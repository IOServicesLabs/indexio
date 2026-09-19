//! Embed pipeline (SPEC-P2 §2.4, rebuilt for throughput in SPEC-P6): chunk
//! visible docs, embed CAS misses, rebuild the vec index with repo-scoped
//! replace.
//!
//! Throughput design (measured: the P5 pipeline spent ~3 ms/chunk in the
//! model's observe and ~1.8 ms/chunk in embed + a file-per-vector CAS, and
//! rebuilt the WHOLE vec/BM25 index once per repo):
//!   * chunking is parallel per doc (rayon), order preserved;
//!   * `Embedder::observe` runs once per repo (stateful embedders handle
//!     their own parallelism internally), then embedding runs as parallel
//!     batches of `EMBED_BATCH` texts;
//!   * the CAS is an append-only pack (see `store.rs`);
//!   * `embed_all` / `embed_repos_with` build the vec index and the BM25
//!     sidecar ONCE for all requested repos instead of once per repo, so a
//!     fleet embed is linear, not quadratic, in total chunk count.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Instant, Duration};

use anyhow::Context;
use rayon::prelude::*;
use tracing::info;

use indexio_index::ShardSet;

use crate::bm25::{Bm25Set, ChunkBm25};
use crate::embed::Embedder;
use crate::index::{VecIndex, VecRowMeta, VecSet};
use crate::store::{chunk_hash, EmbedCas};

/// Chunk size cap per SPEC-P2 §2.4.
pub const MAX_CHARS: usize = 1200;
/// Max texts per `Embedder::embed` call.
const EMBED_BATCH: usize = 64;

#[derive(Clone, Debug, Default)]
pub struct EmbedReport {
    pub repo: String,
    pub chunks: u64,
    pub cas_hits: u64,
    pub cas_misses: u64,
    /// Misses the seen-only cache (SPEC-P10) recognised as already observed:
    /// re-embedded without being fed to the model again.
    pub cas_known: u64,
    pub embedded: u64,
    /// Chunks of files whose rows were already up to date (SPEC-P9
    /// incremental embed): neither looked up nor rewritten.
    pub carried: u64,
    /// Wall time of this repo's own phases (chunk, CAS, observe, embed).
    pub elapsed_ms: u128,
    /// Wall time of the shared vec/BM25 index build that followed; the same
    /// value is stamped on every report of one `embed_repos_with` call.
    pub index_build_ms: u128,
}

fn vec_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("vec")
}

fn bm25_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("bm25")
}

/// Split chunk text (`header + "\n" + body`, see indexio_symbols::chunking) into
/// the (header, body) pair indexed by the BM25F sidecar (SPEC-P5 §A1).
pub(crate) fn split_header_body(text: &str) -> (String, String) {
    match text.split_once('\n') {
        Some((h, b)) => (h.to_string(), b.to_string()),
        None => (text.to_string(), String::new()),
    }
}

/// Repo name of a visible doc.
fn doc_repo(shards: &ShardSet, si: usize, repo_id: u32) -> String {
    shards
        .shard(si)
        .and_then(|s| s.meta().repos.get(repo_id as usize).cloned())
        .unwrap_or_default()
}

/// One chunk awaiting a vector.
struct Work {
    meta: VecRowMeta,
    text: String,
    /// (header, body) of the chunk, for the BM25F sidecar (SPEC-P5 §A2).
    bm25_row: (String, String),
    hash: [u8; 16],
}

/// Chunk every visible doc of `repo` (parallel per doc, doc order kept),
/// or only the docs at `paths` when given.
fn chunk_repo(
    shards: &ShardSet,
    repo: &str,
    max_chars: usize,
    paths: Option<&HashSet<String>>,
) -> anyhow::Result<Vec<Work>> {
    let docs: Vec<(usize, u32, indexio_types::DocMeta)> = shards
        .visible_docs()
        .into_iter()
        .filter(|(si, _, dm)| {
            paths.map_or(true, |p| p.contains(&dm.path)) && doc_repo(shards, *si, dm.repo_id) == repo
        })
        .collect();
    let per_doc: Vec<anyhow::Result<Vec<Work>>> = docs
        .par_iter()
        .map(|(si, docid, dm)| {
            let content = shards
                .content(*si, *docid)
                .with_context(|| format!("read content of {}", dm.path))?;
            Ok(indexio_symbols::chunking::chunks(dm.lang, &dm.path, &content, max_chars)
                .into_iter()
                .map(|c| {
                    let hash = chunk_hash(c.text.as_bytes());
                    Work {
                        hash,
                        bm25_row: split_header_body(&c.text),
                        meta: VecRowMeta {
                            chunk_hash: hash,
                            repo: repo.to_string(),
                            path: dm.path.clone(),
                            start_line: c.start_line,
                            end_line: c.end_line,
                        },
                        text: c.text,
                    }
                })
                .collect())
        })
        .collect();
    let mut works = Vec::new();
    for v in per_doc {
        works.extend(v?);
    }
    Ok(works)
}

/// CAS lookup + observe + embed for `works`; returns one vector per work
/// (in order) and counts hits/misses/embedded into `report`.
fn embed_works(
    cas: &EmbedCas,
    works: &[Work],
    embedder: &dyn Embedder,
    report: &mut EmbedReport,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut vecs: Vec<Option<Vec<f32>>> = (0..works.len()).map(|_| None).collect();
    let mut miss_all: Vec<usize> = Vec::new();
    // A seen-only cache (SPEC-P10) misses every chunk but still knows which
    // ones the model observed before: those are re-embedded, not re-observed.
    let mut known: Vec<bool> = vec![false; works.len()];
    for (i, w) in works.iter().enumerate() {
        if let Some(v) = cas.get(&w.hash) {
            report.cas_hits += 1;
            vecs[i] = Some(v);
        } else {
            report.cas_misses += 1;
            known[i] = cas.contains(&w.hash);
            report.cas_known += u64::from(known[i]);
            miss_all.push(i);
        }
    }
    let mut seen: HashSet<[u8; 16]> = HashSet::new();
    let miss_uniq: Vec<usize> = miss_all
        .iter()
        .copied()
        .filter(|&i| seen.insert(works[i].hash))
        .collect();

    // Stateful embedders (SPEC-P4 §1): ingest the new chunk texts into the
    // model BEFORE embedding them — except transcript chunks (SPEC-P10
    // §18): the `sessions` source is re-rendered every few minutes and its
    // text (tool output, JSON, paths) kept drifting the vocabulary away
    // from the code rows embedded earlier. Transcripts are embedded with
    // the model as it is; only code trains it.
    let miss_texts: Vec<String> = miss_uniq.iter().map(|&i| works[i].text.clone()).collect();
    let new_texts: Vec<String> = miss_uniq
        .iter()
        .filter(|&&i| !known[i] && works[i].meta.repo != NO_OBSERVE_REPO)
        .map(|&i| works[i].text.clone())
        .collect();
    if !new_texts.is_empty() {
        embedder.observe(&new_texts)?;
    }

    // Embed misses: parallel batches of <= EMBED_BATCH, then fill the CAS
    // sequentially (single pack writer).
    let dim = embedder.dim();
    let batches: Vec<anyhow::Result<Vec<Vec<f32>>>> = miss_texts
        .par_chunks(EMBED_BATCH)
        .map(|texts| {
            let out = embedder.embed(texts)?;
            anyhow::ensure!(
                out.len() == texts.len(),
                "embedder returned {} vectors for {} inputs",
                out.len(),
                texts.len()
            );
            for v in &out {
                anyhow::ensure!(v.len() == dim, "embedder returned dim {} != {dim}", v.len());
            }
            Ok(out)
        })
        .collect();
    let mut k = 0usize;
    // vectors computed in this pass, by hash: duplicate misses (the same
    // chunk text in two files) take theirs from here — the seen-only cache
    // (SPEC-P10 §5) never hands vectors back
    let mut computed: HashMap<[u8; 16], Vec<f32>> = HashMap::new();
    for batch in batches {
        for v in batch? {
            let i = miss_uniq[k];
            k += 1;
            cas.put(&works[i].hash, &v);
            computed.insert(works[i].hash, v.clone());
            vecs[i] = Some(v);
            report.embedded += 1;
        }
    }
    cas.flush();
    for &i in &miss_all {
        if vecs[i].is_none() {
            vecs[i] = computed.get(&works[i].hash).cloned().or_else(|| cas.get(&works[i].hash));
        }
    }
    vecs.into_iter()
        .zip(works)
        .map(|(v, w)| v.with_context(|| format!("no embedding for chunk in {}", w.meta.path)))
        .collect()
}

/// Source whose chunks are embedded but never fed to a stateful embedder's
/// model (the rendered session transcripts, SPEC-P10 §18).
pub const NO_OBSERVE_REPO: &str = "sessions";

/// `<vec>/<model>.built`: the model's text count when the plane was last
/// rebuilt whole (SPEC-P10 §20). Written by [`rebuild_all`].
pub fn built_stamp_path(vec_dir: &Path, model_id: &str) -> PathBuf {
    vec_dir.join(format!("{model_id}.built"))
}

fn write_built_stamp(vec_dir: &Path, model_id: &str, texts_seen: u64) {
    let p = built_stamp_path(vec_dir, model_id);
    if let Err(e) = std::fs::write(&p, texts_seen.to_string()) {
        tracing::warn!(error = %e, path = %p.display(), "could not write the plane's built stamp");
    }
}

/// The model must have learned from this many times the texts it had when
/// the plane was built before a full re-embed is due.
pub const REEMBED_GROWTH: f64 = 1.25;
/// And at least this long must have passed since the last full rebuild.
pub const REEMBED_MIN_AGE: Duration = Duration::from_secs(6 * 3600);

/// Whether the semantic plane should be rebuilt whole because the model has
/// drifted from the state its rows were embedded with (SPEC-P10 §20): a
/// stamp exists, the model has grown by [`REEMBED_GROWTH`] since, and the
/// stamp is older than [`REEMBED_MIN_AGE`]. Without a stamp (a plane built
/// before P10, or by a stateless embedder) nothing is due.
pub fn reembed_due(data_dir: &Path, model_id: &str, texts_seen: Option<u64>) -> bool {
    let Some(now) = texts_seen else { return false };
    let p = built_stamp_path(&vec_dir(data_dir), model_id);
    let Ok(meta) = std::fs::metadata(&p) else { return false };
    let young = meta.modified().ok().and_then(|m| m.elapsed().ok()).map_or(false, |age| age < REEMBED_MIN_AGE);
    if young {
        return false;
    }
    let built: u64 = std::fs::read_to_string(&p).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
    built > 0 && (now as f64) >= (built as f64) * REEMBED_GROWTH
}

/// Segments above this count trigger a merge of the small ones.
pub const MAX_SEGMENTS: usize = 8;
/// Tombstoned row fraction above which every segment is merged.
pub const MAX_DEAD_FRACTION: f64 = 0.3;

/// The open vector + BM25 segment sets, or `None` before the first embed.
/// The two are written one after the other by every embed, so another
/// server can be observed between its two writes: a mismatch is an error
/// the caller retries later, never a reason to touch the plane.
fn open_sets(data_dir: &Path, model_id: &str) -> anyhow::Result<Option<(VecSet, Bm25Set)>> {
    let vs = VecSet::open(&vec_dir(data_dir), model_id)?;
    let bs = Bm25Set::open(&bm25_dir(data_dir), model_id)?;
    match (vs, bs) {
        (Some(v), Some(b)) if v.raw_len() == b.raw_len() && v.segments().len() == b.segments().len() => {
            Ok(Some((v, b)))
        }
        (None, None) => Ok(None),
        (v, b) => anyhow::bail!(
            "vector and bm25 segments out of sync ({} vs {} segments); retry",
            v.map_or(0, |v| v.segments().len()),
            b.map_or(0, |b| b.segments().len())
        ),
    }
}

/// Remove every segment (and side file) of `model_id` in both dirs (a
/// model rebuild must start from nothing: unchanged files would otherwise
/// keep rows embedded by the old model).
pub fn clear_segments(data_dir: &Path, model_id: &str) {
    for p in crate::index::segment_paths(&vec_dir(data_dir), model_id) {
        crate::index::remove_or_park(&p).ok();
    }
    for p in crate::bm25::segment_paths(&bm25_dir(data_dir), model_id) {
        crate::index::remove_or_park(&p).ok();
    }
}

/// Embed `repos` INCREMENTALLY (SPEC-P9): every visible doc of each repo is
/// chunked; a file whose chunk hashes equal its live rows' is left alone;
/// changed and new files get their old rows tombstoned and their chunks
/// embedded (CAS first) into ONE new delta segment (vec + aligned BM25);
/// files gone from the repo are tombstoned. Nothing existing is rewritten
/// except the small tombstone side files, so a one-file change costs
/// milliseconds instead of a rebuild of the whole plane.
pub fn embed_repos_with(
    shards: &ShardSet,
    data_dir: &Path,
    repos: &[String],
    embedder: &dyn Embedder,
    max_chars: usize,
) -> anyhow::Result<Vec<EmbedReport>> {
    embed_incremental(shards, data_dir, repos, None, embedder, max_chars)
}

/// Embed only `paths` of `repo` (SPEC-P9 auto-refresh): the docs at those
/// paths are chunked and diffed against their live rows; paths without a
/// visible doc are treated as deleted. Files not named are untouched.
pub fn embed_paths(
    shards: &ShardSet,
    data_dir: &Path,
    repo: &str,
    paths: &[String],
    embedder: &dyn Embedder,
    max_chars: usize,
) -> anyhow::Result<EmbedReport> {
    let set: HashSet<String> = paths.iter().cloned().collect();
    let mut reports =
        embed_incremental(shards, data_dir, &[repo.to_string()], Some(&set), embedder, max_chars)?;
    Ok(reports.remove(0))
}

fn embed_incremental(
    shards: &ShardSet,
    data_dir: &Path,
    repos: &[String],
    only_paths: Option<&HashSet<String>>,
    embedder: &dyn Embedder,
    max_chars: usize,
) -> anyhow::Result<Vec<EmbedReport>> {
    embed_incremental_with(shards, data_dir, repos, only_paths, None, embedder, max_chars)
}

/// [`embed_paths`] with the caller's already-open segment sets (the MCP
/// server's engine cache), so nothing is re-parsed; tombstones go straight
/// to the side files and the caller's next reopen picks them up.
pub fn embed_paths_with_sets(
    shards: &ShardSet,
    data_dir: &Path,
    repo: &str,
    paths: &[String],
    sets: Option<(&VecSet, &Bm25Set)>,
    embedder: &dyn Embedder,
    max_chars: usize,
) -> anyhow::Result<EmbedReport> {
    let set: HashSet<String> = paths.iter().cloned().collect();
    let mut reports = embed_incremental_with(
        shards,
        data_dir,
        &[repo.to_string()],
        Some(&set),
        sets,
        embedder,
        max_chars,
    )?;
    Ok(reports.remove(0))
}

fn embed_incremental_with(
    shards: &ShardSet,
    data_dir: &Path,
    repos: &[String],
    only_paths: Option<&HashSet<String>>,
    given: Option<(&VecSet, &Bm25Set)>,
    embedder: &dyn Embedder,
    max_chars: usize,
) -> anyhow::Result<Vec<EmbedReport>> {
    let model_id = embedder.model_id().to_string();
    let cas = EmbedCas::open_for(data_dir, embedder);
    let t_all = Instant::now();
    // The two halves of the plane must agree. Another server may be
    // between writing its vector delta and its BM25 delta right now, so a
    // mismatch is a "try again" — never a reason to drop the plane (that
    // is what `rebuild_all` / `--rebuild-model` are for). No segments at
    // all means a first embed: other repos join on their own embed call.
    let owned: Option<(VecSet, Bm25Set)> = match given {
        Some(_) => None,
        None => open_sets(data_dir, &model_id)?,
    };
    let sets: Option<(&VecSet, &Bm25Set)> = match (&owned, given) {
        (Some((v, b)), _) => Some((v, b)),
        (None, Some(g)) => {
            anyhow::ensure!(
                g.0.raw_len() == g.1.raw_len() && g.0.segments().len() == g.1.segments().len(),
                "vector and bm25 segments out of sync ({} vs {} rows); retry",
                g.0.raw_len(),
                g.1.raw_len()
            );
            Some(g)
        }
        _ => None,
    };
    let requested: HashSet<&str> = repos.iter().map(String::as_str).collect();

    // Live rows of the requested repos, per file: (global row, chunk hash).
    let mut existing: HashMap<(String, String), Vec<(u32, [u8; 16])>> = HashMap::new();
    if let Some((vset, _)) = &sets {
        for row in vset.live_rows() {
            let m = vset.row_meta(row);
            if requested.contains(m.repo.as_str()) && only_paths.map_or(true, |p| p.contains(&m.path)) {
                existing
                    .entry((m.repo.clone(), m.path.clone()))
                    .or_default()
                    .push((row, m.chunk_hash));
            }
        }
    }

    let mut doomed: HashSet<u32> = HashSet::new();
    let mut new_rows: Vec<(VecRowMeta, Vec<f32>)> = Vec::new();
    let mut new_bm25: Vec<(String, String)> = Vec::new();
    let mut reports = Vec::with_capacity(repos.len());
    for repo in repos {
        let t0 = Instant::now();
        let mut report = EmbedReport {
            repo: repo.to_string(),
            ..Default::default()
        };
        let works = chunk_repo(shards, repo, max_chars, only_paths)?;
        report.chunks = works.len() as u64;
        let mut by_path: BTreeMap<String, Vec<Work>> = BTreeMap::new();
        for w in works {
            by_path.entry(w.meta.path.clone()).or_default().push(w);
        }
        let mut changed: Vec<Work> = Vec::new();
        for (path, ws) in by_path {
            let old = existing.remove(&(repo.to_string(), path));
            let mut new_hashes: Vec<[u8; 16]> = ws.iter().map(|w| w.hash).collect();
            new_hashes.sort_unstable();
            let same = old.as_ref().map_or(false, |o| {
                let mut oh: Vec<[u8; 16]> = o.iter().map(|(_, h)| *h).collect();
                oh.sort_unstable();
                oh == new_hashes
            });
            if same {
                report.carried += ws.len() as u64;
                continue;
            }
            if let Some(o) = old {
                doomed.extend(o.iter().map(|(r, _)| *r));
            }
            changed.extend(ws);
        }
        // files that vanished from the repo
        let gone: Vec<(String, String)> = existing
            .keys()
            .filter(|(r, _)| r == repo)
            .cloned()
            .collect();
        for key in gone {
            if let Some(o) = existing.remove(&key) {
                doomed.extend(o.iter().map(|(r, _)| *r));
            }
        }
        let vecs = embed_works(&cas, &changed, embedder, &mut report)?;
        for (w, v) in changed.into_iter().zip(vecs) {
            new_rows.push((w.meta, v));
            new_bm25.push(w.bm25_row);
        }
        report.elapsed_ms = t0.elapsed().as_millis();
        reports.push(report);
    }

    let t_idx = Instant::now();
    if let (false, Some((vset, bset))) = (doomed.is_empty(), sets) {
        vset.tombstone_on_disk(&doomed)?;
        bset.tombstone_on_disk(&doomed)?;
    }
    drop(owned);
    if !new_rows.is_empty() {
        let seg = crate::index::next_delta_path(&vec_dir(data_dir), &model_id);
        let bseg = crate::bm25::aligned_path(&bm25_dir(data_dir), &seg);
        VecIndex::create_segment(&seg, embedder.dim(), new_rows)?;
        ChunkBm25::create_path(&bseg, &new_bm25)?;
    }
    embedder.flush()?;
    let build_ms = t_idx.elapsed().as_millis();
    info!(
        repos = repos.len(),
        tombstoned = doomed.len(),
        appended = new_bm25.len(),
        segment_ms = build_ms as u64,
        total_ms = t_all.elapsed().as_millis() as u64,
        "embed incremental"
    );
    for r in &mut reports {
        r.index_build_ms = build_ms;
    }
    Ok(reports)
}

/// Full rebuild of the semantic plane for `repos` (every other repo's rows
/// are dropped): chunk everything, embed CAS misses, write ONE fresh
/// segment for vec + BM25, remove the old segments. Used by the first
/// embed, `indexio embed --all` and model rebuilds.
pub fn rebuild_all(
    shards: &ShardSet,
    data_dir: &Path,
    repos: &[String],
    embedder: &dyn Embedder,
    max_chars: usize,
) -> anyhow::Result<Vec<EmbedReport>> {
    let model_id = embedder.model_id().to_string();
    let cas = EmbedCas::open_for(data_dir, embedder);
    let mut rows: Vec<(VecRowMeta, Vec<f32>)> = Vec::new();
    let mut bm25_rows: Vec<(String, String)> = Vec::new();
    let mut reports = Vec::with_capacity(repos.len());
    for repo in repos {
        let t0 = Instant::now();
        let mut report = EmbedReport {
            repo: repo.to_string(),
            ..Default::default()
        };
        let works = chunk_repo(shards, repo, max_chars, None)?;
        report.chunks = works.len() as u64;
        let vecs = embed_works(&cas, &works, embedder, &mut report)?;
        for (w, v) in works.into_iter().zip(vecs) {
            rows.push((w.meta, v));
            bm25_rows.push(w.bm25_row);
        }
        report.elapsed_ms = t0.elapsed().as_millis();
        reports.push(report);
    }
    let t_idx = Instant::now();
    let old_v = crate::index::segment_paths(&vec_dir(data_dir), &model_id);
    let old_b = crate::bm25::segment_paths(&bm25_dir(data_dir), &model_id);
    let seg = crate::index::next_delta_path(&vec_dir(data_dir), &model_id);
    let bseg = crate::bm25::aligned_path(&bm25_dir(data_dir), &seg);
    VecIndex::create_segment(&seg, embedder.dim(), rows)?;
    ChunkBm25::create_path(&bseg, &bm25_rows)?;
    for p in old_v.iter().chain(old_b.iter()) {
        crate::index::remove_or_park(p)?;
    }
    embedder.flush()?;
    // the model state every row was embedded with (SPEC-P10 §20)
    if let Some(n) = embedder.texts_seen() {
        write_built_stamp(&vec_dir(data_dir), &model_id, n);
    }
    let build_ms = t_idx.elapsed().as_millis();
    info!(
        repos = repos.len(),
        rows = bm25_rows.len(),
        build_ms = build_ms as u64,
        "embed full rebuild"
    );
    for r in &mut reports {
        r.index_build_ms = build_ms;
    }
    Ok(reports)
}

/// Fold segments together (SPEC-P9): with more than [`MAX_SEGMENTS`]
/// segments, every segment but the largest is merged into one; with more
/// than [`MAX_DEAD_FRACTION`] tombstoned rows, all of them are. Vectors
/// are carried as they are and BM25 rows are recovered from the inverted
/// lists, so no source is re-chunked and no embedder is needed. Returns
/// whether a merge happened. Merged inputs another process still maps are
/// parked (`.stale`) and reaped later.
pub fn compact_vectors(data_dir: &Path, model_id: &str) -> anyhow::Result<bool> {
    compact_vectors_opts(data_dir, model_id, false)
}

/// [`compact_vectors`] with `force`: merge every segment into one regardless
/// of the thresholds (re-encodes pre-P10 f32 segments as int8).
pub fn compact_vectors_opts(data_dir: &Path, model_id: &str, force: bool) -> anyhow::Result<bool> {
    crate::index::reap_stale(&vec_dir(data_dir));
    crate::index::reap_stale(&bm25_dir(data_dir));
    let Some((vset, bset)) = open_sets(data_dir, model_id)? else {
        return Ok(false);
    };
    // aligned pairs only: a merge over a half-written pair would misalign rows
    let n = vset.segments().len();
    let dead = vset.dead_fraction();
    let merge: Vec<usize> = if force || dead > MAX_DEAD_FRACTION {
        (0..n).collect()
    } else if n > MAX_SEGMENTS {
        let largest = (0..n)
            .max_by_key(|&i| vset.segments()[i].raw_len())
            .expect("n > 0");
        (0..n).filter(|&i| i != largest).collect()
    } else {
        return Ok(false);
    };
    if merge.is_empty() || (merge.len() < 2 && dead <= MAX_DEAD_FRACTION && !force) {
        return Ok(false);
    }
    let t0 = Instant::now();
    let dim = vset.dim();
    let mut rows: Vec<(VecRowMeta, Vec<f32>)> = Vec::new();
    let mut tf_rows: Vec<(BTreeMap<String, u32>, BTreeMap<String, u32>)> = Vec::new();
    // origin[new row] = (segment, local row): tombstones another server
    // writes to an input while we merge are forwarded afterwards
    let mut origin: Vec<(usize, u32)> = Vec::new();
    for &i in &merge {
        let vs = &vset.segments()[i];
        let mut bs_rows = bset.rows_tf_of(i);
        let off = vset_offset(&vset, i);
        for r in 0..vs.raw_len() as u32 {
            if vset.is_deleted(off + r) {
                continue;
            }
            origin.push((i, r));
            rows.push((vs.row_meta(r).clone(), vs.row_vector(r)));
            tf_rows.push(std::mem::take(&mut bs_rows[r as usize]));
        }
    }
    // group rows by repo (then path, line) so a repo-scoped search scans
    // one run of the merged segment (SPEC-P10 §11); BM25 rows and origins
    // follow the same permutation (rows are aligned across the sidecars)
    {
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by(|&a, &b| {
            let (ma, mb) = (&rows[a].0, &rows[b].0);
            ma.repo.cmp(&mb.repo).then_with(|| ma.path.cmp(&mb.path)).then_with(|| ma.start_line.cmp(&mb.start_line))
        });
        let mut rows_s = Vec::with_capacity(rows.len());
        let mut tf_s = Vec::with_capacity(tf_rows.len());
        let mut origin_s = Vec::with_capacity(origin.len());
        let mut rows_opt: Vec<Option<(VecRowMeta, Vec<f32>)>> = rows.into_iter().map(Some).collect();
        let mut tf_opt: Vec<Option<(BTreeMap<String, u32>, BTreeMap<String, u32>)>> = tf_rows.into_iter().map(Some).collect();
        for i in order {
            rows_s.push(rows_opt[i].take().expect("row moved once"));
            tf_s.push(tf_opt[i].take().expect("tf row moved once"));
            origin_s.push(origin[i]);
        }
        rows = rows_s;
        tf_rows = tf_s;
        origin = origin_s;
    }
    let merged_v: Vec<PathBuf> = merge.iter().map(|&i| vset.segments()[i].path().to_path_buf()).collect();
    let merged_b: Vec<PathBuf> = merge.iter().map(|&i| bset.segments()[i].path().to_path_buf()).collect();
    let carried = rows.len();
    drop(vset);
    drop(bset);
    let seg = crate::index::next_delta_path(&vec_dir(data_dir), model_id);
    let bseg = crate::bm25::aligned_path(&bm25_dir(data_dir), &seg);
    VecIndex::create_segment(&seg, dim, rows)?;
    ChunkBm25::create_from_tf(&bseg, tf_rows)?;
    // rows tombstoned in an input since we read it
    let late: std::collections::BTreeSet<u32> = {
        let mut now: Vec<std::collections::HashSet<u32>> = Vec::with_capacity(merge.len());
        for p in &merged_v {
            now.push(crate::index::read_tomb_side(p).into_iter().collect());
        }
        origin
            .iter()
            .enumerate()
            .filter(|(_, &(i, r))| {
                let k = merge.iter().position(|&m| m == i).expect("origin segment is merged");
                now[k].contains(&r)
            })
            .map(|(new_row, _)| new_row as u32)
            .collect()
    };
    if !late.is_empty() {
        crate::index::write_tomb_side(&seg, &late)?;
        crate::index::write_tomb_side(&bseg, &late)?;
        info!(forwarded = late.len(), "tombstones applied to the merged segment after the merge");
    }
    for p in merged_v.iter().chain(merged_b.iter()) {
        crate::index::remove_or_park(p)?;
    }
    info!(
        merged = merge.len(),
        rows = carried,
        ms = t0.elapsed().as_millis() as u64,
        "semantic segments compacted"
    );
    Ok(true)
}

/// Global id of segment `i`'s first row.
fn vset_offset(v: &VecSet, i: usize) -> u32 {
    v.segments()[..i].iter().map(|s| s.raw_len() as u32).sum()
}

/// Cross-process compaction lock (SPEC-P9): `<data_dir>/compact.lock`,
/// created exclusively; a lock older than [`LOCK_STALE_SECS`] is assumed
/// to belong to a dead process and is taken over. Dropping releases it.
pub struct CompactLock {
    path: PathBuf,
}

/// Age past which a compaction lock is considered abandoned.
pub const LOCK_STALE_SECS: u64 = 15 * 60;

impl CompactLock {
    /// `None` when another live process holds the lock.
    pub fn try_acquire(data_dir: &Path) -> Option<CompactLock> {
        let path = data_dir.join("compact.lock");
        for _ in 0..2 {
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut f) => {
                    use std::io::Write as _;
                    let _ = write!(f, "{}", std::process::id());
                    return Some(CompactLock { path });
                }
                Err(_) => {
                    let stale = std::fs::metadata(&path)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .map_or(true, |age| age.as_secs() > LOCK_STALE_SECS);
                    if !stale {
                        return None;
                    }
                    tracing::warn!(path = %path.display(), "taking over a stale compaction lock");
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        None
    }
}

impl Drop for CompactLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Whether a semantic plane for `model_id` exists at all (SPEC-P10 §28):
/// the one case a sync should rebuild whole rather than append.
pub fn plane_exists(data_dir: &Path, model_id: &str) -> bool {
    !crate::index::segment_paths(&vec_dir(data_dir), model_id).is_empty()
}

/// Whether [`compact_vectors`] would do anything.
pub fn needs_compaction(data_dir: &Path, model_id: &str) -> bool {
    let (n, rows, dead) = crate::index::segment_stats(&vec_dir(data_dir), model_id);
    n > MAX_SEGMENTS || (rows > 0 && dead as f64 / rows as f64 > MAX_DEAD_FRACTION)
}

/// Embed the chunks of `repo` whose docs are visible in `shards`
/// (incremental, see [`embed_repos_with`]).
pub fn embed_repo(
    shards: &ShardSet,
    data_dir: &Path,
    repo: &str,
    embedder: &dyn Embedder,
) -> anyhow::Result<EmbedReport> {
    embed_repo_with(shards, data_dir, repo, embedder, MAX_CHARS)
}

/// `embed_repo` with a caller-chosen chunk size cap (SPEC-P2 §4 CLI
/// `--max-chars`; the library default stays [`MAX_CHARS`]).
pub fn embed_repo_with(
    shards: &ShardSet,
    data_dir: &Path,
    repo: &str,
    embedder: &dyn Embedder,
    max_chars: usize,
) -> anyhow::Result<EmbedReport> {
    let mut reports =
        embed_repos_with(shards, data_dir, &[repo.to_string()], embedder, max_chars)?;
    anyhow::ensure!(!reports.is_empty(), "repo '{repo}' has no visible docs");
    Ok(reports.remove(0))
}

/// Full rebuild for every repo with visible docs in the shard set — one
/// fresh segment for the whole fleet (the weekly/cron path).
pub fn embed_all(
    shards: &ShardSet,
    data_dir: &Path,
    embedder: &dyn Embedder,
) -> anyhow::Result<Vec<EmbedReport>> {
    embed_all_with(shards, data_dir, embedder, MAX_CHARS)
}

/// Repos with visible docs in the shard set, sorted.
fn repos_of(shards: &ShardSet) -> BTreeSet<String> {
    let mut repos: BTreeSet<String> = BTreeSet::new();
    for (si, _, dm) in shards.visible_docs() {
        repos.insert(doc_repo(shards, si, dm.repo_id));
    }
    repos
}

/// `embed_all` with a caller-chosen chunk size cap (see [`embed_repo_with`]).
pub fn embed_all_with(
    shards: &ShardSet,
    data_dir: &Path,
    embedder: &dyn Embedder,
    max_chars: usize,
) -> anyhow::Result<Vec<EmbedReport>> {
    let repos: Vec<String> = repos_of(shards).into_iter().collect();
    if repos.is_empty() {
        return Ok(Vec::new());
    }
    rebuild_all(shards, data_dir, &repos, embedder, max_chars)
}


// ---------------------------------------------------------------------------
// Tests — these call indexio_symbols::chunking::chunks, currently a `todo!()` stub
// being implemented on a parallel branch. They are written in full and marked
// indexio-embed -- --ignored`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::{dot, Embedder, HashEmbedder};
    use indexio_index::{ShardSet, ShardWriter};
    use indexio_types::{BlobId, DocMeta, ExtractedArtifact, Lang};

    /// Build a single-shard ShardSet: (repo_id, path, content) triples.
    fn build_shardset(
        dir: &Path,
        repos: &[&str],
        docs: &[(u32, &str, &[u8])],
    ) -> ShardSet {
        let shards_dir = dir.join("shards");
        std::fs::create_dir_all(&shards_dir).unwrap();
        let mut w = ShardWriter::new(&shards_dir).unwrap();
        for &(repo_id, path, content) in docs {
            let meta = DocMeta {
                blob: BlobId::from_content(content),
                repo_id,
                path: path.to_string(),
                lang: Lang::from_path(path),
                raw_len: content.len() as u32,
            };
            let art = ExtractedArtifact {
                lang: meta.lang,
                raw_len: content.len() as u32,
                ..Default::default()
            };
            w.add_doc(&meta, content, &art).unwrap();
        }
        let repos: Vec<String> = repos.iter().map(|s| s.to_string()).collect();
        w.finish(&repos).unwrap();
        ShardSet::open_dir(&shards_dir).unwrap()
    }

    const SHARED: &[u8] = b"pub fn shared_helper(x: i32) -> i32 { x + 1 }\n";

    #[test]
    fn embed_repo_cas_dedup_across_two_repos() {
        let tmp = tempfile::tempdir().unwrap();
        let shards = build_shardset(
            tmp.path(),
            &["r1", "r2"],
            &[
                (0, "src/shared.rs", SHARED),
                (0, "src/only_r1.rs", b"fn alpha_unique() {}\n"),
                (1, "src/shared.rs", SHARED), // identical file in repo 2
                (1, "src/only_r2.rs", b"fn beta_unique() {}\n"),
            ],
        );
        let e = HashEmbedder::new(512);

        let rep1 = embed_repo(&shards, tmp.path(), "r1", &e).unwrap();
        assert!(rep1.chunks > 0);
        assert_eq!(rep1.cas_hits, 0, "first run: cold CAS");
        assert_eq!(rep1.cas_misses, rep1.chunks);
        assert!(rep1.embedded > 0);

        // HashEmbedder is cheap to recompute: the cache is seen-only, so
        // the shared file is a miss the cache recognises (SPEC-P10)
        let rep2 = embed_repo(&shards, tmp.path(), "r2", &e).unwrap();
        assert_eq!(rep2.cas_hits, 0, "seen-only cache never serves vectors: {rep2:?}");
        assert!(rep2.cas_known > 0, "shared file is known to the cache: {rep2:?}");
        assert!(rep2.cas_known < rep2.chunks, "only the shared chunks are known: {rep2:?}");
        // every miss is embedded (unique hashes at least once)
        assert!(rep2.embedded > 0 && rep2.embedded <= rep2.cas_misses);

        // Re-embedding an unchanged repo carries every row (SPEC-P9): no
        // CAS lookups, nothing embedded, nothing rewritten.
        let rep1b = embed_repo(&shards, tmp.path(), "r1", &e).unwrap();
        assert_eq!(rep1b.carried, rep1b.chunks, "{rep1b:?}");
        assert_eq!((rep1b.cas_hits, rep1b.embedded), (0, 0));

        // embed_all covers both repos.
        let reps = embed_all(&shards, tmp.path(), &e).unwrap();
        assert_eq!(reps.len(), 2);
    }

    #[test]
    fn reembed_due_needs_stamp_growth_and_age() {
        let tmp = tempfile::tempdir().unwrap();
        let vd = vec_dir(tmp.path());
        std::fs::create_dir_all(&vd).unwrap();
        // no stamp, stateless embedder: never
        assert!(!reembed_due(tmp.path(), "m", None));
        assert!(!reembed_due(tmp.path(), "m", Some(1_000_000)));
        write_built_stamp(&vd, "m", 1000);
        // fresh stamp: not yet, however much the model grew
        assert!(!reembed_due(tmp.path(), "m", Some(5000)));
        // age the stamp past the minimum
        let old = std::time::SystemTime::now() - REEMBED_MIN_AGE - Duration::from_secs(60);
        let f = std::fs::File::options().write(true).open(built_stamp_path(&vd, "m")).unwrap();
        f.set_modified(old).unwrap();
        assert!(!reembed_due(tmp.path(), "m", Some(1200)), "under the growth factor");
        assert!(reembed_due(tmp.path(), "m", Some(1250)), "at the growth factor");
        assert!(reembed_due(tmp.path(), "m", Some(5000)));
    }

    #[test]
    fn rebuild_writes_the_built_stamp() {
        let tmp = tempfile::tempdir().unwrap();
        let shards = build_shardset(tmp.path(), &["r1"], &[(0, "a.rs", b"fn alpha_one() {}\n")]);
        let e = crate::rindex::RandomIndexingEmbedder::open(tmp.path()).unwrap();
        embed_all(&shards, tmp.path(), &e).unwrap();
        let stamp = std::fs::read_to_string(built_stamp_path(&vec_dir(tmp.path()), e.model_id())).unwrap();
        assert_eq!(stamp.trim().parse::<u64>().unwrap(), e.texts_seen().unwrap());
        assert!(e.texts_seen().unwrap() > 0);
    }

    #[test]
    fn duplicate_chunks_in_one_pass_all_get_vectors() {
        // SPEC-P10 §5 regression: the seen-only cache never returns
        // vectors, so a chunk text that occurs twice in one pass (the same
        // file in two places) must take its vector from the pass itself
        let tmp = tempfile::tempdir().unwrap();
        let same = b"fn shared_twice() -> u32 { 42 }
";
        let shards = build_shardset(
            tmp.path(),
            &["r1"],
            &[(0, "a/dup.rs", same), (0, "b/dup.rs", same), (0, "c/other.rs", b"fn other_one() {}
")],
        );
        let e = HashEmbedder::new(64);
        let rep = embed_repo(&shards, tmp.path(), "r1", &e).unwrap();
        assert!(rep.chunks >= 3, "{rep:?}");
        let idx = crate::index::VecSet::open(&tmp.path().join("vec"), e.model_id()).unwrap().unwrap();
        assert_eq!(idx.len() as u64, rep.chunks, "every chunk has a row: {rep:?}");
    }

    #[test]
    fn repo_scoped_replace_on_reembed() {
        let tmp = tempfile::tempdir().unwrap();
        let shards_v1 = build_shardset(
            tmp.path(),
            &["r1", "r2"],
            &[
                (0, "src/old.rs", b"fn stale_function() {}\n"),
                (1, "src/stable.rs", b"fn stable_function() {}\n"),
            ],
        );
        let e = HashEmbedder::new(512);
        embed_repo(&shards_v1, tmp.path(), "r1", &e).unwrap();
        embed_repo(&shards_v1, tmp.path(), "r2", &e).unwrap();
        drop(shards_v1);

        // r1 changes: old.rs replaced by new.rs. Simulate the re-ingested
        // index by clearing the shard dir and writing the fresh shard set.
        for f in std::fs::read_dir(tmp.path().join("shards")).unwrap() {
            let p = f.unwrap().path();
            if p.extension().and_then(|e| e.to_str()) == Some("cidx") {
                std::fs::remove_file(p).unwrap();
            }
        }
        let shards_v2 = build_shardset(
            tmp.path(),
            &["r1", "r2"],
            &[
                (0, "src/new.rs", b"fn fresh_function() {}\n"),
                (1, "src/stable.rs", b"fn stable_function() {}\n"),
            ],
        );
        embed_repo(&shards_v2, tmp.path(), "r1", &e).unwrap();

        let idx = VecSet::open(&tmp.path().join("vec"), "hash-v1")
            .unwrap()
            .unwrap();
        let mut r1_paths: Vec<&str> = Vec::new();
        let mut r2_count = 0;
        for row in 0..idx.raw_len() as u32 {
            if idx.is_deleted(row) {
                continue;
            }
            let m = idx.row_meta(row);
            if m.repo == "r1" {
                r1_paths.push(m.path.as_str());
            } else if m.repo == "r2" {
                r2_count += 1;
            }
        }
        assert!(
            !r1_paths.iter().any(|p| p.contains("old.rs")),
            "stale rows for r1 must be gone: {r1_paths:?}"
        );
        assert!(
            r1_paths.iter().any(|p| p.contains("new.rs")),
            "new rows present: {r1_paths:?}"
        );
        assert!(r2_count > 0, "other repos' rows are preserved");

        // Semantic check: a query matching the fresh function ranks the
        // fresh chunk first.
        let q = e
            .embed(&["fn fresh_function".to_string()])
            .unwrap()
            .remove(0);
        let hits = idx.search(&q, 3);
        assert!(!hits.is_empty());
        let top = idx.row_meta(hits[0].0);
        assert_eq!(top.repo, "r1");
        assert!(top.path.contains("new.rs"), "top hit: {top:?}");
        assert!(dot(&q, &q) > 0.99);
    }

    /// Row slots via public API (live rows + tombstoned, what the aligned
    /// BM25 set reports as `raw_len`).
    fn idx_len(idx: &VecSet) -> usize {
        idx.raw_len()
    }

    /// SPEC-P5 §A2: embed_repo writes a BM25F sidecar aligned row-for-row
    /// with the vec index; re-embedding a repo replaces only that repo's
    /// rows in BOTH indexes, and carried-over other-repo rows keep their
    /// (header, body) text (recovered by chunk_hash).
    #[test]
    fn bm25_sidecar_aligned_and_repo_replaced() {

        let tmp = tempfile::tempdir().unwrap();
        let shards_v1 = build_shardset(
            tmp.path(),
            &["r1", "r2"],
            &[
                (0, "src/a.rs", b"fn alpha_unique_token() {}\n"),
                (0, "src/b.rs", b"fn beta_helper() {}\n"),
                (1, "src/c.rs", b"fn gamma_unique_token() {}\n"),
                (1, "src/d.rs", b"fn delta_helper() {}\n"),
            ],
        );
        let e = HashEmbedder::new(512);
        embed_repo(&shards_v1, tmp.path(), "r1", &e).unwrap();
        embed_repo(&shards_v1, tmp.path(), "r2", &e).unwrap();

        let bm = Bm25Set::open(&tmp.path().join("bm25"), "hash-v1")
            .unwrap()
            .expect("pipeline must write the bm25 sidecar");
        let idx = VecSet::open(&tmp.path().join("vec"), "hash-v1")
            .unwrap()
            .unwrap();
        assert_eq!(bm.raw_len(), idx_len(&idx), "row-for-row alignment");
        assert_eq!(bm.len(), idx.len());

        // A planted df-1 term surfaces exactly its chunk's row, resolvable
        // through the aligned VecIndex row meta.
        let hits = bm.search("alpha", 5);
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert!(idx.row_meta(hits[0].0).path.contains("a.rs"));

        // r1 re-ingested with a.rs replaced; re-embed only r1.
        drop(shards_v1);
        for f in std::fs::read_dir(tmp.path().join("shards")).unwrap() {
            let p = f.unwrap().path();
            if p.extension().and_then(|e| e.to_str()) == Some("cidx") {
                std::fs::remove_file(p).unwrap();
            }
        }
        let shards_v2 = build_shardset(
            tmp.path(),
            &["r1", "r2"],
            &[
                (0, "src/a2.rs", b"fn epsilon_fresh_token() {}\n"),
                (0, "src/b.rs", b"fn beta_helper() {}\n"),
                (1, "src/c.rs", b"fn gamma_unique_token() {}\n"),
                (1, "src/d.rs", b"fn delta_helper() {}\n"),
            ],
        );
        embed_repo(&shards_v2, tmp.path(), "r1", &e).unwrap();

        let bm = Bm25Set::open(&tmp.path().join("bm25"), "hash-v1")
            .unwrap()
            .unwrap();
        let idx = VecSet::open(&tmp.path().join("vec"), "hash-v1")
            .unwrap()
            .unwrap();
        assert_eq!(bm.raw_len(), idx_len(&idx), "alignment after replace");
        // b.rs did not change: its rows were carried, not re-embedded
        // r1, r2, then the changed a.rs: three segments, nothing rewritten
        assert_eq!(idx.segments().len(), 3, "one delta segment per embed call");
        // Stale r1 term is gone; fresh r1 term is indexed.
        assert!(bm.search("alpha", 5).is_empty(), "stale rows replaced");
        let hits = bm.search("epsilon", 5);
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert!(idx.row_meta(hits[0].0).path.contains("a2.rs"));
        // Carried-over r2 rows kept their text via the chunk_hash map.
        let hits = bm.search("gamma", 5);
        assert_eq!(hits.len(), 1, "carried r2 rows keep text: {hits:?}");
        let m = idx.row_meta(hits[0].0);
        assert_eq!(m.repo, "r2");
        assert!(m.path.contains("c.rs"));
    }
}
