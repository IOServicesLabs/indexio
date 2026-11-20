//! Embed pipeline: chunk
//! visible docs, embed CAS misses, rebuild the vec index with repo-scoped
//! replace.
//!
//! Throughput design (measured: the earlier pipeline spent ~3 ms/chunk in the
//! model's observe and ~1.8 ms/chunk in embed + a file-per-vector CAS, and
//! rebuilt the WHOLE vec/BM25 index once per repo):
//!   * chunking is parallel per doc (rayon), order preserved;
//!   * `Embedder::observe` runs once per repo (stateful embedders handle
//!     their own parallelism internally), then embedding runs as parallel
//!     batches of `EMBED_BATCH` texts;
//!   * the CAS is an append-only pack (see `store.rs`);
//!   * `embed_all` / `embed_repos_with` build the vec index and the BM25
//!     sidecar ONCE for all requested repos instead of once per repo, so a
//!     fleet embed is linear, not quadratic, in total chunk count;
//!   * the rows of a pass go to the segment writers as each repo is
//!     embedded (`RepoPass`): a pass holds a batch of chunks and a round of
//!     vectors, not the rows it writes.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::{Instant, Duration};

use anyhow::Context;
use rayon::prelude::*;
use tracing::info;

use indexio_index::ShardSet;

use crate::bm25::{Bm25Set, Bm25Writer};
use crate::embed::Embedder;
use crate::index::{RowMeta, SegmentWriter, VecIndex, VecSet};
use crate::spool::Spool;
use crate::store::{chunk_hash, EmbedCas};

/// Chunk size cap.
pub const MAX_CHARS: usize = 1200;
/// Max texts per `Embedder::embed` call.
const EMBED_BATCH: usize = 64;

#[derive(Clone, Debug, Default)]
pub struct EmbedReport {
    pub repo: String,
    pub chunks: u64,
    pub cas_hits: u64,
    pub cas_misses: u64,
    /// Misses the seen-only cache recognised as already observed:
    /// re-embedded without being fed to the model again.
    pub cas_known: u64,
    pub embedded: u64,
    /// Chunks of files whose rows were already up to date
    /// (incremental embed): neither looked up nor rewritten.
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
/// the (header, body) pair indexed by the BM25F sidecar.
pub(crate) fn split_header_body(text: &str) -> (&str, &str) {
    text.split_once('\n').unwrap_or((text, ""))
}

/// Repo name of a visible doc.
fn doc_repo(shards: &ShardSet, si: usize, repo_id: u32) -> String {
    shards
        .shard(si)
        .and_then(|s| s.meta().repos.get(repo_id as usize).cloned())
        .unwrap_or_default()
}

/// A visible doc: (shard, doc id, metadata).
type Doc = (usize, u32, indexio_types::DocMeta);

/// One chunk of a repo awaiting a vector.
struct Work {
    path: String,
    start_line: u32,
    end_line: u32,
    text: String,
    hash: [u8; 16],
}

/// Every visible doc of `repo`, or only the docs at `paths` when given.
fn repo_docs(shards: &ShardSet, repo: &str, paths: Option<&HashSet<String>>) -> Vec<Doc> {
    shards
        .visible_docs()
        .into_iter()
        .filter(|(si, _, dm)| {
            paths.is_none_or(|p| p.contains(&dm.path)) && doc_repo(shards, *si, dm.repo_id) == repo
        })
        .collect()
}

/// Source text chunked at a time, and docs: the chunks of a repo used to
/// be made, and held with their text, for the whole repo at once.
const CHUNK_BATCH_BYTES: u64 = 16 << 20;
#[cfg(not(test))]
const CHUNK_BATCH_DOCS: usize = 4096;
/// Under test a repo of a few dozen files takes several batches.
#[cfg(test)]
const CHUNK_BATCH_DOCS: usize = 7;

/// How many of the docs at the front of `docs` make the next batch (at
/// least one, when there is one).
fn next_batch(docs: &[Doc]) -> usize {
    let mut bytes = 0u64;
    let mut n = 0;
    while n < docs.len() && n < CHUNK_BATCH_DOCS {
        bytes += u64::from(docs[n].2.raw_len);
        if n > 0 && bytes > CHUNK_BATCH_BYTES {
            break;
        }
        n += 1;
    }
    n
}

/// Chunk `docs` (parallel per doc): the chunks of each doc, in doc order.
fn chunk_docs(shards: &ShardSet, docs: &[Doc], max_chars: usize) -> anyhow::Result<Vec<Vec<Work>>> {
    docs.par_iter()
        .map(|(si, docid, dm)| {
            let content = shards
                .content(*si, *docid)
                .with_context(|| format!("read content of {}", dm.path))?;
            Ok(indexio_symbols::chunking::chunks(dm.lang, &dm.path, &content, max_chars)
                .into_iter()
                .map(|c| Work {
                    hash: chunk_hash(c.text.as_bytes()),
                    path: dm.path.clone(),
                    start_line: c.start_line,
                    end_line: c.end_line,
                    text: c.text,
                })
                .collect())
        })
        .collect()
}

/// Chunks of a repo held in memory between the two passes of its embed
/// before they move to a scratch file.
#[cfg(not(test))]
const WORKS_IN_MEMORY: usize = 16 << 20;
/// Under test the chunks of a small repo go through the scratch file.
#[cfg(test)]
const WORKS_IN_MEMORY: usize = 4096;

/// Texts embedded in one round: a whole number of [`EMBED_BATCH`]es, so
/// the batches handed to the embedder are the ones a single round over
/// every miss of the repo would make. Their vectors (8 KB each at dim
/// 2048) are all that is held in f32 at a time.
#[cfg(not(test))]
const EMBED_ROUND: usize = 32 * EMBED_BATCH;
/// Under test a repo of a few hundred chunks takes several rounds.
#[cfg(test)]
const EMBED_ROUND: usize = 2 * EMBED_BATCH;

/// In the miss table: no chunk with this hash has been read back yet.
const UNSEEN: u32 = u32::MAX;
/// In the miss table: the first chunk with this hash is waiting for the
/// round that embeds it. Any other value is the row that chunk was written to.
const QUEUED: u32 = u32::MAX - 1;

/// The segment pair an embed writes: one vector row and one BM25 row per
/// chunk, in the same order, committed together.
struct PlaneWriter {
    vec: SegmentWriter,
    bm25: Bm25Writer,
}

impl PlaneWriter {
    fn new(data_dir: &Path, dim: usize) -> anyhow::Result<Self> {
        Ok(PlaneWriter { vec: SegmentWriter::new(&vec_dir(data_dir), dim)?, bm25: Bm25Writer::new(&bm25_dir(data_dir)) })
    }

    /// Rows written so far.
    fn rows(&self) -> u32 {
        self.vec.rows()
    }

    /// Write the pair: BM25 first, the vector file last (the commit).
    fn commit(self, seg: &Path, bseg: &Path) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.vec.rows() == self.bm25.rows(),
            "vector and bm25 rows out of step ({} vs {})",
            self.vec.rows(),
            self.bm25.rows()
        );
        let PlaneWriter { vec, bm25 } = self;
        create_pair(seg, bseg, |p| vec.finish(p), |p| bm25.finish(p))
    }
}

/// Where a chunk read back in the second pass takes its vector from.
enum Source {
    /// The cache had it.
    Cache,
    /// It is embedded in the coming round (the first chunk with its text).
    Round,
    /// A chunk with the same text came earlier in this repo: its row is copied.
    Earlier,
}

/// A chunk whose BM25 row is written and whose vector row waits for the
/// round in progress.
struct Pending {
    path: String,
    start_line: u32,
    end_line: u32,
    hash: [u8; 16],
    source: Source,
}

/// CAS lookup + observe + embed for the chunks of one repo, straight into
/// the segment pair being written.
///
/// This used to be one call over every chunk of the repo that returned
/// every vector: the chunks with two more copies of their text, a
/// `Vec<f32>` per chunk and a second copy of each in a map by hash, all
/// alive until the segment was built from them (18 KB per chunk at dim
/// 2048). The steps are the same and happen in the same order; they are
/// split in two passes around the one `observe` call the embedder gets per
/// repo:
///
/// 1. [`add`](Self::add), a batch of chunks at a time: look each chunk up
///    in the cache, ask which of the misses the model has seen, keep the
///    text of those it has not (first occurrence only), and put the chunk
///    in a spool.
/// 2. [`finish`](Self::finish): observe those texts, then read the chunks
///    back in order. Each gets its BM25 row at once. The misses are
///    embedded [`EMBED_ROUND`] texts at a time and rows are written as
///    their vectors arrive, so a round's vectors are the only f32 rows
///    alive. A chunk whose text already occurred in the repo copies the
///    row of the first one instead of keeping its vector around.
struct RepoPass<'a> {
    cas: &'a EmbedCas,
    embedder: &'a dyn Embedder,
    repo: &'a str,
    works: Spool,
    n_works: usize,
    /// Every chunk hash the cache missed. First pass: the set of them.
    /// Second pass: [`UNSEEN`], [`QUEUED`] or the row of the first chunk.
    misses: HashMap<[u8; 16], u32>,
    /// Texts the model has not seen, for `observe`.
    new_texts: Vec<String>,
    record: Vec<u8>,
}

impl<'a> RepoPass<'a> {
    fn new(cas: &'a EmbedCas, embedder: &'a dyn Embedder, repo: &'a str, data_dir: &Path) -> Self {
        RepoPass {
            cas,
            embedder,
            repo,
            works: Spool::new(&vec_dir(data_dir), "embed-chunks", WORKS_IN_MEMORY),
            n_works: 0,
            misses: HashMap::new(),
            new_texts: Vec::new(),
            record: Vec::new(),
        }
    }

    /// First pass over a batch of the repo's chunks, in row order.
    fn add(&mut self, works: Vec<Work>, report: &mut EmbedReport) -> anyhow::Result<()> {
        let mut missed: Vec<([u8; 16], String)> = Vec::new();
        for w in works {
            let hit = self.cas.get(&w.hash).is_some();
            let r = &mut self.record;
            r.clear();
            r.push(u8::from(hit));
            r.extend_from_slice(&w.hash);
            r.extend_from_slice(&w.start_line.to_le_bytes());
            r.extend_from_slice(&w.end_line.to_le_bytes());
            for s in [&w.path, &w.text] {
                let len = u32::try_from(s.len()).context("chunk too large")?;
                r.extend_from_slice(&len.to_le_bytes());
                r.extend_from_slice(s.as_bytes());
            }
            self.works.write(r)?;
            self.n_works += 1;
            if hit {
                report.cas_hits += 1;
            } else {
                report.cas_misses += 1;
                missed.push((w.hash, w.text));
            }
        }
        // all misses of the batch in one question: a seen-only cache answers
        // it with one pass over its file. A seen-only cache misses every
        // chunk but still knows which ones the model observed before:
        // those are re-embedded, not re-observed.
        let hashes: Vec<[u8; 16]> = missed.iter().map(|m| m.0).collect();
        for ((hash, text), seen) in missed.into_iter().zip(self.cas.contains_many(&hashes)) {
            report.cas_known += u64::from(seen);
            let first = self.misses.insert(hash, UNSEEN).is_none();
            // Stateful embedders: ingest the new chunk texts into the
            // model BEFORE embedding them — except transcript chunks:
            // the `sessions` source is re-rendered every few minutes and its
            // text (tool output, JSON, paths) kept drifting the vocabulary away
            // from the code rows embedded earlier. Transcripts are embedded with
            // the model as it is; only code trains it.
            if first && !seen && self.repo != NO_OBSERVE_REPO {
                self.new_texts.push(text);
            }
        }
        Ok(())
    }

    /// Observe, then the second pass: every chunk added becomes a row of
    /// `plane`, in the order added.
    fn finish(self, plane: &mut PlaneWriter, report: &mut EmbedReport) -> anyhow::Result<()> {
        let RepoPass { cas, embedder, repo, mut works, n_works, mut misses, new_texts, .. } = self;
        if !new_texts.is_empty() {
            embedder.observe(&new_texts)?;
        }
        drop(new_texts);
        let mut round: Vec<String> = Vec::new();
        let mut pending: VecDeque<Pending> = VecDeque::new();
        let mut chunks = works.reader()?;
        for _ in 0..n_works {
            let (hit, w) = read_work(&mut chunks)?;
            let (header, body) = split_header_body(&w.text);
            plane.bm25.push(header, body)?;
            let source = if hit {
                Source::Cache
            } else {
                let state = misses.get_mut(&w.hash).with_context(|| format!("no embedding for chunk in {}", w.path))?;
                if *state == UNSEEN {
                    // the first chunk with this text: duplicate misses (the
                    // same chunk text in two files) take its row — the
                    // seen-only cache never hands vectors back
                    *state = QUEUED;
                    round.push(w.text);
                    Source::Round
                } else {
                    Source::Earlier
                }
            };
            pending.push_back(Pending { path: w.path, start_line: w.start_line, end_line: w.end_line, hash: w.hash, source });
            // rows wait for nothing when no miss is outstanding
            if round.is_empty() || round.len() == EMBED_ROUND {
                write_rows(cas, embedder, repo, plane, &mut pending, &mut round, &mut misses, report)?;
            }
        }
        write_rows(cas, embedder, repo, plane, &mut pending, &mut round, &mut misses, report)?;
        cas.flush();
        Ok(())
    }
}

/// Embed the texts of `round` and write the rows of `pending`, in order:
/// every vector they wait for is there once the round is embedded.
#[allow(clippy::too_many_arguments)]
fn write_rows(
    cas: &EmbedCas,
    embedder: &dyn Embedder,
    repo: &str,
    plane: &mut PlaneWriter,
    pending: &mut VecDeque<Pending>,
    round: &mut Vec<String>,
    misses: &mut HashMap<[u8; 16], u32>,
    report: &mut EmbedReport,
) -> anyhow::Result<()> {
    // a batch that failed stops the pass where its first vector is due:
    // the vectors of the batches before it are in the cache by then, as
    // they were when every batch was collected before the first put
    let mut batches = embed_round(embedder, round).into_iter();
    round.clear();
    let mut vectors = Vec::new().into_iter();
    for p in pending.drain(..) {
        let meta = RowMeta { chunk_hash: p.hash, repo, path: &p.path, start_line: p.start_line, end_line: p.end_line };
        let missing = || format!("no embedding for chunk in {}", p.path);
        match p.source {
            Source::Cache => {
                let v = cas.get(&p.hash).with_context(missing)?;
                plane.vec.push(&meta, &v)?;
            }
            Source::Round => {
                let v = loop {
                    if let Some(v) = vectors.next() {
                        break v;
                    }
                    vectors = batches.next().with_context(missing)??.into_iter();
                };
                cas.put(&p.hash, &v);
                report.embedded += 1;
                let row = plane.vec.push(&meta, &v)?;
                misses.insert(p.hash, row);
            }
            Source::Earlier => {
                let row = misses.get(&p.hash).copied().filter(|&r| r < QUEUED).with_context(missing)?;
                plane.vec.push_copy_of(&meta, row)?;
            }
        }
    }
    Ok(())
}

/// One chunk back from the spool of [`RepoPass`]: (the cache had it, the chunk).
fn read_work(r: &mut impl std::io::Read) -> anyhow::Result<(bool, Work)> {
    let mut head = [0u8; 1 + 16 + 4 + 4];
    r.read_exact(&mut head)?;
    let mut hash = [0u8; 16];
    hash.copy_from_slice(&head[1..17]);
    let mut string = || -> anyhow::Result<String> {
        let mut len = [0u8; 4];
        r.read_exact(&mut len)?;
        let mut bytes = vec![0u8; u32::from_le_bytes(len) as usize];
        r.read_exact(&mut bytes)?;
        Ok(String::from_utf8(bytes)?)
    };
    let path = string()?;
    let text = string()?;
    Ok((
        head[0] != 0,
        Work {
            path,
            start_line: u32::from_le_bytes(head[17..21].try_into().unwrap()),
            end_line: u32::from_le_bytes(head[21..25].try_into().unwrap()),
            text,
            hash,
        },
    ))
}

/// Embed `texts`: parallel batches of <= EMBED_BATCH, in order, one vector
/// per text of a batch.
fn embed_round(embedder: &dyn Embedder, texts: &[String]) -> Vec<anyhow::Result<Vec<Vec<f32>>>> {
    let dim = embedder.dim();
    texts
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
        .collect()
}

/// Source whose chunks are embedded but never fed to a stateful embedder's
/// model (the rendered session transcripts).
pub const NO_OBSERVE_REPO: &str = "sessions";

/// `<vec>/<model>.built`: the model's text count when the plane was last
/// rebuilt whole. Written by [`rebuild_all`].
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
/// drifted from the state its rows were embedded with: a
/// stamp exists, the model has grown by [`REEMBED_GROWTH`] since, and the
/// stamp is older than [`REEMBED_MIN_AGE`]. Without a stamp (a plane built
/// by an older version, or by a stateless embedder) nothing is due.
pub fn reembed_due(data_dir: &Path, model_id: &str, texts_seen: Option<u64>) -> bool {
    let Some(now) = texts_seen else { return false };
    let p = built_stamp_path(&vec_dir(data_dir), model_id);
    let Ok(meta) = std::fs::metadata(&p) else { return false };
    let young = meta.modified().ok().and_then(|m| m.elapsed().ok()).is_some_and(|age| age < REEMBED_MIN_AGE);
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

/// A segment under 1/this of the largest one's rows is folded into the
/// others on a compaction; bigger ones are left as they are.
const SMALL_SEGMENT_RATIO: usize = 8;

/// Age past which half of a segment pair is an orphan (its writer died
/// between the two files) rather than a write in flight.
pub const ORPHAN_AGE: Duration = Duration::from_secs(10 * 60);

/// The committed segments of the semantic plane: the vector and BM25 files
/// that exist as a pair (same stem), in segment order, plus the unpaired
/// files. A writer creates the BM25 file first and the vector file last,
/// so the vector rename is the commit point; a half-written pair is simply
/// not part of the plane yet, and a pair whose writer died never becomes
/// one (it is an orphan, see [`reap_orphans`]).
pub struct PlanePaths {
    pub vec: Vec<PathBuf>,
    pub bm25: Vec<PathBuf>,
    pub orphans: Vec<PathBuf>,
}

pub fn plane_paths(data_dir: &Path, model_id: &str) -> PlanePaths {
    let (vd, bd) = (vec_dir(data_dir), bm25_dir(data_dir));
    let vs = crate::index::segment_paths(&vd, model_id);
    let mut bs: HashSet<PathBuf> = crate::bm25::segment_paths(&bd, model_id).into_iter().collect();
    let mut out = PlanePaths {
        vec: Vec::with_capacity(vs.len()),
        bm25: Vec::with_capacity(vs.len()),
        orphans: Vec::new(),
    };
    for v in vs {
        // the legacy base pairs with the legacy BM25 base; deltas by stem
        let b = if v == VecIndex::file_path(&vd, model_id) {
            crate::bm25::file_path(&bd, model_id)
        } else {
            crate::bm25::aligned_path(&bd, &v)
        };
        if bs.remove(&b) {
            out.vec.push(v);
            out.bm25.push(b);
        } else {
            out.orphans.push(v);
        }
    }
    out.orphans.extend(bs);
    out
}

/// Remove (or park, when still mapped) unpaired segment files older than
/// `min_age`: what a writer that died between its BM25 and vector writes
/// left behind. Their rows were never part of the plane; the docs whose
/// old rows that writer had already tombstoned show up in
/// [`unembedded_docs`] and the coverage repair embeds them again.
pub fn reap_orphans(data_dir: &Path, model_id: &str, min_age: Duration) -> usize {
    let mut n = 0;
    for p in plane_paths(data_dir, model_id).orphans {
        let old = std::fs::metadata(&p)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age >= min_age);
        if old && crate::index::remove_or_park(&p).is_ok() {
            tracing::warn!(path = %p.display(), "removed an orphaned semantic segment (its writer died mid-write)");
            n += 1;
        }
    }
    n
}

/// Create one segment pair: BM25 first, the vector file last (the commit).
fn create_pair(
    seg: &Path,
    bseg: &Path,
    write_vec: impl FnOnce(&Path) -> anyhow::Result<()>,
    write_bm25: impl FnOnce(&Path) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    write_bm25(bseg)?;
    if let Err(e) = write_vec(seg) {
        std::fs::remove_file(bseg).ok();
        return Err(e);
    }
    Ok(())
}

/// The open vector + BM25 segment sets (committed pairs only), or `None`
/// before the first embed. Rows line up by construction; a row-count
/// mismatch inside a pair means a damaged file and is an error.
fn open_sets(data_dir: &Path, model_id: &str) -> anyhow::Result<Option<(VecSet, Bm25Set)>> {
    let pp = plane_paths(data_dir, model_id);
    let vs = VecSet::reopen_paths(None, &vec_dir(data_dir), model_id, &pp.vec)?;
    let bs = Bm25Set::reopen_paths(None, &pp.bm25)?;
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

/// Embed `repos` INCREMENTALLY: every visible doc of each repo is
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

/// Visible docs with no live row in the semantic plane, as (repo, path):
/// what a lexical-only writer (the Bash hook's pre-deny reindex, a CLI
/// `reindex`, an older binary) indexed and nothing embedded since. Their
/// files are invisible to the semantic and BM25 legs. Empty without a plane.
pub fn unembedded_docs(shards: &ShardSet, data_dir: &Path, model_id: &str) -> anyhow::Result<Vec<(String, String)>> {
    let Some((vset, _)) = open_sets(data_dir, model_id)? else {
        return Ok(Vec::new());
    };
    // one entry per file, borrowed from the segments: the rows are not walked
    let have: HashSet<(&str, &str)> = vset.live_files().collect();
    Ok(shards
        .visible_docs()
        .into_iter()
        .map(|(si, _, dm)| (doc_repo(shards, si, dm.repo_id), dm.path))
        .filter(|(repo, path)| !have.contains(&(repo.as_str(), path.as_str())))
        .collect())
}

/// Embed `docs` (repo, path) — the [`unembedded_docs`] repair. Docs that
/// chunk to nothing (an empty `__init__.py`) stay unembedded.
pub fn embed_docs(
    shards: &ShardSet,
    data_dir: &Path,
    docs: &[(String, String)],
    embedder: &dyn Embedder,
    max_chars: usize,
) -> anyhow::Result<Vec<EmbedReport>> {
    let mut repos: Vec<String> = docs.iter().map(|(r, _)| r.clone()).collect();
    repos.sort();
    repos.dedup();
    if repos.is_empty() {
        return Ok(Vec::new());
    }
    // paths of other repos in the set are carried unchanged (same hashes)
    let paths: HashSet<String> = docs.iter().map(|(_, p)| p.clone()).collect();
    embed_incremental(shards, data_dir, &repos, Some(&paths), embedder, max_chars)
}

/// Embed only `paths` of `repo` (auto-refresh): the docs at those
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

/// `<vec>/<model>.embed-error`: the last failed embed's message, removed by
/// the next one that succeeds. Embeds run on background threads whose
/// warnings nobody reads, so `index_stats` shows this instead.
fn embed_error_path(data_dir: &Path, model_id: &str) -> PathBuf {
    vec_dir(data_dir).join(format!("{model_id}.embed-error"))
}

/// The last embed failure (message, age) if the plane is failing to update.
pub fn last_embed_error(data_dir: &Path, model_id: &str) -> Option<(String, Duration)> {
    let p = embed_error_path(data_dir, model_id);
    let msg = std::fs::read_to_string(&p).ok()?;
    let age = std::fs::metadata(&p).and_then(|m| m.modified()).ok()?.elapsed().unwrap_or_default();
    Some((msg.trim().to_string(), age))
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
    let r = embed_incremental_inner(shards, data_dir, repos, only_paths, given, embedder, max_chars);
    let marker = embed_error_path(data_dir, embedder.model_id());
    match &r {
        Ok(_) => {
            if marker.exists() {
                std::fs::remove_file(&marker).ok();
            }
        }
        Err(e) => {
            std::fs::write(&marker, format!("{e:#}")).ok();
        }
    }
    r
}

fn embed_incremental_inner(
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
        None => {
            reap_orphans(data_dir, &model_id, ORPHAN_AGE);
            open_sets(data_dir, &model_id)?
        }
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
    // Only the requested repos' runs are walked, file by file; a row is
    // looked at (its hash alone) once its file passed the filters, so an
    // edit of one file does not read the metadata of the whole plane.
    let mut existing: HashMap<(String, String), Vec<(u32, [u8; 16])>> = HashMap::new();
    if let Some((vset, _)) = &sets {
        for (i, seg) in vset.segments().iter().enumerate() {
            let (off, dead) = (vset.offset(i), vset.dead(i));
            for (repo, repo_rows) in seg.repo_runs() {
                if !requested.contains(repo) {
                    continue;
                }
                for (file, rows) in seg.file_runs(repo_rows) {
                    if only_paths.is_some_and(|p| !p.contains(file.path)) {
                        continue;
                    }
                    let mut live = rows.filter(|r| !dead.contains(r)).map(|r| (off + r, seg.row_hash(r))).peekable();
                    if live.peek().is_some() {
                        existing.entry((file.repo.to_string(), file.path.to_string())).or_default().extend(live);
                    }
                }
            }
        }
    }

    let mut doomed: HashSet<u32> = HashSet::new();
    // one delta segment for every repo of the call, written as rows arrive
    let mut plane = PlaneWriter::new(data_dir, embedder.dim())?;
    let mut reports = Vec::with_capacity(repos.len());
    for repo in repos {
        let t0 = Instant::now();
        let mut report = EmbedReport {
            repo: repo.to_string(),
            ..Default::default()
        };
        // files in path order: the order their rows take in the segment
        // (docs of one path, should there be several, keep theirs)
        let mut docs = repo_docs(shards, repo, only_paths);
        docs.sort_by(|a, b| a.2.path.cmp(&b.2.path));
        let mut pass = RepoPass::new(&cas, embedder, repo, data_dir);
        let mut rest = &docs[..];
        while !rest.is_empty() {
            let mut n = next_batch(rest);
            // a file is compared whole
            while n < rest.len() && rest[n].2.path == rest[n - 1].2.path {
                n += 1;
            }
            let (batch, later) = rest.split_at(n);
            rest = later;
            let mut chunked = chunk_docs(shards, batch, max_chars)?.into_iter();
            let mut changed: Vec<Work> = Vec::new();
            let mut at = 0;
            while at < batch.len() {
                let path = &batch[at].2.path;
                let mut ws: Vec<Work> = Vec::new();
                while at < batch.len() && batch[at].2.path == *path {
                    ws.extend(chunked.next().expect("one chunk list per doc"));
                    at += 1;
                }
                report.chunks += ws.len() as u64;
                if ws.is_empty() {
                    // no chunk, no rows: what the file had goes with the
                    // files that vanished
                    continue;
                }
                let old = existing.remove(&(repo.to_string(), path.clone()));
                let mut new_hashes: Vec<[u8; 16]> = ws.iter().map(|w| w.hash).collect();
                new_hashes.sort_unstable();
                let same = old.as_ref().is_some_and(|o| {
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
            pass.add(changed, &mut report)?;
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
        pass.finish(&mut plane, &mut report)?;
        report.elapsed_ms = t0.elapsed().as_millis();
        reports.push(report);
    }

    let t_idx = Instant::now();
    if let (false, Some((vset, bset))) = (doomed.is_empty(), sets) {
        vset.tombstone_on_disk(&doomed)?;
        bset.tombstone_on_disk(&doomed)?;
    }
    drop(owned);
    let appended = plane.rows();
    if appended > 0 {
        let seg = crate::index::next_delta_path(&vec_dir(data_dir), &model_id);
        let bseg = crate::bm25::aligned_path(&bm25_dir(data_dir), &seg);
        plane.commit(&seg, &bseg)?;
    }
    embedder.flush()?;
    let build_ms = t_idx.elapsed().as_millis();
    info!(
        repos = repos.len(),
        tombstoned = doomed.len(),
        appended,
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
///
/// Rows go to the segment writers as each repo is embedded: what is held
/// is a batch of chunks, a round of vectors and the writers' buffers, not
/// the rows of the corpus (8.2 KB of f32 and the chunk text for each of
/// them, until every repo was done).
pub fn rebuild_all(
    shards: &ShardSet,
    data_dir: &Path,
    repos: &[String],
    embedder: &dyn Embedder,
    max_chars: usize,
) -> anyhow::Result<Vec<EmbedReport>> {
    let model_id = embedder.model_id().to_string();
    let cas = EmbedCas::open_for(data_dir, embedder);
    let mut plane = PlaneWriter::new(data_dir, embedder.dim())?;
    let mut reports = Vec::with_capacity(repos.len());
    for repo in repos {
        let t0 = Instant::now();
        let mut report = EmbedReport {
            repo: repo.to_string(),
            ..Default::default()
        };
        let docs = repo_docs(shards, repo, None);
        let mut pass = RepoPass::new(&cas, embedder, repo, data_dir);
        let mut rest = &docs[..];
        while !rest.is_empty() {
            let (batch, later) = rest.split_at(next_batch(rest));
            rest = later;
            let works: Vec<Work> = chunk_docs(shards, batch, max_chars)?.into_iter().flatten().collect();
            report.chunks += works.len() as u64;
            pass.add(works, &mut report)?;
        }
        pass.finish(&mut plane, &mut report)?;
        report.elapsed_ms = t0.elapsed().as_millis();
        reports.push(report);
    }
    let t_idx = Instant::now();
    let rows = plane.rows();
    let old_v = crate::index::segment_paths(&vec_dir(data_dir), &model_id);
    let old_b = crate::bm25::segment_paths(&bm25_dir(data_dir), &model_id);
    let seg = crate::index::next_delta_path(&vec_dir(data_dir), &model_id);
    let bseg = crate::bm25::aligned_path(&bm25_dir(data_dir), &seg);
    plane.commit(&seg, &bseg)?;
    for p in old_v.iter().chain(old_b.iter()) {
        crate::index::remove_or_park(p)?;
    }
    embedder.flush()?;
    // the model state every row was embedded with
    if let Some(n) = embedder.texts_seen() {
        write_built_stamp(&vec_dir(data_dir), &model_id, n);
    }
    let build_ms = t_idx.elapsed().as_millis();
    info!(
        repos = repos.len(),
        rows,
        build_ms = build_ms as u64,
        "embed full rebuild"
    );
    for r in &mut reports {
        r.index_build_ms = build_ms;
    }
    Ok(reports)
}

/// Fold segments together: with more than [`MAX_SEGMENTS`]
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
/// of the thresholds (re-encodes older f32 segments as int8).
pub fn compact_vectors_opts(data_dir: &Path, model_id: &str, force: bool) -> anyhow::Result<bool> {
    crate::index::reap_stale(&vec_dir(data_dir));
    crate::index::reap_stale(&bm25_dir(data_dir));
    reap_orphans(data_dir, model_id, ORPHAN_AGE);
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
        // the small segments (refresh deltas) alone when folding them is
        // enough: every big one is carried as it is instead of re-encoded
        // (13 → 3 segments wrote 0.4 MB, where all-but-the-largest rewrote a
        // 535 MB segment: 11 s and 2.8 GB of memory per compaction)
        let big = vset.segments()[largest].raw_len();
        let small: Vec<usize> = (0..n).filter(|&i| vset.segments()[i].raw_len() * SMALL_SEGMENT_RATIO < big).collect();
        if small.len() >= 2 && n - small.len() < MAX_SEGMENTS {
            small
        } else {
            (0..n).filter(|&i| i != largest).collect()
        }
    } else {
        return Ok(false);
    };
    if merge.is_empty() || (merge.len() < 2 && dead <= MAX_DEAD_FRACTION && !force) {
        return Ok(false);
    }
    let t0 = Instant::now();
    let dim = vset.dim();
    // The rows carried, in merge order (segment by segment, row by row),
    // each with what it sorts by: its file and its first line. A file is
    // looked at once per run of its rows, a row for its line alone; the
    // rows themselves stay in the mapped segments until they are written.
    struct Carried {
        file: u32,
        start_line: u32,
        /// Position of the row's segment in `merge`, and its row there.
        seg: u32,
        row: u32,
    }
    let mut files: Vec<(&str, &str)> = Vec::new();
    let mut carried: Vec<Carried> = Vec::new();
    for (k, &i) in merge.iter().enumerate() {
        let vs = &vset.segments()[i];
        let dead = vset.dead(i);
        for (file, rows) in vs.file_runs(0..vs.raw_len() as u32) {
            let mut listed = false;
            for r in rows {
                if dead.contains(&r) {
                    continue;
                }
                if !listed {
                    files.push((file.repo, file.path));
                    listed = true;
                }
                carried.push(Carried { file: files.len() as u32 - 1, start_line: vs.row_meta(r).start_line, seg: k as u32, row: r });
            }
        }
    }
    // group rows by repo (then path, line) so a repo-scoped search scans
    // one run of the merged segment; rows that tie keep their merge order.
    // BM25 rows follow the same order (rows are aligned across the sidecars)
    {
        let mut by_name: Vec<u32> = (0..files.len() as u32).collect();
        by_name.sort_unstable_by_key(|&f| files[f as usize]);
        let mut rank = vec![0u32; files.len()];
        for j in 1..by_name.len() {
            let (prev, f) = (by_name[j - 1] as usize, by_name[j] as usize);
            rank[f] = rank[prev] + u32::from(files[f] != files[prev]);
        }
        carried.sort_unstable_by_key(|c| (rank[c.file as usize], c.start_line, c.seg, c.row));
    }
    drop(files);
    // order[new row] = (segment, local row): the rows to write, and where
    // tombstones another server writes to an input while we merge are
    // forwarded afterwards
    let order: Vec<(u32, u32)> = carried.iter().map(|c| (c.seg, c.row)).collect();
    drop(carried);
    // per input, where each of its rows goes in the merged BM25 segment: a
    // row tombstoned on the BM25 side alone is carried without its terms
    let mut row_maps: Vec<Vec<u32>> = merge
        .iter()
        .map(|&i| vec![crate::bm25::DROPPED; bset.segments()[i].raw_len()])
        .collect();
    for (new_row, &(k, r)) in order.iter().enumerate() {
        if !bset.dead(merge[k as usize]).contains(&r) {
            row_maps[k as usize][r as usize] = new_row as u32;
        }
    }
    let merged_v: Vec<PathBuf> = merge.iter().map(|&i| vset.segments()[i].path().to_path_buf()).collect();
    let merged_b: Vec<PathBuf> = merge.iter().map(|&i| bset.segments()[i].path().to_path_buf()).collect();
    let seg = crate::index::next_delta_path(&vec_dir(data_dir), model_id);
    let bseg = crate::bm25::aligned_path(&bm25_dir(data_dir), &seg);
    {
        let inputs_v: Vec<&VecIndex> = merge.iter().map(|&i| &*vset.segments()[i]).collect();
        let inputs_b: Vec<(&crate::bm25::ChunkBm25, crate::bm25::RowMap<'_>)> = merge
            .iter()
            .zip(&row_maps)
            .map(|(&i, map)| (&*bset.segments()[i], crate::bm25::RowMap::Table(map)))
            .collect();
        create_pair(
            &seg,
            &bseg,
            |p| crate::index::write_carried(p, dim, &inputs_v, &order),
            |p| crate::bm25::merge_segments(p, order.len() as u32, &inputs_b),
        )?;
    }
    drop(row_maps);
    // the inputs are read from their mappings up to here; unmapped, they
    // can be removed instead of parked
    drop(vset);
    drop(bset);
    // rows tombstoned in an input since we read it
    let late: std::collections::BTreeSet<u32> = {
        let mut now: Vec<std::collections::HashSet<u32>> = Vec::with_capacity(merge.len());
        for p in &merged_v {
            now.push(crate::index::read_tomb_side(p).into_iter().collect());
        }
        order
            .iter()
            .enumerate()
            .filter(|(_, &(k, r))| now[k as usize].contains(&r))
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
        rows = order.len(),
        ms = t0.elapsed().as_millis() as u64,
        "semantic segments compacted"
    );
    Ok(true)
}

/// The cross-process compaction lock lives in [`crate::lock`]; re-exported
/// here, where callers have always found it.
pub use crate::lock::{CompactLock, LOCK_STALE_SECS};

/// Whether a semantic plane for `model_id` exists at all:
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

/// `embed_repo` with a caller-chosen chunk size cap (CLI
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
// The pipeline before segments were streamed
// ---------------------------------------------------------------------------

/// The embed and compaction passes as they were when every row of a pass
/// was held in memory until its segment was written: each repo's chunks
/// with their text three times over, a `Vec<f32>` per row, the term maps of
/// every BM25 row. Kept, as they were, for the tests that run them next to
/// the streaming passes and compare the files written. They write through
/// the writers kept in `index::reference` and `bm25::reference`.
#[cfg(test)]
mod reference {
    use std::collections::BTreeMap;

    use super::*;
    use crate::index::VecRowMeta;

    /// The `INDEXIO_VEC_F32` knob, as `create_segment` read it.
    fn f32_rows() -> bool {
        std::env::var_os("INDEXIO_VEC_F32").is_some()
    }

    fn split_header_body(text: &str) -> (String, String) {
        match text.split_once('\n') {
            Some((h, b)) => (h.to_string(), b.to_string()),
            None => (text.to_string(), String::new()),
        }
    }

    /// One chunk awaiting a vector.
    struct Work {
        meta: VecRowMeta,
        text: String,
        /// (header, body) of the chunk, for the BM25F sidecar.
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
                paths.is_none_or(|p| p.contains(&dm.path)) && doc_repo(shards, *si, dm.repo_id) == repo
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
        // A seen-only cache misses every chunk but still knows which
        // ones the model observed before: those are re-embedded, not re-observed.
        let mut known: Vec<bool> = vec![false; works.len()];
        for (i, w) in works.iter().enumerate() {
            if let Some(v) = cas.get(&w.hash) {
                report.cas_hits += 1;
                vecs[i] = Some(v);
            } else {
                report.cas_misses += 1;
                miss_all.push(i);
            }
        }
        // all misses in one question: a seen-only cache answers it with one
        // pass over its file instead of loading every hash the model ever saw
        let miss_hashes: Vec<[u8; 16]> = miss_all.iter().map(|&i| works[i].hash).collect();
        for (&i, seen) in miss_all.iter().zip(cas.contains_many(&miss_hashes)) {
            known[i] = seen;
            report.cas_known += u64::from(seen);
        }
        let mut seen: HashSet<[u8; 16]> = HashSet::new();
        let miss_uniq: Vec<usize> = miss_all
            .iter()
            .copied()
            .filter(|&i| seen.insert(works[i].hash))
            .collect();

        // Stateful embedders: ingest the new chunk texts into the
        // model BEFORE embedding them — except transcript chunks:
        // the `sessions` source is re-rendered every few minutes and its
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
        // never hands vectors back
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

    pub fn embed_incremental_inner(
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
        let owned: Option<(VecSet, Bm25Set)> = match given {
            Some(_) => None,
            None => {
                reap_orphans(data_dir, &model_id, ORPHAN_AGE);
                open_sets(data_dir, &model_id)?
            }
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

        let mut existing: HashMap<(String, String), Vec<(u32, [u8; 16])>> = HashMap::new();
        if let Some((vset, _)) = &sets {
            for (i, seg) in vset.segments().iter().enumerate() {
                let (off, dead) = (vset.offset(i), vset.dead(i));
                for (repo, repo_rows) in seg.repo_runs() {
                    if !requested.contains(repo) {
                        continue;
                    }
                    for (file, rows) in seg.file_runs(repo_rows) {
                        if only_paths.is_some_and(|p| !p.contains(file.path)) {
                            continue;
                        }
                        let mut live = rows.filter(|r| !dead.contains(r)).map(|r| (off + r, seg.row_hash(r))).peekable();
                        if live.peek().is_some() {
                            existing.entry((file.repo.to_string(), file.path.to_string())).or_default().extend(live);
                        }
                    }
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
                let same = old.as_ref().is_some_and(|o| {
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
            create_pair(
                &seg,
                &bseg,
                |p| crate::index::reference::create_segment(p, embedder.dim(), new_rows, f32_rows()),
                |p| crate::bm25::reference::create_path(p, &new_bm25),
            )?;
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
        create_pair(
            &seg,
            &bseg,
            |p| crate::index::reference::create_segment(p, embedder.dim(), rows, f32_rows()),
            |p| crate::bm25::reference::create_path(p, &bm25_rows),
        )?;
        for p in old_v.iter().chain(old_b.iter()) {
            crate::index::remove_or_park(p)?;
        }
        embedder.flush()?;
        // the model state every row was embedded with
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

    pub fn compact_vectors_opts(data_dir: &Path, model_id: &str, force: bool) -> anyhow::Result<bool> {
        crate::index::reap_stale(&vec_dir(data_dir));
        crate::index::reap_stale(&bm25_dir(data_dir));
        reap_orphans(data_dir, model_id, ORPHAN_AGE);
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
            let big = vset.segments()[largest].raw_len();
            let small: Vec<usize> = (0..n).filter(|&i| vset.segments()[i].raw_len() * SMALL_SEGMENT_RATIO < big).collect();
            if small.len() >= 2 && n - small.len() < MAX_SEGMENTS {
                small
            } else {
                (0..n).filter(|&i| i != largest).collect()
            }
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
        // what an int8 input stores for each carried row: a compaction
        // copies it (scale, values, code) instead of quantizing again
        let mut stored: Vec<Option<(f32, Vec<u8>, Vec<u8>)>> = Vec::new();
        for &i in &merge {
            let vs = &vset.segments()[i];
            let mut bs_rows = crate::bm25::reference::rows_tf_of(&bset, i);
            let off = vset.offset(i);
            for r in 0..vs.raw_len() as u32 {
                if vset.is_deleted(off + r) {
                    continue;
                }
                origin.push((i, r));
                rows.push((vs.row_meta(r).to_owned_meta(), vs.row_vector(r)));
                stored.push(vs.stored_quant(r).map(|(s, q, c)| (s, q.to_vec(), c.to_vec())));
                tf_rows.push(std::mem::take(&mut bs_rows[r as usize]));
            }
        }
        // group rows by repo (then path, line) so a repo-scoped search scans
        // one run of the merged segment; BM25 rows and origins
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
            let mut stored_s = Vec::with_capacity(stored.len());
            let mut rows_opt: Vec<Option<(VecRowMeta, Vec<f32>)>> = rows.into_iter().map(Some).collect();
            let mut tf_opt: Vec<Option<(BTreeMap<String, u32>, BTreeMap<String, u32>)>> = tf_rows.into_iter().map(Some).collect();
            for i in order {
                rows_s.push(rows_opt[i].take().expect("row moved once"));
                tf_s.push(tf_opt[i].take().expect("tf row moved once"));
                origin_s.push(origin[i]);
                stored_s.push(stored[i].take());
            }
            stored = stored_s;
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
        create_pair(
            &seg,
            &bseg,
            |p| {
                crate::index::reference::create_segment(p, dim, rows, f32_rows())?;
                if f32_rows() {
                    return Ok(());
                }
                let mut file = std::fs::read(p)?;
                let (scale_off, q_off, code_off) = crate::index::reference::v3_sections(&file);
                let cb = dim.div_ceil(8);
                for (row, kept) in stored.iter().enumerate() {
                    let Some((scale, q, code)) = kept else { continue };
                    file[scale_off + 4 * row..scale_off + 4 * row + 4].copy_from_slice(&scale.to_le_bytes());
                    file[q_off + row * dim..q_off + (row + 1) * dim].copy_from_slice(q);
                    file[code_off + row * cb..code_off + (row + 1) * cb].copy_from_slice(code);
                }
                Ok(std::fs::write(p, file)?)
            },
            |p| crate::bm25::reference::create_from_tf(p, tf_rows),
        )?;
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
}

// ---------------------------------------------------------------------------
// Tests — these call indexio_symbols::chunking::chunks, currently a `todo!()` stub
// being implemented on a parallel branch. They are written in full and marked
// indexio-embed -- --ignored`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::ChunkBm25;
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
        // the shared file is a miss the cache recognises
        let rep2 = embed_repo(&shards, tmp.path(), "r2", &e).unwrap();
        assert_eq!(rep2.cas_hits, 0, "seen-only cache never serves vectors: {rep2:?}");
        assert!(rep2.cas_known > 0, "shared file is known to the cache: {rep2:?}");
        assert!(rep2.cas_known < rep2.chunks, "only the shared chunks are known: {rep2:?}");
        // every miss is embedded (unique hashes at least once)
        assert!(rep2.embedded > 0 && rep2.embedded <= rep2.cas_misses);

        // Re-embedding an unchanged repo carries every row: no
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
        // Regression: the seen-only cache never returns
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

    /// A doc indexed after the last embed (a lexical-only writer) has no
    /// rows until the repair embeds it; an empty doc stays unembedded, and
    /// other docs' rows are untouched.
    #[test]
    fn coverage_repair_embeds_docs_without_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let v1 = build_shardset(tmp.path(), &["r1", "r2"], &[(0, "src/a.rs", b"fn alpha_function() {}\n"), (1, "src/b.rs", b"fn beta_function() {}\n")]);
        let e = HashEmbedder::new(512);
        embed_repo(&v1, tmp.path(), "r1", &e).unwrap();
        embed_repo(&v1, tmp.path(), "r2", &e).unwrap();
        drop(v1);
        for f in std::fs::read_dir(tmp.path().join("shards")).unwrap() {
            let p = f.unwrap().path();
            if p.extension().and_then(|e| e.to_str()) == Some("cidx") {
                std::fs::remove_file(p).unwrap();
            }
        }
        let v2 = build_shardset(
            tmp.path(),
            &["r1", "r2"],
            &[(0, "src/a.rs", b"fn alpha_function() {}\n"), (0, "src/new.rs", b"fn gamma_function() {}\n"), (0, "src/empty.py", b""), (1, "src/b.rs", b"fn beta_function() {}\n")],
        );
        let mut missing = unembedded_docs(&v2, tmp.path(), "hash-v1").unwrap();
        missing.sort();
        assert!(missing.contains(&("r1".into(), "src/new.rs".into())), "{missing:?}");
        assert!(!missing.iter().any(|(_, p)| p == "src/a.rs" || p == "src/b.rs"), "{missing:?}");
        let live_before = VecSet::open(&tmp.path().join("vec"), "hash-v1").unwrap().unwrap().len();
        embed_docs(&v2, tmp.path(), &missing, &e, MAX_CHARS).unwrap();
        let after = unembedded_docs(&v2, tmp.path(), "hash-v1").unwrap();
        assert!(after.iter().all(|(_, p)| p == "src/empty.py"), "{after:?}");
        let idx = VecSet::open(&tmp.path().join("vec"), "hash-v1").unwrap().unwrap();
        assert_eq!(idx.len(), live_before + 1, "only new.rs was added");
        // nothing left to do: no new segment
        let segs = idx.segments().len();
        embed_docs(&v2, tmp.path(), &after, &e, MAX_CHARS).unwrap();
        assert_eq!(VecSet::open(&tmp.path().join("vec"), "hash-v1").unwrap().unwrap().segments().len(), segs);
    }

    /// A writer that died between its two segment files (seen live: a
    /// vector delta without its BM25 twin wedged every later embed with
    /// "out of sync; retry" for ten days) leaves an orphan that is not part
    /// of the plane: embeds keep working, the orphan is reaped once old,
    /// and its docs are re-embedded by the coverage repair.
    #[test]
    fn orphaned_half_pair_neither_blocks_embeds_nor_survives() {
        let tmp = tempfile::tempdir().unwrap();
        let shards = build_shardset(tmp.path(), &["r1", "r2"], &[(0, "src/a.rs", b"fn alpha_function() {}\n"), (1, "src/b.rs", b"fn beta_function() {}\n")]);
        let e = HashEmbedder::new(512);
        embed_repo(&shards, tmp.path(), "r1", &e).unwrap();
        embed_repo(&shards, tmp.path(), "r2", &e).unwrap();
        // kill the newest pair's BM25 half (r2's rows)
        let pp = plane_paths(tmp.path(), "hash-v1");
        assert_eq!((pp.vec.len(), pp.bm25.len(), pp.orphans.len()), (2, 2, 0));
        std::fs::remove_file(&pp.bm25[1]).unwrap();
        let pp = plane_paths(tmp.path(), "hash-v1");
        assert_eq!((pp.vec.len(), pp.orphans.len()), (1, 1));
        // the plane opens (committed pairs only) and an embed succeeds
        let (v, b) = open_sets(tmp.path(), "hash-v1").unwrap().unwrap();
        assert_eq!(v.raw_len(), b.raw_len());
        drop((v, b));
        embed_repo(&shards, tmp.path(), "r1", &e).unwrap();
        assert!(last_embed_error(tmp.path(), "hash-v1").is_none());
        // young orphans are left alone (a write may be in flight) ...
        assert_eq!(reap_orphans(tmp.path(), "hash-v1", ORPHAN_AGE), 0);
        // ... old ones are removed, and the repair restores r2's rows
        assert_eq!(reap_orphans(tmp.path(), "hash-v1", Duration::ZERO), 1);
        assert!(plane_paths(tmp.path(), "hash-v1").orphans.is_empty());
        let missing = unembedded_docs(&shards, tmp.path(), "hash-v1").unwrap();
        assert_eq!(missing, vec![("r2".to_string(), "src/b.rs".to_string())]);
        embed_docs(&shards, tmp.path(), &missing, &e, MAX_CHARS).unwrap();
        assert!(unembedded_docs(&shards, tmp.path(), "hash-v1").unwrap().is_empty());
        let (v, b) = open_sets(tmp.path(), "hash-v1").unwrap().unwrap();
        assert_eq!(v.raw_len(), b.raw_len());
    }

    /// A vector file is only written once its BM25 twin exists, so a reader
    /// never sees a vector segment without BM25 rows.
    #[test]
    fn a_failed_vector_write_leaves_no_half_pair() {
        let tmp = tempfile::tempdir().unwrap();
        let (seg, bseg) = (tmp.path().join("vec").join("m.d1.civec"), tmp.path().join("bm25").join("m.d1.cibm25"));
        let r = create_pair(&seg, &bseg, |_| anyhow::bail!("disk full"), |p| ChunkBm25::create_path(p, &[("h".into(), "body".into())]));
        assert!(r.is_err());
        assert!(!seg.exists() && !bseg.exists());
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
                r1_paths.push(m.path);
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

    #[test]
    fn compaction_folds_small_segments_and_carries_the_big_one() {
        let tmp = tempfile::tempdir().unwrap();
        let mut repos: Vec<String> = vec!["big".into()];
        repos.extend((0..9).map(|k| format!("s{k}")));
        let names: Vec<&str> = repos.iter().map(String::as_str).collect();
        let bodies: Vec<(usize, String, Vec<u8>)> = (0..40)
            .map(|i| (0, format!("src/f{i}.rs"), format!("fn big_function_{i}() {{ {i} }}\n").into_bytes()))
            .chain((0..9).map(|k| (k + 1, "src/x.rs".to_string(), format!("fn small_{k}() {{}}\n").into_bytes())))
            .collect();
        let docs: Vec<(u32, &str, &[u8])> = bodies.iter().map(|(r, p, b)| (*r as u32, p.as_str(), b.as_slice())).collect();
        let shards = build_shardset(tmp.path(), &names, &docs);
        let e = HashEmbedder::new(64);
        for r in &names {
            embed_repo(&shards, tmp.path(), r, &e).unwrap();
        }
        let before = VecSet::open(&tmp.path().join("vec"), "hash-v1").unwrap().unwrap();
        assert_eq!(before.segments().len(), 10);
        let big_path = before.segments().iter().max_by_key(|s| s.raw_len()).unwrap().path().to_path_buf();
        let live = (0..before.raw_len() as u32).filter(|&r| !before.is_deleted(r)).count();
        drop(before);
        assert!(compact_vectors(tmp.path(), "hash-v1").unwrap());
        let after = VecSet::open(&tmp.path().join("vec"), "hash-v1").unwrap().unwrap();
        assert_eq!(after.segments().len(), 2, "the nine small ones folded into one");
        assert!(big_path.exists(), "the big segment is carried as it is");
        assert_eq!((0..after.raw_len() as u32).filter(|&r| !after.is_deleted(r)).count(), live);
        let q = e.embed(&["fn small_4".to_string()]).unwrap().remove(0);
        assert_eq!(after.row_meta(after.search(&q, 1)[0].0).repo, "s4");
    }

    /// Row slots via public API (live rows + tombstoned, what the aligned
    /// BM25 set reports as `raw_len`).
    fn idx_len(idx: &VecSet) -> usize {
        idx.raw_len()
    }

    /// embed_repo writes a BM25F sidecar aligned row-for-row
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

    // ---- the streaming passes against the passes they replace -------------
    //
    // Each scenario below runs twice, in two data dirs holding the same
    // shards: once through the passes kept in `reference` (every row of a
    // pass in memory, the writers of `index::reference` and
    // `bm25::reference`), once through the passes above. After every step
    // the two planes must hold the same bytes.

    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    /// What a [`Probe`] was asked.
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum Call {
        Observe(Vec<String>),
        /// How many texts the model had observed, and the texts of the batch.
        Embed(u64, Vec<String>),
    }

    /// An embedder that shows whatever a pass does differently: a vector
    /// depends on its text and on how many texts were observed before it was
    /// computed, every call is recorded, and the vectors are dense, with
    /// components of both signs down to nothing (rows whose int8 values
    /// hold zeros that were negative).
    struct Probe {
        id: &'static str,
        dim: usize,
        cheap: bool,
        /// Fail the batch that holds a text containing this.
        fail_on: Option<&'static str>,
        observed: AtomicU64,
        calls: Mutex<Vec<Call>>,
    }

    impl Probe {
        fn new(id: &'static str, cheap: bool) -> Self {
            Probe { id, dim: 24, cheap, fail_on: None, observed: AtomicU64::new(0), calls: Mutex::new(Vec::new()) }
        }

        /// The observe calls in order, and the embed batches sorted (they
        /// run in parallel).
        fn calls(&self) -> (Vec<Call>, Vec<Call>) {
            let (observes, mut embeds): (Vec<Call>, Vec<Call>) =
                self.calls.lock().unwrap().iter().cloned().partition(|c| matches!(c, Call::Observe(_)));
            embeds.sort();
            (observes, embeds)
        }
    }

    fn probe_vector(text: &str, state: u64, dim: usize) -> Vec<f32> {
        let mut h = blake3::Hasher::new();
        h.update(&state.to_le_bytes());
        h.update(text.as_bytes());
        let mut x = u64::from_le_bytes(h.finalize().as_bytes()[..8].try_into().unwrap()) | 1;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut v: Vec<f32> = (0..dim)
            .map(|_| {
                let sign = if next() & 1 == 0 { 1.0 } else { -1.0 };
                sign * 10f32.powf(-4.0 * (next() % 1000) as f32 / 1000.0)
            })
            .collect();
        crate::embed::l2_normalize(&mut v);
        v
    }

    impl Embedder for Probe {
        fn model_id(&self) -> &str {
            self.id
        }

        fn dim(&self) -> usize {
            self.dim
        }

        fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
            let state = self.observed.load(Ordering::SeqCst);
            self.calls.lock().unwrap().push(Call::Embed(state, texts.to_vec()));
            if let Some(marker) = self.fail_on {
                anyhow::ensure!(!texts.iter().any(|t| t.contains(marker)), "the probe refuses {marker}");
            }
            Ok(texts.iter().map(|t| probe_vector(t, state, self.dim)).collect())
        }

        fn observe(&self, texts: &[String]) -> anyhow::Result<()> {
            self.observed.fetch_add(texts.len() as u64, Ordering::SeqCst);
            self.calls.lock().unwrap().push(Call::Observe(texts.to_vec()));
            Ok(())
        }

        fn recompute_is_cheap(&self) -> bool {
            self.cheap
        }

        fn texts_seen(&self) -> Option<u64> {
            Some(self.observed.load(Ordering::SeqCst))
        }
    }

    /// Source files of a generated corpus: (repo, path, content).
    type Files = Vec<(u32, String, Vec<u8>)>;

    /// The repos of a generated corpus; the transcripts' is one of them.
    const CORPUS_REPOS: [&str; 4] = ["alpha", "beta", NO_OBSERVE_REPO, "zeta"];

    fn rust_fn(name: &str, lines: usize) -> String {
        let mut s = format!("pub fn {name}(input: u32) -> u32 {{\n    let mut total = input;\n");
        for i in 0..lines {
            s.push_str(&format!("    total = total.wrapping_mul({i}).wrapping_add(helper_{name}_{i}(total));\n"));
        }
        s.push_str("    total\n}\n\n");
        s
    }

    /// A corpus with what the passes treat one by one: files of one chunk
    /// and of many, a function too long for a chunk, a chunk whose text
    /// occurs twice, a file every repo has, text that is not code, files
    /// with nothing to embed. From one `round` to the next a file and the
    /// notes are edited and a file is new; after the first some files are
    /// gone, after the second one is emptied.
    fn corpus(files_per_repo: usize, round: u32) -> Files {
        let mut files = Files::new();
        for repo in 0..CORPUS_REPOS.len() as u32 {
            for k in 0..files_per_repo {
                if k % 10 == 9 && round > 0 {
                    continue; // gone after the first round
                }
                let version = if k == 1 { round } else { 0 };
                let mut text = String::new();
                if k % 10 == 8 && round >= 2 {
                    text.push_str("\n   \n"); // emptied: nothing left to embed
                } else {
                    for j in 0..1 + k % 6 {
                        let lines = if (k + j) % 9 == 4 { 70 } else { 2 + (k * 7 + j) % 12 };
                        text.push_str(&rust_fn(&format!("r{repo}_f{k}_g{j}_v{version}"), lines));
                    }
                    if k % 7 == 2 {
                        // the same definition twice: two chunks, one text
                        let twice = rust_fn("defined_twice", 3);
                        text.push_str(&twice);
                        text.push_str(&twice);
                    }
                }
                files.push((repo, format!("src/m{}/f{k}.rs", k % 5), text.into_bytes()));
            }
            files.push((repo, "src/shared.rs".into(), rust_fn("the_same_in_every_repo", 6).into_bytes()));
            let notes: String = (0..40).map(|i| format!("Notes of repo {repo}, round {round}: line {i} says a few words about item {}.\n", i * 3)).collect();
            files.push((repo, "docs/notes.md".into(), notes.into_bytes()));
            files.push((repo, "data/empty.txt".into(), Vec::new()));
            for r in 1..=round {
                files.push((repo, format!("src/new/r{r}.rs"), rust_fn(&format!("added_in_round_{r}_to_{repo}"), 5).into_bytes()));
            }
        }
        files
    }

    /// The shard set of `files`, replacing the one in `dir`.
    fn reshard(dir: &Path, files: &Files) -> ShardSet {
        if let Ok(rd) = std::fs::read_dir(dir.join("shards")) {
            for p in rd.flatten().map(|e| e.path()) {
                if p.extension().and_then(|e| e.to_str()) == Some("cidx") {
                    std::fs::remove_file(p).unwrap();
                }
            }
        }
        let docs: Vec<(u32, &str, &[u8])> = files.iter().map(|(r, p, c)| (*r, p.as_str(), c.as_slice())).collect();
        build_shardset(dir, &CORPUS_REPOS, &docs)
    }

    /// Files of `dir` by name, with their bytes.
    fn files_of(dir: &Path) -> Vec<(String, Vec<u8>)> {
        let mut out: Vec<(String, Vec<u8>)> = match std::fs::read_dir(dir) {
            Ok(rd) => rd
                .flatten()
                .filter(|e| e.path().is_file())
                .map(|e| (e.file_name().to_string_lossy().into_owned(), std::fs::read(e.path()).unwrap()))
                .collect(),
            Err(_) => Vec::new(),
        };
        out.sort();
        out
    }

    /// The semantic plane on disk after a step: the bytes of every segment
    /// in segment order (a segment's name carries the time it was written,
    /// so segments are compared by position), the tombstones beside each,
    /// and the embedding cache.
    struct Snapshot {
        step: String,
        /// What the step returned: the reports' counts, or whether a
        /// compaction merged anything.
        outcome: String,
        vec: Vec<Vec<u8>>,
        bm25: Vec<Vec<u8>>,
        tombs: Vec<Vec<u32>>,
        cas: Vec<(String, Vec<u8>)>,
    }

    fn snapshot(data_dir: &Path, model: &str, step: &str, outcome: String) -> Snapshot {
        // the next segment gets a later name
        std::thread::sleep(Duration::from_millis(3));
        let vec_paths = crate::index::segment_paths(&vec_dir(data_dir), model);
        let bm25_paths = crate::bm25::segment_paths(&bm25_dir(data_dir), model);
        Snapshot {
            step: step.to_string(),
            outcome,
            vec: vec_paths.iter().map(|p| std::fs::read(p).unwrap()).collect(),
            bm25: bm25_paths.iter().map(|p| std::fs::read(p).unwrap()).collect(),
            tombs: vec_paths.iter().chain(&bm25_paths).map(|p| crate::index::read_tomb_side(p)).collect(),
            cas: files_of(&data_dir.join("embcas").join(model)),
        }
    }

    fn counts(reports: &[EmbedReport]) -> String {
        reports
            .iter()
            .map(|r| format!("{}: chunks {} hits {} misses {} known {} embedded {} carried {}", r.repo, r.chunks, r.cas_hits, r.cas_misses, r.cas_known, r.embedded, r.carried))
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Where two lists of files first differ.
    fn first_difference(old: &[Vec<u8>], new: &[Vec<u8>]) -> String {
        if old.len() != new.len() {
            return format!("{} segments before, {} now", old.len(), new.len());
        }
        for (i, (a, b)) in old.iter().zip(new).enumerate() {
            if a != b {
                let at = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
                return format!("segment {i} of {}: {} bytes before, {} now, first difference at byte {at}", old.len(), a.len(), b.len());
            }
        }
        "no difference".to_string()
    }

    fn assert_same_planes(old: &[Snapshot], new: &[Snapshot]) {
        assert_eq!(old.len(), new.len());
        for (a, b) in old.iter().zip(new) {
            let step = &a.step;
            assert_eq!(a.step, b.step);
            assert_eq!(a.outcome, b.outcome, "{step}: the pass reports something else");
            assert!(a.bm25 == b.bm25, "{step}: bm25 segments differ: {}", first_difference(&a.bm25, &b.bm25));
            assert!(a.vec == b.vec, "{step}: vector segments differ: {}", first_difference(&a.vec, &b.vec));
            assert_eq!(a.tombs, b.tombs, "{step}: tombstones differ");
            let names = |s: &Snapshot| s.cas.iter().map(|f| f.0.clone()).collect::<Vec<_>>();
            assert_eq!(names(a), names(b), "{step}: the embedding cache holds other files");
            for (fa, fb) in a.cas.iter().zip(&b.cas) {
                assert!(fa.1 == fb.1, "{step}: {} of the embedding cache differs", fa.0);
            }
        }
    }

    /// The passes of one side: the ones kept in `reference`, or the
    /// streaming ones.
    struct Passes<'a> {
        old: bool,
        dir: &'a Path,
        embedder: &'a dyn Embedder,
    }

    impl Passes<'_> {
        fn rebuild(&self, shards: &ShardSet, repos: &[String]) -> anyhow::Result<Vec<EmbedReport>> {
            if self.old {
                reference::rebuild_all(shards, self.dir, repos, self.embedder, MAX_CHARS)
            } else {
                rebuild_all(shards, self.dir, repos, self.embedder, MAX_CHARS)
            }
        }

        fn incremental(
            &self,
            shards: &ShardSet,
            repos: &[String],
            paths: Option<&HashSet<String>>,
            given: Option<(&VecSet, &Bm25Set)>,
        ) -> anyhow::Result<Vec<EmbedReport>> {
            if self.old {
                reference::embed_incremental_inner(shards, self.dir, repos, paths, given, self.embedder, MAX_CHARS)
            } else {
                embed_incremental_inner(shards, self.dir, repos, paths, given, self.embedder, MAX_CHARS)
            }
        }

        fn compact(&self, force: bool) -> bool {
            let model = self.embedder.model_id();
            if self.old {
                reference::compact_vectors_opts(self.dir, model, force).unwrap()
            } else {
                compact_vectors_opts(self.dir, model, force).unwrap()
            }
        }
    }

    /// A rebuild, incremental embeds of every kind, a fold of the small
    /// segments, a full merge and a last rebuild: the plane after each.
    fn scenario(dir: &Path, embedder: &dyn Embedder, old: bool, files_per_repo: usize) -> Vec<Snapshot> {
        let model = embedder.model_id().to_string();
        let pass = Passes { old, dir, embedder };
        let repos: Vec<String> = CORPUS_REPOS.iter().map(|r| r.to_string()).collect();
        let mut out: Vec<Snapshot> = Vec::new();
        let mut done = |step: &str, outcome: String| out.push(snapshot(dir, &model, step, outcome));

        // the first embed: every repo, nothing in the cache
        let mut shards = reshard(dir, &corpus(files_per_repo, 0));
        done("rebuild", counts(&pass.rebuild(&shards, &repos).unwrap()));
        // again, for three of the repos: every chunk is in the cache or known to it
        done("rebuild of three repos", counts(&pass.rebuild(&shards, &repos[..3]).unwrap()));

        // files edited, gone and new, in every repo (the fourth has no row left)
        drop(shards);
        shards = reshard(dir, &corpus(files_per_repo, 1));
        done("incremental, every repo", counts(&pass.incremental(&shards, &repos, None, None).unwrap()));
        // a few files of one repo: an edited one, the emptied one, one that never existed
        drop(shards);
        shards = reshard(dir, &corpus(files_per_repo, 2));
        let paths: HashSet<String> = ["src/m1/f1.rs", "docs/notes.md", "src/m3/f8.rs", "src/new/r2.rs", "no/such/file.rs"].into_iter().map(String::from).collect();
        done("incremental, some files", counts(&pass.incremental(&shards, &repos[1..2], Some(&paths), None).unwrap()));
        // one repo at a time, until there are more segments than a plane keeps
        for round in 3..=9u32 {
            drop(shards);
            shards = reshard(dir, &corpus(files_per_repo, round));
            let repo = &repos[round as usize % repos.len()];
            done(&format!("incremental, round {round}"), counts(&pass.incremental(&shards, std::slice::from_ref(repo), None, None).unwrap()));
        }
        done("fold", format!("merged: {}", pass.compact(false)));

        // an embed handed the sets its caller has open
        drop(shards);
        shards = reshard(dir, &corpus(files_per_repo, 10));
        {
            let (v, b) = open_sets(dir, &model).unwrap().unwrap();
            done("incremental, the caller's sets", counts(&pass.incremental(&shards, &repos[..2], None, Some((&v, &b))).unwrap()));
        }
        done("incremental, nothing to do", counts(&pass.incremental(&shards, &repos[..2], None, None).unwrap()));
        // rows tombstoned on the BM25 side alone are carried without their terms
        let first_bm25 = plane_paths(dir, &model).bm25[0].clone();
        crate::index::write_tomb_side(&first_bm25, &[1u32, 5].into_iter().collect()).unwrap();
        done("full merge", format!("merged: {}", pass.compact(true)));
        done("compaction, nothing to do", format!("merged: {}", pass.compact(false)));

        // and everything again from the shards
        done("last rebuild", counts(&pass.rebuild(&shards, &repos).unwrap()));
        out
    }

    /// Run [`scenario`] through the old passes and the new, each with its
    /// own embedder and data dir, and hold the planes against each other.
    /// Returns the two embedders and the snapshots of the new side.
    fn old_and_new<E: Embedder>(make: impl Fn(&Path) -> E, files_per_repo: usize) -> (E, E, Vec<Snapshot>) {
        let (old_dir, new_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let (old_e, new_e) = (make(old_dir.path()), make(new_dir.path()));
        let old = scenario(old_dir.path(), &old_e, true, files_per_repo);
        let new = scenario(new_dir.path(), &new_e, false, files_per_repo);
        assert_same_planes(&old, &new);
        // the scenario went where it was meant to go
        let segments = |step: &str| new.iter().find(|s| s.step == step).unwrap().vec.len();
        assert_eq!(segments("rebuild"), 1);
        assert_eq!(segments("incremental, round 9"), 10);
        let folded = segments("fold");
        assert!((2..=3).contains(&folded), "{folded} segments: the small ones were to be folded, the large carried");
        assert_eq!(segments("incremental, nothing to do"), folded + 1);
        assert_eq!(segments("full merge"), 1);
        assert_eq!(segments("last rebuild"), 1);
        assert!(new.iter().any(|s| s.tombs.iter().any(|t| !t.is_empty())), "no step tombstoned a row");
        // nothing but segments, their tombstones and the stamps is left
        for dir in [vec_dir(new_dir.path()), bm25_dir(new_dir.path())] {
            for (name, _) in files_of(&dir) {
                assert!(!name.ends_with(".tmp"), "{name} was left behind");
            }
        }
        (old_e, new_e, new)
    }

    /// Rows of the first segment of a snapshot that repeat an earlier row's
    /// chunk (the same text twice in one repo), and rows in all.
    fn repeated_rows(dir: &Path, bytes: &[u8]) -> (usize, usize) {
        let p = dir.join("look.civec");
        std::fs::write(&p, bytes).unwrap();
        let seg = VecIndex::open_segment(&p).unwrap();
        let mut seen = HashSet::new();
        let repeats = (0..seg.raw_len() as u32).filter(|&r| !seen.insert((seg.row_meta(r).repo.to_string(), seg.row_hash(r)))).count();
        (repeats, seg.raw_len())
    }

    /// A stateless embedder whose cache keeps hashes only.
    #[test]
    fn streaming_passes_write_the_same_plane_hash_embedder() {
        let (_, _, new) = old_and_new(|_| HashEmbedder::new(48), 40);
        let tmp = tempfile::tempdir().unwrap();
        let (repeats, rows) = repeated_rows(tmp.path(), &new[0].vec[0]);
        assert!(repeats > 0, "no chunk text occurs twice in the corpus");
        // more than a round of misses per repo, and more than a batch of docs
        assert!(rows > 4 * EMBED_ROUND && 40 > 2 * CHUNK_BATCH_DOCS, "{rows} rows");
    }

    /// A stateful embedder: the passes observe the same texts in the same
    /// calls and embed the same batches under the same model state.
    #[test]
    fn streaming_passes_write_the_same_plane_stateful_embedder() {
        let (old_e, new_e, _) = old_and_new(|_| Probe::new("probe", true), 40);
        let (old_calls, new_calls) = (old_e.calls(), new_e.calls());
        assert!(old_calls.0 == new_calls.0, "observe was called differently");
        assert!(old_calls.1 == new_calls.1, "other batches were embedded, or under another model state");
        assert!(old_calls.0.len() > 8 && old_calls.1.len() > 40);
        // the transcripts are embedded, never observed
        assert!(!old_calls.0.iter().any(|c| matches!(c, Call::Observe(t) if t.iter().any(|t| t.contains("r2_f")))));
    }

    /// An embedder whose vectors are kept: chunks come back from the cache.
    #[test]
    fn streaming_passes_write_the_same_plane_cached_vectors() {
        let (old_e, new_e, new) = old_and_new(|_| Probe::new("probe-kept", false), 40);
        assert_eq!(old_e.calls(), new_e.calls());
        assert!(new[1].outcome.contains("misses 0"), "the second rebuild missed the cache: {}", new[1].outcome);
    }

    /// The model the index runs on.
    #[test]
    fn streaming_passes_write_the_same_plane_random_indexing() {
        old_and_new(|dir| crate::rindex::RandomIndexingEmbedder::open(dir).unwrap(), 12);
    }

    /// A pass whose embedder fails writes no segment and leaves no scratch
    /// file; the cache holds the vectors of the batches before the failed
    /// one, as it did.
    #[test]
    fn a_failed_embed_leaves_what_it_left_before() {
        for cheap in [true, false] {
            let run = |old: bool| {
                let tmp = tempfile::tempdir().unwrap();
                let dir = tmp.path();
                let files = corpus(40, 0);
                let shards = reshard(dir, &files);
                let repos: Vec<String> = CORPUS_REPOS.iter().map(|r| r.to_string()).collect();
                let mut e = Probe::new("probe-fails", cheap);
                let pass = Passes { old, dir, embedder: &e };
                let mut snaps = vec![snapshot(dir, e.id, "rebuild", counts(&pass.rebuild(&shards, &repos[..1]).unwrap()))];
                // a chunk in the middle of the second repo
                e.fail_on = Some("r1_f21_g0");
                let pass = Passes { old, dir, embedder: &e };
                let err = pass.rebuild(&shards, &repos).unwrap_err();
                snaps.push(snapshot(dir, e.id, "failed rebuild", format!("{err:#}")));
                let err = pass.incremental(&shards, &repos[1..3], None, None).unwrap_err();
                snaps.push(snapshot(dir, e.id, "failed incremental", format!("{err:#}")));
                for d in [vec_dir(dir), bm25_dir(dir)] {
                    assert!(!files_of(&d).iter().any(|(name, _)| name.ends_with(".tmp")), "a scratch file was left");
                }
                snaps
            };
            let (old, new) = (run(true), run(false));
            assert_same_planes(&old, &new);
            assert!(new[1].outcome.contains("the probe refuses"), "{}", new[1].outcome);
            assert_eq!(new[2].vec.len(), 1, "a failed pass wrote a segment");
        }
    }
}
