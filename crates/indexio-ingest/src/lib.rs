//! indexio-ingest: git crawl (gix), global CAS, and delta re-indexing.
//!
//! Contract: docs/SPEC.md, section "indexio-ingest".
//!
//! Data dir layout:
//!   <data_dir>/shards/           immutable shard files (indexio-index)
//!   <data_dir>/cas/              global content-addressed store
//!   <data_dir>/repos/<name>.json RepoState (written atomically)
//!
//! Indexing is commit-deterministic: file contents are read from the HEAD
//! commit's tree via gix, never from the working directory.
//!
//! Delta semantics (`reindex_repo`): the trees of `last_commit` and HEAD are
//! diffed by path→oid maps. Added/modified files go through the extraction
//! pipeline (CAS first) into a new delta shard; docs for deleted files and
//! superseded old versions are tombstoned in the shards that contain them.
//!
//! Full re-index fallback: if `last_commit` is missing from the object
//! database (e.g. history was rewritten / force-pushed and the old commit
//! was GC'd) or is not an ancestor of HEAD (non-fast-forward), the delta is
//! meaningless, so `reindex_repo` re-indexes the whole HEAD tree and
//! tombstones all previously visible docs of the repo. The report reflects
//! this: `docs_added` = current doc count, `docs_deleted` = previous doc
//! count, `docs_unchanged` = 0.

#![deny(unsafe_code)]

mod cas;
pub use cas::Cas;

pub mod org_sync;
pub mod sources;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use indexio_core::grams::{self, CommonGrams, MAX_DOC_BYTES};
use indexio_index::{ShardSet, ShardWriter};
use indexio_types::{BlobId, DocMeta, ExtractedArtifact, Lang};

/// Registered-repo bookkeeping, JSON at `<data_dir>/repos/<name>.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RepoState {
    pub name: String,
    pub path: PathBuf,
    /// Hex of the commit whose tree was last indexed (None only for states
    /// written before any successful index — treated as "full re-index").
    pub last_commit: Option<String>,
    /// ISO-8601 UTC timestamp of the last successful index run.
    pub indexed_at: String,
    /// True for a plain directory (no git): indexed from the filesystem,
    /// delta by content hash instead of by commit (SPEC-P7).
    #[serde(default)]
    pub plain: bool,
    /// True when the visible docs were last taken from the WORKING TREE
    /// (SPEC-P9 `reindex_worktree`) and may differ from `last_commit`'s
    /// tree; the next HEAD-based `reindex_repo` then reconciles by content
    /// hash instead of by tree diff.
    #[serde(default)]
    pub worktree: bool,
}

#[derive(Clone, Debug, Default)]
pub struct IndexReport {
    pub repo: String,
    pub commit: String,
    pub docs_added: u64,
    pub docs_deleted: u64,
    pub docs_unchanged: u64,
    pub cas_hits: u64,
    pub cas_misses: u64,
    pub bytes_indexed: u64,
    pub elapsed_ms: u64,
    /// Repo-relative paths whose docs were added, replaced or deleted by
    /// this run (SPEC-P9): what the semantic plane has to re-embed.
    pub changed_paths: Vec<String>,
}

/// Cap for [`Lang::Text`] files (SPEC-P9): a data dump or generated
/// document larger than this is noise, not something a session looks up.
pub const MAX_TEXT_BYTES: usize = 256 * 1024;

/// Max commits walked when checking that last_commit is an ancestor of HEAD.
const ANCESTRY_WALK_LIMIT: usize = 100_000;

// ---------------------------------------------------------------------------
// gix helpers
// ---------------------------------------------------------------------------

/// One blob entry of a git tree (recursively walked).
struct TreeFile {
    path: String,
    oid: gix::hash::ObjectId,
}

/// Recursively collect all blob entries of a tree, sorted by path for
/// deterministic shard doc order. Path = tree-relative, '/'-separated.
fn tree_files(repo: &gix::Repository, tree_id: gix::hash::ObjectId) -> anyhow::Result<Vec<TreeFile>> {
    let tree = repo.find_tree(tree_id)?;
    let records = tree.traverse().breadthfirst.files()?;
    let mut out = Vec::with_capacity(records.len());
    for r in records {
        // Regular + executable files only: skips subdirs (if any were
        // recorded), symlinks and submodule gitlinks.
        if !r.mode.is_blob() {
            continue;
        }
        out.push(TreeFile {
            path: String::from_utf8_lossy(&r.filepath).into_owned(),
            oid: r.oid,
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

/// Read a blob's full contents from the object store.
fn read_blob(repo: &gix::Repository, oid: gix::hash::ObjectId) -> anyhow::Result<Vec<u8>> {
    let obj = repo.find_object(oid)?;
    let mut blob = obj
        .try_into_blob()
        .map_err(|e| anyhow!("object {oid} is not a blob: {e}"))?;
    Ok(std::mem::take(&mut blob.data))
}

/// (commit hex, tree id) of HEAD.
fn head(repo: &gix::Repository) -> anyhow::Result<(String, gix::hash::ObjectId)> {
    let commit_id = repo.head_id().context("repo has no HEAD commit")?;
    let tree_id = repo.head_tree_id().context("repo has no HEAD tree")?;
    Ok((commit_id.detach().to_string(), tree_id.detach()))
}

/// True if `old` is HEAD's ancestor (or HEAD itself), i.e. the history is
/// fast-forward from the indexed commit. False = history rewrite.
fn is_ancestor(repo: &gix::Repository, old: gix::hash::ObjectId, head_hex: &str) -> bool {
    let Ok(head_oid) = gix::hash::ObjectId::from_hex(head_hex.as_bytes()) else {
        return false;
    };
    if head_oid == old {
        return true;
    }
    let Ok(commit) = repo.find_commit(head_oid) else {
        return false;
    };
    let Ok(walk) = commit.ancestors().all() else {
        return false;
    };
    walk.take(ANCESTRY_WALK_LIMIT)
        .filter_map(|info| info.ok())
        .any(|info| info.id == old)
}

// ---------------------------------------------------------------------------
// extraction pipeline
// ---------------------------------------------------------------------------

/// A file that survived all skip rules, with its extracted artifact.
struct PreparedDoc {
    meta: DocMeta,
    content: Vec<u8>,
    art: ExtractedArtifact,
    cas_hit: bool,
}

fn is_binary(content: &[u8]) -> bool {
    let prefix = &content[..content.len().min(8192)];
    memchr::memchr(0, prefix).is_some()
}

/// Extract (CAS-first) all candidate files in parallel (rayon, 2 threads).
/// Skip rules (SPEC): unknown lang, binary (NUL in first 8KiB), >4MiB.
fn extract_docs(files: Vec<(String, Vec<u8>)>, cas: &Cas) -> Vec<PreparedDoc> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("rayon thread pool");
    pool.install(|| {
        files
            .into_par_iter()
            .filter_map(|(path, content)| {
                let lang = Lang::from_path(&path);
                if lang == Lang::Unknown {
                    debug!(%path, "skip: unknown language");
                    return None;
                }
                if content.len() > MAX_DOC_BYTES {
                    debug!(%path, "skip: >4MiB");
                    return None;
                }
                if lang == Lang::Text && content.len() > MAX_TEXT_BYTES {
                    debug!(%path, "skip: text file over the cap");
                    return None;
                }
                if is_binary(&content) {
                    debug!(%path, "skip: binary");
                    return None;
                }
                let blob = BlobId::from_content(&content);
                let raw_len = content.len() as u32;
                let (art, cas_hit) = match cas.get(&blob) {
                    Some(art) => (art, true),
                    None => {
                        let ngrams = grams::extract(&content, &CommonGrams::empty());
                        let (symbols, calls) = if indexio_symbols::supported(lang) {
                            indexio_symbols::extract(lang, &content)
                        } else {
                            (Vec::new(), Vec::new())
                        };
                        let art = ExtractedArtifact {
                            ngrams,
                            symbols,
                            calls,
                            raw_len,
                            lang,
                        };
                        // A failed put only costs a future CAS miss.
                        if let Err(e) = cas.put(&blob, &art) {
                            warn!(%path, error = %e, "CAS put failed");
                        }
                        (art, false)
                    }
                };
                Some(PreparedDoc {
                    meta: DocMeta {
                        blob,
                        repo_id: 0,
                        path,
                        lang,
                        raw_len,
                    },
                    content,
                    art,
                    cas_hit,
                })
            })
            .collect()
    })
}

/// Write `docs` as one shard (repo_id = 0, repos = [name]).
/// No shard is written when there is nothing to add (delta with zero
/// indexable changes).
fn write_shard(shards_dir: &Path, name: &str, docs: &[PreparedDoc]) -> anyhow::Result<()> {
    if docs.is_empty() {
        return Ok(());
    }
    let mut writer = ShardWriter::new(shards_dir)?;
    for d in docs {
        writer.add_doc(&d.meta, &d.content, &d.art)?;
    }
    let path = writer.finish(&[name.to_string()])?;
    debug!(?path, docs = docs.len(), "shard written");
    Ok(())
}

/// Read blob contents for the given tree files (sequential object-store
/// reads; the CPU-heavy extraction afterwards is parallel).
fn read_all(repo: &gix::Repository, files: Vec<TreeFile>) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
    let mut out = Vec::with_capacity(files.len());
    for f in files {
        let content = read_blob(repo, f.oid).with_context(|| format!("reading blob {}", f.path))?;
        out.push((f.path, content));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// RepoState persistence
// ---------------------------------------------------------------------------

fn repos_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("repos")
}

fn state_path(data_dir: &Path, name: &str) -> PathBuf {
    repos_dir(data_dir).join(format!("{name}.json"))
}

fn load_state(data_dir: &Path, name: &str) -> io::Result<RepoState> {
    let bytes = fs::read(state_path(data_dir, name))?;
    serde_json::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Read a registered repo's state (SPEC-P6: surfaces resolve a repo name
/// to its working-tree path for `impact --diff`).
pub fn repo_state(data_dir: &Path, name: &str) -> io::Result<RepoState> {
    load_state(data_dir, name)
}

/// Atomic write: tmp file in the same directory + rename.
fn save_state(data_dir: &Path, state: &RepoState) -> io::Result<()> {
    let dir = repos_dir(data_dir);
    fs::create_dir_all(&dir)?;
    let path = state_path(data_dir, &state.name);
    let tmp = dir.join(format!(".{}.json.tmp", state.name));
    let bytes = serde_json::to_vec_pretty(state)?;
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, &path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// time (no chrono/time dep: minimal unix -> ISO-8601 UTC)
// ---------------------------------------------------------------------------

fn unix_to_iso(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let mut y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    if mo <= 2 {
        y += 1;
    }
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// `YYYY-MM-DD` (UTC) of a unix timestamp.
pub fn unix_to_iso_date(secs: u64) -> String {
    unix_to_iso(secs).chars().take(10).collect()
}

fn iso_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    unix_to_iso(secs)
}

// ---------------------------------------------------------------------------
// public API (SPEC indexio-ingest)
// ---------------------------------------------------------------------------

/// Full index of a local git repo at HEAD (commit-deterministic: blobs are
/// read from the HEAD tree, not the working directory).
///
/// Errors if `<data_dir>/repos/<name>.json` already exists — use
/// `reindex_repo` for registered repos.
pub fn index_repo(
    repo_path: &Path,
    name: &str,
    data_dir: &Path,
    cas: &Cas,
) -> anyhow::Result<IndexReport> {
    let t0 = Instant::now();
    if state_path(data_dir, name).exists() {
        bail!("repo '{name}' is already registered; use reindex_repo");
    }
    let shards_dir = data_dir.join("shards");
    fs::create_dir_all(&shards_dir)?;
    fs::create_dir_all(repos_dir(data_dir))?;

    let repo = match gix::open(repo_path) {
        Ok(r) => r,
        // Not a git repository: index the directory tree as-is (SPEC-P7).
        Err(_) if repo_path.is_dir() && !repo_path.join(".git").exists() => {
            return index_dir(repo_path, name, data_dir, cas)
        }
        Err(e) => return Err(e).with_context(|| format!("opening {}", repo_path.display())),
    };
    let (commit, tree_id) = head(&repo)?;
    let files = tree_files(&repo, tree_id)?;
    let contents = read_all(&repo, files)?;
    let docs = extract_docs(contents, cas);

    let cas_hits = docs.iter().filter(|d| d.cas_hit).count() as u64;
    let docs_added = docs.len() as u64;
    let bytes_indexed = docs.iter().map(|d| d.content.len() as u64).sum();
    write_shard(&shards_dir, name, &docs)?;
    let report = IndexReport {
        repo: name.to_string(),
        commit,
        docs_added,
        docs_deleted: 0,
        docs_unchanged: 0,
        cas_hits,
        cas_misses: docs_added - cas_hits,
        bytes_indexed,
        elapsed_ms: t0.elapsed().as_millis() as u64,
        changed_paths: docs.iter().map(|d| d.meta.path.clone()).collect(),
    };
    save_state(
        data_dir,
        &RepoState {
            name: name.to_string(),
            path: repo_path.to_path_buf(),
            last_commit: Some(report.commit.clone()),
            indexed_at: iso_now(),
            plain: false,
            worktree: false,
        },
    )?;
    Ok(report)
}

// ---------------------------------------------------------------------------
// Plain directories (no git) — SPEC-P7
// ---------------------------------------------------------------------------

/// Directory names never descended into when indexing a plain tree (build
/// output, dependency caches, VCS internals). Hidden directories are skipped
/// too.
pub const SKIP_DIRS: &[&str] = &[
    "node_modules", "target", "dist", "build", "out", "vendor", "__pycache__",
    "venv", ".venv", "env", "site-packages", "bower_components", "coverage",
    "obj", "bin", "Pods", "DerivedData",
];

/// Whether a directory entry should be skipped by the plain-tree walker.
pub fn skip_dir_name(name: &str) -> bool {
    name.starts_with('.') || SKIP_DIRS.contains(&name)
}

/// Whether `dir` is a cache/build-output directory by its own declaration
/// (SPEC-P10 §24): cargo stamps every target directory — `target/`,
/// `--target-dir target-bench-B`, `CARGO_TARGET_DIR=…` — with a
/// `CACHEDIR.TAG` file, as do pip, pytest, Gradle and other cache writers.
/// [`SKIP_DIRS`] only knows the conventional names: one checkout had
/// 1,034 cargo dep-info `.d` files from `target-bench-B/` indexed as D
/// source, 55 % of the repo's docs.
pub fn is_cache_dir(dir: &Path) -> bool {
    dir.join("CACHEDIR.TAG").is_file()
}

/// Whether the file at `rel` (a '/'-separated path under `root`) lies
/// under a directory that [`is_cache_dir`]; `memo` caches the answer per
/// directory across one pass so a tree costs one stat per directory.
pub fn under_cache_dir(root: &Path, rel: &str, memo: &mut HashMap<String, bool>) -> bool {
    let mut prefix = String::new();
    let mut segs = rel.split('/').peekable();
    while let Some(seg) = segs.next() {
        if segs.peek().is_none() {
            break; // the file itself
        }
        if !prefix.is_empty() {
            prefix.push('/');
        }
        prefix.push_str(seg);
        let cached = match memo.get(&prefix) {
            Some(&b) => b,
            None => {
                let b = is_cache_dir(&root.join(&prefix));
                memo.insert(prefix.clone(), b);
                b
            }
        };
        if cached {
            return true;
        }
    }
    false
}

/// Every regular file under `root` (relative '/'-separated paths, sorted),
/// skipping [`SKIP_DIRS`] and hidden directories; symlinks are not followed.
pub fn walk_plain_tree(root: &Path) -> anyhow::Result<Vec<String>> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) -> anyhow::Result<()> {
        for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
            let entry = entry?;
            let ft = entry.file_type()?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if ft.is_dir() {
                if !skip_dir_name(&name) && !is_cache_dir(&entry.path()) {
                    walk(root, &entry.path(), out)?;
                }
            } else if ft.is_file() {
                let rel = entry
                    .path()
                    .strip_prefix(root)
                    .map(|p| p.to_string_lossy().replace('\\', "/"))
                    .unwrap_or(name);
                out.push(rel);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(root, root, &mut out)?;
    out.sort();
    Ok(out)
}

/// Read the indexable files of a plain tree: (relative path, content) for
/// files with a known language and <= MAX_DOC_BYTES (binary detection
/// happens in `extract_docs`).
fn read_plain_tree(root: &Path) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
    let mut out = Vec::new();
    for rel in walk_plain_tree(root)? {
        if Lang::from_path(&rel) == Lang::Unknown {
            continue;
        }
        let path = root.join(&rel);
        let Ok(meta) = fs::metadata(&path) else { continue };
        if meta.len() as usize > MAX_DOC_BYTES {
            continue;
        }
        match fs::read(&path) {
            Ok(content) => out.push((rel, content)),
            Err(e) => warn!(path = %path.display(), error = %e, "skip: unreadable"),
        }
    }
    Ok(out)
}

/// The indexable files of a plain tree as [`WtFile`]s, with a
/// [`WorktreeCache`] so a file whose mtime and length are unchanged since
/// the last pass is only stat'ed (SPEC-P10: the `sessions` folder and
/// plain-folder repos were re-read and re-hashed whole on every refresh).
fn read_plain_tree_cached(root: &Path, mut cache: Option<&mut WorktreeCache>) -> anyhow::Result<Vec<WtFile>> {
    let mut files = Vec::new();
    for rel in walk_plain_tree(root)? {
        if Lang::from_path(&rel) == Lang::Unknown {
            continue;
        }
        let path = root.join(&rel);
        let Ok(meta) = fs::metadata(&path) else { continue };
        if !meta.is_file() || meta.len() as usize > MAX_DOC_BYTES {
            continue;
        }
        let stamp = (meta.modified().ok(), meta.len());
        if let Some(c) = cache.as_deref() {
            if let Some((m, l, blob)) = c.files.get(&rel) {
                if (*m, *l) == stamp {
                    files.push(WtFile { rel, blob: *blob, content: None });
                    continue;
                }
            }
        }
        match fs::read(&path) {
            Ok(content) => {
                let blob = match normalize_crlf(&content) {
                    Some(lf) => BlobId::from_content(&lf),
                    None => BlobId::from_content(&content),
                };
                if let Some(c) = cache.as_deref_mut() {
                    c.files.insert(rel.clone(), (stamp.0, stamp.1, blob));
                }
                files.push(WtFile { rel, blob, content: Some(content) });
            }
            Err(e) => warn!(path = %path.display(), error = %e, "skip: unreadable"),
        }
    }
    if let Some(c) = cache.as_deref_mut() {
        c.files.retain(|rel, _| files.iter().any(|f| f.rel == *rel));
    }
    Ok(files)
}

/// Index a plain directory (no git) as repo `name`. Files are read from the
/// filesystem; the state records `plain: true` and no commit.
pub fn index_dir(
    dir: &Path,
    name: &str,
    data_dir: &Path,
    cas: &Cas,
) -> anyhow::Result<IndexReport> {
    let t0 = Instant::now();
    if state_path(data_dir, name).exists() {
        bail!("repo '{name}' is already registered; use reindex_repo");
    }
    anyhow::ensure!(dir.is_dir(), "{} is not a directory", dir.display());
    let shards_dir = data_dir.join("shards");
    fs::create_dir_all(&shards_dir)?;
    fs::create_dir_all(repos_dir(data_dir))?;
    let contents = read_plain_tree(dir)?;
    let docs = extract_docs(contents, cas);
    let cas_hits = docs.iter().filter(|d| d.cas_hit).count() as u64;
    let docs_added = docs.len() as u64;
    let bytes_indexed = docs.iter().map(|d| d.content.len() as u64).sum();
    write_shard(&shards_dir, name, &docs)?;
    save_state(
        data_dir,
        &RepoState {
            name: name.to_string(),
            path: dir.to_path_buf(),
            last_commit: None,
            indexed_at: iso_now(),
            plain: true,
            worktree: false,
        },
    )?;
    Ok(IndexReport {
        repo: name.to_string(),
        commit: String::new(),
        docs_added,
        docs_deleted: 0,
        docs_unchanged: 0,
        cas_hits,
        cas_misses: docs_added - cas_hits,
        bytes_indexed,
        elapsed_ms: t0.elapsed().as_millis() as u64,
        changed_paths: docs.iter().map(|d| d.meta.path.clone()).collect(),
    })
}

/// Delta re-index of a plain directory: files whose content hash equals
/// the indexed doc's blob are unchanged; changed/added files go into a new
/// shard; deleted and superseded docs are tombstoned.
fn reindex_dir(
    state: &RepoState,
    data_dir: &Path,
    cas: &Cas,
    cache: Option<&mut WorktreeCache>,
) -> anyhow::Result<IndexReport> {
    let current = read_plain_tree_cached(&state.path, cache)?;
    let next = RepoState {
        indexed_at: iso_now(),
        ..state.clone()
    };
    delta_by_content(state, next, data_dir, cas, current, String::new())
}

/// One current file for [`delta_by_content`]: either its bytes, or just
/// the blake3 of its (LF-normalised) content when a [`WorktreeCache`] says
/// the file is unchanged on disk since it was last read.
pub struct WtFile {
    rel: String,
    blob: BlobId,
    content: Option<Vec<u8>>,
}

impl WtFile {
    fn from_bytes((rel, content): (String, Vec<u8>)) -> WtFile {
        WtFile {
            rel,
            blob: BlobId::from_content(&content),
            content: Some(content),
        }
    }
}

/// Per-file `(mtime, len, blob)` remembered between working-tree passes of
/// one long-lived process (the MCP server's auto-refresh, SPEC-P9): a file
/// whose mtime and length are unchanged is not re-read or re-hashed.
#[derive(Default)]
pub struct WorktreeCache {
    files: HashMap<String, (Option<SystemTime>, u64, BlobId)>,
}

/// Delta re-index against whatever `current` files say (SPEC-P7 plain
/// folders, SPEC-P9 working trees and post-worktree reconciliation): a file
/// whose blake3 hash equals the visible doc's blob is unchanged; changed
/// and added files go into a new shard; deleted and superseded docs are
/// tombstoned. `next` is the state to save on success.
fn delta_by_content(
    state: &RepoState,
    next: RepoState,
    data_dir: &Path,
    cas: &Cas,
    current: Vec<WtFile>,
    commit: String,
) -> anyhow::Result<IndexReport> {
    let t0 = Instant::now();
    let name = state.name.as_str();
    // One writer per repo at a time (SPEC-P9 §23): a session's auto-refresh,
    // the Bash hook's freshen and the CLI can all delta the same repo
    // within the same second; two concurrent passes would each add the
    // changed doc and leave a duplicate live. A writer still busy when the
    // wait runs out is doing this very work: report a no-op.
    let Some(_lock) = RepoLock::acquire(data_dir, name, REPO_LOCK_WAIT) else {
        debug!(repo = %name, "another process is re-indexing this repo; skipping");
        return Ok(IndexReport {
            repo: name.to_string(),
            commit,
            docs_added: 0,
            docs_deleted: 0,
            docs_unchanged: 0,
            cas_hits: 0,
            cas_misses: 0,
            bytes_indexed: 0,
            elapsed_ms: t0.elapsed().as_millis() as u64,
            changed_paths: Vec::new(),
        });
    };
    let shards_dir = data_dir.join("shards");
    fs::create_dir_all(&shards_dir)?;
    let mut set = ShardSet::open_dir(&shards_dir)?;
    let prev_docs: Vec<(usize, u32, DocMeta)> = set
        .visible_docs()
        .into_iter()
        .filter(|(si, _, dm)| repo_name(&set, *si, dm.repo_id) == name)
        .collect();
    let prev_blob: HashMap<&str, BlobId> =
        prev_docs.iter().map(|(_, _, dm)| (dm.path.as_str(), dm.blob)).collect();

    let mut to_index: Vec<(String, Vec<u8>)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut doomed: HashSet<String> = HashSet::new();
    for f in current {
        let rel = f.rel;
        seen.insert(rel.clone());
        let old = prev_blob.get(rel.as_str()).copied();
        let content = match f.content {
            Some(c) => c,
            None => {
                if old == Some(f.blob) {
                    continue; // unchanged on disk and still the visible doc
                }
                // unchanged on disk but its doc is gone (a HEAD sync from
                // another process tombstoned it): read it after all
                match fs::read(state.path.join(&rel)) {
                    Ok(c) => c,
                    Err(_) => continue,
                }
            }
        };
        if old == Some(BlobId::from_content(&content)) {
            continue; // byte-identical
        }
        // Windows checkouts carry CRLF on disk while git blobs (and the
        // docs indexed from them) are LF: compare and index the
        // LF-normalised bytes so line endings alone never re-index a file.
        let content = match normalize_crlf(&content) {
            Some(lf) => {
                if old == Some(BlobId::from_content(&lf)) {
                    continue;
                }
                lf
            }
            None => content,
        };
        if old.is_some() {
            doomed.insert(rel.clone());
        }
        to_index.push((rel, content));
    }
    for old_path in prev_blob.keys() {
        if seen.contains(*old_path) {
            continue;
        }
        // A doc this binary's walker does not recognise (a file type a
        // newer binary indexed) is not a deleted file: leave it to the
        // binary that knows it, or two versions serving one data dir
        // add and tombstone the same docs in turns.
        if Lang::from_path(old_path) == Lang::Unknown && state.path.join(old_path).is_file() {
            continue;
        }
        doomed.insert((*old_path).to_string());
    }
    let mut changed_paths: Vec<String> = to_index.iter().map(|(p, _)| p.clone()).collect();
    changed_paths.extend(doomed.iter().cloned());
    changed_paths.sort();
    changed_paths.dedup();
    let docs = extract_docs(to_index, cas);
    let cas_hits = docs.iter().filter(|d| d.cas_hit).count() as u64;
    let docs_added = docs.len() as u64;
    let bytes_indexed = docs.iter().map(|d| d.content.len() as u64).sum();
    if !docs.is_empty() {
        // a no-op pass must not leave an empty delta shard behind
        write_shard(&shards_dir, name, &docs)?;
    }
    let docs_deleted = tombstone_matching(&mut set, &prev_docs, &|p| doomed.contains(p))?;
    let docs_unchanged = (prev_docs.len() as u64).saturating_sub(docs_deleted);
    save_state(data_dir, &next)?;
    Ok(IndexReport {
        repo: name.to_string(),
        commit,
        docs_added,
        docs_deleted,
        docs_unchanged,
        cas_hits,
        cas_misses: docs_added - cas_hits,
        bytes_indexed,
        elapsed_ms: t0.elapsed().as_millis() as u64,
        changed_paths,
    })
}

/// Whether a registered git repo's HEAD differs from the commit its index
/// was built from (SPEC-P10 §21): a ref resolve, no tree walk. `false` for
/// plain folders, repos indexed from their working tree (another server's
/// auto-refresh owns those), unreadable repos and unchanged HEADs — the
/// cases a background sync should leave alone.
pub fn head_moved(data_dir: &Path, name: &str) -> bool {
    let Ok(state) = load_state(data_dir, name) else { return false };
    if state.plain || state.worktree {
        return false;
    }
    let Ok(repo) = gix::open(&state.path) else { return false };
    match head(&repo) {
        Ok((commit, _)) => state.last_commit.as_deref() != Some(commit.as_str()),
        Err(_) => false,
    }
}

/// Cross-process lock for the background sync of every registered repo
/// (SPEC-P10 §21): `<data_dir>/sync.lock`, created exclusively; a lock older
/// than [`SYNC_LOCK_STALE_SECS`] is assumed abandoned. Dropping releases it.
pub struct SyncLock {
    path: PathBuf,
}

pub const SYNC_LOCK_STALE_SECS: u64 = 30 * 60;

impl SyncLock {
    /// `None` when another live process is syncing.
    pub fn try_acquire(data_dir: &Path) -> Option<SyncLock> {
        let path = data_dir.join("sync.lock");
        for _ in 0..2 {
            match fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut f) => {
                    use std::io::Write as _;
                    let _ = write!(f, "{}", std::process::id());
                    return Some(SyncLock { path });
                }
                Err(_) => {
                    let stale = fs::metadata(&path)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .map_or(true, |age| age.as_secs() > SYNC_LOCK_STALE_SECS);
                    if !stale {
                        return None;
                    }
                    warn!(path = %path.display(), "taking over a stale sync lock");
                    let _ = fs::remove_file(&path);
                }
            }
        }
        None
    }
}

impl Drop for SyncLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// How long a delta pass waits for another writer of the same repo.
const REPO_LOCK_WAIT: Duration = Duration::from_secs(5);
/// Age past which a repo lock is assumed abandoned (a killed process).
const REPO_LOCK_STALE_SECS: u64 = 120;

/// Cross-process per-repo write lock: `<data_dir>/repos/<name>.lock`,
/// created exclusively; dropping releases it.
pub struct RepoLock {
    path: PathBuf,
}

impl RepoLock {
    /// Wait up to `wait` for the lock; `None` if it is still held.
    pub fn acquire(data_dir: &Path, repo: &str, wait: Duration) -> Option<RepoLock> {
        let dir = data_dir.join("repos");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join(format!("{repo}.lock"));
        let deadline = Instant::now() + wait;
        loop {
            match fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut f) => {
                    use std::io::Write as _;
                    let _ = write!(f, "{}", std::process::id());
                    return Some(RepoLock { path });
                }
                Err(_) => {
                    let stale = fs::metadata(&path)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .map_or(true, |age| age.as_secs() > REPO_LOCK_STALE_SECS);
                    if stale {
                        warn!(path = %path.display(), "taking over a stale repo lock");
                        let _ = fs::remove_file(&path);
                        continue;
                    }
                    if Instant::now() >= deadline {
                        return None;
                    }
                    std::thread::sleep(Duration::from_millis(25));
                }
            }
        }
    }
}

impl Drop for RepoLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// `content` with every CRLF turned into LF; `None` when it has no CR
/// (the common case, no copy made).
fn normalize_crlf(content: &[u8]) -> Option<Vec<u8>> {
    if !content.contains(&CR) {
        return None;
    }
    let mut out = Vec::with_capacity(content.len());
    let mut i = 0;
    while i < content.len() {
        if content[i] == CR && content.get(i + 1) == Some(&LF) {
            i += 1;
            continue;
        }
        out.push(content[i]);
        i += 1;
    }
    Some(out)
}

const CR: u8 = 13;
const LF: u8 = 10;

/// The files of a git WORKING TREE: tracked plus untracked-but-not-ignored
/// (`git ls-files --cached --others --exclude-standard`), filtered like a
/// plain tree (known language, <= MAX_DOC_BYTES, readable). Files staged
/// for deletion or removed from disk are skipped, i.e. treated as deleted.
fn read_worktree(root: &Path, mut cache: Option<&mut WorktreeCache>) -> anyhow::Result<Vec<WtFile>> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z", "--cached", "--others", "--exclude-standard"])
        .output()
        .with_context(|| format!("running git ls-files in {}", root.display()))?;
    if !out.status.success() {
        bail!(
            "git ls-files failed in {}: {}",
            root.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let mut files = Vec::new();
    let mut rels: Vec<String> = out
        .stdout
        .split(|&b| b == 0)
        .filter(|p| !p.is_empty())
        .map(|p| String::from_utf8_lossy(p).into_owned())
        .collect();
    rels.sort();
    rels.dedup();
    let mut cache_dirs: HashMap<String, bool> = HashMap::new();
    for rel in rels {
        if rel.split('/').any(|seg| skip_dir_name(seg) && seg != rel) {
            continue; // build output that slipped past .gitignore
        }
        if under_cache_dir(root, &rel, &mut cache_dirs) {
            continue; // a target dir under another name (CACHEDIR.TAG)
        }
        if Lang::from_path(&rel) == Lang::Unknown {
            continue;
        }
        let path = root.join(&rel);
        let Ok(meta) = fs::metadata(&path) else { continue };
        if !meta.is_file() || meta.len() as usize > MAX_DOC_BYTES {
            continue;
        }
        let stamp = (meta.modified().ok(), meta.len());
        if let Some(c) = cache.as_deref() {
            if let Some((m, l, blob)) = c.files.get(&rel) {
                if (*m, *l) == stamp {
                    files.push(WtFile { rel, blob: *blob, content: None });
                    continue;
                }
            }
        }
        match fs::read(&path) {
            Ok(content) => {
                // the cache remembers the LF-normalised hash, which is what
                // the indexed doc carries
                let blob = match normalize_crlf(&content) {
                    Some(lf) => BlobId::from_content(&lf),
                    None => BlobId::from_content(&content),
                };
                if let Some(c) = cache.as_deref_mut() {
                    c.files.insert(rel.clone(), (stamp.0, stamp.1, blob));
                }
                files.push(WtFile { rel, blob, content: Some(content) });
            }
            Err(e) => warn!(path = %path.display(), error = %e, "skip: unreadable"),
        }
    }
    if let Some(c) = cache.as_deref_mut() {
        c.files.retain(|rel, _| files.iter().any(|f| f.rel == *rel));
    }
    Ok(files)
}

/// Re-index a registered repo from its WORKING TREE (SPEC-P9): what the
/// agent is editing right now, uncommitted changes included, delta by
/// content hash against the visible docs. The saved state keeps
/// `last_commit` and sets `worktree: true` so a later HEAD-based
/// `reindex_repo` reconciles by content instead of trusting the tree diff.
/// Plain folders take their usual path.
pub fn reindex_worktree(name: &str, data_dir: &Path, cas: &Cas) -> anyhow::Result<IndexReport> {
    reindex_worktree_cached(name, data_dir, cas, None)
}

/// [`reindex_worktree`] with a [`WorktreeCache`] carried between calls so
/// unchanged files are only stat'ed, not read.
pub fn reindex_worktree_cached(
    name: &str,
    data_dir: &Path,
    cas: &Cas,
    cache: Option<&mut WorktreeCache>,
) -> anyhow::Result<IndexReport> {
    let state = load_state(data_dir, name)
        .map_err(|e| anyhow!("repo '{name}' is not registered (run index_repo first): {e}"))?;
    if state.plain {
        return reindex_dir(&state, data_dir, cas, cache);
    }
    let current = read_worktree(&state.path, cache)?;
    let next = RepoState {
        indexed_at: iso_now(),
        worktree: true,
        ..state.clone()
    };
    delta_by_content(&state, next, data_dir, cas, current, "worktree".to_string())
}

/// Delta re-index of a registered repo: tree-diff `last_commit..HEAD`.
/// Added/modified files are extracted (CAS first) into a delta shard;
/// deleted files and superseded old doc versions are tombstoned.
///
/// Falls back to a full re-index (all previous docs tombstoned) when
/// `last_commit` is missing or is not an ancestor of HEAD
/// (non-fast-forward / rewritten history).
pub fn reindex_repo(name: &str, data_dir: &Path, cas: &Cas) -> anyhow::Result<IndexReport> {
    let t0 = Instant::now();
    let state = load_state(data_dir, name)
        .map_err(|e| anyhow!("repo '{name}' is not registered (run index_repo first): {e}"))?;
    if state.plain {
        return reindex_dir(&state, data_dir, cas, None);
    }
    let shards_dir = data_dir.join("shards");
    fs::create_dir_all(&shards_dir)?;

    let repo = gix::open(&state.path).with_context(|| format!("opening {}", state.path.display()))?;
    let (commit, tree_id) = head(&repo)?;
    let head_files = tree_files(&repo, tree_id)?;
    if state.worktree {
        // The visible docs came from the working tree: the tree diff since
        // last_commit says nothing about them. Reconcile by content hash.
        debug!(repo = %name, "docs are worktree-derived; reconciling HEAD by content");
        let current = read_all(&repo, head_files)?
            .into_iter()
            .map(WtFile::from_bytes)
            .collect();
        let next = RepoState {
            last_commit: Some(commit.clone()),
            indexed_at: iso_now(),
            worktree: false,
            ..state.clone()
        };
        return delta_by_content(&state, next, data_dir, cas, current, commit);
    }
    let head_map: HashMap<String, gix::hash::ObjectId> =
        head_files.iter().map(|f| (f.path.clone(), f.oid)).collect();

    // Previously visible docs of this repo: (shard idx, docid, meta).
    let mut set = ShardSet::open_dir(&shards_dir)?;
    let prev_docs: Vec<(usize, u32, DocMeta)> = set
        .visible_docs()
        .into_iter()
        .filter(|(si, _, dm)| repo_name(&set, *si, dm.repo_id) == name)
        .collect();

    // Resolve the previously indexed commit. Full re-index if it is gone
    // from the object store or the history is not fast-forward.
    let old_tree = state
        .last_commit
        .as_deref()
        .and_then(|hex| gix::hash::ObjectId::from_hex(hex.as_bytes()).ok())
        .filter(|oid| is_ancestor(&repo, *oid, &commit))
        .and_then(|oid| repo.find_commit(oid).ok())
        .and_then(|c| c.tree_id().ok())
        .map(|id| id.detach());

    // Plan: which files to (re-)index, and which previous docs to tombstone.
    enum Tombstones {
        /// Delta: tombstone previous docs at these paths (deleted files +
        /// superseded versions of modified files).
        Paths(HashSet<String>),
        /// Full re-index: tombstone every previously visible doc.
        All,
    }

    let (docs, tombstones): (Vec<PreparedDoc>, Tombstones) = match old_tree {
        Some(old_tree_id) => {
            let old_files = tree_files(&repo, old_tree_id)?;
            let old_map: HashMap<String, gix::hash::ObjectId> =
                old_files.into_iter().map(|f| (f.path, f.oid)).collect();

            // added + modified (old doc versions are superseded)
            let mut to_index: Vec<TreeFile> = Vec::new();
            let mut doomed_paths: HashSet<String> = HashSet::new();
            for f in head_files {
                match old_map.get(&f.path) {
                    None => to_index.push(f),
                    Some(old_oid) if *old_oid != f.oid => {
                        doomed_paths.insert(f.path.clone());
                        to_index.push(f);
                    }
                    _ => {} // unchanged
                }
            }
            // deleted
            for old_path in old_map.keys() {
                if !head_map.contains_key(old_path) {
                    doomed_paths.insert(old_path.clone());
                }
            }

            let contents = read_all(&repo, to_index)?;
            (extract_docs(contents, cas), Tombstones::Paths(doomed_paths))
        }
        None => {
            debug!(repo = %name, "last_commit missing or non-fast-forward; full re-index");
            let contents = read_all(&repo, head_files)?;
            (extract_docs(contents, cas), Tombstones::All)
        }
    };

    let cas_hits = docs.iter().filter(|d| d.cas_hit).count() as u64;
    let docs_added = docs.len() as u64;
    let bytes_indexed = docs.iter().map(|d| d.content.len() as u64).sum();

    // Write the delta shard first, then tombstone the superseded/deleted old
    // docs (a crash in between leaves the old doc merely shadowed by the
    // newer shard — newest-wins — instead of losing the old doc entirely).
    if !docs.is_empty() {
        write_shard(&shards_dir, name, &docs)?;
    }
    let docs_deleted = match &tombstones {
        Tombstones::Paths(paths) => {
            tombstone_matching(&mut set, &prev_docs, &|p| paths.contains(p))?
        }
        Tombstones::All => tombstone_matching(&mut set, &prev_docs, &|_| true)?,
    };
    let docs_unchanged = (prev_docs.len() as u64).saturating_sub(docs_deleted);
    let mut changed_paths: Vec<String> = docs.iter().map(|d| d.meta.path.clone()).collect();
    match &tombstones {
        Tombstones::Paths(paths) => changed_paths.extend(paths.iter().cloned()),
        Tombstones::All => changed_paths.extend(prev_docs.iter().map(|(_, _, dm)| dm.path.clone())),
    }
    changed_paths.sort();
    changed_paths.dedup();

    let report = IndexReport {
        repo: name.to_string(),
        commit: commit.clone(),
        docs_added,
        docs_deleted,
        docs_unchanged,
        cas_hits,
        cas_misses: docs_added - cas_hits,
        bytes_indexed,
        elapsed_ms: t0.elapsed().as_millis() as u64,
        changed_paths,
    };
    save_state(
        data_dir,
        &RepoState {
            name: name.to_string(),
            path: state.path.clone(),
            last_commit: Some(report.commit.clone()),
            indexed_at: iso_now(),
            plain: false,
            worktree: false,
        },
    )?;
    Ok(report)
}

/// Repo name for a doc (repo_id is shard-local).
fn repo_name(set: &ShardSet, shard_idx: usize, repo_id: u32) -> String {
    set.shard(shard_idx)
        .and_then(|s| s.meta().repos.get(repo_id as usize).cloned())
        .unwrap_or_default()
}

/// Tombstone all docs in `prev_docs` whose path matches `pred`.
/// Returns the number of docs tombstoned.
fn tombstone_matching(
    set: &mut ShardSet,
    prev_docs: &[(usize, u32, DocMeta)],
    pred: &dyn Fn(&str) -> bool,
) -> anyhow::Result<u64> {
    let mut by_shard: HashMap<usize, Vec<u32>> = HashMap::new();
    let mut count = 0u64;
    for (si, docid, dm) in prev_docs {
        if pred(&dm.path) {
            by_shard.entry(*si).or_default().push(*docid);
            count += 1;
        }
    }
    for (si, docids) in by_shard {
        set.delete_docs(si, &docids)?;
    }
    Ok(count)
}

// ---------------------------------------------------------------------------
// tests (fixtures use the git CLI)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn git(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("HOME", dir)
            .output()
            .expect("spawn git");
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// Init a git repo in a fresh tempdir with the given files committed.
    fn make_repo(files: &[(&str, &[u8])]) -> tempfile::TempDir {
        let td = tempfile::tempdir().unwrap();
        git(td.path(), &["init", "-q"]);
        git(td.path(), &["config", "user.email", "t@example.com"]);
        git(td.path(), &["config", "user.name", "t"]);
        git(td.path(), &["config", "commit.gpgsign", "false"]);
        write_files(td.path(), files);
        commit_all(td.path(), "init");
        td
    }

    fn write_files(repo: &Path, files: &[(&str, &[u8])]) {
        for (path, content) in files {
            let p = repo.join(path);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(&p, content).unwrap();
        }
    }

    fn commit_all(repo: &Path, msg: &str) {
        git(repo, &["add", "-A"]);
        git(repo, &["commit", "-q", "-m", msg]);
    }

    fn head_hex(repo: &Path) -> String {
        git(repo, &["rev-parse", "HEAD"])
    }

    fn shard_files(data_dir: &Path) -> Vec<PathBuf> {
        let dir = data_dir.join("shards");
        let mut out: Vec<PathBuf> = fs::read_dir(&dir)
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("cidx"))
                    .collect()
            })
            .unwrap_or_default();
        out.sort();
        out
    }

    fn load_repo_state(data_dir: &Path, name: &str) -> RepoState {
        load_state(data_dir, name).unwrap()
    }

    /// Paths of the currently visible docs of `repo`.
    fn visible_paths(data_dir: &Path, repo: &str) -> Vec<String> {
        let set = ShardSet::open_dir(&data_dir.join("shards")).unwrap();
        let mut out: Vec<String> = set
            .visible_docs()
            .into_iter()
            .filter(|(si, _, dm)| repo_name(&set, *si, dm.repo_id) == repo)
            .map(|(_, _, dm)| dm.path)
            .collect();
        out.sort();
        out
    }

    /// Is `needle` present in any visible doc of `repo`?
    fn content_contains(data_dir: &Path, repo: &str, needle: &[u8]) -> bool {
        let set = ShardSet::open_dir(&data_dir.join("shards")).unwrap();
        for (si, docid, dm) in set.visible_docs() {
            if repo_name(&set, si, dm.repo_id) != repo {
                continue;
            }
            let content = set.content(si, docid).unwrap();
            if content
                .windows(needle.len())
                .any(|w| w == needle)
            {
                return true;
            }
        }
        false
    }

    /// Live docs listing a single gram across all shards.
    fn gram_posting_count(data_dir: &Path, gram: &[u8]) -> usize {
        let set = ShardSet::open_dir(&data_dir.join("shards")).unwrap();
        set.posting_doc_ids(gram).len()
    }

    fn open_cas(data_dir: &Path) -> Cas {
        Cas::open(&data_dir.join("cas")).unwrap()
    }

    /// Two binaries with different extension maps share one data dir: the
    /// one that does not know a type must not tombstone the other's docs.
    #[test]
    fn delta_keeps_docs_of_file_types_this_walker_does_not_know() {
        let dir = tempfile::tempdir().unwrap();
        write_files(dir.path(), &[("util.py", UTIL_PY), ("notes.zzz", b"ZZZ_TOKEN = 1\n")]);
        let data = tempfile::tempdir().unwrap();
        let cas = open_cas(data.path());
        index_dir(dir.path(), "p1", data.path(), &cas).unwrap();
        assert_eq!(visible_paths(data.path(), "p1"), vec!["util.py"], "zzz is unknown to this build");

        // a newer binary indexed notes.zzz: the same doc under that path
        let mut docs = extract_docs(vec![("notes.py".to_string(), b"ZZZ_TOKEN = 1\n".to_vec())], &cas);
        docs[0].meta.path = "notes.zzz".to_string();
        write_shard(&data.path().join("shards"), "p1", &docs).unwrap();
        assert_eq!(visible_paths(data.path(), "p1"), vec!["notes.zzz", "util.py"]);

        // this build's delta walks past notes.zzz but the file exists: kept
        let state = load_state(data.path(), "p1").unwrap();
        let r = reindex_dir(&state, data.path(), &cas, None).unwrap();
        assert_eq!((r.docs_added, r.docs_deleted), (0, 0), "{r:?}");
        assert_eq!(visible_paths(data.path(), "p1"), vec!["notes.zzz", "util.py"]);

        // the file is deleted: now it goes
        fs::remove_file(dir.path().join("notes.zzz")).unwrap();
        let r = reindex_dir(&state, data.path(), &cas, None).unwrap();
        assert_eq!((r.docs_added, r.docs_deleted), (0, 1), "{r:?}");
        assert_eq!(visible_paths(data.path(), "p1"), vec!["util.py"]);
    }

    #[test]
    fn head_moved_only_for_committed_git_repos_whose_head_changed() {
        let data_td = tempfile::tempdir().unwrap();
        let data = data_td.path();
        // a plain folder: never
        let plain = tempfile::tempdir().unwrap();
        write_files(plain.path(), &[("a.rs", MAIN_RS)]);
        index_dir(plain.path(), "plain", data, &open_cas(data)).unwrap();
        assert!(!head_moved(data, "plain"));
        // a git repo indexed at HEAD: not moved until a new commit lands
        let repo = make_repo(&[("src/main.rs", MAIN_RS)]);
        index_repo(repo.path(), "repo", data, &open_cas(data)).unwrap();
        assert!(!head_moved(data, "repo"));
        write_files(repo.path(), &[("src/lib.rs", LIB_RS)]);
        commit_all(repo.path(), "lib");
        assert!(head_moved(data, "repo"));
        reindex_repo("repo", data, &open_cas(data)).unwrap();
        assert!(!head_moved(data, "repo"));
        // a worktree-derived state belongs to the auto-refresh: skipped
        reindex_worktree("repo", data, &open_cas(data)).unwrap();
        write_files(repo.path(), &[("src/x.rs", b"fn x() {}
")]);
        commit_all(repo.path(), "x");
        assert!(!head_moved(data, "repo"));
        assert!(!head_moved(data, "no-such-repo"));
    }

    const MAIN_RS: &[u8] = b"fn main() {\n    println!(\"hello world\");\n}\n";
    const LIB_RS: &[u8] = b"pub fn library_function() -> i32 { 42 }\n";
    const UTIL_PY: &[u8] = b"def util_function():\n    return 'util'\n";

    #[test]
    fn index_repo_full() {
        let repo = make_repo(&[
            ("src/main.rs", MAIN_RS),
            ("src/lib.rs", LIB_RS),
            ("util.py", UTIL_PY),
            ("README.md", b"# readme\n"),              // text: indexed (SPEC-P9)
            ("logo.png", b"PNG\n"),                    // unknown lang: skipped
            ("bin.py", b"def f():\x00 pass\n"),        // binary: skipped
        ]);
        let data = tempfile::tempdir().unwrap();
        let cas = open_cas(data.path());

        let report = index_repo(repo.path(), "r1", data.path(), &cas).unwrap();
        assert_eq!(report.repo, "r1");
        assert_eq!(report.commit, head_hex(repo.path()));
        assert_eq!(report.docs_added, 4, "{report:?}");
        assert_eq!(report.docs_deleted, 0);
        assert_eq!(report.docs_unchanged, 0);
        assert_eq!(report.cas_misses, 4);
        assert_eq!(report.cas_hits, 0);
        assert_eq!(
            report.bytes_indexed,
            (MAIN_RS.len() + LIB_RS.len() + UTIL_PY.len() + "# readme\n".len()) as u64
        );

        assert_eq!(shard_files(data.path()).len(), 1);
        let (entries, bytes) = cas.stats();
        assert_eq!(entries, 4);
        assert!(bytes > 0);

        let state = load_repo_state(data.path(), "r1");
        assert_eq!(state.name, "r1");
        assert_eq!(state.path, repo.path());
        assert_eq!(state.last_commit.as_deref(), Some(report.commit.as_str()));
        assert!(state.indexed_at.ends_with('Z'));

        assert_eq!(
            visible_paths(data.path(), "r1"),
            vec!["README.md", "src/lib.rs", "src/main.rs", "util.py"]
        );
    }

    #[test]
    fn index_repo_twice_errors() {
        let repo = make_repo(&[("src/main.rs", MAIN_RS)]);
        let data = tempfile::tempdir().unwrap();
        let cas = open_cas(data.path());
        index_repo(repo.path(), "r1", data.path(), &cas).unwrap();
        let err = index_repo(repo.path(), "r1", data.path(), &cas).unwrap_err();
        assert!(
            err.to_string().contains("already registered"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn reindex_worktree_sees_uncommitted_edits_then_head_reconciles() {
        let repo = make_repo(&[("src/main.rs", MAIN_RS), ("util.py", UTIL_PY)]);
        let data = tempfile::tempdir().unwrap();
        let cas = open_cas(data.path());
        index_repo(repo.path(), "r1", data.path(), &cas).unwrap();

        // Uncommitted: edit main.rs (CRLF, as a Windows checkout would have
        // it), add an untracked file, add an ignored one.
        write_files(
            repo.path(),
            &[
                ("src/main.rs", b"fn main() {\r\n    println!(\"worktree_only_token\");\r\n}\r\n"),
                ("fresh.py", b"FRESH_UNTRACKED_TOKEN = 1\n"),
                (".gitignore", b"ignored.py\n"),
                ("ignored.py", b"IGNORED_TOKEN = 1\n"),
            ],
        );
        let r = reindex_worktree("r1", data.path(), &cas).unwrap();
        assert_eq!(r.commit, "worktree");
        assert_eq!((r.docs_added, r.docs_deleted, r.docs_unchanged), (2, 1, 1), "{r:?}");
        assert!(content_contains(data.path(), "r1", b"worktree_only_token"));
        assert!(content_contains(data.path(), "r1", b"FRESH_UNTRACKED_TOKEN"));
        assert!(!content_contains(data.path(), "r1", b"IGNORED_TOKEN"));
        assert!(!content_contains(data.path(), "r1", b"hello world"));
        // stored LF-normalised
        assert!(!content_contains(data.path(), "r1", b"\r\n"));
        let st = load_repo_state(data.path(), "r1");
        assert!(st.worktree);

        // A second worktree pass is a no-op (CRLF on disk must not count).
        let r2 = reindex_worktree("r1", data.path(), &cas).unwrap();
        assert_eq!((r2.docs_added, r2.docs_deleted, r2.docs_unchanged), (0, 0, 3), "{r2:?}");

        // HEAD-based reindex reconciles by content: the uncommitted edit and
        // the untracked file disappear again, util.py stays untouched.
        let r3 = reindex_repo("r1", data.path(), &cas).unwrap();
        assert_eq!(r3.commit, head_hex(repo.path()));
        assert_eq!((r3.docs_added, r3.docs_deleted, r3.docs_unchanged), (1, 2, 1), "{r3:?}");
        assert!(content_contains(data.path(), "r1", b"hello world"));
        assert!(!content_contains(data.path(), "r1", b"worktree_only_token"));
        assert!(!content_contains(data.path(), "r1", b"FRESH_UNTRACKED_TOKEN"));
        assert!(!load_repo_state(data.path(), "r1").worktree);

        // Commit the edits: a worktree pass then a HEAD pass agree.
        commit_all(repo.path(), "edits");
        let r4 = reindex_worktree("r1", data.path(), &cas).unwrap();
        assert_eq!((r4.docs_added, r4.docs_deleted), (2, 1), "{r4:?}");
        let r5 = reindex_repo("r1", data.path(), &cas).unwrap();
        assert_eq!((r5.docs_added, r5.docs_deleted, r5.docs_unchanged), (0, 0, 3), "{r5:?}");
        assert_eq!(
            visible_paths(data.path(), "r1"),
            vec!["fresh.py", "src/main.rs", "util.py"]
        );
    }

    /// SPEC-P10 §24: an untracked cargo target dir under a custom name is
    /// build output (it carries `CACHEDIR.TAG`), not source — for the
    /// working-tree listing and the plain-tree walk alike.
    #[test]
    fn cache_tagged_dirs_are_not_indexed() {
        let repo = make_repo(&[("src/main.rs", MAIN_RS)]);
        write_files(
            repo.path(),
            &[
                ("target-bench-B/CACHEDIR.TAG", b"Signature: 8a477f597d28d172789f06886806bc55\n"),
                ("target-bench-B/debug/deps/anyhow-1.d", b"anyhow.rlib: src/lib.rs\n"),
                ("bench/keep.py", b"KEEP_TOKEN = 1\n"),
            ],
        );
        let data = tempfile::tempdir().unwrap();
        let cas = open_cas(data.path());
        index_repo(repo.path(), "r1", data.path(), &cas).unwrap();
        let r = reindex_worktree("r1", data.path(), &cas).unwrap();
        assert_eq!(r.docs_added, 1, "{r:?}");
        assert_eq!(visible_paths(data.path(), "r1"), vec!["bench/keep.py", "src/main.rs"]);
        let walked = walk_plain_tree(repo.path()).unwrap();
        assert!(walked.iter().any(|p| p == "bench/keep.py"));
        assert!(!walked.iter().any(|p| p.starts_with("target-bench-B")), "{walked:?}");
        let mut memo = HashMap::new();
        assert!(under_cache_dir(repo.path(), "target-bench-B/debug/deps/anyhow-1.d", &mut memo));
        assert!(!under_cache_dir(repo.path(), "bench/keep.py", &mut memo));
        assert_eq!(memo.len(), 2, "one stat per directory: {memo:?}");
    }

    #[test]
    fn reindex_delta_add_modify_delete() {
        let repo = make_repo(&[
            ("src/main.rs", MAIN_RS),
            ("src/lib.rs", LIB_RS),
            ("util.py", UTIL_PY),
        ]);
        let data = tempfile::tempdir().unwrap();
        let cas = open_cas(data.path());
        index_repo(repo.path(), "r1", data.path(), &cas).unwrap();
        assert!(content_contains(data.path(), "r1", b"hello world"));

        // modify main.rs, delete lib.rs, add extra.py
        const NEW_MAIN: &[u8] = b"fn main() {\n    println!(\"brand new token xyz\");\n}\n";
        write_files(repo.path(), &[("src/main.rs", NEW_MAIN), ("extra.py", b"EXTRA_PYTHON_TOKEN = 1\n")]);
        fs::remove_file(repo.path().join("src/lib.rs")).unwrap();
        commit_all(repo.path(), "delta");

        let report = reindex_repo("r1", data.path(), &cas).unwrap();
        assert_eq!(report.commit, head_hex(repo.path()));
        assert_eq!(report.docs_added, 2, "{report:?}"); // new main.rs + extra.py
        assert_eq!(report.docs_deleted, 2, "{report:?}"); // old main.rs + lib.rs
        assert_eq!(report.docs_unchanged, 1, "{report:?}"); // util.py
        assert_eq!(report.cas_hits, 0);
        assert_eq!(report.cas_misses, 2);

        // one shard per run
        assert_eq!(shard_files(data.path()).len(), 2);

        // old content is unfindable (tombstoned)
        assert!(!content_contains(data.path(), "r1", b"hello world"));
        assert!(!content_contains(data.path(), "r1", b"library_function"));
        assert_eq!(gram_posting_count(data.path(), b"hel"), 0);
        assert_eq!(gram_posting_count(data.path(), b"lib"), 0);
        // new + unchanged content is findable
        assert!(content_contains(data.path(), "r1", b"brand new token xyz"));
        assert!(content_contains(data.path(), "r1", b"EXTRA_PYTHON_TOKEN"));
        assert!(content_contains(data.path(), "r1", b"util_function"));

        assert_eq!(
            visible_paths(data.path(), "r1"),
            vec!["extra.py", "src/main.rs", "util.py"]
        );

        // state moved to the new commit
        let state = load_repo_state(data.path(), "r1");
        assert_eq!(state.last_commit.as_deref(), Some(report.commit.as_str()));

        // no-op reindex: nothing changed
        let report = reindex_repo("r1", data.path(), &cas).unwrap();
        assert_eq!(report.docs_added, 0, "{report:?}");
        assert_eq!(report.docs_deleted, 0, "{report:?}");
        assert_eq!(report.docs_unchanged, 3, "{report:?}");
        assert_eq!(shard_files(data.path()).len(), 2, "no empty delta shard");
    }

    #[test]
    fn fork_dedup_all_cas_hits() {
        let files: &[(&str, &[u8])] = &[
            ("src/a.rs", b"pub fn alpha() -> i32 { 1 }\n"),
            ("src/b.rs", b"pub fn beta() -> i32 { 2 }\n"),
            ("tool.py", b"def tool():\n    return 3\n"),
        ];
        let repo_a = make_repo(files);
        let repo_b = make_repo(files); // separate git init, identical content
        let data = tempfile::tempdir().unwrap();
        let cas = open_cas(data.path());

        let ra = index_repo(repo_a.path(), "fork-a", data.path(), &cas).unwrap();
        assert_eq!(ra.docs_added, 3);
        assert_eq!(ra.cas_misses, 3);
        assert_eq!(ra.cas_hits, 0);

        // Headline feature: the fork pays zero extraction work.
        let rb = index_repo(repo_b.path(), "fork-b", data.path(), &cas).unwrap();
        assert_eq!(rb.docs_added, 3);
        assert_eq!(rb.cas_misses, 0, "{rb:?}");
        assert_eq!(rb.cas_hits, 3, "{rb:?}");

        // CAS not bloated by the fork.
        assert_eq!(cas.stats().0, 3);
    }

    #[test]
    fn reindex_missing_old_commit_falls_back_to_full() {
        let repo = make_repo(&[("src/main.rs", MAIN_RS)]);
        let data = tempfile::tempdir().unwrap();
        let cas = open_cas(data.path());
        index_repo(repo.path(), "r1", data.path(), &cas).unwrap();

        write_files(repo.path(), &[("src/new.rs", b"fn new_stuff() {}\n")]);
        commit_all(repo.path(), "second");

        // Simulate GC of the previously indexed commit.
        let mut state = load_repo_state(data.path(), "r1");
        state.last_commit = Some("deadbeef".repeat(5));
        save_state(data.path(), &state).unwrap();

        let report = reindex_repo("r1", data.path(), &cas).unwrap();
        assert_eq!(report.commit, head_hex(repo.path()));
        assert_eq!(report.docs_added, 2, "{report:?}");
        assert_eq!(report.docs_deleted, 1, "{report:?}"); // previous docs tombstoned
        assert_eq!(report.docs_unchanged, 0, "{report:?}");
        assert_eq!(
            visible_paths(data.path(), "r1"),
            vec!["src/main.rs", "src/new.rs"]
        );
        // main.rs content identical across commits -> CAS hit this time
        assert_eq!(report.cas_hits, 1, "{report:?}");
        assert_eq!(report.cas_misses, 1, "{report:?}");

        let state = load_repo_state(data.path(), "r1");
        assert_eq!(state.last_commit.as_deref(), Some(report.commit.as_str()));
    }

    #[test]
    fn reindex_non_fast_forward_falls_back_to_full() {
        let repo = make_repo(&[("src/main.rs", MAIN_RS)]);
        let data = tempfile::tempdir().unwrap();
        let cas = open_cas(data.path());
        index_repo(repo.path(), "r1", data.path(), &cas).unwrap();

        // Rewrite history: amend replaces the indexed commit (old commit
        // still exists in the object store but is not a HEAD ancestor).
        write_files(repo.path(), &[("src/main.rs", b"fn main() {\n    println!(\"rewritten history\");\n}\n")]);
        git(repo.path(), &["add", "-A"]);
        git(repo.path(), &["commit", "-q", "--amend", "-m", "amended"]);

        let report = reindex_repo("r1", data.path(), &cas).unwrap();
        assert_eq!(report.commit, head_hex(repo.path()));
        assert_eq!(report.docs_added, 1, "{report:?}");
        assert_eq!(report.docs_deleted, 1, "{report:?}");
        assert_eq!(report.docs_unchanged, 0, "{report:?}");
        assert!(!content_contains(data.path(), "r1", b"hello world"));
        assert!(content_contains(data.path(), "r1", b"rewritten history"));
    }

    #[test]
    fn reindex_unregistered_repo_errors() {
        let data = tempfile::tempdir().unwrap();
        let cas = open_cas(data.path());
        let err = reindex_repo("nope", data.path(), &cas).unwrap_err();
        assert!(
            err.to_string().contains("not registered"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn cas_corrupt_entry_is_a_miss() {
        let td = tempfile::tempdir().unwrap();
        let cas = Cas::open(td.path()).unwrap();
        let content = b"pub fn corrupt_me() {}\n";
        let id = BlobId::from_content(content);
        assert!(cas.get(&id).is_none());

        // write garbage at the entry path
        let hex = id.hex();
        let sub = td.path().join(&hex[..2]);
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join(&hex[2..]), b"not bincode").unwrap();
        assert!(cas.get(&id).is_none(), "corrupt entry must be a miss");

        // put overwrites the corrupt entry
        let art = ExtractedArtifact {
            ngrams: vec![b"abc".to_vec()],
            ..Default::default()
        };
        cas.put(&id, &art).unwrap();
        let got = cas.get(&id).unwrap();
        assert_eq!(got.ngrams, art.ngrams);
        assert_eq!(cas.stats().0, 1);
    }

    #[test]
    fn iso_timestamp_format() {
        assert_eq!(unix_to_iso(0), "1970-01-01T00:00:00Z");
        assert_eq!(unix_to_iso(1_700_000_000), "2023-11-14T22:13:20Z");
        assert_eq!(unix_to_iso(951_782_399), "2000-02-28T23:59:59Z");
        assert_eq!(unix_to_iso(951_782_400), "2000-02-29T00:00:00Z"); // leap day
    }
}
