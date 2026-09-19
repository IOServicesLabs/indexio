//! MCP server over stdio: hand-rolled JSON-RPC 2.0 with serde_json only.
//!
//! Transport: newline-delimited JSON — each stdin line is one complete
//! JSON-RPC message (a single object, or an array for batch requests).
//! Content-Length framing is intentionally NOT supported.
//!
//! Contract: docs/SPEC.md, section "indexio — binary"; SPEC-P8 (harness
//! integration): the `initialize` result carries `instructions` that tell
//! the harness to prefer these tools over its own grep/glob/read, the tool
//! descriptions say which built-in they replace, `list_files` covers glob,
//! `refresh_index` lets the model repair staleness itself, and
//! `index_stats` reports per-repo freshness.
//!
//! SECURITY NOTE (SPEC-P3 §3): the MCP stdio channel is the TRUSTED local
//! interface — it runs as the invoking user over a pipe, so it is
//! intentionally NOT authenticated and NOT repo-ACL filtered. HTTP auth
//! and ACLs live only in `indexio serve`. Do not expose this handler over a
//! network transport.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use serde_json::{json, Value};

use indexio_embed::embed::Embedder;
use indexio_embed::rerank::Reranker;
use indexio_query::{Engine, FusionAlgo, SearchMode};

use crate::mcp_text;

const PARSE_ERROR: i64 = -32700;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32603;

const DEFAULT_LIMIT: usize = 50;

/// `code_grep` lists this many matching lines per file by default
/// (`grep -n` semantics, SPEC-P9 §19); `code_search` lists 1 (the densest
/// line) unless asked with `lines`.
const GREP_LINES_PER_FILE: usize = 20;
/// Upper bound for `lines`.
const MAX_LINES_PER_FILE: usize = 100;

/// `find_symbol` hits that get a definition end line (a cached
/// tree-sitter parse per distinct file).
const RANGE_HITS: usize = 20;

/// `refresh_index` merges the oldest shards once more than this many exist…
const COMPACT_ABOVE: usize = 12;
/// …down to this many.
const COMPACT_TO: usize = 6;

/// How tool results are rendered (SPEC-P9): `Text` is the token-dense
/// default for models; `Json` the compact JSON envelope for scripts/tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
}

impl OutputFormat {
    /// `INDEXIO_MCP_FORMAT=json|text` (default text).
    pub fn from_env() -> OutputFormat {
        match std::env::var("INDEXIO_MCP_FORMAT").as_deref() {
            Ok("json") | Ok("JSON") => OutputFormat::Json,
            _ => OutputFormat::Text,
        }
    }

    fn parse(s: &str) -> Option<OutputFormat> {
        match s {
            "json" => Some(OutputFormat::Json),
            "text" => Some(OutputFormat::Text),
            _ => None,
        }
    }
}

/// One MCP server instance: the engine (reloadable after `refresh_index`),
/// the embedder for the semantic plane and the reranker stage.
pub struct McpServer {
    engine: RwLock<Engine>,
    embedder: Arc<dyn Embedder>,
    reranker: Arc<dyn Reranker>,
    format: OutputFormat,
    /// The registered repo this session works in (SPEC-P9): its hits are
    /// listed first and its impact call sites are shown in full. `None`
    /// until the working directory is under a registered repo — resolved
    /// at start and, while still `None`, again once a minute (SPEC-P10
    /// §23: a session started before its folder was registered).
    current_repo: RwLock<Option<String>>,
    /// Filesystem watch on the current repo: when it reports changes, the
    /// next tool call re-indexes the working tree first (auto-refresh).
    watch: Mutex<Option<crate::watch::RepoWatch>>,
    /// Last late-resolution attempt (see `current_repo`).
    repo_checked: Mutex<Option<std::time::Instant>>,
    /// Last periodic shard-count check (SPEC-P10 §29).
    compact_checked: Mutex<Option<std::time::Instant>>,
    /// The model save running off the call path (SPEC-P10 §30), joined
    /// before the next one starts and at shutdown.
    saver: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Stat cache for the current repo's working tree so a refresh only
    /// reads files that changed.
    wt_cache: Mutex<indexio_ingest::WorktreeCache>,
    /// A background shard compaction is running.
    compacting: Arc<AtomicBool>,
    /// A background compaction finished: reopen the engine on the next call.
    reload_needed: Arc<AtomicBool>,
    /// The compaction thread, joined on graceful shutdown so a merge is
    /// never abandoned between writing the merged segment and removing
    /// its inputs.
    compactor: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Engine whose sidecars were opened by `prewarm_sidecars`, waiting
    /// to be adopted.
    warm: Arc<Mutex<Option<Engine>>>,
    /// Paths of the current repo whose semantic rows still need embedding
    /// (an embed failed, e.g. another server was mid-append): retried on
    /// the next call.
    pending_embed: Mutex<std::collections::BTreeSet<String>>,
    /// (count, total bytes, newest mtime) of the shard files at the last
    /// check; a change means another process wrote or merged shards.
    shards_seen: Mutex<(usize, u64, Option<std::time::SystemTime>)>,
    /// Last transcript import (SPEC-P9 recall); throttled to
    /// [`SESSIONS_IMPORT_EVERY`] and run on a background thread.
    sessions_last: Mutex<Option<std::time::Instant>>,
    sessions_busy: Arc<AtomicBool>,
    /// Stat cache for the `sessions` folder (SPEC-P10): only rewritten
    /// transcript parts are re-read by the periodic import.
    sessions_cache: Arc<Mutex<indexio_ingest::WorktreeCache>>,
    /// The same for the `runs` folder (SPEC-P10 §31).
    runs_cache: Arc<Mutex<indexio_ingest::WorktreeCache>>,
    /// Last background sync of the other registered repos (SPEC-P10 §21);
    /// throttled to [`SYNC_EVERY`], one server at a time across the data dir.
    sync_last: Mutex<Option<std::time::Instant>>,
    sync_busy: Arc<AtomicBool>,
    /// The binary this server was started from and its (len, mtime) then:
    /// a newer install is noticed on a later call and reported once
    /// (SPEC-P10 §12 — sessions kept running week-old servers for days).
    exe: Option<(std::path::PathBuf, (u64, Option<std::time::SystemTime>))>,
    exe_checked: Mutex<Option<std::time::Instant>>,
    stale_noted: AtomicBool,
    /// Files this session has already been charged a whole-file read for
    /// (SPEC-P10 §22): only the first `read_span` of a file is measured
    /// against the harness reading it whole.
    read_seen: Mutex<HashSet<(String, String)>>,
}

type ReadSeen = Mutex<HashSet<(String, String)>>;

/// How often the running server re-imports session transcripts.
const SESSIONS_IMPORT_EVERY: std::time::Duration = std::time::Duration::from_secs(180);
/// How often a server probes the other registered repos for moved HEADs.
const SYNC_EVERY: std::time::Duration = std::time::Duration::from_secs(600);
/// How often a server without a current repo looks for one (SPEC-P10 §23).
const ADOPT_EVERY: std::time::Duration = std::time::Duration::from_secs(60);
/// How often a server counts the shards for a compaction it did not cause.
const COMPACT_CHECK_EVERY: std::time::Duration = std::time::Duration::from_secs(60);

/// Guidance block for a harness's project/user instructions file
/// (`indexio setup claude` prints/appends it).
pub const HARNESS_GUIDANCE: &str = "\
## indexio (code index MCP server)

When the `indexio` MCP tools are available, use them BEFORE the built-in Grep / Glob /
Read / find tools for anything inside an indexed repo — they answer in milliseconds from
a pre-built index of every repo (source AND docs, configs, scripts, SQL, markup), ranked,
with symbols and call graph:

- `code_search` (mode \"hybrid\" for questions, \"lexical\" for exact identifiers or regex)
  instead of grep/rg
- `find_symbol` / `who_calls` for definitions and call sites instead of grepping for them
- `list_files` instead of Glob/find
- `file_outline` then `read_span` instead of reading whole files: outline first, then only
  the lines you need
- `impact_of_symbol` BEFORE changing a function or type; `impact_of_diff` AFTER editing
  (it reads the uncommitted working tree) to find every caller and importer to update
- `recall` to find earlier decisions, findings and commands from this or any other
  session (all transcripts are indexed) instead of re-deriving them — especially after a
  context compaction; keep pointers (repo:path:lines, session ids) in context, not content

The repo you are working in is re-indexed automatically from its working tree before each
call whenever files changed, so your own edits (uncommitted, untracked) are searchable
without any refresh. Call `refresh_index` only for other repos or when results look
stale. Use the local tools only for repos that are not indexed.
";

impl McpServer {
    pub fn new(engine: Engine, embedder: Arc<dyn Embedder>, reranker: Arc<dyn Reranker>) -> Self {
        Self::with_format(engine, embedder, reranker, OutputFormat::from_env())
    }

    pub fn with_format(
        engine: Engine,
        embedder: Arc<dyn Embedder>,
        reranker: Arc<dyn Reranker>,
        format: OutputFormat,
    ) -> Self {
        McpServer {
            engine: RwLock::new(engine),
            embedder,
            reranker,
            format,
            current_repo: RwLock::new(None),
            watch: Mutex::new(None),
            repo_checked: Mutex::new(None),
            compact_checked: Mutex::new(None),
            saver: Mutex::new(None),
            wt_cache: Mutex::new(indexio_ingest::WorktreeCache::default()),
            compacting: Arc::new(AtomicBool::new(false)),
            reload_needed: Arc::new(AtomicBool::new(false)),
            compactor: Mutex::new(None),
            warm: Arc::new(Mutex::new(None)),
            pending_embed: Mutex::new(std::collections::BTreeSet::new()),
            sessions_last: Mutex::new(None),
            sessions_busy: Arc::new(AtomicBool::new(false)),
            sessions_cache: Arc::new(Mutex::new(indexio_ingest::WorktreeCache::default())),
            runs_cache: Arc::new(Mutex::new(indexio_ingest::WorktreeCache::default())),
            shards_seen: Mutex::new((0, 0, None)),
            exe: std::env::current_exe().ok().and_then(|p| {
                let m = std::fs::metadata(&p).ok()?;
                Some((p, (m.len(), m.modified().ok())))
            }),
            exe_checked: Mutex::new(None),
            stale_noted: AtomicBool::new(false),
            sync_last: Mutex::new(None),
            sync_busy: Arc::new(AtomicBool::new(false)),
            read_seen: Mutex::new(HashSet::new()),
        }
    }

    /// Keep the repos this session is NOT working in current (SPEC-P10
    /// §21): every [`SYNC_EVERY`], on a background thread, the registered
    /// git repos whose HEAD moved since they were indexed get a HEAD-based
    /// delta and an embed. The session's own repo is the auto-refresh's
    /// (working tree, per call); plain folders and repos another server
    /// indexes from a working tree are skipped; one server across the data
    /// dir does it at a time. A pass with nothing changed costs one ref
    /// resolve per repo.
    fn maybe_sync_others(&self) {
        {
            let mut last = self.sync_last.lock().expect("sync_last poisoned");
            if last.map_or(false, |t| t.elapsed() < SYNC_EVERY) {
                return;
            }
            *last = Some(std::time::Instant::now());
        }
        if self.sync_busy.swap(true, Ordering::AcqRel) {
            return;
        }
        let data_dir: Option<std::path::PathBuf> =
            self.engine.read().expect("engine lock poisoned").data_dir().map(|d| d.to_path_buf());
        let Some(data_dir) = data_dir else {
            self.sync_busy.store(false, Ordering::Release);
            return;
        };
        let current = self.current();
        let embedder = Arc::clone(&self.embedder);
        let busy = Arc::clone(&self.sync_busy);
        let reload = Arc::clone(&self.reload_needed);
        std::thread::spawn(move || {
            let t0 = std::time::Instant::now();
            let Some(_lock) = indexio_ingest::SyncLock::try_acquire(&data_dir) else {
                busy.store(false, Ordering::Release);
                return;
            };
            let names = indexio_ingest::sources::registered_repo_names(&data_dir).unwrap_or_default();
            let moved: Vec<String> = names
                .into_iter()
                .filter(|n| current.as_deref() != Some(n.as_str()) && n != crate::sessions::REPO)
                .filter(|n| indexio_ingest::head_moved(&data_dir, n))
                .collect();
            if moved.is_empty() {
                busy.store(false, Ordering::Release);
                return;
            }
            let mut changed: Vec<String> = Vec::new();
            if let Ok(cas) = indexio_ingest::Cas::open(&data_dir.join("cas")) {
                for n in &moved {
                    match indexio_ingest::reindex_repo(n, &data_dir, &cas) {
                        Ok(r) if r.docs_added > 0 || r.docs_deleted > 0 => changed.push(n.clone()),
                        Ok(_) => {}
                        Err(e) => tracing::warn!(repo = %n, error = %e, "background sync failed"),
                    }
                }
            }
            if !changed.is_empty() {
                if let Ok(set) = indexio_index::ShardSet::open_dir(&data_dir.join("shards")) {
                    if let Err(e) = indexio_embed::pipeline::embed_repos_with(
                        &set,
                        &data_dir,
                        &changed,
                        embedder.as_ref(),
                        indexio_embed::pipeline::MAX_CHARS,
                    ) {
                        tracing::warn!(error = %e, "background sync: embed failed");
                    }
                }
                reload.store(true, Ordering::Release);
            }
            tracing::info!(
                probed = moved.len(),
                changed = changed.len(),
                ms = t0.elapsed().as_millis() as u64,
                "background sync of other repos"
            );
            busy.store(false, Ordering::Release);
        });
    }

    /// Whether a different indexio binary now sits at this server's exe
    /// path (a deploy renamed ours aside and copied a new one in). Checked
    /// at most once a minute; `true` is returned once, so the note lands
    /// on exactly one tool result.
    fn binary_updated_once(&self) -> bool {
        let Some((path, stamp)) = &self.exe else { return false };
        if self.stale_noted.load(Ordering::Relaxed) {
            return false;
        }
        {
            let mut last = self.exe_checked.lock().expect("exe_checked poisoned");
            if last.map_or(false, |t| t.elapsed() < std::time::Duration::from_secs(60)) {
                return false;
            }
            *last = Some(std::time::Instant::now());
        }
        let now = std::fs::metadata(path).ok().map(|m| (m.len(), m.modified().ok()));
        match now {
            Some(n) if n != *stamp => {
                self.stale_noted.store(true, Ordering::Relaxed);
                true
            }
            _ => false,
        }
    }

    /// Re-import this project's transcripts (and index + embed what
    /// changed) on a background thread, at most every few minutes and
    /// never on the request path. The next call adopts the result through
    /// `reload_needed`.
    fn maybe_import_sessions(&self) {
        {
            let mut last = self.sessions_last.lock().expect("sessions_last poisoned");
            if last.map_or(false, |t| t.elapsed() < SESSIONS_IMPORT_EVERY) {
                return;
            }
            *last = Some(std::time::Instant::now());
        }
        if self.sessions_busy.swap(true, Ordering::AcqRel) {
            return;
        }
        let Some(claude) = crate::sessions::default_claude_dir() else {
            self.sessions_busy.store(false, Ordering::Release);
            return;
        };
        let data_dir: Option<std::path::PathBuf> = {
            let guard = self.engine.read().expect("engine lock poisoned");
            guard.data_dir().map(|d| d.to_path_buf())
        };
        let Some(data_dir) = data_dir else {
            self.sessions_busy.store(false, Ordering::Release);
            return;
        };
        let slug = std::env::current_dir().ok().map(|d| crate::sessions::project_slug(&d));
        let embedder = Arc::clone(&self.embedder);
        let busy = Arc::clone(&self.sessions_busy);
        let reload = Arc::clone(&self.reload_needed);
        let cache = Arc::clone(&self.sessions_cache);
        let runs_cache = Arc::clone(&self.runs_cache);
        std::thread::spawn(move || {
            let t0 = std::time::Instant::now();
            // SPEC-P10 §31: the command-output source rides the same cycle —
            // roll off old logs, delta-index the folder, embed what changed
            let rolled = crate::runs::retain(&data_dir, crate::runs::retain_days());
            if data_dir.join(crate::runs::REPO).is_dir() {
                let indexed = indexio_ingest::Cas::open(&data_dir.join("cas"))
                    .map_err(anyhow::Error::from)
                    .and_then(|cas| {
                        let mut c = runs_cache.lock().expect("runs_cache poisoned");
                        crate::runs::index_runs(&data_dir, &cas, Some(&mut c))
                    });
                match indexed {
                    Ok(ir) if ir.docs_added > 0 || ir.docs_deleted > 0 => {
                        if !ir.changed_paths.is_empty() {
                            if let Ok(set) = indexio_index::ShardSet::open_dir(&data_dir.join("shards")) {
                                if let Err(e) = indexio_embed::pipeline::embed_paths(
                                    &set,
                                    &data_dir,
                                    crate::runs::REPO,
                                    &ir.changed_paths,
                                    embedder.as_ref(),
                                    indexio_embed::pipeline::MAX_CHARS,
                                ) {
                                    tracing::warn!(error = %e, "runs: embed failed");
                                }
                            }
                        }
                        tracing::info!(added = ir.docs_added, deleted = ir.docs_deleted, rolled, "runs indexed");
                        reload.store(true, Ordering::Release);
                    }
                    Ok(_) => {}
                    Err(e) => tracing::warn!(error = %e, "runs: index failed"),
                }
            }
            let out = data_dir.join("sessions");
            match crate::sessions::import(&claude, &out, slug.as_deref()) {
                Ok(r) if r.sessions_imported > 0 || r.sessions_rolled_off > 0 => {
                    let indexed = indexio_ingest::Cas::open(&data_dir.join("cas"))
                        .map_err(anyhow::Error::from)
                        .and_then(|cas| {
                            let mut c = cache.lock().expect("sessions_cache poisoned");
                            crate::sessions::index_sessions(&data_dir, &cas, Some(&mut c))
                        });
                    match indexed {
                        Ok(ir) => {
                            if !ir.changed_paths.is_empty() {
                                if let Ok(set) = indexio_index::ShardSet::open_dir(&data_dir.join("shards")) {
                                    if let Err(e) = indexio_embed::pipeline::embed_paths(
                                        &set,
                                        &data_dir,
                                        crate::sessions::REPO,
                                        &ir.changed_paths,
                                        embedder.as_ref(),
                                        indexio_embed::pipeline::MAX_CHARS,
                                    ) {
                                        tracing::warn!(error = %e, "sessions: embed failed");
                                    }
                                }
                            }
                            tracing::info!(
                                imported = r.sessions_imported,
                                parts = r.parts_written,
                                added = ir.docs_added,
                                ms = t0.elapsed().as_millis() as u64,
                                "sessions imported"
                            );
                            reload.store(true, Ordering::Release);
                        }
                        Err(e) => tracing::warn!(error = %e, "sessions: index failed"),
                    }
                }
                Ok(_) => {}
                Err(e) => tracing::warn!(error = %e, "sessions: import failed"),
            }
            busy.store(false, Ordering::Release);
        });
    }

    /// Embed `paths` of the current repo into the semantic plane through
    /// the engine's cached segment sets; paths that could not be embedded
    /// stay queued for the next call.
    fn embed_current_paths(&self, repo: &str, data_dir: &std::path::Path, paths: &[String]) {
        let mut queue = self.pending_embed.lock().expect("pending embed poisoned");
        queue.extend(paths.iter().cloned());
        if queue.is_empty() {
            return;
        }
        let batch: Vec<String> = queue.iter().cloned().collect();
        let set = match indexio_index::ShardSet::open_dir(&data_dir.join("shards")) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "auto-refresh: cannot open shards for embed");
                return;
            }
        };
        // the engine's cached (already parsed) segment sets
        let model = self.embedder.model_id();
        let (vs, bs) = {
            let eng = self.engine.read().expect("engine lock poisoned");
            (eng.vec_index(model).ok().flatten(), eng.bm25_index(model).ok().flatten())
        };
        let sets = match (&vs, &bs) {
            (Some(v), Some(b)) => Some((&**v, &**b)),
            _ => None,
        };
        match indexio_embed::pipeline::embed_paths_with_sets(
            &set,
            data_dir,
            repo,
            &batch,
            sets,
            self.embedder.as_ref(),
            indexio_embed::pipeline::MAX_CHARS,
        ) {
            Ok(_) => queue.clear(),
            Err(e) => tracing::warn!(error = %e, pending = queue.len(), "auto-refresh: embed deferred"),
        }
    }

    /// Wait for a running background compaction and persist embedder state
    /// (graceful shutdown).
    pub fn finish_background_work(&self) {
        if let Some(h) = self.compactor.lock().expect("compactor lock poisoned").take() {
            let _ = h.join();
        }
        if let Some(h) = self.saver.lock().expect("saver lock poisoned").take() {
            let _ = h.join();
        }
        if let Err(e) = self.embedder.persist() {
            tracing::warn!(error = %e, "model save at shutdown failed");
        }
        if let Err(e) = self.embedder.flush() {
            tracing::warn!(error = %e, "embedder flush at shutdown failed");
        }
    }

    /// Save the semantic model on a background thread once its lazy
    /// interval has passed (SPEC-P10 §30). The save holds the model's
    /// write lock for its duration, so a hybrid query in that window waits
    /// for it; every other tool is unaffected, and no tool call carries it.
    fn maybe_save_model(&self) {
        if !self.embedder.save_due() {
            return;
        }
        let mut slot = self.saver.lock().expect("saver lock poisoned");
        if slot.as_ref().is_some_and(|h| !h.is_finished()) {
            return;
        }
        if let Some(prev) = slot.take() {
            let _ = prev.join();
        }
        let embedder = Arc::clone(&self.embedder);
        *slot = Some(std::thread::spawn(move || {
            let t0 = std::time::Instant::now();
            match embedder.persist() {
                Ok(()) => tracing::info!(ms = t0.elapsed().as_millis() as u64, "semantic model saved in the background"),
                Err(e) => tracing::warn!(error = %e, "background model save failed"),
            }
        }));
    }

    /// Fold the oldest shards together in the background once more than
    /// [`COMPACT_ABOVE`] exist (SPEC-P9): every refresh that changed
    /// something adds a delta shard and every lookup probes each shard.
    /// The merge (~20 s on 100 MB of source) never blocks a tool call; the
    /// engine is reopened by the next call after it finishes. Shards this
    /// or another process still has mapped are parked by the merge and
    /// reaped by later opens.
    fn maybe_compact(&self) {
        let data_dir: Option<std::path::PathBuf> = {
            let guard = self.engine.read().expect("engine lock poisoned");
            guard.data_dir().map(|d| d.to_path_buf())
        };
        let Some(data_dir) = data_dir else { return };
        let shards_dir = data_dir.join("shards");
        let model_id = self.embedder.model_id().to_string();
        // a directory listing, not an open of every shard
        let n = shards_stamp(&shards_dir).0;
        let shards_due = n > COMPACT_ABOVE;
        let vectors_due = indexio_embed::pipeline::needs_compaction(&data_dir, &model_id);
        // SPEC-P10 §20: the model has learned a lot since the rows were
        // embedded — rebuild the plane whole (30 s for 218k rows)
        let reembed_due = indexio_embed::pipeline::reembed_due(&data_dir, &model_id, self.embedder.texts_seen());
        if !(shards_due || vectors_due || reembed_due) || self.compacting.swap(true, Ordering::AcqRel) {
            return;
        }
        let compacting = Arc::clone(&self.compacting);
        let reload = Arc::clone(&self.reload_needed);
        let embedder = Arc::clone(&self.embedder);
        let handle = std::thread::spawn(move || {
            let t0 = std::time::Instant::now();
            // one compaction at a time across every server sharing the data dir
            let Some(_lock) = indexio_embed::pipeline::CompactLock::try_acquire(&data_dir) else {
                tracing::debug!("compaction skipped: another process holds the lock");
                compacting.store(false, Ordering::Release);
                return;
            };
            if shards_due {
                match indexio_index::ShardSet::open_dir(&shards_dir) {
                    Ok(mut set) => match set.merge(&shards_dir, COMPACT_TO) {
                        Ok(()) => tracing::info!(from = n, to = COMPACT_TO, ms = t0.elapsed().as_millis() as u64, "shards compacted"),
                        Err(e) => tracing::warn!(error = %e, "shard compaction failed"),
                    },
                    Err(e) => tracing::warn!(error = %e, "shard compaction: cannot open shards"),
                }
            }
            if reembed_due {
                match indexio_index::ShardSet::open_dir(&shards_dir)
                    .map_err(anyhow::Error::from)
                    .and_then(|set| indexio_embed::pipeline::embed_all(&set, &data_dir, embedder.as_ref()))
                {
                    Ok(reps) => tracing::info!(
                        rows = reps.iter().map(|r| r.chunks).sum::<u64>(),
                        ms = t0.elapsed().as_millis() as u64,
                        "semantic plane re-embedded (model drift)"
                    ),
                    Err(e) => tracing::warn!(error = %e, "re-embed failed"),
                }
            } else if vectors_due {
                match indexio_embed::pipeline::compact_vectors(&data_dir, &model_id) {
                    Ok(true) => tracing::info!(ms = t0.elapsed().as_millis() as u64, "vector segments compacted"),
                    Ok(false) => {}
                    Err(e) => tracing::warn!(error = %e, "vector compaction failed"),
                }
            }
            reload.store(true, Ordering::Release);
            compacting.store(false, Ordering::Release);
        });
        let mut slot = self.compactor.lock().expect("compactor lock poisoned");
        if let Some(prev) = slot.take() {
            let _ = prev.join(); // finished already (the flag was clear)
        }
        *slot = Some(handle);
    }

    /// [`maybe_compact`](Self::maybe_compact) once a minute regardless of
    /// who wrote the shards (SPEC-P10 §29): the Bash hook, `indexio sync`,
    /// the background sync and other sessions' servers all add delta
    /// shards, and a server whose own repo is idle never refreshed — the
    /// data dir reached 30 shards with every lookup probing each one.
    fn maybe_compact_periodic(&self) {
        {
            let mut last = self.compact_checked.lock().expect("compact_checked poisoned");
            if last.is_some_and(|t| t.elapsed() < COMPACT_CHECK_EVERY) {
                return;
            }
            *last = Some(std::time::Instant::now());
        }
        self.maybe_compact();
    }

    /// Replace the engine with a fresh open of `data_dir`, handing over the
    /// parsed semantic sidecars so they are reused, not re-parsed.
    fn swap_engine(&self, data_dir: &std::path::Path) -> anyhow::Result<()> {
        let fresh = Engine::open(data_dir)?;
        let mut guard = self.engine.write().expect("engine lock poisoned");
        fresh.inherit_sidecars(&guard);
        *guard = fresh;
        Ok(())
    }

    /// Reopen the engine if a background compaction finished, or if the
    /// shard directory changed under us (SPEC-P9): another session's
    /// auto-refresh, the CLI, the Bash hook or a sync may have written new
    /// shards for repos this session reads. One `read_dir` per call.
    fn reload_if_needed(&self) {
        let dir: Option<std::path::PathBuf> =
            self.engine.read().expect("engine lock poisoned").data_dir().map(|d| d.join("shards"));
        let stamp = dir.as_deref().map(shards_stamp).unwrap_or_default();
        let changed = {
            let mut last = self.shards_seen.lock().expect("shards stamp poisoned");
            if *last != stamp {
                *last = stamp;
                true
            } else {
                false
            }
        };
        if !self.reload_needed.swap(false, Ordering::AcqRel) && !changed {
            return;
        }
        // The read guard must be gone before the write lock is taken: an
        // `if let` scrutinee would keep it alive through the body.
        let dir: Option<std::path::PathBuf> =
            self.engine.read().expect("engine lock poisoned").data_dir().map(|d| d.to_path_buf());
        if let Some(d) = dir {
            if let Err(e) = self.swap_engine(&d) {
                tracing::warn!(error = %e, "reopen after compaction failed");
            }
        }
    }

    /// Open the semantic sidecars on a background thread so the first
    /// hybrid query of a session does not pay the parse (~100-200 ms on a
    /// 1.6 GB plane). The warmed engine is adopted by the next tool call.
    pub fn prewarm_sidecars(&self) {
        let Some(dir) = self.engine.read().expect("engine lock poisoned").data_dir().map(|d| d.to_path_buf()) else {
            return;
        };
        let model = self.embedder.model_id().to_string();
        let slot = Arc::clone(&self.warm);
        std::thread::spawn(move || {
            if let Ok(e) = Engine::open(&dir) {
                let _ = e.vec_index(&model);
                let _ = e.bm25_index(&model);
                *slot.lock().expect("warm slot poisoned") = Some(e);
            }
        });
    }

    /// Append one JSON line per tool call to `<data_dir>/usage/<date>.jsonl`
    /// (SPEC-P9 telemetry): what the sessions sharing this index actually
    /// ask for, how big the answers are and how long they take. `indexio
    /// usage` aggregates it. Never fails a call.
    /// One usage line per call: what was returned, how long it took and,
    /// when the call has a mechanical built-in equivalent, what that would
    /// have returned (SPEC-P10 §22; `indexio usage`, `index_stats`).
    fn log_usage(&self, tool: &str, ms: u64, bytes: usize, builtin: Option<u64>, ok: bool) {
        // benchmarks and sweeps set INDEXIO_NO_USAGE so the stats stay real traffic
        if std::env::var_os("INDEXIO_NO_USAGE").is_some() {
            return;
        }
        let Some(dir) = self.engine.read().expect("engine lock poisoned").data_dir().map(|d| d.join("usage")) else {
            return;
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let day = indexio_ingest::unix_to_iso_date(now);
        let line = json!({
            "ts": now, "pid": std::process::id(), "repo": self.current(),
            "tool": tool, "ms": ms, "bytes": bytes, "builtin": builtin, "ok": ok,
        });
        let _ = std::fs::create_dir_all(&dir);
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(format!("{day}.jsonl"))) {
            use std::io::Write as _;
            let _ = writeln!(f, "{line}");
        }
    }

    /// Take over the pre-warmed sidecars if the background open finished.
    fn adopt_warm(&self) {
        if let Some(w) = self.warm.lock().expect("warm slot poisoned").take() {
            self.engine.read().expect("engine lock poisoned").inherit_sidecars(&w);
        }
    }

    /// Set the session's repo (see [`resolve_current_repo`]) and start
    /// watching its folder for the auto-refresh.
    pub fn with_current_repo(self, repo: Option<String>) -> Self {
        self.set_current_repo(repo);
        self
    }

    /// The session's repo, if any (a short clone; the lock is never held
    /// across a call).
    pub fn current(&self) -> Option<String> {
        self.current_repo.read().expect("current_repo poisoned").clone()
    }

    fn set_current_repo(&self, repo: Option<String>) {
        let mut watch = None;
        if let Some(name) = &repo {
            let root = self
                .engine
                .read()
                .expect("engine lock poisoned")
                .data_dir()
                .and_then(|d| indexio_ingest::repo_state(d, name).ok())
                .map(|st| st.path);
            if let Some(root) = root {
                match crate::watch::RepoWatch::start(&root) {
                    Ok(w) => watch = Some(w),
                    Err(e) => tracing::warn!(error = %e, root = %root.display(), "cannot watch repo; auto-refresh off"),
                }
            }
        }
        *self.watch.lock().expect("watch poisoned") = watch;
        *self.current_repo.write().expect("current_repo poisoned") = repo;
    }

    /// A server that started outside any registered repo looks again, at
    /// most once a minute (SPEC-P10 §23): the folder is often registered
    /// (`indexio add`) after the session that works in it has started,
    /// and until then that session gets no auto-refresh, no repo-first
    /// ranking and no per-repo usage. Costs one `repos/` listing per
    /// minute while unresolved, nothing once resolved.
    fn maybe_adopt_repo(&self) {
        if self.current_repo.read().expect("current_repo poisoned").is_some() {
            return;
        }
        {
            let mut last = self.repo_checked.lock().expect("repo_checked poisoned");
            if last.map_or(false, |t| t.elapsed() < ADOPT_EVERY) {
                return;
            }
            *last = Some(std::time::Instant::now());
        }
        let Ok(cwd) = std::env::current_dir() else { return };
        self.adopt_repo_at(&cwd);
    }

    /// [`maybe_adopt_repo`](Self::maybe_adopt_repo) for an explicit
    /// working directory.
    fn adopt_repo_at(&self, cwd: &std::path::Path) {
        let data_dir: Option<std::path::PathBuf> =
            self.engine.read().expect("engine lock poisoned").data_dir().map(|d| d.to_path_buf());
        let Some(data_dir) = data_dir else { return };
        if let Some(name) = repo_containing(&data_dir, cwd) {
            tracing::info!(repo = %name, "working directory is now under a registered repo; adopting it");
            self.set_current_repo(Some(name));
        }
    }

    /// Re-index the current repo's working tree if the watcher saw a
    /// change since the last call (SPEC-P9 auto-refresh): stat-cached
    /// delta, lexical plane only, engine reopened when docs moved. Never
    /// fails the caller's tool call.
    fn auto_refresh(&self) {
        let Some(repo) = self.current() else { return };
        let repo = repo.as_str();
        let watch = self.watch.lock().expect("watch poisoned");
        let Some(w) = watch.as_ref() else { return };
        let data_dir: Option<std::path::PathBuf> = {
            let guard = self.engine.read().expect("engine lock poisoned");
            guard.data_dir().map(|d| d.to_path_buf())
        };
        let Some(data_dir) = data_dir else { return };
        if !w.take_dirty() {
            // nothing new on disk, but a deferred embed may be waiting
            let pending = !self.pending_embed.lock().expect("pending embed poisoned").is_empty();
            if pending {
                self.embed_current_paths(repo, &data_dir, &[]);
                if let Err(e) = self.swap_engine(&data_dir) {
                    tracing::warn!(error = %e, "auto-refresh: reopen failed");
                }
            }
            return;
        }
        let t0 = std::time::Instant::now();
        let cas = match indexio_ingest::Cas::open(&data_dir.join("cas")) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "auto-refresh: cannot open cas");
                return;
            }
        };
        let mut cache = self.wt_cache.lock().expect("worktree cache poisoned");
        match indexio_ingest::reindex_worktree_cached(repo, &data_dir, &cas, Some(&mut cache)) {
            Ok(r) if r.docs_added > 0 || r.docs_deleted > 0 => {
                // Semantic plane too (SPEC-P9 incremental embed): only the
                // changed paths are chunked, diffed and appended.
                let t_embed = std::time::Instant::now();
                self.embed_current_paths(repo, &data_dir, &r.changed_paths);
                let embed_ms = t_embed.elapsed().as_millis() as u64;
                if let Err(e) = self.swap_engine(&data_dir) {
                    tracing::warn!(error = %e, "auto-refresh: reopen failed");
                }
                drop(cache);
                drop(watch);
                self.maybe_compact();
                tracing::debug!(embed_ms, "auto-refresh: semantic plane updated");
                tracing::debug!(
                    repo,
                    added = r.docs_added,
                    deleted = r.docs_deleted,
                    ms = t0.elapsed().as_millis() as u64,
                    "auto-refresh"
                );
            }
            Ok(_) => {}
            Err(e) => tracing::warn!(error = %e, "auto-refresh failed"),
        }
    }

    /// Handle one input line; `None` means "no response" (notifications,
    /// empty lines, all-notification batches).
    pub fn handle(&self, line: &str) -> Option<String> {
        let line = line.trim();
        if line.is_empty() {
            return None;
        }
        let msg: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                return Some(
                    error_response(Value::Null, PARSE_ERROR, &format!("parse error: {e}"))
                        .to_string(),
                )
            }
        };
        if let Value::Array(batch) = msg {
            let responses: Vec<Value> = batch.iter().filter_map(|m| self.handle_one(m)).collect();
            if responses.is_empty() {
                None
            } else {
                Some(Value::Array(responses).to_string())
            }
        } else {
            self.handle_one(&msg).map(|r| r.to_string())
        }
    }

    /// Handle one JSON-RPC message object. `None` = notification (no reply).
    fn handle_one(&self, msg: &Value) -> Option<Value> {
        let id = msg.get("id").cloned();
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let is_notification = id.is_none();
        let reply = |result: Value| {
            if is_notification {
                None
            } else {
                Some(result_response(id.clone().unwrap_or(Value::Null), result))
            }
        };
        let reply_err = |code: i64, message: &str| {
            if is_notification {
                None
            } else {
                Some(error_response(id.clone().unwrap_or(Value::Null), code, message))
            }
        };

        match method {
            "notifications/initialized" => None,
            "initialize" => {
                let engine = self.engine.read().expect("engine lock poisoned");
                reply(json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "indexio", "version": "0.1.0" },
                    "instructions": server_instructions(&engine, self.current().as_deref()),
                }))
            }
            "ping" => reply(json!({})),
            "tools/list" => reply(tools_list()),
            "tools/call" => {
                let params = msg.get("params").cloned().unwrap_or_else(|| json!({}));
                let t0 = std::time::Instant::now();
                let tool = params.get("name").and_then(Value::as_str).unwrap_or("?").to_string();
                match self.call_tool(&params) {
                    Ok((mut result, builtin)) => {
                        if self.binary_updated_once() {
                            if let Some(t) = result["content"][0]["text"].as_str() {
                                let noted = format!("{t}\n\n[indexio: a newer indexio binary was installed since this session's server started; restart the session to use it]");
                                result["content"][0]["text"] = json!(noted);
                            }
                        }
                        let bytes = result["content"]
                            .as_array()
                            .map(|c| c.iter().filter_map(|x| x["text"].as_str()).map(str::len).sum())
                            .unwrap_or(0);
                        self.log_usage(&tool, t0.elapsed().as_millis() as u64, bytes, builtin, true);
                        reply(result)
                    }
                    Err((code, message)) => {
                        self.log_usage(&tool, t0.elapsed().as_millis() as u64, message.len(), None, false);
                        reply_err(code, &message)
                    }
                }
            }
            _ => reply_err(METHOD_NOT_FOUND, &format!("method not found: {method}")),
        }
    }

    /// The tool result and, when the call has one, the size of the
    /// built-in tool's output for it (SPEC-P10 §22).
    fn call_tool(&self, params: &Value) -> Result<(Value, Option<u64>), (i64, String)> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("missing 'name' parameter"))?;
        let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
        let format = opt_format(&args, self.format)?;
        if name == "refresh_index" {
            return self.refresh_index(&args, format).map(|v| (v, None));
        }
        self.adopt_warm();
        self.reload_if_needed();
        self.maybe_adopt_repo();
        self.auto_refresh();
        self.maybe_compact_periodic();
        self.maybe_save_model();
        self.maybe_import_sessions();
        self.maybe_sync_others();
        let engine = self.engine.read().expect("engine lock poisoned");
        let current = self.current();
        let mut baseline = None;
        let v = call_tool(
            &engine,
            self.embedder.as_ref(),
            self.reranker.as_ref(),
            name,
            &args,
            format,
            current.as_deref(),
            &self.read_seen,
            &mut baseline,
        )?;
        Ok((v, baseline))
    }

    /// Delta re-index every registered repo (or one), embed the repos that
    /// changed, then reopen the engine so the new shards are visible.
    fn refresh_index(&self, args: &Value, format: OutputFormat) -> Result<Value, (i64, String)> {
        let data_dir = self
            .engine
            .read()
            .expect("engine lock poisoned")
            .data_dir()
            .map(|p| p.to_path_buf())
            .ok_or_else(|| internal("engine has no data dir"))?;
        let only: Option<String> = match args.get("repo") {
            None | Some(Value::Null) => None,
            Some(v) => Some(
                v.as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| invalid("'repo' must be a string"))?
                    .to_string(),
            ),
        };
        let embed = match args.get("embed") {
            None | Some(Value::Null) => true,
            Some(Value::Bool(b)) => *b,
            Some(_) => return Err(invalid("'embed' must be a boolean")),
        };
        // SPEC-P9: an agent refreshes to see what it is editing, so the
        // working tree is the default; worktree=false re-indexes HEAD.
        let worktree = match args.get("worktree") {
            None | Some(Value::Null) => true,
            Some(Value::Bool(b)) => *b,
            Some(_) => return Err(invalid("'worktree' must be a boolean")),
        };
        let names = match &only {
            Some(n) => vec![n.clone()],
            None => indexio_ingest::sources::registered_repo_names(&data_dir)
                .map_err(|e| internal(format!("{e:#}")))?,
        };
        let cas = indexio_ingest::Cas::open(&data_dir.join("cas"))
            .map_err(|e| internal(format!("{e:#}")))?;
        let mut repos = Vec::new();
        let mut changed: Vec<String> = Vec::new();
        let mut failed = Vec::new();
        for n in &names {
            let res = if worktree {
                if self.current().as_deref() == Some(n.as_str()) {
                    if let Some(w) = self.watch.lock().expect("watch poisoned").as_ref() {
                        w.take_dirty();
                    }
                    let mut cache = self.wt_cache.lock().expect("worktree cache poisoned");
                    indexio_ingest::reindex_worktree_cached(n, &data_dir, &cas, Some(&mut cache))
                } else {
                    indexio_ingest::reindex_worktree(n, &data_dir, &cas)
                }
            } else {
                indexio_ingest::reindex_repo(n, &data_dir, &cas)
            };
            match res {
                Ok(r) => {
                    if r.docs_added > 0 || r.docs_deleted > 0 {
                        changed.push(n.clone());
                    }
                    repos.push(json!({
                        "repo": r.repo, "docs_added": r.docs_added, "docs_deleted": r.docs_deleted,
                        "docs_unchanged": r.docs_unchanged, "elapsed_ms": r.elapsed_ms,
                    }));
                }
                Err(e) => failed.push(json!({ "repo": n, "error": format!("{e:#}") })),
            }
        }
        let mut embedded = 0u64;
        if embed && !changed.is_empty() {
            let set = indexio_index::ShardSet::open_dir(&data_dir.join("shards"))
                .map_err(|e| internal(format!("{e:#}")))?;
            let reports = indexio_embed::pipeline::embed_repos_with(
                &set,
                &data_dir,
                &changed,
                self.embedder.as_ref(),
                indexio_embed::pipeline::MAX_CHARS,
            )
            .map_err(|e| internal(format!("embed failed: {e:#}")))?;
            embedded = reports.iter().map(|r| r.embedded).sum();
        }
        self.swap_engine(&data_dir).map_err(|e| internal(format!("reopen failed: {e}")))?;
        self.maybe_compact();
        let compacting = self.compacting.load(Ordering::Acquire);
        if format == OutputFormat::Text {
            let rows: Vec<(String, u64, u64, u64, u64)> = repos
                .iter()
                .map(|r| {
                    (
                        r["repo"].as_str().unwrap_or("").to_string(),
                        r["docs_added"].as_u64().unwrap_or(0),
                        r["docs_deleted"].as_u64().unwrap_or(0),
                        r["docs_unchanged"].as_u64().unwrap_or(0),
                        r["elapsed_ms"].as_u64().unwrap_or(0),
                    )
                })
                .collect();
            let fails: Vec<(String, String)> = failed
                .iter()
                .map(|f| {
                    (
                        f["repo"].as_str().unwrap_or("").to_string(),
                        f["error"].as_str().unwrap_or("").to_string(),
                    )
                })
                .collect();
            let mut s = mcp_text::refresh(&rows, &fails, embedded);
            if compacting {
                s.push_str(", shard compaction running in the background");
            }
            return Ok(plain_result(s));
        }
        Ok(text_result(json!({
            "repos": repos, "failed": failed, "changed_repos": changed,
            "embedded_chunks": embedded, "reloaded": true, "compacting": compacting,
        })))
    }
}

/// Single-call entry point for tests: builds a server around a fresh
/// engine on the same data dir.
#[cfg(test)]
pub fn handle(engine: &Engine, embedder: &dyn Embedder, reranker: &dyn Reranker, line: &str) -> Option<String> {
    let _ = (embedder, reranker);
    let dir = engine.data_dir().expect("engine opened from a data dir");
    let server = McpServer::with_format(
        Engine::open(dir).expect("reopen"),
        Arc::new(indexio_embed::embed::HashEmbedder::new(512)),
        Arc::new(indexio_embed::rerank::OverlapReranker),
        OutputFormat::Json,
    );
    server.handle(line)
}

/// Same as [`handle`] but rendering the default text format.
#[cfg(test)]
pub fn handle_text(engine: &Engine, line: &str) -> Option<String> {
    let dir = engine.data_dir().expect("engine opened from a data dir");
    let server = McpServer::with_format(
        Engine::open(dir).expect("reopen"),
        Arc::new(indexio_embed::embed::HashEmbedder::new(512)),
        Arc::new(indexio_embed::rerank::OverlapReranker),
        OutputFormat::Text,
    );
    server.handle(line)
}

/// The `initialize` instructions (SPEC-P8): what is indexed, how fresh it
/// is, and when to prefer these tools over the harness's own. Kept short —
/// harnesses inject it into every system prompt.
pub fn server_instructions(engine: &Engine, current: Option<&str>) -> String {
    let stats = engine.stats();
    // Names only: paths and sync times are one index_stats call away, and
    // this string is paid for in every prompt of every session.
    let mut repo_lines: Vec<String> = stats.repos.clone();
    repo_lines.sort();
    let shown: Vec<String> = repo_lines.iter().take(80).cloned().collect();
    let more = if repo_lines.len() > 80 {
        format!(" … and {} more (see index_stats)", repo_lines.len() - 80)
    } else {
        String::new()
    };
    let here = match current {
        Some(r) => format!(
            " You are in repo \"{r}\": its hits come first (pass repo: to look elsewhere) \
             and its working tree — uncommitted and untracked edits included — is re-indexed \
             automatically before each call, so never call refresh_index for it."
        ),
        None => String::new(),
    };
    format!(
        "indexio is a pre-built index of {} repo(s), {} files: lexical + symbols + call \
         graph + semantic search over source and every other text file (docs, configs, \
         scripts, SQL, markup). Indexed repos: {}{}. Paths are repo-relative.{here}\n\
         PREFER these tools over grep/rg/Glob/find/Read/cat/sed for anything in an indexed \
         repo: code_search (\"hybrid\" for questions, \"lexical\" for identifiers/regex) or \
         code_grep instead of grep; find_symbol/who_calls for definitions and call sites; \
         list_files instead of Glob; file_outline then read_span instead of reading whole \
         files; impact_of_symbol BEFORE changing a function/type, impact_of_diff AFTER \
         editing. recall searches past conversation (this and other sessions) instead of \
         re-deriving it.\n\
         Other repos reflect their last sync/refresh: call refresh_index {{repo}} for them \
         when results look stale. Local tools only for repos not listed above.",
        stats.repos.len(),
        stats.doc_count,
        shown.join(", "),
        more
    )
}

fn result_response(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    })
}

fn tools_list() -> Value {
    let string_prop = |desc: &str| json!({ "type": "string", "description": desc });
    let limit_prop = json!({ "type": "integer", "description": "max results (default 50)", "minimum": 1 });
    let tool = |name: &str, desc: &str, props: Value, required: &[&str]| {
        json!({
            "name": name,
            "description": desc,
            "inputSchema": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        })
    };
    json!({
        "tools": [
            tool(
                "code_search",
                "USE THIS INSTEAD OF grep/rg: ranked search over every indexed repo. mode lexical \
                 (default): exact identifiers, \"phrases\", /regex/, repo:/lang:/path: filters. mode \
                 hybrid: natural-language questions (exact + BM25 + semantic; use when unsure of \
                 the identifier). mode semantic: concepts only. Hits are `repo:path` + `line: \
                 snippet`; follow up with read_span. lines=N (lexical) lists up to N matching lines \
                 per file like `grep -n`.",
                json!({
                    "query": string_prop("search query"),
                    "limit": limit_prop,
                    "lines": json!({ "type": "integer", "description": "lexical: matching lines per file (default 1, max 100)", "minimum": 1 }),
                    "mode": json!({ "type": "string", "enum": ["lexical", "semantic", "hybrid"] }),
                    "rerank": json!({ "type": "boolean", "description": "hybrid only (default false); needs a configured rerank endpoint, otherwise no effect" }),
                    "fusion": json!({ "type": "string", "enum": ["rrf", "combmnz"], "description": "hybrid only (default rrf)" }),
                }),
                &["query"],
            ),
            tool(
                "semantic_search",
                "Pure vector search over code chunks for natural-language concept queries \
                 when the identifier is unknown. For identifiers or regex use code_search.",
                json!({
                    "query": string_prop("natural-language query"),
                    "k": json!({ "type": "integer", "description": "nearest chunks (default 10)", "minimum": 1 }),
                }),
                &["query"],
            ),
            tool(
                "code_grep",
                "`grep -n` over every indexed repo instead of grep/rg: up to `lines` matching lines \
                 per file (default 20), files ranked. Regex; append ` repo:X path:Y` filters to the \
                 pattern to narrow.",
                json!({
                    "pattern": string_prop("regex (may end with repo:/path:/lang: filters)"),
                    "limit": limit_prop,
                    "lines": json!({ "type": "integer", "description": "matching lines per file (default 20, max 100)", "minimum": 1 }),
                }),
                &["pattern"],
            ),
            tool(
                "list_files",
                "USE THIS INSTEAD OF Glob/find/ls: indexed paths matching a glob (`**/*.rs`, \
                 `src/**/handler*`) or a substring, optionally in one repo; folded per directory.",
                json!({
                    "pattern": string_prop("glob (* ? **) or substring, case-insensitive; omitted = every file"),
                    "repo": string_prop("only this repo (optional)"),
                    "limit": json!({ "type": "integer", "minimum": 1, "description": "max paths (default 200)" }),
                }),
                &[],
            ),
            tool(
                "find_symbol",
                "Where a function/method/class/struct/trait/enum is DEFINED (exact name, substring \
                 fallback), instead of grepping for `fn name` / `def name` / `class name`. Rows \
                 read `start-end: signature`; read_span with that start returns the definition. \
                 Your repo first; a common name lists your repo and counts the others (pass repo: \
                 to list one).",
                json!({ "name": string_prop("symbol name"), "repo": string_prop("only this repo (substring)") }),
                &["name"],
            ),
            tool(
                "who_calls",
                "Call sites of a function/method (name-based call graph; qualifiers that cannot be \
                 a repo definition, such as `tokio::spawn`, are left out), instead of grepping for \
                 `name(`. Transitive blast radius: impact_of_symbol.",
                json!({ "name": string_prop("callee name"), "repo": string_prop("only this repo (substring)") }),
                &["name"],
            ),
            tool(
                "index_stats",
                "What is indexed and how fresh: repos grouped by sync date (folders that \
                 differ from root/name spelled out), file/shard counts, last week's usage.",
                json!({}),
                &[],
            ),
            tool(
                "recall",
                "Search past conversation and command output: every Claude Code session on this \
                 machine (this one included, minutes old) is indexed: requests, answers, tool calls, \
                 the first lines of results, compaction summaries; so is every long command output \
                 kept by `indexio run` (builds, tests, scripts). USE THIS INSTEAD OF re-deriving or \
                 asking again: earlier decisions, findings, commands that worked, what a build said, \
                 what another session did here; especially after a context compaction. Hits are \
                 `sessions:<project>/<session>.md` or `runs:<repo>/<log>` + `line: text`; a lone \
                 hit comes with its exchange, else read_span for it.",
                json!({
                    "query": string_prop("what to look for (natural language or exact words)"),
                    "limit": json!({ "type": "integer", "minimum": 1, "description": "max hits (default 8)" }),
                    "all_projects": json!({ "type": "boolean", "description": "every project's sessions (default: this project first)" }),
                }),
                &["query"],
            ),
            tool(
                "refresh_index",
                "Delta re-index a repo (or all) from its WORKING TREE (uncommitted edits and new \
                 files included), embed what changed, reload. Your own repo refreshes itself; call \
                 this for other repos or when results look stale.",
                json!({
                    "repo": string_prop("only this repo (optional; default all)"),
                    "embed": json!({ "type": "boolean", "description": "also embed changed repos (default true)" }),
                    "worktree": json!({ "type": "boolean", "description": "index the working tree (default true); false = committed HEAD only" }),
                }),
                &[],
            ),
            tool(
                "impact_of_symbol",
                "Blast radius of changing symbols: reverse call graph across ALL repos (depth 1 = \
                 direct callers, 2 = their callers). Definitions, then call sites per file with the \
                 enclosing caller. Use BEFORE editing a function/method/type.",
                json!({
                    "names": json!({ "type": "array", "items": { "type": "string" }, "description": "symbol names to start from" }),
                    "depth": json!({ "type": "integer", "minimum": 1, "description": "reverse-call hops (default 2)" }),
                    "max_sites": json!({ "type": "integer", "minimum": 1, "description": "stop after this many call sites (default 500)" }),
                }),
                &["names"],
            ),
            tool(
                "impact_of_diff",
                "Impact of a change: maps a unified diff to the definitions it touches, walks their \
                 callers across all repos, lists importers of the changed files. Pass `diff`, or \
                 omit it to diff the repo's working tree against `base` (HEAD). Use AFTER editing \
                 to find what to update.",
                json!({
                    "repo": string_prop("registered repo name"),
                    "diff": string_prop("unified diff text (optional: omit to diff the working tree)"),
                    "base": string_prop("git ref to diff against (default HEAD)"),
                    "depth": json!({ "type": "integer", "minimum": 1, "description": "reverse-call hops (default 2)" }),
                    "max_sites": json!({ "type": "integer", "minimum": 1, "description": "stop after this many call sites (default 500)" }),
                }),
                &["repo"],
            ),
            tool(
                "file_outline",
                "USE THIS BEFORE READING A FILE: every definition with kind, scope and start-end \
                 lines in a few hundred tokens; then read_span only the lines you need.",
                json!({
                    "repo": string_prop("registered repo name"),
                    "path": string_prop("repo-relative file path"),
                }),
                &["repo", "path"],
            ),
            tool(
                "read_span",
                "USE THIS INSTEAD OF reading a whole file: lines [start, end] (max 400) from the \
                 index. Without `end`: the whole definition containing `start`, else 60 lines. Rows \
                 are `N<TAB>text`; N appears on the first, the last and every 5th line (other rows \
                 start with the tab), count from the nearest one.",
                json!({
                    "repo": string_prop("registered repo name"),
                    "path": string_prop("repo-relative file path"),
                    "start": json!({ "type": "integer", "minimum": 1, "description": "first line (1-based; default 1)" }),
                    "end": json!({ "type": "integer", "minimum": 1, "description": "last line inclusive (default: end of the definition at start, else start+59)" }),
                }),
                &["repo", "path"],
            ),
        ]
    })
}

/// MCP tool result envelope: compact JSON as a single text block (SPEC-P6
/// §3: compact, not pretty — the consumer is a model, and whitespace is
/// paid for in tokens).
fn text_result(payload: Value) -> Value {
    let text = serde_json::to_string(&payload).unwrap_or_else(|_| payload.to_string());
    json!({ "content": [ { "type": "text", "text": text } ] })
}

/// MCP tool result envelope for an already-rendered text block.
fn plain_result(text: String) -> Value {
    json!({ "content": [ { "type": "text", "text": text } ] })
}

/// Per-call `format` override ("text" | "json"); default from the server.
fn opt_format(args: &Value, default: OutputFormat) -> Result<OutputFormat, (i64, String)> {
    match args.get("format") {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_str()
            .and_then(OutputFormat::parse)
            .ok_or_else(|| invalid("'format' must be \"text\" or \"json\"")),
    }
}

fn invalid(msg: impl Into<String>) -> (i64, String) {
    (INVALID_PARAMS, msg.into())
}

/// Fallible internals (semantic plane embed/index I/O) map to -32603.
fn internal(msg: impl Into<String>) -> (i64, String) {
    (INTERNAL_ERROR, msg.into())
}

fn req_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, (i64, String)> {
    args.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid(format!("missing or invalid '{key}' parameter")))
}

fn opt_limit(args: &Value) -> Result<usize, (i64, String)> {
    match args.get("limit") {
        None | Some(Value::Null) => Ok(DEFAULT_LIMIT),
        Some(v) => v
            .as_u64()
            .filter(|&n| n >= 1)
            .map(|n| n as usize)
            .ok_or_else(|| invalid("'limit' must be a positive integer")),
    }
}

fn opt_usize(args: &Value, key: &str, default: usize) -> Result<usize, (i64, String)> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_u64()
            .filter(|&n| n >= 1)
            .map(|n| n as usize)
            .ok_or_else(|| invalid(format!("'{key}' must be a positive integer"))),
    }
}

/// index_stats payload: EngineStats plus per-repo freshness (SPEC-P8).
fn stats_payload(engine: &Engine) -> Value {
    let stats = engine.stats();
    let mut v = serde_json::to_value(&stats).expect("EngineStats is Serialize");
    if let Some(dir) = engine.data_dir() {
        let detail: Vec<Value> = stats
            .repos
            .iter()
            .map(|name| match indexio_ingest::repo_state(dir, name) {
                Ok(st) => json!({
                    "name": name, "path": st.path.display().to_string(),
                    "commit": st.last_commit, "indexed_at": st.indexed_at, "plain": st.plain,
                }),
                Err(_) => json!({ "name": name }),
            })
            .collect();
        v["repos_detail"] = Value::Array(detail);
    }
    v
}

/// index_stats as text (SPEC-P9).
fn stats_text(engine: &Engine) -> String {
    let stats = engine.stats();
    let mut detail = Vec::new();
    if let Some(dir) = engine.data_dir() {
        for name in &stats.repos {
            if let Ok(st) = indexio_ingest::repo_state(dir, name) {
                detail.push((
                    name.clone(),
                    st.path.display().to_string(),
                    st.last_commit.clone().unwrap_or_default(),
                    st.indexed_at.clone(),
                    st.plain,
                ));
            }
        }
    }
    mcp_text::stats(&stats, &detail)
}

/// `baseline` receives the size of the harness's own tool's output for
/// the same call where one exists (SPEC-P10 §22, see `usage`).
#[allow(clippy::too_many_arguments)]
fn call_tool(
    engine: &Engine,
    embedder: &dyn Embedder,
    reranker: &dyn Reranker,
    name: &str,
    args: &Value,
    format: OutputFormat,
    current: Option<&str>,
    read_seen: &ReadSeen,
    baseline: &mut Option<u64>,
) -> Result<Value, (i64, String)> {
    let text = format == OutputFormat::Text;
    // Over-fetch so the session's own repo is never cut off by `limit`
    // before it is moved to the front (the engine collects every hit and
    // only truncates at the end, so this costs nothing extra).
    let fetch = |limit: usize| if current.is_some() { limit.saturating_mul(8).max(400) } else { limit };
    match name {
        "code_search" => {
            let q = req_str(args, "query")?;
            let limit = opt_limit(args)?;
            let mode = match args.get("mode") {
                None | Some(Value::Null) => SearchMode::Lexical,
                Some(v) => {
                    let m = v
                        .as_str()
                        .ok_or_else(|| invalid("'mode' must be a string"))?;
                    SearchMode::parse(m).map_err(invalid)?
                }
            };
            let rerank = match args.get("rerank") {
                None | Some(Value::Null) => false,
                Some(Value::Bool(b)) => *b,
                Some(_) => return Err(invalid("'rerank' must be a boolean")),
            };
            if rerank && mode != SearchMode::Hybrid {
                return Err(invalid("rerank=true requires mode=hybrid"));
            }
            let fusion = match args.get("fusion") {
                None | Some(Value::Null) => FusionAlgo::Rrf,
                Some(v) => {
                    let f = v
                        .as_str()
                        .ok_or_else(|| invalid("'fusion' must be a string"))?;
                    FusionAlgo::parse(f).map_err(invalid)?
                }
            };
            if fusion != FusionAlgo::Rrf && mode != SearchMode::Hybrid {
                return Err(invalid("fusion requires mode=hybrid"));
            }
            let lines = opt_usize(args, "lines", 1)?.clamp(1, MAX_LINES_PER_FILE);
            match mode {
                SearchMode::Lexical => {
                    let query =
                        indexio_query::parse(q).map_err(|e| invalid(format!("invalid query: {e}")))?;
                    let mut res = engine.search_lines(&query, fetch(limit), lines);
                    // SPEC-P10 §27: `a|b|c`, `width: 3[5-9]` — a grep-style
                    // pattern handed to lexical mode as if it were code_grep;
                    // as literal text it matches nothing, so run it as the
                    // regex it is
                    let mut as_regex = false;
                    if res.hits.is_empty() && query.regexes.is_empty() && looks_like_regex(q) {
                        if let Ok(rq) = grep_query(q) {
                            let r2 = engine.search_lines(&rq, fetch(limit), lines);
                            if !r2.hits.is_empty() {
                                res = r2;
                                as_regex = true;
                            }
                        }
                    }
                    *baseline = Some(grep_baseline(&res.hits, current));
                    if wants_sessions(q) {
                        res.hits.truncate(limit);
                    } else {
                        prefer_repo(&mut res.hits, current, limit);
                    }
                    if as_regex {
                        if text {
                            let mut out = String::from("no literal match; treated as a regex:\n");
                            out.push_str(&mcp_text::hits(&res.hits, true, res.truncated, "hits"));
                            return Ok(plain_result(out));
                        }
                        let mut v = crate::serve::search_result_to_json(&res);
                        v["fallback"] = json!("regex");
                        return Ok(text_result(v));
                    }
                    // SPEC-P10 §25: several words and nothing matches them
                    // all — the model meant "any of these"; answer with the
                    // hybrid ranking instead of an empty result it retries
                    if res.hits.is_empty()
                        && query.literals.len() >= 2
                        && query.regexes.is_empty()
                        && !wants_sessions(q)
                    {
                        let scope = repo_scope(engine, q);
                        if let Ok(mut fused) = engine.search_hybrid_in(q, limit.saturating_mul(2), embedder, FusionAlgo::Rrf, scope.as_deref()) {
                            fused.retain(|h| h.hit.repo != crate::sessions::REPO);
                            if let Some(cur) = current {
                                let (mut here, rest): (Vec<_>, Vec<_>) = fused.into_iter().partition(|h| h.hit.repo == cur);
                                here.extend(rest);
                                fused = here;
                            }
                            fused.truncate(limit);
                            if !fused.is_empty() {
                                // a hybrid answer has no built-in equivalent
                                // (the literal grep it replaces was empty)
                                *baseline = None;
                                if text {
                                    let mut out = String::from("no line has every word; ranked by any of them (hybrid):\n");
                                    out.push_str(&mcp_text::hybrid_hits(&fused, "hits"));
                                    return Ok(plain_result(out));
                                }
                                return Ok(text_result(json!({
                                    "fallback": "hybrid",
                                    "hits": fused.iter().map(crate::serve::hybrid_hit_to_json).collect::<Vec<_>>(),
                                })));
                            }
                        }
                    }
                    if text {
                        return Ok(plain_result(mcp_text::hits(&res.hits, true, res.truncated, "hits")));
                    }
                    Ok(text_result(crate::serve::search_result_to_json(&res)))
                }
                SearchMode::Semantic => {
                    let scope = repo_scope(engine, q);
                    let mut hits = engine
                        .search_semantic_in(q, fetch(limit), embedder, scope.as_deref())
                        .map_err(|e| internal(format!("semantic search failed: {e}")))?;
                    prefer_repo(&mut hits, current, limit);
                    if text {
                        return Ok(plain_result(mcp_text::hits(&hits, false, false, "hits")));
                    }
                    Ok(text_result(
                        json!({ "hits": hits.iter().map(crate::serve::hit_to_json).collect::<Vec<_>>() }),
                    ))
                }
                SearchMode::Hybrid => {
                    let n = if current.is_some() { limit.saturating_mul(3) } else { limit.saturating_mul(2) };
                    let scope = repo_scope(engine, q);
                    let mut fused = if rerank {
                        engine
                            .search_hybrid_fused_reranked(q, n, embedder, reranker, fusion)
                            .map_err(|e| internal(format!("hybrid rerank failed: {e}")))?
                    } else {
                        engine
                            .search_hybrid_in(q, n, embedder, fusion, scope.as_deref())
                            .map_err(|e| internal(format!("hybrid search failed: {e}")))?
                    };
                    if !wants_sessions(q) {
                        fused.retain(|h| h.hit.repo != crate::sessions::REPO);
                    }
                    if let Some(cur) = current {
                        // stable: the fused order is kept inside each half
                        let (mut here, rest): (Vec<_>, Vec<_>) =
                            fused.into_iter().partition(|h| h.hit.repo == cur);
                        here.extend(rest);
                        fused = here;
                    }
                    fused.truncate(limit);
                    if text {
                        return Ok(plain_result(mcp_text::hybrid_hits(&fused, "hits")));
                    }
                    Ok(text_result(
                        json!({ "hits": fused.iter().map(crate::serve::hybrid_hit_to_json).collect::<Vec<_>>() }),
                    ))
                }
            }
        }
        "semantic_search" => {
            let q = req_str(args, "query")?;
            let k = opt_usize(args, "k", 10)?;
            let mut hits = engine
                .search_semantic(q, fetch(k), embedder)
                .map_err(|e| internal(format!("semantic search failed: {e}")))?;
            prefer_repo(&mut hits, current, k);
            if text {
                return Ok(plain_result(mcp_text::hits(&hits, false, false, "hits")));
            }
            Ok(text_result(
                json!({ "hits": hits.iter().map(crate::serve::hit_to_json).collect::<Vec<_>>() }),
            ))
        }
        "code_grep" => {
            let pattern = req_str(args, "pattern")?;
            let limit = opt_limit(args)?;
            let lines = opt_usize(args, "lines", GREP_LINES_PER_FILE)?.clamp(1, MAX_LINES_PER_FILE);
            let query = grep_query(pattern).map_err(|e| invalid(format!("invalid pattern: {e}")))?;
            let mut res = engine.search_lines(&query, fetch(limit), lines);
            *baseline = Some(grep_baseline(&res.hits, current));
            prefer_repo(&mut res.hits, current, limit);
            if text {
                return Ok(plain_result(mcp_text::hits(&res.hits, true, res.truncated, "matches")));
            }
            Ok(text_result(crate::serve::search_result_to_json(&res)))
        }
        "list_files" => {
            // no pattern (a worker sent `{}`) lists the session's repo, not an error
            let pattern = match args.get("pattern") {
                None | Some(Value::Null) => "**",
                Some(v) => v.as_str().ok_or_else(|| invalid("'pattern' must be a string"))?,
            };
            let repo = match args.get("repo") {
                None | Some(Value::Null) => None,
                Some(v) => Some(v.as_str().ok_or_else(|| invalid("'repo' must be a string"))?),
            };
            let limit = opt_usize(args, "limit", 200)?;
            let (mut files, truncated) = match (repo, current) {
                (None, Some(cur)) => {
                    // the session's repo first, then everything else
                    let (mut files, t1) = engine.list_paths(pattern, Some(cur), limit);
                    let (others, t2) = engine.list_paths(pattern, None, limit);
                    let mut truncated = t1 || t2;
                    for f in others {
                        if f.0 == cur {
                            continue;
                        }
                        if files.len() >= limit {
                            truncated = true;
                            break;
                        }
                        files.push(f);
                    }
                    (files, truncated)
                }
                _ => engine.list_paths(pattern, repo, limit),
            };
            if repo.is_none() {
                files.retain(|(r, _)| r != crate::sessions::REPO);
            }
            // Glob lists the session's repo (all of them without one) as
            // absolute paths, one per line
            let mut roots: HashMap<String, usize> = HashMap::new();
            let mut root_len = |r: &str| -> usize {
                if let Some(n) = roots.get(r) {
                    return *n;
                }
                let n = engine
                    .data_dir()
                    .and_then(|d| indexio_ingest::repo_state(d, r).ok())
                    .map_or(0, |st| st.path.as_os_str().len());
                roots.insert(r.to_string(), n);
                n
            };
            let listed: Vec<(&str, &str)> = files
                .iter()
                .filter(|(r, _)| current.map_or(true, |c| c == r))
                .map(|(r, p)| (r.as_str(), p.as_str()))
                .collect();
            let mut b = 0u64;
            for (r, p) in &listed {
                b += crate::usage::glob_bytes(std::iter::once(*p), root_len(r));
            }
            *baseline = Some(b);
            if text {
                return Ok(plain_result(mcp_text::files(&files, truncated)));
            }
            Ok(text_result(json!({
                "files": files.iter().map(|(r, p)| json!({ "repo": r, "path": p })).collect::<Vec<_>>(),
                "truncated": truncated,
            })))
        }
        "find_symbol" => {
            let name = req_str(args, "name")?;
            let repo = opt_repo(args)?;
            let mut hits = engine.find_symbol(name, fetch(DEFAULT_LIMIT).max(HUB_FETCH));
            // a grep for `fn name` in the session's repo: the definition lines
            *baseline = Some(grep_baseline(&hits, current));
            let note = focus_repo(&mut hits, current, repo.as_deref(), DEFAULT_LIMIT, "definitions");
            if text {
                // definition end lines for the first hits (one cached parse per file)
                let ends: HashMap<(String, String, u32), u32> = hits
                    .iter()
                    .take(RANGE_HITS)
                    .filter_map(|h| {
                        let (s, e) = engine.definition_range(&h.repo, &h.path, h.line)?;
                        (s == h.line).then_some(((h.repo.clone(), h.path.clone(), h.line), e))
                    })
                    .collect();
                let end_of = |h: &indexio_types::SearchHit| ends.get(&(h.repo.clone(), h.path.clone(), h.line)).copied();
                let mut out = mcp_text::hits_with_ends(
                    &hits,
                    true,
                    hits.len() >= DEFAULT_LIMIT,
                    "definitions",
                    &end_of,
                );
                out.push_str(&note);
                return Ok(plain_result(out));
            }
            Ok(text_result(json!(hits)))
        }
        "who_calls" => {
            let name = req_str(args, "name")?;
            let repo = opt_repo(args)?;
            let mut hits = engine.who_calls(name, fetch(DEFAULT_LIMIT).max(HUB_FETCH));
            // a grep for `name(` in the session's repo: the call lines
            *baseline = Some(grep_baseline(&hits, current));
            let note = focus_repo(&mut hits, current, repo.as_deref(), DEFAULT_LIMIT, "call sites");
            if text {
                let mut out = mcp_text::call_sites(&hits, hits.len() >= DEFAULT_LIMIT, "call sites");
                out.push_str(&note);
                return Ok(plain_result(out));
            }
            Ok(text_result(json!(hits)))
        }
        "recall" => {
            let q = req_str(args, "query")?;
            let limit = opt_usize(args, "limit", 8)?;
            // every leg scoped to the transcripts (SPEC-P10): before, the
            // legs ran over the whole corpus and the session hits were
            // whatever survived in a 60-deep fused list; the command-output
            // source (SPEC-P10 §31) is searched the same way and merged
            let n = limit.saturating_mul(3).max(24);
            let mut fused = engine
                .search_hybrid_in(q, n, embedder, FusionAlgo::Rrf, Some(crate::sessions::REPO))
                .map_err(|e| internal(format!("recall failed: {e}")))?;
            if engine.stats().repos.iter().any(|r| r == crate::runs::REPO) {
                if let Ok(runs) = engine.search_hybrid_in(q, n, embedder, FusionAlgo::Rrf, Some(crate::runs::REPO)) {
                    fused.extend(runs);
                    fused.sort_by(|a, b| b.rrf.partial_cmp(&a.rrf).unwrap_or(std::cmp::Ordering::Equal));
                }
            }
            let slug = std::env::current_dir().ok().map(|d| crate::sessions::project_slug(&d));
            let mut hits: Vec<indexio_types::SearchHit> = fused
                .into_iter()
                .map(|h| h.hit)
                // an earlier recall of the same question is not an answer
                .filter(|h| !h.snippet.contains("mcp__indexio__recall"))
                .collect();
            // this project's sessions and runs first (stable), then the others
            let mine = |h: &indexio_types::SearchHit| {
                (h.repo == crate::sessions::REPO && slug.as_ref().is_some_and(|s| h.path.starts_with(&format!("{s}/"))))
                    || (h.repo == crate::runs::REPO && current.is_some_and(|c| h.path.starts_with(&format!("{c}/"))))
            };
            let (mut here, rest): (Vec<_>, Vec<_>) = hits.into_iter().partition(mine);
            here.extend(rest);
            hits = here;
            hits.truncate(limit);
            // SPEC-P10 §31: when one exchange clearly answers, hand it back
            // whole instead of a pointer the model has to follow
            let exchange = hits.first().filter(|_| hits.len() == 1 || limit == 1).and_then(|h| {
                let start = h.line.saturating_sub(RECALL_EXCHANGE_BEFORE).max(1);
                engine.read_span(&h.repo, &h.path, start, h.line + RECALL_EXCHANGE_AFTER).map(|(body, last)| (h.repo.clone(), h.path.clone(), start, last, body))
            });
            if text {
                let mut out = mcp_text::hits(&hits, false, false, "recalled exchanges");
                if let Some((repo, path, start, last, body)) = &exchange {
                    out.push_str("\n\n");
                    out.push_str(&mcp_text::span(repo, path, *start, *last, body));
                }
                return Ok(plain_result(out));
            }
            Ok(text_result(json!({
                "hits": hits.iter().map(crate::serve::hit_to_json).collect::<Vec<_>>(),
                "exchange": exchange.map(|(repo, path, start, end, text)| json!({"repo": repo, "path": path, "start": start, "end": end, "text": text})),
            })))
        }
        "index_stats" => {
            let usage = engine.data_dir().map(|d| crate::usage::aggregate(d, USAGE_DAYS));
            if text {
                let mut s = stats_text(engine);
                if let Some(cur) = current {
                    s.push_str(&format!("\ncurrent repo (this session's working directory): {cur}"));
                }
                // the two summary lines only: the per-tool table is
                // `indexio usage`, and index_stats is called at most
                // session starts
                if let Some(u) = &usage {
                    if u.total.calls > 0 {
                        s.push('\n');
                        s.push_str(&crate::usage::render_summary(u));
                    }
                }
                return Ok(plain_result(s));
            }
            let mut v = stats_payload(engine);
            if let Some(u) = &usage {
                v["usage"] = crate::usage::to_json(u);
            }
            Ok(text_result(v))
        }
        "impact_of_symbol" => {
            let names: Vec<String> = match args.get("names") {
                Some(Value::Array(a)) => a
                    .iter()
                    .map(|v| {
                        v.as_str()
                            .map(str::to_string)
                            .ok_or_else(|| invalid("'names' must be an array of strings"))
                    })
                    .collect::<Result<_, _>>()?,
                Some(Value::String(s)) => vec![s.clone()],
                _ => return Err(invalid("missing or invalid 'names' parameter")),
            };
            if names.iter().all(|n| n.trim().is_empty()) {
                return Err(invalid("'names' must contain at least one symbol"));
            }
            let opts = impact_opts(args)?;
            let report = engine.impact_symbols(&names, &opts);
            if text {
                return Ok(plain_result(mcp_text::impact(None, &report, current)));
            }
            Ok(text_result(crate::impact_cli::impact_to_json(None, &report)))
        }
        "impact_of_diff" => {
            let repo = req_str(args, "repo")?;
            let opts = impact_opts(args)?;
            let repo_path = engine
                .data_dir()
                .and_then(|d| crate::impact_cli::resolve_repo_path(d, repo).ok());
            let diff = match args.get("diff") {
                Some(Value::String(s)) if !s.trim().is_empty() => s.clone(),
                Some(Value::Null) | None | Some(Value::String(_)) => {
                    let Some(path) = repo_path.clone() else {
                        return Err(invalid(format!(
                            "repo '{repo}' is not registered here; pass 'diff' text explicitly"
                        )));
                    };
                    let base = match args.get("base") {
                        None | Some(Value::Null) => "HEAD",
                        Some(v) => v.as_str().ok_or_else(|| invalid("'base' must be a string"))?,
                    };
                    crate::impact_cli::git_diff(&path, base)
                        .map_err(|e| internal(format!("git diff failed: {e:#}")))?
                }
                Some(_) => return Err(invalid("'diff' must be a string")),
            };
            let provider = crate::impact_cli::working_tree_provider(repo_path);
            let (changed, report) = engine.impact_diff(repo, &diff, &provider, &opts);
            if text {
                return Ok(plain_result(mcp_text::impact(Some(&changed), &report, Some(repo))));
            }
            Ok(text_result(crate::impact_cli::impact_to_json(Some(&changed), &report)))
        }
        "file_outline" => {
            let repo = req_str(args, "repo")?;
            let path = req_str(args, "path")?;
            let items = engine
                .outline(repo, path)
                .ok_or_else(|| invalid(format!("{repo}:{path} is not indexed")))?;
            // the harness has no outline: this call is the cost of reading
            // by span instead of whole
            *baseline = Some(0);
            if text {
                return Ok(plain_result(mcp_text::outline(repo, path, &items)));
            }
            Ok(text_result(crate::impact_cli::outline_to_json(repo, path, &items)))
        }
        "read_span" => {
            let repo = req_str(args, "repo")?;
            let path = req_str(args, "path")?;
            // no start (a worker omitted it) reads from the top, not an error
            let start = opt_usize(args, "start", 1)?.max(1);
            // SPEC-P9: no `end` → the enclosing definition's end (a whole
            // function in one read), else the classic 60-line window.
            let default_end = engine
                .definition_range(repo, path, start as u32)
                .map(|(_, e)| (e as usize).max(start))
                .unwrap_or(start + 59);
            let end = opt_usize(args, "end", default_end)?;
            let (body, last) = engine
                .read_span(repo, path, start as u32, end as u32)
                .ok_or_else(|| invalid(format!("{repo}:{path} is not indexed")))?;
            // the harness would have read the whole file — once
            let first = read_seen
                .lock()
                .map(|mut s| s.insert((repo.to_string(), path.to_string())))
                .unwrap_or(false);
            *baseline = Some(if first {
                engine.file_content(repo, path).map_or(0, |c| crate::usage::read_bytes(&c))
            } else {
                0
            });
            if text {
                return Ok(plain_result(mcp_text::span(repo, path, start as u32, last, &body)));
            }
            Ok(text_result(json!({
                "repo": repo, "path": path, "start": start, "end": last, "text": body,
            })))
        }
        other => Err(invalid(format!("unknown tool: {other}"))),
    }
}

/// Move the session repo's hits to the front (stable within each half),
/// drop transcript hits (the `sessions` source is for `recall`, not for
/// code lookups — a symbol name occurs in every transcript that touched
/// it), then truncate to `limit`.
fn prefer_repo(hits: &mut Vec<indexio_types::SearchHit>, current: Option<&str>, limit: usize) {
    hits.retain(|h| h.repo != crate::sessions::REPO);
    if let Some(cur) = current {
        let (mut here, rest): (Vec<_>, Vec<_>) = std::mem::take(hits).into_iter().partition(|h| h.repo == cur);
        here.extend(rest);
        *hits = here;
    }
    hits.truncate(limit);
}

/// Days of the usage log `index_stats` summarises.
const USAGE_DAYS: u64 = 7;
/// Lines of transcript handed back around a lone recall hit (SPEC-P10 §31).
const RECALL_EXCHANGE_BEFORE: u32 = 4;
const RECALL_EXCHANGE_AFTER: u32 = 36;

/// What a built-in `Grep` in the session's repo (every repo without one)
/// would have printed for these rows (SPEC-P10 §22).
fn grep_baseline(hits: &[indexio_types::SearchHit], current: Option<&str>) -> u64 {
    crate::usage::grep_bytes(
        hits.iter()
            .filter(|h| h.repo != crate::sessions::REPO && current.map_or(true, |c| h.repo == c))
            .map(|h| (h.path.as_str(), h.line, h.snippet.as_str())),
    )
}

/// The query for a `code_grep` pattern: the pattern as one regex, with any
/// trailing ` repo:X path:Y lang:Z case:` tokens as filters.
fn grep_query(pattern: &str) -> Result<indexio_query::Query, String> {
    let mut pattern = pattern.trim_end();
    let mut filters = String::new();
    while let Some((head, tail)) = pattern.rsplit_once(char::is_whitespace) {
        if ["repo:", "path:", "lang:", "case:"].iter().any(|p| tail.starts_with(p)) {
            filters.insert_str(0, &format!(" {tail}"));
            pattern = head.trim_end();
        } else {
            break;
        }
    }
    let q = format!("/{}/{filters}", pattern.replace('/', "\\/"));
    indexio_query::parse(&q).map_err(|e| e.to_string())
}

/// Whether a lexical query reads as a grep pattern rather than words: an
/// alternation, a character class, a quantifier, an anchor or an escape
/// (SPEC-P10 §27). `/…/` is already a regex to the parser and is excluded.
fn looks_like_regex(q: &str) -> bool {
    let mut toks: Vec<&str> = q.split_whitespace().collect();
    while toks.last().is_some_and(|t| ["repo:", "path:", "lang:", "case:"].iter().any(|p| t.starts_with(p))) {
        toks.pop();
    }
    let body = toks.join(" ");
    if body.starts_with('/') {
        return false;
    }
    body.contains('|')
        || body.contains('[')
        || body.contains(".*")
        || body.contains('\\')
        || body.ends_with('$')
        || body.starts_with('^')
        || body.contains(")?")
        || body.contains(")+")
        || body.contains(")*")
}

/// Rows fetched for a symbol lookup before the repo focus is applied (hub
/// names have hundreds of definitions across the index).
const HUB_FETCH: usize = 400;
/// Symbol lookups with at most this many rows are listed in full across
/// repos; above it the session's repo is listed and the others are counted.
const COLLAPSE_SYMBOLS_ABOVE: usize = 12;

/// Optional `repo` argument (substring, case-insensitive like `repo:`).
fn opt_repo(args: &Value) -> Result<Option<String>, (i64, String)> {
    match args.get("repo") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_str()
            .map(|s| Some(s.to_ascii_lowercase()))
            .ok_or_else(|| invalid("'repo' must be a string")),
    }
}

/// Focus symbol-lookup rows (SPEC-P9 §20): an explicit `repo` keeps only
/// that repo; otherwise a hub name (> [`COLLAPSE_SYMBOLS_ABOVE`] rows)
/// keeps the session's repo in full and collapses the others to per-repo
/// counts in the returned note (empty when nothing was collapsed). Rows
/// are truncated to `limit`.
fn focus_repo(
    hits: &mut Vec<indexio_types::SearchHit>,
    current: Option<&str>,
    repo: Option<&str>,
    limit: usize,
    what: &str,
) -> String {
    hits.retain(|h| h.repo != crate::sessions::REPO);
    if let Some(r) = repo {
        hits.retain(|h| h.repo.to_ascii_lowercase().contains(r));
        hits.truncate(limit);
        return String::new();
    }
    let Some(cur) = current else {
        hits.truncate(limit);
        return String::new();
    };
    let (mut here, elsewhere): (Vec<_>, Vec<_>) = std::mem::take(hits).into_iter().partition(|h| h.repo == cur);
    let n_here = here.len();
    if n_here + elsewhere.len() <= COLLAPSE_SYMBOLS_ABOVE || (n_here == 0 && elsewhere.len() <= limit) {
        here.extend(elsewhere);
        here.truncate(limit);
        *hits = here;
        return String::new();
    }
    let rest: Vec<indexio_types::SearchHit> = if n_here == 0 {
        // nothing in the session's repo: a page of the others, the rest counted
        let mut all = elsewhere;
        let rest = all.split_off(limit.min(all.len()));
        here = all;
        rest
    } else {
        let mut rest = here.split_off(limit.min(n_here));
        rest.extend(elsewhere);
        rest
    };
    *hits = here;
    if rest.is_empty() {
        return String::new();
    }
    let mut by_repo: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for h in &rest {
        *by_repo.entry(h.repo.as_str()).or_insert(0) += 1;
    }
    let mut repos: Vec<(&str, usize)> = by_repo.into_iter().collect();
    repos.sort_by_key(|(name, n)| (std::cmp::Reverse(*n), *name));
    let list: Vec<String> = repos.iter().map(|(name, n)| format!("{name} {n}")).collect();
    format!("\n… {} more {what} elsewhere (pass repo: to list them): {}", rest.len(), list.join(", "))
}

/// Cheap fingerprint of a shard directory: (files, bytes, newest mtime).
fn shards_stamp(dir: &std::path::Path) -> (usize, u64, Option<std::time::SystemTime>) {
    let mut n = 0usize;
    let mut bytes = 0u64;
    let mut newest: Option<std::time::SystemTime> = None;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if e.path().extension().and_then(|x| x.to_str()) != Some("cidx") {
                continue;
            }
            if let Ok(m) = e.metadata() {
                n += 1;
                bytes += m.len();
                if let Ok(t) = m.modified() {
                    newest = Some(newest.map_or(t, |c| c.max(t)));
                }
            }
        }
    }
    (n, bytes, newest)
}

/// Whether a lexical query explicitly asks for the transcripts.
/// The single registered repo an explicit `repo:X` token in `q` names
/// (case-insensitive substring, like the lexical filter); `None` when there
/// is no token or it matches several repos. Scopes every hybrid leg
/// (SPEC-P10) instead of only the lexical one.
fn repo_scope(engine: &Engine, q: &str) -> Option<String> {
    engine.repo_scope(q)
}

fn wants_sessions(q: &str) -> bool {
    q.split_whitespace()
        .any(|t| t.strip_prefix("repo:").map_or(false, |r| crate::sessions::REPO.contains(&r.to_ascii_lowercase())))
}

/// The repo an MCP session works in: `explicit` (`--repo`) if registered,
/// else `INDEXIO_REPO`, else the registered repo whose folder contains the
/// process's working directory (harnesses start MCP servers in the project
/// directory). `None` when nothing matches.
pub fn resolve_current_repo(data_dir: &std::path::Path, explicit: Option<&str>) -> Option<String> {
    let names = indexio_ingest::sources::registered_repo_names(data_dir).unwrap_or_default();
    let pick = explicit
        .map(str::to_string)
        .or_else(|| std::env::var("INDEXIO_REPO").ok().filter(|s| !s.is_empty()));
    if let Some(p) = pick {
        return if names.iter().any(|n| *n == p) {
            Some(p)
        } else {
            tracing::warn!(repo = %p, "current repo is not registered; ignoring");
            None
        };
    }
    let cwd = std::env::current_dir().ok()?;
    repo_containing(data_dir, &cwd)
}

/// The deepest registered repo whose folder contains `dir` (junctions and
/// symlinks resolved on both sides), if any.
pub fn repo_containing(data_dir: &std::path::Path, dir: &std::path::Path) -> Option<String> {
    let names = indexio_ingest::sources::registered_repo_names(data_dir).unwrap_or_default();
    let cwd = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let mut best: Option<(usize, String)> = None;
    for name in names {
        let Ok(st) = indexio_ingest::repo_state(data_dir, &name) else { continue };
        let root = st.path.canonicalize().unwrap_or(st.path.clone());
        if cwd.starts_with(&root) {
            let depth = root.components().count();
            if best.as_ref().map_or(true, |(d, _)| depth > *d) {
                best = Some((depth, name));
            }
        }
    }
    best.map(|(_, n)| n)
}

/// `depth` / `max_sites` (optional) → ImpactOptions (SPEC-P6 §3).
fn impact_opts(args: &Value) -> Result<indexio_query::ImpactOptions, (i64, String)> {
    let d = indexio_query::ImpactOptions::default();
    let depth = opt_usize(args, "depth", d.depth as usize)?;
    let max_sites = opt_usize(args, "max_sites", d.max_sites)?;
    Ok(crate::impact_cli::options(depth as u32, max_sites, d.max_fanout))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use indexio_core::grams::{self, CommonGrams};
    use indexio_index::ShardWriter;
    use indexio_types::{BlobId, CallRec, DocMeta, ExtractedArtifact, Lang, SymbolKind, SymbolRec};

    /// Fixture engine: one shard, repo "alpha", two docs with a symbol and
    /// a call posting (same pattern as the indexio-query tests).
    fn fixture_engine() -> (tempfile::TempDir, Engine) {
        let tmp = tempfile::tempdir().unwrap();
        let shards = tmp.path().join("shards");
        std::fs::create_dir_all(&shards).unwrap();

        let docs = [
            (
                "src/foo.rs",
                "fn foo_bar_123(x: i32) -> i32 {\n    helper(x)\n}\n",
                vec![SymbolRec {
                    name: "foo_bar_123".into(),
                    kind: SymbolKind::Fn,
                    line: 1,
                    col: 0,
                    scope: String::new(),
                }],
                vec![],
            ),
            (
                "src/caller.rs",
                "fn run() {\n    foo_bar_123(3);\n}\n",
                vec![SymbolRec {
                    name: "run".into(),
                    kind: SymbolKind::Fn,
                    line: 1,
                    col: 0,
                    scope: String::new(),
                }],
                vec![CallRec {
                    callee: "foo_bar_123".into(),
                    caller: "run".into(),
                    line: 2,
                }],
            ),
        ];
        write_shard(&shards, &docs, &["alpha".to_string()]);
        let engine = Engine::open(tmp.path()).unwrap();
        (tmp, engine)
    }

    type Doc<'a> = (
        &'a str,
        &'a str,
        Vec<SymbolRec>,
        Vec<CallRec>,
    );

    fn write_shard(dir: &Path, docs: &[Doc], repos: &[String]) {
        let mut w = ShardWriter::new(dir).unwrap();
        for (path, content, symbols, calls) in docs {
            let content = content.as_bytes();
            let art = ExtractedArtifact {
                ngrams: grams::extract(content, &CommonGrams::empty()),
                symbols: symbols.clone(),
                calls: calls.clone(),
                raw_len: content.len() as u32,
                lang: Lang::Rust,
            };
            let meta = DocMeta {
                blob: BlobId::from_content(content),
                repo_id: 0,
                path: path.to_string(),
                lang: Lang::Rust,
                raw_len: content.len() as u32,
            };
            w.add_doc(&meta, content, &art).unwrap();
        }
        w.finish(repos).unwrap();
    }

    fn parse_resp(s: &str) -> Value {
        serde_json::from_str(s).unwrap()
    }

    /// Offline embedder for the semantic plane in tests.
    fn hash_emb() -> indexio_embed::embed::HashEmbedder {
        indexio_embed::embed::HashEmbedder::new(512)
    }

    /// Offline reranker for the rerank stage in tests.
    fn rr() -> indexio_embed::rerank::OverlapReranker {
        indexio_embed::rerank::OverlapReranker
    }

    #[test]
    fn initialize_handshake_shape() {
        let (_t, e) = fixture_engine();
        let out = handle(&e, &hash_emb(), &rr(), r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#).unwrap();
        let v = parse_resp(&out);
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["id"], 1);
        assert_eq!(v["result"]["protocolVersion"], "2024-11-05");
        assert!(v["result"]["capabilities"]["tools"].is_object());
        assert_eq!(v["result"]["serverInfo"]["name"], "indexio");
        assert_eq!(v["result"]["serverInfo"]["version"], "0.1.0");
    }

    #[test]
    fn ping_returns_empty_object() {
        let (_t, e) = fixture_engine();
        let out = handle(&e, &hash_emb(), &rr(), r#"{"jsonrpc":"2.0","id":2,"method":"ping"}"#).unwrap();
        let v = parse_resp(&out);
        assert_eq!(v["id"], 2);
        assert_eq!(v["result"], json!({}));
    }

    #[test]
    fn initialized_notification_gets_no_reply() {
        let (_t, e) = fixture_engine();
        assert!(handle(&e, &hash_emb(), &rr(), r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).is_none());
        // any id-less message is a notification, even known methods
        assert!(handle(&e, &hash_emb(), &rr(), r#"{"jsonrpc":"2.0","method":"ping"}"#).is_none());
        assert!(handle(&e, &hash_emb(), &rr(), "").is_none());
    }

    #[test]
    fn tools_list_has_all_twelve_tools_with_schemas() {
        let (_t, e) = fixture_engine();
        let out = handle(&e, &hash_emb(), &rr(), r#"{"jsonrpc":"2.0","id":3,"method":"tools/list"}"#).unwrap();
        let v = parse_resp(&out);
        let tools = v["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 13);
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        for want in [
            "code_search",
            "semantic_search",
            "code_grep",
            "find_symbol",
            "who_calls",
            "index_stats",
            "impact_of_symbol",
            "impact_of_diff",
            "file_outline",
            "read_span",
            "list_files",
            "refresh_index",
        ] {
            assert!(names.contains(&want), "missing tool {want}");
        }
        for t in tools {
            assert_eq!(t["inputSchema"]["type"], "object");
            assert!(t["inputSchema"]["properties"].is_object(), "{}", t["name"]);
            assert!(t["description"].is_string());
        }
        // required params
        let cs = tools.iter().find(|t| t["name"] == "code_search").unwrap();
        assert_eq!(cs["inputSchema"]["required"], json!(["query"]));
        // code_search gained the optional mode enum
        assert_eq!(
            cs["inputSchema"]["properties"]["mode"]["enum"],
            json!(["lexical", "semantic", "hybrid"])
        );
        // descriptions tell the agent WHEN to use each mode
        let cs_desc = cs["description"].as_str().unwrap();
        for kw in ["lexical", "semantic", "hybrid"] {
            assert!(cs_desc.contains(kw), "code_search description lacks '{kw}'");
        }
        let ss = tools.iter().find(|t| t["name"] == "semantic_search").unwrap();
        assert_eq!(ss["inputSchema"]["required"], json!(["query"]));
        assert!(ss["inputSchema"]["properties"]["k"].is_object());
        assert!(
            ss["description"].as_str().unwrap().contains("natural-language"),
            "semantic_search description must say when to use it"
        );
    }

    #[test]
    fn tools_call_code_search_against_fixture() {
        let (_t, e) = fixture_engine();
        let req = r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"code_search","arguments":{"query":"foo_bar_123","limit":10}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        assert_eq!(v["id"], 4);
        let content = v["result"]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "text");
        let payload: Value = serde_json::from_str(content[0]["text"].as_str().unwrap()).unwrap();
        let paths: Vec<&str> = payload["hits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| h["path"].as_str().unwrap())
            .collect();
        assert!(paths.contains(&"src/foo.rs"), "{paths:?}");
        assert!(paths.contains(&"src/caller.rs"), "{paths:?}");
        assert_eq!(payload["truncated"], false);
        assert!(payload["took_ms"].is_u64());
    }

    /// SPEC-P10 §29: shards written by other processes are folded by any
    /// server's periodic check, not only after its own refresh.
    #[test]
    fn periodic_check_compacts_shards_other_writers_left() {
        let (t, e) = fixture_engine();
        let dir = e.data_dir().unwrap().to_path_buf();
        let shards = dir.join("shards");
        // 14 one-doc delta shards, as hooks and syncs would leave them
        for i in 0..14 {
            let path = format!("src/extra_{i}.rs");
            let body = format!("fn extra_{i}() {{}}\n");
            write_shard(&shards, &[(path.as_str(), body.as_str(), vec![], vec![])], &["alpha".to_string()]);
        }
        assert!(shards_stamp(&shards).0 > COMPACT_ABOVE);
        let server = McpServer::with_format(Engine::open(&dir).unwrap(), Arc::new(hash_emb()), Arc::new(rr()), OutputFormat::Text);
        let _ = t.path();
        // an ordinary call, no edit of the session's own repo
        let _ = server.handle(r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"find_symbol","arguments":{"name":"foo_bar_123"}}}"#);
        server.finish_background_work();
        assert!(shards_stamp(&shards).0 <= COMPACT_TO, "shards left: {}", shards_stamp(&shards).0);
        // and the merged index still answers
        let v = parse_resp(&server.handle(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"code_search","arguments":{"query":"extra_7","mode":"lexical"}}}"#).unwrap());
        assert!(v["result"]["content"][0]["text"].as_str().unwrap().contains("extra_7.rs"), "{v}");
    }

    #[test]
    fn tools_call_symbol_calls_and_stats() {
        let (_t, e) = fixture_engine();
        let find = r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"find_symbol","arguments":{"name":"foo_bar_123"}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), find).unwrap());
        let payload: Value =
            serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(payload.as_array().unwrap()[0]["path"], "src/foo.rs");

        let calls = r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"who_calls","arguments":{"name":"foo_bar_123"}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), calls).unwrap());
        let payload: Value =
            serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(payload.as_array().unwrap()[0]["path"], "src/caller.rs");

        let stats = r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"index_stats","arguments":{}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), stats).unwrap());
        let payload: Value =
            serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(payload["doc_count"], 2);
        assert_eq!(payload["repos"], json!(["alpha"]));
    }

    #[test]
    fn tools_call_code_grep_wraps_pattern() {
        let (_t, e) = fixture_engine();
        let req = r#"{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"code_grep","arguments":{"pattern":"foo_bar_[0-9]+"}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        let payload: Value =
            serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert!(!payload["hits"].as_array().unwrap().is_empty());
    }

    #[test]
    fn unknown_method_is_method_not_found() {
        let (_t, e) = fixture_engine();
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), r#"{"jsonrpc":"2.0","id":9,"method":"bogus"}"#).unwrap());
        assert_eq!(v["error"]["code"], -32601);
        assert_eq!(v["id"], 9);
    }

    #[test]
    fn invalid_params_are_minus_32602() {
        let (_t, e) = fixture_engine();
        // missing arguments.query
        let v = parse_resp(
            &handle(&e, &hash_emb(), &rr(), r#"{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"code_search","arguments":{}}}"#).unwrap(),
        );
        assert_eq!(v["error"]["code"], -32602);
        // unknown tool
        let v = parse_resp(
            &handle(&e, &hash_emb(), &rr(), r#"{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"nope"}}"#).unwrap(),
        );
        assert_eq!(v["error"]["code"], -32602);
        // unparsable query (bad regex)
        let v = parse_resp(
            &handle(&e, &hash_emb(), &rr(), r#"{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{"name":"code_grep","arguments":{"pattern":"(bad"}}}"#).unwrap(),
        );
        assert_eq!(v["error"]["code"], -32602);
    }

    /// Fixture with the vec sidecar embedded for repo "alpha".
    fn fixture_engine_embedded() -> (tempfile::TempDir, Engine) {
        let (tmp, e) = fixture_engine();
        let emb = hash_emb();
        let set = indexio_index::ShardSet::open_dir(&tmp.path().join("shards")).unwrap();
        indexio_embed::pipeline::embed_repo(&set, tmp.path(), "alpha", &emb).unwrap();
        (tmp, e)
    }

    #[test]
    fn tools_call_code_search_hybrid_mode() {
        let (_t, e) = fixture_engine_embedded();
        let req = r#"{"jsonrpc":"2.0","id":20,"method":"tools/call","params":{"name":"code_search","arguments":{"query":"foo_bar_123","mode":"hybrid","limit":10}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        assert_eq!(v["id"], 20);
        assert!(v.get("error").is_none(), "{v}");
        let payload: Value =
            serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        let hits = payload["hits"].as_array().unwrap();
        assert!(!hits.is_empty(), "{payload}");
        // Hybrid envelope: fused score + per-list ranks. Both fixture docs
        // contain the token, so both are in both lists (ranks may tie).
        let first = &hits[0];
        assert!(first["hit"]["path"].is_string(), "{first}");
        assert!(first["rrf"].is_f64());
        assert!(first["lex_rank"].is_u64(), "{first}");
        assert!(first["sem_rank"].is_u64(), "vec index embedded: {first}");
        assert!(
            hits[0]["rrf"].as_f64().unwrap() >= hits[1]["rrf"].as_f64().unwrap(),
            "rrf sorted desc: {hits:?}"
        );
        // The defining doc appears.
        let paths: Vec<&str> = hits
            .iter()
            .map(|h| h["hit"]["path"].as_str().unwrap())
            .collect();
        assert!(paths.contains(&"src/foo.rs"), "{paths:?}");
    }

    /// SPEC-P10 §27: a grep-style alternation given to lexical mode is run
    /// as the regex it is once the literal reading finds nothing.
    #[test]
    fn lexical_regex_looking_query_without_hits_is_retried_as_a_regex() {
        let (_t, e) = fixture_engine();
        let req = r#"{"jsonrpc":"2.0","id":24,"method":"tools/call","params":{"name":"code_search","arguments":{"query":"zzz_absent|foo_bar_123","mode":"lexical"}}}"#;
        let v = parse_resp(&handle_text(&e, req).unwrap());
        let t = v["result"]["content"][0]["text"].as_str().unwrap();
        assert!(t.starts_with("no literal match; treated as a regex:"), "{t}");
        assert!(t.contains("src/foo.rs"), "{t}");
        // a character class too, and the filters survive
        let req = r#"{"jsonrpc":"2.0","id":25,"method":"tools/call","params":{"name":"code_search","arguments":{"query":"foo_bar_[0-9]+ repo:alpha","mode":"lexical"}}}"#;
        let v = parse_resp(&handle_text(&e, req).unwrap());
        let t = v["result"]["content"][0]["text"].as_str().unwrap();
        assert!(t.contains("src/foo.rs"), "{t}");
        // plain words are not regexes
        assert!(!looks_like_regex("storage_state_set session_save"));
        assert!(!looks_like_regex("/a|b/"));
        assert!(looks_like_regex("onHover|onMouseEnter|:hover repo:x"));
        assert!(looks_like_regex("width: 3[5-9][0-9]"));
    }

    /// SPEC-P10 §25: a multi-word lexical query that matches no line as a
    /// conjunction falls through to the hybrid ranking with a note, instead
    /// of `no hits` (which the model answered with a reworded retry).
    #[test]
    fn lexical_conjunction_without_hits_falls_back_to_hybrid() {
        let (_t, e) = fixture_engine_embedded();
        let req = r#"{"jsonrpc":"2.0","id":22,"method":"tools/call","params":{"name":"code_search","arguments":{"query":"foo_bar_123 zzz_not_in_fixture","mode":"lexical"}}}"#;
        let out = handle_text(&e, req).unwrap();
        let v = parse_resp(&out);
        let t = v["result"]["content"][0]["text"].as_str().unwrap();
        assert!(t.starts_with("no line has every word"), "{t}");
        assert!(t.contains("src/foo.rs"), "{t}");
        // a single word or a regex still reports no hits
        let req = r#"{"jsonrpc":"2.0","id":23,"method":"tools/call","params":{"name":"code_search","arguments":{"query":"zzz_not_in_fixture","mode":"lexical"}}}"#;
        let v = parse_resp(&handle_text(&e, req).unwrap());
        assert_eq!(v["result"]["content"][0]["text"], "no hits");
    }

    #[test]
    fn tools_call_semantic_search_tool() {
        let (_t, e) = fixture_engine_embedded();
        // "helper" appears only in src/foo.rs.
        let req = r#"{"jsonrpc":"2.0","id":21,"method":"tools/call","params":{"name":"semantic_search","arguments":{"query":"helper","k":5}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        let payload: Value =
            serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        let hits = payload["hits"].as_array().unwrap();
        assert!(!hits.is_empty(), "{payload}");
        assert_eq!(hits[0]["path"], "src/foo.rs", "{hits:?}");
        assert!(hits[0]["score"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn code_search_invalid_mode_is_invalid_params() {
        let (_t, e) = fixture_engine();
        let req = r#"{"jsonrpc":"2.0","id":22,"method":"tools/call","params":{"name":"code_search","arguments":{"query":"foo","mode":"bogus"}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        assert_eq!(v["error"]["code"], -32602);
        assert!(v["error"]["message"].as_str().unwrap().contains("bogus"));
    }

    #[test]
    fn semantic_modes_degrade_without_vec_index() {
        let (_t, e) = fixture_engine(); // no embed run
        // pure semantic: graceful empty result, not an error
        let req = r#"{"jsonrpc":"2.0","id":23,"method":"tools/call","params":{"name":"code_search","arguments":{"query":"foo_bar_123","mode":"semantic"}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        assert!(v.get("error").is_none(), "{v}");
        let payload: Value =
            serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(payload["hits"].as_array().unwrap().len(), 0);
        // hybrid: lexical-only, sem_rank null everywhere
        let req = r#"{"jsonrpc":"2.0","id":24,"method":"tools/call","params":{"name":"code_search","arguments":{"query":"foo_bar_123","mode":"hybrid"}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        let payload: Value =
            serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        let hits = payload["hits"].as_array().unwrap();
        assert!(!hits.is_empty());
        for h in hits {
            assert_eq!(h["sem_rank"], Value::Null, "{h}");
            assert!(h["lex_rank"].is_u64());
        }
    }

    #[test]
    fn batch_request_returns_array() {
        let (_t, e) = fixture_engine();
        let batch = r#"[{"jsonrpc":"2.0","id":1,"method":"initialize"},{"jsonrpc":"2.0","id":2,"method":"ping"},{"jsonrpc":"2.0","method":"notifications/initialized"}]"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), batch).unwrap());
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2, "notification must not produce a response");
        assert_eq!(arr[0]["id"], 1);
        assert_eq!(arr[1]["id"], 2);
        // all-notification batch -> no output line at all
        let all_notif = r#"[{"jsonrpc":"2.0","method":"notifications/initialized"}]"#;
        assert!(handle(&e, &hash_emb(), &rr(), all_notif).is_none());
    }

    #[test]
    fn code_search_rerank_requires_hybrid_mode() {
        let (_t, e) = fixture_engine();
        // rerank=true + mode=lexical -> -32602 invalid params (SPEC-P3 §2)
        let req = r#"{"jsonrpc":"2.0","id":30,"method":"tools/call","params":{"name":"code_search","arguments":{"query":"foo_bar_123","mode":"lexical","rerank":true}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        assert_eq!(v["error"]["code"], -32602);
        assert!(v["error"]["message"].as_str().unwrap().contains("hybrid"));
        // rerank=true + default (lexical) mode -> also -32602
        let req = r#"{"jsonrpc":"2.0","id":31,"method":"tools/call","params":{"name":"code_search","arguments":{"query":"foo_bar_123","rerank":true}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        assert_eq!(v["error"]["code"], -32602);
        // rerank=true + mode=semantic -> -32602
        let req = r#"{"jsonrpc":"2.0","id":32,"method":"tools/call","params":{"name":"code_search","arguments":{"query":"foo_bar_123","mode":"semantic","rerank":true}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        assert_eq!(v["error"]["code"], -32602);
        // non-boolean rerank -> -32602
        let req = r#"{"jsonrpc":"2.0","id":33,"method":"tools/call","params":{"name":"code_search","arguments":{"query":"foo_bar_123","mode":"hybrid","rerank":"yes"}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        assert_eq!(v["error"]["code"], -32602);
    }

    #[test]
    fn code_search_hybrid_rerank_returns_rerank_scores() {
        let (_t, e) = fixture_engine_embedded();
        let req = r#"{"jsonrpc":"2.0","id":34,"method":"tools/call","params":{"name":"code_search","arguments":{"query":"foo_bar_123","mode":"hybrid","rerank":true,"limit":10}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        assert!(v.get("error").is_none(), "{v}");
        let payload: Value =
            serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        let hits = payload["hits"].as_array().unwrap();
        assert!(!hits.is_empty(), "{payload}");
        for h in hits {
            assert!(h["rerank_score"].is_f64(), "rerank_score filled: {h}");
        }
        // rerank=false (or absent) keeps rerank_score null
        let req = r#"{"jsonrpc":"2.0","id":35,"method":"tools/call","params":{"name":"code_search","arguments":{"query":"foo_bar_123","mode":"hybrid","rerank":false}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        let payload: Value =
            serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        for h in payload["hits"].as_array().unwrap() {
            assert_eq!(h["rerank_score"], Value::Null);
        }
    }

    #[test]
    fn tools_list_code_search_has_rerank_schema() {
        let (_t, e) = fixture_engine();
        let out = handle(&e, &hash_emb(), &rr(), r#"{"jsonrpc":"2.0","id":36,"method":"tools/list"}"#).unwrap();
        let v = parse_resp(&out);
        let tools = v["result"]["tools"].as_array().unwrap();
        let cs = tools.iter().find(|t| t["name"] == "code_search").unwrap();
        assert_eq!(cs["inputSchema"]["properties"]["rerank"]["type"], "boolean");
        assert_eq!(
            cs["inputSchema"]["properties"]["rerank"]["description"],
            "hybrid only (default false); needs a configured rerank endpoint, otherwise no effect"
        );
        // still optional: only query is required
        assert_eq!(cs["inputSchema"]["required"], json!(["query"]));
    }

    // ------------------------------------------------------------------
    // SPEC-P6 §3: impact / outline / span tools
    // ------------------------------------------------------------------

    fn payload_of(v: &Value) -> Value {
        serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap()).unwrap()
    }

    #[test]
    fn tools_call_impact_of_symbol() {
        let (_t, e) = fixture_engine();
        let req = r#"{"jsonrpc":"2.0","id":40,"method":"tools/call","params":{"name":"impact_of_symbol","arguments":{"names":["foo_bar_123"],"depth":1}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        assert!(v.get("error").is_none(), "{v}");
        let p = payload_of(&v);
        assert_eq!(p["roots"], json!(["foo_bar_123"]));
        assert_eq!(p["definitions"][0]["path"], "src/foo.rs");
        assert_eq!(p["sites"][0]["path"], "src/caller.rs");
        assert_eq!(p["sites"][0]["caller"], "run");
        assert_eq!(p["sites"][0]["depth"], 1);
        assert_eq!(p["files"][0]["path"], "src/caller.rs");
        assert_eq!(p["truncated"], false);
        assert!(p.get("changed").is_none());
        // a bare string is accepted for `names`
        let req = r#"{"jsonrpc":"2.0","id":41,"method":"tools/call","params":{"name":"impact_of_symbol","arguments":{"names":"foo_bar_123"}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        assert!(v.get("error").is_none(), "{v}");
        // missing names -> -32602
        let req = r#"{"jsonrpc":"2.0","id":42,"method":"tools/call","params":{"name":"impact_of_symbol","arguments":{}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        assert_eq!(v["error"]["code"], -32602);
    }

    #[test]
    fn tools_call_impact_of_diff_with_patch_text() {
        let (_t, e) = fixture_engine();
        // Edit inside foo_bar_123 (line 2 of src/foo.rs); the repo is not
        // registered, so the pre-change side maps through the index.
        let req = json!({
            "jsonrpc": "2.0", "id": 43, "method": "tools/call",
            "params": { "name": "impact_of_diff", "arguments": {
                "repo": "alpha",
                "diff": "--- a/src/foo.rs\n+++ b/src/foo.rs\n@@ -2,1 +2,1 @@\n-    helper(x)\n+    helper(x + 1)\n",
                "depth": 1
            }}
        })
        .to_string();
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), &req).unwrap());
        assert!(v.get("error").is_none(), "{v}");
        let p = payload_of(&v);
        assert_eq!(p["changed"][0]["name"], "foo_bar_123");
        assert_eq!(p["changed"][0]["lines"], json!([2]));
        assert_eq!(p["sites"][0]["path"], "src/caller.rs");
        // no diff + unregistered repo -> -32602 with guidance
        let req = r#"{"jsonrpc":"2.0","id":44,"method":"tools/call","params":{"name":"impact_of_diff","arguments":{"repo":"alpha"}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        assert_eq!(v["error"]["code"], -32602);
        assert!(v["error"]["message"].as_str().unwrap().contains("not registered"));
    }

    #[test]
    fn tools_call_file_outline_and_read_span() {
        let (_t, e) = fixture_engine();
        let req = r#"{"jsonrpc":"2.0","id":45,"method":"tools/call","params":{"name":"file_outline","arguments":{"repo":"alpha","path":"src/foo.rs"}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        assert!(v.get("error").is_none(), "{v}");
        let p = payload_of(&v);
        assert_eq!(p["items"][0]["name"], "foo_bar_123");
        assert_eq!(p["items"][0]["start_line"], 1);
        assert_eq!(p["items"][0]["end_line"], 3);
        let req = r#"{"jsonrpc":"2.0","id":46,"method":"tools/call","params":{"name":"read_span","arguments":{"repo":"alpha","path":"src/foo.rs","start":2,"end":3}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        let p = payload_of(&v);
        assert_eq!(p["text"], "    helper(x)\n}\n");
        assert_eq!(p["end"], 3);
        // unknown file -> -32602; missing start -> -32602
        let req = r#"{"jsonrpc":"2.0","id":47,"method":"tools/call","params":{"name":"file_outline","arguments":{"repo":"alpha","path":"nope.rs"}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        assert_eq!(v["error"]["code"], -32602);
        // no start: the top of the file (a worker omitted it and lost the turn)
        let req = r#"{"jsonrpc":"2.0","id":48,"method":"tools/call","params":{"name":"read_span","arguments":{"repo":"alpha","path":"src/foo.rs"}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        assert!(v.get("error").is_none(), "{v}");
        assert_eq!(payload_of(&v)["start"], 1);
    }

    #[test]
    fn tool_results_are_compact_json() {
        let (_t, e) = fixture_engine();
        let req = r#"{"jsonrpc":"2.0","id":49,"method":"tools/call","params":{"name":"index_stats","arguments":{}}}"#;
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap());
        let text = v["result"]["content"][0]["text"].as_str().unwrap();
        assert!(!text.contains('\n'), "compact JSON expected: {text}");
        assert!(serde_json::from_str::<Value>(text).is_ok());
    }

    #[test]
    fn parse_error_is_minus_32700() {
        let (_t, e) = fixture_engine();
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), "{not json").unwrap());
        assert_eq!(v["error"]["code"], -32700);
        assert_eq!(v["id"], Value::Null);
    }

    // ------------------------------------------------------------------
    // SPEC-P8: harness integration
    // ------------------------------------------------------------------

    #[test]
    fn initialize_carries_instructions_with_repos_and_guidance() {
        let (_t, e) = fixture_engine();
        let out = handle(&e, &hash_emb(), &rr(), r#"{"jsonrpc":"2.0","id":50,"method":"initialize"}"#).unwrap();
        let v = parse_resp(&out);
        let ins = v["result"]["instructions"].as_str().unwrap();
        assert!(ins.contains("alpha"), "{ins}");
        for kw in ["code_search", "list_files", "file_outline", "read_span", "impact_of_diff", "refresh_index", "grep"] {
            assert!(ins.contains(kw), "instructions lack '{kw}': {ins}");
        }
        assert!(ins.len() < 3000, "instructions must stay short: {}", ins.len());
        assert!(HARNESS_GUIDANCE.contains("code_search") && HARNESS_GUIDANCE.contains("Glob"));
    }

    #[test]
    fn tool_descriptions_name_the_builtins_they_replace() {
        let (_t, e) = fixture_engine();
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), r#"{"jsonrpc":"2.0","id":51,"method":"tools/list"}"#).unwrap());
        let desc = |n: &str| {
            v["result"]["tools"].as_array().unwrap().iter().find(|t| t["name"] == n).unwrap()["description"]
                .as_str().unwrap().to_string()
        };
        assert!(desc("code_search").contains("INSTEAD OF grep"));
        assert!(desc("list_files").contains("INSTEAD OF Glob"));
        assert!(desc("read_span").contains("INSTEAD OF reading a whole file"));
        assert!(desc("file_outline").contains("BEFORE READING A FILE"));
        assert!(desc("refresh_index").contains("stale"));
    }

    #[test]
    fn text_format_renders_grouped_hits() {
        let (_t, e) = fixture_engine();
        let out = handle_text(&e, r#"{"jsonrpc":"2.0","id":60,"method":"tools/call","params":{"name":"find_symbol","arguments":{"name":"foo_bar_123"}}}"#).unwrap();
        let v = parse_resp(&out);
        let text = v["result"]["content"][0]["text"].as_str().unwrap();
        // definition rows carry their end line (SPEC-P9): the fn spans lines 1-3
        assert_eq!(text, "alpha:src/foo.rs\n  1-3: fn foo_bar_123(x: i32) -> i32 {\n-- 1 definitions in 1 file");
        // read_span without `end` returns exactly that definition
        let out = handle_text(&e, r#"{"jsonrpc":"2.0","id":62,"method":"tools/call","params":{"name":"read_span","arguments":{"repo":"alpha","path":"src/foo.rs","start":1}}}"#).unwrap();
        let v = parse_resp(&out);
        let span = v["result"]["content"][0]["text"].as_str().unwrap();
        assert!(span.starts_with("alpha:src/foo.rs L1-3\n1\tfn foo_bar_123"), "{span}");
        assert_eq!(span.lines().count(), 4, "{span}");
        // per-call override back to JSON
        let out = handle_text(&e, r#"{"jsonrpc":"2.0","id":61,"method":"tools/call","params":{"name":"find_symbol","arguments":{"name":"foo_bar_123","format":"json"}}}"#).unwrap();
        let v = parse_resp(&out);
        let p: Value = serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(p[0]["path"], "src/foo.rs");
    }

    #[test]
    fn tools_call_list_files_glob_substring_repo() {
        let (_t, e) = fixture_engine();
        let call = |args: &str| {
            let req = format!(r#"{{"jsonrpc":"2.0","id":52,"method":"tools/call","params":{{"name":"list_files","arguments":{args}}}}}"#);
            let v = parse_resp(&handle(&e, &hash_emb(), &rr(), &req).unwrap());
            assert!(v.get("error").is_none(), "{v}");
            let p = payload_of(&v);
            p["files"].as_array().unwrap().iter().map(|f| f["path"].as_str().unwrap().to_string()).collect::<Vec<_>>()
        };
        assert_eq!(call(r#"{"pattern":"**/*.rs"}"#), vec!["src/caller.rs", "src/foo.rs"]);
        assert_eq!(call(r#"{"pattern":"src/foo*"}"#), vec!["src/foo.rs"]);
        assert_eq!(call(r#"{"pattern":"CALLER"}"#), vec!["src/caller.rs"], "substring, case-insensitive");
        assert_eq!(call(r#"{"pattern":"*.rs"}"#), Vec::<String>::new(), "single star does not cross '/'");
        assert_eq!(call(r#"{"pattern":"**/*.rs","repo":"nope"}"#), Vec::<String>::new());
        assert_eq!(call(r#"{"pattern":"**/*.rs","repo":"alpha","limit":1}"#).len(), 1);
        // no pattern at all lists everything (a worker sent `{}`; an error
        // cost it the turn)
        assert_eq!(call("{}"), vec!["src/caller.rs", "src/foo.rs"]);
        let req = r#"{"jsonrpc":"2.0","id":53,"method":"tools/call","params":{"name":"list_files","arguments":{"pattern":7}}}"#;
        assert_eq!(parse_resp(&handle(&e, &hash_emb(), &rr(), req).unwrap())["error"]["code"], -32602);
    }

    /// SPEC-P10 §23: a server started outside any registered repo adopts
    /// the repo its working directory lands in once that repo is
    /// registered; a subdirectory counts, an unrelated folder does not.
    #[test]
    fn server_adopts_a_repo_registered_after_it_started() {
        let (t, e) = fixture_engine();
        let dir = e.data_dir().unwrap().to_path_buf();
        let server = McpServer::with_format(
            Engine::open(&dir).unwrap(),
            Arc::new(hash_emb()),
            Arc::new(rr()),
            OutputFormat::Text,
        )
        .with_current_repo(None);
        let root = t.path().join("checkout");
        std::fs::create_dir_all(root.join("src")).unwrap();
        server.adopt_repo_at(&root.join("src"));
        assert_eq!(server.current(), None, "nothing registered yet");
        // register it (what `indexio add` writes)
        std::fs::create_dir_all(dir.join("repos")).unwrap();
        std::fs::write(
            dir.join("repos").join("alpha.json"),
            serde_json::to_string(&indexio_ingest::RepoState {
                name: "alpha".into(),
                path: root.clone(),
                last_commit: None,
                indexed_at: String::new(),
                plain: true,
                worktree: false,
            })
            .unwrap(),
        )
        .unwrap();
        server.adopt_repo_at(&t.path().join("elsewhere"));
        assert_eq!(server.current(), None, "an unrelated folder");
        server.adopt_repo_at(&root.join("src"));
        assert_eq!(server.current().as_deref(), Some("alpha"));
        assert!(server.watch.lock().unwrap().is_some(), "auto-refresh watch started");
        // the repo now leads the initialize instructions and usage rows
        let v = parse_resp(&server.handle(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#).unwrap());
        assert!(v["result"]["instructions"].as_str().unwrap().contains("alpha"));
    }

    #[test]
    fn index_stats_reports_freshness_and_refresh_index_reloads() {
        let (_t, e) = fixture_engine();
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), r#"{"jsonrpc":"2.0","id":54,"method":"tools/call","params":{"name":"index_stats","arguments":{}}}"#).unwrap());
        let p = payload_of(&v);
        assert_eq!(p["repos"], json!(["alpha"]));
        assert!(p["repos_detail"].is_array(), "{p}");
        // No registered repos in the fixture (shards only): refresh is a
        // harmless no-op that still reloads the engine.
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), r#"{"jsonrpc":"2.0","id":55,"method":"tools/call","params":{"name":"refresh_index","arguments":{}}}"#).unwrap());
        assert!(v.get("error").is_none(), "{v}");
        let p = payload_of(&v);
        assert_eq!(p["reloaded"], true);
        assert_eq!(p["repos"].as_array().unwrap().len(), 0);
        let v = parse_resp(&handle(&e, &hash_emb(), &rr(), r#"{"jsonrpc":"2.0","id":56,"method":"tools/call","params":{"name":"refresh_index","arguments":{"embed":"yes"}}}"#).unwrap());
        assert_eq!(v["error"]["code"], -32602);
    }
}
