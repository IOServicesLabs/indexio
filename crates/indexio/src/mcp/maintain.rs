//! Heavy maintenance of a data dir: the shard merge, the semantic segment
//! compaction and the full re-embed after model drift.
//!
//! A pass is short and large: gigabytes of transient memory for seconds to
//! minutes. Run inside a long-lived MCP server, the peak became that server's
//! committed memory for the rest of its life (the process heap keeps most of
//! what it frees), and since the lock only decides which server runs a pass,
//! over a day every server had run some. The server therefore hands a due
//! pass to a short-lived child, `indexio maintain`, which takes the
//! cross-process lock itself, works with every CPU, and exits: the peak and
//! whatever the heap would have kept leave with the process. The server
//! finds the results the way it finds any other process's writes (the shards
//! stamp, the sidecars' segment stamps).
//!
//! [`run`] is the pass itself and is the same code in the child and in the
//! server's in-process fallback (the child could not be started, or
//! `INDEXIO_MAINT_INPROCESS` is set).

use std::path::Path;
use std::time::{Duration, Instant};

use indexio_embed::embed::Embedder;
use indexio_embed::pipeline::{self, CompactLock};

use super::{shards_stamp, COMPACT_ABOVE, COMPACT_TO};

/// Exit status of `indexio maintain` when another process holds the lock.
pub(crate) const EXIT_BUSY: i32 = 3;

/// How long a maintenance process may run before it gives up (a pass takes
/// seconds to a few minutes; one that is still going after this is stuck,
/// and while it holds the lock nobody else compacts).
const CHILD_DEADLINE: Duration = Duration::from_secs(3600);

/// What a pass has to do.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Due {
    /// More than [`COMPACT_ABOVE`] shards.
    pub shards: bool,
    /// Too many semantic segments, or too many tombstoned rows.
    pub vectors: bool,
    /// The model drifted from the state the rows were embedded with.
    pub reembed: bool,
}

impl Due {
    pub(crate) fn any(self) -> bool {
        self.shards || self.vectors || self.reembed
    }
}

/// What is due right now: a directory listing and the segment stats, nothing
/// is opened.
pub(crate) fn due(data_dir: &Path, model_id: &str, texts_seen: Option<u64>) -> Due {
    Due {
        shards: shards_stamp(&data_dir.join("shards")).0 > COMPACT_ABOVE,
        vectors: pipeline::needs_compaction(data_dir, model_id),
        reembed: pipeline::reembed_due(data_dir, model_id, texts_seen),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// Another process holds the compaction lock.
    Busy,
    /// Nothing was due once the lock was held (another process just did it).
    Nothing,
    /// The work that was due and was attempted.
    Done(Due),
}

/// One maintenance pass in this process: take the cross-process lock, look
/// again at what is due (the decision to start was made without the lock,
/// possibly in another process), and do it. `model_id` names the semantic
/// plane; a re-embed needs `embedder` to be that model.
pub(crate) fn run(data_dir: &Path, embedder: &dyn Embedder, model_id: &str) -> Outcome {
    let Some(_lock) = CompactLock::try_acquire(data_dir) else {
        tracing::debug!("compaction skipped: another process holds the lock");
        return Outcome::Busy;
    };
    let t0 = Instant::now();
    let texts_seen = if embedder.model_id() == model_id { embedder.texts_seen() } else { None };
    let d = due(data_dir, model_id, texts_seen);
    if !d.any() {
        return Outcome::Nothing;
    }
    let shards_dir = data_dir.join("shards");
    if d.shards {
        let n = shards_stamp(&shards_dir).0;
        match indexio_index::ShardSet::open_dir(&shards_dir) {
            Ok(mut set) => match set.merge(&shards_dir, COMPACT_TO) {
                Ok(()) => tracing::info!(from = n, to = COMPACT_TO, ms = t0.elapsed().as_millis() as u64, "shards compacted"),
                Err(e) => tracing::warn!(error = %e, "shard compaction failed"),
            },
            Err(e) => tracing::warn!(error = %e, "shard compaction: cannot open shards"),
        }
    }
    if d.reembed {
        match indexio_index::ShardSet::open_dir(&shards_dir)
            .map_err(anyhow::Error::from)
            .and_then(|set| pipeline::embed_all(&set, data_dir, embedder))
        {
            Ok(reps) => tracing::info!(
                rows = reps.iter().map(|r| r.chunks).sum::<u64>(),
                ms = t0.elapsed().as_millis() as u64,
                "semantic plane re-embedded (model drift)"
            ),
            Err(e) => tracing::warn!(error = %e, "re-embed failed"),
        }
    } else if d.vectors {
        match pipeline::compact_vectors(data_dir, model_id) {
            Ok(true) => tracing::info!(ms = t0.elapsed().as_millis() as u64, "vector segments compacted"),
            Ok(false) => {}
            Err(e) => tracing::warn!(error = %e, "vector compaction failed"),
        }
    }
    Outcome::Done(d)
}

/// Whether the server keeps maintenance in its own process:
/// `INDEXIO_MAINT_INPROCESS` (any value but `0`), for A/B runs and as the
/// way back should the child misbehave somewhere. Unit tests always do: their
/// `current_exe` is the test runner, not indexio.
pub(crate) fn in_process() -> bool {
    cfg!(test) || std::env::var("INDEXIO_MAINT_INPROCESS").is_ok_and(|v| v != "0")
}

/// Start `indexio --data-dir D maintain --model M` cut loose from this
/// server: no inherited stdio (stdout is the JSON-RPC channel), its own
/// process group, out of the harness's job where that is allowed, and below
/// normal priority (it uses every CPU; tool calls should not wait for it).
/// The child outlives the server if the session ends meanwhile.
pub(crate) fn spawn_child(data_dir: &Path, model_id: &str) -> std::io::Result<std::process::Child> {
    use std::process::{Command, Stdio};
    let mut c = Command::new(std::env::current_exe()?);
    c.arg("--data-dir").arg(data_dir).args(["maintain", "--model", model_id]);
    c.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
        const CREATE_BREAKAWAY_FROM_JOB: u32 = 0x0100_0000;
        let flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | BELOW_NORMAL_PRIORITY_CLASS;
        c.creation_flags(flags | CREATE_BREAKAWAY_FROM_JOB);
        if let Ok(child) = c.spawn() {
            return Ok(child);
        }
        // the job does not allow breakaway
        c.creation_flags(flags);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // not in the server's process group: a signal sent to the session's
        // group must not end a pass halfway
        c.process_group(0);
    }
    c.spawn()
}

/// A pass that never ends would hold the lock for good: end the process
/// instead after [`CHILD_DEADLINE`] (`INDEXIO_MAINT_TIMEOUT_SECS`). The lock
/// goes with the process and half-written files are reaped by the next open,
/// as after any kill.
pub(crate) fn arm_child_deadline() {
    let deadline = std::env::var("INDEXIO_MAINT_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|s| *s > 0)
        .map_or(CHILD_DEADLINE, Duration::from_secs);
    let _ = std::thread::Builder::new().name("maintain-deadline".into()).spawn(move || {
        std::thread::sleep(deadline);
        tracing::error!(secs = deadline.as_secs(), "maintenance pass exceeded its deadline; giving up");
        std::process::exit(4);
    });
}

/// `indexio maintain`: one pass as a process of its own. Returns the exit
/// status: 0 when the pass ran or nothing was due, [`EXIT_BUSY`] when another
/// process holds the lock.
pub(crate) fn run_child(data_dir: &Path, embedder: &dyn Embedder, model: Option<&str>) -> i32 {
    let model_id = model.unwrap_or_else(|| embedder.model_id()).to_string();
    let t0 = Instant::now();
    let shards_before = shards_stamp(&data_dir.join("shards")).0;
    let outcome = run(data_dir, embedder, &model_id);
    log_pass(data_dir, &outcome, shards_before, t0.elapsed());
    match outcome {
        Outcome::Busy => EXIT_BUSY,
        Outcome::Nothing | Outcome::Done(_) => 0,
    }
}

/// One JSON line per pass that did something in `<data_dir>/maintain.log`:
/// what ran, how long it took and the process's peak memory, which no longer
/// shows in any server. Best effort; rolled over at 256 KiB.
fn log_pass(data_dir: &Path, outcome: &Outcome, shards_before: usize, took: Duration) {
    let Outcome::Done(d) = outcome else { return };
    let path = data_dir.join("maintain.log");
    if std::fs::metadata(&path).is_ok_and(|m| m.len() > 256 * 1024) {
        let _ = std::fs::rename(&path, data_dir.join("maintain.log.1"));
    }
    let mem = crate::sysmem::process_memory().unwrap_or_default();
    let mb = |b: u64| (b as f64 / 1048576.0).round() as u64;
    let line = serde_json::json!({
        "ts": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |t| t.as_secs()),
        "pid": std::process::id(),
        "shards": d.shards, "vectors": d.vectors, "reembed": d.reembed,
        "shards_before": shards_before,
        "shards_after": shards_stamp(&data_dir.join("shards")).0,
        "ms": took.as_millis() as u64,
        "peak_private_mb": mb(mem.peak_private),
        "peak_resident_mb": mb(mem.peak_resident),
    });
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        use std::io::Write as _;
        let _ = writeln!(f, "{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexio_embed::embed::HashEmbedder;
    use indexio_index::ShardWriter;
    use indexio_types::{BlobId, DocMeta, ExtractedArtifact, Lang};

    /// `n` one-doc shards of repo `r` under `<dir>/shards`.
    fn write_shards(dir: &Path, n: usize) {
        let shards = dir.join("shards");
        std::fs::create_dir_all(&shards).unwrap();
        for i in 0..n {
            let mut w = ShardWriter::new(&shards).unwrap();
            let content = format!("fn shard_doc_{i}() {{}}\n");
            let path = format!("src/f{i}.rs");
            let meta = DocMeta {
                blob: BlobId::from_content(content.as_bytes()),
                repo_id: 0,
                path: path.clone(),
                lang: Lang::from_path(&path),
                raw_len: content.len() as u32,
            };
            let art = ExtractedArtifact { lang: meta.lang, raw_len: meta.raw_len, ..Default::default() };
            w.add_doc(&meta, content.as_bytes(), &art).unwrap();
            w.finish(&["r".to_string()]).unwrap();
        }
    }

    #[test]
    fn nothing_is_due_on_a_small_data_dir() {
        let tmp = tempfile::tempdir().unwrap();
        write_shards(tmp.path(), 3);
        assert_eq!(due(tmp.path(), "hash-v1", None), Due::default());
        assert_eq!(run(tmp.path(), &HashEmbedder::new(64), "hash-v1"), Outcome::Nothing);
        assert!(!tmp.path().join("compact.lock").exists(), "the pass must release the lock");
    }

    #[test]
    fn a_pass_merges_shards_and_a_second_one_finds_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        write_shards(tmp.path(), COMPACT_ABOVE + 2);
        let d = due(tmp.path(), "hash-v1", None);
        assert!(d.shards && !d.vectors && !d.reembed, "{d:?}");
        assert_eq!(run(tmp.path(), &HashEmbedder::new(64), "hash-v1"), Outcome::Done(d));
        let left = shards_stamp(&tmp.path().join("shards")).0;
        assert!(left <= COMPACT_TO, "{left} shards left");
        // every doc survived the merge
        let set = indexio_index::ShardSet::open_dir(&tmp.path().join("shards")).unwrap();
        assert_eq!(set.visible_docs().len(), COMPACT_ABOVE + 2);
        // the decision is taken again under the lock: a pass started for
        // work another process has finished meanwhile does nothing
        assert_eq!(run(tmp.path(), &HashEmbedder::new(64), "hash-v1"), Outcome::Nothing);
    }

    #[test]
    fn a_pass_stands_back_while_another_process_holds_the_lock() {
        let tmp = tempfile::tempdir().unwrap();
        write_shards(tmp.path(), COMPACT_ABOVE + 2);
        let held = CompactLock::try_acquire(tmp.path()).unwrap();
        assert_eq!(run(tmp.path(), &HashEmbedder::new(64), "hash-v1"), Outcome::Busy);
        assert_eq!(shards_stamp(&tmp.path().join("shards")).0, COMPACT_ABOVE + 2, "nothing may be merged without the lock");
        drop(held);
        assert!(matches!(run(tmp.path(), &HashEmbedder::new(64), "hash-v1"), Outcome::Done(_)));
    }

    #[test]
    fn the_child_logs_a_pass_that_did_something() {
        let tmp = tempfile::tempdir().unwrap();
        write_shards(tmp.path(), COMPACT_ABOVE + 1);
        assert_eq!(run_child(tmp.path(), &HashEmbedder::new(64), Some("hash-v1")), 0);
        let log = std::fs::read_to_string(tmp.path().join("maintain.log")).unwrap();
        let line: serde_json::Value = serde_json::from_str(log.lines().next().unwrap()).unwrap();
        assert_eq!(line["shards"], true);
        assert_eq!(line["shards_before"], (COMPACT_ABOVE + 1) as u64);
        assert!(line["shards_after"].as_u64().unwrap() <= COMPACT_TO as u64, "{line}");
        // nothing due: status 0 and no second line
        assert_eq!(run_child(tmp.path(), &HashEmbedder::new(64), Some("hash-v1")), 0);
        assert_eq!(std::fs::read_to_string(tmp.path().join("maintain.log")).unwrap().lines().count(), 1);
        // lock held elsewhere: the busy status
        let _held = CompactLock::try_acquire(tmp.path()).unwrap();
        assert_eq!(run_child(tmp.path(), &HashEmbedder::new(64), Some("hash-v1")), EXIT_BUSY);
    }
}
