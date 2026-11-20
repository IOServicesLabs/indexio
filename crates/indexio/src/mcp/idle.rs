//! What an MCP server does when nobody is calling it.
//!
//! One server runs per agent session and most of them are idle most of the
//! time, each still holding what its last burst of calls left behind: the
//! decompressed-content and outline caches, parsed semantic sidecars, an
//! engine opened ahead for a call that never came, and heap the allocator
//! keeps after the transients. Nothing ever dropped any of it, because every
//! job of the server rode on a tool call.
//!
//! * [`Activity`] records calls. The shards watch asks it whether anyone is
//!   likely to call soon before it opens an engine for another process's
//!   write (an idle server reopens on its next call instead).
//! * The idle timer ([`McpServer::run_idle_timer`]) is the server's one
//!   timer: a single thread parked on a condvar. After `idle_after` without
//!   a call it puts the server back into the state of a freshly started one
//!   ([`McpServer::trim_idle`]) and parks until the next call. It holds the
//!   activity lock while it trims, and a call takes that lock before it
//!   does anything else, so a call never runs into a trim: it either
//!   prevents it or waits the few milliseconds it takes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use indexio_query::Engine;

use super::{shards_stamp, McpServer};

/// Idle time after which a server drops what it can rebuild
/// (`INDEXIO_IDLE_TRIM_SECS`, `0` = never).
const IDLE_TRIM_AFTER: Duration = Duration::from_secs(600);

/// How recently a call must have been made for the shards watch to open an
/// engine ahead of the next one (`INDEXIO_PREPARE_WINDOW_SECS`, `0` = never
/// ahead, always on the call).
const PREPARE_WINDOW: Duration = Duration::from_secs(300);

/// How long the timer stands back when it finds a background job of the
/// server (import, sync, embed, compaction) still running.
const BUSY_RECHECK: Duration = Duration::from_secs(30);

fn env_secs(name: &str, default: Duration) -> Duration {
    match std::env::var(name).ok().and_then(|s| s.trim().parse::<u64>().ok()) {
        Some(s) => Duration::from_secs(s),
        None => default,
    }
}

/// `None` = the idle trim is off.
pub(super) fn idle_trim_after() -> Option<Duration> {
    Some(env_secs("INDEXIO_IDLE_TRIM_SECS", IDLE_TRIM_AFTER)).filter(|d| !d.is_zero())
}

pub(super) fn prepare_window() -> Duration {
    env_secs("INDEXIO_PREPARE_WINDOW_SECS", PREPARE_WINDOW)
}

/// Call activity of one server, shared with its idle timer and its shards
/// watch.
pub(super) struct Activity {
    state: Mutex<State>,
    wake: Condvar,
    epoch: Instant,
    /// Milliseconds after `epoch` of the last call's start or end, for the
    /// watch callback, which must never wait for a lock a trim holds.
    last_ms: AtomicU64,
}

struct State {
    /// Calls in flight.
    in_call: usize,
    /// When the last call ended (the server's start before the first).
    last: Instant,
    /// The trim ran and no call has been made since: nothing to do until
    /// one is.
    trimmed: bool,
    stop: bool,
}

/// A call in flight; dropping it ends the call.
pub(super) struct CallGuard<'a>(&'a Activity);

impl Drop for CallGuard<'_> {
    fn drop(&mut self) {
        let mut st = self.0.lock();
        st.in_call = st.in_call.saturating_sub(1);
        st.last = Instant::now();
        self.0.touch();
    }
}

impl Activity {
    pub(super) fn new() -> Activity {
        let now = Instant::now();
        Activity {
            state: Mutex::new(State { in_call: 0, last: now, trimmed: false, stop: false }),
            wake: Condvar::new(),
            epoch: now,
            last_ms: AtomicU64::new(0),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn touch(&self) {
        self.last_ms.store(self.epoch.elapsed().as_millis() as u64, Ordering::Release);
    }

    /// Start of a call: waits for a trim in progress, and says whether this
    /// is the first call since one (the caller then warms up again, as a
    /// new server does).
    pub(super) fn begin(&self) -> (CallGuard<'_>, bool) {
        let mut st = self.lock();
        st.in_call += 1;
        let after_trim = std::mem::take(&mut st.trimmed);
        self.touch();
        drop(st);
        if after_trim {
            // the timer is parked without a timeout: set it going again
            self.wake.notify_all();
        }
        (CallGuard(self), after_trim)
    }

    /// Whether a call started or ended within `window`.
    pub(super) fn recent(&self, window: Duration) -> bool {
        let now = self.epoch.elapsed().as_millis() as u64;
        now.saturating_sub(self.last_ms.load(Ordering::Acquire)) <= window.as_millis() as u64
    }

    /// End the idle timer (server shutdown).
    pub(super) fn stop(&self) {
        self.lock().stop = true;
        self.wake.notify_all();
    }

    #[cfg(test)]
    pub(super) fn is_trimmed(&self) -> bool {
        self.lock().trimmed
    }
}

/// What one idle trim did, for the log line and the tests.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TrimReport {
    pub ms: u64,
    /// Private bytes before and after (0 where the system does not say).
    pub private_before: u64,
    pub private_after: u64,
    /// After the caches were dropped, before the heap was asked to give
    /// memory back: the difference to `private_after` is the allocator's.
    pub private_dropped: u64,
}

impl McpServer {
    /// Test servers choose their own idle period and prepare window.
    #[cfg(test)]
    pub(super) fn with_idle(mut self, idle_after: Option<Duration>, prepare_window: Duration) -> Self {
        self.idle_after = idle_after;
        self.prepare_window = prepare_window;
        self
    }

    /// The idle timer. Runs until [`stop_idle_timer`](Self::stop_idle_timer);
    /// returns at once when the trim is off. One thread, parked: while calls
    /// keep coming it wakes at most once per idle period to look at the
    /// clock, and after a trim it sleeps without a timeout until the next
    /// call.
    pub fn run_idle_timer(&self) {
        let Some(after) = self.idle_after else { return };
        let act = &*self.activity;
        let mut st = act.lock();
        loop {
            if st.stop {
                return;
            }
            if st.trimmed {
                st = act.wake.wait(st).unwrap_or_else(|e| e.into_inner());
                continue;
            }
            let idle = if st.in_call > 0 { Duration::ZERO } else { st.last.elapsed() };
            if idle < after {
                st = act.wake.wait_timeout(st, after - idle).unwrap_or_else(|e| e.into_inner()).0;
                continue;
            }
            // a job the last call started is still writing: its memory is
            // not garbage yet, and the heap is busy
            if self.background_busy() {
                st = act.wake.wait_timeout(st, BUSY_RECHECK.min(after)).unwrap_or_else(|e| e.into_inner()).0;
                continue;
            }
            // with the activity lock held: a call that arrives now waits in
            // `begin` until the trim is done
            self.trim_idle();
            st.trimmed = true;
        }
    }

    pub fn stop_idle_timer(&self) {
        self.activity.stop();
    }

    /// Whether a background job of this server is at work. A maintenance
    /// pass in a child process is not: `compacting` stays set for it, but
    /// its memory is its own.
    fn background_busy(&self) -> bool {
        let child = self.maint_child.lock().is_ok_and(|c| c.is_some());
        let pass_here = self.compacting.load(Ordering::Acquire) && !child;
        let saving = self.saver.lock().is_ok_and(|s| s.as_ref().is_some_and(|h| !h.is_finished()));
        let warming = self.warming.0.lock().is_ok_and(|n| *n > 0);
        pass_here
            || saving
            || warming
            || [&self.embed_busy, &self.sessions_busy, &self.sync_busy, &self.prepare_busy]
                .iter()
                .any(|b| b.load(Ordering::Acquire))
    }

    /// Put the server back into the state of one that has just started:
    ///
    /// * the engine is replaced by a cold open of the same data dir, which
    ///   drops the content and outline caches, the parsed semantic sidecars,
    ///   the doc views and the mappings of shards that were merged away
    ///   since (they could not be deleted while this server mapped them);
    /// * the engines parked for a call that did not come (`warm`,
    ///   `prepared`) and the per-folder stamp caches of the background sync
    ///   (reloaded from disk by its next pass) are dropped;
    /// * the allocator is asked to return what that freed.
    ///
    /// Everything comes back on demand; the first call afterwards starts the
    /// sidecar warm-up a new server runs at start.
    pub(crate) fn trim_idle(&self) -> TrimReport {
        let t0 = Instant::now();
        self.trimming.store(true, Ordering::Release);
        let private = || crate::sysmem::process_memory().map_or(0, |m| m.private);
        let private_before = private();
        drop(self.warm.lock().unwrap_or_else(|e| e.into_inner()).take());
        drop(self.prepared.lock().unwrap_or_else(|e| e.into_inner()).take());
        if let Ok(mut caches) = self.plain_caches.try_lock() {
            caches.clear();
        }
        let data_dir = self.engine.read().expect("engine lock poisoned").data_dir().map(|d| d.to_path_buf());
        if let Some(dir) = data_dir {
            // stamped before the open: a write that lands after it still
            // shows as a change to the next call
            let stamp = shards_stamp(&dir.join("shards"));
            match Engine::open(&dir) {
                Ok(cold) => {
                    let old = std::mem::replace(&mut *self.engine.write().expect("engine lock poisoned"), cold);
                    drop(old);
                    *self.shards_seen.lock().expect("shards stamp poisoned") = stamp;
                    self.reload_needed.store(false, Ordering::Release);
                }
                Err(e) => tracing::debug!(error = %e, "idle trim: cannot reopen the index; caches kept"),
            }
        }
        let private_dropped = private();
        if self.release_heap {
            crate::sysmem::release_free_heap();
        }
        let report = TrimReport {
            ms: t0.elapsed().as_millis() as u64,
            private_before,
            private_after: private(),
            private_dropped,
        };
        self.trims.fetch_add(1, Ordering::Relaxed);
        self.trimming.store(false, Ordering::Release);
        let mb = |b: u64| b / 1048576;
        tracing::info!(
            ms = report.ms,
            private_mb_before = mb(report.private_before),
            private_mb_dropped = mb(report.private_dropped),
            private_mb_after = mb(report.private_after),
            "idle: caches dropped, freed heap returned"
        );
        #[cfg(feature = "memstat")]
        if crate::memstat::enabled() {
            crate::memstat::report("trim");
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::Arc;

    use indexio_core::grams::{self, CommonGrams};
    use indexio_index::ShardWriter;
    use indexio_types::{BlobId, DocMeta, ExtractedArtifact, Lang};

    use crate::mcp::OutputFormat;

    /// One shard of `n` small Rust files of repo `alpha` under `<dir>/shards`,
    /// named with `tag` so later shards add new files.
    fn write_shard(dir: &Path, tag: &str, n: usize) {
        let shards = dir.join("shards");
        std::fs::create_dir_all(&shards).unwrap();
        let mut w = ShardWriter::new(&shards).unwrap();
        for i in 0..n {
            let path = format!("src/{tag}_{i}.rs");
            let content = format!("/// doc {tag} {i}\npub fn {tag}_fn_{i}(x: u32) -> u32 {{\n    x + {i}\n}}\n");
            let bytes = content.as_bytes();
            let art = ExtractedArtifact {
                ngrams: grams::extract(bytes, &CommonGrams::empty()),
                raw_len: bytes.len() as u32,
                lang: Lang::Rust,
                ..Default::default()
            };
            let meta = DocMeta {
                blob: BlobId::from_content(bytes),
                repo_id: 0,
                path,
                lang: Lang::Rust,
                raw_len: bytes.len() as u32,
            };
            w.add_doc(&meta, bytes, &art).unwrap();
        }
        w.finish(&["alpha".to_string()]).unwrap();
    }

    fn server(dir: &Path, idle_after: Option<Duration>, prepare_window: Duration) -> McpServer {
        let s = McpServer::with_format(
            Engine::open(dir).unwrap(),
            Arc::new(indexio_embed::embed::HashEmbedder::new(64)),
            Arc::new(indexio_embed::rerank::OverlapReranker),
            OutputFormat::Json,
        )
        .with_idle(idle_after, prepare_window);
        // not due in these tests: the periodic import of this machine's
        // transcripts and memory notes, and the sync of other repos (both
        // would add docs half-way through, and with them change the scores)
        *s.sessions_last.lock().unwrap() = Some(Instant::now());
        *s.sync_last.lock().unwrap() = Some(Instant::now());
        s
    }

    /// The answer's text, without the one field that is a measurement.
    fn call(s: &McpServer, tool: &str, args: serde_json::Value) -> String {
        let msg = serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": args}});
        let out = s.handle(&msg.to_string()).expect("a reply");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert!(v.get("error").is_none(), "{v}");
        let text = v["result"]["content"][0]["text"].as_str().unwrap();
        match serde_json::from_str::<serde_json::Value>(text) {
            Ok(serde_json::Value::Object(mut o)) => {
                o.remove("took_ms");
                serde_json::Value::Object(o).to_string()
            }
            _ => text.to_string(),
        }
    }

    /// The calls of one "session": a read, an outline and a search.
    fn session(s: &McpServer) -> Vec<String> {
        vec![
            call(s, "read_span", serde_json::json!({"repo": "alpha", "path": "src/base_3.rs", "start": 1, "end": 4})),
            call(s, "file_outline", serde_json::json!({"repo": "alpha", "path": "src/base_5.rs"})),
            call(s, "code_search", serde_json::json!({"query": "base_fn_7", "mode": "lexical"})),
        ]
    }

    /// Stops the idle timer when the test's scope is left, by a failed
    /// assertion too: the scope waits for the timer thread.
    struct StopTimer<'a>(&'a McpServer);
    impl Drop for StopTimer<'_> {
        fn drop(&mut self) {
            self.0.stop_idle_timer();
        }
    }

    fn wait_for(what: &str, cond: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while !cond() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn activity_knows_recent_calls_and_a_call_waits_for_the_lock() {
        let act = Activity::new();
        assert!(act.recent(Duration::from_secs(5)));
        std::thread::sleep(Duration::from_millis(30));
        assert!(!act.recent(Duration::from_millis(5)), "no call was made for 30 ms");
        {
            let (_call, after_trim) = act.begin();
            assert!(!after_trim);
            assert!(act.recent(Duration::from_millis(5)));
        }
        // what the timer holds while it trims: a call must wait for it
        let held = act.lock();
        std::thread::scope(|s| {
            let caller = s.spawn(|| {
                let (_call, _) = act.begin();
            });
            std::thread::sleep(Duration::from_millis(80));
            assert!(!caller.is_finished(), "a call started while the trim's lock was held");
            drop(held);
            caller.join().unwrap();
        });
    }

    #[test]
    fn an_idle_server_is_trimmed_once_and_answers_the_same_afterwards() {
        let tmp = tempfile::tempdir().unwrap();
        write_shard(tmp.path(), "base", 12);
        let s = server(tmp.path(), Some(Duration::from_millis(60)), Duration::ZERO);
        std::thread::scope(|scope| {
            scope.spawn(|| s.run_idle_timer());
            let _stop = StopTimer(&s);
            let before = session(&s);
            // what a busy session leaves parked beside the live engine (set
            // up as part of a call, so that the timer waits for it)
            let trims = {
                let (_call, _) = s.activity.begin();
                *s.warm.lock().unwrap() = Some(Engine::open(tmp.path()).unwrap());
                *s.prepared.lock().unwrap() =
                    Some((shards_stamp(&tmp.path().join("shards")), Engine::open(tmp.path()).unwrap()));
                s.plain_caches.lock().unwrap().insert("docs".to_string(), Default::default());
                s.trims.load(Ordering::Relaxed)
            };

            wait_for("the trim", || s.trims.load(Ordering::Relaxed) == trims + 1);
            assert!(s.warm.lock().unwrap().is_none(), "the warm engine survived the trim");
            assert!(s.prepared.lock().unwrap().is_none(), "the prepared engine survived the trim");
            assert!(s.plain_caches.lock().unwrap().is_empty());
            assert!(s.activity.is_trimmed());

            // parked: no second trim however long nobody calls
            std::thread::sleep(Duration::from_millis(400));
            assert_eq!(s.trims.load(Ordering::Relaxed), trims + 1, "the timer trimmed an already trimmed server");

            // everything comes back on demand, same answers
            assert_eq!(session(&s), before);
            // and the timer is armed again by those calls
            wait_for("the next trim", || s.trims.load(Ordering::Relaxed) > trims + 1 && s.activity.is_trimmed());
            assert_eq!(session(&s), before);
        });
    }

    /// Calls and the timer at full tilt: a trim never overlaps a call (the
    /// debug assertion in `call_tool`), and no answer changes.
    #[test]
    fn calls_never_run_into_a_trim() {
        let tmp = tempfile::tempdir().unwrap();
        write_shard(tmp.path(), "base", 12);
        let s = server(tmp.path(), Some(Duration::from_millis(1)), Duration::ZERO);
        let expected = session(&s);
        std::thread::scope(|scope| {
            scope.spawn(|| s.run_idle_timer());
            let _stop = StopTimer(&s);
            let callers: Vec<_> = (0..3u64)
                .map(|t| {
                    let (s, expected) = (&s, &expected);
                    scope.spawn(move || {
                        for i in 0..150u64 {
                            assert_eq!(&session(s), expected, "caller {t}, round {i}");
                            // leave gaps of 0 to 4 ms for the timer to slip into
                            std::thread::sleep(Duration::from_micros((i * 37 + t * 11) % 5 * 1000));
                        }
                    })
                })
                .collect();
            for c in callers {
                c.join().unwrap();
            }
            wait_for("a trim after the last call", || s.activity.is_trimmed());
        });
        assert!(s.trims.load(Ordering::Relaxed) > 1, "the timer never got a turn: the test proved nothing");
    }

    #[test]
    fn the_timer_is_off_without_an_idle_period() {
        let tmp = tempfile::tempdir().unwrap();
        write_shard(tmp.path(), "base", 3);
        let s = server(tmp.path(), None, Duration::ZERO);
        s.run_idle_timer(); // returns at once
        assert_eq!(s.trims.load(Ordering::Relaxed), 0);
    }

    /// Another process writes a shard. A server called a moment ago opens
    /// an engine for it off the call path; one that was not does nothing
    /// until its next call, which finds the new docs all the same.
    #[test]
    fn only_a_recently_called_server_prepares_an_engine_for_a_shard_write() {
        for (window, expect_prepared) in [(Duration::from_secs(300), true), (Duration::ZERO, false)] {
            let tmp = tempfile::tempdir().unwrap();
            write_shard(tmp.path(), "base", 4);
            let s = server(tmp.path(), None, window);
            session_first_call(&s);
            std::thread::sleep(Duration::from_millis(20)); // "not recent" for the zero window
            write_shard(tmp.path(), "later", 2);
            if expect_prepared {
                wait_for("the prepared engine", || s.prepared.lock().unwrap().is_some());
            } else {
                std::thread::sleep(Duration::from_millis(500));
                assert!(s.prepared.lock().unwrap().is_none(), "an idle server opened an engine for a shard write");
            }
            let hits = call(&s, "code_search", serde_json::json!({"query": "later_fn_1", "mode": "lexical"}));
            assert!(hits.contains("src/later_1.rs"), "window {window:?}: {hits}");
        }
    }

    /// A warm-up is counted while it runs and leaves its engine behind; a
    /// semantic call that arrives meanwhile waits for it, any other call
    /// does not.
    #[test]
    fn a_semantic_call_waits_for_a_warm_up_in_flight() {
        let tmp = tempfile::tempdir().unwrap();
        write_shard(tmp.path(), "base", 4);
        let s = server(tmp.path(), None, Duration::ZERO);
        let in_flight = || *s.warming.0.lock().unwrap();
        s.prewarm_sidecars();
        wait_for("the warm-up", || in_flight() == 0);
        assert!(s.warm.lock().unwrap().is_some(), "the warm-up left no engine");

        // a warm-up that ends when the test says so
        *s.warming.0.lock().unwrap() += 1;
        let (go, wait) = std::sync::mpsc::channel::<()>();
        std::thread::scope(|scope| {
            let done = crate::mcp::WarmDone(Arc::clone(&s.warming));
            scope.spawn(move || {
                let _done = done;
                let _ = wait.recv();
                std::thread::sleep(Duration::from_millis(100));
            });
            let t0 = Instant::now();
            session_first_call(&s); // lexical
            assert_eq!(in_flight(), 1);
            assert!(t0.elapsed() < Duration::from_secs(2), "a lexical call waited for the warm-up");
            go.send(()).unwrap();
            let msg = serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "code_search", "arguments": {"query": "base fn", "mode": "hybrid"}}});
            s.handle(&msg.to_string()).expect("a reply");
            assert_eq!(in_flight(), 0, "the hybrid call did not wait for the warm-up");
        });
    }

    fn session_first_call(s: &McpServer) {
        // starts the shards watch
        let hits = call(s, "code_search", serde_json::json!({"query": "base_fn_1", "mode": "lexical"}));
        assert!(hits.contains("src/base_1.rs"), "{hits}");
    }

    /// The server's own refresh writes shards too, and has reopened by the
    /// time the watch fires: nothing is prepared for a state it already has.
    #[test]
    fn nothing_is_prepared_for_the_state_the_server_is_at() {
        let tmp = tempfile::tempdir().unwrap();
        write_shard(tmp.path(), "base", 4);
        let s = server(tmp.path(), None, Duration::from_secs(300));
        session_first_call(&s);
        write_shard(tmp.path(), "own", 2);
        // what `auto_refresh` does after its own write
        *s.shards_seen.lock().unwrap() = shards_stamp(&tmp.path().join("shards"));
        std::thread::sleep(Duration::from_millis(500));
        assert!(s.prepared.lock().unwrap().is_none(), "an engine was prepared for the server's own write");
    }
}
