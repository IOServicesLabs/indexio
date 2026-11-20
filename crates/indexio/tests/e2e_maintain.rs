//! End-to-end tests for maintenance in a child process: an MCP server that
//! finds a compaction pass due starts `indexio maintain` instead of merging
//! in its own process, the child takes the cross-process lock, and the
//! server picks the result up like any other process's writes. Requires
//! `git` on PATH. Fully offline (built-in embedder).

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

fn git(repo: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@t")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@t")
        .output()
        .expect("git on PATH");
    assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
}

fn indexio(data: &Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_indexio"));
    c.arg("--data-dir").arg(data).env_remove("INDEXIO_EMBED_BASE").env("INDEXIO_NO_USAGE", "1");
    // no transcripts or memory notes of this machine in the test's index
    // (the server's periodic import would add shards of its own)
    c.env("CLAUDE_CONFIG_DIR", data.join("no-claude-dir"));
    c
}

fn run_ok(c: &mut Command) -> String {
    let out = c.output().unwrap();
    assert!(out.status.success(), "{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn shard_count(data: &Path) -> usize {
    std::fs::read_dir(data.join("shards"))
        .unwrap()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "cidx"))
        .count()
}

/// A registered repo whose data dir holds `shards` shards: the first index
/// plus one delta shard per working-tree edit, as hooks, syncs and other
/// sessions' refreshes leave them.
fn data_dir_with_shards(tmp: &Path, shards: usize) -> (PathBuf, PathBuf) {
    let data = tmp.join("data");
    let repo = tmp.join("svc");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn original_thing() -> u32 {\n    1\n}\n").unwrap();
    git(&repo, &["init", "-q", "-b", "main"]);
    git(&repo, &["add", "."]);
    git(&repo, &["-c", "commit.gpgsign=false", "commit", "-qm", "init"]);
    run_ok(indexio(&data).args(["add", repo.to_str().unwrap()]));
    let mut i = 0;
    while shard_count(&data) < shards {
        std::fs::write(repo.join(format!("src/edit_{i}.rs")), format!("pub fn edited_thing_{i}() -> u32 {{\n    {i}\n}}\n")).unwrap();
        run_ok(indexio(&data).args(["reindex", "--repo", "svc", "--worktree"]));
        i += 1;
        assert!(i < 100, "edits do not add shards");
    }
    (data, repo)
}

fn maintain_log(data: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(data.join("maintain.log"))
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

struct Mcp {
    child: std::process::Child,
    /// `None` once closed.
    stdin: Option<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
    id: u64,
}

impl Mcp {
    fn start(data: &Path, cwd: &Path, env: &[(&str, &str)]) -> Mcp {
        let mut c = indexio(data);
        c.arg("mcp").current_dir(cwd).env("INDEXIO_MCP_FORMAT", "json");
        for (k, v) in env {
            c.env(k, v);
        }
        let mut child = c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
        let stdin = child.stdin.take();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut m = Mcp { child, stdin, stdout, id: 0 };
        m.call("initialize", serde_json::json!({}));
        m
    }

    fn call(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        self.id += 1;
        let msg = serde_json::json!({"jsonrpc": "2.0", "id": self.id, "method": method, "params": params});
        let stdin = self.stdin.as_mut().expect("stdin is closed");
        writeln!(stdin, "{msg}").unwrap();
        stdin.flush().unwrap();
        loop {
            let mut line = String::new();
            assert!(self.stdout.read_line(&mut line).unwrap() > 0, "server closed");
            let v: serde_json::Value = serde_json::from_str(&line).unwrap();
            if v["id"] == self.id {
                return v;
            }
        }
    }

    fn tool(&mut self, name: &str, args: serde_json::Value) -> serde_json::Value {
        let v = self.call("tools/call", serde_json::json!({"name": name, "arguments": args}));
        assert!(v.get("error").is_none(), "{v}");
        serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap()).unwrap()
    }

    fn finds(&mut self, symbol: &str) -> bool {
        !self.tool("find_symbol", serde_json::json!({"name": symbol})).as_array().unwrap().is_empty()
    }
}

impl Drop for Mcp {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

fn wait_for(what: &str, secs: u64, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(secs);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// `indexio maintain` on its own: merges what is due under the lock, logs
/// the pass, releases the lock, and finds nothing the second time.
#[test]
fn the_child_merges_due_shards_under_the_lock() {
    let tmp = tempfile::tempdir().unwrap();
    let (data, _repo) = data_dir_with_shards(tmp.path(), 14);
    run_ok(indexio(&data).arg("maintain"));
    let left = shard_count(&data);
    assert!(left <= 6, "{left} shards left");
    assert!(!data.join("compact.lock").exists(), "the pass left its lock behind");
    let log = maintain_log(&data);
    assert_eq!(log.len(), 1, "{log:?}");
    assert_eq!(log[0]["shards"], true);
    assert_eq!(log[0]["shards_before"], 14);
    // every doc is still there
    let hits = run_ok(indexio(&data).args(["search", "edited_thing_3"]));
    assert!(hits.contains("src/edit_3.rs"), "{hits}");
    let hits = run_ok(indexio(&data).args(["search", "original_thing"]));
    assert!(hits.contains("src/lib.rs"), "{hits}");
    // nothing due: no work, no log line
    run_ok(indexio(&data).arg("maintain"));
    assert_eq!(shard_count(&data), left);
    assert_eq!(maintain_log(&data).len(), 1);
}

/// A lock another process holds (here: one an older server created, which
/// is the file and nothing else) makes the child stand back with status 3
/// and touch nothing; `indexio compact` waits for it instead, and a lock
/// whose holder died long ago is taken over.
#[test]
fn a_held_lock_stops_the_child_and_a_stale_one_is_taken_over() {
    let tmp = tempfile::tempdir().unwrap();
    let (data, _repo) = data_dir_with_shards(tmp.path(), 14);
    let lock = data.join("compact.lock");
    std::fs::write(&lock, "4242").unwrap();
    let out = indexio(&data).arg("maintain").output().unwrap();
    assert_eq!(out.status.code(), Some(3), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(shard_count(&data), 14, "merged without the lock");
    assert!(maintain_log(&data).is_empty());
    assert_eq!(std::fs::read_to_string(&lock).unwrap(), "4242", "the other holder's lock was touched");

    // `compact` waits; the holder finishes (removes its lock) a moment later
    let releaser = {
        let lock = lock.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(1500));
            std::fs::remove_file(lock).unwrap();
        })
    };
    let t0 = Instant::now();
    let out = indexio(&data).args(["compact", "--max-shards", "10"]).output().unwrap();
    releaser.join().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(t0.elapsed() >= Duration::from_millis(1400), "compact did not wait for the lock");
    assert!(String::from_utf8_lossy(&out.stderr).contains("waiting"), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(shard_count(&data), 10);
    assert!(!lock.exists());

    // a holder that died 20 minutes ago
    std::fs::write(&lock, "4242").unwrap();
    let f = std::fs::OpenOptions::new().write(true).open(&lock).unwrap();
    f.set_modified(SystemTime::now() - Duration::from_secs(20 * 60)).unwrap();
    drop(f);
    run_ok(indexio(&data).args(["compact", "--max-shards", "4"]));
    assert_eq!(shard_count(&data), 4);
    assert!(!lock.exists());
}

/// The server's side: a call finds the pass due and a child does it. The
/// log line (only a `maintain` process writes one) carries a pid that is
/// not the server's, the merged index answers through the server's normal
/// reload, and the server exits without waiting for anything.
#[test]
fn a_server_hands_a_due_pass_to_a_child_process() {
    let tmp = tempfile::tempdir().unwrap();
    let (data, repo) = data_dir_with_shards(tmp.path(), 14);
    let mut mcp = Mcp::start(&data, &repo, &[]);
    assert!(mcp.finds("edited_thing_2"));
    wait_for("the child's pass", 60, || shard_count(&data) <= 6 && maintain_log(&data).len() == 1);
    let log = maintain_log(&data);
    assert_ne!(log[0]["pid"].as_u64().unwrap(), u64::from(mcp.child.id()), "the server merged in its own process");
    assert_eq!(log[0]["shards_before"], 14);
    wait_for("the lock's release", 20, || !data.join("compact.lock").exists());
    // the server sees the merged shards on its next calls
    for sym in ["edited_thing_0", "edited_thing_7", "original_thing"] {
        assert!(mcp.finds(sym), "{sym} is gone after the merge");
    }
    // a later edit is still picked up (auto-refresh on the merged index)
    std::fs::write(repo.join("src/after.rs"), "pub fn after_the_merge() -> u32 {\n    9\n}\n").unwrap();
    std::thread::sleep(Duration::from_millis(1500)); // the watcher's debounce
    assert!(mcp.finds("after_the_merge"));
    // closing stdin ends the server
    drop(mcp.stdin.take());
    wait_for("the server's exit", 30, || mcp.child.try_wait().unwrap().is_some());
}

/// `INDEXIO_MAINT_INPROCESS=1` keeps the pass in the server (the A/B switch
/// and the fallback path): same result, no child, no log line.
#[test]
fn the_in_process_switch_keeps_the_pass_in_the_server() {
    let tmp = tempfile::tempdir().unwrap();
    let (data, repo) = data_dir_with_shards(tmp.path(), 14);
    let mut mcp = Mcp::start(&data, &repo, &[("INDEXIO_MAINT_INPROCESS", "1")]);
    assert!(mcp.finds("edited_thing_2"));
    wait_for("the in-process pass", 60, || shard_count(&data) <= 6 && !data.join("compact.lock").exists());
    assert!(maintain_log(&data).is_empty(), "a child ran although the pass was forced in-process");
    for sym in ["edited_thing_0", "edited_thing_7", "original_thing"] {
        assert!(mcp.finds(sym), "{sym} is gone after the merge");
    }
}

/// Two servers on one data dir find the same pass due at the same moment:
/// one pass runs, whichever child gets the lock.
#[test]
fn two_servers_deciding_at_once_run_one_pass() {
    let tmp = tempfile::tempdir().unwrap();
    let (data, repo) = data_dir_with_shards(tmp.path(), 14);
    let mut a = Mcp::start(&data, &repo, &[]);
    let mut b = Mcp::start(&data, &repo, &[]);
    assert!(a.finds("edited_thing_2"));
    assert!(b.finds("edited_thing_2"));
    wait_for("the pass", 60, || shard_count(&data) <= 6 && !data.join("compact.lock").exists());
    std::thread::sleep(Duration::from_millis(1500)); // a second child, had one been started, has run by now
    assert_eq!(maintain_log(&data).len(), 1, "{:?}", maintain_log(&data));
    for sym in ["edited_thing_0", "edited_thing_7", "original_thing"] {
        assert!(a.finds(sym) && b.finds(sym), "{sym} is gone after the merge");
    }
}
