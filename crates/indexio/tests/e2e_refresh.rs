//! End-to-end test for the MCP server's working-tree auto-refresh (SPEC-P9)
//! with the semantic plane embedded off the call path (SPEC-P10 §37): an
//! edit is visible to the lexical tools on the very next call, and to
//! semantic search shortly after, without a refresh_index call. Requires
//! `git` on PATH. Fully offline (built-in embedder).

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

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

struct Mcp {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    id: u64,
}

impl Mcp {
    fn start(data: &Path, cwd: &Path) -> Mcp {
        let mut child = Command::new(env!("CARGO_BIN_EXE_indexio"))
            .arg("--data-dir")
            .arg(data)
            .arg("mcp")
            .current_dir(cwd)
            .env_remove("INDEXIO_EMBED_BASE")
            .env("INDEXIO_MCP_FORMAT", "json")
            .env("INDEXIO_NO_USAGE", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut m = Mcp { child, stdin, stdout, id: 0 };
        m.call("initialize", serde_json::json!({}));
        m
    }

    fn call(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        self.id += 1;
        let msg = serde_json::json!({"jsonrpc": "2.0", "id": self.id, "method": method, "params": params});
        writeln!(self.stdin, "{msg}").unwrap();
        self.stdin.flush().unwrap();
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
}

impl Drop for Mcp {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

#[test]
fn an_edit_is_lexically_visible_on_the_next_call_and_semantically_soon_after() {
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("data");
    let repo = tmp.path().join("svc");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn original_thing() -> u32 {\n    1\n}\n").unwrap();
    git(&repo, &["init", "-q", "-b", "main"]);
    git(&repo, &["add", "."]);
    git(&repo, &["-c", "commit.gpgsign=false", "commit", "-qm", "init"]);
    let out = Command::new(env!("CARGO_BIN_EXE_indexio"))
        .arg("--data-dir")
        .arg(&data)
        .args(["add", repo.to_str().unwrap()])
        .env_remove("INDEXIO_EMBED_BASE")
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    let mut mcp = Mcp::start(&data, &repo);
    let before = mcp.tool("find_symbol", serde_json::json!({"name": "freshly_added_helper"}));
    assert_eq!(before.as_array().unwrap().len(), 0, "{before}");

    // an uncommitted edit, as an agent's Write tool would make it
    std::fs::write(
        repo.join("src/fresh.rs"),
        "/// Coalesces retry backoff jitter for the outbound mailer queue.\npub fn freshly_added_helper() -> u32 {\n    2\n}\n",
    )
    .unwrap();
    std::thread::sleep(Duration::from_millis(1500)); // the watcher's debounce

    // lexical: fresh on this very call
    let after = mcp.tool("find_symbol", serde_json::json!({"name": "freshly_added_helper"}));
    assert_eq!(after[0]["path"], "src/fresh.rs", "{after}");

    // semantic: the background embed lands within a few calls
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let hits = mcp.tool("code_search", serde_json::json!({"query": "retry backoff jitter mailer queue", "mode": "semantic", "limit": 5}));
        let found = hits["hits"].as_array().unwrap().iter().any(|h| h["path"] == "src/fresh.rs");
        if found {
            break;
        }
        assert!(Instant::now() < deadline, "semantic plane never caught up: {hits}");
        std::thread::sleep(Duration::from_millis(250));
    }
}
