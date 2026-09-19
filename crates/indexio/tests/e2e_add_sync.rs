//! End-to-end test for SPEC-P7 onboarding: `indexio add <folder>` discovers git
//! repos at any depth (plus a plain folder), `indexio sync` delta re-indexes
//! after commits and plain-file edits, `indexio sources` / `indexio remove` manage
//! the sources file. Requires `git` on PATH. Fully offline.

use std::path::Path;
use std::process::Command;

fn ci(data_dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_indexio"))
        .arg("--data-dir")
        .arg(data_dir)
        .args(args)
        .env_remove("INDEXIO_EMBED_BASE")
        .env_remove("GITHUB_TOKEN")
        .output()
        .unwrap()
}

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

fn make_repo(root: &Path, file: &str, content: &str) {
    std::fs::create_dir_all(root.join(Path::new(file).parent().unwrap())).unwrap();
    std::fs::write(root.join(file), content).unwrap();
    git(root, &["init", "-q", "-b", "main"]);
    git(root, &["add", "."]);
    git(root, &["-c", "commit.gpgsign=false", "commit", "-qm", "init"]);
}

fn stdout(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn add_folder_discovers_repos_then_sync_is_incremental() {
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("data");
    let code = tmp.path().join("code");
    // two git repos at different depths, one nested under a team folder,
    // plus a repo inside node_modules (must be ignored)
    make_repo(&code.join("api"), "src/main.py", "def handler():\n    return 1\n");
    make_repo(&code.join("team/web"), "src/app.ts", "export function boot() { return 1; }\n");
    make_repo(&code.join("node_modules/dep"), "index.js", "module.exports = 1;\n");

    let out = ci(&data, &["add", code.to_str().unwrap(), "--no-embed"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = stdout(&out);
    assert!(text.contains("added source"), "{text}");
    assert!(text.contains("2 repos"), "{text}");
    assert!(text.contains("api") && text.contains("web"), "{text}");
    assert!(!text.contains("dep"), "node_modules repo must be ignored: {text}");
    assert!(text.contains("ready. try:"), "{text}");

    let stats: serde_json::Value = serde_json::from_slice(&ci(&data, &["stats", "--json"]).stdout).unwrap();
    let mut repos: Vec<String> = stats["repos"].as_array().unwrap().iter().map(|r| r.as_str().unwrap().to_string()).collect();
    repos.sort();
    assert_eq!(repos, vec!["api".to_string(), "web".to_string()]);

    // sources + registered repos are listed
    let text = stdout(&ci(&data, &["sources"]));
    assert!(text.contains("sources (1):"), "{text}");
    assert!(text.contains("registered repos (2):"), "{text}");

    // commit a change in one repo, add a brand-new repo: sync picks up both
    std::fs::write(code.join("api/src/main.py"), "def handler():\n    return 2\n").unwrap();
    git(&code.join("api"), &["-c", "commit.gpgsign=false", "commit", "-qam", "change"]);
    make_repo(&code.join("tools/cli"), "main.go", "package main\n");
    let out = ci(&data, &["sync", "--no-embed"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = stdout(&out);
    assert!(text.contains("3 repos"), "{text}");
    // api: 1 added (new version) 1 deleted (old version); web unchanged; cli new
    assert!(text.lines().any(|l| l.contains("api") && l.contains("+1 -1 =0")), "{text}");
    assert!(text.lines().any(|l| l.contains("web") && l.contains("+0 -0 =1")), "{text}");
    assert!(text.lines().any(|l| l.contains("cli") && l.contains("+1 -0 =0")), "{text}");

    // second sync is a no-op
    let text = stdout(&ci(&data, &["sync", "--no-embed"]));
    assert!(text.contains("0 files added, 0 deleted, 3 unchanged"), "{text}");

    // plain folder (no git) as a source: indexed as one repo, delta by hash
    let plain = tmp.path().join("notes-code");
    std::fs::create_dir_all(plain.join("lib")).unwrap();
    std::fs::write(plain.join("lib/util.py"), "def util():\n    pass\n").unwrap();
    let out = ci(&data, &["add", plain.to_str().unwrap(), "--no-embed"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout(&out).contains("notes-code"), "{}", stdout(&out));
    let text = stdout(&ci(&data, &["sources"]));
    assert!(text.contains("[plain folder]"), "{text}");
    std::fs::write(plain.join("lib/more.py"), "x = 1\n").unwrap();
    let text = stdout(&ci(&data, &["sync", "--no-embed"]));
    assert!(text.lines().any(|l| l.contains("notes-code") && l.contains("+1 -0 =1")), "{text}");

    // plain `indexio index` of a folder without git also works
    let other = tmp.path().join("loose");
    std::fs::create_dir_all(&other).unwrap();
    std::fs::write(other.join("a.rs"), "fn a() {}\n").unwrap();
    let out = ci(&data, &["index", other.to_str().unwrap(), "--name", "loose"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    // ...and `indexio sync` re-indexes it even though no source owns it
    let text = stdout(&ci(&data, &["sync", "--no-embed"]));
    assert!(text.contains("outside any source"), "{text}");
    assert!(text.lines().any(|l| l.contains("loose") && l.contains("=1")), "{text}");

    // remove + unknown source errors
    let out = ci(&data, &["remove", code.to_str().unwrap()]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let out = ci(&data, &["remove", "github:nobody-here"]);
    assert!(!out.status.success());
    let text = stdout(&ci(&data, &["sources"]));
    assert!(text.contains("sources (1):"), "{text}");
    // a bad source is rejected up front
    let out = ci(&data, &["add", "/no/such/folder/at/all"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("not a folder"));

    // MCP refresh_index picks up a new commit without restarting the server,
    // and the instructions list the indexed repos.
    std::fs::write(code.join("team/web/src/fresh.ts"), "export function freshlyCommittedFn() {}
").unwrap();
    git(&code.join("team/web"), &["add", "."]);
    git(&code.join("team/web"), &["-c", "commit.gpgsign=false", "commit", "-qm", "fresh"]);
    let msgs = [
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#.to_string(),
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"find_symbol","arguments":{"name":"freshlyCommittedFn"}}}"#.to_string(),
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"refresh_index","arguments":{"embed":false}}}"#.to_string(),
        r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"find_symbol","arguments":{"name":"freshlyCommittedFn"}}}"#.to_string(),
        r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"list_files","arguments":{"pattern":"**/*.ts","repo":"web"}}}"#.to_string(),
    ];
    let mut child = Command::new(env!("CARGO_BIN_EXE_indexio"))
        .arg("--data-dir").arg(&data).arg("mcp")
        .env_remove("INDEXIO_EMBED_BASE")
        .env("INDEXIO_MCP_FORMAT", "json")
        .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped())
        .spawn().unwrap();
    {
        use std::io::Write as _;
        let mut stdin = child.stdin.take().unwrap();
        for m in &msgs { writeln!(stdin, "{m}").unwrap(); }
    }
    let out = child.wait_with_output().unwrap();
    let lines: Vec<serde_json::Value> = String::from_utf8_lossy(&out.stdout).lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines.len(), 5, "{:?}", lines);
    let ins = lines[0]["result"]["instructions"].as_str().unwrap();
    assert!(ins.contains("web") && ins.contains("refresh_index"), "{ins}");
    let payload = |i: usize| -> serde_json::Value { serde_json::from_str(lines[i]["result"]["content"][0]["text"].as_str().unwrap()).unwrap() };
    assert_eq!(payload(1).as_array().unwrap().len(), 0, "not indexed before refresh");
    let r = payload(2);
    assert_eq!(r["reloaded"], true);
    assert_eq!(r["changed_repos"], serde_json::json!(["web"]), "{r}");
    assert_eq!(payload(3)[0]["path"], "src/fresh.ts", "visible after refresh, same process");
    let files: Vec<String> = payload(4)["files"].as_array().unwrap().iter().map(|f| f["path"].as_str().unwrap().to_string()).collect();
    assert_eq!(files, vec!["src/app.ts".to_string(), "src/fresh.ts".to_string()]);
}
