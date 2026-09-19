//! End-to-end CLI test for SPEC-P6: a real git repo is created and
//! indexed (`indexio index`), a function body is edited in the working tree,
//! and `indexio impact --diff` must map the uncommitted change to the edited
//! definition and list its callers. Also covers `indexio outline`, `indexio span`,
//! `indexio impact --symbol` and `indexio impact --file`. Requires `git` on PATH.

use std::path::Path;
use std::process::Command;

fn ci(data_dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_indexio"))
        .arg("--data-dir")
        .arg(data_dir)
        .args(args)
        .env_remove("INDEXIO_EMBED_BASE")
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
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

const CONFIG_RS: &str = "pub fn parse_config(s: &str) -> u32 {\n    s.len() as u32\n}\n\npub fn load(p: &str) -> u32 {\n    parse_config(p)\n}\n";
const MAIN_RS: &str = "mod config;\nuse config::load;\n\nfn main() {\n    let n = load(\"x\");\n    println!(\"{n}\");\n}\n";

fn make_repo(root: &Path) {
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/config.rs"), CONFIG_RS).unwrap();
    std::fs::write(root.join("src/main.rs"), MAIN_RS).unwrap();
    git(root, &["init", "-q", "-b", "main"]);
    git(root, &["add", "."]);
    git(root, &["-c", "commit.gpgsign=false", "commit", "-qm", "init"]);
}

#[test]
fn impact_diff_outline_span_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("data");
    let repo = tmp.path().join("core");
    make_repo(&repo);

    let out = ci(&data, &["index", repo.to_str().unwrap(), "--name", "core"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    // outline: two fns with ranges
    let out = ci(&data, &["outline", "core:src/config.rs", "--json"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let items = v["items"].as_array().unwrap();
    assert_eq!(items.len(), 2, "{v}");
    assert_eq!(items[0]["name"], "parse_config");
    assert_eq!((items[0]["start_line"].as_u64(), items[0]["end_line"].as_u64()), (Some(1), Some(3)));

    // span: exact lines from the index
    let out = ci(&data, &["span", "core:src/config.rs", "5:6"]);
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "pub fn load(p: &str) -> u32 {\n    parse_config(p)\n"
    );

    // symbol impact: parse_config <- load <- main
    let out = ci(&data, &["impact", "--symbol", "parse_config", "--depth", "2", "--json"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let sites = v["sites"].as_array().unwrap();
    assert!(sites.iter().any(|s| s["path"] == "src/config.rs" && s["caller"] == "load" && s["depth"] == 1), "{v}");
    assert!(sites.iter().any(|s| s["path"] == "src/main.rs" && s["caller"] == "main" && s["depth"] == 2), "{v}");

    // Edit the working tree (uncommitted): change parse_config's body.
    std::fs::write(
        repo.join("src/config.rs"),
        CONFIG_RS.replace("s.len() as u32", "s.len() as u32 + 1"),
    )
    .unwrap();
    let out = ci(&data, &["impact", "--diff", "core", "--json"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["changed"][0]["name"], "parse_config", "{v}");
    assert_eq!(v["changed"][0]["lines"], serde_json::json!([2]));
    // the changed file is excluded from sites; main.rs (depth 2 via load) is reported
    let sites = v["sites"].as_array().unwrap();
    assert!(sites.iter().all(|s| s["path"] != "src/config.rs"), "{v}");
    assert!(sites.iter().any(|s| s["path"] == "src/main.rs"), "{v}");
    // importers: `mod config;` / `use config::load;` in main.rs
    assert!(v["importers"].as_array().unwrap().iter().any(|h| h["path"] == "src/main.rs"), "{v}");

    // human output smoke
    let out = ci(&data, &["impact", "--diff", "core"]);
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("changed definitions (1):"), "{text}");
    assert!(text.contains("src/main.rs"), "{text}");

    // --file: importers + impact of every definition
    let out = ci(&data, &["impact", "--file", "core:src/config.rs", "--json"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["roots"].as_array().unwrap().iter().any(|r| r == "load"));
    assert!(v["sites"].as_array().unwrap().iter().any(|s| s["path"] == "src/main.rs" && s["depth"] == 1), "{v}");

    // --diff-file from stdin-less path + validation errors
    let patch = tmp.path().join("p.diff");
    std::fs::write(&patch, "--- a/src/config.rs\n+++ b/src/config.rs\n@@ -6,1 +6,1 @@\n-    parse_config(p)\n+    parse_config(p) + 0\n").unwrap();
    let out = ci(&data, &["impact", "--diff-file", patch.to_str().unwrap(), "--repo", "core", "--json"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["changed"][0]["name"], "load", "{v}");
    let out = ci(&data, &["impact"]);
    assert!(!out.status.success());
    let out = ci(&data, &["impact", "--diff", "not-registered"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("not registered"));
}
