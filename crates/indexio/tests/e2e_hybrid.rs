//! End-to-end CLI test for the semantic plane (SPEC-P2 §5, SPEC-P4 §2):
//! build a 2-repo shard fixture (ShardWriter, no git needed), run
//! `indexio embed --all`, then `indexio search --mode hybrid --json` and assert
//! exit 0 + non-empty hits. Offline: the default self-contained
//! RandomIndexingEmbedder (INDEXIO_EMBED_BASE is scrubbed from the child env).

use std::path::Path;
use std::process::Command;

use indexio_core::grams::{self, CommonGrams};
use indexio_index::ShardWriter;
use indexio_types::{BlobId, DocMeta, ExtractedArtifact, Lang};

/// Write one shard containing `docs` ((path, content) pairs) for `repo`.
fn write_shard(shards_dir: &Path, repo: &str, docs: &[(&str, &str)]) {
    let mut w = ShardWriter::new(shards_dir).unwrap();
    for (path, content) in docs {
        let content = content.as_bytes();
        let art = ExtractedArtifact {
            ngrams: grams::extract(content, &CommonGrams::empty()),
            raw_len: content.len() as u32,
            lang: Lang::Rust,
            ..Default::default()
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
    w.finish(&[repo.to_string()]).unwrap();
}

fn ci(data_dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_indexio"))
        .arg("--data-dir")
        .arg(data_dir)
        .args(args)
        .env_remove("INDEXIO_EMBED_BASE") // force the offline default (rindex)
        .output()
        .unwrap()
}

fn build_fixture(tmp: &Path) {
    let shards = tmp.join("shards");
    std::fs::create_dir_all(&shards).unwrap();
    write_shard(
        &shards,
        "alpha",
        &[(
            "src/vector.rs",
            "fn foo_bar_123() {\n    // cosine similarity over embedding vectors\n    rank_by_cosine();\n}\n",
        )],
    );
    write_shard(
        &shards,
        "beta",
        &[(
            "src/pool.rs",
            "fn database_pool() {\n    // postgres connection pooling with tls timeouts\n}\n",
        )],
    );
}

#[test]
fn embed_then_hybrid_search_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path());

    // 1. embed --all (default target), prints the report table.
    let out = ci(tmp.path(), &["embed", "--all"]);
    assert!(
        out.status.success(),
        "indexio embed failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("cas_hits"), "table header: {stdout}");
    assert!(stdout.contains("alpha"), "{stdout}");
    assert!(stdout.contains("beta"), "{stdout}");

    // 2. embcas-stats reflects the embed run.
    let out = ci(tmp.path(), &["embcas-stats", "--json"]);
    assert!(out.status.success());
    let v: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&out.stdout)).unwrap();
    assert_eq!(v["model_id"], "rindex-v2");
    assert!(v["entries"].as_u64().unwrap() > 0, "{v}");

    // 3. hybrid search returns JSON hits (exit 0, non-empty).
    let out = ci(
        tmp.path(),
        &["search", "foo_bar_123", "--mode", "hybrid", "--json"],
    );
    assert!(
        out.status.success(),
        "indexio search failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&out.stdout)).unwrap();
    let hits = v["hits"].as_array().unwrap();
    assert!(!hits.is_empty(), "{v}");
    assert_eq!(hits[0]["hit"]["path"], "src/vector.rs", "{hits:?}");
    assert_eq!(hits[0]["lex_rank"], 1);
    assert!(hits[0]["sem_rank"].is_u64(), "{hits:?}");

    // 4. semantic mode works too.
    let out = ci(
        tmp.path(),
        &["search", "cosine embedding vector", "--mode", "semantic", "--json"],
    );
    assert!(out.status.success());
    let v: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&out.stdout)).unwrap();
    assert_eq!(
        v["hits"].as_array().unwrap()[0]["path"],
        "src/vector.rs",
        "{v}"
    );

    // 5. lexical mode unchanged; invalid mode errors (non-zero exit).
    let out = ci(tmp.path(), &["search", "foo_bar_123", "--json"]);
    assert!(out.status.success());
    let out = ci(tmp.path(), &["search", "foo", "--mode", "bogus"]);
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("invalid mode"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // 6. embed --repo scopes to one repo; --repo/--all conflict is a CLI error.
    let out = ci(tmp.path(), &["embed", "--repo", "alpha", "--all"]);
    assert!(!out.status.success());
}

#[test]
fn org_sync_help_and_flag_smoke() {
    let tmp = tempfile::tempdir().unwrap();
    // --help works and documents the flags (no network touched).
    let out = ci(tmp.path(), &["org-sync", "--help"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    for flag in [
        "--org",
        "--token-env",
        "--dest",
        "--limit",
        "--include-forks",
        "--include-archived",
        "--full-clone",
    ] {
        assert!(stdout.contains(flag), "missing {flag} in help:\n{stdout}");
    }
    // --org is required.
    let out = ci(tmp.path(), &["org-sync"]);
    assert!(!out.status.success());
}
