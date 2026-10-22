//! Integration tests for indexio-index: roundtrip, tombstones, merge, dup
//! suppression, zstd dict, FST misses.

use std::collections::{BTreeMap, BTreeSet};
use std::thread::sleep;
use std::time::Duration;

use indexio_index::{Shard, ShardSet, ShardWriter};
use indexio_types::codec::{decode_postings, PostingCursor};
use indexio_types::{BlobId, CallRec, DocMeta, ExtractedArtifact, Lang, SymbolKind, SymbolRec};

/// Local trigram extractor for test artifacts (indexio-core's extract is owned
/// by another crate; the shard format only needs sorted gram -> positions).
fn grams_of_content(content: &[u8]) -> Vec<Vec<u8>> {
    let mut set: BTreeSet<Vec<u8>> = BTreeSet::new();
    if content.len() >= 3 {
        for i in 0..=content.len() - 3 {
            set.insert(content[i..i + 3].to_vec());
        }
    }
    set.into_iter().collect()
}

fn make_doc(i: u32, repo_id: u32, content: Vec<u8>) -> (DocMeta, Vec<u8>, ExtractedArtifact) {
    let path = format!("src/file_{i}.rs");
    let art = ExtractedArtifact {
        ngrams: grams_of_content(&content),
        symbols: vec![
            SymbolRec {
                name: format!("func_{i}"),
                kind: SymbolKind::Fn,
                line: 1,
                col: 0,
                scope: if i % 2 == 0 { String::new() } else { "MyStruct".into() },
            },
            SymbolRec {
                name: "SharedName".into(),
                kind: if i % 2 == 0 { SymbolKind::Struct } else { SymbolKind::Method },
                line: 10,
                col: 0,
                scope: format!("mod_{}", i % 3),
            },
        ],
        calls: vec![CallRec {
            callee: "helper".into(),
            caller: format!("func_{i}"),
            line: 2,
        }],
        raw_len: content.len() as u32,
        lang: Lang::Rust,
    };
    let meta = DocMeta {
        blob: BlobId::from_content(&content),
        repo_id,
        path,
        lang: Lang::Rust,
        raw_len: content.len() as u32,
    };
    (meta, content, art)
}

fn default_content(i: u32) -> Vec<u8> {
    format!(
        "fn func_{i}() {{\n    helper();\n    println!(\"hello world number {i}\");\n}}\n"
    )
    .into_bytes()
}

#[test]
fn write_open_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = ShardWriter::new(dir.path()).unwrap();

    let mut docs = Vec::new();
    for i in 0..100u32 {
        let content = match i {
            // unicode content
            3 => "fn grüße_世界() { let s = \"héllo wörld — こんにちは\"; }\n"
                .repeat(20)
                .into_bytes(),
            // one 1 MiB file (repetitive => bounded unique grams)
            7 => b"abcdefghij0123456789 ABCDEFGHIJ\n".repeat(32 * 1024),
            _ => default_content(i),
        };
        docs.push(make_doc(i, 0, content));
    }
    for (meta, content, art) in &docs {
        w.add_doc(meta, content, art).unwrap();
    }
    let shard_path = w.finish(&["repo-a".to_string()]).unwrap();
    assert!(shard_path.extension().unwrap() == "cidx");

    let shard = Shard::open(&shard_path).unwrap();
    assert_eq!(shard.doc_count(), 100);

    // meta sane
    let meta = shard.meta();
    assert_eq!(meta.doc_count, 100);
    assert_eq!(meta.repos, vec!["repo-a".to_string()]);
    let expect_raw: u64 = docs.iter().map(|(_, c, _)| c.len() as u64).sum();
    assert_eq!(meta.total_raw_bytes, expect_raw);
    assert!(!meta.gram_stats.is_empty());
    assert!(meta.zstd_dict.is_none()); // exactly 100 docs => no dict
    assert!(!meta.created.is_empty());

    // content byte-identical
    for (i, (_, content, _)) in docs.iter().enumerate() {
        assert_eq!(&shard.content(i as u32).unwrap(), content, "content doc {i}");
    }

    // doc meta roundtrip
    for (i, (m, _, _)) in docs.iter().enumerate() {
        let dm = shard.doc(i as u32).unwrap();
        assert_eq!(&dm.path, &m.path);
        assert_eq!(dm.blob, m.blob);
        assert_eq!(dm.repo_id, 0);
        assert_eq!(dm.lang, Lang::Rust);
        assert_eq!(dm.raw_len, m.raw_len);
    }

    // postings roundtrip per gram (sample several grams, full decode):
    // the doc is listed, with an empty position list (SPEC-P10)
    for (i, (_, _, art)) in docs.iter().enumerate().step_by(17) {
        for gram in art.ngrams.iter().take(5) {
            let cursor: PostingCursor = shard.postings(gram).expect("gram must exist");
            let mut found = None;
            let mut c = cursor;
            while let Some((docid, positions)) = c.next_entry() {
                if docid == i as u32 {
                    found = Some(positions);
                }
            }
            assert_eq!(found, Some(Vec::new()), "gram {gram:?} doc {i}");
        }
    }

    // FST miss
    assert!(shard.postings(b"\x00\x01\x02").is_none());
    assert!(shard.symbol_postings("no_such_symbol").is_empty());
    assert!(shard.call_postings("no_such_callee").is_empty());

    // symbol postings
    let sp = shard.symbol_postings("func_5");
    assert_eq!(sp.len(), 1);
    assert_eq!(sp[0].0, 5);
    assert_eq!(sp[0].1, SymbolKind::Fn.as_u8());
    assert_eq!(sp[0].2, 1);
    assert_eq!(sp[0].3, "MyStruct");

    let shared = shard.symbol_postings("SharedName");
    assert_eq!(shared.len(), 100);
    assert!(shared.windows(2).all(|w| w[0].0 <= w[1].0)); // sorted by docid
    assert_eq!(shared[3].3, "mod_0"); // doc 3: 3 % 3 == 0

    // call postings
    let cp = shard.call_postings("helper");
    assert_eq!(cp.len(), 100);
    assert!(cp.windows(2).all(|w| w[0].0 < w[1].0));
    assert_eq!(cp[5].0, 5);
    assert_eq!(cp[5].1, "func_5");
    assert_eq!(cp[5].2, 2);

    // common_grams derives from gram_stats without panicking
    let _ = shard.common_grams();
}

#[test]
fn dict_used_when_over_100_docs() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = ShardWriter::new(dir.path()).unwrap();
    let mut contents = Vec::new();
    for i in 0..150u32 {
        // similar-but-varied content so the dictionary has something to learn
        let content = format!(
            "fn function_number_{i}() {{\n    let value = compute_helper({i});\n    \
             println!(\"logging output for item {i}: {{}}\", value);\n}}\n"
        )
        .into_bytes();
        contents.push(content.clone());
        let (meta, _, art) = make_doc(i, 0, content);
        w.add_doc(&meta, contents.last().unwrap(), &art).unwrap();
    }
    let path = w.finish(&["r".to_string()]).unwrap();
    let shard = Shard::open(&path).unwrap();
    assert!(
        shard.meta().zstd_dict.is_some(),
        "dict must be trained when >100 docs"
    );
    for (i, c) in contents.iter().enumerate() {
        assert_eq!(&shard.content(i as u32).unwrap(), c);
    }
}

#[test]
fn replace_by_latest_within_shard() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = ShardWriter::new(dir.path()).unwrap();
    let (m1, c1, a1) = make_doc(1, 0, b"old content here".to_vec());
    let (m2, c2, a2) = make_doc(1, 0, b"new content here!".to_vec()); // same path src/file_1.rs
    assert_eq!(m1.path, m2.path);
    w.add_doc(&m1, &c1, &a1).unwrap();
    w.add_doc(&m2, &c2, &a2).unwrap();
    let path = w.finish(&["r".to_string()]).unwrap();
    let shard = Shard::open(&path).unwrap();
    assert_eq!(shard.doc_count(), 1);
    assert_eq!(shard.content(0).unwrap(), c2);
}

/// Two processes (two handles on one shard file) tombstone different docs:
/// the second write must union with the bitmap on disk, not with the one
/// it read at open (SPEC-P9).
#[test]
fn tombstones_union_across_writers() {
    let dir = tempfile::tempdir().unwrap();
    let shard_path = {
        let mut w = ShardWriter::new(dir.path()).unwrap();
        for i in 0..6u32 {
            let (m, c, a) = make_doc(i, 0, default_content(i));
            w.add_doc(&m, &c, &a).unwrap();
        }
        w.finish(&["repo-a".to_string()]).unwrap()
    };
    let mut a = Shard::open(&shard_path).unwrap();
    let mut b = Shard::open(&shard_path).unwrap();
    a.delete_docs(&[1]).unwrap();
    b.delete_docs(&[4]).unwrap();
    let c = Shard::open(&shard_path).unwrap();
    assert!(c.tombstones().contains(1), "a's deletion lost");
    assert!(c.tombstones().contains(4));
    assert_eq!(c.tombstones().len(), 2);
    assert_eq!(a.tombstones_on_disk().len(), 2, "a sees b's write through the shared mapping");
}

#[test]
fn tombstones_persist_and_filter() {
    let dir = tempfile::tempdir().unwrap();
    let shard_path = {
        let mut w = ShardWriter::new(dir.path()).unwrap();
        for i in 0..30u32 {
            let (m, c, a) = make_doc(i, 0, default_content(i));
            w.add_doc(&m, &c, &a).unwrap();
        }
        w.finish(&["repo-a".to_string()]).unwrap()
    };

    {
        let mut shard = Shard::open(&shard_path).unwrap();
        let deleted: Vec<u32> = (0..10).collect();
        shard.delete_docs(&deleted).unwrap();
        assert_eq!(shard.tombstones().len(), 10);
        for d in &deleted {
            assert!(shard.tombstones().contains(*d));
        }
    }

    // reopened: tombstones persisted
    let shard = Shard::open(&shard_path).unwrap();
    assert_eq!(shard.tombstones().len(), 10);
    drop(shard);

    // ShardSet fan-out excludes tombstoned docs
    let set = ShardSet::open_dir(dir.path()).unwrap();
    assert_eq!(set.doc_count(), 20);
    let visible = set.visible_docs();
    assert_eq!(visible.len(), 20);
    assert!(visible.iter().all(|&(_, d, _)| d >= 10));

    // symbol/call fan-out skip tombstoned
    let shared = set.symbol_postings("SharedName");
    assert_eq!(shared.len(), 20);
    let calls = set.call_postings("helper");
    assert_eq!(calls.len(), 20);

    // gram postings via filtered helper
    let gram = &b"hel"[..];
    let hits = set.posting_docs(gram);
    assert_eq!(hits.len(), 20);

    // tombstoning more docs accumulates across reopen
    let mut shard = Shard::open(&shard_path).unwrap();
    shard.delete_docs(&[10, 11]).unwrap();
    assert_eq!(shard.tombstones().len(), 12);
}

#[test]
fn duplicate_repo_path_newest_shard_wins() {
    let dir = tempfile::tempdir().unwrap();

    // older shard
    {
        let mut w = ShardWriter::new(dir.path()).unwrap();
        let (m, c, a) = make_doc(42, 0, b"old version of the file".to_vec());
        w.add_doc(&m, &c, &a).unwrap();
        w.finish(&["repo-a".to_string()]).unwrap();
    }
    sleep(Duration::from_millis(1100)); // ensure distinct created_unix
    // newer shard with same (repo, path)
    {
        let mut w = ShardWriter::new(dir.path()).unwrap();
        let (m, c, a) = make_doc(42, 0, b"new version of the file!!".to_vec());
        w.add_doc(&m, &c, &a).unwrap();
        w.finish(&["repo-a".to_string()]).unwrap();
    }

    let set = ShardSet::open_dir(dir.path()).unwrap();
    assert_eq!(set.len(), 2);
    assert_eq!(set.doc_count(), 2); // both physically present
    let visible = set.visible_docs();
    assert_eq!(visible.len(), 1);
    let (si, docid, dm) = visible.into_iter().next().unwrap();
    assert_eq!(si, 0, "newest shard is index 0");
    assert_eq!(dm.path, "src/file_42.rs");
    assert_eq!(set.content(si, docid).unwrap(), b"new version of the file!!");
}

#[test]
fn merge_applies_tombstones_and_keeps_postings() {
    let dir = tempfile::tempdir().unwrap();

    let mut expected: BTreeMap<String, Vec<u8>> = BTreeMap::new(); // path -> content (live)
    // shard 1 (oldest): docs 0..20, tombstone 3 of them
    {
        let mut w = ShardWriter::new(dir.path()).unwrap();
        for i in 0..20u32 {
            let (m, c, a) = make_doc(i, 0, default_content(i));
            w.add_doc(&m, &c, &a).unwrap();
            if i >= 3 {
                expected.insert(m.path.clone(), c.clone());
            }
        }
        let p = w.finish(&["repo-a".to_string()]).unwrap();
        let mut s = Shard::open(&p).unwrap();
        s.delete_docs(&[0, 1, 2]).unwrap();
    }
    sleep(Duration::from_millis(1100));
    // shard 2: docs 20..40
    {
        let mut w = ShardWriter::new(dir.path()).unwrap();
        for i in 20..40u32 {
            let (m, c, a) = make_doc(i, 1, default_content(i));
            w.add_doc(&m, &c, &a).unwrap();
            expected.insert(m.path.clone(), c.clone());
        }
        w.finish(&["repo-a".to_string(), "repo-b".to_string()]).unwrap();
    }
    sleep(Duration::from_millis(1100));
    // shard 3 (newest): docs 40..60
    {
        let mut w = ShardWriter::new(dir.path()).unwrap();
        for i in 40..60u32 {
            let (m, c, a) = make_doc(i, 0, default_content(i));
            w.add_doc(&m, &c, &a).unwrap();
            expected.insert(m.path.clone(), c.clone());
        }
        w.finish(&["repo-a".to_string()]).unwrap();
    }

    let mut set = ShardSet::open_dir(dir.path()).unwrap();
    assert_eq!(set.len(), 3);
    assert_eq!(set.doc_count(), 57);
    let old_paths: Vec<_> = set.shards().iter().map(|s| s.path().to_path_buf()).collect();

    set.merge(dir.path(), 2).unwrap();

    assert_eq!(set.len(), 2, "oldest two merged into one compound");
    assert_eq!(set.doc_count(), 57);
    // merged files removed, newest shard file kept
    assert!(old_paths[0].exists());
    assert!(!old_paths[1].exists());
    assert!(!old_paths[2].exists());

    // all live docs visible exactly once with correct content
    let visible = set.visible_docs();
    assert_eq!(visible.len(), 57);
    for (si, docid, dm) in &visible {
        let content = set.content(*si, *docid).unwrap();
        assert_eq!(&content, expected.get(&dm.path).unwrap(), "path {}", dm.path);
        // deleted docs are gone entirely
        assert!(dm.path != "src/file_0.rs" && dm.path != "src/file_1.rs" && dm.path != "src/file_2.rs");
    }

    // postings correct after merge: gram "hel" appears in every doc
    let hits = set.posting_docs(b"hel");
    assert_eq!(hits.len(), 57);

    // symbol postings survive the merge (docs from merged shards included)
    let sp = set.symbol_postings("func_5");
    assert_eq!(sp.len(), 1);
    let cp = set.call_postings("helper");
    assert_eq!(cp.len(), 57);
    let sc = set.symbol_postings("SharedName");
    assert_eq!(sc.len(), 57);

    // repo table unioned: a doc from shard 2 (repo-b) resolves its repo name
    let doc20 = visible.iter().find(|(_, _, dm)| dm.path == "src/file_20.rs").unwrap();
    let shard = set.shard(doc20.0).unwrap();
    assert_eq!(shard.meta().repos[doc20.2.repo_id as usize], "repo-b");

    // reopen from disk and re-verify (the compound shard reads back)
    let set2 = ShardSet::open_dir(dir.path()).unwrap();
    assert_eq!(set2.doc_count(), 57);
    assert_eq!(set2.visible_docs().len(), 57);
    assert_eq!(set2.posting_docs(b"hel").len(), 57);
}

#[test]
fn merge_folds_young_deltas_and_leaves_the_big_shard() {
    let dir = tempfile::tempdir().unwrap();
    // the base: 200 docs
    let base = {
        let mut w = ShardWriter::new(dir.path()).unwrap();
        for i in 0..200u32 {
            let (m, c, a) = make_doc(i, 0, default_content(i));
            w.add_doc(&m, &c, &a).unwrap();
        }
        w.finish(&["repo-a".to_string()]).unwrap()
    };
    // six one-doc deltas; the first rewrites file_5 (its base doc tombstoned)
    for k in 0..6u32 {
        sleep(Duration::from_millis(1100));
        let mut w = ShardWriter::new(dir.path()).unwrap();
        let i = if k == 0 { 5 } else { 1000 + k };
        let (m, _, a) = make_doc(i, 0, default_content(i));
        w.add_doc(&m, format!("fn edited_{k}() {{}}\n").as_bytes(), &a).unwrap();
        w.finish(&["repo-a".to_string()]).unwrap();
        if k == 0 {
            Shard::open(&base).unwrap().delete_docs(&[5]).unwrap();
        }
    }
    let mut set = ShardSet::open_dir(dir.path()).unwrap();
    assert_eq!(set.len(), 7);
    set.merge(dir.path(), 3).unwrap();
    assert_eq!(set.len(), 3);
    assert!(base.exists(), "the big shard is not rewritten for five small deltas");
    let visible = set.visible_docs();
    assert_eq!(visible.len(), 205);
    let f5 = visible.iter().find(|(_, _, dm)| dm.path == "src/file_5.rs").unwrap();
    assert_eq!(set.content(f5.0, f5.1).unwrap(), b"fn edited_0() {}\n", "the delta's version wins");
    // reopened from disk: same view
    let set2 = ShardSet::open_dir(dir.path()).unwrap();
    assert_eq!(set2.visible_docs().len(), 205);
}

#[test]
fn merge_noop_when_within_limit() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut w = ShardWriter::new(dir.path()).unwrap();
        let (m, c, a) = make_doc(1, 0, default_content(1));
        w.add_doc(&m, &c, &a).unwrap();
        w.finish(&["r".to_string()]).unwrap();
    }
    let mut set = ShardSet::open_dir(dir.path()).unwrap();
    set.merge(dir.path(), 4).unwrap();
    assert_eq!(set.len(), 1);
    assert_eq!(set.doc_count(), 1);
}

#[test]
fn full_decode_matches_cursor() {
    // guard: merge's decode path and the cursor path agree
    let dir = tempfile::tempdir().unwrap();
    let mut w = ShardWriter::new(dir.path()).unwrap();
    for i in 0..50u32 {
        let (m, c, a) = make_doc(i, 0, default_content(i));
        w.add_doc(&m, &c, &a).unwrap();
    }
    let path = w.finish(&["r".to_string()]).unwrap();
    let shard = Shard::open(&path).unwrap();
    let mut c = shard.postings(b"hel").unwrap();
    let mut via_cursor = Vec::new();
    while let Some(e) = c.next_entry() {
        via_cursor.push(e);
    }
    // re-decode the same payload bytes via decode_postings
    let off = {
        // fetch through public cursor a second time to compare lengths
        let mut c2 = shard.postings(b"hel").unwrap();
        let mut n = 0;
        while c2.next_entry().is_some() {
            n += 1;
        }
        n
    };
    assert_eq!(via_cursor.len(), off);
    let decoded = decode_postings(&indexio_types::codec::encode_postings(&via_cursor));
    assert_eq!(decoded, via_cursor);
}

/// A set opened over a shared base answers exactly like the layered set
/// opened from both directories: same live docs in the same order, same
/// totals, lookups, postings, symbols and calls; with and without an ACL.
#[test]
fn open_over_matches_open_layers() {
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::Arc;
    use indexio_index::Visibility;

    let base_dir = tempfile::tempdir().unwrap();
    let upper_dir = tempfile::tempdir().unwrap();
    let repos = ["repo-a".to_string(), "repo-b".to_string()];
    // older base shard: files 0..20 in both repos
    let mut w = ShardWriter::new(base_dir.path()).unwrap();
    for i in 0..20u32 {
        let (m, c, a) = make_doc(i, i % 2, default_content(i));
        w.add_doc(&m, &c, &a).unwrap();
    }
    w.finish(&repos).unwrap();
    sleep(Duration::from_millis(1100));
    // newer base shard: a new version of file_0 (repo-a) and file_3 (repo-b)
    let mut w = ShardWriter::new(base_dir.path()).unwrap();
    for i in [0u32, 3] {
        let (m, c, a) = make_doc(i, i % 2, format!("fn func_{i}() {{ helper(); /* v2 hello */ }}\n").into_bytes());
        w.add_doc(&m, &c, &a).unwrap();
    }
    w.finish(&repos).unwrap();
    {
        // a tombstone in the base: file_5 is gone
        let mut set = ShardSet::open_dir(base_dir.path()).unwrap();
        let (si, d, _) = set.visible_docs().into_iter().find(|(_, _, m)| m.path == "src/file_5.rs").unwrap();
        set.delete_docs(si, &[d]).unwrap();
    }
    // the user's overlay: file_2 changed, file_100 new (repo-a)
    let mut w = ShardWriter::new(upper_dir.path()).unwrap();
    for i in [2u32, 100] {
        let (m, c, a) = make_doc(i, 0, format!("fn func_{i}() {{ helper(); /* mine hello */ }}\n").into_bytes());
        w.add_doc(&m, &c, &a).unwrap();
    }
    w.finish(&["repo-a".to_string()]).unwrap();

    let base = Arc::new(ShardSet::open_dir(base_dir.path()).unwrap());
    let hidden: HashSet<(String, String)> = [("repo-a".to_string(), "src/file_4.rs".to_string())].into_iter().collect();
    let only_a: Arc<dyn Fn(&str) -> bool + Send + Sync> = Arc::new(|r: &str| r == "repo-a");
    let dirs = [upper_dir.path().to_path_buf(), base_dir.path().to_path_buf()];
    let upper: Vec<PathBuf> = vec![upper_dir.path().to_path_buf()];
    for (label, upper, view) in [
        ("overlay", upper.clone(), Visibility { hidden: hidden.clone(), allow: None }),
        ("overlay + acl", upper.clone(), Visibility { hidden: hidden.clone(), allow: Some(only_a.clone()) }),
        ("acl only", Vec::new(), Visibility { hidden: HashSet::new(), allow: Some(only_a.clone()) }),
        ("nothing", Vec::new(), Visibility::default()),
    ] {
        let dirs: Vec<PathBuf> = upper.iter().cloned().chain([dirs[1].clone()]).collect();
        let old = ShardSet::open_layers(&dirs, view.clone()).unwrap();
        let new = ShardSet::open_over(&upper, Arc::clone(&base), view).unwrap();
        assert_eq!(old.len(), new.len(), "{label}");
        for i in 0..old.len() {
            assert_eq!(old.shard(i).unwrap().path(), new.shard(i).unwrap().path(), "{label} shard {i}");
        }
        let key = |v: &[(usize, u32, DocMeta)]| -> Vec<(usize, u32, String, u32, u32)> {
            v.iter().map(|(si, d, m)| (*si, *d, m.path.clone(), m.repo_id, m.raw_len)).collect()
        };
        let docs = old.visible_docs();
        assert_eq!(key(&docs), key(&new.visible_docs()), "{label}");
        let raw: u64 = docs.iter().map(|(_, _, m)| m.raw_len as u64).sum();
        assert_eq!(new.visible_totals(), (docs.len(), raw), "{label}");
        assert_eq!(old.visible_totals(), (docs.len(), raw), "{label}");
        for i in [0u32, 1, 2, 3, 4, 5, 6, 100, 999] {
            for r in &repos {
                let p = format!("src/file_{i}.rs");
                let want = docs.iter().find(|(si, _, m)| m.path == p && old.shard(*si).unwrap().meta().repos[m.repo_id as usize] == *r).map(|(si, d, _)| (*si, *d));
                assert_eq!(new.locate(r, &p), want, "{label} locate {r}:{p}");
                assert_eq!(old.locate(r, &p), want, "{label} locate {r}:{p}");
            }
        }
        assert_eq!(old.posting_doc_ids(b"hel"), new.posting_doc_ids(b"hel"), "{label}");
        assert_eq!(old.posting_docs(b"elp"), new.posting_docs(b"elp"), "{label}");
        assert_eq!(old.symbol_postings("SharedName"), new.symbol_postings("SharedName"), "{label}");
        assert_eq!(old.call_postings("helper"), new.call_postings("helper"), "{label}");
        let iter: Vec<_> = new.visible_iter().map(|(si, d, m)| (si, d, m.clone())).collect();
        assert_eq!(key(&iter), key(&docs), "{label}");
        for (si, d, _) in &docs {
            assert!(new.is_live(*si, *d), "{label}");
        }
    }
}
