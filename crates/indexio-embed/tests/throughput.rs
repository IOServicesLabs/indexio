//! Embedding throughput of the rindex model on real source text, for
//! before/after comparisons of changes to the embed path. Not a regression
//! test (timings depend on the machine), so it is ignored by default:
//!
//! ```text
//! cargo test --release -p indexio-embed --test throughput -- --ignored --nocapture
//! ```
//!
//! `INDEXIO_BENCH_DIR` names the tree to read (default: this workspace's
//! `crates/`); `RAYON_NUM_THREADS=1` gives the single-threaded numbers;
//! `INDEXIO_BENCH_OBSERVE_RUNS=1` shortens a run that is about `embed`.

use std::time::Instant;

use indexio_embed::embed::Embedder;
use indexio_embed::rindex::{tokenize, RandomIndexingEmbedder};

mod common;
use common::{bench_root, chunks};

const RUNS: usize = 5;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn report(what: &str, runs: &[f64], texts: usize, tokens: usize) {
    let best = runs.iter().copied().fold(f64::INFINITY, f64::min);
    let all: Vec<String> = runs.iter().map(|r| format!("{r:.1}")).collect();
    println!(
        "{what:<28} best {best:8.1} ms  first {:8.1} ms  {:9.0} texts/s  {:6.0} ns/token   runs [{}]",
        runs[0],
        texts as f64 / (best / 1000.0),
        best * 1e6 / tokens as f64,
        all.join(", ")
    );
}

#[test]
#[ignore = "benchmark: run with --release --ignored --nocapture"]
fn embed_and_observe_throughput() {
    let texts = chunks(&bench_root());
    assert!(texts.len() > 100, "no source text under {}", bench_root().display());
    let bytes: usize = texts.iter().map(|t| t.len()).sum();
    let t = Instant::now();
    let tokens: usize = texts.iter().map(|t| tokenize(t).len()).sum();
    let tokenize_ms = ms(t);
    println!(
        "corpus: {} chunks, {:.1} MB, {} tokens (tokenize alone, serial: {:.1} ms, {:.0} ns/token), rayon threads {}",
        texts.len(),
        bytes as f64 / 1e6,
        tokens,
        tokenize_ms,
        tokenize_ms * 1e6 / tokens as f64,
        rayon::current_num_threads()
    );

    // observe: a fresh model per run (the first run also pays whatever is
    // cached per process)
    let observe_runs = std::env::var("INDEXIO_BENCH_OBSERVE_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(RUNS).max(1);
    let mut observe = Vec::new();
    let mut keep = None;
    for _ in 0..observe_runs {
        let dir = tempfile::tempdir().unwrap();
        let e = RandomIndexingEmbedder::open(dir.path()).unwrap();
        let t = Instant::now();
        e.observe(&texts).unwrap();
        observe.push(ms(t));
        keep = Some((dir, e));
    }
    report("observe (fresh model)", &observe, texts.len(), tokens);
    let (_dir, e) = keep.unwrap();
    println!("vocabulary after observe: {} terms", e.vocab_len());

    // observe in the small calls of a server refresh (8 texts per call)
    let mut small = Vec::new();
    for _ in 0..observe_runs {
        let dir = tempfile::tempdir().unwrap();
        let e = RandomIndexingEmbedder::open(dir.path()).unwrap();
        let t = Instant::now();
        for part in texts.chunks(8) {
            e.observe(part).unwrap();
        }
        small.push(ms(t));
    }
    report("observe (8 texts per call)", &small, texts.len(), tokens);

    // embed, first with the vocabulary in memory (as observed just now),
    // then mapped from the saved file (a server's steady state)
    for state in ["in memory", "mapped"] {
        if state == "mapped" {
            e.save_now().unwrap();
        }
        // the whole batch (parallel inside the embedder)
        let mut embed = Vec::new();
        for _ in 0..RUNS {
            let t = Instant::now();
            let v = e.embed(&texts).unwrap();
            embed.push(ms(t));
            assert_eq!(v.len(), texts.len());
        }
        report(&format!("embed batch, {state}"), &embed, texts.len(), tokens);

        // below the embedder's parallel threshold, on this thread
        let mut serial = Vec::new();
        for _ in 0..RUNS {
            let t = Instant::now();
            let mut n = 0;
            for part in texts.chunks(4) {
                n += e.embed(part).unwrap().len();
            }
            serial.push(ms(t));
            assert_eq!(n, texts.len());
        }
        report(&format!("embed 4 per call, {state}"), &serial, texts.len(), tokens);
    }

    // a checksum of the vectors, to compare two builds on the same corpus
    let v = e.embed(&texts).unwrap();
    let mut h = blake3::Hasher::new();
    for row in &v {
        for x in row {
            h.update(&x.to_le_bytes());
        }
    }
    println!("vectors blake3 {}", h.finalize().to_hex());
}
