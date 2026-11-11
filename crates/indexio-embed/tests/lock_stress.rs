//! The embedder's model lock must never be waited on from inside the rayon
//! pool while its holder waits for that same pool.
//!
//! A long-lived MCP server embeds from several threads at once (the
//! auto-refresh embed, the transcript import, the other-repos sync, the
//! compaction's re-embed) and saves the model from another. `embed_works`
//! runs `embedder.embed` on every rayon worker; a concurrent `observe` (or
//! save) takes the model's write lock and then needs the pool for its own
//! parallel step. With every worker parked on the read lock the writer never
//! gets a worker, never releases, and every later tool call blocks on the
//! same lock for good.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use indexio_embed::embed::Embedder;
use indexio_embed::rindex::RandomIndexingEmbedder;
use rayon::prelude::*;

/// `n` distinct code-like texts.
fn corpus(n: usize, seed: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            let k = seed * 1_000_003 + i;
            format!(
                "fn handle_{k}(request_{a}: Payload{b}) -> Result<Response{c}> {{\n    let session_{d} = open_store_{a}(config_{b})?;\n    session_{d}.commit_batch_{c}(request_{a})\n}}",
                a = k % 97,
                b = k % 389,
                c = k % 1021,
                d = k % 53,
            )
        })
        .collect()
}

/// No embed pass and no model write finishing for this long is a deadlock.
const STALL: Duration = Duration::from_secs(8);
const RUN: Duration = Duration::from_secs(20);

fn run_until_stalled(e: Arc<RandomIndexingEmbedder>, writer: impl Fn(&RandomIndexingEmbedder, usize) + Send + 'static) {
    let stop = Arc::new(AtomicBool::new(false));
    let embeds = Arc::new(AtomicU64::new(0));
    let writes = Arc::new(AtomicU64::new(0));

    // what pipeline::embed_works does with the chunks the cache missed
    let reader = {
        let (e, stop, embeds) = (Arc::clone(&e), Arc::clone(&stop), Arc::clone(&embeds));
        std::thread::spawn(move || {
            let texts = corpus(64 * 4 * rayon::current_num_threads().max(2), 1);
            while !stop.load(Ordering::Acquire) {
                let out: Vec<Vec<Vec<f32>>> = texts.par_chunks(64).map(|t| e.embed(t).expect("embed")).collect();
                assert_eq!(out.len(), texts.len().div_ceil(64));
                embeds.fetch_add(1, Ordering::AcqRel);
            }
        })
    };
    // what another pass's observe, or the background model save, does
    let w = {
        let (e, stop, writes) = (Arc::clone(&e), Arc::clone(&stop), Arc::clone(&writes));
        std::thread::spawn(move || {
            let mut i = 0usize;
            while !stop.load(Ordering::Acquire) {
                writer(&e, i);
                writes.fetch_add(1, Ordering::AcqRel);
                i += 1;
                std::thread::sleep(Duration::from_millis(2));
            }
        })
    };

    let t0 = Instant::now();
    let mut last = (0u64, 0u64);
    let mut last_change = Instant::now();
    while t0.elapsed() < RUN {
        std::thread::sleep(Duration::from_millis(100));
        let now = (embeds.load(Ordering::Acquire), writes.load(Ordering::Acquire));
        if now != last {
            last = now;
            last_change = Instant::now();
        } else if last_change.elapsed() > STALL {
            panic!(
                "deadlock after {:.1}s: nothing finished for {}s (embed passes {}, model writes {})",
                t0.elapsed().as_secs_f32(),
                STALL.as_secs(),
                now.0,
                now.1
            );
        }
    }
    stop.store(true, Ordering::Release);
    reader.join().expect("reader thread");
    w.join().expect("writer thread");
    assert!(last.0 > 0 && last.1 > 0, "both sides must have run (embed passes {}, model writes {})", last.0, last.1);
}

fn seeded() -> (tempfile::TempDir, Arc<RandomIndexingEmbedder>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let e = RandomIndexingEmbedder::open(dir.path()).expect("open");
    e.observe(&corpus(2000, 0)).expect("seed observe");
    (dir, Arc::new(e))
}

#[test]
fn parallel_embed_and_observe_do_not_deadlock() {
    let (_dir, e) = seeded();
    run_until_stalled(e, |e, i| e.observe(&corpus(8, 1000 + i)).expect("observe"));
}

#[test]
fn parallel_embed_and_model_save_do_not_deadlock() {
    let (_dir, e) = seeded();
    run_until_stalled(e, |e, i| {
        e.observe(&corpus(1, 5000 + i)).expect("observe");
        e.save_now().expect("save");
    });
}
