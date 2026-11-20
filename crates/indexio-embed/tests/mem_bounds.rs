//! Heap bounds of the rindex model, measured with a counting allocator.
//!
//! A long-lived server keeps one embedder per session, so three costs that
//! grew with the vocabulary instead of with the work at hand are pinned
//! here:
//! - labels: embedding text with many distinct tokens must not leave a
//!   per-token cache behind (it used to keep up to 2,000,000 entries,
//!   ~450 MB, for the life of the process);
//! - save: writing the model must not copy the vocabulary or build the
//!   output file in memory (it used to take about twice the file size,
//!   ~0.53 GB for the 253 MB model);
//! - cap: an observe on a vocabulary at its cap must not allocate per
//!   vocabulary term.
//!
//! `real_model_measurements` (ignored) prints the same numbers for a real
//! model file: `INDEXIO_REAL_MODEL=<path to rindex-v2.rimodel>`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::Instant;

use indexio_embed::embed::Embedder;
use indexio_embed::rindex::RandomIndexingEmbedder;

mod common;
use common::{bench_root, chunks, synthetic_texts, word};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn add(n: usize) {
    let now = LIVE.fetch_add(n, Relaxed) + n;
    if now > PEAK.load(Relaxed) {
        PEAK.fetch_max(now, Relaxed);
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            add(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(l);
        if !p.is_null() {
            add(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        LIVE.fetch_sub(l.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = System.realloc(p, l, new);
        if !q.is_null() {
            LIVE.fetch_sub(l.size(), Relaxed);
            add(new);
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// The counters are process-wide: one measured test at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

const MIB: usize = 1 << 20;

fn mib(n: usize) -> f64 {
    n as f64 / MIB as f64
}

/// Heap use of `f`: its result, the highest live size above the level at
/// entry, and what it left allocated (negative when it freed more).
fn measure<R>(f: impl FnOnce() -> R) -> (R, usize, isize) {
    let before = LIVE.load(Relaxed);
    PEAK.store(before, Relaxed);
    let r = f();
    let peak = PEAK.load(Relaxed).saturating_sub(before);
    let kept = LIVE.load(Relaxed) as isize - before as isize;
    (r, peak, kept)
}

fn file_hash(path: &std::path::Path) -> String {
    let mut h = blake3::Hasher::new();
    let mut f = std::fs::File::open(path).unwrap();
    let mut buf = vec![0u8; MIB];
    loop {
        let n = std::io::Read::read(&mut f, &mut buf).unwrap();
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    h.finalize().to_hex().to_string()
}

fn vectors_hash(v: &[Vec<f32>]) -> String {
    let mut h = blake3::Hasher::new();
    for row in v {
        for x in row {
            h.update(&x.to_le_bytes());
        }
    }
    h.finalize().to_hex().to_string()
}

/// Embedding 300,000 distinct tokens (and as many distinct bigrams) leaves
/// nothing per token behind.
#[test]
fn labels_of_many_distinct_tokens_leave_no_heap_behind() {
    const BOUND: usize = 16 * MIB;
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let e = RandomIndexingEmbedder::open(dir.path()).unwrap();
    // the embedder's threads and their first allocations are not the cache
    e.embed(&synthetic_texts(64, 40, 1000, 7)).unwrap();
    let texts: Vec<String> = (0..3000)
        .map(|i| (0..100).map(|j| word(1_000_000 + i * 100 + j)).collect::<Vec<_>>().join(" "))
        .collect();
    let (_, peak, kept) = measure(|| {
        for part in texts.chunks(256) {
            assert_eq!(e.embed(part).unwrap().len(), part.len());
        }
    });
    println!("labels: 300000 distinct tokens embedded, kept {:.2} MiB, peak {:.2} MiB", mib(kept.max(0) as usize), mib(peak));
    assert!(
        kept <= BOUND as isize,
        "embedding 300,000 distinct tokens left {:.1} MiB allocated (bound {} MiB): a per-token cache is back",
        mib(kept.max(0) as usize),
        BOUND / MIB
    );
}

/// A save costs a bucket table and the sorted keys of what changed, not a
/// copy of the vocabulary and of the file.
#[test]
fn save_peak_heap_does_not_grow_with_the_model_file() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let e = RandomIndexingEmbedder::open(dir.path()).unwrap();
    e.observe(&synthetic_texts(1500, 40, 20_000, 11)).unwrap();
    let terms = e.vocab_len();
    assert!(terms >= 30_000, "the generated vocabulary is too small to scale from: {terms}");
    // 48 B per term (16 B of bucket table, 24 B per sorted key, slack) and
    // 1 MiB for buffers: 12.4 MiB at the 250,000-term cap
    let bound = terms * 48 + MIB;

    // first save: the whole vocabulary is in memory
    let (r, peak_first, _) = measure(|| e.save_now());
    r.unwrap();
    let path = RandomIndexingEmbedder::model_path(dir.path());
    let file = std::fs::metadata(&path).unwrap().len() as usize;

    // the server's case: a few changed terms on top of the mapped file
    e.observe(&synthetic_texts(32, 40, 20_000, 13)).unwrap();
    let (r, peak_second, _) = measure(|| e.save_now());
    r.unwrap();

    println!(
        "save: {terms} terms, file {:.1} MiB, peak heap first save {:.2} MiB, later save {:.2} MiB (bound {:.2} MiB)",
        mib(file),
        mib(peak_first),
        mib(peak_second),
        mib(bound)
    );
    assert!(file > 8 * bound, "the model file ({file} B) must dwarf the bound ({bound} B) for this test to mean anything");
    for (which, peak) in [("first", peak_first), ("later", peak_second)] {
        assert!(
            peak <= bound,
            "{which} save of a {terms}-term model ({:.1} MiB file) peaked at {:.2} MiB of heap (bound {:.2} MiB): the save is materializing the model again",
            mib(file),
            mib(peak),
            mib(bound)
        );
    }
    // and what it wrote reopens as the same model
    let again = RandomIndexingEmbedder::open(dir.path()).unwrap();
    assert_eq!(again.vocab_len(), e.vocab_len());
    let probe = synthetic_texts(8, 40, 20_000, 17);
    assert_eq!(again.embed(&probe).unwrap(), e.embed(&probe).unwrap());
}

/// An observe on a vocabulary at its cap evicts as many terms as it added
/// without a copy of every term (which was 32 B and a string each, on every
/// observe: ~20 MB a time at the 250,000-term cap).
#[test]
fn observe_at_the_cap_allocates_nothing_per_vocabulary_term() {
    const CAP: usize = 60_000;
    const BOUND: usize = MIB;
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let e = RandomIndexingEmbedder::open_with_cap(dir.path(), CAP).unwrap();
    e.observe(&synthetic_texts(1500, 40, 20_000, 21)).unwrap();
    assert_eq!(e.vocab_len(), CAP, "the generated vocabulary must reach the cap");
    // a server's state: the vocabulary mapped from the file, at its cap
    e.save_now().unwrap();
    e.observe(&synthetic_texts(1, 40, 20_000, 22)).unwrap();
    let text = synthetic_texts(1, 40, 20_000, 23);
    let (r, peak, _) = measure(|| e.observe(&text));
    r.unwrap();
    assert_eq!(e.vocab_len(), CAP, "new terms came in and as many were evicted");
    println!("cap: observe of one text on a {CAP}-term vocabulary at its cap peaked at {:.2} MiB (bound {:.2} MiB)", mib(peak), mib(BOUND));
    assert!(
        peak <= BOUND,
        "observing one text on a {CAP}-term vocabulary at its cap peaked at {:.2} MiB of heap (bound {:.2} MiB; a copy of the terms is at least {:.2} MiB): the eviction is copying the vocabulary again",
        mib(peak),
        mib(BOUND),
        mib(CAP * 32)
    );
}

/// The numbers above for a real model (a copy of it: the original is only
/// read): label heap after embedding real source text, peak heap of an
/// observe at the vocabulary cap and of a save, and hashes of the saved
/// files and of the vectors to compare two builds.
#[test]
#[ignore = "measurement: INDEXIO_REAL_MODEL=<rindex-v2.rimodel> ... --ignored --nocapture"]
fn real_model_measurements() {
    let Some(model) = std::env::var_os("INDEXIO_REAL_MODEL") else {
        println!("INDEXIO_REAL_MODEL is not set: nothing measured");
        return;
    };
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let dir = match std::env::var_os("INDEXIO_BENCH_TMP") {
        Some(d) => tempfile::tempdir_in(d).unwrap(),
        None => tempfile::tempdir().unwrap(),
    };
    let path = RandomIndexingEmbedder::model_path(dir.path());
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::copy(&model, &path).unwrap();
    let texts = chunks(&bench_root());
    assert!(texts.len() > 1000, "too little source text under {}", bench_root().display());

    let (e, _, kept) = measure(|| RandomIndexingEmbedder::open(dir.path()).unwrap());
    println!("open: {} terms, file {:.1} MiB, heap kept {:.2} MiB", e.vocab_len(), mib(std::fs::metadata(&path).unwrap().len() as usize), mib(kept.max(0) as usize));
    e.embed(&texts[..64]).unwrap();

    let t = Instant::now();
    let (_, peak, kept) = measure(|| {
        for part in texts.chunks(256) {
            e.embed(part).unwrap();
        }
    });
    println!("embed {} chunks: {:.0} ms, heap kept {:.2} MiB, peak {:.2} MiB", texts.len(), t.elapsed().as_secs_f64() * 1e3, mib(kept.max(0) as usize), mib(peak));

    // observes of a server refresh at the vocabulary cap: 8 texts per call
    let (mut worst, mut total_ms) = (0usize, 0.0f64);
    for part in texts[..160].chunks(8) {
        let t = Instant::now();
        let (r, peak, _) = measure(|| e.observe(part));
        r.unwrap();
        total_ms += t.elapsed().as_secs_f64() * 1e3;
        worst = worst.max(peak);
    }
    println!("observe 20 x 8 texts: {:.1} ms per call, worst peak heap {:.2} MiB, vocabulary {}", total_ms / 20.0, mib(worst), e.vocab_len());

    let save = |what: &str| {
        let t = Instant::now();
        let (r, peak, kept) = measure(|| e.save_now());
        r.unwrap();
        println!(
            "save ({what}): {:.0} ms, peak heap {:.2} MiB, heap kept {:.2} MiB, file {:.1} MiB blake3 {}",
            t.elapsed().as_secs_f64() * 1e3,
            mib(peak),
            kept as f64 / MIB as f64,
            mib(std::fs::metadata(&path).unwrap().len() as usize),
            file_hash(&path)
        );
    };
    save("after 160 observed texts");

    let t = Instant::now();
    let (r, peak, _) = measure(|| e.observe(&texts[160..1184]));
    r.unwrap();
    println!("observe 1024 texts in one call: {:.0} ms, peak heap {:.2} MiB, vocabulary {}", t.elapsed().as_secs_f64() * 1e3, mib(peak), e.vocab_len());
    save("after 1024 more");

    e.observe(&texts[1184..1187]).unwrap();
    save("after 3 more");

    println!("vectors of the first 512 chunks: blake3 {}", vectors_hash(&e.embed(&texts[..512]).unwrap()));
    let leftovers: Vec<String> = std::fs::read_dir(path.parent().unwrap())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".tmp") || n.contains(".stale"))
        .collect();
    println!("temp or parked files left behind: {leftovers:?}");
}
