//! Heap accounting for memory work (cargo feature `memstat`, off by default).
//!
//! A counting wrapper around the system allocator: live bytes, the peak of
//! live bytes, and the number of allocations. `INDEXIO_MEMSTAT=1` makes the
//! MCP stdio loop print one `memstat` line to stderr after every answered
//! request, and `INDEXIO_MEMSTAT_EVERY=<secs>` prints one on a timer, so the
//! growth caused by background jobs shows up between calls.
//!
//! The point of the numbers: private bytes of the process minus `live` is
//! memory the allocator holds but the program does not use (fragmentation and
//! retention after a spike), which no data-structure change can give back.

use std::alloc::{GlobalAlloc, Layout};

/// The allocator that is counted: the system's, or mimalloc with that
/// feature.
#[cfg(not(feature = "mimalloc"))]
use std::alloc::System as Inner;
#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc as Inner;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

pub struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCS: AtomicU64 = AtomicU64::new(0);

#[inline]
fn add(n: usize) {
    let now = LIVE.fetch_add(n, Relaxed) + n;
    if now > PEAK.load(Relaxed) {
        PEAK.fetch_max(now, Relaxed);
    }
    ALLOCS.fetch_add(1, Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = Inner.alloc(l);
        if !p.is_null() {
            add(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = Inner.alloc_zeroed(l);
        if !p.is_null() {
            add(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        Inner.dealloc(p, l);
        LIVE.fetch_sub(l.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = Inner.realloc(p, l, new);
        if !q.is_null() {
            LIVE.fetch_sub(l.size(), Relaxed);
            add(new);
        }
        q
    }
}

/// (live bytes, peak live bytes since the last `reset_peak`, allocations).
pub fn snapshot() -> (usize, usize, u64) {
    (LIVE.load(Relaxed), PEAK.load(Relaxed), ALLOCS.load(Relaxed))
}

/// Start a new peak window at the current live size.
pub fn reset_peak() {
    PEAK.store(LIVE.load(Relaxed), Relaxed);
}

pub fn enabled() -> bool {
    std::env::var_os("INDEXIO_MEMSTAT").is_some()
}

/// One line on stderr: `memstat <tag> live_mb=.. peak_mb=.. allocs=..`; the
/// peak is the highest live size since the previous line.
pub fn report(tag: &str) {
    let (live, peak, allocs) = snapshot();
    reset_peak();
    eprintln!(
        "memstat {tag} live_mb={:.1} peak_mb={:.1} allocs={allocs}",
        live as f64 / 1048576.0,
        peak as f64 / 1048576.0
    );
}

/// The timer thread behind `INDEXIO_MEMSTAT_EVERY`.
pub fn start_timer() {
    let Some(secs) = std::env::var("INDEXIO_MEMSTAT_EVERY").ok().and_then(|s| s.parse::<u64>().ok()) else {
        return;
    };
    let _ = std::thread::Builder::new().name("memstat".into()).spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_secs(secs.max(1)));
        report("tick");
    });
}
