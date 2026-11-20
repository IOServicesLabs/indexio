//! Process-wide worker-thread budget.
//!
//! The binary decides once, before any thread pool exists, how many worker
//! threads each pool of the process gets, and every pool reads the number
//! when it is built: the global rayon pool (queries, pipelines) and the
//! private model pool of the random-indexing embedder.
//!
//! A long-lived `indexio mcp` server caps both: one server per agent session
//! is running, most of them idle, and a pool per logical CPU for each of them
//! is a hundred parked threads apiece. Short-lived CLI runs (`sync`, `embed`,
//! `compact`, the maintenance child the server spawns) leave it unset and
//! keep one worker per logical CPU.

use std::sync::atomic::{AtomicUsize, Ordering};

/// 0 = unset: each pool takes its own default (one worker per logical CPU,
/// or `RAYON_NUM_THREADS`).
static WORKERS: AtomicUsize = AtomicUsize::new(0);

/// Set the size every worker pool of this process is built with. Call it
/// before the first parallel section; a pool that already exists keeps the
/// size it was built with.
pub fn set_worker_threads(n: usize) {
    WORKERS.store(n, Ordering::Release);
}

/// The configured pool size, `0` when none was set (the value
/// `rayon::ThreadPoolBuilder::num_threads` takes for "the default").
pub fn worker_threads() -> usize {
    WORKERS.load(Ordering::Acquire)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_is_zero_and_set_is_read_back() {
        // (the only test of this process that touches the setting)
        assert_eq!(worker_threads(), 0);
        set_worker_threads(6);
        assert_eq!(worker_threads(), 6);
        set_worker_threads(0);
        assert_eq!(worker_threads(), 0);
    }
}
