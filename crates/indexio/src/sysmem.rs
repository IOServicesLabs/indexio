//! What the operating system says about this process's memory, and the one
//! call that hands freed heap back to it.
//!
//! The maintenance child reports its own peak through [`process_memory`]
//! (the number that used to stay inside whichever server ran the pass), and
//! the server's idle trim calls [`release_free_heap`] after dropping its
//! caches.

// (with the experimental `mimalloc` feature the system heap is not ours to trim)
#![cfg_attr(feature = "mimalloc", allow(dead_code))]

/// Bytes, as the system accounts them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProcessMemory {
    /// Committed private memory (Windows "private bytes"; anonymous resident
    /// memory on Linux).
    pub private: u64,
    /// The highest `private` since the process started (0 where unknown).
    pub peak_private: u64,
    /// Resident memory (working set).
    pub resident: u64,
    /// The highest `resident` since the process started.
    pub peak_resident: u64,
}

#[cfg(windows)]
#[allow(unsafe_code, non_snake_case)]
mod sys {
    use super::ProcessMemory;
    use std::ffi::c_void;

    /// `PROCESS_MEMORY_COUNTERS_EX`
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(process: *mut c_void, counters: *mut Counters, cb: u32) -> i32;
        fn GetProcessHeap() -> *mut c_void;
        fn HeapCompact(heap: *mut c_void, flags: u32) -> usize;
    }

    pub fn process_memory() -> Option<ProcessMemory> {
        let mut c = Counters { cb: std::mem::size_of::<Counters>() as u32, ..Default::default() };
        // SAFETY: `c` is a correctly sized, writable PROCESS_MEMORY_COUNTERS_EX
        // and the pseudo handle of the current process is always valid.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
        (ok != 0).then_some(ProcessMemory {
            private: c.private_usage as u64,
            peak_private: c.peak_pagefile_usage as u64,
            resident: c.working_set_size as u64,
            peak_resident: c.peak_working_set_size as u64,
        })
    }

    /// `HeapCompact` on the default process heap, the one Rust's allocator
    /// and the C runtime allocate from: adjacent free blocks are coalesced
    /// and large free ones decommitted. Returns the size of the largest
    /// committed free block left.
    ///
    /// Only that heap: the others belong to system libraries, and one
    /// created without serialization must not be touched from a thread its
    /// owner does not know about.
    pub fn release_free_heap() -> usize {
        // SAFETY: the default heap exists for the life of the process and is
        // serialized; with no flags this call takes its lock like any other.
        unsafe { HeapCompact(GetProcessHeap(), 0) }
    }
}

#[cfg(target_os = "linux")]
mod sys {
    use super::ProcessMemory;

    fn kb(status: &str, key: &str) -> u64 {
        status
            .lines()
            .find_map(|l| l.strip_prefix(key))
            .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
            .map_or(0, |n| n * 1024)
    }

    pub fn process_memory() -> Option<ProcessMemory> {
        let s = std::fs::read_to_string("/proc/self/status").ok()?;
        Some(ProcessMemory {
            private: kb(&s, "RssAnon:"),
            peak_private: 0,
            resident: kb(&s, "VmRSS:"),
            peak_resident: kb(&s, "VmHWM:"),
        })
    }

    /// glibc keeps freed memory below the top of the heap; `malloc_trim`
    /// returns what it can. Other C libraries have no equivalent.
    #[cfg(target_env = "gnu")]
    #[allow(unsafe_code)]
    pub fn release_free_heap() -> usize {
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        // SAFETY: no arguments beyond the padding to keep; thread-safe in glibc.
        unsafe { malloc_trim(0) as usize }
    }

    #[cfg(not(target_env = "gnu"))]
    pub fn release_free_heap() -> usize {
        0
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod sys {
    use super::ProcessMemory;

    pub fn process_memory() -> Option<ProcessMemory> {
        None
    }

    pub fn release_free_heap() -> usize {
        0
    }
}

/// This process's memory as the system sees it; `None` where there is no
/// cheap way to ask.
pub fn process_memory() -> Option<ProcessMemory> {
    sys::process_memory()
}

/// Ask the allocator to give freed memory back to the system (Windows:
/// `HeapCompact` on the process heap; glibc: `malloc_trim`). The return
/// value is the platform call's own and only good for a log line.
#[cfg(not(feature = "mimalloc"))]
pub fn release_free_heap() -> usize {
    sys::release_free_heap()
}

/// With the `mimalloc` feature the heap is mimalloc's: collect it.
#[cfg(feature = "mimalloc")]
#[allow(unsafe_code)]
pub fn release_free_heap() -> usize {
    extern "C" {
        fn mi_collect(force: bool);
    }
    // SAFETY: a mimalloc entry point without preconditions; thread-safe.
    unsafe { mi_collect(true) };
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_memory_is_plausible_and_the_trim_is_harmless() {
        // allocate, free, trim: nothing may break and the numbers stay sane
        let big: Vec<Vec<u8>> = (0..128).map(|i| vec![1 + i as u8; 256 * 1024]).collect();
        let with = process_memory();
        drop(big);
        release_free_heap();
        let after = process_memory();
        if cfg!(any(windows, target_os = "linux")) {
            let (with, after) = (with.expect("memory counters"), after.expect("memory counters"));
            assert!(with.resident > 16 * 1024 * 1024, "{with:?}");
            assert!(after.peak_resident >= with.resident, "{after:?} vs {with:?}");
        }
    }
}
