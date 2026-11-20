//! Opening a vector segment must not put its row metadata on the heap.
//!
//! A `.civec` segment used to deserialize its metadata block into one
//! `VecRowMeta` per row: two heap strings and 72 inline bytes each. On a
//! 986k-row plane that was 193 of the 208 MB a freshly started MCP server
//! held, and every writer that opened the sets built the same 190 MB again
//! as a transient. Rows are now read from the mapping; what stays on the
//! heap is one `u32` per row and the file/repo run tables.
//!
//! This binary has its own counting allocator and its tests take one lock,
//! so the numbers are those of the calls under measurement.
//!
//! `measure_real_plane` (ignored) prints the same costs for a real data dir:
//! `INDEXIO_MEASURE_DATA=<data dir> cargo test --release -p indexio-embed
//! --test meta_alloc -- --ignored --nocapture`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use indexio_embed::bm25::Bm25Set;
use indexio_embed::index::{VecIndex, VecRowMeta, VecSet};
use indexio_embed::store::{chunk_hash, EmbedCas};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCS: AtomicU64 = AtomicU64::new(0);

fn grew(by: usize) {
    let now = LIVE.fetch_add(by, Ordering::Relaxed) + by;
    PEAK.fetch_max(now, Ordering::Relaxed);
    ALLOCS.fetch_add(1, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, layout, new_size) };
        if !q.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            grew(new_size);
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Heap cost of a call: bytes still held when it returned, the most it held
/// at once, and how many blocks it asked for.
#[derive(Debug)]
struct Cost {
    kept: usize,
    peak: usize,
    allocs: u64,
}

/// The counters are process-wide: one measuring test at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn measure<T>(f: impl FnOnce() -> T) -> (T, Cost) {
    let live0 = LIVE.load(Ordering::Relaxed);
    PEAK.store(live0, Ordering::Relaxed);
    let allocs0 = ALLOCS.load(Ordering::Relaxed);
    let out = f();
    let cost = Cost {
        kept: LIVE.load(Ordering::Relaxed).saturating_sub(live0),
        peak: PEAK.load(Ordering::Relaxed).saturating_sub(live0),
        allocs: ALLOCS.load(Ordering::Relaxed) - allocs0,
    };
    (out, cost)
}

const DIM: usize = 8;

/// `n` rows laid out file by file, 1 to 8 chunks per file, 400 files per repo.
fn rows(n: usize, tag: &str) -> Vec<(VecRowMeta, Vec<f32>)> {
    let mut out = Vec::with_capacity(n);
    let mut file = 0usize;
    while out.len() < n {
        let repo = format!("repo-{tag}-{}", file / 400);
        let path = format!("src/module_{file}/nested/directory/source_file_{file}.rs");
        for chunk in 0..1 + file % 8 {
            if out.len() == n {
                break;
            }
            let mut chunk_hash = [0u8; 16];
            chunk_hash[..8].copy_from_slice(&(out.len() as u64).to_le_bytes());
            let v: Vec<f32> = (0..DIM).map(|d| ((out.len() * 31 + d * 7) % 17) as f32 - 8.0).collect();
            out.push((
                VecRowMeta {
                    chunk_hash,
                    repo: repo.clone(),
                    path: path.clone(),
                    start_line: chunk as u32 * 40 + 1,
                    end_line: chunk as u32 * 40 + 40,
                },
                v,
            ));
        }
        file += 1;
    }
    out
}

#[test]
fn opening_a_segment_keeps_a_few_bytes_per_row() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    const N: usize = 60_000;
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let (seg_a, seg_b) = (dir.join("m.d0000000000001.civec"), dir.join("m.d0000000000002.civec"));
    VecIndex::create_segment(&seg_a, DIM, rows(N, "a")).unwrap();
    VecIndex::create_segment(&seg_b, DIM, rows(N / 3, "b")).unwrap();

    // the counter sees what the old representation cost: N owned rows
    let (owned, cost) = measure(|| rows(N, "a").into_iter().map(|(m, _)| m).collect::<Vec<VecRowMeta>>());
    assert!(cost.kept > 100 * N && cost.allocs > 2 * N as u64, "counting allocator is not counting: {cost:?}");
    drop(owned);

    // one segment: 4 bytes per row of offsets plus the run tables
    let (idx, cost) = measure(|| VecIndex::open_segment(&seg_a).unwrap());
    assert_eq!(idx.raw_len(), N);
    assert!(cost.kept <= 8 * N + 16 * 1024, "open keeps {} bytes for {N} rows: {cost:?}", cost.kept);
    assert!(cost.peak <= 12 * N + 16 * 1024, "open peaks at {} bytes for {N} rows: {cost:?}", cost.peak);
    assert!(cost.allocs < 100, "open made {} allocations for {N} rows", cost.allocs);

    // reading rows allocates nothing
    let (seen, cost) = measure(|| {
        let mut bytes = 0usize;
        for r in 0..N as u32 {
            let m = idx.row_meta(r);
            bytes += m.repo.len() + m.path.len() + usize::from(idx.row_hash(r)[0]) + m.start_line as usize;
        }
        let files = idx.file_runs(0..N as u32).count();
        let repos = idx.repo_runs().count();
        (bytes, files, repos)
    });
    assert!(seen.0 > 40 * N && seen.1 > N / 8 && seen.2 > 10);
    assert_eq!(cost.allocs, 0, "row access allocated: {cost:?}");
    drop(idx);

    // the whole set, opened the way the server and the writers open it
    let total = N + N / 3;
    let (set, cost) = measure(|| VecSet::open(dir, "m").unwrap().unwrap());
    assert_eq!((set.segments().len(), set.raw_len()), (2, total));
    assert!(cost.kept <= 8 * total + 32 * 1024, "set open keeps {} bytes for {total} rows: {cost:?}", cost.kept);
    assert!(cost.peak <= 12 * total + 32 * 1024, "set open peaks at {} bytes for {total} rows: {cost:?}", cost.peak);
    assert!(cost.allocs < 200, "set open made {} allocations for {total} rows", cost.allocs);

    // a writer opening the plane while the engine holds it parses nothing
    let (again, cost) = measure(|| VecSet::open(dir, "m").unwrap().unwrap());
    assert_eq!(again.raw_len(), total);
    assert!(cost.kept <= 16 * 1024 && cost.peak <= 32 * 1024, "a second open of the same segments cost {cost:?}");
    drop(again);

    // the per-file view writers use instead of a walk over every row
    let (files, cost) = measure(|| set.live_files().count());
    assert!(files > total / 8);
    assert_eq!(cost.allocs, 0, "live_files allocated: {cost:?}");
    let (ranges, cost) = measure(|| set.repo_rows("repo-b-3"));
    assert_eq!(ranges.len(), 1);
    assert!(cost.allocs <= 4, "repo_rows allocated {} blocks", cost.allocs);
}

/// Asking a seen-only embedding cache about a few chunks must not load the
/// hash of every chunk the model ever observed (1.37 M hashes on a real
/// data dir: 21 MB read into a 36-55 MB set, on each embed with a changed
/// chunk).
#[test]
fn seen_only_lookup_does_not_load_the_hash_file() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    const N: u32 = 400_000;
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("embcas").join("m");
    std::fs::create_dir_all(&dir).unwrap();
    let hash = |i: u32| chunk_hash(&i.to_le_bytes());
    let bytes: Vec<u8> = (0..N).flat_map(hash).collect();
    std::fs::write(dir.join("seen.idx"), &bytes).unwrap();
    let file_len = bytes.len();
    drop(bytes);

    // the chunks of one edited file: some seen before, some new
    let ask: Vec<[u8; 16]> = (0..40).map(|i| hash(if i % 2 == 0 { i * 9_001 } else { N + i })).collect();
    let cas = EmbedCas::open_seen_only(tmp.path(), "m");
    let (seen, cost) = measure(|| cas.contains_many(&ask));
    assert!(seen.iter().enumerate().all(|(i, &s)| s == (i % 2 == 0)));
    assert!(file_len > 6 << 20);
    assert!(cost.peak < 1 << 20, "a lookup of 40 hashes in a {file_len}-byte file peaked at {cost:?}");
    assert!(cost.kept < 64 << 10, "a lookup of 40 hashes kept {cost:?}");

    // recording the new ones appends them and loads nothing either
    let (_, cost) = measure(|| {
        for h in &ask {
            cas.put(h, &[0.0]);
        }
        cas.flush();
    });
    assert!(cost.peak < 1 << 20, "puts peaked at {cost:?}");
    drop(cas);
    assert_eq!(std::fs::metadata(dir.join("seen.idx")).unwrap().len(), file_len as u64 + 20 * 16);
}

/// What opening the semantic plane of a real data dir costs, next to what
/// deserializing its metadata blocks (the previous open) costs.
#[test]
#[ignore = "measurement: set INDEXIO_MEASURE_DATA to a data dir"]
fn measure_real_plane() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(data) = std::env::var_os("INDEXIO_MEASURE_DATA").map(std::path::PathBuf::from) else {
        eprintln!("INDEXIO_MEASURE_DATA is not set");
        return;
    };
    let model = std::env::var("INDEXIO_MEASURE_MODEL").unwrap_or_else(|_| "rindex-v2".into());
    let mb = |b: usize| b as f64 / 1048576.0;
    let timed = |what: &str, cost: &Cost, t: Instant| {
        println!(
            "{what:<44} {:>8.1} ms  kept {:>7.2} MB  peak {:>7.2} MB  {:>8} allocs",
            t.elapsed().as_secs_f64() * 1e3,
            mb(cost.kept),
            mb(cost.peak),
            cost.allocs
        );
    };
    let pp = indexio_embed::pipeline::plane_paths(&data, &model);
    println!("{} vector segments, {} bm25 segments", pp.vec.len(), pp.bm25.len());

    for round in ["cold", "warm"] {
        let t = Instant::now();
        let (set, cost) = measure(|| VecSet::reopen_paths(None, &data.join("vec"), &model, &pp.vec).unwrap().unwrap());
        timed(&format!("VecSet open ({round}), {} rows", set.raw_len()), &cost, t);
        let t = Instant::now();
        let (bset, cost) = measure(|| Bm25Set::reopen_paths(None, &pp.bm25).unwrap().unwrap());
        timed(&format!("Bm25Set open ({round}), {} rows", bset.raw_len()), &cost, t);
        if round == "warm" {
            let t = Instant::now();
            let (again, cost) = measure(|| {
                (
                    VecSet::reopen_paths(None, &data.join("vec"), &model, &pp.vec).unwrap().unwrap(),
                    Bm25Set::reopen_paths(None, &pp.bm25).unwrap().unwrap(),
                )
            });
            timed("both sets again while the first are open", &cost, t);
            drop(again);
            let t = Instant::now();
            let (files, cost) = measure(|| set.live_files().collect::<std::collections::HashSet<(&str, &str)>>());
            timed(&format!("live_files into a set ({} files)", files.len()), &cost, t);
            drop(files);
            let t = Instant::now();
            let (bytes, cost) = measure(|| (0..set.raw_len() as u32).map(|g| set.row_meta(g)).map(|m| m.repo.len() + m.path.len()).sum::<usize>());
            timed(&format!("row_meta of every row ({} string bytes)", bytes), &cost, t);
            let t = Instant::now();
            let (ranges, cost) = measure(|| set.repo_rows("indexio"));
            timed(&format!("repo_rows(\"indexio\") ({} ranges)", ranges.len()), &cost, t);
        }
    }

    // the seen-only embedding cache: a lookup of a few chunk hashes
    let cas = EmbedCas::open_seen_only(&data, &model);
    let ask: Vec<[u8; 16]> = (0..64u32).map(|i| chunk_hash(&i.to_le_bytes())).collect();
    let t = Instant::now();
    let (seen, cost) = measure(|| cas.contains_many(&ask));
    timed(&format!("EmbedCas lookup of 64 hashes ({} seen)", seen.iter().filter(|s| **s).count()), &cost, t);
    let t = Instant::now();
    let (stats, cost) = measure(|| cas.stats());
    timed(&format!("EmbedCas stats ({} hashes)", stats.0), &cost, t);
    drop(cas);
    let t = Instant::now();
    let (n, cost) = measure(|| {
        let bytes = std::fs::read(data.join("embcas").join(&model).join("seen.idx")).unwrap_or_default();
        let set: std::collections::HashSet<[u8; 16]> = bytes.chunks_exact(16).map(|c| c.try_into().unwrap()).collect();
        set.len()
    });
    timed(&format!("loading {n} hashes into a set (before)"), &cost, t);

    // the previous open: one owned VecRowMeta per row
    let (mut kept, mut allocs, mut ms) = (0usize, 0u64, 0.0f64);
    for p in &pp.vec {
        let file = std::fs::File::open(p).unwrap();
        let map = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        let meta_len = u64::from_le_bytes(map[20..28].try_into().unwrap()) as usize;
        let t = Instant::now();
        let (rows, cost) = measure(|| bincode::deserialize::<Vec<VecRowMeta>>(&map[28..28 + meta_len]).unwrap());
        ms += t.elapsed().as_secs_f64() * 1e3;
        kept += cost.kept;
        allocs += cost.allocs;
        drop(rows);
    }
    println!("{:<44} {ms:>8.1} ms  kept {:>7.2} MB  {allocs:>26} allocs", "deserializing every meta block (before)", mb(kept));
}
