//! Writing the semantic plane must not hold the rows it writes.
//!
//! Every path that wrote a `.civec` / `.cibm25` pair used to collect its
//! rows first: a `Vec<f32>` and an owned metadata record per vector row, a
//! pair of term maps per BM25 row, then the file in one buffer. A rebuild of
//! a 986k-row plane reached 9.7 GB, a compaction 0.7 to 2.8 GB. The writers
//! now take a row at a time and keep their sections in spools.
//!
//! Each test here runs one write path at two sizes under a counting
//! allocator and holds the peak to an explicit bound: a fixed part (buffers,
//! one batch) plus a few bytes per row, where the old paths took kilobytes
//! per row. This binary has its own allocator and its tests take one lock,
//! so the numbers are those of the calls under measurement.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use indexio_embed::bm25::{Bm25Set, Bm25Writer};
use indexio_embed::embed::HashEmbedder;
use indexio_embed::index::{RowMeta, SegmentWriter, VecSet};
use indexio_embed::pipeline;

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    let now = LIVE.fetch_add(by, Ordering::Relaxed) + by;
    PEAK.fetch_max(now, Ordering::Relaxed);
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

/// The counters are process-wide: one measuring test at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// The most heap `f` held at once, over what was held when it started.
fn peak_of<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let live0 = LIVE.load(Ordering::Relaxed);
    PEAK.store(live0, Ordering::Relaxed);
    let out = f();
    (out, PEAK.load(Ordering::Relaxed).saturating_sub(live0))
}

const MB: usize = 1 << 20;
const DIM: usize = 256;
const ROWS_PER_FILE: usize = 5;

/// Row `i` of a generated plane, handed to `f` without being kept: 5 rows
/// per file, 300 files per repo, a dense vector with both signs.
fn with_row<T>(i: usize, scratch: &mut (String, String, Vec<f32>), f: impl FnOnce(&RowMeta<'_>, &[f32]) -> T) -> T {
    let file = i / ROWS_PER_FILE;
    let (repo, path, v) = scratch;
    if i % ROWS_PER_FILE == 0 || repo.is_empty() {
        *repo = format!("repo-{:03}", file / 300);
        *path = format!("src/module_{file:06}/nested/directory/source_file_{file}.rs");
    }
    v.clear();
    v.extend((0..DIM).map(|d| (((i * 31 + d * 7) % 17) as f32 - 8.0) * (1.0 + (d % 5) as f32)));
    let mut chunk_hash = [0u8; 16];
    chunk_hash[..8].copy_from_slice(&(i as u64).to_le_bytes());
    let chunk = (i % ROWS_PER_FILE) as u32;
    f(&RowMeta { chunk_hash, repo, path, start_line: chunk * 40 + 1, end_line: chunk * 40 + 40 }, v)
}

/// The (header, body) of row `i`: a path header and some 40 words of a
/// vocabulary with common and rare members.
fn text_of(i: usize, header: &mut String, body: &mut String) {
    use std::fmt::Write as _;
    header.clear();
    body.clear();
    let file = i / ROWS_PER_FILE;
    write!(header, "src/module_{file}/source_file_{file}.rs > fn handler_{i}").unwrap();
    let mut x = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for _ in 0..40 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let u = (x % 1000) as f64 / 1000.0;
        write!(body, "w{} ", (u * u * u * 20_000.0) as u64).unwrap();
    }
}

/// Write rows `from..to` as one segment pair of `data_dir` through the
/// streaming writers.
fn write_pair(data_dir: &Path, model: &str, rows: std::ops::Range<usize>) {
    let (vec_dir, bm25_dir) = (data_dir.join("vec"), data_dir.join("bm25"));
    let seg = indexio_embed::index::next_delta_path(&vec_dir, model);
    let bseg = indexio_embed::bm25::aligned_path(&bm25_dir, &seg);
    let mut vectors = SegmentWriter::new(&vec_dir, DIM).unwrap();
    let mut terms = Bm25Writer::new(&bm25_dir);
    let mut scratch = (String::new(), String::new(), Vec::new());
    let (mut header, mut body) = (String::new(), String::new());
    for i in rows {
        with_row(i, &mut scratch, |meta, v| vectors.push(meta, v)).unwrap();
        text_of(i, &mut header, &mut body);
        terms.push(&header, &body).unwrap();
    }
    terms.finish(&bseg).unwrap();
    vectors.finish(&seg).unwrap();
    // the next segment gets a later name
    std::thread::sleep(std::time::Duration::from_millis(3));
}

/// A vector segment written a row at a time: the rows that fit in memory
/// (4 MB), the smaller sections (1 MB each) and the file buffers, however
/// many rows there are. The writer it replaces took the rows as f32 (1 KB
/// each at this dim, 8 KB at the real one) and then the file twice.
#[test]
fn a_streamed_segment_holds_a_few_megabytes_whatever_its_rows() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().unwrap();
    let write = |name: &str, n: usize| {
        let path = tmp.path().join(name);
        let (_, peak) = peak_of(|| {
            let mut w = SegmentWriter::new(tmp.path(), DIM).unwrap();
            let mut scratch = (String::new(), String::new(), Vec::new());
            for i in 0..n {
                with_row(i, &mut scratch, |meta, v| w.push(meta, v)).unwrap();
            }
            // a chunk whose text came earlier in the pass copies that row
            w.push_copy_of(&RowMeta { chunk_hash: [7; 16], repo: "repo-000", path: "again.rs", start_line: 1, end_line: 2 }, 3).unwrap();
            w.finish(&path).unwrap();
        });
        (peak, std::fs::metadata(&path).unwrap().len() as usize)
    };
    const N: usize = 40_000;
    let (small, small_file) = write("m.d0000000000001.civec", N);
    let (large, large_file) = write("m.d0000000000002.civec", 2 * N);
    println!("streamed segment: {N} rows, {small_file} bytes: peak {small}; {} rows, {large_file} bytes: peak {large}", 2 * N);
    assert!(small_file > 10 * MB && large_file > 2 * small_file - MB, "{small_file} and {large_file} bytes written");
    // the rows as f32 alone, which the old writer was handed, are 4 x DIM x N
    assert!(small < 12 * MB, "{N} rows ({small_file} bytes) peaked at {small} bytes");
    assert!(large < 12 * MB, "{} rows ({large_file} bytes) peaked at {large} bytes", 2 * N);
    assert!(large < small + MB / 4, "twice the rows took {large} bytes, up from {small}");
    let set = VecSet::open(tmp.path(), "m").unwrap().unwrap();
    assert_eq!(set.raw_len(), 3 * N + 2);
}

/// A BM25 segment built a row at a time holds its term table up to the
/// budget, then merges the parts it set aside: the postings that fit in
/// memory (4 MB), the dictionary, and per row its lengths (twice, 12
/// bytes) and its posting in the list of the term being merged (here some
/// terms are in every row). The builder it replaces held two term maps per
/// row (some 3 KB for these rows) and the postings twice.
#[test]
fn a_bm25_build_holds_its_table_whatever_its_rows() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().unwrap();
    const BUDGET: usize = MB;
    let build = |name: &str, n: usize| {
        let path = tmp.path().join(name);
        let (_, peak) = peak_of(|| {
            let mut w = Bm25Writer::with_budget(tmp.path(), BUDGET);
            let (mut header, mut body) = (String::new(), String::new());
            for i in 0..n {
                text_of(i, &mut header, &mut body);
                w.push(&header, &body).unwrap();
            }
            w.finish(&path).unwrap();
        });
        (peak, std::fs::metadata(&path).unwrap().len() as usize)
    };
    const N: usize = 30_000;
    let (small, small_file) = build("m.d0000000000001.cibm25", N);
    let (large, large_file) = build("m.d0000000000002.cibm25", 2 * N);
    println!("bm25 build: {N} rows, {small_file} bytes: peak {small}; {} rows, {large_file} bytes: peak {large}", 2 * N);
    assert!(small_file > 2 * MB && large_file > small_file + small_file / 2, "{small_file} and {large_file} bytes written");
    // the table and its bookkeeping, or the postings in memory, the
    // dictionary being built and the file buffers
    let fixed = 2 * BUDGET + 8 * MB;
    assert!(small < fixed + 48 * N, "{N} rows ({small_file} bytes) peaked at {small} bytes");
    assert!(large < fixed + 48 * 2 * N, "{} rows ({large_file} bytes) peaked at {large} bytes", 2 * N);
    assert!(large < small + 48 * N + MB, "twice the rows took {large} bytes, up from {small}");
    let set = Bm25Set::open(tmp.path(), "m").unwrap().unwrap();
    assert_eq!(set.raw_len(), 3 * N);
    assert!(!set.search("w3 w17 handler", 5).is_empty());
}

/// A compaction carries rows from the mapped inputs into the merged pair:
/// what it holds per row is where the row comes from and where it goes
/// (some 60 bytes), not the row (1.2 KB of f32 and metadata at this dim,
/// 8.2 KB at the real one, plus two term maps).
#[test]
fn a_compaction_holds_a_few_dozen_bytes_per_row() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let compact = |n: usize| {
        let tmp = tempfile::tempdir().unwrap();
        // six segments, the last files first (the merge puts them in
        // order), a tenth of the rows tombstoned
        let per_segment = n / 6;
        for s in (0..6).rev() {
            write_pair(tmp.path(), "m", s * per_segment..(s + 1) * per_segment);
        }
        let mut set = VecSet::open(&tmp.path().join("vec"), "m").unwrap().unwrap();
        let mut terms = Bm25Set::open(&tmp.path().join("bm25"), "m").unwrap().unwrap();
        assert_eq!((set.segments().len(), set.raw_len(), terms.raw_len()), (6, 6 * per_segment, 6 * per_segment));
        set.delete_where(|m| m.chunk_hash[0] % 10 == 3);
        set.save_tombstones().unwrap();
        let dead: std::collections::HashSet<u32> = (0..set.raw_len() as u32).filter(|&r| set.is_deleted(r)).collect();
        terms.delete_where(|r| dead.contains(&r));
        terms.save_tombstones().unwrap();
        let live = set.len();
        drop((set, terms));
        let (merged, peak) = peak_of(|| pipeline::compact_vectors_opts(tmp.path(), "m", true).unwrap());
        assert!(merged);
        let set = VecSet::open(&tmp.path().join("vec"), "m").unwrap().unwrap();
        assert_eq!((set.segments().len(), set.raw_len()), (1, live));
        let bytes: u64 = set.segments().iter().map(|s| std::fs::metadata(s.path()).unwrap().len()).sum();
        (peak, live, bytes as usize)
    };
    const N: usize = 30_000;
    let (small, small_rows, small_file) = compact(N);
    let (large, large_rows, large_file) = compact(2 * N);
    println!("compaction: {small_rows} rows, {small_file} bytes: peak {small}; {large_rows} rows, {large_file} bytes: peak {large}");
    assert!(small_rows > N * 8 / 10 && large_rows > 2 * N * 8 / 10);
    assert!(small_file > 6 * MB && large_file > 12 * MB, "{small_file} and {large_file} bytes written");
    // buffers of the two files being written, the dictionary, a term's postings
    let fixed = 8 * MB;
    assert!(small < fixed + 96 * N, "{small_rows} rows ({small_file} bytes) peaked at {small} bytes");
    assert!(large < fixed + 96 * 2 * N, "{large_rows} rows ({large_file} bytes) peaked at {large} bytes");
    assert!(large < small + 96 * N + MB, "twice the rows took {large} bytes, up from {small}");
}

/// A shard set of `files` generated source files in one repo, a short
/// function (one chunk) each; returns it with the bytes of source it holds.
fn shards_of(dir: &Path, files: usize) -> (indexio_index::ShardSet, usize) {
    use indexio_types::{BlobId, DocMeta, ExtractedArtifact, Lang};
    let shards_dir = dir.join("shards");
    std::fs::create_dir_all(&shards_dir).unwrap();
    let mut w = indexio_index::ShardWriter::new(&shards_dir).unwrap();
    let mut source = 0usize;
    for k in 0..files {
        let mut text = String::new();
        for j in 0..1 {
            text.push_str(&format!("pub fn handler_{k}_{j}(input: u32) -> u32 {{
"));
            for line in 0..2 {
                text.push_str(&format!("    let step_{line} = input.wrapping_mul({line}).wrapping_add(lookup_{k}_{j}_{line}(input));
"));
            }
            text.push_str("    input
}

");
        }
        source += text.len();
        let path = format!("src/module_{}/file_{k}.rs", k % 40);
        let meta = DocMeta {
            blob: BlobId::from_content(text.as_bytes()),
            repo_id: 0,
            path: path.clone(),
            lang: Lang::from_path(&path),
            raw_len: text.len() as u32,
        };
        let art = ExtractedArtifact { lang: meta.lang, raw_len: text.len() as u32, ..Default::default() };
        w.add_doc(&meta, text.as_bytes(), &art).unwrap();
    }
    w.finish(&["generated".to_string()]).unwrap();
    (indexio_index::ShardSet::open_dir(&shards_dir).unwrap(), source)
}

/// A full rebuild embeds a round of chunks at a time and hands the rows to
/// the writers. Its peak is that round of vectors and the writers'
/// buffers, plus the text of the repo being embedded: once in the spool of
/// its chunks while that is small enough to stay in memory, once for the
/// single `observe` call a model gets per repo when the text is new to it.
/// The old pass held, for every chunk of every repo until the segment was
/// written, its vector twice (2 x 8 KB of f32 at this dim) and its text
/// three times.
#[test]
fn a_rebuild_holds_a_round_of_vectors_not_the_corpus() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    const EMBED_DIM: usize = 2048;
    let rebuild = |files: usize| {
        let tmp = tempfile::tempdir().unwrap();
        let (shards, source) = shards_of(tmp.path(), files);
        let e = HashEmbedder::new(EMBED_DIM);
        let (reports, peak) = peak_of(|| pipeline::embed_all(&shards, tmp.path(), &e).unwrap());
        let chunks = reports[0].chunks as usize;
        assert_eq!(VecSet::open(&tmp.path().join("vec"), "hash-v1").unwrap().unwrap().raw_len(), chunks);
        // chunk text: the source and a header line per chunk
        (peak, chunks, source + 64 * chunks)
    };
    // the thread pool and the parsers start outside the measurement
    rebuild(8);
    const FILES: usize = 10_000;
    let (small, small_chunks, small_text) = rebuild(FILES);
    let (large, large_chunks, large_text) = rebuild(2 * FILES);
    println!("rebuild: {small_chunks} chunks, {small_text} bytes of text: peak {small}; {large_chunks} chunks, {large_text} bytes of text: peak {large}");
    assert_eq!((small_chunks, large_chunks), (FILES, 2 * FILES));
    let as_f32 = |chunks: usize| chunks * EMBED_DIM * 4;
    // one round of vectors (2048 of them), the spools' memory, the file buffers
    let fixed = 2048 * EMBED_DIM * 4 + 16 * MB;
    assert!(small < fixed + 3 * small_text, "{small_chunks} chunks ({small_text} bytes of text) peaked at {small} bytes");
    assert!(large < fixed + 3 * large_text, "{large_chunks} chunks ({large_text} bytes of text) peaked at {large} bytes");
    assert!(large < small + 3 * (large_text - small_text) + 2 * MB, "twice the chunks took {large} bytes, up from {small}");
    assert!(large < as_f32(large_chunks) / 3, "{large} bytes held for {} bytes of vectors", as_f32(large_chunks));
}
