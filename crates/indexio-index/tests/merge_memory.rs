//! The merge works in bounded memory: its peak heap is a small fraction of
//! the raw content it merges, where the first implementation held all of it
//! (and a heap block per (doc, gram)) until the compound was written.
//!
//! One test in a binary of its own, with a counting allocator: nothing else
//! allocates while it measures.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use indexio_index::{Shard, ShardSet, ShardWriter};
use indexio_types::{BlobId, CallRec, DocMeta, ExtractedArtifact, Lang, SymbolKind, SymbolRec};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(n: usize) {
    let now = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(now, Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            grew(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(l);
        if !p.is_null() {
            grew(l.size());
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
            grew(new);
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// The highest the live heap rose above its level at entry while `f` ran.
fn peak_heap_of<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let out = f();
    (out, PEAK.load(Relaxed).saturating_sub(base))
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// Text of `len` bytes over a vocabulary of a few thousand words: source-like
/// in how many distinct trigrams a doc has, and compressible.
fn text(rng: &mut Lcg, len: usize) -> Vec<u8> {
    const SYLLABLES: [&str; 16] = ["ka", "to", "mi", "re", "sun", "del", "pro", "fix", "ar", "ne", "lo", "vi", "st", "qu", "em", "ox"];
    let mut out = Vec::with_capacity(len + 16);
    while out.len() < len {
        let w = rng.next();
        for k in 0..2 + (w % 3) {
            out.extend_from_slice(SYLLABLES[((w >> (4 + 4 * k)) & 15) as usize].as_bytes());
        }
        out.push(if w % 11 == 0 { b'\n' } else { b' ' });
    }
    out.truncate(len);
    out
}

/// Sorted unique trigrams of `content`; `seen` is a 2^24 table, left clear.
fn trigrams(content: &[u8], seen: &mut [bool]) -> Vec<Vec<u8>> {
    let mut found: Vec<u32> = Vec::new();
    for w in content.windows(3) {
        let key = (w[0] as u32) << 16 | (w[1] as u32) << 8 | w[2] as u32;
        if !seen[key as usize] {
            seen[key as usize] = true;
            found.push(key);
        }
    }
    found.sort_unstable();
    found
        .into_iter()
        .map(|key| {
            seen[key as usize] = false;
            vec![(key >> 16) as u8, (key >> 8) as u8, key as u8]
        })
        .collect()
}

/// One shard of `docs` docs of `doc_len` bytes at `path(i)`; returns the raw
/// bytes written. Every seventh doc is tombstoned afterwards.
fn write_shard(dir: &Path, rng: &mut Lcg, seen: &mut [bool], docs: usize, doc_len: usize, path: impl Fn(usize) -> String) -> u64 {
    let mut w = ShardWriter::new(dir).unwrap();
    let mut raw = 0u64;
    for i in 0..docs {
        let content = text(rng, doc_len);
        let symbols = (0..12)
            .map(|k| SymbolRec {
                name: format!("sym_{}", rng.next() % 500),
                kind: SymbolKind::Fn,
                line: k * 7 + 1,
                col: 0,
                scope: format!("Scope{}", rng.next() % 40),
            })
            .collect();
        let calls = (0..30)
            .map(|k| CallRec { callee: format!("sym_{}", rng.next() % 500), caller: format!("Scope{}::f{}", rng.next() % 40, k % 5), line: k + 1 })
            .collect();
        let meta = DocMeta { blob: BlobId::from_content(&content), repo_id: 0, path: path(i), lang: Lang::Rust, raw_len: content.len() as u32 };
        let art = ExtractedArtifact { ngrams: trigrams(&content, seen), symbols, calls, raw_len: content.len() as u32, lang: Lang::Rust };
        w.add_doc(&meta, &content, &art).unwrap();
        raw += content.len() as u64;
    }
    let shard_path = w.finish(&["repo".to_string()]).unwrap();
    let dead: Vec<u32> = (0..docs as u32).filter(|d| d % 7 == 3).collect();
    Shard::open(&shard_path).unwrap().delete_docs(&dead).unwrap();
    raw
}

/// Write `shards` shards and merge them into one; (raw bytes of the inputs,
/// peak heap of the merge, docs of the compound).
fn merge_of(shards: usize, docs: usize, doc_len: usize) -> (u64, usize, u64) {
    let dir = tempfile::tempdir().unwrap();
    let mut rng = Lcg(42);
    let mut seen = vec![false; 1 << 24];
    let mut raw = 0u64;
    for k in 0..shards {
        // a tenth of the paths of each shard are rewritten by the next one
        raw += write_shard(dir.path(), &mut rng, &mut seen, docs, doc_len, |i| {
            if i % 10 == 0 { format!("src/shared/f{i}.rs") } else { format!("src/s{k}/f{i}.rs") }
        });
    }
    drop(seen);
    let mut set = ShardSet::open_dir(dir.path()).unwrap();
    assert_eq!(set.len(), shards);
    let live_before = set.visible_docs().len();
    let ((), peak) = peak_heap_of(|| set.merge(dir.path(), 1).unwrap());
    assert_eq!(set.len(), 1);
    assert_eq!(set.visible_docs().len(), live_before);
    let kept = set.shards()[0].doc_count();
    assert_eq!(kept as usize, live_before);
    // the compound reads back
    let (si, docid, dm) = set.visible_docs().into_iter().last().unwrap();
    assert_eq!(BlobId::from_content(&set.content(si, docid).unwrap()), dm.blob);
    (raw, peak, kept)
}

#[test]
fn merge_peak_heap_is_a_fraction_of_the_raw_content() {
    const MB: f64 = (1 << 20) as f64;

    // Few large docs (no content dictionary below 101 docs): the merge holds
    // one doc and its frame at a time, 1 MiB of 48 MiB raw when this was
    // written. The in-memory merge held all of the kept content at once.
    let (raw, peak, kept) = merge_of(4, 24, 512 << 10);
    println!("large docs: {kept} docs kept, raw {:.1} MiB, merge peak heap {:.2} MiB", raw as f64 / MB, peak as f64 / MB);
    assert!(raw >= 48 << 20);
    assert!((peak as u64) < raw / 16, "merge peak heap {peak} bytes for {raw} raw bytes: over a sixteenth");

    // Many small docs: a dictionary is trained on the first 512 kept docs,
    // which are held back to back meanwhile (what zstd allocates itself is
    // not counted here). That buffer is the peak; the rest stays small.
    let doc_len = 6 << 10;
    let (raw, peak, kept) = merge_of(3, 500, doc_len);
    println!("small docs: {kept} docs kept, raw {:.1} MiB, merge peak heap {:.2} MiB", raw as f64 / MB, peak as f64 / MB);
    assert!(kept > 512 && raw >= 8 << 20);
    let samples = 512 * doc_len as u64;
    assert!((peak as u64) < samples + raw / 8, "merge peak heap {peak} bytes for {raw} raw bytes and {samples} bytes of dictionary samples");
}
