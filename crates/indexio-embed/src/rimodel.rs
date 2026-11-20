//! Flat, memory-mappable rindex model file.
//!
//! The bincode model (`Vec<(term, df, ctx)>`) had to be read and
//! deserialised whole — 250 MB of reads and ~300 MB of heap per server
//! start, once per session, for a vocabulary a query touches a few hundred
//! entries of. This layout is mapped instead: opening costs the header,
//! lookups touch one hash bucket, one entry record, the term bytes and the
//! context block, and every session shares the page cache.
//!
//! ```text
//! magic "RIMDL003" | u32 dim | u32 n_terms | u64 n_texts_seen | u32 n_buckets
//! | u32 reserved | u64 off_buckets | u64 off_entries | u64 off_terms | u64 off_ctx
//! | (pad to 64 bytes)
//! buckets: n_buckets x u32   term index + 1 (0 = empty), open addressing,
//!                            linear probing, FNV-1a of the term bytes
//! entries: n_terms x 16 B    u32 term_off | u16 term_len | u16 ctx_len
//!                            | u32 df | u32 ctx_off (in components)
//! terms:   concatenated UTF-8 term bytes (entries are sorted by term)
//! ctx:     components, 8 B each: u32 dim | f32 value  (8-byte aligned)
//! ```
//! The file is written to a temp path and renamed into place, never
//! modified — the same immutability argument as the shard and vector
//! mappings.
//!
//! Writing streams: [`Snapshot::write_sorted`] walks its (sorted) source
//! once per section and holds only the bucket table, so a save of the
//! 250,000-term model costs about 2 MiB of heap instead of a copy of the
//! vocabulary plus the whole file image (~0.5 GB). Every save writes its
//! own temp file (`<model>.tmp.<pid>.<n>`): the servers of several sessions
//! share one model file and save on their own schedule.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{anyhow, bail, ensure, Context};
use memmap2::Mmap;

const MAGIC: &[u8; 8] = b"RIMDL003";
const HEADER_LEN: usize = 64;
const ENTRY_LEN: usize = 16;

/// Bytes buffered in front of the temp file while a snapshot is written.
const WRITE_BUFFER: usize = 256 * 1024;

/// A temp file nobody has written to for this long belongs to a writer that
/// died: a live one finishes within seconds of creating its file.
const STALE_TEMP_AFTER: Duration = Duration::from_secs(15 * 60);

/// One entry of a snapshot, borrowed from wherever it lives: `(term, df,
/// ctx)`.
pub type EntryView<'a> = (&'a str, u32, &'a [Comp]);

/// One sparse context component: `(dim, value)`, the on-disk layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comp {
    pub d: u32,
    pub v: f32,
}

/// FNV-1a over the term bytes (the vocabulary shard hash, reused for the
/// bucket table so it needs no hasher state).
pub fn fnv1a(term: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in term.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A mapped model file.
pub struct Snapshot {
    map: Mmap,
    dim: u32,
    n_terms: usize,
    n_texts_seen: u64,
    n_buckets: usize,
    off_buckets: usize,
    off_entries: usize,
    off_terms: usize,
    off_ctx: usize,
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}
fn u16_at(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(b[off..off + 2].try_into().unwrap())
}
fn u64_at(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

/// Whether `path` holds a flat model (magic check on the first 8 bytes).
pub fn is_snapshot(path: &Path) -> bool {
    let mut magic = [0u8; 8];
    fs::File::open(path)
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic))
        .is_ok_and(|_| &magic == MAGIC)
}

impl Snapshot {
    /// Map `path`; validates the header and every block bound.
    ///
    /// # Safety (why the mapping is sound)
    /// The file is written once (tmp + rename) and never modified in place;
    /// a replacement lands under the same name as a new inode, which an
    /// existing mapping does not observe.
    #[allow(unsafe_code)]
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let map = unsafe { Mmap::map(&file) }.with_context(|| format!("map {}", path.display()))?;
        let bad = || anyhow!("{}: corrupt rindex model file", path.display());
        ensure!(map.len() >= HEADER_LEN && &map[..8] == MAGIC, "{}: not a flat rindex model", path.display());
        let dim = u32_at(&map, 8);
        let n_terms = u32_at(&map, 12) as usize;
        let n_texts_seen = u64_at(&map, 16);
        let n_buckets = u32_at(&map, 24) as usize;
        let off_buckets = u64_at(&map, 32) as usize;
        let off_entries = u64_at(&map, 40) as usize;
        let off_terms = u64_at(&map, 48) as usize;
        let off_ctx = u64_at(&map, 56) as usize;
        let end_buckets = off_buckets.checked_add(n_buckets.checked_mul(4).ok_or_else(bad)?).ok_or_else(bad)?;
        let end_entries = off_entries.checked_add(n_terms.checked_mul(ENTRY_LEN).ok_or_else(bad)?).ok_or_else(bad)?;
        ensure!(end_buckets <= map.len() && end_entries <= map.len() && off_terms <= map.len() && off_ctx <= map.len(), bad());
        ensure!(n_buckets.is_power_of_two() || n_terms == 0, bad());
        ensure!((map.as_ptr() as usize + off_ctx) % 8 == 0 && (map.as_ptr() as usize + off_buckets) % 4 == 0, bad());
        Ok(Snapshot { map, dim, n_terms, n_texts_seen, n_buckets, off_buckets, off_entries, off_terms, off_ctx })
    }

    pub fn dim(&self) -> u32 {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.n_terms
    }

    pub fn is_empty(&self) -> bool {
        self.n_terms == 0
    }

    pub fn n_texts_seen(&self) -> u64 {
        self.n_texts_seen
    }

    /// `(term, df, ctx)` of entry `i` (entries are sorted by term).
    #[allow(unsafe_code)]
    pub fn entry(&self, i: usize) -> (&str, u32, &[Comp]) {
        let e = self.off_entries + i * ENTRY_LEN;
        let b = &self.map[..];
        let term_off = u32_at(b, e) as usize;
        let term_len = u16_at(b, e + 4) as usize;
        let ctx_len = u16_at(b, e + 6) as usize;
        let df = u32_at(b, e + 8);
        let ctx_off = u32_at(b, e + 12) as usize;
        let term = std::str::from_utf8(&b[self.off_terms + term_off..self.off_terms + term_off + term_len]).unwrap_or("");
        let start = self.off_ctx + ctx_off * 8;
        let bytes = &b[start..start + ctx_len * 8];
        // `Comp` is repr(C) {u32, f32} = 8 bytes; the block is 8-aligned (checked in `open`)
        let ctx = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const Comp, ctx_len) };
        (term, df, ctx)
    }

    /// Bucket-table lookup: `(df, ctx)` of `term`.
    pub fn get(&self, term: &str) -> Option<(u32, &[Comp])> {
        if self.n_terms == 0 {
            return None;
        }
        let mask = self.n_buckets - 1;
        let mut i = (fnv1a(term) as usize) & mask;
        loop {
            let slot = u32_at(&self.map, self.off_buckets + i * 4) as usize;
            if slot == 0 {
                return None;
            }
            let (t, df, ctx) = self.entry(slot - 1);
            if t == term {
                return Some((df, ctx));
            }
            i = (i + 1) & mask;
        }
    }

    /// Write a snapshot: `entries` sorted by term. Temp file + rename.
    pub fn write(path: &Path, dim: u32, n_texts_seen: u64, entries: &[(String, u32, Vec<Comp>)]) -> anyhow::Result<()> {
        Self::write_sorted(path, dim, n_texts_seen, || entries.iter().map(|(t, df, ctx)| (t.as_str(), *df, ctx.as_slice())))
    }

    /// Write a snapshot of the entries `entries()` yields, without holding
    /// them or the file image in memory.
    ///
    /// `entries` is called once per pass (sizes, bucket table, entry
    /// records, term bytes, contexts) and must yield the same sequence each
    /// time, sorted by term: the caller keeps its source unchanged for the
    /// duration (the model saves under its write lock). Heap use is the
    /// bucket table (4 B per bucket, at most 16 B per term) and the write
    /// buffer.
    ///
    /// The bytes go to a temp file of this call's own
    /// (`<path>.tmp.<pid>.<n>`, created new), which is renamed into place
    /// when complete and removed on any error.
    pub fn write_sorted<'a, I>(path: &Path, dim: u32, n_texts_seen: u64, entries: impl Fn() -> I) -> anyhow::Result<()>
    where
        I: Iterator<Item = EntryView<'a>>,
    {
        const CHANGED: &str = "the model changed while it was being written";
        // sizes
        let (mut n, mut term_bytes, mut ctx_comps) = (0usize, 0usize, 0usize);
        let mut prev: Option<&str> = None;
        for (t, _, ctx) in entries() {
            ensure!(t.len() <= u16::MAX as usize && ctx.len() <= u16::MAX as usize, "term or context too long");
            debug_assert!(prev.is_none_or(|p| p < t), "entries must be sorted by term");
            prev = Some(t);
            n += 1;
            term_bytes += t.len();
            ctx_comps += ctx.len();
        }
        ensure!(
            n < u32::MAX as usize && term_bytes <= u32::MAX as usize && ctx_comps <= u32::MAX as usize,
            "model too large for the snapshot layout"
        );
        // bucket table
        let n_buckets = (n * 2).next_power_of_two().max(16);
        let mut buckets = vec![0u32; n_buckets];
        let mask = n_buckets - 1;
        let mut seen = 0usize;
        for (t, _, _) in entries() {
            ensure!(seen < n, CHANGED);
            let mut b = (fnv1a(t) as usize) & mask;
            while buckets[b] != 0 {
                b = (b + 1) & mask;
            }
            seen += 1;
            buckets[b] = seen as u32;
        }
        ensure!(seen == n, CHANGED);
        let align8 = |v: usize| (v + 7) & !7;
        let off_buckets = HEADER_LEN;
        let off_entries = align8(off_buckets + n_buckets * 4);
        let off_terms = off_entries + n * ENTRY_LEN;
        let off_ctx = align8(off_terms + term_bytes);
        let total = off_ctx + ctx_comps * 8;
        let mut header = [0u8; HEADER_LEN];
        header[..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&dim.to_le_bytes());
        header[12..16].copy_from_slice(&(n as u32).to_le_bytes());
        header[16..24].copy_from_slice(&n_texts_seen.to_le_bytes());
        header[24..28].copy_from_slice(&(n_buckets as u32).to_le_bytes());
        header[32..40].copy_from_slice(&(off_buckets as u64).to_le_bytes());
        header[40..48].copy_from_slice(&(off_entries as u64).to_le_bytes());
        header[48..56].copy_from_slice(&(off_terms as u64).to_le_bytes());
        header[56..64].copy_from_slice(&(off_ctx as u64).to_le_bytes());

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let (mut tmp, file) = TempFile::create(path)?;
        let written = (|| -> anyhow::Result<()> {
            let mut out = Out { w: BufWriter::with_capacity(WRITE_BUFFER, &file), at: 0 };
            out.put(&header)?;
            for b in &buckets {
                out.put(&b.to_le_bytes())?;
            }
            out.pad_to(off_entries)?;
            let (mut term_off, mut ctx_off, mut count) = (0usize, 0usize, 0usize);
            for (t, df, ctx) in entries() {
                let mut rec = [0u8; ENTRY_LEN];
                rec[..4].copy_from_slice(&(term_off as u32).to_le_bytes());
                rec[4..6].copy_from_slice(&(t.len() as u16).to_le_bytes());
                rec[6..8].copy_from_slice(&(ctx.len() as u16).to_le_bytes());
                rec[8..12].copy_from_slice(&df.to_le_bytes());
                rec[12..].copy_from_slice(&(ctx_off as u32).to_le_bytes());
                out.put(&rec)?;
                term_off += t.len();
                ctx_off += ctx.len();
                count += 1;
            }
            ensure!(count == n && term_off == term_bytes && ctx_off == ctx_comps, CHANGED);
            for (t, _, _) in entries() {
                out.put(t.as_bytes())?;
            }
            ensure!(out.at == off_terms + term_bytes, CHANGED);
            out.pad_to(off_ctx)?;
            // (a context is at most CTX_TOP = 128 components: one buffer)
            let mut buf = [0u8; 8 * 128];
            for (_, _, ctx) in entries() {
                for part in ctx.chunks(buf.len() / 8) {
                    for (c, b) in part.iter().zip(buf.chunks_exact_mut(8)) {
                        b[..4].copy_from_slice(&c.d.to_le_bytes());
                        b[4..].copy_from_slice(&c.v.to_le_bytes());
                    }
                    out.put(&buf[..part.len() * 8])?;
                }
            }
            ensure!(out.at == total, CHANGED);
            out.w.flush().context("writing the model snapshot")?;
            file.sync_all().ok();
            Ok(())
        })();
        // closed before anything else: an open temp file can be neither
        // renamed nor (on an error, by `tmp`) removed on Windows
        drop(file);
        drop(buckets);
        written.with_context(|| format!("writing {}", tmp.path.display()))?;
        // The previous file may be mapped by other servers: on Windows a
        // rename over it fails, so park it first (reaped by `reap_parked`).
        if path.exists() {
            let parked = path.with_extension(format!(
                "rimodel.stale{}.{}",
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
                std::process::id()
            ));
            let _ = fs::rename(path, &parked);
        }
        fs::rename(&tmp.path, path).with_context(|| format!("renaming to {}", path.display()))?;
        tmp.renamed = true;
        reap_parked(path);
        reap_stale_temps(path);
        Ok(())
    }

    /// The file image the first writer built in memory before writing it
    /// (it also held `entries` as owned copies): what
    /// [`write_sorted`](Self::write_sorted) must reproduce byte for byte.
    /// (It asserted, in debug builds only, that `entries` are sorted; the
    /// tests also use it to forge a snapshot that is not.)
    #[cfg(test)]
    pub(crate) fn materialized_image(dim: u32, n_texts_seen: u64, entries: &[(String, u32, Vec<Comp>)]) -> anyhow::Result<Vec<u8>> {
        let n = entries.len();
        let n_buckets = (n * 2).next_power_of_two().max(16);
        let mut buckets = vec![0u32; n_buckets];
        let mask = n_buckets - 1;
        for (i, (t, _, _)) in entries.iter().enumerate() {
            let mut b = (fnv1a(t) as usize) & mask;
            while buckets[b] != 0 {
                b = (b + 1) & mask;
            }
            buckets[b] = i as u32 + 1;
        }
        let mut rec = Vec::with_capacity(n * ENTRY_LEN);
        let mut terms = Vec::new();
        let mut ctx_comps: usize = 0;
        for (t, df, ctx) in entries {
            ensure!(t.len() <= u16::MAX as usize && ctx.len() <= u16::MAX as usize, "term or context too long");
            rec.extend_from_slice(&(terms.len() as u32).to_le_bytes());
            rec.extend_from_slice(&(t.len() as u16).to_le_bytes());
            rec.extend_from_slice(&(ctx.len() as u16).to_le_bytes());
            rec.extend_from_slice(&df.to_le_bytes());
            rec.extend_from_slice(&(ctx_comps as u32).to_le_bytes());
            terms.extend_from_slice(t.as_bytes());
            ctx_comps += ctx.len();
        }
        let align8 = |v: usize| (v + 7) & !7;
        let off_buckets = HEADER_LEN;
        let off_entries = align8(off_buckets + n_buckets * 4);
        let off_terms = off_entries + n * ENTRY_LEN;
        let off_ctx = align8(off_terms + terms.len());
        let total = off_ctx + ctx_comps * 8;
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&dim.to_le_bytes());
        out.extend_from_slice(&(n as u32).to_le_bytes());
        out.extend_from_slice(&n_texts_seen.to_le_bytes());
        out.extend_from_slice(&(n_buckets as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(off_buckets as u64).to_le_bytes());
        out.extend_from_slice(&(off_entries as u64).to_le_bytes());
        out.extend_from_slice(&(off_terms as u64).to_le_bytes());
        out.extend_from_slice(&(off_ctx as u64).to_le_bytes());
        out.resize(HEADER_LEN, 0);
        for b in &buckets {
            out.extend_from_slice(&b.to_le_bytes());
        }
        out.resize(off_entries, 0);
        out.extend_from_slice(&rec);
        out.extend_from_slice(&terms);
        out.resize(off_ctx, 0);
        for (_, _, ctx) in entries {
            for c in ctx {
                out.extend_from_slice(&c.d.to_le_bytes());
                out.extend_from_slice(&c.v.to_le_bytes());
            }
        }
        Ok(out)
    }
}

/// The temp file's writer, counting what went through it.
struct Out<'f> {
    w: BufWriter<&'f fs::File>,
    at: usize,
}

impl Out<'_> {
    fn put(&mut self, bytes: &[u8]) -> anyhow::Result<()> {
        self.w.write_all(bytes).context("writing the model snapshot")?;
        self.at += bytes.len();
        Ok(())
    }

    /// Zero bytes up to file offset `off` (section alignment).
    fn pad_to(&mut self, off: usize) -> anyhow::Result<()> {
        ensure!(off >= self.at && off - self.at < 8, "snapshot section out of place");
        self.put(&[0u8; 8][..off - self.at])
    }
}

/// The temp file of one save, removed again unless it was renamed into
/// place.
struct TempFile {
    path: PathBuf,
    renamed: bool,
}

impl TempFile {
    /// Create a temp file next to `path` under a name no other save uses
    /// (the process id and a per-process counter), new: a leftover of a
    /// dead process that had the same id is never written into. On Windows
    /// nothing else can delete, rename or write the file while it is open.
    fn create(path: &Path) -> anyhow::Result<(Self, fs::File)> {
        static SAVES: AtomicU64 = AtomicU64::new(0);
        for _ in 0..64 {
            let mut name = path.as_os_str().to_owned();
            name.push(format!(".tmp.{}.{}", std::process::id(), SAVES.fetch_add(1, Ordering::Relaxed)));
            let tmp = PathBuf::from(name);
            let mut options = fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(windows)]
            {
                use std::os::windows::fs::OpenOptionsExt;
                const FILE_SHARE_READ: u32 = 1;
                options.share_mode(FILE_SHARE_READ);
            }
            match options.open(&tmp) {
                Ok(f) => return Ok((TempFile { path: tmp, renamed: false }, f)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e).with_context(|| format!("writing {}", tmp.display())),
            }
        }
        bail!("no free temp file name next to {}", path.display())
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        if !self.renamed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Remove parked (`.stale*`) predecessors of `path` that no process maps
/// any more; the ones still mapped are left for a later call.
pub fn reap_parked(path: &Path) {
    let Some(dir) = path.parent() else { return };
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else { return };
    let prefix = format!("{name}.stale");
    if let Ok(rd) = fs::read_dir(dir) {
        for e in rd.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&prefix) {
                let _ = fs::remove_file(e.path());
            }
        }
    }
}

/// Remove the temp files of saves that never finished because their writer
/// died: `<path>.tmp.<pid>.<n>`, and `<path>.tmp`, the one name every
/// process shared up to release 0.1.14. Only files not written to for
/// [`STALE_TEMP_AFTER`] go: a live writer's file is seconds old (and on
/// Windows cannot be removed while it is open), and a 0.1.14 server may be
/// writing `<path>.tmp` right now.
pub fn reap_stale_temps(path: &Path) {
    reap_temps_older_than(path, STALE_TEMP_AFTER);
}

fn reap_temps_older_than(path: &Path, age: Duration) {
    let Some(dir) = path.parent() else { return };
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else { return };
    let prefix = format!("{name}.tmp");
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        let Some(rest) = name.strip_prefix(&prefix) else { continue };
        if !(rest.is_empty() || rest.starts_with('.')) {
            continue;
        }
        let idle = e.metadata().ok().and_then(|m| m.modified().ok()).and_then(|t| t.elapsed().ok());
        if idle.is_some_and(|idle| idle >= age) {
            let _ = fs::remove_file(e.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_lookup_and_iteration() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("sem").join("m.rimodel");
        let mut entries: Vec<(String, u32, Vec<Comp>)> = (0..1000)
            .map(|i| {
                (
                    format!("term{i:04}"),
                    i as u32 % 7,
                    (0..(i % 5)).map(|k| Comp { d: (k * 10 + i % 3) as u32, v: i as f32 * 0.5 + k as f32 }).collect(),
                )
            })
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Snapshot::write(&path, 2048, 42, &entries).unwrap();
        assert!(is_snapshot(&path));
        let s = Snapshot::open(&path).unwrap();
        assert_eq!((s.dim(), s.len(), s.n_texts_seen()), (2048, 1000, 42));
        for (i, (t, df, ctx)) in entries.iter().enumerate() {
            let (t2, df2, ctx2) = s.entry(i);
            assert_eq!((t2, df2, ctx2), (t.as_str(), *df, ctx.as_slice()));
            assert_eq!(s.get(t), Some((*df, ctx.as_slice())));
        }
        assert!(s.get("nope").is_none());
        assert!(s.get("").is_none());
        // rewrite parks the old file and replaces it
        Snapshot::write(&path, 2048, 43, &entries[..10]).unwrap();
        let s2 = Snapshot::open(&path).unwrap();
        assert_eq!((s2.len(), s2.n_texts_seen()), (10, 43));
        drop(s);
        reap_parked(&path);
        let stale: Vec<_> = fs::read_dir(path.parent().unwrap()).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().contains("stale")).collect();
        assert!(stale.is_empty(), "{stale:?}");
    }

    #[test]
    fn empty_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("m.rimodel");
        Snapshot::write(&path, 2048, 0, &[]).unwrap();
        let s = Snapshot::open(&path).unwrap();
        assert!(s.is_empty());
        assert!(s.get("x").is_none());
    }

    /// `n` sorted entries with terms of every length class (ASCII and not),
    /// empty contexts, and contexts longer than the writer's 128-component
    /// buffer.
    fn generated(n: usize, seed: u64) -> Vec<(String, u32, Vec<Comp>)> {
        let mut x = seed.max(1);
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut entries: Vec<(String, u32, Vec<Comp>)> = (0..n)
            .map(|i| {
                let term = match next() % 5 {
                    0 => format!("t{i}"),
                    1 => format!("größe{i} straße"),
                    2 => format!("ident_{i}_{}", "x".repeat((next() % 40) as usize)),
                    3 => format!("日本{i}"),
                    _ => format!("{i:07} bigram{}", next() % 100),
                };
                let len = match next() % 8 {
                    0 => 0,
                    1 => 128,
                    2 => 129 + (next() % 200) as usize,
                    _ => (next() % 128) as usize,
                };
                let ctx = (0..len).map(|k| Comp { d: k as u32 * 3 + (next() % 3) as u32, v: (next() % 2000) as f32 / 7.0 - 100.0 }).collect();
                (term, (next() % 1000) as u32, ctx)
            })
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
    }

    fn temp_files(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp"))
            .collect();
        names.sort();
        names
    }

    /// The streamed file is the image the materializing writer built, byte
    /// for byte, for every size (each count moves the section paddings).
    #[test]
    fn streamed_file_equals_the_materialized_image() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("m.rimodel");
        for (n, seed) in [(0usize, 1u64), (1, 2), (2, 3), (3, 4), (7, 5), (8, 6), (9, 7), (64, 8), (1000, 9), (5000, 10), (5001, 11)] {
            let entries = generated(n, seed);
            let want = Snapshot::materialized_image(2048, 1000 + n as u64, &entries).unwrap();
            Snapshot::write(&path, 2048, 1000 + n as u64, &entries).unwrap();
            let got = fs::read(&path).unwrap();
            assert!(got == want, "{n} entries: streamed file ({} B) differs from the materialized image ({} B)", got.len(), want.len());
            // and it reads back
            let s = Snapshot::open(&path).unwrap();
            assert_eq!(s.len(), n);
            for (i, (t, df, ctx)) in entries.iter().enumerate() {
                assert_eq!(s.entry(i), (t.as_str(), *df, ctx.as_slice()));
                assert_eq!(s.get(t), Some((*df, ctx.as_slice())));
            }
            drop(s);
            assert_eq!(temp_files(tmp.path()), Vec::<String>::new(), "a finished save leaves no temp file");
        }
    }

    /// A source that does not repeat itself is refused (and its temp file
    /// removed) instead of written with offsets that do not match.
    #[test]
    fn a_source_that_changes_between_passes_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("m.rimodel");
        let entries = generated(50, 3);
        let pass = std::cell::Cell::new(0usize);
        let err = Snapshot::write_sorted(&path, 2048, 1, || {
            pass.set(pass.get() + 1);
            // one entry fewer from the third pass on
            let n = if pass.get() >= 3 { 49 } else { 50 };
            entries[..n].iter().map(|(t, df, ctx)| (t.as_str(), *df, ctx.as_slice()))
        })
        .unwrap_err();
        assert!(format!("{err:#}").contains("changed while it was being written"), "{err:#}");
        assert!(!path.exists(), "nothing was renamed into place");
        assert_eq!(temp_files(tmp.path()), Vec::<String>::new(), "the temp file of a failed save is removed");
    }

    /// Saves racing on one path (several servers share a data dir, and each
    /// saves on its own) never mix their bytes: each writes its own temp
    /// file, so the file in place is always one writer's whole snapshot.
    /// With the single shared temp name they wrote into each other's file.
    #[test]
    fn racing_saves_never_mix_their_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("sem").join("m.rimodel");
        let sets: Vec<Vec<(String, u32, Vec<Comp>)>> = (0..6).map(|k| generated(300 + 40 * k, 100 + k as u64)).collect();
        let images: Vec<Vec<u8>> =
            sets.iter().enumerate().map(|(k, e)| Snapshot::materialized_image(2048, k as u64, e).unwrap()).collect();
        let saved = AtomicU64::new(0);
        std::thread::scope(|s| {
            for (k, entries) in sets.iter().enumerate() {
                let (path, saved) = (&path, &saved);
                s.spawn(move || {
                    for _ in 0..25 {
                        // (a rename can lose to another thread's on Windows:
                        // that save fails whole, which is fine)
                        if Snapshot::write(path, 2048, k as u64, entries).is_ok() {
                            saved.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
            // readers map whatever is in place while the writers race
            let (path, images) = (&path, &images);
            s.spawn(move || {
                for _ in 0..200 {
                    if let Ok(bytes) = fs::read(path) {
                        assert!(images.iter().any(|i| *i == bytes), "a reader saw a file that is no writer's snapshot ({} B)", bytes.len());
                    }
                    std::thread::yield_now();
                }
            });
        });
        assert!(saved.load(Ordering::Relaxed) > 0, "no save succeeded");
        let bytes = fs::read(&path).unwrap();
        assert!(images.iter().any(|i| *i == bytes), "the file in place is no writer's snapshot");
        assert_eq!(temp_files(path.parent().unwrap()), Vec::<String>::new(), "no temp file is left behind");
    }

    /// Temp files of writers that died are removed once they are old; a
    /// live writer's file is not touched, whatever its age looks like.
    #[test]
    fn temp_files_of_dead_writers_are_reaped_and_live_ones_kept() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("m.rimodel");
        Snapshot::write(&path, 2048, 1, &generated(10, 1)).unwrap();
        let dir = tmp.path();
        let old = std::time::SystemTime::now() - 2 * STALE_TEMP_AFTER;
        let make = |name: &str, mtime: Option<std::time::SystemTime>| {
            let p = dir.join(name);
            fs::write(&p, b"half a snapshot").unwrap();
            if let Some(t) = mtime {
                fs::File::options().write(true).open(&p).unwrap().set_modified(t).unwrap();
            }
            p
        };
        // left by dead processes: the name every process shared up to
        // 0.1.14, and a per-save name
        let dead_shared = make("m.rimodel.tmp", Some(old));
        let dead_own = make("m.rimodel.tmp.4242.7", Some(old));
        // written to just now: a live writer's
        let live_own = make("m.rimodel.tmp.1.0", None);
        // not temp files of this model at all
        let other_model = make("n.rimodel.tmp", Some(old));
        let lookalike = make("m.rimodel.tmpfile", Some(old));
        let expand = make("m.expand.json.tmp", Some(old));

        reap_stale_temps(&path);
        assert!(!dead_shared.exists(), "the old shared temp name is reaped once stale");
        assert!(!dead_own.exists(), "a dead writer's temp file is reaped once stale");
        assert!(live_own.exists(), "a temp file written to just now is a live writer's");
        assert!(other_model.exists() && lookalike.exists() && expand.exists(), "only this model's temp files are candidates");
        assert!(path.exists());
        // a 0.1.14 server saving right now writes the shared name
        let live_shared = make("m.rimodel.tmp", None);
        reap_stale_temps(&path);
        assert!(live_shared.exists(), "the shared temp name is left alone while it is being written");
        fs::remove_file(&live_shared).unwrap();

        // a writer that has its file open keeps it even if the clock says
        // it is old (Windows: the file cannot be removed while open; other
        // systems rely on the age alone)
        let (mut guard, file) = TempFile::create(&path).unwrap();
        file.set_modified(old).unwrap();
        reap_stale_temps(&path);
        #[cfg(windows)]
        assert!(guard.path.exists(), "an open temp file must survive the reaper");
        drop(file);
        guard.renamed = true; // (leave it to the reaper)
        let orphan = guard.path.clone();
        drop(guard);
        if orphan.exists() {
            fs::File::options().write(true).open(&orphan).unwrap().set_modified(old).unwrap();
            reap_stale_temps(&path);
        }
        assert!(!orphan.exists(), "once its writer is gone the file is reaped");

        // the next save cleans up on its own way out, and names differ per save
        let again = make("m.rimodel.tmp.99.0", Some(old));
        Snapshot::write(&path, 2048, 2, &generated(10, 2)).unwrap();
        assert!(!again.exists(), "a save reaps stale temp files");
        let (a, fa) = TempFile::create(&path).unwrap();
        let (b, fb) = TempFile::create(&path).unwrap();
        assert_ne!(a.path, b.path, "two saves of one process never share a temp file");
        assert!(a.path.to_string_lossy().contains(&format!(".tmp.{}.", std::process::id())));
        drop((fa, fb));
    }
}
