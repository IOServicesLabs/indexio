//! Flat, memory-mappable rindex model file (SPEC-P10).
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

use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::{anyhow, ensure, Context};
use memmap2::Mmap;

const MAGIC: &[u8; 8] = b"RIMDL003";
const HEADER_LEN: usize = 64;
const ENTRY_LEN: usize = 16;

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
        .map_or(false, |_| &magic == MAGIC)
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
        debug_assert!(entries.windows(2).all(|w| w[0].0 < w[1].0), "entries must be sorted by term");
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
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let tmp = path.with_extension("rimodel.tmp");
        {
            let mut f = fs::File::create(&tmp).with_context(|| format!("writing {}", tmp.display()))?;
            f.write_all(&out)?;
            f.sync_all().ok();
        }
        // The previous file may be mapped by other servers: on Windows a
        // rename over it fails, so park it first (reaped by `reap_parked`).
        if path.exists() {
            let parked = path.with_extension(format!(
                "rimodel.stale{}",
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
            ));
            let _ = fs::rename(path, &parked);
        }
        fs::rename(&tmp, path).with_context(|| format!("renaming to {}", path.display()))?;
        reap_parked(path);
        Ok(())
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
}
