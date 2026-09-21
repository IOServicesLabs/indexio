//! Embedding CAS — org-wide content-addressed embedding cache (SPEC-P2 §2.2).
//!
//! Layout (SPEC-P6 perf revision): one append-only pack per model,
//! `<data_dir>/embcas/<model_id>/pack.bin`, records `[hash 16][dim u32 LE]
//! [f32 x dim LE]`, plus `pack.idx` with `[hash 16][offset u64 LE][dim u32
//! LE]` entries so an open costs one small sequential read instead of a
//! directory walk. Writes are buffered and flushed by `flush()` / `Drop`.
//! The pre-P6 one-file-per-vector layout (`<hh>/<rest>.bin`, bincode) is
//! still READ as a fallback so existing caches keep their hits; new
//! vectors only ever go to the pack. Single writer per model directory
//! (as before); the CAS is a cache — a lost put only costs a re-embed.
//!
//! Seen-only mode (SPEC-P10): for an embedder whose vectors are cheaper to
//! recompute than to read back (the in-process rindex model embeds a chunk
//! in well under a millisecond; its pack held 8 KB per chunk, 2 GB for 250k
//! chunks — 40 % of the data dir), the cache keeps only the 16-byte chunk
//! hashes in `seen.idx`. What the pipeline needs from it is the set of
//! chunks the model has already *observed*, so unchanged chunks of an
//! edited file are not fed to the model twice; the vectors are recomputed.
//! An existing pack for such a model is migrated on first open: its hashes
//! become `seen.idx` and the pack files are removed.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// chunk_hash = blake3(text)[..16]. Stable across runs, repos and processes.
pub fn chunk_hash(text: &[u8]) -> [u8; 16] {
    let h = blake3::hash(text);
    let mut out = [0u8; 16];
    out.copy_from_slice(&h.as_bytes()[..16]);
    out
}

fn hex16(h: &[u8; 16]) -> String {
    let mut s = String::with_capacity(32);
    for b in h {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

const IDX_REC: usize = 16 + 8 + 4;

struct Pack {
    loaded: bool,
    /// Seen-only mode: `seen` holds the hashes, the pack fields are unused.
    seen: HashSet<[u8; 16]>,
    seen_writer: Option<BufWriter<File>>,
    /// hash -> (offset of the f32 payload in pack.bin, dim)
    index: HashMap<[u8; 16], (u64, u32)>,
    /// Current logical end of pack.bin (including buffered writes).
    end: u64,
    writer: Option<BufWriter<File>>,
    idx_writer: Option<BufWriter<File>>,
    reader: Option<File>,
    /// Bytes not yet flushed: reads at offsets >= flushed_end must flush first.
    flushed_end: u64,
}

/// Content-addressed embedding cache (see module docs).
pub struct EmbedCas {
    dir: PathBuf,
    pack: Mutex<Pack>,
    /// Keep chunk hashes only; `get` always misses (see module docs).
    seen_only: bool,
}

const SEEN_FILE: &str = "seen.idx";

impl EmbedCas {
    /// Opens (lazily creates on first `put`) the CAS for `model_id`.
    pub fn open(data_dir: &Path, model_id: &str) -> Self {
        Self::open_mode(data_dir, model_id, false)
    }

    /// Opens the CAS for `model_id` in seen-only mode (SPEC-P10): only chunk
    /// hashes are kept, `get` never hits, `contains` tells whether the chunk
    /// was already observed. Migrates a vector pack found in the directory.
    pub fn open_seen_only(data_dir: &Path, model_id: &str) -> Self {
        Self::open_mode(data_dir, model_id, true)
    }

    /// The right mode for `embedder`: seen-only when recomputing is cheap.
    pub fn open_for(data_dir: &Path, embedder: &dyn crate::embed::Embedder) -> Self {
        Self::open_mode(data_dir, embedder.model_id(), embedder.recompute_is_cheap())
    }

    fn open_mode(data_dir: &Path, model_id: &str, seen_only: bool) -> Self {
        EmbedCas {
            dir: data_dir.join("embcas").join(model_id),
            pack: Mutex::new(Pack {
                loaded: false,
                seen: HashSet::new(),
                seen_writer: None,
                index: HashMap::new(),
                end: 0,
                writer: None,
                idx_writer: None,
                reader: None,
                flushed_end: 0,
            }),
            seen_only,
        }
    }

    fn seen_path(&self) -> PathBuf {
        self.dir.join(SEEN_FILE)
    }

    /// Seen-only load: `seen.idx` hashes, plus a one-time migration of a
    /// vector pack's hashes (the pack is then removed).
    fn ensure_loaded_seen(&self, p: &mut Pack) {
        if p.loaded {
            return;
        }
        p.loaded = true;
        if let Ok(bytes) = fs::read(self.seen_path()) {
            for rec in bytes.chunks_exact(16) {
                let mut h = [0u8; 16];
                h.copy_from_slice(rec);
                p.seen.insert(h);
            }
        }
        if let Ok(bytes) = fs::read(self.idx_path()) {
            let mut fresh: Vec<u8> = Vec::new();
            for rec in bytes.chunks_exact(IDX_REC) {
                let mut h = [0u8; 16];
                h.copy_from_slice(&rec[..16]);
                if p.seen.insert(h) {
                    fresh.extend_from_slice(&h);
                }
            }
            let mut ok = true;
            if !fresh.is_empty() {
                ok = OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(self.seen_path())
                    .and_then(|mut f| f.write_all(&fresh))
                    .is_ok();
            }
            if ok {
                let _ = fs::remove_file(self.idx_path());
                let _ = fs::remove_file(self.pack_path());
                tracing::info!(
                    dir = %self.dir.display(),
                    hashes = p.seen.len(),
                    "embcas: migrated the vector pack to seen-only (hashes kept, vectors dropped)"
                );
            }
        }
    }

    /// Whether `chunk_hash` is in the cache (either mode): the model has
    /// already observed this chunk text.
    pub fn contains(&self, chunk_hash: &[u8; 16]) -> bool {
        let mut p = self.pack.lock().expect("embcas lock poisoned");
        if self.seen_only {
            self.ensure_loaded_seen(&mut p);
            return p.seen.contains(chunk_hash);
        }
        self.ensure_loaded(&mut p);
        if p.index.contains_key(chunk_hash) {
            return true;
        }
        drop(p);
        self.legacy_path(chunk_hash).is_file()
    }

    fn pack_path(&self) -> PathBuf {
        self.dir.join("pack.bin")
    }

    fn idx_path(&self) -> PathBuf {
        self.dir.join("pack.idx")
    }

    /// Legacy per-file path (read fallback only).
    fn legacy_path(&self, chunk_hash: &[u8; 16]) -> PathBuf {
        let hex = hex16(chunk_hash);
        self.dir.join(&hex[..2]).join(format!("{}.bin", &hex[2..]))
    }

    /// Load pack.idx once. A truncated trailing record is ignored; entries
    /// pointing past the pack's real length are dropped (crash during an
    /// unflushed write).
    fn ensure_loaded(&self, p: &mut Pack) {
        if p.loaded {
            return;
        }
        p.loaded = true;
        let pack_len = fs::metadata(self.pack_path()).map(|m| m.len()).unwrap_or(0);
        p.end = pack_len;
        p.flushed_end = pack_len;
        if let Ok(bytes) = fs::read(self.idx_path()) {
            for rec in bytes.chunks_exact(IDX_REC) {
                let mut h = [0u8; 16];
                h.copy_from_slice(&rec[..16]);
                let off = u64::from_le_bytes(rec[16..24].try_into().unwrap());
                let dim = u32::from_le_bytes(rec[24..28].try_into().unwrap());
                if off + u64::from(dim) * 4 <= pack_len {
                    p.index.insert(h, (off, dim));
                }
            }
        }
    }

    fn flush_locked(p: &mut Pack) {
        if let Some(w) = p.seen_writer.as_mut() {
            let _ = w.flush();
        }
        if let Some(w) = p.writer.as_mut() {
            let _ = w.flush();
        }
        if let Some(w) = p.idx_writer.as_mut() {
            let _ = w.flush();
        }
        p.flushed_end = p.end;
    }

    pub fn get(&self, chunk_hash: &[u8; 16]) -> Option<Vec<f32>> {
        if self.seen_only {
            return None;
        }
        let mut p = self.pack.lock().expect("embcas lock poisoned");
        self.ensure_loaded(&mut p);
        if let Some(&(off, dim)) = p.index.get(chunk_hash) {
            if off + u64::from(dim) * 4 > p.flushed_end {
                Self::flush_locked(&mut p);
            }
            if p.reader.is_none() {
                p.reader = File::open(self.pack_path()).ok();
            }
            let reader = p.reader.as_mut()?;
            let mut buf = vec![0u8; dim as usize * 4];
            reader.seek(SeekFrom::Start(off)).ok()?;
            reader.read_exact(&mut buf).ok()?;
            return Some(
                buf.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            );
        }
        drop(p);
        // Legacy layout fallback.
        let bytes = fs::read(self.legacy_path(chunk_hash)).ok()?;
        bincode::deserialize(&bytes).ok()
    }

    /// Best-effort store (signature per SPEC-P2 returns ()). Appends to the
    /// pack; failures are logged, not fatal.
    pub fn put(&self, chunk_hash: &[u8; 16], vec: &[f32]) {
        let mut p = self.pack.lock().expect("embcas lock poisoned");
        if self.seen_only {
            self.ensure_loaded_seen(&mut p);
            if !p.seen.insert(*chunk_hash) {
                return;
            }
            if p.seen_writer.is_none() {
                if let Err(e) = fs::create_dir_all(&self.dir) {
                    tracing::warn!(error = %e, "embcas: create_dir_all failed");
                    return;
                }
                match OpenOptions::new().append(true).create(true).open(self.seen_path()) {
                    Ok(f) => p.seen_writer = Some(BufWriter::with_capacity(1 << 16, f)),
                    Err(e) => {
                        tracing::warn!(error = %e, "embcas: opening seen.idx failed");
                        return;
                    }
                }
            }
            if p.seen_writer.as_mut().is_none_or(|w| w.write_all(chunk_hash).is_err()) {
                tracing::warn!("embcas: put failed (seen.idx write)");
            }
            return;
        }
        self.ensure_loaded(&mut p);
        if p.index.contains_key(chunk_hash) {
            return;
        }
        if p.writer.is_none() {
            if let Err(e) = fs::create_dir_all(&self.dir) {
                tracing::warn!(error = %e, "embcas: create_dir_all failed");
                return;
            }
            let open = |path: PathBuf| OpenOptions::new().append(true).create(true).open(path);
            match (open(self.pack_path()), open(self.idx_path())) {
                (Ok(a), Ok(b)) => {
                    p.writer = Some(BufWriter::with_capacity(1 << 20, a));
                    p.idx_writer = Some(BufWriter::with_capacity(1 << 16, b));
                }
                (Err(e), _) | (_, Err(e)) => {
                    tracing::warn!(error = %e, "embcas: opening pack failed");
                    return;
                }
            }
        }
        let dim = vec.len() as u32;
        let payload_off = p.end + 16 + 4;
        let mut rec = Vec::with_capacity(20 + vec.len() * 4);
        rec.extend_from_slice(chunk_hash);
        rec.extend_from_slice(&dim.to_le_bytes());
        for x in vec {
            rec.extend_from_slice(&x.to_le_bytes());
        }
        let mut irec = [0u8; IDX_REC];
        irec[..16].copy_from_slice(chunk_hash);
        irec[16..24].copy_from_slice(&payload_off.to_le_bytes());
        irec[24..28].copy_from_slice(&dim.to_le_bytes());
        let ok = p
            .writer
            .as_mut()
            .map(|w| w.write_all(&rec).is_ok())
            .unwrap_or(false)
            && p
                .idx_writer
                .as_mut()
                .map(|w| w.write_all(&irec).is_ok())
                .unwrap_or(false);
        if ok {
            p.end += rec.len() as u64;
            p.index.insert(*chunk_hash, (payload_off, dim));
        } else {
            tracing::warn!("embcas: put failed (pack write)");
        }
    }

    /// Flush buffered pack writes (the pipeline calls this after each repo).
    pub fn flush(&self) {
        let mut p = self.pack.lock().expect("embcas lock poisoned");
        Self::flush_locked(&mut p);
    }

    /// (entries, total payload bytes): pack entries + legacy files, or the
    /// seen hashes (16 bytes each) in seen-only mode.
    pub fn stats(&self) -> (u64, u64) {
        let (mut entries, mut bytes) = {
            let mut p = self.pack.lock().expect("embcas lock poisoned");
            if self.seen_only {
                self.ensure_loaded_seen(&mut p);
                (p.seen.len() as u64, p.seen.len() as u64 * 16)
            } else {
                self.ensure_loaded(&mut p);
                let n = p.index.len() as u64;
                let b: u64 = p.index.values().map(|&(_, d)| u64::from(d) * 4).sum();
                (n, b)
            }
        };
        fn walk(dir: &Path, entries: &mut u64, bytes: &mut u64) {
            let Ok(rd) = fs::read_dir(dir) else { return };
            for entry in rd.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    walk(&p, entries, bytes);
                } else if p.extension().and_then(|e| e.to_str()) == Some("bin")
                    && p.file_name().and_then(|f| f.to_str()) != Some("pack.bin")
                {
                    *entries += 1;
                    *bytes += entry.metadata().map(|m| m.len()).unwrap_or(0);
                }
            }
        }
        walk(&self.dir, &mut entries, &mut bytes);
        (entries, bytes)
    }
}

impl Drop for EmbedCas {
    fn drop(&mut self) {
        if let Ok(mut p) = self.pack.lock() {
            Self::flush_locked(&mut p);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_hash_stability() {
        let a = chunk_hash(b"fn main() {}");
        let b = chunk_hash(b"fn main() {}");
        assert_eq!(a, b, "chunk_hash must be stable");
        assert_eq!(a.len(), 16);
        let c = chunk_hash(b"fn main() { }");
        assert_ne!(a, c, "different text -> different hash");
    }

    #[test]
    fn cas_put_get_stats() {
        let tmp = tempfile::tempdir().unwrap();
        let cas = EmbedCas::open(tmp.path(), "hash-v1");

        let h1 = chunk_hash(b"chunk one");
        let h2 = chunk_hash(b"chunk two");
        let h3 = chunk_hash(b"never stored");

        assert_eq!(cas.get(&h1), None, "miss before put");

        let v1 = vec![0.5f32, -0.5, 0.71];
        let v2 = vec![1.0f32; 512];
        cas.put(&h1, &v1);
        cas.put(&h2, &v2);

        // Reads see unflushed writes (flush-on-demand).
        assert_eq!(cas.get(&h1), Some(v1.clone()));
        assert_eq!(cas.get(&h2), Some(v2.clone()));
        assert_eq!(cas.get(&h3), None);

        // Idempotent put (same key) does not duplicate.
        cas.put(&h1, &v1);

        let (entries, bytes) = cas.stats();
        assert_eq!(entries, 2);
        assert_eq!(bytes, (v1.len() * 4 + v2.len() * 4) as u64);

        // Persistence: a fresh handle reads the pack + idx back.
        drop(cas);
        let cas = EmbedCas::open(tmp.path(), "hash-v1");
        assert_eq!(cas.get(&h1), Some(v1.clone()));
        assert_eq!(cas.get(&h2), Some(v2.clone()));
        assert_eq!(cas.stats().0, 2);
        // Appending after reopen keeps earlier entries.
        cas.put(&h3, &[9.0]);
        cas.flush();
        let cas2 = EmbedCas::open(tmp.path(), "hash-v1");
        assert_eq!(cas2.get(&h3), Some(vec![9.0]));
        assert_eq!(cas2.get(&h1), Some(v1));
        assert_eq!(cas2.stats().0, 3);

        // Separate model id => separate namespace.
        let other = EmbedCas::open(tmp.path(), "http:other-model");
        assert_eq!(other.get(&h1), None);
        assert_eq!(other.stats(), (0, 0));
    }

    #[test]
    fn seen_only_keeps_hashes_and_migrates_a_pack() {
        let tmp = tempfile::tempdir().unwrap();
        let h1 = chunk_hash(b"one");
        let h2 = chunk_hash(b"two");
        {
            let cas = EmbedCas::open(tmp.path(), "m");
            cas.put(&h1, &[1.0, 0.0]);
            cas.flush();
        }
        assert!(tmp.path().join("embcas/m/pack.bin").is_file());
        {
            let cas = EmbedCas::open_seen_only(tmp.path(), "m");
            assert!(cas.contains(&h1), "pack hash migrated");
            assert!(cas.get(&h1).is_none(), "seen-only never returns vectors");
            assert!(!cas.contains(&h2));
            cas.put(&h2, &[0.0, 1.0]);
            assert!(cas.contains(&h2));
            assert_eq!(cas.stats(), (2, 32));
        }
        assert!(!tmp.path().join("embcas/m/pack.bin").exists(), "pack removed");
        assert!(!tmp.path().join("embcas/m/pack.idx").exists());
        assert_eq!(fs::metadata(tmp.path().join("embcas/m/seen.idx")).unwrap().len(), 32);
        let cas = EmbedCas::open_seen_only(tmp.path(), "m");
        assert!(cas.contains(&h1) && cas.contains(&h2), "persisted");
    }

    #[test]
    fn legacy_per_file_layout_is_still_readable() {
        let tmp = tempfile::tempdir().unwrap();
        let cas = EmbedCas::open(tmp.path(), "rindex-v2");
        let h = chunk_hash(b"old chunk");
        let v = vec![0.25f32; 8];
        let path = cas.legacy_path(&h);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bincode::serialize(&v).unwrap()).unwrap();
        assert_eq!(cas.get(&h), Some(v.clone()));
        let (entries, bytes) = cas.stats();
        assert_eq!(entries, 1);
        assert_eq!(bytes, bincode::serialize(&v).unwrap().len() as u64);
        // A pack put of the same hash coexists; pack wins on read.
        cas.put(&h, &[1.0, 2.0]);
        assert_eq!(cas.get(&h), Some(vec![1.0, 2.0]));
        assert_eq!(cas.stats().0, 2);
    }

    #[test]
    fn truncated_idx_and_pack_are_tolerated() {
        let tmp = tempfile::tempdir().unwrap();
        let cas = EmbedCas::open(tmp.path(), "m");
        let h1 = chunk_hash(b"a");
        let h2 = chunk_hash(b"b");
        cas.put(&h1, &[1.0, 1.0]);
        cas.put(&h2, &[2.0, 2.0]);
        drop(cas);
        // Chop the pack so the second record is incomplete; append junk to idx.
        let pack = tmp.path().join("embcas/m/pack.bin");
        let bytes = fs::read(&pack).unwrap();
        fs::write(&pack, &bytes[..bytes.len() - 3]).unwrap();
        let idx = tmp.path().join("embcas/m/pack.idx");
        let mut ib = fs::read(&idx).unwrap();
        ib.extend_from_slice(&[7u8; 5]);
        fs::write(&idx, ib).unwrap();
        let cas = EmbedCas::open(tmp.path(), "m");
        assert_eq!(cas.get(&h1), Some(vec![1.0, 1.0]));
        assert_eq!(cas.get(&h2), None, "incomplete record is a miss");
        // Re-putting the missing one appends cleanly.
        cas.put(&h2, &[2.0, 2.0]);
        assert_eq!(cas.get(&h2), Some(vec![2.0, 2.0]));
    }
}
