//! Embedding CAS — org-wide content-addressed embedding cache.
//!
//! Layout: one append-only pack per model,
//! `<data_dir>/embcas/<model_id>/pack.bin`, records `[hash 16][dim u32 LE]
//! [f32 x dim LE]`, plus `pack.idx` with `[hash 16][offset u64 LE][dim u32
//! LE]` entries so an open costs one small sequential read instead of a
//! directory walk. Writes are buffered and flushed by `flush()` / `Drop`.
//! The older one-file-per-vector layout (`<hh>/<rest>.bin`, bincode) is
//! still READ as a fallback so existing caches keep their hits; new
//! vectors only ever go to the pack. Single writer per model directory
//! (as before); the CAS is a cache — a lost put only costs a re-embed.
//!
//! Seen-only mode: for an embedder whose vectors are cheaper to
//! recompute than to read back (the in-process rindex model embeds a chunk
//! in well under a millisecond; its pack held 8 KB per chunk, 2 GB for 250k
//! chunks — 40 % of the data dir), the cache keeps only the 16-byte chunk
//! hashes in `seen.idx`. What the pipeline needs from it is the set of
//! chunks the model has already *observed*, so unchanged chunks of an
//! edited file are not fed to the model twice; the vectors are recomputed.
//! An existing pack for such a model is migrated on first open: its hashes
//! become `seen.idx` and the pack files are removed.
//!
//! `seen.idx` is never loaded: an embed asks about the chunks of the files
//! it is embedding, a few hashes against a file of every chunk the model
//! ever saw (1.37 M hashes on a 57-repo data dir: 21 MB read into a 36-55 MB
//! set, on each embed with a changed chunk). The hashes asked about are
//! looked up in one sequential pass over the file ([`EmbedCas::contains_many`])
//! and only their answers are kept.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
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
    /// Seen-only mode (the pack fields are unused): the hashes this handle
    /// has looked up in `seen.idx` or appended to it, and whether the file
    /// has them. The file's own hashes are not held.
    seen: HashMap<[u8; 16], bool>,
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

    /// Opens the CAS for `model_id` in seen-only mode: only chunk
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
                seen: HashMap::new(),
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

    /// Seen-only mode, once per handle: migrate a vector pack found in the
    /// directory. Its hashes join `seen.idx` and the pack is removed. This
    /// is the one place the whole of `seen.idx` is read into memory, and
    /// only when there is a pack to migrate.
    fn migrate_pack(&self, p: &mut Pack) {
        if p.loaded {
            return;
        }
        p.loaded = true;
        let Ok(bytes) = fs::read(self.idx_path()) else { return };
        let mut have: HashSet<[u8; 16]> = HashSet::new();
        if let Ok(seen) = fs::read(self.seen_path()) {
            for rec in seen.chunks_exact(16) {
                let mut h = [0u8; 16];
                h.copy_from_slice(rec);
                have.insert(h);
            }
        }
        let mut fresh: Vec<u8> = Vec::new();
        for rec in bytes.chunks_exact(IDX_REC) {
            let mut h = [0u8; 16];
            h.copy_from_slice(&rec[..16]);
            if have.insert(h) {
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
                hashes = have.len(),
                "embcas: migrated the vector pack to seen-only (hashes kept, vectors dropped)"
            );
        }
    }

    /// Seen-only mode: settle, for every hash of `hashes` this handle has
    /// no answer for yet, whether `seen.idx` has it. One sequential pass
    /// over the file through a fixed buffer; nothing of the file is kept
    /// but the answers. A record is tested against the 16-bit prefixes of
    /// the wanted hashes first, so a lookup of a few chunks dismisses
    /// nearly every record on one bit.
    fn look_up(&self, p: &mut Pack, hashes: &[[u8; 16]]) {
        self.migrate_pack(p);
        let want: HashSet<[u8; 16]> = hashes.iter().filter(|h| !p.seen.contains_key(*h)).copied().collect();
        if want.is_empty() {
            return;
        }
        let prefix_of = |h: &[u8; 16]| usize::from(u16::from_le_bytes([h[0], h[1]]));
        let mut prefixes = vec![0u64; (1 << 16) / 64];
        for h in &want {
            prefixes[prefix_of(h) / 64] |= 1 << (prefix_of(h) % 64);
        }
        if let Ok(f) = File::open(self.seen_path()) {
            let mut file = BufReader::with_capacity(1 << 18, f);
            let mut rec = [0u8; 16];
            let mut found = 0usize;
            // a trailing partial record (another process mid-append) ends the pass
            while found < want.len() && file.read_exact(&mut rec).is_ok() {
                let i = prefix_of(&rec);
                if prefixes[i / 64] & (1 << (i % 64)) != 0 && want.contains(&rec) && p.seen.insert(rec, true).is_none() {
                    found += 1;
                }
            }
        }
        for h in want {
            p.seen.entry(h).or_insert(false);
        }
    }

    /// Whether `chunk_hash` is in the cache (either mode): the model has
    /// already observed this chunk text. In seen-only mode a hash not asked
    /// about before costs a pass over `seen.idx`: ask for many at once with
    /// [`contains_many`](Self::contains_many).
    pub fn contains(&self, chunk_hash: &[u8; 16]) -> bool {
        let mut p = self.pack.lock().expect("embcas lock poisoned");
        if self.seen_only {
            self.look_up(&mut p, std::slice::from_ref(chunk_hash));
            return p.seen.get(chunk_hash).copied().unwrap_or(false);
        }
        self.ensure_loaded(&mut p);
        if p.index.contains_key(chunk_hash) {
            return true;
        }
        drop(p);
        self.legacy_path(chunk_hash).is_file()
    }

    /// [`contains`](Self::contains) for each of `hashes`, in order. In
    /// seen-only mode all of them are answered by one pass over `seen.idx`.
    pub fn contains_many(&self, hashes: &[[u8; 16]]) -> Vec<bool> {
        if !self.seen_only {
            return hashes.iter().map(|h| self.contains(h)).collect();
        }
        let mut p = self.pack.lock().expect("embcas lock poisoned");
        self.look_up(&mut p, hashes);
        hashes.iter().map(|h| p.seen.get(h).copied().unwrap_or(false)).collect()
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

    /// Best-effort store (returns ()). Appends to the
    /// pack; failures are logged, not fatal.
    pub fn put(&self, chunk_hash: &[u8; 16], vec: &[f32]) {
        let mut p = self.pack.lock().expect("embcas lock poisoned");
        if self.seen_only {
            self.look_up(&mut p, std::slice::from_ref(chunk_hash));
            if p.seen.insert(*chunk_hash, true) == Some(true) {
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
                self.migrate_pack(&mut p);
                Self::flush_locked(&mut p);
                let n = distinct_hashes(&self.seen_path());
                (n, n * 16)
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

/// Distinct 16-byte records of a `seen.idx` (two processes can append the
/// same hash): the file sorted in one buffer of its own size.
fn distinct_hashes(path: &Path) -> u64 {
    let Ok(f) = File::open(path) else { return 0 };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let mut all: Vec<[u8; 16]> = Vec::with_capacity((len / 16) as usize);
    let mut file = BufReader::with_capacity(1 << 18, f);
    let mut rec = [0u8; 16];
    while file.read_exact(&mut rec).is_ok() {
        all.push(rec);
    }
    all.sort_unstable();
    all.dedup();
    all.len() as u64
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

    /// Lookups against a `seen.idx` written by hand: first, last and middle
    /// records, a duplicate, absent hashes, a torn trailing record.
    #[test]
    fn seen_only_answers_from_the_file_without_loading_it() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("embcas/m");
        fs::create_dir_all(&dir).unwrap();
        let hash = |i: u32| chunk_hash(format!("chunk {i}").as_bytes());
        let n = 50_000u32;
        let mut bytes: Vec<u8> = (0..n).flat_map(hash).collect();
        bytes.extend_from_slice(&hash(7)); // a duplicate, as two writers leave
        bytes.extend_from_slice(&[0xee; 9]); // a record another process is still writing
        fs::write(dir.join(SEEN_FILE), &bytes).unwrap();

        let cas = EmbedCas::open_seen_only(tmp.path(), "m");
        let ask: Vec<[u8; 16]> = vec![hash(0), hash(n), hash(n - 1), hash(7), hash(n + 5), hash(31_337), hash(7)];
        assert_eq!(cas.contains_many(&ask), [true, false, true, true, false, true, true]);
        assert!(cas.contains(&hash(12)) && !cas.contains(&hash(n + 99)));
        assert!(cas.get(&hash(0)).is_none(), "seen-only never returns vectors");
        // what the handle keeps is what it was asked, not the file
        assert_eq!(cas.pack.lock().unwrap().seen.len(), 8);
        assert_eq!(cas.stats(), (u64::from(n), u64::from(n) * 16), "distinct whole records");
        // every hash at once, and a second question costs no second pass
        let probe: Vec<[u8; 16]> = (0..n + 10).map(hash).collect();
        let got = cas.contains_many(&probe);
        assert!(got.iter().enumerate().all(|(i, &g)| g == (i < n as usize)));
        fs::remove_file(dir.join(SEEN_FILE)).unwrap();
        assert_eq!(cas.contains_many(&probe), got, "answers are kept by the handle");

        // no file at all: nothing is seen
        let empty = EmbedCas::open_seen_only(tmp.path(), "other");
        assert_eq!(empty.contains_many(&[hash(1), hash(2)]), [false, false]);
        assert_eq!(empty.stats(), (0, 0));
    }

    /// A put appends exactly the hashes the file does not have, whether or
    /// not the handle was asked about them before.
    #[test]
    fn seen_only_put_appends_only_new_hashes() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("embcas/m");
        fs::create_dir_all(&dir).unwrap();
        let hash = |i: u32| chunk_hash(format!("chunk {i}").as_bytes());
        let n = 3_000u32;
        let bytes: Vec<u8> = (0..n).flat_map(hash).collect();
        fs::write(dir.join(SEEN_FILE), &bytes).unwrap();

        let cas = EmbedCas::open_seen_only(tmp.path(), "m");
        assert_eq!(cas.contains_many(&[hash(0), hash(n)]), [true, false]);
        cas.put(&hash(0), &[1.0]); // known present
        cas.put(&hash(n), &[1.0]); // known absent
        cas.put(&hash(n + 1), &[1.0]); // never asked about, absent
        cas.put(&hash(n + 1), &[1.0]);
        cas.put(&hash(2_000), &[1.0]); // never asked about, present
        assert!(cas.contains(&hash(n)) && cas.contains(&hash(n + 1)), "unflushed puts are seen");
        assert_eq!(cas.stats().0, u64::from(n) + 2, "stats counts the appended hashes");
        drop(cas);
        let after = fs::read(dir.join(SEEN_FILE)).unwrap();
        assert_eq!(after.len(), bytes.len() + 32);
        assert_eq!(&after[..bytes.len()], &bytes[..], "the file is only appended to");
        assert_eq!(&after[bytes.len()..bytes.len() + 16], &hash(n));
        assert_eq!(&after[bytes.len() + 16..], &hash(n + 1));

        // another handle (another server) reads them back
        let cas = EmbedCas::open_seen_only(tmp.path(), "m");
        assert_eq!(cas.contains_many(&[hash(n + 1), hash(n), hash(n + 2)]), [true, true, false]);
        assert_eq!(cas.stats(), (u64::from(n) + 2, (u64::from(n) + 2) * 16));
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
