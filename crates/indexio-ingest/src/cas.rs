//! Global content-addressed store for `ExtractedArtifact`s.
//!
//! Layout (SPEC indexio-ingest, revised SPEC-P10): `<dir>/v2/<hex[0..2]>/
//! <hex[2..]>`, one file per blob: a 4-byte magic then the zstd frame of the
//! bincode-serialized `ExtractedArtifact`. The CAS dedups *extraction work*
//! (grams + tree-sitter) across repos, branches and forks — file content
//! itself is not stored here (it lives in shards).
//!
//! The pre-P10 layout (`<dir>/<hh>/<rest>`, raw bincode with per-occurrence
//! gram positions) averaged ~100 KB per blob — 7× the compressed content
//! it described. Those entries cannot be parsed by the position-free
//! artifact type, so `open` sweeps the legacy fan-out directories once; the
//! CAS is a cache, and a swept entry costs one re-extraction of that blob
//! the next time it is seen.
//!
//! Concurrency: `get`/`put` take `&self` and are safe to call from multiple
//! threads (rayon extraction pool). `put` writes a uniquely-named temp file
//! in the target fan-out directory and atomically renames it over the
//! destination (last writer wins — entries are content-keyed, so concurrent
//! writers store identical values).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use indexio_types::{BlobId, ExtractedArtifact};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Entry magic: `CAS2` + zstd frame of the bincode artifact.
const MAGIC: &[u8; 4] = b"CAS2";
/// zstd level for entries: fast, and bincode's length prefixes fold away.
const ZSTD_LEVEL: i32 = 3;
/// Versioned subdirectory holding the fan-out.
const LAYOUT: &str = "v2";

pub struct Cas {
    dir: PathBuf,
}

impl Cas {
    /// Open (creating if needed) the CAS rooted at `dir`; sweeps a pre-P10
    /// layout found there.
    pub fn open(dir: &Path) -> io::Result<Self> {
        fs::create_dir_all(dir.join(LAYOUT))?;
        Self::sweep_legacy(dir);
        Ok(Cas {
            dir: dir.to_path_buf(),
        })
    }

    /// Remove the legacy `<dir>/<hh>/` fan-out (two lowercase hex chars) if
    /// present. Best effort, logged once.
    fn sweep_legacy(dir: &Path) {
        let Ok(rd) = fs::read_dir(dir) else { return };
        let mut removed = 0usize;
        for d in rd.flatten() {
            let name = d.file_name();
            let name = name.to_string_lossy();
            let legacy = name.len() == 2 && name.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
            if legacy && d.path().is_dir() && fs::remove_dir_all(d.path()).is_ok() {
                removed += 1;
            }
        }
        if removed > 0 {
            tracing::info!(dir = %dir.display(), dirs = removed, "swept the pre-P10 extraction cache");
        }
    }

    fn entry_path(&self, id: &BlobId) -> PathBuf {
        let hex = id.hex();
        self.dir.join(LAYOUT).join(&hex[..2]).join(&hex[2..])
    }

    /// Look up an artifact. A missing *or corrupt* entry is a miss (the
    /// caller recomputes and `put` overwrites the corrupt entry).
    pub fn get(&self, id: &BlobId) -> Option<ExtractedArtifact> {
        let path = self.entry_path(id);
        let bytes = fs::read(&path).ok()?;
        let frame = bytes.strip_prefix(MAGIC)?;
        let raw = zstd::stream::decode_all(frame).ok()?;
        bincode::deserialize(&raw).ok()
    }

    /// Store an artifact (tmp file + atomic rename). Idempotent: putting the
    /// same id twice just overwrites.
    pub fn put(&self, id: &BlobId, art: &ExtractedArtifact) -> io::Result<()> {
        let path = self.entry_path(id);
        let sub = path.parent().expect("entry path has a parent");
        fs::create_dir_all(sub)?;
        let raw = bincode::serialize(art).map_err(io::Error::other)?;
        let mut bytes = Vec::with_capacity(4 + raw.len() / 4);
        bytes.extend_from_slice(MAGIC);
        zstd::stream::copy_encode(&raw[..], &mut bytes, ZSTD_LEVEL)?;
        let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp = sub.join(format!(".put-{}-{n}", std::process::id()));
        fs::write(&tmp, &bytes)?;
        // POSIX rename atomically replaces an existing destination.
        // (Windows can't; remove first — best effort.)
        #[cfg(windows)]
        {
            let _ = fs::remove_file(&path);
        }
        fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// (entry count, total bytes) — used by `indexio cas-stats`.
    pub fn stats(&self) -> (u64, u64) {
        let mut entries = 0u64;
        let mut bytes = 0u64;
        if let Ok(dirs) = fs::read_dir(self.dir.join(LAYOUT)) {
            for d in dirs.flatten() {
                let dp = d.path();
                if !dp.is_dir() {
                    continue;
                }
                if let Ok(files) = fs::read_dir(&dp) {
                    for f in files.flatten() {
                        let fp = f.path();
                        // Skip in-flight temp files.
                        let fname = f.file_name();
                        let fname = fname.to_string_lossy();
                        if fp.is_file() && !fname.starts_with(".put-") {
                            entries += 1;
                            bytes += f.metadata().map(|m| m.len()).unwrap_or(0);
                        }
                    }
                }
            }
        }
        (entries, bytes)
    }
}
