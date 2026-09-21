//! Binary-quantized flat vector index sidecar (SPEC-P2 §2.3).
//!
//! On-disk (v2, written since SPEC-P6): `<dir>/<model_id>.civec`
//! (pipeline passes `<data_dir>/vec`):
//! ```text
//! magic "CIVEC002" (8B) | u32 dim | u32 n_rows | u32 n_deleted
//! | u64 meta_len | bincode Vec<VecRowMeta>
//! | zero pad to an 8-byte boundary
//! | n_rows x dim f32 (little-endian) vectors
//! | n_rows x ceil(dim/8) binary sign-bit codes
//! | u64 tomb_len | bincode sorted Vec<u32> tombstones
//! ```
//! v1 files (`CIVEC001`, unpadded) are still opened, into owned memory.
//!
//! Segments written since SPEC-P10 are v3 (`CIVEC003`): the same header and
//! meta block, then `n_rows` f32 per-row scales, `n_rows x dim` **int8**
//! rows (`x = q * scale`, scale = max|x| / 127 per row), the binary codes
//! and the tombstones. A row is 2 KB instead of 8 at dim 2048, the rescore
//! reads a quarter of the bytes, and the score is `scale * dot(q, row_i8)`.
//! The quantisation round-trips exactly through a compaction (the maximum
//! component maps to 127, every other one to the same integer again). v2
//! files stay readable; `create_with_options` (the HNSW full build) still
//! writes v2 because the graph walks f32 rows.
//!
//! Opening a v2 file memory-maps it (SPEC-P6 perf): the vector block is
//! never copied, so opening a multi-GB index costs the meta parse only and
//! a query touches just the binary codes plus the rescored candidates. The
//! mapping is the crate's single `unsafe` site (`mapped`), under the same
//! policy as indexio-index shards: the file is immutable after its atomic rename.
//!
//! Documented deviations from SPEC-P2:
//! - Tombstones are a bincode sorted `Vec<u32>` rather than a
//!   `RoaringBitmap` to avoid a new dependency; the `.civec` file is a
//!   rebuildable sidecar, so the encoding is internal to this crate.
//!
//! Query: hamming-distance prescan on the binary codes keeps the top
//! `max(k*8, 256)` candidates (capped at live rows), rescores with exact
//! f32 dot, returns top k. Tombstoned rows are skipped.

use std::collections::BTreeSet;
use std::fs;
use std::sync::OnceLock;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, ensure, Context};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::embed::dot;
use crate::hnsw::{Hnsw, HnswOptions};

const MAGIC_V1: &[u8; 8] = b"CIVEC001";
const MAGIC: &[u8; 8] = b"CIVEC002";
const MAGIC_V3: &[u8; 8] = b"CIVEC003";

/// Per-row symmetric int8 quantisation: `(scale, q)` with `x ≈ q * scale`.
fn quantize_row(v: &[f32]) -> (f32, Vec<i8>) {
    let amax = v.iter().fold(0f32, |m, &x| m.max(x.abs()));
    if amax == 0.0 {
        return (0.0, vec![0; v.len()]);
    }
    let scale = amax / 127.0;
    let inv = 127.0 / amax;
    let q = v.iter().map(|&x| (x * inv).round().clamp(-127.0, 127.0) as i8).collect();
    (scale, q)
}

/// `scale * dot(q, row)` for an int8 row.
fn dot_i8(q: &[f32], row: &[i8], scale: f32) -> f32 {
    debug_assert_eq!(q.len(), row.len());
    let mut acc = 0f32;
    for (a, &b) in q.iter().zip(row) {
        acc += a * b as f32;
    }
    acc * scale
}

/// Environment override for the HNSW build threshold (rows).
pub const HNSW_THRESHOLD_ENV: &str = "INDEXIO_HNSW_THRESHOLD";

/// Options for [`VecIndex::create_with_options`] (SPEC-P3 §1, additive).
#[derive(Clone, Debug)]
pub struct IndexOptions {
    /// Build + save an HNSW `.cihnsw` sidecar when the row count is at or
    /// above this threshold. Default: `HnswOptions::default().threshold`
    /// (1,000,000 rows), overridable with `INDEXIO_HNSW_THRESHOLD`. The flat
    /// binary-code prescan answers a 200k-row index in tens of ms, while
    /// the (single-threaded) graph build costs minutes at that size.
    pub hnsw_threshold: usize,
}

impl Default for IndexOptions {
    fn default() -> Self {
        let from_env = std::env::var(HNSW_THRESHOLD_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok());
        IndexOptions {
            hnsw_threshold: from_env.unwrap_or(HnswOptions::default().threshold),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VecRowMeta {
    pub chunk_hash: [u8; 16],
    pub repo: String,
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
}

/// Binary sign-bit code: bit i set iff `v[i] >= 0.0`.
fn encode_code(v: &[f32]) -> Vec<u8> {
    let mut code = vec![0u8; v.len().div_ceil(8)];
    for (i, &x) in v.iter().enumerate() {
        if x >= 0.0 {
            code[i / 8] |= 1 << (i % 8);
        }
    }
    code
}

/// Smallest row range worth a rayon task in the binary prescan.
const PRESCAN_MIN_RANGE: usize = 8192;

/// Hamming distance between two binary codes (u64 chunks + byte tail).
fn hamming(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    let mut d = 0u32;
    for (x, y) in (&mut ca).zip(&mut cb) {
        let x = u64::from_le_bytes(x.try_into().unwrap());
        let y = u64::from_le_bytes(y.try_into().unwrap());
        d += (x ^ y).count_ones();
    }
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        d += (x ^ y).count_ones();
    }
    d
}

// ---------------------------------------------------------------------------
// Memory-mapped storage (the crate's only unsafe)
// ---------------------------------------------------------------------------

#[allow(unsafe_code)]
mod mapped {
    use std::fs::File;
    use std::io;

    use memmap2::Mmap;

    /// A read-only mapping of a v2 `.civec` with the offsets of its vector
    /// and code blocks.
    pub struct MappedVectors {
        map: Mmap,
        vec_off: usize,
        n_f32: usize,
        code_off: usize,
        code_len: usize,
    }

    impl MappedVectors {
        /// Map `file` and validate that `[vec_off, vec_off + 4*n_f32)` and
        /// `[code_off, code_off + code_len)` lie inside it and that the
        /// vector block is 4-byte aligned in memory.
        ///
        /// # Safety (why this is sound)
        /// `memmap2` is unsafe because another process could mutate the file
        /// under the mapping. `.civec` files are written to a temp path and
        /// atomically renamed into place; they are never modified in place
        /// (tombstone saves rewrite through the same rename), so the mapping
        /// cannot be invalidated by this crate's own writers.
        pub fn map(
            file: &File,
            vec_off: usize,
            n_f32: usize,
            code_off: usize,
            code_len: usize,
        ) -> io::Result<Self> {
            let map = unsafe { Mmap::map(file)? };
            let vec_end = vec_off
                .checked_add(n_f32.checked_mul(4).ok_or_else(bad)?)
                .ok_or_else(bad)?;
            let code_end = code_off.checked_add(code_len).ok_or_else(bad)?;
            if vec_end > map.len() || code_end > map.len() {
                return Err(bad());
            }
            if (map.as_ptr() as usize + vec_off) % 4 != 0 {
                return Err(bad());
            }
            Ok(MappedVectors {
                map,
                vec_off,
                n_f32,
                code_off,
                code_len,
            })
        }

        pub fn bytes(&self) -> &[u8] {
            &self.map[..]
        }

        /// The vector block as `&[f32]` (little-endian file on a
        /// little-endian host; alignment checked in `map`).
        #[cfg(target_endian = "little")]
        pub fn vectors(&self) -> &[f32] {
            let ptr = self.map[self.vec_off..].as_ptr() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, self.n_f32) }
        }

        pub fn codes(&self) -> &[u8] {
            &self.map[self.code_off..self.code_off + self.code_len]
        }
    }

    /// A read-only mapping of a v3 `.civec`: per-row scales, int8 rows and
    /// binary codes (SPEC-P10). Same soundness argument as
    /// [`MappedVectors`].
    pub struct MappedQuant {
        map: Mmap,
        scale_off: usize,
        q_off: usize,
        n_rows: usize,
        dim: usize,
        code_off: usize,
        code_len: usize,
    }

    impl MappedQuant {
        pub fn map(
            file: &File,
            scale_off: usize,
            q_off: usize,
            n_rows: usize,
            dim: usize,
            code_off: usize,
            code_len: usize,
        ) -> io::Result<Self> {
            let map = unsafe { Mmap::map(file)? };
            let scale_end = scale_off.checked_add(n_rows.checked_mul(4).ok_or_else(bad)?).ok_or_else(bad)?;
            let q_end = q_off.checked_add(n_rows.checked_mul(dim).ok_or_else(bad)?).ok_or_else(bad)?;
            let code_end = code_off.checked_add(code_len).ok_or_else(bad)?;
            if scale_end > map.len() || q_end > map.len() || code_end > map.len() {
                return Err(bad());
            }
            if (map.as_ptr() as usize + scale_off) % 4 != 0 {
                return Err(bad());
            }
            Ok(MappedQuant { map, scale_off, q_off, n_rows, dim, code_off, code_len })
        }

        pub fn bytes(&self) -> &[u8] {
            &self.map[..]
        }

        #[cfg(target_endian = "little")]
        pub fn scales(&self) -> &[f32] {
            let ptr = self.map[self.scale_off..].as_ptr() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, self.n_rows) }
        }

        pub fn row(&self, r: usize) -> &[i8] {
            let start = self.q_off + r * self.dim;
            let bytes = &self.map[start..start + self.dim];
            // i8 and u8 have the same layout
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, self.dim) }
        }

        pub fn codes(&self) -> &[u8] {
            &self.map[self.code_off..self.code_off + self.code_len]
        }
    }

    fn bad() -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, "civec: mapped block out of bounds")
    }
}

use mapped::{MappedQuant, MappedVectors};

/// Where the vector + code blocks live.
enum Storage {
    Owned { vectors: Vec<f32>, codes: Vec<u8> },
    Mapped(MappedVectors),
    /// v3: int8 rows + per-row scales (SPEC-P10).
    Quant(MappedQuant),
}

/// Flat binary-quantized vector index with exact rescore.
pub struct VecIndex {
    path: PathBuf,
    dim: usize,
    metas: Vec<VecRowMeta>,
    /// n_rows * dim f32 (row-major) and n_rows * ceil(dim/8) codes.
    store: Storage,
    tombstones: BTreeSet<u32>,
    /// HNSW ANN graph (SPEC-P3 §1); `None` below threshold / when absent.
    hnsw: Option<Hnsw>,
    /// Runs of consecutive rows with the same repo, `(repo, start, end)`
    /// half-open (SPEC-P10 §11): a compaction writes rows grouped by repo,
    /// so a repo-scoped search scans one run instead of testing every row.
    repo_runs: OnceLock<Vec<(String, u32, u32)>>,
}

impl VecIndex {
    fn file_path(dir: &Path, model_id: &str) -> PathBuf {
        dir.join(format!("{model_id}.civec"))
    }

    fn hnsw_path(dir: &Path, model_id: &str) -> PathBuf {
        dir.join(format!("{model_id}.cihnsw"))
    }

    fn code_bytes(&self) -> usize {
        self.dim.div_ceil(8)
    }

    /// The f32 row block, when the store has one (v1/v2; `None` for int8).
    fn vectors(&self) -> Option<&[f32]> {
        match &self.store {
            Storage::Owned { vectors, .. } => Some(vectors),
            Storage::Mapped(m) => Some(m.vectors()),
            Storage::Quant(_) => None,
        }
    }

    fn codes(&self) -> &[u8] {
        match &self.store {
            Storage::Owned { codes, .. } => codes,
            Storage::Mapped(m) => m.codes(),
            Storage::Quant(m) => m.codes(),
        }
    }

    /// Exact score of `q` against row `r` (cosine on unit vectors).
    fn score(&self, q: &[f32], r: usize) -> f32 {
        match &self.store {
            Storage::Owned { vectors, .. } => dot(q, &vectors[r * self.dim..(r + 1) * self.dim]),
            Storage::Mapped(m) => dot(q, &m.vectors()[r * self.dim..(r + 1) * self.dim]),
            Storage::Quant(m) => dot_i8(q, m.row(r), m.scales()[r]),
        }
    }

    /// Full build: writes `<dir>/<model_id>.civec` atomically (tmp+rename).
    /// Delegates to [`VecIndex::create_with_options`] with default options.
    pub fn create(
        dir: &Path,
        model_id: &str,
        dim: usize,
        rows: Vec<(VecRowMeta, Vec<f32>)>,
    ) -> anyhow::Result<Self> {
        Self::create_with_options(dir, model_id, dim, rows, &IndexOptions::default())
    }

    /// Full build with options (SPEC-P3 §1). When
    /// `rows.len() >= opts.hnsw_threshold`, also builds a deterministic HNSW
    /// graph and saves `<dir>/<model_id>.cihnsw` alongside the `.civec`;
    /// otherwise any stale `.cihnsw` is removed so a shrunken rebuild never
    /// picks up an outdated graph.
    pub fn create_with_options(
        dir: &Path,
        model_id: &str,
        dim: usize,
        rows: Vec<(VecRowMeta, Vec<f32>)>,
        opts: &IndexOptions,
    ) -> anyhow::Result<Self> {
        ensure!(dim > 0, "dim must be > 0");
        for (meta, v) in &rows {
            ensure!(
                v.len() == dim,
                "row {:?} has dim {}, expected {dim}",
                meta.path,
                v.len()
            );
        }
        fs::create_dir_all(dir)
            .with_context(|| format!("create vec dir {}", dir.display()))?;
        let mut vectors: Vec<f32> = Vec::with_capacity(rows.len() * dim);
        let mut codes: Vec<u8> = Vec::with_capacity(rows.len() * dim.div_ceil(8));
        let mut metas = Vec::with_capacity(rows.len());
        for (m, v) in rows {
            codes.extend(encode_code(&v));
            vectors.extend_from_slice(&v);
            metas.push(m);
        }
        let n_rows = metas.len();
        let idx = VecIndex {
            path: Self::file_path(dir, model_id),
            dim,
            metas,
            store: Storage::Owned { vectors, codes },
            tombstones: BTreeSet::new(),
            hnsw: None,
            repo_runs: OnceLock::new(),
        };
        idx.write_file()?;

        let hpath = Self::hnsw_path(dir, model_id);
        let hnsw = if n_rows >= opts.hnsw_threshold {
            let h = Hnsw::build_flat(dim, idx.vectors().expect("owned f32 store"), &HnswOptions::default());
            h.save(&hpath)
                .with_context(|| format!("save {}", hpath.display()))?;
            Some(h)
        } else {
            // Below threshold: flat path only; drop any stale graph sidecar.
            if hpath.exists() {
                fs::remove_file(&hpath).ok();
            }
            None
        };
        Ok(VecIndex { hnsw, ..idx })
    }

    /// Opens the index; `Ok(None)` if the file does not exist. v2 files are
    /// memory-mapped; v1 files are read into owned memory.
    pub fn open(dir: &Path, model_id: &str) -> anyhow::Result<Option<Self>> {
        let path = Self::file_path(dir, model_id);
        if !path.exists() {
            return Ok(None);
        }
        let mut idx = Self::open_file(path)?;
        // Load the HNSW sidecar if present (SPEC-P3 §1). It is rebuildable,
        // so a corrupt or stale (row-count mismatch) graph is ignored with a
        // warning rather than failing the open — search falls back to the
        // flat BQ path.
        let hpath = Self::hnsw_path(dir, model_id);
        idx.hnsw = if hpath.exists() {
            match Hnsw::load(&hpath) {
                Ok(h) if h.n_nodes() == idx.metas.len() => Some(h),
                Ok(h) => {
                    tracing::warn!(
                        "{}: stale .cihnsw ({} nodes, {} rows) — ignoring",
                        hpath.display(),
                        h.n_nodes(),
                        idx.metas.len()
                    );
                    None
                }
                Err(e) => {
                    tracing::warn!("{}: unreadable .cihnsw ({e:#}) — ignoring", hpath.display());
                    None
                }
            }
        } else {
            None
        };
        Ok(Some(idx))
    }

    fn open_file(path: PathBuf) -> anyhow::Result<Self> {
        let file = fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let mut magic = [0u8; 8];
        {
            use std::io::Read as _;
            (&file).read_exact(&mut magic).map_err(|_| anyhow!("{}: truncated .civec", path.display()))?;
        }
        if &magic == MAGIC_V1 {
            let buf = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            return Self::parse_v1(path, &buf);
        }
        if &magic == MAGIC_V3 {
            return Self::open_v3(path, file);
        }
        ensure!(&magic == MAGIC, "{}: bad .civec magic", path.display());
        // Header + metas first (small), then map the whole file for the
        // vector/code blocks.
        let header_len = 8 + 4 + 4 + 4 + 8;
        let mut head = vec![0u8; header_len];
        {
            use std::io::{Read as _, Seek as _, SeekFrom};
            let mut f = &file;
            f.seek(SeekFrom::Start(0))?;
            f.read_exact(&mut head).map_err(|_| anyhow!("{}: truncated .civec", path.display()))?;
        }
        let bad = || anyhow!("{}: corrupt .civec file", path.display());
        let dim = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
        let n_rows = u32::from_le_bytes(head[12..16].try_into().unwrap()) as usize;
        let n_deleted = u32::from_le_bytes(head[16..20].try_into().unwrap()) as usize;
        let meta_len = u64::from_le_bytes(head[20..28].try_into().unwrap()) as usize;
        ensure!(dim > 0, "dim must be > 0");
        let vec_off = align8(header_len + meta_len);
        let n_f32 = n_rows.checked_mul(dim).ok_or_else(bad)?;
        let code_off = vec_off + n_f32 * 4;
        let code_len = n_rows.checked_mul(dim.div_ceil(8)).ok_or_else(bad)?;
        let mapped = MappedVectors::map(&file, vec_off, n_f32, code_off, code_len)
            .with_context(|| format!("map {}", path.display()))?;
        let bytes = mapped.bytes();
        let metas: Vec<VecRowMeta> = bincode::deserialize(
            bytes.get(header_len..header_len + meta_len).ok_or_else(bad)?,
        )
        .map_err(|_| bad())?;
        ensure!(metas.len() == n_rows, "meta count != n_rows");
        let mut off = code_off + code_len;
        let tomb_len = u64::from_le_bytes(
            bytes.get(off..off + 8).ok_or_else(bad)?.try_into().unwrap(),
        ) as usize;
        off += 8;
        let tombs: Vec<u32> =
            bincode::deserialize(bytes.get(off..off + tomb_len).ok_or_else(bad)?)
                .map_err(|_| bad())?;
        let mut tombstones: BTreeSet<u32> = tombs.into_iter().collect();
        ensure!(
            tombstones.len() == n_deleted,
            "n_deleted header != tombstone count"
        );
        tombstones.extend(read_tomb_side(&path));
        Ok(VecIndex {
            path,
            dim,
            metas,
            store: Storage::Mapped(mapped),
            tombstones,
            hnsw: None,
            repo_runs: OnceLock::new(),
        })
    }

    /// v3 (SPEC-P10): header | metas | pad | scales f32 | rows i8 | codes | tombs.
    fn open_v3(path: PathBuf, file: fs::File) -> anyhow::Result<Self> {
        let header_len = 8 + 4 + 4 + 4 + 8;
        let mut head = vec![0u8; header_len];
        {
            use std::io::{Read as _, Seek as _, SeekFrom};
            let mut f = &file;
            f.seek(SeekFrom::Start(0))?;
            f.read_exact(&mut head).map_err(|_| anyhow!("{}: truncated .civec", path.display()))?;
        }
        let bad = || anyhow!("{}: corrupt .civec file", path.display());
        let dim = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
        let n_rows = u32::from_le_bytes(head[12..16].try_into().unwrap()) as usize;
        let n_deleted = u32::from_le_bytes(head[16..20].try_into().unwrap()) as usize;
        let meta_len = u64::from_le_bytes(head[20..28].try_into().unwrap()) as usize;
        ensure!(dim > 0, "dim must be > 0");
        let scale_off = align8(header_len + meta_len);
        let q_off = scale_off + n_rows * 4;
        let code_off = align8(q_off + n_rows.checked_mul(dim).ok_or_else(bad)?);
        let code_len = n_rows.checked_mul(dim.div_ceil(8)).ok_or_else(bad)?;
        let mapped = MappedQuant::map(&file, scale_off, q_off, n_rows, dim, code_off, code_len)
            .with_context(|| format!("map {}", path.display()))?;
        let bytes = mapped.bytes();
        let metas: Vec<VecRowMeta> =
            bincode::deserialize(bytes.get(header_len..header_len + meta_len).ok_or_else(bad)?).map_err(|_| bad())?;
        ensure!(metas.len() == n_rows, "meta count != n_rows");
        let mut off = code_off + code_len;
        let tomb_len = u64::from_le_bytes(bytes.get(off..off + 8).ok_or_else(bad)?.try_into().unwrap()) as usize;
        off += 8;
        let tombs: Vec<u32> = bincode::deserialize(bytes.get(off..off + tomb_len).ok_or_else(bad)?).map_err(|_| bad())?;
        let mut tombstones: BTreeSet<u32> = tombs.into_iter().collect();
        ensure!(tombstones.len() == n_deleted, "n_deleted header != tombstone count");
        tombstones.extend(read_tomb_side(&path));
        Ok(VecIndex { path, dim, metas, store: Storage::Quant(mapped), tombstones, hnsw: None, repo_runs: OnceLock::new() })
    }

    /// Open one segment file by path (no HNSW sidecar lookup).
    pub fn open_segment(path: &Path) -> anyhow::Result<Self> {
        Self::open_file(path.to_path_buf())
    }

    /// Write one flat segment file at `path` (tmp+rename; no HNSW).
    pub fn create_segment(
        path: &Path,
        dim: usize,
        rows: Vec<(VecRowMeta, Vec<f32>)>,
    ) -> anyhow::Result<()> {
        ensure!(dim > 0, "dim must be > 0");
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).with_context(|| format!("create vec dir {}", dir.display()))?;
        }
        if std::env::var_os("INDEXIO_VEC_F32").is_some() {
            // A/B knob: the pre-P10 f32 layout (v2)
            let mut vectors: Vec<f32> = Vec::with_capacity(rows.len() * dim);
            let mut codes: Vec<u8> = Vec::with_capacity(rows.len() * dim.div_ceil(8));
            let mut metas = Vec::with_capacity(rows.len());
            for (m, v) in rows {
                ensure!(v.len() == dim, "row {:?} has dim {}, expected {dim}", m.path, v.len());
                codes.extend(encode_code(&v));
                vectors.extend_from_slice(&v);
                metas.push(m);
            }
            let idx = VecIndex {
                path: path.to_path_buf(),
                dim,
                metas,
                store: Storage::Owned { vectors, codes },
                tombstones: BTreeSet::new(),
                hnsw: None,
                repo_runs: OnceLock::new(),
            };
            idx.write_file()?;
            let side = tomb_side_path(path);
            if side.exists() {
                fs::remove_file(&side).ok();
            }
            return Ok(());
        }
        // v3 (SPEC-P10): int8 rows with a per-row scale
        let mut scales: Vec<u8> = Vec::with_capacity(rows.len() * 4);
        let mut qrows: Vec<u8> = Vec::with_capacity(rows.len() * dim);
        let mut codes: Vec<u8> = Vec::with_capacity(rows.len() * dim.div_ceil(8));
        let mut metas = Vec::with_capacity(rows.len());
        for (m, v) in rows {
            ensure!(v.len() == dim, "row {:?} has dim {}, expected {dim}", m.path, v.len());
            codes.extend(encode_code(&v));
            let (scale, q) = quantize_row(&v);
            scales.extend_from_slice(&scale.to_le_bytes());
            qrows.extend(q.into_iter().map(|x| x as u8));
            metas.push(m);
        }
        let metas_bytes = bincode::serialize(&metas)?;
        let tombs = bincode::serialize(&Vec::<u32>::new())?;
        let header_len = 8 + 4 + 4 + 4 + 8;
        let scale_off = align8(header_len + metas_bytes.len());
        let code_off = align8(scale_off + scales.len() + qrows.len());
        let mut out = Vec::with_capacity(code_off + codes.len() + 8 + tombs.len());
        out.extend_from_slice(MAGIC_V3);
        out.extend_from_slice(&(dim as u32).to_le_bytes());
        out.extend_from_slice(&(metas.len() as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(metas_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&metas_bytes);
        out.resize(scale_off, 0);
        out.extend_from_slice(&scales);
        out.extend_from_slice(&qrows);
        out.resize(code_off, 0);
        out.extend_from_slice(&codes);
        out.extend_from_slice(&(tombs.len() as u64).to_le_bytes());
        out.extend_from_slice(&tombs);
        let tmp = path.with_extension("civec.tmp");
        {
            let mut f = fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
            f.write_all(&out)?;
            f.sync_all().ok();
        }
        fs::rename(&tmp, path).with_context(|| format!("rename to {}", path.display()))?;
        let side = tomb_side_path(path);
        if side.exists() {
            fs::remove_file(&side).ok();
        }
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// v1 (`CIVEC001`, unpadded) reader: owned memory.
    fn parse_v1(path: PathBuf, buf: &[u8]) -> anyhow::Result<Self> {
        let bad = || anyhow!("{}: corrupt .civec file", path.display());
        let mut off = 0usize;
        let take = |off: &mut usize, n: usize| -> anyhow::Result<&[u8]> {
            let s = buf.get(*off..*off + n).ok_or_else(bad)?;
            *off += n;
            Ok(s)
        };
        ensure!(take(&mut off, 8)? == MAGIC_V1, "bad .civec magic");
        let dim = u32::from_le_bytes(take(&mut off, 4)?.try_into().unwrap()) as usize;
        let n_rows = u32::from_le_bytes(take(&mut off, 4)?.try_into().unwrap()) as usize;
        let n_deleted = u32::from_le_bytes(take(&mut off, 4)?.try_into().unwrap()) as usize;
        ensure!(dim > 0, "dim must be > 0");

        let meta_len = u64::from_le_bytes(take(&mut off, 8)?.try_into().unwrap()) as usize;
        let metas: Vec<VecRowMeta> = bincode::deserialize(take(&mut off, meta_len)?)
            .map_err(|_| bad())?;
        ensure!(metas.len() == n_rows, "meta count != n_rows");

        let vec_bytes = n_rows
            .checked_mul(dim)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(bad)?;
        let vec_raw = take(&mut off, vec_bytes)?;
        let mut vectors = Vec::with_capacity(n_rows * dim);
        for c in vec_raw.chunks_exact(4) {
            vectors.push(f32::from_le_bytes(c.try_into().unwrap()));
        }

        let code_bytes = dim.div_ceil(8);
        let codes = take(&mut off, n_rows.checked_mul(code_bytes).ok_or_else(bad)?)?.to_vec();

        let tomb_len = u64::from_le_bytes(take(&mut off, 8)?.try_into().unwrap()) as usize;
        let tombs: Vec<u32> = bincode::deserialize(take(&mut off, tomb_len)?)
            .map_err(|_| bad())?;
        let mut tombstones: BTreeSet<u32> = tombs.into_iter().collect();
        ensure!(
            tombstones.len() == n_deleted,
            "n_deleted header != tombstone count"
        );
        tombstones.extend(read_tomb_side(&path));

        Ok(VecIndex {
            path,
            dim,
            metas,
            store: Storage::Owned { vectors, codes },
            tombstones,
            hnsw: None,
            repo_runs: OnceLock::new(),
        })
    }

    /// Serializes the full current state (rows + tombstones) as v2 via
    /// tmp+rename. Callers with a mapped store must drop the mapping first
    /// (`save_tombstones` does) — Windows refuses to replace a mapped file.
    fn write_file(&self) -> anyhow::Result<()> {
        let metas = bincode::serialize(&self.metas)?;
        let tombs: Vec<u32> = self.tombstones.iter().copied().collect();
        let tombs = bincode::serialize(&tombs)?;
        let vectors = self.vectors().ok_or_else(|| anyhow!("int8 segments are written by create_segment"))?;
        let codes = self.codes();
        let header_len = 8 + 4 + 4 + 4 + 8;
        let vec_off = align8(header_len + metas.len());

        let mut out = Vec::with_capacity(vec_off + vectors.len() * 4 + codes.len() + 8 + tombs.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.metas.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.tombstones.len() as u32).to_le_bytes());
        out.extend_from_slice(&(metas.len() as u64).to_le_bytes());
        out.extend_from_slice(&metas);
        out.resize(vec_off, 0);
        for x in vectors {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out.extend_from_slice(codes);
        out.extend_from_slice(&(tombs.len() as u64).to_le_bytes());
        out.extend_from_slice(&tombs);

        let tmp = self.path.with_extension("civec.tmp");
        {
            let mut f = fs::File::create(&tmp)
                .with_context(|| format!("create {}", tmp.display()))?;
            f.write_all(&out)?;
            f.sync_all().ok();
        }
        fs::rename(&tmp, &self.path)
            .with_context(|| format!("rename to {}", self.path.display()))?;
        Ok(())
    }

    /// Exact top-k cosine (dot on L2-normalized vectors). Returns
    /// `(row, score)` sorted by score desc, ties by row asc.
    ///
    /// Two candidate paths (SPEC-P3 §1):
    /// - HNSW present: approximate `hnsw.search` with `ef = max(k*10, 100)`,
    ///   then exact f32 rescore of candidates, skipping tombstoned rows.
    /// - Otherwise: binary prescan keeps `max(k*8, 256)` candidates (capped
    ///   at live rows), rescored exactly. Unchanged from SPEC-P2.
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(u32, f32)> {
        self.search_excluding(q, k, &BTreeSet::new())
    }

    /// [`search`](Self::search) that also skips the rows in `dead` — the
    /// tombstones a [`VecSet`] keeps outside the shared segment.
    pub fn search_excluding(&self, q: &[f32], k: usize, dead: &BTreeSet<u32>) -> Vec<(u32, f32)> {
        self.search_where(q, k, dead, &|_| true)
    }

    /// Runs of same-repo rows (computed once from the metas).
    fn repo_runs(&self) -> &[(String, u32, u32)] {
        self.repo_runs.get_or_init(|| {
            let mut runs: Vec<(String, u32, u32)> = Vec::new();
            for (i, m) in self.metas.iter().enumerate() {
                match runs.last_mut() {
                    Some((r, _, end)) if *r == m.repo => *end = i as u32 + 1,
                    _ => runs.push((m.repo.clone(), i as u32, i as u32 + 1)),
                }
            }
            runs
        })
    }

    /// Rows of `repo` as runs; `None` when the repo has no rows here.
    fn repo_ranges(&self, repo: &str) -> Option<Vec<(u32, u32)>> {
        let v: Vec<(u32, u32)> = self.repo_runs().iter().filter(|(r, _, _)| r == repo).map(|&(_, s, e)| (s, e)).collect();
        (!v.is_empty()).then_some(v)
    }

    /// [`search_excluding`](Self::search_excluding) over the rows of `repo`
    /// only (SPEC-P10 §11): the prescan walks that repo's row runs — one
    /// run after a compaction — instead of testing every row of the segment.
    pub fn search_in_repo(&self, q: &[f32], k: usize, dead: &BTreeSet<u32>, repo: &str) -> Vec<(u32, f32)> {
        let Some(ranges) = self.repo_ranges(repo) else { return Vec::new() };
        if k == 0 || q.len() != self.dim {
            return Vec::new();
        }
        let is_dead = |r: u32| self.tombstones.contains(&r) || dead.contains(&r);
        let qcode = encode_code(q);
        let cb = self.code_bytes();
        let codes = self.codes();
        let total: usize = ranges.iter().map(|&(s, e)| (e - s) as usize).sum();
        let pool = (k * 8).max(256).min(total.max(1));
        let mut cands: Vec<(u32, u32)> = ranges
            .par_iter()
            .map(|&(lo, hi)| {
                let mut v: Vec<(u32, u32)> = (lo..hi)
                    .filter(|&r| !is_dead(r))
                    .map(|r| {
                        let row_code = &codes[r as usize * cb..(r as usize + 1) * cb];
                        (hamming(row_code, &qcode), r)
                    })
                    .collect();
                if v.len() > pool {
                    v.select_nth_unstable(pool - 1);
                    v.truncate(pool);
                }
                v
            })
            .reduce(Vec::new, |mut a, b| {
                a.extend(b);
                a
            });
        if cands.len() > pool {
            cands.select_nth_unstable(pool - 1);
            cands.truncate(pool);
        }
        cands.sort_unstable();
        let mut scored: Vec<(u32, f32)> = cands.into_iter().map(|(_, r)| (r, self.score(q, r as usize))).collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.truncate(k);
        scored
    }

    /// [`search_excluding`](Self::search_excluding) restricted to rows for
    /// which `keep(row)` holds (SPEC-P10: a repo-scoped semantic leg — the
    /// prescan pool is then made of matching rows only, instead of a
    /// post-filter over a pool that a large corpus fills with other repos).
    pub fn search_where(&self, q: &[f32], k: usize, dead: &BTreeSet<u32>, keep: &(dyn Fn(u32) -> bool + Sync)) -> Vec<(u32, f32)> {
        let n = self.metas.len();
        let live = n.saturating_sub(self.tombstones.len().max(dead.len()));
        if k == 0 || live == 0 || q.len() != self.dim {
            return Vec::new();
        }
        let is_dead = |r: u32| self.tombstones.contains(&r) || dead.contains(&r) || !keep(r);
        let dim = self.dim;

        if let (Some(h), Some(vectors)) = (&self.hnsw, self.vectors()) {
            let ef = (k * 10).max(100);
            let cands = h.search_flat(dim, vectors, q, ef);
            let mut scored: Vec<(u32, f32)> = cands
                .into_iter()
                .filter(|&r| !is_dead(r))
                .map(|r| (r, self.score(q, r as usize)))
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            scored.truncate(k);
            return scored;
        }

        let qcode = encode_code(q);
        let cb = self.code_bytes();
        let codes = self.codes();

        // Prescan: hamming distance over binary codes; keep the best `pool`
        // with a partial selection instead of a full sort. The scan is
        // split into row ranges scored in parallel (each keeps its own
        // best `pool`), then the partial winners are merged: identical
        // result to the serial scan, since selection is by (distance, row).
        let pool = (k * 8).max(256).min(live);
        let select_pool = |mut v: Vec<(u32, u32)>| {
            if v.len() > pool {
                v.select_nth_unstable(pool - 1);
                v.truncate(pool);
            }
            v
        };
        let range_len = (n / rayon::current_num_threads().max(1)).max(PRESCAN_MIN_RANGE);
        let ranges: Vec<(u32, u32)> = (0..n)
            .step_by(range_len)
            .map(|s| (s as u32, (s + range_len).min(n) as u32))
            .collect();
        let mut cands: Vec<(u32, u32)> = if ranges.len() <= 1 {
            select_pool(
                (0..n as u32)
                    .filter(|&r| !is_dead(r))
                    .map(|r| {
                        let row_code = &codes[r as usize * cb..(r as usize + 1) * cb];
                        (hamming(row_code, &qcode), r)
                    })
                    .collect(),
            )
        } else {
            let partial: Vec<Vec<(u32, u32)>> = ranges
                .par_iter()
                .map(|&(lo, hi)| {
                    select_pool(
                        (lo..hi)
                            .filter(|&r| !is_dead(r))
                            .map(|r| {
                                let row_code = &codes[r as usize * cb..(r as usize + 1) * cb];
                                (hamming(row_code, &qcode), r)
                            })
                            .collect(),
                    )
                })
                .collect();
            select_pool(partial.into_iter().flatten().collect())
        };
        cands.sort_unstable();

        // Rescore exactly (f32 dot, or scale * int8 dot for v3 rows).
        let mut scored: Vec<(u32, f32)> = cands.into_iter().map(|(_, r)| (r, self.score(q, r as usize))).collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.truncate(k);
        scored
    }

    pub fn row_meta(&self, row: u32) -> &VecRowMeta {
        &self.metas[row as usize]
    }

    /// Rows tombstoned inside the file or its side file at open time.
    pub fn tombstones(&self) -> &BTreeSet<u32> {
        &self.tombstones
    }

    /// Tombstone every row matching `pred`. Returns rows newly deleted.
    pub fn delete_where(&mut self, pred: impl Fn(&VecRowMeta) -> bool) -> u64 {
        let mut n = 0u64;
        for (i, m) in self.metas.iter().enumerate() {
            if !self.tombstones.contains(&(i as u32)) && pred(m) {
                self.tombstones.insert(i as u32);
                n += 1;
            }
        }
        n
    }

    /// Persists tombstones (SPEC-P9): the full set goes to the small
    /// `<file>.tomb` side file (tmp+rename); the segment itself — possibly
    /// gigabytes, possibly mapped by other processes — is never rewritten.
    pub fn save_tombstones(&mut self) -> anyhow::Result<()> {
        write_tomb_side(&self.path, &self.tombstones)
    }

    /// Number of live (non-tombstoned) rows.
    pub fn len(&self) -> usize {
        self.metas.len() - self.tombstones.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total row slots including tombstoned (pipeline rebuild iterates
    /// this; indexio-query uses it to guard the row-aligned BM25 sidecar).
    pub fn raw_len(&self) -> usize {
        self.metas.len()
    }

    pub fn is_deleted(&self, row: u32) -> bool {
        self.tombstones.contains(&row)
    }

    /// The row as f32 (dequantised for v3 segments): what a compaction carries.
    pub(crate) fn row_vector(&self, row: u32) -> Vec<f32> {
        let r = row as usize;
        match &self.store {
            Storage::Quant(m) => {
                let s = m.scales()[r];
                m.row(r).iter().map(|&x| x as f32 * s).collect()
            }
            _ => self.vectors().expect("f32 store")[r * self.dim..(r + 1) * self.dim].to_vec(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }
}

fn align8(v: usize) -> usize {
    (v + 7) & !7
}

/// `<segment>.tomb`: bincode `Vec<u32>` of tombstoned rows (SPEC-P9).
pub(crate) fn tomb_side_path(seg: &Path) -> PathBuf {
    let mut s = seg.as_os_str().to_owned();
    s.push(".tomb");
    PathBuf::from(s)
}

pub(crate) fn read_tomb_side(seg: &Path) -> Vec<u32> {
    let side = tomb_side_path(seg);
    match fs::read(&side) {
        Ok(buf) => bincode::deserialize::<Vec<u32>>(&buf).unwrap_or_else(|_| {
            tracing::warn!("{}: corrupt tombstone side file — ignoring", side.display());
            Vec::new()
        }),
        Err(_) => Vec::new(),
    }
}

pub(crate) fn write_tomb_side(seg: &Path, tombs: &BTreeSet<u32>) -> anyhow::Result<()> {
    let side = tomb_side_path(seg);
    // union with what another server wrote since this set was opened
    let mut all: BTreeSet<u32> = read_tomb_side(seg).into_iter().collect();
    all.extend(tombs.iter().copied());
    let bytes = bincode::serialize(&all.into_iter().collect::<Vec<u32>>())?;
    let tmp = {
        let mut s = side.as_os_str().to_owned();
        s.push(".tmp");
        PathBuf::from(s)
    };
    fs::write(&tmp, &bytes).with_context(|| format!("write {}", tmp.display()))?;
    fs::rename(&tmp, &side).with_context(|| format!("rename to {}", side.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// VecSet: the segmented vector index (SPEC-P9)
// ---------------------------------------------------------------------------

/// Extension merged-but-still-mapped segments are parked under.
pub const STALE_EXT: &str = "stale";

/// Segment files of `model_id` in `dir`: the legacy base `<model>.civec`
/// (if present) first, then `<model>.d<NNNN>.civec` ascending.
pub fn segment_paths(dir: &Path, model_id: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let base = VecIndex::file_path(dir, model_id);
    if base.exists() {
        out.push(base);
    }
    let prefix = format!("{model_id}.d");
    let mut deltas: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()) == Some("civec")
                    && p
                        .file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with(&prefix))
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    deltas.sort();
    out.extend(deltas);
    out
}

/// A fresh delta segment path. The name is `d<ms since epoch, 13
/// digits><pid, 6 digits><counter, 2 digits>`: several server processes
/// share one data dir (one per agent session) and refresh concurrently, so
/// a `max + 1` scheme could hand two of them the same name and have the
/// second overwrite the first's rows. Fixed widths keep the lexical order
/// chronological, which is all `segment_paths` relies on.
pub fn next_delta_path(dir: &Path, model_id: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let pid = std::process::id() % 1_000_000;
    let seq = SEQ.fetch_add(1, Ordering::Relaxed) % 100;
    dir.join(format!("{model_id}.d{ms:013}{pid:06}{seq:02}.civec"))
}

/// Cache stamp over every segment and tombstone side file: (total bytes,
/// newest mtime). Changes whenever a refresh touched the plane.
pub fn set_stamp(dir: &Path, model_id: &str) -> Option<(u64, Option<std::time::SystemTime>)> {
    let paths = segment_paths(dir, model_id);
    if paths.is_empty() {
        return None;
    }
    let mut len = 0u64;
    let mut newest: Option<std::time::SystemTime> = None;
    for p in paths.iter().chain(paths.iter().map(|p| tomb_side_path(p)).collect::<Vec<_>>().iter()) {
        if let Ok(m) = fs::metadata(p) {
            len += m.len();
            if let Ok(t) = m.modified() {
                newest = Some(newest.map_or(t, |n| n.max(t)));
            }
        }
    }
    Some((len, newest))
}

/// All segments of one model, addressed by a global row id (segment
/// offsets are cumulative `raw_len`s). Rows never move: a refresh appends
/// a segment and tombstones rows through side files; compaction rewrites
/// the small segments into one. Segments are shared (`Arc`) so a reopen
/// after a refresh reuses every segment whose file is unchanged and only
/// reloads the side-file tombstones — a 1.6 GB base is parsed once per
/// process, not once per edit.
pub struct VecSet {
    segs: Vec<std::sync::Arc<VecIndex>>,
    stamps: Vec<(u64, Option<std::time::SystemTime>)>,
    /// Per segment: in-file tombstones ∪ side-file tombstones.
    tombs: Vec<BTreeSet<u32>>,
    offsets: Vec<u32>,
    dim: usize,
}

fn file_stamp(p: &Path) -> (u64, Option<std::time::SystemTime>) {
    match fs::metadata(p) {
        Ok(m) => (m.len(), m.modified().ok()),
        Err(_) => (0, None),
    }
}

impl VecSet {
    /// Open every segment; `Ok(None)` when the model has no segment yet.
    pub fn open(dir: &Path, model_id: &str) -> anyhow::Result<Option<Self>> {
        Self::reopen(None, dir, model_id)
    }

    /// Open the current segment list, reusing `prev`'s parsed segments
    /// whose files are byte-identical (same length and mtime).
    pub fn reopen(prev: Option<&VecSet>, dir: &Path, model_id: &str) -> anyhow::Result<Option<Self>> {
        let paths = segment_paths(dir, model_id);
        if paths.is_empty() {
            return Ok(None);
        }
        let mut segs = Vec::with_capacity(paths.len());
        let mut stamps = Vec::with_capacity(paths.len());
        let mut tombs = Vec::with_capacity(paths.len());
        for (i, p) in paths.iter().enumerate() {
            let stamp = file_stamp(p);
            let reused = prev.and_then(|pv| {
                pv.segs
                    .iter()
                    .zip(&pv.stamps)
                    .find(|(s, st)| s.path() == p.as_path() && **st == stamp)
                    .map(|(s, _)| std::sync::Arc::clone(s))
            });
            let seg = match reused {
                Some(s) => s,
                None => std::sync::Arc::new(if i == 0 && p == &VecIndex::file_path(dir, model_id) {
                    // the legacy base may carry an HNSW sidecar
                    VecIndex::open(dir, model_id)?.ok_or_else(|| anyhow!("{} vanished", p.display()))?
                } else {
                    VecIndex::open_segment(p)?
                }),
            };
            let mut t = seg.tombstones().clone();
            t.extend(read_tomb_side(p));
            segs.push(seg);
            stamps.push(stamp);
            tombs.push(t);
        }
        let dim = segs[0].dim;
        ensure!(segs.iter().all(|s| s.dim == dim), "vector segments disagree on dim");
        let mut offsets = Vec::with_capacity(segs.len());
        let mut acc = 0u32;
        for s in &segs {
            offsets.push(acc);
            acc = acc
                .checked_add(s.raw_len() as u32)
                .ok_or_else(|| anyhow!("too many vector rows"))?;
        }
        Ok(Some(VecSet { segs, stamps, tombs, offsets, dim }))
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn segments(&self) -> &[std::sync::Arc<VecIndex>] {
        &self.segs
    }

    /// Total row slots across segments (tombstoned included).
    pub fn raw_len(&self) -> usize {
        self.segs.iter().map(|s| s.raw_len()).sum()
    }

    /// Live rows.
    pub fn len(&self) -> usize {
        self.segs
            .iter()
            .zip(&self.tombs)
            .map(|(s, t)| s.raw_len().saturating_sub(t.len()))
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// (segment index, local row) of a global row.
    pub fn locate(&self, row: u32) -> (usize, u32) {
        let i = match self.offsets.binary_search(&row) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        (i, row - self.offsets[i])
    }

    pub fn row_meta(&self, row: u32) -> &VecRowMeta {
        let (s, r) = self.locate(row);
        self.segs[s].row_meta(r)
    }

    pub fn is_deleted(&self, row: u32) -> bool {
        let (s, r) = self.locate(row);
        self.tombs[s].contains(&r)
    }

    pub fn row_vector(&self, row: u32) -> Vec<f32> {
        let (s, r) = self.locate(row);
        self.segs[s].row_vector(r)
    }

    /// Global ids of every live row.
    pub fn live_rows(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.len());
        for (i, s) in self.segs.iter().enumerate() {
            for r in 0..s.raw_len() as u32 {
                if !self.tombs[i].contains(&r) {
                    out.push(self.offsets[i] + r);
                }
            }
        }
        out
    }

    /// Top-k over all segments (cosine scores are comparable across them).
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(u32, f32)> {
        self.search_where(q, k, &|_| true)
    }

    /// [`search`](Self::search) over the rows of `repo` only, walking each
    /// segment's runs of that repo (SPEC-P10 §11).
    pub fn search_repo(&self, q: &[f32], k: usize, repo: &str) -> Vec<(u32, f32)> {
        let mut out: Vec<(u32, f32)> = Vec::new();
        for (i, s) in self.segs.iter().enumerate() {
            for (r, score) in s.search_in_repo(q, k, &self.tombs[i], repo) {
                out.push((self.offsets[i] + r, score));
            }
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(k);
        out
    }

    /// [`search`](Self::search) over the rows whose meta satisfies `keep`.
    pub fn search_where(&self, q: &[f32], k: usize, keep: &(dyn Fn(&VecRowMeta) -> bool + Sync)) -> Vec<(u32, f32)> {
        let mut out: Vec<(u32, f32)> = Vec::new();
        for (i, s) in self.segs.iter().enumerate() {
            let pred = |r: u32| keep(s.row_meta(r));
            for (r, score) in s.search_where(q, k, &self.tombs[i], &pred) {
                out.push((self.offsets[i] + r, score));
            }
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(k);
        out
    }

    /// Tombstone every live row matching `pred`; returns rows newly deleted.
    pub fn delete_where(&mut self, pred: impl Fn(&VecRowMeta) -> bool) -> u64 {
        let mut n = 0u64;
        for (i, s) in self.segs.iter().enumerate() {
            for r in 0..s.raw_len() as u32 {
                if !self.tombs[i].contains(&r) && pred(s.row_meta(r)) {
                    self.tombs[i].insert(r);
                    n += 1;
                }
            }
        }
        n
    }

    /// Tombstone the given global rows; returns rows newly deleted.
    pub fn delete_rows(&mut self, rows: &std::collections::HashSet<u32>) -> u64 {
        let mut n = 0u64;
        for &g in rows {
            let (s, r) = self.locate(g);
            if self.tombs[s].insert(r) {
                n += 1;
            }
        }
        n
    }

    /// Persist every segment's tombstones (side files).
    pub fn save_tombstones(&self) -> anyhow::Result<()> {
        for (s, t) in self.segs.iter().zip(&self.tombs) {
            write_tomb_side(s.path(), t)?;
        }
        Ok(())
    }

    /// Write side files with `rows` added to the current tombstones,
    /// WITHOUT mutating this (possibly shared) set: the caller reopens.
    pub fn tombstone_on_disk(&self, rows: &std::collections::HashSet<u32>) -> anyhow::Result<()> {
        let mut per_seg: Vec<BTreeSet<u32>> = self.tombs.clone();
        for &g in rows {
            let (s, r) = self.locate(g);
            per_seg[s].insert(r);
        }
        for (i, s) in self.segs.iter().enumerate() {
            if per_seg[i].len() != self.tombs[i].len() {
                write_tomb_side(s.path(), &per_seg[i])?;
            }
        }
        Ok(())
    }

    /// Tombstoned fraction of all row slots.
    pub fn dead_fraction(&self) -> f64 {
        let raw = self.raw_len();
        if raw == 0 {
            0.0
        } else {
            1.0 - self.len() as f64 / raw as f64
        }
    }
}

/// (segments, row slots, tombstoned rows) of a model from the segment
/// headers and side files alone — no vector or meta parsing.
pub fn segment_stats(dir: &Path, model_id: &str) -> (usize, u64, u64) {
    let paths = segment_paths(dir, model_id);
    let mut rows = 0u64;
    let mut dead = 0u64;
    for p in &paths {
        if let Ok(f) = fs::File::open(p) {
            use std::io::Read as _;
            let mut head = [0u8; 20];
            if (&f).read_exact(&mut head).is_ok() && (&head[..8] == MAGIC || &head[..8] == MAGIC_V1 || &head[..8] == MAGIC_V3) {
                rows += u64::from(u32::from_le_bytes(head[12..16].try_into().unwrap()));
                dead += u64::from(u32::from_le_bytes(head[16..20].try_into().unwrap()));
            }
        }
        // the side file supersedes the in-file list (it is a superset)
        let side = read_tomb_side(p);
        if !side.is_empty() {
            dead = dead.max(side.len() as u64);
        }
    }
    (paths.len(), rows, dead)
}

/// Remove a segment file (and its side files); when another process still
/// has it mapped (Windows), park it under [`STALE_EXT`] instead — ignored
/// by [`segment_paths`], reaped by [`reap_stale`].
pub fn remove_or_park(seg: &Path) -> anyhow::Result<()> {
    let side = tomb_side_path(seg);
    fs::remove_file(&side).ok();
    if let Err(e) = fs::remove_file(seg) {
        let parked = seg.with_extension(STALE_EXT);
        fs::rename(seg, &parked)
            .with_context(|| format!("cannot remove {} ({e}) nor park it", seg.display()))?;
    }
    Ok(())
}

/// Delete parked segments whose last mapping is gone.
pub fn reap_stale(dir: &Path) {
    if let Ok(rd) = fs::read_dir(dir) {
        for p in rd.flatten().map(|e| e.path()) {
            if p.extension().and_then(|e| e.to_str()) == Some(STALE_EXT) {
                fs::remove_file(&p).ok();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::chunk_hash;

    fn meta(repo: &str, path: &str) -> VecRowMeta {
        VecRowMeta {
            chunk_hash: chunk_hash(path.as_bytes()),
            repo: repo.to_string(),
            path: path.to_string(),
            start_line: 1,
            end_line: 10,
        }
    }

    fn norm(mut v: Vec<f32>) -> Vec<f32> {
        let n = dot(&v, &v).sqrt();
        if n > 0.0 {
            for x in &mut v {
                *x /= n;
            }
        }
        v
    }

    /// Deterministic PRNG (xorshift64) so tests need no rand dep.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            // uniform in [-1, 1)
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 1.0
        }
    }

    #[test]
    fn hamming_and_code_sanity() {
        let a = encode_code(&[1.0, -1.0, 1.0, -1.0]);
        let b = encode_code(&[1.0, -1.0, 1.0, -1.0]);
        let c = encode_code(&[-1.0, 1.0, -1.0, 1.0]);
        assert_eq!(hamming(&a, &b), 0, "identical vectors -> distance 0");
        assert_eq!(hamming(&a, &c), 4, "opposite signs -> distance = dim");
    }

    #[test]
    fn vecindex_create_open_search_ordering() {
        let tmp = tempfile::tempdir().unwrap();
        let rows = vec![
            (meta("r1", "a.rs"), norm(vec![1.0, 0.05, 0.0, 0.0])),
            (meta("r1", "b.rs"), norm(vec![0.0, 1.0, 0.0, 0.0])),
            (meta("r1", "c.rs"), norm(vec![0.0, 0.0, 1.0, 0.0])),
            (meta("r2", "d.rs"), norm(vec![0.0, 0.0, 0.0, 1.0])),
        ];
        let q = norm(vec![1.0, 0.1, 0.0, 0.0]);
        let expected = ["a.rs", "b.rs", "c.rs"]; // strict dot ordering

        let idx = VecIndex::create(tmp.path(), "hash-v1", 4, rows.clone()).unwrap();
        assert_eq!(idx.len(), 4);
        let hits = idx.search(&q, 3);
        let got: Vec<&str> = hits.iter().map(|(r, _)| idx.row_meta(*r).path.as_str()).collect();
        assert_eq!(got, expected, "planted nearest neighbor must rank first");
        assert!(hits[0].1 > hits[1].1 && hits[1].1 > hits[2].1);

        // Reopen from disk and repeat.
        let idx2 = VecIndex::open(tmp.path(), "hash-v1").unwrap().unwrap();
        assert_eq!(idx2.len(), 4);
        let hits2 = idx2.search(&q, 3);
        let got2: Vec<&str> =
            hits2.iter().map(|(r, _)| idx2.row_meta(*r).path.as_str()).collect();
        assert_eq!(got2, expected, "ordering must survive reopen");

        // Absent index -> Ok(None); k > n -> n results.
        assert!(VecIndex::open(tmp.path(), "no-such-model").unwrap().is_none());
        assert_eq!(idx2.search(&q, 100).len(), 4);
        assert!(idx2.search(&q, 0).is_empty());
    }

    #[test]
    fn int8_segment_matches_f32_ranking() {
        // SPEC-P10: a v3 segment (int8 rows) ranks like the f32 rows it was
        // built from, and its rows round-trip through a compaction exactly.
        let tmp = tempfile::tempdir().unwrap();
        // n below the prescan pool (256): every row is rescored, so any
        // difference from brute force is the quantisation alone
        let (dim, n, k) = (256usize, 250usize, 10usize);
        let mut rng = Rng(0xC0FFEE);
        let rows: Vec<(VecRowMeta, Vec<f32>)> = (0..n)
            .map(|i| (meta("r", &format!("f{i}.rs")), norm((0..dim).map(|_| rng.next_f32()).collect())))
            .collect();
        let seg = tmp.path().join("m.d1.civec");
        VecIndex::create_segment(&seg, dim, rows.clone()).unwrap();
        assert_eq!(&fs::read(&seg).unwrap()[..8], MAGIC_V3);
        let idx = VecIndex::open_segment(&seg).unwrap();
        assert!(matches!(idx.store, Storage::Quant(_)));
        let mut agree_top1 = 0;
        let mut overlap = 0;
        for _ in 0..20 {
            let q = norm((0..dim).map(|_| rng.next_f32()).collect());
            let hits = idx.search(&q, k);
            let mut brute: Vec<(u32, f32)> =
                rows.iter().enumerate().map(|(i, (_, v))| (i as u32, dot(&q, v))).collect();
            brute.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            brute.truncate(k);
            agree_top1 += usize::from(hits[0].0 == brute[0].0);
            overlap += hits.iter().filter(|(r, _)| brute.iter().any(|(b, _)| b == r)).count();
            // int8 scores are within 1e-2 of the exact dot
            for (r, s) in &hits {
                assert!((s - dot(&q, &rows[*r as usize].1)).abs() < 1e-2, "{s} vs exact");
            }
        }
        assert!(agree_top1 >= 19, "top-1 agreement {agree_top1}/20");
        assert!(overlap >= 196, "top-{k} overlap {overlap}/200");
        // exact round-trip: dequantise -> requantise gives the same bytes
        let carried: Vec<(VecRowMeta, Vec<f32>)> = (0..n as u32).map(|r| (idx.row_meta(r).clone(), idx.row_vector(r))).collect();
        let seg2 = tmp.path().join("m.d2.civec");
        VecIndex::create_segment(&seg2, dim, carried).unwrap();
        let idx2 = VecIndex::open_segment(&seg2).unwrap();
        let (Storage::Quant(m1), Storage::Quant(m2)) = (&idx.store, &idx2.store) else { panic!("v3") };
        for r in 0..n {
            assert_eq!(m1.row(r), m2.row(r), "row {r} requantised differently");
            assert!((m1.scales()[r] - m2.scales()[r]).abs() <= f32::EPSILON * m1.scales()[r]);
        }
    }

    #[test]
    fn search_in_repo_matches_filtered_search_over_runs() {
        // rows of three repos interleaved (several runs per repo): the
        // run-scoped search returns exactly what the predicate search does
        let tmp = tempfile::tempdir().unwrap();
        let (dim, n) = (64usize, 600usize);
        let mut rng = Rng(0xBEEF);
        let rows: Vec<(VecRowMeta, Vec<f32>)> = (0..n)
            .map(|i| {
                let repo = ["a", "b", "c"][(i / 50) % 3];
                (meta(repo, &format!("f{i}.rs")), norm((0..dim).map(|_| rng.next_f32()).collect()))
            })
            .collect();
        let seg = tmp.path().join("m.d1.civec");
        VecIndex::create_segment(&seg, dim, rows.clone()).unwrap();
        let idx = VecIndex::open_segment(&seg).unwrap();
        assert_eq!(idx.repo_runs().len(), 12);
        assert_eq!(idx.repo_ranges("b").unwrap().len(), 4);
        assert!(idx.repo_ranges("zzz").is_none());
        let dead = BTreeSet::from([7u32, 51, 52]);
        for _ in 0..10 {
            let q = norm((0..dim).map(|_| rng.next_f32()).collect());
            for repo in ["a", "b", "c"] {
                let scoped = idx.search_in_repo(&q, 10, &dead, repo);
                let filtered = idx.search_where(&q, 10, &dead, &|r| idx.row_meta(r).repo == repo);
                assert_eq!(scoped, filtered, "repo {repo}");
                assert!(scoped.iter().all(|(r, _)| idx.row_meta(*r).repo == repo && !dead.contains(r)));
            }
        }
        let set = VecSet::open(tmp.path(), "m").unwrap().unwrap();
        let q = norm((0..dim).map(|_| rng.next_f32()).collect());
        assert_eq!(set.search_repo(&q, 5, "c"), set.search_where(&q, 5, &|m| m.repo == "c"));
    }

    #[test]
    fn prescan_matches_bruteforce_topk() {
        let tmp = tempfile::tempdir().unwrap();
        let (dim, n, k) = (64usize, 200usize, 7usize);
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let rows: Vec<(VecRowMeta, Vec<f32>)> = (0..n)
            .map(|i| {
                (
                    meta("r", &format!("f{i}.rs")),
                    norm((0..dim).map(|_| rng.next_f32()).collect()),
                )
            })
            .collect();
        let q = norm((0..dim).map(|_| rng.next_f32()).collect());

        let idx = VecIndex::create(tmp.path(), "hash-v1", dim, rows.clone()).unwrap();
        let hits = idx.search(&q, k);

        // Brute-force reference over the same planted vectors.
        let mut brute: Vec<(u32, f32)> = rows
            .iter()
            .enumerate()
            .map(|(i, (_, v))| (i as u32, dot(&q, v)))
            .collect();
        brute.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        brute.truncate(k);

        assert_eq!(
            hits, brute,
            "binary prescan must equal exact brute-force top-k on small index"
        );
    }

    /// SPEC-P3 §1: below-threshold create writes NO .cihnsw and uses the
    /// flat BQ path (result must equal brute-force top-k).
    #[test]
    fn below_threshold_no_cihnsw_flat_path() {
        let tmp = tempfile::tempdir().unwrap();
        // 200 rows < BQ prescan pool (256), so the flat path is exact here
        // (same regime as prescan_matches_bruteforce_topk).
        let (dim, n, k) = (32usize, 200usize, 8usize);
        let mut rng = Rng(0x123456789ABCDEF);
        let rows: Vec<(VecRowMeta, Vec<f32>)> = (0..n)
            .map(|i| {
                (
                    meta("r", &format!("f{i}.rs")),
                    norm((0..dim).map(|_| rng.next_f32()).collect()),
                )
            })
            .collect();
        let q = norm((0..dim).map(|_| rng.next_f32()).collect());

        // Default threshold is 1,000,000 -> no graph for 300 rows.
        let idx = VecIndex::create(tmp.path(), "hash-v1", dim, rows.clone()).unwrap();
        assert!(
            !tmp.path().join("hash-v1.cihnsw").exists(),
            "below-threshold create must not write .cihnsw"
        );
        assert!(idx.hnsw.is_none(), "flat path expected");

        let hits = idx.search(&q, k);
        let mut brute: Vec<(u32, f32)> = rows
            .iter()
            .enumerate()
            .map(|(i, (_, v))| (i as u32, dot(&q, v)))
            .collect();
        brute.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        brute.truncate(k);
        assert_eq!(hits, brute, "flat path must match brute force");

        // Reopen: still no graph, same results.
        let idx2 = VecIndex::open(tmp.path(), "hash-v1").unwrap().unwrap();
        assert!(idx2.hnsw.is_none());
        assert_eq!(idx2.search(&q, k), brute);
    }

    /// SPEC-P3 §1: above-threshold create writes .cihnsw and the HNSW path
    /// ranks a planted nearest neighbor #1 (in-memory and after reopen).
    #[test]
    fn above_threshold_cihnsw_planted_nn() {
        let tmp = tempfile::tempdir().unwrap();
        let (dim, n) = (64usize, 300usize);
        let opts = IndexOptions {
            hnsw_threshold: 100, // lowered so 300 rows trigger the graph
        };
        let mut rng = Rng(0xFEEDFACE1234567);
        let mut rows: Vec<(VecRowMeta, Vec<f32>)> = (0..n)
            .map(|i| {
                (
                    meta("r", &format!("f{i}.rs")),
                    norm((0..dim).map(|_| rng.next_f32()).collect()),
                )
            })
            .collect();
        // Plant a near-duplicate of the query at row 137.
        let q = norm((0..dim).map(|_| rng.next_f32()).collect());
        let planted = norm(
            q.iter()
                .map(|&x| x + 0.001 * rng.next_f32())
                .collect(),
        );
        rows[137] = (meta("r", "planted.rs"), planted);

        let idx =
            VecIndex::create_with_options(tmp.path(), "hash-v1", dim, rows.clone(), &opts)
                .unwrap();
        assert!(
            tmp.path().join("hash-v1.cihnsw").exists(),
            "above-threshold create must write .cihnsw"
        );
        assert!(idx.hnsw.is_some(), "HNSW path expected");

        let hits = idx.search(&q, 10);
        assert_eq!(hits.len(), 10);
        assert_eq!(hits[0].0, 137, "planted nearest neighbor must rank #1");
        assert_eq!(idx.row_meta(hits[0].0).path, "planted.rs");

        // Reopen: graph loads from disk, same winner.
        let idx2 = VecIndex::open(tmp.path(), "hash-v1").unwrap().unwrap();
        assert!(idx2.hnsw.is_some(), "open must load the .cihnsw graph");
        let hits2 = idx2.search(&q, 10);
        assert_eq!(hits2[0].0, 137, "planted NN must rank #1 after reopen");
        assert_eq!(hits, hits2, "in-memory and reloaded graph must agree");

        // Recall sanity on the HNSW path: top-10 should match brute force
        // almost exactly on this small deterministic fixture.
        let mut brute: Vec<(u32, f32)> = rows
            .iter()
            .enumerate()
            .map(|(i, (_, v))| (i as u32, dot(&q, v)))
            .collect();
        brute.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        brute.truncate(10);
        let truth: std::collections::BTreeSet<u32> = brute.iter().map(|h| h.0).collect();
        let got: std::collections::BTreeSet<u32> = hits2.iter().map(|h| h.0).collect();
        let recall = got.intersection(&truth).count() as f64 / 10.0;
        assert!(recall >= 0.9, "HNSW path recall {recall} too low on fixture");
    }

    /// SPEC-P3 §1: tombstoned rows are never returned on the HNSW path.
    #[test]
    fn hnsw_path_excludes_tombstones() {
        let tmp = tempfile::tempdir().unwrap();
        let (dim, n) = (64usize, 200usize);
        let opts = IndexOptions {
            hnsw_threshold: 100,
        };
        let mut rng = Rng(0x0BAD5EED0BAD5EED);
        let mut rows: Vec<(VecRowMeta, Vec<f32>)> = (0..n)
            .map(|i| {
                (
                    meta("r", &format!("f{i}.rs")),
                    norm((0..dim).map(|_| rng.next_f32()).collect()),
                )
            })
            .collect();
        // Plant the two closest vectors to the query; the closest is droppable.
        let q = norm((0..dim).map(|_| rng.next_f32()).collect());
        rows[7] = (
            meta("drop", "nearest.rs"),
            norm(q.iter().map(|&x| x + 0.0005 * rng.next_f32()).collect()),
        );
        rows[42] = (
            meta("keep", "second.rs"),
            norm(q.iter().map(|&x| x + 0.002 * rng.next_f32()).collect()),
        );

        let mut idx =
            VecIndex::create_with_options(tmp.path(), "hash-v1", dim, rows, &opts).unwrap();
        assert!(idx.hnsw.is_some());
        assert_eq!(idx.search(&q, 5)[0].0, 7, "setup: row 7 is the NN");

        let deleted = idx.delete_where(|m| m.repo == "drop");
        assert_eq!(deleted, 1);
        let hits = idx.search(&q, 5);
        assert!(
            hits.iter().all(|(r, _)| *r != 7),
            "tombstoned row must never be returned via HNSW path"
        );
        assert_eq!(hits[0].0, 42, "second-planted row becomes #1");

        // Tombstones persist; HNSW path still excludes the row after reopen.
        idx.save_tombstones().unwrap();
        drop(idx);
        let idx2 = VecIndex::open(tmp.path(), "hash-v1").unwrap().unwrap();
        assert!(idx2.hnsw.is_some());
        let hits2 = idx2.search(&q, 5);
        assert!(hits2.iter().all(|(r, _)| *r != 7));
        assert_eq!(hits2[0].0, 42);
    }

    /// Shrinking a rebuild below the threshold removes a stale .cihnsw.
    #[test]
    fn rebuild_below_threshold_removes_stale_cihnsw() {
        let tmp = tempfile::tempdir().unwrap();
        let dim = 16usize;
        let opts = IndexOptions {
            hnsw_threshold: 100,
        };
        let mut rng = Rng(0x7777);
        let mkrows = |n: usize, rng: &mut Rng| -> Vec<(VecRowMeta, Vec<f32>)> {
            (0..n)
                .map(|i| {
                    (
                        meta("r", &format!("f{i}.rs")),
                        norm((0..dim).map(|_| rng.next_f32()).collect()),
                    )
                })
                .collect()
        };

        let idx =
            VecIndex::create_with_options(tmp.path(), "hash-v1", dim, mkrows(150, &mut rng), &opts)
                .unwrap();
        assert!(idx.hnsw.is_some());
        assert!(tmp.path().join("hash-v1.cihnsw").exists());
        drop(idx);

        // Rebuild with fewer rows: sidecar must be gone and results exact.
        let rows = mkrows(50, &mut rng);
        let idx =
            VecIndex::create_with_options(tmp.path(), "hash-v1", dim, rows.clone(), &opts)
                .unwrap();
        assert!(idx.hnsw.is_none());
        assert!(
            !tmp.path().join("hash-v1.cihnsw").exists(),
            "stale .cihnsw must be removed on below-threshold rebuild"
        );
        let q = norm((0..dim).map(|_| rng.next_f32()).collect());
        let hits = idx.search(&q, 5);
        let mut brute: Vec<(u32, f32)> = rows
            .iter()
            .enumerate()
            .map(|(i, (_, v))| (i as u32, dot(&q, v)))
            .collect();
        brute.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        brute.truncate(5);
        assert_eq!(hits, brute);
    }

    /// Two servers (two opened sets) tombstone different rows of one
    /// segment: neither write may undo the other's (SPEC-P9).
    #[test]
    fn side_file_tombstones_union_across_writers() {
        let tmp = tempfile::tempdir().unwrap();
        let rows = vec![
            (meta("r", "a.rs"), norm(vec![1.0, 0.0, 0.0])),
            (meta("r", "b.rs"), norm(vec![0.0, 1.0, 0.0])),
            (meta("r", "c.rs"), norm(vec![0.0, 0.0, 1.0])),
        ];
        VecIndex::create(tmp.path(), "hash-v1", 3, rows).unwrap();
        let mut a = VecSet::open(tmp.path(), "hash-v1").unwrap().unwrap();
        let mut b = VecSet::open(tmp.path(), "hash-v1").unwrap().unwrap();
        a.delete_rows(&[0u32].into_iter().collect());
        a.save_tombstones().unwrap();
        // b's in-memory view still has row 0 live; its save must not resurrect it
        b.delete_rows(&[2u32].into_iter().collect());
        b.save_tombstones().unwrap();
        let c = VecSet::open(tmp.path(), "hash-v1").unwrap().unwrap();
        assert!(c.is_deleted(0) && c.is_deleted(2) && !c.is_deleted(1));
        // tombstone_on_disk (the non-mutating path) unions the same way
        b.tombstone_on_disk(&[1u32].into_iter().collect()).unwrap();
        let d = VecSet::open(tmp.path(), "hash-v1").unwrap().unwrap();
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn delete_where_and_persistence() {
        let tmp = tempfile::tempdir().unwrap();
        let rows = vec![
            (meta("keep", "a.rs"), norm(vec![1.0, 0.0, 0.0])),
            (meta("drop", "b.rs"), norm(vec![0.9, 0.1, 0.0])),
            (meta("drop", "c.rs"), norm(vec![0.8, 0.2, 0.0])),
            (meta("keep", "d.rs"), norm(vec![0.0, 1.0, 0.0])),
        ];
        let mut idx = VecIndex::create(tmp.path(), "hash-v1", 3, rows).unwrap();

        let q = norm(vec![1.0, 0.0, 0.0]);
        assert_eq!(idx.search(&q, 4).len(), 4);

        let deleted = idx.delete_where(|m| m.repo == "drop");
        assert_eq!(deleted, 2);
        assert_eq!(idx.len(), 2);
        // Tombstoned rows are excluded immediately (in-memory).
        let paths: Vec<&str> = idx
            .search(&q, 4)
            .iter()
            .map(|(r, _)| idx.row_meta(*r).path.as_str())
            .collect();
        assert_eq!(paths, ["a.rs", "d.rs"]);
        // Re-deleting is a no-op.
        assert_eq!(idx.delete_where(|m| m.repo == "drop"), 0);

        idx.save_tombstones().unwrap();
        drop(idx);

        // Tombstones survive reopen.
        let idx2 = VecIndex::open(tmp.path(), "hash-v1").unwrap().unwrap();
        assert_eq!(idx2.len(), 2);
        let paths2: Vec<&str> = idx2
            .search(&q, 4)
            .iter()
            .map(|(r, _)| idx2.row_meta(*r).path.as_str())
            .collect();
        assert_eq!(paths2, ["a.rs", "d.rs"]);
    }
}
