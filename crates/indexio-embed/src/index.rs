//! Binary-quantized flat vector index sidecar.
//!
//! On-disk (v2): `<dir>/<model_id>.civec`
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
//! Newer segments are v3 (`CIVEC003`): the same header and
//! meta block, then `n_rows` f32 per-row scales, `n_rows x dim` **int8**
//! rows (`x = q * scale`, scale = max|x| / 127 per row), the binary codes
//! and the tombstones. A row is 2 KB instead of 8 at dim 2048, the rescore
//! reads a quarter of the bytes, and the score is `scale * dot(q, row_i8)`.
//! The quantisation round-trips exactly through a compaction (the maximum
//! component maps to 127, every other one to the same integer again). v2
//! files stay readable; `create_with_options` (the HNSW full build) still
//! writes v2 because the graph walks f32 rows.
//!
//! Opening a v2 or v3 file memory-maps it: the vector block is never
//! copied and the row metadata is read where it lies (`rowmeta`: one walk
//! of the meta block at open, four heap bytes per row afterwards), so a
//! multi-GB index costs a few MB of heap and a query touches just the
//! binary codes plus the rescored candidates. The mapping is the crate's
//! single `unsafe` site (`mapped`), under the same policy as indexio-index
//! shards: the file is immutable after its atomic rename.
//!
//! Tombstones are a bincode sorted `Vec<u32>` rather than a
//! `RoaringBitmap` to avoid a new dependency; the `.civec` file is a
//! rebuildable sidecar, so the encoding is internal to this crate.
//!
//! Query: hamming-distance prescan on the binary codes keeps the top
//! `max(k*8, 256)` candidates (capped at live rows), rescores with exact
//! f32 dot, returns top k. Tombstoned rows are skipped.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, ensure, Context};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::bm25::Bytes;
use crate::embed::dot;
use crate::hnsw::{Hnsw, HnswOptions};
use crate::rowmeta::{BlockError, MetaStore};

pub use crate::rowmeta::RowMeta;

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

/// Options for [`VecIndex::create_with_options`] (additive).
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
    use std::sync::Arc;

    use memmap2::Mmap;

    /// Map a `.civec` file read-only. The mapping is shared by the vector
    /// blocks and the row metadata of the segment.
    ///
    /// # Safety (why this is sound)
    /// `memmap2` is unsafe because another process could mutate the file
    /// under the mapping. `.civec` files are written to a temp path and
    /// atomically renamed into place; they are never modified in place
    /// (tombstones go to a side file), so the mapping cannot be invalidated
    /// by this crate's own writers.
    pub fn map_file(file: &File) -> io::Result<Arc<Mmap>> {
        Ok(Arc::new(unsafe { Mmap::map(file)? }))
    }

    /// The vector and code blocks of a mapped v2 `.civec`.
    pub struct MappedVectors {
        map: Arc<Mmap>,
        vec_off: usize,
        n_f32: usize,
        code_off: usize,
        code_len: usize,
    }

    impl MappedVectors {
        /// Validate that `[vec_off, vec_off + 4*n_f32)` and
        /// `[code_off, code_off + code_len)` lie inside `map` and that the
        /// vector block is 4-byte aligned in memory.
        pub fn new(
            map: Arc<Mmap>,
            vec_off: usize,
            n_f32: usize,
            code_off: usize,
            code_len: usize,
        ) -> io::Result<Self> {
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

        /// The vector block as `&[f32]` (little-endian file on a
        /// little-endian host; alignment checked in `new`).
        #[cfg(target_endian = "little")]
        pub fn vectors(&self) -> &[f32] {
            let ptr = self.map[self.vec_off..].as_ptr() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, self.n_f32) }
        }

        pub fn codes(&self) -> &[u8] {
            &self.map[self.code_off..self.code_off + self.code_len]
        }
    }

    /// The blocks of a mapped v3 `.civec`: per-row scales, int8 rows and
    /// binary codes.
    pub struct MappedQuant {
        map: Arc<Mmap>,
        scale_off: usize,
        q_off: usize,
        n_rows: usize,
        dim: usize,
        code_off: usize,
        code_len: usize,
    }

    impl MappedQuant {
        pub fn new(
            map: Arc<Mmap>,
            scale_off: usize,
            q_off: usize,
            n_rows: usize,
            dim: usize,
            code_off: usize,
            code_len: usize,
        ) -> io::Result<Self> {
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

use mapped::{map_file, MappedQuant, MappedVectors};

/// Where the vector + code blocks live.
enum Storage {
    Owned { vectors: Vec<f32>, codes: Vec<u8> },
    Mapped(MappedVectors),
    /// v3: int8 rows + per-row scales.
    Quant(MappedQuant),
}

/// Flat binary-quantized vector index with exact rescore.
pub struct VecIndex {
    path: PathBuf,
    dim: usize,
    /// Row metadata (read in place for a mapped segment) with the runs of
    /// consecutive rows of one file and of one repo: a compaction writes
    /// rows grouped by repo, so a repo-scoped search scans one run instead
    /// of testing every row.
    metas: MetaStore,
    /// n_rows * dim f32 (row-major) and n_rows * ceil(dim/8) codes.
    store: Storage,
    tombstones: BTreeSet<u32>,
    /// HNSW ANN graph; `None` below threshold / when absent.
    hnsw: Option<Hnsw>,
}

/// The row metadata of a mapped segment: read in place from
/// `map[at..at + len]`, or (a block too large for `u32` record offsets)
/// deserialized as before. Errors are the ones the deserializer gave.
fn open_metas(map: &Arc<Mmap>, at: usize, len: usize, n_rows: usize, path: &Path) -> anyhow::Result<MetaStore> {
    let bad = || anyhow!("{}: corrupt .civec file", path.display());
    let end = at.checked_add(len).filter(|&e| e <= map.len()).ok_or_else(bad)?;
    match MetaStore::in_place(Bytes::Mapped { map: Arc::clone(map), start: at, end }, n_rows) {
        Ok(m) => Ok(m),
        Err(BlockError::Corrupt) => Err(bad()),
        Err(BlockError::Count(_)) => Err(anyhow!("meta count != n_rows")),
        Err(BlockError::TooLarge) => {
            let metas: Vec<VecRowMeta> = bincode::deserialize(&map[at..end]).map_err(|_| bad())?;
            ensure!(metas.len() == n_rows, "meta count != n_rows");
            Ok(MetaStore::owned(metas))
        }
    }
}

impl VecIndex {
    pub(crate) fn file_path(dir: &Path, model_id: &str) -> PathBuf {
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

    /// Full build with options. When
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
            metas: MetaStore::owned(metas),
            store: Storage::Owned { vectors, codes },
            tombstones: BTreeSet::new(),
            hnsw: None,
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
        // Load the HNSW sidecar if present. It is rebuildable,
        // so a corrupt or stale (row-count mismatch) graph is ignored with a
        // warning rather than failing the open — search falls back to the
        // flat BQ path.
        let hpath = Self::hnsw_path(dir, model_id);
        idx.hnsw = if hpath.exists() {
            match Hnsw::load(&hpath) {
                Ok(h) if h.n_nodes() == idx.raw_len() => Some(h),
                Ok(h) => {
                    tracing::warn!(
                        "{}: stale .cihnsw ({} nodes, {} rows) — ignoring",
                        hpath.display(),
                        h.n_nodes(),
                        idx.raw_len()
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
        let map = map_file(&file).with_context(|| format!("map {}", path.display()))?;
        let mapped = MappedVectors::new(Arc::clone(&map), vec_off, n_f32, code_off, code_len)
            .with_context(|| format!("map {}", path.display()))?;
        let metas = open_metas(&map, header_len, meta_len, n_rows, &path)?;
        let bytes = &map[..];
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
        })
    }

    /// v3: header | metas | pad | scales f32 | rows i8 | codes | tombs.
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
        let map = map_file(&file).with_context(|| format!("map {}", path.display()))?;
        let mapped = MappedQuant::new(Arc::clone(&map), scale_off, q_off, n_rows, dim, code_off, code_len)
            .with_context(|| format!("map {}", path.display()))?;
        let metas = open_metas(&map, header_len, meta_len, n_rows, &path)?;
        let bytes = &map[..];
        let mut off = code_off + code_len;
        let tomb_len = u64::from_le_bytes(bytes.get(off..off + 8).ok_or_else(bad)?.try_into().unwrap()) as usize;
        off += 8;
        let tombs: Vec<u32> = bincode::deserialize(bytes.get(off..off + tomb_len).ok_or_else(bad)?).map_err(|_| bad())?;
        let mut tombstones: BTreeSet<u32> = tombs.into_iter().collect();
        ensure!(tombstones.len() == n_deleted, "n_deleted header != tombstone count");
        tombstones.extend(read_tomb_side(&path));
        Ok(VecIndex { path, dim, metas, store: Storage::Quant(mapped), tombstones, hnsw: None })
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
            // A/B knob: the older f32 layout (v2)
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
                metas: MetaStore::owned(metas),
                store: Storage::Owned { vectors, codes },
                tombstones: BTreeSet::new(),
                hnsw: None,
            };
            idx.write_file()?;
            let side = tomb_side_path(path);
            if side.exists() {
                fs::remove_file(&side).ok();
            }
            return Ok(());
        }
        // v3: int8 rows with a per-row scale
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
            metas: MetaStore::owned(metas),
            store: Storage::Owned { vectors, codes },
            tombstones,
            hnsw: None,
        })
    }

    /// Serializes the full current state (rows + tombstones) as v2 via
    /// tmp+rename. Callers with a mapped store must drop the mapping first
    /// (`save_tombstones` does) — Windows refuses to replace a mapped file.
    fn write_file(&self) -> anyhow::Result<()> {
        let metas = self.metas.encode()?;
        let tombs: Vec<u32> = self.tombstones.iter().copied().collect();
        let tombs = bincode::serialize(&tombs)?;
        let vectors = self.vectors().ok_or_else(|| anyhow!("int8 segments are written by create_segment"))?;
        let codes = self.codes();
        let header_len = 8 + 4 + 4 + 4 + 8;
        let vec_off = align8(header_len + metas.len());

        let mut out = Vec::with_capacity(vec_off + vectors.len() * 4 + codes.len() + 8 + tombs.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.raw_len() as u32).to_le_bytes());
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
    /// Two candidate paths:
    /// - HNSW present: approximate `hnsw.search` with `ef = max(k*10, 100)`,
    ///   then exact f32 rescore of candidates, skipping tombstoned rows.
    /// - Otherwise: binary prescan keeps `max(k*8, 256)` candidates (capped
    ///   at live rows), rescored exactly.
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(u32, f32)> {
        self.search_excluding(q, k, &BTreeSet::new())
    }

    /// [`search`](Self::search) that also skips the rows in `dead` — the
    /// tombstones a [`VecSet`] keeps outside the shared segment.
    pub fn search_excluding(&self, q: &[f32], k: usize, dead: &BTreeSet<u32>) -> Vec<(u32, f32)> {
        self.search_where(q, k, dead, &|_| true)
    }

    /// Runs of consecutive rows of one repo, in row order: `(repo, rows)`.
    /// Found when the segment was opened; nothing is scanned here.
    pub fn repo_runs(&self) -> impl Iterator<Item = (&str, Range<u32>)> + '_ {
        self.metas.repo_runs().map(|rows| (self.metas.row(rows.start).repo, rows))
    }

    /// Runs of consecutive rows of one file inside `rows`, in row order:
    /// the metadata of the run's first row (its `repo` and `path` are the
    /// file's) and the run, clipped to `rows`. Writers lay rows out file by
    /// file, so this visits each file of the range once, not each row.
    pub fn file_runs(&self, rows: Range<u32>) -> impl Iterator<Item = (RowMeta<'_>, Range<u32>)> + '_ {
        self.metas.file_runs(rows).map(|run| (self.metas.row(run.start), run))
    }

    /// Rows of `repo` as runs; `None` when the repo has no rows here.
    fn repo_ranges(&self, repo: &str) -> Option<Vec<(u32, u32)>> {
        let v: Vec<(u32, u32)> = self.repo_runs().filter(|(r, _)| *r == repo).map(|(_, rows)| (rows.start, rows.end)).collect();
        (!v.is_empty()).then_some(v)
    }

    /// [`search_excluding`](Self::search_excluding) over the rows of `repo`
    /// only: the prescan walks that repo's row runs — one
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
    /// which `keep(row)` holds (a repo-scoped semantic leg — the
    /// prescan pool is then made of matching rows only, instead of a
    /// post-filter over a pool that a large corpus fills with other repos).
    pub fn search_where(&self, q: &[f32], k: usize, dead: &BTreeSet<u32>, keep: &(dyn Fn(u32) -> bool + Sync)) -> Vec<(u32, f32)> {
        let n = self.raw_len();
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

    /// Metadata of `row`, borrowed from the segment (decoded from the
    /// mapping on each call: keep it out of loops over every row, see
    /// [`file_runs`](Self::file_runs) and [`row_hash`](Self::row_hash)).
    pub fn row_meta(&self, row: u32) -> RowMeta<'_> {
        self.metas.row(row)
    }

    /// The chunk hash of `row`, without decoding the rest of its metadata.
    pub fn row_hash(&self, row: u32) -> [u8; 16] {
        self.metas.hash(row)
    }

    /// Rows tombstoned inside the file or its side file at open time.
    pub fn tombstones(&self) -> &BTreeSet<u32> {
        &self.tombstones
    }

    /// Tombstone every row matching `pred`. Returns rows newly deleted.
    pub fn delete_where(&mut self, pred: impl Fn(&RowMeta<'_>) -> bool) -> u64 {
        let mut n = 0u64;
        for row in 0..self.raw_len() as u32 {
            if !self.tombstones.contains(&row) && pred(&self.metas.row(row)) {
                self.tombstones.insert(row);
                n += 1;
            }
        }
        n
    }

    /// Persists tombstones: the full set goes to the small
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

/// `<segment>.tomb`: bincode `Vec<u32>` of tombstoned rows.
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
// VecSet: the segmented vector index
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
    set_stamp_paths(&segment_paths(dir, model_id))
}

/// [`set_stamp`] over an explicit segment list (the committed pairs of
/// `pipeline::plane_paths`).
pub fn set_stamp_paths(paths: &[PathBuf]) -> Option<(u64, Option<std::time::SystemTime>)> {
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

/// The segments this process has open, by path.
///
/// The writers open the plane on their own (`pipeline::open_sets`: every
/// incremental embed, the coverage repair, a compaction, `index_stats`)
/// next to a query engine that already holds each of its segments, and
/// opening an immutable file a second time bought nothing. A set being
/// opened takes a segment from here when the file at that path still has
/// the length and mtime the segment was opened at: the rule `reopen_paths`
/// applies to the caller's previous set, extended to every set alive in
/// the process.
///
/// Only the immutable part is shared. The segment list is listed and the
/// tombstone side files are read on every open, so a set is as fresh as
/// one opened from nothing; a writer still sees every row another server
/// tombstoned or appended. Entries are weak: a segment is unmapped when the
/// last set holding it goes, exactly as before.
pub(crate) struct OpenSegments<T> {
    open: std::sync::Mutex<Vec<(PathBuf, (u64, Option<std::time::SystemTime>), std::sync::Weak<T>)>>,
}

impl<T> OpenSegments<T> {
    pub(crate) const fn new() -> Self {
        OpenSegments { open: std::sync::Mutex::new(Vec::new()) }
    }

    /// The open segment of the file at `path`, if it was opened at `stamp`.
    /// A file that could not be stat'ed matches nothing.
    pub(crate) fn get(&self, path: &Path, stamp: (u64, Option<std::time::SystemTime>)) -> Option<Arc<T>> {
        stamp.1?;
        let open = self.open.lock().unwrap_or_else(|e| e.into_inner());
        open.iter().find(|(p, st, _)| p == path && *st == stamp).and_then(|(_, _, seg)| seg.upgrade())
    }

    /// Record `seg`, just opened from `path` at `stamp`; an older segment
    /// of that path and entries whose segment is gone are dropped.
    pub(crate) fn put(&self, path: &Path, stamp: (u64, Option<std::time::SystemTime>), seg: &Arc<T>) {
        if stamp.1.is_none() {
            return;
        }
        let mut open = self.open.lock().unwrap_or_else(|e| e.into_inner());
        open.retain(|(p, _, s)| s.strong_count() > 0 && p != path);
        open.push((path.to_path_buf(), stamp, Arc::downgrade(seg)));
    }
}

static OPEN_SEGMENTS: OpenSegments<VecIndex> = OpenSegments::new();

impl VecSet {
    /// Open every segment; `Ok(None)` when the model has no segment yet.
    pub fn open(dir: &Path, model_id: &str) -> anyhow::Result<Option<Self>> {
        Self::reopen(None, dir, model_id)
    }

    /// Open the current segment list, reusing `prev`'s parsed segments
    /// whose files are byte-identical (same length and mtime), then those
    /// of any other set open in this process ([`OpenSegments`]).
    pub fn reopen(prev: Option<&VecSet>, dir: &Path, model_id: &str) -> anyhow::Result<Option<Self>> {
        Self::reopen_paths(prev, dir, model_id, &segment_paths(dir, model_id))
    }

    /// [`reopen`](Self::reopen) over an explicit segment list.
    pub fn reopen_paths(
        prev: Option<&VecSet>,
        dir: &Path,
        model_id: &str,
        paths: &[PathBuf],
    ) -> anyhow::Result<Option<Self>> {
        if paths.is_empty() {
            return Ok(None);
        }
        let mut segs = Vec::with_capacity(paths.len());
        let mut stamps = Vec::with_capacity(paths.len());
        let mut tombs = Vec::with_capacity(paths.len());
        for (i, p) in paths.iter().enumerate() {
            let stamp = file_stamp(p);
            let reused = prev
                .and_then(|pv| {
                    pv.segs
                        .iter()
                        .zip(&pv.stamps)
                        .find(|(s, st)| s.path() == p.as_path() && **st == stamp)
                        .map(|(s, _)| std::sync::Arc::clone(s))
                })
                .or_else(|| OPEN_SEGMENTS.get(p, stamp));
            let seg = match reused {
                Some(s) => s,
                None => {
                    let seg = std::sync::Arc::new(if i == 0 && p == &VecIndex::file_path(dir, model_id) {
                        // the legacy base may carry an HNSW sidecar
                        VecIndex::open(dir, model_id)?.ok_or_else(|| anyhow!("{} vanished", p.display()))?
                    } else {
                        VecIndex::open_segment(p)?
                    });
                    OPEN_SEGMENTS.put(p, stamp, &seg);
                    seg
                }
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

    /// Global id of segment `seg`'s first row.
    pub fn offset(&self, seg: usize) -> u32 {
        self.offsets[seg]
    }

    /// Tombstoned local rows of segment `seg` (in-file ∪ side file).
    pub fn dead(&self, seg: usize) -> &BTreeSet<u32> {
        &self.tombs[seg]
    }

    /// Metadata of a global row, borrowed from its segment.
    pub fn row_meta(&self, row: u32) -> RowMeta<'_> {
        let (s, r) = self.locate(row);
        self.segs[s].row_meta(r)
    }

    /// `(repo, path)` of every file with a live row, from the segments'
    /// file runs: a file is visited once per run, not once per row (a file
    /// with rows in two segments comes up twice).
    pub fn live_files(&self) -> impl Iterator<Item = (&str, &str)> + '_ {
        self.segs.iter().zip(&self.tombs).flat_map(|(seg, dead)| {
            seg.file_runs(0..seg.raw_len() as u32)
                .filter(move |(_, rows)| dead.range(rows.clone()).count() < rows.len())
                .map(|(m, _)| (m.repo, m.path))
        })
    }

    /// Global rows of `repo` as ascending, disjoint ranges (tombstoned
    /// rows included): what a per-row `row_meta(g).repo == repo` test
    /// selects, without decoding a row.
    pub fn repo_rows(&self, repo: &str) -> Vec<Range<u32>> {
        let mut out: Vec<Range<u32>> = Vec::new();
        for (seg, &off) in self.segs.iter().zip(&self.offsets) {
            out.extend(seg.repo_runs().filter(|(r, _)| *r == repo).map(|(_, rows)| off + rows.start..off + rows.end));
        }
        out
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
        let mut out: Vec<(u32, f32)> = Vec::new();
        for (i, s) in self.segs.iter().enumerate() {
            for (r, score) in s.search_excluding(q, k, &self.tombs[i]) {
                out.push((self.offsets[i] + r, score));
            }
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(k);
        out
    }

    /// [`search`](Self::search) over the rows of `repo` only, walking each
    /// segment's runs of that repo.
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
    /// Every row's metadata is decoded for the test: for a repo filter use
    /// [`search_repo`](Self::search_repo), which walks that repo's runs.
    pub fn search_where(&self, q: &[f32], k: usize, keep: &(dyn Fn(&RowMeta<'_>) -> bool + Sync)) -> Vec<(u32, f32)> {
        let mut out: Vec<(u32, f32)> = Vec::new();
        for (i, s) in self.segs.iter().enumerate() {
            let pred = |r: u32| keep(&s.row_meta(r));
            for (r, score) in s.search_where(q, k, &self.tombs[i], &pred) {
                out.push((self.offsets[i] + r, score));
            }
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(k);
        out
    }

    /// Tombstone every live row matching `pred`; returns rows newly deleted.
    pub fn delete_where(&mut self, pred: impl Fn(&RowMeta<'_>) -> bool) -> u64 {
        let mut n = 0u64;
        for (i, s) in self.segs.iter().enumerate() {
            for r in 0..s.raw_len() as u32 {
                if !self.tombs[i].contains(&r) && pred(&s.row_meta(r)) {
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
            match p.extension().and_then(|e| e.to_str()) {
                Some(STALE_EXT) => {
                    fs::remove_file(&p).ok();
                }
                // `.civec.tmp` / `.cibm25.tmp` of a writer that died mid-write
                Some("tmp") if indexio_index::is_abandoned(&p) => {
                    fs::remove_file(&p).ok();
                }
                _ => {}
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
        let got: Vec<&str> = hits.iter().map(|(r, _)| idx.row_meta(*r).path).collect();
        assert_eq!(got, expected, "planted nearest neighbor must rank first");
        assert!(hits[0].1 > hits[1].1 && hits[1].1 > hits[2].1);

        // Reopen from disk and repeat.
        let idx2 = VecIndex::open(tmp.path(), "hash-v1").unwrap().unwrap();
        assert_eq!(idx2.len(), 4);
        let hits2 = idx2.search(&q, 3);
        let got2: Vec<&str> =
            hits2.iter().map(|(r, _)| idx2.row_meta(*r).path).collect();
        assert_eq!(got2, expected, "ordering must survive reopen");

        // Absent index -> Ok(None); k > n -> n results.
        assert!(VecIndex::open(tmp.path(), "no-such-model").unwrap().is_none());
        assert_eq!(idx2.search(&q, 100).len(), 4);
        assert!(idx2.search(&q, 0).is_empty());
    }

    #[test]
    fn int8_segment_matches_f32_ranking() {
        // a v3 segment (int8 rows) ranks like the f32 rows it was
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
        let carried: Vec<(VecRowMeta, Vec<f32>)> = (0..n as u32).map(|r| (idx.row_meta(r).to_owned_meta(), idx.row_vector(r))).collect();
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
        assert_eq!(idx.repo_runs().count(), 12);
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

    /// below-threshold create writes NO .cihnsw and uses the
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

    /// above-threshold create writes .cihnsw and the HNSW path
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

    /// tombstoned rows are never returned on the HNSW path.
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
    /// segment: neither write may undo the other's.
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
            .map(|(r, _)| idx.row_meta(*r).path)
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
            .map(|(r, _)| idx2.row_meta(*r).path)
            .collect();
        assert_eq!(paths2, ["a.rs", "d.rs"]);
    }

    /// Rows the way writers lay them out (file by file) with empty and
    /// non-ASCII repo and path strings.
    fn varied_rows(n: usize, dim: usize, seed: u64) -> Vec<(VecRowMeta, Vec<f32>)> {
        const REPOS: [&str; 4] = ["alpha", "", "dépôt", "仓库"];
        const PATHS: [&str; 5] = ["src/lib.rs", "", "docs/überblick.md", "数据/模型.py", "🦀 dir/a b.rs"];
        let mut rng = Rng(seed);
        let mut out = Vec::with_capacity(n);
        let mut file = 0usize;
        while out.len() < n {
            let repo = REPOS[(file / 7) % REPOS.len()];
            let path = format!("{}{}", PATHS[file % PATHS.len()], if file % 3 == 0 { String::new() } else { file.to_string() });
            for chunk in 0..1 + file % 4 {
                if out.len() == n {
                    break;
                }
                let v = norm((0..dim).map(|_| rng.next_f32()).collect());
                out.push((
                    VecRowMeta {
                        chunk_hash: chunk_hash(format!("{repo}/{path}#{chunk}").as_bytes()),
                        repo: repo.to_string(),
                        path: path.clone(),
                        start_line: chunk as u32 * 30 + 1,
                        end_line: chunk as u32 * 30 + 30,
                    },
                    v,
                ));
            }
            file += 1;
        }
        out
    }

    /// The metadata read in place from a mapped segment (v2 and v3) is the
    /// metadata that was written, row for row, and the run tables agree
    /// with the rows.
    #[test]
    fn mapped_row_meta_equals_what_was_written() {
        let tmp = tempfile::tempdir().unwrap();
        let rows = varied_rows(700, 16, 0xfeed);
        let want: Vec<VecRowMeta> = rows.iter().map(|(m, _)| m.clone()).collect();

        // v3 (create_segment) and v2 (the full build's write_file)
        let v3 = tmp.path().join("m.d0000000000001.civec");
        VecIndex::create_segment(&v3, 16, rows.clone()).unwrap();
        drop(VecIndex::create(tmp.path(), "v2", 16, rows).unwrap());
        let opened = [VecIndex::open_segment(&v3).unwrap(), VecIndex::open(tmp.path(), "v2").unwrap().unwrap()];
        for idx in &opened {
            assert!(idx.metas.is_in_place(), "a mapped segment keeps no owned rows");
            assert_eq!(idx.raw_len(), want.len());
            for (r, w) in want.iter().enumerate() {
                assert_eq!(idx.row_meta(r as u32), w.as_row(), "row {r}");
                assert_eq!(idx.row_hash(r as u32), w.chunk_hash);
            }
            // file runs: consecutive, covering, one (repo, path) each
            let mut next = 0u32;
            for (m, run) in idx.file_runs(0..want.len() as u32) {
                assert_eq!(run.start, next);
                for r in run.clone() {
                    assert_eq!((want[r as usize].repo.as_str(), want[r as usize].path.as_str()), (m.repo, m.path));
                }
                if let Some(after) = want.get(run.end as usize) {
                    assert!(after.repo != m.repo || after.path != m.path, "run ended inside a file");
                }
                next = run.end;
            }
            assert_eq!(next as usize, want.len());
            // repo runs
            let mut next = 0u32;
            for (repo, run) in idx.repo_runs() {
                assert_eq!(run.start, next);
                assert!(run.clone().all(|r| want[r as usize].repo == repo));
                assert!(want.get(run.end as usize).is_none_or(|m| m.repo != repo));
                next = run.end;
            }
            assert_eq!(next as usize, want.len());
        }
    }

    /// A damaged metadata block fails at open with the errors the
    /// deserializer gave; nothing is handed out from it later.
    #[test]
    fn corrupt_meta_block_is_rejected_at_open() {
        let tmp = tempfile::tempdir().unwrap();
        let rows = varied_rows(40, 8, 7);
        let good = tmp.path().join("good.civec");
        VecIndex::create_segment(&good, 8, rows).unwrap();
        let bytes = fs::read(&good).unwrap();
        let meta_len = u64::from_le_bytes(bytes[20..28].try_into().unwrap()) as usize;
        let (meta_at, meta_end) = (28usize, 28 + meta_len);
        let open_err = |name: &str, b: &[u8]| -> String {
            let p = tmp.path().join(name);
            fs::write(&p, b).unwrap();
            match VecIndex::open_segment(&p) {
                Ok(_) => panic!("{name}: a damaged segment opened"),
                Err(e) => format!("{e:#}"),
            }
        };

        // first string length runs past the block
        let mut b = bytes.clone();
        b[meta_at + 8 + 16..meta_at + 8 + 24].copy_from_slice(&(meta_len as u64 * 2).to_le_bytes());
        assert!(open_err("len.civec", &b).contains("corrupt .civec file"));

        // a string that is not UTF-8 (row 0's repo is "alpha")
        let mut b = bytes.clone();
        b[meta_at + 8 + 24] = 0xff;
        assert!(open_err("utf8.civec", &b).contains("corrupt .civec file"));

        // the last record cut short: the header's meta_len ends inside it
        let mut b = bytes.clone();
        b[20..28].copy_from_slice(&((meta_len - 5) as u64).to_le_bytes());
        assert!(open_err("cut.civec", &b).contains("corrupt .civec file"));

        // a row count the block cannot hold
        let mut b = bytes.clone();
        b[meta_at..meta_at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(open_err("count.civec", &b).contains("corrupt .civec file"));

        // a well-formed block with fewer rows than the header announces
        let fewer = bincode::serialize(&varied_rows(39, 8, 7).into_iter().map(|(m, _)| m).collect::<Vec<_>>()).unwrap();
        let mut b = bytes.clone();
        b[meta_at..meta_at + fewer.len()].copy_from_slice(&fewer);
        assert!(open_err("fewer.civec", &b).contains("meta count != n_rows"));

        // a meta_len past the end of the file
        let mut b = bytes.clone();
        b[20..28].copy_from_slice(&(bytes.len() as u64 * 4).to_le_bytes());
        let _ = open_err("past.civec", &b);

        // the file cut inside the metadata block
        let _ = open_err("short.civec", &bytes[..meta_end - 9]);

        // and the untouched bytes still open
        assert_eq!(VecIndex::open_segment(&good).unwrap().raw_len(), 40);
    }

    /// `live_files` and `repo_rows` say what a walk over every live row's
    /// metadata says.
    #[test]
    fn set_file_and_repo_views_match_a_row_walk() {
        let tmp = tempfile::tempdir().unwrap();
        for (i, seed) in [3u64, 5, 9].into_iter().enumerate() {
            VecIndex::create_segment(&tmp.path().join(format!("m.d{i:013}.civec")), 8, varied_rows(150 + 40 * i, 8, seed)).unwrap();
        }
        let mut set = VecSet::open(tmp.path(), "m").unwrap().unwrap();
        assert_eq!(set.segments().len(), 3);
        // tombstone some whole files and some single rows
        let killed = set.delete_where(|m| m.path.ends_with('7') || m.start_line == 31);
        assert!(killed > 0);

        let mut want: BTreeSet<(String, String)> = BTreeSet::new();
        for g in set.live_rows() {
            let m = set.row_meta(g);
            want.insert((m.repo.to_string(), m.path.to_string()));
        }
        let got: BTreeSet<(String, String)> = set.live_files().map(|(r, p)| (r.to_string(), p.to_string())).collect();
        assert_eq!(got, want);
        let all: BTreeSet<(String, String)> =
            (0..set.raw_len() as u32).map(|g| set.row_meta(g)).map(|m| (m.repo.to_string(), m.path.to_string())).collect();
        assert!(got.len() < all.len(), "a fully tombstoned file is not live");

        for repo in ["alpha", "", "dépôt", "仓库", "absent"] {
            let rows = set.repo_rows(repo);
            assert!(rows.windows(2).all(|w| w[0].end <= w[1].start), "ascending and disjoint");
            let inside = |g: u32| rows.iter().any(|r| r.contains(&g));
            for g in 0..set.raw_len() as u32 {
                assert_eq!(inside(g), set.row_meta(g).repo == repo, "row {g} of {repo:?}");
            }
            let q = norm(vec![0.3, -0.2, 0.9, 0.1, 0.0, 0.4, -0.7, 0.2]);
            assert_eq!(set.search_repo(&q, 9, repo), set.search_where(&q, 9, &|m| m.repo == repo));
        }
        for seg in 0..3 {
            assert_eq!(set.offset(seg), set.segments()[..seg].iter().map(|s| s.raw_len() as u32).sum::<u32>());
        }
    }
    /// Sets opened while another set holds a segment share it; what makes a
    /// set fresh (the segment list, the tombstone side files) is still read
    /// on every open.
    #[test]
    fn sets_share_open_segments_and_stay_fresh() {
        let tmp = tempfile::tempdir().unwrap();
        for i in 0..2u64 {
            VecIndex::create_segment(&tmp.path().join(format!("m.d{i:013}.civec")), 8, varied_rows(60, 8, 20 + i)).unwrap();
        }
        let open = || VecSet::open(tmp.path(), "m").unwrap().unwrap();
        let a = open();
        let b = open();
        assert!((0..2).all(|i| Arc::ptr_eq(&a.segments()[i], &b.segments()[i])), "the second set re-opened a segment");

        // rows tombstoned through one set: the next open sees them, the
        // sets opened before keep their own view
        let doomed: std::collections::HashSet<u32> = [0, 5, a.offset(1) + 2].into_iter().collect();
        a.tombstone_on_disk(&doomed).unwrap();
        let c = open();
        assert!(Arc::ptr_eq(&a.segments()[0], &c.segments()[0]));
        assert_eq!(c.len(), a.len() - 3);
        assert!(doomed.iter().all(|&g| c.is_deleted(g) && !a.is_deleted(g) && !b.is_deleted(g)));
        let q = norm(vec![0.3, -0.2, 0.9, 0.1, 0.0, 0.4, -0.7, 0.2]);
        assert!(c.search(&q, 120).iter().all(|(g, _)| !doomed.contains(g)));

        // a segment written since joins the shared ones
        VecIndex::create_segment(&tmp.path().join("m.d0000000000002.civec"), 8, varied_rows(30, 8, 33)).unwrap();
        let d = open();
        assert_eq!((d.segments().len(), d.raw_len()), (3, 150));
        assert!(Arc::ptr_eq(&a.segments()[1], &d.segments()[1]));
        assert_eq!(d.len(), 147);

        // nothing outlives the sets: the mapping goes with the last one
        let seg0 = a.segments()[0].path().to_path_buf();
        let stamp = file_stamp(&seg0);
        assert!(OPEN_SEGMENTS.get(&seg0, stamp).is_some());
        drop((a, b, c, d));
        assert!(OPEN_SEGMENTS.get(&seg0, stamp).is_none());
        assert_eq!(open().len(), 147);
    }

    #[test]
    fn open_segments_match_on_path_and_stamp() {
        let reg: OpenSegments<String> = OpenSegments::new();
        let t = std::time::SystemTime::now();
        let later = t + std::time::Duration::from_secs(1);
        let (p, q) = (Path::new("a.civec"), Path::new("b.civec"));
        let seg = Arc::new("first".to_string());
        reg.put(p, (10, Some(t)), &seg);
        assert!(reg.get(p, (10, Some(t))).is_some_and(|s| Arc::ptr_eq(&s, &seg)));
        assert!(reg.get(p, (11, Some(t))).is_none(), "another length is another file");
        assert!(reg.get(p, (10, Some(later))).is_none(), "another mtime is another file");
        assert!(reg.get(q, (10, Some(t))).is_none());
        // a file that could not be stat'ed is neither recorded nor matched
        reg.put(q, (0, None), &seg);
        assert!(reg.get(q, (0, None)).is_none() && reg.get(p, (10, None)).is_none());
        // a rewritten file replaces its entry
        let newer = Arc::new("second".to_string());
        reg.put(p, (12, Some(later)), &newer);
        assert!(reg.get(p, (10, Some(t))).is_none());
        assert!(reg.get(p, (12, Some(later))).is_some_and(|s| Arc::ptr_eq(&s, &newer)));
        // entries are weak, and dead ones are dropped on the next insert
        drop(newer);
        assert!(reg.get(p, (12, Some(later))).is_none());
        reg.put(q, (1, Some(t)), &seg);
        assert_eq!(reg.open.lock().unwrap().len(), 1);
    }
}
