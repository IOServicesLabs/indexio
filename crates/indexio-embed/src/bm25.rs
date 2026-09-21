//! Chunk-level BM25F sidecar (SPEC-P5 §A1), aligned row-for-row with the
//! vector index of the same namespace (`<dir>/<ns>.cibm25` next to
//! `<data_dir>/vec/<model_id>.civec`; the pipeline passes
//! `<data_dir>/bm25`).
//!
//! Two fields per chunk: **header** (path + scope line, weight
//! [`HEADER_WEIGHT`]) and **body** (weight 1.0). Tokenizer: the shared
//! [`crate::rindex::tokenize`] (identifier-aware splitting, lowercasing);
//! adjacent-bigram terms (terms containing a space) are skipped — bigram
//! boost scoring is a SPEC-P5 non-goal.
//!
//! On-disk format:
//! ```text
//! magic "CIBM25 1" (8B) | u32 n_rows | u32 n_terms (df-table size)
//! | u32 avg_body_len | u32 avg_header_len (rounded token counts)
//! | u64 fst_len | FST bytes (term -> postings offset)
//! | u64 postings_len | postings: per term LEB128 df, then df x
//!   LEB128 (row_delta, tf_body, tf_header)
//! | u64 rowlens_len | n_rows x (u16 body_len, u8 header_len)
//! | u64 tomb_len | bincode sorted Vec<u32> tombstones
//! ```
//!
//! Documented deviations from SPEC-P5 §A1:
//! - The file is read with safe `std::fs::read` into owned memory at open
//!   instead of being mmap'd, keeping this crate `#![forbid(unsafe_code)]`
//!   (same deviation as the `.civec` index; memmap2 is only sanctioned in
//!   indexio-index).
//! - "u32 df-table | u32 avg fields" from the spec header sketch is realized
//!   as `n_terms` (the df table is the per-list LEB128 df prefix inside the
//!   postings block) plus `avg_body_len`/`avg_header_len`.
//! - BM25F is scored with a single weighted term frequency
//!   `tf' = tf_body + HEADER_WEIGHT * tf_header` normalized by the combined
//!   field length `dl = body_len + header_len` against the combined average
//!   (k1 = [`K1`], b = [`B`]), rather than per-field b factors — the spec
//!   fixes one b for both fields.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, ensure, Context};
use memmap2::Mmap;

/// A byte block that is either owned (a segment being built) or a range of
/// a memory-mapped file (an opened segment, SPEC-P10: the sidecar used to
/// be read and copied whole at every server start).
#[derive(Clone)]
pub enum Bytes {
    Owned(Vec<u8>),
    Mapped { map: Arc<Mmap>, start: usize, end: usize },
}

impl AsRef<[u8]> for Bytes {
    fn as_ref(&self) -> &[u8] {
        match self {
            Bytes::Owned(v) => v,
            Bytes::Mapped { map, start, end } => &map[*start..*end],
        }
    }
}

impl Bytes {
    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

use crate::embed::stopwords;
use crate::rindex::tokenize;

const MAGIC: &[u8; 8] = b"CIBM25 1";
/// BM25F parameters (SPEC-P5 §A1): short, uniform-length chunks justify the
/// low b (p5_dim01_lexical table: b in 0.3-0.4 for passage-like docs).
const K1: f64 = 1.0;
const B: f64 = 0.35;
/// Header (path + scope line) field weight.
const HEADER_WEIGHT: f64 = 3.0;
/// Hub cut: query terms with df > 40% of rows are dropped at search time.
const HUB_DF_FRAC: f64 = 0.4;

/// Chunk-level BM25F index, row-aligned with the VecIndex of the same
/// namespace. Owned in-memory representation of the `.cibm25` file.
pub struct ChunkBm25 {
    path: PathBuf,
    n_rows: u32,
    avg_body_len: f64,
    avg_header_len: f64,
    /// Term -> byte offset into `postings`.
    dict: fst::Map<Bytes>,
    /// Per term at its offset: LEB128 df, then df x (row_delta, tf_body,
    /// tf_header) LEB128 triples.
    postings: Bytes,
    /// Per row: (body_len, header_len) in unigram tokens (saturating).
    row_lens: Vec<(u16, u8)>,
    tombstones: BTreeSet<u32>,
}

fn file_path(dir: &Path, ns: &str) -> PathBuf {
    dir.join(format!("{ns}.cibm25"))
}

// ---------------------------------------------------------------------------
// LEB128
// ---------------------------------------------------------------------------

fn leb128_push(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            break;
        }
        out.push(b | 0x80);
    }
}

fn leb128_read(buf: &[u8], off: &mut usize) -> Option<u64> {
    let mut v = 0u64;
    let mut shift = 0u32;
    loop {
        let b = *buf.get(*off)?;
        *off += 1;
        v |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Some(v);
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
}

/// Decode the posting list at `off`: returns (df, [(row, tf_body,
/// tf_header)]). `None` on truncation/corruption.
fn decode_postings(buf: &[u8], mut off: usize) -> Option<(u32, Vec<(u32, u32, u32)>)> {
    let df = leb128_read(buf, &mut off)? as u32;
    let mut out = Vec::with_capacity(df as usize);
    let mut row = 0u64;
    for _ in 0..df {
        row = row.checked_add(leb128_read(buf, &mut off)?)?;
        let tf_body = leb128_read(buf, &mut off)? as u32;
        let tf_header = leb128_read(buf, &mut off)? as u32;
        out.push((u32::try_from(row).ok()?, tf_body, tf_header));
    }
    Some((df, out))
}

/// Unigram term frequencies of `text` (bigram terms skipped).
fn unigram_tf(text: &str) -> BTreeMap<String, u32> {
    let mut tf: BTreeMap<String, u32> = BTreeMap::new();
    for t in tokenize(text) {
        if t.term.contains(' ') {
            continue; // adjacent-bigram term: not indexed (SPEC-P5 non-goal)
        }
        *tf.entry(t.term).or_insert(0) += 1;
    }
    tf
}

impl ChunkBm25 {
    /// Full build: writes `<dir>/<ns>.cibm25` atomically (tmp+rename).
    ///
    /// `rows` are the (header, body) pairs of the SAME chunks, in the SAME
    /// order, as the rows passed to `VecIndex::create` in the same pipeline
    /// call — row i of this index is row i of the vector index (the
    /// indexio-query hybrid leg resolves BM25 rows through `VecIndex::row_meta`).
    pub fn create(dir: &Path, ns: &str, rows: &[(String, String)]) -> anyhow::Result<()> {
        fs::create_dir_all(dir).with_context(|| format!("create bm25 dir {}", dir.display()))?;
        Self::create_path(&file_path(dir, ns), rows)
    }

    /// [`create`](Self::create) at an explicit segment path (SPEC-P9 deltas).
    pub fn create_path(path: &Path, rows: &[(String, String)]) -> anyhow::Result<()> {
        let tf_rows: Vec<(BTreeMap<String, u32>, BTreeMap<String, u32>)> = rows
            .iter()
            .map(|(header, body)| (unigram_tf(body), unigram_tf(header)))
            .collect();
        Self::create_from_tf(path, tf_rows)
    }

    /// Build from per-row `(body term→tf, header term→tf)` maps — what
    /// [`rows_tf`](Self::rows_tf) recovers from an existing segment, so
    /// segments can be merged without re-chunking any source.
    pub fn create_from_tf(
        path: &Path,
        rows: Vec<(BTreeMap<String, u32>, BTreeMap<String, u32>)>,
    ) -> anyhow::Result<()> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).with_context(|| format!("create bm25 dir {}", dir.display()))?;
        }
        let n_rows = rows.len() as u32;

        // Inverted lists: term -> [(row, tf_body, tf_header)], row-ordered
        // because rows are visited in ascending order.
        let mut terms: BTreeMap<String, Vec<(u32, u32, u32)>> = BTreeMap::new();
        let mut row_lens: Vec<(u16, u8)> = Vec::with_capacity(rows.len());
        let mut tot_body = 0u64;
        let mut tot_head = 0u64;
        for (i, (body_tf, head_tf)) in rows.into_iter().enumerate() {
            let row = i as u32;
            let blen: u32 = body_tf.values().sum();
            let hlen: u32 = head_tf.values().sum();
            row_lens.push((
                blen.min(u16::MAX as u32) as u16,
                hlen.min(u8::MAX as u32) as u8,
            ));
            tot_body += u64::from(blen);
            tot_head += u64::from(hlen);
            for (term, tf) in body_tf {
                terms.entry(term).or_default().push((row, tf, 0));
            }
            for (term, tf) in head_tf {
                let list = terms.entry(term).or_default();
                match list.last_mut() {
                    Some(e) if e.0 == row => e.2 = tf, // same row: body pass ran first
                    _ => list.push((row, 0, tf)),
                }
            }
        }
        let avg_body = if n_rows > 0 { tot_body / u64::from(n_rows) } else { 0 };
        let avg_head = if n_rows > 0 { tot_head / u64::from(n_rows) } else { 0 };

        // Postings block + FST term dictionary (term -> postings offset).
        let mut postings: Vec<u8> = Vec::new();
        let mut fstb = fst::MapBuilder::memory();
        for (term, list) in &terms {
            let off = postings.len() as u64;
            leb128_push(&mut postings, list.len() as u64); // df
            let mut prev = 0u64;
            for &(row, tf_body, tf_header) in list {
                leb128_push(&mut postings, u64::from(row) - prev);
                prev = u64::from(row);
                leb128_push(&mut postings, u64::from(tf_body));
                leb128_push(&mut postings, u64::from(tf_header));
            }
            fstb.insert(term, off)
                .with_context(|| format!("fst insert '{term}'"))?;
        }
        let fst_bytes = fstb
            .into_inner()
            .map_err(|e| anyhow!("building bm25 term dictionary: {e}"))?;

        let idx = ChunkBm25 {
            path: path.to_path_buf(),
            n_rows,
            avg_body_len: avg_body as f64,
            avg_header_len: avg_head as f64,
            dict: fst::Map::new(Bytes::Owned(fst_bytes))
                .map_err(|e| anyhow!("finalizing bm25 term dictionary: {e}"))?,
            postings: Bytes::Owned(postings),
            row_lens,
            tombstones: BTreeSet::new(),
        };
        idx.write_file()?;
        let side = crate::index::tomb_side_path(path);
        if side.exists() {
            fs::remove_file(&side).ok();
        }
        Ok(())
    }

    /// Opens the sidecar; `Ok(None)` if the file does not exist.
    pub fn open(dir: &Path, ns: &str) -> anyhow::Result<Option<Self>> {
        let path = file_path(dir, ns);
        if !path.exists() {
            return Ok(None);
        }
        Self::open_path(&path).map(Some)
    }

    /// Open one segment file by path: memory-mapped, the term dictionary
    /// and postings block are ranges of the mapping (SPEC-P10).
    ///
    /// # Safety (why the mapping is sound)
    /// `.cibm25` files are written to a temp path and renamed into place,
    /// never modified (tombstones live in a side file).
    #[allow(unsafe_code)]
    pub fn open_path(path: &Path) -> anyhow::Result<Self> {
        let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let map = Arc::new(unsafe { Mmap::map(&file) }.with_context(|| format!("map {}", path.display()))?);
        let mut idx = Self::parse(path.to_path_buf(), Some(&map), &map[..])?;
        idx.tombstones.extend(crate::index::read_tomb_side(path));
        Ok(idx)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Per-row `(body term→tf, header term→tf)` recovered from the inverted
    /// lists (tombstoned rows included, as empty maps are fine to carry).
    pub fn rows_tf(&self) -> Vec<(BTreeMap<String, u32>, BTreeMap<String, u32>)> {
        let mut rows: Vec<(BTreeMap<String, u32>, BTreeMap<String, u32>)> =
            (0..self.n_rows).map(|_| (BTreeMap::new(), BTreeMap::new())).collect();
        let mut stream = self.dict.stream();
        use fst::Streamer as _;
        while let Some((term, off)) = stream.next() {
            let term = String::from_utf8_lossy(term).into_owned();
            let Some((_, postings)) = decode_postings(self.postings.as_ref(), off as usize) else {
                continue;
            };
            for (row, tf_body, tf_header) in postings {
                let Some(r) = rows.get_mut(row as usize) else { continue };
                if tf_body > 0 {
                    r.0.insert(term.clone(), tf_body);
                }
                if tf_header > 0 {
                    r.1.insert(term.clone(), tf_header);
                }
            }
        }
        rows
    }

    /// Parse a segment; with `map` given, the dictionary and postings
    /// blocks are ranges of it instead of copies.
    fn parse(path: PathBuf, map: Option<&Arc<Mmap>>, buf: &[u8]) -> anyhow::Result<Self> {
        let bad = || anyhow!("{}: corrupt .cibm25 file", path.display());
        let mut off = 0usize;
        let take = |off: &mut usize, n: usize| -> anyhow::Result<&[u8]> {
            let s = buf.get(*off..*off + n).ok_or_else(bad)?;
            *off += n;
            Ok(s)
        };
        let block = |start: usize, bytes: &[u8]| -> Bytes {
            match map {
                Some(m) => Bytes::Mapped { map: Arc::clone(m), start, end: start + bytes.len() },
                None => Bytes::Owned(bytes.to_vec()),
            }
        };
        ensure!(take(&mut off, 8)? == MAGIC, "bad .cibm25 magic");
        let n_rows = u32::from_le_bytes(take(&mut off, 4)?.try_into().unwrap());
        let _n_terms = u32::from_le_bytes(take(&mut off, 4)?.try_into().unwrap());
        let avg_body_len =
            f64::from(u32::from_le_bytes(take(&mut off, 4)?.try_into().unwrap()));
        let avg_header_len =
            f64::from(u32::from_le_bytes(take(&mut off, 4)?.try_into().unwrap()));

        let fst_len = u64::from_le_bytes(take(&mut off, 8)?.try_into().unwrap()) as usize;
        let fst_start = off;
        let fst_bytes = block(fst_start, take(&mut off, fst_len)?);
        let dict = fst::Map::new(fst_bytes).map_err(|_| bad())?;

        let postings_len = u64::from_le_bytes(take(&mut off, 8)?.try_into().unwrap()) as usize;
        let postings_start = off;
        let postings = block(postings_start, take(&mut off, postings_len)?);

        let rowlens_len = u64::from_le_bytes(take(&mut off, 8)?.try_into().unwrap()) as usize;
        ensure!(rowlens_len == n_rows as usize * 3, "rowlens_len != n_rows * 3");
        let rowlens = take(&mut off, rowlens_len)?;
        let mut row_lens = Vec::with_capacity(n_rows as usize);
        for c in rowlens.chunks_exact(3) {
            row_lens.push((u16::from_le_bytes([c[0], c[1]]), c[2]));
        }

        let tomb_len = u64::from_le_bytes(take(&mut off, 8)?.try_into().unwrap()) as usize;
        let tombs: Vec<u32> = bincode::deserialize(take(&mut off, tomb_len)?)
            .map_err(|_| bad())?;
        let tombstones: BTreeSet<u32> = tombs.into_iter().collect();
        ensure!(
            tombstones.iter().all(|&r| r < n_rows),
            "tombstone row out of range"
        );

        Ok(ChunkBm25 {
            path,
            n_rows,
            avg_body_len,
            avg_header_len,
            dict,
            postings,
            row_lens,
            tombstones,
        })
    }

    /// Serializes the full current state (index + tombstones) via
    /// tmp+rename.
    fn write_file(&self) -> anyhow::Result<()> {
        let tombs: Vec<u32> = self.tombstones.iter().copied().collect();
        let tombs = bincode::serialize(&tombs)?;

        let mut out = Vec::with_capacity(
            44 + self.postings.len() + self.row_lens.len() * 3 + tombs.len(),
        );
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.n_rows.to_le_bytes());
        out.extend_from_slice(&(self.dict.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.avg_body_len as u32).to_le_bytes());
        out.extend_from_slice(&(self.avg_header_len as u32).to_le_bytes());
        let fst_bytes = self.dict.as_fst().as_bytes();
        out.extend_from_slice(&(fst_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(fst_bytes);
        out.extend_from_slice(&(self.postings.len() as u64).to_le_bytes());
        out.extend_from_slice(self.postings.as_ref());
        out.extend_from_slice(&((self.row_lens.len() * 3) as u64).to_le_bytes());
        for &(bl, hl) in &self.row_lens {
            out.extend_from_slice(&bl.to_le_bytes());
            out.push(hl);
        }
        out.extend_from_slice(&(tombs.len() as u64).to_le_bytes());
        out.extend_from_slice(&tombs);

        let tmp = self.path.with_extension("cibm25.tmp");
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

    /// BM25F top-k over live rows. Returns `(row, score)` sorted by score
    /// desc (ties by row asc), score > 0 only.
    ///
    /// Query-side term processing (SPEC-P5 §A1): the shared tokenizer's
    /// unigram terms, deduped, minus English stopwords
    /// ([`crate::embed::stopwords`]) and minus hub terms with df > 40% of
    /// rows. Scoring: idf = ln(1 + (N - df + 0.5)/(df + 0.5)) with N = all
    /// rows; per row, tf' = tf_body + 3.0 * tf_header with the combined
    /// field length normalized against the combined average (k1 = 1.0,
    /// b = 0.35).
    pub fn search(&self, query: &str, k: usize) -> Vec<(u32, f32)> {
        if k == 0 || self.n_rows == 0 || self.is_empty() {
            return Vec::new();
        }
        let terms = query_terms(query);
        let n = f64::from(self.n_rows);
        let hub_df = (HUB_DF_FRAC * n).floor() as u64;
        let avgdl = (self.avg_body_len + self.avg_header_len).max(1.0);
        let mut acc: Vec<f64> = vec![0.0; self.n_rows as usize];
        let mut touched: Vec<u32> = Vec::new();
        for term in &terms {
            let Some(df) = self.term_df(term) else { continue };
            if u64::from(df) > hub_df {
                continue; // hub cut: df > 40% of rows carries no signal
            }
            let idf = idf(n, df);
            self.score_into(term, idf, avgdl, &mut acc, &mut touched);
        }
        collect_top(acc, touched, 0, k)
    }

    /// Document frequency of `term` in this segment (`None` when absent).
    pub fn term_df(&self, term: &str) -> Option<u32> {
        let off = self.dict.get(term)?;
        decode_postings(self.postings.as_ref(), off as usize).map(|(df, _)| df)
    }

    /// Rows and average document length this segment contributes to
    /// global statistics: (n_rows, Σ(body+header lens)).
    pub fn length_stats(&self) -> (u64, f64) {
        (
            u64::from(self.n_rows),
            (self.avg_body_len + self.avg_header_len) * f64::from(self.n_rows),
        )
    }

    /// Add `term`'s BM25F contribution to every live row it occurs in,
    /// with caller-supplied (possibly global) idf and average length.
    /// Dense accumulator + touched list: a map insert per posting dominated
    /// this leg once a query term had a long posting list.
    pub fn score_into(&self, term: &str, idf: f64, avgdl: f64, acc: &mut Vec<f64>, touched: &mut Vec<u32>) {
        self.score_into_excluding(term, idf, avgdl, &BTreeSet::new(), acc, touched)
    }

    /// [`score_into`](Self::score_into) that also skips the rows in `dead`
    /// (the tombstones a [`Bm25Set`] keeps outside the shared segment).
    pub fn score_into_excluding(
        &self,
        term: &str,
        idf: f64,
        avgdl: f64,
        dead: &BTreeSet<u32>,
        acc: &mut Vec<f64>,
        touched: &mut Vec<u32>,
    ) {
        let Some(off) = self.dict.get(term) else { return };
        let Some((_, postings)) = decode_postings(self.postings.as_ref(), off as usize) else {
            return; // corrupt list: skip term rather than fail the query
        };
        for (row, tf_body, tf_header) in postings {
            if self.tombstones.contains(&row) || dead.contains(&row) {
                continue;
            }
            let Some(&(bl, hl)) = self.row_lens.get(row as usize) else {
                continue; // corrupt row id
            };
            let dl = f64::from(bl) + f64::from(hl);
            let tfw = f64::from(tf_body) + HEADER_WEIGHT * f64::from(tf_header);
            let score = idf * tfw * (K1 + 1.0) / (tfw + K1 * (1.0 - B + B * dl / avgdl));
            let slot = &mut acc[row as usize];
            if *slot == 0.0 {
                touched.push(row);
            }
            *slot += score;
        }
    }

    /// Tombstone every row matching `pred` (by row id — the same row ids as
    /// the aligned VecIndex). Returns rows newly deleted.
    pub fn delete_where(&mut self, pred: impl Fn(u32) -> bool) -> u64 {
        let mut n = 0u64;
        for row in 0..self.n_rows {
            if !self.tombstones.contains(&row) && pred(row) {
                self.tombstones.insert(row);
                n += 1;
            }
        }
        n
    }

    /// Persists tombstones to the `<file>.tomb` side file (SPEC-P9); the
    /// segment itself is never rewritten.
    pub fn save_tombstones(&self) -> anyhow::Result<()> {
        crate::index::write_tomb_side(&self.path, &self.tombstones)
    }

    /// Number of live (non-tombstoned) rows.
    pub fn len(&self) -> usize {
        self.n_rows as usize - self.tombstones.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total row slots including tombstoned (= the aligned VecIndex row
    /// count).
    pub fn raw_len(&self) -> usize {
        self.n_rows as usize
    }

    pub fn is_deleted(&self, row: u32) -> bool {
        self.tombstones.contains(&row)
    }

    /// Rows tombstoned inside the file or its side file at open time.
    pub fn tombstones(&self) -> &BTreeSet<u32> {
        &self.tombstones
    }
}

/// Query-side term processing (SPEC-P5 §A1): the shared tokenizer's
/// unigram terms, deduped, minus English stopwords.
pub fn query_terms(query: &str) -> Vec<String> {
    let stop: BTreeSet<&str> = stopwords().iter().copied().collect();
    let mut terms: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for t in tokenize(query) {
        if t.term.contains(' ') || stop.contains(t.term.as_str()) {
            continue;
        }
        if seen.insert(t.term.clone()) {
            terms.push(t.term);
        }
    }
    terms
}

fn idf(n: f64, df: u32) -> f64 {
    (1.0 + (n - f64::from(df) + 0.5) / (f64::from(df) + 0.5)).ln()
}

/// Top-k of a dense score vector; rows are offset into a global id space.
fn collect_top(acc: Vec<f64>, touched: Vec<u32>, offset: u32, k: usize) -> Vec<(u32, f32)> {
    collect_top_where(acc, touched, offset, k, &|_| true)
}

/// [`collect_top`] keeping only global rows for which `keep` holds.
fn collect_top_where(acc: Vec<f64>, mut touched: Vec<u32>, offset: u32, k: usize, keep: &dyn Fn(u32) -> bool) -> Vec<(u32, f32)> {
    touched.sort_unstable();
    touched.dedup();
    let mut out: Vec<(u32, f32)> = touched
        .into_iter()
        .map(|r| (r, acc[r as usize]))
        .filter(|(_, s)| *s > 0.0)
        .map(|(r, s)| (offset + r, s as f32))
        .filter(|(g, _)| keep(*g))
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out.truncate(k);
    out
}

// ---------------------------------------------------------------------------
// Bm25Set: the segmented sidecar (SPEC-P9), row-aligned with VecSet
// ---------------------------------------------------------------------------

/// Segment files of `ns` in `dir`: the legacy base first, then
/// `<ns>.d<NNNN>.cibm25` ascending — the same order as the vector
/// segments, so global row ids line up.
pub fn segment_paths(dir: &Path, ns: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let base = file_path(dir, ns);
    if base.exists() {
        out.push(base);
    }
    let prefix = format!("{ns}.d");
    let mut deltas: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()) == Some("cibm25")
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

/// The BM25 segment path aligned with a vector segment path (same stem).
pub fn aligned_path(dir: &Path, vec_seg: &Path) -> PathBuf {
    let name = vec_seg
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("segment.civec");
    let stem = name.strip_suffix(".civec").unwrap_or(name);
    dir.join(format!("{stem}.cibm25"))
}

/// Cache stamp over every segment and side file.
pub fn set_stamp(dir: &Path, ns: &str) -> Option<(u64, Option<std::time::SystemTime>)> {
    let paths = segment_paths(dir, ns);
    if paths.is_empty() {
        return None;
    }
    let mut len = 0u64;
    let mut newest: Option<std::time::SystemTime> = None;
    let sides: Vec<PathBuf> = paths.iter().map(|p| crate::index::tomb_side_path(p)).collect();
    for p in paths.iter().chain(sides.iter()) {
        if let Ok(m) = fs::metadata(p) {
            len += m.len();
            if let Ok(t) = m.modified() {
                newest = Some(newest.map_or(t, |n| n.max(t)));
            }
        }
    }
    Some((len, newest))
}

pub struct Bm25Set {
    segs: Vec<std::sync::Arc<ChunkBm25>>,
    stamps: Vec<(u64, Option<std::time::SystemTime>)>,
    tombs: Vec<BTreeSet<u32>>,
    offsets: Vec<u32>,
}

fn file_stamp(p: &Path) -> (u64, Option<std::time::SystemTime>) {
    match fs::metadata(p) {
        Ok(m) => (m.len(), m.modified().ok()),
        Err(_) => (0, None),
    }
}

impl Bm25Set {
    pub fn open(dir: &Path, ns: &str) -> anyhow::Result<Option<Self>> {
        Self::reopen(None, dir, ns)
    }

    /// Open the current segment list, reusing `prev`'s parsed segments
    /// whose files are unchanged (see `VecSet::reopen`).
    pub fn reopen(prev: Option<&Bm25Set>, dir: &Path, ns: &str) -> anyhow::Result<Option<Self>> {
        let paths = segment_paths(dir, ns);
        if paths.is_empty() {
            return Ok(None);
        }
        let mut segs = Vec::with_capacity(paths.len());
        let mut stamps = Vec::with_capacity(paths.len());
        let mut tombs = Vec::with_capacity(paths.len());
        for p in &paths {
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
                None => std::sync::Arc::new(ChunkBm25::open_path(p)?),
            };
            let mut t = seg.tombstones().clone();
            t.extend(crate::index::read_tomb_side(p));
            segs.push(seg);
            stamps.push(stamp);
            tombs.push(t);
        }
        let mut offsets = Vec::with_capacity(segs.len());
        let mut acc = 0u32;
        for s in &segs {
            offsets.push(acc);
            acc = acc.checked_add(s.n_rows).ok_or_else(|| anyhow!("too many bm25 rows"))?;
        }
        Ok(Some(Bm25Set { segs, stamps, tombs, offsets }))
    }

    pub fn segments(&self) -> &[std::sync::Arc<ChunkBm25>] {
        &self.segs
    }

    pub fn raw_len(&self) -> usize {
        self.segs.iter().map(|s| s.raw_len()).sum()
    }

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

    pub fn locate(&self, row: u32) -> (usize, u32) {
        let i = match self.offsets.binary_search(&row) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        (i, row - self.offsets[i])
    }

    pub fn is_deleted(&self, row: u32) -> bool {
        let (s, r) = self.locate(row);
        self.tombs[s].contains(&r)
    }

    /// BM25F top-k over all segments with GLOBAL statistics: N and each
    /// term's df are summed over the segments and the average length is
    /// row-weighted, so a row scores the same wherever it lives.
    pub fn search(&self, query: &str, k: usize) -> Vec<(u32, f32)> {
        self.search_where(query, k, &|_| true)
    }

    /// [`search`](Self::search) keeping only the global rows for which
    /// `keep` holds (SPEC-P10: a repo-scoped leg).
    pub fn search_where(&self, query: &str, k: usize, keep: &dyn Fn(u32) -> bool) -> Vec<(u32, f32)> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        let terms = query_terms(query);
        let (mut n_total, mut len_total) = (0u64, 0f64);
        for s in &self.segs {
            let (n, l) = s.length_stats();
            n_total += n;
            len_total += l;
        }
        let n = n_total as f64;
        let hub_df = (HUB_DF_FRAC * n).floor() as u64;
        let avgdl = if n_total > 0 { (len_total / n).max(1.0) } else { 1.0 };
        let mut out: Vec<(u32, f32)> = Vec::new();
        let mut accs: Vec<(Vec<f64>, Vec<u32>)> =
            self.segs.iter().map(|s| (vec![0.0; s.raw_len()], Vec::new())).collect();
        for term in &terms {
            let df: u64 = self.segs.iter().filter_map(|s| s.term_df(term)).map(u64::from).sum();
            if df == 0 || df > hub_df {
                continue;
            }
            let idf = idf(n, df.min(u32::MAX as u64) as u32);
            for (i, s) in self.segs.iter().enumerate() {
                let (acc, touched) = &mut accs[i];
                s.score_into_excluding(term, idf, avgdl, &self.tombs[i], acc, touched);
            }
        }
        for (i, (acc, touched)) in accs.into_iter().enumerate() {
            out.extend(collect_top_where(acc, touched, self.offsets[i], k, keep));
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(k);
        out
    }

    /// Tombstone every live global row matching `pred`.
    pub fn delete_where(&mut self, pred: impl Fn(u32) -> bool) -> u64 {
        let mut n = 0u64;
        for (i, s) in self.segs.iter().enumerate() {
            let off = self.offsets[i];
            for r in 0..s.n_rows {
                if !self.tombs[i].contains(&r) && pred(off + r) {
                    self.tombs[i].insert(r);
                    n += 1;
                }
            }
        }
        n
    }

    pub fn save_tombstones(&self) -> anyhow::Result<()> {
        for (s, t) in self.segs.iter().zip(&self.tombs) {
            crate::index::write_tomb_side(s.path(), t)?;
        }
        Ok(())
    }

    /// Write side files with `rows` added, without mutating this set.
    pub fn tombstone_on_disk(&self, rows: &std::collections::HashSet<u32>) -> anyhow::Result<()> {
        let mut per_seg: Vec<BTreeSet<u32>> = self.tombs.clone();
        for &g in rows {
            let (s, r) = self.locate(g);
            per_seg[s].insert(r);
        }
        for (i, s) in self.segs.iter().enumerate() {
            if per_seg[i].len() != self.tombs[i].len() {
                crate::index::write_tomb_side(s.path(), &per_seg[i])?;
            }
        }
        Ok(())
    }

    /// Per-row term maps of segment `i`, tombstoned rows emptied.
    pub fn rows_tf_of(&self, i: usize) -> Vec<(BTreeMap<String, u32>, BTreeMap<String, u32>)> {
        let mut rows = self.segs[i].rows_tf();
        for &r in &self.tombs[i] {
            if let Some(row) = rows.get_mut(r as usize) {
                *row = (BTreeMap::new(), BTreeMap::new());
            }
        }
        rows
    }
}

// ---------------------------------------------------------------------------
// Tests (SPEC-P5 §A)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(h, b)| (h.to_string(), b.to_string()))
            .collect()
    }

    /// Planted-doc ordering: the doc with the most query-term mass wins,
    /// in-memory and after reopen; k > matches truncates to matches.
    /// (5 rows so the df-2 query terms stay under the 40% hub cut.)
    #[test]
    fn create_open_search_planted_ordering() {
        let tmp = tempfile::tempdir().unwrap();
        let data = rows(&[
            ("src/auth.rs > fn authenticate", "user session login with oauth tokens"),
            ("src/db.rs > fn pool", "postgres connection pooling with tls timeouts"),
            ("src/auth.rs > fn token", "oauth token refresh login session login"),
            ("src/util.rs > fn misc", "gamma delta epsilon"),
            ("src/tail.rs > fn tail", "theta iota kappa"),
        ]);
        ChunkBm25::create(tmp.path(), "m", &data).unwrap();
        assert!(tmp.path().join("m.cibm25").is_file());

        let bm = ChunkBm25::open(tmp.path(), "m").unwrap().unwrap();
        assert_eq!(bm.len(), 5);
        assert_eq!(bm.raw_len(), 5);
        let hits = bm.search("login session", 10);
        assert!(!hits.is_empty());
        // Row 2 mentions both terms (login x2, session x1) -> #1.
        assert_eq!(hits[0].0, 2, "{hits:?}");
        assert!(hits.iter().any(|(r, _)| *r == 0), "{hits:?}");
        // db.rs/util/tail share no query term -> absent (score > 0 only).
        assert!(hits.iter().all(|(r, _)| *r == 0 || *r == 2), "{hits:?}");
        for w in hits.windows(2) {
            assert!(w[0].1 > w[1].1, "{hits:?}");
        }

        // Reopen from disk: identical ordering.
        let bm2 = ChunkBm25::open(tmp.path(), "m").unwrap().unwrap();
        assert_eq!(bm.search("login session", 10), bm2.search("login session", 10));
        // Absent namespace -> Ok(None); k = 0 -> empty.
        assert!(ChunkBm25::open(tmp.path(), "nope").unwrap().is_none());
        assert!(bm2.search("login", 0).is_empty());
    }

    /// Header field boost (SPEC-P5 §A1): a term appearing only in the
    /// header (weight 3.0) beats the same term appearing only in the body
    /// at the same tf, on equal-length rows.
    #[test]
    fn header_field_boost_measurable() {
        let tmp = tempfile::tempdir().unwrap();
        // Equal field lengths (header 4 unigrams, body 3) and tf 1 each;
        // only the field differs. 5 rows so zzqheader's df of 2 stays at
        // the 40% hub-cut boundary (kept).
        let data = rows(&[
            ("src/x.rs > fn zzqheader", "alpha beta gamma"),
            ("src/y.rs > fn other", "alpha beta zzqheader"),
            ("src/f1.rs > fn f1", "delta epsilon zeta"),
            ("src/f2.rs > fn f2", "eta theta iota"),
            ("src/f3.rs > fn f3", "kappa lambda mu"),
        ]);
        ChunkBm25::create(tmp.path(), "m", &data).unwrap();
        let bm = ChunkBm25::open(tmp.path(), "m").unwrap().unwrap();
        let hits = bm.search("zzqheader", 10);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!(hits[0].0, 0, "header-only match must rank first: {hits:?}");
        assert!(hits[0].1 > hits[1].1, "{hits:?}");
        // k1=1, equal lengths (c = 1-b+b*dl/avgdl = 1): tf'=3 vs 1 ->
        // scores 3*2/(3+1)=1.5 vs 1*2/(1+1)=1.0 (times equal idf).
        let ratio = f64::from(hits[0].1) / f64::from(hits[1].1);
        assert!(ratio > 1.4, "header boost too weak: ratio {ratio}");
    }

    /// Hub df cut: a term with df > 40% of rows is dropped from the query
    /// (a doc matching ONLY the hub term must not surface).
    #[test]
    fn hub_df_cut() {
        let tmp = tempfile::tempdir().unwrap();
        // 5 rows; "hubterm" in 3 (60% > 40%) -> cut. "rareterm" in 1.
        let data = rows(&[
            ("src/a.rs", "hubterm alpha"),
            ("src/b.rs", "hubterm beta"),
            ("src/c.rs", "hubterm rareterm"),
            ("src/d.rs", "gamma delta"),
            ("src/e.rs", "epsilon zeta"),
        ]);
        ChunkBm25::create(tmp.path(), "m", &data).unwrap();
        let bm = ChunkBm25::open(tmp.path(), "m").unwrap().unwrap();
        let hits = bm.search("hubterm", 10);
        assert!(hits.is_empty(), "hub term must be cut entirely: {hits:?}");
        let hits = bm.search("hubterm rareterm", 10);
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].0, 2, "only the rare term may score: {hits:?}");
        // Under the cut a term in exactly 40% of rows still scores.
        let data2 = rows(&[
            ("src/a.rs", "edgeterm alpha"),
            ("src/b.rs", "edgeterm beta"),
            ("src/c.rs", "gamma delta"),
            ("src/d.rs", "epsilon zeta"),
            ("src/e.rs", "theta iota"),
        ]);
        ChunkBm25::create(tmp.path(), "m2", &data2).unwrap();
        let bm2 = ChunkBm25::open(tmp.path(), "m2").unwrap().unwrap();
        assert_eq!(bm2.search("edgeterm", 10).len(), 2, "df == 40% is kept");
    }

    /// Stopword strip: stopwords never score; an all-stopword query is
    /// empty even when every doc contains the words.
    #[test]
    fn stopword_strip() {
        let tmp = tempfile::tempdir().unwrap();
        // 5 rows so "fox" (df 2/5 = 40%) survives the hub cut.
        let data = rows(&[
            ("src/a.rs", "the quick brown fox"),
            ("src/b.rs", "the lazy dog sleeps"),
            ("src/c.rs", "the fox terrier runs"),
            ("src/d.rs", "gamma delta epsilon"),
            ("src/e.rs", "zeta eta theta"),
        ]);
        ChunkBm25::create(tmp.path(), "m", &data).unwrap();
        let bm = ChunkBm25::open(tmp.path(), "m").unwrap().unwrap();
        assert!(bm.search("the", 10).is_empty(), "'the' is a stopword");
        assert!(bm.search("the and of", 10).is_empty());
        // Mixed query: only content words contribute; fox docs surface.
        let hits = bm.search("the fox", 10);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!(hits.iter().all(|(r, _)| *r == 0 || *r == 2), "{hits:?}");
    }

    /// Tombstones: delete_where by row id, exclusion from search, len,
    /// persistence across save+reopen.
    #[test]
    fn tombstones_delete_persist() {
        let tmp = tempfile::tempdir().unwrap();
        // 5 rows; "needle" in rows 0 and 1 (df 2/5 = 40%: kept).
        let data = rows(&[
            ("src/a.rs", "needle alpha"),
            ("src/b.rs", "needle beta"),
            ("src/c.rs", "gamma delta"),
            ("src/d.rs", "epsilon zeta"),
            ("src/e.rs", "theta iota"),
        ]);
        ChunkBm25::create(tmp.path(), "m", &data).unwrap();
        let mut bm = ChunkBm25::open(tmp.path(), "m").unwrap().unwrap();
        assert_eq!(bm.search("needle", 10).len(), 2);

        assert_eq!(bm.delete_where(|r| r == 1), 1);
        assert_eq!(bm.delete_where(|r| r == 1), 0, "re-delete is a no-op");
        assert_eq!(bm.len(), 4);
        assert!(bm.is_deleted(1));
        let hits = bm.search("needle", 10);
        assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![0]);

        bm.save_tombstones().unwrap();
        drop(bm);
        let bm2 = ChunkBm25::open(tmp.path(), "m").unwrap().unwrap();
        assert_eq!(bm2.len(), 4);
        assert!(bm2.is_deleted(1));
        assert_eq!(
            bm2.search("needle", 10).iter().map(|h| h.0).collect::<Vec<_>>(),
            vec![0]
        );
    }

    /// Row-alignment invariant with VecIndex (SPEC-P5 §A2): built from the
    /// same rows in the same order, the same row-id tombstone predicate
    /// leaves both indexes with identical live sets.
    #[test]
    fn tombstone_alignment_with_vecindex_rows() {
        use crate::embed::{Embedder, HashEmbedder};
        use crate::index::{VecIndex, VecRowMeta};
        let tmp = tempfile::tempdir().unwrap();
        let texts = [
            ("src/a.rs", "alpha unique token one"),
            ("src/b.rs", "beta unique token two"),
            ("src/c.rs", "gamma unique token three"),
        ];
        let e = HashEmbedder::new(64);
        let vrows: Vec<(VecRowMeta, Vec<f32>)> = texts
            .iter()
            .map(|(path, body)| {
                let text = format!("{path}\n{body}");
                (
                    VecRowMeta {
                        chunk_hash: crate::store::chunk_hash(text.as_bytes()),
                        repo: "r".to_string(),
                        path: path.to_string(),
                        start_line: 1,
                        end_line: 2,
                    },
                    e.embed(&[text]).unwrap().remove(0),
                )
            })
            .collect();
        let brows: Vec<(String, String)> = texts
            .iter()
            .map(|(h, b)| (h.to_string(), b.to_string()))
            .collect();
        let mut vi = VecIndex::create(tmp.path(), "m", 64, vrows).unwrap();
        ChunkBm25::create(tmp.path(), "m", &brows).unwrap();
        let mut bm = ChunkBm25::open(tmp.path(), "m").unwrap().unwrap();
        assert_eq!(vi.len(), bm.len());

        // Same predicate on row ids (row 1 dropped in both).
        let drop_row = 1u32;
        assert_eq!(vi.delete_where(|m| m.path == "src/b.rs"), 1);
        assert_eq!(bm.delete_where(|r| r == drop_row), 1);
        vi.save_tombstones().unwrap();
        bm.save_tombstones().unwrap();
        let live_v: Vec<u32> = (0..3).filter(|&r| !vi.is_deleted(r)).collect();
        let live_b: Vec<u32> = (0..3).filter(|&r| !bm.is_deleted(r)).collect();
        assert_eq!(live_v, live_b, "live row ids must stay aligned");
        assert_eq!(bm.len(), vi.len());
    }

    /// Corrupt file -> Err (not a panic); empty corpus is graceful.
    #[test]
    fn corrupt_and_empty_edge_cases() {
        let tmp = tempfile::tempdir().unwrap();
        ChunkBm25::create(tmp.path(), "empty", &[]).unwrap();
        let bm = ChunkBm25::open(tmp.path(), "empty").unwrap().unwrap();
        assert_eq!(bm.len(), 0);
        assert!(bm.is_empty());
        assert!(bm.search("anything", 10).is_empty());

        let dir = tmp.path().join("bad");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("x.cibm25"), b"CIBM25 1 truncated").unwrap();
        assert!(ChunkBm25::open(&dir, "x").is_err());
    }
}
