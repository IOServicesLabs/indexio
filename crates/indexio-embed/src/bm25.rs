//! Chunk-level BM25F sidecar, aligned row-for-row with the
//! vector index of the same namespace (`<dir>/<ns>.cibm25` next to
//! `<data_dir>/vec/<model_id>.civec`; the pipeline passes
//! `<data_dir>/bm25`).
//!
//! Two fields per chunk: **header** (path + scope line, weight
//! [`HEADER_WEIGHT`]) and **body** (weight 1.0). Tokenizer: the shared
//! [`crate::rindex::tokenize`] (identifier-aware splitting, lowercasing);
//! adjacent-bigram terms (terms containing a space) are skipped — bigram
//! boost scoring is a non-goal.
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
//! Format notes:
//! - The file is read with safe `std::fs::read` into owned memory at open
//!   instead of being mmap'd, keeping this crate `#![forbid(unsafe_code)]`
//!   (same deviation as the `.civec` index; memmap2 is only sanctioned in
//!   indexio-index).
//! - The header holds `n_terms` (the df table is the per-list LEB128 df prefix inside the
//!   postings block) plus `avg_body_len`/`avg_header_len`.
//! - BM25F is scored with a single weighted term frequency
//!   `tf' = tf_body + HEADER_WEIGHT * tf_header` normalized by the combined
//!   field length `dl = body_len + header_len` against the combined average
//!   (k1 = [`K1`], b = [`B`]), rather than per-field b factors — the spec
//!   fixes one b for both fields.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, ensure, Context};
use memmap2::Mmap;

/// A byte block that is either owned (a segment being built) or a range of
/// a memory-mapped file (an opened segment: the sidecar used to
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

#[cfg(test)]
impl Bytes {
    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

use crate::embed::stopwords;
use crate::rindex::tokenize;
use crate::spool::{scratch_file, Spool};

const MAGIC: &[u8; 8] = b"CIBM25 1";
/// BM25F parameters: short, uniform-length chunks justify the
/// low b (b in 0.3-0.4 suits passage-like docs).
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

pub(crate) fn file_path(dir: &Path, ns: &str) -> PathBuf {
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

/// Bytes [`leb128_push`] writes for `v`.
fn leb128_len(v: u64) -> u64 {
    u64::from((64 - v.leading_zeros()).max(1).div_ceil(7))
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
#[cfg(test)]
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

/// Walk the posting list at `off`, handing each `(row, tf_body, tf_header)`
/// to `f`: returns df, `None` on truncation/corruption (where
/// [`decode_postings`] gives `None`; `f` has then seen the postings before
/// the damage). Nothing is allocated: a query term in two rows of five
/// has a posting list of megabytes once decoded.
fn walk_postings(buf: &[u8], mut off: usize, mut f: impl FnMut(u32, u32, u32)) -> Option<u32> {
    let df = leb128_read(buf, &mut off)? as u32;
    let mut row = 0u64;
    for _ in 0..df {
        row = row.checked_add(leb128_read(buf, &mut off)?)?;
        let tf_body = leb128_read(buf, &mut off)? as u32;
        let tf_header = leb128_read(buf, &mut off)? as u32;
        f(u32::try_from(row).ok()?, tf_body, tf_header);
    }
    Some(df)
}

// ---------------------------------------------------------------------------
// Writing segments
// ---------------------------------------------------------------------------

/// The postings of one term while a segment is being built: the LEB128
/// `(row_delta, tf_body, tf_header)` triples as the file will carry them.
#[derive(Default)]
struct TermList {
    df: u32,
    /// Row of the last posting: the next one is written as a delta from it.
    last: u32,
    bytes: Vec<u8>,
}

/// Bytes of the term table (terms and encoded postings) at which a
/// [`Bm25Writer`] sets the rows folded so far aside as a part.
pub const TABLE_BUDGET: usize = 32 << 20;

/// What a term costs in the table besides its text and its postings.
const TERM_OVERHEAD: usize = 96;

/// Postings a merge holds in memory before they move to a scratch file.
const POSTINGS_IN_MEMORY: usize = 4 << 20;

/// Rows of a segment being built that were set aside: a complete segment
/// in a scratch file, mapped. The mapping goes before the file handle that
/// keeps the file alive.
struct Part {
    seg: ChunkBm25,
    _file: fs::File,
}

/// Builds one `.cibm25` segment a row at a time.
///
/// `create_path` used to turn every row into a pair of term maps, invert
/// those into one map of `(row, tf, tf)` lists and serialize the whole file
/// into a buffer: 2.3 to 2.9 GB for the 945k rows of a rebuild. Here a row
/// is folded straight into the term table, whose lists are the encoded
/// bytes the file carries; when the table reaches its budget the rows so
/// far are written out as a part (a segment in a scratch file) and the
/// table starts again. [`finish`](Self::finish) writes the table as the
/// segment, or merges the parts into it term by term. Either way the file
/// is the one `create_path` wrote, byte for byte.
pub struct Bm25Writer {
    dir: PathBuf,
    budget: usize,
    table: HashMap<String, TermList>,
    table_bytes: usize,
    /// One row's (body, header) count per term, emptied after each row.
    row_terms: HashMap<String, (u32, u32)>,
    /// (body, header) token counts of the rows in `table`.
    lens: Vec<(u32, u32)>,
    parts: Vec<Part>,
    n_rows: u32,
}

impl Bm25Writer {
    /// A writer whose scratch files, if it comes to them, live in `dir`:
    /// the directory the segment will be written to.
    pub fn new(dir: &Path) -> Self {
        Self::with_budget(dir, TABLE_BUDGET)
    }

    /// [`new`](Self::new) with a term table of `budget` bytes before rows
    /// are set aside (the file written does not depend on it).
    pub fn with_budget(dir: &Path, budget: usize) -> Self {
        Bm25Writer {
            dir: dir.to_path_buf(),
            budget,
            table: HashMap::new(),
            table_bytes: 0,
            row_terms: HashMap::new(),
            lens: Vec::new(),
            parts: Vec::new(),
            n_rows: 0,
        }
    }

    /// Rows pushed so far.
    pub fn rows(&self) -> u32 {
        self.n_rows
    }

    /// Append the row of a chunk: its header (path + scope line) and body.
    /// Row i here is row i of the vector segment written alongside.
    pub fn push(&mut self, header: &str, body: &str) -> anyhow::Result<()> {
        let row = self.lens.len() as u32;
        let (mut body_len, mut header_len) = (0u32, 0u32);
        for t in tokenize(body) {
            if t.term.contains(' ') {
                continue; // adjacent-bigram term: not indexed (a non-goal)
            }
            body_len += 1;
            self.row_terms.entry(t.term).or_default().0 += 1;
        }
        for t in tokenize(header) {
            if t.term.contains(' ') {
                continue;
            }
            header_len += 1;
            self.row_terms.entry(t.term).or_default().1 += 1;
        }
        self.lens.push((body_len, header_len));
        for (term, (tf_body, tf_header)) in self.row_terms.drain() {
            let list = match self.table.entry(term) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    self.table_bytes += e.key().len() + TERM_OVERHEAD;
                    e.insert(TermList::default())
                }
            };
            let held = list.bytes.capacity();
            leb128_push(&mut list.bytes, u64::from(row - list.last));
            leb128_push(&mut list.bytes, u64::from(tf_body));
            leb128_push(&mut list.bytes, u64::from(tf_header));
            list.last = row;
            list.df += 1;
            self.table_bytes += list.bytes.capacity() - held;
        }
        self.n_rows = self.n_rows.checked_add(1).ok_or_else(|| anyhow!("too many bm25 rows"))?;
        if self.table_bytes >= self.budget {
            self.cut_part()?;
        }
        Ok(())
    }

    /// Set the rows in the table aside as a part and empty the table.
    fn cut_part(&mut self) -> anyhow::Result<()> {
        if self.lens.is_empty() {
            return Ok(());
        }
        let file = scratch_file(&self.dir, "cibm25-part").with_context(|| format!("scratch file in {}", self.dir.display()))?;
        let mut out = std::io::BufWriter::with_capacity(1 << 20, file);
        write_table(&mut out, &self.table, &self.lens)?;
        let file = out.into_inner().map_err(|e| e.into_error())?;
        let seg = map_part(&file)?;
        self.parts.push(Part { seg, _file: file });
        self.table.clear();
        self.table_bytes = 0;
        self.lens.clear();
        Ok(())
    }

    /// Write the segment at `path` (tmp + rename) and remove a tombstone
    /// side file left at that path.
    pub fn finish(mut self, path: &Path) -> anyhow::Result<()> {
        if self.parts.is_empty() {
            return write_segment_file(path, |out| write_table(out, &self.table, &self.lens));
        }
        self.cut_part()?;
        self.table = HashMap::new();
        let mut base = 0u32;
        let mut sources: Vec<(&ChunkBm25, RowMap<'_>)> = Vec::with_capacity(self.parts.len());
        for part in &self.parts {
            sources.push((&part.seg, RowMap::Shift(base)));
            base += part.seg.n_rows;
        }
        merge_segments(path, self.n_rows, &sources)
    }
}

/// Map a part just written to its scratch file.
///
/// # Safety (why the mapping is sound)
/// The file is this writer's alone (a scratch file has no name another
/// process could open it by) and nothing writes to it after this point.
#[allow(unsafe_code)]
fn map_part(file: &fs::File) -> anyhow::Result<ChunkBm25> {
    let map = Arc::new(unsafe { Mmap::map(file) }.context("map a bm25 part")?);
    ChunkBm25::parse(PathBuf::from("bm25 part"), Some(&map), &map[..])
}

/// Write a `.cibm25` file at `path` through `body` (tmp + rename), then
/// remove a tombstone side file left at that path.
fn write_segment_file(path: &Path, body: impl FnOnce(&mut dyn Write) -> anyhow::Result<()>) -> anyhow::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).with_context(|| format!("create bm25 dir {}", dir.display()))?;
    }
    let tmp = path.with_extension("cibm25.tmp");
    let file = fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
    let mut out = std::io::BufWriter::with_capacity(1 << 20, file);
    if let Err(e) = body(&mut out).and_then(|()| Ok(out.flush()?)) {
        drop(out);
        fs::remove_file(&tmp).ok();
        return Err(e);
    }
    out.get_ref().sync_all().ok();
    drop(out);
    fs::rename(&tmp, path).with_context(|| format!("rename to {}", path.display()))?;
    let side = crate::index::tomb_side_path(path);
    if side.exists() {
        fs::remove_file(&side).ok();
    }
    Ok(())
}

/// The file up to its postings: header, term dictionary, postings length.
fn write_head(out: &mut dyn Write, n_rows: u32, n_terms: u32, tot_body: u64, tot_head: u64, dict: &[u8], postings_len: u64) -> std::io::Result<()> {
    let avg = |tot: u64| if n_rows > 0 { tot / u64::from(n_rows) } else { 0 };
    out.write_all(MAGIC)?;
    out.write_all(&n_rows.to_le_bytes())?;
    out.write_all(&n_terms.to_le_bytes())?;
    out.write_all(&((avg(tot_body) as f64) as u32).to_le_bytes())?;
    out.write_all(&((avg(tot_head) as f64) as u32).to_le_bytes())?;
    out.write_all(&(dict.len() as u64).to_le_bytes())?;
    out.write_all(dict)?;
    out.write_all(&postings_len.to_le_bytes())
}

/// The file after its postings: the row lengths (saturating) and no
/// tombstone (bincode's empty `Vec<u32>`).
fn write_tail(out: &mut dyn Write, lens: impl ExactSizeIterator<Item = (u32, u32)>) -> std::io::Result<()> {
    out.write_all(&((lens.len() * 3) as u64).to_le_bytes())?;
    for (body, header) in lens {
        out.write_all(&(body.min(u16::MAX as u32) as u16).to_le_bytes())?;
        out.write_all(&[header.min(u8::MAX as u32) as u8])?;
    }
    out.write_all(&8u64.to_le_bytes())?;
    out.write_all(&0u64.to_le_bytes())
}

/// Write the segment a term table holds. The dictionary precedes the
/// postings in the file and points into them, so the terms are walked
/// twice: once for the dictionary (a term's offset is the size of the
/// lists before it), once to write the lists. No copy of the postings is
/// made.
fn write_table(out: &mut dyn Write, table: &HashMap<String, TermList>, lens: &[(u32, u32)]) -> anyhow::Result<()> {
    let mut terms: Vec<(&str, &TermList)> = table.iter().map(|(t, l)| (t.as_str(), l)).collect();
    terms.sort_unstable_by(|a, b| a.0.cmp(b.0));
    let mut dict = fst::MapBuilder::memory();
    let mut postings_len = 0u64;
    for (term, list) in &terms {
        dict.insert(term, postings_len).with_context(|| format!("fst insert '{term}'"))?;
        postings_len += leb128_len(u64::from(list.df)) + list.bytes.len() as u64;
    }
    let dict = dict.into_inner().map_err(|e| anyhow!("building bm25 term dictionary: {e}"))?;
    let tot_body: u64 = lens.iter().map(|l| u64::from(l.0)).sum();
    let tot_head: u64 = lens.iter().map(|l| u64::from(l.1)).sum();
    write_head(out, lens.len() as u32, terms.len() as u32, tot_body, tot_head, &dict, postings_len)?;
    let mut df = Vec::with_capacity(5);
    for (_, list) in &terms {
        df.clear();
        leb128_push(&mut df, u64::from(list.df));
        out.write_all(&df)?;
        out.write_all(&list.bytes)?;
    }
    write_tail(out, lens.iter().copied())?;
    Ok(())
}

/// A row that is not carried into the merged segment.
pub(crate) const DROPPED: u32 = u32::MAX;

/// Where the rows of a source segment go in the segment being written.
pub(crate) enum RowMap<'a> {
    /// Row `r` becomes row `base + r`: the parts of one build, in row order.
    Shift(u32),
    /// Row `r` becomes row `table[r]`, or is left out when that is
    /// [`DROPPED`] or `r` is past the table.
    Table(&'a [u32]),
}

impl RowMap<'_> {
    fn get(&self, row: u32) -> Option<u32> {
        match self {
            RowMap::Shift(base) => base.checked_add(row),
            RowMap::Table(t) => t.get(row as usize).copied().filter(|&r| r != DROPPED),
        }
    }
}

/// Write the segment at `path` whose `n_rows` rows are rows of `sources`,
/// renumbered by each source's [`RowMap`]: the dictionaries are walked
/// together in term order and each term's postings are gathered from the
/// sources, renumbered, put in row order and written. What is held is one
/// term's postings and two counters per row.
///
/// A compaction used to recover a pair of term maps for every row of every
/// input (`rows_tf`) and build the merged segment from those (11 to 17 KB
/// per row carried, with the vectors). The file written here is the same:
/// a posting `(row, tf_body, tf_header)` of an input is the posting
/// `(new row, tf_body, tf_header)` of the output, a term is in the
/// dictionary when a carried row has it, and a row's lengths are the sums
/// of its term frequencies. Like `rows_tf`, a posting list that does not
/// decode is left out whole and a posting of a row the source does not
/// have is ignored.
pub(crate) fn merge_segments(path: &Path, n_rows: u32, sources: &[(&ChunkBm25, RowMap<'_>)]) -> anyhow::Result<()> {
    use fst::Streamer as _;
    let mut dicts = fst::map::OpBuilder::new();
    for (seg, _) in sources {
        dicts.push(&seg.dict);
    }
    let mut terms = dicts.union();
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut postings = Spool::new(dir, "cibm25-postings", POSTINGS_IN_MEMORY);
    let mut dict = fst::MapBuilder::memory();
    let mut n_terms = 0u32;
    let mut lens: Vec<(u32, u32)> = vec![(0, 0); n_rows as usize];
    // one term at a time: where it is in each source, its postings, their bytes
    let mut found: Vec<(usize, u64)> = Vec::new();
    let mut list: Vec<(u32, u32, u32)> = Vec::new();
    let mut bytes: Vec<u8> = Vec::new();
    while let Some((term, offs)) = terms.next() {
        found.clear();
        found.extend(offs.iter().map(|o| (o.index, o.value)));
        found.sort_unstable();
        list.clear();
        let mut in_order = true;
        for &(src, off) in &found {
            let (seg, map) = &sources[src];
            let buf = seg.postings.as_ref();
            // a damaged list contributes nothing
            if walk_postings(buf, off as usize, |_, _, _| {}).is_none() {
                continue;
            }
            walk_postings(buf, off as usize, |row, tf_body, tf_header| {
                if row >= seg.n_rows {
                    return;
                }
                if let Some(new) = map.get(row).filter(|&r| r < n_rows) {
                    in_order &= list.last().is_none_or(|last| last.0 < new);
                    list.push((new, tf_body, tf_header));
                }
            });
        }
        if !in_order {
            list.sort_by_key(|p| p.0); // stable: repeats of a row keep their order
        }
        bytes.clear();
        let (mut df, mut prev) = (0u64, 0u32);
        let mut at = 0;
        while at < list.len() {
            // a row listed twice (no writer does that) counts once, its
            // last non-zero frequencies standing, as in a map per row
            let (row, mut tf_body, mut tf_header) = list[at];
            at += 1;
            while at < list.len() && list[at].0 == row {
                if list[at].1 > 0 {
                    tf_body = list[at].1;
                }
                if list[at].2 > 0 {
                    tf_header = list[at].2;
                }
                at += 1;
            }
            if tf_body == 0 && tf_header == 0 {
                continue;
            }
            leb128_push(&mut bytes, u64::from(row - prev));
            leb128_push(&mut bytes, u64::from(tf_body));
            leb128_push(&mut bytes, u64::from(tf_header));
            prev = row;
            df += 1;
            let len = &mut lens[row as usize];
            len.0 += tf_body;
            len.1 += tf_header;
        }
        if df == 0 {
            continue; // no carried row has the term
        }
        dict.insert(term, postings.len()).with_context(|| format!("fst insert '{}'", String::from_utf8_lossy(term)))?;
        n_terms += 1;
        let mut head = Vec::with_capacity(5);
        leb128_push(&mut head, df);
        postings.write(&head)?;
        postings.write(&bytes)?;
    }
    drop(terms);
    let dict = dict.into_inner().map_err(|e| anyhow!("building bm25 term dictionary: {e}"))?;
    let tot_body: u64 = lens.iter().map(|l| u64::from(l.0)).sum();
    let tot_head: u64 = lens.iter().map(|l| u64::from(l.1)).sum();
    write_segment_file(path, |out| {
        write_head(out, n_rows, n_terms, tot_body, tot_head, &dict, postings.len())?;
        postings.copy_to(out)?;
        write_tail(out, lens.iter().copied())?;
        Ok(())
    })
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

    /// [`create`](Self::create) at an explicit segment path: every row
    /// through a [`Bm25Writer`]. For callers that hold their rows anyway;
    /// the pipeline feeds the writer itself.
    pub fn create_path(path: &Path, rows: &[(String, String)]) -> anyhow::Result<()> {
        let mut w = Bm25Writer::new(path.parent().unwrap_or(Path::new(".")));
        for (header, body) in rows {
            w.push(header, body)?;
        }
        w.finish(path)
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
    /// and postings block are ranges of the mapping.
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

    /// BM25F top-k over live rows. Returns `(row, score)` sorted by score
    /// desc (ties by row asc), score > 0 only.
    ///
    /// Query-side term processing: the shared tokenizer's
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
        walk_postings(self.postings.as_ref(), off as usize, |_, _, _| {})
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
        let postings = self.postings.as_ref();
        // corrupt list: skip the term rather than fail the query. The list
        // is checked whole first, so a damaged one scores none of its rows.
        if walk_postings(postings, off as usize, |_, _, _| {}).is_none() {
            return;
        }
        walk_postings(postings, off as usize, |row, tf_body, tf_header| {
            if self.tombstones.contains(&row) || dead.contains(&row) {
                return;
            }
            let Some(&(bl, hl)) = self.row_lens.get(row as usize) else {
                return; // corrupt row id
            };
            let dl = f64::from(bl) + f64::from(hl);
            let tfw = f64::from(tf_body) + HEADER_WEIGHT * f64::from(tf_header);
            let score = idf * tfw * (K1 + 1.0) / (tfw + K1 * (1.0 - B + B * dl / avgdl));
            let slot = &mut acc[row as usize];
            if *slot == 0.0 {
                touched.push(row);
            }
            *slot += score;
        });
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

    /// Persists tombstones to the `<file>.tomb` side file; the
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

/// Query-side term processing: the shared tokenizer's
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
// Bm25Set: the segmented sidecar, row-aligned with VecSet
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
    set_stamp_paths(&segment_paths(dir, ns))
}

/// [`set_stamp`] over an explicit segment list.
pub fn set_stamp_paths(paths: &[PathBuf]) -> Option<(u64, Option<std::time::SystemTime>)> {
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

/// The BM25 segments this process has open (see
/// [`crate::index::OpenSegments`]: same rule, same freshness).
static OPEN_SEGMENTS: crate::index::OpenSegments<ChunkBm25> = crate::index::OpenSegments::new();

impl Bm25Set {
    pub fn open(dir: &Path, ns: &str) -> anyhow::Result<Option<Self>> {
        Self::reopen(None, dir, ns)
    }

    /// Open the current segment list, reusing `prev`'s parsed segments
    /// whose files are unchanged (see `VecSet::reopen`).
    pub fn reopen(prev: Option<&Bm25Set>, dir: &Path, ns: &str) -> anyhow::Result<Option<Self>> {
        Self::reopen_paths(prev, &segment_paths(dir, ns))
    }

    /// [`reopen`](Self::reopen) over an explicit segment list.
    pub fn reopen_paths(prev: Option<&Bm25Set>, paths: &[PathBuf]) -> anyhow::Result<Option<Self>> {
        if paths.is_empty() {
            return Ok(None);
        }
        let mut segs = Vec::with_capacity(paths.len());
        let mut stamps = Vec::with_capacity(paths.len());
        let mut tombs = Vec::with_capacity(paths.len());
        for p in paths {
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
                    let seg = std::sync::Arc::new(ChunkBm25::open_path(p)?);
                    OPEN_SEGMENTS.put(p, stamp, &seg);
                    seg
                }
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
    /// `keep` holds (a repo-scoped leg).
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

    /// Tombstoned local rows of segment `seg` (in-file ∪ side file).
    pub fn dead(&self, seg: usize) -> &BTreeSet<u32> {
        &self.tombs[seg]
    }
}

// ---------------------------------------------------------------------------
// The writers before segments were streamed
// ---------------------------------------------------------------------------

/// What built and merged `.cibm25` segments before [`Bm25Writer`] and
/// [`merge_segments`]: a pair of term maps per row, one map of posting
/// lists, the file in one buffer. Kept, as it was, for the tests that hold
/// the streaming writers to the bytes these wrote.
#[cfg(test)]
pub(crate) mod reference {
    use std::collections::BTreeMap;

    use super::*;

    /// Unigram term frequencies of `text` (bigram terms skipped).
    fn unigram_tf(text: &str) -> BTreeMap<String, u32> {
        let mut tf: BTreeMap<String, u32> = BTreeMap::new();
        for t in tokenize(text) {
            if t.term.contains(' ') {
                continue; // adjacent-bigram term: not indexed (a non-goal)
            }
            *tf.entry(t.term).or_insert(0) += 1;
        }
        tf
    }

    pub fn create_path(path: &Path, rows: &[(String, String)]) -> anyhow::Result<()> {
        let tf_rows: Vec<(BTreeMap<String, u32>, BTreeMap<String, u32>)> = rows
            .iter()
            .map(|(header, body)| (unigram_tf(body), unigram_tf(header)))
            .collect();
        create_from_tf(path, tf_rows)
    }

    /// Build from per-row `(body term→tf, header term→tf)` maps — what
    /// [`rows_tf`] recovers from an existing segment.
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
        write_file(&idx)?;
        let side = crate::index::tomb_side_path(path);
        if side.exists() {
            fs::remove_file(&side).ok();
        }
        Ok(())
    }

    /// Per-row `(body term→tf, header term→tf)` recovered from the inverted
    /// lists (tombstoned rows included, as empty maps are fine to carry).
    pub fn rows_tf(seg: &ChunkBm25) -> Vec<(BTreeMap<String, u32>, BTreeMap<String, u32>)> {
        let mut rows: Vec<(BTreeMap<String, u32>, BTreeMap<String, u32>)> =
            (0..seg.n_rows).map(|_| (BTreeMap::new(), BTreeMap::new())).collect();
        let mut stream = seg.dict.stream();
        use fst::Streamer as _;
        while let Some((term, off)) = stream.next() {
            let term = String::from_utf8_lossy(term).into_owned();
            let Some((_, postings)) = decode_postings(seg.postings.as_ref(), off as usize) else {
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

    /// Serializes the full current state (index + tombstones) via
    /// tmp+rename.
    fn write_file(idx: &ChunkBm25) -> anyhow::Result<()> {
        let tombs: Vec<u32> = idx.tombstones.iter().copied().collect();
        let tombs = bincode::serialize(&tombs)?;

        let mut out = Vec::with_capacity(
            44 + idx.postings.len() + idx.row_lens.len() * 3 + tombs.len(),
        );
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&idx.n_rows.to_le_bytes());
        out.extend_from_slice(&(idx.dict.len() as u32).to_le_bytes());
        out.extend_from_slice(&(idx.avg_body_len as u32).to_le_bytes());
        out.extend_from_slice(&(idx.avg_header_len as u32).to_le_bytes());
        let fst_bytes = idx.dict.as_fst().as_bytes();
        out.extend_from_slice(&(fst_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(fst_bytes);
        out.extend_from_slice(&(idx.postings.len() as u64).to_le_bytes());
        out.extend_from_slice(idx.postings.as_ref());
        out.extend_from_slice(&((idx.row_lens.len() * 3) as u64).to_le_bytes());
        for &(bl, hl) in &idx.row_lens {
            out.extend_from_slice(&bl.to_le_bytes());
            out.push(hl);
        }
        out.extend_from_slice(&(tombs.len() as u64).to_le_bytes());
        out.extend_from_slice(&tombs);

        let tmp = idx.path.with_extension("cibm25.tmp");
        {
            let mut f = fs::File::create(&tmp)
                .with_context(|| format!("create {}", tmp.display()))?;
            f.write_all(&out)?;
            f.sync_all().ok();
        }
        fs::rename(&tmp, &idx.path)
            .with_context(|| format!("rename to {}", idx.path.display()))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
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

    /// The walk that allocates nothing sees the postings `decode_postings`
    /// returns and fails wherever it fails.
    #[test]
    fn walking_a_posting_list_matches_decoding_it() {
        let encode = |list: &[(u64, u32, u32)]| {
            let mut buf = vec![0xAA; 3]; // the list starts at offset 3
            leb128_push(&mut buf, list.len() as u64);
            let mut prev = 0u64;
            for &(row, body, header) in list {
                leb128_push(&mut buf, row - prev);
                leb128_push(&mut buf, u64::from(body));
                leb128_push(&mut buf, u64::from(header));
                prev = row;
            }
            buf
        };
        let walk = |buf: &[u8]| {
            let mut seen = Vec::new();
            let df = walk_postings(buf, 3, |r, b, h| seen.push((r, b, h)));
            (df, seen)
        };
        let list: Vec<(u64, u32, u32)> = vec![(0, 1, 0), (7, 3, 1), (137, 300, 0), (1 << 20, 1, 70_000), (u64::from(u32::MAX), 2, 2)];
        let rows: Vec<(u32, u32, u32)> = list.iter().map(|&(r, b, h)| (r as u32, b, h)).collect();
        let buf = encode(&list);
        assert_eq!(walk(&buf), (Some(5), rows.clone()));
        assert_eq!(decode_postings(&buf, 3), Some((5, rows.clone())));
        assert_eq!(walk(&encode(&[])), (Some(0), Vec::new()));
        // cut anywhere: both fail; the walk saw the postings before the cut
        for cut in 3..buf.len() {
            let (df, seen) = walk(&buf[..cut]);
            assert_eq!(df, None, "cut at {cut}");
            assert!(decode_postings(&buf[..cut], 3).is_none(), "cut at {cut}");
            assert_eq!(seen[..], rows[..seen.len()]);
        }
        // a row id past u32
        let over = encode(&[(5, 1, 1), (u64::from(u32::MAX) + 1, 1, 1)]);
        assert_eq!(walk(&over), (None, vec![(5, 1, 1)]));
        assert!(decode_postings(&over, 3).is_none());
    }

    /// A term whose posting list is damaged scores nothing and counts in
    /// no document frequency, as when the list was decoded whole first.
    #[test]
    fn a_damaged_posting_list_scores_no_row() {
        let tmp = tempfile::tempdir().unwrap();
        let data = rows(&[
            ("a.rs", "alpha beta"),
            ("b.rs", "alpha gamma"),
            ("c.rs", "delta"),
            ("d.rs", "epsilon"),
            ("e.rs", "zeta"),
            ("f.rs", "eta"),
        ]);
        let path = tmp.path().join("t.cibm25");
        ChunkBm25::create_path(&path, &data).unwrap();
        let mut idx = ChunkBm25::open_path(&path).unwrap();
        assert_eq!(idx.term_df("alpha"), Some(2));
        let (mut acc, mut touched) = (vec![0.0; idx.raw_len()], Vec::new());
        idx.score_into("alpha", 1.0, 2.0, &mut acc, &mut touched);
        assert_eq!(touched, [0, 1]);

        // the same list cut after its first posting
        let off = idx.dict.get("alpha").unwrap() as usize;
        let whole = idx.postings.as_ref().to_vec();
        let mut end = off;
        leb128_read(&whole, &mut end).unwrap(); // df
        for _ in 0..3 {
            leb128_read(&whole, &mut end).unwrap(); // row, tf_body, tf_header of posting 1
        }
        idx.postings = Bytes::Owned(whole[..end].to_vec());
        assert_eq!(idx.term_df("alpha"), None);
        let (mut acc, mut touched) = (vec![0.0; idx.raw_len()], Vec::new());
        idx.score_into("alpha", 1.0, 2.0, &mut acc, &mut touched);
        assert!(touched.is_empty() && acc.iter().all(|s| *s == 0.0));
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

    /// Header field boost: a term appearing only in the
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

    /// Row-alignment invariant with VecIndex: built from the
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

    // ---- the streaming writers against the writers they replace -----------

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }

        /// A word of a vocabulary with a few very common members and a
        /// long tail, some of them identifiers that split, some not ASCII.
        fn word(&mut self) -> String {
            let u = self.below(1000) as f64 / 1000.0;
            let k = (u * u * u * 4000.0) as u64;
            match k % 11 {
                0 => format!("parseQuery{k}"),
                1 => format!("größe{k}"),
                2 => format!("数据{k}"),
                3 => format!("snake_case_{k}"),
                _ => format!("w{k}"),
            }
        }

        fn words(&mut self, n: u64) -> String {
            (0..n).map(|_| self.word()).collect::<Vec<_>>().join(" ")
        }
    }

    /// (header, body) rows of generated text, with the odd ones: empty
    /// fields, empty rows, and fields longer than the lengths a row stores
    /// (u16 body tokens, u8 header tokens).
    fn generated_rows(n: usize, seed: u64) -> Vec<(String, String)> {
        let mut rng = Rng(seed | 1);
        (0..n)
            .map(|i| {
                let header = format!("src/mod{}/file{}.rs > fn {}", i % 7, i % 31, rng.word());
                match i % 211 {
                    3 => (String::new(), String::new()),
                    5 => (header, String::new()),
                    7 => (String::new(), rng.words(40)),
                    11 => (rng.words(300), rng.words(5)),
                    13 if i % 1000 == 13 => (header, "tok ".repeat(70_000)),
                    _ => {
                        let len = 1 + rng.below(120);
                        (header, rng.words(len))
                    }
                }
            })
            .collect()
    }

    /// Extensions of the files in `dir`, sorted.
    fn extensions(dir: &Path) -> Vec<String> {
        let mut out: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.path().extension().and_then(|x| x.to_str()).unwrap_or("").to_string())
            .collect();
        out.sort();
        out.dedup();
        out
    }

    /// A segment built a row at a time is, byte for byte, the segment built
    /// from a pair of term maps per row, whether the term table held every
    /// row or was set aside as parts along the way.
    #[test]
    fn streamed_segments_are_the_segments_built_from_term_maps() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let (want_path, got_path) = (dir.join("want.cibm25"), dir.join("got.cibm25"));
        for (n, budgets) in [
            (0usize, &[TABLE_BUDGET, 0][..]),
            (1, &[TABLE_BUDGET, 0]),
            (60, &[TABLE_BUDGET, 0, 1, 3000]),
            (3000, &[TABLE_BUDGET, 40_000, 200_000]),
        ] {
            let rows = generated_rows(n, 17 + n as u64);
            reference::create_path(&want_path, &rows).unwrap();
            let want = fs::read(&want_path).unwrap();
            for &budget in budgets {
                let mut w = Bm25Writer::with_budget(dir, budget);
                for (header, body) in &rows {
                    w.push(header, body).unwrap();
                }
                assert_eq!(w.rows() as usize, n);
                let parts = w.parts.len();
                assert_eq!(parts > 0, n > 0 && budget < TABLE_BUDGET, "n={n} budget={budget}: {parts} parts");
                assert!(budget != 40_000 || parts > 10, "{parts} parts");
                w.finish(&got_path).unwrap();
                assert!(fs::read(&got_path).unwrap() == want, "n={n} budget={budget} ({parts} parts)");
            }
            // the wrapper callers with rows in hand use
            ChunkBm25::create_path(&got_path, &rows).unwrap();
            assert!(fs::read(&got_path).unwrap() == want, "create_path, n={n}");
            let seg = ChunkBm25::open_path(&got_path).unwrap();
            assert_eq!(seg.raw_len(), n);
            if n == 3000 {
                // the lengths saturate where the reference's did
                assert!(seg.row_lens.iter().any(|l| l.0 == u16::MAX) && seg.row_lens.iter().any(|l| l.1 == u8::MAX));
            }
        }
        assert_eq!(extensions(dir), ["cibm25"], "a temp or scratch file was left");
    }

    /// The reference merge of a compaction: recover a pair of term maps per
    /// row of each input, pick the carried rows in order (a row tombstoned
    /// on this side alone is carried bare), build the segment from them.
    fn reference_merge(path: &Path, segs: &[ChunkBm25], order: &[(usize, u32)], bare: &dyn Fn(usize, u32) -> bool) {
        let mut recovered: Vec<_> = segs.iter().map(reference::rows_tf).collect();
        let tf_rows = order
            .iter()
            .map(|&(s, r)| if bare(s, r) { Default::default() } else { std::mem::take(&mut recovered[s][r as usize]) })
            .collect();
        reference::create_from_tf(path, tf_rows).unwrap();
    }

    fn streamed_merge(path: &Path, segs: &[ChunkBm25], order: &[(usize, u32)], bare: &dyn Fn(usize, u32) -> bool) {
        let mut maps: Vec<Vec<u32>> = segs.iter().map(|s| vec![DROPPED; s.raw_len()]).collect();
        for (new_row, &(s, r)) in order.iter().enumerate() {
            if !bare(s, r) {
                maps[s][r as usize] = new_row as u32;
            }
        }
        let sources: Vec<(&ChunkBm25, RowMap<'_>)> = segs.iter().zip(&maps).map(|(s, m)| (s, RowMap::Table(m))).collect();
        merge_segments(path, order.len() as u32, &sources).unwrap();
    }

    /// A merge of segments by their posting lists writes, byte for byte,
    /// the segment built from the rows recovered out of them.
    #[test]
    fn merged_segments_are_the_segments_built_from_recovered_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let mut segs: Vec<ChunkBm25> = Vec::new();
        for (i, n) in [700usize, 1, 0, 1100, 55].into_iter().enumerate() {
            let p = dir.join(format!("in{i}.cibm25"));
            ChunkBm25::create_path(&p, &generated_rows(n, 900 + i as u64)).unwrap();
            segs.push(ChunkBm25::open_path(&p).unwrap());
        }
        let (want_path, got_path) = (dir.join("want.cibm25"), dir.join("got.cibm25"));
        let check = |segs: &[ChunkBm25], order: &[(usize, u32)], bare: &dyn Fn(usize, u32) -> bool, what: &str| {
            reference_merge(&want_path, segs, order, bare);
            streamed_merge(&got_path, segs, order, bare);
            assert!(fs::read(&got_path).unwrap() == fs::read(&want_path).unwrap(), "{what}");
        };
        let all: Vec<(usize, u32)> = segs.iter().enumerate().flat_map(|(s, seg)| (0..seg.raw_len() as u32).map(move |r| (s, r))).collect();
        let none = |_: usize, _: u32| false;

        // every row, in segment order: the concatenation
        check(&segs, &all, &none, "every row in order");
        let term = "tok";
        assert_eq!(ChunkBm25::open_path(&got_path).unwrap().term_df(term), Some(3), "three inputs have the long row");
        // most rows, in an order that mixes the segments; some carried bare
        let mut order: Vec<(usize, u32)> = all.iter().copied().filter(|&(s, r)| (r + s as u32) % 9 != 2).collect();
        let mut rng = Rng(0xC0FFEE);
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i as u64 + 1) as usize);
        }
        check(&segs, &order, &|s, r| (r + 2 * s as u32) % 13 == 5, "rows dropped, permuted, some bare");
        // the order a compaction gives: rows of each input kept in order, inputs interleaved
        let mut by_row = order.clone();
        by_row.sort_by_key(|&(s, r)| (r, s));
        check(&segs, &by_row, &none, "inputs interleaved");
        // nothing carried, and one row
        check(&segs, &[], &none, "no row");
        check(&segs, &[(3, 1099)], &none, "one row");

        // an input with a posting list that does not decode: the list is
        // left out whole by both
        let victim = dir.join("in3.cibm25");
        let mut bytes = fs::read(&victim).unwrap();
        let fst_len = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
        let postings_at = 32 + fst_len + 8;
        let off = segs[3].dict.get(term).expect("the long row's term") as usize;
        bytes[postings_at + off..postings_at + off + 5].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0x7F]);
        drop(segs);
        fs::write(&victim, &bytes).unwrap();
        let segs: Vec<ChunkBm25> = (0..5).map(|i| ChunkBm25::open_path(&dir.join(format!("in{i}.cibm25"))).unwrap()).collect();
        check(&segs, &all, &none, "a damaged list");
        let merged = ChunkBm25::open_path(&got_path).unwrap();
        assert_eq!(merged.term_df(term), Some(2), "the damaged list was carried");
        assert_eq!(extensions(dir), ["cibm25"], "a temp or scratch file was left");
    }
}
