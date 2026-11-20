//! Row metadata of a vector segment, read in place.
//!
//! The metadata block of a `.civec` file is `bincode::serialize` of a
//! `Vec<VecRowMeta>` (bincode 1.3 free functions: fixed-width little-endian
//! integers, trailing bytes ignored on read):
//!
//! ```text
//! u64 n_rows
//! n_rows x ( 16B chunk_hash | u64 len + repo | u64 len + path
//!          | u32 start_line | u32 end_line )
//! ```
//!
//! A segment used to deserialize that block into a `Vec<VecRowMeta>`: two
//! heap strings and 72 bytes per row, 193 MB for a 986k-row plane, rebuilt
//! as a transient by every writer that opened the sets. The block is part
//! of the segment's memory mapping, so rows are now decoded where they
//! lie: [`MetaStore`] walks the block once at open (bounds, lengths, UTF-8,
//! row count: a damaged file fails there, as it did in the deserializer),
//! keeps one `u32` record offset per row, and hands out [`RowMeta`] views
//! whose strings borrow from the mapping.
//!
//! The same walk records where the file and the repo change between
//! consecutive rows. Writers lay rows out file by file (a compaction sorts
//! them by repo, path and line), so those run tables are a few KB and let
//! "which files have rows" and "the rows of this repo" be answered without
//! touching every row.
//!
//! Segments held in memory (v1 files, a segment being built, tests) keep
//! their rows as owned `VecRowMeta`s behind the same accessors.

use std::ops::Range;

use crate::bm25::Bytes;
use crate::index::VecRowMeta;

/// One row's metadata, borrowed from its segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowMeta<'a> {
    pub chunk_hash: [u8; 16],
    pub repo: &'a str,
    pub path: &'a str,
    pub start_line: u32,
    pub end_line: u32,
}

impl RowMeta<'_> {
    /// What a row whose record cannot be read decodes to. Unreachable for a
    /// block that passed the walk at open; it resolves to no document.
    const EMPTY: RowMeta<'static> = RowMeta { chunk_hash: [0; 16], repo: "", path: "", start_line: 0, end_line: 0 };

    /// The owned form the write side takes (`VecIndex::create_segment`).
    pub fn to_owned_meta(&self) -> VecRowMeta {
        VecRowMeta {
            chunk_hash: self.chunk_hash,
            repo: self.repo.to_string(),
            path: self.path.to_string(),
            start_line: self.start_line,
            end_line: self.end_line,
        }
    }
}

impl VecRowMeta {
    /// Borrowed view of this row.
    pub fn as_row(&self) -> RowMeta<'_> {
        RowMeta {
            chunk_hash: self.chunk_hash,
            repo: &self.repo,
            path: &self.path,
            start_line: self.start_line,
            end_line: self.end_line,
        }
    }
}

const HASH_LEN: usize = 16;
const LEN_LEN: usize = 8;
const LINES_LEN: usize = 8;
/// A record with two empty strings.
const MIN_RECORD: usize = HASH_LEN + 2 * LEN_LEN + LINES_LEN;

/// Why a metadata block cannot be read in place.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum BlockError {
    /// Truncated, a length running past the block, or a string that is not UTF-8.
    Corrupt,
    /// A well-formed block of this many rows where the header says otherwise.
    Count(u64),
    /// Record offsets would not fit `u32` (a block over 4 GB): the caller
    /// falls back to owned rows.
    TooLarge,
}

fn le_u32(b: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(off..off.checked_add(4)?)?.try_into().ok()?))
}

fn le_u64(b: &[u8], off: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(off..off.checked_add(8)?)?.try_into().ok()?))
}

/// The length prefix at `off` and the end of the bytes it announces.
fn string_end(b: &[u8], off: usize) -> Option<usize> {
    let len = usize::try_from(le_u64(b, off)?).ok()?;
    let end = off.checked_add(LEN_LEN)?.checked_add(len)?;
    (end <= b.len()).then_some(end)
}

/// Decode the record at `off`. `None` only for a block that did not pass
/// [`walk`] (every slice is checked: no input can make this panic).
fn decode(block: &[u8], off: usize) -> Option<RowMeta<'_>> {
    let chunk_hash: [u8; HASH_LEN] = block.get(off..off.checked_add(HASH_LEN)?)?.try_into().ok()?;
    let repo_at = off + HASH_LEN;
    let repo_end = string_end(block, repo_at)?;
    let path_end = string_end(block, repo_end)?;
    Some(RowMeta {
        chunk_hash,
        repo: std::str::from_utf8(block.get(repo_at + LEN_LEN..repo_end)?).ok()?,
        path: std::str::from_utf8(block.get(repo_end + LEN_LEN..path_end)?).ok()?,
        start_line: le_u32(block, path_end)?,
        end_line: le_u32(block, path_end + 4)?,
    })
}

/// What one pass over a block yields.
struct Walk {
    /// Offset of each row's record from the start of the block.
    offs: Vec<u32>,
    files: Vec<u32>,
    repos: Vec<u32>,
    /// Bytes the encoding occupies; what follows is ignored, as bincode does.
    len: usize,
}

/// Validate `block` and index its records.
fn walk(block: &[u8], n_rows: usize) -> Result<Walk, BlockError> {
    use BlockError::Corrupt;
    if block.len() > u32::MAX as usize {
        return Err(BlockError::TooLarge);
    }
    let count = le_u64(block, 0).ok_or(Corrupt)?;
    // a count the block cannot hold is damage, not a reason to reserve for it
    if count > ((block.len() - LEN_LEN) / MIN_RECORD) as u64 {
        return Err(Corrupt);
    }
    let mut offs: Vec<u32> = Vec::with_capacity(count as usize);
    let (mut files, mut repos) = (Vec::new(), Vec::new());
    // `len + repo + len + path` of the previous row, and how much of it is
    // `len + repo`: equal bytes are the same file (the encoding is
    // injective), so a file's strings are compared, and validated, once
    let mut prev_key: &[u8] = &[];
    let mut prev_repo_key = 0usize;
    let mut off = LEN_LEN;
    for row in 0..count as u32 {
        let repo_at = off.checked_add(HASH_LEN).ok_or(Corrupt)?;
        let repo_end = string_end(block, repo_at).ok_or(Corrupt)?;
        let path_end = string_end(block, repo_end).ok_or(Corrupt)?;
        let end = path_end.checked_add(LINES_LEN).filter(|&e| e <= block.len()).ok_or(Corrupt)?;
        let key = &block[repo_at..path_end];
        if key != prev_key {
            std::str::from_utf8(&block[repo_at + LEN_LEN..repo_end]).map_err(|_| Corrupt)?;
            std::str::from_utf8(&block[repo_end + LEN_LEN..path_end]).map_err(|_| Corrupt)?;
            files.push(row);
            let repo_key = repo_end - repo_at;
            if key[..repo_key] != prev_key[..prev_repo_key] {
                repos.push(row);
            }
            prev_key = key;
            prev_repo_key = repo_key;
        }
        offs.push(off as u32);
        off = end;
    }
    if count != n_rows as u64 {
        return Err(BlockError::Count(count));
    }
    Ok(Walk { offs, files, repos, len: off })
}

enum Rows {
    Owned(Vec<VecRowMeta>),
    /// `block` is the encoding itself (a range of the segment's mapping);
    /// `offs[row]` is where the row's record starts in it.
    InPlace { block: Bytes, offs: Vec<u32> },
}

/// The rows' metadata of one segment plus its file and repo run tables.
pub(crate) struct MetaStore {
    rows: Rows,
    /// First row of every run of consecutive rows of one (repo, path).
    files: Vec<u32>,
    /// First row of every run of consecutive rows of one repo.
    repos: Vec<u32>,
}

impl MetaStore {
    /// Rows held in memory.
    pub fn owned(rows: Vec<VecRowMeta>) -> Self {
        let (mut files, mut repos) = (Vec::new(), Vec::new());
        for (i, m) in rows.iter().enumerate() {
            let prev = i.checked_sub(1).map(|p| &rows[p]);
            if prev.is_none_or(|p| p.repo != m.repo) {
                repos.push(i as u32);
            }
            if prev.is_none_or(|p| p.repo != m.repo || p.path != m.path) {
                files.push(i as u32);
            }
        }
        MetaStore { rows: Rows::Owned(rows), files, repos }
    }

    /// Rows read in place from `block`, the metadata block of a segment
    /// whose header announces `n_rows`.
    pub fn in_place(block: Bytes, n_rows: usize) -> Result<Self, BlockError> {
        let Walk { offs, files, repos, len } = walk(block.as_ref(), n_rows)?;
        let block = match block {
            Bytes::Owned(mut v) => {
                v.truncate(len);
                Bytes::Owned(v)
            }
            Bytes::Mapped { map, start, .. } => Bytes::Mapped { map, start, end: start + len },
        };
        Ok(MetaStore { rows: Rows::InPlace { block, offs }, files, repos })
    }

    pub fn len(&self) -> usize {
        match &self.rows {
            Rows::Owned(v) => v.len(),
            Rows::InPlace { offs, .. } => offs.len(),
        }
    }

    /// Metadata of `row` (panics past the last row, like a slice index).
    pub fn row(&self, row: u32) -> RowMeta<'_> {
        match &self.rows {
            Rows::Owned(v) => v[row as usize].as_row(),
            Rows::InPlace { block, offs } => decode(block.as_ref(), offs[row as usize] as usize).unwrap_or(RowMeta::EMPTY),
        }
    }

    /// The chunk hash of `row` alone: no string is looked at.
    pub fn hash(&self, row: u32) -> [u8; 16] {
        match &self.rows {
            Rows::Owned(v) => v[row as usize].chunk_hash,
            Rows::InPlace { block, offs } => {
                let off = offs[row as usize] as usize;
                block.as_ref().get(off..off + HASH_LEN).and_then(|h| h.try_into().ok()).unwrap_or([0; HASH_LEN])
            }
        }
    }

    /// Runs of `starts` (ascending first rows) that overlap `rows`, each
    /// clipped to it.
    fn runs<'a>(&'a self, starts: &'a [u32], rows: Range<u32>) -> impl Iterator<Item = Range<u32>> + 'a {
        let n = self.len() as u32;
        let rows = rows.start..rows.end.min(n);
        // the run holding `rows.start` starts at or before it
        let first = starts.partition_point(|&s| s <= rows.start).saturating_sub(1);
        (first..starts.len())
            .map(move |i| (starts[i], starts.get(i + 1).copied().unwrap_or(n)))
            .take_while({
                let end = rows.end;
                move |&(s, _)| s < end
            })
            .map(move |(s, e)| s.max(rows.start)..e.min(rows.end))
            .filter(|r| r.start < r.end)
    }

    /// Runs of consecutive rows of one file inside `rows`.
    pub fn file_runs(&self, rows: Range<u32>) -> impl Iterator<Item = Range<u32>> + '_ {
        self.runs(&self.files, rows)
    }

    /// Runs of consecutive rows of one repo.
    pub fn repo_runs(&self) -> impl Iterator<Item = Range<u32>> + '_ {
        self.runs(&self.repos, 0..self.len() as u32)
    }

    /// The bincode block a segment file carries for these rows.
    pub fn encode(&self) -> bincode::Result<Vec<u8>> {
        match &self.rows {
            Rows::Owned(v) => bincode::serialize(v),
            Rows::InPlace { block, .. } => Ok(block.as_ref().to_vec()),
        }
    }

    /// Whether rows are read from the block (no owned row exists).
    #[cfg(test)]
    pub fn is_in_place(&self) -> bool {
        matches!(self.rows, Rows::InPlace { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    const REPOS: [&str; 5] = ["", "indexio", "dépôt-é", "仓库", "r"];
    const PATHS: [&str; 7] = ["", "src/lib.rs", "docs/überblick.md", "数据/模型.py", "a", "src/a b/c d.rs", "🦀/main.rs"];

    /// Rows laid out the way writers do (file by file), with empty and
    /// non-ASCII strings and the occasional file that comes back later.
    fn rows(n: usize, seed: u64) -> Vec<VecRowMeta> {
        let mut rng = Rng(seed | 1);
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let repo = REPOS[(rng.next() % REPOS.len() as u64) as usize];
            let path = PATHS[(rng.next() % PATHS.len() as u64) as usize];
            let path = if rng.next() % 3 == 0 { format!("{path}/{}", rng.next() % 50) } else { path.to_string() };
            for chunk in 0..1 + rng.next() % 6 {
                if out.len() == n {
                    break;
                }
                let mut chunk_hash = [0u8; 16];
                chunk_hash[..8].copy_from_slice(&rng.next().to_le_bytes());
                chunk_hash[8..].copy_from_slice(&rng.next().to_le_bytes());
                out.push(VecRowMeta {
                    chunk_hash,
                    repo: repo.to_string(),
                    path: path.clone(),
                    start_line: chunk as u32 * 40 + 1,
                    end_line: (rng.next() % u64::from(u32::MAX)) as u32,
                });
            }
        }
        out
    }

    fn in_place(rows: &[VecRowMeta]) -> MetaStore {
        MetaStore::in_place(Bytes::Owned(bincode::serialize(rows).unwrap()), rows.len()).unwrap()
    }

    fn collect(s: &MetaStore) -> Vec<VecRowMeta> {
        (0..s.len() as u32).map(|r| s.row(r).to_owned_meta()).collect()
    }

    #[test]
    fn in_place_rows_equal_the_owned_decode() {
        for (n, seed) in [(0usize, 1u64), (1, 2), (2, 3), (57, 4), (1000, 5), (5000, 6)] {
            let want = rows(n, seed);
            let bytes = bincode::serialize(&want).unwrap();
            let decoded: Vec<VecRowMeta> = bincode::deserialize(&bytes).unwrap();
            let store = in_place(&want);
            assert!(store.is_in_place());
            assert_eq!(store.len(), n);
            let got = collect(&store);
            for (r, (g, d)) in got.iter().zip(&decoded).enumerate() {
                assert_eq!(g.as_row(), d.as_row(), "row {r} of {n}");
                assert_eq!(store.hash(r as u32), d.chunk_hash);
            }
            assert_eq!(got.len(), decoded.len());
            // and the block written back is the block read
            assert_eq!(store.encode().unwrap(), bytes);
            assert_eq!(MetaStore::owned(want).encode().unwrap(), bytes);
        }
    }

    #[test]
    fn run_tables_match_between_owned_and_in_place() {
        let want = rows(3000, 11);
        let (a, b) = (MetaStore::owned(want.clone()), in_place(&want));
        assert_eq!(a.files, b.files);
        assert_eq!(a.repos, b.repos);
        let n = want.len() as u32;
        // every run is one file (resp. repo), and neighbours differ
        for s in [&a, &b] {
            let files: Vec<Range<u32>> = s.file_runs(0..n).collect();
            assert_eq!(files.iter().map(|r| r.len()).sum::<usize>(), want.len());
            for (i, run) in files.iter().enumerate() {
                let first = &want[run.start as usize];
                assert!(want[run.start as usize..run.end as usize].iter().all(|m| m.repo == first.repo && m.path == first.path));
                if let Some(next) = files.get(i + 1) {
                    assert_eq!(run.end, next.start);
                    let m = &want[next.start as usize];
                    assert!(m.repo != first.repo || m.path != first.path);
                }
            }
            let repos: Vec<Range<u32>> = s.repo_runs().collect();
            assert_eq!(repos.iter().map(|r| r.len()).sum::<usize>(), want.len());
            for (i, run) in repos.iter().enumerate() {
                let first = &want[run.start as usize];
                assert!(want[run.start as usize..run.end as usize].iter().all(|m| m.repo == first.repo));
                if let Some(next) = repos.get(i + 1) {
                    assert_ne!(want[next.start as usize].repo, first.repo);
                }
            }
        }
        // a window is the full list clipped to it
        let mut rng = Rng(99);
        for _ in 0..200 {
            let (x, y) = ((rng.next() % u64::from(n + 5)) as u32, (rng.next() % u64::from(n + 5)) as u32);
            let (lo, hi) = (x.min(y), x.max(y));
            let got: Vec<Range<u32>> = b.file_runs(lo..hi).collect();
            let want_runs: Vec<Range<u32>> = b
                .file_runs(0..n)
                .map(|r| r.start.max(lo)..r.end.min(hi))
                .filter(|r| r.start < r.end)
                .collect();
            assert_eq!(got, want_runs, "window {lo}..{hi}");
        }
    }

    #[test]
    fn trailing_bytes_are_ignored_like_bincode_does() {
        let want = rows(40, 21);
        let mut bytes = bincode::serialize(&want).unwrap();
        let len = bytes.len();
        bytes.extend_from_slice(&[0xff; 13]);
        assert!(bincode::deserialize::<Vec<VecRowMeta>>(&bytes).is_ok());
        let store = MetaStore::in_place(Bytes::Owned(bytes), want.len()).unwrap();
        assert_eq!(collect(&store).len(), 40);
        assert_eq!(store.encode().unwrap().len(), len);
    }

    #[test]
    fn damaged_blocks_are_rejected_wherever_bincode_rejects_them() {
        let want = rows(60, 31);
        let bytes = bincode::serialize(&want).unwrap();
        let both = |b: &[u8]| {
            let ours = MetaStore::in_place(Bytes::Owned(b.to_vec()), want.len()).map(|s| collect(&s));
            let theirs = bincode::deserialize::<Vec<VecRowMeta>>(b);
            (ours, theirs)
        };
        // every truncation
        for cut in 0..bytes.len() {
            let (ours, theirs) = both(&bytes[..cut]);
            assert!(theirs.is_err(), "bincode accepted a cut at {cut}");
            assert_eq!(ours.err(), Some(BlockError::Corrupt), "cut at {cut}");
        }
        // every single-byte change: same verdict, and the same rows when both read it
        for i in 0..bytes.len() {
            let mut b = bytes.clone();
            b[i] ^= 0x80;
            let (ours, theirs) = both(&b);
            match (ours, theirs) {
                (Ok(o), Ok(t)) => {
                    assert_eq!(o.len(), t.len());
                    assert!(o.iter().zip(&t).all(|(x, y)| x.as_row() == y.as_row()), "byte {i}");
                }
                (Err(BlockError::Count(c)), Ok(t)) => assert_eq!(c as usize, t.len(), "byte {i}"),
                (Err(BlockError::Corrupt), Err(_)) => {}
                (o, t) => panic!("byte {i}: in place {:?}, bincode {:?}", o.map(|v| v.len()), t.map(|v| v.len())),
            }
        }
        // invalid UTF-8 in a string
        let mut b = bytes.clone();
        let at = 8 + 16 + 8; // first repo byte of row 0, when it has one
        let first_repo_len = want[0].repo.len();
        if first_repo_len > 0 {
            b[at] = 0xff;
            assert_eq!(both(&b).0.err(), Some(BlockError::Corrupt));
        }
        // a count the block cannot hold must not reserve for it
        let mut b = bytes.clone();
        b[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(both(&b).0.err(), Some(BlockError::Corrupt));
        // a shorter, well-formed block against the header's count
        let fewer = bincode::serialize(&want[..10]).unwrap();
        assert_eq!(MetaStore::in_place(Bytes::Owned(fewer), want.len()).err(), Some(BlockError::Count(10)));
        // empty input
        assert_eq!(MetaStore::in_place(Bytes::Owned(Vec::new()), 0).err(), Some(BlockError::Corrupt));
    }

    #[test]
    fn decode_never_panics_on_arbitrary_offsets() {
        let bytes = bincode::serialize(&rows(30, 41)).unwrap();
        for off in 0..bytes.len() + 40 {
            let _ = decode(&bytes, off);
        }
        assert!(decode(&bytes, usize::MAX - 3).is_none());
    }
}
