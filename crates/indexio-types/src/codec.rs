//! Posting-list byte codec (SPEC: indexio-types).
//!
//! Format (little-endian): a posting list is a sequence of blocks of <=128
//! docs. Per block: `u32 last_docid`, `u32 block_bytes`, then per doc entry:
//! `varint(doc_delta_from_prev_in_block)`, `varint(pos_count)`,
//! `pos_count` x `varint(pos_delta)`. varint = unsigned LEB128.
//!
//! Since SPEC-P10 the writer emits `pos_count = 0` for every entry
//! ([`encode_doc_postings`]): readers only ever needed the docids. Lists
//! written with positions by older shards still decode.

const BLOCK_DOCS: usize = 128;

pub fn write_varint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(b);
            return;
        }
        buf.push(b | 0x80);
    }
}

pub fn read_varint(buf: &[u8], pos: &mut usize) -> Option<u64> {
    let mut v: u64 = 0;
    let mut shift = 0u32;
    loop {
        let b = *buf.get(*pos)?;
        *pos += 1;
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

/// Encode `(docid, positions)` pairs (must be sorted by docid, positions sorted,
/// docids unique) into the block format.
pub fn encode_postings(docs: &[(u32, Vec<u32>)]) -> Vec<u8> {
    let mut out = Vec::new();
    for block in docs.chunks(BLOCK_DOCS) {
        let last_docid = block.last().unwrap().0;
        let mut body = Vec::new();
        let mut prev = 0u32;
        for (docid, positions) in block {
            write_varint(&mut body, (docid - prev) as u64);
            prev = *docid;
            write_varint(&mut body, positions.len() as u64);
            let mut pprev = 0u32;
            for p in positions {
                write_varint(&mut body, (p - pprev) as u64);
                pprev = *p;
            }
        }
        out.extend_from_slice(&last_docid.to_le_bytes());
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
    }
    out
}

/// Encode a sorted, unique docid list without positions: the same block
/// format with `pos_count = 0` per entry, ~1-2 bytes per docid.
pub fn encode_doc_postings(docids: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(docids.len() * 2);
    for block in docids.chunks(BLOCK_DOCS) {
        let last_docid = *block.last().unwrap();
        let mut body = Vec::with_capacity(block.len() * 3);
        let mut prev = 0u32;
        for &docid in block {
            write_varint(&mut body, (docid - prev) as u64);
            prev = docid;
            body.push(0); // pos_count
        }
        out.extend_from_slice(&last_docid.to_le_bytes());
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
    }
    out
}

/// Full decode (used by tests and by ShardSet::merge).
pub fn decode_postings(buf: &[u8]) -> Vec<(u32, Vec<u32>)> {
    let mut out = Vec::new();
    let mut cursor = PostingCursor::new(buf);
    while let Some((docid, positions)) = cursor.next_entry() {
        out.push((docid, positions));
    }
    out
}

/// Zero-copy forward cursor with block-level `seek`.
pub struct PostingCursor<'a> {
    buf: &'a [u8],
    pos: usize,
    /// Position of the current block's body end (exclusive).
    block_end: usize,
    last_docid_in_block: u32,
    prev_docid: u32,
    done: bool,
}

impl<'a> PostingCursor<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        let mut c = PostingCursor {
            buf,
            pos: 0,
            block_end: 0,
            last_docid_in_block: 0,
            prev_docid: 0,
            done: buf.is_empty(),
        };
        if !c.done {
            c.load_block_header();
        }
        c
    }

    fn load_block_header(&mut self) -> bool {
        if self.pos + 8 > self.buf.len() {
            self.done = true;
            return false;
        }
        self.last_docid_in_block =
            u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        let block_bytes =
            u32::from_le_bytes(self.buf[self.pos + 4..self.pos + 8].try_into().unwrap()) as usize;
        self.pos += 8;
        self.block_end = self.pos + block_bytes;
        self.prev_docid = 0;
        if block_bytes == 0 {
            self.pos = self.block_end;
            return self.load_block_header();
        }
        true
    }

    /// Advance to the next entry. Returns None at end of list.
    pub fn next_entry(&mut self) -> Option<(u32, Vec<u32>)> {
        loop {
            if self.done {
                return None;
            }
            if self.pos >= self.block_end {
                if !self.load_block_header() {
                    return None;
                }
                continue;
            }
            let delta = read_varint(self.buf, &mut self.pos)?;
            let docid = self.prev_docid.checked_add(delta as u32)?;
            self.prev_docid = docid;
            let count = read_varint(self.buf, &mut self.pos)? as usize;
            let mut positions = Vec::with_capacity(count.min(1 << 16));
            let mut pprev = 0u32;
            for _ in 0..count {
                let pd = read_varint(self.buf, &mut self.pos)?;
                pprev = pprev.checked_add(pd as u32)?;
                positions.push(pprev);
            }
            return Some((docid, positions));
        }
    }

    /// Advance to the next entry and return only its docid, skipping the
    /// position deltas without materialising them (no allocation). For
    /// candidate intersection, where positions are never consulted.
    pub fn next_docid(&mut self) -> Option<u32> {
        loop {
            if self.done {
                return None;
            }
            if self.pos >= self.block_end {
                if !self.load_block_header() {
                    return None;
                }
                continue;
            }
            let delta = read_varint(self.buf, &mut self.pos)?;
            let docid = self.prev_docid.checked_add(delta as u32)?;
            self.prev_docid = docid;
            let count = read_varint(self.buf, &mut self.pos)? as usize;
            for _ in 0..count {
                read_varint(self.buf, &mut self.pos)?;
            }
            return Some(docid);
        }
    }

    /// Skip blocks while their last docid is strictly below `target`,
    /// then scan within the block. Positions the cursor so the next
    /// `next_entry()` returns the first entry with docid >= target.
    /// Returns that entry, or None.
    pub fn seek(&mut self, target: u32) -> Option<(u32, Vec<u32>)> {
        if self.done {
            return None;
        }
        // Block-level skip (only valid if we haven't passed the block already).
        loop {
            if self.done {
                return None;
            }
            if self.last_docid_in_block >= target {
                break;
            }
            self.pos = self.block_end;
            if !self.load_block_header() {
                return None;
            }
        }
        // Within-block scan: we may have already consumed some entries;
        // prev_docid tracks position. Entries already consumed with
        // docid >= target cannot be revisited (forward-only) — callers use
        // seek only with monotonically increasing targets.
        loop {
            let entry = self.next_entry()?;
            if entry.0 >= target {
                return Some(entry);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doc_postings_decode_as_empty_positions() {
        let ids: Vec<u32> = (0..300).map(|i| i * 7 + 3).collect();
        let enc = encode_doc_postings(&ids);
        // pos_count 0: 2 bytes per doc + a header and an absolute first
        // docid per 128-doc block
        assert!(enc.len() <= ids.len() * 2 + 3 * 12, "{}", enc.len());
        let dec = decode_postings(&enc);
        assert_eq!(dec.iter().map(|(d, _)| *d).collect::<Vec<_>>(), ids);
        assert!(dec.iter().all(|(_, p)| p.is_empty()));
        let mut c = PostingCursor::new(&enc);
        assert_eq!(c.seek(ids[200]), Some((ids[200], Vec::new())));
    }

    #[test]
    fn varint_roundtrip() {
        for v in [0u64, 1, 127, 128, 255, 300, 16384, u32::MAX as u64, u64::MAX >> 1] {
            let mut buf = Vec::new();
            write_varint(&mut buf, v);
            let mut pos = 0;
            assert_eq!(read_varint(&buf, &mut pos), Some(v));
            assert_eq!(pos, buf.len());
        }
    }

    #[test]
    fn postings_roundtrip() {
        let docs: Vec<(u32, Vec<u32>)> = (0..1000u32)
            .map(|i| (i * 3, vec![i * 10, i * 10 + 5, i * 10 + 900]))
            .collect();
        let enc = encode_postings(&docs);
        let dec = decode_postings(&enc);
        assert_eq!(docs, dec);
    }

    #[test]
    fn postings_empty_and_single() {
        assert_eq!(decode_postings(&encode_postings(&[])), Vec::<(u32, Vec<u32>)>::new());
        let one = vec![(42u32, vec![0u32, 1, 2])];
        assert_eq!(decode_postings(&encode_postings(&one)), one);
    }

    #[test]
    fn seek_skips_blocks() {
        let docs: Vec<(u32, Vec<u32>)> = (0..1000u32).map(|i| (i, vec![i])).collect();
        let enc = encode_postings(&docs);
        let mut c = PostingCursor::new(&enc);
        assert_eq!(c.seek(500).unwrap().0, 500);
        assert_eq!(c.seek(750).unwrap().0, 750);
        assert_eq!(c.next_entry().unwrap().0, 751);
        assert!(c.seek(2000).is_none());
    }

    #[test]
    fn seek_between_docids() {
        let docs = vec![(10u32, vec![1u32]), (20, vec![2]), (30, vec![3])];
        let enc = encode_postings(&docs);
        let mut c = PostingCursor::new(&enc);
        assert_eq!(c.seek(15).unwrap().0, 20);
    }
}
