//! Shard: mmap reader with in-place tombstone updates.

use std::fs::OpenOptions;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use indexio_core::grams::CommonGrams;
use indexio_types::codec::{read_varint, PostingCursor};
use indexio_types::{BlobId, DocMeta, Lang};
use memmap2::MmapMut;
use roaring::RoaringBitmap;
use tracing::debug;

use crate::format::{self, kind, get_u16, get_u32, get_u64, DOC_REC_LEN, TOMBSTONES_LEN};
use crate::{invalid, ShardMeta};

mod mmap_impl {
    use memmap2::MmapMut;

    /// Map a shard file read-write.
    ///
    /// # Safety (the only unsafe code in indexio-index)
    /// `memmap2` is unsafe because the OS cannot prevent another process
    /// from mutating the file while it is mapped. Shards are immutable
    /// after their atomic rename; the only in-place mutation is our own
    /// `delete_docs` (which writes through this same mapping), so the
    /// mapping cannot be invalidated by design. Callers never hold
    /// references across `delete_docs`.
    #[allow(unsafe_code)]
    pub fn map(file: &std::fs::File) -> std::io::Result<MmapMut> {
        unsafe { MmapMut::map_mut(file) }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DocRec {
    pub blob: [u8; 16],
    pub repo_id: u32,
    pub path_off: u32,
    pub path_len: u32,
    pub lang: u16,
    pub content_off: u64,
    pub content_len: u32,
    pub raw_len: u32,
}

pub struct Shard {
    path: PathBuf,
    mmap: MmapMut,
    doc_count: u64,
    created_unix: u64,
    /// Section (offset, len) indexed by section kind.
    sections: [Option<(usize, usize)>; kind::COUNT],
    /// Absolute offset of the STRINGS blob (after the repo table).
    strings_blob: usize,
    meta: ShardMeta,
    tomb: RoaringBitmap,
}

impl Shard {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let mmap = mmap_impl::map(&file)?;
        let hdr = format::decode_header(&mmap[..])?;
        let dir_len = hdr.section_count as usize * format::DIR_ENTRY_LEN;
        let dir_end = format::HEADER_LEN + dir_len;
        if mmap.len() < dir_end {
            return Err(invalid("shard: truncated section directory"));
        }
        let mut sections: [Option<(usize, usize)>; kind::COUNT] = [None; kind::COUNT];
        for i in 0..hdr.section_count as usize {
            let e = format::HEADER_LEN + i * format::DIR_ENTRY_LEN;
            let k = get_u32(&mmap, e) as usize;
            let off = get_u64(&mmap, e + 8) as usize;
            let len = get_u64(&mmap, e + 16) as usize;
            if off.checked_add(len).is_none_or(|end| end > mmap.len()) {
                return Err(invalid("shard: section out of bounds"));
            }
            if k < kind::COUNT {
                sections[k] = Some((off, len));
            }
        }
        for required in [kind::DOCS, kind::STRINGS, kind::CONTENT, kind::META] {
            if sections[required as usize].is_none() {
                return Err(invalid(format!("shard: missing section kind {required}")));
            }
        }

        let meta: ShardMeta = {
            let s = section_slice_raw(&mmap, &sections, kind::META)
                .ok_or_else(|| invalid("shard: missing META"))?;
            serde_json::from_slice(s).map_err(|e| invalid(format!("shard: bad META json: {e}")))?
        };
        if meta.doc_count != hdr.doc_count {
            return Err(invalid("shard: META doc_count disagrees with header"));
        }
        let tomb = match section_slice_raw(&mmap, &sections, kind::TOMBSTONES) {
            Some(s) => RoaringBitmap::deserialize_from(s)
                .map_err(|e| invalid(format!("shard: bad tombstones: {e}")))?,
            None => RoaringBitmap::new(),
        };
        let strings_blob = {
            let s = section_slice_raw(&mmap, &sections, kind::STRINGS)
                .ok_or_else(|| invalid("shard: missing STRINGS"))?;
            if s.len() < 4 {
                return Err(invalid("shard: STRINGS too small"));
            }
            let repo_count = get_u32(s, 0) as usize;
            let base = 4 + repo_count * 8;
            if s.len() < base {
                return Err(invalid("shard: STRINGS repo table truncated"));
            }
            sections[kind::STRINGS as usize].unwrap().0 + base
        };

        debug!(?path, docs = hdr.doc_count, "shard opened");
        Ok(Shard {
            path: path.to_path_buf(),
            mmap,
            doc_count: hdr.doc_count,
            created_unix: hdr.created_unix,
            sections,
            strings_blob,
            meta,
            tomb,
        })
    }

    pub(crate) fn section_slice(&self, k: u32) -> Option<&[u8]> {
        let (off, len) = self.sections[k as usize]?;
        self.mmap.get(off..off + len)
    }

    pub fn doc_count(&self) -> u64 {
        self.doc_count
    }

    pub(crate) fn created_unix(&self) -> u64 {
        self.created_unix
    }

    pub fn meta(&self) -> &ShardMeta {
        &self.meta
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn rec(&self, docid: u32) -> Option<DocRec> {
        let docs = self.section_slice(kind::DOCS)?;
        let start = docid as usize * DOC_REC_LEN;
        let r = docs.get(start..start + DOC_REC_LEN)?;
        let mut blob = [0u8; 16];
        blob.copy_from_slice(&r[0..16]);
        Some(DocRec {
            blob,
            repo_id: get_u32(r, 16),
            path_off: get_u32(r, 20),
            path_len: get_u32(r, 24),
            lang: get_u16(r, 28),
            content_off: get_u64(r, 32),
            content_len: get_u32(r, 40),
            raw_len: get_u32(r, 44),
        })
    }

    fn string_at(&self, off: u32, len: u32) -> Option<String> {
        let start = self.strings_blob.checked_add(off as usize)?;
        let end = start.checked_add(len as usize)?;
        let bytes = self.mmap.get(start..end)?;
        Some(String::from_utf8_lossy(bytes).into_owned())
    }

    /// Read a NUL-terminated string from the STRINGS blob (scope/caller).
    pub(crate) fn cstr_at(&self, off: u32) -> Option<String> {
        let start = self.strings_blob.checked_add(off as usize)?;
        let rest = self.mmap.get(start..)?;
        let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
        Some(String::from_utf8_lossy(&rest[..end]).into_owned())
    }

    pub fn doc(&self, docid: u32) -> Option<DocMeta> {
        if docid as u64 >= self.doc_count {
            return None;
        }
        let r = self.rec(docid)?;
        let path = self.string_at(r.path_off, r.path_len)?;
        Some(DocMeta {
            blob: BlobId(r.blob),
            repo_id: r.repo_id,
            path,
            lang: Lang::from_u16(r.lang),
            raw_len: r.raw_len,
        })
    }

    pub fn content(&self, docid: u32) -> io::Result<Vec<u8>> {
        let r = self
            .rec(docid)
            .filter(|_| (docid as u64) < self.doc_count)
            .ok_or_else(|| invalid(format!("shard: docid {docid} out of range")))?;
        let c = self
            .section_slice(kind::CONTENT)
            .ok_or_else(|| invalid("shard: missing CONTENT"))?;
        let start = r.content_off as usize;
        let frame = c
            .get(start..start + r.content_len as usize)
            .ok_or_else(|| invalid("shard: content frame out of bounds"))?;
        match &self.meta.zstd_dict {
            None => zstd::stream::decode_all(frame),
            Some(d) => {
                let mut dec = zstd::stream::Decoder::with_dictionary(frame, &d[..])?;
                let mut out = Vec::with_capacity(r.raw_len as usize);
                dec.read_to_end(&mut out)?;
                Ok(out)
            }
        }
    }

    /// Length-prefixed posting payload at `post_off` inside a POST section.
    pub(crate) fn posting_payload(&self, post_kind: u32, post_off: u64) -> Option<&[u8]> {
        let post = self.section_slice(post_kind)?;
        let off = post_off as usize;
        let len_bytes = post.get(off..off.checked_add(4)?)?;
        let len = u32::from_le_bytes(len_bytes.try_into().ok()?) as usize;
        post.get(off + 4..off.checked_add(4)?.checked_add(len)?)
    }

    fn fst_get(&self, fst_kind: u32, key: &[u8]) -> Option<u64> {
        let fst = self.section_slice(fst_kind)?;
        let map = fst::Map::new(fst).ok()?;
        map.get(key)
    }

    /// Byte length of the posting payload for `gram` (0 when absent): a
    /// decode-free proxy for its document frequency (SPEC-P9 planner).
    pub fn posting_bytes(&self, gram: &[u8]) -> usize {
        let Some(off) = self.fst_get(kind::NGRAM_FST, gram) else {
            return 0;
        };
        self.posting_payload(kind::NGRAM_POST, off).map_or(0, |p| p.len())
    }

    pub fn postings(&self, gram: &[u8]) -> Option<PostingCursor<'_>> {
        let off = self.fst_get(kind::NGRAM_FST, gram)?;
        let payload = self.posting_payload(kind::NGRAM_POST, off)?;
        Some(PostingCursor::new(payload))
    }

    pub fn symbol_postings(&self, name: &str) -> Vec<(u32, u8, u32, String)> {
        let mut out = Vec::new();
        let Some(off) = self.fst_get(kind::SYM_FST, name.as_bytes()) else {
            return out;
        };
        let Some(payload) = self.posting_payload(kind::SYM_POST, off) else {
            return out;
        };
        for (docid, line, kind, scope_off) in decode_sym_payload(payload) {
            let scope = self.cstr_at(scope_off).unwrap_or_default();
            out.push((docid, kind, line, scope));
        }
        out
    }

    pub fn call_postings(&self, callee: &str) -> Vec<(u32, String, u32)> {
        let mut out = Vec::new();
        let Some(off) = self.fst_get(kind::CALL_FST, callee.as_bytes()) else {
            return out;
        };
        let Some(payload) = self.posting_payload(kind::CALL_POST, off) else {
            return out;
        };
        for (docid, caller_off, line) in decode_call_payload(payload) {
            let caller = self.cstr_at(caller_off).unwrap_or_default();
            out.push((docid, caller, line));
        }
        out
    }

    pub fn tombstones(&self) -> &RoaringBitmap {
        &self.tomb
    }

    /// The tombstone bitmap as it is on disk RIGHT NOW (the mapping is
    /// shared, so another process's `delete_docs` shows up here), unlike
    /// [`tombstones`](Self::tombstones), which is the bitmap read at open.
    pub fn tombstones_on_disk(&self) -> RoaringBitmap {
        section_slice_raw(&self.mmap, &self.sections, kind::TOMBSTONES)
            .and_then(|s| RoaringBitmap::deserialize_from(s).ok())
            .unwrap_or_else(|| self.tomb.clone())
    }

    pub fn common_grams(&self) -> CommonGrams {
        let stats: Vec<(Vec<u8>, u64)> = self
            .meta
            .gram_stats
            .iter()
            .map(|(g, c)| (g.clone().into_bytes(), *c))
            .collect();
        CommonGrams::from_stats(&stats, self.doc_count, 0.005)
    }

    /// Rewrite the TOMBSTONES section in place (pre-sized 64KiB; error if
    /// the serialized bitmap no longer fits).
    pub fn delete_docs(&mut self, docids: &[u32]) -> io::Result<()> {
        // Several processes tombstone docs of one shard (each agent session
        // refreshes its own repo; a sync touches them all). The mapping is
        // shared, so the section holds whatever was written last: union
        // with it instead of with the bitmap read at open, or a
        // concurrent writer's deletions would be silently undone.
        let mut bm = match section_slice_raw(&self.mmap, &self.sections, kind::TOMBSTONES)
            .and_then(|s| RoaringBitmap::deserialize_from(s).ok())
        {
            Some(on_disk) => on_disk | &self.tomb,
            None => self.tomb.clone(),
        };
        for &d in docids {
            bm.insert(d);
        }
        let mut bytes = Vec::with_capacity(self.tomb.serialized_size().max(64));
        bm.serialize_into(&mut bytes)?;
        if bytes.len() > TOMBSTONES_LEN {
            return Err(invalid(format!(
                "shard: tombstone bitmap {} bytes exceeds 64KiB section",
                bytes.len()
            )));
        }
        let (off, len) = self.sections[kind::TOMBSTONES as usize]
            .ok_or_else(|| invalid("shard: missing TOMBSTONES section"))?;
        if len < TOMBSTONES_LEN || off.checked_add(TOMBSTONES_LEN).is_none_or(|e| e > self.mmap.len()) {
            return Err(invalid("shard: TOMBSTONES section smaller than 64KiB"));
        }
        self.mmap[off..off + bytes.len()].copy_from_slice(&bytes);
        for b in &mut self.mmap[off + bytes.len()..off + TOMBSTONES_LEN] {
            *b = 0;
        }
        self.mmap.flush_range(off, TOMBSTONES_LEN)?; // msync
        self.tomb = bm;
        Ok(())
    }

    /// Stream all entries of an FST section as (key, post_off). Used by merge.
    pub(crate) fn fst_entries(&self, fst_kind: u32) -> Vec<(Vec<u8>, u64)> {
        use fst::Streamer;
        let mut out = Vec::new();
        let Some(fst) = self.section_slice(fst_kind) else {
            return out;
        };
        let Ok(map) = fst::Map::new(fst) else {
            return out;
        };
        let mut stream = map.stream();
        while let Some((k, v)) = stream.next() {
            out.push((k.to_vec(), v));
        }
        out
    }
}

fn section_slice_raw<'a>(
    mmap: &'a [u8],
    sections: &[Option<(usize, usize)>; kind::COUNT],
    k: u32,
) -> Option<&'a [u8]> {
    let (off, len) = sections[k as usize]?;
    mmap.get(off..off + len)
}

/// Decode a SYM payload: entries varint(doc_delta), varint(line), u8 kind,
/// varint(scope_off). Stops cleanly on truncation.
pub(crate) fn decode_sym_payload(payload: &[u8]) -> Vec<(u32, u32, u8, u32)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut prev = 0u32;
    while pos < payload.len() {
        let Some(delta) = read_varint(payload, &mut pos) else { break };
        prev = prev.wrapping_add(delta as u32);
        let Some(line) = read_varint(payload, &mut pos) else { break };
        let Some(&k) = payload.get(pos) else { break };
        pos += 1;
        let Some(scope_off) = read_varint(payload, &mut pos) else { break };
        out.push((prev, line as u32, k, scope_off as u32));
    }
    out
}

/// Decode a CALL payload: entries varint(doc_delta), varint(caller_off),
/// varint(line).
pub(crate) fn decode_call_payload(payload: &[u8]) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut prev = 0u32;
    while pos < payload.len() {
        let Some(delta) = read_varint(payload, &mut pos) else { break };
        prev = prev.wrapping_add(delta as u32);
        let Some(caller_off) = read_varint(payload, &mut pos) else { break };
        let Some(line) = read_varint(payload, &mut pos) else { break };
        out.push((prev, caller_off as u32, line as u32));
    }
    out
}
