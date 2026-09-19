//! On-disk layout constants and header encode/decode (SPEC indexio-index).

use std::io;

use crate::invalid;

pub const MAGIC: &[u8; 8] = b"CIDXSHD1";
pub const VERSION: u32 = 1;
pub const HEADER_LEN: usize = 64;
pub const DIR_ENTRY_LEN: usize = 24;
pub const SECTION_ALIGN: u64 = 8;
/// TOMBSTONES section is pre-sized to 64KiB (SPEC).
pub const TOMBSTONES_LEN: usize = 64 * 1024;

/// Section kinds (SPEC).
pub mod kind {
    pub const DOCS: u32 = 1;
    pub const STRINGS: u32 = 2;
    pub const CONTENT: u32 = 3;
    pub const NGRAM_FST: u32 = 4;
    pub const NGRAM_POST: u32 = 5;
    pub const SYM_FST: u32 = 6;
    pub const SYM_POST: u32 = 7;
    pub const CALL_FST: u32 = 8;
    pub const CALL_POST: u32 = 9;
    pub const TOMBSTONES: u32 = 10;
    pub const META: u32 = 11;
    /// Upper bound for indexing the section table by kind.
    pub const COUNT: usize = 12;
}

/// Doc record size. SPEC's explicit field list
/// (16+4+4+4+2+2+8+4+4) sums to 48 bytes; see crate docs.
pub const DOC_REC_LEN: usize = 48;

pub fn align8(v: u64) -> u64 {
    (v + SECTION_ALIGN - 1) & !(SECTION_ALIGN - 1)
}

pub fn get_u16(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(buf[off..off + 2].try_into().unwrap())
}
pub fn get_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}
pub fn get_u64(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}
pub fn put_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}
pub fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
pub fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub struct Header {
    pub section_count: u32,
    pub doc_count: u64,
    pub created_unix: u64,
    pub flags: u32,
}

pub fn encode_header(h: &Header) -> [u8; HEADER_LEN] {
    let mut out = [0u8; HEADER_LEN];
    out[0..8].copy_from_slice(MAGIC);
    out[8..12].copy_from_slice(&VERSION.to_le_bytes());
    out[12..16].copy_from_slice(&h.section_count.to_le_bytes());
    out[16..24].copy_from_slice(&h.doc_count.to_le_bytes());
    out[24..32].copy_from_slice(&h.created_unix.to_le_bytes());
    out[32..36].copy_from_slice(&h.flags.to_le_bytes());
    // [36..40) header_reserved = 0
    // [40..48) common_grams_off = 0 (none: CommonGrams derive from META)
    // [48..56) common_grams_len = 0
    out
}

pub fn decode_header(buf: &[u8]) -> io::Result<Header> {
    if buf.len() < HEADER_LEN {
        return Err(invalid("shard: file smaller than 64-byte header"));
    }
    if &buf[0..8] != MAGIC {
        return Err(invalid("shard: bad magic (not CIDXSHD1)"));
    }
    let version = get_u32(buf, 8);
    if version != VERSION {
        return Err(invalid(format!("shard: unsupported version {version}")));
    }
    Ok(Header {
        section_count: get_u32(buf, 12),
        doc_count: get_u64(buf, 16),
        created_unix: get_u64(buf, 24),
        flags: get_u32(buf, 32),
    })
}

/// Format unix seconds as ISO-8601 UTC ("YYYY-MM-DDTHH:MM:SSZ").
/// Civil-from-days algorithm (H. Hinnant, public domain).
pub fn unix_to_iso(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}
