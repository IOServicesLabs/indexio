//! indexio-index: immutable single-file shard format — writer, mmap reader,
//! in-place tombstones, ShardSet fan-out and compaction (merge).
//!
//! Byte format contract: docs/SPEC.md, section "indexio-index".
//!   - magic "CIDXSHD1", 64-byte little-endian header, 24-byte directory
//!     entries, sections 8-byte aligned, kinds 1..=11.
//!   - DOCS records: SPEC lists the fields explicitly
//!     (blob[16], u32 repo_id, u32 path_off, u32 path_len, u16 lang,
//!     u16 flags, u64 content_off, u32 content_len, u32 raw_len); those
//!     fields sum to 48 bytes, so DOC_REC_LEN is 48 (SPEC's "40B" label is
//!     an arithmetic slip; the explicit field list is authoritative).
//!   - content_off in a doc record is relative to the start of the CONTENT
//!     section. path_off/scope_off/caller_off are relative to the STRINGS
//!     blob (which starts after the repo table).
//!   - scope/caller strings are stored NUL-terminated in the STRINGS blob
//!     (their posting entries carry only an offset, no length).
//!   - TOMBSTONES is pre-sized to 64KiB: portable-serialized roaring bitmap
//!     at offset 0, zero-padded.

#![allow(clippy::type_complexity)]
#![deny(unsafe_code)]

mod format;
mod set;
mod shard;
mod writer;

pub use set::ShardSet;
pub use shard::Shard;
pub use writer::ShardWriter;

// Re-exported: it appears in the public API (Shard::common_grams).
pub use indexio_core::grams::CommonGrams;
pub use indexio_types::codec::PostingCursor;

use serde::{Deserialize, Serialize};

/// META section payload: UTF-8 JSON (SPEC indexio-index).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardMeta {
    pub repos: Vec<String>,
    pub doc_count: u64,
    pub total_raw_bytes: u64,
    /// Top-10k grams by document frequency: (gram as lossy UTF-8, doc_count).
    pub gram_stats: Vec<(String, u64)>,
    /// Trained zstd dictionary, base64-encoded in JSON. Present iff the
    /// shard was built from >100 docs and training succeeded.
    #[serde(default, with = "b64_opt")]
    pub zstd_dict: Option<Vec<u8>>,
    /// ISO-8601 UTC creation time.
    pub created: String,
}

/// base64 serde for `Option<Vec<u8>>` (SPEC: zstd_dict is base64 in JSON).
mod b64_opt {
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Option<Vec<u8>>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            None => s.serialize_none(),
            Some(bytes) => s.serialize_some(&STANDARD.encode(bytes)),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<u8>>, D::Error> {
        let opt: Option<String> = Option::deserialize(d)?;
        opt.map(|s| STANDARD.decode(s).map_err(serde::de::Error::custom))
            .transpose()
    }
}

pub(crate) fn invalid(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.into())
}
