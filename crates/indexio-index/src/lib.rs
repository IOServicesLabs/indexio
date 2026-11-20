//! indexio-index: immutable single-file shard format — writer, mmap reader,
//! in-place tombstones, ShardSet fan-out and compaction (merge).
//!
//! Byte format:
//!   - magic "CIDXSHD1", 64-byte little-endian header, 24-byte directory
//!     entries, sections 8-byte aligned, kinds 1..=11.
//!   - DOCS records:
//!     (blob[16], u32 repo_id, u32 path_off, u32 path_len, u16 lang,
//!     u16 flags, u64 content_off, u32 content_len, u32 raw_len); those
//!     fields sum to 48 bytes, so DOC_REC_LEN is 48.
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

pub use set::{is_abandoned, ShardSet, Visibility};
pub use shard::Shard;
pub use writer::ShardWriter;

// Re-exported: it appears in the public API (Shard::common_grams).
pub use indexio_core::grams::CommonGrams;
pub use indexio_types::codec::PostingCursor;

use serde::{Deserialize, Serialize};

/// META section payload: UTF-8 JSON.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardMeta {
    pub repos: Vec<String>,
    pub doc_count: u64,
    pub total_raw_bytes: u64,
    /// Always empty. Shards up to 0.1.14 stored the top-10k grams by document
    /// frequency here (up to 100 kB of JSON per shard, 10k Strings in every
    /// process that opened it) and nothing but tests read them. The key stays
    /// in the JSON because 0.1.14 refuses a META without it: it is written
    /// as `[]` and whatever an older shard carries is skipped on read.
    /// [`Shard::gram_stats`] computes the list from the postings on demand.
    #[serde(default, deserialize_with = "skip_value")]
    pub gram_stats: Vec<(String, u64)>,
    /// Trained zstd dictionary, base64-encoded in JSON. Present iff the
    /// shard was built from >100 docs and training succeeded.
    #[serde(default, with = "b64_opt")]
    pub zstd_dict: Option<Vec<u8>>,
    /// ISO-8601 UTC creation time.
    pub created: String,
}

/// Deserialize a value by skipping it (nothing is allocated for it).
fn skip_value<'de, D: serde::Deserializer<'de>, T: Default>(d: D) -> Result<T, D::Error> {
    serde::de::IgnoredAny::deserialize(d)?;
    Ok(T::default())
}

/// base64 serde for `Option<Vec<u8>>` (zstd_dict is base64 in JSON).
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

#[cfg(test)]
mod tests {
    use super::*;

    /// META as 0.1.14 declares it: `gram_stats` is a required key.
    #[derive(Deserialize)]
    struct Meta0114 {
        repos: Vec<String>,
        doc_count: u64,
        total_raw_bytes: u64,
        gram_stats: Vec<(String, u64)>,
        #[serde(default, with = "b64_opt")]
        zstd_dict: Option<Vec<u8>>,
        created: String,
    }

    #[test]
    fn gram_stats_of_an_older_shard_are_skipped_on_read() {
        let json = r#"{"repos":["a"],"doc_count":3,"total_raw_bytes":9,"gram_stats":[["abc",3],["bcd",2]],"zstd_dict":"AAEC","created":"2026-01-01T00:00:00Z"}"#;
        let m: ShardMeta = serde_json::from_str(json).unwrap();
        assert!(m.gram_stats.is_empty());
        assert_eq!((m.repos.len(), m.doc_count, m.total_raw_bytes), (1, 3, 9));
        assert_eq!(m.zstd_dict.as_deref(), Some(&[0u8, 1, 2][..]));
        // a META without the key reads too
        let m: ShardMeta = serde_json::from_str(r#"{"repos":[],"doc_count":0,"total_raw_bytes":0,"created":""}"#).unwrap();
        assert!(m.gram_stats.is_empty() && m.zstd_dict.is_none());
    }

    #[test]
    fn meta_written_now_reads_as_0_1_14_expects() {
        let meta = ShardMeta {
            repos: vec!["a".into(), "b".into()],
            doc_count: 7,
            total_raw_bytes: 1234,
            gram_stats: Vec::new(),
            zstd_dict: Some(vec![9, 8, 7]),
            created: "2026-11-20T00:00:00Z".into(),
        };
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains(r#""gram_stats":[]"#), "{json}");
        let old: Meta0114 = serde_json::from_str(&json).unwrap();
        assert_eq!(old.repos, meta.repos);
        assert_eq!((old.doc_count, old.total_raw_bytes), (7, 1234));
        assert!(old.gram_stats.is_empty());
        assert_eq!(old.zstd_dict, meta.zstd_dict);
        assert_eq!(old.created, meta.created);
    }
}
