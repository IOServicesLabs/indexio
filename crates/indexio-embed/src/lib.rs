//! indexio-embed: semantic/vector sidecar (SPEC-P2 §2).
//!
//! Modules:
//! - [`embed`]   — pluggable embedders (`HashEmbedder` offline default,
//!   `HttpEmbedder` for OpenAI-compatible endpoints).
//! - [`store`]   — org-wide content-addressed embedding cache (CAS).
//! - [`index`]   — binary-quantized flat vector index sidecar (`.civec`).
//! - [`hnsw`]    — deterministic HNSW ANN graph sidecar (`.cihnsw`, SPEC-P3 §1).
//! - [`pipeline`]— chunk → embed (CAS-deduped) → rebuild vec index.
//! - [`rerank`] — reranker stage after RRF fusion (SPEC-P3 §2): offline
//!   `OverlapReranker`, `NoopReranker`, and `HttpReranker` for
//!   TEI/vLLM/Jina-style `/rerank` endpoints.
//! - [`rindex`] — self-contained CPU-only random-indexing semantic model
//!   (SPEC-P4 §2, SPEC-P5 B1/B2); the default offline embedder (`rindex-v2`).
//! - [`bm25`] — chunk-level BM25F lexical sidecar (SPEC-P5 §A1), aligned
//!   row-for-row with the vector index of the same namespace.
//!
//! Unsafe policy: `#![deny(unsafe_code)]` with exactly one `#[allow]` — the
//! read-only mapping of `.civec` files in `index::mapped` (SPEC-P6 perf),
//! under the same immutability argument as indexio-index's shard mapping.
//!
//! Documented deviations from SPEC-P2:
//! - Tombstones trail the file as a length-prefixed bincode sorted
//!   `Vec<u32>` rather than a `RoaringBitmap`, avoiding a new dependency;
//!   the sidecar is rebuildable so the format is internal to this crate.

#![allow(clippy::type_complexity)]
#![deny(unsafe_code)]

pub mod bm25;
pub mod embed;
pub mod hnsw;
pub mod index;
pub mod pipeline;
pub mod rerank;
pub mod rimodel;
pub mod rindex;
pub mod store;
