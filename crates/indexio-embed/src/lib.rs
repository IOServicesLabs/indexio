//! indexio-embed: semantic/vector sidecar.
//!
//! Modules:
//! - [`embed`]   — pluggable embedders (`HashEmbedder` offline default,
//!   `HttpEmbedder` for OpenAI-compatible endpoints).
//! - [`store`]   — org-wide content-addressed embedding cache (CAS).
//! - [`index`]   — binary-quantized flat vector index sidecar (`.civec`).
//! - [`hnsw`]    — deterministic HNSW ANN graph sidecar (`.cihnsw`).
//! - [`pipeline`]— chunk → embed (CAS-deduped) → rebuild vec index.
//! - [`rerank`] — reranker stage after RRF fusion: offline
//!   `OverlapReranker`, `NoopReranker`, and `HttpReranker` for
//!   TEI/vLLM/Jina-style `/rerank` endpoints.
//! - [`rindex`] — self-contained CPU-only random-indexing semantic model
//!; the default offline embedder (`rindex-v2`).
//! - [`bm25`] — chunk-level BM25F lexical sidecar, aligned
//!   row-for-row with the vector index of the same namespace.
//!
//! Unsafe policy: `#![deny(unsafe_code)]` with exactly one `#[allow]` — the
//! read-only mapping of `.civec` files in `index::mapped`,
//! under the same immutability argument as indexio-index's shard mapping.
//!
//! Tombstones trail the file as a length-prefixed bincode sorted
//! `Vec<u32>` rather than a `RoaringBitmap`, avoiding a new dependency;
//! the sidecar is rebuildable so the format is internal to this crate.

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
