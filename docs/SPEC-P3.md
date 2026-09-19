# SPEC-P3 — Scale-out & Production Hardening Addendum

Addendum to SPEC.md / SPEC-P2.md. All prior engineering rules apply. **No model weights or
model downloads anywhere in P3** — all model-touching paths are HTTP clients against
external endpoints, tested with local mock servers only. New external deps: none allowed.

P3 scope (the three named non-goals from P2, now implemented):
1. **HNSW ANN layer** in indexio-embed (pure Rust, in-crate, no deps) — scale-out path for the
   flat+BQ vector index at 1M+ vectors.
2. **Reranker stage** after RRF fusion — `Reranker` trait in indexio-embed with three impls
   (offline overlap, noop, HTTP client for TEI/vLLM/Jina-style `/rerank`), wired through
   indexio-query into CLI/HTTP/MCP.
3. **HTTP auth + repo-level ACLs** on `indexio serve` — bearer token auth, per-token repo
   allowlists, admin-route gating. MCP (local stdio) stays trusted-by-design; document.

---

## 1. indexio-embed::hnsw (Agent H)

In-crate HNSW (Hierarchical Navigable Small World) graph over the f32 vectors already
stored in `.civec`. Rebuildable sidecar — deleting it only costs a rebuild.

```rust
pub mod hnsw {
    pub struct HnswOptions { pub m: usize, pub m0: usize, pub ef_construction: usize, pub threshold: usize }
    // defaults: m=16, m0=32, ef_construction=200, threshold=50_000
    impl Default for HnswOptions { ... }

    /// On-disk: <data_dir>/vec/<model_id>.cihnsw
    ///   magic "CIHNSW1" (8B) | u32 n_nodes | u32 m | u32 m0 | u32 entry_point
    ///   | per node: u8 level, then per level l=0..=level: u32 nbr_count + nbr_count×u32
    pub struct Hnsw { /* in-memory graph */ }
    impl Hnsw {
        /// Build from row vectors (row i = node i). Deterministic: level assignment uses a
        /// xorshift64 PRNG seeded from (n, dim) — no rand dep, reproducible builds.
        pub fn build(vectors: &[Vec<f32>], opts: &HnswOptions) -> Self;
        pub fn save(&self, path: &Path) -> anyhow::Result<()>;
        pub fn load(path: &Path) -> anyhow::Result<Self>;
        /// Approximate top-`ef` candidate row ids for query (L2-normalized dot = cosine).
        pub fn search(&self, vectors: &[Vec<f32>], q: &[f32], ef: usize) -> Vec<u32>;
    }
}
```

Integration into `index::VecIndex` (additive, public API unchanged):
- `IndexOptions { pub hnsw_threshold: usize }` with `Default` (50_000);
  `VecIndex::create_with_options(dir, model_id, dim, rows, opts)`; existing `create`
  delegates with defaults.
- On `create*`: if `rows.len() >= threshold`, build + save `.cihnsw` alongside `.civec`.
- On `open`: if `.cihnsw` exists, load it.
- `search`: if HNSW present → `hnsw.search(q, ef = max(k*10, 100))` → exact f32 rescore of
  candidates → top k (skip tombstoned). Else → existing BQ prescan path (unchanged).

Tests (≥5): recall@10 ≥ 0.95 vs exact brute-force top-10 on 5,000 random 128-dim
L2-normalized vectors (deterministic xorshift PRNG); persistence round-trip (save/load →
identical search results); below-threshold index uses flat path (no .cihnsw written);
above-threshold create → .cihnsw exists and search returns the planted nearest neighbor;
tombstoned rows never returned via HNSW path.

## 2. indexio-embed::rerank + indexio-query wiring (Agent R)

```rust
pub mod rerank {
    pub trait Reranker: Send + Sync {
        fn model_id(&self) -> &str;
        /// Score docs against query. Returns (doc_index, score) sorted by score desc.
        fn rerank(&self, query: &str, docs: &[String]) -> anyhow::Result<Vec<(usize, f64)>>;
    }
    /// Offline deterministic reranker: token overlap between query and doc, weighted by
    /// sqrt(1/(1+doc_token_count)) length normalization + bigram bonus. L2-free, no deps.
    pub struct OverlapReranker;
    /// Identity order (pass-through), for A/B and tests.
    pub struct NoopReranker;
    /// TEI/vLLM/Jina-style: POST {base}/rerank {"model":..,"query":..,"documents":[..]}
    /// → {"results":[{"index":i,"relevance_score":s},...]} via ureq, bearer if key set.
    pub struct HttpReranker { base: String, model: String, api_key: String }
    impl HttpReranker { pub fn from_env() -> anyhow::Result<Option<Self>>; }
    // Some(INDEXIO_RERANK_BASE set; requires INDEXIO_RERANK_MODEL) else None
}
```

indexio-query (additive):
```rust
impl Engine {
    /// Hybrid then rerank: RRF top max(limit*3, 30) → doc text = "path\n"+snippet →
    /// reranker → reorder (ties keep RRF order) → top limit. HybridHit gains
    /// `rerank_score: Option<f64>` (None when no reranker).
    pub fn search_hybrid_reranked(&self, q: &str, limit: usize,
        embedder: &dyn indexio_embed::embed::Embedder, reranker: &dyn indexio_embed::rerank::Reranker)
        -> anyhow::Result<Vec<HybridHit>>;
}
```

crates/indexio surface:
- CLI: `indexio search <q> --mode hybrid --rerank` (flag; only valid with hybrid mode — error
  otherwise). Reranker selection: `HttpReranker::from_env()?` if Some, else OverlapReranker.
- HTTP: `/search?q=&mode=hybrid&rerank=1` (same selection, built once in AppState).
- MCP: `code_search` gains optional `rerank: boolean` (hybrid mode only; ignored otherwise
  with a note in the result? No — return -32602 invalid params if rerank=true and mode !=
  hybrid). Update tool description.

Tests (≥5): OverlapReranker ranks term-containing doc above distractor + length
normalization sanity; NoopReranker identity; HttpReranker against local TcpListener mock
(request shape asserted, scores parsed, reorder applied); search_hybrid_reranked on a
2-repo fixture reorders a constructed RRF list (rerank favorite moves to #1) and fills
rerank_score; CLI flag validation (`--rerank` with `--mode lexical` errors); MCP
rerank=true + mode=lexical → -32602.

## 3. HTTP auth + repo ACLs (also Agent R — single owner for crates/indexio)

crates/indexio `serve.rs` + `main.rs` only (axum middleware + clap flags):

- `indexio serve --auth-token T` or env `INDEXIO_AUTH_TOKEN`: all routes except `GET /health`
  require `Authorization: Bearer T`; else 401 `{"error":"unauthorized"}`.
- `indexio serve --acl-file PATH` (supersedes --auth-token): JSON
  `{"tokens":{"tokA":{"allow":["core","team-*"]},"tokB":{"allow":["*"]}}}`.
  Token must exist in map (else 401). Hit filtering: after search/symbol/calls, drop
  SearchHits whose `repo` doesn't match any allow pattern. Pattern language: `*` matches
  anything; `prefix-*` prefix glob; `*-suffix` suffix glob; exact otherwise. Implement a
  small `fn repo_allowed(patterns: &[String], repo: &str) -> bool` — no glob crate.
- Admin routes (`POST /admin/reload`, `POST /admin/embed`): require a token whose allow
  contains exactly `*`; else 403 `{"error":"forbidden"}`.
- Both flags optional; neither → current open behavior. MCP unchanged (stdio = trusted
  local user) — add a rustdoc/README note.

Tests (≥5): no-auth baseline still open; wrong/missing token → 401, right token → 200;
ACL filters repos out of /search results (2-repo fixture); prefix glob + `*` cases for
repo_allowed (pure unit); admin route 403 with non-`*` token, 200 with `*` token.

## 4. Non-goals for P3 (unchanged)

No SCIP/LSIF, no CodeQL/CPG, no in-process model inference, no MCP auth (documented),
no web UI, no distributed query fan-out.
