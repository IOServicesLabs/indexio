# SPEC-P2 — Semantic Plane Addendum

Addendum to SPEC.md. All SPEC.md engineering rules still apply (workspace layout, pinned deps,
`unsafe` policy, CARGO_BUILD_JOBS=2, commit conventions, additive-only changes to other crates).
P2 adds the semantic/vector sidecar: chunking → embedding (with org-wide CAS dedup) →
quantized vector index → hybrid (RRF) fusion in indexio-query → CLI/HTTP/MCP surface. Plus a
GitHub org ingestion command.

Embedding weights CANNOT be downloaded in this environment. Therefore the embedder is
provider-pluggable: a deterministic built-in `HashEmbedder` (default; offline; used for all
tests and benchmarks) and an `HttpEmbedder` that calls any OpenAI-compatible `/v1/embeddings`
endpoint (production path: self-hosted Qwen3-Embedding-8B/0.6B via vLLM or TEI). No model
weights are fetched at build or test time. Do NOT add candle/onnxruntime/hnsw_rs deps.

## New/changed crates

- `crates/indexio-symbols` — ADDITIVE: new public `chunking` module (below).
- `crates/indexio-embed` — NEW crate: embedders, embedding CAS, vector index, embed pipeline.
- `crates/indexio-ingest` — ADDITIVE: `org_sync` module (GitHub org listing + clone + index loop).
- `crates/indexio-query` — ADDITIVE: hybrid/semantic search + RRF fusion (Wave B).
- `crates/indexio` — ADDITIVE: `embed`, `org-sync` commands; `--mode` on search; HTTP/MCP surface (Wave B).

Workspace Cargo.toml: add `crates/indexio-embed` to members. New pinned deps allowed:
`ureq = "2"` (with "json" feature) for HTTP; no other new external deps without need.
indexio-embed depends on: indexio-types, indexio-core, indexio-index, indexio-symbols, serde, bincode 1.3, blake3,
memmap2, anyhow, rayon (already allowed? if not pinned, pin `rayon = "1"`).

---

## 1. indexio-symbols::chunking (Agent A)

cAST-lite chunker: split files at syntax boundaries using the tree-sitter parsers already in
this crate. Public API:

```rust
pub mod chunking {
    use indexio_types::Lang;

    /// One semantic chunk of a file.
    #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    pub struct ChunkRec {
        pub start_line: u32,   // 1-based, inclusive
        pub end_line: u32,     // 1-based, inclusive
        pub header: String,    // e.g. "src/parser.rs > mod query > fn parse"
        pub text: String,      // header + "\n" + source slice
    }

    /// Chunk `content` into semantic pieces of at most `max_chars` source chars each
    /// (header not counted). Files shorter than `max_chars` yield exactly one chunk whose
    /// header is the path. Returns empty vec for empty/whitespace-only files.
    /// Unknown/unsupported languages fall back to line-window chunking (below).
    pub fn chunks(lang: Lang, path: &str, content: &[u8], max_chars: usize) -> Vec<ChunkRec>;
}
```

Algorithm (supported languages):
1. Parse with the existing per-language parser (reuse the OnceLock parser infra; add a
   function to get a `&'static`-backed tree-sitter Parser per Lang — refactor internals only,
   no public API breakage).
2. Walk top-level def nodes (reuse each language's def-node kinds already used by `extract`):
   - Nodes whose text ≤ max_chars become one chunk. Header: `path > scope-chain > kind name`
     where scope-chain comes from ancestor def nodes (same logic as symbol scopes).
   - Nodes whose text > max_chars: recurse into child def nodes; if a node is still too big at
     leaf level, split by line windows of ≤ max_chars with ~10% overlap, header gets `#partN`.
   - Runs of small adjacent top-level non-def siblings (imports, comments, consts < 200 chars
     each) are merged into a single "preamble" chunk up to max_chars.
3. Fallback for Lang::Unknown or parse failure: sliding line window, chunks of ≤ max_chars,
   10% line overlap, header = `path > lines A-B`.
4. Every chunk's `text` = `header + "\n" + source-slice`. Empty results for empty files.

Tests (≥6): rust file with several fns (each fn its own chunk, headers contain path and fn
name); huge fn > max_chars splits with `#partN` and overlap; python class methods carry
class scope in header; tiny file → single chunk; unknown language → window fallback;
empty file → no chunks; max_chars respected (no chunk source-slice exceeds it by >10%).

## 2. indexio-embed (Agent B)

### 2.1 Embedders

```rust
pub mod embed {
    pub trait Embedder: Send + Sync {
        fn model_id(&self) -> &str;          // e.g. "hash-v1" or "http:Qwen3-Embedding-8B"
        fn dim(&self) -> usize;
        fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>; // L2-normalized
    }

    /// Deterministic offline embedder. Tokenize on non-alphanumeric, lowercase, hash each
    /// token and each token bigram via blake3[..8] into dim buckets with signed hashing
    /// (+1/-1 by a second hash bit), L2-normalize. dim default 512.
    pub struct HashEmbedder { dim: usize }
    impl HashEmbedder { pub fn new(dim: usize) -> Self; }

    /// OpenAI-compatible endpoint. POST {base}/v1/embeddings {model, input:[...]} via ureq,
    /// bearer `api_key` if non-empty. Batch ≤ 32 texts per request. L2-normalize outputs.
    pub struct HttpEmbedder { base: String, model: String, api_key: String, dim: usize }
    impl HttpEmbedder {
        /// Reads INDEXIO_EMBED_BASE, INDEXIO_EMBED_MODEL, INDEXIO_EMBED_KEY, INDEXIO_EMBED_DIM env vars.
        pub fn from_env() -> anyhow::Result<Self>;
    }

    /// cosine(a,b) for L2-normalized vecs = dot product.
    pub fn dot(a: &[f32], b: &[f32]) -> f32;
}
```

### 2.2 Embedding CAS (org-wide dedup)

```rust
pub mod store {
    /// Content-addressed embedding cache: key = (chunk_hash, model_id) where
    /// chunk_hash = blake3(chunk.text)[..16] hex. Layout:
    ///   <data_dir>/embcas/<model_id>/<hh>/<rest>.bin   (bincode Vec<f32>, tmp+rename atomic)
    pub struct EmbedCas { dir: PathBuf }
    impl EmbedCas {
        pub fn open(data_dir: &Path, model_id: &str) -> Self;
        pub fn get(&self, chunk_hash: &[u8;16]) -> Option<Vec<f32>>;
        pub fn put(&self, chunk_hash: &[u8;16], vec: &[f32]);
        pub fn stats(&self) -> (u64 /*entries*/, u64 /*bytes*/);
    }
}
```

### 2.3 Vector index sidecar (rebuildable, mmap)

Binary-quantized flat index with rescore — the report's P2 architecture. Flat is exact and
plenty at demo scale; the format leaves room for an HNSW layer later (P3 note, not built).

```rust
pub mod index {
    use indexio_types::BlobId;

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    pub struct VecRowMeta {
        pub chunk_hash: [u8;16],
        pub repo: String,
        pub path: String,
        pub start_line: u32,
        pub end_line: u32,
    }

    /// On-disk: <data_dir>/vec/<model_id>.civec
    ///   magic "CIVEC001" (8B) | u32 dim | u32 n_rows | u32 n_deleted
    ///   | bincode Vec<VecRowMeta> (length-prefixed u64)
    ///   | n_rows × dim f32 vectors (aligned block, mmap'd)
    ///   | n_rows × ceil(dim/8) binary codes (sign bits, mmap'd)
    /// Query: hamming-distance prescan on binary codes keeps top (k*8, min 256) candidates,
    /// rescore with exact f32 dot, return top k. Deleted rows (tombstone bitset appended
    /// as trailing bincode RoaringBitmap<u32>) are skipped.
    pub struct VecIndex { /* mmap-backed */ }
    impl VecIndex {
        pub fn create(dir: &Path, model_id: &str, dim: usize,
                      rows: Vec<(VecRowMeta, Vec<f32>)>) -> anyhow::Result<Self>; // full build
        pub fn open(dir: &Path, model_id: &str) -> anyhow::Result<Option<Self>>; // None if absent
        pub fn search(&self, q: &[f32], k: usize) -> Vec<(u32 /*row*/, f32 /*score*/)>;
        pub fn row_meta(&self, row: u32) -> &VecRowMeta;
        pub fn delete_where(&mut self, pred: impl Fn(&VecRowMeta) -> bool) -> u64; // tombstone
        pub fn save_tombstones(&self) -> anyhow::Result<()>;
        pub fn len(&self) -> usize;
    }
}
```

### 2.4 Embed pipeline

```rust
pub mod pipeline {
    use indexio_types::Lang;
    use indexio_index::ShardSet;
    use crate::{embed::Embedder, store::EmbedCas, index::VecIndex};

    #[derive(Clone, Debug, Default)]
    pub struct EmbedReport {
        pub repo: String, pub chunks: u64, pub cas_hits: u64, pub cas_misses: u64,
        pub embedded: u64, pub elapsed_ms: u128,
    }

    /// Chunk every doc of `repo` visible in `shards`, embed chunks missing from the CAS
    /// (batch ≤ 64 texts per Embedder::embed call), then rebuild the vec index for
    /// `model_id` as: all existing rows for OTHER repos + all rows for `repo`
    /// (i.e. repo-scoped replace; simple and correct). max_chars = 1200.
    /// Returns report. Creates/overwrites <data_dir>/vec/<model_id>.civec.
    pub fn embed_repo(shards: &ShardSet, data_dir: &Path, repo: &str,
                      embedder: &dyn Embedder) -> anyhow::Result<EmbedReport>;

    /// Convenience: embed_repo for every repo present in the shard set.
    pub fn embed_all(shards: &ShardSet, data_dir: &Path,
                     embedder: &dyn Embedder) -> anyhow::Result<Vec<EmbedReport>>;
}
```

indexio-embed tests (≥8, HashEmbedder only, no network): hash embedder determinism + dim +
normalization + similar-texts-score-higher-than-dissimilar sanity; CAS put/get/stats;
chunk_hash stability; VecIndex create/open/search returns the planted nearest neighbor first
(distinct planted vectors, k=3, verify ordering); binary-prescan correctness (result equals
brute-force f32 top-k on random small index); delete_where + reopen persistence; embed_repo
on a tiny ShardSet built via indexio-index::ShardWriter (2 repos sharing an identical file →
second repo's embed run shows cas_hits>0 and cas_misses lower); repo-scoped replace
(re-embed repo after change → no stale rows for that repo).

## 3. indexio-ingest::org_sync (Agent C)

```rust
pub mod org_sync {
    #[derive(Clone, Debug)]
    pub struct OrgSyncOptions {
        pub org: String,
        pub token: Option<String>,   // GitHub PAT; None = unauthenticated (60 req/h)
        pub dest_dir: PathBuf,       // repos cloned here as dest_dir/<repo>
        pub include_forks: bool,     // default false
        pub include_archived: bool,  // default false
        pub limit: Option<usize>,    // cap repo count (for trials)
        pub shallow: bool,           // clone --depth 1, default true
    }
    #[derive(Clone, Debug, Default)]
    pub struct OrgSyncReport {
        pub repos_listed: u64, pub repos_cloned: u64, pub repos_skipped: u64,
        pub repos_failed: Vec<(String, String)>, pub indexed: Vec<IndexReport>,
    }

    /// 1. GET https://api.github.com/orgs/{org}/repos?per_page=100&page=N via ureq
    ///    (Authorization: Bearer token if set; User-Agent: "indexio"; follow pages
    ///    until < 100 results). Filter forks/archived per options; apply limit.
    /// 2. For each repo: if dest_dir/<name>/.git exists, run `git -C ... pull --ff-only`
    ///    else `git clone [--depth 1] https_clone_url dest_dir/<name>` (std::process::Command;
    ///    10-min per-repo timeout not required — rely on git). Failures recorded, not fatal.
    /// 3. Index each cloned/updated repo with crate::index_repo (name = repo name).
    pub fn sync_org(opts: &OrgSyncOptions, data_dir: &Path, cas: &crate::Cas)
        -> anyhow::Result<OrgSyncReport>;
}
```

Tests (≥3): URL/pagination parsing against a local mock HTTP server (tiny std::net TcpListener
in the test serving two paginated JSON pages — assert both fetched and merged); filter logic
(forks/archived excluded); skip-existing logic with a pre-created fake .git dir and a stubbed
clone (factor clone into a testable fn taking a `clone_fn: &dyn Fn(&str,&Path)->io::Result<()>`).
Do NOT hit github.com in tests.

## 4. indexio-query + indexio CLI/HTTP/MCP (Wave B — after A, B, C merge)

indexio-query additive API:

```rust
pub struct HybridHit { pub hit: SearchHit, pub rrf: f64, pub lex_rank: Option<usize>, pub sem_rank: Option<usize> }
pub enum SearchMode { Lexical, Semantic, Hybrid }   // parse "mode" param; default Lexical

impl Engine {
    /// Semantic: embed query with `embedder`, VecIndex::search, map rows → SearchHit
    /// (snippet = 2 lines around start_line via shard content lookup; score = cosine).
    pub fn search_semantic(&self, q: &str, k: usize, embedder: &dyn indexio_embed::embed::Embedder)
        -> anyhow::Result<Vec<SearchHit>>;
    /// Hybrid: lexical top-N + semantic top-N, fuse with RRF (k=60):
    /// score = Σ 1/(60 + rank). Key = (repo, path). Best snippet/score kept per key.
    pub fn search_hybrid(&self, q: &str, limit: usize, embedder: &dyn indexio_embed::embed::Embedder)
        -> anyhow::Result<Vec<HybridHit>>;
}
```

CLI:
- `indexio embed [--repo NAME | --all] [--max-chars N]` — runs indexio_embed::pipeline; prints table.
- `indexio org-sync --org NAME [--token-env GITHUB_TOKEN] [--dest DIR] [--limit N] [--include-forks] [--full-clone]`
  then indexes; prints OrgSyncReport.
- `indexio search <q> --mode lexical|semantic|hybrid` (default lexical). Embedder selection for
  semantic/hybrid: HttpEmbedder::from_env() if INDEXIO_EMBED_BASE set, else HashEmbedder(512).
- `indexio embcas-stats`.

HTTP: `/search?q=&limit=&mode=`; `/embed` POST (admin); `/embcas/stats`.
MCP: `code_search` gains optional `mode` ("lexical"|"semantic"|"hybrid"); new tool
`semantic_search` (query, k). Tool descriptions must explain when to use semantic vs lexical
(lexical = exact identifiers/regex; semantic = concept/natural-language; hybrid = both).

## 5. Wave B test requirements (≥6, HashEmbedder only)

search_semantic returns the planted relevant chunk first for a natural-language-ish query on
a 2-repo fixture; search_hybrid merges lexical + semantic with RRF ordering correct on a
constructed fixture (doc only in lexical top-N still surfaces; doc in both ranks highest);
mode parsing; CLI `indexio embed` then `indexio search --mode hybrid` end-to-end on fixture repos;
MCP code_search with mode=hybrid returns hits JSON; org-sync CLI help smoke.

## 6. Data layout additions

```
<data_dir>/
  embcas/<model_id>/<hh>/<rest>.bin   # embedding CAS
  vec/<model_id>.civec                # binary-quantized flat vector index
```

## 7. Explicit non-goals for P2

No neural weights in-repo or downloaded; no cross-encoder reranker (HttpReranker is P3);
no HNSW (flat+BQ now, ANN layer later); no pgvector/Qdrant external services (sidecar file
is the store); no multi-vector-per-doc ColBERT. Document these in README.
