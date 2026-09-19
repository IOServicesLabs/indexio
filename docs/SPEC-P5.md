# SPEC-P5 — CPU Retrieval Quality Maximization

Addendum to SPEC/SPEC-P2/P3/P4. All engineering rules apply. No new external deps, no
models, no GPU. Parameter choices are grounded in `/mnt/agents/output/research/p5_dim01_lexical.md`
and `/mnt/agents/output/research/p5_dim02_rindex.md` (agents MUST read the parameter tables).
Eval harness: `bench/known_answers20.json` (12 hard + 4 hard + 4 medium concept queries)
via `bench/eval_modes.py`. Every lever lands behind measurable tests; integration eval
decides defaults.

Baseline (P4, 12 hard queries): lexical MRR 0.083 / semantic 0.278 / hybrid R@5 5-of-12.
Target (20 queries): hybrid MRR ≥ 0.45, recall@5 ≥ 50%.

## Agent A — chunk-BM25 leg + 3-leg hybrid fusion

### A1. indexio-embed/src/bm25.rs (new)

Chunk-level BM25F sidecar aligned row-for-row with the vector index of the same namespace
(`<data_dir>/bm25/<model_id>.cibm25`):

```rust
pub struct ChunkBm25 { /* mmap'd */ }
impl ChunkBm25 {
    /// rows: same (header, body) pairs, same order, as the VecIndex rows built in the
    /// same pipeline call. Tokenizer: crate::rindex::tokenize (make it pub(crate)+).
    /// Two fields per chunk: header (path+scope line, weight 3.0) and body (weight 1.0).
    /// Format: magic "CIBM25 1" | u32 n_rows | u32 df-table | u32 avg fields
    /// | FST term → posting offset | postings: LEB128 (row_delta, tf_body, tf_header)
    /// | per-row: u16 body_len, u8 header_len | tombstone bitmap (bincode Vec<u32>).
    pub fn create(dir: &Path, ns: &str, rows: &[(String, String)]) -> anyhow::Result<()>;
    pub fn open(dir: &Path, ns: &str) -> anyhow::Result<Option<Self>>;
    /// BM25F scoring: k1=1.0, b=0.35 (short chunks), header weight 3.0.
    /// Query-side: drop English stopwords (shared list from indexio-embed::embed::stopwords()
    /// — add ~40-word static list) AND drop terms with df > 40% of rows (hub cut).
    /// Returns top-k (row, score), score > 0 only.
    pub fn search(&self, query: &str, k: usize) -> Vec<(u32, f32)>;
    pub fn delete_where(&mut self, pred: impl Fn(u32) -> bool) -> u64;  // by row id
    pub fn save_tombstones(&self) -> anyhow::Result<()>;
    pub fn len(&self) -> usize;  // live rows
}
```

### A2. Pipeline hook (indexio-embed/src/pipeline.rs)

In the same place VecIndex rows are built: collect (header, body) pairs in identical row
order; call `ChunkBm25::create(dir, model_id, &rows)`; apply the same repo-replace +
tombstone semantics as VecIndex (reuse the predicate on row ids after VecIndex filtering —
rows vectors are identical by construction; document the invariant).

### A3. indexio-query 3-leg hybrid + fusion options (Agent A owns indexio-query except the 3 lines in B)

```rust
#[derive(Clone, Copy)] pub enum FusionAlgo { Rrf, CombMnz }  // default Rrf (k=60)

impl Engine {
    /// Legs (all optional, graceful when absent):
    ///   1. file-lexical: existing conjunctive search (fires on identifier/regex queries)
    ///   2. chunk-BM25: ChunkBm25::search (disjunctive, scored — the NL leg)
    ///   3. semantic: embedder + VecIndex (with rindex query expansion, see B)
    /// Key (repo,path) as now. Rrf: Σ 1/(60+rank) per leg. CombMnz: min-max normalize each
    /// leg's scores to [0,1], sum × number of legs the doc appears in (MNZ).
    pub fn search_hybrid_fused(&self, q: &str, limit: usize,
        embedder: &dyn Embedder, algo: FusionAlgo) -> anyhow::Result<Vec<HybridHit>>;
    // search_hybrid keeps working (delegates: Rrf, 3 legs).
}
```

crates/indexio: `indexio search --fusion rrf|combmnz` (default rrf). HTTP `&fusion=`. MCP optional
`fusion` arg. Rerank stage unchanged (applies after fusion when requested).

### A4. Query stopword strip (shared)

`indexio-embed::embed::stopwords() -> &'static [&'static str]` (~40 English function words).
Applied to query text in: ChunkBm25::search (A1), and indexio-query semantic path before
embedding (B coordinates: Agent A adds the strip call around the embed call in
search_semantic/search_hybrid_fused — single helper `fn clean_query(q)->String` in indexio-query).
Lexical mode untouched (conjunctive precision is a feature).

Tests (≥7): bm25 create/open/search planted-doc ordering; header field boost measurable
(term only in header beats term only in body, same tf); hub df-cut; stopword strip;
tombstone alignment with VecIndex rows; 3-leg fusion: doc only in bm25 leg surfaces,
doc in 3 legs beats doc in 1; CombMnz correctness on constructed lists.

## Agent B — Random Indexing v2 + thesaurus query expansion

### B1. indexio-embed/src/rindex.rs upgrades (model_id → "rindex-v2")

- **dim 2048, 8 nonzeros** per label (Sahlgren consensus; p5_dim02 table).
- **True coordinate permutations** replace ±1 dim rotation: two fixed permutations π, π'
  of 0..2048 derived deterministically (xorshift64 seeded blake3(b"ri-perm-v2")); left
  context applies π, right applies π' to the label dims. (Random Permutation Model
  evidence — p5_dim02 §1.)
- **Hub cut**: tokens with df > max(100, 40% of n_texts_seen) get zero weight at embed
  time (their label AND context excluded). Evaluated at embed time only (df known then) —
  no two-pass observe.
- **Header field weighting ×2.5**: chunk text is `header + "\n" + body` — split at the
  first newline; header tokens weighted 2.5 in the pooling sum (BM25F/SIF evidence).
- **Vocabulary cap/min-df unchanged**; keep P4 unit-normalization (regression tests must
  stay green); keep `rindex-v1` loading ability? NO — clean break: `open()` loads only
  `sem/rindex-v2.rimodel`; v1 files ignored (document migration = `indexio embed --rebuild-model`).
- Model format unchanged structurally (dim field already stored; bump magic comment).
  Persist dim; error if file dim ≠ expected 2048.

### B2. Thesaurus query expansion

```rust
impl RandomIndexingEmbedder {
    /// Nearest vocabulary tokens by context-vector cosine; sim ≥ min_sim; excludes
    /// hub-cut tokens and the query token itself.
    pub fn nearest_tokens(&self, token: &str, k: usize, min_sim: f32) -> Vec<(String, f32)>;
    /// For each content token (df≥2, not hub-cut, len≥3): append its top-3 neighbors
    /// (min_sim 0.35) once each. Returns expanded query text (original tokens first).
    pub fn expand_query(&self, query: &str) -> String;
}
```

Embedder trait: add defaulted `fn expand_query(&self, q: &str) -> String { q.to_string() }`.
indexio-query semantic path (COORDINATE WITH AGENT A — you add ONLY this): call
`embedder.expand_query(&clean_query(q))` before `embedder.embed` in search_semantic and the
semantic leg of search_hybrid_fused (if A hasn't landed the helper, apply expansion around
the existing embed call; keep the diff ≤15 lines in indexio-query).

Tests (≥7): permutation determinism + π≠identity + π'≠π; dim/nonzeros invariants; hub-cut
token excluded from embedding (construct 60%-df token, assert its presence/absence doesn't
change a probe vector); header×2.5 measurable (term-in-header beats term-in-body probe);
nearest_tokens on synthetic corpus (login↔authentication as neighbors after the P4-style
fixture); expand_query adds bridging synonym and preserves original tokens first;
end-to-end: semantic search WITH expansion finds the zero-lexical-overlap chunk (extend
the P4 semantic-transfer fixture with a query token whose only path to the target is a
2-hop bridge); bit-for-bit determinism incl. expansion.

## Coordination rules (both agents)

- indexio-embed/src/lib.rs: both add `pub mod` lines (bm25 / nothing new for B) — keep both on merge.
- indexio-query/src/lib.rs: Agent A owns; Agent B adds ≤15 lines (expansion call) and MUST NOT
  restructure anything. If A's 3-leg signature isn't visible in your worktree, code against
  `search_hybrid`'s existing semantic leg.
- English stopword list is authored by Agent A in indexio-embed::embed; Agent B uses
  `crate::embed::stopwords()` in expand_query (if absent at your compile time, define a
  private const fallback with a TODO comment — do not duplicate long-term).
- model namespaces: rindex-v2 changes → fresh data dir for eval (expected).

## Non-goals

RM3/PRF (research flags it high-risk at our first-pass quality; revisit after P5 numbers),
proximity/bigram boost scoring, earliness prior, in-process neural anything, SPLADE/static
embedding downloads.
