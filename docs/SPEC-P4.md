# SPEC-P4 — Self-Contained Semantic Plane (CPU-only, no sidecar)

Addendum to SPEC.md / SPEC-P2.md / SPEC-P3.md. All engineering rules apply. **No model
downloads, no GPU, no external services, no new external crate deps.** Goal: make the
semantic plane fully standalone — a corpus-native embedding model built and updated
in-process on CPU, replacing the HashEmbedder stand-in as the default offline path.

## Technique: Random Indexing (RI)

Vector-space semantics without SVD or neural training. Each token type gets a fixed sparse
ternary "label" vector; each token also accumulates a "context" vector = weighted sum of
the label vectors of tokens it co-occurs with (direction- and distance-sensitive).
Chunk/query embedding = idf-weighted sum of its tokens' context vectors. Single streaming
pass to build; each new chunk is an O(tokens) incremental update — matching the engine's
ongoing-indexing model. Literature: Kanerva et al. 2000; Sahlgren (S-Space); competitive
with LSA on synonymy tests at a fraction of the cost. Corpus-native: learns the org's own
identifiers and co-occurrence patterns.

## 1. indexio-embed: trait extensions (additive, non-breaking)

```rust
pub mod embed {
    pub trait Embedder: Send + Sync {
        fn model_id(&self) -> &str;
        fn dim(&self) -> usize;
        fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;
        /// Stateful embedders ingest new texts into their model before embedding.
        /// Default no-op (stateless embedders). Called by the pipeline with each repo's
        /// new (CAS-miss) chunk texts, or ALL chunk texts on model rebuild.
        fn observe(&self, _texts: &[String]) -> anyhow::Result<()> { Ok(()) }
        /// Persist any model state. Default no-op. Called by the pipeline after each repo.
        fn flush(&self) -> anyhow::Result<()> { Ok(()) }
    }
}
```

## 2. indexio-embed::rindex (Agent 1)

```rust
pub mod rindex {
    use crate::embed::Embedder;

    /// Random-indexing semantic model + embedder. model_id() = "rindex-v1".
    /// Model file: <data_dir>/sem/rindex-v1.rimodel (bincode, tmp+rename atomic).
    pub struct RandomIndexingEmbedder { /* RwLock<Model> */ }
    impl RandomIndexingEmbedder {
        /// Load model if <data_dir>/sem/rindex-v1.rimodel exists, else start empty. dim=1024.
        pub fn open(data_dir: &Path) -> anyhow::Result<Self>;
        pub fn vocab_len(&self) -> usize;
    }
    impl Embedder for RandomIndexingEmbedder { ... }  // incl. observe + flush
}
```

### Model internals
- **Tokenizer** (pub for tests): split chunk text (header + body) on non-alphanumeric;
  then identifier-aware sub-splitting: camelCase/PascalCase boundaries, snake_case,
  SCREAMING_CASE, digit/alpha boundaries. Emit BOTH the lowercase unsplit identifier and
  its parts (unsplit at 0.5 weight). Lowercase everything. Drop tokens of len 1 and pure
  digit tokens. Also emit adjacent bigrams of the final token stream (weight 1.0).
- **Label vector** per token type: deterministic from blake3(b"ri-label"||token): choose
  6 distinct dims in [0,1024) with signs from digest bits (sparse ternary). Right-context
  uses the label rotated +1 dim (permute), left-context rotated -1 — direction sensitivity.
- **observe(texts)**: streaming over tokens of each text in order: for token at position i,
  for d in 1..=3: context(token_i) += (1/(1+d)) * rotated_label(token_{i±d}). Accumulators
  are sparse Vec<(u32,f32)> truncated to top-128 components by |v| after each text.
  Track df per token (texts containing it) and n_texts_seen.
- **Vocabulary cap**: min df ≥ 2 for context accumulation (singletons still embed via their
  own label); hard cap 250_000 types by frequency (LRU-ish eviction acceptable: drop lowest
  df). Document choice.
- **embed(texts)**: for each text: Σ over tokens idf(t)·context(t) + 0.25·Σ label(t),
  idf(t) = ln(1 + n_texts_seen / df(t)); densify to dim-1024 f32, L2-normalize. Tokens with
  no context yet contribute only their label (graceful cold start).
- **Persistence**: bincode {u32 dim, u64 n_texts_seen, Vec<(String, u32 df, Vec<(u32,f32)>)>};
  tmp+rename; corrupt file → start empty with tracing::warn.

### Pipeline + selection wiring
- `pipeline::embed_repo_with` / `embed_all_with`: collect this repo's chunk texts; call
  `embedder.observe(&miss_texts)` BEFORE embedding misses (CAS hits were observed when
  first embedded); `embedder.flush()` after the repo's index is written. Add
  `indexio embed --rebuild-model`: deletes sem model + that model's embcas namespace, observes
  ALL chunks (needed because RI model state includes now-deleted docs' contributions —
  document this drift note).
- Embedder selection (crates/indexio main.rs, serve.rs; shared helper):
  1. `INDEXIO_EMBED_BASE` set → HttpEmbedder (unchanged)
  2. `--embedder hash` flag → HashEmbedder (A/B escape hatch; flag on `indexio embed`+`indexio search`)
  3. default → RandomIndexingEmbedder::open(data_dir) (creates model on first `indexio embed`)
  Vec/embcas namespaces already keyed by model_id, so hash-v1 / rindex-v1 / http:* coexist.

### Tests (≥8)
tokenizer (camel/snake/digit splitting, unsplit+parts emitted); label determinism +
sparsity (6 nonzeros) + rotation direction; observe→embed determinism (same corpus, same
vectors bit-for-bit); dim/normalization; **semantic transfer test**: synthetic corpus
chunk A {login password session}, chunk B {database pool connection}, chunk C
{authentication login oauth} — after observe(all), embed query "authentication" must rank
A above B (login bridges A↔C; pure lexical overlap with A is zero); cold-start: empty
model embeds without panic and identical texts match; persistence round-trip (observe,
flush, reopen, embed → identical); vocab cap respected; pipeline integration: embed_repo
on 2-repo fixture with rindex embedder — model file written, second repo's shared chunks
are CAS hits, search_semantic via Engine works end-to-end.

## 3. Eval harness (Agent 2 — read-only on code, writes bench artifacts)

Produce `/mnt/agents/output/indexio/bench/known_answers.json` +
`/mnt/agents/output/indexio/bench/eval_modes.py`:
- Study the benchmark corpora at /mnt/agents/bench-repos/{ripgrep,serde} (read code).
- Write 12 natural-language concept queries, each with ONE known-answer file
  (repo + path) and a one-line justification grounded in the actual code. Requirements:
  the answer file must NOT contain the query's key noun verbatim in its path (we want
  queries that need semantics, not filename lookup); spread across crates/languages;
  mix of "how does X work" / "where is Y implemented" phrasings. JSON:
  `[{"q": "...", "repo": "ripgrep", "path_suffix": "crates/printer/src/standard.rs", "why": "..."}]`
- eval_modes.py: runs `indexio search "<q>" --mode M --limit 10 --json` for M in
  [lexical, semantic, hybrid] over a data dir, computes per-mode MRR and recall@5 against
  (repo, path_suffix), prints a table. No third-party python deps.

## 4. Explicit non-goals

No SVD/PCA, no word2vec/fastText training loop (RI replaces it), no GPU path changes
(HTTP embedder stays as the optional upgrade), no cross-encoder changes, no PRF/RM3
query expansion (candidate for P5), no stemming/lemmatizer deps.
