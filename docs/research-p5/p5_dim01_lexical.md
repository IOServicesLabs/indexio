# CPU-Only Lexical/Statistical Retrieval Improvements for Code Search — Parameter Guidance

Context: sparse trigram index (conjunctive literals) + BM25-lite file ranking + RRF hybrid with vector leg; lexical MRR 0.083 on 12 hard NL concept queries. Scope: implementable techniques with concrete parameters.

---

## 1. BM25 variants for code / NL queries

**Variant choice matters less than parameter tuning.** On BEIR, all four mainstream variants score within ~0.5 nDCG@10 of each other (BM25S, k1=1.2, b=0.75: Robertson 39.87, ATIRE 39.92, BM25+ 39.92, BM25L 39.48 avg) [^1^]. Guidance:

- **Default: classic/Lucene BM25.** Most tested; safe for fusion (bounded, monotone) [^2^].
- **BM25+ (delta 0.5–1.0)** if your corpus mixes very short chunks with long files — it lower-bounds the TF component so long docs are never over-penalized to zero [^3^][^4^]. With uniform ~30–80-line chunks this pathology is rare, so BM25+ is optional.
- **BM25L** only for extreme document-length variance (long-doc corpora); it slightly *under*performs on BEIR [^1^].

**k1 / b for short code chunks (~30–80 lines, i.e. passage-like docs):**

| Setting | k1 | b | Source |
|---|---|---|---|
| IR-book defaults | 1.2–2.0 | 0.75 | [^5^] |
| Lucene/Zoekt (code search production) | 1.2 | 0.75 | [^6^] |
| BEIR (Anserini Lucene defaults) | 0.9 | 0.4 | [^7^] |
| MS MARCO passage (tuned) | 0.82 | 0.68 | [^8^] |

Short, uniform-length docs → **lower b (0.3–0.5)** because length normalization has little signal and a high b just adds noise; moderate **k1 ≈ 0.9–1.2** since NL concept queries repeat few terms. Sweep k1 ∈ {0.6, 0.9, 1.2}, b ∈ {0.3, 0.4, 0.6, 0.75} on your 12 queries — with n=12 treat deltas < ~0.05 MRR as noise.

**Notable production trick (Zoekt):** for keyword-style queries Zoekt computes BM25 with **IDF dropped entirely** (treated as constant) — their evals showed IDF over-down-weights some keywords and worsens ranking for code [^6^]. Worth an A/B on concept queries (probably keep IDF for NL queries, but it's a one-line change).

**BM25F field weighting** — the single highest-ROI lexical change for code, "zero additional ML cost, +10–20% precision on structured documents" [^9^]:

- Sourcegraph: **5× boost on symbol/filename matches**, "worked well right out-of-the-box" [^9^]; Zoekt implements exactly this: filename or symbol match → TF += 5, plus **÷5 penalty for low-priority files (tests, generated code)** [^6^].
- Generic structured-doc defaults: title 3.0, heading 2.0, metadata/tags 1.5, body 1.0 [^9^][^10^].
- For a code engine, recommended fields: **symbol/identifier names 5.0, filename/path 3.0–5.0, docstring/comment 2.0, code body 1.0**. Implementation: field-tagged postings so the scorer knows which field a match came from [^9^]; Zoekt's simpler version just inflates term frequency for filename/symbol matches before scoring [^6^].

## 2. RM3 / pseudo-relevance feedback

**Formula.** Estimate RM1 over feedback set F with Dirichlet-smoothed doc models:
P(w|R) ∝ Σ_{d∈F} P(w|θ_d) · P(θ_d) · Π_i P(q_i|θ_d), then interpolate with the original query: **P(w|θ′_q) = (1−λ)·P(w|θ_q) + λ·P(w|R)** [^11^][^12^]. (Watch the λ-direction: Anserini's `originalQueryWeight=0.5` is the *query* weight [^13^]; Terrier's `fb_lambda=0.6` is the *feedback* weight [^14^]. Don't mix conventions.)

**Standard parameters:**

| Parameter | Terrier/PyTerrier | Anserini (Indri defaults) | Tuned grids in literature |
|---|---|---|---|
| Feedback docs K | 3 | 10 | {5, 10}; sometimes 20 [^15^] |
| Expansion terms | 10 | 10 | {10, 20}; BioASQ gridsearched [^15^][^16^] |
| λ (orig-query weight) | 0.4 | 0.5 | {0.5, 0.6} [^15^][^17^] |

**Expected gains over plain BM25:** modest but real on average — BEIR nDCG@10 0.410 → 0.420 [^17^]; MS MARCO MRR@10 0.184 → 0.192 [^17^]; NFCorpus +0.9 nDCG@10 (0.267 → 0.276, best config K=5, 20 terms, λ_query=0.6) [^15^].

**Failure modes — critical for your setting:**
- **Query drift when first-pass ranking is poor.** On TREC ToT (hard, sparse queries with unreliable BM25 rankings), RM3 *collapsed* Recall@1000 from 0.77 → 0.51 [^18^]. With your lexical MRR at 0.083, most feedback docs will be non-relevant → RM3 will amplify noise. This is the dominant risk.
- Single-relevant-document regimes make PRF errors catastrophic [^18^].
- Expansion can regress already-saturated rankers (SPLADE on SciFact −1.7 nDCG) [^15^].

**Recommendation for a weak baseline:** if you implement RM3, use the conservative corner: **K=3–5 feedback docs, 10 expansion terms, λ_query ≥ 0.5–0.6**, and gate it (skip expansion when top-1 score < threshold or top-k scores are flat, indicating a failed first pass). Expect +1–2 nDCG when it works, negative when it doesn't — measure per-query. Rocchio is a competitive alternative that recent work found ~1 point better than RM3 on BM25 feedback because it preserves original-query weight more stably [^19^].

## 3. Conjunctive vs disjunctive multi-term handling

- Pure AND on multi-term NL queries destroys recall under vocabulary mismatch — this is very likely a contributor to your 0.083 MRR on *concept* queries. Move to disjunctive (OR) scoring with a **minimum-should-match** floor.
- **Elasticsearch `mm` conventions:** `mm="75%"` requires ⌊75%⌋ of clauses; negative percentages round the *optional* count instead — "75%" and "-25%" coincide at 4 clauses but diverge at 5 (3 vs 4 required) [^20^]. Combined specs like `"3<75%"` mean: ≤3 terms → all required; >3 terms → 75%. Precision-oriented engines commonly use **mm = 60%–75%** [^21^][^22^].
- **Practical recommendation for NL concept queries on code:** mm ≈ **"2<60%"** (require all terms for ≤2-term queries, 60% above) as the precision floor for the lexical leg; keep pure AND only for exact-literal/identifier queries where the user typed code tokens. Recall lost by AND is not recoverable by fusion if the doc never enters the candidate pool [^23^].
- **Stopwords for code corpora:** don't use a hand-curated English list alone. Best practice: **DF-based stopword filter** — Vespa's `weakAnd` with `stopwordLimit = 0.05` (drop query terms appearing in >5% of docs) gave **+0.0136 MRR@10 on MS MARCO** and halved latency; 0.02 was too aggressive and dropped content words [^24^]. For code, additionally treat **language keywords (`if`, `return`, `for`, `public`…) as stopwords** — a DF threshold achieves this automatically since keywords are ultra-high-DF. Bug-localization literature standardly does stopword removal + punctuation removal + camelCase splitting on both query and code side, and reports **stemming has mixed findings** (recall↑, precision↓) — often skipped [^25^].

## 4. Fusion: CombSUM/CombMNZ vs RRF with heterogeneous legs

- **RRF(k=60) remains the robust default** with incomparable score scales: rank-only, outlier-immune, one knob [^26^][^27^]. Cormack et al. showed ranks alone beat normalized-score fusion [^26^].
- **Evidence that score-based fusion can win when normalization is done right:**
  - Convex combination with min-max normalization at α=0.5: Recall@5 0.726 vs RRF(k=60) 0.695; **lower k helps RRF** (k=10 → 0.716) [^28^].
  - Code-search-specific: on Solidity retrieval, **CombSUM/CombMNZ (0.523 top-1) beat RRF (0.494)** and Borda (0.480) — "score-based methods outperform rank-based methods" when score magnitudes carry signal [^29^].
  - CombMNZ needs per-list normalization first: min-max is standard; z-score is more robust to outliers, which matters because BM25 top scores are heavy-tailed [^30^][^31^].
- **Weighted RRF:** start equal-weight; production guidance is "only deviate when measurement justifies it — often equal-weight RRF is good enough that tuning doesn't pay off" [^27^]. When tuning, 50–100 labeled query-doc pairs suffice to set α [^32^]; grid-searched hybrids often land near **dense 0.7 / sparse 0.3** for NL-heavy query mixes [^33^]. Given your *lexical* leg is the weak one, do not up-weight it by default.
- **Actionable:** keep RRF but **sweep k ∈ {10, 20, 60}** (smaller k emphasizes top ranks — good when one leg has high top-precision) and per-leg weights ∈ {0.3/0.7, 0.5/0.5, 0.7/0.3}. Try CombMNZ + min-max as a cheap second candidate; per-list normalization quality, not the Comb* variant, is what determines success [^31^]. One caution: equal-weight RRF on a weak leg can *degrade* vs the strong leg alone (−5.6% nDCG@10 reported on scientific corpora) [^34^] — always compare hybrid vs best single leg.

## 5. Other cheap, proven wins

1. **Proximity scoring.** Adding a continuous proximity feature to BM25 gave robust MS MARCO gains: Vespa `nativeProximity` weight **w_prox = 10, wide plateau 8–14** (real gain, not noise-fitted) [^24^]. CPU-cheap DIY version: score adjacent query-bigram phrase frequencies and add as a bonus term [^24^].
2. **Bigram field boost.** Store query bigrams in a parallel field with ~**2× boost** over unigram sum; bigrams have naturally low DF, so without compensation an exact phrase match scores *below* scattered unigrams — multiply bigram contribution by 1.5–2× [^35^]. Requires position storage or a bigram token field; cap ~25 bigrams/doc [^35^].
3. **Earliness prior.** Rewarding matches near the start of a field (analogous to docstring-at-top) added +0.0189 MRR over the proximity-only anchor at weight ~8 on MS MARCO [^36^]. For code: matches in the leading comment/docstring block are a strong signal.
4. **Identifier splitting (must-have).** Split camelCase/PascalCase/snake_case identifiers into subwords at *both* index and query time (index both the whole identifier and parts) — standard in IR-based bug localization and code search [^25^]; ~7.5% of identifiers resist heuristic splitting [^37^]. Without this, NL concept queries can't match the vocabulary where code actually stores meaning.
5. **Document/file-type prior.** Zoekt down-weights test/generated files 5× and boosts symbol definitions — a static per-file quality prior multiplied into the lexical score [^6^]. Also: filename-length and word-boundary match quality as priors [^38^].
6. **Query-term coverage as a rerank feature.** A deterministic feature reranker over the fused top-20 (30% normalized BM25, 25% cosine, **20% IDF-weighted query-term coverage**, 10% title coverage, 7.5% adjacent-bigram rate, 5% proximity, 2.5% identifier coverage) is a proven CPU-only pattern [^39^]. IDF-weighted coverage ("fraction of query-term IDF mass present") is the cheapest strong signal you're probably not computing.

---

## Recommended parameter table

| Component | Parameter | Recommended value | Notes / evidence |
|---|---|---|---|
| BM25 core | variant | Lucene/classic (BM25+ δ=0.5 if long files present) | variants ≈ tie on BEIR [^1^] |
| | k1 | 0.9–1.2 (start 0.9) | short chunks; BEIR 0.9 [^7^], Lucene 1.2 [^6^] |
| | b | 0.3–0.4 | short uniform chunks; BEIR b=0.4 [^7^] |
| BM25F fields | symbol/identifier names | 5.0 | Sourcegraph/Zoekt [^6^][^9^] |
| | filename/path | 3.0–5.0 | [^6^][^9^] |
| | docstring/comment | 2.0 | [^9^] |
| | code body | 1.0 | baseline |
| File prior | test/generated penalty | ÷5 | Zoekt [^6^] |
| Multi-term | mm (NL queries) | "2<60%" (i.e., all for ≤2 terms, 60% above) | ES mm conventions [^20^][^21^] |
| | mm (literal/identifier queries) | 100% (keep AND) | trigram literal path unchanged |
| Stopwords | DF-based limit | drop terms with DF > 5% of corpus | +0.0136 MRR@10 MS MARCO [^24^] |
| | stemming | off (or query-side only) | mixed findings in code IR [^25^] |
| RM3 (gated) | K feedback docs | 3–5 | weak first pass → small K [^14^][^18^] |
| | expansion terms | 10 | Terrier/Anserini default [^13^][^14^] |
| | λ (orig-query weight) | 0.5–0.6 | [^15^][^17^] |
| | gate | skip if top-k score flat/low | drift risk at MRR 0.083 [^18^] |
| Fusion | method | RRF, sweep k ∈ {10, 20, 60} | k=10 > k=60 in one head-to-head [^28^] |
| | alt method | CombMNZ + min-max per leg | beat RRF on code-search fusion [^29^] |
| | weights | 0.5/0.5 start; grid {0.3,0.5,0.7} per leg | equal default [^27^]; tune w/ 50–100 labels [^32^] |
| | candidate depth | 50–100 per leg | [^23^] |
| Proximity | bigram/span boost | w ≈ 10× normalized proximity, or bigram field boost 1.5–2× | plateau 8–14 [^24^]; bigram IDF fix [^35^] |
| Earliness | docstring/first-block bonus | small additive, w ≈ 5–8 on normalized scale | +0.019 MRR [^36^] |
| Tokenization | identifier splitting | index whole + subwords, both sides | [^25^][^37^] |

**Priority order for MRR 0.083 → up:** (1) relax AND → mm-based disjunctive; (2) identifier splitting; (3) BM25F symbol/filename boosts; (4) DF-based stopword filter; (5) b ↓ to 0.3–0.4 + k1 sweep; (6) proximity/bigram bonus; (7) RRF k + weight sweep; (8) gated RM3 last (highest risk).

---

## References

[^1^]: BM25S / Baguetter BEIR benchmark, variant comparison at k1=1.2, b=0.75 — https://www.mixedbread.ai/blog/intro-bmx and https://aiqianji.com/blog/article/4460
[^2^]: bm25s variant docs (Kamphuis et al. 2020) — https://pypi.org/project/bm25s-j/
[^3^]: BM25 variant guide (BM25L for length variance, BM25+ lower bound δ default 0.5) — https://github.com/alessandrobenigni/BM25-Turbo-Rust-Python-WASM-CLI
[^4^]: BM25/BM25L/BM25+ definitions — https://djamriska.github.io/bm25.html
[^5^]: "IR book recommends k1 between 1.2 and 2.0, b=0.75" — https://pypi.org/project/bm25s-j/
[^6^]: Zoekt `index/score.go` (k=1.2, b=0.75; IDF skipped; importantTermBoost=5 for filename/symbol; lowPriorityFilePenalty=5) — https://raw.githubusercontent.com/sourcegraph/zoekt/main/index/score.go ; design doc — https://github.com/sourcegraph/zoekt/blob/main/doc/design.md
[^7^]: BEIR paper setup: "Anserini's default Lucene parameters (k=0.9, b=0.4)" — https://blog.csdn.net/zag666/article/details/128336349 (BEIR, Thakur et al. 2021)
[^8^]: Anserini MS MARCO passage tuned BM25 k1=0.82, b=0.68 — https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md
[^9^]: Strata BM25F issue: "+10–20% precision on structured documents"; Sourcegraph 5× symbol/filename boost; default weights title 3.0 / heading 2.0 / metadata 1.5 / body 1.0 — https://github.com/stratalab/strata-core/issues/2272
[^10^]: BM25F field boosts title 3×, metadata 2×, body 1× — https://seowarroom.app/encyclopedia/sem-information-retrieval/bm25-and-probabilistic-ir
[^11^]: RM1/RM3 formulas — https://arxiv.org/pdf/1606.00615.pdf
[^12^]: RM3 interpolation formula — https://utheme.univ-tlse3.fr/access/files/original/4d7a29abbffd06d52ddca3cfe0acc4e8eaef4cd7.pdf
[^13^]: Anserini RM3 defaults: fbTerms=10, fbDocs=10, originalQueryWeight=0.5 — https://mintlify.com/castorini/anserini/search/reranking
[^14^]: PyTerrier RM3 defaults: fb_terms=10, fb_docs=3, fb_lambda=0.6 — https://pyterrier.readthedocs.io/en/stable/_modules/pyterrier/terrier/rewrite.html
[^15^]: RM3 grid K∈{5,10}, terms∈{10,20}, λ_orig∈{0.5,0.6}; NFCorpus 0.267→0.276 — https://arxiv.org/html/2605.11374v1
[^16^]: BioASQ BM25+RM3 hyperparameter optimization — https://ceur-ws.org/Vol-3497/paper-013.pdf
[^17^]: TW-BERT results table: BM25 0.184 → BM25+RM3 0.192 (MS MARCO MRR@10); BEIR 0.410 → 0.420 — https://oplclaw.com/paper/202133
[^18^]: PRF failure on TREC ToT: RM3 Recall@1000 0.5093 vs 0.7705 baseline; drift amplification — https://arxiv.org/pdf/2602.10321
[^19^]: Rocchio > RM3 (~1 pt) on BM25 feedback, 13 BEIR tasks — https://www.alphaxiv.org/abs/2511.19349 ; https://cs.uwaterloo.ca/~jimmylin/publications/Jedidi_Lin_SIGIR2026.pdf
[^20^]: Elasticsearch minimum_should_match docs: "75%" vs "-25%" rounding at 4 vs 5 clauses — https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-minimum-should-match.html
[^21^]: mm percentage usage ("at least 75% of the query terms") — https://opster.com/guides/elasticsearch/search-apis/elasticsearch-minimum-should-match/
[^22^]: match query mm syntax reference — https://pulse.support/kb/elasticsearch-match-query
[^23^]: Candidate depth before fusion: 50–100 per retriever; fusion cannot recover missing candidates — https://bcloud.ai/vector-database-hybrid-search/
[^24^]: Vespa MS MARCO BM25 tuning: weakAnd stopwordLimit=0.05 → +0.0136 MRR@10; nativeProximity w=10 (plateau 8–14) — https://blog.vespa.ai/re-autoresearching-msmarco-bm25-on-vespa/
[^25^]: IR-based bug localization preprocessing: stopword removal, camelCase splitting, stemming avoided (mixed findings) — https://web.cs.dal.ca/~masud/papers/masud-EMSE2021.pdf
[^26^]: RRF robustness vs normalized-score fusion; Cormack 2009 — https://bigdataboutique.com/blog/reciprocal-rank-fusion-how-it-works-and-when-to-use-it
[^27^]: Production RRF practice: equal weights until data justifies otherwise — https://relevantsearch.ai/volumes/vol-09-llm-augmented/
[^28^]: Convex combination α=0.5 (0.726 R@5) vs RRF k=60 (0.695), k=10 (0.716) — https://arxiv.org/pdf/2604.01733
[^29^]: Code-search fusion on Solidity: CombSUM/CombMNZ 0.523 vs RRF 0.494 top-1 — https://arxiv.org/html/2504.07740v1
[^30^]: Fusion formulas + normalization robustness (min-max, min-sum, min-var) — https://pmc.ncbi.nlm.nih.gov/articles/PMC5267596/
[^31^]: CombMNZ with min-max normalization; complex normalizations no better than simple CombMNZ — https://terpconnect.umd.edu/~oard/pdf/acl08.pdf
[^32^]: "50–100 labeled query-doc pairs sufficient to tune α"; dense-heavy α=0.7 starting point — https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/blob/main/docs/bge-m3-model.md
[^33^]: Grid-searched RRF weights α_dense=0.7, α_sparse=0.3 — https://arxiv.org/pdf/2603.20534
[^34^]: Equal-weight RRF degrading nDCG@10 by 5.6% vs best leg on scientific corpora — https://cseit2026.org/spm/papers
[^35^]: Bigram field with ~2× boost; IDF-only bigram under-scores, multiply by 1.5–2×; cap 25 bigrams/doc — https://wal.sh/research/pocket-es/sip-ngram-research
[^36^]: fieldMatch earliness w=8, +0.0189 paired MRR — https://blog.vespa.ai/re-autoresearching-msmarco-bm25-on-vespa/
[^37^]: Identifier splitting difficulty (~7.5% unsplittable by heuristics) — https://arxiv.org/pdf/1805.11651
[^38^]: Zoekt ranking signals list (word-boundary match quality, filename length, symbol definitions) — https://github.com/sourcegraph/zoekt/blob/main/doc/design.md
[^39^]: Deterministic feature reranker weights (30% BM25 / 25% cosine / 20% IDF-weighted coverage / …) — https://knowledge-base.software/guides/vector-search-for-knowledge-bases/
