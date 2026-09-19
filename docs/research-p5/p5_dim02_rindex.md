# Random Indexing tuning scan for code search (dim=1024, 6-nonzero ternary labels, window ±3, 1/(1+d) decay, idf-weighted sums, 250k vocab)

Baseline: semantic-only MRR 0.278, recall@5 4/12 on 12 hard NL concept queries (ripgrep+serde). Findings below are parameter-focused; recommendations at the end.

## 1. RI parameter evidence (Kanerva / Sahlgren lineage)

**Dimensionality.** Sahlgren's original and follow-on work reports dimensionalities "around 2 000 seem to be optimal" for document-based RI, using 20 nonzeros (10× +1, 10× −1) in the 2000-d label vectors; replications confirm 2000 as "a useful starting value" [^1^]. The Random Permutation model of Sahlgren et al. uses far sparser labels — only two +1s and two −1s (4 nonzeros) at ~2000 dims [^2^]. Theoretical analysis (QasemiZadeh & Handschuh) shows that with m ≈ 1600, as few as **two nonzeros per index vector suffice** to build a working VSM — sparsity controls collision noise, not capacity, and denser labels raise cross-term interference (see below) [^3^]. Velldal's re-examination likewise frames k (dimensionality) and ε (nonzeros) as the only two free parameters and confirms the standard 1000–2000 / ~2–20-nonzero range [^4^]. Kanerva's original recommendation: "say 10 non-zero elements in a 1000-dimensional vector" [^5^].

Practical reading: at 1024 dims your 6-nonzero labels (0.6%) are in the sane band — 2000-d labels in the literature use 0.2–1% density. Moving to 2048 dims with ~10 nonzeros tracks Sahlgren's "optimal" config and roughly quadruples capacity for separable contexts at 2× memory — the cheapest quality lever in the classic literature.

**Window size & direction.** Sahlgren (2006): small windows → paradigmatic/similarity (synonyms), large windows → syntagmatic/topical relatedness [^6^]; Bullinaria & Levy and Lapesa & Evert's large-scale evaluation broadly confirm best results cluster around small-to-mid windows (2–8) with distance-weighted, dynamically-sampled windows preferable [^7^][^8^]. For NL-concept-to-code matching you mostly want **topical** (syntagmatic) association (query "case-insensitive matching" should land near `to_lowercase`/`is_case_sensitive`), so a *moderate* window with distance decay is right; your ±3 is reasonable, and word2vec-style *dynamic* windows (uniformly sample actual width in [1,3] per occurrence, equivalently weighting 1/d) are the evidence-backed variant of your 1/(1+d) scheme [^8^]. Note GloVe uses 1/d and word2vec uses (w−d+1)/w — both linear-ish decays; nothing suggests 1/(1+d) is worse than 1/d [^9^].

**Direction sensitivity.** For order/role encoding, Sahlgren et al.'s RPM finds **optimal performance with only two permutations** — one for "before", one for "after" — rather than a distinct permutation per distance [^2^]. This directly supports your ±1-dim rotation / direction tagging: 2 rotations suffice, per-distance rotation is overkill and dilutes signal. (Your "±1 dim rotation" is not a permutation — a coordinate rotation mixes only 2 dims. Consider using an actual *coordinate permutation* of the label vector, which is the operation with published evidence, since permutation is what RPM/Predication-based Semantic Indexing used successfully [^10^].)

**Weighting.** Gorman & Curran's scaling study is the key warning: RI is "robust for small corpora, but larger corpora require that the contexts be weighted to maintain accuracy" — unweighted accumulation degrades as data grows [^11^]. Your idf weighting already does this; keep it.

## 2. Semantic thesaurus / query expansion with RI

- Context-vector nearest-neighbor expansion is established practice: Reflective Random Indexing was used explicitly to mine related terms as query-expansion candidates in e-discovery [^12^]; WordNet/co-occurrence-thesaurus expansion studies show the universal pattern — **recall rises markedly (e.g. 0.23 → 0.44–0.48), precision drops**, so expansion is a recall lever, not an MRR lever unless re-ranking is score-aware [^13^][^14^].
- Rocchio/PRF canonical parameters: α=1.0, β≈0.4–0.75, top 5–10 feedback docs, 10–100 expansion terms; a widely-used safe recipe is **top-5 docs, ≤10 new terms, α=1, β=0.75** [^15^][^16^].
- For a thesaurus-based (nearest-neighbor) expansion over your 250k-term space: expand each query term with k≈3–5 nearest terms weighted by ~0.3–0.5 × cosine similarity (expansion weight clearly below original terms, since originals "usually carry the most weight" [^17^]). Given your task is recall@5-limited, expansion is the single most promising lever, but it will inject **hubs** (high-frequency, high-centrality terms recur as neighbors of everything) — see §3.

## 3. Hubness and frequency effects

- Hubness is intrinsic to high-d inner-product spaces; hubs sit near the centroid [^18^]. The large empirical comparison by Feldbauer & Flexer (50 datasets) recommends, in order of robustness: **mutual proximity (MP), local scaling (LS), NICDM, and DisSimLocal**; LS/NICDM/DSL cost O(n²) or better, MP needs the Gaussian approximation for speed. Shared-nearest-neighbor rescaling reduces hubness but *damages* semantic accuracy — avoid [^19^].
- For a retrieval pipeline the cheapest effective anti-hub trick for a 250k-vocab thesaurus is **CSLS** (cross-domain similarity local scaling, k≈10): `CSLS(q,x) = 2·cos(q,x) − μ_k(q) − μ_k(x)` [^20^], or NICDM local scaling `d(x,y)/sqrt(μ_x·μ_y)` — both give the same neighbor-symmetrization effect and consistently improve NN classification accuracy in the comparison [^19^][^21^]. For query→document scoring, simpler: **subtract the centroid (centering)** or penalize each item's mean similarity to all items ("querybank/dual-softmax"-style normalization), which was shown effective for text specifically [^19^][^22^].
- **Frequency ↔ hubness coupling**: word embeddings encode frequency even after L2 normalization, and neighbor quality is worst for very low- and very high-frequency terms; mid-frequency terms (10³–10⁴) are most stable [^23^]. RI practice filters out the most frequent context units (function words) before accumulating [^24^]. Recommendations: impose a **max-df** cutoff (drop or down-weight the top ~50–100 most frequent tokens / tokens in >X% of docs — Baroni/Evert exclude the 500 most frequent features [^25^]), keep min-df 2 (raising it slightly, e.g. 3–5, on a 2-repo corpus is likely noise-reduction), and prefer **sublinear idf** (e.g. `1+ln(idf)` or `idf` on log-scaled tf) so ultra-frequent tokens don't dominate context-vector accumulation. Unit-normalizing each context vector before summing (already done) plus max-df is the standard hub-suppression combo.

## 4. Field/structure weighting

- The measured evidence is almost entirely from *lexical* retrieval: BM25F per-field weighting (title/name ≈ 2–3× body) is "usually a larger relevance win than any k1 or b adjustment" when documents have structure [^26^][^27^]. Code-search tooling applies the same priors — file-path match boost, symbol/signature fields weighted above body content [^28^][^29^].
- For **semantic** vector spaces the transferable, evidence-backed practices are: (a) **pool with SIF-style or idf weights rather than flat mean** — Model2Vec's post-training regularization (frequency weighting → PCA → SIF weighting) is what makes static mean-pooled embeddings competitive on retrieval [^30^]; (b) **concatenate separately-pooled field vectors** (or sum field vectors with a field multiplier ~2–3 for path/identifier/comment tokens) — engineering-consistent with BM25F gains, no published ablation contradicts it; (c) build the expansion thesaurus over the *whole* vocabulary but weight path/identifier tokens higher when composing document vectors. Identifiers carry the most query-alignable semantics in code, so a 2–3× multiplier is the defensible default.

## 5. Cheap augmentations

- **Random projection vs direct sparse-dense:** Achlioptas ternary projections ({±√3, 0} with probabilities 1/6, 2/3, 1/6) provably preserve pairwise distances (JL) at 3× sparsity of Gaussian — your sparse ternary labels *are* an Achlioptas-style projection, so a separate RP dimensionality-reduction stage adds nothing [^31^][^32^]. The better move is increasing dim (§1).
- **Binarization/ternarization with Hamming prescan:** "Near-lossless Binarization of Word Embeddings" reports ~**2% accuracy loss** for learned binarization with large size reduction [^33^]; sign-binarization of CNN features with Hamming matching shows <1% retrieval loss [^34^]; production guidance is Hamming-prescan-then-float-rerank (binary candidates → rerank top-k with full precision) [^35^]. Model2Vec int8 quantization + Matryoshka dimension truncation shows "near-identical retrieval quality" at 25% memory [^36^]. For your 1024-d float vectors: 1024-bit sign codes + Hamming prescan + cosine rerank of top ~200 is a safe ~32× memory/compute cut with ≲2% quality cost.

## 6. CPU-feasible alternatives that may beat RI outright

- **PPMI + truncated (randomized) SVD**: Levy, Goldberg & Dagan show hyperparameters (dynamic window, subsampling, context distribution smoothing 0.75, adding context vectors w+c) matter more than the algorithm, and properly-tuned count-based SVD matches or beats SGNS on similarity tasks; PPMI/SVD specifically outperforms skipgram on rare-word and semantic-relation tasks [^37^][^38^]. On a 2-repo corpus (millions of tokens) this is minutes of CPU (sparse co-occurrence + randomized SVD to 300–500 dims). Levy & Goldberg (2014) prove SGNS implicitly factorizes a shifted PMI matrix — i.e., your RI space is a noisy version of the same object; a direct PPMI-SVD removes the sampling noise [^39^].
- **SGNS/word2vec via gensim (CPU)**: trains in minutes on corpora this size; SGNS is "a robust baseline," preferred to CBOW and GloVe [^38^]. GloVe has a fast multithreaded C implementation; training cost is comparable, quality not better than SGNS per Levy et al.
- **Static pretrained embeddings (Model2Vec/potion)**: if "corpus-native" is a soft constraint, potion-retrieval-32M reaches 35.06 MTEB Retrieval (82% of all-MiniLM-L6-v2 at 42.92) with pure token-lookup + weighted mean on CPU, orders of magnitude faster than transformers [^30^][^40^]; potion-base-32M hits 93% of MiniLM's all-task average [^41^]. Caveat: on small personal corpora, potion models showed noticeably weaker real-world retrieval than MiniLM (32% top-result overlap) — pretrained static embeddings shine as a *base* but benefit from corpus-specific blending [^42^].
- **Honest bottom line:** RI with tuned hyperparameters is roughly equivalent to count-based DSMs (Levy et al. result cuts both ways). The genuinely-free upgrade path is: keep RI machinery, add PPMI-style association weighting (or fold a small SVD over the RI term matrix), adopt dynamic-window sampling, and use the space only for query expansion + a semantic channel fused with lexical scores. RI's published strength is exactly small-to-mid corpora and incremental indexing [^11^] — matching your scenario.

## Recommended parameter table

| Parameter | Current | Recommended | Evidence |
|---|---|---|---|
| Label dimensionality | 1024 | **2048** (4096 marginal) | Sahlgren "2000 optimal" [^1^]; capacity vs cost |
| Label nonzeros | 6 (0.59%) | **8–12 at 2048 (~0.5%)**, balanced +/− | Sahlgren 20/2000 [^1^]; RPM 4/2000 [^2^]; theory: 2–10 suffice [^3^] |
| Window | ±3, 1/(1+d) | **Keep ±3; sample width dynamically in [1,3] (= distance decay) or widen to ±4–5** for topical matching | Levy et al. dynamic window [^8^]; small=paradigmatic/large=topical [^6^] |
| Direction encoding | ±1 dim rotation | **Two permutations only (before/after)** — not per-distance | RPM optimal with 2 permutations [^2^] |
| Context weighting | idf, unit-norm contexts | Keep; add **sublinear idf** and **max-df cutoff** (drop top ~100–500 df tokens) | Gorman & Curran weighted contexts [^11^]; RI filters frequent contexts [^24^][^25^] |
| min-df | 2 | 2–5 (raise if hub lists dominated by junk) | frequency-variation results [^23^] |
| Query expansion | none | **k=3–5 nearest terms, weight 0.3–0.5·cos; or Rocchio PRF α=1, β=0.4–0.75, top-5 docs, ≤10 terms** | recall +8–24pp typical [^13^][^15^][^16^] |
| Hubness mitigation | none | **Centering (subtract centroid) for doc vectors; CSLS (k≈10) or NICDM for thesaurus NN lists** | Feldbauer & Flexer ranking [^19^]; centering for text [^19^][^22^] |
| Field weighting | none | **2–3× weight for path/identifier/signature tokens** in doc pooling; SIF/idf-weighted pooling | BM25F gains [^26^][^27^]; Model2Vec SIF pooling [^30^] |
| Compression | none (float32) | Optional: **sign-binarize + Hamming prescan, cosine rerank top-200** (≲2% loss, ~32× smaller) | [^33^][^34^][^35^] |
| Alternative model | RI only | A/B test **PPMI + randomized SVD (300–500d)** and/or **gensim SGNS** (minutes CPU); blend **potion-retrieval-32M** static embeddings if pretrained allowed | Levy et al. count vs predict parity [^37^][^38^]; potion numbers [^30^] |

## References

[^1^]: Karlgren, Holst & Sahlgren (2005), "Filaments of Meaning in Word Space" (EACL Wkshp), ACL Anthology W05-1711. https://aclanthology.org/W05-1711.pdf — "k=2 000, with 20 non-zero elements... Sahlgren has reported (Sahlgren, 2004) that... dimensionalities around 2 000 seem to be optimal."
[^2^]: Cohen, Widdows et al., "Predication-based Semantic Indexing: Permutations as a Means to Encode Predications in Semantic Space," PMC2815384; and Recchia, Sahlgren, Kanerva & Jones, "Encoding Sequential Information in Semantic Space Models" (PMC4405220) — RPM uses sparse ternary labels (two +1s, two −1s), window ±2, and "optimal performance when the order-encoding mechanism is restricted to direction information (only two distinct permutations)."
[^3^]: QasemiZadeh & Handschuh, "Random Indexing Explained with High Probability" — "for m = 1600, two non-zero elements per index vector are sufficient to construct a VSM."
[^4^]: Velldal, "Random Indexing Re-Hashed" — parameters of RI: number of non-zeros ε and dimensionality k of ternary index vectors.
[^5^]: CxGs+NLP 2023 proceedings (ACL Anthology 2023.cxgsnlp-1) describing Kanerva/Sahlgren RI — "10 non-zero elements in a 1000-dimensional vector"; random index vectors nearly orthogonal.
[^6^]: Sahlgren (2006) The Word-Space Model, as summarized in "Exploratory analysis of semantic categories" (Springer, s40469-015-0001-1) — small context → paradigmatic, larger → syntagmatic; also Bullinaria & Levy (2007).
[^7^]: Lapesa & Evert (2014), "A Large Scale Evaluation of Distributional Semantic Models: Parameters, Interactions and Model Selection," TACL 2. https://aclanthology.org/people/stefan-evert/
[^8^]: Levy, Goldberg & Dagan (2015) via Makrai thesis survey (hlt.bme.hu/media/pdf/makrai23-phd.pdf) and arXiv 1704.05781 "Redefining Context Windows" — word2vec dynamic window uniformly samples width in [1,L]; weights closer words more (w/w, (w−1)/w, ...); GloVe 1/d.
[^9^]: Same as [^8^]: GloVe uses 1/distance; word2vec triangular weighting.
[^10^]: Sahlgren, Holst & Kanerva (2008) Random Permutation model, summarized in Lapesa PhD thesis (osnadocs.../thesis_lapesa.pdf) — permutations have properties of circular convolution at lower cost; RPM trained on full Wikipedia with significant improvements.
[^11^]: Gorman & Curran (2006), "Scaling Distributional Similarity to Large Corpora," ACL W06-16 — "Random Index is robust for small corpora, but larger corpora require that the contexts be weighted to maintain accuracy"; TOEFL 48–51% document-based → 62–70% with narrow windows.
[^12^]: Ranganathan et al., "Discovery of Related Terms in a Corpus using Reflective Random Indexing" (umiacs.umd.edu/~oard/desi4/papers/rangan.pdf) — RI semantic space mined for query-expansion candidates.
[^13^]: Lu et al., "Synonym, Topic Model and Predicate-Based Query Expansion" (PMC3540443) — expansion raised recall 0.23→0.44–0.48, F +5–14pp, precision dropped.
[^14^]: Mandala, Tokunaga & Tanaka (1998), "The Use of WordNet in Information Retrieval" (W98-0704) — thesaurus expansion increases recall, degrades precision; combined thesauri best.
[^15^]: Bouma et al., CLEF 2007 QA run — Rocchio α=1, β=0.75, top-5 docs, max 10 new keywords, decay 0.1.
[^16^]: Matos et al., "Study of Query Expansion Techniques... Biomedical IR" (PMC3958669) — Rocchio parameter ranges M∈[10,100] docs, K∈[10,100] terms, α∈(0,4]; Terrier default ROCCHIO_BETA=0.4.
[^17^]: "What Is Query Expansion" (ITU Online) — original words usually carry the most weight; expansion raises recall, risks precision.
[^18^]: Radovanović et al. (2010), cited in Feldbauer & Flexer (2019) — hubs lie near data centroid; spatial centrality correlates with k-occurrence.
[^19^]: Feldbauer & Flexer (2019), "A comprehensive empirical comparison of hubness reduction in high-dimensional spaces," Knowl Inf Syst (PMC7327987) — recommends MP, LS, NICDM, DSL; SNN reduces hubness but hurts semantics; centering effective for text; MP-Gaussian approximation gives quadratic complexity.
[^20^]: Lample et al. (2018) CSLS, summarized in Obraczka & Rahm "Evaluation of Hubness Reduction Methods for Entity Resolution" (dbs.uni-leipzig.de) — CSLS = 2·d_xy − μ_x − μ_y.
[^21^]: *SEM 2016 (S16-2010) — NI local scaling d_xy/sqrt(μ_x·μ_y) mitigates hubness in embedding nearest-neighbor search.
[^22^]: ACL 2025 (2025.acl-long.1156) survey of anti-hubness for text embeddings — Globally Corrected Rank, Inverted Softmax, CSLS, Querybank Normalisation, DBNorm, NN normalization.
[^23^]: Pierrejean PhD thesis (hal.science tel-03148513) — embeddings encode frequency even after length normalization (Schnabel et al. 2015); mid-frequency words (10³–10⁴) most stable neighbors.
[^24^]: Basirat & Nivre, "Real-valued syntactic word vectors" (JETAI 2020) — "RI filters out the highly frequent context units, i.e., the function words."
[^25^]: Baroni, Evert & Lenci, ESSLLI 2008 course "Bridging the Gap" — excluded the 500 most frequent tokens as features; MI weighting beats log/log-entropy; small windows best for clustering.
[^26^]: DataAspirant, "BM25 Explained" — BM25F per-field weighting "usually a larger relevance win than any k1 or b adjustment."
[^27^]: RelevantSearch.AI Search Patterns Catalog — BM25F per-field weighting (title 3×); RRF k=60 fusion of lexical+vector lists.
[^28^]: know-cli (PyPI) — FTS5 BM25F field weighting plus file-path match boost for code retrieval.
[^29^]: pluck (GitHub hunhee98/pluck) — BM25F over symbol/signature/content fields for AST-chunk code search.
[^30^]: Model2Vec potion-retrieval-32M model card (ModelScope mirror) — static embeddings; post-training re-regularization = token frequency weighting + PCA + SIF weighting; MTEB Retrieval 35.06 vs all-MiniLM-L6-v2 42.92, orders of magnitude faster.
[^31^]: Achlioptas (2003), "Database-friendly random projections: Johnson-Lindenstrauss with binary coins," JCSS 66:671–687 — ternary {±√3,0} with prob {1/6,2/3,1/6} satisfies JL; 3× sparser than Gaussian.
[^32^]: sklearn-compatible SparseRandomProjection description (github.com/guillaume-osmo/mlx-addons) — Achlioptas ternary matrix, density 1/√d default; one sparse matmul.
[^33^]: Tissier, Gravier & Habrard, "Near-lossless Binarization of Word Embeddings" (AAAI 2019) — binarization costs ~2% accuracy with large size reduction; top-k benchmarks confirm.
[^34^]: MDPI Electronics (2218-?) binary CNN features — simple sign-binarization + Hamming matching, negligible performance loss (<1%).
[^35^]: "Binary Embeddings for Fast Search" (2024) — rank by Hamming, then recover what binary loses with float rerank.
[^36^]: blakecrosley.com Obsidian AI Search guide — Model2Vec int8 quantization (25% size) and 256→128 Matryoshka truncation with minimal quality loss; potion-base-8M MTEB 51.32 vs MiniLM 55.80 at 500× speed.
[^37^]: Levy, Goldberg & Dagan (2015), "Improving Distributional Similarity with Lessons Learned from Word Embeddings," TACL 3 — hyperparameters matter more than algorithm; PPMI-SVD competitive with SGNS; recommendations: context distribution smoothing 0.75, add w+c, SGNS robust baseline, many negatives, 3CosMul > 3CosAdd.
[^38^]: embedding-fairness repo study — PPMI/SVD "generally outperformed skipgram in rare word representation and semantic relationships"; skipgram more efficient only as data scales.
[^39^]: Levy & Goldberg (2014), "Neural Word Embedding as Implicit Matrix Factorization," NeurIPS — SGNS implicitly factorizes shifted PMI matrix.
[^40^]: Model2Vec GitHub (MinishLab) — up to 500× faster than sentence transformer on CPU; ~30 MB models; state of the art among static embeddings.
[^41^]: minish.ai Model2Vec results page — potion-base-32M = 93.21% of all-MiniLM-L6-v2 average MTEB (52.13) while orders of magnitude faster.
[^42^]: allaboutken.com, "Semantic search on a static site" (2026) — potion 2M/4M/8M missed MiniLM's correct hits on a small personal corpus; ~32% top-result overlap; short generic pages become false attractors under mean pooling.
