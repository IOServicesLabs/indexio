//! Embedders (SPEC-P2 §2.1).

use anyhow::{anyhow, Context};

/// Pluggable embedding provider. Outputs are L2-normalized, so cosine
/// similarity is the plain dot product.
pub trait Embedder: Send + Sync {
    /// e.g. "hash-v1" or "http:Qwen3-Embedding-8B".
    fn model_id(&self) -> &str;
    fn dim(&self) -> usize;
    /// Embed a batch of texts; returns one L2-normalized vector per input.
    fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;
    /// Stateful embedders ingest new texts into their model before
    /// embedding (SPEC-P4 §1). Default no-op (stateless embedders). Called
    /// by the pipeline with each repo's new (CAS-miss) chunk texts, or ALL
    /// chunk texts on a model rebuild.
    fn observe(&self, _texts: &[String]) -> anyhow::Result<()> {
        Ok(())
    }
    /// Persist any model state. Default no-op. Called by the pipeline
    /// after each repo. A lazily persisted embedder (the MCP server's) only
    /// notes that a save is due here; see [`save_due`](Self::save_due).
    fn flush(&self) -> anyhow::Result<()> {
        Ok(())
    }
    /// Whether a lazily persisted model has unsaved changes older than its
    /// interval (SPEC-P10 §30): the owner runs [`persist`](Self::persist)
    /// on a background thread instead of paying the save inside a call.
    fn save_due(&self) -> bool {
        false
    }
    /// Save the model now if it changed. Default no-op.
    fn persist(&self) -> anyhow::Result<()> {
        Ok(())
    }
    /// True when re-embedding a chunk is cheaper than reading its cached
    /// vector back from disk (in-process models): the embedding cache then
    /// keeps only chunk hashes (SPEC-P10). Default false (remote / paid
    /// embedders keep their vectors).
    fn recompute_is_cheap(&self) -> bool {
        false
    }
    /// Thesaurus query expansion (SPEC-P5 B2): stateful embedders with a
    /// learned vocabulary may append nearest-neighbor terms; the default
    /// returns the query unchanged (stateless embedders).
    fn expand_query(&self, q: &str) -> String {
        q.to_string()
    }
    /// How many texts a stateful embedder's model has learned from
    /// (SPEC-P10 §20): rows embedded long before the model's current state
    /// drift away from it; `None` for stateless embedders.
    fn texts_seen(&self) -> Option<u64> {
        None
    }
}

/// English function words dropped from QUERY text (never from indexed
/// content) by the BM25F leg (crate::bm25) and the indexio-query semantic path
/// (SPEC-P5 §A4). ~40 words; query-side only — the conjunctive lexical mode
/// is untouched on purpose (precision is a feature there).
pub fn stopwords() -> &'static [&'static str] {
    &STOPWORDS
}

static STOPWORDS: [&str; 49] = [
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
    "does", "for", "from", "had", "has", "have", "how", "i", "if", "in",
    "into", "is", "it", "its", "not", "of", "on", "or", "so", "that", "the",
    "their", "them", "there", "these", "they", "this", "to", "was", "we",
    "what", "when", "where", "which", "who", "will", "with", "you",
];

/// cosine(a,b) for L2-normalized vecs = dot product.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2-normalize in place. Zero vectors stay zero (no NaN).
pub(crate) fn l2_normalize(v: &mut [f32]) {
    let n = dot(v, v).sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

// ---------------------------------------------------------------------------
// HashEmbedder — deterministic, offline
// ---------------------------------------------------------------------------

/// Deterministic offline embedder (SPEC-P2 §2.1): tokenize on
/// non-alphanumerics, lowercase, hash each token and each token bigram via
/// blake3[..8] into `dim` buckets with signed hashing (+1/-1 from a second
/// hash bit — byte 8 of the same 32-byte blake3 digest), L2-normalize.
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    /// `dim` is clamped to >= 1. Default convention: 512.
    pub fn new(dim: usize) -> Self {
        HashEmbedder { dim: dim.max(1) }
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        HashEmbedder::new(512)
    }
}

fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

fn add_hashed(acc: &mut [f32], key: &[u8]) {
    let h = blake3::hash(key);
    let b = h.as_bytes();
    let bucket = u64::from_le_bytes(b[..8].try_into().expect("blake3 is 32 bytes")) as usize
        % acc.len();
    // Signed hashing: a "second hash bit" — bit 0 of byte 8 of the digest.
    let sign = if b[8] & 1 == 0 { 1.0f32 } else { -1.0f32 };
    acc[bucket] += sign;
}

impl Embedder for HashEmbedder {
    fn model_id(&self) -> &str {
        "hash-v1"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn recompute_is_cheap(&self) -> bool {
        true
    }

    fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let toks = tokens(t);
                let mut v = vec![0.0f32; self.dim];
                for w in &toks {
                    add_hashed(&mut v, w.as_bytes());
                }
                for pair in toks.windows(2) {
                    let mut key = String::with_capacity(pair[0].len() + 1 + pair[1].len());
                    key.push_str(&pair[0]);
                    key.push(' ');
                    key.push_str(&pair[1]);
                    add_hashed(&mut v, key.as_bytes());
                }
                l2_normalize(&mut v);
                v
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// HttpEmbedder — OpenAI-compatible /v1/embeddings
// ---------------------------------------------------------------------------

/// OpenAI-compatible embeddings endpoint (production path: self-hosted
/// Qwen3-Embedding via vLLM or TEI). Batches <= 32 texts per request.
pub struct HttpEmbedder {
    base: String,
    model: String,
    api_key: String,
    dim: usize,
    /// Cached "http:<model>" for `Embedder::model_id` (needs &str return).
    model_id: String,
}

const HTTP_BATCH: usize = 32;

impl HttpEmbedder {
    /// Reads INDEXIO_EMBED_BASE (required), INDEXIO_EMBED_MODEL, INDEXIO_EMBED_KEY,
    /// INDEXIO_EMBED_DIM (default 1024) env vars.
    pub fn from_env() -> anyhow::Result<Self> {
        let base = std::env::var("INDEXIO_EMBED_BASE")
            .map_err(|_| anyhow!("INDEXIO_EMBED_BASE is not set"))?;
        let model = std::env::var("INDEXIO_EMBED_MODEL")
            .map_err(|_| anyhow!("INDEXIO_EMBED_MODEL is not set"))?;
        let api_key = std::env::var("INDEXIO_EMBED_KEY").unwrap_or_default();
        let dim = std::env::var("INDEXIO_EMBED_DIM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1024);
        Ok(HttpEmbedder {
            base: base.trim_end_matches('/').to_string(),
            model_id: format!("http:{model}"),
            model,
            api_key,
            dim: dim.max(1),
        })
    }
}

impl Embedder for HttpEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        use ureq::serde_json::{json, Value};
        let url = format!("{}/v1/embeddings", self.base);
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for batch in texts.chunks(HTTP_BATCH) {
            let body = json!({ "model": self.model, "input": batch });
            let mut req = ureq::post(&url);
            if !self.api_key.is_empty() {
                req = req.set("Authorization", &format!("Bearer {}", self.api_key));
            }
            let resp = req
                .send_json(body)
                .map_err(|e| anyhow!("embeddings request to {url} failed: {e}"))?;
            let v: Value = resp
                .into_json()
                .with_context(|| format!("embeddings response from {url} is not JSON"))?;
            let data = v
                .get("data")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("embeddings response missing \"data\" array"))?;
            anyhow::ensure!(
                data.len() == batch.len(),
                "embeddings endpoint returned {} vectors for {} inputs",
                data.len(),
                batch.len()
            );
            // OpenAI order matches input; honor the "index" field when present.
            let mut items: Vec<(usize, Vec<f32>)> = Vec::with_capacity(batch.len());
            for (pos, item) in data.iter().enumerate() {
                let idx = item
                    .get("index")
                    .and_then(Value::as_u64)
                    .map(|i| i as usize)
                    .unwrap_or(pos);
                let arr = item
                    .get("embedding")
                    .and_then(Value::as_array)
                    .ok_or_else(|| anyhow!("embeddings response item missing \"embedding\""))?;
                let mut vec: Vec<f32> = Vec::with_capacity(arr.len());
                for x in arr {
                    vec.push(
                        x.as_f64()
                            .ok_or_else(|| anyhow!("embedding element is not a number"))?
                            as f32,
                    );
                }
                anyhow::ensure!(
                    vec.len() == self.dim,
                    "embedding dim {} != configured INDEXIO_EMBED_DIM {}",
                    vec.len(),
                    self.dim
                );
                l2_normalize(&mut vec);
                items.push((idx, vec));
            }
            items.sort_by_key(|(i, _)| *i);
            out.extend(items.into_iter().map(|(_, v)| v));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn emb(e: &dyn Embedder, text: &str) -> Vec<f32> {
        e.embed(&[text.to_string()]).unwrap().remove(0)
    }

    #[test]
    fn stopwords_list_shape() {
        let sw = stopwords();
        assert!((35..=60).contains(&sw.len()), "~40 words: {}", sw.len());
        for w in sw {
            assert!(!w.is_empty() && w.chars().all(|c| c.is_ascii_lowercase()));
        }
        // Core English function words are covered.
        for w in ["the", "of", "and", "to", "in", "is", "for", "with"] {
            assert!(sw.contains(&w), "missing '{w}'");
        }
        // Content words are NOT stopwords.
        for w in ["login", "parse", "cache", "retry"] {
            assert!(!sw.contains(&w), "'{w}' must not be a stopword");
        }
    }

    #[test]
    fn hash_embedder_determinism() {
        let e = HashEmbedder::new(512);
        let a = emb(&e, "fn parse_query(input: &str) -> Result<Query, QueryError>");
        let b = emb(&e, "fn parse_query(input: &str) -> Result<Query, QueryError>");
        assert_eq!(a, b, "same text must embed bit-identically");
        // Different texts should (overwhelmingly) differ.
        let c = emb(&e, "completely unrelated database connection pooling");
        assert_ne!(a, c);
    }

    #[test]
    fn hash_embedder_dim_and_normalization() {
        for dim in [8, 512, 1024] {
            let e = HashEmbedder::new(dim);
            assert_eq!(e.dim(), dim);
            let v = emb(&e, "hello world embedding dimension test");
            assert_eq!(v.len(), dim);
            let norm = dot(&v, &v).sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "norm={norm} for dim={dim}");
        }
        // Batch embed returns one vector per input.
        let e = HashEmbedder::new(64);
        let out = e
            .embed(&["a".to_string(), "b".to_string(), "c".to_string()])
            .unwrap();
        assert_eq!(out.len(), 3);
        // Empty text -> zero vector, must not NaN.
        let z = emb(&e, "   !!!   ");
        assert!(z.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn hash_embedder_similarity_sanity() {
        let e = HashEmbedder::new(512);
        let a = emb(
            &e,
            "the quick brown fox jumps over the lazy dog while running fast",
        );
        let similar = emb(&e, "quick brown fox jumps over lazy dog running");
        let dissimilar = emb(
            &e,
            "kubernetes pod eviction policy gracperful shutdown webhook tls",
        );
        let s_sim = dot(&a, &similar);
        let s_dis = dot(&a, &dissimilar);
        assert!(
            s_sim > s_dis,
            "similar text should score higher: {s_sim} vs {s_dis}"
        );
        // Identical text scores ~1.0 (self cosine).
        assert!(dot(&a, &a) > 0.99);
        // dot() itself: orthogonal/unit basics.
        assert_eq!(dot(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
        assert_eq!(dot(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
    }
}
