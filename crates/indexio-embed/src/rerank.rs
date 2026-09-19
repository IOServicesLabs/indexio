//! Reranker stage (SPEC-P3 §2): a pluggable second-pass ranker applied
//! after hybrid RRF fusion.
//!
//! Three implementations:
//! - [`OverlapReranker`] — offline, deterministic token-overlap scorer with
//!   sqrt length normalization and a bigram bonus. No deps, no I/O.
//! - [`NoopReranker`] — identity pass-through (A/B baseline, tests).
//! - [`HttpReranker`] — TEI/vLLM/Jina-style `POST {base}/rerank` client
//!   (production path; tested only against local mock servers — no model
//!   weights or downloads anywhere in this crate).

use anyhow::{anyhow, Context as _};

/// Score docs against a query. Returns `(doc_index, score)` pairs sorted by
/// score descending; ties keep the input order (stable sort).
pub trait Reranker: Send + Sync {
    fn model_id(&self) -> &str;
    /// Score `docs` against `query`. Implementations return one entry per
    /// doc (callers tolerate missing entries by keeping RRF order for them).
    fn rerank(&self, query: &str, docs: &[String]) -> anyhow::Result<Vec<(usize, f64)>>;
}

// ---------------------------------------------------------------------------
// OverlapReranker — deterministic offline token overlap
// ---------------------------------------------------------------------------

/// Offline deterministic reranker: count of distinct query tokens present
/// in the doc, plus a bonus per query bigram present in the doc, weighted
/// by `sqrt(1 / (1 + doc_token_count))` length normalization. L2-free,
/// no deps, fully deterministic.
pub struct OverlapReranker;

/// Score added per query bigram found (in order) in the doc.
const BIGRAM_BONUS: f64 = 1.0;

/// Tokenize: lowercase, split on non-alphanumeric, drop empties.
fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Overlap score for one doc (pure; exposed for tests).
fn overlap_score(query_tokens: &[String], doc_tokens: &[String]) -> f64 {
    let mut distinct_hits = 0usize;
    for (i, qt) in query_tokens.iter().enumerate() {
        if !query_tokens[..i].contains(qt) && doc_tokens.contains(qt) {
            distinct_hits += 1;
        }
    }
    let mut bigram_hits = 0usize;
    for pair in query_tokens.windows(2) {
        if doc_tokens.windows(2).any(|w| w == pair) {
            bigram_hits += 1;
        }
    }
    let norm = (1.0 / (1.0 + doc_tokens.len() as f64)).sqrt();
    (distinct_hits as f64 + BIGRAM_BONUS * bigram_hits as f64) * norm
}

impl Reranker for OverlapReranker {
    fn model_id(&self) -> &str {
        "overlap-v1"
    }

    fn rerank(&self, query: &str, docs: &[String]) -> anyhow::Result<Vec<(usize, f64)>> {
        let q = tokens(query);
        let mut scored: Vec<(usize, f64)> = docs
            .iter()
            .enumerate()
            .map(|(i, d)| (i, overlap_score(&q, &tokens(d))))
            .collect();
        // Stable sort: ties keep input (RRF) order.
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(scored)
    }
}

// ---------------------------------------------------------------------------
// NoopReranker — identity order
// ---------------------------------------------------------------------------

/// Identity order (pass-through), for A/B and tests: every doc scores 0.0,
/// so the stable ordering is exactly the input order.
pub struct NoopReranker;

impl Reranker for NoopReranker {
    fn model_id(&self) -> &str {
        "noop"
    }

    fn rerank(&self, _query: &str, docs: &[String]) -> anyhow::Result<Vec<(usize, f64)>> {
        Ok((0..docs.len()).map(|i| (i, 0.0)).collect())
    }
}

// ---------------------------------------------------------------------------
// HttpReranker — TEI/vLLM/Jina-style /rerank client
// ---------------------------------------------------------------------------

/// TEI/vLLM/Jina-style rerank endpoint: `POST {base}/rerank` with
/// `{"model":..,"query":..,"documents":[..]}`, expecting
/// `{"results":[{"index":i,"relevance_score":s},...]}`. Bearer auth when
/// the API key is non-empty.
pub struct HttpReranker {
    base: String,
    model: String,
    api_key: String,
    /// Cached "http:<model>" for `Reranker::model_id` (needs &str return).
    model_id: String,
}

impl HttpReranker {
    /// Env configuration: `Some` when INDEXIO_RERANK_BASE is set (then
    /// INDEXIO_RERANK_MODEL is required; INDEXIO_RERANK_KEY optional), else `None`.
    pub fn from_env() -> anyhow::Result<Option<Self>> {
        let Ok(base) = std::env::var("INDEXIO_RERANK_BASE") else {
            return Ok(None);
        };
        if base.trim().is_empty() {
            return Ok(None);
        }
        let model = std::env::var("INDEXIO_RERANK_MODEL")
            .map_err(|_| anyhow!("INDEXIO_RERANK_BASE is set but INDEXIO_RERANK_MODEL is not"))?;
        let api_key = std::env::var("INDEXIO_RERANK_KEY").unwrap_or_default();
        Ok(Some(HttpReranker {
            base: base.trim_end_matches('/').to_string(),
            model_id: format!("http:{model}"),
            model,
            api_key,
        }))
    }
}

impl Reranker for HttpReranker {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn rerank(&self, query: &str, docs: &[String]) -> anyhow::Result<Vec<(usize, f64)>> {
        use ureq::serde_json::{json, Value};
        let url = format!("{}/rerank", self.base);
        let body = json!({
            "model": self.model,
            "query": query,
            "documents": docs,
        });
        let mut req = ureq::post(&url);
        if !self.api_key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", self.api_key));
        }
        let resp = req
            .send_json(body)
            .map_err(|e| anyhow!("rerank request to {url} failed: {e}"))?;
        let v: Value = resp
            .into_json()
            .with_context(|| format!("rerank response from {url} is not JSON"))?;
        let results = v
            .get("results")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("rerank response missing \"results\" array"))?;
        let mut out: Vec<(usize, f64)> = Vec::with_capacity(results.len());
        for r in results {
            let index = r
                .get("index")
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow!("rerank result missing \"index\""))?
                as usize;
            let score = r
                .get("relevance_score")
                .and_then(Value::as_f64)
                .ok_or_else(|| anyhow!("rerank result missing \"relevance_score\""))?;
            anyhow::ensure!(
                index < docs.len(),
                "rerank result index {index} out of range ({} docs)",
                docs.len()
            );
            out.push((index, score));
        }
        // Stable sort desc (server order is already desc, but don't rely on it).
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead as _, BufReader, Read as _, Write as _};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    #[test]
    fn overlap_ranks_term_doc_above_distractor() {
        let r = OverlapReranker;
        let docs = vec![
            "completely unrelated yaml billing configuration loader".to_string(),
            "embedding vector cosine ranking over code chunks".to_string(),
        ];
        let ranked = r
            .rerank("embedding vector cosine ranking", &docs)
            .unwrap();
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].0, 1, "term-containing doc must rank first");
        assert!(ranked[0].1 > 0.0);
        assert_eq!(ranked[1].0, 0);
        assert_eq!(ranked[1].1, 0.0, "distractor has zero overlap");
    }

    #[test]
    fn overlap_length_normalization_prefers_shorter_doc() {
        // Both docs contain the query term exactly once; the shorter doc
        // must win via the sqrt(1/(1+n)) length normalization.
        let short = tokens("vector search");
        let mut long = tokens("vector search");
        long.extend(std::iter::repeat("filler".to_string()).take(50));
        let s_short = overlap_score(&tokens("vector"), &short);
        let s_long = overlap_score(&tokens("vector"), &long);
        assert!(s_short > s_long, "{s_short} vs {s_long}");
        // Sanity: normalization is sqrt(1/(1+n)), so score shrinks with n.
        let expected = 1.0 * (1.0 / (1.0 + short.len() as f64)).sqrt();
        assert!((s_short - expected).abs() < 1e-12);
        // Bigram bonus: a doc preserving the query bigram beats one that
        // merely contains both tokens out of order.
        let ordered = overlap_score(&tokens("cosine similarity"), &tokens("cosine similarity"));
        let shuffled = overlap_score(&tokens("cosine similarity"), &tokens("similarity cosine"));
        assert!(ordered > shuffled, "{ordered} vs {shuffled}");
    }

    #[test]
    fn noop_reranker_is_identity() {
        let r = NoopReranker;
        let docs: Vec<String> = (0..5).map(|i| format!("doc {i}")).collect();
        let ranked = r.rerank("anything", &docs).unwrap();
        let idxs: Vec<usize> = ranked.iter().map(|(i, _)| *i).collect();
        assert_eq!(idxs, vec![0, 1, 2, 3, 4], "identity order");
        assert!(ranked.iter().all(|(_, s)| *s == 0.0));
        assert_eq!(r.model_id(), "noop");
        assert!(r.rerank("q", &[]).unwrap().is_empty());
    }

    #[test]
    fn overlap_deterministic_and_stable_ties() {
        let r = OverlapReranker;
        let docs = vec![
            "alpha beta".to_string(),
            "gamma delta".to_string(), // tie: zero overlap, keeps position
            "alpha beta".to_string(),  // tie with doc 0
        ];
        let ranked = r.rerank("nothing here", &docs).unwrap();
        let idxs: Vec<usize> = ranked.iter().map(|(i, _)| *i).collect();
        assert_eq!(idxs, vec![0, 1, 2], "stable input order on ties");
    }

    // ---- HttpReranker against a local mock server -------------------------

    struct Captured {
        request_line: String,
        headers: Vec<(String, String)>,
        body: String,
    }

    /// One-shot mock TEI server: captures the request, replies with `body`.
    fn start_mock(body: String) -> (String, Arc<Mutex<Vec<Captured>>>, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let captured: Arc<Mutex<Vec<Captured>>> = Arc::new(Mutex::new(Vec::new()));
        let cap = Arc::clone(&captured);
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            let mut headers = Vec::new();
            let mut content_len = 0usize;
            let mut line = String::new();
            loop {
                line.clear();
                reader.read_line(&mut line).unwrap();
                let t = line.trim_end();
                if t.is_empty() {
                    break;
                }
                if let Some((k, v)) = t.split_once(':') {
                    if k.trim().eq_ignore_ascii_case("content-length") {
                        content_len = v.trim().parse().unwrap_or(0);
                    }
                    headers.push((k.trim().to_string(), v.trim().to_string()));
                }
            }
            let mut body_bytes = vec![0u8; content_len];
            reader.read_exact(&mut body_bytes).unwrap();
            cap.lock().unwrap().push(Captured {
                request_line: request_line.trim_end().to_string(),
                headers,
                body: String::from_utf8_lossy(&body_bytes).into_owned(),
            });
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(resp.as_bytes()).unwrap();
            let _ = stream.flush();
        });
        (format!("http://127.0.0.1:{port}"), captured, handle)
    }

    #[test]
    fn http_reranker_request_shape_and_reorder() {
        // Server scores doc 1 highest, doc 2 mid, doc 0 lowest.
        let body =
            r#"{"results":[{"index":1,"relevance_score":0.91},{"index":2,"relevance_score":0.5},{"index":0,"relevance_score":0.01}]}"#
                .to_string();
        let (base, captured, handle) = start_mock(body);
        let r = HttpReranker {
            base: base.clone(),
            model: "bge-reranker-v2".to_string(),
            api_key: "secret-key".to_string(),
            model_id: "http:bge-reranker-v2".to_string(),
        };
        assert_eq!(r.model_id(), "http:bge-reranker-v2");
        let docs = vec![
            "first doc".to_string(),
            "second doc".to_string(),
            "third doc".to_string(),
        ];
        let ranked = r.rerank("rank these docs", &docs).unwrap();
        handle.join().unwrap();
        // Reorder applied: server order (desc relevance) is honored.
        let idxs: Vec<usize> = ranked.iter().map(|(i, _)| *i).collect();
        assert_eq!(idxs, vec![1, 2, 0]);
        assert!((ranked[0].1 - 0.91).abs() < 1e-12);

        // Request shape asserted.
        let cap = captured.lock().unwrap();
        let req = &cap[0];
        assert_eq!(req.request_line, "POST /rerank HTTP/1.1");
        let auth = req
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
            .map(|(_, v)| v.as_str());
        assert_eq!(auth, Some("Bearer secret-key"));
        let v: ureq::serde_json::Value =
            ureq::serde_json::from_str(&req.body).unwrap();
        assert_eq!(v["model"], "bge-reranker-v2");
        assert_eq!(v["query"], "rank these docs");
        assert_eq!(
            v["documents"],
            ureq::serde_json::json!(["first doc", "second doc", "third doc"])
        );
    }

    #[test]
    fn http_reranker_rejects_out_of_range_index() {
        let body = r#"{"results":[{"index":7,"relevance_score":0.9}]}"#.to_string();
        let (base, _cap, _h) = start_mock(body);
        let r = HttpReranker {
            base,
            model: "m".to_string(),
            api_key: String::new(),
            model_id: "http:m".to_string(),
        };
        let err = r.rerank("q", &["only doc".to_string()]).unwrap_err();
        assert!(format!("{err}").contains("out of range"), "{err}");
    }

    #[test]
    fn from_env_semantics() {
        // Unset base -> None. (No other test touches INDEXIO_RERANK_* vars.)
        std::env::remove_var("INDEXIO_RERANK_BASE");
        std::env::remove_var("INDEXIO_RERANK_MODEL");
        std::env::remove_var("INDEXIO_RERANK_KEY");
        assert!(HttpReranker::from_env().unwrap().is_none());
        // Base without model -> error.
        std::env::set_var("INDEXIO_RERANK_BASE", "http://127.0.0.1:1");
        let err = match HttpReranker::from_env() {
            Err(e) => e,
            Ok(_) => panic!("expected error when INDEXIO_RERANK_MODEL is unset"),
        };
        assert!(format!("{err}").contains("INDEXIO_RERANK_MODEL"), "{err}");
        // Base + model -> Some; key optional; trailing slash trimmed.
        std::env::set_var("INDEXIO_RERANK_BASE", "http://127.0.0.1:1/");
        std::env::set_var("INDEXIO_RERANK_MODEL", "jina-reranker-v2");
        let r = HttpReranker::from_env().unwrap().unwrap();
        assert_eq!(r.base, "http://127.0.0.1:1");
        assert_eq!(r.model, "jina-reranker-v2");
        assert!(r.api_key.is_empty());
        std::env::remove_var("INDEXIO_RERANK_BASE");
        std::env::remove_var("INDEXIO_RERANK_MODEL");
    }
}
