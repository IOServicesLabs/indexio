//! HTTP server (axum 0.8, tokio): JSON query API + admin reload.
//!
//! Routes (contract: docs/SPEC.md + SPEC-P2 §4, section "indexio — binary"):
//!   GET  /health                 -> {"ok":true}
//!   GET  /stats                  -> EngineStats (merged with CAS stats)
//!   GET  /search?q=&limit=&mode=&rerank=&fusion= -> {"hits":[...]} (mode: lexical|semantic|hybrid,
//!                                         default lexical; rerank=1 only with mode=hybrid;
//!                                         fusion: rrf|combmnz, default rrf, hybrid only)
//!   GET  /symbol/{name}          -> [SearchHit...]
//!   GET  /calls/{name}           -> [SearchHit...]
//!   POST /admin/reload           -> reopen the Engine (drops mmap'd shards)
//!   POST /admin/embed            -> embed all repos, returns {"reports":[EmbedReport...]}
//!   GET  /embcas/stats           -> {"model_id":...,"entries":u64,"bytes":u64}
//!   GET  /impact/symbol?name=a,b&depth=&max_sites= -> ImpactReport (SPEC-P6 §3)
//!   POST /impact/diff {"repo","diff","depth"?,"max_sites"?} -> ImpactReport + "changed"
//!   GET  /outline?repo=&path=    -> {"repo","path","items":[OutlineItem...]}
//!   GET  /span?repo=&path=&start=&end= -> {"repo","path","start","end","text"}
//!
//! Bound to 127.0.0.1 unless `--bind` names another address, which the
//! CLI only allows together with a bearer token or an ACL file.
//!
//! Auth + repo-level ACLs (SPEC-P3 §3): with `--auth-token`/INDEXIO_AUTH_TOKEN
//! every route except `GET /health` requires `Authorization: Bearer T`
//! (else 401 `{"error":"unauthorized"}`). `--acl-file` supersedes it: a
//! per-token repo allowlist map; unknown tokens get 401, search/symbol/calls
//! hits are filtered by the allow patterns, and admin routes additionally
//! require an allow entry of exactly `*` (else 403 `{"error":"forbidden"}`).
//! Neither flag -> open behavior (unchanged). MCP over stdio is the
//! trusted local channel and is intentionally NOT authenticated.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::Context;
use axum::extract::{Path, Query, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse as _, Json, Response};
use axum::routing::{get, post};
use axum::{Extension, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tower_http::trace::TraceLayer;

use indexio_embed::embed::Embedder;
use indexio_embed::pipeline::{embed_all, EmbedReport};
use indexio_embed::rerank::Reranker;
use indexio_embed::store::EmbedCas;
use indexio_query::{Engine, FusionAlgo, HybridHit, SearchMode, SearchResult};
use indexio_types::SearchHit;

/// Default hit limit when `limit` is absent (search) or for symbol/calls.
const DEFAULT_LIMIT: usize = 50;

// ---------------------------------------------------------------------------
// Auth + repo ACLs (SPEC-P3 §3)
// ---------------------------------------------------------------------------

/// Server-side auth configuration (selected once at startup).
#[derive(Clone)]
pub enum AuthConfig {
    /// No flags: current open behavior.
    Open,
    /// `--auth-token T` / INDEXIO_AUTH_TOKEN: one bearer token for everything.
    Token(String),
    /// `--acl-file`: token -> repo allow patterns (supersedes Token).
    Acl(Arc<HashMap<String, Vec<String>>>),
}

/// Allow patterns of the authenticated ACL token, attached to the request
/// by the auth middleware (ACL mode only).
#[derive(Clone)]
pub struct AclAllow(pub Arc<Vec<String>>);

/// Load an ACL file: `{"tokens":{"tokA":{"allow":["core","team-*"]},...}}`.
pub fn load_acl_file(path: &std::path::Path) -> anyhow::Result<HashMap<String, Vec<String>>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading ACL file {}", path.display()))?;
    let v: Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing ACL file {} as JSON", path.display()))?;
    let tokens = v
        .get("tokens")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("ACL file missing top-level \"tokens\" object"))?;
    let mut map = HashMap::with_capacity(tokens.len());
    for (tok, spec) in tokens {
        let allow = spec
            .get("allow")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("ACL token \"{tok}\" missing \"allow\" array"))?;
        let mut pats = Vec::with_capacity(allow.len());
        for p in allow {
            pats.push(
                p.as_str()
                    .ok_or_else(|| anyhow::anyhow!("ACL token \"{tok}\" has a non-string allow pattern"))?
                    .to_string(),
            );
        }
        map.insert(tok.clone(), pats);
    }
    Ok(map)
}

/// Repo allow-pattern match (SPEC-P3 §3): `*` matches anything; `prefix-*`
/// is a prefix glob; `*-suffix` a suffix glob; `*mid*` contains; anything
/// else is an exact match. Hand-rolled — no glob crate.
pub fn repo_allowed(patterns: &[String], repo: &str) -> bool {
    patterns.iter().any(|p| {
        if p == "*" {
            return true;
        }
        let lead = p.starts_with('*');
        let trail = p.ends_with('*') && p.len() > 1;
        match (lead, trail) {
            (true, true) => repo.contains(&p[1..p.len() - 1]),
            (true, false) => repo.ends_with(&p[1..]),
            (false, true) => repo.starts_with(&p[..p.len() - 1]),
            (false, false) => p == repo,
        }
    })
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({ "error": "unauthorized" })),
    )
        .into_response()
}

fn forbidden() -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(json!({ "error": "forbidden" })),
    )
        .into_response()
}

/// Extract the bearer token from the Authorization header.
fn bearer_token(req: &Request) -> Option<&str> {
    req.headers()
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
}

/// Auth middleware: `GET /health` is always open; everything else needs a
/// valid bearer token when auth is configured. In ACL mode the token's
/// allow patterns are attached as a request extension for hit filtering,
/// and `/admin/*` additionally requires an allow entry of exactly `*`.
async fn auth_middleware(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Response {
    let auth = state.auth.clone();
    if matches!(auth.as_ref(), AuthConfig::Open) {
        return next.run(req).await;
    }
    if req.method() == axum::http::Method::GET && req.uri().path() == "/health" {
        return next.run(req).await;
    }
    let Some(tok) = bearer_token(&req).map(str::to_string) else {
        return unauthorized();
    };
    match auth.as_ref() {
        AuthConfig::Open => unreachable!("handled above"),
        AuthConfig::Token(expected) => {
            if tok == *expected {
                next.run(req).await
            } else {
                unauthorized()
            }
        }
        AuthConfig::Acl(map) => {
            let Some(patterns) = map.get(&tok) else {
                return unauthorized();
            };
            if req.uri().path().starts_with("/admin/")
                && !patterns.iter().any(|p| p == "*")
            {
                return forbidden();
            }
            req.extensions_mut()
                .insert(AclAllow(Arc::new(patterns.clone())));
            next.run(req).await
        }
    }
}

/// Hit filter: in ACL mode keep only hits whose repo matches an allow
/// pattern; otherwise (open/token) everything passes.
fn hit_allowed(acl: &Option<Extension<AclAllow>>, repo: &str) -> bool {
    match acl {
        None => true,
        Some(Extension(allow)) => repo_allowed(&allow.0, repo),
    }
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    /// Engine holds mmap'd shards; /admin/reload replaces the contents of
    /// this lock with a freshly opened Engine.
    engine: Arc<RwLock<Engine>>,
    /// Embedder for semantic/hybrid search and /admin/embed, held behind
    /// the same RwLock pattern as the engine so it can be swapped later.
    embedder: Arc<RwLock<Arc<dyn Embedder>>>,
    /// Reranker for `/search?...&rerank=1`, selected once at startup
    /// (SPEC-P3 §2: HttpReranker::from_env() if Some, else OverlapReranker).
    reranker: Arc<dyn Reranker>,
    /// Auth/ACL configuration (SPEC-P3 §3).
    auth: Arc<AuthConfig>,
    data_dir: Arc<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct SearchParams {
    q: Option<String>,
    limit: Option<usize>,
    mode: Option<String>,
    rerank: Option<String>,
    /// SPEC-P5 §A3: fusion algorithm for mode=hybrid (rrf|combmnz,
    /// default rrf).
    fusion: Option<String>,
}

/// Serialize a hit to JSON (pure; SearchHit is Serialize).
pub fn hit_to_json(h: &SearchHit) -> Value {
    serde_json::to_value(h).expect("SearchHit is Serialize")
}

/// Serialize a fused hybrid hit (SPEC-P2 §4 + P3 §2 `rerank_score`;
/// shared by HTTP, CLI and MCP).
pub fn hybrid_hit_to_json(h: &HybridHit) -> Value {
    json!({
        "hit": hit_to_json(&h.hit),
        "rrf": h.rrf,
        "lex_rank": h.lex_rank,
        "bm25_rank": h.bm25_rank,
        "sem_rank": h.sem_rank,
        "rerank_score": h.rerank_score,
    })
}

/// Serialize an EmbedReport (not serde; pure).
fn embed_report_to_json(r: &EmbedReport) -> Value {
    json!({
        "repo": r.repo,
        "chunks": r.chunks,
        "cas_hits": r.cas_hits,
        "cas_misses": r.cas_misses,
        "embedded": r.embedded,
        "elapsed_ms": r.elapsed_ms,
        "index_build_ms": r.index_build_ms,
    })
}

/// Serialize a SearchResult to the HTTP/CLI JSON shape (pure).
pub fn search_result_to_json(res: &SearchResult) -> Value {
    json!({
        "hits": res.hits.iter().map(hit_to_json).collect::<Vec<_>>(),
        "truncated": res.truncated,
        "took_ms": res.took_ms,
    })
}

fn bad_request(msg: impl Into<String>) -> (StatusCode, Json<Value>) {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": msg.into() })))
}

fn internal_error(msg: impl Into<String>) -> (StatusCode, Json<Value>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({ "error": msg.into() })),
    )
}

async fn health() -> Json<Value> {
    Json(json!({ "ok": true }))
}

async fn stats(State(state): State<AppState>) -> Json<Value> {
    let stats = crate::merged_stats(&state.data_dir).unwrap_or_default();
    Json(serde_json::to_value(stats).expect("EngineStats is Serialize"))
}

/// Parse the `rerank` query param: absent/0/false -> off, 1/true -> on.
fn parse_rerank_param(v: Option<&str>) -> Result<bool, (StatusCode, Json<Value>)> {
    match v {
        None | Some("0") | Some("false") => Ok(false),
        Some("1") | Some("true") => Ok(true),
        Some(other) => Err(bad_request(format!(
            "invalid rerank value '{other}' (expected 0|1)"
        ))),
    }
}

async fn search(
    State(state): State<AppState>,
    acl: Option<Extension<AclAllow>>,
    Query(params): Query<SearchParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let Some(q) = params.q else {
        return Err(bad_request("missing query parameter 'q'"));
    };
    let limit = params.limit.unwrap_or(DEFAULT_LIMIT);
    let mode = match &params.mode {
        None => SearchMode::Lexical,
        Some(m) => SearchMode::parse(m).map_err(bad_request)?,
    };
    let rerank = parse_rerank_param(params.rerank.as_deref())?;
    if rerank && mode != SearchMode::Hybrid {
        return Err(bad_request("rerank=1 requires mode=hybrid"));
    }
    // SPEC-P5 §A3: optional fusion algorithm (hybrid mode only).
    let fusion = match &params.fusion {
        None => FusionAlgo::Rrf,
        Some(f) => FusionAlgo::parse(f).map_err(bad_request)?,
    };
    if fusion != FusionAlgo::Rrf && mode != SearchMode::Hybrid {
        return Err(bad_request("fusion requires mode=hybrid"));
    }
    let engine = state.engine.read().expect("engine lock poisoned");
    match mode {
        SearchMode::Lexical => {
            let query = indexio_query::parse(&q).map_err(|e| bad_request(e.to_string()))?;
            let mut res = engine.search(&query, limit);
            res.hits.retain(|h| hit_allowed(&acl, &h.repo));
            Ok(Json(search_result_to_json(&res)))
        }
        SearchMode::Semantic => {
            let embedder = state.embedder.read().expect("embedder lock poisoned").clone();
            let mut hits = engine
                .search_semantic(&q, limit, embedder.as_ref())
                .map_err(|e| internal_error(format!("semantic search failed: {e}")))?;
            hits.retain(|h| hit_allowed(&acl, &h.repo));
            Ok(Json(
                json!({ "hits": hits.iter().map(hit_to_json).collect::<Vec<_>>() }),
            ))
        }
        SearchMode::Hybrid => {
            let embedder = state.embedder.read().expect("embedder lock poisoned").clone();
            let mut fused = if rerank {
                engine
                    .search_hybrid_fused_reranked(
                        &q,
                        limit,
                        embedder.as_ref(),
                        state.reranker.as_ref(),
                        fusion,
                    )
                    .map_err(|e| internal_error(format!("hybrid rerank failed: {e}")))?
            } else {
                engine
                    .search_hybrid_fused(&q, limit, embedder.as_ref(), fusion)
                    .map_err(|e| internal_error(format!("hybrid search failed: {e}")))?
            };
            fused.retain(|h| hit_allowed(&acl, &h.hit.repo));
            Ok(Json(
                json!({ "hits": fused.iter().map(hybrid_hit_to_json).collect::<Vec<_>>() }),
            ))
        }
    }
}

async fn symbol(
    State(state): State<AppState>,
    acl: Option<Extension<AclAllow>>,
    Path(name): Path<String>,
) -> Json<Value> {
    let engine = state.engine.read().expect("engine lock poisoned");
    let mut hits = engine.find_symbol(&name, DEFAULT_LIMIT);
    hits.retain(|h| hit_allowed(&acl, &h.repo));
    Json(Value::Array(hits.iter().map(hit_to_json).collect()))
}

async fn calls(
    State(state): State<AppState>,
    acl: Option<Extension<AclAllow>>,
    Path(name): Path<String>,
) -> Json<Value> {
    let engine = state.engine.read().expect("engine lock poisoned");
    let mut hits = engine.who_calls(&name, DEFAULT_LIMIT);
    hits.retain(|h| hit_allowed(&acl, &h.repo));
    Json(Value::Array(hits.iter().map(hit_to_json).collect()))
}

async fn reload(
    State(state): State<AppState>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let fresh = Engine::open(&state.data_dir)
        .map_err(|e| internal_error(format!("reopen failed: {e}")))?;
    let mut guard = state.engine.write().expect("engine lock poisoned");
    *guard = fresh;
    Ok(Json(json!({ "ok": true })))
}

/// Run embed_all over the current shard set (admin; may be slow on large
/// corpora — held under the engine read lock, shards are immutable).
async fn admin_embed(
    State(state): State<AppState>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let embedder = state.embedder.read().expect("embedder lock poisoned").clone();
    let reports = {
        let engine = state.engine.read().expect("engine lock poisoned");
        embed_all(engine.shard_set(), &state.data_dir, embedder.as_ref())
            .map_err(|e| internal_error(format!("embed failed: {e}")))?
    };
    Ok(Json(
        json!({ "reports": reports.iter().map(embed_report_to_json).collect::<Vec<_>>() }),
    ))
}

/// Embedding CAS stats for the active embedder's model.
async fn embcas_stats(State(state): State<AppState>) -> Json<Value> {
    let embedder = state.embedder.read().expect("embedder lock poisoned").clone();
    let (entries, bytes) = EmbedCas::open_for(&state.data_dir, embedder.as_ref()).stats();
    Json(json!({
        "model_id": embedder.model_id(),
        "entries": entries,
        "bytes": bytes,
    }))
}

// ---------------------------------------------------------------------------
// SPEC-P6 §3: impact / outline / span
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ImpactSymbolParams {
    /// Comma-separated symbol names.
    name: Option<String>,
    depth: Option<u32>,
    max_sites: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ImpactDiffBody {
    repo: String,
    diff: String,
    depth: Option<u32>,
    max_sites: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct FileParams {
    repo: Option<String>,
    path: Option<String>,
    start: Option<u32>,
    end: Option<u32>,
}

fn impact_options(depth: Option<u32>, max_sites: Option<usize>) -> indexio_query::ImpactOptions {
    let d = indexio_query::ImpactOptions::default();
    crate::impact_cli::options(
        depth.unwrap_or(d.depth),
        max_sites.unwrap_or(d.max_sites),
        d.max_fanout,
    )
}

/// ACL filtering of every repo-bearing list in a report.
fn filter_report(acl: &Option<Extension<AclAllow>>, r: &mut indexio_query::ImpactReport) {
    r.definitions.retain(|h| hit_allowed(acl, &h.repo));
    r.sites.retain(|s| hit_allowed(acl, &s.repo));
    r.files.retain(|f| hit_allowed(acl, &f.repo));
    r.importers.retain(|h| hit_allowed(acl, &h.repo));
}

async fn impact_symbol(
    State(state): State<AppState>,
    acl: Option<Extension<AclAllow>>,
    Query(params): Query<ImpactSymbolParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let names: Vec<String> = params
        .name
        .as_deref()
        .unwrap_or("")
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if names.is_empty() {
        return Err(bad_request("missing query parameter 'name'"));
    }
    let opts = impact_options(params.depth, params.max_sites);
    let engine = state.engine.read().expect("engine lock poisoned");
    let mut report = engine.impact_symbols(&names, &opts);
    filter_report(&acl, &mut report);
    Ok(Json(crate::impact_cli::impact_to_json(None, &report)))
}

/// POST /impact/diff: the client supplies the patch text. The post-change
/// side is read from the repo's working tree when it is registered on this
/// server; otherwise only the pre-change side (the index) is mapped.
async fn impact_diff(
    State(state): State<AppState>,
    acl: Option<Extension<AclAllow>>,
    Json(body): Json<ImpactDiffBody>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if body.repo.trim().is_empty() {
        return Err(bad_request("missing 'repo'"));
    }
    if !hit_allowed(&acl, &body.repo) {
        return Err((StatusCode::FORBIDDEN, Json(json!({ "error": "forbidden" }))));
    }
    let opts = impact_options(body.depth, body.max_sites);
    let repo_path = crate::impact_cli::resolve_repo_path(&state.data_dir, &body.repo).ok();
    let provider = crate::impact_cli::working_tree_provider(repo_path);
    let engine = state.engine.read().expect("engine lock poisoned");
    let (changed, mut report) = engine.impact_diff(&body.repo, &body.diff, &provider, &opts);
    filter_report(&acl, &mut report);
    Ok(Json(crate::impact_cli::impact_to_json(Some(&changed), &report)))
}

fn file_params(p: &FileParams) -> Result<(&str, &str), (StatusCode, Json<Value>)> {
    match (p.repo.as_deref(), p.path.as_deref()) {
        (Some(r), Some(f)) if !r.is_empty() && !f.is_empty() => Ok((r, f)),
        _ => Err(bad_request("missing query parameters 'repo' and 'path'")),
    }
}

async fn outline(
    State(state): State<AppState>,
    acl: Option<Extension<AclAllow>>,
    Query(params): Query<FileParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let (repo, path) = file_params(&params)?;
    if !hit_allowed(&acl, repo) {
        return Err((StatusCode::FORBIDDEN, Json(json!({ "error": "forbidden" }))));
    }
    let engine = state.engine.read().expect("engine lock poisoned");
    let items = engine
        .outline(repo, path)
        .ok_or_else(|| (StatusCode::NOT_FOUND, Json(json!({ "error": "not indexed" }))))?;
    Ok(Json(crate::impact_cli::outline_to_json(repo, path, &items)))
}

async fn span(
    State(state): State<AppState>,
    acl: Option<Extension<AclAllow>>,
    Query(params): Query<FileParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let (repo, path) = file_params(&params)?;
    if !hit_allowed(&acl, repo) {
        return Err((StatusCode::FORBIDDEN, Json(json!({ "error": "forbidden" }))));
    }
    let start = params.start.unwrap_or(1).max(1);
    let end = params.end.unwrap_or(start + 59);
    let engine = state.engine.read().expect("engine lock poisoned");
    let (text, last) = engine
        .read_span(repo, path, start, end)
        .ok_or_else(|| (StatusCode::NOT_FOUND, Json(json!({ "error": "not indexed" }))))?;
    Ok(Json(json!({
        "repo": repo, "path": path, "start": start, "end": last, "text": text,
    })))
}

/// Build the route table + auth middleware around a prepared AppState.
fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/search", get(search))
        .route("/symbol/{name}", get(symbol))
        .route("/calls/{name}", get(calls))
        .route("/impact/symbol", get(impact_symbol))
        .route("/impact/diff", post(impact_diff))
        .route("/outline", get(outline))
        .route("/span", get(span))
        .route("/admin/reload", post(reload))
        .route("/admin/embed", post(admin_embed))
        .route("/embcas/stats", get(embcas_stats))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn run_server(data_dir: PathBuf, bind: std::net::IpAddr, port: u16, auth: AuthConfig) -> anyhow::Result<()> {
    let engine = Engine::open(&data_dir)
        .with_context(|| format!("opening index at {}", data_dir.display()))?;
    // Embedder selection (SPEC-P4 §2): HTTP when INDEXIO_EMBED_BASE is set,
    // else the self-contained RandomIndexingEmbedder on <data_dir>/sem;
    // falls back to HashEmbedder (with a warning) if init fails.
    let embedder = crate::select_embedder_or_fallback(&data_dir);
    // Reranker selection (SPEC-P3 §2): HttpReranker::from_env() when
    // INDEXIO_RERANK_BASE is set, else the offline OverlapReranker.
    let reranker = crate::select_reranker_or_fallback();
    let state = AppState {
        engine: Arc::new(RwLock::new(engine)),
        embedder: Arc::new(RwLock::new(embedder)),
        reranker,
        auth: Arc::new(auth),
        data_dir: Arc::new(data_dir),
    };
    let app = build_router(state);
    let addr = std::net::SocketAddr::from((bind, port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("serving on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexio_types::Lang;

    fn sample_hit() -> SearchHit {
        SearchHit {
            repo: "alpha".into(),
            path: "src/main.rs".into(),
            line: 2,
            col: 4,
            snippet: "println!(\"hello world\");".into(),
            score: 3.5,
            lang: Lang::Rust,
        }
    }

    #[test]
    fn hit_to_json_shape() {
        let v = hit_to_json(&sample_hit());
        assert_eq!(v["repo"], "alpha");
        assert_eq!(v["path"], "src/main.rs");
        assert_eq!(v["line"], 2);
        assert_eq!(v["col"], 4);
        assert_eq!(v["score"], 3.5);
        assert_eq!(v["lang"], "Rust");
    }

    #[test]
    fn search_result_to_json_shape() {
        let res = SearchResult {
            hits: vec![sample_hit()],
            truncated: true,
            took_ms: 7,
        };
        let v = search_result_to_json(&res);
        assert_eq!(v["hits"].as_array().unwrap().len(), 1);
        assert_eq!(v["hits"][0]["path"], "src/main.rs");
        assert_eq!(v["truncated"], true);
        assert_eq!(v["took_ms"], 7);
    }

    #[test]
    fn search_result_to_json_empty() {
        let v = search_result_to_json(&SearchResult::default());
        assert_eq!(v["hits"].as_array().unwrap().len(), 0);
        assert_eq!(v["truncated"], false);
        assert_eq!(v["took_ms"], 0);
    }

    #[test]
    fn hybrid_hit_to_json_shape() {
        let h = indexio_query::HybridHit {
            hit: sample_hit(),
            rrf: 0.032,
            lex_rank: Some(1),
            bm25_rank: None,
            sem_rank: None,
            rerank_score: None,
        };
        let v = hybrid_hit_to_json(&h);
        assert_eq!(v["hit"]["path"], "src/main.rs");
        assert_eq!(v["rrf"], 0.032);
        assert_eq!(v["lex_rank"], 1);
        assert_eq!(v["bm25_rank"], Value::Null);
        assert_eq!(v["sem_rank"], Value::Null);
        assert_eq!(v["rerank_score"], Value::Null);
        // rerank_score surfaces when a reranker ran (SPEC-P3 §2).
        let h2 = indexio_query::HybridHit {
            rerank_score: Some(0.75),
            ..h
        };
        assert_eq!(hybrid_hit_to_json(&h2)["rerank_score"], 0.75);
    }

    #[test]
    fn embed_report_to_json_shape() {
        let r = EmbedReport {
            repo: "alpha".into(),
            chunks: 7,
            cas_hits: 3,
            cas_misses: 4,
            cas_known: 0,
            embedded: 4,
            elapsed_ms: 12,
            index_build_ms: 3,
            carried: 0,
        };
        let v = embed_report_to_json(&r);
        assert_eq!(v["index_build_ms"], 3);
        assert_eq!(v["repo"], "alpha");
        assert_eq!(v["chunks"], 7);
        assert_eq!(v["cas_hits"], 3);
        assert_eq!(v["cas_misses"], 4);
        assert_eq!(v["embedded"], 4);
        assert_eq!(v["elapsed_ms"], 12);
    }

    #[test]
    fn search_mode_param_parsing() {
        assert_eq!(SearchMode::parse("hybrid").unwrap(), SearchMode::Hybrid);
        assert!(SearchMode::parse("nope").is_err());
    }

    // ------------------------------------------------------------------
    // SPEC-P3 §3: auth + ACLs
    // ------------------------------------------------------------------

    #[test]
    fn repo_allowed_glob_cases() {
        let p = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        // `*` matches anything.
        assert!(repo_allowed(&p(&["*"]), "anything"));
        assert!(repo_allowed(&p(&["*"]), ""));
        // prefix glob
        assert!(repo_allowed(&p(&["team-*"]), "team-core"));
        assert!(repo_allowed(&p(&["team-*"]), "team-"));
        assert!(!repo_allowed(&p(&["team-*"]), "team"));
        assert!(!repo_allowed(&p(&["team-*"]), "x-team-core"));
        // suffix glob
        assert!(repo_allowed(&p(&["*-suffix"]), "core-suffix"));
        assert!(!repo_allowed(&p(&["*-suffix"]), "core-suffixx"));
        assert!(!repo_allowed(&p(&["*-suffix"]), "suffix"));
        // exact otherwise
        assert!(repo_allowed(&p(&["core"]), "core"));
        assert!(!repo_allowed(&p(&["core"]), "core-2"));
        assert!(!repo_allowed(&p(&["core*team"]), "coreXteam")); // inner * is literal
        // any pattern in the list matches
        assert!(repo_allowed(&p(&["nope", "team-*", "alsono"]), "team-x"));
        assert!(!repo_allowed(&p(&[]), "core"));
        // contains glob (defensive extension of the pattern language)
        assert!(repo_allowed(&p(&["*-mid-*"]), "a-mid-b"));
    }

    #[test]
    fn load_acl_file_parses_token_map() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("acl.json");
        std::fs::write(
            &path,
            r#"{"tokens":{"tokA":{"allow":["core","team-*"]},"tokB":{"allow":["*"]}}}"#,
        )
        .unwrap();
        let map = load_acl_file(&path).unwrap();
        assert_eq!(map["tokA"], vec!["core".to_string(), "team-*".to_string()]);
        assert_eq!(map["tokB"], vec!["*".to_string()]);
        // malformed files are errors, not panics
        std::fs::write(&path, r#"{"nope":{}}"#).unwrap();
        assert!(load_acl_file(&path).is_err());
        std::fs::write(&path, "not json").unwrap();
        assert!(load_acl_file(&path).is_err());
        assert!(load_acl_file(&tmp.path().join("missing.json")).is_err());
    }

    /// 2-repo fixture: alpha/src/a.rs and beta/src/b.rs both contain
    /// `sharedtoken`; returns the TempDir keeping the data dir alive.
    fn fixture_data_dir() -> tempfile::TempDir {
        use indexio_core::grams::{self, CommonGrams};
        use indexio_index::ShardWriter;
        use indexio_types::{BlobId, DocMeta, ExtractedArtifact};
        let tmp = tempfile::tempdir().unwrap();
        let shards = tmp.path().join("shards");
        std::fs::create_dir_all(&shards).unwrap();
        for (repo, path, content) in [
            ("alpha", "src/a.rs", "fn alpha_sharedtoken() {\n    // sharedtoken lives in alpha\n}\n"),
            ("beta", "src/b.rs", "fn beta_sharedtoken() {\n    // sharedtoken lives in beta\n}\n"),
        ] {
            let mut w = ShardWriter::new(&shards).unwrap();
            let art = ExtractedArtifact {
                ngrams: grams::extract(content.as_bytes(), &CommonGrams::empty()),
                symbols: vec![],
                calls: vec![],
                raw_len: content.len() as u32,
                lang: Lang::Rust,
            };
            let meta = DocMeta {
                blob: BlobId::from_content(content.as_bytes()),
                repo_id: 0,
                path: path.to_string(),
                lang: Lang::Rust,
                raw_len: content.len() as u32,
            };
            w.add_doc(&meta, content.as_bytes(), &art).unwrap();
            w.finish(&[repo.to_string()]).unwrap();
        }
        tmp
    }

    /// SPEC-P6 fixture: real tree-sitter extraction so call edges exist.
    /// core/src/config.rs defines parse_config + load (load calls
    /// parse_config); app/src/svc.rs calls load and imports config.
    fn fixture_impact_dir() -> tempfile::TempDir {
        use indexio_core::grams::{self, CommonGrams};
        use indexio_index::ShardWriter;
        use indexio_types::{BlobId, DocMeta, ExtractedArtifact};
        let tmp = tempfile::tempdir().unwrap();
        let shards = tmp.path().join("shards");
        std::fs::create_dir_all(&shards).unwrap();
        for (repo, path, content) in [
            (
                "core",
                "src/config.rs",
                "pub fn parse_config(s: &str) -> u32 {\n    s.len() as u32\n}\n\npub fn load(p: &str) -> u32 {\n    parse_config(p)\n}\n",
            ),
            (
                "app",
                "src/svc.rs",
                "use core_lib::config::load;\n\nfn boot() {\n    load(\"svc.toml\");\n}\n",
            ),
        ] {
            let mut w = ShardWriter::new(&shards).unwrap();
            let (symbols, calls) = indexio_symbols::extract(Lang::Rust, content.as_bytes());
            let art = ExtractedArtifact {
                ngrams: grams::extract(content.as_bytes(), &CommonGrams::empty()),
                symbols,
                calls,
                raw_len: content.len() as u32,
                lang: Lang::Rust,
            };
            let meta = DocMeta {
                blob: BlobId::from_content(content.as_bytes()),
                repo_id: 0,
                path: path.to_string(),
                lang: Lang::Rust,
                raw_len: content.len() as u32,
            };
            w.add_doc(&meta, content.as_bytes(), &art).unwrap();
            w.finish(&[repo.to_string()]).unwrap();
        }
        tmp
    }

    /// Spawn a test server on an ephemeral port with the given auth config.
    async fn start_server(auth: AuthConfig) -> (tempfile::TempDir, u16) {
        start_server_in(auth, fixture_data_dir()).await
    }

    async fn start_server_in(auth: AuthConfig, tmp: tempfile::TempDir) -> (tempfile::TempDir, u16) {
        let engine = Engine::open(tmp.path()).unwrap();
        let state = AppState {
            engine: Arc::new(RwLock::new(engine)),
            embedder: Arc::new(RwLock::new(
                Arc::new(indexio_embed::embed::HashEmbedder::new(512)) as Arc<dyn Embedder>
            )),
            reranker: Arc::new(indexio_embed::rerank::OverlapReranker),
            auth: Arc::new(auth),
            data_dir: Arc::new(tmp.path().to_path_buf()),
        };
        let app = build_router(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (tmp, port)
    }

    /// Minimal blocking HTTP/1.0-style request helper (no new deps).
    fn http_req(
        port: u16,
        method: &str,
        path: &str,
        bearer: Option<&str>,
    ) -> (u16, String) {
        http_req_body(port, method, path, bearer, None)
    }

    /// As `http_req`, with an optional JSON body.
    fn http_req_body(
        port: u16,
        method: &str,
        path: &str,
        bearer: Option<&str>,
        body: Option<&str>,
    ) -> (u16, String) {
        use std::io::{Read as _, Write as _};
        let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        let mut req =
            format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n");
        if let Some(t) = bearer {
            req.push_str(&format!("Authorization: Bearer {t}\r\n"));
        }
        let body = body.unwrap_or("");
        if !body.is_empty() {
            req.push_str("Content-Type: application/json\r\n");
        }
        req.push_str(&format!("Content-Length: {}\r\n\r\n{body}", body.len()));
        s.write_all(req.as_bytes()).unwrap();
        let mut buf = Vec::new();
        s.read_to_end(&mut buf).unwrap();
        let text = String::from_utf8_lossy(&buf);
        let status: u16 = text
            .split_whitespace()
            .nth(1)
            .and_then(|c| c.parse().ok())
            .unwrap_or(0);
        let body = text
            .split("\r\n\r\n")
            .nth(1)
            .unwrap_or("")
            .to_string();
        (status, body)
    }

    fn hit_repos(body: &str) -> Vec<String> {
        let v: Value = serde_json::from_str(body).unwrap();
        v["hits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| h["repo"].as_str().unwrap().to_string())
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn open_baseline_needs_no_token() {
        let (_tmp, port) = start_server(AuthConfig::Open).await;
        let (status, body) = http_req(port, "GET", "/health", None);
        assert_eq!(status, 200);
        assert!(body.contains("\"ok\":true"), "{body}");
        let (status, body) = http_req(port, "GET", "/search?q=sharedtoken", None);
        assert_eq!(status, 200, "{body}");
        let mut repos = hit_repos(&body);
        repos.sort();
        assert_eq!(repos, vec!["alpha".to_string(), "beta".to_string()]);
        // admin routes are open too (current behavior unchanged)
        let (status, _) = http_req(port, "POST", "/admin/reload", None);
        assert_eq!(status, 200);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bearer_token_401_and_200() {
        let (_tmp, port) =
            start_server(AuthConfig::Token("s3cret".to_string())).await;
        // /health stays open without a token.
        let (status, _) = http_req(port, "GET", "/health", None);
        assert_eq!(status, 200);
        // missing token -> 401 JSON
        let (status, body) = http_req(port, "GET", "/search?q=sharedtoken", None);
        assert_eq!(status, 401);
        assert!(body.contains("\"unauthorized\""), "{body}");
        // wrong token -> 401
        let (status, _) = http_req(port, "GET", "/search?q=sharedtoken", Some("nope"));
        assert_eq!(status, 401);
        // right token -> 200 (also on admin routes in token mode)
        let (status, body) = http_req(port, "GET", "/search?q=sharedtoken", Some("s3cret"));
        assert_eq!(status, 200, "{body}");
        assert_eq!(hit_repos(&body).len(), 2);
        let (status, _) = http_req(port, "POST", "/admin/reload", Some("s3cret"));
        assert_eq!(status, 200);
        let (status, _) = http_req(port, "POST", "/admin/reload", Some("nope"));
        assert_eq!(status, 401);
    }

    fn test_acl() -> AuthConfig {
        let mut map = HashMap::new();
        map.insert("tokA".to_string(), vec!["alpha".to_string()]);
        map.insert("tokB".to_string(), vec!["*".to_string()]);
        AuthConfig::Acl(Arc::new(map))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn acl_filters_search_hits_by_repo() {
        let (_tmp, port) = start_server(test_acl()).await;
        // unknown token -> 401
        let (status, _) = http_req(port, "GET", "/search?q=sharedtoken", Some("ghost"));
        assert_eq!(status, 401);
        // tokA only sees repo alpha
        let (status, body) = http_req(port, "GET", "/search?q=sharedtoken", Some("tokA"));
        assert_eq!(status, 200, "{body}");
        assert_eq!(hit_repos(&body), vec!["alpha".to_string()]);
        // tokB (allow ["*"]) sees both repos
        let (status, body) = http_req(port, "GET", "/search?q=sharedtoken", Some("tokB"));
        assert_eq!(status, 200);
        let mut repos = hit_repos(&body);
        repos.sort();
        assert_eq!(repos, vec!["alpha".to_string(), "beta".to_string()]);
        // missing token -> 401
        let (status, _) = http_req(port, "GET", "/search?q=sharedtoken", None);
        assert_eq!(status, 401);
        // /health still open
        let (status, _) = http_req(port, "GET", "/health", None);
        assert_eq!(status, 200);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn acl_admin_routes_require_star_allow() {
        let (_tmp, port) = start_server(test_acl()).await;
        // tokA lacks an exact `*` allow -> 403 forbidden JSON
        let (status, body) = http_req(port, "POST", "/admin/reload", Some("tokA"));
        assert_eq!(status, 403);
        assert!(body.contains("\"forbidden\""), "{body}");
        // tokB has `*` -> 200
        let (status, _) = http_req(port, "POST", "/admin/reload", Some("tokB"));
        assert_eq!(status, 200);
        // unknown token -> 401 (auth before authorization)
        let (status, _) = http_req(port, "POST", "/admin/reload", Some("ghost"));
        assert_eq!(status, 401);
    }

    // ------------------------------------------------------------------
    // SPEC-P6 §3: impact / outline / span routes
    // ------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn impact_routes_open() {
        let (_tmp, port) = start_server_in(AuthConfig::Open, fixture_impact_dir()).await;
        // GET /impact/symbol
        let (status, body) = http_req(port, "GET", "/impact/symbol?name=parse_config&depth=2", None);
        assert_eq!(status, 200, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["roots"], json!(["parse_config"]));
        assert_eq!(v["definitions"][0]["path"], "src/config.rs");
        let sites = v["sites"].as_array().unwrap();
        assert!(sites.iter().any(|s| s["path"] == "src/config.rs" && s["depth"] == 1), "{body}");
        assert!(sites.iter().any(|s| s["repo"] == "app" && s["path"] == "src/svc.rs" && s["depth"] == 2), "{body}");
        let (status, _) = http_req(port, "GET", "/impact/symbol", None);
        assert_eq!(status, 400);
        // POST /impact/diff
        let diff = "--- a/src/config.rs\n+++ b/src/config.rs\n@@ -2,1 +2,1 @@\n-    s.len() as u32\n+    s.len() as u32 + 1\n";
        let payload = json!({ "repo": "core", "diff": diff, "depth": 2 }).to_string();
        let (status, body) = http_req_body(port, "POST", "/impact/diff", None, Some(&payload));
        assert_eq!(status, 200, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["changed"][0]["name"], "parse_config");
        assert!(v["sites"].as_array().unwrap().iter().all(|s| s["path"] != "src/config.rs"), "{body}");
        assert!(v["sites"].as_array().unwrap().iter().any(|s| s["repo"] == "app"), "{body}");
        assert_eq!(v["importers"][0]["repo"], "app", "{body}");
        // GET /outline + /span
        let (status, body) = http_req(port, "GET", "/outline?repo=core&path=src/config.rs", None);
        assert_eq!(status, 200, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["items"][0]["name"], "parse_config");
        assert_eq!(v["items"][0]["end_line"], 3);
        let (status, _) = http_req(port, "GET", "/outline?repo=core&path=nope.rs", None);
        assert_eq!(status, 404);
        let (status, body) = http_req(port, "GET", "/span?repo=core&path=src/config.rs&start=5&end=7", None);
        assert_eq!(status, 200, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["text"], "pub fn load(p: &str) -> u32 {\n    parse_config(p)\n}\n");
        assert_eq!(v["end"], 7);
        let (status, _) = http_req(port, "GET", "/span?repo=core", None);
        assert_eq!(status, 400);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn impact_routes_respect_acl() {
        let mut map = HashMap::new();
        map.insert("tokCore".to_string(), vec!["core".to_string()]);
        map.insert("tokAll".to_string(), vec!["*".to_string()]);
        let (_tmp, port) =
            start_server_in(AuthConfig::Acl(Arc::new(map)), fixture_impact_dir()).await;
        // tokCore never sees repo "app" in any list
        let (status, body) = http_req(port, "GET", "/impact/symbol?name=parse_config", Some("tokCore"));
        assert_eq!(status, 200, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        for key in ["definitions", "sites", "files", "importers"] {
            assert!(v[key].as_array().unwrap().iter().all(|x| x["repo"] == "core"), "{key}: {body}");
        }
        let (_, body) = http_req(port, "GET", "/impact/symbol?name=parse_config", Some("tokAll"));
        let v: Value = serde_json::from_str(&body).unwrap();
        assert!(v["sites"].as_array().unwrap().iter().any(|s| s["repo"] == "app"), "{body}");
        // outline/span of a repo outside the allowlist -> 403; diff for it -> 403
        let (status, _) = http_req(port, "GET", "/outline?repo=app&path=src/svc.rs", Some("tokCore"));
        assert_eq!(status, 403);
        let (status, _) = http_req(port, "GET", "/span?repo=app&path=src/svc.rs&start=1", Some("tokCore"));
        assert_eq!(status, 403);
        let payload = json!({ "repo": "app", "diff": "" }).to_string();
        let (status, _) = http_req_body(port, "POST", "/impact/diff", Some("tokCore"), Some(&payload));
        assert_eq!(status, 403);
        // no token -> 401
        let (status, _) = http_req(port, "GET", "/outline?repo=core&path=src/config.rs", None);
        assert_eq!(status, 401);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rerank_param_requires_hybrid_mode() {
        let (_tmp, port) = start_server(AuthConfig::Open).await;
        // rerank=1 with lexical -> 400
        let (status, body) = http_req(port, "GET", "/search?q=sharedtoken&rerank=1", None);
        assert_eq!(status, 400, "{body}");
        // invalid rerank value -> 400
        let (status, _) = http_req(port, "GET", "/search?q=sharedtoken&mode=hybrid&rerank=yes", None);
        assert_eq!(status, 400);
        // hybrid + rerank=1 -> 200, hits carry rerank_score
        let (status, body) =
            http_req(port, "GET", "/search?q=sharedtoken&mode=hybrid&rerank=1", None);
        assert_eq!(status, 200, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        let hits = v["hits"].as_array().unwrap();
        assert!(!hits.is_empty(), "{body}");
        for h in hits {
            assert!(h["rerank_score"].is_f64(), "{h}");
        }
        // plain hybrid keeps rerank_score null
        let (status, body) = http_req(port, "GET", "/search?q=sharedtoken&mode=hybrid", None);
        assert_eq!(status, 200);
        let v: Value = serde_json::from_str(&body).unwrap();
        for h in v["hits"].as_array().unwrap() {
            assert_eq!(h["rerank_score"], Value::Null);
        }
    }
}
