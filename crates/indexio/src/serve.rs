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
//!   POST /team/worktree          -> apply the caller's worktree patch as their overlay
//!                                   (headers X-Indexio-Repo, X-Indexio-Base; body =
//!                                   `git diff --binary <base>`; empty body clears it)
//!   GET  /team/overlays          -> the caller's overlays
//!   POST /mcp                    -> MCP over streamable HTTP (JSON responses)
//!
//! Team overlays: a request made as a user who has overlays is answered by
//! an engine that layers them over the shared index (their changed files
//! shadow the base, their deleted files are hidden). The user is the ACL
//! entry's `"user"` in ACL mode, else the `X-Indexio-User` header.
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};
use std::time::{Duration, Instant};

use anyhow::Context;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, Query, Request, State};
use axum::http::{HeaderMap, StatusCode};
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
    /// OIDC SSO: JWT bearer tokens validated against the identity
    /// provider's JWKS; the ACL file's "users" section maps identities to
    /// allow patterns (SPEC-P12).
    Oidc(Arc<OidcAuth>),
}

/// An OIDC-authenticated identity from the ACL file's "users" section:
/// `{"alice@corp": {"allow": ["team-*"], "user": "alice"}}`. The optional
/// "user" overrides the team-overlay name (otherwise the identity itself is
/// used, so emails work as overlay keys).
#[derive(Clone)]
pub struct OidcUser {
    pub allow: Vec<String>,
    pub user: Option<String>,
}

/// Identity -> allow patterns for OIDC mode (SPEC-P12 §2).
pub fn load_acl_oidc_users(
    path: &std::path::Path,
) -> anyhow::Result<HashMap<String, OidcUser>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading ACL file {}", path.display()))?;
    let v: Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing ACL file {} as JSON", path.display()))?;
    let users = v
        .get("users")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("ACL file missing top-level \"users\" object"))?;
    let mut map = HashMap::with_capacity(users.len());
    for (ident, spec) in users {
        let allow = spec
            .get("allow")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("ACL user \"{ident}\" missing \"allow\" array"))?;
        let mut pats = Vec::with_capacity(allow.len());
        for p in allow {
            pats.push(
                p.as_str()
                    .ok_or_else(|| anyhow::anyhow!("ACL user \"{ident}\" allow entries must be strings"))?
                    .to_string(),
            );
        }
        let user = spec
            .get("user")
            .and_then(Value::as_str)
            .map(str::to_string);
        let name = user.as_deref().unwrap_or(ident);
        anyhow::ensure!(
            indexio_ingest::team::valid_user(name),
            "ACL user \"{ident}\" maps to an invalid user name \"{name}\""
        );
        map.insert(ident.clone(), OidcUser { allow: pats, user });
    }
    Ok(map)
}

/// OIDC bearer-token validation (SPEC-P12 §2): discovery at
/// `<issuer>/.well-known/openid-configuration`, JWKS cached and refetched on
/// `kid` miss, RS256 only, issuer/audience/expiry enforced, identity must be
/// in the users map (fail closed). Plain blocking HTTP — call through
/// `tokio::task::spawn_blocking`.
pub struct OidcAuth {
    issuer: String,
    audience: String,
    jwks_uri: String,
    /// Cached RSA signing keys, refetched when an unknown `kid` arrives.
    keys: std::sync::Mutex<Vec<OidcJwk>>,
    /// Identity (email, else `sub`) -> allow patterns + overlay-name override.
    users: HashMap<String, OidcUser>,
}

struct OidcJwk {
    kid: String,
    n: String,
    e: String,
}

#[derive(serde::Deserialize)]
struct OidcClaims {
    email: Option<String>,
    sub: Option<String>,
}

impl OidcAuth {
    /// Discover the provider and prefill the key cache. A provider that is
    /// briefly unreachable at startup is not fatal: the cache stays empty
    /// and the first `kid` miss retries the fetch.
    pub fn discover(
        issuer: &str,
        audience: &str,
        users: HashMap<String, OidcUser>,
    ) -> anyhow::Result<Arc<Self>> {
        anyhow::ensure!(!users.is_empty(), "OIDC mode needs at least one entry in the ACL file's \"users\" section");
        let issuer = issuer.trim_end_matches('/').to_string();
        anyhow::ensure!(issuer.starts_with("http"), "--oidc-issuer must be a URL");
        let doc: Value = ureq::get(&format!("{issuer}/.well-known/openid-configuration"))
            .timeout(std::time::Duration::from_secs(5))
            .call()?
            .into_json()?;
        let jwks_uri = doc
            .get("jwks_uri")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("OIDC discovery document has no jwks_uri"))?
            .to_string();
        let auth = Arc::new(Self {
            issuer,
            audience: audience.to_string(),
            jwks_uri,
            keys: std::sync::Mutex::new(Vec::new()),
            users,
        });
        if let Err(e) = auth.refresh_keys() {
            tracing::warn!("OIDC JWKS prefetch failed (will retry on demand): {e:#}");
        }
        Ok(auth)
    }

    fn refresh_keys(&self) -> anyhow::Result<()> {
        let v: Value = ureq::get(&self.jwks_uri)
            .timeout(std::time::Duration::from_secs(5))
            .call()?
            .into_json()?;
        let mut keys = Vec::new();
        for k in v
            .get("keys")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("JWKS response has no \"keys\" array"))?
        {
            if k.get("kty").and_then(Value::as_str) != Some("RSA") {
                continue;
            }
            let (Some(kid), Some(n), Some(e)) = (
                k.get("kid").and_then(Value::as_str),
                k.get("n").and_then(Value::as_str),
                k.get("e").and_then(Value::as_str),
            ) else {
                continue;
            };
            keys.push(OidcJwk {
                kid: kid.to_string(),
                n: n.to_string(),
                e: e.to_string(),
            });
        }
        anyhow::ensure!(!keys.is_empty(), "JWKS contained no usable RSA keys");
        *self.keys.lock().expect("jwks mutex") = keys;
        Ok(())
    }

    fn decoding_key(&self, kid: &str) -> Option<jsonwebtoken::DecodingKey> {
        let found = self
            .keys
            .lock()
            .expect("jwks mutex")
            .iter()
            .find(|k| k.kid == kid)
            .map(|k| jsonwebtoken::DecodingKey::from_rsa_components(&k.n, &k.e))
            .and_then(|r| r.ok());
        if found.is_none() {
            // key rotation: refetch once, then retry the lookup
            if self.refresh_keys().is_ok() {
                return self
                    .keys
                    .lock()
                    .expect("jwks mutex")
                    .iter()
                    .find(|k| k.kid == kid)
                    .map(|k| jsonwebtoken::DecodingKey::from_rsa_components(&k.n, &k.e))
                    .and_then(|r| r.ok());
            }
        }
        found
    }

    /// Validate a bearer token; returns the team-overlay user name and the
    /// allow patterns, or `None` for every failure mode (fail closed).
    pub fn validate(&self, token: &str) -> Option<(String, Arc<Vec<String>>)> {
        let header = jsonwebtoken::decode_header(token).ok()?;
        if header.alg != jsonwebtoken::Algorithm::RS256 {
            tracing::warn!(alg = ?header.alg, "OIDC token with disallowed algorithm");
            return None;
        }
        let kid = header.kid?;
        let key = self.decoding_key(&kid)?;
        let mut validation = jsonwebtoken::Validation::new(jsonwebtoken::Algorithm::RS256);
        validation.set_issuer(std::slice::from_ref(&self.issuer));
        validation.set_audience(std::slice::from_ref(&self.audience));
        let claims = jsonwebtoken::decode::<OidcClaims>(token, &key, &validation).ok()?;
        let identity = claims.claims.email.or(claims.claims.sub)?;
        let user = self.users.get(&identity)?;
        let name = user.user.clone().unwrap_or_else(|| identity.clone());
        Some((name, Arc::new(user.allow.clone())))
    }
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

/// Token -> user name map of an ACL file: entries with a `"user"` field
/// (team overlays are keyed by it; a token without one has no overlay).
pub fn load_acl_users(path: &std::path::Path) -> anyhow::Result<HashMap<String, String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading ACL file {}", path.display()))?;
    let v: Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing ACL file {} as JSON", path.display()))?;
    let mut map = HashMap::new();
    if let Some(tokens) = v.get("tokens").and_then(Value::as_object) {
        for (tok, spec) in tokens {
            if let Some(u) = spec.get("user").and_then(Value::as_str) {
                anyhow::ensure!(
                    indexio_ingest::team::valid_user(u),
                    "ACL token \"{tok}\" has an invalid user name \"{u}\""
                );
                map.insert(tok.clone(), u.to_string());
            }
        }
    }
    Ok(map)
}

/// The user a request acts as (team overlays), attached by the auth
/// middleware.
#[derive(Clone)]
pub struct TeamUser(pub String);

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
    // Identity for team overlays: an ACL token's mapped user; with a shared
    // token (or none) the client names itself.
    let header_user = req
        .headers()
        .get("x-indexio-user")
        .and_then(|v| v.to_str().ok())
        .filter(|u| indexio_ingest::team::valid_user(u))
        .map(str::to_string);
    if matches!(auth.as_ref(), AuthConfig::Open) {
        if let Some(u) = header_user {
            req.extensions_mut().insert(TeamUser(u));
        }
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
                if let Some(u) = header_user {
                    req.extensions_mut().insert(TeamUser(u));
                }
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
            if let Some(u) = state.users.get(&tok) {
                req.extensions_mut().insert(TeamUser(u.clone()));
            }
            next.run(req).await
        }
        AuthConfig::Oidc(oidc) => {
            // JWT verification and the JWKS refetch are blocking HTTP/crypto.
            let oidc = oidc.clone();
            let outcome = tokio::task::spawn_blocking(move || oidc.validate(&tok)).await;
            let Some((user, allow)) = outcome.ok().flatten() else {
                return unauthorized();
            };
            if req.uri().path().starts_with("/admin/") && !allow.iter().any(|p| p == "*") {
                return forbidden();
            }
            req.extensions_mut().insert(AclAllow(allow));
            req.extensions_mut().insert(TeamUser(user));
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
    /// ACL token -> team user (ACL entries with a `"user"`).
    users: Arc<HashMap<String, String>>,
    /// Per-(user, ACL) engines layering overlays over the shared index.
    team: Arc<TeamEngines>,
}

// ---------------------------------------------------------------------------
// Team overlays: per-user engines
// ---------------------------------------------------------------------------

/// Most per-user engines kept open (each maps the base shards once more).
const TEAM_ENGINE_CAP: usize = 64;

#[derive(Default)]
struct TeamEngines {
    /// Bumped by /admin/reload: every cached view is rebuilt on next use.
    base_gen: AtomicU64,
    map: Mutex<HashMap<String, TeamEntry>>,
    /// Origin URL -> repo name, with the time it was read.
    remotes: Mutex<Option<(Instant, HashMap<String, String>)>>,
}

struct TeamEntry {
    stamp: u64,
    base_gen: u64,
    used: Instant,
    engine: Arc<Engine>,
}

/// The engine a request reads: the shared one, or the caller's layered view.
enum View<'a> {
    Base(RwLockReadGuard<'a, Engine>),
    Layered(Arc<Engine>),
}

impl std::ops::Deref for View<'_> {
    type Target = Engine;
    fn deref(&self) -> &Engine {
        match self {
            View::Base(g) => g,
            View::Layered(e) => e,
        }
    }
}

/// The engine for `user` with `allow` enforced at the doc level (`None` =
/// every repo; the REST routes filter hits instead, MCP output is rendered
/// text and cannot be filtered after the fact). The shared engine when
/// neither applies.
fn view<'a>(state: &'a AppState, user: Option<&str>, allow: Option<&[String]>) -> Result<View<'a>, (StatusCode, Json<Value>)> {
    let allow = allow.filter(|a| !a.iter().any(|p| p == "*"));
    let stamp = user.map_or(0, |u| indexio_ingest::team::user_stamp(&state.data_dir, u));
    if stamp == 0 && allow.is_none() {
        return Ok(View::Base(state.engine.read().expect("engine lock poisoned")));
    }
    let key = format!("{}\0{}", user.unwrap_or(""), allow.map(|a| a.join(",")).unwrap_or_default());
    let gen = state.team.base_gen.load(Ordering::Acquire);
    {
        let mut map = state.team.map.lock().expect("team map poisoned");
        if let Some(e) = map.get_mut(&key) {
            if e.stamp == stamp && e.base_gen == gen {
                e.used = Instant::now();
                return Ok(View::Layered(Arc::clone(&e.engine)));
            }
        }
    }
    let (upper, hidden) = match user.filter(|_| stamp != 0) {
        Some(u) => {
            let layer = indexio_ingest::team::user_layer(&state.data_dir, u);
            (layer.shards_dir.into_iter().collect::<Vec<_>>(), layer.hidden)
        }
        None => (Vec::new(), Default::default()),
    };
    let allow_fn: Option<Arc<dyn Fn(&str) -> bool + Send + Sync>> = allow.map(|a| {
        let a = a.to_vec();
        Arc::new(move |r: &str| repo_allowed(&a, r)) as Arc<dyn Fn(&str) -> bool + Send + Sync>
    });
    let engine = Engine::open_layered(&state.data_dir, &upper, indexio_index::Visibility { hidden, allow: allow_fn })
        .map_err(|e| internal_error(format!("opening overlay view failed: {e}")))?;
    engine.inherit_sidecars(&state.engine.read().expect("engine lock poisoned"));
    let engine = Arc::new(engine);
    let mut map = state.team.map.lock().expect("team map poisoned");
    if map.len() >= TEAM_ENGINE_CAP && !map.contains_key(&key) {
        if let Some(oldest) = map.iter().min_by_key(|(_, e)| e.used).map(|(k, _)| k.clone()) {
            map.remove(&oldest);
        }
    }
    map.insert(key, TeamEntry { stamp, base_gen: gen, used: Instant::now(), engine: Arc::clone(&engine) });
    Ok(View::Layered(engine))
}

fn user_of(u: &Option<Extension<TeamUser>>) -> Option<&str> {
    u.as_ref().map(|Extension(TeamUser(n))| n.as_str())
}

/// Resolve a client's repo hint (name or origin URL); the origin map is
/// re-read at most every 30 s, and only when the hint is not found.
fn resolve_team_repo(state: &AppState, hint: &str) -> Option<String> {
    let dd = state.data_dir.as_path();
    let mut cache = state.team.remotes.lock().expect("remotes poisoned");
    if let Some((_, m)) = cache.as_ref() {
        if let Some(r) = indexio_ingest::team::resolve_repo(dd, m, hint) {
            return Some(r);
        }
    }
    if cache.as_ref().is_some_and(|(t, _)| t.elapsed() < Duration::from_secs(30)) {
        return None;
    }
    let m = indexio_ingest::team::remote_map(dd);
    let r = indexio_ingest::team::resolve_repo(dd, &m, hint);
    *cache = Some((Instant::now(), m));
    r
}

fn header<'h>(h: &'h HeaderMap, name: &str) -> Option<&'h str> {
    h.get(name).and_then(|v| v.to_str().ok()).map(str::trim).filter(|s| !s.is_empty())
}

/// POST /team/worktree: the caller's worktree patch becomes their overlay
/// of the repo (replacing the previous one; an empty body clears it).
async fn team_worktree(
    State(state): State<AppState>,
    user: Option<Extension<TeamUser>>,
    acl: Option<Extension<AclAllow>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let Some(user) = user_of(&user).map(str::to_string) else {
        return Err(bad_request("no user: map this token to a \"user\" in the ACL file, or send X-Indexio-User"));
    };
    let hint = header(&headers, "x-indexio-repo").ok_or_else(|| bad_request("missing X-Indexio-Repo"))?;
    let base = header(&headers, "x-indexio-base").ok_or_else(|| bad_request("missing X-Indexio-Base"))?.to_string();
    let Some(repo) = resolve_team_repo(&state, hint) else {
        return Err((StatusCode::NOT_FOUND, Json(json!({ "error": format!("no registered repo matches '{hint}'") }))));
    };
    if !hit_allowed(&acl, &repo) {
        return Err(forbidden_err());
    }
    let dd = Arc::clone(&state.data_dir);
    let res = tokio::task::spawn_blocking(move || {
        let cas = indexio_ingest::Cas::open(&dd.join("cas"))?;
        indexio_ingest::team::apply_worktree_patch(&dd, &cas, &repo, &user, &base, &body)
    })
    .await
    .map_err(|e| internal_error(format!("overlay task failed: {e}")))?;
    match res {
        Ok(r) => Ok(Json(serde_json::to_value(r).expect("OverlayReport is Serialize"))),
        Err(e) => {
            use indexio_ingest::team::OverlayError as E;
            let code = match e.downcast_ref::<E>() {
                Some(E::UnknownRepo(_)) => StatusCode::NOT_FOUND,
                Some(E::UnknownBase(_)) => StatusCode::CONFLICT,
                Some(E::Invalid(_)) => StatusCode::BAD_REQUEST,
                None => StatusCode::INTERNAL_SERVER_ERROR,
            };
            Err((code, Json(json!({ "error": format!("{e:#}") }))))
        }
    }
}

/// GET /team/overlays: the caller's overlays (repo, base, changed and
/// hidden paths, last update).
async fn team_overlays(
    State(state): State<AppState>,
    user: Option<Extension<TeamUser>>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let Some(user) = user_of(&user) else {
        return Err(bad_request("no user for this request"));
    };
    let list = indexio_ingest::team::user_overlays(&state.data_dir, user);
    Ok(Json(json!({ "user": user, "overlays": list })))
}

/// POST /mcp: MCP over streamable HTTP, answered with plain JSON (no SSE).
/// Every tool runs against the caller's view: their overlay on top, their
/// ACL applied to the docs themselves.
async fn mcp_post(
    State(state): State<AppState>,
    user: Option<Extension<TeamUser>>,
    acl: Option<Extension<AclAllow>>,
    body: Bytes,
) -> Response {
    let msg: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "jsonrpc": "2.0", "id": null, "error": { "code": -32700, "message": format!("parse error: {e}") } })),
            )
                .into_response()
        }
    };
    let allow = acl.as_ref().map(|Extension(a)| a.0.as_slice());
    let engine = match view(&state, user_of(&user), allow) {
        Ok(v) => v,
        Err(e) => return e.into_response(),
    };
    let embedder = state.embedder.read().expect("embedder lock poisoned").clone();
    let one = |m: &Value| crate::mcp::handle_remote(&engine, embedder.as_ref(), state.reranker.as_ref(), m);
    let reply = match &msg {
        Value::Array(batch) => {
            let out: Vec<Value> = batch.iter().filter_map(one).collect();
            (!out.is_empty()).then_some(Value::Array(out))
        }
        m => one(m),
    };
    match reply {
        Some(v) => Json(v).into_response(),
        None => StatusCode::ACCEPTED.into_response(),
    }
}

/// GET/DELETE /mcp: no server-initiated stream, no sessions.
async fn mcp_not_allowed() -> Response {
    (StatusCode::METHOD_NOT_ALLOWED, [(axum::http::header::ALLOW, "POST")]).into_response()
}

fn forbidden_err() -> (StatusCode, Json<Value>) {
    (StatusCode::FORBIDDEN, Json(json!({ "error": "forbidden" })))
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
    user: Option<Extension<TeamUser>>,
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
    let engine = view(&state, user_of(&user), None)?;
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
    user: Option<Extension<TeamUser>>,
    Path(name): Path<String>,
) -> Json<Value> {
    let Ok(engine) = view(&state, user_of(&user), None) else {
        return Json(Value::Array(Vec::new()));
    };
    let mut hits = engine.find_symbol(&name, DEFAULT_LIMIT);
    hits.retain(|h| hit_allowed(&acl, &h.repo));
    Json(Value::Array(hits.iter().map(hit_to_json).collect()))
}

async fn calls(
    State(state): State<AppState>,
    acl: Option<Extension<AclAllow>>,
    user: Option<Extension<TeamUser>>,
    Path(name): Path<String>,
) -> Json<Value> {
    let Ok(engine) = view(&state, user_of(&user), None) else {
        return Json(Value::Array(Vec::new()));
    };
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
    drop(guard);
    state.team.base_gen.fetch_add(1, Ordering::AcqRel);
    state.team.map.lock().expect("team map poisoned").clear();
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
    user: Option<Extension<TeamUser>>,
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
    let engine = view(&state, user_of(&user), None)?;
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
    user: Option<Extension<TeamUser>>,
    Query(params): Query<FileParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let (repo, path) = file_params(&params)?;
    if !hit_allowed(&acl, repo) {
        return Err((StatusCode::FORBIDDEN, Json(json!({ "error": "forbidden" }))));
    }
    let engine = view(&state, user_of(&user), None)?;
    let items = engine
        .outline(repo, path)
        .ok_or_else(|| (StatusCode::NOT_FOUND, Json(json!({ "error": "not indexed" }))))?;
    Ok(Json(crate::impact_cli::outline_to_json(repo, path, &items)))
}

async fn span(
    State(state): State<AppState>,
    acl: Option<Extension<AclAllow>>,
    user: Option<Extension<TeamUser>>,
    Query(params): Query<FileParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let (repo, path) = file_params(&params)?;
    if !hit_allowed(&acl, repo) {
        return Err((StatusCode::FORBIDDEN, Json(json!({ "error": "forbidden" }))));
    }
    let start = params.start.unwrap_or(1).max(1);
    let end = params.end.unwrap_or(start + 59);
    let engine = view(&state, user_of(&user), None)?;
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
        .route(
            "/team/worktree",
            post(team_worktree).layer(DefaultBodyLimit::max(indexio_ingest::team::MAX_PATCH_BYTES)),
        )
        .route("/team/overlays", get(team_overlays))
        .route("/mcp", post(mcp_post).get(mcp_not_allowed).delete(mcp_not_allowed))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Days an overlay nobody updated is kept (`INDEXIO_TEAM_TTL_DAYS`).
fn team_ttl() -> Duration {
    let days = std::env::var("INDEXIO_TEAM_TTL_DAYS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(7);
    Duration::from_secs(days.max(1) * 86_400)
}

pub async fn run_server(
    data_dir: PathBuf,
    bind: std::net::IpAddr,
    port: u16,
    auth: AuthConfig,
    users: HashMap<String, String>,
) -> anyhow::Result<()> {
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
        users: Arc::new(users),
        team: Arc::new(TeamEngines::default()),
    };
    // stale team overlays go after INDEXIO_TEAM_TTL_DAYS without an update
    let dd = Arc::clone(&state.data_dir);
    tokio::spawn(async move {
        let ttl = team_ttl();
        loop {
            let d = Arc::clone(&dd);
            let _ = tokio::task::spawn_blocking(move || indexio_ingest::team::prune_overlays(&d, ttl)).await;
            tokio::time::sleep(Duration::from_secs(600)).await;
        }
    });
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
        start_server_users(auth, HashMap::new(), tmp).await
    }

    async fn start_server_users(
        auth: AuthConfig,
        users: HashMap<String, String>,
        tmp: tempfile::TempDir,
    ) -> (tempfile::TempDir, u16) {
        let engine = Engine::open(tmp.path()).unwrap();
        let state = AppState {
            engine: Arc::new(RwLock::new(engine)),
            embedder: Arc::new(RwLock::new(
                Arc::new(indexio_embed::embed::HashEmbedder::new(512)) as Arc<dyn Embedder>
            )),
            reranker: Arc::new(indexio_embed::rerank::OverlapReranker),
            auth: Arc::new(auth),
            data_dir: Arc::new(tmp.path().to_path_buf()),
            users: Arc::new(users),
            team: Arc::new(TeamEngines::default()),
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

    // -----------------------------------------------------------------------
    // team overlays + remote MCP
    // -----------------------------------------------------------------------

    fn git(dir: &std::path::Path, args: &[&str]) -> String {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("HOME", dir)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// A data dir (the returned TempDir) with git repo `app` (origin
    /// git@example.com:acme/app.git) and plain folder `secret`, plus a
    /// developer clone of `app` and its base commit.
    fn team_fixture() -> (tempfile::TempDir, tempfile::TempDir, std::path::PathBuf, String) {
        let src_td = tempfile::tempdir().unwrap();
        let src = src_td.path().join("app");
        std::fs::create_dir_all(src.join("src")).unwrap();
        git(&src, &["init", "-q"]);
        for (k, v) in [("user.email", "t@example.com"), ("user.name", "t"), ("commit.gpgsign", "false")] {
            git(&src, &["config", k, v]);
        }
        std::fs::write(src.join("src/lib.rs"), "pub fn shared_entry() { helper_v1(); }\nfn helper_v1() {}\n").unwrap();
        std::fs::write(src.join("src/old.rs"), "pub fn retired_function() {}\n").unwrap();
        git(&src, &["add", "-A"]);
        git(&src, &["commit", "-q", "-m", "init"]);
        let base = git(&src, &["rev-parse", "HEAD"]);
        let dev = src_td.path().join("dev");
        git(src_td.path(), &["clone", "-q", src.to_str().unwrap(), dev.to_str().unwrap()]);
        git(&src, &["remote", "add", "origin", "git@example.com:acme/app.git"]);
        // the server's on-demand fetch reaches the repo itself, not the network
        let local = format!("url.{}.insteadOf", src.display());
        git(&src, &["config", &local, "git@example.com:acme/app.git"]);
        let secret = src_td.path().join("secret");
        std::fs::create_dir_all(&secret).unwrap();
        std::fs::write(secret.join("keys.py"), "def classified_routine():\n    pass\n").unwrap();
        let data = tempfile::tempdir().unwrap();
        let cas = indexio_ingest::Cas::open(&data.path().join("cas")).unwrap();
        indexio_ingest::index_repo(&src, "app", data.path(), &cas).unwrap();
        indexio_ingest::index_repo(&secret, "secret", data.path(), &cas).unwrap();
        (src_td, data, dev, base)
    }

    /// `git diff --binary base` of a working tree, untracked files included,
    /// built the way the Claude Code hook builds it.
    fn hook_patch(dev: &std::path::Path, base: &str) -> Vec<u8> {
        let ix = dev.join(".git").join("indexio-index");
        let run = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(dev)
                .args(args)
                .env("GIT_INDEX_FILE", &ix)
                .output()
                .unwrap();
            assert!(out.status.success());
            out.stdout
        };
        run(&["read-tree", base]);
        run(&["add", "-A"]);
        run(&["diff", "--cached", "--binary", base])
    }

    fn http_raw(port: u16, method: &str, path: &str, headers: &[(&str, &str)], body: &[u8]) -> (u16, String) {
        use std::io::{Read as _, Write as _};
        let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        let mut req = format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n");
        for (k, v) in headers {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        req.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
        let mut bytes = req.into_bytes();
        bytes.extend_from_slice(body);
        s.write_all(&bytes).unwrap();
        let mut buf = Vec::new();
        s.read_to_end(&mut buf).unwrap();
        let text = String::from_utf8_lossy(&buf);
        let status = text.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
        (status, text.split("\r\n\r\n").nth(1).unwrap_or("").to_string())
    }

    /// One MCP tools/call over POST /mcp as `token`; the tool's text.
    fn mcp_call(port: u16, token: &str, tool: &str, args: Value) -> String {
        let body = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": { "name": tool, "arguments": args } })
        .to_string();
        let auth = format!("Bearer {token}");
        let (status, out) = http_raw(
            port,
            "POST",
            "/mcp",
            &[("Authorization", &auth), ("Content-Type", "application/json"), ("Accept", "application/json, text/event-stream")],
            body.as_bytes(),
        );
        assert_eq!(status, 200, "{out}");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v.get("error").is_none(), "{v}");
        v["result"]["content"][0]["text"].as_str().unwrap_or("").to_string()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn team_overlays_are_per_user_over_mcp() {
        let (_src, data, dev, base) = team_fixture();
        let acl: HashMap<String, Vec<String>> = [
            ("tok-alice".to_string(), vec!["*".to_string()]),
            ("tok-bob".to_string(), vec!["*".to_string()]),
            ("tok-app".to_string(), vec!["app".to_string()]),
        ]
        .into();
        let users: HashMap<String, String> =
            [("tok-alice".to_string(), "alice".to_string()), ("tok-bob".to_string(), "bob".to_string())].into();
        let (_data, port) = start_server_users(AuthConfig::Acl(Arc::new(acl)), users, data).await;
        let post_patch = |token: &str, repo: &str, base: &str, body: &[u8]| {
            let auth = format!("Bearer {token}");
            http_raw(port, "POST", "/team/worktree", &[("Authorization", &auth), ("X-Indexio-Repo", repo), ("X-Indexio-Base", base)], body)
        };

        // alice edits a file, adds one, deletes one, commits nothing
        std::fs::write(dev.join("src/lib.rs"), "pub fn shared_entry() { alice_wip_helper(); }\nfn alice_wip_helper() {}\n").unwrap();
        std::fs::write(dev.join("src/draft.rs"), "pub fn untracked_alice_draft() {}\n").unwrap();
        std::fs::remove_file(dev.join("src/old.rs")).unwrap();
        let patch = hook_patch(&dev, &base);
        let (status, out) = post_patch("tok-alice", "https://example.com/Acme/app", &base, &patch);
        assert_eq!(status, 200, "{out}");
        let r: Value = serde_json::from_str(&out).unwrap();
        assert_eq!((r["changed"].as_u64(), r["deleted"].as_u64()), (Some(2), Some(1)), "{r}");

        // alice sees her working tree
        let a = mcp_call(port, "tok-alice", "code_search", json!({ "query": "alice_wip_helper" }));
        assert!(a.contains("src/lib.rs"), "{a}");
        let a = mcp_call(port, "tok-alice", "find_symbol", json!({ "name": "untracked_alice_draft" }));
        assert!(a.contains("src/draft.rs"), "{a}");
        let a = mcp_call(port, "tok-alice", "code_search", json!({ "query": "retired_function" }));
        assert!(!a.contains("old.rs"), "deleted file still visible to alice: {a}");
        let a = mcp_call(port, "tok-alice", "code_search", json!({ "query": "helper_v1" }));
        assert!(!a.contains("src/lib.rs"), "alice sees the base version of her edited file: {a}");

        // bob sees the shared default branch only
        let b = mcp_call(port, "tok-bob", "code_search", json!({ "query": "alice_wip_helper" }));
        assert!(!b.contains("src/lib.rs"), "{b}");
        let b = mcp_call(port, "tok-bob", "code_search", json!({ "query": "retired_function" }));
        assert!(b.contains("old.rs"), "{b}");

        // a token limited to `app` cannot see `secret`, whatever the tool
        let c = mcp_call(port, "tok-app", "list_files", json!({ "pattern": "**/*.py" }));
        assert!(!c.contains("keys.py"), "{c}");
        let c = mcp_call(port, "tok-app", "find_symbol", json!({ "name": "classified_routine" }));
        assert!(!c.contains("keys.py"), "{c}");
        let a = mcp_call(port, "tok-alice", "find_symbol", json!({ "name": "classified_routine" }));
        assert!(a.contains("keys.py"), "{a}");

        // tools that act on the server's own machine are not offered
        let (_, list) = http_raw(port, "POST", "/mcp", &[("Authorization", "Bearer tok-bob")], br#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#);
        assert!(list.contains("code_search") && !list.contains("refresh_index"), "{list}");
        // a notification gets 202 and no body
        let (status, _) = http_raw(port, "POST", "/mcp", &[("Authorization", "Bearer tok-bob")], br#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
        assert_eq!(status, 202);

        // an unknown base is a 409; an empty patch clears the overlay
        let (status, _) = post_patch("tok-alice", "app", &"b".repeat(40), b"x");
        assert_eq!(status, 409);
        let (status, _) = post_patch("tok-alice", "app", &base, b"");
        assert_eq!(status, 200);
        let a = mcp_call(port, "tok-alice", "code_search", json!({ "query": "retired_function" }));
        assert!(a.contains("old.rs"), "{a}");
        // a token without a mapped user cannot upload
        let (status, _) = post_patch("tok-app", "app", &base, &patch);
        assert_eq!(status, 400);
    }

    // -----------------------------------------------------------------------
    // SPEC-P12: enterprise SSO (OIDC) over the whole surface
    // -----------------------------------------------------------------------

    /// Static 2048-bit RSA test key (PKCS#1); the mock provider serves the
    /// matching public JWK. Test material only, never a real key.
    const OIDC_TEST_PEM: &str = "-----BEGIN RSA PRIVATE KEY-----\n\
        MIIEpAIBAAKCAQEArHXrrl2UXuUJ/neaJ//qwFM8uwm3tkc/1ciJlnDNY62pi/gZ\n\
        HhwRkNCRDDCgiWmANxA0oynVXggesmNM0ASiNipcW71QY1MxyT9o9kUlhgtVLKi9\n\
        Z7QYeFeyGx84XpQq75BL0NxnYCrMULjUUYE9tJkaFKJYpt2Lm14avb5Ok9C7ED24\n\
        gbMUXSF0eE1jQ23nry4QpsboKMKtdelAWkx6hj6paltQ5HE/U23Xec4K7358ha4T\n\
        /J9Q2Pf9qvuz8dYEilrby66XrsH57WVCUgfVPy6zrPxDPyYcU8gbCNmlq4zBUkpi\n\
        C4VazaQOm7cOFW4KfOOk8dOaYQJiru0EeElXdwIDAQABAoIBABc2Y/t7Iv5Gy7qR\n\
        dJFPs9QhH/p4y15gZqoqrMIv+qUg+cIaKZ9Q3dhlCjDe1qzII3bF2p/fgJWAeElA\n\
        blVNWlv6BaZfa9OCnh/dRg5nri5FljhFmgC8T9La0uEtqZOpU8Ic5Od+0vcxq4Bt\n\
        8D3sLFcDiGwgkdgb16+Y0faaB8+DPyLeZIE8d0o5m6juwl/CVfw2QBBXoyrKgRQr\n\
        vp1JKqx+4Bp11bOO0/E7iJgbjwNuxv5uxw43Kc2/mcY1FLetr8w5ECwWnO/0sSVt\n\
        +I4o2raXk5BygfGU1B4fYKDEVy6Q/ozH1MKmosMhfFZT/7O2JgQhw7LEMXTjnVfX\n\
        6cFmbrkCgYEA2PEOT03cj3dm4D/tI/gQ8CSbM4iKVGmigSQMWzUElRUYNWP+HHGL\n\
        XncBCRhu5uhOvsN9JOFYVZ7crcqasYnSJCCCeNwxC87jW+ztH5lllFxhotVgBo3P\n\
        H884PCQbQ3pnwWK/1dZGYSslsIQPlrytOUoBY8/potxNCEbXog9RbZMCgYEAy4K2\n\
        0Kzqyj0z+tqnI2O4gjhKXBIV0YVwrtbuqZ8P4rlIN5AhtK8hrW7TUNBnb/pdcB6a\n\
        TCZxus0tO3NoJqHfYfCRW+hqVVOIGcaRU61NsCRaQGh0jZo+wxXb6iZVYnwhoT6d\n\
        ONCWYFlhgaY6UJx3DVLQYcuivcNOWTt/s86CfQ0CgYEAw1RIDh+M96AKgN8OJdS1\n\
        a4OKOlw2MMrsBlruxTB3b8QOiAQASJvzYJrF0+qr8Dw6qohZpVtArdbb258Qqcnt\n\
        65lZ4HhhsMAW9i3dUxZK38pOHs8AJuaIF5v8hin8YkVUJktDbsX/mH3A8a32W0KG\n\
        tY5ssfIB6yFwOoOOo9wm9QECgYB9xrOmFLindVwC1dAmlyMZmCCc9rB1ZbtW0499\n\
        VclDnq97Z6DtQq/VuIDxmVvUYTAOc1t5ZOk1QkmKTLE57yFYLo4n92SAh7e99nMq\n\
        /BjfnBgLZoNiYMoZWBEqjbaHv6ApP8F7s668rYEN1+aCm7EYku4nAuv5zBNIIvWx\n\
        8xfCoQKBgQDXbZYandHlOdMwmew3plr7BDmGJcP+eDTYnx5qlUMvotIZ42ZW8rKG\n\
        5IGDZInNfrqIiY6aQ9ikiuoQhQlzTSJ6jn7GxeCi/53PAml22gVwahtNgXvDuWV3\n\
        nDN6H28w8LD8F5pdfHCRsWJ/Rnmvvi7YfQNOi6vMYAG1sW8jgqH6Wg==\n\
        -----END RSA PRIVATE KEY-----\n";
    const OIDC_TEST_KID: &str = "pilot-test-key-1";
    const OIDC_TEST_N: &str = "rHXrrl2UXuUJ_neaJ__qwFM8uwm3tkc_1ciJlnDNY62pi_gZHhwRkNCRDDCgiWmANxA0oynVXggesmNM0ASiNipcW71QY1MxyT9o9kUlhgtVLKi9Z7QYeFeyGx84XpQq75BL0NxnYCrMULjUUYE9tJkaFKJYpt2Lm14avb5Ok9C7ED24gbMUXSF0eE1jQ23nry4QpsboKMKtdelAWkx6hj6paltQ5HE_U23Xec4K7358ha4T_J9Q2Pf9qvuz8dYEilrby66XrsH57WVCUgfVPy6zrPxDPyYcU8gbCNmlq4zBUkpiC4VazaQOm7cOFW4KfOOk8dOaYQJiru0EeElXdw";
    const OIDC_TEST_E: &str = "AQAB";

    /// A throwaway OIDC provider: discovery document + JWKS holding the
    /// static test key.
    async fn start_oidc_provider() -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = axum::Router::new()
            .route(
                "/.well-known/openid-configuration",
                axum::routing::get(move || {
                    async move {
                        axum::Json(serde_json::json!({
                            "jwks_uri": format!("http://127.0.0.1:{port}/jwks"),
                        }))
                    }
                }),
            )
            .route(
                "/jwks",
                axum::routing::get(|| async {
                    axum::Json(serde_json::json!({ "keys": [{
                        "kty": "RSA", "use": "sig", "alg": "RS256",
                        "kid": OIDC_TEST_KID, "n": OIDC_TEST_N, "e": OIDC_TEST_E,
                    }] }))
                }),
            );
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        port
    }

    /// Sign a bearer token the way an IdP would: RS256 over the test RSA key,
    /// or HS256 for the algorithm-rejection case.
    fn oidc_sign(
        kid: &str,
        email: &str,
        iss: &str,
        aud: &str,
        exp_in_secs: i64,
        alg: jsonwebtoken::Algorithm,
    ) -> String {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let claims = serde_json::json!({
            "sub": format!("sub-{email}"),
            "email": email,
            "iss": iss,
            "aud": aud,
            "iat": now - 60,
            "nbf": now - 60,
            "exp": now + exp_in_secs,
        });
        let mut header = jsonwebtoken::Header::new(alg);
        header.kid = Some(kid.to_string());
        let key = match alg {
            jsonwebtoken::Algorithm::HS256 => jsonwebtoken::EncodingKey::from_secret(b"oidc-test-hs256-secret"),
            _ => jsonwebtoken::EncodingKey::from_rsa_pem(OIDC_TEST_PEM.as_bytes()).unwrap(),
        };
        jsonwebtoken::encode(&header, &claims, &key).unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn oidc_sso_gates_mcp_rest_team_and_admin() {
        let oidc_port = start_oidc_provider().await;
        let issuer = format!("http://127.0.0.1:{oidc_port}");
        let users: HashMap<String, OidcUser> = [
            (
                "alice@corp.example".to_string(),
                OidcUser { allow: vec!["*".to_string()], user: Some("alice".to_string()) },
            ),
            (
                "bob@corp.example".to_string(),
                OidcUser { allow: vec!["app".to_string()], user: None },
            ),
        ]
        .into();
        let oidc = OidcAuth::discover(&issuer, "indexio-api", users).unwrap();
        let (_src, data, dev, base) = team_fixture();
        let (_data, port) = start_server_in(AuthConfig::Oidc(oidc), data).await;

        let alice = oidc_sign(OIDC_TEST_KID, "alice@corp.example", &issuer, "indexio-api", 3600, jsonwebtoken::Algorithm::RS256);
        let bob = oidc_sign(OIDC_TEST_KID, "bob@corp.example", &issuer, "indexio-api", 3600, jsonwebtoken::Algorithm::RS256);

        // alice (allow `*`) works over MCP and REST, and reaches /admin.
        let a = mcp_call(port, &alice, "code_search", json!({ "query": "shared_entry" }));
        assert!(a.contains("src/lib.rs"), "{a}");
        let (status, body) = http_req(port, "GET", "/search?q=classified_routine", Some(&alice));
        assert_eq!(status, 200, "{body}");
        assert_eq!(hit_repos(&body), vec!["secret".to_string()]);
        let (status, _) = http_req(port, "POST", "/admin/reload", Some(&alice));
        assert_eq!(status, 200);

        // bob (allow `app`) is hit-filtered on REST and MCP, and is not admin.
        let (status, body) = http_req(port, "GET", "/search?q=classified_routine", Some(&bob));
        assert_eq!(status, 200, "{body}");
        assert!(hit_repos(&body).is_empty(), "bob saw a repo outside his allow: {body}");
        let b = mcp_call(port, &bob, "list_files", json!({ "pattern": "**/*" }));
        assert!(!b.contains("keys.py"), "{b}");
        let (status, _) = http_req(port, "POST", "/admin/reload", Some(&bob));
        assert_eq!(status, 403);

        // team overlays key on the SSO identity (or its override): alice's
        // overlay is "alice", bob's is his bare email.
        std::fs::write(dev.join("src/draft.rs"), "pub fn alice_sso_draft() {}\n").unwrap();
        let patch = hook_patch(&dev, &base);
        let auth = format!("Bearer {alice}");
        let (status, out) = http_raw(
            port,
            "POST",
            "/team/worktree",
            &[("Authorization", &auth), ("X-Indexio-Repo", "app"), ("X-Indexio-Base", &base)],
            &patch,
        );
        assert_eq!(status, 200, "{out}");
        let (status, body) = http_req(port, "GET", "/team/overlays", Some(&alice));
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("\"user\":\"alice\""), "override not applied: {body}");
        // bob has no overlay: alice's WIP is invisible to him.
        let b = mcp_call(port, &bob, "code_search", json!({ "query": "alice_sso_draft" }));
        assert!(!b.contains("src/draft.rs"), "{b}");
        // bob pushes his own overlay under his email identity.
        let authb = format!("Bearer {bob}");
        let (status, out) = http_raw(
            port,
            "POST",
            "/team/worktree",
            &[("Authorization", &authb), ("X-Indexio-Repo", "app"), ("X-Indexio-Base", &base)],
            &patch,
        );
        assert_eq!(status, 200, "{out}");
        let (status, body) = http_req(port, "GET", "/team/overlays", Some(&bob));
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("bob@corp.example"), "{body}");

        // every failure mode is a plain 401 (fail closed).
        let carol = oidc_sign(OIDC_TEST_KID, "carol@corp.example", &issuer, "indexio-api", 3600, jsonwebtoken::Algorithm::RS256);
        let bad_aud = oidc_sign(OIDC_TEST_KID, "alice@corp.example", &issuer, "other-api", 3600, jsonwebtoken::Algorithm::RS256);
        let bad_iss = oidc_sign(OIDC_TEST_KID, "alice@corp.example", "http://evil.example", "indexio-api", 3600, jsonwebtoken::Algorithm::RS256);
        let expired = oidc_sign(OIDC_TEST_KID, "alice@corp.example", &issuer, "indexio-api", -3600, jsonwebtoken::Algorithm::RS256);
        let hs256 = oidc_sign(OIDC_TEST_KID, "alice@corp.example", &issuer, "indexio-api", 3600, jsonwebtoken::Algorithm::HS256);
        let wrong_kid = oidc_sign("rotated-key", "alice@corp.example", &issuer, "indexio-api", 3600, jsonwebtoken::Algorithm::RS256);
        // a valid token with one payload char changed: signature no longer matches
        let mut tampered = alice.clone();
        let dot = tampered.find('.').unwrap();
        let idx = tampered[dot..].find('a').map(|i| dot + i).unwrap();
        tampered.replace_range(idx..idx + 1, "b");
        for (label, tok) in [
            ("unknown user", &carol),
            ("wrong audience", &bad_aud),
            ("wrong issuer", &bad_iss),
            ("expired", &expired),
            ("HS256", &hs256),
            ("unknown kid", &wrong_kid),
            ("tampered payload", &tampered),
        ] {
            let (status, body) = http_req(port, "GET", "/search?q=x", Some(tok));
            assert_eq!(status, 401, "{label} unexpectedly accepted: {body}");
        }
        let (status, _) = http_req(port, "GET", "/search?q=x", None);
        assert_eq!(status, 401);
        // /health stays open.
        let (status, _) = http_req(port, "GET", "/health", None);
        assert_eq!(status, 200);
    }
}
