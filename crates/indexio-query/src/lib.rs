//! indexio-query: query parsing, planning (the moat), candidate generation,
//! verification and ranking. Contract: docs/SPEC.md, section "indexio-query".
//!
//! Documented API adjustments vs SPEC (allowed):
//!   - `Engine::search` returns `SearchResult { hits, truncated, took_ms }`
//!     instead of a bare `Vec<SearchHit>` (SPEC step 1 explicitly permits
//!     this side channel for the `truncated` flag).
//!   - A query consisting only of filters (no literals, no regexes) is
//!     rejected as `QueryError::Empty`, like the empty query.
//!   - Smart-case is per-literal: a literal is case-insensitive iff it
//!     contains no uppercase ASCII bytes (SPEC says "query is
//!     all-lowercase"; per-literal is the strictly more useful reading and
//!     matches the per-literal `case_insensitive` field in the contract).
//!   - `Engine::open(dir)` opens `<dir>/shards` when that subdirectory
//!     exists, otherwise `dir` itself (so tests / tools can point straight
//!     at a shard directory).
//!
//! Planner invariants (docs/SPEC.md steps 1-6):
//!   1. Grams are computed with `CommonGrams::empty()`. Shards are built at
//!      ingest time with `CommonGrams::empty()` (indexio-ingest / tests), so
//!      `grams::grams_of(needle, empty)` yields exactly the gram keys that
//!      `grams::extract(content, empty)` indexed — every content occurrence
//!      of a case-sensitive literal with >= 1 gram is in the intersection.
//!   2. Literals with >= 1 gram and regex `required_literals` with grams
//!      drive posting-list intersection (SPEC-P9: only the rarest
//!      `MAX_CONSTRAINT_GRAMS` trigrams, rarity read off the posting payload
//!      size; shortest list first, galloping seek-merge). Case-INSENSITIVE
//!      literals union the posting lists of every ASCII case variant of
//!      each trigram (the index is case-sensitive), which keeps full recall
//!      without the brute scan. Only needles too short for any gram
//!      degrade to the bounded brute scan over filter-passing docs
//!      (<= 64 MiB of content, then `truncated = true`).
//!   3. Phrase literals are verified as exact byte substrings on content
//!      (smart-case applies); positional gram pre-checks are skipped in
//!      favour of content verification.
//!   4. Filters (repo substring ci, lang equality, path substring ci) are
//!      applied at doc-meta level before any content load.
//!   5. Candidates are capped at `CANDIDATE_CAP` (2000) before verify;
//!      exceeding the cap sets `truncated = true`.
//!   6. Ranking is BM25-lite (idf per literal from its pre-verify document
//!      frequency, tf = capped match count, k1 = 1.2, b = 0.75, document
//!      length norm against the corpus average) with multiplicative boosts:
//!      path contains literal x5.0, exact symbol-name match x2.5. Sort
//!      desc; stable tiebreak by (repo, path, line).

#![allow(clippy::type_complexity)]

#![forbid(unsafe_code)]

pub mod impact;
pub use impact::{
    ChangedSymbol, FileChange, FileImpact, ImpactOptions, ImpactReport, ImpactSite,
    MAX_SPAN_LINES,
};

use std::collections::{HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, SystemTime};

use rayon::prelude::*;
use indexio_core::grams::{self, CommonGrams};
use indexio_core::verify;
use indexio_embed::bm25::Bm25Set;
use indexio_embed::embed::Embedder;
use indexio_embed::index::VecSet;
use indexio_index::ShardSet;
use indexio_types::{BlobId, DocMeta, EngineStats, Lang, SearchHit};

/// Maximum candidates verified per search (SPEC step 5).
pub const CANDIDATE_CAP: usize = 2000;
/// Brute-scan fallback budget: at most this many content bytes are scanned
/// when no gram constraint exists (SPEC step 1).
pub const SCAN_BUDGET_BYTES: u64 = 64 * 1024 * 1024;
/// Snippet length cap (SPEC step 5).
pub const SNIPPET_MAX_CHARS: usize = 240;
/// Per-doc match cap for verification / tf counting.
const MAX_HITS_PER_DOC: usize = 256;
/// BM25-lite parameters (SPEC step 6).
const K1: f64 = 1.2;
const B: f64 = 0.75;
/// Ranking boosts (SPEC step 6).
const PATH_BOOST: f64 = 5.0;
const SYMBOL_BOOST: f64 = 2.5;

// ---------------------------------------------------------------------------
// Query model + parser
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct Query {
    pub literals: Vec<Literal>,
    pub regexes: Vec<String>,
    pub filters: Filters,
}

#[derive(Clone, Debug)]
pub struct Literal {
    pub text: Vec<u8>,
    pub phrase: bool,
    pub case_insensitive: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Filters {
    pub repo: Option<String>,
    pub lang: Option<Lang>,
    pub path: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    #[error("empty query")]
    Empty,
    #[error("invalid regex /{0}/")]
    InvalidRegex(String),
}

enum Token {
    Word(String),
    Phrase(String),
    Regex(String),
}

/// Split input into whitespace-separated tokens, honouring `"phrase"` and
/// `/regex/` spans. Unterminated quotes/slashes run to end of input; `\/`
/// inside a regex is an escaped slash.
fn tokenize(input: &str) -> Vec<Token> {
    let chars: Vec<char> = input.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        match chars[i] {
            '"' => {
                i += 1;
                let start = i;
                while i < chars.len() && chars[i] != '"' {
                    i += 1;
                }
                toks.push(Token::Phrase(chars[start..i].iter().collect()));
                if i < chars.len() {
                    i += 1; // closing quote
                }
            }
            '/' => {
                i += 1;
                let mut pat = String::new();
                while i < chars.len() {
                    if chars[i] == '\\' && i + 1 < chars.len() && chars[i + 1] == '/' {
                        pat.push('/');
                        i += 2;
                    } else if chars[i] == '/' {
                        break;
                    } else {
                        pat.push(chars[i]);
                        i += 1;
                    }
                }
                if i < chars.len() {
                    i += 1; // closing slash
                }
                toks.push(Token::Regex(pat));
            }
            _ => {
                let start = i;
                while i < chars.len() && !chars[i].is_whitespace() {
                    i += 1;
                }
                toks.push(Token::Word(chars[start..i].iter().collect()));
            }
        }
    }
    toks
}

/// Smart-case rule: case-insensitive iff the literal has no uppercase
/// ASCII bytes.
fn smart_case_insensitive(text: &[u8]) -> bool {
    !text.iter().any(|b| b.is_ascii_uppercase())
}

/// Validate a regex pattern by compiling it (`find_regex` returns `None`
/// only when the pattern fails to compile).
fn regex_valid(pattern: &str) -> bool {
    verify::find_regex(b"", pattern, 1).is_some()
}

pub fn parse(input: &str) -> Result<Query, QueryError> {
    let mut q = Query::default();
    // case:yes = case-sensitive, case:no = case-insensitive (overrides smart-case).
    let mut case_override: Option<bool> = None;
    for tok in tokenize(input) {
        match tok {
            Token::Phrase(s) => {
                if !s.is_empty() {
                    q.literals.push(Literal {
                        text: s.into_bytes(),
                        phrase: true,
                        case_insensitive: false, // resolved below
                    });
                }
            }
            Token::Regex(pat) => {
                if !regex_valid(&pat) {
                    return Err(QueryError::InvalidRegex(pat));
                }
                q.regexes.push(pat);
            }
            Token::Word(w) => {
                if let Some(v) = w.strip_prefix("repo:") {
                    q.filters.repo = Some(v.to_string());
                } else if let Some(v) = w.strip_prefix("lang:") {
                    q.filters.lang = Some(Lang::from_name(v));
                } else if let Some(v) = w.strip_prefix("path:") {
                    q.filters.path = Some(v.to_string());
                } else if let Some(v) = w.strip_prefix("case:") {
                    match v {
                        "yes" => case_override = Some(true),
                        "no" => case_override = Some(false),
                        _ => {} // unknown value: ignore, keep smart-case
                    }
                } else if !w.is_empty() {
                    q.literals.push(Literal {
                        text: w.into_bytes(),
                        phrase: false,
                        case_insensitive: false, // resolved below
                    });
                }
            }
        }
    }
    if q.literals.is_empty() && q.regexes.is_empty() {
        return Err(QueryError::Empty);
    }
    for lit in &mut q.literals {
        lit.case_insensitive = match case_override {
            Some(sensitive) => !sensitive,
            None => smart_case_insensitive(&lit.text),
        };
    }
    Ok(q)
}

// ---------------------------------------------------------------------------
// Search result
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct SearchResult {
    pub hits: Vec<SearchHit>,
    /// True when the candidate cap or the brute-scan byte budget cut the
    /// result set short.
    pub truncated: bool,
    pub took_ms: u64,
}

// ---------------------------------------------------------------------------
// Semantic plane (SPEC-P2 §4)
// ---------------------------------------------------------------------------

/// Search mode for the CLI/HTTP/MCP `mode` parameter. Default: Lexical.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SearchMode {
    /// Trigram/BM25 lexical search (exact identifiers, phrases, regex).
    #[default]
    Lexical,
    /// Pure vector search over the embedding sidecar (natural-language
    /// concept queries).
    Semantic,
    /// RRF (k=60) fusion of lexical and semantic top-N lists.
    Hybrid,
}

impl SearchMode {
    /// Parse a `mode` string; Err on anything but the three known modes.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "lexical" => Ok(SearchMode::Lexical),
            "semantic" => Ok(SearchMode::Semantic),
            "hybrid" => Ok(SearchMode::Hybrid),
            other => Err(format!(
                "invalid mode '{other}' (expected lexical|semantic|hybrid)"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            SearchMode::Lexical => "lexical",
            SearchMode::Semantic => "semantic",
            SearchMode::Hybrid => "hybrid",
        }
    }
}

impl std::str::FromStr for SearchMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        SearchMode::parse(s)
    }
}

/// Fusion algorithm for the hybrid legs (SPEC-P5 §A3). Default: Rrf.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FusionAlgo {
    /// Reciprocal-rank fusion, k = 60: score = Σ_legs 1/(60 + rank).
    #[default]
    Rrf,
    /// Min-max normalize each leg's scores to [0,1] (a flat leg maps to
    /// 1.0), sum, then multiply by the number of legs the doc appears in
    /// (MNZ). Score magnitudes carry signal; rewards multi-leg presence.
    CombMnz,
}

impl FusionAlgo {
    /// Parse a `fusion` string; Err on anything but the two known algos.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "rrf" => Ok(FusionAlgo::Rrf),
            "combmnz" => Ok(FusionAlgo::CombMnz),
            other => Err(format!(
                "invalid fusion '{other}' (expected rrf|combmnz)"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            FusionAlgo::Rrf => "rrf",
            FusionAlgo::CombMnz => "combmnz",
        }
    }
}

impl std::str::FromStr for FusionAlgo {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        FusionAlgo::parse(s)
    }
}

/// SPEC-P5 §A4: strip English stopwords (indexio-embed::embed::stopwords) from
/// query text before the semantic/BM25 legs. Query-side only; the
/// conjunctive lexical mode is untouched (precision there is a feature).
pub fn clean_query(q: &str) -> String {
    let stop: std::collections::BTreeSet<&str> =
        indexio_embed::embed::stopwords().iter().copied().collect();
    q.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .filter(|s| !stop.contains(s.to_ascii_lowercase().as_str()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// One hybrid (fused) hit. `rrf` is the fused score (the CombMNZ score when
/// `FusionAlgo::CombMnz` is selected — the field name predates P5 and is
/// kept for API/JSON compatibility); `lex_rank` / `bm25_rank` / `sem_rank`
/// are 1-based ranks in the contributing legs (`None` when the hit came
/// from other legs — e.g. no vec index yet => all `sem_rank` are `None`).
/// `rerank_score` is `Some` only when a reranker stage ran (SPEC-P3 §2).
#[derive(Clone, Debug)]
pub struct HybridHit {
    pub hit: SearchHit,
    pub rrf: f64,
    pub lex_rank: Option<usize>,
    /// 1-based rank in the chunk-BM25F leg (SPEC-P5 §A3; `None` when the
    /// sidecar is absent or the doc only surfaces elsewhere).
    pub bm25_rank: Option<usize>,
    pub sem_rank: Option<usize>,
    /// Reranker score (None when no reranker ran).
    pub rerank_score: Option<f64>,
}

/// RRF constant (SPEC-P2 §4, SPEC-P5 §A3).
pub const RRF_K: f64 = 60.0;

/// Per-leg fusion weights `(lexical, bm25, semantic)` (SPEC-P10 §15): the
/// legs are not equally trustworthy on natural-language questions — on 80
/// model-written questions over the user's repos, chunk BM25 alone beat the
/// equal-weight fusion, and these weights recovered it (recall@5 61 → 68 %
/// with the repo known, 53 → 58 % without). `INDEXIO_FUSE_WEIGHTS=lex,bm25,sem`
/// overrides for experiments.
pub const FUSE_WEIGHTS: (f64, f64, f64) = (0.5, 1.0, 0.3);

fn fuse_weights() -> (f64, f64, f64) {
    static W: std::sync::OnceLock<(f64, f64, f64)> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("INDEXIO_FUSE_WEIGHTS")
            .ok()
            .and_then(|s| {
                let v: Vec<f64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                (v.len() == 3).then(|| (v[0], v[1], v[2]))
            })
            .unwrap_or(FUSE_WEIGHTS)
    })
}

/// Which retrieval leg a hit list came from (SPEC-P5 §A3).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Leg {
    Lex,
    Bm25,
    Sem,
}

/// Generic fusion of per-leg hit lists, keyed on (repo, path) (SPEC-P5
/// §A3). Rrf: Σ 1/(RRF_K + 1-based rank) per leg. CombMnz: each leg's
/// scores min-max normalized to [0,1] (a flat leg contributes 1.0 per
/// entry), summed, then multiplied by the number of legs containing the
/// doc. A doc appearing in one leg several times (multiple chunks of one
/// file in the BM25/semantic legs) accumulates its occurrences with
/// damped weights 1, 1/2, 1/4 (SPEC-P9; unbounded before); its leg COUNT
/// for MNZ still rises by one. The hit with the best (lowest) rank across
/// legs supplies snippet/line.
pub(crate) fn fuse_legs(
    legs: &[(&[SearchHit], Leg)],
    algo: FusionAlgo,
    limit: usize,
    weights: (f64, f64, f64),
) -> Vec<HybridHit> {
    struct Acc {
        hit: SearchHit,
        best_rank: usize,
        score: f64,
        n_legs: u32,
        lex_rank: Option<usize>,
        bm25_rank: Option<usize>,
        sem_rank: Option<usize>,
    }
    let mut map: HashMap<(String, String), Acc> = HashMap::new();
    for &(hits, leg) in legs {
        let weight = match leg {
            Leg::Lex => weights.0,
            Leg::Bm25 => weights.1,
            Leg::Sem => weights.2,
        };
        let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
        for h in hits {
            mn = mn.min(h.score);
            mx = mx.max(h.score);
        }
        let flat = !(mx > mn); // empty or constant-score leg
        // SPEC-P9: a file's repeat chunks in one leg are damped (1, 1/2,
        // 1/4, then nothing) instead of piling up without bound — a big
        // file with a dozen mediocre chunks in the semantic top-50 used to
        // bury a small file that was rank 1 in BOTH lexical legs.
        let mut repeats: HashMap<(String, String), u32> = HashMap::new();
        for (i, h) in hits.iter().enumerate() {
            let rank = i + 1;
            let key = (h.repo.clone(), h.path.clone());
            let n = repeats.entry(key.clone()).or_insert(0);
            let damp = match *n {
                0 => 1.0,
                1 => 0.5,
                2 => 0.25,
                _ => 0.0,
            };
            *n += 1;
            let acc = map.entry(key).or_insert_with(|| Acc {
                hit: h.clone(),
                best_rank: rank,
                score: 0.0,
                n_legs: 0,
                lex_rank: None,
                bm25_rank: None,
                sem_rank: None,
            });
            acc.score += damp
                * weight
                * match algo {
                    FusionAlgo::Rrf => 1.0 / (RRF_K + rank as f64),
                    FusionAlgo::CombMnz => {
                        if flat {
                            1.0
                        } else {
                            f64::from(h.score - mn) / f64::from(mx - mn)
                        }
                    }
                };
            let slot = match leg {
                Leg::Lex => &mut acc.lex_rank,
                Leg::Bm25 => &mut acc.bm25_rank,
                Leg::Sem => &mut acc.sem_rank,
            };
            if slot.is_none() {
                acc.n_legs += 1;
            }
            *slot = Some(slot.map_or(rank, |r: usize| r.min(rank)));
            if rank < acc.best_rank {
                acc.best_rank = rank;
                acc.hit = h.clone(); // best snippet/line per key
            }
        }
    }
    let mut out: Vec<HybridHit> = map
        .into_values()
        .map(|a| HybridHit {
            hit: a.hit,
            rrf: match algo {
                FusionAlgo::Rrf => a.score,
                FusionAlgo::CombMnz => a.score * f64::from(a.n_legs),
            },
            lex_rank: a.lex_rank,
            bm25_rank: a.bm25_rank,
            sem_rank: a.sem_rank,
            rerank_score: None,
        })
        .collect();
    out.sort_by(|a, b| {
        b.rrf
            .total_cmp(&a.rrf)
            .then_with(|| a.hit.repo.cmp(&b.hit.repo))
            .then_with(|| a.hit.path.cmp(&b.hit.path))
    });
    out.truncate(limit);
    out
}

/// Reciprocal-rank fusion of a lexical and a semantic hit list (SPEC-P2 §4
/// two-leg form; kept for the existing unit test — the hybrid search path
/// uses the 3-leg [`fuse_legs`] via `search_hybrid_fused`).
#[cfg(test)]
pub(crate) fn rrf_fuse(lex: &[SearchHit], sem: &[SearchHit], limit: usize) -> Vec<HybridHit> {
    fuse_legs(&[(lex, Leg::Lex), (sem, Leg::Sem)], FusionAlgo::Rrf, limit, (1.0, 1.0, 1.0))
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Trigrams decoded per literal for the candidate intersection (the
/// rarest ones, by posting payload size).
const MAX_CONSTRAINT_GRAMS: usize = 6;

/// Every ASCII case spelling of `gram` (non-letters kept as-is): the keys
/// a case-insensitive literal has to union over.
fn case_variants(gram: &[u8]) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = vec![Vec::with_capacity(gram.len())];
    for &b in gram {
        if b.is_ascii_alphabetic() {
            let mut next = Vec::with_capacity(out.len() * 2);
            for v in &out {
                let mut lo = v.clone();
                lo.push(b.to_ascii_lowercase());
                next.push(lo);
                let mut up = v.clone();
                up.push(b.to_ascii_uppercase());
                next.push(up);
            }
            out = next;
        } else {
            for v in &mut out {
                v.push(b);
            }
        }
    }
    out
}

/// The (line, col) a hit should point at (SPEC-P9). One constraint: its
/// first match. Several (a multi-word query): the line matched by the most
/// distinct constraints, ties broken by the rarest constraint present, then
/// the earliest line — the earliest match of *any* word was usually a
/// shebang or licence header for natural-language queries.
fn snippet_position(found: &[(f64, Vec<(u32, u32)>)]) -> Option<(u32, u32)> {
    if found.len() == 1 {
        return found[0].1.first().copied();
    }
    // line -> (distinct constraints, min df, first col)
    let mut lines: HashMap<u32, (u32, f64, u32)> = HashMap::new();
    for (df, matches) in found {
        let mut seen_line: Option<u32> = None;
        for &(line, col) in matches {
            if seen_line == Some(line) {
                continue;
            }
            seen_line = Some(line);
            let e = lines.entry(line).or_insert((0, f64::INFINITY, col));
            e.0 += 1;
            if *df < e.1 {
                e.1 = *df;
                e.2 = col;
            }
        }
    }
    lines
        .into_iter()
        .max_by(|(la, (ca, dfa, _)), (lb, (cb, dfb, _))| {
            ca.cmp(cb)
                .then_with(|| dfb.total_cmp(dfa))
                .then_with(|| lb.cmp(la))
        })
        .map(|(line, (_, _, col))| (line, col))
}

/// Candidate counts below this are verified on the calling thread; the
/// rayon fan-out only pays off once decompress + scan dominates.
const PAR_VERIFY_MIN: usize = 16;

/// Per-query state shared by every `verify_doc` call.
struct VerifyCtx<'a> {
    q: &'a Query,
    lit_df: &'a [Option<usize>],
    regex_dfs: &'a [usize],
    regexes: &'a [regex::bytes::Regex],
    symbol_docs: &'a [HashSet<DocRef>],
    n: f64,
    avgdl: f64,
}

pub struct Engine {
    set: ShardSet,
    /// Data dir the shard set was opened from (parent of `shards/`); the
    /// vector sidecar lives at `<data_dir>/vec/<model_id>.civec`. `None`
    /// for engines built via `from_shard_set` unless `with_data_dir` is set.
    data_dir: Option<PathBuf>,
    /// Opened semantic sidecars, keyed by model id and validated against
    /// the file's (len, mtime) on every use (SPEC-P6 perf): a long-lived
    /// engine (HTTP/MCP) pays the open once per rebuild, not per query.
    sidecars: RwLock<SidecarCache>,
    /// (repo, path) -> live doc, built on first use (the shard set is
    /// immutable for the engine's lifetime; `indexio serve` reloads by building
    /// a new Engine).
    doc_map: std::sync::OnceLock<HashMap<(String, String), DocRef>>,
    /// Parsed outlines per content blob (SPEC-P9): `read_span` sizes its
    /// default range by the enclosing definition and `find_symbol` reports
    /// definition ends; a tree-sitter parse per lookup would cost ms. Keyed
    /// by (blob, lang) rather than doc so it survives an engine reload
    /// (`inherit_caches`): a reload happens on every edit and every
    /// transcript import, and unchanged files keep their parse.
    outline_cache: Mutex<HashMap<(BlobId, Lang), Arc<Vec<indexio_symbols::OutlineItem>>>>,
    /// Decompressed doc contents for snippet extraction (SPEC-P9): the
    /// semantic/BM25 legs and symbol lookups each pull one line out of up
    /// to 50 docs per call, and agents keep returning to the same files.
    /// Keyed by blob (content-addressed, never stale); bounded, cleared
    /// wholesale when full.
    content_cache: Mutex<HashMap<BlobId, Arc<Vec<u8>>>>,
}

/// Max decompressed docs kept by `Engine::content_cache`.
const CONTENT_CACHE_CAP: usize = 512;

/// (file length, mtime) of a sidecar at open time.
type FileStamp = (u64, Option<SystemTime>);

#[derive(Default)]
struct SidecarCache {
    vec: HashMap<String, (FileStamp, Arc<VecSet>)>,
    bm25: HashMap<String, (FileStamp, Arc<Bm25Set>)>,
}


/// A doc identified across the shard set.
type DocRef = (usize, u32);

impl Engine {
    /// Open the shard set at `<data_dir>/shards` (falling back to
    /// `data_dir` itself when it directly contains `.cidx` files).
    pub fn open(data_dir: &Path) -> io::Result<Self> {
        let shards_dir = data_dir.join("shards");
        let dir = if shards_dir.is_dir() {
            shards_dir
        } else {
            data_dir.to_path_buf()
        };
        Ok(Engine {
            set: ShardSet::open_dir(&dir)?,
            data_dir: Some(data_dir.to_path_buf()),
            sidecars: RwLock::new(SidecarCache::default()),
            doc_map: std::sync::OnceLock::new(),
            outline_cache: Mutex::new(HashMap::new()),
            content_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Take over `from`'s opened semantic sidecars (SPEC-P9): a reloaded
    /// engine then reuses every parsed segment whose file is unchanged
    /// instead of re-parsing gigabytes after each refresh.
    pub fn inherit_sidecars(&self, from: &Engine) {
        let theirs = from.sidecars.read().expect("sidecar cache poisoned");
        let mut mine = self.sidecars.write().expect("sidecar cache poisoned");
        for (k, v) in &theirs.vec {
            mine.vec.entry(k.clone()).or_insert_with(|| v.clone());
        }
        for (k, v) in &theirs.bm25 {
            mine.bm25.entry(k.clone()).or_insert_with(|| v.clone());
        }
        drop(mine);
        drop(theirs);
        self.inherit_caches(from);
    }

    /// Carry the content and outline caches over from the engine being
    /// replaced (SPEC-P9 §22). Both are keyed by content blob, so nothing
    /// can be stale; a file that did not change keeps its parse.
    pub fn inherit_caches(&self, from: &Engine) {
        if let (Ok(theirs), Ok(mut mine)) = (from.content_cache.lock(), self.content_cache.lock()) {
            if mine.is_empty() {
                *mine = theirs.clone();
            }
        }
        if let (Ok(theirs), Ok(mut mine)) = (from.outline_cache.lock(), self.outline_cache.lock()) {
            if mine.is_empty() {
                *mine = theirs.clone();
            }
        }
    }

    /// Build an engine directly from an opened shard set (used by tests).
    pub fn from_shard_set(set: ShardSet) -> Self {
        Engine {
            set,
            data_dir: None,
            sidecars: RwLock::new(SidecarCache::default()),
            doc_map: std::sync::OnceLock::new(),
            outline_cache: Mutex::new(HashMap::new()),
            content_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Point the semantic plane at `dir` (for engines not built via `open`).
    pub fn with_data_dir(mut self, dir: &Path) -> Self {
        self.data_dir = Some(dir.to_path_buf());
        self
    }

    pub fn shard_set(&self) -> &ShardSet {
        &self.set
    }

    /// The vector index for `model_id`, opened once and reused until the
    /// file changes on disk (`None` when no embed run has produced one).
    pub fn vec_index(&self, model_id: &str) -> anyhow::Result<Option<Arc<VecSet>>> {
        let Some(dir) = &self.data_dir else {
            return Ok(None);
        };
        // stamp over every segment + tombstone side file (SPEC-P9)
        let Some(stamp) = indexio_embed::index::set_stamp(&dir.join("vec"), model_id) else {
            return Ok(None);
        };
        let prev: Option<Arc<VecSet>> = {
            let cache = self.sidecars.read().expect("sidecar cache poisoned");
            match cache.vec.get(model_id) {
                Some((st, idx)) if *st == stamp => return Ok(Some(idx.clone())),
                Some((_, idx)) => Some(idx.clone()),
                None => None,
            }
        };
        // reuse the parsed segments whose files did not change
        let Some(idx) = VecSet::reopen(prev.as_deref(), &dir.join("vec"), model_id)? else {
            return Ok(None);
        };
        let idx = Arc::new(idx);
        self.sidecars
            .write()
            .expect("sidecar cache poisoned")
            .vec
            .insert(model_id.to_string(), (stamp, idx.clone()));
        Ok(Some(idx))
    }

    /// The BM25F sidecar for `model_id`, cached like [`Engine::vec_index`].
    pub fn bm25_index(&self, model_id: &str) -> anyhow::Result<Option<Arc<Bm25Set>>> {
        let Some(dir) = &self.data_dir else {
            return Ok(None);
        };
        let Some(stamp) = indexio_embed::bm25::set_stamp(&dir.join("bm25"), model_id) else {
            return Ok(None);
        };
        let prev: Option<Arc<Bm25Set>> = {
            let cache = self.sidecars.read().expect("sidecar cache poisoned");
            match cache.bm25.get(model_id) {
                Some((st, bm)) if *st == stamp => return Ok(Some(bm.clone())),
                Some((_, bm)) => Some(bm.clone()),
                None => None,
            }
        };
        let Some(bm) = Bm25Set::reopen(prev.as_deref(), &dir.join("bm25"), model_id)? else {
            return Ok(None);
        };
        let bm = Arc::new(bm);
        self.sidecars
            .write()
            .expect("sidecar cache poisoned")
            .bm25
            .insert(model_id.to_string(), (stamp, bm.clone()));
        Ok(Some(bm))
    }

    /// Data dir this engine was opened from (`None` for `from_shard_set`
    /// engines without `with_data_dir`). Surfaces use it to find
    /// `<data_dir>/repos/<name>.json` (SPEC-P6 diff impact).
    pub fn data_dir(&self) -> Option<&Path> {
        self.data_dir.as_deref()
    }

    fn repo_name(&self, si: usize, dm: &DocMeta) -> String {
        self.set
            .shard(si)
            .and_then(|s| s.meta().repos.get(dm.repo_id as usize).cloned())
            .unwrap_or_default()
    }

    fn filters_pass(&self, q: &Query, si: usize, dm: &DocMeta) -> bool {
        if let Some(repo_sub) = &q.filters.repo {
            let repo = self.repo_name(si, dm);
            if !repo
                .to_ascii_lowercase()
                .contains(&repo_sub.to_ascii_lowercase())
            {
                return false;
            }
        }
        if let Some(lang) = q.filters.lang {
            if dm.lang != lang {
                return false;
            }
        }
        if let Some(path_sub) = &q.filters.path {
            // `path:a.py|b.py` — any of several substrings (a grep over a
            // handful of files, SPEC-P9 §19)
            let path = dm.path.to_ascii_lowercase();
            if !path_sub.split('|').any(|p| !p.is_empty() && path.contains(&p.to_ascii_lowercase())) {
                return false;
            }
        }
        true
    }

    /// Gram constraint for one needle (SPEC-P9 planner): the intersection
    /// of the posting lists of its rarest trigrams. Rarity is read off the
    /// posting payload length (no decode) and at most
    /// [`MAX_CONSTRAINT_GRAMS`] trigrams are decoded: the rarest few bound
    /// the candidate set as tightly as all of them would, and verification
    /// is exact anyway. Case-insensitive literals use the union of each
    /// trigram's ASCII case variants, so smart-case (all-lowercase) queries
    /// no longer fall to the brute scan. `None` when the needle yields no
    /// grams (cannot constrain); `Some(empty)` when some trigram occurs in
    /// no shard at all (cannot match).
    fn gram_candidates(&self, needle: &[u8], case_insensitive: bool) -> Option<Vec<DocRef>> {
        let gram_list = grams::grams_of(needle, &CommonGrams::empty());
        if gram_list.is_empty() {
            return None;
        }
        let mut groups: Vec<(usize, Vec<Vec<u8>>)> = Vec::with_capacity(gram_list.len());
        for g in gram_list {
            let variants = if case_insensitive { case_variants(&g) } else { vec![g] };
            let cost: usize = variants.iter().map(|v| self.set.posting_bytes(v)).sum();
            if cost == 0 {
                return Some(Vec::new());
            }
            groups.push((cost, variants));
        }
        groups.sort_by_key(|(c, _)| *c);
        groups.truncate(MAX_CONSTRAINT_GRAMS);
        let mut lists: Vec<Vec<DocRef>> = groups
            .par_iter()
            .map(|(_, variants)| {
                let mut v: Vec<DocRef> = Vec::new();
                for g in variants {
                    v.extend(self.set.posting_doc_ids(g));
                }
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();
        // Shortest list first; intersect the rest via galloping seek-merge.
        lists.sort_by_key(|l| l.len());
        let mut acc = std::mem::take(&mut lists[0]);
        for l in &lists[1..] {
            acc = galloping_intersect(&acc, l);
            if acc.is_empty() {
                break;
            }
        }
        Some(acc)
    }

    pub fn search(&self, q: &Query, limit: usize) -> SearchResult {
        self.search_with_cap_lines(q, limit, CANDIDATE_CAP, 1)
    }

    /// `search` listing up to `per_doc` matching lines of every hit
    /// document as separate hits (`grep -n` style, SPEC-P9 §19): a file's
    /// lines share its score and come out ascending, files stay in ranked
    /// order. `per_doc == 1` is `search` (the densest line per file).
    pub fn search_lines(&self, q: &Query, limit: usize, per_doc: usize) -> SearchResult {
        self.search_with_cap_lines(q, limit, CANDIDATE_CAP, per_doc.max(1))
    }

    /// `search` with an injectable candidate cap (planner step 5); exposed
    /// so the cap path can be exercised in tests without building a
    /// 2000-doc corpus.
    pub fn search_with_cap(&self, q: &Query, limit: usize, cap: usize) -> SearchResult {
        self.search_with_cap_lines(q, limit, cap, 1)
    }

    fn search_with_cap_lines(&self, q: &Query, limit: usize, cap: usize, per_doc: usize) -> SearchResult {
        let t0 = Instant::now();
        let mut truncated = false;

        let visible = self.set.visible_slice();
        let total_docs = visible.len();
        let total_raw: u64 = visible.iter().map(|(_, _, dm)| dm.raw_len as u64).sum();
        let avgdl = if total_docs > 0 {
            (total_raw as f64 / total_docs as f64).max(1.0)
        } else {
            1.0
        };

        // ---- steps 1-3: gram-constrained candidate sets -----------------
        // Each entry: (constraint's candidate list, df for idf).
        let mut constraint_lists: Vec<Vec<DocRef>> = Vec::new();
        // df per literal (pre-verify document frequency), indexed parallel
        // to q.literals; resolved after candidates are known.
        let mut lit_df: Vec<Option<usize>> = vec![None; q.literals.len()];

        for (i, lit) in q.literals.iter().enumerate() {
            if let Some(cands) = self.gram_candidates(&lit.text, lit.case_insensitive) {
                lit_df[i] = Some(cands.len());
                constraint_lists.push(cands);
            }
        }
        let mut regex_dfs: Vec<usize> = Vec::with_capacity(q.regexes.len());
        for pat in &q.regexes {
            let mut regex_cands: Option<Vec<DocRef>> = None;
            if let Some(lits) = verify::required_literals(pat) {
                // `lits` is DISJUNCTIVE (one guaranteed prefix per
                // alternation branch: every match starts with at least one
                // of them), so the candidate set is the UNION of the
                // per-literal gram intersections. A literal too short to
                // yield grams leaves the whole regex unconstrained — the
                // brute-scan fallback keeps recall.
                let mut union: Vec<DocRef> = Vec::new();
                let mut constrained = true;
                for rl in &lits {
                    match self.gram_candidates(rl, false) {
                        Some(cands) => union.extend(cands),
                        None => {
                            constrained = false;
                            break;
                        }
                    }
                }
                if constrained {
                    union.sort_unstable();
                    union.dedup();
                    regex_cands = Some(union);
                }
            }
            if let Some(cands) = regex_cands {
                regex_dfs.push(cands.len());
                constraint_lists.push(cands);
            } else {
                regex_dfs.push(usize::MAX); // resolved below
            }
        }

        // ---- candidates: intersection, or bounded brute scan ------------
        let candidates: Vec<DocRef>;
        if constraint_lists.is_empty() {
            // No usable gram constraint: brute-scan fallback over
            // filter-passing docs, bounded by content bytes (step 1).
            let mut scanned_bytes: u64 = 0;
            let mut cands = Vec::new();
            for (si, docid, dm) in visible.iter() {
                if !self.filters_pass(q, *si, dm) {
                    continue;
                }
                if scanned_bytes >= SCAN_BUDGET_BYTES {
                    truncated = true;
                    break;
                }
                scanned_bytes += dm.raw_len as u64;
                cands.push((*si, *docid));
            }
            let scan_df = cands.len().max(1);
            for d in lit_df.iter_mut() {
                if d.is_none() {
                    *d = Some(scan_df);
                }
            }
            for d in regex_dfs.iter_mut() {
                if *d == usize::MAX {
                    *d = scan_df;
                }
            }
            candidates = cands;
        } else {
            constraint_lists.sort_by_key(|l| l.len());
            let mut acc = constraint_lists[0].clone();
            for l in &constraint_lists[1..] {
                acc = galloping_intersect(&acc, l);
                if acc.is_empty() {
                    break;
                }
            }
            // Unresolved dfs (ci literals, gram-less regexes): use the
            // pre-filter intersection size as the pragmatic df (step 6).
            let inter_df = acc.len().max(1);
            for d in lit_df.iter_mut() {
                if d.is_none() {
                    *d = Some(inter_df);
                }
            }
            for d in regex_dfs.iter_mut() {
                if *d == usize::MAX {
                    *d = inter_df;
                }
            }
            // ---- step 4: doc-meta filters before content load -----------
            candidates = acc
                .into_iter()
                .filter(|&(si, docid)| match self.set.doc(si, docid) {
                    Some(dm) => self.filters_pass(q, si, &dm),
                    None => false,
                })
                .collect();
        }

        // ---- step 5: candidate cap pre-verify ---------------------------
        let pre_verify_count = candidates.len();
        if pre_verify_count > cap {
            truncated = true;
        }
        let candidates: Vec<DocRef> = candidates.into_iter().take(cap).collect();

        // ---- step 5/6: verify + score (parallel over candidates) --------
        // Regexes compile once per query; the symbol-boost lookup (one FST
        // probe per shard) resolves once per literal, not once per doc.
        let n = total_docs.max(1) as f64;
        let mut regexes: Vec<regex::bytes::Regex> = Vec::with_capacity(q.regexes.len());
        for pat in &q.regexes {
            match verify::compile_regex(pat) {
                Ok(re) => regexes.push(re),
                Err(_) => {
                    return SearchResult { hits: Vec::new(), truncated, took_ms: t0.elapsed().as_millis() as u64 };
                }
            }
        }
        let symbol_docs: Vec<HashSet<DocRef>> = q
            .literals
            .iter()
            .map(|lit| {
                let name = String::from_utf8_lossy(&lit.text);
                if name.is_empty() {
                    HashSet::new()
                } else {
                    self.set
                        .symbol_postings(&name)
                        .into_iter()
                        .map(|(s, d, _, _, _)| (s, d))
                        .collect()
                }
            })
            .collect();
        let ctx = VerifyCtx {
            q,
            lit_df: &lit_df,
            regex_dfs: &regex_dfs,
            regexes: &regexes,
            symbol_docs: &symbol_docs,
            n,
            avgdl,
        };
        let mut hits: Vec<SearchHit> = if per_doc <= 1 {
            if candidates.len() < PAR_VERIFY_MIN {
                candidates.iter().filter_map(|&(si, docid)| self.verify_doc(&ctx, si, docid)).collect()
            } else {
                candidates.par_iter().filter_map(|&(si, docid)| self.verify_doc(&ctx, si, docid)).collect()
            }
        } else {
            let per: Vec<Vec<SearchHit>> = if candidates.len() < PAR_VERIFY_MIN {
                candidates.iter().filter_map(|&(si, docid)| self.verify_doc_lines(&ctx, si, docid, per_doc)).collect()
            } else {
                candidates
                    .par_iter()
                    .filter_map(|&(si, docid)| self.verify_doc_lines(&ctx, si, docid, per_doc))
                    .collect()
            };
            per.into_iter().flatten().collect()
        };

        sort_hits(&mut hits);
        if hits.len() > limit {
            truncated = true;
        }
        hits.truncate(limit);
        SearchResult {
            hits,
            truncated,
            took_ms: t0.elapsed().as_millis() as u64,
        }
    }

    /// Verify one candidate doc against every literal/regex of the query
    /// and score it (BM25-lite + path/symbol boosts). `None` = not a hit.
    fn verify_doc(&self, c: &VerifyCtx<'_>, si: usize, docid: u32) -> Option<SearchHit> {
        self.verify_doc_lines(c, si, docid, 1).and_then(|mut v| v.pop())
    }

    /// `verify_doc` that lists up to `per_doc` distinct matching lines of
    /// the document as separate hits (all with the document's score, lines
    /// ascending, the leftmost match column per line). With `per_doc == 1`
    /// the single hit is the densest line, exactly as `verify_doc`.
    fn verify_doc_lines(&self, c: &VerifyCtx<'_>, si: usize, docid: u32, per_doc: usize) -> Option<Vec<SearchHit>> {
        let dm = self.set.doc(si, docid)?;
        let content = self.set.content(si, docid).ok()?;
        let mut score: f64 = 0.0;
        // (df, matches) per constraint, for the snippet line choice below.
        let mut found: Vec<(f64, Vec<(u32, u32)>)> =
            Vec::with_capacity(c.q.literals.len() + c.regexes.len());
        for (i, lit) in c.q.literals.iter().enumerate() {
            let m = verify::find_literal(&content, &lit.text, lit.case_insensitive, MAX_HITS_PER_DOC);
            if m.is_empty() {
                return None;
            }
            let df = c.lit_df[i].unwrap_or(1).max(1) as f64;
            score += bm25(c.n, df, m.len() as f64, dm.raw_len as f64, c.avgdl);
            found.push((df, m));
        }
        for (j, re) in c.regexes.iter().enumerate() {
            let m = verify::find_regex_with(&content, re, MAX_HITS_PER_DOC);
            if m.is_empty() {
                return None;
            }
            let df = c.regex_dfs[j].max(1) as f64;
            score += bm25(c.n, df, m.len() as f64, dm.raw_len as f64, c.avgdl);
            found.push((df, m));
        }
        let first_match = snippet_position(&found);

        // ---- boosts (step 6) ------------------------------------------
        let mut boosted = false;
        for lit in &c.q.literals {
            let path_hit = if lit.case_insensitive {
                let needle = String::from_utf8_lossy(&lit.text).to_ascii_lowercase();
                !needle.is_empty() && dm.path.to_ascii_lowercase().contains(&needle)
            } else {
                dm.path
                    .as_bytes()
                    .windows(lit.text.len())
                    .any(|w| w == lit.text.as_slice())
            };
            if path_hit && !boosted {
                score *= PATH_BOOST;
                boosted = true;
            }
        }
        if c.symbol_docs.iter().any(|set| set.contains(&(si, docid))) {
            score *= SYMBOL_BOOST;
        }

        let (line, col) = first_match.unwrap_or((1, 0));
        let snippet = snippet_line(&content, line);
        let best = SearchHit {
            repo: self.repo_name(si, &dm),
            path: dm.path.clone(),
            line,
            col,
            snippet,
            score: score as f32,
            lang: dm.lang,
        };
        if per_doc <= 1 {
            return Some(vec![best]);
        }
        // every line with a match of any constraint, ascending, leftmost
        // column per line
        let mut lines: Vec<(u32, u32)> = found.iter().flat_map(|(_, m)| m.iter().copied()).collect();
        lines.sort_unstable();
        lines.dedup_by_key(|(l, _)| *l);
        lines.truncate(per_doc);
        Some(
            lines
                .into_iter()
                .map(|(l, c)| SearchHit {
                    line: l,
                    col: c,
                    snippet: if l == best.line { best.snippet.clone() } else { snippet_line(&content, l) },
                    ..best.clone()
                })
                .collect(),
        )
    }

    /// Exact symbol lookup across shards; falls back to substring matching
    /// (score 1.0) when the exact lookup is empty. Exact-name entries score
    /// 10.0 (SPEC: find_symbol).
    pub fn find_symbol(&self, name: &str, limit: usize) -> Vec<SearchHit> {
        let mut hits = self.symbol_hits(name, 10.0);
        if hits.is_empty() {
            hits = self.substring_symbol_hits(name);
        }
        sort_hits(&mut hits);
        hits.truncate(limit);
        hits
    }

    /// Map exact `symbol_postings(name)` entries to hits.
    fn symbol_hits(&self, name: &str, score: f32) -> Vec<SearchHit> {
        let postings = self.set.symbol_postings(name);
        postings
            .par_iter()
            .filter_map(|(si, docid, _kind, line, _scope)| self.posting_hit(*si, *docid, *line, score))
            .collect()
    }

    /// Substring variant: find docs whose content contains `name`, extract
    /// identifiers containing `name` from the matching lines, and confirm
    /// each candidate identifier against the exact symbol index. (The shard
    /// format exposes exact symbol lookup only.)
    fn substring_symbol_hits(&self, name: &str) -> Vec<SearchHit> {
        let mut out: Vec<SearchHit> = Vec::new();
        let mut seen: std::collections::HashSet<(usize, u32, u32)> =
            std::collections::HashSet::new();
        let mut q = Query::default();
        q.literals.push(Literal {
            text: name.as_bytes().to_vec(),
            phrase: false,
            case_insensitive: false,
        });
        let res = self.search(&q, CANDIDATE_CAP);
        for h in res.hits {
            let Some((si, docid)) = self.locate_doc(&h.repo, &h.path) else {
                continue;
            };
            let Ok(content) = self.set.content(si, docid) else {
                continue;
            };
            for m in verify::find_literal(&content, name.as_bytes(), false, MAX_HITS_PER_DOC) {
                let line_text = content_line(&content, m.0);
                for ident in identifiers_containing(&line_text, name) {
                    for (si2, docid2, _, line, _) in self.set.symbol_postings(&ident) {
                        if seen.insert((si2, docid2, line)) {
                            if let Some(hit) = self.posting_hit(si2, docid2, line, 1.0) {
                                out.push(hit);
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Locate a live doc by (repo, path) through a lazily built map (one
    /// visible-docs scan per engine instead of one per lookup).
    fn locate_doc(&self, repo: &str, path: &str) -> Option<DocRef> {
        let map = self.doc_map.get_or_init(|| {
            self.set
                .visible_docs()
                .into_iter()
                .map(|(si, docid, dm)| ((self.repo_name(si, &dm), dm.path), (si, docid)))
                .collect()
        });
        map.get(&(repo.to_string(), path.to_string())).copied()
    }

    /// Indexed file paths matching `pattern` (SPEC-P8 `list_files`): a glob
    /// when it contains `*` or `?` (`**` crosses directories, `*`/`?` do
    /// not; anchored, case-insensitive), otherwise a case-insensitive
    /// substring. Optionally restricted to one repo. Sorted by (repo, path);
    /// `truncated` when more than `limit` matched.
    pub fn list_paths(&self, pattern: &str, repo: Option<&str>, limit: usize) -> (Vec<(String, String)>, bool) {
        let is_glob = pattern.contains('*') || pattern.contains('?');
        let matcher: Box<dyn Fn(&str) -> bool> = if is_glob {
            let mut re = String::from("(?i)^");
            let mut chars = pattern.chars().peekable();
            while let Some(c) = chars.next() {
                match c {
                    '*' => {
                        if chars.peek() == Some(&'*') {
                            chars.next();
                            if chars.peek() == Some(&'/') {
                                chars.next();
                                re.push_str("(?:.*/)?");
                            } else {
                                re.push_str(".*");
                            }
                        } else {
                            re.push_str("[^/]*");
                        }
                    }
                    '?' => re.push_str("[^/]"),
                    other => re.push_str(&regex::escape(&other.to_string())),
                }
            }
            re.push('$');
            match regex::Regex::new(&re) {
                Ok(r) => Box::new(move |p: &str| r.is_match(p)),
                Err(_) => Box::new(|_| false),
            }
        } else {
            let needle = pattern.to_ascii_lowercase();
            Box::new(move |p: &str| p.to_ascii_lowercase().contains(&needle))
        };
        let mut out: Vec<(String, String)> = self
            .set
            .visible_docs()
            .into_iter()
            .filter_map(|(si, _, dm)| {
                let r = self.repo_name(si, &dm);
                if repo.is_some_and(|want| want != r) || !matcher(&dm.path) {
                    return None;
                }
                Some((r, dm.path))
            })
            .collect();
        out.sort();
        let truncated = out.len() > limit;
        out.truncate(limit);
        (out, truncated)
    }

    /// Callers of `name`: one hit per call posting (snippet = call line,
    /// score 1.0; SPEC: who_calls).
    pub fn who_calls(&self, name: &str, limit: usize) -> Vec<SearchHit> {
        let postings = self.set.call_postings(name);
        // name-based edges judged by their qualifier against the
        // definitions (SPEC-P9 §20): `tokio::spawn(` is not a call of the
        // repo's `spawn`
        let shape = self.root_shape(name);
        let mut hits: Vec<SearchHit> = postings
            .par_iter()
            .filter_map(|(si, docid, _caller, line)| {
                let content = self.cached_content(*si, *docid)?;
                if !impact::site_plausible(&content_line(&content, *line), name, &shape) {
                    return None;
                }
                self.posting_hit(*si, *docid, *line, 1.0)
            })
            .collect();
        sort_hits(&mut hits);
        hits.truncate(limit);
        hits
    }

    /// Decompressed content of a doc through the bounded engine cache.
    fn cached_content(&self, si: usize, docid: u32) -> Option<Arc<Vec<u8>>> {
        let blob = self.set.doc(si, docid)?.blob;
        if let Some(c) = self.content_cache.lock().ok()?.get(&blob) {
            return Some(Arc::clone(c));
        }
        let content = Arc::new(self.set.content(si, docid).ok()?);
        if let Ok(mut cache) = self.content_cache.lock() {
            if cache.len() >= CONTENT_CACHE_CAP {
                cache.clear();
            }
            cache.insert(blob, Arc::clone(&content));
        }
        Some(content)
    }

    /// Build a hit for a (doc, 1-based line) posting entry.
    fn posting_hit(&self, si: usize, docid: u32, line: u32, score: f32) -> Option<SearchHit> {
        let dm = self.set.doc(si, docid)?;
        let content = self.cached_content(si, docid)?;
        let snippet = snippet_line(&content, line);
        Some(SearchHit {
            repo: self.repo_name(si, &dm),
            path: dm.path.clone(),
            line,
            col: 0,
            snippet,
            score,
            lang: dm.lang,
        })
    }

    /// Map a vec-index row to a hit (repo/path/start_line from the row
    /// meta, snippet = up to 2 lines around start_line via shard content
    /// lookup). Stale rows (doc deleted since the embed run) return None.
    fn row_hit(&self, idx: &VecSet, row: u32, score: f32) -> Option<SearchHit> {
        let m = idx.row_meta(row);
        let (si, docid) = self.locate_doc(&m.repo, &m.path)?;
        let dm = self.set.doc(si, docid)?;
        let content = self.cached_content(si, docid)?;
        let (line, snippet) = semantic_snippet(&content, m.start_line, m.end_line);
        Some(SearchHit {
            repo: m.repo.clone(),
            path: m.path.clone(),
            line,
            col: 0,
            snippet,
            score,
            lang: dm.lang,
        })
    }

    /// Semantic search (SPEC-P2 §4): embed `q` with `embedder`, run
    /// `VecIndex::search` over `<data_dir>/vec/<model_id>.civec`, map rows
    /// back to hits (repo/path/start_line from the row meta, snippet = up to
    /// 2 lines around start_line via shard content lookup, score = cosine).
    ///
    /// SPEC-P5 §A4: query text is stopword-stripped (`clean_query`) before
    /// embedding. (SPEC-P5 §B2 hook: the embedder's query expansion wraps
    /// this exact call site.)
    ///
    /// Graceful degradation: returns an empty vec (not an error) when the
    /// engine has no data dir or no vec index exists for this model yet.
    pub fn search_semantic(
        &self,
        q: &str,
        k: usize,
        embedder: &dyn Embedder,
    ) -> anyhow::Result<Vec<SearchHit>> {
        self.search_semantic_in(q, k, embedder, None)
    }

    /// [`search_semantic`](Self::search_semantic) restricted to rows of
    /// `repo` when given (SPEC-P10): the vector prescan skips other repos'
    /// rows, so a small repo (the session transcripts, say) is searched at
    /// full depth instead of competing with the whole corpus for the pool.
    pub fn search_semantic_in(
        &self,
        q: &str,
        k: usize,
        embedder: &dyn Embedder,
        repo: Option<&str>,
    ) -> anyhow::Result<Vec<SearchHit>> {
        let Some(idx) = self.vec_index(embedder.model_id())? else {
            return Ok(Vec::new()); // no data dir / no embed run yet
        };
        if k == 0 {
            return Ok(Vec::new());
        }
        // SPEC-P5 A4+B2 integration: stopword-strip, then thesaurus-expand,
        // then embed. search_hybrid's semantic leg flows through here.
        let cleaned = clean_query(q);
        if cleaned.is_empty() {
            return Ok(Vec::new()); // all-stopword query: no content signal
        }
        let t = Instant::now();
        // Thesaurus expansion is opt-in (INDEXIO_EXPAND=1): on 80
        // model-written questions it cost the semantic leg 9 points of
        // recall@5 and hybrid 4 (SPEC-P10 §18), and its first-time
        // vocabulary scan was the slowest part of a cold hybrid query.
        let expanded = if std::env::var("INDEXIO_EXPAND").is_ok_and(|v| !v.trim().is_empty()) {
            embedder.expand_query(&cleaned)
        } else {
            cleaned.clone()
        };
        let t_expand = t.elapsed().as_millis() as u64;
        let qv = embedder.embed(&[expanded])?.remove(0);
        let t_embed = t.elapsed().as_millis() as u64 - t_expand;
        let rows = match repo {
            Some(r) => idx.search_repo(&qv, k, r),
            None => idx.search(&qv, k),
        };
        let t_search = t.elapsed().as_millis() as u64 - t_expand - t_embed;
        let hits: Vec<SearchHit> = rows
            .par_iter()
            .filter_map(|&(row, score)| self.row_hit(&idx, row, score))
            .collect();
        tracing::debug!(
            expand_ms = t_expand,
            embed_ms = t_embed,
            vec_search_ms = t_search,
            hits_ms = t.elapsed().as_millis() as u64 - t_expand - t_embed - t_search,
            "semantic leg"
        );
        Ok(hits)
    }

    /// Chunk-BM25F leg (SPEC-P5 §A3): `ChunkBm25::search` over
    /// `<data_dir>/bm25/<model_id>.cibm25`, disjunctive and scored — the
    /// natural-language lexical leg. Rows are resolved through the
    /// row-aligned VecIndex of the same namespace (SPEC-P5 §A2 invariant).
    ///
    /// Graceful degradation: empty vec when the engine has no data dir, or
    /// the BM25 sidecar / vec index does not exist for this model yet.
    pub fn search_bm25(
        &self,
        q: &str,
        k: usize,
        model_id: &str,
    ) -> anyhow::Result<Vec<SearchHit>> {
        self.search_bm25_in(q, k, model_id, None)
    }

    /// [`search_bm25`](Self::search_bm25) restricted to rows of `repo`.
    pub fn search_bm25_in(
        &self,
        q: &str,
        k: usize,
        model_id: &str,
        repo: Option<&str>,
    ) -> anyhow::Result<Vec<SearchHit>> {
        let Some(bm) = self.bm25_index(model_id)? else {
            return Ok(Vec::new()); // no data dir / no embed run yet
        };
        let Some(idx) = self.vec_index(model_id)? else {
            return Ok(Vec::new());
        };
        let rows = match repo {
            Some(r) => bm.search_where(q, k, &|g| (g as usize) < idx.raw_len() && idx.row_meta(g).repo == r),
            None => bm.search(q, k),
        };
        let hits: Vec<SearchHit> = rows
            .par_iter()
            // sidecar newer than the vec index: skip stray rows
            .filter(|(row, _)| (*row as usize) < idx.raw_len())
            .filter_map(|&(row, score)| self.row_hit(&idx, row, score))
            .collect();
        Ok(hits)
    }

    /// Hybrid search (SPEC-P2 §4 / SPEC-P5 §A3): delegates to
    /// [`Engine::search_hybrid_fused`] with `FusionAlgo::Rrf` (3 legs).
    pub fn search_hybrid(
        &self,
        q: &str,
        limit: usize,
        embedder: &dyn Embedder,
    ) -> anyhow::Result<Vec<HybridHit>> {
        self.search_hybrid_fused(q, limit, embedder, FusionAlgo::Rrf)
    }

    /// 3-leg hybrid with a selectable fusion algorithm (SPEC-P5 §A3):
    ///   1. file-lexical: the existing conjunctive search (fires on
    ///      identifier/regex queries; a query that fails lexical parse
    ///      contributes no lexical list),
    ///   2. chunk-BM25: `ChunkBm25::search` (disjunctive, scored — the NL
    ///      leg; absent sidecar => no list),
    ///   3. semantic: embedder + VecIndex (absent vec index => no list).
    /// Each leg contributes its top `max(limit, 50)` hits; fusion is keyed
    /// on (repo, path). All legs are optional: any subset (including none)
    /// fuses gracefully.
    pub fn search_hybrid_fused(
        &self,
        q: &str,
        limit: usize,
        embedder: &dyn Embedder,
        algo: FusionAlgo,
    ) -> anyhow::Result<Vec<HybridHit>> {
        self.search_hybrid_in(q, limit, embedder, algo, None)
    }

    /// [`search_hybrid_fused`](Self::search_hybrid_fused) with every leg
    /// restricted to `repo` when given (SPEC-P10; `recall` uses it for the
    /// session transcripts).
    pub fn search_hybrid_in(
        &self,
        q: &str,
        limit: usize,
        embedder: &dyn Embedder,
        algo: FusionAlgo,
        repo: Option<&str>,
    ) -> anyhow::Result<Vec<HybridHit>> {
        let n = limit.max(50);
        let t = Instant::now();
        let lex_q = match repo {
            Some(r) => format!("{q} repo:{r}"),
            None => q.to_string(),
        };
        // The three legs are independent reads: run them concurrently
        // (SPEC-P9); wall time is the slowest leg, not the sum.
        let (lex, (bm, sem)) = rayon::join(
            || match parse(&lex_q) {
                Ok(query) => self.search(&query, n).hits,
                Err(_) => Vec::new(),
            },
            || {
                rayon::join(
                    || self.search_bm25_in(q, n, embedder.model_id(), repo),
                    || self.search_semantic_in(q, n, embedder, repo),
                )
            },
        );
        let bm = bm?;
        let sem = sem?;
        tracing::debug!(legs_ms = t.elapsed().as_millis() as u64, lex = lex.len(), bm25 = bm.len(), sem = sem.len(), "hybrid legs (parallel)");
        Ok(fuse_legs(
            &[(&lex, Leg::Lex), (&bm, Leg::Bm25), (&sem, Leg::Sem)],
            algo,
            limit,
            fuse_weights(),
        ))
    }

    /// Hybrid then rerank (SPEC-P3 §2): delegates to
    /// [`Engine::search_hybrid_fused_reranked`] with `FusionAlgo::Rrf`.
    pub fn search_hybrid_reranked(
        &self,
        q: &str,
        limit: usize,
        embedder: &dyn Embedder,
        reranker: &dyn indexio_embed::rerank::Reranker,
    ) -> anyhow::Result<Vec<HybridHit>> {
        self.search_hybrid_fused_reranked(q, limit, embedder, reranker, FusionAlgo::Rrf)
    }

    /// Hybrid then rerank (SPEC-P3 §2 + SPEC-P5 §A3): the fused top
    /// `max(limit*3, 30)` candidates (fusion algorithm `algo`) are reranked
    /// by `reranker` against doc text `path + "\n" + snippet`, reordered by
    /// rerank score (ties — and docs missing from the reranker's output —
    /// keep fused order), then truncated to `limit`. Each returned hit
    /// carries the reranker score in `rerank_score`.
    pub fn search_hybrid_fused_reranked(
        &self,
        q: &str,
        limit: usize,
        embedder: &dyn Embedder,
        reranker: &dyn indexio_embed::rerank::Reranker,
        algo: FusionAlgo,
    ) -> anyhow::Result<Vec<HybridHit>> {
        let pool = limit.saturating_mul(3).max(30);
        let fused = self.search_hybrid_fused(q, pool, embedder, algo)?;
        if fused.is_empty() {
            return Ok(fused);
        }
        let docs: Vec<String> = fused
            .iter()
            .map(|h| format!("{}\n{}", h.hit.path, h.hit.snippet))
            .collect();
        let ranked = reranker.rerank(q, &docs)?;
        let mut scores: Vec<Option<f64>> = vec![None; fused.len()];
        for (i, s) in ranked {
            if i < scores.len() {
                scores[i] = Some(s);
            }
        }
        // Reorder: scored docs by score desc (stable: RRF order breaks
        // ties), unscored docs keep RRF order after the scored ones.
        let mut order: Vec<usize> = (0..fused.len()).collect();
        order.sort_by(|&a, &b| match (scores[a], scores[b]) {
            (Some(x), Some(y)) => y.total_cmp(&x),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        });
        let mut out: Vec<HybridHit> = order
            .into_iter()
            .take(limit)
            .map(|i| {
                let mut h = fused[i].clone();
                h.rerank_score = scores[i];
                h
            })
            .collect();
        // RRF-order tiebreak already applied; truncate to limit.
        out.truncate(limit);
        Ok(out)
    }

    /// The single registered repo an explicit `repo:X` token in `q` names
    /// (case-insensitive substring, like the lexical filter; the `sessions`
    /// source counts); `None` without a token or when it matches several
    /// repos. Scopes every hybrid/semantic leg (SPEC-P10 §9), not only the
    /// lexical one.
    pub fn repo_scope(&self, q: &str) -> Option<String> {
        let want = q.split_whitespace().find_map(|t| t.strip_prefix("repo:"))?.to_ascii_lowercase();
        if want.is_empty() {
            return None;
        }
        let mut names = self.stats().repos;
        if !names.iter().any(|n| n == "sessions") {
            names.push("sessions".to_string());
        }
        let mut hits = names.into_iter().filter(|n| n.to_ascii_lowercase().contains(&want));
        let first = hits.next()?;
        if hits.next().is_some() {
            return None;
        }
        Some(first)
    }

    /// Engine stats from shard-set meta; CAS fields are 0 (CAS stats live
    /// in indexio-ingest; the CLI merges them).
    pub fn stats(&self) -> EngineStats {
        let mut repos: Vec<String> = Vec::new();
        let mut total_raw = 0u64;
        let mut index_bytes = 0u64;
        for shard in self.set.shards() {
            for r in &shard.meta().repos {
                if !repos.contains(r) {
                    repos.push(r.clone());
                }
            }
            total_raw += shard.meta().total_raw_bytes;
            index_bytes += std::fs::metadata(shard.path()).map(|m| m.len()).unwrap_or(0);
        }
        EngineStats {
            repos,
            doc_count: self.set.doc_count(),
            total_raw_bytes: total_raw,
            shard_count: self.set.len() as u64,
            cas_entries: 0,
            cas_bytes: 0,
            index_bytes,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Galloping seek-merge intersection of two sorted doc-ref lists.
fn galloping_intersect(a: &[DocRef], b: &[DocRef]) -> Vec<DocRef> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let mut out = Vec::new();
    let mut lo = 0usize; // lower bound into `long`
    for &x in short {
        // Gallop: exponential probe for the first index `hi` with
        // long[hi] >= x, then binary search the window long[lo..=hi].
        let mut step = 1usize;
        let mut hi = lo;
        while hi < long.len() && long[hi] < x {
            hi = lo + step;
            step <<= 1;
        }
        let end = if hi >= long.len() { long.len() } else { hi + 1 };
        match long[lo..end].binary_search(&x) {
            Ok(rel) => {
                out.push(x);
                lo += rel + 1;
            }
            Err(rel) => {
                lo += rel;
            }
        }
        if lo >= long.len() {
            break;
        }
    }
    out
}

/// BM25-lite term score (SPEC step 6).
fn bm25(n: f64, df: f64, tf: f64, dl: f64, avgdl: f64) -> f64 {
    let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
    let norm = tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / avgdl));
    idf * norm
}

/// Sort by score desc; stable tiebreak by (repo, path, line) asc.
fn sort_hits(hits: &mut [SearchHit]) {
    hits.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.repo.cmp(&b.repo))
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.line.cmp(&b.line))
    });
}

/// The raw text of 1-based `line` in `content` ("" when out of range).
fn content_line(content: &[u8], line: u32) -> String {
    let idx = line.saturating_sub(1) as usize;
    let mut it = content.split(|&b| b == b'\n');
    match it.nth(idx) {
        Some(bytes) => {
            let bytes = bytes.strip_suffix(b"\r").unwrap_or(bytes);
            String::from_utf8_lossy(bytes).into_owned()
        }
        None => String::new(),
    }
}

/// Snippet: the raw matching line, trimmed, capped at SNIPPET_MAX_CHARS
/// with an ellipsis.
fn snippet_line(content: &[u8], line: u32) -> String {
    let line_text = content_line(content, line);
    let trimmed = line_text.trim();
    if trimmed.chars().count() <= SNIPPET_MAX_CHARS {
        trimmed.to_string()
    } else {
        let prefix: String = trimmed.chars().take(SNIPPET_MAX_CHARS - 3).collect();
        format!("{prefix}...")
    }
}

/// Semantic snippet: up to 2 lines starting at 1-based `start_line`,
/// each trimmed, joined by a newline, capped at SNIPPET_MAX_CHARS.
fn semantic_snippet(content: &[u8], start_line: u32, end_line: u32) -> (u32, String) {
    // SPEC-P9: a chunk's first line is often a shebang, a comment banner, an
    // attribute, a decorator or a lone `}`; point the hit at the first line
    // that looks like code, scanning a few lines past the chunk end if the
    // chunk itself has none, and fall back to the first line.
    let _ = end_line;
    let first = content_line(content, start_line);
    let last = start_line.saturating_add(CHUNK_SNIPPET_SCAN);
    let mut chosen: Option<(u32, String)> = None;
    for ln in start_line..=last {
        let l = if ln == start_line { first.clone() } else { content_line(content, ln) };
        let t = l.trim();
        if t.is_empty() {
            if ln > start_line && ln as usize > content_line_count(content) {
                break;
            }
            continue;
        }
        if is_comment_like(t) {
            continue;
        }
        chosen = Some((ln, l));
        break;
    }
    let (line, l1) = chosen.unwrap_or((start_line, first));
    // Two lines: the MCP text layer shows the first, the reranker and the
    // HTTP/CLI consumers get the second for context.
    let l2 = content_line(content, line.saturating_add(1));
    let joined = if l2.trim().is_empty() {
        l1.trim().to_string()
    } else {
        format!("{}
{}", l1.trim(), l2.trim())
    };
    if joined.chars().count() <= SNIPPET_MAX_CHARS {
        (line, joined)
    } else {
        let prefix: String = joined.chars().take(SNIPPET_MAX_CHARS - 3).collect();
        (line, format!("{prefix}..."))
    }
}

/// Number of lines in `content` (a trailing newline does not add one).
fn content_line_count(content: &[u8]) -> usize {
    let body = content.strip_suffix(b"\n").unwrap_or(content);
    if body.is_empty() {
        0
    } else {
        body.iter().filter(|&&b| b == b'\n').count() + 1
    }
}

/// Lines a chunk snippet may skip past looking for code.
const CHUNK_SNIPPET_SCAN: u32 = 12;

/// Comment, doc, attribute, decorator, shebang or bracket-only lines: not
/// worth being the one line a hit shows.
fn is_comment_like(t: &str) -> bool {
    const PREFIXES: [&str; 12] = [
        "#!", "//", "/*", "*", "#[", "#![", "@", "\"\"\"", "'''", "<!--", "--", "<?",
    ];
    if PREFIXES.iter().any(|p| t.starts_with(p)) {
        return true;
    }
    // `#` comments (Python/Ruby/shell) but not C preprocessor lines.
    if t.starts_with('#') && !t.starts_with("#include") && !t.starts_with("#define") && !t.starts_with("#if") {
        return true;
    }
    t.chars().all(|c| matches!(c, '{' | '}' | '(' | ')' | '[' | ']' | ';' | ','))
}

/// Identifiers on `line` that contain `needle` as a substring.
fn identifiers_containing(line: &str, needle: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if !cur.is_empty() && cur.contains(needle) && !out.contains(cur) {
            out.push(cur.clone());
        }
        cur.clear();
    };
    for c in line.chars() {
        if c.is_alphanumeric() || c == '_' {
            cur.push(c);
        } else {
            flush(&mut cur, &mut out);
        }
    }
    flush(&mut cur, &mut out);
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use indexio_core::grams;
    use indexio_index::ShardWriter;
    use indexio_types::{BlobId, CallRec, DocMeta, ExtractedArtifact, SymbolKind, SymbolRec};

    struct Doc {
        path: &'static str,
        lang: Lang,
        content: String,
        symbols: Vec<SymbolRec>,
        calls: Vec<CallRec>,
    }

    fn sym(name: &str, kind: SymbolKind, line: u32) -> SymbolRec {
        SymbolRec {
            name: name.to_string(),
            kind,
            line,
            col: 0,
            scope: String::new(),
        }
    }

    /// Two shards: shard 1 = repo "alpha", shard 2 = repo "beta".
    fn build_engine() -> (tempfile::TempDir, Engine) {
        let tmp = tempfile::tempdir().unwrap();
        let shards = tmp.path().join("shards");
        std::fs::create_dir_all(&shards).unwrap();

        let shard1_docs = vec![
            Doc {
                path: "src/hello.rs",
                lang: Lang::Rust,
                content: "fn main() {\n    println!(\"hello world\");\n    let targettoken = 1;\n}\n"
                    .to_string(),
                symbols: vec![sym("main", SymbolKind::Fn, 1)],
                calls: vec![],
            },
            Doc {
                path: "src/regex.rs",
                lang: Lang::Rust,
                content: "this tests regex engines\nand regx too\n".to_string(),
                symbols: vec![],
                calls: vec![],
            },
            Doc {
                path: "src/foo.rs",
                lang: Lang::Rust,
                content: "fn foo_bar_123(x: i32) -> i32 {\n    helper(x)\n}\n".to_string(),
                symbols: vec![sym("foo_bar_123", SymbolKind::Fn, 1)],
                calls: vec![],
            },
            Doc {
                path: "src/caller.rs",
                lang: Lang::Rust,
                content: "fn run() {\n    foo_bar_123(3);\n}\n".to_string(),
                symbols: vec![sym("run", SymbolKind::Fn, 1)],
                calls: vec![CallRec {
                    callee: "foo_bar_123".to_string(),
                    caller: "run".to_string(),
                    line: 2,
                }],
            },
            Doc {
                path: "src/targettoken.rs",
                lang: Lang::Rust,
                content: "pub fn targettoken() {}\n".to_string(),
                symbols: vec![sym("targettoken", SymbolKind::Fn, 1)],
                calls: vec![],
            },
            Doc {
                path: "notes/xy.txt",
                lang: Lang::Unknown,
                content: "xy\nab xy cd\n".to_string(),
                symbols: vec![],
                calls: vec![],
            },
            Doc {
                path: "docs/uni.md",
                lang: Lang::Unknown,
                content: "héllo wörld — ünïcode\nhello 你好\n".to_string(),
                symbols: vec![],
                calls: vec![],
            },
        ];
        write_shard(&shards, &shard1_docs, &["alpha".to_string()]);

        let mut shard2_docs = vec![
            Doc {
                path: "src/hello_world.py",
                lang: Lang::Python,
                content: "class HelloWorld:\n    def hello_world(self):\n        pass\n".to_string(),
                symbols: vec![
                    sym("HelloWorld", SymbolKind::Class, 1),
                    sym("hello_world", SymbolKind::Method, 2),
                ],
                calls: vec![],
            },
            Doc {
                path: "src/util.go",
                lang: Lang::Go,
                content: "package util\n\nfunc hello_world() {}\n".to_string(),
                symbols: vec![sym("hello_world", SymbolKind::Fn, 3)],
                calls: vec![],
            },
        ];
        for i in 0..11 {
            shard2_docs.push(Doc {
                path: Box::leak(
                    format!("filler/f{i}.py").into_boxed_str(),
                ),
                lang: Lang::Python,
                content: format!("filler document number {i}\nnothing interesting here\n"),
                symbols: vec![],
                calls: vec![],
            });
        }
        write_shard(&shards, &shard2_docs, &["beta".to_string()]);

        let engine = Engine::open(tmp.path()).unwrap();
        (tmp, engine)
    }

    fn write_shard(dir: &Path, docs: &[Doc], repos: &[String]) {
        let mut w = ShardWriter::new(dir).unwrap();
        for d in docs {
            let content = d.content.as_bytes();
            let art = ExtractedArtifact {
                ngrams: grams::extract(content, &CommonGrams::empty()),
                symbols: d.symbols.clone(),
                calls: d.calls.clone(),
                raw_len: content.len() as u32,
                lang: d.lang,
            };
            let meta = DocMeta {
                blob: BlobId::from_content(content),
                repo_id: 0,
                path: d.path.to_string(),
                lang: d.lang,
                raw_len: content.len() as u32,
            };
            w.add_doc(&meta, content, &art).unwrap();
        }
        w.finish(repos).unwrap();
    }

    fn paths(res: &SearchResult) -> Vec<&str> {
        res.hits.iter().map(|h| h.path.as_str()).collect()
    }

    // ---- parser ----------------------------------------------------------

    #[test]
    fn parse_empty_query_errors() {
        assert!(matches!(parse(""), Err(QueryError::Empty)));
        assert!(matches!(parse("   "), Err(QueryError::Empty)));
        // filters-only is also empty
        assert!(matches!(parse("repo:alpha"), Err(QueryError::Empty)));
    }

    #[test]
    fn parse_literals_phrases_regexes_filters() {
        let q = parse("hello").unwrap();
        assert_eq!(q.literals.len(), 1);
        assert_eq!(q.literals[0].text, b"hello");
        assert!(q.literals[0].case_insensitive); // smart-case: lowercase
        assert!(!q.literals[0].phrase);

        let q = parse("Hello").unwrap();
        assert!(!q.literals[0].case_insensitive); // uppercase -> sensitive

        let q = parse("\"hello world\"").unwrap();
        assert!(q.literals[0].phrase);
        assert_eq!(q.literals[0].text, b"hello world");

        let q = parse("/rege?x/").unwrap();
        assert_eq!(q.regexes, vec!["rege?x".to_string()]);

        let q = parse("foo repo:alpha lang:rust path:src case:yes").unwrap();
        assert_eq!(q.filters.repo.as_deref(), Some("alpha"));
        assert_eq!(q.filters.lang, Some(Lang::Rust));
        assert_eq!(q.filters.path.as_deref(), Some("src"));
        assert!(!q.literals[0].case_insensitive); // case:yes forces sensitive

        let q = parse("Hello case:no").unwrap();
        assert!(q.literals[0].case_insensitive); // case:no forces insensitive

        assert!(matches!(parse("/(bad/"), Err(QueryError::InvalidRegex(_))));
    }

    // ---- search: literals, phrases, regexes ------------------------------

    #[test]
    fn search_single_literal_case_sensitive() {
        let (_tmp, e) = build_engine();
        let res = e.search(&parse("HelloWorld").unwrap(), 10);
        assert!(!res.truncated);
        assert_eq!(paths(&res), vec!["src/hello_world.py"]);
        assert_eq!(res.hits[0].line, 1);
        assert_eq!(res.hits[0].snippet, "class HelloWorld:");
    }

    #[test]
    fn search_phrase_and_snippet() {
        let (_tmp, e) = build_engine();
        let res = e.search(&parse("\"hello world\"").unwrap(), 10);
        assert_eq!(paths(&res), vec!["src/hello.rs"]);
        let h = &res.hits[0];
        assert_eq!(h.line, 2);
        assert_eq!(h.snippet, "println!(\"hello world\");");
        assert_eq!(h.repo, "alpha");
        assert_eq!(h.lang, Lang::Rust);
    }

    /// Regression (SPEC-P6): a regex's required literals are one-per-branch
    /// alternatives, so docs containing only ONE branch must still match.
    #[test]
    fn regex_alternation_prefixes_are_disjunctive() {
        let (_tmp, e) = build_engine();
        // "targettoken" is in hello.rs (as `let targettoken`) and
        // targettoken.rs (as `fn targettoken`); "HelloWorld" only in the
        // python doc. A conjunctive planner would demand both strings.
        let res = e.search(&parse("/(?:HelloWorld|targettoken)/").unwrap(), 10);
        let mut got = paths(&res);
        got.sort();
        assert_eq!(
            got,
            vec!["src/hello.rs", "src/hello_world.py", "src/targettoken.rs"]
        );
        assert!(!res.truncated);
    }

    #[test]
    fn search_regex_terms() {
        let (_tmp, e) = build_engine();
        let res = e.search(&parse("/rege?x/").unwrap(), 10);
        assert_eq!(paths(&res), vec!["src/regex.rs"]);

        let res = e.search(&parse("/fn foo_bar_123\\(/").unwrap(), 10);
        assert_eq!(paths(&res), vec!["src/foo.rs"]);
        assert_eq!(res.hits[0].snippet, "fn foo_bar_123(x: i32) -> i32 {");

        // regex whose required literal narrows candidates, verified on
        // content: only "number 10" matches `1\d` (fillers are 0..=10).
        let res = e.search(&parse("/filler document number 1\\d/").unwrap(), 10);
        assert_eq!(paths(&res), vec!["filler/f10.py"]);

        // regex with no required literal at all: bounded scan fallback
        let res = e.search(&parse("/number \\d+/").unwrap(), 20);
        assert_eq!(res.hits.len(), 11);
    }

    // ---- filters ----------------------------------------------------------

    #[test]
    fn search_repo_lang_path_filters() {
        let (_tmp, e) = build_engine();
        // "hello" ci occurs in: alpha/src/hello.rs, alpha/docs/uni.md,
        // beta/src/hello_world.py, beta/src/util.go
        let res = e.search(&parse("hello repo:alpha").unwrap(), 10);
        assert_eq!(res.hits.len(), 2);
        assert!(res.hits.iter().all(|h| h.repo == "alpha"));

        let res = e.search(&parse("hello repo:ALPHA").unwrap(), 10);
        assert_eq!(res.hits.len(), 2); // repo filter is case-insensitive

        let res = e.search(&parse("hello lang:go").unwrap(), 10);
        assert_eq!(paths(&res), vec!["src/util.go"]);

        let res = e.search(&parse("hello path:UTIL").unwrap(), 10);
        assert_eq!(paths(&res), vec!["src/util.go"]); // path filter ci

        let res = e.search(&parse("hello repo:beta path:hello").unwrap(), 10);
        assert_eq!(paths(&res), vec!["src/hello_world.py"]);
    }

    // ---- smart case --------------------------------------------------------

    #[test]
    fn smart_case_behavior() {
        let (_tmp, e) = build_engine();
        let ci = e.search(&parse("hello").unwrap(), 10);
        assert_eq!(ci.hits.len(), 4); // hello.rs, uni.md, hello_world.py, util.go

        let cs = e.search(&parse("HELLO").unwrap(), 10);
        assert_eq!(cs.hits.len(), 0);

        // case:no forces insensitive matching on an uppercase term.
        let forced_ci = e.search(&parse("HELLO case:no").unwrap(), 10);
        assert_eq!(forced_ci.hits.len(), 4);

        let ci2 = e.search(&parse("helloworld").unwrap(), 10);
        assert_eq!(paths(&ci2), vec!["src/hello_world.py"]);

        // case:yes forces sensitive matching: only exact-case bytes match.
        let forced_cs = e.search(&parse("helloworld case:yes").unwrap(), 10);
        assert_eq!(forced_cs.hits.len(), 0);
    }

    // ---- brute fallback -----------------------------------------------------

    #[test]
    fn brute_fallback_two_char_token() {
        let (_tmp, e) = build_engine();
        let res = e.search(&parse("xy").unwrap(), 10);
        assert_eq!(paths(&res), vec!["notes/xy.txt"]);
        assert!(!res.truncated);
        assert_eq!(res.hits[0].line, 1);
        assert_eq!(res.hits[0].snippet, "xy");
    }

    #[test]
    fn unicode_content_searchable() {
        let (_tmp, e) = build_engine();
        let res = e.search(&parse("你好").unwrap(), 10);
        assert_eq!(paths(&res), vec!["docs/uni.md"]);
        let res = e.search(&parse("wörld").unwrap(), 10);
        assert_eq!(paths(&res), vec!["docs/uni.md"]);
    }

    // ---- grep -n style: several lines per file ------------------------------

    #[test]
    fn search_lines_lists_every_matching_line_per_file() {
        let (_tmp, e) = build_engine();
        // `/regex|regx/` matches two lines of src/regex.rs; `search` gives one
        let q = parse("/regex|regx/").unwrap();
        let one = e.search(&q, 10);
        assert_eq!(one.hits.iter().filter(|h| h.path == "src/regex.rs").count(), 1);
        let all = e.search_lines(&q, 10, 20);
        let rows: Vec<(u32, &str)> = all
            .hits
            .iter()
            .filter(|h| h.path == "src/regex.rs")
            .map(|h| (h.line, h.snippet.as_str()))
            .collect();
        assert_eq!(rows, vec![(1, "this tests regex engines"), (2, "and regx too")]);
        // the file's lines share one score and stay adjacent
        let scores: Vec<f32> = all.hits.iter().filter(|h| h.path == "src/regex.rs").map(|h| h.score).collect();
        assert_eq!(scores[0], scores[1]);
        // per-file cap and the total limit both apply; limit sets truncated
        let capped = e.search_lines(&q, 10, 1);
        assert_eq!(capped.hits.iter().filter(|h| h.path == "src/regex.rs").count(), 1);
        let limited = e.search_lines(&parse("/hello|Hello/").unwrap(), 2, 20);
        assert_eq!(limited.hits.len(), 2);
        assert!(limited.truncated);
    }

    // ---- candidate cap ------------------------------------------------------

    #[test]
    fn candidate_cap_sets_truncated() {
        let (_tmp, e) = build_engine();
        let q = parse("hello").unwrap();
        let res = e.search_with_cap(&q, 10, 1);
        assert!(res.truncated);
        assert!(res.hits.len() <= 1);
        // With the default cap the same query is complete.
        let res = e.search(&q, 10);
        assert!(!res.truncated);
        assert_eq!(res.hits.len(), 4);
    }

    // ---- ranking ------------------------------------------------------------

    #[test]
    fn path_boosted_doc_outranks_unboosted() {
        let (_tmp, e) = build_engine();
        let res = e.search(&parse("targettoken").unwrap(), 10);
        assert_eq!(res.hits.len(), 2);
        assert_eq!(res.hits[0].path, "src/targettoken.rs"); // path + symbol boost
        assert_eq!(res.hits[1].path, "src/hello.rs");
        assert!(res.hits[0].score > res.hits[1].score);
    }

    #[test]
    fn limit_is_respected() {
        let (_tmp, e) = build_engine();
        let res = e.search(&parse("hello").unwrap(), 2);
        assert_eq!(res.hits.len(), 2);
    }

    // ---- symbols / calls ------------------------------------------------------

    #[test]
    fn find_symbol_exact() {
        let (_tmp, e) = build_engine();
        let hits = e.find_symbol("foo_bar_123", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "src/foo.rs");
        assert_eq!(hits[0].line, 1);
        assert_eq!(hits[0].score, 10.0);
        assert_eq!(hits[0].snippet, "fn foo_bar_123(x: i32) -> i32 {");

        let hits = e.find_symbol("hello_world", 10);
        assert_eq!(hits.len(), 2); // py method + go fn
        assert!(hits.iter().all(|h| h.score == 10.0));
    }

    #[test]
    fn find_symbol_substring_fallback() {
        let (_tmp, e) = build_engine();
        let hits = e.find_symbol("foo_bar", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "src/foo.rs");
        assert_eq!(hits[0].line, 1);
        assert_eq!(hits[0].score, 1.0);
    }

    #[test]
    fn who_calls_reports_callers() {
        let (_tmp, e) = build_engine();
        let hits = e.who_calls("foo_bar_123", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "src/caller.rs");
        assert_eq!(hits[0].line, 2);
        assert_eq!(hits[0].snippet, "foo_bar_123(3);");
        assert_eq!(hits[0].score, 1.0);

        assert!(e.who_calls("nonexistent", 10).is_empty());
    }

    // ---- stats ------------------------------------------------------------------

    #[test]
    fn stats_aggregates_shards() {
        let (_tmp, e) = build_engine();
        let s = e.stats();
        assert_eq!(s.shard_count, 2);
        assert_eq!(s.doc_count, 20);
        assert!(s.repos.contains(&"alpha".to_string()));
        assert!(s.repos.contains(&"beta".to_string()));
        assert_eq!(s.cas_entries, 0);
        assert_eq!(s.cas_bytes, 0);
        assert!(s.total_raw_bytes > 0);
        assert!(s.index_bytes > 0);
    }

    // ---- semantic plane (SPEC-P2 §4/§5) -----------------------------------

    use indexio_embed::embed::HashEmbedder;
    use indexio_embed::pipeline::embed_repo;

    /// 2-repo fixture with an embedded vec index: alpha = embedding/vector
    /// code, beta = database pooling code. Data dir = tmp (shards/ + vec/).
    fn build_semantic_engine() -> (tempfile::TempDir, Engine, HashEmbedder) {
        let tmp = tempfile::tempdir().unwrap();
        let shards = tmp.path().join("shards");
        std::fs::create_dir_all(&shards).unwrap();
        write_shard(
            &shards,
            &[Doc {
                path: "src/vector.rs",
                lang: Lang::Rust,
                content: "fn vector_search() {\n    // cosine similarity over embedding vectors ranks semantic neighbors\n    let embedding = embed_document(text);\n    rank_by_cosine(&embedding);\n}\n"
                    .to_string(),
                symbols: vec![sym("vector_search", SymbolKind::Fn, 1)],
                calls: vec![],
            }],
            &["alpha".to_string()],
        );
        write_shard(
            &shards,
            &[Doc {
                path: "src/pool.rs",
                lang: Lang::Rust,
                content: "fn database_pool() {\n    // postgres connection pooling with tls timeouts and retries\n    let pool = connect_postgres(dsn);\n}\n"
                    .to_string(),
                symbols: vec![sym("database_pool", SymbolKind::Fn, 1)],
                calls: vec![],
            }],
            &["beta".to_string()],
        );
        let e = HashEmbedder::new(512);
        let set = ShardSet::open_dir(&shards).unwrap();
        embed_repo(&set, tmp.path(), "alpha", &e).unwrap();
        embed_repo(&set, tmp.path(), "beta", &e).unwrap();
        let engine = Engine::open(tmp.path()).unwrap();
        (tmp, engine, e)
    }

    #[test]
    fn mode_parsing() {
        assert_eq!(SearchMode::parse("lexical").unwrap(), SearchMode::Lexical);
        assert_eq!(SearchMode::parse("semantic").unwrap(), SearchMode::Semantic);
        assert_eq!(SearchMode::parse("hybrid").unwrap(), SearchMode::Hybrid);
        assert_eq!(SearchMode::default(), SearchMode::Lexical);
        // invalid mode -> error naming the bad value
        let err = SearchMode::parse("fuzzy").unwrap_err();
        assert!(err.contains("fuzzy"), "{err}");
        assert!(SearchMode::parse("").is_err());
        assert!(SearchMode::parse("LEXICAL").is_err());
        // FromStr + as_str roundtrip
        for m in [SearchMode::Lexical, SearchMode::Semantic, SearchMode::Hybrid] {
            assert_eq!(m.as_str().parse::<SearchMode>().unwrap(), m);
        }
    }

    #[test]
    fn search_semantic_ranks_planted_chunk_first() {
        let (_tmp, e, emb) = build_semantic_engine();
        let hits = e
            .search_semantic("cosine similarity embedding vector ranking", 5, &emb)
            .unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].repo, "alpha", "{:?}", hits[0]);
        assert_eq!(hits[0].path, "src/vector.rs");
        assert_eq!(hits[0].line, 1, "chunk start_line");
        assert!(hits[0].score > 0.0);
        assert!(hits[0].snippet.contains("vector_search"), "{}", hits[0].snippet);
        // The irrelevant repo's chunk scores strictly lower.
        let beta = hits.iter().find(|h| h.repo == "beta").unwrap();
        assert!(hits[0].score > beta.score);
    }

    #[test]
    fn search_semantic_no_vec_index_is_graceful() {
        // build_engine has shards but no vec index: empty, not an error.
        let (_tmp, e) = build_engine();
        let emb = HashEmbedder::new(512);
        let hits = e.search_semantic("hello world", 5, &emb).unwrap();
        assert!(hits.is_empty());
        // Engine without a data dir at all: also empty.
        let e2 = Engine::from_shard_set(ShardSet::open_dir(&_tmp.path().join("shards")).unwrap());
        assert!(e2.search_semantic("hello", 5, &emb).unwrap().is_empty());
        // Hybrid degrades to lexical-only: sem_rank is None everywhere.
        let fused = e.search_hybrid("targettoken", 10, &emb).unwrap();
        assert!(!fused.is_empty());
        for h in &fused {
            assert!(h.sem_rank.is_none(), "{:?}", h.hit);
            assert!(h.lex_rank.is_some());
        }
        assert!(fused[0].hit.path.contains("targettoken"));
        // lexical leg only, rank 1, at the production lexical weight
        assert_eq!(fused[0].rrf, FUSE_WEIGHTS.0 / (RRF_K + 1.0));
    }

    #[test]
    fn rrf_fuse_ordering_and_single_list_survival() {
        let mk = |repo: &str, path: &str, score: f32| SearchHit {
            repo: repo.into(),
            path: path.into(),
            line: 1,
            col: 0,
            snippet: format!("snip {path}"),
            score,
            lang: Lang::Rust,
        };
        // both.rs: lex rank 1 + sem rank 2; semonly.rs: sem rank 1;
        // lexonly.rs: lex rank 2 (absent from semantic list).
        let lex = vec![mk("r", "both.rs", 9.0), mk("r", "lexonly.rs", 5.0)];
        let sem = vec![mk("r", "semonly.rs", 0.9), mk("r", "both.rs", 0.8)];
        let fused = rrf_fuse(&lex, &sem, 10);
        assert_eq!(fused.len(), 3);
        // doc in both lists ranks above any doc in a single list.
        assert_eq!(fused[0].hit.path, "both.rs");
        assert_eq!(fused[0].lex_rank, Some(1));
        assert_eq!(fused[0].sem_rank, Some(2));
        let both_expected = 1.0 / 61.0 + 1.0 / 62.0;
        assert!((fused[0].rrf - both_expected).abs() < 1e-12);
        assert!(fused[0].rrf > fused[1].rrf && fused[1].rrf >= fused[2].rrf);
        // lexical-only doc still surfaces, with sem_rank None.
        let lexonly = fused.iter().find(|h| h.hit.path == "lexonly.rs").unwrap();
        assert_eq!(lexonly.lex_rank, Some(2));
        assert_eq!(lexonly.sem_rank, None);
        assert!((lexonly.rrf - 1.0 / 62.0).abs() < 1e-12);
        // semantic-only doc surfaces with lex_rank None.
        let semonly = fused.iter().find(|h| h.hit.path == "semonly.rs").unwrap();
        assert_eq!(semonly.lex_rank, None);
        assert_eq!(semonly.sem_rank, Some(1));
        // limit truncation
        assert_eq!(rrf_fuse(&lex, &sem, 1).len(), 1);
    }

    #[test]
    fn search_hybrid_merges_lexical_and_semantic() {
        let tmp = tempfile::tempdir().unwrap();
        let shards = tmp.path().join("shards");
        std::fs::create_dir_all(&shards).unwrap();
        // alpha doc: has the rare literal AND the concept terms (both lists).
        write_shard(
            &shards,
            &[Doc {
                path: "src/hybrid.rs",
                lang: Lang::Rust,
                content: "fn hybridtoken() {\n    // embedding vector cosine ranking for semantic code search\n    hybridtoken_rank();\n}\n"
                    .to_string(),
                symbols: vec![sym("hybridtoken", SymbolKind::Fn, 1)],
                calls: vec![],
            }],
            &["alpha".to_string()],
        );
        // beta doc 1: concept terms only (semantic-only); doc 2: unrelated.
        write_shard(
            &shards,
            &[
                Doc {
                    path: "src/concepts.rs",
                    lang: Lang::Rust,
                    content: "fn semantic_ranker() {\n    // embedding vector cosine ranking over code chunks\n}\n"
                        .to_string(),
                    symbols: vec![],
                    calls: vec![],
                },
                Doc {
                    path: "src/unrelated.rs",
                    lang: Lang::Rust,
                    content: "fn unrelated() {\n    // yaml config loader for the billing service\n}\n"
                        .to_string(),
                    symbols: vec![],
                    calls: vec![],
                },
            ],
            &["beta".to_string()],
        );
        let emb = HashEmbedder::new(512);
        let set = ShardSet::open_dir(&shards).unwrap();
        embed_repo(&set, tmp.path(), "alpha", &emb).unwrap();
        embed_repo(&set, tmp.path(), "beta", &emb).unwrap();
        let engine = Engine::open(tmp.path()).unwrap();

        // Lexical parse of this query requires ALL literals -> only alpha
        // matches lexically; semantic lifts the concept doc too.
        let fused = engine
            .search_hybrid("hybridtoken embedding vector cosine ranking", 10, &emb)
            .unwrap();
        assert!(!fused.is_empty());
        let top = &fused[0];
        assert_eq!(top.hit.path, "src/hybrid.rs", "{top:?}");
        assert!(top.lex_rank.is_some() && top.sem_rank.is_some(), "{top:?}");
        // The semantic-only concept doc surfaces via the semantic list.
        let concept = fused
            .iter()
            .find(|h| h.hit.path == "src/concepts.rs")
            .expect("semantic-only doc must surface");
        assert_eq!(concept.lex_rank, None);
        assert!(concept.sem_rank.is_some());
        // rrf strictly ordered desc.
        for w in fused.windows(2) {
            assert!(w[0].rrf >= w[1].rrf, "{:?} vs {:?}", w[0].rrf, w[1].rrf);
        }
    }

    // ---- reranker stage (SPEC-P3 §2) -------------------------------------

    /// Same 2-repo fixture as `search_hybrid_merges_lexical_and_semantic`:
    /// alpha/src/hybrid.rs wins RRF (in both lists), beta/src/concepts.rs
    /// has far more query-term overlap (rerank favorite).
    fn build_rerank_fixture() -> (tempfile::TempDir, Engine, HashEmbedder) {
        let tmp = tempfile::tempdir().unwrap();
        let shards = tmp.path().join("shards");
        std::fs::create_dir_all(&shards).unwrap();
        write_shard(
            &shards,
            &[Doc {
                path: "src/hybrid.rs",
                lang: Lang::Rust,
                content: "fn hybridtoken() {\n    // embedding vector\n    // cosine ranking for semantic code search\n    hybridtoken_rank();\n}\n"
                    .to_string(),
                symbols: vec![sym("hybridtoken", SymbolKind::Fn, 1)],
                calls: vec![],
            }],
            &["alpha".to_string()],
        );
        write_shard(
            &shards,
            &[
                Doc {
                    path: "src/concepts.rs",
                    lang: Lang::Rust,
                    content: "fn semantic_ranker() {\n    // embedding vector cosine ranking over code chunks\n}\n"
                        .to_string(),
                    symbols: vec![],
                    calls: vec![],
                },
                Doc {
                    path: "src/unrelated.rs",
                    lang: Lang::Rust,
                    content: "fn unrelated() {\n    // yaml config loader for the billing service\n}\n"
                        .to_string(),
                    symbols: vec![],
                    calls: vec![],
                },
            ],
            &["beta".to_string()],
        );
        let emb = HashEmbedder::new(512);
        let set = ShardSet::open_dir(&shards).unwrap();
        embed_repo(&set, tmp.path(), "alpha", &emb).unwrap();
        embed_repo(&set, tmp.path(), "beta", &emb).unwrap();
        let engine = Engine::open(tmp.path()).unwrap();
        (tmp, engine, emb)
    }

    #[test]
    fn search_hybrid_reranked_reorders_and_fills_scores() {
        let (_tmp, engine, emb) = build_rerank_fixture();
        let q = "hybridtoken embedding vector cosine ranking";
        // Baseline RRF: the doc in both lists is #1.
        let fused = engine.search_hybrid(q, 10, &emb).unwrap();
        assert_eq!(fused[0].hit.path, "src/hybrid.rs");
        assert!(fused.iter().all(|h| h.rerank_score.is_none()));

        // With the overlap reranker, the concept doc (4 query tokens +
        // bigrams in path+snippet) overtakes the RRF #1.
        let reranker = indexio_embed::rerank::OverlapReranker;
        let reranked = engine
            .search_hybrid_reranked(q, 10, &emb, &reranker)
            .unwrap();
        assert_eq!(reranked.len(), fused.len());
        assert_eq!(
            reranked[0].hit.path, "src/concepts.rs",
            "rerank favorite must move to #1: {:?}",
            reranked.iter().map(|h| &h.hit.path).collect::<Vec<_>>()
        );
        // rerank_score filled for every hit; ordering is score desc.
        for h in &reranked {
            assert!(h.rerank_score.is_some(), "{h:?}");
        }
        for w in reranked.windows(2) {
            assert!(
                w[0].rerank_score.unwrap() >= w[1].rerank_score.unwrap(),
                "{:?} vs {:?}",
                w[0].rerank_score,
                w[1].rerank_score
            );
        }
        // The RRF #1 is still present, just demoted.
        let demoted = reranked
            .iter()
            .find(|h| h.hit.path == "src/hybrid.rs")
            .expect("RRF top doc must survive rerank");
        assert!(demoted.lex_rank.is_some());
        assert!(
            reranked[0].rerank_score.unwrap() > demoted.rerank_score.unwrap()
        );
    }

    // ---- SPEC-P5 §A3/A4: 3-leg fusion, fusion algos, query cleaning -----

    #[test]
    fn fusion_algo_parsing() {
        assert_eq!(FusionAlgo::parse("rrf").unwrap(), FusionAlgo::Rrf);
        assert_eq!(FusionAlgo::parse("combmnz").unwrap(), FusionAlgo::CombMnz);
        assert_eq!(FusionAlgo::default(), FusionAlgo::Rrf);
        let err = FusionAlgo::parse("borda").unwrap_err();
        assert!(err.contains("borda"), "{err}");
        for a in [FusionAlgo::Rrf, FusionAlgo::CombMnz] {
            assert_eq!(a.as_str().parse::<FusionAlgo>().unwrap(), a);
        }
    }

    #[test]
    fn clean_query_strips_stopwords() {
        assert_eq!(clean_query("how do I parse the config"), "parse config");
        assert_eq!(clean_query("where is the retry logic"), "retry logic");
        // Punctuation is split; case-insensitive stopword match.
        assert_eq!(clean_query("cache (with ttl)"), "cache ttl");
        // Content words survive; an all-stopword query empties out.
        assert_eq!(clean_query("login session"), "login session");
        assert_eq!(clean_query("the and of"), "");
        // Lexical-mode queries are NOT cleaned (conjunctive precision is a
        // feature): parse() keeps stopwords as literals.
        assert_eq!(parse("the").unwrap().literals.len(), 1);
    }

    /// CombMNZ on constructed lists (SPEC-P5 §A3 test): min-max
    /// normalization per leg, flat legs map to 1.0, sum x #legs (MNZ).
    #[test]
    fn combmnz_correctness_constructed_lists() {
        let mk = |path: &str, score: f32| SearchHit {
            repo: "r".into(),
            path: path.into(),
            line: 1,
            col: 0,
            snippet: format!("snip {path}"),
            score,
            lang: Lang::Rust,
        };
        // min-max per leg: lex [q=100 -> 1.0, p=50 -> 0.0];
        // bm25 [p=5 -> 1.0, r=1 -> 0.0]; sem flat single entry -> 1.0.
        let lex = vec![mk("q.rs", 100.0), mk("p.rs", 50.0)];
        let bm25 = vec![mk("p.rs", 5.0), mk("r.rs", 1.0)];
        let sem = vec![mk("q.rs", 0.42)];
        let fused = fuse_legs(
            &[(&lex, Leg::Lex), (&bm25, Leg::Bm25), (&sem, Leg::Sem)],
            FusionAlgo::CombMnz,
            10, (1.0, 1.0, 1.0));
        let get = |path: &str| fused.iter().find(|h| h.hit.path == path).unwrap().clone();
        // q.rs: (1.0 + 1.0) * 2 legs = 4.0; p.rs: (0.0 + 1.0) * 2 = 2.0;
        // r.rs: 0.0 * 1 = 0.0.
        assert!((get("q.rs").rrf - 4.0).abs() < 1e-9, "{:?}", get("q.rs"));
        assert!((get("p.rs").rrf - 2.0).abs() < 1e-9, "{:?}", get("p.rs"));
        assert!(get("r.rs").rrf.abs() < 1e-9, "{:?}", get("r.rs"));
        assert_eq!(get("p.rs").bm25_rank, Some(1));
        assert_eq!(get("p.rs").lex_rank, Some(2));
        assert_eq!(get("q.rs").sem_rank, Some(1));
        // Order: q (4.0) > p (2.0) > r (0.0).
        let order: Vec<&str> = fused.iter().map(|h| h.hit.path.as_str()).collect();
        assert_eq!(order, ["q.rs", "p.rs", "r.rs"]);
    }

    /// 3-leg fusion on constructed lists (SPEC-P5 §A3 test): a doc present
    /// ONLY in the bm25 leg still surfaces, and a doc in all 3 legs beats a
    /// doc in fewer legs.
    #[test]
    fn three_legs_bm25_only_surfaces_and_multi_leg_wins() {
        let mk = |path: &str, score: f32| SearchHit {
            repo: "r".into(),
            path: path.into(),
            line: 1,
            col: 0,
            snippet: format!("snip {path}"),
            score,
            lang: Lang::Rust,
        };
        let lex = vec![mk("three.rs", 9.0), mk("one.rs", 8.0)];
        let bm25 = vec![mk("three.rs", 7.0), mk("bmonly.rs", 6.0)];
        let sem = vec![mk("three.rs", 0.9), mk("one.rs", 0.8)];
        let fused = fuse_legs(
            &[(&lex, Leg::Lex), (&bm25, Leg::Bm25), (&sem, Leg::Sem)],
            FusionAlgo::Rrf,
            10, (1.0, 1.0, 1.0));
        // bm25-only doc surfaces with only bm25_rank set.
        let bm = fused.iter().find(|h| h.hit.path == "bmonly.rs").unwrap();
        assert_eq!(bm.bm25_rank, Some(2));
        assert_eq!(bm.lex_rank, None);
        assert_eq!(bm.sem_rank, None);
        assert!((bm.rrf - 1.0 / (RRF_K + 2.0)).abs() < 1e-12);
        // 3-leg doc beats every 1- or 2-leg doc.
        assert_eq!(fused[0].hit.path, "three.rs");
        assert_eq!(fused[0].lex_rank, Some(1));
        assert_eq!(fused[0].bm25_rank, Some(1));
        assert_eq!(fused[0].sem_rank, Some(1));
        let expected = 3.0 / (RRF_K + 1.0);
        assert!((fused[0].rrf - expected).abs() < 1e-12);
        let one = fused.iter().find(|h| h.hit.path == "one.rs").unwrap();
        assert!(fused[0].rrf > one.rrf, "3 legs > 2 legs");
    }

    /// Engine-level 3-leg test (SPEC-P5 §A3): the disjunctive BM25 leg
    /// surfaces a doc the conjunctive lexical leg rejects (missing one
    /// literal); the doc matching all legs wins; an unrelated doc rides
    /// only the semantic leg. Runs under both fusion algos.
    #[test]
    fn search_hybrid_fused_three_legs_end_to_end() {
        let tmp = tempfile::tempdir().unwrap();
        let shards = tmp.path().join("shards");
        std::fs::create_dir_all(&shards).unwrap();
        let doc = |path: &'static str, content: &str| Doc {
            path,
            lang: Lang::Rust,
            content: content.to_string(),
            symbols: vec![],
            calls: vec![],
        };
        write_shard(
            &shards,
            &[
                doc(
                    "src/three.rs",
                    "fn triword() {\n    // embedding vector cosine ranking over code chunks\n    triword_run();\n}\n",
                ),
                doc(
                    "src/bmtwo.rs",
                    "fn partial() {\n    // embedding vector cosine over chunks\n}\n",
                ),
                doc(
                    "src/unrel.rs",
                    "fn unrelated() {\n    // yaml config loader for the billing service\n}\n",
                ),
            ],
            &["alpha".to_string()],
        );
        // Fillers with disjoint vocab so df-1 query terms stay under the
        // 40% hub cut and the semantic leg has some spread.
        write_shard(
            &shards,
            &[
                doc("src/f1.rs", "fn f1() {\n    quux corge grault\n}\n"),
                doc("src/f2.rs", "fn f2() {\n    plugh xyzzy thud\n}\n"),
                doc("src/f3.rs", "fn f3() {\n    waldo fred garply\n}\n"),
            ],
            &["beta".to_string()],
        );
        let emb = HashEmbedder::new(512);
        let set = ShardSet::open_dir(&shards).unwrap();
        embed_repo(&set, tmp.path(), "alpha", &emb).unwrap();
        embed_repo(&set, tmp.path(), "beta", &emb).unwrap();
        let engine = Engine::open(tmp.path()).unwrap();

        let q = "triword embedding vector cosine ranking";
        for algo in [FusionAlgo::Rrf, FusionAlgo::CombMnz] {
            let fused = engine
                .search_hybrid_fused(q, 10, &emb, algo)
                .unwrap_or_else(|e| panic!("{algo:?}: {e}"));
            assert!(!fused.is_empty(), "{algo:?}");
            // Doc matching all 3 legs wins.
            let top = &fused[0];
            assert_eq!(top.hit.path, "src/three.rs", "{algo:?} {top:?}");
            assert!(
                top.lex_rank.is_some() && top.bm25_rank.is_some() && top.sem_rank.is_some(),
                "{algo:?} {top:?}"
            );
            // The doc the conjunctive lexical leg rejects (no triword, no
            // ranking) surfaces via the disjunctive BM25 leg.
            let partial = fused
                .iter()
                .find(|h| h.hit.path == "src/bmtwo.rs")
                .unwrap_or_else(|| panic!("{algo:?}: bm25 leg doc must surface"));
            assert_eq!(partial.lex_rank, None, "{algo:?} {partial:?}");
            assert!(partial.bm25_rank.is_some(), "{algo:?} {partial:?}");
            // The unrelated doc rides the semantic leg only.
            let unrel = fused
                .iter()
                .find(|h| h.hit.path == "src/unrel.rs")
                .unwrap_or_else(|| panic!("{algo:?}: semantic-only doc"));
            assert_eq!(unrel.lex_rank, None, "{algo:?}");
            assert_eq!(unrel.bm25_rank, None, "{algo:?}");
            assert!(unrel.sem_rank.is_some(), "{algo:?}");
            assert!(top.rrf > unrel.rrf, "{algo:?}: 3 legs > 1 leg");
        }

        // search_hybrid delegates to Rrf 3-leg fusion.
        let delegated = engine.search_hybrid(q, 10, &emb).unwrap();
        let direct = engine
            .search_hybrid_fused(q, 10, &emb, FusionAlgo::Rrf)
            .unwrap();
        let a: Vec<&str> = delegated.iter().map(|h| h.hit.path.as_str()).collect();
        let b: Vec<&str> = direct.iter().map(|h| h.hit.path.as_str()).collect();
        assert_eq!(a, b, "search_hybrid must delegate to Rrf fusion");

        // Graceful degradation: no data dir -> bm25/sem legs absent, but
        // the lexical leg still fuses.
        let e2 = Engine::from_shard_set(ShardSet::open_dir(&shards).unwrap());
        let fused = e2.search_hybrid_fused(q, 10, &emb, FusionAlgo::Rrf).unwrap();
        assert!(!fused.is_empty());
        assert!(fused.iter().all(|h| h.bm25_rank.is_none() && h.sem_rank.is_none()));
        assert_eq!(fused[0].hit.path, "src/three.rs");
    }

    #[test]
    fn search_hybrid_reranked_noop_keeps_rrf_order_and_respects_limit() {
        let (_tmp, engine, emb) = build_rerank_fixture();
        let q = "hybridtoken embedding vector cosine ranking";
        let fused = engine.search_hybrid(q, 10, &emb).unwrap();
        let noop = indexio_embed::rerank::NoopReranker;
        let reranked = engine
            .search_hybrid_reranked(q, 10, &emb, &noop)
            .unwrap();
        let a: Vec<&str> = fused.iter().map(|h| h.hit.path.as_str()).collect();
        let b: Vec<&str> = reranked.iter().map(|h| h.hit.path.as_str()).collect();
        assert_eq!(a, b, "noop reranker must preserve RRF order");
        // limit honored: pool is max(3*limit, 30) but output <= limit.
        let reranked1 = engine
            .search_hybrid_reranked(q, 1, &emb, &noop)
            .unwrap();
        assert_eq!(reranked1.len(), 1);
        assert_eq!(reranked1[0].hit.path, "src/hybrid.rs");
        assert!(reranked1[0].rerank_score.is_some());
    }
}
