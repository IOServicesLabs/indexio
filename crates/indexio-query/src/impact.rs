//! Impact analysis + agent context economy (SPEC-P6).
//!
//! Everything here is computed at query time from what the shards already
//! hold — symbol postings, name-based call postings, and file content — so
//! no shard format change and no persistent AST (cost-ladder band 1×).
//!
//!   * `impact_symbols`: BFS over the reverse call graph (callee → call
//!     sites → their enclosing functions → ...), across every repo.
//!   * `parse_unified_diff` + `changed_symbols` + `impact_diff`: map a patch
//!     (working-tree diff, PR diff) to the definitions it touches, then walk
//!     their blast radius; changed files themselves are excluded.
//!   * `who_imports`: grep-grade file-level dependents from import syntax.
//!   * `outline` / `read_span`: the two calls that let an agent avoid
//!     reading whole files.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use indexio_symbols::{enclosing, outline as ts_outline, OutlineItem};
use indexio_types::{DocMeta, Lang, SearchHit, SymbolKind};

use crate::{content_line, snippet_line, sort_hits, DocRef, Engine, Filters, Literal, Query};

/// Hard cap on lines returned by `read_span` (SPEC-P6 §2.4).
pub const MAX_SPAN_LINES: u32 = 400;
/// Changed lines examined per diff range when mapping to definitions.
const MAX_RANGE_LINES: u32 = 5_000;
/// Direct call postings of a root judged for plausibility, as a multiple of
/// `max_fanout` (hub names have thousands; the rest is sampled as before).
const PLAUSIBILITY_BUDGET: usize = 5;

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

/// Walk limits (SPEC-P6 §2.1).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ImpactOptions {
    /// Hops of the reverse call graph to follow (1 = direct callers).
    pub depth: u32,
    /// Total call sites collected before the walk stops (`truncated`).
    pub max_sites: usize,
    /// Call sites consumed per symbol name (hub names like `new`, `get`).
    pub max_fanout: usize,
}

impl Default for ImpactOptions {
    fn default() -> Self {
        ImpactOptions {
            depth: 2,
            max_sites: 500,
            max_fanout: 200,
        }
    }
}

/// One call site reached by the walk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImpactSite {
    pub repo: String,
    pub path: String,
    pub line: u32,
    pub lang: Lang,
    /// Callee name matched at this hop (name-based edge: `Foo::new` and
    /// `Bar::new` both resolve to `new`).
    pub symbol: String,
    /// Enclosing function of the call site; "" at top level.
    pub caller: String,
    /// 1 = direct caller of a root symbol.
    pub depth: u32,
    pub snippet: String,
}

/// Per-file aggregate of the walk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileImpact {
    pub repo: String,
    pub path: String,
    pub sites: u32,
    pub min_depth: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ImpactReport {
    pub roots: Vec<String>,
    pub definitions: Vec<SearchHit>,
    pub sites: Vec<ImpactSite>,
    pub files: Vec<FileImpact>,
    /// File-level dependents (`who_imports`); populated by `impact_diff`
    /// and `impact_file`.
    pub importers: Vec<SearchHit>,
    pub truncated: bool,
    pub took_ms: u64,
}

/// A definition touched by a diff (SPEC-P6 §2.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangedSymbol {
    pub path: String,
    pub name: String,
    pub kind: SymbolKind,
    pub scope: String,
    /// Changed lines that fell inside this definition (capped at 20).
    pub lines: Vec<u32>,
    /// First line of the definition (its signature), for the call-site
    /// judgement of §20 (`def search(self, ..)` takes a receiver).
    #[serde(default)]
    pub signature: String,
}

/// One file of a unified diff.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileChange {
    /// Post-change path (pre-change path for deletions).
    pub path: String,
    /// Pre-change path when it differs (renames); None otherwise.
    pub old_path: Option<String>,
    /// 1-based inclusive line ranges on the pre-change side.
    pub old_ranges: Vec<(u32, u32)>,
    /// 1-based inclusive line ranges on the post-change side.
    pub new_ranges: Vec<(u32, u32)>,
    pub added: bool,
    pub deleted: bool,
}

// ---------------------------------------------------------------------------
// Unified diff parsing (pure)
// ---------------------------------------------------------------------------

fn strip_diff_prefix(p: &str) -> Option<String> {
    let p = p.trim();
    // `--- a/x` / `+++ b/x`; tabs may follow the path (timestamps).
    let p = p.split('\t').next().unwrap_or(p).trim();
    if p == "/dev/null" {
        return None;
    }
    let p = p
        .strip_prefix("a/")
        .or_else(|| p.strip_prefix("b/"))
        .unwrap_or(p);
    let p = p.trim_matches('"');
    if p.is_empty() {
        None
    } else {
        Some(p.to_string())
    }
}

/// `-a,b` / `+c,d` → (start, len); a missing `,len` means 1.
fn parse_hunk_side(s: &str) -> Option<(u32, u32)> {
    let s = s.trim_start_matches(['-', '+']);
    let mut it = s.split(',');
    let start: u32 = it.next()?.trim().parse().ok()?;
    let len: u32 = match it.next() {
        Some(n) => n.trim().parse().ok()?,
        None => 1,
    };
    Some((start, len))
}

/// Parse a unified diff (git or plain) into per-file changed line ranges.
/// Zero-length hunks: a pure insertion has no pre-change lines (skipped on
/// the old side); a pure deletion maps to the single post-change line it
/// sits on (`c.max(1)`), so an enclosing definition is still found.
pub fn parse_unified_diff(diff: &str) -> Vec<FileChange> {
    let mut out: Vec<FileChange> = Vec::new();
    let mut cur: Option<FileChange> = None;
    let mut pending_old: Option<Option<String>> = None; // Some(None) = /dev/null

    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("--- ") {
            if let Some(c) = cur.take() {
                out.push(c);
            }
            pending_old = Some(strip_diff_prefix(rest));
            continue;
        }
        if let Some(rest) = line.strip_prefix("+++ ") {
            let new_path = strip_diff_prefix(rest);
            let old_path = pending_old.take().unwrap_or(None);
            let (path, added, deleted, old_path) = match (old_path, new_path) {
                (None, Some(n)) => (n, true, false, None),
                (Some(o), None) => (o, false, true, None),
                (Some(o), Some(n)) => {
                    let renamed = o != n;
                    (n, false, false, if renamed { Some(o) } else { None })
                }
                (None, None) => continue,
            };
            cur = Some(FileChange {
                path,
                old_path,
                added,
                deleted,
                ..Default::default()
            });
            continue;
        }
        if let Some(rest) = line.strip_prefix("@@ ") {
            let Some(c) = cur.as_mut() else { continue };
            let mut parts = rest.split_whitespace();
            let (Some(old), Some(new)) = (parts.next(), parts.next()) else {
                continue;
            };
            if let Some((a, b)) = parse_hunk_side(old) {
                if b > 0 {
                    c.old_ranges.push((a.max(1), a.max(1) + b - 1));
                }
            }
            if let Some((cc, d)) = parse_hunk_side(new) {
                if d > 0 {
                    c.new_ranges.push((cc.max(1), cc.max(1) + d - 1));
                } else {
                    c.new_ranges.push((cc.max(1), cc.max(1)));
                }
            }
        }
    }
    if let Some(c) = cur.take() {
        out.push(c);
    }
    out
}

// ---------------------------------------------------------------------------
// Import patterns (SPEC-P6 §2.3)
// ---------------------------------------------------------------------------

fn file_stem(path: &str) -> &str {
    let base = path.rsplit('/').next().unwrap_or(path);
    match base.rfind('.') {
        Some(i) if i > 0 => &base[..i],
        _ => base,
    }
}

fn parent_dir_name(path: &str) -> Option<&str> {
    let dir = path.rsplit_once('/')?.0;
    Some(dir.rsplit('/').next().unwrap_or(dir)).filter(|d| !d.is_empty())
}

fn ident_like(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
}

/// (module key, regex) for "files that import `path`", or None when the
/// file has no import identity we can grep for (`main.rs`, root Go file).
pub fn import_pattern(lang: Lang, path: &str) -> Option<(String, String)> {
    let stem = file_stem(path);
    let key: String = match lang {
        Lang::Rust => match stem {
            "mod" | "lib" => parent_dir_name(path)?.to_string(),
            "main" | "build" => return None,
            s => s.to_string(),
        },
        Lang::Python => match stem {
            "__init__" | "__main__" => parent_dir_name(path)?.to_string(),
            s => s.to_string(),
        },
        Lang::Go => parent_dir_name(path)?.to_string(),
        Lang::TsJs => match stem {
            "index" => parent_dir_name(path)?.to_string(),
            s => s.to_string(),
        },
        Lang::Java | Lang::Cpp => stem.to_string(),
        Lang::Unknown | Lang::Text => return None,
    };
    if !ident_like(&key) {
        return None;
    }
    let k = regex::escape(&key);
    let pat = match lang {
        Lang::Rust => format!(r"\b(?:mod|use)\b[^;]*\b{k}\b"),
        Lang::Python => format!(r"(?:from|import)\s+[\w.]*\b{k}\b"),
        Lang::Go => format!(r#""[^"\n]*/{k}""#),
        Lang::TsJs => {
            format!(r#"(?:from|import|require\()\s*['"][^'"\n]*/{k}(?:\.[cm]?[jt]sx?)?['"]"#)
        }
        Lang::Java => format!(r"import\s+(?:static\s+)?[\w.]*\b{k}\b"),
        Lang::Cpp => format!(r#"#include\s*["<][^">\n]*\b{k}\.h"#),
        Lang::Unknown | Lang::Text => return None,
    };
    Some((key, pat))
}

// ---------------------------------------------------------------------------
// Engine methods
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Call-site plausibility (SPEC-P9 §20)
// ---------------------------------------------------------------------------

/// What the definitions of a root name look like, so a name-based call
/// edge can be judged by its qualifier: `tokio::spawn(` cannot be the
/// repo's `net::spawn`, `cmd.spawn()` cannot be a free function.
#[derive(Debug, Default)]
pub(crate) struct RootShape {
    /// Type/container names owning a method or associated definition
    /// (`Client::spawn` → "Client"), plus module names owning free ones
    /// (`net/mod.rs` → "net", `executor.rs` → "executor").
    owners: HashSet<String>,
    /// Some definition takes a receiver (`&self`, `self`, `cls`, or is a
    /// method in a language where every method does): callable as `x.name()`.
    has_receiver: bool,
    /// No definition known — judge nothing.
    unknown: bool,
}

/// Whether a definition line declares a receiver parameter.
fn takes_receiver(lang: Lang, def_line: &str, scoped: bool) -> bool {
    let params = def_line.find('(').map(|i| def_line[i + 1..].trim_start()).unwrap_or("");
    match lang {
        Lang::Rust => {
            params.starts_with("&self")
                || params.starts_with("&mut self")
                || params.starts_with("self")
                || params.starts_with("mut self")
        }
        Lang::Python => params.starts_with("self") || params.starts_with("cls"),
        // Go: `func (r *T) name(`; JS/TS/Java/C#…: any scoped definition is a method
        _ => scoped || def_line.trim_start().starts_with("func ("),
    }
}

impl RootShape {
    fn add_definition(&mut self, name: &str, path: &str, lang: Lang, scope: &str, def_line: &str) {
        // Rust method scopes are "Type::method" (the name itself is the
        // last segment); Python/TS/Java scopes are the container chain.
        let scoped = !scope.is_empty();
        if scoped {
            if let Some(owner) = scope.split("::").filter(|s| !s.is_empty() && *s != name).last() {
                self.owners.insert(owner.to_string());
            }
        }
        if takes_receiver(lang, def_line, scoped) {
            self.has_receiver = true;
        }
        // module-style qualifiers for free functions and for Python/JS
        // modules (`net.spawn(`, `executor::spawn(`)
        let stem = file_stem(path);
        if matches!(stem, "mod" | "lib" | "main" | "__init__" | "index") {
            if let Some(d) = parent_dir_name(path) {
                self.owners.insert(d.to_string());
            }
        } else {
            self.owners.insert(stem.to_string());
        }
    }
}

/// Whether a call site line can refer to one of the root's definitions.
/// Bare calls and unresolvable receivers pass; only a qualifier that names
/// something the repo's definitions are not is rejected.
pub(crate) fn site_plausible(line: &str, name: &str, shape: &RootShape) -> bool {
    if shape.unknown || name.is_empty() {
        return true;
    }
    let bytes = line.as_bytes();
    let mut from = 0;
    let mut saw_call = false;
    while let Some(off) = line[from..].find(name) {
        let start = from + off;
        let end = start + name.len();
        from = end;
        let before_ok = start == 0 || !(bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_');
        let after = &line[end..];
        let is_call = after.starts_with('(') || after.starts_with("::<") || after.starts_with("!(");
        if !before_ok || !is_call {
            continue;
        }
        saw_call = true;
        let before = &line[..start];
        if let Some(q) = before.strip_suffix("::") {
            let qual = trailing_ident(q);
            if matches!(qual, "self" | "super" | "crate" | "Self" | "") || shape.owners.contains(qual) {
                return true;
            }
            continue;
        }
        let recv = before
            .strip_suffix("?.")
            .or_else(|| before.strip_suffix("->"))
            .or_else(|| before.strip_suffix('.'));
        if let Some(r) = recv {
            let ident = trailing_ident(r);
            if ident.is_empty() || shape.owners.contains(ident) {
                return true;
            }
            // A receiver that is visibly not an instance of any owner:
            // a library module (`re.search`, `tokio::task`), a module-level
            // constant (`YEAR_RE.search`), or a type / constructor that is
            // not an owner (`OpenAlex().search`, `Engine.search(eng, ..)`).
            if is_module_like(ident) || is_constant_like(ident) || (is_type_like(ident) && !shape.owners.contains(ident)) {
                continue;
            }
            if shape.has_receiver {
                return true;
            }
            continue;
        }
        return true; // bare call
    }
    !saw_call
}

/// The identifier a string ends with, looking through trailing call or
/// index groups: `"tokio::task"` → "task", `"OpenAlex()"` → "OpenAlex",
/// `"_RSS_FIELD[\"title\"]"` → "_RSS_FIELD".
fn trailing_ident(s: &str) -> &str {
    let mut end = s.trim_end();
    loop {
        let Some(last) = end.chars().last() else { break };
        let open = match last {
            ')' => '(',
            ']' => '[',
            _ => break,
        };
        // find the matching opener (no string awareness; good enough for a line)
        let mut depth = 0i32;
        let mut cut = None;
        for (i, c) in end.char_indices().rev() {
            if c == last {
                depth += 1;
            } else if c == open {
                depth -= 1;
                if depth == 0 {
                    cut = Some(i);
                    break;
                }
            }
        }
        match cut {
            Some(i) => end = end[..i].trim_end(),
            None => break,
        }
    }
    let start = end
        .char_indices()
        .rev()
        .take_while(|(_, c)| c.is_alphanumeric() || *c == '_')
        .last()
        .map(|(i, _)| i)
        .unwrap_or(end.len());
    &end[start..]
}

/// Library modules that are receivers of method-looking calls (`re.search`,
/// `os.path.join`, `JSON.parse`, `tokio::spawn`): never an instance of a
/// repo type.
fn is_module_like(ident: &str) -> bool {
    matches!(
        ident,
        "re" | "os" | "sys" | "json" | "time" | "random" | "subprocess" | "asyncio" | "logging"
            | "math" | "shutil" | "pathlib" | "itertools" | "functools" | "datetime" | "collections"
            | "typing" | "string" | "struct" | "socket" | "hashlib" | "base64" | "urllib" | "http"
            | "threading" | "queue" | "glob" | "io" | "copy" | "pickle" | "uuid" | "tempfile"
            | "np" | "pd" | "plt" | "torch" | "tf" | "httpx" | "requests" | "aiohttp"
            | "std" | "core" | "alloc" | "tokio" | "serde_json" | "serde" | "anyhow" | "regex"
            | "JSON" | "Math" | "Object" | "Array" | "Promise" | "Number" | "String" | "Date"
            | "console" | "document" | "window" | "process" | "fs" | "path" | "os_" | "util"
            | "fmt" | "strings" | "strconv" | "bytes" | "errors" | "context" | "sync" | "log"
    )
}

/// `YEAR_RE`, `_TITLE_RE`, `MAX_N`: a module-level constant.
fn is_constant_like(ident: &str) -> bool {
    ident.len() > 1
        && ident.chars().any(|c| c.is_ascii_uppercase())
        && ident.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// `OpenAlex`, `Engine`, `HttpClient`: a type name (capitalised, mixed case).
fn is_type_like(ident: &str) -> bool {
    let mut chars = ident.chars();
    chars.next().map_or(false, |c| c.is_ascii_uppercase()) && ident.chars().any(|c| c.is_ascii_lowercase())
}

/// Per-walk decompressed-content cache: many sites land in the same file.
struct DocCache<'a> {
    engine: &'a Engine,
    docs: HashMap<DocRef, Option<(DocMeta, String, Vec<u8>)>>,
}

impl<'a> DocCache<'a> {
    fn new(engine: &'a Engine) -> Self {
        DocCache {
            engine,
            docs: HashMap::new(),
        }
    }

    fn get(&mut self, si: usize, docid: u32) -> Option<&(DocMeta, String, Vec<u8>)> {
        let engine = self.engine;
        self.docs
            .entry((si, docid))
            .or_insert_with(|| {
                let dm = engine.set.doc(si, docid)?;
                let content = engine.set.content(si, docid).ok()?;
                let repo = engine.repo_name(si, &dm);
                Some((dm, repo, content))
            })
            .as_ref()
    }
}

impl Engine {
    /// Definition shapes of the roots (see `RootShape`).
    fn root_shapes(
        &self,
        roots: &[String],
        visible: &HashSet<DocRef>,
        cache: &mut DocCache<'_>,
    ) -> HashMap<String, RootShape> {
        let mut out = HashMap::new();
        for n in roots {
            let mut shape = RootShape::default();
            let mut any = false;
            for (si, docid, _kind, line, scope) in self.set.symbol_postings_visible(n, visible) {
                let Some((dm, _, content)) = cache.get(si, docid) else { continue };
                any = true;
                let def_line = content_line(content, line);
                shape.add_definition(n, &dm.path, dm.lang, &scope, &def_line);
            }
            shape.unknown = !any;
            out.insert(n.clone(), shape);
        }
        out
    }

    /// Shape of one name's definitions across every shard (for `who_calls`;
    /// tombstoned definitions may contribute an owner, which is harmless).
    pub(crate) fn root_shape(&self, name: &str) -> RootShape {
        let mut shape = RootShape::default();
        let mut any = false;
        for (si, docid, _kind, line, scope) in self.set.symbol_postings(name) {
            let Some(dm) = self.set.doc(si, docid) else { continue };
            let Some(content) = self.cached_content(si, docid) else { continue };
            any = true;
            shape.add_definition(name, &dm.path, dm.lang, &scope, &content_line(&content, line));
        }
        shape.unknown = !any;
        shape
    }

    /// Transitive reverse call graph from `names` (SPEC-P6 §2.1).
    pub fn impact_symbols(&self, names: &[String], opts: &ImpactOptions) -> ImpactReport {
        let t0 = Instant::now();
        let roots = dedup_names(names);
        let visible = self.set.visible_ids();
        let mut definitions = Vec::new();
        for n in &roots {
            for (si, docid, _kind, line, _scope) in self.set.symbol_postings_visible(n, &visible) {
                if let Some(h) = self.posting_hit(si, docid, line, 10.0) {
                    definitions.push(h);
                }
            }
        }
        sort_hits(&mut definitions);
        let (sites, truncated) = self.walk(&roots, opts, &visible, &HashSet::new(), HashMap::new());
        let mut report = finish_report(roots, definitions, sites, truncated);
        report.took_ms = t0.elapsed().as_millis() as u64;
        report
    }

    /// BFS core. `exclude` = (repo, path) pairs never reported as sites
    /// (the changed files themselves in `impact_diff`).
    fn walk(
        &self,
        roots: &[String],
        opts: &ImpactOptions,
        visible: &HashSet<DocRef>,
        exclude: &HashSet<(String, String)>,
        given: HashMap<String, RootShape>,
    ) -> (Vec<ImpactSite>, bool) {
        let mut cache = DocCache::new(self);
        // shapes handed in (a diff knows exactly which definitions changed)
        // win over the every-definition-of-that-name view
        let missing: Vec<String> = roots.iter().filter(|r| !given.contains_key(*r)).cloned().collect();
        let mut shapes = self.root_shapes(&missing, visible, &mut cache);
        shapes.extend(given);
        let mut visited: HashSet<String> = roots.iter().cloned().collect();
        let mut frontier: Vec<String> = roots.to_vec();
        let mut sites: Vec<ImpactSite> = Vec::new();
        let mut truncated = false;
        let mut depth = 0u32;
        'outer: while !frontier.is_empty() && depth < opts.depth {
            depth += 1;
            let mut next: Vec<String> = Vec::new();
            for name in &frontier {
                let mut postings = self.set.call_postings_visible(name, visible);
                // Direct sites of a root: drop the ones whose qualifier
                // says they are someone else's `name` (`tokio::spawn`,
                // `cmd.spawn()` for a free fn), before the hub sampling.
                if let Some(shape) = shapes.get(name).filter(|s| !s.unknown) {
                    let budget = opts.max_fanout.saturating_mul(PLAUSIBILITY_BUDGET);
                    if postings.len() > budget {
                        truncated = true;
                        postings.truncate(budget);
                    }
                    postings.retain(|(si, docid, _, line)| {
                        cache
                            .get(*si, *docid)
                            .map_or(false, |(_, _, content)| site_plausible(&content_line(content, *line), name, shape))
                    });
                }
                // Hub protection: over the per-symbol cap, sample the
                // posting list evenly (every k-th entry) instead of taking a
                // shard-order prefix, so every repo/file keeps representation.
                let stride = if postings.len() > opts.max_fanout {
                    truncated = true;
                    postings.len().div_ceil(opts.max_fanout)
                } else {
                    1
                };
                for (si, docid, caller, line) in
                    postings.into_iter().step_by(stride).take(opts.max_fanout)
                {
                    if sites.len() >= opts.max_sites {
                        truncated = true;
                        break 'outer;
                    }
                    let Some((dm, repo, content)) = cache.get(si, docid) else {
                        continue;
                    };
                    // Excluded files (the diff's own files) are not reported
                    // as sites, but their enclosing functions still propagate:
                    // a caller inside a changed file has callers of its own.
                    if !exclude.contains(&(repo.clone(), dm.path.clone())) {
                        sites.push(ImpactSite {
                            repo: repo.clone(),
                            path: dm.path.clone(),
                            line,
                            lang: dm.lang,
                            symbol: name.clone(),
                            caller: caller.clone(),
                            depth,
                            snippet: snippet_line(content, line),
                        });
                    }
                    if !caller.is_empty() && caller != *name && visited.insert(caller.clone()) {
                        next.push(caller);
                    }
                }
            }
            frontier = next;
        }
        (sites, truncated)
    }

    /// Grep-grade file-level dependents of `path` (SPEC-P6 §2.3): one
    /// lexical regex search, `lang:` filtered, module key as a required
    /// literal so the planner intersects on it. The file itself is excluded.
    pub fn who_imports(&self, repo: &str, path: &str, limit: usize) -> Vec<SearchHit> {
        let lang = Lang::from_path(path);
        let Some((key, pat)) = import_pattern(lang, path) else {
            return Vec::new();
        };
        let q = Query {
            literals: vec![Literal {
                text: key.into_bytes(),
                phrase: false,
                case_insensitive: false,
            }],
            regexes: vec![pat],
            filters: Filters {
                repo: None,
                lang: Some(lang),
                path: None,
            },
        };
        let mut hits = self.search(&q, limit.saturating_add(1)).hits;
        hits.retain(|h| !(h.repo == repo && h.path == path));
        hits.truncate(limit);
        hits
    }

    /// Definitions with ranges for one indexed file (SPEC-P6 §2.4).
    pub fn outline(&self, repo: &str, path: &str) -> Option<Vec<OutlineItem>> {
        self.outline_cached(repo, path).map(|o| (*o).clone())
    }

    /// [`outline`](Self::outline) through the per-engine cache (SPEC-P9).
    pub fn outline_cached(&self, repo: &str, path: &str) -> Option<std::sync::Arc<Vec<OutlineItem>>> {
        const CAP: usize = 512;
        let (si, docid) = self.locate_doc(repo, path)?;
        let dm = self.set.doc(si, docid)?;
        let key = (dm.blob, dm.lang);
        if let Some(o) = self.outline_cache.lock().ok()?.get(&key) {
            return Some(std::sync::Arc::clone(o));
        }
        let content = self.set.content(si, docid).ok()?;
        let items = std::sync::Arc::new(ts_outline(dm.lang, &content));
        if let Ok(mut cache) = self.outline_cache.lock() {
            if cache.len() >= CAP {
                cache.clear();
            }
            cache.insert(key, std::sync::Arc::clone(&items));
        }
        Some(items)
    }

    /// `(start, end)` of the innermost definition containing `line`
    /// (SPEC-P9): what `read_span` returns when no `end` is given and what
    /// `find_symbol` shows next to a hit. `None` outside any definition.
    pub fn definition_range(&self, repo: &str, path: &str, line: u32) -> Option<(u32, u32)> {
        let items = self.outline_cached(repo, path)?;
        items
            .iter()
            .filter(|it| it.start_line <= line && line <= it.end_line)
            .max_by_key(|it| (it.start_line, std::cmp::Reverse(it.end_line)))
            .map(|it| (it.start_line, it.end_line))
    }

    /// The raw bytes of one indexed file (`None` when it is not indexed):
    /// what a whole-file read would have returned (SPEC-P10 §22).
    pub fn file_content(&self, repo: &str, path: &str) -> Option<Vec<u8>> {
        let (si, docid) = self.locate_doc(repo, path)?;
        self.set.content(si, docid).ok()
    }

    /// Lines `[start, end]` (1-based inclusive) of one indexed file, capped
    /// at [`MAX_SPAN_LINES`]; returns `(text, last_line_returned)`. `None`
    /// when the file is not indexed.
    pub fn read_span(&self, repo: &str, path: &str, start: u32, end: u32) -> Option<(String, u32)> {
        let (si, docid) = self.locate_doc(repo, path)?;
        let content = self.set.content(si, docid).ok()?;
        let start = start.max(1);
        let end = end.max(start).min(start + MAX_SPAN_LINES - 1);
        let mut out = String::new();
        let mut last = start.saturating_sub(1);
        // A trailing newline does not start a phantom empty last line.
        let body = content.strip_suffix(b"\n").unwrap_or(&content);
        for (i, raw) in body.split(|&b| b == b'\n').enumerate() {
            let ln = i as u32 + 1;
            if ln < start {
                continue;
            }
            if ln > end {
                break;
            }
            let raw = raw.strip_suffix(b"\r").unwrap_or(raw);
            out.push_str(&String::from_utf8_lossy(raw));
            out.push('\n');
            last = ln;
        }
        Some((out, last))
    }

    /// Map a diff's changed line ranges to the definitions they fall in
    /// (SPEC-P6 §2.2). Post-change content comes from `new_content(path)`
    /// (the working tree); pre-change content from the index.
    pub fn changed_symbols(
        &self,
        repo: &str,
        changes: &[FileChange],
        new_content: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Vec<ChangedSymbol> {
        let mut acc: HashMap<(String, String, u8, String), Vec<u32>> = HashMap::new();
        let mut order: Vec<(String, String, u8, String)> = Vec::new();
        let mut sigs: HashMap<(String, String, u8, String), String> = HashMap::new();
        let mut note = |path: &str, items: &[OutlineItem], content: &[u8], ranges: &[(u32, u32)]| {
            for &(a, b) in ranges {
                let b = b.min(a.saturating_add(MAX_RANGE_LINES));
                for line in a..=b {
                    if let Some(it) = enclosing(items, line) {
                        let key = (path.to_string(), it.name.clone(), it.kind.as_u8(), it.scope.clone());
                        let e = acc.entry(key.clone()).or_insert_with(|| {
                            sigs.insert(key.clone(), content_line(content, it.start_line).trim().to_string());
                            order.push(key);
                            Vec::new()
                        });
                        if e.len() < 20 && !e.contains(&line) {
                            e.push(line);
                        }
                    }
                }
            }
        };
        for c in changes {
            let lang = Lang::from_path(&c.path);
            if !lang.is_code() {
                continue;
            }
            if !c.deleted && !c.new_ranges.is_empty() {
                if let Some(content) = new_content(&c.path) {
                    let items = ts_outline(lang, &content);
                    note(&c.path, &items, &content, &c.new_ranges);
                }
            }
            if !c.added && !c.old_ranges.is_empty() {
                let old_path = c.old_path.as_deref().unwrap_or(&c.path);
                if let Some((si, docid)) = self.locate_doc(repo, old_path) {
                    if let Ok(content) = self.set.content(si, docid) {
                        let items = ts_outline(lang, &content);
                        note(&c.path, &items, &content, &c.old_ranges);
                    }
                }
            }
        }
        order
            .into_iter()
            .map(|key| {
                let mut lines = acc.remove(&key).unwrap_or_default();
                lines.sort_unstable();
                let signature = sigs.remove(&key).unwrap_or_default();
                ChangedSymbol {
                    path: key.0,
                    name: key.1,
                    kind: SymbolKind::from_u8(key.2),
                    scope: key.3,
                    lines,
                    signature,
                }
            })
            .collect()
    }

    /// The "I am about to change this" flow (SPEC-P6 §2.2): changed
    /// definitions → reverse-call walk (changed files excluded from sites)
    /// + importers of every changed file.
    pub fn impact_diff(
        &self,
        repo: &str,
        diff: &str,
        new_content: &dyn Fn(&str) -> Option<Vec<u8>>,
        opts: &ImpactOptions,
    ) -> (Vec<ChangedSymbol>, ImpactReport) {
        let t0 = Instant::now();
        let changes = parse_unified_diff(diff);
        let changed = self.changed_symbols(repo, &changes, new_content);
        let exclude: HashSet<(String, String)> = changes
            .iter()
            .map(|c| (repo.to_string(), c.path.clone()))
            .collect();
        let roots = roots_for(
            changed
                .iter()
                .map(|c| (c.name.clone(), c.kind, c.scope.clone())),
        );
        let visible = self.set.visible_ids();
        // Only the changed definitions are roots: `search` here means the
        // three provider classes that changed, not every `search` in the
        // index, so `re.search(` and `OtherProvider().search(` are out.
        let mut given: HashMap<String, RootShape> = HashMap::new();
        for c in &changed {
            let lang = Lang::from_path(&c.path);
            given
                .entry(c.name.clone())
                .or_default()
                .add_definition(&c.name, &c.path, lang, &c.scope, &c.signature);
        }
        let (sites, truncated) = self.walk(&roots, opts, &visible, &exclude, given);
        let mut importers: Vec<SearchHit> = Vec::new();
        let mut seen: HashSet<(String, String)> = HashSet::new();
        for c in &changes {
            if c.added {
                continue; // nothing could import it yet
            }
            for h in self.who_imports(repo, &c.path, 50) {
                let key = (h.repo.clone(), h.path.clone());
                if !exclude.contains(&key) && seen.insert(key) {
                    importers.push(h);
                }
            }
        }
        sort_hits(&mut importers);
        let mut report = finish_report(roots, Vec::new(), sites, truncated);
        report.importers = importers;
        report.took_ms = t0.elapsed().as_millis() as u64;
        (changed, report)
    }

    /// Impact of every definition in one indexed file + its importers.
    pub fn impact_file(&self, repo: &str, path: &str, opts: &ImpactOptions) -> ImpactReport {
        let t0 = Instant::now();
        let roots = roots_for(
            self.outline(repo, path)
                .unwrap_or_default()
                .into_iter()
                .map(|it| (it.name, it.kind, it.scope)),
        );
        let visible = self.set.visible_ids();
        let exclude: HashSet<(String, String)> =
            [(repo.to_string(), path.to_string())].into_iter().collect();
        let (sites, truncated) = self.walk(&roots, opts, &visible, &exclude, HashMap::new());
        let mut report = finish_report(roots, Vec::new(), sites, truncated);
        report.importers = self.who_imports(repo, path, 50);
        report.took_ms = t0.elapsed().as_millis() as u64;
        report
    }
}

/// Constructor methods are invoked through the type name (`Client(...)`,
/// `new Foo()`, `Foo::new()` is already `new`): editing one must also walk
/// the type's call sites. Returns the type name for such methods.
fn constructor_alias(name: &str, kind: SymbolKind, scope: &str) -> Option<String> {
    if kind != SymbolKind::Method || scope.is_empty() {
        return None;
    }
    if !matches!(name, "__init__" | "__new__" | "constructor") {
        return None;
    }
    // Rust scopes look like "Type::method"; Python/TS/Java scopes are the
    // container chain "Outer::Inner" — the last non-method segment is the
    // type in both shapes.
    scope
        .split("::")
        .filter(|s| !s.is_empty() && *s != name)
        .last()
        .map(str::to_string)
}

/// Root names for a set of changed/listed definitions: the names themselves
/// plus constructor aliases (SPEC-P6 §2.2).
fn roots_for(items: impl Iterator<Item = (String, SymbolKind, String)>) -> Vec<String> {
    let mut names = Vec::new();
    for (name, kind, scope) in items {
        if let Some(alias) = constructor_alias(&name, kind, &scope) {
            names.push(alias);
        }
        names.push(name);
    }
    dedup_names(&names)
}

fn dedup_names(names: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    names
        .iter()
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty() && seen.insert(n.clone()))
        .collect()
}

/// Sort sites (depth, repo, path, line) and aggregate per file.
fn finish_report(
    roots: Vec<String>,
    definitions: Vec<SearchHit>,
    mut sites: Vec<ImpactSite>,
    truncated: bool,
) -> ImpactReport {
    sites.sort_by(|a, b| {
        a.depth
            .cmp(&b.depth)
            .then_with(|| a.repo.cmp(&b.repo))
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.line.cmp(&b.line))
    });
    let mut agg: HashMap<(String, String), (u32, u32)> = HashMap::new();
    for s in &sites {
        let e = agg
            .entry((s.repo.clone(), s.path.clone()))
            .or_insert((0, s.depth));
        e.0 += 1;
        e.1 = e.1.min(s.depth);
    }
    let mut files: Vec<FileImpact> = agg
        .into_iter()
        .map(|((repo, path), (n, d))| FileImpact {
            repo,
            path,
            sites: n,
            min_depth: d,
        })
        .collect();
    files.sort_by(|a, b| {
        a.min_depth
            .cmp(&b.min_depth)
            .then_with(|| b.sites.cmp(&a.sites))
            .then_with(|| a.repo.cmp(&b.repo))
            .then_with(|| a.path.cmp(&b.path))
    });
    ImpactReport {
        roots,
        definitions,
        sites,
        files,
        importers: Vec::new(),
        truncated,
        took_ms: 0,
    }
}

/// Text of 1-based `line` (used by tests and the CLI outline printer).
pub fn line_text(content: &[u8], line: u32) -> String {
    content_line(content, line)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use indexio_core::grams::{self, CommonGrams};
    use indexio_index::ShardWriter;
    use indexio_types::{BlobId, ExtractedArtifact};
    use std::path::Path;

    /// Real extraction (tree-sitter) so call edges/callers match production.
    fn write_repo(dir: &Path, repo: &str, docs: &[(&str, &str)]) {
        let mut w = ShardWriter::new(dir).unwrap();
        for (path, src) in docs {
            let content = src.as_bytes();
            let lang = Lang::from_path(path);
            let (symbols, calls) = indexio_symbols::extract(lang, content);
            let art = ExtractedArtifact {
                ngrams: grams::extract(content, &CommonGrams::empty()),
                symbols,
                calls,
                raw_len: content.len() as u32,
                lang,
            };
            let meta = DocMeta {
                blob: BlobId::from_content(content),
                repo_id: 0,
                path: path.to_string(),
                lang,
                raw_len: content.len() as u32,
            };
            w.add_doc(&meta, content, &art).unwrap();
        }
        w.finish(&[repo.to_string()]).unwrap();
    }

    const CORE_LIB: &str = "pub mod parse;\npub fn parse_config(s: &str) -> u32 {\n    s.len() as u32\n}\n\npub fn load(path: &str) -> u32 {\n    parse_config(path)\n}\n";
    const CORE_MAIN: &str = "use crate::config::parse_config;\nfn main() {\n    let n = load(\"x\");\n    println!(\"{n}\");\n}\n";
    const APP_SVC: &str = "use core_lib::config::load;\n\nfn boot() {\n    load(\"svc.toml\");\n}\n\nfn unrelated() {\n    other();\n}\n";

    fn engine() -> (tempfile::TempDir, Engine) {
        let tmp = tempfile::tempdir().unwrap();
        let shards = tmp.path().join("shards");
        std::fs::create_dir_all(&shards).unwrap();
        write_repo(
            &shards,
            "core",
            &[("src/config.rs", CORE_LIB), ("src/main.rs", CORE_MAIN)],
        );
        write_repo(&shards, "app", &[("src/svc.rs", APP_SVC)]);
        let e = Engine::open(tmp.path()).unwrap();
        (tmp, e)
    }

    #[test]
    fn qualified_sites_are_judged_by_the_definitions() {
        let mut free = RootShape::default();
        free.add_definition("spawn", "crates/x/src/net/mod.rs", Lang::Rust, "", "pub fn spawn(rt: &Runtime) -> Handle {");
        assert!(free.owners.contains("net"));
        assert!(!free.has_receiver);
        assert!(site_plausible("    tokio::spawn(async move {", "spawn", &free) == false);
        assert!(!site_plausible("    std::thread::spawn(move || {", "spawn", &free));
        assert!(!site_plausible("    let child = cmd.spawn()?;", "spawn", &free));
        assert!(site_plausible("    let h = net::spawn(&rt);", "spawn", &free));
        assert!(site_plausible("    let h = crate::net::spawn(&rt);", "spawn", &free));
        assert!(site_plausible("    let h = spawn(&rt);", "spawn", &free));
        assert!(site_plausible("    tasks.push(spawn);", "spawn", &free)); // not a call: keep
        assert!(site_plausible("    respawn(x);", "spawn", &free)); // other identifier: keep

        let mut method = RootShape::default();
        method.add_definition("spawn", "crates/x/src/client.rs", Lang::Rust, "Client::spawn", "    pub fn spawn(&self, x: u32) {");
        assert!(method.has_receiver);
        assert!(method.owners.contains("Client"), "{:?}", method.owners);
        assert!(site_plausible("    self.client.spawn(3);", "spawn", &method));
        assert!(site_plausible("    Client::spawn(&c, 3);", "spawn", &method));
        assert!(!site_plausible("    tokio::spawn(fut);", "spawn", &method));

        let mut assoc = RootShape::default();
        assoc.add_definition("spawn", "crates/x/src/client.rs", Lang::Rust, "Client::spawn", "    pub async fn spawn(server: S) -> Result<Self> {");
        assert!(!assoc.has_receiver);
        assert!(site_plausible("    Client::spawn(s).await", "spawn", &assoc));
        assert!(!site_plausible("    cmd.spawn()?", "spawn", &assoc));

        let mut py = RootShape::default();
        py.add_definition("spawn", "pkg/worker.py", Lang::Python, "Worker", "    def spawn(self, n):");
        assert!(site_plausible("        self.spawn(2)", "spawn", &py));
        assert!(site_plausible("        worker.spawn(2)", "spawn", &py)); // receiver unknown: kept
        assert!(!site_plausible("        subprocess.spawn(2)", "spawn", &py)); // a library module
        assert!(site_plausible("        Worker().spawn(2)", "spawn", &py));
        assert!(!site_plausible("        Pool().spawn(2)", "spawn", &py)); // another type

        // the providers diff: `def search(self, q, ctx)` on three provider classes
        let mut search = RootShape::default();
        search.add_definition("search", "src/app/providers/apis.py", Lang::Python, "HackerNews", "    async def search(self, q: Query, ctx) -> list[Doc]:");
        search.add_definition("search", "src/app/providers/apis.py", Lang::Python, "Crossref", "    async def search(self, q: Query, ctx) -> list[Doc]:");
        assert!(!site_plausible("    m = re.search(r\"<title[^>]*>\", html, re.I)", "search", &search));
        assert!(!site_plausible("    tm = _TITLE_RE.search(body[:8192])", "search", &search));
        assert!(!site_plausible("    docs = await OpenAlex().search(Query(text=\"x\"), ctx)", "search", &search));
        assert!(!site_plausible("    asyncio.run(Engine.search(eng, \"heat pump\"))", "search", &search));
        assert!(site_plausible("    docs = await Crossref().search(_q(), ctx)", "search", &search));
        assert!(site_plausible("    res = await provider.search(q, ctx)", "search", &search));
        assert!(site_plausible("    title = _RSS_FIELD[\"title\"].search(block)", "search", &search) == false);
        assert_eq!(trailing_ident("OpenAlex()"), "OpenAlex");
        assert_eq!(trailing_ident("_RSS_FIELD[\"title\"]"), "_RSS_FIELD");
        assert_eq!(trailing_ident("foo(a, b(c))"), "foo");
        assert_eq!(trailing_ident("tokio::task"), "task");

        let unknown = RootShape { unknown: true, ..Default::default() };
        assert!(site_plausible("    tokio::spawn(fut);", "spawn", &unknown));
    }

    #[test]
    fn impact_depth1_direct_callers_and_definitions() {
        let (_t, e) = engine();
        let opts = ImpactOptions {
            depth: 1,
            ..Default::default()
        };
        let r = e.impact_symbols(&["parse_config".into()], &opts);
        assert_eq!(r.roots, vec!["parse_config".to_string()]);
        assert_eq!(r.definitions.len(), 1, "{:?}", r.definitions);
        assert_eq!(r.definitions[0].path, "src/config.rs");
        // Direct caller: load() in config.rs, line 7.
        assert_eq!(r.sites.len(), 1, "{:?}", r.sites);
        let s = &r.sites[0];
        assert_eq!((s.repo.as_str(), s.path.as_str(), s.line, s.depth), ("core", "src/config.rs", 7, 1));
        assert_eq!(s.caller, "load");
        assert_eq!(s.symbol, "parse_config");
        assert!(!r.truncated);
        assert_eq!(r.files.len(), 1);
        assert_eq!(r.files[0].sites, 1);
    }

    #[test]
    fn impact_depth2_crosses_repos_via_caller() {
        let (_t, e) = engine();
        let r = e.impact_symbols(&["parse_config".into()], &ImpactOptions::default());
        // depth 2: callers of `load` — main.rs (core) and svc.rs (app).
        let d2: Vec<(&str, &str, u32)> = r
            .sites
            .iter()
            .filter(|s| s.depth == 2)
            .map(|s| (s.repo.as_str(), s.path.as_str(), s.line))
            .collect();
        assert!(d2.contains(&("core", "src/main.rs", 3)), "{d2:?}");
        assert!(d2.contains(&("app", "src/svc.rs", 4)), "{d2:?}");
        // sites are depth-ordered; files aggregate with min_depth
        assert!(r.sites.windows(2).all(|w| w[0].depth <= w[1].depth));
        let f = r.files.iter().find(|f| f.path == "src/svc.rs").unwrap();
        assert_eq!((f.repo.as_str(), f.min_depth, f.sites), ("app", 2, 1));
        // depth-2 site names the hop's callee
        assert!(r.sites.iter().filter(|s| s.depth == 2).all(|s| s.symbol == "load"));
    }

    #[test]
    fn impact_cycle_terminates_and_caps() {
        let tmp = tempfile::tempdir().unwrap();
        let shards = tmp.path().join("shards");
        std::fs::create_dir_all(&shards).unwrap();
        // a -> b -> a recursion plus a hub with many callers.
        let mut hub = String::new();
        for i in 0..30 {
            hub.push_str(&format!("fn c{i}() {{ hub(); }}\n"));
        }
        write_repo(
            &shards,
            "r",
            &[
                ("src/cyc.rs", "fn a() { b(); }\nfn b() { a(); }\n"),
                ("src/hub.rs", &hub),
            ],
        );
        let e = Engine::open(tmp.path()).unwrap();
        let r = e.impact_symbols(
            &["a".into()],
            &ImpactOptions {
                depth: 10,
                ..Default::default()
            },
        );
        assert!(!r.truncated);
        // a <- b (depth 1), b <- a (depth 2); then `a` is visited: stop.
        assert_eq!(r.sites.len(), 2, "{:?}", r.sites);
        let r = e.impact_symbols(
            &["hub".into()],
            &ImpactOptions {
                depth: 1,
                max_sites: 10,
                max_fanout: 100,
            },
        );
        assert!(r.truncated);
        assert_eq!(r.sites.len(), 10);
        let r = e.impact_symbols(
            &["hub".into()],
            &ImpactOptions {
                depth: 1,
                max_sites: 1000,
                max_fanout: 5,
            },
        );
        assert!(r.truncated);
        assert_eq!(r.sites.len(), 5);
        // Even sampling: 30 postings / cap 5 = stride 6 -> c0, c6, ..., c24
        // rather than the first five in shard order.
        let callers: Vec<&str> = r.sites.iter().map(|s| s.caller.as_str()).collect();
        assert_eq!(callers, vec!["c0", "c6", "c12", "c18", "c24"], "{callers:?}");
    }

    #[test]
    fn impact_empty_and_unknown_names() {
        let (_t, e) = engine();
        let r = e.impact_symbols(&["".into(), "   ".into()], &ImpactOptions::default());
        assert!(r.roots.is_empty() && r.sites.is_empty() && r.definitions.is_empty());
        let r = e.impact_symbols(&["does_not_exist_anywhere".into()], &ImpactOptions::default());
        assert_eq!(r.roots.len(), 1);
        assert!(r.sites.is_empty());
    }

    #[test]
    fn parse_diff_hunks_add_modify_delete_rename() {
        let diff = "\
diff --git a/src/config.rs b/src/config.rs
--- a/src/config.rs
+++ b/src/config.rs
@@ -3,1 +3,2 @@ pub fn parse_config
-    s.len() as u32
+    let n = s.len();
+    n as u32
@@ -7 +8,0 @@
-    parse_config(path)
diff --git a/new.py b/new.py
--- /dev/null
+++ b/new.py
@@ -0,0 +1,3 @@
+a
+b
+c
diff --git a/gone.go b/gone.go
--- a/gone.go
+++ /dev/null
@@ -1,2 +0,0 @@
-x
-y
--- a/old_name.ts	2024-01-01
+++ b/new_name.ts	2024-01-02
@@ -10,3 +12,3 @@
";
        let fc = parse_unified_diff(diff);
        assert_eq!(fc.len(), 4, "{fc:?}");
        assert_eq!(fc[0].path, "src/config.rs");
        assert_eq!(fc[0].old_ranges, vec![(3, 3), (7, 7)]);
        // pure deletion on the new side maps to the single line it sits on
        assert_eq!(fc[0].new_ranges, vec![(3, 4), (8, 8)]);
        assert!(!fc[0].added && !fc[0].deleted);
        assert!(fc[1].added && fc[1].old_ranges.is_empty());
        assert_eq!(fc[1].new_ranges, vec![(1, 3)]);
        assert!(fc[2].deleted && fc[2].new_ranges == vec![(1, 1)]);
        assert_eq!(fc[2].old_ranges, vec![(1, 2)]);
        assert_eq!(fc[3].path, "new_name.ts");
        assert_eq!(fc[3].old_path.as_deref(), Some("old_name.ts"));
        assert_eq!(fc[3].old_ranges, vec![(10, 12)]);
        assert!(parse_unified_diff("").is_empty());
        assert!(parse_unified_diff("not a diff\n").is_empty());
    }

    #[test]
    fn changed_symbols_maps_lines_to_definitions() {
        let (_t, e) = engine();
        // Working tree: body of parse_config changed (line 3 → 3-4);
        // module-level `pub mod parse;` untouched.
        let new_src = "pub mod parse;\npub fn parse_config(s: &str) -> u32 {\n    let n = s.len();\n    n as u32\n}\n\npub fn load(path: &str) -> u32 {\n    parse_config(path)\n}\n";
        let diff = "--- a/src/config.rs\n+++ b/src/config.rs\n@@ -3,1 +3,2 @@\n-    s.len() as u32\n+    let n = s.len();\n+    n as u32\n";
        let provider = |p: &str| (p == "src/config.rs").then(|| new_src.as_bytes().to_vec());
        let changed = e.changed_symbols("core", &parse_unified_diff(diff), &provider);
        assert_eq!(changed.len(), 1, "{changed:?}");
        assert_eq!(changed[0].name, "parse_config");
        assert_eq!(changed[0].kind, SymbolKind::Fn);
        assert_eq!(changed[0].lines, vec![3, 4]);
        // An edit outside any definition (the blank line 5) yields nothing.
        let diff = "--- a/src/config.rs\n+++ b/src/config.rs\n@@ -5,1 +5,1 @@\n-\n+// comment\n";
        let orig = |p: &str| (p == "src/config.rs").then(|| CORE_LIB.as_bytes().to_vec());
        let changed = e.changed_symbols("core", &parse_unified_diff(diff), &orig);
        assert!(changed.is_empty(), "{changed:?}");
        // Deleted lines are mapped through the indexed (pre-change) content
        // even when the working-tree file is unavailable.
        let diff = "--- a/src/config.rs\n+++ b/src/config.rs\n@@ -7,1 +7,0 @@\n-    parse_config(path)\n";
        let none = |_: &str| None;
        let changed = e.changed_symbols("core", &parse_unified_diff(diff), &none);
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].name, "load");
    }

    #[test]
    fn impact_diff_excludes_changed_file_and_finds_importers() {
        let (_t, e) = engine();
        let diff = "--- a/src/config.rs\n+++ b/src/config.rs\n@@ -3,1 +3,1 @@\n-    s.len() as u32\n+    s.len() as u32 + 1\n";
        let provider = |p: &str| (p == "src/config.rs").then(|| CORE_LIB.as_bytes().to_vec());
        let (changed, r) = e.impact_diff("core", diff, &provider, &ImpactOptions::default());
        assert_eq!(changed.len(), 1);
        assert_eq!(r.roots, vec!["parse_config".to_string()]);
        // load() lives in the changed file: excluded as a site, but still
        // walked — its callers show up at depth 2.
        assert!(r.sites.iter().all(|s| s.path != "src/config.rs"), "{:?}", r.sites);
        let paths: Vec<(&str, &str)> = r.sites.iter().map(|s| (s.repo.as_str(), s.path.as_str())).collect();
        assert!(paths.contains(&("core", "src/main.rs")), "{paths:?}");
        assert!(paths.contains(&("app", "src/svc.rs")), "{paths:?}");
        // importers: `use crate::config::parse_config` / `use core_lib::config::load`
        let imp: Vec<(&str, &str)> = r.importers.iter().map(|h| (h.repo.as_str(), h.path.as_str())).collect();
        assert!(imp.contains(&("core", "src/main.rs")), "{imp:?}");
        assert!(imp.contains(&("app", "src/svc.rs")), "{imp:?}");
        assert!(!imp.contains(&("core", "src/config.rs")));
    }

    #[test]
    fn who_imports_patterns_per_language() {
        assert_eq!(import_pattern(Lang::Rust, "src/config.rs").unwrap().0, "config");
        assert_eq!(import_pattern(Lang::Rust, "src/net/mod.rs").unwrap().0, "net");
        assert!(import_pattern(Lang::Rust, "src/main.rs").is_none());
        assert_eq!(import_pattern(Lang::Python, "pkg/util.py").unwrap().0, "util");
        assert_eq!(import_pattern(Lang::Python, "pkg/sub/__init__.py").unwrap().0, "sub");
        assert_eq!(import_pattern(Lang::Go, "internal/store/db.go").unwrap().0, "store");
        assert!(import_pattern(Lang::Go, "main.go").is_none());
        assert_eq!(import_pattern(Lang::TsJs, "src/lib/index.ts").unwrap().0, "lib");
        assert_eq!(import_pattern(Lang::TsJs, "src/lib/api.tsx").unwrap().0, "api");
        assert_eq!(import_pattern(Lang::Java, "com/x/Foo.java").unwrap().0, "Foo");
        assert_eq!(import_pattern(Lang::Cpp, "inc/util.h").unwrap().0, "util");
        assert!(import_pattern(Lang::Unknown, "README.md").is_none());
        // every pattern compiles
        for (lang, p) in [
            (Lang::Rust, "a/b.rs"),
            (Lang::Python, "a/b.py"),
            (Lang::Go, "a/b.go"),
            (Lang::TsJs, "a/b.ts"),
            (Lang::Java, "a/B.java"),
            (Lang::Cpp, "a/b.cpp"),
        ] {
            let (_, pat) = import_pattern(lang, p).unwrap();
            assert!(regex::bytes::Regex::new(&pat).is_ok(), "{lang:?}: {pat}");
        }
    }

    #[test]
    fn who_imports_python_and_ts() {
        let tmp = tempfile::tempdir().unwrap();
        let shards = tmp.path().join("shards");
        std::fs::create_dir_all(&shards).unwrap();
        write_repo(
            &shards,
            "py",
            &[
                ("pkg/util.py", "def helper():\n    return 1\n"),
                ("pkg/app.py", "from pkg.util import helper\n\ndef run():\n    helper()\n"),
                ("pkg/other.py", "import os\n"),
                ("web/api.ts", "export const x = 1;\n"),
                ("web/page.tsx", "import { x } from './api';\nconst y = x;\n"),
                ("web/none.ts", "import fs from 'fs';\n"),
            ],
        );
        let e = Engine::open(tmp.path()).unwrap();
        let h = e.who_imports("py", "pkg/util.py", 10);
        assert_eq!(h.iter().map(|h| h.path.as_str()).collect::<Vec<_>>(), vec!["pkg/app.py"]);
        let h = e.who_imports("py", "web/api.ts", 10);
        assert_eq!(h.iter().map(|h| h.path.as_str()).collect::<Vec<_>>(), vec!["web/page.tsx"]);
    }

    #[test]
    fn outline_and_read_span() {
        let (_t, e) = engine();
        let items = e.outline("core", "src/config.rs").unwrap();
        let names: Vec<(&str, u32, u32)> = items
            .iter()
            .map(|i| (i.name.as_str(), i.start_line, i.end_line))
            .collect();
        assert!(names.contains(&("parse_config", 2, 4)), "{names:?}");
        assert!(names.contains(&("load", 6, 8)), "{names:?}");
        assert!(e.outline("core", "nope.rs").is_none());

        let (text, last) = e.read_span("core", "src/config.rs", 2, 3).unwrap();
        assert_eq!(text, "pub fn parse_config(s: &str) -> u32 {\n    s.len() as u32\n");
        assert_eq!(last, 3);
        // clamps: start 0 -> 1; end past EOF -> last real line
        let (text, last) = e.read_span("core", "src/config.rs", 0, 999).unwrap();
        assert_eq!(last, 8, "{text:?}");
        assert!(text.starts_with("pub mod parse;\n"));
        // cap
        let (_, last) = e.read_span("core", "src/config.rs", 1, 1 + MAX_SPAN_LINES + 50).unwrap();
        assert!(last <= MAX_SPAN_LINES);
        assert!(e.read_span("core", "missing.rs", 1, 2).is_none());
    }

    #[test]
    fn constructor_edits_walk_type_call_sites() {
        assert_eq!(
            constructor_alias("__init__", SymbolKind::Method, "Client").as_deref(),
            Some("Client")
        );
        assert_eq!(
            constructor_alias("constructor", SymbolKind::Method, "Outer::Client").as_deref(),
            Some("Client")
        );
        assert!(constructor_alias("__init__", SymbolKind::Method, "").is_none());
        assert!(constructor_alias("run", SymbolKind::Method, "Client").is_none());
        assert!(constructor_alias("__init__", SymbolKind::Fn, "x").is_none());

        let tmp = tempfile::tempdir().unwrap();
        let shards = tmp.path().join("shards");
        std::fs::create_dir_all(&shards).unwrap();
        write_repo(
            &shards,
            "py",
            &[
                (
                    "pkg/client.py",
                    "class Client:\n    def __init__(self, url):\n        self.url = url\n\n    def get(self):\n        return self.url\n",
                ),
                (
                    "pkg/app.py",
                    "from pkg.client import Client\n\ndef main():\n    c = Client(\"http://x\")\n    c.get()\n",
                ),
            ],
        );
        let e = Engine::open(tmp.path()).unwrap();
        let diff = "--- a/pkg/client.py\n+++ b/pkg/client.py\n@@ -3,1 +3,1 @@\n-        self.url = url\n+        self.url = url.strip()\n";
        let none = |_: &str| None;
        let (changed, r) = e.impact_diff("py", diff, &none, &ImpactOptions::default());
        assert_eq!(changed[0].name, "__init__");
        assert_eq!(changed[0].scope, "Client");
        assert_eq!(r.roots, vec!["Client".to_string(), "__init__".to_string()]);
        assert!(
            r.sites.iter().any(|s| s.path == "pkg/app.py" && s.symbol == "Client" && s.line == 4),
            "{:?}",
            r.sites
        );
    }

    #[test]
    fn impact_file_walks_every_definition() {
        let (_t, e) = engine();
        let r = e.impact_file("core", "src/config.rs", &ImpactOptions::default());
        assert!(r.roots.contains(&"parse_config".to_string()));
        assert!(r.roots.contains(&"load".to_string()));
        assert!(r.sites.iter().all(|s| s.path != "src/config.rs"));
        assert!(r.sites.iter().any(|s| s.path == "src/svc.rs" && s.depth == 1));
        assert!(!r.importers.is_empty());
    }
}
