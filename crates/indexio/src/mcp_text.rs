//! Compact plain-text renderers for MCP tool results.
//!
//! The consumer of an MCP tool result is a language model that pays per
//! token, so the default MCP output is not JSON but a dense, grep-like text
//! layout: one `repo:path` header per file, one `line: snippet` row per
//! hit, directory-folded file lists, and no scores / ranks / nulls the
//! model never acts on. Measured on a real repo (bench/mcp_tokens.py) this
//! is 40–75% fewer tokens than the compact-JSON envelope for the same
//! information. `INDEXIO_MCP_FORMAT=json` (or a per-call `format: "json"`
//! argument) restores the JSON envelope for A/B tests and scripts.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use indexio_query::{ChangedSymbol, HybridHit, ImpactReport, ImpactSite};
use indexio_symbols::OutlineItem;
use indexio_types::{EngineStats, SearchHit};

/// Longest snippet we keep per hit (bytes, on a char boundary).
const MAX_SNIPPET: usize = 200;

/// Call sites listed per file in an impact report; the rest is summarised
/// as a count so hub names (`new`, `get`) stay bounded while every affected
/// file is still named.
const MAX_SITES_PER_FILE: usize = 6;

/// `read_span` rows carry their line number only every this many lines
/// (plus the first and the last row): −12 % tokens on a span, and a row
/// is at most four lines from a numbered one.
const SPAN_NUMBER_EVERY: u32 = 5;

/// Transitive (depth >= 2) sites are listed individually up to this many;
/// beyond it they are collapsed per intermediate symbol.
const COLLAPSE_TRANSITIVE_ABOVE: usize = 40;

/// Files named per collapsed intermediate symbol / repo.
const COLLAPSED_TOP_FILES: usize = 4;

/// With a current repo, direct call sites in OTHER repos are collapsed to
/// per-repo counts once there are more than this many direct sites overall.
const COLLAPSE_DIRECT_ABOVE: usize = 30;

/// Definitions listed in full; past this the session's repo keeps its rows
/// (up to [`MAX_DEFS_SHOWN`]) and other repos collapse to per-repo counts
/// (SPEC-P9 §20: `main` had 305 definitions across 41 repos).
const COLLAPSE_DEFS_ABOVE: usize = 12;
const MAX_DEFS_SHOWN: usize = 20;

/// Direct-caller rows shown in full for one repo; the remaining files are
/// listed as `path (n)` tallies. Bounds a hub name's payload.
const MAX_DIRECT_ROWS: usize = 60;

/// Snippet cap for call-site rows (the caller name already carries context).
const MAX_SITE_SNIPPET: usize = 120;

/// One-line, whitespace-trimmed, length-capped snippet.
pub fn snippet(s: &str) -> String {
    snippet_capped(s, MAX_SNIPPET)
}

fn snippet_capped(s: &str, cap: usize) -> String {
    let s = s.trim();
    let first = s.lines().next().unwrap_or("");
    let mut out: String = first.split_whitespace().collect::<Vec<_>>().join(" ");
    if out.len() > cap {
        let mut cut = cap;
        while !out.is_char_boundary(cut) {
            cut -= 1;
        }
        out.truncate(cut);
        out.push('…');
    }
    out
}

/// Hit list as `repo:path` headers with `  line: snippet` rows.
///
/// `grouped = true` (lexical/grep/symbol lookups): hits are gathered per
/// file, files ordered by their best hit, lines ascending — the rg
/// layout. `grouped = false` (ranked semantic/hybrid lists): rank order is
/// kept and only consecutive hits in the same file share a header.
pub fn hits(hits: &[SearchHit], grouped: bool, truncated: bool, what: &str) -> String {
    hits_with_ends(hits, grouped, truncated, what, &|_| None)
}

/// [`hits`] where `end_of(hit)` may supply the last line of the definition
/// a hit starts (rendered `start-end: snippet`), so the agent can read
/// exactly that range next.
pub fn hits_with_ends(
    hits: &[SearchHit],
    grouped: bool,
    truncated: bool,
    what: &str,
    end_of: &dyn Fn(&SearchHit) -> Option<u32>,
) -> String {
    if hits.is_empty() {
        return format!("no {what}");
    }
    let row = |h: &SearchHit| match end_of(h) {
        Some(end) if end > h.line => format!("  {}-{}: {}", h.line, end, snippet(&h.snippet)),
        _ => format!("  {}: {}", h.line, snippet(&h.snippet)),
    };
    let mut out = String::new();
    let mut files = 0usize;
    if grouped {
        // first-appearance order of files, then lines ascending inside
        let mut order: Vec<(&str, &str)> = Vec::new();
        let mut by_file: BTreeMap<(&str, &str), Vec<&SearchHit>> = BTreeMap::new();
        for h in hits {
            let key = (h.repo.as_str(), h.path.as_str());
            if !by_file.contains_key(&key) {
                order.push(key);
            }
            by_file.entry(key).or_default().push(h);
        }
        for key in order {
            let mut rows = by_file.remove(&key).unwrap_or_default();
            rows.sort_by_key(|h| h.line);
            rows.dedup_by_key(|h| h.line);
            files += 1;
            let _ = writeln!(out, "{}:{}", key.0, key.1);
            for h in rows {
                let _ = writeln!(out, "{}", row(h));
            }
        }
    } else {
        let mut last: Option<(&str, &str)> = None;
        for h in hits {
            let key = (h.repo.as_str(), h.path.as_str());
            if last != Some(key) {
                files += 1;
                let _ = writeln!(out, "{}:{}", key.0, key.1);
                last = Some(key);
            }
            let _ = writeln!(out, "{}", row(h));
        }
    }
    let _ = write!(out, "-- {} {what} in {} file{}", hits.len(), files, plural(files));
    if truncated {
        out.push_str(" (truncated; raise limit or narrow the query)");
    }
    out
}

/// Call-site rows per file are capped at this many; the rest of a file's
/// sites are listed as bare line numbers (SPEC-P10: `register` had 16
/// near-identical `Registry.register(...)` rows in one file — the
/// locations matter, the repeated text does not).
const MAX_CALL_ROWS_PER_FILE: usize = 6;

/// [`hits`] for call sites: grouped per file, at most
/// [`MAX_CALL_ROWS_PER_FILE`] snippet rows per file, the remaining sites of
/// that file as `  … +N more at l1 l2 l3`.
pub fn call_sites(hits: &[SearchHit], truncated: bool, what: &str) -> String {
    if hits.is_empty() {
        return format!("no {what}");
    }
    let mut out = String::new();
    let mut order: Vec<(&str, &str)> = Vec::new();
    let mut by_file: BTreeMap<(&str, &str), Vec<&SearchHit>> = BTreeMap::new();
    for h in hits {
        let key = (h.repo.as_str(), h.path.as_str());
        if !by_file.contains_key(&key) {
            order.push(key);
        }
        by_file.entry(key).or_default().push(h);
    }
    let files = order.len();
    for key in order {
        let mut rows = by_file.remove(&key).unwrap_or_default();
        rows.sort_by_key(|h| h.line);
        rows.dedup_by_key(|h| h.line);
        let _ = writeln!(out, "{}:{}", key.0, key.1);
        for h in rows.iter().take(MAX_CALL_ROWS_PER_FILE) {
            let _ = writeln!(out, "  {}: {}", h.line, snippet_capped(&h.snippet, MAX_SITE_SNIPPET));
        }
        if rows.len() > MAX_CALL_ROWS_PER_FILE {
            let rest: Vec<String> = rows[MAX_CALL_ROWS_PER_FILE..].iter().map(|h| h.line.to_string()).collect();
            let _ = writeln!(out, "  … +{} more at {}", rest.len(), rest.join(" "));
        }
    }
    let _ = write!(out, "-- {} {what} in {} file{}", hits.len(), files, plural(files));
    if truncated {
        out.push_str(" (truncated; raise limit or narrow the query)");
    }
    out
}

/// Fused hybrid hits: rank order, same layout as [`hits`].
pub fn hybrid_hits(fused: &[HybridHit], what: &str) -> String {
    let plain: Vec<SearchHit> = fused.iter().map(|h| h.hit.clone()).collect();
    hits(&plain, false, false, what)
}

/// File list folded by directory: `repo:dir/  a.rs b.rs c.rs`.
pub fn files(list: &[(String, String)], truncated: bool) -> String {
    if list.is_empty() {
        return "no files match".to_string();
    }
    let mut out = String::new();
    let mut last_dir: Option<(&str, &str)> = None;
    let mut on_line = 0usize;
    for (repo, path) in list {
        let (dir, name) = match path.rfind('/') {
            Some(i) => (&path[..=i], &path[i + 1..]),
            None => ("", path.as_str()),
        };
        let key = (repo.as_str(), dir);
        if last_dir != Some(key) {
            if last_dir.is_some() {
                out.push('\n');
            }
            let _ = write!(out, "{repo}:{dir} ");
            last_dir = Some(key);
            on_line = 0;
        }
        if on_line > 0 {
            out.push(' ');
        }
        if name.contains(' ') {
            let _ = write!(out, "\"{name}\"");
        } else {
            out.push_str(name);
        }
        on_line += 1;
    }
    let _ = write!(out, "\n-- {} file{}", list.len(), plural(list.len()));
    if truncated {
        out.push_str(" (truncated; raise limit or narrow the pattern)");
    }
    out
}

/// Outline: `start-end kind name`, indented by scope depth.
pub fn outline(repo: &str, path: &str, items: &[OutlineItem]) -> String {
    let mut out = format!(
        "{repo}:{path}  {} definition{}\n",
        items.len(),
        plural(items.len())
    );
    for it in items {
        let depth = if it.scope.is_empty() {
            0
        } else {
            it.scope.split("::").count()
        };
        let _ = writeln!(
            out,
            "{}{}-{} {} {}",
            "  ".repeat(depth),
            it.start_line,
            it.end_line,
            it.kind.name(),
            it.name
        );
    }
    out.truncate(out.trim_end().len());
    out
}

/// A line range: header + `line<TAB>text` rows (a tab costs no more than
/// the JSON `"\n"` escapes it replaces and keeps numbers unambiguous).
pub fn span(repo: &str, path: &str, start: u32, last: u32, text: &str) -> String {
    let mut out = format!("{repo}:{path} L{start}-{last}\n");
    let body = text.strip_suffix('\n').unwrap_or(text);
    if last >= start {
        // Every line is `N<TAB>text`? No: the number is a fifth of the
        // tokens of a short line (SPEC-P9 §21), so only the first, the
        // last and every multiple of [`SPAN_NUMBER_EVERY`] carry it; the
        // other rows start with the tab alone (columns stay aligned, a row
        // never starts with digits by accident).
        let n = body.split('\n').count();
        for (i, line) in body.split('\n').enumerate() {
            let ln = start + i as u32;
            if i == 0 || i + 1 == n || ln % SPAN_NUMBER_EVERY == 0 {
                let _ = writeln!(out, "{ln}\t{line}");
            } else {
                let _ = writeln!(out, "\t{line}");
            }
        }
    }
    out.truncate(out.trim_end_matches('\n').len());
    out
}

/// Index stats: one summary line, then one line per repo.
pub fn stats(stats: &EngineStats, detail: &[(String, String, String, String, bool)]) -> String {
    let mut out = format!(
        "{} repo{}, {} files, {} source, {} shard{}, index {}",
        stats.repos.len(),
        plural(stats.repos.len()),
        stats.doc_count,
        human_bytes(stats.total_raw_bytes),
        stats.shard_count,
        plural(stats.shard_count as usize),
        human_bytes(stats.index_bytes)
    );
    if detail.is_empty() {
        for r in &stats.repos {
            let _ = write!(out, "\n{r}");
        }
        return out;
    }
    // SPEC-P10 §25: one line per sync day listing the repo names (`*` =
    // plain folder), and one line for the repos whose folder is not
    // `<root>/<name>` — the root being the folder most repos share. The
    // commit hashes are in the JSON payload; a session start does not
    // need them. (One row per repo with folder and commit was 2.9 KB for
    // 45 repos, ~700 tokens at every session start.)
    let root = majority_dir(detail.iter().map(|d| (d.0.as_str(), d.1.as_str())));
    if !root.is_empty() {
        let _ = write!(out, "\nroot: {root}");
    }
    let mut by_day: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut elsewhere: Vec<String> = Vec::new();
    for (name, path, _commit, at, plain) in detail {
        let day: String = at.chars().take(10).collect();
        let label = if *plain { format!("{name}*") } else { name.clone() };
        by_day.entry(if day.is_empty() { "-".to_string() } else { day }).or_default().push(label);
        let standard = !root.is_empty() && strip_sep(path.strip_prefix(&root).unwrap_or("")) == name;
        if !standard {
            elsewhere.push(format!("{name}={path}"));
        }
    }
    for (day, names) in by_day.iter().rev() {
        let _ = write!(out, "\nsynced {day}: {}", names.join(" "));
    }
    if !elsewhere.is_empty() {
        let _ = write!(out, "\nnot under root: {}", elsewhere.join(" "));
    }
    if detail.iter().any(|d| d.4) {
        out.push_str("\n* plain folder (no git)");
    }
    out
}

fn strip_sep(s: &str) -> &str {
    s.trim_start_matches(['/', '\\']).trim_end_matches(['/', '\\'])
}

/// The parent folder `<dir>/<name>` that the most repos sit in directly,
/// with its trailing separator; "" when no two repos share one.
fn majority_dir<'a>(repos: impl Iterator<Item = (&'a str, &'a str)>) -> String {
    let mut count: BTreeMap<String, usize> = BTreeMap::new();
    for (name, path) in repos {
        let Some(parent) = path.strip_suffix(name) else { continue };
        if parent.ends_with(['/', '\\']) {
            *count.entry(parent.to_string()).or_insert(0) += 1;
        }
    }
    count
        .into_iter()
        .filter(|(_, n)| *n >= 2)
        .max_by_key(|(p, n)| (*n, std::cmp::Reverse(p.clone())))
        .map(|(p, _)| p)
        .unwrap_or_default()
}

/// refresh_index summary.
pub fn refresh(
    repos: &[(String, u64, u64, u64, u64)],
    failed: &[(String, String)],
    embedded: u64,
) -> String {
    let mut out = String::new();
    let mut changed = 0usize;
    for (repo, added, deleted, unchanged, ms) in repos {
        if *added > 0 || *deleted > 0 {
            changed += 1;
            let _ = writeln!(out, "{repo}: +{added} -{deleted} ={unchanged} ({ms} ms)");
        }
    }
    for (repo, err) in failed {
        let _ = writeln!(out, "{repo}: FAILED {err}");
    }
    let _ = write!(
        out,
        "-- {} repo{} checked, {changed} changed, {embedded} chunks embedded, index reloaded",
        repos.len(),
        plural(repos.len())
    );
    out
}

/// Impact report (SPEC-P9): the changed definitions, the roots'
/// definitions, then the DIRECT call sites grouped per file (at most
/// [`MAX_SITES_PER_FILE`] each, with the enclosing caller), then the
/// transitive hops. Transitive sites are listed like direct ones while they
/// are few; past [`COLLAPSE_TRANSITIVE_ABOVE`] they are summarised per
/// intermediate symbol (`via start(): 280 sites in 190 files`, top files),
/// because on a name-based call graph a hub name at hop 2 fans out to most
/// of the corpus and the agent needs the shape, not 300 lines of it.
pub fn impact(changed: Option<&[ChangedSymbol]>, r: &ImpactReport, current: Option<&str>) -> String {
    let mut out = String::new();
    if let Some(changed) = changed {
        let _ = writeln!(out, "changed definitions ({}):", changed.len());
        for c in changed {
            let scope = if c.scope.is_empty() {
                String::new()
            } else {
                format!(" [{}]", c.scope)
            };
            let lines: Vec<String> = c.lines.iter().map(|l| l.to_string()).collect();
            let _ = writeln!(
                out,
                "  {} {} {}{} lines {}",
                c.path,
                c.kind.name(),
                c.name,
                scope,
                lines.join(",")
            );
        }
    }
    let _ = writeln!(out, "roots: {}", r.roots.join(", "));
    if !r.definitions.is_empty() {
        let _ = writeln!(out, "definitions ({}):", r.definitions.len());
        let (shown, rest): (Vec<&SearchHit>, Vec<&SearchHit>) = if r.definitions.len() <= COLLAPSE_DEFS_ABOVE {
            (r.definitions.iter().collect(), Vec::new())
        } else {
            // the session's repo first (all of it, capped), then the rest
            // only as counts
            let (here, elsewhere): (Vec<&SearchHit>, Vec<&SearchHit>) =
                r.definitions.iter().partition(|d| current == Some(d.repo.as_str()));
            if here.is_empty() {
                let (a, b) = r.definitions.split_at(MAX_DEFS_SHOWN.min(r.definitions.len()));
                (a.iter().collect(), b.iter().collect())
            } else {
                let mut shown = here;
                let rest: Vec<&SearchHit> = shown.split_off(MAX_DEFS_SHOWN.min(shown.len()));
                (shown, rest.into_iter().chain(elsewhere).collect())
            }
        };
        for d in &shown {
            let _ = writeln!(out, "  {}:{}:{} {}", d.repo, d.path, d.line, snippet(&d.snippet));
        }
        if !rest.is_empty() {
            let mut by_repo: BTreeMap<&str, usize> = BTreeMap::new();
            for d in &rest {
                *by_repo.entry(d.repo.as_str()).or_insert(0) += 1;
            }
            let mut repos: Vec<(&str, usize)> = by_repo.into_iter().collect();
            repos.sort_by_key(|(name, n)| (std::cmp::Reverse(*n), *name));
            let list: Vec<String> = repos.iter().map(|(name, n)| format!("{name} {n}")).collect();
            let _ = writeln!(out, "  … {} more (pass repo: to list them): {}", rest.len(), list.join(", "));
        }
    }

    let direct: Vec<&ImpactSite> = r.sites.iter().filter(|s| s.depth <= 1).collect();
    let transitive: Vec<&ImpactSite> = r.sites.iter().filter(|s| s.depth > 1).collect();
    let n_files = |sites: &[&ImpactSite]| {
        sites
            .iter()
            .map(|s| (s.repo.as_str(), s.path.as_str()))
            .collect::<BTreeSet<_>>()
            .len()
    };
    let (nd, nt) = (n_files(&direct), n_files(&transitive));
    let _ = writeln!(
        out,
        "call sites: {} direct in {} file{}, {} transitive in {} file{}",
        direct.len(),
        nd,
        plural(nd),
        transitive.len(),
        nt,
        plural(nt)
    );
    let single_root = r.roots.len() == 1;
    let implied = |s: &ImpactSite| single_root && s.symbol == r.roots[0];
    if !direct.is_empty() {
        match current {
            // The session's repo in full; other repos collapsed to counts
            // once the direct callers are many (a hub name).
            Some(cur) if direct.len() > COLLAPSE_DIRECT_ABOVE => {
                let (here, elsewhere): (Vec<&ImpactSite>, Vec<&ImpactSite>) =
                    direct.iter().partition(|s| s.repo == cur);
                if here.is_empty() {
                    let _ = writeln!(out, "direct callers: none in {cur}");
                } else {
                    let _ = writeln!(out, "direct callers in {cur}:");
                    sites_by_file_budget(&mut out, &here, implied, MAX_DIRECT_ROWS);
                }
                if !elsewhere.is_empty() {
                    let _ = writeln!(out, "direct callers elsewhere (collapsed; pass repo: to expand):");
                    per_repo_summary(&mut out, &elsewhere);
                }
            }
            _ => {
                let _ = writeln!(out, "direct callers:");
                sites_by_file_budget(&mut out, &direct, implied, MAX_DIRECT_ROWS);
            }
        }
    }
    if !transitive.is_empty() {
        if transitive.len() <= COLLAPSE_TRANSITIVE_ABOVE {
            let _ = writeln!(out, "transitive callers (dN = hops from a root):");
            sites_by_file(&mut out, &transitive, |_| false);
        } else {
            // group by the intermediate symbol the hop went through
            let mut order: Vec<&str> = Vec::new();
            let mut by_sym: BTreeMap<&str, Vec<&ImpactSite>> = BTreeMap::new();
            for s in &transitive {
                if !by_sym.contains_key(s.symbol.as_str()) {
                    order.push(s.symbol.as_str());
                }
                by_sym.entry(s.symbol.as_str()).or_default().push(s);
            }
            order.sort_by_key(|sym| std::cmp::Reverse(by_sym[sym].len()));
            let _ = writeln!(
                out,
                "transitive callers, collapsed by the symbol they reach a root through \
                 (run impact_of_symbol on one to expand it):"
            );
            // symbols reached through a single site are grouped per file
            // at the end (`main.rs: run_all, route_command`), not one
            // `via` line each
            let mut singles: BTreeMap<(&str, &str), Vec<&str>> = BTreeMap::new();
            for sym in order {
                let sites = &by_sym[sym];
                if sites.len() == 1 {
                    let s = sites[0];
                    singles.entry((s.repo.as_str(), s.path.as_str())).or_default().push(sym);
                    continue;
                }
                let mut per_file: BTreeMap<(&str, &str), (u32, u32)> = BTreeMap::new();
                for s in sites {
                    let e = per_file
                        .entry((s.repo.as_str(), s.path.as_str()))
                        .or_insert((0, u32::MAX));
                    e.0 += 1;
                    e.1 = e.1.min(s.depth);
                }
                let mut files: Vec<((&str, &str), (u32, u32))> = per_file.into_iter().collect();
                files.sort_by_key(|(k, (n, d))| (*d, std::cmp::Reverse(*n), *k));
                let shown: Vec<String> = files
                    .iter()
                    .take(COLLAPSED_TOP_FILES)
                    .map(|((repo, path), (n, d))| format!("{repo}:{path} (d{d}, {n})"))
                    .collect();
                let more = files.len().saturating_sub(COLLAPSED_TOP_FILES);
                let _ = writeln!(
                    out,
                    "  via {}(): {} site{} in {} file{}: {}{}",
                    sym,
                    sites.len(),
                    plural(sites.len()),
                    files.len(),
                    plural(files.len()),
                    shown.join(", "),
                    if more > 0 { format!(", +{more} more") } else { String::new() }
                );
            }
            if !singles.is_empty() {
                let n: usize = singles.values().map(Vec::len).sum();
                let _ = writeln!(out, "  via one site each ({n}):");
                for ((repo, path), syms) in &singles {
                    let _ = writeln!(out, "    {repo}:{path}: {}", syms.join(", "));
                }
            }
        }
    }
    if !r.importers.is_empty() {
        let _ = writeln!(out, "importers ({}):", r.importers.len());
        for h in &r.importers {
            let _ = writeln!(out, "  {}:{}:{} {}", h.repo, h.path, h.line, snippet(&h.snippet));
        }
    }
    if r.truncated {
        out.push_str("-- truncated: a fan-out cap was hit (raise max_sites, lower depth, or pick a less common name)");
    } else {
        out.push_str("-- complete");
    }
    out
}

/// One line per repo: `repo: N sites in M files: a.rs (5), b.rs (3), +K more`.
fn per_repo_summary(out: &mut String, sites: &[&ImpactSite]) {
    let mut by_repo: BTreeMap<&str, BTreeMap<&str, u32>> = BTreeMap::new();
    for s in sites {
        *by_repo
            .entry(s.repo.as_str())
            .or_default()
            .entry(s.path.as_str())
            .or_insert(0) += 1;
    }
    let mut repos: Vec<(&str, BTreeMap<&str, u32>)> = by_repo.into_iter().collect();
    repos.sort_by_key(|(name, files)| (std::cmp::Reverse(files.values().sum::<u32>()), *name));
    for (repo, files) in repos {
        let total: u32 = files.values().sum();
        let mut list: Vec<(&str, u32)> = files.into_iter().collect();
        list.sort_by_key(|(p, n)| (std::cmp::Reverse(*n), *p));
        let shown: Vec<String> = list
            .iter()
            .take(COLLAPSED_TOP_FILES)
            .map(|(p, n)| if *n > 1 { format!("{p} ({n})") } else { p.to_string() })
            .collect();
        let more = list.len().saturating_sub(COLLAPSED_TOP_FILES);
        let _ = writeln!(
            out,
            "  {repo}: {total} site{} in {} file{}: {}{}",
            plural(total as usize),
            list.len(),
            plural(list.len()),
            shown.join(", "),
            if more > 0 { format!(", +{more} more") } else { String::new() }
        );
    }
}

/// `repo:path` headers with `  dN line: [symbol ]in caller(): snippet` rows,
/// files in first-appearance (depth-first) order, at most
/// [`MAX_SITES_PER_FILE`] rows per file. `implied(site)` says when the
/// matched symbol is obvious from context and can be left out.
fn sites_by_file(out: &mut String, sites: &[&ImpactSite], implied: impl Fn(&ImpactSite) -> bool) {
    sites_by_file_budget(out, sites, implied, usize::MAX)
}

/// `sites_by_file` that stops listing rows after `budget` and tallies the
/// remaining files as `repo:path (n)`.
fn sites_by_file_budget(
    out: &mut String,
    sites: &[&ImpactSite],
    implied: impl Fn(&ImpactSite) -> bool,
    budget: usize,
) {
    let mut order: Vec<(&str, &str)> = Vec::new();
    let mut by_file: BTreeMap<(&str, &str), Vec<&ImpactSite>> = BTreeMap::new();
    for s in sites {
        let key = (s.repo.as_str(), s.path.as_str());
        if !by_file.contains_key(&key) {
            order.push(key);
        }
        by_file.entry(key).or_default().push(s);
    }
    let mut written = 0usize;
    let mut tally: Vec<String> = Vec::new();
    for key in &order {
        let mut rows = by_file.remove(key).unwrap_or_default();
        if written >= budget {
            tally.push(if rows.len() > 1 { format!("{}:{} ({})", key.0, key.1, rows.len()) } else { format!("{}:{}", key.0, key.1) });
            continue;
        }
        rows.sort_by_key(|s| (s.depth, s.line));
        let _ = writeln!(out, "{}:{}", key.0, key.1);
        let total = rows.len();
        written += total.min(MAX_SITES_PER_FILE);
        for s in rows.into_iter().take(MAX_SITES_PER_FILE) {
            let sym = if implied(s) { String::new() } else { format!("{} ", s.symbol) };
            let caller = if s.caller.is_empty() {
                String::new()
            } else {
                format!("in {}(): ", s.caller)
            };
            let _ = writeln!(
                out,
                "  d{} {}: {}{}{}",
                s.depth,
                s.line,
                sym,
                caller,
                snippet_capped(&s.snippet, MAX_SITE_SNIPPET)
            );
        }
        if total > MAX_SITES_PER_FILE {
            let _ = writeln!(out, "  … +{} more in this file", total - MAX_SITES_PER_FILE);
        }
    }
    if !tally.is_empty() {
        let _ = writeln!(out, "  … {} more file{}: {}", tally.len(), plural(tally.len()), tally.join(", "));
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

fn human_bytes(b: u64) -> String {
    const K: f64 = 1024.0;
    let b = b as f64;
    if b >= K * K * K {
        format!("{:.1} GB", b / (K * K * K))
    } else if b >= K * K {
        format!("{:.1} MB", b / (K * K))
    } else if b >= K {
        format!("{:.0} KB", b / K)
    } else {
        format!("{b} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexio_types::Lang;

    fn hit(repo: &str, path: &str, line: u32, snip: &str) -> SearchHit {
        SearchHit {
            repo: repo.into(),
            path: path.into(),
            line,
            col: 0,
            snippet: snip.into(),
            score: 1.0,
            lang: Lang::Rust,
        }
    }

    #[test]
    fn hits_group_by_file_and_sort_lines() {
        let h = vec![
            hit("r", "b.rs", 9, "  x  "),
            hit("r", "a.rs", 20, "second"),
            hit("r", "a.rs", 3, "first"),
        ];
        let s = hits(&h, true, false, "hits");
        assert_eq!(
            s,
            "r:b.rs\n  9: x\nr:a.rs\n  3: first\n  20: second\n-- 3 hits in 2 files"
        );
        assert!(hits(&h, true, true, "hits").ends_with("(truncated; raise limit or narrow the query)"));
        assert_eq!(hits(&[], true, false, "hits"), "no hits");
    }

    #[test]
    fn call_sites_cap_rows_per_file_and_keep_line_numbers() {
        let many: Vec<SearchHit> = (1..=10).map(|i| hit("r", "a.rs", i * 10, "reg.register(x)")).collect();
        let s = call_sites(&many, false, "call sites");
        assert_eq!(s.matches("reg.register").count(), MAX_CALL_ROWS_PER_FILE);
        assert!(s.contains("… +4 more at 70 80 90 100"), "{s}");
        assert!(s.ends_with("-- 10 call sites in 1 file"));
        let few = vec![hit("r", "b.rs", 3, "f()"), hit("r", "b.rs", 1, "g()")];
        assert!(call_sites(&few, false, "call sites").starts_with("r:b.rs
  1: g()
  3: f()
"));
    }

    #[test]
    fn ranked_hits_keep_order() {
        let h = vec![hit("r", "a.rs", 20, "s"), hit("r", "b.rs", 1, "t"), hit("r", "a.rs", 3, "f")];
        let s = hits(&h, false, false, "hits");
        assert_eq!(s, "r:a.rs\n  20: s\nr:b.rs\n  1: t\nr:a.rs\n  3: f\n-- 3 hits in 3 files");
    }

    #[test]
    fn snippet_is_one_trimmed_capped_line() {
        assert_eq!(snippet("   a   b\n c"), "a b");
        let long = "x".repeat(300);
        let s = snippet(&long);
        assert!(s.len() < 210 && s.ends_with('…'));
    }

    #[test]
    fn files_fold_directories() {
        let l = vec![
            ("r".to_string(), "src/a.rs".to_string()),
            ("r".to_string(), "src/b c.rs".to_string()),
            ("r".to_string(), "src/x/c.rs".to_string()),
            ("r".to_string(), "README".to_string()),
        ];
        assert_eq!(
            files(&l, false),
            "r:src/ a.rs \"b c.rs\"\nr:src/x/ c.rs\nr: README\n-- 4 files"
        );
        assert_eq!(files(&[], false), "no files match");
    }

    #[test]
    fn span_numbers_lines_with_tabs() {
        assert_eq!(span("r", "a.rs", 3, 4, "x\ny\n"), "r:a.rs L3-4\n3\tx\n4\ty");
        assert_eq!(span("r", "a.rs", 9, 8, ""), "r:a.rs L9-8");
        // first, last and every 5th line are numbered; the rest keep the tab
        let body = (7..=16).map(|i| format!("l{i}")).collect::<Vec<_>>().join("\n");
        assert_eq!(
            span("r", "a.rs", 7, 16, &body),
            "r:a.rs L7-16\n7\tl7\n\tl8\n\tl9\n10\tl10\n\tl11\n\tl12\n\tl13\n\tl14\n15\tl15\n16\tl16"
        );
    }

    #[test]
    fn stats_groups_repos_by_sync_day_under_the_majority_root() {
        let s = EngineStats {
            repos: vec!["a".into(), "b".into(), "notes".into(), "sessions".into()],
            doc_count: 10,
            total_raw_bytes: 1000,
            shard_count: 1,
            index_bytes: 500,
            ..Default::default()
        };
        let detail = vec![
            ("a".to_string(), r"C:\gh\a".to_string(), "abc".to_string(), "2026-09-17T01:00:00Z".to_string(), false),
            ("b".to_string(), r"C:\gh\b".to_string(), "def".to_string(), "2026-09-16T01:00:00Z".to_string(), false),
            ("notes".to_string(), r"C:\docs\notes".to_string(), String::new(), "2026-09-17T02:00:00Z".to_string(), true),
            ("sessions".to_string(), r"C:\u\.indexio\sessions".to_string(), String::new(), "2026-09-17T03:00:00Z".to_string(), true),
        ];
        let t = stats(&s, &detail);
        assert!(t.contains(r"root: C:\gh\"), "{t}");
        assert!(t.contains("synced 2026-09-17: a notes* sessions*"), "{t}");
        assert!(t.contains("synced 2026-09-16: b"), "{t}");
        assert!(t.contains(r"not under root: notes=C:\docs\notes sessions=C:\u\.indexio\sessions"), "{t}");
        assert!(!t.contains("abc"), "commit hashes are for the JSON payload: {t}");
        assert!(t.contains("* plain folder"), "{t}");
    }

    #[test]
    fn human_bytes_scales() {
        assert_eq!(human_bytes(10), "10 B");
        assert_eq!(human_bytes(2048), "2 KB");
        assert_eq!(human_bytes(11_323_949), "10.8 MB");
    }
}
