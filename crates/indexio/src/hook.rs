//! `indexio hook bash` — a Claude Code PreToolUse hook for the Bash tool
//! (SPEC-P9 §17).
//!
//! Measured on a day of real sessions: two thirds of all tool-result
//! tokens were `cat` / `sed -n` / `head` / `tail` and `grep` / `rg` run
//! through Bash against files the index already held — the sessions had
//! stopped using the built-in Grep/Read tools but shelled out instead,
//! which bypasses both the guidance and the index. This hook reads the
//! harness's hook JSON on stdin, and when the command is a plain read or
//! search of something indexio can serve, answers with a `deny` whose
//! reason spells out the equivalent indexio call. Anything else — writes,
//! `sed -i`, greps on a pipeline, files the index does not have, commands
//! carrying `# raw` — is allowed untouched. Never fails the call: any
//! error means "allow".

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use indexio_query::Engine;

/// Escape hint, appended only when the SAME command is denied a second
/// time in a session (SPEC-P10): shown on every deny, the model learned it
/// from the first reason and wrote `# raw` on its next shell read instead
/// of the indexio call.
const RAW: &str = " (You already saw this: if the shell is really needed, add `# raw` to the command.)";

/// Registered repos as (name, canonical root).
fn repos(data_dir: &Path) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for name in indexio_ingest::sources::registered_repo_names(data_dir).unwrap_or_default() {
        if name == crate::sessions::REPO {
            continue;
        }
        if let Ok(st) = indexio_ingest::repo_state(data_dir, &name) {
            let root = st.path.canonicalize().unwrap_or(st.path.clone());
            out.push((name, root));
        }
    }
    out
}

/// (repo, repo-relative path with '/') for an absolute path inside a repo.
fn locate(repos: &[(String, PathBuf)], abs: &Path) -> Option<(String, String)> {
    let abs = abs.canonicalize().unwrap_or_else(|_| abs.to_path_buf());
    let mut best: Option<(usize, String, String)> = None;
    for (name, root) in repos {
        if let Ok(rel) = abs.strip_prefix(root) {
            let depth = root.components().count();
            if best.as_ref().map_or(true, |(d, _, _)| depth > *d) {
                let rel = rel.to_string_lossy().replace('\\', "/");
                best = Some((depth, name.clone(), rel));
            }
        }
    }
    best.map(|(_, n, r)| (n, r))
}

/// Split a command line into pipelines on `&&`, `||`, `;`, then each
/// pipeline into `|` stages; quotes are respected, everything is trimmed.
fn split(cmd: &str) -> Vec<Vec<String>> {
    let mut pipelines: Vec<Vec<String>> = Vec::new();
    let mut stages: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let chars: Vec<char> = cmd.chars().collect();
    let mut i = 0;
    let flush_stage = |cur: &mut String, stages: &mut Vec<String>| {
        let t = cur.trim().to_string();
        if !t.is_empty() {
            stages.push(t);
        }
        cur.clear();
    };
    while i < chars.len() {
        let c = chars[i];
        match quote {
            // a backslash escapes the next char outside quotes and inside
            // double quotes (`"\"..."`), never inside single quotes
            Some('"') | None if c == '\\' && i + 1 < chars.len() => {
                cur.push(c);
                cur.push(chars[i + 1]);
                i += 1;
            }
            Some(q) => {
                cur.push(c);
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                '"' | '\'' => {
                    quote = Some(c);
                    cur.push(c);
                }
                '&' if chars.get(i + 1) == Some(&'&') => {
                    flush_stage(&mut cur, &mut stages);
                    pipelines.push(std::mem::take(&mut stages));
                    i += 1;
                }
                '|' if chars.get(i + 1) == Some(&'|') => {
                    flush_stage(&mut cur, &mut stages);
                    pipelines.push(std::mem::take(&mut stages));
                    i += 1;
                }
                ';' | '\n' => {
                    flush_stage(&mut cur, &mut stages);
                    pipelines.push(std::mem::take(&mut stages));
                }
                '|' => flush_stage(&mut cur, &mut stages),
                _ => cur.push(c),
            },
        }
        i += 1;
    }
    flush_stage(&mut cur, &mut stages);
    if !stages.is_empty() {
        pipelines.push(stages);
    }
    pipelines.into_iter().filter(|p| !p.is_empty()).collect()
}

/// Shell words of one stage (quotes stripped, no globbing).
fn words(stage: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut had = false;
    let mut it = stage.chars().peekable();
    while let Some(c) = it.next() {
        match quote {
            // bash escapes: outside quotes `\x` is x; inside double quotes
            // only `\"`, `\\`, `\$` and `` \` `` lose the backslash (`"\|"`
            // stays `\|`, which is what grep sees)
            None if c == '\\' => {
                if let Some(n) = it.next() {
                    cur.push(n);
                }
            }
            Some('"') if c == '\\' && matches!(it.peek(), Some('"' | '\\' | '$' | '`')) => {
                cur.push(it.next().unwrap_or('\\'));
            }
            Some(q) if c == q => quote = None,
            Some(_) => cur.push(c),
            None if c == '"' || c == '\'' => {
                quote = Some(c);
                had = true;
            }
            None if c.is_whitespace() => {
                if had || !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                    had = false;
                }
            }
            None => cur.push(c),
        }
    }
    if had || !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn resolve(cwd: &Path, arg: &str) -> PathBuf {
    let p = Path::new(arg);
    if p.is_absolute() || arg.starts_with('/') {
        // Git-Bash `/c/Users/...` spelling
        if let Some(rest) = arg.strip_prefix('/') {
            let mut it = rest.splitn(2, '/');
            if let (Some(drive), Some(tail)) = (it.next(), it.next()) {
                if drive.len() == 1 && drive.chars().all(|c| c.is_ascii_alphabetic()) {
                    return PathBuf::from(format!("{}:/{}", drive.to_ascii_uppercase(), tail));
                }
            }
        }
        p.to_path_buf()
    } else {
        cwd.join(p)
    }
}

/// `>`, `>>`, `<<` outside quotes.
fn has_redirect(stage: &str) -> bool {
    let mut quote: Option<char> = None;
    let cs: Vec<char> = stage.chars().collect();
    for i in 0..cs.len() {
        match quote {
            Some(q) if cs[i] == q => quote = None,
            Some(_) => {}
            None if cs[i] == '"' || cs[i] == '\'' => quote = Some(cs[i]),
            None if cs[i] == '>' => return true,
            None if cs[i] == '<' && cs.get(i + 1) == Some(&'<') => return true,
            None => {}
        }
    }
    false
}

/// A pipeline stage that only views or filters text (so the pipeline is
/// still a read, not a program consuming the file).
fn is_viewer(stage: &str) -> bool {
    let w = words(stage);
    matches!(
        w.first().map(String::as_str),
        Some("head" | "tail" | "grep" | "rg" | "egrep" | "fgrep" | "sed" | "awk" | "cut" | "sort" | "uniq" | "wc" | "less" | "more" | "tr" | "nl" | "cat" | "column" | "tee")
    ) && !w.iter().any(|a| a == "-i")
}

/// A `sed -n 'A,Bp'` / `sed -n Ap` range.
fn sed_range(args: &[String]) -> Option<(u32, u32)> {
    for a in args {
        let a = a.trim_matches(|c| c == '\'' || c == '"');
        if let Some(body) = a.strip_suffix('p') {
            if let Some((s, e)) = body.split_once(',') {
                if let (Ok(s), Ok(e)) = (s.parse::<u32>(), e.parse::<u32>()) {
                    return Some((s, e));
                }
            } else if let Ok(s) = body.parse::<u32>() {
                return Some((s, s));
            }
        }
    }
    None
}

/// Positional (non-flag) arguments.
fn positional(args: &[String]) -> Vec<String> {
    args.iter().filter(|a| !a.starts_with('-')).cloned().collect()
}

/// A parsed grep/rg command line: the pattern, the path operands, the
/// `-A` context and whether the pattern is already extended syntax.
struct GrepArgs {
    pattern: Option<String>,
    paths: Vec<String>,
    after: Option<u32>,
    ere: bool,
}

/// grep/rg options that take a value (so the value is not a path).
const GREP_VALUE_OPTS: &[&str] = &[
    "-A", "-B", "-C", "-m", "-e", "-f", "-g", "-t", "-T", "-d", "-D", "--include", "--exclude",
    "--exclude-dir", "--glob", "--iglob", "--type", "--type-not", "--max-count", "--context",
    "--after-context", "--before-context", "--regexp", "--file", "--max-depth", "--color",
    "--colour", "--threads", "-j",
];

fn grep_args(cmd: &str, args: &[String]) -> GrepArgs {
    let mut g = GrepArgs {
        pattern: None,
        paths: Vec::new(),
        after: None,
        ere: matches!(cmd, "rg" | "egrep" | "ag" | "ack"),
    };
    let mut i = 0;
    let mut only_operands = false;
    while i < args.len() {
        let a = &args[i];
        if only_operands || !a.starts_with('-') || a == "-" {
            if g.pattern.is_none() {
                g.pattern = Some(a.clone());
            } else {
                g.paths.push(a.clone());
            }
            i += 1;
            continue;
        }
        if a == "--" {
            only_operands = true;
            i += 1;
            continue;
        }
        if a == "-E" || a == "-P" || a == "--extended-regexp" || a == "--perl-regexp" {
            g.ere = true;
        }
        // `--opt=value`, `-A3`, `-e pat`, `-A 3`
        if let Some((k, v)) = a.split_once('=') {
            if k == "--regexp" {
                g.pattern.get_or_insert(v.to_string());
            } else if k == "--after-context" || k == "--context" {
                g.after = v.parse().ok();
            }
            i += 1;
            continue;
        }
        if a.len() > 2 && !a.starts_with("--") {
            let (k, v) = a.split_at(2);
            if k == "-e" {
                g.pattern.get_or_insert(v.to_string());
                i += 1;
                continue;
            }
            if k == "-A" || k == "-C" {
                g.after = v.parse().ok();
                i += 1;
                continue;
            }
            if GREP_VALUE_OPTS.contains(&k) {
                i += 1;
                continue;
            }
            // bundled flags like `-rn`; `-rne pat` is not handled (rare)
            i += 1;
            continue;
        }
        if GREP_VALUE_OPTS.contains(&a.as_str()) {
            if let Some(v) = args.get(i + 1) {
                match a.as_str() {
                    "-e" | "--regexp" => {
                        g.pattern.get_or_insert(v.clone());
                    }
                    "-A" | "-C" | "--after-context" | "--context" => g.after = v.parse().ok(),
                    _ => {}
                }
            }
            i += 2;
            continue;
        }
        i += 1;
    }
    g
}

/// POSIX basic regex (plain `grep`) to the extended syntax indexio's
/// `/regex/` queries use: `\|`, `\(`, `\)`, `\{`, `\}`, `\+`, `\?` become
/// operators and the bare characters become literals.
fn bre_to_ere(p: &str) -> String {
    let mut out = String::with_capacity(p.len() + 4);
    let mut it = p.chars().peekable();
    while let Some(c) = it.next() {
        match c {
            '\\' => match it.next() {
                Some(n @ ('|' | '(' | ')' | '{' | '}' | '+' | '?')) => out.push(n),
                Some(n) => {
                    out.push('\\');
                    out.push(n);
                }
                None => out.push('\\'),
            },
            '|' | '(' | ')' | '{' | '}' | '+' | '?' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

/// `def NAME`, `fn NAME`, `class NAME`, `struct NAME`… → the symbol a
/// `find_symbol` answers directly (with its definition's line range).
fn definition_name(pattern: &str) -> Option<String> {
    let p = pattern.trim_start_matches('^').trim();
    let mut w = p.split_whitespace().filter(|t| !matches!(*t, "pub" | "async" | "export" | "static" | "const" | "default" | "pub(crate)"));
    let kw = w.next()?;
    if !matches!(kw, "def" | "fn" | "class" | "struct" | "enum" | "trait" | "impl" | "function" | "interface" | "type" | "func") {
        return None;
    }
    let name: String = w.next()?.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
    if name.len() < 2 || w.next().is_some() {
        return None;
    }
    Some(name)
}

/// `^class \|^def \|^async def ` (every alternative anchored at a
/// definition keyword or decorator) is a hand-rolled outline of a file:
/// `file_outline` is the exact answer.
fn is_outline_pattern(ere: &str) -> bool {
    let alts: Vec<&str> = ere.split('|').map(str::trim).filter(|a| !a.is_empty()).collect();
    !alts.is_empty()
        && alts.iter().all(|a| {
            let Some(rest) = a.strip_prefix('^') else { return false };
            let rest = rest.trim_start_matches("\\s*").trim_start_matches("\\s+").trim();
            let word: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '@' || *c == '_').collect();
            matches!(
                word.as_str(),
                "def" | "async" | "class" | "fn" | "pub" | "struct" | "enum" | "impl" | "trait" | "function"
                    | "export" | "interface" | "type" | "func" | "mod"
            ) || word.starts_with('@')
        })
}

/// Bring `repo`'s working tree into the index before a redirected lookup
/// runs (the session that issued the shell read may not be the one that
/// auto-refreshes this repo). Lexical only, a stat pass when nothing
/// changed; errors are ignored.
fn freshen(data_dir: &Path, repo: &str) {
    if let Ok(cas) = indexio_ingest::Cas::open(&data_dir.join("cas")) {
        let _ = indexio_ingest::reindex_worktree(repo, data_dir, &cas);
    }
}

/// The verdict for one Bash command: `Some(reason)` = deny.
pub fn verdict(data_dir: &Path, cwd: &Path, command: &str) -> Option<String> {
    let mut reason = judge(data_dir, cwd, command)?;
    // the model bundles several reads per call (`sed -n … ; grep -n … | head`):
    // the deny is for the servable one, the rest is fine to rerun alone
    let others = split(command)
        .iter()
        .filter(|p| p.first().map_or(false, |s| !s.starts_with("cd ") && s != "cd"))
        .count();
    if others > 1 {
        reason.push_str(" The rest of this call is fine to run on its own.");
    }
    Some(reason)
}

/// Commands the hook can serve from the index, and the harmless helpers
/// that surround them in one call. A pipeline starting with anything else
/// — `python`, `bash x.sh`, `cargo`, `git`, a heredoc, `sed -i`, a
/// redirect — is work only the shell can do.
const LOOKUP_OR_HARMLESS: &[&str] = &[
    "cat", "head", "tail", "sed", "less", "more", "nl", "grep", "rg", "egrep", "fgrep", "ag", "ack", "find",
    "ls", "dir", "wc", "sort", "uniq", "cut", "tr", "awk", "column", "echo", "printf", "true", "pwd", "cd",
    "type", "test", "[", "xargs", "tee",
];

/// Whether this pipeline is work the shell has to do (SPEC-P10 §26). A
/// call that contains such work is allowed whole even when another of its
/// pipelines is a servable lookup: denying it made the model re-run the
/// work part alone — 22 of 58 Bash denials in a day were `python patch.py
/// && grep -n … ` or a heredoc, and a re-run applies the patch twice.
fn is_other_work(pipeline: &[String]) -> bool {
    if pipeline.iter().any(|st| has_redirect(st)) {
        return true;
    }
    let Some(first) = pipeline.first() else { return false };
    let w = words(first);
    let mut cmd = w.first().map(String::as_str).unwrap_or("");
    let mut args: &[String] = if w.is_empty() { &[] } else { &w[1..] };
    if cmd == "rtk" {
        // `rtk <cmd>` (the rtk hook's rewrite) or `rtk proxy <cmd>`
        let skip = usize::from(args.first().map(String::as_str) == Some("proxy"));
        cmd = args.get(skip).map(String::as_str).unwrap_or("");
        args = args.get(skip + 1..).unwrap_or(&[]);
    }
    if cmd == "sed" && args.iter().any(|a| a == "-i" || a.starts_with("-i")) {
        return true;
    }
    !cmd.is_empty() && !LOOKUP_OR_HARMLESS.contains(&cmd)
}

fn judge(data_dir: &Path, cwd: &Path, command: &str) -> Option<String> {
    if command.contains("INDEXIO_RAW") {
        return None;
    }
    // the shell is needed anyway: let the whole call through
    if split(command).iter().any(|p| is_other_work(p)) {
        return None;
    }
    let repos = repos(data_dir);
    let debug = std::env::var_os("INDEXIO_HOOK_DEBUG").is_some();
    if debug {
        eprintln!("hook: cwd={} repos={} cmd={command:?}", cwd.display(), repos.len());
    }
    if repos.is_empty() {
        return None;
    }
    let mut engine: Option<Engine> = None;
    let mut indexed = |repo: &str, rel: &str| -> bool {
        if engine.is_none() {
            engine = Engine::open(data_dir).ok();
        }
        engine.as_ref().map_or(false, |e| e.outline(repo, rel).is_some())
    };
    let mut cwd = cwd.to_path_buf();
    for pipeline in split(command) {
        let Some(first) = pipeline.first() else { continue };
        // A redirection or heredoc makes it a write (`cat > f <<EOF`), and a
        // read whose output feeds a program (`cat f | python -`) is not a
        // lookup: only reads that end in the terminal or in viewers/filters
        // are the index's business.
        if pipeline.iter().any(|st| has_redirect(st)) {
            continue;
        }
        if pipeline[1..].iter().any(|st| !is_viewer(st)) {
            continue;
        }
        let w = words(first);
        let Some(cmd0) = w.first() else { continue };
        let mut cmd = cmd0.as_str();
        let mut args: &[String] = &w[1..];
        if cmd == "rtk" && args.first().map(String::as_str) == Some("proxy") {
            // `rtk proxy <cmd>`: judge the wrapped command
            if let Some(c) = args.get(1) {
                cmd = c.as_str();
                args = &args[2..];
            }
        }
        match cmd {
            "cd" => {
                if let Some(dir) = args.first() {
                    cwd = resolve(&cwd, dir);
                }
                continue;
            }
            "cat" | "head" | "tail" | "sed" | "less" | "more" => {
                if cmd == "sed" && args.iter().any(|a| a == "-i" || a.starts_with("-i")) {
                    continue; // an edit
                }
                let files: Vec<(String, String, PathBuf)> = positional(args)
                    .into_iter()
                    .filter(|a| !(cmd == "sed" && a.ends_with('p')))
                    .map(|a| (a.clone(), a, PathBuf::new()))
                    .map(|(a, _, _)| {
                        let abs = resolve(&cwd, &a);
                        (a, String::new(), abs)
                    })
                    .collect();
                for (a, _, abs) in files {
                    if debug {
                        eprintln!("hook: read candidate {a} -> {} file={} located={:?}", abs.display(), abs.is_file(), locate(&repos, &abs));
                    }
                    if !abs.is_file() {
                        continue;
                    }
                    let Some((repo, rel)) = locate(&repos, &abs) else { continue };
                    if !indexed(&repo, &rel) {
                        if debug {
                            eprintln!("hook: {repo}:{rel} not indexed");
                        }
                        continue;
                    }
                    let range = if cmd == "sed" {
                        sed_range(args)
                    } else if cmd == "head" {
                        args.iter()
                            .position(|x| x == "-n")
                            .and_then(|i| args.get(i + 1))
                            .and_then(|n| n.parse::<u32>().ok())
                            .or_else(|| args.iter().find_map(|x| x.strip_prefix("-n").and_then(|n| n.parse().ok())))
                            .map(|n| (1, n))
                    } else {
                        None
                    };
                    let call = match range {
                        Some((s, e)) => format!(
                            "mcp__indexio__read_span {{repo:\"{repo}\", path:\"{rel}\", start:{s}, end:{e}}}"
                        ),
                        None => format!(
                            "mcp__indexio__file_outline {{repo:\"{repo}\", path:\"{rel}\"}} then read_span for the lines you need"
                        ),
                    };
                    freshen(data_dir, &repo);
                    return Some(format!("{a} is indexed: use {call} instead (numbered, fresh)"));
                }
            }
            "grep" | "rg" | "egrep" | "fgrep" | "ag" | "ack" => {
                // a grep on a pipeline (`x | grep`) is never first here; this
                // is a filesystem search: pattern = first positional, paths =
                // the rest (none = cwd)
                let g = grep_args(cmd, args);
                let Some(pattern) = g.pattern.as_deref() else { continue };
                let paths: Vec<PathBuf> = if !g.paths.is_empty() {
                    g.paths.iter().map(|p| resolve(&cwd, p)).collect()
                } else {
                    vec![cwd.clone()]
                };
                let mut targets: Vec<(String, String)> = Vec::new();
                for p in &paths {
                    match locate(&repos, p) {
                        // a file must itself be indexed (a .log inside the repo is not)
                        Some((repo, rel)) if p.is_file() && !indexed(&repo, &rel) => return None,
                        Some(t) => targets.push(t),
                        None => return None, // something outside the index: allow
                    }
                }
                let repo_filter = {
                    let mut names: Vec<&str> = targets.iter().map(|(r, _)| r.as_str()).collect();
                    names.sort();
                    names.dedup();
                    if names.len() == 1 { format!(" repo:{}", names[0]) } else { String::new() }
                };
                // files/dirs → `path:a|b|c` (any-of substrings); a whole
                // repo → no path filter
                let path_filter = {
                    let mut rels: Vec<&str> = targets.iter().map(|(_, r)| r.as_str()).filter(|r| !r.is_empty()).collect();
                    rels.sort();
                    rels.dedup();
                    if rels.is_empty() || rels.len() < targets.len() {
                        String::new()
                    } else {
                        format!(" path:{}", rels.join("|"))
                    }
                };
                for (repo, _) in &targets {
                    freshen(data_dir, repo);
                }
                let fixed = args.iter().any(|a| a == "-F" || a == "--fixed-strings");
                let ere = if g.ere { pattern.to_string() } else { bre_to_ere(pattern) };
                let is_regex = !fixed && ere.chars().any(|c| "\\^$.*+?()[]{}|".contains(c));
                // `grep "def foo" -A 12` is a definition lookup: find_symbol
                // gives the definition's line range in one call
                if let Some(name) = definition_name(pattern) {
                    let body = match g.after {
                        Some(n) => format!(", then read_span {{repo, path, start, end:start+{n}}} for the body"),
                        None => " (rows carry start-end lines; read_span reads the body)".to_string(),
                    };
                    return Some(format!(
                        "Definition lookup in an indexed repo: use mcp__indexio__find_symbol {{name:\"{name}\"}} instead{body}"
                    ));
                }
                // `grep "^class \|^def " file` is an outline by hand
                if let [(repo, rel)] = targets.as_slice() {
                    if !rel.is_empty() && is_outline_pattern(&ere) {
                        return Some(format!(
                            "Outline of an indexed file: use mcp__indexio__file_outline {{repo:\"{repo}\", path:\"{rel}\"}} instead (every definition with start-end lines)"
                        ));
                    }
                }
                // a grep lists every matching line: code_search with
                // `lines` (literal) or code_grep (regex) do the same, ranked
                // per file
                let call = if !is_regex {
                    let lit = if fixed { pattern.to_string() } else { ere.replace('\\', "") };
                    format!(
                        "mcp__indexio__code_search {{query:\"{}\"{repo_filter}{path_filter}, lines:20}}",
                        lit.replace('"', "\\\"")
                    )
                } else {
                    format!(
                        "mcp__indexio__code_grep {{pattern:\"{}{repo_filter}{path_filter}\"}}",
                        ere.replace('"', "\\\"")
                    )
                };
                let ctx = match g.after {
                    Some(n) => format!(", then read_span {{repo, path, start:line, end:line+{n}}} for context"),
                    None => String::new(),
                };
                return Some(format!(
                    "Indexed files: use {call} instead (every matching line per file, ranked, current with your edits{ctx})"
                ));
            }
            "find" => {
                let pos = positional(args);
                let dir = pos.first().map(|d| resolve(&cwd, d)).unwrap_or_else(|| cwd.clone());
                if let Some((repo, rel)) = locate(&repos, &dir) {
                    let name = args
                        .iter()
                        .position(|a| a == "-name" || a == "-iname")
                        .and_then(|i| args.get(i + 1))
                        .cloned()
                        .unwrap_or_else(|| "*".to_string());
                    let pattern = if rel.is_empty() { format!("**/{name}") } else { format!("{rel}/**/{name}") };
                    return Some(format!(
                        "Indexed folder: use mcp__indexio__list_files {{pattern:\"{pattern}\", repo:\"{repo}\"}} instead"
                    ));
                }
            }
            _ => {}
        }
    }
    None
}

/// The command string Claude Code should run for this binary. Claude Code
/// spawns hook commands through Git Bash on Windows too, where a backslash
/// path collapses (`C:\Users\x\indexio.exe` → `C:Usersxindexio.exe: command
/// not found` on every call, and a non-blocking hook error is the only
/// trace): forward slashes work in every shell, and a path with spaces is
/// quoted.
pub fn hook_command(exe: &Path, args: &str) -> String {
    let exe = exe.to_string_lossy().replace('\\', "/");
    if exe.contains(' ') {
        format!("\"{exe}\" {args}")
    } else {
        format!("{exe} {args}")
    }
}

/// The hooks this binary provides: (event, tool matcher, args, timeout s).
const HOOKS: [(&str, Option<&str>, &str, u64); 3] = [
    ("PreToolUse", Some("Bash"), "hook bash", 10),
    ("PreToolUse", Some("Read"), "hook read", 10),
    ("PreCompact", None, "sessions --quiet", 120),
];

/// Add (or repair) indexio's hooks in a Claude Code `settings.json` value;
/// returns one line per change, none when everything was already right.
/// An existing entry is recognised by `indexio` plus the hook's args in its
/// command, so a stale or shell-hostile path is rewritten in place and the
/// user's other hooks are left alone.
pub fn install_settings(settings: &mut Value, exe: &Path) -> Vec<String> {
    let mut changes = Vec::new();
    if !settings.is_object() {
        *settings = json!({});
    }
    let hooks = settings
        .as_object_mut()
        .unwrap()
        .entry("hooks")
        .or_insert_with(|| json!({}));
    if !hooks.is_object() {
        *hooks = json!({});
    }
    for (event, matcher, args, timeout) in HOOKS {
        let want = hook_command(exe, args);
        let groups = hooks
            .as_object_mut()
            .unwrap()
            .entry(event)
            .or_insert_with(|| json!([]));
        if !groups.is_array() {
            *groups = json!([]);
        }
        let mut found = false;
        for group in groups.as_array_mut().unwrap() {
            let Some(list) = group.get_mut("hooks").and_then(Value::as_array_mut) else { continue };
            for h in list.iter_mut() {
                let cmd = h["command"].as_str().unwrap_or("").to_string();
                if !(cmd.contains("indexio") && cmd.contains(args)) {
                    continue;
                }
                found = true;
                if cmd != want {
                    changes.push(format!("{event}: `{cmd}` -> `{want}`"));
                    h["command"] = json!(want);
                }
                if h.get("timeout").is_none() {
                    h["timeout"] = json!(timeout);
                }
            }
        }
        if !found {
            let mut group = json!({ "hooks": [{ "type": "command", "command": want, "timeout": timeout }] });
            if let Some(m) = matcher {
                group["matcher"] = json!(m);
            }
            groups.as_array_mut().unwrap().push(group);
            changes.push(format!("{event}: added `{want}`"));
        }
    }
    changes
}

/// `indexio hook install`: write the hooks into `<claude dir>/settings.json`.
pub fn install(claude_dir: &Path, exe: &Path, print_only: bool) -> anyhow::Result<()> {
    let path = claude_dir.join("settings.json");
    let mut settings: Value = match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => json!({}),
        Err(e) => return Err(anyhow::anyhow!("{}: {e}", path.display())),
    };
    let changes = install_settings(&mut settings, exe);
    if changes.is_empty() {
        println!("{}: indexio hooks already installed", path.display());
        return Ok(());
    }
    for c in &changes {
        println!("{c}");
    }
    if print_only {
        println!("(print-only: {} not written)", path.display());
        return Ok(());
    }
    std::fs::create_dir_all(claude_dir)?;
    std::fs::write(&path, format!("{}\n", serde_json::to_string_pretty(&settings)?))?;
    println!("wrote {} (Claude Code picks hook changes up on the next tool call)", path.display());
    Ok(())
}

/// `<data_dir>/hook/<session>.last`: the hash of the last denied command.
fn last_denied_path(data_dir: &Path, session: &str) -> PathBuf {
    let safe: String = session.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-').take(64).collect();
    data_dir.join("hook").join(format!("{safe}.last"))
}

/// The command without its `# raw` marker (and surrounding blanks), the
/// form whose denial the marker is answering.
fn without_raw(command: &str) -> String {
    let mut s = command.replace("# raw", "");
    while s.contains("  ") {
        s = s.replace("  ", " ");
    }
    s.trim().to_string()
}

/// Whether `# raw` on this command is a retry of the command the hook last
/// denied in `session` — the only case the marker escapes the hook
/// (SPEC-P10 §13: a model that had seen the marker once put it on every
/// later grep, unprompted).
fn raw_is_retry(data_dir: &Path, session: &str, command: &str) -> bool {
    if !command.contains("# raw") {
        return false;
    }
    let hash = indexio_types::BlobId::from_content(without_raw(command).as_bytes()).hex();
    std::fs::read_to_string(last_denied_path(data_dir, session)).map_or(false, |prev| prev.trim() == hash)
}

/// The verdict for one `Read` tool call (SPEC-P10 §16): a file the index
/// holds is served by `file_outline` + `read_span` instead — a whole-file
/// Read of a source file was the largest single result in a session hour
/// (2,000 lines by default). `offset`/`limit` (1-based line, count) map to
/// a `read_span` range; without them the outline comes first. Files the
/// index does not have (images, logs, task outputs, unindexed folders) are
/// allowed untouched.
pub fn read_verdict(data_dir: &Path, file_path: &str, offset: Option<u64>, limit: Option<u64>) -> Option<String> {
    let abs = PathBuf::from(file_path);
    if !abs.is_file() {
        return None;
    }
    let repos = repos(data_dir);
    let (repo, rel) = locate(&repos, &abs)?;
    let engine = Engine::open(data_dir).ok()?;
    engine.outline(&repo, &rel)?;
    freshen(data_dir, &repo);
    let name = abs.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_else(|| rel.clone());
    Some(match (offset, limit) {
        (Some(start), Some(n)) if n > 0 => format!(
            "{name} is indexed: use mcp__indexio__read_span {{repo:\"{repo}\", path:\"{rel}\", start:{start}, end:{}}} instead (numbered, fresh)",
            start.max(1) + n - 1
        ),
        (Some(start), None) => format!(
            "{name} is indexed: use mcp__indexio__read_span {{repo:\"{repo}\", path:\"{rel}\", start:{}}} instead (the definition at that line, numbered, fresh)",
            start.max(1)
        ),
        _ => format!(
            "{name} is indexed: use mcp__indexio__file_outline {{repo:\"{repo}\", path:\"{rel}\"}} then read_span for the lines you need, instead of the whole file"
        ),
    })
}

/// Entry point for `hook read`: hook JSON on stdin, decision JSON on stdout.
pub fn run_read_hook(data_dir: &Path) -> anyhow::Result<()> {
    let mut input = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)?;
    let v: Value = serde_json::from_str(&input).unwrap_or(Value::Null);
    if v["tool_name"].as_str() != Some("Read") {
        return Ok(());
    }
    let Some(file_path) = v["tool_input"]["file_path"].as_str() else { return Ok(()) };
    let offset = v["tool_input"]["offset"].as_u64();
    let limit = v["tool_input"]["limit"].as_u64();
    if let Some(reason) = read_verdict(data_dir, file_path, offset, limit) {
        println!(
            "{}",
            json!({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            })
        );
    }
    Ok(())
}

/// Whether this exact command was the last one denied in `session`; records
/// it either way.
fn repeated_denial(data_dir: &Path, session: &str, command: &str) -> bool {
    let path = last_denied_path(data_dir, session);
    let hash = indexio_types::BlobId::from_content(without_raw(command).as_bytes()).hex();
    let same = std::fs::read_to_string(&path).map_or(false, |prev| prev.trim() == hash);
    if !same {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(&path, hash);
    }
    same
}

/// Commands whose output belongs in the runs source rather than in the
/// model's context (SPEC-P10 §31): scripts and builds that print at
/// length. `cargo`, `git` and the other commands rtk rewrites are left to
/// it (its rewrite and ours would race on the same call).
const RUNNERS: &[&str] = &[
    "python", "python3", "py", "node", "npx", "npm", "pnpm", "yarn", "bun", "deno", "pytest", "go", "make",
    "gradle", "gradlew", "mvn", "dotnet", "tsc", "java", "ruby", "bundle", "php", "swift", "flutter", "dart",
    "expo", "turbo", "vitest", "jest", "mocha", "cmake", "ninja", "zig",
];

/// The `updatedInput` command that runs `command` through `indexio run`,
/// or `None` when it should run as typed: a pipeline or redirect, a
/// command already wrapped, an rtk-handled one, or anything outside
/// [`RUNNERS`]. A leading `cd X &&` is kept in front.
pub fn runner_rewrite(exe: &Path, command: &str) -> Option<String> {
    if command.contains("indexio") || command.contains("# raw") || command.contains("INDEXIO_RAW") {
        return None;
    }
    let pipelines = split(command);
    let (cds, last) = pipelines.split_at(pipelines.len().checked_sub(1)?);
    let last = last.first()?;
    if last.len() != 1 || has_redirect(&last[0]) || last[0].contains("<<") {
        return None;
    }
    if !cds.iter().all(|p| p.len() == 1 && (p[0] == "cd" || p[0].starts_with("cd "))) {
        return None;
    }
    let w = words(&last[0]);
    let head = w.first()?;
    let base = head.rsplit(['/', '\\']).next().unwrap_or(head).trim_end_matches(".exe");
    if !RUNNERS.contains(&base) {
        return None;
    }
    // `python -c '…'` and REPL-style one-liners print little: leave them
    if base.starts_with("python") && w.get(1).is_some_and(|a| a == "-c") {
        return None;
    }
    let prefix = if cds.is_empty() {
        String::new()
    } else {
        // the original text of the cd stages, in order
        let mut s = String::new();
        for p in cds {
            s.push_str(&p[0]);
            s.push_str(" && ");
        }
        s
    };
    // the whole stage as ONE argument, so its own quotes survive the shell
    let quoted = format!("'{}'", last[0].replace('\'', "'\\''"));
    Some(format!("{prefix}{} run -- {quoted}", hook_command(exe, "").trim_end()))
}

/// Entry point: hook JSON on stdin, decision JSON on stdout, exit 0.
pub fn run_bash_hook(data_dir: &Path) -> anyhow::Result<()> {
    let mut input = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)?;
    let v: Value = serde_json::from_str(&input).unwrap_or(Value::Null);
    if v["tool_name"].as_str() != Some("Bash") {
        return Ok(());
    }
    let command = v["tool_input"]["command"].as_str().unwrap_or("");
    let cwd = v["cwd"]
        .as_str()
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_default();
    let session = v["session_id"].as_str().unwrap_or("nosession");
    if raw_is_retry(data_dir, session, command) {
        return Ok(());
    }
    // SPEC-P10 §31: a script or build runs through `indexio run`, which
    // keeps a long output in the runs source and returns a digest
    if let Some(exe) = std::env::current_exe().ok() {
        if let Some(rewritten) = runner_rewrite(&exe, command) {
            println!(
                "{}",
                json!({
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecisionReason": "indexio run: long output is stored and digested",
                        "updatedInput": { "command": rewritten },
                    }
                })
            );
            return Ok(());
        }
    }
    // judge the command as the shell will run it: the marker is a comment
    let judged = without_raw(command);
    if let Some(mut reason) = verdict(data_dir, &cwd, &judged) {
        if repeated_denial(data_dir, session, command) {
            reason.push_str(RAW);
        }
        println!(
            "{}",
            json!({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            })
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_pipelines_and_stages() {
        let p = split("cd x && cat a.rs | head -5; grep -n foo src");
        assert_eq!(p.len(), 3);
        assert_eq!(p[0], vec!["cd x"]);
        assert_eq!(p[1], vec!["cat a.rs", "head -5"]);
        assert_eq!(p[2], vec!["grep -n foo src"]);
        let q = split("echo 'a; b | c' && ls");
        assert_eq!(q[0], vec!["echo 'a; b | c'"]);
    }

    #[test]
    fn words_strip_quotes() {
        assert_eq!(words("sed -n '10,20p' \"my file.rs\""), vec!["sed", "-n", "10,20p", "my file.rs"]);
    }

    #[test]
    fn sed_ranges() {
        assert_eq!(sed_range(&["-n".into(), "10,20p".into()]), Some((10, 20)));
        assert_eq!(sed_range(&["-n".into(), "'7p'".into()]), Some((7, 7)));
        assert_eq!(sed_range(&["-n".into(), "s/a/b/p".into()]), None);
    }

    #[test]
    fn redirects_and_consumers_are_not_reads() {
        assert!(has_redirect("cat > out.txt"));
        assert!(has_redirect("cat file <<EOF"));
        assert!(!has_redirect("echo '>' x"));
        assert!(is_viewer("head -20"));
        assert!(is_viewer("grep -n foo"));
        assert!(!is_viewer("python -"));
        assert!(!is_viewer("sed -i s/a/b/"));
    }

    #[test]
    fn grep_option_values_are_not_paths() {
        let a: Vec<String> = ["-n", "class Blocked", "-A", "12", "src/errors.py"].iter().map(|s| s.to_string()).collect();
        let g = grep_args("grep", &a);
        assert_eq!(g.pattern.as_deref(), Some("class Blocked"));
        assert_eq!(g.paths, vec!["src/errors.py"]);
        assert_eq!(g.after, Some(12));
        assert!(!g.ere);
        let b: Vec<String> = ["-rn", "--include=*.rs", "-e", "foo|bar", "-A3", "crates", "docs"].iter().map(|s| s.to_string()).collect();
        let g = grep_args("rg", &b);
        assert_eq!(g.pattern.as_deref(), Some("foo|bar"));
        assert_eq!(g.paths, vec!["crates", "docs"]);
        assert_eq!(g.after, Some(3));
        assert!(g.ere);
    }

    #[test]
    fn basic_regex_becomes_extended() {
        assert_eq!(bre_to_ere("raise Blocked\\|Blocked("), "raise Blocked|Blocked\\(");
        assert_eq!(bre_to_ere("^def _x\\(self"), "^def _x(self");
        assert_eq!(bre_to_ere("a\\+b"), "a+b");
        assert_eq!(bre_to_ere("plain_name"), "plain_name");
    }

    #[test]
    fn definition_patterns_name_the_symbol() {
        assert_eq!(definition_name("def _primary_kind").as_deref(), Some("_primary_kind"));
        assert_eq!(definition_name("^async def test_fetch").as_deref(), Some("test_fetch"));
        assert_eq!(definition_name("pub fn reindex_worktree(").as_deref(), Some("reindex_worktree"));
        assert_eq!(definition_name("class Blocked").as_deref(), Some("Blocked"));
        assert_eq!(definition_name("raise Blocked"), None);
        assert_eq!(definition_name("def"), None);
        assert_eq!(definition_name("def a b"), None);
    }

    #[test]
    fn escaped_quotes_do_not_end_the_word() {
        // bash: grep -n "add(\"--id\"\|x" f | head  →  pattern add("--id"\|x
        let cmd = "grep -n \"add(\\\"--id\\\"\\|x\" src/a.py | head -3; grep -n y b.py";
        let p = split(cmd);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].len(), 2, "{:?}", p);
        let w = words(&p[0][0]);
        assert_eq!(w, vec!["grep", "-n", "add(\"--id\"\\|x", "src/a.py"]);
        assert_eq!(words("grep -n foo\\ bar f"), vec!["grep", "-n", "foo bar", "f"]);
        assert_eq!(words("grep 'a\\|b' f"), vec!["grep", "a\\|b", "f"]);
    }

    #[test]
    fn outline_patterns() {
        assert!(is_outline_pattern("^class |^def |^async def |^@pytest"));
        assert!(is_outline_pattern("^pub fn |^impl |^struct "));
        assert!(!is_outline_pattern("^class Foo|bar"));
        assert!(!is_outline_pattern("def test_"));
    }

    #[test]
    fn install_repairs_backslash_paths_and_keeps_other_hooks() {
        let exe = Path::new("C:\\Users\\x\\.cargo\\bin\\indexio.exe");
        let mut s = json!({
            "permissions": {"allow": ["mcp__indexio"]},
            "hooks": {"PreToolUse": [
                {"matcher": "Bash", "hooks": [{"type": "command", "command": "rtk hook claude"}]},
                {"matcher": "Bash", "hooks": [{"type": "command",
                    "command": "C:\\Users\\x\\.cargo\\bin\\indexio.exe hook bash", "timeout": 10}]}
            ]}
        });
        let changes = install_settings(&mut s, exe);
        assert_eq!(changes.len(), 3, "{changes:?}");
        let pre = s["hooks"]["PreToolUse"].as_array().unwrap();
        assert_eq!(pre.len(), 3);
        assert_eq!(pre[0]["hooks"][0]["command"], "rtk hook claude");
        assert_eq!(pre[1]["hooks"][0]["command"], "C:/Users/x/.cargo/bin/indexio.exe hook bash");
        assert_eq!(pre[2]["matcher"], "Read");
        assert_eq!(pre[2]["hooks"][0]["command"], "C:/Users/x/.cargo/bin/indexio.exe hook read");
        assert_eq!(s["hooks"]["PreCompact"][0]["hooks"][0]["command"], "C:/Users/x/.cargo/bin/indexio.exe sessions --quiet");
        assert!(s["hooks"]["PreCompact"][0].get("matcher").is_none());
        assert_eq!(s["permissions"]["allow"][0], "mcp__indexio");
        // idempotent
        assert!(install_settings(&mut s, exe).is_empty());
        // spaces are quoted
        assert_eq!(hook_command(Path::new("C:\\Program Files\\indexio.exe"), "hook bash"), "\"C:/Program Files/indexio.exe\" hook bash");
    }

    #[test]
    fn raw_marker_only_escapes_a_retry() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        let cmd = "sed -n 1,5p src/lib.rs";
        // a fresh `# raw` on a never-denied command is not a retry
        assert!(!raw_is_retry(d, "s1", "sed -n 1,5p src/lib.rs # raw"));
        // first denial records the command; the second identical one repeats
        assert!(!repeated_denial(d, "s1", cmd));
        assert!(repeated_denial(d, "s1", cmd));
        // now the marker on that command is a retry, on another it is not
        assert!(raw_is_retry(d, "s1", "sed -n 1,5p src/lib.rs # raw"));
        assert!(raw_is_retry(d, "s1", "sed -n 1,5p src/lib.rs  # raw"));
        assert!(!raw_is_retry(d, "s1", "sed -n 1,9p src/lib.rs # raw"));
        assert!(!raw_is_retry(d, "s2", "sed -n 1,5p src/lib.rs # raw"), "per session");
        assert_eq!(without_raw("grep -n foo a.rs | head # raw"), "grep -n foo a.rs | head");
    }

    /// SPEC-P10 §26: a call with a non-lookup pipeline is the shell's.
    #[test]
    fn calls_with_real_work_are_left_to_the_shell() {
        let mixed = |cmd: &str| split(cmd).iter().any(|p| is_other_work(p));
        assert!(mixed("python - <<'PY'\nimport io\ngrep = 1\nPY\ngrep -n foo src"));
        assert!(mixed("python patch.py && grep -n \"repo_runs\" crates/x.rs"));
        assert!(mixed("sed -i s/a/b/ f.rs && cat f.rs"));
        assert!(mixed("rtk git status; cat a.rs"));
        assert!(mixed("rtk proxy cargo build 2>&1 | grep error"));
        assert!(mixed("grep -n foo src > out.txt"));
        assert!(!mixed("cat a.rs; grep -n foo src"));
        assert!(!mixed("cd x && cat a.rs | head -5"));
        assert!(!mixed("wc -l a.py && grep -n foo a.py; echo ---"));
        assert!(!mixed("rtk grep -n foo src"));
    }

    /// SPEC-P10 §31: scripts and builds are routed through `indexio run`;
    /// pipelines, redirects, rtk-handled and already-wrapped commands are not.
    #[test]
    fn runners_are_routed_through_indexio_run() {
        let exe = Path::new("C:\\Users\\x\\.cargo\\bin\\indexio.exe");
        let rw = |c: &str| runner_rewrite(exe, c);
        assert_eq!(rw("python scripts/probe.py --fast").as_deref(), Some("C:/Users/x/.cargo/bin/indexio.exe run -- 'python scripts/probe.py --fast'"));
        assert_eq!(rw("cd \"C:/x\" && npm test").as_deref(), Some("cd \"C:/x\" && C:/Users/x/.cargo/bin/indexio.exe run -- 'npm test'"));
        assert_eq!(rw("node -e 'x(1)'").as_deref(), Some(r"C:/Users/x/.cargo/bin/indexio.exe run -- 'node -e '\''x(1)'\'''"), "inner quotes escaped");
        assert_eq!(rw("cargo test -p x"), None, "rtk's");
        assert_eq!(rw("git status"), None);
        assert_eq!(rw("python x.py | head -5"), None, "pipeline");
        assert_eq!(rw("python x.py > out.txt"), None, "redirect");
        assert_eq!(rw("python -c 'print(1)'"), None, "one-liner");
        assert_eq!(rw("C:/Users/x/.cargo/bin/indexio.exe run -- python x.py"), None, "already wrapped");
        assert_eq!(rw("python x.py # raw"), None);
        assert_eq!(rw("ls; python x.py"), None, "a non-cd stage before the runner");
    }

    #[test]
    fn git_bash_paths_resolve() {
        assert_eq!(resolve(Path::new("C:/x"), "/c/Users/me/a.rs"), PathBuf::from("C:/Users/me/a.rs"));
        assert_eq!(resolve(Path::new("C:/x"), "src/a.rs"), PathBuf::from("C:/x").join("src/a.rs"));
    }
}
