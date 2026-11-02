//! `indexio hook bash` — a Claude Code PreToolUse hook for the Bash tool
//! (SPEC-P9 §17).
//!
//! Measured on a day of real sessions: two thirds of all tool-result
//! tokens were `cat` / `sed -n` / `head` / `tail` and `grep` / `rg` run
//! through Bash against files the index already held — the sessions had
//! stopped using the built-in Grep/Read tools but shelled out instead,
//! which bypasses both the guidance and the index. This hook reads the
//! harness's hook JSON on stdin, and when a pipeline of the command is a
//! plain read or search the index answers exactly, rewrites that pipeline
//! to `indexio tool …` (`{ indexio tool … || <typed pipeline>; }`) and
//! allows the call: the answer arrives in the same round trip, and when
//! the index cannot answer the typed command runs. A deny would cost a
//! round trip and strand callers that cannot call the indexio tools (MCP
//! server down, deferred or missing in a subagent). Anything else — writes,
//! `sed -i`, filters the index cannot reproduce, files the index does not
//! have, commands carrying `# raw` — is left untouched. Never fails the
//! call: any error means "allow".

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use indexio_query::Engine;

#[path = "hook_ps.rs"]
mod ps;

/// Registered repos as (name, root as registered). [`locate`] matches
/// against these first and canonicalizes them only when that fails:
/// canonicalizing all ~50 roots cost every hook call a few milliseconds.
fn repos(data_dir: &Path) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for name in indexio_ingest::sources::registered_repo_names(data_dir).unwrap_or_default() {
        if name == crate::sessions::REPO {
            continue;
        }
        if let Ok(st) = indexio_ingest::repo_state(data_dir, &name) {
            out.push((name, st.path));
        }
    }
    out
}

/// A canonical path without the Windows verbatim prefix (`\\?\C:\..`
/// -> `C:\..`), comparable with a root as it was registered.
fn plain(p: &Path) -> PathBuf {
    let s = p.to_string_lossy();
    match s.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC\\") => PathBuf::from(rest),
        _ => p.to_path_buf(),
    }
}

/// The path whose indexed contents a read of `abs` (at `rel` in `repo`)
/// gets: `rel` itself, or for a file in a Claude Code agent worktree
/// (`<repo>/.claude/worktrees/<name>/<path>`, a second checkout the index
/// does not hold) `<path>` when the file there is byte for byte the indexed
/// one (823 worktree Reads, 7.2M chars, went past the hooks in two weeks;
/// an edited file stays the agent's own and is left to the Read tool).
pub(crate) fn indexed_rel(engine: &Engine, repo: &str, rel: &str, abs: &Path) -> Option<String> {
    if contents_indexed(engine, repo, rel) {
        return Some(rel.to_string());
    }
    let (_, twin) = rel.strip_prefix(".claude/worktrees/")?.split_once('/')?;
    let indexed = engine.file_content(repo, twin)?;
    let disk = std::fs::read(abs).ok()?;
    let lf = |b: &[u8]| -> Vec<u8> {
        let mut out = Vec::with_capacity(b.len());
        for (i, &c) in b.iter().enumerate() {
            if !(c == b'\r' && b.get(i + 1) == Some(&b'\n')) {
                out.push(c);
            }
        }
        out
    };
    (lf(&disk) == lf(&indexed) && contents_indexed(engine, repo, twin)).then(|| twin.to_string())
}

/// Whether the index holds `rel`'s CONTENTS as the file itself: a name-only
/// entry (a document, a text file over the cap) and a PDF's extracted text
/// layer stand in for the file, and the file itself is what a Read or a
/// `cat` of it should get.
fn contents_indexed(engine: &Engine, repo: &str, rel: &str) -> bool {
    // one scoped lookup: a hook opens a fresh engine per call, and building
    // the whole (repo, path) map plus an outline parse to answer this took
    // ~30 ms of a deny
    let set = engine.shard_set();
    let one: std::collections::HashSet<String> = [rel.to_string()].into_iter().collect();
    let Some((si, docid, _)) = set.visible_at(repo, &one).into_iter().next() else {
        return false;
    };
    let Ok(content) = set.content(si, docid) else {
        return true;
    };
    let first = content.split(|&b| b == b'\n').next().unwrap_or_default();
    let first = String::from_utf8_lossy(first.strip_suffix(b"\r").unwrap_or(first));
    !(first.contains(indexio_ingest::NAME_ONLY_MARKER) || first.contains(indexio_ingest::PDF_TEXT_MARKER))
}

/// The last `n` lines of an indexed file as a 1-based range (the index has
/// the line count), so `tail -n 12 f` / `Get-Content f -Tail 12` is
/// redirected to the exact `read_span` instead of an outline.
fn tail_range(engine: &Engine, repo: &str, rel: &str, n: u64) -> Option<(u64, u64)> {
    let set = engine.shard_set();
    let one: std::collections::HashSet<String> = [rel.to_string()].into_iter().collect();
    let (si, docid, _) = set.visible_at(repo, &one).into_iter().next()?;
    let content = set.content(si, docid).ok()?;
    let mut lines = content.iter().filter(|&&b| b == b'\n').count() as u64;
    if content.last().is_some_and(|&b| b != b'\n') {
        lines += 1;
    }
    (lines > 0).then(|| (lines.saturating_sub(n.max(1)) + 1, lines))
}

/// Whether an indexed file is short enough to read whole: a deny and the
/// outline + span calls it asks for cost more than its lines (whole Reads
/// of 1-2 KB memory notes were denied).
pub(crate) fn is_small_file(engine: &Engine, repo: &str, rel: &str) -> bool {
    tail_range(engine, repo, rel, 1).is_some_and(|(_, lines)| lines <= SMALL_READ_LINES)
}

/// `tail -n N` / `tail -N` / `tail -nN`: N (`-n +K`, from line K, is not a
/// count and returns `None`).
fn tail_count(args: &[String]) -> Option<u64> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let v = if a == "-n" || a == "--lines" {
            it.next().map(String::as_str)
        } else if let Some(v) = a.strip_prefix("--lines=") {
            Some(v)
        } else if let Some(v) = a.strip_prefix("-n") {
            Some(v)
        } else {
            a.strip_prefix('-').filter(|v| v.chars().all(|c| c.is_ascii_digit()))
        };
        if let Some(v) = v {
            // `+K` (from line K on) parses as a u64, but is not a tail count
            return v.strip_prefix('+').map_or_else(|| v.parse().ok(), |_| None);
        }
    }
    // plain `tail f` is the last 10 lines
    Some(10)
}

/// (repo, repo-relative path with '/') for an absolute path inside a repo.
fn locate(repos: &[(String, PathBuf)], abs: &Path) -> Option<(String, String)> {
    let canonical = abs.canonicalize().ok();
    let exists = canonical.is_some();
    let abs = canonical.unwrap_or_else(|| abs.to_path_buf());
    let deepest = |roots: &mut dyn Iterator<Item = (&String, PathBuf)>, abs: &Path| {
        let mut best: Option<(usize, String, String)> = None;
        for (name, root) in roots {
            if let Ok(rel) = abs.strip_prefix(&root) {
                let depth = root.components().count();
                if best.as_ref().is_none_or(|(d, _, _)| depth > *d) {
                    let rel = rel.to_string_lossy().replace('\\', "/");
                    best = Some((depth, name.clone(), rel));
                }
            }
        }
        best.map(|(_, n, r)| (n, r))
    };
    // an existing path against the registered roots as they are (no
    // syscalls), then against the canonicalized roots (a root reached
    // through a link or spelled otherwise), which is all a path that does
    // not exist is compared with
    exists
        .then(|| deepest(&mut repos.iter().map(|(n, r)| (n, r.clone())), &plain(&abs)))
        .flatten()
        .or_else(|| deepest(&mut repos.iter().map(|(n, r)| (n, r.canonicalize().unwrap_or_else(|_| r.clone()))), &abs))
}

/// Split a command line into pipelines on `&&`, `||`, `;`, then each
/// pipeline into `|` stages; quotes are respected, everything is trimmed.
fn split(cmd: &str) -> Vec<Vec<String>> {
    split_spans(cmd).into_iter().map(|(_, p)| p).collect()
}

/// [`split`] with each pipeline's byte range in `cmd` (first to last
/// non-blank character), so a pipeline can be replaced in place.
fn split_spans(cmd: &str) -> Vec<(std::ops::Range<usize>, Vec<String>)> {
    let mut pipelines: Vec<(std::ops::Range<usize>, Vec<String>)> = Vec::new();
    let mut stages: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let chars: Vec<(usize, char)> = cmd.char_indices().collect();
    let mut start: Option<usize> = None;
    let mut end = 0usize;
    let mut i = 0;
    let flush_stage = |cur: &mut String, stages: &mut Vec<String>| {
        let t = cur.trim().to_string();
        if !t.is_empty() {
            stages.push(t);
        }
        cur.clear();
    };
    let mut close = |stages: &mut Vec<String>, start: &mut Option<usize>, end: usize| {
        let s = std::mem::take(stages);
        if let (false, Some(b)) = (s.is_empty(), start.take()) {
            pipelines.push((b..end, s));
        }
        *start = None;
    };
    while i < chars.len() {
        let (at, c) = chars[i];
        // the text this char belongs to (not a separator)
        let mark = |upto: usize, start: &mut Option<usize>, end: &mut usize| {
            if start.is_none() {
                *start = Some(at);
            }
            *end = upto;
        };
        let after = |k: usize| chars.get(k + 1).map_or(cmd.len(), |&(b, _)| b);
        match quote {
            // a backslash escapes the next char outside quotes and inside
            // double quotes (`"\"..."`), never inside single quotes
            Some('"') | None if c == '\\' && i + 1 < chars.len() => {
                cur.push(c);
                cur.push(chars[i + 1].1);
                i += 1;
                mark(after(i), &mut start, &mut end);
            }
            Some(q) => {
                cur.push(c);
                if c == q {
                    quote = None;
                }
                mark(after(i), &mut start, &mut end);
            }
            None => match c {
                '"' | '\'' => {
                    quote = Some(c);
                    cur.push(c);
                    mark(after(i), &mut start, &mut end);
                }
                '&' if chars.get(i + 1).map(|x| x.1) == Some('&') => {
                    flush_stage(&mut cur, &mut stages);
                    close(&mut stages, &mut start, end);
                    i += 1;
                }
                '|' if chars.get(i + 1).map(|x| x.1) == Some('|') => {
                    flush_stage(&mut cur, &mut stages);
                    close(&mut stages, &mut start, end);
                    i += 1;
                }
                ';' | '\n' => {
                    flush_stage(&mut cur, &mut stages);
                    close(&mut stages, &mut start, end);
                }
                '|' => flush_stage(&mut cur, &mut stages),
                _ => {
                    cur.push(c);
                    if !c.is_whitespace() {
                        mark(after(i), &mut start, &mut end);
                    }
                }
            },
        }
        i += 1;
    }
    flush_stage(&mut cur, &mut stages);
    close(&mut stages, &mut start, end);
    pipelines
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

/// Short flags of one bundle (`-rnA` -> `rnA`); empty for a long option or
/// an operand.
fn short_flags(a: &str) -> &str {
    match a.strip_prefix('-') {
        Some(f) if !f.starts_with('-') => f,
        _ => "",
    }
}

/// `cat -A` / `-v` / `-e` / `-t`, `head -c`, `tail -c` / `-f`: output the
/// index does not have (control characters, bytes, a growing file).
fn shows_bytes(cmd: &str, args: &[String]) -> bool {
    match cmd {
        "cat" => args.iter().any(|a| short_flags(a).contains(['A', 'v', 'e', 'E', 't', 'T']) || a.starts_with("--show")),
        "head" => args.iter().any(|a| short_flags(a).starts_with('c') || a.starts_with("--bytes")),
        "tail" => args.iter().any(|a| {
            let f = short_flags(a);
            f.starts_with('c') || f.contains(['f', 'F']) || a.starts_with("--bytes") || a.starts_with("--follow")
        }),
        _ => false,
    }
}

/// The ranges of a `sed -n 'A,Bp'` / `sed -n Ap` script, several when it
/// prints more than one (`'A,Bp;C,Dp'`, `-e 'A,Bp' -e 'C,Dp'`); empty when
/// any command in it is something else.
fn sed_ranges(args: &[String]) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for a in args {
        let a = a.trim_matches(|c| c == '\'' || c == '"');
        if a.starts_with('-') || !a.ends_with('p') {
            continue;
        }
        for cmd in a.split(';').map(str::trim).filter(|c| !c.is_empty()) {
            let Some(body) = cmd.strip_suffix('p') else { return Vec::new() };
            let range = match body.split_once(',') {
                Some((s, e)) => s.parse::<u32>().ok().zip(e.parse::<u32>().ok()),
                None => body.parse::<u32>().ok().map(|s| (s, s)),
            };
            match range {
                Some(r) => out.push(r),
                None => return Vec::new(),
            }
        }
    }
    out
}

/// Positional (non-flag) arguments.
fn positional(args: &[String]) -> Vec<String> {
    args.iter().filter(|a| !a.starts_with('-')).cloned().collect()
}

/// A grep/rg command line the index answers exactly: one pattern over
/// files or folders, printing the matching lines. [`parse_grep`] returns
/// `None` for anything else — an inverted match, counts, file lists,
/// context before a match, include/type filters, several patterns, an
/// unknown flag — and the shell runs it as typed.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct GrepArgs {
    pattern: String,
    paths: Vec<String>,
    /// `-A n` / `-C n`: only a definition lookup is answered with it
    after: Option<u32>,
    ere: bool,
    fixed: bool,
    icase: bool,
    word: bool,
    whole_line: bool,
    max: Option<u32>,
    recursive: bool,
}

fn parse_grep(cmd: &str, args: &[String]) -> Option<GrepArgs> {
    let rg = cmd == "rg";
    let mut g = GrepArgs {
        ere: matches!(cmd, "rg" | "egrep" | "ag" | "ack"),
        fixed: cmd == "fgrep",
        recursive: matches!(cmd, "rg" | "ag" | "ack"),
        ..Default::default()
    };
    let mut pattern: Option<String> = None;
    let mut operands: Vec<String> = Vec::new();
    let mut only_operands = false;
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        i += 1;
        if only_operands || !a.starts_with('-') || a == "-" {
            operands.push(a.clone());
            continue;
        }
        if a == "--" {
            only_operands = true;
            continue;
        }
        if let Some(long) = a.strip_prefix("--") {
            let (k, inline) = match long.split_once('=') {
                Some((k, v)) => (k, Some(v.to_string())),
                None => (long, None),
            };
            let mut value = || -> Option<String> {
                inline.clone().or_else(|| {
                    let v = args.get(i).cloned();
                    i += 1;
                    v
                })
            };
            match k {
                "line-number" | "with-filename" | "no-filename" | "no-messages" | "color" | "colour" | "smart-case"
                | "heading" | "no-heading" => {}
                "recursive" | "dereference-recursive" if !rg => g.recursive = true,
                "ignore-case" => g.icase = true,
                "extended-regexp" | "perl-regexp" => g.ere = true,
                "fixed-strings" => g.fixed = true,
                "word-regexp" => g.word = true,
                "line-regexp" => g.whole_line = true,
                "binary-files" if inline.as_deref() == Some("without-match") => {}
                "max-count" => g.max = Some(value()?.parse().ok()?),
                "after-context" | "context" => g.after = Some(value()?.parse().ok()?),
                "regexp" => {
                    if pattern.replace(value()?).is_some() {
                        return None;
                    }
                }
                _ => return None,
            }
            continue;
        }
        // a bundle of short flags; a value flag takes the rest of the
        // bundle (`-A3`, `-efoo`) or the next word
        let flags: Vec<char> = a[1..].chars().collect();
        let mut j = 0;
        while j < flags.len() {
            let c = flags[j];
            j += 1;
            match c {
                'n' | 'H' | 'h' | 's' | 'I' => {}
                // `rg -r` is a replacement, not a recursion
                'r' | 'R' if !rg => g.recursive = true,
                'S' if rg => {}
                'i' => g.icase = true,
                'E' | 'P' => g.ere = true,
                'F' => g.fixed = true,
                'w' => g.word = true,
                'x' => g.whole_line = true,
                'A' | 'C' | 'm' | 'e' => {
                    let rest: String = flags[j..].iter().collect();
                    j = flags.len();
                    let v = if rest.is_empty() {
                        let v = args.get(i).cloned();
                        i += 1;
                        v?
                    } else {
                        rest
                    };
                    match c {
                        'A' | 'C' => g.after = Some(v.parse().ok()?),
                        'm' => g.max = Some(v.parse().ok()?),
                        _ => {
                            if pattern.replace(v).is_some() {
                                return None;
                            }
                        }
                    }
                }
                _ => return None,
            }
        }
    }
    let mut ops = operands.into_iter();
    g.pattern = match pattern {
        Some(p) => p,
        None => ops.next()?,
    };
    g.paths = ops.collect();
    // `grep x` with no file reads stdin
    if g.pattern.is_empty() || (g.paths.is_empty() && !g.recursive) {
        return None;
    }
    Some(g)
}

/// `re` with every class that could match a newline kept to one line:
/// grep matches line by line, the index over the whole file, where
/// `^\s*mod` would match from the blank line before `mod`. `\s` becomes
/// `[^\S\n]`, `\W` `[^\w\n]`, `\D` `[^\d\n]`, and a negated bracket
/// `[^…]` excludes `\n` too.
fn line_local(re: &str) -> String {
    let c: Vec<char> = re.chars().collect();
    let mut out = String::with_capacity(re.len() + 8);
    let mut in_class = false;
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        if ch == '\\' && i + 1 < c.len() {
            let e = c[i + 1];
            i += 2;
            match (e, in_class) {
                ('s', false) => out.push_str(r"[^\S\n]"),
                ('W', false) => out.push_str(r"[^\w\n]"),
                ('D', false) => out.push_str(r"[^\d\n]"),
                // inside a class: `[\s,]` → `[ \t\r\f\v,]`
                ('s', true) => out.push_str(r" \t\r\f\v"),
                _ => {
                    out.push('\\');
                    out.push(e);
                }
            }
            continue;
        }
        if !in_class && ch == '[' {
            in_class = true;
            out.push('[');
            i += 1;
            if c.get(i) == Some(&'^') {
                out.push_str(r"^\n");
                i += 1;
            }
            // a `]` first in the class is a literal
            if c.get(i) == Some(&']') {
                out.push(']');
                i += 1;
            }
            continue;
        }
        if in_class && ch == ']' {
            in_class = false;
        }
        // `[[:space:]]` and friends: the bracket inside is not a new class
        if in_class && ch == '[' && c.get(i + 1) == Some(&':') {
            if let Some(end) = (i + 2..c.len().saturating_sub(1)).find(|&k| c[k] == ':' && c[k + 1] == ']') {
                let name: String = c[i..end + 2].iter().collect();
                out.push_str(if name == "[:space:]" { r" \t\r\f\v" } else { &name });
                i = end + 2;
                continue;
            }
        }
        out.push(ch);
        i += 1;
    }
    out
}

/// The grep's pattern as the extended regex `code_grep` takes.
fn grep_regex(g: &GrepArgs) -> String {
    let mut re = if g.fixed {
        grep_literal(&g.pattern)
    } else if g.ere {
        g.pattern.clone()
    } else {
        bre_to_ere(&g.pattern)
    };
    re = line_local(&re);
    if g.word {
        re = format!(r"\b(?:{re})\b");
    }
    if g.whole_line {
        re = format!("^(?:{re})$");
    }
    re
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

/// Bring `repo`'s working tree (or just `rel`) into the index before the
/// redirected lookup runs, in a detached `hook freshen` process: the deny
/// is printed at once and the model's next call comes seconds later. Done
/// inline, the pass sat on the hook's clock — an old binary without the
/// saved worktree cache re-hashed the whole repo on every deny and hit
/// Claude Code's 10 s hook timeout 33 times in a week, so the read ran
/// anyway after a 10 s stall. The session's own repo is skipped: its MCP
/// server refreshes it before every call.
fn freshen_later(data_dir: &Path, own: Option<&str>, repo: &str, rel: Option<&str>) {
    // (in unit tests current_exe is the test runner)
    if own == Some(repo) || cfg!(test) {
        return;
    }
    if std::env::var_os("INDEXIO_HOOK_INLINE_FRESHEN").is_some() {
        freshen_paths(data_dir, repo, rel);
        return;
    }
    // never inline as a fallback: the watchdog may end this process, and
    // a writer killed under its repo lock blocks other writers for minutes
    let _ = spawn_detached(data_dir, repo, rel);
}

/// `indexio --data-dir D hook freshen --repo R [--path P]`, cut loose from
/// the hook: no inherited stdio (Claude Code waits for its pipes to close),
/// its own process group, out of the hook's job where that is allowed.
fn spawn_detached(data_dir: &Path, repo: &str, rel: Option<&str>) -> std::io::Result<()> {
    use std::process::{Command, Stdio};
    let mut c = Command::new(std::env::current_exe()?);
    c.arg("--data-dir").arg(data_dir).args(["hook", "freshen", "--repo", repo]);
    if let Some(rel) = rel {
        c.args(["--path", rel]);
    }
    c.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_BREAKAWAY_FROM_JOB: u32 = 0x0100_0000;
        c.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB);
        if c.spawn().is_ok() {
            return Ok(());
        }
        // the job does not allow breakaway
        c.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
    }
    c.spawn().map(|_| ())
}

/// Entry point for `hook freshen` (spawned by [`freshen_later`]).
pub fn run_freshen(data_dir: &Path, repo: &str, rel: Option<&str>) {
    freshen_paths(data_dir, repo, rel);
}

/// Lexical only; with the repo's saved worktree cache a whole-repo pass is
/// a stat pass when nothing changed. A known file is stat'ed alone (and
/// re-indexed when it changed), with the whole-tree pass as the fallback
/// for a file the saved cache does not list. Errors are ignored.
fn freshen_paths(data_dir: &Path, repo: &str, rel: Option<&str>) {
    let t0 = std::time::Instant::now();
    if let (Some(rel), Ok(cas)) = (rel, indexio_ingest::Cas::open(&data_dir.join("cas"))) {
        let mut cache = indexio_ingest::WorktreeCache::default();
        cache.load_once(data_dir, repo);
        let one: std::collections::HashSet<String> = [rel.to_string()].into_iter().collect();
        if let Ok(Some(_)) = indexio_ingest::reindex_worktree_paths(repo, data_dir, &cas, &mut cache, &one) {
            cache.save_if_dirty(data_dir, repo);
            if std::env::var_os("INDEXIO_HOOK_DEBUG").is_some() {
                eprintln!("hook: freshen {repo}:{rel} ({} us)", t0.elapsed().as_micros());
            }
            return;
        }
    }
    if let Ok(cas) = indexio_ingest::Cas::open(&data_dir.join("cas")) {
        // the saved cache is loaded, and saved again when the pass changed it
        let mut cache = indexio_ingest::WorktreeCache::default();
        let _ = indexio_ingest::reindex_worktree_cached(repo, data_dir, &cas, Some(&mut cache));
    }
    if std::env::var_os("INDEXIO_HOOK_DEBUG").is_some() {
        eprintln!("hook: freshen {repo} ({} us)", t0.elapsed().as_micros());
    }
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

/// The inline code of an interpreter one-liner: `python [flags] -c CODE`,
/// `node -e CODE` / `-p CODE`, `ruby -e`, `perl -e`, `php -r`. `None` for
/// a script file or a REPL.
fn one_liner_code(w: &[String]) -> Option<&str> {
    let head = w.first()?;
    let base = head.rsplit(['/', '\\']).next().unwrap_or(head).trim_end_matches(".exe");
    let flags: &[&str] = if base.starts_with("python") || base == "py" {
        &["-c"]
    } else if base == "node" || base == "deno" || base == "bun" {
        &["-e", "-p", "--eval", "--print"]
    } else if base == "ruby" || base == "perl" {
        &["-e", "-E"]
    } else if base == "php" {
        &["-r"]
    } else {
        return None;
    };
    // the code follows the first such flag; everything before it is flags
    let i = w[1..].iter().position(|a| flags.contains(&a.as_str()))? + 1;
    if w[1..i].iter().any(|a| !a.starts_with('-') && !["utf8", "utf-8"].contains(&a.as_str())) {
        return None; // a script file came first
    }
    w.get(i + 1).map(String::as_str)
}

/// The first string literal after `marker` in `code` (`open(r'…'`,
/// `readFileSync("…"`, `.find('…'`): raw/byte prefixes skipped, escapes
/// left as they are.
fn literal_after(code: &str, marker: &str) -> Option<String> {
    let start = code.find(marker)? + marker.len();
    let rest = code[start..].trim_start_matches([' ', 'r', 'b', 'R', 'B', 'u', 'f']);
    let q = rest.chars().next()?;
    if q != '\'' && q != '"' {
        return None;
    }
    let body = &rest[1..];
    let end = body.find(q)?;
    Some(body[..end].to_string())
}

/// A one-liner that opens a file to print part of it is a read in
/// disguise (seen after the `cat`/`sed` denials: `python -X utf8 -c
/// "src=open(r'…/lib.rs').read(); i=src.find('\"goto\" =>'); print(src[i:i+3000])"`).
/// `Some((path, needle))` when the code only reads: the file it opens and
/// the literal it looks for, if any. Anything that writes, spawns or edits
/// is left to the shell.
fn script_read(stage: &str) -> Option<(String, Option<String>)> {
    let w = words(stage);
    let code = one_liner_code(&w)?;
    const WRITES: &[&str] = &[
        ".write(", "writeFileSync", "writeFile(", "appendFile", "os.remove", "os.rename", "os.unlink", "shutil",
        "subprocess", "os.system", "mkdir", "rmtree", "unlinkSync", "renameSync", "child_process", "'w'", "\"w\"",
        "'a'", "\"a\"", "'w+'", "\"w+\"", "'r+'", "\"r+\"", "'wb'", "\"wb\"",
    ];
    // byte-level questions (line endings, counts, escapes): the answer is
    // not the file's text, and the index holds text with CRLF folded —
    // `b.count(b'\r\n')` was denied twice in a row
    const ANALYSES: &[&str] = &["'rb'", "\"rb\"", ".count(", "repr(", "ascii("];
    if WRITES.iter().chain(ANALYSES).any(|m| code.contains(m)) {
        return None;
    }
    let path = ["open(", "readFileSync(", "readFile(", "Path(", "read_text("]
        .iter()
        .find_map(|m| literal_after(code, m))?;
    if path.is_empty() || path.contains('*') {
        return None;
    }
    let needle = [".find(", ".index(", ".rfind(", ".indexOf(", "re.search(", ".search(", "re.finditer(", "re.findall("]
        .iter()
        .find_map(|m| literal_after(code, m))
        .filter(|n| !n.trim().is_empty());
    Some((path, needle))
}

/// `needle` as a `code_grep` pattern: every regex metacharacter escaped.
fn grep_literal(needle: &str) -> String {
    let mut out = String::with_capacity(needle.len() + 8);
    for c in needle.chars() {
        if "\\^$.*+?()[]{}|".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

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
    if script_read(first).is_some() {
        return false; // a one-liner that only opens a file to print it
    }
    !cmd.is_empty() && !LOOKUP_OR_HARMLESS.contains(&cmd)
}

/// What the index answers for one shell pipeline (Bash) or statement
/// (PowerShell): tool calls made in order by `indexio tool`, which brings
/// the files they read up to date first. The shell hooks rewrite the
/// command to run that call instead of denying it (a deny shows as a hook
/// error, costs a second round trip, and is a dead end for a caller that
/// cannot reach the indexio MCP tools).
#[derive(Debug)]
pub(crate) struct Serve {
    calls: Vec<(&'static str, Value)>,
    /// (repo, path) to freshen first; an empty path = the whole repo
    fresh: Vec<(String, String)>,
    /// a line printed after the answer
    note: Option<String>,
}

impl Serve {
    /// `(tool name, base64url JSON arguments)` for `indexio tool`.
    fn payload(&self, current: Option<&str>) -> (String, String) {
        let (name, mut args) = if self.calls.len() == 1 {
            (self.calls[0].0.to_string(), self.calls[0].1.clone())
        } else {
            let calls: Vec<Value> = self.calls.iter().map(|(n, a)| json!({"name": n, "args": a})).collect();
            ("batch".to_string(), json!({ "calls": calls }))
        };
        if let Some(c) = current {
            args["_current"] = json!(c);
        }
        if !self.fresh.is_empty() {
            args["_fresh"] = json!(self.fresh);
        }
        if let Some(n) = &self.note {
            args["_note"] = json!(n);
        }
        (name, b64url_encode(args.to_string().as_bytes()))
    }
}

/// The most lines one `read_span` returns.
const MAX_SPAN: u64 = 400;

/// What a read of one file asks for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Want {
    /// lines first..=last (1-based)
    Range(u64, u64),
    /// the last n lines
    Tail(u64),
    /// the whole file
    Whole,
}

/// Registered repos and the index, opened on first use.
pub(crate) struct Ctx<'a> {
    pub(crate) data_dir: &'a Path,
    pub(crate) repos: Vec<(String, PathBuf)>,
    engine: Option<Option<Engine>>,
}

impl<'a> Ctx<'a> {
    pub(crate) fn new(data_dir: &'a Path) -> Self {
        Ctx { data_dir, repos: repos(data_dir), engine: None }
    }
    pub(crate) fn engine(&mut self) -> Option<&Engine> {
        if self.engine.is_none() {
            self.engine = Some(Engine::open(self.data_dir).ok());
        }
        self.engine.as_ref()?.as_ref()
    }
    /// (repo, path the index holds) for an existing file it can answer for.
    fn indexed_file(&mut self, abs: &Path) -> Option<(String, String)> {
        if !abs.is_file() {
            return None;
        }
        let (repo, rel) = locate(&self.repos, abs)?;
        let rel = indexed_rel(self.engine()?, &repo, &rel, abs)?;
        Some((repo, rel))
    }
}

/// Calls answering `wants` of one indexed file, the note to print after
/// them, and how many lines they return (a whole read of a structured file
/// counts as large: its outline is the answer).
fn file_calls(engine: &Engine, repo: &str, rel: &str, wants: &[Want]) -> Option<(Vec<(&'static str, Value)>, Option<String>, u64)> {
    let total = tail_range(engine, repo, rel, 1).map(|(_, e)| e)?;
    let span = |s: u64, e: u64| ("read_span", json!({"repo": repo, "path": rel, "start": s, "end": e}));
    let mut calls = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    let mut lines = 0;
    for &w in wants {
        let (s, e) = match w {
            Want::Range(s, e) => (s.max(1), e.min(total)),
            Want::Tail(n) => tail_range(engine, repo, rel, n)?,
            Want::Whole if total <= SMALL_READ_LINES => (1, total),
            Want::Whole => {
                // a file with a few definitions or more: the outline first, as
                // the deny used to ask for; otherwise its lines
                if engine.outline(repo, rel).is_some_and(|o| o.len() >= 3) {
                    calls.push(("file_outline", json!({"repo": repo, "path": rel})));
                    notes.push(format!(
                        "({rel}: {total} lines, outline above; read_span or `sed -n A,Bp` for the lines you need)"
                    ));
                    lines += SMALL_READ_LINES + 1;
                    continue;
                }
                (1, total)
            }
        };
        if s > e {
            continue;
        }
        let end = e.min(s + MAX_SPAN - 1);
        if end < e {
            notes.push(format!("({rel}: lines {s}-{end} of {s}-{e}; `sed -n {},{e}p` for the rest)", end + 1));
        }
        calls.push(span(s, end));
        lines += end - s + 1;
    }
    if calls.is_empty() {
        return None;
    }
    Some((calls, (!notes.is_empty()).then(|| notes.join("\n")), lines))
}

/// The answer to reading `files`, each with `wants`; `None` when a file is
/// not one the index answers for, or when the read is small enough that
/// the shell's own output is as cheap.
pub(crate) fn serve_files(ctx: &mut Ctx, files: &[PathBuf], wants: &[Want]) -> Option<Serve> {
    let mut serve = Serve { calls: Vec::new(), fresh: Vec::new(), note: None };
    let mut notes = Vec::new();
    let mut lines = 0;
    for abs in files {
        let (repo, rel) = ctx.indexed_file(abs)?;
        let (calls, note, n) = file_calls(ctx.engine()?, &repo, &rel, wants)?;
        serve.calls.extend(calls);
        notes.extend(note);
        lines += n;
        serve.fresh.push((repo, rel));
    }
    if serve.calls.is_empty() || lines <= SMALL_READ_LINES {
        return None;
    }
    serve.note = (!notes.is_empty()).then(|| notes.join("\n"));
    Some(serve)
}

/// The answer to a search of `targets` (files or folders, all in one repo)
/// with grep semantics: `code_grep` (case-sensitive unless `icase`), a
/// definition's lines for `grep "fn x" -A n`, or the outline for a grep of
/// definition keywords in one file. `lines` = the matching lines per file
/// the caller wants at most.
pub(crate) fn serve_search(ctx: &mut Ctx, targets: &[PathBuf], g: &GrepArgs, lines: Option<u32>) -> Option<Serve> {
    let mut repo: Option<String> = None;
    let mut rels: Vec<String> = Vec::new();
    let mut whole_repo = false;
    let mut anchors: Vec<String> = Vec::new();
    for p in targets {
        let (r, rel) = if p.is_file() {
            ctx.indexed_file(p)?
        } else if p.is_dir() {
            locate(&ctx.repos, p)?
        } else {
            return None;
        };
        if repo.as_ref().is_some_and(|x| *x != r) {
            return None; // two repos: the shell's
        }
        repo = Some(r);
        if rel.is_empty() {
            whole_repo = true;
        }
        // anchored: `path:Cargo.toml` alone would take every crate's Cargo.toml
        anchors.push(if p.is_file() { format!("^{rel}$") } else { format!("^{rel}/") });
        rels.push(rel);
    }
    let repo = repo?;
    rels.sort();
    rels.dedup();
    anchors.sort();
    anchors.dedup();
    let all_files = targets.iter().all(|p| p.is_file());
    let fresh: Vec<(String, String)> = if all_files {
        rels.iter().map(|r| (repo.clone(), r.clone())).collect()
    } else {
        vec![(repo.clone(), String::new())]
    };
    // `grep -n "fn foo" -A 40 f`: the definition's lines
    if let Some(n) = g.after {
        let name = definition_name(&g.pattern)?;
        if !all_files {
            return None;
        }
        let engine = ctx.engine()?;
        let mut hits = Vec::new();
        for rel in &rels {
            for item in engine.outline(&repo, rel).unwrap_or_default() {
                if item.name == name {
                    hits.push((rel.clone(), item.start_line, item.end_line));
                }
            }
        }
        let [(rel, s, e)] = hits.as_slice() else { return None };
        let end = (*e).min(s + n).min(s + MAX_SPAN as u32 - 1);
        return Some(Serve {
            calls: vec![("read_span", json!({"repo": repo, "path": rel, "start": s, "end": end}))],
            fresh,
            note: None,
        });
    }
    let re = grep_regex(g);
    // the query language splits on whitespace: such a path is the shell's
    if rels.iter().any(|r| r.contains(char::is_whitespace) || r.contains('|')) {
        return None;
    }
    let path_filter = if whole_repo { String::new() } else { format!(" path:{}", anchors.join("|")) };
    let case = if g.icase { "no" } else { "yes" };
    let mut args = json!({ "pattern": format!("{re} case:{case} repo:{repo}{path_filter}") });
    // `code_grep` shows at most 100 lines a file: a longer cut is the shell's
    if let Some(n) = g.max.or(lines) {
        if n > 100 {
            return None;
        }
        args["lines"] = json!(n.max(1));
    }
    Some(Serve { calls: vec![("code_grep", args)], fresh, note: None })
}

/// `head` / `head -N` / `head -n N` / `tail …` after a lookup: the count of
/// lines kept, which an answer can be cut to. `None` for anything else.
fn line_cut(stage: &str) -> Option<(bool, u64)> {
    let w = words(stage);
    let head = match w.first()?.as_str() {
        "head" => true,
        "tail" => false,
        _ => return None,
    };
    if shows_bytes(&w[0], &w[1..]) {
        return None;
    }
    let n = if head { head_count(&w[1..])? } else { tail_count(&w[1..])? };
    // `head -5 f`: a file operand makes it a read, not a cut
    if !operands(&w[1..]).is_empty() {
        return None;
    }
    Some((head, n))
}

/// `head` line count: `-n N`, `-nN`, `-N`, `--lines=N`, else 10; `None`
/// for `-n -K` (all but the last K).
fn head_count(args: &[String]) -> Option<u64> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let v = if a == "-n" || a == "--lines" {
            it.next().map(String::as_str)
        } else if let Some(v) = a.strip_prefix("--lines=") {
            Some(v)
        } else if let Some(v) = a.strip_prefix("-n") {
            Some(v)
        } else {
            a.strip_prefix('-').filter(|v| !v.is_empty() && v.chars().all(|c| c.is_ascii_digit()))
        };
        if let Some(v) = v {
            return v.parse().ok();
        }
    }
    Some(10)
}

/// The file operands of `head`/`tail` (the value after `-n` is not one).
fn operands(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "-n" || a == "--lines" {
            it.next();
        } else if !a.starts_with('-') || a == "-" {
            out.push(a.clone());
        }
    }
    out
}

/// `sed -n` with line-range scripts only: (ranges, file operands).
fn sed_args(args: &[String]) -> Option<(Vec<(u32, u32)>, Vec<String>)> {
    let mut quiet = false;
    let mut scripts = Vec::new();
    let mut ops = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-n" | "--quiet" | "--silent" => quiet = true,
            "-e" | "--expression" => scripts.push(it.next()?.clone()),
            "-E" | "-r" | "-s" | "-u" | "--posix" => {}
            s if s.starts_with('-') && s.len() > 1 => return None,
            _ => ops.push(a.clone()),
        }
    }
    if !quiet {
        return None;
    }
    if scripts.is_empty() {
        if ops.is_empty() {
            return None;
        }
        scripts.push(ops.remove(0));
    }
    let ranges = sed_ranges(&scripts);
    (!ranges.is_empty()).then_some((ranges, ops))
}

/// Unquoted `#` (a comment), `$(`, `` ` `` or `<(`: text the splitter does
/// not model, so nothing in the command is rewritten.
fn has_unmodelled_syntax(cmd: &str) -> bool {
    let cs: Vec<char> = cmd.chars().collect();
    let mut quote: Option<char> = None;
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        match quote {
            Some('\'') if c == '\'' => quote = None,
            Some('\'') => {}
            Some(_) if c == '\\' => i += 1,
            Some(_) if c == '"' => quote = None,
            Some(_) if c == '`' || (c == '$' && cs.get(i + 1) == Some(&'(')) => return true,
            Some(_) => {}
            None if c == '\\' => i += 1,
            None if c == '\'' || c == '"' => quote = Some(c),
            None if c == '`' => return true,
            None if (c == '$' || c == '<') && cs.get(i + 1) == Some(&'(') => return true,
            None if c == '#' && (i == 0 || cs[i - 1].is_whitespace()) => return true,
            None => {}
        }
        i += 1;
    }
    false
}

/// One pipeline the index answers: the answer, and the stages kept after
/// it (`| head -30` on a search) — `None` when the answer replaces the
/// whole pipeline.
type Served = (Serve, Option<String>);

/// The answer for one pipeline, or `None` (the shell runs it as typed).
fn serve_pipeline(ctx: &mut Ctx, cwd: &Path, pipeline: &[String]) -> Option<Served> {
    let first = pipeline.first()?;
    let w = words(first);
    let (mut cmd, mut args): (&str, &[String]) = (w.first()?.as_str(), &w[1..]);
    if cmd == "rtk" && args.first().map(String::as_str) == Some("proxy") {
        cmd = args.get(1)?.as_str();
        args = &args[2..];
    }
    let rest = &pipeline[1..];
    match cmd {
        "cat" | "less" | "more" | "head" | "tail" | "sed" => {
            if shows_bytes(cmd, args) {
                return None;
            }
            let (files, mut wants): (Vec<String>, Vec<Want>) = match cmd {
                "head" => (operands(args), vec![Want::Range(1, head_count(args)?.max(1))]),
                "tail" => {
                    if args.iter().any(|a| a.starts_with('+')) {
                        return None;
                    }
                    (operands(args), vec![Want::Tail(tail_count(args)?)])
                }
                "sed" => {
                    let (ranges, files) = sed_args(args)?;
                    (files, ranges.into_iter().map(|(s, e)| Want::Range(u64::from(s), u64::from(e))).collect())
                }
                _ => {
                    if args.iter().any(|a| a.starts_with('-') && a != "-n") {
                        return None;
                    }
                    (positional(args), vec![Want::Whole])
                }
            };
            // `cat f | head -150` is the first 150 lines; any other stage
            // after a read is the shell's
            match rest {
                [] => {}
                [cut] if wants == [Want::Whole] => {
                    let (head, n) = line_cut(cut)?;
                    wants = vec![if head { Want::Range(1, n.max(1)) } else { Want::Tail(n) }];
                }
                _ => return None,
            }
            if files.is_empty() {
                return None;
            }
            let abs: Vec<PathBuf> = files.iter().map(|f| resolve(cwd, f)).collect();
            Some((serve_files(ctx, &abs, &wants)?, None))
        }
        "grep" | "rg" | "egrep" | "fgrep" | "ag" | "ack" => {
            let g = parse_grep(cmd, args)?;
            let mut cut = None;
            for st in rest {
                let (head, n) = line_cut(st)?;
                if head {
                    cut = Some(cut.map_or(n, |c: u64| c.min(n)));
                }
            }
            let targets: Vec<PathBuf> =
                if g.paths.is_empty() { vec![cwd.to_path_buf()] } else { g.paths.iter().map(|p| resolve(cwd, p)).collect() };
            let lines = cut.map(|n| u32::try_from(n).unwrap_or(u32::MAX));
            let serve = serve_search(ctx, &targets, &g, lines)?;
            // one file: `lines` is the whole cut. Several: the answer adds a
            // `repo:path` line per file, so `head -N` becomes `head -(2N+1)`
            // (every match in its own file, and the count line) and never
            // drops a line the typed cut would show.
            let one_file = g.paths.len() == 1 && targets[0].is_file();
            let keep: Vec<String> = rest
                .iter()
                .filter_map(|st| {
                    let (head, n) = line_cut(st)?;
                    (!(one_file && head)).then(|| format!("{} -n {}", if head { "head" } else { "tail" }, 2 * n + 1))
                })
                .collect();
            Some((serve, (!keep.is_empty()).then(|| keep.join(" | "))))
        }
        "find" => {
            for st in rest {
                line_cut(st)?;
            }
            let mut dir: Option<&String> = None;
            let mut name = "*".to_string();
            let mut it = args.iter();
            while let Some(a) = it.next() {
                match a.as_str() {
                    "-name" | "-iname" => name = it.next()?.clone(),
                    "-type" if it.next()? == "f" => {}
                    s if s.starts_with('-') => return None,
                    _ if dir.is_none() => dir = Some(a),
                    _ => return None,
                }
            }
            if name.contains('/') {
                return None;
            }
            let d = dir.map(|d| resolve(cwd, d)).unwrap_or_else(|| cwd.to_path_buf());
            if !d.is_dir() {
                return None;
            }
            let (repo, rel) = locate(&ctx.repos, &d)?;
            let pattern = if rel.is_empty() { format!("**/{name}") } else { format!("{rel}/**/{name}") };
            let serve = Serve {
                calls: vec![("list_files", json!({"pattern": pattern, "repo": repo}))],
                fresh: vec![(repo, String::new())],
                note: None,
            };
            Some((serve, (!rest.is_empty()).then(|| rest.join(" | "))))
        }
        _ => None,
    }
}

/// The pipelines of `command` the index answers: (pipeline index in
/// [`split_spans`], answer). Empty when the call does other work too (a
/// build, a write, a script: the shell is needed anyway, and a re-run
/// of half of it is worse than its reads) or when its syntax goes beyond
/// what the splitter models.
fn plan(data_dir: &Path, cwd: &Path, command: &str) -> Vec<(usize, Served)> {
    if command.contains("INDEXIO_RAW") || command.contains("# raw") || has_unmodelled_syntax(command) {
        return Vec::new();
    }
    let pipelines = split_spans(command);
    if pipelines.iter().any(|(_, p)| is_other_work(p)) {
        return Vec::new();
    }
    let mut ctx = Ctx::new(data_dir);
    if ctx.repos.is_empty() {
        return Vec::new();
    }
    let mut cwd = cwd.to_path_buf();
    let mut out = Vec::new();
    for (i, (_, pipeline)) in pipelines.iter().enumerate() {
        if has_redirect_any(pipeline) {
            continue;
        }
        let w = pipeline.first().map(|s| words(s)).unwrap_or_default();
        if w.first().map(String::as_str) == Some("cd") && pipeline.len() == 1 {
            if let Some(dir) = w.get(1) {
                cwd = resolve(&cwd, dir);
            }
            continue;
        }
        if let Some(s) = serve_pipeline(&mut ctx, &cwd, pipeline) {
            out.push((i, s));
        }
    }
    out
}

fn has_redirect_any(pipeline: &[String]) -> bool {
    pipeline.iter().any(|st| has_redirect(st))
}

/// `indexio --data-dir D` as a shell word sequence every shell accepts.
fn tool_prefix(exe: &Path, data_dir: &Path) -> String {
    let dd = data_dir.to_string_lossy().replace('\\', "/").replace('\'', "'\\''");
    hook_command(exe, &format!("--data-dir '{dd}'"))
}

/// `command` with each answered pipeline replaced by its `indexio tool`
/// call; the original pipeline runs if the call fails.
fn bash_rewrite(exe: &Path, data_dir: &Path, command: &str, current: Option<&str>, served: &[(usize, Served)]) -> String {
    let spans = split_spans(command);
    let prefix = tool_prefix(exe, data_dir);
    let mut out = command.to_string();
    for (i, (serve, keep)) in served.iter().rev() {
        let (range, stages) = &spans[*i];
        let (name, b64) = serve.payload(current);
        let call = format!("{prefix} tool {name} {b64}");
        let rep = match keep {
            Some(k) => format!("{{ {call} || {}; }} | {k}", stages[0]),
            None => format!("{{ {call} || {}; }}", &command[range.clone()]),
        };
        out.replace_range(range.clone(), &rep);
    }
    out
}

/// The command to run instead of `command`, when the index answers a read
/// in it (SPEC-P11): `None` = run it as typed.
pub fn rewrite_bash(data_dir: &Path, cwd: &Path, command: &str, exe: &Path) -> Option<String> {
    let served = plan(data_dir, cwd, command);
    if served.is_empty() {
        return None;
    }
    let current = locate(&repos(data_dir), cwd).map(|(r, _)| r);
    Some(bash_rewrite(exe, data_dir, command, current.as_deref(), &served))
}

/// The base64url alphabet (RFC 4648 §5): shell-safe in Bash, Git Bash's
/// path conversion and PowerShell alike, so the tool's JSON arguments need
/// no quoting.
const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

pub(crate) fn b64url_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let n = (u32::from(chunk[0]) << 16) | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8) | u32::from(*chunk.get(2).unwrap_or(&0));
        for k in 0..=chunk.len() {
            out.push(B64[((n >> (18 - 6 * k)) & 63) as usize] as char);
        }
    }
    out
}

pub(crate) fn b64url_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.trim_end_matches('=').bytes() {
        let v = B64.iter().position(|&b| b == c)? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// Entry point for `indexio tool NAME [ARGS]`: freshen what the call reads,
/// run it, print the answer (and the hook's note). Returns the exit status:
/// 0, or 1 when the tool failed (its message on stderr, nothing on stdout,
/// so a hook-written fallback prints the original command's output alone).
pub fn run_tool(data_dir: &Path, name: &str, args: Option<&str>, current: Option<&str>) -> i32 {
    use std::io::Write as _;
    let parsed = match args.map(str::trim) {
        None | Some("") => Some(json!({})),
        Some(a) if a.starts_with('{') => serde_json::from_str::<Value>(a).ok(),
        Some(a) => b64url_decode(a).and_then(|b| serde_json::from_slice::<Value>(&b).ok()),
    };
    let Some(Value::Object(mut obj)) = parsed else {
        eprintln!("indexio tool: the arguments are neither a JSON object nor one base64url-encoded");
        return 2;
    };
    let cur = obj.remove("_current").and_then(|v| v.as_str().map(str::to_string)).or(current.map(str::to_string));
    let note = obj.remove("_note").and_then(|v| v.as_str().map(str::to_string));
    let fresh: Vec<(String, String)> = obj.remove("_fresh").and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default();
    let current = crate::mcp::resolve_current_repo(data_dir, cur.as_deref());
    // the index must be current with the files on disk: the MCP server
    // refreshes before each call, and this call has no server
    for (repo, rel) in &fresh {
        freshen_paths(data_dir, repo, (!rel.is_empty()).then_some(rel.as_str()));
    }
    let args = Value::Object(obj);
    let calls: Vec<(String, Value)> = if name == "batch" {
        args["calls"]
            .as_array()
            .map(|a| a.iter().map(|c| (c["name"].as_str().unwrap_or("").to_string(), c["args"].clone())).collect())
            .unwrap_or_default()
    } else {
        vec![(name.to_string(), args)]
    };
    let mut out = String::new();
    for (n, a) in &calls {
        match crate::mcp::call_once(data_dir, n, a, current.as_deref()) {
            // a grep that finds nothing exits 1 (`grep x f || echo none`):
            // the typed grep runs instead, prints nothing and exits 1
            Ok(text) if n == "code_grep" && calls.len() == 1 && text.trim_start().starts_with("no matches") => return 1,
            Ok(text) => {
                out.push_str(&text);
                if !text.ends_with('\n') {
                    out.push('\n');
                }
            }
            Err(e) => {
                eprintln!("indexio {n}: {e}");
                return 1;
            }
        }
    }
    if let Some(n) = note {
        out.push_str(&n);
        out.push('\n');
    }
    // a reader that stops early (`| head`) is not an error
    let mut so = std::io::stdout().lock();
    let _ = so.write_all(out.as_bytes());
    let _ = so.flush();
    0
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
const HOOKS: [(&str, Option<&str>, &str, u64); 5] = [
    ("PreToolUse", Some("Bash"), "hook bash", 10),
    ("PreToolUse", Some("PowerShell"), "hook powershell", 10),
    ("PreToolUse", Some("Read"), "hook read", 10),
    ("PreCompact", None, "sessions --quiet", 120),
    ("SessionStart", None, "hook session-start", 10),
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

/// The verdict for one `Read` tool call (SPEC-P10 §16): a file the index
/// holds is served by `file_outline` + `read_span` instead — a whole-file
/// Read of a source file was the largest single result in a session hour
/// (2,000 lines by default). `offset`/`limit` (1-based line, count) map to
/// a `read_span` range; without them the outline comes first. Files the
/// index does not have (images, logs, task outputs, unindexed folders) are
/// allowed untouched, and so is a Read of at most [`SMALL_READ_LINES`]
/// lines: `read_span` saves ~1.5 tokens a line there, the deny and the
/// second call cost ~110.
pub fn read_verdict(data_dir: &Path, cwd: Option<&Path>, file_path: &str, offset: Option<u64>, limit: Option<u64>) -> Option<String> {
    if limit.is_some_and(|n| n <= SMALL_READ_LINES) {
        return None;
    }
    let abs = PathBuf::from(file_path);
    if !abs.is_file() {
        return None;
    }
    let repos = repos(data_dir);
    let (repo, rel) = locate(&repos, &abs)?;
    let engine = Engine::open(data_dir).ok()?;
    let asked = rel;
    let rel = indexed_rel(&engine, &repo, &asked, &abs)?;
    if limit.is_none() && is_small_file(&engine, &repo, &rel) {
        return None;
    }
    let own = cwd.and_then(|c| locate(&repos, c)).map(|(r, _)| r);
    freshen_later(data_dir, own.as_deref(), &repo, Some(&rel));
    let mut name = abs.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_else(|| rel.clone());
    if rel != asked {
        name.push_str(" (this worktree's copy is byte-identical to the repo's)");
    }
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

/// Reads of this many lines or fewer are left to the Read tool.
pub const SMALL_READ_LINES: u64 = 60;

/// How long a hook may think before it lets the call through
/// (`INDEXIO_HOOK_BUDGET_MS`, default 3000). Claude Code waits up to the
/// hook's timeout (10 s) and then runs the call anyway, so a slow verdict
/// only ever cost time. The verdict is read-only (the freshen runs in its
/// own process), so ending it mid-way is safe.
fn within_budget<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let ms = std::env::var("INDEXIO_HOOK_BUDGET_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(3000u64);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    let out = rx.recv_timeout(std::time::Duration::from_millis(ms)).ok();
    if out.is_none() && std::env::var_os("INDEXIO_HOOK_DEBUG").is_some() {
        eprintln!("hook: over the {ms} ms budget, allowing");
    }
    out
}

/// Entry point for `hook read`: hook JSON on stdin, decision JSON on stdout.
pub fn run_read_hook(data_dir: &Path) -> anyhow::Result<()> {
    let mut input = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)?;
    let v: Value = serde_json::from_str(&input).unwrap_or(Value::Null);
    if v["tool_name"].as_str() != Some("Read") {
        return Ok(());
    }
    let Some(file_path) = v["tool_input"]["file_path"].as_str().map(str::to_string) else { return Ok(()) };
    let offset = v["tool_input"]["offset"].as_u64();
    let limit = v["tool_input"]["limit"].as_u64();
    let cwd = v["cwd"].as_str().map(PathBuf::from);
    let session = v["session_id"].as_str().unwrap_or("nosession").to_string();
    let data_dir = data_dir.to_path_buf();
    let verdict = within_budget(move || {
        let reason = read_verdict(&data_dir, cwd.as_deref(), &file_path, offset, limit)?;
        // a deny names tools this caller cannot reach: let the Read through
        if indexio_unreachable(&v, cwd.as_deref()) {
            return None;
        }
        let key = format!("Read {file_path} {offset:?} {limit:?}");
        if repeated_denial(&data_dir, &session, &key) {
            return None;
        }
        // a subagent's indexio tools are deferred (a ToolSearch first); its
        // shell reads are answered from the index in one call
        if v["agent_id"].is_string() {
            let path = file_path.replace('\\', "/");
            let shell = match (offset, limit) {
                (Some(s), Some(n)) if n > 0 => format!("sed -n {},{}p '{path}'", s.max(1), s.max(1) + n - 1),
                _ => format!("cat '{path}'"),
            };
            return Some(format!("{reason} (or Bash `{shell}`, answered from the index)"));
        }
        Some(reason)
    })
    .flatten();
    if let Some(reason) = verdict {
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

/// Agent types whose tool list has no MCP tools.
const AGENTS_WITHOUT_MCP: &[&str] = &["claude-code-guide", "statusline-setup"];

/// Whether the caller of a hooked call cannot use the indexio MCP tools: an
/// agent type whose tool list leaves them out, or a transcript whose last
/// indexio call failed as unavailable (the server dropped mid-session: the
/// Read deny then sent the model to tools it no longer had, and it wrote
/// scripts to read files).
fn indexio_unreachable(hook_input: &Value, cwd: Option<&Path>) -> bool {
    if let Some(t) = hook_input["agent_type"].as_str() {
        if agent_lacks_indexio(t, cwd, crate::sessions::default_claude_dir().as_deref()) {
            return true;
        }
    }
    let Some(t) = hook_input["transcript_path"].as_str() else { return false };
    // a subagent's hook input names the session's transcript
    let path = match hook_input["agent_id"].as_str() {
        Some(a) => Path::new(t).with_extension("").join("subagents").join(format!("agent-{a}.jsonl")),
        None => PathBuf::from(t),
    };
    transcript_says_unavailable(&path)
}

/// An agent definition (`.claude/agents/<type>.md`, project then user)
/// whose `tools:` list has no indexio tool, or a built-in type without MCP.
fn agent_lacks_indexio(agent_type: &str, cwd: Option<&Path>, claude_dir: Option<&Path>) -> bool {
    if AGENTS_WITHOUT_MCP.contains(&agent_type) {
        return true;
    }
    if agent_type.contains(['/', '\\', ':']) {
        return false; // a plugin's agent: not looked up
    }
    let dirs = [cwd.map(|c| c.join(".claude").join("agents")), claude_dir.map(|c| c.join("agents"))];
    for dir in dirs.into_iter().flatten() {
        let Ok(text) = std::fs::read_to_string(dir.join(format!("{agent_type}.md"))) else { continue };
        let Some(front) = text.strip_prefix("---").and_then(|r| r.split_once("\n---")).map(|(f, _)| f) else {
            return false;
        };
        return front.lines().find_map(|l| l.trim().strip_prefix("tools:")).is_some_and(|tools| {
            let t = tools.trim();
            !(t.is_empty() || t == "*" || t.contains("mcp__indexio"))
        });
    }
    false
}

/// Whether the transcript's last indexio call failed as unavailable ("No
/// such tool available: mcp__indexio__…", with no indexio call after it).
/// Reads the transcript's tail only.
fn transcript_says_unavailable(path: &Path) -> bool {
    use std::io::{Read as _, Seek as _, SeekFrom};
    const TAIL: u64 = 512 * 1024;
    let Ok(mut f) = std::fs::File::open(path) else { return false };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    if f.seek(SeekFrom::Start(len.saturating_sub(TAIL))).is_err() {
        return false;
    }
    let mut buf = Vec::new();
    if f.read_to_end(&mut buf).is_err() {
        return false;
    }
    let text = String::from_utf8_lossy(&buf);
    let Some(at) = text.rfind("No such tool available: mcp__indexio__") else { return false };
    !text[at..].contains("\"name\":\"mcp__indexio__")
}

/// Touched when a session's context is compacted or cleared.
pub(crate) fn compacted_marker(data_dir: &Path) -> PathBuf {
    data_dir.join("hook").join("compacted")
}

/// Entry point for `hook session-start` (SessionStart): hook JSON on stdin,
/// a few lines of past context as `additionalContext` on stdout
/// (sessions::session_start_context). Prints nothing when there is none.
pub fn run_session_start_hook(claude_dir: &Path, data_dir: &Path) -> anyhow::Result<()> {
    let mut input = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)?;
    let v: Value = serde_json::from_str(&input).unwrap_or(Value::Null);
    let cwd = v["cwd"].as_str().map(PathBuf::from).unwrap_or(std::env::current_dir()?);
    let session = v["session_id"].as_str().unwrap_or("");
    let transcript = v["transcript_path"].as_str().map(PathBuf::from);
    let source = v["source"].as_str().unwrap_or("startup");
    // the context was rebuilt: the MCP servers stop treating earlier
    // read_span answers as still in view
    if matches!(source, "compact" | "clear") {
        let marker = compacted_marker(data_dir);
        if let Some(dir) = marker.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(&marker, session);
    }
    if let Some(ctx) = crate::sessions::session_start_context(claude_dir, &cwd, session, transcript.as_deref(), source) {
        println!(
            "{}",
            json!({
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": ctx,
                }
            })
        );
    }
    Ok(())
}

/// Whether this exact call was the last one denied in `session`: a model
/// that re-sends a denied call unchanged has read the reason and still
/// wants it, and a second deny only cost it another round trip (25 such
/// re-sends in a week). The record is cleared then, so the next deny of it
/// starts over; otherwise this call is recorded as the last denied.
fn repeated_denial(data_dir: &Path, session: &str, command: &str) -> bool {
    let path = last_denied_path(data_dir, session);
    let hash = indexio_types::BlobId::from_content(without_raw(command).as_bytes()).hex();
    let same = std::fs::read_to_string(&path).is_ok_and(|prev| prev.trim() == hash);
    if same {
        let _ = std::fs::remove_file(&path);
    } else {
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
    // `python [-X utf8] -c '…'`, `node -e` and other one-liners print what
    // they were asked for: leave them (a read-only one is judge's business)
    if one_liner_code(&w).is_some() {
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

/// Entry point for `hook powershell`: the PowerShell tool's commands judged
/// by [`ps::verdict`] (no runner rewrite: `indexio run` is a POSIX-shell
/// wrapper).
pub fn run_powershell_hook(data_dir: &Path) -> anyhow::Result<()> {
    let mut input = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)?;
    let v: Value = serde_json::from_str(&input).unwrap_or(Value::Null);
    if v["tool_name"].as_str() != Some("PowerShell") {
        return Ok(());
    }
    let command = v["tool_input"]["command"].as_str().unwrap_or("").to_string();
    let cwd = v["cwd"].as_str().map(PathBuf::from).or_else(|| std::env::current_dir().ok()).unwrap_or_default();
    let Ok(exe) = std::env::current_exe() else { return Ok(()) };
    let dd = data_dir.to_path_buf();
    let decided = within_budget(move || ps::rewrite(&dd, &cwd, &command, &exe));
    if let Some(Some(rewritten)) = decided {
        print_served(&rewritten);
    }
    Ok(())
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
    let Ok(exe) = std::env::current_exe() else { return Ok(()) };
    // SPEC-P10 §31: a script or build runs through `indexio run`, which
    // keeps a long output in the runs source and returns a digest
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
    // SPEC-P11: a read the index answers runs as `indexio tool` in place
    let (dd, cmd) = (data_dir.to_path_buf(), command.to_string());
    let decided = within_budget(move || rewrite_bash(&dd, &cwd, &cmd, &exe));
    if let Some(Some(rewritten)) = decided {
        print_served(&rewritten);
    }
    Ok(())
}

/// The hook answer that runs `rewritten` instead of the call. `allow` is
/// what lets it win over another hook's rewrite of the same call (rtk turns
/// `cat`/`grep` into `rtk read`/`rtk grep`; Claude Code applies the
/// `updatedInput` of the hook that also decided).
fn print_served(rewritten: &str) {
    println!(
        "{}",
        json!({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": "indexio: answered from the index",
                "updatedInput": { "command": rewritten },
            }
        })
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_liners_are_recognised_with_any_flag_order() {
        let w = |s: &str| words(s);
        assert_eq!(one_liner_code(&w("python -c 'print(1)'")), Some("print(1)"));
        assert_eq!(one_liner_code(&w("python -X utf8 -c \"x=1\"")), Some("x=1"));
        assert_eq!(one_liner_code(&w("python3 -u -c 'x'")), Some("x"));
        assert_eq!(one_liner_code(&w("node -e 'console.log(1)'")), Some("console.log(1)"));
        assert_eq!(one_liner_code(&w("python scripts/probe.py -c")), None, "a script file");
        assert_eq!(one_liner_code(&w("python -m pytest -c x")), None);
        assert_eq!(one_liner_code(&w("cargo build")), None);
        // and the runner rewrite leaves them alone
        let rw = |c: &str| runner_rewrite(Path::new("C:/Users/x/.cargo/bin/indexio.exe"), c);
        assert_eq!(rw("python -X utf8 -c \"print(open('a').read())\""), None);
    }

    #[test]
    fn read_only_one_liners_are_reads_in_disguise() {
        let stage = "python -X utf8 -c \"\nsrc=open(r'C:/x/crates/se-serve/src/lib.rs',encoding='utf-8').read()\ni=src.find('\\\"goto\\\" =>')\nprint(src[i:i+3000])\n\"";
        let (path, needle) = script_read(stage).expect("a read");
        assert_eq!(path, "C:/x/crates/se-serve/src/lib.rs");
        assert_eq!(needle.as_deref(), Some("\"goto\" =>"));
        assert_eq!(grep_literal("fn settle("), "fn settle\\(");
        // no needle: outline + span
        let (p, n) = script_read("python -c \"print(open('src/a.rs').read()[:400])\"").unwrap();
        assert_eq!((p.as_str(), n), ("src/a.rs", None));
        // node too
        let (p, n) = script_read("node -e \"const s=require('fs').readFileSync('src/a.ts','utf8'); console.log(s.indexOf('foo'))\"").unwrap();
        assert_eq!((p.as_str(), n.as_deref()), ("src/a.ts", Some("foo")));
        // writes, spawns and variables are not reads
        assert!(script_read("python -c \"open('a.rs','w').write('x')\"").is_none());
        assert!(script_read("python -c \"import subprocess; open('a.rs').read()\"").is_none());
        assert!(script_read("python -c \"open(p).read()\"").is_none());
        assert!(script_read("python patch.py").is_none());
        assert!(!is_other_work(&["python -c \"print(open('a.rs').read())\"".to_string()]));
        assert!(is_other_work(&["python patch.py".to_string()]));
        // byte questions are not reads: the shell answers them
        assert!(script_read("python -c \"b=open('src/a.rs','rb').read(); print(b.count(b'\\r\\n'))\"").is_none());
        assert!(script_read("python -c \"print(repr(open('a.rs').read()[:80]))\"").is_none());
    }

    #[test]
    fn tail_counts() {
        let s = |x: &str| x.split(' ').filter(|w| !w.is_empty()).map(String::from).collect::<Vec<_>>();
        assert_eq!(tail_count(&s("-n 20 a.rs")), Some(20));
        assert_eq!(tail_count(&s("-20 a.rs")), Some(20));
        assert_eq!(tail_count(&s("-n5 a.rs")), Some(5));
        assert_eq!(tail_count(&s("--lines=7 a.rs")), Some(7));
        assert_eq!(tail_count(&s("a.rs")), Some(10));
        assert_eq!(tail_count(&s("-n +40 a.rs")), None);
    }

    #[test]
    fn counts_and_byte_views_are_not_line_reads() {
        let s = |x: &str| x.split(' ').map(String::from).collect::<Vec<_>>();
        assert!(shows_bytes("cat", &s("-A a.rs")));
        assert!(shows_bytes("cat", &s("-vet a.rs")));
        assert!(!shows_bytes("cat", &s("-n a.rs")));
        assert!(shows_bytes("tail", &s("-f run.log")));
        assert!(shows_bytes("head", &s("-c 200 a.bin")));
        assert!(!shows_bytes("head", &s("-n 20 a.rs")));
        // counts, file lists, matched parts, inverted matches, filters and
        // context before a match are not lines the index returns
        for (cmd, args) in [
            ("grep", "-c foo a.rs"),
            ("grep", "-rl foo src"),
            ("rg", "-o foo"),
            ("grep", "-rhoE x src"),
            ("grep", "--count foo a"),
            ("grep", "-v foo a.rs"),
            ("grep", "-rn --include=*.rs foo src"),
            ("rg", "-t rust foo"),
            ("grep", "-B3 foo a.rs"),
            ("grep", "-e a -e b a.rs"),
            ("grep", "foo"),
            ("rg", "-r x foo"),
        ] {
            assert!(parse_grep(cmd, &s(args)).is_none(), "{cmd} {args}");
        }
        for (cmd, args) in [("grep", "-n foo a.rs"), ("grep", "-rn -A3 foo src"), ("rg", "-n foo"), ("grep", "-rni foo .")] {
            assert!(parse_grep(cmd, &s(args)).is_some(), "{cmd} {args}");
        }
    }

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
    fn sed_scripts() {
        let a = |x: &str| x.split(' ').map(String::from).collect::<Vec<_>>();
        assert_eq!(sed_ranges(&a("-n 10,20p")), vec![(10, 20)]);
        assert_eq!(sed_ranges(&a("-n '7p'")), vec![(7, 7)]);
        assert_eq!(sed_ranges(&a("-n s/a/b/p")), vec![]);
        assert_eq!(sed_ranges(&a("-n '3160,3163p;3574,3577p' f.rs")), vec![(3160, 3163), (3574, 3577)]);
        assert_eq!(sed_ranges(&a("-n -e 5,9p -e 20,30p f.rs")), vec![(5, 9), (20, 30)]);
        assert_eq!(sed_ranges(&a("-n 5,9p;/x/p f.rs")), vec![]);
    }

    #[test]
    fn redirects_and_consumers_are_not_reads() {
        assert!(has_redirect("cat > out.txt"));
        assert!(has_redirect("cat file <<EOF"));
        assert!(!has_redirect("echo '>' x"));
    }

    #[test]
    fn grep_option_values_are_not_paths() {
        let a: Vec<String> = ["-n", "class Blocked", "-A", "12", "src/errors.py"].iter().map(|s| s.to_string()).collect();
        let g = parse_grep("grep", &a).unwrap();
        assert_eq!(g.pattern, "class Blocked");
        assert_eq!(g.paths, vec!["src/errors.py"]);
        assert_eq!(g.after, Some(12));
        assert!(!g.ere);
        let b: Vec<String> = ["-n", "-e", "foo|bar", "-A3", "crates", "docs"].iter().map(|s| s.to_string()).collect();
        let g = parse_grep("rg", &b).unwrap();
        assert_eq!(g.pattern, "foo|bar");
        assert_eq!(g.paths, vec!["crates", "docs"]);
        assert_eq!(g.after, Some(3));
        assert!(g.ere);
        // `-E` inside a bundle is extended syntax too (`-nE "^a|^b"` was
        // turned into `^a\|^b`, a literal bar)
        let c: Vec<String> = ["-nE", "^stage 1|^trial +15 ", "funnel.log"].iter().map(|s| s.to_string()).collect();
        let g = parse_grep("grep", &c).unwrap();
        assert!(g.ere);
        assert_eq!(grep_regex(&g), "^stage 1|^trial +15 ");
        let d: Vec<String> = ["-rniw", "foo", "src"].iter().map(|s| s.to_string()).collect();
        let g = parse_grep("grep", &d).unwrap();
        assert!(g.icase && g.recursive);
        assert_eq!(grep_regex(&g), r"\b(?:foo)\b");
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

    /// A plain folder's PDF is indexed by name only: a Read of it is the
    /// way to its contents and is allowed; its text neighbour is redirected.
    /// A nested git worktree (`.git` file) is not part of the folder.
    #[test]
    fn read_hook_allows_name_only_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data");
        let root = tmp.path().join("folder");
        std::fs::create_dir_all(root.join("wt")).unwrap();
        std::fs::write(root.join("notes.md"), format!("# Notes\n{}", "some text\n".repeat(100))).unwrap();
        std::fs::write(root.join("short.md"), b"# Short\nsome text\n").unwrap();
        std::fs::write(root.join("spec.pdf"), b"%PDF\0binary").unwrap();
        std::fs::write(root.join("wt/.git"), b"gitdir: elsewhere\n").unwrap();
        std::fs::write(root.join("wt/lib.rs"), b"fn x() {}\n").unwrap();
        let cas = indexio_ingest::Cas::open(&data.join("cas")).unwrap();
        let r = indexio_ingest::index_dir(&root, "folder", &data, &cas).unwrap();
        assert_eq!(r.docs_added, 3, "notes.md, short.md + spec.pdf by name; the worktree is skipped");
        assert_eq!(read_verdict(&data, None, &root.join("short.md").to_string_lossy(), None, None), None, "a short file read whole");
        assert!(read_verdict(&data, None, &root.join("notes.md").to_string_lossy(), None, None).is_some());
        assert_eq!(read_verdict(&data, None, &root.join("spec.pdf").to_string_lossy(), None, None), None);
        // a short read is cheaper as it is than through a deny
        assert_eq!(read_verdict(&data, None, &root.join("notes.md").to_string_lossy(), Some(1), Some(40)), None);
        assert!(read_verdict(&data, None, &root.join("notes.md").to_string_lossy(), Some(1), Some(200)).is_some());
    }

    #[test]
    fn a_worktree_copy_identical_to_the_indexed_file_is_redirected() {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data");
        let root = tmp.path().join("folder");
        let body = format!("# Notes\n{}", "some text\n".repeat(100));
        std::fs::create_dir_all(root.join("docs")).unwrap();
        std::fs::write(root.join("docs/notes.md"), &body).unwrap();
        for (wt, text) in [("agent-1", body.replace('\n', "\r\n")), ("agent-2", format!("{body}edited\n"))] {
            let dir = root.join(".claude/worktrees").join(wt);
            std::fs::create_dir_all(dir.join("docs")).unwrap();
            std::fs::write(dir.join(".git"), b"gitdir: elsewhere\n").unwrap();
            std::fs::write(dir.join("docs/notes.md"), text).unwrap();
        }
        let cas = indexio_ingest::Cas::open(&data.join("cas")).unwrap();
        let r = indexio_ingest::index_dir(&root, "folder", &data, &cas).unwrap();
        assert_eq!(r.docs_added, 1, "the worktrees are not indexed");
        let at = |wt: &str| root.join(".claude/worktrees").join(wt).join("docs/notes.md").to_string_lossy().into_owned();
        let same = read_verdict(&data, None, &at("agent-1"), None, None).expect("same bytes (CRLF aside): denied");
        assert!(same.contains("path:\"docs/notes.md\"") && same.contains("byte-identical"), "{same}");
        assert_eq!(read_verdict(&data, None, &at("agent-2"), None, None), None, "the agent's edit is its own");
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
        assert_eq!(changes.len(), 5, "{changes:?}");
        let pre = s["hooks"]["PreToolUse"].as_array().unwrap();
        assert_eq!(pre.len(), 4);
        assert_eq!(pre[0]["hooks"][0]["command"], "rtk hook claude");
        assert_eq!(pre[1]["hooks"][0]["command"], "C:/Users/x/.cargo/bin/indexio.exe hook bash");
        assert_eq!(pre[2]["matcher"], "PowerShell");
        assert_eq!(pre[2]["hooks"][0]["command"], "C:/Users/x/.cargo/bin/indexio.exe hook powershell");
        assert_eq!(pre[3]["matcher"], "Read");
        assert_eq!(pre[3]["hooks"][0]["command"], "C:/Users/x/.cargo/bin/indexio.exe hook read");
        assert_eq!(s["hooks"]["PreCompact"][0]["hooks"][0]["command"], "C:/Users/x/.cargo/bin/indexio.exe sessions --quiet");
        assert!(s["hooks"]["PreCompact"][0].get("matcher").is_none());
        assert_eq!(s["hooks"]["SessionStart"][0]["hooks"][0]["command"], "C:/Users/x/.cargo/bin/indexio.exe hook session-start");
        assert_eq!(s["permissions"]["allow"][0], "mcp__indexio");
        // idempotent
        assert!(install_settings(&mut s, exe).is_empty());
        // spaces are quoted
        assert_eq!(hook_command(Path::new("C:\\Program Files\\indexio.exe"), "hook bash"), "\"C:/Program Files/indexio.exe\" hook bash");
    }

    #[test]
    fn a_resent_read_goes_through() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        let cmd = "Read src/lib.rs None None";
        // first denial records the call
        assert!(!repeated_denial(d, "s1", cmd));
        assert!(!repeated_denial(d, "s2", cmd), "per session");
        // the identical re-send repeats (and goes through); the one after starts over
        assert!(repeated_denial(d, "s1", cmd));
        assert!(!repeated_denial(d, "s1", cmd));
        assert!(repeated_denial(d, "s1", cmd));
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
        assert_eq!(rw("node run.js 'x(1)'").as_deref(), Some(r"C:/Users/x/.cargo/bin/indexio.exe run -- 'node run.js '\''x(1)'\'''"), "inner quotes escaped");
        assert_eq!(rw("node -e 'x(1)'"), None, "a one-liner prints what it was asked for");
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
    fn pipelines_keep_their_place_in_the_command() {
        let cmd = "cd x && cat 'a b.rs' | head -5;  grep -n \"x;y\" src || echo no";
        let spans = split_spans(cmd);
        let texts: Vec<&str> = spans.iter().map(|(r, _)| &cmd[r.clone()]).collect();
        assert_eq!(texts, vec!["cd x", "cat 'a b.rs' | head -5", "grep -n \"x;y\" src", "echo no"]);
        assert_eq!(spans[1].1, vec!["cat 'a b.rs'", "head -5"]);
        // multi-byte text before a pipeline
        let cmd = "echo 'é→' && cat f";
        let spans = split_spans(cmd);
        assert_eq!(&cmd[spans[1].0.clone()], "cat f");
    }

    #[test]
    fn grep_classes_stay_on_one_line() {
        assert_eq!(line_local(r"^\s*mod "), r"^[^\S\n]*mod ");
        assert_eq!(line_local(r"a[^;]*;"), r"a[^\n;]*;");
        assert_eq!(line_local(r"[\s,]x\W\D\.\\s"), r"[ \t\r\f\v,]x[^\w\n][^\d\n]\.\\s");
        assert_eq!(line_local(r"[]a]b[^]c]"), r"[]a]b[^\n]c]");
        assert_eq!(line_local(r"^[[:space:]]*#"), r"^[ \t\r\f\v]*#");
        assert_eq!(line_local(r"[[:alpha:]_]+"), r"[[:alpha:]_]+");
        for re in [r"^\s*mod ", r"a[^;]*;", r"[\s,]x\W\D", r"[]a]b[^]c]", r"^[[:space:]]*#"] {
            assert!(regex::Regex::new(&line_local(re)).is_ok(), "{re}");
        }
    }

    #[test]
    fn base64url_round_trips() {
        for s in ["", "a", "ab", "abc", "{\"repo\":\"x y\",\"pattern\":\"a|b'c\\\"\"}", "é→\u{1F600}"] {
            let e = b64url_encode(s.as_bytes());
            assert!(e.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'), "{e}");
            assert_eq!(b64url_decode(&e).unwrap(), s.as_bytes());
        }
        assert!(b64url_decode("a+b").is_none());
    }

    #[test]
    fn comments_and_substitutions_are_left_alone() {
        assert!(has_unmodelled_syntax("cat f # look"));
        assert!(has_unmodelled_syntax("cat $(ls | head -1)"));
        assert!(has_unmodelled_syntax("cat `ls`"));
        assert!(has_unmodelled_syntax("diff <(cat a) b"));
        assert!(!has_unmodelled_syntax("grep -n 'a#b $(x)' f"));
        assert!(!has_unmodelled_syntax("grep -n \"issue#4\" f"));
        assert!(!has_unmodelled_syntax("sed -n 1,80p a.rs"));
    }

    /// An indexed folder: `code.rs` (90 lines, 6 functions), `notes.md`
    /// (100 lines, one heading), `short.md` (3 lines).
    fn served_fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data");
        let root = tmp.path().join("folder");
        std::fs::create_dir_all(root.join("src")).unwrap();
        let mut code = String::new();
        for f in 0..6 {
            code.push_str(&format!("fn f{f}() {{\n"));
            for l in 0..13 {
                code.push_str(&format!("    let x{l} = {l}; // Needle{f}\n"));
            }
            code.push_str("}\n");
        }
        std::fs::write(root.join("src/code.rs"), &code).unwrap();
        std::fs::write(root.join("notes.md"), format!("# Notes\n{}", "some text\n".repeat(99))).unwrap();
        std::fs::write(root.join("short.md"), "# Short\nsome text\n").unwrap();
        let cas = indexio_ingest::Cas::open(&data.join("cas")).unwrap();
        indexio_ingest::index_dir(&root, "folder", &data, &cas).unwrap();
        (tmp, data, root)
    }

    /// The calls a command's reads are answered with: one line per served
    /// pipeline (`index: tool args…`), `[]` when it runs as typed.
    fn served(data: &Path, cwd: &Path, cmd: &str) -> Vec<String> {
        plan(data, cwd, cmd)
            .into_iter()
            .map(|(i, (s, keep))| {
                let calls: Vec<String> = s.calls.iter().map(|(n, a)| format!("{n} {a}")).collect();
                format!("{i}: {}{}", calls.join(" + "), keep.map(|k| format!(" | {k}")).unwrap_or_default())
            })
            .collect()
    }

    #[test]
    fn shell_reads_are_answered_in_place() {
        let (_t, data, root) = served_fixture();
        let s = |c: &str| served(&data, &root, c);
        // a whole read of a file with definitions: its outline (and a note)
        assert_eq!(s("cat src/code.rs"), vec![r#"0: file_outline {"path":"src/code.rs","repo":"folder"}"#]);
        let p = plan(&data, &root, "cat src/code.rs");
        assert!(p[0].1 .0.note.as_deref().is_some_and(|n| n.contains("90 lines")), "{:?}", p[0].1 .0.note);
        // one without: its lines
        assert_eq!(s("cat notes.md"), vec![r#"0: read_span {"end":100,"path":"notes.md","repo":"folder","start":1}"#]);
        // `| head -N` folds into the range; a small read stays the shell's
        assert_eq!(s("cat notes.md | head -80"), vec![r#"0: read_span {"end":80,"path":"notes.md","repo":"folder","start":1}"#]);
        assert!(s("cat notes.md | head -20").is_empty());
        assert!(s("cat short.md").is_empty());
        assert!(s("head -20 src/code.rs").is_empty());
        assert_eq!(s("sed -n 5,90p src/code.rs"), vec![r#"0: read_span {"end":90,"path":"src/code.rs","repo":"folder","start":5}"#]);
        assert_eq!(s("tail -n 70 notes.md"), vec![r#"0: read_span {"end":100,"path":"notes.md","repo":"folder","start":31}"#]);
        // a grep: case-sensitive like grep, `-i` not, cut by `| head`
        assert_eq!(
            s("grep -n Needle3 src/code.rs | head -5"),
            vec![r#"0: code_grep {"lines":5,"pattern":"Needle3 case:yes repo:folder path:^src/code.rs$"}"#]
        );
        assert_eq!(s("grep -rni needle src"), vec![r#"0: code_grep {"pattern":"needle case:no repo:folder path:^src/"}"#]);
        // a definition with its body
        assert_eq!(s("grep -n 'fn f2' -A 40 src/code.rs"), vec![r#"0: read_span {"end":45,"path":"src/code.rs","repo":"folder","start":31}"#]);
        // several pipelines: each answered in place, the rest left as typed
        assert_eq!(
            s("cd src && sed -n 1,80p code.rs; echo ---; grep -n Needle1 code.rs"),
            vec![
                r#"1: read_span {"end":80,"path":"src/code.rs","repo":"folder","start":1}"#.to_string(),
                r#"3: code_grep {"pattern":"Needle1 case:yes repo:folder path:^src/code.rs$"}"#.to_string(),
            ]
        );
        // what the index cannot answer as asked stays the shell's
        for cmd in [
            "cat notes.md | grep text",          // a filter after a read
            "grep -v Needle src/code.rs",        // inverted
            "grep -c Needle src/code.rs",        // a count
            "grep -n Needle -B 2 src/code.rs",   // context before
            "sed 's/a/b/' notes.md",             // a transform
            "sed -n '/fn/,/}/p' src/code.rs",    // regex addresses
            "cat notes.md > copy.md",            // a write
            "cat notes.md # why",                // a comment
            "python build.py && cat notes.md",   // other work in the call
            "find . -name '*.rs' -delete",       // not a listing
            "grep -n Needle missing.rs",         // not a file of the index
        ] {
            assert!(s(cmd).is_empty(), "{cmd}: {:?}", s(cmd));
        }
        assert_eq!(s("find . -name '*.rs' | head -3"), vec![r#"0: list_files {"pattern":"**/*.rs","repo":"folder"} | head -3"#]);
    }

    #[test]
    fn the_rewrite_falls_back_to_the_original() {
        let (_t, data, root) = served_fixture();
        let exe = Path::new("C:/Program Files/indexio/indexio.exe");
        let out = rewrite_bash(&data, &root, "cd src && grep -n Needle1 code.rs | head -3 && echo done", exe).unwrap();
        assert!(out.starts_with("cd src && { \"C:/Program Files/indexio/indexio.exe\" --data-dir '"), "{out}");
        assert!(out.ends_with(" tool code_grep ") || out.contains(" tool code_grep "), "{out}");
        assert!(out.contains(" || grep -n Needle1 code.rs | head -3; } && echo done"), "{out}");
        // over a folder the cut stays, widened for the per-file lines
        let dir = rewrite_bash(&data, &root, "grep -rn Needle1 src | head -3", exe).unwrap();
        assert!(dir.ends_with(" || grep -rn Needle1 src; } | head -n 7"), "{dir}");
        // the payload carries the repo to freshen and the session's repo
        let b64 = out.split(" tool code_grep ").nth(1).unwrap().split(' ').next().unwrap();
        let v: Value = serde_json::from_slice(&b64url_decode(b64).unwrap()).unwrap();
        assert_eq!(v["_fresh"], json!([["folder", "src/code.rs"]]));
        assert_eq!(v["_current"], json!("folder"));
        let out = rewrite_bash(&data, &root, "sed -n 1,80p src/code.rs", exe).unwrap();
        assert!(out.ends_with(" || sed -n 1,80p src/code.rs; }"), "{out}");
        assert_eq!(rewrite_bash(&data, &root, "cat short.md", exe), None);
    }

    #[test]
    fn powershell_reads_are_answered_in_place() {
        let (_t, data, root) = served_fixture();
        let exe = Path::new("C:/x/indexio.exe");
        let rw = |c: &str| ps::rewrite(&data, &root, c, exe);
        let out = rw("Select-String -Path src/code.rs -Pattern 'Needle4' | Select-Object -Last 6").unwrap();
        assert!(out.starts_with("$global:LASTEXITCODE = 1; & 'C:/x/indexio.exe' --data-dir '"), "{out}");
        assert!(out.contains(" tool code_grep ") && out.contains(" | Select-Object -Last 13; if ($LASTEXITCODE -ne 0) { Select-String"), "{out}");
        assert!(rw("Get-Content src/code.rs -TotalCount 80").unwrap().contains(" tool read_span "));
        // shapes the answer would not keep: the shell's
        assert_eq!(rw("Get-Content notes.md | Where-Object { $_ -match 'x' }"), None);
        assert_eq!(rw("Select-String -Path src/code.rs -Pattern 'x' | ForEach-Object { $_.LineNumber }"), None);
        assert_eq!(rw("Get-Content short.md"), None);
    }

    #[test]
    fn a_caller_without_indexio_tools_reads_files_itself() {
        let tmp = tempfile::tempdir().unwrap();
        let t = tmp.path().join("s1.jsonl");
        let fail = r#"{"type":"user","message":{"content":[{"type":"tool_result","content":"<tool_use_error>Error: No such tool available: mcp__indexio__read_span. Its MCP server 'indexio' has disconnected.</tool_use_error>"}]}}"#;
        let call = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","name":"mcp__indexio__read_span","input":{}}]}}"#;
        std::fs::write(&t, format!("{call}\n{fail}\n")).unwrap();
        assert!(transcript_says_unavailable(&t));
        std::fs::write(&t, format!("{call}\n{fail}\n{call}\n")).unwrap();
        assert!(!transcript_says_unavailable(&t), "a later call: the server is back");
        // a subagent's own transcript
        let sub = tmp.path().join("s1").join("subagents");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("agent-a1.jsonl"), format!("{call}\n{fail}\n")).unwrap();
        std::fs::write(&t, format!("{call}\n")).unwrap();
        let input = json!({"transcript_path": t.to_string_lossy(), "agent_id": "a1", "agent_type": "general-purpose"});
        assert!(indexio_unreachable(&input, None));
        assert!(!indexio_unreachable(&json!({"transcript_path": t.to_string_lossy()}), None));
        // agent definitions
        let agents = tmp.path().join(".claude").join("agents");
        std::fs::create_dir_all(&agents).unwrap();
        std::fs::write(agents.join("reader.md"), "---\nname: reader\ntools: Read, Grep\n---\nbody").unwrap();
        std::fs::write(agents.join("full.md"), "---\nname: full\n---\nbody").unwrap();
        std::fs::write(agents.join("idx.md"), "---\ntools: Read, mcp__indexio__read_span\n---\n").unwrap();
        let cwd = Some(tmp.path());
        assert!(agent_lacks_indexio("reader", cwd, None));
        assert!(!agent_lacks_indexio("full", cwd, None));
        assert!(!agent_lacks_indexio("idx", cwd, None));
        assert!(agent_lacks_indexio("claude-code-guide", None, None));
        assert!(!agent_lacks_indexio("Explore", cwd, None));
    }

    #[test]
    fn git_bash_paths_resolve() {
        assert_eq!(resolve(Path::new("C:/x"), "/c/Users/me/a.rs"), PathBuf::from("C:/Users/me/a.rs"));
        assert_eq!(resolve(Path::new("C:/x"), "src/a.rs"), PathBuf::from("C:/x").join("src/a.rs"));
    }
}
