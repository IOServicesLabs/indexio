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

#[path = "hook_ps.rs"]
mod ps;

/// Escape hint, appended only when the SAME command is denied a second
/// time in a session (SPEC-P10): shown on every deny, the model learned it
/// from the first reason and wrote `# raw` on its next shell read instead
/// of the indexio call.

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
/// still a read, not a program consuming the file). Counting (`wc`,
/// `uniq -c`) and byte views (`cat -A`) are not: their answer is a number
/// or the invisible characters, which no indexio call returns — denying
/// `grep -n x f | cat -A` made the model send it again.
fn is_viewer(stage: &str) -> bool {
    let w = words(stage);
    let Some(cmd) = w.first().map(String::as_str) else { return false };
    matches!(
        cmd,
        "head" | "tail" | "grep" | "rg" | "egrep" | "fgrep" | "sed" | "awk" | "cut" | "sort" | "uniq" | "less" | "more" | "tr" | "nl" | "cat" | "column" | "tee"
    ) && !w.iter().any(|a| a == "-i")
        && !shows_bytes(cmd, &w[1..])
        && !(cmd == "uniq" && w[1..].iter().any(|a| a == "-c" || a == "--count"))
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

/// A parsed grep/rg command line: the pattern, the path operands, the
/// `-A` context and whether the pattern is already extended syntax.
struct GrepArgs {
    pattern: Option<String>,
    paths: Vec<String>,
    after: Option<u32>,
    ere: bool,
    /// `-c`, `-l`, `-L`, `-q`, `-o`: a count, file names, a status or the
    /// matched parts only — all smaller than the lines an indexio call
    /// returns, so the grep is the cheaper answer.
    not_lines: bool,
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
        not_lines: false,
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
        if matches!(
            a.split('=').next().unwrap_or(""),
            "--count" | "--files-with-matches" | "--files-without-match" | "--quiet" | "--only-matching" | "--count-matches"
        ) {
            g.not_lines = true;
        }
        // a bundle's flags up to the first one that takes a value
        // (`-rlc`, `-rho`; in `-A3c` the `3c` is a value)
        for c in short_flags(a).chars() {
            if matches!(c, 'c' | 'l' | 'L' | 'q' | 'o') {
                g.not_lines = true;
            }
            if matches!(c, 'A' | 'B' | 'C' | 'm' | 'e' | 'f' | 'g' | 't' | 'T' | 'd' | 'D' | 'j') {
                break;
            }
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

/// The verdict for one Bash command: `Some(reason)` = deny.
pub fn verdict(data_dir: &Path, cwd: &Path, command: &str) -> Option<String> {
    let mut reason = judge(data_dir, cwd, command)?;
    // the model bundles several reads per call (`sed -n … ; grep -n … | head`):
    // the deny is for the servable one, the rest is fine to rerun alone
    let others = split(command)
        .iter()
        .filter(|p| p.first().is_some_and(|s| !s.starts_with("cd ") && s != "cd"))
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

fn judge(data_dir: &Path, cwd: &Path, command: &str) -> Option<String> {
    if command.contains("INDEXIO_RAW") {
        return None;
    }
    // the shell is needed anyway: let the whole call through
    if split(command).iter().any(|p| is_other_work(p)) {
        return None;
    }
    let t0 = std::time::Instant::now();
    let repos = repos(data_dir);
    let debug = std::env::var_os("INDEXIO_HOOK_DEBUG").is_some();
    if debug {
        eprintln!("hook: cwd={} repos={} cmd={command:?} ({} us)", cwd.display(), repos.len(), t0.elapsed().as_micros());
    }
    if repos.is_empty() {
        return None;
    }
    // the session's repo: its MCP server refreshes it, the hook need not
    let own = locate(&repos, cwd).map(|(r, _)| r);
    // opened on the first file that needs it
    let mut engine: Option<Engine> = None;
    let indexed = |engine: &mut Option<Engine>, repo: &str, rel: &str| -> bool {
        if engine.is_none() {
            *engine = Engine::open(data_dir).ok();
        }
        engine.as_ref().is_some_and(|e| contents_indexed(e, repo, rel))
    };
    // the same for a file read: the path to point at (see `indexed_rel`)
    let indexed_as = |engine: &mut Option<Engine>, repo: &str, rel: &str, abs: &Path| -> Option<String> {
        if engine.is_none() {
            *engine = Engine::open(data_dir).ok();
        }
        indexed_rel(engine.as_ref()?, repo, rel, abs)
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
                if shows_bytes(cmd, args) {
                    continue;
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
                    if debug {
                        eprintln!("hook: located ({} us)", t0.elapsed().as_micros());
                    }
                    let Some(rel) = indexed_as(&mut engine, &repo, &rel, &abs) else {
                        if debug {
                            eprintln!("hook: {repo}:{rel} not indexed");
                        }
                        continue;
                    };
                    if debug {
                        eprintln!("hook: indexed checked ({} us)", t0.elapsed().as_micros());
                    }
                    let ranges: Vec<(u32, u32)> = if cmd == "sed" {
                        sed_ranges(args)
                    } else if cmd == "head" {
                        args.iter()
                            .position(|x| x == "-n")
                            .and_then(|i| args.get(i + 1))
                            .and_then(|n| n.parse::<u32>().ok())
                            .or_else(|| args.iter().find_map(|x| x.strip_prefix("-n").and_then(|n| n.parse().ok())))
                            .map(|n| (1, n))
                            .or(Some((1, 10))) // plain `head f`
                            .into_iter()
                            .collect()
                    } else if cmd == "tail" {
                        let e = engine.as_ref();
                        tail_count(args)
                            .and_then(|n| tail_range(e?, &repo, &rel, n))
                            .map(|(s, e)| (s as u32, e as u32))
                            .into_iter()
                            .collect()
                    } else {
                        Vec::new()
                    };
                    // a few lines are cheaper read than denied (as for Read)
                    let lines: u64 = ranges.iter().map(|&(s, e)| u64::from(e.saturating_sub(s)) + 1).sum();
                    if !ranges.is_empty() && lines <= SMALL_READ_LINES {
                        continue;
                    }
                    if ranges.is_empty() && engine.as_ref().is_some_and(|e| is_small_file(e, &repo, &rel)) {
                        continue;
                    }
                    let call = if ranges.is_empty() {
                        format!("mcp__indexio__file_outline {{repo:\"{repo}\", path:\"{rel}\"}} then read_span for the lines you need")
                    } else {
                        ranges
                            .iter()
                            .map(|(s, e)| format!("mcp__indexio__read_span {{repo:\"{repo}\", path:\"{rel}\", start:{s}, end:{e}}}"))
                            .collect::<Vec<_>>()
                            .join(" + ")
                    };
                    freshen_later(data_dir, own.as_deref(), &repo, Some(&rel));
                    return Some(format!("{a} is indexed: use {call} instead (numbered, fresh)"));
                }
            }
            "grep" | "rg" | "egrep" | "fgrep" | "ag" | "ack" => {
                // a grep on a pipeline (`x | grep`) is never first here; this
                // is a filesystem search: pattern = first positional, paths =
                // the rest (none = cwd)
                let g = grep_args(cmd, args);
                if g.not_lines {
                    continue;
                }
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
                        Some((repo, rel)) if p.is_file() && !indexed(&mut engine, &repo, &rel) => return None,
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
                let mut fresh: Vec<&str> = targets.iter().map(|(r, _)| r.as_str()).collect();
                fresh.dedup();
                for repo in fresh {
                    freshen_later(data_dir, own.as_deref(), repo, None);
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
            "python" | "python3" | "py" | "node" | "deno" | "bun" | "ruby" | "perl" | "php" => {
                let Some((path, needle)) = script_read(first) else { continue };
                let abs = resolve(&cwd, &path);
                if !abs.is_file() {
                    continue;
                }
                let Some((repo, rel)) = locate(&repos, &abs) else { continue };
                let Some(rel) = indexed_as(&mut engine, &repo, &rel, &abs) else { continue };
                freshen_later(data_dir, own.as_deref(), &repo, Some(&rel));
                let call = match needle {
                    Some(n) => format!(
                        "mcp__indexio__code_grep {{pattern:\"{}\", repo:\"{repo}\", path:\"{rel}\"}} then read_span {{repo, path, start:line, end:line+60}}",
                        grep_literal(&n).replace('"', "\\\"")
                    ),
                    None => format!(
                        "mcp__indexio__file_outline {{repo:\"{repo}\", path:\"{rel}\"}} then read_span for the lines you need"
                    ),
                };
                return Some(format!("{path} is indexed: use {call} instead of a script that opens it (numbered, fresh)"));
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

/// Whether `# raw` on this command is a retry of the command the hook last
/// denied in `session` — the only case the marker escapes the hook
/// (SPEC-P10 §13: a model that had seen the marker once put it on every
/// later grep, unprompted).
fn raw_is_retry(data_dir: &Path, session: &str, command: &str) -> bool {
    if !command.contains("# raw") {
        return false;
    }
    let hash = indexio_types::BlobId::from_content(without_raw(command).as_bytes()).hex();
    std::fs::read_to_string(last_denied_path(data_dir, session)).is_ok_and(|prev| prev.trim() == hash)
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
        let key = format!("Read {file_path} {offset:?} {limit:?}");
        (!repeated_denial(&data_dir, &session, &key)).then_some(reason)
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
    let session = v["session_id"].as_str().unwrap_or("nosession").to_string();
    if raw_is_retry(data_dir, &session, &command) {
        return Ok(());
    }
    let dd = data_dir.to_path_buf();
    let decided = within_budget(move || {
        let reason = ps::verdict(&dd, &cwd, &without_raw(&command))?;
        (!repeated_denial(&dd, &session, &command)).then_some(reason)
    });
    if let Some(Some(reason)) = decided {
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
    if let Ok(exe) = std::env::current_exe() {
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
    let (dd, cmd, sess) = (data_dir.to_path_buf(), command.to_string(), session.to_string());
    let decided = within_budget(move || {
        let reason = verdict(&dd, &cwd, &judged)?;
        (!repeated_denial(&dd, &sess, &cmd)).then_some(reason)
    });
    if let Some(Some(reason)) = decided {
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
        assert!(!is_viewer("cat -A"));
        assert!(!is_viewer("wc -l"));
        assert!(!is_viewer("uniq -c"));
        assert!(is_viewer("head -20"));
        for (cmd, args) in [("grep", "-c foo a.rs"), ("grep", "-rl foo src"), ("rg", "-o foo"), ("grep", "-rhoE x src"), ("grep", "--count foo a")] {
            assert!(grep_args(cmd, &s(args)).not_lines, "{cmd} {args}");
        }
        for (cmd, args) in [("grep", "-n foo a.rs"), ("grep", "-rn -A3 foo src"), ("rg", "-n foo")] {
            assert!(!grep_args(cmd, &s(args)).not_lines, "{cmd} {args}");
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
    fn raw_marker_only_escapes_a_retry() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        let cmd = "sed -n 1,5p src/lib.rs";
        // a fresh `# raw` on a never-denied command is not a retry
        assert!(!raw_is_retry(d, "s1", "sed -n 1,5p src/lib.rs # raw"));
        // first denial records the command
        assert!(!repeated_denial(d, "s1", cmd));
        // now the marker on that command is a retry, on another it is not
        assert!(raw_is_retry(d, "s1", "sed -n 1,5p src/lib.rs # raw"));
        assert!(raw_is_retry(d, "s1", "sed -n 1,5p src/lib.rs  # raw"));
        assert!(!raw_is_retry(d, "s1", "sed -n 1,9p src/lib.rs # raw"));
        assert!(!raw_is_retry(d, "s2", "sed -n 1,5p src/lib.rs # raw"), "per session");
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
    fn git_bash_paths_resolve() {
        assert_eq!(resolve(Path::new("C:/x"), "/c/Users/me/a.rs"), PathBuf::from("C:/Users/me/a.rs"));
        assert_eq!(resolve(Path::new("C:/x"), "src/a.rs"), PathBuf::from("C:/x").join("src/a.rs"));
    }
}
