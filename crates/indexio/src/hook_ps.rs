//! `indexio hook powershell` — the Bash hook's rules for the PowerShell
//! tool (Claude Code on Windows offers both shells, and sessions there use
//! PowerShell for a third of their shell calls).
//!
//! Only plain lookups are judged: `Get-Content` / `gc` / `cat` / `type` of
//! one file (with `-TotalCount`, `-Tail`, `| Select-Object -First/-Skip`
//! or `(Get-Content f)[a..b]`), and `Select-String` over files
//! (`-Path`, or `Get-ChildItem -Recurse | Select-String`). The statements
//! around them may only change the location or set string variables; any
//! other statement (a build, a script, a write, a block) lets the whole call
//! through, as in the Bash hook.

use std::path::{Path, PathBuf};

use super::{contents_indexed, freshen_later, locate, repos, tail_range};
use indexio_query::Engine;

/// Split `s` at top-level occurrences of any of `seps` (outside quotes,
/// braces, parens and brackets). PowerShell quoting: `'…'` (with `''` for a
/// quote), `"…"` (with a backtick escape).
fn split_top(s: &str, seps: &[char]) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut depth = 0i32;
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        match quote {
            Some('"') if c == '`' => {
                cur.push(c);
                if let Some(n) = it.next() {
                    cur.push(n);
                }
                continue;
            }
            Some(q) if c == q => {
                cur.push(c);
                if q == '\'' && it.peek() == Some(&'\'') {
                    cur.push(it.next().unwrap());
                    continue;
                }
                quote = None;
                continue;
            }
            Some(_) => {
                cur.push(c);
                continue;
            }
            None => {}
        }
        match c {
            '\'' | '"' => quote = Some(c),
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => depth -= 1,
            _ if depth == 0 && seps.contains(&c) => {
                let t = cur.trim().to_string();
                if !t.is_empty() {
                    out.push(t);
                }
                cur.clear();
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    let t = cur.trim().to_string();
    if !t.is_empty() {
        out.push(t);
    }
    out
}

/// Words of one pipeline stage, quotes removed (`'it''s'` -> `it's`); a
/// list written with spaces (`a.rs, b.rs`) is one word.
fn ps_words(stage: &str) -> Vec<String> {
    let mut joined: Vec<String> = Vec::new();
    for w in split_top(stage, &[' ', '\t']) {
        match joined.last_mut() {
            Some(prev) if prev.ends_with(',') || w.starts_with(',') => prev.push_str(&w),
            _ => joined.push(w),
        }
    }
    let mut out = Vec::new();
    for w in joined {
        // a list (`'a','b'`) keeps its quotes for the caller to split
        let unq = if split_top(&w, &[',']).len() > 1 {
            w
        } else if w.len() >= 2 && w.starts_with('\'') && w.ends_with('\'') {
            w[1..w.len() - 1].replace("''", "'")
        } else if w.len() >= 2 && w.starts_with('"') && w.ends_with('"') {
            w[1..w.len() - 1].to_string()
        } else {
            w
        };
        out.push(unq);
    }
    out
}

/// Value of `-Name V` (any of `names`), and the positional words.
struct Args {
    named: Vec<(String, String)>,
    switches: Vec<String>,
    positional: Vec<String>,
}

fn parse_args(words: &[String], value_params: &[&str]) -> Args {
    let mut a = Args { named: Vec::new(), switches: Vec::new(), positional: Vec::new() };
    let mut i = 0;
    while i < words.len() {
        let w = &words[i];
        if let Some(name) = w.strip_prefix('-').filter(|n| n.chars().next().is_some_and(char::is_alphabetic)) {
            // `-Name:value`
            if let Some((n, v)) = name.split_once(':') {
                a.named.push((n.to_ascii_lowercase(), v.to_string()));
            } else if value_params.iter().any(|p| p.eq_ignore_ascii_case(name)) {
                if let Some(v) = words.get(i + 1) {
                    a.named.push((name.to_ascii_lowercase(), v.clone()));
                }
                i += 1;
            } else {
                a.switches.push(name.to_ascii_lowercase());
            }
        } else {
            a.positional.push(w.clone());
        }
        i += 1;
    }
    a
}

impl Args {
    fn get(&self, names: &[&str]) -> Option<&str> {
        self.named.iter().find(|(n, _)| names.iter().any(|x| x.eq_ignore_ascii_case(n))).map(|(_, v)| v.as_str())
    }
    fn has(&self, names: &[&str]) -> bool {
        self.switches.iter().any(|s| names.iter().any(|x| x.eq_ignore_ascii_case(s)))
    }
}

fn cmd_is(w: &str, names: &[&str]) -> bool {
    names.iter().any(|n| n.eq_ignore_ascii_case(w))
}

const GET_CONTENT: &[&str] = &["Get-Content", "gc", "cat", "type"];
const SELECT_STRING: &[&str] = &["Select-String", "sls"];
const SET_LOCATION: &[&str] = &["cd", "Set-Location", "sl", "chdir", "Push-Location", "pushd"];
const GET_CHILD: &[&str] = &["Get-ChildItem", "gci", "ls", "dir"];
/// Later pipeline stages that only shape the text a read produced.
const VIEWERS: &[&str] = &[
    "Select-Object", "select", "ForEach-Object", "%", "foreach", "Where-Object", "?", "where", "Format-Table", "ft",
    "Format-List", "fl", "Out-String", "Out-Host", "Sort-Object", "sort",
];

/// The variables a statement like `$p = 'C:\x\y.rs'` sets, applied to a
/// later word (`$p`, `"$p\z"`, `$($p)`): `None` if an unknown `$` remains.
fn expand(w: &str, vars: &[(String, String)]) -> Option<String> {
    let mut s = w.to_string();
    for (k, v) in vars {
        s = s.replace(&format!("$({k})"), v).replace(&format!("${{{k}}}"), v);
        // longest names first is overkill here; a name is followed by a
        // non-word character or the end
        let pat = format!("${k}");
        let mut out = String::new();
        let mut rest = s.as_str();
        while let Some(i) = rest.find(&pat) {
            let after = &rest[i + pat.len()..];
            if after.chars().next().is_some_and(|c| c.is_alphanumeric() || c == '_' || c == ':') {
                out.push_str(&rest[..i + pat.len()]);
            } else {
                out.push_str(&rest[..i]);
                out.push_str(v);
            }
            rest = after;
        }
        out.push_str(rest);
        s = out;
    }
    (!s.contains('$')).then_some(s)
}

fn resolve(cwd: &Path, p: &str) -> PathBuf {
    let p = p.trim();
    let p = if let Some(rest) = p.strip_prefix("~") {
        match std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
            Some(h) => format!("{}{rest}", PathBuf::from(h).display()),
            None => p.to_string(),
        }
    } else {
        p.to_string()
    };
    let path = PathBuf::from(&p);
    if path.is_absolute() { path } else { cwd.join(path) }
}

/// What one lookup statement asks for.
enum Lookup {
    /// lines of one file: (path, Some((first, last)) or the whole file, or
    /// the last n lines)
    Read(PathBuf, Option<(u64, u64)>, Option<u64>),
    /// several line ranges of one file
    Ranges(PathBuf, Vec<(u64, u64)>),
    /// a pattern over files or folders
    Search { paths: Vec<PathBuf>, pattern: String, simple: bool, case_sensitive: bool, after: Option<u32> },
}

/// `Select-Object -First n [-Skip m]` / `-Index a..b` after a read, as a
/// line range; `Some(None)` = a viewer that keeps every line.
fn select_range(stage_words: &[String]) -> Option<Option<(u64, u64)>> {
    let head = stage_words.first()?;
    if !cmd_is(head, VIEWERS) {
        return None;
    }
    if cmd_is(head, &["Select-Object", "select"]) {
        let a = parse_args(&stage_words[1..], &["First", "Last", "Skip", "Index", "ExpandProperty", "Property"]);
        if a.get(&["Last"]).is_some() {
            return Some(None);
        }
        let skip: u64 = a.get(&["Skip"]).and_then(|v| v.parse().ok()).unwrap_or(0);
        if let Some(n) = a.get(&["First"]).and_then(|v| v.parse::<u64>().ok()) {
            return Some(Some((skip + 1, skip + n.max(1))));
        }
        if let Some((x, y)) = a.get(&["Index"]).and_then(|v| v.split_once("..")) {
            if let (Ok(x), Ok(y)) = (x.trim().parse::<u64>(), y.trim().parse::<u64>()) {
                return Some(Some((x + 1, y.max(x) + 1)));
            }
        }
    }
    Some(None)
}

/// `Select-Object -Last n`.
fn select_last(stage_words: &[String]) -> Option<u64> {
    if !cmd_is(stage_words.first()?, &["Select-Object", "select"]) {
        return None;
    }
    parse_args(&stage_words[1..], &["First", "Last", "Skip", "Index", "ExpandProperty", "Property"]).get(&["Last"])?.parse().ok()
}

/// One statement (a pipeline) as a lookup, or `Err(())` when it is work.
/// `Ok(None)` = harmless (location change, string variable, echo).
fn statement(stmt: &str, cwd: &mut PathBuf, vars: &mut Vec<(String, String)>) -> Result<Option<Lookup>, ()> {
    // `$x = 'literal'` / `$x = "literal"` (no expansion inside)
    if let Some((lhs, rhs)) = stmt.split_once('=') {
        let lhs = lhs.trim();
        let rhs = rhs.trim();
        if lhs.starts_with('$') && !lhs.contains(['.', '[', ':']) && !rhs.starts_with('=') {
            let quoted = rhs.len() >= 2
                && ((rhs.starts_with('\'') && rhs.ends_with('\'')) || (rhs.starts_with('"') && rhs.ends_with('"')));
            if quoted && split_top(rhs, &[' ']).len() == 1 {
                let v = expand(&rhs[1..rhs.len() - 1], vars).ok_or(())?;
                vars.push((lhs[1..].to_string(), v));
                return Ok(None);
            }
            return Err(()); // a computed value: work
        }
    }
    // an echo or a separator string
    if (stmt.starts_with('\'') || stmt.starts_with('"')) && split_top(stmt, &[' ']).len() == 1 {
        return Ok(None);
    }
    // `(Get-Content f)[a..b]`
    if let Some(inner) = stmt.strip_prefix('(') {
        if let Some((call, idx)) = inner.split_once(")[") {
            let w = ps_words(call);
            if w.first().is_some_and(|c| cmd_is(c, GET_CONTENT)) {
                let a = parse_args(&w[1..], &["Path", "LiteralPath", "Encoding", "TotalCount", "Head", "First", "Tail", "Last", "ReadCount"]);
                let p = a.get(&["Path", "LiteralPath"]).map(str::to_string).or_else(|| a.positional.first().cloned()).ok_or(())?;
                let p = expand(&p, vars).ok_or(())?;
                let idx = idx.trim_end_matches(']');
                let range = idx.split_once("..").and_then(|(x, y)| Some((x.trim().parse::<u64>().ok()?, y.trim().parse::<u64>().ok()?)));
                let (x, y) = range.ok_or(())?;
                return Ok(Some(Lookup::Read(resolve(cwd, &p), Some((x + 1, y.max(x) + 1)), None)));
            }
        }
        return Err(());
    }
    if stmt.contains(['{', '}']) && !stmt.contains('|') {
        return Err(()); // a block: `if`, `foreach`, a script block
    }
    let stages = split_top(stmt, &['|']);
    let first = ps_words(stages.first().ok_or(())?);
    let head = first.first().ok_or(())?;
    if cmd_is(head, &["Write-Output", "Write-Host", "echo"]) && stages.len() == 1 {
        return Ok(None);
    }
    if cmd_is(head, SET_LOCATION) && stages.len() == 1 {
        let a = parse_args(&first[1..], &["Path", "LiteralPath"]);
        let p = a.get(&["Path", "LiteralPath"]).map(str::to_string).or_else(|| a.positional.first().cloned()).ok_or(())?;
        *cwd = resolve(cwd, &expand(&p, vars).ok_or(())?);
        return Ok(None);
    }
    let rest: Vec<Vec<String>> = stages[1..].iter().map(|s| ps_words(s)).collect();
    if cmd_is(head, GET_CONTENT) {
        let a = parse_args(&first[1..], &["Path", "LiteralPath", "Encoding", "TotalCount", "Head", "First", "Tail", "Last", "ReadCount", "Delimiter"]);
        if a.has(&["Wait", "Stream", "AsByteStream"]) || a.get(&["Encoding"]).is_some_and(|e| e.eq_ignore_ascii_case("byte")) {
            return Err(());
        }
        let p = a.get(&["Path", "LiteralPath"]).map(str::to_string).or_else(|| a.positional.first().cloned()).ok_or(())?;
        if a.positional.len() > 1 || p.contains([',', '*', '?']) {
            return Err(());
        }
        let p = expand(&p, vars).ok_or(())?;
        let mut range = a.get(&["TotalCount", "Head", "First"]).and_then(|n| n.parse::<u64>().ok()).map(|n| (1, n.max(1)));
        let mut tail = a.get(&["Tail", "Last"]).and_then(|n| n.parse::<u64>().ok());
        if tail.is_some() {
            range = None;
        }
        // `| Select-String x` on one file is a grep of it
        if let Some(sel) = rest.first().filter(|w| w.first().is_some_and(|c| cmd_is(c, SELECT_STRING))) {
            if rest.len() > 1 && !rest[1..].iter().all(|w| select_range(w).is_some()) {
                return Err(());
            }
            return search(&sel[1..], vec![resolve(cwd, &p)], vars);
        }
        for w in &rest {
            if let Some(n) = select_last(w) {
                if range.is_none() && tail.is_none() {
                    tail = Some(n);
                }
                continue;
            }
            match select_range(w) {
                Some(Some(r)) if range.is_none() && tail.is_none() => range = Some(r),
                Some(_) => {}
                None => return Err(()), // Measure-Object, a consumer, a write
            }
        }
        return Ok(Some(Lookup::Read(resolve(cwd, &p), range, tail)));
    }
    if cmd_is(head, SELECT_STRING) {
        if !rest.iter().all(|w| select_range(w).is_some()) {
            return Err(());
        }
        return search(&first[1..], Vec::new(), vars).map(|l| match l {
            Some(Lookup::Search { paths, pattern, simple, case_sensitive, after }) => Some(Lookup::Search {
                paths: paths.into_iter().map(|p| resolve(cwd, &p.to_string_lossy())).collect(),
                pattern,
                simple,
                case_sensitive,
                after,
            }),
            other => other,
        });
    }
    // `Get-ChildItem D -Recurse [-Include x] | Select-String p`
    if cmd_is(head, GET_CHILD) && rest.first().is_some_and(|w| w.first().is_some_and(|c| cmd_is(c, SELECT_STRING))) {
        if !rest[1..].iter().all(|w| select_range(w).is_some()) {
            return Err(());
        }
        let a = parse_args(&first[1..], &["Path", "LiteralPath", "Include", "Filter", "Exclude", "Depth"]);
        let d = a.get(&["Path", "LiteralPath"]).map(str::to_string).or_else(|| a.positional.first().cloned()).unwrap_or_else(|| ".".into());
        let d = expand(&d, vars).ok_or(())?;
        if d.contains([',', '*']) {
            return Err(());
        }
        return search(&rest[0][1..], vec![resolve(cwd, &d)], vars);
    }
    Err(())
}

/// `Select-String` arguments (the stage's words after the command) as a
/// search; `given` = the piped-in files.
fn search(words: &[String], given: Vec<PathBuf>, vars: &[(String, String)]) -> Result<Option<Lookup>, ()> {
    let a = parse_args(words, &["Pattern", "Path", "LiteralPath", "Context", "Encoding", "Include", "Exclude"]);
    if a.has(&["NotMatch", "List", "Quiet", "Raw"]) {
        return Err(()); // inverse matches, file lists, a status: smaller than lines
    }
    let mut pos = a.positional.iter();
    let pattern = match a.get(&["Pattern"]) {
        Some(p) => p.to_string(),
        None => pos.next().ok_or(())?.clone(),
    };
    // several patterns (`'a','b'`) become one alternation
    let parts = split_top(&pattern, &[',']);
    let pattern = parts.iter().map(|p| p.trim_matches(['\'', '"'])).collect::<Vec<_>>().join("|");
    // `"a \"b\""`: PowerShell does not escape with a backslash, so the
    // string ended early and the rest is not the pattern the model meant
    if pattern.contains('"') {
        return Err(());
    }
    let paths: Vec<PathBuf> = if given.is_empty() {
        let p = a.get(&["Path", "LiteralPath"]).map(str::to_string).or_else(|| pos.next().cloned()).ok_or(())?;
        let mut out = Vec::new();
        for one in split_top(&p, &[',']) {
            let one = expand(one.trim_matches(['\'', '"']), vars).ok_or(())?;
            out.push(PathBuf::from(one));
        }
        out
    } else {
        given
    };
    let after = a.get(&["Context"]).and_then(|c| c.rsplit(',').next()).and_then(|n| n.trim().parse().ok());
    Ok(Some(Lookup::Search {
        paths,
        pattern,
        simple: a.has(&["SimpleMatch"]),
        case_sensitive: a.has(&["CaseSensitive"]),
        after,
    }))
}

/// A range end meaning "the last line".
const TO_END: u64 = u64::MAX;

/// A numbered read of one file through a variable: `$c = Get-Content f`
/// then `$c[a..b]`, `for ($k=A; $k -le B; $k++) { … $c[$k-1] }` or a loop
/// over `@(@(a,b),…)` pairs (249 such calls, 728k chars in two weeks). The
/// whole call must only print: anything that writes, filters, runs a
/// program or uses the variable some other way leaves it to the shell.
fn var_read(command: &str, cwd: &Path) -> Option<Lookup> {
    const WORK: &[&str] = &[
        ">", "&", "set-", "out-file", "add-", "remove-", "copy-", "move-", "new-", "invoke-", "start-", "stop-",
        "rename-", "clear-", "export-", "import-", "[system.", "[io.", ".exe", "-replace", "-match", "-like",
        "where-object", "?{", "? {", "sort-object", "convert", "measure-object", "select-string", "sls ",
    ];
    let lower = command.to_ascii_lowercase();
    if WORK.iter().any(|w| lower.contains(w)) {
        return None;
    }
    let mut here = cwd.to_path_buf();
    let mut vars: Vec<(String, String)> = Vec::new();
    let mut read: Option<(String, PathBuf)> = None;
    let mut rest = String::new();
    for stmt in split_top(command, &[';', '\n', '\r']) {
        let stmt = stmt.trim();
        if read.is_some() {
            rest.push_str(stmt);
            rest.push('\n');
            continue;
        }
        if let Some((lhs, rhs)) = stmt.split_once('=') {
            let (lhs, rhs) = (lhs.trim(), rhs.trim());
            let w = ps_words(rhs);
            if lhs.starts_with('$') && w.first().is_some_and(|c| cmd_is(c, GET_CONTENT)) {
                let a = parse_args(&w[1..], &["Path", "LiteralPath", "Encoding"]);
                let p = a.get(&["Path", "LiteralPath"]).map(str::to_string).or_else(|| a.positional.first().cloned())?;
                if a.positional.len() > 1 || p.contains([',', '*', '?', '(']) {
                    return None;
                }
                read = Some((lhs[1..].to_string(), resolve(&here, &expand(&p, &vars)?)));
                continue;
            }
        }
        // what may come before the read: locations and string variables
        statement(stmt, &mut here, &mut vars).ok()?.is_none().then_some(())?;
    }
    let (var, path) = read?;
    let v = regex::escape(&var);
    let num = |m: Option<regex::Match>| m.and_then(|m| m.as_str().parse::<u64>().ok());
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    // `$c[a..b]` (0-based)
    for c in regex::Regex::new(&format!(r"\${v}\[(\d+)\s*\.\.\s*(\d+)\]")).ok()?.captures_iter(&rest) {
        ranges.push((num(c.get(1))? + 1, num(c.get(2))? + 1));
    }
    // `$c[a..($c.Count-1)]`: from a to the end
    for c in regex::Regex::new(&format!(r"\${v}\[(\d+)\s*\.\.\s*\(\${v}\.Count\s*-\s*1\)\]")).ok()?.captures_iter(&rest) {
        ranges.push((num(c.get(1))? + 1, TO_END));
    }
    // `for ($k=A; $k -le B; …) { … $c[$k-1] }` (1-based) or `$c[$k]` (0-based);
    // B may be `$c.Count` (to the end)
    let bound = format!(r"(\d+|\${v}\.Count)");
    let for_re = format!(r"for\s*\(\s*\$(\w+)\s*=\s*(\d+)\s*;\s*\$\w+\s*-(le|lt)\s*{bound}");
    for c in regex::Regex::new(&for_re).ok()?.captures_iter(&rest) {
        let k = &c[1];
        let base = u64::from(!rest.contains(&format!("[${k}-1]")) && !rest.contains(&format!("[${k} - 1]")));
        let a = num(c.get(2))? + base;
        let b = match num(c.get(4)) {
            Some(b) => b + base - u64::from(&c[3] == "lt"),
            None => TO_END,
        };
        ranges.push((a.max(1), b));
    }
    // `A..B | % { … $c[$_] }` (0-based)
    for c in regex::Regex::new(r"(?:^|[\s(])(\d+)\s*\.\.\s*(\d+)\s*\|").ok()?.captures_iter(&rest) {
        ranges.push((num(c.get(1))? + 1, num(c.get(2))? + 1));
    }
    // `foreach ($r in @(@(a,b),…)) { for ($k=$r[0]; $k -le $r[1]; …) }`
    if rest.contains("[0]") && rest.contains("[1]") {
        for c in regex::Regex::new(r"@\(\s*(\d+)\s*,\s*(\d+)\s*\)").ok()?.captures_iter(&rest) {
            ranges.push((num(c.get(1))?, num(c.get(2))?));
        }
    }
    let uses = rest.matches(&format!("${var}")).count();
    let indexed_uses = rest.matches(&format!("${var}[")).count();
    // the variable printed whole (`$c`, `$c | Select-Object …`)
    let whole = regex::Regex::new(&format!(r"(?m)(^|\|\s*)\${v}\s*(\||$)")).ok()?.is_match(&rest);
    if indexed_uses == 0 && !whole {
        return None; // only `$c.Count` and the like: a number, not the file
    }
    if uses > indexed_uses + rest.matches(&format!("${var}.Count")).count() + usize::from(whole) * uses {
        return None;
    }
    ranges.retain(|&(a, b)| a >= 1 && b >= a);
    // a loop over every line is the whole file
    if whole || ranges.is_empty() || ranges.iter().any(|&r| r == (1, TO_END)) {
        return Some(Lookup::Read(path, None, None));
    }
    ranges.sort_unstable();
    ranges.dedup();
    Some(Lookup::Ranges(path, ranges))
}

/// The verdict for one PowerShell command: `Some(reason)` = deny.
pub fn verdict(data_dir: &Path, cwd: &Path, command: &str) -> Option<String> {
    if command.contains("INDEXIO_RAW") || command.contains("# raw") {
        return None;
    }
    if let Some(l) = var_read(command, cwd) {
        return judge_lookup(data_dir, cwd, l);
    }
    let mut here = cwd.to_path_buf();
    let mut vars: Vec<(String, String)> = Vec::new();
    let mut lookups = Vec::new();
    for stmt in split_top(command, &[';', '\n', '\r']) {
        match statement(&stmt, &mut here, &mut vars) {
            Ok(Some(l)) => lookups.push(l),
            Ok(None) => {}
            Err(()) => return None, // the shell is needed anyway
        }
    }
    let more = lookups.len() > 1;
    let first = lookups.into_iter().next()?;
    let reason = judge_lookup(data_dir, cwd, first)?;
    Some(if more { format!("{reason} The rest of this call is fine to run on its own.") } else { reason })
}

fn judge_lookup(data_dir: &Path, cwd: &Path, first: Lookup) -> Option<String> {
    let repos = repos(data_dir);
    if repos.is_empty() {
        return None;
    }
    let own = locate(&repos, cwd).map(|(r, _)| r);
    let engine = Engine::open(data_dir).ok()?;
    match first {
        Lookup::Read(abs, range, tail) => {
            // a wildcard path that resolves to nothing, a folder: not ours
            if !abs.is_file() {
                return None;
            }
            let (repo, rel) = locate(&repos, &abs)?;
            let rel = super::indexed_rel(&engine, &repo, &rel, &abs)?;
            if range.is_none() && tail.is_none() && super::is_small_file(&engine, &repo, &rel) {
                return None; // a short file read whole
            }
            let range = range.or_else(|| tail_range(&engine, &repo, &rel, tail?));
            // a few lines are cheaper read than denied (as for Read)
            if range.is_some_and(|(s, e)| e.saturating_sub(s) < super::SMALL_READ_LINES) {
                return None;
            }
            freshen_later(data_dir, own.as_deref(), &repo, Some(&rel));
            let name = abs.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_else(|| rel.clone());
            Some(match range {
                Some((s, e)) => format!(
                    "{name} is indexed: use mcp__indexio__read_span {{repo:\"{repo}\", path:\"{rel}\", start:{s}, end:{e}}} instead (numbered, fresh)"
                ),
                None => format!(
                    "{name} is indexed: use mcp__indexio__file_outline {{repo:\"{repo}\", path:\"{rel}\"}} then read_span for the lines you need (numbered, fresh)"
                ),
            })
        }
        Lookup::Ranges(abs, ranges) => {
            if !abs.is_file() {
                return None;
            }
            let (repo, rel) = locate(&repos, &abs)?;
            let rel = super::indexed_rel(&engine, &repo, &rel, &abs)?;
            let last = tail_range(&engine, &repo, &rel, 1).map_or(TO_END, |(_, e)| e);
            let ranges: Vec<(u64, u64)> = ranges.into_iter().map(|(s, e)| (s, e.min(last))).filter(|(s, e)| s <= e).collect();
            if ranges.iter().map(|(s, e)| e - s + 1).sum::<u64>() <= super::SMALL_READ_LINES {
                return None;
            }
            freshen_later(data_dir, own.as_deref(), &repo, Some(&rel));
            let name = abs.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_else(|| rel.clone());
            let calls: Vec<String> = ranges
                .iter()
                .map(|(s, e)| format!("mcp__indexio__read_span {{repo:\"{repo}\", path:\"{rel}\", start:{s}, end:{e}}}"))
                .collect();
            Some(format!("{name} is indexed: use {} instead (numbered, fresh)", calls.join(" + ")))
        }
        Lookup::Search { paths, pattern, simple, case_sensitive, after } => {
            let mut targets: Vec<(String, String)> = Vec::new();
            for p in &paths {
                if p.to_string_lossy().contains(['*', '?']) {
                    // `src\*.rs`: the folder, if it is indexed
                    let dir = p.parent()?;
                    let (repo, rel) = locate(&repos, dir)?;
                    targets.push((repo, rel));
                    continue;
                }
                let (repo, rel) = locate(&repos, p)?;
                if p.is_file() && !contents_indexed(&engine, &repo, &rel) {
                    return None;
                }
                if !p.exists() {
                    return None;
                }
                targets.push((repo, rel));
            }
            if targets.is_empty() {
                return None;
            }
            let mut names: Vec<&str> = targets.iter().map(|(r, _)| r.as_str()).collect();
            names.sort();
            names.dedup();
            let repo_filter = if names.len() == 1 { format!(" repo:{}", names[0]) } else { String::new() };
            let mut rels: Vec<&str> = targets.iter().map(|(_, r)| r.as_str()).filter(|r| !r.is_empty()).collect();
            rels.sort();
            rels.dedup();
            let path_filter =
                if rels.is_empty() || rels.len() < targets.len() { String::new() } else { format!(" path:{}", rels.join("|")) };
            for repo in names {
                freshen_later(data_dir, own.as_deref(), repo, None);
            }
            // Select-String ignores case unless -CaseSensitive; code_grep is
            // smart-case, so only a pattern with capitals needs the flag
            let ci = !case_sensitive && pattern.chars().any(|c| c.is_uppercase());
            let call = if simple {
                format!("mcp__indexio__code_search {{query:\"{}\"{repo_filter}{path_filter}, lines:20}}", pattern.replace('"', "\\\""))
            } else {
                format!(
                    "mcp__indexio__code_grep {{pattern:\"{}{}{repo_filter}{path_filter}\"}}",
                    if ci { "(?i)" } else { "" },
                    pattern.replace('"', "\\\"")
                )
            };
            let ctx = match after {
                Some(n) if n > 0 => format!(", then read_span {{repo, path, start:line, end:line+{n}}} for context"),
                _ => String::new(),
            };
            Some(format!("Indexed files: use {call} instead (every matching line per file, ranked, current with your edits{ctx})"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn look(cmd: &str) -> Vec<String> {
        let mut cwd = PathBuf::from("C:/r");
        let mut vars = Vec::new();
        let mut out = Vec::new();
        for s in split_top(cmd, &[';', '\n']) {
            match statement(&s, &mut cwd, &mut vars) {
                Ok(Some(Lookup::Read(p, r, tail))) => out.push(format!("read {} {r:?}{}", p.display(), tail.map(|n| format!(" tail {n}")).unwrap_or_default())),
                Ok(Some(Lookup::Search { paths, pattern, simple, .. })) => {
                    out.push(format!("search {pattern} simple={simple} in {}", paths.len()))
                }
                Ok(Some(Lookup::Ranges(p, r))) => out.push(format!("ranges {} {r:?}", p.display())),
                Ok(None) => {}
                Err(()) => return vec!["WORK".into()],
            }
        }
        out
    }

    #[test]
    fn reads_and_ranges() {
        assert_eq!(look("Get-Content src\\a.rs"), vec!["read C:/r\\src\\a.rs None"]);
        assert_eq!(look("Get-Content -Path src/a.rs -TotalCount 40"), vec!["read C:/r\\src/a.rs Some((1, 40))"]);
        assert_eq!(look("gc a.rs | Select-Object -Skip 99 -First 20"), vec!["read C:/r\\a.rs Some((100, 119))"]);
        assert_eq!(look("(Get-Content a.rs)[10..19]"), vec!["read C:/r\\a.rs Some((11, 20))"]);
        assert_eq!(look("$p='C:\\x\\a.rs'; Get-Content $p -Raw"), vec!["read C:\\x\\a.rs None"]);
        assert_eq!(look("cd C:\\x; Get-Content a.rs"), vec!["read C:\\x\\a.rs None"]);
    }

    #[test]
    fn searches() {
        assert_eq!(look("Select-String -Path a.rs -Pattern 'fn main'"), vec!["search fn main simple=false in 1"]);
        assert_eq!(look("Select-String -Path a.rs,b.rs -Pattern 'x','y' -SimpleMatch"), vec!["search x|y simple=true in 2"]);
        assert_eq!(look("Get-ChildItem src -Recurse -Include *.rs | Select-String 'Foo' | Select-Object -First 30"), vec!["search Foo simple=false in 1"]);
        assert_eq!(
            look("Select-String -Path $p -Pattern 'a|b' | ForEach-Object { \"{0}: {1}\" -f $_.LineNumber, $_.Line }"),
            vec!["WORK"] // $p unknown
        );
    }

    fn var(cmd: &str) -> String {
        match var_read(cmd, Path::new("C:/r")) {
            Some(Lookup::Read(p, r, _)) => format!("read {} {r:?}", p.display()),
            Some(Lookup::Ranges(p, r)) => format!("ranges {} {r:?}", p.display()),
            Some(Lookup::Search { .. }) => "search".into(),
            None => "WORK".into(),
        }
    }

    #[test]
    fn reads_through_a_variable() {
        assert_eq!(var("$c = Get-Content src\\a.rs; $c[8446..8496]; '=== tests ==='; $c[11160..11300]"), "ranges C:/r\\src\\a.rs [(8447, 8497), (11161, 11301)]");
        assert_eq!(
            var("$lines = Get-Content a.rs; for ($k=390; $k -le 680; $k++) { \"{0,4} {1}\" -f $k, $lines[$k-1] }"),
            "ranges C:/r\\a.rs [(390, 680)]"
        );
        assert_eq!(
            var("$lines = Get-Content a.rs; for ($k=1; $k -le $lines.Count; $k++) { \"{0,4} {1}\" -f $k, $lines[$k-1] }"),
            "read C:/r\\a.rs None"
        );
        assert_eq!(
            var("$f='crates/x.rs'; $lines = Get-Content $f; foreach ($r in @(@(96,110),@(255,300))) { \"-- $($r[0])\"; for ($k=$r[0]; $k -le $r[1]; $k++) { \"{0,4} {1}\" -f $k, $lines[$k-1] } }"),
            "ranges C:/r\\crates/x.rs [(96, 110), (255, 300)]"
        );
        assert_eq!(var("$l=Get-Content a.tsx; 185..205 | % { \"{0}: {1}\" -f ($_+1), $l[$_] }"), "ranges C:/r\\a.tsx [(186, 206)]");
        assert_eq!(var("$c = Get-Content a.md; \"TOTAL=$($c.Count)\"; $c | Select-Object -First 40"), "read C:/r\\a.md None");
        assert_eq!(
            var("$c = Get-Content ci.yml; for ($i=88; $i -lt 115 -and $i -lt $c.Count; $i++) { \"{0}: {1}\" -f ($i+1), $c[$i] }"),
            "ranges C:/r\\ci.yml [(89, 115)]"
        );
        assert_eq!(
            var("$lines = Get-Content m.rs; for ($k=60; $k -le $lines.Count; $k++) { \"{0,4} {1}\" -f $k, $lines[$k-1] }"),
            format!("ranges C:/r\\m.rs [(60, {TO_END})]")
        );
        assert_eq!(var("$c = Get-Content a.rs; \"TOTAL: $($c.Count)\"; $c[125..($c.Count-1)]"), format!("ranges C:/r\\a.rs [(126, {TO_END})]"));
        // not reads
        for cmd in [
            "$c = Get-Content a.rs; $c.Count",
            "$c = Get-Content a.rs -Raw; $c.Replace('a','b') | Set-Content a.rs",
            "$log = Get-Content run.log; for ($i=0; $i -lt $log.Count; $i++) { if ($log[$i] -match 'x') { $i } }",
            "$j = Get-Content -Raw x.json | ConvertFrom-Json; $j.result",
            "cargo build; $c = Get-Content a.rs; $c[1..5]",
            "$c = Get-Content a.rs; $d = $c | Where-Object { $_ }; $d",
        ] {
            assert_eq!(var(cmd), "WORK", "{cmd}");
        }
    }

    #[test]
    fn work_is_left_alone() {
        for cmd in [
            "cargo build 2>&1 | Select-String -Pattern 'error'",
            "Get-Content a.rs | Measure-Object -Line",
            "$c = Get-Content a.rs -Raw; $c.Replace('a','b') | Set-Content a.rs",
            "if (Test-Path a) { Get-Content a }",
            "Get-Content run.log -Wait",
            "Select-String -Path a.rs -Pattern x -List",
            "git status; Get-Content a.rs",
            "Get-Content src\\*.rs",
        ] {
            assert_eq!(look(cmd), vec!["WORK"], "{cmd}");
        }
    }
}
