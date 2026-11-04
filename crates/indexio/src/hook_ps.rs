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

use super::{locate, serve_files, serve_search, Ctx, GrepArgs, Serve, Want};

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

/// `Select-Object -First/-Last/-Skip/-Index` and nothing else.
fn is_selector(w: &[String]) -> bool {
    w.first().is_some_and(|c| cmd_is(c, &["Select-Object", "select"])) && {
        let a = parse_args(&w[1..], &["First", "Last", "Skip", "Index"]);
        a.switches.is_empty() && a.positional.is_empty() && !a.named.is_empty()
    }
}

/// How a lookup statement's later stages relate to its answer:
/// `Some(None)` = folded into the lookup (or there are none),
/// `Some(Some(text))` = stages to keep after the answer (`| Select-Object
/// -Last 6` on a search), `None` = they reshape it (`ForEach-Object`,
/// `Where-Object`, a filtered folder listing): the shell's.
fn stmt_tail(stmt: &str) -> Option<Option<String>> {
    if stmt.starts_with('(') {
        return Some(None);
    }
    let stages = split_top(stmt, &['|']);
    let words: Vec<Vec<String>> = stages.iter().map(|s| ps_words(s)).collect();
    let head = words.first()?.first()?;
    let search_at = if cmd_is(head, SELECT_STRING) {
        Some(0)
    } else if words.get(1).and_then(|w| w.first()).is_some_and(|c| cmd_is(c, SELECT_STRING)) {
        if cmd_is(head, GET_CHILD) {
            let a = parse_args(&words[0][1..], &["Path", "LiteralPath", "Include", "Filter", "Exclude", "Depth"]);
            if a.get(&["Include", "Filter", "Exclude", "Depth"]).is_some() {
                return None;
            }
        }
        Some(1)
    } else {
        None
    };
    match search_at {
        Some(k) => {
            let first_last = |w: &Vec<String>| {
                is_selector(w) && parse_args(&w[1..], &["First", "Last", "Skip", "Index"]).get(&["Skip", "Index"]).is_none()
            };
            if !words[k + 1..].iter().all(first_last) {
                return None;
            }
            // the answer adds a `repo:path` line per file and a count line:
            // `-First/-Last n` becomes 2n+1, which never drops a match the
            // typed cut would show
            let mut after = Vec::new();
            for w in &words[k + 1..] {
                let a = parse_args(&w[1..], &["First", "Last"]);
                let mut stage = "Select-Object".to_string();
                for p in ["First", "Last"] {
                    if let Some(v) = a.get(&[p]) {
                        let n: u64 = v.parse().ok()?;
                        stage.push_str(&format!(" -{p} {}", 2 * n + 1));
                    }
                }
                after.push(stage);
            }
            Some((!after.is_empty()).then(|| after.join(" | ")))
        }
        None => words[1..].iter().all(|w| is_selector(w)).then_some(None),
    }
}

/// The index's answer to one lookup, or `None` (the shell's).
fn serve_lookup(ctx: &mut Ctx, l: Lookup) -> Option<Serve> {
    match l {
        Lookup::Read(abs, range, tail) => {
            let want = match (range, tail) {
                (Some((s, e)), _) => Want::Range(s, e),
                (None, Some(n)) => Want::Tail(n),
                _ => Want::Whole,
            };
            serve_files(ctx, &[abs], &[want])
        }
        Lookup::Ranges(abs, ranges) => {
            let wants: Vec<Want> = ranges.into_iter().map(|(s, e)| Want::Range(s, e)).collect();
            serve_files(ctx, &[abs], &wants)
        }
        Lookup::Search { paths, pattern, simple, case_sensitive, after } => {
            // context lines, and a wildcard (its extension filter), are the shell's
            if after.is_some_and(|n| n > 0) || paths.iter().any(|p| p.to_string_lossy().contains(['*', '?'])) {
                return None;
            }
            // Select-String: .NET regex, case-insensitive unless -CaseSensitive
            let g = GrepArgs { pattern, ere: true, fixed: simple, icase: !case_sensitive, ..Default::default() };
            serve_search(ctx, &paths, &g, None)
        }
    }
}

/// `s` in single quotes for PowerShell.
fn ps_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// The command to run instead of `command` when the index answers a read
/// in it: each answered statement becomes an `indexio tool`
/// call, with the statement itself as the fallback. `None` = as typed.
pub fn rewrite(data_dir: &Path, cwd: &Path, command: &str, exe: &Path) -> Option<String> {
    if command.contains("INDEXIO_RAW") || command.contains("# raw") {
        return None;
    }
    let mut ctx = Ctx::new(data_dir);
    if ctx.repos.is_empty() {
        return None;
    }
    let current = locate(&ctx.repos, cwd).map(|(r, _)| r);
    let exe = ps_quote(&exe.to_string_lossy().replace('\\', "/"));
    let dd = ps_quote(&data_dir.to_string_lossy().replace('\\', "/"));
    // a missing binary leaves the code at 1, so the fallback runs
    let call = |s: &Serve, keep: Option<&str>| {
        let (name, b64) = s.payload(current.as_deref());
        let keep = keep.map(|k| format!(" | {k}")).unwrap_or_default();
        format!("$global:LASTEXITCODE = 1; & {exe} --data-dir {dd} tool {name} {b64}{keep}")
    };
    if let Some(l) = var_read(command, cwd) {
        let s = serve_lookup(&mut ctx, l)?;
        return Some(format!("{}; if ($LASTEXITCODE -ne 0) {{ {command} }}", call(&s, None)));
    }
    let mut here = cwd.to_path_buf();
    let mut vars: Vec<(String, String)> = Vec::new();
    let mut out = Vec::new();
    let mut any = false;
    for stmt in split_top(command, &[';', '\n', '\r']) {
        match statement(&stmt, &mut here, &mut vars) {
            Err(()) => return None, // the shell is needed anyway
            Ok(None) => out.push(stmt),
            Ok(Some(l)) => {
                let served = stmt_tail(&stmt).and_then(|keep| Some((serve_lookup(&mut ctx, l)?, keep)));
                match served {
                    Some((s, keep)) => {
                        any = true;
                        out.push(format!("{}; if ($LASTEXITCODE -ne 0) {{ {stmt} }}", call(&s, keep.as_deref())));
                    }
                    None => out.push(stmt),
                }
            }
        }
    }
    any.then(|| out.join("\n"))
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
