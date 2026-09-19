//! What the harness's own tools would have put in the model's context for
//! the same call (SPEC-P10 §22), and the aggregate of the usage log that
//! `index_stats` and `indexio usage` report.
//!
//! Every successful tool call is logged with the bytes it returned and,
//! where the equivalent built-in tool's output can be sized without extra
//! work, the bytes that output would have had. The baselines reproduce the
//! harness tools as they render for the model: `Grep` prints every match
//! as `path:line:text` (first [`BUILTIN_GREP_LINES`] rows), `Read` prints
//! the whole file as `N<TAB>line` (first [`BUILTIN_READ_LINES`]), `Glob`
//! one absolute path per line. They are deliberately conservative: a grep
//! is sized from the rows the index returned (never more than it found), a
//! whole-file read is charged only the first time a session opens a file
//! (the harness would have had it in context after that), the outline that
//! precedes a span is pure cost, and tools without a mechanical equivalent
//! (hybrid search, recall, impact) get no baseline and count as unchanged.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

/// Rows a built-in `Grep` shows before truncating (its default head limit).
pub const BUILTIN_GREP_LINES: usize = 250;
/// Lines a built-in `Read` shows before truncating.
pub const BUILTIN_READ_LINES: usize = 2000;

fn digits(n: u32) -> u64 {
    if n == 0 {
        1
    } else {
        u64::from(n.ilog10() + 1)
    }
}

/// Bytes of a built-in `Grep` over `rows` = (repo-relative path, line, text):
/// `path:line:text\n` for the first [`BUILTIN_GREP_LINES`] rows.
pub fn grep_bytes<'a>(rows: impl Iterator<Item = (&'a str, u32, &'a str)>) -> u64 {
    rows.take(BUILTIN_GREP_LINES)
        .map(|(p, l, t)| p.len() as u64 + 1 + digits(l) + 1 + t.len() as u64 + 1)
        .sum()
}

/// Bytes of a built-in `Read` of `content`: `N<TAB>line\n` for the first
/// [`BUILTIN_READ_LINES`] lines.
pub fn read_bytes(content: &[u8]) -> u64 {
    let body = content.strip_suffix(b"\n").unwrap_or(content);
    if body.is_empty() {
        return 0;
    }
    body.split(|&b| b == b'\n')
        .take(BUILTIN_READ_LINES)
        .enumerate()
        .map(|(i, l)| digits(i as u32 + 1) + 1 + l.len() as u64 + 1)
        .sum()
}

/// Bytes of a built-in `Glob`: one absolute path per line, `root_len` bytes
/// of checkout root plus a separator before each repo-relative path.
pub fn glob_bytes<'a>(paths: impl Iterator<Item = &'a str>, root_len: usize) -> u64 {
    paths.map(|p| root_len as u64 + 1 + p.len() as u64 + 1).sum()
}

#[derive(Default, Clone)]
pub struct Agg {
    pub calls: u64,
    pub errors: u64,
    /// Bytes the tools returned.
    pub bytes: u64,
    /// Calls with a built-in baseline, and what those calls returned.
    pub compared: u64,
    pub compared_bytes: u64,
    /// What the built-in tools would have returned for the compared calls.
    pub builtin: u64,
    pub ms: Vec<u64>,
}

impl Agg {
    fn add(&mut self, ok: bool, bytes: u64, builtin: Option<u64>, ms: u64) {
        self.calls += 1;
        self.errors += u64::from(!ok);
        self.bytes += bytes;
        if let Some(b) = builtin {
            self.compared += 1;
            self.compared_bytes += bytes;
            self.builtin += b;
        }
        self.ms.push(ms);
    }
    pub fn avg_ms(&self) -> u64 {
        if self.calls == 0 {
            0
        } else {
            self.ms.iter().sum::<u64>() / self.calls
        }
    }
    pub fn p95_ms(&self) -> u64 {
        if self.ms.is_empty() {
            return 0;
        }
        let mut v = self.ms.clone();
        v.sort_unstable();
        v[(v.len() * 95 / 100).min(v.len() - 1)]
    }
    /// Percent change of the compared calls against the built-in tools
    /// (negative = fewer bytes), `None` without compared calls.
    pub fn change_pct(&self) -> Option<i64> {
        (self.builtin > 0).then(|| {
            ((self.compared_bytes as f64 - self.builtin as f64) / self.builtin as f64 * 100.0).round() as i64
        })
    }
}

pub struct Report {
    pub days: u64,
    pub sessions: usize,
    pub by_tool: BTreeMap<String, Agg>,
    pub by_repo: BTreeMap<String, Agg>,
    pub total: Agg,
}

/// Aggregate `<data_dir>/usage/*.jsonl` over the last `days` days.
pub fn aggregate(data_dir: &Path, days: u64) -> Report {
    let dir = data_dir.join("usage");
    let since = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
        .saturating_sub(days * 86_400);
    let mut r = Report { days, sessions: 0, by_tool: BTreeMap::new(), by_repo: BTreeMap::new(), total: Agg::default() };
    let mut sessions: BTreeSet<u64> = BTreeSet::new();
    let Ok(rd) = std::fs::read_dir(&dir) else { return r };
    for e in rd.flatten() {
        let Ok(text) = std::fs::read_to_string(e.path()) else { continue };
        for line in text.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
            let ts = v["ts"].as_u64().unwrap_or(0);
            if ts < since {
                continue;
            }
            let tool = v["tool"].as_str().unwrap_or("?").to_string();
            let repo = v["repo"].as_str().unwrap_or("-").to_string();
            let ms = v["ms"].as_u64().unwrap_or(0);
            let bytes = v["bytes"].as_u64().unwrap_or(0);
            let ok = v["ok"].as_bool().unwrap_or(true);
            let builtin = v["builtin"].as_u64();
            sessions.insert(v["pid"].as_u64().unwrap_or(0));
            r.by_tool.entry(tool).or_default().add(ok, bytes, builtin, ms);
            r.by_repo.entry(repo).or_default().add(ok, bytes, builtin, ms);
            r.total.add(ok, bytes, builtin, ms);
        }
    }
    r.sessions = sessions.len();
    r
}

/// Tokens from bytes, the estimate used throughout (about 4 bytes per
/// token of code and paths under cl100k).
pub fn est_tokens(bytes: u64) -> u64 {
    bytes / 4
}

/// Estimated tokens as a short figure: `812`, `4.1k`, `55k`.
pub fn ktok(bytes: u64) -> String {
    let t = est_tokens(bytes);
    if t >= 10_000 {
        format!("{}k", t / 1000)
    } else if t >= 1000 {
        format!("{:.1}k", t as f64 / 1000.0)
    } else {
        t.to_string()
    }
}

/// The one- or two-line summary `index_stats` carries: calls, tokens
/// returned and the change against the built-in tools.
pub fn render_summary(r: &Report) -> String {
    let mut out = format!(
        "usage, last {} day{}: {} call{} from {} server process{}, {} tokens returned",
        r.days,
        if r.days == 1 { "" } else { "s" },
        r.total.calls,
        if r.total.calls == 1 { "" } else { "s" },
        r.sessions,
        if r.sessions == 1 { "" } else { "es" },
        ktok(r.total.bytes),
    );
    if let Some(pct) = r.total.change_pct() {
        let _ = write!(
            out,
            "\nvs the built-in tools on the {} call{} with a mechanical equivalent: {} tokens instead of {} ({pct:+}%)",
            r.total.compared,
            if r.total.compared == 1 { "" } else { "s" },
            ktok(r.total.compared_bytes),
            ktok(r.total.builtin),
        );
    }
    out
}

/// One table row: name, calls, tokens, built-in tokens, change, avg ms.
/// A tool compared but with nothing on the other side (`file_outline`)
/// shows `0` and `cost`; an uncompared one `-`.
pub fn render_row(name: &str, a: &Agg) -> String {
    let (b, c) = match (a.compared, a.change_pct()) {
        (_, Some(pct)) => (ktok(a.builtin), format!("{pct:+}%")),
        (n, None) if n > 0 => ("0".to_string(), "cost".to_string()),
        _ => ("-".to_string(), "-".to_string()),
    };
    format!("{:<18}{:>6}{:>9}{:>10}{:>8}{:>8}", name, a.calls, ktok(a.bytes), b, c, a.avg_ms())
}

pub const ROW_HEADER: (&str, &str, &str, &str, &str, &str) = ("tool", "calls", "tokens", "built-in", "change", "avg ms");

fn header(first: &str) -> String {
    let h = ROW_HEADER;
    format!("{:<18}{:>6}{:>9}{:>10}{:>8}{:>8}", first, h.1, h.2, h.3, h.4, h.5)
}

/// The report as text: the summary, then one row per tool and per repo.
pub fn render(r: &Report) -> String {
    let mut out = render_summary(r);
    if r.by_tool.is_empty() {
        return out;
    }
    let _ = write!(out, "\n{}", header("tool"));
    for (name, a) in &r.by_tool {
        let _ = write!(out, "\n{}", render_row(name, a));
    }
    let _ = write!(out, "\n{}", header("repo"));
    for (name, a) in &r.by_repo {
        let _ = write!(out, "\n{}", render_row(name, a));
    }
    out
}

/// Machine-readable form of the report (`indexio usage --json`).
pub fn to_json(r: &Report) -> serde_json::Value {
    let row = |name: &str, a: &Agg| {
        serde_json::json!({
            "name": name, "calls": a.calls, "errors": a.errors, "bytes": a.bytes,
            "est_tokens": est_tokens(a.bytes), "avg_ms": a.avg_ms(), "p95_ms": a.p95_ms(),
            "compared_calls": a.compared, "compared_bytes": a.compared_bytes,
            "builtin_bytes": a.builtin, "change_pct": a.change_pct(),
        })
    };
    serde_json::json!({
        "days": r.days, "sessions": r.sessions,
        "tools": r.by_tool.iter().map(|(k, a)| row(k, a)).collect::<Vec<_>>(),
        "repos": r.by_repo.iter().map(|(k, a)| row(k, a)).collect::<Vec<_>>(),
        "total": row("total", &r.total),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baselines_reproduce_the_harness_renderings() {
        // Grep: `path:line:text\n`
        assert_eq!(grep_bytes([("a.rs", 12, "fn x()")].into_iter()), 4 + 1 + 2 + 1 + 6 + 1);
        // capped at the harness's head limit
        let many: Vec<(&str, u32, &str)> = (0..1000).map(|_| ("p", 1, "t")).collect();
        assert_eq!(grep_bytes(many.iter().copied()), BUILTIN_GREP_LINES as u64 * 6);
        // Read: `N<TAB>line\n`, a trailing newline is not a phantom line
        assert_eq!(read_bytes(b"ab\ncd\n"), 2 * (1 + 1 + 2 + 1));
        assert_eq!(read_bytes(b""), 0);
        let big = "x\n".repeat(3000);
        assert_eq!(read_bytes(big.as_bytes()), (1..=BUILTIN_READ_LINES as u32).map(|n| digits(n) + 3).sum::<u64>());
        // Glob: root + separator + path + newline
        assert_eq!(glob_bytes(["src/a.rs", "b.rs"].into_iter(), 10), (11 + 8 + 1) + (11 + 4 + 1));
    }

    #[test]
    fn aggregate_compares_only_calls_with_a_baseline() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("usage");
        std::fs::create_dir_all(&dir).unwrap();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        std::fs::write(
            dir.join("today.jsonl"),
            format!(
                "{}\n{}\n{}\n",
                serde_json::json!({"ts": now, "pid": 1, "repo": "r", "tool": "read_span", "ms": 3, "bytes": 400, "builtin": 4000, "ok": true}),
                serde_json::json!({"ts": now, "pid": 1, "repo": "r", "tool": "recall", "ms": 30, "bytes": 2000, "ok": true}),
                serde_json::json!({"ts": now - 10 * 86_400, "pid": 2, "repo": "r", "tool": "read_span", "ms": 3, "bytes": 1, "builtin": 1, "ok": true}),
            ),
        )
        .unwrap();
        let r = aggregate(tmp.path(), 7);
        assert_eq!(r.total.calls, 2);
        assert_eq!(r.sessions, 1);
        assert_eq!(r.total.compared, 1);
        assert_eq!(r.total.compared_bytes, 400);
        assert_eq!(r.total.builtin, 4000);
        assert_eq!(r.total.change_pct(), Some(-90));
        assert_eq!(r.by_tool["recall"].change_pct(), None);
        let text = render(&r);
        assert!(text.contains("2 calls from 1 server process"), "{text}");
        assert!(text.contains("(-90%)"), "{text}");
        assert!(text.contains("recall"), "{text}");
        assert!(render_summary(&r).lines().count() == 2, "{text}");
        let mut outline = Agg::default();
        outline.add(true, 800, Some(0), 1);
        assert!(render_row("file_outline", &outline).contains("cost"));
        let j = to_json(&r);
        assert_eq!(j["total"]["change_pct"], -90);
        assert_eq!(j["tools"][0]["name"], "read_span");
    }
}
