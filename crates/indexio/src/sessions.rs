//! Session transcripts as an indexed source (SPEC-P9 §16, "recall").
//!
//! Claude Code writes every session to
//! `<claude dir>/projects/<project slug>/<session id>.jsonl`, one JSON
//! object per line. This module turns each transcript into compact
//! Markdown under `<data_dir>/sessions/<slug>/<session id>[.partNN].md`:
//! user and assistant text, tool calls with their arguments, the first
//! lines of each tool result, and compaction summaries — the parts a later
//! turn (or a later session) may need to recall. Thinking blocks,
//! harness reminders and bookkeeping records are dropped. The folder is
//! registered as the plain-folder source `sessions`, so the ordinary
//! refresh machinery indexes it (lexical + semantic) and the `recall`
//! MCP tool searches it.
//!
//! Everything stays local under the data dir; it is a second copy of what
//! the sessions already wrote to disk.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Output file size cap per part (well under the text-file index cap).
const PART_BYTES: usize = 200 * 1024;
/// Caps per entry.
const TEXT_CAP: usize = 4000;
const TOOL_INPUT_CAP: usize = 300;
const TOOL_RESULT_CAP: usize = 400;
const TOOL_RESULT_LINES: usize = 6;
/// Thinking blocks (SPEC-P10): the harness does not always persist the
/// assistant's visible text, but a short narration of each turn lands in
/// its thinking block — the decisions `recall` is asked about.
const THINK_CAP: usize = 400;

#[derive(Default, Debug)]
pub struct ImportReport {
    pub sessions_seen: usize,
    pub sessions_imported: usize,
    pub parts_written: usize,
    pub bytes_written: u64,
    /// Transcripts idle longer than the retention window whose rendered
    /// parts were removed (SPEC-P10 §31).
    pub sessions_rolled_off: usize,
}

#[derive(Default, Serialize, Deserialize)]
struct State {
    /// transcript path -> (len, mtime ms) at last import
    files: HashMap<String, (u64, u64)>,
}

/// Claude Code's project folder name for a working directory: every
/// non-alphanumeric character becomes `-`.
pub fn project_slug(cwd: &Path) -> String {
    cwd.to_string_lossy()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

/// `~/.claude`, honouring `CLAUDE_CONFIG_DIR`.
pub fn default_claude_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("CLAUDE_CONFIG_DIR") {
        return Some(PathBuf::from(d));
    }
    ["HOME", "USERPROFILE"]
        .iter()
        .filter_map(std::env::var_os)
        .find(|h| !h.is_empty())
        .map(|h| PathBuf::from(h).join(".claude"))
}

/// Name of the plain-folder source the transcripts are indexed under.
pub const REPO: &str = "sessions";

/// Register `<data_dir>/sessions` as a plain-folder source (once) and
/// delta re-index it by content. Lexical only; the MCP server embeds the
/// changed paths itself, `indexio sync` embeds it like any other repo.
pub fn index_sessions(
    data_dir: &Path,
    cas: &indexio_ingest::Cas,
    cache: Option<&mut indexio_ingest::WorktreeCache>,
) -> anyhow::Result<indexio_ingest::IndexReport> {
    let out = data_dir.join("sessions");
    fs::create_dir_all(&out)?;
    if indexio_ingest::repo_state(data_dir, REPO).is_err() {
        let src = indexio_ingest::sources::parse_source(&out.to_string_lossy())?;
        indexio_ingest::sources::add_source(data_dir, src)?;
        return indexio_ingest::index_repo(&out, REPO, data_dir, cas);
    }
    // a plain folder: with the server's cache, unchanged parts are stat'ed,
    // not re-read (SPEC-P10)
    indexio_ingest::reindex_worktree_cached(REPO, data_dir, cas, cache)
}

/// Remove every rendered part of one session.
fn remove_parts(dir: &Path, session_id: &str) {
    if let Ok(old) = fs::read_dir(dir) {
        for f in old.flatten() {
            let n = f.file_name().to_string_lossy().into_owned();
            if n.starts_with(session_id) && n.ends_with(".md") {
                let _ = fs::remove_file(f.path());
            }
        }
    }
}

fn mtime_ms(m: &fs::Metadata) -> u64 {
    m.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Import every transcript that changed since the last run. `only_slug`
/// restricts the pass to one project folder (the running session's).
pub fn import(claude_dir: &Path, out_dir: &Path, only_slug: Option<&str>) -> anyhow::Result<ImportReport> {
    let projects = claude_dir.join("projects");
    let mut report = ImportReport::default();
    if !projects.is_dir() {
        return Ok(report);
    }
    fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;

    for proj in fs::read_dir(&projects)?.flatten() {
        let slug = proj.file_name().to_string_lossy().into_owned();
        if only_slug.is_some_and(|s| s != slug) {
            continue;
        }
        // one state file per project: servers of different sessions import
        // concurrently and must not overwrite each other's stamps
        let state_path = out_dir.join(&slug).join(".import-state.json");
        let mut state: State = fs::read(&state_path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        let Ok(rd) = fs::read_dir(proj.path()) else { continue };
        // transcripts rendered before cards existed: render each once more
        // (a marker, not a state field — older binaries rewrite the state)
        let cards_marker = out_dir.join(&slug).join(CARDS_MARKER);
        let backfill = !cards_marker.exists();
        let retain = crate::runs::retain_days();
        let cutoff_ms = (retain > 0).then(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0)
                .saturating_sub(retain * 86_400_000)
        });
        for entry in rd.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            report.sessions_seen += 1;
            let Ok(meta) = entry.metadata() else { continue };
            let stamp = (meta.len(), mtime_ms(&meta));
            let key = path.to_string_lossy().into_owned();
            let session_id = path.file_stem().and_then(|s| s.to_str()).unwrap_or("session").to_string();
            // SPEC-P10 §31: a transcript idle longer than the retention
            // window rolls off — its rendered parts are removed and it is
            // not rendered again until it changes
            if cutoff_ms.is_some_and(|c| stamp.1 < c) {
                if state.files.remove(&key).is_some() {
                    remove_parts(&out_dir.join(&slug), &session_id);
                    report.sessions_rolled_off += 1;
                }
                continue;
            }
            if !backfill && state.files.get(&key) == Some(&stamp) {
                continue;
            }
            match render(&path, &slug, &session_id) {
                Ok((parts, card)) => {
                    let dir = out_dir.join(&slug);
                    fs::create_dir_all(&dir)?;
                    // drop stale parts of an earlier, longer import
                    remove_parts(&dir, &session_id);
                    if let Some(card) = card {
                        report.bytes_written += card.len() as u64;
                        fs::write(dir.join(format!("{session_id}{CARD_SUFFIX}")), card)?;
                    }
                    let n = parts.len();
                    for (i, text) in parts.into_iter().enumerate() {
                        let name = if n == 1 {
                            format!("{session_id}.md")
                        } else {
                            format!("{session_id}.part{:02}.md", i + 1)
                        };
                        report.bytes_written += text.len() as u64;
                        fs::write(dir.join(name), text)?;
                        report.parts_written += 1;
                    }
                    report.sessions_imported += 1;
                    state.files.insert(key, stamp);
                }
                Err(e) => tracing::warn!(path = %path.display(), error = %e, "transcript skipped"),
            }
        }
        if let Some(parent) = state_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&state_path, serde_json::to_vec(&state)?)?;
        if backfill {
            fs::write(&cards_marker, b"")?;
        }
    }
    Ok(report)
}

/// File name suffix of a session card, next to its parts.
pub const CARD_SUFFIX: &str = ".card.md";
/// Per-project marker: every transcript there has been rendered with cards.
const CARDS_MARKER: &str = ".cards-v1";

/// Caps of a session card (the compressed form of a whole session).
const CARD_ASKS: usize = 8;
const CARD_ASK_CAP: usize = 220;
const CARD_FILES: usize = 25;
const CARD_COMMITS: usize = 12;
const CARD_END_CAP: usize = 900;

/// What a session did, gathered while rendering it: the few hundred tokens a
/// later session needs to pick up where it left off (`recall {brief:true}`,
/// `hook session-start`), instead of the transcript's hundreds of kB.
#[derive(Default)]
struct Card {
    first_ts: String,
    last_ts: String,
    turns: usize,
    tool_calls: usize,
    asks: Vec<String>,
    /// edited file -> edits, in first-edit order
    files: Vec<(String, usize)>,
    commits: Vec<String>,
    last_answer: String,
}

impl Card {
    fn note_ask(&mut self, t: &str) {
        // harness turns (background-task wakeups, local-command echoes) are
        // not requests; a pasted table or log starts at its first worded line
        let t = t.trim();
        if t.is_empty() || ["[Request interrupted", "Caveat:", "<task-notification>", "<local-command"].iter().any(|p| t.starts_with(p)) {
            return;
        }
        // a bare slash command (`/clear`, `/model`) asks nothing
        // (rendered "command: /clear  clear": the name, then its message)
        if let Some(c) = t.strip_prefix("command: /") {
            let w: Vec<&str> = c.split_whitespace().collect();
            if w.len() == 1 || (w.len() == 2 && w[0] == w[1]) {
                return;
            }
        }
        self.turns += 1;
        let tabular = |l: &str| l.trim_start().starts_with('|') || l.chars().any(|c| ('\u{2500}'..='\u{257f}').contains(&c));
        let worded = t.lines().find(|l| !tabular(l) && l.chars().filter(|c| c.is_alphabetic()).count() >= 3).unwrap_or(t);
        let t = &t[t.find(worded).unwrap_or(0)..];
        let a = cap(t, CARD_ASK_CAP * 2);
        // a /loop re-fires the same prompt every tick
        if !self.asks.contains(&a) {
            self.asks.push(a);
        }
    }

    fn tool_use(&mut self, name: &str, input: &Value) {
        self.tool_calls += 1;
        match name {
            "Edit" | "Write" | "MultiEdit" | "NotebookEdit" => {
                let p = input["file_path"].as_str().or_else(|| input["notebook_path"].as_str());
                if let Some(p) = p {
                    match self.files.iter_mut().find(|(f, _)| f == p) {
                        Some((_, n)) => *n += 1,
                        None => self.files.push((p.to_string(), 1)),
                    }
                }
            }
            "Bash" | "PowerShell" => {
                if let Some(m) = input["command"].as_str().and_then(commit_message) {
                    if !self.commits.contains(&m) {
                        self.commits.push(m);
                    }
                }
            }
            _ => {}
        }
    }

    fn render(&self, session_id: &str, title: &str, cwd: &str, slug: &str) -> Option<String> {
        if self.asks.is_empty() {
            return None;
        }
        let mut s = format!(
            "# card {session_id}{}\nproject: {} ({slug})\nwhen: {} → {}; {} turns, {} tool calls\n",
            if title.is_empty() { String::new() } else { format!(" — {title}") },
            if cwd.is_empty() { "?" } else { cwd },
            self.first_ts,
            self.last_ts,
            self.turns,
            self.tool_calls,
        );
        s.push_str("\n## asked\n");
        // the first requests set the task, the last ones where it ended
        let n = self.asks.len();
        let half = CARD_ASKS / 2;
        for (i, a) in self.asks.iter().enumerate() {
            if n > CARD_ASKS && i == half {
                s.push_str(&format!("- … {} more\n", n - CARD_ASKS));
            }
            if n <= CARD_ASKS || i < half || i >= n - half {
                s.push_str(&format!("- {}\n", cap(&a.replace('\n', " "), CARD_ASK_CAP)));
            }
        }
        // project files, most-edited first; scratch files elsewhere are counted
        let root = cwd.replace('\\', "/").trim_end_matches('/').to_lowercase();
        let mut inside: Vec<(String, usize)> = Vec::new();
        let mut outside = 0;
        for (f, k) in &self.files {
            let f = f.replace('\\', "/");
            if !root.is_empty() && f.to_lowercase().starts_with(&format!("{root}/")) {
                inside.push((f[root.len() + 1..].to_string(), *k));
            } else if root.is_empty() {
                inside.push((f, *k));
            } else {
                outside += 1;
            }
        }
        inside.sort_by(|a, b| b.1.cmp(&a.1));
        if !inside.is_empty() || outside > 0 {
            s.push_str("\n## changed\n");
            for (f, k) in inside.iter().take(CARD_FILES) {
                s.push_str(&format!("- {f}{}\n", if *k > 1 { format!(" ×{k}") } else { String::new() }));
            }
            if inside.len() > CARD_FILES {
                s.push_str(&format!("- … {} more\n", inside.len() - CARD_FILES));
            }
            if outside > 0 {
                s.push_str(&format!("- ({outside} file{} outside the project)\n", if outside == 1 { "" } else { "s" }));
            }
        }
        if !self.commits.is_empty() {
            s.push_str("\n## commits\n");
            let skip = self.commits.len().saturating_sub(CARD_COMMITS);
            for c in &self.commits[skip..] {
                s.push_str(&format!("- {c}\n"));
            }
        }
        if !self.last_answer.is_empty() {
            s.push_str(&format!("\n## ended with\n{}\n", cap(&self.last_answer, CARD_END_CAP)));
        }
        Some(crate::redact::redact(&s).into_owned())
    }
}

/// The subject line of a `git commit -m …` command, if it is one.
fn commit_message(cmd: &str) -> Option<String> {
    let i = cmd.find("git commit")?;
    let rest = &cmd[i..];
    let m = rest.find(" -m")? + 3;
    let rest = rest[m..].trim_start();
    // -m "$(cat <<'EOF' … EOF)" : the first line of the heredoc body
    let (body, quote) = if rest.starts_with("\"$(") || rest.starts_with("$(") || rest.starts_with("@'") {
        (rest.split_once('\n').map(|(_, b)| b).unwrap_or(""), None)
    } else {
        let q = rest.chars().next().filter(|c| *c == '"' || *c == '\'');
        (rest.trim_start_matches(['"', '\'']), q)
    };
    let mut line = body.lines().map(str::trim).find(|l| !l.is_empty())?;
    // a one-line message ends at its closing quote (then `&& git push …`)
    if let Some(q) = quote {
        if let Some(j) = line.find(q) {
            line = &line[..j];
        }
    }
    let line = line.trim_end_matches(['"', '\'']).trim();
    (!line.is_empty()).then(|| cap(line, 140))
}

/// Render one transcript into ≤ [`PART_BYTES`] Markdown parts, plus its
/// card (`None` when the session asked nothing).
fn render(path: &Path, slug: &str, session_id: &str) -> anyhow::Result<(Vec<String>, Option<String>)> {
    let r = render_full(path, slug, session_id)?;
    let card = r.card.render(session_id, &r.title, &r.cwd, slug);
    Ok((r.parts, card))
}

/// A rendered transcript: its parts and what its card is made of.
struct Rendered {
    parts: Vec<String>,
    card: Card,
    title: String,
    cwd: String,
}

fn render_full(path: &Path, slug: &str, session_id: &str) -> anyhow::Result<Rendered> {
    let file = fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    use std::io::BufRead as _;
    let mut title = String::new();
    let mut cwd = String::new();
    let mut entries: Vec<String> = Vec::new();
    let mut card = Card::default();
    for line in reader.lines() {
        let Ok(line) = line else { continue };
        let Ok(o) = serde_json::from_str::<Value>(&line) else { continue };
        match o["type"].as_str().unwrap_or("") {
            "ai-title" => {
                if let Some(t) = o["aiTitle"].as_str() {
                    title = t.to_string();
                }
            }
            "summary" => {
                if let Some(s) = o["summary"].as_str() {
                    entries.push(format!("## compaction summary\n{}\n", crate::redact::redact(&cap(s, TEXT_CAP))));
                }
            }
            "user" | "assistant" => {
                if cwd.is_empty() {
                    if let Some(c) = o["cwd"].as_str() {
                        cwd = c.to_string();
                    }
                }
                let side = if o["isSidechain"].as_bool().unwrap_or(false) { " (subagent)" } else { "" };
                let ts = o["timestamp"].as_str().map(short_ts).unwrap_or_default();
                let role = o["type"].as_str().unwrap_or("");
                if !ts.is_empty() {
                    if card.first_ts.is_empty() {
                        card.first_ts = ts.clone();
                    }
                    card.last_ts = ts.clone();
                }
                let main_thread = side.is_empty();
                let mut body = String::new();
                match &o["message"]["content"] {
                    Value::String(s) => {
                        let t = clean_user_text(s);
                        if !t.is_empty() {
                            body.push_str(&cap(&t, TEXT_CAP));
                            body.push('\n');
                            if main_thread && role == "user" {
                                card.note_ask(&t);
                            }
                        }
                    }
                    Value::Array(blocks) => {
                        for b in blocks {
                            match b["type"].as_str().unwrap_or("") {
                                "text" => {
                                    let t = clean_user_text(b["text"].as_str().unwrap_or(""));
                                    if !t.is_empty() {
                                        body.push_str(&cap(&t, TEXT_CAP));
                                        body.push('\n');
                                        if main_thread && role == "user" {
                                            card.note_ask(&t);
                                        } else if main_thread && role == "assistant" {
                                            card.last_answer = t;
                                        }
                                    }
                                }
                                "tool_use" => {
                                    let name = b["name"].as_str().unwrap_or("?");
                                    card.tool_use(name, &b["input"]);
                                    let input = b["input"].to_string();
                                    body.push_str(&format!("-> {name} {}\n", cap(&input, TOOL_INPUT_CAP)));
                                }
                                "tool_result" => {
                                    let text = match &b["content"] {
                                        Value::String(s) => s.clone(),
                                        Value::Array(parts) => parts
                                            .iter()
                                            .filter_map(|p| p["text"].as_str())
                                            .collect::<Vec<_>>()
                                            .join("\n"),
                                        _ => String::new(),
                                    };
                                    // noise lines a Windows shell adds to every result
                                    // (SPEC-P10 §21): they carry no recall value and
                                    // used to crowd out the result's real first lines
                                    let head: Vec<&str> = text.lines().filter(|l| !is_noise_line(l)).take(TOOL_RESULT_LINES).collect();
                                    let head = cap(&head.join("\n"), TOOL_RESULT_CAP);
                                    if !head.trim().is_empty() {
                                        body.push_str(&format!("<- {}\n", head.trim_end()));
                                    }
                                }
                                "thinking" => {
                                    let t = b["thinking"].as_str().unwrap_or("").trim();
                                    if !t.is_empty() {
                                        body.push_str(&format!("~ {}
", cap(t, THINK_CAP)));
                                    }
                                }
                                _ => {} // images, ...
                            }
                        }
                    }
                    _ => {}
                }
                if !body.trim().is_empty() {
                    // what the agent saw may include what it must not keep (§39)
                    let body = crate::redact::redact(&body);
                    entries.push(format!("## {ts} {role}{side}\n{body}"));
                }
            }
            _ => {}
        }
    }
    let header = format!(
        "# session {session_id}{}\nproject: {} ({slug})\n\n",
        if title.is_empty() { String::new() } else { format!(" — {title}") },
        if cwd.is_empty() { "?" } else { cwd.as_str() }
    );
    let mut parts: Vec<String> = Vec::new();
    let mut cur = header.clone();
    let mut part_no = 1;
    for e in entries {
        if cur.len() + e.len() > PART_BYTES && cur.len() > header.len() {
            parts.push(std::mem::take(&mut cur));
            part_no += 1;
            cur = header.replace("\nproject:", &format!(" (part {part_no})\nproject:"));
        }
        cur.push_str(&e);
        cur.push('\n');
    }
    if cur.len() > header.len() {
        parts.push(cur);
    }
    Ok(Rendered { parts, card, title, cwd })
}

/// Caps of the session-start context (it is paid on every start).
const START_FILES: usize = 12;
const START_COMMITS: usize = 5;
/// A previous session older than this is not worth pointing at.
const START_MAX_AGE_DAYS: u64 = 14;

/// Relative path of `f` under `cwd`, else `None`.
fn under(cwd: &str, f: &str) -> Option<String> {
    let root = cwd.replace('\\', "/").trim_end_matches('/').to_lowercase();
    let f = f.replace('\\', "/");
    (!root.is_empty() && f.to_lowercase().starts_with(&format!("{root}/"))).then(|| f[root.len() + 1..].to_string())
}

/// `hook session-start` (SessionStart): a few lines of past context, so a
/// session starts from pointers instead of re-exploring.
/// - `startup` / `clear`: the previous session in this project — title,
///   when, how much it changed, its last commit — and the `recall` call that
///   returns its card. ~50 tokens.
/// - `compact`: this session's working set (files changed, commits), which a
///   compaction summary tends to lose. ~150-300 tokens.
/// - `resume` (or anything else): nothing; the session has its history.
pub fn session_start_context(claude_dir: &Path, cwd: &Path, session_id: &str, transcript: Option<&Path>, source: &str) -> Option<String> {
    let slug = project_slug(cwd);
    match source {
        "compact" => {
            let path = transcript.map(Path::to_path_buf).unwrap_or_else(|| claude_dir.join("projects").join(&slug).join(format!("{session_id}.jsonl")));
            let r = render_full(&path, &slug, session_id).ok()?;
            let cwd_s = if r.cwd.is_empty() { cwd.to_string_lossy().into_owned() } else { r.cwd.clone() };
            let mut files: Vec<(String, usize)> = r.card.files.iter().filter_map(|(f, k)| under(&cwd_s, f).map(|p| (p, *k))).collect();
            if files.is_empty() && r.card.commits.is_empty() {
                return None;
            }
            files.sort_by(|a, b| b.1.cmp(&a.1));
            let mut s = String::from("indexio — this session before compaction:");
            if !files.is_empty() {
                let shown: Vec<String> = files.iter().take(START_FILES).map(|(f, k)| if *k > 1 { format!("{f} ×{k}") } else { f.clone() }).collect();
                s.push_str(&format!("\nfiles changed: {}", shown.join(", ")));
                if files.len() > START_FILES {
                    s.push_str(&format!(", +{} more", files.len() - START_FILES));
                }
            }
            if !r.card.commits.is_empty() {
                let skip = r.card.commits.len().saturating_sub(START_COMMITS);
                s.push_str(&format!("\ncommits: {}", r.card.commits[skip..].join(" | ")));
            }
            Some(crate::redact::redact(&s).into_owned())
        }
        "startup" | "clear" => {
            // the newest other transcript of this project that asked something
            let dir = claude_dir.join("projects").join(&slug);
            let now = std::time::SystemTime::now();
            let mut cands: Vec<(std::time::SystemTime, PathBuf)> = fs::read_dir(&dir)
                .ok()?
                .flatten()
                .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("jsonl"))
                .filter(|e| e.path().file_stem().and_then(|s| s.to_str()) != Some(session_id))
                .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
                .filter(|(t, _)| now.duration_since(*t).map(|d| d.as_secs() < START_MAX_AGE_DAYS * 86_400).unwrap_or(true))
                .collect();
            cands.sort_by(|a, b| b.0.cmp(&a.0));
            for (_, path) in cands.into_iter().take(4) {
                let id = path.file_stem()?.to_string_lossy().into_owned();
                let Ok(r) = render_full(&path, &slug, &id) else { continue };
                if r.card.asks.is_empty() {
                    continue;
                }
                let cwd_s = if r.cwd.is_empty() { cwd.to_string_lossy().into_owned() } else { r.cwd.clone() };
                let files = r.card.files.iter().filter(|(f, _)| under(&cwd_s, f).is_some()).count();
                let title = if r.title.is_empty() { cap(&r.card.asks[0].replace('\n', " "), 80) } else { r.title.clone() };
                let mut s = format!("indexio — previous session here: \"{title}\" ({} → {})", r.card.first_ts, r.card.last_ts);
                if files > 0 {
                    s.push_str(&format!(", {files} file{} changed", if files == 1 { "" } else { "s" }));
                }
                let n = r.card.commits.len();
                if n > 0 {
                    s.push_str(&format!(", {n} commit{}", if n == 1 { "" } else { "s" }));
                }
                if let Some(c) = r.card.commits.last() {
                    s.push_str(&format!(", last: \"{}\"", cap(c, 90)));
                }
                s.push_str(".\nIts summary (asks, files, commits, outcome): mcp__indexio__recall {brief:true, query:\"\"}");
                return Some(crate::redact::redact(&s).into_owned());
            }
            None
        }
        _ => None,
    }
}

/// `2026-09-15T14:03:22.123Z` -> `2026-09-15 14:03`.
fn short_ts(ts: &str) -> String {
    let (d, t) = ts.split_once('T').unwrap_or((ts, ""));
    format!("{d} {}", t.chars().take(5).collect::<String>())
}

/// Harness noise a user message carries: `<system-reminder>` blocks and
/// slash-command wrappers. Keeps the command arguments.
fn clean_user_text(s: &str) -> String {
    // an injected skill document (`# /loop — …`, `# /commit …`): harness
    // material, not conversation
    if s.trim_start().starts_with("# /") {
        return String::new();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    loop {
        match rest.find("<system-reminder>") {
            Some(i) => {
                out.push_str(&rest[..i]);
                match rest[i..].find("</system-reminder>") {
                    Some(j) => rest = &rest[i + j + "</system-reminder>".len()..],
                    None => break, // unterminated: drop the tail
                }
            }
            None => {
                out.push_str(rest);
                break;
            }
        }
    }
    let out = out
        .replace("<command-message>", "")
        .replace("</command-message>", "")
        .replace("<command-name>", "command: ")
        .replace("</command-name>", "")
        .replace("<command-args>", "")
        .replace("</command-args>", "");
    // collapse runs of blank lines
    let mut collapsed = String::with_capacity(out.len());
    let mut blank = 0;
    for line in out.lines() {
        if line.trim().is_empty() {
            blank += 1;
            if blank > 1 {
                continue;
            }
        } else {
            blank = 0;
        }
        collapsed.push_str(line.trim_end());
        collapsed.push('\n');
    }
    collapsed.trim().to_string()
}

/// Lines that say nothing about the exchange: shell housekeeping, git's
/// line-ending warnings, rtk's log pointers.
fn is_noise_line(l: &str) -> bool {
    let t = l.trim();
    t.is_empty()
        || t.starts_with("Shell cwd was reset")
        || t.starts_with("warning: LF will be replaced by CRLF")
        || t.starts_with("warning: CRLF will be replaced by LF")
        || t == "The file will have its original line endings in your working directory"
        || t.starts_with("[full output:")
        || t.starts_with("[exited with code")
}

fn cap(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let mut cut = n;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…", &s[..cut])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_matches_claude_code() {
        assert_eq!(
            project_slug(Path::new(r"C:\Users\me\code\indexio")),
            "C--Users-me-code-indexio"
        );
        assert_eq!(project_slug(Path::new("/home/u/code/x.y")), "-home-u-code-x-y");
    }

    #[test]
    fn strips_reminders_and_keeps_commands() {
        let s = "hello <system-reminder>noise\nmore</system-reminder> world\n\n\n\n<command-name>/loop</command-name><command-args>do x</command-args>";
        assert_eq!(clean_user_text(s), "hello  world\n\ncommand: /loopdo x");
    }

    #[test]
    fn result_noise_lines_are_dropped() {
        assert!(is_noise_line("Shell cwd was reset to C:/x"));
        assert!(is_noise_line("warning: LF will be replaced by CRLF in a.rs."));
        assert!(is_noise_line("The file will have its original line endings in your working directory"));
        assert!(is_noise_line("[full output: ~/x.log]"));
        assert!(is_noise_line("   "));
        assert!(!is_noise_line("fn main() {}"));
        assert!(!is_noise_line("warning: unused variable"));
    }

    /// SPEC-P10 §31: a transcript older than the retention window has its
    /// rendered parts removed and is not re-rendered; a fresh one is kept.
    #[test]
    fn old_transcripts_roll_off() {
        let claude = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        let proj = claude.path().join("projects").join("C--p");
        fs::create_dir_all(&proj).unwrap();
        let line = r#"{"type":"user","timestamp":"2026-09-01T00:00:00Z","message":{"role":"user","content":"hello there indexio"}}"#;
        fs::write(proj.join("old.jsonl"), format!("{line}\n")).unwrap();
        fs::write(proj.join("new.jsonl"), format!("{line}\n")).unwrap();
        std::env::set_var("INDEXIO_RETAIN_DAYS", "30");
        let r = import(claude.path(), out.path(), None).unwrap();
        assert_eq!(r.sessions_imported, 2);
        assert!(out.path().join("C--p").join("old.md").exists());
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(40 * 86_400);
        fs::File::options().write(true).open(proj.join("old.jsonl")).unwrap().set_modified(old).unwrap();
        let r = import(claude.path(), out.path(), None).unwrap();
        assert_eq!(r.sessions_rolled_off, 1, "{r:?}");
        assert!(!out.path().join("C--p").join("old.md").exists());
        assert!(out.path().join("C--p").join("new.md").exists());
        // idempotent: nothing left to roll off, nothing re-rendered
        let r = import(claude.path(), out.path(), None).unwrap();
        assert_eq!((r.sessions_rolled_off, r.sessions_imported), (0, 0));
        std::env::remove_var("INDEXIO_RETAIN_DAYS");
    }

    #[test]
    fn commit_subjects_are_parsed() {
        assert_eq!(commit_message(r#"git add a && git commit -q -m "fix: x" && git push"#).as_deref(), Some("fix: x"));
        assert_eq!(commit_message("git commit -m 'one' ").as_deref(), Some("one"));
        assert_eq!(
            commit_message("git commit -m \"$(cat <<'EOF'\nfeat: heredoc subject\n\nbody\nEOF\n)\"").as_deref(),
            Some("feat: heredoc subject")
        );
        assert_eq!(commit_message("git commit -m \"subject\n\nbody\""), Some("subject".into()));
        assert_eq!(commit_message("git status"), None);
    }

    /// A card: requests (harness turns and /loop repeats dropped), project
    /// files by edit count, scratch files counted, commits, the last answer.
    #[test]
    fn a_session_card_is_written() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = tmp.path().join("projects").join("C--x");
        fs::create_dir_all(&proj).unwrap();
        let lines = [
            r#"{"type":"ai-title","aiTitle":"Retry policy"}"#,
            r#"{"type":"user","cwd":"C:\\x","timestamp":"2026-09-15T10:00:00Z","message":{"role":"user","content":"┌──┐\n│ table │\nadd retries to uploads"}}"#,
            r#"{"type":"assistant","timestamp":"2026-09-15T10:00:05Z","message":{"role":"assistant","content":[{"type":"tool_use","name":"Edit","input":{"file_path":"C:\\x\\src\\up.rs"}},{"type":"tool_use","name":"Edit","input":{"file_path":"C:\\x\\src\\up.rs"}},{"type":"tool_use","name":"Write","input":{"file_path":"C:\\tmp\\scratch.py"}},{"type":"tool_use","name":"Bash","input":{"command":"git commit -m \"feat: upload retries\" && git push"}}]}}"#,
            r#"{"type":"user","timestamp":"2026-09-15T10:01:00Z","message":{"role":"user","content":"<task-notification> done </task-notification>"}}"#,
            r#"{"type":"user","timestamp":"2026-09-15T10:01:30Z","message":{"role":"user","content":"<command-name>/clear</command-name>\n<command-message>clear</command-message>"}}"#,
            r#"{"type":"user","timestamp":"2026-09-15T10:02:00Z","message":{"role":"user","content":"add retries to uploads"}}"#,
            r#"{"type":"assistant","timestamp":"2026-09-15T10:03:00Z","message":{"role":"assistant","content":[{"type":"text","text":"Done: 3 tries with backoff."}]}}"#,
        ];
        fs::write(proj.join("s1.jsonl"), lines.join("\n")).unwrap();
        let out = tmp.path().join("sessions");
        import(tmp.path(), &out, None).unwrap();
        let c = fs::read_to_string(out.join("C--x").join(format!("s1{CARD_SUFFIX}"))).unwrap();
        assert!(c.starts_with("# card s1 — Retry policy\nproject: C:\\x (C--x)\nwhen: 2026-09-15 10:00 → 2026-09-15 10:03"), "{c}");
        assert_eq!(c.matches("add retries to uploads").count(), 1, "{c}");
        assert!(!c.contains("task-notification") && !c.contains("┌") && !c.contains("/clear"), "{c}");
        assert!(c.contains("- src/up.rs ×2\n- (1 file outside the project)"), "{c}");
        assert!(c.contains("## commits\n- feat: upload retries\n"), "{c}");
        assert!(c.contains("## ended with\nDone: 3 tries with backoff."), "{c}");
        // unchanged: skipped; a data dir from before cards (no marker, no
        // card) renders every transcript once more
        assert_eq!(import(tmp.path(), &out, None).unwrap().sessions_imported, 0);
        let card = out.join("C--x").join(format!("s1{CARD_SUFFIX}"));
        fs::remove_file(&card).unwrap();
        fs::remove_file(out.join("C--x").join(CARDS_MARKER)).unwrap();
        assert_eq!(import(tmp.path(), &out, None).unwrap().sessions_imported, 1);
        assert!(card.exists());
        assert_eq!(import(tmp.path(), &out, None).unwrap().sessions_imported, 0);
    }

    /// SessionStart: `startup` points at the newest OTHER session of the
    /// project, `compact` lists this session's own files and commits,
    /// `resume` adds nothing.
    #[test]
    fn session_start_context_by_source() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = Path::new(r"C:\x");
        let proj = tmp.path().join("projects").join(project_slug(cwd));
        fs::create_dir_all(&proj).unwrap();
        let prev = [
            r#"{"type":"ai-title","aiTitle":"Retry policy"}"#,
            r#"{"type":"user","cwd":"C:\\x","timestamp":"2099-01-01T10:00:00Z","message":{"role":"user","content":"add retries"}}"#,
            r#"{"type":"assistant","timestamp":"2099-01-01T10:05:00Z","message":{"role":"assistant","content":[{"type":"tool_use","name":"Edit","input":{"file_path":"C:\\x\\src\\up.rs"}},{"type":"tool_use","name":"Bash","input":{"command":"git commit -m \"feat: retries\""}}]}}"#,
        ];
        fs::write(proj.join("old.jsonl"), prev.join("\n")).unwrap();
        let now = r#"{"type":"user","cwd":"C:\\x","timestamp":"2099-01-02T10:00:00Z","message":{"role":"user","content":"next task"}}"#;
        fs::write(proj.join("cur.jsonl"), now).unwrap();

        let s = session_start_context(tmp.path(), cwd, "cur", None, "startup").unwrap();
        assert!(s.contains("\"Retry policy\" (2099-01-01 10:00 → 2099-01-01 10:05), 1 file changed, 1 commit, last: \"feat: retries\""), "{s}");
        assert!(s.contains("recall {brief:true"), "{s}");
        let c = session_start_context(tmp.path(), cwd, "old", None, "compact").unwrap();
        assert!(c.contains("files changed: src/up.rs") && c.contains("commits: feat: retries"), "{c}");
        assert_eq!(session_start_context(tmp.path(), cwd, "cur", None, "compact"), None, "nothing changed yet");
        assert_eq!(session_start_context(tmp.path(), cwd, "cur", None, "resume"), None);
    }

    #[test]
    fn renders_and_splits_a_transcript() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = tmp.path().join("projects").join("C--x");
        fs::create_dir_all(&proj).unwrap();
        let mut lines = vec![
            r#"{"type":"ai-title","aiTitle":"Retry policy"}"#.to_string(),
            r#"{"type":"user","cwd":"C:\\x","timestamp":"2026-09-15T10:00:00Z","message":{"role":"user","content":"should we retry uploads? <system-reminder>ignore</system-reminder>"}}"#.to_string(),
            r#"{"type":"assistant","timestamp":"2026-09-15T10:00:05Z","message":{"role":"assistant","content":[{"type":"thinking","thinking":"secret"},{"type":"text","text":"Yes: exponential backoff, 3 tries."},{"type":"tool_use","name":"read_span","input":{"repo":"r","path":"a.rs","start":1}}]}}"#.to_string(),
            r#"{"type":"user","timestamp":"2026-09-15T10:00:06Z","message":{"role":"user","content":[{"type":"tool_result","content":"line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8"}]}}"#.to_string(),
            r#"{"type":"summary","summary":"We chose backoff."}"#.to_string(),
            r#"{"type":"cost-state","totalCostUSD":1}"#.to_string(),
        ];
        // pad to force a second part
        for i in 0..400 {
            lines.push(format!(
                r#"{{"type":"assistant","timestamp":"2026-09-15T10:{:02}:00Z","message":{{"role":"assistant","content":[{{"type":"text","text":"{}"}}]}}}}"#,
                i % 60,
                "x".repeat(1000)
            ));
        }
        fs::write(proj.join("s1.jsonl"), lines.join("\n")).unwrap();
        let out = tmp.path().join("sessions");
        let r = import(tmp.path(), &out, None).unwrap();
        assert_eq!((r.sessions_seen, r.sessions_imported), (1, 1));
        assert!(r.parts_written >= 2, "{r:?}");
        let p1 = fs::read_to_string(out.join("C--x").join("s1.part01.md")).unwrap();
        assert!(p1.starts_with("# session s1 — Retry policy\nproject: C:\\x (C--x)"), "{p1}");
        assert!(p1.contains("should we retry uploads?") && !p1.contains("ignore"));
        assert!(p1.contains("Yes: exponential backoff"));
        assert!(p1.contains("~ secret"), "thinking is rendered, capped (SPEC-P10)");
        assert!(p1.contains("-> read_span {\"path\":\"a.rs\""));
        assert!(p1.contains("<- line1\nline2") && !p1.contains("line7"));
        assert!(p1.contains("## compaction summary\nWe chose backoff."));
        // unchanged transcript: skipped next time
        let r2 = import(tmp.path(), &out, None).unwrap();
        assert_eq!((r2.sessions_seen, r2.sessions_imported), (1, 0));
        // only_slug filter
        let r3 = import(tmp.path(), &out, Some("other")).unwrap();
        assert_eq!(r3.sessions_seen, 0);
    }
}
