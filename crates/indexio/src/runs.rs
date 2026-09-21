//! Command output kept out of the model's context (SPEC-P10 §31).
//!
//! `indexio run -- <command>` runs the command through the shell and, when
//! the output is long, stores the whole of it under
//! `<data_dir>/runs/<repo>/<stamp>-<slug>.log` and prints a digest: exit
//! code, size, the first lines, the lines that look like errors, the last
//! lines, and the `runs:<repo>/<file>` pointer to `read_span` for anything
//! else. Short output is printed as it is. The `runs` folder is a plain
//! source indexed by the server's periodic import, searchable and reachable
//! by `recall`, and rolled off after [`retain_days`] days so it never
//! grows without bound. The Bash hook rewrites script and build commands
//! to go through here (`hook::runner_rewrite`).

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context;

/// Name of the plain-folder source the outputs are indexed under.
pub const REPO: &str = "runs";

/// Output up to this many lines and bytes is printed whole and not stored.
pub const INLINE_LINES: usize = 60;
pub const INLINE_BYTES: usize = 6 * 1024;
/// Digest shape for longer output.
const HEAD_LINES: usize = 12;
const TAIL_LINES: usize = 25;
const FLAGGED_LINES: usize = 25;
const LINE_CAP: usize = 240;
/// Stored output is capped so a runaway command cannot fill the disk.
const STORE_CAP: usize = 8 * 1024 * 1024;

/// Days a run log or a rendered session transcript is kept
/// (`INDEXIO_RETAIN_DAYS`, default 30; 0 keeps everything).
pub fn retain_days() -> u64 {
    std::env::var("INDEXIO_RETAIN_DAYS").ok().and_then(|s| s.trim().parse().ok()).unwrap_or(30)
}

/// One shell word per argument: arguments the shell would split or
/// interpret are single-quoted (`'` becomes `'\''`), so
/// `indexio run -- python -c "print(1)"` reaches the shell as typed. A
/// single argument (the hook passes the whole command as one) is used
/// as it is.
pub fn shell_join(args: &[String]) -> String {
    if args.len() == 1 {
        return args[0].clone();
    }
    args.iter()
        .map(|a| {
            let plain = !a.is_empty() && a.chars().all(|c| c.is_ascii_alphanumeric() || "_-./:=@%+,".contains(c));
            if plain { a.clone() } else { format!("'{}'", a.replace('\'', "'\\''")) }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The registered repo whose folder contains `cwd`, else `_`.
fn repo_for(data_dir: &Path, cwd: &Path) -> String {
    crate::mcp::repo_containing(data_dir, cwd).unwrap_or_else(|| "_".to_string())
}

/// A file-name-safe slug of the command: its first words, lower-cased.
fn slug(command: &str) -> String {
    let mut s: String = command
        .split_whitespace()
        .take(4)
        .collect::<Vec<_>>()
        .join("_")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' { c.to_ascii_lowercase() } else { '-' })
        .collect();
    s.truncate(48);
    if s.is_empty() {
        s.push_str("run");
    }
    s
}

fn stamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // civil date from the epoch (Howard Hinnant's algorithm), no chrono dep
    let days = secs / 86_400;
    let rem = secs % 86_400;
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}", rem / 3600, (rem % 3600) / 60, rem % 60)
}

fn looks_flagged(line: &str) -> bool {
    let l = line.to_ascii_lowercase();
    ["error", "warning:", "failed", "failure", "panicked", "traceback", "exception", "assert", " fail", "fatal", "cannot ", "not found", "denied"]
        .iter()
        .any(|k| l.contains(k))
}

fn cap_line(l: &str) -> String {
    if l.chars().count() > LINE_CAP {
        let mut s: String = l.chars().take(LINE_CAP).collect();
        s.push('…');
        s
    } else {
        l.to_string()
    }
}

/// The digest of a long output: head, flagged lines with their numbers,
/// tail, and the pointer. `pointer` is the `repo:path` of the stored log.
pub fn digest(output: &str, pointer: &str, exit: i32, ms: u128) -> String {
    let lines: Vec<&str> = output.lines().collect();
    let n = lines.len();
    let mut out = format!(
        "[indexio run] exit {exit}, {} s, {n} lines, {} KB -> {pointer} (read_span for the rest)\n",
        ms / 1000,
        output.len() / 1024
    );
    let head = HEAD_LINES.min(n);
    for l in &lines[..head] {
        out.push_str(&cap_line(l));
        out.push('\n');
    }
    let tail_start = n.saturating_sub(TAIL_LINES).max(head);
    let mut flagged = 0usize;
    let mut skipped = 0usize;
    for (i, l) in lines.iter().enumerate().take(tail_start).skip(head) {
        if flagged < FLAGGED_LINES && looks_flagged(l) {
            if skipped > 0 {
                out.push_str(&format!("… {skipped} lines\n"));
                skipped = 0;
            }
            out.push_str(&format!("{}: {}\n", i + 1, cap_line(l)));
            flagged += 1;
        } else {
            skipped += 1;
        }
    }
    if skipped > 0 {
        out.push_str(&format!("… {skipped} lines\n"));
    }
    if tail_start < n {
        if tail_start > head {
            out.push_str(&format!("--- last {} lines (from {})\n", n - tail_start, tail_start + 1));
        }
        for l in &lines[tail_start..] {
            out.push_str(&cap_line(l));
            out.push('\n');
        }
    }
    out
}

/// Run `command` in `cwd` through the shell. Short output is printed as it
/// is; long output is stored under the runs source and a digest is
/// printed. Returns the command's exit code.
pub fn run(data_dir: &Path, cwd: &Path, command: &str) -> anyhow::Result<i32> {
    let t0 = std::time::Instant::now();
    let shell = std::env::var("SHELL").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "bash".to_string());
    // The command text travels in an environment variable and is `eval`ed:
    // an argv string would go through the MSYS runtime's quote conversion
    // on Windows (single quotes are stripped), an env var is untouched.
    // stderr is interleaved with stdout, as the terminal would show it.
    let script = "( eval \"$INDEXIO_RUN_CMD\" ) 2>&1";
    let spawn = |sh: &str| {
        std::process::Command::new(sh)
            .arg("-c")
            .arg(script)
            .env("INDEXIO_RUN_CMD", command)
            .current_dir(cwd)
            .output()
    };
    let out = spawn(&shell)
        .or_else(|_| spawn("sh"))
        .with_context(|| format!("running `{command}` through {shell}"))?;
    let exit = out.status.code().unwrap_or(-1);
    let ms = t0.elapsed().as_millis();
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let lines = text.lines().count();
    let mut stdout = std::io::stdout().lock();
    if lines <= INLINE_LINES && text.len() <= INLINE_BYTES {
        stdout.write_all(text.as_bytes())?;
        if exit != 0 {
            writeln!(stdout, "[indexio run] exit {exit}")?;
        }
        return Ok(exit);
    }
    // the stored log is indexed and searchable for 30 days: secrets the
    // command printed (an env dump, a connection string) stay out of it
    let text = crate::redact::redact(&text).into_owned();
    let repo = repo_for(data_dir, cwd);
    let dir = data_dir.join(REPO).join(&repo);
    fs::create_dir_all(&dir)?;
    let name = format!("{}-{}.log", stamp(), slug(command));
    let path = dir.join(&name);
    let mut stored = String::with_capacity(text.len().min(STORE_CAP) + 256);
    stored.push_str(&format!("# {command}\n# cwd {}  exit {exit}  {} s\n", cwd.display(), ms / 1000));
    if text.len() > STORE_CAP {
        stored.push_str(&text[..STORE_CAP]);
        stored.push_str("\n# [truncated]\n");
    } else {
        stored.push_str(&text);
    }
    fs::write(&path, stored)?;
    let pointer = format!("{REPO}:{repo}/{name}");
    stdout.write_all(digest(&text, &pointer, exit, ms).as_bytes())?;
    Ok(exit)
}

/// Delete run logs older than `days` (SPEC-P10 §31); returns how many.
pub fn retain(data_dir: &Path, days: u64) -> usize {
    if days == 0 {
        return 0;
    }
    let root = data_dir.join(REPO);
    let cutoff = std::time::SystemTime::now() - std::time::Duration::from_secs(days * 86_400);
    let mut removed = 0;
    let Ok(repos) = fs::read_dir(&root) else { return 0 };
    for repo in repos.flatten() {
        let Ok(files) = fs::read_dir(repo.path()) else { continue };
        for f in files.flatten() {
            let old = f.metadata().and_then(|m| m.modified()).map(|t| t < cutoff).unwrap_or(false);
            if old && fs::remove_file(f.path()).is_ok() {
                removed += 1;
            }
        }
    }
    removed
}

/// Register `<data_dir>/runs` as a plain-folder source (once) and delta
/// re-index it, like the sessions source.
pub fn index_runs(
    data_dir: &Path,
    cas: &indexio_ingest::Cas,
    cache: Option<&mut indexio_ingest::WorktreeCache>,
) -> anyhow::Result<indexio_ingest::IndexReport> {
    let out: PathBuf = data_dir.join(REPO);
    fs::create_dir_all(&out)?;
    if indexio_ingest::repo_state(data_dir, REPO).is_err() {
        let src = indexio_ingest::sources::parse_source(&out.to_string_lossy())?;
        indexio_ingest::sources::add_source(data_dir, src)?;
        return indexio_ingest::index_repo(&out, REPO, data_dir, cas);
    }
    indexio_ingest::reindex_worktree_cached(REPO, data_dir, cas, cache)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_keeps_head_flagged_and_tail() {
        let mut lines: Vec<String> = (1..=200).map(|i| format!("line {i}")).collect();
        lines[99] = "error[E0308]: mismatched types".into();
        lines[150] = "test foo ... FAILED".into();
        let text = lines.join("\n");
        let d = digest(&text, "runs:r/x.log", 101, 2500);
        assert!(d.starts_with("[indexio run] exit 101, 2 s, 200 lines"), "{d}");
        assert!(d.contains("runs:r/x.log"));
        assert!(d.contains("line 1\n") && d.contains("line 12\n"), "head");
        assert!(d.contains("100: error[E0308]") && d.contains("151: test foo ... FAILED"), "flagged with numbers: {d}");
        assert!(d.contains("line 200\n") && d.contains("--- last 25 lines"), "tail: {d}");
        assert!(!d.contains("line 50\n"), "the middle is elided");
        assert!(d.len() < text.len() / 2);
    }

    #[test]
    fn retention_removes_only_old_logs() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join(REPO).join("r");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("new.log"), "x").unwrap();
        fs::write(dir.join("old.log"), "x").unwrap();
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(40 * 86_400);
        fs::File::options().write(true).open(dir.join("old.log")).unwrap().set_modified(old).unwrap();
        assert_eq!(retain(tmp.path(), 30), 1);
        assert!(dir.join("new.log").exists());
        assert!(!dir.join("old.log").exists());
        assert_eq!(retain(tmp.path(), 0), 0, "0 keeps everything");
    }

    #[test]
    fn shell_join_keeps_quoted_arguments_intact() {
        let a = |v: &[&str]| shell_join(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert_eq!(a(&["python", "-c", "print('hi'); print(2)"]), r"python -c 'print('\''hi'\''); print(2)'");
        assert_eq!(a(&["npm", "run", "build:web"]), "npm run build:web");
        assert_eq!(a(&["python scripts/probe.py --fast"]), "python scripts/probe.py --fast", "one argument is the command");
    }

    #[test]
    fn slug_and_stamp_are_file_safe() {
        assert_eq!(slug("python scripts/probe.py --fast \"a b\""), "python_scripts-probe.py_--fast_-a");
        let s = stamp();
        assert_eq!(s.len(), 15, "{s}");
        assert!(s.chars().all(|c| c.is_ascii_digit() || c == '-'));
    }
}
