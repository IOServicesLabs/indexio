//! Team worktree overlays: a central server's view of each developer's
//! uncommitted work, fed by the harness (a Claude Code hook sends
//! `git diff --binary <base>`), with nothing installed on the developer's
//! machine.
//!
//! Layout, per user, beside the shared index:
//!   <data_dir>/team/<user>/shards/*.cidx        one shard per repo overlay
//!   <data_dir>/team/<user>/repos/<repo>.json    [`OverlayState`]
//!
//! A patch is applied to the base commit in a scratch git index of the
//! server's clone (`read-tree` + `apply --cached` + `write-tree`), so the
//! server needs neither a checkout per user nor a patch parser of its own.
//! The files the patch changed are indexed into a fresh overlay shard that
//! replaces the user's previous one for that repo; the files it deleted
//! are recorded so the query side hides the base docs under them. The
//! shared index is only ever read.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead as _, BufReader, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{bail, Context};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use indexio_types::Lang;

use crate::{extract_docs, iso_now, load_state, normalize_crlf, under_skipped_dir, write_shard, Cas, RepoLock};

/// How long an overlay update waits for another update of the same one.
const OVERLAY_LOCK_WAIT: Duration = Duration::from_secs(10);

/// Largest patch accepted (a worktree diff past this is a vendored tree or
/// a generated dump, not work in progress).
pub const MAX_PATCH_BYTES: usize = 32 * 1024 * 1024;

/// One user's overlay of one repo, JSON at `team/<user>/repos/<repo>.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OverlayState {
    pub repo: String,
    pub user: String,
    /// Commit the patch was taken against.
    pub base: String,
    /// File name of the overlay shard under `team/<user>/shards/` (`None`
    /// when the overlay only deletes files).
    pub shard: Option<String>,
    /// Paths the patch added or modified, sorted.
    pub changed: Vec<String>,
    /// Paths hidden from the base: deleted by the patch, or changed into
    /// something no longer indexable (binary, too big).
    pub deleted: Vec<String>,
    /// blake3 of the patch last applied: a repeat of it is a no-op.
    pub patch_hash: String,
    pub updated_at: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct OverlayReport {
    pub repo: String,
    pub user: String,
    pub base: String,
    /// Docs in the new overlay shard.
    pub docs: u64,
    pub changed: u64,
    pub deleted: u64,
    /// True when the patch was identical to the last one (nothing done).
    pub unchanged: bool,
    /// True when an empty patch removed the overlay.
    pub cleared: bool,
    pub elapsed_ms: u64,
}

/// Why an update was refused; the server maps these to HTTP statuses.
#[derive(Debug, thiserror::Error)]
pub enum OverlayError {
    #[error("unknown repo '{0}': not registered on this server")]
    UnknownRepo(String),
    #[error("base commit {0} is not on the server (push it, or wait for the next sync)")]
    UnknownBase(String),
    #[error("{0}")]
    Invalid(String),
}

/// A user name safe to use as a directory name: 1-64 of `[A-Za-z0-9._@-]`,
/// not starting with a dot. `@` is included so OIDC identities (email
/// addresses) can key team overlays directly.
pub fn valid_user(user: &str) -> bool {
    !user.is_empty()
        && user.len() <= 64
        && !user.starts_with('.')
        && user
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'@'))
}

fn is_hex_commit(s: &str) -> bool {
    (s.len() == 40 || s.len() == 64) && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// `team/<user>` under the data dir.
pub fn user_dir(data_dir: &Path, user: &str) -> PathBuf {
    data_dir.join("team").join(user)
}

fn overlay_state_path(data_dir: &Path, user: &str, repo: &str) -> PathBuf {
    user_dir(data_dir, user).join("repos").join(format!("{repo}.json"))
}

fn save_overlay(data_dir: &Path, st: &OverlayState) -> anyhow::Result<()> {
    let path = overlay_state_path(data_dir, &st.user, &st.repo);
    let dir = path.parent().expect("state path has a parent");
    fs::create_dir_all(dir)?;
    let tmp = dir.join(format!(".{}.json.tmp", st.repo));
    fs::write(&tmp, serde_json::to_vec_pretty(st)?)?;
    fs::rename(&tmp, &path)?;
    Ok(())
}

fn load_overlay(data_dir: &Path, user: &str, repo: &str) -> Option<OverlayState> {
    let bytes = fs::read(overlay_state_path(data_dir, user, repo)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Every overlay of `user`, one per repo.
pub fn user_overlays(data_dir: &Path, user: &str) -> Vec<OverlayState> {
    let Ok(rd) = fs::read_dir(user_dir(data_dir, user).join("repos")) else {
        return Vec::new();
    };
    let mut out: Vec<OverlayState> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| serde_json::from_slice(&fs::read(e.path()).ok()?).ok())
        .collect();
    out.sort_by(|a, b| a.repo.cmp(&b.repo));
    out
}

/// What a query engine needs to layer `user`'s overlays over the shared
/// index: their shard dir (when any shard exists) and the (repo, path)
/// keys they hide.
pub struct UserLayer {
    pub shards_dir: Option<PathBuf>,
    pub hidden: HashSet<(String, String)>,
}

pub fn user_layer(data_dir: &Path, user: &str) -> UserLayer {
    let overlays = user_overlays(data_dir, user);
    let hidden = overlays
        .iter()
        .flat_map(|o| o.deleted.iter().map(move |p| (o.repo.clone(), p.clone())))
        .collect();
    let dir = user_dir(data_dir, user).join("shards");
    UserLayer {
        shards_dir: overlays.iter().any(|o| o.shard.is_some()).then_some(dir),
        hidden,
    }
}

/// A cheap change stamp of `user`'s overlays (their state files' names,
/// sizes and mtimes): a server caches one engine per user and rebuilds it
/// when this moves.
pub fn user_stamp(data_dir: &Path, user: &str) -> u64 {
    let Ok(rd) = fs::read_dir(user_dir(data_dir, user).join("repos")) else {
        return 0;
    };
    let mut h = blake3::Hasher::new();
    let mut rows: Vec<(String, u64, u128)> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            let t = m.modified().ok()?.duration_since(SystemTime::UNIX_EPOCH).ok()?.as_nanos();
            Some((e.file_name().to_string_lossy().into_owned(), m.len(), t))
        })
        .collect();
    if rows.is_empty() {
        return 0;
    }
    rows.sort();
    for (n, l, t) in rows {
        h.update(n.as_bytes());
        h.update(&l.to_le_bytes());
        h.update(&t.to_le_bytes());
    }
    u64::from_le_bytes(h.finalize().as_bytes()[..8].try_into().expect("8 bytes"))
}

/// Canonical form of a git remote URL, so `git@github.com:Org/Repo.git`,
/// `https://github.com/org/repo` and `ssh://git@github.com/org/repo/` all
/// name one repository: `github.com/org/repo`.
pub fn normalize_remote(url: &str) -> String {
    let mut s = url.trim().to_ascii_lowercase();
    if let Some(i) = s.find("://") {
        s = s[i + 3..].to_string();
    } else if let Some(i) = s.find(':') {
        // scp-like `user@host:path`
        if !s[..i].contains('/') {
            s = format!("{}/{}", &s[..i], &s[i + 1..]);
        }
    }
    if let Some(i) = s.find('@') {
        if !s[..i].contains('/') {
            s = s[i + 1..].to_string();
        }
    }
    // drop a port on the host (`host:22/org/repo`)
    if let Some(slash) = s.find('/') {
        if let Some(colon) = s[..slash].find(':') {
            s = format!("{}{}", &s[..colon], &s[slash..]);
        }
    }
    let s = s.trim_end_matches('/');
    s.strip_suffix(".git").unwrap_or(s).trim_end_matches('/').to_string()
}

/// `normalize_remote(origin url) -> repo name` for every registered git
/// repo (read from each clone's config, no subprocess).
pub fn remote_map(data_dir: &Path) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for name in crate::sources::registered_repo_names(data_dir).unwrap_or_default() {
        let Ok(st) = load_state(data_dir, &name) else { continue };
        if st.plain {
            continue;
        }
        let Ok(repo) = gix::open(&st.path) else { continue };
        if let Some(url) = repo.config_snapshot().string("remote.origin.url") {
            out.insert(normalize_remote(&url.to_string()), name);
        }
    }
    out
}

/// The registered repo a client names: its registered name, or its origin
/// URL in any common spelling (looked up in `remotes`, see [`remote_map`]).
pub fn resolve_repo(data_dir: &Path, remotes: &HashMap<String, String>, hint: &str) -> Option<String> {
    let hint = hint.trim();
    if hint.is_empty() {
        return None;
    }
    if valid_repo_name(hint) && load_state(data_dir, hint).is_ok() {
        return Some(hint.to_string());
    }
    remotes.get(&normalize_remote(hint)).cloned()
}

fn valid_repo_name(s: &str) -> bool {
    !s.is_empty() && !s.contains(['/', '\\', ':']) && !s.starts_with('.')
}

/// Run git in `repo` with an optional scratch index and stdin; stdout on
/// success.
fn git(repo: &Path, index: Option<&Path>, args: &[&str], stdin: Option<&[u8]>) -> anyhow::Result<Vec<u8>> {
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(repo).args(args);
    if let Some(ix) = index {
        cmd.env("GIT_INDEX_FILE", ix);
    }
    cmd.stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().with_context(|| format!("running git {}", args.join(" ")))?;
    if let Some(input) = stdin {
        let mut pipe = child.stdin.take().expect("piped stdin");
        let input = input.to_vec();
        // write on a thread: a large patch would otherwise deadlock
        // against git filling its stdout/stderr pipes
        std::thread::spawn(move || {
            let _ = pipe.write_all(&input);
        });
    }
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!("git {} failed: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(out.stdout)
}

/// Least time between two on-demand fetches of one repo: an unknown base
/// from a client must not turn every request into a network round trip.
const FETCH_EVERY: Duration = Duration::from_secs(60);
/// Longest an on-demand fetch may run before it is killed.
const FETCH_TIMEOUT: Duration = Duration::from_secs(20);

/// True (and the attempt recorded) when `repo` was not fetched on demand
/// within [`FETCH_EVERY`]; the marker is `team/.fetch/<repo>`'s mtime.
fn fetch_due(data_dir: &Path, repo: &str) -> bool {
    let dir = data_dir.join("team").join(".fetch");
    let marker = dir.join(repo);
    let recent = fs::metadata(&marker)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .is_some_and(|age| age < FETCH_EVERY);
    if recent {
        return false;
    }
    let _ = fs::create_dir_all(&dir);
    let _ = fs::write(&marker, b"");
    true
}

/// `git fetch origin` that never prompts and is killed after
/// [`FETCH_TIMEOUT`] (an unreachable remote must not hold a request).
fn fetch_origin(repo: &Path) {
    let child = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["fetch", "--quiet", "origin"])
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=10")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    let Ok(mut child) = child else { return };
    let deadline = Instant::now() + FETCH_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(50)),
            _ => {
                warn!(repo = %repo.display(), "on-demand fetch timed out");
                let _ = child.kill();
                let _ = child.wait();
                return;
            }
        }
    }
}

fn has_commit(repo: &Path, commit: &str) -> bool {
    git(repo, None, &["cat-file", "-e", &format!("{commit}^{{commit}}")], None).is_ok()
}

/// Contents of `paths` in `tree` via one `git cat-file --batch`. A path
/// missing from the tree (or not a blob) is absent from the result.
fn read_tree_blobs(repo: &Path, tree: &str, paths: &[String]) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
    let mut child = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["cat-file", "--batch"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("running git cat-file --batch")?;
    let mut input = String::new();
    for p in paths {
        input.push_str(&format!("{tree}:{p}\n"));
    }
    let mut pipe = child.stdin.take().expect("piped stdin");
    let writer = std::thread::spawn(move || {
        let _ = pipe.write_all(input.as_bytes());
    });
    let mut rd = BufReader::new(child.stdout.take().expect("piped stdout"));
    let mut out = Vec::with_capacity(paths.len());
    for p in paths {
        let mut header = String::new();
        if rd.read_line(&mut header)? == 0 {
            break;
        }
        let parts: Vec<&str> = header.split_whitespace().collect();
        // `<oid> blob <size>` or `<spec> missing`
        if parts.len() == 3 {
            let size: usize = parts[2].parse().context("cat-file size")?;
            let mut buf = vec![0u8; size + 1]; // content + trailing LF
            rd.read_exact(&mut buf)?;
            buf.truncate(size);
            if parts[1] == "blob" {
                out.push((p.clone(), buf));
            }
        }
    }
    drop(rd);
    let _ = writer.join();
    let _ = child.wait();
    Ok(out)
}

/// Remove a shard file of a replaced overlay; one still mapped by a live
/// engine (Windows refuses to unlink it) is parked as `.stale` and reaped
/// by the next directory open.
fn retire_shard(path: &Path) {
    if fs::remove_file(path).is_err() {
        let _ = fs::rename(path, path.with_extension("stale"));
    }
}

/// Apply `patch` (the output of `git diff --binary <base>` over the user's
/// working tree, untracked files included) as `user`'s overlay of `repo`,
/// replacing their previous one. An empty patch removes the overlay.
pub fn apply_worktree_patch(
    data_dir: &Path,
    cas: &Cas,
    repo: &str,
    user: &str,
    base: &str,
    patch: &[u8],
) -> anyhow::Result<OverlayReport> {
    let t0 = Instant::now();
    if !valid_user(user) {
        return Err(OverlayError::Invalid(format!("invalid user name '{user}'")).into());
    }
    if !is_hex_commit(base) {
        return Err(OverlayError::Invalid(format!("base must be a full commit id, got '{base}'")).into());
    }
    if patch.len() > MAX_PATCH_BYTES {
        return Err(OverlayError::Invalid(format!("patch over {MAX_PATCH_BYTES} bytes")).into());
    }
    let state = load_state(data_dir, repo).map_err(|_| OverlayError::UnknownRepo(repo.to_string()))?;
    if state.plain {
        return Err(OverlayError::Invalid(format!("'{repo}' is a plain folder, not a git repo")).into());
    }
    let base = base.to_ascii_lowercase();
    let udir = user_dir(data_dir, user);
    fs::create_dir_all(udir.join("shards"))?;
    let Some(_lock) = RepoLock::acquire(&udir, repo, OVERLAY_LOCK_WAIT) else {
        bail!("another update of this overlay is still running");
    };
    let mut report = OverlayReport {
        repo: repo.to_string(),
        user: user.to_string(),
        base: base.clone(),
        ..OverlayReport::default()
    };
    let prev = load_overlay(data_dir, user, repo);
    let patch_hash = blake3::hash(patch).to_hex().to_string();
    if let Some(p) = &prev {
        if p.patch_hash == patch_hash && p.base == base {
            report.unchanged = true;
            report.docs = p.changed.len() as u64;
            report.changed = p.changed.len() as u64;
            report.deleted = p.deleted.len() as u64;
            report.elapsed_ms = t0.elapsed().as_millis() as u64;
            return Ok(report);
        }
    }

    if patch.iter().all(u8::is_ascii_whitespace) {
        if let Some(p) = prev {
            if let Some(s) = &p.shard {
                retire_shard(&udir.join("shards").join(s));
            }
            let _ = fs::remove_file(overlay_state_path(data_dir, user, repo));
        }
        report.cleared = true;
        report.elapsed_ms = t0.elapsed().as_millis() as u64;
        return Ok(report);
    }

    // The base must be in the server's clone: a commit the user made
    // locally and never pushed is not, so clients diff against their
    // merge-base with the remote. One fetch covers a push the sync has not
    // picked up yet.
    let clone = state.path.as_path();
    if !has_commit(clone, &base) {
        if fetch_due(data_dir, repo) {
            fetch_origin(clone);
        }
        if !has_commit(clone, &base) {
            return Err(OverlayError::UnknownBase(base).into());
        }
    }

    let tmp = udir.join("tmp");
    fs::create_dir_all(&tmp)?;
    let index = tmp.join(format!("{repo}.index"));
    let _ = fs::remove_file(&index);
    git(clone, Some(&index), &["read-tree", &base], None)?;
    git(
        clone,
        Some(&index),
        &["apply", "--cached", "--binary", "--whitespace=nowarn", "--recount", "-"],
        Some(patch),
    )
    .map_err(|e| OverlayError::Invalid(format!("patch does not apply to {base}: {e:#}")))?;
    let tree = String::from_utf8(git(clone, Some(&index), &["write-tree"], None)?)?.trim().to_string();
    let _ = fs::remove_file(&index);

    let status = git(clone, None, &["diff-tree", "-r", "-z", "--no-renames", "--name-status", &base, &tree], None)?;
    let mut changed: Vec<String> = Vec::new();
    let mut deleted: Vec<String> = Vec::new();
    let mut fields = status.split(|&b| b == 0).filter(|f| !f.is_empty());
    while let (Some(st), Some(path)) = (fields.next(), fields.next()) {
        let path = String::from_utf8_lossy(path).into_owned();
        if under_skipped_dir(&path) {
            continue;
        }
        match st.first() {
            Some(b'D') => deleted.push(path),
            Some(_) => changed.push(path),
            None => {}
        }
    }
    let wanted: Vec<String> = changed
        .iter()
        .filter(|p| !Lang::is_secret_path(p) && !p.contains('\n'))
        .cloned()
        .collect();
    let contents: Vec<(String, Vec<u8>)> = read_tree_blobs(clone, &tree, &wanted)?
        .into_iter()
        .map(|(p, c)| {
            let c = normalize_crlf(&c).unwrap_or(c);
            (p, c)
        })
        .collect();
    let docs = extract_docs(contents, cas);
    let indexed: HashSet<&str> = docs.iter().map(|d| d.meta.path.as_str()).collect();
    // changed into something not indexable: the base doc is stale, hide it
    for p in &changed {
        if !indexed.contains(p.as_str()) {
            deleted.push(p.clone());
        }
    }
    changed.retain(|p| indexed.contains(p.as_str()));
    changed.sort();
    deleted.sort();
    deleted.dedup();

    let shard_path = write_shard(&udir.join("shards"), repo, &docs)?;
    let st = OverlayState {
        repo: repo.to_string(),
        user: user.to_string(),
        base: base.clone(),
        shard: shard_path
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned()),
        changed,
        deleted,
        patch_hash,
        updated_at: iso_now(),
    };
    save_overlay(data_dir, &st)?;
    if let Some(old) = prev.and_then(|p| p.shard) {
        if Some(&old) != st.shard.as_ref() {
            retire_shard(&udir.join("shards").join(old));
        }
    }
    report.docs = docs.len() as u64;
    report.changed = st.changed.len() as u64;
    report.deleted = st.deleted.len() as u64;
    report.elapsed_ms = t0.elapsed().as_millis() as u64;
    debug!(repo, user, docs = report.docs, deleted = report.deleted, ms = report.elapsed_ms, "team overlay");
    Ok(report)
}

/// Drop overlays not updated for `max_age` (a developer who stopped
/// working on a repo, or left): their shard and state go, the base shows
/// through again. Returns the number removed.
pub fn prune_overlays(data_dir: &Path, max_age: Duration) -> usize {
    let Ok(users) = fs::read_dir(data_dir.join("team")) else {
        return 0;
    };
    let mut removed = 0;
    for u in users.filter_map(|e| e.ok()) {
        let user = u.file_name().to_string_lossy().into_owned();
        if !valid_user(&user) {
            continue;
        }
        let Ok(rd) = fs::read_dir(u.path().join("repos")) else { continue };
        for e in rd.filter_map(|e| e.ok()) {
            let path = e.path();
            if path.extension().is_none_or(|x| x != "json") {
                continue;
            }
            let old = e
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.elapsed().ok())
                .is_some_and(|age| age > max_age);
            if !old {
                continue;
            }
            match fs::read(&path).ok().and_then(|b| serde_json::from_slice::<OverlayState>(&b).ok()) {
                Some(st) => {
                    if let Some(s) = &st.shard {
                        retire_shard(&u.path().join("shards").join(s));
                    }
                }
                None => warn!(path = %path.display(), "unreadable overlay state; removing"),
            }
            if fs::remove_file(&path).is_ok() {
                removed += 1;
            }
        }
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn git_t(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("HOME", dir)
            .output()
            .expect("spawn git");
        assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// A registered repo in `data` plus a developer clone of it.
    fn fixture() -> (tempfile::TempDir, PathBuf, PathBuf, String) {
        let td = tempfile::tempdir().unwrap();
        let origin = td.path().join("origin");
        fs::create_dir_all(origin.join("src")).unwrap();
        git_t(&origin, &["init", "-q"]);
        git_t(&origin, &["config", "user.email", "t@example.com"]);
        git_t(&origin, &["config", "user.name", "t"]);
        git_t(&origin, &["config", "commit.gpgsign", "false"]);
        fs::write(origin.join("src/main.rs"), "fn main() { base_only(); }\n").unwrap();
        fs::write(origin.join("src/gone.rs"), "fn doomed_function() {}\n").unwrap();
        git_t(&origin, &["add", "-A"]);
        git_t(&origin, &["commit", "-q", "-m", "init"]);
        let base = git_t(&origin, &["rev-parse", "HEAD"]);
        let data = td.path().join("data");
        let cas = Cas::open(&data.join("cas")).unwrap();
        crate::index_repo(&origin, "demo", &data, &cas).unwrap();
        let dev = td.path().join("dev");
        git_t(td.path(), &["clone", "-q", origin.to_str().unwrap(), dev.to_str().unwrap()]);
        (td, data, dev, base)
    }

    fn worktree_patch(dev: &Path, base: &str) -> Vec<u8> {
        let ix = dev.join(".git").join("indexio-index");
        let run = |args: &[&str]| {
            let out = Command::new("git")
                .arg("-C")
                .arg(dev)
                .args(args)
                .env("GIT_INDEX_FILE", &ix)
                .output()
                .unwrap();
            assert!(out.status.success(), "{args:?}: {}", String::from_utf8_lossy(&out.stderr));
            out.stdout
        };
        run(&["read-tree", base]);
        run(&["add", "-A"]);
        run(&["diff", "--cached", "--binary", base])
    }

    #[test]
    fn remote_urls_normalize_to_one_key() {
        let k = "github.com/org/repo";
        for u in [
            "git@github.com:Org/Repo.git",
            "https://github.com/org/repo",
            "https://user@github.com/org/repo.git/",
            "ssh://git@github.com:22/org/repo",
        ] {
            assert_eq!(normalize_remote(u), k, "{u}");
        }
    }

    #[test]
    fn user_names_are_path_safe() {
        assert!(valid_user("alice.smith-2"));
        // OIDC identities (emails) are valid overlay keys
        assert!(valid_user("alice.smith@corp.example"));
        for bad in ["", ".hidden", "a/b", "..", "a b", "x\\y", "a@b/c"] {
            assert!(!valid_user(bad), "{bad}");
        }
    }

    #[test]
    fn overlay_shadows_hides_and_clears() {
        let (_td, data, dev, base) = fixture();
        let cas = Cas::open(&data.join("cas")).unwrap();
        fs::write(dev.join("src/main.rs"), "fn main() { alice_edit(); }\n").unwrap();
        fs::write(dev.join("src/new.rs"), "fn brand_new_untracked() {}\n").unwrap();
        fs::remove_file(dev.join("src/gone.rs")).unwrap();
        let patch = worktree_patch(&dev, &base);

        let r = apply_worktree_patch(&data, &cas, "demo", "alice", &base, &patch).unwrap();
        assert_eq!((r.docs, r.changed, r.deleted), (2, 2, 1), "{r:?}");
        let layer = user_layer(&data, "alice");
        assert!(layer.hidden.contains(&("demo".into(), "src/gone.rs".into())));
        assert!(user_stamp(&data, "alice") != 0);

        // the same patch again is a no-op
        let again = apply_worktree_patch(&data, &cas, "demo", "alice", &base, &patch).unwrap();
        assert!(again.unchanged);

        // an empty patch clears the overlay
        let cleared = apply_worktree_patch(&data, &cas, "demo", "alice", &base, b"").unwrap();
        assert!(cleared.cleared);
        assert!(user_overlays(&data, "alice").is_empty());
        assert_eq!(user_stamp(&data, "alice"), 0);
    }

    #[test]
    fn unknown_base_and_repo_are_refused() {
        let (_td, data, _dev, _base) = fixture();
        let cas = Cas::open(&data.join("cas")).unwrap();
        let e = apply_worktree_patch(&data, &cas, "demo", "bob", &"a".repeat(40), b"x").unwrap_err();
        assert!(matches!(e.downcast_ref::<OverlayError>(), Some(OverlayError::UnknownBase(_))), "{e:#}");
        let e = apply_worktree_patch(&data, &cas, "nope", "bob", &"a".repeat(40), b"x").unwrap_err();
        assert!(matches!(e.downcast_ref::<OverlayError>(), Some(OverlayError::UnknownRepo(_))), "{e:#}");
    }

    #[test]
    fn resolve_by_name_or_origin_url() {
        let (td, data, _dev, _base) = fixture();
        let origin = td.path().join("origin");
        git_t(&origin, &["remote", "add", "origin", "git@github.com:Acme/Demo.git"]);
        let remotes = remote_map(&data);
        assert_eq!(resolve_repo(&data, &remotes, "demo").as_deref(), Some("demo"));
        assert_eq!(resolve_repo(&data, &remotes, "https://github.com/acme/demo").as_deref(), Some("demo"));
        assert_eq!(resolve_repo(&data, &remotes, "https://github.com/acme/other"), None);
    }
}
