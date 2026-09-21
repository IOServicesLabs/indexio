//! Sources (SPEC-P7): the one-command onboarding surface.
//!
//! A *source* is something that expands to repositories:
//!
//! | spec | kind |
//! |---|---|
//! | `~/code`, `C:\src` (any path) | every git repo under it, any depth; a folder with no git repo is indexed as a plain tree |
//! | `github:org` / `github:user` | every repo of the org (or user) |
//! | `github:owner/repo`, `https://github.com/owner/repo(.git)` | one repo |
//! | `azdo:org/project`, `https://dev.azure.com/org/project` | every repo of the Azure DevOps project |
//! | `azdo:org/project/repo`, `https://dev.azure.com/org/project/_git/repo`, `https://org.visualstudio.com/project/_git/repo` | one repo |
//! | any other `https://…` / `git@…` | one repo, cloned as-is |
//!
//! Sources are remembered in `<data_dir>/sources.json`; `sync_all` clones or
//! fast-forwards every remote repo, discovers local ones, (re)indexes them
//! all incrementally, and is the only thing a cron job needs to run.
//! Tokens come from the environment (`GITHUB_TOKEN`/`GH_TOKEN`,
//! `AZDO_TOKEN`/`AZURE_DEVOPS_EXT_PAT`/`SYSTEM_ACCESSTOKEN`) and are passed
//! to git as an `http.extraheader` — never on the command line's URL, so
//! they do not appear in process listings or logs.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::org_sync::{is_existing_clone, run_git};
use crate::{Cas, IndexReport};

const USER_AGENT: &str = "indexio";
const PER_PAGE: usize = 100;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// A local folder: every git repo below it (or the folder itself).
    LocalDir,
    /// `github:NAME`: all repos of an organization or user.
    GithubOwner,
    /// `github:owner/repo`.
    GithubRepo,
    /// `azdo:org/project`: all repos of an Azure DevOps project.
    AzdoProject,
    /// `azdo:org/project/repo`.
    AzdoRepo,
    /// Any other clone URL.
    GitUrl,
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    pub kind: SourceKind,
    /// Normalized identity: a path, `owner`, `owner/repo`, `org/project`,
    /// `org/project/repo`, or a URL.
    pub spec: String,
    /// Clone destination for remote sources (default
    /// `<data_dir>/remotes/<host>/<owner>/…`).
    #[serde(default)]
    pub dest: Option<PathBuf>,
    #[serde(default)]
    pub include_forks: bool,
    #[serde(default)]
    pub include_archived: bool,
    #[serde(default = "default_true")]
    pub shallow: bool,
}

impl Source {
    fn new(kind: SourceKind, spec: impl Into<String>) -> Self {
        Source {
            kind,
            spec: spec.into(),
            dest: None,
            include_forks: false,
            include_archived: false,
            shallow: true,
        }
    }

    /// Human label.
    pub fn label(&self) -> String {
        match self.kind {
            SourceKind::LocalDir => self.spec.clone(),
            SourceKind::GithubOwner | SourceKind::GithubRepo => format!("github:{}", self.spec),
            SourceKind::AzdoProject | SourceKind::AzdoRepo => format!("azdo:{}", self.spec),
            SourceKind::GitUrl => self.spec.clone(),
        }
    }
}

fn trim_git_suffix(s: &str) -> &str {
    s.trim_end_matches('/').trim_end_matches(".git")
}

/// Parse a user-typed source spec (see module docs). Local paths must exist.
pub fn parse_source(input: &str) -> anyhow::Result<Source> {
    let raw = input.trim();
    if raw.is_empty() {
        bail!("empty source");
    }
    if let Some(rest) = raw.strip_prefix("github:") {
        let rest = trim_git_suffix(rest.trim_matches('/'));
        return match rest.split('/').filter(|s| !s.is_empty()).collect::<Vec<_>>()[..] {
            [owner] => Ok(Source::new(SourceKind::GithubOwner, owner.to_string())),
            [owner, repo] => Ok(Source::new(SourceKind::GithubRepo, format!("{owner}/{repo}"))),
            _ => bail!("expected github:OWNER or github:OWNER/REPO, got '{raw}'"),
        };
    }
    if let Some(rest) = raw.strip_prefix("azdo:").or_else(|| raw.strip_prefix("azure:")) {
        let rest = trim_git_suffix(rest.trim_matches('/'));
        return match rest.split('/').filter(|s| !s.is_empty()).collect::<Vec<_>>()[..] {
            [org, project] => Ok(Source::new(SourceKind::AzdoProject, format!("{org}/{project}"))),
            [org, project, repo] => Ok(Source::new(
                SourceKind::AzdoRepo,
                format!("{org}/{project}/{repo}"),
            )),
            _ => bail!("expected azdo:ORG/PROJECT or azdo:ORG/PROJECT/REPO, got '{raw}'"),
        };
    }
    if raw.starts_with("http://") || raw.starts_with("https://") {
        let no_scheme = raw.split_once("://").map(|(_, r)| r).unwrap_or(raw);
        let (host, path) = no_scheme.split_once('/').unwrap_or((no_scheme, ""));
        let host = host.rsplit('@').next().unwrap_or(host).to_ascii_lowercase();
        let parts: Vec<&str> = trim_git_suffix(path)
            .split('/')
            .filter(|s| !s.is_empty())
            .collect();
        if host == "github.com" {
            return match parts[..] {
                [owner] => Ok(Source::new(SourceKind::GithubOwner, owner.to_string())),
                [owner, repo, ..] => {
                    Ok(Source::new(SourceKind::GithubRepo, format!("{owner}/{repo}")))
                }
                _ => bail!("cannot parse GitHub URL '{raw}'"),
            };
        }
        if host == "dev.azure.com" {
            return match parts[..] {
                [org, project] => {
                    Ok(Source::new(SourceKind::AzdoProject, format!("{org}/{project}")))
                }
                [org, project, "_git", repo, ..] => Ok(Source::new(
                    SourceKind::AzdoRepo,
                    format!("{org}/{project}/{repo}"),
                )),
                _ => bail!("cannot parse Azure DevOps URL '{raw}'"),
            };
        }
        if let Some(org) = host.strip_suffix(".visualstudio.com") {
            return match parts[..] {
                [project] => Ok(Source::new(SourceKind::AzdoProject, format!("{org}/{project}"))),
                [project, "_git", repo, ..] => Ok(Source::new(
                    SourceKind::AzdoRepo,
                    format!("{org}/{project}/{repo}"),
                )),
                _ => bail!("cannot parse Azure DevOps URL '{raw}'"),
            };
        }
        return Ok(Source::new(SourceKind::GitUrl, raw.to_string()));
    }
    if raw.starts_with("git@") || raw.starts_with("ssh://") {
        return Ok(Source::new(SourceKind::GitUrl, raw.to_string()));
    }
    let path = expand_tilde(raw);
    if !path.is_dir() {
        bail!(
            "'{raw}' is not a folder (and not a github:/azdo:/URL source); \
             folders must exist"
        );
    }
    let canon = fs::canonicalize(&path).unwrap_or(path);
    Ok(Source::new(SourceKind::LocalDir, strip_verbatim(&canon)))
}

fn expand_tilde(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/").or_else(|| p.strip_prefix("~\\")) {
        if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(p)
}

/// Windows `canonicalize` returns `\\?\C:\…`; keep the familiar form.
fn strip_verbatim(p: &Path) -> String {
    let s = p.to_string_lossy();
    s.strip_prefix(r"\\?\").unwrap_or(&s).to_string()
}

// ---------------------------------------------------------------------------
// sources.json
// ---------------------------------------------------------------------------

fn sources_path(data_dir: &Path) -> PathBuf {
    data_dir.join("sources.json")
}

pub fn load_sources(data_dir: &Path) -> anyhow::Result<Vec<Source>> {
    let path = sources_path(data_dir);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

pub fn save_sources(data_dir: &Path, sources: &[Source]) -> anyhow::Result<()> {
    fs::create_dir_all(data_dir)?;
    let path = sources_path(data_dir);
    let tmp = data_dir.join(".sources.json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(sources)?)?;
    fs::rename(&tmp, &path)?;
    Ok(())
}

/// Add (or update the options of) a source. Returns true when it was new.
pub fn add_source(data_dir: &Path, src: Source) -> anyhow::Result<bool> {
    let mut all = load_sources(data_dir)?;
    if let Some(existing) = all.iter_mut().find(|s| s.kind == src.kind && s.spec == src.spec) {
        *existing = src;
        save_sources(data_dir, &all)?;
        return Ok(false);
    }
    all.push(src);
    save_sources(data_dir, &all)?;
    Ok(true)
}

/// Remove a source by its spec (as typed or as normalized). Returns true
/// when something was removed. Indexed repos stay until `indexio sync --prune`.
pub fn remove_source(data_dir: &Path, spec: &str) -> anyhow::Result<bool> {
    let mut all = load_sources(data_dir)?;
    let before = all.len();
    let parsed = parse_source(spec).ok();
    all.retain(|s| {
        !(s.spec == spec
            || s.label() == spec
            || parsed.as_ref().is_some_and(|p| p.kind == s.kind && p.spec == s.spec))
    });
    if all.len() != before {
        save_sources(data_dir, &all)?;
    }
    Ok(all.len() != before)
}

// ---------------------------------------------------------------------------
// Local discovery
// ---------------------------------------------------------------------------

/// Git repositories under `root` at any depth (a repo is a directory with
/// `.git`); the walk does not descend into repos, hidden dirs or
/// [`crate::SKIP_DIRS`]. `root` itself being a repo yields just `[root]`.
pub fn discover_repos(root: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        if dir.join(".git").exists() {
            out.push(dir.to_path_buf());
            return;
        }
        let Ok(rd) = fs::read_dir(dir) else { return };
        let mut subdirs: Vec<PathBuf> = rd
            .flatten()
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .filter(|e| !crate::skip_dir_name(&e.file_name().to_string_lossy()))
            .map(|e| e.path())
            .collect();
        subdirs.sort();
        for d in subdirs {
            walk(&d, out);
        }
    }
    let mut out = Vec::new();
    walk(root, &mut out);
    out
}

// ---------------------------------------------------------------------------
// Remote listings
// ---------------------------------------------------------------------------

/// One remote repository to clone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteRepo {
    /// Name used for the clone directory and (uniquified) as the repo name.
    pub name: String,
    pub clone_url: String,
    pub fork: bool,
    pub archived: bool,
}

#[derive(Deserialize)]
struct GhRepo {
    name: String,
    clone_url: String,
    #[serde(default)]
    fork: bool,
    #[serde(default)]
    archived: bool,
}

fn github_get(url: &str, token: Option<&str>) -> Result<ureq::Response, ureq::Error> {
    let mut req = ureq::get(url).set("User-Agent", USER_AGENT);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    req.call()
}

/// All repos of a GitHub org or user (`/orgs/{x}/repos`, falling back to
/// `/users/{x}/repos` on 404), paginated.
fn list_github_owner(base: &str, owner: &str, token: Option<&str>) -> anyhow::Result<Vec<RemoteRepo>> {
    let mut out = Vec::new();
    for endpoint in ["orgs", "users"] {
        out.clear();
        let mut page = 1usize;
        let mut not_found = false;
        loop {
            let url = format!("{base}/{endpoint}/{owner}/repos?per_page={PER_PAGE}&page={page}&type=all");
            let resp = match github_get(&url, token) {
                Ok(r) => r,
                Err(ureq::Error::Status(404, _)) => {
                    not_found = true;
                    break;
                }
                Err(e) => return Err(anyhow!("GET {url} failed: {e}")),
            };
            let batch: Vec<GhRepo> = resp
                .into_json()
                .with_context(|| format!("decoding repo list from {url}"))?;
            let short = batch.len() < PER_PAGE;
            out.extend(batch.into_iter().map(|r| RemoteRepo {
                name: r.name,
                clone_url: r.clone_url,
                fork: r.fork,
                archived: r.archived,
            }));
            if short {
                break;
            }
            page += 1;
        }
        if !not_found {
            return Ok(out);
        }
    }
    bail!("GitHub owner '{owner}' not found (as an org or a user)")
}

#[derive(Deserialize)]
struct AzdoList {
    value: Vec<AzdoRepoJson>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AzdoRepoJson {
    name: String,
    remote_url: String,
    #[serde(default)]
    is_disabled: bool,
}

fn percent_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn azdo_basic(token: &str) -> String {
    format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD.encode(format!(":{token}"))
    )
}

/// All repos of an Azure DevOps project
/// (`GET {base}/{org}/{project}/_apis/git/repositories?api-version=7.1`).
fn list_azdo_project(
    base: &str,
    org: &str,
    project: &str,
    token: Option<&str>,
) -> anyhow::Result<Vec<RemoteRepo>> {
    let url = format!(
        "{base}/{}/{}/_apis/git/repositories?api-version=7.1",
        percent_encode(org),
        percent_encode(project)
    );
    let mut req = ureq::get(&url).set("User-Agent", USER_AGENT);
    if let Some(t) = token {
        req = req.set("Authorization", &azdo_basic(t));
    }
    let list: AzdoList = req
        .call()
        .map_err(|e| anyhow!("GET {url} failed: {e} (is AZDO_TOKEN set to a PAT with Code:Read?)"))?
        .into_json()
        .with_context(|| format!("decoding repo list from {url}"))?;
    Ok(list
        .value
        .into_iter()
        .map(|r| RemoteRepo {
            name: r.name,
            clone_url: r.remote_url,
            fork: false,
            archived: r.is_disabled,
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Clone with token
// ---------------------------------------------------------------------------

/// Authorization header value for a clone URL, if a token applies.
fn auth_header_for(url: &str, opts: &SyncOptions) -> Option<String> {
    let host = url
        .split_once("://")
        .map(|(_, r)| r)
        .unwrap_or(url)
        .split('/')
        .next()
        .unwrap_or("")
        .rsplit('@')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    if host == "github.com" || host.ends_with(".github.com") {
        return opts.github_token.as_deref().map(|t| {
            format!(
                "Basic {}",
                base64::engine::general_purpose::STANDARD.encode(format!("x-access-token:{t}"))
            )
        });
    }
    if host == "dev.azure.com" || host.ends_with(".visualstudio.com") {
        return opts.azdo_token.as_deref().map(azdo_basic);
    }
    None
}

/// `git -c http.extraheader="AUTHORIZATION: …" clone/pull`. The token never
/// appears in the URL.
pub fn clone_or_pull_auth(
    url: &str,
    dest: &Path,
    shallow: bool,
    auth: Option<&str>,
) -> io::Result<()> {
    let mut cmd = std::process::Command::new("git");
    if let Some(h) = auth {
        cmd.arg("-c").arg(format!("http.extraheader=AUTHORIZATION: {h}"));
    }
    if is_existing_clone(dest) {
        cmd.arg("-C").arg(dest).args(["pull", "--ff-only", "--quiet"]);
    } else {
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        cmd.args(["clone", "--quiet"]);
        if shallow {
            cmd.args(["--depth", "1"]);
        }
        cmd.arg(url).arg(dest);
    }
    run_git(&mut cmd)
}

// ---------------------------------------------------------------------------
// Sync
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SyncOptions {
    pub github_token: Option<String>,
    pub azdo_token: Option<String>,
    pub github_api_base: String,
    pub azdo_api_base: String,
    /// Cap repos per remote source (trial runs).
    pub limit: Option<usize>,
}

impl Default for SyncOptions {
    /// Tokens from `GITHUB_TOKEN`/`GH_TOKEN` and
    /// `AZDO_TOKEN`/`AZURE_DEVOPS_EXT_PAT`/`SYSTEM_ACCESSTOKEN`.
    fn default() -> Self {
        let env = |names: &[&str]| {
            names
                .iter()
                .find_map(|n| std::env::var(n).ok().filter(|v| !v.trim().is_empty()))
        };
        SyncOptions {
            github_token: env(&["GITHUB_TOKEN", "GH_TOKEN"]),
            azdo_token: env(&["AZDO_TOKEN", "AZURE_DEVOPS_EXT_PAT", "SYSTEM_ACCESSTOKEN"]),
            github_api_base: "https://api.github.com".to_string(),
            azdo_api_base: "https://dev.azure.com".to_string(),
            limit: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SyncReport {
    pub source: String,
    /// Repos the source expanded to (before fork/archived filtering).
    pub discovered: u64,
    pub skipped: u64,
    pub indexed: Vec<IndexReport>,
    /// (repo, error) — per-repo failures, never fatal.
    pub failed: Vec<(String, String)>,
}

/// Registered repo name for `path`: an existing registration at this exact
/// path keeps its name; otherwise the basename, disambiguated with the parent
/// directory name and then a counter when another path already owns it.
pub fn repo_name_for(data_dir: &Path, path: &Path) -> String {
    let same_path = |name: &str| {
        crate::repo_state(data_dir, name)
            .map(|s| paths_equal(&s.path, path))
            .unwrap_or(false)
    };
    let registered = |name: &str| crate::repo_state(data_dir, name).is_ok();
    let clean = |s: &str| {
        s.chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '-' })
            .collect::<String>()
    };
    let base = path
        .file_name()
        .map(|s| clean(&s.to_string_lossy()))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "repo".to_string());
    // Reuse an existing registration of this path under any candidate name.
    let parent = path
        .parent()
        .and_then(|p| p.file_name())
        .map(|s| clean(&s.to_string_lossy()))
        .filter(|s| !s.is_empty());
    let mut candidates = vec![base.clone()];
    if let Some(p) = &parent {
        candidates.push(format!("{p}-{base}"));
    }
    for n in 2..100 {
        candidates.push(format!("{base}-{n}"));
    }
    if let Some(c) = candidates.iter().find(|c| same_path(c)) {
        return c.clone();
    }
    candidates
        .into_iter()
        .find(|c| !registered(c))
        .unwrap_or(base)
}

fn paths_equal(a: &Path, b: &Path) -> bool {
    let norm = |p: &Path| {
        fs::canonicalize(p)
            .map(|c| strip_verbatim(&c))
            .unwrap_or_else(|_| p.to_string_lossy().into_owned())
            .to_ascii_lowercase()
    };
    norm(a) == norm(b)
}

/// Index a path under its registered name (delta re-index when already
/// registered at that path).
fn index_or_reindex(path: &Path, data_dir: &Path, cas: &Cas) -> anyhow::Result<IndexReport> {
    let name = repo_name_for(data_dir, path);
    match crate::repo_state(data_dir, &name) {
        Ok(state) if paths_equal(&state.path, path) => crate::reindex_repo(&name, data_dir, cas),
        _ => crate::index_repo(path, &name, data_dir, cas),
    }
}

fn default_dest(data_dir: &Path, src: &Source, repo: &str) -> PathBuf {
    let base = src
        .dest
        .clone()
        .unwrap_or_else(|| data_dir.join("remotes"));
    match src.kind {
        SourceKind::GithubOwner | SourceKind::GithubRepo => {
            let owner = src.spec.split('/').next().unwrap_or("github");
            base.join("github.com").join(owner).join(repo)
        }
        SourceKind::AzdoProject | SourceKind::AzdoRepo => {
            let mut it = src.spec.split('/');
            let org = it.next().unwrap_or("azdo");
            let project = it.next().unwrap_or("project");
            base.join("dev.azure.com").join(org).join(project).join(repo)
        }
        SourceKind::GitUrl => base.join("git").join(repo),
        SourceKind::LocalDir => base.join(repo),
    }
}

/// Repo name implied by a clone URL (`…/owner/repo.git` → `repo`).
fn url_repo_name(url: &str) -> String {
    trim_git_suffix(url)
        .rsplit(['/', ':'])
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("repo")
        .to_string()
}

/// Expand a source into (repo name hint, clone url) pairs for remote kinds.
fn list_remote(src: &Source, opts: &SyncOptions) -> anyhow::Result<Vec<RemoteRepo>> {
    let one = |name: String, clone_url: String| RemoteRepo { name, clone_url, fork: false, archived: false };
    Ok(match src.kind {
        SourceKind::GithubOwner => {
            list_github_owner(&opts.github_api_base, &src.spec, opts.github_token.as_deref())?
        }
        SourceKind::GithubRepo => {
            let (owner, repo) = src.spec.split_once('/').unwrap_or((&src.spec, "repo"));
            let host = opts
                .github_api_base
                .strip_prefix("https://api.")
                .map(|h| format!("https://{h}"))
                .unwrap_or_else(|| "https://github.com".to_string());
            vec![one(repo.to_string(), format!("{host}/{owner}/{repo}.git"))]
        }
        SourceKind::AzdoProject => {
            let (org, project) = src.spec.split_once('/').unwrap_or((&src.spec, ""));
            list_azdo_project(&opts.azdo_api_base, org, project, opts.azdo_token.as_deref())?
        }
        SourceKind::AzdoRepo => {
            let parts: Vec<&str> = src.spec.splitn(3, '/').collect();
            let (org, project, repo) = (parts[0], parts.get(1).copied().unwrap_or(""), parts.get(2).copied().unwrap_or(""));
            vec![one(
                repo.to_string(),
                format!(
                    "{}/{}/{}/_git/{}",
                    opts.azdo_api_base,
                    percent_encode(org),
                    percent_encode(project),
                    percent_encode(repo)
                ),
            )]
        }
        SourceKind::GitUrl => vec![one(url_repo_name(&src.spec), src.spec.clone())],
        SourceKind::LocalDir => Vec::new(),
    })
}

/// Sync one source with an injectable clone step (tests stub it).
pub fn sync_source_with(
    src: &Source,
    data_dir: &Path,
    cas: &Cas,
    opts: &SyncOptions,
    clone_fn: &dyn Fn(&str, &Path, bool, Option<&str>) -> io::Result<()>,
) -> anyhow::Result<SyncReport> {
    let mut report = SyncReport {
        source: src.label(),
        ..Default::default()
    };
    if src.kind == SourceKind::LocalDir {
        let root = PathBuf::from(&src.spec);
        if !root.is_dir() {
            bail!("{}: folder no longer exists", root.display());
        }
        let mut repos = discover_repos(&root);
        if repos.is_empty() {
            repos.push(root.clone()); // plain tree
        }
        report.discovered = repos.len() as u64;
        for r in repos {
            match index_or_reindex(&r, data_dir, cas) {
                Ok(ir) => report.indexed.push(ir),
                Err(e) => {
                    warn!(path = %r.display(), error = %e, "index failed");
                    report.failed.push((r.display().to_string(), format!("{e:#}")));
                }
            }
        }
        return Ok(report);
    }

    let listed = list_remote(src, opts)?;
    report.discovered = listed.len() as u64;
    let mut repos: Vec<RemoteRepo> = listed
        .into_iter()
        .filter(|r| (src.include_forks || !r.fork) && (src.include_archived || !r.archived))
        .collect();
    if let Some(limit) = opts.limit {
        repos.truncate(limit);
    }
    report.skipped = report.discovered - repos.len() as u64;
    for repo in repos {
        let dest = default_dest(data_dir, src, &repo.name);
        let auth = auth_header_for(&repo.clone_url, opts);
        if let Err(e) = clone_fn(&repo.clone_url, &dest, src.shallow, auth.as_deref()) {
            warn!(repo = %repo.name, error = %e, "clone failed");
            report.failed.push((repo.name.clone(), format!("clone: {e}")));
            continue;
        }
        debug!(repo = %repo.name, dest = %dest.display(), "cloned/updated");
        match index_or_reindex(&dest, data_dir, cas) {
            Ok(ir) => report.indexed.push(ir),
            Err(e) => report.failed.push((repo.name.clone(), format!("index: {e:#}"))),
        }
    }
    Ok(report)
}

/// Sync one source (real git).
pub fn sync_source(
    src: &Source,
    data_dir: &Path,
    cas: &Cas,
    opts: &SyncOptions,
) -> anyhow::Result<SyncReport> {
    sync_source_with(src, data_dir, cas, opts, &clone_or_pull_auth)
}

/// Sync every remembered source, then delta re-index any registered repo
/// no source touched (repos added with plain `indexio index`). Source-level
/// failures (a listing that fails) are reported, not fatal.
pub fn sync_all(data_dir: &Path, cas: &Cas, opts: &SyncOptions) -> anyhow::Result<Vec<SyncReport>> {
    let mut reports = Vec::new();
    let mut touched: HashSet<String> = HashSet::new();
    for src in load_sources(data_dir)? {
        match sync_source(&src, data_dir, cas, opts) {
            Ok(r) => {
                touched.extend(r.indexed.iter().map(|i| i.repo.clone()));
                reports.push(r);
            }
            Err(e) => reports.push(SyncReport {
                source: src.label(),
                failed: vec![(src.label(), format!("{e:#}"))],
                ..Default::default()
            }),
        }
    }
    let mut rest = SyncReport {
        source: "(registered repos outside any source)".to_string(),
        ..Default::default()
    };
    for name in registered_repo_names(data_dir)? {
        if touched.contains(&name) {
            continue;
        }
        rest.discovered += 1;
        match crate::reindex_repo(&name, data_dir, cas) {
            Ok(ir) => rest.indexed.push(ir),
            Err(e) => rest.failed.push((name, format!("{e:#}"))),
        }
    }
    if rest.discovered > 0 {
        reports.push(rest);
    }
    Ok(reports)
}

/// Names of registered repos (`<data_dir>/repos/*.json`), sorted.
pub fn registered_repo_names(data_dir: &Path) -> anyhow::Result<Vec<String>> {
    let dir = data_dir.join("repos");
    let mut out = Vec::new();
    if dir.is_dir() {
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if !stem.starts_with('.') {
                        out.push(stem.to_string());
                    }
                }
            }
        }
    }
    out.sort();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead as _, BufReader, Write as _};
    use std::net::TcpListener;

    #[test]
    fn parse_github_and_azdo_and_urls() {
        let p = |s: &str| parse_source(s).unwrap();
        assert_eq!(p("github:acme").kind, SourceKind::GithubOwner);
        assert_eq!(p("github:acme").spec, "acme");
        assert_eq!(p("github:acme/widgets").kind, SourceKind::GithubRepo);
        assert_eq!(p("github:acme/widgets.git").spec, "acme/widgets");
        assert_eq!(p("https://github.com/acme/widgets.git").spec, "acme/widgets");
        assert_eq!(p("https://github.com/acme/widgets/tree/main").spec, "acme/widgets");
        assert_eq!(p("https://github.com/acme").kind, SourceKind::GithubOwner);
        assert_eq!(p("azdo:acme/Platform").kind, SourceKind::AzdoProject);
        assert_eq!(p("azdo:acme/Platform/api").kind, SourceKind::AzdoRepo);
        assert_eq!(p("azure:acme/Platform/api").spec, "acme/Platform/api");
        assert_eq!(p("https://dev.azure.com/acme/Platform").spec, "acme/Platform");
        assert_eq!(p("https://dev.azure.com/acme/Platform/_git/api").spec, "acme/Platform/api");
        assert_eq!(p("https://acme@dev.azure.com/acme/Platform/_git/api").kind, SourceKind::AzdoRepo);
        assert_eq!(p("https://acme.visualstudio.com/Platform/_git/api").spec, "acme/Platform/api");
        assert_eq!(p("https://acme.visualstudio.com/Platform").kind, SourceKind::AzdoProject);
        assert_eq!(p("https://gitlab.example.com/g/r.git").kind, SourceKind::GitUrl);
        assert_eq!(p("git@github.com:acme/widgets.git").kind, SourceKind::GitUrl);
        assert!(parse_source("github:").is_err());
        assert!(parse_source("azdo:onlyorg").is_err());
        assert!(parse_source("").is_err());
        assert!(parse_source("/definitely/not/a/folder/xyz").is_err());
        let tmp = tempfile::tempdir().unwrap();
        let s = p(tmp.path().to_str().unwrap());
        assert_eq!(s.kind, SourceKind::LocalDir);
        assert!(!s.spec.starts_with(r"\\?\"), "{}", s.spec);
        assert_eq!(s.label(), s.spec);
        assert_eq!(p("github:acme").label(), "github:acme");
    }

    #[test]
    fn sources_file_round_trip_add_remove() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        assert!(load_sources(d).unwrap().is_empty());
        assert!(add_source(d, parse_source("github:acme").unwrap()).unwrap());
        assert!(add_source(d, parse_source("azdo:acme/Platform").unwrap()).unwrap());
        // same source again: updated, not duplicated
        let mut again = parse_source("github:acme").unwrap();
        again.include_forks = true;
        assert!(!add_source(d, again).unwrap());
        let all = load_sources(d).unwrap();
        assert_eq!(all.len(), 2);
        assert!(all[0].include_forks);
        assert!(remove_source(d, "github:acme").unwrap());
        assert!(!remove_source(d, "github:acme").unwrap());
        assert_eq!(load_sources(d).unwrap().len(), 1);
        assert!(remove_source(d, "https://dev.azure.com/acme/Platform").unwrap());
        assert!(load_sources(d).unwrap().is_empty());
    }

    #[test]
    fn discover_repos_any_depth_skips_noise() {
        let tmp = tempfile::tempdir().unwrap();
        let r = tmp.path();
        for p in ["a/.git", "b/sub/c/.git", "node_modules/x/.git", ".hidden/y/.git", "plain/src"] {
            fs::create_dir_all(r.join(p)).unwrap();
        }
        // a nested repo inside a repo is not descended into
        fs::create_dir_all(r.join("a/vendor/inner/.git")).unwrap();
        let found: Vec<String> = discover_repos(r)
            .iter()
            .map(|p| p.strip_prefix(r).unwrap().to_string_lossy().replace('\\', "/"))
            .collect();
        assert_eq!(found, vec!["a".to_string(), "b/sub/c".to_string()]);
        // root itself a repo
        assert_eq!(discover_repos(&r.join("a")), vec![r.join("a")]);
        // nothing at all
        assert!(discover_repos(&r.join("plain")).is_empty());
    }

    #[test]
    fn repo_names_are_unique_and_stable() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path().join("data");
        let cas = Cas::open(&d.join("cas")).unwrap();
        let t1 = tmp.path().join("teamA/api");
        let t2 = tmp.path().join("teamB/api");
        for t in [&t1, &t2] {
            fs::create_dir_all(t.join("src")).unwrap();
            fs::write(t.join("src/main.py"), b"def main():\n    pass\n").unwrap();
        }
        assert_eq!(repo_name_for(&d, &t1), "api");
        crate::index_dir(&t1, "api", &d, &cas).unwrap();
        assert_eq!(repo_name_for(&d, &t1), "api", "same path keeps its name");
        assert_eq!(repo_name_for(&d, &t2), "teamB-api", "other path is disambiguated");
        crate::index_dir(&t2, "teamB-api", &d, &cas).unwrap();
        assert_eq!(repo_name_for(&d, &t2), "teamB-api");
        assert_eq!(url_repo_name("https://github.com/acme/widgets.git"), "widgets");
        assert_eq!(url_repo_name("git@github.com:acme/widgets.git"), "widgets");
    }

    #[test]
    fn local_source_indexes_plain_tree_and_reindexes_delta() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path().join("data");
        let cas = Cas::open(&d.join("cas")).unwrap();
        let root = tmp.path().join("code");
        fs::create_dir_all(root.join("svc/src")).unwrap();
        fs::create_dir_all(root.join("svc/node_modules/dep")).unwrap();
        fs::write(root.join("svc/src/app.py"), b"def handler():\n    return 1\n").unwrap();
        fs::write(root.join("svc/src/util.ts"), b"export const x = 1;\n").unwrap();
        fs::write(root.join("svc/node_modules/dep/index.js"), b"module.exports = 1;\n").unwrap();
        fs::write(root.join("svc/README.md"), b"# nope\n").unwrap(); // text: indexed
        fs::write(root.join("svc/logo.png"), b"PNG\n").unwrap(); // unknown: skipped
        let src = parse_source(root.to_str().unwrap()).unwrap();
        let opts = SyncOptions { github_token: None, azdo_token: None, ..Default::default() };
        let r = sync_source_with(&src, &d, &cas, &opts, &|_, _, _, _| Ok(())).unwrap();
        assert_eq!(r.discovered, 1, "no git repos -> the folder itself, once");
        assert_eq!(r.indexed[0].repo, "code");
        assert_eq!(r.indexed[0].docs_added, 3, "py + ts + md; png unknown; node_modules skipped");
        assert!(r.failed.is_empty(), "{:?}", r.failed);
        // delta: one file changed, one added, one removed
        fs::write(root.join("svc/src/app.py"), b"def handler():\n    return 2\n").unwrap();
        fs::write(root.join("svc/src/new.go"), b"package svc\n").unwrap();
        fs::remove_file(root.join("svc/src/util.ts")).unwrap();
        let r = sync_source_with(&src, &d, &cas, &opts, &|_, _, _, _| Ok(())).unwrap();
        let ir = &r.indexed[0];
        assert_eq!((ir.docs_added, ir.docs_deleted, ir.docs_unchanged), (2, 2, 1), "{ir:?}"); // README.md unchanged
        // no-op
        let r = sync_source_with(&src, &d, &cas, &opts, &|_, _, _, _| Ok(())).unwrap();
        let ir = &r.indexed[0];
        assert_eq!((ir.docs_added, ir.docs_deleted, ir.docs_unchanged), (0, 0, 3), "{ir:?}");
        // sync_all covers it too and re-indexes repos registered outside sources
        add_source(&d, src.clone()).unwrap();
        let reps = sync_all(&d, &cas, &opts).unwrap();
        assert_eq!(reps.len(), 1);
        assert_eq!(reps[0].indexed[0].docs_unchanged, 3);
    }

    /// Minimal one-shot JSON server; returns (base url, captured request
    /// line + headers).
    fn serve_json(body: String) -> (String, std::sync::mpsc::Receiver<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut lines = Vec::new();
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let t = line.trim_end().to_string();
                if t.is_empty() {
                    break;
                }
                lines.push(t);
            }
            tx.send(lines).unwrap();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(resp.as_bytes()).unwrap();
        });
        (format!("http://127.0.0.1:{port}"), rx)
    }

    #[test]
    fn azdo_project_listing_auth_and_clone_dest() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path().join("data");
        let cas = Cas::open(&d.join("cas")).unwrap();
        let body = r#"{"value":[
            {"name":"api","remoteUrl":"https://acme@dev.azure.com/acme/Platform/_git/api","isDisabled":false},
            {"name":"old","remoteUrl":"https://acme@dev.azure.com/acme/Platform/_git/old","isDisabled":true}
        ]}"#;
        let (base, rx) = serve_json(body.to_string());
        let opts = SyncOptions {
            github_token: None,
            azdo_token: Some("pat123".into()),
            azdo_api_base: base,
            ..Default::default()
        };
        let src = parse_source("azdo:acme/My Platform").unwrap();
        let cloned = std::sync::Mutex::new(Vec::new());
        let r = sync_source_with(&src, &d, &cas, &opts, &|url, dest, shallow, auth| {
            // stub clone: materialize a plain tree so indexing has something
            fs::create_dir_all(dest.join("src")).unwrap();
            fs::write(dest.join("src/a.py"), b"def a():\n    pass\n").unwrap();
            cloned.lock().unwrap().push((url.to_string(), dest.to_path_buf(), shallow, auth.map(str::to_string)));
            Ok(())
        })
        .unwrap();
        let req = rx.recv().unwrap();
        assert!(req[0].contains("/acme/My%20Platform/_apis/git/repositories?api-version=7.1"), "{req:?}");
        let expect = azdo_basic("pat123");
        assert!(req.iter().any(|h| h == &format!("Authorization: {expect}")), "{req:?}");
        assert_eq!(r.discovered, 2);
        assert_eq!(r.skipped, 1, "disabled repo skipped");
        let c = cloned.lock().unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].0, "https://acme@dev.azure.com/acme/Platform/_git/api");
        assert!(c[0].1.ends_with(Path::new("remotes/dev.azure.com/acme/My Platform/api")), "{:?}", c[0].1);
        assert!(c[0].2, "shallow by default");
        assert_eq!(c[0].3.as_deref(), Some(expect.as_str()), "PAT goes to git as a header");
        assert_eq!(r.indexed.len(), 1);
        assert_eq!(r.indexed[0].repo, "api");
    }

    #[test]
    fn github_owner_listing_falls_back_to_users() {
        // First server answers /orgs with 404 -> code retries /users.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let bodies = [
                (404, "{\"message\":\"Not Found\"}".to_string()),
                (200, r#"[{"name":"w","clone_url":"https://github.com/me/w.git","fork":false,"archived":false},{"name":"f","clone_url":"https://github.com/me/f.git","fork":true,"archived":false}]"#.to_string()),
            ];
            for (status, body) in bodies {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line.trim_end().is_empty() {
                        break;
                    }
                }
                let resp = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(resp.as_bytes()).unwrap();
            }
        });
        let repos = list_github_owner(&format!("http://127.0.0.1:{port}"), "me", None).unwrap();
        assert_eq!(repos.len(), 2);
        assert!(repos[1].fork);
        // auth header for github clones uses the token, never the URL
        let opts = SyncOptions {
            github_token: Some("ghp_x".into()),
            azdo_token: None,
            ..Default::default()
        };
        assert!(auth_header_for("https://github.com/me/w.git", &opts).unwrap().starts_with("Basic "));
        assert!(auth_header_for("https://gitlab.com/me/w.git", &opts).is_none());
    }

    #[test]
    fn git_url_source_names_and_dest() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path().join("data");
        let src = parse_source("https://gitlab.example.com/grp/tool.git").unwrap();
        assert!(default_dest(&d, &src, "tool").ends_with(Path::new("remotes/git/tool")));
        let src = parse_source("github:acme/widgets").unwrap();
        assert!(default_dest(&d, &src, "widgets").ends_with(Path::new("remotes/github.com/acme/widgets")));
        let mut src = parse_source("github:acme").unwrap();
        src.dest = Some(PathBuf::from("/srv/repos"));
        assert_eq!(default_dest(&d, &src, "w"), PathBuf::from("/srv/repos/github.com/acme/w"));
    }
}
