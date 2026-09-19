//! GitHub organization ingestion (SPEC-P2 §3).
//!
//! `sync_org` lists the repos of a GitHub org (paginated REST API), clones or
//! fast-forward-updates each one into `dest_dir`, then indexes every repo with
//! [`crate::index_repo`]. Per-repo failures are recorded in the report and are
//! not fatal.
//!
//! The HTTP listing and the git clone/pull step are factored behind injectable
//! function pointers (`sync_with`), so tests run against a local mock server
//! and a stubbed clone — they never touch github.com or run real clones.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};
use tracing::{debug, warn};

use crate::{Cas, IndexReport};

/// Default GitHub REST API base.
const GITHUB_API_BASE: &str = "https://api.github.com";
/// Results per page; a page shorter than this terminates pagination.
const PER_PAGE: usize = 100;
/// Mandatory User-Agent (GitHub rejects requests without one).
const USER_AGENT: &str = "indexio";

#[derive(Clone, Debug)]
pub struct OrgSyncOptions {
    pub org: String,
    /// GitHub PAT; None = unauthenticated (60 req/h).
    pub token: Option<String>,
    /// Repos are cloned here as `dest_dir/<repo>`.
    pub dest_dir: PathBuf,
    /// Include forked repos (default false).
    pub include_forks: bool,
    /// Include archived repos (default false).
    pub include_archived: bool,
    /// Cap repo count (for trials).
    pub limit: Option<usize>,
    /// Clone with `--depth 1` (default true).
    pub shallow: bool,
}

impl Default for OrgSyncOptions {
    fn default() -> Self {
        OrgSyncOptions {
            org: String::new(),
            token: None,
            dest_dir: PathBuf::new(),
            include_forks: false,
            include_archived: false,
            limit: None,
            shallow: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct OrgSyncReport {
    pub repos_listed: u64,
    pub repos_cloned: u64,
    pub repos_skipped: u64,
    /// (repo name, error message) — per-repo failures, never fatal.
    pub repos_failed: Vec<(String, String)>,
    pub indexed: Vec<IndexReport>,
}

/// One repo entry of the `/orgs/{org}/repos` API response (subset of fields).
#[derive(Clone, Debug, serde::Deserialize)]
struct OrgRepo {
    name: String,
    clone_url: String,
    #[serde(default)]
    fork: bool,
    #[serde(default)]
    archived: bool,
}

/// Fetch one page of the org repo listing.
fn fetch_page(base: &str, opts: &OrgSyncOptions, page: usize) -> anyhow::Result<Vec<OrgRepo>> {
    let url = format!(
        "{base}/orgs/{}/repos?per_page={PER_PAGE}&page={page}",
        opts.org
    );
    let mut req = ureq::get(&url).set("User-Agent", USER_AGENT);
    if let Some(token) = &opts.token {
        req = req.set("Authorization", &format!("Bearer {token}"));
    }
    let repos = req
        .call()
        .map_err(|e| anyhow!("GET {url} failed: {e}"))?
        .into_json::<Vec<OrgRepo>>()
        .with_context(|| format!("decoding repo list from {url}"))?;
    Ok(repos)
}

/// List all repos of the org: follow pages until a short (< PER_PAGE) page.
/// Fork/archived filtering and `limit` are applied by the caller
/// ([`apply_filters`]).
fn list_org_repos(opts: &OrgSyncOptions, base: &str) -> anyhow::Result<Vec<OrgRepo>> {
    let mut out = Vec::new();
    let mut page = 1usize;
    loop {
        let batch = fetch_page(base, opts, page)?;
        let short = batch.len() < PER_PAGE;
        debug!(org = %opts.org, page, got = batch.len(), "org repos page");
        out.extend(batch);
        if short {
            break;
        }
        page += 1;
    }
    Ok(out)
}

/// Apply fork/archived filters and the `limit` cap. Returns the kept repos.
fn apply_filters(repos: Vec<OrgRepo>, opts: &OrgSyncOptions) -> Vec<OrgRepo> {
    let mut kept: Vec<OrgRepo> = repos
        .into_iter()
        .filter(|r| (opts.include_forks || !r.fork) && (opts.include_archived || !r.archived))
        .collect();
    if let Some(limit) = opts.limit {
        kept.truncate(limit);
    }
    kept
}

/// Whether `dest` is an existing clone that should be updated instead of
/// re-cloned.
pub(crate) fn is_existing_clone(dest: &Path) -> bool {
    dest.join(".git").exists()
}

/// Run a git command; non-zero exit status is an io error carrying stderr.
pub(crate) fn run_git(cmd: &mut std::process::Command) -> io::Result<()> {
    let out = cmd
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .map_err(|e| io::Error::new(e.kind(), format!("spawn git: {e}")))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "git exited with {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        )))
    }
}

/// Production clone: `git -C dest pull --ff-only` for an existing clone, else
/// `git clone [--depth 1] <url> <dest>`.
fn clone_or_pull(url: &str, dest: &Path, shallow: bool) -> io::Result<()> {
    if is_existing_clone(dest) {
        let mut cmd = std::process::Command::new("git");
        cmd.arg("-C").arg(dest).args(["pull", "--ff-only"]);
        run_git(&mut cmd)
    } else {
        let mut cmd = std::process::Command::new("git");
        cmd.arg("clone");
        if shallow {
            cmd.args(["--depth", "1"]);
        }
        cmd.arg(url).arg(dest);
        run_git(&mut cmd)
    }
}

/// Injectable pipeline. `list_fn` returns the raw (unfiltered) repo listing;
/// `clone_fn` clones or updates one repo into the given destination;
/// `index_fn` indexes one repo directory under the given name.
fn sync_with(
    opts: &OrgSyncOptions,
    list_fn: &dyn Fn(&OrgSyncOptions) -> anyhow::Result<Vec<OrgRepo>>,
    clone_fn: &dyn Fn(&str, &Path) -> io::Result<()>,
    index_fn: &dyn Fn(&Path, &str) -> anyhow::Result<IndexReport>,
) -> anyhow::Result<OrgSyncReport> {
    // A listing failure is fatal: nothing sensible to do without the repo set.
    let listed = list_fn(opts)?;
    let mut report = OrgSyncReport {
        repos_listed: listed.len() as u64,
        ..Default::default()
    };
    let repos = apply_filters(listed, opts);
    report.repos_skipped = report.repos_listed - repos.len() as u64;

    fs::create_dir_all(&opts.dest_dir)?;
    for repo in repos {
        let dest = opts.dest_dir.join(&repo.name);
        if let Err(e) = clone_fn(&repo.clone_url, &dest) {
            warn!(repo = %repo.name, error = %e, "clone failed");
            report
                .repos_failed
                .push((repo.name.clone(), format!("clone: {e}")));
            continue;
        }
        report.repos_cloned += 1;
        match index_fn(&dest, &repo.name) {
            Ok(r) => report.indexed.push(r),
            Err(e) => {
                warn!(repo = %repo.name, error = %e, "index failed");
                report
                    .repos_failed
                    .push((repo.name.clone(), format!("index: {e}")));
            }
        }
    }
    Ok(report)
}

/// Sync a GitHub org into `opts.dest_dir` and index every repo.
///
/// 1. GET `https://api.github.com/orgs/{org}/repos?per_page=100&page=N` via
///    ureq (`Authorization: Bearer` when `opts.token` is set; mandatory
///    `User-Agent: indexio`), following pages until a short page.
///    Forks/archived are filtered per options; `limit` caps the count.
/// 2. Each repo: `git -C <dest> pull --ff-only` when `<dest>/.git` exists,
///    else `git clone [--depth 1] <clone_url> <dest>`. Failures are recorded
///    in the report, not fatal.
/// 3. Every cloned/updated repo is indexed with [`crate::index_repo`]
///    (name = repo name).
pub fn sync_org(opts: &OrgSyncOptions, data_dir: &Path, cas: &Cas) -> anyhow::Result<OrgSyncReport> {
    let shallow = opts.shallow;
    sync_with(
        opts,
        &|o| list_org_repos(o, GITHUB_API_BASE),
        &|url, dest| clone_or_pull(url, dest, shallow),
        &|path, name| crate::index_repo(path, name, data_dir, cas),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    fn opts(org: &str, dest: &Path) -> OrgSyncOptions {
        OrgSyncOptions {
            org: org.to_string(),
            dest_dir: dest.to_path_buf(),
            ..Default::default()
        }
    }

    fn repo_json(name: &str, fork: bool, archived: bool) -> String {
        format!(
            r#"{{"name":"{name}","clone_url":"https://example.com/{name}.git","fork":{fork},"archived":{archived}}}"#
        )
    }

    fn repo(name: &str) -> OrgRepo {
        OrgRepo {
            name: name.to_string(),
            clone_url: format!("https://example.com/{name}.git"),
            fork: false,
            archived: false,
        }
    }

    // -----------------------------------------------------------------------
    // mock GitHub API server (std::net::TcpListener, two paginated pages)
    // -----------------------------------------------------------------------

    #[derive(Debug)]
    struct CapturedRequest {
        request_line: String,
        headers: Vec<(String, String)>,
    }

    struct MockServer {
        base: String,
        requests: Arc<Mutex<Vec<CapturedRequest>>>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    /// Serve `page1` (exactly PER_PAGE entries) then `page2` (short page).
    fn start_mock(page1: String, page2: String) -> MockServer {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let requests: Arc<Mutex<Vec<CapturedRequest>>> = Arc::new(Mutex::new(Vec::new()));
        let reqs = Arc::clone(&requests);
        let handle = std::thread::spawn(move || {
            let bodies = [page1, page2];
            let mut served = 0usize;
            for stream in listener.incoming() {
                let mut stream = match stream {
                    Ok(s) => s,
                    Err(_) => break,
                };
                if served >= bodies.len() {
                    break;
                }
                // Read the request headers (GET: no body).
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                let mut headers = Vec::new();
                let mut line = String::new();
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some((k, v)) = trimmed.split_once(':') {
                        headers.push((k.trim().to_string(), v.trim().to_string()));
                    }
                }
                reqs.lock().unwrap().push(CapturedRequest {
                    request_line: request_line.trim_end().to_string(),
                    headers,
                });
                let body = bodies[served].clone();
                served += 1;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(resp.as_bytes()).unwrap();
                let _ = stream.flush();
                if served >= bodies.len() {
                    break;
                }
            }
        });
        MockServer {
            base: format!("http://127.0.0.1:{port}"),
            requests,
            handle: Some(handle),
        }
    }

    #[test]
    fn pagination_fetches_and_merges_all_pages() {
        // Page 1: exactly PER_PAGE entries -> pagination must continue.
        let page1 = format!(
            "[{}]",
            (0..PER_PAGE)
                .map(|i| repo_json(&format!("repo-{i:03}"), false, false))
                .collect::<Vec<_>>()
                .join(",")
        );
        // Page 2: short page (2 entries) -> pagination stops here.
        let page2 = format!(
            "[{},{}]",
            repo_json("extra-1", false, false),
            repo_json("extra-2", false, false)
        );
        let server = start_mock(page1, page2);

        let mut o = opts("myorg", Path::new("/unused"));
        o.token = Some("secret-token".to_string());
        let repos = list_org_repos(&o, &server.base).unwrap();

        // Both pages fetched and merged: 100 + 2.
        assert_eq!(repos.len(), PER_PAGE + 2);
        assert_eq!(repos[0].name, "repo-000");
        assert_eq!(repos[PER_PAGE - 1].name, format!("repo-{:03}", PER_PAGE - 1));
        assert_eq!(repos[PER_PAGE].name, "extra-1");
        assert_eq!(repos[PER_PAGE + 1].name, "extra-2");
        assert_eq!(repos[7].clone_url, "https://example.com/repo-007.git");

        // Exactly two requests, correct URLs, mandatory headers.
        let reqs = server.requests.lock().unwrap();
        assert_eq!(reqs.len(), 2, "{reqs:?}");
        assert_eq!(
            reqs[0].request_line,
            format!("GET /orgs/myorg/repos?per_page={PER_PAGE}&page=1 HTTP/1.1")
        );
        assert_eq!(
            reqs[1].request_line,
            format!("GET /orgs/myorg/repos?per_page={PER_PAGE}&page=2 HTTP/1.1")
        );
        for r in reqs.iter() {
            let has = |k: &str, v: &str| {
                r.headers
                    .iter()
                    .any(|(hk, hv)| hk.eq_ignore_ascii_case(k) && hv == v)
            };
            assert!(has("user-agent", "indexio"), "{:?}", r.headers);
            assert!(has("authorization", "Bearer secret-token"), "{:?}", r.headers);
        }
        drop(reqs);
        server.handle.unwrap().join().unwrap();
    }

    #[test]
    fn filters_exclude_forks_and_archived_and_apply_limit() {
        let mut plain = repo("plain");
        plain.clone_url = "u-plain".to_string();
        let mut fork = repo("forked");
        fork.fork = true;
        let mut arch = repo("archived");
        arch.archived = true;
        let mut both = repo("fork-archived");
        both.fork = true;
        both.archived = true;
        let all = vec![plain, fork, arch, both];

        let dest = Path::new("/unused");
        let o = opts("org", dest);
        let kept = apply_filters(all.clone(), &o);
        assert_eq!(
            kept.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
            vec!["plain"]
        );

        let o = OrgSyncOptions {
            include_forks: true,
            ..opts("org", dest)
        };
        let kept = apply_filters(all.clone(), &o);
        assert_eq!(
            kept.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
            vec!["plain", "forked"]
        );

        let o = OrgSyncOptions {
            include_forks: true,
            include_archived: true,
            ..opts("org", dest)
        };
        assert_eq!(apply_filters(all.clone(), &o).len(), 4);

        // limit caps the kept set (forks/archived excluded first).
        let o = OrgSyncOptions {
            limit: Some(0),
            ..opts("org", dest)
        };
        assert!(apply_filters(all.clone(), &o).is_empty());
    }

    // -----------------------------------------------------------------------
    // sync_with: stubbed list + clone + index (no network, no real git)
    // -----------------------------------------------------------------------

    fn index_report(name: &str) -> IndexReport {
        IndexReport {
            repo: name.to_string(),
            docs_added: 1,
            ..Default::default()
        }
    }

    #[test]
    fn sync_skips_existing_via_pull_stub_and_counts_index_calls() {
        let td = tempfile::tempdir().unwrap();
        let dest_dir = td.path().join("repos");

        // Pre-created fake clone: repo "old" must take the pull path.
        let existing = dest_dir.join("old");
        fs::create_dir_all(existing.join(".git")).unwrap();

        let o = OrgSyncOptions {
            include_forks: false,
            ..opts("myorg", &dest_dir)
        };
        let listed: Vec<OrgRepo> = vec![
            repo("old"),
            repo("fresh"),
            {
                let mut r = repo("afork");
                r.fork = true;
                r
            },
        ];

        // Stub clone: emulate the production decision (pull iff .git exists)
        // and record every call.
        let calls: Arc<Mutex<Vec<(String, PathBuf, bool)>>> = Arc::new(Mutex::new(Vec::new()));
        let calls2 = Arc::clone(&calls);
        let clone_fn = move |url: &str, dest: &Path| -> io::Result<()> {
            let pull = is_existing_clone(dest);
            if !pull {
                fs::create_dir_all(dest.join(".git"))?;
            }
            calls2
                .lock()
                .unwrap()
                .push((url.to_string(), dest.to_path_buf(), pull));
            Ok(())
        };

        let index_count = Arc::new(AtomicUsize::new(0));
        let index_count2 = Arc::clone(&index_count);
        let index_fn = move |_path: &Path, name: &str| -> anyhow::Result<IndexReport> {
            index_count2.fetch_add(1, Ordering::SeqCst);
            Ok(index_report(name))
        };

        let report = sync_with(&o, &|_| Ok(listed.clone()), &clone_fn, &index_fn).unwrap();

        assert_eq!(report.repos_listed, 3);
        assert_eq!(report.repos_skipped, 1, "the fork is filtered out");
        assert_eq!(report.repos_cloned, 2);
        assert!(report.repos_failed.is_empty());
        assert_eq!(index_count.load(Ordering::SeqCst), 2);
        assert_eq!(
            report.indexed.iter().map(|r| r.repo.as_str()).collect::<Vec<_>>(),
            vec!["old", "fresh"]
        );

        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        // existing clone -> pull path (no re-clone); new repo -> clone path.
        assert_eq!(calls[0].0, "https://example.com/old.git");
        assert_eq!(calls[0].1, existing);
        assert!(calls[0].2, "repo with pre-created .git must take the pull path");
        assert_eq!(calls[1].0, "https://example.com/fresh.git");
        assert_eq!(calls[1].1, dest_dir.join("fresh"));
        assert!(!calls[1].2, "new repo must take the clone path");
    }

    #[test]
    fn sync_records_per_repo_failures_without_aborting() {
        let td = tempfile::tempdir().unwrap();
        let dest_dir = td.path().join("repos");
        let o = opts("myorg", &dest_dir);
        let listed: Vec<OrgRepo> = vec![repo("good"), repo("bad-clone"), repo("bad-index"), repo("last")];

        let clone_fn = |url: &str, dest: &Path| -> io::Result<()> {
            if url.contains("bad-clone") {
                return Err(io::Error::other("simulated clone failure"));
            }
            fs::create_dir_all(dest.join(".git"))
        };
        let index_fn = |_path: &Path, name: &str| -> anyhow::Result<IndexReport> {
            if name == "bad-index" {
                anyhow::bail!("simulated index failure");
            }
            Ok(index_report(name))
        };

        let report = sync_with(&o, &|_| Ok(listed.clone()), &clone_fn, &index_fn).unwrap();

        assert_eq!(report.repos_listed, 4);
        assert_eq!(report.repos_skipped, 0);
        // bad-clone never cloned; bad-index cloned but failed at index time.
        assert_eq!(report.repos_cloned, 3);
        assert_eq!(
            report
                .repos_failed
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["bad-clone", "bad-index"]
        );
        assert!(report.repos_failed[0].1.contains("simulated clone failure"));
        assert!(report.repos_failed[1].1.contains("simulated index failure"));
        // Repos after a failure are still processed.
        assert_eq!(
            report.indexed.iter().map(|r| r.repo.as_str()).collect::<Vec<_>>(),
            vec!["good", "last"]
        );
    }

    #[test]
    fn production_clone_decision_prefers_pull_for_existing_dot_git() {
        // The decision behind the production clone_fn (pull vs clone) is pure
        // path inspection — verify it without running git.
        let td = tempfile::tempdir().unwrap();
        let dest = td.path().join("r");
        assert!(!is_existing_clone(&dest));
        fs::create_dir_all(dest.join(".git")).unwrap();
        assert!(is_existing_clone(&dest));
    }

    #[test]
    fn listing_failure_is_fatal() {
        let td = tempfile::tempdir().unwrap();
        let o = opts("myorg", &td.path().join("repos"));
        let err = sync_with(
            &o,
            &|_| Err(anyhow!("boom")),
            &|_, _| Ok(()),
            &|_, _| Ok(index_report("x")),
        )
        .unwrap_err();
        assert!(err.to_string().contains("boom"));
    }
}
