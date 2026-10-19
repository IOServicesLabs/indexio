//! Filesystem watch on the session's repo (SPEC-P9 auto-refresh).
//!
//! The MCP server starts one recursive watcher on the current repo's
//! folder. Events for source files (known language, outside `.git`, build
//! output and hidden dirs) only set a dirty flag; the next tool call then
//! runs a working-tree delta re-index before answering, so an agent's own
//! edits are searchable without it ever calling `refresh_index`. Idle
//! sessions pay nothing: no polling, no stat storms.

use std::collections::HashSet;
use std::path::{Component, Path};
use std::sync::{Arc, Mutex};

use notify::{RecommendedWatcher, RecursiveMode, Watcher};

use indexio_types::Lang;

/// Paths remembered per call before the watch gives up on naming them and
/// asks for a full pass (a checkout or a formatter run touches hundreds).
const MAX_TRACKED: usize = 256;

/// What changed under the repo since the last call: the repo-relative
/// paths (forward slashes), or `full` when they are not all known (an
/// overflow, too many paths, a `.gitignore` edit that changes what is
/// indexed at all).
#[derive(Debug, Default)]
pub struct Changes {
    pub paths: HashSet<String>,
    pub full: bool,
}

pub struct RepoWatch {
    changes: Arc<Mutex<Option<Changes>>>,
    // Dropping the watcher stops it; keep it alive with the server.
    _watcher: RecommendedWatcher,
}

impl RepoWatch {
    pub fn start(root: &Path) -> anyhow::Result<Self> {
        let changes: Arc<Mutex<Option<Changes>>> = Arc::new(Mutex::new(None));
        let slot = Arc::clone(&changes);
        let base = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        let watch_root = base.clone();
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            let mut c = slot.lock().unwrap_or_else(|e| e.into_inner());
            match res {
                Ok(ev) => {
                    // judge only the part below the repo root: the root
                    // itself may live under a dotted or "build" folder
                    for p in &ev.paths {
                        let rel = p.strip_prefix(&base).unwrap_or(p);
                        if is_ignore_file(rel) {
                            c.get_or_insert_with(Changes::default).full = true;
                        } else if interesting(rel) {
                            let ch = c.get_or_insert_with(Changes::default);
                            if !ch.full {
                                ch.paths.insert(rel_string(rel));
                                if ch.paths.len() > MAX_TRACKED {
                                    ch.full = true;
                                    ch.paths.clear();
                                }
                            }
                        }
                    }
                }
                // An overflow means events were lost: assume anything changed.
                Err(_) => c.get_or_insert_with(Changes::default).full = true,
            }
        })?;
        watcher.watch(&watch_root, RecursiveMode::Recursive)?;
        Ok(RepoWatch {
            changes,
            _watcher: watcher,
        })
    }

    /// What changed since the last call (and reset); `None` = nothing.
    pub fn take_changes(&self) -> Option<Changes> {
        self.changes.lock().unwrap_or_else(|e| e.into_inner()).take()
    }

    /// Whether anything changed since the last call (and reset).
    #[cfg(test)]
    pub fn take_dirty(&self) -> bool {
        self.take_changes().is_some()
    }
}

/// `a/b/c.rs` from a relative path, whatever the platform separator.
fn rel_string(rel: &Path) -> String {
    rel.components()
        .filter_map(|c| match c {
            Component::Normal(n) => Some(n.to_string_lossy()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

/// A `.gitignore` outside skipped dirs: editing one changes which files the
/// working tree lists, which only a full pass sees.
fn is_ignore_file(p: &Path) -> bool {
    p.file_name().is_some_and(|n| n == ".gitignore")
        && !p.components().any(|c| match c {
            Component::Normal(name) => indexio_ingest::skip_dir_name(&name.to_string_lossy()) && !p.ends_with(name),
            _ => false,
        })
}

/// A path the index would care about: a file of a known language outside
/// `.git`, hidden directories and the build/dependency dirs the indexer
/// skips. Editor scratch files (`.swp`, `~`, `.tmp`) fail the language
/// test and never trigger a refresh.
fn interesting(p: &Path) -> bool {
    let noisy_dir = p.components().any(|c| match c {
        Component::Normal(name) => {
            let n = name.to_string_lossy();
            indexio_ingest::skip_dir_name(&n) && !p.ends_with(name)
        }
        _ => false,
    });
    if noisy_dir {
        return false;
    }
    let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    if name.starts_with('.') {
        return false;
    }
    if Lang::from_path(name) == Lang::Unknown {
        return false;
    }
    // a build writing thousands of dep-info files into a custom target dir
    // is not an edit (SPEC-P10 §24); one stat per ancestor, events are rare
    // outside builds and builds are exactly what this filters
    !p.ancestors().skip(1).any(indexio_ingest::is_cache_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interesting_filters_noise() {
        assert!(interesting(Path::new("C:/r/src/main.rs")));
        assert!(interesting(Path::new("/r/app/x.py")));
        assert!(!interesting(Path::new("/r/.git/index")));
        assert!(!interesting(Path::new("/r/target/debug/x.rs")));
        assert!(!interesting(Path::new("/r/node_modules/a/b.js")));
        assert!(!interesting(Path::new("/r/src/.main.rs.swp")));
        assert!(!interesting(Path::new("/r/src/main.rs~")));
        assert!(interesting(Path::new("/r/README")), "text files are indexed too (SPEC-P9)");
        assert!(!interesting(Path::new("/r/logo.png")));
        assert!(!interesting(Path::new("/r/package-lock.json")));
    }

    #[test]
    fn watch_sets_dirty_on_source_edit() {
        // tempdir names start with ".tmp": the relative-path check must
        // not treat the root itself as a hidden directory
        let tmp = tempfile::tempdir().unwrap();
        let w = RepoWatch::start(tmp.path()).unwrap();
        assert!(!w.take_dirty());
        std::fs::write(tmp.path().join("a.rs"), b"fn a() {}\n").unwrap();
        let mut seen = false;
        for _ in 0..50 {
            std::thread::sleep(std::time::Duration::from_millis(20));
            if w.take_dirty() {
                seen = true;
                break;
            }
        }
        assert!(seen, "edit was not noticed (root {})", tmp.path().display());
        std::fs::write(tmp.path().join("photo.png"), b"x").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(!w.take_dirty(), "unknown-language file must not dirty");
    }

    fn wait_changes(w: &RepoWatch) -> Changes {
        for _ in 0..50 {
            std::thread::sleep(std::time::Duration::from_millis(20));
            if let Some(c) = w.take_changes() {
                // let the rest of a burst land
                std::thread::sleep(std::time::Duration::from_millis(100));
                let mut c = c;
                if let Some(more) = w.take_changes() {
                    c.full |= more.full;
                    c.paths.extend(more.paths);
                }
                return c;
            }
        }
        panic!("no change noticed");
    }

    #[test]
    fn watch_names_the_changed_paths() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src/deep")).unwrap();
        let w = RepoWatch::start(tmp.path()).unwrap();
        std::fs::write(tmp.path().join("src/deep/a.rs"), b"fn a() {}\n").unwrap();
        let c = wait_changes(&w);
        assert!(!c.full);
        assert!(c.paths.contains("src/deep/a.rs"), "{c:?}");
        std::fs::write(tmp.path().join(".gitignore"), b"out/\n").unwrap();
        assert!(wait_changes(&w).full, ".gitignore edits need a full pass");
    }
}
