//! Filesystem watch on the session's repo (SPEC-P9 auto-refresh).
//!
//! The MCP server starts one recursive watcher on the current repo's
//! folder. Events for source files (known language, outside `.git`, build
//! output and hidden dirs) only set a dirty flag; the next tool call then
//! runs a working-tree delta re-index before answering, so an agent's own
//! edits are searchable without it ever calling `refresh_index`. Idle
//! sessions pay nothing: no polling, no stat storms.

use std::path::{Component, Path};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use notify::{RecommendedWatcher, RecursiveMode, Watcher};

use indexio_types::Lang;

pub struct RepoWatch {
    dirty: Arc<AtomicBool>,
    // Dropping the watcher stops it; keep it alive with the server.
    _watcher: RecommendedWatcher,
}

impl RepoWatch {
    pub fn start(root: &Path) -> anyhow::Result<Self> {
        let dirty = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&dirty);
        let base = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        let watch_root = base.clone();
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            match res {
                Ok(ev) => {
                    // judge only the part below the repo root: the root
                    // itself may live under a dotted or "build" folder
                    if ev.paths.iter().any(|p| {
                        let rel = p.strip_prefix(&base).unwrap_or(p);
                        interesting(rel)
                    }) {
                        flag.store(true, Ordering::Relaxed);
                    }
                }
                // An overflow means events were lost: assume something changed.
                Err(_) => flag.store(true, Ordering::Relaxed),
            }
        })?;
        watcher.watch(&watch_root, RecursiveMode::Recursive)?;
        Ok(RepoWatch {
            dirty,
            _watcher: watcher,
        })
    }

    /// Whether anything changed since the last call (and reset).
    pub fn take_dirty(&self) -> bool {
        self.dirty.swap(false, Ordering::Relaxed)
    }
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
}
