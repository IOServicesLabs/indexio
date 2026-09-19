//! Shared helpers for the SPEC-P6 surfaces (CLI, HTTP, MCP): repo-name →
//! working-tree resolution, `git diff` capture, target parsing and the
//! human-readable impact printer.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Context;
use serde_json::{json, Value};

use indexio_query::{ChangedSymbol, ImpactOptions, ImpactReport};

/// `REPO:PATH` → (repo, path). The path keeps any further colons.
pub(crate) fn parse_repo_path(target: &str) -> anyhow::Result<(String, String)> {
    match target.split_once(':') {
        Some((r, p)) if !r.is_empty() && !p.is_empty() => Ok((r.to_string(), p.to_string())),
        _ => anyhow::bail!("expected REPO:PATH, got '{target}'"),
    }
}

/// `START[:END]` → (start, end); END defaults to START + 60 - 1.
pub(crate) fn parse_line_range(range: &str) -> anyhow::Result<(u32, u32)> {
    let (a, b) = match range.split_once(':') {
        Some((a, b)) => (a, Some(b)),
        None => (range, None),
    };
    let start: u32 = a.trim().parse().context("START must be a positive integer")?;
    let end: u32 = match b {
        Some(b) => b.trim().parse().context("END must be a positive integer")?,
        None => start.saturating_add(59),
    };
    Ok((start.max(1), end.max(start.max(1))))
}

/// Working-tree path of a registered repo (`<data_dir>/repos/<name>.json`).
pub(crate) fn resolve_repo_path(data_dir: &Path, name: &str) -> anyhow::Result<PathBuf> {
    let state = indexio_ingest::repo_state(data_dir, name)
        .with_context(|| format!("repo '{name}' is not registered in {}", data_dir.display()))?;
    Ok(state.path)
}

/// `git -C <repo> diff -U0 --no-color --no-ext-diff <base>`: every
/// uncommitted change (staged + unstaged) relative to `base`. Untracked
/// files are not part of a diff and therefore not analysed.
pub(crate) fn git_diff(repo_path: &Path, base: &str) -> anyhow::Result<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo_path)
        .args(["diff", "-U0", "--no-color", "--no-ext-diff", base])
        .output()
        .context("running git (is it on PATH?)")?;
    if !out.status.success() {
        anyhow::bail!(
            "git diff failed in {}: {}",
            repo_path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Post-change file provider for `impact_diff`: the working tree of the
/// registered repo (None when unknown/unreadable → old-side mapping only).
pub(crate) fn working_tree_provider(repo_path: Option<PathBuf>) -> impl Fn(&str) -> Option<Vec<u8>> {
    move |rel: &str| {
        let root = repo_path.as_ref()?;
        // Reject traversal outside the repo (diff text is untrusted input).
        if rel.starts_with('/') || rel.contains("..") || rel.contains(':') {
            return None;
        }
        std::fs::read(root.join(rel)).ok()
    }
}

/// JSON envelope shared by the CLI (`--json`), HTTP and MCP.
pub(crate) fn impact_to_json(changed: Option<&[ChangedSymbol]>, report: &ImpactReport) -> Value {
    let mut v = serde_json::to_value(report).expect("ImpactReport is Serialize");
    if let Some(c) = changed {
        v["changed"] = serde_json::to_value(c).expect("ChangedSymbol is Serialize");
    }
    v
}

pub(crate) fn options(depth: u32, max_sites: usize, max_fanout: usize) -> ImpactOptions {
    ImpactOptions {
        depth: depth.max(1),
        max_sites: max_sites.max(1),
        max_fanout: max_fanout.max(1),
    }
}

/// Human output for `indexio impact`.
pub(crate) fn print_impact(changed: Option<&[ChangedSymbol]>, r: &ImpactReport) {
    if let Some(changed) = changed {
        println!("changed definitions ({}):", changed.len());
        for c in changed {
            let scope = if c.scope.is_empty() {
                String::new()
            } else {
                format!("  [{}]", c.scope)
            };
            println!(
                "  {}  {} {}{}  lines {}",
                c.path,
                c.kind.name(),
                c.name,
                scope,
                c.lines
                    .iter()
                    .map(|l| l.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
    }
    println!("roots: {}", r.roots.join(", "));
    if !r.definitions.is_empty() {
        println!("definitions ({}):", r.definitions.len());
        for d in &r.definitions {
            println!("  {}:{}:{}  {}", d.repo, d.path, d.line, d.snippet);
        }
    }
    println!("call sites ({}):", r.sites.len());
    for s in &r.sites {
        let caller = if s.caller.is_empty() {
            "<top-level>".to_string()
        } else {
            format!("{}()", s.caller)
        };
        println!(
            "  d{}  {}:{}:{}  {}  in {}  | {}",
            s.depth, s.repo, s.path, s.line, s.symbol, caller, s.snippet
        );
    }
    println!("files ({}):", r.files.len());
    for f in &r.files {
        println!(
            "  {}:{}  sites={} min_depth={}",
            f.repo, f.path, f.sites, f.min_depth
        );
    }
    if !r.importers.is_empty() {
        println!("importers ({}):", r.importers.len());
        for h in &r.importers {
            println!("  {}:{}:{}  {}", h.repo, h.path, h.line, h.snippet);
        }
    }
    if r.truncated {
        println!("(truncated: a fan-out cap was hit; raise --max-sites/--max-fanout or lower --depth)");
    }
    println!("took {} ms", r.took_ms);
}

/// Human output for `indexio outline`.
pub(crate) fn print_outline(repo: &str, path: &str, items: &[indexio_symbols::OutlineItem]) {
    println!("{repo}:{path}  ({} definitions)", items.len());
    for it in items {
        let depth = if it.scope.is_empty() {
            0
        } else {
            it.scope.split("::").count()
        };
        println!(
            "{:>5}-{:<5} {}{} {}",
            it.start_line,
            it.end_line,
            "  ".repeat(depth),
            it.kind.name(),
            it.name
        );
    }
}

pub(crate) fn outline_to_json(repo: &str, path: &str, items: &[indexio_symbols::OutlineItem]) -> Value {
    json!({ "repo": repo, "path": path, "items": items })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_targets() {
        assert_eq!(
            parse_repo_path("core:src/a.rs").unwrap(),
            ("core".to_string(), "src/a.rs".to_string())
        );
        assert_eq!(
            parse_repo_path("core:src/a:b.rs").unwrap().1,
            "src/a:b.rs"
        );
        assert!(parse_repo_path("nocolon").is_err());
        assert!(parse_repo_path(":x").is_err());
        assert!(parse_repo_path("x:").is_err());
        assert_eq!(parse_line_range("10").unwrap(), (10, 69));
        assert_eq!(parse_line_range("10:20").unwrap(), (10, 20));
        assert_eq!(parse_line_range("0:0").unwrap(), (1, 1));
        assert_eq!(parse_line_range("20:10").unwrap(), (20, 20));
        assert!(parse_line_range("a").is_err());
    }

    #[test]
    fn working_tree_provider_reads_and_guards() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join("src/a.rs"), b"fn a() {}\n").unwrap();
        let p = working_tree_provider(Some(tmp.path().to_path_buf()));
        assert_eq!(p("src/a.rs").unwrap(), b"fn a() {}\n");
        assert!(p("src/missing.rs").is_none());
        assert!(p("../etc/passwd").is_none());
        assert!(p("/abs").is_none());
        let none = working_tree_provider(None);
        assert!(none("src/a.rs").is_none());
    }

    #[test]
    fn options_clamp() {
        let o = options(0, 0, 0);
        assert_eq!((o.depth, o.max_sites, o.max_fanout), (1, 1, 1));
    }
}
