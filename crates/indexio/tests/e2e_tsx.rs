//! End-to-end CLI test for TypeScript/React apps: a small Next.js-shaped
//! repo is indexed (`indexio index`), then `indexio symbol` must find type
//! aliases by their exact name (not a longer component name), and
//! `indexio calls` / `indexio impact` must list the JSX elements that render
//! a component. Requires `git` on PATH. Fully offline.

use std::path::Path;
use std::process::Command;

fn ci(data_dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_indexio"))
        .arg("--data-dir")
        .arg(data_dir)
        .args(args)
        .env_remove("INDEXIO_EMBED_BASE")
        .output()
        .unwrap()
}

fn git(repo: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@t")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@t")
        .output()
        .expect("git on PATH");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

const FILES: &[(&str, &str)] = &[
    (
        "lib/types.ts",
        "export type Status = 'todo' | 'done';\nexport type Task = { id: string; status: Status };\n",
    ),
    (
        "components/StatusChart.tsx",
        "import type { Status } from '../lib/types';\n\nexport default function StatusChart({ s }: { s: Status }) {\n  return <div>{s}</div>;\n}\n",
    ),
    (
        "components/Board.tsx",
        "import StatusChart from './StatusChart';\n\nexport const Board = ({ n }: { n: number }) => (\n  <section>\n    <StatusChart s=\"todo\" />\n  </section>\n);\n",
    ),
    (
        "app/page.tsx",
        "import { Board } from '../components/Board';\n\nexport default function Page() {\n  return (\n    <main>\n      <Board n={1} />\n    </main>\n  );\n}\n",
    ),
];

fn make_repo(root: &Path) {
    for (path, content) in FILES {
        let p = root.join(path);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, content).unwrap();
    }
    git(root, &["init", "-q", "-b", "main"]);
    git(root, &["add", "."]);
    git(root, &["-c", "commit.gpgsign=false", "commit", "-qm", "init"]);
}

fn stdout(out: &std::process::Output) -> String {
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn tsx_aliases_exact_names_and_render_sites() {
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("data");
    let repo = tmp.path().join("web");
    make_repo(&repo);
    stdout(&ci(&data, &["index", repo.to_str().unwrap(), "--name", "web"]));

    // type alias by exact name; `StatusChart` is not part of the answer
    let out = ci(&data, &["symbol", "Status"]);
    let text = stdout(&out);
    assert!(text.contains("web:lib/types.ts:1:"), "{text}");
    assert!(!text.contains("StatusChart"), "{text}");
    assert!(String::from_utf8_lossy(&out.stderr).is_empty());
    let text = stdout(&ci(&data, &["symbol", "Task"]));
    assert!(text.contains("web:lib/types.ts:2:"), "{text}");
    // a name with no exact definition: the fallback, flagged on stderr
    let out = ci(&data, &["symbol", "StatusCh"]);
    assert!(stdout(&out).contains("StatusChart"));
    assert!(String::from_utf8_lossy(&out.stderr).contains("no definition named exactly `StatusCh`"));

    // JSX render sites are call sites
    let text = stdout(&ci(&data, &["calls", "Board"]));
    assert!(text.contains("web:app/page.tsx:6:"), "{text}");
    let text = stdout(&ci(&data, &["calls", "StatusChart"]));
    assert!(text.contains("web:components/Board.tsx:5:"), "{text}");
    assert!(!stdout(&ci(&data, &["calls", "section"])).contains("Board.tsx"), "intrinsic element");

    // impact of the chart file: rendered by Board (a const arrow component),
    // which is rendered by Page
    let out = ci(&data, &["impact", "--file", "web:components/StatusChart.tsx", "--depth", "2", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&out)).unwrap();
    let sites = v["sites"].as_array().unwrap();
    assert!(
        sites.iter().any(|s| s["path"] == "components/Board.tsx" && s["caller"] == "Board" && s["depth"] == 1),
        "{v}"
    );
    assert!(
        sites.iter().any(|s| s["path"] == "app/page.tsx" && s["caller"] == "Page" && s["depth"] == 2),
        "{v}"
    );
}
