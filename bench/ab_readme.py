#!/usr/bin/env python3
"""indexio vs the coding agent's built-in tools, on the same tasks.

Reproduces what Claude Code's built-in tools return — `Grep` (every match as
`path:line:content`), `Read` (the whole file as `N<TAB>line`, 2,000-line
default), `Glob` (one path per line) — and runs the same task through the
indexio MCP tools, measuring the tokens (tiktoken cl100k_base) that land in
the model's context and the wall time. Tasks are the ones agents actually
issue: find where a symbol is defined, list its call sites, read a function,
list files, and answer a natural-language question about the code (from
`known_answers_local.json`, model-written).

    python bench/ab_readme.py --repo indexio [--n-questions 12] [--md]

The built-in side is measured by re-implementing the tool semantics over the
repo's working tree (ripgrep-equivalent walk of tracked files); the tokens of
those results are what the agent would have received.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from mcp_io import Mcp  # noqa: E402

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def ntok(s):
        return len(_enc.encode(s, disallowed_special=()))
except Exception:  # pragma: no cover
    def ntok(s):
        return max(1, len(s) // 4)


def tracked_files(root):
    out = subprocess.run(["git", "-C", root, "ls-files", "-z"], capture_output=True).stdout
    return [p.decode("utf-8", "replace") for p in out.split(b"\0") if p]


def builtin_grep(root, files, pattern, path_prefix=None):
    """Claude Code `Grep` content mode: `path:line:text` for every match."""
    rx = re.compile(pattern)
    lines = []
    for rel in files:
        if path_prefix and not rel.startswith(path_prefix):
            continue
        try:
            with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if rx.search(line):
                        lines.append("%s:%d:%s" % (rel.replace("/", "\\"), i, line.rstrip("\n")))
        except OSError:
            continue
    return "\n".join(lines)


def builtin_read(root, rel, offset=None, limit=None):
    """Claude Code `Read`: `N<TAB>line`, whole file (2,000 lines) or a range."""
    with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    start = (offset or 1) - 1
    end = start + (limit or 2000)
    return "\n".join("%d\t%s" % (i + 1, l) for i, l in enumerate(lines[start:end], start))


def builtin_glob(files, pattern):
    import fnmatch
    return "\n".join(rel.replace("/", "\\") for rel in files if fnmatch.fnmatch(rel, pattern))


def symbols_of(root, files, n, seed=3):
    """Function/struct/class names defined in the repo, sampled."""
    import random
    rx = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|class|def|function|interface|type)\s+([A-Za-z_][A-Za-z0-9_]{4,})")
    found = {}
    for rel in files:
        if not rel.endswith((".rs", ".py", ".ts", ".tsx", ".js", ".go", ".java")):
            continue
        try:
            with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    m = rx.match(line)
                    if m and m.group(1) not in found:
                        found[m.group(1)] = (rel, i)
        except OSError:
            continue
    names = sorted(found)
    random.Random(seed).shuffle(names)
    # prefer names that occur in more than one file (real "who calls" targets)
    return [(nm, found[nm]) for nm in names[: n * 3]][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="registered repo name")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/.indexio"))
    ap.add_argument("--ci", default=os.path.expanduser("~/.cargo/bin/indexio.exe"))
    ap.add_argument("--n", type=int, default=10, help="symbols / files per task type")
    ap.add_argument("--n-questions", type=int, default=12)
    ap.add_argument("--answers", help="known-answers JSON (default bench/known_answers_local.json, from gen_questions.py)")
    ap.add_argument("--md", action="store_true", help="print a Markdown table")
    a = ap.parse_args()
    root = None
    for f in os.listdir(os.path.join(a.data_dir, "repos")):
        st = json.load(open(os.path.join(a.data_dir, "repos", f), encoding="utf-8"))
        if st.get("name") == a.repo:
            root = st["path"]
    if not root:
        sys.exit("repo not registered: " + a.repo)
    files = tracked_files(root)
    m = Mcp(a.ci, a.data_dir, root)
    m.call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "ab", "version": "0"}})
    time.sleep(2)

    def idx(tool, args):
        t0 = time.perf_counter()
        r, _ = m.call("tools/call", {"name": tool, "arguments": args})
        txt = r["result"]["content"][0]["text"] if "result" in r else json.dumps(r)
        return txt, time.perf_counter() - t0

    rows = []  # (task, builtin_tokens, builtin_s, indexio_tokens, indexio_s, n)

    # 1. where is X defined  — built-in: Grep `fn X|def X|class X|struct X` over the repo
    syms = symbols_of(root, files, a.n)
    bt = bs = it = is_ = 0.0
    for name, _ in syms:
        t0 = time.perf_counter()
        g = builtin_grep(root, files, r"\b(fn|def|class|struct|enum|trait|function|type|interface)\s+%s\b" % re.escape(name))
        bs += time.perf_counter() - t0
        bt += ntok(g)
        txt, dt = idx("find_symbol", {"name": name})
        it += ntok(txt)
        is_ += dt
    rows.append(("find where a symbol is defined", bt, bs, it, is_, len(syms)))

    # 2. who calls X — built-in: Grep `X(` over the repo
    bt = bs = it = is_ = 0.0
    for name, _ in syms:
        t0 = time.perf_counter()
        g = builtin_grep(root, files, r"\b%s\s*\(" % re.escape(name))
        bs += time.perf_counter() - t0
        bt += ntok(g)
        txt, dt = idx("who_calls", {"name": name})
        it += ntok(txt)
        is_ += dt
    rows.append(("list a symbol's call sites", bt, bs, it, is_, len(syms)))

    # 3. read the function — built-in: Read the whole file (an agent does not know the range);
    #    indexio: file_outline then read_span of the definition
    bt = bs = it = is_ = 0.0
    for name, (rel, line) in syms:
        t0 = time.perf_counter()
        r = builtin_read(root, rel)
        bs += time.perf_counter() - t0
        bt += ntok(r)
        o, d1 = idx("file_outline", {"repo": a.repo, "path": rel})
        s, d2 = idx("read_span", {"repo": a.repo, "path": rel, "start": line})
        it += ntok(o) + ntok(s)
        is_ += d1 + d2
    rows.append(("read one function (agent knows the file, not the lines)", bt, bs, it, is_, len(syms)))

    # 4. list files — built-in: Glob **/*.<ext>
    exts = {}
    for rel in files:
        e = os.path.splitext(rel)[1]
        if e:
            exts[e] = exts.get(e, 0) + 1
    top = [e for e, _ in sorted(exts.items(), key=lambda kv: -kv[1])[:3]]
    bt = bs = it = is_ = 0.0
    for e in top:
        t0 = time.perf_counter()
        g = builtin_glob(files, "*" + e)
        bs += time.perf_counter() - t0
        bt += ntok(g)
        txt, dt = idx("list_files", {"pattern": "**/*" + e, "repo": a.repo})
        it += ntok(txt)
        is_ += dt
    rows.append(("list the repo's files of a type", bt, bs, it, is_, len(top)))

    # 5. answer a question — built-in: the agent greps the question's 3 most specific
    #    words (what agents do with Grep on NL questions); indexio: hybrid code_search
    # questions about this machine's repos: generate them with bench/gen_questions.py
    qa_path = a.answers or os.path.join(os.path.dirname(__file__), "known_answers_local.json")
    qa = [q for q in json.load(open(qa_path, encoding="utf-8")) if q["repo"] == a.repo] if os.path.exists(qa_path) else []
    qa = qa[: a.n_questions]
    stop = {"how", "to", "the", "a", "an", "and", "or", "of", "in", "for", "with", "using", "from", "into", "on", "when", "that", "is", "are", "by", "as", "at", "it", "its", "via"}
    bt = bs = it = is_ = 0.0
    b_hit = i_hit = 0
    for q in qa:
        words = [w for w in re.findall(r"[A-Za-z]{4,}", q["q"].lower()) if w not in stop]
        words = sorted(words, key=len, reverse=True)[:3]
        t0 = time.perf_counter()
        g = builtin_grep(root, files, "|".join(re.escape(w) for w in words)) if words else ""
        bs += time.perf_counter() - t0
        bt += ntok(g)
        b_hit += q["path_suffix"].replace("/", "\\") in g
        txt, dt = idx("code_search", {"query": q["q"], "mode": "hybrid", "limit": 5})
        it += ntok(txt)
        is_ += dt
        i_hit += q["path_suffix"] in txt
    rows.append(("answer a question about the code (%d/%d vs %d/%d found the file)" % (b_hit, len(qa), i_hit, len(qa)), bt, bs, it, is_, len(qa)))
    m.close()

    tb = sum(r[1] for r in rows)
    ti = sum(r[3] for r in rows)
    if a.md:
        print("| task (%s, n) | built-in tokens | indexio tokens | reduction |" % a.repo)
        print("|---|---|---|---|")
        for t, b, bs_, i, is__, n in rows:
            print("| %s (n=%d) | %s | %s | **%d%%** |" % (t, n, "{:,}".format(int(b)), "{:,}".format(int(i)), round(100 * (1 - i / max(b, 1)))))
        print("| **total** | **%s** | **%s** | **%d%%** |" % ("{:,}".format(int(tb)), "{:,}".format(int(ti)), round(100 * (1 - ti / max(tb, 1)))))
    else:
        print("%-62s %5s %10s %10s %8s %8s" % ("task", "n", "builtin", "indexio", "b ms", "i ms"))
        for t, b, bs_, i, is__, n in rows:
            print("%-62s %5d %10d %10d %8.0f %8.0f" % (t[:62], n, b, i, 1000 * bs_ / max(n, 1), 1000 * is__ / max(n, 1)))
        print("%-62s %5s %10d %10d   -> %d%% fewer tokens" % ("TOTAL", "", tb, ti, round(100 * (1 - ti / max(tb, 1)))))


if __name__ == "__main__":
    main()
