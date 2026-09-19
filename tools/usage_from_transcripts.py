#!/usr/bin/env python3
"""Real tool usage from Claude Code transcripts (ground truth even when the
indexio usage log is empty).

Parses ~/.claude/projects/*/*.jsonl, pairs every tool_use with its
tool_result, and reports calls + estimated result tokens (bytes/4) per tool,
per project, and the Bash commands broken down by what they run (cat/sed,
grep/rg, cargo, python, ...). Also counts the Bash hook's denials and `# raw`
escapes.

    python tools/usage_from_transcripts.py --since 2026-09-15T21:20 [--claude-dir DIR]
"""
import argparse
import collections
import glob
import json
import os
import re


def cat_of(cmd: str) -> str:
    c = cmd.strip()
    c = re.sub(r'^cd\s+"?[^;&|]+"?\s*(&&|;)\s*', "", c)
    c = re.sub(r"^rtk\s+proxy\s+", "", c)
    first = re.split(r"[\s|;&]", c, 1)[0].lower()
    if first in ("python", "python3", "py"):
        return "python script"
    if first == "cargo":
        m = re.match(r"cargo\s+(\w+)", c)
        return "cargo " + (m.group(1) if m else "")
    if first in ("grep", "rg", "egrep", "fgrep"):
        return "grep/rg"
    if first in ("sed", "head", "tail", "cat", "less", "awk", "cut"):
        return "cat/sed/head/awk"
    if first in ("find", "ls", "tree", "du", "wc"):
        return "find/ls/wc"
    if first == "git":
        m = re.match(r"git\s+(\w+)", c)
        return "git " + (m.group(1) if m else "")
    if first in ("printf", "echo"):
        return "echo/printf"
    if first in ("npm", "npx", "node", "pnpm", "yarn"):
        return "node/npm"
    if first in ("pytest", "uv", "pip"):
        return "pytest/uv/pip"
    return first[:20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="", help="ISO timestamp (UTC) lower bound")
    ap.add_argument("--claude-dir", default=os.environ.get("CLAUDE_CONFIG_DIR", os.path.expanduser("~/.claude")))
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    per_tool = collections.defaultdict(lambda: [0, 0])
    per_proj = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    bash = collections.defaultdict(lambda: [0, 0])
    # per project: shell reads/searches (what the hook targets) calls, bytes
    proj_reads = collections.defaultdict(lambda: [0, 0])
    newest = collections.defaultdict(str)
    ids = {}
    denied = collections.Counter()
    raw = 0
    for f in glob.glob(os.path.join(a.claude_dir, "projects", "*", "*.jsonl")):
        proj = os.path.basename(os.path.dirname(f))
        try:
            fh = open(f, encoding="utf-8")
        except OSError:
            continue
        for line in fh:
            if '"tool_use"' not in line and '"tool_result"' not in line:
                continue
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if o.get("timestamp", "") < a.since:
                continue
            c = (o.get("message") or {}).get("content")
            if not isinstance(c, list):
                continue
            for b in c:
                t = b.get("type")
                if t == "tool_use":
                    n = b.get("name", "?")
                    cmd = (b.get("input") or {}).get("command", "") if n == "Bash" else ""
                    ids[b.get("id")] = (proj, n, cmd)
                    per_tool[n][0] += 1
                    per_proj[proj][n][0] += 1
                    newest[proj] = max(newest[proj], o.get("timestamp", ""))
                    if n == "Bash":
                        k = cat_of(cmd)
                        bash[k][0] += 1
                        if k in ("cat/sed/head/awk", "grep/rg", "find/ls/wc"):
                            proj_reads[proj][0] += 1
                        if "# raw" in cmd:
                            raw += 1
                elif t == "tool_result":
                    pn = ids.get(b.get("tool_use_id"))
                    if not pn:
                        continue
                    cont = b.get("content")
                    text = cont if isinstance(cont, str) else " ".join(p.get("text", "") for p in cont if isinstance(p, dict))
                    per_tool[pn[1]][1] += len(text)
                    per_proj[pn[0]][pn[1]][1] += len(text)
                    if pn[1] == "Bash":
                        k = cat_of(pn[2])
                        bash[k][1] += len(text)
                        if k in ("cat/sed/head/awk", "grep/rg", "find/ls/wc"):
                            proj_reads[pn[0]][1] += len(text)
                        if "mcp__indexio__" in text and "to force the shell" in text:  # the hook's RAW tail
                            denied[pn[0]] += 1
    tot_calls = sum(v[0] for v in per_tool.values())
    tot_bytes = sum(v[1] for v in per_tool.values())
    print(f"since {a.since or 'the beginning'}: {tot_calls} tool calls, ~{tot_bytes // 4 // 1000}k tokens of results")
    print(f"\n{'tool':34}{'calls':>7}{'est_ktok':>10}{'%tok':>6}")
    for n, (c, b) in sorted(per_tool.items(), key=lambda x: -x[1][1])[: a.top]:
        print(f"{n:34}{c:7}{b // 4 // 1000:10}{100 * b / max(tot_bytes, 1):6.1f}")
    btot = sum(v[1] for v in bash.values()) or 1
    print(f"\n{'bash by command':34}{'calls':>7}{'est_ktok':>10}{'%bash':>6}")
    for k, (c, b) in sorted(bash.items(), key=lambda x: -x[1][1])[: a.top]:
        print(f"{k:34}{c:7}{b // 4 // 1000:10}{100 * b / btot:6.1f}")
    print(f"\nbash hook: denials={sum(denied.values())} {dict(denied)}  '# raw' escapes={raw}")
    print("\nper project: indexio vs built-in lookup tools")
    for proj, tools in sorted(per_proj.items()):
        ix = sum(v[0] for n, v in tools.items() if n.startswith("mcp__indexio"))
        ib = sum(v[1] for n, v in tools.items() if n.startswith("mcp__indexio"))
        gr = sum(v[0] for n, v in tools.items() if n in ("Grep", "Glob", "Read"))
        rb = sum(v[1] for n, v in tools.items() if n in ("Grep", "Glob", "Read"))
        sr, sb = proj_reads.get(proj, [0, 0])
        if ix + gr + sr:
            print(
                f"  {proj[-44:]:44} indexio={ix:4} ({ib // 4 // 1000:4}k tok)  Grep/Glob/Read={gr:4} ({rb // 4 // 1000:4}k tok)"
                f"  shell reads={sr:4} ({sb // 4 // 1000:4}k tok)  last={newest[proj][:16]}"
            )


if __name__ == "__main__":
    main()
