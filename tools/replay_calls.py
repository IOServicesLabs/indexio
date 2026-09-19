#!/usr/bin/env python3
"""Replay a project's real indexio tool calls against the installed server
and compare result sizes: what a session on an old server would get from a
restart (SPEC-P10 §16).

    python tools/replay_calls.py --project C--Users-me-code-my-service --since 2026-09-16T12:00
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
from mcp_io import Mcp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Claude project slug under ~/.claude/projects")
    ap.add_argument("--since", required=True)
    ap.add_argument("--ci", default=os.path.expanduser("~/.cargo/bin/indexio.exe"))
    ap.add_argument("--data-dir", default=os.path.expanduser("~/.indexio"))
    ap.add_argument("--cwd", help="working directory for the server (the session's repo)")
    ap.add_argument("--max", type=int, default=400)
    a = ap.parse_args()
    proj = os.path.expanduser("~/.claude/projects/" + a.project)
    uses, results = {}, {}
    for f in glob.glob(os.path.join(proj, "*.jsonl")):
        for line in open(f, encoding="utf-8", errors="replace"):
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
                if b.get("type") == "tool_use" and (b.get("name") or "").startswith("mcp__indexio__"):
                    uses[b["id"]] = (b["name"][len("mcp__indexio__"):], b.get("input") or {})
                elif b.get("type") == "tool_result" and b.get("tool_use_id") in uses:
                    cont = b.get("content")
                    if isinstance(cont, list):
                        cont = " ".join(x.get("text", "") for x in cont if isinstance(x, dict))
                    results[b["tool_use_id"]] = len(str(cont or ""))
    calls = [(uses[k][0], uses[k][1], results[k]) for k in uses if k in results][: a.max]
    if not calls:
        sys.exit("no calls found")
    cwd = a.cwd
    if not cwd:
        # the session's repo from ~/.indexio/repos: the slug's last segment
        name = a.project.split("-")[-1]
        for rf in glob.glob(os.path.join(a.data_dir, "repos", "*.json")):
            st = json.load(open(rf, encoding="utf-8"))
            if st.get("name", "").lower().endswith(name.lower()):
                cwd = st["path"]
                break
    m = Mcp(a.ci, a.data_dir, cwd)
    m.call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "replay", "version": "0"}})
    time.sleep(2)
    per = {}
    old_total = new_total = 0
    for tool, args, old_len in calls:
        if tool in ("refresh_index",):
            continue
        r, _ = m.call("tools/call", {"name": tool, "arguments": args})
        txt = r["result"]["content"][0]["text"] if "result" in r else json.dumps(r)
        new_len = len(txt)
        p = per.setdefault(tool, [0, 0, 0])
        p[0] += 1
        p[1] += old_len
        p[2] += new_len
        old_total += old_len
        new_total += new_len
    m.close()
    print("%-16s %5s %10s %10s %7s" % ("tool", "calls", "old ktok", "new ktok", "change"))
    for t, (n, o, nw) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        print("%-16s %5d %10.1f %10.1f %6.0f%%" % (t, n, o / 4000, nw / 4000, 100 * (nw - o) / max(o, 1)))
    print("%-16s %5d %10.1f %10.1f %6.0f%%" % ("TOTAL", sum(v[0] for v in per.values()), old_total / 4000, new_total / 4000, 100 * (new_total - old_total) / max(old_total, 1)))


if __name__ == "__main__":
    main()
