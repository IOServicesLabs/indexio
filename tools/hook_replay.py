"""Replay a session's real Bash commands through the hook: verdict + reason."""
import glob, json, os, re, subprocess, sys, time, collections
EXE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "target", "debug", "indexio.exe")
proj = os.path.expanduser("~/.claude/projects/C--Users-me-code-my-service")
since = "2026-09-15T18:00"
cmds = []
for f in glob.glob(os.path.join(proj, "*.jsonl")):
    for line in open(f, encoding="utf-8"):
        if '"tool_use"' not in line:
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if o.get("timestamp", "") < since:
            continue
        c = (o.get("message") or {}).get("content")
        if not isinstance(c, list):
            continue
        for b in c:
            if b.get("type") == "tool_use" and b.get("name") == "Bash":
                cmds.append((b.get("input") or {}).get("command", ""))
reads = [c for c in cmds if re.match(r"^(cd [^;&|]+\s*(&&|;)\s*)?(rtk proxy )?(cat|sed|head|tail|grep|rg|find)\b", c.strip())]
print(len(cmds), "bash calls;", len(reads), "shell reads")
kinds = collections.Counter()
tot = 0.0
for c in reads:
    inp = json.dumps({"tool_name": "Bash", "tool_input": {"command": c}, "cwd": "~/code/repo-b"})
    t = time.perf_counter()
    r = subprocess.run([EXE, "hook", "bash", "--data-dir", os.path.expanduser("~/.indexio")], input=inp, capture_output=True, text=True, encoding="utf-8")
    ms = (time.perf_counter() - t) * 1000
    tot += ms
    out = r.stdout.strip()
    if out:
        reason = json.loads(out)["hookSpecificOutput"]["permissionDecisionReason"]
        call = re.search(r"mcp__indexio__(\w+) (\{[^}]*\})", reason)
        kinds["deny:" + (call.group(1) if call else "?")] += 1
        print("DENY %5.0fms | %s\n      -> %s" % (ms, c[:110].replace("\n", " "), (call.group(0) if call else reason)[:160]))
    else:
        kinds["allow"] += 1
        print("allow %5.0fms | %s" % (ms, c[:120].replace("\n", " ")))
print(dict(kinds), "avg %.0f ms" % (tot / max(len(reads), 1)))
