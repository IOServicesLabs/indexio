"""Replay a project's Bash calls through the hook and weigh them by the tokens
their results actually cost: how much of the shell-read spend would the hook
have redirected?  python hook_potential.py <exe> <project-slug> <since-iso>"""
import glob, json, os, re, subprocess, sys, collections
EXE, PROJ, SINCE = sys.argv[1], sys.argv[2], sys.argv[3]
proj = os.path.expanduser("~/.claude/projects/" + PROJ)
cwd = "~/" + PROJ.replace("C--Users-me-", "").replace("-", "/")
uses, results = {}, {}
for f in glob.glob(os.path.join(proj, "*.jsonl")):
    for line in open(f, encoding="utf-8", errors="replace"):
        if '"tool_use"' not in line and '"tool_result"' not in line:
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if o.get("timestamp", "") < SINCE:
            continue
        c = (o.get("message") or {}).get("content")
        if not isinstance(c, list):
            continue
        for b in c:
            if b.get("type") == "tool_use" and b.get("name") == "Bash":
                uses[b["id"]] = (b.get("input") or {}).get("command", "")
            elif b.get("type") == "tool_result":
                cont = b.get("content")
                if isinstance(cont, list):
                    cont = " ".join(x.get("text", "") for x in cont if isinstance(x, dict))
                results[b.get("tool_use_id")] = len(str(cont or ""))
tot = collections.Counter(); n = collections.Counter()
rows = []
for uid, cmd in uses.items():
    tok = results.get(uid, 0) / 4
    if not re.match(r"^(cd [^;&|]+\s*(&&|;)\s*)?(rtk proxy )?(cat|sed|head|tail|grep|rg|find)\b", cmd.strip()):
        tot["other bash"] += tok; n["other bash"] += 1
        continue
    inp = json.dumps({"tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": cwd})
    r = subprocess.run([EXE, "hook", "bash", "--data-dir", os.path.expanduser("~/.indexio")], input=inp, capture_output=True, text=True, encoding="utf-8")
    out = r.stdout.strip()
    if out:
        reason = json.loads(out)["hookSpecificOutput"]["permissionDecisionReason"]
        call = re.search(r"mcp__indexio__(\w+)", reason)
        k = "deny:" + (call.group(1) if call else "?")
    else:
        k = "allow:read"
    tot[k] += tok; n[k] += 1
    rows.append((tok, k, cmd[:110].replace("\n", " ")))
print("%-22s %5s %8s" % ("verdict", "calls", "tokens"))
for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
    print("%-22s %5d %8.0f" % (k, n[k], v))
print("\nlargest allowed reads:")
for tok, k, cmd in sorted(rows, reverse=True)[:12]:
    if k.startswith("allow"):
        print("%6.0f %s" % (tok, cmd))
