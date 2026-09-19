"""Latency of cache-dependent calls right after an engine reload (caused by
an external shard write, e.g. another session's transcript import)."""
import json, os, subprocess, sys, time
DATA = os.path.expanduser("~/.indexio")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # this checkout, registered as "indexio"
MARKF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "SPEC-P9.md")

def server(exe):
    p = subprocess.Popen([exe, "mcp", "--data-dir", DATA], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", cwd=REPO)
    rid = [0]
    def call(method, params):
        rid[0] += 1
        p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid[0], "method": method, "params": params}) + "\n"); p.stdin.flush()
        while True:
            o = json.loads(p.stdout.readline())
            if o.get("id") == rid[0]:
                return o
    call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})
    return p, call

def timed(call, name, args):
    t = time.perf_counter(); call("tools/call", {"name": name, "arguments": args}); return (time.perf_counter() - t) * 1000

calls = [
    ("find_symbol", {"name": "main"}),
    ("find_symbol", {"name": "spawn"}),
    ("read_span", {"repo": "indexio", "path": "crates/indexio/src/mcp.rs", "start": 1200}),
    ("read_span", {"repo": "indexio", "path": "crates/indexio/src/main.rs", "start": 149}),
    ("who_calls", {"name": "spawn"}),
]
EXE_OLD = os.path.expanduser("~/.cargo/bin/indexio.exe")
EXE_NEW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "target", "release", "indexio.exe")
CLI = EXE_NEW
for label, exe in (("old", EXE_OLD), ("new", EXE_NEW)):
    p, call = server(exe)
    warm = [timed(call, n, a) for n, a in calls]  # cold
    warm = [timed(call, n, a) for n, a in calls]  # warm
    # external shard write -> reload on next call
    with open(MARKF, "a", encoding="utf-8") as f:
        f.write("\nProbe marker reloadprobe_%d.\n" % int(time.time()))
    subprocess.run([CLI, "reindex", "--data-dir", DATA, "--repo", "indexio", "--worktree"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    after = [timed(call, n, a) for n, a in calls]
    print(f"{label}: warm {sum(warm):6.1f} ms  after reload {sum(after):6.1f} ms  " + " ".join(f"{n[:9]}={w:.0f}/{a:.0f}" for (n, _), w, a in zip(calls, warm, after)))
    p.stdin.close(); p.wait(timeout=30)
# clean the probe lines
s = open(MARKF, encoding="utf-8").read()
lines = [l for l in s.split("\n") if not l.startswith("Probe marker reloadprobe_")]
open(MARKF, "w", encoding="utf-8", newline="\n").write("\n".join(lines).rstrip("\n") + "\n")
subprocess.run([CLI, "reindex", "--data-dir", DATA, "--repo", "indexio", "--worktree"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
