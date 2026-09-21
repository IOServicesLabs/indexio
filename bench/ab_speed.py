#!/usr/bin/env python3
"""A/B wall-clock: shell tools vs the indexio MCP server for the same lookups.

    python bench/ab_speed.py                 # this checkout, registered as "indexio"
    python bench/ab_speed.py --repo NAME --dir PATH --n 10

Both sides see the same tree. The shell side runs each command through
`bash -c` the way an agent's Bash tool would (rg, grep, sed, cat, find); the
indexio side talks to one warm `indexio mcp` server over stdio, so the
numbers include the JSON-RPC transport. Medians of `--n` runs. The last three
rows are the fixed cost every Bash tool call pays before its command runs:
the shell spawn and the two PreToolUse hooks.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="indexio", help="registered repo name")
    ap.add_argument("--dir", default=HERE, help="its checkout")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/.indexio"))
    ap.add_argument("--exe", default=os.path.join(HERE, "target", "release", "indexio" + (".exe" if os.name == "nt" else "")))
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    if not os.path.exists(a.exe):
        a.exe = "indexio"
    env = dict(os.environ, INDEXIO_NO_USAGE="1")

    def timed(fn):
        xs = []
        for _ in range(a.n):
            t = time.perf_counter()
            fn()
            xs.append((time.perf_counter() - t) * 1000)
        return statistics.median(xs)

    def sh(cmd):
        return lambda: subprocess.run(["bash", "-c", cmd], cwd=a.dir, capture_output=True, env=env)

    p = subprocess.Popen(
        [a.exe, "mcp", "--data-dir", a.data_dir],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", cwd=a.dir, env=env,
    )
    rid = [0]

    def rpc(method, params):
        rid[0] += 1
        p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid[0], "method": method, "params": params}) + "\n")
        p.stdin.flush()
        while True:
            line = p.stdout.readline()
            if not line:
                sys.exit("server died")
            o = json.loads(line)
            if o.get("id") == rid[0]:
                return o

    def call(name, args):
        return lambda: rpc("tools/call", {"name": name, "arguments": args})

    rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "ab", "version": "0"}})
    call("index_stats", {})()  # warm
    r = a.repo
    cases = [
        ("identifier grep (rg)", sh("rg -n 'fn call_tool' crates"), call("code_grep", {"pattern": "fn call_tool", "repo": r})),
        ("regex grep (rg)", sh("rg -n 'pub fn \\w+_repo' crates"), call("code_grep", {"pattern": "pub fn \\w+_repo", "repo": r})),
        ("grep -rn (GNU)", sh("grep -rn 'call_tool' crates --include='*.rs'"), call("code_grep", {"pattern": "call_tool", "repo": r})),
        ("read 120 lines (sed)", sh("sed -n '1,120p' crates/indexio/src/mcp.rs"), call("read_span", {"repo": r, "path": "crates/indexio/src/mcp.rs", "start": 1, "end": 120})),
        ("whole file (cat) / outline", sh("cat crates/indexio/src/runs.rs"), call("file_outline", {"repo": r, "path": "crates/indexio/src/runs.rs"})),
        ("find -name / list_files", sh("find crates -name '*.rs'"), call("list_files", {"pattern": "**/*.rs", "repo": r})),
        ("definition (rg) / find_symbol", sh("rg -n 'fn shell_join' crates"), call("find_symbol", {"name": "shell_join"})),
    ]
    print(f"{'case':30} {'bash ms':>8} {'indexio ms':>11}   (medians of {a.n}, wall-clock incl. transport)")
    for name, b, i in cases:
        print(f"{name:30} {timed(b):8.0f} {timed(i):11.1f}")
    print()
    print(f"{'bash -c true':30} {timed(sh('true')):8.0f}   shell spawn alone")
    hook_in = json.dumps({"tool_name": "Bash", "tool_input": {"command": "echo hi"}, "cwd": a.dir, "session_id": "ab"})
    hook = lambda: subprocess.run([a.exe, "hook", "bash"], input=hook_in, capture_output=True, text=True, env=env)
    print(f"{'indexio hook bash':30} {timed(hook):8.0f}   per Bash call (PreToolUse)")
    try:
        rtk = lambda: subprocess.run(["rtk", "hook", "claude"], input=hook_in, capture_output=True, text=True, env=env)
        print(f"{'rtk hook claude':30} {timed(rtk):8.0f}   per Bash call (PreToolUse), if installed")
    except OSError:
        pass
    p.kill()


if __name__ == "__main__":
    main()
