"""A/B: grep commands as agents issue them vs the hook's indexio replacement.
Tokens (cl100k) of the shell output vs the MCP text result, and whether the
MCP result covers the same (file, line) pairs."""
import json, os, re, subprocess, sys, time
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
EXE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "target", "release", "indexio.exe")
DATA = os.path.expanduser("~/.indexio")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # this checkout, registered as "indexio"

cases = [
    # (grep argv, mcp tool, mcp args) -- the shapes seen in real sessions, on this repo
    (["grep", "-n", r"fn call_tool\|fn log_usage", "crates/indexio/src/mcp.rs"], "code_grep", {"pattern": "fn call_tool|fn log_usage repo:indexio path:crates/indexio/src/mcp.rs"}),
    (["grep", "-n", r"anyhow!(\|bail!(", "crates/indexio-ingest/src/lib.rs"], "code_grep", {"pattern": r"anyhow!\(|bail!\( repo:indexio path:crates/indexio-ingest/src/lib.rs"}),
    (["grep", "-n", "tombstone", "crates/indexio-index/src/lib.rs", "crates/indexio-ingest/src/lib.rs"], "code_search", {"query": "\"tombstone\" repo:indexio path:crates/indexio-index/src/lib.rs|crates/indexio-ingest/src/lib.rs", "lines": 20}),
    (["grep", "-n", r"INLINE_LINES", "crates/indexio/src/runs.rs"], "code_grep", {"pattern": r"INLINE_LINES repo:indexio path:crates/indexio/src/runs.rs"}),
    (["grep", "-rn", "reload_if_needed", "crates/"], "code_search", {"query": "\"reload_if_needed\" repo:indexio", "lines": 20}),
    (["grep", "-n", r"sync_busy\|sync_last\|SYNC_EVERY", "crates/indexio/src/mcp.rs"], "code_grep", {"pattern": "sync_busy|sync_last|SYNC_EVERY repo:indexio path:crates/indexio/src/mcp.rs"}),
    (["grep", "-n", r"^pub fn \|^fn \|^pub struct \|^impl ", "crates/indexio/src/hook.rs"], "file_outline", {"repo": "indexio", "path": "crates/indexio/src/hook.rs"}),
    (["grep", "-n", "pub fn shell_join", "-A", "14", "crates/indexio/src/runs.rs"], "find_symbol", {"name": "shell_join"}),
    (["grep", "-n", "INDEXIO_RETAIN_DAYS", "crates/indexio/src/runs.rs"], "code_search", {"query": "\"INDEXIO_RETAIN_DAYS\" repo:indexio path:crates/indexio/src/runs.rs", "lines": 20}),
    (["grep", "-n", r"#\[test\]\|fn [a-z_]*(", "crates/indexio/src/usage.rs"], "code_grep", {"pattern": r"#\[test\]|fn [a-z_]*\( repo:indexio path:crates/indexio/src/usage.rs"}),
]

p = subprocess.Popen([EXE, "mcp", "--data-dir", DATA], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=REPO, text=True, encoding="utf-8")
rid = 0
def call(method, params):
    global rid
    rid += 1
    p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}) + "\n"); p.stdin.flush()
    while True:
        line = p.stdout.readline()
        if not line:
            raise SystemExit("server died")
        o = json.loads(line)
        if o.get("id") == rid:
            return o
call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})

tot_g = tot_m = 0
print(f"{'grep':60} {'g_tok':>6} {'m_tok':>6} {'ms':>5}  lines cover")
for argv, tool, args in cases:
    g = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    t = time.perf_counter()
    r = call("tools/call", {"name": tool, "arguments": args})
    ms = (time.perf_counter() - t) * 1000
    m = r["result"]["content"][0]["text"]
    gt, mt = len(enc.encode(g)), len(enc.encode(m))
    tot_g += gt; tot_m += mt
    # coverage: grep's (file,line) pairs found in the mcp text
    glines = set()
    for ln in g.splitlines():
        mm = re.match(r"^(?:([^:]+):)?(\d+):", ln)
        if mm:
            glines.add((mm.group(1) or argv[-1], int(mm.group(2))))
    mlines = set()
    cur = None
    for ln in m.splitlines():
        if not ln.startswith(" ") and ":" in ln:
            cur = ln.split(":", 1)[1].split(" ")[0]
        mm = re.match(r"^\s+(\d+)(?:-(\d+))?[: ]", ln)
        if mm and cur:
            mlines.add((cur, int(mm.group(1))))
    cov = len(glines & mlines) / max(len(glines), 1)
    print(f"{' '.join(argv)[:60]:60} {gt:6} {mt:6} {ms:5.0f}  {len(glines):3}/{len(mlines):<3} {cov:4.0%}")
print(f"{'TOTAL':60} {tot_g:6} {tot_m:6}   delta {100*(tot_m-tot_g)/max(tot_g,1):+.0f}%")
p.stdin.close(); p.wait(timeout=30)
