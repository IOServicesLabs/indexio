"""A/B impact_of_symbol payloads: old vs new binary, several names."""
import os
import json, subprocess, sys, time, tiktoken
enc = tiktoken.get_encoding("cl100k_base")
OLD = os.path.expanduser("~/.cargo/bin/indexio.exe")
NEW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "target", "release", "indexio.exe")
DATA = os.path.expanduser("~/.indexio")
CWD = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/code/repo-a")
names = sys.argv[2:] or ["spawn", "new", "run", "handle", "send", "load", "parse", "write", "main", "get"]

def server(exe):
    p = subprocess.Popen([exe, "mcp", "--data-dir", DATA], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", cwd=CWD)
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

def measure(call, name):
    best = None
    for _ in range(2):
        t = time.perf_counter()
        r = call("tools/call", {"name": "impact_of_symbol", "arguments": {"names": [name]}})
        ms = (time.perf_counter() - t) * 1000
        best = ms if best is None else min(best, ms)
    txt = r["result"]["content"][0]["text"]
    direct = next((l for l in txt.splitlines() if l.startswith("call sites:")), "")
    return len(enc.encode(txt)), best, direct, txt

po, co = server(OLD)
pn, cn = server(NEW)
print(f"{'name':10} {'old_tok':>8} {'new_tok':>8} {'d%':>5} {'old_ms':>7} {'new_ms':>7}  sites old -> new")
to = tn = 0
for n in names:
    a, ams, ad, _ = measure(co, n)
    b, bms, bd, bt = measure(cn, n)
    to += a; tn += b
    print(f"{n:10} {a:8} {b:8} {100*(b-a)/max(a,1):+4.0f}% {ams:7.0f} {bms:7.0f}  {ad[11:40]} -> {bd[11:40]}")
    if n == names[0]:
        open(sys.argv[2] if len(sys.argv) > 2 else "impact_fixture.diff", encoding="utf-8").write(bt)
print(f"{'TOTAL':10} {to:8} {tn:8} {100*(tn-to)/max(to,1):+4.0f}%")
po.stdin.close(); pn.stdin.close()
