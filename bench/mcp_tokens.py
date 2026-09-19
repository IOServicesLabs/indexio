#!/usr/bin/env python3
"""A/B token + latency benchmark for the indexio MCP server.

Drives `indexio mcp` over stdio with a fixed set of representative tool calls
(the ones a coding agent issues in a session) and reports, per call, the
size of the tool result in tokens (tiktoken cl100k_base, a close proxy for
Claude's tokenizer) and the server-side latency. Also reports the fixed
per-session cost: `initialize.instructions` + `tools/list`.

    python bench/mcp_tokens.py --ci ~/.cargo/bin/indexio.exe --data-dir ~/.indexio \
        --repo indexio --label baseline --out bench/mcp_tokens.baseline.json
    python bench/mcp_tokens.py ... --label candidate --compare bench/mcp_tokens.baseline.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def ntok(s: str) -> int:
        return len(_enc.encode(s, disallowed_special=()))

except Exception:  # pragma: no cover

    def ntok(s: str) -> int:
        return max(1, len(s) // 4)


class Mcp:
    def __init__(self, ci, data_dir, cwd=None):
        env = dict(os.environ, INDEXIO_NO_USAGE="1")  # benchmark calls are not usage
        self.p = subprocess.Popen(
            [ci, "mcp", "--data-dir", data_dir],
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.n = 0

    def call(self, method, params=None):
        self.n += 1
        msg = {"jsonrpc": "2.0", "id": self.n, "method": method}
        if params is not None:
            msg["params"] = params
        t0 = time.perf_counter()
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        dt = (time.perf_counter() - t0) * 1000.0
        return json.loads(line), dt

    def tool(self, tool_name, args):
        r, dt = self.call("tools/call", {"name": tool_name, "arguments": args})
        if "error" in r:
            return None, r["error"], dt
        content = r["result"]["content"]
        text = "".join(c.get("text", "") for c in content)
        return text, None, dt

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def workload(repo, sample_path, sample_symbol, sample_callee, queries):
    """The calls a session typically makes. Returns (label, tool, args)."""
    w = [
        ("stats", "index_stats", {}),
        ("list_glob", "list_files", {"pattern": "**/*.py", "repo": repo}),
        ("list_sub", "list_files", {"pattern": "test", "repo": repo}),
        ("find_sym", "find_symbol", {"name": sample_symbol}),
        ("who_calls", "who_calls", {"name": sample_callee}),
        ("outline", "file_outline", {"repo": repo, "path": sample_path}),
        ("span60", "read_span", {"repo": repo, "path": sample_path, "start": 1}),
        ("span200", "read_span", {"repo": repo, "path": sample_path, "start": 1, "end": 200}),
        ("grep", "code_grep", {"pattern": "def \\w+_handler", "limit": 20}),
        ("impact_sym", "impact_of_symbol", {"names": [sample_callee], "depth": 2}),
    ]
    for i, q in enumerate(queries):
        w.append((f"lex{i}", "code_search", {"query": q["lexical"], "limit": 20}))
        w.append((f"hyb{i}", "code_search", {"query": q["hybrid"], "mode": "hybrid", "limit": 10}))
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", default=os.path.expanduser("~/.cargo/bin/indexio.exe"))
    ap.add_argument("--data-dir", default=os.path.expanduser("~/.indexio"))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--path", required=True, help="a repo-relative source file with several definitions")
    ap.add_argument("--symbol", required=True, help="a defined symbol name")
    ap.add_argument("--callee", required=True, help="a frequently-called function name")
    ap.add_argument("--queries", default=os.path.join(os.path.dirname(__file__), "mcp_queries.json"))
    ap.add_argument("--label", default="run")
    ap.add_argument("--out")
    ap.add_argument("--compare")
    ap.add_argument("--reps", type=int, default=3, help="latency reps per call (min is reported)")
    ap.add_argument("--show", action="store_true", help="print each result text")
    ap.add_argument("--cwd", help="start the server in this folder (a session's project dir)")
    a = ap.parse_args()

    with open(a.queries, encoding="utf-8") as f:
        queries = json.load(f)

    t0 = time.perf_counter()
    m = Mcp(a.ci, a.data_dir, cwd=a.cwd)
    init, init_ms = m.call(
        "initialize",
        {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "bench", "version": "0"}},
    )
    startup_ms = (time.perf_counter() - t0) * 1000.0
    instr = init["result"].get("instructions", "")
    tl, _ = m.call("tools/list")
    tools_json = json.dumps(tl["result"]["tools"], separators=(",", ":"))

    rows = {}
    rows["_session/instructions"] = {"tokens": ntok(instr), "ms": 0.0}
    rows["_session/tools_list"] = {"tokens": ntok(tools_json), "ms": 0.0}
    rows["_session/startup"] = {"tokens": 0, "ms": startup_ms}

    for label, tool, args in workload(a.repo, a.path, a.symbol, a.callee, queries):
        best = None
        text = None
        err = None
        for _ in range(a.reps):
            text, err, dt = m.tool(tool, args)
            best = dt if best is None else min(best, dt)
        if err:
            rows[label] = {"tokens": 0, "ms": best, "error": err.get("message")}
            continue
        rows[label] = {"tokens": ntok(text), "ms": best, "bytes": len(text)}
        if a.show:
            print(f"--- {label} ({tool} {json.dumps(args)})\n{text[:1500]}\n")
    m.close()

    base = None
    if a.compare:
        with open(a.compare, encoding="utf-8") as f:
            base = json.load(f)["rows"]

    tot_tok = sum(r["tokens"] for k, r in rows.items() if not k.startswith("_session"))
    tot_ms = sum(r["ms"] for k, r in rows.items() if not k.startswith("_session"))
    print(f"\n{a.label}: per-session fixed = {rows['_session/instructions']['tokens']} (instructions) + "
          f"{rows['_session/tools_list']['tokens']} (tools/list) tokens; startup {startup_ms:.0f} ms")
    hdr = f"{'call':<24}{'tokens':>8}{'ms':>9}"
    if base:
        hdr += f"{'base_tok':>10}{'dtok':>8}{'base_ms':>9}"
    print(hdr)
    for k, r in rows.items():
        line = f"{k:<24}{r['tokens']:>8}{r['ms']:>9.1f}"
        if base and k in base:
            b = base[k]
            d = r["tokens"] - b["tokens"]
            pct = (100.0 * d / b["tokens"]) if b["tokens"] else 0.0
            line += f"{b['tokens']:>10}{d:>+8}{b['ms']:>9.1f}  ({pct:+.0f}%)"
        if r.get("error"):
            line += f"   ERROR {r['error']}"
        print(line)
    line = f"{'TOTAL (tool calls)':<24}{tot_tok:>8}{tot_ms:>9.1f}"
    if base:
        btot = sum(r["tokens"] for k, r in base.items() if not k.startswith("_session"))
        bms = sum(r["ms"] for k, r in base.items() if not k.startswith("_session"))
        line += f"{btot:>10}{tot_tok - btot:>+8}{bms:>9.1f}  ({100.0 * (tot_tok - btot) / btot:+.0f}%)"
    print(line)

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"label": a.label, "rows": rows}, f, indent=1)
        print(f"saved {a.out}")


if __name__ == "__main__":
    sys.exit(main())
