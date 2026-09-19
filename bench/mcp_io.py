#!/usr/bin/env python3
"""I/O profile of the indexio MCP server: read syscalls, bytes read and RSS
after startup and after each representative call (psutil io_counters on the
server process). What a slow disk pays.

    python bench/mcp_io.py [--ci EXE] [--data-dir DIR] [--repo indexio] [--cwd DIR]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import psutil


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
        self.ps = psutil.Process(self.p.pid)
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
        return json.loads(line), (time.perf_counter() - t0) * 1000.0

    def io(self):
        c = self.ps.io_counters()
        m = self.ps.memory_info()
        return c.read_count, c.read_bytes, c.other_count, m.rss

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", default=os.path.expanduser("~/.cargo/bin/indexio.exe"))
    ap.add_argument("--data-dir", default=os.path.expanduser("~/.indexio"))
    ap.add_argument("--repo", default="indexio")
    ap.add_argument("--path", default="crates/indexio/src/main.rs")
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--settle", type=float, default=3.0, help="seconds to let prewarm threads finish")
    a = ap.parse_args()
    m = Mcp(a.ci, a.data_dir, a.cwd)
    t0 = time.perf_counter()
    m.call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "io", "version": "0"}})
    init_ms = (time.perf_counter() - t0) * 1000
    time.sleep(a.settle)  # prewarm (rindex model, vec index) runs in the background
    prev = m.io()
    print("%-14s %7s %9s %8s %8s %8s" % ("step", "ms", "reads", "MB read", "other", "RSS MB"))
    print("%-14s %7.0f %9d %8.1f %8d %8.0f" % ("startup", init_ms, prev[0], prev[1] / 1e6, prev[2], prev[3] / 1e6))
    calls = [
        ("stats", "index_stats", {}),
        ("list_glob", "list_files", {"pattern": "**/*.rs", "repo": a.repo}),
        ("find_sym", "find_symbol", {"name": "main"}),
        ("who_calls", "who_calls", {"name": "spawn"}),
        ("outline", "file_outline", {"repo": a.repo, "path": a.path}),
        ("span60", "read_span", {"repo": a.repo, "path": a.path, "start": 1, "end": 60}),
        ("grep", "code_grep", {"pattern": "def \\w+_handler", "limit": 20}),
        ("lex0", "code_search", {"query": "Command::new", "mode": "lexical"}),
        ("lex1", "code_search", {"query": "fn handle_request", "mode": "lexical"}),
        ("hyb0", "code_search", {"query": "how are sidecar python processes started", "mode": "hybrid"}),
        ("hyb1", "code_search", {"query": "where is the oauth token refreshed", "mode": "hybrid"}),
        ("hyb2", "code_search", {"query": "rate limiting for outbound http requests", "mode": "hybrid"}),
        ("impact_sym", "impact_of_symbol", {"symbols": ["main"]}),
    ]
    for label, tool, args in calls:
        _, ms = m.call("tools/call", {"name": tool, "arguments": args})
        cur = m.io()
        print("%-14s %7.1f %9d %8.1f %8d %8.0f" % (label, ms, cur[0] - prev[0], (cur[1] - prev[1]) / 1e6, cur[2] - prev[2], cur[3] / 1e6))
        prev = cur
    m.close()


if __name__ == "__main__":
    main()
