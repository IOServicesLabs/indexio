#!/usr/bin/env python3
"""A/B of vector-row storage (SPEC-P10 int8 segments vs the f32 segments they
were built from): copies the live data dir's semantic plane, re-encodes the
copy with `compact --vectors`, then runs the same semantic + hybrid queries
against both and reports top-1 agreement, top-k overlap, size and latency.

    python bench/ab_quant.py --exe target/release/indexio.exe [--data-dir ~/.indexio] [--k 10]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

QUERIES = [
    "how are sidecar python processes started",
    "where is the oauth token refreshed",
    "how does the coordinator accept or reject a plan",
    "rate limiting for outbound http requests",
    "parse a unified diff into changed line ranges",
    "retry with exponential backoff on network errors",
    "read the config file and merge environment overrides",
    "websocket reconnect logic",
    "how are search results deduplicated by canonical url",
    "tree-sitter symbol extraction for python",
    "write a shard file atomically with a temp rename",
    "hamming distance prescan over binary codes",
    "spawn a worker task and wait for its result",
    "price feed subscription and update handling",
    "jwt authentication middleware",
    "database connection pool setup",
    "render markdown transcript of a session",
    "cache eviction when the vocabulary is full",
    "stream audio samples to the browser",
    "compute bm25 score for a document",
    "load a model from disk and warn if corrupt",
    "command line argument parsing with subcommands",
    "background thread that compacts segments",
    "convert crlf line endings before hashing",
    "expo react native breathing exercise screen",
    "fetch listings from a marketplace and parse prices",
    "kill a stale lock left by a crashed process",
    "hnsw graph search with ef parameter",
    "serialize the report as json for the http api",
    "unit test for the query parser",
]


def run(exe, data_dir, q, mode, k):
    t0 = time.perf_counter()
    r = subprocess.run([exe, "search", q, "--mode", mode, "--limit", str(k), "--json", "--data-dir", data_dir],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ms = (time.perf_counter() - t0) * 1000
    try:
        hits = json.loads(r.stdout)
        if isinstance(hits, dict):
            hits = hits.get("hits", [])
        hits = [h.get("hit", h) for h in hits]  # hybrid wraps the hit with its ranks
        keys = [(h.get("repo"), h.get("path"), h.get("line")) for h in hits]
    except ValueError:
        keys = []
    return keys, ms


def du(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="target/release/indexio.exe")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/.indexio"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    # two copies of the semantic plane, each folded into ONE segment: f32
    # (INDEXIO_VEC_F32) and int8 — so the only difference is the row format
    dirs = {}
    for label, env in [("f32", {"INDEXIO_VEC_F32": "1"}), ("int8", {})]:
        tmp = tempfile.mkdtemp(prefix=f"indexio-quant-{label}-")
        for sub in ["vec", "bm25", "sem", "shards", "repos", "embcas"]:
            src = os.path.join(a.data_dir, sub)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(tmp, sub))
        shutil.copy(os.path.join(a.data_dir, "sources.json"), tmp)
        t0 = time.perf_counter()
        out = subprocess.run([a.exe, "compact", "--vectors", "--max-shards", "999", "--data-dir", tmp],
                             capture_output=True, text=True, env={**os.environ, **env})
        print(label + ":", out.stdout.strip().splitlines()[-1], "(%.1f s)" % (time.perf_counter() - t0))
        print("  vec/: %.0f MB" % (du(os.path.join(tmp, "vec")) / 1e6))
        dirs[label] = tmp
    for mode in ["semantic", "hybrid"]:
        top1 = overlap = n = 0
        ms_a = ms_b = 0.0
        for q in QUERIES:
            ka, ta = run(a.exe, dirs["f32"], q, mode, a.k)
            kb, tb = run(a.exe, dirs["int8"], q, mode, a.k)
            if not ka or not kb:
                continue
            n += 1
            top1 += ka[0] == kb[0]
            overlap += len(set(ka) & set(kb)) / max(len(ka), 1)
            ms_a += ta
            ms_b += tb
        print("%-9s queries=%d  top1 agree=%d/%d  top%d overlap=%.1f%%  ms f32=%.0f int8=%.0f (whole CLI runs)" % (
            mode, n, top1, n, a.k, 100 * overlap / max(n, 1), ms_a / max(n, 1), ms_b / max(n, 1)))
    for tmp in dirs.values():
        if a.keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
