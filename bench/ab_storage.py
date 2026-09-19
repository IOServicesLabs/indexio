#!/usr/bin/env python3
"""A/B the on-disk footprint and build time of a lexical index: index the same
repos with two binaries into fresh scratch data dirs (`add --no-embed`), then
report shards / cas sizes per section and the wall time.

    python bench/ab_storage.py --old ~/.cargo/bin/indexio.old.exe --new target/release/indexio.exe \
        --repos ~/code/repo-a ~/code/repo-b
"""
import argparse
import glob
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

KINDS = {1: "DOCS", 2: "STRINGS", 3: "CONTENT", 4: "NGRAM_FST", 5: "NGRAM_POST", 6: "SYM_FST", 7: "SYM_POST",
         8: "CALL_FST", 9: "CALL_POST", 10: "TOMBSTONES", 11: "META"}


def du(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def sections(shards_dir):
    tot = {}
    raw = 0
    for f in glob.glob(os.path.join(shards_dir, "*.cidx")):
        with open(f, "rb") as fh:
            b = fh.read(64 + 12 * 24)
            n = struct.unpack_from("<I", b, 12)[0]
            for i in range(n):
                k, _, off, ln = struct.unpack_from("<IIQQ", b, 64 + i * 24)
                tot[KINDS.get(k, k)] = tot.get(KINDS.get(k, k), 0) + ln
                if KINDS.get(k) == "DOCS":
                    fh.seek(off)
                    d = fh.read(ln)
                    raw += sum(struct.unpack_from("<I", d, j * 48 + 44)[0] for j in range(ln // 48))
    return tot, raw


def build(exe, repos, label):
    d = tempfile.mkdtemp(prefix=f"indexio-ab-{label}-")
    t0 = time.perf_counter()
    for r in repos:
        subprocess.run([exe, "add", r, "--data-dir", d, "--no-embed"], check=True, capture_output=True)
    secs = time.perf_counter() - t0
    tot, raw = sections(os.path.join(d, "shards"))
    out = {"label": label, "build_s": secs, "raw_text": raw, "shards": du(os.path.join(d, "shards")),
           "cas": du(os.path.join(d, "cas")), "sections": tot,
           "cas_files": sum(len(f) for _, _, f in os.walk(os.path.join(d, "cas")))}
    shutil.rmtree(d, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=os.path.expanduser("~/.cargo/bin/indexio.old.exe"))
    ap.add_argument("--new", default="target/release/indexio.exe")
    ap.add_argument("--repos", nargs="+", required=True)
    a = ap.parse_args()
    rows = [build(a.old, a.repos, "old"), build(a.new, a.repos, "new")]
    mb = lambda v: "%.1f" % (v / 1e6)
    print("%-12s %8s %10s %10s %10s %10s %10s" % ("", "build s", "raw MB", "shards MB", "cas MB", "cas files", "NGRAM_POST"))
    for r in rows:
        print("%-12s %8.1f %10s %10s %10s %10d %10s" % (r["label"], r["build_s"], mb(r["raw_text"]), mb(r["shards"]),
                                                          mb(r["cas"]), r["cas_files"], mb(r["sections"].get("NGRAM_POST", 0))))
    o, n = rows
    print("\nshards %s -> %s MB (%.1fx), cas %s -> %s MB (%.1fx), build %.1f -> %.1f s" % (
        mb(o["shards"]), mb(n["shards"]), o["shards"] / max(n["shards"], 1),
        mb(o["cas"]), mb(n["cas"]), o["cas"] / max(n["cas"], 1), o["build_s"], n["build_s"]))
    print("sections (MB):")
    for k in sorted(set(o["sections"]) | set(n["sections"]), key=lambda k: -o["sections"].get(k, 0)):
        print("  %-11s %8s -> %8s" % (k, mb(o["sections"].get(k, 0)), mb(n["sections"].get(k, 0))))


if __name__ == "__main__":
    main()
