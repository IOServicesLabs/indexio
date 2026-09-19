#!/usr/bin/env python3
"""eval_modes.py — known-answer evaluation of the indexio engine's search modes.

For each natural-language concept query in known_answers.json and each search
mode, runs:

    indexio search "<q>" --mode M --limit N --json --data-dir D

and finds the first rank (1-based) at which a hit's repo matches the query's
repo and the hit's path ends with the query's path_suffix. Prints a per-query
rank table, per-mode MRR and recall@5, and a final comparison table.

Usage:
    python3 eval_modes.py --ci <path-to-ci-binary> --data-dir <dir> \
        [--modes lexical,semantic,hybrid] [--limit 10] [--answers known_answers.json]

Stdlib only. Search failures (nonzero exit, unparseable JSON) are reported as
rank None and never abort the run.
"""

import argparse
import json
import re
import os
import subprocess
import sys

DEFAULT_MODES = ["lexical", "semantic", "hybrid"]
# Hits from these repos never count (the MCP server drops transcript hits
# from code searches; the CLI does not).
SKIP_REPOS = {"sessions"}


def load_answers(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("known_answers.json must be a JSON list")
    for i, item in enumerate(data):
        for key in ("q", "repo", "path_suffix"):
            if key not in item:
                raise ValueError("entry %d missing key %r" % (i, key))
    return data


def extract_hits(payload):
    """Pull the hit list out of a parsed JSON payload.

    Tolerates:
      - {"hits": [...]}                      (lexical / semantic / hybrid CLI)
      - a bare top-level list of hits
    Each hit may be a flat SearchHit or a hybrid envelope {"hit": {...}, ...}.
    Returns a list of flat hit dicts.
    """
    if isinstance(payload, dict):
        raw = payload.get("hits", [])
    elif isinstance(payload, list):
        raw = payload
    else:
        return []
    hits = []
    for h in raw:
        if not isinstance(h, dict):
            continue
        if isinstance(h.get("hit"), dict):  # hybrid envelope
            hits.append(h["hit"])
        else:
            hits.append(h)
    return hits


STOPWORDS = {
    "how", "does", "do", "is", "are", "the", "a", "an", "of", "in", "inside",
    "to", "for", "and", "or", "with", "that", "this", "when", "where", "what",
    "which", "it", "its", "on", "by", "be", "can", "capable", "actually",
    "implemented", "implementation", "code", "like", "while", "during",
    "decide", "decides", "support", "efficiently", "arbitrary", "user",
}


def keywordize(query):
    """Extract content terms for lexical mode (agents query lexical with
    identifiers/keywords, not full NL sentences — the parser ANDs literals).
    Keeps code-ish tokens like #[serde(...)] fragments and $1/$name."""
    toks = re.findall(r"[A-Za-z0-9_#$]{2,}", query.lower())
    kept = [t for t in toks if t not in STOPWORDS and not t.isdigit()]
    return " ".join(kept) if kept else query


EXTRA_ARGS = []


def run_search(ci, data_dir, query, mode, limit, timeout=120):
    """Run one search; return (hits, error). error is None on success."""
    if mode == "lexical":
        query = keywordize(query)
    cmd = [
        ci, "search", query,
        "--mode", mode,
        "--limit", str(limit),
        "--json",
        "--data-dir", data_dir,
    ] + (EXTRA_ARGS if mode == "hybrid" else [])
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",  # snippets are arbitrary bytes; Windows defaults to cp1252
        )
    except subprocess.TimeoutExpired:
        return [], "timeout after %ds" % timeout
    except OSError as e:
        return [], "exec failed: %s" % e
    out = proc.stdout.strip()
    if not out:
        err = proc.stderr.strip().splitlines()
        return [], "no output (rc=%d)%s" % (
            proc.returncode, (": " + err[-1]) if err else "")
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        # Some tools emit log lines before the JSON; try the last JSON-looking line.
        payload = None
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{") or line.startswith("["):
                try:
                    payload = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if payload is None:
            return [], "unparseable JSON (rc=%d)" % proc.returncode
    hits = [h for h in extract_hits(payload) if h.get("repo") not in SKIP_REPOS]
    if proc.returncode != 0:
        # Still try to score whatever hits we got, but record the failure.
        return hits, "rc=%d" % proc.returncode
    return hits, None


def hit_matches(hit, repo, path_suffix):
    h_repo = hit.get("repo")
    h_path = hit.get("path")
    if not isinstance(h_repo, str) or not isinstance(h_path, str):
        return False
    if h_repo != repo:
        return False
    norm = h_path.replace(os.sep, "/")
    return norm.endswith(path_suffix)


def first_rank(hits, repo, path_suffix):
    for i, hit in enumerate(hits, start=1):
        if hit_matches(hit, repo, path_suffix):
            return i
    return None


def fmt_rank(rank, err):
    if rank is not None:
        return str(rank)
    return "miss(%s)" % err if err else "miss"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ci", required=True, help="path to the ci binary")
    ap.add_argument("--data-dir", required=True, help="engine data directory")
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES),
                    help="comma-separated modes (default: lexical,semantic,hybrid)")
    ap.add_argument("--limit", type=int, default=10, help="search hit limit")
    ap.add_argument("--repo-scope", action="store_true",
                    help="append ` repo:<repo>` to each query (the session's repo is known to the MCP server)")
    ap.add_argument("--extra", default="", help="extra CLI flags for hybrid runs, e.g. \"--fusion combmnz\" or \"--rerank\"")
    ap.add_argument("--answers",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "known_answers.json"),
                    help="path to known_answers.json")
    args = ap.parse_args()
    EXTRA_ARGS[:] = args.extra.split()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        print("error: no modes given", file=sys.stderr)
        return 2
    try:
        answers = load_answers(args.answers)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print("error: cannot load %s: %s" % (args.answers, e), file=sys.stderr)
        return 2

    # results[mode] = list of (query, rank, error)
    results = {m: [] for m in modes}
    for qi, qa in enumerate(answers):
        for mode in modes:
            q = qa["q"] + (" repo:%s" % qa["repo"] if args.repo_scope else "")
            hits, err = run_search(args.ci, args.data_dir, q, mode, args.limit)
            rank = first_rank(hits, qa["repo"], qa["path_suffix"])
            results[mode].append((qa, rank, err))
            sys.stderr.write(
                "\r[%d/%d] %-8s rank=%s\033[K" % (
                    qi + 1, len(answers), mode, fmt_rank(rank, err)))
            sys.stderr.flush()
    sys.stderr.write("\n")

    # Per-query rank table (queries as rows, modes as columns).
    qwidth = max(len(qa["q"]) for qa in answers)
    qwidth = min(qwidth, 72)
    header = "  ".join(["query".ljust(qwidth)] + [m.rjust(12) for m in modes])
    print(header)
    print("-" * len(header))
    for qi, qa in enumerate(answers):
        q = qa["q"]
        if len(q) > qwidth:
            q = q[: qwidth - 1] + "…"
        cells = []
        for mode in modes:
            _, rank, err = results[mode][qi]
            cells.append(fmt_rank(rank, err).rjust(12))
        print("  ".join([q.ljust(qwidth)] + cells))
    print()

    # Per-mode metrics.
    def mrr(ranks):
        vals = [1.0 / r for r in ranks if r is not None]
        return sum(vals) / len(ranks) if ranks else 0.0

    def recall_at(ranks, k):
        if not ranks:
            return 0.0
        return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)

    print("mode".ljust(12) + "MRR".rjust(8) + "recall@5".rjust(12)
          + "recall@10".rjust(12) + "errors".rjust(9))
    print("-" * 53)
    summary = {}
    for mode in modes:
        ranks = [rank for _, rank, _ in results[mode]]
        errs = sum(1 for _, _, e in results[mode] if e)
        summary[mode] = (mrr(ranks), recall_at(ranks, 5), recall_at(ranks, 10), errs)
        print(mode.ljust(12)
              + ("%.3f" % summary[mode][0]).rjust(8)
              + ("%.3f" % summary[mode][1]).rjust(12)
              + ("%.3f" % summary[mode][2]).rjust(12)
              + str(errs).rjust(9))
    print()

    # Final comparison table.
    print("=== comparison ===")
    print("mode".ljust(12) + "MRR".rjust(8) + "recall@5".rjust(12))
    best_mrr = max((v[0] for v in summary.values()), default=0.0)
    for mode in modes:
        mark = " *" if summary[mode][0] == best_mrr and best_mrr > 0 else ""
        print(mode.ljust(12) + ("%.3f" % summary[mode][0]).rjust(8)
              + ("%.3f" % summary[mode][1]).rjust(12) + mark)
    return 0


if __name__ == "__main__":
    sys.exit(main())
