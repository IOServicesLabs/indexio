#!/usr/bin/env python3
"""Known-answer questions for the user's OWN repos, written by an LLM
(SPEC-P10 §14): sample code windows from the registered repos, ask the model
for the natural-language question a developer would type to find that code
(no identifiers copied verbatim), and save `{q, repo, path_suffix}` rows that
`eval_modes.py` scores as MRR / recall@k per search mode.

Any OpenAI-compatible chat endpoint. Configuration by environment only — the
key never touches the repo:

    MODEL_API_KEY            required
    MODEL_API_BASE           default https://api.openai.com   (/v1/chat/completions is appended)
    MODEL_NAME               default gpt-4o-mini
    MODEL_REASONING_EFFORT   optional (minimal|low|medium|...); unset sends nothing

    python bench/gen_questions.py --data-dir ~/.indexio --n 80 --out bench/known_answers_local.json
"""
import argparse
import glob
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

BASE = (os.environ.get("MODEL_API_BASE") or "https://api.openai.com").rstrip("/")
API = BASE if BASE.endswith("/chat/completions") else BASE + "/v1/chat/completions"
MODEL = os.environ.get("MODEL_NAME") or "gpt-4o-mini"
SKIP_DIRS = {"node_modules", "target", "dist", "build", "vendor", ".git", "__pycache__", ".venv", "venv", "bench-work"}
EXTS = {".rs", ".py", ".ts", ".tsx", ".js", ".go", ".java", ".cpp", ".c", ".h", ".cs"}

SYSTEM = (
    "You write search queries for a code search engine benchmark. Given a code excerpt, "
    "reply with JSON {\"q\": ..., \"why\": ...}. `q` is the natural-language question a "
    "developer who has NOT seen this code would type to find it: 8-20 words, describes what "
    "the code does or handles, no function/type/variable names copied verbatim, no file "
    "names. `why` is one sentence naming what in the excerpt answers it."
)


def ask(key: str, user: str, max_tokens: int = 1500) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
    }
    # Muse Spark reasons by default and max_tokens is shared with the
    # reasoning: a short budget returns EMPTY content. Keep reasoning minimal
    # for this task unless told otherwise.
    effort = os.environ.get("MODEL_REASONING_EFFORT") or ""
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(API, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    msg = data["choices"][0].get("message") or {}
    text = (msg.get("content") or "").strip()
    if not text:
        raise RuntimeError("empty content (finish_reason=%s)" % data["choices"][0].get("finish_reason"))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        return json.loads(text[s:e + 1])


def repos(data_dir):
    out = []
    for f in glob.glob(os.path.join(data_dir, "repos", "*.json")):
        try:
            st = json.load(open(f, encoding="utf-8"))
        except ValueError:
            continue
        if st.get("name") == "sessions" or not os.path.isdir(st.get("path", "")):
            continue
        out.append((st["name"], st["path"]))
    return out


def code_files(root, limit=4000):
    files = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in fns:
            if os.path.splitext(fn)[1] in EXTS:
                p = os.path.join(dp, fn)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                if 2000 <= sz <= 120_000:
                    files.append(p)
        if len(files) > limit:
            break
    return files


def window(path, rng):
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    if len(lines) < 30:
        return None
    n = rng.randint(30, 70)
    start = rng.randint(0, max(0, len(lines) - n))
    body = "\n".join(lines[start:start + n])
    if len(body.strip()) < 400:
        return None
    return start + 1, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/.indexio"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="bench/known_answers_local.json")
    ap.add_argument("--repos", nargs="*", help="restrict to these registered repos")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        sys.exit("MODEL_API_KEY is not set")
    rng = random.Random(a.seed)
    cands = [(n, p) for n, p in repos(a.data_dir) if not a.repos or n in a.repos]
    if not cands:
        sys.exit("no registered repos found")
    rows = []
    if os.path.exists(a.out):
        rows = json.load(open(a.out, encoding="utf-8"))
        print("continuing", a.out, "with", len(rows), "rows")
    tries = 0
    while len(rows) < a.n and tries < a.n * 4:
        tries += 1
        name, root = rng.choice(cands)
        files = code_files(root)
        if not files:
            continue
        path = rng.choice(files)
        w = window(path, rng)
        if not w:
            continue
        start, body = w
        rel = os.path.relpath(path, root).replace("\\", "/")
        user = f"Repository: {name}\nFile: {rel} (lines {start}-{start + body.count(chr(10))})\n\n```\n{body}\n```"
        try:
            t0 = time.perf_counter()
            r = ask(key, user)
            q = (r.get("q") or "").strip()
            if not (20 <= len(q) <= 200):
                continue
            rows.append({"q": q, "repo": name, "path_suffix": rel, "line": start, "why": (r.get("why") or "")[:200]})
            print("%3d %5.1fs %-20s %-45s %s" % (len(rows), time.perf_counter() - t0, name, rel[-45:], q[:70]))
            json.dump(rows, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, ValueError, KeyError) as e:
            detail = e.read().decode("utf-8", "replace")[:300] if isinstance(e, urllib.error.HTTPError) else str(e)
            print("  skip:", detail)
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403):
                sys.exit("authentication failed")
            time.sleep(1)
    print("wrote", len(rows), "questions to", a.out)


if __name__ == "__main__":
    main()
