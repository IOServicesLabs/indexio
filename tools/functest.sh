#!/usr/bin/env bash
# Functional test of the whole `indexio` binary against real repos (bash; works in
# Git Bash on Windows). Exercises: index, duplicate rejection, stats, every
# lexical query form, symbol/calls, embed (cold + warm), semantic/hybrid/
# rerank, delta re-index (modify/add/delete), no-op re-index, compaction,
# impact/outline/span, HTTP with ACLs, MCP over stdio.
#
#   tools/functest.sh <repo-path> [<repo-path> ...]
#
# The FIRST repo is copied to a scratch dir and committed to (delta test);
# the others are only read. Needs: git, python3 (or python), curl.
# Data dir: $FT_DIR (default: a fresh temp dir). Exit 1 on any FAIL.
set -u
export MSYS_NO_PATHCONV=1
# `pwd -W` gives a Windows path under MSYS/Git Bash (python needs it); plain pwd elsewhere.
HERE="$(cd "$(dirname "$0")/.." && (pwd -W 2>/dev/null || pwd))"
CI="${INDEXIO_BIN:-$HERE/target/release/indexio}"
[ -x "$CI.exe" ] && CI="$CI.exe"   # Windows: python's CreateProcess needs the extension
[ -x "$CI" ] || { echo "build first: cargo build --release"; exit 2; }
PY="$(command -v python3 || command -v python)"
[ $# -ge 1 ] || { echo "usage: $0 <repo-path> [more repos]"; exit 2; }

FT="${FT_DIR:-$(mktemp -d)}"; mkdir -p "$FT"
export INDEXIO_DATA_DIR="$FT/data"; rm -rf "$INDEXIO_DATA_DIR"
export GIT_AUTHOR_NAME=functest GIT_AUTHOR_EMAIL=f@t GIT_COMMITTER_NAME=functest GIT_COMMITTER_EMAIL=f@t
FAILS=0
pass(){ echo "PASS  $1"; }
fail(){ echo "FAIL  $1"; FAILS=$((FAILS+1)); }
check(){ if "$@" >/dev/null 2>&1; then pass "$NAME"; else fail "$NAME"; fi; }
py(){ "$PY" -c "$@"; }

FIRST="$1"; shift
FIRST_NAME="$(basename "$FIRST")"
K="$FT/$FIRST_NAME"; rm -rf "$K"; cp -r "$FIRST" "$K"
git -C "$K" rev-parse HEAD >/dev/null 2>&1 || { echo "$FIRST is not a git repo"; exit 2; }

echo "### index"
for r in "$FIRST" "$@"; do
  n="$(basename "$r")"; src="$r"; [ "$r" = "$FIRST" ] && src="$K"
  out="$("$CI" index "$src" --name "$n" 2>&1)"
  echo "$out" | grep -q "docs:" && pass "index $n ($(echo "$out" | grep elapsed | tr -s ' '))" || { fail "index $n"; echo "$out"; }
done
NAME="duplicate registration rejected"; "$CI" index "$K" --name "$FIRST_NAME" >/dev/null 2>&1 && fail "$NAME" || pass "$NAME"
NAME="stats json"; check sh -c "'$CI' stats --json | '$PY' -c 'import json,sys; s=json.load(sys.stdin); assert s[\"doc_count\"]>0 and s[\"shard_count\"]>0'"

# A symbol + a file to probe: the most-defined-in file of the first repo.
PROBE="$("$CI" search "/def |fn |function |class / repo:$FIRST_NAME" --json --limit 200 | "$PY" -c "
import json,sys,collections
h=json.load(sys.stdin)['hits']; c=collections.Counter(x['path'] for x in h); print(c.most_common(1)[0][0] if c else '')")"
[ -n "$PROBE" ] || { echo "no probe file found in $FIRST_NAME"; exit 2; }
SYM="$("$CI" outline "$FIRST_NAME:$PROBE" --json | "$PY" -c "
import json,sys; it=[i for i in json.load(sys.stdin)['items'] if i['kind'] in ('Fn','Method','Class','Struct') and len(i['name'])>3]; print(it[0]['name'] if it else '')")"
echo "probe: $FIRST_NAME:$PROBE symbol=$SYM"

echo "### lexical search"
NAME="literal"; check sh -c "'$CI' search '$SYM' --json | '$PY' -c 'import json,sys; assert json.load(sys.stdin)[\"hits\"]'"
NAME="regex + repo filter"; check sh -c "'$CI' search '/\\b$SYM\\b/ repo:$FIRST_NAME' --json | '$PY' -c 'import json,sys; h=json.load(sys.stdin)[\"hits\"]; assert h and all(x[\"repo\"]==\"$FIRST_NAME\" for x in h)'"
NAME="phrase"; check sh -c "'$CI' search '\"$SYM\"' --json | '$PY' -c 'import json,sys; assert json.load(sys.stdin)[\"hits\"]'"
NAME="case:no"; check sh -c "'$CI' search '$(echo "$SYM" | tr A-Z a-z) case:no' --json | '$PY' -c 'import json,sys; assert json.load(sys.stdin)[\"hits\"]'"
NAME="empty query rejected"; "$CI" search "" >/dev/null 2>&1 && fail "$NAME" || pass "$NAME"
NAME="bad regex rejected"; "$CI" search "/(bad/" >/dev/null 2>&1 && fail "$NAME" || pass "$NAME"

echo "### symbol + calls"
NAME="symbol exact"; check sh -c "'$CI' symbol '$SYM' | grep -q ."
NAME="symbol substring fallback"; check sh -c "'$CI' symbol '${SYM:0:$((${#SYM}-1))}' | grep -q ."
"$CI" calls "$SYM" | head -3

echo "### embed + semantic + hybrid (cold embed can take minutes on large repos)"
"$CI" embed --all | tail -n +1
NAME="embcas stats"; check sh -c "'$CI' embcas-stats --json | '$PY' -c 'import json,sys; assert json.load(sys.stdin)[\"entries\"]>0'"
NAME="semantic"; check sh -c "'$CI' search 'where is $SYM defined and used' --mode semantic --limit 5 --json | '$PY' -c 'import json,sys; assert json.load(sys.stdin)[\"hits\"]'"
NAME="hybrid"; check sh -c "'$CI' search '$SYM' --mode hybrid --limit 5 --json | '$PY' -c 'import json,sys; h=json.load(sys.stdin)[\"hits\"]; assert h and h[0][\"rrf\"]>0'"
NAME="hybrid combmnz"; check sh -c "'$CI' search '$SYM' --mode hybrid --fusion combmnz --limit 3 --json | '$PY' -c 'import json,sys; assert json.load(sys.stdin)[\"hits\"]'"
NAME="hybrid rerank"; check sh -c "'$CI' search '$SYM' --mode hybrid --rerank --limit 3 --json | '$PY' -c 'import json,sys; h=json.load(sys.stdin)[\"hits\"]; assert h and h[0][\"rerank_score\"] is not None'"
NAME="rerank requires hybrid"; "$CI" search x --rerank >/dev/null 2>&1 && fail "$NAME" || pass "$NAME"
NAME="warm re-embed computes 0 vectors"; check sh -c "'$CI' embed --all | awk 'NR>1 && \$2 ~ /^[0-9]+$/ && \$5!=0 {bad=1} END {exit bad}'"

echo "### delta re-index"
printf '\n# functest delta\n' >> "$K/$PROBE"
printf 'def functest_brand_new_fn():\n    return 42\n' > "$K/functest_newmod.py"
DEL="$(git -C "$K" ls-files | grep -v "^$PROBE$" | grep -E '\.(py|rs|ts|tsx|js|go|java)$' | head -1)"
[ -n "$DEL" ] && git -C "$K" rm -q "$DEL"
git -C "$K" add -A && git -C "$K" -c commit.gpgsign=false commit -qm functest
out="$("$CI" reindex --repo "$FIRST_NAME" 2>&1)"; echo "$out" | grep -E "docs:|elapsed"
NAME="delta counts (2 added, 1-2 deleted)"; echo "$out" | grep -Eq "2 added, [12] deleted" && pass "$NAME" || fail "$NAME"
NAME="new file searchable"; check sh -c "'$CI' symbol functest_brand_new_fn | grep -q functest_newmod.py"
if [ -n "$DEL" ]; then NAME="deleted file tombstoned"; check sh -c "'$CI' search 'path:$DEL $(basename "$DEL" | cut -d. -f1)' --json | '$PY' -c 'import json,sys; assert all(h[\"path\"]!=\"$DEL\" for h in json.load(sys.stdin)[\"hits\"])'"; fi
NAME="no-op re-index"; check sh -c "'$CI' reindex --repo '$FIRST_NAME' | grep -q '0 added, 0 deleted'"
NAME="embed after delta only computes new chunks"; check sh -c "'$CI' embed --repo '$FIRST_NAME' | awk 'NR>1 && \$2 ~ /^[0-9]+$/ && \$5>50 {bad=1} END {exit bad}'"

echo "### compact"
"$CI" compact --max-shards 2
NAME="search after compact"; check sh -c "'$CI' search '$SYM' --json | '$PY' -c 'import json,sys; assert json.load(sys.stdin)[\"hits\"]'"
NAME="symbols after compact"; check sh -c "'$CI' symbol functest_brand_new_fn | grep -q functest_newmod.py"

echo "### impact / outline / span"
NAME="outline"; check sh -c "'$CI' outline '$FIRST_NAME:$PROBE' --json | '$PY' -c 'import json,sys; assert json.load(sys.stdin)[\"items\"]'"
NAME="span"; check sh -c "'$CI' span '$FIRST_NAME:$PROBE' 1:5 | grep -q ."
NAME="impact --symbol"; check sh -c "'$CI' impact --symbol '$SYM' --depth 2 --json | '$PY' -c 'import json,sys; r=json.load(sys.stdin); assert r[\"roots\"]==[\"$SYM\"]'"
NAME="impact --file"; check sh -c "'$CI' impact --file '$FIRST_NAME:$PROBE' --json | '$PY' -c 'import json,sys; r=json.load(sys.stdin); assert r[\"roots\"]'"
# uncommitted working-tree edit inside the first definition of the probe file
LINE="$("$CI" outline "$FIRST_NAME:$PROBE" --json | "$PY" -c "
import json,sys; it=[i for i in json.load(sys.stdin)['items'] if i['kind'] in ('Fn','Method') and i['end_line']>i['start_line']]; print(it[0]['start_line']+1 if it else 0)")"
if [ "$LINE" -gt 0 ]; then
  "$PY" -c "
import sys; p=sys.argv[1]; n=int(sys.argv[2]); L=open(p,encoding='utf-8',errors='surrogateescape').read().split('\n'); L.insert(n-1, L[n-1][:len(L[n-1])-len(L[n-1].lstrip())]+'# functest wt edit'); open(p,'w',encoding='utf-8',errors='surrogateescape').write('\n'.join(L))" "$K/$PROBE" "$LINE"
  NAME="impact --diff maps working-tree edit to a definition"; check sh -c "'$CI' impact --diff '$FIRST_NAME' --json | '$PY' -c 'import json,sys; r=json.load(sys.stdin); assert r[\"changed\"], r'"
fi

echo "### HTTP with ACL"
cat > "$FT/acl.json" <<EOF
{"tokens":{"reader":{"allow":["$FIRST_NAME"]},"admin":{"allow":["*"]}}}
EOF
PORT=$((20000 + RANDOM % 20000))
"$CI" serve --port "$PORT" --acl-file "$FT/acl.json" > "$FT/serve.log" 2>&1 &
SP=$!
for _ in $(seq 1 60); do curl -s -o /dev/null "http://127.0.0.1:$PORT/health" && break; sleep 0.25; done
"$PY" - "$PORT" "$FIRST_NAME" "$PROBE" "$SYM" <<'EOF'
import urllib.request, json, sys
port,repo,probe,sym=sys.argv[1:5]; B=f"http://127.0.0.1:{port}"
def call(path, tok=None, data=None):
    hdr={};
    if tok: hdr["Authorization"]="Bearer "+tok
    if data is not None: hdr["Content-Type"]="application/json"
    req=urllib.request.Request(B+path, data=(json.dumps(data).encode() if data is not None else None), headers=hdr, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, None
fails=0
def chk(n,c):
    global fails; print(("PASS  " if c else "FAIL  ")+n); fails+= (not c)
s,b=call("/health"); chk("/health open", s==200)
s,_=call(f"/search?q={sym}"); chk("no token -> 401", s==401)
s,b=call(f"/search?q={sym}","reader"); chk("reader sees only own repo", s==200 and all(h["repo"]==repo for h in b["hits"]))
s,b=call(f"/search?q={sym}&mode=hybrid&limit=3","admin"); chk("hybrid over HTTP", s==200 and b["hits"] and "rrf" in b["hits"][0])
s,_=call("/search?q=x&rerank=1","admin"); chk("rerank w/o hybrid -> 400", s==400)
s,b=call(f"/symbol/{sym}","admin"); chk("/symbol", s==200 and b)
s,b=call("/stats","admin"); chk("/stats", s==200 and b["doc_count"]>0)
s,_=call("/admin/reload","reader",{}); chk("reader admin -> 403", s==403)
s,b=call("/admin/reload","admin",{}); chk("admin reload", s==200)
s,b=call(f"/impact/symbol?name={sym}","admin"); chk("/impact/symbol", s==200 and b["roots"]==[sym])
s,b=call(f"/outline?repo={repo}&path={probe}","reader"); chk("/outline", s==200 and b["items"])
s,_=call("/outline?repo=__other__&path=x","reader"); chk("/outline outside ACL -> 403", s==403)
s,b=call(f"/span?repo={repo}&path={probe}&start=1&end=3","reader"); chk("/span", s==200 and b["text"])
s,b=call("/impact/diff","admin",{"repo":repo,"diff":"","depth":1}); chk("POST /impact/diff", s==200 and "sites" in b)
sys.exit(fails)
EOF
[ $? -eq 0 ] || FAILS=$((FAILS+1))
kill $SP 2>/dev/null; wait $SP 2>/dev/null

echo "### MCP"
"$PY" - "$CI" "$FIRST_NAME" "$PROBE" "$SYM" <<'EOF'
import subprocess, json, sys
ci,repo,probe,sym=sys.argv[1:5]
calls=[("code_search",{"query":sym,"limit":3}),("code_search",{"query":sym,"mode":"hybrid","limit":3}),("semantic_search",{"query":sym,"k":3}),("code_grep",{"pattern":sym,"limit":3}),("find_symbol",{"name":sym}),("who_calls",{"name":sym}),("index_stats",{}),("impact_of_symbol",{"names":[sym],"depth":2}),("impact_of_diff",{"repo":repo,"depth":1}),("file_outline",{"repo":repo,"path":probe}),("read_span",{"repo":repo,"path":probe,"start":1,"end":3})]
msgs=[{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"0"}}},{"jsonrpc":"2.0","method":"notifications/initialized"},{"jsonrpc":"2.0","id":2,"method":"tools/list"}]
for i,(n,a) in enumerate(calls): msgs.append({"jsonrpc":"2.0","id":10+i,"method":"tools/call","params":{"name":n,"arguments":a}})
p=subprocess.run([ci,"mcp"],input=("\n".join(json.dumps(m) for m in msgs)+"\n").encode(),capture_output=True)
resp={}
for line in p.stdout.decode("utf-8",errors="replace").splitlines():
    v=json.loads(line); resp[v["id"]]=v
fails=0
def chk(n,c):
    global fails; print(("PASS  " if c else "FAIL  ")+n); fails+=(not c)
chk("initialize", resp.get(1,{}).get("result",{}).get("serverInfo",{}).get("name")=="indexio")
chk("tools/list = 10 tools", len(resp.get(2,{}).get("result",{}).get("tools",[]))==10)
for i,(n,a) in enumerate(calls):
    v=resp.get(10+i); ok = v is not None and "error" not in v and json.loads(v["result"]["content"][0]["text"]) is not None
    chk("tool %s"%n, ok)
chk("no stderr noise", not p.stderr.strip())
sys.exit(fails)
EOF
[ $? -eq 0 ] || FAILS=$((FAILS+1))

echo
if [ "$FAILS" -eq 0 ]; then echo "ALL PASS  (data dir: $INDEXIO_DATA_DIR)"; else echo "$FAILS FAILURE(S)  (data dir: $INDEXIO_DATA_DIR)"; exit 1; fi
