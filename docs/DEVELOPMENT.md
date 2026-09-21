# Development guide

How to build, test and benchmark indexio, how to plug in real embedding and reranker
models, and where to continue development. Read `README.md` for the user guide; this
document is the operator manual.

## 0. What you have

A single Cargo workspace, 8 crates, about 30k lines of Rust, 303 tests, no external runtime
services. Everything lives in one binary (`indexio`) + one data directory.

| Crate | Owns | Key files |
|---|---|---|
| `indexio-types` | Shared types, posting-list codec (LEB128 varints, 128-doc blocks) | `src/lib.rs`, `src/codec.rs` |
| `indexio-core` | Sparse n-gram extraction (`grams`), literal/regex verify, required-literal extraction | `src/lib.rs` |
| `indexio-index` | Immutable mmap'd shards (`.cidx`), tombstones, compound merge | `src/lib.rs` |
| `indexio-symbols` | tree-sitter parsers for 6 languages; symbol + call-edge extraction; cAST-lite chunking | `src/lib.rs`, `src/chunking.rs` |
| `indexio-ingest` | Global CAS, git HEAD-tree/diff delta indexing, GitHub `org_sync` | `src/lib.rs`, `src/cas.rs`, `src/org_sync.rs` |
| `indexio-embed` | Embedders (`RandomIndexing` default / `Hash` / `Http`), embedding CAS, vector index (flat+BQ / HNSW), rerankers, embed pipeline | `src/rindex.rs`, `src/embed.rs`, `src/store.rs`, `src/index.rs`, `src/hnsw.rs`, `src/rerank.rs`, `src/pipeline.rs` |
| `indexio-query` | Query parser, planner, BM25-lite ranking, RRF hybrid fusion + rerank stage | `src/lib.rs` |
| `indexio` | CLI (clap), HTTP (axum), MCP (stdio JSON-RPC) | `src/main.rs`, `src/serve.rs`, `src/mcp.rs` |

Design contracts: `SPEC.md` (P0/P1), `SPEC-P2.md` (semantic plane + org-sync), `SPEC-P3.md`
(evaluation), `SPEC-P6.md` (impact analysis + perf), `SPEC-P7.md` (sources / onboarding),
`SPEC-P8.md` (harness integration), `SPEC-P9.md` and `SPEC-P10.md` (the agent-facing tools,
hooks and their measurements), `SPEC-P11.md` (central index design). The `research-p5/`
folder holds the lexical and vector-index studies.

## 1. Local setup

```bash
# toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # or your package manager
rustup default stable          # built/tested on 1.98; any recent stable works

# build + full test suite (first build ~5–10 min; tests ~2–5 min)
cd indexio
cargo build --release
cargo test --workspace         # expect: 236 passed, 0 failed
```

You do **not** need: Docker, Postgres, Qdrant, any model weights, or any network service
for the core system. The default offline embedder/reranker let the whole pipeline run
air-gapped; real models are external HTTP endpoints you add later (§4).

## 2. Smoke-test checklist (10 minutes)

Automated version of everything below plus delta re-index, compaction, ACL'd HTTP and a
full MCP tool sweep — 60 checks against real repos of yours (the first repo is copied to a
scratch dir and committed to; the others are only read):

```bash
tools/functest.sh ~/src/some-small-repo ~/src/another-repo     # expect: ALL PASS
```


```bash
CI=target/release/indexio
export INDEXIO_DATA_DIR=$HOME/.indexio

# 2.1 index a real repo (uses your local clones)
git clone --depth 1 https://github.com/BurntSushi/ripgrep /tmp/ripgrep
$CI index /tmp/ripgrep --name ripgrep

# 2.2 lexical queries — expect ms latencies printed per query
$CI search "WalkState" --json | head -30
$CI search "/fn\s+parse_/" --limit 5
$CI search 'error lang:rust' --limit 5

# 2.3 symbols + call graph
$CI symbol Searcher
$CI calls "Printer::write"

# 2.4 delta re-index — make a commit, re-run, expect ~1 doc changed in <1s
cd /tmp/ripgrep && git checkout -b test && echo "// hi" >> crates/core/main.rs \
  && git commit -aqm test && cd -
$CI reindex ripgrep

# 2.5 semantic plane (Random Indexing — CPU-only, in-process, no downloads)
$CI embed --all                 # builds + trains rindex-v1 model from YOUR corpus
$CI embcas-stats
$CI search "how does the printer buffer output" --mode semantic --limit 5
$CI search "how does the printer buffer output" --mode hybrid --rerank --limit 5

# 2.6 HTTP + auth
$CI serve --port 7717 &
curl 'http://127.0.0.1:7717/search?q=printer&mode=hybrid&limit=3'
kill %1

# 2.7 impact analysis (SPEC-P6) — what does a change touch?
$CI outline ripgrep:crates/core/flags/parse.rs         # skeleton with line ranges
$CI span ripgrep:crates/core/flags/parse.rs 1:40       # exact lines from the index
$CI impact --symbol parse --depth 2                    # transitive callers, all repos
$CI impact --diff ripgrep                              # the uncommitted edit from 2.4

# 2.8 MCP handshake (what Claude Code will do)
printf '%s\n%s\n' \
 '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | $CI mcp | python3 -m json.tool | grep '"name"'
```

If 2.1–2.8 all behave, your build is healthy. Expected index size: ~1–3× the source size
(shard + CAS on disk); queries in the single/double-digit ms range on a laptop.

## 3. Pulling real code: sources (SPEC-P7)

```bash
$CI add ~/code                            # all git repos under it, any depth; plain folders too
export GITHUB_TOKEN=ghp_...               # read scope is enough; public repos need none
$CI add github:YOUR_ORG --limit 10        # trial: 10 repos first
$CI add github:YOUR_ORG                   # then the whole org (or a user, or OWNER/REPO)
export AZDO_TOKEN=...                     # Azure DevOps PAT with Code: Read
$CI add azdo:YOUR_ORG/YOUR_PROJECT        # every repo of the project
$CI sources                               # what is remembered + every registered repo
$CI sync                                  # freshness loop: cron/systemd timer every 15 min
```

`indexio sync` is incremental end to end: clones are ff-pulled, new repos discovered, git deltas
re-indexed by commit, plain folders by content hash, unchanged chunks are CAS hits (0
recomputed embeddings). Per-repo failures (e.g. a repo with no commits) are listed as
`FAILED` and never stop the run. `indexio org-sync` still exists as a legacy alias.

Watch the reported `cas_hits` ratio on the second full run — on any real org with
forks/vendoring you should see the dedup effect immediately (our bench: 100% CAS hits
indexing a fork; 76% of chunk embeddings deduped across 4 repos).

## 4. Semantic plane: standalone first, neural optional

**Default (no setup): Random Indexing.** `indexio embed --all` builds a corpus-native semantic
model in-process on CPU (`<data-dir>/sem/rindex-v1.rimodel`) — single streaming pass,
incremental per chunk thereafter, no GPU/sidecar/downloads. It learns your org's own
vocabulary. Evaluate it on your corpus with the bench harness:

```bash
python3 bench/eval_modes.py --ci target/release/indexio --data-dir $INDEXIO_DATA_DIR \
  --answers bench/known_answers20.json --modes lexical,semantic,hybrid --limit 10
# our ripgrep+serde reference numbers (P5): lexical MRR 0.050 / semantic 0.293 /
# hybrid MRR 0.397, R@5 60% (20 queries: 16 hard + 4 medium)
```

Embedder selection order: `INDEXIO_EMBED_BASE` set → HTTP endpoint · `--embedder hash` → legacy
hash (A/B) · default → Random Indexing (`rindex-v2`: 2048 dims, direction permutations,
hub cut, header×2.5, thesaurus query expansion). Model namespaces (`rindex-v2`, `hash-v1`,
`http:*`) keep separate `vec/` + `embcas/` + `bm25/` stores and coexist. Weekly cron:
`indexio embed --all --rebuild-model` to purge deleted-doc drift from the RI model.

Hybrid fusion is 3-leg RRF (file-lexical + chunk-BM25F + semantic); `--fusion combmnz`
exists for experiments (measured worse: 0.34 vs 0.397 MRR). Do NOT use `--rerank` without
a real `INDEXIO_RERANK_BASE` endpoint — the offline overlap reranker measurably hurts
(0.195 MRR vs 0.397).

**Optional upgrade: neural endpoint.** The engine never runs neural inference in-process —
it calls OpenAI-compatible endpoints. One GPU box (or a CPU box with the 0.6B variants)
serves the whole fleet.

### Embedding endpoint (pick one)

```bash
# vLLM (OpenAI-compatible /v1/embeddings)
pip install vllm
vllm serve Qwen/Qwen3-Embedding-8B --port 8080        # or Qwen3-Embedding-0.6B on CPU-ish setups

# or HuggingFace TEI (docker)
docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id Qwen/Qwen3-Embedding-8B
```

Then:

```bash
export INDEXIO_EMBED_BASE=http://localhost:8080     # base URL; /v1/embeddings is appended
export INDEXIO_EMBED_MODEL=Qwen/Qwen3-Embedding-8B
export INDEXIO_EMBED_DIM=1024                       # 1024 for 8B, 1024 for 0.6B (check config)
# export INDEXIO_EMBED_KEY=...                      # only if your gateway requires it
$CI embed --all                                # builds vec/<model-id>.civec + .cihnsw at ≥50k chunks
$CI search "refresh expired oauth tokens" --mode hybrid --limit 10
```

Model namespaces are keyed by `model-id`, so the old `hash-v1` index and the new
`http:Qwen...` index coexist; re-embedding only computes chunks missing from that model's
CAS namespace. Switching models never touches the lexical index.

### Reranker endpoint (optional but recommended)

```bash
docker run -p 8081:80 ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id Qwen/Qwen3-Reranker-8B            # exposes POST /rerank

export INDEXIO_RERANK_BASE=http://localhost:8081
export INDEXIO_RERANK_MODEL=Qwen/Qwen3-Reranker-8B
$CI search "how is backpressure handled" --mode hybrid --rerank --limit 10
```

### Quality A/B you should run once

Index 5–10 repos you know well. Take 10 questions where you know the answer file.
Compare `lexical` vs `semantic` vs `hybrid` vs `hybrid --rerank` (with real endpoints)
on where the answer file ranks. Expect: lexical wins on exact identifiers, hybrid+rerank
wins on concept questions — that combination is the production default to put in your
MCP client instructions.

## 5. Claude Code / agent integration (recap)

```bash
claude mcp add indexio -- /path/to/indexio mcp --data-dir /var/lib/ciindex
# or .mcp.json: {"mcpServers":{"indexio":{"command":"/path/to/indexio","args":["mcp","--data-dir","/var/lib/ciindex"]}}}
```

```bash
indexio setup claude --claude-md ~/.claude/CLAUDE.md   # registers the server + appends the guidance block
```

12 tools: `code_search` (with `mode` + `rerank`), `semantic_search`, `code_grep`, `list_files`,
`refresh_index`,
`find_symbol`, `who_calls`, `index_stats`, and the SPEC-P6 set: `impact_of_symbol`,
`impact_of_diff` (pass a patch, or omit `diff` to analyse the registered repo's
working tree), `file_outline`, `read_span`. Tool descriptions already coach the agent on
when to use lexical vs semantic vs hybrid and to prefer outline+span over reading files.

## 6. Performance knobs & operational notes

- **Chunk size**: `indexio embed --max-chars N` (default 1200 source chars). Smaller = more
  precise retrieval, more vectors; bigger = more context per hit.
- **RI knobs** (`indexio-embed/src/rindex.rs` consts): DIM=2048, LABEL_NNZ=8, window ±3,
  header weight 2.5, hub cut df>40%, expansion top-3 @ 0.35. Each was tuned against
  `bench/eval_modes.py` — re-run it after any change.
- **BM25F knobs** (`indexio-embed/src/bm25.rs`): k1=1.0, b=0.35, header weight 3.0, hub cut
  df>40% of rows.
- **HNSW**: opt-in above `INDEXIO_HNSW_THRESHOLD` rows (default 1,000,000). The flat
  binary-code prescan answers a 184k-row index in tens of ms, while the single-threaded
  graph build costs ~5 min at that size; measured HNSW recall@10 = 0.955 at ef=100,
  0.995 at ef=200 if you do enable it. Delete `vec/*.cihnsw` to force a graph rebuild;
  it's a pure sidecar of `.civec`.
- **Embed throughput**: `RUST_LOG=indexio_embed=info indexio embed --all` prints per-repo phase
  timings (chunk / CAS lookup / observe / embed+put) and the one-off index build time.
  Measured on a 32-core laptop: ~0.15 ms per chunk end-to-end for the Random Indexing
  model; the whole 200k-chunk GitHub folder embeds cold in well under a minute.
- **Compaction**: `indexio compact` merges shards (LSM-style). Run nightly if you re-index
  often; query fan-out cost grows with shard count.
- **Tombstones**: deletions are lazy (roaring bitmap). Compaction reclaims space.
- **Data dir portability**: the whole `<data-dir>` is self-contained — rsync it to any
  machine with the same binary and serve immediately. Great for CI runners: build the
  index centrally, distribute read-only.
- **Memory**: shards and the `.civec` vector index are mmap'd (page cache, not heap);
  a query touches the binary codes (256 B per row at dim=2048) plus ~256 rescored
  vectors. The Random Indexing model file (`sem/*.rimodel`, up to ~250 MB at the 250k
  vocabulary cap) IS loaded into RAM by every process that embeds queries — one-off for
  `indexio serve` / `indexio mcp`, per invocation for the CLI.

## 7. Where to continue (suggested roadmap, cheapest first)

Done in P6 (`SPEC-P6.md`): impact analysis (`indexio impact`, `impact_of_symbol`,
`impact_of_diff`), file outline + span, compact MCP payloads, and a planner fix
(regex alternation prefixes were intersected instead of unioned — `/(a|b)x/` lost recall).

1. **Scope-aware impact edges** — call postings only store the callee's last identifier
   segment. Storing the receiver/qualifier (`Foo::new` vs `Bar::new`) needs a CALL payload
   extension (format version bump) but would remove most hub noise from `indexio impact`.
2. **Import edges at index time** — `who_imports` is a query-time regex; persisting
   per-file import lists (new shard section) would make `impact --file` exact and enable
   a module dependency graph.
3. **Watch daemon** — `indexio watch` wrapping org-sync/reindex/embed on inotify or a poll
   loop; today it's cron. Entry point: `indexio-ingest::reindex_repo` + `indexio-embed::pipeline`.
4. **More languages for symbols/chunks** — add a grammar crate + a defs query in
   `indexio-symbols/src/lib.rs` (`LangSpec` table); chunking, outline and impact inherit.
5. **Per-repo ACLs for MCP** — mirror `serve.rs`'s `repo_allowed` into `mcp.rs` if you
   expose MCP over a network transport.
6. **Query-time repo/path boosting config** — ranking constants live at the top of
   `indexio-query/src/lib.rs`; they're one-line experiments.
7. **SCIP ingestion (big, optional)** — the report's cost ladder says only do this if
   cross-repo *precise* references become a hard requirement.
8. **Web UI** — the HTTP API is already JSON-shaped for it.

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `indexio search --mode semantic` returns nothing | No vec index yet → run `indexio embed --all` (degrades silently by design) |
| `HttpEmbedder` errors at startup | `INDEXIO_EMBED_BASE`/`INDEXIO_EMBED_MODEL` unset or endpoint down; `curl $INDEXIO_EMBED_BASE/v1/embeddings` |
| Embedding feels slow | Run with `RUST_LOG=indexio_embed=info` to see per-phase timings. Second run should be ~100% `cas_hits`. If `index build` dominates, a `.cihnsw` is being built: raise `INDEXIO_HNSW_THRESHOLD` |
| `--rerank` rejected | Only valid with `--mode hybrid` (by design) |
| 403 on `/admin/*` | Your ACL token lacks `allow: ["*"]` |
| Index bigger than expected | Run `indexio compact`; check `indexio cas-stats` for vendored duplication (it's deduped, that's the point) |
| Weird query errors | Query grammar is in `SPEC.md` §query-language: literals, `"phrases"`, `/regex/`, `repo:/lang:/path:/case:` filters |
| `indexio add azdo:…` fails with 401/203 | Set `AZDO_TOKEN` to a PAT with *Code: Read* for that org; SSO-backed orgs need the PAT created under the same identity |
| `indexio add github:org` lists 0 private repos | `GITHUB_TOKEN` needs `repo` scope (classic) or *Contents: Read* on the repos (fine-grained) |
| Two repos with the same folder name | The second is registered as `<parent>-<name>` (see `indexio sources`) |

## 9. Verifying integrity after the transfer

```bash
cd indexio
git log --oneline | head -30        # 30+ commits, latest = docs commit on main
git status                          # clean tree
cargo test --workspace 2>&1 | tail -3
```

Test suite anatomy (what's covered, all offline): posting codec round-trips, gram
extraction consistency, shard write/read/merge/tombstones, 6-language symbol extraction,
chunking edge cases, CAS + git-delta ingest (incl. fork dedup), vector index
(BQ-prescan-vs-brute-force equivalence, HNSW recall≥0.95, persistence), embedder/reranker
HTTP clients against mock servers, RRF fusion ordering, org-sync against a mock GitHub
API, CLI e2e (index→embed→search), HTTP auth/ACL, MCP protocol flows.
