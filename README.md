# indexio

indexio is a code search engine for AI coding agents. It indexes all the repositories of an
organization one time. It answers a query in milliseconds. It gives the index to agents as
a tool through the Model Context Protocol (MCP) and through an HTTP API.

indexio is one binary, written in Rust. It does not need a database or an external
service. The index is a set of files in one data directory. You can copy that directory to
another machine.

## Install

Pick one. Each one gives you the `indexio` command.

| Platform | Command |
|---|---|
| Linux, macOS | `curl -fsSL https://raw.githubusercontent.com/IOServicesLabs/indexio/main/install.sh \| sh` |
| Windows (PowerShell) | `irm https://raw.githubusercontent.com/IOServicesLabs/indexio/main/install.ps1 \| iex` |
| npm (any platform) | `npm install -g indexio` |
| Docker | `docker pull ghcr.io/ioserviceslabs/indexio` |
| From source (Rust 1.85+) | `cargo install --git https://github.com/IOServicesLabs/indexio indexio` |

The scripts and the npm package download the binary of the latest
[release](https://github.com/IOServicesLabs/indexio/releases) for your OS and CPU and
check its SHA-256. `git` must be on the PATH. No service, no database and no model download
is necessary.

## Quick start

Three commands. The third one is only for Claude Code.

```bash
indexio add ~/code            # 1. index every git repository under a folder
indexio search parse_config   # 2. search it (add --mode hybrid for a question in words)
indexio setup claude          # 3. register the MCP server with Claude Code
```

Then restart Claude Code. Its sessions now have the indexio tools. Two optional commands
make the savings larger:

```bash
indexio hook install          # reads and greps of indexed files go through the index
indexio sync                  # run from cron or a timer to keep the index current
```

`indexio stats` shows what is indexed. `indexio usage` shows the tokens the index served
against what the built-in tools would have returned.

All data goes into one directory: `--data-dir`, `$INDEXIO_DATA_DIR`, or `~/.indexio`.

## Contents

1. [What indexio does](#what-indexio-does)
2. [Add sources](#add-sources)
3. [Keep the index current](#keep-the-index-current)
4. [Embeddings](#embeddings)
5. [Secure the HTTP API](#secure-the-http-api)
6. [Use indexio with Claude Code](#use-indexio-with-claude-code)
7. [Token savings, measured](#token-savings-measured)
8. [Impact analysis](#impact-analysis)
9. [Use indexio from other tools](#use-indexio-from-other-tools)
10. [Command reference](#command-reference)
11. [Environment variables](#environment-variables)
12. [Deploy with Docker](#deploy-with-docker)
13. [Architecture](#architecture)
14. [Measured performance](#measured-performance)
15. [Limits](#limits)
16. [Development and releases](#development-and-releases)

## What indexio does

indexio answers three kinds of question about the code of an organization.

| Kind | Example | Method |
|---|---|---|
| Lexical | Where is `WalkState::new` defined? Which Rust files match `/fn parse_*/`? | Trigram index, regex, exact identifiers |
| Semantic | How do we buffer printer output? Where do we refresh the auth token? | Vector search over syntax-aware code chunks |
| Hybrid | Any question | Lexical, BM25 and semantic results, fused by Reciprocal Rank Fusion (RRF) |

indexio also knows the definitions and the call graph of the code. It can tell you where a
symbol is defined, who calls it, and what breaks if you change it.

## Add sources

Each `add` indexes the source immediately and builds the semantic plane.

```bash
indexio add ~/code                             # every git repository under the folder, at any depth
indexio add /srv/checkouts/legacy-monolith     # a folder without git: indexed as-is
indexio add github:my-org                      # a GitHub organization or user
indexio add github:my-org/payments             # one GitHub repository
indexio add azdo:my-org/Platform               # an Azure DevOps project
indexio add https://gitlab.example.com/g/r.git # any git host
```

Search:

```bash
indexio search parse_config
indexio search "how is backpressure applied" --mode hybrid
```

Serve the index to agents:

```bash
indexio mcp                           # MCP over stdio
indexio serve                         # HTTP API on http://127.0.0.1:7717
```

indexio remembers each source in `<data-dir>/sources.json`. `indexio sources` lists the
sources and the registered repositories. `indexio remove <source>` forgets a source.

Useful flags for `add`: `--limit 20` (trial run on a large organization), `--include-forks`,
`--include-archived`, `--full-clone`, `--dest DIR` (where remote clones go), `--no-embed`.

**Credentials.** indexio reads credentials from the environment only: `GITHUB_TOKEN` (or
`GH_TOKEN`) and `AZDO_TOKEN` (or `AZURE_DEVOPS_EXT_PAT`, `SYSTEM_ACCESSTOKEN`). It sends
them to the listing APIs and to `git` as an authorization header. It never puts them in a
clone URL, in a log, or in `.git/config`. Public repositories do not need a token.

## Keep the index current

Run `indexio sync` from cron or a timer, for example every 15 minutes. Each run is
incremental:

- New repositories are cloned.
- Changed git repositories are delta-indexed by commit.
- Changed plain folders are delta-indexed by content hash.
- New and changed chunks are embedded. All other chunks are cache hits.

A folder tree of 156 repositories and 17,530 files syncs as a no-op in 11 s.

A running `indexio mcp` server does more on its own:

- It watches the repository the session works in. It re-indexes the working tree before
  each call when files changed. Uncommitted and untracked edits are searchable at once;
  their embeddings follow on a background thread within a few hundred milliseconds.
- Every 10 minutes it probes the other repositories for a moved HEAD and re-indexes the
  ones that moved.
- A session that starts before its folder is registered adopts the repository within one
  minute of `indexio add`. No restart is necessary.

## Embeddings

### The built-in model

The default embedder is Random Indexing (`rindex`). It is a semantic model that indexio
builds in-process from your own code, on the CPU, in one streaming pass. It needs no
training infrastructure, no download and no GPU. It learns the vocabulary of your
organization: internal service names, code names and APIs that no pretrained model knows.

Each new chunk is an incremental update, so `indexio sync` stays cheap.

Measured on 20 hard concept queries (`bench/`): hybrid recall@5 is 60 % and MRR is 0.397.
Lexical search alone gives 5 % and 0.050 on the same queries.

### An optional neural endpoint

For maximum retrieval quality, point indexio at any OpenAI-compatible embeddings endpoint,
for example Qwen3-Embedding on vLLM or TEI. The two model namespaces coexist.

```bash
export INDEXIO_EMBED_BASE=http://embed.internal.example:8080
export INDEXIO_EMBED_MODEL=Qwen/Qwen3-Embedding-8B
export INDEXIO_EMBED_DIM=1024
export INDEXIO_EMBED_KEY=...                                    # only if the gateway needs it
indexio embed                                                   # builds vec/<model-id>.*.civec
```

Embeddings are content-addressed by `(chunk hash, model id)`. Two repositories that share a
file share its embedding. Duplicate chunks across an organization cost nothing extra.

### An optional reranker

Add a reranker endpoint, for example Qwen3-Reranker on TEI or vLLM. indexio uses it when a
query asks for it: `indexio search --mode hybrid --rerank`, `rerank=1` over HTTP, or
`rerank: true` over MCP. Without an endpoint, `--rerank` has no effect.

```bash
export INDEXIO_RERANK_BASE=http://rerank.internal.example:8081
export INDEXIO_RERANK_MODEL=Qwen/Qwen3-Reranker-8B
export INDEXIO_RERANK_KEY=...
```

## Secure the HTTP API

`indexio serve` binds to the loopback address by default.

```bash
indexio serve --auth-token $(openssl rand -hex 32)               # one bearer token
```

For per-token repository access, use an ACL file:

```bash
cat > acl.json <<'EOF'
{"tokens":{"indexio-reader":{"allow":["team-*","core"]},"indexio-admin":{"allow":["*"]}}}
EOF
indexio serve --acl-file acl.json
```

With an ACL file:

- An unknown token gets HTTP 401.
- Hits outside the allowed patterns are removed from `/search`, `/symbol` and `/calls`.
- Admin routes need a token with `allow: ["*"]`. Other tokens get HTTP 403.
- `GET /health` is always open.

**Caution.** A bind address other than loopback (`--bind 0.0.0.0`) is refused unless you
also set `--auth-token`, `$INDEXIO_AUTH_TOKEN` or `--acl-file`. The MCP server over stdio
runs as the local user and has no authentication by design. Put a network-facing MCP
server behind your own gateway.

## Use indexio with Claude Code

indexio speaks MCP over stdio. This is the native tool interface of Claude Code, of the
Claude Agent SDK and of most coding agents.

### Register the server

Do one of these:

- Run `indexio setup claude`. It runs `claude mcp add` for this binary and data directory,
  and it prints the guidance block for your `CLAUDE.md`. Add `--claude-md ~/.claude/CLAUDE.md`
  to append the block.
- Run `claude mcp add indexio -- /usr/local/bin/indexio mcp --data-dir /var/lib/ciindex`.
- Commit a `.mcp.json` to the project:

  ```json
  {
    "mcpServers": {
      "indexio": {
        "command": "/usr/local/bin/indexio",
        "args": ["mcp", "--data-dir", "/var/lib/ciindex"]
      }
    }
  }
  ```

### Install the hooks

```bash
indexio hook install
```

This writes three Claude Code hooks into `~/.claude/settings.json`:

| Hook | Effect |
|---|---|
| PreToolUse, Bash | A shell read or search of an indexed file (`cat`, `sed -n`, `head`, `grep`, `rg`, `find`, a `python -c` or `node -e` one-liner that only opens the file) is refused. The refusal names the indexio call that gives the same result. A command that also does other work, for example a script or a build, runs as typed. |
| PreToolUse, Read | A whole-file Read of an indexed file is refused. The refusal names `file_outline` and `read_span`. |
| PreCompact | Session transcripts are imported for the `recall` tool. |

The Bash hook also routes scripts and builds (`python`, `node`, `npm`, `pytest`, `go`,
`make` …) through `indexio run`, which keeps a long output out of the context: the first
lines, the error lines and the last lines are returned with a `runs:<repo>/<log>` pointer,
and the full output stays searchable for 30 days.

To run a refused command as typed, add `# raw` to it and send it again, or set
`INDEXIO_RAW=1`.

### The tools

| Tool | Use it for |
|---|---|
| `code_search` | Literal or regex search (`mode: "lexical"`), concept search (`"semantic"`), or both (`"hybrid"`). A lexical query that looks like a grep pattern and matches nothing runs as a regex. A lexical query of several words that matches nothing runs as hybrid. |
| `code_grep` | `grep -n` across all repositories: every matching line per file, ranked. `repo` and `path` arguments, or trailing `repo:`, `path:` and `lang:` filters. An unbalanced parenthesis is read as a literal. |
| `find_symbol` | Where a function, struct, class, trait or enum is defined. |
| `who_calls` | Every recorded caller of a symbol. |
| `semantic_search` | Vector search only. |
| `list_files` | A glob or substring over indexed paths, folded by directory. No pattern lists every file. |
| `file_outline` | Every definition of a file with its start and end lines. A few hundred tokens instead of the file. |
| `read_span` | An exact line range of an indexed file. Without an end line, the whole definition at the start line. |
| `impact_of_symbol` | The transitive callers of a symbol across all repositories. |
| `impact_of_diff` | The callers and importers touched by a patch or by the uncommitted working tree. |
| `refresh_index` | A delta re-index, embed and reload from inside the session. |
| `recall` | Search of earlier sessions and of stored command output: requests, answers, tool calls, results, build and test logs. A lone hit comes with its exchange. |
| `index_stats` | What is indexed, how current it is, and the last week of usage against the built-in tools. |

The server tells Claude which built-in tool each indexio tool replaces, and which
repositories are indexed. Claude then routes an identifier to lexical search and a
question to hybrid search on its own. It reads a file by outline and span instead of whole.

Results are dense plain text: grouped `repo:path` and `line: snippet` rows, folded file
lists, numbered spans. No scores, ranks or JSON punctuation. This costs 54 % fewer tokens
than compact JSON for the same hits. Set `INDEXIO_MCP_FORMAT=json` for the JSON form.

## Token savings, measured

### Against the built-in tools, on the same tasks

`bench/ab_readme.py` runs the tasks a coding agent does through the built-in tools of
Claude Code and through indexio, on the same repositories. `Grep` prints every match as
`path:line:text`. `Read` prints the whole file, 2,000 lines at most. `Glob` prints one path
per line. Tokens are counted with tiktoken. Three real repositories, 44 in the index, 10
symbols and 3 file types per repository, questions written by a model.

| Task | Built-in | indexio | Change |
|---|---|---|---|
| Find where a symbol is defined (30) | 814 | 1,421 | Worse. `find_symbol` lists every repository. A targeted grep finds one. |
| List the call sites of a symbol (30) | 2,018 | 1,409 | −30 % |
| Read one function, file known, lines unknown (30) | 419,380 | 41,612 | −90 % |
| List the files of one type (9) | 12,216 | 3,887 | −68 % |
| Answer a question about the code (11) | 1,144,842 | 1,883 | −99.8 % |
| All tasks | 1,579,270 | 50,212 | −97 % |

Where the built-in tool is cheaper, the difference is a few hundred tokens. Where it is
expensive, it is expensive by orders of magnitude, because `Read` returns files and `Grep`
returns every match of every word.

### On real traffic

The server logs every call with the bytes it returned. Where the built-in tool has a
mechanical equivalent, the server also logs what that tool would have returned:

| Call | Built-in equivalent |
|---|---|
| `read_span` | `Read` of the whole file, charged once per file version: again after the file changes |
| `find_symbol`, `who_calls`, `code_grep`, lexical `code_search` | `Grep` in the session's repository, first 250 rows |
| `list_files` | `Glob` as absolute paths |
| `file_outline` | None. Counted as cost. |
| hybrid search, `recall`, impact tools | None. Not compared. |

`index_stats` ends with the last seven days across all sessions. `indexio usage` prints
the same per tool and per repository. One day of real coding sessions on the author's
machine: 423 compared calls, 220k tokens served where the built-in tools would have
returned 596k, a reduction of 63 %. `read_span` alone: −71 %.

## Impact analysis

```bash
indexio impact --symbol RateLimiter::acquire --depth 2     # transitive callers, all repositories
indexio impact --diff my-service                           # uncommitted working-tree changes
indexio impact --diff-file pr.patch --repo my-service      # a patch
indexio impact --file my-service:src/auth/token.rs         # every definition in a file, plus importers
indexio outline my-service:src/auth/token.rs               # definitions with line ranges
indexio span my-service:src/auth/token.rs 120:160          # exact lines
```

The walk is a breadth-first search over the reverse call graph that the index holds. It
costs nothing at index time and answers in milliseconds. A patch is mapped to definitions
with a tree-sitter pass over the changed files only. Changed files are excluded from the
result. Their importers are found with a language-aware import query.

Edges are name-based. `Foo::new` and `Bar::new` share the node `new`. Each site carries its
`caller`, `path` and `symbol`, so an agent can filter. Fan-out caps keep hub names bounded;
`truncated: true` tells you when a cap fired.

The HTTP API has the same surface: `GET /impact/symbol`, `POST /impact/diff`,
`GET /outline`, `GET /span`. All of them apply the ACL.

## Use indexio from other tools

Any agent framework can use the HTTP API.

```bash
curl 'http://127.0.0.1:7717/search?q=auth%20token%20refresh&mode=hybrid&limit=10'
curl 'http://127.0.0.1:7717/symbol/RateLimiter'
curl 'http://127.0.0.1:7717/calls/RateLimiter::acquire'
```

A response is a JSON list of hits: repository, path, line, column, snippet, score.

## Command reference

Each command accepts `--data-dir DIR`. The default is `$INDEXIO_DATA_DIR`, then `~/.indexio`.

### Index code

```bash
indexio add ~/code                          # every git repository under a folder
indexio add ~/notes                         # a folder without git: indexed as-is
indexio add github:my-org                   # a GitHub organization
indexio add github:my-org/payments          # one repository
indexio add azdo:my-org/Platform            # an Azure DevOps project
indexio add https://gitlab.example.com/g/r.git
indexio add github:my-org --limit 20 --no-embed     # trial run: 20 repositories, lexical only
indexio add github:my-org --dest /srv/clones        # where remote clones go
indexio index ~/code/one-repo --name one            # one repository, no source bookkeeping
indexio sources                             # sources and registered repositories
indexio remove github:my-org                # forget a source
```

### Keep it current

```bash
indexio sync                                # pull, discover, delta re-index, embed. Run from cron.
indexio sync --no-embed                     # lexical plane only
indexio reindex --repo payments             # one repository, from HEAD
indexio reindex --repo payments --worktree  # from the working tree, uncommitted edits included
indexio reindex --all
```

### Search

```bash
indexio search parse_config                                  # exact identifier
indexio search 'parse_config repo:payments lang:rust'        # filters: repo: lang: path: case:
indexio search '"token refresh"'                             # phrase
indexio search '/fn (parse|load)_config\(/'                  # regex
indexio search 'how is backpressure applied' --mode hybrid   # question
indexio search 'retry with exponential backoff' --mode semantic --limit 10
indexio search 'rate limiter' --mode hybrid --fusion combmnz # alternative fusion
indexio search 'rate limiter' --mode hybrid --rerank         # needs INDEXIO_RERANK_BASE
indexio search parse_config --json
```

### Symbols and blast radius

```bash
indexio symbol parse_config                 # definitions
indexio calls parse_config                  # call sites
indexio impact --symbol parse_config        # transitive callers and importers, 2 hops
indexio impact --symbol parse_config --depth 3 --max-sites 1000
indexio impact --diff payments              # the uncommitted working tree
indexio impact --diff payments --base origin/main
git diff main | indexio impact --diff-file - --repo payments
indexio impact --file payments:src/config.rs
indexio impact --symbol parse_config --json
```

### Read from the index

```bash
indexio outline payments:src/config.rs      # definitions with start and end lines
indexio span payments:src/config.rs 120     # the definition at line 120
indexio span payments:src/config.rs 120:160 # a range, 400 lines at most
```

### Embeddings

```bash
indexio embed                               # embed every repository, incrementally
indexio embed --repo payments
indexio embed --all --rebuild-model         # retrain the built-in model from scratch
indexio embed --max-chars 800               # smaller chunks
indexio embcas-stats                        # embedding cache statistics
```

### Serve

```bash
indexio mcp                                 # MCP over stdio
indexio mcp --repo payments                 # pin the session's repository
indexio serve                               # HTTP on 127.0.0.1:7717
indexio serve --port 8080 --auth-token $(openssl rand -hex 32)
indexio serve --bind 0.0.0.0 --auth-token "$TOKEN"          # beyond localhost: a token is mandatory
indexio serve --acl-file acl.json
```

### Agent setup

```bash
indexio setup claude
indexio setup claude --claude-md ~/.claude/CLAUDE.md
indexio hook install
indexio hook install --print-only
indexio sessions                            # import Claude Code transcripts for recall
indexio sessions --project C--Users-me-code-payments
```

### Run commands without filling the context

```bash
indexio run -- python scripts/probe.py --fast    # short output printed as is; long output stored + digested
indexio run -- npm test                          # the Bash hook does this rewrite for you
indexio run --cwd ~/code/payments -- pytest -q
```

A stored log is `runs:<repo>/<stamp>-<command>.log`: `read_span` it for the lines the
digest left out, `recall` finds it later. Logs roll off after `INDEXIO_RETAIN_DAYS`.

### Measure and maintain

```bash
indexio stats                               # repositories, files, shards, index size
indexio usage                               # last 7 days of MCP calls, against the built-in tools
indexio usage --days 1 --json
indexio compact                             # merge shards
indexio compact --vectors                   # also merge vector segments
indexio cas-stats
```

## Environment variables

| Variable | Effect |
|---|---|
| `INDEXIO_DATA_DIR` | Data directory. Default `~/.indexio`. |
| `INDEXIO_REPO` | The repository an MCP session works in. Default: the one that contains the working directory. |
| `INDEXIO_BIND` | Bind address of `serve`. Not loopback needs a token or an ACL. |
| `INDEXIO_AUTH_TOKEN` | Bearer token of `serve`. |
| `INDEXIO_EMBED_BASE`, `INDEXIO_EMBED_MODEL`, `INDEXIO_EMBED_DIM`, `INDEXIO_EMBED_KEY` | An OpenAI-compatible embeddings endpoint. |
| `INDEXIO_RERANK_BASE`, `INDEXIO_RERANK_MODEL`, `INDEXIO_RERANK_KEY` | A reranker endpoint. |
| `GITHUB_TOKEN` or `GH_TOKEN`, `AZDO_TOKEN` | Credentials for remote sources. Never written to disk. |
| `INDEXIO_MCP_FORMAT` | `text` (default) or `json` tool results. |
| `INDEXIO_FUSE_WEIGHTS` | Hybrid leg weights `lex,bm25,sem`. Default `0.5,1.0,0.3`. |
| `INDEXIO_EXPAND` | Enable thesaurus query expansion. Off by default; it measured worse. |
| `INDEXIO_RETAIN_DAYS` | Days that run logs and rendered session transcripts are kept. Default 30, 0 keeps everything. |
| `INDEXIO_NO_USAGE` | Do not write the usage log. For benchmarks. |
| `INDEXIO_RAW` | Run a hooked Bash command as typed. |
| `INDEXIO_HOOK_DEBUG` | The Bash hook explains its decision on stderr. |

## Deploy with Docker

The release workflow publishes `ghcr.io/ioserviceslabs/indexio` for `linux/amd64` and
`linux/arm64`. The image runs as an unprivileged user. The data directory is `/data`.
`git` and CA certificates are included.

Index a folder on this machine and search it, with nothing installed but Docker:

```bash
docker run --rm -v indexio-data:/data -v ~/code:/repos:ro ghcr.io/ioserviceslabs/indexio add /repos
docker run --rm -v indexio-data:/data ghcr.io/ioserviceslabs/indexio search parse_config
docker run --rm -v indexio-data:/data ghcr.io/ioserviceslabs/indexio stats
```

Serve the HTTP API:

```bash
docker run -d --name indexio -p 127.0.0.1:7717:7717 -v indexio-data:/data   -e INDEXIO_AUTH_TOKEN="$(openssl rand -hex 32)" ghcr.io/ioserviceslabs/indexio
curl -H "Authorization: Bearer $TOKEN" 'http://127.0.0.1:7717/search?q=parse_config'
```

Give the container to an agent as an MCP server (stdio through `docker run -i`):

```bash
claude mcp add indexio -- docker run -i --rm -v indexio-data:/data -v ~/code:/repos:ro ghcr.io/ioserviceslabs/indexio mcp
```

The container sees your code under `/repos`, not under its host path. The hooks and the
working-tree refresh of the session's own repository need the host binary. Use the
container for a shared index; use the host binary next to the agent.

`docker-compose.yml` runs two services on one volume:

- `indexio`: the HTTP API, published on localhost only.
- `sync`: `indexio sync` every 15 minutes. `GITHUB_TOKEN` and `AZDO_TOKEN` pass through.

1. Put the folders to index under `./repos`.
2. Start the services: `INDEXIO_AUTH_TOKEN=$(openssl rand -hex 32) docker compose up -d`
3. Index the folders one time: `docker compose exec sync indexio add /repos`

The container binds `0.0.0.0`. The binary refuses to start without `INDEXIO_AUTH_TOKEN` or
an ACL file. `docker build -t indexio .` builds the same image from source.

## Architecture

```
            ┌──────────────────────────── data dir ────────────────────────────┐
 git repos  │  shards/*.cidx   immutable mmap'd index shards (single file)      │
    │       │  cas/            global content-addressed store (blake3 keyed)    │
    ▼       │  embcas/         embedding CAS: (chunk-hash, model-id) → vector   │
 indexio-ingest  │  vec/*.civec     int8 vector segments + binary prescan            │
  git delta │  repos/*.json    per-repo indexing state (last commit)            │
    │       └──────────────────────────────────────────────────────────────────┘
    ▼
 Lexical plane                             Semantic plane
 ─────────────────────                     ──────────────────────────
 trigram inverted index                    tree-sitter chunking
 FST term dictionary, LEB128 postings      pluggable embedder (built-in model
 symbols + call edges (tree-sitter,          or any OpenAI-compatible endpoint)
 6 languages, AST discarded at once)       embedding CAS (org-wide dedup)
 query planner: required literals,         mmap'd binary prescan + int8 rescore
  galloping intersection, verify, BM25
            └─── indexio-query: RRF fusion → optional reranker ───┘
                                   │
                    CLI · HTTP (axum) · MCP (stdio JSON-RPC)
```

Five design decisions:

1. **A trigram index, not a resident AST.** indexio parses the AST, takes symbols, call
   edges and chunk boundaries from it, and discards it. A resident AST costs 20 to 50 times
   the text size. Trigram shards cost 1 to 3 times the corpus and answer in milliseconds.
2. **A global content-addressed store.** Extracted artifacts and embeddings are keyed by
   content hash across the organization. A fork, a vendored copy or a duplicated file is
   indexed and embedded one time.
3. **Immutable mmap'd shards with tombstones.** An index build writes new shards. A
   deletion is a bitmap tombstone. `indexio compact` merges shards in the background.
   Readers never block writers.
4. **Git-native delta indexing.** A re-index walks the diff of the HEAD tree. Unchanged
   blobs are not read, parsed or embedded again. A one-file commit re-indexes in less than
   one second.
5. **An independent semantic plane.** The vector index is derived from the store and the
   shards. You can change the embedding model or rebuild the plane without touching the
   lexical index.

Data layout:

```
<data-dir>/
  shards/*.cidx          immutable index shards (mmap'd)
  cas/v2/<hh>/<rest>     content-addressed extracted artifacts (zstd)
  embcas/<model>/        content-addressed embeddings, one pack per model
  vec/<model>.*.civec    int8 vector segments with a binary prescan
  bm25/*.cibm25          chunk-level BM25 sidecar (mmap'd)
  sem/*.rimodel          the built-in semantic model (mmap'd)
  repos/<name>.json      per-repository state
  sessions/              rendered session transcripts for recall (30-day window)
  runs/<repo>/*.log      command output kept by `indexio run` (30-day window)
  usage/<day>.jsonl      the MCP usage log
```

## Measured performance

Storage and start-up, measured on the author's machine on a 44-repository, 10,000-file
data directory (`docs/SPEC-P10.md`):

| Metric | Before | After |
|---|---|---|
| Data directory for 143 MB of source | 5.3 GB | 1.0 GB |
| Server start: time, bytes read, RSS | 188 ms, 302 MB, 417 MB | 10 ms, 0 MB, 74 MB |
| Hybrid query, warm process | 130 to 190 ms | 7 to 14 ms |
| Regex grep, warm process | 424 ms | 28 ms |
| `find_symbol`, `who_calls` | 12 ms | 0.5 ms |
| Working-tree re-index after an edit | not available | about 25 ms |

The same lookups through the shell and through the index (`bench/ab_speed.py`, this
repository, a 1,300-file TypeScript application gives the same picture). Wall-clock,
medians of 10, the index side includes the MCP transport:

| Lookup | Shell | indexio |
|---|---|---|
| Identifier grep (`rg -n`) | 32 ms | 0.6 ms |
| Regex grep (`rg -n`) | 34 ms | 2.5 ms |
| `grep -rn` over an application | 26 to 860 ms | 0.7 ms |
| Read 120 lines (`sed -n`) | 23 ms | 0.3 ms |
| `find -name` | 25 to 410 ms | 0.9 ms |
| Definition lookup | 33 ms | 0.1 ms |

Each Bash tool call also pays about 15 ms for the shell and about 30 ms for the two
PreToolUse hooks before its command runs. An MCP call pays neither. On real traffic
(5,355 calls in 7 days) the median is 0 ms for `read_span`, 12 ms for `code_search` and
29 ms for `code_grep`; the 90th percentile stays under 130 ms.

Duplication, measured on a sandbox with 2 cores and 4 GB of RAM:

| Scenario | Result |
|---|---|
| Index `serde`, then an identical fork | Fork: 100 % cache hits, 3.0 s instead of 10.0 s |
| Index `ripgrep`, then a fork with one changed file | Fork: 100 % cache hits, 2.5 s instead of 6.6 s |
| Re-embed 4 repositories, 19,977 chunk instances | 4,731 unique embeddings, 76 % deduplicated |

Lexical latency on the same sandbox: rare identifier 21 ms, common term 81 ms, regex
254 ms. Delta re-index of a one-file commit: 792 ms.

Semantic quality on 20 concept queries (`bench/known_answers20.json`, `bench/eval_modes.py`):

| Mode | MRR | recall@5 |
|---|---|---|
| lexical | 0.050 | 5 % |
| semantic | 0.293 | 40 % |
| hybrid | 0.397 | 60 % |

Measured and rejected: CombMNZ fusion scored below RRF. An offline overlap reranker scored
below hybrid alone. Thesaurus query expansion scored below plain semantic search. An IVF
prescan gave no speed-up. These stay off by default.

## Limits

- **No type-resolved cross-references.** Call edges are name-based. Overloaded names such
  as `new`, `get` and `run` produce noise. Filter by `caller` and `path`.
- **Untracked files in `impact --diff`.** `git diff` does not include them. Stage them, or
  pass a patch.
- **The built-in model is not neural.** It is useful and cheap. For neural quality, set
  `INDEXIO_EMBED_BASE`.
- **MCP has no authentication.** It runs as the local user over stdio. Do not expose it to
  a network without a gateway.
- **No web UI and no distributed query.** One binary, one data directory.

## Development and releases

```
cargo test --workspace                  # every crate, 303 tests
cargo clippy --workspace --all-targets
tools/build-release.sh                  # release build; remaps local paths out of the binary
```

Crates: `indexio-types` (shared types, posting codec), `indexio-core` (n-grams, verification,
query literals), `indexio-index` (shards, merge, tombstones), `indexio-symbols` (tree-sitter
symbols, calls and chunks for 6 languages), `indexio-ingest` (git delta, content store,
sources), `indexio-embed` (embedders, embedding store, vector index, pipeline),
`indexio-query` (planner, ranking, fusion, impact), `indexio` (CLI, HTTP, MCP, hooks). The
design notes are in `docs/`; `docs/DEVELOPMENT.md` is the operator manual.

CI (`.github/workflows/ci.yml`) runs on each push and pull request. It checks that the
manifests parse, the scripts compile and the workflows are valid. It is killed after 90
seconds. The build and the test suite run locally; they do not fit that budget.

Releases (`.github/workflows/release.yml`) come from `main`:

1. Set `version` in `Cargo.toml` (`[workspace.package]`).
2. Merge to `main`. In less than 90 seconds the workflow creates the tag `v<version>` and a
   GitHub release with generated notes. A merge without a version change does nothing.
3. The same workflow then builds the binaries for Linux (x86_64, aarch64), macOS (Apple
   silicon, Intel) and Windows (x86_64), attaches each archive with a `.sha256` to the
   release, publishes the container image to `ghcr.io`, and publishes the npm package when
   the `NPM_TOKEN` secret exists. These jobs compile Rust and take some minutes.

`tools/release-assets.sh` builds and attaches one asset from a maintainer machine, for a
platform the workflow does not cover.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
