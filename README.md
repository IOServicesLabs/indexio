# indexio

[![release](https://img.shields.io/github/v/release/IOServicesLabs/indexio?label=release)](https://github.com/IOServicesLabs/indexio/releases)
[![npm](https://img.shields.io/npm/v/indexio?label=npm)](https://www.npmjs.com/package/indexio)
[![PyPI](https://img.shields.io/pypi/v/indexio-cli?label=pypi)](https://pypi.org/project/indexio-cli/)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

indexio is a code search engine for AI coding agents. It indexes all the repositories of an
organization one time. It answers a query in milliseconds. It gives the index to agents as
a tool through the Model Context Protocol (MCP) and through an HTTP API.

indexio is one binary, written in Rust. It does not need a database or an external
service. The index is a set of files in one data directory. You can copy that directory to
another machine.

## Let your agent install it

Paste the block below to Claude Code, Cursor, Codex or any coding agent that can run
shell commands. It installs indexio, indexes your code, registers the MCP server and
verifies the result. Replace `~/code` with the folder that holds your repositories.

````text
Install indexio (https://github.com/IOServicesLabs/indexio) on this machine and connect it
to me. Do these steps in order and stop at the first failure with the exact error.

1. Install the binary.
   - Linux or macOS:  curl -fsSL https://raw.githubusercontent.com/IOServicesLabs/indexio/main/install.sh | sh
   - Windows (PowerShell):  irm https://raw.githubusercontent.com/IOServicesLabs/indexio/main/install.ps1 | iex
   - If neither works, use `pip install indexio-cli`, `npm install -g indexio`, or `cargo install --git https://github.com/IOServicesLabs/indexio indexio`.
   The binary lands in ~/.local/bin (Linux, macOS) or %LOCALAPPDATA%\Programs\indexio (Windows).
   If `indexio --version` is not found afterwards, add that folder to PATH and open a new shell.
   `git` must be on PATH.

2. Index my code:  indexio add ~/code
   This registers every git repository under the folder at any depth and builds the index.
   Nothing is uploaded; the index lives in ~/.indexio.

3. Register the MCP server with the agent you are.
   - Claude Code:  indexio setup claude --claude-md ~/.claude/CLAUDE.md
     then  indexio hook install
     (the hooks route file reads and greps through the index; scripts and builds run as before)
   - Any other MCP client: add this server to its MCP configuration and restart the client:
       command: indexio   args: ["mcp"]
     Use the absolute path of the binary if the client does not search PATH.

4. Verify. All four must succeed:
   - indexio --version                  prints a version
   - indexio stats                      lists the repositories and a file count above zero
   - indexio search "TODO" --limit 3    returns hits from my code
   - indexio usage                      prints a report (it is empty until I use the tools)

5. Tell me what was installed, where the data directory is, which repositories were
   indexed, and that I must restart my agent session so it picks up the new MCP server.
   Do not modify any file inside my repositories.
````

What the agent gets afterwards: `code_search`, `code_grep`, `find_symbol`, `who_calls`,
`file_outline`, `read_span`, `impact_of_symbol`, `impact_of_diff` and `recall` as tools,
answered from the index in milliseconds. The [tools table](#the-tools) describes each one.

## Install

Pick one. Each one gives you the `indexio` command.

| Platform | Command |
|---|---|
| Linux, macOS | `curl -fsSL https://raw.githubusercontent.com/IOServicesLabs/indexio/main/install.sh \| sh` |
| Windows (PowerShell) | `irm https://raw.githubusercontent.com/IOServicesLabs/indexio/main/install.ps1 \| iex` |
| npm (any platform) | `npm install -g indexio` |
| pip (any platform) | `pip install indexio-cli` |
| Docker | `docker pull ghcr.io/ioserviceslabs/indexio` |
| Homebrew, winget | not yet; use a script or a package above |
| From source (Rust 1.85+) | `cargo install --git https://github.com/IOServicesLabs/indexio indexio` |

The scripts and the npm package download the binary of the latest
[release](https://github.com/IOServicesLabs/indexio/releases) for your OS and CPU and
check its SHA-256. The pip wheels carry the binary inside them, one wheel per platform.
The pip package is called `indexio-cli` because `indexio` on PyPI belongs to an unrelated
project; the command it installs is `indexio`. The container image works with no install
at all: see [Deploy with Docker](#deploy-with-docker). `git` must be on the PATH. No service, no database and no model download
is necessary.

## Upgrade

Run the same install command again. The index in `~/.indexio` stays, and you do not
index again.

| Installed with | Upgrade command |
|---|---|
| Script, Linux or macOS | `curl -fsSL https://raw.githubusercontent.com/IOServicesLabs/indexio/main/install.sh \| sh` |
| Script, Windows | `irm https://raw.githubusercontent.com/IOServicesLabs/indexio/main/install.ps1 \| iex` |
| npm | `npm install -g indexio@latest` |
| pip | `pip install -U indexio-cli` |
| Docker | `docker pull ghcr.io/ioserviceslabs/indexio:latest`, then recreate the container |
| Source | `cargo install --git https://github.com/IOServicesLabs/indexio indexio --force` |

Then do two more steps:

1. Run `indexio hook install`. It adds the hooks that the new version brings and repairs the
   others. It keeps your own hooks.
2. Restart your agent sessions. A running session keeps its indexio server until it
   restarts.

Running sessions do not stop you from upgrading:

- The two scripts replace the binary while sessions use it. On Windows, `install.ps1` moves
  the running `indexio.exe` to `indexio.old.exe` and copies the new one in.
- On Windows, npm and pip try to overwrite the binary in place, and Windows does not allow
  that while it runs. Close your agent sessions first, or use the script.
- Sessions that still run the old version can read an index that the new version wrote.
- The first new server does its one-time work in the background: it embeds files that have
  no vectors and it writes session cards. Calls are not blocked.

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

1. [Let your agent install it](#let-your-agent-install-it)
2. [What indexio does](#what-indexio-does)
3. [Add sources](#add-sources)
4. [Keep the index current](#keep-the-index-current)
5. [Embeddings](#embeddings)
6. [What is never indexed](#what-is-never-indexed)
7. [Secure the HTTP API](#secure-the-http-api)
8. [Team indexing: live worktree overlays](#team-indexing-live-worktree-overlays)
9. [Enterprise setup](#enterprise-setup)
10. [Use indexio with Claude Code](#use-indexio-with-claude-code)
11. [Token savings, measured](#token-savings-measured)
12. [Impact analysis](#impact-analysis)
13. [Use indexio from other tools](#use-indexio-from-other-tools)
14. [Command reference](#command-reference)
15. [Environment variables](#environment-variables)
16. [Deploy with Docker](#deploy-with-docker)
17. [Architecture](#architecture)
18. [Measured performance](#measured-performance)
19. [Limits](#limits)
20. [Development and releases](#development-and-releases)

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
`--include-archived`, `--full-clone`, `--dest DIR` (where remote clones go), `--no-embed`,
`--exclude PATH`.

**Folders with repositories and other files.** A folder often holds git repositories and
files that are not in any repository: notes, specs, exports, PDFs. `indexio add` indexes
each repository as a repository, and the other files as one more repository with the
folder's name:

```
projectA/                     indexio add ~/projectA
├── backend/   (git)    →     repository "backend"
├── frontend/  (git)    →     repository "frontend"
├── docs/notes.md       →     repository "projectA": the contents
├── docs/spec.pdf       →     repository "projectA": the name only
└── old-copy/           →     left out with --exclude old-copy
```

- Text files are indexed by contents, up to the size limit (256 KiB for text, 4 MiB for code).
- Documents (PDF, Word, Excel, PowerPoint, OpenDocument, RTF, EPUB) and text files over the
  limit are indexed by name only. `list_files` and a search for words of the name find them.
  `read_span` on one says that its contents are not indexed; read the file itself. The hooks
  let a Read of such a file through.
- Images, media, archives and other binary files are not indexed.
- `--exclude PATH` (repeatable, relative to the folder) keeps a folder out: it is not a
  repository and its files are not indexed. Excludes are remembered with the source and
  accumulate when you run `add` again.
- A folder that is registered on its own is not indexed a second time.

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
  ones that moved, and re-indexes changed plain folders by content (after the first pass,
  an unchanged file costs one stat).
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

## What is never indexed

- Files without a known code, document, configuration, script or data extension, such as
  databases, images, archives and compiled output.
- Binary content: a NUL byte in the first 8 KiB skips the file. Files over 4 MiB, and
  text files over 256 KiB. In a plain folder, these files and documents (PDF, Office) are
  indexed by name only (see [Add sources](#add-sources)).
- Build output, dependency and cache folders (`node_modules`, `target`, `dist`, `vendor`,
  any folder with a `CACHEDIR.TAG`), git-ignored files, hidden folders except the
  configuration ones (`.github`, `.cargo`, `.vscode`, `.devcontainer` …), lock, minified
  and source-map files.
- Credentials: `.env` files, private keys and certificates (`pem`, `key`, `p12`, `pfx`,
  `jks`, `gpg` …), `.netrc`, `.npmrc`, Terraform state. A credential file indexed by an
  earlier version is removed by the next sync. `.env.example` and `*.pub` are kept.

The session transcripts and run logs that `recall` searches are redacted before they are
stored: values of secret-looking settings, passwords in URLs, API keys, bearer tokens,
JWTs and private-key blocks are replaced with `<redacted>`.

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

## Team indexing: live worktree overlays

`indexio serve` can layer every developer's uncommitted changes on top of the
shared index, so an agent answering *you* sees *your* working tree — edits,
untracked files, deletions — while everyone else keeps seeing the pushed
state. Nothing to install on the developer machine beyond a shell hook in
the agent harness: no indexio client.

### How it works

1. The server indexes the pushed repositories (the usual `add` + `sync`).
2. A `PreToolUse` hook in the agent harness runs
   `integrations/claude-code/indexio-team-sync.sh` before every indexio tool
   call. The script diffs the working tree (untracked files included)
   against the merge-base with the remote default branch and POSTs the patch
   to `/team/worktree`.
3. The server applies the patch into a small overlay shard keyed by the
   caller's user name, layered over the base shards: changed paths shadow
   the base index, deleted paths are hidden, everything else reads from the
   shared index. Applying uses git plumbing on a scratch index — no
   checkouts, no worktrees.
4. A hash of the last accepted patch is kept on disk, so repeated calls
   with unchanged work cost nothing. The hook prints nothing and always
   exits 0: it can never block a tool call or pollute the agent's context.

Overlays expire after `INDEXIO_TEAM_TTL_DAYS` (default 7) without an update.

### Server setup

Serve with an ACL file whose tokens carry a `"user"` — that name keys the
overlay:

```bash
cat > acl.json <<'EOF'
{"tokens":{
  "tok-alice":{"allow":["*"],"user":"alice"},
  "tok-bob":  {"allow":["*"],"user":"bob"},
  "tok-ci":   {"allow":["*"]}
}}
EOF
indexio serve --acl-file acl.json --bind 0.0.0.0
```

A token without `"user"` works for reading but has no overlay (CI, above).
With a single `--auth-token` instead of an ACL file, clients pick their user
with the `X-Indexio-User` header.

### Developer setup

```bash
indexio setup claude --team-sync              # writes the hook script, prints the entry
export INDEXIO_URL=https://indexio.example.com
export INDEXIO_TOKEN=tok-alice
```

`setup claude --team-sync` writes the hook script into the Claude config
dir and prints the `PreToolUse` entry for `settings.json` — matcher
`mcp__indexio__.*`, so the sync runs right before any indexio tool call.
With one shared token instead of per-user tokens, also set `INDEXIO_USER`
to the name to act as.

### What is enforced server-side

- Credential paths (`Lang::is_secret_path`) are never accepted into an
  overlay, and neither are patches over 32 MiB.
- ACL `allow` patterns filter what each user reads — on the documents
  themselves for MCP (rendered text cannot be un-shown), as hit-filtering
  on the REST API.
- Overlay user names are constrained to `[A-Za-z0-9._@-]`, no leading dot,
  so overlay state stays path-safe (SSO identities are usually emails).

### Limits

- Overlays are lexical planes. Vector search ranks the base content; edits
  to tracked files surface through every tool, while brand-new untracked
  files and deletions are visible to the lexical, symbol and outline tools.
- The base must be a commit the server has (the merge-base with
  `origin/HEAD`, else `origin/main`, `origin/master`, or the upstream
  branch), otherwise the upload is refused with HTTP 409 and retried on the
  next call. The server fetches the origin itself, rate-limited to once a
  minute.
- The hook ships tuned for Claude Code; the `PreToolUse` contract is the
  one other harnesses are converging on, but only Claude Code is tested.

## Enterprise setup

Everything above runs on one machine. The same binary scales to a team
server or an organization-wide deployment; the moving parts are the
network edge and authentication.

### Inside the network (LAN / VPN)

The simplest topology: one indexio server on an internal host, bound to a
private interface; TLS optional on a trusted network.

```bash
indexio serve --bind 10.0.0.5 --acl-file acl.json
```

Developers aim their team-sync hook at it with `INDEXIO_URL` +
`INDEXIO_TOKEN`. Start with an ACL file; move to SSO (below) when
handing out tokens becomes the chore.

### Facing the internet

The server speaks plain HTTP and refuses to bind wide without
credentials, so put it behind a reverse proxy that terminates TLS and
forwards to the loopback port. A minimal Caddy site:

```
indexio.example.com {
	reverse_proxy 127.0.0.1:7717
}
```

Then serve with SSO rather than static tokens: a short-lived identity
token beats a long-lived bearer on a laptop.

### SSO with an OIDC provider (Entra ID, Google, Keycloak)

`--oidc-issuer` + `--oidc-audience` replace the static-token modes. The
server discovers the provider at
`<issuer>/.well-known/openid-configuration`, caches its JWKS, and
validates every bearer token (RS256, issuer, audience, expiry) before
looking the identity up — fail closed:

```bash
cat > users.json <<'EOF'
{"users":{
  "alice@corp.example":{"allow":["*"],"user":"alice"},
  "bob@corp.example":{"allow":["payments-*"]}
}}
EOF
indexio serve \
  --oidc-issuer https://login.microsoftonline.com/<tenant-id>/v2.0 \
  --oidc-audience <application-id-uri> \
  --acl-file users.json --bind 0.0.0.0
```

- The identity is the token's `email` claim (else `sub`). An identity
  not present in `users.json` gets HTTP 401: provisioning is your
  existing directory, mirrored into this file by hand or by a sync job.
- `allow` works exactly like the token ACL; `/admin` still needs `*`.
  `"user"` sets the team-overlay name — without it the email is used.
- IdP specifics. Entra ID: an app registration with an exposed API; the
  audience is the Application ID URI. Google Workspace: a Web
  application OAuth client; the audience is the client ID. Keycloak: a
  confidential client in your realm; the audience is the client ID and
  the issuer `https://<host>/realms/<realm>`.
- Signing keys rotate without a restart: an unknown `kid` refetches the
  JWKS on demand.
- OIDC mode supersedes `--auth-token`/`INDEXIO_AUTH_TOKEN`; leave them
  unset. `--oidc-audience` requires `--oidc-issuer`.

### Try it: Google Workspace in ten minutes

The exact flow, verified end-to-end against a live server on 2026-09-27.

1. Google Cloud console: create a project (`indexio-pilot`, say).
2. **APIs & Services → OAuth consent screen**: External; app name
   `indexio pilot`; scopes `openid` and `email`; add your own Google
   account under **Test users** — while the app is in "Testing", only
   test users can consent. This is the step everyone forgets.
3. **Credentials → Create OAuth client ID → Web application**: add
   authorized redirect URI `https://developers.google.com/oauthplayground`.
4. Mint a real ID token at
   https://developers.google.com/oauthplayground: gear icon → "Use your
   own OAuth credentials" → paste the client ID and secret → scope
   `openid email` → **Authorize APIs** → consent → **Exchange
   authorization code for tokens** → copy the `id_token`.
5. Serve with the client ID as the audience and your email in the users
   file:

   ```bash
   cat > users.json <<'EOF'
   {"users":{"you@corp.example":{"allow":["*"],"user":"pilot-you"}}}
   EOF
   indexio serve --bind 0.0.0.0 \
     --oidc-issuer https://accounts.google.com \
     --oidc-audience <client-id>.apps.googleusercontent.com \
     --acl-file users.json
   ```

6. Smoke test from anywhere:

   ```bash
   curl -H "Authorization: Bearer $ID_TOKEN" http://indexio.example.com:7717/search?q=parse
   # 200 with the token; 401 without. The token lives one hour — re-run
   # the playground exchange for a fresh one.
   ```

Beyond the playground, anything that can run the OAuth consent flow can
produce a bearer the server accepts: an IDE plugin, a small login page
in front of the server, or a CI job.

### GitHub

GitHub is not a general OIDC provider for your own API, but two
patterns cover it:

- **GitHub Actions → indexio is OIDC already.** A job with
  `permissions: id-token: write` requests a JWT (issuer
  `https://token.actions.githubusercontent.com`, audience of your
  choosing) and hands it to indexio clients as the bearer token. The
  server validates it like any OIDC token:

  ```bash
  indexio serve --oidc-issuer https://token.actions.githubusercontent.com \
    --oidc-audience indexio --acl-file users.json
  ```

  CI machines hold no static token, and the allow list in `users.json`
  scopes what each workflow may read.
- **Humans with GitHub accounts.** Put an OIDC gateway in front
  (oauth2-proxy configured for GitHub) and keep ACL-file tokens between
  gateway and indexio, or move human access to Entra ID / Google, where
  directory-backed provisioning belongs.

### Humans vs agents

Keep the two populations on different credentials: humans on SSO
short-lived tokens, agents and CI on ACL-file tokens scoped to the
repositories they touch (`allow: ["payments-*"]`). Revoking a human is
disabling their directory account; revoking an agent is deleting one
line of the ACL file.

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

This writes four Claude Code hooks into `~/.claude/settings.json`:

| Hook | Effect |
|---|---|
| PreToolUse, Bash | A shell read or search of an indexed file (`cat`, `sed -n`, `head`, `grep`, `rg`, `find`, a `python -c` or `node -e` one-liner that only opens the file) is refused. The refusal names the indexio call that gives the same result. A command that also does other work, for example a script or a build, runs as typed. |
| PreToolUse, Read | A whole-file Read of an indexed file is refused. The refusal names `file_outline` and `read_span`. |
| PreCompact | Session transcripts are imported for the `recall` tool. |
| SessionStart | A new session gets one line about the previous session in the same project: its title, dates, files changed, commits and the `recall` call for its card (70–115 tokens). After a compaction, the session gets its own changed files and commits back (75–300 tokens). A resumed session gets nothing. |

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
| `file_outline` | Every definition of a file with its start and end lines, or the headings of a markdown file. A few hundred tokens instead of the file. The tests of a `mod tests` or a `Test…` class are on one line: name and start line. |
| `read_span` | An exact line range of an indexed file. Without an end line, the whole definition at the start line. |
| `impact_of_symbol` | The transitive callers of a symbol across all repositories. |
| `impact_of_diff` | The callers and importers touched by a patch or by the uncommitted working tree. |
| `refresh_index` | A delta re-index, embed and reload from inside the session. |
| `recall` | Search of earlier sessions and of stored command output: requests, answers, tool calls, results, build and test logs. A lone hit comes with its exchange. With `brief: true`, it returns session cards instead: about 1,000 tokens per session with the requests, the files changed, the commits and the outcome. `query: ""` gives the latest sessions of this project. |
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
indexio embcas-stats --coverage             # per repository: indexed files without vectors
```

### Serve

```bash
indexio mcp                                 # MCP over stdio
indexio mcp --repo payments                 # pin the session's repository
indexio serve                               # HTTP on 127.0.0.1:7717
indexio serve --port 8080 --auth-token $(openssl rand -hex 32)
indexio serve --bind 0.0.0.0 --auth-token "$TOKEN"          # beyond localhost: a token is mandatory
indexio serve --acl-file acl.json
indexio serve --oidc-issuer https://idp.example.com --oidc-audience indexio-api --acl-file users.json
# team worktree overlays: POST /team/worktree (see "Team indexing" above)
```

### Agent setup

```bash
indexio setup claude
indexio setup claude --claude-md ~/.claude/CLAUDE.md
indexio setup claude --team-sync            # also install the team worktree-sync hook
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
| `INDEXIO_TEAM_TTL_DAYS` | Days a team worktree overlay is kept without an update. Default 7. |
| `INDEXIO_URL`, `INDEXIO_TOKEN`, `INDEXIO_USER` | The team-sync hook's server, bearer token, and user to act as (only with a shared token). |
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

**If you are indexing your own machine for your own agent, install the binary instead.**
The container is for a shared index that several people or machines query over HTTP. On
your own laptop it costs you the Bash hooks, the instant working-tree refresh and a layer
of path translation, and buys nothing. Use [Install](#install) above. The rest of this
section is for the shared case, and for trying indexio without installing anything.

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

### How the container indexes code on your machine

The container has no access to your disk beyond what you bind-mount into it. Two mounts
do the work, and they are not symmetric:

| Mount | Purpose | Access needed |
|---|---|---|
| `-v ~/code:/repos:ro` | The code to index. Mount anything: one repository, a checkout root, several `-v` flags. | Read-only is enough. Indexing never writes to your source. |
| `-v indexio-data:/data` | The index itself. | Writable. This is a Docker volume, not a folder on your host. |

`indexio add /repos` walks the mount, registers every git repository under it at any
depth, and indexes each one. A folder with no git repository in it is indexed as a plain
tree. Each repository is recorded in `/data/repos/<name>.json` with the path **as the
container sees it** — `/repos/my-service`, never `C:\Users\you\code\my-service`. Results
are repository-relative (`my-service:src/main.rs`), so searching is unaffected, but the
host path is nowhere in the index.

Three consequences worth knowing before you build a large index:

- **A container index and a host index are not interchangeable.** If you point the host
  binary at a data directory built in a container, the recorded `/repos/...` paths do not
  exist on the host, so re-indexing and `read_span` on unindexed content fail. Pick one
  and stay with it, or keep two data directories.
- **Freshness comes from the sync loop, not from a file watcher.** `indexio sync` on a
  local folder re-discovers the repositories under the root and delta re-indexes whatever
  is on disk right now. It does not clone, pull, or write to your source — only the remote
  source kinds (`github:`, `azdo:`, a git URL) fetch anything. The `sync` service in
  `docker-compose.yml` runs that every 15 minutes.
- **The auto-refresh does not survive the mount.** `indexio mcp` normally watches the
  current repository and re-indexes the working tree in about 25 ms after an edit.
  Filesystem events do not cross a Docker Desktop bind mount from a Windows or macOS host
  reliably, so in a container treat the index as fresh to the last `sync` or
  `refresh_index` call, not to the last keystroke. This is the main reason to run the host
  binary next to an agent that is editing code.

The Bash hooks (`indexio hook install`) also need the host binary: they rewrite commands
in your shell, which a container cannot see.

On Windows, generate the serve token in PowerShell rather than with the bash line above:

```powershell
$bytes = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
$env:INDEXIO_AUTH_TOKEN = [System.BitConverter]::ToString($bytes).Replace('-','').ToLower()
```

The image sets `INDEXIO_BIND=0.0.0.0`, so `serve` refuses to start without
`INDEXIO_AUTH_TOKEN` or `--acl-file`: an index of all your source is not something to
publish unauthenticated. Do not "fix" that by setting `INDEXIO_BIND=127.0.0.1` in the
container — the service then binds the container's own loopback and published ports
cannot reach it. Keep the bind at `0.0.0.0`, set a token, and publish to
`127.0.0.1:7717:7717` so only your machine can connect.

### Continuously index one folder

Say the code lives in `C:\codebuilds` and you want it re-indexed on a timer, with an
agent querying it over MCP. You need no auth token for this: the token guards `serve`,
the HTTP API, and this setup does not run it.

Register the folder and build the index once. `add` records `/repos` as a persistent
source, which is what later syncs re-read:

```powershell
docker run --rm -v indexio-data:/data -v C:/codebuilds:/repos:ro `
  ghcr.io/ioserviceslabs/indexio add /repos
```

Then leave a container running that re-syncs on a loop. The image's entrypoint is the
binary, so override it to get a shell:

```powershell
docker run -d --name indexio-sync --restart unless-stopped `
  -v indexio-data:/data -v C:/codebuilds:/repos:ro `
  --entrypoint /bin/sh ghcr.io/ioserviceslabs/indexio `
  -c "while true; do indexio sync || true; sleep 900; done"
```

Point your agent at the same volume. The MCP server is stdio, so it needs no port and no
token:

```powershell
claude mcp add indexio -- docker run -i --rm -v indexio-data:/data -v C:/codebuilds:/repos:ro ghcr.io/ioserviceslabs/indexio mcp
```

Check it with `docker run --rm -v indexio-data:/data ghcr.io/ioserviceslabs/indexio stats`,
which should list every repository found under `C:\codebuilds`.

Use forward slashes in the mount (`C:/codebuilds`), and make sure the drive is shared in
Docker Desktop's file-sharing settings. Lower the `sleep` if 15 minutes is too coarse —
but an edit is visible only after the next sync, so if you want the index to track your
edits as you make them, run the host binary instead.

To use `docker-compose.yml` for the same thing, set the folder and start only the sync
service:

```powershell
$env:INDEXIO_REPOS = "C:/codebuilds"
docker compose up -d sync
docker compose exec sync indexio add /repos
```

Compose interpolates the whole file before it selects a service, so
`INDEXIO_AUTH_TOKEN` must still be set for the file to parse even when you are not
starting the `indexio` service. Set it as above, or use the plain `docker run` form.

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
cargo test --workspace                  # every crate, 325 tests
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
   silicon, Intel) and Windows (x86_64), attaches each archive with a `.sha256` and a pip
   wheel to the release, publishes the container image to `ghcr.io`, publishes the npm
   package when the `NPM_TOKEN` secret exists, and publishes the `indexio-cli` wheels to
   PyPI when the `PYPI_API_TOKEN` secret exists. These jobs compile Rust and take some
   minutes.

`tools/release-assets.sh` builds and attaches one asset from a maintainer machine, for a
platform the workflow does not cover.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
