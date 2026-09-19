# SPEC-P8 — Harness integration: make the index the default, not grep

Addendum to SPEC/SPEC-P2..P7. Motivation (user): "I would like the MCP server to be more
included, so instead of always running grep and relying on the harness we can use the
index." A coding agent decides between its built-in tools (Grep, Glob, Read, find) and MCP
tools from three signals: the server's `instructions` (injected into the system prompt by
Claude Code and others), each tool's description, and any project/user guidance file.
P8 works all three, and closes the two gaps that made an agent fall back to grep: no way
to list files, and no way to fix a stale index without leaving the session.

## 1. `initialize` → `instructions` (dynamic)

`McpServer` answers `initialize` with an `instructions` string built from the live index:
repo count, file count, the first 20 repos as `name (path, synced <time>)`, then the rule
"PREFER these tools over grep/rg/Glob/find/Read for anything in an indexed repo", the
tool-by-tool mapping, and the freshness contract (last synced commit; uncommitted edits
only visible to `impact_of_diff`; call `refresh_index` when stale; local tools only for
files/repos not indexed). Kept under ~3 KB — it lands in every prompt.

## 2. Tool descriptions name the built-in they replace

`code_search`: "USE THIS INSTEAD OF grep/rg/Grep …"; `list_files`: "… INSTEAD OF
Glob/find/ls"; `file_outline`: "USE THIS BEFORE READING A FILE"; `read_span`: "… INSTEAD
OF reading a whole file"; `code_grep`, `find_symbol`, `who_calls`: "use instead of
grepping for …"; `refresh_index`: "call this when results look stale …".

## 3. New tools

- `list_files {pattern, repo?, limit?=200}` → `{files:[{repo,path}], truncated}`.
  `Engine::list_paths`: a glob when the pattern contains `*`/`?` (`**` crosses
  directories, `*`/`?` do not; anchored; case-insensitive), else a case-insensitive
  substring; sorted by (repo, path).
- `refresh_index {repo?, embed?=true}` → delta re-index of every registered repo (or one)
  from its current HEAD via `reindex_repo`, `embed_repos_with` for the repos whose docs
  changed, then the server swaps in a freshly opened `Engine` (`RwLock<Engine>`), so the
  next call sees the new shards without a restart. Returns per-repo counts,
  `changed_repos`, `embedded_chunks`, `reloaded`.
- `index_stats` gains `repos_detail: [{name, path, commit, indexed_at, plain}]`.

`McpServer { engine: RwLock<Engine>, embedder, reranker }` replaces the free `handle`
function (kept under `#[cfg(test)]` for the unit tests).

## 4. Guidance file + setup command

`mcp::HARNESS_GUIDANCE` is the block for a CLAUDE.md (project or `~/.claude/CLAUDE.md`).
`indexio setup claude [--scope user|project|local] [--claude-md PATH] [--print-only]` runs
`claude mcp add --scope … indexio -- <this exe> mcp --data-dir <data dir>` and prints the
block (or appends it, idempotently, with `--claude-md`).

## 5. Tests

instructions contain repos + every tool name + "grep" and stay short; descriptions contain
the "INSTEAD OF" phrases; `list_files` glob/substring/repo/limit/`*` not crossing `/`;
`index_stats.repos_detail`; `refresh_index` no-op reload + argument validation; CLI parse
of `setup`; end-to-end: commit in a registered repo → `find_symbol` misses →
`refresh_index` → same MCP process finds it, `list_files` lists it.
