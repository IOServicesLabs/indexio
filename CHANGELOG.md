# Changelog

## 0.1.15

TypeScript and React apps: type aliases are definitions, JSX elements are call
sites, and a symbol lookup no longer answers with a longer name when the exact
one exists. Measured on copies of three generated Next.js apps (132 TS/TSX
files) against grep:

| Case | 0.1.14 | 0.1.15 |
|---|---|---|
| `type X = …` aliases found by `symbol X` | 0 of 34 | 34 of 34 |
| JSX render sites (`<Board …/>`) found by `calls Board`, context providers aside | 0 of 62 | 62 of 62 |
| Render sites listed by `impact --file components/X.tsx` (depth 1) | 0 of 33 | 33 of 33 |
| Alias names answered only with longer names (`Status` → `StatusChart`) | 19 of 22 | 0 of 22 |

- `type X = …` and `export type X = …` in `.ts`/`.tsx`/`.js`/`.jsx` are
  indexed as definitions of kind `type`, so `find_symbol`, `file_outline` and
  `read_span` see them.
- A JSX element that renders a component is a call of that component:
  `<Board />` and `<AppShell>…</AppShell>` (one site, at the opening tag), and
  `<ui.Button>` as a call of `Button` (a member name matches by its last
  segment, as `a.b()` does). Lowercase intrinsic elements (`<div>`,
  `<motion.div>`), `<Ctx.Provider>`/`<Ctx.Consumer>` and an old-style `<Foo>x`
  type assertion in a `.ts` file are not. `who_calls`, `calls` and `impact`
  now list the files and functions that render a component.
- A function bound to a module-level `const` is its calls' caller:
  `const Board = () => …`, `export const Card = memo(function () {…})`. Their
  calls were filed under the top level before, so an impact walk stopped there.
  Functions bound with `let` or `var`, and an anonymous `export default
  function`, still file their calls under the top level.
- `Status` returned `StatusChart`: the lookup was already exact first, but
  `Status` was a type alias, not indexed, so no definition matched exactly and
  the substring fallback answered. With aliases indexed the exact definition
  answers alone. When the fallback does answer, `find_symbol` now says so on
  its first line (`no definition named exactly …`) and `indexio symbol` prints
  that note on stderr, so a longer name is not read as the definition.

### Upgrading an existing index

Indexes built before 0.1.15 hold no aliases or render sites for files that have
not changed since. No manual step is needed: the extraction cache keys TS/JS
entries by an extractor generation, and each repo records the generation it was
built with. The next `indexio sync`, `indexio reindex` or MCP auto-refresh of a
repo re-extracts its unchanged TS/JS files once, and only those (other
languages keep their cached results). The background sync of an MCP server
treats a repo built by an older version as due. Until a repo has had that pass,
its TS/JS symbols and calls are the 0.1.14 ones. The shard format is unchanged,
and an older binary still reads the index. Restart or upgrade every indexio MCP
server that shares the data dir: an older binary drops the recorded generation
each time it rewrites a repo's state, so until then 0.1.15 re-extracts that
repo's TS/JS files on its next sync.

### Memory

Memory. One MCP server runs per agent session, and ten of them on one data
dir held 8 to 9 GB between them, single servers 1 to 2 GB, with peaks near
10 GB. A server now starts in under 20 MB, stays in the low hundreds under
constant edits, and goes back to about 15 MB when it is idle. No change to the
on-disk formats; 0.1.14 servers can share the data dir with new ones.

Measured with one server on a copy of a live data dir (57 repos, 986k vector
rows), an edit every 2 seconds and tool calls throughout, for 12 minutes:

| | 0.1.14 | 0.1.15 |
|---|---|---|
| Fresh server, before any call | 250 MB | 6–19 MB |
| Peak during the run | 2,328 MB | 238 MB |
| End of the run | 1,400 MB | 220 MB |
| Heap actually in use at the end | 288 MB | 55 MB |
| Largest maintenance spike | ~2,000 MB, inside the server | 50 MB, in a short-lived child |
| Threads | 100 | 37 |
| Hybrid search, median | 440 ms | 166 ms |
| Lexical search, median | 227 ms | 90 ms |

Memory figures are private bytes of the process.

| Operation | 0.1.14 | 0.1.15 |
|---|---|---|
| Lexical merge of a real 8-shard window | 1,674 MB, 19.3 s | 81 MB, 10 s |
| Full vector merge, 137k rows | 1,692 MB, 7.8 s | 23 MB, 2.5 s |
| Full vector merge, 986k rows | not run (about 10 GB by extrapolation) | 82 MB, 20 s |
| Full re-embed, 137k rows | 2,378 MB | 814 MB |
| Model save | 500 MB | 2.6 MB |
| CLI `embed` with nothing to do | 545 MB | 80 MB |
| `index_stats`, peak | 436–445 MB | 58 MB |

- Vector row metadata (hash, repo, path, lines of every row) is read from the
  mapped segment instead of being parsed into the heap: 193 MB per server at
  986k rows, and as much again on every embed, which parsed a second copy.
  Servers and passes in one process share an open segment.
- Lexical shard merges stream from the mapped inputs into the new shard
  (output byte-identical to 0.1.14's) and the merge window takes the small
  shards first, so a few megabytes of deltas no longer rewrite the largest
  shard. Compaction starts above 10 shards and merges down to 8 (was 12 and
  6). A posting list found out of order is sorted instead of failing the merge.
- Vector and BM25 segments are written a row at a time: embeds, folds, full
  merges and rebuilds no longer hold every row of the plane. A rebuild needs
  about twice the segment size in free disk space while it runs.
- A compaction copies the int8 rows it carries. Until now it took each row
  through f32 and the quantizer again, which set the binary-code bit of every
  component stored as zero (about 6% of the code bits of a real segment) and
  so changed what the prescan of a semantic search picked after each
  compaction.
- The model's label cache is a fixed 4 MiB table (it grew without bound, up to
  several hundred MB), and a model save streams the snapshot instead of
  building it in memory. A server saves the model when it is dirty, 5 minutes
  have passed since the last save, and either 64 new texts were observed or
  the oldest unsaved one has waited an hour.
- Compaction, vector folds and the drift re-embed run in a short-lived child
  process (`indexio maintain`, below-normal priority, one at a time per data
  dir under `compact.lock`), so their memory goes back to the system when
  they finish instead of staying in the server's heap. Each pass writes a line
  to `<data dir>/maintain.log`.
- A server idle for 10 minutes drops its caches, parked engines and free heap,
  and no longer opens an engine for every shard another process writes. The
  first call after that pays for the reload (about 100 ms more on a hybrid
  search in the test).
- `indexio mcp` no longer starts a tokio runtime (32 threads), and its two
  worker pools are capped at 16 threads.
- The server's per-session maps (`read_span` history, repair attempts) are
  bounded.

Settings (environment):

| Variable | Default | Meaning |
|---|---|---|
| `INDEXIO_MCP_THREADS` | 16 | Threads per worker pool of an MCP server; 0 = no cap |
| `INDEXIO_IDLE_TRIM_SECS` | 600 | Idle time before a server drops what it can rebuild; 0 = never |
| `INDEXIO_PREPARE_WINDOW_SECS` | 300 | How recently a call must have been made for a server to open an engine ahead of the next one |
| `INDEXIO_IDLE_HEAP_RELEASE` | on | Ask the allocator to return free memory at the idle trim |
| `INDEXIO_MAINT_INPROCESS` | off | Run maintenance inside the server, as 0.1.14 did |
| `INDEXIO_MAINT_TIMEOUT_SECS` | 3600 | Deadline of one maintenance child |

Checked on copies of a live data dir: 636 MCP calls replayed from real
sessions, 619 answers byte-identical to 0.1.14 and the other 17 identical once
both versions read the same copy (the two copies had synced the live repos at
different moments); a new server and 0.1.14 servers on one data dir for 5
minutes without an error; no file lost its vector rows.

Known and unchanged: a semantic search rescans at most `max(8k, 256)`
candidates per vector segment whatever the segment's size, so its answers
depend on how the rows are spread over segments and change when segments are
merged, in 0.1.15 as in 0.1.14.

## 0.1.14

Fixes a deadlock that left a long-running MCP server silent until it was
restarted. No change to search results or tool output.

- A background embed (a transcript import, a sync of other repos) that
  overlapped an edit's embed or the periodic model save could block every
  worker thread on the semantic model's lock while the holder of that lock
  waited for those same threads. The server then used no CPU and answered no
  further tool call; the harness gave up on each one after its 30 minute idle
  timeout. Work done under the model lock now runs on its own thread pool, the
  write lock is held per batch, and the server's call path no longer takes the
  lock at all.
- The per-repo write lock gives up at its deadline when the lock file can be
  neither created nor removed, instead of retrying without a pause.
- A/B against 0.1.12 on copies of a live data dir: 102 MCP calls with the same
  answers, median 29.0 → 31.6 ms, and a full `embed --rebuild-model` that
  writes a byte-identical model.

## 0.1.13

Cleanup of the CLI help and the docs, plus an rtk hook fix. No change to search
results or tool output.

- `indexio --help` and the subcommand help no longer carry internal design-doc
  tags such as `(SPEC-P7)`. The hook help now describes the in-place rewrite
  instead of the deny that 0.1.11 replaced, and lists the `powershell` and
  `session-start` hooks. The `--rerank` help says what it does without a
  reranker endpoint: nothing.
- `indexio hook install` keeps rtk's command in settings under a matcher that
  never runs, so rtk sees its hook and stops printing "No hook installed", and
  `rtk init -g` does not add the competing hook back.
- Docs: macOS is listed as build-from-source, since releases carry no macOS
  binaries. The README describes the PowerShell hook and `# raw` as they work
  now, and `docs/DEVELOPMENT.md` drops stale counts and commands.

## 0.1.12

0.1.11's in-place answers lost to rtk: with both hooks installed, a `grep`,
`cat`, `head` or `tail` that indexio answered was also rewritten by rtk's
Bash hook (`rtk grep`, `rtk read`), and Claude Code ran rtk's version. That
was 204 of the 224 commands 0.1.11 answers in the A/B replay.

- `indexio hook install` moves rtk's `rtk hook claude` entry into indexio's
  Bash hook (`hook bash --then 'rtk hook claude'`): indexio answers what the
  index can, and every other command goes to rtk unchanged, so each call gets
  one rewrite. Run `indexio hook install` again after `rtk init`.
- `indexio hook bash --then '<hook command>'` chains any hook the same way.

## 0.1.11

Shell reads are answered from the index in place instead of being denied.
A deny showed up as a hook error, cost a round trip, and stranded sessions
that could not call the indexio tools (MCP server disconnected, tools deferred
or missing in a subagent). A/B tested against 0.1.10 on a copy of the real
index by replaying the 570 Bash and PowerShell commands the hooks denied over
two weeks of Claude Code sessions.

### A/B results

| Case | 0.1.10 | 0.1.11 |
|---|---|---|
| Denied shell commands (570 replayed) | 302 still denied | 0 denied: 225 answered from the index, the rest run as typed |
| Output of the 217 answered commands that could be re-run | 218.7k tokens | 137.4k tokens (-37%) |
| Exit status differing from the typed command | — | 0 of 217 |
| `grep -n` answers missing a line the typed grep prints | — | 0 of 126 |
| Hook time, median | 37 ms | 14 ms |

### Hooks

- The Bash and PowerShell hooks rewrite a read or search the index answers
  exactly to `{ indexio tool <name> <args> || <typed pipeline>; }` and allow
  it. The typed command runs if the index cannot answer (tool error, a grep
  with no matches, a grep longer than the index lists).
- Only exact equivalents are served: grep flag bundles (`-nE`, `-rniw`),
  `-F/-w/-x/-i/-m/-A` are modelled; `-v/-c/-l/-B`, `--include`, several
  patterns, `sed` transforms, a filter after a read, comments and command
  substitutions run as typed. `grep -E` patterns are no longer mangled as
  basic regexes, and `\s`/`[^…]` never match across lines.
- PowerShell: `| Select-Object -First/-Last N` after `Select-String` is kept;
  `| ForEach-Object`/`Where-Object` and filtered `Get-ChildItem` run as typed.
- The Read hook still refuses whole-file reads of long indexed files, but lets
  them through when the caller cannot call indexio: its transcript shows
  `No such tool available: mcp__indexio__…`, or it is an agent whose `tools:`
  exclude indexio.
- New `indexio tool <name> [json|base64url]` runs one MCP tool from the shell.

### Search

- `code_grep` with `case:no` is case-insensitive (it was ignored: the pattern
  is a regex and `case:` only reached literals).
- `path:` filters take anchors: `path:^Cargo.toml$` is that one file,
  `path:^src/` a folder; a bare `path:x` is still a substring.

## 0.1.10

Hooks that follow how agents actually read files, a PowerShell hook, Claude Code
memory notes in `recall`, fewer repeated answers, and compaction that no longer
rewrites the whole index. 30 changes, every one A/B tested against 0.1.9 on a
copy of a real 49-repo, 12k-file index by replaying real calls from two weeks of
Claude Code sessions.

### A/B results

Measured on Windows 11, 32 cores. "Before" is 0.1.9.

| Case | 0.1.9 | 0.1.10 |
|---|---|---|
| Greps with 5+ alternatives (1,972 real calls) | some files dropped | 82 calls gain lines, 0 lose any |
| Hook that cannot decide in time | waits for the 10.7 s timeout | allows after 3 s |
| Time to deny a read | 40.8 ms median | 17.8 ms median |
| PowerShell reads of indexed files | not hooked | routed to indexio (108 denies over 2,065 calls, 394.6k chars) |
| Identical `read_span` answers repeated in a session | sent in full | one line; -10.3% `read_span` output |
| Repeated `code_grep` / `code_search` / `file_outline` answers | sent in full | -5.9% / -12.3% / -15.1% output |
| Reads of an agent worktree's identical copy | not hooked | routed to indexio (0.93M of 1.75M chars) |
| Failed calls from real sessions (55) | 39 answered | 45 answered, 0 broken |
| Grep across all repos (`fn locate`) | 17.4 ms | 9.7 ms |
| Shard compaction while shards are being written | 28-30 s, ~2.5 GB | 13-25 ms, 220 MB or less |
| `indexio compact` of 48 shards | 32.0 s | 0.9 s |
| Vector compaction of 13 segments | 11.2 s, 2.8 GB | 0.8 s, 101 MB |
| Hybrid answers after vector compaction (vs before it) | 83% identical | 100% identical |

1,500 real hook calls and 1,520 real search calls return the same verdicts and
answers as 0.1.9 wherever a change was not meant to alter them.

### Hooks

- New PreToolUse hook for PowerShell: `Get-Content`, `Select-String`,
  `Get-ChildItem | Select-String`, `-Tail N`, and `$c = Get-Content f` followed
  by line ranges or loops get the same treatment as the Bash hook. Run
  `indexio hook install` again to add it.
- A hook gives a verdict within 3 s or allows the call.
- The re-index after a deny runs in a detached process, and is skipped for the
  session's own repo (the server already watches it).
- Reads of 60 lines or less go through, whether a Read range, `sed -n`,
  `head`, `tail`, `Get-Content`, or a whole file that short.
- Sending a denied call again unchanged lets it through.
- Byte views and counts (`cat -A`, `head -c`, `tail -f`, `| wc`,
  `| uniq -c`), `grep -c/-l/-L/-q/-o` and Python byte-analysis one-liners are
  no longer denied.
- `tail -n N` and multi-range `sed -n 'A,Bp;C,Dp'` are answered with exact
  `read_span` ranges instead of a whole-file outline.
- A read of a file in `.claude/worktrees/<wt>/` that is byte-identical to the
  indexed file is redirected to it; an edited copy is left alone.

### Search and lookups

- Regex prefilters keep every alternation branch (5+ branches lost files).
- `(?i)` regexes use the trigram index instead of a brute-force scan.
- Lookaround regexes return a superset with a note instead of an error.
- `code_grep` and lexical `code_search` leave session transcripts out of the
  search itself unless asked for (`repo:sessions`).
- File tools without `repo` use the session repo when it has the path, else
  the one repo that does; `repo` plus a path that starts with the repo folder
  name resolves.
- `code_search` accepts `code_grep`'s `pattern`, `repo` and `path`.
- An answer identical to one the session got in the last 20 minutes from
  `read_span`, `code_grep`, `code_search` or `file_outline` (400+ chars) comes
  back as one line. Sending it again returns the text; a compaction or
  `/clear` resets this.
- `read_span`, `code_grep`, `file_outline`, `code_search`, `recall` and
  `list_files` are marked always-load, so Claude Code skips a ToolSearch round
  trip before using them.

### Sources

- Claude Code memory notes (`~/.claude/projects/*/memory/`) are indexed with
  the transcripts; `recall` searches them, this project's notes first.
- A session whose unregistered checkout sits beside three or more registered
  repos adds it (git or a plain project folder up to 20,000 files, never a
  folder of repos). Opt out with `INDEXIO_NO_AUTO_ADD` or
  `<data_dir>/auto-add-skip`.
- A repo with no commit yet is indexed from its working tree.
- Config files opened by name are indexed: `Caddyfile`, `Dockerfile.*`,
  `.gitignore`, `.dockerignore`, `*.example`, `*.inf`, `*.patch` and similar.
  Secret files are still skipped.

### Compaction

- Shard compaction merges the cheapest run of adjacent shards instead of
  always the oldest, so the big base shard is not rewritten for every few
  small deltas. Once tombstones pass 30% the oldest run is merged, which drops
  dead docs.
- Vector compaction merges only the small segments when that is enough and
  carries the large one over untouched.

## 0.1.9

Faster freshness, a self-healing semantic plane, a team server built for many
developers at once, and a much cheaper `impact_of_diff`. 57 changes, every one
A/B tested against the build before it on a copy of a real 49-repo, 12k-file
index. Replays of 1,500+ real MCP calls from a week of Claude Code sessions
return byte-identical output except where a change meant to alter it.

### A/B results

Measured on Windows 11, 32 cores. "Before" is 0.1.8 unless the row says
otherwise.

**Local sessions (stdio MCP server)**

| Case | 0.1.8 | 0.1.9 |
|---|---|---|
| Next call after a file is touched | 50 ms | 0.8 ms |
| First call after an edit | 93 ms | 50 ms |
| Second call after an edit | 20 ms | 0.5 ms |
| First call after another session writes the index | 22 ms | 3 ms |
| First refresh in a new session (large repo, cold) | 5.4 s | 1 ms |
| Transcript import before a compaction | 1.2 s | 0.2 s |
| Files with no semantic vectors (wedged plane) | 986 | 36 (empty files) |
| `impact_of_diff` on a 158-file working tree | 2.8 s, 140 KB | 0.4–0.6 s, 23 KB |
| `impact_of_diff` text, small / medium / large diff | 756 / 5.8k / 12.8k tokens | 603 / 4.6k / 4.9k tokens |
| `code_search` p50 over 1,529 real calls | 14.5 ms | 12.4 ms |
| Bash/Read hook denying a file read | 115 ms | 21–23 ms |
| Bash hook denying a grep | 115 ms | 57 ms |

Replay of 1,862 real calls: 9 calls that failed now answer, 4 greps that
returned nothing find the real code, 1 grep that matched the wrong file
matches the right one, 11 empty greps say the path filter matches no indexed
file, outlines are 2.5–4.3% smaller. 600 real hook calls: identical decisions.

**Team server (`indexio serve`)**

| Case | 0.1.8 | 0.1.9 |
|---|---|---|
| Light call while 16 users run heavy queries (p50 / p95) | 182 / 630 ms | 1.3 / 14 ms |
| `/health` p99 under that load | 493 ms | 0.4 ms |
| Heavy-query throughput, 16 users | 1x | 5.9x (3.3x, then 1.8x, each A/B'd against the build before) |
| A sync's changes visible to users | after `/admin/reload` | ~1 ms after the sync |
| First hybrid search after startup | 77–192 ms | 20 ms |
| Memory per developer with an overlay (24 users) | 33.6 MB | 1.2 MB |
| First call after a developer's edit | 36 ms | 0.5 ms |
| Server time to apply an uploaded edit | ~166 ms | ~95 ms |
| Team-sync hook on every indexio call (no new edits) | 281 ms | 113 ms |
| Bytes on the wire for 300 real calls | 824 KB | 370 KB |
| Remote answers identical to a local session's (400 calls) | 96% | 100% |

### Semantic plane

- Vector and BM25 segments are written as a pair and become visible together;
  a half-written pair left by a crash is reaped, which un-wedges embeds.
- Embed failures leave a marker and `index_stats` reports an unhealthy plane.
- Abandoned `.tmp` files are cleaned up.

### Freshness

- After an edit only the files the watcher saw change are re-checked; a
  `.gitignore` edit still triggers a full pass.
- Per-repo stat caches are saved to disk and shared by the server, the hooks,
  `refresh_index`, transcript and log imports, plain folders and the CLI.
- Index writes by other processes are reopened and warmed in the background;
  no double reopen after the server's own edit, none after a background embed.
- One server per 10 minutes runs the background sync of other repos.

### Search and lookups

- `read_span` and `file_outline` default to the session's repo and accept
  `repo:path` and absolute paths.
- `code_grep`: doubled backslashes retried as single escapes, path filters
  with a repo name or an absolute path, `(a|b)` groups, and a note when the
  path filter matches no indexed file.
- Impact and `who_calls` ignore call sites in other language families;
  `who_calls` judges sites in output order and stops at the limit (hub names
  ~15x less CPU).
- Literals are verified rarest first, so candidates missing a rare word fail
  before the common words are scanned.
- Prose documents up to 2 MiB are indexed (was 256 KiB).
- Outlines drop redundant `N-N` ranges and `method` labels; impact changed
  lines print as ranges.
- Content and outline caches keep their working set (two generations) instead
  of clearing when full; the content cache is bounded by bytes.

### `impact_of_diff`

- Both sides of every changed file are parsed in parallel through a
  content-keyed outline cache; import searches run once per module key, side
  by side; the caller walk looks names up in parallel chunks.
- The report lists changed definitions once per file, names roots only when
  they add something, and caps very long lists (150 definitions, 40
  importers) with counts for the rest.

### Team server

- Requests run off the async runtime, and the engine is swapped without
  holding a lock across a request.
- The server follows `<data_dir>/shards` by itself and parses the semantic
  plane at startup and after each embed.
- Developer overlays sit on the one shared index instead of copying it, and
  are prebuilt right after each upload.
- One shared content cache, `INDEXIO_CONTENT_CACHE_MB` (default 1024), serves
  every user's searches.
- Small lookups stay off the shared thread pool while the server is busy.
- Responses are gzipped; the hook uploads large patches gzipped.
- Remote sessions get a current repo (`X-Indexio-Repo`, else the repo they
  last synced) and `impact_of_diff` over their synced worktree.
- Applying an upload takes 3 git calls instead of 6; overlay stamps are kept
  in memory.
- The team-sync hook needs two git calls when nothing changed.
- A relative `--data-dir` no longer breaks overlays.

### Hooks

- The indexed check is one scoped lookup; a denied single-file read refreshes
  that file only; repo roots are matched before canonicalizing them.

### Build

- `tools/build-release.sh` also retries rustc's illegal-instruction crash.
