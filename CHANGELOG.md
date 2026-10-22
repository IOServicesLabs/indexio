# Changelog

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
