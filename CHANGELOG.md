# Changelog

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
