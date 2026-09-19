# SPEC-P9 — Token-dense MCP results and the query hot path

Addendum to SPEC/SPEC-P2..P8. Motivation (user): the MCP server is the context source
for every other Claude Code session on the machine, so the goal is "the highest token
reduction possible and speed when doing lookups". Every byte of a tool result is paid
for in model tokens, and every millisecond is paid once per call per session.

Measurement harness: `bench/mcp_tokens.py` drives `indexio mcp` over stdio with a fixed
workload (stats, list_files, find_symbol, who_calls, file_outline, read_span, code_grep,
impact_of_symbol, 4 lexical + 4 hybrid searches from `bench/mcp_queries.json`), counts
tiktoken `cl100k_base` tokens per result and the min-of-3 latency, and diffs against a
saved run (`--compare`). Saved runs live next to it as `bench/mcp_tokens.<label>.json`.

## 1. Text results by default (`crates/indexio/src/mcp_text.rs`)

MCP tool results are rendered as dense plain text instead of the compact-JSON envelope.
`INDEXIO_MCP_FORMAT=json` on the server, or `format: "json"` in any call's arguments,
restores JSON (used by the e2e tests and by scripts).

| Tool | Text layout |
|---|---|
| `code_search` (lexical), `code_grep`, `find_symbol`, `who_calls` | grouped per file: `repo:path` header, `  line: snippet` rows, files in best-hit order, lines ascending; trailer `-- N hits in M files` (+ `truncated` hint) |
| `code_search` (hybrid/semantic), `semantic_search` | rank order kept; consecutive hits in one file share a header; no scores/ranks/nulls |
| `list_files` | one line per directory: `repo:dir/ a.rs b.rs c.rs` (names with spaces quoted) |
| `file_outline` | `start-end kind name`, indented by scope depth |
| `read_span` | `repo:path Lstart-end` header, then `line<TAB>text` (raw text, no JSON escaping) |
| `index_stats` | one summary line, the shared root folder once, then `name folder commit date` per repo |
| `impact_of_symbol` / `impact_of_diff` | definitions, then call sites grouped per file (`dN line: [symbol] in caller(): snippet`, at most 8 per file + `… +K more`), per-file tallies only for files not listed, importers, `-- complete` / `-- truncated` |
| `refresh_index` | one line per changed/failed repo + a summary line |

Snippets are trimmed to one whitespace-collapsed line, capped at 200 bytes. Hit
headers use the same `repo:path` spelling `read_span`/`file_outline` take as arguments.

Measured on the repo-a repo (bench workload, same hits): **−54 % tokens overall**
(list_files −71 %, hybrid hits −68 %, who_calls −57 %, impact −56 %, outline −61 %,
find_symbol −48 %; read_span ±0 because line numbers cost what JSON escaping did).

## 2. Per-session fixed cost

`initialize.instructions` lists repo *names* only (paths and sync times are one
`index_stats` away): 976 → 408 tokens with 41 repos. Tool descriptions were tightened
(1656 → 1392 tokens) while keeping every "INSTEAD OF …" cue SPEC-P8 relies on.

## 3. Query hot path (`indexio-index::ShardSet`, `indexio-query::Engine`)

Profiling on a 41-repo / 17.5k-file / 39-shard set showed the lexical plane was
dominated by bookkeeping, not by matching:

1. **`ShardSet::posting_docs` rebuilt the visible-doc set (a full scan of every shard,
   allocating two strings per doc) for every n-gram of every query.** The visible view
   is now computed once per `ShardSet` (`OnceLock<VisibleCache>`: the doc list plus its
   `(shard, docid)` set) and dropped by the mutations (`delete_docs`, `merge`).
   `visible_slice()` / `visible_set()` borrow it; the old owned accessors still exist.
2. **Candidate intersection decoded and allocated every posting's positions.**
   `PostingCursor::next_docid` skips them; `ShardSet::posting_doc_ids` uses it.
3. **The symbol boost probed every shard's symbol FST once per candidate doc.** It is
   resolved once per literal into a `HashSet<DocRef>` before verification.
4. **Regexes were compiled once per candidate doc.** `verify::find_regex_with` takes a
   pre-compiled `regex::bytes::Regex`; the engine compiles once per query.
5. **Verification (zstd decompress + scan + score) is parallel** over candidates with
   rayon once there are ≥ 16 of them (`verify_doc` is a pure per-doc function).
6. **Case-insensitive literals never used the gram index.** Smart-case makes every
   all-lowercase query (most natural-language and many identifier queries)
   case-insensitive, and those fell to the 64 MiB brute scan: ~40 ms, and on a 41-repo
   index the scan budget cut off *before the matching files*, so `fn handle_request`
   returned nothing with `truncated: true`. The planner now unions the posting lists of
   each trigram's ASCII case variants (≤ 8 per trigram), which is an exact superset.
7. **Every trigram of a literal was decoded.** Only the `MAX_CONSTRAINT_GRAMS` (6)
   rarest are, rarity read off the posting payload length without decoding; a trigram
   absent from every shard short-circuits to an empty result (2 ms for a no-hit query).
8. **The hybrid legs ran sequentially**; they are independent reads and now run under
   `rayon::join`, so wall time is the slowest leg.
9. **BM25 leg** accumulated scores in a `BTreeMap` (one insert per posting); it now uses
   a dense per-row accumulator plus a touched list.
10. **Semantic leg**: the binary prescan is split into row ranges scored in parallel
    (each keeping its own top pool, then merged: identical result); the thesaurus
    expansion caches each query term's neighbours (a full-vocabulary scan otherwise,
    ~10 ms per term); snippet extraction for the 50 rows of each leg runs in parallel
    through a bounded decompressed-content cache shared with `find_symbol`/`who_calls`.

Result on the same 41-repo index and workload (hits identical except where the brute
scan had lost them):

| call | before | after |
|---|---|---|
| regex grep (`def \w+_handler`) | 424 ms | 28 ms |
| `Command::new` | 92 ms | 3 ms |
| `fn handle_request` (all-lowercase) | 40 ms, 0 hits (truncated) | 11 ms, hits |
| `find_symbol` / `who_calls` (50 hits) | 12 ms | 0.5 ms (warm cache), ~3 ms cold |
| hybrid natural-language query | 130–190 ms | 7–14 ms |
| whole 22-call workload | 1465 ms | 111 ms |

## 4. Impact reports and snippets (second pass)

- `impact_of_symbol` / `impact_of_diff` text: direct call sites are listed per file (≤ 6
  rows each, enclosing caller shown); transitive hops are listed the same way while there
  are ≤ 40 of them and otherwise collapsed per intermediate symbol
  (`via start(): 280 sites in 190 files: top files…`). On a name-based call graph a hub
  name at hop 2 reaches most of the corpus; the agent needs its shape, not 300 rows. Hub
  symbol `spawn` at depth 2: 14.4k → 6.2k tokens.
- Lexical hit position: the line matched by the most distinct query terms (ties: rarest
  term, then earliest), instead of the earliest match of any term, which for multi-word
  queries was usually a shebang or licence header.
- Chunk (semantic/BM25) hits point at the first code-looking line at or after the chunk
  start (skipping blank, comment, attribute, decorator, shebang and bracket-only lines),
  and report that line number.
- Chunker: a split's final window was emitted even when it consisted only of the overlap
  lines, so ~8 % of all chunks (16k of 205k) were verbatim duplicates of the previous
  chunk's tail; and a tail that added only closing brackets became a `}` chunk that
  embedded as its path header. Both are folded away (`window_split`).

## 5. Working-tree freshness

Dogfooding exposed the biggest practical gap: the index reflected each repo's HEAD, so an
agent could not search the file it had just edited. `reindex_worktree` (CLI
`indexio reindex --worktree`, MCP `refresh_index` with `worktree` defaulting to true)
lists the checkout with `git ls-files --cached --others --exclude-standard` (tracked plus
untracked, gitignore respected, SKIP_DIRS applied), and runs the same content-hash delta
as plain folders. Contents are compared and stored LF-normalised, because Windows
checkouts carry CRLF while blobs are LF — without that every file re-indexed (and
re-embedded) on each refresh. The repo state records `worktree: true`; the next HEAD-based
`reindex_repo` (cron `sync`) then reconciles by content hash instead of the commit tree
diff, so the two modes can alternate and only genuinely changed files move. Measured on
this repo: worktree refresh 50–160 ms, HEAD ↔ worktree round trip touches only the edited
files.

## 6. The session's repo comes first

Every harness session runs inside one repo, but the server ranked all 41 equally, so
`find_symbol main` returned 50 definitions from other repos and `list_files **/*.py`
filled its 200 slots alphabetically. `indexio mcp` now resolves the session's repo: `--repo`,
else `$INDEXIO_REPO`, else the registered repo whose folder contains the process's working
directory (harnesses start MCP servers in the project directory). With a current repo:

- `initialize` instructions name it; `index_stats` reports it;
- `code_search` (all modes), `code_grep`, `find_symbol`, `who_calls`, `semantic_search`
  over-fetch and move the current repo's hits to the front (stable within each half)
  before truncating to the limit, so they are never cut off;
- `list_files` lists the current repo's matches first, then the rest;
- impact reports show the current repo's direct call sites in full and collapse the other
  repos to `repo: N sites in M files: top files` once there are more than 30 direct sites
  (`spawn` from repo-a: 6.2k → 4.7k tokens); `impact_of_diff` uses the diff's repo.

## 7. Shard compaction inside `refresh_index`

Every refresh that changes something writes a delta shard, and every lookup fans out over
all shards (one FST probe each); a day of worktree refreshes across sessions would leave
hundreds. `refresh_index` now merges the oldest shards down to 6 once more than 12 exist
(48 → 6 on this machine in ~20 s, transparent to the caller). `ShardSet::merge` could not
unlink a merged shard another process still has mapped (Windows), so it now parks such
files under `.stale`, which `open_dir` ignores and reaps on later opens.

## 8. Auto-refresh: the session's repo is always fresh

Even with `refresh_index` indexing the working tree, an agent had to remember to call it.
`indexio mcp` now starts one recursive filesystem watch (`notify`) on the current repo's
folder. Events for source files (known language, outside `.git`, hidden and build/dependency
directories) only set a dirty flag; the next tool call runs the stat-cached working-tree
delta before answering and reopens the engine when docs moved. Idle sessions pay nothing;
a call after an edit pays ~25 ms (the changed files are read, everything else is only
stat'ed through `WorktreeCache`); the very first refresh in a process reads every file once
(~100 ms). Editor scratch files (`.swp`, `~`) never trigger. The semantic plane is not
touched by the auto-refresh (an embed rebuilds the whole vector index, ~10 s); hybrid hits
for just-edited files update on `refresh_index`/`sync`, and the instructions say so.

Two findings on the way: a no-op refresh used to write an empty delta shard every time
(`write_shard` is now skipped when nothing changed), and compaction moved off the request
path — `maybe_compact` runs the merge on a background thread once more than 12 shards
exist; calls keep answering at their usual latency and the next call after the merge
reopens the engine (measured: 14 → 6 shards while calls stayed at 0.2 ms).

## 9. Segmented semantic plane: incremental embeds

`refresh_index` with embedding took 18 s for a one-file change (and for a delete): the
pipeline carried every other row (1.5 GB of f32), re-chunked every other repo to recover
BM25 text, and rewrote both sidecars. The semantic plane is now segmented like the shards:

- `vec/<model>.civec` (legacy base) plus `vec/<model>.d<NNNN>.civec` deltas, each a normal
  `VecIndex`; `bm25/` mirrors the list (`aligned_path`), so a global row id is
  `segment offset + local row` in both. `VecSet` / `Bm25Set` open the list, address rows
  globally, merge per-segment top-k (cosine is comparable; BM25 uses **global** N, df and
  average length summed over segments so a row scores the same wherever it lives).
- Tombstones live in `<segment>.tomb` side files (bincode `Vec<u32>`); a segment is never
  rewritten. `search_excluding` / `score_into_excluding` take the set's tombstones so
  parsed segments can be shared: `VecSet::reopen` reuses every segment whose file length
  and mtime are unchanged and only reloads side files, and a reloaded `Engine` inherits the
  old engine's sidecars (`inherit_sidecars`), so a 1.6 GB base is parsed once per process.
- `embed_repos_with` is incremental: each repo is chunked, a file whose chunk-hash multiset
  equals its live rows' is carried (`EmbedReport::carried`), changed/new files have their
  rows tombstoned and their chunks embedded (CAS first) into ONE new delta pair, vanished
  files are tombstoned. `embed_paths` restricts this to given paths (`IndexReport::
  changed_paths` from the lexical delta), which is what the auto-refresh calls — so the
  semantic plane is fresh after every edit too. `rebuild_all` (`embed --all`,
  `--rebuild-model`, which also clears segments) writes one fresh segment and removes the rest.
- `compact_vectors` merges every segment but the largest once there are more than 8, or
  all of them past 30 % tombstoned rows; vectors are carried and BM25 rows are recovered
  from the inverted lists (`rows_tf`), so no source is re-chunked. It runs on the same
  background thread as shard compaction, and the server joins that thread on shutdown.
- The random-indexing model is saved lazily by the server (`with_lazy_flush`, 5 min):
  serialising the 250 MB model per edit dominated the refresh; embeddings are always in
  the CAS, so an unsaved tail is at most a little context-vector drift.
- Fusion (`fuse_legs`) damps a file's repeat chunks within one leg (1, 1/2, 1/4, then 0):
  unbounded accumulation let a big file with a dozen mediocre semantic chunks bury a small
  file that was rank 1 in both lexical legs.

Measured on the 41-repo index: `refresh_index` with embedding after a one-file change
18 s → 55 ms; auto-refresh after an edit, lexical + semantic, 66–120 ms (first one in a
process ~200 ms); hybrid queries unchanged at 5–15 ms with 2–16 segments; 16 → 2 segments
compacted in 91 ms in the background.

## 10. Many servers, one data dir

Every agent session runs its own `indexio mcp` against the same `~/.indexio`; on this
machine six were live during the work. Two races were closed:

- delta segment names were `max + 1` over the directory listing, so two servers refreshing
  at the same moment could pick the same name and the second `tmp+rename` would replace
  the first one's rows (and misalign the vec/BM25 pair). Names are now
  `d<ms since epoch><pid><counter>` with fixed widths (lexical order stays chronological);
- shard and vector compaction take `<data_dir>/compact.lock` (exclusive create, taken
  over after 15 minutes), so two servers never merge the same inputs into duplicate rows.

Tombstones are lost-update safe: `Shard::delete_docs` unions with the bitmap currently in
the shared mapping (not the one read at open), the `.tomb` side files are unioned with
the file at write time, and both merges re-read their inputs' tombstones after writing
the output and forward anything that landed meanwhile (`tombstones_on_disk`, `origin`
maps). A vec/BM25 segment-count mismatch — another server observed between its two
writes — is an error the caller retries (the auto-refresh queues the paths in
`pending_embed`), never a reason to drop the plane.

## 11. Startup, first query, telemetry

- `prewarm_sidecars` opens the semantic segments on a background thread right after
  start; the next tool call adopts them (`inherit_sidecars`), so the first hybrid query of a
  session does not pay the ~100 ms parse.
- What remains of the first-query cost is thesaurus expansion: one full scan of the
  ~250k-term vocabulary per new query word (~20 ms each, sparse dot products). The scan now
  handles all of a query's uncached words in one pass, the cache is no longer dropped on
  every `observe` (only after 2000 observed texts, so the per-edit auto-refresh keeps it),
  and it is persisted to `sem/<model>.expand.json` (merged across servers) and loaded at
  start. Measured: a repeated natural-language query in a fresh session 57–66 ms → 16–20 ms.
- Every tool call appends `{ts, pid, repo, tool, ms, bytes, ok}` to
  `<data_dir>/usage/<date>.jsonl`; `indexio usage [--days N] [--json]` aggregates calls,
  estimated tokens returned (bytes/4), average and p95 latency per tool and per repo — the
  A/B instrument for the real sessions rather than the synthetic bench.

## 12. Definition-sized reads

`find_symbol` rows now read `start-end: signature` (the definition's range from a cached
tree-sitter outline, for the first 20 hits), and `read_span` without `end` returns the
whole definition containing `start` (capped at 400 lines) instead of a blind 60-line
window — outside any definition the 60-line default stays. One `find_symbol` + one
`read_span` now yields exactly a function, with neither a truncated tail (a second call)
nor 40 unrelated lines after a short one. Outlines are cached per engine (`outline_cached`,
512 files), so the extra cost is one parse per distinct file.

## 13. Every text file is indexed

Only the six grammar languages were indexed, so a session searching for a string in a
Markdown, JSON, YAML, TOML, shell, SQL, HTML/CSS or Ruby/PHP/Kotlin/… file got nothing
from indexio and had to grep — the single biggest reason a session could not "always"
use the index. `Lang::Text` (`indexio-types`) now classifies every such file (by
extension, plus Dockerfile/Makefile/README-style names): lexical index, window-chunked
semantic rows, no symbols/calls/outline (`Lang::is_code`). Lock files, minified bundles,
source maps and text files over 256 KB stay out. On the 41 repos this added 2,782 docs
(+44 %) and 23.8k chunks; the per-repo incremental embed took 59 s for all of them, and
the instructions/guidance now say what is covered so the model does not fall back to grep.

`indexio compact` now also folds the vector/BM25 segments (44 → 1 in 15 s after the
text rollout; the segments were 47 % tombstoned because the base predated the chunker fix).

## 14. Recall: the conversation as a queryable store

Asked whether indexio could "compact the context window" of a running session: an MCP
server cannot touch the live context, but it can make everything the context might drop
re-queryable, which lets compaction be aggressive. The code half was already true (any
file content ever read is a `read_span` away). The conversation half is this section.

- `indexio sessions` (`crates/indexio/src/sessions.rs`) renders every Claude Code
  transcript (`<claude dir>/projects/<slug>/<id>.jsonl`) into compact Markdown under
  `<data_dir>/sessions/<slug>/<id>[.partNN].md`: user and assistant text, tool calls with
  arguments (300 chars), the first 6 lines of each tool result (400 chars), and compaction
  summaries. Thinking blocks, `<system-reminder>` blocks, injected skill documents and
  bookkeeping records are dropped. Parts stay under 200 KB (the text-file cap is 256 KB).
  Incremental by (len, mtime); 939 MB of transcripts → 30 MB in 2 s. The folder is
  registered as the plain-folder source `sessions` and indexed like any repo (30.7k
  chunks embedded in 11 s).
- The MCP server re-imports its own project's transcripts on a background thread at most
  every 3 minutes (`maybe_import_sessions`), and a `PreCompact` hook in
  `~/.claude/settings.json` runs the import right before Claude Code compacts, so nothing
  that gets summarized away is lost.
- `recall {query, limit, all_projects}`: a hybrid search restricted to the `sessions`
  source, this project's sessions first; hits are `sessions:<slug>/<id>.md` + `line: text`,
  and `read_span` restores the exchange. Instructions and the harness guidance tell the
  model to recall instead of re-deriving, and to keep pointers rather than content.

Everything stays local under the data dir; it is a second copy of the transcripts Claude
Code already keeps on disk.

## 15. The Bash hook: where the tokens really went

A day of real transcripts (parsed from `~/.claude/projects`, 1,918 tool calls, ~606k
result tokens) showed the sessions had stopped using the built-in Grep/Glob/Read tools
(13 calls) — but **Bash produced 81 % of all result tokens, and two thirds of that was
`cat` / `sed -n` / `head` (243k tokens) and `grep` / `rg` (83k tokens) on files inside
indexed repos**. Guidance about tools does not reach a model that shells out (bypass mode
even tells it to read with `sed -n`).

`indexio hook bash` (`crates/indexio/src/hook.rs`) is a PreToolUse hook for the Bash
tool: it splits the command into pipelines (quotes respected, `cd` tracked, `rtk proxy`
unwrapped), and when a pipeline's first stage is a plain read (`cat`, `sed -n`, `head`,
`tail`) of a file the index has, or a filesystem search (`grep`/`rg`/`find`) inside a
registered repo, it answers with a `deny` whose reason spells out the equivalent call —
`read_span {repo, path, start, end}` for a `sed -n 'A,Bp'`, `code_search {query …
repo:… path:…}` for a grep, `list_files` for a find. Writes (`sed -i`), greps on a
pipeline, files the index does not have (logs, temp, unindexed folders) and commands
carrying `# raw` are allowed untouched; any internal error allows. 6–40 ms per call.
Registered in `~/.claude/settings.json` next to the rtk hook.

## 16. Index coverage

Every top-level git repo under the user's projects folder is registered as its own
source (41 repos). Nested A/B benchmark copies under `bench-work/`, `ab*-builds/` were
left out on purpose: 116 near-duplicate checkouts would swamp every ranked list.

## 17. Tests

`mcp_text` unit tests pin the layouts (grouping/sorting, ranked order, snippet trimming,
directory folding, tab-numbered spans, byte formatting). The MCP unit tests keep using
`OutputFormat::Json` through the `handle` helper; `e2e_add_sync` sets
`INDEXIO_MCP_FORMAT=json` for its stdio round-trip. Engine changes are covered by the
existing query/index/e2e suites; `reindex_worktree_sees_uncommitted_edits_then_head_reconciles`
covers the working-tree path (CRLF edit, untracked file, ignored file, HEAD reconcile,
commit); `watch::tests` cover the event filter and a real edit being noticed;
`text_format_renders_grouped_hits` pins the text layout and the per-call JSON override.
263 tests, all passing.

## 18. Freshness across processes

Each Claude Code session runs its own `indexio mcp`, and only the session whose cwd is a
repo keeps that repo's working tree indexed (the `notify` watcher + auto-refresh of §12).
A session that reads repo B from repo A — or whose cwd is outside the index altogether
(a second checkout of repo-b working on the first) — used to see B as of B's last
refresh by *someone*. Two changes close that gap without any per-session bookkeeping:

- **Shard-directory stamp.** `reload_if_needed` (called on every tool call) now
  fingerprints `<data_dir>/shards` — file count, total bytes, newest mtime of the
  `.cidx` files, one `read_dir` over ~10 entries — and reopens the engine when the stamp
  moves. New shards written by any process (another session's auto-refresh, the CLI, a
  `sync`, the hook below) become visible on the next call: a server started in repo-a
  saw an edit to `indexio/docs/SPEC-P9.md` indexed by `indexio reindex --worktree`
  15.6 ms after the CLI returned, without `refresh_index`; the reopen itself costs
  ~10 ms once, subsequent calls are back to 5 ms. The bench workload is unchanged
  (v15 vs v14: 152 vs 164 ms total).
- **The hook freshens before it redirects.** When `indexio hook bash` denies a `sed -n`
  / `cat` / `grep` on an indexed file, it first runs `reindex_worktree` for that repo
  (lexical only, a stat pass when nothing changed), so the `read_span` / `code_search`
  the model makes next is fresh even if no server watches that repo. Measured: append a
  line to an indexed file, hook a `sed -n` on it — 75 ms, and the new line is searchable
  before the deny reason is printed. Files the index does not have are still allowed
  through to the shell (that is the right fallback for brand-new files; the next
  auto-refresh or hook pass indexes them).

## 19. `grep -n` semantics, and the hook against real commands

Replaying the 165 shell reads of one live session (`repo-b`, six hours)
through the hook showed what a session actually greps for, and where the redirect fell
short:

- **One hit per file was not a grep.** `code_search` and `code_grep` returned the densest
  line of each matching file; a session running `grep -n "a\|b\|c" file` wants every
  matching line, and would grep again when the one snippet did not answer.
  `Engine::search_lines(q, limit, per_doc)` expands each verified document into up to
  `per_doc` matching lines (shared score, ascending, leftmost column; `per_doc == 1` is
  the old `search`). `code_grep` lists 20 lines per file by default and takes trailing
  `repo:/path:/lang:` filters inside the pattern; `code_search` takes `lines: N` (default
  1, so ranked identifier searches cost what they did). `path:a.py|b.py` matches any of
  several files, so a grep over three files stays a grep over three files.
- **Parsing bugs on real commands.** A `\"` inside double quotes ended the word (the rest
  of the command became the pattern); `-A 12` made `12` a path operand (`path:12`); BRE
  patterns (`\|` alternation, literal `(`) were passed as extended regexes, which either
  failed to parse or matched nothing. `grep_args` now knows the value-taking options,
  `bre_to_ere` converts plain-grep syntax, and `split`/`words` honour backslash escapes.
- **Better targets than a grep.** `grep "def foo" -A 14` is a definition lookup →
  `find_symbol {name}` (start-end rows, then `read_span` for the body): 242 → 47 tokens.
  `grep "^class \|^def \|^async def "` is an outline by hand → `file_outline`.
- **Reasons cost tokens too.** Every deny is paid for in the model's context; the
  reasons were cut from ~100 to 50–78 tokens (the call is most of it). A bundled command
  (`sed -n …; grep -n … | head`) gets one extra sentence saying the rest may run alone.

A/B on ten real greps of that session (tiktoken, MCP text vs `grep -n` output):

| grep | grep tok | indexio tok | note |
|---|---|---|---|
| `_error\|arguments` in one file (9 lines) | 167 | 197 | line-for-line, header + trailer |
| `warmup` in three files | 246 | 229 | `path:a\|b\|c` |
| `def _provider_preflight -A 14` | 242 | 47 | `find_symbol` |
| `^class \|^def \|^async def` (53 lines) | 670 | 893 | `file_outline` (74 definitions with end lines) |
| ten greps total | 1975 | 2137 | +8 %, 100 % line coverage where the cap allows |

So a grep-for-grep replacement is token-neutral (the `repo:path` header and `-- N matches`
trailer cost what the `cd … &&` prefix and unranked `| head` cost the other way); the
savings come from the reads a grep leads to (`read_span` of a definition instead of
`sed -n` guesses, `find_symbol` instead of `-A 14`), and from the model learning, from
the first deny, to call indexio directly. Bench workload unchanged (v16 = v15).

## 20. Hub names: qualifier-checked call sites and repo focus

`impact_of_symbol` was the most expensive call in the usage log (3.5k tokens per call).
Looking at `spawn` in repo-a showed why: the call graph is name-based, so the 167
"direct callers" were mostly `tokio::spawn(`, `std::thread::spawn(` and
`Command::new(..).spawn()` — none of them a call of the repo's eight `spawn` functions —
and the real callers were buried under them or sampled away by the fan-out cap. Three
changes, all at query time (no shard format change):

- **Call-site plausibility** (`impact.rs`: `RootShape`, `site_plausible`). The root's
  definitions give an owner set (types from a method's scope — Rust scopes are
  `Type::method`, so the owner is the segment before the name — and module names from the
  file: `net/mod.rs` → `net`, `executor.rs` → `executor`) and whether any definition takes a
  receiver (`&self`, `self`, `cls`; Go/JS/Java methods always). A depth-1 posting is kept
  when its call text is bare (`spawn(`), qualified by an owner or `self`/`super`/`crate`
  (`net::spawn(`, `Client::spawn(`), or a method call on an unknown receiver when some
  definition has one; it is dropped when the qualifier names something else
  (`tokio::spawn(`) or when it is a method call and no definition has a receiver
  (`cmd.spawn()`). Non-call mentions (`map(spawn)`) pass. Up to 5 × `max_fanout` postings
  are judged before the hub sampling; `who_calls` applies the same judgement.
- **Definitions capped.** `main` has 305 definitions across the 41 repos and the list was
  printed whole (8.6k tokens). Past 12, the session's repo is listed (up to 20) and the
  rest is one line of per-repo counts. `find_symbol` and `who_calls` do the same past 12
  rows (the session's repo up to 50, then counts) and take an optional `repo` argument.
- **Direct rows and singleton hops.** Direct callers are capped at 60 rows per section,
  the remaining files tallied; transitive symbols reached through a single site are
  grouped per file (`main.rs: run_all, route_command, …`) instead of one `via` line each.

A/B on ten hub names in repo-a (`impact_of_symbol`, tokens, old → new): spawn 4571 →
1890 (25 real direct sites instead of 167), new 13397 → 4352 (104 real sites in 30 files),
main 8632 → 1406, write 6324 → 1649, load 6831 → 2425; ten names 58.6k → 25.5k (−57 %),
latency 25–40 → 30–70 ms (content loads for the judgement). `find_symbol` on eight names
6527 → 4555, `who_calls` 6855 → 4869 (−30 %; for `who_calls` the rows are limit-capped, so
the gain is precision: `handle` 1445 → 167 with the junk gone). Bench workload 14,955 →
11,542 tokens (−23 % this iteration, −77 % against the JSON baseline).

## 21. Span numbering and the per-session fixed cost

`read_span` is the largest tool by tokens in the usage log (1.4k per call on average).
On five real spans the `N<TAB>` prefix was 14–21 % of the tokens (a four-digit number and
a tab on a ten-token line). Rows now carry the number only on the first row, the last
row and every 5th line; the other rows start with the tab alone, so columns stay aligned
and a row never starts with digits by accident. A row is at most four lines from a
numbered one. Measured: five spans 5,165 → 4,553 tokens (−12 %); bench spans −7 %. The
tool description says how to count.

The per-session fixed cost (paid in every prompt of every session) was creeping up with
each new argument: tools/list 1,691 → 1,946 tokens, instructions 565. The descriptions
were rewritten with the same cues in fewer words (1,703) and the server instructions
dropped the stale "call refresh_index after editing" paragraph, which contradicted the
auto-refresh sentence above it (440). Fixed cost 2,472 → 2,143 per session.

Bench v19: 11,336 tokens for the 22-call workload (−78 % vs the JSON baseline).

## 22. Caches that survive a reload

§18 made every server reopen its engine whenever the shard directory changes. That is
now frequent: every own edit (auto-refresh), every other session's transcript import
(≤ 3 min apart while a session is active), every hook freshen. The engine's content and
outline caches were keyed by `(shard, doc)` and died with the engine, so the first
`find_symbol` after a reload re-parsed every file it reports a range for: five typical
calls (`find_symbol main`, `find_symbol spawn`, two `read_span`s with the default
definition-sized range, `who_calls spawn`) took 206 ms right after a reload against 4 ms
warm.

Both caches are now keyed by content blob (`BlobId`, plus `Lang` for outlines). A blob
never changes meaning, so the maps can be handed to the next engine wholesale
(`inherit_caches`, called from `inherit_sidecars` on every swap); a file that did not
change keeps its parse and its decompressed bytes, a changed file misses once. The same
five calls after a reload: 14 ms, of which ~10 ms is the reopen itself
(`bench/ab_reload.py`).

## 23. One writer per repo

With §18 the same edit can be delta-indexed by several processes within the same second:
the session's own auto-refresh, the Bash hook's freshen (a different session shelling into
the repo), the CLI, two sessions open in one folder. Three concurrent
`reindex --worktree` of one edit on the previous binary: two passes each added the
changed doc (a duplicate live doc until the file changes again) and one failed with
"The system cannot find the file specified" (a state file renamed under it).

`delta_by_content` now takes `<data_dir>/repos/<name>.lock` (exclusive create, PID
inside; waits up to 5 s in 25 ms steps; a lock older than 120 s is taken over as
abandoned). The working-tree listing and hashing happen before the lock, the shard set is
opened after it, so a pass that waited compares against the winner's result and reports
0 added. Four concurrent passes on the new binary, three rounds: one pass adds the doc,
the other three report 0 added, no errors, one live doc (`bench/ab_concurrent.py`).

## 24. A real diff: receivers and diff-specific roots

The repo-b session edited `search()` on three provider classes (HackerNews, Crossref,
StackExchange) plus two private helpers. `impact_of_diff` on that working tree returned
4.1k tokens, 135 "direct callers", and most of them were `re.search(`, `_TITLE_RE.search(`,
`OpenAlex().search(`, `Engine.search(eng, ..)`: the §20 judgement let every method call on
an unknown receiver through, and the root's owner set came from every `search` definition
in the index (so `DuckDuckGo` was an owner too). Two refinements:

- **Receivers that cannot be an instance.** For `x.name(`, `x` is rejected when it is a
  library module (`re`, `os`, `asyncio`, `tokio`, `JSON`, … a short list), a
  SCREAMING_CASE constant (`_TITLE_RE`), or a type / constructor that is not one of the
  owners (`OpenAlex()`, `Engine`). `trailing_ident` looks through trailing `(...)` and
  `[...]` groups, so `OpenAlex().search` and `_RSS_FIELD["title"].search` resolve to
  `OpenAlex` and `_RSS_FIELD`. Lower-case variables (`eng.search`, `p.search`) stay in:
  `p.search(q, ctx)` in the router is the real caller.
- **Diff-specific roots.** `impact_of_diff` builds the root shapes from the changed
  definitions only (`ChangedSymbol` now carries its signature line): `search` means those
  three classes, and `DuckDuckGo().search` is out. `impact_of_symbol` keeps the
  every-definition view, since there the name itself is the question.

On that diff (`bench/fixtures_repo-b_providers.diff`): 4,263 → 3,379 tokens (−21 %), the
regex and other-provider sites gone, the same real callers listed. Bench workload
unchanged (v20 = v19).

## 25. The hook never ran: a backslash path under Git Bash

The first real-usage check after the sessions restarted (`tools/usage_from_transcripts.py
--since 2026-09-15T20:20`) showed the repo-b session doing 131 shell reads (45k tokens)
in twenty minutes with zero denials, while `indexio hook bash` replayed on the same
commands denied them all. The transcript had the answer as `hook_non_blocking_error`
attachments on every Bash call:

    /usr/bin/bash: C:Usersme.cargobinindexio.exe: command not found

Claude Code spawns hook commands through Git Bash on Windows too, and the
`C:\Users\…\indexio.exe` path written into `~/.claude/settings.json` lost its backslashes
as escapes. The hook had been a no-op since it was installed (§15); so had the PreCompact
`sessions --quiet` import (§14) — only the MCP server's own 3-minute import kept `recall`
fresh. A non-blocking hook error is shown to the user, not the model, so nothing in the
sessions noticed.

Fix: forward slashes (`C:/Users/…/indexio.exe hook bash`), which every shell on Windows
accepts. `indexio hook install` now writes both hooks that way from `current_exe()`
(quoted when the path has spaces), repairs an existing indexio entry in place, leaves the
user's other hooks alone and is idempotent; `--print-only` shows the changes. Claude Code
re-reads `settings.json` hooks live: the first `sed -n` in this session after the edit
was denied without a restart, so the running sessions get the hook on their next call.

Also found while reading the repo-b transcript: `code_grep` compiled its regex without
multi-line mode, so `^## 24` or `\{$` only matched at the very start or end of a file
(`grep -n` semantics anchor every line). `verify::compile_regex` now builds every content
regex with `multi_line(true)`; the trigram prefilter already treated anchors as
transparent, so candidates were right and only the verification changed.
