# SPEC-P10: storage and I/O — what opening the index costs, and what it should

Goal: the index must open fast on a slow disk, not thrash IOPS, and scale to
thousands of repos. Measured on the live data dir on 2026-09-16 (43 repos, 9,981
files, **143 MB of raw text**): the data dir was **5.3 GB** — 37× the text it
describes — and a server start read 302 MB through syscalls.

## 1. What happens when the files are opened

`bench/mcp_io.py` drives `indexio mcp` and samples the server process's
`io_counters` after startup and after each representative call (page cache warm):

| step | ms | read syscalls | MB read | RSS MB |
|---|---|---|---|---|
| startup | 188 | 146 | 302 | 417 |
| find_symbol | 149 | 1737 | 41.6 | 436 |
| first hybrid query | 133 | 165 | 48.9 | 619 |
| every other call | <1 | 1 | 0 | — |

- **Shards (`.cidx`) and vector segments (`.civec`) are memory-mapped.** Opening
  one reads the 64-byte header, the section table, the META JSON and the 64 KB
  tombstone bitmap; everything else is a page fault on demand. That part is right.
- **The rindex model is not.** `sem/rindex-v2.rimodel` (261 MB of bincode) is
  `fs::read` + deserialised into HashMaps at every server start — that is the 302 MB
  and most of the 417 MB RSS, paid **per session** (six sessions ≈ 1.8 GB of startup
  reads and 3.6 GB of RAM for the same bytes).
- **The first hybrid query reads the 48 MB of binary codes** (189k rows × 256 B)
  with plain reads; rescoring then faults in ~256 random 8 KB f32 rows (≈2 MB of
  random I/O per query — fine on an SSD, seconds on a spinning disk).
- The 41.6 MB at `find_symbol` was the background transcript import running at
  that moment (§6).

## 2. Where the 5.3 GB went

| component | size | content | what it needs to be |
|---|---|---|---|
| `embcas/` | 2.0 GB | f32 vector per chunk ever embedded (8 KB each) | a hash set (§5) |
| `vec/` | 1.8 GB | f32 rows 8 KB/chunk + 256 B binary codes | int8 rows (§7) |
| `cas/` | 962 MB | bincode `ExtractedArtifact` per blob, uncompressed, with every gram's byte positions | compressed gram list (§4) |
| `shards/` | 317 MB | NGRAM_POST 270 MB (positions), CONTENT 34 MB (zstd+dict, 4.3×) | docid-only postings (§3) |
| `sem/` | 261 MB | rindex vocabulary: 250k terms × 128 context components | same bytes, mmap-able (§7) |
| `bm25/` | 39 MB | BM25F inverted lists per chunk | fine |
| `sessions/` | 30 MB | rendered transcripts | fine |

## 3. Positions were never read

The gram postings stored every byte offset of every gram occurrence
(`varint(pos_count)` + deltas per doc): 270 MB for 143 MB of text. Every caller was
checked — the query planner intersects `posting_doc_ids` (docids only) and the
verifier rescans the decompressed content for line/column; `decode_postings` with
positions was used by tests and by the shard merge, which only re-encoded them.

Change: `grams::extract` returns the sorted distinct grams (`Vec<Vec<u8>>`, no
`HashMap<gram, Vec<u32>>` during extraction), `ExtractedArtifact.ngrams` is that
list, the writer emits `encode_doc_postings` (the same block format with
`pos_count = 0`, so every reader is unchanged and old shards still decode), and the
merge inverts through `next_docid` — a compaction also strips positions from
pre-P10 shards.

`bench/ab_storage.py` (same four repos, 32.5 MB raw, `add --no-embed`, old vs new
binary into fresh data dirs):

| | build s | shards MB | NGRAM_POST MB | cas MB | cas files |
|---|---|---|---|---|---|
| before | 9.1 | 84.0 | 68.3 | 225.6 | 2350 |
| after | 8.4 | **29.4** | **13.7** | **14.6** | 2350 |

Shards 2.9× smaller (NGRAM_POST 5×), extraction slightly faster.

## 4. Extraction cache v2

`cas/` held the same positions again, as raw bincode (every gram a `Vec<u8>` with an
8-byte length prefix, every position list another): ~100 KB per blob, 7× the
compressed content it described, and it would have been ~70 GB at 8,000 repos.
Entries are now `cas/v2/<hh>/<rest>`: a `CAS2` magic plus the zstd (level 3) frame of
the bincode artifact — 15× smaller in the A/B above. The old fan-out cannot be parsed
by the position-free type, so `Cas::open` sweeps the legacy `<hh>/` directories once
(it is a cache; a swept blob costs one re-extraction if it is ever seen again — and
blobs already in a shard are never looked up).

## 5. Seen-only embedding cache

`embcas/` cached every chunk's f32 vector so a chunk seen again (another repo, a
rebuild) would not be re-embedded. For the in-process rindex model that is a bad
trade: embedding a chunk costs ~0.4 ms, reading its 8 KB back costs more on any disk,
and the pack was 40 % of the data dir. What the pipeline actually needs from the
cache is the set of chunks the model has already **observed**, so the unchanged
chunks of an edited file are not fed to the model twice (drift). `Embedder::
recompute_is_cheap()` (true for rindex and the hash embedder, false for HTTP
embedders that cost money) selects a seen-only cache: `embcas/<model>/seen.idx`, 16
bytes per chunk, `contains` instead of `get`; misses the cache knows are re-embedded
but not re-observed (`EmbedReport.cas_known`). An existing pack is migrated on first
open (hashes copied to `seen.idx`, `pack.bin`/`pack.idx` removed): 2.09 GB → 4 MB.

## 6. Background transcript import

The 41.6 MB read at `find_symbol` was `maybe_import_sessions`: every 3 minutes the
server re-renders the transcripts that changed (the active session's own, ~12 MB) and
then delta-indexes the `sessions` folder — a plain-folder source, and `reindex_dir`
read and hashed **every** part in it (30 MB) to find the two that changed. Plain-folder
repos had the same cost on every auto-refresh, because `reindex_worktree_cached`
ignored its stat cache for them. `read_plain_tree_cached` now applies the same
`(mtime, len) → blob` cache as the git working-tree reader; the server keeps one for
the sessions folder, so an import stats 300 files and reads the rewritten parts only.

## Live migration (2026-09-16, the 43-repo data dir)

`embcas-stats` migrated the pack (2.09 GB → 3.9 MB, 255k hashes, 0.3 s); the first
`Cas::open` swept the legacy extraction cache (962 MB → 266 KB); `indexio compact
--max-shards 1` re-encoded every posting without positions (14 shards, 317 MB → one
shard, 83 MB, 12.8 s; the vector segments folded 10 → 2 at the same time). Data dir
5.3 GB → 2.1 GB, of which vectors 1.8 GB and the model 261 MB — §7's targets. The
bench workload is 170 → 145 ms on the compacted index (v22), tokens unchanged.

## 7. int8 vector rows

Segments are now v3 (`CIVEC003`): per-row scale (`max|x| / 127`) + `dim` int8, the
binary codes and tombstones as before; the rescore is `scale · dot(q, row_i8)`, and a
compaction round-trips the rows exactly (the maximum maps back to 127, every other
component to the same integer). `create_with_options` (the HNSW full build, unused by
the pipeline) still writes v2; both stay readable. `bench/ab_quant.py` folds two copies
of the live plane into ONE segment each — f32 (`INDEXIO_VEC_F32=1`) and int8 — so the
row format is the only difference, then runs 30 natural-language queries against both:

| | vec/ | semantic top-1 | semantic top-10 overlap | hybrid top-1 | hybrid top-10 overlap |
|---|---|---|---|---|---|
| f32 → int8 | 1835 → **517 MB** (3.55×) | 30/30 | 98.0 % | 29/30 | 100 % |

(The first, naive comparison against the live 4-segment f32 plane showed 84 % overlap:
that was the merge itself — one segment rescoring one 256-candidate pool instead of four
— not the quantisation.) Live: `compact --vectors` re-encoded 218k rows in 11.5 s;
`vec/` 1.8 GB → 493 MB; the data dir is now **911 MB** (from 5.3 GB); the bench
workload's hybrid results and latency are unchanged (v23).

Sessions whose server predates this build cannot open v3 segments (`bad .civec magic`):
their hybrid/semantic searches error until the session restarts; lexical, symbols,
outlines and spans are unaffected.

## 8. The model and the BM25 sidecar are mapped too

`sem/rindex-v2.rimodel` is now a flat file (`RIMDL003`, `rimodel.rs`): a bucket table
(FNV-1a, linear probing), 16-byte entry records, the term bytes and the context
components, 8-byte aligned. The embedder maps it; a term lookup touches one bucket, one
record, the term and its components. Updates from `observe` go to an in-memory overlay
(the sharded HashMap the model always had, now holding only what changed since the
snapshot); `save` merges overlay and snapshot into a new file (tmp + rename, the mapped
predecessor parked as `.stale*` and reaped later, as the vector segments do) and
continues on top of it. A pre-P10 bincode model is converted on the first open. The
thesaurus scan (`nearest_many`) walks the snapshot in parallel ranges and skips the two
per-entry overlay probes while the overlay is empty (a read-only server).

The BM25 sidecar (`.cibm25`) had its 39 MB read and copied at every start as well; its
term dictionary and postings block are now ranges of a mapping (`bm25::Bytes`).

| server start (`bench/mcp_io.py`) | ms | MB read | RSS MB |
|---|---|---|---|
| this morning (bincode model, copied sidecar) | 188 | 302 | 417 |
| mapped model (4fed13c) | 14 | 40 | 121 |
| + mapped sidecar (e3e049d) | **10** | **0** | **74** |

Hybrid queries are unchanged in results and warm latency (7–11 ms); the first query
that needs an uncached thesaurus term pays the snapshot page-in once per process
(~30 ms, was the 250 MB read at startup instead). The three long-running servers on the
old binary were holding 720–985 MB each; the pages of the mapped files are shared
across sessions.

## 9. Real usage after the hook, and recall

With the hook live, the other sessions' result tokens since 03:55Z were 53 % `read_span`
(85 calls, 735–930 tokens each, 1 % of lines re-read), 6 % `code_grep`, the rest
small. The spans are what the model asked for and the rows are already bare (numbers on
every 5th line only); indentation is 18 % of the characters but cannot be touched — the
model pastes span text into `Edit` old_strings. The tool outputs are at their floor; the
remaining waste is the sessions still on the pre-text-format server (their `file_outline`
comes back as 681 tokens of JSON where the text form is ~200).

`recall` was a corpus-wide hybrid search post-filtered to the transcripts: 60 fused
candidates of mostly code, few of them transcripts, 62 ms. Every leg now takes a repo
scope (`VecSet::search_where` keeps the prescan pool to matching rows, `Bm25Set::
search_where` filters the collected rows, the lexical leg gets `repo:`), and an explicit
`repo:X` in a hybrid/semantic `code_search` scopes its legs the same way. On 8 questions
about this day's work: 1/8 → 3/8 answered in the top 3, 62 → 12 ms. The misses are facts
that were only in the assistant's visible text — which this harness does not always
persist to the transcript (18 text blocks against 358 tool calls in this session, while
a short narration of each turn lands in its `thinking` block). The renderer now keeps
thinking blocks, capped at 400 characters (`~ …` lines); an earlier recall of the same
question is dropped from its own results.

## 10. Tried and dropped: an IVF prescan

For the 8,000-repo case the flat binary prescan is the first thing that breaks
(O(rows): 8 GB of codes per query at 32M rows). An IVF sidecar was built and measured
— k-means++ on a 20k-row sample, k ≈ √rows partitions, every row assigned by dot
product, the query probing the nearest 5 % of partitions (`.civf` next to the segment,
built at compaction). On the live plane (218k rows, k = 467, 24 partitions probed) the
semantic top-10 overlap with the flat scan was **57.7 %** (top-1 16/30), hybrid 78 %,
and the semantic leg was no faster (32 ms either way — at this size the scan is not
where the time goes). rindex vectors are sums of sparse ternary labels in 2048
dimensions: near-isotropic, with no cluster structure a coarse quantiser can exploit,
so reaching the flat scan's recall would mean probing a large fraction of the
partitions and giving the speed-up back. Removed.

The scale path that fits this data is structural instead: group a compacted
segment's rows by repo and keep a repo → row-range table, so a repo-scoped search (the
default preference is already "the session's repo first") scans only that repo's codes;
cross-repo queries scan everything and stay the rare, slower case. The scoped legs
from §9 are the query-side half of that; the row grouping is the next step.

## 11. Rows grouped by repo; scoped searches walk runs

A compaction now sorts the merged rows by (repo, path, line) — the BM25 rows and the
tombstone-forwarding origins follow the same permutation — and every segment computes,
once, its runs of consecutive same-repo rows from the metas (no format change; a delta
segment simply has one run per file group). A repo-scoped semantic search
(`VecIndex::search_in_repo` / `VecSet::search_repo`, used by the §9 scoped legs) walks
those runs in parallel instead of testing a predicate on every row: identical results
to the predicate scan (12/12 queries on the live plane, a unit test over interleaved
repos), 12.9 → 11.2 ms for repo-a and 6.7 → 5.1 ms for repo-b today, and at 8,000
repos the cost of a scoped query is that repo's rows rather than the corpus's. The
unscoped default still scans everything; making "the session's repo first" a scoped
pass with a global fallback is the remaining switch for that scale.

## 12. A server that outlives a deploy says so

Every improvement in this document reached the sessions only after a restart, and the
sessions kept running servers from before the text format, the hook, the int8 segments
and the mapped model for a day or more — nobody in the loop could see it. The server now
records its binary's `(len, mtime)` at start and, at most once a minute, stats the same
path; when a deploy has replaced the file (renamed aside, new one copied in — the
running process keeps its original path), the next tool result carries one trailer:
`[indexio: a newer indexio binary was installed since this session's server started;
restart the session to use it]`. Once per server, so it cannot become noise; the model
relays it. Verified with a rename + copy against a running server.

## 13. `# raw` is a retry, not a habit

By the second night the repo-b session was writing `# raw` on every grep it issued —
unprompted, learned from the deny reasons it had seen the day before. The marker now
escapes the hook only when the command (marker stripped) is the one the hook last
denied in that session (`<data_dir>/hook/<session>.last`): a fresh `# raw` command is
judged like any other and gets the redirect; repeating it with the marker is allowed.
`INDEXIO_RAW` in the command stays an unconditional escape for scripts. The marker is
also stripped before judging — a trailing `# raw` used to read as extra grep operands
outside every repo, which allowed the command before the explicit check even ran.

## 14. Known answers for the user's own repos, written by a model

Every ranking change so far was judged by proxies (top-k overlap between two builds) or
by the bundled ripgrep/serde answer set, which is not in this index. `bench/
gen_questions.py` samples 30–70-line windows from the registered repos and asks an
OpenAI-compatible chat model (Muse Spark 1.3 contributor by default; endpoint, model and
reasoning effort by environment, the key only ever in `MODEL_API_KEY`) for the question a
developer who has not seen the code would type to find it — no identifiers copied. The
rows (`bench/known_answers_local.json`, `{q, repo, path_suffix, line, why}`) feed
`bench/eval_modes.py`, which now ignores transcript hits the way the MCP server does, and
report MRR and recall@5 per search mode on the live index. Muse reasons by default and
shares `max_tokens` with the reasoning, so the generator asks for `minimal` effort and a
1500-token budget; ~4 s and a fraction of a cent per question.

## 15. What the questions showed

80 questions over 30 of the registered repos, scored on the live index
(`eval_modes.py`, recall@5 = the answering file within the top 5 files):

| mode | repo unknown | repo known (`repo:` scope) |
|---|---|---|
| lexical (keywordised) | 7.5 % | 1.3 % |
| semantic | 12.5 % | 41.2 % |
| chunk BM25 alone | — | **66.2 %** (MRR 0.526) |
| hybrid, equal weights | 52.5 % | 61.3 % (MRR 0.450) |
| hybrid, CombMNZ | — | 57.5 % |

Two things. The repo scope matters more than any ranking change: a third of the
unscoped misses were the same UI component in another Next.js site outranking the
asked-for one — the MCP server's "session repo first" and the §9 scoping are what an agent
actually gets. And on natural-language questions the chunk-BM25 leg alone beats the
three-leg fusion: equal-weight RRF lets the weaker semantic and lexical legs displace
BM25's hits. Fusion weights per leg are now a constant (`FUSE_WEIGHTS`,
`INDEXIO_FUSE_WEIGHTS=lex,bm25,sem` to experiment); the sweep:

| weights (lex, bm25, sem) | repo known: MRR / recall@5 | repo unknown |
|---|---|---|
| 1, 1, 1 (before) | 0.450 / 61.3 % | 0.314 / 52.5 % |
| 1, 1, 0.5 | 0.485 / 65.0 % | 0.365 / 57.5 % |
| 1, 1, 0.3 | 0.513 / 67.5 % | — |
| **0.5, 1, 0.3** | **0.513 / 67.5 %** | **0.418 / 57.5 %** |
| 0.3, 1, 0.3 | 0.513 / 67.5 % | — |
| 0, 1, 0.5 | 0.485 / 65.0 % | 0.369 / 57.5 % |

`(0.5, 1, 0.3)` is the default now (confirmed on the deployed binary). The semantic leg
still earns its place — BM25 alone is 66.2 %, the weighted fusion 67.5 % with a better
MRR — but as a tie-breaker, not a peer. The `--rerank` option (the bundled overlap
reranker) was measured too: 35.0 % recall@5 on top of the same fusion, i.e. it halves
the quality; it stays off by default and should be replaced or removed. The bench
workload is unchanged in latency (175 ms) and +65 tokens on its four hybrid calls,
which now return different (better-ranked) rows.

## 16. The morning's sessions: a `Read` hook, and what a restart is worth

The first working hour of the 16th (`tools/usage_from_transcripts.py --since 12:00Z`):
repo-c 131 indexio calls / 93k tokens, repo-d 92 / 55k, repo-a 68 /
32k — and repo-a also pulled **24k tokens through 13 `Read` tool calls** on source
files (`commands.rs`, `main.rs`, `app.rs`, `types.rs`, …), whole files at Claude Code's
2,000-line default. Nothing intercepted that: the Bash hook covers shell reads, the
guidance covers the model's choices, the built-in `Read` tool sat between them.

`indexio hook read` is the third hook (`hook install` adds it, matcher `Read`): a file the
index holds is answered with `file_outline` + `read_span` (a Read with `offset`/`limit`
maps to the exact `read_span` range; `offset` alone to the definition at that line);
images, logs, task outputs and unindexed folders pass. Installed at 08:2x local; live for
every session on its next call.

`tools/replay_calls.py` replays a session's real tool calls against the installed server
and compares result sizes — the restart question, answered per session instead of by
guess. For the two sessions still on the 15th's morning server: repo-c −4 % (130
calls: `file_outline` −57 %, `list_files` −24 %, `read_span` unchanged),
repo-d −8 % (`code_grep` −34 %, `file_outline` −64 %, `list_files` −77 %). Small,
because `read_span` is 80 % of both — the reads themselves are the cost, which is what
the Read hook and the per-file caps address; the restart's real gains there are memory
(repo-d's server sits at 1.2 GB) and working hybrid search.

`# raw` after §13: 30 uses this hour across four sessions, none of them retries of a
denied command — every one was judged normally and redirected.

## 17. The offline reranker is no longer the default

`rerank: true` (and the CLI `--rerank`) fell back to the bundled `OverlapReranker` when
no `INDEXIO_RERANK_BASE` endpoint was configured — the heuristic that halved recall@5 in
§15. The fallback is now the no-op: without a configured endpoint the flag leaves the
fused order alone, and the tool schema says so (+11 tokens on `tools/list`). The overlap
reranker stays in the crate for its unit tests and for anyone who wires it up on purpose.

## 18. The semantic leg: stale rows, and an expansion that hurt

Re-running the §15 evaluation two hours later, the semantic leg had fallen from 41 %
to 22 % recall@5 with nothing in the search code changed. The rindex model learns from
every chunk it embeds, and the transcript source (`sessions`) re-renders every few
minutes — 12,700 chunks of tool output, JSON and paths observed in three hours — so
the vocabulary's context vectors kept moving while the rows embedded earlier stayed
where they were: queries embedded with today's model no longer lined up with rows
embedded with this morning's. Re-embedding every row with the current model
(`indexio embed --all`, 30 s for 218k rows) brought the leg back to 35 %.

Two changes. Transcript chunks are still embedded and searchable but no longer feed
the model (`pipeline::NO_OBSERVE_REPO`): only code trains the vocabulary, so the
drift source with the highest volume and the least code in it is gone, and the
per-import 249 MB model saves with it. And the thesaurus expansion (SPEC-P5 B2) is now
opt-in (`INDEXIO_EXPAND=1`): measured after the re-embed, with the repo known —

| | semantic MRR / recall@5 | hybrid MRR / recall@5 |
|---|---|---|
| expansion on | 0.268 / 35.0 % | 0.486 / 67.5 % |
| **expansion off** | **0.349 / 43.8 %** | **0.526 / 71.3 %** |

— it lowered both legs, and its first-time vocabulary scan was the slowest part of a
cold hybrid query (§8). Also fixed on the way: with the seen-only embedding cache (§5) a
chunk text occurring twice in one embed pass got no vector for its second occurrence
and the whole pass failed (`no embedding for chunk`) — which is why the first re-embed
attempt did nothing; duplicates now take their vector from the pass itself (test added).

Remaining drift: code changes still train the model while unchanged rows keep their
old vectors. A re-embed when the model has grown a lot since the plane was built is the
follow-up; for now `indexio embed --all` is the reset.

## 19. Two servers, one model file

Every session's server observes the code it sees edited and saves the model on its lazy
flush; each merged its own overlay onto the snapshot it had opened, so with two sessions
the last save silently dropped the other's updates (the follow-up left at §9). A save
now re-opens the snapshot on disk first and, when another writer has been there since
(a larger `n_texts_seen`), rebases onto it: the other server's terms survive, this
server's overlay is applied on top (a term both touched keeps this server's state),
and the text count adds this server's share. It also covers the first snapshot of a
fresh data dir being written by a sibling. Test: two embedders on one data dir observe
different texts and save in turn; the reopened model holds both.

## 20. The plane re-embeds itself when the model has moved on

§18 fixed the largest drift source; code changes still train the model while unchanged
rows keep the vectors they were embedded with. `rebuild_all` now writes
`<vec>/<model>.built` — the model's text count when every row was last embedded — and
the server's background compaction has a third trigger: when the model has learned from
1.25× the texts it had at the last full rebuild, and that rebuild is more than six hours
old, the plane is re-embedded whole under the compaction lock (30 s for 218k rows; the
next call reopens it). Stateless embedders report no text count and never trigger it; a
plane built before this has no stamp and is left alone until the next `embed --all`.

## 21. Other repos stay current on their own

A session's own repo is refreshed from its working tree before every call; the other
42 were only as fresh as the last `indexio sync` or `refresh_index`, and nothing ran
either — a session reading `repo-b` from the `repo-b-engine` session saw
whatever state the last manual sync had left. Every 10 minutes a server now probes the
registered git repos it is not working in (`ingest::head_moved`: one ref resolve each,
no tree walk; plain folders and repos another server indexes from a working tree are
skipped), delta-indexes and embeds the ones whose HEAD moved, and reopens. One server at a
time across the data dir (`sync.lock`); a pass with nothing changed costs ~50 ms for 43
repos. Transcript rendering also drops the shell/git noise lines (`Shell cwd was
reset`, CRLF warnings, log pointers) that used to fill a result's six kept lines.

## 22. The server measures itself against the built-in tools

`bench/ab_readme.py` compares the two paths on synthetic tasks; the same comparison now
runs on real traffic. Every successful tool call is logged (`usage/<day>.jsonl`) with the
bytes it returned and, where the harness tool's output for that exact call can be sized
without extra work, the bytes that output would have had (`"builtin"`):

| call | built-in equivalent | how it is sized |
|---|---|---|
| `read_span` | `Read` of the whole file (`N<TAB>line`, first 2,000 lines) | from the file's content, charged only the first time a session spans that file — after that the harness would have had it in context |
| `file_outline` | none | baseline 0: the cost of reading by span instead of whole |
| `find_symbol`, `who_calls` | `Grep` for `fn name` / `name(` in the session's repo | `path:line:text` of the definition / call rows in that repo, first 250 (the harness's default cap) |
| `code_grep`, lexical `code_search` | `Grep` of the pattern | the same, from the rows the index returned (never more than it found) |
| `list_files` | `Glob` | one absolute path per line for the session's repo |
| hybrid/semantic search, `recall`, `impact_*`, `index_stats` | none | not compared (counted as unchanged) |

`indexio usage` and the end of `index_stats` aggregate the last N days across every server
process on the data dir: calls, tokens returned, and for the compared calls the tokens the
built-in tools would have returned and the change. Old log rows without the field are
uncompared. The baselines err low on purpose: a grep's real output is uncapped per file and
includes lines the index would not rank; a whole-file `Read` is charged once even when the
harness would have re-read the file after edits.

## 23. A server adopts its repo late

`resolve_current_repo` ran once, at start. The `repo-b-engine` session had started at
16:59 in a folder that was registered at 17:20: for the next four hours its server ran
with no current repo — no working-tree auto-refresh, no repo-first ranking, `repo: null`
in every usage row — and nothing but a session restart would have fixed it (the same
happens to any session started before `indexio add` of its folder, and to a session
whose folder is reached through a junction registered under the other path only when
canonicalisation fails). While its repo is `None`, a server now re-resolves once a minute
from `repos/` (one directory listing), and on a match starts the watch and takes the repo
for the rest of the session. Test: a server started with nothing registered, a repo
registered under its working directory afterwards, adopted on the next look; an unrelated
folder is not.

## 24. Build output under a name the skip list does not know

`SKIP_DIRS` drops `target/`, `node_modules/`, `dist/` … by name. repo-a's checkout had a
cargo target directory called `target-bench-B/`, untracked and not ignored: the
working-tree listing (`git ls-files --others`) took it, and its 1,034 dep-info `.d` files
were indexed as D source — 55 % of the repo's docs, 10 % of the whole index, matching
every crate name a lexical query mentions (a real `code_grep` in that session returned
`target-bench-B/…` rows), and every build re-triggered the auto-refresh. Cargo stamps every
target directory with `CACHEDIR.TAG` (so do pip, pytest, Gradle …): the plain-tree walker
now stops at a directory carrying it, the working-tree listing skips files under one (one
stat per directory per pass, memoised), and the watcher ignores events under one. Test: a
tagged `target-bench-B/debug/deps/x.d` next to a `bench/keep.py` — the latter is indexed,
the former is not, by both listings.

## 25. Three small things from a day of real traffic

- **A multi-word lexical query that matches nothing falls back to hybrid.** Six of 216
  real search calls in one afternoon returned `no hits`; every one was a lexical query of
  three to six words (`storage_state_set session_save session_load async def`) — the model
  using lexical mode as a bag of words, where the parser wants every literal on one line.
  Each empty result was followed by a reworded retry. Now, when a lexical query with two
  or more literals and no regex matches no line, the server answers with the hybrid
  ranking for the same words under a one-line note (`no line has every word; ranked by
  any of them (hybrid):`); a single word or a regex still says `no hits`.
- **`index_stats` is a fifth of the size.** It is called about once per server start (43
  processes in a day). One row per repo with folder, commit and date was 2.9 KB for 45
  repos; now repos are grouped by sync day under the folder most of them share, only the
  ones elsewhere spell out a path, plain folders carry a `*`, and the commit hashes stay
  in the JSON payload. ~700 tokens → ~150 per session start.
- **Benchmarks stay out of the usage stats.** `INDEXIO_NO_USAGE=1` (set by the bench
  harnesses) skips the usage log, so `index_stats` and `indexio usage` report only what
  sessions did — a sweep listing 10k files had pulled the measured reduction from −74 % to
  −52 % in an hour.

## 26. The Bash hook leaves mixed calls alone

Of 58 Bash denials in one day, 22 were calls with a pipeline the index cannot serve next
to one it can: `python patch.py && grep -n … crates/x.rs`, a python heredoc whose body
mentioned `grep`, `python -c … ; grep -n numpy requirements.txt`, `cat a.sh; bash a.sh`.
The deny sent the model back to run the work part alone — an extra turn, and for a patch
script a second application. A call whose pipelines include anything outside the lookup
set (`cat head tail sed grep rg find ls wc sort uniq cut tr awk echo printf cd …`, `rtk`
unwrapped), a `sed -i`, a redirect or a heredoc is now allowed whole; only calls that are
lookups end to end are redirected to the index. Test: the classifier on the shapes above.

(Also this evening: the known-answers set on the cleaned index gives hybrid MRR 0.507 /
recall@5 67.5 % — within three questions of the 0.526 / 71.3 % measured before the
repo-a junk was removed and the model kept learning; not a regression to chase.)

## 27. A grep pattern handed to lexical search runs as a regex

An agent-swarm worker on the second build asked `code_search` (lexical by default) for
`onHover|onMouseEnter|:hover|hover:`, `minHeight|minTouch|44` and
`width: 3[5-9][0-9]|minWidth: 3[5-9][0-9]` — grep alternations and classes written the way
`code_grep` takes them, without the `/…/` the lexical parser needs. As literal text every
one matched nothing; each `no hits` was followed by the worker's own `search_grep`. When a
lexical query with no regex reads like one (an unescaped `|`, a `[`, `.*`, a backslash, an
anchor or a group quantifier, filters aside) and the literal reading finds nothing, the
server now runs it through the `code_grep` path and answers under
`no literal match; treated as a regex:`. Plain words are untouched, and §25's hybrid
fallback still covers the bag-of-words case.

(Also from that run: the swarm's global MCP config had carried `--repo <folder>` for the
previous build, so the second build — in another folder — ran its server against the wrong
repo: no working-tree refresh of the folder the workers were editing, searches for a
function written minutes earlier found nothing. The flag is gone; the server resolves the
repo from the working directory the swarm pins, which both folders now register.)

## 28. `add` and `sync` embed what changed

`indexio add <folder>` ended with `embed_all`: every repo's chunks re-embedded into one
fresh 500 MB segment, 25 s per registration, and every running server reopening the plane
— including the pre-v3 servers that could not read the new format at all, which is how
registering one folder on the 17th took vectors away from every old session at once. The
embed after a sync is now the same incremental path the server's auto-refresh and
background sync use: only the repos whose docs were added or deleted get a delta segment;
a data dir with no plane for the model yet is built whole. Registering a 233-file folder
now costs its own chunks, not the fleet's.

## 29. Compaction no longer depends on who wrote the shards

A server folded shards only after its own auto-refresh had changed something. Every
other writer — the Bash hook's freshen, `indexio sync`, the background sync, another
session's server — adds a delta shard and compacts nothing, and a server whose own repo
sits idle never looks. The data dir reached 30 shards (the threshold is 12); every lookup
probed each one, and a manual `indexio compact` folded 25 of them in 22 s. Every server now
counts the shard files once a minute (a directory listing, not an open of each shard) and
runs the same background compaction under the cross-process lock when the count is over the
threshold. Test: fourteen one-doc shards written by "someone else", one ordinary tool call,
the count is back under the target and the merged index still answers.

## 30. The model save leaves the call path

The server persists its Random Indexing model lazily: `flush` after an incremental embed
saved the ~250 MB snapshot when the model was dirty and the last save was older than five
minutes. That save — overlay merged into the mapped snapshot, a new file written, the write
lock held throughout — landed inside whichever tool call happened to embed first after the
interval: a plain-folder session logged a 2.7 s and a 5.7 s `read_span` in one afternoon,
against a median of 0 ms. `flush` in lazy mode now only notes the state; the server checks
`save_due` after each call and runs `persist` on its own thread, joined before the next save
and at shutdown. Lexical tools never wait; a hybrid query that arrives during the save
waits for the write lock, as before. Test: a lazy embedder is dirty after `flush`, due once
the interval has passed, clean after `persist`, and reopens with the observed term.

## 31. Command output, retention, and what else the index holds

Measured on a day of the author's coding sessions after the hooks took over reads and
greps: 46 % of the tokens returned to the model came from the index, 32 % from Bash
results, 10 % from `Read`. Of the Bash share, script runs were 51k tokens, and `sed`, `cat`
and `grep` on files the index did not hold were 82k — firmware (`.ino`), template and data
files with no language mapping, so the hooks let them through. Three changes:

- **More file types.** Firmware, shader, assembly, Fortran, template and tabular
  extensions map to `Lang::Text` and are indexed like docs; the Read and Bash hooks then
  redirect them.
- **`indexio run -- <command>`** runs a command through the shell and keeps its output out
  of the context: up to 60 lines are printed as they are; longer output is stored under
  `<data-dir>/runs/<repo>/<stamp>-<slug>.log` (the repo is the registered one containing
  the working directory) and a digest is printed instead — exit code, size, the first 12
  lines, up to 25 lines that look like errors with their line numbers, the last 25 lines,
  and the `runs:<repo>/<log>` pointer for `read_span`. The Bash hook rewrites script and
  build commands (`python`, `node`, `npm`, `pytest`, `go`, `make`, `gradle`, `dotnet` …) to
  go through it via the hook's `updatedInput`; pipelines, redirects, one-liners and the
  commands rtk rewrites are left alone. The command text travels in an environment variable
  and is `eval`ed, because the MSYS runtime strips single quotes from argv on Windows. The
  `runs` folder is a plain source: the server's three-minute import cycle delta-indexes and
  embeds it, `recall` searches it alongside the transcripts (the session's own repo first),
  and its logs are searchable like any file.
- **A 30-day window.** `INDEXIO_RETAIN_DAYS` (default 30, 0 = keep) bounds both sources:
  run logs older than the window are deleted and tombstoned by the next delta, and a
  transcript idle longer than the window has its rendered parts removed and is not
  rendered again until it changes. Neither the transcripts themselves nor the code index
  are touched.
- **A lone recall hit comes with its exchange.** When `recall` returns one hit (or was
  asked for one), the transcript lines around it are appended, so the model does not spend
  a second call on the pointer it was just given.

## 32. A missed path names the file it probably meant

In one session (236 calls, −50 % against the built-in tools) three `file_outline` calls
failed with `is not indexed`: the agent had guessed `components/Recurring.tsx` for a file
that lives under `pages/`, and two names that do not exist at all. Each miss cost a
`list_files` or `code_grep` round trip before the right call. The miss now carries the
indexed paths of that repo with the same file name (else the same stem, up to five):
`repo:lib/foo.rs is not indexed; did you mean src/foo.rs`. Both `file_outline` and
`read_span` use it; a name with no relative is reported as before.

## 33. Two more shapes seen in the wild

- **An unbalanced parenthesis in a grep pattern.** `export function rateLimit|return
  async|):.*=>` — the `)` meant a literal — was refused as an invalid regex, and the agent
  spent a turn on it. `code_grep` now escapes the parentheses that have no partner (outside
  classes, not already escaped) and retries; the result is prefixed with `unbalanced
  parenthesis treated as literal:`. A pattern broken for another reason still fails.
- **`start` after `end` in `read_span`.** `360..130` and `336..325` came back as one line
  each. The pair is now read as swapped: the lines between them, which is what the agent
  asked for the second time in both cases.
- **`path` and `repo` as arguments of `code_grep`.** The same agent sent
  `path: "app/backend/src/middleware/rate-limit.ts"` next to the pattern, the way the
  built-in Grep takes it; the server ignored the argument and answered from the whole
  repo. Both are accepted now and appended as the `repo:` and `path:` filters.
- **Two binaries, one data dir.** A firmware `.ino` file was indexed by the new binary
  (§31 extension map) and tombstoned again by the running session's older server, whose
  walker did not know the type and took the doc for a deleted file. A content delta now
  keeps a doc whose type it does not recognise while the file exists on disk; only a file
  that is gone is tombstoned. The older binary in that session keeps its old behaviour
  until the session restarts, which the `binary updated` note on its next result asks for.
- **A read in disguise.** Once `cat` and `sed -n` were refused, one session read files with
  `python -X utf8 -c "src=open(r'…/lib.rs').read(); i=src.find('\"goto\" =>');
  print(src[i:i+3000])"`. Two things went wrong: the `-X utf8` before `-c` hid the
  one-liner from the runner rewrite, so the 77 lines the agent asked for came back as a
  digest; and the read itself bypassed the index. Now a one-liner is recognised whatever
  flags precede `-c` (`-e`, `-p`, `-r` for node, ruby, perl, php) and is never wrapped;
  and a one-liner that only opens a file to print it — an `open(`/`readFileSync(` literal,
  no write, spawn or edit — is judged like `cat`: when the file is indexed the call is
  refused with `code_grep {pattern:<the .find literal>, repo, path} then read_span`, or
  `file_outline` then `read_span` when there is no literal. A one-liner that writes,
  spawns or opens a variable path runs as typed.

## 34. The Read baseline follows the file's version

§22 charged the whole-file `Read` once per file per session, so a session that came
back to the same files hour after hour (one session: +40 % in its second day, −50 % overall)
was scored as if the harness had read each file once and remembered it. The harness
does not: an agent re-reads a file after editing it, and again after a compaction. The
baseline now keys on the file's content id (`Engine::file_version`, the doc's blob):
the first span of a version is charged the whole file, a repeat span of the same
version nothing, and a span after the file changed the whole file again. Compactions and
re-indexes of unchanged content keep the id.

## 35. Other repos are folded behind the session's repo

A swarm building one project on a machine that also holds an older copy of it got every
`code_grep` and lexical `code_search` answer twice: the copy's files carried the same
identifiers, and the built-in Grep it replaces would have searched one folder. Measured
against that baseline the answers were 20–40 % larger, one of them 3.9 KB against 246
bytes. When the session's repo has hits, other repos now contribute at most three rows
(the best-ranked, one file each) and the rest is a count: `+7 more in 2 other repos
(pass repo: to search one)`. Without a session repo, or when it has no hits, every repo
is listed as before, so a cross-repo lookup still works.

## 36. Hidden folders that hold configuration

A session asked `read_span` for `.github/workflows/ci.yml` and was told the file was not
indexed. The walker skipped every dot-directory, so CI workflows, `.cargo/config.toml`,
dev-container and editor configuration — tracked, small, and exactly what an agent asks
about when a build fails — were never in the index. Worse, the HEAD-tree listing did not
skip them while the working-tree listing did, so a HEAD sync added them and the next
working-tree refresh tombstoned them. One rule now applies to HEAD trees, working trees
and plain folders (`under_skipped_dir`): an allow-list of hidden folders is indexed
(`.github`, `.gitlab`, `.circleci`, `.cargo`, `.devcontainer`, `.vscode`, `.config`,
`.husky`, `.changeset`, `.storybook`, `.well-known`), every other dot-directory is
skipped as before.

## 37. Next

- Product quantisation (≈256 B/chunk with the binary codes as the prescan) when the
  corpus outgrows int8.
- The rindex model is still one file per data dir written by whichever server saves
  last (each server's overlay merges onto its own base): concurrent sessions' model
  updates overwrite each other, as before P10.
- **Scale estimate** at 8,000 repos (≈24 GB raw at the current mix): content ≈6 GB,
  postings ≈7 GB, vectors ≈80 GB as int8 / ≈10 GB with PQ, versus ≈900 GB with the
  pre-P10 layout.
