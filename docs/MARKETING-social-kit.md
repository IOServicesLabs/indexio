# indexio — Viral Marketing Kit (Social)

**Campaign number: 97% fewer tokens.**
Source: `bench/ab_readme.py`, published in the README — the tasks a coding agent
actually performs, run through Claude Code's built-in `Grep`/`Read`/`Glob` and
through indexio on the same repositories. 1,579,270 tokens in, 50,212 out.
Three real repositories, 44 in the index, n=30 per task, tiktoken-counted.

Backup numbers: **2,515,228 tokens saved** in seven days of real agent traffic
(−50% on the 3,074 calls with a built-in equivalent, verified twice), **310×**
faster definition lookups (31 ms → 0.1 ms), **10 ms** server start reading **0
bytes**, **5.3 GB → 911 MB** index, **13** MCP tools, **325** tests, **one**
Rust binary, **zero** bytes uploaded.

0.1.6 numbers (see the "0.1.6 — what's new" section): past sessions kept as **~1k-token
cards** (86 sessions: **44.9 MB → 265 KB**), **4%** of indexed files were invisible to
semantic search and now aren't, best-answer rank **0.565 → 0.674** on questions asked from
inside a repo, outlines **13%** smaller, **8%** fewer tokens on replayed real calls.

One-liner: *indexio is the Rust code index for AI coding agents — it indexes every
repo you own once, answers in milliseconds, and cuts 97% of the tokens your agent
spends reading code. One binary. No database. Nothing leaves your machine.*

Install lines to repeat everywhere:
- Linux/macOS: `curl -fsSL https://raw.githubusercontent.com/IOServicesLabs/indexio/main/install.sh | sh`
- Windows: `irm https://raw.githubusercontent.com/IOServicesLabs/indexio/main/install.ps1 | iex`
- npm: `npm install -g indexio` · pip: `pip install indexio-cli`
- Docker: `docker pull ghcr.io/ioserviceslabs/indexio`
- Then: `indexio add ~/code` → `indexio setup claude` → `indexio hook install`
- Upgrade: run the same install line again, then `indexio hook install`; the index carries over
- Repo: `github.com/IOServicesLabs/indexio`

**The install hook worth leading with:** you don't install indexio. You paste a
block into your agent and *it* installs itself, indexes your code, registers its
own MCP server and verifies the result. The README has the block.

---

## 0.1.6 — what's new (post copy)

Source for every figure below: A/B replays of real agent calls against the 0.1.5 binary on a
copy of the same index (one machine, 48 repos, 10,991 files), plus a 20-question eval asked
from inside the repo. The release notes carry the same table.

| Change | Result |
|---|---|
| Fewer rows when a search matches nothing; recall runs its two searches in parallel | code_search 9–23% fewer tokens; recall about 1.5–2× faster |
| Session cards + `recall {brief:true}` | 86 past sessions summarized in 265 KB, vs 44.9 MB of rendered transcripts |
| Search results that just read `}` removed | Less noise, search quality unchanged |
| `repo:` filter fix + your own repo searched first in hybrid | Best-answer rank score 0.565 → 0.674, right file in the top 5: 80% → 85% |
| Missing-vector repair + `embcas-stats --coverage` | Fixes 441 files (4%) that semantic search couldn't find |
| Test modules folded in outlines | file_outline 13% fewer tokens |
| Session-start hook | 71–112 tokens pointing a new session at the previous one; 76–291 tokens of working set after a compaction |

### X / Twitter — 0.1.6 single post

> Your agent compacts, and forgets which files it was editing.
> indexio 0.1.6 keeps every past session as a ~1k-token card — what was asked, what changed, what shipped — and hands a new session a one-line pointer to the last one.
> 86 sessions: 44.9 MB of transcript → 265 KB of cards. Still one Rust binary. Still nothing uploaded.

### X / Twitter — 0.1.6 thread

**1/**
indexio 0.1.6 is out. The theme: an agent's past is context too, and it was the most expensive context it had. 🧵

**2/**
Every session now gets a card: the requests, the files changed (most-edited first), the commits, how it ended. About 1k tokens. `recall {brief:true}` returns them. 86 of my sessions went from 44.9 MB of transcript to 265 KB of cards.

**3/**
A new session gets one line pointing at the previous one — title, when, how much changed, the last commit — for 71–112 tokens. After a compaction, the session gets back its own working set: the files it was editing and what it committed.

**4/**
We also found 4% of our own code was invisible to semantic search. A lexical-only writer (our shell hook re-indexes before it blocks a `cat`) indexed files that nothing ever embedded. The background sync now repairs it; `indexio embcas-stats --coverage` shows the gap.

**5/**
And a scoping bug: `repo:indexio` searched every repo, because another repo's name contains "indexio". Exact names win now, and your own repo gets its own ranked pass. On questions asked from inside the repo: best-answer rank 0.565 → 0.674.

**6/**
One idea we measured and didn't ship: answering a re-read with "only what changed since you last read it". It would save at most 2%, and the server can't tell whether the model's context was compacted in between. A partial answer to a model that lost the file is worse than a full one.

**7/**
Upgrade: run the same install line again, then `indexio hook install`. The index carries over. github.com/IOServicesLabs/indexio

### X / Twitter — 0.1.6 standalone posts

> 4% of our code was invisible to our own semantic search.
> Our shell hook re-indexes a repo before it tells the agent "use the index instead of cat". It updated the lexical index. Nothing ever embedded those files.
> Every search leg looked healthy. It just never returned them.

> `repo:indexio` searched every repository we own.
> The filter matches by substring, and another repo is called `themlisten_indexio`. Two matches means "ambiguous", ambiguous means "no filter", and no filter means no error.

> We had a feature that would have cut re-reads to just the changed lines.
> We measured it first: ≤2% of read tokens, and it can hand a partial file to a model that just compacted and lost the original.
> Not shipped. The measurement was the feature.

> Search results that just said `}`.
> The last chunk of a file is often nothing but closing braces. Its vector sits at the average of everything, so it matches every query a little. We stopped returning chunks with no words in them.

### LinkedIn — 0.1.6

> **Your coding agent's most expensive context is its own past.**
>
> When a long session compacts, the model keeps a summary and loses the working set: which files it was editing, what it already committed. A new session in the same repo starts from nothing and re-explores.
>
> indexio 0.1.6 keeps every past session as a card of about 1,000 tokens: what was asked, which files changed, the commits, how it ended. 86 real sessions went from 44.9 MB of transcript to 265 KB of cards. A new session gets a one-line pointer to the last one (71–112 tokens); a compacted one gets its working set back.
>
> Two bugs we found while building it are worth more than the feature:
> • 4% of indexed files had never been embedded, so semantic search could not return them. A lexical-only writer indexed them and nothing noticed. The background sync now repairs it.
> • A repo filter matched by substring, so `repo:indexio` quietly searched every repo. Fixing it, and giving the session's own repo its own ranked pass, moved best-answer rank from 0.565 to 0.674.
>
> Neither produced an error. Both just made the agent a little worse at its job.
>
> Apache-2.0, one Rust binary, nothing uploaded: github.com/IOServicesLabs/indexio

### Discord — 0.1.6 announcement

> :crab: **indexio 0.1.6**
>
> • **Session cards:** every past session as ~1k tokens (asks, files, commits, outcome). `recall {brief:true, query:""}` for the latest here
> • **Session-start hook:** new sessions get a pointer to the previous one; compacted sessions get their working set back
> • **4% of files were invisible to semantic search** — fixed and self-repairing; check with `indexio embcas-stats --coverage`
> • `repo:` filters scope correctly, your own repo is searched first, outlines fold test modules
>
> Upgrade: same install line as before, then `indexio hook install`. Restart your agent sessions to pick it up.

---

## X / Twitter — single post

> Your coding agent just read 2,000 lines to find 12.
> It didn't know which 12 until it looked. So it paid for all of them.
> indexio indexes every repo you own and returns the span. **97% fewer tokens.**
> One Rust binary. No database. Nothing leaves your machine.

### X / Twitter — thread (longer reach)

**1/**
Your AI coding agent is broke and it's not the model's fault. Ask it where a function lives and it greps, gets 40 matches, reads a 2,000-line file to find 12 lines, and pays for every one. We indexed the whole thing instead. 🧵

**2/**
The number: across the tasks agents actually issue — find a symbol, list its callers, read a function, answer a question about the code — built-in tools cost 1,579,270 tokens. indexio cost 50,212. **97% fewer.**

**3/**
The biggest single win isn't glamorous. It's bounded reads. Agents don't want files, they want spans — they just don't know which span until they look. 419,380 tokens of whole-file reads became 41,612 of exactly the right lines.

**4/**
The second win is the one grep can't do at all. "Where is parallel directory traversal implemented?" has no regex. Hybrid search answered 11 of those for 1,883 tokens. Grepping the words cost 1,144,842. That's **−99.8%**.

**5/**
Speed, because tokens aren't the only tax. Definition lookup: 31 ms → **0.1 ms**. Identifier grep: 31 ms → 0.7 ms. And every Bash call your agent makes pays ~25 ms of shell + hook overhead before the command even runs. An MCP call pays zero.

**6/**
It's one Rust binary. No database, no daemon, no service, no cloud. The index is a folder — copy it to another machine and it works. Nothing is ever uploaded. Your proprietary code stays on your disk, which is not true of most code search you can buy.

**7/**
13 MCP tools: `code_search`, `code_grep`, `find_symbol`, `who_calls`, `file_outline`, `read_span`, `impact_of_symbol`, `impact_of_diff`, `recall`, + 4 more. Your agent picks the right one — the server tells it which built-in tool each replaces.

**8/**
`impact_of_symbol` is the one people don't expect. Every transitive caller and importer of a symbol, across every repo you own, in milliseconds. You cannot grep for this. Agents try, badly, until they run out of context.

**9/**
You don't install it. You paste a block into Claude Code / Cursor / Codex and the agent installs it, indexes your repos, registers its own MCP server, and verifies. Then it's faster at its own job.

**10/**
Seven days of real traffic on one machine: 6,893 logged calls, **2,515,228 tokens saved**. Not a benchmark — just a week of work.

Apache-2.0. `github.com/IOServicesLabs/indexio` — star it, index something huge, tell me what broke.

### X / Twitter — standalone posts (no thread needed)

> Our code search was extremely fast at returning nothing.
> The scan budget ran out before it reached the matching files, so every all-lowercase query came back empty — and the agent concluded the function didn't exist.
> The latency dashboard was green the whole time.

> We found 270 MB in our index that no code path had ever read.
> The only thing that touched it was the shard merge, which decoded it in order to re-encode it.

> We deleted a 2 GB cache and the pipeline got faster.
> Embedding a chunk: 0.4 ms. Reading its cached vector back: slower than that.
> A cache whose hit is slower than its miss is just a disk leak.

> 42% of everything our coding agent did in a week was read a file.
> Not search. Not analyze. Read.
> Make reading cheap and you've fixed 90% of the bill.

---

## LinkedIn

> **My AI coding agent was spending 97% of its context window on code it didn't need.**
>
> Here's the uncomfortable math. Ask an agent where a function is defined. It runs a grep, gets forty matches, picks a file, and reads all 2,000 lines of it to find the twelve that matter. It pays for every line — and then pays again on the next question, because nothing it read was kept.
>
> We built indexio to answer the narrow question instead of shipping the file.
>
> What it does:
> 🦀 Indexes every repository your organization owns, one time. One Rust binary, no database, no service to operate.
> 📉 Cuts the tokens an agent spends reading code by 97% on our benchmark — 1,579,270 → 50,212 across the tasks agents actually issue.
> ⚡ Answers in milliseconds. Definition lookup went from 31 ms through the shell to 0.1 ms through the index.
> 🔒 Nothing is uploaded. The index is a folder on your disk. That matters if your code is the product.
> 🤖 Speaks MCP natively — 13 tools for Claude Code, Cursor, or any MCP client.
>
> Then we ran it for a week of ordinary work and logged every call against what the standard tooling would have returned: **2,515,228 tokens saved** across 6,893 calls.
>
> Two things I'd want to know if I were reading this:
>
> 90% of that saving came from one unglamorous feature — bounded file reads. Not the call graph, not the semantic search. If you take one idea from this, take that one; it probably applies to your stack whether or not you use our tool.
>
> And three of our tools came out *worse* than what they replaced. One costs 24,000 tokens the baseline didn't spend. We published those rows, because a benchmark table with no losing row is an advertisement.
>
> Apache-2.0, benchmarks in the repo: github.com/IOServicesLabs/indexio
> If you're running coding agents at any scale, this is the layer under them. Run it, and tell me where it loses.

---

## Reddit — r/rust

**Title:** Show r/rust: I built a code index for AI coding agents — one mmap'd binary, 0.1 ms symbol lookups, 97% fewer tokens

**Body:**

> Most "give the AI your codebase" products are a vector DB, a service to run, and your source on someone else's disk. I wanted one binary and a folder.
>
> indexio is a trigram index + symbol table + call graph + semantic plane over every repo you own. The design decisions that mattered:
>
> - **Trigram index, not a resident AST.** We parse with tree-sitter, take symbols, call edges and chunk boundaries, and throw the AST away. A resident AST costs 20–50× the text size; trigram shards cost 1–3×.
> - **Immutable mmap'd shards with tombstone bitmaps.** A build writes new shards, a delete is a bitmap flip, compaction merges in the background. Readers never block writers.
> - **Content-addressed store keyed by blake3.** A fork, a vendored copy or a duplicated file is indexed and embedded once. Indexing `serde` then a fork of it hits 100% cache: 3.0 s instead of 10.0 s.
> - **Everything is mapped, including the model.** We had a 261 MB bincode blob that was `fs::read` + deserialised at every server start — fine for one process, but we run one per agent session, so six sessions meant 1.8 GB of startup reads and 3.6 GB of RSS holding six copies of the same bytes. Rewrote it as a flat mapped file (FNV-1a bucket table, linear probing, 16-byte records, 8-byte aligned). **188 ms → 10 ms start, 302 MB read → 0, 417 MB RSS → 74 MB**, and the pages are now shared across sessions.
>
> Two things we deleted rather than optimised. Our postings stored every byte offset of every trigram occurrence — 270 MB for 143 MB of text — and the only consumer was the shard merge, decoding them to re-encode them. Gone: postings 5× smaller. And a 2 GB f32 embedding cache whose hits were slower than recomputing the value (0.4 ms to embed a chunk, longer than that to read 8 KB back). Replaced with a 16-byte-per-chunk seen-set: 2.09 GB → 4 MB.
>
> Vector rows are int8 with a per-row scale (`max|x|/127`), rescore is `scale · dot(q, row_i8)`. 3.55× smaller, 98% top-10 overlap with f32.
>
> Data dir went 5.3 GB → 911 MB for the same index. Whole benchmark workload 1,465 ms → 111 ms.
>
> Apache-2.0, 325 tests, `cargo install --git https://github.com/IOServicesLabs/indexio indexio`.
> Repo: https://github.com/IOServicesLabs/indexio — happy to talk about the posting codec, the tombstone/compaction design, or why `.stale` parking exists (Windows won't let you unlink a file another process still has mapped).

## Reddit — r/programming

**Title:** A case-insensitivity optimization made our code search silently return zero results

**Body:**

> We use smart-case: an all-lowercase query is treated as case-insensitive. Our trigram index is case-sensitive, so those queries couldn't use it and fell through to a brute scan with a 64 MiB budget.
>
> On a small index that's just slow — about 40 ms. The failure mode only appears at scale: on a large index the scan budget is exhausted *before it reaches the files that actually match*. The query returns zero hits with a `truncated: true` flag that nothing surfaces, and every caller reads "zero hits" as "no such symbol."
>
> A slow search tells you it's slow. A fast, empty search tells you nothing is there.
>
> The fix was to stop treating case-insensitivity as a reason to bypass the index. For each trigram of the literal, union the posting lists of its ASCII case variants — at most 8 per trigram, since a trigram has at most 3 cased letters. That union is an exact superset of the case-insensitive match set, so verification still decides correctness. 40 ms / 0 hits became 11 ms / hits.
>
> The part I keep chewing on: our latency benchmarks were green throughout, because not one of them asserted on the result set. A performance test that doesn't check correctness will happily certify a search engine that has stopped finding things.
>
> Context: this is from a code index for AI coding agents, where an empty result doesn't produce an error — it produces a confident model concluding the function doesn't exist. Write-up and benchmarks: https://github.com/IOServicesLabs/indexio

## Reddit — r/LocalLLaMA

**Title:** Stop letting your agent read whole files — indexio gives it symbol lookup, call graphs and spans, 97% fewer tokens, fully local

**Body:**

> The silent killer of local coding agents isn't the model, it's that every "look at this code" call lands a whole file in context. Ask where a function is defined and you get 2,000 lines to find 12.
>
> indexio indexes your repos once and gives the agent 13 MCP tools instead:
>
> - `read_span` — exact line range, or the whole definition at a line. In our bench, 419,380 tokens of whole-file reads became 41,612.
> - `file_outline` — every definition with line ranges, a few hundred tokens instead of the file
> - `find_symbol` / `who_calls` — definitions and call sites, 0.1 ms
> - `impact_of_symbol` — transitive callers across every repo. No grep equivalent exists.
> - hybrid search — "where is parallel directory traversal implemented?" has no regex. 11 such questions: 1,883 tokens vs 1,144,842 for grepping the words. −99.8%.
>
> All tasks: 1,579,270 → 50,212 tokens. **97% fewer.**
>
> Fully local — one Rust binary, no database, no service, nothing uploaded. The index is a folder you can copy between machines. Works with any MCP client, so Claude/Cursor and your Ollama-backed setup get the same tools.
>
> Seven days of real usage: 6,893 calls, 2,515,228 tokens saved. Three of our tools came out worse than the built-in equivalent and those rows are published too.
>
> `pip install indexio-cli` then `indexio add ~/code`.
> Repo: https://github.com/IOServicesLabs/indexio

---

## Hacker News — Show HN

**Title:** Show HN: indexio – a code index for AI coding agents (Rust, MCP, runs entirely local)

**Body:**

> I got tired of watching agents read 2,000-line files to find twelve lines, so I built the index I wanted.
>
> - One Rust binary. No database, no daemon, no cloud. The index is a directory you can copy to another machine. Nothing is uploaded.
> - Trigram index + tree-sitter symbols + a name-based call graph + an optional semantic plane, over every repo you own. The AST is parsed and discarded — a resident AST costs 20–50× the text, trigram shards cost 1–3×.
> - 13 MCP tools. The server tells the model which built-in tool each one replaces, so the agent routes identifiers to lexical search and questions to hybrid search on its own, and reads files by outline and span instead of whole.
> - Bench (`bench/ab_readme.py`, in the repo): the tasks agents actually issue cost 1,579,270 tokens through Claude Code's built-in Grep/Read/Glob and 50,212 through indexio. Seven days of real logged traffic: 2,515,228 tokens saved across 6,893 calls.
> - Speed: definition lookup 31 ms → 0.1 ms, hybrid query 130–190 ms → 7–14 ms, server start 188 ms → 10 ms reading zero bytes.
>
> Three of our own tools cost *more* tokens than the built-in equivalent and those rows are in the README, along with a duplicate-row bug we found in our own usage logger while double-checking the numbers.
>
> Install: `curl -fsSL .../install.sh | sh`, `npm i -g indexio`, `pip install indexio-cli`, or Docker. Then `indexio add ~/code`. There's also a block you paste into your agent that makes *it* do the install, the indexing and the MCP registration.
>
> Apache-2.0: https://github.com/IOServicesLabs/indexio
> The benchmarks are in /bench and the design notes in /docs. What would you want to see measured before trusting this on a real codebase?

**Alternative title, if we'd rather lead with the bug story than the launch:**
`Our code search was fast at returning nothing`

---

## Product Hunt

**Tagline:** The Rust code index that cuts your AI agent's token bill by 97%

**Description:**

> Your AI coding agent shouldn't read 2,000 lines to find twelve.
>
> indexio indexes every repository your team owns — one time — and gives your agent symbol lookup, call graphs, impact analysis and exact line spans instead of whole files. On our benchmark, the tasks agents actually perform cost 1,579,270 tokens through built-in tools and 50,212 through indexio.
>
> 🦀 One Rust binary. No database, no service, no cloud bill.
> 🔒 Nothing leaves your machine. The index is a folder on your disk.
> ⚡ 0.1 ms symbol lookups, 10 ms cold start, millisecond queries across 40+ repos
> 🤖 13 native MCP tools for Claude Code, Cursor, and anything that speaks MCP
> 📊 Ships its own usage ledger — it tells you what it saved, including where it lost
>
> Install in 60 seconds: `pip install indexio-cli` then `indexio add ~/code`.
> Or paste one block into your coding agent and let it install itself.
>
> Apache-2.0, built by IOServicesLabs.

**First comment (maker):** "Happy to get into any of it — the trigram/AST tradeoff, why the whole index is mmap'd, or the case-insensitivity bug that made our search return zero results silently on large indexes. Fastest way to judge it: `indexio add` your biggest monorepo and ask your agent something it would normally need five greps for."

---

## dev.to / blog intro

> ## Your coding agent is reading 97% garbage. Here's the fix.
>
> Every agent harness ships a "read this file" tool, and almost all of them make the same mistake: they hand the model the whole file. The agent asked where `parse_config` is defined. It got two thousand lines, because the tool had no way to return twelve.
>
> This is the context tax — the gap between what the agent needed and what it had to read to find it. And it compounds, because nothing the agent read is kept. The next question pays again.
>
> We built **indexio** on a simple bet: agents don't need files, they need answers to narrow questions. So we index the code once and let the tool return the span, the signature, the call site, the blast radius.
>
> Measured on our own bench across three real repositories: the tasks agents actually issue cost **1,579,270 tokens** through the built-in Grep/Read/Glob and **50,212** through indexio. A 97% reduction. The single biggest line item is the least interesting one — bounded reads took 419,380 tokens down to 41,612.
>
> Under the hood it's one Rust binary: a trigram inverted index with an FST term dictionary, tree-sitter symbols and call edges (the AST is parsed and thrown away), immutable memory-mapped shards with tombstone bitmaps, a blake3 content-addressed store so a forked repo is indexed once, and an optional int8 vector plane for the questions that have no regex.
>
> In this post: why we discarded the AST, how deleting 270 MB of index data that nothing read made queries 5× cheaper, the cache we removed because its hits were slower than its misses, and the case-insensitivity bug that made our search return zero results without telling anyone. `pip install indexio-cli` — let's go.

---

## YouTube

**Title options:**
1. "I cut my AI agent's token bill by 97% (Rust)"
2. "Your AI reads 2,000 lines to find 12"
3. "My code search was fast at returning nothing"

**Description:**

> Your coding agent reads whole files to find single functions. indexio indexes every repo you own and returns the exact span — 1,579,270 tokens down to 50,212 on our benchmark. 97% fewer.
>
> One Rust binary, no database, nothing uploaded. 13 MCP tools for Claude Code, Cursor, and anything that speaks MCP. 0.1 ms symbol lookups.
>
> 0:00 The 2,000-lines-to-find-12 problem
> 1:15 What an index gives an agent that grep can't
> 3:00 The 97% number, measured
> 4:40 The bug: fast at returning nothing
> 6:30 MCP setup in 60 seconds
> 8:00 Where indexio loses (the rows we published)
>
> Repo: github.com/IOServicesLabs/indexio
> `pip install indexio-cli && indexio add ~/code`

**30-second hook script:**

> [Cold open, terminal visible]
> "I asked my AI where a function was defined. Watch what it reads." [`Read` dumps 2,000 lines]
> "Two thousand lines. It needed twelve. And it paid for all of them — then forgot, and paid again on the next question."
> [cut: `indexio span`, twelve lines appear]
> "Same answer. Twelve lines. Across our whole benchmark that's ninety-seven percent fewer tokens, and the lookup takes a tenth of a millisecond."
> [cut: install command typing itself]
> "One Rust binary. No database. Nothing leaves your machine."
> "Link in the description. Your context window will thank you."

---

## TikTok / Instagram Reels / Shorts

**15-second script:**

> **(0–2s, text on screen: "2,000 lines")** "This is how much code your AI just read to find one function."
> **(2–6s, terminal: `indexio span` returns 12 lines)** "This is what it actually needed. Twelve lines."
> **(6–10s)** "Ninety-seven percent fewer tokens. A tenth of a millisecond. Written in Rust."
> **(10–15s, install command on screen)** "One binary. Nothing leaves your machine. Link in bio."

**Alternative hook — the bug (higher engagement with a dev audience):**

> **(0–3s, terminal shows `0 results` in 40ms)** "This search is broken and it looks perfect."
> **(3–8s, zoom on `truncated: true`)** "Zero results. Forty milliseconds. The function exists."
> **(8–12s)** "Our scan budget ran out before it reached the matching files. Every perf test stayed green — none of them checked the results."
> **(12–15s)** "A slow search tells you it's slow. A fast empty one tells you nothing's there."

**Caption:**
> 97% fewer tokens. 0.1ms lookups. 1 Rust binary. 🦀 Nothing leaves your machine. Your coding agent's missing layer — link in bio. #rustlang #ai #llm #coding #techtok

---

## Discord (server announcement)

> :crab: **indexio is live — the local code index for AI coding agents**
>
> The pitch in one number: the tasks your agent actually performs cost **1,579,270 tokens** through built-in tools and **50,212** through indexio. **97% fewer**, same answers.
>
> • Indexes every repo you own, once — one Rust binary, no database, no service
> • **Nothing is uploaded.** The index is a folder on your disk you can copy anywhere
> • 13 MCP tools: `code_search`, `find_symbol`, `who_calls`, `read_span`, `impact_of_symbol`, `recall`, + 7 more
> • 0.1 ms symbol lookups, 10 ms cold start, millisecond queries across 40+ repos
> • Ships its own usage ledger — it reports what it saved, including where it lost
>
> Try it in two lines:
> ```
> pip install indexio-cli
> indexio add ~/code
> ```
> Or paste one block into Claude Code and let it install itself.
> Repo: <https://github.com/IOServicesLabs/indexio> — star it, index your biggest monorepo, then tell me what broke.

---

## GitHub (release notes / repo social preview)

**Social preview text (card):**
> indexio — 97% fewer tokens. 0.1 ms lookups. 1 Rust binary. 0 bytes uploaded.

**Release note blurb:**

> ## Your agent's context window just got 30x bigger
>
> Whole files in → exact spans out. On our bench, the tasks a coding agent actually issues cost **1,579,270 tokens** through Claude Code's built-in Grep/Read/Glob and **50,212** through indexio — 97% fewer, same answers.
>
> Seven days of real logged traffic: **2,515,228 tokens saved** across 6,893 calls. 90% of that comes from `read_span` returning the lines you asked for instead of the whole file.
>
> Where we cost more than the baseline: `file_outline` (+24k tokens over 218 calls, buying spans and scope nesting instead of regex matches), `code_search` on its lexical path (a tie with grep), and `find_symbol` (worse than a targeted grep — it lists every repository). Those rows are in the README.
>
> Also in this cycle: data directory 5.3 GB → 911 MB, server start 188 ms → 10 ms reading zero bytes, benchmark workload 1,465 ms → 111 ms. Fixed a case-insensitivity path that could return zero results on large indexes.
>
> `pip install indexio-cli` · `npm i -g indexio` · `docker pull ghcr.io/ioserviceslabs/indexio`

---

## Hashtag / keyword bank

- #RustLang #LLM #AIAgents #MCP #ContextEngineering #BuildInPublic #DevTools
- X/LinkedIn: #artificialintelligence #opensource #tokenoptimization #codesearch #claudecode
- TikTok/IG: #rust #coding #ai #techtok #learnontiktok #programming
- Reddit: none — read the room, flaired posts only
- HN: none

---

## Pre-publish verification checklist

- [ ] **Reconcile the two token numbers before anything ships.** The README bench says **−97%** (1,579,270 → 50,212, n=30, three repos). A fresh single-repo run with the question row missing gave **−92%**. Both are real, from the same harness, different scopes. Pick one as the campaign number, put it in the README and in this kit, and don't let a second figure circulate.
- [ ] **Reconcile the two live-traffic numbers too.** The README quotes one day: 423 compared calls, −63%. This kit quotes seven days: 3,074 compared calls, −50%, 2,515,228 tokens. Use the seven-day figure — it was verified twice, against the tool's own accounting and an independent recount from the raw log — and update the README to match.
- [ ] **"97%" always travels with its denominator.** It is the token cost of a task mix, dominated by the read-a-function and answer-a-question rows. Never post it bare beside a claim about total context consumption.
- [ ] **"50%" is on the 3,074 calls with a mechanical built-in equivalent** — not on all 6,893, and not on the agent's total context. Say it before someone else does.
- [ ] **Call counts carry an asterisk.** The usage logger double-writes ~7% of rows (387 duplicate signatures, 530 extra rows). Token *ratios* are unaffected — a duplicated row duplicates both sides of the comparison — but absolute counts are overstated. Say "6,893 logged calls," never "6,893 distinct calls." Fix the logger before the next campaign.
- [ ] **Never claim end-to-end speedup.** We measured tool calls and lookups, not task completion. "310× faster" is that lookup. "10× faster agent" is not defensible and HN will ask.
- [ ] **"Rust" framing is clean here** — it genuinely is one Rust binary, unlike searchio's Rust-sidecar-plus-Python orchestrator. Say "one Rust binary" freely; it is a real differentiator.
- [ ] **The local/privacy claim is the sharpest wedge against hosted code search** — nothing uploaded, index is a copyable folder. Confirm no telemetry ships enabled by default before leaning on it in paid copy.
- [ ] **Keep the losing rows in.** `file_outline` (+24k), lexical `code_search` (a tie with grep), `find_symbol` (worse than a targeted grep). They are ~1% of the total and they are the reason the other numbers get believed. Don't let anyone edit them out for a cleaner table.
- [ ] **n=1 machine, one developer, Windows 11, 42-repo index.** State it up front rather than letting a commenter discover it.
- [ ] Swap the repo URL for a landing page if one exists at launch time.
- [ ] **0.1.6 figures carry their denominators.** "8% fewer tokens" is on replayed real calls where output changed or stayed identical (read_span, 85% of all tokens, did not change). "0.565 → 0.674" is a 20-question eval written by us, on one repo, asked from inside it. "44.9 MB → 265 KB" is storage of 86 sessions, not context saved per session.
- [ ] **Session cards save context only when used.** The hook's pointer costs 71–112 tokens per session start; the saving (sessions not re-exploring) has not been measured yet. Don't claim a context reduction for cards until it is.
- [ ] **Fresher live-traffic figure exists:** `index_stats` on 2026-09-29 reported 5,347 calls in 7 days, 3.92M tokens vs 12.58M for the built-in equivalents (−69%). Check the logger duplicate-row fix first, then reconcile it with the 2,515,228 / −50% figure above before either circulates.
