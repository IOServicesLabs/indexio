# SPEC-P6 — Impact analysis + agent context economy

Addendum to SPEC/SPEC-P2..P5. All engineering rules apply: no new external deps beyond
the workspace set, no persistent AST, no shard format change (everything in this phase is
computed at query time from data the shards already hold: symbol postings, call postings,
and the zstd'd file content).

Motivation (product goal): an LLM harness that is about to change code must be able to
ask, cheaply, "**what else does this touch?**" — across every indexed repo — and must be
able to pull *exactly* the lines it needs instead of whole files. P6 adds both.

## 1. indexio-symbols — symbol ranges (query-time)

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutlineItem {
    pub name: String,
    pub kind: SymbolKind,
    pub scope: String,        // as SymbolRec.scope
    pub start_line: u32,      // 1-based inclusive: line of the definition node
    pub end_line: u32,        // 1-based inclusive: last line of the definition node
}
/// Same definitions `extract` reports, with the full node range. Sorted by
/// (start_line asc, end_line desc) so parents precede children.
pub fn outline(lang: Lang, content: &[u8]) -> Vec<OutlineItem>;
/// Innermost item whose range covers `line` (smallest span wins).
pub fn enclosing(items: &[OutlineItem], line: u32) -> Option<&OutlineItem>;
```

`extract` is unchanged in behaviour (same symbols, same lines, same dedup): `outline` is
a second view over the same tree-sitter pass. Unsupported language → empty vec. Never
panics (catch_unwind, as `extract`).

## 2. indexio-query — `impact` module

### 2.1 Symbol impact (transitive reverse call graph)

```rust
pub struct ImpactSite {
    pub repo: String, pub path: String, pub line: u32, pub lang: Lang,
    pub symbol: String,   // the callee name matched at this hop
    pub caller: String,   // enclosing function of the call site ("" = top level)
    pub depth: u32,       // 1 = direct caller of a root symbol
    pub snippet: String,
}
pub struct FileImpact { pub repo: String, pub path: String, pub sites: u32, pub min_depth: u32 }
pub struct ImpactReport {
    pub roots: Vec<String>,          // symbol names the walk started from
    pub definitions: Vec<SearchHit>, // exact definitions of the roots (find_symbol, exact only)
    pub sites: Vec<ImpactSite>,      // BFS order (depth asc, then repo/path/line)
    pub files: Vec<FileImpact>,      // aggregated, sorted by (min_depth asc, sites desc)
    pub importers: Vec<SearchHit>,   // file-level dependents (see 2.3); empty for symbol walks
    pub truncated: bool,             // a fan-out cap was hit
    pub took_ms: u64,
}
pub struct ImpactOptions { pub depth: u32 /*default 2*/, pub max_sites: usize /*default 500*/,
                           pub max_fanout: usize /*per symbol, default 200*/ }
impl Engine {
    pub fn impact_symbols(&self, names: &[String], opts: &ImpactOptions) -> ImpactReport;
}
```

Algorithm: BFS over `ShardSet::call_postings`. Frontier₀ = `names`. For each name at
depth d < `opts.depth`: every call posting (doc, caller, line) becomes an `ImpactSite`
at depth d+1; its `caller` (if non-empty, not yet visited) joins frontier_{d+1}. A name's
postings are cut at `max_fanout` (truncated=true). Total sites are cut at `max_sites`
(truncated=true). Visited set on names prevents cycles. Name-based edges are best-effort
(SPEC indexio-symbols): `Foo::new` and `Bar::new` both resolve to `new` — the report says so
through the `symbol` field so an agent can filter by scope.

### 2.2 Diff impact (the "I am about to change this" flow)

```rust
pub struct ChangedSymbol { pub path: String, pub name: String, pub kind: SymbolKind,
                           pub scope: String, pub lines: Vec<u32> /* changed lines hit */ }
pub struct FileChange { pub path: String, pub old_ranges: Vec<(u32,u32)>, pub new_ranges: Vec<(u32,u32)>,
                        pub added: bool, pub deleted: bool }
/// Pure unified-diff parser: `--- a/`, `+++ b/`, `@@ -a,b +c,d @@`. Zero-length hunks
/// (pure insert/delete) map to the single line they touch.
pub fn parse_unified_diff(diff: &str) -> Vec<FileChange>;
/// Map changed line ranges to enclosing definitions. `new_content(path)` yields the
/// post-change file (working tree); the pre-change side comes from the index.
pub fn changed_symbols(&self, repo: &str, changes: &[FileChange],
                       new_content: &dyn Fn(&str) -> Option<Vec<u8>>) -> Vec<ChangedSymbol>;
/// changed_symbols → impact_symbols on the union of names, plus importers of every
/// changed file. Changed files themselves are excluded from `sites`.
pub fn impact_diff(&self, repo: &str, diff: &str,
                   new_content: &dyn Fn(&str) -> Option<Vec<u8>>, opts: &ImpactOptions)
                   -> (Vec<ChangedSymbol>, ImpactReport);
```

Changed lines that fall outside any definition (module-level code, imports) yield no
symbol; the file still contributes importers.

### 2.3 Importers (file-level dependents, best-effort)

`Engine::who_imports(&self, repo: &str, path: &str, limit) -> Vec<SearchHit>` builds a
language-specific regex from the file's module identity and runs it through the existing
lexical planner (`lang:` filtered, so it costs one regex search):

| Lang | module key | pattern (conceptually) |
|---|---|---|
| Rust | stem (`mod.rs`/`lib.rs` → dir) | `\b(mod|use)\b[^;]*\bKEY\b` |
| Python | stem (`__init__.py` → dir) | `^\s*(from|import)\s+[\w.]*\bKEY\b` |
| Go | parent dir | `"[^"]*/KEY"` inside `import` |
| TS/JS | stem (`index.*` → dir) | `(from|require\()\s*['"][^'"]*\/KEY(\.[cm]?[jt]sx?)?['"]` |
| Java | stem | `import\s+[\w.]*\.KEY\s*;` |
| C/C++ | stem | `#include\s*["<][^">]*\bKEY\.h` |

The file itself is excluded. Precision is "grep-grade" by design (cost-ladder band 1×);
the report labels the list `importers`.

### 2.4 Context economy

```rust
impl Engine {
    /// Definitions with ranges for one indexed file (None if not indexed).
    pub fn outline(&self, repo: &str, path: &str) -> Option<Vec<OutlineItem>>;
    /// Lines [start, end] (1-based inclusive) of one indexed file, capped at
    /// MAX_SPAN_LINES (400); returns (text, actual_end).
    pub fn read_span(&self, repo: &str, path: &str, start: u32, end: u32) -> Option<(String, u32)>;
}
```

## 3. Surfaces

CLI:
```
indexio impact --symbol NAME [--symbol NAME2 ...] [--depth N] [--max-sites N] [--json]
indexio impact --diff REPO [--base REF]        # runs `git diff -U0 REF` in the registered repo
indexio impact --diff-file PATCH --repo REPO   # patch from file ("-" = stdin)
indexio impact --file REPO:PATH                # importers + impact of every definition in the file
indexio outline REPO:PATH [--json]
indexio span REPO:PATH START[:END]
```
HTTP (same auth/ACL rules as /search; ACL filters sites/files/importers by repo):
```
GET  /impact/symbol?name=&name=&depth=&max_sites=
POST /impact/diff   {"repo":..., "diff":..., "depth":..}   (diff text supplied by client)
GET  /outline?repo=&path=
GET  /span?repo=&path=&start=&end=
```
MCP tools (in addition to the six existing ones):
- `impact_of_symbol {names[], depth?, max_sites?}`
- `impact_of_diff {repo, diff?, base?, depth?}` — with no `diff`, the server runs
  `git diff -U0 <base|HEAD>` in the registered repo's working tree (stdio MCP is the
  trusted local channel, SPEC-P3 §3).
- `file_outline {repo, path}`
- `read_span {repo, path, start, end}`

MCP tool results switch from pretty-printed to compact JSON (same payloads): ~25–35%
fewer tokens per call, measured on the fixture responses.

## 4. Tests (≥ 14)

outline: ranges for nested Rust impl/method, Python class/method, TS class; `enclosing`
picks innermost; unsupported lang empty. impact: depth-1 finds direct callers; depth-2
reaches transitive caller through the `caller` field; cycle terminates; fan-out cap
sets truncated; cross-repo edge (caller in shard/repo B of callee in repo A).
diff: hunk parsing (add/modify/delete/new file/deleted file, zero-length hunks);
changed_symbols maps a modified line to its method and ignores whitespace-only
outside-defs edits; impact_diff excludes the changed file from sites. importers: Rust
`use crate::foo::bar` and Python `from pkg.mod import x` resolve. span: clamps and caps.
Surfaces: CLI parse, HTTP route JSON shape + ACL filtering, MCP tools/list (10 tools) +
one call each.

## 5. Embed pipeline throughput (added after the P6 functional test)

Profiling (`RUST_LOG=indexio_embed=info indexio embed --all` prints per-phase timings) showed the
P5 pipeline spending, per chunk, ~3 ms in `RandomIndexingEmbedder::observe`, ~1.8 ms in
`embed` + a one-file-per-vector CAS write, and rebuilding the whole vec/BM25 index once
per repo (quadratic in fleet size). Changes, all result-preserving (vectors are verified
bit-for-bit identical to the P5 binary's; model files stay loadable and deterministic):

- **rindex `observe`** is split into a pure parallel half (`compute_update`: tokenize,
  cached labels, contributions pre-summed per (term, dim) in stream-visit order) and a
  sequential-equivalent apply. The vocabulary is sharded by term hash (`Vocab`, 32
  shards) so the apply runs shards in parallel; a term's context depends only on its own
  contribution sequence, so this is exactly equivalent to a serial pass. CTX_TOP
  truncation uses O(n) selection under the same total order as the former sort.
- **Label cache**: `label()` (blake3 XOF + sort) is memoised process-wide.
- **`embed`** is rayon-parallel across texts under the model read lock; the pipeline
  embeds CAS misses as parallel batches of 64.
- **Embedding CAS** is an append-only pack (`embcas/<model>/pack.bin` + `pack.idx`)
  instead of a file per vector; the old layout is still read as a fallback.
- **One index build per `embed --all`** (`embed_repos_with`): chunk/observe/embed each
  repo in turn, then write the vec index and BM25 sidecar once. `EmbedReport` gained
  `index_build_ms` (shared value on every report of the call). Single-repo embeds still
  carry the other repos' rows (text recovered by chunk_hash, now parallel per doc).
- **HNSW is opt-in at fleet scale**: the graph build is single-threaded (~5 min for
  184k rows of 2048 dims on this machine) while the flat binary prescan over the same
  rows takes tens of ms, so `IndexOptions::default().hnsw_threshold` is now 1,000,000
  rows, overridable with `INDEXIO_HNSW_THRESHOLD`.
- **`.civec` v2 is memory-mapped** (`CIVEC002`: vector block 8-byte aligned; v1 files
  still open into owned memory). Opening a 1.7 GB index costs the meta parse only; a
  query touches the binary codes plus the rescored candidates. This is indexio-embed's single
  `unsafe` site (`index::mapped`), same immutability argument as indexio-index shards.
- **Engine caches opened sidecars** (`Engine::vec_index` / `bm25_index`), validated by
  the file's (len, mtime) on each use, so a long-lived HTTP/MCP process opens them once
  per rebuild instead of once per query.
