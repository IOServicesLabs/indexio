# SPEC.md — indexio (P0+P1)

A high-performance enterprise code indexing engine in Rust. Sparse n-gram lexical core + symbol plane + global content-addressed store (CAS) + git delta ingest. Interfaces: CLI, HTTP (axum), MCP (JSON-RPC stdio). Derived from the research report (`/mnt/agents/output/enterprise-code-indexing.agent.final.md`, chapters 5 and 7).

This spec is the contract. Implement interfaces EXACTLY as written. Do not rename public items or change byte formats. If something is impossible, stop and report rather than improvising.

## Workspace layout

```
indexio/
├── Cargo.toml            # workspace, members below (pre-created)
├── crates/
│   ├── indexio-types/         # shared types + posting codec (pre-created by orchestrator; DO NOT MODIFY)
│   ├── indexio-core/          # n-gram extraction, sparse grams, verifiers
│   ├── indexio-index/         # shard format: write/read(mmap)/merge/tombstones
│   ├── indexio-symbols/       # tree-sitter symbol + call-site extraction
│   ├── indexio-ingest/        # git crawl (gix), CAS, delta ingest
│   ├── indexio-query/         # query parse/plan/candidates/verify/rank
│   └── indexio/               # binary: CLI (clap) + HTTP (axum) + MCP (stdio JSON-RPC)
└── tests/                # workspace integration tests (orchestrator stage)
```

Dependency direction: `indexio-types` ← everything; `indexio-core` ← `indexio-index`, `indexio-query`, `indexio-symbols`(no), `indexio-ingest`; `indexio-index` ← `indexio-query`, `indexio-ingest`; `indexio-symbols` ← `indexio-ingest`; `indexio-query` ← `indexio`; `indexio-ingest` ← `indexio`. No cycles.

Pinned dependency versions (workspace Cargo.toml, already set by orchestrator — agents may add deps ONLY to their own crate, prefer listed ones):
- serde 1 (derive), bincode 1.3, thiserror 2, anyhow 1, byteorder 1, memmap2 0.9, zstd 0.13, fst 0.4, roaring 0.10, aho-corasick 1, memchr 2, regex 1, blake3 1, rayon 1, tracing 0.1, tracing-subscriber 0.3
- gix 0.66 (default-features = false, features = ["parallel","serde"]) for git access
- tree-sitter 0.24 + grammars: tree-sitter-rust 0.23, tree-sitter-python 0.23, tree-sitter-go 0.23, tree-sitter-typescript 0.23 (tsx), tree-sitter-java 0.23, tree-sitter-cpp 0.23
- clap 4 (derive), axum 0.8, tokio 1 (full), serde_json 1, tower-http 0.6 (cors,trace)

Build with CARGO_BUILD_JOBS=2 (4GB RAM machine). Run tests ONLY for your own crate: `cargo test -p <crate>`.

---

## indexio-types (exists; authoritative — summary for implementers)

- `BlobId([u8;16])` — blake3(content)[..16]; global CAS key. `BlobId::from_content(&[u8])`.
- `Lang(u16)` enum-like constants: `UNKNOWN=0, RUST=1, PYTHON=2, GO=3, TS_JS=4, JAVA=5, C_CPP=6`. `Lang::from_path(&str) -> Lang` (by extension: rs, py, go, ts/tsx/js/jsx/mts/cts, java, c/h/cc/cpp/cxx/hpp).
- `SymbolKind(u8)`: `FN=0, STRUCT=1, CLASS=2, ENUM=3, TRAIT=4, IMPL=5, MOD=6, CONST=7, TYPE=8, METHOD=9, VAR=10, INTERFACE=11`.
- `SymbolRec { name: String, kind: SymbolKind, line: u32, col: u32, scope: String }` (scope = enclosing fn/class path or "").
- `CallRec { callee: String, caller: String, line: u32 }` (best-effort, name-based; caller may be "" for top-level).
- `ExtractedArtifact { ngrams: Vec<(Vec<u8>, Vec<u32>)>, symbols: Vec<SymbolRec>, calls: Vec<CallRec>, raw_len: u32, lang: Lang }` — serde+bincode. `ngrams` = sorted unique gram (3–8 bytes) → sorted byte-offset positions.
- `DocMeta { blob: BlobId, repo_id: u32, path: String, lang: Lang, raw_len: u32 }`.
- `SearchHit { repo: String, path: String, line: u32, col: u32, snippet: String, score: f32, lang: Lang }`.
- Posting codec (exact byte format below): `encode_postings(&[(u32, Vec<u32>)]) -> Vec<u8>`, `decode_postings(&[u8]) -> Vec<(u32, Vec<u32>)>`, `PostingCursor` with `next()`, `seek(docid)`.

### Posting byte format (little-endian, no padding)
A posting list = sequence of **blocks of ≤128 docs**. Per block: `u32 last_docid`, `u32 block_bytes` (bytes of body that follow). Body per doc entry: `varint(doc_delta_from_prev, 0 for first in block)`, `varint(pos_count)`, then `pos_count` × `varint(pos_delta)`. varint = LEB128 unsigned. Skip via `block_bytes`. `seek(docid)` skips blocks whose `last_docid < target` then scans.

---

## indexio-core — grams + verification

Public API (exact):
```rust
pub mod grams {
    /// Extract positional trigrams, extending to 4..=8-byte grams for grams listed
    /// in `common` (sparse-gram strategy: common grams get longer, rare stay trigrams).
    /// Returns sorted, deduped (gram, sorted byte offsets). Skips docs > MAX_DOC_BYTES (4 MiB).
    pub fn extract(text: &[u8], common: &CommonGrams) -> Vec<(Vec<u8>, Vec<u32>)>;
    /// All trigrams of `needle` (or extended grams where `common` says so), in order.
    pub fn grams_of(needle: &[u8], common: &CommonGrams) -> Vec<Vec<u8>>;
    pub struct CommonGrams { /* trigram -> required extension length (0=stay trigram) */ }
    impl CommonGrams { pub fn empty() -> Self; pub fn from_stats(counts: &[(Vec<u8>, u64)], total_docs: u64, threshold_doc_frac: f64) -> Self; pub fn ext_len(&self, tri: &[u8]) -> usize; }
}
pub mod verify {
    /// Literal verification: returns (line, col, snippet_line_text) matches. SIMD via memchr/aho-corasick.
    pub fn find_literal(content: &[u8], needle: &[u8], case_insensitive: bool, max_hits: usize) -> Vec<(u32,u32)>;
    /// Regex verification (regex crate, bytes). Returns (line,col) of each match start.
    pub fn find_regex(content: &[u8], pattern: &str, max_hits: usize) -> Option<Vec<(u32,u32)>>; // None = invalid regex
    /// Literal extraction from a regex (RE2-style required literals). Returns None if no usable literal.
    pub fn required_literals(pattern: &str) -> Option<Vec<Vec<u8>>>;
    pub fn line_col(content: &[u8], offset: u32) -> (u32, u32); // 1-based line, 0-based col
}
```
Rules: binary files (NUL in first 8KiB) are skipped upstream; `extract` assumes text. Non-ASCII is fine — operate on bytes. Grams crossing no boundaries (whole doc). Unit tests: roundtrip extract/grams_of consistency (every gram of needle "hello world" appears in extract of a doc containing it), common-gram extension, LEB128 edge cases (in indexio-types, already tested), literal/regex verify on tricky inputs (CRLF, unicode, >64KiB lines).

## indexio-index — shard format

Single-file immutable shards, mmap-read. Little-endian, 8-byte section alignment.

```
[0..64)   Header: magic "CIDXSHD1"(8B), u32 version=1, u32 section_count, u64 doc_count,
          u64 created_unix, u32 flags, u32 header_reserved, u64 common_grams_off (0=none), u64 common_grams_len
[64..)    Section directory: section_count × {u32 kind, u32 pad, u64 offset, u64 len}
Sections: DOCS=1, STRINGS=2, CONTENT=3, NGRAM_FST=4, NGRAM_POST=5, SYM_FST=6, SYM_POST=7,
          CALL_FST=8, CALL_POST=9, TOMBSTONES=10, META=11
```
- DOCS: doc_count records × 40B: `blob[16], u32 repo_id, u32 path_off, u32 path_len, u16 lang, u16 flags, u64 content_off, u32 content_len, u32 raw_len`. docid = record index.
- STRINGS: u32 repo_count, then repo_count × {u32 off,u32 len} into trailing string blob; path strings also in the blob (paths referenced from DOCS).
- CONTENT: zstd-compressed blob contents (one frame per doc, dictionary-compressed if dict present in META).
- NGRAM_FST: fst::Map<u64> gram(bytes) → (post_off<<20)|(post_len) … simpler: map gram → u64 = post_off; lengths implicit via next-offset table NGRAM_POST header. Simplify further: fst maps gram → post_off; each posting at post_off is length-prefixed: `u32 bytes` + posting payload. Same pattern for SYM and CALL sections.
- SYM_POST payload: `u8 kind, u32 line, u32 scope_off` per entry after docid deltas — use: varint(doc_delta), varint(line), u8 kind, varint(scope_off into STRINGS blob).
- CALL_POST payload per entry: varint(doc_delta), varint(caller_off into STRINGS), varint(line).
- TOMBSTONES: serialized roaring::RoaringBitmap of deleted docids.
- META: UTF-8 JSON `ShardMeta { repos: Vec<String>, doc_count: u64, total_raw_bytes: u64, gram_stats: Vec<(String,u64)> (top-10k common grams with doc counts), zstd_dict: Option<Vec<u8>> (base64), created: String }`.

Public API (exact):
```rust
pub struct ShardWriter { /* writes to a temp path, atomic rename on finish */ }
impl ShardWriter {
    pub fn new(dir: &Path) -> io::Result<Self>;
    pub fn add_doc(&mut self, meta: &DocMeta, content: &[u8], art: &ExtractedArtifact) -> io::Result<u32 /*docid*/>;
    pub fn finish(self, repos: &[String]) -> io::Result<PathBuf>; // builds FSTs, postings, compresses content (zstd level 3, trained dict if >100 docs), writes shard file `shard-<ulid>.cidx`
}
pub struct Shard { /* mmap */ }
impl Shard {
    pub fn open(path: &Path) -> io::Result<Self>;
    pub fn doc_count(&self) -> u64;
    pub fn meta(&self) -> &ShardMeta;
    pub fn doc(&self, docid: u32) -> Option<DocMeta>;
    pub fn content(&self, docid: u32) -> io::Result<Vec<u8>>;        // zstd-decompress
    pub fn postings(&self, gram: &[u8]) -> Option<PostingCursor>;
    pub fn symbol_postings(&self, name: &str) -> Vec<(u32 /*docid*/, u8 kind, u32 line, String scope)>;
    pub fn call_postings(&self, callee: &str) -> Vec<(u32, String caller, u32 line)>;
    pub fn tombstones(&self) -> &RoaringBitmap;
    pub fn common_grams(&self) -> CommonGrams;                        // from META gram_stats
    pub fn delete_docs(&mut self, docids: &[u32]) -> io::Result<()>;  // rewrites TOMBSTONES section in place (section is pre-sized to 64KiB, fits bitmap for ≤ ~1M tombstones; else error)
    pub fn path(&self) -> &Path;
}
pub struct ShardSet { shards: Vec<Shard> } // newest-first
impl ShardSet {
    pub fn open_dir(dir: &Path) -> io::Result<Self>; // loads all *.cidx, sorted newest first
    pub fn merge(&mut self, out_dir: &Path, max_shards: usize) -> io::Result<()>; // merge oldest shards beyond max_shards into one compound shard; applies tombstones (drops deleted docs); atomic swap
    // + the same read APIs as Shard, fan-out across shards (newest wins for duplicate blob+path+repo)
}
```
Tests: write 100 docs incl. unicode + 1MiB file, reopen, verify postings/content/symbols; tombstone + merge; duplicate suppression newest-wins.

## indexio-symbols — tree-sitter extraction

Public API (exact):
```rust
pub fn extract(lang: Lang, content: &[u8]) -> (Vec<SymbolRec>, Vec<CallRec>);
pub fn supported(lang: Lang) -> bool;
```
Implementation notes: tree-sitter 0.24, grammars as listed (TSX grammar covers TS+JS). Extract via queries (write `.scm`-style `Query` inline per language): definitions (fn/method/struct/class/enum/trait/impl/const/type/interface) with `name` captures; `scope` = enclosing definition name chain (walk ancestors). Calls: call-expression nodes → callee = callee identifier text (last segment for qualified names, e.g. `a.b.c` → `c` AND store full text too? Store last segment only — best-effort, documented). caller = enclosing fn/method name or "". UTF-8 safe (work on byte ranges, convert with from_utf8_lossy). NEVER persist trees. Panic-safe: wrap parse in catch_unwind, grammar load failure → return empty + `supported()==false` path. Tests: per language, a small source file with known defs/calls; assert exact names/kinds/lines.

## indexio-ingest — git + CAS + delta

CAS layout under `<data_dir>/cas/`: `<hex[0..2]>/<hex[2..]>` files, bincode-serialized `ExtractedArtifact`. Content itself is NOT in CAS (it lives in shards); CAS dedups the *extraction work* across repos/branches/forks.

Public API (exact):
```rust
pub struct Cas { dir: PathBuf }
impl Cas { pub fn open(dir:&Path)->io::Result<Self>; pub fn get(&self, id:&BlobId)->Option<ExtractedArtifact>; pub fn put(&self, id:&BlobId, art:&ExtractedArtifact)->io::Result<()>; /* write tmp + atomic rename */ pub fn stats(&self)->(u64 /*entries*/, u64 /*bytes*/); }

pub struct RepoState { pub name: String, pub path: PathBuf, pub last_commit: Option<String>, pub indexed_at: String } // JSON, stored in <data_dir>/repos/<name>.json

/// Full index of a local git repo (working tree at HEAD; use gix to read blobs from HEAD tree — NOT the filesystem, so results are commit-deterministic).
pub fn index_repo(repo_path: &Path, name: &str, data_dir: &Path, cas: &Cas) -> anyhow::Result<IndexReport>;
/// Delta re-index: diff last_commit..HEAD trees; add/modify → re-extract (CAS first); delete → tombstone; writes a delta shard; updates RepoState.
pub fn reindex_repo(name: &str, data_dir: &Path, cas: &Cas) -> anyhow::Result<IndexReport>;
pub struct IndexReport { pub repo: String, pub commit: String, pub docs_added: u64, pub docs_deleted: u64, pub docs_unchanged: u64, pub cas_hits: u64, pub cas_misses: u64, pub bytes_indexed: u64, pub elapsed_ms: u64 }
```
Rules: skip binary (NUL in first 8KiB), >4MiB, and non-source (no Lang match → index as UNKNOWN only if extension allowlist? No: skip UNKNOWN-lang files EXCEPT common text configs? Keep simple: index only files whose `Lang::from_path != UNKNOWN`; count skipped). Walk HEAD tree with gix recursively. Parallel extraction with rayon (jobs=2). After writing shard, update RepoState atomically. `index_repo` on an already-registered repo = error (use reindex). Tests: create temp git repo via git CLI (available), commit files, index; modify+commit, reindex; assert delta counts, CAS hits on second identical repo (fork simulation: two repos with same files → 2nd is ~all CAS hits).

## indexio-query — planner + execution

Query language: whitespace-separated terms; `"phrase"` = ordered literal; `/regex/` = regex; filters `repo:NAME` (substring), `lang:rust`, `path:substr`, `case:yes` (default smart-case: insensitive iff query is all-lowercase). Bare term = substring literal.

Public API (exact):
```rust
pub struct Query { pub literals: Vec<Literal>, pub regexes: Vec<String>, pub filters: Filters }
pub struct Literal { pub text: Vec<u8>, pub phrase: bool, pub case_insensitive: bool }
pub struct Filters { pub repo: Option<String>, pub lang: Option<Lang>, pub path: Option<String> }
pub fn parse(input: &str) -> Result<Query, QueryError>;
pub struct Engine { set: ShardSet, cas-hint: () }
impl Engine {
    pub fn open(data_dir: &Path) -> io::Result<Self>;          // opens <data_dir>/shards
    pub fn search(&self, q: &Query, limit: usize) -> Vec<SearchHit>;
    pub fn find_symbol(&self, name: &str, limit: usize) -> Vec<SearchHit>;
    pub fn who_calls(&self, name: &str, limit: usize) -> Vec<SearchHit>;
    pub fn stats(&self) -> EngineStats;
}
```
Planner contract (THIS IS THE MOAT — implement carefully):
1. For each literal, compute `grams_of(literal, common)`. If literal <3 bytes and no other usable constraint → brute-scan fallback over docs passing filters (bounded: scan ≤ 64MiB of content, then return partial with `truncated: true` in a side channel — add `pub truncated: bool` to a `SearchResult { hits, truncated }` returned instead of bare Vec if easier; adjust API accordingly but document).
2. Candidate docids = intersection of posting lists (galloping: seek smallest-first; use roaring bitmap for ≥3 lists? Keep: sort cursors by estimated len ascending, intersect first two via seek-merge, then filter rest).
3. Phrase: verify order/adjacency of gram positions before content load when possible (positional check on posting positions), else verify on content.
4. Apply Filters at doc-meta level (repo substring, lang, path substring) before content load.
5. Verify: load content per candidate (limit candidates to 2000 before verify), `find_literal`/`find_regex`, build SearchHit with snippet = full matching line (max 240 chars, ellipsis).
6. Rank: score = BM25-lite: `idf = ln(1 + (N - df + 0.5)/(df + 0.5))` summed per literal (tf = match count, document-length norm k1=1.2,b=0.75) + boosts: path contains literal text ×5, symbol name exact match ×2.5 (check symbol_postings), earlier line ×1. Sort desc, take limit.
`find_symbol`: exact name → symbol_postings across shards; also substring variant when exact is empty. `who_calls`: call_postings(callee). Both map docid→SearchHit (line/col from posting, snippet from content line).

Tests: build a shard via indexio-index with 20 small docs with known contents; assert planner returns exactly the right docs for: single literal, phrase, /rege(x)/, repo:/lang:/path: filters, case smartness, 2-char brute fallback, >2000-candidate bound.

## ci — binary (CLI + HTTP + MCP)

CLI (clap derive), data dir default `~/.indexio` (`--data-dir` global, env `INDEXIO_DATA_DIR`):
```
indexio index <repo-path> [--name NAME]     # default name = dir basename
indexio reindex [--repo NAME|--all]
indexio search <query> [--limit N] [--json]
indexio symbol <name>  |  indexio calls <name>
indexio compact [--max-shards N]            # default 4
indexio stats [--json]
indexio serve [--port 7717]
indexio mcp                                 # stdio JSON-RPC MCP server
indexio cas-stats
```
HTTP (axum 0.8, tokio): `GET /health` → `{"ok":true}`; `GET /stats`; `GET /search?q=..&limit=..` → `{"hits":[SearchHit...],"truncated":bool,"took_ms":u64}`; `GET /symbol/{name}`; `GET /calls/{name}`. Bind 127.0.0.1 default. Engine wrapped in `Arc`, RwLock reload on `POST /admin/reload`.

MCP over stdio, hand-rolled JSON-RPC 2.0 (serde_json only, line-delimited):
- `initialize` → `{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"indexio","version":"0.1.0"}}`; `notifications/initialized` → ignore (no response); `ping` → `{}`.
- `tools/list` → tools: `code_search`(query,limit), `code_grep`(pattern,limit), `find_symbol`(name), `who_calls`(name), `index_stats`. Each with JSON Schema for input.
- `tools/call` → `{"content":[{"type":"text","text":<JSON of results>}]}`. Errors → JSON-RPC error -32602/-32603.
- Unknown method → -32601. Batch requests: handle arrays. No SSE/HTTP transport (stdio only).
Tests: unit-test the JSON-RPC dispatcher with captured stdin/stdout strings (factor handler as `fn handle(&Engine, &str) -> String`).

## Global engineering rules
- Rust edition 2021. `unsafe` forbidden except memmap2 usage inside indexio-index (justify in comment).
- No panics on untrusted input paths: all public fns return Result/Option as specified.
- tracing for logs (no println except CLI output).
- Every crate: `#![forbid(unsafe_code)]` except indexio-index (`#![deny(unsafe_code)]` with explicit allows).
- CARGO_BUILD_JOBS=2 cargo test -p <crate> must pass before you commit.
- Commit on your branch with conventional commits (`feat(indexio-core): ...`). Do not touch other crates or the workspace Cargo.toml.
