# SPEC-P11: one central index, every machine's working tree, every branch

Status: design. Nothing in this document is implemented yet; it records the shape the
storage already has and the smallest additions that turn one data dir into a fleet.

## 1. The problem

A central indexio server can index every repository of an organisation on a schedule
(`indexio add github:org`, `indexio sync` from cron). Two things it cannot see:

- **Work in progress on a developer's machine.** Uncommitted edits, untracked files, local
  branches. Today the MCP server on that machine re-indexes the working tree of the
  repository the session sits in before every call, so the developer's own index sees it;
  the central one does not, and the developer's index does not see the other 7,999 repos
  unless it indexed them too.
- **Branches.** A repository is indexed at one HEAD (the checkout's, or the default branch
  of a clone). A pushed feature branch is invisible until it merges.

## 2. What the storage already gives us

- Shards (`shards/*.cidx`) are immutable single files, opened by `mmap`, discovered by a
  directory listing. A shard set is any collection of such files. Deletions are tombstone
  bitmaps beside a shard, never rewrites.
- Documents are keyed by `(repo, path)` and resolved through a lazily built map
  (`Engine::locate_doc`); later shards shadow earlier ones for the same key by design
  (that is how a delta re-index works).
- The content store (`cas/v2`) and the embedding store are keyed by content hash. Two
  repos, two branches or two machines that hold the same bytes hold them once.
- The semantic plane (`vec/*.civec`, `bm25/*.cibm25`) is a set of segments discovered the
  same way, with per-segment tombstones.
- Every server on a machine shares one data dir and reopens it when the shard listing
  changes (`reload_if_needed`).

So "two data dirs" is not a new storage model; it is one shard set assembled from two
directories.

## 3. Design

### 3.1 Overlay data dirs

`INDEXIO_DATA_DIR` names the **local** data dir, as today. A new `INDEXIO_BASE_DIRS`
(colon-separated, read-only) names one or more **base** data dirs. `Engine::open` builds
its shard set from every base's `shards/` first, then the local `shards/`, so a local
document shadows a base document with the same `(repo, path)`. The same rule applies to
the vector and BM25 segments, the repo states (`repos/*.json`: local wins) and the
sessions and runs sources (local only). Tombstones of a base shard are honoured; the local
dir never writes into a base.

Publishing a base is copying its `shards/`, `vec/`, `bm25/`, `sem/` and `repos/` folders.
Because every file is immutable and named by a ULID or a model id, a client can `rsync`,
fetch by manifest, or mount the folder read-only over the network. A compaction on the
central server produces new files and parks the old ones with a `.stale` suffix that a
client ignores; a client that still maps a parked file keeps working until its next open.

The local dir holds what the machine changed: working-tree deltas of its own checkouts
(`reindex_worktree`, which the server already runs), the sessions and runs sources, and
nothing else. It stays small; `indexio compact` never touches a base.

### 3.2 Branches on the central server

A registered git repo gains an optional `refs` list in its state (default: the checkout's
HEAD, as today). For each ref the sync indexes the tree at that commit under the repo name
`<repo>@<branch>` for every branch except the default, which keeps the bare name. The delta
walks `last_commit..<ref>` per ref as it does for HEAD, so a branch that shares 99 % of its
tree with main costs 1 % of the extraction and nothing in the content store. Branches that
are deleted on the remote are tombstoned at the next sync; branches idle longer than
`INDEXIO_RETAIN_DAYS` (§P10.31) are dropped from the index, not from git.

Queries get a `ref:` filter (`ref:main`, `ref:feature/x`, `ref:*`) that the MCP tools
expose as an optional `ref` argument. Without it the default branch is searched and a
`@<branch>` repo is listed only when it holds the only hit, so the default view stays as
compact as it is now. `impact_of_diff --base origin/main` against a branch's index answers
"what does this branch touch" without a checkout.

### 3.3 The developer's machine

The MCP server on a developer's machine runs as today with two settings:

```
INDEXIO_DATA_DIR=~/.indexio            # local: working trees, sessions, runs
INDEXIO_BASE_DIRS=/mnt/indexio-base    # central: every repo, every branch, read-only
```

The session's own repo is refreshed from the working tree before each call (unchanged).
The background sync of "other repos" (§P10.21) is skipped for repos that the base holds,
because the central server keeps those current. `refresh_index` on a base-held repo
returns the base's sync time instead of re-indexing. `index_stats` reports the base's
freshness next to the local one.

Nothing leaves the machine: the base is read, the local dir is written, and the central
server never learns what a developer has open. A developer who wants their branch visible
to others pushes it; the central sync picks it up within its interval.

### 3.4 Remote instead of mounted

Where a shared mount is not available, `INDEXIO_BASE_URL` points at a central `indexio
serve` and the client keeps a **mirror** of the base under `<local>/base/`: on start and
every `INDEXIO_BASE_SYNC` minutes it fetches the base's manifest (`GET /manifest`: file
names, sizes, hashes) and downloads new files, deleting parked ones. The manifest route
applies the same bearer-token and ACL rules as `/search`, so a token that may see only
`team-*` repos receives only those shards. The mirror is then an ordinary base dir.

## 4. What does not change

The tools, the hooks, the text renderings, the usage accounting and the semantic model all
work unchanged over an overlay set: they see one engine. The sessions and runs sources stay
per machine; a central server has none.

## 5. Order of work

1. `Engine::open` over several shard roots with local-wins shadowing; the same for
   segments and repo states. Tests: a base doc shadowed by a local doc, a base tombstone
   honoured, a base compaction survived.
2. `INDEXIO_BASE_DIRS` in the server and the CLI; `index_stats` freshness per root;
   background sync skips base-held repos.
3. `refs` in the repo state, `<repo>@<branch>` naming, the `ref:` filter, branch
   retention.
4. `GET /manifest` and the mirror.
