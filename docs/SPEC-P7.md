# SPEC-P7 — Sources: one-command onboarding

Addendum to SPEC/SPEC-P2..P6. Motivation: "make it dead simple for someone to index local
repos (a whole folder and all subfolders) and remote repos in Azure DevOps or GitHub."
Before P7 a user had to know `indexio index` per repo, `indexio org-sync` for GitHub orgs only, and
`indexio embed`. After P7 there are two commands: `indexio add <source>` and `indexio sync`.

## 1. Sources (indexio-ingest `sources` module)

A source expands to repositories:

| `indexio add …` | kind | expands to |
|---|---|---|
| `~/code`, `C:\src`, `.` | `local_dir` | every git repo below the folder at any depth (a repo is a dir with `.git`; the walk does not descend into repos, hidden dirs, `node_modules`, `target`, `dist`, `build`, `vendor`, venvs, …). A folder containing no repo is indexed as one **plain tree** |
| `github:ORG` / `github:USER` | `github_owner` | all repos (`/orgs/{x}/repos`, falling back to `/users/{x}/repos`), paginated |
| `github:OWNER/REPO`, `https://github.com/OWNER/REPO(.git)` | `github_repo` | one repo |
| `azdo:ORG/PROJECT`, `https://dev.azure.com/ORG/PROJECT`, `https://ORG.visualstudio.com/PROJECT` | `azdo_project` | all repos of the project (`_apis/git/repositories?api-version=7.1`); disabled repos skipped unless `--include-archived` |
| `azdo:ORG/PROJECT/REPO`, `…/_git/REPO` | `azdo_repo` | one repo |
| any other `https://…`, `git@…`, `ssh://…` | `git_url` | one repo, cloned as-is |

Sources persist in `<data_dir>/sources.json` (`Vec<Source>`: kind, normalized spec, optional
`dest`, `include_forks`, `include_archived`, `shallow`). `add_source` updates an existing
(kind, spec) in place; `remove_source` accepts the spec as typed or normalized.

Remote clones land in `<data_dir>/remotes/<host>/<owner>/<repo>` (`--dest DIR` overrides the
root), shallow by default. Existing clones are `git pull --ff-only`.

**Credentials** come from the environment only: `GITHUB_TOKEN` / `GH_TOKEN`, and
`AZDO_TOKEN` / `AZURE_DEVOPS_EXT_PAT` / `SYSTEM_ACCESSTOKEN`. They are sent to the listing
APIs as `Authorization` headers and to git as `-c http.extraheader="AUTHORIZATION: Basic …"`
(GitHub: `x-access-token:<token>`, Azure DevOps: `:<PAT>`), never embedded in a URL, so
they never appear in process lists, logs, or `.git/config`.

## 2. Plain trees (folders without git)

`index_repo` on a directory that is not a git repository now indexes the filesystem tree
(`index_dir`): files with a known language, ≤ 4 MiB, non-binary, skipping the directory
list above. `RepoState.plain = true`, `last_commit = None`. `reindex_repo` dispatches to a
content-hash delta (`reindex_dir`): a file whose blake3 equals the indexed doc's blob is
unchanged; changed/added files go into a new shard; deleted and superseded docs are
tombstoned. Same report shape as git deltas.

## 3. Repo naming

`repo_name_for(data_dir, path)`: a path already registered keeps its name; otherwise the
directory basename, then `<parent>-<basename>` when another path owns it, then `-2`, `-3`, …
(non `[A-Za-z0-9-_.]` characters become `-`). Names are what `repo:` filters, `REPO:PATH`
targets and ACLs use.

## 4. Sync

`sync_source` (per source) and `sync_all`:

1. every source in order — local: discover + index/re-index; remote: list → filter →
   clone/pull → index/re-index; per-repo failures are collected, never fatal; a listing
   failure fails that source only;
2. every registered repo no source touched (e.g. added with plain `indexio index`) is delta
   re-indexed;
3. the CLI then runs `embed --all` unless `--no-embed`.

## 5. CLI

```
indexio add <SOURCE> [--dest DIR] [--include-forks] [--include-archived] [--full-clone]
                [--limit N] [--no-sync] [--no-embed]
indexio sync [--no-embed] [--limit N]        # the only thing to cron
indexio sources                              # sources + registered repos (path, plain?, indexed at)
indexio remove <SOURCE>
indexio index <DIR>                          # now also accepts a folder without git
```

`indexio org-sync` remains as a thin legacy alias of `indexio add github:ORG --dest …`.

## 6. Tests

parse_source (github/azdo/URL/visualstudio/ssh/paths/errors), sources.json add/update/
remove round trip, discover_repos (depth, nested repo, node_modules, hidden), unique repo
naming, plain-tree index + hash delta (+ no-op, + sync_all coverage of untouched repos),
Azure DevOps listing against a mock server (URL encoding, Basic auth header, disabled
repos skipped, clone dest/header/shallow), GitHub owner listing 404-fallback, clone auth
header host rules, CLI parse, and an end-to-end `indexio add <folder>` / `indexio sync` / `indexio
sources` / `indexio remove` run with real git repos.
