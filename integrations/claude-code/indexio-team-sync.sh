#!/usr/bin/env bash
# indexio team sync: a Claude Code PreToolUse hook that sends this repo's
# uncommitted work to a shared indexio server right before each indexio
# tool call, so the answer includes the developer's own working tree.
# Needs only git and curl. Prints nothing and always exits 0: a hook's
# output would land in the agent's context, and a failing hook must never
# block the tool call.
#
#   INDEXIO_URL    server base URL, e.g. https://indexio.example.com
#   INDEXIO_TOKEN  bearer token (an ACL entry with a "user")
#   INDEXIO_USER   only with a shared token: the name to act as
#
# It runs before every indexio call, so the common case (nothing changed
# since the last send) is two git calls: one rev-parse and one `add -A` on
# a scratch index that persists between runs.

exec >/dev/null 2>&1
cat >/dev/null  # the hook's JSON on stdin is not needed

[ -n "$INDEXIO_URL" ] || exit 0
command -v git >/dev/null && command -v curl >/dev/null || exit 0
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
{ read -r root; read -r gitdir; read -r head; } < <(git rev-parse --show-toplevel --absolute-git-dir HEAD)
[ -n "$root" ] && [ -n "$gitdir" ] && [ -n "$head" ] || exit 0
cd "$root" || exit 0

# The base must be a commit the server has: the merge-base with the remote
# default branch (always synced there), else with the upstream branch. It
# only moves with HEAD or the remote refs, so it is cached with the origin
# URL and recomputed when HEAD differs or a ref file is newer than the cache.
state="$gitdir/indexio-team.base"
fresh=0
if [ -f "$state" ]; then
  { read -r c_head; read -r base; read -r origin; } <"$state"
  [ "$c_head" = "$head" ] && [ -n "$base" ] && [ -n "$origin" ] && fresh=1
  for f in "$gitdir/packed-refs" "$gitdir/config" "$gitdir/refs/remotes/origin/HEAD" \
           "$gitdir/refs/remotes/origin/main" "$gitdir/refs/remotes/origin/master"; do
    [ "$f" -nt "$state" ] && fresh=0
  done
fi
if [ "$fresh" = 0 ]; then
  origin=$(git config --get remote.origin.url) || exit 0
  base=""
  for ref in origin/HEAD origin/main origin/master '@{upstream}'; do
    base=$(git merge-base HEAD "$ref") && [ -n "$base" ] && break
    base=""
  done
  [ -n "$base" ] || exit 0
  printf '%s\n%s\n%s\n' "$head" "$base" "$origin" >"$state"
fi

# Stage the working tree, untracked files included, into a scratch index
# kept between runs: its stat cache means `add -A` hashes only what changed
# since the last run, and the developer's own index is never touched. It
# starts over from the real index whenever that one moves (add, commit,
# checkout). A concurrent run holding it falls back to a private copy.
ix="$gitdir/indexio-team.index"
patch="$gitdir/indexio-team.$$.patch"
gz="$patch.gz"
trap 'rm -f "$patch" "$gz" "$gitdir/indexio-team.$$.index"' EXIT
if [ ! -f "$ix" ] || [ "$gitdir/index" -nt "$ix" ]; then
  cp "$gitdir/index" "$ix" 2>/dev/null || rm -f "$ix"
fi
if ! GIT_INDEX_FILE="$ix" git add -A; then
  ix="$gitdir/indexio-team.$$.index"
  cp "$gitdir/index" "$ix" 2>/dev/null || rm -f "$ix"
  GIT_INDEX_FILE="$ix" git add -A || exit 0
fi

# Unchanged since the last accepted send (same base, same staged tree):
# nothing to diff or send.
sent="$gitdir/indexio-team.sent"
ixsum=$(cksum <"$ix")
if [ -f "$sent" ]; then
  { read -r s_base; read -r s_ix; read -r _; } <"$sent"
  [ "$s_base" = "$base" ] && [ "$s_ix" = "$ixsum" ] && exit 0
fi
GIT_INDEX_FILE="$ix" git diff --cached --binary --no-color --no-ext-diff "$base" >"$patch" || exit 0
patchsum=$(cksum <"$patch")
if [ -f "$sent" ]; then
  # the tree moved but the patch did not (an edit undone): record, no send
  { read -r s_base; read -r _; read -r s_patch; } <"$sent"
  if [ "$s_base" = "$base" ] && [ "$s_patch" = "$patchsum" ]; then
    printf '%s\n%s\n%s\n' "$base" "$ixsum" "$patchsum" >"$sent"
    exit 0
  fi
fi

post() { # $1 body file, $2 extra curl args
  curl -sS -m 10 -o /dev/null -w '%{http_code}' -X POST \
    ${INDEXIO_TOKEN:+-H "Authorization: Bearer $INDEXIO_TOKEN"} \
    ${INDEXIO_USER:+-H "X-Indexio-User: $INDEXIO_USER"} \
    -H "X-Indexio-Repo: $origin" \
    -H "X-Indexio-Base: $base" \
    -H "Content-Type: text/x-diff" \
    $2 --data-binary @"$1" \
    "${INDEXIO_URL%/}/team/worktree"
}
# Large patches go gzipped; a server that cannot take that (4xx other than
# 404/409) gets the plain body.
code=""
if [ "$(wc -c <"$patch")" -gt 32768 ] && command -v gzip >/dev/null && gzip -c "$patch" >"$gz"; then
  code=$(post "$gz" "-H Content-Encoding:gzip")
  case "$code" in 4??) [ "$code" = 404 ] || [ "$code" = 409 ] || code="" ;; esac
fi
[ -n "$code" ] || code=$(post "$patch" "")

# 2xx: accepted. 404: the server does not index this repo; do not retry
# until the work changes. Anything else (409 base not there yet, network)
# is retried on the next call.
case "$code" in
  2??|404) printf '%s\n%s\n%s\n' "$base" "$ixsum" "$patchsum" >"$sent" ;;
esac
exit 0
