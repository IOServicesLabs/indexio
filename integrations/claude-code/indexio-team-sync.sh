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

exec >/dev/null 2>&1
cat >/dev/null  # the hook's JSON on stdin is not needed

[ -n "$INDEXIO_URL" ] || exit 0
command -v git >/dev/null && command -v curl >/dev/null || exit 0
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
root=$(git rev-parse --show-toplevel) || exit 0
cd "$root" || exit 0
gitdir=$(git rev-parse --absolute-git-dir) || exit 0
origin=$(git config --get remote.origin.url) || exit 0

# The base must be a commit the server has: the merge-base with the remote
# default branch (always synced there), else with the upstream branch.
base=""
for ref in origin/HEAD origin/main origin/master '@{upstream}'; do
  base=$(git merge-base HEAD "$ref") && [ -n "$base" ] && break
  base=""
done
[ -n "$base" ] || exit 0

# Diff the working tree, untracked files included, against the base with a
# scratch copy of the real index: its stat cache means `add -A` hashes only
# files that changed, and the developer's own index is never touched.
ix="$gitdir/indexio-team.$$.index"
patch="$gitdir/indexio-team.$$.patch"
trap 'rm -f "$ix" "$patch"' EXIT
cp "$gitdir/index" "$ix" 2>/dev/null || rm -f "$ix"
GIT_INDEX_FILE="$ix" git add -A || exit 0
GIT_INDEX_FILE="$ix" git diff --cached --binary --no-color --no-ext-diff "$base" >"$patch" || exit 0

# Unchanged since the last accepted send: nothing to do.
sent="$gitdir/indexio-team.sent"
sig="$base $(git hash-object "$patch")"
[ -f "$sent" ] && [ "$(cat "$sent")" = "$sig" ] && exit 0

code=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' -X POST \
  ${INDEXIO_TOKEN:+-H "Authorization: Bearer $INDEXIO_TOKEN"} \
  ${INDEXIO_USER:+-H "X-Indexio-User: $INDEXIO_USER"} \
  -H "X-Indexio-Repo: $origin" \
  -H "X-Indexio-Base: $base" \
  -H "Content-Type: text/x-diff" \
  --data-binary @"$patch" \
  "${INDEXIO_URL%/}/team/worktree")

# 2xx: accepted. 404: the server does not index this repo; do not retry
# until the work changes. Anything else (409 base not there yet, network)
# is retried on the next call.
case "$code" in
  2??|404) printf '%s' "$sig" >"$sent" ;;
esac
exit 0
