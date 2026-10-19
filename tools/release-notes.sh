#!/usr/bin/env bash
# Print the CHANGELOG.md section of one version (the lines between its
# `## <version>` heading and the next `## ` heading). Prints nothing when
# the version has no section. Used by the release workflow.
#   tools/release-notes.sh 0.1.9
version="$1"
cd "$(dirname "$0")/.." || exit 0
[ -n "$version" ] && [ -f CHANGELOG.md ] || exit 0
awk -v v="## $version" '
  $0 == v { on = 1; next }
  on && /^## / { exit }
  on { print }
' CHANGELOG.md
