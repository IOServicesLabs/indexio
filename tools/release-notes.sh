#!/usr/bin/env bash
# Print the CHANGELOG.md section of one version (the lines between its
# `## <version>` heading and the next `## ` heading), with hard-wrapped
# paragraphs and list items joined into single lines: a GitHub release page
# shows every newline as a line break. Prints nothing when the version has
# no section. Used by the release workflow.
#   tools/release-notes.sh 0.1.9
version="$1"
cd "$(dirname "$0")/.." || exit 0
[ -n "$version" ] && [ -f CHANGELOG.md ] || exit 0
awk -v v="## $version" '
  function flush() { if (have) print buf; have = 0; buf = "" }
  $0 == v { on = 1; next }
  on && /^## / { exit }
  !on { next }
  # a continuation: text right under text that is not a table row or heading
  have && $0 != "" && buf != "" && $0 !~ /^[ \t]*([|#]|- |\*\*)/ && buf !~ /^[ \t]*[|#]/ {
    sub(/^[ \t]+/, ""); buf = buf " " $0; next
  }
  { flush(); buf = $0; have = 1 }
  END { flush() }
' CHANGELOG.md
