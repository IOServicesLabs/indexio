#!/usr/bin/env bash
# cargo build --release with two extras:
#  - the home, cargo and rustup directories are remapped, so the binary does
#    not embed this machine's user name in panic locations and debug paths;
#  - retries: rustc on Windows occasionally dies with STATUS_ACCESS_VIOLATION
#    during the thin-LTO link, and a plain retry succeeds.
cd "$(dirname "$0")/.." || exit 1

remap=""
for d in "${HOME:-}" "${CARGO_HOME:-$HOME/.cargo}" "${RUSTUP_HOME:-$HOME/.rustup}"; do
  [ -n "$d" ] || continue
  remap="$remap --remap-path-prefix=$d=/build"
  if command -v cygpath > /dev/null 2>&1; then
    remap="$remap --remap-path-prefix=$(cygpath -w "$d")=C:/build"
  fi
done
export RUSTFLAGS="${RUSTFLAGS:-}$remap"

for i in 1 2 3 4 5 6 7 8; do
  out=$(cargo build --release "$@" 2>&1)
  if echo "$out" | grep -q "Finished"; then echo "$out" | grep -E "^warning|Finished" | head -5; exit 0; fi
  if echo "$out" | grep -qE "ACCESS_VIOLATION|STACK_BUFFER_OVERRUN|0xc0000409|0xc0000005"; then echo "attempt $i: rustc access violation, retrying"; continue; fi
  echo "$out" | grep -E "^(error|warning)" -A8 | head -60; exit 1
done
echo "build failed after 8 attempts"; exit 1
