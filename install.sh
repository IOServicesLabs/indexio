#!/bin/sh
# Install the latest indexio release on Linux or macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/IOServicesLabs/indexio/main/install.sh | sh
#
# Options (environment):
#   INDEXIO_VERSION      a tag such as v0.1.0 (default: the latest release)
#   INDEXIO_INSTALL_DIR  where the binary goes (default: ~/.local/bin)
#
# The script downloads the archive for this OS and CPU from the GitHub
# release, checks its SHA-256 against the published checksum, and puts the
# `indexio` binary in the install directory. It does not need root.
set -eu

repo="IOServicesLabs/indexio"
dir="${INDEXIO_INSTALL_DIR:-$HOME/.local/bin}"

os="$(uname -s)"
case "$os" in
  Linux) os=linux ;;
  Darwin) os=macos ;;
  *) echo "install.sh: unsupported OS '$os' (use install.ps1 on Windows)" >&2; exit 1 ;;
esac
arch="$(uname -m)"
case "$arch" in
  x86_64|amd64) arch=x86_64 ;;
  arm64|aarch64) arch=aarch64 ;;
  *) echo "install.sh: unsupported CPU '$arch'" >&2; exit 1 ;;
esac

fetch() {
  if command -v curl > /dev/null 2>&1; then curl -fsSL "$1"
  elif command -v wget > /dev/null 2>&1; then wget -qO- "$1"
  else echo "install.sh: curl or wget is required" >&2; exit 1
  fi
}

tag="${INDEXIO_VERSION:-}"
if [ -z "$tag" ]; then
  tag="$(fetch "https://api.github.com/repos/$repo/releases/latest" \
    | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1)"
  [ -n "$tag" ] || { echo "install.sh: cannot find the latest release" >&2; exit 1; }
fi
version="${tag#v}"
name="indexio-${version}-${os}-${arch}"
base="https://github.com/$repo/releases/download/$tag"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "downloading $name.tar.gz"
fetch "$base/$name.tar.gz" > "$tmp/$name.tar.gz"
fetch "$base/$name.tar.gz.sha256" > "$tmp/$name.tar.gz.sha256"

want="$(awk '{print $1}' "$tmp/$name.tar.gz.sha256")"
if command -v sha256sum > /dev/null 2>&1; then have="$(sha256sum "$tmp/$name.tar.gz" | awk '{print $1}')"
else have="$(shasum -a 256 "$tmp/$name.tar.gz" | awk '{print $1}')"
fi
[ "$want" = "$have" ] || { echo "install.sh: checksum mismatch" >&2; exit 1; }

tar -xzf "$tmp/$name.tar.gz" -C "$tmp"
mkdir -p "$dir"
install -m 755 "$tmp/$name/indexio" "$dir/indexio"
echo "installed $dir/indexio ($tag)"

case ":$PATH:" in
  *":$dir:"*) ;;
  *) echo "add it to your PATH:  export PATH=\"$dir:\$PATH\"" ;;
esac
echo "next:  indexio add ~/code && indexio setup claude && indexio hook install"
