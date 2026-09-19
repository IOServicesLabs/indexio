#!/usr/bin/env bash
# Build the release binary on this machine and attach it to the GitHub
# release of the version in Cargo.toml. The release workflow does this for
# every supported platform; use this script for a platform it does not
# cover or to replace one asset.
#
#   tools/release-assets.sh              # build, package, upload
#   tools/release-assets.sh --no-build   # package and upload an existing target/release build
#
# Needs `gh` logged in with write access to the repository. The build goes
# through tools/build-release.sh, which remaps local paths out of the binary.
set -euo pipefail
cd "$(dirname "$0")/.."

version="$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' Cargo.toml crates/indexio/Cargo.toml | head -1)"
tag="v${version}"
os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"
case "$os" in
  mingw*|msys*|cygwin*) os=windows; bin=indexio.exe ;;
  darwin) os=macos; bin=indexio ;;
  *) os=linux; bin=indexio ;;
esac
case "$arch" in
  arm64|aarch64) arch=aarch64 ;;
  *) arch=x86_64 ;;
esac

if [ "${1:-}" != "--no-build" ]; then
  if [ -x tools/build-release.sh ]; then bash tools/build-release.sh; else cargo build --release -p indexio; fi
fi
[ -f "target/release/${bin}" ] || { echo "target/release/${bin} missing"; exit 1; }

name="indexio-${version}-${os}-${arch}"
rm -rf "dist/${name}" && mkdir -p "dist/${name}"
cp "target/release/${bin}" README.md LICENSE NOTICE "dist/${name}/"
(
  cd dist
  if [ "$os" = windows ]; then
    rm -f "${name}.zip"
    if command -v 7z > /dev/null; then 7z a -tzip "${name}.zip" "${name}" > /dev/null; else powershell -NoProfile -Command "Compress-Archive -Path '${name}' -DestinationPath '${name}.zip' -Force"; fi
    asset="${name}.zip"
  else
    tar -czf "${name}.tar.gz" "${name}"
    asset="${name}.tar.gz"
  fi
  sha256sum "$asset" > "${asset}.sha256"
  gh release upload "$tag" "$asset" "${asset}.sha256" --clobber
  echo "attached ${asset} to ${tag}"
)
