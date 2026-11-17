#!/usr/bin/env bash
set -euo pipefail

# Install one pinned Linux tool into the runner's temporary directory. Verify
# the archive before extraction and publish its bin directory for later steps.
: "${RUNNER_TEMP:?Runner temporary directory is required}"
: "${RUNNER_ARCH:?Runner architecture is required}"
: "${GITHUB_PATH:?GitHub path file is required}"

if [[ $# != 1 ]]; then
  printf 'usage: %s go|node\n' "$0" >&2
  exit 2
fi

case "$RUNNER_ARCH" in
  X64)
    go_arch=amd64
    node_arch=x64
    go_sha=7226736c2339719cda28fbf3a54d71aa63bcefe376e98cef00034268dab24840
    node_sha=5c063e46bab0fbc1c17cc445d7ff129cbdcd0471be6ed83b6ab864f8143bc864
    ;;
  ARM64)
    go_arch=arm64
    node_arch=arm64
    go_sha=3392582988a273a87ab5043ee1b2c9c17a2c3c00580bf9326c47a2e0aaaeac92
    node_sha=89e1853138db51e8b82d3e98ba9c0d3f2956a28e5e55ea1b7e12d25aae92d5dd
    ;;
  *)
    printf 'unsupported CI runner architecture: %s\n' "$RUNNER_ARCH" >&2
    exit 2
    ;;
esac

case "$1" in
  go)
    release=go1.27.4
    archive="$release.linux-$go_arch.tar.gz"
    url="https://go.dev/dl/$archive"
    sha256="$go_sha"
    ;;
  node)
    release=node-v26.13.1
    archive="$release-linux-$node_arch.tar.xz"
    url="https://nodejs.org/dist/v26.13.1/$archive"
    sha256="$node_sha"
    ;;
  *)
    printf 'unsupported CI tool: %s\n' "$1" >&2
    exit 2
    ;;
esac

download="$RUNNER_TEMP/$archive"
tool_root="$RUNNER_TEMP/$release-ci"
curl --fail --silent --show-error --location --output "$download" "$url"
printf '%s  %s\n' "$sha256" "$download" | sha256sum --check --strict -
mkdir "$tool_root"
tar -xf "$download" -C "$tool_root" --strip-components=1
case "$1" in
  go)
    version=$(env -u GOROOT GOTOOLCHAIN=local "$tool_root/bin/go" version)
    test "$version" = "go version go1.27.4 linux/$go_arch"
    ;;
  node)
    version=$("$tool_root/bin/node" --version)
    test "$version" = v26.13.1
    ;;
esac
printf '%s\n' "$version"
printf '%s\n' "$tool_root/bin" >> "$GITHUB_PATH"
