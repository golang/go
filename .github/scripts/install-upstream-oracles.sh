#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_ENV:?GitHub environment file is required}"
: "${RUNNER_TEMP:?Runner temporary directory is required}"

printf 'UPROOT1274=%s\n' "$(go env GOROOT)" >> "$GITHUB_ENV"
while read -r version sha256 root; do
  archive="$RUNNER_TEMP/go$version.linux-amd64.tar.gz"
  curl --fail --silent --show-error --location --output "$archive" "https://go.dev/dl/go$version.linux-amd64.tar.gz"
  printf '%s  %s\n' "$sha256" "$archive" | sha256sum --check --strict -
  mkdir "$RUNNER_TEMP/go$version"
  tar -xzf "$archive" -C "$RUNNER_TEMP/go$version" --strip-components=1
  printf '%s=%s\n' "$root" "$RUNNER_TEMP/go$version" >> "$GITHUB_ENV"
done <<'PINS'
1.27.2 ecbadb99091a3f46e31f5f934b068b1864eafa7995211b39eaddf76996045fe5 UPROOT1272
1.27.3 65efbc80fdc1fa2cd2631cad5b481777a68e8725f34e89f50fc6ac714e379b40 UPROOT1273
PINS
