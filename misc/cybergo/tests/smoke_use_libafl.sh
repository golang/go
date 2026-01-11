#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

bootstrap_goroot="${GOROOT_BOOTSTRAP:-}"
if [[ -z "${bootstrap_goroot}" ]]; then
  if command -v go >/dev/null 2>&1; then
    bootstrap_goroot="$(go env GOROOT)"
  else
    bootstrap_ver="go1.25.5"
    os="$(uname -s | tr '[:upper:]' '[:lower:]')"
    arch="$(uname -m)"
    case "${arch}" in
      x86_64) arch="amd64" ;;
      aarch64 | arm64) arch="arm64" ;;
      *) echo "unsupported arch for auto-bootstrap: ${arch}" >&2; exit 1 ;;
    esac
    case "${os}" in
      linux | darwin) ;;
      *) echo "unsupported os for auto-bootstrap: ${os}" >&2; exit 1 ;;
    esac
    cache_root="${XDG_CACHE_HOME:-$HOME/.cache}/cybergo/go-bootstrap/${bootstrap_ver}"
    bootstrap_goroot="${cache_root}/go"
    if [[ ! -x "${bootstrap_goroot}/bin/go" ]]; then
      mkdir -p "${cache_root}"
      archive="${bootstrap_ver}.${os}-${arch}.tar.gz"
      curl -fsSL -o "${cache_root}/${archive}" "https://go.dev/dl/${archive}"
      tar -C "${cache_root}" -xzf "${cache_root}/${archive}"
    fi
  fi
fi

cd "${ROOT_DIR}/src"
GOROOT_BOOTSTRAP="${bootstrap_goroot}" ./make.bash

bash "${ROOT_DIR}/misc/cybergo/tests/smoke_use_libafl_panic.sh"
bash "${ROOT_DIR}/misc/cybergo/tests/smoke_use_libafl_overflow.sh"
bash "${ROOT_DIR}/misc/cybergo/tests/smoke_use_libafl_multiargs.sh"
bash "${ROOT_DIR}/misc/cybergo/tests/smoke_use_libafl_multiparams.sh"
bash "${ROOT_DIR}/misc/cybergo/tests/smoke_use_libafl_reverse.sh"
bash "${ROOT_DIR}/misc/cybergo/tests/smoke_use_libafl_config.sh"
