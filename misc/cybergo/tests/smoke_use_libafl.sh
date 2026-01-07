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

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
export GOCACHE="${tmp_dir}/gocache"

crash_re='Found [1-9][0-9]* crashing input\(s\)\. Saved to'

run_expect_crash() {
  local example_dir="${1}"
  local fuzz_name="${2}"
  local timeout_dur="${3}"
  local output_file="${4}"

  cd "${ROOT_DIR}/test/cybergo/examples/${example_dir}"
  set +e
  CGO_ENABLED=1 timeout "${timeout_dur}" "${ROOT_DIR}/bin/go" test -fuzz="${fuzz_name}" --use-libafl 2>&1 | tee "${output_file}"
  local status="${PIPESTATUS[0]}"
  set -e

  if [[ "${status}" -eq 0 ]]; then
    echo "expected fuzz run to fail (panic/crash), but it exited 0"
    exit 1
  fi

  if ! grep -Eq "${crash_re}" "${output_file}"; then
    echo "expected output to contain: Found N crashing input(s). Saved to ..."
    exit 1
  fi
}

libafl_input_dir() {
  local pkg
  pkg="$("${ROOT_DIR}/bin/go" list -f '{{.ImportPath}}')"
  echo "${GOCACHE}/fuzz/${pkg}/libafl/input"
}

run_expect_crash panic FuzzPanic 10m "${tmp_dir}/output.txt"
run_expect_crash multiargs FuzzMultiArgs 2m "${tmp_dir}/output-multiargs.txt"

cd "${ROOT_DIR}/test/cybergo/examples/multiparams"
in_dir="$(libafl_input_dir)"
mkdir -p "${in_dir}"
printf '\x06libafl\x07cybergo\x69\x7a\x01' > "${in_dir}/seed-crash"
run_expect_crash multiparams FuzzMultiParams 2m "${tmp_dir}/output-multiparams.txt"

cd "${ROOT_DIR}/test/cybergo/examples/reverse"
in_dir="$(libafl_input_dir)"
mkdir -p "${in_dir}"
printf 'FUZZING!' > "${in_dir}/seed-crash"
run_expect_crash reverse FuzzReverse 2m "${tmp_dir}/output-reverse.txt"
