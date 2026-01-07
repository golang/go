#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT_DIR}/misc/cybergo/tests/smoke_use_libafl_common.sh"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
export GOCACHE="${tmp_dir}/gocache"

cd "${ROOT_DIR}/test/cybergo/examples/reverse"
in_dir="$(libafl_input_dir)"
mkdir -p "${in_dir}"
printf 'FUZZING!' > "${in_dir}/seed-crash"

run_expect_crash reverse FuzzReverse 2m "${tmp_dir}/output.txt"
