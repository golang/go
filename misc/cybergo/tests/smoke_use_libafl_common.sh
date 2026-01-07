#!/usr/bin/env bash
#
# Common helpers for the cybergo --use-libafl smoke tests.
#
# Expected environment:
# - ROOT_DIR: repo root
# - GOCACHE: set by the caller (typically to a temp dir)

CYBERGO_LIBAFL_CRASH_RE='Found [1-9][0-9]* crashing input\(s\)\. Saved to'

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

  if ! grep -Eq "${CYBERGO_LIBAFL_CRASH_RE}" "${output_file}"; then
    echo "expected output to contain: Found N crashing input(s). Saved to ..."
    exit 1
  fi
}

libafl_input_dir() {
  local pkg
  pkg="$("${ROOT_DIR}/bin/go" list -f '{{.ImportPath}}')"
  echo "${GOCACHE}/fuzz/${pkg}/libafl/input"
}
