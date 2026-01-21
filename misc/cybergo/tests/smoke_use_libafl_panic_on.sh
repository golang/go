#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT_DIR}/misc/cybergo/tests/smoke_use_libafl_common.sh"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
export GOCACHE="${tmp_dir}/gocache"

target_sym="cybergo.example/panic_on/myLog.myCustomError"
target_sym2="cybergo.example/panic_on/myLog.myCustomWarning"
target_sym_wildcard="cybergo.example/panic_on/myLog.myCustom*"
target_sym_nodot="test_go_panicon.(*Logger).Error"

# Invalid --panic-on should error out early.
cd "${ROOT_DIR}/test/cybergo/examples/panic_on/myLog"
set +e
CGO_ENABLED=1 timeout 30s "${ROOT_DIR}/bin/go" test -fuzz=FuzzMyCustomError --use-libafl --focus-on-new-code=false --panic-on=run 2>&1 | tee "${tmp_dir}/output-invalid-panic-on.txt"
status="${PIPESTATUS[0]}"
set -e
if [[ "${status}" -eq 0 ]]; then
  echo "expected invalid --panic-on to fail, but it exited 0"
  exit 1
fi
grep -Fq 'invalid -panic-on pattern "run"' "${tmp_dir}/output-invalid-panic-on.txt"

# With --panic-on, we expect myLog.myCustomError (an error-returning function) to be
# treated as a crash.
run_expect_crash panic_on/myLog FuzzMyCustomError 10m "${tmp_dir}/output-panic-on.txt" "--panic-on=${target_sym}"
grep -Fq "detected function: ${target_sym}()" "${tmp_dir}/output-panic-on.txt"
grep -Fq "panic-on-call: ${target_sym}" "${tmp_dir}/output-panic-on.txt"

# Multiple patterns should be supported (comma-separated).
cd "${ROOT_DIR}/test/cybergo/examples/panic_on/myLog"
set +e
timeout 30s "${ROOT_DIR}/bin/go" test -count=1 --panic-on="${target_sym},${target_sym2}" 2>&1 | tee "${tmp_dir}/output-panic-on-multi.txt"
status="${PIPESTATUS[0]}"
set -e
if [[ "${status}" -eq 0 ]]; then
  echo "expected --panic-on with multiple patterns to fail (panic), but it exited 0"
  exit 1
fi
grep -Fq "detected function: ${target_sym}()" "${tmp_dir}/output-panic-on-multi.txt"
grep -Fq "detected function: ${target_sym2}()" "${tmp_dir}/output-panic-on-multi.txt"
grep -Fq "panic-on-call: ${target_sym}" "${tmp_dir}/output-panic-on-multi.txt"

# Wildcard match should be supported (prefix*).
set +e
timeout 30s "${ROOT_DIR}/bin/go" test -count=1 --panic-on="${target_sym_wildcard}" 2>&1 | tee "${tmp_dir}/output-panic-on-wildcard.txt"
status="${PIPESTATUS[0]}"
set -e
if [[ "${status}" -eq 0 ]]; then
  echo "expected --panic-on with wildcard pattern to fail (panic), but it exited 0"
  exit 1
fi
grep -Fq "detected function: ${target_sym}()" "${tmp_dir}/output-panic-on-wildcard.txt"
grep -Fq "detected function: ${target_sym2}()" "${tmp_dir}/output-panic-on-wildcard.txt"
grep -Fq "panic-on-call: ${target_sym}" "${tmp_dir}/output-panic-on-wildcard.txt"

# Dotless module paths (e.g. "unit_test", "test_go_panicon") should still be
# instrumented, and inlining of matched targets must be prevented so the SSA
# pass can see the call site.
run_expect_crash panic_on_nodot FuzzLoggerError 10m "${tmp_dir}/output-panic-on-nodot.txt" "--panic-on=${target_sym_nodot}"
grep -Fq "detected function: ${target_sym_nodot}()" "${tmp_dir}/output-panic-on-nodot.txt"
grep -Fq "panic-on-call: ${target_sym_nodot}" "${tmp_dir}/output-panic-on-nodot.txt"

# Without --panic-on, the same harness should not crash.
cd "${ROOT_DIR}/test/cybergo/examples/panic_on/myLog"
output_no_panic_on="${tmp_dir}/output-no-panic-on.txt"

set +e
CGO_ENABLED=1 timeout 5m "${ROOT_DIR}/bin/go" test -fuzz=FuzzMyCustomError --use-libafl --focus-on-new-code=false 2>&1 | tee "${output_no_panic_on}"
status="${PIPESTATUS[0]}"
set -e

if ! grep -Fq "Loading from" "${output_no_panic_on}"; then
  echo "expected libafl fuzz run to start (\"Loading from\"), but it didn't"
  exit 1
fi

if grep -Eq "${CYBERGO_LIBAFL_CRASH_RE}" "${output_no_panic_on}"; then
  echo "expected fuzz run to NOT find crashing inputs, but it did"
  exit 1
fi

if [[ "${status}" -ne 0 && "${status}" -ne 124 ]]; then
  echo "expected fuzz run to exit 0 or timeout (124), got ${status}"
  exit 1
fi
