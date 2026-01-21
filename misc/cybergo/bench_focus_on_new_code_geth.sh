#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bench_focus_on_new_code_geth.sh [--trials N] [--warmup SECONDS] [--timeout SECONDS] [--workdir DIR] [--keep]

Benchmarks cybergo's --focus-on-new-code=true on a shallow clone of go-ethereum (geth).

Workflow:
  1) clone geth to /tmp
  2) add a fuzz target in package rlp (committed with an old date)
  3) warmup: fuzz baseline (no crash) for --warmup seconds
  4) commit a single-line crash marked "RECENT_BUG"
  5) build the LibAFL harness + golibafl runner
  6) run N paired trials: baseline vs git-aware; report median time-to-first-crash
EOF
}

trials=5
warmup_s=30
timeout_s=120
workdir=""
keep=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--trials)
      trials="${2:?missing N}"
      shift 2
      ;;
    -w|--warmup)
      warmup_s="${2:?missing seconds}"
      shift 2
      ;;
    -t|--timeout)
      timeout_s="${2:?missing seconds}"
      shift 2
      ;;
    --workdir)
      workdir="${2:?missing dir}"
      shift 2
      ;;
    --keep)
      keep=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1"
      usage
      exit 2
      ;;
  esac
done

CYBERGO_GO="${ROOT_DIR}/bin/go"
GOLIBAFL_DIR="${ROOT_DIR}/golibafl"

if [[ ! -x "${CYBERGO_GO}" ]]; then
  echo "missing cybergo binary: ${CYBERGO_GO}"
  echo "build it with: (cd src && ./make.bash)"
  exit 1
fi

if [[ -z "${workdir}" ]]; then
  workdir="$(mktemp -d /tmp/cybergo-bench-focus-on-new-code-geth.XXXXXX)"
fi

if [[ "${keep}" != "true" ]]; then
  trap 'rm -rf "${workdir}"' EXIT
fi

repo_dir="${workdir}/go-ethereum"
cache_dir="${workdir}/cache"
mkdir -p "${cache_dir}/gocache" "${cache_dir}/gomodcache"

export GOCACHE="${cache_dir}/gocache"
export GOMODCACHE="${cache_dir}/gomodcache"
export CGO_ENABLED=1

echo "workdir: ${workdir}"
echo "cloning geth..."
git clone --depth 1 https://github.com/ethereum/go-ethereum.git "${repo_dir}"

cd "${repo_dir}"
git config user.email "cybergo-bench@example.com"
git config user.name "cybergo-bench"

cat > rlp/cybergo_focus_new_code_fuzz_test.go <<'EOF'
package rlp

import "testing"

func FuzzCybergoFocusNewCode(f *testing.F) {
	seed, _ := EncodeToBytes(uint32(0))
	f.Add(seed)
	f.Fuzz(func(t *testing.T, data []byte) {
		CybergoFocusNewCodeTarget(data)
	})
}
EOF

cat > rlp/cybergo_focus_new_code_target.go <<'EOF'
package rlp

const cybergoFocusNewCodeMagic = uint32(0x13371337)

func CybergoFocusNewCodeTarget(data []byte) {
	var x uint32
	if err := DecodeBytes(data, &x); err != nil {
		return
	}
	if x == cybergoFocusNewCodeMagic { return } // CYBERGO_PLACEHOLDER
}
EOF

git add rlp/cybergo_focus_new_code_fuzz_test.go rlp/cybergo_focus_new_code_target.go
GIT_AUTHOR_DATE="2000-01-01T00:00:00Z" GIT_COMMITTER_DATE="2000-01-01T00:00:00Z" \
  git commit -m "cybergo bench: add fuzz target (old)"

build_harness() {
  local out_dir="${1}"
  mkdir -p "${out_dir}"
  "${CYBERGO_GO}" test ./rlp -c -fuzz=FuzzCybergoFocusNewCode --use-libafl --focus-on-new-code=false -o "${out_dir}/libharness.a"
}

build_golibafl() {
  local harness_lib="${1}"
  local out_bin="${2}"
  (cd "${GOLIBAFL_DIR}" && HARNESS_LIB="${harness_lib}" cargo build --release >/dev/null)
  cp "${GOLIBAFL_DIR}/target/release/golibafl" "${out_bin}"
  chmod +x "${out_bin}"
}

warmup_in="${workdir}/warmup/input"
warmup_out="${workdir}/warmup/out"
mkdir -p "${warmup_in}" "${warmup_out}"

harness_no_bug="${workdir}/harness-no-bug"
build_harness "${harness_no_bug}"

golibafl_no_bug="${workdir}/golibafl-no-bug"
build_golibafl "${harness_no_bug}/libharness.a" "${golibafl_no_bug}"

echo "warmup: ${warmup_s}s (baseline, no crash)"
set +e
LIBAFL_SEED_DIR="${warmup_in}" GOLIBAFL_FOCUS_ON_NEW_CODE=false \
  timeout --signal=INT --kill-after=5s "${warmup_s}" \
  "${golibafl_no_bug}" fuzz -j 0 -i "${warmup_in}" -o "${warmup_out}" \
  >"${workdir}/warmup.log" 2>&1
set -e

warmup_queue_dir="$(find "${warmup_out}/queue" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1 || true)"
if [[ -z "${warmup_queue_dir}" ]]; then
  echo "warmup did not produce a queue corpus; warmup log:"
  tail -n 200 "${workdir}/warmup.log" || true
  exit 1
fi
echo "warmup corpus: ${warmup_queue_dir}"

echo "committing RECENT_BUG (single-line crash)..."
perl -pi -e 's/if x == cybergoFocusNewCodeMagic \\{ return \\} \\/\\/ CYBERGO_PLACEHOLDER/if x == cybergoFocusNewCodeMagic { panic(\"RECENT_BUG\") } \\/\\/ RECENT_BUG/' \
  rlp/cybergo_focus_new_code_target.go
git add rlp/cybergo_focus_new_code_target.go
git commit -m "RECENT_BUG"

harness_bug="${workdir}/harness-bug"
build_harness "${harness_bug}"

golibafl_bug="${workdir}/golibafl-bug"
build_golibafl "${harness_bug}/libharness.a" "${golibafl_bug}"

git_map="${workdir}/git_recency_map.bin"
target_dir="${repo_dir}/rlp"

run_until_crash_ms() {
  local label="${1}"
  local out_dir="${2}"
  local focus="${3}"

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  local start_ns
  start_ns="$(date +%s%N)"

  if [[ "${focus}" == "true" ]]; then
    HARNESS_LIB="${harness_bug}/libharness.a" \
      GOLIBAFL_FOCUS_ON_NEW_CODE=true \
      GOLIBAFL_TARGET_DIR="${target_dir}" \
      LIBAFL_GIT_RECENCY_MAPPING_PATH="${git_map}" \
      "${golibafl_bug}" fuzz -j 0 -i "${warmup_queue_dir}" -o "${out_dir}" \
      >"${out_dir}/run.log" 2>&1 &
  else
    GOLIBAFL_FOCUS_ON_NEW_CODE=false \
      "${golibafl_bug}" fuzz -j 0 -i "${warmup_queue_dir}" -o "${out_dir}" \
      >"${out_dir}/run.log" 2>&1 &
  fi
  local pid="$!"

  local deadline=$(( $(date +%s) + timeout_s ))
  local crash_file=""
  while true; do
    if [[ -d "${out_dir}/crashes" ]]; then
      crash_file="$(find "${out_dir}/crashes" -maxdepth 1 -type f -not -name '.*' -print -quit 2>/dev/null || true)"
      if [[ -n "${crash_file}" ]]; then
        break
      fi
    fi

    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" || true
      if [[ -d "${out_dir}/crashes" ]]; then
        crash_file="$(find "${out_dir}/crashes" -maxdepth 1 -type f -not -name '.*' -print -quit 2>/dev/null || true)"
        if [[ -n "${crash_file}" ]]; then
          break
        fi
      fi
      echo "${label}: exited without a crash; log tail:"
      tail -n 100 "${out_dir}/run.log" || true
      return 1
    fi

    if [[ "$(date +%s)" -ge "${deadline}" ]]; then
      kill -INT "${pid}" 2>/dev/null || true
      sleep 1
      kill -KILL "${pid}" 2>/dev/null || true
      wait "${pid}" || true
      echo "${label}: timeout (${timeout_s}s); log tail:"
      tail -n 50 "${out_dir}/run.log" || true
      return 1
    fi

    sleep 0.05
  done

  local end_ns
  end_ns="$(date +%s%N)"
  local dur_ms=$(( (end_ns - start_ns) / 1000000 ))

  wait "${pid}" || true
  echo "${dur_ms}"
}

baseline_ms=()
gitaware_ms=()

mkdir -p "${workdir}/trials"
for i in $(seq 1 "${trials}"); do
  echo "trial ${i}/${trials}: baseline"
  b_ms="$(run_until_crash_ms "baseline_${i}" "${workdir}/trials/baseline_${i}" "false")"
  baseline_ms+=("${b_ms}")
  echo "  baseline_${i}: ${b_ms}ms"

  echo "trial ${i}/${trials}: git-aware (--focus-on-new-code=true)"
  g_ms="$(run_until_crash_ms "gitaware_${i}" "${workdir}/trials/gitaware_${i}" "true")"
  gitaware_ms+=("${g_ms}")
  echo "  gitaware_${i}: ${g_ms}ms"
done

python3 - <<PY
import statistics

baseline = list(map(int, """${baseline_ms[*]}""".split()))
gitaware = list(map(int, """${gitaware_ms[*]}""".split()))

if len(baseline) != len(gitaware):
    raise SystemExit(f"internal error: baseline trials={len(baseline)} git-aware trials={len(gitaware)}")

b_med_ms = statistics.median(baseline)
g_med_ms = statistics.median(gitaware)

print("")
print(f"baseline times (ms): {baseline}")
print(f"git-aware times (ms): {gitaware}")
print("")
print(f"baseline median: {b_med_ms/1000.0:.3f}s")
print(f"git-aware median: {g_med_ms/1000.0:.3f}s")
print(f"speedup (baseline/git-aware): {b_med_ms/g_med_ms:.2f}x")
PY
