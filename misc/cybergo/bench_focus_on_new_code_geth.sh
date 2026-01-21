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
  6) precompute the git recency mapping (git-aware) once
  7) run N paired trials: baseline vs git-aware; report median time-to-first-crash
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
  cleanup() {
    # Go's module cache can be made read-only; make it writable so the temp dir
    # can be removed cleanly.
    chmod -R u+w "${workdir}" 2>/dev/null || true
    rm -rf "${workdir}"
  }
  trap cleanup EXIT
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

func CybergoFocusNewCodeTarget(data []byte) {
	// Exercise some real geth code (rlp decoding), but don't gate on it:
	// the benchmark is about scheduling, not RLP validity.
	var dummy uint32
	_ = DecodeBytes(data, &dummy)

	if len(data) < 16 { return }
	if data[0] != 0x13 { return }
	if data[1] != 0x37 { return }
	if data[2] != 0x13 { return }
	if data[3] != 0x37 { return }
	if data[4] != 0x13 { return }
	if data[5] != 0x37 { return }
	if data[6] != 0xde { return }
	if data[7] != 0xad { return }
	if data[8] != 0xbe { return }
	if data[9] != 0xef { return }
	if data[10] != 0xca { return }
	if data[11] != 0xfe { return }
	if data[12] != 0xba { return }
	if data[13] != 0xbe { return }
	if data[14] != 0x42 { return }
	if data[15] == 0x99 { return } // CYBERGO_PLACEHOLDER
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
python3 - <<'PY'
from pathlib import Path

p = Path("rlp/cybergo_focus_new_code_target.go")
text = p.read_text()
old = "if data[15] == 0x99 { return } // CYBERGO_PLACEHOLDER\n"
new = "if data[15] == 0x99 { panic(\"RECENT_BUG\") } // RECENT_BUG\n"
if old not in text:
    raise SystemExit("RECENT_BUG placeholder line not found")
p.write_text(text.replace(old, new))
PY
git add rlp/cybergo_focus_new_code_target.go
git commit -m "RECENT_BUG"

harness_bug="${workdir}/harness-bug"
build_harness "${harness_bug}"

golibafl_bug="${workdir}/golibafl-bug"
build_golibafl "${harness_bug}/libharness.a" "${golibafl_bug}"

git_map="${workdir}/git_recency_map.bin"
target_dir="${repo_dir}/rlp"

precompute_git_recency_map() {
  if [[ -s "${git_map}" ]]; then
    return 0
  fi

  echo "precomputing git recency mapping (one-time; not counted in trials)..."
  local out_dir="${workdir}/git_map_pregen"
  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  HARNESS_LIB="${harness_bug}/libharness.a" \
    GOLIBAFL_FOCUS_ON_NEW_CODE=true \
    GOLIBAFL_TARGET_DIR="${target_dir}" \
    LIBAFL_GIT_RECENCY_MAPPING_PATH="${git_map}" \
    setsid "${golibafl_bug}" fuzz -j 0 -i "${warmup_queue_dir}" -o "${out_dir}" \
    >"${out_dir}/run.log" 2>&1 &
  local pid="$!"

  local deadline=$(( $(date +%s) + 600 )) # 10 minutes should be plenty on a local clone.
  while [[ ! -s "${git_map}" ]]; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" || true
      echo "failed to generate git recency mapping; log tail:" >&2
      tail -n 200 "${out_dir}/run.log" >&2 || true
      exit 1
    fi
    if [[ "$(date +%s)" -ge "${deadline}" ]]; then
      kill -INT -- "-${pid}" 2>/dev/null || true
      sleep 1
      kill -KILL -- "-${pid}" 2>/dev/null || true
      wait "${pid}" || true
      echo "timed out generating git recency mapping; log tail:" >&2
      tail -n 200 "${out_dir}/run.log" >&2 || true
      exit 1
    fi
    sleep 0.1
  done

  # Wait for the file size to stabilize (avoid killing while it's still writing).
  local last_size=0
  local stable=0
  while [[ "${stable}" -lt 5 ]]; do
    local size
    size="$(wc -c < "${git_map}" 2>/dev/null || echo 0)"
    if [[ "${size}" -ge 16 && "${size}" -eq "${last_size}" ]]; then
      stable=$(( stable + 1 ))
    else
      stable=0
      last_size="${size}"
    fi
    sleep 0.1
  done

  kill -INT "${pid}" 2>/dev/null || true
  for _ in {1..50}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    kill -INT -- "-${pid}" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  wait "${pid}" || true

  echo "git recency map: ${git_map} (${last_size} bytes)"
}

run_until_crash_ms() {
  local label="${1}"
  local out_dir="${2}"
  local focus="${3}"

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  local start_ns
  start_ns="$(date +%s%N)"

  # Run the fuzzer in its own process group so timeouts can reliably kill the
  # whole tree (broker/clients), not just the parent PID.
  if [[ "${focus}" == "true" ]]; then
    HARNESS_LIB="${harness_bug}/libharness.a" \
      GOLIBAFL_FOCUS_ON_NEW_CODE=true \
      GOLIBAFL_TARGET_DIR="${target_dir}" \
      LIBAFL_GIT_RECENCY_MAPPING_PATH="${git_map}" \
      setsid "${golibafl_bug}" fuzz -j 0 -i "${warmup_queue_dir}" -o "${out_dir}" \
      >"${out_dir}/run.log" 2>&1 &
  else
    GOLIBAFL_FOCUS_ON_NEW_CODE=false \
      setsid "${golibafl_bug}" fuzz -j 0 -i "${warmup_queue_dir}" -o "${out_dir}" \
      >"${out_dir}/run.log" 2>&1 &
  fi
  local pid="$!"

  local deadline=$(( $(date +%s) + timeout_s ))
  local crash_file=""
  local status="running"
  while true; do
    if [[ -d "${out_dir}/crashes" ]]; then
      crash_file="$(find "${out_dir}/crashes" -maxdepth 1 -type f -not -name '.*' -print -quit 2>/dev/null || true)"
      if [[ -n "${crash_file}" ]]; then
        status="crash"
        break
      fi
    fi

    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" || true
      if [[ -d "${out_dir}/crashes" ]]; then
        crash_file="$(find "${out_dir}/crashes" -maxdepth 1 -type f -not -name '.*' -print -quit 2>/dev/null || true)"
        if [[ -n "${crash_file}" ]]; then
          status="crash"
          break
        fi
      fi
      status="error"
      break
    fi

    if [[ "$(date +%s)" -ge "${deadline}" ]]; then
      status="timeout"
      break
    fi

    sleep 0.05
  done

  local end_ns
  end_ns="$(date +%s%N)"
  local dur_ms=$(( (end_ns - start_ns) / 1000000 ))

  if [[ "${status}" == "crash" ]]; then
    # Prefer a clean stop so golibafl can finalize crash artifacts, but do not
    # hang forever if it keeps running.
    local stop_deadline=$(( $(date +%s) + 10 ))
    while kill -0 "${pid}" 2>/dev/null && [[ "$(date +%s)" -lt "${stop_deadline}" ]]; do
      sleep 0.05
    done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -INT "${pid}" 2>/dev/null || true
      sleep 1
      if kill -0 "${pid}" 2>/dev/null; then
        kill -INT -- "-${pid}" 2>/dev/null || true
        sleep 1
        kill -KILL -- "-${pid}" 2>/dev/null || true
      fi
    fi
    wait "${pid}" || true
    echo "crash ${dur_ms}"
    return 0
  fi

  if [[ "${status}" == "timeout" ]]; then
    kill -INT "${pid}" 2>/dev/null || true
    for _ in {1..50}; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -INT -- "-${pid}" 2>/dev/null || true
      sleep 1
      kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
    wait "${pid}" || true
    echo "${label}: timeout (${timeout_s}s); log tail:" >&2
    tail -n 50 "${out_dir}/run.log" >&2 || true
    echo "timeout $(( timeout_s * 1000 ))"
    return 0
  fi

  wait "${pid}" || true
  echo "${label}: exited without a crash; log tail:" >&2
  tail -n 100 "${out_dir}/run.log" >&2 || true
  echo "error ${dur_ms}"
  return 0
}

baseline_status=()
baseline_ms=()
gitaware_status=()
gitaware_ms=()

precompute_git_recency_map

mkdir -p "${workdir}/trials"
for i in $(seq 1 "${trials}"); do
  echo "trial ${i}/${trials}: baseline"
  read -r b_status b_ms < <(run_until_crash_ms "baseline_${i}" "${workdir}/trials/baseline_${i}" "false")
  baseline_status+=("${b_status}")
  baseline_ms+=("${b_ms}")
  echo "  baseline_${i}: ${b_status} (${b_ms}ms)"

  echo "trial ${i}/${trials}: git-aware (--focus-on-new-code=true)"
  read -r g_status g_ms < <(run_until_crash_ms "gitaware_${i}" "${workdir}/trials/gitaware_${i}" "true")
  gitaware_status+=("${g_status}")
  gitaware_ms+=("${g_ms}")
  echo "  gitaware_${i}: ${g_status} (${g_ms}ms)"
done

python3 - <<PY
import statistics

timeout_ms = ${timeout_s} * 1000

baseline_status = """${baseline_status[*]}""".split()
baseline_ms = list(map(int, """${baseline_ms[*]}""".split()))
gitaware_status = """${gitaware_status[*]}""".split()
gitaware_ms = list(map(int, """${gitaware_ms[*]}""".split()))

if len(baseline_status) != len(baseline_ms):
    raise SystemExit(f"internal error: baseline status={len(baseline_status)} ms={len(baseline_ms)}")
if len(gitaware_status) != len(gitaware_ms):
    raise SystemExit(f"internal error: git-aware status={len(gitaware_status)} ms={len(gitaware_ms)}")
if len(baseline_ms) != len(gitaware_ms):
    raise SystemExit(f"internal error: baseline trials={len(baseline_ms)} git-aware trials={len(gitaware_ms)}")

def summarize(label: str, st: list[str], ms: list[int]):
    pairs = list(zip(st, ms))
    crashes = [m for (s, m) in pairs if s == "crash"]
    timeouts = sum(1 for (s, _m) in pairs if s == "timeout")
    errors = sum(1 for (s, _m) in pairs if s == "error")
    ok = len(crashes)

    print(f"{label} results:")
    for i, (s, m) in enumerate(pairs, 1):
        if s == "timeout":
            print(f"  trial {i}: timeout ({timeout_ms}ms)")
        elif s == "error":
            print(f"  trial {i}: error ({m}ms)")
        else:
            print(f"  trial {i}: crash ({m}ms)")

    med = statistics.median(crashes) if crashes else None
    print(f"{label} crashes: {ok}/{len(pairs)} (timeouts={timeouts}, errors={errors})")
    if med is None:
        print(f"{label} median (crashes only): N/A")
    else:
        print(f"{label} median (crashes only): {med/1000.0:.3f}s")
    print("")
    return med

b_med_ms = summarize("baseline", baseline_status, baseline_ms)
g_med_ms = summarize("git-aware", gitaware_status, gitaware_ms)

print("")
if b_med_ms is not None and g_med_ms is not None:
    print(f"speedup (baseline/git-aware): {b_med_ms/g_med_ms:.2f}x")
else:
    print("speedup (baseline/git-aware): N/A (need at least one crash in each mode)")
PY
