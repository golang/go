#!/usr/bin/env bash
set -euo pipefail

# Every hyperfine invocation retains the Go benchmark's own work timer.
# Process startup and shell overhead alone cannot satisfy the one-second gate.
if [[ ${1-} == "__sample" ]]; then
  shift
  binary=$1
  regex=$2
  iterations=$3
  captures=$4
  output=$(mktemp "$captures/invocation.XXXXXX.log")
  exec "$binary" \
    -test.run='^$' \
    "-test.bench=$regex" \
    "-test.benchtime=${iterations}x" \
    -test.count=1 \
    -test.benchmem >"$output" 2>&1
fi

base=$(realpath "$1")
head=$(realpath "$2")
out=$(realpath -m "$3")
primary=$4
script=$(realpath "$0")
run_id="${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}-${HEAD_SHA:0:12}"
run="$out/$run_id"
mkdir -p "$out"
mkdir "$run"
sha256sum "$script" >"$run/harness.sha256"

declare -A roots=([base]="$base" [head]="$head")
timed_cases=(
  "BenchmarkLogosGCScan/Self"
  "BenchmarkLogosGCScan/Nil"
  "BenchmarkLogosGCScan/External"
  "BenchmarkLogosGCScanSmall/Pointer/Self"
  "BenchmarkLogosGCScanSmall/Pointer/Nil"
  "BenchmarkLogosGCScanSmall/Pointer/OtherArena"
)

found=false
for name in "${timed_cases[@]}"; do
  [[ "$name" != "$primary" ]] || found=true
done
$found || {
  printf 'Unsupported primary benchmark: %s\n' "$primary" >&2
  exit 1
}

[[ $(git -C "$base" rev-parse HEAD) == "$BASE_SHA" ]]
[[ $(git -C "$head" rev-parse HEAD) == "$HEAD_SHA" ]]
[[ $(uname -m) == x86_64 ]]

bootstrap=$(go env GOROOT)
unset GOROOT GOGC GOMEMLIMIT GODEBUG GOFLAGS
export GOENV=off GOTOOLCHAIN=local GOAMD64=v1 CGO_ENABLED=1 LC_ALL=C

jq -n \
  --arg run "$run_id" \
  --arg base "$BASE_SHA" \
  --arg head "$HEAD_SHA" \
  --arg primary "$primary" \
  '{
    run: $run, base: $base, head: $head, primary: $primary,
    metric: "fixed-work process latency",
    gomaxprocs: 1, goamd64: "v1", experiment: "logosarena",
    fullG6: "pending independent admission, floor, and parity review"
  }' >"$run/manifest.json"
jq -c . "$run/manifest.json" >>"$out/log.jsonl"
sha256sum "$script" >"$run/harness.sha256"

{
  date -u
  uname -a
  lscpu
  hyperfine --version
  jq --version
  "$bootstrap/bin/go" version
} >"$run/environment.txt"

test_manifest() {
  (
    cd "$1"
    git ls-files -z -- 'src/runtime/*logos*test.go' |
      sort -z |
      xargs -0 sha256sum
  )
}

source_manifest() {
  (
    cd "$1"
    git ls-files -z -- \
      src/runtime src/internal/runtime src/internal/abi \
      src/internal/goexperiment |
      sort -z |
      xargs -0 sha256sum
  )
}

# Both exact commits must already contain the frozen test baseline.
# Never overlay head fixtures onto an older baseline silently.
for version in base head; do
  root=${roots[$version]}
  mkdir "$run/$version"
  test_manifest "$root" >"$run/$version/frozen-tests.sha256"
  source_manifest "$root" >"$run/$version/sources.sha256"
done
test -s "$run/base/frozen-tests.sha256"
cmp "$run/base/frozen-tests.sha256" "$run/head/frozen-tests.sha256"

for path in \
  export_logos_gcscan_test.go \
  logos_gcscan_test.go \
  export_logos_gcscan_bounds_test.go \
  logos_gcscan_bounds_test.go
do
  test -f "$base/src/runtime/$path"
  test -f "$head/src/runtime/$path"
done

# Builds are serialized, and finish before any timed invocation.
for version in base head; do
  root=${roots[$version]}
  (
    cd "$root/src"
    env GOEXPERIMENT= GOMAXPROCS=2 GOFLAGS=-p=2 \
      GOROOT_BOOTSTRAP="$bootstrap" ./make.bash
  ) >"$run/$version/build.log" 2>&1
  (
    cd "$root/src"
    env GOROOT="$root" GOEXPERIMENT=logosarena \
      GOMAXPROCS=2 GOFLAGS=-p=2 \
      "$root/bin/go" test -c \
      -o "$run/$version/runtime.test" runtime
  ) >"$run/$version/test-build.log" 2>&1
  sha256sum "$run/$version/runtime.test" \
    >"$run/$version/binary.sha256"
  stat -c '%s' "$run/$version/runtime.test" \
    >"$run/$version/binary-bytes.txt"
  env GOROOT="$root" "$root/bin/go" tool objdump \
    -s '^runtime\.logosArenaScanObject$' "$run/$version/runtime.test" \
    >"$run/$version/scanner-assembly.txt"
  env GOROOT="$root" GOEXPERIMENT=logosarena \
    "$root/bin/go" env -json \
    GOOS GOARCH GOAMD64 GOEXPERIMENT CGO_ENABLED GOVERSION GOROOT \
    >"$run/$version/go-env.json"
done

export GOMAXPROCS=1 GOEXPERIMENT=logosarena
mkdir "$run/cases"

# Fail if the regexp selects an unexpected leaf or a different work count.
rows() {
  local file=$1 name=$2 n=$3
  awk -v wanted="$name" -v wanted_n="$n" '
    $1 ~ /^Benchmark/ && $4 == "ns/op" {
      name = $1
      sub(/-[0-9]+$/, "", name)
      if (name != wanted || $2 != wanted_n) {
        print "unexpected benchmark or iteration count" > "/dev/stderr"
        exit 2
      }
      bytes = allocs = -1
      for (i = 5; i <= NF; i++) {
        if ($i == "B/op") bytes = $(i-1)
        if ($i == "allocs/op") allocs = $(i-1)
      }
      if (bytes < 0 || allocs < 0) exit 3
      printf "{\"iterations\":%.0f,\"nsPerOp\":%s," \
             "\"bytesPerOp\":%s,\"allocsPerOp\":%s," \
             "\"workSeconds\":%.9f}\n",
             $2, $3, bytes, allocs, $2 * $3 / 1000000000
      seen++
    }
    END {
      if (seen == 0) exit 4
    }
  ' "$file" | jq -s .
}

minimum_work() {
  jq -s 'add | map(.workSeconds) | min' "$@"
}

increase_n() {
  local n=$1 seconds=$2
  awk -v n="$n" -v seconds="$seconds" '
    BEGIN {
      if (seconds <= 0) exit 1
      next_n = int(n * 1.35 / seconds) + 1
      if (next_n <= n) next_n = n + 1
      printf "%.0f\n", next_n
    }
  '
}

timed_pair() {
  local dir=$1 name=$2 regex=$3 n=$4
  mkdir -p "$dir/base/captures" "$dir/head/captures"
  local base_command head_command
  printf -v base_command '%q ' \
    bash "$script" __sample "$run/base/runtime.test" \
    "$regex" "$n" "$dir/base/captures"
  printf -v head_command '%q ' \
    bash "$script" __sample "$run/head/runtime.test" \
    "$regex" "$n" "$dir/head/captures"

  hyperfine --shell bash --style basic --sort command \
    --warmup 3 --min-runs 10 \
    --export-json "$dir/hyperfine.json" \
    "$base_command" "$head_command" \
    >"$dir/hyperfine.txt" 2>&1

  for version in base head; do
    : >"$dir/$version/invocations.jsonl"
    for file in "$dir/$version/captures/"*.log; do
      rows "$file" "$name" "$n" |
        jq -ec 'if length == 1 then .[0] else error("sample count") end' \
        >>"$dir/$version/invocations.jsonl"
    done
    jq -s 'if length >= 13 then . else error("missing runs") end' \
      "$dir/$version/invocations.jsonl" \
      >"$dir/$version/invocations.json"
  done
}

measure() {
  local name=$1 timed=$2
  local id=${name//\//_}
  local case_dir="$run/cases/$id"
  local regex n=1000000 accepted=false
  mkdir "$case_dir"
  regex=$(awk -v name="$name" 'BEGIN {
    count = split(name, part, "/")
    for (i = 1; i <= count; i++)
      printf "%s^%s$", i == 1 ? "" : "/", part[i]
  }')

  for attempt in 1 2 3 4 5; do
    local trial="$case_dir/attempt-$attempt-N$n"
    mkdir "$trial"

    # Calibration never contributes a reported performance sample.
    for version in base head; do
      "$run/$version/runtime.test" \
        -test.run='^$' "-test.bench=$regex" \
        "-test.benchtime=${n}x" -test.count=1 -test.benchmem \
        >"$trial/$version-calibration.log" 2>&1
      rows "$trial/$version-calibration.log" "$name" "$n" \
        >"$trial/$version-calibration.json"
      jq -e 'length == 1' "$trial/$version-calibration.json" >/dev/null
    done

    local seconds
    seconds=$(minimum_work \
      "$trial/base-calibration.json" "$trial/head-calibration.json")
    if ! jq -en --argjson s "$seconds" '$s >= 1.25' >/dev/null; then
      n=$(increase_n "$n" "$seconds")
      continue
    fi

    # One frozen N for both versions, corroboration, and integration.
    jq -n --arg name "$name" --argjson n "$n" \
      '{benchmark:$name, iterations:$n}' >"$trial/frozen-work.json"

    for version in base head; do
      local root=${roots[$version]}
      (
        cd "$root/src"
        env GOROOT="$root" GOFLAGS=-p=2 "$root/bin/go" test \
          -run='^$' "-bench=$regex" "-benchtime=${n}x" \
          -count=10 -benchmem runtime
      ) >"$trial/$version-benchmem.log" 2>&1
      rows "$trial/$version-benchmem.log" "$name" "$n" \
        >"$trial/$version-benchmem.json"
      jq -e 'length == 10' "$trial/$version-benchmem.json" >/dev/null
    done

    local minimum_files=(
      "$trial/base-benchmem.json"
      "$trial/head-benchmem.json"
    )

    if [[ "$timed" == true ]]; then
      timed_pair "$trial/comparison" "$name" "$regex" "$n"
      minimum_files+=(
        "$trial/comparison/base/invocations.json"
        "$trial/comparison/head/invocations.json"
      )
      if [[ "$name" == "$primary" ]]; then
        timed_pair "$trial/integration" "$name" "$regex" "$n"
        minimum_files+=(
          "$trial/integration/base/invocations.json"
          "$trial/integration/head/invocations.json"
        )
      fi
    fi

    seconds=$(minimum_work "${minimum_files[@]}")
    if ! jq -en --argjson s "$seconds" '$s >= 1' >/dev/null; then
      jq -n --argjson minimum "$seconds" \
        '{status:"insufficient useful work; recapture both sides",
          minimumUsefulSeconds:$minimum}' >"$trial/rejected.json"
      n=$(increase_n "$n" "$seconds")
      continue
    fi

    printf 'null\n' >"$trial/no-timing.json"
    local profile="$trial/no-timing.json"
    if [[ "$name" == "$primary" ]]; then
      # Profiles are separate invocations, outside all hyperfine samples.
      for version in base head; do
        local root=${roots[$version]}
        "$run/$version/runtime.test" \
          -test.run='^$' "-test.bench=$regex" \
          "-test.benchtime=${n}x" -test.count=1 -test.benchmem \
          "-test.cpuprofile=$trial/$version-cpu.out" \
          >"$trial/$version-profile.log" 2>&1
        env GOROOT="$root" "$root/bin/go" tool pprof \
          -top -nodecount=0 -nodefraction=0 \
          "$run/$version/runtime.test" "$trial/$version-cpu.out" \
          >"$trial/$version-profile.txt"
        awk '
          $6 == "runtime.logosArenaScanObject" {
            sub(/%$/, "", $5)
            print $5
            found = 1
          }
          END { if (!found) print "null" }
        ' "$trial/$version-profile.txt" \
          >"$trial/$version-scanner-cumulative-percent.json"
      done
      profile="$trial/profile-summary.json"
      jq -n \
        --slurpfile base "$trial/base-scanner-cumulative-percent.json" \
        --slurpfile head "$trial/head-scanner-cumulative-percent.json" \
        '{
          scannerCumulativePercent:{base:$base[0], head:$head[0]},
          admission:"pending; profile share alone does not establish the floor"
        }' >"$profile"
    fi

    local comparison="$trial/no-timing.json"
    local integration="$trial/no-timing.json"
    [[ "$timed" != true ]] ||
      comparison="$trial/comparison/hyperfine.json"
    [[ "$name" != "$primary" ]] ||
      integration="$trial/integration/hyperfine.json"

    jq -n \
      --arg name "$name" --arg run "$run_id" --arg trial "$trial" \
      --argjson n "$n" --argjson minimum "$seconds" \
      --slurpfile comparison "$comparison" \
      --slurpfile integration "$integration" \
      --slurpfile profile "$profile" \
      --slurpfile base "$trial/base-benchmem.json" \
      --slurpfile head "$trial/head-benchmem.json" \
      '{
        run:$run, benchmark:$name, trial:$trial, iterations:$n,
        minimumUsefulSeconds:$minimum,
        comparison:$comparison[0], integration:$integration[0],
        profile:$profile[0],
        benchmem:{base:$base[0], head:$head[0]},
        fullG6:"pending admission, floor, and parity review"
      }' >"$case_dir/result.json"
    jq -c . "$case_dir/result.json" >>"$out/log.jsonl"
    accepted=true
    break
  done

  $accepted || {
    printf 'Could not freeze sufficient useful work for %s\n' "$name" >&2
    return 1
  }
}

for name in "${timed_cases[@]}"; do
  measure "$name" true
done

# Additional small layouts have calibrated count10 corroboration only.
# Their ns/op samples do not replace the six hyperfine comparisons.
for layout in Interface String Slice Irregular3; do
  for mode in Self Nil OtherArena; do
    measure "BenchmarkLogosGCScanSmall/$layout/$mode" false
  done
done

for version in base head; do
  root=${roots[$version]}
  test_manifest "$root" >"$run/$version/frozen-tests-after.sha256"
  source_manifest "$root" >"$run/$version/sources-after.sha256"
  cmp "$run/$version/frozen-tests.sha256" \
      "$run/$version/frozen-tests-after.sha256"
  cmp "$run/$version/sources.sha256" \
      "$run/$version/sources-after.sha256"
done

jq -s . "$run/cases/"*/result.json >"$run/results.json"

# This gate checks timing criteria only. It does not infer safety=1,
# hot-path admission, a valid lower-bound argument, or full G6 acceptance.
jq --arg primary "$primary" '
  def valid_timing:
    (.results | length) == 2 and
    all(.results[];
      (.times | length) >= 10 and
      .median > 0 and
      (.stddev / .median) <= 0.20 and
      all(.exit_codes[]; . == 0));
  def ratio: .results[0].median / .results[1].median;
  . as $cases |
  ($cases | map(select(.benchmark == $primary)) | .[0]) as $p |
  {
    primary: $primary,
    comparisonRatio: ($p.comparison | ratio),
    integrationRatio: ($p.integration | ratio),
    primaryProfile: $p.profile,
    noisePass: all($cases[];
      (.comparison == null or (.comparison | valid_timing)) and
      (.integration == null or (.integration | valid_timing))),
    fullG6: "PENDING: independent hot-path, floor, parity, and promotion review",
    controlPolicy: "Report every control; no automatic acceptance of regressions"
  } |
  .timingGate = (
    .noisePass and .comparisonRatio >= 1.05 and .integrationRatio >= 1.05
  )
' "$run/results.json" >"$run/timing-gate.json"

{
  printf '## GC fixed-work evidence\n\n'
  printf 'Primary: %s\n\n' "$primary"
  printf 'Full G6 remains pending admission, floor, and parity review.\n\n'
  printf '| Case | N | Base median s | Head median s | Base/head | Head latency change | Base noise | Head noise |\n'
  printf '|---|---:|---:|---:|---:|---:|---:|---:|\n'
  jq -r '
    .[] | select(.comparison != null) |
    .comparison.results[0] as $b |
    .comparison.results[1] as $h |
    "| \(.benchmark) | \(.iterations) | \($b.median) | \($h.median) | \($b.median/$h.median) | \(100*($h.median/$b.median-1))% | \(100*$b.stddev/$b.median)% | \(100*$h.stddev/$h.median)% |"
  ' "$run/results.json"
  printf '\nPrimary integration, profiles, and all raw samples are in the artifact.\n\n'
  cat "$run/timing-gate.json"
} >"$run/summary.md"

if [[ -n ${GITHUB_STEP_SUMMARY:-} ]]; then
  cat "$run/summary.md" >>"$GITHUB_STEP_SUMMARY"
fi
# Keep a compact report in the job log as well as the retained artifact.
cat "$run/summary.md"
printf '\nBenchmark allocation and useful-work evidence:\n'
jq -c '.[] | {
  benchmark, iterations, minimumUsefulSeconds,
  baseBytes: ([.benchmem.base[].bytesPerOp] | unique),
  headBytes: ([.benchmem.head[].bytesPerOp] | unique),
  baseAllocs: ([.benchmem.base[].allocsPerOp] | unique),
  headAllocs: ([.benchmem.head[].allocsPerOp] | unique)
}' "$run/results.json"
printf '\nScanner assembly change:\n'
diff -u "$run/base/scanner-assembly.txt" "$run/head/scanner-assembly.txt" || test "$?" -eq 1
jq -e '.timingGate' "$run/timing-gate.json" >/dev/null
