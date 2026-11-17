#!/usr/bin/env bash
# Run one frozen autovectorization benchmark against two built toolchains.
set -euo pipefail

# Repeat a completed campaign with its exact work count. A short integration
# sample fails and requires a fresh campaign for all sides.
fixed_n=${AUTOVEC_FIXED_N:-}
if [[ -n "$fixed_n" ]]; then
	[[ "$fixed_n" =~ ^[1-9][0-9]*$ && ${#fixed_n} -le 10 ]] || { echo "invalid fixed iteration count" >&2; exit 2; }
	(( fixed_n <= 1073741824 )) || { echo "fixed iteration count is too large" >&2; exit 2; }
fi

if [[ $# != 7 ]]; then
	printf 'usage: %s RUN_ID BENCH BASE_GO CAND_GO FIXTURE BASE_COMMIT CAND_COMMIT\n' "$0" >&2
	exit 2
fi
run_id=$1
bench=$2
base_go=$(readlink -f "$3")
cand_go=$(readlink -f "$4")
fixture=$(readlink -f "$5")
base_commit=$6
cand_commit=$7
case "$run_id" in ''|*[!a-zA-Z0-9_.-]*) echo "invalid run ID" >&2; exit 2;; esac
case "$bench" in
	BenchmarkAutovecFixedPoint|BenchmarkAutovecStartFixed|BenchmarkAutovecStartVariable) ;;
	*) echo "unknown benchmark" >&2; exit 2;;
esac
for commit in "$base_commit" "$cand_commit"; do
	[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "expected full commit SHA" >&2; exit 2; }
done
for cmd in git sha256sum jq hyperfine awk sort xargs; do
	command -v "$cmd" >/dev/null || { echo "missing $cmd" >&2; exit 1; }
done
[[ $(hyperfine --version) == "hyperfine 2.0.0" ]] || { echo "hyperfine 2.0.0 required" >&2; exit 1; }
[[ -x "$base_go" && -x "$cand_go" && -f "$fixture" ]] || { echo "missing input" >&2; exit 1; }

unset GOROOT GOOS GOARCH
export GOENV=off GOFLAGS= GOTOOLCHAIN=local GOEXPERIMENT=simd CGO_ENABLED=0 GOMAXPROCS=1
base_root=$("$base_go" env GOROOT)
cand_root=$("$cand_go" env GOROOT)
base_root=$(readlink -f "$base_root")
cand_root=$(readlink -f "$cand_root")
for side in base cand; do
	root_var="${side}_root"
	commit_var="${side}_commit"
	root=${!root_var}
	expected=${!commit_var}
	actual=$(git -C "$root" rev-parse HEAD)
	[[ "$actual" == "$expected" ]] || { echo "$side checkout is $actual, expected $expected" >&2; exit 1; }
	git -C "$root" diff --quiet
	git -C "$root" diff --cached --quiet
done
base_platform=$("$base_go" env GOOS GOARCH | tr '\n' ' ')
cand_platform=$("$cand_go" env GOOS GOARCH | tr '\n' ' ')
[[ "$base_platform" == "$cand_platform" ]] || { echo "toolchain targets differ" >&2; exit 1; }
if [[ "$("$base_go" env GOARCH)" == amd64 ]]; then
	export GOAMD64=v3
fi

campaign=${AUTOVEC_OUT_ROOT:-"$cand_root/.outline/optimize/autovec"}
out="$campaign/$run_id"
if [[ -e "$out" ]]; then
	echo "run directory already exists: $out" >&2
	exit 1
fi
mkdir -p "$out" "$campaign"
record() {
	printf '%s\n' "$1" | tee -a "$campaign/log.jsonl" >> "$out/events.jsonl"
}
status=blocked
finish() {
	code=$?
	if (( code != 0 )); then
		printf '\n=== Failure diagnostics ===\n' >&2
		for log in "$out"/{base,cand}/{build,parity}.log; do
			if [[ -f "$log" ]]; then
				printf '%s\n' "$log" >&2
				tail -n 40 "$log" >&2
			fi
		done
		if [[ -n "${phase:-}" ]]; then
			for log in "$phase"/{base,cand}-gotest.txt; do
				if [[ -f "$log" ]]; then
					printf '%s\n' "$log" >&2
					tail -n 40 "$log" >&2
				fi
			done
		fi
	fi
	record "$(jq -cn --arg run "$run_id" --arg benchmark "$bench" --arg status "$status" --argjson exit_code "$code" \
		'{event:"finish",run:$run,benchmark:$benchmark,status:$status,exit_code:$exit_code}')"
}
trap finish EXIT

cp "$fixture" "$out/workload_test.go"
fixture="$out/workload_test.go"
sha256sum "$fixture" > "$out/workload.sha256"
for side in base cand; do
	root_var="${side}_root"
	go_var="${side}_go"
	root=${!root_var}
	go=${!go_var}
	mkdir "$out/$side"
	(
		cd "$root"
		git ls-files -z | sort -z | xargs -0 sha256sum
	) > "$out/$side/tracked.sha256"
	git -C "$root" status --porcelain=v1 > "$out/$side/status.txt"
	"$go" version > "$out/$side/go-version.txt"
	"$go" env -json GOOS GOARCH GOAMD64 GOARM64 GOEXPERIMENT GOROOT CGO_ENABLED > "$out/$side/go-env.json"
	"$go" tool compile -V=full > "$out/$side/compiler-version.txt"
	sha256sum "$go" "$("$go" env GOTOOLDIR)/compile" > "$out/$side/toolchain.sha256"
	# The base compiler predates the autovec phase; it rejects the debug
	# option, so only the candidate compiles with it. Everything else in the
	# workload build stays identical across both sides.
	dbg=; [[ $side == cand ]] && dbg=-d=ssa/autovec/debug=1
	"$go" test -c -gcflags="$dbg" -o "$out/$side/workload.test" "$fixture" > "$out/$side/build.log" 2>&1
	sha256sum "$out/$side/workload.test" > "$out/$side/binary.sha256"
	wc -c < "$out/$side/workload.test" > "$out/$side/binary.bytes"
	"$out/$side/workload.test" -test.run='^TestAutovecBenchmarkParity$' > "$out/$side/parity.log" 2>&1
	"$go" tool objdump -s 'benchFixedPoint|benchStartFixed|benchStartVariable' "$out/$side/workload.test" > "$out/$side/disassembly.txt"
done
uname -a > "$out/uname.txt"
if command -v lscpu >/dev/null; then lscpu > "$out/cpu.txt"; fi
hyperfine --version > "$out/hyperfine-version.txt"
sha256sum "$(command -v hyperfine)" > "$out/hyperfine.sha256"
record "$(jq -cn --arg run "$run_id" --arg benchmark "$bench" --arg base "$base_commit" --arg candidate "$cand_commit" \
	--arg workload "$(sha256sum "$fixture" | awk '{print $1}')" \
	'{event:"fingerprint",run:$run,benchmark:$benchmark,base:$base,candidate:$candidate,workload_sha256:$workload,metric:"latency",gomaxprocs:1,parity:"fixture-only",admission:"profile-and-contract-floor-review-required",candidate_scope:"one local compiler mechanism per candidate; G6 candidate-lane exception requires separate review"}')"

# Read nanoseconds per operation, rejecting missing or malformed output.
ns_per_op() {
	awk '/^Benchmark/ { for (i=1; i<=NF; i++) if ($i=="ns/op") { print $(i-1); found=1; exit } } END { if (!found) exit 1 }' "$1"
}

n=${fixed_n:-1}
if [[ -z "$fixed_n" ]]; then
	while :; do
		mkdir "$out/calibration-$n"
		ready=1
		for side in base cand; do
			"$out/$side/workload.test" -test.run='^$' -test.bench="^$bench\$" -test.benchtime="${n}x" -test.benchmem > "$out/calibration-$n/$side.txt"
			ns=$(ns_per_op "$out/calibration-$n/$side.txt")
			if ! awk -v n="$n" -v ns="$ns" 'BEGIN { exit !(n * ns >= 1.2e9) }'; then
				ready=0
			fi
		done
		[[ $ready == 1 ]] && break
		(( n < 1073741824 )) || { echo "calibration did not converge" >&2; exit 1; }
		n=$((n * 2))
	done
fi

# If any corroboration sample falls below one second of useful work, retain
# this phase as evidence, double the count, and recapture both sides.
while :; do
	phase="$out/N$n"
	mkdir "$phase"
	record "$(jq -cn --arg run "$run_id" --arg benchmark "$bench" --argjson n "$n" \
		'{event:"measurement-start",run:$run,benchmark:$benchmark,iterations:$n}')"
	for side in base cand; do
		go_var="${side}_go"
		go=${!go_var}
		binary="$out/$side/workload.test"
		printf -v quoted_binary '%q' "$binary"
		command="$quoted_binary '-test.run=^$' '-test.bench=^$bench\$' '-test.benchtime=${n}x' '-test.benchmem'"
		hyperfine --shell=bash --warmup 3 --min-runs 10 --max-runs 10 --export-json "$phase/$side.json" "$command" > "$phase/$side-hyperfine.txt"
		dbg=; [[ $side == cand ]] && dbg=-d=ssa/autovec/debug=1
		"$go" test -run='^$' -bench="^$bench\$" -benchtime="${n}x" -count=10 -benchmem \
			-gcflags="$dbg" "$fixture" > "$phase/$side-gotest.txt" 2>&1
	done
	minimum_ok=1
	for side in base cand; do
		if ! awk -v n="$n" '
			/^Benchmark/ {
				for (i=1; i<=NF; i++) if ($i=="ns/op") {
					count++
					if (n * $(i-1) < 1e9) bad=1
				}
			}
			END { exit !(count == 10 && !bad) }
		' "$phase/$side-gotest.txt"; then minimum_ok=0; fi
	done
	[[ $minimum_ok == 1 ]] && break
	if [[ -n "$fixed_n" ]]; then
		echo "integration sample below one second; recapture the complete campaign with a larger count" >&2
		exit 1
	fi
	record "$(jq -cn --arg run "$run_id" --arg benchmark "$bench" --argjson n "$n" \
		'{event:"recapture",run:$run,benchmark:$benchmark,iterations:$n,reason:"useful-work sample below one second or missing samples"}')"
	(( n < 1073741824 )) || { echo "measurement did not converge" >&2; exit 1; }
	n=$((n * 2))
done

for side in base cand; do
	go_var="${side}_go"
	go=${!go_var}
	binary="$out/$side/workload.test"
	"$binary" -test.run='^$' -test.bench="^$bench\$" -test.benchtime="${n}x" -test.cpuprofile="$phase/$side.cpu" > "$phase/$side-profile.txt"
	"$go" tool pprof -top "$binary" "$phase/$side.cpu" > "$phase/$side-pprof.txt"
	awk '/^Benchmark/ {
		for (i=1; i<=NF; i++) {
			if ($i=="B/op") { bytes++; if ($(i-1)!=0) bad=1 }
			if ($i=="allocs/op") { allocs++; if ($(i-1)!=0) bad=1 }
		}
	} END { exit !(bytes == 10 && allocs == 10 && !bad) }' "$phase/$side-gotest.txt" || {
		echo "$side had missing allocation statistics or nonzero allocations" >&2
		exit 1
	}
done

summary=$(jq -cn --slurpfile b "$phase/base.json" --slurpfile c "$phase/cand.json" \
	--arg run "$run_id" --arg benchmark "$bench" --argjson n "$n" '
	def positive_number:
		type == "number" and isfinite and . > 0;
	def nonnegative_number:
		type == "number" and isfinite and . >= 0;
	def wall_clock_summary:
		if .schema_version != 2 or .primary_metric != "time_wall_clock" then
			error("expected hyperfine schema 2 with wall-clock primary metric")
		elif (.results | type) != "array" then
			error("hyperfine results is not an array")
		elif (.results | length) != 1 then
			error("expected exactly one hyperfine result")
		else
			.results[0] as $r
			| if ($r.measurements | type) != "array" then
				error("hyperfine measurements is not an array")
			elif ($r.measurements | length) != 10 then
				error("expected exactly ten hyperfine samples")
			elif (all($r.measurements[];
				.exit_code == 0
				and .time_wall_clock.unit == "second"
				and (.time_wall_clock.value | positive_number)) | not) then
				error("invalid hyperfine wall-clock sample")
			else
				$r.summary.time_wall_clock as $s
				| if ($s | type) != "object" then
					error("missing hyperfine wall-clock summary")
				elif $s.unit != "second" or $s.count != 10
					or (($s.mean | positive_number) | not)
					or (($s.median | positive_number) | not)
					or (($s.min | positive_number) | not)
					or (($s.max | positive_number) | not)
					or (($s.stddev | nonnegative_number) | not)
					or $s.min > $s.median or $s.median > $s.max then
					error("invalid hyperfine wall-clock summary")
				else $s end
			end
		end;
	if ($b | length) != 1 or ($c | length) != 1 then
		error("expected one JSON document per side")
	else . end
	| ($b[0] | wall_clock_summary) as $base
	| ($c[0] | wall_clock_summary) as $cand
	|
	{event:"measurement",run:$run,benchmark:$benchmark,iterations:$n,shell:"bash",
	 base:{median:$base.median,stddev:$base.stddev,min:$base.min,max:$base.max},
	 candidate:{median:$cand.median,stddev:$cand.stddev,min:$cand.min,max:$cand.max},
	 ratio:($base.median/$cand.median),
	 noise_ok:($base.stddev <= 0.2*$base.median and $cand.stddev <= 0.2*$cand.median),
	 status:"measured; full parity, hot-path admission, floor, ranking and integration review remain"}')
record "$summary"

# GitHub Actions logs remain useful when the artifact archive cannot be opened.
# The complete files remain the primary evidence.
printf '\n=== Fingerprint and workload ===\n'
cat "$out/workload.sha256" "$out/uname.txt" "$out/hyperfine-version.txt" "$out/hyperfine.sha256"
if [[ -f "$out/cpu.txt" ]]; then sed -n '1,35p' "$out/cpu.txt"; fi
printf 'benchmark=%s iterations=%s shell=bash GOMAXPROCS=%s\n' "$bench" "$n" "$GOMAXPROCS"
for side in base cand; do
	printf '\n=== %s provenance ===\n' "$side"
	cat "$out/$side/go-version.txt" "$out/$side/compiler-version.txt" "$out/$side/toolchain.sha256" "$out/$side/binary.sha256"
	printf 'binary bytes: '
	cat "$out/$side/binary.bytes"
	printf 'tracked-source fingerprint: '
	sha256sum "$out/$side/tracked.sha256"
	cat "$out/$side/go-env.json"
	printf '\n=== %s autovectorization build diagnostics ===\n' "$side"
	awk '/autovectorized/ { print; found=1 } END { if (!found) print "(none)" }' "$out/$side/build.log"
	printf '\n=== %s kernel disassembly ===\n' "$side"
	sed -n '1,240p' "$out/$side/disassembly.txt"
	printf '\n=== %s hyperfine JSON ===\n' "$side"
	jq -c . "$phase/$side.json"
	printf '\n=== %s go test corroboration ===\n' "$side"
	awk '/^Benchmark/ { print }' "$phase/$side-gotest.txt"
	printf '\n=== %s CPU profile top ===\n' "$side"
	sed -n '1,35p' "$phase/$side-pprof.txt"
done
printf '\n=== Measurement summary ===\n'
printf '%s\n' "$summary"
if [[ $(printf '%s\n' "$summary" | jq -r .noise_ok) != true ]]; then
	echo "noise exceeds 20% of the median; measurement is blocked" >&2
	exit 1
fi
status=measured
