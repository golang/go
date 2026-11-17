# Scalar optimization measurements

B means baseline and C means candidate. Ratios are **baseline median / candidate median**. Above 1 favors the candidate; below 1 favors the baseline. Each paired comparison uses one compiler and one source snapshot, with `-d=logosopt=0` versus `-d=logosopt=1`. Compare variants within a campaign. Compiler and fixture hashes can change across campaigns.

Hyperfine measures the complete fixed-work test-binary invocation. Go values are medians of ten fixed-iteration samples. PackedDescriptors, PackedHeaders, and CorrelatedDescriptors process 1,024 records per operation; their ns/op values are per batch.

## Final campaign results

| Campaign | Runtime benchmark | Fixed N | HF median seconds B / C | HF ratio | Go median ns/op B / C | Go ratio |
|---|---|---:|---:|---:|---:|---:|
| F | BenchmarkMaskedArithmetic | 1,271,186,441 | 1.552385 / 1.635747 | 0.949037 | 1.233 / 1.2135 | 1.016069 |
| F | BenchmarkRotatedField | 1,258,389,262 | 1.540492 / 1.628073 | 0.946206 | 1.2935 / 1.288 | 1.004270 |
| F | BenchmarkCorrelated | 1,129,518,073 | 1.692463 / 1.403335 | 1.206029 | 1.444 / 1.446 | 0.998617 |
| F | BenchmarkFixedPointDispatch | 359,023,457 | 2.303594 / 1.388902 | 1.658571 | 6.439 / 3.944 | 1.632606 |
| F | BenchmarkPackedDescriptors | 2,592,465 | 1.719035 / 1.611389 | 1.066803 | 673.75 / 580.25 | 1.161137 |
| F | BenchmarkPackedHeaders | 2,753,810 | 2.012385 / 1.317048 | 1.527951 | 708.75 / 404.9 | 1.750432 |
| F | BenchmarkCorrelatedDescriptors | 4,342,791 | 2.560183 / 1.342733 | 1.906696 | 563.8 / 319.6 | 1.764080 |

| Campaign | Compile target | Compiles per batch | Median ms/compile B / C | HF ratio | Peak RSS KiB B / C |
|---|---|---:|---:|---:|---:|
| F | go/parser | 5 | 367.626000 / 359.983959 | 1.021229 | 71772 / 73440 |
| F | encoding/json | 13 | 145.869751 / 143.761574 | 1.014664 | 62244 / 59732 |

These package medians do not establish a broad compiler-speed improvement. Read them with the raw distributions and single-run RSS scope.

### Retention decision and scope

Retain the selected implementation on the strength of four loop kernels: `BenchmarkFixedPointDispatch`, `BenchmarkPackedDescriptors`, `BenchmarkPackedHeaders`, and `BenchmarkCorrelatedDescriptors`. Each has a baseline/candidate median ratio of at least **1.05 under both Hyperfine and the separately collected Go benchmark samples**. The final packed-header comparison favors the candidate under both methods; its initial regression is retained below.

The three tiny-call cases, `BenchmarkMaskedArithmetic`, `BenchmarkRotatedField`, and `BenchmarkCorrelated`, remain inconclusive speed claims because the timers disagree in direction. Both observations are preserved. These measurements establish neither whole-application speedups nor general compiler-speed improvements.

Separate CPU profiles place 96.38% to 99.04% of baseline cumulative samples in the four retained kernels, including inlinees; candidate shares range from 87.59% to 99.33%. The [profile summaries](results.json) support workload materiality. They do not isolate an exact compiler pass cost or establish a hardware cause. The processing fixtures retain their linear scans and required input/payload reads, with zero timed allocations.

The [design review](../../SCALAR-OPTIMIZATION.md) assessed 20 techniques. These campaigns measure the selected implementation, rather than 20 separately implemented variants. Correctness, mutation, cross-target, and regression evidence is tracked separately in [VALIDATION.md](VALIDATION.md).

## Initial campaign results, retained

| Campaign | Runtime benchmark | Fixed N | HF median seconds B / C | HF ratio | Go median ns/op B / C | Go ratio |
|---|---|---:|---:|---:|---:|---:|
| C1 | BenchmarkMaskedArithmetic | 1,445,086,706 | 1.531256 / 1.743852 | 0.878088 | 0.9877 / 1.2355 | 0.799433 |
| C1 | BenchmarkRotatedField | 1,516,836,890 | 1.518606 / 1.900727 | 0.798960 | 1.046 / 1.318 | 0.793627 |
| C1 | BenchmarkCorrelated | 1,241,721,855 | 1.680809 / 1.568426 | 1.071653 | 1.2305 / 1.2675 | 0.970809 |
| C1 | BenchmarkFixedPointDispatch | 347,302,617 | 2.073513 / 1.363912 | 1.520268 | 5.9815 / 3.9645 | 1.508765 |
| C2 | BenchmarkPackedDescriptors | 2,658,632 | 1.734171 / 1.525976 | 1.136434 | 664.6 / 554.05 | 1.199531 |
| C2 | BenchmarkPackedHeaders | 2,296,036 | 1.450100 / 1.845296 | 0.785836 | 632.35 / 785.45 | 0.805080 |
| C2 | BenchmarkCorrelatedDescriptors | 4,611,129 | 2.545104 / 1.493560 | 1.704052 | 592.45 / 317.4 | 1.866572 |

| Campaign | Compile target | Compiles per batch | Median ms/compile B / C | HF ratio | Peak RSS KiB B / C |
|---|---|---:|---:|---:|---:|
| C1 | go/parser | 6 | 303.921462 / 300.895380 | 1.010057 | 66640 / 66424 |
| C1 | encoding/json | 12 | 127.009437 / 127.483830 | 0.996279 | 50712 / 53924 |

### Initial negatives and inconclusive results

- Tiny `BenchmarkCorrelated`: Hyperfine ratio 1.071653 is not corroborated by the Go ratio 0.970809. Do not present it as a demonstrated gain.
- Tiny `BenchmarkMaskedArithmetic` and `BenchmarkRotatedField`: both timers favor the baseline. The measurement alone does not identify the cause.
- `BenchmarkPackedHeaders`: Hyperfine ratio 0.785836 and Go ratio 0.805080 both favor the baseline. The Hyperfine median represents 27.253% more candidate time. This initial regression motivated the projection work and remains part of the record.
- Initial compiler timing ratios, 1.010057 for go/parser and 0.996279 for encoding/json, are small relative to their sample spread. They do not establish a broad compiler-speed improvement. The encoding/json single-run RSS diagnostic increased from 50,712 to 53,924 KiB.

## Allocations and binary sizes

Every recorded runtime sample reports 0 B/op and 0 allocs/op. Processing-fixture setup precedes `ResetTimer` and is excluded.

| Campaign | Baseline test binary bytes | Candidate bytes | Candidate minus baseline |
|---|---:|---:|---:|
| F | 4,719,597 | 4,719,389 | -208 |
| C1 | 4,701,041 | 4,701,385 | +344 |
| C2 | 4,719,597 | 4,719,157 | -440 |

These executable sizes include the test harness and fixture code. Archived binary sizes and SHA-256 values were checked against their campaign logs; parity outputs are retained in the sidecar.

## Measurement protocol and validation

Runtime binaries are built before measurement and checked with `^TestScalar`. Each comparison calibrates one fixed N for both variants. Primary Hyperfine samples invoke the compiled binaries with `-test.benchtime=Nx -test.count=1 -test.benchmem -test.cpu=1`. Go corroboration uses the same N with `-count=10`. Exact recorded workload commands are in the JSON; reconstructed Go commands are labeled as such.

Compiler cost uses direct serial compiler invocations against already-built dependency archives, bypassing the action cache for the target package. It is not a full clean build. CPU/allocation profiles and one unprofiled RSS diagnostic per variant are collected separately from primary samples. RSS and sparse single-invocation compiler profiles do not establish precise pass costs or memory distributions.

Extraction checked 18 comparisons, 36 Hyperfine distributions with 360 raw times, and 280 Go samples. Reported means, medians, standard deviations, minima, and maxima were recomputed from raw times. Go output rows were checked against logged samples and fixed N, and their medians were recomputed.

Numeric duration/noise gates: **PASS**. The floor is one second per sample and Hyperfine standard deviation at most 20% of median. A valid measurement can still be a regression or fail to corroborate a speedup.

### Per-campaign protocol and environment

Unset means the variable was recorded as absent. “Not recorded” means the earlier campaign did not capture that field. Meaningful Go settings and public freeze metadata are retained in the sidecar. Private paths use labeled portable placeholders; host identifiers and raw machine log wrappers have been removed. Source hashes are preserved separately.

| Setting | F | C1 | C2 |
|---|---|---|---|
| CPU model | AMD EPYC 9V45 96-Core Processor | AMD EPYC 9V45 96-Core Processor | AMD EPYC 9V45 96-Core Processor |
| Go version | go version go1.27.4 linux/amd64 | go version go1.27.4 linux/amd64 | go version go1.27.4 linux/amd64 |
| Hyperfine version | hyperfine 1.19.0 | hyperfine 1.19.0 | hyperfine 1.19.0 |
| GOAMD64 (resolved) | v1 | v1 | v1 |
| GOEXPERIMENT (resolved) | (empty) | (empty) | (empty) |
| GOMAXPROCS | 1 | 1 | 1 |
| GOGC | 100 | Not recorded | Not recorded |
| GOMEMLIMIT | off | Not recorded | Not recorded |
| GODEBUG | unset | Not recorded | Not recorded |
| TMPDIR | <TEST_TMPDIR> | Not recorded | Not recorded |
| GOTMPDIR | <GO_TMPDIR> | Not recorded | Not recorded |
| CPU affinity | [8] | Not recorded | Not recorded |
| Target seconds | 1.5 | 1.5 | 1.5 |
| Warmups per side | 3 | 3 | 3 |
| Minimum runs per side | 10 | 10 | 10 |
| Order | baseline then candidate | baseline then candidate | baseline then candidate |
| Protocol record source | freeze event | preserved runner snapshot | preserved runner snapshot |

## Limits and claim boundaries

- Baseline-first/candidate-second order permits temporal drift. An affinity record describes eligible CPUs; it does not prove an idle host, fixed frequency, or exclusive execution.
- Initial campaigns lack the extended environment and affinity records. Do not fill those fields from the final campaign.
- Ratios are descriptive medians. No significance test or confidence interval is supplied. The runtime corpus is synthetic, including repeated 1,024-record buffers on the recorded host.
- C1/C2 recorded target compiler-input hashes without repeated input checks or explicit imported-archive hashes. For campaigns with `compile_input_freeze` events, the matching input manifests and their stated recheck scope are retained below. Do not apply C1/C2’s provenance limitation to a later campaign that records the additional checks.
- Git HEAD alone does not identify the modified working tree. Use each campaign’s manifest and executable hashes. The campaign folders do not themselves archive a complete compiler source tree or executable.
- Preserve campaign boundaries. Do not pool raw samples, reuse an earlier baseline for a later candidate, or assign old compiler-cost results to a different compiler hash.

### Compiler-input verification by campaign

| Campaign | Target | Hashed input entries | Recorded verification |
|---|---|---:|---|
| F | go/parser | 71 | before timing and after profiling, including imported archives |
| F | encoding/json | 82 | before timing and after profiling, including imported archives |
| C1 | compiler workloads | n/a | No extended input-freeze event |
| C2 | no compiler workloads | n/a | Not applicable |

## Reproduce the measurement protocol

Use a Linux/amd64 host with a built fork, Python 3.11 or later available through `uv`, Hyperfine, and `taskset`. Follow the repository's [build instructions](../../../README.md) first. Start measurement after competing builds and tests have finished. The runner builds both benchmark binaries before timing, checks their scalar semantic tests, calibrates a shared N, then records both variants in a fresh campaign directory.

Run this from the repository root or one of its subdirectories. The CPU is chosen from the current process's permitted affinity set and held fixed for the complete campaign.

```bash
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
hyperfine_bin=$(command -v hyperfine)
tool_dir=$(env GOENV=off GOTOOLCHAIN=local GOFLAGS= GOOS=linux GOARCH=amd64 \
  GOAMD64=v1 GOEXPERIMENT= GOROOT="$repo_root" "$repo_root/bin/go" env GOTOOLDIR)
env GOENV=off GOTOOLCHAIN=local GOFLAGS= GOOS=linux GOARCH=amd64 \
  GOAMD64=v1 GOEXPERIMENT= GOROOT="$repo_root" \
  "$repo_root/bin/go" build -o "$tool_dir/pprof" cmd/pprof
benchmark_cpu=$(uv run --no-project python -c 'import os; print(min(os.sched_getaffinity(0)))')
evidence_root=$(mktemp -d "${TMPDIR:-/tmp}/logos-scalar-evidence.XXXXXX")
mkdir -p "$evidence_root/tmp" "$evidence_root/work" "$evidence_root/cache"

env -u GODEBUG \
  GOOS=linux GOARCH=amd64 GOAMD64=v1 GOEXPERIMENT= \
  GOMAXPROCS=1 GOGC=100 GOMEMLIMIT=off \
  TMPDIR="$evidence_root/tmp" GOTMPDIR="$evidence_root/work" \
  GOCACHE="$evidence_root/cache" \
  taskset -c "$benchmark_cpu" \
  uv run --no-project python \
  "$repo_root/logos-doc/measurements/scalar-opt/benchmark.py" \
  --phase all \
  --repo "$repo_root" \
  --evidence "$evidence_root" \
  --hyperfine "$hyperfine_bin" \
  --target-seconds 1.5
```

The runner prints the fresh evidence directory and preserves all raw observations, profiles, manifests, fixed counts, and commands. It sets `GOROOT` to the supplied repository, `GOENV=off`, `GOTOOLCHAIN=local`, and empty `GOFLAGS`. The final recorded campaign used CPU 8; the portable command selects a CPU permitted on the new host. The new campaign's own environment and affinity records determine its provenance.

The explicit profiler build handles toolchains whose normal build omits that
optional tool. The runner checks its availability before timing and includes
its binary hash in the frozen inputs. Historical runner hashes in
`validation.json` identify the original PR #28 version; the current runner
also supports the follow-up compiler comparison and throughput columns.

For the full corpus, omit `--benchmark` and `--compile-package` as shown. The defaults now cover thirteen runtime kernels, including six added by [the scalar fact follow-up](FOLLOWUP.md), and the go/parser and encoding/json compiler workloads. The original campaign above measured seven runtime kernels. A new host, compiler revision, or environment requires a new paired campaign; do not reuse earlier baseline samples. The correctness and full regression procedure and its recorded status belong to [VALIDATION.md](VALIDATION.md).

To compare two compiler revisions at actual O3, supply both
`--baseline-compiler /path/to/original/compile` and
`--candidate-compiler /path/to/new/compile`. The runner selects each compiler
through `-toolexec`, rejects different binaries with colliding tool identities,
freezes both binaries,
and uses `-O=3` with `logosopt=1` on both sides. The initial builds and Go
corroboration share those exact compiler selections and flags. This mode is
separate from the default single-compiler flag-off/flag-on comparison.

## Public-data redaction

Private absolute workspace, toolchain, cache, temporary, and evidence paths have been replaced by labeled portable placeholders such as `<REPO_ROOT>`, `<CAMPAIGN_F>`, and `<GO_CACHE>`. Local campaign directory identifiers are labeled F, C1, and C2. Host identifiers and kernel build metadata are removed. Raw process logs are replaced by public benchmark rows, outcomes, elapsed observations, and profile summaries with identity headers omitted.

Every original JSON numeric token was checked recursively before and after redaction and remains byte-for-byte unchanged. This includes every sample, timing statistic, iteration count, allocation observation, recorded size, and RSS measurement. All original SHA-256 values are also unchanged. Those hashes and recorded byte counts identify the original measured artifacts; they do not describe the redacted representation. CPU model, OS/architecture, flags, affinity, and runtime configuration needed to interpret the results remain available.

Placeholders in historical command records are labels, not commands to execute. Use the portable reproduction procedure above to create a new campaign. The exact measured runner source and private raw logs are retained outside this public payload; [validation.json](validation.json) contains the small public validation summary, runner hashes, portable commands, and probe summary.

## Provenance

| Campaign | Artifact | SHA-256 |
|---|---|---|
| F | compiler | `f33648781c282a3546bb3a221991e0dd1a8590eed5d075931361a6c57490f03a` |
| F | Go driver | `4c4d2f8ca51d5a4960009f5f24256a4dc7f28b305ee74185ff2e0e3eb0fab902` |
| F | runner | `e1821be01625e17dc778d12b3c632b0497b134734316fa77fb8b979f6061ec10` |
| F | processing_test.go | `75d4fc314797bf34fcf89af9e6fe4e258782067a3a541471477497f23f4d6ddd` |
| F | scalaropt_test.go | `be97210b775ab079987df7d69aab97598b7ba130ebd2f9884d3cbf444e4b6af7` |
| C1 | compiler | `d16455a25bf4490a03d9169156c5de1bc475e514af764706402a94b96abd8068` |
| C1 | Go driver | `b817765a81bc4c021d21b6a6196213a080d972d5d976d8c1ba5311a8fa5b027a` |
| C1 | runner | `158b7066fce5a114f5a69dab5777708ca10f6b1847a16830cc7d2fb0622320e9` |
| C1 | scalaropt_test.go | `be97210b775ab079987df7d69aab97598b7ba130ebd2f9884d3cbf444e4b6af7` |
| C2 | compiler | `d16455a25bf4490a03d9169156c5de1bc475e514af764706402a94b96abd8068` |
| C2 | Go driver | `b817765a81bc4c021d21b6a6196213a080d972d5d976d8c1ba5311a8fa5b027a` |
| C2 | runner | `dcee64999a1948e488982782ec891ba9e81077c11b8be55cc25b3a0584612d9b` |
| C2 | processing_test.go | `75d4fc314797bf34fcf89af9e6fe4e258782067a3a541471477497f23f4d6ddd` |
| C2 | scalaropt_test.go | `be97210b775ab079987df7d69aab97598b7ba130ebd2f9884d3cbf444e4b6af7` |

- **F**, final: `campaign-F`; recorded Git HEAD `b0bbbc05d5a7dee54e6501415163e78290f72231`.
- **C1**, initial: `campaign-C1`; recorded Git HEAD `b0bbbc05d5a7dee54e6501415163e78290f72231`.
- **C2**, initial: `campaign-C2`; recorded Git HEAD `b0bbbc05d5a7dee54e6501415163e78290f72231`.

The [raw JSON sidecar](results.json) retains every distribution statistic, raw timing, exit code, Go sample, public benchmark rows and outcome summaries, fixed N, command, binary size/hash, compiler RSS record, compile input hash, profile summary, and full source manifest represented by shared hashes plus campaign additions/overrides. The [reproduction runner](benchmark.py) is the portable published runner. The measured and published runner hashes and the publication-equivalence probe summary are recorded in [validation.json](validation.json). Exact measured runner source and private raw logs are not embedded in this public evidence bundle.
