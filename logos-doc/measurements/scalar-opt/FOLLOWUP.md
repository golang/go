# Scalar fact closure: incremental validation and measurements

Issue: [#50](https://github.com/gosuda/LoGOs/issues/50).
Parent: [PR #28](https://github.com/gosuda/LoGOs/pull/28).
Baseline compiler source: `7089362d7b971fcaa7a8fb7c29034bb84ff37c95`.

This campaign compares the existing scalar optimization against the follow-up,
with actual O3 enabled on both sides. It does not reuse the original
flag-off/flag-on timing distributions.

## Frozen attacks

All attack sources below were frozen before implementation. The baseline
failed on missing optimization behavior; dynamic controls and runtime
semantics passed. Pre-freeze review corrected architecture annotation syntax
and added actual Power10 assertions. Production review then corrected opcode
width handling before the first GREEN run. No frozen attack was weakened.

| Repository path | SHA-256 |
|---|---|
| `src/cmd/compile/internal/ssa/bswap_logos_test.go` | `eac0c5aa68fdd3f7b914f0561fbb8a78d8f958c3646416c68117adec5b3255aa` |
| `src/cmd/compile/internal/ssa/known_bits_intrinsics_logos_test.go` | `540de1119792d204b97e6bd4230f3bee5aa1b9d1d2c904699dba98bef93f4319` |
| `src/cmd/compile/internal/test/testdata/scalaropt/bitfacts_test.go` | `eda5ca404021d4685d49aed2f8ff8e1475fcc918454e27d6f876ca34f7bda6f2` |
| `src/cmd/compile/internal/test/testdata/scalaropt/bswap_test.go` | `6e2d7270713a38db1427a70b21202b78a1d6d9af5ecebf9cf52fbe771876db0c` |
| `src/cmd/compile/internal/test/testdata/scalaropt/phi01_test.go` | `89f8073ad10ffa69db5de852238108a37bf428ffa23beed9da68893de45ca0c3` |
| `test/logos/codegen/scalar_bitfacts.go` | `105be8712177f326c82e84c87112aeed3b9f213ec45b454c6705c4de00951491` |
| `test/logos/codegen/scalar_bitfacts.lgo` | `105be8712177f326c82e84c87112aeed3b9f213ec45b454c6705c4de00951491` |
| `test/logos/codegen/scalar_bswap.go` | `51c4cb49230a63cacbcced2eb2dc211cf45084f432024b3facade5cb1f74a886` |
| `test/logos/codegen/scalar_bswap.lgo` | `51c4cb49230a63cacbcced2eb2dc211cf45084f432024b3facade5cb1f74a886` |
| `test/logos/codegen/scalar_phi01.go` | `a72a8ad9482be29fd4c63d8306d1e11c4e4cf6bbac7ab67bdf9f6ebfbedcb40f` |
| `test/logos/codegen/scalar_phi01.lgo` | `a72a8ad9482be29fd4c63d8306d1e11c4e4cf6bbac7ab67bdf9f6ebfbedcb40f` |

## Validation protocol

Run the complete toolchain build, uncached SSA short suite, runtime parity
with both scalar flag values and SSA checking, all declared source-codegen
targets, actual linux/386 runtime tests, native/ejected behavior, and vet.
Power10 codegen covers ppc64 and ppc64le; it is compilation evidence, not
Power10 runtime evidence. Keep the full regression gate required.

Mutation probes must fail behaviorally, restore the production source
byte-for-byte, and return the frozen tests to GREEN. Test hashes are verified
again after integration.

## Measurement protocol

The paired-compiler runner uses actual `-O=3`, identical sources, inputs,
target ISA, runtime settings, and per-side compiler selection for both the
compiled Hyperfine workload and Go corroboration. It rejects compiler tool-ID
collisions. Freeze one common work count per workload, run at least one
second of useful work per sample, use three warmups and at least ten timed
runs, and collect CPU profiles separately. All heavy work is serialized.

The six new processing workloads each consume 1024 records or words per
iteration. Payloads remain dynamic and observed; no timed setup or instruction
padding is added. Previous processing workloads provide regression controls.
Runtime latency is the speed metric. Report binary bytes and allocations
separately, and retain every non-win. These are synthetic kernels on one host.

Raw environment, command, and workspace logs stay private. Publish only
repository-relative source paths, hashes, numeric samples, portable commands,
and outcome summaries. Original evidence remains in README.md/results.json.

## Implementation and correctness results

The follow-up closes three gaps in the existing analysis:

1. Correlated constants run after generic CSE and before phiopt. This preserves
   the predecessor alignment of zero/one phis long enough to prove that
   `(0, 7)` and `(1, 6)` both sum to seven. The existing traversal and cleanup
   remain sufficient.
2. Byte swaps permute both values and known-bit masks. Bits above the operation
   width are zero even when Power10 represents Bswap16 with a wider result.
   SCCP also folds byte-swap operands that become constant late, while retaining
   the original result type and leaving the instruction intact until convergence.
3. BitLen, Ctz/CtzNonZero, and PopCount derive a result interval from mandatory
   and possible input bits. Its common prefix supplies additional known bits.
   The opcode determines input width; the result type determines representation.
   The transfer takes constant work and reuses the existing fact cache.

| Gate | Observed result |
|---|---|
| Complete toolchain rebuild | Passed |
| Rebuilt, uncached full SSA short suite | Passed, 125.325 seconds |
| Scalar runtime wrapper, both flag values and SSA checking | Passed |
| Direct native scalar fixture, both flag values | Passed |
| Go and LoGOs scalar source codegen | Passed on all declared targets: amd64 v1-v4, arm64 v8.0/v8.1, 386 softfloat/sse2, and both Power10 endian targets |
| 386 and wasm scalar fixture compilation with SSA checking | Passed |
| Actual 386 fixture execution | Unavailable locally: exec-format error and no i386 emulator; both flag values are now selected in CI |
| Native/ejected scalar smoke | Existing scalar script passed; this is not execution of ejected copies of the three new fixtures |
| Selected standard library at O3 | Passed: math/bits, encoding/binary, math/big, strconv, go/parser, encoding/json, sort, strings, bytes |
| Vet, Go formatting, whitespace, and frozen-source hashes | Passed |
| Python runner | Syntax, command-line handling, real throughput-column parsing, and the runtime campaign passed; Ruff was unavailable in the offline tool cache |
| Full regression | Required before merge; no successful full-floor result is claimed |

Two mutation probes failed as required. Dropping the known-zero upper bits of
widened Bswap16 failed `TestLogosBswapKnownBits/16_uint`. Removing the narrow
Ctz zero-input cap failed `TestLogosKnownBitsIntrinsicsExtremes`. Both production
files were restored byte-for-byte and the focused tests passed again. Initial
source RED and rebuilt GREEN pin the pass order; no separate pass-order mutation
was run.

Two initial integration builds terminated compiler subprocesses before tests
could run. Smaller heap settings and serialized retries passed. The termination
cause was not established. A native/ejected script also initially inherited
unrelated parent Git metadata through a workspace temporary directory; rerunning
with a temporary directory outside that checkout passed. These failed attempts
remain part of the evidence.

The parent PR's [full CI rerun](https://github.com/gosuda/LoGOs/actions/runs/43841164938)
passed testdir but failed regression in three SIMD command scripts. Two failures
tried to execute v4-targeted binaries on a host without v4. The third executed an
unsupported AVX-512 instruction in a SIMD slice program. The job's outer target
setting does not establish that program's target: script tests reset the
environment, and the unchanged slice script uses the driver's default target.
A separate nested-sum reproducer explicitly sets `GOAMD64=v1` and confirms that
vector initialization can escape its runtime feature guard at that baseline.

[Issue #51](https://github.com/gosuda/LoGOs/issues/51) tracks the correctness
dependency. The broader repair in [PR #54](https://github.com/gosuda/LoGOs/pull/54)
addresses both zero-entry hoisting and short-arm SIMD speculation by branch
elimination. Nested reductions alone do not test that second path. The scalar
fact-closure changes do not resolve either defect, and a fresh complete
regression pass remains required before merge. The independent zero-only repair
is superseded by #54; its focused SSA results are not an installed-compiler or
full-regression pass.

The follow-up's own [CI run](https://github.com/gosuda/LoGOs/actions/runs/43855555345)
at `635b0126` subsequently failed the same three scripts in its full regression
job (`cmd/go`, 1672.425 seconds). Its separate testdir, lint, and scripts jobs
passed. Those passes do not replace the failed full-regression gate.

## Incremental runtime measurements

The complete numeric distributions and source manifests are in
[followup-results.json](followup-results.json). Each ratio is baseline median
divided by candidate median, so values above one favor the candidate. Both
compilers use actual O3. The target is Linux/amd64 v2, with one permitted CPU,
GOMAXPROCS=1, GOGC=100, and GOMEMLIMIT=1GiB. Hyperfine uses three warmups and ten
timed samples per side. Go corroboration uses ten separate samples at the same
fixed N. Profiles are separate from the timed samples. Heavy local work was
serialized, and both methods run the baseline before the candidate.

| New workload | Fixed N | Hyperfine ratio | Go ratio |
|---|---:|---:|---:|
| Phi01Descriptors | 2,793,817 | 1.426 | 1.488 |
| BswapLateDispatch | 4,870,130 | 1.981 | 1.907 |
| BswapNetworkByteFields | 3,406,769 | 1.371 | 1.587 |
| AlignedBitCounts | 3,402,133 | 1.156 | 1.123 |
| HeaderBitLengths | 3,116,560 | 2.265 | 2.335 |
| SparsePopulationClasses | 2,982,701 | 1.335 | 1.335 |

Every new workload processes 1,024 records or words per benchmark iteration.
The benchmark retains dynamic payload loads, reductions, and an observed sink.
All ten runtime workloads completed both methods and separate profiling. Their
samples met the one-second useful-work floor and the runner's noise limit.
All 200 Go samples reported 0 B/op and 0 allocs/op. The benchmark executable
changed from 5,397,753 to 5,396,217 bytes, a reduction of 1,536 bytes. Separate
profiles place 97.71%-100% of cumulative sampled CPU time in each new workload's
processing kernel. These observations establish that the kernels are hot in
this corpus; they do not establish application-wide benefit.

The codegen tests establish specific removed instructions. In BswapLateDispatch,
the old compiler already hoisted the byte swap out of the inner loop. The new
compiler removes the repeated arithmetic, comparison, and conditional selection;
the workload must not be described as removing 1,024 byte swaps per iteration.

### Controls limit performance attribution

| Existing control | Hyperfine ratio | Go ratio | Interpretation |
|---|---:|---:|---|
| FixedPointDispatch | 0.985 | 1.170 | Methods disagree; inconclusive |
| PackedDescriptors | 1.737 | 1.689 | Unexplained timing change |
| PackedHeaders | 1.240 | 1.372 | Unexplained timing change |
| CorrelatedDescriptors | 1.808 | 1.817 | Unexplained timing change |

Disassembly of the exact measured binaries shows identical machine-code bytes,
sizes, and instruction counts in all four control kernels. Their benchmark
callers also have identical instructions after address normalization. All eight
symbols move by -416 bytes in the candidate. Placement is an observed difference;
its causal contribution to timing has not been established.

These controls prevent attributing the whole measured gain of the new workloads
to the new transfer rules. The six rows are paired-binary observations, supported
by independent instruction-removal tests. They are not isolated causal estimates
or whole-application speed claims. No samples were removed, and the control gains
are not counted as additional optimization wins.

No significance test, confidence interval, or numeric hardware time floor is
supplied. Source and assembly retain work proportional to the number of dynamic
records, but that does not establish a nanosecond lower bound. Consequently this
record does not claim completion of G6's hardware-floor ranking requirement.
The bounded rules are retained for their proved fact propagation and observed
codegen improvements; broader performance acceptance remains open.

## Compiler cost and retained non-results

Direct compilation of `go/parser` used the same three compilations per timed
sample. Its median ratio was 0.980, which establishes no compiler-speed gain.
Separate profiles and single-child peak-RSS observations are retained; a single
RSS observation is not a memory distribution.

The `encoding/json` facade used twelve compilations per sample. Its candidate
standard deviation divided by median was about 0.238, above the 0.20 limit.
All ten samples per side are retained, but the result is unscored. The runner
stopped at this check, after all runtime workloads and the parser measurement
had completed. JSON profiling and RSS were therefore not collected in that run.

A fresh JSON-only retry requested longer, equally sized batches with a
three-second target. It failed to calibrate within five attempts and produced
no timing distribution. This failed retry is recorded separately. No old baseline
was reused, no samples were pooled across campaigns, and no compiler-cost gain
is claimed.

## Input retention audit

On 2026-11-16, a separate SIMD repair rebuild exhausted disk space. Clearing the
task's build cache also removed imported package archives referenced by these
compiler measurements. Sources, manifests, measured executables, profiles, and
numeric results were retained.

The follow-up's 81 unique imported archives were regenerated using the retained
measurement driver and candidate compiler, with the original plain export-list
preparation at `GOAMD64=v2`. Every one of the 94 unique inputs across the three unchanged
input manifests matched its original SHA-256 afterward. The archives were also
copied outside the build cache and verified again. No benchmark was rerun, no
historical manifest was rewritten, and no timing sample changed.

This restoration covers the follow-up main campaign and its JSON calibration
retry. It does not restore the earlier campaign's archives; that separate
retention gap is recorded in [VALIDATION.md](VALIDATION.md#historical-input-retention).

## Reproduce a fresh comparison

Build the baseline revision and this revision in separate checkouts using the
repository build procedure. Save each resulting `pkg/tool/linux_amd64/compile`
binary. The candidate checkout supplies the driver, linker, source corpus, and
profiler; `-toolexec` selects the compiler for each side. Set the two compiler
paths below to those saved binaries before running this command from the
candidate checkout.

```bash
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
baseline_compiler=/path/to/baseline/compile
candidate_compiler=/path/to/candidate/compile
hyperfine_bin=$(command -v hyperfine)
tool_dir=$(env GOENV=off GOTOOLCHAIN=local GOFLAGS= GOOS=linux GOARCH=amd64 \
  GOAMD64=v2 GOEXPERIMENT= GOROOT="$repo_root" "$repo_root/bin/go" env GOTOOLDIR)
env GOENV=off GOTOOLCHAIN=local GOFLAGS= GOOS=linux GOARCH=amd64 \
  GOAMD64=v2 GOEXPERIMENT= GOROOT="$repo_root" \
  "$repo_root/bin/go" build -o "$tool_dir/pprof" cmd/pprof
benchmark_cpu=$(uv run --no-project python -c 'import os; print(min(os.sched_getaffinity(0)))')
evidence_root=$(mktemp -d "${TMPDIR:-/tmp}/logos-scalar-followup.XXXXXX")
mkdir -p "$evidence_root/tmp" "$evidence_root/work" "$evidence_root/cache"

env -u GODEBUG \
  GOOS=linux GOARCH=amd64 GOAMD64=v2 GOEXPERIMENT= \
  GOMAXPROCS=1 GOGC=100 GOMEMLIMIT=1GiB \
  TMPDIR="$evidence_root/tmp" GOTMPDIR="$evidence_root/work" \
  GOCACHE="$evidence_root/cache" \
  taskset -c "$benchmark_cpu" \
  uv run --no-project python \
  "$repo_root/logos-doc/measurements/scalar-opt/benchmark.py" \
  --phase all --repo "$repo_root" --evidence "$evidence_root" \
  --hyperfine "$hyperfine_bin" --target-seconds 1.5 \
  --baseline-compiler "$baseline_compiler" \
  --candidate-compiler "$candidate_compiler" \
  --benchmark BenchmarkPhi01Descriptors \
  --benchmark BenchmarkBswapLateDispatch \
  --benchmark BenchmarkBswapNetworkByteFields \
  --benchmark BenchmarkAlignedBitCounts \
  --benchmark BenchmarkHeaderBitLengths \
  --benchmark BenchmarkSparsePopulationClasses \
  --benchmark BenchmarkFixedPointDispatch \
  --benchmark BenchmarkPackedDescriptors \
  --benchmark BenchmarkPackedHeaders \
  --benchmark BenchmarkCorrelatedDescriptors
```

The runner fails closed when calibration, sample duration, noise, compiler
identity, or frozen inputs fail validation. Such a failure must be retained with
the completed per-workload results. A successful fresh run does not retroactively
change the outcomes reported here.
