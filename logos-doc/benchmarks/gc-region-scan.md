# Region GC scan measurement

Date: 2026-11-16. Scope: M18-A-S21, RFC-305 section 3.5.2.

## Decision

Keep the same-allocation lookup shortcut for the measured self-reference path. Its final fixed-work speed ratio is **2.178**, above G6's 1.05 gate. The change preserves tracing behavior in the targeted tests and mutation probe.

This is a workload-specific improvement. The final nil and other-arena controls have **4.1% and 5.1% higher median latency**, respectively. Those costs remain part of the result; this record does not establish a universal GC improvement. The separate root-work accounting correction makes no speed claim.

## Workload and source boundary

The baseline is `dc95954659ede58433ba7df962cefc4ebb36794a`, which already contains the accounting correction. Both binaries include the identical `export_logos_gcscan_test.go` and `logos_gcscan_test.go` fixtures. The sole measured production delta adds `p-obj < size` to the scanner's existing nil filter, after the slot's scan work is recorded.

`BenchmarkLogosGCScan` calls the actual runtime scanner over 512 pointer slots in one 4,096-byte arena allocation, repeated **1,000,000 times** per sample. It visits the same source block repeatedly, so this is a cache-resident scanner workload. Automatic GC is disabled during measurement. Setup is outside the Go benchmark timer. Hyperfine measures the complete fixed-work process, including its small startup and setup cost.

- `Self`: every pointer targets the source allocation, including its later elements.
- `Nil`: every pointer is nil.
- `External`: every pointer targets another live manual arena allocation. It does **not** measure heap marking.

The helper has no heap work buffer or stack scan state. Heap referent survival, scalar non-roots, and release behavior are covered separately by the behavioral tests.

## Environment and procedure

Linux/amd64, AMD EPYC 9V45 96-Core Processor, the fork's Go 1.27.4, `GOAMD64=v1`, `GOEXPERIMENT=logosarena`, and `GOMAXPROCS=1`. Both binaries have matching compiler and build settings. Hyperfine is 1.19.0.

Each series uses a compiled test binary, three warmups and ten measured runs. No builds or tests run concurrently with the timed samples. CPU profiles are collected separately. The final integration run compares baseline and candidate adjacently for each input, in the order Self, Nil, External. Earlier phase measurements are retained in the JSON because the controls changed between phases, indicating possible host-load drift.

All final series have standard deviation below 20% of their median. Every timed run exceeds one second. The minimum candidate Go benchmark sample is 1,302 ns/op, or 1.302 seconds of timed work at the frozen iteration count.

The baseline profile attributes 99.58% cumulative CPU time to `logosArenaScanObject`, satisfying hot-path admission. Calibration's nil-input proxy was 1,900 ns/op against 3,992 ns/op for Self, a ratio of 2.101. Visiting all 512 typed slots is required by the contract; the nil measurement is an empirical floor proxy, not a proof of a machine-level lower bound. The single-function, single-concern exception in G6 applies. Readability cost is 0.05, verified parity gives safety 1, and the final score is 2.145.

## Final adjacent comparison

Times below are seconds for the same one million scans. A positive latency change means the candidate took longer.

| Input | Baseline median | Candidate median | Latency change | Baseline / candidate |
|---|---:|---:|---:|---:|
| Self | 3.254167 | 1.494322 | -54.08% | 2.178 |
| Nil | 1.489952 | 1.550643 | +4.07% | 0.961 |
| Other arena (`External`) | 3.180422 | 3.343736 | +5.13% | 0.951 |

| Series | Standard deviation | Minimum | Maximum | Standard deviation / median |
|---|---:|---:|---:|---:|
| Self baseline | 0.332158 | 2.939211 | 4.045230 | 10.21% |
| Self candidate | 0.157445 | 1.306106 | 1.769607 | 10.54% |
| Nil baseline | 0.130858 | 1.285571 | 1.688800 | 8.78% |
| Nil candidate | 0.172634 | 1.475668 | 2.027915 | 11.13% |
| External baseline | 0.122921 | 2.987279 | 3.395816 | 3.86% |
| External candidate | 0.273257 | 3.095365 | 3.816177 | 8.17% |

Ten `go test -benchmem` samples per input and per version all report **0 B/op and 0 allocs/op**. These corroborating samples were collected in the earlier baseline and candidate phases, so their ns/op medians must not replace the adjacent comparison above. The runtime test binary grows from 15,997,847 to 15,998,583 bytes, an increase of 736 bytes. Binary size is reported separately from speed.

The candidate Self profile no longer shows the heap/span lookup helpers among its top samples. Its remaining cost is predominantly typed-pointer iteration. This supports the mechanism; neither profile measures whole-GC duration or application throughput.

## Correctness evidence

The unchanged behavioral fixture passes before and after the optimization. A semantics-preserving speed change has no missing-behavior RED. The accounting commit has its own genuine failing tests: a single nil pointer reports 64 bytes instead of 72, and a leased chunk adds zero instead of 65,536 bytes to the root-work bound.

For the optimization's mutation probe, an overlay changes the filter to skip every non-nil pointer. Both arena and region cases fail because live referent 0 is collected. The unmutated binary passes under the same constrained runtime environment. All six new test files retain their frozen hashes.

The rebuilt toolchain passes the targeted runtime suite, race detection, the `nogreenteagc` variant, `runtime/metrics`, and vet. The `logos_arena_go` and `logos_region_eject_parity` scripts pass; ejected fixtures also build with upstream Go 1.27.2. Linux/arm64 cross-compilation passes. ARM64 execution and the full `run.bash` milestone gate were not run.

Several parallel compiler jobs and one initial serialized mutation build were killed. The successful retries used lower compilation concurrency and a memory budget. Those compiler failures are not treated as behavioral test failures.

## Reproduction

Create a detached worktree at the baseline commit, copy only the two GC scan benchmark/liveness test files from this change into it, and rebuild both toolchains with upstream Go 1.27.4 as `GOROOT_BOOTSTRAP`. The baseline commit already contains the other four accounting test files. Use the recorded source hashes to check the production delta and fixture identity.

In each tree's `src` directory, build and corroborate with its own installed `bin/go`. Give `out` a different directory for each version and run:

```bash
GOEXPERIMENT=logosarena ../bin/go test -c -o "$out/runtime.test" runtime
GOMAXPROCS=1 GOEXPERIMENT=logosarena ../bin/go test \
  -run='^$' -bench='^BenchmarkLogosGCScan$' \
  -benchtime=1000000x -count=10 -benchmem runtime
```

Set `baseline` and `candidate` to the respective compiled binary paths. Run each pair serially on an otherwise idle host, retaining every output file:

```bash
for input in Self Nil External; do
  GOMAXPROCS=1 hyperfine --warmup 3 --min-runs 10 \
    --export-json "$out/$input.json" \
    "\"$baseline\" -test.run='^$' -test.bench='^BenchmarkLogosGCScan/$input$' -test.benchtime=1000000x -test.benchmem" \
    "\"$candidate\" -test.run='^$' -test.bench='^BenchmarkLogosGCScan/$input$' -test.benchtime=1000000x -test.benchmem"
done
```

Collect profiles in separate runs with `-test.cpuprofile`, never in the timed hyperfine commands. Preserve the fixed work count on both sides. If a faster machine completes below one second, raise the count for both versions and start a new measurement record.

[The JSON record](gc-region-scan.json) contains exact source and binary hashes, frozen test hashes, all timing samples, benchmem samples, profile summaries, and the gate decision. Shared-host variability and the control costs limit the conclusion to the measured scanner workload. No pause-latency, peak-memory, heap-marking, or application-speed improvement is established here.
