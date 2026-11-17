# Candidate: initialize the region scan bitmap once

## Integration status, 2026-11-17

This candidate is deferred. PR #35 restores the scanner implementation from
`release-branch.go1.27` because the required caller-contract floor has not been
established. The region-close ownership repair and every regression test remain.
The measurements below describe the original candidate revisions and remain
unchanged; they do not admit either scanner optimization into the updated PR.
A future candidate must complete its floor and parity evidence before promotion.

Date: 2026-11-16

Status: draft; native parity and repeated timing verified; full upstream and G6 admission tracked separately.

Related: #21, #26, and #35. This candidate targets the stability-test baseline and is independent of the deferred-extent candidate.

## Problem and change

`logosArenaScanObject` initializes `typePointers.mask` with `readUintptr(getGCMask(typ))` and immediately calls `fastForward(0, limit)`. For the typed path, `fastForward` replaces that mask before returning it to the scanner.

Remove only the initial `mask` field. Keep `fastForward`, the type descriptor, object addresses, byte limit, pointer dispatch, and extent accounting unchanged. This is a single-function candidate within the G6 exception; there is no new iterator or persistent cache.

## Contract and proof obligations

The existing scanner preconditions remain: a live typed manual allocation; a valid object range; initialized readable padding for any final partial pointer word; valid collector state for targets that require marking.

For a nonempty typed range, `fastForward` obtains the mask through `getGCMask` and applies the same clipping. Its old incoming mask is not read on this path. For an empty range it returns an empty iterator before either mask can be consumed. The proposed initializer therefore supplies the same iterator state to the scan loop.

The lazy mask-resolution protocol remains in `getGCMask`; direct access to `typ.GCData` is not substituted. A zero-length call no longer eagerly resolves a mask that it never scans. No contract requires that incidental initialization.

The postcondition remains the same pointer visits, marking effects, and returned scan extent. Nil slots, self pointers, scalar tails, bitmap-word boundaries, and partial pointer words retain their existing meaning.

## Required evidence

Run #35's native, race, Green Tea on/off, heap-graph, boundary, and upstream gates against this exact head. Keep the frozen tests unchanged. This semantics-preserving candidate starts from a parity baseline; it does not claim a behavioral RED.

Inspect emitted code before accepting a speed claim. The compiler may already eliminate some of the repeated source work. Use the small-object scanner benchmark to expose initialization cost and retain the 512-slot Self, Nil, and External controls. External is another manual allocation and does not measure heap marking.

Compare compiled base/head binaries with identical fixed work, `GOMAXPROCS=1`, separate CPU profiles, hyperfine warmup 3/min-runs 10, and ten benchmem samples. Require the repository's G6 admission, noise, and integrated speed-ratio gates. Record controls and reject a non-win. Do not infer application or full-collection speed from isolated scanner timing.

## Executed evidence, 2026-11-16

The measured baseline is `45e2f57a37190ba5c10e7888b32671ea631bf393`; the measured candidate is `0866c25300d1268d2b4a96fd209a1b6777ad467b`. This record applies to those exact source/workload fingerprints. Later host-ISA fixture, timeout, and ledger corrections are prerequisites for the separate full upstream rerun; they are not silently substituted into this measurement.

- [Measurement run 43829350925](https://github.com/gosuda/LoGOs/actions/runs/43829350925/job/133170631575)
- [Retained raw artifact](https://github.com/gosuda/LoGOs/actions/runs/43829350925/artifacts/14208759747)
- [Verified artifact readback](https://github.com/gosuda/LoGOs/actions/runs/43834529472/job/133187747022)
- [Native runtime and upstream gate run](https://github.com/gosuda/LoGOs/actions/runs/43829350876)

Environment: Intel Xeon 6975P-C under KVM, four exposed vCPUs, Linux amd64, Go 1.27.4, hyperfine 1.18.0, `GOAMD64=v1`, `GOEXPERIMENT=logosarena`, `GOMAXPROCS=1`. The harness clears GOGC, GOMEMLIMIT, and GODEBUG during timing. Profiles run separately. The artifact records source/input hashes, full environment, raw samples, binary bytes, assembly, and separate comparison/integration records. Readback checked the run attempt, exact source pair, unchanged source fingerprints, and equality of the frozen tests.

### Primary latency

Primary: `BenchmarkLogosGCScanSmall/Pointer/Self`, fixed N = **166440637** for both sides and all primary timing series. Hyperfine excludes three warmups and retains ten measurements per side. Noise below is standard deviation divided by median.

| Series | Baseline median (s) | Candidate median (s) | Base/head ratio | Latency change | Baseline / candidate noise |
| --- | ---: | ---: | ---: | ---: | ---: |
| Comparison | 1.39561908042 | 1.27778363492 | 1.0922186 | -8.4432% | 0.2886% / 0.4640% |
| Integration repeat | 1.39057724996 | 1.28158512246 | 1.0850448 | -7.8379% | 0.2848% / 0.3158% |

The independent ten benchmem samples give median **8.301 -> 7.655 ns/op**, a **7.7822%** latency decrease. The separate profiles place **90.65% / 89.84%** of baseline/candidate sampled CPU time in the scanner cumulatively.

### Controls

Each row uses its own calibrated, frozen N. Ratios across different rows are not a common-work ranking.

| Workload | N | Baseline median (s) | Candidate median (s) | Hyperfine latency change | Count10 median (ns/op) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small/Pointer/Nil | 170756388 | 1.43327415 | 1.32157401 | -7.7934% | 8.3255 -> 7.657 |
| Small/Pointer/OtherArena | 106973059 | 1.38764145 | 1.32214995 | -4.7196% | 12.825 -> 12.344999999999999 |
| Small/Pointer/Self | 166440637 | 1.39561908 | 1.27778363 | -8.4432% | 8.301 -> 7.655 |
| Large/External | 1000000 | 3.88007759 | 3.89874232 | 0.4810% | 3857.5 -> 3892.5 |
| Large/Nil | 1000000 | 1.43551644 | 1.42386632 | -0.8116% | 1430.5 -> 1411 |
| Large/Self | 1000000 | 1.54658487 | 1.56547164 | 1.2212% | 1542 -> 1556.5 |

All 18 cases have ten benchmem samples per side, **360 samples total**, with **0 B/op and 0 allocs/op** in every sample. Every retained useful-work invocation meets one second; the minimum across cases is **1.264615960 s**. All 14 hyperfine series have ten zero-exit samples; their maximum noise is **0.6162%**, below the existing 20% limit.

All count10 medians are retained here, including layouts without a separate hyperfine campaign:

| Workload | N | Baseline ns/op | Candidate ns/op | Latency change |
| --- | ---: | ---: | ---: | ---: |
| Small/Interface/Nil | 170454546 | 8.3735 | 7.7219999999999995 | -7.7805% |
| Small/Interface/OtherArena | 104166667 | 12.925 | 12.335 | -4.5648% |
| Small/Interface/Self | 168560370 | 8.315000000000001 | 7.641 | -8.1058% |
| Small/Irregular3/Nil | 150183558 | 9.638 | 8.964500000000001 | -6.9880% |
| Small/Irregular3/OtherArena | 69587629 | 19.52 | 19.315 | -1.0502% |
| Small/Irregular3/Self | 149303252 | 9.684999999999999 | 9.12 | -5.8338% |
| Small/Pointer/Nil | 170756388 | 8.3255 | 7.657 | -8.0295% |
| Small/Pointer/OtherArena | 106973059 | 12.825 | 12.344999999999999 | -3.7427% |
| Small/Pointer/Self | 166440637 | 8.301 | 7.655 | -7.7822% |
| Small/Slice/Nil | 171602899 | 8.316 | 7.656499999999999 | -7.9305% |
| Small/Slice/OtherArena | 108259824 | 12.925 | 12.21 | -5.5319% |
| Small/Slice/Self | 177678337 | 8.341000000000001 | 7.7044999999999995 | -7.6310% |
| Small/String/Nil | 169512808 | 8.280999999999999 | 7.6335 | -7.8191% |
| Small/String/OtherArena | 112406329 | 12.795 | 12.31 | -3.7905% |
| Small/String/Self | 176539820 | 8.317499999999999 | 7.631 | -8.2537% |
| Large/External | 1000000 | 3857.5 | 3892.5 | 0.9073% |
| Large/Nil | 1000000 | 1430.5 | 1411 | -1.3632% |
| Large/Self | 1000000 | 1542 | 1556.5 | 0.9403% |

### Generated code and size

Scanner machine-code span: **620 -> 556 bytes (-64)**. Whole `runtime.test` file size: **16,042,582 -> 16,043,974 bytes (+1,392)**. The complete executables use different untrimmed source roots and include debug/build metadata; whole-file differences are not attributed solely to this function.

The emitted amd64 code removes the initial TFlag branch, GCData/on-demand-mask resolution, and bitmap load before fastForward. The typed fastForward call and lazy-mask protocol remain. The compiler had not eliminated this first initialization.

### Correctness and admission decision

All four native amd64/arm64 x Green Tea on/off runtime jobs passed on the measured revision: build, formatting/vet, runtime regression, race, repeated boundary/lifecycle checks, noinline synchronous-close stress, and checkmark. The Green Tea jobs also retained the stability branch's exact-original ownership mutation result. The full upstream run failed shared prerequisites: a script timeout, ledger formatting, and an unsupported-host SIMD fixture where that script reached execution. The subsequent correction is tracked in #35 and #47; consult current PR checks for its result.

The one-pointer benefit is corroborated at about 7.8% lower latency. Keep the approximately 1% large-Self and large-External costs visible; this is a distribution-dependent tradeoff, not an unconditional scanner improvement.

The timing and hot-path checks pass. **Full G6 admission is still pending the caller-contract lower bound and full upstream parity.** A high scanner profile share does not establish that floor. The previous large Self/Nil proxy is not reused for a new workload; Nil still performs iterator work. CPU identification and BogoMIPS do not establish effective clock or latency bounds. This evidence supports a reviewable draft, not an unsupported whole-GC or application claim.
