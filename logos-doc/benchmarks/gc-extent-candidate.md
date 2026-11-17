# Candidate: compute region scan extent after traversal

## Integration status, 2026-11-17

This candidate is deferred. PR #35 restores the scanner implementation from
`release-branch.go1.27` because the required caller-contract floor has not been
established. The region-close ownership repair and every regression test remain.
The measurements below describe the original candidate revisions and remain
unchanged; they do not admit either scanner optimization into the updated PR.
A future candidate must complete its floor and parity evidence before promotion.

Date: 2026-11-16

Status: draft; native parity and repeated timing verified; full upstream and G6 admission tracked separately.

Related: #21, #26, and #35. This candidate targets the stability-test baseline and is independent of the bitmap-initialization candidate.

## Problem and change

The scanner currently calculates `addr - obj + PtrSize` for every typed pointer slot, although only the last value is returned. Retain the last visited address and compute the extent after traversal. Return zero if no slot was visited.

Assign `last = addr` before loading the slot and before every nil, self, stack, or heap dispatch branch. Keep the bitmap iterator, scan limit, target classification, marking calls, and work propagation unchanged.

This is a single-function candidate within the G6 exception. It adds no cache, object metadata, allocation, or new marking behavior.

## Contract and loop invariant

The existing scanner preconditions remain: a live typed manual allocation; a valid object range; initialized readable padding for a partial final pointer word; valid collector state for targets that require marking.

After zero visited slots, the old `scanned` and new `last` are both zero. After any positive number of visited slots, let A be the final visited address. The old value is `uintptr(A - obj + PtrSize)`; the new value is `last == A`. No intervening operation reads that accounting value. Computing the same expression at return preserves the result.

The iterator already uses address zero as its exhaustion sentinel, so testing `last == 0` adds no restriction to valid inputs. Keep subtraction and addition in `uintptr`, then convert the result to `int64`.

The postcondition includes nil and internally directed slots, excludes scalar tails, and counts a final pointer word in full when its start is below the byte limit. Do not clamp the result to the requested byte count. A leading scalar prefix or zero-size range with no visited pointer still returns zero.

## Required evidence

Run #35's independent byte-limit oracle, native/race/Green Tea on/off suites, real heap-graph checks, and upstream gate. Preserve the frozen tests. This semantics-preserving candidate has a parity baseline, not a claimed behavioral RED.

Inspect emitted code: the compiler may already simplify arithmetic, share offsets with heap dispatch, or change register allocation. The extra exit test can cost time on zero- and one-slot scans.

Measure independently against this candidate's immediate parent, using both the 512-slot Self/Nil/External workload and small typed allocations. External and OtherArena target manual allocations; neither measures heap-object marking. Use identical fixed work, `GOMAXPROCS=1`, separate profiles, hyperfine warmup 3/min-runs 10, and ten benchmem samples. Apply the repository's G6 admission, noise, and integrated speed-ratio requirements and record every control.

PR #26's roughly 4-5% Nil/other-arena control costs make those results material. Reject a non-win or an unacceptable distribution-dependent regression; a focused scanner gain cannot establish whole-GC or application speed.

## Executed evidence, 2026-11-16

The measured baseline is `45e2f57a37190ba5c10e7888b32671ea631bf393`; the measured candidate is `cba52d5d93465a32195e992449c7fe5034c35425`. This record applies to those exact source/workload fingerprints. Later host-ISA fixture, timeout, and ledger corrections are prerequisites for the separate full upstream rerun; they are not silently substituted into this measurement.

- [Measurement run 43829360508](https://github.com/gosuda/LoGOs/actions/runs/43829360508/job/133170661979)
- [Retained raw artifact](https://github.com/gosuda/LoGOs/actions/runs/43829360508/artifacts/14208884858)
- [Verified artifact readback](https://github.com/gosuda/LoGOs/actions/runs/43834529472/job/133187747469)
- [Native runtime and upstream gate run](https://github.com/gosuda/LoGOs/actions/runs/43829360531)

Environment: Intel Xeon 6975P-C under KVM, four exposed vCPUs, Linux amd64, Go 1.27.4, hyperfine 1.18.0, `GOAMD64=v1`, `GOEXPERIMENT=logosarena`, `GOMAXPROCS=1`. The harness clears GOGC, GOMEMLIMIT, and GODEBUG during timing. Profiles run separately. The artifact records source/input hashes, full environment, raw samples, binary bytes, assembly, and separate comparison/integration records. Readback checked the run attempt, exact source pair, unchanged source fingerprints, and equality of the frozen tests.

### Primary latency

Primary: `BenchmarkLogosGCScan/Self`, fixed N = **1000000** for both sides and all primary timing series. Hyperfine excludes three warmups and retains ten measurements per side. Noise below is standard deviation divided by median.

| Series | Baseline median (s) | Candidate median (s) | Base/head ratio | Latency change | Baseline / candidate noise |
| --- | ---: | ---: | ---: | ---: | ---: |
| Comparison | 1.54835793084 | 1.43526265234 | 1.0787976 | -7.3042% | 0.1176% / 0.1279% |
| Integration repeat | 1.54773062946 | 1.43453011146 | 1.0789112 | -7.3140% | 0.1244% / 0.1055% |

The independent ten benchmem samples give median **1542 -> 1427.5 ns/op**, a **7.4254%** latency decrease. The separate profiles place **99.35% / 100%** of baseline/candidate sampled CPU time in the scanner cumulatively.

### Controls

Each row uses its own calibrated, frozen N. Ratios across different rows are not a common-work ranking.

| Workload | N | Baseline median (s) | Candidate median (s) | Hyperfine latency change | Count10 median (ns/op) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small/Pointer/Nil | 181940701 | 1.52162766 | 1.31384130 | -13.6555% | 8.363 -> 7.159 |
| Small/Pointer/OtherArena | 118213661 | 1.52302859 | 1.35404583 | -11.0952% | 12.81 -> 11.42 |
| Small/Pointer/Self | 184375854 | 1.53984939 | 1.35172712 | -12.2169% | 8.305 -> 7.297000000000001 |
| Large/External | 1000000 | 3.88284716 | 3.83439751 | -1.2478% | 3874 -> 3819 |
| Large/Nil | 1000000 | 1.43598103 | 1.43216610 | -0.2657% | 1429.5 -> 1427 |
| Large/Self | 1000000 | 1.54835793 | 1.43526265 | -7.3042% | 1542 -> 1427.5 |

All 18 cases have ten benchmem samples per side, **360 samples total**, with **0 B/op and 0 allocs/op** in every sample. Every retained useful-work invocation meets one second; the minimum across cases is **1.299238546 s**. All 14 hyperfine series have ten zero-exit samples; their maximum noise is **0.5619%**, below the existing 20% limit.

All count10 medians are retained here, including layouts without a separate hyperfine campaign:

| Workload | N | Baseline ns/op | Candidate ns/op | Latency change |
| --- | ---: | ---: | ---: | ---: |
| Small/Interface/Nil | 189793337 | 8.292 | 7.1565 | -13.6939% |
| Small/Interface/OtherArena | 116079106 | 12.93 | 11.4 | -11.8329% |
| Small/Interface/Self | 185032895 | 8.3305 | 7.3100000000000005 | -12.2502% |
| Small/Irregular3/Nil | 167681034 | 9.5805 | 7.984500000000001 | -16.6588% |
| Small/Irregular3/OtherArena | 78079815 | 19.535 | 17.095 | -12.4904% |
| Small/Irregular3/Self | 158357772 | 9.7845 | 8.45 | -13.6389% |
| Small/Pointer/Nil | 181940701 | 8.363 | 7.159 | -14.3967% |
| Small/Pointer/OtherArena | 118213661 | 12.81 | 11.42 | -10.8509% |
| Small/Pointer/Self | 184375854 | 8.305 | 7.297000000000001 | -12.1373% |
| Small/Slice/Nil | 184653263 | 8.2735 | 7.174 | -13.2894% |
| Small/Slice/OtherArena | 117289314 | 12.79 | 11.41 | -10.7897% |
| Small/Slice/Self | 180240321 | 8.2895 | 7.2885 | -12.0755% |
| Small/String/Nil | 189208129 | 8.358 | 7.173 | -14.1780% |
| Small/String/OtherArena | 117903931 | 12.895 | 11.4 | -11.5936% |
| Small/String/Self | 185465037 | 8.312000000000001 | 7.3015 | -12.1571% |
| Large/External | 1000000 | 3874 | 3819 | -1.4197% |
| Large/Nil | 1000000 | 1429.5 | 1427 | -0.1749% |
| Large/Self | 1000000 | 1542 | 1427.5 | -7.4254% |

### Generated code and size

Scanner machine-code span: **620 -> 648 bytes (+28); frame reservation 0x70 -> 0x78 (+8)**. Whole `runtime.test` file size: **16,042,582 -> 16,042,398 bytes (-184)**. The complete executables use different untrimmed source roots and include debug/build metadata; whole-file differences are not attributed solely to this function.

The emitted amd64 code removes repeated extent subtraction/addition from the self/nil loop path and performs it at return. The larger frame reservation and function text are recorded costs. Pointer iteration and external marking dispatch remain unchanged.

### Correctness and admission decision

All four native amd64/arm64 x Green Tea on/off runtime jobs passed on the measured revision: build, formatting/vet, runtime regression, race, repeated boundary/lifecycle checks, noinline synchronous-close stress, and checkmark. The Green Tea jobs also retained the stability branch's exact-original ownership mutation result. The full upstream run failed shared prerequisites: a script timeout, ledger formatting, and an unsupported-host SIMD fixture where that script reached execution. The subsequent correction is tracked in #35 and #47; consult current PR checks for its result.

The large self-pointer benefit is corroborated at about 7.3% lower latency. None of the six hyperfine controls regressed in this run; large Nil is effectively flat. No combined result with the mask-initialization candidate is claimed.

The timing and hot-path checks pass. **Full G6 admission is still pending the caller-contract lower bound and full upstream parity.** A high scanner profile share does not establish that floor. The previous large Self/Nil proxy is not reused for a new workload; Nil still performs iterator work. CPU identification and BogoMIPS do not establish effective clock or latency bounds. This evidence supports a reviewable draft, not an unsupported whole-GC or application claim.
