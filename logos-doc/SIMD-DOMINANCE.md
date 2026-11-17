# SIMD feature dominance

## Contract and scope

A CPU-feature test must dominate every SIMD instruction that needs that
feature, including a zero initializer and an instruction whose result is a
scalar but whose input is a vector or mask. This contract applies to ordinary
Go and to loGo/s because they share the SSA backend.

Issue #51 exposed a zero initializer before an AVX feature test. The repair
preserves existing scalar optimization and ordinary dominance-aware CSE. It
makes no performance claim and adds no optimization flag.

This is a SIMD barrier based on checked SSA types. It does not introduce
general hardware-feature metadata for scalar instructions; the existing
upstream issue #34950 remains outside this repair.

## Two independent defects

`zcse` moves every eligible zero-argument value to the entry block. That is
safe for ordinary constants, but not for `OpZeroSIMD`: target lowering can
turn the zero into an instruction that needs the feature test. Later
`tighten` does not sink it into a deeper loop, so tightening cannot repair
this movement reliably. Sibling guarded blocks can also lose their guards
when their zero values are combined at entry.

`branchelim` can speculate short conditional arms. Its safety predicate did
not reject SIMD values or SIMD inputs. A zero followed by scalar lane
extraction can therefore fit the speculation budget and lose its guard.
Checking only the result type misses lane extraction and mask conversion.

The repair excludes `OpZeroSIMD` from entry-hoisting CSE and adds one shared
SIMD-type predicate to speculative execution. Both branch-elimination forms
use that predicate. General CSE may still reuse an existing dominating
value; it does not create a new definition above the feature test.

## Frozen attacks

The attacks were written before the repair. Their SHA-256 hashes are:

| File | SHA-256 |
| --- | --- |
| `src/cmd/compile/internal/ssa/zcse_logos_test.go` | `beac5ae975345cc4df77a4d2bf8fd94d0fb4ee8b409bc157fb4b3c2ade980f63` |
| `src/cmd/compile/internal/ssa/branchelim_logos_test.go` | `152eecfdb74ea8eb1bb29f279dff704f9f71e0dc4775434aa1ab05e61153be40` |
| `src/cmd/go/testdata/script/logos_simd_zero_dominance.txt` | `697209c8e3f160fe4cf9b6e8c4a0da4050e18580f3804e9c925099fdadadf54a` |

The SSA tests check valid graphs before and after optimization. They cover
128-, 256-, and 512-bit vectors in nested loops and sibling guards, vector
and mask results, and scalar results with vector or mask operands. Scalar
controls require ordinary constant sharing and conditional moves to survive.

The source script compiles explicit Go SIMD and loGo/s slice reductions at
`GOAMD64=v1`. Its disassembly checker builds a control-flow graph, removes
the feature-true edge, and rejects any reachable vector or mask instruction.
It also requires both a feature check and vector instructions, so a scalar
fallback cannot make the test pass. Runtime cases compare scalar oracles at
lengths 0, 1, 7, 8, 9, 63, 64, 65, and 129, with 0, 1, and 3 repetitions,
both normally and with AVX2/AVX512F disabled.

The disassembly checker was corrected in a test-only commit before the
production repair: Go can inline a CPU check as `CMPB` followed by `JE` or
`JNE`, while the loGo/s form can retain a call and result test. Both forms
must identify one feature branch. The correction changed recognition of
the guard, not the dominance requirement. The corrected checker observed
unguarded `VPXOR` and `VPXORQ` instructions in the baseline.

## Evidence recorded on 2026-11-16

The starting revision is `949e633789fe5122c548d6c9f39c06fbb2769702`,
the head of the separate host-capability fixture repair in PR #47.
PR #54 is stacked on that branch. Its first two commits contain only tests
and CI, preserving an independently reviewable pre-repair state.

The fork's baseline `zcse.go` and `branchelim.go` were byte-identical to the
pinned upstream Go 1.27.4 copies. An overlay onto that installed upstream
compiler source allowed the SSA tests to run while the fork bootstrapped.
No upstream source file was modified.

| Check | Observed result |
| --- | --- |
| Baseline SSA guard attacks | Failed: SIMD zeros moved outside their feature regions; branch elimination removed feature guards. Scalar controls passed. |
| Baseline installed-fork disassembly | Failed for Go and loGo/s at both widths: vector zero instructions were reachable without their AVX2 or AVX512F guard. |
| Candidate SSA guard attacks | Passed unchanged. |
| Restore old zero-CSE admission | Failed the zero-dominance attacks. |
| Restore old speculation admission | Failed the branch-elimination attacks. |
| Restore both repairs | Passed unchanged. |
| Installed fork, complete uncached SSA short suite | Passed in 84.341 seconds. |
| Formatting and SSA vet | Passed. |
| Race-enabled source and native/ejected script gate | Passed all six selected scripts in 45.737 seconds, including the new dominance script, three existing feature-gate scripts, ejected slice reductions, and README consistency. |
| Go and loGo/s source oracles | Passed both widths, normal execution, and execution with AVX2/AVX512F disabled. |
| Hosted focused workflow | Passed, including a fresh rebuild, formatting, vet, complete SSA short suite, and race-enabled source/ejection scripts (job 133294001990). |
| Hosted full regression, first run | Every reported package except cmd/go passed. cmd/go reached its 540-second outer deadline at 541.663 seconds: 593 scripts reported cancellation and four active commands received SIGQUIT. This is not a pass. |
| Local full regression | Unix-domain socket tests report `operation not permitted` in this environment. This is not a pass. |
| Hosted full regression, corrected outer budget | Pending; an incomplete result remains a merge blocker. |

These observations prove the two optimizer defects and the targeted repair.
They do not substitute for the full regression gate.
The initial local bootstrap attempts were killed under memory pressure;
using a disk-backed cache and temporary directory allowed bootstrap to
continue. The baseline and repaired toolchains both rebuilt successfully.
An initial script invocation put temporary outputs below GOROOT, where the
cmd/go test harness correctly rejects writes; the unchanged scripts passed
with a temporary directory outside GOROOT. Those environment failures are
not counted as semantic test failures.

The repository contains 1,444 cmd/go scripts, including 511 loGo/s scripts.
The first hosted full run used upstream's nine-minute cmd/go outer deadline,
while the existing dedicated language-script CI already budgets an hour.
The full workflow now sets the distribution harness's supported
`GO_TEST_TIMEOUT_SCALE=8`, giving cmd/go a 72-minute outer deadline within
the job's 90-minute limit. Test assertions and performance acceptance
thresholds are unchanged. The canceled run is retained as evidence, and a
completed rerun is still required.

Hosted evidence: [focused job](https://github.com/gosuda/LoGOs/actions/runs/43865827289/job/133294001990)
and [first full regression job](https://github.com/gosuda/LoGOs/actions/runs/43865827289/job/133294001373).

## Reproduction and merge gate

After `src/make.bash` rebuilds and installs the fork, put its `bin` first on
`PATH` and run from `src`:

```sh
go test -count=1 -short cmd/compile/internal/ssa
go vet cmd/compile/internal/ssa
go test -count=1 -race -timeout=40m -v cmd/go \
  -run '^TestScript$/^(README|logos_simd_(zero_dominance|feature_gate|gate_avx512|gate_branch)|logos_eject_simd_slice)$'
GO_TEST_TIMEOUT_SCALE=8 ./run.bash --no-rebuild
```

The source/ejection scripts use the repository's pinned upstream 1.27.2,
1.27.3, and 1.27.4 toolchains as independent oracles. Changed Go files must
also pass `gofmt`. In the combined stabilization candidate,
`simd-host-gates.yml` runs both independent source-dominance fixtures,
the complete SSA short suite, and the native/ejected controls.
The normal `logos.yml` workflow owns the full upstream regression job with
the same timeout scale. Queued or incomplete CI is not a pass and does not
permit merging.
