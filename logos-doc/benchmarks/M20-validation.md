# M20 validation follow-up

This follow-up keeps the four default-off cgo optimization candidates and repairs their validation. It changes no production compiler or runtime behavior. Issue #48 and PR #49 track this work.

## Rebuilt test configuration

The inherited `selfTestBin` called `cgoSpanToolFlags`, which deliberately disables fast-cgo for the span workload. A fast-enabled parent therefore inspected a fast-disabled child.

Test-only commit `aad4fbc7657b6ded06f72cd863e4b7f43401d091`, tree `e88ea33e85d14afd040135c1aad8c54270eccf1e`, adds an actual child-execution check and runs the existing structural tests in all four independent cgo flag combinations. A clean local toolchain build succeeded with stock Go 1.27.4 before the observed RED at 09:43:04 UTC on 2026-11-16.

The new test required `fast=true opt=true` and observed `fast=false opt=true`. OneTrampoline reported zero definitions instead of one; Structure and NoClockInFastBody found no fast trampoline disassembly. The inline control passed. The complete raw failure is in [M20-validation-red.jsonl](M20-validation-red.jsonl); its SHA-256 is `ecaa6ccd2300310d6b61d1e68ca70b84483d0f772dbc9c77ac2749a6a6fcb2bb`. HEAD, tree, tracked index, and frozen source hashes were unchanged throughout RED.

The repair rebuilds from the running binary's embedded GOEXPERIMENT and instrumentation settings. Explicit false race/ASan/MSan arguments prevent inherited GOFLAGS from adding another mode; recorded settings then restore the original mode. GOENV is disabled for the child build so an empty recorded experiment cannot pick up a different settings file. C compiler variables remain explicit in CI.

### First local repair results

| Configuration | Required top-level checks | Result |
|---|---:|---|
| Both experiments on | 5 | Passed |
| Cgo optimization only | 5 | Passed |
| Fast-cgo only | 5 | Passed |
| Both experiments off | 5 | Passed |
| Both on, race | 5 | Passed |
| Both on, cgocheck2 | 2 | Passed |
| Both on, ASan | 2 | Passed |

This table records the first repair published at `b9161ef1babd011d26a0ac1264b96b919d95f822`. All tests ran with `-count=1`. The existing fast-body disassembly subtest skips when fast-cgo is off; its parent still checks calls and results. Every required top-level check passed.

A helper-only mutation changed the GOEXPERIMENT switch key to an unmatched key. The new test then observed `fast=false opt=false`, as predicted. Restoration ran in finally; the original helper hash was restored and the test passed again. Frozen attack tests and the CI compatibility script remained unchanged. The repair's Go source is the same source used by these checks.

Clang and gccgo were not on the initial PATH, so the first repair did not cover MSan or gccgo generation. The cached gccgo frontend was later found and checked as recorded below. MSan still requires CI. The local compatibility result is not a completed full-regression claim.

## Shared-helper integration

Parent commit `77ffa86046903bffd86473a8484220325841240d` separates the span and fastcall entry points. This follow-up preserves that structure and moves complete build-setting replay into their shared helper. Recorded experiments are replayed before the caller's final fast-cgo token. Span workloads deliberately turn fast-cgo off; fastcall inspection matches the compiled parent's bit. Other recorded experiments and instrumentation remain intact.

The final helper was checked again with both span callers added to each selection:

| Configuration | Required top-level result |
|---|---|
| Both experiments on, normal | 7 passed |
| Cgo optimization only, normal | 7 passed |
| Fast-cgo only, normal | 7 passed |
| Both experiments off, normal | 7 passed |
| Both on, race | 7 passed |
| Both on, cgocheck2 | 4 passed |
| Both on, ASan | 3 passed, 1 failed |

This is 42 passing required executions and one failed required execution. The ASan configuration/inline/codegen checks passed. The span pointer fixture printed its success marker, then LeakSanitizer failed during shutdown. The unchanged failure output is in [M20-validation-asan.jsonl](M20-validation-asan.jsonl).

One diagnostic-only repeat retained leak detection and enabled `LSAN_OPTIONS=verbosity=1:log_threads=1`. It reported `Could not attach to thread ... (errno 1)` for five threads, followed by the LeakSanitizer fatal error. The attach operations were denied with EPERM. The fixture remains a failed local sanitizer check; its assertions and exit-status requirement were not weakened. A runner that permits LeakSanitizer thread attachment must complete this gate.

The new helper-only mutation again ignored the GOEXPERIMENT setting. The caller still supplied `fast=true`, but the child lost `opt=true`. The frozen test caught the exact `fast=true opt=false` mismatch. Restoration recovered the exact helper bytes and GREEN.

Two initial infrastructure failures are also preserved. A full home tmpfs first made module indexing report the package as missing; a command from the cmd module exposed the underlying ENOSPC error. After moving GOCACHE, a child Go command faulted while writing its telemetry mmap on that same full filesystem, before the C fixture ran. Moving XDG_CONFIG_HOME to the workspace corrected that path too. No indexing, telemetry, sanitizer, or test assertion was disabled.

Final off/on vet of the changed test package, gofmt, and diff checks passed. A cached GCC 14.2.0-19 frontend also ran TestCgoOptGccgoUnchanged and both gccgo/godefs package-contract comparisons successfully, without skips. This establishes generation compatibility; no full gccgo linking result is claimed.

The final Go-input aggregate is `767e3be86d094373c26bf8aeb0dad3bd5622171869c7241bb430ad53e397b2c9` across 13,251 files. Changes from the completed runtime matrix are confined to the two test-helper files. The frozen self-build attack remains Git blob `553d6b66f4c8cd4c11b5a69eadc9657d75bffae1`; production runtime and compiler sources are unchanged.

## Stronger span attack

The combined tree includes test-only commit `1c632c287c1eed95c4b0a08c6be7ee9bb509880c`. Its PrefixInterior case places an unpinned pointer before the passed address. Starting a scan at the argument would miss that pointer; scanning the complete containing object must reject it.

## Combined runtime and C checks

The integrated runtime candidates passed all 54 required top-level executions: five runtime tests and four actual-C/codegen tests in each of six configurations. Fast-cgo was disabled to keep these checks on the ordinary-cgo path.

| Cgo optimization | Normal | Race | cgocheck2 |
|---|---:|---:|---:|
| Enabled | 9 passed | 9 passed | 9 passed |
| Disabled | 9 passed | 9 passed | 9 passed |

No required test skipped. The span cases include the stronger PrefixInterior attack; the typed-address cases cover eligible values, pointer-bearing neighbors, and container boundaries. The evidence record names every test and records each command and output hash.

The first eight commands completed. The next shell process exited 137 during the disabled runtime/race command, with no buffered child output available. Its kill cause is unproved, and the interrupted attempt remains recorded. Only the four unfinished commands were rerun with build parallelism reduced from two to one; GOMAXPROCS stayed two. All four passed with the original tests and instrumentation.

Across the completed matrix, 13,251 Go, C, assembly, module, and version inputs remained unchanged, with aggregate SHA-256 `a420c41fe530ad594ef686cd69b5290f27a782156be60d2851b727de1e3c298c`. Production runtime blobs and frozen test hashes are recorded in [M20-validation-evidence.jsonl](M20-validation-evidence.jsonl). The subsequent shared-helper integration changes only test-build configuration; its caller checks are recorded separately.

## Benchmark recorder and format

The prior recorder exported to `hyperfine.json` and used `hyperfine` as the command-log prefix. The command logger consequently replaced each export with process metadata.

Both affected campaigns are invalidated:

| Run / job | Frozen work for the first row | Secondary Go medians, baseline / candidate / integration |
|---|---:|---|
| [43831729270 / 133188676400](https://github.com/gosuda/LoGOs/actions/runs/43831729270/job/133188676400) | 88,888,681 | 27.365 / 27.060 / 27.090 ns/op |
| [43832392680 / 133190928032](https://github.com/gosuda/LoGOs/actions/runs/43832392680/job/133190928032) | 89,398,006 | 27.095 / 26.795 / 26.815 ns/op |

These medians describe BenchmarkCgoOptNocallback only. The package-contract campaign never reached BenchmarkCgoOptLateContract. No primary timing sample survived, so no row receives a latency verdict. All secondary samples and artifact identities are retained in [M20-validation-evidence.jsonl](M20-validation-evidence.jsonl).

The original parent and package-contract full-dist jobs were later cancelled when their branches advanced. Their logs contained no test failure before cancellation, but none completed the full gate.

The follow-up preserves external repair `eb7aa44a0f804ffb993adc73a6ec7baa3d7966f4` unchanged. It separates `hyperfine-export.json` from command metadata and reads Hyperfine 2's per-measurement JSON. The exact schema is confirmed by the [official v2.0.0 exporter](https://github.com/sharkdp/hyperfine/blob/b33755fe1fca8bcc1f182c773ca3ab040bfe517d/src/export/json.rs) and its [integration snapshot](https://github.com/sharkdp/hyperfine/blob/b33755fe1fca8bcc1f182c773ca3ab040bfe517d/tests/snapshots/integration_tests__json_export_basic.snap).

Local verification used the actual Hyperfine 2.0.0 release archive with SHA-256 `38e59087bd65520930a13cf513d8a493b7043741e8ad62349a8a9929eb61bc5c`. A small serialization workload produced ten real samples. The new parser read them while command metadata remained separate; wrong units, a wrong primary metric, and the old schema were rejected. The 14 checker tests passed. This exercise verifies the recorder and format, not cgo performance.

Warmups, sample counts, frozen work, phase order, the one-second useful-work rule, the 20% noise limit, and both 1.05 ratios are unchanged. The statistics and acceptance functions are byte-identical to their earlier versions.

## Current package-contract CI

[Run 43842772742, job 133215508482](https://github.com/gosuda/LoGOs/actions/runs/43842772742/job/133215508482) passed the current S2/S3 focused gate: build, format/full off/on vet, all three mutation/restore pairs, off/on normal/race/cgocheck2 families and existing pointer checks, direct off/on ASan/MSan, and the stock fixture. The merge checkout and PR head have the identical tree `68be5a7a2cc68566436f18cc886ae923d43a08b6`. Artifact `14216032289` has SHA-256 `496f346bc4c674cca61c522ac252e0d21515470bff2b9f9e0184ead4228ebc6b`.

That result covers the return-frame and package-contract candidates. Combined runtime-candidate CI, full regression, and performance remain separate gates.

## Remaining admission

A fresh complete campaign must measure every target and reference row from a new baseline. Prior samples cannot be selected or reused to manufacture a pass. Full regression, required instrumented CI, profile admission, and a justified same-contract floor remain independent gates. No accepted speedup, G6 promotion, default enablement, or complete stability is claimed.
