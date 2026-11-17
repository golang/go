# Installed acceptance for source 33752c03

## Tested identity and scope

This packet preserves one normal fresh build and **19 completed installed
gate records** for the combined source tree
`33752c03819ed5afcb37f87633a3cac668e181a8`. At execution, the reviewed index
was `098de80a4ddf1e59b2750c1306ef495414f42d1e`, based on HEAD
`2535486fe9596a9212d525a5007963ef9e8682f0` with the recorded staged changes.
It was a frozen **uncommitted combined snapshot**. The build and gates do
not prove any intermediate slice commit or a clean published release tip.
Each original receipt preserves that exact before/after boundary. Later
documentation edits or commits do not retroactively change it.

The second fresh `src/make.bash` completed with exit 0 in 480.027 seconds.
Its complete log, source identities, six installed tool hashes, timestamps,
environment, resource samples and wrapper are under `builds/`. Normal final
compiler PGO and dist cleanup remained enabled. GOMAXPROCS was 2; both
compiler phases retained their recorded `-c=1` flags. The record explicitly
states that make/dist clear GOFLAGS, so a recorded `-p=2` is not represented
as their scheduling cap. The new postbuild cache was empty when make ended.

The first make has no completion receipt. Its interrupted log, resource
samples, start/input records and interruption observation remain intact.
The termination cause is unconfirmed. The restart preparation records the
bounded concurrency change and removal of reproducible intermediate build
outputs. No implementation or attack changed between the two invocations;
the interruption is not another implementation candidate or a passing build.

## Actual installed outcomes

Every recorded expected outcome matched. The void-hook probe intentionally
exits 1; all other commands exit 0. Duration includes each complete recorded
command, not only the test function's internal timer.

| Gate | Exit | Seconds |
|---|---:|---:|
| alias-race-001 | 0 | 69.663 |
| alias-scripts-race-001 | 0 | 75.578 |
| cgo-ejection-jsonv2-race-001 | 0 | 49.336 |
| cgo-quoted-nojsonv2-race-001 | 0 | 71.497 |
| cgo-type-const-001 | 0 | 8.343 |
| driver-go-help-001 | 0 | 0.240 |
| driver-logos-help-001 | 0 | 0.192 |
| gofmt-001 | 0 | 0.294 |
| loader-packages-race-001 | 0 | 38.255 |
| normal-vet-001 | 0 | 17.466 |
| panic-compile-001 | 0 | 13.373 |
| panic-native-001 | 0 | 0.329 |
| panic-restored-001 | 0 | 0.312 |
| parity-short-race-001 | 0 | 72.524 |
| projection-generate-001 | 0 | 29.531 |
| projection-scripts-race-001 | 0 | 48.781 |
| types-packages-001 | 0 | 21.257 |
| void-hook-probe-001 | 1 | 0.351 |
| void-mutant-compile-001 | 0 | 84.843 |

The alias gate executes all eight frozen subcases under race detection.
Its separate command gate executes all twelve selected scripts. The
projection command gate executes all nine selected panic/C/ejection
scripts, including their frozen upstream checks. The four complete short
loader package race suites pass without skips.

The complete short `go/types` and `cmd/compile/internal/types2` suites pass.
Each has exactly four existing skips: `TestFixedbugs/issue78346.go`, whose
files are excluded by its build tags, and the short-mode `TestStdlib`,
`TestStdTest` and `TestStdFixed` gates. `TestGenerate` and the existing
`TestLogos/narrowing.lgo` fixture actually execute and pass. The public
`go/types/builtins.go` and `go/types/typestring.go` bytes retain their
pre-repair frozen hashes.

C source emission is checked under explicit jsonv2 and nojsonv2 profiles
with CGO enabled and race detection. The selected tests execute without
GoBuild or CGO skips. Installed acceptance clears GO_GCFLAGS and
BOOT_GO_GCFLAGS after make. Normal vet completes without diagnostics.

## Native void-hook mutation

One separately built diagnostic compiler removes exactly the call to
`check.logosRecordVoidPanic(call)` through an external overlay. No installed
compiler or production source is replaced. The mutation's source, overlay,
preparation and goroot layout description are retained; its executable and
symlink tree are excluded.

The compile receipt hashes the assertion binary. Native, diagnostic and
restored runs all freeze that same binary and compile receipt. The mutant
fails exactly the two real-builtin void-function cells with
`nil_deref_unproven` and no object output. The other seven controls pass.
Restoring selection to the original installed compiler passes all nine.
Preparation-time pending fields in the original layout/prepared records
remain historical; the later successful build and run receipts establish
the completed outcome.

## Short parity execution and its limits

The complete short race package run passes in 72.524 seconds. The queue
suite proves the N-1/N/N+1 boundary, cross-row sharing, three seconds of
queue wait outside a two-second child budget, and permit return after
success, three missing-executable errors and three ready-child timeouts.
The unchanged budget test and all three process/generator cleanup controls
execute and pass.

The parity corpus runs **31 programs through three fork configurations and
two upstream rows**. The failure corpus and normal ledger-row validation
pass. All nineteen mutant-definition subtests pass, and sweep discovery
reports 998 eligible files. Those definition/discovery checks are not
execution of the release mutants or the upstream sweep.

The only parity skips are exactly `TestParityMutants`,
`TestParityUpstreamSweep` and `TestRedteamLedgerRelease`. Each requires
release mode. This short run requests no stamp and proves no clean-tip
release. Full unfiltered, non-short, oracle-on release parity, actual
mutants, actual upstream sweep and its unchanged clean-tip stamp remain
required.

## Remaining acceptance boundaries

- The complete plain-Go `run.bash` regression on the repaired source is not
  part of these focused local commands.
- SWIG-dependent full-regression cases have not been established by this
  packet. The focused cgo results do not establish SWIG generator execution.
- Other release optimization, second-host reproducibility and guarded
  runtime obligations require their own actual results. Existing blocked
  language work and release exceptions are unchanged.
- Actual runtime proof here is native Linux. The independent source review
  records existing Windows harness assumptions and the new fixture's
  suffix/error mismatch. No blanket skip was added. No Windows or Plan 9
  execution result is claimed, and the broader Go-semantic and Windows
  driver-installation contracts are not waived.

## Contents and integrity

`gates/` holds all nineteen original directories: inputs, result, stdout,
stderr and resource samples. `specs/` holds the exact commands and selected
environments. `recorders/` contains the final installed recorder and both
imported modules. `builds/` retains both complete available make records,
their exact wrappers and restart preparation. `provenance/` holds the
integration manifest, binding review and pinned toolchain identities.
`reviews/` preserves all four independent review checkpoints, including
the source-only portability and contract-scope notes. Historical pending
states and raw logs are never rewritten.

All payloads are regular files. No executable binary, cache, active
worktree, diagnostic goroot symlink or archive is included. `source-map.json`
maps every copied file to its original absolute path and hash. Original
absolute paths remain in the raw records and overlays; replay must create
new path bindings for its own checkout rather than edit historical proof.
`results.json` is a derived index, not a substitute for the raw streams.

From this packet directory, `sha256sum --check --strict files.sha256`
verifies every payload. `manifest.json` also covers that checksum list.
The manifest itself is identified by the handoff SHA-256 outside the packet.
