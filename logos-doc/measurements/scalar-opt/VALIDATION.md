# Scalar optimization validation

**The focused correctness and integration gates passed. The full upstream regression floor is incomplete, and CI is pending.** This record covers the final compiler, including rotate projection, against base revision `b0bbbc05d5a7dee54e6501415163e78290f72231`.

Focused integration completed with exit status 0 on **2026-11-16 at 04:46:38 UTC**. The full regression run was interrupted at 05:18:57 UTC after encountering the failures described below. Test durations are reported package durations, not performance measurements. Performance results are separate in the [measurement report](README.md).

The public [validation.json](validation.json) retains outcomes, hashes, probe results, portable reproduction commands, and summarized failure evidence. Full raw execution artifacts remain outside the public PR. Log filenames below identify those retained artifacts; their contents and original environment/command captures are not embedded in the public record.

## Focused results

| Gate | Result | Evidence record |
|---|---|---|
| Complete toolchain build with projection | Passed; installed toolchain at 04:30:41 UTC | `build-projection.log` |
| Full SSA package, uncached, short | Passed, 74.944 s | `green-final-unit.log` |
| Optimization settings package | Passed, 0.021 s | `green-final-unit.log` |
| Work package with matching temporary roots | Passed, 0.041 s | `green-work-final.log` |
| Runtime test, both flag values with SSA checking | Passed, 1.872 s | `green-runtime-final.log` |
| Disabled-feature codegen control | Expected behavioral RED, 187.529 s | `red-codegen-corrected.log` |
| Enabled-feature codegen, Go and loGo/s | Passed, 1.660 s; all declared amd64 v1–v4 and arm64 v8.0/v8.1 targets | `green-codegen-final.log` |
| Optimization scripts | All five selected scripts passed, 229.432 s | `green-script-final.log` |
| Selected standard packages with feature enabled | All nine passed with `-gcflags=all=-d=logosopt=1` | `green-stdlib-opt.log` |
| SCCP source probes with SSA checking | Compilation passed for linux/386, linux/arm64 and js/wasm | `green-sccp-386.log`, `green-sccp-arm64.log`, `green-sccp-wasm.log` |
| Vet | Passed for all five selected packages | `green-vet.log` |
| Diff check | `git diff --check` passed | Successful integration command completion |
| Settings/work race tests | Earlier pass, 1.030 s / 1.179 s, at 03:26:47 UTC | `green-race.log` |

The race result predates projection; projection does not change the settings or work packages. Cross-target source probes establish compilation, not execution on those targets. Empty probe/vet logs count with the observed successful command completion, not from their emptiness alone.

The combined final unit command **failed** in `cmd/go/internal/work`; its successful SSA and settings lines do not make that combined command a pass. The failure is preserved in `unit-split-temp-roots-failure.log`. The separate successful work rerun supplies that package's result.

## Full regression floor: incomplete

`./run.bash --no-rebuild -k` started at **05:00:35 UTC** and was interrupted at **05:18:57 UTC**, exiting **130**. Before interruption, **203 package suites reported success**. This is not a passing full regression gate. `full-regression-floor.log` records two distinct failures:

- `log/syslog` could not create UNIX stream/datagram sockets: `socket: operation not permitted`.
- Compiler processes compiling `runtime` and `net` reported `signal: killed`. The socket diagnostic does not explain these killed processes.

The socket restriction was reproduced independently with unmodified upstream Go 1.27.4 using `go test -count=1 -short log/syslog -run '^TestWithSimulated$'`. It exited **1**, with a reported package duration of **0.405 s**. Both `TestWithSimulated/unix` and `/unixgram` failed at `syslog_test.go:152` with the same operation and error text; only generated temporary directory names differed. All five `log/syslog` Go source/test files have identical SHA-256 hashes between the fork and upstream. Records: `upstream-syslog-environment.log` and `upstream-syslog-source-hashes.txt`. This confirms the socket restriction only.

The [CI workflow](../../../.github/workflows/logos.yml) adds a `regression` matrix entry that builds the toolchain and executes `./run.bash --no-rebuild` on its runner. **That CI result is pending.** No successful full-floor result is claimed.

## Behavioral RED provenance

Historical full command lines for the earliest RED runs were not retained reliably. These records identify the actual output and test families. Commands below are reproduction instructions, not reconstructed historical argv.

| Surface | Accepted evidence | Observed missing behavior |
|---|---|---|
| Correlated phis | `red-ssa.log` | Equal edge-aligned arithmetic/comparison results remained operations; predecessor counts 7 and 8 did not fold. Negative controls produced no failures. |
| SCCP extensions | `red-ssa.log` | Bit-length/popcount, selections, boolean transfer and enabled-gate cases did not propagate expected constants. |
| Carry and rotate known bits | `red-knownbits.log` | Arithmetic domain/extreme cases and rotate facts remained unknown; masked arithmetic did not become constant true. |
| Tier scope | `red-settings.log` | O3 in Go mode, logos mode and configuration omitted `logosopt=1`. |
| First valid source codegen control | `pre-projection-validation/red-codegen-corrected.log` | With `-O=3 -d=logosopt=0`, required constants were absent and forbidden arithmetic/rotate instructions remained. Duration: 1.569 s. |
| Rotate projection matrix | `red-projection-corrected.log` | Width/count matrix, packed-header case and zero-count cycle still returned shifts instead of the input. SSA validation ran before and after the pass. Duration: 0.161 s. |
| Projection source fixture against the installed preprojection compiler | `red-projection-codegen.log` | `projectedHeader` retained `ROLQ`/`RORX` plus `SHRQ` on amd64, and `ROR` plus `LSR` on arm64. Both source kinds failed. Duration: 2.473 s. |

Invalid attempts are retained and excluded from behavioral RED claims: `red-codegen.log` had comma-separated asmcheck patterns; `red-codegen-invalid-harness-flag.log` passed unsupported `-gcflags` to the harness and encountered VCS errors. The corrected control puts `-d=logosopt=0` in a temporary fixture's asmcheck header and rejects syntax, flag, build and killed-process failures. The first `red-projection.log` was superseded after adding scalar-constant normalization and SSA validation, before projection implementation.

Initial full SSA/settings GREEN is in `green-unit.log` (76.910 s / 0.020 s). Projection GREEN is in `green-projection.log` (0.209 s). Final installed-compiler results supersede those earlier passes. Earlier integration logs retained under `pre-projection-validation/` remain historical evidence; their copied modification times do not date the original executions.

## Frozen tests

Retained formatted snapshots byte-match the final files below. The SSA filenames are under `src/cmd/compile/internal/ssa`; the settings file is under `src/cmd/go/internal/logos/settings`.

| Test file | SHA-256 |
|---|---|
| `ssa/known_bits_logos_test.go` | `edf03f1ec16955b519aad3115adc726a3c0312c9967809dd1a6e5fdca95ecf40` |
| `ssa/correlated_constants_logos_test.go` | `5a1365cda15074d41f6151b6bc70b0fa385f4d302b19ac9427e62a06dbd81671` |
| `ssa/sccp_logos_test.go` | `edbf0a7b78de74958f7c8cf6816c26c2fb86e7196474c58822b51ea3349b2969` |
| `ssa/known_bits_projection_logos_test.go` | `80472041332c9693d0555e6e96a6236051f1828f64bf3f5f87490ac641fe191f` |
| `settings/scalaropt_test.go` | `7a083b53b73beaeb0221b958e286ce5f6d59513cf11c206f3b343c34ea8380f5` |
| `test/logos/codegen/scalar_opt.go` and `.lgo` | `628d42563cf979f1b331cc09348c66a051af8f9a89641295706a88e716e3f4a3` |

The first codegen snapshot has hash `efe4cc921c2e8fcbbfcf2e4de2804fd52a1ed11ab3b5dd476848afe3f89d9190` for both source kinds. The final snapshot only appends `projectedHeader`; existing checks are unchanged. Earlier formatting is not a behavioral test change.

## Mutation probes

| Deliberate defect | Test that caught it | Restoration evidence |
|---|---|---|
| Drop the input carry from known-bits addition | `TestLogosKnownBitsArithmeticExtremes` fails subtraction/negation boundaries | `mutations-independent/restored-green.log` |
| Remove the same-block requirement for correlated phis | `TestLogosCorrelatedConstantsNonlocal` observes an invalid constant 4 | Same restored run |
| Reverse SCCP's constant-condition selection | `TestLogosSCCPSelections` observes the wrong selected value | Same restored run |
| Remove projection's requirement that high bits be known | `TestLogosKnownBitsProjectionGuards` fails unknown-high and one-unknown-high cases across widths | `mutation-projection/restored-green.log` |

The first three independent defects were compiled together to bound host memory use, with a distinct targeted test for each mechanism. The probe script restored all three files, verified their exact bytes with `cmp`, and reran the selection successfully (0.017 s). Projection followed the same restore/check discipline; all projection tests then passed in 0.174 s. Probe outcomes and restoration results are summarized in [validation.json](validation.json).

An earlier mutation attempt was killed while compiling the test binary. `mutation-summary.log` correctly records that it did not catch a behavioral failure. Only the later successful probes count.

## Environment and reproduction

The successful final build used upstream Go 1.27.4 as bootstrap, `GOMAXPROCS=2`, `GOFLAGS=-p=1`, `GOGC=40`, and `GOMEMLIMIT=900MiB`. An earlier killed build is preserved in `build.log`; successful earlier and final builds are in `build-serial.log` and `build-projection.log`.

The final combined unit run used `GOMAXPROCS=2`; runtime/integration and the work rerun used `GOMAXPROCS=1`. They used fork `GOROOT`, `GOTOOLCHAIN=local`, a task-local Go cache, and the fork's `bin` first on `PATH`.

The failed work run used distinct `GOTMPDIR` and `TMPDIR` roots. [testing.TempDir](../../../src/testing/testing.go) creates fixtures under `GOTMPDIR`, while [isLocalLinkerPath](../../../src/cmd/go/internal/work/security.go) recognizes the root from `os.TempDir`. The linker-security fixtures fell outside the expected local roots. The successful rerun used `bin/go test -count=1 -short cmd/go/internal/work` with both roots identical, without a source or test change.

For portable reproduction, start at the checkout root and set `UPROOT1272`, `UPROOT1273`, and `UPROOT1274` to installed upstream toolchain roots. The first two are needed by eject checks; 1.27.4 is the bootstrap and behavior oracle. The commands use a fresh common temporary root:

```bash
checkout="$PWD"
export UPROOT1272 UPROOT1273 UPROOT1274
validation_tmp="$(mktemp -d)"
export GOCACHE="$validation_tmp/cache"
export GOTMPDIR="$validation_tmp/tmp" TMPDIR="$validation_tmp/tmp"
mkdir -p "$GOCACHE" "$GOTMPDIR"
export GOTOOLCHAIN=local GOFLAGS=-p=1 GOGC=40 GOMEMLIMIT=900MiB
cd src
env -u GOROOT GOROOT_BOOTSTRAP="$UPROOT1274" GOMAXPROCS=2 ./make.bash
export GOROOT="$checkout" GOMAXPROCS=1
export PATH="$checkout/bin:$checkout/lib/wasm:$PATH"

go test -count=1 -short cmd/compile/internal/ssa cmd/go/internal/logos/settings cmd/go/internal/work
go test -count=1 cmd/compile/internal/test -run '^TestLogosScalarOptimizationRuntime$'
go test -race -count=1 -short cmd/go/internal/logos/settings cmd/go/internal/work
go test -count=1 cmd/internal/testdir -run '^Test$/logos/codegen/scalar_opt' -all_codegen
go test -count=1 cmd/go -run 'Script/^logos_opt_(scalar|levels|go_package|level_values|scope)$'
go test -count=1 -short -gcflags=all=-d=logosopt=1 math/bits math/big strconv bytes strings encoding/binary encoding/asn1 hash/crc32 hash/maphash
go vet cmd/compile/internal/base cmd/compile/internal/ssa cmd/compile/internal/test cmd/go/internal/logos/settings cmd/go/internal/work
git diff --check
```

The runtime parent invokes the fixture twice with `-count=1` and package-scoped `-d=logosopt=0,ssa/check/on` or `-d=logosopt=1,ssa/check/on`. It checks scalar and processing-loop parity, dynamic negative controls, nil/index/division panics, and boolean evaluation effects.

The full-floor command clears target, experiment, debug and inherited build-flag overrides. Its local interrupted outcome remains the one reported above:

```bash
env -u GOROOT -u GOFLAGS -u GO_GCFLAGS -u GOOS -u GOARCH \
  -u GOEXPERIMENT -u GODEBUG GO_TEST_TIMEOUT_SCALE=3 \
  ./run.bash --no-rebuild -k
```

The upstream socket diagnostic uses the same cache, temporary roots and resource settings, with the upstream executable and `GOROOT` selected explicitly:

```bash
GOROOT="$UPROOT1274" "$UPROOT1274/bin/go" \
  test -count=1 -short log/syslog -run '^TestWithSimulated$'
```

## Historical input retention

A task-cache cleanup on 2026-11-16 removed imported package archives used by
the original compiler-cost campaigns. The 04:46 campaign had recorded hashes
for 81 unique archives. Those archives have not been restored: its exact driver
was not retained, and no assembler fingerprint was recorded. The 03:43 campaign
also lost its importcfg-referenced archives, which were not included in that
campaign's original input-hash manifest.

Source files, the archived compiler, import configurations, manifests, measured
benchmark executables, profiles, logs, and numeric observations remain. The
measurement-time validation is unchanged, but these earlier compiler workloads
are not currently reproducible from a complete retained set of imported archive
bytes. Historical hashes and results were not rewritten to hide that limit.
The later follow-up's 81 archives were restored and verified separately, as
recorded in [FOLLOWUP.md](FOLLOWUP.md#input-retention-audit).
