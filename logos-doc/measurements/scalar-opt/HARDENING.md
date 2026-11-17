# Full-regression harness hardening

The [first PR #28 CI run](https://github.com/gosuda/LoGOs/actions/runs/43811707272)
passed the unit, scripts, testdir, and lint jobs. The regression job failed
in two packages. The complete gate remains required; these repairs preserve
its test selection and expected outputs.

## Isolate the wasm fixture environment

`cmd/compile/internal/logos` failed three wasm test families before their
programs entered `main`: the command line and inherited environment exceeded
Go's wasm startup buffer. `runWasmNode` reused the build environment helper,
which copied unrelated host and CI variables into the fixture.

The runner now passes only `GOMAXPROCS=1`. Its Node executable and launcher
paths are already resolved before process creation. The fixture programs
have no ambient environment inputs. The public wasm launcher and the
compiler's build environment are unchanged.

`TestWasmNodeIsolatesParentEnvironment` sets a 16 KiB sentinel in the parent.
Its program must run, report that the sentinel is absent, and observe one
processor. The unchanged attack failed before the repair with the same
startup-buffer error as CI. The attack and the three affected existing test
families then passed together in 14.975 seconds.

The mutation probe restored ambient inheritance. The attack failed again
with the startup-buffer error. Restoring the repaired helper byte for byte
returned the attack to green in 0.250 seconds. Frozen attack SHA-256:
`de7d159728ece17575e0f2b97abe3a75d6f6a90a152c2c6ad59fae7d3e2f433c`.

The complete toolchain rebuild passed. The compiler binary retained its
original SHA-256, confirming that this repair changes the test harness
rather than compiler production code. The four test families passed again
after the rebuild in 1.933 seconds; package vet, gofmt, and whitespace
checks also passed.

## Budget for the expanded command suite

The default `dist test` package budget is ten minutes. `cmd/go` reserves
ten percent of that deadline for script subprocess cleanup, which explains
the mass deadline failures around 542 seconds in the regression log.

The separate script job completed successfully in **1500.810 seconds**
under its explicit 60-minute package deadline. The regression job must
accommodate that same expanded suite. It now invokes:

```bash
GO_TEST_TIMEOUT_SCALE=6 ./run.bash --no-rebuild
```

This uses the existing upstream timeout-scaling mechanism. It neither
selects fewer tests nor changes their assertions. The workflow still has
its existing 90-minute job limit.

## Reproduce the focused check

With the built fork first on `PATH`, run from `src`:

```bash
go test -count=1 -short cmd/compile/internal/logos \
  -run '^TestWasm(NodeIsolatesParentEnvironment|ProgramRunsUnderNode|WrongMayYieldTraps|YieldFreeSourcesRunUnderNode)$'
```

The successful focused check does not establish a full-regression pass.
The CI run for the repaired commit supplies that result. This public record
contains summarized evidence and a frozen-source hash; private execution
payloads remain outside the PR.
