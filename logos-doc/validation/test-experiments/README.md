# Test subprocess experiment identity

`logos test` built its test binary with the effective arena experiment but
launched the binary with the original environment. A test that invoked the
same toolchain's plain `go` therefore lost the experiment. The environment
lookup used for test-cache hashing also read the original value.

The repair appends the effective `cfg.RawGOEXPERIMENT` only for logos test
processes and uses that value for their `GOEXPERIMENT` cache lookup. Plain-Go
environment handling, optimization flags, and all experiment defaults remain
unchanged.

## Test-first evidence

The baseline helper bytes match source base
`ec2f6dab` exactly. The unit attack reports seven missing or stale environment
values and an unchanged cache hash after the effective experiment changes.
The native environment fixture reports empty experiments in both the test and
its child. The corrected plugin fixture fails on the
`internal/goexperiment` package identity under matching O1 compilation.

The original plugin test had an additional, invalid expectation that a default
O2 logos host would accept a plugin built by the O1 plain-Go child. Production
was rolled back before a separate test-only correction. The
[correction record](TEST-CORRECTION.md), original frozen script, failure logs,
and three causal probes preserve that history. The correction changes the
host's declared optimization precondition; no plugin assertion was removed or
relaxed. The corrected tests were frozen again and observed RED before the
same environment repair was reapplied.

## Component results

All results below use the recorded isolated cmd/go overlay driver and the
existing installed compiler/runtime. This is component acceptance; a fresh
combined toolchain build and installed script/race gates remain pending.

| Check | Result |
| --- | --- |
| Complete `cmd/go/internal/test`, uncached | Pass |
| Complete package with the race detector, uncached | Pass |
| `go vet cmd/go/internal/test` | Pass |
| Seven native environment/profile invocations | All pass |
| Three native plugin invocations with matching O1 settings | All pass |
| Remove environment injection in a source overlay | Seven environment cases fail |
| Invert the cache-mode guard in a source overlay | Environment/cache cases fail |
| Restore source and compare corrected frozen test hashes | Pass |

The seven environment invocations cover logos defaults, ordinary Go defaults,
`none` with arena enabled, `nojsonv2`, explicit `jsonv2`, `none` with arena
disabled, and plain Go with `none`. They assert the actual compiled arena and
JSON tags, the test's environment, and a child `go env GOEXPERIMENT` value.

The three plugin invocations execute the plain-Go control, logos with arena,
and logos with explicit `none` and arena disabled. Each compiles a plugin in a
child process, loads it, checks the exported function type, and calls it to
require the value 42. No skips are present in those executed fixture functions.

The driver SHA-256 is
`831fcd79bc87bc2aaa0f9cb4fb25f2352b56611ff8b147ad6f0c9f92c867dbb5`.
`source-identity.json` records source and test hashes. `native-matrix.json` and
`plugin-matched-green.json` retain the exact commands, environment, durations,
exit codes, and log hashes. Absolute paths name the original run directories;
the corresponding logs are copied here under `native-logs/` and
`plugin-green-logs/`.

## Installed acceptance commands

After integrating the change, from `src/` with the rebuilt fork first on PATH:

```sh
./make.bash
go test -count=1 -short cmd/go/internal/test
go test -count=1 -race -short cmd/go/internal/test
go test -count=1 -race -parallel=1 -timeout=20m cmd/go \
  -run 'Script/^logos_test_(experiment_env|plugin_experiment)$'
go vet cmd/go/internal/test
```

The complete upstream `run.bash --no-rebuild` gate is separate. No complete
upstream pass is claimed by these component results.

## Mixed-driver plugin boundary

The upstream xxhash source stays unchanged. A default logos host uses O2 while
its plain-Go child uses O1. That unconfigured corpus cell remains a failure;
this experiment repair does not make mismatched compiler options compatible.
`src/plugin/plugin.go` documents the need for matching toolchain, tags, and
relevant flags and environment values.

Two explicit profiles establish the intended matching precondition:

```sh
logos test -O=1 ./...
GOFLAGS=-O=2 logos test ./...
```

The first selects O1 for the host to match the child's default. The second
selects O2 for both through the inherited flag. These are separate invocations
with recorded settings; they do not replace the original unconfigured result.
