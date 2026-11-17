# Preserve the inherited experiment contract

The implicit propagation change in `22ab6f7a169e905d9fde5e345db101735a321486`
was a regression. M18-S9 already says that user test, run, generator and
module-tool processes retain the caller's `GOEXPERIMENT`. Commit
[`fd146c5550`](https://github.com/gosuda/LoGOs/commit/fd146c5550fe49770e84eb11fb1bfb394d9be306)
also requires the test cache to distinguish an absent variable from a
present-but-empty variable.

The earlier M0-A-S14 tests assumed that a test process should receive the
experiment used to compile its binary. That assumption contradicted the
existing contract. Successful runs of those tests prove their former oracle,
not the inherited-environment behavior. The original files are retained in
`original/`; the earlier component evidence under `../test-experiments/`
remains unchanged and is superseded for this contract.

## Established failure

GitHub Actions
[run 43903529332, scripts job 133426172718](https://github.com/gosuda/LoGOs/actions/runs/43903529332/job/133426172718)
fails `TestScript/logos_test_cache_experiment` at line 14. After
`exec env -u GOEXPERIMENT logos test -v .`, the test prints
`GOEXPERIMENT present=true`; the unchanged fixture requires false.

`pr71-scripts-ci.log` is the decoded job log fetched directly from the
GitHub API. `baseline.json` records its run/job IDs and the Git blob IDs
of the immutable inputs. No unavailable local log has been reconstructed.

## Correction and frozen boundaries

The first correction commit changes only the newly introduced tests,
their contradictory design text, and this evidence. The separate production
commit removes the six lines that appended and hashed
`cfg.RawGOEXPERIMENT` for test processes. The resulting `logos_env.go`
must match the existing release blob
`cab15397914ffb0680488d9d61b83a8fde7d8bc7`.

The original `logos_test_cache_experiment.txt` stays byte-identical. It
already checks first execution, cache reuse and absent/empty distinction
under both command names. The three arena-off codegen fixtures cited by
M18-S9 also stay byte-identical.

The corrected unit matrix covers inherited absence, empty values, explicit
tokens, arena opt-out, duplicates and platform-specific name matching.
Its cache controls keep the effective experiment fixed when comparing
absent, empty and explicit inherited values. A separate check changes only
the effective compilation experiment and requires the getenv input hash
to stay unchanged. Compilation settings belong to the test binary's build
identity; the getenv input hash represents what the running test observes.

The native environment script retains its effective compiled-tag assertions
and checks the caller's value and presence in the user test. A plain-Go
child receives that inherited value. The plugin fixture and its assertions
stay unchanged. Its LoGOs invocation uses `-O=1` and explicitly supplies
`GOEXPERIMENT=logosarena`, so the plain-Go child builds with compatible
settings. The arena-disabled `GOEXPERIMENT=none` control remains.
Unconfigured mixed-driver plugin settings are not claimed compatible.

## Execution gates remain pending

The workspace became unavailable before corrected-test execution.
This branch records the existing CI failure and a source-level correction;
it does not claim a new build, GREEN run or mutation result.

Run with the freshly installed integrated toolchain first on PATH:

```sh
cd src
./make.bash

go test -count=1 -short cmd/go/internal/test
go test -count=1 -short -race cmd/go/internal/test
go vet cmd/go/internal/test

go test -count=1 -race -parallel=1 cmd/go \
  -run 'Script/^logos_test_(cache_experiment|experiment_env|plugin_experiment)$'

go test -count=1 -parallel=1 cmd/internal/testdir \
  -run '^Test$/^logos$/^codegen$/^(arena_place_off\.(go|lgo)|arena_frame_local_object_off\.lgo)$'
logos test -count=1 -parallel=1 cmd/internal/testdir \
  -run '^Test$/^logos$/^codegen$/^(arena_place_off\.(go|lgo)|arena_frame_local_object_off\.lgo)$'

./run.bash --no-rebuild -k
```

For the negative implementation control, overlay the archived
`original/logos_env.go` onto `src/cmd/go/internal/test/logos_env.go` while
keeping the corrected tests frozen. The inherited-environment unit suite
must fail. Remove the overlay and rerun it successfully. The original
cache script must also pass on the final installed driver. Record those
results against the final source and binary identities before marking
this slice tested.
