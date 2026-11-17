# Ejection target facts

Issue: [#57](https://github.com/gosuda/LoGOs/issues/57).
Slice: M24-A-S2, REDTEAM #393, AUD-197.
Repair base: `153705a0134c32a6a27344f63bdf3cdddd50216d`.

## Invariant and boundary

An inactive file's selected build context must agree with the real target
settings passed to its helper. Architecture implications and target-specific
experiments are fixed facts. They cannot be toggled as custom tags. Explicit
user tags retain ordinary Go semantics, including a user-supplied tag whose
spelling matches an architecture feature.

The architecture-tag implementation is shared with `internal/buildcfg`'s
normal tool-tag path. Experiment facts use its existing `ParseGOEXPERIMENT`.
The selector keeps the Boolean constraint solver, complete helper tag set,
host-tag preference, supported GOOS/GOARCH check, and final `MatchFile` check.
Architecture settings are included in helper deduplication and subprocess
environment construction.

## Frozen failures

Both test files in `frozen-tests.sha256` were frozen before production edits.
Baseline component failures included:

- `amd64.v3 && !amd64.v2` was accepted by adding `amd64.v3` to user tags.
- `arm64.v9.1 && !arm64.v8.6` was accepted despite the required implication.
- A cross-target arm64 context retained `amd64.v1`; a 386 context retained
  the host's register-ABI experiment fact.
- Unsupported feature levels and `gccgo` could become synthetic user tags.
- The real helper still reported `GOAMD64=v1` and lacked `amd64.v2` while
  receiving a manufactured `amd64.v3` user tag.
- Valid `arm64 && !amd64.v1` and `386 && !goexperiment.regabiargs` contexts
  could not be selected.

The baseline command script failed at ejection because no context selected
`arm64 && !amd64.v1`. The explicit user-tag and baseline-level controls passed.

## Review correction: raw GO386

Review found that the ordinary tool-tag initializer receives an unvalidated
`GO386` string. The first repair wrongly treated every global architecture
setting as already parsed and could panic before `go env` or the compiler's
existing diagnostic ran. The pre-change driver and pristine Go 1.27.4 both
return `invalid` from `GOARCH=386 GO386=invalid go env GO386`; a build reports
`unsupported setting GO386=invalid` through the normal compiler boundary.

The added `feature_tags_logos_test.go` and
`logos_eject_target_env_compat.txt` were frozen against the first repair before
the correction. Both reproduced the unexpected initialization panic. Their
controls retain strict candidate validation, architecture-setting boundaries,
valid 386 modes, and the existing raw-env/error behavior of other settings.
The second production attempt shares one implication implementation while
preserving the ordinary raw-GO386 path and validating selector candidates.

## Executed validation

All runs use `-count=1`, GOMAXPROCS=2, package parallelism 1, and a 512 MiB Go
memory limit. The first production attempt passed the frozen feature tests,
the unchanged tag-solver tests, and existing `internal/buildcfg` controls.

The freshly compiled cmd/go script runner passed
`TestScript/^logos_eject_target_features$` in 47.905 seconds. This runs the
actual variant helper path and pristine Go 1.27.2 builds for amd64 v1/v3 and
arm64 v8.0/v9.1, and checks rejection of the contradictory feature constraint.
The eject package race gate passed in 5.782 seconds; internal/buildcfg passed
in 1.011 seconds. The independently frozen, still-unrepaired inactive-package
tests for issue #56 were excluded from this issue #57 package gate.

After the review correction, both packages passed race again in 5.795 and
1.011 seconds. The freshly compiled driver passed both target scripts in
0.774 seconds, including the raw-GO386 command and diagnostic checks.
Making the ordinary initializer strict again was caught by
`TestGogoarchTagsPreservesRawGO386`; the unmutated packages and vet then passed.

These component and freshly compiled driver results use the installed root
compiler through the isolated worktree's GOROOT. They are not a claim that a
complete new toolchain or the full upstream suite passed. The parent
stabilization branch owns those integrated gates.

The mutation disabled the selected architecture environment in
`spawnVariantEjects`. The frozen helper test caught the missing setting, wrong
ToolTags, and failure to select the source. The unchanged production source
then passed the same feature and tag-solver tests again in 0.047 seconds.
`go vet` passed both changed packages, `gofmt -l` was empty, frozen hashes
matched, and the embedded/JSONL ledgers matched with unique IDs.

Local complete logs are retained under `tooling-evidence/ejection` in the
stabilization workspace.

## Acceptance follow-up and manifest succession

The original `frozen-tests.sha256` remains unchanged. The later configured lint
gate found an unchecked cleanup error and fixed host parameters in a tuple
adapter used only by tests. The acceptance follow-up moves that adapter to
`eject_test.go`, names its Linux/amd64 host explicitly, and checks `RemoveAll`'s
error in the existing cleanup. The production selector, tested inputs, and
existing semantic assertions remain unchanged.

`frozen-tests-v2.sha256` records the mechanically updated target-feature test
and the new adapter location. The other three original entries are unchanged.
[Acceptance follow-ups](../acceptance-followups/README.md) records the exact
transformation proof and the actual lint/race/vet results. The historical
manifest remains the evidence for the original repair; the versioned successor
identifies the tests used by the follow-up.
