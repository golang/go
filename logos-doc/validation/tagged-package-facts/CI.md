# Draft-PR acceptance gate

`.github/workflows/tagged-facts.yml` runs on draft pull requests when the compiler,
SIMD sources, this validation directory, or this workflow changes. It is separate
from the existing native workflow, which skips drafts. It does not require the
corpus runner, corpus manifests, external modules, or another workflow's files.

The workflow checks out the exact PR head with credential persistence disabled.
It uses the repository's pinned checkout and Go setup actions, Go 1.27.4, and
SHA-256-verified uv 0.13.8. Python tools have exact version pins in
`test-requirements.txt`. The job uses Ubuntu 24.04, `GOMAXPROCS=2`,
`GOMEMLIMIT=2GiB`, `GOGC=50`, and `GOFLAGS=-p=1`.

## Required sequence

1. Verify the two frozen new Go test files, the two original 48-case files,
   `old-optfacts.go.txt`, and `frozen.json` against fixed SHA-256 values in
   `check_ci.py`. Require the clean checkout to match the requested PR head.
2. Run the Python acceptance tests and strict type/style checks.
3. Build the candidate toolchain with the official Go 1.27.4 bootstrap.
4. Overlay only the archived old production file. Select
   `TestFactScopeTaggedFunctionAssertions/Project@simd128/mask=0` and require
   the named missing-metadata diagnostic, exit status 1, and the exact two
   run/fail events for the test and its root.
5. Run all selected candidate tests once, without `-short`. Require the exact
   150 named run/pass events: 142 leaf cases and eight nonleaf roots. This includes
   the original 48 cases, the new 92 unit cases, and both real SIMD archive cases.
6. Recheck source and tool identities after execution. Accept only if both
   phases passed their respective oracle and these identities are unchanged.

There are no implicit parent events for slash-separated strings passed to one
`t.Run` call. `testing.T.Run` creates one child `T` with the complete supplied
name; `test2json` preserves that name. The checker includes only actual calls
present in the frozen tests. Its Python controls reject invented intermediate
parents as unexpected events.

A compilation error, timeout, skipped test, malformed JSON event, build-failure
event, wrong package, nonempty stderr, duplicate event, missing event, unexpected
test name, or incorrect negative-control diagnostic fails acceptance. The
old-source control must fail because the named tagged function lacks known
assertion metadata, not because the test failed to build or run.

The old-control test timeout is 10 minutes, with a 15-minute process deadline.
The candidate test timeout is 30 minutes, with a 35-minute process deadline.
Expired process groups are terminated. The workflow has a 90-minute job limit
and separate limits on the build and gate steps. Partial evidence remains a
failure, not a completed acceptance result.

## Evidence

An always-run artifact step retains the evidence directory for 14 days:

- Source head, full tree, source tree, frozen hashes, workflow/helper hashes,
  and resource profile before the build.
- Bootstrap and installed command versions, build output, and verified uv
  archive receipt.
- Separate stdout JSON and stderr for each Go command, command arguments,
  exit status, timeout state, elapsed time, and log hashes.
- Source identities and SHA-256 values for installed `go`, `logos`, `compile`,
  and `link` before and after the gate.
- Per-phase acceptance status and the final acceptance result.

The helper creates new evidence files rather than overwriting an earlier run.
The final acceptance flag is false unless the complete candidate event set and
the expected old-source failure both satisfy the checker. Raw logs remain
available when either phase is rejected.

## Local validation boundary

The helper's 52 Python tests passed locally, including Hypothesis samples for
missing and duplicate events, exact positive and negative controls, invented
slash parents, malformed/build-failure streams, incorrect diagnostics, nonempty
stderr, a changed copy of every pinned source file, and retained rejection
status for malformed array/null output from a Python subprocess. Strict Pyright, Ruff,
format checks, and actionlint 1.7.12 passed. Local actionlint did not have
shellcheck available; the workflow's shell was also reviewed directly.

No local Go command, toolchain build, compiler test, or SIMD archive build was
run for this CI change. The required named old-source failure and candidate
passes remain hosted-CI obligations. The frozen Go tests and their original
evidence files are unchanged.
