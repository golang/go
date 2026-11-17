# External test package capability ownership

The purego linker hang repair exposed a second defect: the main module's
external test package was blamed as a dependency. A module root `m` owns
packages named `m` or `m/...`; the synthetic external test name `m_test`
matches neither. Conversely, a nested dependency's synthetic root name can
match its parent's main-module prefix and incorrectly become exempt.

## Frozen inputs and baseline

The source parent is the capability discovery repair, based on default
branch `9a628f5b87f3c69a392ba539d11c7c9a37b98566`. Its original ownership
code is unchanged at this baseline. The two attack files were written,
formatted, hashed, and run before changing the ownership implementation.
`frozen-tests.sha256` records their final unchanged bytes.

The unit test checks 15 cases: main roots and subpackages, ordinary and
nested dependencies, workspace precedence, missing or unrelated test
provenance, real packages whose names end in `_test`, exact-name bounds,
standard-library tests, command-line packages, and non-test binaries.

The command fixture contains actual Go modules. Its external and internal
test bodies compile and execute with `-short`. The process-launch call is
reachable in the compiled body, while short mode avoids launching it.

| Frozen native case | Baseline | Expected |
| --- | --- | --- |
| Main root external test | Exit 1, falsely denied | Exit 0 |
| Nested dependency external test | Exit 0, incorrectly exempt | Exit 1 with the dependency's denial |
| Real dependency named `app_test` | Exit 1 | Retain the denial |

`red.log`, `cli-red.json`, and the `*-red` streams preserve those failures.

## Repair and proof boundaries

The driver derives an origin only from a generated test main's own direct
external-test import: `ForTest`, the `_test` import path, and the `.test`
main path must agree. It adds that exact origin to the link action hash and
passes it as one internal linker argument. Plain `go` does neither.

The linker applies the origin only to its exact synthetic package name,
only in a test binary. Standard-library classification remains based on
object provenance. The existing module ownership rule then decides whether
the tested package belongs to a main module or a dependency. No suffix-wide
exemption, policy severity change, or allowlist is added.

Candidate one passed all unit ownership cases and repaired the main root.
The frozen native nested-dependency control remained red. A dependency's
test executable names that dependency as its build-info main module, so
it is absent from the build-info `dep` rows. The driver already sends its
ownership in `nested_modules`, but the linker did not decode that field.

Candidate two decodes that existing field and combines it with build-info
dependency roots. Longest-main-module ownership still takes precedence.
`nested-policy.json` retains the policy that established the missing input.
The two attempts use the same frozen tests; the correction budget was not
reset and no diagnostic expectation was weakened.

## Focused checks

Run date: 2026-11-17 UTC. Go 1.27.4, Linux amd64.
Commands used `GOMAXPROCS=2`, `GOMEMLIMIT=384MiB`, `GOGC=30`, `-p=1`, and
`-count=1`. Explicit source overlays redirected only the selected changed
files into the prior installed tree, preserving that checkout. The native
fixture uses a separately built candidate driver and a toolexec wrapper
that substitutes only the separately built candidate linker.

| Check | Result | Evidence |
| --- | --- | --- |
| Baseline ownership units | Four semantic failures | `red.log` |
| All capability and graph unit cases | Passed | `green.log`, `restored.log` |
| Candidate one native replay | Root repaired; nested control still fails | `cli-candidate1.json` |
| Candidate two native replay | Main root passes; nested and real suffixed dependencies denied; plain Go passes all three | `cli-candidate2.json`, `*-candidate2` |
| Broaden exact origin match to a prefix | Caught by suffixed child and longer-name controls | `probe.log`, `probe2.log` |
| Vet linker and cmd/go/work | Passed | `vet.log` |
| Build scoped driver and linker | Passed | `driver-build.log`, `link-build.log`, `link-build2.log` |

The candidate-two replay checks the exact `os.startProcess` diagnostic for
both forbidden dependency cases and rejects any capability diagnostic for
the first-party and plain-Go cases.

## Installed acceptance

The fresh combined `make.bash`, actual `cmd/go` script harness under race,
complete affected-package checks, and final purego FFI runtime replay are
pending at this component checkpoint. The overlay checks above do not
claim a fresh full-toolchain build or a full upstream regression pass.
Later acceptance evidence must name the integrated source revision and
installed binary hashes.
