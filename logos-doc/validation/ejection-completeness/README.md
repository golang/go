# Complete ejection inventories

Issue: [#56](https://github.com/gosuda/LoGOs/issues/56).
Slice: M24-A-S1, REDTEAM #392, AUD-196.
Reviewed source base: `153705a0134c32a6a27344f63bdf3cdddd50216d`.
The target-context prerequisite is M24-A-S2 with its GO386 review correction.

## Frozen regressions and fixture correction

The files in `frozen-tests.sha256` cover unloaded OS/architecture/custom-tag
packages, an entirely host-inactive module, inactive-only tests, dependency
module discovery, ownership exclusions, real declaration provenance, and
unique formatting of repeated variant outputs.

The original baseline omitted the Windows-only package in the real command
script. The unit census returned no unloaded packages. The owned-provenance
test rejected a declaration selected by a real alternate load inventory.

The first production attempt passed the complete inactive-package script,
including its pristine-Go dependency, cross-target, test-only, and empty-host
checks. It exposed two test setup errors: the existing
`ownedTestMainPackage` convenience helper appends `/sub`, but two new fixtures
wrote sources at the supplied root. The production delta was saved and
reverted. Only those fixture directory fields were corrected; all behavioral
assertions stayed unchanged. A separate test commit records that correction.

The same attempt also exposed repeated gofmt paths from several actual load
inventories. Four unchanged command scripts failed when parallel formatter
workers raced on the same output. A deterministic formatter-argument attack
was added while production remained reverted.

On the corrected pre-repair tree, all five component tests fail for their
predicted semantic reasons, and both real command scripts omit the expected
inactive consumer output. The final hashes were frozen at that point.
The initial owned-script spelling used an ordinary imported function where
the language requires a `const func`; that was corrected before any issue #56
production edit and before the semantic missing-output RED was accepted.

## Package-clause review and final repair

Attempt 2 passes the corrected component suite (0.615s), all seven initial
command scripts (11.559s), and the race suite (5.841s). Independent review then
identified a legal case absent from the original attacks: mutually exclusive
platform files may use different package clauses, and consumers may import
the common path under an explicit alias.

The separate `logos_eject_variant_package_names.txt` attack first builds its
source with `logos` under Linux and Windows. Both pass, including compile-time
assertions of the expected constant values. Ejection then fails:

```text
owned request package "windowsowner" does not match owner package "linuxowner"
```

`TestInactiveOwnedRequestPackageName` reproduces the selection failure directly;
its invented-name rejection control passes. A second independent attack,
`TestInactiveOwnedGroupsPackageNames`, reaches the collision phase directly.
It rejects two distinct platform package clauses for both declaring `Shared`,
while the same-package rejection control passes. The two new files were hashed
before attempt 3; none of the six earlier frozen files changed.

Attempt 3 selects the actual retained package inventory by import path and
package clause when the host clause differs. The original mismatch diagnostic
remains when no inventory matches. Collision groups use output directory and
package clause. Existing standard-library, module, source-selection, and
same-package collision checks remain active.

## Completeness and termination invariant

The parent independently inventories eligible loGo/s source directories in
the main module and currently known ejected dependencies. It respects nested
modules, module ignore rules, vendor/testdata, dot/underscore paths, and
symlink boundaries. A source already selected by the host load cannot become
an inactive fallback when its expected compiler output is missing.

Each missing source receives a real target context from the existing selector.
The existing helper loads those package roots and uses the ordinary compiler
and test-build phases. It returns its actual package source inventories and
raw staged output, including owned-request sidecars. These records extend only
emission and declaration provenance, never compiler roots. The existing module
writer emits the combined tree and rewrites the discovered dependencies.

The visited set grows monotonically by module-owned source and full target
context. A tuple can be requested once, and each completed batch must provide
every requested canonical staged file. Missing output or a repeated tuple
fails with the source name. Only modules from actual helper loads expand the
next census. The formatter receives each final output path once.

## Final local evidence

The commands below use the root's installed compiler with this worktree as
GOROOT, a fresh command test driver, `GOMAXPROCS=2`, `GOMEMLIMIT=512MiB`, and
`GOFLAGS=-p=1`. `GOTMPDIR` is a real directory outside workspace VCS ancestry.
Pristine output checks use Go 1.27.2. Tests use `-count=1`.

| Gate | Result | Local log |
|---|---|---|
| Complete eject package, attempt 3 | PASS, 0.499s | `inactive-attempt3.log` |
| Eight command scripts below | PASS, 8.465s | `inactive-scripts-attempt3.log` |
| Complete eject package with `-race` | PASS, 6.185s | `inactive-attempt3-race.log` |
| Complete eject package after the mutation probes | PASS, 0.526s | `inactive-attempt3-restored.log` |
| `go vet cmd/go/internal/logos/eject` | Exit 0, no output | `inactive-attempt3-vet.log` |
| `gofmt -l` on all changed Go files | No output | Checked before commit |
| Frozen regressions and embedded/file ledger parity | Eight hashes unchanged; ledgers identical with unique IDs | Checked before commit |

The command gate is:

```sh
go test -count=1 -short -parallel=1 cmd/go \
  -run 'TestScript/^logos_eject_(inactive_packages|inactive_owned|variant_package_names|target_features|target_env_compat|variant_tags|pr2_platform|test_files)$' -v
```

It covers the new complete-module and owned-provenance attacks, native aliased
imports, pristine output at the relevant OS/architecture/custom-tag settings,
and existing target facts, empty-tag helpers, platform files, and test-file
ejection behavior. The complete component and race gates also retain all
existing owned-helper rejection and target-selector controls.

Four isolated source overlays were deliberately wrong:

| Mutation | Frozen detection |
|---|---|
| Skip directories absent from the loaded package map | All three `TestInactiveCensus*` cases lose required sources |
| Remove formatter sorting and compaction | `TestInactiveVariantsFormatEachOutputOnce` observes repeated paths |
| Disable matching an alternate package clause | `TestInactiveOwnedRequestPackageName/windowsowner` rejects the valid loaded owner; invented-name rejection still passes |
| Collapse collision groups to directory alone | `TestInactiveOwnedGroupsPackageNames/distinct` reports a false collision; same-package rejection still passes |

The mutation commands fail for those semantic assertions. Production files
were never replaced by the overlays; the restored gates omit `-overlay`.
All eight frozen test hashes are checked after the final repair and probes.

## Validation boundary

Local complete logs and the saved first production attempt remain in
`tooling-evidence/ejection` in the stabilization workspace. The repair commit
records the unchanged-test GREEN, race/vet/format, and caught/restored mutation
results. This slice meets the three-attempt correction limit. Full toolchain
rebuilding and the complete upstream regression suite remain integration gates
owned by the parent stabilization branch; the local component and command
results do not claim those broader gates.
