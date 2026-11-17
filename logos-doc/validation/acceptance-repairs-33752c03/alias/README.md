# Preserve alias errors at the package boundary

M0-A-S20 / REDTEAM #442 / AUD-210. One production candidate has been
implemented. This bundle preserves component RED, GREEN, a caught mutation,
and restored race GREEN. Fresh installed-driver scripts and the complete
regression gate remain pending. Nothing in this bundle assigns those passes.

## Source boundary

The candidate worktree was based on local
`2535486fe9596a9212d525a5007963ef9e8682f0`, whose compiler source tree is
`e237c399aae8f8065b724e8a0e27585686c7da18`. The component commands used the
previously installed e96 toolchain and explicit source overlays. They did
not replace the installed compiler or driver.

The installed driver SHA-256 in the raw bindings is
`34a04896c6bdbedb2a8dbdb491b836b41e80efa47a255286733de021c776e6f9`;
the installed compiler SHA-256 is
`b8d0742e328093b87fec566a68fb810d7043446ce0b575a81f304432bee87b6e`.
The recorded command, environment, source mappings, timestamps, runner hash,
and before/after bindings are retained without rewriting their original
workspace paths.

`candidate/tests.patch` adds the frozen new regression to the baseline.
`candidate/production.patch` is the sole production attempt and has SHA-256
`32735f3a77a2b5f9c33172707607f78074bf4714776c1fb12f10b9b847b9ad3a`.
It changes four files. `candidate/source-files.sha256` and its receipt bind
their candidate contents; `history/baseline-sources.sha256` binds the
original source contents. The historical preparation records precede the
root-owned formatting step and are preserved separately from the final
freeze.

## Frozen behavior

The new `TestScanModuleKindsDuplicateSiblings` has eight cells. Its source
SHA-256 is
`3a07ac609dc04cd1b0b250b0a366ae535ac25138dfb425365e67e101f992e19f`.
The cases call existing `ImportDir`, `scanModuleKinds`, and
`reportMixedModule` functions inside isolated subprocesses.

Five attacks require a classifiable duplicate sibling to contribute its
source kind without rejecting an unrelated package: canonical/alias,
alias/alias, variation-selector alias, external-test-only alias, and a
Go-selected module whose duplicate LoGOs sibling must still trigger
`mixed_module`. The other three cells retain malformed-header and
mixed-package failures and plain-Go alias exclusion. Normal package
selection must still return its original error.

The `frozen` directory contains the exact new test and all twelve unchanged
original script controls. `frozen-inputs.sha256` verifies those local copies.
The original `logos_alias.txt` keeps its invalid `dup` and `dup2` siblings in
place. No frozen assertion, skip, timeout, or invalid fixture was moved or
weakened. Inclusion of a script in this bundle does not claim it was rerun
against an installed candidate.

## Recorded execution

| Evidence directory | Result | Duration | Scope |
|---|---|---:|---|
| `execution/original-script-red` | Expected exit 1 | 0.415 s | Existing installed race test binary rejects the first positive `./scroll` build because of unselected `example.com/m/dup` |
| `execution/alias-new-attack-red-001` | Expected exit 1 | 20.411 s | Baseline production fails all five attacks for duplicate-census errors; all three controls pass |
| `execution/alias-component-green-001` | Exit 0 | 7.747 s | All eight unchanged cells pass with candidate source overlay |
| `execution/alias-kind-mutation-001` | Expected exit 1 | 1.983 s | Skipping the duplicate directory loses its source kind and suppresses the required mixed-module error |
| `execution/alias-restored-race-001` | Exit 0 | 20.534 s | All eight unchanged cells pass under race with the candidate overlay restored |

The component records explicitly reject build failures and skipped test
markers. Raw stdout and stderr hashes were checked before copying. The
mutation existed only in a separate overlay source file. The candidate
source and frozen tests remained unchanged; the restored run selected the
original candidate overlay. `mutation/kind-loss.patch` makes the mutation
boundary reviewable without installing or reproducing it.

## Design and remaining gates

The shared typed duplicate-stem error preserves diagnostic text and records
that file discovery and source-header checks completed. Only the module
census tolerates that package validation error, and it still accounts for
every admitted file kind. Normal package loading remains invalid. Malformed
headers, mixed packages, module ownership, tags, overlays, vendor, nested
modules, and plain-Go behavior retain their existing policy.

The required remaining installed script selector is
`^TestScript$/^logos_(alias|module_census.*|nomix|interop|lint_alias_extension|extension_normalization|module_zip_alias|plain_lgo_selection)$`.
Changed-package tests, formatting, vet, the fresh toolchain build, and the
complete regression gate must be reported against their actual source and
tool identities. The later combined build is distinct from the component
proof preserved here.

`spec-table-t-draft.md` contains wording proposed for root review. It is not
an assertion that the normative documents were changed or that pending
gates passed. `bundle.json` is the exact-copy inventory. `files.sha256`
seals the portable bundle. No binaries, caches, live temporary modules, or
unrelated external corpus are included.
