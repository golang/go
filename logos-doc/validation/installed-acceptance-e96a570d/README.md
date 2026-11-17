# Installed acceptance of the combined source

This record binds the second fresh local build to the exact combined source.
It does not inherit behavioral results from the earlier source-45 toolchain.

## Source and build

| Identity | Value |
|---|---|
| Local source commit | `e96a570da271c43a80e23bb1a188aac054acff25` |
| Identical GitHub source commit | `071705bf963bf54467e1ab9e3c39d78fca2253fb` |
| Complete tree | `0841c3d2a92b7d6091a5e248b82bd820326947a8` |
| Source tree | `e237c399aae8f8065b724e8a0e27585686c7da18` |
| Build command | `./make.bash`, from `src` |
| Start and end | 2026-11-16 18:23:09 to 18:44:39 UTC |
| Exit and duration | 0; 1290.122 seconds |
| Source before and after | Clean, exact same commit and trees |
| Acceptance cache at completion | Empty and separate from build cache |

The normal bootstrap, final PGO stage and dist cleanup ran unchanged. The
build used pinned stock Go 1.27.4, one worker, compiler concurrency one and
a 512 MiB soft Go heap limit. Resource observations are retained; the build
was not killed or shortened at a resource threshold. Both installed commands
report version Go 1.27.4 with the expected `go` and `logos` command names.

## Installed binary SHA-256

| Binary | SHA-256 |
|---|---|
| `bin/go`, `bin/logos` | `34a04896c6bdbedb2a8dbdb491b836b41e80efa47a255286733de021c776e6f9` |
| `compile` | `b8d0742e328093b87fec566a68fb810d7043446ce0b575a81f304432bee87b6e` |
| `link` | `0520fc2422f2d09ccb460c856d98f624164939d4e84ec78c8331e1d0afa59729` |
| `asm` | `cf19e608ea29badc7c52e96ccddc64fb736098547dbf3bf6848c7b27a588aa3a` |
| `cgo` | `323a1c76a840a00b36a0b0b1938ecd2b403bad5fea981cffa4f717c2e0a959bd` |

[The build receipt](build/result.json) records the inferred GOROOT and full
compiler build ID. [The input record](build/inputs.json), raw log, resource
samples, launcher and publication mapping are frozen by
[build-files.sha256](build-files.sha256). The source-publication record
describes object creation before the final branch update; it does not claim
that the PR ref moved at that earlier point.

## Behavioral acceptance

[Array-index acceptance](array-index/README.md) passed the complete lint
package with all 96 frozen cells, fixture completeness, both native errorcheck
fixtures and their eight diagnostics, the guarded-enum command script under
race, vet and formatting. The external reporting-condition mutation produced
both predicted false positives; the unchanged restored cells passed. All
source, frozen-input and installed-tool identities remained unchanged.

[Tooling and runtime acceptance](tooling-runtime/README.md) passed the installed
overlay-ejection race script (5.636 seconds), 25 linker top-level tests and 75
reported cells (0.912 seconds), normal linker vet and five native capability
scripts (87.423 seconds). Both native cgo paths executed, and the final linker
received the capability manifest. The five-second graph child limit stayed
unchanged. No selected test skipped.

The same final command test binary executed the arm64 sum program under pinned
QEMU and compared all 44 output lines to the native run (10.432 seconds).
The command and comparison are present in the transcript. The cross-target
phase did not force `CGO_ENABLED`. Final source, tests, six tools, retained test
binary and QEMU identities stayed unchanged.

[Final JSON acceptance](json-runtime/README.md) passes the complete runtime
race suite and normal runtime vet in both profiles. JSONv2 passes 141 top-level
tests and 4,268 PASS records without skips. Classic passes 128 of 129 started
top-level tests and 4,255 PASS records; its sole existing depth-bound skip is
recorded explicitly. The quoted-C test runs and passes under race with CGO
enabled in both profiles, without skips. All ten commands exit zero, and the
source, reviewed composition, frozen C fixture and installed tools remain
unchanged.

The broad main workflow, all six driver/optimization matrices, stamped parity
tier, real LSP/Wasm legs and second-host reproducibility remain separate
requirements. Their hosted outcomes belong to the immutable tip tested by
those runs, with this local source identity retained as a separate proof.

M6-S3b remains blocked under G1 item 7. This successful build does not establish
the unresolved default-application contract or authorize another binding
implementation attempt.
