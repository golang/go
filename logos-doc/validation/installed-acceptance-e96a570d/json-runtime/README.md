# Installed JSON acceptance at e96a570d

Both JSON profiles passed the assigned installed gates on 2026-11-16:
the complete runtime package under race, normal runtime vet, and explicit
execution of `TestEjectQuotedCgoTypes` under race with CGO enabled. Every
command exited 0, without a timeout or interruption. No overlay, source
change, frozen-test change, or new mutation was used.

## Exact source and tools

The tested commit is `e96a570da271c43a80e23bb1a188aac054acff25`, full tree
`0841c3d2a92b7d6091a5e248b82bd820326947a8`, compiler source tree
`e237c399aae8f8065b724e8a0e27585686c7da18`. Root's normal fresh make passed
before these tests. Its installed identity and the expected JSON composition
record are copied under `provenance/`.

Installed go/logos SHA-256:
`34a04896c6bdbedb2a8dbdb491b836b41e80efa47a255286733de021c776e6f9`.
Installed compiler SHA-256:
`b8d0742e328093b87fec566a68fb810d7043446ce0b575a81f304432bee87b6e`.

Before and after bindings verify a clean exact commit, all six installed
driver/compiler/tool hashes, and these four reviewed composition inputs:

| File | SHA-256 |
|---|---|
| `stream_v2_test.go` | `08729986d528dd8a0c003893b0069639b5f3d7201b122a4105be162bfa088232` |
| `scan.go` | `a4fe21da48eea8ce7ff96faed3ff4ec10d0685f98930a73cf83cd51798226854` |
| `path_v1.go` | `d07f9a34d01110e84a2bf13138af6e23c9ff8a7d71baa0b1b3ba1131b63ddd5c` |
| `path_v2.go` | `937c332fdc99c1f5f28b2879495a6fc2dcf2baad67019e9af83c2ee28c6a00de` |

The quoted-C fixture also remained byte-identical, SHA-256
`bc1903980a8fa11ad0724c8eada997d0b5bdbfb5713efbe46d27cffdacac3084`.
The before and after records match exactly.

## Observed results

All durations below are outer command times in seconds. Compilation and
execution are separate records, so test time is not confused with building
the race test binary.

| Profile | Runtime compile | Runtime race run | Quoted-C compile | Quoted-C race run | Runtime vet |
|---|---:|---:|---:|---:|---:|
| `jsonv2` | 4.381 | 48.338 | 74.693 | 2.789 | 0.466 |
| `nojsonv2` | 39.560 | 47.550 | 82.763 | 37.671 | 6.376 |

JSONv2 ran and passed 141 top-level tests, with 4,268 PASS records and no
skips. The complete suite includes both frozen Unicode/append regressions
and the depth-bound control, all explicitly required to run and pass.

Classic JSON started 129 top-level tests: 128 passed, and the existing
`TestWalkEnterFrameDepthBound` skipped because `encoding/json` v1 has no
depth bound. There were 4,255 PASS records, no failures, and no other skips.
This existing profile distinction was retained without changing its test.

`TestEjectQuotedCgoTypes` was selected exactly and executed in each profile;
both explicit test records show one RUN and one PASS, with no skip. Its
unchanged ordinary-Go and ejected-program assertions check `quoted: 7 9`
under classic JSON and `quoted: 7 9 11 11` under JSONv2. Both use the final
fork's installed driver and compiler. This result is explicit test execution,
not an inference from a quiet package PASS.

Normal `go vet -p=1 logos/rt/json` passed in both profiles. This record does
not claim that the complete compiler frontend race suite or frontend vet
ran in this lane.

## Execution and evidence

`run_final_json_acceptance.py` is the exact executed runner. It checked the
source/tool identity and composition before launch and after completion.
Each stage retains exact argv, selected environment, UTC times, resource
readings, raw stdout/stderr, exit and timeout status, and stream hashes.
The test executions add explicit required-test, full-suite count, and skip
records. No raw stream was decoded and rewritten during copying.

Race binaries were compiled serially with disk `GOTMPDIR` and the task's
`final-postbuild-cache`. Executed binaries used a unique real directory under
`/home/agent` for both `TMPDIR` and `GOTMPDIR`: this fork's `testing.TempDir`
uses `GOTMPDIR`, and the quoted-C test creates ordinary modules there.
Nested `go run` commands inherit that source directory's environment. No
VCS suppression was used. CC was `/usr/bin/cc`, CGO was explicitly enabled,
and each profile was explicit in `GOEXPERIMENT`.

The runner used GOMAXPROCS=1, GOGC=20, GOMEMLIMIT=512MiB, serial package
compilation, count 1, test parallelism 1, a 20-minute test timeout, and a
21-minute outer command bound. Bootstrap compiler flags were absent.
Absolute paths in the logs describe this observed execution and remain
unchanged in the portable copy.

Test executables, caches, and live temp directories are intentionally excluded.
Their recorded binary receipts remain in the compile-stage directories.
Verify all portable bytes with `sha256sum -c evidence.sha256` from this
directory. The manifest excludes itself.

## Scope

These gates prove the final composition of the element-type helper signatures,
shared permissive Unicode decoder, preserved reflection-fallback removal,
corrected regression placement, and quoted-C source printing. The existing
source45 semantic RED and caught/restored pointer mutation remain preserved
in `logos-doc/validation/acceptance-followups/json-text/`; they were not rerun
or rebound to the final source here.

This is the fresh installed M11-A-S9 / AUD-203 / REDTEAM #419 composition
acceptance and later installed evidence for #391. Broader hosted frontend,
fixture, command-script, and full Go release gates remain separate.
