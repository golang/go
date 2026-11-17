# Acceptance cache audit

Source: public commit `45c29ff215477f7e403188d7452fc79aef05d791`, whose tree
matches the tested merge `f2ad58be8742b8ec350db8f06d9dc794ea3302e4`.
Scope: the current Linux amd64 and arm64 hosted toolchain builds.

## Verdict

The suspected stock-bootstrap cache collision is excluded by the existing
fork tool-identity hooks. No workflow cache change is justified. This is a
source-level assessment, not an executed cache-hit or cache-miss trace.

1. [make.bash](https://github.com/gosuda/LoGOs/blob/45c29ff215477f7e403188d7452fc79aef05d791/src/make.bash#L155)
   builds the first cmd/dist with stock Go and the inherited cache. That
   initial cache population is real. Dist then selects its private cache;
   [the VERSION branch](https://github.com/gosuda/LoGOs/blob/45c29ff215477f7e403188d7452fc79aef05d791/src/cmd/dist/build.go#L1602)
   does not restore the old cache when VERSION exists. The parent shell
   retains its own environment independently of the child process.
2. [forkReportsBuildID](https://github.com/gosuda/LoGOs/blob/45c29ff215477f7e403188d7452fc79aef05d791/src/cmd/internal/objabi/flag_logos.go#L7)
   is true, so [the version flag](https://github.com/gosuda/LoGOs/blob/45c29ff215477f7e403188d7452fc79aef05d791/src/cmd/internal/objabi/flag.go#L231)
   emits a build ID even for release-version tools.
3. [work.toolID](https://github.com/gosuda/LoGOs/blob/45c29ff215477f7e403188d7452fc79aef05d791/src/cmd/go/internal/work/buildid.go#L176)
   recognizes that suffix and uses its content ID. Every Go compilation
   [hashes the compiler tool ID](https://github.com/gosuda/LoGOs/blob/45c29ff215477f7e403188d7452fc79aef05d791/src/cmd/go/internal/work/exec.go#L378).
   The stock release-version key therefore differs from the installed
   fork's key, including for identical standard-library source.
4. [Tool linking stamps objabi.buildID](https://github.com/gosuda/LoGOs/blob/45c29ff215477f7e403188d7452fc79aef05d791/src/cmd/go/internal/work/gc.go#L631).
   [MustLinkExternal](https://github.com/gosuda/LoGOs/blob/45c29ff215477f7e403188d7452fc79aef05d791/src/internal/platform/supported.go#L101)
   returns false for the audited Linux amd64/arm64 tool builds without cgo;
   the final toolchain2/3 sequence supplies build IDs. Link actions also
   include the linker's tool ID.

GOROOT separation is not the decisive protection: standard-package compile
keys deliberately omit the root path. Source hashes, dependency content
IDs, target settings and experiment flags provide additional distinctions,
but compiler content identity already excludes the proposed collision.

## Correction and evidence boundary

The earlier warning that an unchanged VERSION could mask compiler changes
was overbroad for this fork. It relied on an inherited upstream comment
without accounting for the existing release-build-ID hooks. Correct that
warning in the final acceptance report; it does not invalidate the current
hosted results. The deliberately empty local post-build cache remains useful
isolation, not a repair required to make hosted acceptance valid.

No Go command, build, test, cache inspection or mutation was performed for
this audit. No claim is made about unexamined externally linked host ports.
