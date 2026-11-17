# Capability ancestry discovery repair

The purego FFI corpus exposed a nonterminating linker capability audit.
Relocations can name an inlined owner before its incoming path is recorded.
The first-discovery parent links then form a cycle, and `capGraph.depth`
never returns. That same representation also misses a later root or a
shorter incoming path.

## Source and attack history

The fresh baseline is `9a628f5b87f3c69a392ba539d11c7c9a37b98566`.
Its two capability implementation files match the prior dogfood baseline:

- `capabilities_chains_logos.go`: SHA-256
  `8a91295a9fef0fd767738d502db5814966b577a6edb4f14049fa4b032e304540`.
- `capabilities_logos.go`: SHA-256
  `a83cde4cd9c03f79c9c3046fd29cd825604c8a1cfe1bf2d7b1c7ef75cab3648f`.

The discovery test was already frozen by the earlier, unpublished dogfood
investigation. Its SHA-256 remains
`884666bdbf5956560246c837a619e6157ea7ff8ad2f4bc27e3ef1fdaa2a8a1f6`.
The repair preserves that investigation's first implementation attempt
unchanged. It does not restart the three-attempt correction budget.

The new run applies explicit source overlays to the earlier installed fork.
The overlay redirects both capability implementation files and the frozen
new test to this branch. It leaves the earlier checkout untouched. This
verifies the selected package source; it is not a fresh full-toolchain build.

## Contract and implementation

Discovery retains the complete predecessor graph and its roots. Before
reporting, a breadth-first traversal from all roots derives shortest depths
and a parent forest. Each parent has depth one less than its child, so a
parent walk terminates. Later discovery invalidates the prepared paths.
The reverse blame search excludes unrooted predecessors.

This changes no protected sink, severity, allowlist, or exemption. The
existing rule still reports one chain per dependency and sink, with runtime
walls and the nearest dependency frame preserved.

## Focused verification

Run date: 2026-11-17 UTC. Toolchain: Go 1.27.4, Linux amd64.
`GOMAXPROCS=2`, `GOMEMLIMIT=384MiB`, `GOGC=30`, and `-p=1` bound the build.
All test commands use `-count=1`.

| Check | Result | Evidence |
| --- | --- | --- |
| Baseline frozen discovery tests | Failed as predicted: four cycle timeouts, three nonminimal chains, unknown-parent panic, shuffled-oracle timeout | `red.log` |
| Candidate all `TestCapGraph` tests | Passed | `green.log` |
| Remove the rooted-predecessor guard | Caught by `unrooted_incoming_cycle` and shuffled cases: reported chains do not start at a root | `probe.log` |
| Restore candidate | Passed with frozen tests unchanged | `restored.log`, `frozen-tests.sha256` |
| Vet affected package | Passed | `vet.log` |
| Build candidate linker | Passed | `link-build.log` |
| Formatting and whitespace | Passed | no output from gofmt or `git diff --check` |

The frozen suite has 16 named discovery cases and 1000 deterministic
shuffled graphs checked after each visit. The prior 3000 complete-graph
oracle cases and existing named graph controls remain unchanged.

The normal command after installation is:

```sh
go test -p=1 -count=1 -short -timeout=3m cmd/link/internal/ld -run '^TestCapGraph'
go vet -p=1 cmd/link/internal/ld
```

## Actual FFI linker replay

`ffi-baseline.json` and `ffi-baseline.stderr` retain the prior exact linker
probe. With the original `logosarena` object experiment restored, it timed
out at 30 seconds. The SIGQUIT stack is in `capGraph.depth`.

The candidate replay keeps the same object archive, import configuration,
capability policy, and recorded environment. Their hashes match the
baseline. Only the linker executable and output path change; the candidate
linker was built with the explicit source overlay described above.

It returns in 0.63 seconds, with five capability policy diagnostics.
`ffi-link.json` records the command, hashes, status, and observed duration.
This is a termination result, not a successful package-runtime result.
The diagnostics name `github.com/ebitengine/purego_test` as a transitive
dependency; main-module external-test identity needs separate investigation.
No allowlist or policy was weakened to make this replay pass.

## Remaining gates

A fresh build of the integrated published source, the complete short linker
package, existing `logos_cap_*` scripts, and full pinned FFI runtime replay
remain pending at this component checkpoint. The full upstream Go regression
is not claimed. Later integrated evidence must identify the exact source
revision and installed binary hashes; the focused source-overlay checks do
not substitute for those gates.
