# Preserve the identity of each inline call site

## Observed failure

Corpus CI run [43928753813](https://github.com/gosuda/LoGOs/actions/runs/43928753813)
built `golang.org/x/tools@v0.52.0` with stock Go, fork Go, and LoGOs. The first
two builds passed. `logos build ./...` exited 1 with 44 capability errors,
including this network path:

```text
golang.org/x/sync/errgroup.(*Group).Go.func1
sync.(*Once).Do
net.initMPTCPavailable
net.sysSocket
```

The module pin is commit `16517d951552611e7bd358c48ee1051fd0e57b0c`, module sum
`h1:ryxg4U63oaUYk7uOLTF37n43f4qgtGvqV4Ioit1BaEs=`. The compiler source tree was
`61483affd1671d27e29a1a1aadc5299ba675f789`. The artifact ZIP has SHA-256
`8d6f69a36646775185ca3c7292bcfe3c7ce2bb662300cfcdd76e313483b271ed`; the failing
stderr has SHA-256
`d8c559f2ae777b062b7768204b563b45eda9426ca851cb27a1d15526b1be0c70`.

This is a capability rejection. It is not a measured compiler crash or a
timeout. The collector and acceptance checker correctly retain and reject the
nonzero candidate build. No corpus policy is changed by this test-only freeze.

## Source-level cause

In `src/cmd/link/internal/ld/capabilities_logos.go`, `followRelocs` asks
`inlineSite(s).at(relocationOffset)` for the call owner. That method decodes the
host's `pcinline` spans but returns the global `fi.InlTree(index).Func` symbol.
`followInlTree` also records edges between these global function symbols.
Different inline copies of one function therefore share a graph node and its
outgoing edges.

`sync.Once.Do` has no reference to `net`. Its fast path is intentionally
inlineable. `net.supportsMultipathTCP` passes `initMPTCPavailable` to its own
`sync.Once`; `errgroup.Go.func1` passes a closure that records an error and
cancels its context. Their distinct inline copies must not compose into a path
from the latter closure to the former callback.

SPEC section 11.1 requires the package of the innermost inline function to own
its call. Retaining that package does not permit merging unrelated inline
instances. Existing graph tests check traversal after edges have been supplied,
and `TestLogosCapInlineIndex` checks span decoding. They do not exercise this
cross-host identity boundary.

The artifact does not contain the compiler object files or their PC tables.
The mechanism above is established from the source and the observed diagnostic;
the new installed test still requires execution to prove its exact generated
relocations reproduce the defect.

## Frozen native test

`src/cmd/go/testdata/script/logos_cap_inline_context.txt` contains three build
profiles. All dependencies are local replacements, so it downloads nothing.
It uses Linux socket declarations with `CGO_ENABLED=0` and the ordinary LoGOs
optimization default. Plain fork controls explicitly use `-O=2`.

| Case | Expected result | Required evidence |
| --- | --- | --- |
| Harmless dependency `sync.Once` plus unrelated first-party `sync.Once` and reachable networking | Build succeeds, with no capability diagnostics | Compiler output confirms both `Once.Do` copies were inlined; the executable runs its harmless callback and exits normally without opening a socket |
| Dependency `sync.Once` callback calls `syscall.Socket` | LoGOs build fails | Exact dependency and `syscall.Socket` capability diagnostic; compiler output confirms `Once.Do` was inlined |
| Dependency socket wrapper is inlined into first-party `main` | LoGOs build fails | Exact dependency and `syscall.Socket` diagnostic; compiler output confirms the dependency wrapper was inlined |

The denied executables are never run. The passing executable retains the
network branch behind a runtime argument check but runs with no argument.

The first case must fail on the archived production source with an
`unauthorized_net` diagnostic naming `example.com/inlinecapdep/safe`, then pass
after the repair without altering the fixture. The two denial controls prevent
an implementation that drops inline ownership or all inline edges from passing.

`frozen.json` records the new test, archived production source, and unchanged
existing validation files. `original/capabilities_logos.go` preserves the old
auditor for an installed-linker negative control.

## Required execution gates

No Go build or test was run while preparing this freeze. The original corpus
failure is preserved; it is not a substitute for the new fixture's RED result.

After an installed toolchain build, run from `src`, with that toolchain first
on `PATH`:

```sh
go test -count=1 -race -parallel=1 -timeout 20m cmd/go -run 'Script/^logos_cap_inline_context$'
go test -count=1 -short -race -timeout 20m cmd/link/internal/ld
go test -count=1 -parallel=1 -timeout 30m cmd/go -run 'Script/^logos_cap_(inline|chains|deadcode|net_socket|shortest_frames|xtest_owner)$'
go vet cmd/link/internal/ld
```

Retain the named old-linker failure, candidate success, unchanged fixture hashes,
and exact source/tool identities. An old-linker failure caused by compilation,
missing tooling, or a skip does not satisfy the negative control.

Rerun the pinned `tools` corpus after the repair. Its real telemetry process
launches and its conservative interface-dispatch paths are separate from the
inline-copy defect; this freeze does not claim the repair resolves every
capability error or authorize any additional dependency capability.
