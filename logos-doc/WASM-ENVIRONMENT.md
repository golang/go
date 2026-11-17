# Wasm test guest environment

## Status and scope

M16-S1a, anchored by Tony Hoare, repairs only the Wasm test execution helper.
The frozen attack, the first helper candidate, and an original-runner overlay
establish RED/GREEN/PROBE; restoring the candidate passes the same tests.
At publication, combined CI acceptance had not been established. Use the
candidate's [PR #54 checks](https://github.com/gosuda/LoGOs/pull/54/checks)
for that result; the local proof below does not replace the full upstream gate.

## Contract and repair

Unrelated host variables must not enter the guest or consume its startup space.
`runWasmNode` previously reused the build environment from `wasmGoEnv`.
The Node launcher copies its environment into the Go guest, so this exposed
host-only values and could overflow the runtime's fixed 8 KiB startup region.

Attempt 1 changes only the execution helper, with a two-line explanation:

```go
cmd.Env = []string{"GOMAXPROCS=1", "TMPDIR=" + tb.TempDir()}
```

The helper explicitly supplies only the fixed scheduler value and a
per-call writable directory. `buildWasm`, `goExports`, the build environment,
production runtime, startup limit, and existing semantic/trap assertions
are unchanged. No compiler or language implementation changes in this slice.

## Frozen attack and source identity

`src/cmd/compile/internal/logos/wasm_env_test.go` builds one plain-Go guest
before changing the host environment. Its clean, sentinel, and 16 KiB cases
require the host-only value to be absent, `GOMAXPROCS=1`, and a temporary
file to be created, closed, and removed successfully. The test is unchanged.

| File or tool | SHA-256 |
| --- | --- |
| Frozen `wasm_env_test.go` | `0a5176044f8f55cd3ad62d7694ad16e6600f5cd463f3d56e23cb4eee5c4a54e7` |
| Original `wasm_test.go` | `64e2c6a45ee50bc43e1a3340f567571009871df1fbc57a1d739a92556d8221c1` |
| Candidate `wasm_test.go` | `00110006ef36b4de0baa9134ccc8b9dd767ea6a20a922156222cb57de0f6d695` |
| Unchanged `lib/wasm/wasm_exec.js` | `0c949f4996f9a89698e4b5c586de32249c3b69b7baadb64d220073cc04acba14` |
| Unchanged `lib/wasm/wasm_exec_node.js` | `86b8ce546ecaa462544f3736adc4c19264eedd257c2ec427c9abc6debc1ee184` |
| Installed compiler | `4aaa7bd74c4b01a326abae70117d3009058e7d515f6c8856ecae35c117a8a5da` |
| Installed `go` and `logos` | `293129e95b816bf74be72596a4ac7993d67b872bacacfb7c3730f5911507c03c` |

The local tools report Go/LoGOs 1.27.4 and Node 26.13.1. A fresh full
`make.bash` succeeded on `447d6ad7` source plus the frozen attack
([build log](validation/wasm-environment/build.log)). Before the repair,
the runner and runtime bytes matched integrated head `b154571b`.
This establishes the tested helper's source identity, not a full build
or upstream regression result for every integrated change.

## Observed evidence

| Stage | Observed result | Evidence |
| --- | --- | --- |
| Original runner RED | 27.318s. Clean control passed; sentinel caused `host-only LOGOS_WASM_TEST_PADDING reached the wasm guest`; 16 KiB case overflowed startup. All three existing Wasm top-level tests passed. No skips. | [red.log](validation/wasm-environment/red.log) |
| First candidate GREEN | All selected tests passed in 1.850s. | [proof.json](validation/wasm-environment/proof.json) |
| Controlled GREEN | All selected tests passed in 2.789s. | [green.log](validation/wasm-environment/green.log) |
| Original-runner overlay PROBE | 1.882s. The same sentinel and overflow failures returned; the positive control passed. | [probe.log](validation/wasm-environment/probe.log) |
| Restored candidate | All four top-level tests and all subtests passed in 1.698s, with no skips. | [restored.log](validation/wasm-environment/restored.log) |

Formatting and `go vet cmd/compile/internal/logos` passed. The unchanged
`TestRedteamLedgerRows` passed in 0.030s on the local repaired-ledger
snapshot. [proof.json](validation/wasm-environment/proof.json) records the
supporting verification; candidate-specific CI remains a separate gate.

The first overlay launch was rejected before tests ran because `env -i`
left `GOMODCACHE` empty. That launch is excluded from semantic proof and
retained as [overlay-setup-rejected.log](validation/wasm-environment/overlay-setup-rejected.log).
Controlled GREEN, PROBE, and restoration used an absolute owned
`GOMODCACHE`. No assertion or frozen fixture changed between those runs.

## Focused selection and full acceptance

```sh
go test -count=1 -timeout=10m -v cmd/compile/internal/logos \
  -run '^TestWasm(RunIgnoresHostEnvironment|ProgramRunsUnderNode|WrongMayYieldTraps|YieldFreeSourcesRunUnderNode)$'
```

The selection covers the new environment attack, ordinary Node execution,
the wrong-may-yield trap, and yield-free sources. An environment cleanup
outside the test cannot satisfy the attack: its subtests add the sentinel
and large value after startup and after the guest has been built.

At publication, the `b154571b` [SIMD/Wasm run](https://github.com/gosuda/LoGOs/actions/runs/43922367072)
and [language run](https://github.com/gosuda/LoGOs/actions/runs/43922367067)
were queued and had never started. Those runs precede this helper candidate;
acceptance must be verified on the candidate identified by the live
[PR #54 checks](https://github.com/gosuda/LoGOs/pull/54/checks).

The earlier `08646b8` full Go regression passed. The `eaf5e3e` child passed
its builds and LoGOs gates but failed its full upstream gate on the Wasm
host environment and ledger schema. Those historical results remain distinct
from combined acceptance. See [SIMD-DOMINANCE.md](SIMD-DOMINANCE.md) and
[DEFAULT-BINDINGS.md](DEFAULT-BINDINGS.md) for the original evidence.

## Measurement and attempt boundaries

Host benchmark `ns/op` includes the new temporary-directory setup.
Custom timing inside the guest loop excludes that setup. No speed claim
or benchmark improvement is inferred from the correctness results.

REDTEAM row #406 records the tested helper invariant. M6-S3a remains at
its third and final implementation attempt; M6-S3b remains blocked after
its third attempt. This independent test-helper repair does not reopen
either slice or alter the frozen binding archive.
