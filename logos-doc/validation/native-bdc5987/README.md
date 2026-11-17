# Completed native run before the combined repairs

[Run 43927955055](https://github.com/gosuda/LoGOs/actions/runs/43927955055)
finished with **failure** on head `bdc5987c526a9e9aa1b43975290f69a2370ae37a`.
The actual Actions checkout was `df9692747da8b773c8714f5497a5d0f61f06e956`;
its complete tree equals the head tree `b1eca92a0c54e14b8e37974c6516a882d13d1442`.
The source tree is `61483affd1671d27e29a1a1aadc5299ba675f789`.

| Job | Result | Observed result |
| --- | --- | --- |
| Unit, 133512371888 | Pass | Fresh build, unit and race tests, compiler fixtures, formatting, vet, lint, both intended old-code failures and restored passes |
| Scripts, 133512372234 | Fail | The 1265.083-second LoGOs script suite reports only `logos_cgo_shared`: its external `cc` rejects the unchanged `-std=c23` flag; the shared library and header were built successfully |
| Upstream, 133512372425 | Fail | Exactly two package failures: the compiler's Wasm host tests and the `cmd/go` script context deadline |

The separate installed subprocess/race gate in the scripts job still ran and
passed in 28.577 seconds. It includes the unchanged inherited-environment cache
fixture. The unit pass does not replace either failed broad gate.

## Upstream failures

The three Wasm roots are `TestWasmProgramRunsUnderNode`,
`TestWasmWrongMayYieldTraps`, and `TestWasmYieldFreeSourcesRunUnderNode`.
Both plain-fork and LoGOs profiles fail before guest execution because the
Go JavaScript host's fixed 8 KiB argument/environment staging region overflows.
The test helper forwards almost the complete host environment. The logs do
not show a Node engine defect. The unit job did not explicitly install Node,
and its nonverbose result does not establish that those individual tests ran.

The observed upstream command is `./run.bash --no-rebuild -k`. The internal
`go test` argv was not echoed. Source reconstruction establishes a default
ten-minute package timeout: the combined batch contains packages that cause
dist to omit its custom timeout, and the scale is one. The script harness
reserves two 5% cleanup intervals, leaving about 540 seconds. The observed
`cmd/go` duration is 542.698 seconds.

Of 676 script failure lines, 673 explicitly report a context deadline.
Two more include SIGQUIT stacks during builds. The remaining
`logos_stackbuf_cgo_unsafe` exit 1 has no stderr and remains unclassified beyond
its timing. No earlier failed semantic assertion was found. The LoGOs-only
script suite's separately observed 1265.083-second duration already exceeds
the combined batch's context allowance.

These observations motivate a compact CI environment and a separate bounded
`cmd/go` partition. They do not establish that the subsequent profile passes.
Existing Go/JavaScript fixtures and assertions remain the acceptance criteria.

## Evidence

[native-43927955055.zip](native-43927955055.zip) retains all 25 collected files,
including complete unit, script, and upstream logs; API snapshots; independent
audits; and exact-head source files used to reconstruct timeout and environment
behavior. Its contents were read back and compared byte for byte after packing.

The uncompressed upstream log has SHA-256
`4e32724648b62e16bd63d0cf76f3395d83fdffdead2eef2e7018ff14e782c733`.
The [unit summary](unit.summary.json), [upstream summary](upstream.summary.json),
and [checkout identity proof](checkout-identities.json) are also directly
reviewable. [SHA256SUMS](SHA256SUMS) binds the archive and these files.

The initial snapshots inside the archive retain their original collection-time
statements. This completed account supersedes their pending-job status only.
