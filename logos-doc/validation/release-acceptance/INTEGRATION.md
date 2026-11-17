# Integrating independent release acceptance

M22-A-S6 and AUD-209 add execution coverage for the existing release contracts.
They add no language rule, REDTEAM attack row, or SPEC exception. The two task
ledgers contain 302 identical entries. REDTEAM retains 346 unique rows: 337
tested, two satisfied upstream, no planned rows and seven blocked rows.

## Source and evidence boundaries

The [request](../release-acceptance-request.json) binds the hosted jobs to
source tree `e237c399aae8f8065b724e8a0e27585686c7da18` and the unchanged fixture
hashes. A normal fresh build of that source at local e96a570d has already
passed. [Installed acceptance](../installed-acceptance-e96a570d/README.md)
records the build, array-index, overlay, capability, JSON and actual QEMU
results with their source and tool identities.

The new [workflow](../../../.github/workflows/release-acceptance.yml) tests an
exact clean PR head or explicitly requested commit and records the event
identity separately. The [executor](../../../.github/scripts/release-acceptance.sh)
verifies the requested source tree before work and does not edit fixtures.
Metadata-only commits can retain the source binding; a compiler change needs
a reviewed request update and fresh evidence.

## Hosted work still required

| Gate | Required proof |
|---|---|
| Six driver/optimization cells | Both drivers at O1/O2/O3, complete `std cmd` inventory without short mode, the unchanged 31-name explicit exclusion contract, and separate `os` execution. All six cells can run independently on separate VMs. |
| Complete parity | No planned rows; the entire oracle-enabled release suite, mutants and upstream sweep; a real stamp equal to the unchanged clean tip. A filtered or short run cannot satisfy it. |
| Two-host reproducibility | Distinct host fingerprints; identical source and target recipes; matching clean-cache digests for the frozen repro module and actual order-service example on linux/amd64, linux/arm64 and js/wasm. The size record remains specific to the frozen repro module. |
| Real LSP and guest execution | Verified gopls, QEMU and Node prerequisites; verbose uncached race scripts; all eleven LSP messages, actual native/arm64 output comparison, and the selected Wasm runtime legs, including stack switching. Missing prerequisites or script skips cannot pass the job. |
| Broad Go regression | The existing main workflow retains the complete command/compiler-fixture suites, configured lint and `GO_TEST_TIMEOUT_SCALE=8 ./run.bash --no-rebuild`. Its ordinary short configuration does not replace the release-only cells above. |

These new hosted outcomes are pending. The static checker, source comparison,
mutation probe and workflow linters do not execute or certify them. The final
local arm64 sum result is already valid evidence for its exact installed
source, independently of the hosted runtime job and the still-unprovisioned
local gopls dependency cache.

The parity harness retains all configurations and its child concurrency.
Measured resource preflights do not establish its peak requirement. A host
that lacks capacity must produce an explicit infrastructure result; the gate
does not reduce configurations or weaken assertions to fit.

## Frozen contract and versioned corrections

The [test-first record](TEST-FIRST.md) and unchanged
[checker](check_release_contract.py) precede the production proposal. Baseline
RED records missing release coverage. Candidate static GREEN, a caught
one-line `-short` mutation and exact restoration establish the static contract.
Actual actionlint1.7.12, ShellCheck0.11.0 and Bash syntax checks pass as recorded.

The reviewed preparation corrected the example's txtar member to `m/go.mod`
and preserved its `order_service.lgo` basename. A file-only extraction check
proves the exact module bytes and source hash. The evidence bundle preserves
raw non-UTF-8 log bytes, checks its archive digest, and rejects corruption.
Those original outcomes remain under [V1 history](history/v1/README.md).

Independent review then found an effective arm64 recipe mismatch. The cmd/go
script environment does not forward GOAMD64 or GOARM64, so the unchanged
`logos_repro` script uses the deterministic LoGOs defaults v3/v8.2. V1 selected
and recorded v8.0 for the direct example. The
[V2 correction](arm64-recipe-v2/arm64-recipe.patch) changes exactly those two
executor values to v8.2. Workflow, request, checker and assertions stay
unchanged. Both paths keep CGO_ENABLED=0 and the same empty experiment input;
normal LoGOs default experiments remain enabled in both paths.

[Fresh V2 results](arm64-recipe-v2/results.json) include unchanged-checker
GREEN, the caught/restored short-mode probe and both actual linters. The
[V2 manifest](arm64-recipe-v2/files.sha256) is the verifier for the integrated
production and evidence files. V1 records and manifests retain their original
bytes and mappings as historical evidence; they have not been rewritten to
claim V2 execution. The corrected executor SHA-256 is
`a9a5fdd226443ee84ac22bb9ccef3f6c52278d55cb7973b0062a73a2a156a7b3`.

## Release closure remains separate

M22's historical blocked exceptions are #216, #217, #218 and #226, with their
existing mechanisms and matching matrix obligations. M6-S3b rows #403, #404
and #405 are additional unresolved binding contracts. Their original patch,
assertions and three-attempt disposition remain unchanged.

The workflow reports this stricter closure status independently, so those
blocks do not suppress ordinary Go, profile, parity, reproducibility or runtime
acceptance. Even if every independent behavioral job passes, a parity stamp
alone does not resolve the extra M6 binding contracts or establish complete
language implementation. No further binding implementation is attempted by
this CI slice.
