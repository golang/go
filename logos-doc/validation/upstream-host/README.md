# Upstream CI host and scheduling controls

These helpers change the CI invocation boundary. They leave the compiler,
runtime, Go tests, JavaScript host, test assertions, and dist selection rules
unchanged. Native execution remains pending until the hosted job records it.

## Observed failures and the bounded correction

[Run 43927955055](https://github.com/gosuda/LoGOs/actions/runs/43927955055)
reported `total length of command line and environment variables exceeds limit`
in the three Node-backed Wasm roots. `wasmGoEnv` inherits most host variables;
`wasm_exec_node.js` forwards them into the guest. `wasm_exec.js` reserves 8 KiB
for the combined argument/environment strings and pointers. This is a Go host
staging limit, not an intrinsic Node engine limit. The specific variable that
crossed the old limit has not been isolated.

The same upstream job ran `cmd/go` for 542.698 seconds; 673 script failures
explicitly reported the shared context deadline. The existing LoGOs-only script
job had taken 1265.083 seconds with its existing 60-minute limit. Those records
justify a separate bounded driver budget. They do not establish that every
failure in the upstream job was caused by its deadline.

The old full dist batch included `cmd/cgo/internal/testshared` and
`internal/godebugs`, which cause `registerStdTest` to choose timeout zero.
At timeout scale one, `buildArgs` omits `-timeout`, leaving the default ten
minutes. `TestScript` reserves two five-percent cleanup intervals, producing
roughly 540 seconds of script context. An isolated `cmd/go` registration uses
180 seconds multiplied by three; scale four gives a 36-minute package budget
and roughly 32.4 minutes of script context. The complement keeps scale one.

## Environment boundary

`.github/scripts/logos-test-env.sh COMMAND...` executes the unchanged argument
vector under an explicit environment allowlist. It keeps each allowed value
exactly, including the distinction between an absent variable and an empty
one. This includes `GOEXPERIMENT`, actual `GO_BUILDER_NAME`, Go selectors and
module policy, compiler settings, caches and resource controls, all three
`UPROOT` oracles, and the existing `HOME`, `USER`, `PATH`, and `TMPDIR`. It does
not create a replacement home or invent a builder identity.

The Linux wrapper uses absolute interpreter and `env` paths, so an empty caller
`PATH` does not break the wrapper itself. Only exported settings are copied;
Bash's unexported default for a missing `PATH` is not added to the child.

The wrapper removes Actions metadata, arbitrary credential/session variables,
proxy transport variables, and `NODE_OPTIONS`. It does not fetch inputs or
establish sessions. Configured Go module policy, including `GOAUTH`, remains
available to Go; the evidence receipt does not serialize these values. A valid
wrapper invocation writes nothing of its own to the child's output streams.

## Exact Wasm execution evidence

After a fresh toolchain build and Node installation, run from the checkout root:

```sh
uv run --no-project --no-config --python 3.12 --no-python-downloads python \
  logos-doc/validation/upstream-host/check_wasm.py \
  --repo "$PWD" --evidence "$RUNNER_TEMP/logos-regression-evidence/wasm"
```

The verifier invokes the installed `bin/go` through the wrapper with
`test -json -count=1 -short -timeout=10m` and an exact three-root selector. It
requires one run and one pass for each of these seven identities:

| Root | Additional named children |
| --- | --- |
| `TestWasmProgramRunsUnderNode` | `resume_points_everywhere`, `selective_resume_points` |
| `TestWasmWrongMayYieldTraps` | None |
| `TestWasmYieldFreeSourcesRunUnderNode` | `plain_go`, `logos` |

These are three roots and five leaves, with seven named events of each action.
Map iteration order is allowed to vary. Missing, duplicate, skipped, failed,
unbuilt, malformed, or out-of-lifetime evidence is rejected, as are crash or
timeout diagnostics at package, root, child, or build level. The unchanged
wrong-MayYield test still requires its deliberately invalid program to trap;
that expected subprocess failure is handled inside the test and is not a
passing skip.

`frozen.json` pins the existing Wasm test, both JavaScript files, dist test
driver, script-test driver, and `run.bash`. The checker verifies all six before
starting Go and requires clean tracked source. It retains raw JSON/stdout and
stderr, exit status, argv, elapsed time, environment presence controls, and
before/after source, Node, Go, logos, compiler, and linker hashes. Source/tool
changes reject acceptance. The wrapper's environment-byte receipt is only an
environment estimate; it does not claim to measure the final guest argv.

## Complete dist partition

From `src`, capture one unfiltered inventory through the same entrypoint as
all three executions:

```sh
bash ../.github/scripts/logos-upstream-identity.sh before "$GITHUB_WORKSPACE" "$evidence"
../.github/scripts/logos-test-env.sh ./run.bash --no-rebuild -list > "$evidence/all-tests.txt"
uv run --no-project --no-config --python 3.12 --no-python-downloads python \
  ../logos-doc/validation/upstream-host/partition_dist.py \
  --inventory "$evidence/all-tests.txt" --evidence "$evidence/partition"
GO_TEST_TIMEOUT_SCALE=1 ../.github/scripts/logos-test-env.sh \
  ./run.bash --no-rebuild -k -run='!^(cmd/go|cmd/internal/testdir:(0|[1-9][0-9]*)_([1-9][0-9]*))$'
GO_TEST_TIMEOUT_SCALE=2 ../.github/scripts/logos-test-env.sh \
  ./run.bash --no-rebuild -k -run='^cmd/internal/testdir:(0|[1-9][0-9]*)_([1-9][0-9]*)$'
GO_TEST_TIMEOUT_SCALE=4 ../.github/scripts/logos-test-env.sh \
  ./run.bash --no-rebuild -k -run='^cmd/go$'
bash ../.github/scripts/logos-upstream-identity.sh after "$GITHUB_WORKSPACE" "$evidence"
```

The workflow must run all three partitions independently, retain all exit statuses
and raw logs, and keep every other selection-affecting setting identical. The
inventory uses `run.bash` too, because that script sets `GOENV=off` and evaluates
the installed dist environment before registration. `dist -list -run=...` is
not a filtered inventory: its list path prints all registered names.

`partition_dist.py` requires unique nonempty names, exactly one literal
`cmd/go`, and a complete coherent set of canonical `cmd/internal/testdir:I_N`
shards. Every shard must have the same positive N and indices must cover exactly
0 through N-1. It checks the observed count before allocating index coverage.
Prefixes, other variants, and noncanonical spellings such as `00_1` and `0_01`
stay in the complement. The parser and the three workflow filters use the same
Go-compatible regexp grammar. Tests bind the actual workflow commands to those
filters and timeout scales. The receipt records a disjoint complete union and
explicitly makes no execution claim.

The original 666-name inventory has one command-driver name, one testdir shard,
and 664 complement names. The observed testdir timeout is retained in
[testdir-timeout](testdir-timeout/README.md), together with the bounded capacity
rationale. Testdir alone gets a 20-minute package timeout and a 25-minute step.
The complement keeps scale 1, cmd/go keeps scale 4, and the job keeps 90 minutes.
The original `frozen.json` retains its historical two-way plan and existing
native source pins; the new evidence and selection controls supersede only
that partition plan. [Provenance notes](testdir-timeout/NOTES.md) distinguish the
failed-run source from the archived implementation-base workflow.

`logos-upstream-identity.sh` records exact HEAD/src/test trees, clean source and
fixture status, and hashes of both drivers plus compile/link/asm/cgo. It changes
to the explicit repository root before inspecting `src test`. Baseline receipts
cannot be overwritten. After all partitions, the same source and tool identities
must match even after an ordinary suite failure. Cancellation or a failed build
does not run the after step. The old run had no installed-tool receipts; this
new boundary supplies them for the next hosted run.

## Local verifier checks

The focused Python tests require the existing pinned
`../tagged-package-facts/test-requirements.txt`. Run pytest, Pyright, and the
full Ruff check/format check from the repository root using uv. Shell syntax is
checked with `bash -n`; hosted CI also runs ShellCheck. No local Go execution
was used to develop this slice.

The pre-correction controls exposed 51 failures: 18 configured-value/presence
losses, 28 accepted crash-diagnostic placements, four accepted lifetime
violations, and one split crash diagnostic. The corrected suite retains these
controls. A hosted missing/skip result, changed frozen source, altered tools,
incomplete inventory, nonzero partition exit, or repeated deadline remains a
failure; none is converted to success by these helpers.
