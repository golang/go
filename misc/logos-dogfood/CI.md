# Pinned corpus acceptance

`.github/workflows/dogfood.yml` rebuilds the compiler and executes the corpus on
Linux/amd64. It uses the pull request's exact head and a full reviewed gglint
commit, rather than a moving gglint branch. The independent `logos.yml` workflow
retains the native unit, script, race, mutation, and full upstream gates.

## Reviewed gglint revision

The workflow pins `4456e61c4dbdf636fca2b5bd77787873e2d1c1e8`, tree
`dfb46c72af5ce9695844c5d567cbb8aab1568183`. All four gates in
[run 43963396252](https://github.com/gosuda/gglint/actions/runs/43963396252)
passed on a merge with that exact tree: 39 packages tested with the race detector,
the unchanged nine-worker vettool stress case, vet/lint/self-lint, and the frozen default-select
regression with two independent Chi copies. The strict Chi result removes only
the targeted receive false positive; basic and standard findings are unchanged.
This pin replaces `a95099aae13ff7ffb4b3825bd63f8510bd184c69`. Historical results
for that earlier revision remain preserved under their original identities.

## What must pass

The compile and lint sweep contains **196 primary cells**: 28 cases multiplied
by stock Go, fork Go, LoGOs, native lint, and three gglint modes. It uses
`--operation build`. Compilation alone does not establish runtime correctness.
The semantic fixture contains only test files, so its build cells check package
selection. The separate required runtime step compiles and executes those tests.
Native lint and gglint also have different test-file scopes, as described in the
main corpus README.

For 27 cases, all three compilers must succeed. All four lint configurations
must return valid diagnostics with the corresponding exit status, then repeat
with the same complete diagnostic multiset. Findings are retained; a linter is
not required to call adversarial or third-party code clean. gglint duplicate
findings, report-schema errors, crashes, timeouts, and unavailable dependencies
fail the gate.

`accept_corpus.py` checks every required case and mode, the source and policy
pins, recorded command, private source directory, package patterns, build tags,
cgo setting, toolchain PATH, offline/read-only settings, raw-log hashes, repeat
inputs, and recomputed assessments. Its CLI also checks the unchanged runner
hash and the executable hashes. Removing a mode or changing a label cannot
substitute for executing the requested configuration.

### One explicit failed-load control

The pinned `gitlab.com/bosi/decorder@v0.4.2` depends on
`golang.org/x/tools@v0.20.0`. Stock Go 1.27.4 rejects that dependency's
`internal/tokeninternal/tokeninternal.go:64:9` with:

```text
invalid array length -delta * delta (constant -256 of type int64)
```

The seven `decorder` cells retain their failure classifications. The oracle
must fail with that exact compiler error; fork and LoGOs results remain
`uncompared`; native lint and every gglint mode remain `load-failed`. Structured
errors must identify the same source, position, and message. Other compiler
errors, unexpected gglint stderr, transport errors, and timeouts are rejected.

The runner therefore exits one even for an accepted campaign. The workflow
records that exit and the console-capture exit independently, then invokes the
acceptance checker. A blanket successful exit or ignored command failure cannot
satisfy this gate. `acceptance.json` explicitly records 27 accepted positive
cases and `decorder` as an incomplete analysis. It does not rewrite `report.json`.

## Reviewed capability profile

`corpus-ci.json` preserves every original case, source pin, build tag, cgo
setting, package pattern, and operation. It selects the ten existing capability
profiles, uses the existing linker profiles for errorlint and exhaustive, and
selects `policies/modernc-sql-runtime.toml` for modernc SQLite and
`policies/tools-build.toml` for tools. The checker rejects a changed corpus
selection or source pin.

The modernc runtime profile has SHA-256
`5666c90bc1f25d5f0d7d0e5b53e80c2fa8e7ebe341286f7d7b0b1c5651513096`.
It retains the six reviewed unsafe package grants, permits the raw-syscall
callers `golang.org/x/sys/unix` and `modernc.org/libc` to use execution and
network capabilities, and permits execution by the test helper package
`modernc.org/fileutil/ccgo`. SPEC section 11.1 classifies raw syscall entries as
both execution and network sinks. The pinned fileutil helper uses
`exec.LookPath`, `exec.CommandContext`, and `Run`.

The tools build profile has SHA-256
`c57b54d9a78993eac2b2614730cfdf05b12e47c1dbbc77c243f16887d67d7def`.
[TOOLS_POLICY.md](TOOLS_POLICY.md) records the exact reviewed owners and the
conservative interface paths behind its grants. The original strict policy
still drives the early three-driver capture and the separate unlimited linker
diagnostic command. That helper binds the pinned source, baseline commands,
raw streams, resource settings, and before/after source/tool identities.
Completed diagnostic capture is distinct from a successful build. Full corpus
acceptance and every runtime gate remain required.

These explicit grants apply only to the private native replay copies. Stock
Go, fork Go, and gglint receive pristine source. No lint severity, diagnostic
budget, or capability classification is changed. The default manifest, earlier
policies, and historical denial evidence remain unchanged. This CI campaign
tests compatibility under the reviewed grants; the separate native capability
script gate tests policy enforcement.

## Runtime gates

Independent steps execute the frozen semantic and C ABI fixtures; the xxhash
assembly, purego-tag, and cgo-plugin profiles; selected go-sqlite3 and modernc
SQLite tests; eight named purego FFI tests; LoGo execution and its ejected stock
Go program; and the exact native ownership/recursion diagnostic fixtures.

The backend replay runs stock Go first. Candidate results require that same
profile's successful oracle, matching package-qualified test run/pass events,
and no skipped or failed test events. The plugin profile explicitly aligns
experiments and optimization settings without editing upstream tests. The
modernc SQL selection requires all seven named top-level tests and eight
subtests. See the [backend runtime documentation](README.md#focused-backend-runtime-comparisons)
for exact tests, profiles, and configuration hashes.

Runtime steps remain independent after a corpus failure, so one failure does
not discard evidence from the remaining backends. The final job still fails
when any required step fails.

## Inputs, resources, and retained evidence

The workflow verifies the Go 1.27.4 and uv 0.13.8 distribution archive hashes
before execution. `prepare.py` fetches exact source pins and dependency versions
without editing upstream module files or executing their code. See
`PREPARE.md` for the fetch and integrity contract.

The main compile/lint profile uses Ubuntu 24.04, two workers, `GOMEMLIMIT=2GiB`,
`GOGC=50`, and a 600-second command deadline. Backend tests use one worker and
a 900-second process deadline. Their Go test binary deadline remains 120 seconds.
The existing focused FFI and language/lint replay scripts retain their own
recorded memory limits and deadlines. These are distinct from historical local
profiles; a longer CI deadline does not turn an earlier timeout into a pass.

The always-run artifact step retains source-preparation receipts, original
module files, source and executable identities, resource settings, policies,
reports, and raw stdout/stderr. Source copies and module/build caches are not
published. GitHub retains each artifact for 14 days; the source manifests and
replay code remain in the repository.

The committed [historical evidence](../../logos-doc/validation/historical-22ab/README.md)
records the interrupted campaigns and earlier failures separately. That evidence
does not establish acceptance of a newer compiler or linter revision.
