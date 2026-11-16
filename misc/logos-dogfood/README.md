# Differential dogfood corpus

This replay compares the same pinned source and build settings through stock Go,
the fork's `go`, and `logos`. It also checks the report contracts of `logos lint`
and, when supplied, gglint in its basic, standard, and strict modes.

The manifest has **24 external sources and two local adversarial fixtures**.
Assembly/purego variants of xxhash and cgo/stub variants of go-sqlite3 bring the
total to **28 cases**. These are explicit samples, not exhaustive coverage of the
Go ecosystem or a claim that the third-party projects contain no defects.

## Inputs

`corpus.json` records each module, upstream URL, revision or module version,
checksum where applicable, selection cohort, backend, cgo setting, build tags,
package patterns, and operation. Inputs remain outside this repository except
for the small original fixtures in `fixtures/`.

| Cohort | Selection purpose | Examples |
|---|---|---|
| `popular` | Widely used implementation paths | chi, zap, protobuf, go-sqlite3, xxhash, UUID |
| `reference` | Small libraries or useful implementation practices | go-cmp, multierr, TOML, errorlint, scfg, exhaustive |
| `stress` | Generated code, unsafe/FFI, legacy control flow, or deliberately vulnerable code | modernc, purego FFI, YAML, x/sys, GoVWA |

The cohorts are sampling labels. They do not certify code quality. GoVWA is an
intentionally vulnerable application; the replay only builds and analyzes it.
It does not launch its server.

The upstream origins cover GitHub, GitLab, Codeberg, SourceHut, and
go.googlesource.com. A module cache entry represents its recorded origin, not an
additional independent source host.

### Source verification

- `git`: prepare a checkout under `--source-root/<source>`. The runner requires
  the exact HEAD, matching origin URL, and a clean worktree. It copies tracked
  files only. The manifest pins full commit IDs, never moving branch names.
- `module`: prepare the exact module version in `--module-cache`. The runner
  checks the cached `.info` version and origin commit when a revision is pinned,
  then recomputes Go's `h1` directory checksum from the actual extracted files.
  Reading a `.ziphash` marker alone does not verify the source being tested.
- `local`: resolve the fixture relative to the manifest and verify its pinned
  `h1` content checksum. Refresh that pin deliberately when changing a fixture.

The runner never fetches sources. Prepare Git checkouts at the manifest's full
commit IDs or populate the module cache with `go mod download -json
MODULE@VERSION`. Automatic module resolution and downloads are disabled for
replay commands through `GOPROXY=off` and `GOSUMDB=off`; these settings are not a
network sandbox. Missing dependencies remain visible as blocked cells. Module
inputs are verified against their pinned content sums before use.

Each command receives a fresh private copy of its source. The copy is checked
against the verified digest before execution. Commands run outside the corpus
checkout, and each subsequent command receives a new copy without earlier
commands' output files. This copying does not provide a filesystem sandbox.
Only one source copy remains on disk at a time; the copy is removed after its
case. Symlinks and newline-containing source filenames are rejected explicitly.

## Run

Use Python 3.11 or newer through `uv`, Git, a stock Go 1.27.4 toolchain, the built
fork, and a C compiler for cgo cases. Build the fork after changing its compiler,
runtime, or command packages. From this repository root:

```sh
uv run --no-project python misc/logos-dogfood/run.py \
  --go-oracle /opt/go1.27.4/bin/go \
  --go-candidate "$PWD/bin/go" \
  --logos "$PWD/bin/logos" \
  --gglint /opt/gglint/gglint \
  --source-root /opt/dogfood/git \
  --module-cache /opt/dogfood/modules \
  --cache-root /tmp/logos-dogfood-cache \
  --out /tmp/logos-dogfood-evidence
```

Omit `--gglint` to run the three toolchain stages and the built-in lint stage.
Use a new evidence directory for each invocation; an existing `report.json` is
never overwritten. Build caches can be reused through `--cache-root`, with
separate subdirectories for the oracle, fork, logos, and gglint.

Repeat `--only CASE` to select cases. Repeat `--stage oracle`, `--stage go`,
`--stage logos`, `--stage logos-lint`, or `--stage gglint` to select stages. For
example, this checks only the local semantic and ABI fixtures with both Go
drivers:

```sh
uv run --no-project python misc/logos-dogfood/run.py \
  --go-oracle /opt/go1.27.4/bin/go \
  --go-candidate "$PWD/bin/go" \
  --source-root /opt/dogfood/git \
  --module-cache /opt/dogfood/modules \
  --only fixture-semantics --only fixture-cgo \
  --stage oracle --stage go \
  --out /tmp/logos-fixture-evidence
```

`--operation build` overrides the per-case operation for a compile-only sweep.
The normal `test` operation uses **`go test -count=1 -short`**, with a test
deadline below the enclosing command deadline. This executes upstream short
tests; it is not a full upstream integration test suite. `--timeout` defaults to
180 seconds per command, and `--jobs` defaults to two. On POSIX systems, a
deadline kills the entire command process group and preserves partial logs.

Package patterns, cgo settings, and build tags are identical for all compiler
stages. The gglint stage uses stock Go on PATH and an explicit configuration with
the same build tags. Its report includes test-file diagnostics. `logos lint`
follows its own package scope and does not imply that the same test files were
analyzed.

### Replay with reviewed capability grants

`corpus-permissions.json` is a separate manifest for ten pinned sources whose
default-policy builds reported capability denials. It preserves their original
source pins and selects reviewed files under `policies/`. These grants authorize
replay of this pinned corpus only; they do not certify the dependencies or grant
permissions to other versions. Keep the original default-policy evidence.

Each policy has paired `logos_policy` and `logos_policy_sha256` fields. The
runner verifies the exact bytes, saves a separate policy artifact in the new
evidence directory, and applies it as `logos.toml` only in the private copies
for `logos` and `logos lint`. It refuses to overwrite an upstream configuration.
Oracle Go, fork Go, and gglint receive the pristine source. Both the original
source digest and each command's effective input digest are recorded, including
the repeated lint input.

The accepted policy format contains only `[capabilities]` and exact package
paths in `transitive_unsafe_allowlist`, `transitive_net_allowlist`, or
`transitive_exec_allowlist`. Paths with
globs, duplicate grants, escaping or symlinked policy files, severity changes,
build settings, and lint overrides are rejected. Default denials and reflection
warnings remain enabled. The existing configuration semantics are documented
in [the language specification](../../logos-doc/SPEC.md), sections 8 and 11.

Run the separate compile and lint comparison with a successful stock oracle:

```sh
uv run --no-project python misc/logos-dogfood/run.py \
  --manifest misc/logos-dogfood/corpus-permissions.json \
  --go-oracle /opt/go1.27.4/bin/go \
  --go-candidate "$PWD/bin/go" \
  --logos "$PWD/bin/logos" \
  --source-root /opt/dogfood/git \
  --module-cache /opt/dogfood/modules \
  --cache-root /tmp/logos-dogfood-cache \
  --operation build --jobs 1 \
  --out /tmp/logos-permission-evidence
```

A capability denial can occur before lint emits its JSON report. Its original
failed or invalid-report cell remains evidence of that invocation. A successful
permission replay is a separate result under its recorded grants. Additional
linker denials require review before a new policy and replay are recorded.

The separate `corpus-link-permissions.json` records that next review for
errorlint and exhaustive, using new files in `policies/link/`. Their pinned
`golang.org/x/tools/go/packages` and `golang.org/x/tools/internal/gocommand`
dependencies need process-launch authority. These package grants include their
configured external drivers and helper commands. The initial policy files and
manifest remain unchanged. Use the new manifest and another new evidence
directory for this linker comparison; retain the earlier denial records.

## Evidence and interpretation

`report.json` is saved atomically after each cell. It starts with `complete:
false` and becomes complete only after every selected case has been attempted.
Complete means the replay finished, including any failed or blocked cells.

The report includes the runner and manifest hashes; selected cases and stages; source pins
and verified digests; driver and compiler/linker/assembler/cgo binary hashes;
tool versions and Go environment; compiler source digest and Git revision when
available; full commands and effective build environment overrides; exit codes;
deadlines and elapsed time; diagnostic counts; and log paths and hashes. Raw
stdout and stderr remain separate files beside the report.

| Status | Meaning |
|---|---|
| `pass` | Oracle succeeded, or candidate succeeded with a successful oracle in this invocation |
| `clean` | Lint returned a valid empty report and the corresponding success status |
| `findings` | Lint returned valid diagnostic records and the corresponding exit status |
| `upstream-failed` | The stock Go build or short-test baseline failed |
| `candidate-failed` | Candidate failed against a successful baseline, crashed, or returned an invalid driver exit status |
| `blocked` | A source, dependency, toolchain, native compiler, or disk prerequisite was unavailable |
| `load-failed` | A requested pattern was empty, or ordinary compiler/typecheck errors prevented complete lint analysis |
| `timeout` | The command exceeded its deadline |
| `invalid-report` | Diagnostic JSON, fields, severity, count, or exit status violated the report contract |
| `nondeterministic` | Repeating lint changed the diagnostic multiset or exit status |
| `uncompared` | A compiler stage ran without a successful oracle in this invocation |

Every successful lint report is replayed on a fresh copy with the same cache.
Comparison sorts complete diagnostic records and preserves duplicate
multiplicity. Raw log hashes also expose ordering differences. gglint's promise
to emit each identical finding once is checked. A nonzero status alone is never
accepted as evidence that a lint completed: its parsed records must explain the
status. Package banners in logos stderr must be followed by diagnostic records;
other unstructured stderr is not silently discarded.

Native ordinary compiler errors intentionally use an empty `rule_id`. The
runner verifies their full point-span record, zero provenance, empty fixes,
deny severity, and failed exit status before classifying them as `load-failed`.
Malformed or incomplete anonymous records remain `invalid-report`. This is a
loading failure, including when mixed with valid lint findings, and never a
successful lint result. Existing evidence retains the classification made by
the runner version recorded in that report.

`findings` establishes report integrity and repeatability, not diagnostic
precision. Confirming a true positive or false positive requires source review
and a minimized paired regression. The runner does not set diagnostic budgets,
suppress difficult rules, or treat a large finding count as an upstream defect.

The command exits nonzero if any selected stage is blocked, failed, timed out,
invalid, nondeterministic, or uncompared. Selecting `--stage go` without
`--stage oracle` is useful for collecting evidence, but cannot produce a
differential pass. Such invocations remain visibly `uncompared`.

## Coverage boundaries

`corpus-runtime.json` is a separate xxhash runtime comparison. Its first two
cases run the complete root package with cgo disabled, both with assembly and
with the `purego` tag. The third runs every package with cgo enabled, including
the upstream dynamic-plugin test. The original cgo-disabled `./...` cases stay
unchanged: their root tests pass, but their plugin `TestMain` requires external
linking and fails under stock Go. The supplemental cases preserve the same
source pin and record their different package patterns and cgo settings.

`replay_ffi.py` runs eight named upstream purego tests with JSON execution proof:
symbol lookup, nested C++ loading, a native call, a callback from C, mixed
floating-point and integer callback arguments, a wide callback result with
stack arguments, a stack callback, and stack alignment. It requires all eight
tests to start and pass without skips under each driver, with a passing stock
baseline. The full purego short suite remains separate evidence; its credential
and all-thread syscall tests can fail in a restricted host even under stock Go.
The focused replay does not replace those failures or claim the full suite
passes. Run `uv run --no-project python misc/logos-dogfood/replay_ffi.py --help`
for the required explicit tool, source, cache, and output paths.

- Go-only code, the `purego` build tag, ebitengine's purego FFI implementation,
  native cgo, and modernc-generated Go are separate cases.
- The go-sqlite3 cgo-disabled case builds its four ordinary stub source files.
  Upstream's cgo-dependent test files do not compile in that configuration.
  This case does not claim SQLite functionality without cgo.
- modernc SQLite uses its own pinned `modernc.org/libc` dependency. The runner
  does not upgrade or replace that dependency to make tests pass.
- Architecture and OS variants need explicit separate invocations. The default
  replay is for the current host; it does not prove cross-platform execution.
- The external matrix contains Go source. The original `.lgo` language fixture
  under `fixtures/language` must be checked through native logos and the ejected
  Go result separately because stock Go cannot parse `.lgo` source.
- `fixtures/semantics` contains test code only. A clean `logos lint` result for
  that fixture does not establish that its test functions were analyzed. The
  deliberate safe/unsafe diagnostic pairs in `fixtures/lint` are outside this
  execution matrix and need a separate exact-diagnostic assertion.

## Test the replay contract

```sh
uv run --no-project python -m unittest discover \
  -s misc/logos-dogfood -p test_run.py
uvx pyright misc/logos-dogfood/run.py misc/logos-dogfood/test_run.py
```

Both Python files enable Pyright's strict mode directly.

These tests cover false success on exit 1, partial-error masking, missing package
patterns, schema and severity mismatches, typecheck failures, duplicate findings,
repeat ordering and multiplicity, absent baselines, source hashes, actual
subprocess timeout and interruption cleanup, and the capability-policy boundary.
