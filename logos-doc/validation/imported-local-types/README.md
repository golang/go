# Imported generic local type validation

The real-code corpus exposed an internal compiler error in valid Go:
`golang.org/x/tools/cmd/digraph` imported a package containing a local type
that captures a generic receiver's type parameter. The LoGOs fact walker
resolved that compiler-only local symbol through the public type importer,
which correctly rejected the missing enclosing dictionary.

Production revision `39c62004a991501a25f93eae7a63dc7c7fac5646` restricts
eager scope lookups to valid source declaration identifiers. All 48 new cells
and all 61 earlier dogfood regression cells pass with the rebuilt installed
toolchain. The five affected packages pass their full short test suites,
and the pinned upstream `digraph` package passes all seven top-level tests
under stock Go, fork Go, and LoGOs. The initially killed automatic vet process
passed its dedicated retry; the normal package gate and affected-package vet
also pass.

The [gate record](gates.json) preserves commands, environment overrides, exit
codes, and log hashes, including the interrupted combined command.
Raw `.log` files retain the compiler and test runner's exact whitespace.
This directory's `.gitattributes` marks those logs as verbatim data for Git
whitespace checking; authored source, documentation, and replay code use the
normal checks.

## Real source and reduction

The original source is `golang.org/x/tools v0.52.0`, from the
[Go source mirror](https://go.googlesource.com/tools), revision
`16517d951552611e7bd358c48ee1051fd0e57b0c`, with verified module digest
`h1:ryxg4U63oaUYk7uOLTF37n43f4qgtGvqV4Ioit1BaEs=`. The corpus manifest and
report retain this provenance. Its `internal/graph/graphfmt/dot.go` contains a
local `cluster` type inside `Dot[NodeID].Sprint`; that local type contains a
`[]NodeID` field. The focused replay uses only the original module files,
`cmd/digraph`, and `internal/graph`; all 24 copied files are byte-identical to
the verified module source and recorded in [upstream-input.sha256](upstream-input.sha256).

The reduced imported library is:

```go
package lib

type Box[T any] struct{}

func (Box[T]) Pass(value T) T {
    type local struct{ value T }
    return local{value}.value
}
```

A separate main package imports it, calls `(lib.Box[string]{}).Pass("ok")`, and checks
the returned value. Stock Go 1.27.4 executes this program and builds the exact
upstream `cmd/digraph` source. The pre-repair fork crashes on both. The preserved
[real-code stack](digraph-red.log) locates the call chain:
`ApplyFuncFacts`, `reachableFuncFactObjects`, `Scope.Lookup`, and then
`importer.objDictIdx`.

Imported scopes contain lazy compiler symbols such as `local·1`. These are not
source declaration names, but the old filter excluded only names containing
an ASCII period. Resolving a hoisted generic local type requires its enclosing
dictionary. Fact application has no reason to force this local symbol.

## Invariant and repair

Before looking up scope objects, fact traversal and overlay cloning now retain
only `token.IsIdentifier` names. The same condition guards the direct readonly
fact lookup. Private declarations and valid Unicode identifiers remain
reachable. Methods continue to be reached through their declaring types.

The public importer assertion and the export format are unchanged. The scope's
fresh sorted name slice is filtered in place, preserving declaration order.
The [production digest](production.sha256) identifies the single changed
production file.

## Test-first evidence

The attack tests were added before changing production code and remain at
their [frozen hashes](frozen-tests.sha256). There are **48 new cells**:

- **28 scope cells** cross four compiler-only symbol forms with function
  facts, readonly facts, empty facts, declaration cloning, structural cloning,
  and direct readonly lookup. A lazy resolver fails immediately if invoked.
  Each operation also checks the source declarations and applicable metadata
  it must preserve, including `private`, `Ω`, interface method facts, and a
  structural const overlay.
- **20 compiler cells** cross generic method locals, generic function locals,
  nested locals capturing two type parameters, and two valid nearby controls
  with plain Go, LoGOs handling Go, Go-to-LoGOs interoperation, and a native
  structural overlay. Every cell runs a fresh module and requires its exact
  `generic local ok` output.

| Stage | Observed result | Evidence |
| --- | --- | --- |
| Stock Go oracle | Reduced program executes; exact upstream `cmd/digraph` builds. | [Minimal oracle](minimal-oracle.log), [upstream oracle](digraph-oracle.log) |
| Pre-repair fork | Reduced and upstream imports crash with an implicit type parameter error. | [Minimal RED](minimal-red.log), [upstream RED](digraph-red.log) |
| Reconstructed baseline production | Both cases reproduce the same error with baseline compiler source. | [Minimal baseline](minimal-baseline-red.log), [upstream baseline](digraph-baseline-red.log) |
| Frozen regression tests | 16 scope cells force forbidden lazy objects; 12 generic compiler cells crash. The other 20 cells pass. | [Frozen RED](red.log) |
| Attempt one, scope tests | All 28 scope cells pass. | [Unit GREEN](scope-green.log) |
| Clean restoration | New scope tests and existing Apply, CloneWith, and CopyFunc tests pass with `-count=1`. | [Restored gate](restored.log) |
| Rebuilt installed compiler | All 20 compiler/runtime cells and all 28 scope cells pass, with no skips. | [Installed gate](installed.log) |
| Full short package tests | Logos, importer, noder, lint, and types2 test suites pass individually. The combined command exits 1 because SSA's automatic vet process was killed. | [Interrupted combined gate](full-interrupted.log) |
| Dedicated retry | Standalone SSA vet exits zero. Noder's full short suite then exits zero with normal automatic vet enabled. | [SSA vet](ssa-vet-retry.log), [noder retry](noder-retry.log) |
| Earlier repairs and final checks | All 61 closer/panic regression cells pass; vet for all five affected packages, formatting, source identity, and frozen hashes pass. | [Earlier regressions](prior-regressions.log), [vet](vet.log), [formatting](format.log), [hash checks](checks.log) |
| Upstream runtime | All seven top-level tests and 33 subtests pass under each of three drivers, with original input bytes preserved. | [Runtime report](digraph/report.json) |

The empty stock-Go oracle logs are successful commands with no output. They
are build and reduction evidence; they are not included in the three upstream
package-test invocations below. The final vet and format logs are also empty
successful results.

### Baseline reconstruction

The pristine fork production baseline is
`bd318f1c558f3aa73ad434929fba7f237db9877f`. To distinguish this defect from
earlier dogfood repairs without replacing the installed toolchain, a Go source
overlay restored the six production files changed since that baseline. No
other non-test compiler production file differed. Their exact baseline bytes
are recorded in [baseline-source.sha256](baseline-source.sha256).

The [overlay](baseline-overlay.json) built a separate compiler, identified by
[its digest](baseline-binary.sha256). A [transparent wrapper](baseline-toolexec.sh)
selected that binary only for compilation; all other tools remained the
installed tools. This source reconstruction used the installed host toolchain;
its build metadata can differ from an original baseline build. The compiler
was invoked through a symlink named `compile`,
as required by Go's tool-version handshake. An initial launch under the name
`compile-baseline` was rejected by that handshake. Its [log](baseline-launch-rejected.log)
is retained but is excluded from behavioral evidence. The subsequent reduced
and real-code failures came from the correctly launched baseline compiler.

The checked-in overlay and wrapper preserve original evidence paths. Recreate
them with local paths when replaying; obtain each listed source from
`git show bd318f1c558f3aa73ad434929fba7f237db9877f:<path>`.

### Mutation controls

Both probes use source overlays and leave the production tree unchanged.

| Mutation | Observed failure | Evidence |
| --- | --- | --- |
| Invert the declaration identifier filter. | All 28 scope cells fail on forced compiler symbols or lost source declarations. | [Scope probe](scope-probe.log) |
| Invert the direct readonly lookup guard. | 12 cells fail on lost interface method metadata or forced compiler symbols. | [Readonly probe](readonly-probe.log) |

The unmodified production source passes the restoration gate afterward. This
slice has used one implementation candidate; the killed bootstrap process
does not change the implementation or establish a test result.

## Rebuilt toolchain

The first full rebuild was [interrupted](build-interrupted.log) when its
compiler process was killed during toolchain three. A serial retry used
`GOMAXPROCS=1`, `GOFLAGS=-p=1`, `GOMEMLIMIT=384MiB`, `GOGC=30`, and a disk-backed
temporary directory. The [successful build log](build.log) records dist's final
completion banner. The shell session's exit metadata was lost during session
compaction, so the [observation record](build-observation.json) preserves the
observed banner and removed bootstrap tool without inventing an exit status.

Compiler source remained fixed at the [recorded revision](build-source.txt).
The six final tool hashes are in [binaries.sha256](binaries.sha256), with their
reported [versions](versions.json). An initial test launch encountered a
dangling cache symlink because dist had removed its private cache at exit.
The [setup error](cache-launch-failed.log) occurred before any test started.
The cache was replaced with an independent directory before the passing
installed gate; this setup failure contributes no test count.

The first combined package command produced five passing test-package results
but exited 1 because SSA's automatic vet process was killed. The exact raw
failure is retained. Other validation jobs were paused for the standalone SSA
vet retry and the subsequent noder retry; both exited zero. No analyzer or
test was disabled. The final explicit vet command also passed for logos,
importer, noder, lint, and types2.

## Pinned upstream runtime

The [replay script](replay_digraph.py) verifies the complete module against its
pinned digest, checks the focused input manifest, creates an independent
private source copy for each driver, and runs:

```sh
go test -p=1 -count=1 -json -timeout=120s ./cmd/digraph
```

It requires exactly the seven expected top-level test names, a passing package
event, matching run/pass events for every subtest, and no failed, skipped, or
duplicate test events. The original input file digests must remain unchanged.
The report records the actual command, environment, exit code, elapsed time,
output hashes, driver/tool hashes, and compiler source digest.

| Driver | Package-test invocations | Top-level tests | Subtests | Result |
| --- | ---: | ---: | ---: | --- |
| Stock Go 1.27.4 | 1 | 7 | 33 | Pass, 1.534s |
| Rebuilt fork Go | 1 | 7 | 33 | Pass, 1.782s |
| Rebuilt LoGOs | 1 | 7 | 33 | Pass, 1.522s |

`TestToDot` executes the precise generic receiver method that exposed the
compiler error and checks the complete DOT output, including escaped labels.
The other tests exercise graph queries, path and cycle handling, quoting, and
focus operations. These are **three successful upstream package-test
invocations**. The unit checks, compiler-only cases, build commands, and vet
commands are excluded from that count.

## Reproduction

Put the repository's built toolchain first on `PATH`, set `GOROOT` to the
checkout, and use pristine Go 1.27.4 for `GOROOT_BOOTSTRAP`. The recorded runs
use linux/amd64, `GOTOOLCHAIN=local`, one package worker, and at most two Go
execution threads. These limits do not filter test cases.

From `src/`, the package gates are:

```sh
GOROOT_BOOTSTRAP=/path/to/pristine/go1.27.4 ./make.bash

go test -p=1 -count=1 -short -timeout=30m \
  cmd/compile/internal/logos \
  cmd/compile/internal/importer \
  cmd/compile/internal/noder \
  cmd/compile/internal/logos/lint cmd/compile/internal/types2

go vet -p=1 cmd/compile/internal/logos \
  cmd/compile/internal/importer cmd/compile/internal/noder \
  cmd/compile/internal/logos/lint cmd/compile/internal/types2
```

The dedicated retry used `go vet -p=1 cmd/compile/internal/ssa`, followed by
`go test -p=1 -count=1 -short -timeout=30m cmd/compile/internal/noder`.

The focused attack command is:

```sh
go test -p=1 -count=1 -short cmd/compile/internal/logos \
  -run '^Test(CompileImportedLocalType|FactScopeLeavesCompilerSymbolsLazy)$'
```

From the repository root, verify frozen tests and production input with:

```sh
sha256sum -c logos-doc/validation/imported-local-types/frozen-tests.sha256
sha256sum -c logos-doc/validation/imported-local-types/production.sha256
```

Replay the upstream runtime check with a new output directory and prepared
module/build caches:

```sh
uv run --no-project python \
  logos-doc/validation/imported-local-types/replay_digraph.py \
  --oracle /path/to/pristine/go1.27.4/bin/go \
  --candidate "$PWD/bin/go" --logos "$PWD/bin/logos" \
  --module-cache /path/to/module-cache --cache-root /path/to/build-caches \
  --tmpdir /path/to/temp --out /path/to/new-output
```

The replay passed Ruff 0.17.5 formatting/lint checks and `ty` 0.0.104
[typechecking](replay-checks.log). The [file manifest](files.sha256) covers the
preserved evidence and replay script.

## Scope

The repair is confined to imported fact and overlay traversal. These checks
do not claim a full upstream `run.bash` pass or a performance improvement.
The earlier lint repairs and runtime campaign retain their separate
[compiler evidence](../dogfood-lint/README.md) and
[runtime evidence](../../../misc/logos-dogfood/evidence/2026-11-16/runtime/README.md).
