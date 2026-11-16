# Dogfood compiler and lint validation

Two defect classes are repaired: ordinary error handling after `Close` no
longer creates a spurious leak finding, and a returning function named
`panic` no longer proves an option non-nil or hides infinite recursion.

The final compiler uses production source from local revision
`4cbfbf5af30d99609d385afbecfa305026a32307`. The baseline was
`bd318f1c558f3aa73ad434929fba7f237db9877f`. Publication can give equivalent
source a different commit identity; [production hashes](production.sha256)
and [built binary hashes](binaries.sha256) identify the verified inputs.

## Repaired behavior

### Close results

`if err := f.Close(); err != nil { ... }` calls `Close` before handling its
result. The old rule recognized expression and return statements but
missed direct calls used as assignment or variable-declaration values.

The repair uses the existing object and alias tracking for direct values,
including parallel assignments and `if`, `for`, and `switch` initializers.
It does not descend into conditional expression operands. The controls
keep distinct and shadowed resources separate and retain the existing
behavior for asynchronous calls and `errdefer`.

The two old `explicitClose` fixtures incorrectly required a leak finding.
Their executable source stayed unchanged; the incorrect expectations were
removed in a separate test-only commit before implementation.

### Panic identity and option safety

Both lint flow analysis and option narrowing trusted the spelling `panic`.
For example, a user-defined `func panic(any) {}` could return normally from
a nil branch, yet the compiler accepted a later optional-pointer field
access and wrote an object file. Parentheses also hid the real builtin
from the recursion rule.

Lint now resolves the checked builtin and shares one termination helper.
The option checker uses its recorded builtin-call identity. The full
types2 suite caught an incomplete second candidate: upstream records
panic calls for result-bearing function termination, while loGo/s also
needs the facts in functions without results. The final candidate records
them for all loGo/s function result arities. Plain-Go registration retains
its previous condition.

## Test-first sequence

| Stage | Observed result | Evidence |
| --- | --- | --- |
| Unchanged baseline | 24 assigned-close cells report extra leaks; three typed-nil and six recursion identity cells fail; both assigned-close compiler cases fail. | [Initial RED](red.log) |
| Native option attack | Three unsafe shadowed-panic programs compile successfully and produce object files. | [Compiler RED](compiler-red.log) |
| First panic candidate | Fixing lint alone leaves three typed-nil attacks red because types2 already erased the option. | [Candidate one](candidate1.log) |
| Second panic candidate | The complete lint package passes, but the unchanged `TestLogos/narrowing.lgo` rejects a valid void-function panic guard. | [Candidate two](candidate2.log) |
| Void preservation attack | Both real builtin guards in the new native void-function test fail against candidate two. | [Void RED](void-red.log) |
| Final candidate | Rebuild, full short lint/types2 suites, installed compiler cases, relevant lint fixtures, vet, formatting, and whitespace checks pass. | [Build](build.log), [suite GREEN](green.log), [fixtures](fixtures.log), [vet](vet.log) |

The four new test files contain **61 attack and control cells**: 32 resource
cells, 18 flow-identity cells, and 11 installed-compiler cells. Their
[frozen hashes](frozen-tests.sha256) have not changed since their respective
RED executions. No final test command uses a skip filter.

## Mutation controls

The probes use Go source overlays. The repository's production files stay
unchanged while the altered package is compiled and tested.

| Mutation | Required failure | Evidence |
| --- | --- | --- |
| Invert the `closeValues` nil guard. | Assigned-close cases regain spurious leak findings. | [Closer probe](closer-probe.log) |
| Invert the option checker's `check.isPanic[call]` predicate. | Shadowed and ordinary returning calls erase required typed-nil findings. | [Panic probe](panic-probe.log) |
| Remove loGo/s registration from the builtin panic condition. | Unchanged `TestLogos/narrowing.lgo` rejects `earlyPanic`. | [Void probe](void-probe.log) |

Clean-source runs restore [the new unit attacks](restored.log) and
[the existing narrowing fixture](void-restored.log). An initial malformed
mutation left an unused local; its [build error](invalid-probe.log) is
retained and is not counted as a behavioral probe.

## Reproduction

Use the repository's built toolchain first on `PATH`, set `GOROOT` to this
checkout, and use a pristine Go 1.27.4 tree for `GOROOT_BOOTSTRAP`.
The recorded run used linux/amd64, `GOTOOLCHAIN=local`, one package worker,
and at most two Go execution threads. These are resource limits, not test
filters.

From `src/`:

```sh
GOROOT_BOOTSTRAP=/path/to/pristine/go1.27.4 ./make.bash

go test -p=1 -count=1 -short \
  cmd/compile/internal/logos/lint cmd/compile/internal/types2

go test -p=1 -count=1 -timeout 10m cmd/internal/testdir \
  -run '^Test$/^logos$/^lint$/^unclosed_io_closer'

go vet -p=1 cmd/compile/internal/logos/lint cmd/compile/internal/types2
```

From the repository root:

```sh
sha256sum -c logos-doc/validation/dogfood-lint/frozen-tests.sha256
sha256sum -c logos-doc/validation/dogfood-lint/production.sha256
```

## Scope

The closer rule remains its existing syntactic ownership analysis. These
changes do not establish that every control-flow path closes a resource.
The compiler and lint gates above are scoped correctness evidence; they
do not claim a complete upstream `run.bash` pass or a performance gain.
The external corpus runner and its separate results live under
[`misc/logos-dogfood`](../../../misc/logos-dogfood/README.md).
