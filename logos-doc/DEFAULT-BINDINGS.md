# Struct defaults: retained validator and blocked binding work

## Status and scope

The declaration validator, M6-S3a, is retained. The binding implementation,
M6-S3b, is blocked and has been removed after its third implementation
attempt failed installed-compiler tests on 2026-11-16. This follows G1's
three-attempt limit, required by AGENTS.md. The following source-level
results describe the rejected candidates; they do not establish a working
native binding implementation.

The rejected snapshot is commit
`96a5a8cc599cbcc55263d32c4ceb2348abc6f0fd`. Its direct lowering suite passed,
but both installed exported-package tests and the allocation probe failed
with an internal compiler panic. The exact diagnosis and recovery are
recorded below. The retained validator still needs its separate fresh
installed-toolchain gate and the full hosted regression.

The repair has two obligations. A default must be checked in the scope where
it was declared, even if its type is never constructed. Applying that default
must preserve the declaration's bindings and exact value types when a
constructor appears in another scope or package. Explicit initializers must
win, and separate composite defaults must create separate mutable storage.
The corresponding contract is [SPEC.md](SPEC.md), Table T row
T-104.5, and slices M6-S3a and M6-S3b in [IMPLEMENTATION.md](IMPLEMENTATION.md).

The main checkout used for baseline probes retained the original language
implementation. It included the independent SIMD repair and was based on
`949e633789fe5122c548d6c9f39c06fbb2769702`. Candidate language work lived in
the separate `LoGOs-defaults` worktree.

Raw logs named in this report are session artifacts under
`/workspace/scratch/fd0d13fb6a48/evidence`. They are not part of this repository.
The test sources and their hashes below identify the reproducible attacks.
Here, RED means that the relevant semantic assertion failed before its repair;
a test compilation error, skipped run, or environment failure is identified
separately.

## Confirmed failures

| Check | Observed result before repair | Evidence |
| --- | --- | --- |
| Unused invalid declarations | `uint8` defaults below 0 and above 255, wrong scalar and composite types, and an effectful map key were accepted. The 0 and 255 controls passed. | `defaults-baseline-red.log` |
| Declaration scope | Imported `time.Second` failed with `undefined: time`; a local constant failed with `undefined: local`. | `defaults-baseline-red.log` |
| Application scope | Native and ejected constructors used constructor-local constants instead of declaration constants. An alias to an anonymous struct had the same failure. | `defaults-baseline-red.log`, `defaults-shapes-red.log` |
| Generic default checking | A default `[]T{}` in a generic struct failed with `undefined: T`. | `defaults-shapes-red.log` |
| Exported bindings | The consumer program panicked because defaults captured consumer names. | `defaults-export-red.log`, `defaults-export-ejected-disk-red.log` |
| Exported private fields | Lowering produced a consumer struct literal that tried to name the producer's unexported field. Native compilation and corrected-environment ejection rejected it. | `defaults-export-ejected-disk-red.log` |
| Blank-field lowering | Generated code contained both keyed `_:` elements and assignments such as `_logos_v._`; neither is a legal field reference. | `defaults-blank-lowering-native-red.log`, `defaults-blank-lowering-eject-red.log` |
| Blank-field source identity | The first validator repair selected a field type by its name. Repeated `_` fields therefore used the first blank field's type: valid declarations were rejected and invalid ones accepted. Six attack cases failed while two controls passed. This was a defect in the intermediate validator, not a separate untouched-baseline result. | `structdefault-blank-red.log` |
| Helper metadata | The old decoder did not recognize `defaulthelper`; the old importer accepted legacy expression-only defaults without a helper. The four new wire tests failed before the boundary change. | `default-helper-wire-red.log` |

The retained blank-field validator uses the field's source position to find
its checked type. The proposed omission of generated blank assignments was
part of the blocked binding implementation and is not retained.

Additional candidate attacks exposed local-derived type handling, generic
substitution, inferred array spelling, capture planning across files, and
generated-name capture. These failures were retained rather than hidden by
changing their assertions. Relevant logs include `default-origin-red.log`,
`defaults-hygiene-red.log`, `default-multifile-red.log`, and
`default-implicit-shadow-red.log`. Their concrete diagnostics included
`undefined: int0`, `undefined: Item`, invalid use of `[...]` outside a
composite literal, and a constructor-local `Cfg` or `Alias` being used as a
type.

The local-derived, inferred-array, and implicit-element type-shadow fixtures
were also run verbatim with the baseline compiler. Native execution, ejection,
and execution of ejected Go all passed. Those results identify preservation
regressions in the candidate, rather than new language requirements. The logs
are `defaults-derived-baseline-*`, `defaults-inferred-array-baseline-*`, and
`defaults-implicit-element-baseline-*`.

## Fixture and environment calibration

### Helper-boundary test compilation

The first helper-boundary harness declared `Info.Scopes` with
`map[*syntax.Node]*types2.Scope`. The API requires
`map[syntax.Node]*types2.Scope`, because `syntax.Node` is already an interface.
`default-origin-test-compile-error.log` records that compilation error. The
map declaration was corrected without changing the helper signatures,
hostile metadata, or behavioral assertions. The manifest below records the
corrected file. This harness error is not evidence of a compiler defect or a
semantic RED result.

### Ejection and temporary directories

The first exported ejection runs stopped with `error obtaining VCS status:
exit status 128`. The temporary module inherited an unusable VCS ancestor
under `/workspace`. Although `GOFLAGS` contained `-buildvcs=false`, the eject
command did not register that flag, so it did not suppress this path.

Both `TMPDIR` and `GOTMPDIR` were moved inside a real worktree, under its
ignored `pkg/validation-tmp-fd0d13fb6a48` directory. Setting only `TMPDIR` was
insufficient because the test harness also used `GOTMPDIR`. The directory was
on disk; `/tmp` and `/var/tmp` were memory-backed in this environment. The
fork's `bin` directory also had to be first on `PATH` when running saved test
binaries, so `internal/testenv` found the intended Go tool.

After those environment corrections, exported ejection reached the actual
binding and private-field failures recorded in
`defaults-export-ejected-disk-red.log`. Earlier VCS failures and skipped runs
do not count as semantic RED evidence.

### Predeclared Boolean names

The original package-level fixture declared `type bool = int` and
`const true = 7` alongside two defaulted scalar fields. It failed on the
baseline because existing automatic serializer code used the source name
`bool`. The diagnostics were `cannot use false ... as bool` and an invalid
`!` operation on an integer. That failure did not isolate the new presence
mask.

The calibrated fixture adds only an unused `map[int]int` field. This field
disables automatic JSON and binary generation for the fixture. The original
defaulted fields, values, source shadows, constructors, and assertions remain.
The calibrated package-level source passed baseline native execution,
ejection, and ejected execution; both runs printed `ok predeclared shadows`.
The candidate then failed specifically in its generated presence mask, as
recorded in `defaults-predeclared-calibrated-red.log`.

The retained uncalibrated test has SHA-256
`e3e7155137550ad4bb733813a7d9a8a4ab24cf0e1d8b0a2c82310eab57fe46cb`.
The calibrated frozen test has SHA-256
`0fdcb69a5a56c832fcc02a0432252b9eb333457e41016aab1cf7772a9c53381f`.
This isolates default-mask hygiene; it does not claim to repair serializer
behavior under package-level predeclared-name shadowing.

### Final shadowed-`new` attack

Candidate 2 converted `&Cfg{}` into a call wrapped by unqualified `new`.
It also used unqualified `new` for implicit pointer elements. Under a legal
local `new := 0`, both generated forms resolved to the local integer.
`default-new-shadow-red.log` records two `cannot call new (variable of type
int)` diagnostics in each of the native-tree and ejected-text runs.

`TestStructDefaultNewShadow` keeps both forms, explicit-field controls, exact
pointer types, and independent-storage assertions. Its combined baseline
result needs a precise label: the baseline compiled both forms, but failed
the value assertion because it did not apply the default to the implicit
element in `[]*Cfg{{}}`. Baseline ejection shows `&Cfg{N: 17}` for the explicit
address and leaves `[]*Cfg{{}, {N: 29}}` unchanged.

A separate raw explicit-address-only control passed baseline native execution,
ejection, and ejected execution. Both executions printed
`ok explicit shadowed new`. Thus explicit addressing is a confirmed
preservation regression; implicit pointer default omission is a pre-existing
defect. The combined frozen test was not weakened to conflate those results.

The explicit control source has SHA-256
`4caa2d35910808689296acf53f6dbe4711107c12fa236ca04496ea90a513953c`.
Both passing execution logs have SHA-256
`63121f824d6069c33b54c8e94e584a986bb9619f55473993e630894335f06460`.
They are named `defaults-new-shadow-explicit-baseline-native.log` and
`defaults-new-shadow-explicit-baseline-ejected-run.log`. The combined baseline
logs are `defaults-new-shadow-baseline-native.log` and
`defaults-new-shadow-baseline-ejected-run.log`.

## Frozen attack manifest

Paths in this table are relative to `src/cmd/compile/internal/logos/`.
These hashes identify the corrected and calibrated fixtures used for the
candidate review. No assertion was removed to obtain a passing result.
They match `default-attacks-final.sha256`, whose own SHA-256 is
`e4a4b1b47c9169c8e033439312dd9a093fcfd2f146accc444506baf6c7b300ac`.

| Test file | SHA-256 |
| --- | --- |
| `structdefault_binding_test.go` | `d6db2aef195c19b1072d8716082382c0c689da781cd026b506a197284034dcb4` |
| `structdefault_blank_lowering_test.go` | `caf46dae632857e01dd125ae2ff995181cf8d2fae8fedfd3a975dd075620fd7a` |
| `structdefault_blank_test.go` | `36288ae8a2ff903f8592b398927778c5585ca657f532581ca468e8ed8384edfe` |
| `structdefault_derived_test.go` | `48bd7ac9ffaf6242c25409d280feaab801cda4d9a02c98cf4e61cd6a128b130a` |
| `structdefault_generic_capture_test.go` | `c94f84908a7e08dd716c5dae39d779738e0b41e6c01ae2825b78ca0cdbef2b0c` |
| `structdefault_helper_boundary_test.go` | `d58563417b5139381021db6cb7374fe0179ac793a9cde0ff9b1dae0a333edee8` |
| `structdefault_implicit_shadow_test.go` | `7de5824b61181c5908dd2900caf3ad1fb52b5bbdf67f6d87ff6e0d93ece9c8a4` |
| `structdefault_inferred_array_test.go` | `70e699dd376929b10ed778feb1462435b4da9a40141c091f9a47b9d87c55cd6b` |
| `structdefault_multifile_test.go` | `39de51957596c655080da953e50c888fcf4bd291913332a7d09b98dae88f417c` |
| `structdefault_new_shadow_test.go` | `b79814adff6e1a870d4c6d304afc0cdceadab4701bb46a8fbb0deafa5da7423a` |
| `structdefault_origin_test.go` | `ee859ed1ea15e70497c3c860d6e8de4c70cb32af63ba8e15e3fad990aed90f5c` |
| `structdefault_predeclared_test.go` | `0fdcb69a5a56c832fcc02a0432252b9eb333457e41016aab1cf7772a9c53381f` |
| `structdefault_shapes_test.go` | `977fac5dbf5fe1cb4fd270f844cea481feeaae4ae4be28b35767dfea45c018b8` |
| `facts/default_helper_wire_test.go` | `9c0ec0aec4b33fa502713f106d145ea7461c1fc7cc219819e6e451a7327863c6` |

## Review of the rejected binding candidate

These findings describe the proposed implementation. The installed native
failure exposed a type-information storage assumption that the direct
harness and this source review did not catch. The helper APIs below are
not present in the retained implementation.

The review covered `structdefault_binding.go`, `structdefault.go`,
`facts/write_default.go`, the facts import boundary, and JSON reuse. The
following invariants were checked against the frozen attacks.

**Declaration checking.** Delayed checking restores the declaration scope,
uses the checked field type, proves recursive constant values and map keys,
and performs the ordinary assignment check. Callers may omit `Info.Types`;
the validator creates the temporary map it needs and restores the caller's
nil state. Repeated blank fields are distinguished by source identity.

**Producer ownership.** Facts writing and helper emission use the same
checked-origin collector. Package helpers execute defaults in the source
file and package scope. They are ordinary functions, so adding them does not
change the original type's method set. Renamed generated type parameters
avoid hiding names used by the original declaration.

**Imported helper boundary.** The decoder accepts only identifier-shaped
helper names in the reserved `Logos_` namespace. Exact and conflicting
duplicate type mappings are rejected. Import checks the fingerprint and
requires a helper mapping for every described default-bearing type before
changing the registry. Schema 3 is retained with fail-closed behavior: an old
reader rejects the new record; a new importer rejects legacy defaults without
helpers and requests a dependency rebuild.

The consumer requires a function object, exact generic arity, valid
instantiation constraints, and the nonvariadic signature `(T, [N]bool) T`,
where `N` is the struct's total field count. Type identity includes aliases
and private producer types. Imported `DefaultFact.Text` has no execution or
reparse path in the consumer. The hostile-text fixture runs an ordinary
checked producer function and verifies that the injected expression never
appears in either generated form.

**Constructor behavior.** The original literal remains the helper's first
argument, preserving explicit initializer evaluation and ordering. Presence
masks preserve explicit fields and copied spread fields. Blank fields are
never assigned. Composite expressions execute separately for each omitted
field application, preserving fresh mutable storage. A transparent runtime
alias to the built-in Boolean type and constant comparisons prevent source
`bool` and `true` declarations from changing generated masks.

**Scope and exact types.** Local defaults capture constants and types in
their declaration scope without inserting runtime variables that would alter
goto legality. Capture planning precedes per-file lowering. Inferred array
aliases use the checked length. Local derived types convert through
declaration-bound base and result aliases. Reserved consumer import aliases
avoid constructor-local import shadows; implicit named elements use reserved
type aliases before their contextual type is materialized. Generic named
arguments retain type-argument inference.

**JSON reuse.** JSON default initialization calls the producer helper before
decoding fields, retaining the existing non-defaulted fields and applying
the same checked defaults. This removes a second path that emitted direct
blank-field assignments or rebound default expression text.

Candidate 2 had no further concrete blocker in these reviewed paths. Its
unqualified `new` handling remained a blocker and required the final
correction. Source review is not a substitute for the installed compiler
and ejection gates.

## Candidate 2 results

`default-binding-candidate2-unit.log` shows passes for the direct declaration,
lowering, helper-signature, hostile-text, blank-field, derived-type, generic,
inferred-array, cross-file, and calibrated name-shadow tests present in that
run. The separate shadowed-`new` attack was added after this run and failed
as described above.

The candidate 2 package command still exited FAIL. The two exported tests
invoke the installed `bin/logos`, which had not yet been rebuilt with the
binding repair; they repeated the consumer-name and private-field failures.
Those tests remained pending for the rebuilt compiler. The direct tests are
useful source-level evidence, but this run is not an overall passing gate.

The complete facts package passed in `default-facts-candidate1.log`, including
the four new wire tests. Later changes still require the final applicable
gates. The important evidence hashes are:

| Log | SHA-256 |
| --- | --- |
| `defaults-baseline-red.log` | `df172f87e674e700383a88a3ac59c7eb9add69f673ee7d273cf07ee6115069e4` |
| `default-helper-wire-red.log` | `e3192730786227a42ac59223ffa0ba57c076e073de5572706ccc50e011ef8ed4` |
| `default-facts-candidate1.log` | `6286299b01edf791c11674a9d656299d36ec061e91137dfd157795dd72bb4bf1` |
| `default-binding-candidate2-unit.log` | `4b24412107faccec7735fd658f84419e55a59ae02d7c74248149cdfafde182ed` |
| `default-new-shadow-red.log` | `78cec597eea76e5ac85a43d8bd562bf583fb2360a8c865b995cf508433c889fb` |

## Candidate 3 source tests and mutation checks

The final pointer correction uses the reserved qualified
`logos/rt.Logos_DefaultAddr` helper, which returns the address of its value
parameter. Defaulted pointer constructors no longer need to resolve `new`
in the constructor's scope. The helper retains exact type inference and
creates a fresh variable on each call.

The following source-level test command passed, including the unchanged
combined `TestStructDefaultNewShadow` attack:

```sh
go test -count=1 -short cmd/compile/internal/logos \
  -run '^TestStructDefault' -skip '^TestStructDefaultExported'
```

`default-binding-candidate3-unit.log` records PASS in 5.063 seconds. The
exported cases were explicitly excluded because they launch the installed
fork and need the rebuild. The full facts package passed in 0.018 seconds,
and the full types2 package passed in 23.057 seconds, recorded in
`defaults-facts-types2-green.log`.

Four separate `go -overlay` mutations each restored one fault without editing
the production files. Each probe failed for its intended semantic reason:

| Isolated fault | Observed failure | Log |
| --- | --- | --- |
| Select the default field by name instead of source position | Six repeated-blank identity cases fail; valid and invalid declarations are both affected. | `default-mutant-field-identity.log` |
| Disable the required-helper check | Both missing-mapping cases are accepted, and both detect unwanted registry insertion. | `default-mutant-missing-helper.log` |
| Disable mask type identity validation | Short, long, wrong-element, and slice masks are incorrectly accepted. | `default-mutant-helper-mask.log` |
| Invert the producer's `!present` condition | Declaration-binding values fail in both native-tree and ejected-text runs. | `default-mutant-helper-fill.log` |

These results show that the tests detect the guarded faults. They do not
replace the installed-fork tests or the full regression gate.

| Log | SHA-256 |
| --- | --- |
| `default-binding-candidate3-unit.log` | `c522fab42f547058704d9b0c6d9c3b08cdea6b4dffe43683924a314cdda8c8eb` |
| `defaults-facts-types2-green.log` | `99e131215afc595b00f49d82722d71b867771e53e9673ea8dbcf2edf20888b71` |
| `default-mutant-field-identity.log` | `7b5f35a83a1b7c3939b52ae56e5d17e40951b2505f1981a9da1da8a29441a84b` |
| `default-mutant-missing-helper.log` | `9e0077c616f9cb1365b3ace226ea2291df8f172a7603fddca8f764495b0a0919` |
| `default-mutant-helper-mask.log` | `8b178c8b93f5b75c765b171a4efe20a69fca8518107086e59b7b434b2e073e6d` |
| `default-mutant-helper-fill.log` | `9d27abe2af5793beb0d3845f7e9f13e054e312801749b736d5d9cd847a243134` |

## Allocation baseline

A separate probe measured one representative nonescaping construction on the
unchanged language compiler. It declares 32 scalar defaults, a four-element
slice default, and a two-entry map default. A noinline function constructs,
mutates, and reads the value locally, then returns only an integer checksum.
`testing.AllocsPerRun(10000, ...)` measures that work; printing is outside the
measured closure. The probe uses two-result map indexing, as required for
ordinary value reads in loGo/s. An initial one-result-map diagnostic was
corrected before recording the measurement.

The command exited 0 with:

```text
default_fields=34 repetitions=10000 allocations_per_run=0.000000 checksum=557
```

The toolchain reported `go1.27.4-X:logosarena`, linux/amd64, `GOAMD64=v1`, and
`CGO_ENABLED=0`. The saved command was:

```sh
bash /workspace/scratch/fd0d13fb6a48/evidence/run-defaults-alloc-probe.sh \
  /workspace/scratch/fd0d13fb6a48/LoGOs \
  /workspace/scratch/fd0d13fb6a48/evidence/defaults-alloc-baseline.bin \
  > /workspace/scratch/fd0d13fb6a48/evidence/defaults-alloc-baseline.log 2>&1
```

| Artifact | SHA-256 |
| --- | --- |
| Probe source | `6cb0f0c37c863e8382c3b5aa07e1e834c445e71735c4064c80dfe08d1b03f051` |
| Baseline compiler | `b47afa6b98908ca155a1a3e2ebf60230238be7fd96bc2f0c6109edf67b4569ab` |
| Measurement log | `db24ccdf56705db991445fe42fb8ed661c0b007a7e46244dfd9a4901c84d2481` |
| Probe executable | `88a9e52294c47c475e3bb19689d83bd1638d0fc64a07b4ae0c6e174feac206cf` |

This is a baseline observation. The installed candidate failed to compile
the identical workload, so it produced no allocation measurement or timing
result. No allocation improvement or regression can be inferred from that
failed run. The workload is retained to check the independent validator
and any future binding design.

## Retained validator and unresolved binding behavior

The retained validator commands must run without cached results. The
installed toolchain must be rebuilt after removing the rejected lowering:

```sh
# From src/, with the fork's bin directory first on PATH:
./make.bash
go test -count=1 -short cmd/compile/internal/logos -run '^TestStructDefault(LexicalScope|RejectsInvalidDeclaration|BlankFieldIdentity)$'
go test -count=1 -short cmd/compile/internal/logos/facts
go test -count=1 -short cmd/compile/internal/types2
go test -count=1 cmd/go -run 'Script/^logos_(m6_addr_defaults|eject_struct_default)$'
go test -count=1 cmd/internal/testdir -run 'Test/logos/'
./run.bash --no-rebuild
```

Local probes bounded build resources with `GOFLAGS='-p=1 -buildvcs=false'`,
`GOMAXPROCS=2`, `GOGC=50`, and `GOMEMLIMIT=1GiB`. Both temporary-directory
variables pointed inside the valid disk-backed worktree. Package tests,
script tests, and successful isolated examples do not replace the full Go
regression floor. The full rebuilt regression run remains a merge gate.

The original application-binding, private-field, and blank-lowering defects
above remain unresolved. Two further language behaviors also remain outside
the retained validator. Exported anonymous
structs nested in container types need declaration provenance by exported
type path; the current named/default records do not identify that origin.
Also, a default containing a literal of another default-bearing type retains
the previous nested behavior. The baseline reproduction in
`default-nested-baseline.log` printed `7 0` for direct `Inner{}` and an outer
default containing `Inner{}`. Neither behavior is claimed as repaired here.

Ordinary Go struct literals continue to use Go zero initialization.

## Installed failure and bounded recovery

Two local `make.bash` attempts ended with a killed SSA compiler process.
The first used two workers and a 1536 MiB soft memory limit and stopped
during toolchain3. The second used one worker and a 768 MiB limit. It
completed toolchain1, toolchain2, toolchain3, and standard-library building,
then stopped during final command installation. These are failed build
attempts, not successful G2 gates; neither produced a source diagnostic.
The retained logs are `defaults-candidate-build-resource-failure.log` and
`defaults-candidate-build.log`.

The documented incremental recovery command, `go install std cmd`, passed
against the retained build cache with one worker, `GOGC=20`, and a 512 MiB
memory limit. The installed Go and logos commands both reported Go 1.27.4.
The rejected compiler binary had SHA-256
`673bdff8dd21dda65d8164b5253fff134cce205c869fa08306dcc05a13d54eab`.

The subsequent uncached installed gate failed in
`TestStructDefaultExportedConstantBindings` and
`TestStructDefaultExportedPrivateFields`, in both native and ejection
operations. The allocation probe failed during compilation too. The logs
are `defaults-installed-focused-rejected.log`,
`defaults-alloc-candidate3.log`, and `defaults-installed-panic-trace.log`.
The traced failure is:

```text
panic: assignment to entry in nil map
cmd/compile/internal/logos.(*fileLower).fillStructDefaults
    cmd/compile/internal/logos/structdefault.go:153
```

Native noder supplies `Info{StoreTypesInSyntax: true}` without `Info.Types`.
The rejected constructor rewrite read and assigned the optional map
directly. Its capture planner also read constant values only from that map.
An empty replacement map is not a repair: `Info.TypeOf` falls back to syntax
storage only while the map is nil, so allocating one would hide previously
checked node types. A future design must use the existing dual-storage
type-information API and test the actual native configuration.

Under G1, the eight modified binding implementation files were restored and
the three new implementation files removed. The independent validator was
kept unchanged. Its lexical-scope and invalid-declaration tests, plus their
two execution helpers, were moved with byte-identical function bodies to
`structdefault_validation_test.go`. The original blank-field identity file
was retained unchanged.

[M6-S3b.patch](blocked/M6-S3b.patch) contains only the blocked tests. It
removes the extracted validator test file before restoring the original
combined file, avoiding duplicate test functions. A clean replay check
restored all 14 hashes in the frozen manifest above. Its SHA-256 is
`3d1bd84ac779294170db7d8e5e3d6403d005be31b94a634f5f9e4a16ae37d012`.
The helper-boundary tests depend on the internal API in the rejected
snapshot; the patch alone is an attack archive, not a repaired compiler.

Fresh installed validation of the retained checker and the full hosted
regression remain merge gates. No fourth binding repair was attempted.

The exact allocation workload is also checked in as
[struct-default-allocations.lgo](benchmarks/struct-default-allocations.lgo).
It retains the frozen probe-source hash above. Allocation observations do
not establish a latency improvement.
