# Tagged receiver method facts: frozen regression slice

## Status and boundary

This is a test-only slice based on
`2c337536177f90957a8786e533ccd7a901596747`. No production code or existing
test is changed. No Go test, compiler build, or target program was executed
while preparing this slice. The current reachability conclusion comes from
source inspection; execution and the resulting RED/GREEN evidence are pending.

The chosen repair preserves the method spelling already emitted and consumed:
`Box@simd128.Read` and `(*Box@simd128).Read`. The codec must accept these two
receiver-tag forms while retaining every legacy accepted form, including the
distinct method-tag forms `Box.Read@simd128` and `(*Box).Read@simd128`. It must
not normalize or merge these identities. The package-declaration filter must
also reject any name containing a dot, before a scope lookup can resolve a
lazy method symbol. Existing frozen package-declaration tests remain unchanged.

An initial, uncommitted draft considered moving all tags to the end of the
method key. The completed consumer inventory showed that this would require
unnecessary changes across several subsystems. That direction was withdrawn
before freezing or executing any tests; the final tests preserve existing
construction and lookup behavior.

## Current source-derived trigger

The existing `src/simd/testdata/iface/iface.go` source defines the non-generic
`VectorC` type with a `simd.Float32s` field, value methods `Foo` and `Bar`, and
the pointer-receiver `Q.Qux` method. These methods have explicit parameters
and bodies. The file is copied without modification into a private ordinary
Go module by `TestCompileSIMDTaggedMethodFacts`.

The path to the incompatible key is:

1. `midway/analysis.go` marks the receiver types and methods SIMD dependent.
   `midway/deepcopy.go` tags dependent type names but deliberately preserves
   method names. `midway/midway.go` supplies sizes 0, 128, 256, and 512 on amd64.
2. `noder/irgen.go` rechecks the rewritten declarations before
   `logosWriteCheckedFacts` runs.
3. `cmd/go/internal/work/logos_lint.go:logosGoFactsFlags` enables
   `-logos.gofacts` when `cfg.Logos` is true, the package is ordinary Go, and
   it has no standard-library provenance. This is why the private module
   matters. A normal standard-library `simd` build is a different profile.
4. `facts.EmitGoMutations` emits explicit masks for exported methods with
   bodies and at least one explicit parameter. It does not require a generic
   receiver. `constFuncName` currently calls `MethodObject` with the already
   tagged receiver name.
5. `facts.Encode` writes the nonempty method record without validating the
   object name. `facts.Decode` splits at `@` first, rejects the dot or closing
   parenthesis in the remaining tag, and reports a bad object.
6. `noder.readImportFileImpl` calls `readLogosFacts` and `facts.Import` before
   choosing either the Go or loGo/s metadata consumer. An invalid method
   record therefore prevents importing the archive.

`VectorC.Foo`, `VectorC.Bar`, and `(*Q).Qux` pass or return reference-free
vector values. `constBodyScan.mark` leaves explicit value arguments unmarked
through `types2.LogosReferenceFree`. The integration test therefore requires
an explicit `ParamMutated=0` for all twelve tagged method records. It also
requires the current conservative receiver flags: true for `Foo` and `Bar`,
false for `Qux`. The first two call `Add`; its standard-library package has no
Go mutation record, so `calleeFacts` returns an unknown receiver effect and
`doFuncCall` uses `markMentionWrite` on the receiver. `Qux` makes no call and
does not reference its receiver. An independent source review confirmed this
distinction before freezing. No receiver-purity assumption is substituted for
the actual writer behavior. An absent record, unknown mask, duplicate,
malformed archive, or failed import cannot satisfy the test.

The fork-Go baseline builds the same private source and consumer with
`GOEXPERIMENT=none,simd`. Its driver does not enable Go mutation facts. The
test checks that any baseline function records have no `ParamMutated` field.
The `logos` profile then builds and decodes the real archive and compiles the
consumer. Neither profile executes a target program.

For this source, the rewrite creates exactly one tag per method key. A
SIMD-dependent receiver makes `HasDependentSignature` true, so
`Analyzer.markIfDependent` marks its method in `isDependentMethod`.
`DeepCopier.OnName` then preserves the method name while tagging the receiver
type. The body-only dependency phase can instead mark a method with an
independent signature; it marks the function, not the receiver type. The
rewriter also runs only once before rechecking (`didMidway`). The grammar
controls reject repeated tags and keys tagged at both receiver and method.
They preserve legacy method-local tags as separate identities. This argument
is scoped to the typed dependent declarations under review, not arbitrary
synthetic names supplied by future compiler transforms.

## Other writer conditions

`writeAsserts` runs after the SIMD rewrite for both drivers. Its method branch
requires a function or receiver type parameter and known assertion effects.
The current SIMD example's concrete methods do not meet that condition.
A new generic SIMD-dependent receiver could reach it in plain fork-Go, but
that is not needed for the existing-source `logos` reproduction.

`facts.Write` produces readonly and option/const declaration facts during
loGo/s lowering, before the SIMD rewrite. Those passes do not by themselves
prove a current post-rewrite failure for ordinary Go SIMD methods. Synthetic
writer and lookup controls nevertheless freeze their method-key contract.
The lock writer and wasm yield writer validate names before emission; their
current handling of receiver-local tags omits those records rather than
emitting an invalid record. This slice does not claim a measured failure in
those separate profiles.

## Key construction and lookup inventory

| Location | Role |
| --- | --- |
| `facts/methodobject.go:MethodObject` | Shared type-name and receiver constructor. Used by exported signatures, assertion facts, nested readonly facts, mutation facts, nil-error facts, and concrete overlay method lookup. |
| `facts/methodobject.go:ObjectName` | Independent constructor used by nil-error consumers, comptime record/lookup, and lock record/lookup. |
| `facts/write_const.go:emitReadonlyFacts` | Separate raw concatenation for named interface method records. |
| `optfacts.go:collectNamedFuncFacts` | Shared constructor for concrete methods; separate raw concatenation for named interface methods. |
| `optfacts.go:overlayGraph.prepare` and `addShellMethod` | Reconstruct exact method keys for qualifier routing, readonly paths, assertion metadata, and cloned method metadata. |
| `gocall.go:optParamSet` | Independent value/pointer method key for imported option parameters. |
| `facts/write_const.go:constMethodName` | Imported mutation lookup through the shared constructor. |
| `types2/const_logos.go:logosConstMethodObject` | Independent method key passed to `LogosImportedParamProvenZero`; `facts/checker.go` compares it verbatim. |
| `noder/logos_allocfacts.go:logosTypes2FactName` | Imported allocation fact key; the producer uses `lint.DeclName` with syntax receiver spelling. |
| `wasmyield/wasmyield.go:objectName` and `wasmyield/imports.go:importedYield` | Linker-symbol-derived keys for yield facts and their lookup. |
| `types2/const_receiver_logos.go:logosProofObject` | Same-shaped readonly proof handoff key; this is not an imported facts wire lookup. |

Existing codec tests pin ordinary `S.F` and `(*S).Get` method spellings and
the tagged package function `Float32sFromArch@simd0`. They did not pin a
tagged receiver method round trip. `IMPLEMENTATION.md` M3-A-S3 requires an
identifier tag but does not separately specify its placement on a method.

## Frozen controls

| Test | Required behavior |
| --- | --- |
| `TestTaggedMethodObjectConstructors` | Existing receiver-tag output is preserved for value/pointer receivers, aliases, and supported const wrappers; generated records round-trip with known-zero masks. `ObjectName` retains its existing empty result for const-wrapped receivers. |
| `TestTaggedMethodCodecGrammar` | Legacy spellings and the two emitted receiver-tag forms are accepted; malformed/repeated/double tags, local compiler symbols, and generic shapes remain rejected. |
| `TestTaggedInterfaceReadonlyFactWriter` | A tagged named interface's existing readonly method record round-trips without renaming. |
| `TestTaggedMethodFactLookup` | Decoded method facts reach the exact canonical value/pointer method without replacing declarations or resolving lazy package method symbols. |
| `TestTaggedMethodFactRejectsInvalidMask` | A canonical method lookup rejects an out-of-arity mask without publishing any metadata. |
| `TestTaggedInterfaceMethodFactLookup` | The readonly record reaches a tagged interface method while its lazy package-scope method symbol remains unresolved. This is the named old-package-guard negative control. |
| `TestTaggedMethodStructuralFactLookup` | Canonical method facts survive structural cloning; original signatures and metadata remain unchanged. |
| `TestTaggedMethodOptionFactLookup` | Imported option parameters match the canonical method only, excluding an unrelated method's parameter. |
| `TestTaggedMethodIdentityRemainsDistinct` | Receiver-local and method-local tags retain distinct keys, objects, and metadata. |
| `TestCompileSIMDTaggedMethodFacts` | The unchanged real SIMD source passes the fork-Go baseline, emits twelve exact known-zero method facts under `logos`, decodes, and imports successfully. |

`frozen.json` records exact file hashes, the base commit, the unmodified SIMD
source hash, and two named negative controls. `old-facts.go.txt` and
`old-optfacts.go.txt` are exact production files from the base commit.
Restoring only the old codec must make the
`TestTaggedMethodCodecGrammar/Box@simd128.Read` control fail. After the codec
extension, restoring only the old package guard must make
`TestTaggedInterfaceMethodFactLookup` fail by resolving the lazy
`Reader@simd128.Read` symbol. The earlier package-tag predicate repair is
already present in that archived overlay file, so this negative isolates the
new need to exclude method separators before lookup.

## Separate unexecuted follow-ups

The inventory found two pre-existing behaviors outside this slice. First,
`ObjectName` does not unwrap `ConstView`, while `MethodObject` supports its
outer receiver wrapper. This slice preserves those contracts. Second,
`noder.logosTypes2FactName` uses `types2.TypeString(receiver, nil)`, whose nil
qualifier includes the package path, while the allocation-fact producer uses
`lint.DeclName`'s local receiver syntax. Imported ordinary methods therefore
deserve a separate focused allocation-fact reproduction. Neither observation
was executed or repaired here; neither requires normalizing tagged identities.

## Pending execution

After a fresh toolchain build that includes the eventual production repair,
the focused gates are:

```sh
go test -json -count=1 -p=1 -parallel=1 cmd/compile/internal/logos/facts -run '^(TestTaggedMethodObjectConstructors|TestTaggedMethodCodecGrammar|TestTaggedInterfaceReadonlyFactWriter)$'
go test -json -count=1 -p=1 -parallel=1 cmd/compile/internal/logos -run '^(TestTaggedMethodFactLookup|TestTaggedMethodFactRejectsInvalidMask|TestTaggedInterfaceMethodFactLookup|TestTaggedMethodStructuralFactLookup|TestTaggedMethodOptionFactLookup|TestTaggedMethodIdentityRemainsDistinct|TestCompileSIMDTaggedMethodFacts)$'
```

The integration command must not use `-short` as a substitute for its named
PASS events. Missing tests, skips, compiler unavailability, and failed commands
are not acceptance. Preserve pre-repair failures separately from the final
positive and old-overlay negative evidence.
