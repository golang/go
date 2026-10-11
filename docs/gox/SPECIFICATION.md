# GoX language and transformation specification

Status: proposed implementation contract; no GoX implementation is delivered by
this documentation change. Syntax choices below make the plan actionable and
are proposals, not claims that every detail was already approved.

Repository baseline: `alfu32/go-shared`, `feature/shared-toolchain`, commit
`64f1787400cb9507e355a13bf5f25c7158dabe4d` (2026-10-11 inspection).
Implementation sequence: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## 1. Objective and boundaries

Build a standalone GoX-to-Go source transformer in Go, reusing adapted copies
of Go's public scanner and parser. Later embed the same transformation engine
into this fork's Go CLI. Preserve the existing compiler, runtime, native builds,
shared-library tools, and Python/batch launcher behavior.

All four feature families belong to the first language milestone:

1. String expressions (SEs), including recursively nested interpolations.
2. Named local functions, lexical capture, and self-recursion.
3. Explicitly typed block expressions containing ordinary statements.
4. A minimal JSX-like element/component expression with nested expressions.

Transformation happens at build time; generated expressions execute at runtime.
This is not arbitrary compile-time execution, a macro interpreter, a reactive
UI runtime, or a new type system. Deferred evaluation uses explicit functions.
No automatic dependency tracking, rendering, hooks, or snapshot capture is
implied by the syntax.

## 2. Source compatibility and syntax overview

`.gox` files opt into extensions. Ordinary `.go` files retain stock semantics.
In GoX mode, `s` is a prefix only when immediately followed by a quote or
backtick; `do` and `view` are contextual introducers only in their forms below.
Existing uses such as `s`, `do()`, `view.Field`, and `view < limit` remain legal.
Ambiguous contextual parses must use bounded lookahead and report errors without
silently reinterpreting malformed extensions as unrelated Go expressions.

```gox
package example

func describe(name string) string {
    func suffix(n int) string {
        return s"{name}: {n}"
    }
    return s"Result: {do string {
        if name == "" { return "anonymous" }
        return suffix(2)
    }}"
}
```

Proposed grammar additions, where Type, Expression and Block mean Go grammar
extended recursively with these productions:

```ebnf
StringExpr     = "s" ( QuotedSE | RawSE ) .
Interpolation  = "{" Expression "}" .
LocalFunction  = "func" identifier Signature Block .
BlockExpr      = "do" Type Block .
ViewExpr       = "view" Element .
Element        = "<" Tag { Attribute } ( "/>" | ">" { Child } "</" Tag ">" ) .
Tag            = identifier { "." identifier } .
Attribute      = identifier "=" ( string_lit | "{" Expression "}" ) .
Child          = Element | "{" Expression "}" | Text .
```

Local functions have no receivers or type parameter lists in v1. Block
expressions have exactly one unnamed result type; tuple/void blocks and inferred
result types are deferred. New constructs are expressions wherever their
result type is valid, not implicitly compile-time constants.

## 3. String expressions

### 3.1 Delimiters and escaping

Both `s"text {expr}"` and raw, multiline ``s`text {expr}``` are supported.
Quoted text follows Go interpreted-string escapes. Raw text follows Go raw
string rules, including carriage-return handling. `{{` and `}}` encode literal
braces in text mode. A single opening brace begins an interpolation; a single
closing brace in text mode is an error. Backslash is not an additional brace
escape. An interpolation must contain exactly one value-producing expression;
empty and comma-separated interpolations are errors.

Quotes, rune literals, raw strings, comments, composite literals, function
literals, block expressions and nested SEs inside interpolation are scanned
as code. Their internal braces do not terminate the enclosing interpolation.
An ordinary unescaped newline in quoted text is invalid; newlines inside its
interpolation are legal subject to expression grammar. Raw SEs may contain
newlines in text. Comments inside interpolation retain normal Go behavior.

### 3.2 Evaluation and formatting

An SE immediately produces a string. Each interpolation is evaluated exactly
once, in lexical left-to-right order. Each value is formatted with Go `fmt`'s
`%v` semantics immediately after its expression is evaluated, before the next
interpolation expression is evaluated. Side effects in Formatter/Stringer
implementations therefore have a defined order. Formatting can panic as it can
in ordinary fmt usage; GoX introduces no exception handling.

Literal text is never a format string. Percent signs in it are ordinary text.
There is no implicit HTML/SQL/shell escaping. Format specifier syntax is not
part of v1: explicit calls such as `{fmt.Sprintf("%.2f", amount)}` are legal.

Illustrative lowering (actual identifiers are generated hygienically):

```go
func() string {
    var gxBuilder strings.Builder
    gxBuilder.WriteString("Hello ")
    fmt.Fprint(&gxBuilder, name)
    gxBuilder.WriteString(", ")
    fmt.Fprint(&gxBuilder, next())
    return gxBuilder.String()
}()
```

Separate Fprint calls matter: a single Sprintf with several arguments would
evaluate all arguments before formatting any of them. Do not introduce that
optimization without proving the specified behavior unchanged.

An SE without interpolations lowers to an ordinary string literal and can be a
constant. An SE with interpolations is not a constant, even if all operands are.
The normal type checker rejects use in const declarations and other constant
contexts. SEs with values not usable as a single fmt operand are rejected by
ordinary type checking. The lowerer must prevent a multi-result expression
from accidentally expanding into variadic formatting arguments.

### 3.3 Capture and loops

`text := s"{name}"` stores an already computed string. To defer evaluation:

```gox
render := func() string { return s"{name}" }
```

Capture is Go lexical capture, not a generated parameter snapshot. Functions
remain at their original block location. Go's language-version-dependent loop
rules apply: variables declared by a loop have per-iteration identity under
Go 1.22+ semantics, while externally declared variables reused with `=` are
shared. Pointers, maps and slices still refer to shared underlying data.

## 4. Named local functions

Within a statement block:

```gox
func factorial(n int) int {
    if n <= 1 { return 1 }
    return n * factorial(n - 1)
}
```

lowers in place to:

```go
var factorial func(n int) int
factorial = func(n int) int {
    if n <= 1 { return 1 }
    return n * factorial(n - 1)
}
```

The name is visible in its body and subsequent statements in the same block,
not before the declaration. Same-block duplicate declarations are errors;
shadowing an outer binding follows Go rules. Names are function-valued local
variables: assignment to them is allowed. Their enclosing locals are captured
by the native Go compiler. No function lifting, closure environment generation,
or explicit free-variable parameter lists are needed.

Mutual recursion remains available through explicit Go variable declarations
followed by function literal assignments. Automatic forward declarations and
whole-block hoisting are not v1 semantics. This avoids calling a nil function
before its generated initializer executes. Local generic functions, methods,
and declaration-only local function signatures are rejected with clear errors.
Lowering must preserve labels and avoid accepting gotos that would otherwise
jump into a local's scope.

## 5. Typed block expressions

```gox
label := do string {
    if user.Admin { return s"Administrator {user.Name}" }
    return s"User {user.Name}"
}
```

lowers to an immediately invoked `func() string` literal. Type syntax includes
qualified types, pointers, slices, maps, channels, structs and instantiated
generic types supported by the baseline Go version. Parsing must distinguish
braces inside the type from the following statement block.

The body supports ordinary Go statements and all GoX constructs recursively.
Return completeness and result assignability are checked by Go. `return`
returns from this block function; `defer` runs at its exit. Break/continue/goto
cannot target constructs outside this function. Nested loops and labels work
normally. Panic/recover have ordinary function-call semantics. Free variables
are captured, and mutation of them is observable after evaluation.

This is not textual macro substitution. An SE or block expression in a `go` or
`defer` call retains the Go distinction between evaluating call arguments now
and invoking the deferred/launched function later.

## 6. Minimal JSX-like expressions

This section resolves the previously discussed alternative into a concrete,
minimal proposal, so milestone 1 does not silently omit it. The `view` prefix
disambiguates markup from Go comparisons and type arguments.

```gox
node := view <panel title={s"Hello {name}"}>
    <label>{name}</label>
    <ui.Button caption="Save" />
</panel>
```

A small, separately versioned `goxview` Go package defines the default contract:

```go
type Node interface { /* closed set of runtime node implementations */ }
type Attr struct { Name string; Value any }
func Element(tag string, attrs []Attr, children []Node) Node
func Text(text string) Node
func Child(value any) Node
```

The exact closed-interface method is a runtime implementation detail; document
its public constructors. Lowercase single-identifier tags call Element with a
literal tag name. Uppercase or qualified tags denote Go functions with the
signature `func([]goxview.Attr, []goxview.Node) goxview.Node`; invoke them in the
original lexical scope, so local capturing component functions work.

Attribute names are identifier spellings, case-sensitive. Duplicate attributes
are errors. Attribute strings use Go quoted/raw string syntax; attribute
expressions each produce one value. Source order is preserved with sequential
temporaries: evaluate the component function value first, then attributes,
then children left-to-right, then invoke the constructor/component. A child
expression accepts Node, string, or nil; Child converts string to Text and nil
to an empty child, and panics with a documented type error for other values.
Slice spreading, conditional directives, fragments, boolean shorthand, HTML
entities and implicit list flattening are deferred. Conditions and iteration
can be expressed in ordinary Go functions or typed blocks returning Node.

Text between tags is literal UTF-8 with normalized CRLF; preserve all other
whitespace, including indentation. A `{...}` child is code, and nested tags
are elements. Use `{"{"}`, `{"<"}` etc. for special text characters; do not
interpret markup as HTML. Matching close tags are mandatory. A trailing
slash closes a leaf element. Markup inside an interpolation starts with `view`;
nested child elements do not require it.

The default runtime stores a tree only. No renderer, DOM, event loop, React
compatibility, effect scheduler, or HTML serialization is included. Runtime
imports must be explicit in dependency metadata; generated imports alone do
not authorize automatic go.mod edits or network installs. Use a pinned module
version for standalone consumers and a reproducible vendored dependency for
toolchain distribution if the runtime is bundled.

## 7. Scanner/parser implementation architecture

Initial standalone module location: `gox/` at repository root, outside the
special `std` and `cmd` modules under src. Proposed module path:
`github.com/alfu32/go-shared/gox`. Final distribution/versioning is established
in milestone 2; implementation must not assume root Go module semantics.

Adapt copies of `src/go/scanner` and `src/go/parser` with tests and licenses.
Record the baseline SHA and all import adaptations in an UPSTREAM document.
Audit API requirements: this checkout's modules say Go 1.28 and its parser
calls scanner.End; do not claim compatibility with older Go releases without
checking and testing every used API. Avoid importing restricted compiler
internal packages. Reuse go/token positions and ordinary go/ast nodes.

Scanner modes form a stack: normal Go, quoted SE text, raw SE text,
interpolation code, markup tag and markup text. Each frame tracks its opening
position and delimiter depth. Nested comments/strings follow Go's own lexical
rules (Go block comments do not themselves nest). Use a private token-kind
wrapper for extension tokens; do not modify standard token constants globally.

Semicolon insertion must match Go in ordinary code and block bodies. An SE or
view end behaves as an expression-ending token. At a depth-zero interpolation
closing delimiter, allow the embedding parser to consume its optional final
inserted semicolon; do not disable Go semicolon rules inside arbitrary nested
code. Include newline, comment, EOF and return-expression cases in tests.

The extended parser accepts all constructs before lowering. Public go/ast
interfaces have private marker methods, so external packages cannot implement
new expression nodes directly. Use ordinary placeholder nodes registered by
identity in an explicit extension-record table. Records contain kind, child
ASTs/records, source spans and raw text. Placeholder bindings are private
implementation data, never parsed magic identifiers. A recursive lowering pass
replaces every placeholder and local-function record with valid Go nodes.
Assert the extension table is fully consumed before printing/type checking.

This modest intermediate representation supports recursive constructs and
source-aware diagnostics without duplicating the entire Go AST. A future
GoX-preserving formatter may need a concrete syntax tree; go/format formats
the generated Go only and is not a GoX formatter.

## 8. Hygiene, evaluation and diagnostics

Before naming generated identifiers, collect all identifier spellings from
the original file and extension records. Choose deterministic unused names
for temporaries, imports and helper functions. Never rely solely on an unusual
prefix. In package processing also exclude package-level identifiers from all
selected sibling .go/.gox files. Generated names are stable across runs.

Imports require binding awareness. If an existing fmt/strings/runtime import
is shadowed at an expansion site, relocate its reference into a hygienically
named file-level helper with explicit arguments; do not capture or change the
meaning of user identifiers. Reuse compatible imports and avoid duplicate
paths. Dot imports of generated dependency paths receive an explicit v1
diagnostic unless safe binding resolution is implemented. Blank imports may
be converted to a fresh named import only with preserved side-effect semantics.
Expression-only transform APIs return required imports/helpers, not just text.

Generated sequential declarations enforce specified evaluation order, including
ordinary variable reads relative to calls. Enforce exactly one value per
interpolation/attribute/child using a single-target initialization or equivalent
typed helper contract before a variadic formatting call. Untyped nil must be
accepted as a fmt value; avoid inferring a standalone variable type from nil.

Diagnostics have stable code, severity, message, primary source span and
optional related spans. Report unmatched delimiters at both open and failure
locations. Recovery always advances or exits; bound nesting and output growth.
Return multiple useful errors without producing a supposedly valid build on
failure. Keep generated wrapper locations distinct from original expression
locations. Source maps use byte offsets and include generated-to-original ranges;
line/column presentation must handle UTF-8 consistently. Emit //line directives
where useful, but retain precise sidecar mappings because formatting changes
layout and directives alone cannot represent every generated token.

## 9. Standalone API and CLI

Proposed API responsibilities (names may settle during implementation):

```go
TransformFile(filename string, src []byte, opts Options) (Result, error)
TransformPackage(inputs []Source, opts Options) (PackageResult, error)
```

Options contain language version, build context, runtime import path, limits
and deterministic emission settings. Results contain generated source,
diagnostics, dependency metadata, source maps and transformer identity.
Calls are reentrant and safe for parallel independent packages; no process-wide
mutable parser, working directory, environment, or exit behavior.

Commands planned for milestone 2:

```sh
gox emit example.gox --output example_gen.go
gox check ./examples/basic
gox build ./examples/basic
gox run ./examples/basic -- program-argument
gox test ./...
```

Emit is syntax/lowering validation; check includes package type checking using
the selected Go toolchain. Build/run/test create isolated generated package
trees and invoke the configured Go command without a shell. Preserve module
identity, relative replace directives, embeds and package layout. Do not
promise transparent arbitrary dependency discovery in the first wrapper:
support and document local module/workspace scope in M2, then dependency graph
integration in M3. Never overwrite user files without an explicit output path.
Exit 0 means success, 1 source/build failure, 2 invalid CLI usage; build/run/test
propagate subprocess failure status where available, including program status.

## 10. Native Go CLI integration

Milestone 3 invokes the library in-process. No mandatory external gox executable
or custom native launcher is added. Package loading must recognize .gox-only
packages, selected platform suffixes, build tags, internal/external _test.gox
files, workspace replacements and GoX dependencies before final import graph
construction. Scan headers before parsing inactive files. Generated imports
participate in graph construction and cycle detection.

Candidate integration surfaces verified at the baseline:

| Path | Responsibility |
| --- | --- |
| src/go/build/build.go and read.go | File classification, headers, constraints/import scanning |
| src/cmd/go/internal/modindex | Indexed discovery; avoid divergence from uncached loading |
| src/cmd/go/internal/load | Package metadata and dependency graph |
| src/cmd/go/internal/work/exec.go | Build inputs, instrumentation and generated source |
| src/cmd/go/internal/cache | Transformer/input identity and generated artifacts |
| src/cmd/go/testdata/script | CLI integration/regression tests |

This is an investigation map, not authorization to alter all public go/build
APIs. Prefer a GoX-aware internal discovery layer; change common loaders only
where required to keep indexed/nonindexed behavior consistent. Transform before
cgo, coverage and compilation; preserve directives and map error locations.

Cache identity includes exact input bytes, selected file set, language version,
transformer version, target/build settings and runtime/lowering configuration.
Writes are atomic, concurrent builders cannot share partially generated files,
and changing ignored files does not incorrectly change selected package inputs.
Go's normal cache still accounts for downstream dependencies. Handle editor
overlays explicitly; do not assume stock overlays discover nonexistent .gox
packages. Generated paths must not leak machine-specific roots into reproducible
artifacts. list JSON exposes original GoX inputs and generated compile inputs
through documented metadata. go vet consumes generated Go with mapped diagnostics.

The reusable module will be vendored/pinned into the cmd toolchain module via
its normal dependency workflow; no duplicate hand-maintained transformation
engine. Verify bootstrap compatibility before selecting that dependency version.
Stock gofmt/gopls do not automatically gain GoX support; editor integration and
a source-preserving formatter are subsequent projects, not implicit deliverables.

## 11. Validation contract and compatibility limits

Acceptance requires ordinary-Go scanner/parser differential tests, recursive
mixed-construct tests, deterministic output, successful standard Go compilation,
and executable tests of capture, mutation, evaluation and formatting order.
See the plan for the complete feature matrix and milestone gates.

Use stock Go as the semantic oracle after lowering, not hand-written expectations
of optimizer behavior. No zero-allocation or performance-equivalence promise
is made; measure cold and warm transformation/build times before optimizing.
Do not rewrite closures based on presumed escape behavior.

v1 exclusions: implicit lazy SEs, arbitrary compile-time execution, whole-block
function hoisting, inferred block types, local generics, markup rendering,
automatic dependency downloads, and modifying compiler/backend semantics.
These exclusions do not defer the four syntax families required in milestone 1.
