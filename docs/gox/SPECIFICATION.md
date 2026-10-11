# GoX language and transformation specification

Status: experimental implementation on `feature/gox-spec-plan`. The standalone
engine, explicit syntax, bare JSX, headless runtime and native package-loader
adapter are implemented. This remains the target contract; unchecked release
gates and deviations are tracked in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
See [the runnable guide](../../gox/README.md) for the implemented API and commands.

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
4. Structured `X"..."` JSX-like expressions, component declarations, and the
   minimal persistent-instance/hook runtime needed to execute them.

Milestone 2 adds bare JSX as alternative syntax using the same representation,
lowering and runtime. It does not introduce a second component implementation.

Transformation happens at build time; generated expressions execute at runtime.
The scope now includes a minimal component runtime: instance storage, state,
refs, context, reconciliation and scheduled rerendering. A headless host verifies
its behavior; production DOM/GPU/native renderers remain separate. Arbitrary
compile-time execution, a macro interpreter and a new type system remain out
of scope. Formatting itself is eager; repeated evaluation belongs to components.

## 2. Source compatibility and syntax overview

`.gox` files opt into extensions. Ordinary `.go` files retain stock semantics.
In GoX mode, uppercase `F` and `X` introduce literals only when immediately
adjacent to their opening delimiter. `F` supports quotes and backticks; the
initial `X` form uses double quotes. `do` and `component` are contextual
introducers. Ordinary identifiers such as `F`, `X`, `do()` and `component.Field`
remain legal. The earlier lowercase s prefix and view introducer are superseded,
not compatibility aliases. Malformed recognized extensions produce diagnostics.

```gox
package example

func describe(name string) string {
    func suffix(n int) string {
        return F"{name}: {n}"
    }
    return F"Result: {do string {
        if name == "" { return "anonymouF" }
        return suffix(2)
    }}"
}
```

Proposed grammar additions, where Type, Expression and Block mean Go grammar
extended recursively with these productions:

```ebnf
StringExpr     = "F" ( QuotedSE | RawSE ) .
Interpolation  = "{" Expression "}" .
LocalFunction  = "func" identifier Signature Block .
BlockExpr      = "do" Type Block .
XExpr          = "X" QuotedMarkup .
BareJSX        = Element .  (* milestone 2; expression-start position only *)
ComponentDecl  = "component" identifier "(" [ Parameters ] ")" Block .
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

Both `F"text {expr}"` and raw, multiline ``F`text {expr}``` are supported.
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

`text := F"{name}"` stores an already computed string. To defer evaluation:

```gox
render := func() string { return F"{name}" }
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
    if user.Admin { return F"Administrator {user.Name}" }
    return F"User {user.Name}"
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

## 6. Structured X markup and component runtime

### 6.1 Explicit markup literals, then bare JSX

Milestone 1 accepts a structured literal containing exactly one root element:

```gox
node := X"<panel title={F"Hello {name}"}>
    <label>{name}</label>
    <ui.Button caption="Save" />
</panel>"
```

X is not an ordinary Go string scanned and unescaped as a whole. The opening
quote enters markup mode. Attribute quotes are owned by the attribute parser;
quotes inside `{...}` are owned by the Go expression parser. After the root
element closes, optional whitespace and the outer closing quote end X. Multiline
markup is allowed. Quotes in child text are literal; text after the root other
than whitespace is an error. Quoted attributes use Go string escapes; child
text does not. Preserve text whitespace, normalizing CRLF only. Use `{"{"}` or
`{"<"}` for structural characters. Raw/backtick X delimiters are deferred.

Nested child elements need no prefix. Interpolation expressions can contain
F, X, typed blocks and ordinary Go closures recursively. An unterminated
attribute, interpolation, root or wrapper gets a distinct source diagnostic.
Literal markup is parsed at build time; values never become source text.

Milestone 2 also accepts:

```gox
node := <panel title={F"Hello {name}"}><label>{name}</label></panel>
```

The parser enters markup mode at `<` only when a primary expression is expected.
After an operand, `<` remains comparison; `<-` remains channel receive. Use the
same element records and lowering for both entry forms. Test `a < b`, `a <= b`,
`<-ch`, generic brackets, return/assignment/call contexts, comments and newlines.
The parser must coordinate scanner mode entry before lookahead consumes markup.
Markup errors cannot silently fall back to comparison parsing. Bare JSX is
recognized only in GoX files, not by changing the ordinary Go grammar globally.

### 6.2 Nodes and parameterized template lowering

A separately versioned `goxview` package owns node descriptions, component
definitions and runtime instances. Lowercase single-identifier tags construct
host element descriptions. Uppercase or qualified tags refer to declared
component definitions. A component element describes an invocation; it does
not directly execute the component while building its parent's node tree.
The runtime resolves its instance and invokes it during reconciliation.

X lowering builds a generated function/template with parameter slots for its
interpolations and calls it with values evaluated in the surrounding scope.
Each hole produces exactly one value, evaluated once, depth-first in source
order (attributes before children). Resolve each component definition at its
tag before that tag's attributes. Use sequential temporaries to preserve reads
relative to side effects. Construct the final template after evaluating holes;
component bodies execute later in the runtime, not as interpolation side effects.

Do not stringify values into Go/markup source. For a concrete baseline ABI,
template slots may use `any`; generated single-target typed initializations
(`var slot any = expr`) accept nil and reject multi-result expressions. Known
component properties are checked against declared parameter types in generated
Go code before boxing. Generated render adapters recover those exact types.
Function callbacks retain their concrete type inside host attribute values.
This avoids assuming Go can infer parameter types on generated function literals.
A later optimization may specialize slots if it preserves behavior.

Attribute names are case-sensitive identifiers. Duplicates are errors. Component
attributes map by name to declared parameters: required arguments must appear,
unknown arguments are errors, key is prohibited as a declared input name, and
variadic component parameters are excluded in
v1. A reserved `children` parameter of type `[]goxview.Node` receives child
nodes, including an empty slice when absent; explicit children= is rejected.
Supplying children to a component without that parameter is an error. Native
host attributes remain named values validated by the host adapter.

Child expressions support Node, string, nil and []Node; strings become text,
nil is omitted and slices flatten one level in order. Other values produce a
documented runtime type error where their static type was erased. F explicitly
formats numbers and arbitrary values. A minimal Node representation and public
constructors must be documented; no implicit HTML entity decoding, spreading,
fragments or conditional directives are included. Loops can build []Node and
insert them as a child expression.

### 6.3 Component declaration and generated instance parameter

The component declaration syntax remains a concrete proposal, while the
persistent-instance model and implicit instance injection are agreed design:

```gox
component Counter(title string) {
    count, setCount := useState(0)
    ref := useRef(0)
    theme := useContext(Theme)
    _ = ref
    _ = theme
    return X"<button onClick={func() {
        setCount(count + 1)
    }}>{F"{title}: {count}"}</button>"
}
```

A component returns one Node implicitly in its generated signature. Lower the
body to an ordinary Go function receiving `instance *goxview.Instance` followed
by typed inputs. Rewrite recognized hook forms using that instance and a
compiler-generated hook slot. Illustrative, not final exported API spelling:

```go
func renderCounter(instance *goxview.Instance, title string) goxview.Node {
    count, setCount := goxview.UseState[int](instance, countSlot, 0)
    // Remaining body lowered in place.
}
```

Do not copy outer variable declarations into this body. Initializers and local
bindings already inside it execute again on each render; free variables retain
ordinary lexical capture. Local component declarations lower to local definition
values with stable compiler-assigned definition IDs and capturing render
callbacks. The runtime refreshes current inputs/callbacks on parent updates;
closure pointer addresses must not define component identity. Package-level
components have stable definition IDs as well. No local component generics or
receiver declarations are introduced in v1.

A registry is owned by a runtime/root, never one process-global mutable singleton.
Each live instance holds definition ID, parent identity, current inputs/render
callback, hook storage, context subscriptions and committed child instances.
A function-valued field is sufficient; dynamically attaching Go methods is not
required. Registry entries are not newly allocated VNode descriptions.

### 6.4 Identity, reconciliation and loops

Instance matching uses parent instance, component definition, and either an
explicit sibling key or the structural child position. Structural position
includes intervening host-node/container paths so distinct child lists cannot
collide merely because they share a nearest component ancestor. `key={...}` is runtime
metadata, not an ordinary component input. v1 keys are strings or integers,
normalized with their type category; duplicate sibling keys are errors. Keys
are scoped to siblings. Changing a key or component definition unmounts the old
instance and mounts a new one. Stable keyed siblings retain state across reorder;
unkeyed children use position and can reset/move state as positions change.

One source location can create many instances in a loop. A compiler source-site
ID cannot identify them alone. The compiler identifies definitions/hook slots;
the runtime identifies actual occurrences. Unmount recursively removes registry
entries and context subscriptions and invalidates stale setters. Updates to an
unmounted instance are ignored, optionally reported in debug mode, and must never
update a new instance that reuses a key. Use a generation/lifetime token.

### 6.5 Hook semantics and restrictions

| Form | Persistent association | Update behavior |
| --- | --- | --- |
| useState(initial) | Instance + compiler slot stores T | Setter queues T replacement and rerender |
| useRef(initial) | Instance + compiler slot stores stable Ref[T] | Mutation alone does not schedule render |
| useContext(context) | Nearest ancestor provider for typed context identity | Subscribe instance; provider updates schedule consumers |

Initial state/ref storage is installed once per mount, even though an eager
initializer expression is evaluated each time execution reaches the call.
Use explicit generic type arguments when Go inference cannot determine T, such
as untyped nil. Repeated queued state replacements apply in order; even equal
values may schedule a render in v1 (no generic equality assumption). A separate
functional updater operation on the setter, `setCount.Update(func(T) T)`, applies to the latest stored
state and avoids stale callback snapshots. Callback locals such as count are
snapshots of the render that created the callback; the setter remains attached
to the instance and slot. Implement the callable setter as a named function
type with an Update method, or settle an equivalent explicit API before coding;
plain Go function values cannot acquire arbitrary instance methods. Reconciliation replaces host callbacks on commit.

Typed context objects expose a default and a provider constructor accepting a
value and child descriptions. The provider is resolved through committed/render
ancestry, not a global variable. Nested providers shadow outer providers of the
same context identity. Missing providers yield the default. Provider updates
invalidate subscribed descendants; subscriptions refresh per successful render.

Recognize hooks by resolved special forms in component bodies, not blind textual
replacement of every similarly named function. A user binding with the same
name follows ordinary scope resolution. v1 hook calls must occur as direct,
unconditional top-level component statements before conditional returns. Reject
hook calls in loops, branches, nested closures, block expressions, callbacks or
ordinary functions. Custom hook functions and compiler-propagated instance
parameters through arbitrary calls are deferred. Slot IDs are deterministic
within a component build; state migration across live code edits is not promised.

### 6.6 Scheduling, host boundary and lifetime

The runtime serializes render/reconcile/commit work per root. State setters enqueue
updates and mark instances dirty; they do not recursively invoke render. A Flush
operation drains queued work for deterministic tests; a pluggable scheduler wakes
an application event loop. Workers use `StateSetter.Enqueue` or `EnqueueUpdate` to queue safely even
during rendering. Ordinary callable setters and `Update` reject render-phase
calls. This explicit distinction avoids depending on unsupported goroutine-ID
introspection. Render
bodies and host commit calls run on the root owner. Ref writes require the same
owner discipline or caller synchronization; they are not automatically atomic.

Reconcile fresh descriptions against persistent instances, render child components,
then commit the resulting host tree. No host mutation during render. Render-phase
state updates are rejected in v1 to prevent immediate feedback loops. A render
panic aborts that pending commit and restores the root's rendering bookkeeping;
the previous host tree remains committed. Runtime error delivery must be explicit.
Mount/unmount bookkeeping releases subscriptions and callbacks; arbitrary effects,
useEffect, cleanup hooks and asynchronous rendering are deferred.

Provide a headless host adapter with observable node trees and event dispatch
for acceptance tests. Production DOM/native/GPU renderers, React compatibility,
HTML serialization, memoization and automatic dependency tracking are not promised.
Runtime module versions are pinned; generated imports do not authorize implicit
go.mod edits or dependency downloads. The transformer itself remains independent
of a live rendering runtime and must not run component code during compilation.

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
interpolation code, X wrapper, markup tag and markup text. Milestone 2 adds
parser-directed bare-markup entry into the same modes. Each frame tracks its opening
position and delimiter depth. Nested comments/strings follow Go's own lexical
rules (Go block comments do not themselves nest). Use a private token-kind
wrapper for extension tokens; do not modify standard token constants globally.

Semicolon insertion must match Go in ordinary code and block bodies. An SE or
X/bare JSX end behaves as an expression-ending token. At a depth-zero interpolation
closing delimiter, allow the embedding parser to consume its optional final
inserted semicolon; do not disable Go semicolon rules inside arbitrary nested
code. Include newline, comment, EOF and return-expression cases in tests.

The extended parser accepts all constructs before lowering. Public go/ast
interfaces have private marker methods, so external packages cannot implement
new expression nodes directly. Use ordinary placeholder nodes registered by
identity in an explicit extension-record table. Records contain kind, child
ASTs/records, source spans and raw text. Placeholder bindings are private
implementation data, never parsed magic identifiers. A recursive lowering pass
replaces every placeholder, local-function and component record with valid Go
nodes, passing explicit instance context only to component/hook lowering.
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
TransformPackage(ctx context.Context, files map[string][]byte, opts Options) (PackageResult, error)
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
the selected Go toolchain. Build/run/test create immutable generated files and a temporary Go filesystem
overlay, then invoke the configured Go command without a shell. This preserves
the existing module/workspace tree instead of copying it. Preserve module
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
and executable tests of capture, mutation, evaluation and formatting order,
component identity, hooks, keyed reconciliation and rerender scheduling. Bare
JSX and X forms must have equivalent lowering/runtime behavior in milestone 2.
See the plan for the complete feature matrix and milestone gates.

Use stock Go as the semantic oracle after lowering, not hand-written expectations
of optimizer behavior. No zero-allocation or performance-equivalence promise
is made; measure cold and warm transformation/build times before optimizing.
Do not rewrite closures based on presumed escape behavior.

v1 exclusions: implicit lazy SEs, arbitrary compile-time execution, whole-block
function hoisting, inferred block types, local generics, production host renderers,
automatic dependency downloads, and modifying compiler/backend semantics.
These exclusions do not defer the four syntax families required in milestone 1.
