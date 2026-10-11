# GoX

GoX translates extended Go source into ordinary Go. This module contains a
standalone command, an API, the adapted Go scanner/parser and a headless component
runtime. The fork also integrates this same engine into `go` package discovery.
Go's compiler backend and closure implementation are unchanged.

Requires Go 1.26 or newer. This is an experimental implementation; the detailed
release gates and outstanding work are in
[the implementation plan](../docs/gox/IMPLEMENTATION_PLAN.md).

## Try it

From this directory, using a stock Go installation:

```sh
go test ./...
go build -o gox ./cmd/gox
cd examples/counter
../../gox run .
```

On Windows, use `../../gox.exe`. The example has its own module and a local
replacement for the runtime, so no published GoX release is needed. Expected:

```text
Counter 42: 0 (render 1)
Counter 42: 1 (render 2)
Another: 0 (render 2)
```

Install from the checkout with `go install ./cmd/gox`. For an application outside
this checkout, add `github.com/alfu32/go-shared/gox` to its module requirements and
replace it with this checkout's `gox` directory until a version is published.
The transformer never edits module manifests automatically.

## Language

```go
name := "Ada"
message := F"Hello {name}; twice: {F"{name} {name}"}"
raw := F`Braces: {{literal}}, value: {name}`

func fact(n int) int {
    if n < 2 { return 1 }
    return n * fact(n-1)
}
answer := do int { defer cleanup(); return fact(5) }
node := X"<label title={message}>{F"answer: {answer}"}</label>"
other := <label>{message}</label>
```

* `F` is uppercase and adjacent to the quote or backtick. Holes contain Go
  expressions, including nested GoX expressions. `{{` and `}}` produce braces.
  Each hole is evaluated once and immediately formatted with `fmt.Sprint` before
  the next hole is evaluated. A literal without holes remains a Go constant.
* Named local functions retain lexical captures and self recursion. They become
  visible at their declaration, without forward declarations or implicit mutual
  recursion. Go's normal loop capture semantics apply.
* `do Type { ... }` is an immediately called function with the indicated result
  type. Its return/defer/break/goto boundaries are normal function boundaries.
* `X"<...>"` and bare markup have the same lowering. Bare markup starts only where
  an expression may start. Comparisons and channel receives keep Go's meanings.
  Attributes require `name="text"` or `name={expression}`. Literal markup text,
  including whitespace, is preserved. There is exactly one root element.
* Lowercase tags create host nodes; capitalized or qualified tags refer to
  components. Children accept strings, nodes, node slices and nil. Other child
  values panic rather than being implicitly formatted; use `F` explicitly.
* Markup holes are evaluated in source order into explicit parameters of a
  generated template function. Values and callbacks remain typed Go values;
  they are never pasted into source text.

```go
component Counter(title string) {
    n, set := useState(0)
    ref := useRef(0)
    theme := useContext(Theme)
    ref.Current++
    return <button theme={theme} onClick={func() {
        set.Update(func(latest int) int { return latest + 1 })
    }}>{F"{title}: {n}"}</button>
}
```

Component properties are mandatory, statically checked and passed by name.
Unknown/missing properties are rejected by generated Go types, including across
package boundaries. An optional `children []view.Node` parameter accepts nested
children; `key` is reserved occurrence metadata. Components may be declared in
functions and capture those functions' variables. Their definitions describe
render code; each mounted occurrence owns a separate persistent instance.

Hooks must be direct, unconditional assignments at the beginning of component
execution, before control-flow statements. Ordinary locally bound functions
named `useState`, `useRef` or `useContext` retain their meaning. Hook calls in
branches, loops, callbacks and after conditional control flow are rejected.

The generated component value exposes a `GoXDefinition` runtime definition,
per-property typed adapters and a structural validation function. This ABI is
experimental. Ordinary Go can consume emitted code and use
`view.Component(Counter.GoXDefinition, attrs, children)`; that low-level API
performs runtime property checks, whereas markup performs static checks.

## Runtime

`goxview.NewRoot(commit, wake)` creates a root. `Mount(node)` renders;
`Tree()` returns the headless tree; attribute callback values serve as event
handlers. Invoke a callback, then call `Flush()` to process its queued updates.
`Unmount()` invalidates old setters. The counter example demonstrates this host
adapter without depending on a browser or UI framework.

Identity is parent + component definition + explicit key or sibling position.
Keys are strings or integers and must be unique among siblings. Keyed reordering
preserves state; changing a key or definition resets it. Refs persist without
scheduling. `NewContext(default).Provider(value, children...)` supplies the nearest
context value. Every scheduled render currently traverses the whole root, so
context consumers update correctly without selective subscription optimization.

A single host goroutine owns `Mount`, `Flush`, `Tree`, `Unmount` and refs. Ordinary
`set(value)` and `set.Update(f)` reject updates during rendering. Workers use
`set.Enqueue(value)` or `set.EnqueueUpdate(f)`, which are safe even while a render
is in progress. The wake callback should signal the host event loop, not render
synchronously. Work queued during a render or commit waits for a later `Flush`.
Callbacks capture the render's state snapshot; functional updates see the latest
queued state. Render panics return errors and leave the previous host tree intact;
user side effects, including ref mutations, are not rolled back.

Effects, live-edit state migration, DOM/GPU rendering and arbitrary compile-time
execution are outside this runtime. Templates and blocks run at application
runtime, not while compiling.

## Commands and project handling

| Command | Action |
| --- | --- |
| `gox emit file.gox [--output file.go]` | Inspect formatted Go; atomic explicit output |
| `gox check ./...` | Go vet/type checking, including tests, without executing user tests |
| `gox build`, `run`, `test`, `list`, `vet` | Forward to the selected Go tool with a generated overlay |

Use `--` to separate GoX source arguments from program arguments. Set `GOX_GO` to
an alternative Go executable or a shared distribution's Python launcher; set
`GOX_PYTHON` when a specific Python executable is needed. No shell command strings
are used. Child process streams and exit status are preserved.

The standalone command asks Go for the module/workspace graph and transforms
selected `.gox` files in those modules, including local replacements. It uses
filesystem overlays, preserving assets, embeds, relative paths and manifests.
Files such as `widget_linux.gox` and `widget_test.gox` keep their selection rules.
A source `widget.gox` conflicts with an existing `widget.go` and is rejected.
Because standalone preparation scans selected modules, a syntax error in an
otherwise unused package can fail preparation. Native package discovery avoids
that eager whole-module transformation.

Cache entries include source bytes, transformation options, logical module/file
identity and a hash of the transformer executable. Writes are atomic. Diagnostic
`//line` directives point back to GoX. The API exposes line-level source mappings;
generated helper locations are approximate. There is no GoX-aware gopls/formatter
integration yet: inspect emitted Go for type errors involving generated adapters.

`TransformFile` operates on bytes. `TransformPackage` accepts selected files,
a context for cancellation, and returns outputs and imports. File selection and
filesystem orchestration live outside the parser. The parser/scanner derive from
the pinned Go sources documented in [UPSTREAM.md](UPSTREAM.md).

## Fork integration

After building this repository, normal `go build/run/test/list/vet` discover GoX
before import loading. `go gox emit file.gox` inspects the same lowering.
`go list -json` adds `GoXFiles` and `GoXGeneratedFiles` (virtual Go → original GoX).
The standard `GoFiles` fields contain virtual `.go` paths, while the filesystem
adapter resolves them to immutable generated files. Coverage also receives those
actual paths. The cmd module uses a vendored copy of this module; refresh it with:

```sh
python gox/tools/sync_vendor.py
```

Run this from the repository root. CI checks that the vendored copy stays in sync.
The shared C/Python launchers keep their existing entry points and argument flow.
