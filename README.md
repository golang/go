# Go Tools

[![PkgGoDev](https://pkg.go.dev/badge/golang.org/x/tools)](https://pkg.go.dev/golang.org/x/tools)

This repository provides the `golang.org/x/tools` module, comprising
various tools and packages mostly for static analysis of Go programs,
some of which are listed below.
Use the "Go reference" link above for more information about any package.

It also contains the
[`golang.org/x/tools/gopls`](https://pkg.go.dev/golang.org/x/tools/gopls)
module, whose root package is a language-server protocol (LSP) server for Go.
An LSP server analyses the source code of a project and
responds to requests from a wide range of editors such as VSCode and
Vim, allowing them to support IDE-like functionality.

<!-- List only packages of general interest below. -->

Selected commands:

- `cmd/goimports` formats a Go program like `go fmt` and additionally
  inserts import statements for any packages required by the file
  after it is edited.
- `cmd/callgraph` prints the call graph of a Go program.
- `cmd/digraph` is a utility for manipulating directed graphs in textual notation.
- `cmd/stringer` generates declarations (including a `String` method) for "enum" types.
- `cmd/toolstash` is a utility to simplify working with multiple versions of the Go toolchain.

These commands may be fetched with a command such as
```
go install golang.org/x/tools/cmd/goimports@latest
```

Selected packages:

- `go/ssa` provides a static single-assignment form (SSA) intermediate
  representation (IR) for Go programs, similar to a typical compiler,
  for use by analysis tools.

- `go/packages` provides a simple interface for loading, parsing, and
  type checking a complete Go program from source code.

- `go/analysis` provides a framework for modular static analysis of Go
  programs.

- `go/callgraph` provides call graphs of Go programs using a variety
  of algorithms with different trade-offs.

- `go/ast/inspector` provides an optimized means of traversing a Go
  parse tree for use in analysis tools.

- `go/cfg` provides a simple control-flow graph (CFG) for a Go function.

- `go/expect` reads Go source files used as test inputs and interprets
  special comments within them as queries or assertions for testing.

- `go/gcexportdata` and `go/gccgoexportdata` read and write the binary
  files containing type information used by the standard and `gccgo` compilers.

- `go/types/objectpath` provides a stable naming scheme for named
  entities ("objects") in the `go/types` API.

Numerous other packages provide more esoteric functionality.

<!-- Some that didn't make the cut: 

golang.org/x/tools/benchmark/parse
golang.org/x/tools/go/ast/astutil
golang.org/x/tools/go/types/typeutil
golang.org/x/tools/go/vcs
golang.org/x/tools/godoc
golang.org/x/tools/playground
golang.org/x/tools/present
golang.org/x/tools/refactor/importgraph
golang.org/x/tools/refactor/rename
golang.org/x/tools/refactor/satisfy
golang.org/x/tools/txtar

-->

## Contributing

This repository uses Gerrit for code changes.
To learn how to submit changes, see https://golang.org/doc/contribute.html.

The main issue tracker for the tools repository is located at
https://github.com/golang/go/issues. Prefix your issue with "x/tools/(your
subdir):" in the subject line, so it is easy to find.

### JavaScript and CSS Formatting

This repository uses [prettier](https://prettier.io/) to format JS and CSS files.

The version of `prettier` used is 1.18.2.

It is encouraged that all JS and CSS code be run through this before submitting
a change. However, it is not a strict requirement enforced by CI.
