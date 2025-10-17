## Tools {#tools}

### Go command {#go-command}

<!-- go.dev/issue/74667 -->
`cmd/doc`, and `go tool doc` have been deleted. `go doc` can be used as
a replacement for `go tool doc`: it takes the same flags and arguments and
has the same behavior.

<!-- go.dev/issue/75432 -->
The `go fix` command, following the pattern of `go vet` in Go 1.10,
now uses the Go analysis framework (`golang.org/x/tools/go/analysis`).
This means the same analyzers that provide diagnostics in `go vet`
can be used to suggest and apply fixes in `go fix`.
The `go fix` command's historical fixers, all of which were obsolete,
have been removed and replaced by a suite of new analyzers that
offer fixes to use newer features of the language and library.
<!-- I'll write a blog post that discusses this at length. --adonovan -->

### Cgo {#cgo}

