## Tools {#tools}

### Go command {#go-command}

Setting the `GOROOT_FINAL` environment variable no longer has an effect
([#62047](https://go.dev/issue/62047)).
Distributions that install the `go` command to a location other than
`$GOROOT/bin/go` should install a symlink instead of relocating
or copying the `go` binary.

### Cgo {#cgo}

### Vet

The new `tests` analyzer reports common mistakes in declarations of
tests, fuzzers, benchmarks, and examples in test packages, such as
malformed names, wrong signatures, or examples that document
non-existent identifiers. Some of these mistakes may cause tests not
to run.

This analyzer is among the subset of analyzers that are run by `go test`.
