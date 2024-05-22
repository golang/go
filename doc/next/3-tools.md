## Tools {#tools}

### Go command {#go-command}

Setting the `GOROOT_FINAL` environment variable no longer has an effect
([#62047](/issue/62047)).
Distributions that install the `go` command to a location other than
`$GOROOT/bin/go` should install a symlink instead of relocating
or copying the `go` binary.

<!-- go.dev/issue/34208, CL 563137, CL 586095 -->
The new `go` `env` `-changed` flag causes the command to print only
those settings whose effective value differs from the default value
that would be obtained in an empty environment with no prior uses of the `-w` flag.

<!-- go.dev/issue/27005, CL 585401 -->
The new `go` `mod` `tidy` `-diff` flag causes the command not to modify
the files but instead print the necessary changes as a unified diff.
It exits with a non-zero code if updates are needed.

### Vet {#vet}

The `go vet` subcommand now includes the
[stdversion](https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/stdversion)
analyzer, which flags references to symbols that are too new for the version
of Go in effect in the referring file. (The effective version is determined
by the `go` directive in the file's enclosing `go.mod` file, and
by any [`//go:build` constraints](https://pkg.go.dev/cmd/go#hdr-Build_constraints)
in the file.)

For example, it will report a diagnostic for a reference to the
`reflect.TypeFor` function (introduced in go1.22) from a file in a
module whose go.mod file specifies `go 1.21`.

### Cgo {#cgo}

