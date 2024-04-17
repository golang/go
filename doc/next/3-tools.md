## Tools {#tools}

### Go command {#go-command}

Setting the `GOROOT_FINAL` environment variable no longer has an effect
([#62047](https://go.dev/issue/62047)).
Distributions that install the `go` command to a location other than
`$GOROOT/bin/go` should install a symlink instead of relocating
or copying the `go` binary.

New Flag -changed only prints all non-default settings in go env,
do not print GOOS and GOARCH.

### Vet {#vet}

The `go vet` subcommand now includes the
[stdversion](https://beta.pkg.go.dev/golang.org/x/tools/go/analysis/passes/stdversion)
analyzer, which flags references to symbols that are too new for the version
of Go in effect in the referring file. (The effective version is determined
by the `go` directive in the file's enclosing `go.mod` file, and
by any [`//go:build` constraints](https://pkg.go.dev/cmd/go#hdr-Build_constraints)
in the file.)

For example, it will report a diagnostic for a reference to the
`reflect.TypeFor` function (introduced in go1.22) from a file in a
module whose go.mod file specifies `go 1.21`.

### Cgo {#cgo}

