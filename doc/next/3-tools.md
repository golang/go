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

<!-- go.dev/issue/52792, CL 562775 -->
<!-- TODO: Improve this if needed. -->
The `go` `list` `-m` `-json` command now includes new `Sum` and `GoModSum` fields.
This is similar to the existing behavior of the `go` `mod` `download` `-json` command.

<!-- go.dev/issue/67111 ("cmd/go: add go telemetry subcommand") -->
The new `go` `telemetry` command should be documented here,
as well as `GOTELEMETRY` and `GOTELEMETRYDIR` environment variables.
<!-- go.dev/issue/58894 ("all: add opt-in transparent telemetry to Go toolchain") -->
<!-- TODO: document Go 1.23 behavior (from https://go.dev/cl/559199, https://go.dev/cl/559519, https://go.dev/cl/559795, https://go.dev/cl/562715, https://go.dev/cl/562735, https://go.dev/cl/564555, https://go.dev/cl/570679, https://go.dev/cl/570736, https://go.dev/cl/582695, https://go.dev/cl/584276, https://go.dev/cl/585235, https://go.dev/cl/586138) -->

<!-- go.dev/issue/65573 ("cmd/go: separate default GODEBUGs from go language version") -->
<!-- TODO: document Go 1.23 behavior (from https://go.dev/cl/584218, https://go.dev/cl/584300, https://go.dev/cl/584475, https://go.dev/cl/584476) -->

### Vet {#vet}

<!-- go.dev/issue/46136 -->
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

<!-- go.dev/issue/66456 -->
[cmd/cgo] supports the new `-ldflags` flag for passing flags to the C linker.
The `go` command uses it automatically, avoiding "argument list too long"
errors with a very large `CGO_LDFLAGS`.
