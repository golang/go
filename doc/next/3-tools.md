## Tools {#tools}

### Telemetry

<!-- go.dev/issue/58894, go.dev/issue/67111 -->
Starting in Go 1.23, the Go toolchain can collect usage and breakage
statistics that help the Go team understand how the Go toolchain is
used and how well it is working. We refer to these statistics as
[Go telemetry](/doc/telemetry).

Go telemetry is an _opt-in system_, controlled by the
[`go` `telemetry` command](/cmd/go/#hdr-Manage_telemetry_data_and_settings).
By default, the toolchain programs
collect statistics in counter files that can be inspected locally
but are otherwise unused (`go` `telemetry` `local`).

To help us keep Go working well and understand Go usage,
please consider opting in to Go telemetry by running
`go` `telemetry` `on`.
In that mode,
anonymous counter reports are uploaded to
[telemetry.go.dev](https://telemetry.go.dev) weekly,
where they are aggregated into graphs and also made
available for download by any Go contributors or users
wanting to analyze the data.
See “[Go Telemetry](/doc/telemetry)” for more details
about the Go Telemetry system.

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
The `go` `list` `-m` `-json` command now includes new `Sum` and `GoModSum` fields.
This is similar to the existing behavior of the `go` `mod` `download` `-json` command.

<!-- go.dev/issue/65573 ("cmd/go: separate default GODEBUGs from go language version") -->
The new `godebug` directive in `go.mod` and `go.work` declares a
[GODEBUG setting](/doc/godebug) to apply for the work module or workspace in use.

### Vet {#vet}

<!-- go.dev/issue/46136 -->
The `go vet` subcommand now includes the
[stdversion](https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/stdversion)
analyzer, which flags references to symbols that are too new for the version
of Go in effect in the referring file. (The effective version is determined
by the `go` directive in the file's enclosing `go.mod` file, and
by any [`//go:build` constraints](/cmd/go#hdr-Build_constraints)
in the file.)

For example, it will report a diagnostic for a reference to the
`reflect.TypeFor` function (introduced in go1.22) from a file in a
module whose go.mod file specifies `go 1.21`.

### Cgo {#cgo}

<!-- go.dev/issue/66456 -->
[cmd/cgo] supports the new `-ldflags` flag for passing flags to the C linker.
The `go` command uses it automatically, avoiding "argument list too long"
errors with a very large `CGO_LDFLAGS`.

### Trace {#trace}

<!-- go.dev/issue/65316 -->
The `trace` tool now better tolerates partially broken traces by attempting to
recover what trace data it can. This functionality is particularly helpful when
viewing a trace that was collected during a program crash, since the trace data
leading up to the crash will now [be recoverable](/issue/65319) under most
circumstances.
