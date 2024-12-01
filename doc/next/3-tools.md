## Tools {#tools}

### Go command {#go-command}

<!-- go.dev/issue/48429 -->

Go modules can now track executable dependencies using `tool` directives in
go.mod. This removes the need for the previous workaround of adding tools as
blank imports to a file conventionally named "tools.go". The `go tool`
command can now run these tools in addition to tools shipped with the Go
distribution. For more information see [the
documentation](/doc/modules/managing-dependencies#tools).

The new `-tool` flag for `go get` causes a tool directive to be added to the
current module for named packages in addition to adding require directives.

The new [`tool` meta-pattern](/cmd/go#hdr-Package_lists_and_patterns) refers to
all tools in the current module. This can be used to upgrade them all with `go
get -u tool` or to install them into your GOBIN directory with `go install
tool`.

<!-- go.dev/issue/69290 -->

Executables created by `go run` and the new behavior for `go tool` are now
cached in the Go build cache. This makes repeated executions faster at the
expense of making the cache larger. See [#69290](/issue/69290).

<!-- go.dev/issue/62067 -->

The `go build` and `go install` commands now accept a `-json` flag that reports
build output and failures as structured JSON output on standard output.
For details of the reporting format, see `go help buildjson`.

Furthermore, `go test -json` now reports build output and failures in JSON,
interleaved with test result JSON.
These are distinguished by new `Action` types, but if they cause problems in
a test integration system, you can revert to the text build output by setting
`GODEBUG=gotestjsonbuildtext=1`.

### Cgo {#cgo}

<!-- go.dev/issue/56378, CL 579955 -->
Cgo supports new annotations for C functions to improve run time
performance.
`#cgo noescape cFunctionName` tells the compiler that memory passed to
the C function `cFunctionname` does not escape.
`#cgo nocallback cFunctionName` tells the compiler that the C function
`cFunctionName` does not call back to any Go functions.
For more information, see [the cgo documentation](/pkg/cmd/cgo#hdr-Optimizing_calls_of_C_code).

<!-- go.dev/issue/67699 -->
Cgo currently refuses to compile calls to a C function which has multiple
incompatible declarations. For instance, if `f` is declared as both `void f(int)`
and `void f(double)`, cgo will report an error instead of possibly generating an
incorrect call sequence for `f(0)`. New in this release is a better detector for
this error condition when the incompatible declarations appear in different
files. See [#67699](/issue/67699).

### Vet

<!-- go.dev/issue/44251 -->
The new `tests` analyzer reports common mistakes in declarations of
tests, fuzzers, benchmarks, and examples in test packages, such as
malformed names, incorrect signatures, or examples that document
non-existent identifiers. Some of these mistakes may cause tests not
to run.
This analyzer is among the subset of analyzers that are run by `go test`.

<!-- go.dev/issue/60529 -->
The existing `printf` analyzer now reports a diagnostic for calls of
the form `fmt.Printf(s)`, where `s` is a non-constant format string,
with no other arguments. Such calls are nearly always a mistake
as the value of `s` may contain the `%` symbol; use `fmt.Print` instead.
See [#60529](/issue/60529).

<!-- go.dev/issue/64127 -->
The existing `buildtag` analyzer now reports a diagnostic when
there is an invalid Go [major version build constraint](/pkg/cmd/go#hdr-Build_constraints)
within a `//go:build` directive. For example, `//go:build go1.23.1` refers to
a point release; use `//go:build go1.23` instead.
See [#64127](/issue/64127).

<!-- go.dev/issue/66387 -->
The existing `copylock` analyzer now reports a diagnostic when a
variable declared in a 3-clause "for" loop such as
`for i := iter(); done(i); i = next(i) { ... }` contains a `sync.Locker`,
such as a `sync.Mutex`. [Go 1.22](/doc/go1.22#language) changed the behavior
of these loops to create a new variable for each iteration, copying the
value from the previous iteration; this copy operation is not safe for locks.
See [#66387](/issue/66387).

### GOCACHEPROG

<!-- go.dev/issue/64876 -->
The `cmd/go` internal binary and test caching mechanism can now be implemented
by child processes implementing a JSON protocol between the `cmd/go` tool
and the child process named by the `GOCACHEPROG` environment variable.
This was previously behind a GOEXPERIMENT.
For protocol details, see [#59719](/issue/59719).
