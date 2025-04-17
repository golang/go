## Tools {#tools}

### Go command {#go-command}

The `go build` `-asan` option now defaults to doing leak detection at
program exit.
This will report an error if memory allocated by C is not freed and is
not referenced by any other memory allocated by either C or Go.
These new error reports may be disabled by setting
`ASAN_OPTIONS=detect_leaks=0` in the environment when running the
program.

<!-- go.dev/issue/71294 -->

The new `work` package pattern matches all packages in the work (formerly called main)
modules: either the single work module in module mode or the set of workspace modules
in workspace mode.

<!-- go.dev/issue/65847 -->

When the go command updates the `go` line in a `go.mod` or `go.work` file,
it [no longer](/ref/mod#go-mod-file-toolchain) adds a toolchain line
specifying the command's current version.

### Cgo {#cgo}

### Vet {#vet}

<!-- go.dev/issue/18022 -->

The `go vet` command now includes the
[waitgroup](https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/waitgroup)
analyzer, which reports misplaced calls to [sync.WaitGroup.Add].


