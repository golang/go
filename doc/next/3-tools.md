## Tools {#tools}

### Go command {#go-command}

The `go build` `-asan` option now defaults to doing leak detection at
program exit.
This will report an error if memory allocated by C is not freed and is
not referenced by any other memory allocated by either C or Go.
These new error reports may be disabled by setting
`ASAN_OPTIONS=detect_leaks=0` in the environment when running the
program.

### Cgo {#cgo}

