## Runtime {#runtime}

<!-- CL 742580 -->

Tracebacks for modules with `go` directives configuring Go 1.27 or later will now
include [runtime/pprof](https://pkg.go.dev/runtime/pprof) goroutine labels in
the header line. This behavior can be disabled with `GODEBUG=tracebacklabels=0`
(added in [Go 1.26](/doc/godebug#go-126)). This opt-out is expected to be
kept indefinitely in case goroutine labels acquire sensitive information that
shouldn't be made available in tracebacks.

<!-- CL 781580 -->

The `asynctimerchan` GODEBUG setting (added in [Go 1.23](/doc/godebug#go-123))
has been removed permanently. Channels created by package [time](https://pkg.go.dev/time)
are now always unbuffered (synchronous), irrespective of GODEBUG settings.

### Faster memory allocation

<!-- go.dev.issue/79286 -->

The compiler will now generate calls to size-specialized memory allocation
routines, reducing the cost of some small (<80 byte) memory allocations by
up to 30%.
Improvements vary depending on the workload, but the overall improvement is
expected to be ~1% in real allocation-heavy programs.
This causes the binary size to increase by about 60 KB (independent of the
workload).
Please [file an issue](/issue/new) if you notice any regressions.
You may set `GOEXPERIMENT=nosizespecializedmalloc` at build time to disable
it.
This opt-out setting is expected to be removed in Go 1.28.
