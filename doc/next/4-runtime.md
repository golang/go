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
