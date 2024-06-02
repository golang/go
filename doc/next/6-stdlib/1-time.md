### Timer changes

Go 1.23 makes two significant changes to the implementation of
[time.Timer] and [time.Ticker].

<!-- go.dev/issue/61542 -->
First, `Timer`s and `Ticker`s that are no longer referred to by the program
become eligible for garbage collection immediately, even if their
`Stop` methods have not been called.
Earlier versions of Go did not collect unstopped `Timer`s until after
they had fired and never collected unstopped `Ticker`s.

<!-- go.dev/issue/37196 -->
Second, the timer channel associated with a `Timer` or `Ticker` is
now unbuffered, with capacity 0.
The main effect of this change is that Go now guarantees
that for any call to a `Reset` or `Stop` method, no stale values
prepared before that call will be sent or received after the call.
Earlier versions of Go used channels with a one-element buffer,
making it difficult to use `Reset` and `Stop` correctly.
A visible effect of this change is that `len` and `cap` of timer channels
now returns 0 instead of 1, which may affect programs that
poll the length to decide whether a receive on the timer channel
will succeed.
Such code should use a non-blocking receive instead.

These new behaviors are only enabled when the main Go program
is in a module with a `go.mod` `go` line using Go 1.23.0 or later.
When Go 1.23 builds older programs, the old behaviors remain in effect.
The new [GODEBUG setting](/doc/godebug) [`asynctimerchan=1`](/pkg/time/#NewTimer)
can be used to revert back to asynchronous channel behaviors
even when a program names Go 1.23.0 or later in its `go.mod` file.
