This is a very incomplete and probably out-of-date guide to
programming in the Go runtime and how it differs from writing normal
Go.

Runtime-only compiler directives
================================

In addition to the "//go:" directives documented in "go doc compile",
the compiler supports additional directives only in the runtime.

go:systemstack
--------------

`go:systemstack` indicates that a function must run on the system
stack. This is checked dynamically by a special function prologue.

go:nowritebarrier
-----------------

`go:nowritebarrier` directs the compiler to emit an error if the
following function contains any write barriers. (It *does not*
suppress the generation of write barriers; it is simply an assertion.)

Usually you want `go:nowritebarrierrec`. `go:nowritebarrier` is
primarily useful in situations where it's "nice" not to have write
barriers, but not required for correctness.

go:nowritebarrierrec and go:yeswritebarrierrec
----------------------------------------------

`go:nowritebarrierrec` directs the compiler to emit an error if the
following function or any function it calls recursively, up to a
`go:yeswritebarrierrec`, contains a write barrier.

Logically, the compiler floods the call graph starting from each
`go:nowritebarrierrec` function and produces an error if it encounters
a function containing a write barrier. This flood stops at
`go:yeswritebarrierrec` functions.

`go:nowritebarrierrec` is used in the implementation of the write
barrier to prevent infinite loops.

Both directives are used in the scheduler. The write barrier requires
an active P (`getg().m.p != nil`) and scheduler code often runs
without an active P. In this case, `go:nowritebarrierrec` is used on
functions that release the P or may run without a P and
`go:yeswritebarrierrec` is used when code re-acquires an active P.
Since these are function-level annotations, code that releases or
acquires a P may need to be split across two functions.
