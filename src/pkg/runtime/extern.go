// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package runtime contains operations that interact with Go's runtime system,
such as functions to control goroutines. It also includes the low-level type information
used by the reflect package; see reflect's documentation for the programmable
interface to the run-time type system.

Environment Variables

The following environment variables ($name or %name%, depending on the host
operating system) control the run-time behavior of Go programs. The meanings
and use may change from release to release.

The GOGC variable sets the initial garbage collection target percentage.
A collection is triggered when the ratio of freshly allocated data to live data
remaining after the previous collection reaches this percentage. The default
is GOGC=100. Setting GOGC=off disables the garbage collector entirely.
The runtime/debug package's SetGCPercent function allows changing this
percentage at run time. See http://golang.org/pkg/runtime/debug/#SetGCPercent.

The GODEBUG variable controls debug output from the runtime. GODEBUG value is
a comma-separated list of name=val pairs. Supported names are:

	allocfreetrace: setting allocfreetrace=1 causes every allocation to be
	profiled and a stack trace printed on each object's allocation and free.

	efence: setting efence=1 causes the allocator to run in a mode
	where each object is allocated on a unique page and addresses are
	never recycled.

	gctrace: setting gctrace=1 causes the garbage collector to emit a single line to standard
	error at each collection, summarizing the amount of memory collected and the
	length of the pause. Setting gctrace=2 emits the same summary but also
	repeats each collection.

	gcdead: setting gcdead=1 causes the garbage collector to clobber all stack slots
	that it thinks are dead.

	scheddetail: setting schedtrace=X and scheddetail=1 causes the scheduler to emit
	detailed multiline info every X milliseconds, describing state of the scheduler,
	processors, threads and goroutines.

	schedtrace: setting schedtrace=X causes the scheduler to emit a single line to standard
	error every X milliseconds, summarizing the scheduler state.

	scavenge: scavenge=1 enables debugging mode of heap scavenger.

The GOMAXPROCS variable limits the number of operating system threads that
can execute user-level Go code simultaneously. There is no limit to the number of threads
that can be blocked in system calls on behalf of Go code; those do not count against
the GOMAXPROCS limit. This package's GOMAXPROCS function queries and changes
the limit.

The GOTRACEBACK variable controls the amount of output generated when a Go
program fails due to an unrecovered panic or an unexpected runtime condition.
By default, a failure prints a stack trace for every extant goroutine, eliding functions
internal to the run-time system, and then exits with exit code 2.
If GOTRACEBACK=0, the per-goroutine stack traces are omitted entirely.
If GOTRACEBACK=1, the default behavior is used.
If GOTRACEBACK=2, the per-goroutine stack traces include run-time functions.
If GOTRACEBACK=crash, the per-goroutine stack traces include run-time functions,
and if possible the program crashes in an operating-specific manner instead of
exiting. For example, on Unix systems, the program raises SIGABRT to trigger a
core dump.

The GOARCH, GOOS, GOPATH, and GOROOT environment variables complete
the set of Go environment variables. They influence the building of Go programs
(see http://golang.org/cmd/go and http://golang.org/pkg/go/build).
GOARCH, GOOS, and GOROOT are recorded at compile time and made available by
constants or functions in this package, but they do not influence the execution
of the run-time system.
*/
package runtime

import "unsafe"

// Goexit terminates the goroutine that calls it.  No other goroutine is affected.
// Goexit runs all deferred calls before terminating the goroutine.
//
// Calling Goexit from the main goroutine terminates that goroutine
// without func main returning. Since func main has not returned,
// the program continues execution of other goroutines.
// If all other goroutines exit, the program crashes.
func Goexit()

// We assume that all architectures turn faults and the like
// into apparent calls to runtime.sigpanic.  If we see a "call"
// to runtime.sigpanic, we do not back up the PC to find the
// line number of the CALL instruction, because there is no CALL.
var sigpanic byte

// Caller reports file and line number information about function invocations on
// the calling goroutine's stack.  The argument skip is the number of stack frames
// to ascend, with 0 identifying the caller of Caller.  (For historical reasons the
// meaning of skip differs between Caller and Callers.) The return values report the
// program counter, file name, and line number within the file of the corresponding
// call.  The boolean ok is false if it was not possible to recover the information.
func Caller(skip int) (pc uintptr, file string, line int, ok bool) {
	// Ask for two PCs: the one we were asked for
	// and what it called, so that we can see if it
	// "called" sigpanic.
	var rpc [2]uintptr
	if callers(int32(1+skip-1), &rpc[0], 2) < 2 {
		return
	}
	f := findfunc(rpc[1])
	if f == nil {
		// TODO(rsc): Probably a bug?
		// The C version said "have retpc at least"
		// but actually returned pc=0.
		ok = true
		return
	}
	pc = rpc[1]
	xpc := pc
	g := findfunc(rpc[0])
	if xpc > f.entry && (g == nil || g.entry != uintptr(unsafe.Pointer(&sigpanic))) {
		xpc--
	}
	line = int(funcline(f, xpc, &file))
	ok = true
	return
}

func findfunc(uintptr) *_func

//go:noescape
func funcline(*_func, uintptr, *string) int32

// Callers fills the slice pc with the program counters of function invocations
// on the calling goroutine's stack.  The argument skip is the number of stack frames
// to skip before recording in pc, with 0 identifying the frame for Callers itself and
// 1 identifying the caller of Callers.
// It returns the number of entries written to pc.
func Callers(skip int, pc []uintptr) int {
	// runtime.callers uses pc.array==nil as a signal
	// to print a stack trace.  Pick off 0-length pc here
	// so that we don't let a nil pc slice get to it.
	if len(pc) == 0 {
		return 0
	}
	return int(callers(int32(skip), &pc[0], int32(len(pc))))
}

//go:noescape
func callers(int32, *uintptr, int32) int32

//go:noescape
func gcallers(*g, int32, *uintptr, int32) int32

//go:noescape
func gentraceback(uintptr, uintptr, uintptr, *g, int32, *uintptr, int32, unsafe.Pointer, unsafe.Pointer, bool) int32

func getgoroot() string

// GOROOT returns the root of the Go tree.
// It uses the GOROOT environment variable, if set,
// or else the root used during the Go build.
func GOROOT() string {
	s := gogetenv("GOROOT")
	if s != "" {
		return s
	}
	return defaultGoroot
}

// Version returns the Go tree's version string.
// It is either the commit hash and date at the time of the build or,
// when possible, a release tag like "go1.3".
func Version() string {
	return theVersion
}

// GOOS is the running program's operating system target:
// one of darwin, freebsd, linux, and so on.
const GOOS string = theGoos

// GOARCH is the running program's architecture target:
// 386, amd64, or arm.
const GOARCH string = theGoarch
