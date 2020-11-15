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
percentage at run time. See https://golang.org/pkg/runtime/debug/#SetGCPercent.

The GODEBUG variable controls debugging variables within the runtime.
It is a comma-separated list of name=val pairs setting these named variables:

	allocfreetrace: setting allocfreetrace=1 causes every allocation to be
	profiled and a stack trace printed on each object's allocation and free.

	clobberfree: setting clobberfree=1 causes the garbage collector to
	clobber the memory content of an object with bad content when it frees
	the object.

	cgocheck: setting cgocheck=0 disables all checks for packages
	using cgo to incorrectly pass Go pointers to non-Go code.
	Setting cgocheck=1 (the default) enables relatively cheap
	checks that may miss some errors.  Setting cgocheck=2 enables
	expensive checks that should not miss any errors, but will
	cause your program to run slower.

	efence: setting efence=1 causes the allocator to run in a mode
	where each object is allocated on a unique page and addresses are
	never recycled.

	gccheckmark: setting gccheckmark=1 enables verification of the
	garbage collector's concurrent mark phase by performing a
	second mark pass while the world is stopped.  If the second
	pass finds a reachable object that was not found by concurrent
	mark, the garbage collector will panic.

	gcpacertrace: setting gcpacertrace=1 causes the garbage collector to
	print information about the internal state of the concurrent pacer.

	gcshrinkstackoff: setting gcshrinkstackoff=1 disables moving goroutines
	onto smaller stacks. In this mode, a goroutine's stack can only grow.

	gcstoptheworld: setting gcstoptheworld=1 disables concurrent garbage collection,
	making every garbage collection a stop-the-world event. Setting gcstoptheworld=2
	also disables concurrent sweeping after the garbage collection finishes.

	gctrace: setting gctrace=1 causes the garbage collector to emit a single line to standard
	error at each collection, summarizing the amount of memory collected and the
	length of the pause. The format of this line is subject to change.
	Currently, it is:
		gc # @#s #%: #+#+# ms clock, #+#/#/#+# ms cpu, #->#-># MB, # MB goal, # P
	where the fields are as follows:
		gc #        the GC number, incremented at each GC
		@#s         time in seconds since program start
		#%          percentage of time spent in GC since program start
		#+...+#     wall-clock/CPU times for the phases of the GC
		#->#-># MB  heap size at GC start, at GC end, and live heap
		# MB goal   goal heap size
		# P         number of processors used
	The phases are stop-the-world (STW) sweep termination, concurrent
	mark and scan, and STW mark termination. The CPU times
	for mark/scan are broken down in to assist time (GC performed in
	line with allocation), background GC time, and idle GC time.
	If the line ends with "(forced)", this GC was forced by a
	runtime.GC() call.

	inittrace: setting inittrace=1 causes the runtime to emit a single line to standard
	error for each package with init work, summarizing the execution time and memory
	allocation. No information is printed for inits executed as part of plugin loading
	and for packages without both user defined and compiler generated init work.
	The format of this line is subject to change. Currently, it is:
		init # @#ms, # ms clock, # bytes, # allocs
	where the fields are as follows:
		init #      the package name
		@# ms       time in milliseconds when the init started since program start
		# clock     wall-clock time for package initialization work
		# bytes     memory allocated on the heap
		# allocs    number of heap allocations

	madvdontneed: setting madvdontneed=0 will use MADV_FREE
	instead of MADV_DONTNEED on Linux when returning memory to the
	kernel. This is more efficient, but means RSS numbers will
	drop only when the OS is under memory pressure.

	memprofilerate: setting memprofilerate=X will update the value of runtime.MemProfileRate.
	When set to 0 memory profiling is disabled.  Refer to the description of
	MemProfileRate for the default value.

	invalidptr: invalidptr=1 (the default) causes the garbage collector and stack
	copier to crash the program if an invalid pointer value (for example, 1)
	is found in a pointer-typed location. Setting invalidptr=0 disables this check.
	This should only be used as a temporary workaround to diagnose buggy code.
	The real fix is to not store integers in pointer-typed locations.

	sbrk: setting sbrk=1 replaces the memory allocator and garbage collector
	with a trivial allocator that obtains memory from the operating system and
	never reclaims any memory.

	scavenge: scavenge=1 enables debugging mode of heap scavenger.

	scavtrace: setting scavtrace=1 causes the runtime to emit a single line to standard
	error, roughly once per GC cycle, summarizing the amount of work done by the
	scavenger as well as the total amount of memory returned to the operating system
	and an estimate of physical memory utilization. The format of this line is subject
	to change, but currently it is:
		scav # # KiB work, # KiB total, #% util
	where the fields are as follows:
		scav #       the scavenge cycle number
		# KiB work   the amount of memory returned to the OS since the last line
		# KiB total  the total amount of memory returned to the OS
		#% util      the fraction of all unscavenged memory which is in-use
	If the line ends with "(forced)", then scavenging was forced by a
	debug.FreeOSMemory() call.

	scheddetail: setting schedtrace=X and scheddetail=1 causes the scheduler to emit
	detailed multiline info every X milliseconds, describing state of the scheduler,
	processors, threads and goroutines.

	schedtrace: setting schedtrace=X causes the scheduler to emit a single line to standard
	error every X milliseconds, summarizing the scheduler state.

	tracebackancestors: setting tracebackancestors=N extends tracebacks with the stacks at
	which goroutines were created, where N limits the number of ancestor goroutines to
	report. This also extends the information returned by runtime.Stack. Ancestor's goroutine
	IDs will refer to the ID of the goroutine at the time of creation; it's possible for this
	ID to be reused for another goroutine. Setting N to 0 will report no ancestry information.

	asyncpreemptoff: asyncpreemptoff=1 disables signal-based
	asynchronous goroutine preemption. This makes some loops
	non-preemptible for long periods, which may delay GC and
	goroutine scheduling. This is useful for debugging GC issues
	because it also disables the conservative stack scanning used
	for asynchronously preempted goroutines.

The net, net/http, and crypto/tls packages also refer to debugging variables in GODEBUG.
See the documentation for those packages for details.

The GOMAXPROCS variable limits the number of operating system threads that
can execute user-level Go code simultaneously. There is no limit to the number of threads
that can be blocked in system calls on behalf of Go code; those do not count against
the GOMAXPROCS limit. This package's GOMAXPROCS function queries and changes
the limit.

The GORACE variable configures the race detector, for programs built using -race.
See https://golang.org/doc/articles/race_detector.html for details.

The GOTRACEBACK variable controls the amount of output generated when a Go
program fails due to an unrecovered panic or an unexpected runtime condition.
By default, a failure prints a stack trace for the current goroutine,
eliding functions internal to the run-time system, and then exits with exit code 2.
The failure prints stack traces for all goroutines if there is no current goroutine
or the failure is internal to the run-time.
GOTRACEBACK=none omits the goroutine stack traces entirely.
GOTRACEBACK=single (the default) behaves as described above.
GOTRACEBACK=all adds stack traces for all user-created goroutines.
GOTRACEBACK=system is like ``all'' but adds stack frames for run-time functions
and shows goroutines created internally by the run-time.
GOTRACEBACK=crash is like ``system'' but crashes in an operating system-specific
manner instead of exiting. For example, on Unix systems, the crash raises
SIGABRT to trigger a core dump.
For historical reasons, the GOTRACEBACK settings 0, 1, and 2 are synonyms for
none, all, and system, respectively.
The runtime/debug package's SetTraceback function allows increasing the
amount of output at run time, but it cannot reduce the amount below that
specified by the environment variable.
See https://golang.org/pkg/runtime/debug/#SetTraceback.

The GOARCH, GOOS, GOPATH, and GOROOT environment variables complete
the set of Go environment variables. They influence the building of Go programs
(see https://golang.org/cmd/go and https://golang.org/pkg/go/build).
GOARCH, GOOS, and GOROOT are recorded at compile time and made available by
constants or functions in this package, but they do not influence the execution
of the run-time system.
*/
package runtime

import "runtime/internal/sys"

// Caller reports file and line number information about function invocations on
// the calling goroutine's stack. The argument skip is the number of stack frames
// to ascend, with 0 identifying the caller of Caller.  (For historical reasons the
// meaning of skip differs between Caller and Callers.) The return values report the
// program counter, file name, and line number within the file of the corresponding
// call. The boolean ok is false if it was not possible to recover the information.
func Caller(skip int) (pc uintptr, file string, line int, ok bool) {
	rpc := make([]uintptr, 1)
	n := callers(skip+1, rpc[:])
	if n < 1 {
		return
	}
	frame, _ := CallersFrames(rpc).Next()
	return frame.PC, frame.File, frame.Line, frame.PC != 0
}

// Callers fills the slice pc with the return program counters of function invocations
// on the calling goroutine's stack. The argument skip is the number of stack frames
// to skip before recording in pc, with 0 identifying the frame for Callers itself and
// 1 identifying the caller of Callers.
// It returns the number of entries written to pc.
//
// To translate these PCs into symbolic information such as function
// names and line numbers, use CallersFrames. CallersFrames accounts
// for inlined functions and adjusts the return program counters into
// call program counters. Iterating over the returned slice of PCs
// directly is discouraged, as is using FuncForPC on any of the
// returned PCs, since these cannot account for inlining or return
// program counter adjustment.
func Callers(skip int, pc []uintptr) int {
	// runtime.callers uses pc.array==nil as a signal
	// to print a stack trace. Pick off 0-length pc here
	// so that we don't let a nil pc slice get to it.
	if len(pc) == 0 {
		return 0
	}
	return callers(skip, pc)
}

// GOROOT returns the root of the Go tree. It uses the
// GOROOT environment variable, if set at process start,
// or else the root used during the Go build.
func GOROOT() string {
	s := gogetenv("GOROOT")
	if s != "" {
		return s
	}
	return sys.DefaultGoroot
}

// Version returns the Go tree's version string.
// It is either the commit hash and date at the time of the build or,
// when possible, a release tag like "go1.3".
func Version() string {
	return sys.TheVersion
}

// GOOS is the running program's operating system target:
// one of darwin, freebsd, linux, and so on.
// To view possible combinations of GOOS and GOARCH, run "go tool dist list".
const GOOS string = sys.GOOS

// GOARCH is the running program's architecture target:
// one of 386, amd64, arm, s390x, and so on.
const GOARCH string = sys.GOARCH
