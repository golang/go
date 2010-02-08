// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
	The runtime package contains operations that interact with Go's runtime system,
	such as functions to control goroutines. It also includes the low-level type information
	used by the reflect package; see reflect's documentation for the programmable
	interface to the run-time type system.
*/
package runtime

// These functions are implemented in the base runtime library, ../../runtime/.

// Gosched yields the processor, allowing other goroutines to run.  It does not
// suspend the current goroutine, so execution resumes automatically.
func Gosched()

// Goexit terminates the goroutine that calls it.  No other goroutine is affected.
func Goexit()

// Breakpoint() executes a breakpoint trap.
func Breakpoint()

// Caller reports file and line number information about function invocations on
// the calling goroutine's stack.  The argument is the number of stack frames to
// ascend, with 1 identifying the the caller of Caller.  The return values report the
// program counter, file name, and line number within the file of the corresponding
// call.  The boolean ok is false if it was not possible to recover the information.
func Caller(n int) (pc uintptr, file string, line int, ok bool)

// mid returns the current os thread (m) id.
func mid() uint32

// LockOSThread wires the calling goroutine to its current operating system thread.
// Until the calling goroutine exits or calls UnlockOSThread, it will always
// execute in that thread, and no other goroutine can.
// LockOSThread cannot be used during init functions.
func LockOSThread()

// UnlockOSThread unwires the calling goroutine from its fixed operating system thread.
// If the calling goroutine has not called LockOSThread, UnlockOSThread is a no-op.
func UnlockOSThread()

// GOMAXPROCS sets the maximum number of CPUs that can be executing
// simultaneously.   This call will go away when the scheduler improves.
func GOMAXPROCS(n int)

// Cgocalls returns the number of cgo calls made by the current process.
func Cgocalls() int64

// Semacquire waits until *s > 0 and then atomically decrements it.
// It is intended as a simple sleep primitive for use by the synchronization
// library and should not be used directly.
func Semacquire(s *uint32)

// Semrelease atomically increments *s and notifies a waiting goroutine
// if one is blocked in Semacquire.
// It is intended as a simple wakeup primitive for use by the synchronization
// library and should not be used directly.
func Semrelease(s *uint32)

// Sigrecv returns a bitmask of signals that have arrived since the last call to Sigrecv.
// It blocks until at least one signal arrives.
func Sigrecv() uint32

// Signame returns a string describing the signal, or "" if the signal is unknown.
func Signame(sig int32) string

// Siginit enables receipt of signals via Sigrecv.  It should typically
// be called during initialization.
func Siginit()

type MemStatsType struct {
	Alloc      uint64
	TotalAlloc uint64
	Sys        uint64
	Stacks     uint64
	InusePages uint64
	NextGC     uint64
	Lookups    uint64
	Mallocs    uint64
	PauseNs    uint64
	NumGC      uint32
	EnableGC   bool
	DebugGC    bool
	BySize     [67]struct {
		Size    uint32
		Mallocs uint64
		Frees   uint64
	}
}

// MemStats holds statistics about the memory system.
// The statistics are only approximate, as they are not interlocked on update.
var MemStats MemStatsType

// Alloc allocates a block of the given size.
// FOR TESTING AND DEBUGGING ONLY.
func Alloc(uintptr) *byte

// Free frees the block starting at the given pointer.
// FOR TESTING AND DEBUGGING ONLY.
func Free(*byte)

// Lookup returns the base and size of the block containing the given pointer.
// FOR TESTING AND DEBUGGING ONLY.
func Lookup(*byte) (*byte, uintptr)

// GC runs a garbage collection.
func GC()

// SetFinalizer sets the finalizer associated with x to f.
// When the garbage collector finds an unreachable block
// with an associated finalizer, it clears the association and creates
// a new goroutine running f(x).  Creating the new goroutine makes
// x reachable again, but now without an associated finalizer.
// Assuming that SetFinalizer is not called again, the next time
// the garbage collector sees that x is unreachable, it will free x.
//
// SetFinalizer(x, nil) clears any finalizer associated with f.
//
// The argument x must be a pointer to an object allocated by
// calling new or by taking the address of a composite literal.
// The argument f must be a function that takes a single argument
// of x's type and returns no arguments.  If either of these is not
// true, SetFinalizer aborts the program.
//
// Finalizers are run in dependency order: if A points at B, both have
// finalizers, and they are otherwise unreachable, only the finalizer
// for A runs; once A is freed, the finalizer for B can run.
// If a cyclic structure includes a block with a finalizer, that
// cycle is not guaranteed to be garbage collected and the finalizer
// is not guaranteed to run, because there is no ordering that
// respects the dependencies.
//
// The finalizer for x is scheduled to run at some arbitrary time after
// x becomes unreachable.
// There is no guarantee that finalizers will run before a program exits,
// so typically they are useful only for releasing non-memory resources
// associated with an object during a long-running program.
// For example, an os.File object could use a finalizer to close the
// associated operating system file descriptor when a program discards
// an os.File without calling Close, but it would be a mistake
// to depend on a finalizer to flush an in-memory I/O buffer such as a
// bufio.Writer, because the buffer would not be flushed at program exit.
//
// TODO(rsc): make os.File use SetFinalizer
// TODO(rsc): allow f to have (ignored) return values
//
func SetFinalizer(x, f interface{})
