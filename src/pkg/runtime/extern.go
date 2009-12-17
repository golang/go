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
