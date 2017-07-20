// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Support for memory sanitizer. See runtime/cgo/sigaction.go.

// +build linux,amd64

package runtime

import "unsafe"

// _cgo_sigaction is filled in by runtime/cgo when it is linked into the
// program, so it is only non-nil when using cgo.
//go:linkname _cgo_sigaction _cgo_sigaction
var _cgo_sigaction unsafe.Pointer

//go:nosplit
//go:nowritebarrierrec
func rt_sigaction(sig uintptr, new, old *sigactiont, size uintptr) int32 {
	// The runtime package is explicitly blacklisted from sanitizer
	// instrumentation in racewalk.go, but we might be calling into instrumented C
	// functions here — so we need the pointer parameters to be properly marked.
	//
	// Mark the input as having been written before the call and the output as
	// read after.
	if msanenabled && new != nil {
		msanwrite(unsafe.Pointer(new), unsafe.Sizeof(*new))
	}

	var ret int32

	if _cgo_sigaction == nil || inForkedChild {
		ret = sysSigaction(sig, new, old, size)
	} else {
		// We need to call _cgo_sigaction, which means we need a big enough stack
		// for C.  To complicate matters, we may be in libpreinit (before the
		// runtime has been initialized) or in an asynchronous signal handler (with
		// the current thread in transition between goroutines, or with the g0
		// system stack already in use).

		g := getg()
		sp := uintptr(unsafe.Pointer(&sig))
		switch {
		case g == nil:
			// No g: we're on a C stack or a signal stack.
			ret = callCgoSigaction(sig, new, old)
		case sp < g.stack.lo || sp >= g.stack.hi:
			// We're no longer on g's stack, so we must be handling a signal.  It's
			// possible that we interrupted the thread during a transition between g
			// and g0, so we should stay on the current stack to avoid corrupting g0.
			ret = callCgoSigaction(sig, new, old)
		default:
			// We're running on g's stack, so either we're not in a signal handler or
			// the signal handler has set the correct g.  If we're on gsignal or g0,
			// systemstack will make the call directly; otherwise, it will switch to
			// g0 to ensure we have enough room to call a libc function.
			//
			// The function literal that we pass to systemstack is not nosplit, but
			// that's ok: we'll be running on a fresh, clean system stack so the stack
			// check will always succeed anyway.
			systemstack(func() {
				ret = callCgoSigaction(sig, new, old)
			})
		}

		const EINVAL = 22
		if ret == EINVAL {
			// libc reserves certain signals — normally 32-33 — for pthreads, and
			// returns EINVAL for sigaction calls on those signals.  If we get EINVAL,
			// fall back to making the syscall directly.
			ret = sysSigaction(sig, new, old, size)
		}
	}

	if msanenabled && old != nil && ret == 0 {
		msanread(unsafe.Pointer(old), unsafe.Sizeof(*old))
	}
	return ret
}

// sysSigaction calls the rt_sigaction system call. It is implemented in assembly.
//go:noescape
func sysSigaction(sig uintptr, new, old *sigactiont, size uintptr) int32

// callCgoSigaction calls the sigaction function in the runtime/cgo package
// using the GCC calling convention. It is implemented in assembly.
//go:noescape
func callCgoSigaction(sig uintptr, new, old *sigactiont) int32
