// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Support for sanitizers. See runtime/cgo/sigaction.go.
// Also used on linux/386 to clear the SA_RESTORER flag
// when using cgo; see issue #75253.

//go:build (linux && (386 || amd64 || arm64 || loong64 || ppc64 || ppc64le || riscv64)) || (freebsd && amd64)

package runtime

import (
	"internal/goexperiment"
	"unsafe"
)

// _cgo_sigaction is filled in by runtime/cgo when it is linked into the
// program, so it is only non-nil when using cgo.
//
//go:linkname _cgo_sigaction _cgo_sigaction
var _cgo_sigaction unsafe.Pointer

// cgoSigaction is the Go equivalent of _cgo_sigaction, used on platforms
// where runtime/cgo implements it in Go. noG reports whether the call must
// stay on the current stack without touching g or TLS (for example, during
// libpreinit before runtime initialization or in an async signal handler).
//
//go:linknamestd cgoSigaction
var cgoSigaction func(sig uintptr, new, old *sigactiont, noG bool) int32

//go:nosplit
//go:nowritebarrierrec
func sigaction(sig uint32, new, old *sigactiont) {
	// racewalk.go avoids adding sanitizing instrumentation to package runtime,
	// but we might be calling into instrumented C functions here,
	// so we need the pointer parameters to be properly marked.
	//
	// Mark the input as having been written before the call
	// and the output as read after.
	if msanenabled && new != nil {
		msanwrite(unsafe.Pointer(new), unsafe.Sizeof(*new))
	}
	if asanenabled && new != nil {
		asanwrite(unsafe.Pointer(new), unsafe.Sizeof(*new))
	}
	var haveCgoSigaction bool
	if goexperiment.CgoBindings {
		haveCgoSigaction = cgoSigaction != nil
	} else {
		haveCgoSigaction = _cgo_sigaction != nil
	}
	if !haveCgoSigaction || inForkedChild {
		sysSigaction(sig, new, old)
	} else {
		// We need to call _cgo_sigaction, which means we need a big enough stack
		// for C.  To complicate matters, we may be in libpreinit (before the
		// runtime has been initialized) or in an asynchronous signal handler (with
		// the current thread in transition between goroutines, or with the g0
		// system stack already in use).

		var ret int32

		fixSigactionForCgo(new)

		var g *g
		if mainStarted {
			g = getg()
		}
		sp := uintptr(unsafe.Pointer(&sig))
		switch {
		case g == nil:
			// No g: we're on a C stack or a signal stack.
			ret = doCgoSigaction(uintptr(sig), new, old, true)
		case sp < g.stack.lo || sp >= g.stack.hi:
			// We're no longer on g's stack, so we must be handling a signal.  It's
			// possible that we interrupted the thread during a transition between g
			// and g0, so we should stay on the current stack to avoid corrupting g0.
			ret = doCgoSigaction(uintptr(sig), new, old, true)
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
				ret = doCgoSigaction(uintptr(sig), new, old, false)
			})
		}

		const EINVAL = 22
		if ret == EINVAL {
			// libc reserves certain signals — normally 32-33 — for pthreads, and
			// returns EINVAL for sigaction calls on those signals.  If we get EINVAL,
			// fall back to making the syscall directly.
			sysSigaction(sig, new, old)
		}
	}

	if msanenabled && old != nil {
		msanread(unsafe.Pointer(old), unsafe.Sizeof(*old))
	}
	if asanenabled && old != nil {
		asanread(unsafe.Pointer(old), unsafe.Sizeof(*old))
	}
}

// doCgoSigaction calls the sigaction implementation in runtime/cgo, using the
// Go cgoSigaction hook when available or falling back to _cgo_sigaction via
// callCgoSigaction.
//
//go:nosplit
//go:nowritebarrierrec
func doCgoSigaction(sig uintptr, new, old *sigactiont, noG bool) int32 {
	if goexperiment.CgoBindings {
		return cgoSigaction(sig, noEscapePtr(new), noEscapePtr(old), noG)
	}
	return callCgoSigaction(sig, new, old)
}

// callCgoSigaction calls the sigaction function in the runtime/cgo package
// using the GCC calling convention. It is implemented in assembly.
//
//go:noescape
func callCgoSigaction(sig uintptr, new, old *sigactiont) int32
