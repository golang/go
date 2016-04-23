// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd

package runtime

import "unsafe"

//go:noescape
func sigfwd(fn uintptr, sig uint32, info *siginfo, ctx unsafe.Pointer)

// Determines if the signal should be handled by Go and if not, forwards the
// signal to the handler that was installed before Go's. Returns whether the
// signal was forwarded.
// This is called by the signal handler, and the world may be stopped.
//go:nosplit
//go:nowritebarrierrec
func sigfwdgo(sig uint32, info *siginfo, ctx unsafe.Pointer) bool {
	if sig >= uint32(len(sigtable)) {
		return false
	}
	fwdFn := fwdSig[sig]

	if !signalsOK {
		// The only way we can get here is if we are in a
		// library or archive, we installed a signal handler
		// at program startup, but the Go runtime has not yet
		// been initialized.
		if fwdFn == _SIG_DFL {
			dieFromSignal(int32(sig))
		} else {
			sigfwd(fwdFn, sig, info, ctx)
		}
		return true
	}

	flags := sigtable[sig].flags

	// If there is no handler to forward to, no need to forward.
	if fwdFn == _SIG_DFL {
		return false
	}

	// If we aren't handling the signal, forward it.
	if flags&_SigHandling == 0 {
		sigfwd(fwdFn, sig, info, ctx)
		return true
	}

	// Only forward synchronous signals.
	c := &sigctxt{info, ctx}
	if c.sigcode() == _SI_USER || flags&_SigPanic == 0 {
		return false
	}
	// Determine if the signal occurred inside Go code. We test that:
	//   (1) we were in a goroutine (i.e., m.curg != nil), and
	//   (2) we weren't in CGO (i.e., m.curg.syscallsp == 0).
	g := getg()
	if g != nil && g.m != nil && g.m.curg != nil && g.m.curg.syscallsp == 0 {
		return false
	}
	// Signal not handled by Go, forward it.
	if fwdFn != _SIG_IGN {
		sigfwd(fwdFn, sig, info, ctx)
	}
	return true
}
