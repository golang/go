// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package runtime

import "unsafe"

//go:linkname os_sigpipe os.sigpipe
func os_sigpipe() {
	systemstack(sigpipe)
}

// Determines if the signal should be handled by Go and if not, forwards the
// signal to the handler that was installed before Go's.  Returns whether the
// signal was forwarded.
//go:nosplit
func sigfwdgo(sig uint32, info *siginfo, ctx unsafe.Pointer) bool {
	g := getg()
	c := &sigctxt{info, ctx}
	if sig >= uint32(len(sigtable)) {
		return false
	}
	fwdFn := fwdSig[sig]
	flags := sigtable[sig].flags

	// If there is no handler to forward to, no need to forward.
	if fwdFn == _SIG_DFL {
		return false
	}
	// Only forward synchronous signals.
	if c.sigcode() == _SI_USER || flags&_SigPanic == 0 {
		return false
	}
	// Determine if the signal occurred inside Go code.  We test that:
	//   (1) we were in a goroutine (i.e., m.curg != nil), and
	//   (2) we weren't in CGO (i.e., m.curg.syscallsp == 0).
	if g != nil && g.m != nil && g.m.curg != nil && g.m.curg.syscallsp == 0 {
		return false
	}
	// Signal not handled by Go, forward it.
	if fwdFn != _SIG_IGN {
		sigfwd(fwdFn, sig, info, ctx)
	}
	return true
}
