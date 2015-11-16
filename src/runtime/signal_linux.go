// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Continuation of the (assembly) sigtramp() logic.
//go:nosplit
func sigtrampgo(sig uint32, info *siginfo, ctx unsafe.Pointer) {
	if sigfwdgo(sig, info, ctx) {
		return
	}
	g := getg()
	if g == nil {
		badsignal(uintptr(sig))
		return
	}
	setg(g.m.gsignal)
	sighandler(sig, info, ctx, g)
	setg(g)
}
