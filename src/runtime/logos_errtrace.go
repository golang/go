// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// This file implements the per-goroutine error-trace ring backing
// logos/rt.TraceErr and ErrorTrace. Each goroutine records up to 8
// propagation sites for its latest error in its g fields; a different
// error value resets the ring. logosErrLast holds only the error's data
// word as a uintptr identity, never a reference, so the trace neither
// keeps the error alive nor needs a write barrier for it.
//
// Both functions are referenced from logos/rt via //go:linkname pull and
// carry push-style //go:linkname marks here, so the linker's linkname
// check (-checklinkname, on by default) accepts the pull: neither name is
// in the linker's blocked map, and a pushed definition allows the pull.

//go:linkname logosTraceRecord
func logosTraceRecord(site unsafe.Pointer, errData uintptr) {
	gp := getg()
	if errData != gp.logosErrLast {
		gp.logosErrLen = 0
		gp.logosErrLast = errData
	}
	if gp.logosErrLen < uint8(len(gp.logosErrSite)) {
		gp.logosErrSite[gp.logosErrLen] = site
		gp.logosErrLen++
	} else {
		// Full: drop the oldest site, keep the last 8 in order.
		copy(gp.logosErrSite[:], gp.logosErrSite[1:])
		gp.logosErrSite[len(gp.logosErrSite)-1] = site
	}
}

//go:linkname logosTraceRead
func logosTraceRead(buf []unsafe.Pointer) int {
	gp := getg()
	n := gp.logosErrLen
	if n > uint8(len(gp.logosErrSite)) {
		n = uint8(len(gp.logosErrSite))
	}
	copy(buf, gp.logosErrSite[:n])
	return int(n)
}

// logosClearErrTrace zeroes gp's error-trace ring. Its single caller is
// gdestroy, so every recycled g starts empty: gfget hands dead gs to new
// goroutines, and a stale logosErrLast could match a new error's recycled
// address and skip the reset. It runs on g0 with a P held like the rest
// of gdestroy and uses normal write barriers for the pointer slots, so it
// must not run in nowritebarrierrec contexts.
func logosClearErrTrace(gp *g) {
	gp.logosErrSite = [8]unsafe.Pointer{}
	gp.logosErrLen = 0
	gp.logosErrLast = 0
}
