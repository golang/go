// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Test seam for M20-S1: opaque wrappers over the fast-call entry and
// exit helpers for package runtime_test, which cannot name m or p.

// LogosEnterFastCallForTest records a fast-call entry on the current M
// and P and reports the observed pin slots, timestamp, and flag.
func LogosEnterFastCallForTest(arg unsafe.Pointer, now int64) (slots [4]uintptr, stamp int64, flag, ok bool) {
	mp := getg().m
	pp := mp.p.ptr()
	if pp == nil {
		return slots, stamp, flag, false
	}
	logosEnterFastCall(mp, pp, arg, now)
	return pp.cgoPinnedPtrs, pp.fastCallEnterNano, mp.inFastCall, true
}

// LogosExitFastCallForTest clears a fast-call entry on the current M
// and P and reports the observed pin slots, timestamp, and flag.
func LogosExitFastCallForTest() (slots [4]uintptr, stamp int64, flag, ok bool) {
	mp := getg().m
	pp := mp.p.ptr()
	if pp == nil {
		return slots, stamp, flag, false
	}
	logosExitFastCall(mp, pp)
	return pp.cgoPinnedPtrs, pp.fastCallEnterNano, mp.inFastCall, true
}
