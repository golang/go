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

// LogosGrowsliceBufForTest calls growsliceBuf for elements of the dynamic
// type of elem, using buf (bufLen elements, already holding newLen-num
// elements) as the backing store, and reports the resulting slice header.
func LogosGrowsliceBufForTest(elem any, buf unsafe.Pointer, bufLen, newLen, num int) (array unsafe.Pointer, length, capacity int) {
	s := growsliceBuf(buf, newLen, bufLen, num, efaceOf(&elem)._type, buf, bufLen)
	return s.array, s.len, s.cap
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
