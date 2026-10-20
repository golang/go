// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"testing"
	"unsafe"
)

// TestLogosGrowsliceBufCapStaysInBuffer checks that growsliceBuf never
// reports a capacity larger than the stack buffer it hands back. Rounding
// the length up to a malloc size class can overshoot the buffer: a
// pointerful buffer above 512 bytes pays an 8-byte malloc header, and a
// buffer whose byte size is not a size class rounds up to the next class.
// An oversized capacity makes the clear in growsliceBuf and every later
// append write past the buffer, over the caller's saved frame pointer and
// return address. A -d=variablemakethreshold above the 32-byte default
// (as the -O=3 tier sets) makes such buffers reachable.
func TestLogosGrowsliceBufCapStaysInBuffer(t *testing.T) {
	t.Run("pointerful4096", func(t *testing.T) {
		// 256 strings fill a 4096-byte buffer, which crosses the
		// 512-byte malloc header threshold: roundupsize(4096) = 4856.
		var buf [256 + 64]string
		checkGrowsliceBuf(t, "", unsafe.Pointer(&buf[0]), unsafe.Sizeof(buf[0]), 256, len(buf), func(i int) { buf[i] = "canary" }, func(i int) bool { return buf[i] == "canary" })
	})
	t.Run("noscanBetweenSizeClasses", func(t *testing.T) {
		// 7 elements of 14 bytes are 98 bytes, which round up to the
		// 112-byte size class (8 elements).
		var buf [7 + 16][14]byte
		checkGrowsliceBuf(t, [14]byte{}, unsafe.Pointer(&buf[0]), unsafe.Sizeof(buf[0]), 7, len(buf), func(i int) { buf[i][0] = 0xaa }, func(i int) bool { return buf[i][0] == 0xaa })
	})
}

// checkGrowsliceBuf grows a bufLen-element buffer one element at a time
// up to bufLen and checks, after each call, that the capacity fits the
// buffer and that the canary elements past the buffer are untouched.
// total is the number of elements in the backing array, canary elements
// included.
func checkGrowsliceBuf(t *testing.T, elem any, buf unsafe.Pointer, size uintptr, bufLen, total int, setCanary func(int), canaryOK func(int) bool) {
	t.Helper()
	for i := bufLen; i < total; i++ {
		setCanary(i)
	}
	for newLen := 1; newLen <= bufLen; newLen++ {
		array, n, c := runtime.LogosGrowsliceBufForTest(elem, buf, bufLen, newLen, 1)
		if array != buf || n != newLen {
			t.Fatalf("newLen=%d: got array=%p len=%d, want array=%p len=%d", newLen, array, n, buf, newLen)
		}
		if c < newLen || c > bufLen {
			t.Fatalf("newLen=%d: cap=%d outside [%d, %d] for a %d-byte buffer", newLen, c, newLen, bufLen, uintptr(bufLen)*size)
		}
		for i := bufLen; i < total; i++ {
			if !canaryOK(i) {
				t.Fatalf("newLen=%d: element %d past the %d-element buffer was overwritten", newLen, i, bufLen)
			}
		}
	}
}
