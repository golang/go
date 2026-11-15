// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"testing"
	"unsafe"
)

func TestLogosGrowsliceBufCapStaysInBuffer(t *testing.T) {
	t.Run("pointerful4096", func(t *testing.T) {
		var buf [256 + 64]string
		checkGrowsliceBuf(t, "", unsafe.Pointer(&buf[0]), unsafe.Sizeof(buf[0]), 256, len(buf), func(i int) { buf[i] = "canary" }, func(i int) bool { return buf[i] == "canary" })
	})
	t.Run("noscanBetweenSizeClasses", func(t *testing.T) {
		var buf [7 + 16][14]byte
		checkGrowsliceBuf(t, [14]byte{}, unsafe.Pointer(&buf[0]), unsafe.Sizeof(buf[0]), 7, len(buf), func(i int) { buf[i][0] = 0xaa }, func(i int) bool { return buf[i][0] == 0xaa })
	})
}

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
