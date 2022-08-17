// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher

import (
	"internal/cpu"
	"unsafe"
)

const wordSize = int(unsafe.Sizeof(uintptr(0)))

var hasNEON = cpu.HWCap&(1<<12) != 0

func isAligned(a *byte) bool {
	return uintptr(unsafe.Pointer(a))%uintptr(wordSize) == 0
}

// xorBytes xors the bytes in a and b. The destination should have enough
// space, otherwise xorBytes will panic. Returns the number of bytes xor'd.
func xorBytes(dst, a, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	// make sure dst has enough space
	_ = dst[n-1]

	if hasNEON {
		xorBytesNEON32(&dst[0], &a[0], &b[0], n)
	} else if isAligned(&dst[0]) && isAligned(&a[0]) && isAligned(&b[0]) {
		xorBytesARM32(&dst[0], &a[0], &b[0], n)
	} else {
		safeXORBytes(dst, a, b, n)
	}
	return n
}

// n needs to be smaller or equal than the length of a and b.
func safeXORBytes(dst, a, b []byte, n int) {
	for i := 0; i < n; i++ {
		dst[i] = a[i] ^ b[i]
	}
}

func xorWords(dst, a, b []byte) {
	xorBytes(dst, a, b)
}

//go:noescape
func xorBytesARM32(dst, a, b *byte, n int)

//go:noescape
func xorBytesNEON32(dst, a, b *byte, n int)
