// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (mips || mipsle || mips64 || mips64le) && !purego

package subtle

import (
	"unsafe"
)

const wordSize = unsafe.Sizeof(uintptr(0))

// dst,a,b,n, all be wordSize aligned, and n must be > 0 and a multiple of wordSize
//
//go:noescape
func xorBytesAligned(dst, a, b *byte, n int)

func xorBytes(dstb, xb, yb *byte, n int) {
	xp := uintptr(unsafe.Pointer(xb)) % wordSize
	yp := uintptr(unsafe.Pointer(yb)) % wordSize
	dp := uintptr(unsafe.Pointer(dstb)) % wordSize
	if xp != yp || xp != dp {
		dst := unsafe.Slice(dstb, n)
		x := unsafe.Slice(xb, n)
		y := unsafe.Slice(yb, n)
		xorLoop(dst, x, y)
		return
	}
	for (uintptr(unsafe.Pointer(dstb))%wordSize) != 0 && n > 0 {
		*dstb = *xb ^ *yb
		dstb = (*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(dstb)) + 1))
		xb = (*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(xb)) + 1))
		yb = (*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(yb)) + 1))
		n--
	}
	alignedN := n & -int(wordSize)
	if alignedN > 0 {
		xorBytesAligned(dstb, xb, yb, alignedN)
		dstb = (*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(dstb)) + uintptr(alignedN)))
		xb = (*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(xb)) + uintptr(alignedN)))
		yb = (*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(yb)) + uintptr(alignedN)))
		n -= alignedN
	}
	for n > 0 {
		*dstb = *xb ^ *yb
		dstb = (*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(dstb)) + 1))
		xb = (*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(xb)) + 1))
		yb = (*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(yb)) + 1))
		n--
	}
}

func xorLoop[T byte | uintptr](dst, x, y []T) {
	x = x[:len(dst)] // remove bounds check in loop
	y = y[:len(dst)] // remove bounds check in loop
	for i := range dst {
		dst[i] = x[i] ^ y[i]
	}
}
