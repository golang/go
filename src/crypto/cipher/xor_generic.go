// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && !ppc64 && !ppc64le && !arm64

package cipher

import (
	"runtime"
	"unsafe"
)

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

	switch {
	case supportsUnaligned:
		fastXORBytes(dst, a, b, n)
	default:
		// TODO(hanwen): if (dst, a, b) have common alignment
		// we could still try fastXORBytes. It is not clear
		// how often this happens, and it's only worth it if
		// the block encryption itself is hardware
		// accelerated.
		safeXORBytes(dst, a, b, n)
	}
	return n
}

const wordSize = int(unsafe.Sizeof(uintptr(0)))
const supportsUnaligned = runtime.GOARCH == "386" || runtime.GOARCH == "ppc64" || runtime.GOARCH == "ppc64le" || runtime.GOARCH == "s390x"

// fastXORBytes xors in bulk. It only works on architectures that
// support unaligned read/writes.
// n needs to be smaller or equal than the length of a and b.
func fastXORBytes(dst, a, b []byte, n int) {
	// Assert dst has enough space
	_ = dst[n-1]

	w := n / wordSize
	if w > 0 {
		dw := *(*[]uintptr)(unsafe.Pointer(&dst))
		aw := *(*[]uintptr)(unsafe.Pointer(&a))
		bw := *(*[]uintptr)(unsafe.Pointer(&b))
		for i := 0; i < w; i++ {
			dw[i] = aw[i] ^ bw[i]
		}
	}

	for i := (n - n%wordSize); i < n; i++ {
		dst[i] = a[i] ^ b[i]
	}
}

// n needs to be smaller or equal than the length of a and b.
func safeXORBytes(dst, a, b []byte, n int) {
	// Load multiple bytes so the compiler can recognize
	// and optimize them into single multi-byte loads
	w := n / 8
	for i := 0; i < w; i++ {
		offset := i * 8
		first32 := uint32(a[offset]) | uint32(a[offset+1])<<8 | uint32(a[offset+2])<<16 | uint32(a[offset+3])<<24
		first32 ^= uint32(b[offset]) | uint32(b[offset+1])<<8 | uint32(b[offset+2])<<16 | uint32(b[offset+3])<<24
		second32 := uint32(a[offset+4]) | uint32(a[offset+5])<<8 | uint32(a[offset+6])<<16 | uint32(a[offset+7])<<24
		second32 ^= uint32(b[offset+4]) | uint32(b[offset+5])<<8 | uint32(b[offset+6])<<16 | uint32(b[offset+7])<<24
		for j := 0; j < 4; j++ {
			dst[offset+j] = byte(first32 >> (j * 8))
			dst[offset+j+4] = byte(second32 >> (j * 8))
		}
	}

	for i := w * 8; i < n; i++ {
		dst[i] = a[i] ^ b[i]
	}
}

// fastXORWords XORs multiples of 4 or 8 bytes (depending on architecture.)
// The arguments are assumed to be of equal length.
func fastXORWords(dst, a, b []byte) {
	dw := *(*[]uintptr)(unsafe.Pointer(&dst))
	aw := *(*[]uintptr)(unsafe.Pointer(&a))
	bw := *(*[]uintptr)(unsafe.Pointer(&b))
	n := len(b) / wordSize
	for i := 0; i < n; i++ {
		dw[i] = aw[i] ^ bw[i]
	}
}

// fastXORWords XORs multiples of 4 or 8 bytes (depending on architecture.)
// The slice arguments a and b are assumed to be of equal length.
func xorWords(dst, a, b []byte) {
	if supportsUnaligned {
		fastXORWords(dst, a, b)
	} else {
		safeXORBytes(dst, a, b, len(b))
	}
}
