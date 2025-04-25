// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!amd64 && !arm && !arm64 && !loong64 && !mips && !mipsle && !mips64 && !mips64le && !ppc64 && !ppc64le && !riscv64) || purego

package subtle

import (
	"runtime"
	"unsafe"
)

const wordSize = unsafe.Sizeof(uintptr(0))

const supportsUnaligned = runtime.GOARCH == "386" ||
	runtime.GOARCH == "amd64" ||
	runtime.GOARCH == "ppc64" ||
	runtime.GOARCH == "ppc64le" ||
	runtime.GOARCH == "s390x"

func xorBytes(dstb, xb, yb *byte, n int) {
	// xorBytes assembly is written using pointers and n. Back to slices.
	dst := unsafe.Slice(dstb, n)
	x := unsafe.Slice(xb, n)
	y := unsafe.Slice(yb, n)

	if supportsUnaligned || aligned(dstb, xb, yb) {
		xorLoop(words(dst), words(x), words(y))
		if uintptr(n)%wordSize == 0 {
			return
		}
		done := n &^ int(wordSize-1)
		dst = dst[done:]
		x = x[done:]
		y = y[done:]
	}
	xorLoop(dst, x, y)
}

// aligned reports whether dst, x, and y are all word-aligned pointers.
func aligned(dst, x, y *byte) bool {
	return (uintptr(unsafe.Pointer(dst))|uintptr(unsafe.Pointer(x))|uintptr(unsafe.Pointer(y)))&(wordSize-1) == 0
}

// words returns a []uintptr pointing at the same data as x,
// with any trailing partial word removed.
func words(x []byte) []uintptr {
	n := uintptr(len(x)) / wordSize
	if n == 0 {
		// Avoid creating a *uintptr that refers to data smaller than a uintptr;
		// see issue 59334.
		return nil
	}
	return unsafe.Slice((*uintptr)(unsafe.Pointer(&x[0])), n)
}

func xorLoop[T byte | uintptr](dst, x, y []T) {
	x = x[:len(dst)] // remove bounds check in loop
	y = y[:len(dst)] // remove bounds check in loop
	for i := range dst {
		dst[i] = x[i] ^ y[i]
	}
}
