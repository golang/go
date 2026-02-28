// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
typedef void *HANDLE;

struct HWND__{int unused;}; typedef struct HWND__ *HWND;
*/
import "C"

import (
	"testing"
	"unsafe"
)

func test42018(t *testing.T) {
	// Test that Windows handles are marked go:notinheap, by growing the
	// stack and checking for pointer adjustments. Trick from
	// test/fixedbugs/issue40954.go.
	var i int
	handle := C.HANDLE(unsafe.Pointer(uintptr(unsafe.Pointer(&i))))
	recurseHANDLE(100, handle, uintptr(unsafe.Pointer(&i)))
	hwnd := C.HWND(unsafe.Pointer(uintptr(unsafe.Pointer(&i))))
	recurseHWND(400, hwnd, uintptr(unsafe.Pointer(&i)))
}

func recurseHANDLE(n int, p C.HANDLE, v uintptr) {
	if n > 0 {
		recurseHANDLE(n-1, p, v)
	}
	if uintptr(unsafe.Pointer(p)) != v {
		panic("adjusted notinheap pointer")
	}
}

func recurseHWND(n int, p C.HWND, v uintptr) {
	if n > 0 {
		recurseHWND(n-1, p, v)
	}
	if uintptr(unsafe.Pointer(p)) != v {
		panic("adjusted notinheap pointer")
	}
}
