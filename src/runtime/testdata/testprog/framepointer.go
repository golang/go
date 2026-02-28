// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64

package main

import "unsafe"

func init() {
	register("FramePointerAdjust", FramePointerAdjust)
}

func FramePointerAdjust() { framePointerAdjust1(0) }

//go:noinline
func framePointerAdjust1(x int) {
	argp := uintptr(unsafe.Pointer(&x))
	fp := *getFP()
	if !(argp-0x100 <= fp && fp <= argp+0x100) {
		print("saved FP=", fp, " &x=", argp, "\n")
		panic("FAIL")
	}

	// grow the stack
	grow(10000)

	// check again
	argp = uintptr(unsafe.Pointer(&x))
	fp = *getFP()
	if !(argp-0x100 <= fp && fp <= argp+0x100) {
		print("saved FP=", fp, " &x=", argp, "\n")
		panic("FAIL")
	}
}

func grow(n int) {
	if n > 0 {
		grow(n - 1)
	}
}

func getFP() *uintptr
