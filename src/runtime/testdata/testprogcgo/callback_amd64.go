// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Regression test to verify that cgocallback restores X15 = 0 as required by
// ABIInternal.

/*
#include <stdint.h>

void go_callback_amd64();

static void call_go_callback_amd64() {
	// Clobber X15.
	uint64_t val = 42;
	asm volatile(
		"vmovdqu %0, %%xmm15;"
		:
		: "m" (val)
		: "xmm15");

	go_callback_amd64();
}
*/
import "C"

import (
	"runtime/testdata/testprogcgo/goasm"
)

func init() {
	register("CgoCallbackX15", CgoCallbackX15)
}

//export go_callback_amd64
func go_callback_amd64() {
	v := goasm.ReadX15()
	if v != 0 {
		println("X15 =", v)
		panic("non-zero X15")
	}
}

func CgoCallbackX15() {
	C.call_go_callback_amd64()

	println("OK")
}
