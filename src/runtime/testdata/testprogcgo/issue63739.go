// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9 && !windows
// +build !plan9,!windows

package main

// This is for issue #56378.
// After we retiring _Cgo_use for parameters, the compiler will treat the
// parameters, start from the second, as non-alive. Then, they will be marked
// as scalar in stackmap, which means the pointer won't be copied correctly
// in copyStack.

/*
int add_from_multiple_pointers(int *a, int *b, int *c) {
	return *a + *b + *c;
}
#cgo noescape add_from_multiple_pointers
#cgo nocallback add_from_multiple_pointers
*/
import "C"

import (
	"fmt"
	"os"
	"time"
)

const (
	maxStack = 1024
)

func init() {
	register("CgoEscapeWithMultiplePointers", CgoEscapeWithMultiplePointers)
}

func CgoEscapeWithMultiplePointers() {
	stackGrow(maxStack)
	fmt.Println("OK")
}

//go:noinline
func testCWithMultiplePointers() {
	var a C.int = 1
	var b C.int = 2
	var c C.int = 3
	v := C.add_from_multiple_pointers(&a, &b, &c)
	if v != 6 {
		fmt.Printf("%d + %d + %d != %d", a, b, c, v)
	}
}

func stackGrow(n int) {
	if n == 0 {
		return
	}
	testCWithMultiplePointers()
	stackGrow(n - 1)
}
