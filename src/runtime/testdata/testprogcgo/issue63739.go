// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This is for issue #56378.
// After we retiring _Cgo_use for parameters, the compiler will treat the
// parameters, start from the second, as non-alive. Then, they will be marked
// as scalar in stackmap, which means the pointer won't be copied correctly
// in copystack.

/*
int add_from_multiple_pointers(int *a, int *b, int *c) {
	*a = *a + 1;
	*b = *b + 1;
	*c = *c + 1;
	return *a + *b + *c;
}
#cgo noescape add_from_multiple_pointers
#cgo nocallback add_from_multiple_pointers
*/
import "C"

import (
	"fmt"
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
	if v != 9 || a != 2 || b != 3 || c != 4 {
		fmt.Printf("%d + %d + %d != %d\n", a, b, c, v)
	}
}

func stackGrow(n int) {
	if n == 0 {
		return
	}
	testCWithMultiplePointers()
	stackGrow(n - 1)
}
