// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This is for issue #63739.
// Ensure that parameters are kept alive until the end of the C call. If not,
// then a stack copy at just the right time while calling into C might think
// that any stack pointers are not alive and fail to update them, causing the C
// function to see the old, no longer correct, pointer values.

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
