// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <stdlib.h>
#include <stdio.h>

void test(int* a) {
	// Access Go pointer out of bounds.
	a[3] = 300;          // BOOM
	// We shouldn't get here; asan should stop us first.
	printf("a[3]=%d\n", a[3]);
}*/
import "C"

func main() {
	var cIntArray [2]C.int
	C.test(&cIntArray[0]) // cIntArray is moved to heap.
}
