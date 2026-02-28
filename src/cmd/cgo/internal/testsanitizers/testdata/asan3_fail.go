// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <stdlib.h>
#include <stdio.h>

void test(int *a) {
	// Access Go pointer out of bounds.
	int c = a[5];        // BOOM
	// We shouldn't get here; asan should stop us first.
	printf("a[5]=%d\n", c);
}
*/
import "C"

func main() {
	cIntSlice := []C.int{200, 201, 203, 203, 204}
	C.test(&cIntSlice[0])
}
