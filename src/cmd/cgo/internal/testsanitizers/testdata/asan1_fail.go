// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <stdlib.h>
#include <stdio.h>

int *p;
int* test() {
 p = (int *)malloc(2 * sizeof(int));
 free(p);
 return p;
}
*/
import "C"
import "fmt"

func main() {
	// C passes Go an invalid pointer.
	a := C.test()
	// Use after free
	*a = 2 // BOOM
	// We shouldn't get here; asan should stop us first.
	fmt.Println(*a)
}
