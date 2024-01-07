// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <stdlib.h>
#include <stdio.h>

int *p;
int* f() {
  int i;
  p = (int *)malloc(5*sizeof(int));
  for (i = 0; i < 5; i++) {
    p[i] = i+10;
  }
  return p;
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func main() {
	a := C.f()
	q5 := (*C.int)(unsafe.Add(unsafe.Pointer(a), 4*5))
	// Access to C pointer out of bounds.
	*q5 = 100 // BOOM
	// We shouldn't get here; asan should stop us first.
	fmt.Printf("q5: %d, %x\n", *q5, q5)
}
