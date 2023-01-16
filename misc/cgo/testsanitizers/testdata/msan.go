// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <stdint.h>

void f(int32_t *p, int n) {
  int i;

  for (i = 0; i < n; i++) {
    p[i] = (int32_t)i;
  }
}
*/
import "C"

import (
	"fmt"
	"os"
	"unsafe"
)

func main() {
	a := make([]int32, 10)
	C.f((*C.int32_t)(unsafe.Pointer(&a[0])), C.int(len(a)))
	for i, v := range a {
		if i != int(v) {
			fmt.Printf("bad %d: %v\n", i, a)
			os.Exit(1)
		}
	}
}
