// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

void f(int32_t *p, int n) {
  int32_t * volatile q = (int32_t *)malloc(sizeof(int32_t) * n);
  memcpy(p, q, n * sizeof(*p));
  free(q);
}

void g(int32_t *p, int n) {
  if (p[4] != 1) {
    abort();
  }
}
*/
import "C"

import (
	"unsafe"
)

func main() {
	a := make([]int32, 10)
	C.f((*C.int32_t)(unsafe.Pointer(&a[0])), C.int(len(a)))
	a[4] = 1
	C.g((*C.int32_t)(unsafe.Pointer(&a[0])), C.int(len(a)))
}
