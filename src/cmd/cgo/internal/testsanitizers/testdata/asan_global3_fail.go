// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <stdlib.h>
#include <stdio.h>

int test(int *a) {
	int* p = a+1;
	*p = 10;          // BOOM
	return *p;
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

var cIntV C.int

func main() {
	r := C.test((*C.int)(unsafe.Pointer(&cIntV)))
	fmt.Printf("r value is %d", r)
}
