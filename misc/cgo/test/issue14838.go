// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 14838. add CBytes function

package cgotest

/*
#include <stdlib.h>

int check_cbytes(char *b, size_t l) {
	int i;
	for (i = 0; i < l; i++) {
		if (b[i] != i) {
			return 0;
		}
	}
	return 1;
}
*/
import "C"

import (
	"testing"
	"unsafe"
)

func test14838(t *testing.T) {
	data := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	cData := C.CBytes(data)
	defer C.free(cData)

	if C.check_cbytes((*C.char)(cData), C.size_t(len(data))) == 0 {
		t.Fatalf("mismatched data: expected %v, got %v", data, (*(*[10]byte)(unsafe.Pointer(cData)))[:])
	}
}
