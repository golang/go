// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test a constant in conjunction with pointer checking.

package cgotest

/*
#include <stdlib.h>

#define CheckConstVal 0

typedef struct {
	int *p;
} CheckConstStruct;

static void CheckConstFunc(CheckConstStruct *p, int e) {
}
*/
import "C"

import (
	"testing"
	"unsafe"
)

func testCheckConst(t *testing.T) {
	// The test is that this compiles successfully.
	p := C.malloc(C.size_t(unsafe.Sizeof(C.int(0))))
	defer C.free(p)
	C.CheckConstFunc(&C.CheckConstStruct{(*C.int)(p)}, C.CheckConstVal)
}
