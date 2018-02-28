// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 11925.  Structs with zero-length trailing fields are now
// padded by the Go compiler.

package cgotest

/*
struct a11925 {
	int i;
	char a[0];
	char b[0];
};

struct b11925 {
	int i;
	char a[0];
	char b[];
};
*/
import "C"

import (
	"testing"
	"unsafe"
)

func test11925(t *testing.T) {
	if C.sizeof_struct_a11925 != unsafe.Sizeof(C.struct_a11925{}) {
		t.Errorf("size of a changed: C %d, Go %d", C.sizeof_struct_a11925, unsafe.Sizeof(C.struct_a11925{}))
	}
	if C.sizeof_struct_b11925 != unsafe.Sizeof(C.struct_b11925{}) {
		t.Errorf("size of b changed: C %d, Go %d", C.sizeof_struct_b11925, unsafe.Sizeof(C.struct_b11925{}))
	}
}
