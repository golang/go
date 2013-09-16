// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// #include <stdlib.h>
import "C"

import "testing"

func test6390(t *testing.T) {
	p1 := C.malloc(1024)
	if p1 == nil {
		t.Fatalf("C.malloc(1024) returned nil")
	}
	p2 := C.malloc(0)
	if p2 == nil {
		t.Fatalf("C.malloc(0) returned nil")
	}
	C.free(p1)
	C.free(p2)
}
