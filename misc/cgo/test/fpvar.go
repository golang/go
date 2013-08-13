// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains test cases for cgo with function pointer variables.

package cgotest

/*
typedef int (*intFunc) ();

int
bridge_int_func(intFunc f)
{
	return f();
}

int fortytwo()
{
	return 42;
}

*/
import "C"
import "testing"

func callBridge(f C.intFunc) int {
	return int(C.bridge_int_func(f))
}

func callCBridge(f C.intFunc) C.int {
	return C.bridge_int_func(f)
}

func testFpVar(t *testing.T) {
	const expected = 42
	f := C.intFunc(C.fortytwo)
	res1 := C.bridge_int_func(f)
	if r1 := int(res1); r1 != expected {
		t.Errorf("got %d, want %d", r1, expected)
	}
	res2 := callCBridge(f)
	if r2 := int(res2); r2 != expected {
		t.Errorf("got %d, want %d", r2, expected)
	}
	r3 := callBridge(f)
	if r3 != expected {
		t.Errorf("got %d, want %d", r3, expected)
	}
}
