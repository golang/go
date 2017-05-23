// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 5242.  Cgo incorrectly computed the alignment of structs
// with no Go accessible fields as 0, and then panicked on
// modulo-by-zero computations.

package cgotest

/*
typedef struct {
} foo;

typedef struct {
	int x : 1;
} bar;

int issue5242(foo f, bar b) {
	return 5242;
}
*/
import "C"

import "testing"

func test5242(t *testing.T) {
	if got := C.issue5242(C.foo{}, C.bar{}); got != 5242 {
		t.Errorf("got %v", got)
	}
}
