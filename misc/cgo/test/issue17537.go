// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 17537.  The void* cast introduced by cgo to avoid problems
// with const/volatile qualifiers breaks C preprocessor macros that
// emulate functions.

package cgotest

/*
#include <stdlib.h>

typedef struct {
	int i;
} S17537;

int I17537(S17537 *p);

#define I17537(p) ((p)->i)

// Calling this function used to fail without the cast.
const int F17537(const char **p) {
	return **p;
}

// Calling this function used to trigger an error from the C compiler
// (issue 18298).
void F18298(const void *const *p) {
}

// Test that conversions between typedefs work as they used to.
typedef const void *T18298_1;
struct S18298 { int i; };
typedef const struct S18298 *T18298_2;
void G18298(T18298_1 t) {
}
*/
import "C"

import "testing"

func test17537(t *testing.T) {
	v := C.S17537{i: 17537}
	if got, want := C.I17537(&v), C.int(17537); got != want {
		t.Errorf("got %d, want %d", got, want)
	}

	p := (*C.char)(C.malloc(1))
	*p = 17
	if got, want := C.F17537(&p), C.int(17); got != want {
		t.Errorf("got %d, want %d", got, want)
	}

	C.F18298(nil)
	var v18298 C.T18298_2
	C.G18298(C.T18298_1(v18298))
}
