// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 10303. Pointers passed to C were not marked as escaping (bug in cgo).

package cgotest

import "runtime"

/*
typedef int *intptr;

void setintstar(int *x) {
	*x = 1;
}

void setintptr(intptr x) {
	*x = 1;
}

void setvoidptr(void *x) {
	*(int*)x = 1;
}

typedef struct Struct Struct;
struct Struct {
	int *P;
};

void setstruct(Struct s) {
	*s.P = 1;
}

*/
import "C"

import (
	"testing"
	"unsafe"
)

func test10303(t *testing.T, n int) {
	if runtime.Compiler == "gccgo" {
		t.Skip("gccgo permits C pointers on the stack")
	}

	// Run at a few different stack depths just to avoid an unlucky pass
	// due to variables ending up on different pages.
	if n > 0 {
		test10303(t, n-1)
	}
	if t.Failed() {
		return
	}
	var x, y, z, v, si C.int
	var s C.Struct
	C.setintstar(&x)
	C.setintptr(&y)
	C.setvoidptr(unsafe.Pointer(&v))
	s.P = &si
	C.setstruct(s)

	if uintptr(unsafe.Pointer(&x))&^0xfff == uintptr(unsafe.Pointer(&z))&^0xfff {
		t.Error("C int* argument on stack")
	}
	if uintptr(unsafe.Pointer(&y))&^0xfff == uintptr(unsafe.Pointer(&z))&^0xfff {
		t.Error("C intptr argument on stack")
	}
	if uintptr(unsafe.Pointer(&v))&^0xfff == uintptr(unsafe.Pointer(&z))&^0xfff {
		t.Error("C void* argument on stack")
	}
	if uintptr(unsafe.Pointer(&si))&^0xfff == uintptr(unsafe.Pointer(&z))&^0xfff {
		t.Error("C struct field pointer on stack")
	}
}
