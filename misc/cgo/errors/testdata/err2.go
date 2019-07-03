// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <stdio.h>

typedef struct foo foo_t;
typedef struct bar bar_t;

foo_t *foop;

long double x = 0;

static int transform(int x) { return x; }

typedef void v;
void F(v** p) {}

void fvi(void *p, int x) {}

void fppi(int** p) {}

int i;
void fi(int i) {}
*/
import "C"
import (
	"unsafe"
)

func main() {
	s := ""
	_ = s
	C.malloc(s) // ERROR HERE

	x := (*C.bar_t)(nil)
	C.foop = x // ERROR HERE

	// issue 13129: used to output error about C.unsignedshort with CC=clang
	var x C.ushort
	x = int(0) // ERROR HERE: C\.ushort

	// issue 13423
	_ = C.fopen() // ERROR HERE

	// issue 13467
	var x rune = '✈'
	var _ rune = C.transform(x) // ERROR HERE: C\.int

	// issue 13635: used to output error about C.unsignedchar.
	// This test tests all such types.
	var (
		_ C.uchar         = "uc"  // ERROR HERE: C\.uchar
		_ C.schar         = "sc"  // ERROR HERE: C\.schar
		_ C.ushort        = "us"  // ERROR HERE: C\.ushort
		_ C.uint          = "ui"  // ERROR HERE: C\.uint
		_ C.ulong         = "ul"  // ERROR HERE: C\.ulong
		_ C.longlong      = "ll"  // ERROR HERE: C\.longlong
		_ C.ulonglong     = "ull" // ERROR HERE: C\.ulonglong
		_ C.complexfloat  = "cf"  // ERROR HERE: C\.complexfloat
		_ C.complexdouble = "cd"  // ERROR HERE: C\.complexdouble
	)

	// issue 13830
	// cgo converts C void* to Go unsafe.Pointer, so despite appearances C
	// void** is Go *unsafe.Pointer. This test verifies that we detect the
	// problem at build time.
	{
		type v [0]byte

		f := func(p **v) {
			C.F((**C.v)(unsafe.Pointer(p))) // ERROR HERE
		}
		var p *v
		f(&p)
	}

	// issue 16116
	_ = C.fvi(1) // ERROR HERE

	// Issue 16591: Test that we detect an invalid call that was being
	// hidden by a type conversion inserted by cgo checking.
	{
		type x *C.int
		var p *x
		C.fppi(p) // ERROR HERE
	}

	// issue 26745
	_ = func(i int) int {
		return C.i + 1 // ERROR HERE: :13
	}
	_ = func(i int) {
		C.fi(i) // ERROR HERE: :6
	}

	C.fi = C.fi // ERROR HERE

}
