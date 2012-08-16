// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
// libgcc on ARM might be compiled as thumb code, but our 5l
// can't handle that, so we have to disable this test on arm.
#ifdef __ARMEL__
#include <stdio.h>
int vabs(int x) {
	puts("testLibgcc is disabled on ARM because 5l cannot handle thumb library.");
	return (x < 0) ? -x : x;
}
#else
int __absvsi2(int); // dummy prototype for libgcc function
// we shouldn't name the function abs, as gcc might use
// the builtin one.
int vabs(int x) { return __absvsi2(x); }
#endif
*/
import "C"

import "testing"

func testLibgcc(t *testing.T) {
	var table = []struct {
		in, out C.int
	}{
		{0, 0},
		{1, 1},
		{-42, 42},
		{1000300, 1000300},
		{1 - 1<<31, 1<<31 - 1},
	}
	for _, v := range table {
		if o := C.vabs(v.in); o != v.out {
			t.Fatalf("abs(%d) got %d, should be %d", v.in, o, v.out)
			return
		}
	}
}
