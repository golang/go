// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
// Mac OS X's gcc will generate scattered relocation 2/1 for
// this function on Darwin/386, and 8l couldn't handle it.
// this example is in issue 1635
#include <stdio.h>
void scatter() {
	void *p = scatter;
	printf("scatter = %p\n", p);
}

// Adding this explicit extern declaration makes this a test for
// https://gcc.gnu.org/PR68072 aka https://golang.org/issue/13344 .
// It used to cause a cgo error when building with GCC 6.
extern int hola;

// this example is in issue 3253
int hola = 0;
int testHola() { return hola; }
*/
import "C"

import "testing"

func test1635(t *testing.T) {
	C.scatter()
	if v := C.hola; v != 0 {
		t.Fatalf("C.hola is %d, should be 0", v)
	}
	if v := C.testHola(); v != 0 {
		t.Fatalf("C.testHola() is %d, should be 0", v)
	}
}
