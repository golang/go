// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4417:	cmd/cgo: bool alignment/padding issue.
// bool alignment is wrong and causing wrong arguments when calling functions.
//

package cgotest

/*
#include <stdbool.h>

static int c_bool(bool a, bool b, int c, bool d, bool e)  {
   return c;
}
*/
import "C"
import "testing"

func testBoolAlign(t *testing.T) {
	b := C.c_bool(true, true, 10, true, false)
	if b != 10 {
		t.Fatalf("found %d expected 10\n", b)
	}
	b = C.c_bool(true, true, 5, true, true)
	if b != 5 {
		t.Fatalf("found %d expected 5\n", b)
	}
	b = C.c_bool(true, true, 3, true, false)
	if b != 3 {
		t.Fatalf("found %d expected 3\n", b)
	}
	b = C.c_bool(false, false, 1, true, false)
	if b != 1 {
		t.Fatalf("found %d expected 1\n", b)
	}
	b = C.c_bool(false, true, 200, true, false)
	if b != 200 {
		t.Fatalf("found %d expected 200\n", b)
	}
}
