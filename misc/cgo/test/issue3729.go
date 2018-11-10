// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 3729:	cmd/cgo: access errno from void C function
// void f(void) returns [0]byte, error in Go world.

// +build !windows

package cgotest

/*
#include <errno.h>

void g(void) {
	errno = E2BIG;
}

// try to pass some non-trivial arguments to function g2
const char _expA = 0x42;
const float _expB = 3.14159;
const short _expC = 0x55aa;
const int _expD = 0xdeadbeef;
void g2(int x, char a, float b, short c, int d) {
	if (a == _expA && b == _expB && c == _expC && d == _expD)
		errno = x;
	else
		errno = -1;
}
*/
import "C"

import (
	"syscall"
	"testing"
)

func test3729(t *testing.T) {
	_, e := C.g()
	if e != syscall.E2BIG {
		t.Errorf("got %q, expect %q", e, syscall.E2BIG)
	}
	_, e = C.g2(C.EINVAL, C._expA, C._expB, C._expC, C._expD)
	if e != syscall.EINVAL {
		t.Errorf("got %q, expect %q", e, syscall.EINVAL)
	}
}
