// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !android

package cgotest

/*
#include <complex.h>

complex float complexFloatSquared(complex float a) { return a*a; }
complex double complexDoubleSquared(complex double a) { return a*a; }
*/
import "C"

import (
	"runtime"
	"testing"
)

func test8694(t *testing.T) {
	if runtime.GOARCH == "arm" {
		t.Skip("test8694 is disabled on ARM because 5l cannot handle thumb library.")
	}
	// Really just testing that this compiles, but check answer anyway.
	x := C.complexfloat(2 + 3i)
	x2 := x * x
	cx2 := C.complexFloatSquared(x)
	if cx2 != x2 {
		t.Errorf("C.complexFloatSquared(%v) = %v, want %v", x, cx2, x2)
	}

	y := C.complexdouble(2 + 3i)
	y2 := y * y
	cy2 := C.complexDoubleSquared(y)
	if cy2 != y2 {
		t.Errorf("C.complexDoubleSquared(%v) = %v, want %v", y, cy2, y2)
	}
}
