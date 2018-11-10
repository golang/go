// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test fails on older versions of OS X because they use older buggy
// versions of Clang that emit ambiguous DWARF info.  See issue 8611.
// +build !darwin

package cgotest

// Issue 8428.  Cgo inconsistently translated zero size arrays.

/*
struct issue8428one {
	char b;
	char rest[];
};

struct issue8428two {
	void *p;
	char b;
	char rest[0];
	char pad;
};

struct issue8428three {
	char w[1][2][3][0];
	char x[2][3][0][1];
	char y[3][0][1][2];
	char z[0][1][2][3];
};
*/
import "C"

import "unsafe"

var _ = C.struct_issue8428one{
	b: C.char(0),
	// The trailing rest field is not available in cgo.
	// See issue 11925.
	// rest: [0]C.char{},
}

var _ = C.struct_issue8428two{
	p:    unsafe.Pointer(nil),
	b:    C.char(0),
	rest: [0]C.char{},
}

var _ = C.struct_issue8428three{
	w: [1][2][3][0]C.char{},
	x: [2][3][0][1]C.char{},
	y: [3][0][1][2]C.char{},
	z: [0][1][2][3]C.char{},
}
