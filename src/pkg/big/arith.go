// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides Go implementations of elementary multi-precision
// arithmetic operations on word vectors. Needed for platforms without
// assembly implementations of these routines.

package big

import "unsafe"


// ----------------------------------------------------------------------------
// Elementary operations on words
//
// These operations are used by the vector operations below.

func addWW_s(x, y, c Word) (z1, z0 Word)

// z1<<_W + z0 = x+y+c, with c == 0 or 1
func addWW_g(x, y, c Word) (z1, z0 Word) {
	yc := y+c;
	z0 = x+yc;
	if z0 < x || yc < y {
		z1 = 1;
	}
	return;
}


func subWW_s(x, y, c Word) (z1, z0 Word)

// z1<<_W + z0 = x-y-c, with c == 0 or 1
func subWW_g(x, y, c Word) (z1, z0 Word) {
	yc := y+c;
	z0 = x-yc;
	if z0 > x || yc < y {
		z1 = 1;
	}
	return;
}


// TODO(gri) mulWW_g is not needed anymore. Keep around for
//           now since mulAddWWW_g should use some of the
//           optimizations from mulWW_g eventually.

// z1<<_W + z0 = x*y
func mulWW_g(x, y Word) (z1, z0 Word) {
	// Split x and y into 2 halfWords each, multiply
	// the halfWords separately while avoiding overflow,
	// and return the product as 2 Words.

	if x < y {
		x, y = y, x
	}

	if x < _B2 {
		// y < _B2 because y <= x
		// sub-digits of x and y are (0, x) and (0, y)
		// z = z[0] = x*y
		z0 = x*y;
		return;
	}

	if y < _B2 {
		// sub-digits of x and y are (x1, x0) and (0, y)
		// x = (x1*_B2 + x0)
		// y = (y1*_B2 + y0)
		x1, x0 := x>>_W2, x&_M2;

		// x*y = t2*_B2*_B2 + t1*_B2 + t0
		t0 := x0*y;
		t1 := x1*y;

		// compute result digits but avoid overflow
		// z = z[1]*_B + z[0] = x*y
		z0 = t1<<_W2 + t0;
		z1 = (t1 + t0>>_W2) >> _W2;
		return;
	}

	// general case
	// sub-digits of x and y are (x1, x0) and (y1, y0)
	// x = (x1*_B2 + x0)
	// y = (y1*_B2 + y0)
	x1, x0 := x>>_W2, x&_M2;
	y1, y0 := y>>_W2, y&_M2;

	// x*y = t2*_B2*_B2 + t1*_B2 + t0
	t0 := x0*y0;
	t1 := x1*y0 + x0*y1;
	t2 := x1*y1;

	// compute result digits but avoid overflow
	// z = z[1]*_B + z[0] = x*y
	z0 = t1<<_W2 + t0;
	z1 = t2 + (t1 + t0>>_W2) >> _W2;
	return;
}


// z1<<_W + z0 = x*y + c
func mulAddWWW_g(x, y, c Word) (z1, z0 Word) {
	// Split x and y into 2 halfWords each, multiply
	// the halfWords separately while avoiding overflow,
	// and return the product as 2 Words.

	// TODO(gri) Should implement special cases for faster execution.

	// general case
	// sub-digits of x, y, and c are (x1, x0), (y1, y0), (c1, c0)
	// x = (x1*_B2 + x0)
	// y = (y1*_B2 + y0)
	x1, x0 := x>>_W2, x&_M2;
	y1, y0 := y>>_W2, y&_M2;
	c1, c0 := c>>_W2, c&_M2;

	// x*y + c = t2*_B2*_B2 + t1*_B2 + t0
	t0 := x0*y0 + c0;
	t1 := x1*y0 + x0*y1 + c1;
	t2 := x1*y1;

	// compute result digits but avoid overflow
	// z = z[1]*_B + z[0] = x*y
	z0 = t1<<_W2 + t0;
	z1 = t2 + (t1 + t0>>_W2) >> _W2;
	return;
}


func divWWW_s(x1, x0, y Word) (q, r Word)

// q = (x1<<_W + x0 - r)/y
func divWW_g(x1, x0, y Word) (q, r Word) {
	if x1 == 0 {
		q, r = x0/y, x0%y;
		return;
	}

	// TODO(gri) implement general case w/o assembly code
	q, r = divWWW_s(x1, x0, y);
	return;
}


// ----------------------------------------------------------------------------
// Elementary operations on vectors

// All higher-level functions use these elementary vector operations.
// The function pointers f are initialized with default implementations
// f_g, written in Go for portability. The corresponding assembly routines
// f_s should be installed if they exist.
var (
	// addVV sets z and returns c such that z+c = x+y.
	addVV func(z, x, y *Word, n int) (c Word)	= addVV_g;

	// subVV sets z and returns c such that z-c = x-y.
	subVV func(z, x, y *Word, n int) (c Word)	= subVV_g;

	// addVW sets z and returns c such that z+c = x-y.
	addVW func(z, x *Word, y Word, n int) (c Word)	= addVW_g;

	// subVW sets z and returns c such that z-c = x-y.
	subVW func(z, x *Word, y Word, n int) (c Word)	= subVW_g;

	// mulAddVWW sets z and returns c such that z+c = x*y + r.
	mulAddVWW func(z, x *Word, y, r Word, n int) (c Word)	= mulAddVWW_g;

	// divWVW sets z and returns r such that z-r = (xn<<(n*_W) + x) / y.
	divWVW func(z* Word, xn Word, x *Word, y Word, n int) (r Word)	= divWVW_g;
)


func useAsm() bool

func init() {
	if useAsm() {
		// Install assemby routines.
		// TODO(gri) This should only be done if the assembly routines are present.
		addVV = addVV_s;
		subVV = subVV_s;
		addVW = addVW_s;
		subVW = subVW_s;
		mulAddVWW = mulAddVWW_s;
		divWVW = divWVW_s;
	}
}


func (p *Word) at(i int) *Word {
	return (*Word)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(i)*_S));
}


func addVV_s(z, x, y *Word, n int) (c Word)
func addVV_g(z, x, y *Word, n int) (c Word) {
	for i := 0; i < n; i++ {
		c, *z.at(i) = addWW_g(*x.at(i), *y.at(i), c);
	}
	return
}


func subVV_s(z, x, y *Word, n int) (c Word)
func subVV_g(z, x, y *Word, n int) (c Word) {
	for i := 0; i < n; i++ {
		c, *z.at(i) = subWW_g(*x.at(i), *y.at(i), c);
	}
	return
}


func addVW_s(z, x *Word, y Word, n int) (c Word)
func addVW_g(z, x *Word, y Word, n int) (c Word) {
	c = y;
	for i := 0; i < n; i++ {
		c, *z.at(i) = addWW_g(*x.at(i), c, 0);
	}
	return
}


func subVW_s(z, x *Word, y Word, n int) (c Word)
func subVW_g(z, x *Word, y Word, n int) (c Word) {
	c = y;
	for i := 0; i < n; i++ {
		c, *z.at(i) = subWW_g(*x.at(i), c, 0);
	}
	return
}


func mulAddVWW_s(z, x *Word, y, r Word, n int) (c Word)
func mulAddVWW_g(z, x *Word, y, r Word, n int) (c Word) {
	c = r;
	for i := 0; i < n; i++ {
		c, *z.at(i) = mulAddWWW_g(*x.at(i), y, c);
	}
	return
}


func divWVW_s(z* Word, xn Word, x *Word, y Word, n int) (r Word)
func divWVW_g(z* Word, xn Word, x *Word, y Word, n int) (r Word) {
	r = xn;
	for i := n-1; i >= 0; i-- {
		*z.at(i), r = divWW_g(r, *x.at(i), y);
	}
	return;
}
