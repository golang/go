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

func addWW_s(x, y, c Word) (z1, z0 Word)

// z1<<_W + z0 = x+y+c, with c == 0 or 1
func addWW(x, y, c Word) (z1, z0 Word) {
	yc := y+c;
	z0 = x+yc;
	if z0 < x || yc < y {
		z1 = 1;
	}
	return;
}


func subWW_s(x, y, c Word) (z1, z0 Word)

// z1<<_W + z0 = x-y-c, with c == 0 or 1
func subWW(x, y, c Word) (z1, z0 Word) {
	yc := y+c;
	z0 = x-yc;
	if z0 > x || yc < y {
		z1 = 1;
	}
	return;
}


// z1<<_W + z0 = x*y
func mulW(x, y Word) (z1, z0 Word) {
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
func mulAddWW(x, y, c Word) (z1, z0 Word) {
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


func divWW_s(x1, x0, y Word) (q, r Word)

// q = (x1<<_W + x0 - r)/y
func divWW(x1, x0, y Word) (q, r Word) {
	if x1 == 0 {
		q, r = x0/y, x0%y;
		return;
	}

	// TODO(gri) implement general case w/o assembly code
	q, r = divWW_s(x1, x0, y);
	return;
}


// ----------------------------------------------------------------------------
// Elementary operations on vectors

// For each function f there is a corresponding function f_s which
// implements the same functionality as f but is written in assembly.


func addVV_s(z, x, y *Word, n int) (c Word)

// addVV sets z and returns c such that z+c = x+y.
// z, x, y are n-word vectors.
func addVV(z, x, y *Word, n int) (c Word) {
	for i := 0; i < n; i++ {
		c, *z = addWW(*x, *y, c);
		x = (*Word)(unsafe.Pointer((uintptr(unsafe.Pointer(x)) + _S)));
		y = (*Word)(unsafe.Pointer((uintptr(unsafe.Pointer(y)) + _S)));
		z = (*Word)(unsafe.Pointer((uintptr(unsafe.Pointer(z)) + _S)));

	}
	return
}


func subVV_s(z, x, y *Word, n int) (c Word)

// subVV sets z and returns c such that z-c = x-y.
// z, x, y are n-word vectors.
func subVV(z, x, y *Word, n int) (c Word) {
	for i := 0; i < n; i++ {
		c, *z = subWW(*x, *y, c);
		x = (*Word)(unsafe.Pointer((uintptr(unsafe.Pointer(x)) + _S)));
		y = (*Word)(unsafe.Pointer((uintptr(unsafe.Pointer(y)) + _S)));
		z = (*Word)(unsafe.Pointer((uintptr(unsafe.Pointer(z)) + _S)));
	}
	return
}


func addVW_s(z, x *Word, y Word, n int) (c Word)

// addVW sets z and returns c such that z+c = x-y.
// z, x are n-word vectors.
func addVW(z, x *Word, y Word, n int) (c Word) {
	c = y;
	for i := 0; i < n; i++ {
		c, *z = addWW(*x, c, 0);
		x = (*Word)(unsafe.Pointer((uintptr(unsafe.Pointer(x)) + _S)));
		z = (*Word)(unsafe.Pointer((uintptr(unsafe.Pointer(z)) + _S)));

	}
	return
}

func subVW_s(z, x *Word, y Word, n int) (c Word)

// subVW sets z and returns c such that z-c = x-y.
// z, x are n-word vectors.
func subVW(z, x *Word, y Word, n int) (c Word) {
	c = y;
	for i := 0; i < n; i++ {
		c, *z = subWW(*x, c, 0);
		x = (*Word)(unsafe.Pointer((uintptr(unsafe.Pointer(x)) + _S)));
		z = (*Word)(unsafe.Pointer((uintptr(unsafe.Pointer(z)) + _S)));

	}
	return
}


func mulVW_s(z, x *Word, y Word, n int) (c Word)

// mulVW sets z and returns c such that z+c = x*y.
// z, x are n-word vectors.
func mulVW(z, x *Word, y Word, n int) (c Word) {
	for i := 0; i < n; i++ {
		c, *z = mulAddWW(*x, y, c);
		x = (*Word)(unsafe.Pointer((uintptr(unsafe.Pointer(x)) + _S)));
		z = (*Word)(unsafe.Pointer((uintptr(unsafe.Pointer(z)) + _S)));
	}
	return
}


func divWVW_s(z* Word, xn Word, x *Word, y Word, n int) (r Word)

// divWVW sets z and returns r such that z-r = (xn<<(n*_W) + x) / y.
// z, x are n-word vectors; xn is the extra word x[n] of x.
func divWVW(z* Word, xn Word, x *Word, y Word, n int) (r Word) {
	r = xn;
	x = (*Word)(unsafe.Pointer((uintptr(unsafe.Pointer(x)) + uintptr(n-1)*_S)));
	z = (*Word)(unsafe.Pointer((uintptr(unsafe.Pointer(z)) + uintptr(n-1)*_S)));
	for i := n-1; i >= 0; i-- {
		*z, r = divWW(r, *x, y);
		x = (*Word)(unsafe.Pointer((uintptr(unsafe.Pointer(x)) - _S)));
		z = (*Word)(unsafe.Pointer((uintptr(unsafe.Pointer(z)) - _S)));
	}
	return;
}
