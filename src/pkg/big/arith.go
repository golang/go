// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides Go implementations of elementary multi-precision
// arithmetic operations on word vectors. Needed for platforms without
// assembly implementations of these routines.

package big

import "unsafe"

type Word uintptr

const (
	_S	= uintptr(unsafe.Sizeof(Word(0)));	// TODO(gri) should Sizeof return a uintptr?
	_W	= _S*8;
	_B	= 1<<_W;
	_M	= _B-1;
	_W2	= _W/2;
	_B2	= 1<<_W2;
	_M2	= _B2-1;
)


// ----------------------------------------------------------------------------
// Elementary operations on words
//
// These operations are used by the vector operations below.

// z1<<_W + z0 = x+y+c, with c == 0 or 1
func addWW_g(x, y, c Word) (z1, z0 Word) {
	yc := y+c;
	z0 = x+yc;
	if z0 < x || yc < y {
		z1 = 1;
	}
	return;
}


// z1<<_W + z0 = x-y-c, with c == 0 or 1
func subWW_g(x, y, c Word) (z1, z0 Word) {
	yc := y+c;
	z0 = x-yc;
	if z0 > x || yc < y {
		z1 = 1;
	}
	return;
}


// z1<<_W + z0 = x*y
func mulWW_g(x, y Word) (z1, z0 Word) {
	// Split x and y into 2 halfWords each, multiply
	// the halfWords separately while avoiding overflow,
	// and return the product as 2 Words.

	if x < y {
		x, y = y, x;
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
		z1 = (t1 + t0>>_W2)>>_W2;
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
	// t1 := x1*y0 + x0*y1;
	var c Word;
	t1 := x1*y0;
	t1a := t1;
	t1 += x0*y1;
	if t1 < t1a {
		c++;
	}
	t2 := x1*y1 + c*_B2;

	// compute result digits but avoid overflow
	// z = z[1]*_B + z[0] = x*y
	// This may overflow, but that's ok because we also sum t1 and t0 above
	// and we take care of the overflow there.
	z0 = t1<<_W2 + t0;

	// z1 = t2 + (t1 + t0>>_W2)>>_W2;
	var c3 Word;
	z1 = t1 + t0>>_W2;
	if z1 < t1 {
		c3++;
	}
	z1 >>= _W2;
	z1 += c3*_B2;
	z1 += t2;
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
	// (1<<32-1)^2 == 1<<64 - 1<<33 + 1, so there's space to add c0 in here.
	t0 := x0*y0 + c0;

	// t1 := x1*y0 + x0*y1 + c1;
	var c2 Word;	// extra carry
	t1 := x1*y0 + c1;
	t1a := t1;
	t1 += x0*y1;
	if t1 < t1a {	// If the number got smaller then we overflowed.
		c2++;
	}

	t2 := x1*y1 + c2*_B2;

	// compute result digits but avoid overflow
	// z = z[1]*_B + z[0] = x*y
	// z0 = t1<<_W2 + t0;
	// This may overflow, but that's ok because we also sum t1 and t0 below
	// and we take care of the overflow there.
	z0 = t1<<_W2 + t0;

	var c3 Word;
	z1 = t1 + t0>>_W2;
	if z1 < t1 {
		c3++;
	}
	z1 >>= _W2;
	z1 += t2 + c3*_B2;

	return;
}


// q = (x1<<_W + x0 - r)/y
// The most significant bit of y must be 1.
func divStep(x1, x0, y Word) (q, r Word) {
	d1, d0 := y>>_W2, y&_M2;
	q1, r1 := x1/d1, x1%d1;
	m := q1*d0;
	r1 = r1*_B2 | x0>>_W2;
	if r1 < m {
		q1--;
		r1 += y;
		if r1 >= y && r1 < m {
			q1--;
			r1 += y;
		}
	}
	r1 -= m;

	r0 := r1%d1;
	q0 := r1/d1;
	m = q0*d0;
	r0 = r0*_B2 | x0&_M2;
	if r0 < m {
		q0--;
		r0 += y;
		if r0 >= y && r0 < m {
			q0--;
			r0 += y;
		}
	}
	r0 -= m;

	q = q1*_B2 | q0;
	r = r0;
	return;
}


// Number of leading zeros in x.
func leadingZeros(x Word) (n uint) {
	if x == 0 {
		return uint(_W);
	}
	for x&(1<<(_W-1)) == 0 {
		n++;
		x <<= 1;
	}
	return;
}


// q = (x1<<_W + x0 - r)/y
func divWW_g(x1, x0, y Word) (q, r Word) {
	if x1 == 0 {
		q, r = x0/y, x0%y;
		return;
	}

	var q0, q1 Word;
	z := leadingZeros(y);
	if y > x1 {
		if z != 0 {
			y <<= z;
			x1 = (x1<<z)|(x0>>(uint(_W)-z));
			x0 <<= z;
		}
		q0, x0 = divStep(x1, x0, y);
		q1 = 0;
	} else {
		if z == 0 {
			x1 -= y;
			q1 = 1;
		} else {
			z1 := uint(_W)-z;
			y <<= z;
			x2 := x1>>z1;
			x1 = (x1<<z)|(x0>>z1);
			x0 <<= z;
			q1, x1 = divStep(x2, x1, y);
		}

		q0, x0 = divStep(x1, x0, y);
	}

	r = x0>>z;

	if q1 != 0 {
		panic("div out of range");
	}

	return q0, r;
}


// ----------------------------------------------------------------------------
// Elementary operations on vectors

// All higher-level functions use these elementary vector operations.
// The function pointers f are initialized with default implementations
// f_g, written in Go for portability. The corresponding assembly routines
// f_s should be installed if they exist.
var (
	// addVV sets z and returns c such that z+c = x+y.
	addVV	func(z, x, y *Word, n int) (c Word)	= addVV_g;

	// subVV sets z and returns c such that z-c = x-y.
	subVV	func(z, x, y *Word, n int) (c Word)	= subVV_g;

	// addVW sets z and returns c such that z+c = x-y.
	addVW	func(z, x *Word, y Word, n int) (c Word)	= addVW_g;

	// subVW sets z and returns c such that z-c = x-y.
	subVW	func(z, x *Word, y Word, n int) (c Word)	= subVW_g;

	// mulAddVWW sets z and returns c such that z+c = x*y + r.
	mulAddVWW	func(z, x *Word, y, r Word, n int) (c Word)	= mulAddVWW_g;

	// addMulVVW sets z and returns c such that z+c = z + x*y.
	addMulVVW	func(z, x *Word, y Word, n int) (c Word)	= addMulVVW_g;

	// divWVW sets z and returns r such that z-r = (xn<<(n*_W) + x) / y.
	divWVW	func(z *Word, xn Word, x *Word, y Word, n int) (r Word)	= divWVW_g;
)


// UseAsm returns true if the assembly routines are enabled.
func useAsm() bool

func init() {
	if useAsm() {
		// Install assembly routines.
		addVV = addVV_s;
		subVV = subVV_s;
		addVW = addVW_s;
		subVW = subVW_s;
		mulAddVWW = mulAddVWW_s;
		addMulVVW = addMulVVW_s;
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
	return;
}


func subVV_s(z, x, y *Word, n int) (c Word)
func subVV_g(z, x, y *Word, n int) (c Word) {
	for i := 0; i < n; i++ {
		c, *z.at(i) = subWW_g(*x.at(i), *y.at(i), c);
	}
	return;
}


func addVW_s(z, x *Word, y Word, n int) (c Word)
func addVW_g(z, x *Word, y Word, n int) (c Word) {
	c = y;
	for i := 0; i < n; i++ {
		c, *z.at(i) = addWW_g(*x.at(i), c, 0);
	}
	return;
}


func subVW_s(z, x *Word, y Word, n int) (c Word)
func subVW_g(z, x *Word, y Word, n int) (c Word) {
	c = y;
	for i := 0; i < n; i++ {
		c, *z.at(i) = subWW_g(*x.at(i), c, 0);
	}
	return;
}


func mulAddVWW_s(z, x *Word, y, r Word, n int) (c Word)
func mulAddVWW_g(z, x *Word, y, r Word, n int) (c Word) {
	c = r;
	for i := 0; i < n; i++ {
		c, *z.at(i) = mulAddWWW_g(*x.at(i), y, c);
	}
	return;
}


func addMulVVW_s(z, x *Word, y Word, n int) (c Word)
func addMulVVW_g(z, x *Word, y Word, n int) (c Word) {
	for i := 0; i < n; i++ {
		z1, z0 := mulAddWWW_g(*x.at(i), y, *z.at(i));
		c, *z.at(i) = addWW_g(z0, c, 0);
		c += z1;
	}
	return;
}


func divWVW_s(z *Word, xn Word, x *Word, y Word, n int) (r Word)
func divWVW_g(z *Word, xn Word, x *Word, y Word, n int) (r Word) {
	r = xn;
	for i := n-1; i >= 0; i-- {
		*z.at(i), r = divWW_g(r, *x.at(i), y);
	}
	return;
}
