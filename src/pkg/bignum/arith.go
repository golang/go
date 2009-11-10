// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fast versions of the routines in this file are in fast.arith.s.
// Simply replace this file with arith.s (renamed from fast.arith.s)
// and the bignum package will build and run on a platform that
// supports the assembly routines.

package bignum

import "unsafe"

// z1<<64 + z0 = x*y
func Mul128(x, y uint64) (z1, z0 uint64) {
	// Split x and y into 2 halfwords each, multiply
	// the halfwords separately while avoiding overflow,
	// and return the product as 2 words.

	const (
		W	= uint(unsafe.Sizeof(x)) * 8;
		W2	= W / 2;
		B2	= 1 << W2;
		M2	= B2 - 1;
	)

	if x < y {
		x, y = y, x
	}

	if x < B2 {
		// y < B2 because y <= x
		// sub-digits of x and y are (0, x) and (0, y)
		// z = z[0] = x*y
		z0 = x * y;
		return;
	}

	if y < B2 {
		// sub-digits of x and y are (x1, x0) and (0, y)
		// x = (x1*B2 + x0)
		// y = (y1*B2 + y0)
		x1, x0 := x>>W2, x&M2;

		// x*y = t2*B2*B2 + t1*B2 + t0
		t0 := x0 * y;
		t1 := x1 * y;

		// compute result digits but avoid overflow
		// z = z[1]*B + z[0] = x*y
		z0 = t1<<W2 + t0;
		z1 = (t1 + t0>>W2) >> W2;
		return;
	}

	// general case
	// sub-digits of x and y are (x1, x0) and (y1, y0)
	// x = (x1*B2 + x0)
	// y = (y1*B2 + y0)
	x1, x0 := x>>W2, x&M2;
	y1, y0 := y>>W2, y&M2;

	// x*y = t2*B2*B2 + t1*B2 + t0
	t0 := x0 * y0;
	t1 := x1*y0 + x0*y1;
	t2 := x1 * y1;

	// compute result digits but avoid overflow
	// z = z[1]*B + z[0] = x*y
	z0 = t1<<W2 + t0;
	z1 = t2 + (t1+t0>>W2)>>W2;
	return;
}


// z1<<64 + z0 = x*y + c
func MulAdd128(x, y, c uint64) (z1, z0 uint64) {
	// Split x and y into 2 halfwords each, multiply
	// the halfwords separately while avoiding overflow,
	// and return the product as 2 words.

	const (
		W	= uint(unsafe.Sizeof(x)) * 8;
		W2	= W / 2;
		B2	= 1 << W2;
		M2	= B2 - 1;
	)

	// TODO(gri) Should implement special cases for faster execution.

	// general case
	// sub-digits of x, y, and c are (x1, x0), (y1, y0), (c1, c0)
	// x = (x1*B2 + x0)
	// y = (y1*B2 + y0)
	x1, x0 := x>>W2, x&M2;
	y1, y0 := y>>W2, y&M2;
	c1, c0 := c>>W2, c&M2;

	// x*y + c = t2*B2*B2 + t1*B2 + t0
	t0 := x0*y0 + c0;
	t1 := x1*y0 + x0*y1 + c1;
	t2 := x1 * y1;

	// compute result digits but avoid overflow
	// z = z[1]*B + z[0] = x*y
	z0 = t1<<W2 + t0;
	z1 = t2 + (t1+t0>>W2)>>W2;
	return;
}


// q = (x1<<64 + x0)/y + r
func Div128(x1, x0, y uint64) (q, r uint64) {
	if x1 == 0 {
		q, r = x0/y, x0%y;
		return;
	}

	// TODO(gri) Implement general case.
	panic("Div128 not implemented for x > 1<<64-1");
}
