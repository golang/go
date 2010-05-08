// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides Go implementations of elementary multi-precision
// arithmetic operations on word vectors. Needed for platforms without
// assembly implementations of these routines.

package big

// TODO(gri) Decide if Word needs to remain exported.

type Word uintptr

const (
	// Compute the size _S of a Word in bytes.
	_m    = ^Word(0)
	_logS = _m>>8&1 + _m>>16&1 + _m>>32&1
	_S    = 1 << _logS

	_W = _S << 3 // word size in bits
	_B = 1 << _W // digit base
	_M = _B - 1  // digit mask

	_W2 = _W / 2   // half word size in bits
	_B2 = 1 << _W2 // half digit base
	_M2 = _B2 - 1  // half digit mask
)


// ----------------------------------------------------------------------------
// Elementary operations on words
//
// These operations are used by the vector operations below.

// z1<<_W + z0 = x+y+c, with c == 0 or 1
func addWW_g(x, y, c Word) (z1, z0 Word) {
	yc := y + c
	z0 = x + yc
	if z0 < x || yc < y {
		z1 = 1
	}
	return
}


// z1<<_W + z0 = x-y-c, with c == 0 or 1
func subWW_g(x, y, c Word) (z1, z0 Word) {
	yc := y + c
	z0 = x - yc
	if z0 > x || yc < y {
		z1 = 1
	}
	return
}


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
		z0 = x * y
		return
	}

	if y < _B2 {
		// sub-digits of x and y are (x1, x0) and (0, y)
		// x = (x1*_B2 + x0)
		// y = (y1*_B2 + y0)
		x1, x0 := x>>_W2, x&_M2

		// x*y = t2*_B2*_B2 + t1*_B2 + t0
		t0 := x0 * y
		t1 := x1 * y

		// compute result digits but avoid overflow
		// z = z[1]*_B + z[0] = x*y
		z0 = t1<<_W2 + t0
		z1 = (t1 + t0>>_W2) >> _W2
		return
	}

	// general case
	// sub-digits of x and y are (x1, x0) and (y1, y0)
	// x = (x1*_B2 + x0)
	// y = (y1*_B2 + y0)
	x1, x0 := x>>_W2, x&_M2
	y1, y0 := y>>_W2, y&_M2

	// x*y = t2*_B2*_B2 + t1*_B2 + t0
	t0 := x0 * y0
	// t1 := x1*y0 + x0*y1;
	var c Word
	t1 := x1 * y0
	t1a := t1
	t1 += x0 * y1
	if t1 < t1a {
		c++
	}
	t2 := x1*y1 + c*_B2

	// compute result digits but avoid overflow
	// z = z[1]*_B + z[0] = x*y
	// This may overflow, but that's ok because we also sum t1 and t0 above
	// and we take care of the overflow there.
	z0 = t1<<_W2 + t0

	// z1 = t2 + (t1 + t0>>_W2)>>_W2;
	var c3 Word
	z1 = t1 + t0>>_W2
	if z1 < t1 {
		c3++
	}
	z1 >>= _W2
	z1 += c3 * _B2
	z1 += t2
	return
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
	x1, x0 := x>>_W2, x&_M2
	y1, y0 := y>>_W2, y&_M2
	c1, c0 := c>>_W2, c&_M2

	// x*y + c = t2*_B2*_B2 + t1*_B2 + t0
	// (1<<32-1)^2 == 1<<64 - 1<<33 + 1, so there's space to add c0 in here.
	t0 := x0*y0 + c0

	// t1 := x1*y0 + x0*y1 + c1;
	var c2 Word // extra carry
	t1 := x1*y0 + c1
	t1a := t1
	t1 += x0 * y1
	if t1 < t1a { // If the number got smaller then we overflowed.
		c2++
	}

	t2 := x1*y1 + c2*_B2

	// compute result digits but avoid overflow
	// z = z[1]*_B + z[0] = x*y
	// z0 = t1<<_W2 + t0;
	// This may overflow, but that's ok because we also sum t1 and t0 below
	// and we take care of the overflow there.
	z0 = t1<<_W2 + t0

	var c3 Word
	z1 = t1 + t0>>_W2
	if z1 < t1 {
		c3++
	}
	z1 >>= _W2
	z1 += t2 + c3*_B2

	return
}


// q = (x1<<_W + x0 - r)/y
// The most significant bit of y must be 1.
func divStep(x1, x0, y Word) (q, r Word) {
	d1, d0 := y>>_W2, y&_M2
	q1, r1 := x1/d1, x1%d1
	m := q1 * d0
	r1 = r1*_B2 | x0>>_W2
	if r1 < m {
		q1--
		r1 += y
		if r1 >= y && r1 < m {
			q1--
			r1 += y
		}
	}
	r1 -= m

	r0 := r1 % d1
	q0 := r1 / d1
	m = q0 * d0
	r0 = r0*_B2 | x0&_M2
	if r0 < m {
		q0--
		r0 += y
		if r0 >= y && r0 < m {
			q0--
			r0 += y
		}
	}
	r0 -= m

	q = q1*_B2 | q0
	r = r0
	return
}


// Length of x in bits.
func bitLen(x Word) (n int) {
	for ; x >= 0x100; x >>= 8 {
		n += 8
	}
	for ; x > 0; x >>= 1 {
		n++
	}
	return
}


// log2 computes the integer binary logarithm of x.
// The result is the integer n for which 2^n <= x < 2^(n+1).
// If x == 0, the result is -1.
func log2(x Word) int {
	return bitLen(x) - 1
}


// Number of leading zeros in x.
func leadingZeros(x Word) uint {
	return uint(_W - bitLen(x))
}


// q = (x1<<_W + x0 - r)/y
func divWW_g(x1, x0, y Word) (q, r Word) {
	if x1 == 0 {
		q, r = x0/y, x0%y
		return
	}

	var q0, q1 Word
	z := leadingZeros(y)
	if y > x1 {
		if z != 0 {
			y <<= z
			x1 = (x1 << z) | (x0 >> (_W - z))
			x0 <<= z
		}
		q0, x0 = divStep(x1, x0, y)
		q1 = 0
	} else {
		if z == 0 {
			x1 -= y
			q1 = 1
		} else {
			z1 := _W - z
			y <<= z
			x2 := x1 >> z1
			x1 = (x1 << z) | (x0 >> z1)
			x0 <<= z
			q1, x1 = divStep(x2, x1, y)
		}

		q0, x0 = divStep(x1, x0, y)
	}

	r = x0 >> z

	if q1 != 0 {
		panic("div out of range")
	}

	return q0, r
}


func addVV(z, x, y []Word) (c Word)
func addVV_g(z, x, y []Word) (c Word) {
	for i := range z {
		c, z[i] = addWW_g(x[i], y[i], c)
	}
	return
}


func subVV(z, x, y []Word) (c Word)
func subVV_g(z, x, y []Word) (c Word) {
	for i := range z {
		c, z[i] = subWW_g(x[i], y[i], c)
	}
	return
}


func addVW(z, x []Word, y Word) (c Word)
func addVW_g(z, x []Word, y Word) (c Word) {
	c = y
	for i := range z {
		c, z[i] = addWW_g(x[i], c, 0)
	}
	return
}


func subVW(z, x []Word, y Word) (c Word)
func subVW_g(z, x []Word, y Word) (c Word) {
	c = y
	for i := range z {
		c, z[i] = subWW_g(x[i], c, 0)
	}
	return
}


func shlVW(z, x []Word, s Word) (c Word)
func shlVW_g(z, x []Word, s Word) (c Word) {
	if n := len(z); n > 0 {
		ŝ := _W - s
		w1 := x[n-1]
		c = w1 >> ŝ
		for i := n - 1; i > 0; i-- {
			w := w1
			w1 = x[i-1]
			z[i] = w<<s | w1>>ŝ
		}
		z[0] = w1 << s
	}
	return
}


func shrVW(z, x []Word, s Word) (c Word)
func shrVW_g(z, x []Word, s Word) (c Word) {
	if n := len(z); n > 0 {
		ŝ := _W - s
		w1 := x[0]
		c = w1 << ŝ
		for i := 0; i < n-1; i++ {
			w := w1
			w1 = x[i+1]
			z[i] = w>>s | w1<<ŝ
		}
		z[n-1] = w1 >> s
	}
	return
}


func mulAddVWW(z, x []Word, y, r Word) (c Word)
func mulAddVWW_g(z, x []Word, y, r Word) (c Word) {
	c = r
	for i := range z {
		c, z[i] = mulAddWWW_g(x[i], y, c)
	}
	return
}


func addMulVVW(z, x []Word, y Word) (c Word)
func addMulVVW_g(z, x []Word, y Word) (c Word) {
	for i := range z {
		z1, z0 := mulAddWWW_g(x[i], y, z[i])
		c, z[i] = addWW_g(z0, c, 0)
		c += z1
	}
	return
}


func divWVW(z []Word, xn Word, x []Word, y Word) (r Word)
func divWVW_g(z []Word, xn Word, x []Word, y Word) (r Word) {
	r = xn
	for i := len(z) - 1; i >= 0; i-- {
		z[i], r = divWW_g(r, x[i], y)
	}
	return
}
