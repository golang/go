// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides Go implementations of elementary multi-precision
// arithmetic operations on word vectors. These have the suffix _g.
// These are needed for platforms without assembly implementations of these routines.
// This file also contains elementary operations that can be implemented
// sufficiently efficiently in Go.

package big

import (
	"math/bits"
	_ "unsafe" // for go:linkname
)

// A Word represents a single digit of a multi-precision unsigned integer.
type Word uint

const (
	_S = _W / 8 // word size in bytes

	_W = bits.UintSize // word size in bits
	_B = 1 << _W       // digit base
	_M = _B - 1        // digit mask
)

// Many of the loops in this file are of the form
//   for i := 0; i < len(z) && i < len(x) && i < len(y); i++
// i < len(z) is the real condition.
// However, checking i < len(x) && i < len(y) as well is faster than
// having the compiler do a bounds check in the body of the loop;
// remarkably it is even faster than hoisting the bounds check
// out of the loop, by doing something like
//   _, _ = x[len(z)-1], y[len(z)-1]
// There are other ways to hoist the bounds check out of the loop,
// but the compiler's BCE isn't powerful enough for them (yet?).
// See the discussion in CL 164966.

// ----------------------------------------------------------------------------
// Elementary operations on words
//
// These operations are used by the vector operations below.

// z1<<_W + z0 = x*y
func mulWW(x, y Word) (z1, z0 Word) {
	hi, lo := bits.Mul(uint(x), uint(y))
	return Word(hi), Word(lo)
}

// z1<<_W + z0 = x*y + c
func mulAddWWW_g(x, y, c Word) (z1, z0 Word) {
	hi, lo := bits.Mul(uint(x), uint(y))
	var cc uint
	lo, cc = bits.Add(lo, uint(c), 0)
	return Word(hi + cc), Word(lo)
}

// nlz returns the number of leading zeros in x.
// Wraps bits.LeadingZeros call for convenience.
func nlz(x Word) uint {
	return uint(bits.LeadingZeros(uint(x)))
}

// The resulting carry c is either 0 or 1.
func addVV_g(z, x, y []Word) (c Word) {
	// The comment near the top of this file discusses this for loop condition.
	for i := 0; i < len(z) && i < len(x) && i < len(y); i++ {
		zi, cc := bits.Add(uint(x[i]), uint(y[i]), uint(c))
		z[i] = Word(zi)
		c = Word(cc)
	}
	return
}

// The resulting carry c is either 0 or 1.
func subVV_g(z, x, y []Word) (c Word) {
	// The comment near the top of this file discusses this for loop condition.
	for i := 0; i < len(z) && i < len(x) && i < len(y); i++ {
		zi, cc := bits.Sub(uint(x[i]), uint(y[i]), uint(c))
		z[i] = Word(zi)
		c = Word(cc)
	}
	return
}

// addVW sets z = x + y, returning the final carry c.
// The behavior is undefined if len(x) != len(z).
// If len(z) == 0, c = y; otherwise, c is 0 or 1.
//
// addVW should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/remyoudompheng/bigfft
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname addVW
func addVW(z, x []Word, y Word) (c Word) {
	x = x[:len(z)]
	if len(z) == 0 {
		return y
	}
	zi, cc := bits.Add(uint(x[0]), uint(y), 0)
	z[0] = Word(zi)
	if cc == 0 {
		if &z[0] != &x[0] {
			copy(z[1:], x[1:])
		}
		return 0
	}
	for i := 1; i < len(z); i++ {
		xi := x[i]
		if xi != ^Word(0) {
			z[i] = xi + 1
			if &z[0] != &x[0] {
				copy(z[i+1:], x[i+1:])
			}
			return 0
		}
		z[i] = 0
	}
	return 1
}

// addVW_ref is the reference implementation for addVW, used only for testing.
func addVW_ref(z, x []Word, y Word) (c Word) {
	c = y
	for i := range z {
		zi, cc := bits.Add(uint(x[i]), uint(c), 0)
		z[i] = Word(zi)
		c = Word(cc)
	}
	return
}

// subVW sets z = x - y, returning the final carry c.
// The behavior is undefined if len(x) != len(z).
// If len(z) == 0, c = y; otherwise, c is 0 or 1.
//
// subVW should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/remyoudompheng/bigfft
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname subVW
func subVW(z, x []Word, y Word) (c Word) {
	x = x[:len(z)]
	if len(z) == 0 {
		return y
	}
	zi, cc := bits.Sub(uint(x[0]), uint(y), 0)
	z[0] = Word(zi)
	if cc == 0 {
		if &z[0] != &x[0] {
			copy(z[1:], x[1:])
		}
		return 0
	}
	for i := 1; i < len(z); i++ {
		xi := x[i]
		if xi != 0 {
			z[i] = xi - 1
			if &z[0] != &x[0] {
				copy(z[i+1:], x[i+1:])
			}
			return 0
		}
		z[i] = ^Word(0)
	}
	return 1
}

// subVW_ref is the reference implementation for subVW, used only for testing.
func subVW_ref(z, x []Word, y Word) (c Word) {
	c = y
	for i := range z {
		zi, cc := bits.Sub(uint(x[i]), uint(c), 0)
		z[i] = Word(zi)
		c = Word(cc)
	}
	return c
}

func lshVU_g(z, x []Word, s uint) (c Word) {
	if s == 0 {
		copy(z, x)
		return
	}
	if len(z) == 0 {
		return
	}
	s &= _W - 1 // hint to the compiler that shifts by s don't need guard code
	ŝ := _W - s
	ŝ &= _W - 1 // ditto
	c = x[len(z)-1] >> ŝ
	for i := len(z) - 1; i > 0; i-- {
		z[i] = x[i]<<s | x[i-1]>>ŝ
	}
	z[0] = x[0] << s
	return
}

func rshVU_g(z, x []Word, s uint) (c Word) {
	if s == 0 {
		copy(z, x)
		return
	}
	if len(z) == 0 {
		return
	}
	if len(x) != len(z) {
		// This is an invariant guaranteed by the caller.
		panic("len(x) != len(z)")
	}
	s &= _W - 1 // hint to the compiler that shifts by s don't need guard code
	ŝ := _W - s
	ŝ &= _W - 1 // ditto
	c = x[0] << ŝ
	for i := 1; i < len(z); i++ {
		z[i-1] = x[i-1]>>s | x[i]<<ŝ
	}
	z[len(z)-1] = x[len(z)-1] >> s
	return
}

func mulAddVWW_g(z, x []Word, y, r Word) (c Word) {
	c = r
	// The comment near the top of this file discusses this for loop condition.
	for i := 0; i < len(z) && i < len(x); i++ {
		c, z[i] = mulAddWWW_g(x[i], y, c)
	}
	return
}

func addMulVVWW_g(z, x, y []Word, m, a Word) (c Word) {
	c = a
	// The comment near the top of this file discusses this for loop condition.
	for i := 0; i < len(z) && i < len(x) && i < len(y); i++ {
		z1, z0 := mulAddWWW_g(y[i], m, x[i])
		lo, cc := bits.Add(uint(z0), uint(c), 0)
		c, z[i] = Word(cc), Word(lo)
		c += z1
	}
	return
}

// q = ( x1 << _W + x0 - r)/y. m = floor(( _B^2 - 1 ) / d - _B). Requiring x1<y.
// An approximate reciprocal with a reference to "Improved Division by Invariant Integers
// (IEEE Transactions on Computers, 11 Jun. 2010)"
func divWW(x1, x0, y, m Word) (q, r Word) {
	s := nlz(y)
	if s != 0 {
		x1 = x1<<s | x0>>(_W-s)
		x0 <<= s
		y <<= s
	}
	d := uint(y)
	// We know that
	//   m = ⎣(B^2-1)/d⎦-B
	//   ⎣(B^2-1)/d⎦ = m+B
	//   (B^2-1)/d = m+B+delta1    0 <= delta1 <= (d-1)/d
	//   B^2/d = m+B+delta2        0 <= delta2 <= 1
	// The quotient we're trying to compute is
	//   quotient = ⎣(x1*B+x0)/d⎦
	//            = ⎣(x1*B*(B^2/d)+x0*(B^2/d))/B^2⎦
	//            = ⎣(x1*B*(m+B+delta2)+x0*(m+B+delta2))/B^2⎦
	//            = ⎣(x1*m+x1*B+x0)/B + x0*m/B^2 + delta2*(x1*B+x0)/B^2⎦
	// The latter two terms of this three-term sum are between 0 and 1.
	// So we can compute just the first term, and we will be low by at most 2.
	t1, t0 := bits.Mul(uint(m), uint(x1))
	_, c := bits.Add(t0, uint(x0), 0)
	t1, _ = bits.Add(t1, uint(x1), c)
	// The quotient is either t1, t1+1, or t1+2.
	// We'll try t1 and adjust if needed.
	qq := t1
	// compute remainder r=x-d*q.
	dq1, dq0 := bits.Mul(d, qq)
	r0, b := bits.Sub(uint(x0), dq0, 0)
	r1, _ := bits.Sub(uint(x1), dq1, b)
	// The remainder we just computed is bounded above by B+d:
	// r = x1*B + x0 - d*q.
	//   = x1*B + x0 - d*⎣(x1*m+x1*B+x0)/B⎦
	//   = x1*B + x0 - d*((x1*m+x1*B+x0)/B-alpha)                                   0 <= alpha < 1
	//   = x1*B + x0 - x1*d/B*m                         - x1*d - x0*d/B + d*alpha
	//   = x1*B + x0 - x1*d/B*⎣(B^2-1)/d-B⎦             - x1*d - x0*d/B + d*alpha
	//   = x1*B + x0 - x1*d/B*⎣(B^2-1)/d-B⎦             - x1*d - x0*d/B + d*alpha
	//   = x1*B + x0 - x1*d/B*((B^2-1)/d-B-beta)        - x1*d - x0*d/B + d*alpha   0 <= beta < 1
	//   = x1*B + x0 - x1*B + x1/B + x1*d + x1*d/B*beta - x1*d - x0*d/B + d*alpha
	//   =        x0        + x1/B        + x1*d/B*beta        - x0*d/B + d*alpha
	//   = x0*(1-d/B) + x1*(1+d*beta)/B + d*alpha
	//   <  B*(1-d/B) +  d*B/B          + d          because x0<B (and 1-d/B>0), x1<d, 1+d*beta<=B, alpha<1
	//   =  B - d     +  d              + d
	//   = B+d
	// So r1 can only be 0 or 1. If r1 is 1, then we know q was too small.
	// Add 1 to q and subtract d from r. That guarantees that r is <B, so
	// we no longer need to keep track of r1.
	if r1 != 0 {
		qq++
		r0 -= d
	}
	// If the remainder is still too large, increment q one more time.
	if r0 >= d {
		qq++
		r0 -= d
	}
	return Word(qq), Word(r0 >> s)
}

// reciprocalWord return the reciprocal of the divisor. rec = floor(( _B^2 - 1 ) / u - _B). u = d1 << nlz(d1).
func reciprocalWord(d1 Word) Word {
	u := uint(d1 << nlz(d1))
	x1 := ^u
	x0 := uint(_M)
	rec, _ := bits.Div(x1, x0, u) // (_B^2-1)/U-_B = (_B*(_M-C)+_M)/U
	return Word(rec)
}
