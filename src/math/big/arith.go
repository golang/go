// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides Go implementations of elementary multi-precision
// arithmetic operations on word vectors. Needed for platforms without
// assembly implementations of these routines.

package big

import "math/bits"

// A Word represents a single digit of a multi-precision unsigned integer.
type Word uint

const (
	_S = _W / 8 // word size in bytes

	_W = bits.UintSize // word size in bits
	_B = 1 << _W       // digit base
	_M = _B - 1        // digit mask
)

// ----------------------------------------------------------------------------
// Elementary operations on words
//
// These operations are used by the vector operations below.

// z1<<_W + z0 = x*y
func mulWW_g(x, y Word) (z1, z0 Word) {
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

// q = (u1<<_W + u0 - r)/v
func divWW_g(u1, u0, v Word) (q, r Word) {
	qq, rr := bits.Div(uint(u1), uint(u0), uint(v))
	return Word(qq), Word(rr)
}

// The resulting carry c is either 0 or 1.
func addVV_g(z, x, y []Word) (c Word) {
	for i := range x[:len(z)] {
		zi, cc := bits.Add(uint(x[i]), uint(y[i]), uint(c))
		z[i] = Word(zi)
		c = Word(cc)
	}
	return
}

// The resulting carry c is either 0 or 1.
func subVV_g(z, x, y []Word) (c Word) {
	for i := range x[:len(z)] {
		zi, cc := bits.Sub(uint(x[i]), uint(y[i]), uint(c))
		z[i] = Word(zi)
		c = Word(cc)
	}
	return
}

// The resulting carry c is either 0 or 1.
func addVW_g(z, x []Word, y Word) (c Word) {
	c = y
	for i := range x[:len(z)] {
		zi, cc := bits.Add(uint(x[i]), uint(c), 0)
		z[i] = Word(zi)
		c = Word(cc)
	}
	return
}

func subVW_g(z, x []Word, y Word) (c Word) {
	c = y
	for i := range x[:len(z)] {
		zi, cc := bits.Sub(uint(x[i]), uint(c), 0)
		z[i] = Word(zi)
		c = Word(cc)
	}
	return
}

func shlVU_g(z, x []Word, s uint) (c Word) {
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
	w1 := x[len(z)-1]
	c = w1 >> ŝ
	for i := len(z) - 1; i > 0; i-- {
		w := w1
		w1 = x[i-1]
		z[i] = w<<s | w1>>ŝ
	}
	z[0] = w1 << s
	return
}

func shrVU_g(z, x []Word, s uint) (c Word) {
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
	w1 := x[0]
	c = w1 << ŝ
	for i := 0; i < len(z)-1; i++ {
		w := w1
		w1 = x[i+1]
		z[i] = w>>s | w1<<ŝ
	}
	z[len(z)-1] = w1 >> s
	return
}

func mulAddVWW_g(z, x []Word, y, r Word) (c Word) {
	c = r
	for i := range z {
		c, z[i] = mulAddWWW_g(x[i], y, c)
	}
	return
}

func addMulVVW_g(z, x []Word, y Word) (c Word) {
	for i := range z {
		z1, z0 := mulAddWWW_g(x[i], y, z[i])
		lo, cc := bits.Add(uint(z0), uint(c), 0)
		c, z[i] = Word(cc), Word(lo)
		c += z1
	}
	return
}

func divWVW_g(z []Word, xn Word, x []Word, y Word) (r Word) {
	r = xn
	for i := len(z) - 1; i >= 0; i-- {
		z[i], r = divWW_g(r, x[i], y)
	}
	return
}
