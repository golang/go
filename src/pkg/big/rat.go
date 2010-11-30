// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements multi-precision rational numbers.

package big

import "strings"

// A Rat represents a quotient a/b of arbitrary precision. The zero value for
// a Rat, 0/0, is not a legal Rat.
type Rat struct {
	a Int
	b nat
}


// NewRat creates a new Rat with numerator a and denominator b.
func NewRat(a, b int64) *Rat {
	return new(Rat).SetFrac64(a, b)
}


// SetFrac sets z to a/b and returns z.
func (z *Rat) SetFrac(a, b *Int) *Rat {
	z.a.Set(a)
	z.a.neg = a.neg != b.neg
	z.b = z.b.set(b.abs)
	return z.norm()
}


// SetFrac64 sets z to a/b and returns z.
func (z *Rat) SetFrac64(a, b int64) *Rat {
	z.a.SetInt64(a)
	if b < 0 {
		b = -b
		z.a.neg = !z.a.neg
	}
	z.b = z.b.setUint64(uint64(b))
	return z.norm()
}


// SetInt sets z to x (by making a copy of x) and returns z.
func (z *Rat) SetInt(x *Int) *Rat {
	z.a.Set(x)
	z.b = z.b.setWord(1)
	return z
}


// SetInt64 sets z to x and returns z.
func (z *Rat) SetInt64(x int64) *Rat {
	z.a.SetInt64(x)
	z.b = z.b.setWord(1)
	return z
}


// Sign returns:
//
//	-1 if x <  0
//	 0 if x == 0
//	+1 if x >  0
//
func (x *Rat) Sign() int {
	return x.a.Sign()
}


// IsInt returns true if the denominator of x is 1.
func (x *Rat) IsInt() bool {
	return len(x.b) == 1 && x.b[0] == 1
}


// Num returns the numerator of z; it may be <= 0.
// The result is a reference to z's numerator; it
// may change if a new value is assigned to z.
func (z *Rat) Num() *Int {
	return &z.a
}


// Demom returns the denominator of z; it is always > 0.
// The result is a reference to z's denominator; it
// may change if a new value is assigned to z.
func (z *Rat) Denom() *Int {
	return &Int{false, z.b}
}


func gcd(x, y nat) nat {
	// Euclidean algorithm.
	var a, b nat
	a = a.set(x)
	b = b.set(y)
	for len(b) != 0 {
		var q, r nat
		_, r = q.div(r, a, b)
		a = b
		b = r
	}
	return a
}


func (z *Rat) norm() *Rat {
	f := gcd(z.a.abs, z.b)
	if len(z.a.abs) == 0 {
		// z == 0
		z.a.neg = false // normalize sign
		z.b = z.b.setWord(1)
		return z
	}
	if f.cmp(natOne) != 0 {
		z.a.abs, _ = z.a.abs.div(nil, z.a.abs, f)
		z.b, _ = z.b.div(nil, z.b, f)
	}
	return z
}


func mulNat(x *Int, y nat) *Int {
	var z Int
	z.abs = z.abs.mul(x.abs, y)
	z.neg = len(z.abs) > 0 && x.neg
	return &z
}


// Cmp compares x and y and returns:
//
//   -1 if x <  y
//    0 if x == y
//   +1 if x >  y
//
func (x *Rat) Cmp(y *Rat) (r int) {
	return mulNat(&x.a, y.b).Cmp(mulNat(&y.a, x.b))
}


// Abs sets z to |x| (the absolute value of x) and returns z.
func (z *Rat) Abs(x *Rat) *Rat {
	z.a.Abs(&x.a)
	z.b = z.b.set(x.b)
	return z
}


// Add sets z to the sum x+y and returns z.
func (z *Rat) Add(x, y *Rat) *Rat {
	a1 := mulNat(&x.a, y.b)
	a2 := mulNat(&y.a, x.b)
	z.a.Add(a1, a2)
	z.b = z.b.mul(x.b, y.b)
	return z.norm()
}


// Sub sets z to the difference x-y and returns z.
func (z *Rat) Sub(x, y *Rat) *Rat {
	a1 := mulNat(&x.a, y.b)
	a2 := mulNat(&y.a, x.b)
	z.a.Sub(a1, a2)
	z.b = z.b.mul(x.b, y.b)
	return z.norm()
}


// Mul sets z to the product x*y and returns z.
func (z *Rat) Mul(x, y *Rat) *Rat {
	z.a.Mul(&x.a, &y.a)
	z.b = z.b.mul(x.b, y.b)
	return z.norm()
}


// Quo sets z to the quotient x/y and returns z.
// If y == 0, a division-by-zero run-time panic occurs.
func (z *Rat) Quo(x, y *Rat) *Rat {
	if len(y.a.abs) == 0 {
		panic("division by zero")
	}
	a := mulNat(&x.a, y.b)
	b := mulNat(&y.a, x.b)
	z.a.abs = a.abs
	z.b = b.abs
	z.a.neg = a.neg != b.neg
	return z.norm()
}


// Neg sets z to -x (by making a copy of x if necessary) and returns z.
func (z *Rat) Neg(x *Rat) *Rat {
	z.a.Neg(&x.a)
	z.b = z.b.set(x.b)
	return z
}


// Set sets z to x (by making a copy of x if necessary) and returns z.
func (z *Rat) Set(x *Rat) *Rat {
	z.a.Set(&x.a)
	z.b = z.b.set(x.b)
	return z
}


// SetString sets z to the value of s and returns z and a boolean indicating
// success. s can be given as a fraction "a/b" or as a floating-point number
// optionally followed by an exponent. If the operation failed, the value of z
// is undefined.
func (z *Rat) SetString(s string) (*Rat, bool) {
	if len(s) == 0 {
		return z, false
	}

	// check for a quotient
	sep := strings.Index(s, "/")
	if sep >= 0 {
		if _, ok := z.a.SetString(s[0:sep], 10); !ok {
			return z, false
		}
		s = s[sep+1:]
		var n int
		if z.b, _, n = z.b.scan(s, 10); n != len(s) {
			return z, false
		}
		return z.norm(), true
	}

	// check for a decimal point
	sep = strings.Index(s, ".")
	// check for an exponent
	e := strings.IndexAny(s, "eE")
	var exp Int
	if e >= 0 {
		if e < sep {
			// The E must come after the decimal point.
			return z, false
		}
		if _, ok := exp.SetString(s[e+1:], 10); !ok {
			return z, false
		}
		s = s[0:e]
	}
	if sep >= 0 {
		s = s[0:sep] + s[sep+1:]
		exp.Sub(&exp, NewInt(int64(len(s)-sep)))
	}

	if _, ok := z.a.SetString(s, 10); !ok {
		return z, false
	}
	powTen := nat{}.expNN(natTen, exp.abs, nil)
	if exp.neg {
		z.b = powTen
		z.norm()
	} else {
		z.a.abs = z.a.abs.mul(z.a.abs, powTen)
		z.b = z.b.setWord(1)
	}

	return z, true
}


// String returns a string representation of z in the form "a/b" (even if b == 1).
func (z *Rat) String() string {
	return z.a.String() + "/" + z.b.string(10)
}


// RatString returns a string representation of z in the form "a/b" if b != 1,
// and in the form "a" if b == 1.
func (z *Rat) RatString() string {
	if z.IsInt() {
		return z.a.String()
	}
	return z.String()
}


// FloatString returns a string representation of z in decimal form with prec
// digits of precision after the decimal point and the last digit rounded.
func (z *Rat) FloatString(prec int) string {
	if z.IsInt() {
		return z.a.String()
	}

	q, r := nat{}.div(nat{}, z.a.abs, z.b)

	p := natOne
	if prec > 0 {
		p = nat{}.expNN(natTen, nat{}.setUint64(uint64(prec)), nil)
	}

	r = r.mul(r, p)
	r, r2 := r.div(nat{}, r, z.b)

	// see if we need to round up
	r2 = r2.add(r2, r2)
	if z.b.cmp(r2) <= 0 {
		r = r.add(r, natOne)
		if r.cmp(p) >= 0 {
			q = nat{}.add(q, natOne)
			r = nat{}.sub(r, p)
		}
	}

	s := q.string(10)
	if z.a.neg {
		s = "-" + s
	}

	if prec > 0 {
		rs := r.string(10)
		leadingZeros := prec - len(rs)
		s += "." + strings.Repeat("0", leadingZeros) + rs
	}

	return s
}
