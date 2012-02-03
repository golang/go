// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements multi-precision rational numbers.

package big

import (
	"encoding/binary"
	"errors"
	"fmt"
	"strings"
)

// A Rat represents a quotient a/b of arbitrary precision.
// The zero value for a Rat represents the value 0.
type Rat struct {
	a Int
	b nat // len(b) == 0 acts like b == 1
}

// NewRat creates a new Rat with numerator a and denominator b.
func NewRat(a, b int64) *Rat {
	return new(Rat).SetFrac64(a, b)
}

// SetFrac sets z to a/b and returns z.
func (z *Rat) SetFrac(a, b *Int) *Rat {
	z.a.neg = a.neg != b.neg
	babs := b.abs
	if len(babs) == 0 {
		panic("division by zero")
	}
	if &z.a == b || alias(z.a.abs, babs) {
		babs = nat(nil).set(babs) // make a copy
	}
	z.a.abs = z.a.abs.set(a.abs)
	z.b = z.b.set(babs)
	return z.norm()
}

// SetFrac64 sets z to a/b and returns z.
func (z *Rat) SetFrac64(a, b int64) *Rat {
	z.a.SetInt64(a)
	if b == 0 {
		panic("division by zero")
	}
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
	z.b = z.b.make(0)
	return z
}

// SetInt64 sets z to x and returns z.
func (z *Rat) SetInt64(x int64) *Rat {
	z.a.SetInt64(x)
	z.b = z.b.make(0)
	return z
}

// Set sets z to x (by making a copy of x) and returns z.
func (z *Rat) Set(x *Rat) *Rat {
	if z != x {
		z.a.Set(&x.a)
		z.b = z.b.set(x.b)
	}
	return z
}

// Abs sets z to |x| (the absolute value of x) and returns z.
func (z *Rat) Abs(x *Rat) *Rat {
	z.Set(x)
	z.a.neg = false
	return z
}

// Neg sets z to -x and returns z.
func (z *Rat) Neg(x *Rat) *Rat {
	z.Set(x)
	z.a.neg = len(z.a.abs) > 0 && !z.a.neg // 0 has no sign
	return z
}

// Inv sets z to 1/x and returns z.
func (z *Rat) Inv(x *Rat) *Rat {
	if len(x.a.abs) == 0 {
		panic("division by zero")
	}
	z.Set(x)
	a := z.b
	if len(a) == 0 {
		a = a.setWord(1) // materialize numerator
	}
	b := z.a.abs
	if b.cmp(natOne) == 0 {
		b = b.make(0) // normalize denominator
	}
	z.a.abs, z.b = a, b // sign doesn't change
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
	return len(x.b) == 0 || x.b.cmp(natOne) == 0
}

// Num returns the numerator of x; it may be <= 0.
// The result is a reference to x's numerator; it
// may change if a new value is assigned to x.
func (x *Rat) Num() *Int {
	return &x.a
}

// Denom returns the denominator of x; it is always > 0.
// The result is a reference to x's denominator; it
// may change if a new value is assigned to x.
func (x *Rat) Denom() *Int {
	if len(x.b) == 0 {
		return &Int{abs: nat{1}}
	}
	return &Int{abs: x.b}
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
	switch {
	case len(z.a.abs) == 0:
		// z == 0 - normalize sign and denominator
		z.a.neg = false
		z.b = z.b.make(0)
	case len(z.b) == 0:
		// z is normalized int - nothing to do
	case z.b.cmp(natOne) == 0:
		// z is int - normalize denominator
		z.b = z.b.make(0)
	default:
		if f := gcd(z.a.abs, z.b); f.cmp(natOne) != 0 {
			z.a.abs, _ = z.a.abs.div(nil, z.a.abs, f)
			z.b, _ = z.b.div(nil, z.b, f)
		}
	}
	return z
}

// mulDenom sets z to the denominator product x*y (by taking into
// account that 0 values for x or y must be interpreted as 1) and
// returns z.
func mulDenom(z, x, y nat) nat {
	switch {
	case len(x) == 0:
		return z.set(y)
	case len(y) == 0:
		return z.set(x)
	}
	return z.mul(x, y)
}

// scaleDenom computes x*f.
// If f == 0 (zero value of denominator), the result is (a copy of) x.
func scaleDenom(x *Int, f nat) *Int {
	var z Int
	if len(f) == 0 {
		return z.Set(x)
	}
	z.abs = z.abs.mul(x.abs, f)
	z.neg = x.neg
	return &z
}

// Cmp compares x and y and returns:
//
//   -1 if x <  y
//    0 if x == y
//   +1 if x >  y
//
func (x *Rat) Cmp(y *Rat) int {
	return scaleDenom(&x.a, y.b).Cmp(scaleDenom(&y.a, x.b))
}

// Add sets z to the sum x+y and returns z.
func (z *Rat) Add(x, y *Rat) *Rat {
	a1 := scaleDenom(&x.a, y.b)
	a2 := scaleDenom(&y.a, x.b)
	z.a.Add(a1, a2)
	z.b = mulDenom(z.b, x.b, y.b)
	return z.norm()
}

// Sub sets z to the difference x-y and returns z.
func (z *Rat) Sub(x, y *Rat) *Rat {
	a1 := scaleDenom(&x.a, y.b)
	a2 := scaleDenom(&y.a, x.b)
	z.a.Sub(a1, a2)
	z.b = mulDenom(z.b, x.b, y.b)
	return z.norm()
}

// Mul sets z to the product x*y and returns z.
func (z *Rat) Mul(x, y *Rat) *Rat {
	z.a.Mul(&x.a, &y.a)
	z.b = mulDenom(z.b, x.b, y.b)
	return z.norm()
}

// Quo sets z to the quotient x/y and returns z.
// If y == 0, a division-by-zero run-time panic occurs.
func (z *Rat) Quo(x, y *Rat) *Rat {
	if len(y.a.abs) == 0 {
		panic("division by zero")
	}
	a := scaleDenom(&x.a, y.b)
	b := scaleDenom(&y.a, x.b)
	z.a.abs = a.abs
	z.b = b.abs
	z.a.neg = a.neg != b.neg
	return z.norm()
}

func ratTok(ch rune) bool {
	return strings.IndexRune("+-/0123456789.eE", ch) >= 0
}

// Scan is a support routine for fmt.Scanner. It accepts the formats
// 'e', 'E', 'f', 'F', 'g', 'G', and 'v'. All formats are equivalent.
func (z *Rat) Scan(s fmt.ScanState, ch rune) error {
	tok, err := s.Token(true, ratTok)
	if err != nil {
		return err
	}
	if strings.IndexRune("efgEFGv", ch) < 0 {
		return errors.New("Rat.Scan: invalid verb")
	}
	if _, ok := z.SetString(string(tok)); !ok {
		return errors.New("Rat.Scan: invalid syntax")
	}
	return nil
}

// SetString sets z to the value of s and returns z and a boolean indicating
// success. s can be given as a fraction "a/b" or as a floating-point number
// optionally followed by an exponent. If the operation failed, the value of
// z is undefined but the returned value is nil.
func (z *Rat) SetString(s string) (*Rat, bool) {
	if len(s) == 0 {
		return nil, false
	}

	// check for a quotient
	sep := strings.Index(s, "/")
	if sep >= 0 {
		if _, ok := z.a.SetString(s[0:sep], 10); !ok {
			return nil, false
		}
		s = s[sep+1:]
		var err error
		if z.b, _, err = z.b.scan(strings.NewReader(s), 10); err != nil {
			return nil, false
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
			return nil, false
		}
		if _, ok := exp.SetString(s[e+1:], 10); !ok {
			return nil, false
		}
		s = s[0:e]
	}
	if sep >= 0 {
		s = s[0:sep] + s[sep+1:]
		exp.Sub(&exp, NewInt(int64(len(s)-sep)))
	}

	if _, ok := z.a.SetString(s, 10); !ok {
		return nil, false
	}
	powTen := nat(nil).expNN(natTen, exp.abs, nil)
	if exp.neg {
		z.b = powTen
		z.norm()
	} else {
		z.a.abs = z.a.abs.mul(z.a.abs, powTen)
		z.b = z.b.make(0)
	}

	return z, true
}

// String returns a string representation of z in the form "a/b" (even if b == 1).
func (x *Rat) String() string {
	s := "/1"
	if len(x.b) != 0 {
		s = "/" + x.b.decimalString()
	}
	return x.a.String() + s
}

// RatString returns a string representation of z in the form "a/b" if b != 1,
// and in the form "a" if b == 1.
func (x *Rat) RatString() string {
	if x.IsInt() {
		return x.a.String()
	}
	return x.String()
}

// FloatString returns a string representation of z in decimal form with prec
// digits of precision after the decimal point and the last digit rounded.
func (x *Rat) FloatString(prec int) string {
	if x.IsInt() {
		s := x.a.String()
		if prec > 0 {
			s += "." + strings.Repeat("0", prec)
		}
		return s
	}
	// x.b != 0

	q, r := nat(nil).div(nat(nil), x.a.abs, x.b)

	p := natOne
	if prec > 0 {
		p = nat(nil).expNN(natTen, nat(nil).setUint64(uint64(prec)), nil)
	}

	r = r.mul(r, p)
	r, r2 := r.div(nat(nil), r, x.b)

	// see if we need to round up
	r2 = r2.add(r2, r2)
	if x.b.cmp(r2) <= 0 {
		r = r.add(r, natOne)
		if r.cmp(p) >= 0 {
			q = nat(nil).add(q, natOne)
			r = nat(nil).sub(r, p)
		}
	}

	s := q.decimalString()
	if x.a.neg {
		s = "-" + s
	}

	if prec > 0 {
		rs := r.decimalString()
		leadingZeros := prec - len(rs)
		s += "." + strings.Repeat("0", leadingZeros) + rs
	}

	return s
}

// Gob codec version. Permits backward-compatible changes to the encoding.
const ratGobVersion byte = 1

// GobEncode implements the gob.GobEncoder interface.
func (x *Rat) GobEncode() ([]byte, error) {
	buf := make([]byte, 1+4+(len(x.a.abs)+len(x.b))*_S) // extra bytes for version and sign bit (1), and numerator length (4)
	i := x.b.bytes(buf)
	j := x.a.abs.bytes(buf[0:i])
	n := i - j
	if int(uint32(n)) != n {
		// this should never happen
		return nil, errors.New("Rat.GobEncode: numerator too large")
	}
	binary.BigEndian.PutUint32(buf[j-4:j], uint32(n))
	j -= 1 + 4
	b := ratGobVersion << 1 // make space for sign bit
	if x.a.neg {
		b |= 1
	}
	buf[j] = b
	return buf[j:], nil
}

// GobDecode implements the gob.GobDecoder interface.
func (z *Rat) GobDecode(buf []byte) error {
	if len(buf) == 0 {
		return errors.New("Rat.GobDecode: no data")
	}
	b := buf[0]
	if b>>1 != ratGobVersion {
		return errors.New(fmt.Sprintf("Rat.GobDecode: encoding version %d not supported", b>>1))
	}
	const j = 1 + 4
	i := j + binary.BigEndian.Uint32(buf[j-4:j])
	z.a.neg = b&1 != 0
	z.a.abs = z.a.abs.setBytes(buf[j:i])
	z.b = z.b.setBytes(buf[i:])
	return nil
}
