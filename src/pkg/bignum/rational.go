// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Rational numbers

package bignum

import "fmt"


// Rational represents a quotient a/b of arbitrary precision.
//
type Rational struct {
	a	*Integer;	// numerator
	b	Natural;	// denominator
}


// MakeRat makes a rational number given a numerator a and a denominator b.
//
func MakeRat(a *Integer, b Natural) *Rational {
	f := a.mant.Gcd(b);	// f > 0
	if f.Cmp(Nat(1)) != 0 {
		a = MakeInt(a.sign, a.mant.Div(f));
		b = b.Div(f);
	}
	return &Rational{a, b};
}


// Rat creates a small rational number with value a0/b0.
//
func Rat(a0 int64, b0 int64) *Rational {
	a, b := Int(a0), Int(b0);
	if b.sign {
		a = a.Neg()
	}
	return MakeRat(a, b.mant);
}


// Value returns the numerator and denominator of x.
//
func (x *Rational) Value() (numerator *Integer, denominator Natural) {
	return x.a, x.b
}


// Predicates

// IsZero returns true iff x == 0.
//
func (x *Rational) IsZero() bool	{ return x.a.IsZero() }


// IsNeg returns true iff x < 0.
//
func (x *Rational) IsNeg() bool	{ return x.a.IsNeg() }


// IsPos returns true iff x > 0.
//
func (x *Rational) IsPos() bool	{ return x.a.IsPos() }


// IsInt returns true iff x can be written with a denominator 1
// in the form x == x'/1; i.e., if x is an integer value.
//
func (x *Rational) IsInt() bool	{ return x.b.Cmp(Nat(1)) == 0 }


// Operations

// Neg returns the negated value of x.
//
func (x *Rational) Neg() *Rational	{ return MakeRat(x.a.Neg(), x.b) }


// Add returns the sum x + y.
//
func (x *Rational) Add(y *Rational) *Rational {
	return MakeRat((x.a.MulNat(y.b)).Add(y.a.MulNat(x.b)), x.b.Mul(y.b))
}


// Sub returns the difference x - y.
//
func (x *Rational) Sub(y *Rational) *Rational {
	return MakeRat((x.a.MulNat(y.b)).Sub(y.a.MulNat(x.b)), x.b.Mul(y.b))
}


// Mul returns the product x * y.
//
func (x *Rational) Mul(y *Rational) *Rational	{ return MakeRat(x.a.Mul(y.a), x.b.Mul(y.b)) }


// Quo returns the quotient x / y for y != 0.
// If y == 0, a division-by-zero run-time error occurs.
//
func (x *Rational) Quo(y *Rational) *Rational {
	a := x.a.MulNat(y.b);
	b := y.a.MulNat(x.b);
	if b.IsNeg() {
		a = a.Neg()
	}
	return MakeRat(a, b.mant);
}


// Cmp compares x and y. The result is an int value
//
//   <  0 if x <  y
//   == 0 if x == y
//   >  0 if x >  y
//
func (x *Rational) Cmp(y *Rational) int	{ return (x.a.MulNat(y.b)).Cmp(y.a.MulNat(x.b)) }


// ToString converts x to a string for a given base, with 2 <= base <= 16.
// The string representation is of the form "n" if x is an integer; otherwise
// it is of form "n/d".
//
func (x *Rational) ToString(base uint) string {
	s := x.a.ToString(base);
	if !x.IsInt() {
		s += "/" + x.b.ToString(base)
	}
	return s;
}


// String converts x to its decimal string representation.
// x.String() is the same as x.ToString(10).
//
func (x *Rational) String() string	{ return x.ToString(10) }


// Format is a support routine for fmt.Formatter. It accepts
// the formats 'b' (binary), 'o' (octal), and 'x' (hexadecimal).
//
func (x *Rational) Format(h fmt.State, c int)	{ fmt.Fprintf(h, "%s", x.ToString(fmtbase(c))) }


// RatFromString returns the rational number corresponding to the
// longest possible prefix of s representing a rational number in a
// given conversion base, the actual conversion base used, and the
// prefix length. The syntax of a rational number is:
//
//	rational = mantissa [exponent] .
//	mantissa = integer ('/' natural | '.' natural) .
//	exponent = ('e'|'E') integer .
//
// If the base argument is 0, the string prefix determines the actual
// conversion base for the mantissa. A prefix of ``0x'' or ``0X'' selects
// base 16; the ``0'' prefix selects base 8. Otherwise the selected base is 10.
// If the mantissa is represented via a division, both the numerator and
// denominator may have different base prefixes; in that case the base of
// of the numerator is returned. If the mantissa contains a decimal point,
// the base for the fractional part is the same as for the part before the
// decimal point and the fractional part does not accept a base prefix.
// The base for the exponent is always 10.
//
func RatFromString(s string, base uint) (*Rational, uint, int) {
	// read numerator
	a, abase, alen := IntFromString(s, base);
	b := Nat(1);

	// read denominator or fraction, if any
	var blen int;
	if alen < len(s) {
		ch := s[alen];
		if ch == '/' {
			alen++;
			b, base, blen = NatFromString(s[alen:], base);
		} else if ch == '.' {
			alen++;
			b, base, blen = NatFromString(s[alen:], abase);
			assert(base == abase);
			f := Nat(uint64(base)).Pow(uint(blen));
			a = MakeInt(a.sign, a.mant.Mul(f).Add(b));
			b = f;
		}
	}

	// read exponent, if any
	rlen := alen + blen;
	if rlen < len(s) {
		ch := s[rlen];
		if ch == 'e' || ch == 'E' {
			rlen++;
			e, _, elen := IntFromString(s[rlen:], 10);
			rlen += elen;
			m := Nat(10).Pow(uint(e.mant.Value()));
			if e.sign {
				b = b.Mul(m)
			} else {
				a = a.MulNat(m)
			}
		}
	}

	return MakeRat(a, b), base, rlen;
}
