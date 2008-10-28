// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Bignum

// A package for arbitrary precision arithmethic.
// It implements the following numeric types:
//
// - Natural	unsigned integer numbers
// - Integer	signed integer numbers
// - Rational	rational numbers
// - Number		scaled rational numbers (contain exponent)


// ----------------------------------------------------------------------------
// Support

type Word uint32

const N = 4;
const L = 28;  // = sizeof(Word) * 8
const B = 1 << L;
const M = B - 1;


// TODO replace this with a Go built-in assert
func ASSERT(p bool) {
	if !p {
		panic("ASSERT failed");
	}
}


func IsSmall(x Word) bool {
	return x < 1 << N;
}


func Update(x Word) (Word, Word) {
	return x & M, x >> L;
}


// ----------------------------------------------------------------------------
// Naturals

export type Natural []Word;
export var NatZero *Natural = new(Natural, 0);


func (x *Natural) IsZero() bool {
	return len(x) == 0;
}


func (x *Natural) Add (y *Natural) *Natural {
	xl := len(x);
	yl := len(y);
	if xl < yl {
		return y.Add(x);
	}
	ASSERT(xl >= yl);
	z := new(Natural, xl + 1);

	i := 0;
	c := Word(0);
	for i < yl { z[i], c = Update(x[i] + y[i] + c); i++; }
	for i < xl { z[i], c = Update(x[i] + c); i++; }
	if c != 0 { z[i] = c; i++; }
	z = z[0 : i];

	return z;
}


func (x *Natural) Sub (y *Natural) *Natural {
	xl := len(x);
	yl := len(y);
	ASSERT(xl >= yl);
	z := new(Natural, xl);

	i := 0;
	c := Word(0);
	for i < yl { z[i], c = Update(x[i] - y[i] + c); i++; }
	for i < xl { z[i], c = Update(x[i] + c); i++; }
	ASSERT(c == 0);  // usub(x, y) must be called with x >= y
	for i > 0 && z[i - 1] == 0 { i--; }
	z = z[0 : i];

	return z;
}


// Computes x = x*a + c (in place) for "small" a's.
func (x* Natural) Mul1Add(a, c Word) *Natural {
	ASSERT(IsSmall(a) && IsSmall(c));
	if (x.IsZero() || a == 0) && c == 0 {
		return NatZero;
	}
	xl := len(x);

	z := new(Natural, xl + 1);
	i := 0;
	for i < xl { z[i], c = Update(x[i] * a + c); i++; }
	if c != 0 { z[i] = c; i++; }  
	z = z[0 : i];

	return z;
}


// Returns z = (x * y) div B, c = (x * y) mod B.
func Mul1(x, y Word) (z Word, c Word) {
	const L2 = (L + 1) >> 1;
	const B2 = 1 << L2;
	const M2 = B2 - 1;

	x0 := x & M2;
	x1 := x >> L2;

	y0 := y & M2;
	y1 := y >> L2;

	z10 := x0*y0;
	z21 := x1*y0 + x0*y1 + z10 >> L2;

	cc := x1*y1 + z21 >> L2;  
	zz := z21 & M2 << L2 | z10 & M2;
	return zz, cc
}


func (x *Natural) Mul (y *Natural) *Natural {
	if x.IsZero() || y.IsZero() {
		return NatZero;
	}
	xl := len(x);
	yl := len(y);
	if xl < yl {
		return y.Mul(x);  // for speed
	}
	ASSERT(xl >= yl && yl > 0);

	// initialize z
	zl := xl + yl;
	z := new(Natural, zl);

	k := 0;
	for j := 0; j < yl; j++ {
		d := y[j];
		if d != 0 {
			k = j;
			c := Word(0);
			for i := 0; i < xl; i++ {
				// compute z[k] += x[i] * d + c;
				t := z[k] + c;
				var z1 Word;
				z1, c = Mul1(x[i], d);
				t += z1;
				z[k] = t & M;
				c += t >> L;
				k++;
			}
			if c != 0 {
				z[k] = c;
				k++;
			}
		}
	}
	z = z[0 : k];

	return z;
}


func (x *Natural) Div (y *Natural) *Natural {
	panic("UNIMPLEMENTED");
	return nil;
}


func (x *Natural) Mod (y *Natural) *Natural {
	panic("UNIMPLEMENTED");
	return nil;
}


func (x *Natural) Cmp (y *Natural) int {
	xl := len(x);
	yl := len(y);

	if xl != yl || xl == 0 {
		return xl - yl;
	}

	i := xl - 1;
	for i > 0 && x[i] == y[i] { i--; }
	
	d := 0;
	switch {
	case x[i] < y[i]: d = -1;
	case x[i] > y[i]: d = 1;
	}
	return d;
}


func (x *Natural) Log() int {
	xl := len(x);
	if xl == 0 { return 0; }

	n := (xl - 1) * L;
	for t := x[xl - 1]; t != 0; t >>= 1 { n++ };

	return n;
}


func (x *Natural) And (y *Natural) *Natural {
	xl := len(x);
	yl := len(y);
	if xl < yl {
		return y.And(x);
	}
	ASSERT(xl >= yl);
	z := new(Natural, xl);

	i := 0;
	for i < yl { z[i] = x[i] & y[i]; i++; }
	for i < xl { z[i] = x[i]; i++; }
	for i > 0 && z[i - 1] == 0 { i--; }
	z = z[0 : i];

	return z;
}


func (x *Natural) Or (y *Natural) *Natural {
	xl := len(x);
	yl := len(y);
	if xl < yl {
		return y.Or(x);
	}
	ASSERT(xl >= yl);
	z := new(Natural, xl);

	i := 0;
	for i < yl { z[i] = x[i] | y[i]; i++; }
	for i < xl { z[i] = x[i]; i++; }

	return z;
}


func (x *Natural) Xor (y *Natural) *Natural {
	xl := len(x);
	yl := len(y);
	if xl < yl {
		return y.Xor(x);
	}
	ASSERT(xl >= yl);
	z := new(Natural, xl);

	i := 0;
	for i < yl { z[i] = x[i] ^ y[i]; i++; }
	for i < xl { z[i] = x[i]; i++; }
	for i > 0 && z[i - 1] == 0 { i--; }
	z = z[0 : i];

	return z;
}


// Returns a copy of x with space for one extra digit (for Div/Mod use)
func Copy(x *Natural) *Natural {
	xl := len(x);

	z := new(Natural, xl + 1);  // add space for one extra digit
	for i := 0; i < xl; i++ { z[i] = x[i]; }
	z = z[0 : xl];

	return z;
}


// Computes x = x div d (in place) for "small" d's. Returns updated x, x mod d.
func (x *Natural) Mod1 (d Word) (*Natural, Word) {
	ASSERT(IsSmall(d));
	xl := len(x);
	
	c := Word(0);
	for i := xl - 1; i >= 0; i-- {
		c = c << L + x[i];
		x[i], c = c / d, c %d;
	}
	if xl > 0 && x[xl - 1] == 0 {
		x = x[0 : xl - 1];
	}

	return x, c;
}


func (x *Natural) String() string {
	if x.IsZero() {
		return "0";
	}
	
	// allocate string
	// approx. length: 1 char for 3 bits
	n := x.Log()/3 + 1;  // +1 (round up)
	s := new([]byte, n);

	// convert
	i := n;
	x = Copy(x);  // don't destroy recv
	for !x.IsZero() {
		i--;
		var d Word;
		x, d = x.Mod1(10);
		s[i] = byte(d) + '0';
	};

	return string(s[i : n]);
}


export func NatFromWord(x Word) *Natural {
	var z *Natural;
	switch {
	case x == 0:
		z = NatZero;
	case x < B:
		z = new(Natural, 1);
		z[0] = x;
		return z;
	default:
		z = new(Natural, 2);
		z[0], z[1] = Update(x);
	}
	return z;
}


// Support function for faster factorial computation.
func MulRange(a, b Word) *Natural {
	switch {
	case a > b: return NatFromWord(1);
	case a == b: return NatFromWord(a);
	case a + 1 == b: return NatFromWord(a).Mul(NatFromWord(b));
	}
	m := (a + b) >> 1;
	ASSERT(a <= m && m < b);
	return MulRange(a, m).Mul(MulRange(m + 1, b));
}


export func Fact(n Word) *Natural {
  return MulRange(2, n);
}


export func NatFromString(s string) *Natural {
	x := NatZero;
	for i := 0; i < len(s) && '0' <= s[i] && s[i] <= '9'; i++ {
		x = x.Mul1Add(10, Word(s[i] - '0'));
	}
	return x;
}


// ----------------------------------------------------------------------------
// Integers

export type Integer struct {
	sign bool;
	mant *Natural;
}


func (x *Integer) Add (y *Integer) *Integer {
	var z *Integer;
	if x.sign == y.sign {
		// x + y == x + y
		// (-x) + (-y) == -(x + y)
		z = &Integer{x.sign, x.mant.Add(y.mant)};
	} else {
		// x + (-y) == x - y == -(y - x)
		// (-x) + y == y - x == -(x - y)
		if x.mant.Cmp(y.mant) >= 0 {
			z = &Integer{false, x.mant.Sub(y.mant)};
		} else {
			z = &Integer{true, y.mant.Sub(x.mant)};
		}
	}
	if x.sign {
		z.sign = !z.sign;
	}
	return z;
}


func (x *Integer) Sub (y *Integer) *Integer {
	var z *Integer;
	if x.sign != y.sign {
		// x - (-y) == x + y
		// (-x) - y == -(x + y)
		z = &Integer{x.sign, x.mant.Add(y.mant)};
	} else {
		// x - y == x - y == -(y - x)
		// (-x) - (-y) == y - x == -(x - y)
		if x.mant.Cmp(y.mant) >= 0 {
			z = &Integer{false, x.mant.Sub(y.mant)};
		} else {
			z = &Integer{true, y.mant.Sub(x.mant)};
		}
	}
	if x.sign {
		z.sign = !z.sign;
	}
	return z;
}


func (x *Integer) Mul (y *Integer) *Integer {
	// x * y == x * y
	// x * (-y) == -(x * y)
	// (-x) * y == -(x * y)
	// (-x) * (-y) == x * y
	return &Integer{x.sign != y.sign, x.mant.Mul(y.mant)};
}


func (x *Integer) Div (y *Integer) *Integer {
	panic("UNIMPLEMENTED");
	return nil;
}


func (x *Integer) Mod (y *Integer) *Integer {
	panic("UNIMPLEMENTED");
	return nil;
}


func (x *Integer) Cmp (y *Integer) int {
	panic("UNIMPLEMENTED");
	return 0;
}


func (x *Integer) String() string {
	if x.mant.IsZero() {
		return "0";
	}
	var s string;
	if x.sign {
		s = "-" + x.mant.String();
	} else {
		s = x.mant.String();
	}
	return s;
}

	
export func IntFromString(s string) *Integer {
	// get sign, if any
	sign := false;
	if len(s) > 0 && (s[0] == '-' || s[0] == '+') {
		sign = s[0] == '-';
	}
	return &Integer{sign, NatFromString(s[1 : len(s)])};
}


// ----------------------------------------------------------------------------
// Rationals

export type Rational struct {
	a, b *Integer;  // a = numerator, b = denominator
}


func NewRat(a, b *Integer) *Rational {
	// TODO normalize the rational
	return &Rational{a, b};
}


func (x *Rational) Add (y *Rational) *Rational {
	return NewRat((x.a.Mul(y.b)).Add(x.b.Mul(y.a)), x.b.Mul(y.b));
}


func (x *Rational) Sub (y *Rational) *Rational {
	return NewRat((x.a.Mul(y.b)).Sub(x.b.Mul(y.a)), x.b.Mul(y.b));
}


func (x *Rational) Mul (y *Rational) *Rational {
	return NewRat(x.a.Mul(y.a), x.b.Mul(y.b));
}


func (x *Rational) Div (y *Rational) *Rational {
	return NewRat(x.a.Mul(y.b), x.b.Mul(y.a));
}


func (x *Rational) Mod (y *Rational) *Rational {
	panic("UNIMPLEMENTED");
	return nil;
}


func (x *Rational) Cmp (y *Rational) int {
	panic("UNIMPLEMENTED");
	return 0;
}


export func RatFromString(s string) *Rational {
	panic("UNIMPLEMENTED");
	return nil;
}


// ----------------------------------------------------------------------------
// Numbers

export type Number struct {
	mant *Rational;
	exp Integer;
}
