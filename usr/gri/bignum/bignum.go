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
// Representation

type Word uint64
const LogW = 32;

const LogH = 4;  // bits for a hex digit (= "small" number)
const H = 1 << LogH;

const L = LogW - LogH;  // must be even (for Mul1)
const B = 1 << L;
const M = B - 1;


// For division

const (
	L3 = L / 3;
	B3 = 1 << L3;
	M3 = B3 - 1;
)


type (
	Word3 uint32;
	Natural3 [] Word3;
)


// ----------------------------------------------------------------------------
// Support

// TODO replace this with a Go built-in assert
func assert(p bool) {
	if !p {
		panic("assert failed");
	}
}


func init() {
	assert(L % 2 == 0);  // L must be even
}


func IsSmall(x Word) bool {
	return x < H;
}


func Update(x Word) (Word, Word) {
	return x & M, x >> L;
}


export func Dump(x *[]Word) {
	print("[", len(x), "]");
	for i := len(x) - 1; i >= 0; i-- {
		print(" ", x[i]);
	}
	println();
}


// ----------------------------------------------------------------------------
// Natural numbers

export type Natural []Word;
export var NatZero *Natural = new(Natural, 0);


export func NewNat(x Word) *Natural {
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


func Normalize(x *Natural) *Natural {
	i := len(x);
	for i > 0 && x[i - 1] == 0 { i-- }
	if i < len(x) {
		x = x[0 : i];  // trim leading 0's
	}
	return x;
}


func Normalize3(x *Natural3) *Natural3 {
	i := len(x);
	for i > 0 && x[i - 1] == 0 { i-- }
	if i < len(x) {
		x = x[0 : i];  // trim leading 0's
	}
	return x;
}


func (x *Natural) IsZero() bool {
	return len(x) == 0;
}


func (x *Natural) Add(y *Natural) *Natural {
	n := len(x);
	m := len(y);
	if n < m {
		return y.Add(x);
	}
	assert(n >= m);
	z := new(Natural, n + 1);

	i := 0;
	c := Word(0);
	for i < m { z[i], c = Update(x[i] + y[i] + c); i++; }
	for i < n { z[i], c = Update(x[i] + c); i++; }
	z[i] = c;

	return Normalize(z);
}


func (x *Natural) Sub(y *Natural) *Natural {
	n := len(x);
	m := len(y);
	assert(n >= m);
	z := new(Natural, n);

	i := 0;
	c := Word(0);
	for i < m { z[i], c = Update(x[i] - y[i] + c); i++; }
	for i < n { z[i], c = Update(x[i] + c); i++; }
	assert(c == 0);  // x.Sub(y) must be called with x >= y

	return Normalize(z);
}


// Computes x = x*a + c (in place) for "small" a's.
func (x* Natural) MulAdd1(a, c Word) *Natural {
	assert(IsSmall(a-1) && IsSmall(c));
	if x.IsZero() || a == 0 {
		return NewNat(c);
	}
	n := len(x);

	z := new(Natural, n + 1);
	for i := 0; i < n; i++ { z[i], c = Update(x[i] * a + c); }
	z[n] = c;

	return Normalize(z);
}


// Returns z = (x * y) div B, c = (x * y) mod B.
func Mul1(x, y Word) (Word, Word) {
	const L2 = (L + 1) / 2;  // TODO check if we can run with odd L
	const B2 = 1 << L2;
	const M2 = B2 - 1;
	
	x0 := x & M2;
	x1 := x >> L2;

	y0 := y & M2;
	y1 := y >> L2;

	z0 := x0*y0;
	z1 := x1*y0 + x0*y1 + z0 >> L2;  z0 &= M2;
	z2 := x1*y1 + z1 >> L2;  z1 &= M2;
	
	return z1 << L2 | z0, z2;
}


func (x *Natural) Mul(y *Natural) *Natural {
	if x.IsZero() || y.IsZero() {
		return NatZero;
	}
	xl := len(x);
	yl := len(y);
	if xl < yl {
		return y.Mul(x);  // for speed
	}
	assert(xl >= yl && yl > 0);

	// initialize z
	zl := xl + yl;
	z := new(Natural, zl);

	for j := 0; j < yl; j++ {
		d := y[j];
		if d != 0 {
			k := j;
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
			z[k] = c;
		}
	}

	return Normalize(z);
}


func Shl1(x Word, s int) (Word, Word) {
	return 0, 0
}


func Shr1(x Word, s int) (Word, Word) {
	return 0, 0
}


func (x *Natural) Shl(s int) *Natural {
	panic("incomplete");
	
	if s == 0 {
		return x;
	}
	
	S := s/L;
	s = s%L;
	n := len(x) + S + 1;
	z := new(Natural, n);
	
	c := Word(0);
	for i := 0; i < n; i++ {
		z[i + S], c = Shl1(x[i], s);
	}
	z[n + S] = c;
	
	return Normalize(z);
}


func (x *Natural) Shr(s uint) *Natural {
	panic("incomplete");
	
	if s == 0 {
		return x;
	}
	return nil
}


func SplitBase(x *Natural) *Natural3 {
	xl := len(x);
	z := new(Natural3, xl * 3);
	for i, j := 0, 0; i < xl; i, j = i + 1, j + 3 {
		t := x[i];
		z[j] = Word3(t & M3);  t >>= L3;  j++;
		z[j] = Word3(t & M3);  t >>= L3;  j++;
		z[j] = Word3(t & M3);  t >>= L3;  j++;
	}
	return Normalize3(z);
}


func Scale(x *Natural, f Word) *Natural3 {
	return nil;
}


func TrialDigit(r, d *Natural3, k, m int) Word {
	km := k + m;
	assert(2 <= m && m <= km);
	r3 := (Word(r[km]) << L3 + Word(r[km - 1])) << L3 + Word(r[km - 2]);
	d2 := Word(d[m - 1]) << L3 + Word(d[m - 2]);
	qt := r3 / d2;
	if qt >= B {
		qt = B - 1;
	}
	return qt;
}


func DivMod(x, y *Natural) {
	xl := len(x);
	yl := len(y);
	assert(2 <= yl && yl <= xl);  // use special-case algorithm otherwise
	
	f := B / (y[yl - 1] + 1);
	r := Scale(x, f);
	d := Scale(y, f);
	n := len(r);
	m := len(d);
	
	for k := n - m; k >= 0; k-- {
		qt := TrialDigit(r, d, k, m);
		
	}
}


func (x *Natural) Div(y *Natural) *Natural {
	panic("UNIMPLEMENTED");
	return nil;
}


func (x *Natural) Mod(y *Natural) *Natural {
	panic("UNIMPLEMENTED");
	return nil;
}


func (x *Natural) Cmp(y *Natural) int {
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
	n := len(x);
	if n == 0 { return 0; }
	assert(n > 0);
	
	c := (n - 1) * L;
	for t := x[n - 1]; t != 0; t >>= 1 { c++ };

	return c;
}


func (x *Natural) And(y *Natural) *Natural {
	n := len(x);
	m := len(y);
	if n < m {
		return y.And(x);
	}
	assert(n >= m);
	z := new(Natural, n);

	i := 0;
	for i < m { z[i] = x[i] & y[i]; i++; }
	for i < n { z[i] = x[i]; i++; }

	return Normalize(z);
}


func (x *Natural) Or(y *Natural) *Natural {
	n := len(x);
	m := len(y);
	if n < m {
		return y.Or(x);
	}
	assert(n >= m);
	z := new(Natural, n);

	i := 0;
	for i < m { z[i] = x[i] | y[i]; i++; }
	for i < n { z[i] = x[i]; i++; }

	return Normalize(z);
}


func (x *Natural) Xor(y *Natural) *Natural {
	n := len(x);
	m := len(y);
	if n < m {
		return y.Xor(x);
	}
	assert(n >= m);
	z := new(Natural, n);

	i := 0;
	for i < m { z[i] = x[i] ^ y[i]; i++; }
	for i < n { z[i] = x[i]; i++; }

	return Normalize(z);
}


func Copy(x *Natural) *Natural {
	z := new(Natural, len(x));
	//*z = *x;  // BUG assignment does't work yet
	for i := len(x) - 1; i >= 0; i-- { z[i] = x[i]; }
	return z;
}


// Computes x = x div d (in place - the recv maybe modified) for "small" d's.
// Returns updated x and x mod d.
func (x *Natural) DivMod1(d Word) (*Natural, Word) {
	assert(0 < d && IsSmall(d - 1));

	c := Word(0);
	for i := len(x) - 1; i >= 0; i-- {
		var LL Word = L;  // BUG shift broken for const L
		c = c << LL + x[i];
		x[i] = c / d;
		c %= d;
	}

	return Normalize(x), c;
}


func (x *Natural) String(base Word) string {
	if x.IsZero() {
		return "0";
	}
	
	// allocate string
	// TODO n is too small for bases < 10!!!
	assert(base >= 10);  // for now
	// approx. length: 1 char for 3 bits
	n := x.Log()/3 + 1;  // +1 (round up)
	s := new([]byte, n);

	// convert
	const hex = "0123456789abcdef";
	i := n;
	x = Copy(x);  // don't destroy recv
	for !x.IsZero() {
		i--;
		var d Word;
		x, d = x.DivMod1(base);
		s[i] = hex[d];
	};

	return string(s[i : n]);
}


func MulRange(a, b Word) *Natural {
	switch {
	case a > b: return NewNat(1);
	case a == b: return NewNat(a);
	case a + 1 == b: return NewNat(a).Mul(NewNat(b));
	}
	m := (a + b) >> 1;
	assert(a <= m && m < b);
	return MulRange(a, m).Mul(MulRange(m + 1, b));
}


export func Fact(n Word) *Natural {
	// Using MulRange() instead of the basic for-loop
	// lead to faster factorial computation.
	return MulRange(2, n);
}


func HexValue(ch byte) Word {
	d := Word(H);
	switch {
	case '0' <= ch && ch <= '9': d = Word(ch - '0');
	case 'a' <= ch && ch <= 'f': d = Word(ch - 'a') + 10;
	case 'A' <= ch && ch <= 'F': d = Word(ch - 'A') + 10;
	}
	return d;
}


// TODO auto-detect base if base argument is 0
export func NatFromString(s string, base Word) *Natural {
	x := NatZero;
	for i := 0; i < len(s); i++ {
		d := HexValue(s[i]);
		if d < base {
			x = x.MulAdd1(base, d);
		} else {
			break;
		}
	}
	return x;
}


// ----------------------------------------------------------------------------
// Integer numbers

export type Integer struct {
	sign bool;
	mant *Natural;
}


func (x *Integer) Add(y *Integer) *Integer {
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


func (x *Integer) Sub(y *Integer) *Integer {
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


func (x *Integer) Mul(y *Integer) *Integer {
	// x * y == x * y
	// x * (-y) == -(x * y)
	// (-x) * y == -(x * y)
	// (-x) * (-y) == x * y
	return &Integer{x.sign != y.sign, x.mant.Mul(y.mant)};
}


func (x *Integer) Div(y *Integer) *Integer {
	panic("UNIMPLEMENTED");
	return nil;
}


func (x *Integer) Mod(y *Integer) *Integer {
	panic("UNIMPLEMENTED");
	return nil;
}


func (x *Integer) Cmp(y *Integer) int {
	panic("UNIMPLEMENTED");
	return 0;
}


func (x *Integer) String(base Word) string {
	if x.mant.IsZero() {
		return "0";
	}
	var s string;
	if x.sign {
		s = "-";
	}
	return s + x.mant.String(base);
}

	
export func IntFromString(s string, base Word) *Integer {
	// get sign, if any
	sign := false;
	if len(s) > 0 && (s[0] == '-' || s[0] == '+') {
		sign = s[0] == '-';
	}
	return &Integer{sign, NatFromString(s[1 : len(s)], base)};
}


// ----------------------------------------------------------------------------
// Rational numbers

export type Rational struct {
	a, b *Integer;  // a = numerator, b = denominator
}


func NewRat(a, b *Integer) *Rational {
	// TODO normalize the rational
	return &Rational{a, b};
}


func (x *Rational) Add(y *Rational) *Rational {
	return NewRat((x.a.Mul(y.b)).Add(x.b.Mul(y.a)), x.b.Mul(y.b));
}


func (x *Rational) Sub(y *Rational) *Rational {
	return NewRat((x.a.Mul(y.b)).Sub(x.b.Mul(y.a)), x.b.Mul(y.b));
}


func (x *Rational) Mul(y *Rational) *Rational {
	return NewRat(x.a.Mul(y.a), x.b.Mul(y.b));
}


func (x *Rational) Div(y *Rational) *Rational {
	return NewRat(x.a.Mul(y.b), x.b.Mul(y.a));
}


func (x *Rational) Mod(y *Rational) *Rational {
	panic("UNIMPLEMENTED");
	return nil;
}


func (x *Rational) Cmp(y *Rational) int {
	panic("UNIMPLEMENTED");
	return 0;
}


export func RatFromString(s string) *Rational {
	panic("UNIMPLEMENTED");
	return nil;
}


// ----------------------------------------------------------------------------
// Scaled numbers

export type Number struct {
	mant *Rational;
	exp Integer;
}
