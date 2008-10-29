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


// ----------------------------------------------------------------------------
// Representation
//
// A natural number of the form
//
//   x = x[n-1]*B^(n-1) + x[n-2]*B^(n-2) + ... + x[1]*B + x[0]
//
// with 0 <= x[i] < B and 0 <= i < n is stored in an array of length n,
// with the digits x[i] as the array elements. 0 is represented as an
// empty array (length == 0).
//
// A natural number is normalized if the array contains no leading 0 digits.
// During arithmetic operations, denormalized values may occur which are
// always normalized before returning the final result.
//
// The base B is chosen as large as possible on a given platform but there
// are a few constraints besides the largest unsigned integer type available.
// TODO describe the constraints.

type Word uint64;
const LogW = 64;

const LogH = 4;  // bits for a hex digit (= "small" number)
const H = 1 << LogH;

const LogB = LogW - LogH;
const L = LogB;
const B = 1 << LogB;
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


func IsSmall(x Word) bool {
	return x < H;
}


func Split(x Word) (Word, Word) {
	return x>>L, x&M;
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
		z[1], z[0] = Split(x);
	}
	return z;
}


func Normalize(x *Natural) *Natural {
	n := len(x);
	for n > 0 && x[n - 1] == 0 { n-- }
	if n < len(x) {
		x = x[0 : n];  // trim leading 0's
	}
	return x;
}


func Normalize3(x *Natural3) *Natural3 {
	n := len(x);
	for n > 0 && x[n - 1] == 0 { n-- }
	if n < len(x) {
		x = x[0 : n];  // trim leading 0's
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
	for ; i < m; i++ { c, z[i] = Split(x[i] + y[i] + c); }
	for ; i < n; i++ { c, z[i] = Split(x[i] + c); }
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
	for ; i < m; i++ { c, z[i] = Split(x[i] - y[i] + c); }
	for ; i < n; i++ { c, z[i] = Split(x[i] + c); }
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
	for i := 0; i < n; i++ { c, z[i] = Split(x[i]*a + c); }
	z[n] = c;

	return Normalize(z);
}


// Returns c = x*y div B, z = x*y mod B.
func Mul1(x, y Word) (Word, Word) {
	// Split x and y into 2 sub-digits each (in base sqrt(B)),
	// multiply the digits separately while avoiding overflow,
	// and return the product as two separate digits.

	const L0 = (L + 1)/2;
	const L1 = L - L0;
	const DL = L0 - L1;  // 0 or 1
	const b  = 1<<L0;
	const m  = b - 1;

	// split x and y into sub-digits
	// x = (x1*b + x0)
	// y = (y1*b + y0)
	x1, x0 := x>>L0, x&m;
	y1, y0 := y>>L0, y&m;

	// x*y = t2*b^2 + t1*b + t0
	t0 := x0*y0;
	t1 := x1*y0 + x0*y1;
	t2 := x1*y1;

	// compute the result digits but avoid overflow
	// z = z1*B + z0 = x*y
	z0 := (t1<<L0 + t0)&M;
	z1 := t2<<DL + (t1 + t0>>L0)>>L1;
	
	return z1, z0;
}


func (x *Natural) Mul(y *Natural) *Natural {
	n := len(x);
	m := len(y);
	z := new(Natural, n + m);

	for j := 0; j < m; j++ {
		d := y[j];
		if d != 0 {
			c := Word(0);
			for i := 0; i < n; i++ {
				// z[i+j] += x[i]*d + c;
				z1, z0 := Mul1(x[i], d);
				c, z[i+j] = Split(z[i+j] + z0 + c);
				c += z1;
			}
			z[n+j] = c;
		}
	}

	return Normalize(z);
}


// BUG use these until 6g shifts are working properly
func shl(x Word, s uint) Word {
	return x << s;
}


func shr(x Word, s uint) Word {
	return x >> s;
}


func Shl1(x, c Word, s uint) (Word, Word) {
	assert(s <= LogB);
	return shr(x, (LogB - s)), shl(x, s)&M | c
}


func (x *Natural) Shl(s uint) *Natural {
	n := len(x);
	si := int(s/LogB);
	s = s%LogB;
	z := new(Natural, n + si + 1);
	
	i := 0;
	c := Word(0);
	for ; i < n; i++ { c, z[i+si] = Shl1(x[i], c, s); }
	z[i+si] = c;
	
	return Normalize(z);
}


func (x *Natural) Shr(s uint) *Natural {
	panic("incomplete");
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


func Log1(x Word) int {
	n := -1;
	for x != 0 { x >>= 1; n++; }
	return n;
}


func (x *Natural) Log() int {
	n := len(x);
	if n > 0 {
		n = (n - 1)*L + Log1(x[n - 1]);
	} else {
		n = -1;
	}
	return n;
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
	for ; i < m; i++ { z[i] = x[i] & y[i]; }
	for ; i < n; i++ { z[i] = x[i]; }

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
	for ; i < m; i++ { z[i] = x[i] | y[i]; }
	for ; i < n; i++ { z[i] = x[i]; }

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
	for ; i < m; i++ { z[i] = x[i] ^ y[i]; }
	for ; i < n; i++ { z[i] = x[i]; }

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
		c = c<<L + x[i];
		x[i] = c/d;
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
	n := x.Log()/3 + 10;  // +10 (round up) - what is the right number?
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
	m := (a + b)>>1;
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
