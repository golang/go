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
// with the digits x[i] as the array elements.
//
// A natural number is normalized if the array contains no leading 0 digits.
// During arithmetic operations, denormalized values may occur which are
// always normalized before returning the final result. The normalized
// representation of 0 is the empty array (length = 0).
//
// The base B is chosen as large as possible on a given platform but there
// are a few constraints besides the size of the largest unsigned integer
// type available.
// TODO describe the constraints.

const LogW = 64;
const LogH = 4;  // bits for a hex digit (= "small" number)
const LogB = LogW - LogH;


const (
	L2 = LogB / 2;
	B2 = 1 << L2;
	M2 = B2 - 1;

	L = L2 * 2;
	B = 1 << L;
	M = B - 1;
)


type (
	Digit2 uint32;
	Digit  uint64;
)


// ----------------------------------------------------------------------------
// Support

// TODO replace this with a Go built-in assert
func assert(p bool) {
	if !p {
		panic("assert failed");
	}
}


// ----------------------------------------------------------------------------
// Raw operations

func And1(z, x *[]Digit, y Digit) {
	for i := len(x) - 1; i >= 0; i-- {
		z[i] = x[i] & y;
	}
}


func And(z, x, y *[]Digit) {
	for i := len(x) - 1; i >= 0; i-- {
		z[i] = x[i] & y[i];
	}
}


func Or1(z, x *[]Digit, y Digit) {
	for i := len(x) - 1; i >= 0; i-- {
		z[i] = x[i] | y;
	}
}


func Or(z, x, y *[]Digit) {
	for i := len(x) - 1; i >= 0; i-- {
		z[i] = x[i] | y[i];
	}
}


func Xor1(z, x *[]Digit, y Digit) {
	for i := len(x) - 1; i >= 0; i-- {
		z[i] = x[i] ^ y;
	}
}


func Xor(z, x, y *[]Digit) {
	for i := len(x) - 1; i >= 0; i-- {
		z[i] = x[i] ^ y[i];
	}
}


func Add1(z, x *[]Digit, c Digit) Digit {
	n := len(x);
	for i := 0; i < n; i++ {
		t := c + x[i];
		c, z[i] = t>>L, t&M
	}
	return c;
}


func Add(z, x, y *[]Digit) Digit {
	var c Digit;
	n := len(x);
	for i := 0; i < n; i++ {
		t := c + x[i] + y[i];
		c, z[i] = t>>L, t&M
	}
	return c;
}


func Sub1(z, x *[]Digit, c Digit) Digit {
	n := len(x);
	for i := 0; i < n; i++ {
		t := c + x[i];
		c, z[i] = Digit(int64(t)>>L), t&M;  // arithmetic shift!
	}
	return c;
}


func Sub(z, x, y *[]Digit) Digit {
	var c Digit;
	n := len(x);
	for i := 0; i < n; i++ {
		t := c + x[i] - y[i];
		c, z[i] = Digit(int64(t)>>L), t&M;  // arithmetic shift!
	}
	return c;
}


// Returns c = x*y div B, z = x*y mod B.
func Mul11(x, y Digit) (Digit, Digit) {
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


func Mul(z, x, y *[]Digit) {
	n := len(x);
	m := len(y);
	for j := 0; j < m; j++ {
		d := y[j];
		if d != 0 {
			c := Digit(0);
			for i := 0; i < n; i++ {
				// z[i+j] += c + x[i]*d;
				z1, z0 := Mul11(x[i], d);
				t := c + z[i+j] + z0;
				c, z[i+j] = t>>L, t&M;
				c += z1;
			}
			z[n+j] = c;
		}
	}
}


func Mul1(z, x *[]Digit2, y Digit2) Digit2 {
	n := len(x);
	var c Digit;
	f := Digit(y);
	for i := 0; i < n; i++ {
		t := c + Digit(x[i])*f;
		c, z[i] = t>>L2, Digit2(t&M2);
	}
	return Digit2(c);
}


func Div1(z, x *[]Digit2, y Digit2) Digit2 {
	n := len(x);
	var c Digit;
	d := Digit(y);
	for i := n-1; i >= 0; i-- {
		t := c*B2 + Digit(x[i]);
		c, z[i] = t%d, Digit2(t/d);
	}
	return Digit2(c);
}


func Shl(z, x *[]Digit, s uint) Digit {
	assert(s <= L);
	n := len(x);
	var c Digit;
	for i := 0; i < n; i++ {
		c, z[i] = x[i] >> (L-s), x[i] << s & M | c;
	}
	return c;
}


func Shr(z, x *[]Digit, s uint) Digit {
	assert(s <= L);
	n := len(x);
	var c Digit;
	for i := n - 1; i >= 0; i-- {
		c, z[i] = x[i] << (L-s) & M, x[i] >> s | c;
	}
	return c;
}


// ----------------------------------------------------------------------------
// Support

func IsSmall(x Digit) bool {
	return x < 1<<LogH;
}


func Split(x Digit) (Digit, Digit) {
	return x>>L, x&M;
}


export func Dump(x *[]Digit) {
	print("[", len(x), "]");
	for i := len(x) - 1; i >= 0; i-- {
		print(" ", x[i]);
	}
	println();
}


// ----------------------------------------------------------------------------
// Natural numbers
//
// Naming conventions
//
// B, b   bases
// c      carry
// x, y   operands
// z      result
// n, m   n = len(x), m = len(y)


export type Natural []Digit;
export var NatZero *Natural = new(Natural, 0);


export func Nat(x Digit) *Natural {
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


func Normalize2(x *[]Digit2) *[]Digit2 {
	n := len(x);
	for n > 0 && x[n - 1] == 0 { n-- }
	if n < len(x) {
		x = x[0 : n];  // trim leading 0's
	}
	return x;
}


// Predicates

func (x *Natural) IsZero() bool { return len(x) == 0; }
func (x *Natural) IsOdd() bool { return len(x) > 0 && x[0]&1 != 0; }


func (x *Natural) Add(y *Natural) *Natural {
	n := len(x);
	m := len(y);
	if n < m {
		return y.Add(x);
	}

	z := new(Natural, n + 1);
	c := Add(z[0 : m], x[0 : m], y);
	z[n] = Add1(z[m : n], x[m : n], c);

	return Normalize(z);
}


func (x *Natural) Sub(y *Natural) *Natural {
	n := len(x);
	m := len(y);
	if n < m {
		panic("underflow")
	}

	z := new(Natural, n);
	c := Sub(z[0 : m], x[0 : m], y);
	if Sub1(z[m : n], x[m : n], c) != 0 {
		panic("underflow");
	}

	return Normalize(z);
}


// Computes x = x*a + c (in place) for "small" a's.
func (x* Natural) MulAdd1(a, c Digit) *Natural {
	assert(IsSmall(a-1) && IsSmall(c));
	n := len(x);
	z := new(Natural, n + 1);

	for i := 0; i < n; i++ { c, z[i] = Split(c + x[i]*a); }
	z[n] = c;

	return Normalize(z);
}


func (x *Natural) Mul(y *Natural) *Natural {
	n := len(x);
	m := len(y);

	z := new(Natural, n + m);
	Mul(z, x, y);

	return Normalize(z);
}


func Pop1(x Digit) uint {
	n := uint(0);
	for x != 0 {
		x &= x-1;
		n++;
	}
	return n;
}


func (x *Natural) Pop() uint {
	n := uint(0);
	for i := len(x) - 1; i >= 0; i-- {
		n += Pop1(x[i]);
	}
	return n;
}


func (x *Natural) Pow(n uint) *Natural {
	z := Nat(1);
	for n > 0 {
		// z * x^n == x^n0
		if n&1 == 1 {
			z = z.Mul(x);
		}
		x, n = x.Mul(x), n/2;
	}
	return z;
}


func (x *Natural) Shl(s uint) *Natural {
	n := uint(len(x));
	m := n + s/L;
	z := new(Natural, m+1);
	
	z[m] = Shl(z[m-n : m], x, s%L);
	
	return Normalize(z);
}


func (x *Natural) Shr(s uint) *Natural {
	n := uint(len(x));
	m := n - s/L;
	if m > n {  // check for underflow
		m = 0;
	}
	z := new(Natural, m);
	
	Shr(z, x[n-m : n], s%L);
	
	return Normalize(z);
}


// DivMod needs multi-precision division which is not available if Digit
// is already using the largest uint size. Split base before division,
// and merge again after. Each Digit is split into 2 Digit2's.

func Unpack(x *Natural) *[]Digit2 {
	// TODO Use Log() for better result - don't need Normalize2 at the end!
	n := len(x);
	z := new([]Digit2, n*2 + 1);  // add space for extra digit (used by DivMod)
	for i := 0; i < n; i++ {
		t := x[i];
		z[i*2] = Digit2(t & M2);
		z[i*2 + 1] = Digit2(t >> L2 & M2);
	}
	return Normalize2(z);
}


func Pack(x *[]Digit2) *Natural {
	n := (len(x) + 1) / 2;
	z := new(Natural, n);
	if len(x) & 1 == 1 {
		// handle odd len(x)
		n--;
		z[n] = Digit(x[n*2]);
	}
	for i := 0; i < n; i++ {
		z[i] = Digit(x[i*2 + 1]) << L2 | Digit(x[i*2]);
	}
	return Normalize(z);
}


// Division and modulo computation - destroys x and y. Based on the
// algorithms described in:
//
// 1) D. Knuth, "The Art of Computer Programming. Volume 2. Seminumerical
//    Algorithms." Addison-Wesley, Reading, 1969.
//
// 2) P. Brinch Hansen, Multiple-length division revisited: A tour of the
//    minefield. "Software - Practice and Experience 24", (June 1994),
//    579-601. John Wiley & Sons, Ltd.
//
// Specifically, the inplace computation of quotient and remainder
// is described in 1), while 2) provides the background for a more
// accurate initial guess of the trial digit.

func DivMod2(x, y *[]Digit2) (*[]Digit2, *[]Digit2) {
	const b = B2;
	
	n := len(x);
	m := len(y);
	assert(m > 0);  // division by zero
	assert(n+1 <= cap(x));  // space for one extra digit (should it be == ?)
	x = x[0 : n + 1];
	
	if m == 1 {
		// division by single digit
		// result is shifted left by 1 in place!
		x[0] = Div1(x[1 : n+1], x[0 : n], y[0]);
		
	} else if m > n {
		// quotient = 0, remainder = x
		// TODO in this case we shouldn't even split base - FIX THIS
		m = n;
		
	} else {
		// general case
		assert(2 <= m && m <= n);
		assert(x[n] == 0);
		
		// normalize x and y
		f := b/(Digit(y[m-1]) + 1);
		Mul1(x, x, Digit2(f));
		Mul1(y, y, Digit2(f));
		assert(b/2 <= y[m-1] && y[m-1] < b);  // incorrect scaling
		
		y1, y2 := Digit(y[m-1]), Digit(y[m-2]);
		d2 := Digit(y1)*b + Digit(y2);
		for i := n-m; i >= 0; i-- {
			k := i+m;
			
			// compute trial digit
			var q Digit;
			{	// Knuth
				x0, x1, x2 := Digit(x[k]), Digit(x[k-1]), Digit(x[k-2]);
				if x0 != y1 {
					q = (x0*b + x1)/y1;
				} else {
					q = b-1;
				}
				for y2 * q > (x0*b + x1 - y1*q)*b + x2 {
					q--
				}
			}
			
			// subtract y*q
			c := Digit(0);
			for j := 0; j < m; j++ {
				t := c + Digit(x[i+j]) - Digit(y[j])*q;  // arithmetic shift!
				c, x[i+j] = Digit(int64(t)>>L2), Digit2(t&M2);
			}
			
			// correct if trial digit was too large
			if c + Digit(x[k]) != 0 {
				// add y
				c := Digit(0);
				for j := 0; j < m; j++ {
					t := c + Digit(x[i+j]) + Digit(y[j]);
					c, x[i+j] = uint64(int64(t) >> L2), Digit2(t & M2)
				}
				assert(c + Digit(x[k]) == 0);
				// correct trial digit
				q--;
			}
			
			x[k] = Digit2(q);
		}
		
		// undo normalization for remainder
		c := Div1(x[0 : m], x[0 : m], Digit2(f));
		assert(c == 0);
	}

	return x[m : n+1], x[0 : m];
}


func (x *Natural) Div(y *Natural) *Natural {
	q, r := DivMod2(Unpack(x), Unpack(y));
	return Pack(q);
}


func (x *Natural) Mod(y *Natural) *Natural {
	q, r := DivMod2(Unpack(x), Unpack(y));
	return Pack(r);
}


func (x *Natural) DivMod(y *Natural) (*Natural, *Natural) {
	q, r := DivMod2(Unpack(x), Unpack(y));
	return Pack(q), Pack(r);
}


func (x *Natural) Cmp(y *Natural) int {
	n := len(x);
	m := len(y);

	if n != m || n == 0 {
		return n - m;
	}

	i := n - 1;
	for i > 0 && x[i] == y[i] { i--; }
	
	d := 0;
	switch {
	case x[i] < y[i]: d = -1;
	case x[i] > y[i]: d = 1;
	}

	return d;
}


func Log2(x Digit) int {
	n := -1;
	for x != 0 { x = x >> 1; n++; }  // BUG >>= broken for uint64
	return n;
}


func (x *Natural) Log2() int {
	n := len(x);
	if n > 0 {
		n = (n - 1)*L + Log2(x[n - 1]);
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

	z := new(Natural, n);
	And(z[0 : m], x[0 : m], y);
	Or1(z[m : n], x[m : n], 0);

	return Normalize(z);
}


func (x *Natural) Or(y *Natural) *Natural {
	n := len(x);
	m := len(y);
	if n < m {
		return y.Or(x);
	}

	z := new(Natural, n);
	Or(z[0 : m], x[0 : m], y);
	Or1(z[m : n], x[m : n], 0);

	return Normalize(z);
}


func (x *Natural) Xor(y *Natural) *Natural {
	n := len(x);
	m := len(y);
	if n < m {
		return y.Xor(x);
	}

	z := new(Natural, n);
	Xor(z[0 : m], x[0 : m], y);
	Or1(z[m : n], x[m : n], 0);

	return Normalize(z);
}


// Computes x = x div d (in place - the recv maybe modified) for "small" d's.
// Returns updated x and x mod d.
func (x *Natural) DivMod1(d Digit) (*Natural, Digit) {
	assert(0 < d && IsSmall(d - 1));

	c := Digit(0);
	for i := len(x) - 1; i >= 0; i-- {
		t := c<<L + x[i];
		c, x[i] = t%d, t/d;
	}

	return Normalize(x), c;
}


func (x *Natural) String(base uint) string {
	if x.IsZero() {
		return "0";
	}
	
	// allocate string
	assert(2 <= base && base <= 16);
	n := (x.Log2() + 1) / Log2(Digit(base)) + 1;  // TODO why the +1?
	s := new([]byte, n);

	// convert

	// don't destroy x, make a copy
	t := new(Natural, len(x));
	Or1(t, x, 0);  // copy x
	
	i := n;
	for !t.IsZero() {
		i--;
		var d Digit;
		t, d = t.DivMod1(Digit(base));
		s[i] = "0123456789abcdef"[d];
	};

	return string(s[i : n]);
}


export func MulRange(a, b Digit) *Natural {
	switch {
	case a > b: return Nat(1);
	case a == b: return Nat(a);
	case a + 1 == b: return Nat(a).Mul(Nat(b));
	}
	m := (a + b)>>1;
	assert(a <= m && m < b);
	return MulRange(a, m).Mul(MulRange(m + 1, b));
}


export func Fact(n Digit) *Natural {
	// Using MulRange() instead of the basic for-loop
	// lead to faster factorial computation.
	return MulRange(2, n);
}


func (x *Natural) Gcd(y *Natural) *Natural {
	// Euclidean algorithm.
	for !y.IsZero() {
		x, y = y, x.Mod(y);
	}
	return x;
}


func HexValue(ch byte) uint {
	d := uint(1 << LogH);
	switch {
	case '0' <= ch && ch <= '9': d = uint(ch - '0');
	case 'a' <= ch && ch <= 'f': d = uint(ch - 'a') + 10;
	case 'A' <= ch && ch <= 'F': d = uint(ch - 'A') + 10;
	}
	return d;
}


// TODO auto-detect base if base argument is 0
export func NatFromString(s string, base uint) *Natural {
	x := NatZero;
	for i := 0; i < len(s); i++ {
		d := HexValue(s[i]);
		if d < base {
			x = x.MulAdd1(Digit(base), Digit(d));
		} else {
			break;
		}
	}
	return x;
}


// ----------------------------------------------------------------------------
// Algorithms

export type T interface {
	IsZero() bool;
	Mod(y T) bool;
}

export func Gcd(x, y T) T {
	// Euclidean algorithm.
	for !y.IsZero() {
		x, y = y, x.Mod(y);
	}
	return x;
}


// ----------------------------------------------------------------------------
// Integer numbers

export type Integer struct {
	sign bool;
	mant *Natural;
}


export func Int(x int64) *Integer {
	return nil;
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


func (x *Integer) Quo(y *Integer) *Integer {
	// x / y == x / y
	// x / (-y) == -(x / y)
	// (-x) / y == -(x / y)
	// (-x) / (-y) == x / y
	return &Integer{x.sign != y.sign, x.mant.Div(y.mant)};
}


func (x *Integer) Rem(y *Integer) *Integer {
	// x % y == x % y
	// x % (-y) == x % y
	// (-x) % y == -(x % y)
	// (-x) % (-y) == -(x % y)
	return &Integer{y.sign, x.mant.Mod(y.mant)};
}


func (x *Integer) QuoRem(y *Integer) (*Integer, *Integer) {
	q, r := x.mant.DivMod(y.mant);
	return &Integer{x.sign != y.sign, q}, &Integer{y.sign, q};
}


func (x *Integer) Div(y *Integer) *Integer {
	q, r := x.mant.DivMod(y.mant);
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


func (x *Integer) String(base uint) string {
	if x.mant.IsZero() {
		return "0";
	}
	var s string;
	if x.sign {
		s = "-";
	}
	return s + x.mant.String(base);
}

	
export func IntFromString(s string, base uint) *Integer {
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


func (x *Rational) Normalize() *Rational {
	f := x.a.mant.Gcd(x.b.mant);
	x.a.mant = x.a.mant.Div(f);
	x.b.mant = x.b.mant.Div(f);
	return x;
}


func Rat(a, b *Integer) *Rational {
	return (&Rational{a, b}).Normalize();
}


func (x *Rational) Add(y *Rational) *Rational {
	return Rat((x.a.Mul(y.b)).Add(x.b.Mul(y.a)), x.b.Mul(y.b));
}


func (x *Rational) Sub(y *Rational) *Rational {
	return Rat((x.a.Mul(y.b)).Sub(x.b.Mul(y.a)), x.b.Mul(y.b));
}


func (x *Rational) Mul(y *Rational) *Rational {
	return Rat(x.a.Mul(y.a), x.b.Mul(y.b));
}


func (x *Rational) Div(y *Rational) *Rational {
	return Rat(x.a.Mul(y.b), x.b.Mul(y.a));
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
