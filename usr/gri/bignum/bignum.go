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
	L3 = LogB / 3;
	B3 = 1 << L3;
	M3 = B3 - 1;

	L2 = L3 * 2;
	B2 = 1 << L2;
	M2 = B2 - 1;

	L = L3 * 3;
	B = 1 << L;
	M = B - 1;
)


type (
	Digit3 uint32;
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


export func Dump3(x *[]Digit3) {
	print("[", len(x), "]");
	for i := len(x) - 1; i >= 0; i-- {
		print(" ", x[i]);
	}
	println();
}


// ----------------------------------------------------------------------------
// Natural numbers

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


func Normalize3(x *[]Digit3) *[]Digit3 {
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

	c := Digit(0);
	for i := 0; i < m; i++ { c, z[i] = Split(x[i] + y[i] + c); }
	for i := m; i < n; i++ { c, z[i] = Split(x[i] + c); }
	z[n] = c;

	return Normalize(z);
}


func (x *Natural) Sub(y *Natural) *Natural {
	n := len(x);
	m := len(y);
	assert(n >= m);
	z := new(Natural, n);

	c := Digit(0);
	for i := 0; i < m; i++ { c, z[i] = Split(x[i] - y[i] + c); }  // TODO verify asr!!!
	for i := m; i < n; i++ { c, z[i] = Split(x[i] + c); }
	assert(c == 0);  // x.Sub(y) must be called with x >= y

	return Normalize(z);
}


// Computes x = x*a + c (in place) for "small" a's.
func (x* Natural) MulAdd1(a, c Digit) *Natural {
	assert(IsSmall(a-1) && IsSmall(c));
	n := len(x);
	z := new(Natural, n + 1);

	for i := 0; i < n; i++ { c, z[i] = Split(x[i]*a + c); }
	z[n] = c;

	return Normalize(z);
}


// Returns c = x*y div B, z = x*y mod B.
func Mul1(x, y Digit) (Digit, Digit) {
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
			c := Digit(0);
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


func Shl1(x, c Digit, s uint) (Digit, Digit) {
	assert(s <= LogB);
	return x >> (LogB - s), x << s & M | c
}


func Shr1(x, c Digit, s uint) (Digit, Digit) {
	assert(s <= LogB);
	return x << (LogB - s) & M, x >> s | c
}


func (x *Natural) Shl(s uint) *Natural {
	n := len(x);
	si := int(s / LogB);
	s = s % LogB;
	z := new(Natural, n + si + 1);
	
	c := Digit(0);
	for i := 0; i < n; i++ { c, z[i+si] = Shl1(x[i], c, s); }
	z[n+si] = c;
	
	return Normalize(z);
}


func (x *Natural) Shr(s uint) *Natural {
	n := len(x);
	si := int(s / LogB);
	if si >= n { si = n; }
	s = s % LogB;
	assert(si <= n);
	z := new(Natural, n - si);
	
	c := Digit(0);
	for i := n - 1; i >= si; i-- { c, z[i-si] = Shr1(x[i], c, s); }
	
	return Normalize(z);
}


// DivMod needs multi-precision division which is not available if Digit
// is already using the largest uint size. Split base before division,
// and merge again after. Each Digit is split into 3 Digit3's.

func SplitBase(x *Natural) *[]Digit3 {
	// TODO Use Log() for better result - don't need Normalize3 at the end!
	n := len(x);
	z := new([]Digit3, n*3 + 1);  // add space for extra digit (used by DivMod)
	for i, j := 0, 0; i < n; i, j = i+1, j+3 {
		t := x[i];
		z[j+0] = Digit3(t >> (L3*0) & M3);
		z[j+1] = Digit3(t >> (L3*1) & M3);
		z[j+2] = Digit3(t >> (L3*2) & M3);
	}
	return Normalize3(z);
}


func MergeBase(x *[]Digit3) *Natural {
	i := len(x);
	j := (i+2)/3;
	z := new(Natural, j);

	switch i%3 {
	case 1: z[j-1] = Digit(x[i-1]); i--; j--;
	case 2: z[j-1] = Digit(x[i-1])<<L3 | Digit(x[i-2]); i -= 2; j--;
	case 0:
	}
	
	for i >= 3 {
		z[j-1] = ((Digit(x[i-1])<<L3) | Digit(x[i-2]))<<L3 | Digit(x[i-3]);
		i -= 3;
		j--;
	}
	assert(j == 0);

	return Normalize(z);
}


func Split3(x Digit) (Digit, Digit3) {
	return uint64(int64(x)>>L3), Digit3(x&M3)
}


func Product(x *[]Digit3, y Digit) {
	n := len(x);
	c := Digit(0);
	for i := 0; i < n; i++ { c, x[i] = Split3(Digit(x[i])*y + c) }
	assert(c == 0);
}


func Quotient(x *[]Digit3, y Digit) {
	n := len(x);
	c := Digit(0);
	for i := n-1; i >= 0; i-- {
		t := c*B3 + Digit(x[i]);
		c, x[i] = t%y, Digit3(t/y);
	}
	assert(c == 0);
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

func DivMod(x, y *[]Digit3) (*[]Digit3, *[]Digit3) {
	const b = B3;
	
	n := len(x);
	m := len(y);
	assert(m > 0);  // division by zero
	assert(n+1 <= cap(x));  // space for one extra digit (should it be == ?)
	x = x[0 : n + 1];
	
	if m == 1 {
		// division by single digit
		d := Digit(y[0]);
		c := Digit(0);
		for i := n; i > 0; i-- {
			t := c*b + Digit(x[i-1]);
			c, x[i] = t%d, Digit3(t/d);
		}
		x[0] = Digit3(c);

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
		Product(x, f);
		Product(y, f);
		assert(b/2 <= y[m-1] && y[m-1] < b);  // incorrect scaling
		
		d2 := Digit(y[m-1])*b + Digit(y[m-2]);
		for i := n-m; i >= 0; i-- {
			k := i+m;
			
			// compute trial digit
			r3 := (Digit(x[k])*b + Digit(x[k-1]))*b + Digit(x[k-2]);
			q := r3/d2;
			if q >= b { q = b-1 }
			
			// subtract y*q
			c := Digit(0);
			for j := 0; j < m; j++ {
				c, x[i+j] = Split3(c + Digit(x[i+j]) - Digit(y[j])*q);
			}
			
			// correct if trial digit was too large
			if c + Digit(x[k]) != 0 {
				// add y
				c := Digit(0);
				for j := 0; j < m; j++ {
					c, x[i+j] = Split3(c + Digit(x[i+j]) + Digit(y[j]));
				}
				// correct trial digit
				q--;
			}
			
			x[k] = Digit3(q);
		}
		
		// undo normalization for remainder
		Quotient(x[0 : m], f);
	}

	return x[m : n+1], x[0 : m];
}


func (x *Natural) Div(y *Natural) *Natural {
	q, r := DivMod(SplitBase(x), SplitBase(y));
	return MergeBase(q);
}


func (x *Natural) Mod(y *Natural) *Natural {
	q, r := DivMod(SplitBase(x), SplitBase(y));
	return MergeBase(r);
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


func Log1(x Digit) int {
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

	for i := 0; i < m; i++ { z[i] = x[i] & y[i]; }
	for i := m; i < n; i++ { z[i] = x[i]; }

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

	for i := 0; i < m; i++ { z[i] = x[i] | y[i]; }
	for i := m; i < n; i++ { z[i] = x[i]; }

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

	for i := 0; i < m; i++ { z[i] = x[i] ^ y[i]; }
	for i := m; i < n; i++ { z[i] = x[i]; }

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
func (x *Natural) DivMod1(d Digit) (*Natural, Digit) {
	assert(0 < d && IsSmall(d - 1));

	c := Digit(0);
	for i := len(x) - 1; i >= 0; i-- {
		c = c<<L + x[i];
		x[i] = c/d;
		c %= d;
	}

	return Normalize(x), c;
}


func (x *Natural) String(base Digit) string {
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
		var d Digit;
		x, d = x.DivMod1(base);
		s[i] = hex[d];
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


func HexValue(ch byte) Digit {
	d := Digit(1 << LogH);
	switch {
	case '0' <= ch && ch <= '9': d = Digit(ch - '0');
	case 'a' <= ch && ch <= 'f': d = Digit(ch - 'a') + 10;
	case 'A' <= ch && ch <= 'F': d = Digit(ch - 'A') + 10;
	}
	return d;
}


// TODO auto-detect base if base argument is 0
export func NatFromString(s string, base Digit) *Natural {
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


func (x *Integer) String(base Digit) string {
	if x.mant.IsZero() {
		return "0";
	}
	var s string;
	if x.sign {
		s = "-";
	}
	return s + x.mant.String(base);
}

	
export func IntFromString(s string, base Digit) *Integer {
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
