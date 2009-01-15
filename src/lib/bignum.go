// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bignum

// A package for arbitrary precision arithmethic.
// It implements the following numeric types:
//
// - Natural	unsigned integer numbers
// - Integer	signed integer numbers
// - Rational	rational numbers

import "fmt"


// ----------------------------------------------------------------------------
// Internal representation
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
// The operations for all other numeric types are implemented on top of
// the operations for natural numbers.
//
// The base B is chosen as large as possible on a given platform but there
// are a few constraints besides the size of the largest unsigned integer
// type available:
//
// 1) To improve conversion speed between strings and numbers, the base B
//    is chosen such that division and multiplication by 10 (for decimal
//    string representation) can be done without using extended-precision
//    arithmetic. This makes addition, subtraction, and conversion routines
//    twice as fast. It requires a "buffer" of 4 bits per operand digit.
//    That is, the size of B must be 4 bits smaller then the size of the
//    type (Digit) in which these operations are performed. Having this
//    buffer also allows for trivial (single-bit) carry computation in
//    addition and subtraction (optimization suggested by Ken Thompson).
//
// 2) Long division requires extended-precision (2-digit) division per digit.
//    Instead of sacrificing the largest base type for all other operations,
//    for division the operands are unpacked into "half-digits", and the
//    results are packed again. For faster unpacking/packing, the base size
//    in bits must be even.

export type (
	Digit  uint64;
	Digit2 uint32;  // half-digits for division
)


const _LogW = 64;
const _LogH = 4;  // bits for a hex digit (= "small" number)
const _LogB = _LogW - _LogH;  // largest bit-width available


const (
	// half-digits
	_W2 = _LogB / 2;  // width
	_B2 = 1 << _W2;   // base
	_M2 = _B2 - 1;    // mask

	// full digits
	_W = _W2 * 2;     // width
	_B = 1 << _W;     // base
	_M = _B - 1;      // mask
)


// ----------------------------------------------------------------------------
// Support functions

func assert(p bool) {
	if !p {
		panic("assert failed");
	}
}


func isSmall(x Digit) bool {
	return x < 1<<_LogH;
}


export func Dump(x []Digit) {
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
// c      carry
// x, y   operands
// z      result
// n, m   len(x), len(y)


export type Natural []Digit;

var (
	natZero Natural = Natural{};
	natOne Natural = Natural{1};
	natTwo Natural = Natural{2};
	natTen Natural = Natural{10};
)


// Creation

export func Nat(x uint) Natural {
	switch x {
	case 0: return natZero;
	case 1: return natOne;
	case 2: return natTwo;
	case 10: return natTen;
	}
	assert(Digit(x) < _B);
	return Natural{Digit(x)};
}


// Predicates

func (x Natural) IsOdd() bool {
	return len(x) > 0 && x[0]&1 != 0;
}


func (x Natural) IsZero() bool {
	return len(x) == 0;
}


// Operations

func normalize(x Natural) Natural {
	n := len(x);
	for n > 0 && x[n - 1] == 0 { n-- }
	if n < len(x) {
		x = x[0 : n];  // trim leading 0's
	}
	return x;
}


func (x Natural) Add(y Natural) Natural {
	n := len(x);
	m := len(y);
	if n < m {
		return y.Add(x);
	}

	c := Digit(0);
	z := make(Natural, n + 1);
	i := 0;
	for i < m {
		t := c + x[i] + y[i];
		c, z[i] = t>>_W, t&_M;
		i++;
	}
	for i < n {
		t := c + x[i];
		c, z[i] = t>>_W, t&_M;
		i++;
	}
	if c != 0 {
		z[i] = c;
		i++;
	}

	return z[0 : i];
}


func (x Natural) Sub(y Natural) Natural {
	n := len(x);
	m := len(y);
	if n < m {
		panic("underflow")
	}

	c := Digit(0);
	z := make(Natural, n);
	i := 0;
	for i < m {
		t := c + x[i] - y[i];
		c, z[i] = Digit(int64(t)>>_W), t&_M;  // requires arithmetic shift!
		i++;
	}
	for i < n {
		t := c + x[i];
		c, z[i] = Digit(int64(t)>>_W), t&_M;  // requires arithmetic shift!
		i++;
	}
	for i > 0 && z[i - 1] == 0 {  // normalize
		i--;
	}

	return z[0 : i];
}


// Returns c = x*y div B, z = x*y mod B.
func mul11(x, y Digit) (Digit, Digit) {
	// Split x and y into 2 sub-digits each,
	// multiply the digits separately while avoiding overflow,
	// and return the product as two separate digits.

	// This code also works for non-even bit widths W
	// which is why there are separate constants below
	// for half-digits.
	const W2 = (_W + 1)/2;
	const DW = W2*2 - _W;  // 0 or 1
	const B2  = 1<<W2;
	const M2  = _B2 - 1;

	// split x and y into sub-digits
	// x = (x1*B2 + x0)
	// y = (y1*B2 + y0)
	x1, x0 := x>>W2, x&M2;
	y1, y0 := y>>W2, y&M2;

	// x*y = t2*B2^2 + t1*B2 + t0
	t0 := x0*y0;
	t1 := x1*y0 + x0*y1;
	t2 := x1*y1;

	// compute the result digits but avoid overflow
	// z = z1*B + z0 = x*y
	z0 := (t1<<W2 + t0)&_M;
	z1 := t2<<DW + (t1 + t0>>W2)>>(_W-W2);

	return z1, z0;
}


func (x Natural) Mul(y Natural) Natural {
	n := len(x);
	m := len(y);

	z := make(Natural, n + m);
	for j := 0; j < m; j++ {
		d := y[j];
		if d != 0 {
			c := Digit(0);
			for i := 0; i < n; i++ {
				// z[i+j] += c + x[i]*d;
				z1, z0 := mul11(x[i], d);
				t := c + z[i+j] + z0;
				c, z[i+j] = t>>_W, t&_M;
				c += z1;
			}
			z[n+j] = c;
		}
	}

	return normalize(z);
}


// DivMod needs multi-precision division which is not available if Digit
// is already using the largest uint size. Instead, unpack each operand
// into operands with twice as many digits of half the size (Digit2), do
// DivMod, and then pack the results again.

func unpack(x Natural) []Digit2 {
	n := len(x);
	z := make([]Digit2, n*2 + 1);  // add space for extra digit (used by DivMod)
	for i := 0; i < n; i++ {
		t := x[i];
		z[i*2] = Digit2(t & _M2);
		z[i*2 + 1] = Digit2(t >> _W2 & _M2);
	}

	// normalize result
	k := 2*n;
	for k > 0 && z[k - 1] == 0 { k-- }
	return z[0 : k];  // trim leading 0's
}


func pack(x []Digit2) Natural {
	n := (len(x) + 1) / 2;
	z := make(Natural, n);
	if len(x) & 1 == 1 {
		// handle odd len(x)
		n--;
		z[n] = Digit(x[n*2]);
	}
	for i := 0; i < n; i++ {
		z[i] = Digit(x[i*2 + 1]) << _W2 | Digit(x[i*2]);
	}
	return normalize(z);
}


func mul1(z, x []Digit2, y Digit2) Digit2 {
	n := len(x);
	c := Digit(0);
	f := Digit(y);
	for i := 0; i < n; i++ {
		t := c + Digit(x[i])*f;
		c, z[i] = t>>_W2, Digit2(t&_M2);
	}
	return Digit2(c);
}


func div1(z, x []Digit2, y Digit2) Digit2 {
	n := len(x);
	c := Digit(0);
	d := Digit(y);
	for i := n-1; i >= 0; i-- {
		t := c*_B2 + Digit(x[i]);
		c, z[i] = t%d, Digit2(t/d);
	}
	return Digit2(c);
}


// divmod returns q and r with x = y*q + r and 0 <= r < y.
// x and y are destroyed in the process.
//
// The algorithm used here is based on 1). 2) describes the same algorithm
// in C. A discussion and summary of the relevant theorems can be found in
// 3). 3) also describes an easier way to obtain the trial digit - however
// it relies on tripple-precision arithmetic which is why Knuth's method is
// used here.
//
// 1) D. Knuth, "The Art of Computer Programming. Volume 2. Seminumerical
//    Algorithms." Addison-Wesley, Reading, 1969.
//    (Algorithm D, Sec. 4.3.1)
//
// 2) Henry S. Warren, Jr., "A Hacker's Delight". Addison-Wesley, 2003.
//    (9-2 Multiword Division, p.140ff)
//
// 3) P. Brinch Hansen, Multiple-length division revisited: A tour of the
//    minefield. "Software - Practice and Experience 24", (June 1994),
//    579-601. John Wiley & Sons, Ltd.

func divmod(x, y []Digit2) ([]Digit2, []Digit2) {
	n := len(x);
	m := len(y);
	if m == 0 {
		panic("division by zero");
	}
	assert(n+1 <= cap(x));  // space for one extra digit
	x = x[0 : n + 1];
	assert(x[n] == 0);

	if m == 1 {
		// division by single digit
		// result is shifted left by 1 in place!
		x[0] = div1(x[1 : n+1], x[0 : n], y[0]);

	} else if m > n {
		// y > x => quotient = 0, remainder = x
		// TODO in this case we shouldn't even unpack x and y
		m = n;

	} else {
		// general case
		assert(2 <= m && m <= n);

		// normalize x and y
		// TODO Instead of multiplying, it would be sufficient to
		//      shift y such that the normalization condition is
		//      satisfied (as done in "Hacker's Delight").
		f := _B2 / (Digit(y[m-1]) + 1);
		if f != 1 {
			mul1(x, x, Digit2(f));
			mul1(y, y, Digit2(f));
		}
		assert(_B2/2 <= y[m-1] && y[m-1] < _B2);  // incorrect scaling

		y1, y2 := Digit(y[m-1]), Digit(y[m-2]);
		d2 := Digit(y1)<<_W2 + Digit(y2);
		for i := n-m; i >= 0; i-- {
			k := i+m;

			// compute trial digit (Knuth)
			var q Digit;
			{	x0, x1, x2 := Digit(x[k]), Digit(x[k-1]), Digit(x[k-2]);
				if x0 != y1 {
					q = (x0<<_W2 + x1)/y1;
				} else {
					q = _B2 - 1;
				}
				for y2*q > (x0<<_W2 + x1 - y1*q)<<_W2 + x2 {
					q--
				}
			}

			// subtract y*q
			c := Digit(0);
			for j := 0; j < m; j++ {
				t := c + Digit(x[i+j]) - Digit(y[j])*q;
				c, x[i+j] = Digit(int64(t) >> _W2), Digit2(t & _M2);  // requires arithmetic shift!
			}

			// correct if trial digit was too large
			if c + Digit(x[k]) != 0 {
				// add y
				c := Digit(0);
				for j := 0; j < m; j++ {
					t := c + Digit(x[i+j]) + Digit(y[j]);
					c, x[i+j] = t >> _W2, Digit2(t & _M2)
				}
				assert(c + Digit(x[k]) == 0);
				// correct trial digit
				q--;
			}

			x[k] = Digit2(q);
		}

		// undo normalization for remainder
		if f != 1 {
			c := div1(x[0 : m], x[0 : m], Digit2(f));
			assert(c == 0);
		}
	}

	return x[m : n+1], x[0 : m];
}


func (x Natural) Div(y Natural) Natural {
	q, r := divmod(unpack(x), unpack(y));
	return pack(q);
}


func (x Natural) Mod(y Natural) Natural {
	q, r := divmod(unpack(x), unpack(y));
	return pack(r);
}


func (x Natural) DivMod(y Natural) (Natural, Natural) {
	q, r := divmod(unpack(x), unpack(y));
	return pack(q), pack(r);
}


func shl(z, x []Digit, s uint) Digit {
	assert(s <= _W);
	n := len(x);
	c := Digit(0);
	for i := 0; i < n; i++ {
		c, z[i] = x[i] >> (_W-s), x[i] << s & _M | c;
	}
	return c;
}


func (x Natural) Shl(s uint) Natural {
	n := uint(len(x));
	m := n + s/_W;
	z := make(Natural, m+1);

	z[m] = shl(z[m-n : m], x, s%_W);

	return normalize(z);
}


func shr(z, x []Digit, s uint) Digit {
	assert(s <= _W);
	n := len(x);
	c := Digit(0);
	for i := n - 1; i >= 0; i-- {
		c, z[i] = x[i] << (_W-s) & _M, x[i] >> s | c;
	}
	return c;
}


func (x Natural) Shr(s uint) Natural {
	n := uint(len(x));
	m := n - s/_W;
	if m > n {  // check for underflow
		m = 0;
	}
	z := make(Natural, m);

	shr(z, x[n-m : n], s%_W);

	return normalize(z);
}


func (x Natural) And(y Natural) Natural {
	n := len(x);
	m := len(y);
	if n < m {
		return y.And(x);
	}

	z := make(Natural, m);
	for i := 0; i < m; i++ {
		z[i] = x[i] & y[i];
	}
	// upper bits are 0

	return normalize(z);
}


func copy(z, x []Digit) {
	for i, e := range x {
		z[i] = e
	}
}


func (x Natural) Or(y Natural) Natural {
	n := len(x);
	m := len(y);
	if n < m {
		return y.Or(x);
	}

	z := make(Natural, n);
	for i := 0; i < m; i++ {
		z[i] = x[i] | y[i];
	}
	copy(z[m : n], x[m : n]);

	return z;
}


func (x Natural) Xor(y Natural) Natural {
	n := len(x);
	m := len(y);
	if n < m {
		return y.Xor(x);
	}

	z := make(Natural, n);
	for i := 0; i < m; i++ {
		z[i] = x[i] ^ y[i];
	}
	copy(z[m : n], x[m : n]);

	return normalize(z);
}


func (x Natural) Cmp(y Natural) int {
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


func log2(x Digit) uint {
	assert(x > 0);
	n := uint(0);
	for x > 0 {
		x >>= 1;
		n++;
	}
	return n - 1;
}


func (x Natural) Log2() uint {
	n := len(x);
	if n > 0 {
		return (uint(n) - 1)*_W + log2(x[n - 1]);
	}
	panic("Log2(0)");
}


// Computes x = x div d in place (modifies x) for "small" d's.
// Returns updated x and x mod d.
func divmod1(x Natural, d Digit) (Natural, Digit) {
	assert(0 < d && isSmall(d - 1));

	c := Digit(0);
	for i := len(x) - 1; i >= 0; i-- {
		t := c<<_W + x[i];
		c, x[i] = t%d, t/d;
	}

	return normalize(x), c;
}


func (x Natural) ToString(base uint) string {
	if len(x) == 0 {
		return "0";
	}

	// allocate buffer for conversion
	assert(2 <= base && base <= 16);
	n := (x.Log2() + 1) / log2(Digit(base)) + 1;  // +1: round up
	s := make([]byte, n);

	// don't destroy x
	t := make(Natural, len(x));
	copy(t, x);

	// convert
	i := n;
	for !t.IsZero() {
		i--;
		var d Digit;
		t, d = divmod1(t, Digit(base));
		s[i] = "0123456789abcdef"[d];
	};

	return string(s[i : n]);
}


func (x Natural) String() string {
	return x.ToString(10);
}


func fmtbase(c int) uint {
	switch c {
	case 'b': return 2;
	case 'o': return 8;
	case 'x': return 16;
	}
	return 10;
}


func (x Natural) Format(h fmt.Formatter, c int) {
	fmt.Fprintf(h, "%s", x.ToString(fmtbase(c)));
}


func hexvalue(ch byte) uint {
	d := uint(1 << _LogH);
	switch {
	case '0' <= ch && ch <= '9': d = uint(ch - '0');
	case 'a' <= ch && ch <= 'f': d = uint(ch - 'a') + 10;
	case 'A' <= ch && ch <= 'F': d = uint(ch - 'A') + 10;
	}
	return d;
}


// Computes x = x*d + c for "small" d's.
func muladd1(x Natural, d, c Digit) Natural {
	assert(isSmall(d-1) && isSmall(c));
	n := len(x);
	z := make(Natural, n + 1);

	for i := 0; i < n; i++ {
		t := c + x[i]*d;
		c, z[i] = t>>_W, t&_M;
	}
	z[n] = c;

	return normalize(z);
}


// Determines base (octal, decimal, hexadecimal) if base == 0.
// Returns the number and base.
export func NatFromString(s string, base uint, slen *int) (Natural, uint) {
	// determine base if necessary
	i, n := 0, len(s);
	if base == 0 {
		base = 10;
		if n > 0 && s[0] == '0' {
			if n > 1 && (s[1] == 'x' || s[1] == 'X') {
				base, i = 16, 2;
			} else {
				base, i = 8, 1;
			}
		}
	}

	// convert string
	assert(2 <= base && base <= 16);
	x := Nat(0);
	for ; i < n; i++ {
		d := hexvalue(s[i]);
		if d < base {
			x = muladd1(x, Digit(base), Digit(d));
		} else {
			break;
		}
	}

	// provide number of string bytes consumed if necessary
	if slen != nil {
		*slen = i;
	}

	return x, base;
}


// Natural number functions

func pop1(x Digit) uint {
	n := uint(0);
	for x != 0 {
		x &= x-1;
		n++;
	}
	return n;
}


func (x Natural) Pop() uint {
	n := uint(0);
	for i := len(x) - 1; i >= 0; i-- {
		n += pop1(x[i]);
	}
	return n;
}


func (xp Natural) Pow(n uint) Natural {
	z := Nat(1);
	x := xp;
	for n > 0 {
		// z * x^n == x^n0
		if n&1 == 1 {
			z = z.Mul(x);
		}
		x, n = x.Mul(x), n/2;
	}
	return z;
}


export func MulRange(a, b uint) Natural {
	switch {
	case a > b: return Nat(1);
	case a == b: return Nat(a);
	case a + 1 == b: return Nat(a).Mul(Nat(b));
	}
	m := (a + b)>>1;
	assert(a <= m && m < b);
	return MulRange(a, m).Mul(MulRange(m + 1, b));
}


export func Fact(n uint) Natural {
	// Using MulRange() instead of the basic for-loop
	// lead to faster factorial computation.
	return MulRange(2, n);
}


export func Binomial(n, k uint) Natural {
	return MulRange(n-k+1, n).Div(MulRange(1, k));
}


func (xp Natural) Gcd(y Natural) Natural {
	// Euclidean algorithm.
	x := xp;
	for !y.IsZero() {
		x, y = y, x.Mod(y);
	}
	return x;
}


// ----------------------------------------------------------------------------
// Integer numbers
//
// Integers are normalized if the mantissa is normalized and the sign is
// false for mant == 0. Use MakeInt to create normalized Integers.

export type Integer struct {
	sign bool;
	mant Natural;
}


// Creation

export func MakeInt(sign bool, mant Natural) *Integer {
	if mant.IsZero() {
		sign = false;  // normalize
	}
	return &Integer{sign, mant};
}


export func Int(x int) *Integer {
	sign := false;
	var ux uint;
	if x < 0 {
		sign = true;
		if -x == x {
			// smallest negative integer
			t := ^0;
			ux = ^(uint(t) >> 1);
		} else {
			ux = uint(-x);
		}
	} else {
		ux = uint(x);
	}
	return MakeInt(sign, Nat(ux));
}


// Predicates

func (x *Integer) IsOdd() bool {
	return x.mant.IsOdd();
}


func (x *Integer) IsZero() bool {
	return x.mant.IsZero();
}


func (x *Integer) IsNeg() bool {
	return x.sign && !x.mant.IsZero()
}


func (x *Integer) IsPos() bool {
	return !x.sign && !x.mant.IsZero()
}


// Operations

func (x *Integer) Neg() *Integer {
	return MakeInt(!x.sign, x.mant);
}


func (x *Integer) Add(y *Integer) *Integer {
	var z *Integer;
	if x.sign == y.sign {
		// x + y == x + y
		// (-x) + (-y) == -(x + y)
		z = MakeInt(x.sign, x.mant.Add(y.mant));
	} else {
		// x + (-y) == x - y == -(y - x)
		// (-x) + y == y - x == -(x - y)
		if x.mant.Cmp(y.mant) >= 0 {
			z = MakeInt(false, x.mant.Sub(y.mant));
		} else {
			z = MakeInt(true, y.mant.Sub(x.mant));
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
		z = MakeInt(false, x.mant.Add(y.mant));
	} else {
		// x - y == x - y == -(y - x)
		// (-x) - (-y) == y - x == -(x - y)
		if x.mant.Cmp(y.mant) >= 0 {
			z = MakeInt(false, x.mant.Sub(y.mant));
		} else {
			z = MakeInt(true, y.mant.Sub(x.mant));
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
	return MakeInt(x.sign != y.sign, x.mant.Mul(y.mant));
}


func (x *Integer) MulNat(y Natural) *Integer {
	// x * y == x * y
	// (-x) * y == -(x * y)
	return MakeInt(x.sign, x.mant.Mul(y));
}


// Quo and Rem implement T-division and modulus (like C99):
//
//   q = x.Quo(y) = trunc(x/y)  (truncation towards zero)
//   r = x.Rem(y) = x - y*q
//
// ( Daan Leijen, "Division and Modulus for Computer Scientists". )

func (x *Integer) Quo(y *Integer) *Integer {
	// x / y == x / y
	// x / (-y) == -(x / y)
	// (-x) / y == -(x / y)
	// (-x) / (-y) == x / y
	return MakeInt(x.sign != y.sign, x.mant.Div(y.mant));
}


func (x *Integer) Rem(y *Integer) *Integer {
	// x % y == x % y
	// x % (-y) == x % y
	// (-x) % y == -(x % y)
	// (-x) % (-y) == -(x % y)
	return MakeInt(x.sign, x.mant.Mod(y.mant));
}


func (x *Integer) QuoRem(y *Integer) (*Integer, *Integer) {
	q, r := x.mant.DivMod(y.mant);
	return MakeInt(x.sign != y.sign, q), MakeInt(x.sign, r);
}


// Div and Mod implement Euclidian division and modulus:
//
//   d = x.Div(y)
//   m = x.Mod(y) with: 0 <= m < |d| and: y = x*d + m
//
// ( Raymond T. Boute, The Euclidian definition of the functions
//   div and mod. "ACM Transactions on Programming Languages and
//   Systems (TOPLAS)", 14(2):127-144, New York, NY, USA, 4/1992.
//   ACM press. )


func (x *Integer) Div(y *Integer) *Integer {
	q, r := x.QuoRem(y);
	if r.IsNeg() {
		if y.IsPos() {
			q = q.Sub(Int(1));
		} else {
			q = q.Add(Int(1));
		}
	}
	return q;
}


func (x *Integer) Mod(y *Integer) *Integer {
	r := x.Rem(y);
	if r.IsNeg() {
		if y.IsPos() {
			r = r.Add(y);
		} else {
			r = r.Sub(y);
		}
	}
	return r;
}


func (x *Integer) DivMod(y *Integer) (*Integer, *Integer) {
	q, r := x.QuoRem(y);
	if r.IsNeg() {
		if y.IsPos() {
			q = q.Sub(Int(1));
			r = r.Add(y);
		} else {
			q = q.Add(Int(1));
			r = r.Sub(y);
		}
	}
	return q, r;
}


func (x *Integer) Shl(s uint) *Integer {
	return MakeInt(x.sign, x.mant.Shl(s));
}


func (x *Integer) Shr(s uint) *Integer {
	z := MakeInt(x.sign, x.mant.Shr(s));
	if x.IsNeg() {
		panic("UNIMPLEMENTED Integer.Shr() of negative values");
	}
	return z;
}


func (x *Integer) And(y *Integer) *Integer {
	var z *Integer;
	if !x.sign && !y.sign {
		z = MakeInt(false, x.mant.And(y.mant));
	} else {
		panic("UNIMPLEMENTED Integer.And() of negative values");
	}
	return z;
}


func (x *Integer) Or(y *Integer) *Integer {
	var z *Integer;
	if !x.sign && !y.sign {
		z = MakeInt(false, x.mant.Or(y.mant));
	} else {
		panic("UNIMPLEMENTED Integer.Or() of negative values");
	}
	return z;
}


func (x *Integer) Xor(y *Integer) *Integer {
	var z *Integer;
	if !x.sign && !y.sign {
		z = MakeInt(false, x.mant.Xor(y.mant));
	} else {
		panic("UNIMPLEMENTED Integer.Xor() of negative values");
	}
	return z;
}


func (x *Integer) Cmp(y *Integer) int {
	// x cmp y == x cmp y
	// x cmp (-y) == x
	// (-x) cmp y == y
	// (-x) cmp (-y) == -(x cmp y)
	var r int;
	switch {
	case x.sign == y.sign:
		r = x.mant.Cmp(y.mant);
		if x.sign {
			r = -r;
		}
	case x.sign: r = -1;
	case y.sign: r = 1;
	}
	return r;
}


func (x *Integer) ToString(base uint) string {
	if x.mant.IsZero() {
		return "0";
	}
	var s string;
	if x.sign {
		s = "-";
	}
	return s + x.mant.ToString(base);
}


func (x *Integer) String() string {
	return x.ToString(10);
}


func (x *Integer) Format(h fmt.Formatter, c int) {
	fmt.Fprintf(h, "%s", x.ToString(fmtbase(c)));
}


// Determines base (octal, decimal, hexadecimal) if base == 0.
// Returns the number and base.
export func IntFromString(s string, base uint, slen *int) (*Integer, uint) {
	// get sign, if any
	sign := false;
	if len(s) > 0 && (s[0] == '-' || s[0] == '+') {
		sign = s[0] == '-';
		s = s[1 : len(s)];
	}

	var mant Natural;
	mant, base = NatFromString(s, base, slen);

	// correct slen if necessary
	if slen != nil && sign {
		*slen++;
	}

	return MakeInt(sign, mant), base;
}


// ----------------------------------------------------------------------------
// Rational numbers

export type Rational struct {
	a *Integer;  // numerator
	b Natural;  // denominator
}


// Creation

export func MakeRat(a *Integer, b Natural) *Rational {
	f := a.mant.Gcd(b);  // f > 0
	if f.Cmp(Nat(1)) != 0 {
		a = MakeInt(a.sign, a.mant.Div(f));
		b = b.Div(f);
	}
	return &Rational{a, b};
}


export func Rat(a0 int, b0 int) *Rational {
	a, b := Int(a0), Int(b0);
	if b.sign {
		a = a.Neg();
	}
	return MakeRat(a, b.mant);
}


// Predicates

func (x *Rational) IsZero() bool {
	return x.a.IsZero();
}


func (x *Rational) IsNeg() bool {
	return x.a.IsNeg();
}


func (x *Rational) IsPos() bool {
	return x.a.IsPos();
}


func (x *Rational) IsInt() bool {
	return x.b.Cmp(Nat(1)) == 0;
}


// Operations

func (x *Rational) Neg() *Rational {
	return MakeRat(x.a.Neg(), x.b);
}


func (x *Rational) Add(y *Rational) *Rational {
	return MakeRat((x.a.MulNat(y.b)).Add(y.a.MulNat(x.b)), x.b.Mul(y.b));
}


func (x *Rational) Sub(y *Rational) *Rational {
	return MakeRat((x.a.MulNat(y.b)).Sub(y.a.MulNat(x.b)), x.b.Mul(y.b));
}


func (x *Rational) Mul(y *Rational) *Rational {
	return MakeRat(x.a.Mul(y.a), x.b.Mul(y.b));
}


func (x *Rational) Quo(y *Rational) *Rational {
	a := x.a.MulNat(y.b);
	b := y.a.MulNat(x.b);
	if b.IsNeg() {
		a = a.Neg();
	}
	return MakeRat(a, b.mant);
}


func (x *Rational) Cmp(y *Rational) int {
	return (x.a.MulNat(y.b)).Cmp(y.a.MulNat(x.b));
}


func (x *Rational) ToString(base uint) string {
	s := x.a.ToString(base);
	if !x.IsInt() {
		s += "/" + x.b.ToString(base);
	}
	return s;
}


func (x *Rational) String() string {
	return x.ToString(10);
}


func (x *Rational) Format(h fmt.Formatter, c int) {
	fmt.Fprintf(h, "%s", x.ToString(fmtbase(c)));
}


// Determines base (octal, decimal, hexadecimal) if base == 0.
// Returns the number and base of the nominator.
export func RatFromString(s string, base uint, slen *int) (*Rational, uint) {
	// read nominator
	var alen, blen int;
	a, abase := IntFromString(s, base, &alen);
	b := Nat(1);

	// read denominator or fraction, if any
	if alen < len(s) {
		ch := s[alen];
		if ch == '/' {
			alen++;
			b, base = NatFromString(s[alen : len(s)], base, &blen);
		} else if ch == '.' {
			alen++;
			b, base = NatFromString(s[alen : len(s)], abase, &blen);
			assert(base == abase);
			f := Nat(base).Pow(uint(blen));
			a = MakeInt(a.sign, a.mant.Mul(f).Add(b));
			b = f;
		}
	}

	// provide number of string bytes consumed if necessary
	if slen != nil {
		*slen = alen + blen;
	}

	return MakeRat(a, b), abase;
}
