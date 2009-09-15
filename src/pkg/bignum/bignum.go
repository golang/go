// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A package for arbitrary precision arithmethic.
// It implements the following numeric types:
//
//	- Natural	unsigned integers
//	- Integer	signed integers
//	- Rational	rational numbers
//
package bignum

import (
	"fmt";
)

// TODO(gri) Complete the set of in-place operations.

// ----------------------------------------------------------------------------
// Internal representation
//
// A natural number of the form
//
//   x = x[n-1]*B^(n-1) + x[n-2]*B^(n-2) + ... + x[1]*B + x[0]
//
// with 0 <= x[i] < B and 0 <= i < n is stored in a slice of length n,
// with the digits x[i] as the slice elements.
//
// A natural number is normalized if the slice contains no leading 0 digits.
// During arithmetic operations, denormalized values may occur but are
// always normalized before returning the final result. The normalized
// representation of 0 is the empty slice (length = 0).
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
//    twice as fast. It requires a ``buffer'' of 4 bits per operand digit.
//    That is, the size of B must be 4 bits smaller then the size of the
//    type (digit) in which these operations are performed. Having this
//    buffer also allows for trivial (single-bit) carry computation in
//    addition and subtraction (optimization suggested by Ken Thompson).
//
// 2) Long division requires extended-precision (2-digit) division per digit.
//    Instead of sacrificing the largest base type for all other operations,
//    for division the operands are unpacked into ``half-digits'', and the
//    results are packed again. For faster unpacking/packing, the base size
//    in bits must be even.

type (
	digit  uint64;
	digit2 uint32;  // half-digits for division
)


const (
	logW = 64;  // word width
	logH = 4;  // bits for a hex digit (= small number)
	logB = logW - logH;  // largest bit-width available

	// half-digits
	_W2 = logB / 2;   // width
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


func isSmall(x digit) bool {
	return x < 1<<logH;
}


// For debugging. Keep around.
/*
func dump(x Natural) {
	print("[", len(x), "]");
	for i := len(x) - 1; i >= 0; i-- {
		print(" ", x[i]);
	}
	println();
}
*/


// ----------------------------------------------------------------------------
// Natural numbers

// Natural represents an unsigned integer value of arbitrary precision.
//
type Natural []digit;


// Nat creates a small natural number with value x.
//
func Nat(x uint64) Natural {
	if x == 0 {
		return nil;  // len == 0
	}

	// single-digit values
	// (note: cannot re-use preallocated values because
	//        the in-place operations may overwrite them)
	if x < _B {
		return Natural{digit(x)};
	}

	// compute number of digits required to represent x
	// (this is usually 1 or 2, but the algorithm works
	// for any base)
	n := 0;
	for t := x; t > 0; t >>= _W {
		n++;
	}

	// split x into digits
	z := make(Natural, n);
	for i := 0; i < n; i++ {
		z[i] = digit(x & _M);
		x >>= _W;
	}

	return z;
}


// Value returns the lowest 64bits of x.
//
func (x Natural) Value() uint64 {
	// single-digit values
	n := len(x);
	switch n {
	case 0: return 0;
	case 1: return uint64(x[0]);
	}

	// multi-digit values
	// (this is usually 1 or 2, but the algorithm works
	// for any base)
	z := uint64(0);
	s := uint(0);
	for i := 0; i < n && s < 64; i++ {
		z += uint64(x[i]) << s;
		s += _W;
	}

	return z;
}


// Predicates

// IsEven returns true iff x is divisible by 2.
//
func (x Natural) IsEven() bool {
	return len(x) == 0 || x[0]&1 == 0;
}


// IsOdd returns true iff x is not divisible by 2.
//
func (x Natural) IsOdd() bool {
	return len(x) > 0 && x[0]&1 != 0;
}


// IsZero returns true iff x == 0.
//
func (x Natural) IsZero() bool {
	return len(x) == 0;
}


// Operations
//
// Naming conventions
//
// c      carry
// x, y   operands
// z      result
// n, m   len(x), len(y)

func normalize(x Natural) Natural {
	n := len(x);
	for n > 0 && x[n-1] == 0 { n-- }
	return x[0 : n];
}


// nalloc returns a Natural of n digits. If z is large
// enough, z is resized and returned. Otherwise, a new
// Natural is allocated.
//
func nalloc(z Natural, n int) Natural {
	size := n;
	if size <= 0 {
		size = 4;
	}
	if size <= cap(z) {
		return z[0 : n];
	}
	return make(Natural, n, size);
}


// Nadd sets *zp to the sum x + y.
// *zp may be x or y.
//
func Nadd(zp *Natural, x, y Natural) {
	n := len(x);
	m := len(y);
	if n < m {
		Nadd(zp, y, x);
		return;
	}

	z := nalloc(*zp, n+1);
	c := digit(0);
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
	*zp = z[0 : i]
}


// Add returns the sum z = x + y.
//
func (x Natural) Add(y Natural) Natural {
	var z Natural;
	Nadd(&z, x, y);
	return z;
}


// Nsub sets *zp to the difference x - y for x >= y.
// If x < y, an underflow run-time error occurs (use Cmp to test if x >= y).
// *zp may be x or y.
//
func Nsub(zp *Natural, x, y Natural) {
	n := len(x);
	m := len(y);
	if n < m {
		panic("underflow")
	}

	z := nalloc(*zp, n);
	c := digit(0);
	i := 0;
	for i < m {
		t := c + x[i] - y[i];
		c, z[i] = digit(int64(t)>>_W), t&_M;  // requires arithmetic shift!
		i++;
	}
	for i < n {
		t := c + x[i];
		c, z[i] = digit(int64(t)>>_W), t&_M;  // requires arithmetic shift!
		i++;
	}
	if int64(c) < 0 {
		panic("underflow");
	}
	*zp = normalize(z);
}


// Sub returns the difference x - y for x >= y.
// If x < y, an underflow run-time error occurs (use Cmp to test if x >= y).
//
func (x Natural) Sub(y Natural) Natural {
	var z Natural;
	Nsub(&z, x, y);
	return z;
}


// Returns z1 = (x*y + c) div B, z0 = (x*y + c) mod B.
//
func muladd11(x, y, c digit) (digit, digit) {
	z1, z0 := MulAdd128(uint64(x), uint64(y), uint64(c));
	return digit(z1<<(64 - logB) | z0>>logB), digit(z0&_M);
}


func mul1(z, x Natural, y digit) (c digit) {
	for i := 0; i < len(x); i++ {
		c, z[i] = muladd11(x[i], y, c);
	}
	return;
}


// Nscale sets *z to the scaled value (*z) * d.
//
func Nscale(z *Natural, d uint64) {
	switch {
	case d == 0:
		*z = Nat(0);
		return
	case d == 1:
		return;
	case d >= _B:
		*z = z.Mul1(d);
		return;
	}

	c := mul1(*z, *z, digit(d));

	if c != 0 {
		n := len(*z);
		if n >= cap(*z) {
			zz := make(Natural, n+1);
			for i, d := range *z {
				zz[i] = d;
			}
			*z = zz
		} else {
			*z = (*z)[0 : n+1];
		}
		(*z)[n] = c;
	}
}


// Computes x = x*d + c for small d's.
//
func muladd1(x Natural, d, c digit) Natural {
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


// Mul1 returns the product x * d.
//
func (x Natural) Mul1(d uint64) Natural {
	switch {
	case d == 0: return Nat(0);
	case d == 1: return x;
	case isSmall(digit(d)): muladd1(x, digit(d), 0);
	case d >= _B: return x.Mul(Nat(d));
	}

	z := make(Natural, len(x) + 1);
	c := mul1(z, x, digit(d));
	z[len(x)] = c;
	return normalize(z);
}


// Mul returns the product x * y.
//
func (x Natural) Mul(y Natural) Natural {
	n := len(x);
	m := len(y);
	if n < m {
		return y.Mul(x);
	}

	if m == 0 {
		return Nat(0);
	}

	if m == 1 && y[0] < _B {
		return x.Mul1(uint64(y[0]));
	}

	z := make(Natural, n + m);
	for j := 0; j < m; j++ {
		d := y[j];
		if d != 0 {
			c := digit(0);
			for i := 0; i < n; i++ {
				c, z[i+j] = muladd11(x[i], d, z[i+j] + c);
			}
			z[n+j] = c;
		}
	}

	return normalize(z);
}


// DivMod needs multi-precision division, which is not available if digit
// is already using the largest uint size. Instead, unpack each operand
// into operands with twice as many digits of half the size (digit2), do
// DivMod, and then pack the results again.

func unpack(x Natural) []digit2 {
	n := len(x);
	z := make([]digit2, n*2 + 1);  // add space for extra digit (used by DivMod)
	for i := 0; i < n; i++ {
		t := x[i];
		z[i*2] = digit2(t & _M2);
		z[i*2 + 1] = digit2(t >> _W2 & _M2);
	}

	// normalize result
	k := 2*n;
	for k > 0 && z[k - 1] == 0 { k-- }
	return z[0 : k];  // trim leading 0's
}


func pack(x []digit2) Natural {
	n := (len(x) + 1) / 2;
	z := make(Natural, n);
	if len(x) & 1 == 1 {
		// handle odd len(x)
		n--;
		z[n] = digit(x[n*2]);
	}
	for i := 0; i < n; i++ {
		z[i] = digit(x[i*2 + 1]) << _W2 | digit(x[i*2]);
	}
	return normalize(z);
}


func mul21(z, x []digit2, y digit2) digit2 {
	c := digit(0);
	f := digit(y);
	for i := 0; i < len(x); i++ {
		t := c + digit(x[i])*f;
		c, z[i] = t>>_W2, digit2(t&_M2);
	}
	return digit2(c);
}


func div21(z, x []digit2, y digit2) digit2 {
	c := digit(0);
	d := digit(y);
	for i := len(x)-1; i >= 0; i-- {
		t := c<<_W2 + digit(x[i]);
		c, z[i] = t%d, digit2(t/d);
	}
	return digit2(c);
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
// 1) D. Knuth, The Art of Computer Programming. Volume 2. Seminumerical
//    Algorithms. Addison-Wesley, Reading, 1969.
//    (Algorithm D, Sec. 4.3.1)
//
// 2) Henry S. Warren, Jr., Hacker's Delight. Addison-Wesley, 2003.
//    (9-2 Multiword Division, p.140ff)
//
// 3) P. Brinch Hansen, ``Multiple-length division revisited: A tour of the
//    minefield''. Software - Practice and Experience 24, (June 1994),
//    579-601. John Wiley & Sons, Ltd.

func divmod(x, y []digit2) ([]digit2, []digit2) {
	n := len(x);
	m := len(y);
	if m == 0 {
		panic("division by zero");
	}
	assert(n+1 <= cap(x));  // space for one extra digit
	x = x[0 : n+1];
	assert(x[n] == 0);

	if m == 1 {
		// division by single digit
		// result is shifted left by 1 in place!
		x[0] = div21(x[1 : n+1], x[0 : n], y[0]);

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
		//      satisfied (as done in Hacker's Delight).
		f := _B2 / (digit(y[m-1]) + 1);
		if f != 1 {
			mul21(x, x, digit2(f));
			mul21(y, y, digit2(f));
		}
		assert(_B2/2 <= y[m-1] && y[m-1] < _B2);  // incorrect scaling

		y1, y2 := digit(y[m-1]), digit(y[m-2]);
		for i := n-m; i >= 0; i-- {
			k := i+m;

			// compute trial digit (Knuth)
			var q digit;
			{	x0, x1, x2 := digit(x[k]), digit(x[k-1]), digit(x[k-2]);
				if x0 != y1 {
					q = (x0<<_W2 + x1)/y1;
				} else {
					q = _B2-1;
				}
				for y2*q > (x0<<_W2 + x1 - y1*q)<<_W2 + x2 {
					q--
				}
			}

			// subtract y*q
			c := digit(0);
			for j := 0; j < m; j++ {
				t := c + digit(x[i+j]) - digit(y[j])*q;
				c, x[i+j] = digit(int64(t) >> _W2), digit2(t & _M2);  // requires arithmetic shift!
			}

			// correct if trial digit was too large
			if c + digit(x[k]) != 0 {
				// add y
				c := digit(0);
				for j := 0; j < m; j++ {
					t := c + digit(x[i+j]) + digit(y[j]);
					c, x[i+j] = t >> _W2, digit2(t & _M2)
				}
				assert(c + digit(x[k]) == 0);
				// correct trial digit
				q--;
			}

			x[k] = digit2(q);
		}

		// undo normalization for remainder
		if f != 1 {
			c := div21(x[0 : m], x[0 : m], digit2(f));
			assert(c == 0);
		}
	}

	return x[m : n+1], x[0 : m];
}


// Div returns the quotient q = x / y for y > 0,
// with x = y*q + r and 0 <= r < y.
// If y == 0, a division-by-zero run-time error occurs.
//
func (x Natural) Div(y Natural) Natural {
	q, _ := divmod(unpack(x), unpack(y));
	return pack(q);
}


// Mod returns the modulus r of the division x / y for y > 0,
// with x = y*q + r and 0 <= r < y.
// If y == 0, a division-by-zero run-time error occurs.
//
func (x Natural) Mod(y Natural) Natural {
	_, r := divmod(unpack(x), unpack(y));
	return pack(r);
}


// DivMod returns the pair (x.Div(y), x.Mod(y)) for y > 0.
// If y == 0, a division-by-zero run-time error occurs.
//
func (x Natural) DivMod(y Natural) (Natural, Natural) {
	q, r := divmod(unpack(x), unpack(y));
	return pack(q), pack(r);
}


func shl(z, x Natural, s uint) digit {
	assert(s <= _W);
	n := len(x);
	c := digit(0);
	for i := 0; i < n; i++ {
		c, z[i] = x[i] >> (_W-s), x[i] << s & _M | c;
	}
	return c;
}


// Shl implements ``shift left'' x << s. It returns x * 2^s.
//
func (x Natural) Shl(s uint) Natural {
	n := uint(len(x));
	m := n + s/_W;
	z := make(Natural, m+1);

	z[m] = shl(z[m-n : m], x, s%_W);

	return normalize(z);
}


func shr(z, x Natural, s uint) digit {
	assert(s <= _W);
	n := len(x);
	c := digit(0);
	for i := n - 1; i >= 0; i-- {
		c, z[i] = x[i] << (_W-s) & _M, x[i] >> s | c;
	}
	return c;
}


// Shr implements ``shift right'' x >> s. It returns x / 2^s.
//
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


// And returns the ``bitwise and'' x & y for the 2's-complement representation of x and y.
//
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


func copy(z, x Natural) {
	for i, e := range x {
		z[i] = e
	}
}


// AndNot returns the ``bitwise clear'' x &^ y for the 2's-complement representation of x and y.
//
func (x Natural) AndNot(y Natural) Natural {
	n := len(x);
	m := len(y);
	if n < m {
		m = n;
	}

	z := make(Natural, n);
	for i := 0; i < m; i++ {
		z[i] = x[i] &^ y[i];
	}
	copy(z[m : n], x[m : n]);

	return normalize(z);
}


// Or returns the ``bitwise or'' x | y for the 2's-complement representation of x and y.
//
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


// Xor returns the ``bitwise exclusive or'' x ^ y for the 2's-complement representation of x and y.
//
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


// Cmp compares x and y. The result is an int value
//
//   <  0 if x <  y
//   == 0 if x == y
//   >  0 if x >  y
//
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


// log2 computes the binary logarithm of x for x > 0.
// The result is the integer n for which 2^n <= x < 2^(n+1).
// If x == 0 a run-time error occurs.
//
func log2(x uint64) uint {
	assert(x > 0);
	n := uint(0);
	for x > 0 {
		x >>= 1;
		n++;
	}
	return n - 1;
}


// Log2 computes the binary logarithm of x for x > 0.
// The result is the integer n for which 2^n <= x < 2^(n+1).
// If x == 0 a run-time error occurs.
//
func (x Natural) Log2() uint {
	n := len(x);
	if n > 0 {
		return (uint(n) - 1)*_W + log2(uint64(x[n - 1]));
	}
	panic("Log2(0)");
}


// Computes x = x div d in place (modifies x) for small d's.
// Returns updated x and x mod d.
//
func divmod1(x Natural, d digit) (Natural, digit) {
	assert(0 < d && isSmall(d - 1));

	c := digit(0);
	for i := len(x) - 1; i >= 0; i-- {
		t := c<<_W + x[i];
		c, x[i] = t%d, t/d;
	}

	return normalize(x), c;
}


// ToString converts x to a string for a given base, with 2 <= base <= 16.
//
func (x Natural) ToString(base uint) string {
	if len(x) == 0 {
		return "0";
	}

	// allocate buffer for conversion
	assert(2 <= base && base <= 16);
	n := (x.Log2() + 1) / log2(uint64(base)) + 1;  // +1: round up
	s := make([]byte, n);

	// don't destroy x
	t := make(Natural, len(x));
	copy(t, x);

	// convert
	i := n;
	for !t.IsZero() {
		i--;
		var d digit;
		t, d = divmod1(t, digit(base));
		s[i] = "0123456789abcdef"[d];
	};

	return string(s[i : n]);
}


// String converts x to its decimal string representation.
// x.String() is the same as x.ToString(10).
//
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


// Format is a support routine for fmt.Formatter. It accepts
// the formats 'b' (binary), 'o' (octal), and 'x' (hexadecimal).
//
func (x Natural) Format(h fmt.State, c int) {
	fmt.Fprintf(h, "%s", x.ToString(fmtbase(c)));
}


func hexvalue(ch byte) uint {
	d := uint(1 << logH);
	switch {
	case '0' <= ch && ch <= '9': d = uint(ch - '0');
	case 'a' <= ch && ch <= 'f': d = uint(ch - 'a') + 10;
	case 'A' <= ch && ch <= 'F': d = uint(ch - 'A') + 10;
	}
	return d;
}


// NatFromString returns the natural number corresponding to the
// longest possible prefix of s representing a natural number in a
// given conversion base, the actual conversion base used, and the
// prefix length. The syntax of natural numbers follows the syntax
// of unsigned integer literals in Go.
//
// If the base argument is 0, the string prefix determines the actual
// conversion base. A prefix of ``0x'' or ``0X'' selects base 16; the
// ``0'' prefix selects base 8. Otherwise the selected base is 10.
//
func NatFromString(s string, base uint) (Natural, uint, int) {
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
			x = muladd1(x, digit(base), digit(d));
		} else {
			break;
		}
	}

	return x, base, i;
}


// Natural number functions

func pop1(x digit) uint {
	n := uint(0);
	for x != 0 {
		x &= x-1;
		n++;
	}
	return n;
}


// Pop computes the ``population count'' of (the number of 1 bits in) x.
//
func (x Natural) Pop() uint {
	n := uint(0);
	for i := len(x) - 1; i >= 0; i-- {
		n += pop1(x[i]);
	}
	return n;
}


// Pow computes x to the power of n.
//
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


// MulRange computes the product of all the unsigned integers
// in the range [a, b] inclusively.
//
func MulRange(a, b uint) Natural {
	switch {
	case a > b: return Nat(1);
	case a == b: return Nat(uint64(a));
	case a + 1 == b: return Nat(uint64(a)).Mul(Nat(uint64(b)));
	}
	m := (a + b)>>1;
	assert(a <= m && m < b);
	return MulRange(a, m).Mul(MulRange(m + 1, b));
}


// Fact computes the factorial of n (Fact(n) == MulRange(2, n)).
//
func Fact(n uint) Natural {
	// Using MulRange() instead of the basic for-loop
	// lead to faster factorial computation.
	return MulRange(2, n);
}


// Binomial computes the binomial coefficient of (n, k).
//
func Binomial(n, k uint) Natural {
	return MulRange(n-k+1, n).Div(MulRange(1, k));
}


// Gcd computes the gcd of x and y.
//
func (x Natural) Gcd(y Natural) Natural {
	// Euclidean algorithm.
	a, b := x, y;
	for !b.IsZero() {
		a, b = b, a.Mod(b);
	}
	return a;
}
