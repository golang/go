// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements signed multi-precision integers.

package big

import (
	"errors"
	"fmt"
	"io"
	"math/rand"
	"strings"
)

// An Int represents a signed multi-precision integer.
// The zero value for an Int represents the value 0.
type Int struct {
	neg bool // sign
	abs nat  // absolute value of the integer
}

var intOne = &Int{false, natOne}

// Sign returns:
//
//	-1 if x <  0
//	 0 if x == 0
//	+1 if x >  0
//
func (x *Int) Sign() int {
	if len(x.abs) == 0 {
		return 0
	}
	if x.neg {
		return -1
	}
	return 1
}

// SetInt64 sets z to x and returns z.
func (z *Int) SetInt64(x int64) *Int {
	neg := false
	if x < 0 {
		neg = true
		x = -x
	}
	z.abs = z.abs.setUint64(uint64(x))
	z.neg = neg
	return z
}

// NewInt allocates and returns a new Int set to x.
func NewInt(x int64) *Int {
	return new(Int).SetInt64(x)
}

// Set sets z to x and returns z.
func (z *Int) Set(x *Int) *Int {
	if z != x {
		z.abs = z.abs.set(x.abs)
		z.neg = x.neg
	}
	return z
}

// Bits provides raw (unchecked but fast) access to x by returning its
// absolute value as a little-endian Word slice. The result and x share
// the same underlying array.
// Bits is intended to support implementation of missing low-level Int
// functionality outside this package; it should be avoided otherwise.
func (x *Int) Bits() []Word {
	return x.abs
}

// SetBits provides raw (unchecked but fast) access to z by setting its
// value to abs, interpreted as a little-endian Word slice, and returning
// z. The result and abs share the same underlying array.
// SetBits is intended to support implementation of missing low-level Int
// functionality outside this package; it should be avoided otherwise.
func (z *Int) SetBits(abs []Word) *Int {
	z.abs = nat(abs).norm()
	z.neg = false
	return z
}

// Abs sets z to |x| (the absolute value of x) and returns z.
func (z *Int) Abs(x *Int) *Int {
	z.Set(x)
	z.neg = false
	return z
}

// Neg sets z to -x and returns z.
func (z *Int) Neg(x *Int) *Int {
	z.Set(x)
	z.neg = len(z.abs) > 0 && !z.neg // 0 has no sign
	return z
}

// Add sets z to the sum x+y and returns z.
func (z *Int) Add(x, y *Int) *Int {
	neg := x.neg
	if x.neg == y.neg {
		// x + y == x + y
		// (-x) + (-y) == -(x + y)
		z.abs = z.abs.add(x.abs, y.abs)
	} else {
		// x + (-y) == x - y == -(y - x)
		// (-x) + y == y - x == -(x - y)
		if x.abs.cmp(y.abs) >= 0 {
			z.abs = z.abs.sub(x.abs, y.abs)
		} else {
			neg = !neg
			z.abs = z.abs.sub(y.abs, x.abs)
		}
	}
	z.neg = len(z.abs) > 0 && neg // 0 has no sign
	return z
}

// Sub sets z to the difference x-y and returns z.
func (z *Int) Sub(x, y *Int) *Int {
	neg := x.neg
	if x.neg != y.neg {
		// x - (-y) == x + y
		// (-x) - y == -(x + y)
		z.abs = z.abs.add(x.abs, y.abs)
	} else {
		// x - y == x - y == -(y - x)
		// (-x) - (-y) == y - x == -(x - y)
		if x.abs.cmp(y.abs) >= 0 {
			z.abs = z.abs.sub(x.abs, y.abs)
		} else {
			neg = !neg
			z.abs = z.abs.sub(y.abs, x.abs)
		}
	}
	z.neg = len(z.abs) > 0 && neg // 0 has no sign
	return z
}

// Mul sets z to the product x*y and returns z.
func (z *Int) Mul(x, y *Int) *Int {
	// x * y == x * y
	// x * (-y) == -(x * y)
	// (-x) * y == -(x * y)
	// (-x) * (-y) == x * y
	z.abs = z.abs.mul(x.abs, y.abs)
	z.neg = len(z.abs) > 0 && x.neg != y.neg // 0 has no sign
	return z
}

// MulRange sets z to the product of all integers
// in the range [a, b] inclusively and returns z.
// If a > b (empty range), the result is 1.
func (z *Int) MulRange(a, b int64) *Int {
	switch {
	case a > b:
		return z.SetInt64(1) // empty range
	case a <= 0 && b >= 0:
		return z.SetInt64(0) // range includes 0
	}
	// a <= b && (b < 0 || a > 0)

	neg := false
	if a < 0 {
		neg = (b-a)&1 == 0
		a, b = -b, -a
	}

	z.abs = z.abs.mulRange(uint64(a), uint64(b))
	z.neg = neg
	return z
}

// Binomial sets z to the binomial coefficient of (n, k) and returns z.
func (z *Int) Binomial(n, k int64) *Int {
	var a, b Int
	a.MulRange(n-k+1, n)
	b.MulRange(1, k)
	return z.Quo(&a, &b)
}

// Quo sets z to the quotient x/y for y != 0 and returns z.
// If y == 0, a division-by-zero run-time panic occurs.
// Quo implements truncated division (like Go); see QuoRem for more details.
func (z *Int) Quo(x, y *Int) *Int {
	z.abs, _ = z.abs.div(nil, x.abs, y.abs)
	z.neg = len(z.abs) > 0 && x.neg != y.neg // 0 has no sign
	return z
}

// Rem sets z to the remainder x%y for y != 0 and returns z.
// If y == 0, a division-by-zero run-time panic occurs.
// Rem implements truncated modulus (like Go); see QuoRem for more details.
func (z *Int) Rem(x, y *Int) *Int {
	_, z.abs = nat(nil).div(z.abs, x.abs, y.abs)
	z.neg = len(z.abs) > 0 && x.neg // 0 has no sign
	return z
}

// QuoRem sets z to the quotient x/y and r to the remainder x%y
// and returns the pair (z, r) for y != 0.
// If y == 0, a division-by-zero run-time panic occurs.
//
// QuoRem implements T-division and modulus (like Go):
//
//	q = x/y      with the result truncated to zero
//	r = x - y*q
//
// (See Daan Leijen, ``Division and Modulus for Computer Scientists''.)
// See DivMod for Euclidean division and modulus (unlike Go).
//
func (z *Int) QuoRem(x, y, r *Int) (*Int, *Int) {
	z.abs, r.abs = z.abs.div(r.abs, x.abs, y.abs)
	z.neg, r.neg = len(z.abs) > 0 && x.neg != y.neg, len(r.abs) > 0 && x.neg // 0 has no sign
	return z, r
}

// Div sets z to the quotient x/y for y != 0 and returns z.
// If y == 0, a division-by-zero run-time panic occurs.
// Div implements Euclidean division (unlike Go); see DivMod for more details.
func (z *Int) Div(x, y *Int) *Int {
	y_neg := y.neg // z may be an alias for y
	var r Int
	z.QuoRem(x, y, &r)
	if r.neg {
		if y_neg {
			z.Add(z, intOne)
		} else {
			z.Sub(z, intOne)
		}
	}
	return z
}

// Mod sets z to the modulus x%y for y != 0 and returns z.
// If y == 0, a division-by-zero run-time panic occurs.
// Mod implements Euclidean modulus (unlike Go); see DivMod for more details.
func (z *Int) Mod(x, y *Int) *Int {
	y0 := y // save y
	if z == y || alias(z.abs, y.abs) {
		y0 = new(Int).Set(y)
	}
	var q Int
	q.QuoRem(x, y, z)
	if z.neg {
		if y0.neg {
			z.Sub(z, y0)
		} else {
			z.Add(z, y0)
		}
	}
	return z
}

// DivMod sets z to the quotient x div y and m to the modulus x mod y
// and returns the pair (z, m) for y != 0.
// If y == 0, a division-by-zero run-time panic occurs.
//
// DivMod implements Euclidean division and modulus (unlike Go):
//
//	q = x div y  such that
//	m = x - y*q  with 0 <= m < |q|
//
// (See Raymond T. Boute, ``The Euclidean definition of the functions
// div and mod''. ACM Transactions on Programming Languages and
// Systems (TOPLAS), 14(2):127-144, New York, NY, USA, 4/1992.
// ACM press.)
// See QuoRem for T-division and modulus (like Go).
//
func (z *Int) DivMod(x, y, m *Int) (*Int, *Int) {
	y0 := y // save y
	if z == y || alias(z.abs, y.abs) {
		y0 = new(Int).Set(y)
	}
	z.QuoRem(x, y, m)
	if m.neg {
		if y0.neg {
			z.Add(z, intOne)
			m.Sub(m, y0)
		} else {
			z.Sub(z, intOne)
			m.Add(m, y0)
		}
	}
	return z, m
}

// Cmp compares x and y and returns:
//
//   -1 if x <  y
//    0 if x == y
//   +1 if x >  y
//
func (x *Int) Cmp(y *Int) (r int) {
	// x cmp y == x cmp y
	// x cmp (-y) == x
	// (-x) cmp y == y
	// (-x) cmp (-y) == -(x cmp y)
	switch {
	case x.neg == y.neg:
		r = x.abs.cmp(y.abs)
		if x.neg {
			r = -r
		}
	case x.neg:
		r = -1
	default:
		r = 1
	}
	return
}

func (x *Int) String() string {
	switch {
	case x == nil:
		return "<nil>"
	case x.neg:
		return "-" + x.abs.decimalString()
	}
	return x.abs.decimalString()
}

func charset(ch rune) string {
	switch ch {
	case 'b':
		return lowercaseDigits[0:2]
	case 'o':
		return lowercaseDigits[0:8]
	case 'd', 's', 'v':
		return lowercaseDigits[0:10]
	case 'x':
		return lowercaseDigits[0:16]
	case 'X':
		return uppercaseDigits[0:16]
	}
	return "" // unknown format
}

// write count copies of text to s
func writeMultiple(s fmt.State, text string, count int) {
	if len(text) > 0 {
		b := []byte(text)
		for ; count > 0; count-- {
			s.Write(b)
		}
	}
}

// Format is a support routine for fmt.Formatter. It accepts
// the formats 'b' (binary), 'o' (octal), 'd' (decimal), 'x'
// (lowercase hexadecimal), and 'X' (uppercase hexadecimal).
// Also supported are the full suite of package fmt's format
// verbs for integral types, including '+', '-', and ' '
// for sign control, '#' for leading zero in octal and for
// hexadecimal, a leading "0x" or "0X" for "%#x" and "%#X"
// respectively, specification of minimum digits precision,
// output field width, space or zero padding, and left or
// right justification.
//
func (x *Int) Format(s fmt.State, ch rune) {
	cs := charset(ch)

	// special cases
	switch {
	case cs == "":
		// unknown format
		fmt.Fprintf(s, "%%!%c(big.Int=%s)", ch, x.String())
		return
	case x == nil:
		fmt.Fprint(s, "<nil>")
		return
	}

	// determine sign character
	sign := ""
	switch {
	case x.neg:
		sign = "-"
	case s.Flag('+'): // supersedes ' ' when both specified
		sign = "+"
	case s.Flag(' '):
		sign = " "
	}

	// determine prefix characters for indicating output base
	prefix := ""
	if s.Flag('#') {
		switch ch {
		case 'o': // octal
			prefix = "0"
		case 'x': // hexadecimal
			prefix = "0x"
		case 'X':
			prefix = "0X"
		}
	}

	// determine digits with base set by len(cs) and digit characters from cs
	digits := x.abs.string(cs)

	// number of characters for the three classes of number padding
	var left int   // space characters to left of digits for right justification ("%8d")
	var zeroes int // zero characters (actually cs[0]) as left-most digits ("%.8d")
	var right int  // space characters to right of digits for left justification ("%-8d")

	// determine number padding from precision: the least number of digits to output
	precision, precisionSet := s.Precision()
	if precisionSet {
		switch {
		case len(digits) < precision:
			zeroes = precision - len(digits) // count of zero padding 
		case digits == "0" && precision == 0:
			return // print nothing if zero value (x == 0) and zero precision ("." or ".0")
		}
	}

	// determine field pad from width: the least number of characters to output
	length := len(sign) + len(prefix) + zeroes + len(digits)
	if width, widthSet := s.Width(); widthSet && length < width { // pad as specified
		switch d := width - length; {
		case s.Flag('-'):
			// pad on the right with spaces; supersedes '0' when both specified
			right = d
		case s.Flag('0') && !precisionSet:
			// pad with zeroes unless precision also specified
			zeroes = d
		default:
			// pad on the left with spaces
			left = d
		}
	}

	// print number as [left pad][sign][prefix][zero pad][digits][right pad]
	writeMultiple(s, " ", left)
	writeMultiple(s, sign, 1)
	writeMultiple(s, prefix, 1)
	writeMultiple(s, "0", zeroes)
	writeMultiple(s, digits, 1)
	writeMultiple(s, " ", right)
}

// scan sets z to the integer value corresponding to the longest possible prefix
// read from r representing a signed integer number in a given conversion base.
// It returns z, the actual conversion base used, and an error, if any. In the
// error case, the value of z is undefined but the returned value is nil. The
// syntax follows the syntax of integer literals in Go.
//
// The base argument must be 0 or a value from 2 through MaxBase. If the base
// is 0, the string prefix determines the actual conversion base. A prefix of
// ``0x'' or ``0X'' selects base 16; the ``0'' prefix selects base 8, and a
// ``0b'' or ``0B'' prefix selects base 2. Otherwise the selected base is 10.
//
func (z *Int) scan(r io.RuneScanner, base int) (*Int, int, error) {
	// determine sign
	ch, _, err := r.ReadRune()
	if err != nil {
		return nil, 0, err
	}
	neg := false
	switch ch {
	case '-':
		neg = true
	case '+': // nothing to do
	default:
		r.UnreadRune()
	}

	// determine mantissa
	z.abs, base, err = z.abs.scan(r, base)
	if err != nil {
		return nil, base, err
	}
	z.neg = len(z.abs) > 0 && neg // 0 has no sign

	return z, base, nil
}

// Scan is a support routine for fmt.Scanner; it sets z to the value of
// the scanned number. It accepts the formats 'b' (binary), 'o' (octal),
// 'd' (decimal), 'x' (lowercase hexadecimal), and 'X' (uppercase hexadecimal).
func (z *Int) Scan(s fmt.ScanState, ch rune) error {
	s.SkipSpace() // skip leading space characters
	base := 0
	switch ch {
	case 'b':
		base = 2
	case 'o':
		base = 8
	case 'd':
		base = 10
	case 'x', 'X':
		base = 16
	case 's', 'v':
		// let scan determine the base
	default:
		return errors.New("Int.Scan: invalid verb")
	}
	_, _, err := z.scan(s, base)
	return err
}

// Int64 returns the int64 representation of x.
// If x cannot be represented in an int64, the result is undefined.
func (x *Int) Int64() int64 {
	if len(x.abs) == 0 {
		return 0
	}
	v := int64(x.abs[0])
	if _W == 32 && len(x.abs) > 1 {
		v |= int64(x.abs[1]) << 32
	}
	if x.neg {
		v = -v
	}
	return v
}

// SetString sets z to the value of s, interpreted in the given base,
// and returns z and a boolean indicating success. If SetString fails,
// the value of z is undefined but the returned value is nil.
//
// The base argument must be 0 or a value from 2 through MaxBase. If the base
// is 0, the string prefix determines the actual conversion base. A prefix of
// ``0x'' or ``0X'' selects base 16; the ``0'' prefix selects base 8, and a
// ``0b'' or ``0B'' prefix selects base 2. Otherwise the selected base is 10.
//
func (z *Int) SetString(s string, base int) (*Int, bool) {
	r := strings.NewReader(s)
	_, _, err := z.scan(r, base)
	if err != nil {
		return nil, false
	}
	_, _, err = r.ReadRune()
	if err != io.EOF {
		return nil, false
	}
	return z, true // err == io.EOF => scan consumed all of s
}

// SetBytes interprets buf as the bytes of a big-endian unsigned
// integer, sets z to that value, and returns z.
func (z *Int) SetBytes(buf []byte) *Int {
	z.abs = z.abs.setBytes(buf)
	z.neg = false
	return z
}

// Bytes returns the absolute value of z as a big-endian byte slice.
func (x *Int) Bytes() []byte {
	buf := make([]byte, len(x.abs)*_S)
	return buf[x.abs.bytes(buf):]
}

// BitLen returns the length of the absolute value of z in bits.
// The bit length of 0 is 0.
func (x *Int) BitLen() int {
	return x.abs.bitLen()
}

// Exp sets z = x**y mod m and returns z. If m is nil, z = x**y.
// See Knuth, volume 2, section 4.6.3.
func (z *Int) Exp(x, y, m *Int) *Int {
	if y.neg || len(y.abs) == 0 {
		neg := x.neg
		z.SetInt64(1)
		z.neg = neg
		return z
	}

	var mWords nat
	if m != nil {
		mWords = m.abs
	}

	z.abs = z.abs.expNN(x.abs, y.abs, mWords)
	z.neg = len(z.abs) > 0 && x.neg && y.abs[0]&1 == 1 // 0 has no sign
	return z
}

// GCD sets z to the greatest common divisor of a and b, which must be
// positive numbers, and returns z.
// If x and y are not nil, GCD sets x and y such that z = a*x + b*y.
// If either a or b is not positive, GCD sets z = x = y = 0.
func (z *Int) GCD(x, y, a, b *Int) *Int {
	if a.neg || b.neg {
		z.SetInt64(0)
		if x != nil {
			x.SetInt64(0)
		}
		if y != nil {
			y.SetInt64(0)
		}
		return z
	}

	A := new(Int).Set(a)
	B := new(Int).Set(b)

	X := new(Int)
	Y := new(Int).SetInt64(1)

	lastX := new(Int).SetInt64(1)
	lastY := new(Int)

	q := new(Int)
	temp := new(Int)

	for len(B.abs) > 0 {
		r := new(Int)
		q, r = q.QuoRem(A, B, r)

		A, B = B, r

		temp.Set(X)
		X.Mul(X, q)
		X.neg = !X.neg
		X.Add(X, lastX)
		lastX.Set(temp)

		temp.Set(Y)
		Y.Mul(Y, q)
		Y.neg = !Y.neg
		Y.Add(Y, lastY)
		lastY.Set(temp)
	}

	if x != nil {
		*x = *lastX
	}

	if y != nil {
		*y = *lastY
	}

	*z = *A
	return z
}

// ProbablyPrime performs n Miller-Rabin tests to check whether x is prime.
// If it returns true, x is prime with probability 1 - 1/4^n.
// If it returns false, x is not prime.
func (x *Int) ProbablyPrime(n int) bool {
	return !x.neg && x.abs.probablyPrime(n)
}

// Rand sets z to a pseudo-random number in [0, n) and returns z.
func (z *Int) Rand(rnd *rand.Rand, n *Int) *Int {
	z.neg = false
	if n.neg == true || len(n.abs) == 0 {
		z.abs = nil
		return z
	}
	z.abs = z.abs.random(rnd, n.abs, n.abs.bitLen())
	return z
}

// ModInverse sets z to the multiplicative inverse of g in the group ℤ/pℤ (where
// p is a prime) and returns z.
func (z *Int) ModInverse(g, p *Int) *Int {
	var d Int
	d.GCD(z, nil, g, p)
	// x and y are such that g*x + p*y = d. Since p is prime, d = 1. Taking
	// that modulo p results in g*x = 1, therefore x is the inverse element.
	if z.neg {
		z.Add(z, p)
	}
	return z
}

// Lsh sets z = x << n and returns z.
func (z *Int) Lsh(x *Int, n uint) *Int {
	z.abs = z.abs.shl(x.abs, n)
	z.neg = x.neg
	return z
}

// Rsh sets z = x >> n and returns z.
func (z *Int) Rsh(x *Int, n uint) *Int {
	if x.neg {
		// (-x) >> s == ^(x-1) >> s == ^((x-1) >> s) == -(((x-1) >> s) + 1)
		t := z.abs.sub(x.abs, natOne) // no underflow because |x| > 0
		t = t.shr(t, n)
		z.abs = t.add(t, natOne)
		z.neg = true // z cannot be zero if x is negative
		return z
	}

	z.abs = z.abs.shr(x.abs, n)
	z.neg = false
	return z
}

// Bit returns the value of the i'th bit of x. That is, it
// returns (x>>i)&1. The bit index i must be >= 0.
func (x *Int) Bit(i int) uint {
	if i < 0 {
		panic("negative bit index")
	}
	if x.neg {
		t := nat(nil).sub(x.abs, natOne)
		return t.bit(uint(i)) ^ 1
	}

	return x.abs.bit(uint(i))
}

// SetBit sets z to x, with x's i'th bit set to b (0 or 1).
// That is, if bit is 1 SetBit sets z = x | (1 << i);
// if bit is 0 it sets z = x &^ (1 << i). If bit is not 0 or 1,
// SetBit will panic.
func (z *Int) SetBit(x *Int, i int, b uint) *Int {
	if i < 0 {
		panic("negative bit index")
	}
	if x.neg {
		t := z.abs.sub(x.abs, natOne)
		t = t.setBit(t, uint(i), b^1)
		z.abs = t.add(t, natOne)
		z.neg = len(z.abs) > 0
		return z
	}
	z.abs = z.abs.setBit(x.abs, uint(i), b)
	z.neg = false
	return z
}

// And sets z = x & y and returns z.
func (z *Int) And(x, y *Int) *Int {
	if x.neg == y.neg {
		if x.neg {
			// (-x) & (-y) == ^(x-1) & ^(y-1) == ^((x-1) | (y-1)) == -(((x-1) | (y-1)) + 1)
			x1 := nat(nil).sub(x.abs, natOne)
			y1 := nat(nil).sub(y.abs, natOne)
			z.abs = z.abs.add(z.abs.or(x1, y1), natOne)
			z.neg = true // z cannot be zero if x and y are negative
			return z
		}

		// x & y == x & y
		z.abs = z.abs.and(x.abs, y.abs)
		z.neg = false
		return z
	}

	// x.neg != y.neg
	if x.neg {
		x, y = y, x // & is symmetric
	}

	// x & (-y) == x & ^(y-1) == x &^ (y-1)
	y1 := nat(nil).sub(y.abs, natOne)
	z.abs = z.abs.andNot(x.abs, y1)
	z.neg = false
	return z
}

// AndNot sets z = x &^ y and returns z.
func (z *Int) AndNot(x, y *Int) *Int {
	if x.neg == y.neg {
		if x.neg {
			// (-x) &^ (-y) == ^(x-1) &^ ^(y-1) == ^(x-1) & (y-1) == (y-1) &^ (x-1)
			x1 := nat(nil).sub(x.abs, natOne)
			y1 := nat(nil).sub(y.abs, natOne)
			z.abs = z.abs.andNot(y1, x1)
			z.neg = false
			return z
		}

		// x &^ y == x &^ y
		z.abs = z.abs.andNot(x.abs, y.abs)
		z.neg = false
		return z
	}

	if x.neg {
		// (-x) &^ y == ^(x-1) &^ y == ^(x-1) & ^y == ^((x-1) | y) == -(((x-1) | y) + 1)
		x1 := nat(nil).sub(x.abs, natOne)
		z.abs = z.abs.add(z.abs.or(x1, y.abs), natOne)
		z.neg = true // z cannot be zero if x is negative and y is positive
		return z
	}

	// x &^ (-y) == x &^ ^(y-1) == x & (y-1)
	y1 := nat(nil).add(y.abs, natOne)
	z.abs = z.abs.and(x.abs, y1)
	z.neg = false
	return z
}

// Or sets z = x | y and returns z.
func (z *Int) Or(x, y *Int) *Int {
	if x.neg == y.neg {
		if x.neg {
			// (-x) | (-y) == ^(x-1) | ^(y-1) == ^((x-1) & (y-1)) == -(((x-1) & (y-1)) + 1)
			x1 := nat(nil).sub(x.abs, natOne)
			y1 := nat(nil).sub(y.abs, natOne)
			z.abs = z.abs.add(z.abs.and(x1, y1), natOne)
			z.neg = true // z cannot be zero if x and y are negative
			return z
		}

		// x | y == x | y
		z.abs = z.abs.or(x.abs, y.abs)
		z.neg = false
		return z
	}

	// x.neg != y.neg
	if x.neg {
		x, y = y, x // | is symmetric
	}

	// x | (-y) == x | ^(y-1) == ^((y-1) &^ x) == -(^((y-1) &^ x) + 1)
	y1 := nat(nil).sub(y.abs, natOne)
	z.abs = z.abs.add(z.abs.andNot(y1, x.abs), natOne)
	z.neg = true // z cannot be zero if one of x or y is negative
	return z
}

// Xor sets z = x ^ y and returns z.
func (z *Int) Xor(x, y *Int) *Int {
	if x.neg == y.neg {
		if x.neg {
			// (-x) ^ (-y) == ^(x-1) ^ ^(y-1) == (x-1) ^ (y-1)
			x1 := nat(nil).sub(x.abs, natOne)
			y1 := nat(nil).sub(y.abs, natOne)
			z.abs = z.abs.xor(x1, y1)
			z.neg = false
			return z
		}

		// x ^ y == x ^ y
		z.abs = z.abs.xor(x.abs, y.abs)
		z.neg = false
		return z
	}

	// x.neg != y.neg
	if x.neg {
		x, y = y, x // ^ is symmetric
	}

	// x ^ (-y) == x ^ ^(y-1) == ^(x ^ (y-1)) == -((x ^ (y-1)) + 1)
	y1 := nat(nil).sub(y.abs, natOne)
	z.abs = z.abs.add(z.abs.xor(x.abs, y1), natOne)
	z.neg = true // z cannot be zero if only one of x or y is negative
	return z
}

// Not sets z = ^x and returns z.
func (z *Int) Not(x *Int) *Int {
	if x.neg {
		// ^(-x) == ^(^(x-1)) == x-1
		z.abs = z.abs.sub(x.abs, natOne)
		z.neg = false
		return z
	}

	// ^x == -x-1 == -(x+1)
	z.abs = z.abs.add(x.abs, natOne)
	z.neg = true // z cannot be zero if x is positive
	return z
}

// Gob codec version. Permits backward-compatible changes to the encoding.
const intGobVersion byte = 1

// GobEncode implements the gob.GobEncoder interface.
func (x *Int) GobEncode() ([]byte, error) {
	buf := make([]byte, 1+len(x.abs)*_S) // extra byte for version and sign bit
	i := x.abs.bytes(buf) - 1            // i >= 0
	b := intGobVersion << 1              // make space for sign bit
	if x.neg {
		b |= 1
	}
	buf[i] = b
	return buf[i:], nil
}

// GobDecode implements the gob.GobDecoder interface.
func (z *Int) GobDecode(buf []byte) error {
	if len(buf) == 0 {
		return errors.New("Int.GobDecode: no data")
	}
	b := buf[0]
	if b>>1 != intGobVersion {
		return errors.New(fmt.Sprintf("Int.GobDecode: encoding version %d not supported", b>>1))
	}
	z.neg = b&1 != 0
	z.abs = z.abs.setBytes(buf[1:])
	return nil
}
