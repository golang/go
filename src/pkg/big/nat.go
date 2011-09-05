// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package big implements multi-precision arithmetic (big numbers).
// The following numeric types are supported:
//
//	- Int	signed integers
//	- Rat	rational numbers
//
// All methods on Int take the result as the receiver; if it is one
// of the operands it may be overwritten (and its memory reused).
// To enable chaining of operations, the result is also returned.
//
package big

// This file contains operations on unsigned multi-precision integers.
// These are the building blocks for the operations on signed integers
// and rationals.

import (
	"io"
	"os"
	"rand"
)

// An unsigned integer x of the form
//
//   x = x[n-1]*_B^(n-1) + x[n-2]*_B^(n-2) + ... + x[1]*_B + x[0]
//
// with 0 <= x[i] < _B and 0 <= i < n is stored in a slice of length n,
// with the digits x[i] as the slice elements.
//
// A number is normalized if the slice contains no leading 0 digits.
// During arithmetic operations, denormalized values may occur but are
// always normalized before returning the final result. The normalized
// representation of 0 is the empty or nil slice (length = 0).

type nat []Word

var (
	natOne = nat{1}
	natTwo = nat{2}
	natTen = nat{10}
)

func (z nat) clear() {
	for i := range z {
		z[i] = 0
	}
}

func (z nat) norm() nat {
	i := len(z)
	for i > 0 && z[i-1] == 0 {
		i--
	}
	return z[0:i]
}

func (z nat) make(n int) nat {
	if n <= cap(z) {
		return z[0:n] // reuse z
	}
	// Choosing a good value for e has significant performance impact
	// because it increases the chance that a value can be reused.
	const e = 4 // extra capacity
	return make(nat, n, n+e)
}

func (z nat) setWord(x Word) nat {
	if x == 0 {
		return z.make(0)
	}
	z = z.make(1)
	z[0] = x
	return z
}

func (z nat) setUint64(x uint64) nat {
	// single-digit values
	if w := Word(x); uint64(w) == x {
		return z.setWord(w)
	}

	// compute number of words n required to represent x
	n := 0
	for t := x; t > 0; t >>= _W {
		n++
	}

	// split x into n words
	z = z.make(n)
	for i := range z {
		z[i] = Word(x & _M)
		x >>= _W
	}

	return z
}

func (z nat) set(x nat) nat {
	z = z.make(len(x))
	copy(z, x)
	return z
}

func (z nat) add(x, y nat) nat {
	m := len(x)
	n := len(y)

	switch {
	case m < n:
		return z.add(y, x)
	case m == 0:
		// n == 0 because m >= n; result is 0
		return z.make(0)
	case n == 0:
		// result is x
		return z.set(x)
	}
	// m > 0

	z = z.make(m + 1)
	c := addVV(z[0:n], x, y)
	if m > n {
		c = addVW(z[n:m], x[n:], c)
	}
	z[m] = c

	return z.norm()
}

func (z nat) sub(x, y nat) nat {
	m := len(x)
	n := len(y)

	switch {
	case m < n:
		panic("underflow")
	case m == 0:
		// n == 0 because m >= n; result is 0
		return z.make(0)
	case n == 0:
		// result is x
		return z.set(x)
	}
	// m > 0

	z = z.make(m)
	c := subVV(z[0:n], x, y)
	if m > n {
		c = subVW(z[n:], x[n:], c)
	}
	if c != 0 {
		panic("underflow")
	}

	return z.norm()
}

func (x nat) cmp(y nat) (r int) {
	m := len(x)
	n := len(y)
	if m != n || m == 0 {
		switch {
		case m < n:
			r = -1
		case m > n:
			r = 1
		}
		return
	}

	i := m - 1
	for i > 0 && x[i] == y[i] {
		i--
	}

	switch {
	case x[i] < y[i]:
		r = -1
	case x[i] > y[i]:
		r = 1
	}
	return
}

func (z nat) mulAddWW(x nat, y, r Word) nat {
	m := len(x)
	if m == 0 || y == 0 {
		return z.setWord(r) // result is r
	}
	// m > 0

	z = z.make(m + 1)
	z[m] = mulAddVWW(z[0:m], x, y, r)

	return z.norm()
}

// basicMul multiplies x and y and leaves the result in z.
// The (non-normalized) result is placed in z[0 : len(x) + len(y)].
func basicMul(z, x, y nat) {
	z[0 : len(x)+len(y)].clear() // initialize z
	for i, d := range y {
		if d != 0 {
			z[len(x)+i] = addMulVVW(z[i:i+len(x)], x, d)
		}
	}
}

// Fast version of z[0:n+n>>1].add(z[0:n+n>>1], x[0:n]) w/o bounds checks.
// Factored out for readability - do not use outside karatsuba.
func karatsubaAdd(z, x nat, n int) {
	if c := addVV(z[0:n], z, x); c != 0 {
		addVW(z[n:n+n>>1], z[n:], c)
	}
}

// Like karatsubaAdd, but does subtract.
func karatsubaSub(z, x nat, n int) {
	if c := subVV(z[0:n], z, x); c != 0 {
		subVW(z[n:n+n>>1], z[n:], c)
	}
}

// Operands that are shorter than karatsubaThreshold are multiplied using
// "grade school" multiplication; for longer operands the Karatsuba algorithm
// is used.
var karatsubaThreshold int = 32 // computed by calibrate.go

// karatsuba multiplies x and y and leaves the result in z.
// Both x and y must have the same length n and n must be a
// power of 2. The result vector z must have len(z) >= 6*n.
// The (non-normalized) result is placed in z[0 : 2*n].
func karatsuba(z, x, y nat) {
	n := len(y)

	// Switch to basic multiplication if numbers are odd or small.
	// (n is always even if karatsubaThreshold is even, but be
	// conservative)
	if n&1 != 0 || n < karatsubaThreshold || n < 2 {
		basicMul(z, x, y)
		return
	}
	// n&1 == 0 && n >= karatsubaThreshold && n >= 2

	// Karatsuba multiplication is based on the observation that
	// for two numbers x and y with:
	//
	//   x = x1*b + x0
	//   y = y1*b + y0
	//
	// the product x*y can be obtained with 3 products z2, z1, z0
	// instead of 4:
	//
	//   x*y = x1*y1*b*b + (x1*y0 + x0*y1)*b + x0*y0
	//       =    z2*b*b +              z1*b +    z0
	//
	// with:
	//
	//   xd = x1 - x0
	//   yd = y0 - y1
	//
	//   z1 =      xd*yd                    + z1 + z0
	//      = (x1-x0)*(y0 - y1)             + z1 + z0
	//      = x1*y0 - x1*y1 - x0*y0 + x0*y1 + z1 + z0
	//      = x1*y0 -    z1 -    z0 + x0*y1 + z1 + z0
	//      = x1*y0                 + x0*y1

	// split x, y into "digits"
	n2 := n >> 1              // n2 >= 1
	x1, x0 := x[n2:], x[0:n2] // x = x1*b + y0
	y1, y0 := y[n2:], y[0:n2] // y = y1*b + y0

	// z is used for the result and temporary storage:
	//
	//   6*n     5*n     4*n     3*n     2*n     1*n     0*n
	// z = [z2 copy|z0 copy| xd*yd | yd:xd | x1*y1 | x0*y0 ]
	//
	// For each recursive call of karatsuba, an unused slice of
	// z is passed in that has (at least) half the length of the
	// caller's z.

	// compute z0 and z2 with the result "in place" in z
	karatsuba(z, x0, y0)     // z0 = x0*y0
	karatsuba(z[n:], x1, y1) // z2 = x1*y1

	// compute xd (or the negative value if underflow occurs)
	s := 1 // sign of product xd*yd
	xd := z[2*n : 2*n+n2]
	if subVV(xd, x1, x0) != 0 { // x1-x0
		s = -s
		subVV(xd, x0, x1) // x0-x1
	}

	// compute yd (or the negative value if underflow occurs)
	yd := z[2*n+n2 : 3*n]
	if subVV(yd, y0, y1) != 0 { // y0-y1
		s = -s
		subVV(yd, y1, y0) // y1-y0
	}

	// p = (x1-x0)*(y0-y1) == x1*y0 - x1*y1 - x0*y0 + x0*y1 for s > 0
	// p = (x0-x1)*(y0-y1) == x0*y0 - x0*y1 - x1*y0 + x1*y1 for s < 0
	p := z[n*3:]
	karatsuba(p, xd, yd)

	// save original z2:z0
	// (ok to use upper half of z since we're done recursing)
	r := z[n*4:]
	copy(r, z)

	// add up all partial products
	//
	//   2*n     n     0
	// z = [ z2  | z0  ]
	//   +    [ z0  ]
	//   +    [ z2  ]
	//   +    [  p  ]
	//
	karatsubaAdd(z[n2:], r, n)
	karatsubaAdd(z[n2:], r[n:], n)
	if s > 0 {
		karatsubaAdd(z[n2:], p, n)
	} else {
		karatsubaSub(z[n2:], p, n)
	}
}

// alias returns true if x and y share the same base array.
func alias(x, y nat) bool {
	return cap(x) > 0 && cap(y) > 0 && &x[0:cap(x)][cap(x)-1] == &y[0:cap(y)][cap(y)-1]
}

// addAt implements z += x*(1<<(_W*i)); z must be long enough.
// (we don't use nat.add because we need z to stay the same
// slice, and we don't need to normalize z after each addition)
func addAt(z, x nat, i int) {
	if n := len(x); n > 0 {
		if c := addVV(z[i:i+n], z[i:], x); c != 0 {
			j := i + n
			if j < len(z) {
				addVW(z[j:], z[j:], c)
			}
		}
	}
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

// karatsubaLen computes an approximation to the maximum k <= n such that
// k = p<<i for a number p <= karatsubaThreshold and an i >= 0. Thus, the
// result is the largest number that can be divided repeatedly by 2 before
// becoming about the value of karatsubaThreshold.
func karatsubaLen(n int) int {
	i := uint(0)
	for n > karatsubaThreshold {
		n >>= 1
		i++
	}
	return n << i
}

func (z nat) mul(x, y nat) nat {
	m := len(x)
	n := len(y)

	switch {
	case m < n:
		return z.mul(y, x)
	case m == 0 || n == 0:
		return z.make(0)
	case n == 1:
		return z.mulAddWW(x, y[0], 0)
	}
	// m >= n > 1

	// determine if z can be reused
	if alias(z, x) || alias(z, y) {
		z = nil // z is an alias for x or y - cannot reuse
	}

	// use basic multiplication if the numbers are small
	if n < karatsubaThreshold || n < 2 {
		z = z.make(m + n)
		basicMul(z, x, y)
		return z.norm()
	}
	// m >= n && n >= karatsubaThreshold && n >= 2

	// determine Karatsuba length k such that
	//
	//   x = x1*b + x0
	//   y = y1*b + y0  (and k <= len(y), which implies k <= len(x))
	//   b = 1<<(_W*k)  ("base" of digits xi, yi)
	//
	k := karatsubaLen(n)
	// k <= n

	// multiply x0 and y0 via Karatsuba
	x0 := x[0:k]              // x0 is not normalized
	y0 := y[0:k]              // y0 is not normalized
	z = z.make(max(6*k, m+n)) // enough space for karatsuba of x0*y0 and full result of x*y
	karatsuba(z, x0, y0)
	z = z[0 : m+n] // z has final length but may be incomplete, upper portion is garbage

	// If x1 and/or y1 are not 0, add missing terms to z explicitly:
	//
	//     m+n       2*k       0
	//   z = [   ...   | x0*y0 ]
	//     +   [ x1*y1 ]
	//     +   [ x1*y0 ]
	//     +   [ x0*y1 ]
	//
	if k < n || m != n {
		x1 := x[k:] // x1 is normalized because x is
		y1 := y[k:] // y1 is normalized because y is
		var t nat
		t = t.mul(x1, y1)
		copy(z[2*k:], t)
		z[2*k+len(t):].clear() // upper portion of z is garbage
		t = t.mul(x1, y0.norm())
		addAt(z, t, k)
		t = t.mul(x0.norm(), y1)
		addAt(z, t, k)
	}

	return z.norm()
}

// mulRange computes the product of all the unsigned integers in the
// range [a, b] inclusively. If a > b (empty range), the result is 1.
func (z nat) mulRange(a, b uint64) nat {
	switch {
	case a == 0:
		// cut long ranges short (optimization)
		return z.setUint64(0)
	case a > b:
		return z.setUint64(1)
	case a == b:
		return z.setUint64(a)
	case a+1 == b:
		return z.mul(nat(nil).setUint64(a), nat(nil).setUint64(b))
	}
	m := (a + b) / 2
	return z.mul(nat(nil).mulRange(a, m), nat(nil).mulRange(m+1, b))
}

// q = (x-r)/y, with 0 <= r < y
func (z nat) divW(x nat, y Word) (q nat, r Word) {
	m := len(x)
	switch {
	case y == 0:
		panic("division by zero")
	case y == 1:
		q = z.set(x) // result is x
		return
	case m == 0:
		q = z.make(0) // result is 0
		return
	}
	// m > 0
	z = z.make(m)
	r = divWVW(z, 0, x, y)
	q = z.norm()
	return
}

func (z nat) div(z2, u, v nat) (q, r nat) {
	if len(v) == 0 {
		panic("division by zero")
	}

	if u.cmp(v) < 0 {
		q = z.make(0)
		r = z2.set(u)
		return
	}

	if len(v) == 1 {
		var rprime Word
		q, rprime = z.divW(u, v[0])
		if rprime > 0 {
			r = z2.make(1)
			r[0] = rprime
		} else {
			r = z2.make(0)
		}
		return
	}

	q, r = z.divLarge(z2, u, v)
	return
}

// q = (uIn-r)/v, with 0 <= r < y
// Uses z as storage for q, and u as storage for r if possible.
// See Knuth, Volume 2, section 4.3.1, Algorithm D.
// Preconditions:
//    len(v) >= 2
//    len(uIn) >= len(v)
func (z nat) divLarge(u, uIn, v nat) (q, r nat) {
	n := len(v)
	m := len(uIn) - n

	// determine if z can be reused
	// TODO(gri) should find a better solution - this if statement
	//           is very costly (see e.g. time pidigits -s -n 10000)
	if alias(z, uIn) || alias(z, v) {
		z = nil // z is an alias for uIn or v - cannot reuse
	}
	q = z.make(m + 1)

	qhatv := make(nat, n+1)
	if alias(u, uIn) || alias(u, v) {
		u = nil // u is an alias for uIn or v - cannot reuse
	}
	u = u.make(len(uIn) + 1)
	u.clear()

	// D1.
	shift := leadingZeros(v[n-1])
	if shift > 0 {
		// do not modify v, it may be used by another goroutine simultaneously
		v1 := make(nat, n)
		shlVU(v1, v, shift)
		v = v1
	}
	u[len(uIn)] = shlVU(u[0:len(uIn)], uIn, shift)

	// D2.
	for j := m; j >= 0; j-- {
		// D3.
		qhat := Word(_M)
		if u[j+n] != v[n-1] {
			var rhat Word
			qhat, rhat = divWW(u[j+n], u[j+n-1], v[n-1])

			// x1 | x2 = q̂v_{n-2}
			x1, x2 := mulWW(qhat, v[n-2])
			// test if q̂v_{n-2} > br̂ + u_{j+n-2}
			for greaterThan(x1, x2, rhat, u[j+n-2]) {
				qhat--
				prevRhat := rhat
				rhat += v[n-1]
				// v[n-1] >= 0, so this tests for overflow.
				if rhat < prevRhat {
					break
				}
				x1, x2 = mulWW(qhat, v[n-2])
			}
		}

		// D4.
		qhatv[n] = mulAddVWW(qhatv[0:n], v, qhat, 0)

		c := subVV(u[j:j+len(qhatv)], u[j:], qhatv)
		if c != 0 {
			c := addVV(u[j:j+n], u[j:], v)
			u[j+n] += c
			qhat--
		}

		q[j] = qhat
	}

	q = q.norm()
	shrVU(u, u, shift)
	r = u.norm()

	return q, r
}

// Length of x in bits. x must be normalized.
func (x nat) bitLen() int {
	if i := len(x) - 1; i >= 0 {
		return i*_W + bitLen(x[i])
	}
	return 0
}

// MaxBase is the largest number base accepted for string conversions.
const MaxBase = 'z' - 'a' + 10 + 1 // = hexValue('z') + 1


func hexValue(ch int) Word {
	d := MaxBase + 1 // illegal base
	switch {
	case '0' <= ch && ch <= '9':
		d = ch - '0'
	case 'a' <= ch && ch <= 'z':
		d = ch - 'a' + 10
	case 'A' <= ch && ch <= 'Z':
		d = ch - 'A' + 10
	}
	return Word(d)
}

// scan sets z to the natural number corresponding to the longest possible prefix
// read from r representing an unsigned integer in a given conversion base.
// It returns z, the actual conversion base used, and an error, if any. In the
// error case, the value of z is undefined. The syntax follows the syntax of
// unsigned integer literals in Go.
//
// The base argument must be 0 or a value from 2 through MaxBase. If the base
// is 0, the string prefix determines the actual conversion base. A prefix of
// ``0x'' or ``0X'' selects base 16; the ``0'' prefix selects base 8, and a
// ``0b'' or ``0B'' prefix selects base 2. Otherwise the selected base is 10.
//
func (z nat) scan(r io.RuneScanner, base int) (nat, int, os.Error) {
	// reject illegal bases
	if base < 0 || base == 1 || MaxBase < base {
		return z, 0, os.NewError("illegal number base")
	}

	// one char look-ahead
	ch, _, err := r.ReadRune()
	if err != nil {
		return z, 0, err
	}

	// determine base if necessary
	b := Word(base)
	if base == 0 {
		b = 10
		if ch == '0' {
			switch ch, _, err = r.ReadRune(); err {
			case nil:
				b = 8
				switch ch {
				case 'x', 'X':
					b = 16
				case 'b', 'B':
					b = 2
				}
				if b == 2 || b == 16 {
					if ch, _, err = r.ReadRune(); err != nil {
						return z, 0, err
					}
				}
			case os.EOF:
				return z.make(0), 10, nil
			default:
				return z, 10, err
			}
		}
	}

	// convert string
	// - group as many digits d as possible together into a "super-digit" dd with "super-base" bb
	// - only when bb does not fit into a word anymore, do a full number mulAddWW using bb and dd
	z = z.make(0)
	bb := Word(1)
	dd := Word(0)
	for max := _M / b; ; {
		d := hexValue(ch)
		if d >= b {
			r.UnreadRune() // ch does not belong to number anymore
			break
		}

		if bb <= max {
			bb *= b
			dd = dd*b + d
		} else {
			// bb * b would overflow
			z = z.mulAddWW(z, bb, dd)
			bb = b
			dd = d
		}

		if ch, _, err = r.ReadRune(); err != nil {
			if err != os.EOF {
				return z, int(b), err
			}
			break
		}
	}

	switch {
	case bb > 1:
		// there was at least one mantissa digit
		z = z.mulAddWW(z, bb, dd)
	case base == 0 && b == 8:
		// there was only the octal prefix 0 (possibly followed by digits > 7);
		// return base 10, not 8
		return z, 10, nil
	case base != 0 || b != 8:
		// there was neither a mantissa digit nor the octal prefix 0
		return z, int(b), os.NewError("syntax error scanning number")
	}

	return z.norm(), int(b), nil
}

// Character sets for string conversion.
const (
	lowercaseDigits = "0123456789abcdefghijklmnopqrstuvwxyz"
	uppercaseDigits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

// decimalString returns a decimal representation of x.
// It calls x.string with the charset "0123456789".
func (x nat) decimalString() string {
	return x.string(lowercaseDigits[0:10])
}

// string converts x to a string using digits from a charset; a digit with
// value d is represented by charset[d]. The conversion base is determined
// by len(charset), which must be >= 2.
func (x nat) string(charset string) string {
	b := Word(len(charset))

	// special cases
	switch {
	case b < 2 || b > 256:
		panic("illegal base")
	case len(x) == 0:
		return string(charset[0])
	}

	// allocate buffer for conversion
	i := x.bitLen()/log2(b) + 1 // +1: round up
	s := make([]byte, i)

	// special case: power of two bases can avoid divisions completely
	if b == b&-b {
		// shift is base-b digit size in bits
		shift := uint(trailingZeroBits(b)) // shift > 0 because b >= 2
		mask := Word(1)<<shift - 1
		w := x[0]
		nbits := uint(_W) // number of unprocessed bits in w

		// convert less-significant words
		for k := 1; k < len(x); k++ {
			// convert full digits
			for nbits >= shift {
				i--
				s[i] = charset[w&mask]
				w >>= shift
				nbits -= shift
			}

			// convert any partial leading digit and advance to next word
			if nbits == 0 {
				// no partial digit remaining, just advance
				w = x[k]
				nbits = _W
			} else {
				// partial digit in current (k-1) and next (k) word
				w |= x[k] << nbits
				i--
				s[i] = charset[w&mask]

				// advance
				w = x[k] >> (shift - nbits)
				nbits = _W - (shift - nbits)
			}
		}

		// convert digits of most-significant word (omit leading zeros)
		for nbits >= 0 && w != 0 {
			i--
			s[i] = charset[w&mask]
			w >>= shift
			nbits -= shift
		}

		return string(s[i:])
	}

	// general case: extract groups of digits by multiprecision division

	// maximize ndigits where b**ndigits < 2^_W; bb (big base) is b**ndigits
	bb := Word(1)
	ndigits := 0
	for max := Word(_M / b); bb <= max; bb *= b {
		ndigits++
	}

	// preserve x, create local copy for use in repeated divisions
	q := nat(nil).set(x)
	var r Word

	// convert
	if b == 10 { // hard-coding for 10 here speeds this up by 1.25x
		for len(q) > 0 {
			// extract least significant, base bb "digit"
			q, r = q.divW(q, bb) // N.B. >82% of time is here. Optimize divW
			if len(q) == 0 {
				// skip leading zeros in most-significant group of digits
				for j := 0; j < ndigits && r != 0; j++ {
					i--
					s[i] = charset[r%10]
					r /= 10
				}
			} else {
				for j := 0; j < ndigits; j++ {
					i--
					s[i] = charset[r%10]
					r /= 10
				}
			}
		}
	} else {
		for len(q) > 0 {
			// extract least significant group of digits
			q, r = q.divW(q, bb) // N.B. >82% of time is here. Optimize divW
			if len(q) == 0 {
				// skip leading zeros in most-significant group of digits
				for j := 0; j < ndigits && r != 0; j++ {
					i--
					s[i] = charset[r%b]
					r /= b
				}
			} else {
				for j := 0; j < ndigits; j++ {
					i--
					s[i] = charset[r%b]
					r /= b
				}
			}
		}
	}

	return string(s[i:])
}

const deBruijn32 = 0x077CB531

var deBruijn32Lookup = []byte{
	0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
	31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9,
}

const deBruijn64 = 0x03f79d71b4ca8b09

var deBruijn64Lookup = []byte{
	0, 1, 56, 2, 57, 49, 28, 3, 61, 58, 42, 50, 38, 29, 17, 4,
	62, 47, 59, 36, 45, 43, 51, 22, 53, 39, 33, 30, 24, 18, 12, 5,
	63, 55, 48, 27, 60, 41, 37, 16, 46, 35, 44, 21, 52, 32, 23, 11,
	54, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9, 13, 8, 7, 6,
}

// trailingZeroBits returns the number of consecutive zero bits on the right
// side of the given Word.
// See Knuth, volume 4, section 7.3.1
func trailingZeroBits(x Word) int {
	// x & -x leaves only the right-most bit set in the word. Let k be the
	// index of that bit. Since only a single bit is set, the value is two
	// to the power of k. Multiplying by a power of two is equivalent to
	// left shifting, in this case by k bits.  The de Bruijn constant is
	// such that all six bit, consecutive substrings are distinct.
	// Therefore, if we have a left shifted version of this constant we can
	// find by how many bits it was shifted by looking at which six bit
	// substring ended up at the top of the word.
	switch _W {
	case 32:
		return int(deBruijn32Lookup[((x&-x)*deBruijn32)>>27])
	case 64:
		return int(deBruijn64Lookup[((x&-x)*(deBruijn64&_M))>>58])
	default:
		panic("Unknown word size")
	}

	return 0
}

// z = x << s
func (z nat) shl(x nat, s uint) nat {
	m := len(x)
	if m == 0 {
		return z.make(0)
	}
	// m > 0

	n := m + int(s/_W)
	z = z.make(n + 1)
	z[n] = shlVU(z[n-m:n], x, s%_W)
	z[0 : n-m].clear()

	return z.norm()
}

// z = x >> s
func (z nat) shr(x nat, s uint) nat {
	m := len(x)
	n := m - int(s/_W)
	if n <= 0 {
		return z.make(0)
	}
	// n > 0

	z = z.make(n)
	shrVU(z, x[m-n:], s%_W)

	return z.norm()
}

func (z nat) setBit(x nat, i uint, b uint) nat {
	j := int(i / _W)
	m := Word(1) << (i % _W)
	n := len(x)
	switch b {
	case 0:
		z = z.make(n)
		copy(z, x)
		if j >= n {
			// no need to grow
			return z
		}
		z[j] &^= m
		return z.norm()
	case 1:
		if j >= n {
			n = j + 1
		}
		z = z.make(n)
		copy(z, x)
		z[j] |= m
		// no need to normalize
		return z
	}
	panic("set bit is not 0 or 1")
}

func (z nat) bit(i uint) uint {
	j := int(i / _W)
	if j >= len(z) {
		return 0
	}
	return uint(z[j] >> (i % _W) & 1)
}

func (z nat) and(x, y nat) nat {
	m := len(x)
	n := len(y)
	if m > n {
		m = n
	}
	// m <= n

	z = z.make(m)
	for i := 0; i < m; i++ {
		z[i] = x[i] & y[i]
	}

	return z.norm()
}

func (z nat) andNot(x, y nat) nat {
	m := len(x)
	n := len(y)
	if n > m {
		n = m
	}
	// m >= n

	z = z.make(m)
	for i := 0; i < n; i++ {
		z[i] = x[i] &^ y[i]
	}
	copy(z[n:m], x[n:m])

	return z.norm()
}

func (z nat) or(x, y nat) nat {
	m := len(x)
	n := len(y)
	s := x
	if m < n {
		n, m = m, n
		s = y
	}
	// m >= n

	z = z.make(m)
	for i := 0; i < n; i++ {
		z[i] = x[i] | y[i]
	}
	copy(z[n:m], s[n:m])

	return z.norm()
}

func (z nat) xor(x, y nat) nat {
	m := len(x)
	n := len(y)
	s := x
	if m < n {
		n, m = m, n
		s = y
	}
	// m >= n

	z = z.make(m)
	for i := 0; i < n; i++ {
		z[i] = x[i] ^ y[i]
	}
	copy(z[n:m], s[n:m])

	return z.norm()
}

// greaterThan returns true iff (x1<<_W + x2) > (y1<<_W + y2)
func greaterThan(x1, x2, y1, y2 Word) bool {
	return x1 > y1 || x1 == y1 && x2 > y2
}

// modW returns x % d.
func (x nat) modW(d Word) (r Word) {
	// TODO(agl): we don't actually need to store the q value.
	var q nat
	q = q.make(len(x))
	return divWVW(q, 0, x, d)
}

// powersOfTwoDecompose finds q and k with x = q * 1<<k and q is odd, or q and k are 0.
func (x nat) powersOfTwoDecompose() (q nat, k int) {
	if len(x) == 0 {
		return x, 0
	}

	// One of the words must be non-zero by definition,
	// so this loop will terminate with i < len(x), and
	// i is the number of 0 words.
	i := 0
	for x[i] == 0 {
		i++
	}
	n := trailingZeroBits(x[i]) // x[i] != 0

	q = make(nat, len(x)-i)
	shrVU(q, x[i:], uint(n))

	q = q.norm()
	k = i*_W + n
	return
}

// random creates a random integer in [0..limit), using the space in z if
// possible. n is the bit length of limit.
func (z nat) random(rand *rand.Rand, limit nat, n int) nat {
	bitLengthOfMSW := uint(n % _W)
	if bitLengthOfMSW == 0 {
		bitLengthOfMSW = _W
	}
	mask := Word((1 << bitLengthOfMSW) - 1)
	z = z.make(len(limit))

	for {
		for i := range z {
			switch _W {
			case 32:
				z[i] = Word(rand.Uint32())
			case 64:
				z[i] = Word(rand.Uint32()) | Word(rand.Uint32())<<32
			}
		}

		z[len(limit)-1] &= mask

		if z.cmp(limit) < 0 {
			break
		}
	}

	return z.norm()
}

// If m != nil, expNN calculates x**y mod m. Otherwise it calculates x**y. It
// reuses the storage of z if possible.
func (z nat) expNN(x, y, m nat) nat {
	if alias(z, x) || alias(z, y) {
		// We cannot allow in place modification of x or y.
		z = nil
	}

	if len(y) == 0 {
		z = z.make(1)
		z[0] = 1
		return z
	}

	if m != nil {
		// We likely end up being as long as the modulus.
		z = z.make(len(m))
	}
	z = z.set(x)
	v := y[len(y)-1]
	// It's invalid for the most significant word to be zero, therefore we
	// will find a one bit.
	shift := leadingZeros(v) + 1
	v <<= shift
	var q nat

	const mask = 1 << (_W - 1)

	// We walk through the bits of the exponent one by one. Each time we
	// see a bit, we square, thus doubling the power. If the bit is a one,
	// we also multiply by x, thus adding one to the power.

	w := _W - int(shift)
	for j := 0; j < w; j++ {
		z = z.mul(z, z)

		if v&mask != 0 {
			z = z.mul(z, x)
		}

		if m != nil {
			q, z = q.div(z, z, m)
		}

		v <<= 1
	}

	for i := len(y) - 2; i >= 0; i-- {
		v = y[i]

		for j := 0; j < _W; j++ {
			z = z.mul(z, z)

			if v&mask != 0 {
				z = z.mul(z, x)
			}

			if m != nil {
				q, z = q.div(z, z, m)
			}

			v <<= 1
		}
	}

	return z
}

// probablyPrime performs reps Miller-Rabin tests to check whether n is prime.
// If it returns true, n is prime with probability 1 - 1/4^reps.
// If it returns false, n is not prime.
func (n nat) probablyPrime(reps int) bool {
	if len(n) == 0 {
		return false
	}

	if len(n) == 1 {
		if n[0] < 2 {
			return false
		}

		if n[0]%2 == 0 {
			return n[0] == 2
		}

		// We have to exclude these cases because we reject all
		// multiples of these numbers below.
		switch n[0] {
		case 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53:
			return true
		}
	}

	const primesProduct32 = 0xC0CFD797         // Π {p ∈ primes, 2 < p <= 29}
	const primesProduct64 = 0xE221F97C30E94E1D // Π {p ∈ primes, 2 < p <= 53}

	var r Word
	switch _W {
	case 32:
		r = n.modW(primesProduct32)
	case 64:
		r = n.modW(primesProduct64 & _M)
	default:
		panic("Unknown word size")
	}

	if r%3 == 0 || r%5 == 0 || r%7 == 0 || r%11 == 0 ||
		r%13 == 0 || r%17 == 0 || r%19 == 0 || r%23 == 0 || r%29 == 0 {
		return false
	}

	if _W == 64 && (r%31 == 0 || r%37 == 0 || r%41 == 0 ||
		r%43 == 0 || r%47 == 0 || r%53 == 0) {
		return false
	}

	nm1 := nat(nil).sub(n, natOne)
	// 1<<k * q = nm1;
	q, k := nm1.powersOfTwoDecompose()

	nm3 := nat(nil).sub(nm1, natTwo)
	rand := rand.New(rand.NewSource(int64(n[0])))

	var x, y, quotient nat
	nm3Len := nm3.bitLen()

NextRandom:
	for i := 0; i < reps; i++ {
		x = x.random(rand, nm3, nm3Len)
		x = x.add(x, natTwo)
		y = y.expNN(x, q, n)
		if y.cmp(natOne) == 0 || y.cmp(nm1) == 0 {
			continue
		}
		for j := 1; j < k; j++ {
			y = y.mul(y, y)
			quotient, y = quotient.div(y, y, n)
			if y.cmp(nm1) == 0 {
				continue NextRandom
			}
			if y.cmp(natOne) == 0 {
				return false
			}
		}
		return false
	}

	return true
}

// bytes writes the value of z into buf using big-endian encoding.
// len(buf) must be >= len(z)*_S. The value of z is encoded in the
// slice buf[i:]. The number i of unused bytes at the beginning of
// buf is returned as result.
func (z nat) bytes(buf []byte) (i int) {
	i = len(buf)
	for _, d := range z {
		for j := 0; j < _S; j++ {
			i--
			buf[i] = byte(d)
			d >>= 8
		}
	}

	for i < len(buf) && buf[i] == 0 {
		i++
	}

	return
}

// setBytes interprets buf as the bytes of a big-endian unsigned
// integer, sets z to that value, and returns z.
func (z nat) setBytes(buf []byte) nat {
	z = z.make((len(buf) + _S - 1) / _S)

	k := 0
	s := uint(0)
	var d Word
	for i := len(buf); i > 0; i-- {
		d |= Word(buf[i-1]) << s
		if s += 8; s == _S*8 {
			z[k] = d
			k++
			s = 0
			d = 0
		}
	}
	if k < len(z) {
		z[k] = d
	}

	return z.norm()
}
