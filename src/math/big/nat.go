// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package big implements multi-precision arithmetic (big numbers).
// The following numeric types are supported:
//
//   Int    signed integers
//   Rat    rational numbers
//   Float  floating-point numbers
//
// Methods are typically of the form:
//
//   func (z *T) Unary(x *T) *T        // z = op x
//   func (z *T) Binary(x, y *T) *T    // z = x op y
//   func (x *T) M() T1                // v = x.M()
//
// with T one of Int, Rat, or Float. For unary and binary operations, the
// result is the receiver (usually named z in that case); if it is one of
// the operands x or y it may be overwritten (and its memory reused).
// To enable chaining of operations, the result is also returned. Methods
// returning a result other than *Int, *Rat, or *Float take an operand as
// the receiver (usually named x in that case).
//
package big

// This file contains operations on unsigned multi-precision integers.
// These are the building blocks for the operations on signed integers
// and rationals.

import (
	"errors"
	"io"
	"math"
	"math/rand"
	"sync"
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
//
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
		return z[:n] // reuse z
	}
	// Choosing a good value for e has significant performance impact
	// because it increases the chance that a value can be reused.
	const e = 4 // extra capacity
	return make(nat, n, n+e)
}

func (z nat) setWord(x Word) nat {
	if x == 0 {
		return z[:0]
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
		return z[:0]
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
		return z[:0]
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
var karatsubaThreshold int = 40 // computed by calibrate.go

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
	//   z1 =      xd*yd                    + z2 + z0
	//      = (x1-x0)*(y0 - y1)             + z2 + z0
	//      = x1*y0 - x1*y1 - x0*y0 + x0*y1 + z2 + z0
	//      = x1*y0 -    z2 -    z0 + x0*y1 + z2 + z0
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
	copy(r, z[:n*2])

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

// addAt implements z += x<<(_W*i); z must be long enough.
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
		return z[:0]
	case n == 1:
		return z.mulAddWW(x, y[0], 0)
	}
	// m >= n > 1

	// determine if z can be reused
	if alias(z, x) || alias(z, y) {
		z = nil // z is an alias for x or y - cannot reuse
	}

	// use basic multiplication if the numbers are small
	if n < karatsubaThreshold {
		z = z.make(m + n)
		basicMul(z, x, y)
		return z.norm()
	}
	// m >= n && n >= karatsubaThreshold && n >= 2

	// determine Karatsuba length k such that
	//
	//   x = xh*b + x0  (0 <= x0 < b)
	//   y = yh*b + y0  (0 <= y0 < b)
	//   b = 1<<(_W*k)  ("base" of digits xi, yi)
	//
	k := karatsubaLen(n)
	// k <= n

	// multiply x0 and y0 via Karatsuba
	x0 := x[0:k]              // x0 is not normalized
	y0 := y[0:k]              // y0 is not normalized
	z = z.make(max(6*k, m+n)) // enough space for karatsuba of x0*y0 and full result of x*y
	karatsuba(z, x0, y0)
	z = z[0 : m+n]  // z has final length but may be incomplete
	z[2*k:].clear() // upper portion of z is garbage (and 2*k <= m+n since k <= n <= m)

	// If xh != 0 or yh != 0, add the missing terms to z. For
	//
	//   xh = xi*b^i + ... + x2*b^2 + x1*b (0 <= xi < b)
	//   yh =                         y1*b (0 <= y1 < b)
	//
	// the missing terms are
	//
	//   x0*y1*b and xi*y0*b^i, xi*y1*b^(i+1) for i > 0
	//
	// since all the yi for i > 1 are 0 by choice of k: If any of them
	// were > 0, then yh >= b^2 and thus y >= b^2. Then k' = k*2 would
	// be a larger valid threshold contradicting the assumption about k.
	//
	if k < n || m != n {
		var t nat

		// add x0*y1*b
		x0 := x0.norm()
		y1 := y[k:]       // y1 is normalized because y is
		t = t.mul(x0, y1) // update t so we don't lose t's underlying array
		addAt(z, t, k)

		// add xi*y0<<i, xi*y1*b<<(i+k)
		y0 := y0.norm()
		for i := k; i < len(x); i += k {
			xi := x[i:]
			if len(xi) > k {
				xi = xi[:k]
			}
			xi = xi.norm()
			t = t.mul(xi, y0)
			addAt(z, t, i)
			t = t.mul(xi, y1)
			addAt(z, t, i+k)
		}
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
		q = z[:0] // result is 0
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
		q = z[:0]
		r = z2.set(u)
		return
	}

	if len(v) == 1 {
		var r2 Word
		q, r2 = z.divW(u, v[0])
		r = z2.setWord(r2)
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
	u.clear() // TODO(gri) no need to clear if we allocated a new u

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
const MaxBase = 'z' - 'a' + 10 + 1

// maxPow returns (b**n, n) such that b**n is the largest power b**n <= _M.
// For instance maxPow(10) == (1e19, 19) for 19 decimal digits in a 64bit Word.
// In other words, at most n digits in base b fit into a Word.
// TODO(gri) replace this with a table, generated at build time.
func maxPow(b Word) (p Word, n int) {
	p, n = b, 1 // assuming b <= _M
	for max := _M / b; p <= max; {
		// p == b**n && p <= max
		p *= b
		n++
	}
	// p == b**n && p <= _M
	return
}

// pow returns x**n for n > 0, and 1 otherwise.
func pow(x Word, n int) (p Word) {
	// n == sum of bi * 2**i, for 0 <= i < imax, and bi is 0 or 1
	// thus x**n == product of x**(2**i) for all i where bi == 1
	// (Russian Peasant Method for exponentiation)
	p = 1
	for n > 0 {
		if n&1 != 0 {
			p *= x
		}
		x *= x
		n >>= 1
	}
	return
}

// scan scans the number corresponding to the longest possible prefix
// from r representing an unsigned number in a given conversion base.
// It returns the corresponding natural number res, the actual base b,
// a digit count, and an error err, if any.
//
//	number = [ prefix ] digits | digits "." [ digits ] | "." digits .
//	prefix = "0" [ "x" | "X" | "b" | "B" ] .
//	digits = digit { digit } .
//	digit  = "0" ... "9" | "a" ... "z" | "A" ... "Z" .
//
// The base argument must be a value between 0 and MaxBase (inclusive).
// For base 0, the number prefix determines the actual base: A prefix of
// ``0x'' or ``0X'' selects base 16; the ``0'' prefix selects base 8, and
// a ``0b'' or ``0B'' prefix selects base 2. Otherwise the selected base
// is 10 and no prefix is permitted.
//
// Base argument 1 selects actual base 10 but also enables scanning a number
// with a decimal point.
//
// A result digit count > 0 corresponds to the number of (non-prefix) digits
// parsed. A digit count <= 0 indicates the presence of a decimal point (for
// base == 1, only), and the number of fractional digits is -count. In this
// case, the value of the scanned number is res * 10**count.
//
func (z nat) scan(r io.RuneScanner, base int) (res nat, b, count int, err error) {
	// reject illegal bases
	if base < 0 || base > MaxBase {
		err = errors.New("illegal number base")
		return
	}

	// one char look-ahead
	ch, _, err := r.ReadRune()
	if err != nil {
		return
	}

	// determine actual base
	switch base {
	case 0:
		// actual base is 10 unless there's a base prefix
		b = 10
		if ch == '0' {
			switch ch, _, err = r.ReadRune(); err {
			case nil:
				// possibly one of 0x, 0X, 0b, 0B
				b = 8
				switch ch {
				case 'x', 'X':
					b = 16
				case 'b', 'B':
					b = 2
				}
				if b == 2 || b == 16 {
					if ch, _, err = r.ReadRune(); err != nil {
						// io.EOF is also an error in this case
						return
					}
				}
			case io.EOF:
				// input is "0"
				res = z[:0]
				count = 1
				err = nil
				return
			default:
				// read error
				return
			}
		}
	case 1:
		// actual base is 10 and decimal point is permitted
		b = 10
	default:
		b = base
	}

	// convert string
	// Algorithm: Collect digits in groups of at most n digits in di
	// and then use mulAddWW for every such group to add them to the
	// result.
	z = z[:0]
	b1 := Word(b)
	bn, n := maxPow(b1) // at most n digits in base b1 fit into Word
	di := Word(0)       // 0 <= di < b1**i < bn
	i := 0              // 0 <= i < n
	dp := -1            // position of decimal point
	for {
		if base == 1 && ch == '.' {
			base = 10 // no 2nd decimal point permitted
			dp = count
			// advance
			if ch, _, err = r.ReadRune(); err != nil {
				if err == io.EOF {
					err = nil
					break
				}
				return
			}
		}

		// convert rune into digit value d1
		var d1 Word
		switch {
		case '0' <= ch && ch <= '9':
			d1 = Word(ch - '0')
		case 'a' <= ch && ch <= 'z':
			d1 = Word(ch - 'a' + 10)
		case 'A' <= ch && ch <= 'Z':
			d1 = Word(ch - 'A' + 10)
		default:
			d1 = MaxBase + 1
		}
		if d1 >= b1 {
			r.UnreadRune() // ch does not belong to number anymore
			break
		}
		count++

		// collect d1 in di
		di = di*b1 + d1
		i++

		// if di is "full", add it to the result
		if i == n {
			z = z.mulAddWW(z, bn, di)
			di = 0
			i = 0
		}

		// advance
		if ch, _, err = r.ReadRune(); err != nil {
			if err == io.EOF {
				err = nil
				break
			}
			return
		}
	}

	if count == 0 {
		// no digits found
		switch {
		case base == 0 && b == 8:
			// there was only the octal prefix 0 (possibly followed by digits > 7);
			// count as one digit and return base 10, not 8
			count = 1
			b = 10
		case base != 0 || b != 8:
			// there was neither a mantissa digit nor the octal prefix 0
			err = errors.New("syntax error scanning number")
		}
		return
	}
	// count > 0

	// add remaining digits to result
	if i > 0 {
		z = z.mulAddWW(z, pow(b1, i), di)
	}
	res = z.norm()

	// adjust for fraction, if any
	if dp >= 0 {
		// 0 <= dp <= count > 0
		count = dp - count
	}

	return
}

// Character sets for string conversion.
const (
	lowercaseDigits = "0123456789abcdefghijklmnopqrstuvwxyz"
	uppercaseDigits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

// decimalString returns a decimal representation of x.
// It calls x.string with the charset "0123456789".
func (x nat) decimalString() string {
	return x.string(lowercaseDigits[:10])
}

// hexString returns a hexadecimal representation of x.
// It calls x.string with the charset "0123456789abcdef".
func (x nat) hexString() string {
	return x.string(lowercaseDigits[:16])
}

// string converts x to a string using digits from a charset; a digit with
// value d is represented by charset[d]. The conversion base is determined
// by len(charset), which must be >= 2 and <= 256.
func (x nat) string(charset string) string {
	b := Word(len(charset))

	// special cases
	switch {
	case b < 2 || b > 256:
		panic("invalid character set length")
	case len(x) == 0:
		return string(charset[0])
	}

	// allocate buffer for conversion
	i := int(float64(x.bitLen())/math.Log2(float64(b))) + 1 // off by one at most
	s := make([]byte, i)

	// convert power of two and non power of two bases separately
	if b == b&-b {
		// shift is base-b digit size in bits
		shift := trailingZeroBits(b) // shift > 0 because b >= 2
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

	} else {
		bb, ndigits := maxPow(Word(b))

		// construct table of successive squares of bb*leafSize to use in subdivisions
		// result (table != nil) <=> (len(x) > leafSize > 0)
		table := divisors(len(x), b, ndigits, bb)

		// preserve x, create local copy for use by convertWords
		q := nat(nil).set(x)

		// convert q to string s in base b
		q.convertWords(s, charset, b, ndigits, bb, table)

		// strip leading zeros
		// (x != 0; thus s must contain at least one non-zero digit
		// and the loop will terminate)
		i = 0
		for zero := charset[0]; s[i] == zero; {
			i++
		}
	}

	return string(s[i:])
}

// Convert words of q to base b digits in s. If q is large, it is recursively "split in half"
// by nat/nat division using tabulated divisors. Otherwise, it is converted iteratively using
// repeated nat/Word division.
//
// The iterative method processes n Words by n divW() calls, each of which visits every Word in the
// incrementally shortened q for a total of n + (n-1) + (n-2) ... + 2 + 1, or n(n+1)/2 divW()'s.
// Recursive conversion divides q by its approximate square root, yielding two parts, each half
// the size of q. Using the iterative method on both halves means 2 * (n/2)(n/2 + 1)/2 divW()'s
// plus the expensive long div(). Asymptotically, the ratio is favorable at 1/2 the divW()'s, and
// is made better by splitting the subblocks recursively. Best is to split blocks until one more
// split would take longer (because of the nat/nat div()) than the twice as many divW()'s of the
// iterative approach. This threshold is represented by leafSize. Benchmarking of leafSize in the
// range 2..64 shows that values of 8 and 16 work well, with a 4x speedup at medium lengths and
// ~30x for 20000 digits. Use nat_test.go's BenchmarkLeafSize tests to optimize leafSize for
// specific hardware.
//
func (q nat) convertWords(s []byte, charset string, b Word, ndigits int, bb Word, table []divisor) {
	// split larger blocks recursively
	if table != nil {
		// len(q) > leafSize > 0
		var r nat
		index := len(table) - 1
		for len(q) > leafSize {
			// find divisor close to sqrt(q) if possible, but in any case < q
			maxLength := q.bitLen()     // ~= log2 q, or at of least largest possible q of this bit length
			minLength := maxLength >> 1 // ~= log2 sqrt(q)
			for index > 0 && table[index-1].nbits > minLength {
				index-- // desired
			}
			if table[index].nbits >= maxLength && table[index].bbb.cmp(q) >= 0 {
				index--
				if index < 0 {
					panic("internal inconsistency")
				}
			}

			// split q into the two digit number (q'*bbb + r) to form independent subblocks
			q, r = q.div(r, q, table[index].bbb)

			// convert subblocks and collect results in s[:h] and s[h:]
			h := len(s) - table[index].ndigits
			r.convertWords(s[h:], charset, b, ndigits, bb, table[0:index])
			s = s[:h] // == q.convertWords(s, charset, b, ndigits, bb, table[0:index+1])
		}
	}

	// having split any large blocks now process the remaining (small) block iteratively
	i := len(s)
	var r Word
	if b == 10 {
		// hard-coding for 10 here speeds this up by 1.25x (allows for / and % by constants)
		for len(q) > 0 {
			// extract least significant, base bb "digit"
			q, r = q.divW(q, bb)
			for j := 0; j < ndigits && i > 0; j++ {
				i--
				// avoid % computation since r%10 == r - int(r/10)*10;
				// this appears to be faster for BenchmarkString10000Base10
				// and smaller strings (but a bit slower for larger ones)
				t := r / 10
				s[i] = charset[r-t<<3-t-t] // TODO(gri) replace w/ t*10 once compiler produces better code
				r = t
			}
		}
	} else {
		for len(q) > 0 {
			// extract least significant, base bb "digit"
			q, r = q.divW(q, bb)
			for j := 0; j < ndigits && i > 0; j++ {
				i--
				s[i] = charset[r%b]
				r /= b
			}
		}
	}

	// prepend high-order zeroes
	zero := charset[0]
	for i > 0 { // while need more leading zeroes
		i--
		s[i] = zero
	}
}

// Split blocks greater than leafSize Words (or set to 0 to disable recursive conversion)
// Benchmark and configure leafSize using: go test -bench="Leaf"
//   8 and 16 effective on 3.0 GHz Xeon "Clovertown" CPU (128 byte cache lines)
//   8 and 16 effective on 2.66 GHz Core 2 Duo "Penryn" CPU
var leafSize int = 8 // number of Word-size binary values treat as a monolithic block

type divisor struct {
	bbb     nat // divisor
	nbits   int // bit length of divisor (discounting leading zeroes) ~= log2(bbb)
	ndigits int // digit length of divisor in terms of output base digits
}

var cacheBase10 struct {
	sync.Mutex
	table [64]divisor // cached divisors for base 10
}

// expWW computes x**y
func (z nat) expWW(x, y Word) nat {
	return z.expNN(nat(nil).setWord(x), nat(nil).setWord(y), nil)
}

// construct table of powers of bb*leafSize to use in subdivisions
func divisors(m int, b Word, ndigits int, bb Word) []divisor {
	// only compute table when recursive conversion is enabled and x is large
	if leafSize == 0 || m <= leafSize {
		return nil
	}

	// determine k where (bb**leafSize)**(2**k) >= sqrt(x)
	k := 1
	for words := leafSize; words < m>>1 && k < len(cacheBase10.table); words <<= 1 {
		k++
	}

	// reuse and extend existing table of divisors or create new table as appropriate
	var table []divisor // for b == 10, table overlaps with cacheBase10.table
	if b == 10 {
		cacheBase10.Lock()
		table = cacheBase10.table[0:k] // reuse old table for this conversion
	} else {
		table = make([]divisor, k) // create new table for this conversion
	}

	// extend table
	if table[k-1].ndigits == 0 {
		// add new entries as needed
		var larger nat
		for i := 0; i < k; i++ {
			if table[i].ndigits == 0 {
				if i == 0 {
					table[0].bbb = nat(nil).expWW(bb, Word(leafSize))
					table[0].ndigits = ndigits * leafSize
				} else {
					table[i].bbb = nat(nil).mul(table[i-1].bbb, table[i-1].bbb)
					table[i].ndigits = 2 * table[i-1].ndigits
				}

				// optimization: exploit aggregated extra bits in macro blocks
				larger = nat(nil).set(table[i].bbb)
				for mulAddVWW(larger, larger, b, 0) == 0 {
					table[i].bbb = table[i].bbb.set(larger)
					table[i].ndigits++
				}

				table[i].nbits = table[i].bbb.bitLen()
			}
		}
	}

	if b == 10 {
		cacheBase10.Unlock()
	}

	return table
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

// trailingZeroBits returns the number of consecutive least significant zero
// bits of x.
func trailingZeroBits(x Word) uint {
	// x & -x leaves only the right-most bit set in the word. Let k be the
	// index of that bit. Since only a single bit is set, the value is two
	// to the power of k. Multiplying by a power of two is equivalent to
	// left shifting, in this case by k bits.  The de Bruijn constant is
	// such that all six bit, consecutive substrings are distinct.
	// Therefore, if we have a left shifted version of this constant we can
	// find by how many bits it was shifted by looking at which six bit
	// substring ended up at the top of the word.
	// (Knuth, volume 4, section 7.3.1)
	switch _W {
	case 32:
		return uint(deBruijn32Lookup[((x&-x)*deBruijn32)>>27])
	case 64:
		return uint(deBruijn64Lookup[((x&-x)*(deBruijn64&_M))>>58])
	default:
		panic("unknown word size")
	}
}

// trailingZeroBits returns the number of consecutive least significant zero
// bits of x.
func (x nat) trailingZeroBits() uint {
	if len(x) == 0 {
		return 0
	}
	var i uint
	for x[i] == 0 {
		i++
	}
	// x[i] != 0
	return i*_W + trailingZeroBits(x[i])
}

// z = x << s
func (z nat) shl(x nat, s uint) nat {
	m := len(x)
	if m == 0 {
		return z[:0]
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
		return z[:0]
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
			z = z.make(j + 1)
			z[n:].clear()
		} else {
			z = z.make(n)
		}
		copy(z, x)
		z[j] |= m
		// no need to normalize
		return z
	}
	panic("set bit is not 0 or 1")
}

// bit returns the value of the i'th bit, with lsb == bit 0.
func (x nat) bit(i uint) uint {
	j := i / _W
	if j >= uint(len(x)) {
		return 0
	}
	// 0 <= j < len(x)
	return uint(x[j] >> (i % _W) & 1)
}

// sticky returns 1 if there's a 1 bit within the
// i least significant bits, otherwise it returns 0.
func (x nat) sticky(i uint) uint {
	j := i / _W
	if j >= uint(len(x)) {
		if len(x) == 0 {
			return 0
		}
		return 1
	}
	// 0 <= j < len(x)
	for _, x := range x[:j] {
		if x != 0 {
			return 1
		}
	}
	if x[j]<<(_W-i%_W) != 0 {
		return 1
	}
	return 0
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

// random creates a random integer in [0..limit), using the space in z if
// possible. n is the bit length of limit.
func (z nat) random(rand *rand.Rand, limit nat, n int) nat {
	if alias(z, limit) {
		z = nil // z is an alias for limit - cannot reuse
	}
	z = z.make(len(limit))

	bitLengthOfMSW := uint(n % _W)
	if bitLengthOfMSW == 0 {
		bitLengthOfMSW = _W
	}
	mask := Word((1 << bitLengthOfMSW) - 1)

	for {
		switch _W {
		case 32:
			for i := range z {
				z[i] = Word(rand.Uint32())
			}
		case 64:
			for i := range z {
				z[i] = Word(rand.Uint32()) | Word(rand.Uint32())<<32
			}
		default:
			panic("unknown word size")
		}
		z[len(limit)-1] &= mask
		if z.cmp(limit) < 0 {
			break
		}
	}

	return z.norm()
}

// If m != 0 (i.e., len(m) != 0), expNN sets z to x**y mod m;
// otherwise it sets z to x**y. The result is the value of z.
func (z nat) expNN(x, y, m nat) nat {
	if alias(z, x) || alias(z, y) {
		// We cannot allow in-place modification of x or y.
		z = nil
	}

	// x**y mod 1 == 0
	if len(m) == 1 && m[0] == 1 {
		return z.setWord(0)
	}
	// m == 0 || m > 1

	// x**0 == 1
	if len(y) == 0 {
		return z.setWord(1)
	}
	// y > 0

	if len(m) != 0 {
		// We likely end up being as long as the modulus.
		z = z.make(len(m))
	}
	z = z.set(x)

	// If the base is non-trivial and the exponent is large, we use
	// 4-bit, windowed exponentiation. This involves precomputing 14 values
	// (x^2...x^15) but then reduces the number of multiply-reduces by a
	// third. Even for a 32-bit exponent, this reduces the number of
	// operations.
	if len(x) > 1 && len(y) > 1 && len(m) > 0 {
		return z.expNNWindowed(x, y, m)
	}

	v := y[len(y)-1] // v > 0 because y is normalized and y > 0
	shift := leadingZeros(v) + 1
	v <<= shift
	var q nat

	const mask = 1 << (_W - 1)

	// We walk through the bits of the exponent one by one. Each time we
	// see a bit, we square, thus doubling the power. If the bit is a one,
	// we also multiply by x, thus adding one to the power.

	w := _W - int(shift)
	// zz and r are used to avoid allocating in mul and div as
	// otherwise the arguments would alias.
	var zz, r nat
	for j := 0; j < w; j++ {
		zz = zz.mul(z, z)
		zz, z = z, zz

		if v&mask != 0 {
			zz = zz.mul(z, x)
			zz, z = z, zz
		}

		if len(m) != 0 {
			zz, r = zz.div(r, z, m)
			zz, r, q, z = q, z, zz, r
		}

		v <<= 1
	}

	for i := len(y) - 2; i >= 0; i-- {
		v = y[i]

		for j := 0; j < _W; j++ {
			zz = zz.mul(z, z)
			zz, z = z, zz

			if v&mask != 0 {
				zz = zz.mul(z, x)
				zz, z = z, zz
			}

			if len(m) != 0 {
				zz, r = zz.div(r, z, m)
				zz, r, q, z = q, z, zz, r
			}

			v <<= 1
		}
	}

	return z.norm()
}

// expNNWindowed calculates x**y mod m using a fixed, 4-bit window.
func (z nat) expNNWindowed(x, y, m nat) nat {
	// zz and r are used to avoid allocating in mul and div as otherwise
	// the arguments would alias.
	var zz, r nat

	const n = 4
	// powers[i] contains x^i.
	var powers [1 << n]nat
	powers[0] = natOne
	powers[1] = x
	for i := 2; i < 1<<n; i += 2 {
		p2, p, p1 := &powers[i/2], &powers[i], &powers[i+1]
		*p = p.mul(*p2, *p2)
		zz, r = zz.div(r, *p, m)
		*p, r = r, *p
		*p1 = p1.mul(*p, x)
		zz, r = zz.div(r, *p1, m)
		*p1, r = r, *p1
	}

	z = z.setWord(1)

	for i := len(y) - 1; i >= 0; i-- {
		yi := y[i]
		for j := 0; j < _W; j += n {
			if i != len(y)-1 || j != 0 {
				// Unrolled loop for significant performance
				// gain.  Use go test -bench=".*" in crypto/rsa
				// to check performance before making changes.
				zz = zz.mul(z, z)
				zz, z = z, zz
				zz, r = zz.div(r, z, m)
				z, r = r, z

				zz = zz.mul(z, z)
				zz, z = z, zz
				zz, r = zz.div(r, z, m)
				z, r = r, z

				zz = zz.mul(z, z)
				zz, z = z, zz
				zz, r = zz.div(r, z, m)
				z, r = r, z

				zz = zz.mul(z, z)
				zz, z = z, zz
				zz, r = zz.div(r, z, m)
				z, r = r, z
			}

			zz = zz.mul(z, powers[yi>>(_W-n)])
			zz, z = z, zz
			zz, r = zz.div(r, z, m)
			z, r = r, z

			yi <<= n
		}
	}

	return z.norm()
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

	if n[0]&1 == 0 {
		return false // n is even
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
	// determine q, k such that nm1 = q << k
	k := nm1.trailingZeroBits()
	q := nat(nil).shr(nm1, k)

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
		for j := uint(1); j < k; j++ {
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
