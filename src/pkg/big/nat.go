// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains operations on unsigned multi-precision integers.
// These are the building blocks for the operations on signed integers
// and rationals.

// This package implements multi-precision arithmetic (big numbers).
// The following numeric types are supported:
//
//	- Int	signed integers
//
// All methods on Int take the result as the receiver; if it is one
// of the operands it may be overwritten (and its memory reused).
// To enable chaining of operations, the result is also returned.
//
// If possible, one should use big over bignum as the latter is headed for
// deprecation.
//
package big

import "rand"

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

// TODO(gri) - convert these routines into methods for type 'nat'
//           - decide if type 'nat' should be exported

func normN(z []Word) []Word {
	i := len(z)
	for i > 0 && z[i-1] == 0 {
		i--
	}
	z = z[0:i]
	return z
}


func makeN(z []Word, m int, clear bool) []Word {
	if len(z) > m {
		z = z[0:m] // reuse z - has at least one extra word for a carry, if any
		if clear {
			for i := range z {
				z[i] = 0
			}
		}
		return z
	}

	c := 4 // minimum capacity
	if m > c {
		c = m
	}
	return make([]Word, m, c+1) // +1: extra word for a carry, if any
}


func newN(z []Word, x uint64) []Word {
	if x == 0 {
		return makeN(z, 0, false)
	}

	// single-digit values
	if x == uint64(Word(x)) {
		z = makeN(z, 1, false)
		z[0] = Word(x)
		return z
	}

	// compute number of words n required to represent x
	n := 0
	for t := x; t > 0; t >>= _W {
		n++
	}

	// split x into n words
	z = makeN(z, n, false)
	for i := 0; i < n; i++ {
		z[i] = Word(x & _M)
		x >>= _W
	}

	return z
}


func setN(z, x []Word) []Word {
	z = makeN(z, len(x), false)
	for i, d := range x {
		z[i] = d
	}
	return z
}


func addNN(z, x, y []Word) []Word {
	m := len(x)
	n := len(y)

	switch {
	case m < n:
		return addNN(z, y, x)
	case m == 0:
		// n == 0 because m >= n; result is 0
		return makeN(z, 0, false)
	case n == 0:
		// result is x
		return setN(z, x)
	}
	// m > 0

	z = makeN(z, m, false)
	c := addVV(&z[0], &x[0], &y[0], n)
	if m > n {
		c = addVW(&z[n], &x[n], c, m-n)
	}
	if c > 0 {
		z = z[0 : m+1]
		z[m] = c
	}

	return z
}


func subNN(z, x, y []Word) []Word {
	m := len(x)
	n := len(y)

	switch {
	case m < n:
		panic("underflow")
	case m == 0:
		// n == 0 because m >= n; result is 0
		return makeN(z, 0, false)
	case n == 0:
		// result is x
		return setN(z, x)
	}
	// m > 0

	z = makeN(z, m, false)
	c := subVV(&z[0], &x[0], &y[0], n)
	if m > n {
		c = subVW(&z[n], &x[n], c, m-n)
	}
	if c != 0 {
		panic("underflow")
	}
	z = normN(z)

	return z
}


func cmpNN(x, y []Word) (r int) {
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


func mulAddNWW(z, x []Word, y, r Word) []Word {
	m := len(x)
	if m == 0 || y == 0 {
		return newN(z, uint64(r)) // result is r
	}
	// m > 0

	z = makeN(z, m, false)
	c := mulAddVWW(&z[0], &x[0], y, r, m)
	if c > 0 {
		z = z[0 : m+1]
		z[m] = c
	}

	return z
}


func mulNN(z, x, y []Word) []Word {
	m := len(x)
	n := len(y)

	switch {
	case m < n:
		return mulNN(z, y, x)
	case m == 0 || n == 0:
		return makeN(z, 0, false)
	case n == 1:
		return mulAddNWW(z, x, y[0], 0)
	}
	// m >= n && m > 1 && n > 1

	z = makeN(z, m+n, true)
	if &z[0] == &x[0] || &z[0] == &y[0] {
		z = makeN(nil, m+n, true) // z is an alias for x or y - cannot reuse
	}
	for i := 0; i < n; i++ {
		if f := y[i]; f != 0 {
			z[m+i] = addMulVVW(&z[i], &x[0], f, m)
		}
	}
	z = normN(z)

	return z
}


// q = (x-r)/y, with 0 <= r < y
func divNW(z, x []Word, y Word) (q []Word, r Word) {
	m := len(x)
	switch {
	case y == 0:
		panic("division by zero")
	case y == 1:
		q = setN(z, x) // result is x
		return
	case m == 0:
		q = setN(z, nil) // result is 0
		return
	}
	// m > 0
	z = makeN(z, m, false)
	r = divWVW(&z[0], 0, &x[0], y, m)
	q = normN(z)
	return
}


func divNN(z, z2, u, v []Word) (q, r []Word) {
	if len(v) == 0 {
		panic("Divide by zero undefined")
	}

	if cmpNN(u, v) < 0 {
		q = makeN(z, 0, false)
		r = setN(z2, u)
		return
	}

	if len(v) == 1 {
		var rprime Word
		q, rprime = divNW(z, u, v[0])
		if rprime > 0 {
			r = makeN(z2, 1, false)
			r[0] = rprime
		} else {
			r = makeN(z2, 0, false)
		}
		return
	}

	q, r = divLargeNN(z, z2, u, v)
	return
}


// q = (uIn-r)/v, with 0 <= r < y
// See Knuth, Volume 2, section 4.3.1, Algorithm D.
// Preconditions:
//    len(v) >= 2
//    len(uIn) >= len(v)
func divLargeNN(z, z2, uIn, v []Word) (q, r []Word) {
	n := len(v)
	m := len(uIn) - len(v)

	u := makeN(z2, len(uIn)+1, false)
	qhatv := make([]Word, len(v)+1)
	q = makeN(z, m+1, false)

	// D1.
	shift := leadingZeroBits(v[n-1])
	shiftLeft(v, v, shift)
	shiftLeft(u, uIn, shift)
	u[len(uIn)] = uIn[len(uIn)-1] >> (_W - uint(shift))

	// D2.
	for j := m; j >= 0; j-- {
		// D3.
		var qhat Word
		if u[j+n] == v[n-1] {
			qhat = _B - 1
		} else {
			var rhat Word
			qhat, rhat = divWW_g(u[j+n], u[j+n-1], v[n-1])

			// x1 | x2 = q̂v_{n-2}
			x1, x2 := mulWW_g(qhat, v[n-2])
			// test if q̂v_{n-2} > br̂ + u_{j+n-2}
			for greaterThan(x1, x2, rhat, u[j+n-2]) {
				qhat--
				prevRhat := rhat
				rhat += v[n-1]
				// v[n-1] >= 0, so this tests for overflow.
				if rhat < prevRhat {
					break
				}
				x1, x2 = mulWW_g(qhat, v[n-2])
			}
		}

		// D4.
		qhatv[len(v)] = mulAddVWW(&qhatv[0], &v[0], qhat, 0, len(v))

		c := subVV(&u[j], &u[j], &qhatv[0], len(qhatv))
		if c != 0 {
			c := addVV(&u[j], &u[j], &v[0], len(v))
			u[j+len(v)] += c
			qhat--
		}

		q[j] = qhat
	}

	q = normN(q)
	shiftRight(u, u, shift)
	shiftRight(v, v, shift)
	r = normN(u)

	return q, r
}


// log2 computes the integer binary logarithm of x.
// The result is the integer n for which 2^n <= x < 2^(n+1).
// If x == 0, the result is -1.
func log2(x Word) int {
	n := 0
	for ; x > 0; x >>= 1 {
		n++
	}
	return n - 1
}


// log2N computes the integer binary logarithm of x.
// The result is the integer n for which 2^n <= x < 2^(n+1).
// If x == 0, the result is -1.
func log2N(x []Word) int {
	m := len(x)
	if m > 0 {
		return (m-1)*_W + log2(x[m-1])
	}
	return -1
}


func hexValue(ch byte) int {
	var d byte
	switch {
	case '0' <= ch && ch <= '9':
		d = ch - '0'
	case 'a' <= ch && ch <= 'f':
		d = ch - 'a' + 10
	case 'A' <= ch && ch <= 'F':
		d = ch - 'A' + 10
	default:
		return -1
	}
	return int(d)
}


// scanN returns the natural number corresponding to the
// longest possible prefix of s representing a natural number in a
// given conversion base, the actual conversion base used, and the
// prefix length. The syntax of natural numbers follows the syntax
// of unsigned integer literals in Go.
//
// If the base argument is 0, the string prefix determines the actual
// conversion base. A prefix of ``0x'' or ``0X'' selects base 16; the
// ``0'' prefix selects base 8. Otherwise the selected base is 10.
//
func scanN(z []Word, s string, base int) ([]Word, int, int) {
	// determine base if necessary
	i, n := 0, len(s)
	if base == 0 {
		base = 10
		if n > 0 && s[0] == '0' {
			if n > 1 && (s[1] == 'x' || s[1] == 'X') {
				if n == 2 {
					// Reject a string which is just '0x' as nonsense.
					return nil, 0, 0
				}
				base, i = 16, 2
			} else {
				base, i = 8, 1
			}
		}
	}
	if base < 2 || 16 < base {
		panic("illegal base")
	}

	// convert string
	z = makeN(z, len(z), false)
	for ; i < n; i++ {
		d := hexValue(s[i])
		if 0 <= d && d < base {
			z = mulAddNWW(z, z, Word(base), Word(d))
		} else {
			break
		}
	}

	return z, base, i
}


// string converts x to a string for a given base, with 2 <= base <= 16.
// TODO(gri) in the style of the other routines, perhaps this should take
//           a []byte buffer and return it
func stringN(x []Word, base int) string {
	if base < 2 || 16 < base {
		panic("illegal base")
	}

	if len(x) == 0 {
		return "0"
	}

	// allocate buffer for conversion
	i := (log2N(x)+1)/log2(Word(base)) + 1 // +1: round up
	s := make([]byte, i)

	// don't destroy x
	q := setN(nil, x)

	// convert
	for len(q) > 0 {
		i--
		var r Word
		q, r = divNW(q, q, Word(base))
		s[i] = "0123456789abcdef"[r]
	}

	return string(s[i:])
}


// leadingZeroBits returns the number of leading zero bits in x.
func leadingZeroBits(x Word) int {
	c := 0
	if x < 1<<(_W/2) {
		x <<= _W / 2
		c = _W / 2
	}

	for i := 0; x != 0; i++ {
		if x&(1<<(_W-1)) != 0 {
			return i + c
		}
		x <<= 1
	}

	return _W
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
	// to the power of k. Multipling by a power of two is equivalent to
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


func shiftLeft(dst, src []Word, n int) {
	if len(src) == 0 {
		return
	}

	ñ := _W - uint(n)
	for i := len(src) - 1; i >= 1; i-- {
		dst[i] = src[i] << uint(n)
		dst[i] |= src[i-1] >> ñ
	}
	dst[0] = src[0] << uint(n)
}


func shiftRight(dst, src []Word, n int) {
	if len(src) == 0 {
		return
	}

	ñ := _W - uint(n)
	for i := 0; i < len(src)-1; i++ {
		dst[i] = src[i] >> uint(n)
		dst[i] |= src[i+1] << ñ
	}
	dst[len(src)-1] = src[len(src)-1] >> uint(n)
}


// greaterThan returns true iff (x1<<_W + x2) > (y1<<_W + y2)
func greaterThan(x1, x2, y1, y2 Word) bool { return x1 > y1 || x1 == y1 && x2 > y2 }


// modNW returns x % d.
func modNW(x []Word, d Word) (r Word) {
	// TODO(agl): we don't actually need to store the q value.
	q := makeN(nil, len(x), false)
	return divWVW(&q[0], 0, &x[0], d, len(x))
}


// powersOfTwoDecompose finds q and k such that q * 1<<k = n and q is odd.
func powersOfTwoDecompose(n []Word) (q []Word, k Word) {
	if len(n) == 0 {
		return n, 0
	}

	zeroWords := 0
	for n[zeroWords] == 0 {
		zeroWords++
	}
	// One of the words must be non-zero by invariant, therefore
	// zeroWords < len(n).
	x := trailingZeroBits(n[zeroWords])

	q = makeN(nil, len(n)-zeroWords, false)
	shiftRight(q, n[zeroWords:], x)
	q = normN(q)

	k = Word(_W*zeroWords + x)
	return
}


// randomN creates a random integer in [0..limit), using the space in z if
// possible. n is the bit length of limit.
func randomN(z []Word, rand *rand.Rand, limit []Word, n int) []Word {
	bitLengthOfMSW := uint(n % _W)
	if bitLengthOfMSW == 0 {
		bitLengthOfMSW = _W
	}
	mask := Word((1 << bitLengthOfMSW) - 1)
	z = makeN(z, len(limit), false)

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

		if cmpNN(z, limit) < 0 {
			break
		}
	}

	return normN(z)
}


// If m != nil, expNNN calculates x**y mod m. Otherwise it calculates x**y. It
// reuses the storage of z if possible.
func expNNN(z, x, y, m []Word) []Word {
	if len(y) == 0 {
		z = makeN(z, 1, false)
		z[0] = 1
		return z
	}

	if m != nil {
		// We likely end up being as long as the modulus.
		z = makeN(z, len(m), false)
	}
	z = setN(z, x)
	v := y[len(y)-1]
	// It's invalid for the most significant word to be zero, therefore we
	// will find a one bit.
	shift := leadingZeros(v) + 1
	v <<= shift
	var q []Word

	const mask = 1 << (_W - 1)

	// We walk through the bits of the exponent one by one. Each time we
	// see a bit, we square, thus doubling the power. If the bit is a one,
	// we also multiply by x, thus adding one to the power.

	w := _W - int(shift)
	for j := 0; j < w; j++ {
		z = mulNN(z, z, z)

		if v&mask != 0 {
			z = mulNN(z, z, x)
		}

		if m != nil {
			q, z = divNN(q, z, z, m)
		}

		v <<= 1
	}

	for i := len(y) - 2; i >= 0; i-- {
		v = y[i]

		for j := 0; j < _W; j++ {
			z = mulNN(z, z, z)

			if v&mask != 0 {
				z = mulNN(z, z, x)
			}

			if m != nil {
				q, z = divNN(q, z, z, m)
			}

			v <<= 1
		}
	}

	return z
}


// lenN returns the bit length of z.
func lenN(z []Word) int {
	if len(z) == 0 {
		return 0
	}

	return (len(z)-1)*_W + (_W - leadingZeroBits(z[len(z)-1]))
}


const (
	primesProduct32 = 0xC0CFD797         // Π {p ∈ primes, 2 < p <= 29}
	primesProduct64 = 0xE221F97C30E94E1D // Π {p ∈ primes, 2 < p <= 53}
)

var bigOne = []Word{1}
var bigTwo = []Word{2}

// probablyPrime performs reps Miller-Rabin tests to check whether n is prime.
// If it returns true, n is prime with probability 1 - 1/4^reps.
// If it returns false, n is not prime.
func probablyPrime(n []Word, reps int) bool {
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
		if n[0] == 3 || n[0] == 5 || n[0] == 7 || n[0] == 11 ||
			n[0] == 13 || n[0] == 17 || n[0] == 19 || n[0] == 23 ||
			n[0] == 29 || n[0] == 31 || n[0] == 37 || n[0] == 41 ||
			n[0] == 43 || n[0] == 47 || n[0] == 53 {
			return true
		}
	}

	var r Word
	switch _W {
	case 32:
		r = modNW(n, primesProduct32)
	case 64:
		r = modNW(n, primesProduct64&_M)
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

	nm1 := subNN(nil, n, bigOne)
	// 1<<k * q = nm1;
	q, k := powersOfTwoDecompose(nm1)

	nm3 := subNN(nil, nm1, bigTwo)
	rand := rand.New(rand.NewSource(int64(n[0])))

	var x, y, quotient []Word
	nm3Len := lenN(nm3)

NextRandom:
	for i := 0; i < reps; i++ {
		x = randomN(x, rand, nm3, nm3Len)
		x = addNN(x, x, bigTwo)
		y = expNNN(y, x, q, n)
		if cmpNN(y, bigOne) == 0 || cmpNN(y, nm1) == 0 {
			continue
		}
		for j := Word(1); j < k; j++ {
			y = mulNN(y, y, y)
			quotient, y = divNN(quotient, y, y, n)
			if cmpNN(y, nm1) == 0 {
				continue NextRandom
			}
			if cmpNN(y, bigOne) == 0 {
				return false
			}
		}
		return false
	}

	return true
}
