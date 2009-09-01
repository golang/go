// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains operations on unsigned multi-precision integers.
// These are the building blocks for the operations on signed integers
// and rationals.

//	NOTE: PACKAGE UNDER CONSTRUCTION (use bignum for the time being)
//
// This package implements multi-precision arithmetic (big numbers).
// The following numeric types are supported:
//
//	- Int	signed integers
//
// All methods on Int take the result as the receiver; if it is one
// of the operands it may be overwritten (and its memory reused).
// To enable chaining of operations, the result is also returned.
//
package big

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
	i := len(z);
	for i > 0 && z[i-1] == 0 {
		i--;
	}
	z = z[0 : i];
	return z;
}


func makeN(z []Word, m int, clear bool) []Word {
	if len(z) > m {
		z = z[0 : m];  // reuse z - has at least one extra word for a carry, if any
		if clear {
			for i := range z {
				z[i] = 0;
			}
		}
		return z;
	}

	c := 4;  // minimum capacity
	if m > c {
		c = m;
	}
	return make([]Word, m, c+1);  // +1: extra word for a carry, if any
}


func newN(z []Word, x uint64) []Word {
	if x == 0 {
		return makeN(z, 0, false);
	}

	// single-digit values
	if x == uint64(Word(x)) {
		z = makeN(z, 1, false);
		z[0] = Word(x);
		return z;
	}

	// compute number of words n required to represent x
	n := 0;
	for t := x; t > 0; t >>= _W {
		n++;
	}

	// split x into n words
	z = makeN(z, n, false);
	for i := 0; i < n; i++ {
		z[i] = Word(x & _M);
		x >>= _W;
	}

	return z;
}


func setN(z, x []Word) []Word {
	z = makeN(z, len(x), false);
	for i, d := range x {
		z[i] = d;
	}
	return z;
}


func addNN(z, x, y []Word) []Word {
	m := len(x);
	n := len(y);

	switch {
	case m < n:
		return addNN(z, y, x);
	case m == 0:
		// n == 0 because m >= n; result is 0
		return makeN(z, 0, false);
	case n == 0:
		// result is x
		return setN(z, x);
	}
	// m > 0

	z = makeN(z, m, false);
	c := addVV(&z[0], &x[0], &y[0], n);
	if m > n {
		c = addVW(&z[n], &x[n], c, m-n);
	}
	if c > 0 {
		z = z[0 : m+1];
		z[m] = c;
	}

	return z;
}


func subNN(z, x, y []Word) []Word {
	m := len(x);
	n := len(y);

	switch {
	case m < n:
		panic("underflow");
	case m == 0:
		// n == 0 because m >= n; result is 0
		return makeN(z, 0, false);
	case n == 0:
		// result is x
		return setN(z, x);
	}
	// m > 0

	z = makeN(z, m, false);
	c := subVV(&z[0], &x[0], &y[0], n);
	if m > n {
		c = subVW(&z[n], &x[n], c, m-n);
	}
	if c != 0 {
		panic("underflow");
	}
	z = normN(z);

	return z;
}


func cmpNN(x, y []Word) (r int) {
	m := len(x);
	n := len(y);
	if m != n || m == 0 {
		switch {
		case m < n: r = -1;
		case m > n: r = 1;
		}
		return;
	}

	i := m-1;
	for i > 0 && x[i] == y[i] {
		i--;
	}

	switch {
	case x[i] < y[i]: r = -1;
	case x[i] > y[i]: r = 1;
	}
	return;
}


func mulAddNWW(z, x []Word, y, r Word) []Word {
	m := len(x);
	if m == 0 || y == 0 {
		return newN(z, uint64(r));	// result is r
	}
	// m > 0

	z = makeN(z, m, false);
	c := mulAddVWW(&z[0], &x[0], y, r, m);
	if c > 0 {
		z = z[0 : m+1];
		z[m] = c;
	}

	return z;
}


func mulNN(z, x, y []Word) []Word {
	m := len(x);
	n := len(y);

	switch {
	case m < n:
		return mulNN(z, y, x);
	case m == 0 || n == 0:
		return makeN(z, 0, false);
	case n == 1:
		return mulAddNWW(z, x, y[0], 0);
	}
	// m >= n && m > 1 && n > 1

	z = makeN(z, m+n, true);
	if &z[0] == &x[0] || &z[0] == &y[0] {
		z = makeN(nil, m+n, true);  // z is an alias for x or y - cannot reuse
	}
	for i := 0; i < n; i++ {
		if f := y[i]; f != 0 {
			z[m+i] = addMulVVW(&z[i], &x[0], f, m);
		}
	}
	z = normN(z);

	return z
}


// q = (x-r)/y, with 0 <= r < y
func divNW(z, x []Word, y Word) (q []Word, r Word) {
	m := len(x);
	switch {
	case y == 0:
		panic("division by zero");
	case y == 1:
		q = setN(z, x);  // result is x
		return;
	case m == 0:
		q = setN(z, nil);  // result is 0
		return;
	}
	// m > 0
	z = makeN(z, m, false);
	r = divWVW(&z[0], 0, &x[0], y, m);
	q = normN(z);
	return;
}


// log2 computes the integer binary logarithm of x.
// The result is the integer n for which 2^n <= x < 2^(n+1).
// If x == 0, the result is -1.
func log2(x Word) int {
	n := 0;
	for ; x > 0; x >>= 1 {
		n++;
	}
	return n-1;
}


// log2N computes the integer binary logarithm of x.
// The result is the integer n for which 2^n <= x < 2^(n+1).
// If x == 0, the result is -1.
func log2N(x []Word) int {
	m := len(x);
	if m > 0 {
		return (m-1)*int(_W) + log2(x[m-1]);
	}
	return -1;
}


func hexValue(ch byte) int {
	var d byte;
	switch {
	case '0' <= ch && ch <= '9': d = ch - '0';
	case 'a' <= ch && ch <= 'f': d = ch - 'a' + 10;
	case 'A' <= ch && ch <= 'F': d = ch - 'A' + 10;
	default: return -1;
	}
	return int(d);
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
	if base < 2 || 16 < base {
		panic("illegal base");
	}

	// convert string
	z = makeN(z, len(z), false);
	for ; i < n; i++ {
		d := hexValue(s[i]);
		if 0 <= d && d < base {
			z = mulAddNWW(z, z, Word(base), Word(d));
		} else {
			break;
		}
	}

	return z, base, i;
}


// string converts x to a string for a given base, with 2 <= base <= 16.
// TODO(gri) in the style of the other routines, perhaps this should take
//           a []byte buffer and return it
func stringN(x []Word, base int) string {
	if base < 2 || 16 < base {
		panic("illegal base");
	}

	if len(x) == 0 {
		return "0";
	}

	// allocate buffer for conversion
	i := (log2N(x) + 1) / log2(Word(base)) + 1;  // +1: round up
	s := make([]byte, i);

	// don't destroy x
	q := setN(nil, x);

	// convert
	for len(q) > 0 {
		i--;
		var r Word;
		q, r = divNW(q, q, 10);
		s[i] = "0123456789abcdef"[r];
	};

	return string(s[i : len(s)]);
}
