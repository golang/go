// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements multi-precision decimal numbers.
// The implementation is for float to decimal conversion only;
// not general purpose use.
// The only operations are precise conversion from binary to
// decimal and rounding.
//
// The key observation and some code (shr) is borrowed from
// strconv/decimal.go: conversion of binary fractional values can be done
// precisely in multi-precision decimal because 2 divides 10 (required for
// >> of mantissa); but conversion of decimal floating-point values cannot
// be done precisely in binary representation.
//
// In contrast to strconv/decimal.go, only right shift is implemented in
// decimal format - left shift can be done precisely in binary format.

package big

// A decimal represents an unsigned floating-point number in decimal representation.
// The value of a non-zero decimal d is d.mant * 10**d.exp with 0.5 <= d.mant < 1,
// with the most-significant mantissa digit at index 0. For the zero decimal, the
// mantissa length and exponent are 0.
// The zero value for decimal represents a ready-to-use 0.0.
type decimal struct {
	mant []byte // mantissa ASCII digits, big-endian
	exp  int    // exponent
}

// at returns the i'th mantissa digit, starting with the most significant digit at 0.
func (d *decimal) at(i int) byte {
	if 0 <= i && i < len(d.mant) {
		return d.mant[i]
	}
	return '0'
}

// Maximum shift amount that can be done in one pass without overflow.
// A Word has _W bits and (1<<maxShift - 1)*10 + 9 must fit into Word.
const maxShift = _W - 4

// TODO(gri) Since we know the desired decimal precision when converting
// a floating-point number, we may be able to limit the number of decimal
// digits that need to be computed by init by providing an additional
// precision argument and keeping track of when a number was truncated early
// (equivalent of "sticky bit" in binary rounding).

// TODO(gri) Along the same lines, enforce some limit to shift magnitudes
// to avoid "infinitely" long running conversions (until we run out of space).

// Init initializes x to the decimal representation of m << shift (for
// shift >= 0), or m >> -shift (for shift < 0).
func (x *decimal) init(m nat, shift int) {
	// special case 0
	if len(m) == 0 {
		x.mant = x.mant[:0]
		x.exp = 0
		return
	}

	// Optimization: If we need to shift right, first remove any trailing
	// zero bits from m to reduce shift amount that needs to be done in
	// decimal format (since that is likely slower).
	if shift < 0 {
		ntz := m.trailingZeroBits()
		s := uint(-shift)
		if s >= ntz {
			s = ntz // shift at most ntz bits
		}
		m = nat(nil).shr(m, s)
		shift += int(s)
	}

	// Do any shift left in binary representation.
	if shift > 0 {
		m = nat(nil).shl(m, uint(shift))
		shift = 0
	}

	// Convert mantissa into decimal representation.
	s := m.utoa(10)
	n := len(s)
	x.exp = n
	// Trim trailing zeros; instead the exponent is tracking
	// the decimal point independent of the number of digits.
	for n > 0 && s[n-1] == '0' {
		n--
	}
	x.mant = append(x.mant[:0], s[:n]...)

	// Do any (remaining) shift right in decimal representation.
	if shift < 0 {
		for shift < -maxShift {
			shr(x, maxShift)
			shift += maxShift
		}
		shr(x, uint(-shift))
	}
}

// shr implements x >> s, for s <= maxShift.
func shr(x *decimal, s uint) {
	// Division by 1<<s using shift-and-subtract algorithm.

	// pick up enough leading digits to cover first shift
	r := 0 // read index
	var n Word
	for n>>s == 0 && r < len(x.mant) {
		ch := Word(x.mant[r])
		r++
		n = n*10 + ch - '0'
	}
	if n == 0 {
		// x == 0; shouldn't get here, but handle anyway
		x.mant = x.mant[:0]
		return
	}
	for n>>s == 0 {
		r++
		n *= 10
	}
	x.exp += 1 - r

	// read a digit, write a digit
	w := 0 // write index
	mask := Word(1)<<s - 1
	for r < len(x.mant) {
		ch := Word(x.mant[r])
		r++
		d := n >> s
		n &= mask // n -= d << s
		x.mant[w] = byte(d + '0')
		w++
		n = n*10 + ch - '0'
	}

	// write extra digits that still fit
	for n > 0 && w < len(x.mant) {
		d := n >> s
		n &= mask
		x.mant[w] = byte(d + '0')
		w++
		n = n * 10
	}
	x.mant = x.mant[:w] // the number may be shorter (e.g. 1024 >> 10)

	// append additional digits that didn't fit
	for n > 0 {
		d := n >> s
		n &= mask
		x.mant = append(x.mant, byte(d+'0'))
		n = n * 10
	}

	trim(x)
}

func (x *decimal) String() string {
	if len(x.mant) == 0 {
		return "0"
	}

	var buf []byte
	switch {
	case x.exp <= 0:
		// 0.00ddd
		buf = append(buf, "0."...)
		buf = appendZeros(buf, -x.exp)
		buf = append(buf, x.mant...)

	case /* 0 < */ x.exp < len(x.mant):
		// dd.ddd
		buf = append(buf, x.mant[:x.exp]...)
		buf = append(buf, '.')
		buf = append(buf, x.mant[x.exp:]...)

	default: // len(x.mant) <= x.exp
		// ddd00
		buf = append(buf, x.mant...)
		buf = appendZeros(buf, x.exp-len(x.mant))
	}

	return string(buf)
}

// appendZeros appends n 0 digits to buf and returns buf.
func appendZeros(buf []byte, n int) []byte {
	for ; n > 0; n-- {
		buf = append(buf, '0')
	}
	return buf
}

// shouldRoundUp reports if x should be rounded up
// if shortened to n digits. n must be a valid index
// for x.mant.
func shouldRoundUp(x *decimal, n int) bool {
	if x.mant[n] == '5' && n+1 == len(x.mant) {
		// exactly halfway - round to even
		return n > 0 && (x.mant[n-1]-'0')&1 != 0
	}
	// not halfway - digit tells all (x.mant has no trailing zeros)
	return x.mant[n] >= '5'
}

// round sets x to (at most) n mantissa digits by rounding it
// to the nearest even value with n (or fever) mantissa digits.
// If n < 0, x remains unchanged.
func (x *decimal) round(n int) {
	if n < 0 || n >= len(x.mant) {
		return // nothing to do
	}

	if shouldRoundUp(x, n) {
		x.roundUp(n)
	} else {
		x.roundDown(n)
	}
}

func (x *decimal) roundUp(n int) {
	if n < 0 || n >= len(x.mant) {
		return // nothing to do
	}
	// 0 <= n < len(x.mant)

	// find first digit < '9'
	for n > 0 && x.mant[n-1] >= '9' {
		n--
	}

	if n == 0 {
		// all digits are '9's => round up to '1' and update exponent
		x.mant[0] = '1' // ok since len(x.mant) > n
		x.mant = x.mant[:1]
		x.exp++
		return
	}

	// n > 0 && x.mant[n-1] < '9'
	x.mant[n-1]++
	x.mant = x.mant[:n]
	// x already trimmed
}

func (x *decimal) roundDown(n int) {
	if n < 0 || n >= len(x.mant) {
		return // nothing to do
	}
	x.mant = x.mant[:n]
	trim(x)
}

// trim cuts off any trailing zeros from x's mantissa;
// they are meaningless for the value of x.
func trim(x *decimal) {
	i := len(x.mant)
	for i > 0 && x.mant[i-1] == '0' {
		i--
	}
	x.mant = x.mant[:i]
	if i == 0 {
		x.exp = 0
	}
}
