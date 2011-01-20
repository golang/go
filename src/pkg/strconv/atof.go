// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// decimal to binary floating point conversion.
// Algorithm:
//   1) Store input in multiprecision decimal.
//   2) Multiply/divide decimal by powers of two until in range [0.5, 1)
//   3) Multiply by 2^precision and round to get mantissa.

// The strconv package implements conversions to and from
// string representations of basic data types.
package strconv

import (
	"math"
	"os"
)

var optimize = true // can change for testing

func equalIgnoreCase(s1, s2 string) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i := 0; i < len(s1); i++ {
		c1 := s1[i]
		if 'A' <= c1 && c1 <= 'Z' {
			c1 += 'a' - 'A'
		}
		c2 := s2[i]
		if 'A' <= c2 && c2 <= 'Z' {
			c2 += 'a' - 'A'
		}
		if c1 != c2 {
			return false
		}
	}
	return true
}

func special(s string) (f float64, ok bool) {
	switch {
	case equalIgnoreCase(s, "nan"):
		return math.NaN(), true
	case equalIgnoreCase(s, "-inf"):
		return math.Inf(-1), true
	case equalIgnoreCase(s, "+inf"):
		return math.Inf(1), true
	case equalIgnoreCase(s, "inf"):
		return math.Inf(1), true
	}
	return
}

// TODO(rsc): Better truncation handling.
func stringToDecimal(s string) (neg bool, d *decimal, trunc bool, ok bool) {
	i := 0

	// optional sign
	if i >= len(s) {
		return
	}
	switch {
	case s[i] == '+':
		i++
	case s[i] == '-':
		neg = true
		i++
	}

	// digits
	b := new(decimal)
	sawdot := false
	sawdigits := false
	for ; i < len(s); i++ {
		switch {
		case s[i] == '.':
			if sawdot {
				return
			}
			sawdot = true
			b.dp = b.nd
			continue

		case '0' <= s[i] && s[i] <= '9':
			sawdigits = true
			if s[i] == '0' && b.nd == 0 { // ignore leading zeros
				b.dp--
				continue
			}
			b.d[b.nd] = s[i]
			b.nd++
			continue
		}
		break
	}
	if !sawdigits {
		return
	}
	if !sawdot {
		b.dp = b.nd
	}

	// optional exponent moves decimal point.
	// if we read a very large, very long number,
	// just be sure to move the decimal point by
	// a lot (say, 100000).  it doesn't matter if it's
	// not the exact number.
	if i < len(s) && (s[i] == 'e' || s[i] == 'E') {
		i++
		if i >= len(s) {
			return
		}
		esign := 1
		if s[i] == '+' {
			i++
		} else if s[i] == '-' {
			i++
			esign = -1
		}
		if i >= len(s) || s[i] < '0' || s[i] > '9' {
			return
		}
		e := 0
		for ; i < len(s) && '0' <= s[i] && s[i] <= '9'; i++ {
			if e < 10000 {
				e = e*10 + int(s[i]) - '0'
			}
		}
		b.dp += e * esign
	}

	if i != len(s) {
		return
	}

	d = b
	ok = true
	return
}

// decimal power of ten to binary power of two.
var powtab = []int{1, 3, 6, 9, 13, 16, 19, 23, 26}

func decimalToFloatBits(neg bool, d *decimal, trunc bool, flt *floatInfo) (b uint64, overflow bool) {
	var exp int
	var mant uint64

	// Zero is always a special case.
	if d.nd == 0 {
		mant = 0
		exp = flt.bias
		goto out
	}

	// Obvious overflow/underflow.
	// These bounds are for 64-bit floats.
	// Will have to change if we want to support 80-bit floats in the future.
	if d.dp > 310 {
		goto overflow
	}
	if d.dp < -330 {
		// zero
		mant = 0
		exp = flt.bias
		goto out
	}

	// Scale by powers of two until in range [0.5, 1.0)
	exp = 0
	for d.dp > 0 {
		var n int
		if d.dp >= len(powtab) {
			n = 27
		} else {
			n = powtab[d.dp]
		}
		d.Shift(-n)
		exp += n
	}
	for d.dp < 0 || d.dp == 0 && d.d[0] < '5' {
		var n int
		if -d.dp >= len(powtab) {
			n = 27
		} else {
			n = powtab[-d.dp]
		}
		d.Shift(n)
		exp -= n
	}

	// Our range is [0.5,1) but floating point range is [1,2).
	exp--

	// Minimum representable exponent is flt.bias+1.
	// If the exponent is smaller, move it up and
	// adjust d accordingly.
	if exp < flt.bias+1 {
		n := flt.bias + 1 - exp
		d.Shift(-n)
		exp += n
	}

	if exp-flt.bias >= 1<<flt.expbits-1 {
		goto overflow
	}

	// Extract 1+flt.mantbits bits.
	mant = d.Shift(int(1 + flt.mantbits)).RoundedInteger()

	// Rounding might have added a bit; shift down.
	if mant == 2<<flt.mantbits {
		mant >>= 1
		exp++
		if exp-flt.bias >= 1<<flt.expbits-1 {
			goto overflow
		}
	}

	// Denormalized?
	if mant&(1<<flt.mantbits) == 0 {
		exp = flt.bias
	}
	goto out

overflow:
	// ±Inf
	mant = 0
	exp = 1<<flt.expbits - 1 + flt.bias
	overflow = true

out:
	// Assemble bits.
	bits := mant & (uint64(1)<<flt.mantbits - 1)
	bits |= uint64((exp-flt.bias)&(1<<flt.expbits-1)) << flt.mantbits
	if neg {
		bits |= 1 << flt.mantbits << flt.expbits
	}
	return bits, overflow
}

// Compute exact floating-point integer from d's digits.
// Caller is responsible for avoiding overflow.
func decimalAtof64Int(neg bool, d *decimal) float64 {
	f := 0.0
	for i := 0; i < d.nd; i++ {
		f = f*10 + float64(d.d[i]-'0')
	}
	if neg {
		f *= -1 // BUG work around 6g f = -f.
	}
	return f
}

func decimalAtof32Int(neg bool, d *decimal) float32 {
	f := float32(0)
	for i := 0; i < d.nd; i++ {
		f = f*10 + float32(d.d[i]-'0')
	}
	if neg {
		f *= -1 // BUG work around 6g f = -f.
	}
	return f
}

// Exact powers of 10.
var float64pow10 = []float64{
	1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
	1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
	1e20, 1e21, 1e22,
}
var float32pow10 = []float32{1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10}

// If possible to convert decimal d to 64-bit float f exactly,
// entirely in floating-point math, do so, avoiding the expense of decimalToFloatBits.
// Three common cases:
//	value is exact integer
//	value is exact integer * exact power of ten
//	value is exact integer / exact power of ten
// These all produce potentially inexact but correctly rounded answers.
func decimalAtof64(neg bool, d *decimal, trunc bool) (f float64, ok bool) {
	// Exact integers are <= 10^15.
	// Exact powers of ten are <= 10^22.
	if d.nd > 15 {
		return
	}
	switch {
	case d.dp == d.nd: // int
		f := decimalAtof64Int(neg, d)
		return f, true

	case d.dp > d.nd && d.dp <= 15+22: // int * 10^k
		f := decimalAtof64Int(neg, d)
		k := d.dp - d.nd
		// If exponent is big but number of digits is not,
		// can move a few zeros into the integer part.
		if k > 22 {
			f *= float64pow10[k-22]
			k = 22
		}
		return f * float64pow10[k], true

	case d.dp < d.nd && d.nd-d.dp <= 22: // int / 10^k
		f := decimalAtof64Int(neg, d)
		return f / float64pow10[d.nd-d.dp], true
	}
	return
}

// If possible to convert decimal d to 32-bit float f exactly,
// entirely in floating-point math, do so, avoiding the machinery above.
func decimalAtof32(neg bool, d *decimal, trunc bool) (f float32, ok bool) {
	// Exact integers are <= 10^7.
	// Exact powers of ten are <= 10^10.
	if d.nd > 7 {
		return
	}
	switch {
	case d.dp == d.nd: // int
		f := decimalAtof32Int(neg, d)
		return f, true

	case d.dp > d.nd && d.dp <= 7+10: // int * 10^k
		f := decimalAtof32Int(neg, d)
		k := d.dp - d.nd
		// If exponent is big but number of digits is not,
		// can move a few zeros into the integer part.
		if k > 10 {
			f *= float32pow10[k-10]
			k = 10
		}
		return f * float32pow10[k], true

	case d.dp < d.nd && d.nd-d.dp <= 10: // int / 10^k
		f := decimalAtof32Int(neg, d)
		return f / float32pow10[d.nd-d.dp], true
	}
	return
}

// Atof32 converts the string s to a 32-bit floating-point number.
//
// If s is well-formed and near a valid floating point number,
// Atof32 returns the nearest floating point number rounded
// using IEEE754 unbiased rounding.
//
// The errors that Atof32 returns have concrete type *NumError
// and include err.Num = s.
//
// If s is not syntactically well-formed, Atof32 returns err.Error = os.EINVAL.
//
// If s is syntactically well-formed but is more than 1/2 ULP
// away from the largest floating point number of the given size,
// Atof32 returns f = ±Inf, err.Error = os.ERANGE.
func Atof32(s string) (f float32, err os.Error) {
	if val, ok := special(s); ok {
		return float32(val), nil
	}

	neg, d, trunc, ok := stringToDecimal(s)
	if !ok {
		return 0, &NumError{s, os.EINVAL}
	}
	if optimize {
		if f, ok := decimalAtof32(neg, d, trunc); ok {
			return f, nil
		}
	}
	b, ovf := decimalToFloatBits(neg, d, trunc, &float32info)
	f = math.Float32frombits(uint32(b))
	if ovf {
		err = &NumError{s, os.ERANGE}
	}
	return f, err
}

// Atof64 converts the string s to a 64-bit floating-point number.
// Except for the type of its result, its definition is the same as that
// of Atof32.
func Atof64(s string) (f float64, err os.Error) {
	if val, ok := special(s); ok {
		return val, nil
	}

	neg, d, trunc, ok := stringToDecimal(s)
	if !ok {
		return 0, &NumError{s, os.EINVAL}
	}
	if optimize {
		if f, ok := decimalAtof64(neg, d, trunc); ok {
			return f, nil
		}
	}
	b, ovf := decimalToFloatBits(neg, d, trunc, &float64info)
	f = math.Float64frombits(b)
	if ovf {
		err = &NumError{s, os.ERANGE}
	}
	return f, err
}

// AtofN converts the string s to a 64-bit floating-point number,
// but it rounds the result assuming that it will be stored in a value
// of n bits (32 or 64).
func AtofN(s string, n int) (f float64, err os.Error) {
	if n == 32 {
		f1, err1 := Atof32(s)
		return float64(f1), err1
	}
	f1, err1 := Atof64(s)
	return f1, err1
}
