// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package strconv implements conversions to and from string representations
// of basic data types.
package strconv

// decimal to binary floating point conversion.
// Algorithm:
//   1) Store input in multiprecision decimal.
//   2) Multiply/divide decimal by powers of two until in range [0.5, 1)
//   3) Multiply by 2^precision and round to get mantissa.

import "math"

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
	case equalIgnoreCase(s, "-inf"),
		equalIgnoreCase(s, "-infinity"):
		return math.Inf(-1), true
	case equalIgnoreCase(s, "+inf"),
		equalIgnoreCase(s, "+infinity"),
		equalIgnoreCase(s, "inf"),
		equalIgnoreCase(s, "infinity"):
		return math.Inf(1), true
	}
	return
}

// TODO(rsc): Better truncation handling.
func (b *decimal) set(s string) (ok bool) {
	i := 0
	b.neg = false

	// optional sign
	if i >= len(s) {
		return
	}
	switch {
	case s[i] == '+':
		i++
	case s[i] == '-':
		b.neg = true
		i++
	}

	// digits
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

	ok = true
	return
}

// decimal power of ten to binary power of two.
var powtab = []int{1, 3, 6, 9, 13, 16, 19, 23, 26}

func (d *decimal) floatBits(flt *floatInfo) (b uint64, overflow bool) {
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
	d.Shift(int(1 + flt.mantbits))
	mant = d.RoundedInteger()

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
	if d.neg {
		bits |= 1 << flt.mantbits << flt.expbits
	}
	return bits, overflow
}

// Compute exact floating-point integer from d's digits.
// Caller is responsible for avoiding overflow.
func (d *decimal) atof64int() float64 {
	f := 0.0
	for i := 0; i < d.nd; i++ {
		f = f*10 + float64(d.d[i]-'0')
	}
	if d.neg {
		f = -f
	}
	return f
}

func (d *decimal) atof32int() float32 {
	f := float32(0)
	for i := 0; i < d.nd; i++ {
		f = f*10 + float32(d.d[i]-'0')
	}
	if d.neg {
		f = -f
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
func (d *decimal) atof64() (f float64, ok bool) {
	// Exact integers are <= 10^15.
	// Exact powers of ten are <= 10^22.
	if d.nd > 15 {
		return
	}
	switch {
	case d.dp == d.nd: // int
		f := d.atof64int()
		return f, true

	case d.dp > d.nd && d.dp <= 15+22: // int * 10^k
		f := d.atof64int()
		k := d.dp - d.nd
		// If exponent is big but number of digits is not,
		// can move a few zeros into the integer part.
		if k > 22 {
			f *= float64pow10[k-22]
			k = 22
		}
		return f * float64pow10[k], true

	case d.dp < d.nd && d.nd-d.dp <= 22: // int / 10^k
		f := d.atof64int()
		return f / float64pow10[d.nd-d.dp], true
	}
	return
}

// If possible to convert decimal d to 32-bit float f exactly,
// entirely in floating-point math, do so, avoiding the machinery above.
func (d *decimal) atof32() (f float32, ok bool) {
	// Exact integers are <= 10^7.
	// Exact powers of ten are <= 10^10.
	if d.nd > 7 {
		return
	}
	switch {
	case d.dp == d.nd: // int
		f := d.atof32int()
		return f, true

	case d.dp > d.nd && d.dp <= 7+10: // int * 10^k
		f := d.atof32int()
		k := d.dp - d.nd
		// If exponent is big but number of digits is not,
		// can move a few zeros into the integer part.
		if k > 10 {
			f *= float32pow10[k-10]
			k = 10
		}
		return f * float32pow10[k], true

	case d.dp < d.nd && d.nd-d.dp <= 10: // int / 10^k
		f := d.atof32int()
		return f / float32pow10[d.nd-d.dp], true
	}
	return
}

const fnParseFloat = "ParseFloat"

func atof32(s string) (f float32, err error) {
	if val, ok := special(s); ok {
		return float32(val), nil
	}

	var d decimal
	if !d.set(s) {
		return 0, syntaxError(fnParseFloat, s)
	}
	if optimize {
		if f, ok := d.atof32(); ok {
			return f, nil
		}
	}
	b, ovf := d.floatBits(&float32info)
	f = math.Float32frombits(uint32(b))
	if ovf {
		err = rangeError(fnParseFloat, s)
	}
	return f, err
}

func atof64(s string) (f float64, err error) {
	if val, ok := special(s); ok {
		return val, nil
	}

	var d decimal
	if !d.set(s) {
		return 0, syntaxError(fnParseFloat, s)
	}
	if optimize {
		if f, ok := d.atof64(); ok {
			return f, nil
		}
	}
	b, ovf := d.floatBits(&float64info)
	f = math.Float64frombits(b)
	if ovf {
		err = rangeError(fnParseFloat, s)
	}
	return f, err
}

// ParseFloat converts the string s to a floating-point number
// with the precision specified by bitSize: 32 for float32, or 64 for float64.
// When bitSize=32, the result still has type float64, but it will be
// convertible to float32 without changing its value.
//
// If s is well-formed and near a valid floating point number,
// ParseFloat returns the nearest floating point number rounded
// using IEEE754 unbiased rounding.
//
// The errors that ParseFloat returns have concrete type *NumError
// and include err.Num = s.
//
// If s is not syntactically well-formed, ParseFloat returns err.Error = ErrSyntax.
//
// If s is syntactically well-formed but is more than 1/2 ULP
// away from the largest floating point number of the given size,
// ParseFloat returns f = ±Inf, err.Error = ErrRange.
func ParseFloat(s string, bitSize int) (f float64, err error) {
	if bitSize == 32 {
		f1, err1 := atof32(s)
		return float64(f1), err1
	}
	f1, err1 := atof64(s)
	return f1, err1
}
