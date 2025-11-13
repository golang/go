// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import "math/bits"

// binary to decimal conversion using the Ryū algorithm.
//
// See Ulf Adams, "Ryū: Fast Float-to-String Conversion" (doi:10.1145/3192366.3192369)

// ryuFtoaShortest formats mant*2^exp with prec decimal digits.
func ryuFtoaShortest(d *decimalSlice, mant uint64, exp int, flt *floatInfo) {
	if mant == 0 {
		d.nd, d.dp = 0, 0
		return
	}
	// If input is an exact integer with fewer bits than the mantissa,
	// the previous and next integer are not admissible representations.
	if exp <= 0 && bits.TrailingZeros64(mant) >= -exp {
		mant >>= uint(-exp)
		ryuDigits(d, mant, mant, mant, true, false)
		return
	}
	ml, mc, mu, e2 := computeBounds(mant, exp, flt)
	if e2 == 0 {
		ryuDigits(d, ml, mc, mu, true, false)
		return
	}
	// Find 10^q *larger* than 2^-e2
	q := mulLog10_2(-e2) + 1

	// We are going to multiply by 10^q using 128-bit arithmetic.
	// The exponent is the same for all 3 numbers.
	var dl, dc, du uint64
	var dl0, dc0, du0 bool
	if flt == &float32info {
		var dl32, dc32, du32 uint32
		dl32, _, dl0 = mult64bitPow10(uint32(ml), e2, q)
		dc32, _, dc0 = mult64bitPow10(uint32(mc), e2, q)
		du32, e2, du0 = mult64bitPow10(uint32(mu), e2, q)
		dl, dc, du = uint64(dl32), uint64(dc32), uint64(du32)
	} else {
		dl, _, dl0 = mult128bitPow10(ml, e2, q)
		dc, _, dc0 = mult128bitPow10(mc, e2, q)
		du, e2, du0 = mult128bitPow10(mu, e2, q)
	}
	if e2 >= 0 {
		panic("not enough significant bits after mult128bitPow10")
	}
	// Is it an exact computation?
	if q > 55 {
		// Large positive powers of ten are not exact
		dl0, dc0, du0 = false, false, false
	}
	if q < 0 && q >= -24 {
		// Division by a power of ten may be exact.
		// (note that 5^25 is a 59-bit number so division by 5^25 is never exact).
		if divisiblePow5(ml, -q) {
			dl0 = true
		}
		if divisiblePow5(mc, -q) {
			dc0 = true
		}
		if divisiblePow5(mu, -q) {
			du0 = true
		}
	}
	// Express the results (dl, dc, du)*2^e2 as integers.
	// Extra bits must be removed and rounding hints computed.
	extra := uint(-e2)
	extraMask := uint64(1<<extra - 1)
	// Now compute the floored, integral base 10 mantissas.
	dl, fracl := dl>>extra, dl&extraMask
	dc, fracc := dc>>extra, dc&extraMask
	du, fracu := du>>extra, du&extraMask
	// Is it allowed to use 'du' as a result?
	// It is always allowed when it is truncated, but also
	// if it is exact and the original binary mantissa is even
	// When disallowed, we can subtract 1.
	uok := !du0 || fracu > 0
	if du0 && fracu == 0 {
		uok = mant&1 == 0
	}
	if !uok {
		du--
	}
	// Is 'dc' the correctly rounded base 10 mantissa?
	// The correct rounding might be dc+1
	cup := false // don't round up.
	if dc0 {
		// If we computed an exact product, the half integer
		// should round to next (even) integer if 'dc' is odd.
		cup = fracc > 1<<(extra-1) ||
			(fracc == 1<<(extra-1) && dc&1 == 1)
	} else {
		// otherwise, the result is a lower truncation of the ideal
		// result.
		cup = fracc>>(extra-1) == 1
	}
	// Is 'dl' an allowed representation?
	// Only if it is an exact value, and if the original binary mantissa
	// was even.
	lok := dl0 && fracl == 0 && (mant&1 == 0)
	if !lok {
		dl++
	}
	// We need to remember whether the trimmed digits of 'dc' are zero.
	c0 := dc0 && fracc == 0
	// render digits
	ryuDigits(d, dl, dc, du, c0, cup)
	d.dp -= q
}

// computeBounds returns a floating-point vector (l, c, u)×2^e2
// where the mantissas are 55-bit (or 26-bit) integers, describing the interval
// represented by the input float64 or float32.
func computeBounds(mant uint64, exp int, flt *floatInfo) (lower, central, upper uint64, e2 int) {
	if mant != 1<<flt.mantbits || exp == flt.bias+1-int(flt.mantbits) {
		// regular case (or denormals)
		lower, central, upper = 2*mant-1, 2*mant, 2*mant+1
		e2 = exp - 1
		return
	} else {
		// border of an exponent
		lower, central, upper = 4*mant-1, 4*mant, 4*mant+2
		e2 = exp - 2
		return
	}
}

func ryuDigits(d *decimalSlice, lower, central, upper uint64,
	c0, cup bool) {
	lhi, llo := uint32(lower/1e9), uint32(lower%1e9)
	chi, clo := uint32(central/1e9), uint32(central%1e9)
	uhi, ulo := uint32(upper/1e9), uint32(upper%1e9)
	if uhi == 0 {
		// only low digits (for denormals)
		ryuDigits32(d, llo, clo, ulo, c0, cup, 8)
	} else if lhi < uhi {
		// truncate 9 digits at once.
		if llo != 0 {
			lhi++
		}
		c0 = c0 && clo == 0
		cup = (clo > 5e8) || (clo == 5e8 && cup)
		ryuDigits32(d, lhi, chi, uhi, c0, cup, 8)
		d.dp += 9
	} else {
		d.nd = 0
		// emit high part
		n := uint(9)
		for v := chi; v > 0; {
			v1, v2 := v/10, v%10
			v = v1
			n--
			d.d[n] = byte(v2 + '0')
		}
		d.d = d.d[n:]
		d.nd = int(9 - n)
		// emit low part
		ryuDigits32(d, llo, clo, ulo,
			c0, cup, d.nd+8)
	}
	// trim trailing zeros
	for d.nd > 0 && d.d[d.nd-1] == '0' {
		d.nd--
	}
	// trim initial zeros
	for d.nd > 0 && d.d[0] == '0' {
		d.nd--
		d.dp--
		d.d = d.d[1:]
	}
}

// ryuDigits32 emits decimal digits for a number less than 1e9.
func ryuDigits32(d *decimalSlice, lower, central, upper uint32,
	c0, cup bool, endindex int) {
	if upper == 0 {
		d.dp = endindex + 1
		return
	}
	trimmed := 0
	// Remember last trimmed digit to check for round-up.
	// c0 will be used to remember zeroness of following digits.
	cNextDigit := 0
	for upper > 0 {
		// Repeatedly compute:
		// l = Ceil(lower / 10^k)
		// c = Round(central / 10^k)
		// u = Floor(upper / 10^k)
		// and stop when c goes out of the (l, u) interval.
		l := (lower + 9) / 10
		c, cdigit := central/10, central%10
		u := upper / 10
		if l > u {
			// don't trim the last digit as it is forbidden to go below l
			// other, trim and exit now.
			break
		}
		// Check that we didn't cross the lower boundary.
		// The case where l < u but c == l-1 is essentially impossible,
		// but may happen if:
		//    lower   = ..11
		//    central = ..19
		//    upper   = ..31
		// and means that 'central' is very close but less than
		// an integer ending with many zeros, and usually
		// the "round-up" logic hides the problem.
		if l == c+1 && c < u {
			c++
			cdigit = 0
			cup = false
		}
		trimmed++
		// Remember trimmed digits of c
		c0 = c0 && cNextDigit == 0
		cNextDigit = int(cdigit)
		lower, central, upper = l, c, u
	}
	// should we round up?
	if trimmed > 0 {
		cup = cNextDigit > 5 ||
			(cNextDigit == 5 && !c0) ||
			(cNextDigit == 5 && c0 && central&1 == 1)
	}
	if central < upper && cup {
		central++
	}
	// We know where the number ends, fill directly
	endindex -= trimmed
	v := central
	n := endindex
	for n > d.nd {
		v1, v2 := v/100, v%100
		d.d[n] = smalls[2*v2+1]
		d.d[n-1] = smalls[2*v2+0]
		n -= 2
		v = v1
	}
	if n == d.nd {
		d.d[n] = byte(v + '0')
	}
	d.nd = endindex + 1
	d.dp = d.nd + trimmed
}

// mult64bitPow10 takes a floating-point input with a 25-bit
// mantissa and multiplies it with 10^q. The resulting mantissa
// is m*P >> 57 where P is a 64-bit truncated power of 10.
// It is typically 31 or 32-bit wide.
// The returned boolean is true if all trimmed bits were zero.
//
// That is:
//
//	m*2^e2 * round(10^q) = resM * 2^resE + ε
//	exact = ε == 0
func mult64bitPow10(m uint32, e2, q int) (resM uint32, resE int, exact bool) {
	if q == 0 {
		// P == 1<<63
		return m << 6, e2 - 6, true
	}
	pow, exp2, ok := pow10(q)
	if !ok {
		// This never happens due to the range of float32/float64 exponent
		panic("mult64bitPow10: power of 10 is out of range")
	}
	if q < 0 {
		// Inverse powers of ten must be rounded up.
		pow.Hi++
	}
	hi, lo := bits.Mul64(uint64(m), pow.Hi)
	e2 += exp2 - 64 + 57
	return uint32(hi<<7 | lo>>57), e2, lo<<7 == 0
}

// mult128bitPow10 takes a floating-point input with a 55-bit
// mantissa and multiplies it with 10^q. The resulting mantissa
// is m*P >> 119 where P is a 128-bit truncated power of 10.
// It is typically 63 or 64-bit wide.
// The returned boolean is true is all trimmed bits were zero.
//
// That is:
//
//	m*2^e2 * round(10^q) = resM * 2^resE + ε
//	exact = ε == 0
func mult128bitPow10(m uint64, e2, q int) (resM uint64, resE int, exact bool) {
	if q == 0 {
		// P == 1<<127
		return m << 8, e2 - 8, true
	}
	pow, exp2, ok := pow10(q)
	if !ok {
		// This never happens due to the range of float32/float64 exponent
		panic("mult128bitPow10: power of 10 is out of range")
	}
	if q < 0 {
		// Inverse powers of ten must be rounded up.
		pow.Lo++
	}
	e2 += exp2 - 128 + 119

	hi, mid, lo := umul192(m, pow)
	return hi<<9 | mid>>55, e2, mid<<9 == 0 && lo == 0
}
