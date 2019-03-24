// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import (
	"math/bits"
)

// binary to decimal conversion using the Ryū algorithm.
//
// See Ulf Adams, "Ryū: Fast Float-to-String Conversion" (doi:10.1145/3192366.3192369)
//
// Fixed precision formatting is a variant of the original paper's
// algorithm, where a single multiplication by 10^k is required,
// sharing the same rounding guarantees.

// ryuFtoaFixed32 formats mant*(2^exp) with prec decimal digits.
func ryuFtoaFixed32(d *decimalSlice, mant uint32, exp int, prec int) {
	if prec < 0 {
		panic("ryuFtoaFixed32 called with negative prec")
	}
	if prec > 9 {
		panic("ryuFtoaFixed32 called with prec > 9")
	}
	// Zero input.
	if mant == 0 {
		d.nd, d.dp = 0, 0
		return
	}
	// Renormalize to a 25-bit mantissa.
	e2 := exp
	if b := bits.Len32(mant); b < 25 {
		mant <<= uint(25 - b)
		e2 += int(b) - 25
	}
	// Choose an exponent such that rounded mant*(2^e2)*(10^q) has
	// at least prec decimal digits, i.e
	//     mant*(2^e2)*(10^q) >= 10^(prec-1)
	// Because mant >= 2^24, it is enough to choose:
	//     2^(e2+24) >= 10^(-q+prec-1)
	// or q = -mulByLog2Log10(e2+24) + prec - 1
	q := -mulByLog2Log10(e2+24) + prec - 1

	// Now compute mant*(2^e2)*(10^q).
	// Is it an exact computation?
	// Only small positive powers of 10 are exact (5^28 has 66 bits).
	exact := q <= 27 && q >= 0

	di, dexp2, d0 := mult64bitPow10(mant, e2, q)
	if dexp2 >= 0 {
		panic("not enough significant bits after mult64bitPow10")
	}
	// As a special case, computation might still be exact, if exponent
	// was negative and if it amounts to computing an exact division.
	// In that case, we ignore all lower bits.
	// Note that division by 10^11 cannot be exact as 5^11 has 26 bits.
	if q < 0 && q >= -10 && divisibleByPower5(uint64(mant), -q) {
		exact = true
		d0 = true
	}
	// Remove extra lower bits and keep rounding info.
	extra := uint(-dexp2)
	extraMask := uint32(1<<extra - 1)

	di, dfrac := di>>extra, di&extraMask
	roundUp := false
	if exact {
		// If we computed an exact product, d + 1/2
		// should round to d+1 if 'd' is odd.
		roundUp = dfrac > 1<<(extra-1) ||
			(dfrac == 1<<(extra-1) && !d0) ||
			(dfrac == 1<<(extra-1) && d0 && di&1 == 1)
	} else {
		// otherwise, d+1/2 always rounds up because
		// we truncated below.
		roundUp = dfrac>>(extra-1) == 1
	}
	if dfrac != 0 {
		d0 = false
	}
	// Proceed to the requested number of digits
	formatDecimal(d, uint64(di), !d0, roundUp, prec)
	// Adjust exponent
	d.dp -= q
}

// ryuFtoaFixed64 formats mant*(2^exp) with prec decimal digits.
func ryuFtoaFixed64(d *decimalSlice, mant uint64, exp int, prec int) {
	if prec > 18 {
		panic("ryuFtoaFixed64 called with prec > 18")
	}
	// Zero input.
	if mant == 0 {
		d.nd, d.dp = 0, 0
		return
	}
	// Renormalize to a 55-bit mantissa.
	e2 := exp
	if b := bits.Len64(mant); b < 55 {
		mant = mant << uint(55-b)
		e2 += int(b) - 55
	}
	// Choose an exponent such that rounded mant*(2^e2)*(10^q) has
	// at least prec decimal digits, i.e
	//     mant*(2^e2)*(10^q) >= 10^(prec-1)
	// Because mant >= 2^54, it is enough to choose:
	//     2^(e2+54) >= 10^(-q+prec-1)
	// or q = -mulByLog2Log10(e2+54) + prec - 1
	//
	// The minimal required exponent is -mulByLog2Log10(1025)+18 = -291
	// The maximal required exponent is mulByLog2Log10(1074)+18 = 342
	q := -mulByLog2Log10(e2+54) + prec - 1

	// Now compute mant*(2^e2)*(10^q).
	// Is it an exact computation?
	// Only small positive powers of 10 are exact (5^55 has 128 bits).
	exact := q <= 55 && q >= 0

	di, dexp2, d0 := mult128bitPow10(mant, e2, q)
	if dexp2 >= 0 {
		panic("not enough significant bits after mult128bitPow10")
	}
	// As a special case, computation might still be exact, if exponent
	// was negative and if it amounts to computing an exact division.
	// In that case, we ignore all lower bits.
	// Note that division by 10^23 cannot be exact as 5^23 has 54 bits.
	if q < 0 && q >= -22 && divisibleByPower5(mant, -q) {
		exact = true
		d0 = true
	}
	// Remove extra lower bits and keep rounding info.
	extra := uint(-dexp2)
	extraMask := uint64(1<<extra - 1)

	di, dfrac := di>>extra, di&extraMask
	roundUp := false
	if exact {
		// If we computed an exact product, d + 1/2
		// should round to d+1 if 'd' is odd.
		roundUp = dfrac > 1<<(extra-1) ||
			(dfrac == 1<<(extra-1) && !d0) ||
			(dfrac == 1<<(extra-1) && d0 && di&1 == 1)
	} else {
		// otherwise, d+1/2 always rounds up because
		// we truncated below.
		roundUp = dfrac>>(extra-1) == 1
	}
	if dfrac != 0 {
		d0 = false
	}
	// Proceed to the requested number of digits
	formatDecimal(d, di, !d0, roundUp, prec)
	// Adjust exponent
	d.dp -= q
}

// formatDecimal fills d with at most prec decimal digits
// of mantissa m. The boolean trunc indicates whether m
// is truncated compared to the original number being formatted.
func formatDecimal(d *decimalSlice, m uint64, trunc bool, roundUp bool, prec int) {
	max := uint64pow10[prec]
	trimmed := 0
	for m >= max {
		a, b := m/10, m%10
		m = a
		trimmed++
		if b > 5 {
			roundUp = true
		} else if b < 5 {
			roundUp = false
		} else { // b == 5
			// round up if there are trailing digits,
			// or if the new value of m is odd (round-to-even convention)
			roundUp = trunc || m&1 == 1
		}
		if b != 0 {
			trunc = true
		}
	}
	if roundUp {
		m++
	}
	if m >= max {
		// Happens if di was originally 99999....xx
		m /= 10
		trimmed++
	}
	// render digits (similar to formatBits)
	n := uint(prec)
	d.nd = int(prec)
	v := m
	for v >= 100 {
		var v1, v2 uint64
		if v>>32 == 0 {
			v1, v2 = uint64(uint32(v)/100), uint64(uint32(v)%100)
		} else {
			v1, v2 = v/100, v%100
		}
		n -= 2
		d.d[n+1] = smallsString[2*v2+1]
		d.d[n+0] = smallsString[2*v2+0]
		v = v1
	}
	if v > 0 {
		n--
		d.d[n] = smallsString[2*v+1]
	}
	if v >= 10 {
		n--
		d.d[n] = smallsString[2*v]
	}
	for d.d[d.nd-1] == '0' {
		d.nd--
		trimmed++
	}
	d.dp = d.nd + trimmed
}

// mulByLog2Log10 returns math.Floor(x * log(2)/log(10)) for an integer x in
// the range -1600 <= x && x <= +1600.
//
// The range restriction lets us work in faster integer arithmetic instead of
// slower floating point arithmetic. Correctness is verified by unit tests.
func mulByLog2Log10(x int) int {
	// log(2)/log(10) ≈ 0.30102999566 ≈ 78913 / 2^18
	return (x * 78913) >> 18
}

// mulByLog10Log2 returns math.Floor(x * log(10)/log(2)) for an integer x in
// the range -500 <= x && x <= +500.
//
// The range restriction lets us work in faster integer arithmetic instead of
// slower floating point arithmetic. Correctness is verified by unit tests.
func mulByLog10Log2(x int) int {
	// log(10)/log(2) ≈ 3.32192809489 ≈ 108853 / 2^15
	return (x * 108853) >> 15
}

// mult64bitPow10 takes a floating-point input with a 25-bit
// mantissa and multiplies it with 10^q. The resulting mantissa
// is m*P >> 57 where P is a 64-bit element of the detailedPowersOfTen tables.
// It is typically 31 or 32-bit wide.
// The returned boolean is true if all trimmed bits were zero.
//
// That is:
//     m*2^e2 * round(10^q) = resM * 2^resE + ε
//     exact = ε == 0
func mult64bitPow10(m uint32, e2, q int) (resM uint32, resE int, exact bool) {
	if q == 0 {
		return m << 7, e2 - 7, true
	}
	if q < detailedPowersOfTenMinExp10 || detailedPowersOfTenMaxExp10 < q {
		// This never happens due to the range of float32/float64 exponent
		panic("mult64bitPow10: power of 10 is out of range")
	}
	pow := detailedPowersOfTen[q-detailedPowersOfTenMinExp10][1]
	if q < 0 {
		// Inverse powers of ten must be rounded up.
		pow += 1
	}
	hi, lo := bits.Mul64(uint64(m), pow)
	e2 += mulByLog10Log2(q) - 63 + 57
	return uint32(hi<<7 | lo>>57), e2, lo<<7 == 0
}

// mult128bitPow10 takes a floating-point input with a 55-bit
// mantissa and multiplies it with 10^q. The resulting mantissa
// is m*P >> 119 where P is a 128-bit element of the detailedPowersOfTen tables.
// It is typically 63 or 64-bit wide.
// The returned boolean is true is all trimmed bits were zero.
//
// That is:
//     m*2^e2 * round(10^q) = resM * 2^resE + ε
//     exact = ε == 0
func mult128bitPow10(m uint64, e2, q int) (resM uint64, resE int, exact bool) {
	if q == 0 {
		return m << 9, e2 - 9, true
	}
	if q < detailedPowersOfTenMinExp10 || detailedPowersOfTenMaxExp10 < q {
		// This never happens due to the range of float32/float64 exponent
		panic("mult128bitPow10: power of 10 is out of range")
	}
	pow := detailedPowersOfTen[q-detailedPowersOfTenMinExp10]
	if q < 0 {
		// Inverse powers of ten must be rounded up.
		pow[0] += 1
	}
	e2 += mulByLog10Log2(q) - 127 + 119

	// long multiplication
	l1, l0 := bits.Mul64(m, pow[0])
	h1, h0 := bits.Mul64(m, pow[1])
	mid, carry := bits.Add64(l1, h0, 0)
	h1 += carry
	return h1<<9 | mid>>55, e2, mid<<9 == 0 && l0 == 0
}

func divisibleByPower5(m uint64, k int) bool {
	if m == 0 {
		return true
	}
	for i := 0; i < k; i++ {
		if m%5 != 0 {
			return false
		}
		m /= 5
	}
	return true
}
