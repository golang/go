// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import "math/bits"

var uint64pow10 = [...]uint64{
	1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
	1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
}

// fixedFtoa formats a number of decimal digits of mant*(2^exp) into d,
// where mant > 0 and 1 ≤ digits ≤ 18.
func fixedFtoa(d *decimalSlice, mant uint64, exp, digits int) {
	// The strategy here is to multiply (mant * 2^exp) by a power of 10
	// to make the resulting integer be the number of digits we want.
	//
	// Adams proved in the Ryu paper that 128-bit precision in the
	// power-of-10 constant is sufficient to produce correctly
	// rounded output for all float64s, up to 18 digits.
	// https://dl.acm.org/doi/10.1145/3192366.3192369
	//
	// TODO(rsc): The paper is not focused on, nor terribly clear about,
	// this fact in this context, and the proof seems too complicated.
	// Post a shorter, more direct proof and link to it here.

	if digits > 18 {
		panic("fixedFtoa called with digits > 18")
	}

	// Shift mantissa to have 64 bits,
	// so that the 192-bit product below will
	// have at least 63 bits in its top word.
	b := 64 - bits.Len64(mant)
	mant <<= b
	exp -= b

	// We have f = mant * 2^exp ≥ 2^(63+exp)
	// and we want to multiply it by some 10^p
	// to make it have the number of digits plus one rounding bit:
	//
	//	2 * 10^(digits-1) ≤ f * 10^p < ~2 * 10^digits
	//
	// The lower bound is required, but the upper bound is approximate:
	// we must not have too few digits, but we can round away extra ones.
	//
	//	f * 10^p ≥ 2 * 10^(digits-1)
	//	10^p ≥ 2 * 10^(digits-1) / f                         [dividing by f]
	//	p ≥ (log₁₀ 2) + (digits-1) - log₁₀ f                 [taking log₁₀]
	//	p ≥ (log₁₀ 2) + (digits-1) - log₁₀ (mant * 2^exp)    [expanding f]
	//	p ≥ (log₁₀ 2) + (digits-1) - (log₁₀ 2) * (64 + exp)  [mant < 2⁶⁴]
	//	p ≥ (digits - 1) - (log₁₀ 2) * (63 + exp)            [refactoring]
	//
	// Once we have p, we can compute the scaled value:
	//
	//	dm * 2^de = mant * 2^exp * 10^p
	//	          = mant * 2^exp * pow/2^128 * 2^exp2.
	//	          = (mant * pow/2^128) * 2^(exp+exp2).
	p := (digits - 1) - mulLog10_2(63+exp)
	pow, exp2, ok := pow10(p)
	if !ok {
		// This never happens due to the range of float32/float64 exponent
		panic("fixedFtoa: pow10 out of range")
	}
	if -22 <= p && p < 0 {
		// Special case: Let q=-p. q is in [1,22]. We are dividing by 10^q
		// and the mantissa may be a multiple of 5^q (5^22 < 2^53),
		// in which case the division must be computed exactly and
		// recorded as exact for correct rounding. Our normal computation is:
		//
		//	dm = floor(mant * floor(10^p * 2^s))
		//
		// for some scaling shift s. To make this an exact division,
		// it suffices to change the inner floor to a ceil:
		//
		//	dm = floor(mant * ceil(10^p * 2^s))
		//
		// In the range of values we are using, the floor and ceil
		// cancel each other out and the high 64 bits of the product
		// come out exactly right.
		// (This is the same trick compilers use for division by constants.
		// See Hacker's Delight, 2nd ed., Chapter 10.)
		pow.Lo++
	}
	dm, lo1, lo0 := umul192(mant, pow)
	de := exp + exp2

	// Check whether any bits have been truncated from dm.
	// If so, set dt != 0. If not, leave dt == 0 (meaning dm is exact).
	var dt uint
	switch {
	default:
		// Most powers of 10 use a truncated constant,
		// meaning the result is also truncated.
		dt = 1
	case 0 <= p && p <= 55:
		// Small positive powers of 10 (up to 10⁵⁵) can be represented
		// precisely in a 128-bit mantissa (5⁵⁵ ≤ 2¹²⁸), so the only truncation
		// comes from discarding the low bits of the 192-bit product.
		//
		// TODO(rsc): The new proof mentioned above should also
		// prove that we can't have lo1 == 0 and lo0 != 0.
		// After proving that, drop computation and use of lo0 here.
		dt = bool2uint(lo1|lo0 != 0)
	case -22 <= p && p < 0 && divisiblePow5(mant, -p):
		// If the original mantissa was a multiple of 5^p,
		// the result is exact. (See comment above for pow.Lo++.)
		dt = 0
	}

	// The value we want to format is dm * 2^de, where de < 0.
	// Multply by 2^de by shifting, but leave one extra bit for rounding.
	// After the shift, the "integer part" of dm is dm>>1,
	// the "rounding bit" (the first fractional bit) is dm&1,
	// and the "truncated bit" (have any bits been discarded?) is dt.
	shift := -de - 1
	dt |= bool2uint(dm&(1<<shift-1) != 0)
	dm >>= shift

	// Set decimal point in eventual formatted digits,
	// so we can update it as we adjust the digits.
	d.dp = digits - p

	// Trim excess digit if any, updating truncation and decimal point.
	// The << 1 is leaving room for the rounding bit.
	max := uint64pow10[digits] << 1
	if dm >= max {
		var r uint
		dm, r = dm/10, uint(dm%10)
		dt |= bool2uint(r != 0)
		d.dp++
	}

	// Round and shift away rounding bit.
	// We want to round up when
	// (a) the fractional part is > 0.5 (dm&1 != 0 and dt == 1)
	// (b) or the fractional part is ≥ 0.5 and the integer part is odd
	//     (dm&1 != 0 and dm&2 != 0).
	// The bitwise expression encodes that logic.
	dm += uint64(uint(dm) & (dt | uint(dm)>>1) & 1)
	dm >>= 1
	if dm == max>>1 {
		// 999... rolled over to 1000...
		dm = uint64pow10[digits-1]
		d.dp++
	}

	// Format digits into d.
	formatBase10(d.d[:digits], dm)
	d.nd = digits
	for d.d[d.nd-1] == '0' {
		d.nd--
	}
}
