// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import (
	"math/bits"
)

// Binary to decimal conversion using the Dragonbox algorithm by Junekey Jeon.
// Junekey Jeon has agreed to license this port under a BSD-style license,
// specifically for inclusion in the Go source code.
//
// Fixed precision format is not supported by the Dragonbox algorithm
// so we continue to use Ryū-printf for this purpose.
// See https://github.com/jk-jeon/dragonbox/issues/38 for more details.
//
// For binary to decimal rounding, uses round to nearest, tie to even.
// For decimal to binary rounding, assumes round to nearest, tie to even.
//
// The original paper by Junekey Jeon can be found at:
// https://github.com/jk-jeon/dragonbox/blob/d5dc40ae6a3f1a4559cda816738df2d6255b4e24/other_files/Dragonbox.pdf
//
// The reference implementation in C++ by Junekey Jeon can be found at:
// https://github.com/jk-jeon/dragonbox/blob/6c7c925b571d54486b9ffae8d9d18a822801cbda/subproject/simple/include/simple_dragonbox.h

// dragonboxFtoa computes the decimal significand and exponent
// from the binary significand and exponent using the Dragonbox algorithm
// and formats the decimal floating point number in d.
func dragonboxFtoa(d *decimalSlice, mant uint64, exp int, denorm bool, bitSize int) {
	if bitSize == 32 {
		dragonboxFtoa32(d, uint32(mant), exp, denorm)
		return
	}
	dragonboxFtoa64(d, mant, exp, denorm)
}

func dragonboxFtoa64(d *decimalSlice, mant uint64, exp int, denorm bool) {
	// A floating point number w is represented as w = (-1)^σ * Fw * 2^Ew, where:
	// σ is the sign of w.
	// Fw is the significand of w.
	// Ew is the exponent of w.
	//
	// fc = Fw * 2^p is the adjusted significand of w.
	// e = Ew - p is the adjusted exponent of w.
	// where p is the number of significand bits.
	// mant and exp should be adjusted before the call
	// so that mant = fc and exp = e.
	//
	// I ⊆ ℝ is the interval such that each r ∈ I rounds to w
	// during decimal to binary conversion using round to nearest, tie to even.
	// Δ is length of the interval I.
	// wL = (w^- + w) / 2 is the left endpoint of the interval I, and;
	// wR = (w + w^+) / 2 is right endpoint of the interval I, where:
	// w^+ is the smallest floating point number larger than w
	// w^- is the largest floating point number smaller than w

	// Short path (denormalized and zero mantissa)
	if mant == 0 {
		d.nd, d.dp = 0, 0
		return
	}

	if mant == (1<<mantBits64) && !denorm {
		// Algorithm 5.6 (Skeleton of Dragonbox, part 3)
		// Shorter interval case (Δ = 3*2^(e-2))
		// This is the case iff Fw = 1 and Ew ≠ Emin.

		// Note that in the shorter interval case:
		// k0 = -⌊log10(Δ)⌋
		// x = 10^k0*wL, y = 10^k0*w, z = 10^k0*wR
		// δ = z - x = (10^k0)*Δ

		// Compute -k0 using bit tricks (section 6.3).
		// -k0 = ⌊log10(Δ)⌋ = ⌊log10(3*2^(e-2))⌋
		//     = ⌊(e-2)*log10(2)+log10(3)⌋ = ⌊log10(2^e)-log10(4/3)⌋
		minusK0 := floorLog10Pow2MinusLog10_4Over3(exp) // -k0

		// Compute x^(i) and z^(i) from the precomputed table of φ̃k0 (section 5.2.1).
		beta := exp + floorLog2Pow10(-minusK0)  // β = e + ⌊k0*log2(10)⌋
		phi := getCache64(-minusK0)             // φ̃k0
		xi := computeLeftEndpoint64(phi, beta)  // x^(i)
		zi := computeRightEndpoint64(phi, beta) // z^(i)

		// Compute x̃^(i) and z̃^(i) from x^(i) and z^(i).
		// x̃^(i) = x^(i) if e ∈ [2, 3] and x ∈ 10^(k0)I (page 23)
		// x̃^(i) = x^(i) + 1 otherwise
		// x ∈ 10^(k0)I is always true for round to nearest, tie to even,
		// since Fw is even in the shorter interval case.
		if !(2 <= exp && exp <= 3) {
			xi++ // x̃^(i) = x^(i) + 1
		}
		// z̃^(i) = z^(i) - 1 if e ∈ [0, 3] and z ∉ 10^(k0)I (page 23)
		// z̃^(i) = z^(i) otherwise
		// z ∉ 10^(k0)I is always false for round to nearest, tie to even,
		// since Fw is even in the shorter interval case.
		// Thus we let z̃^(i) = z^(i).

		// Check if I ∩ 10^(-k0+1)ℤ is non-empty.
		// If I ∩ 10^(-k0+1)ℤ is non-empty,
		// the unique element in it has the smallest number of
		// decimal significand digits (corollary 3.3).

		// I ∩ 10^(-k0+1)ℤ is non-empty iff
		// x̃^(i) ≤ ⌊z̃^(i)/10⌋*10 (proposition 5.5). In this case,
		// ⌊z̃^(i)/10⌋*10^(-k0+1) is the unique element in I ∩ 10^(-k0+1)ℤ
		q := zi / 10 // q = ⌊z̃^(i)/10⌋
		if xi <= q*10 {
			mant, exp := removeTrailingZeros64(q, minusK0+1)
			dragonboxDigits64(d, mant, exp)
			return
		}

		// Find elements in I ∩ 10^(-k0)ℤ.
		// I ∩ 10^(-k0)ℤ is guaranteed to be non-empty (proposition 3.1).
		// Since I ∩ 10^(-k0+1)ℤ must be empty at this point,
		// any element in I ∩ 10^(-k0)ℤ has the smallest number of
		// decimal significand digits (corollary 3.3).

		// Compute y^(ru) = ⌊y+1/2⌋ (ru stands for round up).
		// y^(ru) is computed directly unlike the normal interval case.
		yru := computeRoundUp64(phi, beta) // y^(ru) (section 5.2.2)

		// Check if y^(ru) = y^(rd) (page 17).
		// Note that y^(ru) = ⌊y+1/2⌋ and y^(rd) = ⌈y-1/2⌉.
		// y^(ru) = y^(rd)+1 iff the fractional part of y is 1/2
		//                   iff e ∈ [-77, -77] for float64 (section 5.2.4)
		// y^(ru) = y^(rd) otherwise
		// A tie happens between y^(ru) and y^(rd) if y^(ru) = y^(rd)+1,
		// so we need to break the tie according to
		// the binary to decimal rounding mode (round to nearest, tie to even).
		if exp == -77 && yru%2 != 0 {
			yru-- // y^(rd) = y^(ru)-1
		} else {
			// Check if y^(ru) is in 10^(k0)I.
			// Neither y^(ru) nor y^(rd) are guaranteed to be in 10^(k0)I,
			// unlike the normal interval case.
			// So we need to check if x^(i) ≤ y^(ru), y^(rd) ≤ z^(i) explicitly.

			// y^(ru) is guaranteed to be at most z̃^(i), and;
			// y^(ru)+1 is guaranteed to be in 10^(k0)I if
			// y^(ru) is not in 10^(k0)I (page 23). Therefore we have:
			// if x^(i) ≤ y^(ru), then y^(ru)+1 is in 10^(k0)I
			// if y^(ru) < x^(i), then y^(ru) is in 10^(k0)I
			// Since a tie does not happen in this branch,
			// y^(ru) or y^(ru)+1 is the correct element in both cases.
			if yru < xi {
				yru++ // y^(ru)+1
			}
			// Both y^(ru) and y^(rd) are guaranteed to be
			// in 10^(k0)I in the other branch.
			// Since a tie happens iff the fractional part of y is 1/2
			//                     iff both y^(ru) and y^(rd) are equidistant from y
			// Thus no other integers could lie in 10^(k0)I otherwise.
		}
		dragonboxDigits64(d, yru, minusK0)
		return
	}

	// Normal interval case (Δ = 2^e)
	// This is the case iff Fw ≠ 1 or Ew = Emin.

	// Note that in the normal interval case:
	// k = k0 + κ
	// x = 10^k*wL, y = 10^k*w, z = 10^k*wR
	// δ = z-x = (10^k)*Δ

	const kappa = 2           // κ = 2 for float64 (section 5.1.3)
	const largeDivisor = 1000 // 10^(κ+1)
	const smallDivisor = 100  // 10^κ

	// Compute -k0 using bit tricks (section 6.1).
	// -k = -k0 - κ = ⌊log10(Δ)⌋ - κ
	//    = ⌊log10(2^e)⌋ - κ
	minusK := floorLog10Pow2(exp) - kappa // -k

	// Compute z^(i) from the precomputed table of φ̃k (section 5.1.5).
	beta := exp + floorLog2Pow10(-minusK)                   // β = e + ⌊k*log2(10)⌋
	phi := getCache64(-minusK)                              // φ̃k
	zi, zIsInt := computeMul64(uint64(mant*2+1)<<beta, phi) // z^(i), z^(f) = 0

	// Compute δ^(i) from the precomputed table of φ̃k (section 5.1.4)
	deltai := computeDelta64(phi, beta) // δ^(i)

	// Algorithm 5.2 (Skeleton of Dragonbox, part 1)
	// Check if I ∩ 10^(-k0+1)ℤ is non-empty.
	// If I ∩ 10^(-k0+1)ℤ is non-empty,
	// the unique element in it has the smallest number of
	// decimal significand digits (corollary 3.3).

	// Divide z^(i) by 10^(κ+1).
	// This should be optimized by the compiler using bit tricks.
	s := zi / largeDivisor           // s is the quotient
	r := uint32(zi - largeDivisor*s) // r is the remainder

	// Check if I ∩ 10^(-k0+1)ℤ contains s (proposition 5.1):
	// If I = [wL, wR] (i.e., Fw is even for round to nearest, tie to even),
	// I ∩ 10^(-k0+1)ℤ contains s iff r+z^(f) ≤ δ.
	// If I = (wL, wR) (i.e., Fw is odd for round to nearest, tie to even),
	// I ∩ 10^(-k0+1)ℤ contains s iff r+z^(f) < δ and (r ≠ 0 or z^(f) ≠ 0).
	// We must make sure r ≠ 0 or z^(f) ≠ 0 if I = (wL, wR),
	// since s will otherwise overlap with wR ∉ I if r = 0 and z^(f) = 0.
	if r < deltai {
		// r < δ^(i) in this branch.
		// If I = [wL, wR], r < δ^(i) implies r+z^(f) ≤ δ.
		// If I = (wL, wR), r < δ^(i) implies r+z^(f) < δ.
		if r != 0 || !zIsInt || mant%2 == 0 {
			mant, exp := removeTrailingZeros64(s, minusK+kappa+1)
			dragonboxDigits64(d, mant, exp)
			return
		}
		// r = 0 at this point.
		// Compute s̃ and r̃ in advance for the next part (page 17).
		// This ensures D > 0 prior to the division of D by 10^κ (see below).
		s--              // s̃ = s - 1 if r = 0
		r = largeDivisor // r̃ = 10^(κ+1) if r = 0
	} else if r == deltai {
		// r = δ^(i) in this branch.
		// If I = [wL, wR], r = δ^(i) and z^(f) ≤ δ^(f) implies r+z^(f) ≤ δ.
		// If I = (wL, wR), r = δ^(i) and z^(f) < δ^(f) implies r+z^(f) < δ.

		// Check z^(f) < δ^(f) efficiently by the parity of x^(i) (page 15):
		// z^(f) < δ^(f) iff x^(i) is odd
		// z^(f) ≤ δ^(f) iff x^(i) is odd or z^(f) = δ^(f)
		//               iff x^(i) is odd or x^(f) = 0
		xiParity, xIsInt := computeMulParity64(uint64(mant*2-1), phi, beta)
		if xiParity || (xIsInt && mant%2 == 0) {
			mant, exp := removeTrailingZeros64(s, minusK+kappa+1)
			dragonboxDigits64(d, mant, exp)
			return
		}
		// r ≠ 0 at this point since r = δ^(i) and δ^(i) ≥ 10^κ ≠ 0.
		// Thus we let s̃ = s and r̃ = r (page 17).
	}

	// Algorithm 5.4 (Skeleton of Dragonbox, part 2)
	// Find elements in I ∩ 10^(-k0)ℤ.
	// I ∩ 10^(-k0)ℤ is guaranteed to be non-empty (proposition 3.1).
	// Since I ∩ 10^(-k0+1)ℤ must be empty at this point,
	// any element in I ∩ 10^(-k0)ℤ has the smallest number of
	// decimal significand digits (corollary 3.3).

	// Compute D = ⌊r̃+(10^κ/2)-ε^(i)⌋ where ε = δ/2 (page 17).
	// D is a part of the floor term in y^(ru) (see below).
	D := uint32(r + (smallDivisor / 2) - (deltai / 2))

	// Divide D by 10^κ.
	// This should be optimized by the compiler using bit tricks.
	t := uint32(D / smallDivisor) // t is the quotient
	rho := D - t*smallDivisor     // ρ is the remainder

	// Compute y^(ru) = ⌊y/10^κ+1/2⌋
	//                = 10s̃ + ⌊(D+(z^(f)-ε^(f)))/10^κ⌋
	//                = 10s̃+t + ⌊(ρ+(z^(f)-ε^(f)))/10^κ⌋
	// assuming the residue term ⌊(ρ+(z^(f)-ε^(f)))/10^κ⌋ is zero for now.
	yru := 10*s + uint64(t) // y^(ru) = 10s̃+t

	if rho == 0 {
		// The residue term ⌊(ρ+(z^(f)-ε^(f)))/10^κ⌋ in y^(ru) is non-zero
		// (more precisely, equals -1) if ρ = 0 and z^(f) < ε^(f).

		// Check z^(f) < ε^(f) efficiently by the parity of y^(i) (page 17):
		// z^(f) < ε^(f) iff parity of y^(i) ≠ parity of z^(i) - ε^(i)
		//               iff parity of y^(i) ≠ parity of (D - (10^κ)/2)
		// Note that parity of z^(i) - ε^(i) = parity of (D-(10^κ)/2) because:
		// 1. D - (10^κ)/2 = r̃ − ε^(i), and;
		// 2. parity of r̃ = parity of z^(i) (since z^(i) = 10s̃ + r̃ = 2*5s̃ + r̃).
		yiParity, yIsInt := computeMulParity64(mant*2, phi, beta)
		yiParityApprox := (D-smallDivisor/2)%2 != 0
		if yiParity != yiParityApprox {
			yru-- // y^(ru) = 10s̃+t-1
		} else {
			// Check if y^(ru) = y^(rd) (page 17).
			// Note that y^(ru) = ⌊y/10^κ+1/2⌋ and y^(rd) = ⌈y/10^κ-1/2⌉.
			// y^(ru) = y^(rd)+1 iff the fractional part of y/10^κ is 1/2
			//                   iff ρ = 0 and z^(f) - ε^(f) = 0
			//                   iff ρ = 0 and y is an integer
			// y^(ru) = y^(rd) otherwise
			// A tie happens between y^(ru) and y^(rd) if y^(ru) = y^(rd)+1,
			// so we need to break the tie according to
			// the binary to decimal rounding mode (round to nearest, tie to even).
			if yIsInt && yru%2 != 0 {
				yru-- // y^(rd) = y^(ru)-1
			}
			// Since a tie only happens if z^(f) - ε^(f) = 0,
			// it does not happen in the other branch where z^(f) < ε^(f).
		}
	}
	dragonboxDigits64(d, yru, minusK+kappa)
}

// Almost identical to dragonboxFtoa64.
// This is kept as a separate copy to minimize runtime overhead.
func dragonboxFtoa32(d *decimalSlice, mant uint32, exp int, denorm bool) {
	if mant == 0 {
		d.nd, d.dp = 0, 0
		return
	}

	if mant == (1<<mantBits32) && !denorm {
		minusK0 := floorLog10Pow2MinusLog10_4Over3(exp)

		beta := exp + floorLog2Pow10(-minusK0)
		phi := getCache32(-minusK0)
		xi := computeLeftEndpoint32(phi, beta)
		zi := computeRightEndpoint32(phi, beta)

		if !(2 <= exp && exp <= 3) {
			xi++
		}

		q := zi / 10
		if xi <= q*10 {
			mant, exp := removeTrailingZeros32(q, minusK0+1)
			dragonboxDigits32(d, mant, exp)
			return
		}

		yru := computeRoundUp32(phi, beta)
		if exp == -35 && yru%2 != 0 {
			yru--
		} else if yru < xi {
			yru++
		}
		dragonboxDigits32(d, yru, minusK0)
		return
	}

	const kappa = 1
	const bigDivisor = 100
	const smallDivisor = 10

	minusK := floorLog10Pow2(exp) - kappa

	beta := exp + floorLog2Pow10(-minusK)
	phi := getCache32(-minusK)
	zi, zIsInt := computeMul32(uint32(mant*2+1)<<beta, phi)

	deltai := computeDelta32(phi, beta)

	s := zi / bigDivisor
	r := uint32(zi - bigDivisor*s)

	if r < deltai {
		if r != 0 || !zIsInt || mant%2 == 0 {
			mant, exp := removeTrailingZeros32(s, minusK+kappa+1)
			dragonboxDigits32(d, mant, exp)
			return
		}
		s--
		r = bigDivisor
	} else if r == deltai {
		xiParity, xIsInt := computeMulParity32(mant*2-1, phi, beta)
		if xiParity || (xIsInt && mant%2 == 0) {
			mant, exp := removeTrailingZeros32(s, minusK+kappa+1)
			dragonboxDigits32(d, mant, exp)
			return
		}
	}

	D := uint32(r + (smallDivisor / 2) - (deltai / 2))

	t := uint32(D / smallDivisor)
	rho := D - t*smallDivisor

	yru := 10*s + t

	if rho == 0 {
		yiParity, yIsInt := computeMulParity32(mant*2, phi, beta)
		yiParityApprox := (D-smallDivisor/2)%2 != 0
		if yiParity != yiParityApprox {
			yru--
		} else {
			if yIsInt && yru%2 != 0 {
				yru--
			}
		}
	}
	dragonboxDigits32(d, yru, minusK+kappa)
}

// Fast digit generation algorithm adapted from the original implementation
// by Junekey Jeon as part of the Dragonbox algorithm,
// which in turn is inspired by James Anhalt's itoa algorithm.
// This should be faster than (unrolled) lut,
// as this algorithm can reduce the number of multiplications almost by half.
// See https://jk-jeon.github.io/posts/2022/02/jeaiii-algorithm/ for more details.
//
// The original itoa algorithm in C++ by James Anhalt can be found at:
// https://github.com/jeaiii/itoa/blob/69308f65e87a9954f11f952ed04d551eabeee0ae/include/itoa/jeaiii_to_text.h
//
// The reference implementation in C++ by Junekey Jeon can be found at:
// https://github.com/jk-jeon/dragonbox/blob/6c7c925b571d54486b9ffae8d9d18a822801cbda/source/dragonbox_to_chars.cpp

// dragonboxDigits64 emits decimal digits of mant in d for float64
// and adjusts the decimal point based on exp.
func dragonboxDigits64(d *decimalSlice, mant uint64, exp int) {
	// mant should not have any trailing zeroes in decimal.
	// mant has at most ⌈log10(2^(52+1))⌉+1 = 16+1 = 17 decimal digits.
	// Note the +1 in 52+1 since the MSB is implicit in IEEE754, and
	// also the +1 in 16+1 for the round trip guarantee.
	if mant < 100_000_000 {
		// mant has 9 digits or less.
		print9Digits(d, uint32(mant))
	} else {
		// mant has 10 digits or more.
		// Divide mant into two blocks of at most 9 and 8 digits respectively.
		first := uint32(mant / 100_000_000)        // First 9 digits
		second := uint32(mant) - first*100_000_000 // Last 8 digits
		print9Digits(d, first)
		print8Digits(d, second)
	}
	// Adjust decimal point.
	d.dp = d.nd + exp
}

// dragonboxDigits32 emits decimal digits of mant in d for float32
// and adjusts the decimal point based on exp.
func dragonboxDigits32(d *decimalSlice, mant uint32, exp int) {
	// mant has at most ⌈log10(2^(23+1))⌉+1 = 8+1 = 9 decimal digits.
	print9Digits(d, mant)
	// Adjust decimal point.
	d.dp = d.nd + exp
}

// print9Digits emits at most 9 decimal digits of block in d.
func print9Digits(d *decimalSlice, block uint32) {
	buf := d.d
	if block < 100 {
		// block has 1 or 2 digits.
		n := int(block)
		if n >= 10 {
			// block has 2 digits.
			print2Digits(buf, 0, n)
			d.nd += 2
		} else {
			// block has 1 digit.
			buf[0] = byte(n + '0')
			d.nd += 1
		}
	} else if block < 10_000 {
		// block has 3 or 4 digits.
		// 42949673 = ⌈2^32 / 100⌉
		prod := uint64(block) * 42949673
		n := int(prod >> 32)
		if n >= 10 {
			// block has 4 digits.
			print2Digits(buf, 0, n)
			prod = uint64(uint32(prod)) * 100
			print2Digits(buf, 0+2, int(prod>>32))
			d.nd += 4
		} else {
			// block has 3 digits.
			buf[0] = byte(n + '0')
			prod = uint64(uint32(prod)) * 100
			print2Digits(buf, 0+1, int(prod>>32))
			d.nd += 3
		}
	} else if block < 1_000_000 {
		// block has 5 or 6 digits.
		// 429497 = ⌈2^32 / 10,000⌉
		prod := uint64(block) * 429497
		n := int(prod >> 32)
		if n >= 10 {
			// block has 6 digits.
			print2Digits(buf, 0, n)
			prod = uint64(uint32(prod)) * 100
			print2Digits(buf, 0+2, int(prod>>32))
			prod = uint64(uint32(prod)) * 100
			print2Digits(buf, 2+2, int(prod>>32))
			d.nd += 6
		} else {
			// block has 5 digits.
			buf[0] = byte(n + '0')
			prod = uint64(uint32(prod)) * 100
			print2Digits(buf, 0+1, int(prod>>32))
			prod = uint64(uint32(prod)) * 100
			print2Digits(buf, 2+1, int(prod>>32))
			d.nd += 5
		}
	} else if block < 100_000_000 {
		// block has 7 or 8 digits.
		// 281474978 = ⌈2^48 / 1,000,000⌉ + 1
		prod := uint64(block) * 281474978
		prod >>= 16
		n := int(prod >> 32)
		if n >= 10 {
			// block has 8 digits.
			print2Digits(buf, 0, n)
			prod = uint64(uint32(prod)) * 100
			print2Digits(buf, 0+2, int(prod>>32))
			prod = uint64(uint32(prod)) * 100
			print2Digits(buf, 2+2, int(prod>>32))
			prod = uint64(uint32(prod)) * 100
			print2Digits(buf, 4+2, int(prod>>32))
			d.nd += 8
		} else {
			// block has 7 digits.
			buf[0] = byte(n + '0')
			prod = uint64(uint32(prod)) * 100
			print2Digits(buf, 0+1, int(prod>>32))
			prod = uint64(uint32(prod)) * 100
			print2Digits(buf, 2+1, int(prod>>32))
			prod = uint64(uint32(prod)) * 100
			print2Digits(buf, 4+1, int(prod>>32))
			d.nd += 7
		}
	} else {
		// block has 9 digits.
		// 1441151882 = ⌈2^57 / 100,000,000⌉ + 1
		prod := uint64(block) * 1441151882
		prod >>= 25
		n := int(prod >> 32)
		buf[0] = byte(n + '0')
		// Repeated manually since the current compiler fails to
		// unroll the loop with constant number of iterations.
		// The calls to print2Digits should be inlined automatically.
		prod = uint64(uint32(prod)) * 100
		print2Digits(buf, 0+1, int(prod>>32))
		prod = uint64(uint32(prod)) * 100
		print2Digits(buf, 2+1, int(prod>>32))
		prod = uint64(uint32(prod)) * 100
		print2Digits(buf, 4+1, int(prod>>32))
		prod = uint64(uint32(prod)) * 100
		print2Digits(buf, 6+1, int(prod>>32))
		// Written 9 digits.
		d.nd += 9
	}
}

// print9Digits emits at most 8 decimal digits of block in d.
func print8Digits(d *decimalSlice, block uint32) {
	// block has 8 digits.
	buf, ofs := d.d, d.nd
	// 281474978 = ⌈2^48 / 1,000,000⌉ + 1
	prod := uint64(block) * 281474978
	prod >>= 16
	prod++
	// Offset the index by d.nd since this may be called after print9Digits.
	print2Digits(buf, ofs+0, int(prod>>32))
	prod = uint64(uint32(prod)) * 100
	print2Digits(buf, ofs+0+2, int(prod>>32))
	prod = uint64(uint32(prod)) * 100
	print2Digits(buf, ofs+2+2, int(prod>>32))
	prod = uint64(uint32(prod)) * 100
	print2Digits(buf, ofs+4+2, int(prod>>32))
	d.nd += 8
}

// print2Digits emits 2 decimal digits of n in buf starting at i.
// n should be in the range [0, 99].
func print2Digits(buf []byte, i int, n int) {
	buf[i+0] = smallsString[n*2+0]
	buf[i+1] = smallsString[n*2+1]
}

// uint128 represents 128-bit integer as a pair of high/low 64 bits.
type uint128 struct {
	hi, lo uint64
}

// uadd128 returns the full 128 bits of u + n.
func uadd128(u uint128, n uint64) uint128 {
	sum := uint64(u.lo + n)
	// Check if lo is wrapped around.
	if sum < u.lo {
		u.hi++
	}
	u.lo = sum
	return u
}

// umul64 returns the full 64 bits of x * y.
func umul64(x, y uint32) uint64 {
	return uint64(x) * uint64(y)
}

// umul96Upper64 returns the upper 64 bits (out of 96 bits) of x * y.
func umul96Upper64(x uint32, y uint64) uint64 {
	yh := uint32(y >> 32)
	yl := uint32(y)

	xyh := umul64(x, yh)
	xyl := umul64(x, yl)

	return xyh + (xyl >> 32)
}

// umul96Lower64 returns the lower 64 bits (out of 96 bits) of x * y.
func umul96Lower64(x uint32, y uint64) uint64 {
	return uint64(uint64(x) * y)
}

// umul128 returns the full 128 bits of x * y.
func umul128(x, y uint64) uint128 {
	a := uint32(x >> 32)
	b := uint32(x)
	c := uint32(y >> 32)
	d := uint32(y)

	ac := umul64(a, c)
	bc := umul64(b, c)
	ad := umul64(a, d)
	bd := umul64(b, d)

	intermediate := uint64(bd>>32) + uint64(uint32(ad)) + uint64(uint32(bc))

	hi := ac + (intermediate >> 32) + (ad >> 32) + (bc >> 32)
	lo := (intermediate << 32) + uint64(uint32(bd))
	return uint128{hi, lo}
}

// umul128Upper64 returns the upper 64 bits (out of 128 bits) of x * y.
func umul128Upper64(x, y uint64) uint64 {
	a := uint32(x >> 32)
	b := uint32(x)
	c := uint32(y >> 32)
	d := uint32(y)

	ac := umul64(a, c)
	bc := umul64(b, c)
	ad := umul64(a, d)
	bd := umul64(b, d)

	intermediate := (bd >> 32) + uint64(uint32(ad)) + uint64(uint32(bc))

	return ac + (intermediate >> 32) + (ad >> 32) + (bc >> 32)
}

// umul192Upper128 returns the upper 128 bits (out of 192 bits) of x * y.
func umul192Upper128(x uint64, y uint128) uint128 {
	r := umul128(x, y.hi)
	t := umul128Upper64(x, y.lo)
	return uadd128(r, t)
}

// umul192Lower128 returns the lower 128 bits (out of 192 bits) of x * y.
func umul192Lower128(x uint64, y uint128) uint128 {
	high := x * y.hi
	highLow := umul128(x, y.lo)
	return uint128{uint64(high + highLow.hi), highLow.lo}
}

// computeMul64 computes x^(i), y^(i), z^(i)
// from the precomputed value of φ̃k for float64
// and also checks if x^(f), y^(f), z^(f) == 0 (section 5.2.1).
func computeMul64(u uint64, phi uint128) (intPart uint64, isInt bool) {
	r := umul192Upper128(u, phi)
	intPart = r.hi
	isInt = r.lo == 0
	return
}

// computeMul64 computes x^(i), y^(i), z^(i)
// from the precomputed value of φ̃k for float32
// and also checks if x^(f), y^(f), z^(f) == 0 (section 5.2.1).
func computeMul32(u uint32, phi uint64) (intPart uint32, isInt bool) {
	r := umul96Upper64(u, phi)
	intPart = uint32(r >> 32)
	isInt = uint32(r) == 0
	return
}

// computeMul64 computes only the parity of x^(i), y^(i), z^(i)
// from the precomputed value of φ̃k for float64
// and also checks if x^(f), y^(f), z^(f) = 0 (section 5.2.1).
func computeMulParity64(mant2 uint64, phi uint128, beta int) (parity bool, isInt bool) {
	r := umul192Lower128(mant2, phi)
	parity = ((r.hi >> (64 - beta)) & 1) != 0
	isInt = ((uint64(r.hi << beta)) | (r.lo >> (64 - beta))) == 0
	return
}

// computeMul64 computes only the parity of x^(i), y^(i), z^(i)
// from the precomputed value of φ̃k for float32
// and also checks if x^(f), y^(f), z^(f) = 0 (section 5.2.1).
func computeMulParity32(mant2 uint32, phi uint64, beta int) (parity bool, isInt bool) {
	r := umul96Lower64(mant2, phi)
	parity = ((r >> (64 - beta)) & 1) != 0
	isInt = uint32(r>>(32-beta)) == 0
	return
}

// computeDelta64 computes δ^(i) from the precomputed value of φ̃k for float64.
func computeDelta64(phi uint128, beta int) uint32 {
	return uint32(phi.hi >> (cacheBits64/2 - 1 - beta))
}

// computeDelta64 computes δ^(i) from the precomputed value of φ̃k for float32.
func computeDelta32(phi uint64, beta int) uint32 {
	return uint32(phi >> (cacheBits32 - 1 - beta))
}

// floorLog10Pow2 computes ⌊log10(2^e)⌋ = ⌊e*log10(2)⌋ (section 6.1).
func floorLog10Pow2(e int) int {
	// e should be in the range [-2620, 2620].
	return (e * 315653) >> 20
}

// floorLog2Pow10 computes ⌊log2(10^e)⌋ = ⌊e*log2(10)⌋ (section 6.2).
func floorLog2Pow10(e int) int {
	// e should be in the range [-1233, 1233].
	// The formula itself holds on [-4003, 4003],
	// but restricted to avoid overflow.
	return (e * 1741647) >> 19
}

// floorLog10Pow2MinusLog10_4Over3 computes
// ⌊e*log10(2)-log10(4/3)⌋ = ⌊log10(2^e)-log10(4/3)⌋ (section 6.3).
func floorLog10Pow2MinusLog10_4Over3(e int) int {
	// e should be in the range [-2985, 2936].
	return (e*631305 - 261663) >> 21
}

const (
	cacheBits64 = 128 // Q = 2*q = 128 for float64.
	cacheBits32 = 64  // Q = 2*q = 64 for float32.
	mantBits64  = 52  // p = 52 for float64.
	mantBits32  = 23  // p = 23 for flaot32.
)

// computeLeftEndpoint64 computes integer part of the left endpoint x.
func computeLeftEndpoint64(phi uint128, beta int) uint64 {
	return (phi.hi - (phi.hi >> (mantBits64 + 2))) >>
		(cacheBits64/2 - mantBits64 - 1 - beta)
}

// computeLeftEndpoint32 computes integer part of the left endpoint x.
func computeLeftEndpoint32(phi uint64, beta int) uint32 {
	return uint32((phi - (phi >> (mantBits32 + 2))) >>
		(cacheBits32 - mantBits32 - 1 - beta))
}

// computeRightEndpoint64 computes integer part of the right endpoint z.
func computeRightEndpoint64(phi uint128, beta int) uint64 {
	return (phi.hi + (phi.hi >> (mantBits64 + 1))) >>
		(cacheBits64/2 - mantBits64 - 1 - beta)
}

// computeRightEndpoint32 computes integer part of the right endpoint z.
func computeRightEndpoint32(phi uint64, beta int) uint32 {
	return uint32((phi + (phi >> (mantBits32 + 1))) >>
		(cacheBits32 - mantBits32 - 1 - beta))
}

// computeRoundUp64 computes the round up of y (i.e., y^(ru)).
func computeRoundUp64(phi uint128, beta int) uint64 {
	return (phi.hi>>(cacheBits64/2-mantBits64-2-beta) + 1) / 2
}

// computeRoundUp32 computes the round up of y (i.e., y^(ru)).
func computeRoundUp32(phi uint64, beta int) uint32 {
	return uint32(phi>>(cacheBits32-mantBits32-2-beta)+1) / 2
}

// removeTrailingZeros64 removes trailing zeros in decimal digits.
// There are at most 15 trailing zeros for float64 (page 16).
func removeTrailingZeros64(mant uint64, exp int) (uint64, int) {
	r := bits.RotateLeft64(mant*28999941890838049, -8)
	b := r < 184467440738
	s := 0
	if b { // TODO: Make this branchless if necessary.
		s++
		mant = r
	}

	r = bits.RotateLeft64(mant*182622766329724561, -4)
	b = r < 1844674407370956
	s = s * 2
	if b {
		s++
		mant = r
	}

	r = bits.RotateLeft64(mant*10330176681277348905, -2)
	b = r < 184467440737095517
	s = s * 2
	if b {
		s++
		mant = r
	}

	r = bits.RotateLeft64(mant*14757395258967641293, -1)
	b = r < 1844674407370955162
	s = s * 2
	if b {
		s++
		mant = r
	}

	exp += s
	return mant, exp
}

// removeTrailingZeros32 removes trailing zeros in decimal digits.
// There are at most 7 trailing zeros for float32 (page 16).
func removeTrailingZeros32(mant uint32, exp int) (uint32, int) {
	r := bits.RotateLeft32(mant*184254097, -4)
	b := r < 429497
	s := 0
	if b { // TODO: Make this branchless if necessary.
		s++
		mant = r
	}

	r = bits.RotateLeft32(mant*42949673, -2)
	b = r < 42949673
	s = s * 2
	if b {
		s++
		mant = r
	}

	r = bits.RotateLeft32(mant*1288490189, -1)
	b = r < 429496730
	s = s * 2
	if b {
		s++
		mant = r
	}

	exp += s
	return mant, exp
}

const (
	cacheMinK64 = -292 // k ∈ [-292, 326] for float64 (section 6.2).
	cacheMinK32 = -31  // k ∈ [-31, 46] for float32 (section 6.2).
)

// getCache64 gets the precomputed value of φ̃̃k for float64.
func getCache64(k int) uint128 {
	return cache64[k-cacheMinK64]
}

// getCache32 gets the precomputed value of φ̃̃k for float32.
func getCache32(k int) uint64 {
	return cache32[k-cacheMinK32]
}

// The precomputed table of φ̃̃k for float64.
// Note that φ̃̃k = ⌈φk⌉ and φk = 10^k*2^(-e_k),
// where e_k is the unique integer satisfying 2^(128-1) ≤ φk < 2^(128).
//
// φ̃̃k is chosen to satisfy ⌊n*2^(e-1)*10^k⌋ = ⌊2^β*n*φ̃̃k/2^128⌋
// such that expressions of the form ⌊n*2^(e-1)*10^k⌋
// (e.g., x^(i), y^(i), z^(i) and δ^(i))
// can be computed efficiently using bit shifts and multiplications only.
var cache64 = [619]uint128{
	{0xff77b1fcbebcdc4f, 0x25e8e89c13bb0f7b},
	{0x9faacf3df73609b1, 0x77b191618c54e9ad},
	{0xc795830d75038c1d, 0xd59df5b9ef6a2418},
	{0xf97ae3d0d2446f25, 0x4b0573286b44ad1e},
	{0x9becce62836ac577, 0x4ee367f9430aec33},
	{0xc2e801fb244576d5, 0x229c41f793cda740},
	{0xf3a20279ed56d48a, 0x6b43527578c11110},
	{0x9845418c345644d6, 0x830a13896b78aaaa},
	{0xbe5691ef416bd60c, 0x23cc986bc656d554},
	{0xedec366b11c6cb8f, 0x2cbfbe86b7ec8aa9},
	{0x94b3a202eb1c3f39, 0x7bf7d71432f3d6aa},
	{0xb9e08a83a5e34f07, 0xdaf5ccd93fb0cc54},
	{0xe858ad248f5c22c9, 0xd1b3400f8f9cff69},
	{0x91376c36d99995be, 0x23100809b9c21fa2},
	{0xb58547448ffffb2d, 0xabd40a0c2832a78b},
	{0xe2e69915b3fff9f9, 0x16c90c8f323f516d},
	{0x8dd01fad907ffc3b, 0xae3da7d97f6792e4},
	{0xb1442798f49ffb4a, 0x99cd11cfdf41779d},
	{0xdd95317f31c7fa1d, 0x40405643d711d584},
	{0x8a7d3eef7f1cfc52, 0x482835ea666b2573},
	{0xad1c8eab5ee43b66, 0xda3243650005eed0},
	{0xd863b256369d4a40, 0x90bed43e40076a83},
	{0x873e4f75e2224e68, 0x5a7744a6e804a292},
	{0xa90de3535aaae202, 0x711515d0a205cb37},
	{0xd3515c2831559a83, 0x0d5a5b44ca873e04},
	{0x8412d9991ed58091, 0xe858790afe9486c3},
	{0xa5178fff668ae0b6, 0x626e974dbe39a873},
	{0xce5d73ff402d98e3, 0xfb0a3d212dc81290},
	{0x80fa687f881c7f8e, 0x7ce66634bc9d0b9a},
	{0xa139029f6a239f72, 0x1c1fffc1ebc44e81},
	{0xc987434744ac874e, 0xa327ffb266b56221},
	{0xfbe9141915d7a922, 0x4bf1ff9f0062baa9},
	{0x9d71ac8fada6c9b5, 0x6f773fc3603db4aa},
	{0xc4ce17b399107c22, 0xcb550fb4384d21d4},
	{0xf6019da07f549b2b, 0x7e2a53a146606a49},
	{0x99c102844f94e0fb, 0x2eda7444cbfc426e},
	{0xc0314325637a1939, 0xfa911155fefb5309},
	{0xf03d93eebc589f88, 0x793555ab7eba27cb},
	{0x96267c7535b763b5, 0x4bc1558b2f3458df},
	{0xbbb01b9283253ca2, 0x9eb1aaedfb016f17},
	{0xea9c227723ee8bcb, 0x465e15a979c1cadd},
	{0x92a1958a7675175f, 0x0bfacd89ec191eca},
	{0xb749faed14125d36, 0xcef980ec671f667c},
	{0xe51c79a85916f484, 0x82b7e12780e7401b},
	{0x8f31cc0937ae58d2, 0xd1b2ecb8b0908811},
	{0xb2fe3f0b8599ef07, 0x861fa7e6dcb4aa16},
	{0xdfbdcece67006ac9, 0x67a791e093e1d49b},
	{0x8bd6a141006042bd, 0xe0c8bb2c5c6d24e1},
	{0xaecc49914078536d, 0x58fae9f773886e19},
	{0xda7f5bf590966848, 0xaf39a475506a899f},
	{0x888f99797a5e012d, 0x6d8406c952429604},
	{0xaab37fd7d8f58178, 0xc8e5087ba6d33b84},
	{0xd5605fcdcf32e1d6, 0xfb1e4a9a90880a65},
	{0x855c3be0a17fcd26, 0x5cf2eea09a550680},
	{0xa6b34ad8c9dfc06f, 0xf42faa48c0ea481f},
	{0xd0601d8efc57b08b, 0xf13b94daf124da27},
	{0x823c12795db6ce57, 0x76c53d08d6b70859},
	{0xa2cb1717b52481ed, 0x54768c4b0c64ca6f},
	{0xcb7ddcdda26da268, 0xa9942f5dcf7dfd0a},
	{0xfe5d54150b090b02, 0xd3f93b35435d7c4d},
	{0x9efa548d26e5a6e1, 0xc47bc5014a1a6db0},
	{0xc6b8e9b0709f109a, 0x359ab6419ca1091c},
	{0xf867241c8cc6d4c0, 0xc30163d203c94b63},
	{0x9b407691d7fc44f8, 0x79e0de63425dcf1e},
	{0xc21094364dfb5636, 0x985915fc12f542e5},
	{0xf294b943e17a2bc4, 0x3e6f5b7b17b2939e},
	{0x979cf3ca6cec5b5a, 0xa705992ceecf9c43},
	{0xbd8430bd08277231, 0x50c6ff782a838354},
	{0xece53cec4a314ebd, 0xa4f8bf5635246429},
	{0x940f4613ae5ed136, 0x871b7795e136be9a},
	{0xb913179899f68584, 0x28e2557b59846e40},
	{0xe757dd7ec07426e5, 0x331aeada2fe589d0},
	{0x9096ea6f3848984f, 0x3ff0d2c85def7622},
	{0xb4bca50b065abe63, 0x0fed077a756b53aa},
	{0xe1ebce4dc7f16dfb, 0xd3e8495912c62895},
	{0x8d3360f09cf6e4bd, 0x64712dd7abbbd95d},
	{0xb080392cc4349dec, 0xbd8d794d96aacfb4},
	{0xdca04777f541c567, 0xecf0d7a0fc5583a1},
	{0x89e42caaf9491b60, 0xf41686c49db57245},
	{0xac5d37d5b79b6239, 0x311c2875c522ced6},
	{0xd77485cb25823ac7, 0x7d633293366b828c},
	{0x86a8d39ef77164bc, 0xae5dff9c02033198},
	{0xa8530886b54dbdeb, 0xd9f57f830283fdfd},
	{0xd267caa862a12d66, 0xd072df63c324fd7c},
	{0x8380dea93da4bc60, 0x4247cb9e59f71e6e},
	{0xa46116538d0deb78, 0x52d9be85f074e609},
	{0xcd795be870516656, 0x67902e276c921f8c},
	{0x806bd9714632dff6, 0x00ba1cd8a3db53b7},
	{0xa086cfcd97bf97f3, 0x80e8a40eccd228a5},
	{0xc8a883c0fdaf7df0, 0x6122cd128006b2ce},
	{0xfad2a4b13d1b5d6c, 0x796b805720085f82},
	{0x9cc3a6eec6311a63, 0xcbe3303674053bb1},
	{0xc3f490aa77bd60fc, 0xbedbfc4411068a9d},
	{0xf4f1b4d515acb93b, 0xee92fb5515482d45},
	{0x991711052d8bf3c5, 0x751bdd152d4d1c4b},
	{0xbf5cd54678eef0b6, 0xd262d45a78a0635e},
	{0xef340a98172aace4, 0x86fb897116c87c35},
	{0x9580869f0e7aac0e, 0xd45d35e6ae3d4da1},
	{0xbae0a846d2195712, 0x8974836059cca10a},
	{0xe998d258869facd7, 0x2bd1a438703fc94c},
	{0x91ff83775423cc06, 0x7b6306a34627ddd0},
	{0xb67f6455292cbf08, 0x1a3bc84c17b1d543},
	{0xe41f3d6a7377eeca, 0x20caba5f1d9e4a94},
	{0x8e938662882af53e, 0x547eb47b7282ee9d},
	{0xb23867fb2a35b28d, 0xe99e619a4f23aa44},
	{0xdec681f9f4c31f31, 0x6405fa00e2ec94d5},
	{0x8b3c113c38f9f37e, 0xde83bc408dd3dd05},
	{0xae0b158b4738705e, 0x9624ab50b148d446},
	{0xd98ddaee19068c76, 0x3badd624dd9b0958},
	{0x87f8a8d4cfa417c9, 0xe54ca5d70a80e5d7},
	{0xa9f6d30a038d1dbc, 0x5e9fcf4ccd211f4d},
	{0xd47487cc8470652b, 0x7647c32000696720},
	{0x84c8d4dfd2c63f3b, 0x29ecd9f40041e074},
	{0xa5fb0a17c777cf09, 0xf468107100525891},
	{0xcf79cc9db955c2cc, 0x7182148d4066eeb5},
	{0x81ac1fe293d599bf, 0xc6f14cd848405531},
	{0xa21727db38cb002f, 0xb8ada00e5a506a7d},
	{0xca9cf1d206fdc03b, 0xa6d90811f0e4851d},
	{0xfd442e4688bd304a, 0x908f4a166d1da664},
	{0x9e4a9cec15763e2e, 0x9a598e4e043287ff},
	{0xc5dd44271ad3cdba, 0x40eff1e1853f29fe},
	{0xf7549530e188c128, 0xd12bee59e68ef47d},
	{0x9a94dd3e8cf578b9, 0x82bb74f8301958cf},
	{0xc13a148e3032d6e7, 0xe36a52363c1faf02},
	{0xf18899b1bc3f8ca1, 0xdc44e6c3cb279ac2},
	{0x96f5600f15a7b7e5, 0x29ab103a5ef8c0ba},
	{0xbcb2b812db11a5de, 0x7415d448f6b6f0e8},
	{0xebdf661791d60f56, 0x111b495b3464ad22},
	{0x936b9fcebb25c995, 0xcab10dd900beec35},
	{0xb84687c269ef3bfb, 0x3d5d514f40eea743},
	{0xe65829b3046b0afa, 0x0cb4a5a3112a5113},
	{0x8ff71a0fe2c2e6dc, 0x47f0e785eaba72ac},
	{0xb3f4e093db73a093, 0x59ed216765690f57},
	{0xe0f218b8d25088b8, 0x306869c13ec3532d},
	{0x8c974f7383725573, 0x1e414218c73a13fc},
	{0xafbd2350644eeacf, 0xe5d1929ef90898fb},
	{0xdbac6c247d62a583, 0xdf45f746b74abf3a},
	{0x894bc396ce5da772, 0x6b8bba8c328eb784},
	{0xab9eb47c81f5114f, 0x066ea92f3f326565},
	{0xd686619ba27255a2, 0xc80a537b0efefebe},
	{0x8613fd0145877585, 0xbd06742ce95f5f37},
	{0xa798fc4196e952e7, 0x2c48113823b73705},
	{0xd17f3b51fca3a7a0, 0xf75a15862ca504c6},
	{0x82ef85133de648c4, 0x9a984d73dbe722fc},
	{0xa3ab66580d5fdaf5, 0xc13e60d0d2e0ebbb},
	{0xcc963fee10b7d1b3, 0x318df905079926a9},
	{0xffbbcfe994e5c61f, 0xfdf17746497f7053},
	{0x9fd561f1fd0f9bd3, 0xfeb6ea8bedefa634},
	{0xc7caba6e7c5382c8, 0xfe64a52ee96b8fc1},
	{0xf9bd690a1b68637b, 0x3dfdce7aa3c673b1},
	{0x9c1661a651213e2d, 0x06bea10ca65c084f},
	{0xc31bfa0fe5698db8, 0x486e494fcff30a63},
	{0xf3e2f893dec3f126, 0x5a89dba3c3efccfb},
	{0x986ddb5c6b3a76b7, 0xf89629465a75e01d},
	{0xbe89523386091465, 0xf6bbb397f1135824},
	{0xee2ba6c0678b597f, 0x746aa07ded582e2d},
	{0x94db483840b717ef, 0xa8c2a44eb4571cdd},
	{0xba121a4650e4ddeb, 0x92f34d62616ce414},
	{0xe896a0d7e51e1566, 0x77b020baf9c81d18},
	{0x915e2486ef32cd60, 0x0ace1474dc1d122f},
	{0xb5b5ada8aaff80b8, 0x0d819992132456bb},
	{0xe3231912d5bf60e6, 0x10e1fff697ed6c6a},
	{0x8df5efabc5979c8f, 0xca8d3ffa1ef463c2},
	{0xb1736b96b6fd83b3, 0xbd308ff8a6b17cb3},
	{0xddd0467c64bce4a0, 0xac7cb3f6d05ddbdf},
	{0x8aa22c0dbef60ee4, 0x6bcdf07a423aa96c},
	{0xad4ab7112eb3929d, 0x86c16c98d2c953c7},
	{0xd89d64d57a607744, 0xe871c7bf077ba8b8},
	{0x87625f056c7c4a8b, 0x11471cd764ad4973},
	{0xa93af6c6c79b5d2d, 0xd598e40d3dd89bd0},
	{0xd389b47879823479, 0x4aff1d108d4ec2c4},
	{0x843610cb4bf160cb, 0xcedf722a585139bb},
	{0xa54394fe1eedb8fe, 0xc2974eb4ee658829},
	{0xce947a3da6a9273e, 0x733d226229feea33},
	{0x811ccc668829b887, 0x0806357d5a3f5260},
	{0xa163ff802a3426a8, 0xca07c2dcb0cf26f8},
	{0xc9bcff6034c13052, 0xfc89b393dd02f0b6},
	{0xfc2c3f3841f17c67, 0xbbac2078d443ace3},
	{0x9d9ba7832936edc0, 0xd54b944b84aa4c0e},
	{0xc5029163f384a931, 0x0a9e795e65d4df12},
	{0xf64335bcf065d37d, 0x4d4617b5ff4a16d6},
	{0x99ea0196163fa42e, 0x504bced1bf8e4e46},
	{0xc06481fb9bcf8d39, 0xe45ec2862f71e1d7},
	{0xf07da27a82c37088, 0x5d767327bb4e5a4d},
	{0x964e858c91ba2655, 0x3a6a07f8d510f870},
	{0xbbe226efb628afea, 0x890489f70a55368c},
	{0xeadab0aba3b2dbe5, 0x2b45ac74ccea842f},
	{0x92c8ae6b464fc96f, 0x3b0b8bc90012929e},
	{0xb77ada0617e3bbcb, 0x09ce6ebb40173745},
	{0xe55990879ddcaabd, 0xcc420a6a101d0516},
	{0x8f57fa54c2a9eab6, 0x9fa946824a12232e},
	{0xb32df8e9f3546564, 0x47939822dc96abfa},
	{0xdff9772470297ebd, 0x59787e2b93bc56f8},
	{0x8bfbea76c619ef36, 0x57eb4edb3c55b65b},
	{0xaefae51477a06b03, 0xede622920b6b23f2},
	{0xdab99e59958885c4, 0xe95fab368e45ecee},
	{0x88b402f7fd75539b, 0x11dbcb0218ebb415},
	{0xaae103b5fcd2a881, 0xd652bdc29f26a11a},
	{0xd59944a37c0752a2, 0x4be76d3346f04960},
	{0x857fcae62d8493a5, 0x6f70a4400c562ddc},
	{0xa6dfbd9fb8e5b88e, 0xcb4ccd500f6bb953},
	{0xd097ad07a71f26b2, 0x7e2000a41346a7a8},
	{0x825ecc24c873782f, 0x8ed400668c0c28c9},
	{0xa2f67f2dfa90563b, 0x728900802f0f32fb},
	{0xcbb41ef979346bca, 0x4f2b40a03ad2ffba},
	{0xfea126b7d78186bc, 0xe2f610c84987bfa9},
	{0x9f24b832e6b0f436, 0x0dd9ca7d2df4d7ca},
	{0xc6ede63fa05d3143, 0x91503d1c79720dbc},
	{0xf8a95fcf88747d94, 0x75a44c6397ce912b},
	{0x9b69dbe1b548ce7c, 0xc986afbe3ee11abb},
	{0xc24452da229b021b, 0xfbe85badce996169},
	{0xf2d56790ab41c2a2, 0xfae27299423fb9c4},
	{0x97c560ba6b0919a5, 0xdccd879fc967d41b},
	{0xbdb6b8e905cb600f, 0x5400e987bbc1c921},
	{0xed246723473e3813, 0x290123e9aab23b69},
	{0x9436c0760c86e30b, 0xf9a0b6720aaf6522},
	{0xb94470938fa89bce, 0xf808e40e8d5b3e6a},
	{0xe7958cb87392c2c2, 0xb60b1d1230b20e05},
	{0x90bd77f3483bb9b9, 0xb1c6f22b5e6f48c3},
	{0xb4ecd5f01a4aa828, 0x1e38aeb6360b1af4},
	{0xe2280b6c20dd5232, 0x25c6da63c38de1b1},
	{0x8d590723948a535f, 0x579c487e5a38ad0f},
	{0xb0af48ec79ace837, 0x2d835a9df0c6d852},
	{0xdcdb1b2798182244, 0xf8e431456cf88e66},
	{0x8a08f0f8bf0f156b, 0x1b8e9ecb641b5900},
	{0xac8b2d36eed2dac5, 0xe272467e3d222f40},
	{0xd7adf884aa879177, 0x5b0ed81dcc6abb10},
	{0x86ccbb52ea94baea, 0x98e947129fc2b4ea},
	{0xa87fea27a539e9a5, 0x3f2398d747b36225},
	{0xd29fe4b18e88640e, 0x8eec7f0d19a03aae},
	{0x83a3eeeef9153e89, 0x1953cf68300424ad},
	{0xa48ceaaab75a8e2b, 0x5fa8c3423c052dd8},
	{0xcdb02555653131b6, 0x3792f412cb06794e},
	{0x808e17555f3ebf11, 0xe2bbd88bbee40bd1},
	{0xa0b19d2ab70e6ed6, 0x5b6aceaeae9d0ec5},
	{0xc8de047564d20a8b, 0xf245825a5a445276},
	{0xfb158592be068d2e, 0xeed6e2f0f0d56713},
	{0x9ced737bb6c4183d, 0x55464dd69685606c},
	{0xc428d05aa4751e4c, 0xaa97e14c3c26b887},
	{0xf53304714d9265df, 0xd53dd99f4b3066a9},
	{0x993fe2c6d07b7fab, 0xe546a8038efe402a},
	{0xbf8fdb78849a5f96, 0xde98520472bdd034},
	{0xef73d256a5c0f77c, 0x963e66858f6d4441},
	{0x95a8637627989aad, 0xdde7001379a44aa9},
	{0xbb127c53b17ec159, 0x5560c018580d5d53},
	{0xe9d71b689dde71af, 0xaab8f01e6e10b4a7},
	{0x9226712162ab070d, 0xcab3961304ca70e9},
	{0xb6b00d69bb55c8d1, 0x3d607b97c5fd0d23},
	{0xe45c10c42a2b3b05, 0x8cb89a7db77c506b},
	{0x8eb98a7a9a5b04e3, 0x77f3608e92adb243},
	{0xb267ed1940f1c61c, 0x55f038b237591ed4},
	{0xdf01e85f912e37a3, 0x6b6c46dec52f6689},
	{0x8b61313bbabce2c6, 0x2323ac4b3b3da016},
	{0xae397d8aa96c1b77, 0xabec975e0a0d081b},
	{0xd9c7dced53c72255, 0x96e7bd358c904a22},
	{0x881cea14545c7575, 0x7e50d64177da2e55},
	{0xaa242499697392d2, 0xdde50bd1d5d0b9ea},
	{0xd4ad2dbfc3d07787, 0x955e4ec64b44e865},
	{0x84ec3c97da624ab4, 0xbd5af13bef0b113f},
	{0xa6274bbdd0fadd61, 0xecb1ad8aeacdd58f},
	{0xcfb11ead453994ba, 0x67de18eda5814af3},
	{0x81ceb32c4b43fcf4, 0x80eacf948770ced8},
	{0xa2425ff75e14fc31, 0xa1258379a94d028e},
	{0xcad2f7f5359a3b3e, 0x096ee45813a04331},
	{0xfd87b5f28300ca0d, 0x8bca9d6e188853fd},
	{0x9e74d1b791e07e48, 0x775ea264cf55347e},
	{0xc612062576589dda, 0x95364afe032a819e},
	{0xf79687aed3eec551, 0x3a83ddbd83f52205},
	{0x9abe14cd44753b52, 0xc4926a9672793543},
	{0xc16d9a0095928a27, 0x75b7053c0f178294},
	{0xf1c90080baf72cb1, 0x5324c68b12dd6339},
	{0x971da05074da7bee, 0xd3f6fc16ebca5e04},
	{0xbce5086492111aea, 0x88f4bb1ca6bcf585},
	{0xec1e4a7db69561a5, 0x2b31e9e3d06c32e6},
	{0x9392ee8e921d5d07, 0x3aff322e62439fd0},
	{0xb877aa3236a4b449, 0x09befeb9fad487c3},
	{0xe69594bec44de15b, 0x4c2ebe687989a9b4},
	{0x901d7cf73ab0acd9, 0x0f9d37014bf60a11},
	{0xb424dc35095cd80f, 0x538484c19ef38c95},
	{0xe12e13424bb40e13, 0x2865a5f206b06fba},
	{0x8cbccc096f5088cb, 0xf93f87b7442e45d4},
	{0xafebff0bcb24aafe, 0xf78f69a51539d749},
	{0xdbe6fecebdedd5be, 0xb573440e5a884d1c},
	{0x89705f4136b4a597, 0x31680a88f8953031},
	{0xabcc77118461cefc, 0xfdc20d2b36ba7c3e},
	{0xd6bf94d5e57a42bc, 0x3d32907604691b4d},
	{0x8637bd05af6c69b5, 0xa63f9a49c2c1b110},
	{0xa7c5ac471b478423, 0x0fcf80dc33721d54},
	{0xd1b71758e219652b, 0xd3c36113404ea4a9},
	{0x83126e978d4fdf3b, 0x645a1cac083126ea},
	{0xa3d70a3d70a3d70a, 0x3d70a3d70a3d70a4},
	{0xcccccccccccccccc, 0xcccccccccccccccd},
	{0x8000000000000000, 0x0000000000000000},
	{0xa000000000000000, 0x0000000000000000},
	{0xc800000000000000, 0x0000000000000000},
	{0xfa00000000000000, 0x0000000000000000},
	{0x9c40000000000000, 0x0000000000000000},
	{0xc350000000000000, 0x0000000000000000},
	{0xf424000000000000, 0x0000000000000000},
	{0x9896800000000000, 0x0000000000000000},
	{0xbebc200000000000, 0x0000000000000000},
	{0xee6b280000000000, 0x0000000000000000},
	{0x9502f90000000000, 0x0000000000000000},
	{0xba43b74000000000, 0x0000000000000000},
	{0xe8d4a51000000000, 0x0000000000000000},
	{0x9184e72a00000000, 0x0000000000000000},
	{0xb5e620f480000000, 0x0000000000000000},
	{0xe35fa931a0000000, 0x0000000000000000},
	{0x8e1bc9bf04000000, 0x0000000000000000},
	{0xb1a2bc2ec5000000, 0x0000000000000000},
	{0xde0b6b3a76400000, 0x0000000000000000},
	{0x8ac7230489e80000, 0x0000000000000000},
	{0xad78ebc5ac620000, 0x0000000000000000},
	{0xd8d726b7177a8000, 0x0000000000000000},
	{0x878678326eac9000, 0x0000000000000000},
	{0xa968163f0a57b400, 0x0000000000000000},
	{0xd3c21bcecceda100, 0x0000000000000000},
	{0x84595161401484a0, 0x0000000000000000},
	{0xa56fa5b99019a5c8, 0x0000000000000000},
	{0xcecb8f27f4200f3a, 0x0000000000000000},
	{0x813f3978f8940984, 0x4000000000000000},
	{0xa18f07d736b90be5, 0x5000000000000000},
	{0xc9f2c9cd04674ede, 0xa400000000000000},
	{0xfc6f7c4045812296, 0x4d00000000000000},
	{0x9dc5ada82b70b59d, 0xf020000000000000},
	{0xc5371912364ce305, 0x6c28000000000000},
	{0xf684df56c3e01bc6, 0xc732000000000000},
	{0x9a130b963a6c115c, 0x3c7f400000000000},
	{0xc097ce7bc90715b3, 0x4b9f100000000000},
	{0xf0bdc21abb48db20, 0x1e86d40000000000},
	{0x96769950b50d88f4, 0x1314448000000000},
	{0xbc143fa4e250eb31, 0x17d955a000000000},
	{0xeb194f8e1ae525fd, 0x5dcfab0800000000},
	{0x92efd1b8d0cf37be, 0x5aa1cae500000000},
	{0xb7abc627050305ad, 0xf14a3d9e40000000},
	{0xe596b7b0c643c719, 0x6d9ccd05d0000000},
	{0x8f7e32ce7bea5c6f, 0xe4820023a2000000},
	{0xb35dbf821ae4f38b, 0xdda2802c8a800000},
	{0xe0352f62a19e306e, 0xd50b2037ad200000},
	{0x8c213d9da502de45, 0x4526f422cc340000},
	{0xaf298d050e4395d6, 0x9670b12b7f410000},
	{0xdaf3f04651d47b4c, 0x3c0cdd765f114000},
	{0x88d8762bf324cd0f, 0xa5880a69fb6ac800},
	{0xab0e93b6efee0053, 0x8eea0d047a457a00},
	{0xd5d238a4abe98068, 0x72a4904598d6d880},
	{0x85a36366eb71f041, 0x47a6da2b7f864750},
	{0xa70c3c40a64e6c51, 0x999090b65f67d924},
	{0xd0cf4b50cfe20765, 0xfff4b4e3f741cf6d},
	{0x82818f1281ed449f, 0xbff8f10e7a8921a5},
	{0xa321f2d7226895c7, 0xaff72d52192b6a0e},
	{0xcbea6f8ceb02bb39, 0x9bf4f8a69f764491},
	{0xfee50b7025c36a08, 0x02f236d04753d5b5},
	{0x9f4f2726179a2245, 0x01d762422c946591},
	{0xc722f0ef9d80aad6, 0x424d3ad2b7b97ef6},
	{0xf8ebad2b84e0d58b, 0xd2e0898765a7deb3},
	{0x9b934c3b330c8577, 0x63cc55f49f88eb30},
	{0xc2781f49ffcfa6d5, 0x3cbf6b71c76b25fc},
	{0xf316271c7fc3908a, 0x8bef464e3945ef7b},
	{0x97edd871cfda3a56, 0x97758bf0e3cbb5ad},
	{0xbde94e8e43d0c8ec, 0x3d52eeed1cbea318},
	{0xed63a231d4c4fb27, 0x4ca7aaa863ee4bde},
	{0x945e455f24fb1cf8, 0x8fe8caa93e74ef6b},
	{0xb975d6b6ee39e436, 0xb3e2fd538e122b45},
	{0xe7d34c64a9c85d44, 0x60dbbca87196b617},
	{0x90e40fbeea1d3a4a, 0xbc8955e946fe31ce},
	{0xb51d13aea4a488dd, 0x6babab6398bdbe42},
	{0xe264589a4dcdab14, 0xc696963c7eed2dd2},
	{0x8d7eb76070a08aec, 0xfc1e1de5cf543ca3},
	{0xb0de65388cc8ada8, 0x3b25a55f43294bcc},
	{0xdd15fe86affad912, 0x49ef0eb713f39ebf},
	{0x8a2dbf142dfcc7ab, 0x6e3569326c784338},
	{0xacb92ed9397bf996, 0x49c2c37f07965405},
	{0xd7e77a8f87daf7fb, 0xdc33745ec97be907},
	{0x86f0ac99b4e8dafd, 0x69a028bb3ded71a4},
	{0xa8acd7c0222311bc, 0xc40832ea0d68ce0d},
	{0xd2d80db02aabd62b, 0xf50a3fa490c30191},
	{0x83c7088e1aab65db, 0x792667c6da79e0fb},
	{0xa4b8cab1a1563f52, 0x577001b891185939},
	{0xcde6fd5e09abcf26, 0xed4c0226b55e6f87},
	{0x80b05e5ac60b6178, 0x544f8158315b05b5},
	{0xa0dc75f1778e39d6, 0x696361ae3db1c722},
	{0xc913936dd571c84c, 0x03bc3a19cd1e38ea},
	{0xfb5878494ace3a5f, 0x04ab48a04065c724},
	{0x9d174b2dcec0e47b, 0x62eb0d64283f9c77},
	{0xc45d1df942711d9a, 0x3ba5d0bd324f8395},
	{0xf5746577930d6500, 0xca8f44ec7ee3647a},
	{0x9968bf6abbe85f20, 0x7e998b13cf4e1ecc},
	{0xbfc2ef456ae276e8, 0x9e3fedd8c321a67f},
	{0xefb3ab16c59b14a2, 0xc5cfe94ef3ea101f},
	{0x95d04aee3b80ece5, 0xbba1f1d158724a13},
	{0xbb445da9ca61281f, 0x2a8a6e45ae8edc98},
	{0xea1575143cf97226, 0xf52d09d71a3293be},
	{0x924d692ca61be758, 0x593c2626705f9c57},
	{0xb6e0c377cfa2e12e, 0x6f8b2fb00c77836d},
	{0xe498f455c38b997a, 0x0b6dfb9c0f956448},
	{0x8edf98b59a373fec, 0x4724bd4189bd5ead},
	{0xb2977ee300c50fe7, 0x58edec91ec2cb658},
	{0xdf3d5e9bc0f653e1, 0x2f2967b66737e3ee},
	{0x8b865b215899f46c, 0xbd79e0d20082ee75},
	{0xae67f1e9aec07187, 0xecd8590680a3aa12},
	{0xda01ee641a708de9, 0xe80e6f4820cc9496},
	{0x884134fe908658b2, 0x3109058d147fdcde},
	{0xaa51823e34a7eede, 0xbd4b46f0599fd416},
	{0xd4e5e2cdc1d1ea96, 0x6c9e18ac7007c91b},
	{0x850fadc09923329e, 0x03e2cf6bc604ddb1},
	{0xa6539930bf6bff45, 0x84db8346b786151d},
	{0xcfe87f7cef46ff16, 0xe612641865679a64},
	{0x81f14fae158c5f6e, 0x4fcb7e8f3f60c07f},
	{0xa26da3999aef7749, 0xe3be5e330f38f09e},
	{0xcb090c8001ab551c, 0x5cadf5bfd3072cc6},
	{0xfdcb4fa002162a63, 0x73d9732fc7c8f7f7},
	{0x9e9f11c4014dda7e, 0x2867e7fddcdd9afb},
	{0xc646d63501a1511d, 0xb281e1fd541501b9},
	{0xf7d88bc24209a565, 0x1f225a7ca91a4227},
	{0x9ae757596946075f, 0x3375788de9b06959},
	{0xc1a12d2fc3978937, 0x0052d6b1641c83af},
	{0xf209787bb47d6b84, 0xc0678c5dbd23a49b},
	{0x9745eb4d50ce6332, 0xf840b7ba963646e1},
	{0xbd176620a501fbff, 0xb650e5a93bc3d899},
	{0xec5d3fa8ce427aff, 0xa3e51f138ab4cebf},
	{0x93ba47c980e98cdf, 0xc66f336c36b10138},
	{0xb8a8d9bbe123f017, 0xb80b0047445d4185},
	{0xe6d3102ad96cec1d, 0xa60dc059157491e6},
	{0x9043ea1ac7e41392, 0x87c89837ad68db30},
	{0xb454e4a179dd1877, 0x29babe4598c311fc},
	{0xe16a1dc9d8545e94, 0xf4296dd6fef3d67b},
	{0x8ce2529e2734bb1d, 0x1899e4a65f58660d},
	{0xb01ae745b101e9e4, 0x5ec05dcff72e7f90},
	{0xdc21a1171d42645d, 0x76707543f4fa1f74},
	{0x899504ae72497eba, 0x6a06494a791c53a9},
	{0xabfa45da0edbde69, 0x0487db9d17636893},
	{0xd6f8d7509292d603, 0x45a9d2845d3c42b7},
	{0x865b86925b9bc5c2, 0x0b8a2392ba45a9b3},
	{0xa7f26836f282b732, 0x8e6cac7768d7141f},
	{0xd1ef0244af2364ff, 0x3207d795430cd927},
	{0x8335616aed761f1f, 0x7f44e6bd49e807b9},
	{0xa402b9c5a8d3a6e7, 0x5f16206c9c6209a7},
	{0xcd036837130890a1, 0x36dba887c37a8c10},
	{0x802221226be55a64, 0xc2494954da2c978a},
	{0xa02aa96b06deb0fd, 0xf2db9baa10b7bd6d},
	{0xc83553c5c8965d3d, 0x6f92829494e5acc8},
	{0xfa42a8b73abbf48c, 0xcb772339ba1f17fa},
	{0x9c69a97284b578d7, 0xff2a760414536efc},
	{0xc38413cf25e2d70d, 0xfef5138519684abb},
	{0xf46518c2ef5b8cd1, 0x7eb258665fc25d6a},
	{0x98bf2f79d5993802, 0xef2f773ffbd97a62},
	{0xbeeefb584aff8603, 0xaafb550ffacfd8fb},
	{0xeeaaba2e5dbf6784, 0x95ba2a53f983cf39},
	{0x952ab45cfa97a0b2, 0xdd945a747bf26184},
	{0xba756174393d88df, 0x94f971119aeef9e5},
	{0xe912b9d1478ceb17, 0x7a37cd5601aab85e},
	{0x91abb422ccb812ee, 0xac62e055c10ab33b},
	{0xb616a12b7fe617aa, 0x577b986b314d600a},
	{0xe39c49765fdf9d94, 0xed5a7e85fda0b80c},
	{0x8e41ade9fbebc27d, 0x14588f13be847308},
	{0xb1d219647ae6b31c, 0x596eb2d8ae258fc9},
	{0xde469fbd99a05fe3, 0x6fca5f8ed9aef3bc},
	{0x8aec23d680043bee, 0x25de7bb9480d5855},
	{0xada72ccc20054ae9, 0xaf561aa79a10ae6b},
	{0xd910f7ff28069da4, 0x1b2ba1518094da05},
	{0x87aa9aff79042286, 0x90fb44d2f05d0843},
	{0xa99541bf57452b28, 0x353a1607ac744a54},
	{0xd3fa922f2d1675f2, 0x42889b8997915ce9},
	{0x847c9b5d7c2e09b7, 0x69956135febada12},
	{0xa59bc234db398c25, 0x43fab9837e699096},
	{0xcf02b2c21207ef2e, 0x94f967e45e03f4bc},
	{0x8161afb94b44f57d, 0x1d1be0eebac278f6},
	{0xa1ba1ba79e1632dc, 0x6462d92a69731733},
	{0xca28a291859bbf93, 0x7d7b8f7503cfdcff},
	{0xfcb2cb35e702af78, 0x5cda735244c3d43f},
	{0x9defbf01b061adab, 0x3a0888136afa64a8},
	{0xc56baec21c7a1916, 0x088aaa1845b8fdd1},
	{0xf6c69a72a3989f5b, 0x8aad549e57273d46},
	{0x9a3c2087a63f6399, 0x36ac54e2f678864c},
	{0xc0cb28a98fcf3c7f, 0x84576a1bb416a7de},
	{0xf0fdf2d3f3c30b9f, 0x656d44a2a11c51d6},
	{0x969eb7c47859e743, 0x9f644ae5a4b1b326},
	{0xbc4665b596706114, 0x873d5d9f0dde1fef},
	{0xeb57ff22fc0c7959, 0xa90cb506d155a7eb},
	{0x9316ff75dd87cbd8, 0x09a7f12442d588f3},
	{0xb7dcbf5354e9bece, 0x0c11ed6d538aeb30},
	{0xe5d3ef282a242e81, 0x8f1668c8a86da5fb},
	{0x8fa475791a569d10, 0xf96e017d694487bd},
	{0xb38d92d760ec4455, 0x37c981dcc395a9ad},
	{0xe070f78d3927556a, 0x85bbe253f47b1418},
	{0x8c469ab843b89562, 0x93956d7478ccec8f},
	{0xaf58416654a6babb, 0x387ac8d1970027b3},
	{0xdb2e51bfe9d0696a, 0x06997b05fcc0319f},
	{0x88fcf317f22241e2, 0x441fece3bdf81f04},
	{0xab3c2fddeeaad25a, 0xd527e81cad7626c4},
	{0xd60b3bd56a5586f1, 0x8a71e223d8d3b075},
	{0x85c7056562757456, 0xf6872d5667844e4a},
	{0xa738c6bebb12d16c, 0xb428f8ac016561dc},
	{0xd106f86e69d785c7, 0xe13336d701beba53},
	{0x82a45b450226b39c, 0xecc0024661173474},
	{0xa34d721642b06084, 0x27f002d7f95d0191},
	{0xcc20ce9bd35c78a5, 0x31ec038df7b441f5},
	{0xff290242c83396ce, 0x7e67047175a15272},
	{0x9f79a169bd203e41, 0x0f0062c6e984d387},
	{0xc75809c42c684dd1, 0x52c07b78a3e60869},
	{0xf92e0c3537826145, 0xa7709a56ccdf8a83},
	{0x9bbcc7a142b17ccb, 0x88a66076400bb692},
	{0xc2abf989935ddbfe, 0x6acff893d00ea436},
	{0xf356f7ebf83552fe, 0x0583f6b8c4124d44},
	{0x98165af37b2153de, 0xc3727a337a8b704b},
	{0xbe1bf1b059e9a8d6, 0x744f18c0592e4c5d},
	{0xeda2ee1c7064130c, 0x1162def06f79df74},
	{0x9485d4d1c63e8be7, 0x8addcb5645ac2ba9},
	{0xb9a74a0637ce2ee1, 0x6d953e2bd7173693},
	{0xe8111c87c5c1ba99, 0xc8fa8db6ccdd0438},
	{0x910ab1d4db9914a0, 0x1d9c9892400a22a3},
	{0xb54d5e4a127f59c8, 0x2503beb6d00cab4c},
	{0xe2a0b5dc971f303a, 0x2e44ae64840fd61e},
	{0x8da471a9de737e24, 0x5ceaecfed289e5d3},
	{0xb10d8e1456105dad, 0x7425a83e872c5f48},
	{0xdd50f1996b947518, 0xd12f124e28f7771a},
	{0x8a5296ffe33cc92f, 0x82bd6b70d99aaa70},
	{0xace73cbfdc0bfb7b, 0x636cc64d1001550c},
	{0xd8210befd30efa5a, 0x3c47f7e05401aa4f},
	{0x8714a775e3e95c78, 0x65acfaec34810a72},
	{0xa8d9d1535ce3b396, 0x7f1839a741a14d0e},
	{0xd31045a8341ca07c, 0x1ede48111209a051},
	{0x83ea2b892091e44d, 0x934aed0aab460433},
	{0xa4e4b66b68b65d60, 0xf81da84d56178540},
	{0xce1de40642e3f4b9, 0x36251260ab9d668f},
	{0x80d2ae83e9ce78f3, 0xc1d72b7c6b42601a},
	{0xa1075a24e4421730, 0xb24cf65b8612f820},
	{0xc94930ae1d529cfc, 0xdee033f26797b628},
	{0xfb9b7cd9a4a7443c, 0x169840ef017da3b2},
	{0x9d412e0806e88aa5, 0x8e1f289560ee864f},
	{0xc491798a08a2ad4e, 0xf1a6f2bab92a27e3},
	{0xf5b5d7ec8acb58a2, 0xae10af696774b1dc},
	{0x9991a6f3d6bf1765, 0xacca6da1e0a8ef2a},
	{0xbff610b0cc6edd3f, 0x17fd090a58d32af4},
	{0xeff394dcff8a948e, 0xddfc4b4cef07f5b1},
	{0x95f83d0a1fb69cd9, 0x4abdaf101564f98f},
	{0xbb764c4ca7a4440f, 0x9d6d1ad41abe37f2},
	{0xea53df5fd18d5513, 0x84c86189216dc5ee},
	{0x92746b9be2f8552c, 0x32fd3cf5b4e49bb5},
	{0xb7118682dbb66a77, 0x3fbc8c33221dc2a2},
	{0xe4d5e82392a40515, 0x0fabaf3feaa5334b},
	{0x8f05b1163ba6832d, 0x29cb4d87f2a7400f},
	{0xb2c71d5bca9023f8, 0x743e20e9ef511013},
	{0xdf78e4b2bd342cf6, 0x914da9246b255417},
	{0x8bab8eefb6409c1a, 0x1ad089b6c2f7548f},
	{0xae9672aba3d0c320, 0xa184ac2473b529b2},
	{0xda3c0f568cc4f3e8, 0xc9e5d72d90a2741f},
	{0x8865899617fb1871, 0x7e2fa67c7a658893},
	{0xaa7eebfb9df9de8d, 0xddbb901b98feeab8},
	{0xd51ea6fa85785631, 0x552a74227f3ea566},
	{0x8533285c936b35de, 0xd53a88958f872760},
	{0xa67ff273b8460356, 0x8a892abaf368f138},
	{0xd01fef10a657842c, 0x2d2b7569b0432d86},
	{0x8213f56a67f6b29b, 0x9c3b29620e29fc74},
	{0xa298f2c501f45f42, 0x8349f3ba91b47b90},
	{0xcb3f2f7642717713, 0x241c70a936219a74},
	{0xfe0efb53d30dd4d7, 0xed238cd383aa0111},
	{0x9ec95d1463e8a506, 0xf4363804324a40ab},
	{0xc67bb4597ce2ce48, 0xb143c6053edcd0d6},
	{0xf81aa16fdc1b81da, 0xdd94b7868e94050b},
	{0x9b10a4e5e9913128, 0xca7cf2b4191c8327},
	{0xc1d4ce1f63f57d72, 0xfd1c2f611f63a3f1},
	{0xf24a01a73cf2dccf, 0xbc633b39673c8ced},
	{0x976e41088617ca01, 0xd5be0503e085d814},
	{0xbd49d14aa79dbc82, 0x4b2d8644d8a74e19},
	{0xec9c459d51852ba2, 0xddf8e7d60ed1219f},
	{0x93e1ab8252f33b45, 0xcabb90e5c942b504},
	{0xb8da1662e7b00a17, 0x3d6a751f3b936244},
	{0xe7109bfba19c0c9d, 0x0cc512670a783ad5},
	{0x906a617d450187e2, 0x27fb2b80668b24c6},
	{0xb484f9dc9641e9da, 0xb1f9f660802dedf7},
	{0xe1a63853bbd26451, 0x5e7873f8a0396974},
	{0x8d07e33455637eb2, 0xdb0b487b6423e1e9},
	{0xb049dc016abc5e5f, 0x91ce1a9a3d2cda63},
	{0xdc5c5301c56b75f7, 0x7641a140cc7810fc},
	{0x89b9b3e11b6329ba, 0xa9e904c87fcb0a9e},
	{0xac2820d9623bf429, 0x546345fa9fbdcd45},
	{0xd732290fbacaf133, 0xa97c177947ad4096},
	{0x867f59a9d4bed6c0, 0x49ed8eabcccc485e},
	{0xa81f301449ee8c70, 0x5c68f256bfff5a75},
	{0xd226fc195c6a2f8c, 0x73832eec6fff3112},
	{0x83585d8fd9c25db7, 0xc831fd53c5ff7eac},
	{0xa42e74f3d032f525, 0xba3e7ca8b77f5e56},
	{0xcd3a1230c43fb26f, 0x28ce1bd2e55f35ec},
	{0x80444b5e7aa7cf85, 0x7980d163cf5b81b4},
	{0xa0555e361951c366, 0xd7e105bcc3326220},
	{0xc86ab5c39fa63440, 0x8dd9472bf3fefaa8},
	{0xfa856334878fc150, 0xb14f98f6f0feb952},
	{0x9c935e00d4b9d8d2, 0x6ed1bf9a569f33d4},
	{0xc3b8358109e84f07, 0x0a862f80ec4700c9},
	{0xf4a642e14c6262c8, 0xcd27bb612758c0fb},
	{0x98e7e9cccfbd7dbd, 0x8038d51cb897789d},
	{0xbf21e44003acdd2c, 0xe0470a63e6bd56c4},
	{0xeeea5d5004981478, 0x1858ccfce06cac75},
	{0x95527a5202df0ccb, 0x0f37801e0c43ebc9},
	{0xbaa718e68396cffd, 0xd30560258f54e6bb},
	{0xe950df20247c83fd, 0x47c6b82ef32a206a},
	{0x91d28b7416cdd27e, 0x4cdc331d57fa5442},
	{0xb6472e511c81471d, 0xe0133fe4adf8e953},
	{0xe3d8f9e563a198e5, 0x58180fddd97723a7},
	{0x8e679c2f5e44ff8f, 0x570f09eaa7ea7649},
	{0xb201833b35d63f73, 0x2cd2cc6551e513db},
	{0xde81e40a034bcf4f, 0xf8077f7ea65e58d2},
	{0x8b112e86420f6191, 0xfb04afaf27faf783},
	{0xadd57a27d29339f6, 0x79c5db9af1f9b564},
	{0xd94ad8b1c7380874, 0x18375281ae7822bd},
	{0x87cec76f1c830548, 0x8f2293910d0b15b6},
	{0xa9c2794ae3a3c69a, 0xb2eb3875504ddb23},
	{0xd433179d9c8cb841, 0x5fa60692a46151ec},
	{0x849feec281d7f328, 0xdbc7c41ba6bcd334},
	{0xa5c7ea73224deff3, 0x12b9b522906c0801},
	{0xcf39e50feae16bef, 0xd768226b34870a01},
	{0x81842f29f2cce375, 0xe6a1158300d46641},
	{0xa1e53af46f801c53, 0x60495ae3c1097fd1},
	{0xca5e89b18b602368, 0x385bb19cb14bdfc5},
	{0xfcf62c1dee382c42, 0x46729e03dd9ed7b6},
	{0x9e19db92b4e31ba9, 0x6c07a2c26a8346d2},
	{0xc5a05277621be293, 0xc7098b7305241886},
	{0xf70867153aa2db38, 0xb8cbee4fc66d1ea8},
}

// The precomputed table of φ̃̃k for float32.
var cache32 = [78]uint64{
	0x81ceb32c4b43fcf5, 0xa2425ff75e14fc32,
	0xcad2f7f5359a3b3f, 0xfd87b5f28300ca0e,
	0x9e74d1b791e07e49, 0xc612062576589ddb,
	0xf79687aed3eec552, 0x9abe14cd44753b53,
	0xc16d9a0095928a28, 0xf1c90080baf72cb2,
	0x971da05074da7bef, 0xbce5086492111aeb,
	0xec1e4a7db69561a6, 0x9392ee8e921d5d08,
	0xb877aa3236a4b44a, 0xe69594bec44de15c,
	0x901d7cf73ab0acda, 0xb424dc35095cd810,
	0xe12e13424bb40e14, 0x8cbccc096f5088cc,
	0xafebff0bcb24aaff, 0xdbe6fecebdedd5bf,
	0x89705f4136b4a598, 0xabcc77118461cefd,
	0xd6bf94d5e57a42bd, 0x8637bd05af6c69b6,
	0xa7c5ac471b478424, 0xd1b71758e219652c,
	0x83126e978d4fdf3c, 0xa3d70a3d70a3d70b,
	0xcccccccccccccccd, 0x8000000000000000,
	0xa000000000000000, 0xc800000000000000,
	0xfa00000000000000, 0x9c40000000000000,
	0xc350000000000000, 0xf424000000000000,
	0x9896800000000000, 0xbebc200000000000,
	0xee6b280000000000, 0x9502f90000000000,
	0xba43b74000000000, 0xe8d4a51000000000,
	0x9184e72a00000000, 0xb5e620f480000000,
	0xe35fa931a0000000, 0x8e1bc9bf04000000,
	0xb1a2bc2ec5000000, 0xde0b6b3a76400000,
	0x8ac7230489e80000, 0xad78ebc5ac620000,
	0xd8d726b7177a8000, 0x878678326eac9000,
	0xa968163f0a57b400, 0xd3c21bcecceda100,
	0x84595161401484a0, 0xa56fa5b99019a5c8,
	0xcecb8f27f4200f3a, 0x813f3978f8940985,
	0xa18f07d736b90be6, 0xc9f2c9cd04674edf,
	0xfc6f7c4045812297, 0x9dc5ada82b70b59e,
	0xc5371912364ce306, 0xf684df56c3e01bc7,
	0x9a130b963a6c115d, 0xc097ce7bc90715b4,
	0xf0bdc21abb48db21, 0x96769950b50d88f5,
	0xbc143fa4e250eb32, 0xeb194f8e1ae525fe,
	0x92efd1b8d0cf37bf, 0xb7abc627050305ae,
	0xe596b7b0c643c71a, 0x8f7e32ce7bea5c70,
	0xb35dbf821ae4f38c, 0xe0352f62a19e306f,
}
