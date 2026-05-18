// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build riscv64 && !purego

package fiat

import "math/bits"

//go:noescape
func p256Mul(out, a, b *p256MontgomeryDomainFieldElement)

// p256Square squares a field element following ARM64 assembly algorithm of p256Sqr in p256_asm_arm64.s
// This implementation uses a delayed reduction strategy: compute all products first,
// then perform Montgomery reduction in batch. This avoids redundant multiplications
// and improves performance by keeping intermediate values in registers.
func p256Square(out1 *p256MontgomeryDomainFieldElement, arg1 *p256MontgomeryDomainFieldElement) {
	x0 := arg1[0]
	x1 := arg1[1]
	x2 := arg1[2]
	x3 := arg1[3]

	// =====================================
	// Stage 1: Compute cross products (avoiding duplicates)
	// =====================================
	// For x² = (x₀ + x₁·2⁶⁴ + x₂·2¹²⁸ + x₃·2¹⁹²)², we need:
	// - Square terms: x₀², x₁², x₂², x₃²
	// - Cross terms: x₀x₁, x₀x₂, x₀x₃, x₁x₂, x₁x₃, x₂x₃ (each multiplied by 2)
	//
	// We only compute the upper triangular matrix to avoid redundant multiplications.
	// acc1-acc7 will accumulate the 512-bit result of cross products.

	var acc1, acc2, acc3, acc4, acc5, acc6, acc7 uint64
	var carry uint64

	// Compute x₀ × x₁: result goes to acc1 (low) and acc2 (high)
	hi, lo := bits.Mul64(x0, x1)
	acc1 = lo
	acc2 = hi

	// Compute x₀ × x₂: low part adds to acc2, high part goes to acc3
	hi, lo = bits.Mul64(x0, x2)
	acc2, carry = bits.Add64(acc2, lo, 0)
	acc3 = hi

	// Compute x₀ × x₃: low part adds to acc3 with carry, high part goes to acc4
	hi, lo = bits.Mul64(x0, x3)
	acc3, carry = bits.Add64(acc3, lo, carry)
	acc4 = hi
	acc4, carry = bits.Add64(acc4, 0, carry)

	// Compute x₁ × x₂: low part adds to acc3, high part adds to acc4 with carry
	hi, lo = bits.Mul64(x1, x2)
	acc3, carry = bits.Add64(acc3, lo, 0)
	acc4, carry = bits.Add64(acc4, hi, carry)
	acc5 = carry

	// Compute x₁ × x₃: low part adds to acc4, high part adds to acc5 with carry
	hi, lo = bits.Mul64(x1, x3)
	acc4, carry = bits.Add64(acc4, lo, 0)
	acc5, _ = bits.Add64(acc5, hi, carry)

	// Compute x₂ × x₃: low part adds to acc5, high part goes to acc6
	hi, lo = bits.Mul64(x2, x3)
	acc5, carry = bits.Add64(acc5, lo, 0)
	acc6 = hi
	acc6, _ = bits.Add64(acc6, 0, carry)

	acc7 = 0

	// =====================================
	// Stage 2: Multiply cross products by 2
	// =====================================
	// In the square formula (a+b)² = a² + 2ab + b², cross terms need to be doubled.
	// We multiply each accumulator by 2 using addition (x + x = 2x).
	acc1, carry = bits.Add64(acc1, acc1, 0)
	acc2, carry = bits.Add64(acc2, acc2, carry)
	acc3, carry = bits.Add64(acc3, acc3, carry)
	acc4, carry = bits.Add64(acc4, acc4, carry)
	acc5, carry = bits.Add64(acc5, acc5, carry)
	acc6, carry = bits.Add64(acc6, acc6, carry)
	acc7, _ = bits.Add64(acc7, acc7, carry)

	// =====================================
	// Stage 3: Add square terms
	// =====================================
	// Now add the square terms: x₀², x₁², x₂², x₃²
	// acc0 will hold the lowest 64 bits of the result.
	var acc0 uint64

	// x₀²: low part goes to acc0, high part adds to acc1
	hi, lo = bits.Mul64(x0, x0)
	acc0 = lo
	acc1, carry = bits.Add64(acc1, hi, 0)

	// x₁²: low part adds to acc2, high part adds to acc3
	hi, lo = bits.Mul64(x1, x1)
	acc2, carry = bits.Add64(acc2, lo, carry)
	acc3, carry = bits.Add64(acc3, hi, carry)

	// x₂²: low part adds to acc4, high part adds to acc5
	hi, lo = bits.Mul64(x2, x2)
	acc4, carry = bits.Add64(acc4, lo, carry)
	acc5, carry = bits.Add64(acc5, hi, carry)

	// x₃²: low part adds to acc6, high part adds to acc7
	hi, lo = bits.Mul64(x3, x3)
	acc6, carry = bits.Add64(acc6, lo, carry)
	acc7, _ = bits.Add64(acc7, hi, carry)

	// At this point, acc0-acc7 contain the full 512-bit square result.

	// =====================================
	// Stage 4: Montgomery reduction (4 steps)
	// =====================================
	// Reduce the 512-bit result back to 256 bits modulo p.
	// Each reduction step processes one 64-bit limb using the Montgomery constant.
	const const1 = 0xffffffff00000001 // Montgomery constant for P-256

	// Reduction step 1: reduce acc0
	// Shift acc0 left by 32 bits and add to acc1, then compute acc0 * const1
	t0 := acc0 << 32
	acc1, carry = bits.Add64(acc1, t0, 0)
	t1 := acc0 >> 32
	hi, lo = bits.Mul64(acc0, const1)
	t2 := lo
	acc0 = hi // acc0 is overwritten with the high part of acc0 * const1
	acc2, carry = bits.Add64(acc2, t1, carry)
	acc3, carry = bits.Add64(acc3, t2, carry)
	acc0, _ = bits.Add64(acc0, 0, carry)

	// Reduction step 2: reduce acc1
	t0 = acc1 << 32
	acc2, carry = bits.Add64(acc2, t0, 0)
	t1 = acc1 >> 32
	hi, lo = bits.Mul64(acc1, const1)
	t2 = lo
	acc1 = hi
	acc3, carry = bits.Add64(acc3, t1, carry)
	acc0, carry = bits.Add64(acc0, t2, carry)
	acc1, _ = bits.Add64(acc1, 0, carry)

	// Reduction step 3: reduce acc2
	t0 = acc2 << 32
	acc3, carry = bits.Add64(acc3, t0, 0)
	t1 = acc2 >> 32
	hi, lo = bits.Mul64(acc2, const1)
	t2 = lo
	acc2 = hi
	acc0, carry = bits.Add64(acc0, t1, carry)
	acc1, carry = bits.Add64(acc1, t2, carry)
	acc2, _ = bits.Add64(acc2, 0, carry)

	// Reduction step 4: reduce acc3
	t0 = acc3 << 32
	acc0, carry = bits.Add64(acc0, t0, 0)
	t1 = acc3 >> 32
	hi, lo = bits.Mul64(acc3, const1)
	t2 = lo
	acc3 = hi
	acc1, carry = bits.Add64(acc1, t1, carry)
	acc2, carry = bits.Add64(acc2, t2, carry)
	acc3, _ = bits.Add64(acc3, 0, carry)

	// =====================================
	// Stage 5: Add high 256 bits
	// =====================================
	// Add the high 256 bits (acc4-acc7) to the reduced result (acc0-acc3)
	acc0, carry = bits.Add64(acc0, acc4, 0)
	acc1, carry = bits.Add64(acc1, acc5, carry)
	acc2, carry = bits.Add64(acc2, acc6, carry)
	acc3, carry = bits.Add64(acc3, acc7, carry)
	finalCarry := carry

	// =====================================
	// Stage 6: Conditional subtraction
	// =====================================
	// Ensure the result is in the range [0, p) by conditionally subtracting p.
	// This is necessary because the reduction might leave the result >= p.
	const p0 = 0xffffffffffffffff // P-256 prime: bits [63:0]
	const p1 = 0x00000000ffffffff // P-256 prime: bits [95:64]
	const p3 = 0xffffffff00000001 // P-256 prime: bits [255:192]

	// Compute (acc0, acc1, acc2, acc3) - (p0, p1, 0, p3)
	var t0_sub, t1_sub, t2_sub, t3_sub uint64
	var borrow uint64

	t0_sub, borrow = bits.Sub64(acc0, p0, 0)
	t1_sub, borrow = bits.Sub64(acc1, p1, borrow)
	t2_sub, borrow = bits.Sub64(acc2, 0, borrow)
	t3_sub, borrow = bits.Sub64(acc3, p3, borrow)

	// Check if the subtraction produced a borrow (meaning result < p)
	_, finalBorrow := bits.Sub64(finalCarry, 0, borrow)

	// If finalBorrow == 0, the original result was >= p, so use the subtracted value.
	// Otherwise, use the original result.
	if finalBorrow == 0 {
		out1[0] = t0_sub
		out1[1] = t1_sub
		out1[2] = t2_sub
		out1[3] = t3_sub
	} else {
		out1[0] = acc0
		out1[1] = acc1
		out1[2] = acc2
		out1[3] = acc3
	}
}
