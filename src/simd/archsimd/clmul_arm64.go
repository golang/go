// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package archsimd

// CarrylessMultiplyEven computes the carryless
// multiplications of selected even halves of the elements of x and y.
//
// A carryless multiplication uses bitwise XOR instead of
// add-with-carry, for example (in base two):
//
//	11 * 11 = 11 * (10 ^ 1) = (11 * 10) ^ (11 * 1) = 110 ^ 11 = 101
//
// This also models multiplication of polynomials with coefficients
// from GF(2) -- 11 * 11 models (x+1)*(x+1) = x**2 + (1^1)x + 1 =
// x**2 + 0x + 1 = x**2 + 1 modeled by 101.  (Note that "+" adds
// polynomial terms, but coefficients "add" with XOR.)
//
// Asm: PMULL, CPU Feature: PMULL
func (x Uint64x2) CarrylessMultiplyEven(y Uint64x2) Uint64x2 {
	return x.carrylessMultiplyWidenLo(y)
}

// CarrylessMultiplyOdd computes the carryless
// multiplications of selected odd halves of the elements of x and y.
//
// A carryless multiplication uses bitwise XOR instead of
// add-with-carry, for example (in base two):
//
//	11 * 11 = 11 * (10 ^ 1) = (11 * 10) ^ (11 * 1) = 110 ^ 11 = 101
//
// This also models multiplication of polynomials with coefficients
// from GF(2) -- 11 * 11 models (x+1)*(x+1) = x**2 + (1^1)x + 1 =
// x**2 + 0x + 1 = x**2 + 1 modeled by 101.  (Note that "+" adds
// polynomial terms, but coefficients "add" with XOR.)
//
// Asm: PMULL, CPU Feature: PMULL
func (x Uint64x2) CarrylessMultiplyOdd(y Uint64x2) Uint64x2 {
	return x.HiToLo().carrylessMultiplyWidenLo(y.HiToLo())
}

// CarrylessMultiplyOddEven computes the carryless
// multiplications of selected odd half of x's elements and even half of y's elements.
//
// A carryless multiplication uses bitwise XOR instead of
// add-with-carry, for example (in base two):
//
//	11 * 11 = 11 * (10 ^ 1) = (11 * 10) ^ (11 * 1) = 110 ^ 11 = 101
//
// This also models multiplication of polynomials with coefficients
// from GF(2) -- 11 * 11 models (x+1)*(x+1) = x**2 + (1^1)x + 1 =
// x**2 + 0x + 1 = x**2 + 1 modeled by 101.  (Note that "+" adds
// polynomial terms, but coefficients "add" with XOR.)
//
// Asm: PMULL, CPU Feature: PMULL
func (x Uint64x2) CarrylessMultiplyOddEven(y Uint64x2) Uint64x2 {
	return x.HiToLo().carrylessMultiplyWidenLo(y)
}

// CarrylessMultiplyEvenOdd computes the carryless
// multiplications of selected even half of x's elements and odd half of y's elements.
//
// A carryless multiplication uses bitwise XOR instead of
// add-with-carry, for example (in base two):
//
//	11 * 11 = 11 * (10 ^ 1) = (11 * 10) ^ (11 * 1) = 110 ^ 11 = 101
//
// This also models multiplication of polynomials with coefficients
// from GF(2) -- 11 * 11 models (x+1)*(x+1) = x**2 + (1^1)x + 1 =
// x**2 + 0x + 1 = x**2 + 1 modeled by 101.  (Note that "+" adds
// polynomial terms, but coefficients "add" with XOR.)
//
// Asm: PMULL, CPU Feature: PMULL
func (x Uint64x2) CarrylessMultiplyEvenOdd(y Uint64x2) Uint64x2 {
	return x.carrylessMultiplyWidenLo(y.HiToLo())
}
