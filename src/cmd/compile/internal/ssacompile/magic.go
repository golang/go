// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

// So you want to compute x / c for some constant c?
// Machine division instructions are slow, so we try to
// compute this division with a multiplication + a few
// other cheap instructions instead.
// (We assume here that c != 0, +/- 1, or +/- 2^i.  Those
// cases are easy to handle in different ways).

// Technique from https://gmplib.org/~tege/divcnst-pldi94.pdf

// First consider unsigned division.
// Our strategy is to precompute 1/c then do
//   ⎣x / c⎦ = ⎣x * (1/c)⎦.
// 1/c is less than 1, so we can't compute it directly in
// integer arithmetic.  Let's instead compute 2^e/c
// for a value of e TBD (^ = exponentiation).  Then
//   ⎣x / c⎦ = ⎣x * (2^e/c) / 2^e⎦.
// Dividing by 2^e is easy.  2^e/c isn't an integer, unfortunately.
// So we must approximate it.  Let's call its approximation m.
// We'll then compute
//   ⎣x * m / 2^e⎦
// Which we want to be equal to ⎣x / c⎦ for 0 <= x < 2^n-1
// where n is the word size.
// Setting x = c gives us c * m >= 2^e.
// We'll chose m = ⎡2^e/c⎤ to satisfy that equation.
// What remains is to choose e.
// Let m = 2^e/c + delta, 0 <= delta < 1
//   ⎣x * (2^e/c + delta) / 2^e⎦
//   ⎣x / c + x * delta / 2^e⎦
// We must have x * delta / 2^e < 1/c so that this
// additional term never rounds differently than ⎣x / c⎦ does.
// Rearranging,
//   2^e > x * delta * c
// x can be at most 2^n-1 and delta can be at most 1.
// So it is sufficient to have 2^e >= 2^n*c.
// So we'll choose e = n + s, with s = ⎡log2(c)⎤.
//
// An additional complication arises because m has n+1 bits in it.
// Hardware restricts us to n bit by n bit multiplies.
// We divide into 3 cases:
//
// Case 1: m is even.
//   ⎣x / c⎦ = ⎣x * m / 2^(n+s)⎦
//   ⎣x / c⎦ = ⎣x * (m/2) / 2^(n+s-1)⎦
//   ⎣x / c⎦ = ⎣x * (m/2) / 2^n / 2^(s-1)⎦
//   ⎣x / c⎦ = ⎣⎣x * (m/2) / 2^n⎦ / 2^(s-1)⎦
//   multiply + shift
//
// Case 2: c is even.
//   ⎣x / c⎦ = ⎣(x/2) / (c/2)⎦
//   ⎣x / c⎦ = ⎣⎣x/2⎦ / (c/2)⎦
//     This is just the original problem, with x' = ⎣x/2⎦, c' = c/2, n' = n-1.
//       s' = s-1
//       m' = ⎡2^(n'+s')/c'⎤
//          = ⎡2^(n+s-1)/c⎤
//          = ⎡m/2⎤
//   ⎣x / c⎦ = ⎣x' * m' / 2^(n'+s')⎦
//   ⎣x / c⎦ = ⎣⎣x/2⎦ * ⎡m/2⎤ / 2^(n+s-2)⎦
//   ⎣x / c⎦ = ⎣⎣⎣x/2⎦ * ⎡m/2⎤ / 2^n⎦ / 2^(s-2)⎦
//   shift + multiply + shift
//
// Case 3: everything else
//   let k = m - 2^n. k fits in n bits.
//   ⎣x / c⎦ = ⎣x * m / 2^(n+s)⎦
//   ⎣x / c⎦ = ⎣x * (2^n + k) / 2^(n+s)⎦
//   ⎣x / c⎦ = ⎣(x + x * k / 2^n) / 2^s⎦
//   ⎣x / c⎦ = ⎣(x + ⎣x * k / 2^n⎦) / 2^s⎦
//   ⎣x / c⎦ = ⎣(x + ⎣x * k / 2^n⎦) / 2^s⎦
//   ⎣x / c⎦ = ⎣⎣(x + ⎣x * k / 2^n⎦) / 2⎦ / 2^(s-1)⎦
//   multiply + avg + shift
//
// These can be implemented in hardware using:
//  ⎣a * b / 2^n⎦ - aka high n bits of an n-bit by n-bit multiply.
//  ⎣(a+b) / 2⎦   - aka "average" of two n-bit numbers.
//                  (Not just a regular add & shift because the intermediate result
//                   a+b has n+1 bits in it.  Nevertheless, can be done
//                   in 2 instructions on x86.)

// umagicOK reports whether we should strength reduce a n-bit divide by c.
func umagicOK(n uint, c int64) bool {
	// Convert from ConstX auxint values to the real uint64 constant they represent.
	d := uint64(c) << (64 - n) >> (64 - n)

	// Doesn't work for 0.
	// Don't use for powers of 2.
	return d&(d-1) != 0
}
