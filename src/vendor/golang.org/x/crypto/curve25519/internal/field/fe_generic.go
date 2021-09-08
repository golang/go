// Copyright (c) 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package field

import "math/bits"

// uint128 holds a 128-bit number as two 64-bit limbs, for use with the
// bits.Mul64 and bits.Add64 intrinsics.
type uint128 struct {
	lo, hi uint64
}

// mul64 returns a * b.
func mul64(a, b uint64) uint128 {
	hi, lo := bits.Mul64(a, b)
	return uint128{lo, hi}
}

// addMul64 returns v + a * b.
func addMul64(v uint128, a, b uint64) uint128 {
	hi, lo := bits.Mul64(a, b)
	lo, c := bits.Add64(lo, v.lo, 0)
	hi, _ = bits.Add64(hi, v.hi, c)
	return uint128{lo, hi}
}

// shiftRightBy51 returns a >> 51. a is assumed to be at most 115 bits.
func shiftRightBy51(a uint128) uint64 {
	return (a.hi << (64 - 51)) | (a.lo >> 51)
}

func feMulGeneric(v, a, b *Element) {
	a0 := a.l0
	a1 := a.l1
	a2 := a.l2
	a3 := a.l3
	a4 := a.l4

	b0 := b.l0
	b1 := b.l1
	b2 := b.l2
	b3 := b.l3
	b4 := b.l4

	// Limb multiplication works like pen-and-paper columnar multiplication, but
	// with 51-bit limbs instead of digits.
	//
	//                          a4   a3   a2   a1   a0  x
	//                          b4   b3   b2   b1   b0  =
	//                         ------------------------
	//                        a4b0 a3b0 a2b0 a1b0 a0b0  +
	//                   a4b1 a3b1 a2b1 a1b1 a0b1       +
	//              a4b2 a3b2 a2b2 a1b2 a0b2            +
	//         a4b3 a3b3 a2b3 a1b3 a0b3                 +
	//    a4b4 a3b4 a2b4 a1b4 a0b4                      =
	//   ----------------------------------------------
	//      r8   r7   r6   r5   r4   r3   r2   r1   r0
	//
	// We can then use the reduction identity (a * 2²⁵⁵ + b = a * 19 + b) to
	// reduce the limbs that would overflow 255 bits. r5 * 2²⁵⁵ becomes 19 * r5,
	// r6 * 2³⁰⁶ becomes 19 * r6 * 2⁵¹, etc.
	//
	// Reduction can be carried out simultaneously to multiplication. For
	// example, we do not compute r5: whenever the result of a multiplication
	// belongs to r5, like a1b4, we multiply it by 19 and add the result to r0.
	//
	//            a4b0    a3b0    a2b0    a1b0    a0b0  +
	//            a3b1    a2b1    a1b1    a0b1 19×a4b1  +
	//            a2b2    a1b2    a0b2 19×a4b2 19×a3b2  +
	//            a1b3    a0b3 19×a4b3 19×a3b3 19×a2b3  +
	//            a0b4 19×a4b4 19×a3b4 19×a2b4 19×a1b4  =
	//           --------------------------------------
	//              r4      r3      r2      r1      r0
	//
	// Finally we add up the columns into wide, overlapping limbs.

	a1_19 := a1 * 19
	a2_19 := a2 * 19
	a3_19 := a3 * 19
	a4_19 := a4 * 19

	// r0 = a0×b0 + 19×(a1×b4 + a2×b3 + a3×b2 + a4×b1)
	r0 := mul64(a0, b0)
	r0 = addMul64(r0, a1_19, b4)
	r0 = addMul64(r0, a2_19, b3)
	r0 = addMul64(r0, a3_19, b2)
	r0 = addMul64(r0, a4_19, b1)

	// r1 = a0×b1 + a1×b0 + 19×(a2×b4 + a3×b3 + a4×b2)
	r1 := mul64(a0, b1)
	r1 = addMul64(r1, a1, b0)
	r1 = addMul64(r1, a2_19, b4)
	r1 = addMul64(r1, a3_19, b3)
	r1 = addMul64(r1, a4_19, b2)

	// r2 = a0×b2 + a1×b1 + a2×b0 + 19×(a3×b4 + a4×b3)
	r2 := mul64(a0, b2)
	r2 = addMul64(r2, a1, b1)
	r2 = addMul64(r2, a2, b0)
	r2 = addMul64(r2, a3_19, b4)
	r2 = addMul64(r2, a4_19, b3)

	// r3 = a0×b3 + a1×b2 + a2×b1 + a3×b0 + 19×a4×b4
	r3 := mul64(a0, b3)
	r3 = addMul64(r3, a1, b2)
	r3 = addMul64(r3, a2, b1)
	r3 = addMul64(r3, a3, b0)
	r3 = addMul64(r3, a4_19, b4)

	// r4 = a0×b4 + a1×b3 + a2×b2 + a3×b1 + a4×b0
	r4 := mul64(a0, b4)
	r4 = addMul64(r4, a1, b3)
	r4 = addMul64(r4, a2, b2)
	r4 = addMul64(r4, a3, b1)
	r4 = addMul64(r4, a4, b0)

	// After the multiplication, we need to reduce (carry) the five coefficients
	// to obtain a result with limbs that are at most slightly larger than 2⁵¹,
	// to respect the Element invariant.
	//
	// Overall, the reduction works the same as carryPropagate, except with
	// wider inputs: we take the carry for each coefficient by shifting it right
	// by 51, and add it to the limb above it. The top carry is multiplied by 19
	// according to the reduction identity and added to the lowest limb.
	//
	// The largest coefficient (r0) will be at most 111 bits, which guarantees
	// that all carries are at most 111 - 51 = 60 bits, which fits in a uint64.
	//
	//     r0 = a0×b0 + 19×(a1×b4 + a2×b3 + a3×b2 + a4×b1)
	//     r0 < 2⁵²×2⁵² + 19×(2⁵²×2⁵² + 2⁵²×2⁵² + 2⁵²×2⁵² + 2⁵²×2⁵²)
	//     r0 < (1 + 19 × 4) × 2⁵² × 2⁵²
	//     r0 < 2⁷ × 2⁵² × 2⁵²
	//     r0 < 2¹¹¹
	//
	// Moreover, the top coefficient (r4) is at most 107 bits, so c4 is at most
	// 56 bits, and c4 * 19 is at most 61 bits, which again fits in a uint64 and
	// allows us to easily apply the reduction identity.
	//
	//     r4 = a0×b4 + a1×b3 + a2×b2 + a3×b1 + a4×b0
	//     r4 < 5 × 2⁵² × 2⁵²
	//     r4 < 2¹⁰⁷
	//

	c0 := shiftRightBy51(r0)
	c1 := shiftRightBy51(r1)
	c2 := shiftRightBy51(r2)
	c3 := shiftRightBy51(r3)
	c4 := shiftRightBy51(r4)

	rr0 := r0.lo&maskLow51Bits + c4*19
	rr1 := r1.lo&maskLow51Bits + c0
	rr2 := r2.lo&maskLow51Bits + c1
	rr3 := r3.lo&maskLow51Bits + c2
	rr4 := r4.lo&maskLow51Bits + c3

	// Now all coefficients fit into 64-bit registers but are still too large to
	// be passed around as a Element. We therefore do one last carry chain,
	// where the carries will be small enough to fit in the wiggle room above 2⁵¹.
	*v = Element{rr0, rr1, rr2, rr3, rr4}
	v.carryPropagate()
}

func feSquareGeneric(v, a *Element) {
	l0 := a.l0
	l1 := a.l1
	l2 := a.l2
	l3 := a.l3
	l4 := a.l4

	// Squaring works precisely like multiplication above, but thanks to its
	// symmetry we get to group a few terms together.
	//
	//                          l4   l3   l2   l1   l0  x
	//                          l4   l3   l2   l1   l0  =
	//                         ------------------------
	//                        l4l0 l3l0 l2l0 l1l0 l0l0  +
	//                   l4l1 l3l1 l2l1 l1l1 l0l1       +
	//              l4l2 l3l2 l2l2 l1l2 l0l2            +
	//         l4l3 l3l3 l2l3 l1l3 l0l3                 +
	//    l4l4 l3l4 l2l4 l1l4 l0l4                      =
	//   ----------------------------------------------
	//      r8   r7   r6   r5   r4   r3   r2   r1   r0
	//
	//            l4l0    l3l0    l2l0    l1l0    l0l0  +
	//            l3l1    l2l1    l1l1    l0l1 19×l4l1  +
	//            l2l2    l1l2    l0l2 19×l4l2 19×l3l2  +
	//            l1l3    l0l3 19×l4l3 19×l3l3 19×l2l3  +
	//            l0l4 19×l4l4 19×l3l4 19×l2l4 19×l1l4  =
	//           --------------------------------------
	//              r4      r3      r2      r1      r0
	//
	// With precomputed 2×, 19×, and 2×19× terms, we can compute each limb with
	// only three Mul64 and four Add64, instead of five and eight.

	l0_2 := l0 * 2
	l1_2 := l1 * 2

	l1_38 := l1 * 38
	l2_38 := l2 * 38
	l3_38 := l3 * 38

	l3_19 := l3 * 19
	l4_19 := l4 * 19

	// r0 = l0×l0 + 19×(l1×l4 + l2×l3 + l3×l2 + l4×l1) = l0×l0 + 19×2×(l1×l4 + l2×l3)
	r0 := mul64(l0, l0)
	r0 = addMul64(r0, l1_38, l4)
	r0 = addMul64(r0, l2_38, l3)

	// r1 = l0×l1 + l1×l0 + 19×(l2×l4 + l3×l3 + l4×l2) = 2×l0×l1 + 19×2×l2×l4 + 19×l3×l3
	r1 := mul64(l0_2, l1)
	r1 = addMul64(r1, l2_38, l4)
	r1 = addMul64(r1, l3_19, l3)

	// r2 = l0×l2 + l1×l1 + l2×l0 + 19×(l3×l4 + l4×l3) = 2×l0×l2 + l1×l1 + 19×2×l3×l4
	r2 := mul64(l0_2, l2)
	r2 = addMul64(r2, l1, l1)
	r2 = addMul64(r2, l3_38, l4)

	// r3 = l0×l3 + l1×l2 + l2×l1 + l3×l0 + 19×l4×l4 = 2×l0×l3 + 2×l1×l2 + 19×l4×l4
	r3 := mul64(l0_2, l3)
	r3 = addMul64(r3, l1_2, l2)
	r3 = addMul64(r3, l4_19, l4)

	// r4 = l0×l4 + l1×l3 + l2×l2 + l3×l1 + l4×l0 = 2×l0×l4 + 2×l1×l3 + l2×l2
	r4 := mul64(l0_2, l4)
	r4 = addMul64(r4, l1_2, l3)
	r4 = addMul64(r4, l2, l2)

	c0 := shiftRightBy51(r0)
	c1 := shiftRightBy51(r1)
	c2 := shiftRightBy51(r2)
	c3 := shiftRightBy51(r3)
	c4 := shiftRightBy51(r4)

	rr0 := r0.lo&maskLow51Bits + c4*19
	rr1 := r1.lo&maskLow51Bits + c0
	rr2 := r2.lo&maskLow51Bits + c1
	rr3 := r3.lo&maskLow51Bits + c2
	rr4 := r4.lo&maskLow51Bits + c3

	*v = Element{rr0, rr1, rr2, rr3, rr4}
	v.carryPropagate()
}

// carryPropagate brings the limbs below 52 bits by applying the reduction
// identity (a * 2²⁵⁵ + b = a * 19 + b) to the l4 carry. TODO inline
func (v *Element) carryPropagateGeneric() *Element {
	c0 := v.l0 >> 51
	c1 := v.l1 >> 51
	c2 := v.l2 >> 51
	c3 := v.l3 >> 51
	c4 := v.l4 >> 51

	v.l0 = v.l0&maskLow51Bits + c4*19
	v.l1 = v.l1&maskLow51Bits + c0
	v.l2 = v.l2&maskLow51Bits + c1
	v.l3 = v.l3&maskLow51Bits + c2
	v.l4 = v.l4&maskLow51Bits + c3

	return v
}
