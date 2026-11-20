// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable

package archsimd

// The 64-bit polynomial multiply instruction (PMULL with a 1Q arrangement)
// requires FEAT_PMULL, which is optional. Most arm64 CPUs have it, but some
// don't, notably the Raspberry Pi 4 and earlier. The CarrylessMul methods
// use that instruction if it's available, and otherwise fall back to
// carrylessMultiplyP8, which builds the 64-bit multiply out of the 8-bit
// polynomial multiply that is part of baseline Advanced SIMD.

// CarrylessMulEven computes the elementwise carryless multiplication of
// even-indexed elements of x and y. The result of each carryless multiply is
// twice the width of the input elements. The high bits are stored in elements
// z[2*i+1] and the low bits are stored in elements z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i], y[2*i])
//
// A carryless multiplication uses bitwise XOR instead of add-with-carry.
// For example, to compute the carryless multiply of 0b1110 and 0b1011:
//
//	     1110
//	   ⊗ 1011
//	  ───────
//	     1110
//	    1110
//	   0000
//	⊕ 1110
//	  ───────
//	  1100010
//
// Carryless multiply can also be viewed as multiplying polynomials with
// coefficients from GF(2). For example, the above example can be represented as
//
//	  (x^3 + x^2 + x^1) * (x^3 + x^1 + x^0)
//	= (x^6 + x^5 + (1^1)x^4 + (1^1)x^3 + (1^1)x^2 + x^1)
//	= (x^6 + x^5 + x^1)
//
// Asm: PMULL if CPU feature PMULL, Emulated otherwise
func (x Uint64x2) CarrylessMulEven(y Uint64x2) (z Uint64x2) {
	if ARM64.PMULL() {
		return x.carrylessMultiplyWidenLo(y)
	}
	return carrylessMultiplyP8(x.InterleaveLo(x), y.InterleaveLo(y))
}

// CarrylessMulOdd computes the elementwise carryless multiplication of
// odd-indexed elements of x and y. The result of each carryless multiply is
// twice the width of the input elements. The high bits are stored in elements
// z[2*i+1] and the low bits are stored in elements z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i+1], y[2*i+1])
//
// See [CarrylessMulEven] for details about carryless multiply.
//
// Asm: PMULL if CPU feature PMULL, Emulated otherwise
func (x Uint64x2) CarrylessMulOdd(y Uint64x2) (z Uint64x2) {
	if ARM64.PMULL() {
		return x.HiToLo().carrylessMultiplyWidenLo(y.HiToLo())
	}
	return carrylessMultiplyP8(x.InterleaveHi(x), y.InterleaveHi(y))
}

// CarrylessMulOddEven computes the elementwise carryless multiplication of
// odd-indexed elements of x and even-indexed elements of y. The result of each
// carryless multiply is twice the width of the input elements. The high bits
// are stored in elements z[2*i+1] and the low bits are stored in elements
// z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i+1], y[2*i])
//
// See [CarrylessMulEven] for details about carryless multiply.
//
// Asm: PMULL if CPU feature PMULL, Emulated otherwise
func (x Uint64x2) CarrylessMulOddEven(y Uint64x2) (z Uint64x2) {
	if ARM64.PMULL() {
		return x.HiToLo().carrylessMultiplyWidenLo(y)
	}
	return carrylessMultiplyP8(x.InterleaveHi(x), y.InterleaveLo(y))
}

// CarrylessMulEvenOdd computes the elementwise carryless multiplication of
// even-indexed elements of x and odd-indexed elements of y. The result of each
// carryless multiply is twice the width of the input elements. The high bits
// are stored in elements z[2*i+1] and the low bits are stored in elements
// z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i], y[2*i+1])
//
// See [CarrylessMulEven] for details about carryless multiply.
//
// Asm: PMULL if CPU feature PMULL, Emulated otherwise
func (x Uint64x2) CarrylessMulEvenOdd(y Uint64x2) (z Uint64x2) {
	if ARM64.PMULL() {
		return x.carrylessMultiplyWidenLo(y.HiToLo())
	}
	return carrylessMultiplyP8(x.InterleaveLo(x), y.InterleaveHi(y))
}

// clmulP8Place holds the index vectors that carrylessMultiplyP8 uses to put the
// terms of its partial products in place. See there for details.
//
// There are two vectors for each pk, for k = 1 to 4. The first moves the terms
// of pk that didn't wrap around left by k bytes. The second moves the terms that
// did wrap around right by 8-k bytes. Both zero everything else. Each vector is
// listed from byte 0 up, which is the reverse of the diagrams in
// carrylessMultiplyP8.
//
// ___ is an index that is out of range, for which LookupOrZero produces a zero
// byte.
var clmulP8Place = [8][16]uint8{
	{___, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, ___},
	{___, ___, ___, ___, ___, ___, ___, 0xe, 0xf, ___, ___, ___, ___, ___, ___, ___},
	{___, ___, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, ___, ___},
	{___, ___, ___, ___, ___, ___, 0xc, 0xd, 0xe, 0xf, ___, ___, ___, ___, ___, ___},
	{___, ___, ___, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, ___, ___, ___},
	{___, ___, ___, ___, ___, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, ___, ___, ___, ___, ___},
	{___, ___, ___, ___, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, ___, ___, ___, ___},
	{___, ___, ___, ___, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, ___, ___, ___, ___},
}

const ___ = 0xff

// carrylessMultiplyP8 returns the 128-bit carryless product of two 64-bit
// values, x and y, using only the 8-bit polynomial multiply. Both elements of
// a64 must hold x, and both elements of b64 must hold y.
//
// This is constant-time. It follows the approach of "Fast Software Polynomial
// Multiplication on ARM Processors Using the NEON Engine" by Câmara, Gouvêa,
// López, and Dahab, which is also used by the Linux kernel's GHASH for CPUs
// without FEAT_PMULL, except that it puts the partial products in place using
// table lookups.
//
// It's about 45 instructions, so it's not inlined: that keeps the
// CarrylessMul methods, and their fast paths, small enough to inline.
//
//go:noinline
func carrylessMultiplyP8(a64, b64 Uint64x2) Uint64x2 {
	// This is long multiplication without carries, where the digits are the
	// bytes of x and y: x7..x0 and y7..y0. To keep the example on the page,
	// consider 32-bit (4 byte) inputs instead of the full 64 bits.
	//
	// Fully written out as long-hand multiplication, where ⊗ represents
	// carryless multiply, ⊕ represents XOR, and [u⊗v] represents the 15 bit
	// result of the carryless multiply of bytes u and v:
	//
	//                      x3  x2  x1  x0
	//                    ⊗ y3  y2  y1  y0
	//     -------------------------------
	//                 [x3⊗y0]
	//                     [x2⊗y0]
	//                         [x1⊗y0]
	//                             [x0⊗y0]
	//             [x3⊗y1]
	//                 [x2⊗y1]
	//                     [x1⊗y1]
	//                         [x0⊗y1]
	//         [x3⊗y2]
	//             [x2⊗y2]
	//                 [x1⊗y2]
	//                     [x0⊗y2]
	//     [x3⊗y3]
	//         [x2⊗y3]
	//             [x1⊗y3]
	// ⊕               [x0⊗y3]
	//
	// (We draw each of the usual rows of intermediate results on a diagonal
	// because the 15 bit intermediates overlap.)
	//
	// We can rearrange the inputs to the XOR freely, as long as each one
	// stays in its column. Numbering the bytes of the result:
	//
	//      7   6   5   4   3   2   1   0
	//     [x3⊗y3] [x2⊗y2] [x1⊗y1] [x0⊗y0]  = p0
	//
	//         [x2⊗y3] [x1⊗y2] [x0⊗y1]
	//                 [x3⊗y0]
	//         [x3⊗y2] [x2⊗y1] [x1⊗y0]
	// ⊕               [x0⊗y3]              = q1
	//
	//             [x1⊗y3] [x0⊗y2]
	// ⊕           [x3⊗y1] [x2⊗y0]          = q2
	//
	// The result is p0 ⊕ q1 ⊕ q2.
	//
	// We group them this way because of what the 8-bit vector polynomial
	// multiply instruction computes. (In the code it is
	// carrylessMultiplyWidenLo; here, pmull.) It multiplies each byte of
	// the low half of one input by the same byte of the other, and produces
	// the results side by side. Writing xx for the bytes it ignores:
	//
	//      7   6   5   4   3   2   1   0
	//      xx  xx  xx  xx  u3  u2  u1  u0  = u
	//      xx  xx  xx  xx  v3  v2  v1  v0  = v
	//     [u3⊗v3] [u2⊗v2] [u1⊗v1] [u0⊗v0]  = pmull(u, v)
	//
	// So p0 is pmull(x, y).
	//
	// In q1, each byte of x is paired with the byte of y one position up or
	// one position down, wrapping around between 3 and 0. Rotating one
	// input right by a byte before multiplying produces those pairs.
	// Writing rot(y, 1) for y0 y3 y2 y1:
	//
	//      7   6   5   4   3   2   1   0
	//     [x3⊗y0] [x2⊗y3] [x1⊗y2] [x0⊗y1]  = pmull(x, rot(y, 1))
	//     [x0⊗y3] [x3⊗y2] [x2⊗y1] [x1⊗y0]  = pmull(rot(x, 1), y)
	//
	// These are the terms of q1, but in the wrong places. In both lines,
	// the three right-most terms belong one byte to the left, and the
	// left-most term, where the index wrapped around, belongs three bytes
	// to the right. Both lines need the same fix, so first XOR them
	// together into p1. Each 16-bit lane of p1 then holds a pair of terms,
	// one from each line, that belong in the same place:
	//
	//        7       6       5       4       3       2       1       0
	//     [x3⊗y0 ⊕ x0⊗y3] [x2⊗y3 ⊕ x3⊗y2] [x1⊗y2 ⊕ x2⊗y1] [x0⊗y1 ⊕ x1⊗y0]  = p1
	//
	// Put the pairs in place with two table lookups, one for the pairs
	// that didn't wrap around and one for the pair that did, and XOR the
	// results. It takes two lookups because the pair that wrapped around
	// lands on top of other pairs. The first lookup moves the three
	// right-most pairs left by one byte and zeros everything else:
	//
	//        7       6       5       4       3       2       1       0
	//             [x2⊗y3 ⊕ x3⊗y2] [x1⊗y2 ⊕ x2⊗y1] [x0⊗y1 ⊕ x1⊗y0]
	//
	// The second moves the left-most pair right by three bytes and zeros
	// everything else:
	//
	//        7       6       5       4       3       2       1       0
	//                             [x3⊗y0 ⊕ x0⊗y3]
	//
	// The XOR of the two lookups is q1.
	//
	// For q2, rotate an input by two bytes. Rotating x or y gives the same
	// terms, so one multiply is enough:
	//
	//      7   6   5   4   3   2   1   0
	//     [x3⊗y1] [x2⊗y0] [x1⊗y3] [x0⊗y2]  = p2 = pmull(x, rot(y, 2))
	//
	// Here the left-most two terms wrapped around. Each term is two bytes
	// wide. The first lookup moves the right-most two terms (four bytes)
	// left by two bytes and zeros everything else:
	//
	//      7   6   5   4   3   2   1   0
	//             [x1⊗y3] [x0⊗y2]
	//
	// The second moves the left-most two terms (four bytes) right by two
	// bytes and zeros everything else:
	//
	//      7   6   5   4   3   2   1   0
	//             [x3⊗y1] [x2⊗y0]
	//
	// The XOR of the two lookups is q2.
	//
	// In general, the left-most k terms (2k bytes) of pk wrapped around.
	// With n-byte inputs, one lookup moves the other n-k terms left by k
	// bytes, and the other lookup moves those k terms right by n-k bytes.
	//
	// With the full 8 bytes, there are p0 through p4. p1, p2, and p3 take
	// two multiplies each, and p0 and p4 take one, so eight multiplies
	// produce all 64 terms.
	//
	// In the code, x and y are 64 bits, and each one is in both halves of a
	// vector:
	//
	//     byte   15  14  13  12  11  10   9   8   7   6   5   4   3   2   1   0
	//     a      x7  x6  x5  x4  x3  x2  x1  x0  x7  x6  x5  x4  x3  x2  x1  x0
	//     b      y7  y6  y5  y4  y3  y2  y1  y0  y7  y6  y5  y4  y3  y2  y1  y0
	//
	// pmull reads only the low 8 bytes of each input. The upper copy is
	// there to make rot cheap: a window on the concatenation of a with
	// itself that starts at byte k has rot(x, k) in both halves. For
	// example, a1 is
	//
	//     a1     x0  x7  x6  x5  x4  x3  x2  x1  x0  x7  x6  x5  x4  x3  x2  x1
	//
	// So ak and bk below are rot(x, k) and rot(y, k).
	a := a64.ReshapeToUint8s()
	b := b64.ReshapeToUint8s()

	a1 := a.ConcatWindow(a, 1)
	a2 := a.ConcatWindow(a, 2)
	a3 := a.ConcatWindow(a, 3)
	b1 := b.ConcatWindow(b, 1)
	b2 := b.ConcatWindow(b, 2)
	b3 := b.ConcatWindow(b, 3)
	b4 := b.ConcatWindow(b, 4)

	// pk holds the terms for rotating by k, where the multiply leaves them.
	p0 := a.carrylessMultiplyWidenLo(b).ReshapeToUint8s()
	p1 := a.carrylessMultiplyWidenLo(b1).Xor(a1.carrylessMultiplyWidenLo(b)).ReshapeToUint8s()
	p2 := a.carrylessMultiplyWidenLo(b2).Xor(a2.carrylessMultiplyWidenLo(b)).ReshapeToUint8s()
	p3 := a.carrylessMultiplyWidenLo(b3).Xor(a3.carrylessMultiplyWidenLo(b)).ReshapeToUint8s()
	p4 := a.carrylessMultiplyWidenLo(b4).ReshapeToUint8s()

	// qk is pk with its terms in place.
	place := &clmulP8Place
	q1 := p1.LookupOrZero(LoadUint8x16Array(&place[0])).Xor(p1.LookupOrZero(LoadUint8x16Array(&place[1])))
	q2 := p2.LookupOrZero(LoadUint8x16Array(&place[2])).Xor(p2.LookupOrZero(LoadUint8x16Array(&place[3])))
	q3 := p3.LookupOrZero(LoadUint8x16Array(&place[4])).Xor(p3.LookupOrZero(LoadUint8x16Array(&place[5])))
	q4 := p4.LookupOrZero(LoadUint8x16Array(&place[6])).Xor(p4.LookupOrZero(LoadUint8x16Array(&place[7])))
	return p0.Xor(q1.Xor(q2).Xor(q3.Xor(q4))).ReshapeToUint64s()
}
