// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jpeg

// Discrete Cosine Transformation (DCT) implementations using the algorithm from
// Christoph Loeffler, Adriaan Lightenberg, and George S. Mostchytz,
// “Practical Fast 1-D DCT Algorithms with 11 Multiplications,” ICASSP 1989.
// https://ieeexplore.ieee.org/document/266596
//
// Since the paper is paywalled, the rest of this comment gives a summary.
//
// A 1-dimensional forward DCT (1D FDCT) takes as input 8 values x0..x7
// and transforms them in place into the result values.
//
// The mathematical definition of the N-point 1D FDCT is:
//
//	X[k] = α_k Σ_n x[n] * cos (2n+1)*k*π/2N
//
// where α₀ = √2 and α_k = 1 for k > 0.
//
// For our purposes, N=8, so the angles end up being multiples of π/16.
// The most direct implementation of this definition would require 64 multiplications.
//
// Loeffler's paper presents a more efficient computation that requires only
// 11 multiplications and works in terms of three basic operations:
//
//  - A “butterfly” x0, x1 = x0+x1, x0-x1.
//    The inverse is x0, x1 = (x0+x1)/2, (x0-x1)/2.
//
//  - A scaling of x0 by k: x0 *= k. The inverse is scaling by 1/k.
//
//  - A rotation of x0, x1 by θ, defined as:
//    x0, x1 = x0 cos θ + x1 sin θ, -x0 sin θ + x1 cos θ.
//    The inverse is rotation by -θ.
//
// The algorithm proceeds in four stages:
//
// Stage 1:
//  - butterfly x0, x7; x1, x6; x2, x5; x3, x4.
//
// Stage 2:
//  - butterfly x0, x3; x1, x2
//  - rotate x4, x7 by 3π/16
//  - rotate x5, x6 by π/16.
//
// Stage 3:
//  - butterfly x0, x1; x4, x6; x7, x5
//  - rotate x2, x3 by 6π/16 and scale by √2.
//
// Stage 4:
//  - butterfly x7, x4
//  - scale x5, x6 by √2.
//
// Finally, the values are permuted. The permutation can be read as either:
//  - x0, x4, x2, x6, x7, x3, x5, x1 = x0, x1, x2, x3, x4, x5, x6, x7 (paper's form)
//  - x0, x1, x2, x3, x4, x5, x6, x7 = x0, x7, x2, x5, x1, x6, x3, x4 (sorted by LHS)
// The code below uses the second form to make it easier to merge adjacent stores.
// (Note that unlike in recursive FFT implementations, the permutation here is
// not always mapping indexes to their bit reversals.)
//
// As written above, the rotation requires four multiplications, but it can be
// reduced to three by refactoring (see [dctBox] below), and the scaling in
// stage 3 can be merged into the rotation constants, so the overall cost
// of a 1D FDCT is 11 multiplies.
//
// The 1D inverse DCT (IDCT) is the 1D FDCT run backward
// with all the basic operations inverted.

// dctBox implements a 3-multiply, 3-add rotation+scaling.
// Given x0, x1, k*cos θ, and k*sin θ, dctBox returns the
// rotated and scaled coordinates.
// (It is called dctBox because the rotate+scale operation
// is drawn as a box in Figures 1 and 2 in the paper.)
func dctBox(x0, x1, kcos, ksin int32) (y0, y1 int32) {
	// y0 = x0*kcos + x1*ksin
	// y1 = -x0*ksin + x1*kcos
	ksum := kcos * (x0 + x1)
	y0 = ksum + (ksin-kcos)*x1
	y1 = ksum - (kcos+ksin)*x0
	return y0, y1
}

// A block is an 8x8 input to a 2D DCT (either the FDCT or IDCT).
// The input is actually only 8x8 uint8 values, and the outputs are 8x8 int16,
// but it is convenient to use int32s for intermediate storage,
// so we define only a single block type of [8*8]int32.
//
// A 2D DCT is implemented as 1D DCTs over the rows and columns.
//
// dct_test.go defines a String method for nice printing in tests.
type block [blockSize]int32

const blockSize = 8 * 8

// Note on Numerical Precision
//
// The inputs to both the FDCT and IDCT are uint8 values stored in a block,
// and the outputs are int16s in the same block, but the overall operation
// uses int32 values as fixed-point intermediate values.
// In the code comments below, the notation “QN.M” refers to a
// signed value of 1+N+M significant bits, one of which is the sign bit,
// and M of which hold fractional (sub-integer) precision.
// For example, 255 as a Q8.0 value is stored as int32(255),
// while 255 as a Q8.1 value is stored as int32(510),
// and 255.5 as a Q8.1 value is int32(511).
// The notation UQN.M refers to an unsigned value of N+M significant bits.
// See https://en.wikipedia.org/wiki/Q_(number_format) for more.
//
// In general we only need to keep about 16 significant bits, but it is more
// efficient and somewhat more precise to let unnecessary fractional bits
// accumulate and shift them away in bulk rather than after every operation.
// As such, it is important to keep track of the number of fractional bits
// in each variable at different points in the code, to avoid mistakes like
// adding numbers with different fractional precisions, as well as to keep
// track of the total number of bits, to avoid overflow. A comment like:
//
//	// x[123] now Q8.2.
//
// means that x1, x2, and x3 are all Q8.2 (11-bit) values.
// Keeping extra precision bits also reduces the size of the errors introduced
// by using right shift to approximate rounded division.

// Constants needed for the implementation.
// These are all 60-bit precision fixed-point constants.
// The function c(val, b) rounds the constant to b bits.
// c is simple enough that calls to it with constant args
// are inlined and constant-propagated down to an inline constant.
// Each constant is commented with its Ivy definition (see robpike.io/ivy),
// using this scaling helper function:
//
//	op fix x = floor 0.5 + x * 2**60
const (
	cos1          = 1130768441178740757 // fix cos 1*pi/16
	sin1          = 224923827593068887  // fix sin 1*pi/16
	cos3          = 958619196450722178  // fix cos 3*pi/16
	sin3          = 640528868967736374  // fix sin 3*pi/16
	sqrt2         = 1630477228166597777 // fix sqrt 2
	sqrt2_cos6    = 623956622067911264  // fix (sqrt 2)*cos 6*pi/16
	sqrt2_sin6    = 1506364539328854985 // fix (sqrt 2)*sin 6*pi/16
	sqrt2inv      = 815238614083298888  // fix 1/sqrt 2
	sqrt2inv_cos6 = 311978311033955632  // fix (1/sqrt 2)*cos 6*pi/16
	sqrt2inv_sin6 = 753182269664427492  // fix (1/sqrt 2)*sin 6*pi/16
)

func c(x uint64, bits int) int32 {
	return int32((x + (1 << (59 - bits))) >> (60 - bits))
}

// fdct implements the forward DCT.
// Inputs are UQ8.0; outputs are Q13.0.
func fdct(b *block) {
	fdctCols(b)
	fdctRows(b)
}

// fdctCols applies the 1D DCT to the columns of b.
// Inputs are UQ8.0 in [0,255] but interpreted as [-128,127].
// Outputs are Q10.18.
func fdctCols(b *block) {
	for i := range 8 {
		x0 := b[0*8+i]
		x1 := b[1*8+i]
		x2 := b[2*8+i]
		x3 := b[3*8+i]
		x4 := b[4*8+i]
		x5 := b[5*8+i]
		x6 := b[6*8+i]
		x7 := b[7*8+i]

		// x[01234567] are UQ8.0 in [0,255].

		// Stage 1: four butterflies.
		// In general a butterfly of QN.M inputs produces Q(N+1).M outputs.
		// A butterfly of UQN.M inputs produces a UQ(N+1).M sum and a QN.M difference.

		x0, x7 = x0+x7, x0-x7
		x1, x6 = x1+x6, x1-x6
		x2, x5 = x2+x5, x2-x5
		x3, x4 = x3+x4, x3-x4
		// x[0123] now UQ9.0 in [0, 510].
		// x[4567] now Q8.0 in [-255,255].

		// Stage 2: two boxes and two butterflies.
		// A box on QN.M inputs with B-bit constants
		// produces Q(N+1).(M+B) outputs.
		// (The +1 is from the addition.)

		x4, x7 = dctBox(x4, x7, c(cos3, 18), c(sin3, 18))
		x5, x6 = dctBox(x5, x6, c(cos1, 18), c(sin1, 18))
		// x[47] now Q9.18 in [-354, 354].
		// x[56] now Q9.18 in [-300, 300].

		x0, x3 = x0+x3, x0-x3
		x1, x2 = x1+x2, x1-x2
		// x[01] now UQ10.0 in [0, 1020].
		// x[23] now Q9.0 in [-510, 510].

		// Stage 3: one box and three butterflies.

		x2, x3 = dctBox(x2, x3, c(sqrt2_cos6, 18), c(sqrt2_sin6, 18))
		// x[23] now Q10.18 in [-943, 943].

		x0, x1 = x0+x1, x0-x1
		// x0 now UQ11.0 in [0, 2040].
		// x1 now Q10.0 in [-1020, 1020].

		// Store x0, x1, x2, x3 to their permuted targets.
		// The original +128 in every input value
		// has cancelled out except in the “DC signal” x0.
		// Subtracting 128*8 here is equivalent to subtracting 128
		// from every input before we started, but cheaper.
		// It also converts x0 from UQ11.18 to Q10.18.
		b[0*8+i] = (x0 - 128*8) << 18
		b[4*8+i] = x1 << 18
		b[2*8+i] = x2
		b[6*8+i] = x3

		x4, x6 = x4+x6, x4-x6
		x7, x5 = x7+x5, x7-x5
		// x[4567] now Q10.18 in [-654, 654].

		// Stage 4: two √2 scalings and one butterfly.

		x5 = (x5 >> 12) * c(sqrt2, 12)
		x6 = (x6 >> 12) * c(sqrt2, 12)
		// x[56] still Q10.18 in [-925, 925] (= 654√2).
		x7, x4 = x7+x4, x7-x4
		// x[47] still Q10.18 in [-925, 925] (not Q11.18!).
		// This is not obvious at all! See “Note on 925” below.

		// Store x4 x5 x6 x7 to their permuted targets.
		b[1*8+i] = x7
		b[3*8+i] = x5
		b[5*8+i] = x6
		b[7*8+i] = x4
	}
}

// fdctRows applies the 1D DCT to the rows of b.
// Inputs are Q10.18; outputs are Q13.0.
func fdctRows(b *block) {
	for i := range 8 {
		x := b[8*i : 8*i+8 : 8*i+8]
		x0 := x[0]
		x1 := x[1]
		x2 := x[2]
		x3 := x[3]
		x4 := x[4]
		x5 := x[5]
		x6 := x[6]
		x7 := x[7]

		// x[01234567] are Q10.18 [-1020, 1020].

		// Stage 1: four butterflies.

		x0, x7 = x0+x7, x0-x7
		x1, x6 = x1+x6, x1-x6
		x2, x5 = x2+x5, x2-x5
		x3, x4 = x3+x4, x3-x4
		// x[01234567] now Q11.18 in [-2040, 2040].

		// Stage 2: two boxes and two butterflies.

		x4, x7 = dctBox(x4>>14, x7>>14, c(cos3, 14), c(sin3, 14))
		x5, x6 = dctBox(x5>>14, x6>>14, c(cos1, 14), c(sin1, 14))
		// x[47] now Q12.18 in [-2830, 2830].
		// x[56] now Q12.18 in [-2400, 2400].
		x0, x3 = x0+x3, x0-x3
		x1, x2 = x1+x2, x1-x2
		// x[01234567] now Q12.18 in [-4080, 4080].

		// Stage 3: one box and three butterflies.

		x2, x3 = dctBox(x2>>14, x3>>14, c(sqrt2_cos6, 14), c(sqrt2_sin6, 14))
		// x[23] now Q13.18 in [-7539, 7539].
		x0, x1 = x0+x1, x0-x1
		// x[01] now Q13.18 in [-8160, 8160].
		x4, x6 = x4+x6, x4-x6
		x7, x5 = x7+x5, x7-x5
		// x[4567] now Q13.18 in [-5230, 5230].

		// Stage 4: two √2 scalings and one butterfly.

		x5 = (x5 >> 14) * c(sqrt2, 14)
		x6 = (x6 >> 14) * c(sqrt2, 14)
		// x[56] still Q13.18 in [-7397, 7397] (= 5230√2).
		x7, x4 = x7+x4, x7-x4
		// x[47] still Q13.18 in [-7395, 7395] (= 2040*3.6246).
		// See “Note on 925” below.

		// Cut from Q13.18 to Q13.0.
		x0 = (x0 + 1<<17) >> 18
		x1 = (x1 + 1<<17) >> 18
		x2 = (x2 + 1<<17) >> 18
		x3 = (x3 + 1<<17) >> 18
		x4 = (x4 + 1<<17) >> 18
		x5 = (x5 + 1<<17) >> 18
		x6 = (x6 + 1<<17) >> 18
		x7 = (x7 + 1<<17) >> 18

		// Note: Unlike in fdctCols, saved all stores for the end
		// because they are adjacent memory locations and some systems
		// can use multiword stores.
		x[0] = x0
		x[1] = x7
		x[2] = x2
		x[3] = x5
		x[4] = x1
		x[5] = x6
		x[6] = x3
		x[7] = x4
	}
}

// “Note on 925”, deferred from above to avoid interrupting code.
//
// In fdctCols, heading into stage 2, the values x4, x5, x6, x7 are in [-255, 255].
// Let's call those specific values b4, b5, b6, b7, and trace how x[4567] evolve:
//
// Stage 2:
//	x4 = b4*cos3 + b7*sin3
//	x7 = -b4*sin3 + b7*cos3
//	x5 = b5*cos1 + b6*sin1
//	x6 = -b5*sin1 + b6*cos1
//
// Stage 3:
//
//	x4 = x4+x6 =  b4*cos3 + b7*sin3 - b5*sin1 + b6*cos1
//	x6 = x4-x6 =  b4*cos3 + b7*sin3 + b5*sin1 - b6*cos1
//	x7 = x7+x5 = -b4*sin3 + b7*cos3 + b5*cos1 + b6*sin1
//	x5 = x7-x5 = -b4*sin3 + b7*cos3 - b5*cos1 - b6*sin1
//
// Stage 4:
//
//	x7 = x7+x4 = -b4*sin3 + b7*cos3 + b5*cos1 + b6*sin1 + b4*cos3 + b7*sin3 - b5*sin1 + b6*cos1
//	   = b4*(cos3-sin3) + b5*(cos1-sin1) + b6*(cos1+sin1) + b7*(cos3+sin3)
//	   < 255*(0.2759 + 0.7857 + 1.1759 + 1.3871) = 255*3.6246 < 925.
//
//	x4 = x7-x4 = -b4*sin3 + b7*cos3 + b5*cos1 + b6*sin1 - b4*cos3 - b7*sin3 + b5*sin1 - b6*cos1
//	   = -b4*(cos3+sin3) + b5*(cos1+sin1) + b6*(sin1-cos1) + b7*(cos3-sin3)
//	   < same 925.
//
// The fact that x5, x6 are also at most 925 is not a coincidence: we are computing
// the same kinds of numbers for all four, just with different paths to them.
//
// In fdctRows, the same analysis applies, but the initial values are
// in [-2040, 2040] instead of [-255, 255], so the bound is 2040*3.6246 < 7395.

// idct implements the inverse DCT.
// Inputs are UQ8.0; outputs are Q10.3.
func idct(b *block) {
	// A 2D IDCT is a 1D IDCT on rows followed by columns.
	idctRows(b)
	idctCols(b)
}

// idctRows applies the 1D IDCT to the rows of b.
// Inputs are UQ8.0; outputs are Q9.20.
func idctRows(b *block) {
	for i := range 8 {
		x := b[8*i : 8*i+8 : 8*i+8]
		x0 := x[0]
		x7 := x[1]
		x2 := x[2]
		x5 := x[3]
		x1 := x[4]
		x6 := x[5]
		x3 := x[6]
		x4 := x[7]

		// Run FDCT backward.
		// Independent operations have been reordered somewhat
		// to make precision tracking easier.
		//
		// Note that “x0, x1 = x0+x1, x0-x1” is now a reverse butterfly
		// and carries with it an implicit divide by two: the extra bit
		// is added to the precision, not the value size.

		// x[01234567] are UQ8.0 in [0, 255].

		// Stages 4, 3, 2: x0, x1, x2, x3.

		x0 <<= 17
		x1 <<= 17
		// x0, x1 now UQ8.17.
		x0, x1 = x0+x1, x0-x1
		// x0 now UQ8.18 in [0, 255].
		// x1 now Q7.18 in [-127½, 127½].

		// Note: (1/sqrt 2)*((cos 6*pi/16)+(sin 6*pi/16)) < 0.924, so no new high bit.
		x2, x3 = dctBox(x2, x3, c(sqrt2inv_cos6, 18), -c(sqrt2inv_sin6, 18))
		// x[23] now Q8.18 in [-236, 236].
		x1, x2 = x1+x2, x1-x2
		x0, x3 = x0+x3, x0-x3
		// x[0123] now Q8.19 in [-246, 246].

		// Stages 4, 3, 2: x4, x5, x6, x7.

		x4 <<= 7
		x7 <<= 7
		// x[47] now UQ8.7
		x7, x4 = x7+x4, x7-x4
		// x7 now UQ8.8 in [0, 255].
		// x4 now Q7.8 in [-127½, 127½].

		x6 = x6 * c(sqrt2inv, 8)
		x5 = x5 * c(sqrt2inv, 8)
		// x[56] now UQ8.8 in [0, 181].
		// Note that 1/√2 has five 0s in its binary representation after
		// the 8th bit, so this multipliy is actually producing 12 bits of precision.

		x7, x5 = x7+x5, x7-x5
		x4, x6 = x4+x6, x4-x6
		// x[4567] now Q8.9 in [-218, 218].

		x4, x7 = dctBox(x4>>2, x7>>2, c(cos3, 12), -c(sin3, 12))
		x5, x6 = dctBox(x5>>2, x6>>2, c(cos1, 12), -c(sin1, 12))
		// x[4567] now Q9.19 in [-303, 303].

		// Stage 1.

		x0, x7 = x0+x7, x0-x7
		x1, x6 = x1+x6, x1-x6
		x2, x5 = x2+x5, x2-x5
		x3, x4 = x3+x4, x3-x4
		// x[01234567] now Q9.20 in [-275, 275].

		// Note: we don't need all 20 bits of “precision”,
		// but it is faster to let idctCols shift it away as part
		// of other operations rather than downshift here.

		x[0] = x0
		x[1] = x1
		x[2] = x2
		x[3] = x3
		x[4] = x4
		x[5] = x5
		x[6] = x6
		x[7] = x7
	}
}

// idctCols applies the 1D IDCT to the columns of b.
// Inputs are Q9.20.
// Outputs are Q10.3. That is, the result is the IDCT*8.
func idctCols(b *block) {
	for i := range 8 {
		x0 := b[0*8+i]
		x7 := b[1*8+i]
		x2 := b[2*8+i]
		x5 := b[3*8+i]
		x1 := b[4*8+i]
		x6 := b[5*8+i]
		x3 := b[6*8+i]
		x4 := b[7*8+i]

		// x[012345678] are Q9.20.

		// Start by adding 0.5 to x0 (the incoming DC signal).
		// The butterflies will add it to all the other values,
		// and then the final shifts will round properly.
		x0 += 1 << 19

		// Stages 4, 3, 2: x0, x1, x2, x3.

		x0, x1 = (x0+x1)>>2, (x0-x1)>>2
		// x[01] now Q9.19.
		// Note: (1/sqrt 2)*((cos 6*pi/16)+(sin 6*pi/16)) < 1, so no new high bit.
		x2, x3 = dctBox(x2>>13, x3>>13, c(sqrt2inv_cos6, 12), -c(sqrt2inv_sin6, 12))
		// x[0123] now Q9.19.

		x1, x2 = x1+x2, x1-x2
		x0, x3 = x0+x3, x0-x3
		// x[0123] now Q9.20.

		// Stages 4, 3, 2: x4, x5, x6, x7.

		x7, x4 = x7+x4, x7-x4
		// x[47] now Q9.21.

		x5 = (x5 >> 13) * c(sqrt2inv, 14)
		x6 = (x6 >> 13) * c(sqrt2inv, 14)
		// x[56] now Q9.21.

		x7, x5 = x7+x5, x7-x5
		x4, x6 = x4+x6, x4-x6
		// x[4567] now Q9.22.

		x4, x7 = dctBox(x4>>14, x7>>14, c(cos3, 12), -c(sin3, 12))
		x5, x6 = dctBox(x5>>14, x6>>14, c(cos1, 12), -c(sin1, 12))
		// x[4567] now Q10.20.

		x0, x7 = x0+x7, x0-x7
		x1, x6 = x1+x6, x1-x6
		x2, x5 = x2+x5, x2-x5
		x3, x4 = x3+x4, x3-x4
		// x[01234567] now Q10.21.

		x0 >>= 18
		x1 >>= 18
		x2 >>= 18
		x3 >>= 18
		x4 >>= 18
		x5 >>= 18
		x6 >>= 18
		x7 >>= 18
		// x[01234567] now Q10.3.

		b[0*8+i] = x0
		b[1*8+i] = x1
		b[2*8+i] = x2
		b[3*8+i] = x3
		b[4*8+i] = x4
		b[5*8+i] = x5
		b[6*8+i] = x6
		b[7*8+i] = x7
	}
}
