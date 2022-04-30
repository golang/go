// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && !arm64

package elliptic

import "math/big"

// Field elements are represented as nine, unsigned 32-bit words.
//
// The value of a field element is:
//   x[0] + (x[1] * 2**29) + (x[2] * 2**57) + ... + (x[8] * 2**228)
//
// That is, each limb is alternately 29 or 28-bits wide in little-endian
// order.
//
// This means that a field element hits 2**257, rather than 2**256 as we would
// like. A 28, 29, ... pattern would cause us to hit 2**256, but that causes
// problems when multiplying as terms end up one bit short of a limb which
// would require much bit-shifting to correct.
//
// Finally, the values stored in a field element are in Montgomery form. So the
// value |y| is stored as (y*R) mod p, where p is the P-256 prime and R is
// 2**257.

const (
	p256Limbs    = 9
	bottom29Bits = 0x1fffffff
)

var (
	// p256One is the number 1 as a field element.
	p256One  = [p256Limbs]uint32{2, 0, 0, 0xffff800, 0x1fffffff, 0xfffffff, 0x1fbfffff, 0x1ffffff, 0}
	p256Zero = [p256Limbs]uint32{0, 0, 0, 0, 0, 0, 0, 0, 0}
	// p256P is the prime modulus as a field element.
	p256P = [p256Limbs]uint32{0x1fffffff, 0xfffffff, 0x1fffffff, 0x3ff, 0, 0, 0x200000, 0xf000000, 0xfffffff}
	// p2562P is the twice prime modulus as a field element.
	p2562P = [p256Limbs]uint32{0x1ffffffe, 0xfffffff, 0x1fffffff, 0x7ff, 0, 0, 0x400000, 0xe000000, 0x1fffffff}
)

// Field element operations:

const bottom28Bits = 0xfffffff

// nonZeroToAllOnes returns:
//
//	0xffffffff for 0 < x <= 2**31
//	0 for x == 0 or x > 2**31.
func nonZeroToAllOnes(x uint32) uint32 {
	return ((x - 1) >> 31) - 1
}

// p256ReduceCarry adds a multiple of p in order to cancel |carry|,
// which is a term at 2**257.
//
// On entry: carry < 2**3, inout[0,2,...] < 2**29, inout[1,3,...] < 2**28.
// On exit: inout[0,2,..] < 2**30, inout[1,3,...] < 2**29.
func p256ReduceCarry(inout *[p256Limbs]uint32, carry uint32) {
	carry_mask := nonZeroToAllOnes(carry)

	inout[0] += carry << 1
	inout[3] += 0x10000000 & carry_mask
	// carry < 2**3 thus (carry << 11) < 2**14 and we added 2**28 in the
	// previous line therefore this doesn't underflow.
	inout[3] -= carry << 11
	inout[4] += (0x20000000 - 1) & carry_mask
	inout[5] += (0x10000000 - 1) & carry_mask
	inout[6] += (0x20000000 - 1) & carry_mask
	inout[6] -= carry << 22
	// This may underflow if carry is non-zero but, if so, we'll fix it in the
	// next line.
	inout[7] -= 1 & carry_mask
	inout[7] += carry << 25
}

// p256Sum sets out = in+in2.
//
// On entry: in[i]+in2[i] must not overflow a 32-bit word.
// On exit: out[0,2,...] < 2**30, out[1,3,...] < 2**29.
func p256Sum(out, in, in2 *[p256Limbs]uint32) {
	carry := uint32(0)
	for i := 0; ; i++ {
		out[i] = in[i] + in2[i]
		out[i] += carry
		carry = out[i] >> 29
		out[i] &= bottom29Bits

		i++
		if i == p256Limbs {
			break
		}

		out[i] = in[i] + in2[i]
		out[i] += carry
		carry = out[i] >> 28
		out[i] &= bottom28Bits
	}

	p256ReduceCarry(out, carry)
}

const (
	two30m2    = 1<<30 - 1<<2
	two30p13m2 = 1<<30 + 1<<13 - 1<<2
	two31m2    = 1<<31 - 1<<2
	two31m3    = 1<<31 - 1<<3
	two31p24m2 = 1<<31 + 1<<24 - 1<<2
	two30m27m2 = 1<<30 - 1<<27 - 1<<2
)

// p256Zero31 is 0 mod p.
var p256Zero31 = [p256Limbs]uint32{two31m3, two30m2, two31m2, two30p13m2, two31m2, two30m2, two31p24m2, two30m27m2, two31m2}

// p256Diff sets out = in-in2.
//
// On entry: in[0,2,...] < 2**30, in[1,3,...] < 2**29 and
// in2[0,2,...] < 2**30, in2[1,3,...] < 2**29.
// On exit: out[0,2,...] < 2**30, out[1,3,...] < 2**29.
func p256Diff(out, in, in2 *[p256Limbs]uint32) {
	var carry uint32

	for i := 0; ; i++ {
		out[i] = in[i] - in2[i]
		out[i] += p256Zero31[i]
		out[i] += carry
		carry = out[i] >> 29
		out[i] &= bottom29Bits

		i++
		if i == p256Limbs {
			break
		}

		out[i] = in[i] - in2[i]
		out[i] += p256Zero31[i]
		out[i] += carry
		carry = out[i] >> 28
		out[i] &= bottom28Bits
	}

	p256ReduceCarry(out, carry)
}

// p256ReduceDegree sets out = tmp/R mod p where tmp contains 64-bit words with
// the same 29,28,... bit positions as a field element.
//
// The values in field elements are in Montgomery form: x*R mod p where R =
// 2**257. Since we just multiplied two Montgomery values together, the result
// is x*y*R*R mod p. We wish to divide by R in order for the result also to be
// in Montgomery form.
//
// On entry: tmp[i] < 2**64.
// On exit: out[0,2,...] < 2**30, out[1,3,...] < 2**29.
func p256ReduceDegree(out *[p256Limbs]uint32, tmp [17]uint64) {
	// The following table may be helpful when reading this code:
	//
	// Limb number:   0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10...
	// Width (bits):  29| 28| 29| 28| 29| 28| 29| 28| 29| 28| 29
	// Start bit:     0 | 29| 57| 86|114|143|171|200|228|257|285
	//   (odd phase): 0 | 28| 57| 85|114|142|171|199|228|256|285
	var tmp2 [18]uint32
	var carry, x, xMask uint32

	// tmp contains 64-bit words with the same 29,28,29-bit positions as a
	// field element. So the top of an element of tmp might overlap with
	// another element two positions down. The following loop eliminates
	// this overlap.
	tmp2[0] = uint32(tmp[0]) & bottom29Bits

	tmp2[1] = uint32(tmp[0]) >> 29
	tmp2[1] |= (uint32(tmp[0]>>32) << 3) & bottom28Bits
	tmp2[1] += uint32(tmp[1]) & bottom28Bits
	carry = tmp2[1] >> 28
	tmp2[1] &= bottom28Bits

	for i := 2; i < 17; i++ {
		tmp2[i] = (uint32(tmp[i-2] >> 32)) >> 25
		tmp2[i] += (uint32(tmp[i-1])) >> 28
		tmp2[i] += (uint32(tmp[i-1]>>32) << 4) & bottom29Bits
		tmp2[i] += uint32(tmp[i]) & bottom29Bits
		tmp2[i] += carry
		carry = tmp2[i] >> 29
		tmp2[i] &= bottom29Bits

		i++
		if i == 17 {
			break
		}
		tmp2[i] = uint32(tmp[i-2]>>32) >> 25
		tmp2[i] += uint32(tmp[i-1]) >> 29
		tmp2[i] += ((uint32(tmp[i-1] >> 32)) << 3) & bottom28Bits
		tmp2[i] += uint32(tmp[i]) & bottom28Bits
		tmp2[i] += carry
		carry = tmp2[i] >> 28
		tmp2[i] &= bottom28Bits
	}

	tmp2[17] = uint32(tmp[15]>>32) >> 25
	tmp2[17] += uint32(tmp[16]) >> 29
	tmp2[17] += uint32(tmp[16]>>32) << 3
	tmp2[17] += carry

	// Montgomery elimination of terms:
	//
	// Since R is 2**257, we can divide by R with a bitwise shift if we can
	// ensure that the right-most 257 bits are all zero. We can make that true
	// by adding multiplies of p without affecting the value.
	//
	// So we eliminate limbs from right to left. Since the bottom 29 bits of p
	// are all ones, then by adding tmp2[0]*p to tmp2 we'll make tmp2[0] == 0.
	// We can do that for 8 further limbs and then right shift to eliminate the
	// extra factor of R.
	for i := 0; ; i += 2 {
		tmp2[i+1] += tmp2[i] >> 29
		x = tmp2[i] & bottom29Bits
		xMask = nonZeroToAllOnes(x)
		tmp2[i] = 0

		// The bounds calculations for this loop are tricky. Each iteration of
		// the loop eliminates two words by adding values to words to their
		// right.
		//
		// The following table contains the amounts added to each word (as an
		// offset from the value of i at the top of the loop). The amounts are
		// accounted for from the first and second half of the loop separately
		// and are written as, for example, 28 to mean a value <2**28.
		//
		// Word:                   3   4   5   6   7   8   9   10
		// Added in top half:     28  11      29  21  29  28
		//                                        28  29
		//                                            29
		// Added in bottom half:      29  10      28  21  28   28
		//                                            29
		//
		// The value that is currently offset 7 will be offset 5 for the next
		// iteration and then offset 3 for the iteration after that. Therefore
		// the total value added will be the values added at 7, 5 and 3.
		//
		// The following table accumulates these values. The sums at the bottom
		// are written as, for example, 29+28, to mean a value < 2**29+2**28.
		//
		// Word:                   3   4   5   6   7   8   9  10  11  12  13
		//                        28  11  10  29  21  29  28  28  28  28  28
		//                            29  28  11  28  29  28  29  28  29  28
		//                                    29  28  21  21  29  21  29  21
		//                                        10  29  28  21  28  21  28
		//                                        28  29  28  29  28  29  28
		//                                            11  10  29  10  29  10
		//                                            29  28  11  28  11
		//                                                    29      29
		//                        --------------------------------------------
		//                                                30+ 31+ 30+ 31+ 30+
		//                                                28+ 29+ 28+ 29+ 21+
		//                                                21+ 28+ 21+ 28+ 10
		//                                                10  21+ 10  21+
		//                                                    11      11
		//
		// So the greatest amount is added to tmp2[10] and tmp2[12]. If
		// tmp2[10/12] has an initial value of <2**29, then the maximum value
		// will be < 2**31 + 2**30 + 2**28 + 2**21 + 2**11, which is < 2**32,
		// as required.
		tmp2[i+3] += (x << 10) & bottom28Bits
		tmp2[i+4] += (x >> 18)

		tmp2[i+6] += (x << 21) & bottom29Bits
		tmp2[i+7] += x >> 8

		// At position 200, which is the starting bit position for word 7, we
		// have a factor of 0xf000000 = 2**28 - 2**24.
		tmp2[i+7] += 0x10000000 & xMask
		tmp2[i+8] += (x - 1) & xMask
		tmp2[i+7] -= (x << 24) & bottom28Bits
		tmp2[i+8] -= x >> 4

		tmp2[i+8] += 0x20000000 & xMask
		tmp2[i+8] -= x
		tmp2[i+8] += (x << 28) & bottom29Bits
		tmp2[i+9] += ((x >> 1) - 1) & xMask

		if i+1 == p256Limbs {
			break
		}
		tmp2[i+2] += tmp2[i+1] >> 28
		x = tmp2[i+1] & bottom28Bits
		xMask = nonZeroToAllOnes(x)
		tmp2[i+1] = 0

		tmp2[i+4] += (x << 11) & bottom29Bits
		tmp2[i+5] += (x >> 18)

		tmp2[i+7] += (x << 21) & bottom28Bits
		tmp2[i+8] += x >> 7

		// At position 199, which is the starting bit of the 8th word when
		// dealing with a context starting on an odd word, we have a factor of
		// 0x1e000000 = 2**29 - 2**25. Since we have not updated i, the 8th
		// word from i+1 is i+8.
		tmp2[i+8] += 0x20000000 & xMask
		tmp2[i+9] += (x - 1) & xMask
		tmp2[i+8] -= (x << 25) & bottom29Bits
		tmp2[i+9] -= x >> 4

		tmp2[i+9] += 0x10000000 & xMask
		tmp2[i+9] -= x
		tmp2[i+10] += (x - 1) & xMask
	}

	// We merge the right shift with a carry chain. The words above 2**257 have
	// widths of 28,29,... which we need to correct when copying them down.
	carry = 0
	for i := 0; i < 8; i++ {
		// The maximum value of tmp2[i + 9] occurs on the first iteration and
		// is < 2**30+2**29+2**28. Adding 2**29 (from tmp2[i + 10]) is
		// therefore safe.
		out[i] = tmp2[i+9]
		out[i] += carry
		out[i] += (tmp2[i+10] << 28) & bottom29Bits
		carry = out[i] >> 29
		out[i] &= bottom29Bits

		i++
		out[i] = tmp2[i+9] >> 1
		out[i] += carry
		carry = out[i] >> 28
		out[i] &= bottom28Bits
	}

	out[8] = tmp2[17]
	out[8] += carry
	carry = out[8] >> 29
	out[8] &= bottom29Bits

	p256ReduceCarry(out, carry)
}

// p256Square sets out=in*in.
//
// On entry: in[0,2,...] < 2**30, in[1,3,...] < 2**29.
// On exit: out[0,2,...] < 2**30, out[1,3,...] < 2**29.
func p256Square(out, in *[p256Limbs]uint32) {
	var tmp [17]uint64

	tmp[0] = uint64(in[0]) * uint64(in[0])
	tmp[1] = uint64(in[0]) * (uint64(in[1]) << 1)
	tmp[2] = uint64(in[0])*(uint64(in[2])<<1) +
		uint64(in[1])*(uint64(in[1])<<1)
	tmp[3] = uint64(in[0])*(uint64(in[3])<<1) +
		uint64(in[1])*(uint64(in[2])<<1)
	tmp[4] = uint64(in[0])*(uint64(in[4])<<1) +
		uint64(in[1])*(uint64(in[3])<<2) +
		uint64(in[2])*uint64(in[2])
	tmp[5] = uint64(in[0])*(uint64(in[5])<<1) +
		uint64(in[1])*(uint64(in[4])<<1) +
		uint64(in[2])*(uint64(in[3])<<1)
	tmp[6] = uint64(in[0])*(uint64(in[6])<<1) +
		uint64(in[1])*(uint64(in[5])<<2) +
		uint64(in[2])*(uint64(in[4])<<1) +
		uint64(in[3])*(uint64(in[3])<<1)
	tmp[7] = uint64(in[0])*(uint64(in[7])<<1) +
		uint64(in[1])*(uint64(in[6])<<1) +
		uint64(in[2])*(uint64(in[5])<<1) +
		uint64(in[3])*(uint64(in[4])<<1)
	// tmp[8] has the greatest value of 2**61 + 2**60 + 2**61 + 2**60 + 2**60,
	// which is < 2**64 as required.
	tmp[8] = uint64(in[0])*(uint64(in[8])<<1) +
		uint64(in[1])*(uint64(in[7])<<2) +
		uint64(in[2])*(uint64(in[6])<<1) +
		uint64(in[3])*(uint64(in[5])<<2) +
		uint64(in[4])*uint64(in[4])
	tmp[9] = uint64(in[1])*(uint64(in[8])<<1) +
		uint64(in[2])*(uint64(in[7])<<1) +
		uint64(in[3])*(uint64(in[6])<<1) +
		uint64(in[4])*(uint64(in[5])<<1)
	tmp[10] = uint64(in[2])*(uint64(in[8])<<1) +
		uint64(in[3])*(uint64(in[7])<<2) +
		uint64(in[4])*(uint64(in[6])<<1) +
		uint64(in[5])*(uint64(in[5])<<1)
	tmp[11] = uint64(in[3])*(uint64(in[8])<<1) +
		uint64(in[4])*(uint64(in[7])<<1) +
		uint64(in[5])*(uint64(in[6])<<1)
	tmp[12] = uint64(in[4])*(uint64(in[8])<<1) +
		uint64(in[5])*(uint64(in[7])<<2) +
		uint64(in[6])*uint64(in[6])
	tmp[13] = uint64(in[5])*(uint64(in[8])<<1) +
		uint64(in[6])*(uint64(in[7])<<1)
	tmp[14] = uint64(in[6])*(uint64(in[8])<<1) +
		uint64(in[7])*(uint64(in[7])<<1)
	tmp[15] = uint64(in[7]) * (uint64(in[8]) << 1)
	tmp[16] = uint64(in[8]) * uint64(in[8])

	p256ReduceDegree(out, tmp)
}

// p256Mul sets out=in*in2.
//
// On entry: in[0,2,...] < 2**30, in[1,3,...] < 2**29 and
// in2[0,2,...] < 2**30, in2[1,3,...] < 2**29.
// On exit: out[0,2,...] < 2**30, out[1,3,...] < 2**29.
func p256Mul(out, in, in2 *[p256Limbs]uint32) {
	var tmp [17]uint64

	tmp[0] = uint64(in[0]) * uint64(in2[0])
	tmp[1] = uint64(in[0])*(uint64(in2[1])<<0) +
		uint64(in[1])*(uint64(in2[0])<<0)
	tmp[2] = uint64(in[0])*(uint64(in2[2])<<0) +
		uint64(in[1])*(uint64(in2[1])<<1) +
		uint64(in[2])*(uint64(in2[0])<<0)
	tmp[3] = uint64(in[0])*(uint64(in2[3])<<0) +
		uint64(in[1])*(uint64(in2[2])<<0) +
		uint64(in[2])*(uint64(in2[1])<<0) +
		uint64(in[3])*(uint64(in2[0])<<0)
	tmp[4] = uint64(in[0])*(uint64(in2[4])<<0) +
		uint64(in[1])*(uint64(in2[3])<<1) +
		uint64(in[2])*(uint64(in2[2])<<0) +
		uint64(in[3])*(uint64(in2[1])<<1) +
		uint64(in[4])*(uint64(in2[0])<<0)
	tmp[5] = uint64(in[0])*(uint64(in2[5])<<0) +
		uint64(in[1])*(uint64(in2[4])<<0) +
		uint64(in[2])*(uint64(in2[3])<<0) +
		uint64(in[3])*(uint64(in2[2])<<0) +
		uint64(in[4])*(uint64(in2[1])<<0) +
		uint64(in[5])*(uint64(in2[0])<<0)
	tmp[6] = uint64(in[0])*(uint64(in2[6])<<0) +
		uint64(in[1])*(uint64(in2[5])<<1) +
		uint64(in[2])*(uint64(in2[4])<<0) +
		uint64(in[3])*(uint64(in2[3])<<1) +
		uint64(in[4])*(uint64(in2[2])<<0) +
		uint64(in[5])*(uint64(in2[1])<<1) +
		uint64(in[6])*(uint64(in2[0])<<0)
	tmp[7] = uint64(in[0])*(uint64(in2[7])<<0) +
		uint64(in[1])*(uint64(in2[6])<<0) +
		uint64(in[2])*(uint64(in2[5])<<0) +
		uint64(in[3])*(uint64(in2[4])<<0) +
		uint64(in[4])*(uint64(in2[3])<<0) +
		uint64(in[5])*(uint64(in2[2])<<0) +
		uint64(in[6])*(uint64(in2[1])<<0) +
		uint64(in[7])*(uint64(in2[0])<<0)
	// tmp[8] has the greatest value but doesn't overflow. See logic in
	// p256Square.
	tmp[8] = uint64(in[0])*(uint64(in2[8])<<0) +
		uint64(in[1])*(uint64(in2[7])<<1) +
		uint64(in[2])*(uint64(in2[6])<<0) +
		uint64(in[3])*(uint64(in2[5])<<1) +
		uint64(in[4])*(uint64(in2[4])<<0) +
		uint64(in[5])*(uint64(in2[3])<<1) +
		uint64(in[6])*(uint64(in2[2])<<0) +
		uint64(in[7])*(uint64(in2[1])<<1) +
		uint64(in[8])*(uint64(in2[0])<<0)
	tmp[9] = uint64(in[1])*(uint64(in2[8])<<0) +
		uint64(in[2])*(uint64(in2[7])<<0) +
		uint64(in[3])*(uint64(in2[6])<<0) +
		uint64(in[4])*(uint64(in2[5])<<0) +
		uint64(in[5])*(uint64(in2[4])<<0) +
		uint64(in[6])*(uint64(in2[3])<<0) +
		uint64(in[7])*(uint64(in2[2])<<0) +
		uint64(in[8])*(uint64(in2[1])<<0)
	tmp[10] = uint64(in[2])*(uint64(in2[8])<<0) +
		uint64(in[3])*(uint64(in2[7])<<1) +
		uint64(in[4])*(uint64(in2[6])<<0) +
		uint64(in[5])*(uint64(in2[5])<<1) +
		uint64(in[6])*(uint64(in2[4])<<0) +
		uint64(in[7])*(uint64(in2[3])<<1) +
		uint64(in[8])*(uint64(in2[2])<<0)
	tmp[11] = uint64(in[3])*(uint64(in2[8])<<0) +
		uint64(in[4])*(uint64(in2[7])<<0) +
		uint64(in[5])*(uint64(in2[6])<<0) +
		uint64(in[6])*(uint64(in2[5])<<0) +
		uint64(in[7])*(uint64(in2[4])<<0) +
		uint64(in[8])*(uint64(in2[3])<<0)
	tmp[12] = uint64(in[4])*(uint64(in2[8])<<0) +
		uint64(in[5])*(uint64(in2[7])<<1) +
		uint64(in[6])*(uint64(in2[6])<<0) +
		uint64(in[7])*(uint64(in2[5])<<1) +
		uint64(in[8])*(uint64(in2[4])<<0)
	tmp[13] = uint64(in[5])*(uint64(in2[8])<<0) +
		uint64(in[6])*(uint64(in2[7])<<0) +
		uint64(in[7])*(uint64(in2[6])<<0) +
		uint64(in[8])*(uint64(in2[5])<<0)
	tmp[14] = uint64(in[6])*(uint64(in2[8])<<0) +
		uint64(in[7])*(uint64(in2[7])<<1) +
		uint64(in[8])*(uint64(in2[6])<<0)
	tmp[15] = uint64(in[7])*(uint64(in2[8])<<0) +
		uint64(in[8])*(uint64(in2[7])<<0)
	tmp[16] = uint64(in[8]) * (uint64(in2[8]) << 0)

	p256ReduceDegree(out, tmp)
}

func p256Assign(out, in *[p256Limbs]uint32) {
	*out = *in
}

// p256Invert calculates |out| = |in|^{-1}
//
// Based on Fermat's Little Theorem:
//
//	a^p = a (mod p)
//	a^{p-1} = 1 (mod p)
//	a^{p-2} = a^{-1} (mod p)
func p256Invert(out, in *[p256Limbs]uint32) {
	var ftmp, ftmp2 [p256Limbs]uint32

	// each e_I will hold |in|^{2^I - 1}
	var e2, e4, e8, e16, e32, e64 [p256Limbs]uint32

	p256Square(&ftmp, in)     // 2^1
	p256Mul(&ftmp, in, &ftmp) // 2^2 - 2^0
	p256Assign(&e2, &ftmp)
	p256Square(&ftmp, &ftmp)   // 2^3 - 2^1
	p256Square(&ftmp, &ftmp)   // 2^4 - 2^2
	p256Mul(&ftmp, &ftmp, &e2) // 2^4 - 2^0
	p256Assign(&e4, &ftmp)
	p256Square(&ftmp, &ftmp)   // 2^5 - 2^1
	p256Square(&ftmp, &ftmp)   // 2^6 - 2^2
	p256Square(&ftmp, &ftmp)   // 2^7 - 2^3
	p256Square(&ftmp, &ftmp)   // 2^8 - 2^4
	p256Mul(&ftmp, &ftmp, &e4) // 2^8 - 2^0
	p256Assign(&e8, &ftmp)
	for i := 0; i < 8; i++ {
		p256Square(&ftmp, &ftmp)
	} // 2^16 - 2^8
	p256Mul(&ftmp, &ftmp, &e8) // 2^16 - 2^0
	p256Assign(&e16, &ftmp)
	for i := 0; i < 16; i++ {
		p256Square(&ftmp, &ftmp)
	} // 2^32 - 2^16
	p256Mul(&ftmp, &ftmp, &e16) // 2^32 - 2^0
	p256Assign(&e32, &ftmp)
	for i := 0; i < 32; i++ {
		p256Square(&ftmp, &ftmp)
	} // 2^64 - 2^32
	p256Assign(&e64, &ftmp)
	p256Mul(&ftmp, &ftmp, in) // 2^64 - 2^32 + 2^0
	for i := 0; i < 192; i++ {
		p256Square(&ftmp, &ftmp)
	} // 2^256 - 2^224 + 2^192

	p256Mul(&ftmp2, &e64, &e32) // 2^64 - 2^0
	for i := 0; i < 16; i++ {
		p256Square(&ftmp2, &ftmp2)
	} // 2^80 - 2^16
	p256Mul(&ftmp2, &ftmp2, &e16) // 2^80 - 2^0
	for i := 0; i < 8; i++ {
		p256Square(&ftmp2, &ftmp2)
	} // 2^88 - 2^8
	p256Mul(&ftmp2, &ftmp2, &e8) // 2^88 - 2^0
	for i := 0; i < 4; i++ {
		p256Square(&ftmp2, &ftmp2)
	} // 2^92 - 2^4
	p256Mul(&ftmp2, &ftmp2, &e4) // 2^92 - 2^0
	p256Square(&ftmp2, &ftmp2)   // 2^93 - 2^1
	p256Square(&ftmp2, &ftmp2)   // 2^94 - 2^2
	p256Mul(&ftmp2, &ftmp2, &e2) // 2^94 - 2^0
	p256Square(&ftmp2, &ftmp2)   // 2^95 - 2^1
	p256Square(&ftmp2, &ftmp2)   // 2^96 - 2^2
	p256Mul(&ftmp2, &ftmp2, in)  // 2^96 - 3

	p256Mul(out, &ftmp2, &ftmp) // 2^256 - 2^224 + 2^192 + 2^96 - 3
}

// p256Scalar3 sets out=3*out.
//
// On entry: out[0,2,...] < 2**30, out[1,3,...] < 2**29.
// On exit: out[0,2,...] < 2**30, out[1,3,...] < 2**29.
func p256Scalar3(out *[p256Limbs]uint32) {
	var carry uint32

	for i := 0; ; i++ {
		out[i] *= 3
		out[i] += carry
		carry = out[i] >> 29
		out[i] &= bottom29Bits

		i++
		if i == p256Limbs {
			break
		}

		out[i] *= 3
		out[i] += carry
		carry = out[i] >> 28
		out[i] &= bottom28Bits
	}

	p256ReduceCarry(out, carry)
}

// p256Scalar4 sets out=4*out.
//
// On entry: out[0,2,...] < 2**30, out[1,3,...] < 2**29.
// On exit: out[0,2,...] < 2**30, out[1,3,...] < 2**29.
func p256Scalar4(out *[p256Limbs]uint32) {
	var carry, nextCarry uint32

	for i := 0; ; i++ {
		nextCarry = out[i] >> 27
		out[i] <<= 2
		out[i] &= bottom29Bits
		out[i] += carry
		carry = nextCarry + (out[i] >> 29)
		out[i] &= bottom29Bits

		i++
		if i == p256Limbs {
			break
		}
		nextCarry = out[i] >> 26
		out[i] <<= 2
		out[i] &= bottom28Bits
		out[i] += carry
		carry = nextCarry + (out[i] >> 28)
		out[i] &= bottom28Bits
	}

	p256ReduceCarry(out, carry)
}

// p256Scalar8 sets out=8*out.
//
// On entry: out[0,2,...] < 2**30, out[1,3,...] < 2**29.
// On exit: out[0,2,...] < 2**30, out[1,3,...] < 2**29.
func p256Scalar8(out *[p256Limbs]uint32) {
	var carry, nextCarry uint32

	for i := 0; ; i++ {
		nextCarry = out[i] >> 26
		out[i] <<= 3
		out[i] &= bottom29Bits
		out[i] += carry
		carry = nextCarry + (out[i] >> 29)
		out[i] &= bottom29Bits

		i++
		if i == p256Limbs {
			break
		}
		nextCarry = out[i] >> 25
		out[i] <<= 3
		out[i] &= bottom28Bits
		out[i] += carry
		carry = nextCarry + (out[i] >> 28)
		out[i] &= bottom28Bits
	}

	p256ReduceCarry(out, carry)
}

// p256CopyConditional sets out=in if mask = 0xffffffff in constant time.
//
// On entry: mask is either 0 or 0xffffffff.
func p256CopyConditional(out, in *[p256Limbs]uint32, mask uint32) {
	for i := 0; i < p256Limbs; i++ {
		tmp := mask & (in[i] ^ out[i])
		out[i] ^= tmp
	}
}

// p256FromBig sets out = R*in.
func p256FromBig(out *[p256Limbs]uint32, in *big.Int) {
	tmp := new(big.Int).Lsh(in, 257)
	tmp.Mod(tmp, p256Params.P)

	for i := 0; i < p256Limbs; i++ {
		if bits := tmp.Bits(); len(bits) > 0 {
			out[i] = uint32(bits[0]) & bottom29Bits
		} else {
			out[i] = 0
		}
		tmp.Rsh(tmp, 29)

		i++
		if i == p256Limbs {
			break
		}

		if bits := tmp.Bits(); len(bits) > 0 {
			out[i] = uint32(bits[0]) & bottom28Bits
		} else {
			out[i] = 0
		}
		tmp.Rsh(tmp, 28)
	}
}

// p256ToBig returns a *big.Int containing the value of in.
func p256ToBig(in *[p256Limbs]uint32) *big.Int {
	result, tmp := new(big.Int), new(big.Int)

	result.SetInt64(int64(in[p256Limbs-1]))
	for i := p256Limbs - 2; i >= 0; i-- {
		if (i & 1) == 0 {
			result.Lsh(result, 29)
		} else {
			result.Lsh(result, 28)
		}
		tmp.SetInt64(int64(in[i]))
		result.Add(result, tmp)
	}

	result.Mul(result, p256RInverse)
	result.Mod(result, p256Params.P)
	return result
}
