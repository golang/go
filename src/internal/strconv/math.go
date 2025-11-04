// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import "math/bits"

// A uint128 is a 128-bit uint.
// The fields are exported to make them visible to package strconv_test.
type uint128 struct {
	Hi uint64
	Lo uint64
}

// umul128 returns the 128-bit product x*y.
func umul128(x, y uint64) uint128 {
	hi, lo := bits.Mul64(x, y)
	return uint128{hi, lo}
}

// umul192 returns the 192-bit product x*y in three uint64s.
func umul192(x uint64, y uint128) (hi, mid, lo uint64) {
	mid1, lo := bits.Mul64(x, y.Lo)
	hi, mid2 := bits.Mul64(x, y.Hi)
	mid, carry := bits.Add64(mid1, mid2, 0)
	return hi + carry, mid, lo
}

// pow10 returns the 128-bit mantissa and binary exponent of 10**e.
// That is, 10^e = mant/2^128 * 2**exp.
// If e is out of range, pow10 returns ok=false.
func pow10(e int) (mant uint128, exp int, ok bool) {
	if e < pow10Min || e > pow10Max {
		return
	}
	return pow10Tab[e-pow10Min], 1 + mulLog2_10(e), true
}

// mulLog10_2 returns math.Floor(x * log(2)/log(10)) for an integer x in
// the range -1600 <= x && x <= +1600.
//
// The range restriction lets us work in faster integer arithmetic instead of
// slower floating point arithmetic. Correctness is verified by unit tests.
func mulLog10_2(x int) int {
	// log(2)/log(10) ≈ 0.30102999566 ≈ 78913 / 2^18
	return (x * 78913) >> 18
}

// mulLog2_10 returns math.Floor(x * log(10)/log(2)) for an integer x in
// the range -500 <= x && x <= +500.
//
// The range restriction lets us work in faster integer arithmetic instead of
// slower floating point arithmetic. Correctness is verified by unit tests.
func mulLog2_10(x int) int {
	// log(10)/log(2) ≈ 3.32192809489 ≈ 108853 / 2^15
	return (x * 108853) >> 15
}

func bool2uint(b bool) uint {
	if b {
		return 1
	}
	return 0
}

// Exact Division and Remainder Checking
//
// An exact division x/c (exact means x%c == 0)
// can be implemented by x*m where m is the multiplicative inverse of c (m*c == 1).
//
// Since c is also the multiplicative inverse of m, x*m is lossless,
// and all the exact multiples of c map to all of [0, maxUint64/c].
// The non-multiples are forced to map to larger values.
// This also gives a quick test for whether x is an exact multiple of c:
// compute the exact division and check whether it's at most maxUint64/c:
//	x%c == 0 => x*m <= maxUint64/c.
//
// Only odd c have multiplicative inverses mod powers of two.
// To do an exact divide x / (c<<s) we can use (x/c)>>s instead.
// And to check for remainder, we need to check that those low s
// bits are all zero before we shift them away. We can merge that
// with the <= for the exact odd remainder check by rotating the
// shifted bits into the high part instead:
// 	x%(c<<s) == 0 => bits.RotateLeft64(x*m, -s) <= maxUint64/c.
//
// The compiler does this transformation automatically in general,
// but we apply it here by hand in a few ways that the compiler can't help with.
//
// For a more detailed explanation, see
// Henry S. Warren, Jr., Hacker's Delight, 2nd ed., sections 10-16 and 10-17.

// divisiblePow5 reports whether x is divisible by 5^p.
// It returns false for p not in [1, 22],
// because we only care about float64 mantissas, and 5^23 > 2^53.
func divisiblePow5(x uint64, p int) bool {
	return 1 <= p && p <= 22 && x*div5Tab[p-1][0] <= div5Tab[p-1][1]
}

const maxUint64 = 1<<64 - 1

// div5Tab[p-1] is the multiplicative inverse of 5^p and maxUint64/5^p.
var div5Tab = [22][2]uint64{
	{0xcccccccccccccccd, maxUint64 / 5},
	{0x8f5c28f5c28f5c29, maxUint64 / 5 / 5},
	{0x1cac083126e978d5, maxUint64 / 5 / 5 / 5},
	{0xd288ce703afb7e91, maxUint64 / 5 / 5 / 5 / 5},
	{0x5d4e8fb00bcbe61d, maxUint64 / 5 / 5 / 5 / 5 / 5},
	{0x790fb65668c26139, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5},
	{0xe5032477ae8d46a5, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5 / 5},
	{0xc767074b22e90e21, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5},
	{0x8e47ce423a2e9c6d, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5},
	{0x4fa7f60d3ed61f49, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5},
	{0x0fee64690c913975, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5},
	{0x3662e0e1cf503eb1, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5},
	{0xa47a2cf9f6433fbd, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5},
	{0x54186f653140a659, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5},
	{0x7738164770402145, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5},
	{0xe4a4d1417cd9a041, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5},
	{0xc75429d9e5c5200d, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5},
	{0xc1773b91fac10669, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5},
	{0x26b172506559ce15, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5},
	{0xd489e3a9addec2d1, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5},
	{0x90e860bb892c8d5d, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5},
	{0x502e79bf1b6f4f79, maxUint64 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5},
}

// trimZeros trims trailing zeros from x.
// It finds the largest p such that x % 10^p == 0
// and then returns x / 10^p, p.
//
// This is here for reference and tested, because it is an optimization
// used by other ftoa algorithms, but in our implementations it has
// never been benchmarked to be faster than trimming zeros after
// formatting into decimal bytes.
func trimZeros(x uint64) (uint64, int) {
	const (
		div1e8m  = 0xc767074b22e90e21
		div1e8le = maxUint64 / 100000000

		div1e4m  = 0xd288ce703afb7e91
		div1e4le = maxUint64 / 10000

		div1e2m  = 0x8f5c28f5c28f5c29
		div1e2le = maxUint64 / 100

		div1e1m  = 0xcccccccccccccccd
		div1e1le = maxUint64 / 10
	)

	// _ = assert[x - y] asserts at compile time that x == y.
	// Assert that the multiplicative inverses are correct
	// by checking that (div1eNm * 5^N) % 1<<64 == 1.
	var assert [1]struct{}
	_ = assert[(div1e8m*5*5*5*5*5*5*5*5)%(1<<64)-1]
	_ = assert[(div1e4m*5*5*5*5)%(1<<64)-1]
	_ = assert[(div1e2m*5*5)%(1<<64)-1]
	_ = assert[(div1e1m*5)%(1<<64)-1]

	// Cut 8 zeros, then 4, then 2, then 1.
	p := 0
	for d := bits.RotateLeft64(x*div1e8m, -8); d <= div1e8le; d = bits.RotateLeft64(x*div1e8m, -8) {
		x = d
		p += 8
	}
	if d := bits.RotateLeft64(x*div1e4m, -4); d <= div1e4le {
		x = d
		p += 4
	}
	if d := bits.RotateLeft64(x*div1e2m, -2); d <= div1e2le {
		x = d
		p += 2
	}
	if d := bits.RotateLeft64(x*div1e1m, -1); d <= div1e1le {
		x = d
		p += 1
	}
	return x, p
}
