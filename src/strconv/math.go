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

// pow10 returns the 128-bit mantissa and binary exponent of 10**e
// If e is out of range, pow10 returns ok=false.
func pow10(e int) (mant uint128, exp int, ok bool) {
	if e < pow10Min || e > pow10Max {
		return
	}
	return pow10Tab[e-pow10Min], mulLog2_10(e), true
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
