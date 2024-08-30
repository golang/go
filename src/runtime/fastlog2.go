// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// fastlog2 implements a fast approximation to the base 2 log of a
// float64. This is used to compute a geometric distribution for heap
// sampling, without introducing dependencies into package math. This
// uses a very rough approximation using the float64 exponent and the
// first 25 bits of the mantissa. The top 5 bits of the mantissa are
// used to load limits from a table of constants and the rest are used
// to scale linearly between them.
func fastlog2(x float64) float64 {
	const fastlogScaleBits = 20
	const fastlogScaleRatio = 1.0 / (1 << fastlogScaleBits)

	xBits := float64bits(x)
	// Extract the exponent from the IEEE float64, and index a constant
	// table with the first 10 bits from the mantissa.
	xExp := int64((xBits>>52)&0x7FF) - 1023
	xManIndex := (xBits >> (52 - fastlogNumBits)) % (1 << fastlogNumBits)
	xManScale := (xBits >> (52 - fastlogNumBits - fastlogScaleBits)) % (1 << fastlogScaleBits)

	low, high := fastlog2Table[xManIndex], fastlog2Table[xManIndex+1]
	return float64(xExp) + low + (high-low)*float64(xManScale)*fastlogScaleRatio
}
