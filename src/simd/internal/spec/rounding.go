// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//specgen:category Rounding

package spec

import "math"

// Ceil returns the least integer value greater than or equal to each element
// of x.
//
//	z[i] = ceil(x[i])
//
// Special cases follow IEEE 754: Ceil(±0) = ±0, Ceil(±Inf) = ±Inf, Ceil(x) = -0
// for -1 < x < -0, and Ceil(NaN) = NaN. The sign and payload of a NaN result
// are unspecified.
func Ceil[E Floats, W Width](x Vec[E, W]) (z Vec[E, W]) {
	return map1[E, W, E, W](x, func(x E) E {
		return E(math.Ceil(float64(x)))
	})
}

// Floor returns the greatest integer value less than or equal to each element
// of x.
//
//	z[i] = floor(x[i])
//
// Special cases follow IEEE 754: Floor(±0) = ±0, Floor(±Inf) = ±Inf, and
// Floor(NaN) = NaN. The sign and payload of a NaN result are unspecified.
func Floor[E Floats, W Width](x Vec[E, W]) (z Vec[E, W]) {
	return map1[E, W, E, W](x, func(x E) E {
		return E(math.Floor(float64(x)))
	})
}

// Round returns the nearest integer value to each element of x, rounding ties
// to even.
//
//	z[i] = roundToEven(x[i])
//
// Special cases follow IEEE 754: Round(±0) = ±0, Round(±Inf) = ±Inf,
// Round(x) = -0 for -0.5 <= x < -0, and Round(NaN) = NaN. The sign and payload
// of a NaN result are unspecified.
func Round[E Floats, W Width](x Vec[E, W]) (z Vec[E, W]) {
	return map1[E, W, E, W](x, func(x E) E {
		return E(math.RoundToEven(float64(x)))
	})
}

// Trunc returns the integer value of each element of x, rounded toward zero.
//
//	z[i] = trunc(x[i])
//
// Special cases follow IEEE 754: Trunc(±0) = ±0, Trunc(±Inf) = ±Inf,
// Trunc(x) = -0 for -1 < x < -0, and Trunc(NaN) = NaN. The sign and payload of
// a NaN result are unspecified.
func Trunc[E Floats, W Width](x Vec[E, W]) (z Vec[E, W]) {
	return map1[E, W, E, W](x, func(x E) E {
		return E(math.Trunc(float64(x)))
	})
}
