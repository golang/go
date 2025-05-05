// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	float64Mask  = 0x7FF
	float64Shift = 64 - 11 - 1
	float64Bias  = 1023
)

var inf = float64frombits(0x7FF0000000000000)

// isNaN reports whether f is an IEEE 754 “not-a-number” value.
func isNaN(f float64) (is bool) {
	// IEEE 754 says that only NaNs satisfy f != f.
	return f != f
}

// isFinite reports whether f is neither NaN nor an infinity.
func isFinite(f float64) bool {
	return !isNaN(f - f)
}

// isInf reports whether f is an infinity.
func isInf(f float64) bool {
	return !isNaN(f) && !isFinite(f)
}

// abs returns the absolute value of x.
//
// Special cases are:
//
//	abs(±Inf) = +Inf
//	abs(NaN) = NaN
func abs(x float64) float64 {
	const sign = 1 << 63
	return float64frombits(float64bits(x) &^ sign)
}

// copysign returns a value with the magnitude
// of x and the sign of y.
func copysign(x, y float64) float64 {
	const sign = 1 << 63
	return float64frombits(float64bits(x)&^sign | float64bits(y)&sign)
}

// float64bits returns the IEEE 754 binary representation of f.
func float64bits(f float64) uint64 {
	return *(*uint64)(unsafe.Pointer(&f))
}

// float64frombits returns the floating point number corresponding
// the IEEE 754 binary representation b.
func float64frombits(b uint64) float64 {
	return *(*float64)(unsafe.Pointer(&b))
}

// floor returns the greatest integer value less than or equal to x.
//
// Special cases are:
//
//	floor(±0) = ±0
//	floor(±Inf) = ±Inf
//	floor(NaN) = NaN
//
// N.B. Portable floor copied from math. math also has optimized arch-specific
// implementations.
func floor(x float64) float64 {
	if x == 0 || isNaN(x) || isInf(x) {
		return x
	}
	if x < 0 {
		d, fract := modf(-x)
		if fract != 0.0 {
			d = d + 1
		}
		return -d
	}
	d, _ := modf(x)
	return d
}

// ceil returns the least integer value greater than or equal to x.
//
// Special cases are:
//
//	Ceil(±0) = ±0
//	Ceil(±Inf) = ±Inf
//	Ceil(NaN) = NaN
//
// N.B. Portable ceil copied from math. math also has optimized arch-specific
// implementations.
func ceil(x float64) float64 {
	return -floor(-x)
}

// modf returns integer and fractional floating-point numbers
// that sum to f. Both values have the same sign as f.
//
// Special cases are:
//
//	Modf(±Inf) = ±Inf, NaN
//	Modf(NaN) = NaN, NaN
//
// N.B. Portable modf copied from math. math also has optimized arch-specific
// implementations.
func modf(f float64) (int float64, frac float64) {
	if f < 1 {
		switch {
		case f < 0:
			int, frac = modf(-f)
			return -int, -frac
		case f == 0:
			return f, f // Return -0, -0 when f == -0
		}
		return 0, f
	}

	x := float64bits(f)
	e := uint(x>>float64Shift)&float64Mask - float64Bias

	// Keep the top 12+e bits, the integer part; clear the rest.
	if e < 64-12 {
		x &^= 1<<(64-12-e) - 1
	}
	int = float64frombits(x)
	frac = f - int
	return
}
