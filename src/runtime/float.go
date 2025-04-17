// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

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
