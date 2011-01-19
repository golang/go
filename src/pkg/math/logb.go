// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Logb(x) returns the binary exponent of x.
//
// Special cases are:
//	Logb(±Inf) = +Inf
//	Logb(0) = -Inf
//	Logb(NaN) = NaN
func Logb(x float64) float64 {
	// TODO(rsc): Remove manual inlining of IsNaN, IsInf
	// when compiler does it for us
	// special cases
	switch {
	case x == 0:
		return Inf(-1)
	case x < -MaxFloat64 || x > MaxFloat64: // IsInf(x, 0):
		return Inf(1)
	case x != x: // IsNaN(x):
		return x
	}
	return float64(ilogb(x))
}

// Ilogb(x) returns the binary exponent of x as an integer.
//
// Special cases are:
//	Ilogb(±Inf) = MaxInt32
//	Ilogb(0) = MinInt32
//	Ilogb(NaN) = MaxInt32
func Ilogb(x float64) int {
	// TODO(rsc): Remove manual inlining of IsNaN, IsInf
	// when compiler does it for us
	// special cases
	switch {
	case x == 0:
		return MinInt32
	case x != x: // IsNaN(x):
		return MaxInt32
	case x < -MaxFloat64 || x > MaxFloat64: // IsInf(x, 0):
		return MaxInt32
	}
	return ilogb(x)
}

// logb returns the binary exponent of x. It assumes x is finite and
// non-zero.
func ilogb(x float64) int {
	x, exp := normalize(x)
	return int((Float64bits(x)>>shift)&mask) - bias + exp
}
