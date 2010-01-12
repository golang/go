// Copyright 2009-2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math


/*
	Floating-point mod function.
*/

// Fmod returns the floating-point remainder of x/y.
// The magnitude of the result is less than y and its
// sign agrees with that of x.
//
// Special cases are:
//	if x is not finite, Fmod returns NaN
//	if y is 0 or NaN, Fmod returns NaN
func Fmod(x, y float64) float64 {
	// TODO(rsc): Remove manual inlining of IsNaN, IsInf
	// when compiler does it for us.
	if y == 0 || x > MaxFloat64 || x < -MaxFloat64 || x != x || y != y { // y == 0 || IsInf(x, 0) || IsNaN(x) || IsNan(y)
		return NaN()
	}
	if y < 0 {
		y = -y
	}

	yfr, yexp := Frexp(y)
	sign := false
	r := x
	if x < 0 {
		r = -x
		sign = true
	}

	for r >= y {
		rfr, rexp := Frexp(r)
		if rfr < yfr {
			rexp = rexp - 1
		}
		r = r - Ldexp(y, rexp-yexp)
	}
	if sign {
		r = -r
	}
	return r
}
