// Copyright 2009-2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

/*
	Floating-point mod function.
*/

// Mod returns the floating-point remainder of x/y.
// The magnitude of the result is less than y and its
// sign agrees with that of x.
//
// Special cases are:
//	Mod(±Inf, y) = NaN
//	Mod(NaN, y) = NaN
//	Mod(x, 0) = NaN
//	Mod(x, ±Inf) = x
//	Mod(x, NaN) = NaN
func Mod(x, y float64) float64 {
	if haveArchMod {
		return archMod(x, y)
	}
	return mod(x, y)
}

func mod(x, y float64) float64 {
	if y == 0 || IsInf(x, 0) || IsNaN(x) || IsNaN(y) {
		return NaN()
	}
	y = Abs(y)

	yfr, yexp := Frexp(y)
	r := x
	if x < 0 {
		r = -x
	}

	for r >= y {
		rfr, rexp := Frexp(r)
		if rfr < yfr {
			rexp = rexp - 1
		}
		r = r - Ldexp(y, rexp-yexp)
	}
	if x < 0 {
		r = -r
	}
	return r
}
