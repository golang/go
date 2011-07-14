// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

/*
	Floating-point hyperbolic tangent.

	Sinh and Cosh are called except for large arguments, which
	would cause overflow improperly.
*/

// Tanh computes the hyperbolic tangent of x.
func Tanh(x float64) float64 {
	if x < 0 {
		x = -x
		if x > 21 {
			return -1
		}
		return -Sinh(x) / Cosh(x)
	}
	if x > 21 {
		return 1
	}
	return Sinh(x) / Cosh(x)
}
