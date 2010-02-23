// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Nextafter returns the next representable value after x towards y.
// If x == y, then x is returned.
//
// Special cases are:
//	Nextafter(NaN, y) = NaN
//	Nextafter(x, NaN) = NaN
func Nextafter(x, y float64) (r float64) {
	// TODO(rsc): Remove manual inlining of IsNaN
	// when compiler does it for us
	switch {
	case x != x || y != y: // IsNaN(x) || IsNaN(y): // special case
		r = NaN()
	case x == y:
		r = x
	case x == 0:
		r = Copysign(Float64frombits(1), y)
	case (y > x) == (x > 0):
		r = Float64frombits(Float64bits(x) + 1)
	default:
		r = Float64frombits(Float64bits(x) - 1)
	}
	return r
}
