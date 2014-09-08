// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Atan2 returns the arc tangent of y/x, using
// the signs of the two to determine the quadrant
// of the return value.
//
// Special cases are (in order):
//	Atan2(y, NaN) = NaN
//	Atan2(NaN, x) = NaN
//	Atan2(+0, x>=0) = +0
//	Atan2(-0, x>=0) = -0
//	Atan2(+0, x<=-0) = +Pi
//	Atan2(-0, x<=-0) = -Pi
//	Atan2(y>0, 0) = +Pi/2
//	Atan2(y<0, 0) = -Pi/2
//	Atan2(+Inf, +Inf) = +Pi/4
//	Atan2(-Inf, +Inf) = -Pi/4
//	Atan2(+Inf, -Inf) = 3Pi/4
//	Atan2(-Inf, -Inf) = -3Pi/4
//	Atan2(y, +Inf) = 0
//	Atan2(y>0, -Inf) = +Pi
//	Atan2(y<0, -Inf) = -Pi
//	Atan2(+Inf, x) = +Pi/2
//	Atan2(-Inf, x) = -Pi/2
func Atan2(y, x float64) float64

func atan2(y, x float64) float64 {
	// special cases
	switch {
	case IsNaN(y) || IsNaN(x):
		return NaN()
	case y == 0:
		if x >= 0 && !Signbit(x) {
			return Copysign(0, y)
		}
		return Copysign(Pi, y)
	case x == 0:
		return Copysign(Pi/2, y)
	case IsInf(x, 0):
		if IsInf(x, 1) {
			switch {
			case IsInf(y, 0):
				return Copysign(Pi/4, y)
			default:
				return Copysign(0, y)
			}
		}
		switch {
		case IsInf(y, 0):
			return Copysign(3*Pi/4, y)
		default:
			return Copysign(Pi, y)
		}
	case IsInf(y, 0):
		return Copysign(Pi/2, y)
	}

	// Call atan and determine the quadrant.
	q := Atan(y / x)
	if x < 0 {
		if q <= 0 {
			return q + Pi
		}
		return q - Pi
	}
	return q
}
