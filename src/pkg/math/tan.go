// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math


/*
	Floating point tangent.
*/

// Tan returns the tangent of x.
func Tan(x float64) float64 {
	// Coefficients are #4285 from Hart & Cheney. (19.74D)
	const (
		P0 = -.1306820264754825668269611177e+5
		P1 = .1055970901714953193602353981e+4
		P2 = -.1550685653483266376941705728e+2
		P3 = .3422554387241003435328470489e-1
		P4 = .3386638642677172096076369e-4
		Q0 = -.1663895238947119001851464661e+5
		Q1 = .4765751362916483698926655581e+4
		Q2 = -.1555033164031709966900124574e+3
	)

	flag := false
	sign := false
	if x < 0 {
		x = -x
		sign = true
	}
	x = x * (4 / Pi) /* overflow? */
	var e float64
	e, x = Modf(x)
	i := int32(e)

	switch i & 3 {
	case 1:
		x = 1 - x
		flag = true

	case 2:
		sign = !sign
		flag = true

	case 3:
		x = 1 - x
		sign = !sign
	}

	xsq := x * x
	temp := ((((P4*xsq+P3)*xsq+P2)*xsq+P1)*xsq + P0) * x
	temp = temp / (((xsq+Q2)*xsq+Q1)*xsq + Q0)

	if flag {
		if temp == 0 {
			return NaN()
		}
		temp = 1 / temp
	}
	if sign {
		temp = -temp
	}
	return temp
}
