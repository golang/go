// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

/*
	Floating-point hyperbolic sine and cosine.

	The exponential func is called for arguments
	greater in magnitude than 0.5.

	A series is used for arguments smaller in magnitude than 0.5.

	Cosh(x) is computed from the exponential func for
	all arguments.
*/

// Sinh returns the hyperbolic sine of x.
//
// Special cases are:
//	Sinh(±0) = ±0
//	Sinh(±Inf) = ±Inf
//	Sinh(NaN) = NaN
func Sinh(x float64) float64 {
	// The coefficients are #2029 from Hart & Cheney. (20.36D)
	const (
		P0 = -0.6307673640497716991184787251e+6
		P1 = -0.8991272022039509355398013511e+5
		P2 = -0.2894211355989563807284660366e+4
		P3 = -0.2630563213397497062819489e+2
		Q0 = -0.6307673640497716991212077277e+6
		Q1 = 0.1521517378790019070696485176e+5
		Q2 = -0.173678953558233699533450911e+3
	)

	sign := false
	if x < 0 {
		x = -x
		sign = true
	}

	var temp float64
	switch true {
	case x > 21:
		temp = Exp(x) / 2

	case x > 0.5:
		temp = (Exp(x) - Exp(-x)) / 2

	default:
		sq := x * x
		temp = (((P3*sq+P2)*sq+P1)*sq + P0) * x
		temp = temp / (((sq+Q2)*sq+Q1)*sq + Q0)
	}

	if sign {
		temp = -temp
	}
	return temp
}

// Cosh returns the hyperbolic cosine of x.
//
// Special cases are:
//	Cosh(±0) = 1
//	Cosh(±Inf) = +Inf
//	Cosh(NaN) = NaN
func Cosh(x float64) float64 {
	if x < 0 {
		x = -x
	}
	if x > 21 {
		return Exp(x) / 2
	}
	return (Exp(x) + Exp(-x)) / 2
}
