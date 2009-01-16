// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import "math"

/*
 *	sinh(arg) returns the hyperbolic sine of its floating-
 *	point argument.
 *
 *	The exponential func is called for arguments
 *	greater in magnitude than 0.5.
 *
 *	A series is used for arguments smaller in magnitude than 0.5.
 *	The coefficients are #2029 from Hart & Cheney. (20.36D)
 *
 *	cosh(arg) is computed from the exponential func for
 *	all arguments.
 */

const
(
	shp0	= -0.6307673640497716991184787251e+6;
	shp1	= -0.8991272022039509355398013511e+5;
	shp2	= -0.2894211355989563807284660366e+4;
	shp3	= -0.2630563213397497062819489e+2;
	shq0	= -0.6307673640497716991212077277e+6;
	shq1	=  0.1521517378790019070696485176e+5;
	shq2	= -0.173678953558233699533450911e+3;
)

export func Sinh(arg float64) float64 {
	sign := false;
	if arg < 0 {
		arg = -arg;
		sign = true;
	}

	var temp float64;
	switch true {
	case arg > 21:
		temp = Exp(arg)/2;

	case arg > 0.5:
		temp = (Exp(arg) - Exp(-arg))/2;

	default:
		argsq := arg*arg;
		temp = (((shp3*argsq+shp2)*argsq+shp1)*argsq+shp0)*arg;
		temp = temp/(((argsq+shq2)*argsq+shq1)*argsq+shq0);
	}

	if sign {
		temp = -temp;
	}
	return temp;
}

export func Cosh(arg float64) float64 {
	if arg < 0 {
		arg = - arg;
	}
	if arg > 21 {
		return Exp(arg)/2;
	}
	return (Exp(arg) + Exp(-arg))/2;
}
