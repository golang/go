// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import	math "exp"
export	sinh, cosh

/*
	sinh(arg) returns the hyperbolic sine of its floating-
	point argument.

	The exponential func is called for arguments
	greater in magnitude than 0.5.

	A series is used for arguments smaller in magnitude than 0.5.
	The coefficients are #2029 from Hart & Cheney. (20.36D)

	cosh(arg) is computed from the exponential func for
	all arguments.
 */

const
(
	p0	= -0.6307673640497716991184787251e+6;
	p1	= -0.8991272022039509355398013511e+5;
	p2	= -0.2894211355989563807284660366e+4;
	p3	= -0.2630563213397497062819489e+2;
	q0	= -0.6307673640497716991212077277e+6;
	q1	=  0.1521517378790019070696485176e+5;
	q2	= -0.173678953558233699533450911e+3;
)

func
sinh(arg double) double
{
	var temp, argsq double;
	var sign bool;

	sign = false;
	if arg < 0 {
		arg = -arg;
		sign = true;
	}
	switch true {
	case arg > 21:
		temp = exp(arg)/2;

	case arg > 0.5:
//		temp = (exp(arg) - exp(-arg))/2;

	default:
		argsq = arg*arg;
		temp = (((p3*argsq+p2)*argsq+p1)*argsq+p0)*arg;
		temp = temp/(((argsq+q2)*argsq+q1)*argsq+q0);
	}

	if sign {
		temp = -temp;
	}
	return temp;
}

func
cosh(arg double) double
{
	if arg < 0 {
		arg = - arg;
	}
	if arg > 21 {
		return exp(arg)/2;
	}
//	return (exp(arg) + exp(-arg))/2;
}
