// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

/*
 *	floating-point arctangent
 *
 *	atan returns the value of the arctangent of its
 *	argument in the range [-pi/2,pi/2].
 *	there are no error returns.
 *	coefficients are #5077 from Hart & Cheney. (19.56D)
 */


const
(
	p4	= .161536412982230228262e2;
	p3	= .26842548195503973794141e3;
	p2	= .11530293515404850115428136e4;
	p1	= .178040631643319697105464587e4;
	p0	= .89678597403663861959987488e3;
	q4	= .5895697050844462222791e2;
	q3	= .536265374031215315104235e3;
	q2	= .16667838148816337184521798e4;
	q1	= .207933497444540981287275926e4;
	q0	= .89678597403663861962481162e3;
	pio2	= .15707963267948966192313216e1;
	pio4	= .7853981633974483096156608e0;
	sq2p1	= .2414213562373095048802e1;		// sqrt(2)+1
	sq2m1	= .414213562373095048802e0;		// sqrt(2)-1
)

/*
 *	xatan evaluates a series valid in the
 *	range [-0.414...,+0.414...]. (tan(pi/8))
 */
func
xatan(arg float64) float64
{
	var argsq, value float64;

	argsq = arg*arg;
	value = ((((p4*argsq + p3)*argsq + p2)*argsq + p1)*argsq + p0);
	value = value/(((((argsq + q4)*argsq + q3)*argsq + q2)*argsq + q1)*argsq + q0);
	return value*arg;
}

/*
 *	satan reduces its argument (known to be positive)
 *	to the range [0,0.414...] and calls xatan.
 */
func
satan(arg float64) float64
{

	if arg < sq2m1 {
		return xatan(arg);
	}
	if arg > sq2p1 {
		return pio2 - xatan(1/arg);
	}
	return pio4 + xatan((arg-1)/(arg+1));
}

/*
 *	atan makes its argument positive and
 *	calls the inner routine satan.
 */
export func
atan(arg float64) float64
{

	if arg > 0 {
		return satan(arg);
	}
	return -satan(-arg);
}
