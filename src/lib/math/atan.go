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
	ap4	= .161536412982230228262e2;
	ap3	= .26842548195503973794141e3;
	ap2	= .11530293515404850115428136e4;
	ap1	= .178040631643319697105464587e4;
	ap0	= .89678597403663861959987488e3;
	aq4	= .5895697050844462222791e2;
	aq3	= .536265374031215315104235e3;
	aq2	= .16667838148816337184521798e4;
	aq1	= .207933497444540981287275926e4;
	aq0	= .89678597403663861962481162e3;
	apio2	= .15707963267948966192313216e1;
	apio4	= .7853981633974483096156608e0;
	asq2p1	= .2414213562373095048802e1;		// sqrt(2)+1
	asq2m1	= .414213562373095048802e0;		// sqrt(2)-1
)

/*
 *	xatan evaluates a series valid in the
 *	range [-0.414...,+0.414...]. (tan(pi/8))
 */
func xatan(arg float64) float64 {
	argsq := arg*arg;
	value := ((((ap4*argsq + ap3)*argsq + ap2)*argsq + ap1)*argsq + ap0);
	value = value/(((((argsq + aq4)*argsq + aq3)*argsq + aq2)*argsq + aq1)*argsq + aq0);
	return value*arg;
}

/*
 *	satan reduces its argument (known to be positive)
 *	to the range [0,0.414...] and calls xatan.
 */
func satan(arg float64) float64 {
	if arg < asq2m1 {
		return xatan(arg);
	}
	if arg > asq2p1 {
		return apio2 - xatan(1/arg);
	}
	return apio4 + xatan((arg-1)/(arg+1));
}

/*
 *	atan makes its argument positive and
 *	calls the inner routine satan.
 */
export func Atan(arg float64) float64 {
	if arg > 0 {
		return satan(arg);
	}
	return -satan(-arg);
}
