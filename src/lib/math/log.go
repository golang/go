// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package log

import	sys "sys"
export	log, log10

/*
	log returns the natural logarithm of its floating
	point argument.

	The coefficients are #2705 from Hart & Cheney. (19.38D)

	It calls frexp.
*/

const
(
	log2	=   .693147180559945309e0;
	ln10o1	=   .4342944819032518276511;
	sqrto2	=   .707106781186547524e0;
	p0	=  -.240139179559210510e2;
	p1	=   .309572928215376501e2;
	p2	=  -.963769093377840513e1;
	p3	=   .421087371217979714e0;
	q0	=  -.120069589779605255e2;
	q1	=   .194809660700889731e2;
	q2	=  -.891110902798312337e1;
)

func
log(arg double) double
{
	var x, z, zsq, temp double;
	var exp int;

	if arg <= 0 {
		return sys.NaN();
	}

	exp,x = sys.frexp(arg);
	for x < 0.5 {
		x = x*2;
		exp = exp-1;
	}
	if x < sqrto2 {
		x = x*2;
		exp = exp-1;
	}

	z = (x-1) / (x+1);
	zsq = z*z;

	temp = ((p3*zsq + p2)*zsq + p1)*zsq + p0;
	temp = temp/(((zsq + q2)*zsq + q1)*zsq + q0);
	temp = temp*z + double(exp)*log2;
	return temp;
}

func
log10(arg double) double
{

	if arg <= 0 {
		return sys.NaN();
	}
	return log(arg) * ln10o1;
}
