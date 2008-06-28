// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

export		tan

/*
	floating point tangent
	Coefficients are #4285 from Hart & Cheney. (19.74D)
 */

const
(
	p0	= -.1306820264754825668269611177e+5;
	p1	=  .1055970901714953193602353981e+4;
	p2	= -.1550685653483266376941705728e+2;
	p3	=  .3422554387241003435328470489e-1;
	p4	=  .3386638642677172096076369e-4;
	q0	= -.1663895238947119001851464661e+5;
	q1	=  .4765751362916483698926655581e+4;
	q2	= -.1555033164031709966900124574e+3;
        piu4	=  .1273239544735162686151070107e+1;	// 4/pi
)

func
tan(arg double) double
{
	var temp, e, x, xsq double;
	var i long;
	var flag, sign bool;

	flag = false;
	sign = false;
	x = arg;
	if(x < 0) {
		x = -x;
		sign = true;
	}
	x = x * piu4;   /* overflow? */
	e,x = sys.modf(x);
	i = long(e);

	switch i & 3 {
	case 1:
		x = 1 - x;
		flag = true;

	case 2:
		sign = !sign;
		flag = true;

	case 3:
		x = 1 - x;
		sign = !sign;
	}

	xsq = x*x;
	temp = ((((p4*xsq+p3)*xsq+p2)*xsq+p1)*xsq+p0)*x;
	temp = temp/(((xsq+q2)*xsq+q1)*xsq+q0);

	if flag {
		if(temp == 0) {
			panic "return sys.NaN()";
		}
		temp = 1/temp;
	}
	if sign {
		temp = -temp;
	}
	return temp;
}
