// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

export		sqrt

/*
	sqrt returns the square root of its floating
	point argument. Newton's method.

	calls frexp
*/

func
sqrt(arg double) double
{
	var x, temp double;
	var exp, i int;

	if sys.isInf(arg, 1) {
		return arg;
	}

	if arg <= 0 {
		if arg < 0 {
			panic "return sys.NaN()"
		}
		return 0;
	}

	x,exp = sys.frexp(arg);
	for x < 0.5 {
		x = x*2;
		exp = exp-1;
	}

	if exp&1 != 0 {
		x = x*2;
		exp = exp-1;
	}
	temp = 0.5 * (1+x);

	for exp > 60 {
		temp = temp * double(1<<30);
		exp = exp - 60;
	}
	for exp < -60 {
		temp = temp / double(1<<30);
		exp = exp + 60;
	}
	if exp >= 0 {
		temp = temp * double(1 << (exp/2));
	} else {
		temp = temp / double(1 << (-exp/2));
	}

	for i=0; i<=4; i=i+1 {
		temp = 0.5*(temp + arg/temp);
	}
	return temp;
}
