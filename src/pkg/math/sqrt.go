// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math


/*
 *	sqrt returns the square root of its floating
 *	point argument. Newton's method.
 *
 *	calls frexp
 */

// Sqrt returns the square root of x.
//
// Special cases are:
//	Sqrt(+Inf) = +Inf
//	Sqrt(0) = 0
//	Sqrt(x < 0) = NaN
func Sqrt(x float64) float64 {
	if IsInf(x, 1) {
		return x;
	}

	if x <= 0 {
		if x < 0 {
			return NaN();
		}
		return 0;
	}

	y, exp := Frexp(x);
	for y < 0.5 {
		y = y*2;
		exp = exp-1;
	}

	if exp&1 != 0 {
		y = y*2;
		exp = exp-1;
	}
	temp := 0.5 * (1+y);

	for exp > 60 {
		temp = temp * float64(1<<30);
		exp = exp - 60;
	}
	for exp < -60 {
		temp = temp / float64(1<<30);
		exp = exp + 60;
	}
	if exp >= 0 {
		exp = 1 << uint(exp/2);
		temp = temp * float64(exp);
	} else {
		exp = 1 << uint(-exp/2);
		temp = temp / float64(exp);
	}

	for i:=0; i<=4; i++ {
		temp = 0.5*(temp + x/temp);
	}
	return temp;
}
