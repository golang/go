// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

/*
 *	floating-point mod func without infinity or NaN checking
 */

func Fmod(x, y float64) float64 {
	if y == 0 {
		return x;
	}
	if y < 0 {
		y = -y;
	}

	yfr, yexp := sys.Frexp(y);
	sign := false;
	r := x;
	if x < 0 {
		r = -x;
		sign = true;
	}

	for r >= y {
		rfr, rexp := sys.Frexp(r);
		if rfr < yfr {
			rexp = rexp - 1;
		}
		r = r - sys.Ldexp(y, rexp-yexp);
	}
	if sign {
		r = -r;
	}
	return r;
}
