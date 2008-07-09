// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

export	fmod

/*
 *	floating-point mod func without infinity or NaN checking
 */

func
fmod(x, y float64) float64
{
	var yexp, rexp int;
	var r, yfr, rfr float64;
	var sign bool;

	if y == 0 {
		return x;
	}
	if y < 0 {
		y = -y;
	}

	yfr,yexp = sys.frexp(y);
	sign = false;
	if x < 0 {
		r = -x;
		sign = true;
	} else {
		r = x;
	}

	for r >= y {
		rfr,rexp = sys.frexp(r);
		if rfr < yfr {
			rexp = rexp - 1;
		}
		r = r - sys.ldexp(y, rexp-yexp);
	}
	if sign {
		r = -r;
	}
	return r;
}
