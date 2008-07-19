// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import	math "math"

export	asin, acos

/*
 * asin(arg) and acos(arg) return the arcsin, arccos,
 * respectively of their arguments.
 *
 * Arctan is called after appropriate range reduction.
 */

const
(
	pio2 = .15707963267948966192313216e1
)

func
asin(arg float64)float64
{
	var temp, x float64;
	var sign bool;

	sign = false;
	x = arg;
	if x < 0 {
		x = -x;
		sign = true;
	}
	if arg > 1 {
		return sys.NaN();
	}

	temp = sqrt(1 - x*x);
	if x > 0.7 {
		temp = pio2 - atan(temp/x);
	} else {
		temp = atan(x/temp);
	}

	if sign {
		temp = -temp;
	}
	return temp;
}

func
acos(arg float64)float64
{
	if(arg > 1 || arg < -1) {
		return sys.NaN();
	}
	return pio2 - asin(arg);
}
