// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import "math"

/*
 *	atan2 discovers what quadrant the angle
 *	is in and calls atan.
 */

const
(
	pio2	= .15707963267948966192313216e1;
	pi	= .3141592653589793238462643383276e1;
)

export func Atan2(arg1, arg2 float64) float64 {
	if arg1+arg2 == arg1 {
		if arg1 >= 0 {
			return pio2;
		}
		return -pio2;
	}
	x := Atan(arg1/arg2);
	if arg2 < 0 {
		if x <= 0 {
			return x + pi;
		}
		return x - pi;
	}
	return x;
}
