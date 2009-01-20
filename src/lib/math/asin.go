// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import "math"

/*
 * asin(arg) and acos(arg) return the arcsin, arccos,
 * respectively of their arguments.
 *
 * Arctan is called after appropriate range reduction.
 */

func Asin(arg float64) float64 {
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

	temp = Sqrt(1 - x*x);
	if x > 0.7 {
		temp = Pi/2 - Atan(temp/x);
	} else {
		temp = Atan(x/temp);
	}

	if sign {
		temp = -temp;
	}
	return temp;
}

func Acos(arg float64) float64 {
	if arg > 1 || arg < -1 {
		return sys.NaN();
	}
	return Pi/2 - Asin(arg);
}
