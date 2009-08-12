// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math


/*
 * asin(arg) and acos(arg) return the arcsin, arccos,
 * respectively of their arguments.
 *
 * Arctan is called after appropriate range reduction.
 */

// Asin returns the arc sine of x.
func Asin(x float64) float64 {
	sign := false;
	if x < 0 {
		x = -x;
		sign = true;
	}
	if x > 1 {
		return NaN();
	}

	temp := Sqrt(1 - x*x);
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

// Acos returns the arc cosine of x.
func Acos(x float64) float64 {
	if x > 1 || x < -1 {
		return NaN();
	}
	return Pi/2 - Asin(x);
}
