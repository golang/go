// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import "math"

/*
 *	atan2 discovers what quadrant the angle
 *	is in and calls atan.
 */
func Atan2(arg1, arg2 float64) float64 {
	if arg1+arg2 == arg1 {
		if arg1 >= 0 {
			return Pi/2;
		}
		return -Pi/2;
	}
	x := Atan(arg1/arg2);
	if arg2 < 0 {
		if x <= 0 {
			return x + Pi;
		}
		return x - Pi;
	}
	return x;
}
