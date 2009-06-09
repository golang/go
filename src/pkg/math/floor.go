// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import "math"

// Floor returns the greatest integer value less than or equal to x.
func Floor(x float64) float64 {
	if x < 0 {
		d, fract := Modf(-x);
		if fract != 0.0 {
			d = d+1;
		}
		return -d;
	}
	d, fract := Modf(x);
	return d;
}

// Ceil returns the least integer value greater than or equal to x.
func Ceil(x float64) float64 {
	return -Floor(-x);
}
