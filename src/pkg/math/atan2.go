// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math


// Atan returns the arc tangent of y/x, using
// the signs of the two to determine the quadrant
// of the return value.
func Atan2(x, y float64) float64 {
	// Determine the quadrant and call atan.
	if x+y == x {
		if x >= 0 {
			return Pi/2
		}
		return -Pi / 2;
	}
	q := Atan(x/y);
	if y < 0 {
		if q <= 0 {
			return q+Pi
		}
		return q-Pi;
	}
	return q;
}
