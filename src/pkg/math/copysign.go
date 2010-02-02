// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math


// Copysign(x, y) returns a value with the magnitude
// of x and the sign of y.
func Copysign(x, y float64) float64 {
	if x < 0 && y > 0 || x > 0 && y < 0 {
		return -x
	}
	return x
}
