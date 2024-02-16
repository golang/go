// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue9510a

/*
static double csquare(double a, double b) {
	__complex__ double d;
	__real__ d = a;
	__imag__ d = b;
	return __real__ (d * d);
}
*/
import "C"

func F(a, b float64) float64 {
	return float64(C.csquare(C.double(a), C.double(b)))
}
