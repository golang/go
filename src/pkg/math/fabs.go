// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Fabs returns the absolute value of x.
//
// Special cases are:
//	Fabs(+Inf) = +Inf
//	Fabs(-Inf) = +Inf
//	Fabs(NaN) = NaN
func Fabs(x float64) float64 {
	switch {
	case x < 0:
		return -x
	case x == 0:
		return 0 // return correctly fabs(-0)
	}
	return x
}
