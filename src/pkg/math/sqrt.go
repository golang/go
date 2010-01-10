// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Sqrt returns the square root of x.
//
// Special cases are:
//	Sqrt(+Inf) = +Inf
//	Sqrt(0) = 0
//	Sqrt(x < 0) = NaN
//	Sqrt(NaN) = NaN
func Sqrt(x float64) float64 { return sqrtGo(x) }
