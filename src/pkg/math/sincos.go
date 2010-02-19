// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Sincos(x) returns Sin(x), Cos(x).
//
// Special conditions are:
//	Sincos(+Inf) = NaN, NaN
//	Sincos(-Inf) = NaN, NaN
//	Sincos(NaN) = NaN, NaN
func Sincos(x float64) (sin, cos float64) { return Sin(x), Cos(x) }
