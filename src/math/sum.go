// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Sum returns the sum of the values in x.
//
// Special cases are:
//
//	Sum([]) = 0
//	Sum(x) = NaN if any x[i] is NaN
//	Sum(x) = +Inf if sum overflows positive (no -Inf or NaN present)
//	Sum(x) = -Inf if sum overflows negative (no +Inf or NaN present)
//	Sum containing both +Inf and -Inf = NaN
func Sum(x []float64) float64 {
	s := 0.0
	for _, v := range x {
		s += v
	}
	return s
}
