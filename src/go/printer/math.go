// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package printer

import "math"

// log2ish returns a crude approximation to log₂(x).
// The result is only used for heuristic alignment decisions and should
// not be used where precision matters.
// The approximation is guaranteed to produce identical results
// across all architectures.
func log2ish(x float64) float64 {
	f, e := math.Frexp(x)
	return float64(e) + 2*(f-1)
}

// exp2ish returns a crude approximation to 2**x.
// The result is only used for heuristic alignment decisions and should
// not be used where precision matters.
// The approximation is guaranteed to produce identical results
// across all architectures.
func exp2ish(x float64) float64 {
	n := math.Floor(x)
	f := x - n
	return math.Ldexp(1+f, int(n))
}
