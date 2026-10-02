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
	// 0.5 ≤ f < 1, so t = 2f-1 is in [0, 1) and x = 2^{e-1}·(1+t).
	// log₂(1+t) ≈ t(4-t)/3 interpolates log₂(1)=0 and log₂(2)=1 exactly
	// (max error on [1,2) is about 0.01 bits). The previous 2(f-1) fit
	// was piecewise-linear and could push the 2.5 alignment threshold
	// the other way from math.Log/math.Exp.
	t := 2*f - 1
	return float64(e-1) + t*(4-t)/3
}

// exp2ish returns a crude approximation to 2**x.
// The result is only used for heuristic alignment decisions and should
// not be used where precision matters.
// The approximation is guaranteed to produce identical results
// across all architectures.
func exp2ish(x float64) float64 {
	n := math.Floor(x)
	f := x - n
	// 2^f ≈ 1 + f(2+f)/3 interpolates 2^0=1 and 2^1=2 exactly.
	return math.Ldexp(1+f*(2+f)/3, int(n))
}
