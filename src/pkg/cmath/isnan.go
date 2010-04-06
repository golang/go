// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmath

import "math"

// IsNaN returns true if either real(x) or imag(x) is NaN.
func IsNaN(x complex128) bool {
	if math.IsNaN(real(x)) || math.IsNaN(imag(x)) {
		return true
	}
	return false
}

// NaN returns a complex ``not-a-number'' value.
func NaN() complex128 {
	nan := math.NaN()
	return cmplx(nan, nan)
}
