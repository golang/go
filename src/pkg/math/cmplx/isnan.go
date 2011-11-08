// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmplx

import "math"

// IsNaN returns true if either real(x) or imag(x) is NaN
// and neither is an infinity.
func IsNaN(x complex128) bool {
	switch {
	case math.IsInf(real(x), 0) || math.IsInf(imag(x), 0):
		return false
	case math.IsNaN(real(x)) || math.IsNaN(imag(x)):
		return true
	}
	return false
}

// NaN returns a complex ``not-a-number'' value.
func NaN() complex128 {
	nan := math.NaN()
	return complex(nan, nan)
}
