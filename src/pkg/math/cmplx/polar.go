// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmplx

// Polar returns the absolute value r and phase θ of x,
// such that x = r * e**θi.
// The phase is in the range [-Pi, Pi].
func Polar(x complex128) (r, θ float64) {
	return Abs(x), Phase(x)
}
