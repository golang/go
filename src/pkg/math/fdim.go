// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Fdim returns the maximum of x-y or 0.
func Fdim(x, y float64) float64 {
	if x > y {
		return x - y
	}
	return 0
}

// Fmax returns the larger of x or y.
func Fmax(x, y float64) float64 {
	if x > y {
		return x
	}
	return y
}

// Fmin returns the smaller of x or y.
func Fmin(x, y float64) float64 {
	if x < y {
		return x
	}
	return y
}
