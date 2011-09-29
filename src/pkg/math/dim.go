// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Dim returns the maximum of x-y or 0.
func Dim(x, y float64) float64 {
	if x > y {
		return x - y
	}
	return 0
}

// Max returns the larger of x or y.
func Max(x, y float64) float64 {
	if x > y {
		return x
	}
	return y
}

// Min returns the smaller of x or y.
func Min(x, y float64) float64 {
	if x < y {
		return x
	}
	return y
}
