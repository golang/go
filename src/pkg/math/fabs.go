// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Fabs returns the absolute value of x.
func Fabs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
