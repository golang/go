// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Signbit reports whether x is negative or negative zero.
func Signbit(x float64) bool {
	return int64(Float64bits(x)) < 0
}
