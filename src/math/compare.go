// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

func sign[I int32 | int64](a, b I) int {
	if a < b {
		return -1
	}
	if a > b {
		return 1
	}
	return 0
}

// Compare compares a and b such that
// -NaN is ordered before any other value,
// +NaN is ordered after any other value,
// and -0 is ordered before +0.
// In other words, it defines a total order over floats
// (according to the total-ordering predicate in IEEE-754, section 5.10).
// It returns 0 if a == b, -1 if a < b, and +1 if a > b.
func Compare(a, b float64) int {
	// Perform a bitwise comparison (a < b) by casting the float64s into an int64s.
	x := int64(Float64bits(a))
	y := int64(Float64bits(b))

	// If a and b are both negative, flip the comparison so that we check a > b.
	if x < 0 && y < 0 {
		return sign(y, x)
	}
	return sign(x, y)
}

// Compare32 compares a and b such that
// -NaN is ordered before any other value,
// +NaN is ordered after any other value,
// and -0 is ordered before +0.
// In other words, it defines a total order over floats
// (according to the total-ordering predicate in IEEE-754, section 5.10).
// It returns 0 if a == b, -1 if a < b, and +1 if a > b.
func Compare32(a, b float32) int {
	// Perform a bitwise comparison (a < b) by casting the float32s into an int32s.
	x := int32(Float32bits(a))
	y := int32(Float32bits(b))

	// If a and b are both negative, flip the comparison so that we check a > b.
	if x < 0 && y < 0 {
		return sign(y, x)
	}
	return sign(x, y)
}
