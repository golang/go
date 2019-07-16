// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that typed and untyped negative zero floating point constants
// are treated as equivalent to zero constants.

package main

import "math"

const zero = 0.0

func main() {
	x := -zero
	b := math.Float64bits(x)
	if b != 0 {
		panic(b)
	}
	x = -float64(zero)
	b = math.Float64bits(x)
	if b != 0 {
		panic(b)
	}
	v := x
	b = math.Float64bits(-v)
	if b != 0x8000000000000000 {
		panic(b)
	}
}
