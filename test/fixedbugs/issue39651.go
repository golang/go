// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that float -> integer conversion doesn't clobber
// flags.

package main

//go:noinline
func f(x, y float64, a, b *bool, r *int64) {
	*a = x < y    // set flags
	*r = int64(x) // clobber flags
	*b = x == y   // use flags
}

func main() {
	var a, b bool
	var r int64
	f(1, 1, &a, &b, &r)
	if a || !b {
		panic("comparison incorrect")
	}
}
