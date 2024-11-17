// errorcheck -0 -d=ssa/tighten/debug=1

//go:build arm64

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var (
	ga, gb, gc, gd int
)

func moveValuesWithMemoryArg(len int) {
	for n := 0; n < len; n++ {
		// Loads of b and d can be delayed until inside the outer "if".
		a := ga
		b := gb // ERROR "MOVDload is moved$"
		c := gc
		d := gd // ERROR "MOVDload is moved$"
		if a == c {
			if b == d {
				return
			}
		}
	}
}
