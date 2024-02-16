// errorcheck -0 -d=ssa/tighten/debug=1

//go:build arm64

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var (
	e  any
	ts uint16
)

func moveValuesWithMemoryArg(len int) {
	for n := 0; n < len; n++ {
		// Load of e.data is lowed as a MOVDload op, which has a memory
		// argument. It's moved near where it's used.
		_ = e != ts // ERROR "MOVDload is moved$" "MOVDaddr is moved$"
	}
}
