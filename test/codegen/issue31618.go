// asmcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// Make sure we remove both inline marks in the following code.
// Both +5 and +6 should map to real instructions, which can
// be used as inline marks instead of explicit nops.
func f(x int) int {
	// amd64:-"XCHGL"
	x = g(x) + 5
	// amd64:-"XCHGL"
	x = g(x) + 6
	return x
}

func g(x int) int {
	return x >> 3
}
