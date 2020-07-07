// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

var gx, gy int

// Test to make sure that (CMPQ (ANDQ x y) [0]) does not get rewritten to
// (TESTQ x y) if the ANDQ has other uses. If that rewrite happens, then one
// of the args of the ANDQ needs to be saved so it can be used as the arg to TESTQ.
func andWithUse(x, y int) int {
	// Load x,y into registers, so those MOVQ will not appear at the z := x&y line.
	gx, gy = x, y
	// amd64:-"MOVQ"
	z := x & y
	if z == 0 {
		return 77
	}
	// use z by returning it
	return z
}
