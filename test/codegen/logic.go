// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// Test to make sure that (CMPQ (ANDQ x y) [0]) does not get rewritten to
// (TESTQ x y) if the ANDQ has other uses. If that rewrite happens, then one
// of the args of the ANDQ needs to be saved so it can be used as the arg to TESTQ.
func andWithUse(x, y int) int {
	z := x & y
	// amd64:`TESTQ\s(AX, AX|BX, BX|CX, CX|DX, DX|SI, SI|DI, DI|R8, R8|R9, R9|R10, R10|R11, R11|R12, R12|R13, R13|R15, R15)`
	if z == 0 {
		return 77
	}
	// use z by returning it
	return z
}

// Verify (OR x (NOT y)) rewrites to (ORN x y) where supported
func ornot(x, y int) int {
	// ppc64x:"ORN"
	z := x | ^y
	return z
}

// Verify that (OR (NOT x) (NOT y)) rewrites to (NOT (AND x y))
func orDemorgans(x, y int) int {
	// amd64:"AND" -"OR"
	z := ^x | ^y
	return z
}

// Verify that (AND (NOT x) (NOT y)) rewrites to (NOT (OR x y))
func andDemorgans(x, y int) int {
	// amd64:"OR" -"AND"
	z := ^x & ^y
	return z
}
