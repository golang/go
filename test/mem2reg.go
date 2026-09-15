// errorcheck -0 -d=ssa/mem2reg/debug=4

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test for variables accessed through pointers that can
// be promoted to register by the mem2reg pass.

package main

// The following functions are ported from src/cmd/compile/internal/ssa/mem2reg_test.go.

// Single-block mem2reg test

func justReturn0() int {
	var x, y, z int // ERROR "promoted x in single block case" "promoted y in single block case" "promoted z in single block case"
	xp := &x
	yp := &y
	zp := &z
	*xp = 0
	*yp = 1
	*zp = 2
	*yp = 3
	*zp = 4
	*yp = 5
	valX1 := *xp
	*zp = valX1 - 1
	*xp = 1
	*yp = 2
	*xp = 3
	*yp = 4
	*xp = 5
	valZ1 := *zp
	*yp = valZ1 + 1
	*zp = 1
	*xp = 2
	*zp = 3
	*xp = 4
	*zp = 5
	valY1 := *yp
	return valY1
}
