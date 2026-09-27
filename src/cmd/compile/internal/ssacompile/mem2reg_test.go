// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"testing"
)

// Single-block mem2reg test
//
//go:noinline
func justReturn0() int {
	var x, y, z int
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

func TestMem2Reg(t *testing.T) {
	if r := justReturn0(); r != 0 {
		t.Errorf("justReturn0() = %d, want 0", r)
	}
}
