// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// cmp_ssa.go tests compare simplification operations.
package main

import "testing"

//go:noinline
func eq_ssa(a int64) bool {
	return 4+a == 10
}

//go:noinline
func neq_ssa(a int64) bool {
	return 10 != a+4
}

func testCmp(t *testing.T) {
	if wanted, got := true, eq_ssa(6); wanted != got {
		t.Errorf("eq_ssa: expected %v, got %v\n", wanted, got)
	}
	if wanted, got := false, eq_ssa(7); wanted != got {
		t.Errorf("eq_ssa: expected %v, got %v\n", wanted, got)
	}
	if wanted, got := false, neq_ssa(6); wanted != got {
		t.Errorf("neq_ssa: expected %v, got %v\n", wanted, got)
	}
	if wanted, got := true, neq_ssa(7); wanted != got {
		t.Errorf("neq_ssa: expected %v, got %v\n", wanted, got)
	}
}

func TestCmp(t *testing.T) {
	testCmp(t)
}
