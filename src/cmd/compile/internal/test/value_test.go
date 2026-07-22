// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/types"
	"internal/buildcfg"
	"testing"
)

// This file contains tests for ssa values, types and their utility functions.

func TestCanSSA(t *testing.T) {
	i64 := types.Types[types.TINT64]
	v128 := types.TypeVec128
	s1 := mkstruct(i64, mkstruct(i64, i64, i64, i64))
	if ssacore.CanSSA(s1) {
		// Test size check for struct.
		t.Errorf("CanSSA(%v) returned true, expected false", s1)
	}
	a1 := types.NewArray(s1, 1)
	if ssacore.CanSSA(a1) {
		// Test size check for array.
		t.Errorf("CanSSA(%v) returned true, expected false", a1)
	}
	if buildcfg.Experiment.SIMD {
		s2 := mkstruct(v128, v128, v128, v128)
		if !ssacore.CanSSA(s2) {
			// Test size check for SIMD struct special case.
			t.Errorf("CanSSA(%v) returned false, expected true", s2)
		}
		a2 := types.NewArray(s2, 1)
		if !ssacore.CanSSA(a2) {
			// Test size check for SIMD array special case.
			t.Errorf("CanSSA(%v) returned false, expected true", a2)
		}
	}
}
