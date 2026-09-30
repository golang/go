// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"simd"
	"testing"
)

func variadicAny(xs ...any) int {
	_ = simd.BroadcastUint8s(0)
	return len(xs)
}

func variadicInt(xs ...int) int {
	_ = simd.BroadcastUint8s(0)
	sum := 0
	for _, x := range xs {
		sum += x
	}
	return sum
}

func variadicAfterFixed(base int, xs ...int) int {
	_ = simd.BroadcastUint8s(0)
	return base + len(xs)
}

func TestMidwayVariadicDispatch(t *testing.T) {
	// An "any" element type is the interesting case: a []any slice is
	// assignable to an any parameter, so dropping the "..." is not a type
	// error and the wrong count is returned instead.
	if got := variadicAny(1, 2, 3); got != 3 {
		t.Errorf("variadicAny(1, 2, 3) = %d, want 3", got)
	}
	if got := variadicAny(); got != 0 {
		t.Errorf("variadicAny() = %d, want 0", got)
	}
	if got := variadicInt(1, 2, 3); got != 6 {
		t.Errorf("variadicInt(1, 2, 3) = %d, want 6", got)
	}
	if got := variadicInt([]int{4, 5}...); got != 9 {
		t.Errorf("variadicInt([]int{4, 5}...) = %d, want 9", got)
	}
	if got := variadicAfterFixed(10, 1, 2); got != 12 {
		t.Errorf("variadicAfterFixed(10, 1, 2) = %d, want 12", got)
	}
}
