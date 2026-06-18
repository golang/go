// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

// --- Masked: zero elements where mask is false ---

func TestMasked(t *testing.T) {
	// Test Masked for Int8x16
	forSlicePair(t, int8s, 16, func(x, y []int8) bool {
		t.Helper()
		a := archsimd.LoadInt8x16(x)
		mask := archsimd.LoadInt8x16(y).Greater(archsimd.Int8x16{}) // mask: y > 0
		g := make([]int8, 16)
		a.Masked(mask).Store(g)
		w := make([]int8, 16)
		for i := range w {
			if y[i] > 0 {
				w[i] = x[i]
			}
		}
		return checkSlicesLogInput(t, g, w, 0.0, func() { t.Helper(); t.Logf("x=%v, y=%v", x, y) })
	})

	// Test Masked for Float64x2
	forSlicePair(t, float64s, 2, func(x, y []float64) bool {
		t.Helper()
		a := archsimd.LoadFloat64x2(x)
		mask := archsimd.LoadFloat64x2(y).Greater(archsimd.Float64x2{}) // mask: y > 0
		g := make([]float64, 2)
		a.Masked(mask).Store(g)
		w := make([]float64, 2)
		for i := range w {
			if y[i] > 0 {
				w[i] = x[i]
			}
		}
		return checkSlicesLogInput(t, g, w, 0.0, func() { t.Helper(); t.Logf("x=%v, y=%v", x, y) })
	})
}

// --- IfElse: set elements to y where mask is true, keep x where true ---

func TestIfElse(t *testing.T) {
	// Test Merge for Int8x16
	forSliceTriple(t, int8s, 16, func(x, y, m []int8) bool {
		t.Helper()
		a := archsimd.LoadInt8x16(x)
		b := archsimd.LoadInt8x16(y)
		mask := archsimd.LoadInt8x16(m).Greater(archsimd.Int8x16{}) // mask: m > 0
		g := make([]int8, 16)
		a.IfElse(mask, b).Store(g)
		w := make([]int8, 16)
		for i := range w {
			if m[i] > 0 {
				w[i] = y[i]
			} else {
				w[i] = x[i]
			}
		}
		return checkSlicesLogInput(t, g, w, 0.0, func() { t.Helper(); t.Logf("x=%v, y=%v, m=%v", x, y, m) })
	})

	// Test Merge for Float32x4
	forSliceTriple(t, float32s, 4, func(x, y, m []float32) bool {
		t.Helper()
		a := archsimd.LoadFloat32x4(x)
		b := archsimd.LoadFloat32x4(y)
		mask := archsimd.LoadFloat32x4(m).Greater(archsimd.Float32x4{}) // mask: m > 0
		g := make([]float32, 4)
		a.IfElse(mask, b).Store(g)
		w := make([]float32, 4)
		for i := range w {
			if m[i] > 0 {
				w[i] = y[i]
			} else {
				w[i] = x[i]
			}
		}
		return checkSlicesLogInput(t, g, w, 0.0, func() { t.Helper(); t.Logf("x=%v, y=%v, m=%v", x, y, m) })
	})
}
