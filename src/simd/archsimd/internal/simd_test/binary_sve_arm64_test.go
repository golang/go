// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

// SVE binary-op tests. Unlike amd64, SVE has only a handful of (scalable)
// vector types, so there is nothing to generate — these drivers are hand-written
// in the same shape as the generated testXxxBinary helpers. Each loads two input
// windows via the fixed-array API, runs the op, stores the result, and compares
// the lanes the hardware actually populated: the vector's runtime Len() (VL is
// <= the 32-byte backing, enforced at package init).

package simd_test

import (
	"simd/archsimd"
	"testing"
)

// sveMaxBytes is the fixed backing-array size for a scalable vector: the maximum
// vector length simd supports (256 bits).
const sveMaxBytes = 32

// testSVEBinary drives a scalable binary op like the generated testXxxBinary
// helpers. active is the runtime number of live lanes (from the vector's Len()).
func testSVEBinary[T number, V any](t *testing.T, pool []T, elemBytes, active int,
	load func([]T) V, f func(V, V) V, store func(V, []T), want func([]T, []T) []T) {
	t.Helper()
	count := sveMaxBytes / elemBytes // lanes in the fixed backing array
	forSlicePair(t, pool, count, func(x, y []T) bool {
		t.Helper()
		g := make([]T, count)
		store(f(load(x), load(y)), g)
		w := want(x, y)
		return checkSlicesLogInput(t, g[:active], w[:active], 0.0, func() {
			t.Helper()
			t.Logf("x=%v", x)
			t.Logf("y=%v", y)
		})
	})
}

func testInt8sBinary(t *testing.T, f func(_, _ archsimd.Int8s) archsimd.Int8s, want func(_, _ []int8) []int8) {
	var z archsimd.Int8s
	testSVEBinary(t, int8s, 1, z.Len(), archsimd.LoadInt8s, f, archsimd.Int8s.Store, want)
}

func testInt16sBinary(t *testing.T, f func(_, _ archsimd.Int16s) archsimd.Int16s, want func(_, _ []int16) []int16) {
	var z archsimd.Int16s
	testSVEBinary(t, int16s, 2, z.Len(), archsimd.LoadInt16s, f, archsimd.Int16s.Store, want)
}

func testInt32sBinary(t *testing.T, f func(_, _ archsimd.Int32s) archsimd.Int32s, want func(_, _ []int32) []int32) {
	var z archsimd.Int32s
	testSVEBinary(t, int32s, 4, z.Len(), archsimd.LoadInt32s, f, archsimd.Int32s.Store, want)
}

func testInt64sBinary(t *testing.T, f func(_, _ archsimd.Int64s) archsimd.Int64s, want func(_, _ []int64) []int64) {
	var z archsimd.Int64s
	testSVEBinary(t, int64s, 8, z.Len(), archsimd.LoadInt64s, f, archsimd.Int64s.Store, want)
}

func testUint8sBinary(t *testing.T, f func(_, _ archsimd.Uint8s) archsimd.Uint8s, want func(_, _ []uint8) []uint8) {
	var z archsimd.Uint8s
	testSVEBinary(t, uint8s, 1, z.Len(), archsimd.LoadUint8s, f, archsimd.Uint8s.Store, want)
}

func testFloat32sBinary(t *testing.T, f func(_, _ archsimd.Float32s) archsimd.Float32s, want func(_, _ []float32) []float32) {
	var z archsimd.Float32s
	testSVEBinary(t, float32s, 4, z.Len(), archsimd.LoadFloat32s, f, archsimd.Float32s.Store, want)
}

func testFloat64sBinary(t *testing.T, f func(_, _ archsimd.Float64s) archsimd.Float64s, want func(_, _ []float64) []float64) {
	var z archsimd.Float64s
	testSVEBinary(t, float64s, 8, z.Len(), archsimd.LoadFloat64s, f, archsimd.Float64s.Store, want)
}

func TestAddSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testInt8sBinary(t, archsimd.Int8s.Add, addSlice[int8])
	testInt16sBinary(t, archsimd.Int16s.Add, addSlice[int16])
	testInt32sBinary(t, archsimd.Int32s.Add, addSlice[int32])
	testInt64sBinary(t, archsimd.Int64s.Add, addSlice[int64])
	testUint8sBinary(t, archsimd.Uint8s.Add, addSlice[uint8])
	testFloat32sBinary(t, archsimd.Float32s.Add, addSlice[float32])
	testFloat64sBinary(t, archsimd.Float64s.Add, addSlice[float64])
}

func TestSubSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testInt8sBinary(t, archsimd.Int8s.Sub, subSlice[int8])
	testInt16sBinary(t, archsimd.Int16s.Sub, subSlice[int16])
	testInt32sBinary(t, archsimd.Int32s.Sub, subSlice[int32])
	testInt64sBinary(t, archsimd.Int64s.Sub, subSlice[int64])
	testUint8sBinary(t, archsimd.Uint8s.Sub, subSlice[uint8])
	testFloat32sBinary(t, archsimd.Float32s.Sub, subSlice[float32])
	testFloat64sBinary(t, archsimd.Float64s.Sub, subSlice[float64])
}

// testSVEUnary drives a scalable unary op, the one-input counterpart of
// testSVEBinary.
func testSVEUnary[T number, V any](t *testing.T, pool []T, elemBytes, active int,
	load func([]T) V, f func(V) V, store func(V, []T), want func([]T) []T) {
	t.Helper()
	count := sveMaxBytes / elemBytes
	forSlice(t, pool, count, func(x []T) bool {
		t.Helper()
		g := make([]T, count)
		store(f(load(x)), g)
		w := want(x)
		return checkSlicesLogInput(t, g[:active], w[:active], 0.0, func() {
			t.Helper()
			t.Logf("x=%v", x)
		})
	})
}

func TestAbsSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	absSlice := func(x []int8) []int8 {
		r := make([]int8, len(x))
		for i, v := range x {
			if v < 0 {
				v = -v // -128 stays -128, as ABS does
			}
			r[i] = v
		}
		return r
	}
	var z archsimd.Int8s
	testSVEUnary(t, int8s, 1, z.Len(), archsimd.LoadInt8s, archsimd.Int8s.Abs, archsimd.Int8s.Store, absSlice)
}
