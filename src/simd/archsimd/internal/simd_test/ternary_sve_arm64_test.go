// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

// SVE ternary-op tests, the three-input counterpart of the drivers in
// binary_sve_arm64_test.go. flakiness absorbs the float32 double rounding of
// the math.FMA-based emulation, as in the NEON MulAdd tests.

package simd_test

import (
	"simd/archsimd"
	"testing"
)

// testSVETernary drives a scalable ternary op like testSVEBinary. active is the
// runtime number of live lanes (from the vector's Len()).
func testSVETernary[T number, V any](t *testing.T, pool []T, elemBytes, active int, flakiness float64,
	load func([]T) V, f func(V, V, V) V, store func(V, []T), want func(_, _, _ []T) []T) {
	t.Helper()
	count := sveMaxBytes / elemBytes // lanes in the fixed backing array
	forSliceTriple(t, pool, count, func(x, y, z []T) bool {
		t.Helper()
		g := make([]T, count)
		store(f(load(x), load(y), load(z)), g)
		w := want(x, y, z)
		return checkSlicesLogInput(t, g[:active], w[:active], flakiness, func() {
			t.Helper()
			t.Logf("x=%v", x)
			t.Logf("y=%v", y)
			t.Logf("z=%v", z)
		})
	})
}

func TestMulAddSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testSVETernary(t, int8s, 1, archsimd.Int8s{}.Len(), 0, archsimd.LoadInt8s, archsimd.Int8s.MulAdd, archsimd.Int8s.Store, imaSlice[int8])
	testSVETernary(t, int16s, 2, archsimd.Int16s{}.Len(), 0, archsimd.LoadInt16s, archsimd.Int16s.MulAdd, archsimd.Int16s.Store, imaSlice[int16])
	testSVETernary(t, int32s, 4, archsimd.Int32s{}.Len(), 0, archsimd.LoadInt32s, archsimd.Int32s.MulAdd, archsimd.Int32s.Store, imaSlice[int32])
	testSVETernary(t, int64s, 8, archsimd.Int64s{}.Len(), 0, archsimd.LoadInt64s, archsimd.Int64s.MulAdd, archsimd.Int64s.Store, imaSlice[int64])
	testSVETernary(t, uint8s, 1, archsimd.Uint8s{}.Len(), 0, archsimd.LoadUint8s, archsimd.Uint8s.MulAdd, archsimd.Uint8s.Store, imaSlice[uint8])
	testSVETernary(t, uint16s, 2, archsimd.Uint16s{}.Len(), 0, archsimd.LoadUint16s, archsimd.Uint16s.MulAdd, archsimd.Uint16s.Store, imaSlice[uint16])
	testSVETernary(t, uint32s, 4, archsimd.Uint32s{}.Len(), 0, archsimd.LoadUint32s, archsimd.Uint32s.MulAdd, archsimd.Uint32s.Store, imaSlice[uint32])
	testSVETernary(t, uint64s, 8, archsimd.Uint64s{}.Len(), 0, archsimd.LoadUint64s, archsimd.Uint64s.MulAdd, archsimd.Uint64s.Store, imaSlice[uint64])
	testSVETernary(t, float32s, 4, archsimd.Float32s{}.Len(), 0.001, archsimd.LoadFloat32s, archsimd.Float32s.MulAdd, archsimd.Float32s.Store, fmaSlice[float32])
	testSVETernary(t, float64s, 8, archsimd.Float64s{}.Len(), 0.001, archsimd.LoadFloat64s, archsimd.Float64s.MulAdd, archsimd.Float64s.Store, fmaSlice[float64])
}
