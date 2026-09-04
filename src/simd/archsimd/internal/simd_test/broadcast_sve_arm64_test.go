// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

// SVE broadcast tests, in the shape of the binary/unary drivers in
// binary_sve_arm64_test.go: broadcast each scalar drawn from the pool and
// check that every lane the hardware actually populated equals the scalar.

package simd_test

import (
	"simd/archsimd"
	"testing"
)

// testSVEBroadcast drives a scalable broadcast constructor. active is the
// runtime number of live lanes (from the vector's Len()).
func testSVEBroadcast[T number, V any](t *testing.T, pool []T, elemBytes, active int,
	f func(T) V, store func(V, []T)) {
	t.Helper()
	count := sveMaxBytes / elemBytes // lanes in the fixed backing array
	for _, x := range pool {
		g := make([]T, count)
		store(f(x), g)
		w := make([]T, count)
		for i := range w {
			w[i] = x
		}
		if !checkSlicesLogInput(t, g[:active], w[:active], 0.0, func() {
			t.Helper()
			t.Logf("x=%v", x)
		}) {
			return
		}
	}
}

func TestBroadcastSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testSVEBroadcast(t, int8s, 1, archsimd.Int8s{}.Len(), archsimd.BroadcastInt8s, archsimd.Int8s.Store)
	testSVEBroadcast(t, int16s, 2, archsimd.Int16s{}.Len(), archsimd.BroadcastInt16s, archsimd.Int16s.Store)
	testSVEBroadcast(t, int32s, 4, archsimd.Int32s{}.Len(), archsimd.BroadcastInt32s, archsimd.Int32s.Store)
	testSVEBroadcast(t, int64s, 8, archsimd.Int64s{}.Len(), archsimd.BroadcastInt64s, archsimd.Int64s.Store)
	testSVEBroadcast(t, uint8s, 1, archsimd.Uint8s{}.Len(), archsimd.BroadcastUint8s, archsimd.Uint8s.Store)
	testSVEBroadcast(t, uint16s, 2, archsimd.Uint16s{}.Len(), archsimd.BroadcastUint16s, archsimd.Uint16s.Store)
	testSVEBroadcast(t, uint32s, 4, archsimd.Uint32s{}.Len(), archsimd.BroadcastUint32s, archsimd.Uint32s.Store)
	testSVEBroadcast(t, uint64s, 8, archsimd.Uint64s{}.Len(), archsimd.BroadcastUint64s, archsimd.Uint64s.Store)
	testSVEBroadcast(t, float32s, 4, archsimd.Float32s{}.Len(), archsimd.BroadcastFloat32s, archsimd.Float32s.Store)
	testSVEBroadcast(t, float64s, 8, archsimd.Float64s{}.Len(), archsimd.BroadcastFloat64s, archsimd.Float64s.Store)
}
