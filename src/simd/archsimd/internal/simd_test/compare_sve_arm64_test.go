// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

// SVE mask (predicate) tests, in the same utility-function shape as
// compare_amd64_test.go.

package simd_test

import (
	"simd/archsimd"
	"testing"
)

const sveMaskUint16s = sveMaxBytes / 8

// testSVECompare drives a scalable compare that returns a mask. active is the
// runtime number of live lanes; elemBytes is the element width. Lane i is true
// iff bit i*elemBytes of the stored predicate is set.
func testSVECompare[T number, V, M any](t *testing.T, pool []T, elemBytes, active int,
	load func([]T) V, cmp func(V, V) M, store func(M, []uint16), want func(T, T) bool) {
	t.Helper()
	count := sveMaxBytes / elemBytes
	forSlicePair(t, pool, count, func(x, y []T) bool {
		t.Helper()
		bits := make([]uint16, sveMaskUint16s)
		store(cmp(load(x), load(y)), bits)
		for i := 0; i < active; i++ {
			b := i * elemBytes
			got := bits[b/16]>>uint(b%16)&1 == 1
			if got != want(x[i], y[i]) {
				t.Errorf("lane %d: got %v, want %v (x=%v y=%v)", i, got, want(x[i], y[i]), x[i], y[i])
				return false
			}
		}
		return true
	})
}

func testInt8sCompare(t *testing.T, cmp func(_, _ archsimd.Int8s) archsimd.Mask8s, want func(_, _ int8) bool) {
	var z archsimd.Int8s
	testSVECompare(t, int8s, 1, z.Len(), archsimd.LoadInt8s, cmp, archsimd.Mask8s.Store, want)
}

func testInt16sCompare(t *testing.T, cmp func(_, _ archsimd.Int16s) archsimd.Mask16s, want func(_, _ int16) bool) {
	var z archsimd.Int16s
	testSVECompare(t, int16s, 2, z.Len(), archsimd.LoadInt16s, cmp, archsimd.Mask16s.Store, want)
}

func testInt32sCompare(t *testing.T, cmp func(_, _ archsimd.Int32s) archsimd.Mask32s, want func(_, _ int32) bool) {
	var z archsimd.Int32s
	testSVECompare(t, int32s, 4, z.Len(), archsimd.LoadInt32s, cmp, archsimd.Mask32s.Store, want)
}

func testInt64sCompare(t *testing.T, cmp func(_, _ archsimd.Int64s) archsimd.Mask64s, want func(_, _ int64) bool) {
	var z archsimd.Int64s
	testSVECompare(t, int64s, 8, z.Len(), archsimd.LoadInt64s, cmp, archsimd.Mask64s.Store, want)
}

func testUint8sCompare(t *testing.T, cmp func(_, _ archsimd.Uint8s) archsimd.Mask8s, want func(_, _ uint8) bool) {
	var z archsimd.Uint8s
	testSVECompare(t, uint8s, 1, z.Len(), archsimd.LoadUint8s, cmp, archsimd.Mask8s.Store, want)
}

func testFloat32sCompare(t *testing.T, cmp func(_, _ archsimd.Float32s) archsimd.Mask32s, want func(_, _ float32) bool) {
	var z archsimd.Float32s
	testSVECompare(t, float32s, 4, z.Len(), archsimd.LoadFloat32s, cmp, archsimd.Mask32s.Store, want)
}

func testFloat64sCompare(t *testing.T, cmp func(_, _ archsimd.Float64s) archsimd.Mask64s, want func(_, _ float64) bool) {
	var z archsimd.Float64s
	testSVECompare(t, float64s, 8, z.Len(), archsimd.LoadFloat64s, cmp, archsimd.Mask64s.Store, want)
}

func TestGreaterSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no sve")
	}
	testInt8sCompare(t, archsimd.Int8s.Greater, func(a, b int8) bool { return a > b })
	testInt16sCompare(t, archsimd.Int16s.Greater, func(a, b int16) bool { return a > b })
	testInt32sCompare(t, archsimd.Int32s.Greater, func(a, b int32) bool { return a > b })
	testInt64sCompare(t, archsimd.Int64s.Greater, func(a, b int64) bool { return a > b })
	testUint8sCompare(t, archsimd.Uint8s.Greater, func(a, b uint8) bool { return a > b })
	testFloat32sCompare(t, archsimd.Float32s.Greater, func(a, b float32) bool { return a > b })
	testFloat64sCompare(t, archsimd.Float64s.Greater, func(a, b float64) bool { return a > b })
}

// TestMaskStoreLoadPanicSVE checks that the exported mask memory APIs panic when
// the bits slice is too short to hold the whole predicate.
func TestMaskStoreLoadPanicSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no sve")
	}
	var z archsimd.Int8s
	m := z.Greater(z)
	mustPanic(t, "Store short", func() { m.Store(nil) })
	mustPanic(t, "LoadMask8s short", func() { archsimd.LoadMask8s(nil) })
	// A slice long enough for the whole predicate must not panic.
	bits := make([]uint16, sveMaskUint16s)
	m.Store(bits)
	archsimd.LoadMask8s(bits)
}
