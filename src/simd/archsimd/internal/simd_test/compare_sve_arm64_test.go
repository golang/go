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

func testUint16sCompare(t *testing.T, cmp func(_, _ archsimd.Uint16s) archsimd.Mask16s, want func(_, _ uint16) bool) {
	var z archsimd.Uint16s
	testSVECompare(t, uint16s, 2, z.Len(), archsimd.LoadUint16s, cmp, archsimd.Mask16s.Store, want)
}

func testUint32sCompare(t *testing.T, cmp func(_, _ archsimd.Uint32s) archsimd.Mask32s, want func(_, _ uint32) bool) {
	var z archsimd.Uint32s
	testSVECompare(t, uint32s, 4, z.Len(), archsimd.LoadUint32s, cmp, archsimd.Mask32s.Store, want)
}

func testUint64sCompare(t *testing.T, cmp func(_, _ archsimd.Uint64s) archsimd.Mask64s, want func(_, _ uint64) bool) {
	var z archsimd.Uint64s
	testSVECompare(t, uint64s, 8, z.Len(), archsimd.LoadUint64s, cmp, archsimd.Mask64s.Store, want)
}

func gtWant[T number](a, b T) bool { return a > b }
func geWant[T number](a, b T) bool { return a >= b }
func eqWant[T number](a, b T) bool { return a == b }
func neWant[T number](a, b T) bool { return a != b }

func TestGreaterSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testInt8sCompare(t, archsimd.Int8s.Greater, gtWant[int8])
	testInt16sCompare(t, archsimd.Int16s.Greater, gtWant[int16])
	testInt32sCompare(t, archsimd.Int32s.Greater, gtWant[int32])
	testInt64sCompare(t, archsimd.Int64s.Greater, gtWant[int64])
	testUint8sCompare(t, archsimd.Uint8s.Greater, gtWant[uint8])
	testUint16sCompare(t, archsimd.Uint16s.Greater, gtWant[uint16])
	testUint32sCompare(t, archsimd.Uint32s.Greater, gtWant[uint32])
	testUint64sCompare(t, archsimd.Uint64s.Greater, gtWant[uint64])
	testFloat32sCompare(t, archsimd.Float32s.Greater, gtWant[float32])
	testFloat64sCompare(t, archsimd.Float64s.Greater, gtWant[float64])
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

func TestEqualSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testInt8sCompare(t, archsimd.Int8s.Equal, eqWant[int8])
	testInt16sCompare(t, archsimd.Int16s.Equal, eqWant[int16])
	testInt32sCompare(t, archsimd.Int32s.Equal, eqWant[int32])
	testInt64sCompare(t, archsimd.Int64s.Equal, eqWant[int64])
	testUint8sCompare(t, archsimd.Uint8s.Equal, eqWant[uint8])
	testUint16sCompare(t, archsimd.Uint16s.Equal, eqWant[uint16])
	testUint32sCompare(t, archsimd.Uint32s.Equal, eqWant[uint32])
	testUint64sCompare(t, archsimd.Uint64s.Equal, eqWant[uint64])
	testFloat32sCompare(t, archsimd.Float32s.Equal, eqWant[float32])
	testFloat64sCompare(t, archsimd.Float64s.Equal, eqWant[float64])
}

func TestNotEqualSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testInt8sCompare(t, archsimd.Int8s.NotEqual, neWant[int8])
	testInt16sCompare(t, archsimd.Int16s.NotEqual, neWant[int16])
	testInt32sCompare(t, archsimd.Int32s.NotEqual, neWant[int32])
	testInt64sCompare(t, archsimd.Int64s.NotEqual, neWant[int64])
	testUint8sCompare(t, archsimd.Uint8s.NotEqual, neWant[uint8])
	testUint16sCompare(t, archsimd.Uint16s.NotEqual, neWant[uint16])
	testUint32sCompare(t, archsimd.Uint32s.NotEqual, neWant[uint32])
	testUint64sCompare(t, archsimd.Uint64s.NotEqual, neWant[uint64])
	testFloat32sCompare(t, archsimd.Float32s.NotEqual, neWant[float32])
	testFloat64sCompare(t, archsimd.Float64s.NotEqual, neWant[float64])
}

func TestGreaterEqualSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testInt8sCompare(t, archsimd.Int8s.GreaterEqual, geWant[int8])
	testInt16sCompare(t, archsimd.Int16s.GreaterEqual, geWant[int16])
	testInt32sCompare(t, archsimd.Int32s.GreaterEqual, geWant[int32])
	testInt64sCompare(t, archsimd.Int64s.GreaterEqual, geWant[int64])
	testUint8sCompare(t, archsimd.Uint8s.GreaterEqual, geWant[uint8])
	testUint16sCompare(t, archsimd.Uint16s.GreaterEqual, geWant[uint16])
	testUint32sCompare(t, archsimd.Uint32s.GreaterEqual, geWant[uint32])
	testUint64sCompare(t, archsimd.Uint64s.GreaterEqual, geWant[uint64])
	testFloat32sCompare(t, archsimd.Float32s.GreaterEqual, geWant[float32])
	testFloat64sCompare(t, archsimd.Float64s.GreaterEqual, geWant[float64])
}
