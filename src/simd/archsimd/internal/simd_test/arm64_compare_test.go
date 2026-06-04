// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

// --- Equal (hardware: CMEQ / FCMEQ) ---

func TestEqual(t *testing.T) {
	testInt8x16Compare(t, archsimd.Int8x16.Equal, equalSlice[int8])
	testInt16x8Compare(t, archsimd.Int16x8.Equal, equalSlice[int16])
	testInt32x4Compare(t, archsimd.Int32x4.Equal, equalSlice[int32])
	testInt64x2Compare(t, archsimd.Int64x2.Equal, equalSlice[int64])

	testUint8x16Compare(t, archsimd.Uint8x16.Equal, equalSlice[uint8])
	testUint16x8Compare(t, archsimd.Uint16x8.Equal, equalSlice[uint16])
	testUint32x4Compare(t, archsimd.Uint32x4.Equal, equalSlice[uint32])
	testUint64x2Compare(t, archsimd.Uint64x2.Equal, equalSlice[uint64])

	testFloat32x4Compare(t, archsimd.Float32x4.Equal, equalSlice[float32])
	testFloat64x2Compare(t, archsimd.Float64x2.Equal, equalSlice[float64])
}

// --- Greater (hardware: CMGT/CMHI for signed/unsigned, FCMGT for float) ---

func TestGreater(t *testing.T) {
	testInt8x16Compare(t, archsimd.Int8x16.Greater, greaterSlice[int8])
	testInt16x8Compare(t, archsimd.Int16x8.Greater, greaterSlice[int16])
	testInt32x4Compare(t, archsimd.Int32x4.Greater, greaterSlice[int32])
	testInt64x2Compare(t, archsimd.Int64x2.Greater, greaterSlice[int64])

	testUint8x16Compare(t, archsimd.Uint8x16.Greater, greaterSlice[uint8])
	testUint16x8Compare(t, archsimd.Uint16x8.Greater, greaterSlice[uint16])
	testUint32x4Compare(t, archsimd.Uint32x4.Greater, greaterSlice[uint32])
	testUint64x2Compare(t, archsimd.Uint64x2.Greater, greaterSlice[uint64])

	testFloat32x4Compare(t, archsimd.Float32x4.Greater, greaterSlice[float32])
	testFloat64x2Compare(t, archsimd.Float64x2.Greater, greaterSlice[float64])
}

// --- GreaterEqual (hardware: CMGE/CMHS for signed/unsigned, FCMGE for float) ---

func TestGreaterEqual(t *testing.T) {
	testInt8x16Compare(t, archsimd.Int8x16.GreaterEqual, greaterEqualSlice[int8])
	testInt16x8Compare(t, archsimd.Int16x8.GreaterEqual, greaterEqualSlice[int16])
	testInt32x4Compare(t, archsimd.Int32x4.GreaterEqual, greaterEqualSlice[int32])
	testInt64x2Compare(t, archsimd.Int64x2.GreaterEqual, greaterEqualSlice[int64])

	testUint8x16Compare(t, archsimd.Uint8x16.GreaterEqual, greaterEqualSlice[uint8])
	testUint16x8Compare(t, archsimd.Uint16x8.GreaterEqual, greaterEqualSlice[uint16])
	testUint32x4Compare(t, archsimd.Uint32x4.GreaterEqual, greaterEqualSlice[uint32])
	testUint64x2Compare(t, archsimd.Uint64x2.GreaterEqual, greaterEqualSlice[uint64])

	testFloat32x4Compare(t, archsimd.Float32x4.GreaterEqual, greaterEqualSlice[float32])
	testFloat64x2Compare(t, archsimd.Float64x2.GreaterEqual, greaterEqualSlice[float64])
}

// --- Less (derived: y.Greater(x)) ---

func TestLess(t *testing.T) {
	testInt8x16Compare(t, archsimd.Int8x16.Less, lessSlice[int8])
	testInt16x8Compare(t, archsimd.Int16x8.Less, lessSlice[int16])
	testInt32x4Compare(t, archsimd.Int32x4.Less, lessSlice[int32])
	testInt64x2Compare(t, archsimd.Int64x2.Less, lessSlice[int64])

	testUint8x16Compare(t, archsimd.Uint8x16.Less, lessSlice[uint8])
	testUint16x8Compare(t, archsimd.Uint16x8.Less, lessSlice[uint16])
	testUint32x4Compare(t, archsimd.Uint32x4.Less, lessSlice[uint32])
	testUint64x2Compare(t, archsimd.Uint64x2.Less, lessSlice[uint64])

	testFloat32x4Compare(t, archsimd.Float32x4.Less, lessSlice[float32])
	testFloat64x2Compare(t, archsimd.Float64x2.Less, lessSlice[float64])
}

// --- LessEqual (derived: y.GreaterEqual(x)) ---

func TestLessEqual(t *testing.T) {
	testInt8x16Compare(t, archsimd.Int8x16.LessEqual, lessEqualSlice[int8])
	testInt16x8Compare(t, archsimd.Int16x8.LessEqual, lessEqualSlice[int16])
	testInt32x4Compare(t, archsimd.Int32x4.LessEqual, lessEqualSlice[int32])
	testInt64x2Compare(t, archsimd.Int64x2.LessEqual, lessEqualSlice[int64])

	testUint8x16Compare(t, archsimd.Uint8x16.LessEqual, lessEqualSlice[uint8])
	testUint16x8Compare(t, archsimd.Uint16x8.LessEqual, lessEqualSlice[uint16])
	testUint32x4Compare(t, archsimd.Uint32x4.LessEqual, lessEqualSlice[uint32])
	testUint64x2Compare(t, archsimd.Uint64x2.LessEqual, lessEqualSlice[uint64])

	testFloat32x4Compare(t, archsimd.Float32x4.LessEqual, lessEqualSlice[float32])
	testFloat64x2Compare(t, archsimd.Float64x2.LessEqual, lessEqualSlice[float64])
}

// --- NotEqual (derived: Equal().Not()) ---

func TestNotEqual(t *testing.T) {
	testInt8x16Compare(t, archsimd.Int8x16.NotEqual, notEqualSlice[int8])
	testInt16x8Compare(t, archsimd.Int16x8.NotEqual, notEqualSlice[int16])
	testInt32x4Compare(t, archsimd.Int32x4.NotEqual, notEqualSlice[int32])
	testInt64x2Compare(t, archsimd.Int64x2.NotEqual, notEqualSlice[int64])

	testUint8x16Compare(t, archsimd.Uint8x16.NotEqual, notEqualSlice[uint8])
	testUint16x8Compare(t, archsimd.Uint16x8.NotEqual, notEqualSlice[uint16])
	testUint32x4Compare(t, archsimd.Uint32x4.NotEqual, notEqualSlice[uint32])
	testUint64x2Compare(t, archsimd.Uint64x2.NotEqual, notEqualSlice[uint64])

	testFloat32x4Compare(t, archsimd.Float32x4.NotEqual, notEqualSlice[float32])
	testFloat64x2Compare(t, archsimd.Float64x2.NotEqual, notEqualSlice[float64])
}

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
