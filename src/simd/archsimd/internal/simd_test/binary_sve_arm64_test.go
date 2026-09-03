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

func testUint16sBinary(t *testing.T, f func(_, _ archsimd.Uint16s) archsimd.Uint16s, want func(_, _ []uint16) []uint16) {
	var z archsimd.Uint16s
	testSVEBinary(t, uint16s, 2, z.Len(), archsimd.LoadUint16s, f, archsimd.Uint16s.Store, want)
}

func testUint32sBinary(t *testing.T, f func(_, _ archsimd.Uint32s) archsimd.Uint32s, want func(_, _ []uint32) []uint32) {
	var z archsimd.Uint32s
	testSVEBinary(t, uint32s, 4, z.Len(), archsimd.LoadUint32s, f, archsimd.Uint32s.Store, want)
}

func testUint64sBinary(t *testing.T, f func(_, _ archsimd.Uint64s) archsimd.Uint64s, want func(_, _ []uint64) []uint64) {
	var z archsimd.Uint64s
	testSVEBinary(t, uint64s, 8, z.Len(), archsimd.LoadUint64s, f, archsimd.Uint64s.Store, want)
}

func testInt8sUnary(t *testing.T, f func(archsimd.Int8s) archsimd.Int8s, want func([]int8) []int8) {
	var z archsimd.Int8s
	testSVEUnary(t, int8s, 1, z.Len(), archsimd.LoadInt8s, f, archsimd.Int8s.Store, want)
}

func testInt16sUnary(t *testing.T, f func(archsimd.Int16s) archsimd.Int16s, want func([]int16) []int16) {
	var z archsimd.Int16s
	testSVEUnary(t, int16s, 2, z.Len(), archsimd.LoadInt16s, f, archsimd.Int16s.Store, want)
}

func testInt32sUnary(t *testing.T, f func(archsimd.Int32s) archsimd.Int32s, want func([]int32) []int32) {
	var z archsimd.Int32s
	testSVEUnary(t, int32s, 4, z.Len(), archsimd.LoadInt32s, f, archsimd.Int32s.Store, want)
}

func testInt64sUnary(t *testing.T, f func(archsimd.Int64s) archsimd.Int64s, want func([]int64) []int64) {
	var z archsimd.Int64s
	testSVEUnary(t, int64s, 8, z.Len(), archsimd.LoadInt64s, f, archsimd.Int64s.Store, want)
}

func testUint8sUnary(t *testing.T, f func(archsimd.Uint8s) archsimd.Uint8s, want func([]uint8) []uint8) {
	var z archsimd.Uint8s
	testSVEUnary(t, uint8s, 1, z.Len(), archsimd.LoadUint8s, f, archsimd.Uint8s.Store, want)
}

func testUint16sUnary(t *testing.T, f func(archsimd.Uint16s) archsimd.Uint16s, want func([]uint16) []uint16) {
	var z archsimd.Uint16s
	testSVEUnary(t, uint16s, 2, z.Len(), archsimd.LoadUint16s, f, archsimd.Uint16s.Store, want)
}

func testUint32sUnary(t *testing.T, f func(archsimd.Uint32s) archsimd.Uint32s, want func([]uint32) []uint32) {
	var z archsimd.Uint32s
	testSVEUnary(t, uint32s, 4, z.Len(), archsimd.LoadUint32s, f, archsimd.Uint32s.Store, want)
}

func testUint64sUnary(t *testing.T, f func(archsimd.Uint64s) archsimd.Uint64s, want func([]uint64) []uint64) {
	var z archsimd.Uint64s
	testSVEUnary(t, uint64s, 8, z.Len(), archsimd.LoadUint64s, f, archsimd.Uint64s.Store, want)
}

func testFloat32sUnary(t *testing.T, f func(archsimd.Float32s) archsimd.Float32s, want func([]float32) []float32) {
	var z archsimd.Float32s
	testSVEUnary(t, float32s, 4, z.Len(), archsimd.LoadFloat32s, f, archsimd.Float32s.Store, want)
}

func testFloat64sUnary(t *testing.T, f func(archsimd.Float64s) archsimd.Float64s, want func([]float64) []float64) {
	var z archsimd.Float64s
	testSVEUnary(t, float64s, 8, z.Len(), archsimd.LoadFloat64s, f, archsimd.Float64s.Store, want)
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
	testUint16sBinary(t, archsimd.Uint16s.Add, addSlice[uint16])
	testUint32sBinary(t, archsimd.Uint32s.Add, addSlice[uint32])
	testUint64sBinary(t, archsimd.Uint64s.Add, addSlice[uint64])
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
	testUint16sBinary(t, archsimd.Uint16s.Sub, subSlice[uint16])
	testUint32sBinary(t, archsimd.Uint32s.Sub, subSlice[uint32])
	testUint64sBinary(t, archsimd.Uint64s.Sub, subSlice[uint64])
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
	testInt8sUnary(t, archsimd.Int8s.Abs, absSlice[int8])
	testInt16sUnary(t, archsimd.Int16s.Abs, absSlice[int16])
	testInt32sUnary(t, archsimd.Int32s.Abs, absSlice[int32])
	testInt64sUnary(t, archsimd.Int64s.Abs, absSlice[int64])
	testFloat32sUnary(t, archsimd.Float32s.Abs, absSlice[float32])
	testFloat64sUnary(t, archsimd.Float64s.Abs, absSlice[float64])
}

func TestNegSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testInt8sUnary(t, archsimd.Int8s.Neg, negSlice[int8])
	testInt16sUnary(t, archsimd.Int16s.Neg, negSlice[int16])
	testInt32sUnary(t, archsimd.Int32s.Neg, negSlice[int32])
	testInt64sUnary(t, archsimd.Int64s.Neg, negSlice[int64])
	testFloat32sUnary(t, archsimd.Float32s.Neg, negSlice[float32])
	testFloat64sUnary(t, archsimd.Float64s.Neg, negSlice[float64])
}

func TestSqrtSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testFloat32sUnary(t, archsimd.Float32s.Sqrt, sqrtSlice[float32])
	testFloat64sUnary(t, archsimd.Float64s.Sqrt, sqrtSlice[float64])
}

func TestCeilSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testFloat32sUnary(t, archsimd.Float32s.Ceil, ceilSlice[float32])
	testFloat64sUnary(t, archsimd.Float64s.Ceil, ceilSlice[float64])
}

func TestFloorSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testFloat32sUnary(t, archsimd.Float32s.Floor, floorSlice[float32])
	testFloat64sUnary(t, archsimd.Float64s.Floor, floorSlice[float64])
}

func TestTruncSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testFloat32sUnary(t, archsimd.Float32s.Trunc, truncSlice[float32])
	testFloat64sUnary(t, archsimd.Float64s.Trunc, truncSlice[float64])
}

func TestRoundSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testFloat32sUnary(t, archsimd.Float32s.Round, roundSlice[float32])
	testFloat64sUnary(t, archsimd.Float64s.Round, roundSlice[float64])
}

func TestAndSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testInt8sBinary(t, archsimd.Int8s.And, andSlice[int8])
	testInt16sBinary(t, archsimd.Int16s.And, andSlice[int16])
	testInt32sBinary(t, archsimd.Int32s.And, andSlice[int32])
	testInt64sBinary(t, archsimd.Int64s.And, andSlice[int64])
	testUint8sBinary(t, archsimd.Uint8s.And, andSlice[uint8])
	testUint16sBinary(t, archsimd.Uint16s.And, andSlice[uint16])
	testUint32sBinary(t, archsimd.Uint32s.And, andSlice[uint32])
	testUint64sBinary(t, archsimd.Uint64s.And, andSlice[uint64])
}

func TestOrSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testInt8sBinary(t, archsimd.Int8s.Or, orSlice[int8])
	testInt16sBinary(t, archsimd.Int16s.Or, orSlice[int16])
	testInt32sBinary(t, archsimd.Int32s.Or, orSlice[int32])
	testInt64sBinary(t, archsimd.Int64s.Or, orSlice[int64])
	testUint8sBinary(t, archsimd.Uint8s.Or, orSlice[uint8])
	testUint16sBinary(t, archsimd.Uint16s.Or, orSlice[uint16])
	testUint32sBinary(t, archsimd.Uint32s.Or, orSlice[uint32])
	testUint64sBinary(t, archsimd.Uint64s.Or, orSlice[uint64])
}

func TestXorSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testInt8sBinary(t, archsimd.Int8s.Xor, xorSlice[int8])
	testInt16sBinary(t, archsimd.Int16s.Xor, xorSlice[int16])
	testInt32sBinary(t, archsimd.Int32s.Xor, xorSlice[int32])
	testInt64sBinary(t, archsimd.Int64s.Xor, xorSlice[int64])
	testUint8sBinary(t, archsimd.Uint8s.Xor, xorSlice[uint8])
	testUint16sBinary(t, archsimd.Uint16s.Xor, xorSlice[uint16])
	testUint32sBinary(t, archsimd.Uint32s.Xor, xorSlice[uint32])
	testUint64sBinary(t, archsimd.Uint64s.Xor, xorSlice[uint64])
}

func TestAndNotSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testInt8sBinary(t, archsimd.Int8s.AndNot, andNotSlice[int8])
	testInt16sBinary(t, archsimd.Int16s.AndNot, andNotSlice[int16])
	testInt32sBinary(t, archsimd.Int32s.AndNot, andNotSlice[int32])
	testInt64sBinary(t, archsimd.Int64s.AndNot, andNotSlice[int64])
	testUint8sBinary(t, archsimd.Uint8s.AndNot, andNotSlice[uint8])
	testUint16sBinary(t, archsimd.Uint16s.AndNot, andNotSlice[uint16])
	testUint32sBinary(t, archsimd.Uint32s.AndNot, andNotSlice[uint32])
	testUint64sBinary(t, archsimd.Uint64s.AndNot, andNotSlice[uint64])
}

func TestMulSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testFloat32sBinary(t, archsimd.Float32s.Mul, mulSlice[float32])
	testFloat64sBinary(t, archsimd.Float64s.Mul, mulSlice[float64])
	// Ungated: integer Mul compiles to the merging-predicated fallback,
	// correct on any SVE.
	testInt8sBinary(t, archsimd.Int8s.Mul, mulSlice[int8])
	testInt16sBinary(t, archsimd.Int16s.Mul, mulSlice[int16])
	testInt32sBinary(t, archsimd.Int32s.Mul, mulSlice[int32])
	testInt64sBinary(t, archsimd.Int64s.Mul, mulSlice[int64])
	testUint8sBinary(t, archsimd.Uint8s.Mul, mulSlice[uint8])
	testUint16sBinary(t, archsimd.Uint16s.Mul, mulSlice[uint16])
	testUint32sBinary(t, archsimd.Uint32s.Mul, mulSlice[uint32])
	testUint64sBinary(t, archsimd.Uint64s.Mul, mulSlice[uint64])
	if archsimd.ARM64.SVE2() {
		// Gated: this block compiles to the unpredicated SVE2 encoding.
		testInt8sBinary(t, archsimd.Int8s.Mul, mulSlice[int8])
		testInt16sBinary(t, archsimd.Int16s.Mul, mulSlice[int16])
		testInt32sBinary(t, archsimd.Int32s.Mul, mulSlice[int32])
		testInt64sBinary(t, archsimd.Int64s.Mul, mulSlice[int64])
		testUint8sBinary(t, archsimd.Uint8s.Mul, mulSlice[uint8])
		testUint16sBinary(t, archsimd.Uint16s.Mul, mulSlice[uint16])
		testUint32sBinary(t, archsimd.Uint32s.Mul, mulSlice[uint32])
		testUint64sBinary(t, archsimd.Uint64s.Mul, mulSlice[uint64])
	}
}

func TestMulHighSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	// Ungated: compiles to the merging-predicated fallback, correct on any SVE.
	testInt8sBinary(t, archsimd.Int8s.MulHigh, mulHighSlice[int8])
	testInt16sBinary(t, archsimd.Int16s.MulHigh, mulHighSlice[int16])
	testInt32sBinary(t, archsimd.Int32s.MulHigh, mulHighSlice[int32])
	testInt64sBinary(t, archsimd.Int64s.MulHigh, mulHighSlice[int64])
	testUint8sBinary(t, archsimd.Uint8s.MulHigh, mulHighSlice[uint8])
	testUint16sBinary(t, archsimd.Uint16s.MulHigh, mulHighSlice[uint16])
	testUint32sBinary(t, archsimd.Uint32s.MulHigh, mulHighSlice[uint32])
	testUint64sBinary(t, archsimd.Uint64s.MulHigh, mulHighSlice[uint64])
	if archsimd.ARM64.SVE2() {
		// Gated: this block compiles to the unpredicated SVE2 encodings.
		testInt8sBinary(t, archsimd.Int8s.MulHigh, mulHighSlice[int8])
		testInt16sBinary(t, archsimd.Int16s.MulHigh, mulHighSlice[int16])
		testInt32sBinary(t, archsimd.Int32s.MulHigh, mulHighSlice[int32])
		testInt64sBinary(t, archsimd.Int64s.MulHigh, mulHighSlice[int64])
		testUint8sBinary(t, archsimd.Uint8s.MulHigh, mulHighSlice[uint8])
		testUint16sBinary(t, archsimd.Uint16s.MulHigh, mulHighSlice[uint16])
		testUint32sBinary(t, archsimd.Uint32s.MulHigh, mulHighSlice[uint32])
		testUint64sBinary(t, archsimd.Uint64s.MulHigh, mulHighSlice[uint64])
	}
}
