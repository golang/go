// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestBroadcastUint32x4(t *testing.T) {
	s := make([]uint32, 4, 4)
	archsimd.BroadcastUint32x4(123456789).Store(s)
	checkSlices(t, s, []uint32{123456789, 123456789, 123456789, 123456789})
}

func TestBroadcastFloat32x4(t *testing.T) {
	s := make([]float32, 4, 4)
	archsimd.BroadcastFloat32x4(3.14).Store(s)
	checkSlices(t, s, []float32{3.14, 3.14, 3.14, 3.14})
}

func TestBroadcastFloat64x2(t *testing.T) {
	s := make([]float64, 2, 2)
	archsimd.BroadcastFloat64x2(3.14).Store(s)
	checkSlices(t, s, []float64{3.14, 3.14})
}

func TestBroadcastUint64x2(t *testing.T) {
	s := make([]uint64, 2, 2)
	archsimd.BroadcastUint64x2(123456789012345).Store(s)
	checkSlices(t, s, []uint64{123456789012345, 123456789012345})
}

func TestString(t *testing.T) {
	x := archsimd.LoadUint32x4([]uint32{0, 1, 2, 3})
	y := archsimd.LoadInt64x2([]int64{-44, -5})
	z := archsimd.LoadFloat32x4([]float32{0.5, 1.5, -2.5, 3.5e9})
	w := archsimd.LoadFloat64x2([]float64{-2.5, 3.5e9})

	sx := "{0,1,2,3}"
	sy := "{-44,-5}"
	sz := "{0.5,1.5,-2.5,3.5e+09}"
	sw := "{-2.5,3.5e+09}"

	if x.String() != sx {
		t.Errorf("x=%s wanted %s", x, sx)
	}
	if y.String() != sy {
		t.Errorf("y=%s wanted %s", y, sy)
	}
	if z.String() != sz {
		t.Errorf("z=%s wanted %s", z, sz)
	}
	if w.String() != sw {
		t.Errorf("w=%s wanted %s", w, sw)
	}
	t.Logf("w=%s", w)
	t.Logf("x=%s", x)
	t.Logf("y=%s", y)
	t.Logf("z=%s", z)
}

func TestBroadcastUint16x8(t *testing.T) {
	s := make([]uint16, 8, 8)
	archsimd.BroadcastUint16x8(12345).Store(s)
	checkSlices(t, s, []uint16{12345, 12345, 12345, 12345, 12345, 12345, 12345, 12345})
}

func TestBroadcastInt8x16(t *testing.T) {
	s := make([]int8, 16, 16)
	archsimd.BroadcastInt8x16(-123).Store(s)
	checkSlices(t, s, []int8{-123, -123, -123, -123, -123, -123, -123, -123,
		-123, -123, -123, -123, -123, -123, -123, -123})
}

func TestBroadcastUint8x16(t *testing.T) {
	s := make([]uint8, 16, 16)
	archsimd.BroadcastUint8x16(200).Store(s)
	checkSlices(t, s, []uint8{200, 200, 200, 200, 200, 200, 200, 200,
		200, 200, 200, 200, 200, 200, 200, 200})
}

func TestBroadcastInt16x8(t *testing.T) {
	s := make([]int16, 8, 8)
	archsimd.BroadcastInt16x8(-12345).Store(s)
	checkSlices(t, s, []int16{-12345, -12345, -12345, -12345, -12345, -12345, -12345, -12345})
}

func TestBroadcastInt32x4(t *testing.T) {
	s := make([]int32, 4, 4)
	archsimd.BroadcastInt32x4(-123456789).Store(s)
	checkSlices(t, s, []int32{-123456789, -123456789, -123456789, -123456789})
}

func TestBroadcastInt64x2(t *testing.T) {
	s := make([]int64, 2, 2)
	archsimd.BroadcastInt64x2(-123456789).Store(s)
	checkSlices(t, s, []int64{-123456789, -123456789})
}

func TestLookupOrZero(t *testing.T) {
	// Out-of-range indices produce zero lane value.
	x := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	indices := []uint8{7, 6, 5, 4, 3, 2, 1, 0, 0xff, 8, 16, 9, 128, 10, 20, 11}
	want := []uint8{8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 0, 10, 0, 11, 0, 12}
	got := make([]uint8, len(x))
	archsimd.LoadUint8x16(x).LookupOrZero(archsimd.LoadUint8x16(indices)).Store(got)
	checkSlices(t, got, want)
}

func TestLookupOrKeep(t *testing.T) {
	// Out-of-range indices keep the existing (receiver) lane value.
	existing := []int8{100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115}
	table := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	indices := []int8{0, 1, 2, 3, -1, -1, 6, 7, 8, 9, -1, -1, 12, 13, 14, 15}
	want := []int8{1, 2, 3, 4, 104, 105, 7, 8, 9, 10, 110, 111, 13, 14, 15, 16}
	got := make([]int8, 16)
	archsimd.LoadInt8x16(existing).LookupOrKeep(
		archsimd.LoadInt8x16(table),
		archsimd.LoadInt8x16(indices),
	).Store(got)
	checkSlices(t, got, want)
}

func TestClMul(t *testing.T) {
	var x = archsimd.LoadUint64x2([]uint64{1, 5})
	var y = archsimd.LoadUint64x2([]uint64{3, 9})

	foo := func(v archsimd.Uint64x2, s []uint64) {
		r := make([]uint64, 2, 2)
		v.Store(r)
		checkSlices[uint64](t, r, s)
	}

	foo(x.CarrylessMultiplyEven(y), []uint64{3, 0})
	foo(x.CarrylessMultiplyEvenOdd(y), []uint64{9, 0})
	foo(x.CarrylessMultiplyOddEven(y), []uint64{15, 0})
	foo(x.CarrylessMultiplyOdd(y), []uint64{45, 0})
	foo(y.CarrylessMultiplyEven(y), []uint64{5, 0})
}
