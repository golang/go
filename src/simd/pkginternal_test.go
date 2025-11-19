// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"simd"
	"simd/internal/test_helpers"
	"testing"
)

func TestConcatSelectedConstant64(t *testing.T) {
	a := make([]int64, 2)
	x := simd.LoadInt64x2Slice([]int64{4, 5})
	y := simd.LoadInt64x2Slice([]int64{6, 7})
	z := x.ExportTestConcatSelectedConstant(0b10, y)
	z.StoreSlice(a)
	test_helpers.CheckSlices[int64](t, a, []int64{4, 7})
}

func TestConcatSelectedConstantGrouped64(t *testing.T) {
	a := make([]float64, 4)
	x := simd.LoadFloat64x4Slice([]float64{4, 5, 8, 9})
	y := simd.LoadFloat64x4Slice([]float64{6, 7, 10, 11})
	z := x.ExportTestConcatSelectedConstantGrouped(0b_11_10, y)
	z.StoreSlice(a)
	test_helpers.CheckSlices[float64](t, a, []float64{4, 7, 9, 11})
}

func TestConcatSelectedConstant32(t *testing.T) {
	a := make([]float32, 4)
	x := simd.LoadFloat32x4Slice([]float32{4, 5, 8, 9})
	y := simd.LoadFloat32x4Slice([]float32{6, 7, 10, 11})
	z := x.ExportTestConcatSelectedConstant(0b_11_01_10_00, y)
	z.StoreSlice(a)
	test_helpers.CheckSlices[float32](t, a, []float32{4, 8, 7, 11})
}

func TestConcatSelectedConstantGrouped32(t *testing.T) {
	a := make([]uint32, 8)
	x := simd.LoadUint32x8Slice([]uint32{0, 1, 2, 3, 8, 9, 10, 11})
	y := simd.LoadUint32x8Slice([]uint32{4, 5, 6, 7, 12, 13, 14, 15})
	z := x.ExportTestConcatSelectedConstantGrouped(0b_11_01_00_10, y)
	z.StoreSlice(a)
	test_helpers.CheckSlices[uint32](t, a, []uint32{2, 0, 5, 7, 10, 8, 13, 15})
}

func TestTern(t *testing.T) {
	if !simd.X86.AVX512() {
		t.Skip("This test needs AVX512")
	}
	x := simd.LoadInt32x8Slice([]int32{0, 0, 0, 0, 1, 1, 1, 1})
	y := simd.LoadInt32x8Slice([]int32{0, 0, 1, 1, 0, 0, 1, 1})
	z := simd.LoadInt32x8Slice([]int32{0, 1, 0, 1, 0, 1, 0, 1})

	foo := func(w simd.Int32x8, k uint8) {
		a := make([]int32, 8)
		w.StoreSlice(a)
		t.Logf("For k=%0b, w=%v", k, a)
		for i, b := range a {
			if (int32(k)>>i)&1 != b {
				t.Errorf("Element %d of stored slice (=%d) did not match corresponding bit in 0b%b",
					i, b, k)
			}
		}
	}

	foo(x.ExportTestTern(0b1111_0000, y, z), 0b1111_0000)
	foo(x.ExportTestTern(0b1100_1100, y, z), 0b1100_1100)
	foo(x.ExportTestTern(0b1010_1010, y, z), 0b1010_1010)
}

func TestSelect2x4x32(t *testing.T) {
	for a := range uint8(8) {
		for b := range uint8(8) {
			for c := range uint8(8) {
				for d := range uint8(8) {
					x := simd.LoadInt32x4Slice([]int32{0, 1, 2, 3})
					y := simd.LoadInt32x4Slice([]int32{4, 5, 6, 7})
					z := select2x4x32(x, a, b, c, d, y)
					w := make([]int32, 4, 4)
					z.StoreSlice(w)
					if w[0] != int32(a) || w[1] != int32(b) ||
						w[2] != int32(c) || w[3] != int32(d) {
						t.Errorf("Expected [%d %d %d %d] got %v", a, b, c, d, w)
					}
				}
			}
		}
	}
}

func TestSelect2x8x32Grouped(t *testing.T) {
	for a := range uint8(8) {
		for b := range uint8(8) {
			for c := range uint8(8) {
				for d := range uint8(8) {
					x := simd.LoadInt32x8Slice([]int32{0, 1, 2, 3, 10, 11, 12, 13})
					y := simd.LoadInt32x8Slice([]int32{4, 5, 6, 7, 14, 15, 16, 17})
					z := select2x8x32Grouped(x, a, b, c, d, y)
					w := make([]int32, 8, 8)
					z.StoreSlice(w)
					if w[0] != int32(a) || w[1] != int32(b) ||
						w[2] != int32(c) || w[3] != int32(d) ||
						w[4] != int32(10+a) || w[5] != int32(10+b) ||
						w[6] != int32(10+c) || w[7] != int32(10+d) {
						t.Errorf("Expected [%d %d %d %d %d %d %d %d] got %v", a, b, c, d, 10+a, 10+b, 10+c, 10+d, w)
					}
				}
			}
		}
	}
}

// select2x4x32 returns a selection of 4 elements in x and y, numbered
// 0-7, where 0-3 are the four elements of x and 4-7 are the four elements
// of y.
func select2x4x32(x simd.Int32x4, a, b, c, d uint8, y simd.Int32x4) simd.Int32x4 {
	pattern := a>>2 + (b&4)>>1 + (c & 4) + (d&4)<<1

	a, b, c, d = a&3, b&3, c&3, d&3

	switch pattern {
	case simd.LLLL:
		return x.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(a, b, c, d), x)
	case simd.HHHH:
		return y.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(a, b, c, d), y)
	case simd.LLHH:
		return x.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(a, b, c, d), y)
	case simd.HHLL:
		return y.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(a, b, c, d), x)

	case simd.HLLL:
		z := y.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(a, a, b, b), x)
		return z.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(0, 2, c, d), x)
	case simd.LHLL:
		z := x.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(a, a, b, b), y)
		return z.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(0, 2, c, d), x)

	case simd.HLHH:
		z := y.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(a, a, b, b), x)
		return z.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(0, 2, c, d), y)
	case simd.LHHH:
		z := x.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(a, a, b, b), y)
		return z.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(0, 2, c, d), y)

	case simd.LLLH:
		z := x.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(c, c, d, d), y)
		return x.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(a, b, 0, 2), z)
	case simd.LLHL:
		z := y.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(c, c, d, d), x)
		return x.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(a, b, 0, 2), z)
	case simd.HHLH:
		z := x.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(c, c, d, d), y)
		return y.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(a, b, 0, 2), z)
	case simd.HHHL:
		z := y.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(c, c, d, d), x)
		return y.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(a, b, 0, 2), z)

	case simd.LHLH:
		z := x.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(a, c, b, d), y)
		return z.ExportTestConcatSelectedConstant(0b11_01_10_00 /* =simd.ExportTestCscImm4(0, 2, 1, 3) */, z)
	case simd.HLHL:
		z := x.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(b, d, a, c), y)
		return z.ExportTestConcatSelectedConstant(0b01_11_00_10 /* =simd.ExportTestCscImm4(2, 0, 3, 1) */, z)
	case simd.HLLH:
		z := x.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(b, c, a, d), y)
		return z.ExportTestConcatSelectedConstant(0b11_01_00_10 /* =simd.ExportTestCscImm4(2, 0, 1, 3) */, z)
	case simd.LHHL:
		z := x.ExportTestConcatSelectedConstant(simd.ExportTestCscImm4(a, d, b, c), y)
		return z.ExportTestConcatSelectedConstant(0b01_11_10_00 /* =simd.ExportTestCscImm4(0, 2, 3, 1) */, z)
	}
	panic("missing case, switch should be exhaustive")
}

// select2x8x32Grouped returns a pair of selection of 4 elements in x and y,
// numbered 0-7, where 0-3 are the four elements of x's two groups (lower and
// upper 128 bits) and 4-7 are the four elements of y's two groups.

func select2x8x32Grouped(x simd.Int32x8, a, b, c, d uint8, y simd.Int32x8) simd.Int32x8 {
	// selections as being expressible in the ExportTestConcatSelectedConstant pattern,
	// or not. Classification is by H and L, where H is a selection from 4-7
	// and L is a selection from 0-3.
	// simd.LLHH -> CSC(x,y, a, b, c&3, d&3)
	// simd.HHLL -> CSC(y,x, a&3, b&3, c, d)
	// simd.LLLL -> CSC(x,x, a, b, c, d)
	// simd.HHHH -> CSC(y,y, a&3, b&3, c&3, d&3)

	// simd.LLLH -> z = CSC(x, y, c, c, d&3, d&3); CSC(x, z, a, b, 0, 2)
	// simd.LLHL -> z = CSC(x, y, c&3, c&3, d, d); CSC(x, z, a, b, 0, 2)
	// simd.HHLH -> z = CSC(x, y, c, c, d&3, d&3); CSC(y, z, a&3, b&3, 0, 2)
	// simd.HHHL -> z = CSC(x, y, c&3, c&3, d, d); CSC(y, z, a&3, b&3, 0, 2)

	// simd.LHLL -> z = CSC(x, y, a, a, b&3, b&3); CSC(z, x, 0, 2, c, d)
	// etc

	// simd.LHLH -> z = CSC(x, y, a, c, b&3, d&3); CSC(z, z, 0, 2, 1, 3)
	// simd.HLHL -> z = CSC(x, y, b, d, a&3, c&3); CSC(z, z, 2, 0, 3, 1)

	pattern := a>>2 + (b&4)>>1 + (c & 4) + (d&4)<<1

	a, b, c, d = a&3, b&3, c&3, d&3

	switch pattern {
	case simd.LLLL:
		return x.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(a, b, c, d), x)
	case simd.HHHH:
		return y.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(a, b, c, d), y)
	case simd.LLHH:
		return x.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(a, b, c, d), y)
	case simd.HHLL:
		return y.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(a, b, c, d), x)

	case simd.HLLL:
		z := y.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(a, a, b, b), x)
		return z.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(0, 2, c, d), x)
	case simd.LHLL:
		z := x.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(a, a, b, b), y)
		return z.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(0, 2, c, d), x)

	case simd.HLHH:
		z := y.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(a, a, b, b), x)
		return z.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(0, 2, c, d), y)
	case simd.LHHH:
		z := x.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(a, a, b, b), y)
		return z.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(0, 2, c, d), y)

	case simd.LLLH:
		z := x.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(c, c, d, d), y)
		return x.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(a, b, 0, 2), z)
	case simd.LLHL:
		z := y.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(c, c, d, d), x)
		return x.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(a, b, 0, 2), z)
	case simd.HHLH:
		z := x.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(c, c, d, d), y)
		return y.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(a, b, 0, 2), z)
	case simd.HHHL:
		z := y.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(c, c, d, d), x)
		return y.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(a, b, 0, 2), z)

	case simd.LHLH:
		z := x.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(a, c, b, d), y)
		return z.ExportTestConcatSelectedConstantGrouped(0b11_01_10_00 /* =simd.ExportTestCscImm4(0, 2, 1, 3) */, z)
	case simd.HLHL:
		z := x.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(b, d, a, c), y)
		return z.ExportTestConcatSelectedConstantGrouped(0b01_11_00_10 /* =simd.ExportTestCscImm4(2, 0, 3, 1) */, z)
	case simd.HLLH:
		z := x.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(b, c, a, d), y)
		return z.ExportTestConcatSelectedConstantGrouped(0b11_01_00_10 /* =simd.ExportTestCscImm4(2, 0, 1, 3) */, z)
	case simd.LHHL:
		z := x.ExportTestConcatSelectedConstantGrouped(simd.ExportTestCscImm4(a, d, b, c), y)
		return z.ExportTestConcatSelectedConstantGrouped(0b01_11_10_00 /* =simd.ExportTestCscImm4(0, 2, 3, 1) */, z)
	}
	panic("missing case, switch should be exhaustive")
}
