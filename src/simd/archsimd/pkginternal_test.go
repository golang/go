// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package archsimd_test

import (
	"simd/archsimd"
	"simd/archsimd/internal/test_helpers"
	"testing"
)

func TestConcatSelectedConstant64(t *testing.T) {
	a := make([]int64, 2)
	x := archsimd.LoadInt64x2Slice([]int64{4, 5})
	y := archsimd.LoadInt64x2Slice([]int64{6, 7})
	z := x.ExportTestConcatSelectedConstant(0b10, y)
	z.StoreSlice(a)
	test_helpers.CheckSlices[int64](t, a, []int64{4, 7})
}

func TestConcatSelectedConstantGrouped64(t *testing.T) {
	a := make([]float64, 4)
	x := archsimd.LoadFloat64x4Slice([]float64{4, 5, 8, 9})
	y := archsimd.LoadFloat64x4Slice([]float64{6, 7, 10, 11})
	z := x.ExportTestConcatSelectedConstantGrouped(0b_11_10, y)
	z.StoreSlice(a)
	test_helpers.CheckSlices[float64](t, a, []float64{4, 7, 9, 11})
}

func TestConcatSelectedConstant32(t *testing.T) {
	a := make([]float32, 4)
	x := archsimd.LoadFloat32x4Slice([]float32{4, 5, 8, 9})
	y := archsimd.LoadFloat32x4Slice([]float32{6, 7, 10, 11})
	z := x.ExportTestConcatSelectedConstant(0b_11_01_10_00, y)
	z.StoreSlice(a)
	test_helpers.CheckSlices[float32](t, a, []float32{4, 8, 7, 11})
}

func TestConcatSelectedConstantGrouped32(t *testing.T) {
	a := make([]uint32, 8)
	x := archsimd.LoadUint32x8Slice([]uint32{0, 1, 2, 3, 8, 9, 10, 11})
	y := archsimd.LoadUint32x8Slice([]uint32{4, 5, 6, 7, 12, 13, 14, 15})
	z := x.ExportTestConcatSelectedConstantGrouped(0b_11_01_00_10, y)
	z.StoreSlice(a)
	test_helpers.CheckSlices[uint32](t, a, []uint32{2, 0, 5, 7, 10, 8, 13, 15})
}

func TestTern(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("This test needs AVX512")
	}
	x := archsimd.LoadInt32x8Slice([]int32{0, 0, 0, 0, 1, 1, 1, 1})
	y := archsimd.LoadInt32x8Slice([]int32{0, 0, 1, 1, 0, 0, 1, 1})
	z := archsimd.LoadInt32x8Slice([]int32{0, 1, 0, 1, 0, 1, 0, 1})

	foo := func(w archsimd.Int32x8, k uint8) {
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
					x := archsimd.LoadInt32x4Slice([]int32{0, 1, 2, 3})
					y := archsimd.LoadInt32x4Slice([]int32{4, 5, 6, 7})
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
					x := archsimd.LoadInt32x8Slice([]int32{0, 1, 2, 3, 10, 11, 12, 13})
					y := archsimd.LoadInt32x8Slice([]int32{4, 5, 6, 7, 14, 15, 16, 17})
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
func select2x4x32(x archsimd.Int32x4, a, b, c, d uint8, y archsimd.Int32x4) archsimd.Int32x4 {
	pattern := a>>2 + (b&4)>>1 + (c & 4) + (d&4)<<1

	a, b, c, d = a&3, b&3, c&3, d&3

	switch pattern {
	case archsimd.LLLL:
		return x.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(a, b, c, d), x)
	case archsimd.HHHH:
		return y.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(a, b, c, d), y)
	case archsimd.LLHH:
		return x.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(a, b, c, d), y)
	case archsimd.HHLL:
		return y.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(a, b, c, d), x)

	case archsimd.HLLL:
		z := y.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(a, a, b, b), x)
		return z.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(0, 2, c, d), x)
	case archsimd.LHLL:
		z := x.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(a, a, b, b), y)
		return z.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(0, 2, c, d), x)

	case archsimd.HLHH:
		z := y.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(a, a, b, b), x)
		return z.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(0, 2, c, d), y)
	case archsimd.LHHH:
		z := x.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(a, a, b, b), y)
		return z.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(0, 2, c, d), y)

	case archsimd.LLLH:
		z := x.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(c, c, d, d), y)
		return x.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(a, b, 0, 2), z)
	case archsimd.LLHL:
		z := y.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(c, c, d, d), x)
		return x.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(a, b, 0, 2), z)
	case archsimd.HHLH:
		z := x.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(c, c, d, d), y)
		return y.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(a, b, 0, 2), z)
	case archsimd.HHHL:
		z := y.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(c, c, d, d), x)
		return y.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(a, b, 0, 2), z)

	case archsimd.LHLH:
		z := x.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(a, c, b, d), y)
		return z.ExportTestConcatSelectedConstant(0b11_01_10_00 /* =archsimd.ExportTestCscImm4(0, 2, 1, 3) */, z)
	case archsimd.HLHL:
		z := x.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(b, d, a, c), y)
		return z.ExportTestConcatSelectedConstant(0b01_11_00_10 /* =archsimd.ExportTestCscImm4(2, 0, 3, 1) */, z)
	case archsimd.HLLH:
		z := x.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(b, c, a, d), y)
		return z.ExportTestConcatSelectedConstant(0b11_01_00_10 /* =archsimd.ExportTestCscImm4(2, 0, 1, 3) */, z)
	case archsimd.LHHL:
		z := x.ExportTestConcatSelectedConstant(archsimd.ExportTestCscImm4(a, d, b, c), y)
		return z.ExportTestConcatSelectedConstant(0b01_11_10_00 /* =archsimd.ExportTestCscImm4(0, 2, 3, 1) */, z)
	}
	panic("missing case, switch should be exhaustive")
}

// select2x8x32Grouped returns a pair of selection of 4 elements in x and y,
// numbered 0-7, where 0-3 are the four elements of x's two groups (lower and
// upper 128 bits) and 4-7 are the four elements of y's two groups.

func select2x8x32Grouped(x archsimd.Int32x8, a, b, c, d uint8, y archsimd.Int32x8) archsimd.Int32x8 {
	// selections as being expressible in the ExportTestConcatSelectedConstant pattern,
	// or not. Classification is by H and L, where H is a selection from 4-7
	// and L is a selection from 0-3.
	// archsimd.LLHH -> CSC(x,y, a, b, c&3, d&3)
	// archsimd.HHLL -> CSC(y,x, a&3, b&3, c, d)
	// archsimd.LLLL -> CSC(x,x, a, b, c, d)
	// archsimd.HHHH -> CSC(y,y, a&3, b&3, c&3, d&3)

	// archsimd.LLLH -> z = CSC(x, y, c, c, d&3, d&3); CSC(x, z, a, b, 0, 2)
	// archsimd.LLHL -> z = CSC(x, y, c&3, c&3, d, d); CSC(x, z, a, b, 0, 2)
	// archsimd.HHLH -> z = CSC(x, y, c, c, d&3, d&3); CSC(y, z, a&3, b&3, 0, 2)
	// archsimd.HHHL -> z = CSC(x, y, c&3, c&3, d, d); CSC(y, z, a&3, b&3, 0, 2)

	// archsimd.LHLL -> z = CSC(x, y, a, a, b&3, b&3); CSC(z, x, 0, 2, c, d)
	// etc

	// archsimd.LHLH -> z = CSC(x, y, a, c, b&3, d&3); CSC(z, z, 0, 2, 1, 3)
	// archsimd.HLHL -> z = CSC(x, y, b, d, a&3, c&3); CSC(z, z, 2, 0, 3, 1)

	pattern := a>>2 + (b&4)>>1 + (c & 4) + (d&4)<<1

	a, b, c, d = a&3, b&3, c&3, d&3

	switch pattern {
	case archsimd.LLLL:
		return x.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(a, b, c, d), x)
	case archsimd.HHHH:
		return y.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(a, b, c, d), y)
	case archsimd.LLHH:
		return x.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(a, b, c, d), y)
	case archsimd.HHLL:
		return y.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(a, b, c, d), x)

	case archsimd.HLLL:
		z := y.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(a, a, b, b), x)
		return z.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(0, 2, c, d), x)
	case archsimd.LHLL:
		z := x.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(a, a, b, b), y)
		return z.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(0, 2, c, d), x)

	case archsimd.HLHH:
		z := y.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(a, a, b, b), x)
		return z.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(0, 2, c, d), y)
	case archsimd.LHHH:
		z := x.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(a, a, b, b), y)
		return z.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(0, 2, c, d), y)

	case archsimd.LLLH:
		z := x.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(c, c, d, d), y)
		return x.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(a, b, 0, 2), z)
	case archsimd.LLHL:
		z := y.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(c, c, d, d), x)
		return x.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(a, b, 0, 2), z)
	case archsimd.HHLH:
		z := x.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(c, c, d, d), y)
		return y.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(a, b, 0, 2), z)
	case archsimd.HHHL:
		z := y.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(c, c, d, d), x)
		return y.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(a, b, 0, 2), z)

	case archsimd.LHLH:
		z := x.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(a, c, b, d), y)
		return z.ExportTestConcatSelectedConstantGrouped(0b11_01_10_00 /* =archsimd.ExportTestCscImm4(0, 2, 1, 3) */, z)
	case archsimd.HLHL:
		z := x.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(b, d, a, c), y)
		return z.ExportTestConcatSelectedConstantGrouped(0b01_11_00_10 /* =archsimd.ExportTestCscImm4(2, 0, 3, 1) */, z)
	case archsimd.HLLH:
		z := x.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(b, c, a, d), y)
		return z.ExportTestConcatSelectedConstantGrouped(0b11_01_00_10 /* =archsimd.ExportTestCscImm4(2, 0, 1, 3) */, z)
	case archsimd.LHHL:
		z := x.ExportTestConcatSelectedConstantGrouped(archsimd.ExportTestCscImm4(a, d, b, c), y)
		return z.ExportTestConcatSelectedConstantGrouped(0b01_11_10_00 /* =archsimd.ExportTestCscImm4(0, 2, 3, 1) */, z)
	}
	panic("missing case, switch should be exhaustive")
}
