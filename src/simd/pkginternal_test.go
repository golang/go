// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd

import (
	"simd/internal/test_helpers"
	"testing"
)

func TestConcatSelectedConstant64(t *testing.T) {
	a := make([]int64, 2)
	x := LoadInt64x2Slice([]int64{4, 5})
	y := LoadInt64x2Slice([]int64{6, 7})
	z := x.concatSelectedConstant(0b10, y)
	z.StoreSlice(a)
	test_helpers.CheckSlices[int64](t, a, []int64{4, 7})
}

func TestConcatSelectedConstantGrouped64(t *testing.T) {
	a := make([]float64, 4)
	x := LoadFloat64x4Slice([]float64{4, 5, 8, 9})
	y := LoadFloat64x4Slice([]float64{6, 7, 10, 11})
	z := x.concatSelectedConstantGrouped(0b_11_10, y)
	z.StoreSlice(a)
	test_helpers.CheckSlices[float64](t, a, []float64{4, 7, 9, 11})
}

func TestConcatSelectedConstant32(t *testing.T) {
	a := make([]float32, 4)
	x := LoadFloat32x4Slice([]float32{4, 5, 8, 9})
	y := LoadFloat32x4Slice([]float32{6, 7, 10, 11})
	z := x.concatSelectedConstant(0b_11_01_10_00, y)
	z.StoreSlice(a)
	test_helpers.CheckSlices[float32](t, a, []float32{4, 8, 7, 11})
}

func TestConcatSelectedConstantGrouped32(t *testing.T) {
	a := make([]uint32, 8)
	x := LoadUint32x8Slice([]uint32{0, 1, 2, 3, 8, 9, 10, 11})
	y := LoadUint32x8Slice([]uint32{4, 5, 6, 7, 12, 13, 14, 15})
	z := x.concatSelectedConstantGrouped(0b_11_01_00_10, y)
	z.StoreSlice(a)
	test_helpers.CheckSlices[uint32](t, a, []uint32{2, 0, 5, 7, 10, 8, 13, 15})
}

func TestTern(t *testing.T) {
	if !HasAVX512() {
		t.Skip("This test needs AVX512")
	}
	x := LoadInt32x8Slice([]int32{0, 0, 0, 0, 1, 1, 1, 1})
	y := LoadInt32x8Slice([]int32{0, 0, 1, 1, 0, 0, 1, 1})
	z := LoadInt32x8Slice([]int32{0, 1, 0, 1, 0, 1, 0, 1})

	foo := func(w Int32x8, k uint8) {
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

	foo(x.tern(0b1111_0000, y, z), 0b1111_0000)
	foo(x.tern(0b1100_1100, y, z), 0b1100_1100)
	foo(x.tern(0b1010_1010, y, z), 0b1010_1010)
}

func TestSelect2x4x32(t *testing.T) {
	for a := range uint8(8) {
		for b := range uint8(8) {
			for c := range uint8(8) {
				for d := range uint8(8) {
					x := LoadInt32x4Slice([]int32{0, 1, 2, 3})
					y := LoadInt32x4Slice([]int32{4, 5, 6, 7})
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
					x := LoadInt32x8Slice([]int32{0, 1, 2, 3, 10, 11, 12, 13})
					y := LoadInt32x8Slice([]int32{4, 5, 6, 7, 14, 15, 16, 17})
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
func select2x4x32(x Int32x4, a, b, c, d uint8, y Int32x4) Int32x4 {
	pattern := a>>2 + (b&4)>>1 + (c & 4) + (d&4)<<1

	a, b, c, d = a&3, b&3, c&3, d&3

	switch pattern {
	case _LLLL:
		return x.concatSelectedConstant(cscimm4(a, b, c, d), x)
	case _HHHH:
		return y.concatSelectedConstant(cscimm4(a, b, c, d), y)
	case _LLHH:
		return x.concatSelectedConstant(cscimm4(a, b, c, d), y)
	case _HHLL:
		return y.concatSelectedConstant(cscimm4(a, b, c, d), x)

	case _HLLL:
		z := y.concatSelectedConstant(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstant(cscimm4(0, 2, c, d), x)
	case _LHLL:
		z := x.concatSelectedConstant(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstant(cscimm4(0, 2, c, d), x)

	case _HLHH:
		z := y.concatSelectedConstant(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstant(cscimm4(0, 2, c, d), y)
	case _LHHH:
		z := x.concatSelectedConstant(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstant(cscimm4(0, 2, c, d), y)

	case _LLLH:
		z := x.concatSelectedConstant(cscimm4(c, c, d, d), y)
		return x.concatSelectedConstant(cscimm4(a, b, 0, 2), z)
	case _LLHL:
		z := y.concatSelectedConstant(cscimm4(c, c, d, d), x)
		return x.concatSelectedConstant(cscimm4(a, b, 0, 2), z)
	case _HHLH:
		z := x.concatSelectedConstant(cscimm4(c, c, d, d), y)
		return y.concatSelectedConstant(cscimm4(a, b, 0, 2), z)
	case _HHHL:
		z := y.concatSelectedConstant(cscimm4(c, c, d, d), x)
		return y.concatSelectedConstant(cscimm4(a, b, 0, 2), z)

	case _LHLH:
		z := x.concatSelectedConstant(cscimm4(a, c, b, d), y)
		return z.concatSelectedConstant(0b11_01_10_00 /* =cscimm4(0, 2, 1, 3) */, z)
	case _HLHL:
		z := x.concatSelectedConstant(cscimm4(b, d, a, c), y)
		return z.concatSelectedConstant(0b01_11_00_10 /* =cscimm4(2, 0, 3, 1) */, z)
	case _HLLH:
		z := x.concatSelectedConstant(cscimm4(b, c, a, d), y)
		return z.concatSelectedConstant(0b11_01_00_10 /* =cscimm4(2, 0, 1, 3) */, z)
	case _LHHL:
		z := x.concatSelectedConstant(cscimm4(a, d, b, c), y)
		return z.concatSelectedConstant(0b01_11_10_00 /* =cscimm4(0, 2, 3, 1) */, z)
	}
	panic("missing case, switch should be exhaustive")
}

// select2x8x32Grouped returns a pair of selection of 4 elements in x and y,
// numbered 0-7, where 0-3 are the four elements of x's two groups (lower and
// upper 128 bits) and 4-7 are the four elements of y's two groups.

func select2x8x32Grouped(x Int32x8, a, b, c, d uint8, y Int32x8) Int32x8 {
	// selections as being expressible in the concatSelectedConstant pattern,
	// or not. Classification is by H and L, where H is a selection from 4-7
	// and L is a selection from 0-3.
	// _LLHH -> CSC(x,y, a, b, c&3, d&3)
	// _HHLL -> CSC(y,x, a&3, b&3, c, d)
	// _LLLL -> CSC(x,x, a, b, c, d)
	// _HHHH -> CSC(y,y, a&3, b&3, c&3, d&3)

	// _LLLH -> z = CSC(x, y, c, c, d&3, d&3); CSC(x, z, a, b, 0, 2)
	// _LLHL -> z = CSC(x, y, c&3, c&3, d, d); CSC(x, z, a, b, 0, 2)
	// _HHLH -> z = CSC(x, y, c, c, d&3, d&3); CSC(y, z, a&3, b&3, 0, 2)
	// _HHHL -> z = CSC(x, y, c&3, c&3, d, d); CSC(y, z, a&3, b&3, 0, 2)

	// _LHLL -> z = CSC(x, y, a, a, b&3, b&3); CSC(z, x, 0, 2, c, d)
	// etc

	// _LHLH -> z = CSC(x, y, a, c, b&3, d&3); CSC(z, z, 0, 2, 1, 3)
	// _HLHL -> z = CSC(x, y, b, d, a&3, c&3); CSC(z, z, 2, 0, 3, 1)

	pattern := a>>2 + (b&4)>>1 + (c & 4) + (d&4)<<1

	a, b, c, d = a&3, b&3, c&3, d&3

	switch pattern {
	case _LLLL:
		return x.concatSelectedConstantGrouped(cscimm4(a, b, c, d), x)
	case _HHHH:
		return y.concatSelectedConstantGrouped(cscimm4(a, b, c, d), y)
	case _LLHH:
		return x.concatSelectedConstantGrouped(cscimm4(a, b, c, d), y)
	case _HHLL:
		return y.concatSelectedConstantGrouped(cscimm4(a, b, c, d), x)

	case _HLLL:
		z := y.concatSelectedConstantGrouped(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), x)
	case _LHLL:
		z := x.concatSelectedConstantGrouped(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), x)

	case _HLHH:
		z := y.concatSelectedConstantGrouped(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), y)
	case _LHHH:
		z := x.concatSelectedConstantGrouped(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), y)

	case _LLLH:
		z := x.concatSelectedConstantGrouped(cscimm4(c, c, d, d), y)
		return x.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _LLHL:
		z := y.concatSelectedConstantGrouped(cscimm4(c, c, d, d), x)
		return x.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _HHLH:
		z := x.concatSelectedConstantGrouped(cscimm4(c, c, d, d), y)
		return y.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _HHHL:
		z := y.concatSelectedConstantGrouped(cscimm4(c, c, d, d), x)
		return y.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)

	case _LHLH:
		z := x.concatSelectedConstantGrouped(cscimm4(a, c, b, d), y)
		return z.concatSelectedConstantGrouped(0b11_01_10_00 /* =cscimm4(0, 2, 1, 3) */, z)
	case _HLHL:
		z := x.concatSelectedConstantGrouped(cscimm4(b, d, a, c), y)
		return z.concatSelectedConstantGrouped(0b01_11_00_10 /* =cscimm4(2, 0, 3, 1) */, z)
	case _HLLH:
		z := x.concatSelectedConstantGrouped(cscimm4(b, c, a, d), y)
		return z.concatSelectedConstantGrouped(0b11_01_00_10 /* =cscimm4(2, 0, 1, 3) */, z)
	case _LHHL:
		z := x.concatSelectedConstantGrouped(cscimm4(a, d, b, c), y)
		return z.concatSelectedConstantGrouped(0b01_11_10_00 /* =cscimm4(0, 2, 3, 1) */, z)
	}
	panic("missing case, switch should be exhaustive")
}
