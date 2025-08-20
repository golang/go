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
