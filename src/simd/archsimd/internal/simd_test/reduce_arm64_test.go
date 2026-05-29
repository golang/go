// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestReduceSum(t *testing.T) {
	testInt8x16Reduce(t, archsimd.Int8x16.ReduceSum, func(x []int8) int8 { return reduceSlice(x, add[int8]) })
	testInt16x8Reduce(t, archsimd.Int16x8.ReduceSum, func(x []int16) int16 { return reduceSlice(x, add[int16]) })
	testInt32x4Reduce(t, archsimd.Int32x4.ReduceSum, func(x []int32) int32 { return reduceSlice(x, add[int32]) })
	testUint8x16Reduce(t, archsimd.Uint8x16.ReduceSum, func(x []uint8) uint8 { return reduceSlice(x, add[uint8]) })
	testUint16x8Reduce(t, archsimd.Uint16x8.ReduceSum, func(x []uint16) uint16 { return reduceSlice(x, add[uint16]) })
	testUint32x4Reduce(t, archsimd.Uint32x4.ReduceSum, func(x []uint32) uint32 { return reduceSlice(x, add[uint32]) })
	testInt64x2Reduce(t, archsimd.Int64x2.ReduceSum, func(x []int64) int64 { return reduceSlice(x, add[int64]) })
	testUint64x2Reduce(t, archsimd.Uint64x2.ReduceSum, func(x []uint64) uint64 { return reduceSlice(x, add[uint64]) })
	testFloat64x2Reduce(t, archsimd.Float64x2.ReduceSum, func(x []float64) float64 { return reduceSlice(x, add[float64]) })
}

func TestReduceMax(t *testing.T) {
	testInt8x16Reduce(t, archsimd.Int8x16.ReduceMax, func(x []int8) int8 { return reduceSlice(x, max_[int8]) })
	testInt16x8Reduce(t, archsimd.Int16x8.ReduceMax, func(x []int16) int16 { return reduceSlice(x, max_[int16]) })
	testInt32x4Reduce(t, archsimd.Int32x4.ReduceMax, func(x []int32) int32 { return reduceSlice(x, max_[int32]) })
	testUint8x16Reduce(t, archsimd.Uint8x16.ReduceMax, func(x []uint8) uint8 { return reduceSlice(x, max_[uint8]) })
	testUint16x8Reduce(t, archsimd.Uint16x8.ReduceMax, func(x []uint16) uint16 { return reduceSlice(x, max_[uint16]) })
	testUint32x4Reduce(t, archsimd.Uint32x4.ReduceMax, func(x []uint32) uint32 { return reduceSlice(x, max_[uint32]) })
	testFloat32x4Reduce(t, archsimd.Float32x4.ReduceMax, func(x []float32) float32 { return reduceSlice(x, max_[float32]) })
	testFloat64x2Reduce(t, archsimd.Float64x2.ReduceMax, func(x []float64) float64 { return reduceSlice(x, max_[float64]) })
}

func TestReduceMin(t *testing.T) {
	testInt8x16Reduce(t, archsimd.Int8x16.ReduceMin, func(x []int8) int8 { return reduceSlice(x, min_[int8]) })
	testInt16x8Reduce(t, archsimd.Int16x8.ReduceMin, func(x []int16) int16 { return reduceSlice(x, min_[int16]) })
	testInt32x4Reduce(t, archsimd.Int32x4.ReduceMin, func(x []int32) int32 { return reduceSlice(x, min_[int32]) })
	testUint8x16Reduce(t, archsimd.Uint8x16.ReduceMin, func(x []uint8) uint8 { return reduceSlice(x, min_[uint8]) })
	testUint16x8Reduce(t, archsimd.Uint16x8.ReduceMin, func(x []uint16) uint16 { return reduceSlice(x, min_[uint16]) })
	testUint32x4Reduce(t, archsimd.Uint32x4.ReduceMin, func(x []uint32) uint32 { return reduceSlice(x, min_[uint32]) })
	testFloat32x4Reduce(t, archsimd.Float32x4.ReduceMin, func(x []float32) float32 { return reduceSlice(x, min_[float32]) })
	testFloat64x2Reduce(t, archsimd.Float64x2.ReduceMin, func(x []float64) float64 { return reduceSlice(x, min_[float64]) })
}
