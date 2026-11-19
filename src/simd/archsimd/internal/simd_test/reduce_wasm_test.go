// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable && wasm

package simd_test

import (
	"math"
	"simd/archsimd"
	"testing"
)

func TestReduceMaxWasm(t *testing.T) {
	testV2S(t, int8s, archsimd.LoadInt8x16, archsimd.Int8x16.ReduceMax, func(x []int8) int8 { return reduceSlice(x, max_[int8]) })
	testV2S(t, int16s, archsimd.LoadInt16x8, archsimd.Int16x8.ReduceMax, func(x []int16) int16 { return reduceSlice(x, max_[int16]) })
	testV2S(t, int32s, archsimd.LoadInt32x4, archsimd.Int32x4.ReduceMax, func(x []int32) int32 { return reduceSlice(x, max_[int32]) })
	testV2S(t, int64s, archsimd.LoadInt64x2, archsimd.Int64x2.ReduceMax, func(x []int64) int64 { return reduceSlice(x, max_[int64]) })
	testV2S(t, uint8s, archsimd.LoadUint8x16, archsimd.Uint8x16.ReduceMax, func(x []uint8) uint8 { return reduceSlice(x, max_[uint8]) })
	testV2S(t, uint16s, archsimd.LoadUint16x8, archsimd.Uint16x8.ReduceMax, func(x []uint16) uint16 { return reduceSlice(x, max_[uint16]) })
	testV2S(t, uint32s, archsimd.LoadUint32x4, archsimd.Uint32x4.ReduceMax, func(x []uint32) uint32 { return reduceSlice(x, max_[uint32]) })
	testV2S(t, uint64s, archsimd.LoadUint64x2, archsimd.Uint64x2.ReduceMax, func(x []uint64) uint64 { return reduceSlice(x, max_[uint64]) })
	testV2S(t, float32s, archsimd.LoadFloat32x4, archsimd.Float32x4.ReduceMax, func(x []float32) float32 { return reduceSlice(x, max_[float32]) })
	testV2S(t, float64s, archsimd.LoadFloat64x2, archsimd.Float64x2.ReduceMax, func(x []float64) float64 { return reduceSlice(x, max_[float64]) })
}

func TestReduceMinWasm(t *testing.T) {
	testV2S(t, int8s, archsimd.LoadInt8x16, archsimd.Int8x16.ReduceMin, func(x []int8) int8 { return reduceSlice(x, min_[int8]) })
	testV2S(t, int16s, archsimd.LoadInt16x8, archsimd.Int16x8.ReduceMin, func(x []int16) int16 { return reduceSlice(x, min_[int16]) })
	testV2S(t, int32s, archsimd.LoadInt32x4, archsimd.Int32x4.ReduceMin, func(x []int32) int32 { return reduceSlice(x, min_[int32]) })
	testV2S(t, int64s, archsimd.LoadInt64x2, archsimd.Int64x2.ReduceMin, func(x []int64) int64 { return reduceSlice(x, min_[int64]) })
	testV2S(t, uint8s, archsimd.LoadUint8x16, archsimd.Uint8x16.ReduceMin, func(x []uint8) uint8 { return reduceSlice(x, min_[uint8]) })
	testV2S(t, uint16s, archsimd.LoadUint16x8, archsimd.Uint16x8.ReduceMin, func(x []uint16) uint16 { return reduceSlice(x, min_[uint16]) })
	testV2S(t, uint32s, archsimd.LoadUint32x4, archsimd.Uint32x4.ReduceMin, func(x []uint32) uint32 { return reduceSlice(x, min_[uint32]) })
	testV2S(t, uint64s, archsimd.LoadUint64x2, archsimd.Uint64x2.ReduceMin, func(x []uint64) uint64 { return reduceSlice(x, min_[uint64]) })
	testV2S(t, float32s, archsimd.LoadFloat32x4, archsimd.Float32x4.ReduceMin, func(x []float32) float32 { return reduceSlice(x, min_[float32]) })
	testV2S(t, float64s, archsimd.LoadFloat64x2, archsimd.Float64x2.ReduceMin, func(x []float64) float64 { return reduceSlice(x, min_[float64]) })
}

func TestReduceMinMaxFloatEdgeWasm(t *testing.T) {
	negZero32 := float32(math.Copysign(0, -1))
	nan32 := float32(math.NaN())
	for lane := range 4 {
		a := [4]float32{negZero32, negZero32, negZero32, negZero32}
		a[lane] = 0
		if got := archsimd.LoadFloat32x4Array(&a).ReduceMax(); got != 0 || math.Signbit(float64(got)) {
			t.Errorf("Float32x4.ReduceMax(%v) = %v, want +0", a, got)
		}
		b := [4]float32{0, 0, 0, 0}
		b[lane] = negZero32
		if got := archsimd.LoadFloat32x4Array(&b).ReduceMin(); got != 0 || !math.Signbit(float64(got)) {
			t.Errorf("Float32x4.ReduceMin(%v) = %v, want -0", b, got)
		}
		c := [4]float32{1, 2, 3, 4}
		c[lane] = nan32
		if got := archsimd.LoadFloat32x4Array(&c).ReduceMax(); got == got {
			t.Errorf("Float32x4.ReduceMax(%v) = %v, want NaN", c, got)
		}
		if got := archsimd.LoadFloat32x4Array(&c).ReduceMin(); got == got {
			t.Errorf("Float32x4.ReduceMin(%v) = %v, want NaN", c, got)
		}
	}

	negZero64 := math.Copysign(0, -1)
	nan64 := math.NaN()
	for lane := range 2 {
		a := [2]float64{negZero64, negZero64}
		a[lane] = 0
		if got := archsimd.LoadFloat64x2Array(&a).ReduceMax(); got != 0 || math.Signbit(got) {
			t.Errorf("Float64x2.ReduceMax(%v) = %v, want +0", a, got)
		}
		b := [2]float64{0, 0}
		b[lane] = negZero64
		if got := archsimd.LoadFloat64x2Array(&b).ReduceMin(); got != 0 || !math.Signbit(got) {
			t.Errorf("Float64x2.ReduceMin(%v) = %v, want -0", b, got)
		}
		c := [2]float64{1, 2}
		c[lane] = nan64
		if got := archsimd.LoadFloat64x2Array(&c).ReduceMax(); got == got {
			t.Errorf("Float64x2.ReduceMax(%v) = %v, want NaN", c, got)
		}
		if got := archsimd.LoadFloat64x2Array(&c).ReduceMin(); got == got {
			t.Errorf("Float64x2.ReduceMin(%v) = %v, want NaN", c, got)
		}
	}
}
