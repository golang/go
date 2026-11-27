// compile -d=ssa/check/on

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable && wasm

// The Broadcast intrinsics on wasm built their SSA value with a
// memory argument that the op does not take.

package p

import "simd/archsimd"

func f(a int8, b int16, c int32, d int64, e float32, g float64) (archsimd.Int8x16, archsimd.Int16x8, archsimd.Int32x4, archsimd.Int64x2, archsimd.Float32x4, archsimd.Float64x2) {
	return archsimd.BroadcastInt8x16(a), archsimd.BroadcastInt16x8(b), archsimd.BroadcastInt32x4(c), archsimd.BroadcastInt64x2(d), archsimd.BroadcastFloat32x4(e), archsimd.BroadcastFloat64x2(g)
}

func h(x archsimd.Int64x2) archsimd.Int32x4 {
	return x.SaturateToInt32()
}
