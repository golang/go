// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && wasm

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestOnesCountWasm(t *testing.T) {
	testInt16x8Unary(t, archsimd.Int16x8.OnesCount, map1[int16](onesCount))
	testUint16x8Unary(t, archsimd.Uint16x8.OnesCount, map1[uint16](onesCount))
	testInt32x4Unary(t, archsimd.Int32x4.OnesCount, map1[int32](onesCount))
	testUint32x4Unary(t, archsimd.Uint32x4.OnesCount, map1[uint32](onesCount))
	testInt64x2Unary(t, archsimd.Int64x2.OnesCount, map1[int64](onesCount))
	testUint64x2Unary(t, archsimd.Uint64x2.OnesCount, map1[uint64](onesCount))
}
