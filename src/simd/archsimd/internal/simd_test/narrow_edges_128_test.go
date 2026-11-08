// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || arm64)

package simd_test

import (
	"math"
	"simd/archsimd"
	"testing"
)

// Values around the limits of the narrower integer types, which the general
// test data doesn't cover well.
var (
	narrowEdgesI16 = []int16{math.MinInt16, math.MinInt8 - 1, math.MinInt8, -1, 0, 1, math.MaxInt8, math.MaxInt8 + 1, math.MaxUint8, math.MaxUint8 + 1, math.MaxInt16}
	narrowEdgesU16 = []uint16{0, 1, math.MaxInt8, math.MaxInt8 + 1, math.MaxUint8, math.MaxUint8 + 1, math.MaxInt16, math.MaxInt16 + 1, math.MaxUint16}
	narrowEdgesI32 = []int32{math.MinInt32, math.MinInt16 - 1, math.MinInt16, -1, 0, 1, math.MaxInt16, math.MaxInt16 + 1, math.MaxUint16, math.MaxUint16 + 1, math.MaxInt32}
	narrowEdgesU32 = []uint32{0, 1, math.MaxInt16, math.MaxInt16 + 1, math.MaxUint16, math.MaxUint16 + 1, math.MaxInt32, math.MaxInt32 + 1, math.MaxUint32}
	narrowEdgesI64 = []int64{math.MinInt64, math.MinInt32 - 1, math.MinInt32, -1, 0, 1, math.MaxInt32, math.MaxInt32 + 1, math.MaxUint32, math.MaxUint32 + 1, math.MaxInt64}
	narrowEdgesU64 = []uint64{0, 1, math.MaxInt32, math.MaxInt32 + 1, math.MaxUint32, math.MaxUint32 + 1, 1 << 33, math.MaxInt64, math.MaxInt64 + 1, math.MaxUint64}
)

// TestNarrowEdges128 tests the 128-bit 2:1 narrowing conversions on values
// around the limits of the result type.
func TestNarrowEdges128(t *testing.T) {
	testNarrowEdges(t, narrowEdgesI16, archsimd.LoadInt16x8, archsimd.Int16x8.TruncToInt8, toInt8[int16])
	testNarrowEdges(t, narrowEdgesU16, archsimd.LoadUint16x8, archsimd.Uint16x8.TruncToUint8, toUint8[uint16])
	testNarrowEdges(t, narrowEdgesI32, archsimd.LoadInt32x4, archsimd.Int32x4.TruncToInt16, toInt16[int32])
	testNarrowEdges(t, narrowEdgesU32, archsimd.LoadUint32x4, archsimd.Uint32x4.TruncToUint16, toUint16[uint32])
	testNarrowEdges(t, narrowEdgesI64, archsimd.LoadInt64x2, archsimd.Int64x2.TruncToInt32, toInt32[int64])
	testNarrowEdges(t, narrowEdgesU64, archsimd.LoadUint64x2, archsimd.Uint64x2.TruncToUint32, toUint32[uint64])
	testNarrowEdges(t, narrowEdgesI16, archsimd.LoadInt16x8, archsimd.Int16x8.SaturateToInt8, satToInt8[int16])
	testNarrowEdges(t, narrowEdgesI16, archsimd.LoadInt16x8, archsimd.Int16x8.SaturateToUint8, satToUint8[int16])
	testNarrowEdges(t, narrowEdgesU16, archsimd.LoadUint16x8, archsimd.Uint16x8.SaturateToUint8, satToUint8[uint16])
	testNarrowEdges(t, narrowEdgesI32, archsimd.LoadInt32x4, archsimd.Int32x4.SaturateToInt16, satToInt16[int32])
	testNarrowEdges(t, narrowEdgesI32, archsimd.LoadInt32x4, archsimd.Int32x4.SaturateToUint16, satToUint16[int32])
	testNarrowEdges(t, narrowEdgesU32, archsimd.LoadUint32x4, archsimd.Uint32x4.SaturateToUint16, satToUint16[uint32])
	testNarrowEdges(t, narrowEdgesI64, archsimd.LoadInt64x2, archsimd.Int64x2.SaturateToInt32, satToInt32[int64])
	testNarrowEdges(t, narrowEdgesI64, archsimd.LoadInt64x2, archsimd.Int64x2.SaturateToUint32, satToUint32[int64])
	testNarrowEdges(t, narrowEdgesU64, archsimd.LoadUint64x2, archsimd.Uint64x2.SaturateToUint32, satToUint32[uint64])
}
