// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"math"
	"simd"
	"simd/internal/test_helpers"
	"testing"
	"unsafe"
)

// withEdges returns xs followed by values around the limits of the narrower
// integer types, which xs may lack. It repeats the edge values enough that
// every window of the result that TestV2Ve tries puts every edge value in
// every lane, for vectors of up to 64 elements.
func withEdges[X integer](xs []X, edges ...int64) []X {
	const maxLanes = 64
	r := append([]X(nil), xs...)
	for i := range len(edges) + maxLanes {
		r = append(r, X(edges[i%len(edges)]))
	}
	return r
}

func sat[Z integer, X integer](x X) Z {
	// Compare in int64 or uint64, depending on the signedness of x.
	lo, hi := minOf[Z](), maxOf[Z]()
	if x < 0 {
		if int64(x) < lo {
			return Z(lo)
		}
	} else if uint64(x) > hi {
		return Z(hi)
	}
	return Z(x)
}

func minOf[Z integer]() int64 {
	var z Z
	if z-1 > 0 {
		return 0
	}
	return -1 << (8*sizeOf[Z]() - 1)
}

func maxOf[Z integer]() uint64 {
	var z Z
	if z-1 > 0 {
		return 1<<(8*sizeOf[Z]()) - 1
	}
	return 1<<(8*sizeOf[Z]()-1) - 1
}

func sizeOf[Z integer]() uint {
	var z Z
	return uint(unsafe.Sizeof(z))
}

func conv[Z, X number](x X) Z { return Z(x) }

func TestWidthChangingConversions(t *testing.T) {
	edges := []int64{math.MinInt32 - 1, math.MinInt32, math.MinInt16 - 1, math.MinInt16, math.MinInt8 - 1, math.MinInt8,
		-1, 0, 1, math.MaxInt8, math.MaxInt8 + 1, math.MaxUint8, math.MaxUint8 + 1,
		math.MaxInt16, math.MaxInt16 + 1, math.MaxUint16, math.MaxUint16 + 1,
		math.MaxInt32, math.MaxInt32 + 1, math.MaxUint32, math.MaxUint32 + 1, math.MaxInt64, math.MinInt64}
	i8 := withEdges(test_helpers.Int8s(), edges...)
	i16 := withEdges(test_helpers.Int16s(), edges...)
	i32 := withEdges(test_helpers.Int32s(), edges...)
	i64 := withEdges(test_helpers.Int64s(), edges...)
	u8 := withEdges(test_helpers.Uint8s(), edges...)
	u16 := withEdges(test_helpers.Uint16s(), edges...)
	u32 := withEdges(test_helpers.Uint32s(), edges...)
	u64 := withEdges(test_helpers.Uint64s(), edges...)

	test_helpers.TestV2Ve(t, i8, simd.LoadInt8s, simd.Int8s.ExtendToInt16, conv[int16, int8])
	test_helpers.TestV2Ve(t, i16, simd.LoadInt16s, simd.Int16s.ExtendToInt32, conv[int32, int16])
	test_helpers.TestV2Ve(t, i32, simd.LoadInt32s, simd.Int32s.ExtendToInt64, conv[int64, int32])
	test_helpers.TestV2Ve(t, u8, simd.LoadUint8s, simd.Uint8s.ExtendToUint16, conv[uint16, uint8])
	test_helpers.TestV2Ve(t, u16, simd.LoadUint16s, simd.Uint16s.ExtendToUint32, conv[uint32, uint16])
	test_helpers.TestV2Ve(t, u32, simd.LoadUint32s, simd.Uint32s.ExtendToUint64, conv[uint64, uint32])

	test_helpers.TestV2Ve(t, i16, simd.LoadInt16s, simd.Int16s.TruncToInt8, conv[int8, int16])
	test_helpers.TestV2Ve(t, i32, simd.LoadInt32s, simd.Int32s.TruncToInt16, conv[int16, int32])
	test_helpers.TestV2Ve(t, i64, simd.LoadInt64s, simd.Int64s.TruncToInt32, conv[int32, int64])
	test_helpers.TestV2Ve(t, u16, simd.LoadUint16s, simd.Uint16s.TruncToUint8, conv[uint8, uint16])
	test_helpers.TestV2Ve(t, u32, simd.LoadUint32s, simd.Uint32s.TruncToUint16, conv[uint16, uint32])
	test_helpers.TestV2Ve(t, u64, simd.LoadUint64s, simd.Uint64s.TruncToUint32, conv[uint32, uint64])

	test_helpers.TestV2Ve(t, i16, simd.LoadInt16s, simd.Int16s.SaturateToInt8, sat[int8, int16])
	test_helpers.TestV2Ve(t, i32, simd.LoadInt32s, simd.Int32s.SaturateToInt16, sat[int16, int32])
	test_helpers.TestV2Ve(t, i64, simd.LoadInt64s, simd.Int64s.SaturateToInt32, sat[int32, int64])
	test_helpers.TestV2Ve(t, i16, simd.LoadInt16s, simd.Int16s.SaturateToUint8, sat[uint8, int16])
	test_helpers.TestV2Ve(t, i32, simd.LoadInt32s, simd.Int32s.SaturateToUint16, sat[uint16, int32])
	test_helpers.TestV2Ve(t, i64, simd.LoadInt64s, simd.Int64s.SaturateToUint32, sat[uint32, int64])
	test_helpers.TestV2Ve(t, u16, simd.LoadUint16s, simd.Uint16s.SaturateToUint8, sat[uint8, uint16])
	test_helpers.TestV2Ve(t, u32, simd.LoadUint32s, simd.Uint32s.SaturateToUint16, sat[uint16, uint32])
	test_helpers.TestV2Ve(t, u64, simd.LoadUint64s, simd.Uint64s.SaturateToUint32, sat[uint32, uint64])

	test_helpers.TestV2Ve(t, test_helpers.Float32s(), simd.LoadFloat32s, simd.Float32s.ConvertToFloat64, conv[float64, float32])
	test_helpers.TestV2Ve(t, test_helpers.Float64s(), simd.LoadFloat64s, simd.Float64s.ConvertToFloat32, conv[float32, float64])
}
