// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

// SVE shuffle tests. The expected result of a shuffle depends on the runtime
// vector length, so unlike testSVEBinary, these drivers pass only the active
// lanes to want.

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func testSVEShuffle2[T number, V any](t *testing.T, pool []T, elemBytes, active int,
	load func([]T) V, f func(V, V) V, store func(V, []T), want func([]T, []T) []T) {
	t.Helper()
	count := sveMaxBytes / elemBytes
	forSlicePair(t, pool, count, func(x, y []T) bool {
		t.Helper()
		g := make([]T, count)
		store(f(load(x), load(y)), g)
		w := want(x[:active], y[:active])
		return checkSlicesLogInput(t, g[:active], w, 0.0, func() {
			t.Helper()
			t.Logf("x=%v", x[:active])
			t.Logf("y=%v", y[:active])
		})
	})
}

func TestInterleaveLoSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testSVEShuffle2(t, int8s, 1, archsimd.Int8s{}.Len(), archsimd.LoadInt8s, archsimd.Int8s.InterleaveLo, archsimd.Int8s.Store, interleaveSlice[int8](0, false))
	testSVEShuffle2(t, int16s, 2, archsimd.Int16s{}.Len(), archsimd.LoadInt16s, archsimd.Int16s.InterleaveLo, archsimd.Int16s.Store, interleaveSlice[int16](0, false))
	testSVEShuffle2(t, int32s, 4, archsimd.Int32s{}.Len(), archsimd.LoadInt32s, archsimd.Int32s.InterleaveLo, archsimd.Int32s.Store, interleaveSlice[int32](0, false))
	testSVEShuffle2(t, int64s, 8, archsimd.Int64s{}.Len(), archsimd.LoadInt64s, archsimd.Int64s.InterleaveLo, archsimd.Int64s.Store, interleaveSlice[int64](0, false))
	testSVEShuffle2(t, uint8s, 1, archsimd.Uint8s{}.Len(), archsimd.LoadUint8s, archsimd.Uint8s.InterleaveLo, archsimd.Uint8s.Store, interleaveSlice[uint8](0, false))
	testSVEShuffle2(t, uint16s, 2, archsimd.Uint16s{}.Len(), archsimd.LoadUint16s, archsimd.Uint16s.InterleaveLo, archsimd.Uint16s.Store, interleaveSlice[uint16](0, false))
	testSVEShuffle2(t, uint32s, 4, archsimd.Uint32s{}.Len(), archsimd.LoadUint32s, archsimd.Uint32s.InterleaveLo, archsimd.Uint32s.Store, interleaveSlice[uint32](0, false))
	testSVEShuffle2(t, uint64s, 8, archsimd.Uint64s{}.Len(), archsimd.LoadUint64s, archsimd.Uint64s.InterleaveLo, archsimd.Uint64s.Store, interleaveSlice[uint64](0, false))
}

func TestInterleaveHiSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testSVEShuffle2(t, int8s, 1, archsimd.Int8s{}.Len(), archsimd.LoadInt8s, archsimd.Int8s.InterleaveHi, archsimd.Int8s.Store, interleaveSlice[int8](0, true))
	testSVEShuffle2(t, int16s, 2, archsimd.Int16s{}.Len(), archsimd.LoadInt16s, archsimd.Int16s.InterleaveHi, archsimd.Int16s.Store, interleaveSlice[int16](0, true))
	testSVEShuffle2(t, int32s, 4, archsimd.Int32s{}.Len(), archsimd.LoadInt32s, archsimd.Int32s.InterleaveHi, archsimd.Int32s.Store, interleaveSlice[int32](0, true))
	testSVEShuffle2(t, int64s, 8, archsimd.Int64s{}.Len(), archsimd.LoadInt64s, archsimd.Int64s.InterleaveHi, archsimd.Int64s.Store, interleaveSlice[int64](0, true))
	testSVEShuffle2(t, uint8s, 1, archsimd.Uint8s{}.Len(), archsimd.LoadUint8s, archsimd.Uint8s.InterleaveHi, archsimd.Uint8s.Store, interleaveSlice[uint8](0, true))
	testSVEShuffle2(t, uint16s, 2, archsimd.Uint16s{}.Len(), archsimd.LoadUint16s, archsimd.Uint16s.InterleaveHi, archsimd.Uint16s.Store, interleaveSlice[uint16](0, true))
	testSVEShuffle2(t, uint32s, 4, archsimd.Uint32s{}.Len(), archsimd.LoadUint32s, archsimd.Uint32s.InterleaveHi, archsimd.Uint32s.Store, interleaveSlice[uint32](0, true))
	testSVEShuffle2(t, uint64s, 8, archsimd.Uint64s{}.Len(), archsimd.LoadUint64s, archsimd.Uint64s.InterleaveHi, archsimd.Uint64s.Store, interleaveSlice[uint64](0, true))
}

func TestConcatEvenSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testSVEShuffle2(t, int8s, 1, archsimd.Int8s{}.Len(), archsimd.LoadInt8s, archsimd.Int8s.ConcatEven, archsimd.Int8s.Store, deinterleaveSlice[int8](0, false))
	testSVEShuffle2(t, int16s, 2, archsimd.Int16s{}.Len(), archsimd.LoadInt16s, archsimd.Int16s.ConcatEven, archsimd.Int16s.Store, deinterleaveSlice[int16](0, false))
	testSVEShuffle2(t, int32s, 4, archsimd.Int32s{}.Len(), archsimd.LoadInt32s, archsimd.Int32s.ConcatEven, archsimd.Int32s.Store, deinterleaveSlice[int32](0, false))
	testSVEShuffle2(t, int64s, 8, archsimd.Int64s{}.Len(), archsimd.LoadInt64s, archsimd.Int64s.ConcatEven, archsimd.Int64s.Store, deinterleaveSlice[int64](0, false))
	testSVEShuffle2(t, uint8s, 1, archsimd.Uint8s{}.Len(), archsimd.LoadUint8s, archsimd.Uint8s.ConcatEven, archsimd.Uint8s.Store, deinterleaveSlice[uint8](0, false))
	testSVEShuffle2(t, uint16s, 2, archsimd.Uint16s{}.Len(), archsimd.LoadUint16s, archsimd.Uint16s.ConcatEven, archsimd.Uint16s.Store, deinterleaveSlice[uint16](0, false))
	testSVEShuffle2(t, uint32s, 4, archsimd.Uint32s{}.Len(), archsimd.LoadUint32s, archsimd.Uint32s.ConcatEven, archsimd.Uint32s.Store, deinterleaveSlice[uint32](0, false))
	testSVEShuffle2(t, uint64s, 8, archsimd.Uint64s{}.Len(), archsimd.LoadUint64s, archsimd.Uint64s.ConcatEven, archsimd.Uint64s.Store, deinterleaveSlice[uint64](0, false))
}

func TestConcatOddSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testSVEShuffle2(t, int8s, 1, archsimd.Int8s{}.Len(), archsimd.LoadInt8s, archsimd.Int8s.ConcatOdd, archsimd.Int8s.Store, deinterleaveSlice[int8](0, true))
	testSVEShuffle2(t, int16s, 2, archsimd.Int16s{}.Len(), archsimd.LoadInt16s, archsimd.Int16s.ConcatOdd, archsimd.Int16s.Store, deinterleaveSlice[int16](0, true))
	testSVEShuffle2(t, int32s, 4, archsimd.Int32s{}.Len(), archsimd.LoadInt32s, archsimd.Int32s.ConcatOdd, archsimd.Int32s.Store, deinterleaveSlice[int32](0, true))
	testSVEShuffle2(t, int64s, 8, archsimd.Int64s{}.Len(), archsimd.LoadInt64s, archsimd.Int64s.ConcatOdd, archsimd.Int64s.Store, deinterleaveSlice[int64](0, true))
	testSVEShuffle2(t, uint8s, 1, archsimd.Uint8s{}.Len(), archsimd.LoadUint8s, archsimd.Uint8s.ConcatOdd, archsimd.Uint8s.Store, deinterleaveSlice[uint8](0, true))
	testSVEShuffle2(t, uint16s, 2, archsimd.Uint16s{}.Len(), archsimd.LoadUint16s, archsimd.Uint16s.ConcatOdd, archsimd.Uint16s.Store, deinterleaveSlice[uint16](0, true))
	testSVEShuffle2(t, uint32s, 4, archsimd.Uint32s{}.Len(), archsimd.LoadUint32s, archsimd.Uint32s.ConcatOdd, archsimd.Uint32s.Store, deinterleaveSlice[uint32](0, true))
	testSVEShuffle2(t, uint64s, 8, archsimd.Uint64s{}.Len(), archsimd.LoadUint64s, archsimd.Uint64s.ConcatOdd, archsimd.Uint64s.Store, deinterleaveSlice[uint64](0, true))
}
