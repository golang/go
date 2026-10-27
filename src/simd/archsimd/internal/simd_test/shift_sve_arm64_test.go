// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"math"
	"simd/archsimd"
	"testing"
)

type sveShiftCase[V any] struct {
	shift uint64
	f     func(V) V // x.ShiftAllLeft(shift) with shift a constant
}

// testSVEShiftAllLeft checks ShiftAllLeft by every amount in amounts, passed
// as a variable, and by the constant amounts in consts, against Go's <<.
func testSVEShiftAllLeft[T int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64, V any](
	t *testing.T, name string, n int, load func([]T) V, store func(V, []T),
	shift func(V, uint64) V, consts []sveShiftCase[V]) {
	t.Helper()
	xs := make([]T, n)
	for i := range xs {
		xs[i] = T(i*37 - 5) // mixes signs and high bits
	}
	got := make([]T, n)
	check := func(s uint64, how string) {
		t.Helper()
		for i, x := range xs {
			if want := x << s; got[i] != want {
				t.Errorf("%s.ShiftAllLeft(%d) %s: lane %d = %v, want %v", name, s, how, i, got[i], want)
				return
			}
		}
	}
	for _, s := range []uint64{0, 1, 3, 7, 8, 15, 16, 31, 32, 63, 64, 65, 255, 256, 1 << 40, math.MaxUint64} {
		store(shift(load(xs), s), got)
		check(s, "by variable")
	}
	for _, c := range consts {
		store(c.f(load(xs)), got)
		check(c.shift, "by constant")
	}
}

func TestShiftAllLeftSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testSVEShiftAllLeft(t, "Int8s", archsimd.Int8s{}.Len(), archsimd.LoadInt8s, archsimd.Int8s.Store, archsimd.Int8s.ShiftAllLeft,
		[]sveShiftCase[archsimd.Int8s]{
			{3, func(x archsimd.Int8s) archsimd.Int8s { return x.ShiftAllLeft(3) }},
			{7, func(x archsimd.Int8s) archsimd.Int8s { return x.ShiftAllLeft(7) }},
			{8, func(x archsimd.Int8s) archsimd.Int8s { return x.ShiftAllLeft(8) }},
		})
	testSVEShiftAllLeft(t, "Uint8s", archsimd.Uint8s{}.Len(), archsimd.LoadUint8s, archsimd.Uint8s.Store, archsimd.Uint8s.ShiftAllLeft,
		[]sveShiftCase[archsimd.Uint8s]{
			{1, func(x archsimd.Uint8s) archsimd.Uint8s { return x.ShiftAllLeft(1) }},
			{200, func(x archsimd.Uint8s) archsimd.Uint8s { return x.ShiftAllLeft(200) }},
		})
	testSVEShiftAllLeft(t, "Int16s", archsimd.Int16s{}.Len(), archsimd.LoadInt16s, archsimd.Int16s.Store, archsimd.Int16s.ShiftAllLeft,
		[]sveShiftCase[archsimd.Int16s]{
			{15, func(x archsimd.Int16s) archsimd.Int16s { return x.ShiftAllLeft(15) }},
			{16, func(x archsimd.Int16s) archsimd.Int16s { return x.ShiftAllLeft(16) }},
		})
	testSVEShiftAllLeft(t, "Uint16s", archsimd.Uint16s{}.Len(), archsimd.LoadUint16s, archsimd.Uint16s.Store, archsimd.Uint16s.ShiftAllLeft,
		[]sveShiftCase[archsimd.Uint16s]{
			{4, func(x archsimd.Uint16s) archsimd.Uint16s { return x.ShiftAllLeft(4) }},
		})
	testSVEShiftAllLeft(t, "Int32s", archsimd.Int32s{}.Len(), archsimd.LoadInt32s, archsimd.Int32s.Store, archsimd.Int32s.ShiftAllLeft,
		[]sveShiftCase[archsimd.Int32s]{
			{23, func(x archsimd.Int32s) archsimd.Int32s { return x.ShiftAllLeft(23) }},
			{31, func(x archsimd.Int32s) archsimd.Int32s { return x.ShiftAllLeft(31) }},
			{32, func(x archsimd.Int32s) archsimd.Int32s { return x.ShiftAllLeft(32) }},
		})
	testSVEShiftAllLeft(t, "Uint32s", archsimd.Uint32s{}.Len(), archsimd.LoadUint32s, archsimd.Uint32s.Store, archsimd.Uint32s.ShiftAllLeft,
		[]sveShiftCase[archsimd.Uint32s]{
			{0, func(x archsimd.Uint32s) archsimd.Uint32s { return x.ShiftAllLeft(0) }},
		})
	testSVEShiftAllLeft(t, "Int64s", archsimd.Int64s{}.Len(), archsimd.LoadInt64s, archsimd.Int64s.Store, archsimd.Int64s.ShiftAllLeft,
		[]sveShiftCase[archsimd.Int64s]{
			{52, func(x archsimd.Int64s) archsimd.Int64s { return x.ShiftAllLeft(52) }},
			{63, func(x archsimd.Int64s) archsimd.Int64s { return x.ShiftAllLeft(63) }},
			{64, func(x archsimd.Int64s) archsimd.Int64s { return x.ShiftAllLeft(64) }},
		})
	testSVEShiftAllLeft(t, "Uint64s", archsimd.Uint64s{}.Len(), archsimd.LoadUint64s, archsimd.Uint64s.Store, archsimd.Uint64s.ShiftAllLeft,
		[]sveShiftCase[archsimd.Uint64s]{
			{33, func(x archsimd.Uint64s) archsimd.Uint64s { return x.ShiftAllLeft(33) }},
		})
}
