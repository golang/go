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
	f     func(V) V // x.ShiftAllLeft(shift) or x.ShiftAllRight(shift) with shift a constant
}

// testSVEShiftAllLeft checks ShiftAllLeft by every amount in amounts, passed
// as a variable, and by the constant amounts in consts, against Go's <<.
func testSVEShiftAllLeft[T integer, V any](
	t *testing.T, name string, n int, load func([]T) V, store func(V, []T),
	shift func(V, uint64) V, consts []sveShiftCase[V]) {
	t.Helper()
	testSVEShiftAll(t, name+".ShiftAllLeft", n, load, store, shift, consts, func(x T, s uint64) T { return x << s })
}

// testSVEShiftAllRight is testSVEShiftAllLeft for ShiftAllRight and Go's >>.
func testSVEShiftAllRight[T integer, V any](
	t *testing.T, name string, n int, load func([]T) V, store func(V, []T),
	shift func(V, uint64) V, consts []sveShiftCase[V]) {
	t.Helper()
	testSVEShiftAll(t, name+".ShiftAllRight", n, load, store, shift, consts, func(x T, s uint64) T { return x >> s })
}

// testSVEShiftAll checks name, the shift op, by every amount in amounts,
// passed as a variable, and by the constant amounts in consts, against ref.
func testSVEShiftAll[T integer, V any](
	t *testing.T, name string, n int, load func([]T) V, store func(V, []T),
	shift func(V, uint64) V, consts []sveShiftCase[V], ref func(T, uint64) T) {
	t.Helper()
	xs := make([]T, n)
	for i := range xs {
		xs[i] = T(i*37 - 5) // mixes signs and high bits
		if i%2 == 1 {
			xs[i] = ^xs[i]
		}
	}
	got := make([]T, n)
	check := func(s uint64, how string) {
		t.Helper()
		for i, x := range xs {
			if want := ref(x, s); got[i] != want {
				t.Errorf("%s(%d) %s: lane %d = %v, want %v", name, s, how, i, got[i], want)
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

func TestShiftAllRightSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testSVEShiftAllRight(t, "Int8s", archsimd.Int8s{}.Len(), archsimd.LoadInt8s, archsimd.Int8s.Store, archsimd.Int8s.ShiftAllRight,
		[]sveShiftCase[archsimd.Int8s]{
			{0, func(x archsimd.Int8s) archsimd.Int8s { return x.ShiftAllRight(0) }},
			{1, func(x archsimd.Int8s) archsimd.Int8s { return x.ShiftAllRight(1) }},
			{7, func(x archsimd.Int8s) archsimd.Int8s { return x.ShiftAllRight(7) }},
			{8, func(x archsimd.Int8s) archsimd.Int8s { return x.ShiftAllRight(8) }},
			{9, func(x archsimd.Int8s) archsimd.Int8s { return x.ShiftAllRight(9) }},
		})
	testSVEShiftAllRight(t, "Uint8s", archsimd.Uint8s{}.Len(), archsimd.LoadUint8s, archsimd.Uint8s.Store, archsimd.Uint8s.ShiftAllRight,
		[]sveShiftCase[archsimd.Uint8s]{
			{3, func(x archsimd.Uint8s) archsimd.Uint8s { return x.ShiftAllRight(3) }},
			{8, func(x archsimd.Uint8s) archsimd.Uint8s { return x.ShiftAllRight(8) }},
			{200, func(x archsimd.Uint8s) archsimd.Uint8s { return x.ShiftAllRight(200) }},
		})
	testSVEShiftAllRight(t, "Int16s", archsimd.Int16s{}.Len(), archsimd.LoadInt16s, archsimd.Int16s.Store, archsimd.Int16s.ShiftAllRight,
		[]sveShiftCase[archsimd.Int16s]{
			{15, func(x archsimd.Int16s) archsimd.Int16s { return x.ShiftAllRight(15) }},
			{16, func(x archsimd.Int16s) archsimd.Int16s { return x.ShiftAllRight(16) }},
			{100, func(x archsimd.Int16s) archsimd.Int16s { return x.ShiftAllRight(100) }},
		})
	testSVEShiftAllRight(t, "Uint16s", archsimd.Uint16s{}.Len(), archsimd.LoadUint16s, archsimd.Uint16s.Store, archsimd.Uint16s.ShiftAllRight,
		[]sveShiftCase[archsimd.Uint16s]{
			{4, func(x archsimd.Uint16s) archsimd.Uint16s { return x.ShiftAllRight(4) }},
			{16, func(x archsimd.Uint16s) archsimd.Uint16s { return x.ShiftAllRight(16) }},
		})
	testSVEShiftAllRight(t, "Int32s", archsimd.Int32s{}.Len(), archsimd.LoadInt32s, archsimd.Int32s.Store, archsimd.Int32s.ShiftAllRight,
		[]sveShiftCase[archsimd.Int32s]{
			{23, func(x archsimd.Int32s) archsimd.Int32s { return x.ShiftAllRight(23) }},
			{32, func(x archsimd.Int32s) archsimd.Int32s { return x.ShiftAllRight(32) }},
			{1 << 40, func(x archsimd.Int32s) archsimd.Int32s { return x.ShiftAllRight(1 << 40) }},
		})
	testSVEShiftAllRight(t, "Uint32s", archsimd.Uint32s{}.Len(), archsimd.LoadUint32s, archsimd.Uint32s.Store, archsimd.Uint32s.ShiftAllRight,
		[]sveShiftCase[archsimd.Uint32s]{
			{31, func(x archsimd.Uint32s) archsimd.Uint32s { return x.ShiftAllRight(31) }},
			{1 << 40, func(x archsimd.Uint32s) archsimd.Uint32s { return x.ShiftAllRight(1 << 40) }},
		})
	testSVEShiftAllRight(t, "Int64s", archsimd.Int64s{}.Len(), archsimd.LoadInt64s, archsimd.Int64s.Store, archsimd.Int64s.ShiftAllRight,
		[]sveShiftCase[archsimd.Int64s]{
			{52, func(x archsimd.Int64s) archsimd.Int64s { return x.ShiftAllRight(52) }},
			{64, func(x archsimd.Int64s) archsimd.Int64s { return x.ShiftAllRight(64) }},
			{65, func(x archsimd.Int64s) archsimd.Int64s { return x.ShiftAllRight(65) }},
			{math.MaxUint64, func(x archsimd.Int64s) archsimd.Int64s { return x.ShiftAllRight(math.MaxUint64) }},
		})
	testSVEShiftAllRight(t, "Uint64s", archsimd.Uint64s{}.Len(), archsimd.LoadUint64s, archsimd.Uint64s.Store, archsimd.Uint64s.ShiftAllRight,
		[]sveShiftCase[archsimd.Uint64s]{
			{33, func(x archsimd.Uint64s) archsimd.Uint64s { return x.ShiftAllRight(33) }},
			{64, func(x archsimd.Uint64s) archsimd.Uint64s { return x.ShiftAllRight(64) }},
			{65, func(x archsimd.Uint64s) archsimd.Uint64s { return x.ShiftAllRight(65) }},
		})
}

// shiftAllRightIfElse returns, lane by lane, x >> s where x > y and x
// elsewhere: x.ShiftAllRight(s).IfElse(x.Greater(y), x).
func shiftAllRightIfElse[T integer](s uint64) func(x, y []T) []T {
	return func(x, y []T) []T {
		r := shiftAllRightSlice(x, s)
		for i := range r {
			if !(x[i] > y[i]) {
				r[i] = x[i]
			}
		}
		return r
	}
}

// TestShiftAllRightIfElseSVE checks ShiftAllRight under a select, which folds
// into its merging form, by a constant, by a constant past the element width,
// and by a variable.
func TestShiftAllRightIfElseSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	for _, s := range []uint64{5, 70} {
		testInt8sBinary(t, func(x, y archsimd.Int8s) archsimd.Int8s { return x.ShiftAllRight(s).IfElse(x.Greater(y), x) }, shiftAllRightIfElse[int8](s))
		testUint16sBinary(t, func(x, y archsimd.Uint16s) archsimd.Uint16s { return x.ShiftAllRight(s).IfElse(x.Greater(y), x) }, shiftAllRightIfElse[uint16](s))
		testInt64sBinary(t, func(x, y archsimd.Int64s) archsimd.Int64s { return x.ShiftAllRight(s).IfElse(x.Greater(y), x) }, shiftAllRightIfElse[int64](s))
		testUint64sBinary(t, func(x, y archsimd.Uint64s) archsimd.Uint64s { return x.ShiftAllRight(s).IfElse(x.Greater(y), x) }, shiftAllRightIfElse[uint64](s))
	}
	testInt8sBinary(t, func(x, y archsimd.Int8s) archsimd.Int8s { return x.ShiftAllRight(3).IfElse(x.Greater(y), x) }, shiftAllRightIfElse[int8](3))
	testInt8sBinary(t, func(x, y archsimd.Int8s) archsimd.Int8s { return x.ShiftAllRight(9).IfElse(x.Greater(y), x) }, shiftAllRightIfElse[int8](9))
	testUint16sBinary(t, func(x, y archsimd.Uint16s) archsimd.Uint16s { return x.ShiftAllRight(16).IfElse(x.Greater(y), x) }, shiftAllRightIfElse[uint16](16))
	testUint16sBinary(t, func(x, y archsimd.Uint16s) archsimd.Uint16s { return x.ShiftAllRight(17).IfElse(x.Greater(y), x) }, shiftAllRightIfElse[uint16](17))
	testInt64sBinary(t, func(x, y archsimd.Int64s) archsimd.Int64s { return x.ShiftAllRight(64).IfElse(x.Greater(y), x) }, shiftAllRightIfElse[int64](64))
	testInt64sBinary(t, func(x, y archsimd.Int64s) archsimd.Int64s { return x.ShiftAllRight(65).IfElse(x.Greater(y), x) }, shiftAllRightIfElse[int64](65))
	testUint64sBinary(t, func(x, y archsimd.Uint64s) archsimd.Uint64s { return x.ShiftAllRight(3).IfElse(x.Greater(y), x) }, shiftAllRightIfElse[uint64](3))
	testUint64sBinary(t, func(x, y archsimd.Uint64s) archsimd.Uint64s { return x.ShiftAllRight(65).IfElse(x.Greater(y), x) }, shiftAllRightIfElse[uint64](65))
	testInt16sBinary(t, func(x, y archsimd.Int16s) archsimd.Int16s { return x.ShiftAllRight(1<<40).IfElse(x.Greater(y), x) }, shiftAllRightIfElse[int16](1<<40))
	testUint32sBinary(t, func(x, y archsimd.Uint32s) archsimd.Uint32s { return x.ShiftAllRight(1<<40).IfElse(x.Greater(y), x) }, shiftAllRightIfElse[uint32](1<<40))
	testInt32sBinary(t, func(x, y archsimd.Int32s) archsimd.Int32s { return x.ShiftAllRight(7).IfElse(x.Greater(y), x) }, shiftAllRightIfElse[int32](7))
	testUint8sBinary(t, func(x, y archsimd.Uint8s) archsimd.Uint8s { return x.ShiftAllRight(8).IfElse(x.Greater(y), x) }, shiftAllRightIfElse[uint8](8))
	testInt64sBinary(t, func(x, y archsimd.Int64s) archsimd.Int64s {
		return x.ShiftAllRight(math.MaxUint64).IfElse(x.Greater(y), x)
	}, shiftAllRightIfElse[int64](math.MaxUint64))
}

// testSVEShift drives a shift by vector. Every window of xs is paired with
// every window of amounts. active is the runtime number of live lanes.
func testSVEShift[T, S integer, V, A any](t *testing.T, xs []T, amounts []S, elemBytes, active int,
	loadX func([]T) V, loadA func([]S) A, f func(V, A) V, store func(V, []T), want func(T, S) T) {
	t.Helper()
	count := sveMaxBytes / elemBytes // lanes in the fixed backing array
	for i := 0; i < len(xs)-count; i++ {
		for j := 0; j < len(amounts)-count; j++ {
			x, y := xs[i:i+count], amounts[j:j+count]
			g := make([]T, count)
			store(f(loadX(x), loadA(y)), g)
			w := make([]T, count)
			for k := range w {
				w[k] = want(x[k], y[k])
			}
			if !checkSlicesLogInput(t, g[:active], w[:active], 0.0, func() {
				t.Helper()
				t.Logf("x=%v", x)
				t.Logf("shift=%v", y)
			}) {
				return
			}
		}
	}
}

// shiftAmounts returns shift amounts for bits-bit elements: small ones, the
// element width and either side of it, larger ones, and ones with only high
// bits set, since SVE reads the whole element as the amount.
func shiftAmounts[S integer](bits int) []S {
	top := ^S(0)
	return nOf(N, []S{0, 1, 2, 3, 5, S(bits / 2), S(bits - 1), S(bits), S(bits + 1), S(2 * bits),
		S(1) << (bits - 1), S(1) << (bits - 2), top - 1, top})
}

func shiftLeft[T, S integer](x T, s S) T  { return x << s }
func shiftRight[T, S integer](x T, s S) T { return x >> s }

func TestShiftLeftSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testSVEShift(t, int8s, shiftAmounts[uint8](8), 1, archsimd.Int8s{}.Len(),
		archsimd.LoadInt8s, archsimd.LoadUint8s, archsimd.Int8s.ShiftLeft, archsimd.Int8s.Store, shiftLeft[int8, uint8])
	testSVEShift(t, int16s, shiftAmounts[uint16](16), 2, archsimd.Int16s{}.Len(),
		archsimd.LoadInt16s, archsimd.LoadUint16s, archsimd.Int16s.ShiftLeft, archsimd.Int16s.Store, shiftLeft[int16, uint16])
	testSVEShift(t, int32s, shiftAmounts[uint32](32), 4, archsimd.Int32s{}.Len(),
		archsimd.LoadInt32s, archsimd.LoadUint32s, archsimd.Int32s.ShiftLeft, archsimd.Int32s.Store, shiftLeft[int32, uint32])
	testSVEShift(t, int64s, shiftAmounts[uint64](64), 8, archsimd.Int64s{}.Len(),
		archsimd.LoadInt64s, archsimd.LoadUint64s, archsimd.Int64s.ShiftLeft, archsimd.Int64s.Store, shiftLeft[int64, uint64])
	testSVEShift(t, uint8s, shiftAmounts[uint8](8), 1, archsimd.Uint8s{}.Len(),
		archsimd.LoadUint8s, archsimd.LoadUint8s, archsimd.Uint8s.ShiftLeft, archsimd.Uint8s.Store, shiftLeft[uint8, uint8])
	testSVEShift(t, uint16s, shiftAmounts[uint16](16), 2, archsimd.Uint16s{}.Len(),
		archsimd.LoadUint16s, archsimd.LoadUint16s, archsimd.Uint16s.ShiftLeft, archsimd.Uint16s.Store, shiftLeft[uint16, uint16])
	testSVEShift(t, uint32s, shiftAmounts[uint32](32), 4, archsimd.Uint32s{}.Len(),
		archsimd.LoadUint32s, archsimd.LoadUint32s, archsimd.Uint32s.ShiftLeft, archsimd.Uint32s.Store, shiftLeft[uint32, uint32])
	testSVEShift(t, uint64s, shiftAmounts[uint64](64), 8, archsimd.Uint64s{}.Len(),
		archsimd.LoadUint64s, archsimd.LoadUint64s, archsimd.Uint64s.ShiftLeft, archsimd.Uint64s.Store, shiftLeft[uint64, uint64])
}

// shiftIfElse returns, lane by lane, op(x, s) where s is odd and x elsewhere:
// x.Op(s).IfElse(m, x) for m the odd lanes of s.
func shiftIfElse[T, S integer](op func(T, S) T) func(T, S) T {
	return func(x T, s S) T {
		if s&1 != 0 {
			return op(x, s)
		}
		return x
	}
}

// TestShiftLeftIfElseSVE checks ShiftLeft under a select, which folds into its
// merging form.
func TestShiftLeftIfElseSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	odd8 := func(s archsimd.Uint8s) archsimd.Mask8s {
		return s.And(archsimd.BroadcastUint8s(1)).NotEqual(archsimd.Uint8s{})
	}
	odd16 := func(s archsimd.Uint16s) archsimd.Mask16s {
		return s.And(archsimd.BroadcastUint16s(1)).NotEqual(archsimd.Uint16s{})
	}
	odd32 := func(s archsimd.Uint32s) archsimd.Mask32s {
		return s.And(archsimd.BroadcastUint32s(1)).NotEqual(archsimd.Uint32s{})
	}
	odd64 := func(s archsimd.Uint64s) archsimd.Mask64s {
		return s.And(archsimd.BroadcastUint64s(1)).NotEqual(archsimd.Uint64s{})
	}
	testSVEShift(t, int8s, shiftAmounts[uint8](8), 1, archsimd.Int8s{}.Len(), archsimd.LoadInt8s, archsimd.LoadUint8s,
		func(x archsimd.Int8s, s archsimd.Uint8s) archsimd.Int8s { return x.ShiftLeft(s).IfElse(odd8(s), x) },
		archsimd.Int8s.Store, shiftIfElse(shiftLeft[int8, uint8]))
	testSVEShift(t, uint16s, shiftAmounts[uint16](16), 2, archsimd.Uint16s{}.Len(), archsimd.LoadUint16s, archsimd.LoadUint16s,
		func(x archsimd.Uint16s, s archsimd.Uint16s) archsimd.Uint16s {
			return x.ShiftLeft(s).IfElse(odd16(s), x)
		},
		archsimd.Uint16s.Store, shiftIfElse(shiftLeft[uint16, uint16]))
	testSVEShift(t, int32s, shiftAmounts[uint32](32), 4, archsimd.Int32s{}.Len(), archsimd.LoadInt32s, archsimd.LoadUint32s,
		func(x archsimd.Int32s, s archsimd.Uint32s) archsimd.Int32s { return x.ShiftLeft(s).IfElse(odd32(s), x) },
		archsimd.Int32s.Store, shiftIfElse(shiftLeft[int32, uint32]))
	testSVEShift(t, uint64s, shiftAmounts[uint64](64), 8, archsimd.Uint64s{}.Len(), archsimd.LoadUint64s, archsimd.LoadUint64s,
		func(x archsimd.Uint64s, s archsimd.Uint64s) archsimd.Uint64s {
			return x.ShiftLeft(s).IfElse(odd64(s), x)
		},
		archsimd.Uint64s.Store, shiftIfElse(shiftLeft[uint64, uint64]))
}

// TestShiftLeftBroadcastSVE checks 64-bit ShiftLeft by a broadcast constant,
// which shares its instruction with ShiftAllLeft and so its immediate form.
func TestShiftLeftBroadcastSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	n := archsimd.Uint64s{}.Len()
	xs := make([]uint64, n)
	for i := range xs {
		xs[i] = uint64(i)*0x9e3779b97f4a7c15 + 1
	}
	got := make([]uint64, n)
	for _, c := range []struct {
		s uint64
		f func(archsimd.Uint64s) archsimd.Uint64s
	}{
		{5, func(x archsimd.Uint64s) archsimd.Uint64s { return x.ShiftLeft(archsimd.BroadcastUint64s(5)) }},
		{63, func(x archsimd.Uint64s) archsimd.Uint64s { return x.ShiftLeft(archsimd.BroadcastUint64s(63)) }},
		{64, func(x archsimd.Uint64s) archsimd.Uint64s { return x.ShiftLeft(archsimd.BroadcastUint64s(64)) }},
		{math.MaxUint64, func(x archsimd.Uint64s) archsimd.Uint64s {
			return x.ShiftLeft(archsimd.BroadcastUint64s(math.MaxUint64))
		}},
	} {
		c.f(archsimd.LoadUint64s(xs)).Store(got)
		for i, x := range xs {
			if want := x << c.s; got[i] != want {
				t.Errorf("Uint64s.ShiftLeft(Broadcast(%d)): lane %d = %#x, want %#x", c.s, i, got[i], want)
			}
		}
	}
}

func TestShiftRightSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testSVEShift(t, int8s, shiftAmounts[uint8](8), 1, archsimd.Int8s{}.Len(),
		archsimd.LoadInt8s, archsimd.LoadUint8s, archsimd.Int8s.ShiftRight, archsimd.Int8s.Store, shiftRight[int8, uint8])
	testSVEShift(t, int16s, shiftAmounts[uint16](16), 2, archsimd.Int16s{}.Len(),
		archsimd.LoadInt16s, archsimd.LoadUint16s, archsimd.Int16s.ShiftRight, archsimd.Int16s.Store, shiftRight[int16, uint16])
	testSVEShift(t, int32s, shiftAmounts[uint32](32), 4, archsimd.Int32s{}.Len(),
		archsimd.LoadInt32s, archsimd.LoadUint32s, archsimd.Int32s.ShiftRight, archsimd.Int32s.Store, shiftRight[int32, uint32])
	testSVEShift(t, int64s, shiftAmounts[uint64](64), 8, archsimd.Int64s{}.Len(),
		archsimd.LoadInt64s, archsimd.LoadUint64s, archsimd.Int64s.ShiftRight, archsimd.Int64s.Store, shiftRight[int64, uint64])
	testSVEShift(t, uint8s, shiftAmounts[uint8](8), 1, archsimd.Uint8s{}.Len(),
		archsimd.LoadUint8s, archsimd.LoadUint8s, archsimd.Uint8s.ShiftRight, archsimd.Uint8s.Store, shiftRight[uint8, uint8])
	testSVEShift(t, uint16s, shiftAmounts[uint16](16), 2, archsimd.Uint16s{}.Len(),
		archsimd.LoadUint16s, archsimd.LoadUint16s, archsimd.Uint16s.ShiftRight, archsimd.Uint16s.Store, shiftRight[uint16, uint16])
	testSVEShift(t, uint32s, shiftAmounts[uint32](32), 4, archsimd.Uint32s{}.Len(),
		archsimd.LoadUint32s, archsimd.LoadUint32s, archsimd.Uint32s.ShiftRight, archsimd.Uint32s.Store, shiftRight[uint32, uint32])
	testSVEShift(t, uint64s, shiftAmounts[uint64](64), 8, archsimd.Uint64s{}.Len(),
		archsimd.LoadUint64s, archsimd.LoadUint64s, archsimd.Uint64s.ShiftRight, archsimd.Uint64s.Store, shiftRight[uint64, uint64])
}

// TestShiftRightIfElseSVE checks ShiftRight under a select, which folds into
// its merging form.
func TestShiftRightIfElseSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	odd8 := func(s archsimd.Uint8s) archsimd.Mask8s {
		return s.And(archsimd.BroadcastUint8s(1)).NotEqual(archsimd.Uint8s{})
	}
	odd16 := func(s archsimd.Uint16s) archsimd.Mask16s {
		return s.And(archsimd.BroadcastUint16s(1)).NotEqual(archsimd.Uint16s{})
	}
	odd32 := func(s archsimd.Uint32s) archsimd.Mask32s {
		return s.And(archsimd.BroadcastUint32s(1)).NotEqual(archsimd.Uint32s{})
	}
	odd64 := func(s archsimd.Uint64s) archsimd.Mask64s {
		return s.And(archsimd.BroadcastUint64s(1)).NotEqual(archsimd.Uint64s{})
	}
	testSVEShift(t, int8s, shiftAmounts[uint8](8), 1, archsimd.Int8s{}.Len(), archsimd.LoadInt8s, archsimd.LoadUint8s,
		func(x archsimd.Int8s, s archsimd.Uint8s) archsimd.Int8s { return x.ShiftRight(s).IfElse(odd8(s), x) },
		archsimd.Int8s.Store, shiftIfElse(shiftRight[int8, uint8]))
	testSVEShift(t, int16s, shiftAmounts[uint16](16), 2, archsimd.Int16s{}.Len(), archsimd.LoadInt16s, archsimd.LoadUint16s,
		func(x archsimd.Int16s, s archsimd.Uint16s) archsimd.Int16s {
			return x.ShiftRight(s).IfElse(odd16(s), x)
		},
		archsimd.Int16s.Store, shiftIfElse(shiftRight[int16, uint16]))
	testSVEShift(t, uint8s, shiftAmounts[uint8](8), 1, archsimd.Uint8s{}.Len(), archsimd.LoadUint8s, archsimd.LoadUint8s,
		func(x archsimd.Uint8s, s archsimd.Uint8s) archsimd.Uint8s { return x.ShiftRight(s).IfElse(odd8(s), x) },
		archsimd.Uint8s.Store, shiftIfElse(shiftRight[uint8, uint8]))
	testSVEShift(t, uint16s, shiftAmounts[uint16](16), 2, archsimd.Uint16s{}.Len(), archsimd.LoadUint16s, archsimd.LoadUint16s,
		func(x archsimd.Uint16s, s archsimd.Uint16s) archsimd.Uint16s {
			return x.ShiftRight(s).IfElse(odd16(s), x)
		},
		archsimd.Uint16s.Store, shiftIfElse(shiftRight[uint16, uint16]))
	testSVEShift(t, uint32s, shiftAmounts[uint32](32), 4, archsimd.Uint32s{}.Len(), archsimd.LoadUint32s, archsimd.LoadUint32s,
		func(x archsimd.Uint32s, s archsimd.Uint32s) archsimd.Uint32s {
			return x.ShiftRight(s).IfElse(odd32(s), x)
		},
		archsimd.Uint32s.Store, shiftIfElse(shiftRight[uint32, uint32]))
	testSVEShift(t, int32s, shiftAmounts[uint32](32), 4, archsimd.Int32s{}.Len(), archsimd.LoadInt32s, archsimd.LoadUint32s,
		func(x archsimd.Int32s, s archsimd.Uint32s) archsimd.Int32s {
			return x.ShiftRight(s).IfElse(odd32(s), x)
		},
		archsimd.Int32s.Store, shiftIfElse(shiftRight[int32, uint32]))
	testSVEShift(t, int64s, shiftAmounts[uint64](64), 8, archsimd.Int64s{}.Len(), archsimd.LoadInt64s, archsimd.LoadUint64s,
		func(x archsimd.Int64s, s archsimd.Uint64s) archsimd.Int64s {
			return x.ShiftRight(s).IfElse(odd64(s), x)
		},
		archsimd.Int64s.Store, shiftIfElse(shiftRight[int64, uint64]))
	testSVEShift(t, uint64s, shiftAmounts[uint64](64), 8, archsimd.Uint64s{}.Len(), archsimd.LoadUint64s, archsimd.LoadUint64s,
		func(x archsimd.Uint64s, s archsimd.Uint64s) archsimd.Uint64s {
			return x.ShiftRight(s).IfElse(odd64(s), x)
		},
		archsimd.Uint64s.Store, shiftIfElse(shiftRight[uint64, uint64]))
}
