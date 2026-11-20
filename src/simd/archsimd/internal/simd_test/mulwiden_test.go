// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (arm64 || wasm)

package simd_test

import (
	"simd/archsimd"
	"testing"
)

// TestMulWiden tests x.MulWiden(y) and x.HiToLo().MulWiden(y.HiToLo()). The
// latter compiles to a single instruction that multiplies the high halves of x
// and y, so it's written out in full in each closure to exercise that.
func TestMulWiden(t *testing.T) {
	testMulWiden(t, int8s, 16, archsimd.LoadInt8x16, archsimd.Int16x8.Store,
		archsimd.Int8x16.MulWiden,
		func(x, y archsimd.Int8x16) archsimd.Int16x8 { return x.HiToLo().MulWiden(y.HiToLo()) })
	testMulWiden(t, int16s, 8, archsimd.LoadInt16x8, archsimd.Int32x4.Store,
		archsimd.Int16x8.MulWiden,
		func(x, y archsimd.Int16x8) archsimd.Int32x4 { return x.HiToLo().MulWiden(y.HiToLo()) })
	testMulWiden(t, int32s, 4, archsimd.LoadInt32x4, archsimd.Int64x2.Store,
		archsimd.Int32x4.MulWiden,
		func(x, y archsimd.Int32x4) archsimd.Int64x2 { return x.HiToLo().MulWiden(y.HiToLo()) })

	testMulWiden(t, uint8s, 16, archsimd.LoadUint8x16, archsimd.Uint16x8.Store,
		archsimd.Uint8x16.MulWiden,
		func(x, y archsimd.Uint8x16) archsimd.Uint16x8 { return x.HiToLo().MulWiden(y.HiToLo()) })
	testMulWiden(t, uint16s, 8, archsimd.LoadUint16x8, archsimd.Uint32x4.Store,
		archsimd.Uint16x8.MulWiden,
		func(x, y archsimd.Uint16x8) archsimd.Uint32x4 { return x.HiToLo().MulWiden(y.HiToLo()) })
	testMulWiden(t, uint32s, 4, archsimd.LoadUint32x4, archsimd.Uint64x2.Store,
		archsimd.Uint32x4.MulWiden,
		func(x, y archsimd.Uint32x4) archsimd.Uint64x2 { return x.HiToLo().MulWiden(y.HiToLo()) })
}

func testMulWiden[X, Z number, VX, VZ any](t *testing.T, s []X, n int, load func([]X) VX, store func(VZ, []Z), mulWiden, mulWidenHi func(VX, VX) VZ) {
	t.Helper()
	want := func(x, y []X) []Z {
		z := make([]Z, n/2)
		for i := range z {
			z[i] = Z(x[i]) * Z(y[i])
		}
		return z
	}
	forSlicePair(t, s, n, func(x, y []X) bool {
		t.Helper()
		logInput := func() { t.Helper(); t.Logf("x=%v", x); t.Logf("y=%v", y) }

		got := make([]Z, n/2)
		store(mulWiden(load(x), load(y)), got)
		if !checkSlicesLogInput(t, got, want(x, y), 0.0, logInput) {
			return false
		}

		store(mulWidenHi(load(x), load(y)), got)
		return checkSlicesLogInput(t, got, want(x[n/2:], y[n/2:]), 0.0, logInput)
	})
}
