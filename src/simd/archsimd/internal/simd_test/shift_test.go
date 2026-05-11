// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestShift(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("requires AVX2")
	}

	testInt32x4Binary(t,
		func(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftLeft(y.AsUint32x4()) },
		map2(func(x, y int32) int32 { return x << uint32(y) }))
	testInt32x4Binary(t,
		func(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftRight(y.AsUint32x4()) },
		map2(func(x, y int32) int32 { return x >> uint32(y) }))
	testUint32x4Binary(t,
		func(x, y archsimd.Uint32x4) archsimd.Uint32x4 { return x.ShiftRight(y) },
		map2(func(x, y uint32) uint32 { return x >> y }))
}

func TestShiftAll(t *testing.T) {
	// Test both const and non-const shifts.
	// Test both regular and over-shifts.

	hide := hideConst[uint64]

	// ShiftAllLeft

	testInt32x4Unary(t,
		func(x archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllLeft(2) },
		map1(func(x int32) int32 { return x << 2 }))
	testInt32x4Unary(t,
		func(x archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllLeft(hide(2)) },
		map1(func(x int32) int32 { return x << hide(2) }))

	// Ironically, we have to hide the constant in the want function so the
	// compiler doesn't complain about a silly shift.
	testInt32x4Unary(t,
		func(x archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllLeft(0x1000) },
		map1(func(x int32) int32 { return x << hide(0x1000) }))
	testInt32x4Unary(t,
		func(x archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllLeft(hide(0x1000)) },
		map1(func(x int32) int32 { return x << hide(0x1000) }))

	// Signed ShiftAllRight

	testInt32x4Unary(t,
		func(x archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllRight(2) },
		map1(func(x int32) int32 { return x >> 2 }))
	testInt32x4Unary(t,
		func(x archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllRight(hide(2)) },
		map1(func(x int32) int32 { return x >> hide(2) }))

	testInt32x4Unary(t,
		func(x archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllRight(0x1000) },
		map1(func(x int32) int32 { return x >> hide(0x1000) }))
	testInt32x4Unary(t,
		func(x archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllRight(hide(0x1000)) },
		map1(func(x int32) int32 { return x >> hide(0x1000) }))

	// Unsigned ShiftAllRight

	testUint32x4Unary(t,
		func(x archsimd.Uint32x4) archsimd.Uint32x4 { return x.ShiftAllRight(2) },
		map1(func(x uint32) uint32 { return x >> 2 }))
	testUint32x4Unary(t,
		func(x archsimd.Uint32x4) archsimd.Uint32x4 { return x.ShiftAllRight(hide(2)) },
		map1(func(x uint32) uint32 { return x >> hide(2) }))

	testUint32x4Unary(t,
		func(x archsimd.Uint32x4) archsimd.Uint32x4 { return x.ShiftAllRight(0x1000) },
		map1(func(x uint32) uint32 { return x >> hide(0x1000) }))
	testUint32x4Unary(t,
		func(x archsimd.Uint32x4) archsimd.Uint32x4 { return x.ShiftAllRight(hide(0x1000)) },
		map1(func(x uint32) uint32 { return x >> hide(0x1000) }))
}

func concatInt32s(x, y int32) int64 {
	return (int64(x) << 32) | int64(uint32(y))
}

func concatUint32s(x, y uint32) uint64 {
	return (uint64(x) << 32) | uint64(y)
}

func TestShiftAllConcat(t *testing.T) {
	if !archsimd.X86.AVX512VBMI2() {
		t.Skip("requires AVX512-VBMI2")
	}

	// Note that unlike their non-Concat counterparts, these wrap the shift count.

	hide := hideConst[uint8]

	// ShiftAllLeftConcat
	salc := func(shift uint8) func(x, y int32) int32 {
		return func(x, y int32) int32 {
			return int32(concatInt32s(x, y) >> (32 - shift%32))
		}
	}

	testInt32x4Binary(t,
		func(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllLeftConcatMod32(y, 2) },
		map2(salc(2)))
	testInt32x4Binary(t,
		func(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllLeftConcatMod32(y, hide(2)) },
		map2(salc(hide(2))))

	// TODO: If we expand the shift from uint8, add larger cases (e.g., 0x1000).
	// The uint8 conversion is only here so the build will fail if we change it.

	testInt32x4Binary(t,
		func(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllLeftConcatMod32(y, uint8(128)) },
		map2(salc(128)))
	testInt32x4Binary(t,
		func(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllLeftConcatMod32(y, hide(128)) },
		map2(salc(hide(128))))

	// Signed ShiftAllRightConcat
	sarc := func(shift uint8) func(x, y int32) int32 {
		return func(x, y int32) int32 {
			return int32(concatInt32s(y, x) >> (shift % 32))
		}
	}

	testInt32x4Binary(t,
		func(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllRightConcatMod32(y, 2) },
		map2(sarc(2)))
	testInt32x4Binary(t,
		func(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllRightConcatMod32(y, hide(2)) },
		map2(sarc(hide(2))))

	testInt32x4Binary(t,
		func(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllRightConcatMod32(y, uint8(128)) },
		map2(sarc(128)))
	testInt32x4Binary(t,
		func(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllRightConcatMod32(y, hide(128)) },
		map2(sarc(hide(128))))

	// Unsigned ShiftAllRightConcat
	usarc := func(shift uint8) func(x, y uint32) uint32 {
		return func(x, y uint32) uint32 {
			return uint32(concatUint32s(y, x) >> (shift % 32))
		}
	}

	testUint32x4Binary(t,
		func(x, y archsimd.Uint32x4) archsimd.Uint32x4 { return x.ShiftAllRightConcatMod32(y, 2) },
		map2(usarc(2)))
	testUint32x4Binary(t,
		func(x, y archsimd.Uint32x4) archsimd.Uint32x4 { return x.ShiftAllRightConcatMod32(y, hide(2)) },
		map2(usarc(hide(2))))

	testUint32x4Binary(t,
		func(x, y archsimd.Uint32x4) archsimd.Uint32x4 { return x.ShiftAllRightConcatMod32(y, uint8(128)) },
		map2(usarc(128)))
	testUint32x4Binary(t,
		func(x, y archsimd.Uint32x4) archsimd.Uint32x4 { return x.ShiftAllRightConcatMod32(y, hide(128)) },
		map2(usarc(hide(128))))
}

func TestShiftConcat(t *testing.T) {
	if !archsimd.X86.AVX512VBMI2() {
		t.Skip("requires AVX512-VBMI2")
	}

	// Note that unlike their non-Concat counterparts, these wrap the shift count.

	testInt32x4Ternary(t,
		func(x, y, z archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftLeftConcatMod32(y, z.AsUint32x4()) },
		map3(func(x, y, z int32) int32 {
			return int32(concatInt32s(x, y) >> (32 - uint32(z)%32))
		}))

	testInt32x4Ternary(t,
		func(x, y, z archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftRightConcatMod32(y, z.AsUint32x4()) },
		map3(func(x, y, z int32) int32 {
			return int32(concatInt32s(y, x) >> (uint32(z) % 32))
		}))

	testUint32x4Ternary(t,
		func(x, y, z archsimd.Uint32x4) archsimd.Uint32x4 { return x.ShiftRightConcatMod32(y, z) },
		map3(func(x, y, z uint32) uint32 {
			return uint32(concatUint32s(y, x) >> (z % 32))
		}))
}

func TestConcatShiftBytesRight(t *testing.T) {
	hide := hideConst[uint8]

	csbr := func(shift uint8) func(x, y []uint8) []uint8 {
		return func(x, y []uint8) []uint8 {
			z := make([]uint8, len(x))
			for i := range z {
				target := i + int(shift)
				if target < 16 {
					z[i] = y[target]
				} else if target < 32 {
					z[i] = x[(target - 16)]
				}
			}
			return z
		}
	}

	t.Run("Uint8x16", func(t *testing.T) {
		if !archsimd.X86.AVX() {
			t.Skip("requires AVX")
		}
		// TODO: If we expand the shift from uint8, add larger cases (e.g., 0x1000)
		for _, shift := range []uint8{0, 2, 16, 20, 32, 128} {
			t.Log("shift", shift)
			testUint8x16Binary(t,
				func(x, y archsimd.Uint8x16) archsimd.Uint8x16 { return x.ConcatShiftBytesRight(y, shift) },
				csbr(shift))
			testUint8x16Binary(t,
				func(x, y archsimd.Uint8x16) archsimd.Uint8x16 { return x.ConcatShiftBytesRight(y, hide(shift)) },
				csbr(hide(shift)))
		}
	})

	t.Run("Uint8x32", func(t *testing.T) {
		if !archsimd.X86.AVX2() {
			t.Skip("requires AVX2")
		}
		for _, shift := range []uint8{0, 2, 16, 20, 32, 128} {
			t.Log("shift", shift)
			testUint8x32Binary(t,
				func(x, y archsimd.Uint8x32) archsimd.Uint8x32 { return x.ConcatShiftBytesRightGrouped(y, shift) },
				grouped2(csbr(shift)))
			testUint8x32Binary(t,
				func(x, y archsimd.Uint8x32) archsimd.Uint8x32 { return x.ConcatShiftBytesRightGrouped(y, hide(shift)) },
				grouped2(csbr(hide(shift))))
		}
	})

	t.Run("Uint8x64", func(t *testing.T) {
		if !archsimd.X86.AVX512() {
			t.Skip("requires AVX512")
		}
		for _, shift := range []uint8{0, 2, 16, 20, 32, 128} {
			t.Log("shift", shift)
			testUint8x64Binary(t,
				func(x, y archsimd.Uint8x64) archsimd.Uint8x64 { return x.ConcatShiftBytesRightGrouped(y, shift) },
				grouped2(csbr(shift)))
			testUint8x64Binary(t,
				func(x, y archsimd.Uint8x64) archsimd.Uint8x64 { return x.ConcatShiftBytesRightGrouped(y, hide(shift)) },
				grouped2(csbr(hide(shift))))
		}
	})
}
