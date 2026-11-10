// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestLookupOrZeroInt8x32(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("requires AVX2")
	}
	// Out-of-range indices produce zero lane value.
	x := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116}

	indices := []uint8{0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 255, 128, 12, 13, 14, 15,
		12, 13, 14, 15, 0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 255, 128}

	want := []int8{1, 2, 3, 4, 101, 102, 103, 104, 0, 0, 0, 0, 13, 14, 15, 16,
		13, 14, 15, 16, 1, 2, 3, 4, 101, 102, 103, 104, 0, 0, 0, 0}

	got := make([]int8, len(x))
	archsimd.LoadInt8x32(x).LookupOrZero(archsimd.LoadUint8x32(indices)).StorePart(got)
	checkSlices(t, got, want)
}

func TestLookupOrZeroInt8x64(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("requires AVX512")
	}
	// Out-of-range indices produce zero lane value.
	x := []int8{
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
		41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
		61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
	}

	indices := []uint8{
		0, 1, 16, 17, 32, 33, 48, 49, 64, 65, 127, 128, 255, 13, 14, 15,
		48, 49, 0, 1, 16, 17, 32, 33, 64, 65, 127, 128, 255, 29, 30, 31,
		32, 33, 48, 49, 0, 1, 16, 17, 64, 65, 127, 128, 255, 45, 46, 47,
		16, 17, 32, 33, 48, 49, 0, 1, 64, 65, 127, 128, 255, 61, 62, 63,
	}

	want := []int8{
		1, 2, 21, 22, 41, 42, 61, 62, 0, 0, 0, 0, 0, 14, 15, 16,
		61, 62, 1, 2, 21, 22, 41, 42, 0, 0, 0, 0, 0, 34, 35, 36,
		41, 42, 61, 62, 1, 2, 21, 22, 0, 0, 0, 0, 0, 54, 55, 56,
		21, 22, 41, 42, 61, 62, 1, 2, 0, 0, 0, 0, 0, 74, 75, 76,
	}

	got := make([]int8, len(x))
	archsimd.LoadInt8x64(x).LookupOrZero(archsimd.LoadUint8x64(indices)).StorePart(got)
	checkSlices(t, got, want)
}

func TestRotateAllLeftAMD64(t *testing.T) {
	x := uint8(0x81)
	if y := rotl(x, 1); y != 3 {
		t.Errorf("Expected 3, got 0x%x", y)
	}
	if y := rotl(x, 7); y != 0xc0 {
		t.Errorf("Expected 0xc0, got 0x%x", y)
	}
	if y := rotr(x, 4); y != 0x18 {
		t.Errorf("Expected 0x18, got 0x%x", y)
	}

	for i := uint64(0); i < 65; i++ {
		testUint64x4Unary(t, curry2(archsimd.Uint64x4.RotateAllLeft, i), rotlOfSlice[uint64](i))
		testUint32x8Unary(t, curry2(archsimd.Uint32x8.RotateAllLeft, i), rotlOfSlice[uint32](i))
		//		testUint16x16Unary(t, curry2(archsimd.Uint16x16.RotateAllLeft, i), rotlOfSlice[uint16](i))
		//		testUint8x32Unary(t, curry2(archsimd.Uint8x32.RotateAllLeft, i), rotlOfSlice[uint8](i))
	}

}

func TestRotateAllRightAMD64(t *testing.T) {
	x := uint8(0x81)
	if y := rotr(x, 1); y != 0xc0 {
		t.Errorf("Expected 0xc0, got 0x%x", y)
	}
	if y := rotr(x, 7); y != 3 {
		t.Errorf("Expected 3, got 0x%x", y)
	}
	if y := rotr(x, 4); y != 0x18 {
		t.Errorf("Expected 0x18, got 0x%x", y)
	}

	for i := uint64(0); i < 65; i++ {
		testUint64x4Unary(t, curry2(archsimd.Uint64x4.RotateAllRight, i), rotrOfSlice[uint64](i))
		testUint32x8Unary(t, curry2(archsimd.Uint32x8.RotateAllRight, i), rotrOfSlice[uint32](i))
		//		testUint16x16Unary(t, curry2(archsimd.Uint16x16.RotateAllLeft, i), rotlOfSlice[uint16](i))
		//		testUint8x32Unary(t, curry2(archsimd.Uint8x32.RotateAllLeft, i), rotlOfSlice[uint8](i))
	}
}

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

	hide := hideConst[uint64]

	// ShiftAllLeftConcat
	salc := func(shift uint64) func(x, y int32) int32 {
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

	testInt32x4Binary(t,
		func(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllLeftConcatMod32(y, 128) },
		map2(salc(128)))
	testInt32x4Binary(t,
		func(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllLeftConcatMod32(y, hide(128)) },
		map2(salc(hide(128))))

	// Signed ShiftAllRightConcat
	sarc := func(shift uint64) func(x, y int32) int32 {
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
		func(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllRightConcatMod32(y, 128) },
		map2(sarc(128)))
	testInt32x4Binary(t,
		func(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllRightConcatMod32(y, hide(128)) },
		map2(sarc(hide(128))))

	testInt32x4Binary(t,
		func(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllRightConcatMod32(y, 256) },
		map2(sarc(256)))
	testInt32x4Binary(t,
		func(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllRightConcatMod32(y, hide(256)) },
		map2(sarc(hide(256))))

	// Unsigned ShiftAllRightConcat
	usarc := func(shift uint64) func(x, y uint32) uint32 {
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
		func(x, y archsimd.Uint32x4) archsimd.Uint32x4 { return x.ShiftAllRightConcatMod32(y, 128) },
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
	hide := hideConst[uint64]

	csbr := func(shift uint64) func(x, y []uint8) []uint8 {
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
		for _, shift := range []uint64{0, 2, 16, 20, 32, 128} {
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
		for _, shift := range []uint64{0, 2, 16, 20, 32, 128} {
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
		for _, shift := range []uint64{0, 2, 16, 20, 32, 128} {
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

func TestShiftAllAMD64(t *testing.T) {
	if archsimd.X86.AVX2() {
		// ShiftAllLeft
		testInt16x16ShiftAll(t, archsimd.Int16x16.ShiftAllLeft, shiftAllLeftSlice[int16])
		testInt32x8ShiftAll(t, archsimd.Int32x8.ShiftAllLeft, shiftAllLeftSlice[int32])
		testInt64x4ShiftAll(t, archsimd.Int64x4.ShiftAllLeft, shiftAllLeftSlice[int64])
		testUint16x16ShiftAll(t, archsimd.Uint16x16.ShiftAllLeft, shiftAllLeftSlice[uint16])
		testUint32x8ShiftAll(t, archsimd.Uint32x8.ShiftAllLeft, shiftAllLeftSlice[uint32])
		testUint64x4ShiftAll(t, archsimd.Uint64x4.ShiftAllLeft, shiftAllLeftSlice[uint64])

		// ShiftAllRight signed
		testInt16x16ShiftAll(t, archsimd.Int16x16.ShiftAllRight, shiftAllRightSlice[int16])
		testInt32x8ShiftAll(t, archsimd.Int32x8.ShiftAllRight, shiftAllRightSlice[int32])
		// Int64x4 ShiftAllRight requires AVX-512

		// ShiftAllRight unsigned
		testUint16x16ShiftAll(t, archsimd.Uint16x16.ShiftAllRight, shiftAllRightSlice[uint16])
		testUint32x8ShiftAll(t, archsimd.Uint32x8.ShiftAllRight, shiftAllRightSlice[uint32])
		testUint64x4ShiftAll(t, archsimd.Uint64x4.ShiftAllRight, shiftAllRightSlice[uint64])
	}

	if archsimd.X86.AVX512() {
		// 512-bit vectors (AVX512)
		// ShiftAllLeft
		testInt16x32ShiftAll(t, archsimd.Int16x32.ShiftAllLeft, shiftAllLeftSlice[int16])
		testInt32x16ShiftAll(t, archsimd.Int32x16.ShiftAllLeft, shiftAllLeftSlice[int32])
		testInt64x8ShiftAll(t, archsimd.Int64x8.ShiftAllLeft, shiftAllLeftSlice[int64])
		testUint16x32ShiftAll(t, archsimd.Uint16x32.ShiftAllLeft, shiftAllLeftSlice[uint16])
		testUint32x16ShiftAll(t, archsimd.Uint32x16.ShiftAllLeft, shiftAllLeftSlice[uint32])
		testUint64x8ShiftAll(t, archsimd.Uint64x8.ShiftAllLeft, shiftAllLeftSlice[uint64])

		// ShiftAllRight signed
		testInt16x32ShiftAll(t, archsimd.Int16x32.ShiftAllRight, shiftAllRightSlice[int16])
		testInt32x16ShiftAll(t, archsimd.Int32x16.ShiftAllRight, shiftAllRightSlice[int32])
		testInt64x8ShiftAll(t, archsimd.Int64x8.ShiftAllRight, shiftAllRightSlice[int64])
		// 256-bit Int64x4 ShiftAllRight (requires AVX-512)
		testInt64x4ShiftAll(t, archsimd.Int64x4.ShiftAllRight, shiftAllRightSlice[int64])

		// ShiftAllRight unsigned
		testUint16x32ShiftAll(t, archsimd.Uint16x32.ShiftAllRight, shiftAllRightSlice[uint16])
		testUint32x16ShiftAll(t, archsimd.Uint32x16.ShiftAllRight, shiftAllRightSlice[uint32])
		testUint64x8ShiftAll(t, archsimd.Uint64x8.ShiftAllRight, shiftAllRightSlice[uint64])
	}
}
