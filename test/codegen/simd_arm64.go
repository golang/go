// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check ARM64 SIMD code generation and peephole optimizations.

//go:build goexperiment.simd && arm64

package codegen

import (
	"simd/archsimd"
)

//go:noinline
func forceSpill() {}

func spillAroundCall(a archsimd.Int8x16) archsimd.Int8x16 {
	forceSpill()
	// arm64:`FMOVQ` `FMOVQ`
	return a
}

var (
	sinkU8  archsimd.Uint8x16
	sinkI8  archsimd.Int8x16
	sinkU32 archsimd.Uint32x4
	sinkU64 archsimd.Uint64x2
	sinkF32 archsimd.Float32x4
	sinkF64 archsimd.Float64x2
)

func broadcastConstImmFold(k int) {
	switch k {
	case 0:
		// arm64:`VMOVI [$]0,` -`VDUP`
		sinkU8 = archsimd.BroadcastUint8x16(0)
	case 1:
		// arm64:`VMOVI [$]1,` -`VDUP`
		sinkU8 = archsimd.BroadcastUint8x16(1)
	case 127:
		// arm64:`VMOVI [$]127,` -`VDUP`
		sinkI8 = archsimd.BroadcastInt8x16(127)
	case 128:
		// arm64:`VMOVI [$]128,` -`VDUP`
		sinkU8 = archsimd.BroadcastUint8x16(128)
	case -128:
		// arm64:`VMOVI [$]128,` -`VDUP`
		sinkI8 = archsimd.BroadcastInt8x16(-128)
	case 255:
		// arm64:`VMOVI [$]255,` -`VDUP`
		sinkU8 = archsimd.BroadcastUint8x16(255)
	case -1:
		// arm64:`VMOVI [$]255,` -`VDUP`
		sinkI8 = archsimd.BroadcastInt8x16(-1)
	case -2:
		// arm64:`VMOVI [$]254,` -`VDUP`
		sinkI8 = archsimd.BroadcastInt8x16(-2)
	default:
		// arm64:`VMOV R0, V\d+.B\[0\]` `VDUP`
		sinkI8 = archsimd.BroadcastInt8x16(int8(k))
	}
}

func shiftAllImmFold(k int) {
	switch k {
	case 100:
		// arm64:`VMOVI [$]100,` `VSSHL` -`VDUP`
		sinkI8 = sinkI8.ShiftAllLeft(100)
		// arm64:`VMOVI [$]156,` `VUSHL` -`VDUP`
		sinkU8 = sinkU8.ShiftAllRight(100)
	}
}

func setHiUint32(x, lo archsimd.Uint32x4) {
	// arm64:`VMOV V1.D\[0\], V0.D\[1\]`
	sinkU32 = x.SetHi(lo)
}

func setHiFloat64(x, lo archsimd.Float64x2) {
	// arm64:`VMOV V1.D\[0\], V0.D\[1\]`
	sinkF64 = x.SetHi(lo)
}

func getHiFloat32(x archsimd.Float32x4) {
	// arm64:`VDUP V0.D\[1\],`
	sinkF32 = x.GetHi()
}

func getHiFloat64(x archsimd.Float64x2) {
	// arm64:`VDUP V0.D\[1\],`
	sinkF64 = x.GetHi()
}

func foldGetHiSetHiShifts(x archsimd.Uint32x4) archsimd.Uint16x8 {
	shrN := x.ShiftRightNarrowConst(16)         // arm64: `VSHRN [$]16, V0.S4, V[0-9]+.H4`
	trunc := x.ShiftRightNarrowConst(0)         // arm64: `VXTN V0.S4, V[0-9]+.H4` -`VSHRN`
	shlLo := x.ShiftLeftWidenLoConst(1)         // arm64: `VUSHLL [$]1, V0.S2, V[0-9]+.D2`
	shlHi := x.GetHi().ShiftLeftWidenLoConst(1) // arm64: `VUSHLL2 [$]1, V0.S4, V[0-9]+.D2` -`VDUP`
	sum := shrN.Add(trunc)
	combined := sum.SetHi(x.ShiftRightNarrowConst(15)) // arm64: `VSHRN2 [$]15, V0.S4, V[0-9]+.H8` -`VMOV.*D\[`
	sinkU64 = shlLo.Sub(shlHi)
	return combined
}

func foldGetHiSetHiMuls(a, b archsimd.Uint16x8) archsimd.Uint16x8 {
	wLo := a.MulWidenLo(b)                    // arm64: `VUMULL V0.H4, V1.H4, V[0-9].S4`
	wHi := a.GetHi().MulWidenLo(b.GetHi())    // arm64: `VUMULL2 V1.H8, V0.H8, V[0-9].S4` -`VDUP`
	wHiRight := wHi.ShiftRightNarrowConst(16) // arm64: -`.*`
	wLoRight := wLo.ShiftRightNarrowConst(16) // arm64: `VSHRN [$]16, V[0-9]+.S4, V0.H4`
	return wLoRight.SetHi(wHiRight)           // arm64: `VSHRN2 [$]16, V[0-9]+.S4, V0.H8` -`VMOV.*D\[`
}

func carrylessMultiplies(x, y archsimd.Uint64x2) archsimd.Uint64x2 {
	lo := x.CarrylessMultiplyEven(y) // arm64:`VPMULL V` -`VPMULL2`
	hi := x.CarrylessMultiplyOdd(y)  // arm64:`VPMULL2 V` -`VPMULL `
	return lo.Xor(hi)
}

func mergeWithNotMask(x, y archsimd.Int8x16, mask archsimd.Mask8x16, f1, f2 archsimd.Float32x4) {
	// arm64:`VBIF` -`VBIT` -`VNOT`
	sinkI8 = x.IfElse(mask.Not(), y)
	// arm64: `VFCMEQ`
	eq := f1.Equal(f2)
	// The next line `ne` should be CSEd with `eq` above
	ne := f1.NotEqual(f2)    // arm64: -`.*`
	fne := f1.IfElse(eq, f2) // arm64:`VBIT`
	feq := f1.IfElse(ne, f2) // arm64:`VBIF`
	sinkF32 = fne.Add(feq)
}
