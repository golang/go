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
	sinkU16 archsimd.Uint16x8
	sinkU32 archsimd.Uint32x4
	sinkU64 archsimd.Uint64x2
	sinkF32 archsimd.Float32x4
	sinkF64 archsimd.Float64x2

	sinkM8  archsimd.Mask8x16
	sinkM16 archsimd.Mask16x8
	sinkM32 archsimd.Mask32x4
	sinkM64 archsimd.Mask64x2

	sinkScalarI8  int8
	sinkScalarI8b int8
	sinkScalarI64 int64
	sinkScalarU64 uint64
	sinkI64a      int64
	sinkI64b      int64
	sinkI64c      int64
	sinkI64d      int64
	sinkU64b      uint64
	sinkU64c      uint64
	sinkU64d      uint64
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
		// arm64:`VDUP R[0-9]+, V[0-9]+\.B16` -`VMOV`
		sinkI8 = archsimd.BroadcastInt8x16(int8(k))
	}
}

func broadcastConstImmFoldWide(k int) {
	switch k {
	case 0:
		// arm64:`VMOVI [$]66, V\d+\.H8` -`VDUP`
		sinkU16 = archsimd.BroadcastUint16x8(0x42)
	case 1:
		// arm64:`VMOVI [$]16896, V\d+\.H8` -`VDUP`
		sinkU16 = archsimd.BroadcastUint16x8(0x4200)
	case 2:
		// arm64:`VMOVI [$]65280, V\d+\.H8` -`VDUP`
		sinkU16 = archsimd.BroadcastUint16x8(0xFF00)
	case 3:
		// arm64:`VMOVI [$]66, V\d+\.S4` -`VDUP`
		sinkU32 = archsimd.BroadcastUint32x4(0x42)
	case 4:
		// arm64:`VMOVI [$]16896, V\d+\.S4` -`VDUP`
		sinkU32 = archsimd.BroadcastUint32x4(0x4200)
	case 5:
		// arm64:`VMOVI [$]4325376, V\d+\.S4` -`VDUP`
		sinkU32 = archsimd.BroadcastUint32x4(0x420000)
	case 6:
		// arm64:`VMOVI [$]1107296256, V\d+\.S4` -`VDUP`
		sinkU32 = archsimd.BroadcastUint32x4(0x42000000)
	case 7:
		// arm64:`VMOVI [$]17151, V\d+\.S4` -`VDUP`
		sinkU32 = archsimd.BroadcastUint32x4(0x42FF)
	case 8:
		// arm64:`VMOVI [$]196607, V\d+\.S4` -`VDUP`
		sinkU32 = archsimd.BroadcastUint32x4(0x2FFFF)
	case 9:
		// arm64:`VMOVI [$]0, V\d+\.S4` -`VDUP`
		sinkU32 = archsimd.BroadcastUint32x4(0)
	case 10:
		// arm64:`VMOVI [$]-71777214294589696, V\d+\.D2` -`VDUP`
		sinkU64 = archsimd.BroadcastUint64x2(0xFF00FF00FF00FF00)
	case 11:
		// arm64:`VMOVI [$]-1, V\d+\.D2` -`VDUP`
		sinkU64 = archsimd.BroadcastUint64x2(0xFFFFFFFFFFFFFFFF)
	case 12:
		// arm64:`VMOVI [$]0, V\d+\.D2` -`VDUP`
		sinkU64 = archsimd.BroadcastUint64x2(0)
	case 13:
		// arm64:`VMOVI [$]4278190080, V\d+\.S4` -`VDUP`
		sinkU32 = archsimd.BroadcastUint32x4(0xFF000000)
	case 14:
		// arm64:`VDUP R[0-9]+, V[0-9]+\.H8` -`VMOV`
		sinkU16 = archsimd.BroadcastUint16x8(0x4201)
	default:
		// arm64:`VDUP R[0-9]+, V[0-9]+\.S4` -`VMOV`
		sinkU32 = archsimd.BroadcastUint32x4(0x7F7F7F7F)
	}
}

func shiftAllImmFold(k int) {
	switch k {
	case 100:
		// arm64:`VMOVI [$]0,` -`VSSHL` -`VDUP`
		sinkI8 = sinkI8.ShiftAllLeft(100)
		// arm64:`VUSHR [$]8,` -`VUSHL` -`VDUP`
		sinkU8 = sinkU8.ShiftAllRight(100)
	}
}

// Shifts by a constant use the immediate forms.
func shiftAllLeftConst8(x archsimd.Int8x16) archsimd.Int8x16 {
	// arm64:`VSHL [$]3, V\d+.B16` -`VSSHL`
	return x.ShiftAllLeft(3)
}

func shiftAllRightConst16(x archsimd.Int16x8) archsimd.Int16x8 {
	// arm64:`VSSHR [$]7, V\d+.H8` -`VSSHL`
	return x.ShiftAllRight(7)
}

func shiftAllRightConst32(x archsimd.Int32x4) archsimd.Int32x4 {
	// arm64:`VSSHR [$]24, V\d+.S4` -`VSSHL`
	return x.ShiftAllRight(24)
}

func shiftAllRightConstHuge32(x archsimd.Int32x4) archsimd.Int32x4 {
	// arm64:`VSSHR [$]32, V\d+.S4` -`VSSHL`
	return x.ShiftAllRight(1000)
}

func shiftAllRightConstWidth64(x archsimd.Uint64x2) archsimd.Uint64x2 {
	// arm64:`VUSHR [$]64, V\d+.D2` -`VUSHL`
	return x.ShiftAllRight(64)
}

func shiftAllLeftConst64(x archsimd.Uint64x2) archsimd.Uint64x2 {
	// arm64:`VSHL [$]63, V\d+.D2` -`VUSHL`
	return x.ShiftAllLeft(63)
}

// Truncate-of-immediate-right-shift fuses into the single
// shift-right-and-narrow VSHRN for amounts in [1, elemBits/2].
func shiftAllRightConstTrunc16(x archsimd.Uint16x8) archsimd.Uint8x16 {
	// arm64:`VSHRN [$]4, V\d+\.H8, V\d+\.B8` -`VXTN` -`VUSHR`
	return x.ShiftAllRight(4).TruncToUint8()
}

func shiftAllRightConstTruncSigned16(x archsimd.Int16x8) archsimd.Int8x16 {
	// arm64:`VSHRN [$]4, V\d+\.H8, V\d+\.B8` -`VXTN` -`VSSHR`
	return x.ShiftAllRight(4).TruncToInt8()
}

func shiftAllRightConstTrunc32(x archsimd.Uint32x4) archsimd.Uint16x8 {
	// arm64:`VSHRN [$]9, V\d+\.S4, V\d+\.H4` -`VXTN` -`VUSHR`
	return x.ShiftAllRight(9).TruncToUint16()
}

func shiftAllRightConstTrunc64(x archsimd.Uint64x2) archsimd.Uint32x4 {
	// arm64:`VSHRN [$]31, V\d+\.D2, V\d+\.S2` -`VXTN` -`VUSHR`
	return x.ShiftAllRight(31).TruncToUint32()
}

// Above half the lane width the amount is not encodable as VSHRN, so the
// immediate shift and the narrow stay separate instructions.
func shiftAllRightConstTruncNoFusion(x archsimd.Uint16x8) archsimd.Uint8x16 {
	// arm64:`VUSHR [$]9, V\d+\.H8,` `VXTN V\d+\.H8,` -`VSHRN`
	return x.ShiftAllRight(9).TruncToUint8()
}

func setHiUint32(x, lo archsimd.Uint32x4) {
	// arm64:`VMOV V1.D\[0\], V0.D\[1\]`
	sinkU32 = loToHiUint32Vec(x, lo)
}

func setHiFloat64(x, lo archsimd.Float64x2) {
	// arm64:`VMOV V1.D\[0\], V0.D\[1\]`
	sinkF64 = x.SetElem(1, lo.Elem(0))
}

func getHiFloat32(x archsimd.Float32x4) {
	// arm64:`VDUP V0.D\[1\],`
	sinkF32 = x.HiToLo()
}

func getHiFloat64(x archsimd.Float64x2) {
	// arm64:`VDUP V0.D\[1\],`
	sinkF64 = x.HiToLo()
}

func getElemExtFold(x8 archsimd.Int8x16, u8 archsimd.Uint8x16, x16 archsimd.Int16x8, x32 archsimd.Int32x4, u16 archsimd.Uint16x8, u32 archsimd.Uint32x4) {
	// arm64:`VSMOV V0.B\[3\],` -`MOVB`
	sinkI64a = int64(x8.Elem(3))
	// arm64:`VSMOV V0.B\[5\],`
	sinkScalarI8 = x8.Elem(5)
	// arm64:`VSMOV V[0-9]+.H\[5\],` -`MOVH`
	sinkI64b = int64(x16.Elem(5))
	// arm64:`VSMOV V[0-9]+.S\[2\],` -`MOVW`
	sinkI64c = int64(x32.Elem(2))
	// arm64:`VMOV V1.B\[7\],` -`MOVBU`
	sinkScalarU64 = uint64(u8.Elem(7))
	// arm64:`VMOV V[0-9]+.H\[3\],` -`MOVHU`
	sinkU64b = uint64(u16.Elem(3))
	// arm64:`VMOV V[0-9]+.S\[1\],` -`MOVWU`
	sinkU64c = uint64(u32.Elem(1))
	// arm64:`VSMOV V1.B\[7\],` -`MOVB`
	sinkI64d = int64(int8(u8.Elem(7)))
	// arm64:`VMOV V0.B\[6\],` -`VSMOV`
	sinkU64d = uint64(uint8(x8.Elem(6)))
	// arm64:`VSMOV V0.B\[4\],`
	e := x8.Elem(4)
	sinkScalarI64 = int64(e)
	sinkScalarI8b = e
}

// does x.And(y).Equal(zero) peephole to the expected cmtst?
func cmtst(x8, y8 archsimd.Uint8x16, x16, y16 archsimd.Uint16x8,
	x32, y32 archsimd.Uint32x4, x64, y64 archsimd.Uint64x2) {
	var z8 archsimd.Uint8x16
	var z16 archsimd.Uint16x8
	var z32 archsimd.Uint32x4
	var z64 archsimd.Uint64x2
	sinkM8 = x8.And(y8).Equal(z8).Not()     // arm64: `VCMTST V[0-9]+.B16, V[0-9]+.B16, V[0-9]+.B16`
	sinkM16 = x16.And(y16).Equal(z16).Not() // arm64: `VCMTST V[0-9]+.H8, V[0-9]+.H8, V[0-9]+.H8`
	sinkM32 = x32.And(y32).Equal(z32).Not() // arm64: `VCMTST V[0-9]+.S4, V[0-9]+.S4, V[0-9]+.S4`
	sinkM64 = x64.And(y64).Equal(z64).Not() // arm64: `VCMTST V[0-9]+.D2, V[0-9]+.D2, V[0-9]+.D2`
}

// does x.Or(y.Not()) peephole to the expected orn?
func orNot(x8, y8 archsimd.Uint8x16, x32, y32 archsimd.Uint32x4, m, n archsimd.Mask64x2) {
	sinkU8 = x8.Or(y8.Not())    // arm64: `VORN V[0-9]+.B16, V[0-9]+.B16, V[0-9]+.B16` -`VNOT` -`VORR`
	sinkU32 = y32.Not().Or(x32) // arm64: `VORN V[0-9]+.B16, V[0-9]+.B16, V[0-9]+.B16` -`VNOT` -`VORR`
	sinkM64 = m.Or(n.Not())     // arm64: `VORN V[0-9]+.B16, V[0-9]+.B16, V[0-9]+.B16` -`VNOT` -`VORR`
}

func foldGetHiSetHiMuls(a, b archsimd.Uint16x8) archsimd.Uint16x8 {
	wLo := a.MulWiden(b)                       // arm64: `VUMULL V0.H4, V1.H4, V[0-9].S4`
	wHi := a.HiToLo().MulWiden(b.HiToLo())     // arm64: `VUMULL2 V1.H8, V0.H8, V[0-9].S4` -`VDUP`
	narrowLo := wLo.TruncToUint16()            // arm64: `VXTN V[0-9]+.S4, V0.H4`
	narrowHi := wHi.TruncToUint16()            // folded into next line
	return loToHiUint16Vec(narrowLo, narrowHi) // arm64: `VXTN2 V[0-9]+.S4, V0.H8`
}

func carrylessMultiplies(x, y archsimd.Uint64x2) archsimd.Uint64x2 {
	lo := x.CarrylessMulEven(y)                   // arm64:`VPMULL V` -`VPMULL2`
	hi := x.HiToLo().CarrylessMulEven(y.HiToLo()) // arm64:`VPMULL2 V` -`VPMULL `
	return lo.Xor(hi)
}

func mergeWithNotMask(x, y archsimd.Int8x16, mask archsimd.Mask8x16, f1, f2 archsimd.Float32x4) {
	// arm64:`VBIT` -`VBIF` -`VNOT`
	sinkI8 = x.IfElse(mask.Not(), y)
	// arm64: `VFCMEQ`
	eq := f1.Equal(f2)
	// ne is CSEd with eq above, so its line has only a DWARF anchor NOP.
	ne := f1.NotEqual(f2)    // arm64: -`V[A-Z]`
	feq := f1.IfElse(eq, f2) // arm64:`VBIF`
	fne := f1.IfElse(ne, f2) // arm64:`VBIT`
	sinkF32 = fne.Add(feq)
}

// loToHiUint32Vec returns a vector with the lower 64 bits of x preserved and
// the upper 64 bits replaced with the lower 64 bits of lo.
// It routes through Float64x2 to stay in the FP/SIMD register file,
// avoiding a round-trip through a GP register.
func loToHiUint32Vec(x, lo archsimd.Uint32x4) archsimd.Uint32x4 {
	return x.ReshapeToUint64s().BitsToFloat64().SetElem(1, lo.ReshapeToUint64s().BitsToFloat64().Elem(0)).ToBits().ReshapeToUint32s()
}

// loToHiUint16Vec returns a vector with the lower 64 bits of x preserved and
// the upper 64 bits replaced with the lower 64 bits of lo.
// It routes through Float64x2 to stay in the FP/SIMD register file,
// avoiding a round-trip through a GP register.
func loToHiUint16Vec(x, lo archsimd.Uint16x8) archsimd.Uint16x8 {
	return x.ReshapeToUint64s().BitsToFloat64().SetElem(1, lo.ReshapeToUint64s().BitsToFloat64().Elem(0)).ToBits().ReshapeToUint16s()
}

// --- SVE predication peepholes ---
//
// IfElse over an unpredicated operation folds into that operation's
// merging-predicated form. Merging keeps the destination, and an SVE predicated
// instruction is destructive, so an "else" operand that is already one of the
// sources folds into a bare predicated instruction; any other one first needs a
// merging MOVPRFX to put it in the destination.

func sveIfElseFoldsFirstOperand(x, y archsimd.Int8s, m archsimd.Mask8s) archsimd.Int8s {
	// arm64:`ZADD.*P[0-9]+\.M` -`ZSEL` -`ZMOVPRFX`
	return x.Add(y).IfElse(m, x)
}

func sveIfElseFoldsSecondOperand(x, y archsimd.Int8s, m archsimd.Mask8s) archsimd.Int8s {
	// Commutative, so the mirrored form folds too.
	// arm64:`ZADD.*P[0-9]+\.M` -`ZSEL` -`ZMOVPRFX`
	return x.Add(y).IfElse(m, y)
}

func sveIfElseArbitraryElse(x, y, z archsimd.Int8s, m archsimd.Mask8s) archsimd.Int8s {
	// The else operand is neither source, so a merging MOVPRFX puts it in the
	// destination and the destructive add merges over it.
	// arm64:`ZMOVPRFX.*P[0-9]+\.M` `ZADD.*P[0-9]+\.M` -`ZSEL`
	return x.Add(y).IfElse(m, z)
}

func sveMaskedFoldsIntoMerging(x, y archsimd.Int8s, m archsimd.Mask8s) archsimd.Int8s {
	// Masked is a select against zero. ADD has no zeroing-predicated form, so it
	// folds into the merging one with the zero vector as the else operand.
	// arm64:`ZMOVPRFX.*P[0-9]+\.M` `ZADD.*P[0-9]+\.M` -`ZSEL`
	return x.Add(y).Masked(m)
}

//go:noinline
func sinkInt8s(archsimd.Int8s) {}

// A non-commutative operation is more restricted. Its destructive operand is
// fixed, so only an "else" operand that is already that one folds, and there is
// no prefixed form to place any other.

func sveIfElseFoldsSubMinuend(x, y archsimd.Int8s, m archsimd.Mask8s) archsimd.Int8s {
	// arm64:`ZSUB.*P[0-9]+\.M` -`ZSEL`
	return x.Sub(y).IfElse(m, x)
}

func sveIfElseKeepsSelectSubSubtrahend(x, y archsimd.Int8s, m archsimd.Mask8s) archsimd.Int8s {
	// Folding here would compute y-x.
	// arm64:`ZSEL` -`ZSUB.*P[0-9]+\.M`
	return x.Sub(y).IfElse(m, y)
}

func sveIfElseKeepsSelectSubArbitrary(x, y, z archsimd.Int8s, m archsimd.Mask8s) archsimd.Int8s {
	// arm64:`ZSEL` -`ZMOVPRFX` -`ZSUB.*P[0-9]+\.M`
	return x.Sub(y).IfElse(m, z)
}

// An operation that only comes predicated reaches its unpredicated API through
// an all-true predicate. A select over it replaces that predicate instead of
// adding an instruction.

func sveAbsSynthesizesAllTrue(x archsimd.Int8s) archsimd.Uint8s {
	// arm64:`PWHILELT` `ZABS.*P[0-9]+\.M`
	return x.Abs()
}

func sveAbsIfElseFoldsToMerging(x archsimd.Int8s, z archsimd.Uint8s, m archsimd.Mask8s) archsimd.Uint8s {
	// ABS names its destination apart from its source, so the else operand is an
	// operand of the instruction: no select, no MOVPRFX, and the all-true
	// predicate is gone because the select's mask took its place.
	// arm64:`ZABS.*P[0-9]+\.M` -`ZSEL` -`PWHILELT` -`ZMOVPRFX`
	return x.Abs().IfElse(m, z)
}

func sveAbsMaskedFoldsToMerging(x archsimd.Int8s, m archsimd.Mask8s) archsimd.Uint8s {
	// Masked folds through the same rule, with the zero vector as the else
	// operand. Zeroing predication would save the ZDUP, but ABS only has a
	// zeroing encoding from Armv9.6-A on.
	// arm64:`ZABS.*P[0-9]+\.M` -`ZSEL` -`PWHILELT`
	return x.Abs().Masked(m)
}

func sveOnesCountSynthesizesAllTrue(x archsimd.Int8s) archsimd.Int8s {
	// arm64:`PWHILELT` `ZCNT.*P[0-9]+\.M`
	return x.OnesCount()
}

func sveOnesCountIfElseFoldsToMerging(x, z archsimd.Int8s, m archsimd.Mask8s) archsimd.Int8s {
	// CNT is predicated-only, like ABS, and names its destination apart from
	// its source, so the else operand is an operand of the instruction.
	// arm64:`ZCNT.*P[0-9]+\.M` -`ZSEL` -`PWHILELT` -`ZMOVPRFX`
	return x.OnesCount().IfElse(m, z)
}

func sveOnesCountMaskedFoldsToMerging(x archsimd.Int8s, m archsimd.Mask8s) archsimd.Int8s {
	// Masked folds through the same rule, with the zero vector as the else
	// operand, as it does for ABS.
	// arm64:`ZCNT.*P[0-9]+\.M` -`ZSEL` -`PWHILELT`
	return x.OnesCount().Masked(m)
}

func sveCeilIfElseFoldsToMerging(x, z archsimd.Float32s, m archsimd.Mask32s) archsimd.Float32s {
	// FRINTP is predicated-only, like ABS, and folds the same way.
	// arm64:`ZFRINTP.*P[0-9]+\.M` -`ZSEL` -`PWHILELT` -`ZMOVPRFX`
	return x.Ceil().IfElse(m, z)
}

func sveNotSynthesizesAllTrue(x archsimd.Uint8s) archsimd.Uint8s {
	// arm64:`PWHILELT` `ZNOT.*P[0-9]+\.M`
	return x.Not()
}

func sveNotIfElseFoldsToMerging(x, z archsimd.Uint8s, m archsimd.Mask8s) archsimd.Uint8s {
	// NOT is predicated-only, like ABS, and folds the same way.
	// arm64:`ZNOT.*P[0-9]+\.M` -`ZSEL` -`PWHILELT` -`ZMOVPRFX`
	return x.Not().IfElse(m, z)
}

func sveNotMaskedFoldsToMerging(x archsimd.Uint8s, m archsimd.Mask8s) archsimd.Uint8s {
	// arm64:`ZNOT.*P[0-9]+\.M` -`ZSEL` -`PWHILELT`
	return x.Not().Masked(m)
}

func sveNotIfElseFoldsToMerging16(x, z archsimd.Int16s, m archsimd.Mask16s) archsimd.Int16s {
	// arm64:`ZNOT\s+Z[0-9]+\.H, P[0-9]+\.M` -`ZSEL`
	return x.Not().IfElse(m, z)
}

func sveNotIfElseFoldsToMerging32(x, z archsimd.Uint32s, m archsimd.Mask32s) archsimd.Uint32s {
	// arm64:`ZNOT\s+Z[0-9]+\.S, P[0-9]+\.M` -`ZSEL`
	return x.Not().IfElse(m, z)
}

func sveNotIfElseFoldsToMerging64(x, z archsimd.Int64s, m archsimd.Mask64s) archsimd.Int64s {
	// arm64:`ZNOT\s+Z[0-9]+\.D, P[0-9]+\.M` -`ZSEL`
	return x.Not().IfElse(m, z)
}

func sveIfElseFloat(x, y archsimd.Float64s, m archsimd.Mask64s) archsimd.Float64s {
	// arm64:`ZFADD.*P[0-9]+\.M` -`ZSEL`
	return x.Add(y).IfElse(m, x)
}

// Less is Greater with the operands swapped, so it is one compare with the
// sources reversed.
func sveLess(x, y archsimd.Int32s) archsimd.Mask32s {
	// arm64:`ZCMPGT\s+Z0\.S, Z1\.S, P[0-9]+\.Z, P[0-9]+\.S`
	return x.Less(y)
}

// LessEqual is GreaterEqual with the operands swapped.
func sveLessEqual(x, y archsimd.Float64s) archsimd.Mask64s {
	// arm64:`ZFCMGE\s+Z0\.D, Z1\.D, P[0-9]+\.Z, P[0-9]+\.D`
	return x.LessEqual(y)
}

// A reduction leaves its result in the SIMD&FP scalar register that a float
// is returned in.
func sveReduceSum(x archsimd.Float32s) float32 {
	// arm64:`ZFADDVS\s+Z[0-9]+\.S, P[0-9]+, V[0-9]+` -`FMOV`
	return x.ReduceSum()
}

func sveReduceMax(x archsimd.Float64s) float64 {
	// arm64:`ZFMAXVD\s+Z[0-9]+\.D, P[0-9]+, V[0-9]+` -`FMOV`
	return x.ReduceMax()
}

func sveReduceMin(x archsimd.Float32s) float32 {
	// arm64:`ZFMINVS\s+Z[0-9]+\.S, P[0-9]+, V[0-9]+` -`FMOV`
	return x.ReduceMin()
}

// An integer reduction moves its result from the SIMD&FP register to the
// general-purpose register that an integer is returned in.
func sveReduceSumInt16(x archsimd.Int16s) int16 {
	// arm64:`ZUADDVD\s+Z[0-9]+\.H, P[0-9]+, V[0-9]+` `FMOVD\s+F[0-9]+, R[0-9]+`
	return x.ReduceSum()
}

// The moved result is not sign-extended, so widening a signed one extends it.
func sveReduceSumInt8Wide(x archsimd.Int8s) int64 {
	// arm64:`ZUADDVD\s+Z[0-9]+\.B, P[0-9]+, V[0-9]+` `FMOVD\s+F[0-9]+, R[0-9]+` `MOVB\s+R[0-9]+, R[0-9]+`
	return int64(x.ReduceSum())
}

// SMAXV and UMAXV leave a result one element wide, which is moved the same way.
func sveReduceMaxUint32(x archsimd.Uint32s) uint32 {
	// arm64:`ZUMAXVS\s+Z[0-9]+\.S, P[0-9]+, V[0-9]+` `FMOVD\s+F[0-9]+, R[0-9]+`
	return x.ReduceMax()
}

// SMINV and UMINV are moved the same way.
func sveReduceMinInt8(x archsimd.Int8s) int8 {
	// arm64:`ZSMINVB\s+Z[0-9]+\.B, P[0-9]+, V[0-9]+` `FMOVD\s+F[0-9]+, R[0-9]+`
	return x.ReduceMin()
}

// ShiftAllLeft by a constant below the element width is LSL (immediate), and
// by a larger constant the zero vector. A variable amount is broadcast to
// 64-bit lanes and shifts by wide elements, which gives zero for an amount of
// the element width or more, so the amount needs no clamping.

func sveShiftAllLeftConst(x archsimd.Int32s) archsimd.Int32s {
	// arm64:`ZLSL\s+\$23, Z[0-9]+\.S, Z[0-9]+\.S` -`ZDUP`
	return x.ShiftAllLeft(23)
}

func sveShiftAllLeftConst64(x archsimd.Int64s) archsimd.Int64s {
	// arm64:`ZLSL\s+\$63, Z[0-9]+\.D, Z[0-9]+\.D` -`ZDUP` -`PWHILELT`
	return x.ShiftAllLeft(63)
}

func sveShiftAllLeftConstTooBig(x archsimd.Int32s) archsimd.Int32s {
	// arm64:`ZDUP\s+\$0,` -`ZLSL`
	return x.ShiftAllLeft(32)
}

func sveShiftAllLeftConstHuge(x archsimd.Uint64s) archsimd.Uint64s {
	// arm64:`ZDUP\s+\$0,` -`ZLSL` -`PWHILELT`
	return x.ShiftAllLeft(1 << 40)
}

func sveShiftAllLeftConstTooBigIfElse(x archsimd.Int8s, m archsimd.Mask8s) archsimd.Int8s {
	// arm64:`ZCPY\s+\$0, P[0-9]+\.M, Z[0-9]+\.B` -`ZLSL` -`ZSEL`
	return x.ShiftAllLeft(8).IfElse(m, x)
}

func sveShiftAllLeftConstIfElse(x archsimd.Int16s, m archsimd.Mask16s) archsimd.Int16s {
	// arm64:`ZLSL\s+\$3, Z[0-9]+\.H, P[0-9]+\.M, Z[0-9]+\.H` -`ZSEL`
	return x.ShiftAllLeft(3).IfElse(m, x)
}

func sveShiftAllLeftVar(x archsimd.Int16s, s uint64) archsimd.Int16s {
	// arm64:`ZDUP\s+R[0-9]+, Z[0-9]+\.D` `ZLSL\s+Z[0-9]+\.D, Z[0-9]+\.H, Z[0-9]+\.H` -`CSEL`
	return x.ShiftAllLeft(s)
}

func sveShiftAllLeftVar64(x archsimd.Uint64s, s uint64) archsimd.Uint64s {
	// LSL has no wide form for 64-bit elements; LSL (vectors) is predicated.
	// arm64:`ZLSL\s+Z[0-9]+\.D, Z[0-9]+\.D, P[0-9]+\.M, Z[0-9]+\.D`
	return x.ShiftAllLeft(s)
}

// ShiftAllRight by a constant from 1 to the element width is ASR or LSR
// (immediate). By a larger constant, LSR gives the zero vector and ASR the
// sign, which ASR by the element width already gives. A variable amount
// shifts by wide elements, as for ShiftAllLeft.

func sveShiftAllRightConst(x archsimd.Int32s) archsimd.Int32s {
	// arm64:`ZASR\s+\$23, Z[0-9]+\.S, Z[0-9]+\.S` -`ZDUP`
	return x.ShiftAllRight(23)
}

func sveShiftAllRightConstWidth(x archsimd.Uint16s) archsimd.Uint16s {
	// arm64:`ZLSR\s+\$16, Z[0-9]+\.H, Z[0-9]+\.H` -`ZDUP`
	return x.ShiftAllRight(16)
}

func sveShiftAllRightConst64(x archsimd.Int64s) archsimd.Int64s {
	// arm64:`ZASR\s+\$64, Z[0-9]+\.D, Z[0-9]+\.D` -`ZDUP` -`PWHILELT`
	return x.ShiftAllRight(64)
}

func sveShiftAllRightConstTooBig(x archsimd.Int16s) archsimd.Int16s {
	// arm64:`ZASR\s+\$16, Z[0-9]+\.H, Z[0-9]+\.H` -`ZDUP`
	return x.ShiftAllRight(100)
}

func sveShiftAllRightConstHuge(x archsimd.Int64s) archsimd.Int64s {
	// arm64:`ZASR\s+\$64, Z[0-9]+\.D, Z[0-9]+\.D` -`ZDUP` -`PWHILELT`
	return x.ShiftAllRight(1 << 40)
}

func sveShiftAllRightConstTooBigUnsigned(x archsimd.Uint32s) archsimd.Uint32s {
	// arm64:`ZDUP\s+\$0,` -`ZLSR`
	return x.ShiftAllRight(1 << 40)
}

func sveShiftAllRightConstIfElse(x archsimd.Int16s, m archsimd.Mask16s) archsimd.Int16s {
	// arm64:`ZASR\s+\$3, Z[0-9]+\.H, P[0-9]+\.M, Z[0-9]+\.H` -`ZSEL`
	return x.ShiftAllRight(3).IfElse(m, x)
}

func sveShiftAllRightConstTooBigIfElse(x archsimd.Int8s, m archsimd.Mask8s) archsimd.Int8s {
	// arm64:`ZASR\s+\$8, Z[0-9]+\.B, P[0-9]+\.M, Z[0-9]+\.B` -`ZSEL`
	return x.ShiftAllRight(9).IfElse(m, x)
}

func sveShiftAllRightVar(x archsimd.Uint16s, s uint64) archsimd.Uint16s {
	// arm64:`ZDUP\s+R[0-9]+, Z[0-9]+\.D` `ZLSR\s+Z[0-9]+\.D, Z[0-9]+\.H, Z[0-9]+\.H` -`CSEL`
	return x.ShiftAllRight(s)
}

func sveShiftAllRightVar64(x archsimd.Int64s, s uint64) archsimd.Int64s {
	// arm64:`ZASR\s+Z[0-9]+\.D, Z[0-9]+\.D, P[0-9]+\.M, Z[0-9]+\.D`
	return x.ShiftAllRight(s)
}

func sveShiftAllRightVarIfElse(x archsimd.Uint16s, s uint64, m archsimd.Mask16s) archsimd.Uint16s {
	// arm64:`ZLSR\s+Z[0-9]+\.D, Z[0-9]+\.H, P[0-9]+\.M, Z[0-9]+\.H` -`ZSEL`
	return x.ShiftAllRight(s).IfElse(m, x)
}

func sveShiftAllRightVar64IfElse(x archsimd.Int64s, s uint64, m archsimd.Mask64s) archsimd.Int64s {
	// arm64:`ZASR\s+Z[0-9]+\.D, Z[0-9]+\.D, P[0-9]+\.M, Z[0-9]+\.D` -`ZSEL` -`PWHILELT`
	return x.ShiftAllRight(s).IfElse(m, x)
}

// ShiftLeft by vector is LSL (vectors), which is predicated-only, so it runs
// under an all-true predicate. It yields 0 for an amount of the element width
// or more, so nothing clamps the amounts.

func sveShiftLeftVec(x archsimd.Int8s, s archsimd.Uint8s) archsimd.Int8s {
	// arm64:`PWHILELT` `ZLSL\s+Z[0-9]+\.B, Z[0-9]+\.B, P[0-9]+\.M, Z[0-9]+\.B` -`ZSEL` -`ZUMIN`
	return x.ShiftLeft(s)
}

func sveShiftLeftVec64(x archsimd.Uint64s, s archsimd.Uint64s) archsimd.Uint64s {
	// arm64:`PWHILELT` `ZLSL\s+Z[0-9]+\.D, Z[0-9]+\.D, P[0-9]+\.M, Z[0-9]+\.D` -`ZSEL` -`ZUMIN` -`ZDUP`
	return x.ShiftLeft(s)
}

func sveShiftLeftVec64Const(x archsimd.Uint64s) archsimd.Uint64s {
	// 64-bit LSL (vectors) is ShiftAllLeft's instruction too, so a broadcast
	// constant amount takes its immediate form.
	// arm64:`ZLSL\s+\$5, Z[0-9]+\.D, Z[0-9]+\.D` -`PWHILELT`
	return x.ShiftLeft(archsimd.BroadcastUint64s(5))
}

func sveShiftLeftVecIfElse(x archsimd.Int16s, s archsimd.Uint16s, m archsimd.Mask16s) archsimd.Int16s {
	// arm64:`ZLSL\s+Z[0-9]+\.H, Z[0-9]+\.H, P[0-9]+\.M, Z[0-9]+\.H` -`ZSEL` -`PWHILELT`
	return x.ShiftLeft(s).IfElse(m, x)
}

func sveShiftLeftVecIfElseAmount(x, s archsimd.Uint16s, m archsimd.Mask16s) archsimd.Uint16s {
	// The merging form keeps x in the inactive lanes, so a select keeping s stays.
	// arm64:`PWHILELT` `ZSEL`
	return x.ShiftLeft(s).IfElse(m, s)
}

// ShiftRight by vector is LSR or ASR (vectors), the same shape as ShiftLeft.
// For an amount of the element width or more LSR yields 0 and ASR the sign.

func sveShiftRightVec(x archsimd.Int32s, s archsimd.Uint32s) archsimd.Int32s {
	// arm64:`PWHILELT` `ZASR\s+Z[0-9]+\.S, Z[0-9]+\.S, P[0-9]+\.M, Z[0-9]+\.S` -`ZSEL` -`ZUMIN`
	return x.ShiftRight(s)
}

func sveShiftRightVec64(x archsimd.Uint64s, s archsimd.Uint64s) archsimd.Uint64s {
	// arm64:`PWHILELT` `ZLSR\s+Z[0-9]+\.D, Z[0-9]+\.D, P[0-9]+\.M, Z[0-9]+\.D` -`ZSEL` -`ZUMIN` -`ZDUP`
	return x.ShiftRight(s)
}

func sveShiftRightVecIfElse(x archsimd.Uint8s, s archsimd.Uint8s, m archsimd.Mask8s) archsimd.Uint8s {
	// arm64:`ZLSR\s+Z[0-9]+\.B, Z[0-9]+\.B, P[0-9]+\.M, Z[0-9]+\.B` -`ZSEL` -`PWHILELT`
	return x.ShiftRight(s).IfElse(m, x)
}

func sveShiftRightVecIfElse64(x archsimd.Int64s, s archsimd.Uint64s, m archsimd.Mask64s) archsimd.Int64s {
	// arm64:`ZASR\s+Z[0-9]+\.D, Z[0-9]+\.D, P[0-9]+\.M, Z[0-9]+\.D` -`ZSEL` -`PWHILELT`
	return x.ShiftRight(s).IfElse(m, x)
}

// The zero value of a mask is an all-false predicate.
func sveZeroMask() archsimd.Mask8s {
	var m archsimd.Mask8s
	// arm64:`PPFALSE` -`ZDUP`
	return m
}

// The all-true mask is built at the lane size, PBRKB is always .B, and PCNTP
// counts at the lane size.
func sveMaskTrailingZeros8(m archsimd.Mask8s) int {
	// arm64:`PWHILELT.*P[0-9]+\.B` `PBRKB\s+P[0-9]+\.B, P[0-9]+\.Z, P[0-9]+\.B` `PCNTP\s+P[0-9]+\.B, P[0-9]+, R[0-9]+`
	return m.TrailingZeros()
}

func sveMaskTrailingZeros16(m archsimd.Mask16s) int {
	// arm64:`PWHILELT.*P[0-9]+\.H` `PBRKB\s+P[0-9]+\.B, P[0-9]+\.Z, P[0-9]+\.B` `PCNTP\s+P[0-9]+\.H, P[0-9]+, R[0-9]+`
	return m.TrailingZeros()
}

func sveMaskTrailingZeros32(m archsimd.Mask32s) int {
	// arm64:`PWHILELT.*P[0-9]+\.S` `PBRKB\s+P[0-9]+\.B, P[0-9]+\.Z, P[0-9]+\.B` `PCNTP\s+P[0-9]+\.S, P[0-9]+, R[0-9]+`
	return m.TrailingZeros()
}

func sveMaskTrailingZeros64(m archsimd.Mask64s) int {
	// arm64:`PWHILELT.*P[0-9]+\.D` `PBRKB\s+P[0-9]+\.B, P[0-9]+\.Z, P[0-9]+\.B` `PCNTP\s+P[0-9]+\.D, P[0-9]+, R[0-9]+`
	return m.TrailingZeros()
}

// An operation whose unpredicated encoding needs SVE2 lowers to it only in
// blocks where the cpufeatures analysis proves SVE2; elsewhere it lowers to
// its baseline-SVE merging-predicated sibling under an all-true predicate.
func sveMulSVE2Gate(x, y archsimd.Int8s) archsimd.Int8s {
	if archsimd.ARM64.SVE2() {
		return x.Mul(y) // arm64:`ZMUL\s+Z[0-9]+\.B, Z[0-9]+\.B, Z[0-9]+\.B`
	}
	return x.Mul(y) // arm64:`PWHILELT` `ZMUL.*P[0-9]+\.M`
}

//go:noinline
func forceArgSpill(a int8) int8 { return a + 1 }

// Test spill area instructions around call to morestack.
// arm64:`MOVB R0, 40\(RSP\)`
// arm64:2`MOVD \$8\(RSP\), R27`
// arm64:`ZSTR Z0, \(VL\*0\)\(R27\)`
// arm64:2`MOVD \$48\(RSP\), R27`
// arm64:`ZSTR Z1, \(VL\*0\)\(R27\)`
// arm64:`ZLDR \(VL\*0\)\(R27\), Z1`
// arm64:`ZLDR \(VL\*0\)\(R27\), Z0`
// arm64:`FMOVS F2, 80\(RSP\)`
func sveArgSpillMixed(v archsimd.Int8s, a int8, w archsimd.Uint16s, f float32) int8 {
	return forceArgSpill(a)
}

// --- SVE destructive operations under an all-true predicate ---

func sveMaxBothSourcesLive(x, y archsimd.Int8s) archsimd.Int8s {
	// SMAX has only a predicated, destructive encoding. Both sources stay
	// live, so the result goes to a third register through an unpredicated
	// MOVPRFX: with every lane active, no lane of that register survives.
	// arm64:`ZMOVPRFX` `ZSMAX.*P[0-9]+\.M` -`ZMOVPRFX.*P[0-9]+`
	r := x.Max(y)
	sinkInt8s(x)
	sinkInt8s(y)
	return r
}

func sveMaskToInt(m archsimd.Mask8s) archsimd.Int8s {
	// Broadcast(-1).Masked(m) folds into a single zeroing CPY.
	// arm64:`ZCPY\s+\$-1, P[0-9]+\.Z, Z[0-9]+\.B` -`ZDUP` -`ZSEL`
	return m.ToInt8s()
}

// The predicate logic instructions are always .B; the governing all-true
// mask is built at the lane size.
func sveMaskAnd(x, y archsimd.Mask16s) archsimd.Mask16s {
	// arm64:`PWHILELT.*P[0-9]+\.H` `PAND\s+P[0-9]+\.B, P[0-9]+\.B, P[0-9]+\.Z, P[0-9]+\.B`
	return x.And(y)
}

// A Mask8s has a lane for every bit, so x itself governs.
func sveMask8sAnd(x, y archsimd.Mask8s) archsimd.Mask8s {
	// arm64:`PAND\s+P[0-9]+\.B, P[0-9]+\.B, P[0-9]+\.Z, P[0-9]+\.B` -`PWHILELT` -`PTRUE`
	return x.And(y)
}

func sveMask8sAndNot(x, y archsimd.Mask8s) archsimd.Mask8s {
	// arm64:`PBIC\s+P[0-9]+\.B, P[0-9]+\.B, P[0-9]+\.Z, P[0-9]+\.B` -`PWHILELT` -`PTRUE`
	return x.AndNot(y)
}

func sveMaskAndNot(x, y archsimd.Mask16s) archsimd.Mask16s {
	// arm64:`PWHILELT.*P[0-9]+\.H` `PBIC\s+P[0-9]+\.B, P[0-9]+\.B, P[0-9]+\.Z, P[0-9]+\.B`
	return x.AndNot(y)
}

func sveMaskOr(x, y archsimd.Mask16s) archsimd.Mask16s {
	// arm64:`PWHILELT.*P[0-9]+\.H` `PORR\s+P[0-9]+\.B, P[0-9]+\.B, P[0-9]+\.Z, P[0-9]+\.B`
	return x.Or(y)
}

func sveMaskXor(x, y archsimd.Mask16s) archsimd.Mask16s {
	// arm64:`PWHILELT.*P[0-9]+\.H` `PEOR\s+P[0-9]+\.B, P[0-9]+\.B, P[0-9]+\.Z, P[0-9]+\.B`
	return x.Xor(y)
}

func sveMaskNot(x archsimd.Mask16s) archsimd.Mask16s {
	// NOT is an EOR under the all-true mask of the lane size.
	// arm64:`PWHILELT.*P[0-9]+\.H` `PEOR\s+P[0-9]+\.B, P[0-9]+\.B, P[0-9]+\.Z, P[0-9]+\.B`
	return x.Not()
}

// Elem inlines to a bounds check against the runtime length, then LASTB
// under the mask of lanes 0..i.
func sveElem32(x archsimd.Int32s, i uint8) int32 {
	// arm64:`RDVL` `PWHILELT.*P[0-9]+\.S` `ZLASTBW\s+Z[0-9]+\.S, P[0-9]+, R[0-9]+`
	return x.Elem(i)
}

// A constant index folds into the PWHILELT bound, and a signed byte is
// sign-extended where it is widened.
func sveElemConst8(x archsimd.Int8s) int64 {
	// arm64:`PWHILELT.*P[0-9]+\.B` `ZLASTBW\s+Z[0-9]+\.B, P[0-9]+, R[0-9]+` `MOVB\s+R[0-9]+, R[0-9]+` -`ADD\s+\$1,`
	return int64(x.Elem(3))
}

func sveElemFloat64(x archsimd.Float64s, i uint8) float64 {
	// arm64:`PWHILELT.*P[0-9]+\.D` `ZLASTBD\s+Z[0-9]+\.D, P[0-9]+, V[0-9]+`
	return x.Elem(i)
}

// LASTB takes its predicate in three bits, so with ten masks live across
// Elem, the predicate must still be one of P0-P7. The selects after Elem
// use its result, so Elem cannot be scheduled after them.
func sveElemManyLiveMasks(x, a archsimd.Int32s, i uint8, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 int) (int32, archsimd.Int32s) {
	m0 := archsimd.Mask32sFromCount(c0)
	m1 := archsimd.Mask32sFromCount(c1)
	m2 := archsimd.Mask32sFromCount(c2)
	m3 := archsimd.Mask32sFromCount(c3)
	m4 := archsimd.Mask32sFromCount(c4)
	m5 := archsimd.Mask32sFromCount(c5)
	m6 := archsimd.Mask32sFromCount(c6)
	m7 := archsimd.Mask32sFromCount(c7)
	m8 := archsimd.Mask32sFromCount(c8)
	m9 := archsimd.Mask32sFromCount(c9)
	a = a.IfElse(m0, x)
	a = a.IfElse(m1, x)
	a = a.IfElse(m2, x)
	a = a.IfElse(m3, x)
	a = a.IfElse(m4, x)
	a = a.IfElse(m5, x)
	a = a.IfElse(m6, x)
	a = a.IfElse(m7, x)
	a = a.IfElse(m8, x)
	a = a.IfElse(m9, x)
	// arm64:`ZLASTBW\s+Z[0-9]+\.S, P[0-7], R[0-9]+` -`ZLASTBW\s+Z[0-9]+\.S, P(?:[89]|1[0-5]),`
	e := x.Elem(i)
	y := x.Add(archsimd.BroadcastInt32s(e))
	a = a.IfElse(m0, y)
	a = a.IfElse(m1, y)
	a = a.IfElse(m2, y)
	a = a.IfElse(m3, y)
	a = a.IfElse(m4, y)
	a = a.IfElse(m5, y)
	a = a.IfElse(m6, y)
	a = a.IfElse(m7, y)
	a = a.IfElse(m8, y)
	a = a.IfElse(m9, y)
	return e, a
}

// SetElem: Next after the mask of lanes 0..i-1 selects lane i, and the select
// over the broadcast folds into a merging CPY.
func sveSetElem16(x archsimd.Int16s, i uint8, y int16) archsimd.Int16s {
	// arm64:`PWHILELT.*P[0-9]+\.H` `PPNEXT\s+P[0-9]+\.H, P[0-9]+, P[0-9]+\.H` `ZCPYW\s+R[0-9]+, P[0-9]+\.M, Z[0-9]+\.H` -`ZSEL` -`ZDUP`
	return x.SetElem(i, y)
}

func sveSetElemFloat32(x archsimd.Float32s, i uint8, y float32) archsimd.Float32s {
	// arm64:`PPNEXT\s+P[0-9]+\.S` `ZCPYS\s+V[0-9]+, P[0-9]+\.M, Z[0-9]+\.S` -`ZSEL` -`ZDUP`
	return x.SetElem(i, y)
}

func sveBroadcastConstIfElse(m archsimd.Mask16s, z archsimd.Int16s) archsimd.Int16s {
	// arm64:`ZCPY\s+\$5, P[0-9]+\.M, Z[0-9]+\.H` -`ZDUP` -`ZSEL`
	return archsimd.BroadcastInt16s(5).IfElse(m, z)
}

func sveBroadcastGpIfElse32(x int32, m archsimd.Mask32s, z archsimd.Int32s) archsimd.Int32s {
	// arm64:`ZCPYW\s+R[0-9]+, P[0-9]+\.M, Z[0-9]+\.S` -`ZDUP` -`ZSEL`
	return archsimd.BroadcastInt32s(x).IfElse(m, z)
}

func sveBroadcastGpIfElse64(x int64, m archsimd.Mask64s, z archsimd.Int64s) archsimd.Int64s {
	// arm64:`ZCPY\s+R[0-9]+, P[0-9]+\.M, Z[0-9]+\.D` -`ZDUP` -`ZSEL`
	return archsimd.BroadcastInt64s(x).IfElse(m, z)
}

func sveBroadcastFloatIfElse(x float32, m archsimd.Mask32s, z archsimd.Float32s) archsimd.Float32s {
	// arm64:`ZCPYS\s+V[0-9]+, P[0-9]+\.M, Z[0-9]+\.S` -`ZDUP` -`ZSEL`
	return archsimd.BroadcastFloat32s(x).IfElse(m, z)
}

// --- SVE table lookup ---

// SVE TBL takes a single-register table, so it lowers to one instruction with a
// one-register list and no governing predicate.
func sveLookupOrZeroBytes(table, indices archsimd.Uint8s) archsimd.Uint8s {
	// arm64:`ZTBL\s+Z[0-9]+\.B, \[Z[0-9]+\.B\], Z[0-9]+\.B` -`PWHILELT`
	return table.LookupOrZero(indices)
}

func sveLookupOrZeroSigned(table archsimd.Int8s, indices archsimd.Uint8s) archsimd.Int8s {
	// arm64:`ZTBL\s+Z[0-9]+\.B, \[Z[0-9]+\.B\], Z[0-9]+\.B` -`PWHILELT`
	return table.LookupOrZero(indices)
}

// Keep >8 Mask32s values live simultaneously across Pg1013 operations (ZCPYW
// /M) so regalloc must restrict <Pg> to P0-P7 even under predicate pressure.
func sveManyLiveMasks(a archsimd.Int32s, x int32, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 int) archsimd.Int32s {
	m0 := archsimd.Mask32sFromCount(c0)
	m1 := archsimd.Mask32sFromCount(c1)
	m2 := archsimd.Mask32sFromCount(c2)
	m3 := archsimd.Mask32sFromCount(c3)
	m4 := archsimd.Mask32sFromCount(c4)
	m5 := archsimd.Mask32sFromCount(c5)
	m6 := archsimd.Mask32sFromCount(c6)
	m7 := archsimd.Mask32sFromCount(c7)
	m8 := archsimd.Mask32sFromCount(c8)
	m9 := archsimd.Mask32sFromCount(c9)
	a = archsimd.BroadcastInt32s(x).IfElse(m0, a)
	a = archsimd.BroadcastInt32s(x).IfElse(m1, a)
	a = archsimd.BroadcastInt32s(x).IfElse(m2, a)
	a = archsimd.BroadcastInt32s(x).IfElse(m3, a)
	a = archsimd.BroadcastInt32s(x).IfElse(m4, a)
	a = archsimd.BroadcastInt32s(x).IfElse(m5, a)
	a = archsimd.BroadcastInt32s(x).IfElse(m6, a)
	a = archsimd.BroadcastInt32s(x).IfElse(m7, a)
	// arm64:`ZCPYW\s+R[0-9]+, P[0-7]\.M, Z[0-9]+\.S` -`P(?:[89]|1[0-5])\.M`
	a = archsimd.BroadcastInt32s(x).IfElse(m8, a)
	// arm64:`ZCPYW\s+R[0-9]+, P[0-7]\.M, Z[0-9]+\.S` -`P(?:[89]|1[0-5])\.M`
	a = archsimd.BroadcastInt32s(x).IfElse(m9, a)
	a = archsimd.BroadcastInt32s(x).IfElse(m0, a)
	a = archsimd.BroadcastInt32s(x).IfElse(m1, a)
	a = archsimd.BroadcastInt32s(x).IfElse(m2, a)
	a = archsimd.BroadcastInt32s(x).IfElse(m3, a)
	a = archsimd.BroadcastInt32s(x).IfElse(m4, a)
	a = archsimd.BroadcastInt32s(x).IfElse(m5, a)
	a = archsimd.BroadcastInt32s(x).IfElse(m6, a)
	a = archsimd.BroadcastInt32s(x).IfElse(m7, a)
	// arm64:`ZCPYW\s+R[0-9]+, P[0-7]\.M, Z[0-9]+\.S` -`P(?:[89]|1[0-5])\.M`
	a = archsimd.BroadcastInt32s(x).IfElse(m8, a)
	// arm64:`ZCPYW\s+R[0-9]+, P[0-7]\.M, Z[0-9]+\.S` -`P(?:[89]|1[0-5])\.M`
	a = archsimd.BroadcastInt32s(x).IfElse(m9, a)
	return a
}

// Keep >8 Mask32s values live simultaneously across simdgen-generated Pg1013
// operations (ZFABS /M) so regalloc must restrict <Pg> to P0-P7.
func sveManyLiveMasksGen(a, y archsimd.Float32s, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 int) archsimd.Float32s {
	m0 := archsimd.Mask32sFromCount(c0)
	m1 := archsimd.Mask32sFromCount(c1)
	m2 := archsimd.Mask32sFromCount(c2)
	m3 := archsimd.Mask32sFromCount(c3)
	m4 := archsimd.Mask32sFromCount(c4)
	m5 := archsimd.Mask32sFromCount(c5)
	m6 := archsimd.Mask32sFromCount(c6)
	m7 := archsimd.Mask32sFromCount(c7)
	m8 := archsimd.Mask32sFromCount(c8)
	m9 := archsimd.Mask32sFromCount(c9)
	a = a.Abs().IfElse(m0, y)
	a = a.Abs().IfElse(m1, y)
	a = a.Abs().IfElse(m2, y)
	a = a.Abs().IfElse(m3, y)
	a = a.Abs().IfElse(m4, y)
	a = a.Abs().IfElse(m5, y)
	a = a.Abs().IfElse(m6, y)
	a = a.Abs().IfElse(m7, y)
	// arm64:`ZFABS\s+Z[0-9]+\.S, P[0-7]\.M, Z[0-9]+\.S` -`P(?:[89]|1[0-5])\.M`
	a = a.Abs().IfElse(m8, y)
	// arm64:`ZFABS\s+Z[0-9]+\.S, P[0-7]\.M, Z[0-9]+\.S` -`P(?:[89]|1[0-5])\.M`
	a = a.Abs().IfElse(m9, y)
	a = a.Abs().IfElse(m0, y)
	a = a.Abs().IfElse(m1, y)
	a = a.Abs().IfElse(m2, y)
	a = a.Abs().IfElse(m3, y)
	a = a.Abs().IfElse(m4, y)
	a = a.Abs().IfElse(m5, y)
	a = a.Abs().IfElse(m6, y)
	a = a.Abs().IfElse(m7, y)
	// arm64:`ZFABS\s+Z[0-9]+\.S, P[0-7]\.M, Z[0-9]+\.S` -`P(?:[89]|1[0-5])\.M`
	a = a.Abs().IfElse(m8, y)
	// arm64:`ZFABS\s+Z[0-9]+\.S, P[0-7]\.M, Z[0-9]+\.S` -`P(?:[89]|1[0-5])\.M`
	a = a.Abs().IfElse(m9, y)
	return a
}
