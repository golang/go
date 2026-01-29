// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains codegen tests related to bit field
// insertion/extraction simplifications/optimizations.

func extr1(x, x2 uint64) uint64 {
	return x<<7 + x2>>57 // arm64:"EXTR [$]57,"
}

func extr2(x, x2 uint64) uint64 {
	return x<<7 | x2>>57 // arm64:"EXTR [$]57,"
}

func extr3(x, x2 uint64) uint64 {
	return x<<7 ^ x2>>57 // arm64:"EXTR [$]57,"
}

func extr4(x, x2 uint32) uint32 {
	return x<<7 + x2>>25 // arm64:"EXTRW [$]25,"
}

func extr5(x, x2 uint32) uint32 {
	return x<<7 | x2>>25 // arm64:"EXTRW [$]25,"
}

func extr6(x, x2 uint32) uint32 {
	return x<<7 ^ x2>>25 // arm64:"EXTRW [$]25,"
}

// check 32-bit shift masking
func mask32(x uint32) uint32 {
	return (x << 29) >> 29 // arm64:"AND [$]7, R[0-9]+" -"LSR" -"LSL"
}

// check 16-bit shift masking
func mask16(x uint16) uint16 {
	return (x << 14) >> 14 // arm64:"AND [$]3, R[0-9]+" -"LSR" -"LSL"
}

// check 8-bit shift masking
func mask8(x uint8) uint8 {
	return (x << 7) >> 7 // arm64:"AND [$]1, R[0-9]+" -"LSR" -"LSL"
}

func maskshift(x uint64) uint64 {
	// arm64:"AND [$]4095, R[0-9]+" -"LSL" -"LSR" -"UBFIZ" -"UBFX"
	return ((x << 5) & (0xfff << 5)) >> 5
}

// bitfield ops
// bfi
func bfi1(x, y uint64) uint64 {
	// arm64:"BFI [$]4, R[0-9]+, [$]12" -"LSL" -"LSR" -"AND"
	return ((x & 0xfff) << 4) | (y & 0xffffffffffff000f)
}

func bfi2(x, y uint64) uint64 {
	// arm64:"BFI [$]12, R[0-9]+, [$]40" -"LSL" -"LSR" -"AND"
	return (x << 24 >> 12) | (y & 0xfff0000000000fff)
}

// bfxil
func bfxil1(x, y uint64) uint64 {
	// arm64:"BFXIL [$]5, R[0-9]+, [$]12" -"LSL" -"LSR" -"AND"
	return ((x >> 5) & 0xfff) | (y & 0xfffffffffffff000)
}

func bfxil2(x, y uint64) uint64 {
	// arm64:"BFXIL [$]12, R[0-9]+, [$]40" -"LSL" -"LSR" -"AND"
	return (x << 12 >> 24) | (y & 0xffffff0000000000)
}

// sbfiz
// merge shifts into sbfiz: (x << lc) >> rc && lc > rc.
func sbfiz1(x int64) int64 {
	// arm64:"SBFIZ [$]1, R[0-9]+, [$]60" -"LSL" -"ASR"
	return (x << 4) >> 3
}

// merge shift and sign-extension into sbfiz.
func sbfiz2(x int32) int64 {
	return int64(x << 3) // arm64:"SBFIZ [$]3, R[0-9]+, [$]29" -"LSL"
}

func sbfiz3(x int16) int64 {
	return int64(x << 3) // arm64:"SBFIZ [$]3, R[0-9]+, [$]13" -"LSL"
}

func sbfiz4(x int8) int64 {
	return int64(x << 3) // arm64:"SBFIZ [$]3, R[0-9]+, [$]5" -"LSL"
}

// sbfiz combinations.
// merge shift with sbfiz into sbfiz.
func sbfiz5(x int32) int32 {
	// arm64:"SBFIZ [$]1, R[0-9]+, [$]28" -"LSL" -"ASR"
	return (x << 4) >> 3
}

func sbfiz6(x int16) int64 {
	return int64(x+1) << 3 // arm64:"SBFIZ [$]3, R[0-9]+, [$]16" -"LSL"
}

func sbfiz7(x int8) int64 {
	return int64(x+1) << 62 // arm64:"SBFIZ [$]62, R[0-9]+, [$]2" -"LSL"
}

func sbfiz8(x int32) int64 {
	return int64(x+1) << 40 // arm64:"SBFIZ [$]40, R[0-9]+, [$]24" -"LSL"
}

// sbfx
// merge shifts into sbfx: (x << lc) >> rc && lc <= rc.
func sbfx1(x int64) int64 {
	return (x << 3) >> 4 // arm64:"SBFX [$]1, R[0-9]+, [$]60" -"LSL" -"ASR"
}

func sbfx2(x int64) int64 {
	return (x << 60) >> 60 // arm64:"SBFX [$]0, R[0-9]+, [$]4" -"LSL" -"ASR"
}

// merge shift and sign-extension into sbfx.
func sbfx3(x int32) int64 {
	return int64(x) >> 3 // arm64:"SBFX [$]3, R[0-9]+, [$]29" -"ASR"
}

func sbfx4(x int16) int64 {
	return int64(x) >> 3 // arm64:"SBFX [$]3, R[0-9]+, [$]13" -"ASR"
}

func sbfx5(x int8) int64 {
	return int64(x) >> 3 // arm64:"SBFX [$]3, R[0-9]+, [$]5" -"ASR"
}

func sbfx6(x int32) int64 {
	return int64(x >> 30) // arm64:"SBFX [$]30, R[0-9]+, [$]2"
}

func sbfx7(x int16) int64 {
	return int64(x >> 10) // arm64:"SBFX [$]10, R[0-9]+, [$]6"
}

func sbfx8(x int8) int64 {
	return int64(x >> 5) // arm64:"SBFX [$]5, R[0-9]+, [$]3"
}

// sbfx combinations.
// merge shifts with sbfiz into sbfx.
func sbfx9(x int32) int32 {
	return (x << 3) >> 4 // arm64:"SBFX [$]1, R[0-9]+, [$]28" -"LSL" -"ASR"
}

// merge sbfx and sign-extension into sbfx.
func sbfx10(x int32) int64 {
	c := x + 5
	return int64(c >> 20) // arm64"SBFX [$]20, R[0-9]+, [$]12" -"MOVW R[0-9]+, R[0-9]+"
}

// ubfiz
// merge shifts into ubfiz: (x<<lc)>>rc && lc>rc
func ubfiz1(x uint64) uint64 {
	// arm64:"UBFIZ [$]1, R[0-9]+, [$]60" -"LSL" -"LSR"
	// s390x:"RISBGZ [$]3, [$]62, [$]1, " -"SLD" -"SRD"
	return (x << 4) >> 3
}

// merge shift and zero-extension into ubfiz.
func ubfiz2(x uint32) uint64 {
	return uint64(x+1) << 3 // arm64:"UBFIZ [$]3, R[0-9]+, [$]32" -"LSL"
}

func ubfiz3(x uint16) uint64 {
	return uint64(x+1) << 3 // arm64:"UBFIZ [$]3, R[0-9]+, [$]16" -"LSL"
}

func ubfiz4(x uint8) uint64 {
	return uint64(x+1) << 3 // arm64:"UBFIZ [$]3, R[0-9]+, [$]8" -"LSL"
}

func ubfiz5(x uint8) uint64 {
	return uint64(x) << 60 // arm64:"UBFIZ [$]60, R[0-9]+, [$]4" -"LSL"
}

func ubfiz6(x uint32) uint64 {
	return uint64(x << 30) // arm64:"UBFIZ [$]30, R[0-9]+, [$]2",
}

func ubfiz7(x uint16) uint64 {
	return uint64(x << 10) // arm64:"UBFIZ [$]10, R[0-9]+, [$]6",
}

func ubfiz8(x uint8) uint64 {
	return uint64(x << 7) // arm64:"UBFIZ [$]7, R[0-9]+, [$]1",
}

// merge ANDconst into ubfiz.
func ubfiz9(x uint64) uint64 {
	// arm64:"UBFIZ [$]3, R[0-9]+, [$]12" -"LSL" -"AND"
	// s390x:"RISBGZ [$]49, [$]60, [$]3," -"SLD" -"AND"
	return (x & 0xfff) << 3
}

func ubfiz10(x uint64) uint64 {
	// arm64:"UBFIZ [$]4, R[0-9]+, [$]12" -"LSL" -"AND"
	// s390x:"RISBGZ [$]48, [$]59, [$]4," -"SLD" -"AND"
	return (x << 4) & 0xfff0
}

// ubfiz combinations
func ubfiz11(x uint32) uint32 {
	// arm64:"UBFIZ [$]1, R[0-9]+, [$]28" -"LSL" -"LSR"
	return (x << 4) >> 3
}

func ubfiz12(x uint64) uint64 {
	// arm64:"UBFIZ [$]1, R[0-9]+, [$]20" -"LSL" -"LSR"
	// s390x:"RISBGZ [$]43, [$]62, [$]1, " -"SLD" -"SRD" -"AND"
	return ((x & 0xfffff) << 4) >> 3
}

func ubfiz13(x uint64) uint64 {
	// arm64:"UBFIZ [$]5, R[0-9]+, [$]13" -"LSL" -"LSR" -"AND"
	return ((x << 3) & 0xffff) << 2
}

func ubfiz14(x uint64) uint64 {
	// arm64:"UBFIZ [$]7, R[0-9]+, [$]12" -"LSL" -"LSR" -"AND"
	// s390x:"RISBGZ [$]45, [$]56, [$]7, " -"SLD" -"SRD" -"AND"
	return ((x << 5) & (0xfff << 5)) << 2
}

// ubfx
// merge shifts into ubfx: (x<<lc)>>rc && lc<rc
func ubfx1(x uint64) uint64 {
	// arm64:"UBFX [$]1, R[0-9]+, [$]62" -"LSL" -"LSR"
	// s390x:"RISBGZ [$]2, [$]63, [$]63," -"SLD" -"SRD"
	return (x << 1) >> 2
}

// merge shift and zero-extension into ubfx.
func ubfx2(x uint32) uint64 {
	return uint64(x >> 15) // arm64:"UBFX [$]15, R[0-9]+, [$]17" -"LSR"
}

func ubfx3(x uint16) uint64 {
	return uint64(x >> 9) // arm64:"UBFX [$]9, R[0-9]+, [$]7" -"LSR"
}

func ubfx4(x uint8) uint64 {
	return uint64(x >> 3) // arm64:"UBFX [$]3, R[0-9]+, [$]5" -"LSR"
}

func ubfx5(x uint32) uint64 {
	return uint64(x) >> 30 // arm64:"UBFX [$]30, R[0-9]+, [$]2"
}

func ubfx6(x uint16) uint64 {
	return uint64(x) >> 10 // arm64:"UBFX [$]10, R[0-9]+, [$]6"
}

func ubfx7(x uint8) uint64 {
	return uint64(x) >> 3 // arm64:"UBFX [$]3, R[0-9]+, [$]5"
}

// merge ANDconst into ubfx.
func ubfx8(x uint64) uint64 {
	// arm64:"UBFX [$]25, R[0-9]+, [$]10" -"LSR" -"AND"
	// s390x:"RISBGZ [$]54, [$]63, [$]39, " -"SRD" -"AND"
	return (x >> 25) & 1023
}

func ubfx9(x uint64) uint64 {
	// arm64:"UBFX [$]4, R[0-9]+, [$]8" -"LSR" -"AND"
	// s390x:"RISBGZ [$]56, [$]63, [$]60, " -"SRD" -"AND"
	return (x & 0x0ff0) >> 4
}

// ubfx combinations.
func ubfx10(x uint32) uint32 {
	// arm64:"UBFX [$]1, R[0-9]+, [$]30" -"LSL" -"LSR"
	return (x << 1) >> 2
}

func ubfx11(x uint64) uint64 {
	// arm64:"UBFX [$]1, R[0-9]+, [$]12" -"LSL" -"LSR" -"AND"
	// s390x:"RISBGZ [$]52, [$]63, [$]63," -"SLD" -"SRD" -"AND"
	return ((x << 1) >> 2) & 0xfff
}

func ubfx12(x uint64) uint64 {
	// arm64:"UBFX [$]4, R[0-9]+, [$]11" -"LSL" -"LSR" -"AND"
	// s390x:"RISBGZ [$]53, [$]63, [$]60, " -"SLD" -"SRD" -"AND"
	return ((x >> 3) & 0xfff) >> 1
}

func ubfx13(x uint64) uint64 {
	// arm64:"UBFX [$]5, R[0-9]+, [$]56" -"LSL" -"LSR"
	// s390x:"RISBGZ [$]8, [$]63, [$]59, " -"SLD" -"SRD"
	return ((x >> 2) << 5) >> 8
}

func ubfx14(x uint64) uint64 {
	// arm64:"UBFX [$]1, R[0-9]+, [$]19" -"LSL" -"LSR"
	// s390x:"RISBGZ [$]45, [$]63, [$]63, " -"SLD" -"SRD" -"AND"
	return ((x & 0xfffff) << 3) >> 4
}

// merge ubfx and zero-extension into ubfx.
func ubfx15(x uint64) bool {
	midr := x + 10
	part_num := uint16((midr >> 4) & 0xfff)
	if part_num == 0xd0c { // arm64:"UBFX [$]4, R[0-9]+, [$]12" -"MOVHU R[0-9]+, R[0-9]+"
		return true
	}
	return false
}

// merge ANDconst and ubfx into ubfx
func ubfx16(x uint64) uint64 {
	// arm64:"UBFX [$]4, R[0-9]+, [$]6" -"AND [$]63"
	return ((x >> 3) & 0xfff) >> 1 & 0x3f
}

// Check that we don't emit comparisons for constant shifts.
//
//go:nosplit
func shift_no_cmp(x int) int {
	// arm64:`LSL [$]17`,-`CMP`
	// mips64:`SLLV [$]17`,-`SGT`
	return x << 17
}

func rev16(c uint64) (uint64, uint64, uint64) {
	// arm64:`REV16`,-`AND`,-`LSR`,-`AND`,-`ORR R[0-9]+<<8`
	// loong64:`REVB4H`,-`MOVV`,-`AND`,-`SRLV`,-`AND`,-`SLLV`,-`OR`
	b1 := ((c & 0xff00ff00ff00ff00) >> 8) | ((c & 0x00ff00ff00ff00ff) << 8)
	// arm64:-`ADD R[0-9]+<<8`
	// loong64:-`ADDV`
	b2 := ((c & 0xff00ff00ff00ff00) >> 8) + ((c & 0x00ff00ff00ff00ff) << 8)
	// arm64:-`EOR R[0-9]+<<8`
	// loong64:-`XOR`
	b3 := ((c & 0xff00ff00ff00ff00) >> 8) ^ ((c & 0x00ff00ff00ff00ff) << 8)
	return b1, b2, b3
}

func rev16w(c uint32) (uint32, uint32, uint32) {
	// arm64:`REV16W`,-`AND`,-`UBFX`,-`AND`,-`ORR R[0-9]+<<8`
	// loong64:`REVB2H`,-`AND`,-`SRL`,-`AND`,-`SLL`,-`OR`
	b1 := ((c & 0xff00ff00) >> 8) | ((c & 0x00ff00ff) << 8)
	// arm64:-`ADD R[0-9]+<<8`
	// loong64:-`ADDV`
	b2 := ((c & 0xff00ff00) >> 8) + ((c & 0x00ff00ff) << 8)
	// arm64:-`EOR R[0-9]+<<8`
	// loong64:-`XOR`
	b3 := ((c & 0xff00ff00) >> 8) ^ ((c & 0x00ff00ff) << 8)
	return b1, b2, b3
}

func shift(x uint32, y uint16, z uint8) uint64 {
	// arm64:-`MOVWU`,-`LSR [$]32`
	// loong64:-`MOVWU`,-`SRLV [$]32`
	a := uint64(x) >> 32
	// arm64:-`MOVHU
	// loong64:-`MOVHU`,-`SRLV [$]16`
	b := uint64(y) >> 16
	// arm64:-`MOVBU`
	// loong64:-`MOVBU`,-`SRLV [$]8`
	c := uint64(z) >> 8
	// arm64:`MOVD ZR`,-`ADD R[0-9]+>>16`,-`ADD R[0-9]+>>8`,
	// loong64:`MOVV R0`,-`ADDVU`
	return a + b + c
}
