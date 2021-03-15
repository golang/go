// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains codegen tests related to bit field
// insertion/extraction simplifications/optimizations.

func extr1(x, x2 uint64) uint64 {
	return x<<7 + x2>>57 // arm64:"EXTR\t[$]57,"
}

func extr2(x, x2 uint64) uint64 {
	return x<<7 | x2>>57 // arm64:"EXTR\t[$]57,"
}

func extr3(x, x2 uint64) uint64 {
	return x<<7 ^ x2>>57 // arm64:"EXTR\t[$]57,"
}

func extr4(x, x2 uint32) uint32 {
	return x<<7 + x2>>25 // arm64:"EXTRW\t[$]25,"
}

func extr5(x, x2 uint32) uint32 {
	return x<<7 | x2>>25 // arm64:"EXTRW\t[$]25,"
}

func extr6(x, x2 uint32) uint32 {
	return x<<7 ^ x2>>25 // arm64:"EXTRW\t[$]25,"
}

// check 32-bit shift masking
func mask32(x uint32) uint32 {
	return (x << 29) >> 29 // arm64:"AND\t[$]7, R[0-9]+",-"LSR",-"LSL"
}

// check 16-bit shift masking
func mask16(x uint16) uint16 {
	return (x << 14) >> 14 // arm64:"AND\t[$]3, R[0-9]+",-"LSR",-"LSL"
}

// check 8-bit shift masking
func mask8(x uint8) uint8 {
	return (x << 7) >> 7 // arm64:"AND\t[$]1, R[0-9]+",-"LSR",-"LSL"
}

func maskshift(x uint64) uint64 {
	// arm64:"AND\t[$]4095, R[0-9]+",-"LSL",-"LSR",-"UBFIZ",-"UBFX"
	return ((x << 5) & (0xfff << 5)) >> 5
}

// bitfield ops
// bfi
func bfi1(x, y uint64) uint64 {
	// arm64:"BFI\t[$]4, R[0-9]+, [$]12",-"LSL",-"LSR",-"AND"
	return ((x & 0xfff) << 4) | (y & 0xffffffffffff000f)
}

func bfi2(x, y uint64) uint64 {
	// arm64:"BFI\t[$]12, R[0-9]+, [$]40",-"LSL",-"LSR",-"AND"
	return (x << 24 >> 12) | (y & 0xfff0000000000fff)
}

// bfxil
func bfxil1(x, y uint64) uint64 {
	// arm64:"BFXIL\t[$]5, R[0-9]+, [$]12",-"LSL",-"LSR",-"AND"
	return ((x >> 5) & 0xfff) | (y & 0xfffffffffffff000)
}

func bfxil2(x, y uint64) uint64 {
	// arm64:"BFXIL\t[$]12, R[0-9]+, [$]40",-"LSL",-"LSR",-"AND"
	return (x << 12 >> 24) | (y & 0xffffff0000000000)
}

// sbfiz
func sbfiz1(x int64) int64 {
	// arm64:"SBFIZ\t[$]1, R[0-9]+, [$]60",-"LSL",-"ASR"
	return (x << 4) >> 3
}

func sbfiz2(x int32) int64 {
	return int64(x << 3) // arm64:"SBFIZ\t[$]3, R[0-9]+, [$]29",-"LSL"
}

func sbfiz3(x int16) int64 {
	return int64(x << 3) // arm64:"SBFIZ\t[$]3, R[0-9]+, [$]13",-"LSL"
}

func sbfiz4(x int8) int64 {
	return int64(x << 3) // arm64:"SBFIZ\t[$]3, R[0-9]+, [$]5",-"LSL"
}

func sbfiz5(x int32) int32 {
	// arm64:"SBFIZ\t[$]1, R[0-9]+, [$]28",-"LSL",-"ASR"
	return (x << 4) >> 3
}

// sbfx
func sbfx1(x int64) int64 {
	return (x << 3) >> 4 // arm64:"SBFX\t[$]1, R[0-9]+, [$]60",-"LSL",-"ASR"
}

func sbfx2(x int64) int64 {
	return (x << 60) >> 60 // arm64:"SBFX\tZR, R[0-9]+, [$]4",-"LSL",-"ASR"
}

func sbfx3(x int32) int64 {
	return int64(x) >> 3 // arm64:"SBFX\t[$]3, R[0-9]+, [$]29",-"ASR"
}

func sbfx4(x int16) int64 {
	return int64(x) >> 3 // arm64:"SBFX\t[$]3, R[0-9]+, [$]13",-"ASR"
}

func sbfx5(x int8) int64 {
	return int64(x) >> 3 // arm64:"SBFX\t[$]3, R[0-9]+, [$]5",-"ASR"
}

func sbfx6(x int32) int32 {
	return (x << 3) >> 4 // arm64:"SBFX\t[$]1, R[0-9]+, [$]28",-"LSL",-"ASR"
}

// ubfiz
func ubfiz1(x uint64) uint64 {
	// arm64:"UBFIZ\t[$]3, R[0-9]+, [$]12",-"LSL",-"AND"
	// s390x:"RISBGZ\t[$]49, [$]60, [$]3,",-"SLD",-"AND"
	return (x & 0xfff) << 3
}

func ubfiz2(x uint64) uint64 {
	// arm64:"UBFIZ\t[$]4, R[0-9]+, [$]12",-"LSL",-"AND"
	// s390x:"RISBGZ\t[$]48, [$]59, [$]4,",-"SLD",-"AND"
	return (x << 4) & 0xfff0
}

func ubfiz3(x uint32) uint64 {
	return uint64(x+1) << 3 // arm64:"UBFIZ\t[$]3, R[0-9]+, [$]32",-"LSL"
}

func ubfiz4(x uint16) uint64 {
	return uint64(x+1) << 3 // arm64:"UBFIZ\t[$]3, R[0-9]+, [$]16",-"LSL"
}

func ubfiz5(x uint8) uint64 {
	return uint64(x+1) << 3 // arm64:"UBFIZ\t[$]3, R[0-9]+, [$]8",-"LSL"
}

func ubfiz6(x uint64) uint64 {
	// arm64:"UBFIZ\t[$]1, R[0-9]+, [$]60",-"LSL",-"LSR"
	// s390x:"RISBGZ\t[$]3, [$]62, [$]1, ",-"SLD",-"SRD"
	return (x << 4) >> 3
}

func ubfiz7(x uint32) uint32 {
	// arm64:"UBFIZ\t[$]1, R[0-9]+, [$]28",-"LSL",-"LSR"
	return (x << 4) >> 3
}

func ubfiz8(x uint64) uint64 {
	// arm64:"UBFIZ\t[$]1, R[0-9]+, [$]20",-"LSL",-"LSR"
	// s390x:"RISBGZ\t[$]43, [$]62, [$]1, ",-"SLD",-"SRD",-"AND"
	return ((x & 0xfffff) << 4) >> 3
}

func ubfiz9(x uint64) uint64 {
	// arm64:"UBFIZ\t[$]5, R[0-9]+, [$]13",-"LSL",-"LSR",-"AND"
	return ((x << 3) & 0xffff) << 2
}

func ubfiz10(x uint64) uint64 {
	// arm64:"UBFIZ\t[$]7, R[0-9]+, [$]12",-"LSL",-"LSR",-"AND"
	// s390x:"RISBGZ\t[$]45, [$]56, [$]7, ",-"SLD",-"SRD",-"AND"
	return ((x << 5) & (0xfff << 5)) << 2
}

// ubfx
func ubfx1(x uint64) uint64 {
	// arm64:"UBFX\t[$]25, R[0-9]+, [$]10",-"LSR",-"AND"
	// s390x:"RISBGZ\t[$]54, [$]63, [$]39, ",-"SRD",-"AND"
	return (x >> 25) & 1023
}

func ubfx2(x uint64) uint64 {
	// arm64:"UBFX\t[$]4, R[0-9]+, [$]8",-"LSR",-"AND"
	// s390x:"RISBGZ\t[$]56, [$]63, [$]60, ",-"SRD",-"AND"
	return (x & 0x0ff0) >> 4
}

func ubfx3(x uint32) uint64 {
	return uint64(x >> 15) // arm64:"UBFX\t[$]15, R[0-9]+, [$]17",-"LSR"
}

func ubfx4(x uint16) uint64 {
	return uint64(x >> 9) // arm64:"UBFX\t[$]9, R[0-9]+, [$]7",-"LSR"
}

func ubfx5(x uint8) uint64 {
	return uint64(x >> 3) // arm64:"UBFX\t[$]3, R[0-9]+, [$]5",-"LSR"
}

func ubfx6(x uint64) uint64 {
	// arm64:"UBFX\t[$]1, R[0-9]+, [$]62",-"LSL",-"LSR"
	// s390x:"RISBGZ\t[$]2, [$]63, [$]63,",-"SLD",-"SRD"
	return (x << 1) >> 2
}

func ubfx7(x uint32) uint32 {
	// arm64:"UBFX\t[$]1, R[0-9]+, [$]30",-"LSL",-"LSR"
	return (x << 1) >> 2
}

func ubfx8(x uint64) uint64 {
	// arm64:"UBFX\t[$]1, R[0-9]+, [$]12",-"LSL",-"LSR",-"AND"
	// s390x:"RISBGZ\t[$]52, [$]63, [$]63,",-"SLD",-"SRD",-"AND"
	return ((x << 1) >> 2) & 0xfff
}

func ubfx9(x uint64) uint64 {
	// arm64:"UBFX\t[$]4, R[0-9]+, [$]11",-"LSL",-"LSR",-"AND"
	// s390x:"RISBGZ\t[$]53, [$]63, [$]60, ",-"SLD",-"SRD",-"AND"
	return ((x >> 3) & 0xfff) >> 1
}

func ubfx10(x uint64) uint64 {
	// arm64:"UBFX\t[$]5, R[0-9]+, [$]56",-"LSL",-"LSR"
	// s390x:"RISBGZ\t[$]8, [$]63, [$]59, ",-"SLD",-"SRD"
	return ((x >> 2) << 5) >> 8
}

func ubfx11(x uint64) uint64 {
	// arm64:"UBFX\t[$]1, R[0-9]+, [$]19",-"LSL",-"LSR"
	// s390x:"RISBGZ\t[$]45, [$]63, [$]63, ",-"SLD",-"SRD",-"AND"
	return ((x & 0xfffff) << 3) >> 4
}

// Check that we don't emit comparisons for constant shifts.
//go:nosplit
func shift_no_cmp(x int) int {
	// arm64:`LSL\t[$]17`,-`CMP`
	// mips64:`SLLV\t[$]17`,-`SGT`
	return x << 17
}
