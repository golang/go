// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "math/bits"

var sval64 [8]int64
var sval32 [8]int32
var sval16 [8]int16
var sval8 [8]int8
var val64 [8]uint64
var val32 [8]uint32
var val16 [8]uint16
var val8 [8]uint8

// Avoid zero/sign extensions following a load
// which has extended the value correctly.
// Note: No tests are done for int8 since
// an extra extension is usually needed due to
// no signed byte load.

func set16(x8 int8, u8 *uint8, y8 int8, z8 uint8) {
	// Truncate not needed, load does sign/zero extend

	// ppc64x:-"MOVBZ R\\d+,\\sR\\d+"
	val16[0] = uint16(*u8)

	// AND not needed due to size
	// ppc64x:-"ANDCC"
	sval16[1] = 255 & int16(x8+y8)

	// ppc64x:-"ANDCC"
	val16[1] = 255 & uint16(*u8+z8)

}
func shiftidx(u8 *uint8, x16 *int16, u16 *uint16) {

	// ppc64x:-"MOVBZ R\\d+,\\sR\\d+"
	val16[0] = uint16(sval16[*u8>>2])

	// ppc64x:-"MOVH R\\d+,\\sR\\d+"
	sval16[1] = int16(val16[*x16>>1])

	// ppc64x:-"MOVHZ R\\d+,\\sR\\d+"
	val16[1] = uint16(sval16[*u16>>2])

}

func setnox(x8 int8, u8 *uint8, y8 *int8, z8 *uint8, x16 *int16, u16 *uint16, x32 *int32, u32 *uint32) {

	// ppc64x:-"MOVBZ R\\d+,\\sR\\d+"
	val16[0] = uint16(*u8)

	// AND not needed due to size
	// ppc64x:-"ANDCC"
	sval16[1] = 255 & int16(x8+*y8)

	// ppc64x:-"ANDCC"
	val16[1] = 255 & uint16(*u8+*z8)

	// ppc64x:-"MOVH R\\d+,\\sR\\d+"
	sval32[1] = int32(*x16)

	// ppc64x:-"MOVBZ R\\d+,\\sR\\d+"
	val32[0] = uint32(*u8)

	// ppc64x:-"MOVHZ R\\d+,\\sR\\d+"
	val32[1] = uint32(*u16)

	// ppc64x:-"MOVH R\\d+,\\sR\\d+"
	sval64[1] = int64(*x16)

	// ppc64x:-"MOVW R\\d+,\\sR\\d+"
	sval64[2] = int64(*x32)

	// ppc64x:-"MOVBZ R\\d+,\\sR\\d+"
	val64[0] = uint64(*u8)

	// ppc64x:-"MOVHZ R\\d+,\\sR\\d+"
	val64[1] = uint64(*u16)

	// ppc64x:-"MOVWZ R\\d+,\\sR\\d+"
	val64[2] = uint64(*u32)
}

func cmp16(u8 *uint8, x32 *int32, u32 *uint32, x64 *int64, u64 *uint64) bool {

	// ppc64x:-"MOVBZ R\\d+,\\sR\\d+"
	if uint16(*u8) == val16[0] {
		return true
	}

	// ppc64x:-"MOVHZ R\\d+,\\sR\\d+"
	if uint16(*u32>>16) == val16[0] {
		return true
	}

	// ppc64x:-"MOVHZ R\\d+,\\sR\\d+"
	if uint16(*u64>>48) == val16[0] {
		return true
	}

	// Verify the truncates are using the correct sign.
	// ppc64x:-"MOVHZ R\\d+,\\sR\\d+"
	if int16(*x32) == sval16[0] {
		return true
	}

	// ppc64x:-"MOVH R\\d+,\\sR\\d+"
	if uint16(*u32) == val16[0] {
		return true
	}

	// ppc64x:-"MOVHZ R\\d+,\\sR\\d+"
	if int16(*x64) == sval16[0] {
		return true
	}

	// ppc64x:-"MOVH R\\d+,\\sR\\d+"
	if uint16(*u64) == val16[0] {
		return true
	}

	return false
}

func cmp32(u8 *uint8, x16 *int16, u16 *uint16, x64 *int64, u64 *uint64) bool {

	// ppc64x:-"MOVBZ R\\d+,\\sR\\d+"
	if uint32(*u8) == val32[0] {
		return true
	}

	// ppc64x:-"MOVH R\\d+,\\sR\\d+"
	if int32(*x16) == sval32[0] {
		return true
	}

	// ppc64x:-"MOVHZ R\\d+,\\sR\\d+"
	if uint32(*u16) == val32[0] {
		return true
	}

	// Verify the truncates are using the correct sign.
	// ppc64x:-"MOVWZ R\\d+,\\sR\\d+"
	if int32(*x64) == sval32[0] {
		return true
	}

	// ppc64x:-"MOVW R\\d+,\\sR\\d+"
	if uint32(*u64) == val32[0] {
		return true
	}

	return false
}

func cmp64(u8 *uint8, x16 *int16, u16 *uint16, x32 *int32, u32 *uint32) bool {

	// ppc64x:-"MOVBZ R\\d+,\\sR\\d+"
	if uint64(*u8) == val64[0] {
		return true
	}

	// ppc64x:-"MOVH R\\d+,\\sR\\d+"
	if int64(*x16) == sval64[0] {
		return true
	}

	// ppc64x:-"MOVHZ R\\d+,\\sR\\d+"
	if uint64(*u16) == val64[0] {
		return true
	}

	// ppc64x:-"MOVW R\\d+,\\sR\\d+"
	if int64(*x32) == sval64[0] {
		return true
	}

	// ppc64x:-"MOVWZ R\\d+,\\sR\\d+"
	if uint64(*u32) == val64[0] {
		return true
	}
	return false
}

// no unsign extension following 32 bits ops

func noUnsignEXT(t1, t2, t3, t4 uint32, k int64) uint64 {
	var ret uint64

	// arm64:"RORW" -"MOVWU"
	ret += uint64(bits.RotateLeft32(t1, 7))

	// arm64:"MULW" -"MOVWU"
	ret *= uint64(t1 * t2)

	// arm64:"MNEGW" -"MOVWU"
	ret += uint64(-t1 * t3)

	// arm64:"UDIVW" -"MOVWU"
	ret += uint64(t1 / t4)

	// arm64:-"MOVWU"
	ret += uint64(t2 % t3)

	// arm64:"MSUBW" -"MOVWU"
	ret += uint64(t1 - t2*t3)

	// arm64:"MADDW" -"MOVWU"
	ret += uint64(t3*t4 + t2)

	// arm64:"REVW" -"MOVWU"
	ret += uint64(bits.ReverseBytes32(t1))

	// arm64:"RBITW" -"MOVWU"
	ret += uint64(bits.Reverse32(t1))

	// arm64:"CLZW" -"MOVWU"
	ret += uint64(bits.LeadingZeros32(t1))

	// arm64:"REV16W" -"MOVWU"
	ret += uint64(((t1 & 0xff00ff00) >> 8) | ((t1 & 0x00ff00ff) << 8))

	// arm64:"EXTRW" -"MOVWU"
	ret += uint64((t1 << 25) | (t2 >> 7))

	return ret
}

// no sign extension when the upper bits of the result are zero

func noSignEXT(x int) int64 {
	t1 := int32(x)

	var ret int64

	// arm64:-"MOVW"
	ret += int64(t1 & 1)

	// arm64:-"MOVW"
	ret += int64(int32(x & 0x7fffffff))

	// arm64:-"MOVH"
	ret += int64(int16(x & 0x7fff))

	// arm64:-"MOVB"
	ret += int64(int8(x & 0x7f))

	return ret
}

// corner cases that sign extension must not be omitted

func shouldSignEXT(x int) int64 {
	t1 := int32(x)

	var ret int64

	// arm64:"MOVW"
	ret += int64(t1 & (-1))

	// arm64:"MOVW"
	ret += int64(int32(x & 0x80000000))

	// arm64:"MOVW"
	ret += int64(int32(x & 0x1100000011111111))

	// arm64:"MOVH"
	ret += int64(int16(x & 0x1100000000001111))

	// arm64:"MOVB"
	ret += int64(int8(x & 0x1100000000000011))

	return ret
}

func noIntermediateExtension(a, b, c uint32) uint32 {
	// arm64:-"MOVWU"
	return a*b*9 + c
}
