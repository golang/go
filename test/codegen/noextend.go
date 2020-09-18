// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

var sval64 [8]int64
var sval32 [8]int32
var sval16 [8]int16
var sval8 [8]int8
var val64 [8]uint64
var val32 [8]uint32
var val16 [8]uint16
var val8 [8]uint8

// ----------------------------- //
//    avoid zero/sign extensions //
// ----------------------------- //

func set16(x8 int8, u8 uint8, y8 int8, z8 uint8) {
	// Truncate not needed, load does sign/zero extend
	// ppc64:-"MOVB\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVB\tR\\d+,\\sR\\d+"
	sval16[0] = int16(x8)

	// ppc64:-"MOVBZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVBZ\tR\\d+,\\sR\\d+"
	val16[0] = uint16(u8)

	// AND not needed due to size
	// ppc64:-"ANDCC"
	// ppc64le:-"ANDCC"
	sval16[1] = 255 & int16(x8+y8)

	// ppc64:-"ANDCC"
	// ppc64le:-"ANDCC"
	val16[1] = 255 & uint16(u8+z8)

}
func shiftidx(x8 int8, u8 uint8, x16 int16, u16 uint16, x32 int32, u32 uint32) {
	// ppc64:-"MOVB\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVB\tR\\d+,\\sR\\d+"
	sval16[0] = int16(val16[x8>>1])

	// ppc64:-"MOVBZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVBZ\tR\\d+,\\sR\\d+"
	val16[0] = uint16(sval16[u8>>2])

	// ppc64:-"MOVH\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVH\tR\\d+,\\sR\\d+"
	sval16[1] = int16(val16[x16>>1])

	// ppc64:-"MOVHZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVHZ\tR\\d+,\\sR\\d+"
	val16[1] = uint16(sval16[u16>>2])

}

func setnox(x8 int8, u8 uint8, y8 int8, z8 uint8, x16 int16, u16 uint16, x32 int32, u32 uint32) {
	// Truncate not needed due to sign/zero extension on load

	// ppc64:-"MOVB\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVB\tR\\d+,\\sR\\d+"
	sval16[0] = int16(x8)

	// ppc64:-"MOVBZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVBZ\tR\\d+,\\sR\\d+"
	val16[0] = uint16(u8)

	// AND not needed due to size
	// ppc64:-"ANDCC"
	// ppc64le:-"ANDCC"
	sval16[1] = 255 & int16(x8+y8)

	// ppc64:-"ANDCC"
	// ppc64le:-"ANDCC"
	val16[1] = 255 & uint16(u8+z8)

	// ppc64:-"MOVB\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVB\tR\\d+,\\sR\\d+"
	sval32[0] = int32(x8)

	// ppc64:-"MOVH\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVH\tR\\d+,\\sR\\d+"
	sval32[1] = int32(x16)

	//ppc64:-"MOVBZ\tR\\d+,\\sR\\d+"
	//ppc64le:-"MOVBZ\tR\\d+,\\sR\\d+"
	val32[0] = uint32(u8)

	// ppc64:-"MOVHZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVHZ\tR\\d+,\\sR\\d+"
	val32[1] = uint32(u16)

	// ppc64:-"MOVB\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVB\tR\\d+,\\sR\\d+"
	sval64[0] = int64(x8)

	// ppc64:-"MOVH\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVH\tR\\d+,\\sR\\d+"
	sval64[1] = int64(x16)

	// ppc64:-"MOVW\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVW\tR\\d+,\\sR\\d+"
	sval64[2] = int64(x32)

	//ppc64:-"MOVBZ\tR\\d+,\\sR\\d+"
	//ppc64le:-"MOVBZ\tR\\d+,\\sR\\d+"
	val64[0] = uint64(u8)

	// ppc64:-"MOVHZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVHZ\tR\\d+,\\sR\\d+"
	val64[1] = uint64(u16)

	// ppc64:-"MOVWZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVWZ\tR\\d+,\\sR\\d+"
	val64[2] = uint64(u32)
}

func cmp16(x8 int8, u8 uint8, x32 int32, u32 uint32, x64 int64, u64 uint64) bool {
	// ppc64:-"MOVB\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVB\tR\\d+,\\sR\\d+"
	if int16(x8) == sval16[0] {
		return true
	}

	// ppc64:-"MOVBZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVBZ\tR\\d+,\\sR\\d+"
	if uint16(u8) == val16[0] {
		return true
	}

	// ppc64:-"MOVHZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVHZ\tR\\d+,\\sR\\d+"
	if uint16(u32>>16) == val16[0] {
		return true
	}

	// ppc64:-"MOVHZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVHZ\tR\\d+,\\sR\\d+"
	if uint16(u64>>48) == val16[0] {
		return true
	}

	// Verify the truncates are using the correct sign.
	// ppc64:-"MOVHZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVHZ\tR\\d+,\\sR\\d+"
	if int16(x32) == sval16[0] {
		return true
	}

	// ppc64:-"MOVH\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVH\tR\\d+,\\sR\\d+"
	if uint16(u32) == val16[0] {
		return true
	}

	// ppc64:-"MOVHZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVHZ\tR\\d+,\\sR\\d+"
	if int16(x64) == sval16[0] {
		return true
	}

	// ppc64:-"MOVH\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVH\tR\\d+,\\sR\\d+"
	if uint16(u64) == val16[0] {
		return true
	}

	return false
}

func cmp32(x8 int8, u8 uint8, x16 int16, u16 uint16, x64 int64, u64 uint64) bool {
	// ppc64:-"MOVB\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVB\tR\\d+,\\sR\\d+"
	if int32(x8) == sval32[0] {
		return true
	}

	// ppc64:-"MOVBZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVBZ\tR\\d+,\\sR\\d+"
	if uint32(u8) == val32[0] {
		return true
	}

	// ppc64:-"MOVH\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVH\tR\\d+,\\sR\\d+"
	if int32(x16) == sval32[0] {
		return true
	}

	// ppc64:-"MOVHZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVHZ\tR\\d+,\\sR\\d+"
	if uint32(u16) == val32[0] {
		return true
	}

	// Verify the truncates are using the correct sign.
	// ppc64:-"MOVWZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVWZ\tR\\d+,\\sR\\d+"
	if int32(x64) == sval32[0] {
		return true
	}

	// ppc64:-"MOVW\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVW\tR\\d+,\\sR\\d+"
	if uint32(u64) == val32[0] {
		return true
	}

	return false
}

func cmp64(x8 int8, u8 uint8, x16 int16, u16 uint16, x32 int32, u32 uint32) bool {
	// ppc64:-"MOVB\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVB\tR\\d+,\\sR\\d+"
	if int64(x8) == sval64[0] {
		return true
	}

	// ppc64:-"MOVBZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVBZ\tR\\d+,\\sR\\d+"
	if uint64(u8) == val64[0] {
		return true
	}

	// ppc64:-"MOVH\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVH\tR\\d+,\\sR\\d+"
	if int64(x16) == sval64[0] {
		return true
	}

	// ppc64:-"MOVHZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVHZ\tR\\d+,\\sR\\d+"
	if uint64(u16) == val64[0] {
		return true
	}

	// ppc64:-"MOVW\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVW\tR\\d+,\\sR\\d+"
	if int64(x32) == sval64[0] {
		return true
	}

	// ppc64:-"MOVWZ\tR\\d+,\\sR\\d+"
	// ppc64le:-"MOVWZ\tR\\d+,\\sR\\d+"
	if uint64(u32) == val64[0] {
		return true
	}
	return false
}
