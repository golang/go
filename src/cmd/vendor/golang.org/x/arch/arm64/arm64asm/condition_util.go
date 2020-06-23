// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64asm

func extract_bit(value, bit uint32) uint32 {
	return (value >> bit) & 1
}

func bfxpreferred_4(sf, opc1, imms, immr uint32) bool {
	if imms < immr {
		return false
	}
	if (imms>>5 == sf) && (imms&0x1f == 0x1f) {
		return false
	}
	if immr == 0 {
		if sf == 0 && (imms == 7 || imms == 15) {
			return false
		}
		if sf == 1 && opc1 == 0 && (imms == 7 ||
			imms == 15 || imms == 31) {
			return false
		}
	}
	return true
}

func move_wide_preferred_4(sf, N, imms, immr uint32) bool {
	if sf == 1 && N != 1 {
		return false
	}
	if sf == 0 && !(N == 0 && ((imms>>5)&1) == 0) {
		return false
	}
	if imms < 16 {
		return (-immr)%16 <= (15 - imms)
	}
	width := uint32(32)
	if sf == 1 {
		width = uint32(64)
	}
	if imms >= (width - 15) {
		return (immr % 16) <= (imms - (width - 15))
	}
	return false
}

type Sys uint8

const (
	Sys_AT Sys = iota
	Sys_DC
	Sys_IC
	Sys_TLBI
	Sys_SYS
)

func sys_op_4(op1, crn, crm, op2 uint32) Sys {
	// TODO: system instruction
	return Sys_SYS
}

func is_zero(x uint32) bool {
	return x == 0
}

func is_ones_n16(x uint32) bool {
	return x == 0xffff
}

func bit_count(x uint32) uint8 {
	var count uint8
	for count = 0; x > 0; x >>= 1 {
		if (x & 1) == 1 {
			count++
		}
	}
	return count
}
