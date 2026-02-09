// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "math/bits"

//
// 64 bit instructions
//

func bitsCheckConstLeftShiftU64(a uint64) (n int) {
	// amd64:"BTQ [$]63,"
	// arm64:"TBNZ [$]63,"
	// loong64:"MOVV [$]" "AND" "BNE"
	// riscv64:"MOV [$]" "AND" "BNEZ"
	if a&(1<<63) != 0 {
		return 1
	}
	// amd64:"BTQ [$]60,"
	// arm64:"TBNZ [$]60,"
	// loong64:"MOVV [$]" "AND" "BNE"
	// riscv64:"MOV [$]" "AND" "BNEZ"
	if a&(1<<60) != 0 {
		return 1
	}
	// amd64:"BTL [$]0,"
	// arm64:"TBZ [$]0,"
	// loong64:"AND [$]1," "BEQ"
	// riscv64:"ANDI" "BEQZ"
	if a&(1<<0) != 0 {
		return 1
	}
	return 0
}

func bitsCheckConstRightShiftU64(a [8]uint64) (n int) {
	// amd64:"BTQ [$]63,"
	// arm64:"LSR [$]63," "TBNZ [$]0,"
	// loong64:"SRLV [$]63," "AND [$]1," "BNE"
	// riscv64:"SRLI" "ANDI" "BNEZ"
	if (a[0]>>63)&1 != 0 {
		return 1
	}
	// amd64:"BTQ [$]63,"
	// arm64:"LSR [$]63," "CBNZ"
	// loong64:"SRLV [$]63," "BNE"
	// riscv64:"SRLI" "BNEZ"
	if a[1]>>63 != 0 {
		return 1
	}
	// amd64:"BTQ [$]63,"
	// arm64:"LSR [$]63," "CBZ"
	// loong64:"SRLV [$]63," "BEQ"
	// riscv64:"SRLI" "BEQZ"
	if a[2]>>63 == 0 {
		return 1
	}
	// amd64:"BTQ [$]60,"
	// arm64:"LSR [$]60," "TBZ [$]0,"
	// loong64:"SRLV [$]60," "AND [$]1," "BEQ"
	// riscv64:"SRLI", "ANDI" "BEQZ"
	if (a[3]>>60)&1 == 0 {
		return 1
	}
	// amd64:"BTL [$]1,"
	// arm64:"LSR [$]1," "TBZ [$]0,"
	// loong64:"SRLV [$]1," "AND [$]1," "BEQ"
	// riscv64:"SRLI" "ANDI" "BEQZ"
	if (a[4]>>1)&1 == 0 {
		return 1
	}
	// amd64:"BTL [$]0,"
	// arm64:"TBZ [$]0," -"LSR"
	// loong64:"AND [$]1," "BEQ" -"SRLV"
	// riscv64:"ANDI" "BEQZ" -"SRLI"
	if (a[5]>>0)&1 == 0 {
		return 1
	}
	// amd64:"BTL [$]7,"
	// arm64:"LSR [$]5," "TBNZ [$]2,"
	// loong64:"SRLV [$]5," "AND [$]4," "BNE"
	// riscv64:"SRLI" "ANDI" "BNEZ"
	if (a[6]>>5)&4 == 0 {
		return 1
	}
	return 0
}

func bitsCheckVarU64(a, b uint64) (n int) {
	// amd64:"BTQ"
	// arm64:"MOVD [$]1," "LSL" "TST"
	// loong64:"MOVV [$]1," "SLLV R" "AND" "BNE"
	// riscv64:"ANDI [$]63," "SLL " "AND "
	if a&(1<<(b&63)) != 0 {
		return 1
	}
	// amd64:"BTQ" -"BT. [$]0,"
	// arm64:"LSR" "TBZ [$]0,"
	// loong64:"SRLV" "AND [$]1," "BEQ"
	// riscv64:"ANDI [$]63," "SRL" "ANDI [$]1,"
	if (b>>(a&63))&1 != 0 {
		return 1
	}
	return 0
}

func bitsCheckMaskU64(a uint64) (n int) {
	// amd64:"BTQ [$]63,"
	// arm64:"TBNZ [$]63,"
	// loong64:"MOVV [$]" "AND" "BNE"
	// riscv64:"MOV [$]" "AND" "BNEZ"
	if a&0x8000000000000000 != 0 {
		return 1
	}
	// amd64:"BTQ [$]59,"
	// arm64:"TBNZ [$]59,"
	// loong64:"MOVV [$]" "AND" "BNE"
	// riscv64:"MOV [$]" "AND" "BNEZ"
	if a&0x800000000000000 != 0 {
		return 1
	}
	// amd64:"BTL [$]0,"
	// arm64:"TBZ [$]0,"
	// loong64:"AND [$]1," "BEQ"
	// riscv64:"ANDI" "BEQZ"
	if a&0x1 != 0 {
		return 1
	}
	return 0
}

func bitsSetU64(a, b uint64) (n uint64) {
	// amd64:"BTSQ"
	// arm64:"MOVD [$]1," "LSL" "ORR"
	// loong64:"MOVV [$]1," "SLLV" "OR"
	// riscv64:"ANDI" "SLL" "OR"
	n += b | (1 << (a & 63))

	// amd64:"BTSQ [$]63,"
	// arm64:"ORR [$]-9223372036854775808,"
	// loong64:"MOVV [$]" "OR"
	// riscv64:"MOV [$]" "OR "
	n += a | (1 << 63)

	// amd64:"BTSQ [$]60,"
	// arm64:"ORR [$]1152921504606846976,"
	// loong64:"MOVV [$]" "OR"
	// riscv64:"MOV [$]" "OR "
	n += a | (1 << 60)

	// amd64:"ORQ [$]1,"
	// arm64:"ORR [$]1,"
	// loong64:"OR [$]1,"
	// riscv64:"ORI"
	n += a | (1 << 0)

	return n
}

func bitsClearU64(a, b uint64) (n uint64) {
	// amd64:"BTRQ"
	// arm64:"MOVD [$]1," "LSL" "BIC"
	// loong64:"MOVV [$]1," "SLLV" "ANDN"
	// riscv64:"ANDI" "SLL" "ANDN"
	n += b &^ (1 << (a & 63))

	// amd64:"BTRQ [$]63,"
	// arm64:"AND [$]9223372036854775807,"
	// loong64:"MOVV [$]" "AND "
	// riscv64:"MOV [$]" "AND "
	n += a &^ (1 << 63)

	// amd64:"BTRQ [$]60,"
	// arm64:"AND [$]-1152921504606846977,"
	// loong64:"MOVV [$]" "AND "
	// riscv64:"MOV [$]" "AND "
	n += a &^ (1 << 60)

	// amd64:"ANDQ [$]-2"
	// arm64:"AND [$]-2"
	// loong64:"AND [$]-2"
	// riscv64:"ANDI [$]-2"
	n += a &^ (1 << 0)

	return n
}

func bitsClearLowest(x int64, y int32) (int64, int32) {
	// amd64:"ANDQ [$]-2,"
	// arm64:"AND [$]-2,"
	// loong64:"AND [$]-2,"
	// riscv64:"ANDI [$]-2,"
	a := (x >> 1) << 1

	// amd64:"ANDL [$]-2,"
	// arm64:"AND [$]-2,"
	// loong64:"AND [$]-2,"
	// riscv64:"ANDI [$]-2,"
	b := (y >> 1) << 1

	return a, b
}

func bitsFlipU64(a, b uint64) (n uint64) {
	// amd64:"BTCQ"
	// arm64:"MOVD [$]1," "LSL" "EOR"
	// loong64:"MOVV [$]1," "SLLV" "XOR"
	// riscv64:"ANDI" "SLL" "XOR "
	n += b ^ (1 << (a & 63))

	// amd64:"BTCQ [$]63,"
	// arm64:"EOR [$]-9223372036854775808,"
	// loong64:"MOVV [$]" "XOR"
	// riscv64:"MOV [$]" "XOR "
	n += a ^ (1 << 63)

	// amd64:"BTCQ [$]60,"
	// arm64:"EOR [$]1152921504606846976,"
	// loong64:"MOVV [$]" "XOR"
	// riscv64:"MOV [$]" "XOR "
	n += a ^ (1 << 60)

	// amd64:"XORQ [$]1,"
	// arm64:"EOR [$]1,"
	// loong64:"XOR [$]1,"
	// riscv64:"XORI [$]1,"
	n += a ^ (1 << 0)

	return n
}

//
// 32 bit instructions
//

func bitsCheckConstShiftLeftU32(a uint32) (n int) {
	// amd64:"BTL [$]31,"
	// arm64:"TBNZ [$]31,"
	// loong64:"AND [$]" "MOVWU" "BNE"
	// riscv64:"MOV [$]" "AND" "BNEZ"
	if a&(1<<31) != 0 {
		return 1
	}
	// amd64:"BTL [$]28,"
	// arm64:"TBNZ [$]28,"
	// loong64:"AND [$]" "BNE"
	// riscv64:"ANDI" "BNEZ"
	if a&(1<<28) != 0 {
		return 1
	}
	// amd64:"BTL [$]0,"
	// arm64:"TBZ [$]0,"
	// loong64:"AND [$]" "BEQ"
	// riscv64:"ANDI" "BEQZ"
	if a&(1<<0) != 0 {
		return 1
	}
	return 0
}

func bitsCheckConstRightShiftU32(a [8]uint32) (n int) {
	// amd64:"BTL [$]31,"
	// arm64:"UBFX [$]31," "CBNZW"
	// loong64:"SRL [$]31," "AND [$]1," "BNE"
	// riscv64:"SRLI" "ANDI" "BNEZ"
	if (a[0]>>31)&1 != 0 {
		return 1
	}
	// amd64:"BTL [$]31,"
	// arm64:"UBFX [$]31," "CBNZW"
	// loong64:"SRL [$]31," "MOVWU" "BNE"
	// riscv64:"SRLI" "BNEZ"
	if a[1]>>31 != 0 {
		return 1
	}
	// amd64:"BTL [$]31,"
	// arm64:"UBFX [$]31," "CBZW"
	// loong64:"SRL [$]31," "MOVWU" "BEQ"
	// riscv64:"SRLI" "BEQZ"
	if a[2]>>31 == 0 {
		return 1
	}
	// amd64:"BTL [$]28,"
	// arm64:"UBFX [$]28," "TBZ"
	// loong64:"SRL [$]28," "AND [$]1," "BEQ"
	// riscv64:"SRLI" "ANDI" "BEQZ"
	if (a[3]>>28)&1 == 0 {
		return 1
	}
	// amd64:"BTL [$]1,"
	// arm64:"UBFX [$]1," "TBZ"
	// loong64:"SRL [$]1," "AND [$]1," "BEQ"
	// riscv64:"SRLI" "ANDI" "BEQZ"
	if (a[4]>>1)&1 == 0 {
		return 1
	}
	// amd64:"BTL [$]0,"
	// arm64:"TBZ" -"UBFX" -"SRL"
	// loong64:"AND [$]1," "BEQ" -"SRL [$]"
	// riscv64:"ANDI" "BEQZ" -"SRLI "
	if (a[5]>>0)&1 == 0 {
		return 1
	}
	// amd64:"BTL [$]7,"
	// arm64:"UBFX [$]5," "TBNZ"
	// loong64:"SRL [$]5," "AND [$]4," "BNE"
	// riscv64:"SRLI" "ANDI" "BNEZ"
	if (a[6]>>5)&4 == 0 {
		return 1
	}
	return 0
}

func bitsCheckVarU32(a, b uint32) (n int) {
	// amd64:"BTL"
	// arm64:"AND [$]31," "MOVD [$]1," "LSL" "TSTW"
	// loong64:"MOVV [$]1," "SLL R" "AND R" "MOVWU" "BNE"
	// riscv64:"ANDI [$]31," "SLL " "AND "
	if a&(1<<(b&31)) != 0 {
		return 1
	}
	// amd64:"BTL" -"BT. [$]0"
	// arm64:"AND [$]31," "LSR" "TBZ"
	// loong64:"SRL R" "AND [$]1," "BEQ"
	// riscv64:"ANDI [$]31," "SRLW " "ANDI [$]1,"
	if (b>>(a&31))&1 != 0 {
		return 1
	}
	return 0
}

func bitsCheckMaskU32(a uint32) (n int) {
	// amd64:"BTL [$]31,"
	// arm64:"TBNZ [$]31,"
	// loong64:"AND [$]" "MOVWU" "BNE"
	// riscv64:"MOV [$]" "AND" "BNEZ"
	if a&0x80000000 != 0 {
		return 1
	}
	// amd64:"BTL [$]27,"
	// arm64:"TBNZ [$]27,"
	// loong64:"AND [$]" "BNE"
	// riscv64:"ANDI" "BNEZ"
	if a&0x8000000 != 0 {
		return 1
	}
	// amd64:"BTL [$]0,"
	// arm64:"TBZ [$]0,"
	// loong64:"AND [$]1," "BEQ"
	// riscv64:"ANDI" "BEQZ"
	if a&0x1 != 0 {
		return 1
	}
	return 0
}

func bitsSetU32(a, b uint32) (n uint32) {
	// amd64:"BTSL"
	// arm64:"AND [$]31," "MOVD [$]1," "LSL" "ORR"
	// loong64:"MOVV [$]1," "SLL " "OR "
	// riscv64:"ANDI" "SLL" "OR"
	n += b | (1 << (a & 31))

	// amd64:"ORL [$]-2147483648,"
	// arm64:"ORR [$]-2147483648,"
	// loong64:"OR [$]-2147483648,"
	// riscv64:"ORI [$]-2147483648,"
	n += a | (1 << 31)

	// amd64:"ORL [$]268435456,"
	// arm64:"ORR [$]268435456,"
	// loong64:"OR [$]268435456,"
	// riscv64:"ORI [$]268435456,"
	n += a | (1 << 28)

	// amd64:"ORL [$]1,"
	// arm64:"ORR [$]1,"
	// loong64:"OR [$]1,"
	// riscv64:"ORI [$]1,"
	n += a | (1 << 0)

	return n
}

func bitsClearU32(a, b uint32) (n uint32) {
	// amd64:"BTRL"
	// arm64:"AND [$]31," "MOVD [$]1," "LSL" "BIC"
	// loong64:"MOVV [$]1," "SLL R" "ANDN"
	// riscv64:"ANDI" "SLL" "ANDN"
	n += b &^ (1 << (a & 31))

	// amd64:"ANDL [$]2147483647,"
	// arm64:"AND [$]2147483647,"
	// loong64:"AND [$]2147483647,"
	// riscv64:"ANDI [$]2147483647,"
	n += a &^ (1 << 31)

	// amd64:"ANDL [$]-268435457,"
	// arm64:"AND [$]-268435457,"
	// loong64:"AND [$]-268435457,"
	// riscv64:"ANDI [$]-268435457,"
	n += a &^ (1 << 28)

	// amd64:"ANDL [$]-2,"
	// arm64:"AND [$]-2,"
	// loong64:"AND [$]-2,"
	// riscv64:"ANDI [$]-2,"
	n += a &^ (1 << 0)

	return n
}

func bitsFlipU32(a, b uint32) (n uint32) {
	// amd64:"BTCL"
	// arm64:"AND [$]31," "MOVD [$]1," "LSL" "EOR"
	// loong64:"MOVV [$]1," "SLL R" "XOR"
	// riscv64:"ANDI" "SLL" "XOR "
	n += b ^ (1 << (a & 31))

	// amd64:"XORL [$]-2147483648,"
	// arm64:"EOR [$]-2147483648,"
	// loong64:"XOR [$]-2147483648,"
	// riscv64:"XORI [$]-2147483648,"
	n += a ^ (1 << 31)

	// amd64:"XORL [$]268435456,"
	// arm64:"EOR [$]268435456,"
	// loong64:"XOR [$]268435456,"
	// riscv64:"XORI [$]268435456,"
	n += a ^ (1 << 28)

	// amd64:"XORL [$]1,"
	// arm64:"EOR [$]1,"
	// loong64:"XOR [$]1,"
	// riscv64:"XORI [$]1,"
	n += a ^ (1 << 0)

	return n
}

func bitsOpOnMem(a []uint32, b, c, d uint32) {
	// check direct operation on memory with constant

	// amd64:`ANDL\s[$]200,\s\([A-Z][A-Z0-9]+\)`
	a[0] &= 200
	// amd64:`ORL\s[$]220,\s4\([A-Z][A-Z0-9]+\)`
	a[1] |= 220
	// amd64:`XORL\s[$]240,\s8\([A-Z][A-Z0-9]+\)`
	a[2] ^= 240
}

func bitsCheckMostNegative(b uint8) bool {
	// amd64:"TESTB"
	// arm64:"TSTW" "CSET"
	// loong64:"AND [$]128," "SGTU"
	// riscv64:"ANDI [$]128," "SNEZ" -"ADDI"
	return b&0x80 == 0x80
}

func bitsIssue19857a(a uint64) uint64 {
	// arm64:`AND `
	return a & ((1 << 63) - 1)
}

func bitsIssue19857b(a uint64) uint64 {
	// arm64:`AND `
	return a & (1 << 63)
}

func bitsIssue19857c(a, b uint32) (uint32, uint32) {
	// arm/7:`BIC`,-`AND`
	a &= 0xffffaaaa
	// arm/7:`BFC`,-`AND`,-`BIC`
	b &= 0xffc003ff
	return a, b
}

func bitsAndNot(x, y uint32) uint32 {
	// arm64:`BIC `,-`AND`
	// loong64:"ANDN " -"AND "
	// riscv64:"ANDN" -"AND "
	return x &^ y
}

func bitsXorNot(x, y, z uint32, a []uint32, n, m uint64) uint64 {
	// arm64:`EON `,-`EOR`,-`MVN`
	// loong64:"NOR" "XOR"
	// riscv64:"XNOR " -"MOV [$]" -"XOR"
	a[0] = x ^ (y ^ 0xffffffff)

	// arm64:`EON `,-`EOR`,-`MVN`
	// loong64:"XOR" "NOR"
	// riscv64:"XNOR" -"XOR"
	a[1] = ^(y ^ z)

	// arm64:`EON `,-`XOR`
	// loong64:"NOR" "XOR"
	// riscv64:"XNOR" -"XOR" -"NOT"
	a[2] = x ^ ^z

	// arm64:`EON `,-`EOR`,-`MVN`
	// loong64:"NOR" "XOR"
	// riscv64:"XNOR" -"MOV [$]" -"XOR"
	return n ^ (m ^ 0xffffffffffffffff)
}

func bitsOrNot(x, y uint32) uint32 {
	// arm64:"ORN " -"ORR"
	// loong64:"ORN" -"OR "
	// riscv64:"ORN" -"OR "
	return x | ^y
}

func bitsNotOr(x int64, a []int64) {
	// loong64: "MOVV [$]0" "NOR R"
	a[0] = ^(0x1234 | x)
	// loong64:"NOR" -"XOR"
	a[1] = (-1) ^ x
	// loong64: "MOVV [$]-55" -"OR" -"NOR"
	a[2] = ^(0x12 | 0x34)
}

func bitsSetPowerOf2Test(x int) bool {
	// amd64:"BTL [$]3"
	// loong64:"AND [$]8," "SGTU"
	// riscv64:"ANDI [$]8," "SNEZ" -"ADDI"
	return x&8 == 8
}

func bitsSetTest(x int) bool {
	// amd64:"ANDL [$]9, AX"
	// amd64:"CMPQ AX, [$]9"
	// loong64:"AND [$]9," "XOR" "SGTU"
	// riscv64:"ANDI [$]9," "ADDI [$]-9," "SEQZ"
	return x&9 == 9
}

func bitsMaskContiguousOnes64U(x uint64) uint64 {
	// s390x:"RISBGZ [$]16, [$]47, [$]0,"
	return x & 0x0000ffffffff0000
}

func bitsMaskContiguousZeroes64U(x uint64) uint64 {
	// s390x:"RISBGZ [$]48, [$]15, [$]0,"
	return x & 0xffff00000000ffff
}

func bitsIssue44228a(a []int64, i int) bool {
	// amd64: "BTQ", -"SHL"
	return a[i>>6]&(1<<(i&63)) != 0
}

func bitsIssue44228b(a []int32, i int) bool {
	// amd64: "BTL", -"SHL"
	return a[i>>5]&(1<<(i&31)) != 0
}

func bitsIssue48467(x, y uint64) uint64 {
	// arm64: -"NEG"
	d, borrow := bits.Sub64(x, y, 0)
	return x - d&(-borrow)
}

func bitsFoldConst(x, y uint64) uint64 {
	// arm64: "ADDS [$]7" -"MOVD [$]7"
	// ppc64x: "ADDC [$]7,"
	d, b := bits.Add64(x, 7, 0)
	return b & d
}

func bitsFoldConstOutOfRange(a uint64) uint64 {
	// arm64: "MOVD [$]19088744" -"ADD [$]19088744"
	return a + 0x1234568
}

func bitsSignExtendAndMask8to64U(a int8) (s, z uint64) {
	// Verify sign-extended values are not zero-extended under a bit mask (#61297)

	// ppc64x: "MOVB", "ANDCC [$]1015,"
	s = uint64(a) & 0x3F7
	// ppc64x: -"MOVB", "ANDCC [$]247,"
	z = uint64(uint8(a)) & 0x3F7
	return
}

func bitsZeroExtendAndMask8toU64(a int8, b int16) (x, y uint64) {
	// Verify zero-extended values are not sign-extended under a bit mask (#61297)

	// ppc64x: -"MOVB ", -"ANDCC", "MOVBZ"
	x = uint64(a) & 0xFF
	// ppc64x: -"MOVH ", -"ANDCC", "MOVHZ"
	y = uint64(b) & 0xFFFF
	return
}

func bitsRotateAndMask(io64 [8]uint64, io32 [4]uint32, io16 [4]uint16, io8 [4]uint8) {
	// Verify rotate and mask instructions, and further simplified instructions for small types

	// ppc64x: "RLDICR [$]0, R[0-9]*, [$]47, R"
	io64[0] = io64[0] & 0xFFFFFFFFFFFF0000
	// ppc64x: "RLDICL [$]0, R[0-9]*, [$]16, R"
	io64[1] = io64[1] & 0x0000FFFFFFFFFFFF
	// ppc64x: -"SRD", -"AND", "RLDICL [$]60, R[0-9]*, [$]16, R"
	io64[2] = (io64[2] >> 4) & 0x0000FFFFFFFFFFFF
	// ppc64x: -"SRD", -"AND", "RLDICL [$]36, R[0-9]*, [$]28, R"
	io64[3] = (io64[3] >> 28) & 0x0000FFFFFFFFFFFF

	// ppc64x: "MOVWZ", "RLWNM [$]1, R[0-9]*, [$]28, [$]3, R"
	io64[4] = uint64(bits.RotateLeft32(io32[0], 1) & 0xF000000F)

	// ppc64x: "RLWNM [$]0, R[0-9]*, [$]4, [$]19, R"
	io32[0] = io32[0] & 0x0FFFF000
	// ppc64x: "RLWNM [$]0, R[0-9]*, [$]20, [$]3, R"
	io32[1] = io32[1] & 0xF0000FFF
	// ppc64x: -"RLWNM", MOVD, AND
	io32[2] = io32[2] & 0xFFFF0002

	var bigc uint32 = 0x12345678
	// ppc64x: "ANDCC [$]22136"
	io16[0] = io16[0] & uint16(bigc)

	// ppc64x: "ANDCC [$]120"
	io8[0] = io8[0] & uint8(bigc)
}

func bitsOpXor1(x, y uint32) uint32 {
	// arm64: "ORR" "AND" "BIC"
	// loong64: "OR " "AND " "ANDN"
	return (x | y) &^ (x & y)
}

func bitsOpXor2(x, y uint32) uint32 {
	// arm64: "BIC" "ORR"
	// loong64: "ANDN" "OR "
	return (x &^ y) | (^x & y)
}
