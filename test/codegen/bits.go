// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "math/bits"

/************************************
 * 64-bit instructions
 ************************************/

func bitcheck64_constleft(a uint64) (n int) {
	// amd64:"BTQ\t[$]63"
	if a&(1<<63) != 0 {
		return 1
	}
	// amd64:"BTQ\t[$]60"
	if a&(1<<60) != 0 {
		return 1
	}
	// amd64:"BTL\t[$]0"
	if a&(1<<0) != 0 {
		return 1
	}
	return 0
}

func bitcheck64_constright(a [8]uint64) (n int) {
	// amd64:"BTQ\t[$]63"
	if (a[0]>>63)&1 != 0 {
		return 1
	}
	// amd64:"BTQ\t[$]63"
	if a[1]>>63 != 0 {
		return 1
	}
	// amd64:"BTQ\t[$]63"
	if a[2]>>63 == 0 {
		return 1
	}
	// amd64:"BTQ\t[$]60"
	if (a[3]>>60)&1 == 0 {
		return 1
	}
	// amd64:"BTL\t[$]1"
	if (a[4]>>1)&1 == 0 {
		return 1
	}
	// amd64:"BTL\t[$]0"
	if (a[5]>>0)&1 == 0 {
		return 1
	}
	// amd64:"BTL\t[$]7"
	if (a[6]>>5)&4 == 0 {
		return 1
	}
	return 0
}

func bitcheck64_var(a, b uint64) (n int) {
	// amd64:"BTQ"
	if a&(1<<(b&63)) != 0 {
		return 1
	}
	// amd64:"BTQ",-"BT.\t[$]0"
	if (b>>(a&63))&1 != 0 {
		return 1
	}
	return 0
}

func bitcheck64_mask(a uint64) (n int) {
	// amd64:"BTQ\t[$]63"
	if a&0x8000000000000000 != 0 {
		return 1
	}
	// amd64:"BTQ\t[$]59"
	if a&0x800000000000000 != 0 {
		return 1
	}
	// amd64:"BTL\t[$]0"
	if a&0x1 != 0 {
		return 1
	}
	return 0
}

func biton64(a, b uint64) (n uint64) {
	// amd64:"BTSQ"
	n += b | (1 << (a & 63))

	// amd64:"BTSQ\t[$]63"
	n += a | (1 << 63)

	// amd64:"BTSQ\t[$]60"
	n += a | (1 << 60)

	// amd64:"ORQ\t[$]1"
	n += a | (1 << 0)

	return n
}

func bitoff64(a, b uint64) (n uint64) {
	// amd64:"BTRQ"
	n += b &^ (1 << (a & 63))

	// amd64:"BTRQ\t[$]63"
	n += a &^ (1 << 63)

	// amd64:"BTRQ\t[$]60"
	n += a &^ (1 << 60)

	// amd64:"ANDQ\t[$]-2"
	n += a &^ (1 << 0)

	return n
}

func bitcompl64(a, b uint64) (n uint64) {
	// amd64:"BTCQ"
	n += b ^ (1 << (a & 63))

	// amd64:"BTCQ\t[$]63"
	n += a ^ (1 << 63)

	// amd64:"BTCQ\t[$]60"
	n += a ^ (1 << 60)

	// amd64:"XORQ\t[$]1"
	n += a ^ (1 << 0)

	return n
}

/************************************
 * 32-bit instructions
 ************************************/

func bitcheck32_constleft(a uint32) (n int) {
	// amd64:"BTL\t[$]31"
	if a&(1<<31) != 0 {
		return 1
	}
	// amd64:"BTL\t[$]28"
	if a&(1<<28) != 0 {
		return 1
	}
	// amd64:"BTL\t[$]0"
	if a&(1<<0) != 0 {
		return 1
	}
	return 0
}

func bitcheck32_constright(a [8]uint32) (n int) {
	// amd64:"BTL\t[$]31"
	if (a[0]>>31)&1 != 0 {
		return 1
	}
	// amd64:"BTL\t[$]31"
	if a[1]>>31 != 0 {
		return 1
	}
	// amd64:"BTL\t[$]31"
	if a[2]>>31 == 0 {
		return 1
	}
	// amd64:"BTL\t[$]28"
	if (a[3]>>28)&1 == 0 {
		return 1
	}
	// amd64:"BTL\t[$]1"
	if (a[4]>>1)&1 == 0 {
		return 1
	}
	// amd64:"BTL\t[$]0"
	if (a[5]>>0)&1 == 0 {
		return 1
	}
	// amd64:"BTL\t[$]7"
	if (a[6]>>5)&4 == 0 {
		return 1
	}
	return 0
}

func bitcheck32_var(a, b uint32) (n int) {
	// amd64:"BTL"
	if a&(1<<(b&31)) != 0 {
		return 1
	}
	// amd64:"BTL",-"BT.\t[$]0"
	if (b>>(a&31))&1 != 0 {
		return 1
	}
	return 0
}

func bitcheck32_mask(a uint32) (n int) {
	// amd64:"BTL\t[$]31"
	if a&0x80000000 != 0 {
		return 1
	}
	// amd64:"BTL\t[$]27"
	if a&0x8000000 != 0 {
		return 1
	}
	// amd64:"BTL\t[$]0"
	if a&0x1 != 0 {
		return 1
	}
	return 0
}

func biton32(a, b uint32) (n uint32) {
	// amd64:"BTSL"
	n += b | (1 << (a & 31))

	// amd64:"ORL\t[$]-2147483648"
	n += a | (1 << 31)

	// amd64:"ORL\t[$]268435456"
	n += a | (1 << 28)

	// amd64:"ORL\t[$]1"
	n += a | (1 << 0)

	return n
}

func bitoff32(a, b uint32) (n uint32) {
	// amd64:"BTRL"
	n += b &^ (1 << (a & 31))

	// amd64:"ANDL\t[$]2147483647"
	n += a &^ (1 << 31)

	// amd64:"ANDL\t[$]-268435457"
	n += a &^ (1 << 28)

	// amd64:"ANDL\t[$]-2"
	n += a &^ (1 << 0)

	return n
}

func bitcompl32(a, b uint32) (n uint32) {
	// amd64:"BTCL"
	n += b ^ (1 << (a & 31))

	// amd64:"XORL\t[$]-2147483648"
	n += a ^ (1 << 31)

	// amd64:"XORL\t[$]268435456"
	n += a ^ (1 << 28)

	// amd64:"XORL\t[$]1"
	n += a ^ (1 << 0)

	return n
}

// check direct operation on memory with constant and shifted constant sources
func bitOpOnMem(a []uint32, b, c, d uint32) {
	// amd64:`ANDL\s[$]200,\s\([A-Z][A-Z0-9]+\)`
	a[0] &= 200
	// amd64:`ORL\s[$]220,\s4\([A-Z][A-Z0-9]+\)`
	a[1] |= 220
	// amd64:`XORL\s[$]240,\s8\([A-Z][A-Z0-9]+\)`
	a[2] ^= 240
}

func bitcheckMostNegative(b uint8) bool {
	// amd64:"TESTB"
	return b&0x80 == 0x80
}

// Check AND masking on arm64 (Issue #19857)

func and_mask_1(a uint64) uint64 {
	// arm64:`AND\t`
	return a & ((1 << 63) - 1)
}

func and_mask_2(a uint64) uint64 {
	// arm64:`AND\t`
	return a & (1 << 63)
}

func and_mask_3(a, b uint32) (uint32, uint32) {
	// arm/7:`BIC`,-`AND`
	a &= 0xffffaaaa
	// arm/7:`BFC`,-`AND`,-`BIC`
	b &= 0xffc003ff
	return a, b
}

// Check generation of arm64 BIC/EON/ORN instructions

func op_bic(x, y uint32) uint32 {
	// arm64:`BIC\t`,-`AND`
	return x &^ y
}

func op_eon(x, y, z uint32, a []uint32, n, m uint64) uint64 {
	// arm64:`EON\t`,-`EOR`,-`MVN`
	a[0] = x ^ (y ^ 0xffffffff)

	// arm64:`EON\t`,-`EOR`,-`MVN`
	a[1] = ^(y ^ z)

	// arm64:`EON\t`,-`XOR`
	a[2] = x ^ ^z

	// arm64:`EON\t`,-`EOR`,-`MVN`
	return n ^ (m ^ 0xffffffffffffffff)
}

func op_orn(x, y uint32) uint32 {
	// arm64:`ORN\t`,-`ORR`
	return x | ^y
}

// check bitsets
func bitSetPowerOf2Test(x int) bool {
	// amd64:"BTL\t[$]3"
	return x&8 == 8
}

func bitSetTest(x int) bool {
	// amd64:"ANDL\t[$]9, AX"
	// amd64:"CMPQ\tAX, [$]9"
	return x&9 == 9
}

// mask contiguous one bits
func cont1Mask64U(x uint64) uint64 {
	// s390x:"RISBGZ\t[$]16, [$]47, [$]0,"
	return x & 0x0000ffffffff0000
}

// mask contiguous zero bits
func cont0Mask64U(x uint64) uint64 {
	// s390x:"RISBGZ\t[$]48, [$]15, [$]0,"
	return x & 0xffff00000000ffff
}

func issue44228a(a []int64, i int) bool {
	// amd64: "BTQ", -"SHL"
	return a[i>>6]&(1<<(i&63)) != 0
}
func issue44228b(a []int32, i int) bool {
	// amd64: "BTL", -"SHL"
	return a[i>>5]&(1<<(i&31)) != 0
}

func issue48467(x, y uint64) uint64 {
	// arm64: -"NEG"
	d, borrow := bits.Sub64(x, y, 0)
	return x - d&(-borrow)
}

func foldConst(x, y uint64) uint64 {
	// arm64: "ADDS\t[$]7",-"MOVD\t[$]7"
	d, b := bits.Add64(x, 7, 0)
	return b & d
}

func foldConstOutOfRange(a uint64) uint64 {
	// arm64: "MOVD\t[$]19088744",-"ADD\t[$]19088744"
	return a + 0x1234568
}

// Verify sign-extended values are not zero-extended under a bit mask (#61297)
func signextendAndMask8to64(a int8) (s, z uint64) {
	// ppc64x: "MOVB", "ANDCC\t[$]1015,"
	s = uint64(a) & 0x3F7
	// ppc64x: -"MOVB", "ANDCC\t[$]247,"
	z = uint64(uint8(a)) & 0x3F7
	return
}

// Verify zero-extended values are not sign-extended under a bit mask (#61297)
func zeroextendAndMask8to64(a int8, b int16) (x, y uint64) {
	// ppc64x: -"MOVB\t", -"ANDCC", "MOVBZ"
	x = uint64(a) & 0xFF
	// ppc64x: -"MOVH\t", -"ANDCC", "MOVHZ"
	y = uint64(b) & 0xFFFF
	return
}

// Verify rotate and mask instructions, and further simplified instructions for small types
func bitRotateAndMask(io64 [4]uint64, io32 [4]uint32, io16 [4]uint16, io8 [4]uint8) {
	// ppc64x: "RLDICR\t[$]0, R[0-9]*, [$]47, R"
	io64[0] = io64[0] & 0xFFFFFFFFFFFF0000
	// ppc64x: "RLDICL\t[$]0, R[0-9]*, [$]16, R"
	io64[1] = io64[1] & 0x0000FFFFFFFFFFFF
	// ppc64x: -"SRD", -"AND", "RLDICL\t[$]60, R[0-9]*, [$]16, R"
	io64[2] = (io64[2] >> 4) & 0x0000FFFFFFFFFFFF
	// ppc64x: -"SRD", -"AND", "RLDICL\t[$]36, R[0-9]*, [$]28, R"
	io64[3] = (io64[3] >> 28) & 0x0000FFFFFFFFFFFF

	// ppc64x: "RLWNM\t[$]0, R[0-9]*, [$]4, [$]19, R"
	io32[0] = io32[0] & 0x0FFFF000
	// ppc64x: "RLWNM\t[$]0, R[0-9]*, [$]20, [$]3, R"
	io32[1] = io32[1] & 0xF0000FFF
	// ppc64x: -"RLWNM", MOVD, AND
	io32[2] = io32[2] & 0xFFFF0002

	var bigc uint32 = 0x12345678
	// ppc64x: "ANDCC\t[$]22136"
	io16[0] = io16[0] & uint16(bigc)

	// ppc64x: "ANDCC\t[$]120"
	io8[0] = io8[0] & uint8(bigc)
}
