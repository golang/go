// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

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

	// amd64:"BTSL\t[$]31"
	n += a | (1 << 31)

	// amd64:"BTSL\t[$]28"
	n += a | (1 << 28)

	// amd64:"ORL\t[$]1"
	n += a | (1 << 0)

	return n
}

func bitoff32(a, b uint32) (n uint32) {
	// amd64:"BTRL"
	n += b &^ (1 << (a & 31))

	// amd64:"BTRL\t[$]31"
	n += a &^ (1 << 31)

	// amd64:"BTRL\t[$]28"
	n += a &^ (1 << 28)

	// amd64:"ANDL\t[$]-2"
	n += a &^ (1 << 0)

	return n
}

func bitcompl32(a, b uint32) (n uint32) {
	// amd64:"BTCL"
	n += b ^ (1 << (a & 31))

	// amd64:"BTCL\t[$]31"
	n += a ^ (1 << 31)

	// amd64:"BTCL\t[$]28"
	n += a ^ (1 << 28)

	// amd64:"XORL\t[$]1"
	n += a ^ (1 << 0)

	return n
}

// check direct operation on memory with constant and shifted constant sources
func bitOpOnMem(a []uint32, b, c, d uint32) {
	// amd64:`ANDL\s[$]200,\s\([A-Z]+\)`
	a[0] &= 200
	// amd64:`ORL\s[$]220,\s4\([A-Z]+\)`
	a[1] |= 220
	// amd64:`XORL\s[$]240,\s8\([A-Z]+\)`
	a[2] ^= 240
	// amd64:`BTRL\s[$]15,\s12\([A-Z]+\)`,-`ANDL`
	a[3] &= 0xffff7fff
	// amd64:`BTSL\s[$]14,\s16\([A-Z]+\)`,-`ORL`
	a[4] |= 0x4000
	// amd64:`BTCL\s[$]13,\s20\([A-Z]+\)`,-`XORL`
	a[5] ^= 0x2000
	// amd64:`BTRL\s[A-Z]+,\s24\([A-Z]+\)`
	a[6] &^= 1 << (b & 31)
	// amd64:`BTSL\s[A-Z]+,\s28\([A-Z]+\)`
	a[7] |= 1 << (c & 31)
	// amd64:`BTCL\s[A-Z]+,\s32\([A-Z]+\)`
	a[8] ^= 1 << (d & 31)
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
	// amd64:"ANDQ\t[$]9, AX"
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
