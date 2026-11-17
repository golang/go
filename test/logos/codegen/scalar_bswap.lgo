// asmcheck -O=3

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "math/bits"

func bswapLiteral64() uint64 {
	// amd64:-"BSWAPQ"
	// arm64:-"REV"
	// 386:-"BSWAPL"
	return bits.ReverseBytes64(0x0123456789abcdef)
}

func bswapLiteral32() uint32 {
	// amd64:"MOVL [$]1732584193, AX" -"BSWAPL"
	// arm64:-"REVW"
	// 386:-"BSWAPL"
	return bits.ReverseBytes32(0x01234567)
}

func bswapLiteral16() uint16 {
	// amd64:"MOVL [$]8961, AX" -"ROLW"
	// arm64:-"REV"
	// ppc64x/power10:"MOVD [$]8961," -"BRH"
	return bits.ReverseBytes16(0x0123)
}

func bswapMasked64(x uint64) uint64 {
	// amd64:"XORL AX, AX" -"BSWAPQ" -"ANDQ"
	// arm64:-"REV" -"AND"
	return bits.ReverseBytes64(x&0xffff) & 0x0000ffffffffffff
}

func bswapMasked32(x uint32) uint32 {
	// amd64:"XORL AX, AX" -"BSWAPL" -"ANDL"
	// arm64:-"REVW" -"AND"
	return bits.ReverseBytes32(x&0xff) & 0x00ffffff
}

func bswapWidened16(x uint16) uint64 {
	// ppc64x/power10:"BRH"
	return uint64(bits.ReverseBytes16(x))
}

func bswapWidenedPrefix16(x uint16) uint64 {
	// ReverseBytes16 returns an unsigned 16-bit value. Widening keeps
	// every higher bit zero, so the high prefix is independent of x.
	// ppc64x/power10:"MOVD [$]4660," -"BRH"
	return (uint64(bits.ReverseBytes16(x)) | uint64(0x1234)<<32) >> 32
}

func bswapLateDispatch(words []uint64) uint64 {
	state := uint64(1)
	var sum uint64
	for _, word := range words {
		state *= state
		// amd64:-"BSWAPQ" -"CMOVQ" -"CMPQ"
		// arm64:-"REV" -"CSEL" -"CMP"
		if bits.ReverseBytes64(state) == uint64(1)<<56 {
			sum += word
		} else {
			sum ^= word
		}
	}
	return sum
}

func bswapDynamic64(x uint64) uint64 {
	// amd64:"BSWAPQ"
	// arm64:"REV"
	return bits.ReverseBytes64(x)
}
