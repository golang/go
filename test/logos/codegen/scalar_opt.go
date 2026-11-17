// asmcheck -O=3

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func maskedAdd(x, y uint64) uint64 {
	// amd64:"MOVL [$]37, AX" -"ANDQ" -"ADDQ" -"LEAQ"
	// arm64:"MOVD [$]37, R0" -"AND" -"ADD"
	return ((x &^ 255) + (y &^ 255) + 37) & 255
}

func maskedSub(x, y uint64) uint64 {
	// amd64:"MOVL [$]219, AX" -"SUBQ" -"ANDQ" -"LEAQ"
	// arm64:"MOVD [$]219, R0" -"SUB" -"AND"
	return ((x &^ 255) - (y &^ 255) - 37) & 255
}

func maskedNeg(x uint64) uint64 {
	// amd64:"MOVL [$]37, AX" -"NEGQ" -"ANDQ" -"LEAQ"
	// arm64:"MOVD [$]37, R0" -"NEG" -"AND"
	return (-(x &^ 255) + 37) & 255
}

func rotatedField(x uint64) uint64 {
	x &= 255
	// amd64:-"ROLQ" -"RORX" -"MOVBLZX"
	// arm64:-"ROR" -"AND"
	return (x<<8 | x>>56) & 255
}

func correlated(flag bool) uint64 {
	var x, y uint64
	if flag {
		x, y = 1, 3
	} else {
		x, y = 2, 2
	}
	// amd64:"MOVL [$]4, AX" -"ADDQ"
	// arm64:"MOVD [$]4, R0" -"ADD"
	return x + y
}

func dynamicAdd(x, y uint64) uint64 {
	// amd64:"LEAQ|ADDQ"
	// arm64:"ADD"
	return ((x &^ 255) + y + 37) & 255
}

func projectedHeader(key uint32, sequence uint16) uint64 {
	native := uint64(key)<<16 | uint64(sequence)
	// amd64:-"ROLQ" -"RORX" -"SHRQ"
	// arm64:-"ROR" -"LSR"
	return (native<<16 | native>>48) >> 16
}
