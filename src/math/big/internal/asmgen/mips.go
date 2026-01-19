// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asmgen

var ArchMIPS = &Arch{
	Name:          "mipsx",
	Build:         "mips || mipsle",
	WordBits:      32,
	WordBytes:     4,
	CarrySafeLoop: true,

	regs: []string{
		// R0 is 0
		// R23 is the assembler/linker temporary (which we use too).
		// R24 and R25 are our virtual carry flags.
		// R28 is SB.
		// R29 is SP.
		// R30 is g.
		// R31 is LR.
		"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9",
		"R10", "R11", "R12", "R13", "R14", "R15", "R16", "R17", "R18", "R19",
		"R20", "R21", "R22", "R24", "R25",
	},
	reg0:        "R0",
	regTmp:      "R23",
	regCarry:    "R24",
	regAltCarry: "R25",

	mov:      "MOVW",
	add:      "ADDU",
	sltu:     "SGTU", // SGTU args are swapped, so it's really SLTU
	sub:      "SUBU",
	mulWideF: mipsMulWide,
	lsh:      "SLL",
	rsh:      "SRL",
	and:      "AND",
	or:       "OR",
	xor:      "XOR",

	jmpZero:    "BEQ %s, %s",
	jmpNonZero: "BNE %s, %s",
}

func mipsMulWide(a *Asm, src1, src2, dstlo, dsthi Reg) {
	a.Printf("\tMULU %s, %s\n\tMOVW LO, %s\n\tMOVW HI, %s\n", src1, src2, dstlo, dsthi)
}
