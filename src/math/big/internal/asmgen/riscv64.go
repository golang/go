// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asmgen

var ArchRISCV64 = &Arch{
	Name:          "riscv64",
	WordBits:      64,
	WordBytes:     8,
	CarrySafeLoop: true,

	regs: []string{
		// X0 is zero.
		// X1 is LR.
		// X2 is SP.
		// X3 is SB.
		// X4 is TP.
		// X27 is g.
		// X28 and X29 are our virtual carry flags.
		// X31 is the assembler/linker temporary (which we use too).
		"X5", "X6", "X7", "X8", "X9",
		"X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19",
		"X20", "X21", "X22", "X23", "X24", "X25", "X26",
		"X30",
	},

	reg0:        "X0",
	regCarry:    "X28",
	regAltCarry: "X29",
	regTmp:      "X31",

	mov:   "MOV",
	add:   "ADD",
	sub:   "SUB",
	mul:   "MUL",
	mulhi: "MULHU",
	lsh:   "SLL",
	rsh:   "SRL",
	and:   "AND",
	or:    "OR",
	xor:   "XOR",
	sltu:  "SLTU",

	jmpZero:    "BEQZ %s, %s",
	jmpNonZero: "BNEZ %s, %s",
}
