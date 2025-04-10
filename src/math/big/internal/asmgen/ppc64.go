// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asmgen

var ArchPPC64x = &Arch{
	Name:          "ppc64x",
	Build:         "ppc64 || ppc64le",
	WordBits:      64,
	WordBytes:     8,
	CarrySafeLoop: true,

	// Note: The old, hand-written ppc64x assembly used MOVDU
	// to avoid explicit pointer updates in a few routines, but the new
	// generated code runs just as fast, so we haven't bothered to try
	// to add that back. (It's not trivial; you'd have to keep the pointers
	// shifted one word in order to make the semantics work.)
	//
	// The old assembly also used some complex vector instructions
	// to implement lshVU and rshVU, but the generated code that uses
	// ordinary integer instructions is much faster than the vector code was,
	// at least on the power10 gomote.

	regs: []string{
		// R0 is 0 by convention.
		// R1 is SP.
		// R2 is TOC.
		// R30 is g.
		// R31 is the assembler/linker temporary (which we use too).
		"R3", "R4", "R5", "R6", "R7", "R8", "R9",
		"R10", "R11", "R12" /*R13 is TLS*/, "R14", "R15", "R16", "R17", "R18", "R19",
		"R20", "R21", "R22", "R23", "R24", "R25", "R26", "R27", "R28", "R29",
	},
	reg0:   "R0",
	regTmp: "R31",

	// Note: Could write an addF and subF to use ADDZE and SUBZE,
	// but we have R0 so it doesn't seem to matter much.

	mov:   "MOVD",
	add:   "ADD",
	adds:  "ADDC",
	adcs:  "ADDE",
	sub:   "SUB",
	subs:  "SUBC",
	sbcs:  "SUBE",
	mul:   "MULLD",
	mulhi: "MULHDU",
	lsh:   "SLD",
	rsh:   "SRD",
	and:   "ANDCC", // regular AND does not accept immediates
	or:    "OR",
	xor:   "XOR",

	jmpZero:    "CMP %[1]s, $0; BEQ %[2]s",
	jmpNonZero: "CMP %s, $0; BNE %s",

	// Note: Using CTR means that we could free the count register
	// during the loop body, but the portable logic doesn't know that,
	// and we're not hurting for registers.
	loopTop:    "CMP %[1]s, $0; BEQ %[2]s; MOVD %[1]s, CTR",
	loopBottom: "BDNZ %[2]s",
}
