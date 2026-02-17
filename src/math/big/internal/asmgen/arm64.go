// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asmgen

var ArchARM64 = &Arch{
	Name:          "arm64",
	WordBits:      64,
	WordBytes:     8,
	CarrySafeLoop: true,

	regs: []string{
		// R18 is the platform register.
		// R27 is the assembler/linker temporary (which we could potentially use but don't).
		// R28 is g.
		// R29 is FP.
		// R30 is LR.
		"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9",
		"R10", "R11", "R12", "R13", "R14", "R15", "R16", "R17", "R19",
		"R20", "R21", "R22", "R23", "R24", "R25", "R26",
	},
	reg0: "ZR",

	mov:   "MOVD",
	add:   "ADD",
	adds:  "ADDS",
	adc:   "ADC",
	adcs:  "ADCS",
	sub:   "SUB",
	subs:  "SUBS",
	sbc:   "SBC",
	sbcs:  "SBCS",
	mul:   "MUL",
	mulhi: "UMULH",
	lsh:   "LSL",
	rsh:   "LSR",
	and:   "AND",
	or:    "ORR",
	xor:   "EOR",

	addWords: "ADD %[1]s<<3, %[2]s, %[3]s",

	jmpZero:    "CBZ %s, %s",
	jmpNonZero: "CBNZ %s, %s",

	loadIncN:  arm64LoadIncN,
	loadDecN:  arm64LoadDecN,
	storeIncN: arm64StoreIncN,
	storeDecN: arm64StoreDecN,
}

func arm64LoadIncN(a *Asm, p RegPtr, regs []Reg) {
	if len(regs) == 1 {
		a.Printf("\tMOVD.P %d(%s), %s\n", a.Arch.WordBytes, p, regs[0])
		return
	}
	a.Printf("\tLDP.P %d(%s), (%s, %s)\n", len(regs)*a.Arch.WordBytes, p, regs[0], regs[1])
	var i int
	for i = 2; i+2 <= len(regs); i += 2 {
		a.Printf("\tLDP %d(%s), (%s, %s)\n", (i-len(regs))*a.Arch.WordBytes, p, regs[i], regs[i+1])
	}
	if i < len(regs) {
		a.Printf("\tMOVD %d(%s), %s\n", -1*a.Arch.WordBytes, p, regs[i])
	}
}

func arm64LoadDecN(a *Asm, p RegPtr, regs []Reg) {
	if len(regs) == 1 {
		a.Printf("\tMOVD.W -%d(%s), %s\n", a.Arch.WordBytes, p, regs[0])
		return
	}
	a.Printf("\tLDP.W %d(%s), (%s, %s)\n", -len(regs)*a.Arch.WordBytes, p, regs[len(regs)-1], regs[len(regs)-2])
	var i int
	for i = 2; i+2 <= len(regs); i += 2 {
		a.Printf("\tLDP %d(%s), (%s, %s)\n", i*a.Arch.WordBytes, p, regs[len(regs)-1-i], regs[len(regs)-2-i])
	}
	if i < len(regs) {
		a.Printf("\tMOVD %d(%s), %s\n", i*a.Arch.WordBytes, p, regs[0])
	}
}

func arm64StoreIncN(a *Asm, p RegPtr, regs []Reg) {
	if len(regs) == 1 {
		a.Printf("\tMOVD.P %s, %d(%s)\n", regs[0], a.Arch.WordBytes, p)
		return
	}
	a.Printf("\tSTP.P (%s, %s), %d(%s)\n", regs[0], regs[1], len(regs)*a.Arch.WordBytes, p)
	var i int
	for i = 2; i+2 <= len(regs); i += 2 {
		a.Printf("\tSTP (%s, %s), %d(%s)\n", regs[i], regs[i+1], (i-len(regs))*a.Arch.WordBytes, p)
	}
	if i < len(regs) {
		a.Printf("\tMOVD %s, %d(%s)\n", regs[i], -1*a.Arch.WordBytes, p)
	}
}

func arm64StoreDecN(a *Asm, p RegPtr, regs []Reg) {
	if len(regs) == 1 {
		a.Printf("\tMOVD.W %s, -%d(%s)\n", regs[0], a.Arch.WordBytes, p)
		return
	}
	a.Printf("\tSTP.W (%s, %s), %d(%s)\n", regs[len(regs)-1], regs[len(regs)-2], -len(regs)*a.Arch.WordBytes, p)
	var i int
	for i = 2; i+2 <= len(regs); i += 2 {
		a.Printf("\tSTP (%s, %s), %d(%s)\n", regs[len(regs)-1-i], regs[len(regs)-2-i], i*a.Arch.WordBytes, p)
	}
	if i < len(regs) {
		a.Printf("\tMOVD %s, %d(%s)\n", regs[0], i*a.Arch.WordBytes, p)
	}
}
