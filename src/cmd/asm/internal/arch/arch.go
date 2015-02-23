// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arch

import (
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
	"cmd/internal/obj/i386" // == 386
	"cmd/internal/obj/ppc64"
	"cmd/internal/obj/x86" // == amd64
	"fmt"
)

// Pseudo-registers whose names are the constant name without the leading R.
const (
	RFP = -(iota + 1)
	RSB
	RSP
	RPC
)

// Arch wraps the link architecture object with more architecture-specific information.
type Arch struct {
	*obj.LinkArch
	// Map of instruction names to enumeration.
	Instructions map[string]int
	// Map of register names to enumeration.
	Register map[string]int16
	// Table of register prefix names. These are things like R for R(0) and SPR for SPR(268).
	RegisterPrefix map[string]bool
	// RegisterNumber converts R(10) into arm.REG_R10.
	RegisterNumber func(string, int16) (int16, bool)
	// Instructions that take one operand whose result is a destination.
	UnaryDestination map[int]bool
	// Instruction is a jump.
	IsJump func(word string) bool
	// Aconv pretty-prints an instruction opcode for this architecture.
	Aconv func(int) string
	// Dconv pretty-prints an address for this architecture.
	Dconv func(p *obj.Prog, flag int, a *obj.Addr) string
}

// nilRegisterNumber is the register number function for architectures
// that do not accept the R(N) notation. It always returns failure.
func nilRegisterNumber(name string, n int16) (int16, bool) {
	return 0, false
}

var Pseudos = map[string]int{
	"DATA":     obj.ADATA,
	"FUNCDATA": obj.AFUNCDATA,
	"GLOBL":    obj.AGLOBL,
	"PCDATA":   obj.APCDATA,
	"TEXT":     obj.ATEXT,
}

// Set configures the architecture specified by GOARCH and returns its representation.
// It returns nil if GOARCH is not recognized.
func Set(GOARCH string) *Arch {
	// TODO: Is this how to set this up?
	switch GOARCH {
	case "386":
		return arch386()
	case "amd64":
		return archAmd64()
	case "amd64p32":
		a := archAmd64()
		a.LinkArch = &x86.Linkamd64p32
		return a
	case "arm":
		return archArm()
	case "ppc64":
		a := archPPC64()
		a.LinkArch = &ppc64.Linkppc64
		return a
	case "ppc64le":
		a := archPPC64()
		a.LinkArch = &ppc64.Linkppc64le
		return a
	}
	return nil
}

func jump386(word string) bool {
	return word[0] == 'J' || word == "CALL"
}

func arch386() *Arch {
	register := make(map[string]int16)
	// Create maps for easy lookup of instruction names etc.
	// TODO: Should this be done in obj for us?
	for i, s := range i386.Register {
		register[s] = int16(i + i386.REG_AL)
	}
	// Pseudo-registers.
	register["SB"] = RSB
	register["FP"] = RFP
	register["PC"] = RPC
	// Prefixes not used on this architecture.

	instructions := make(map[string]int)
	for i, s := range i386.Anames {
		instructions[s] = i
	}
	// Annoying aliases.
	instructions["JA"] = i386.AJHI
	instructions["JAE"] = i386.AJCC
	instructions["JB"] = i386.AJCS
	instructions["JBE"] = i386.AJLS
	instructions["JC"] = i386.AJCS
	instructions["JE"] = i386.AJEQ
	instructions["JG"] = i386.AJGT
	instructions["JHS"] = i386.AJCC
	instructions["JL"] = i386.AJLT
	instructions["JLO"] = i386.AJCS
	instructions["JNA"] = i386.AJLS
	instructions["JNAE"] = i386.AJCS
	instructions["JNB"] = i386.AJCC
	instructions["JNBE"] = i386.AJHI
	instructions["JNC"] = i386.AJCC
	instructions["JNG"] = i386.AJLE
	instructions["JNGE"] = i386.AJLT
	instructions["JNL"] = i386.AJGE
	instructions["JNLE"] = i386.AJGT
	instructions["JNO"] = i386.AJOC
	instructions["JNP"] = i386.AJPC
	instructions["JNS"] = i386.AJPL
	instructions["JNZ"] = i386.AJNE
	instructions["JO"] = i386.AJOS
	instructions["JP"] = i386.AJPS
	instructions["JPE"] = i386.AJPS
	instructions["JPO"] = i386.AJPC
	instructions["JS"] = i386.AJMI
	instructions["JZ"] = i386.AJEQ
	instructions["MASKMOVDQU"] = i386.AMASKMOVOU
	instructions["MOVOA"] = i386.AMOVO
	instructions["MOVNTDQ"] = i386.AMOVNTO

	unaryDestination := make(map[int]bool) // Instruction takes one operand and result is a destination.
	// These instructions write to prog.To.
	unaryDestination[i386.ABSWAPL] = true
	unaryDestination[i386.ACMPXCHG8B] = true
	unaryDestination[i386.ADECB] = true
	unaryDestination[i386.ADECL] = true
	unaryDestination[i386.ADECW] = true
	unaryDestination[i386.AINCB] = true
	unaryDestination[i386.AINCL] = true
	unaryDestination[i386.AINCW] = true
	unaryDestination[i386.ANEGB] = true
	unaryDestination[i386.ANEGL] = true
	unaryDestination[i386.ANEGW] = true
	unaryDestination[i386.ANOTB] = true
	unaryDestination[i386.ANOTL] = true
	unaryDestination[i386.ANOTW] = true
	unaryDestination[i386.APOPL] = true
	unaryDestination[i386.APOPW] = true
	unaryDestination[i386.ASETCC] = true
	unaryDestination[i386.ASETCS] = true
	unaryDestination[i386.ASETEQ] = true
	unaryDestination[i386.ASETGE] = true
	unaryDestination[i386.ASETGT] = true
	unaryDestination[i386.ASETHI] = true
	unaryDestination[i386.ASETLE] = true
	unaryDestination[i386.ASETLS] = true
	unaryDestination[i386.ASETLT] = true
	unaryDestination[i386.ASETMI] = true
	unaryDestination[i386.ASETNE] = true
	unaryDestination[i386.ASETOC] = true
	unaryDestination[i386.ASETOS] = true
	unaryDestination[i386.ASETPC] = true
	unaryDestination[i386.ASETPL] = true
	unaryDestination[i386.ASETPS] = true
	unaryDestination[i386.AFFREE] = true
	unaryDestination[i386.AFLDENV] = true
	unaryDestination[i386.AFSAVE] = true
	unaryDestination[i386.AFSTCW] = true
	unaryDestination[i386.AFSTENV] = true
	unaryDestination[i386.AFSTSW] = true

	return &Arch{
		LinkArch:         &i386.Link386,
		Instructions:     instructions,
		Register:         register,
		RegisterPrefix:   nil,
		RegisterNumber:   nilRegisterNumber,
		UnaryDestination: unaryDestination,
		IsJump:           jump386,
		Aconv:            i386.Aconv,
		Dconv:            i386.Dconv,
	}
}

func archAmd64() *Arch {
	register := make(map[string]int16)
	// Create maps for easy lookup of instruction names etc.
	// TODO: Should this be done in obj for us?
	for i, s := range x86.Register {
		register[s] = int16(i + x86.REG_AL)
	}
	// Pseudo-registers.
	register["SB"] = RSB
	register["FP"] = RFP
	register["PC"] = RPC
	// Register prefix not used on this architecture.

	instructions := make(map[string]int)
	for i, s := range x86.Anames {
		instructions[s] = i
	}
	// Annoying aliases.
	instructions["JB"] = x86.AJCS
	instructions["JC"] = x86.AJCS
	instructions["JNAE"] = x86.AJCS
	instructions["JLO"] = x86.AJCS
	instructions["JAE"] = x86.AJCC
	instructions["JNB"] = x86.AJCC
	instructions["JNC"] = x86.AJCC
	instructions["JHS"] = x86.AJCC
	instructions["JE"] = x86.AJEQ
	instructions["JZ"] = x86.AJEQ
	instructions["JNZ"] = x86.AJNE
	instructions["JBE"] = x86.AJLS
	instructions["JNA"] = x86.AJLS
	instructions["JA"] = x86.AJHI
	instructions["JNBE"] = x86.AJHI
	instructions["JS"] = x86.AJMI
	instructions["JNS"] = x86.AJPL
	instructions["JP"] = x86.AJPS
	instructions["JPE"] = x86.AJPS
	instructions["JNP"] = x86.AJPC
	instructions["JPO"] = x86.AJPC
	instructions["JL"] = x86.AJLT
	instructions["JNGE"] = x86.AJLT
	instructions["JNL"] = x86.AJGE
	instructions["JNG"] = x86.AJLE
	instructions["JG"] = x86.AJGT
	instructions["JNLE"] = x86.AJGT
	instructions["MASKMOVDQU"] = x86.AMASKMOVOU
	instructions["MOVD"] = x86.AMOVQ
	instructions["MOVDQ2Q"] = x86.AMOVQ

	unaryDestination := make(map[int]bool) // Instruction takes one operand and result is a destination.
	// These instructions write to prog.To.
	unaryDestination[x86.ABSWAPL] = true
	unaryDestination[x86.ABSWAPQ] = true
	unaryDestination[x86.ACMPXCHG8B] = true
	unaryDestination[x86.ADECB] = true
	unaryDestination[x86.ADECL] = true
	unaryDestination[x86.ADECQ] = true
	unaryDestination[x86.ADECW] = true
	unaryDestination[x86.AINCB] = true
	unaryDestination[x86.AINCL] = true
	unaryDestination[x86.AINCQ] = true
	unaryDestination[x86.AINCW] = true
	unaryDestination[x86.ANEGB] = true
	unaryDestination[x86.ANEGL] = true
	unaryDestination[x86.ANEGQ] = true
	unaryDestination[x86.ANEGW] = true
	unaryDestination[x86.ANOTB] = true
	unaryDestination[x86.ANOTL] = true
	unaryDestination[x86.ANOTQ] = true
	unaryDestination[x86.ANOTW] = true
	unaryDestination[x86.APOPL] = true
	unaryDestination[x86.APOPQ] = true
	unaryDestination[x86.APOPW] = true
	unaryDestination[x86.ASETCC] = true
	unaryDestination[x86.ASETCS] = true
	unaryDestination[x86.ASETEQ] = true
	unaryDestination[x86.ASETGE] = true
	unaryDestination[x86.ASETGT] = true
	unaryDestination[x86.ASETHI] = true
	unaryDestination[x86.ASETLE] = true
	unaryDestination[x86.ASETLS] = true
	unaryDestination[x86.ASETLT] = true
	unaryDestination[x86.ASETMI] = true
	unaryDestination[x86.ASETNE] = true
	unaryDestination[x86.ASETOC] = true
	unaryDestination[x86.ASETOS] = true
	unaryDestination[x86.ASETPC] = true
	unaryDestination[x86.ASETPL] = true
	unaryDestination[x86.ASETPS] = true
	unaryDestination[x86.AFFREE] = true
	unaryDestination[x86.AFLDENV] = true
	unaryDestination[x86.AFSAVE] = true
	unaryDestination[x86.AFSTCW] = true
	unaryDestination[x86.AFSTENV] = true
	unaryDestination[x86.AFSTSW] = true
	unaryDestination[x86.AFXSAVE] = true
	unaryDestination[x86.AFXSAVE64] = true
	unaryDestination[x86.ASTMXCSR] = true

	return &Arch{
		LinkArch:         &x86.Linkamd64,
		Instructions:     instructions,
		Register:         register,
		RegisterPrefix:   nil,
		RegisterNumber:   nilRegisterNumber,
		UnaryDestination: unaryDestination,
		IsJump:           jump386,
		Aconv:            x86.Aconv,
		Dconv:            x86.Dconv,
	}
}

func archArm() *Arch {
	register := make(map[string]int16)
	// Create maps for easy lookup of instruction names etc.
	// TODO: Should this be done in obj for us?
	// Note that there is no list of names as there is for 386 and amd64.
	// TODO: Are there aliases we need to add?
	for i := arm.REG_R0; i < arm.REG_SPSR; i++ {
		register[arm.Rconv(i)] = int16(i)
	}
	// Avoid unintentionally clobbering g using R10.
	delete(register, "R10")
	register["g"] = arm.REG_R10
	for i := 0; i < 16; i++ {
		register[fmt.Sprintf("C%d", i)] = int16(i)
	}

	// Pseudo-registers.
	register["SB"] = RSB
	register["FP"] = RFP
	register["PC"] = RPC
	register["SP"] = RSP
	registerPrefix := map[string]bool{
		"F": true,
		"R": true,
	}

	instructions := make(map[string]int)
	for i, s := range arm.Anames {
		instructions[s] = i
	}
	// Annoying aliases.
	instructions["B"] = obj.AJMP
	instructions["BL"] = obj.ACALL

	unaryDestination := make(map[int]bool) // Instruction takes one operand and result is a destination.
	// These instructions write to prog.To.
	// TODO: These are silly. Fix once C assembler is gone.
	unaryDestination[arm.ASWI] = true
	unaryDestination[arm.AWORD] = true

	return &Arch{
		LinkArch:         &arm.Linkarm,
		Instructions:     instructions,
		Register:         register,
		RegisterPrefix:   registerPrefix,
		RegisterNumber:   armRegisterNumber,
		UnaryDestination: unaryDestination,
		IsJump:           jumpArm,
		Aconv:            arm.Aconv,
		Dconv:            arm.Dconv,
	}
}

func archPPC64() *Arch {
	register := make(map[string]int16)
	// Create maps for easy lookup of instruction names etc.
	// TODO: Should this be done in obj for us?
	// Note that there is no list of names as there is for 386 and amd64.
	for i := ppc64.REG_R0; i <= ppc64.REG_R31; i++ {
		register[ppc64.Rconv(i)] = int16(i)
	}
	for i := ppc64.REG_F0; i <= ppc64.REG_F31; i++ {
		register[ppc64.Rconv(i)] = int16(i)
	}
	for i := ppc64.REG_C0; i <= ppc64.REG_C7; i++ {
		// TODO: Rconv prints these as C7 but the input syntax requires CR7.
		register[fmt.Sprintf("CR%d", i-ppc64.REG_C0)] = int16(i)
	}
	for i := ppc64.REG_MSR; i <= ppc64.REG_CR; i++ {
		register[ppc64.Rconv(i)] = int16(i)
	}
	register["CR"] = ppc64.REG_CR
	register["XER"] = ppc64.REG_XER
	register["LR"] = ppc64.REG_LR
	register["CTR"] = ppc64.REG_CTR
	register["FPSCR"] = ppc64.REG_FPSCR
	register["MSR"] = ppc64.REG_MSR
	// Pseudo-registers.
	register["SB"] = RSB
	register["FP"] = RFP
	register["PC"] = RPC
	// Avoid unintentionally clobbering g using R30.
	delete(register, "R30")
	register["g"] = ppc64.REG_R30
	registerPrefix := map[string]bool{
		"CR":  true,
		"F":   true,
		"R":   true,
		"SPR": true,
	}

	instructions := make(map[string]int)
	for i, s := range ppc64.Anames {
		instructions[s] = i
	}
	// Annoying aliases.
	instructions["BR"] = ppc64.ABR
	instructions["BL"] = ppc64.ABL
	instructions["RETURN"] = ppc64.ARETURN

	return &Arch{
		LinkArch:         &ppc64.Linkppc64,
		Instructions:     instructions,
		Register:         register,
		RegisterPrefix:   registerPrefix,
		RegisterNumber:   ppc64RegisterNumber,
		UnaryDestination: nil,
		IsJump:           jumpPPC64,
		Aconv:            ppc64.Aconv,
		Dconv:            ppc64.Dconv,
	}
}
