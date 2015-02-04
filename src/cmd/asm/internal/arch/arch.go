// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arch

import (
	"cmd/internal/obj"
	"cmd/internal/obj/i386" // == 386
	"cmd/internal/obj/x86"  // == amd64
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
	D_INDIR  int // TODO: why not in LinkArch?
	D_CONST2 int // TODO: why not in LinkArch?
	// Register number of hardware stack pointer.
	SP int
	// Encoding of non-address.
	NoAddr obj.Addr
	// Map of instruction names to enumeration.
	Instructions map[string]int
	// Map of register names to enumeration.
	Registers map[string]int
	// Map of pseudo-instructions (TEXT, DATA etc.)  to enumeration.
	Pseudos map[string]int
	// Instructions that take one operand whose result is a destination.
	UnaryDestination map[int]bool
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
	}
	return nil
}

func arch386() *Arch {
	noAddr := obj.Addr{
		Type:  i386.D_NONE,
		Index: i386.D_NONE,
	}

	registers := make(map[string]int)
	// Create maps for easy lookup of instruction names etc.
	// TODO: Should this be done in obj for us?
	for i, s := range i386.Register {
		registers[s] = i
	}
	// Pseudo-registers.
	registers["SB"] = RSB
	registers["FP"] = RFP
	registers["SP"] = RSP
	registers["PC"] = RPC

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

	pseudos := make(map[string]int) // TEXT, DATA etc.
	pseudos["DATA"] = i386.ADATA
	pseudos["FUNCDATA"] = i386.AFUNCDATA
	pseudos["GLOBL"] = i386.AGLOBL
	pseudos["PCDATA"] = i386.APCDATA
	pseudos["TEXT"] = i386.ATEXT

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
		D_INDIR:          i386.D_INDIR,
		D_CONST2:         i386.D_CONST2,
		SP:               i386.D_SP,
		NoAddr:           noAddr,
		Instructions:     instructions,
		Registers:        registers,
		Pseudos:          pseudos,
		UnaryDestination: unaryDestination,
	}
}

func archAmd64() *Arch {
	noAddr := obj.Addr{
		Type:  x86.D_NONE,
		Index: x86.D_NONE,
	}

	registers := make(map[string]int)
	// Create maps for easy lookup of instruction names etc.
	// TODO: Should this be done in obj for us?
	for i, s := range x86.Register {
		registers[s] = i
	}
	// Pseudo-registers.
	registers["SB"] = RSB
	registers["FP"] = RFP
	registers["SP"] = RSP
	registers["PC"] = RPC

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

	pseudos := make(map[string]int) // TEXT, DATA etc.
	pseudos["DATA"] = x86.ADATA
	pseudos["FUNCDATA"] = x86.AFUNCDATA
	pseudos["GLOBL"] = x86.AGLOBL
	pseudos["PCDATA"] = x86.APCDATA
	pseudos["TEXT"] = x86.ATEXT

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
		D_INDIR:          x86.D_INDIR,
		D_CONST2:         x86.D_NONE,
		SP:               x86.D_SP,
		NoAddr:           noAddr,
		Instructions:     instructions,
		Registers:        registers,
		Pseudos:          pseudos,
		UnaryDestination: unaryDestination,
	}
}
