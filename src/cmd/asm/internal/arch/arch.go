// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arch

import (
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
	"cmd/internal/obj/ppc64"
	"cmd/internal/obj/x86"
	"fmt"
	"strings"
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
	// Instruction is a jump.
	IsJump func(word string) bool
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
	switch GOARCH {
	case "386":
		return archX86(&x86.Link386)
	case "amd64":
		return archX86(&x86.Linkamd64)
	case "amd64p32":
		return archX86(&x86.Linkamd64p32)
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

func jumpX86(word string) bool {
	return word[0] == 'J' || word == "CALL" || strings.HasPrefix(word, "LOOP")
}

func archX86(linkArch *obj.LinkArch) *Arch {
	register := make(map[string]int16)
	// Create maps for easy lookup of instruction names etc.
	for i, s := range x86.Register {
		register[s] = int16(i + x86.REG_AL)
	}
	// Pseudo-registers.
	register["SB"] = RSB
	register["FP"] = RFP
	register["PC"] = RPC
	// Register prefix not used on this architecture.

	instructions := make(map[string]int)
	for i, s := range obj.Anames {
		instructions[s] = i
	}
	for i, s := range x86.Anames {
		if i >= obj.A_ARCHSPECIFIC {
			instructions[s] = i + obj.ABaseAMD64
		}
	}
	// Annoying aliases.
	instructions["JA"] = x86.AJHI
	instructions["JAE"] = x86.AJCC
	instructions["JB"] = x86.AJCS
	instructions["JBE"] = x86.AJLS
	instructions["JC"] = x86.AJCS
	instructions["JE"] = x86.AJEQ
	instructions["JG"] = x86.AJGT
	instructions["JHS"] = x86.AJCC
	instructions["JL"] = x86.AJLT
	instructions["JLO"] = x86.AJCS
	instructions["JNA"] = x86.AJLS
	instructions["JNAE"] = x86.AJCS
	instructions["JNB"] = x86.AJCC
	instructions["JNBE"] = x86.AJHI
	instructions["JNC"] = x86.AJCC
	instructions["JNG"] = x86.AJLE
	instructions["JNGE"] = x86.AJLT
	instructions["JNL"] = x86.AJGE
	instructions["JNLE"] = x86.AJGT
	instructions["JNO"] = x86.AJOC
	instructions["JNP"] = x86.AJPC
	instructions["JNS"] = x86.AJPL
	instructions["JNZ"] = x86.AJNE
	instructions["JO"] = x86.AJOS
	instructions["JP"] = x86.AJPS
	instructions["JPE"] = x86.AJPS
	instructions["JPO"] = x86.AJPC
	instructions["JS"] = x86.AJMI
	instructions["JZ"] = x86.AJEQ
	instructions["MASKMOVDQU"] = x86.AMASKMOVOU
	instructions["MOVD"] = x86.AMOVQ
	instructions["MOVDQ2Q"] = x86.AMOVQ
	instructions["MOVNTDQ"] = x86.AMOVNTO
	instructions["MOVOA"] = x86.AMOVO
	instructions["MOVOA"] = x86.AMOVO
	instructions["PF2ID"] = x86.APF2IL
	instructions["PI2FD"] = x86.API2FL
	instructions["PSLLDQ"] = x86.APSLLO
	instructions["PSRLDQ"] = x86.APSRLO

	return &Arch{
		LinkArch:       linkArch,
		Instructions:   instructions,
		Register:       register,
		RegisterPrefix: nil,
		RegisterNumber: nilRegisterNumber,
		IsJump:         jumpX86,
	}
}

func archArm() *Arch {
	register := make(map[string]int16)
	// Create maps for easy lookup of instruction names etc.
	// Note that there is no list of names as there is for x86.
	for i := arm.REG_R0; i < arm.REG_SPSR; i++ {
		register[obj.Rconv(i)] = int16(i)
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
	for i, s := range obj.Anames {
		instructions[s] = i
	}
	for i, s := range arm.Anames {
		if i >= obj.A_ARCHSPECIFIC {
			instructions[s] = i + obj.ABaseARM
		}
	}
	// Annoying aliases.
	instructions["B"] = obj.AJMP
	instructions["BL"] = obj.ACALL

	return &Arch{
		LinkArch:       &arm.Linkarm,
		Instructions:   instructions,
		Register:       register,
		RegisterPrefix: registerPrefix,
		RegisterNumber: armRegisterNumber,
		IsJump:         jumpArm,
	}
}

func archPPC64() *Arch {
	register := make(map[string]int16)
	// Create maps for easy lookup of instruction names etc.
	// Note that there is no list of names as there is for x86.
	for i := ppc64.REG_R0; i <= ppc64.REG_R31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := ppc64.REG_F0; i <= ppc64.REG_F31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := ppc64.REG_CR0; i <= ppc64.REG_CR7; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := ppc64.REG_MSR; i <= ppc64.REG_CR; i++ {
		register[obj.Rconv(i)] = int16(i)
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
	for i, s := range obj.Anames {
		instructions[s] = i
	}
	for i, s := range ppc64.Anames {
		if i >= obj.A_ARCHSPECIFIC {
			instructions[s] = i + obj.ABasePPC64
		}
	}
	// Annoying aliases.
	instructions["BR"] = ppc64.ABR
	instructions["BL"] = ppc64.ABL
	instructions["RETURN"] = ppc64.ARETURN

	return &Arch{
		LinkArch:       &ppc64.Linkppc64,
		Instructions:   instructions,
		Register:       register,
		RegisterPrefix: registerPrefix,
		RegisterNumber: ppc64RegisterNumber,
		IsJump:         jumpPPC64,
	}
}
