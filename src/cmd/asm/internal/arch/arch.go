// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package arch defines architecture-specific information and support functions.
package arch

import (
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
	"cmd/internal/obj/arm64"
	"cmd/internal/obj/loong64"
	"cmd/internal/obj/mips"
	"cmd/internal/obj/ppc64"
	"cmd/internal/obj/riscv"
	"cmd/internal/obj/s390x"
	"cmd/internal/obj/wasm"
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
	Instructions map[string]obj.As
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

// Set configures the architecture specified by GOARCH and returns its representation.
// It returns nil if GOARCH is not recognized.
func Set(GOARCH string, shared bool) *Arch {
	switch GOARCH {
	case "386":
		return archX86(&x86.Link386)
	case "amd64":
		return archX86(&x86.Linkamd64)
	case "arm":
		return archArm()
	case "arm64":
		return archArm64()
	case "loong64":
		return archLoong64(&loong64.Linkloong64)
	case "mips":
		return archMips(&mips.Linkmips)
	case "mipsle":
		return archMips(&mips.Linkmipsle)
	case "mips64":
		return archMips64(&mips.Linkmips64)
	case "mips64le":
		return archMips64(&mips.Linkmips64le)
	case "ppc64":
		return archPPC64(&ppc64.Linkppc64)
	case "ppc64le":
		return archPPC64(&ppc64.Linkppc64le)
	case "riscv64":
		return archRISCV64(shared)
	case "s390x":
		return archS390x()
	case "wasm":
		return archWasm()
	}
	return nil
}

func jumpX86(word string) bool {
	return word[0] == 'J' || word == "CALL" || strings.HasPrefix(word, "LOOP") || word == "XBEGIN"
}

func jumpRISCV(word string) bool {
	switch word {
	case "BEQ", "BEQZ", "BGE", "BGEU", "BGEZ", "BGT", "BGTU", "BGTZ", "BLE", "BLEU", "BLEZ",
		"BLT", "BLTU", "BLTZ", "BNE", "BNEZ", "CALL", "JAL", "JALR", "JMP":
		return true
	}
	return false
}

func jumpWasm(word string) bool {
	return word == "JMP" || word == "CALL" || word == "Call" || word == "Br" || word == "BrIf"
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
	if linkArch == &x86.Linkamd64 {
		// Alias g to R14
		register["g"] = x86.REGG
	}
	// Register prefix not used on this architecture.

	instructions := make(map[string]obj.As)
	for i, s := range obj.Anames {
		instructions[s] = obj.As(i)
	}
	for i, s := range x86.Anames {
		if obj.As(i) >= obj.A_ARCHSPECIFIC {
			instructions[s] = obj.As(i) + obj.ABaseAMD64
		}
	}
	// Annoying aliases.
	instructions["JA"] = x86.AJHI   /* alternate */
	instructions["JAE"] = x86.AJCC  /* alternate */
	instructions["JB"] = x86.AJCS   /* alternate */
	instructions["JBE"] = x86.AJLS  /* alternate */
	instructions["JC"] = x86.AJCS   /* alternate */
	instructions["JCC"] = x86.AJCC  /* carry clear (CF = 0) */
	instructions["JCS"] = x86.AJCS  /* carry set (CF = 1) */
	instructions["JE"] = x86.AJEQ   /* alternate */
	instructions["JEQ"] = x86.AJEQ  /* equal (ZF = 1) */
	instructions["JG"] = x86.AJGT   /* alternate */
	instructions["JGE"] = x86.AJGE  /* greater than or equal (signed) (SF = OF) */
	instructions["JGT"] = x86.AJGT  /* greater than (signed) (ZF = 0 && SF = OF) */
	instructions["JHI"] = x86.AJHI  /* higher (unsigned) (CF = 0 && ZF = 0) */
	instructions["JHS"] = x86.AJCC  /* alternate */
	instructions["JL"] = x86.AJLT   /* alternate */
	instructions["JLE"] = x86.AJLE  /* less than or equal (signed) (ZF = 1 || SF != OF) */
	instructions["JLO"] = x86.AJCS  /* alternate */
	instructions["JLS"] = x86.AJLS  /* lower or same (unsigned) (CF = 1 || ZF = 1) */
	instructions["JLT"] = x86.AJLT  /* less than (signed) (SF != OF) */
	instructions["JMI"] = x86.AJMI  /* negative (minus) (SF = 1) */
	instructions["JNA"] = x86.AJLS  /* alternate */
	instructions["JNAE"] = x86.AJCS /* alternate */
	instructions["JNB"] = x86.AJCC  /* alternate */
	instructions["JNBE"] = x86.AJHI /* alternate */
	instructions["JNC"] = x86.AJCC  /* alternate */
	instructions["JNE"] = x86.AJNE  /* not equal (ZF = 0) */
	instructions["JNG"] = x86.AJLE  /* alternate */
	instructions["JNGE"] = x86.AJLT /* alternate */
	instructions["JNL"] = x86.AJGE  /* alternate */
	instructions["JNLE"] = x86.AJGT /* alternate */
	instructions["JNO"] = x86.AJOC  /* alternate */
	instructions["JNP"] = x86.AJPC  /* alternate */
	instructions["JNS"] = x86.AJPL  /* alternate */
	instructions["JNZ"] = x86.AJNE  /* alternate */
	instructions["JO"] = x86.AJOS   /* alternate */
	instructions["JOC"] = x86.AJOC  /* overflow clear (OF = 0) */
	instructions["JOS"] = x86.AJOS  /* overflow set (OF = 1) */
	instructions["JP"] = x86.AJPS   /* alternate */
	instructions["JPC"] = x86.AJPC  /* parity clear (PF = 0) */
	instructions["JPE"] = x86.AJPS  /* alternate */
	instructions["JPL"] = x86.AJPL  /* non-negative (plus) (SF = 0) */
	instructions["JPO"] = x86.AJPC  /* alternate */
	instructions["JPS"] = x86.AJPS  /* parity set (PF = 1) */
	instructions["JS"] = x86.AJMI   /* alternate */
	instructions["JZ"] = x86.AJEQ   /* alternate */
	instructions["MASKMOVDQU"] = x86.AMASKMOVOU
	instructions["MOVD"] = x86.AMOVQ
	instructions["MOVDQ2Q"] = x86.AMOVQ
	instructions["MOVNTDQ"] = x86.AMOVNTO
	instructions["MOVOA"] = x86.AMOVO
	instructions["PSLLDQ"] = x86.APSLLO
	instructions["PSRLDQ"] = x86.APSRLO
	instructions["PADDD"] = x86.APADDL
	// Spellings originally used in CL 97235.
	instructions["MOVBELL"] = x86.AMOVBEL
	instructions["MOVBEQQ"] = x86.AMOVBEQ
	instructions["MOVBEWW"] = x86.AMOVBEW

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

	// special operands for DMB/DSB instructions
	register["MB_SY"] = arm.REG_MB_SY
	register["MB_ST"] = arm.REG_MB_ST
	register["MB_ISH"] = arm.REG_MB_ISH
	register["MB_ISHST"] = arm.REG_MB_ISHST
	register["MB_NSH"] = arm.REG_MB_NSH
	register["MB_NSHST"] = arm.REG_MB_NSHST
	register["MB_OSH"] = arm.REG_MB_OSH
	register["MB_OSHST"] = arm.REG_MB_OSHST

	instructions := make(map[string]obj.As)
	for i, s := range obj.Anames {
		instructions[s] = obj.As(i)
	}
	for i, s := range arm.Anames {
		if obj.As(i) >= obj.A_ARCHSPECIFIC {
			instructions[s] = obj.As(i) + obj.ABaseARM
		}
	}
	// Annoying aliases.
	instructions["B"] = obj.AJMP
	instructions["BL"] = obj.ACALL
	// MCR differs from MRC by the way fields of the word are encoded.
	// (Details in arm.go). Here we add the instruction so parse will find
	// it, but give it an opcode number known only to us.
	instructions["MCR"] = aMCR

	return &Arch{
		LinkArch:       &arm.Linkarm,
		Instructions:   instructions,
		Register:       register,
		RegisterPrefix: registerPrefix,
		RegisterNumber: armRegisterNumber,
		IsJump:         jumpArm,
	}
}

func archArm64() *Arch {
	register := make(map[string]int16)
	// Create maps for easy lookup of instruction names etc.
	// Note that there is no list of names as there is for 386 and amd64.
	register[obj.Rconv(arm64.REGSP)] = int16(arm64.REGSP)
	for i := arm64.REG_R0; i <= arm64.REG_R31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	// Rename R18 to R18_PLATFORM to avoid accidental use.
	register["R18_PLATFORM"] = register["R18"]
	delete(register, "R18")
	for i := arm64.REG_F0; i <= arm64.REG_F31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := arm64.REG_V0; i <= arm64.REG_V31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}

	// System registers.
	for i := 0; i < len(arm64.SystemReg); i++ {
		register[arm64.SystemReg[i].Name] = arm64.SystemReg[i].Reg
	}

	register["LR"] = arm64.REGLINK

	// Pseudo-registers.
	register["SB"] = RSB
	register["FP"] = RFP
	register["PC"] = RPC
	register["SP"] = RSP
	// Avoid unintentionally clobbering g using R28.
	delete(register, "R28")
	register["g"] = arm64.REG_R28
	registerPrefix := map[string]bool{
		"F": true,
		"R": true,
		"V": true,
	}

	instructions := make(map[string]obj.As)
	for i, s := range obj.Anames {
		instructions[s] = obj.As(i)
	}
	for i, s := range arm64.Anames {
		if obj.As(i) >= obj.A_ARCHSPECIFIC {
			instructions[s] = obj.As(i) + obj.ABaseARM64
		}
	}
	// Annoying aliases.
	instructions["B"] = arm64.AB
	instructions["BL"] = arm64.ABL

	return &Arch{
		LinkArch:       &arm64.Linkarm64,
		Instructions:   instructions,
		Register:       register,
		RegisterPrefix: registerPrefix,
		RegisterNumber: arm64RegisterNumber,
		IsJump:         jumpArm64,
	}

}

func archPPC64(linkArch *obj.LinkArch) *Arch {
	register := make(map[string]int16)
	// Create maps for easy lookup of instruction names etc.
	// Note that there is no list of names as there is for x86.
	for i := ppc64.REG_R0; i <= ppc64.REG_R31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := ppc64.REG_F0; i <= ppc64.REG_F31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := ppc64.REG_V0; i <= ppc64.REG_V31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := ppc64.REG_VS0; i <= ppc64.REG_VS63; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := ppc64.REG_A0; i <= ppc64.REG_A7; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := ppc64.REG_CR0; i <= ppc64.REG_CR7; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := ppc64.REG_MSR; i <= ppc64.REG_CR; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := ppc64.REG_CR0LT; i <= ppc64.REG_CR7SO; i++ {
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

	instructions := make(map[string]obj.As)
	for i, s := range obj.Anames {
		instructions[s] = obj.As(i)
	}
	for i, s := range ppc64.Anames {
		if obj.As(i) >= obj.A_ARCHSPECIFIC {
			instructions[s] = obj.As(i) + obj.ABasePPC64
		}
	}
	// The opcodes generated by x/arch's ppc64map are listed in
	// a separate slice, add them too.
	for i, s := range ppc64.GenAnames {
		instructions[s] = obj.As(i) + ppc64.AFIRSTGEN
	}
	// Annoying aliases.
	instructions["BR"] = ppc64.ABR
	instructions["BL"] = ppc64.ABL

	return &Arch{
		LinkArch:       linkArch,
		Instructions:   instructions,
		Register:       register,
		RegisterPrefix: registerPrefix,
		RegisterNumber: ppc64RegisterNumber,
		IsJump:         jumpPPC64,
	}
}

func archMips(linkArch *obj.LinkArch) *Arch {
	register := make(map[string]int16)
	// Create maps for easy lookup of instruction names etc.
	// Note that there is no list of names as there is for x86.
	for i := mips.REG_R0; i <= mips.REG_R31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}

	for i := mips.REG_F0; i <= mips.REG_F31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := mips.REG_M0; i <= mips.REG_M31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := mips.REG_FCR0; i <= mips.REG_FCR31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	register["HI"] = mips.REG_HI
	register["LO"] = mips.REG_LO
	// Pseudo-registers.
	register["SB"] = RSB
	register["FP"] = RFP
	register["PC"] = RPC
	// Avoid unintentionally clobbering g using R30.
	delete(register, "R30")
	register["g"] = mips.REG_R30

	registerPrefix := map[string]bool{
		"F":   true,
		"FCR": true,
		"M":   true,
		"R":   true,
	}

	instructions := make(map[string]obj.As)
	for i, s := range obj.Anames {
		instructions[s] = obj.As(i)
	}
	for i, s := range mips.Anames {
		if obj.As(i) >= obj.A_ARCHSPECIFIC {
			instructions[s] = obj.As(i) + obj.ABaseMIPS
		}
	}
	// Annoying alias.
	instructions["JAL"] = mips.AJAL

	return &Arch{
		LinkArch:       linkArch,
		Instructions:   instructions,
		Register:       register,
		RegisterPrefix: registerPrefix,
		RegisterNumber: mipsRegisterNumber,
		IsJump:         jumpMIPS,
	}
}

func archMips64(linkArch *obj.LinkArch) *Arch {
	register := make(map[string]int16)
	// Create maps for easy lookup of instruction names etc.
	// Note that there is no list of names as there is for x86.
	for i := mips.REG_R0; i <= mips.REG_R31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := mips.REG_F0; i <= mips.REG_F31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := mips.REG_M0; i <= mips.REG_M31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := mips.REG_FCR0; i <= mips.REG_FCR31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := mips.REG_W0; i <= mips.REG_W31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	register["HI"] = mips.REG_HI
	register["LO"] = mips.REG_LO
	// Pseudo-registers.
	register["SB"] = RSB
	register["FP"] = RFP
	register["PC"] = RPC
	// Avoid unintentionally clobbering g using R30.
	delete(register, "R30")
	register["g"] = mips.REG_R30
	// Avoid unintentionally clobbering RSB using R28.
	delete(register, "R28")
	register["RSB"] = mips.REG_R28
	registerPrefix := map[string]bool{
		"F":   true,
		"FCR": true,
		"M":   true,
		"R":   true,
		"W":   true,
	}

	instructions := make(map[string]obj.As)
	for i, s := range obj.Anames {
		instructions[s] = obj.As(i)
	}
	for i, s := range mips.Anames {
		if obj.As(i) >= obj.A_ARCHSPECIFIC {
			instructions[s] = obj.As(i) + obj.ABaseMIPS
		}
	}
	// Annoying alias.
	instructions["JAL"] = mips.AJAL

	return &Arch{
		LinkArch:       linkArch,
		Instructions:   instructions,
		Register:       register,
		RegisterPrefix: registerPrefix,
		RegisterNumber: mipsRegisterNumber,
		IsJump:         jumpMIPS,
	}
}

func archLoong64(linkArch *obj.LinkArch) *Arch {
	register := make(map[string]int16)
	// Create maps for easy lookup of instruction names etc.
	// Note that there is no list of names as there is for x86.
	for i := loong64.REG_R0; i <= loong64.REG_R31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := loong64.REG_F0; i <= loong64.REG_F31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := loong64.REG_FCSR0; i <= loong64.REG_FCSR31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := loong64.REG_FCC0; i <= loong64.REG_FCC31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	// Pseudo-registers.
	register["SB"] = RSB
	register["FP"] = RFP
	register["PC"] = RPC
	// Avoid unintentionally clobbering g using R22.
	delete(register, "R22")
	register["g"] = loong64.REG_R22
	registerPrefix := map[string]bool{
		"F":    true,
		"FCSR": true,
		"FCC":  true,
		"R":    true,
	}

	instructions := make(map[string]obj.As)
	for i, s := range obj.Anames {
		instructions[s] = obj.As(i)
	}
	for i, s := range loong64.Anames {
		if obj.As(i) >= obj.A_ARCHSPECIFIC {
			instructions[s] = obj.As(i) + obj.ABaseLoong64
		}
	}
	// Annoying alias.
	instructions["JAL"] = loong64.AJAL

	return &Arch{
		LinkArch:       linkArch,
		Instructions:   instructions,
		Register:       register,
		RegisterPrefix: registerPrefix,
		RegisterNumber: loong64RegisterNumber,
		IsJump:         jumpLoong64,
	}
}

func archRISCV64(shared bool) *Arch {
	register := make(map[string]int16)

	// Standard register names.
	for i := riscv.REG_X0; i <= riscv.REG_X31; i++ {
		// Disallow X3 in shared mode, as this will likely be used as the
		// GP register, which could result in problems in non-Go code,
		// including signal handlers.
		if shared && i == riscv.REG_GP {
			continue
		}
		if i == riscv.REG_TP || i == riscv.REG_G {
			continue
		}
		name := fmt.Sprintf("X%d", i-riscv.REG_X0)
		register[name] = int16(i)
	}
	for i := riscv.REG_F0; i <= riscv.REG_F31; i++ {
		name := fmt.Sprintf("F%d", i-riscv.REG_F0)
		register[name] = int16(i)
	}
	for i := riscv.REG_V0; i <= riscv.REG_V31; i++ {
		name := fmt.Sprintf("V%d", i-riscv.REG_V0)
		register[name] = int16(i)
	}

	// General registers with ABI names.
	register["ZERO"] = riscv.REG_ZERO
	register["RA"] = riscv.REG_RA
	register["SP"] = riscv.REG_SP
	register["GP"] = riscv.REG_GP
	register["TP"] = riscv.REG_TP
	register["T0"] = riscv.REG_T0
	register["T1"] = riscv.REG_T1
	register["T2"] = riscv.REG_T2
	register["S0"] = riscv.REG_S0
	register["S1"] = riscv.REG_S1
	register["A0"] = riscv.REG_A0
	register["A1"] = riscv.REG_A1
	register["A2"] = riscv.REG_A2
	register["A3"] = riscv.REG_A3
	register["A4"] = riscv.REG_A4
	register["A5"] = riscv.REG_A5
	register["A6"] = riscv.REG_A6
	register["A7"] = riscv.REG_A7
	register["S2"] = riscv.REG_S2
	register["S3"] = riscv.REG_S3
	register["S4"] = riscv.REG_S4
	register["S5"] = riscv.REG_S5
	register["S6"] = riscv.REG_S6
	register["S7"] = riscv.REG_S7
	register["S8"] = riscv.REG_S8
	register["S9"] = riscv.REG_S9
	register["S10"] = riscv.REG_S10
	// Skip S11 as it is the g register.
	register["T3"] = riscv.REG_T3
	register["T4"] = riscv.REG_T4
	register["T5"] = riscv.REG_T5
	register["T6"] = riscv.REG_T6

	// Go runtime register names.
	register["g"] = riscv.REG_G
	register["CTXT"] = riscv.REG_CTXT
	register["TMP"] = riscv.REG_TMP

	// ABI names for floating point register.
	register["FT0"] = riscv.REG_FT0
	register["FT1"] = riscv.REG_FT1
	register["FT2"] = riscv.REG_FT2
	register["FT3"] = riscv.REG_FT3
	register["FT4"] = riscv.REG_FT4
	register["FT5"] = riscv.REG_FT5
	register["FT6"] = riscv.REG_FT6
	register["FT7"] = riscv.REG_FT7
	register["FS0"] = riscv.REG_FS0
	register["FS1"] = riscv.REG_FS1
	register["FA0"] = riscv.REG_FA0
	register["FA1"] = riscv.REG_FA1
	register["FA2"] = riscv.REG_FA2
	register["FA3"] = riscv.REG_FA3
	register["FA4"] = riscv.REG_FA4
	register["FA5"] = riscv.REG_FA5
	register["FA6"] = riscv.REG_FA6
	register["FA7"] = riscv.REG_FA7
	register["FS2"] = riscv.REG_FS2
	register["FS3"] = riscv.REG_FS3
	register["FS4"] = riscv.REG_FS4
	register["FS5"] = riscv.REG_FS5
	register["FS6"] = riscv.REG_FS6
	register["FS7"] = riscv.REG_FS7
	register["FS8"] = riscv.REG_FS8
	register["FS9"] = riscv.REG_FS9
	register["FS10"] = riscv.REG_FS10
	register["FS11"] = riscv.REG_FS11
	register["FT8"] = riscv.REG_FT8
	register["FT9"] = riscv.REG_FT9
	register["FT10"] = riscv.REG_FT10
	register["FT11"] = riscv.REG_FT11

	// Pseudo-registers.
	register["SB"] = RSB
	register["FP"] = RFP
	register["PC"] = RPC

	instructions := make(map[string]obj.As)
	for i, s := range obj.Anames {
		instructions[s] = obj.As(i)
	}
	for i, s := range riscv.Anames {
		if obj.As(i) >= obj.A_ARCHSPECIFIC {
			instructions[s] = obj.As(i) + obj.ABaseRISCV
		}
	}

	return &Arch{
		LinkArch:       &riscv.LinkRISCV64,
		Instructions:   instructions,
		Register:       register,
		RegisterPrefix: nil,
		RegisterNumber: nilRegisterNumber,
		IsJump:         jumpRISCV,
	}
}

func archS390x() *Arch {
	register := make(map[string]int16)
	// Create maps for easy lookup of instruction names etc.
	// Note that there is no list of names as there is for x86.
	for i := s390x.REG_R0; i <= s390x.REG_R15; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := s390x.REG_F0; i <= s390x.REG_F15; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := s390x.REG_V0; i <= s390x.REG_V31; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	for i := s390x.REG_AR0; i <= s390x.REG_AR15; i++ {
		register[obj.Rconv(i)] = int16(i)
	}
	register["LR"] = s390x.REG_LR
	// Pseudo-registers.
	register["SB"] = RSB
	register["FP"] = RFP
	register["PC"] = RPC
	// Avoid unintentionally clobbering g using R13.
	delete(register, "R13")
	register["g"] = s390x.REG_R13
	registerPrefix := map[string]bool{
		"AR": true,
		"F":  true,
		"R":  true,
	}

	instructions := make(map[string]obj.As)
	for i, s := range obj.Anames {
		instructions[s] = obj.As(i)
	}
	for i, s := range s390x.Anames {
		if obj.As(i) >= obj.A_ARCHSPECIFIC {
			instructions[s] = obj.As(i) + obj.ABaseS390X
		}
	}
	// Annoying aliases.
	instructions["BR"] = s390x.ABR
	instructions["BL"] = s390x.ABL

	return &Arch{
		LinkArch:       &s390x.Links390x,
		Instructions:   instructions,
		Register:       register,
		RegisterPrefix: registerPrefix,
		RegisterNumber: s390xRegisterNumber,
		IsJump:         jumpS390x,
	}
}

func archWasm() *Arch {
	instructions := make(map[string]obj.As)
	for i, s := range obj.Anames {
		instructions[s] = obj.As(i)
	}
	for i, s := range wasm.Anames {
		if obj.As(i) >= obj.A_ARCHSPECIFIC {
			instructions[s] = obj.As(i) + obj.ABaseWasm
		}
	}

	return &Arch{
		LinkArch:       &wasm.Linkwasm,
		Instructions:   instructions,
		Register:       wasm.Register,
		RegisterPrefix: nil,
		RegisterNumber: nilRegisterNumber,
		IsJump:         jumpWasm,
	}
}
