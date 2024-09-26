// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loong64

import (
	"cmd/internal/obj"
)

//go:generate go run ../stringer.go -i $GOFILE -o anames.go -p loong64

const (
	NSNAME = 8
	NSYM   = 50
	NREG   = 32 // number of general registers
	NFREG  = 32 // number of floating point registers
	NVREG  = 32 // number of LSX registers
	NXREG  = 32 // number of LASX registers
)

const (
	REG_R0 = obj.RBaseLOONG64 + iota // must be a multiple of 32
	REG_R1
	REG_R2
	REG_R3
	REG_R4
	REG_R5
	REG_R6
	REG_R7
	REG_R8
	REG_R9
	REG_R10
	REG_R11
	REG_R12
	REG_R13
	REG_R14
	REG_R15
	REG_R16
	REG_R17
	REG_R18
	REG_R19
	REG_R20
	REG_R21
	REG_R22
	REG_R23
	REG_R24
	REG_R25
	REG_R26
	REG_R27
	REG_R28
	REG_R29
	REG_R30
	REG_R31

	REG_F0 // must be a multiple of 32
	REG_F1
	REG_F2
	REG_F3
	REG_F4
	REG_F5
	REG_F6
	REG_F7
	REG_F8
	REG_F9
	REG_F10
	REG_F11
	REG_F12
	REG_F13
	REG_F14
	REG_F15
	REG_F16
	REG_F17
	REG_F18
	REG_F19
	REG_F20
	REG_F21
	REG_F22
	REG_F23
	REG_F24
	REG_F25
	REG_F26
	REG_F27
	REG_F28
	REG_F29
	REG_F30
	REG_F31

	REG_FCSR0 // must be a multiple of 32
	REG_FCSR1
	REG_FCSR2
	REG_FCSR3 // only four registers are needed
	REG_FCSR4
	REG_FCSR5
	REG_FCSR6
	REG_FCSR7
	REG_FCSR8
	REG_FCSR9
	REG_FCSR10
	REG_FCSR11
	REG_FCSR12
	REG_FCSR13
	REG_FCSR14
	REG_FCSR15
	REG_FCSR16
	REG_FCSR17
	REG_FCSR18
	REG_FCSR19
	REG_FCSR20
	REG_FCSR21
	REG_FCSR22
	REG_FCSR23
	REG_FCSR24
	REG_FCSR25
	REG_FCSR26
	REG_FCSR27
	REG_FCSR28
	REG_FCSR29
	REG_FCSR30
	REG_FCSR31

	REG_FCC0 // must be a multiple of 32
	REG_FCC1
	REG_FCC2
	REG_FCC3
	REG_FCC4
	REG_FCC5
	REG_FCC6
	REG_FCC7 // only eight registers are needed
	REG_FCC8
	REG_FCC9
	REG_FCC10
	REG_FCC11
	REG_FCC12
	REG_FCC13
	REG_FCC14
	REG_FCC15
	REG_FCC16
	REG_FCC17
	REG_FCC18
	REG_FCC19
	REG_FCC20
	REG_FCC21
	REG_FCC22
	REG_FCC23
	REG_FCC24
	REG_FCC25
	REG_FCC26
	REG_FCC27
	REG_FCC28
	REG_FCC29
	REG_FCC30
	REG_FCC31

	// LSX: 128-bit vector register
	REG_V0
	REG_V1
	REG_V2
	REG_V3
	REG_V4
	REG_V5
	REG_V6
	REG_V7
	REG_V8
	REG_V9
	REG_V10
	REG_V11
	REG_V12
	REG_V13
	REG_V14
	REG_V15
	REG_V16
	REG_V17
	REG_V18
	REG_V19
	REG_V20
	REG_V21
	REG_V22
	REG_V23
	REG_V24
	REG_V25
	REG_V26
	REG_V27
	REG_V28
	REG_V29
	REG_V30
	REG_V31

	// LASX: 256-bit vector register
	REG_X0
	REG_X1
	REG_X2
	REG_X3
	REG_X4
	REG_X5
	REG_X6
	REG_X7
	REG_X8
	REG_X9
	REG_X10
	REG_X11
	REG_X12
	REG_X13
	REG_X14
	REG_X15
	REG_X16
	REG_X17
	REG_X18
	REG_X19
	REG_X20
	REG_X21
	REG_X22
	REG_X23
	REG_X24
	REG_X25
	REG_X26
	REG_X27
	REG_X28
	REG_X29
	REG_X30
	REG_X31

	REG_LAST = REG_X31 // the last defined register

	REG_SPECIAL = REG_FCSR0

	REGZERO = REG_R0 // set to zero
	REGLINK = REG_R1
	REGSP   = REG_R3
	REGRET  = REG_R20 // not use
	REGARG  = -1      // -1 disables passing the first argument in register
	REGRT1  = REG_R20 // reserved for runtime, duffzero and duffcopy
	REGRT2  = REG_R21 // reserved for runtime, duffcopy
	REGCTXT = REG_R29 // context for closures
	REGG    = REG_R22 // G in loong64
	REGTMP  = REG_R30 // used by the assembler
	FREGRET = REG_F0  // not use
)

var LOONG64DWARFRegisters = map[int16]int16{}

func init() {
	// f assigns dwarfregisters[from:to] = (base):(to-from+base)
	f := func(from, to, base int16) {
		for r := int16(from); r <= to; r++ {
			LOONG64DWARFRegisters[r] = (r - from) + base
		}
	}
	f(REG_R0, REG_R31, 0)
	f(REG_F0, REG_F31, 32)

	// The lower bits of V and X registers are alias to F registers
	f(REG_V0, REG_V31, 32)
	f(REG_X0, REG_X31, 32)
}

const (
	BIG = 2046
)

const (
	// mark flags
	LABEL  = 1 << 0
	LEAF   = 1 << 1
	SYNC   = 1 << 2
	BRANCH = 1 << 3
)

const (
	C_NONE = iota
	C_REG
	C_FREG
	C_FCSRREG
	C_FCCREG
	C_VREG
	C_XREG
	C_ZCON
	C_SCON // 12 bit signed
	C_UCON // 32 bit signed, low 12 bits 0
	C_ADD0CON
	C_AND0CON
	C_ADDCON  // -0x800 <= v < 0
	C_ANDCON  // 0 < v <= 0xFFF
	C_LCON    // other 32
	C_DCON    // other 64 (could subdivide further)
	C_SACON   // $n(REG) where n <= int12
	C_LACON   // $n(REG) where int12 < n <= int32
	C_DACON   // $n(REG) where int32 < n
	C_EXTADDR // external symbol address
	C_BRAN
	C_SAUTO
	C_LAUTO
	C_ZOREG
	C_SOREG
	C_LOREG
	C_ROFF // register offset
	C_ADDR
	C_TLS_LE
	C_TLS_IE
	C_GOTADDR
	C_TEXTSIZE

	C_GOK
	C_NCLASS // must be the last
)

const (
	AABSD = obj.ABaseLoong64 + obj.A_ARCHSPECIFIC + iota
	AABSF
	AADD
	AADDD
	AADDF
	AADDU

	AADDW
	AAND
	ABEQ
	ABGEZ
	ABLEZ
	ABGTZ
	ABLTZ
	ABFPF
	ABFPT

	ABNE
	ABREAK

	ACMPEQD
	ACMPEQF

	ACMPGED // ACMPGED -> fcmp.sle.d
	ACMPGEF // ACMPGEF -> fcmp.sle.s
	ACMPGTD // ACMPGTD -> fcmp.slt.d
	ACMPGTF // ACMPGTF -> fcmp.slt.s

	ALU12IW
	ALU32ID
	ALU52ID
	APCALAU12I
	APCADDU12I
	AJIRL
	ABGE
	ABLT
	ABLTU
	ABGEU

	ADIV
	ADIVD
	ADIVF
	ADIVU
	ADIVW

	ALL
	ALLV

	ALUI

	AMOVB
	AMOVBU

	AMOVD
	AMOVDF
	AMOVDW
	AMOVF
	AMOVFD
	AMOVFW

	AMOVH
	AMOVHU
	AMOVW

	AMOVWD
	AMOVWF

	AMUL
	AMULD
	AMULF
	AMULU
	AMULH
	AMULHU
	AMULW
	ANEGD
	ANEGF

	ANEGW
	ANEGV

	ANOOP // hardware nop
	ANOR
	AOR
	AREM
	AREMU

	ARFE

	ASC
	ASCV

	ASGT
	ASGTU

	ASLL
	ASQRTD
	ASQRTF
	ASRA
	ASRL
	AROTR
	ASUB
	ASUBD
	ASUBF

	ASUBU
	ASUBW
	ADBAR
	ASYSCALL

	ATEQ
	ATNE

	AWORD

	AXOR

	AMASKEQZ
	AMASKNEZ

	// 64-bit
	AMOVV

	ASLLV
	ASRAV
	ASRLV
	AROTRV
	ADIVV
	ADIVVU

	AREMV
	AREMVU

	AMULV
	AMULVU
	AMULHV
	AMULHVU
	AADDV
	AADDVU
	ASUBV
	ASUBVU

	// 64-bit FP
	ATRUNCFV
	ATRUNCDV
	ATRUNCFW
	ATRUNCDW

	AMOVWU
	AMOVFV
	AMOVDV
	AMOVVF
	AMOVVD

	// 2.2.1.8
	AORN
	AANDN

	// 2.2.7. Atomic Memory Access Instructions
	AAMSWAPB
	AAMSWAPH
	AAMSWAPW
	AAMSWAPV
	AAMCASB
	AAMCASH
	AAMCASW
	AAMCASV
	AAMADDW
	AAMADDV
	AAMANDW
	AAMANDV
	AAMORW
	AAMORV
	AAMXORW
	AAMXORV
	AAMMAXW
	AAMMAXV
	AAMMINW
	AAMMINV
	AAMMAXWU
	AAMMAXVU
	AAMMINWU
	AAMMINVU
	AAMSWAPDBB
	AAMSWAPDBH
	AAMSWAPDBW
	AAMSWAPDBV
	AAMCASDBB
	AAMCASDBH
	AAMCASDBW
	AAMCASDBV
	AAMADDDBW
	AAMADDDBV
	AAMANDDBW
	AAMANDDBV
	AAMORDBW
	AAMORDBV
	AAMXORDBW
	AAMXORDBV
	AAMMAXDBW
	AAMMAXDBV
	AAMMINDBW
	AAMMINDBV
	AAMMAXDBWU
	AAMMAXDBVU
	AAMMINDBWU
	AAMMINDBVU

	// 2.2.3.1
	AEXTWB
	AEXTWH

	// 2.2.3.2
	ACLOW
	ACLOV
	ACLZW
	ACLZV
	ACTOW
	ACTOV
	ACTZW
	ACTZV

	// 2.2.3.4
	AREVBV
	AREVB2W
	AREVB4H
	AREVB2H

	// 2.2.3.5
	AREVH2W
	AREVHV

	// 2.2.3.6
	ABITREV4B
	ABITREV8B

	// 2.2.3.7
	ABITREVW
	ABITREVV

	// 2.2.3.8
	ABSTRINSW
	ABSTRINSV

	// 2.2.3.9
	ABSTRPICKW
	ABSTRPICKV

	// 2.2.9. CRC Check Instructions
	ACRCWBW
	ACRCWHW
	ACRCWWW
	ACRCWVW
	ACRCCWBW
	ACRCCWHW
	ACRCCWWW
	ACRCCWVW

	// 2.2.10. Other Miscellaneous Instructions
	ARDTIMELW
	ARDTIMEHW
	ARDTIMED
	ACPUCFG

	// 3.2.1.2
	AFMADDF
	AFMADDD
	AFMSUBF
	AFMSUBD
	AFNMADDF
	AFNMADDD
	AFNMSUBF
	AFNMSUBD

	// 3.2.1.3
	AFMINF
	AFMIND
	AFMAXF
	AFMAXD

	// 3.2.1.7
	AFCOPYSGF
	AFCOPYSGD
	AFSCALEBF
	AFSCALEBD
	AFLOGBF
	AFLOGBD

	// 3.2.1.8
	AFCLASSF
	AFCLASSD

	// 3.2.3.2
	AFFINTFW
	AFFINTFV
	AFFINTDW
	AFFINTDV
	AFTINTWF
	AFTINTWD
	AFTINTVF
	AFTINTVD

	// 3.2.3.3
	AFTINTRPWF
	AFTINTRPWD
	AFTINTRPVF
	AFTINTRPVD
	AFTINTRMWF
	AFTINTRMWD
	AFTINTRMVF
	AFTINTRMVD
	AFTINTRZWF
	AFTINTRZWD
	AFTINTRZVF
	AFTINTRZVD
	AFTINTRNEWF
	AFTINTRNEWD
	AFTINTRNEVF
	AFTINTRNEVD

	// LSX and LASX memory access instructions
	AVMOVQ
	AXVMOVQ

	// LSX and LASX integer comparison instruction
	AVSEQB
	AXVSEQB
	AVSEQH
	AXVSEQH
	AVSEQW
	AXVSEQW
	AVSEQV
	AXVSEQV

	ALAST

	// aliases
	AJMP = obj.AJMP
	AJAL = obj.ACALL
	ARET = obj.ARET
)

func init() {
	// The asm encoder generally assumes that the lowest 5 bits of the
	// REG_XX constants match the machine instruction encoding, i.e.
	// the lowest 5 bits is the register number.
	// Check this here.
	if REG_R0%32 != 0 {
		panic("REG_R0 is not a multiple of 32")
	}
	if REG_F0%32 != 0 {
		panic("REG_F0 is not a multiple of 32")
	}
	if REG_FCSR0%32 != 0 {
		panic("REG_FCSR0 is not a multiple of 32")
	}
	if REG_FCC0%32 != 0 {
		panic("REG_FCC0 is not a multiple of 32")
	}
	if REG_V0%32 != 0 {
		panic("REG_V0 is not a multiple of 32")
	}
	if REG_X0%32 != 0 {
		panic("REG_X0 is not a multiple of 32")
	}
}
