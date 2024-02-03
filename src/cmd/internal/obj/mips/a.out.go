// cmd/9c/9.out.h from Vita Nuova.
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2008 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2008 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package mips

import (
	"cmd/internal/obj"
)

//go:generate go run ../stringer.go -i $GOFILE -o anames.go -p mips

/*
 * mips 64
 */
const (
	NSNAME = 8
	NSYM   = 50
	NREG   = 32 /* number of general registers */
	NFREG  = 32 /* number of floating point registers */
	NWREG  = 32 /* number of MSA registers */
)

const (
	REG_R0 = obj.RBaseMIPS + iota // must be a multiple of 32
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

	// co-processor 0 control registers
	REG_M0 // must be a multiple of 32
	REG_M1
	REG_M2
	REG_M3
	REG_M4
	REG_M5
	REG_M6
	REG_M7
	REG_M8
	REG_M9
	REG_M10
	REG_M11
	REG_M12
	REG_M13
	REG_M14
	REG_M15
	REG_M16
	REG_M17
	REG_M18
	REG_M19
	REG_M20
	REG_M21
	REG_M22
	REG_M23
	REG_M24
	REG_M25
	REG_M26
	REG_M27
	REG_M28
	REG_M29
	REG_M30
	REG_M31

	// FPU control registers
	REG_FCR0 // must be a multiple of 32
	REG_FCR1
	REG_FCR2
	REG_FCR3
	REG_FCR4
	REG_FCR5
	REG_FCR6
	REG_FCR7
	REG_FCR8
	REG_FCR9
	REG_FCR10
	REG_FCR11
	REG_FCR12
	REG_FCR13
	REG_FCR14
	REG_FCR15
	REG_FCR16
	REG_FCR17
	REG_FCR18
	REG_FCR19
	REG_FCR20
	REG_FCR21
	REG_FCR22
	REG_FCR23
	REG_FCR24
	REG_FCR25
	REG_FCR26
	REG_FCR27
	REG_FCR28
	REG_FCR29
	REG_FCR30
	REG_FCR31

	// MSA registers
	// The lower bits of W registers are alias to F registers
	REG_W0 // must be a multiple of 32
	REG_W1
	REG_W2
	REG_W3
	REG_W4
	REG_W5
	REG_W6
	REG_W7
	REG_W8
	REG_W9
	REG_W10
	REG_W11
	REG_W12
	REG_W13
	REG_W14
	REG_W15
	REG_W16
	REG_W17
	REG_W18
	REG_W19
	REG_W20
	REG_W21
	REG_W22
	REG_W23
	REG_W24
	REG_W25
	REG_W26
	REG_W27
	REG_W28
	REG_W29
	REG_W30
	REG_W31

	REG_HI
	REG_LO

	REG_LAST = REG_LO // the last defined register

	REG_SPECIAL = REG_M0

	REGZERO = REG_R0 /* set to zero */
	REGSP   = REG_R29
	REGSB   = REG_R28
	REGLINK = REG_R31
	REGRET  = REG_R1
	REGARG  = -1      /* -1 disables passing the first argument in register */
	REGRT1  = REG_R1  /* reserved for runtime, duffzero and duffcopy */
	REGRT2  = REG_R2  /* reserved for runtime, duffcopy */
	REGCTXT = REG_R22 /* context for closures */
	REGG    = REG_R30 /* G */
	REGTMP  = REG_R23 /* used by the linker */
	FREGRET = REG_F0
)

// https://llvm.org/svn/llvm-project/llvm/trunk/lib/Target/Mips/MipsRegisterInfo.td search for DwarfRegNum
// https://gcc.gnu.org/viewcvs/gcc/trunk/gcc/config/mips/mips.c?view=co&revision=258099&content-type=text%2Fplain search for mips_dwarf_regno
// For now, this is adequate for both 32 and 64 bit.
var MIPSDWARFRegisters = map[int16]int16{}

func init() {
	// f assigns dwarfregisters[from:to] = (base):(to-from+base)
	f := func(from, to, base int16) {
		for r := int16(from); r <= to; r++ {
			MIPSDWARFRegisters[r] = (r - from) + base
		}
	}
	f(REG_R0, REG_R31, 0)
	f(REG_F0, REG_F31, 32) // For 32-bit MIPS, compiler only uses even numbered registers --  see cmd/compile/internal/ssa/gen/MIPSOps.go
	MIPSDWARFRegisters[REG_HI] = 64
	MIPSDWARFRegisters[REG_LO] = 65
	// The lower bits of W registers are alias to F registers
	f(REG_W0, REG_W31, 32)
}

const (
	BIG = 32766
)

const (
	/* mark flags */
	FOLL    = 1 << 0
	LABEL   = 1 << 1
	LEAF    = 1 << 2
	SYNC    = 1 << 3
	BRANCH  = 1 << 4
	LOAD    = 1 << 5
	FCMP    = 1 << 6
	NOSCHED = 1 << 7

	NSCHED = 20
)

const (
	C_NONE = iota
	C_REG
	C_FREG
	C_FCREG
	C_MREG /* special processor register */
	C_WREG /* MSA registers */
	C_HI
	C_LO
	C_ZCON
	C_SCON /* 16 bit signed */
	C_UCON /* 32 bit signed, low 16 bits 0 */
	C_ADD0CON
	C_AND0CON
	C_ADDCON /* -0x8000 <= v < 0 */
	C_ANDCON /* 0 < v <= 0xFFFF */
	C_LCON   /* other 32 */
	C_DCON   /* other 64 (could subdivide further) */
	C_SACON  /* $n(REG) where n <= int16 */
	C_SECON
	C_LACON /* $n(REG) where int16 < n <= int32 */
	C_LECON
	C_DACON /* $n(REG) where int32 < n */
	C_STCON /* $tlsvar */
	C_SBRA
	C_LBRA
	C_SAUTO
	C_LAUTO
	C_SEXT
	C_LEXT
	C_ZOREG
	C_SOREG
	C_LOREG
	C_GOK
	C_ADDR
	C_TLS
	C_TEXTSIZE

	C_NCLASS /* must be the last */
)

const (
	AABSD = obj.ABaseMIPS + obj.A_ARCHSPECIFIC + iota
	AABSF
	AABSW
	AADD
	AADDD
	AADDF
	AADDU
	AADDW
	AAND
	ABEQ
	ABFPF
	ABFPT
	ABGEZ
	ABGEZAL
	ABGTZ
	ABLEZ
	ABLTZ
	ABLTZAL
	ABNE
	ABREAK
	ACLO
	ACLZ
	ACMOVF
	ACMOVN
	ACMOVT
	ACMOVZ
	ACMPEQD
	ACMPEQF
	ACMPGED
	ACMPGEF
	ACMPGTD
	ACMPGTF
	ADIV
	ADIVD
	ADIVF
	ADIVU
	ADIVW
	AGOK
	ALL
	ALLV
	ALUI
	AMADD
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
	AMOVWL
	AMOVWR
	AMSUB
	AMUL
	AMULD
	AMULF
	AMULU
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
	AROTR
	AROTRV
	ASC
	ASCV
	ASEB
	ASEH
	ASGT
	ASGTU
	ASLL
	ASQRTD
	ASQRTF
	ASRA
	ASRL
	ASUB
	ASUBD
	ASUBF
	ASUBU
	ASUBW
	ASYNC
	ASYSCALL
	ATEQ
	ATLBP
	ATLBR
	ATLBWI
	ATLBWR
	ATNE
	AWORD
	AWSBH
	AXOR

	/* 64-bit */
	AMOVV
	AMOVVL
	AMOVVR
	ASLLV
	ASRAV
	ASRLV
	ADIVV
	ADIVVU
	AREMV
	AREMVU
	AMULV
	AMULVU
	AADDV
	AADDVU
	ASUBV
	ASUBVU
	ADSBH
	ADSHD

	/* 64-bit FP */
	ATRUNCFV
	ATRUNCDV
	ATRUNCFW
	ATRUNCDW
	AMOVWU
	AMOVFV
	AMOVDV
	AMOVVF
	AMOVVD

	/* MSA */
	AVMOVB
	AVMOVH
	AVMOVW
	AVMOVD

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
	if REG_M0%32 != 0 {
		panic("REG_M0 is not a multiple of 32")
	}
	if REG_FCR0%32 != 0 {
		panic("REG_FCR0 is not a multiple of 32")
	}
	if REG_W0%32 != 0 {
		panic("REG_W0 is not a multiple of 32")
	}
}
