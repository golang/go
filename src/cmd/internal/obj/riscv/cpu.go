//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2008 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2008 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
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

package riscv

import "cmd/internal/obj"

//go:generate go run ../stringer.go -i $GOFILE -o anames.go -p riscv

const (
	// Base register numberings.
	REG_X0 = obj.RBaseRISCV + iota
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

	// FP register numberings.
	REG_F0
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

	// Special/control registers.
	// TODO(myenik) Read more and add the ones we need...

	// This marks the end of the register numbering.
	REG_END

	// General registers reassigned to ABI names.
	REG_ZERO = REG_X0
	REG_RA   = REG_X1
	REG_FP   = REG_X2
	REG_S0   = REG_X2 // FP and S0 are the same.
	REG_S1   = REG_X3
	REG_S2   = REG_X4
	REG_S3   = REG_X5
	REG_S4   = REG_X6
	REG_S5   = REG_X7
	REG_S6   = REG_X8
	REG_S7   = REG_X9
	REG_S8   = REG_X10
	REG_S9   = REG_X11
	REG_S10  = REG_X12
	REG_S11  = REG_X13
	REG_SP   = REG_X14
	REG_TP   = REG_X15
	REG_V0   = REG_X16
	REG_V1   = REG_X17
	REG_A0   = REG_X18
	REG_A1   = REG_X19
	REG_A2   = REG_X20
	REG_A3   = REG_X21
	REG_A4   = REG_X22
	REG_A5   = REG_X23
	REG_A6   = REG_X24
	REG_A7   = REG_X25
	REG_T0   = REG_X26
	REG_T1   = REG_X27
	REG_T2   = REG_X28
	REG_T3   = REG_X29
	REG_T4   = REG_X30
	REG_GP   = REG_X31

	// Golang runtime register names.
	// TODO(myenik) Revisit these mappings.
	REG_SB   = REG_S1 // First saved register reserved for SB.
	REG_RT1  = REG_S2 // Reserved for runtime (duffzero and duffcopy), second saved register.
	REG_RT2  = REG_S3 // Reserved for runtime (duffcopy), third saved register.
	REG_CTXT = REG_S4 // Context for closures.
	REG_G    = REG_S5 // G pointer.

	// ABI names for floating point registers.
	REG_FS0  = REG_F0
	REG_FS1  = REG_F1
	REG_FS2  = REG_F2
	REG_FS3  = REG_F3
	REG_FS4  = REG_F4
	REG_FS5  = REG_F5
	REG_FS6  = REG_F6
	REG_FS7  = REG_F7
	REG_FS8  = REG_F8
	REG_FS9  = REG_F9
	REG_FS10 = REG_F10
	REG_FS11 = REG_F11
	REG_FS12 = REG_F12
	REG_FS13 = REG_F13
	REG_FS14 = REG_F14
	REG_FS15 = REG_F15
	REG_FV0  = REG_F16
	REG_FV1  = REG_F17
	REG_FA0  = REG_F18
	REG_FA1  = REG_F19
	REG_FA2  = REG_F20
	REG_FA3  = REG_F21
	REG_FA4  = REG_F22
	REG_FA5  = REG_F23
	REG_FA6  = REG_F24
	REG_FA7  = REG_F25
	REG_FT0  = REG_F26
	REG_FT1  = REG_F27
	REG_FT2  = REG_F28
	REG_FT3  = REG_F29
	REG_FT4  = REG_F30
	REG_FT5  = REG_F31
)

// TEXTFLAG definitions.
const (
	/* mark flags */
	LABEL   = 1 << 0
	LEAF    = 1 << 1
	FLOAT   = 1 << 2
	BRANCH  = 1 << 3
	LOAD    = 1 << 4
	FCMP    = 1 << 5
	SYNC    = 1 << 6
	LIST    = 1 << 7
	FOLL    = 1 << 8
	NOSCHED = 1 << 9
)

// These are taken straight from table 8.2 in the RISCV user ISA v2.0 document.
// Instructions that are commented out were duplicates in other sections of the
// table.
// TODO(myenik) remove these duplicates if they are not needed.
const (
	// RV32I
	ALUI = obj.ABaseRISCV + obj.A_ARCHSPECIFIC + iota
	AAUIPC
	AJAL
	AJALR
	ABEQ
	ABNE
	ABLT
	ABGE
	ABLTU
	ABGEU
	ALB
	ALH
	ALBU
	ALHU
	ASB
	ASH
	ASW
	AADDI
	ASLTI
	ASLTIU
	AXORI
	AORI
	AANDI
	ASLLI
	ASRLI
	ASRAI
	AADD
	ASUB
	ASLL
	ASLT
	ASLTU
	AXOR
	ASRL
	ASRA
	AOR
	AAND
	AFENCE
	AFENCEI
	ASCALL
	ASBREAK
	ARDCYCLE
	ARDCYCLEH
	ARDTIME
	ARDTIMEH
	ARDINSTRET
	ARDINSTRETH

	// RV64I
	ALWU
	ALD
	ASD
	// SLLI
	// SRLI
	// SRAI
	ASLLIW
	ASRLIW
	ASRAIW
	AADDW
	ASUBW
	ASLLW
	ASRLW
	ASRAW

	// RV32M
	AMUL
	AMULH
	AMULHSU
	AMULHU
	ADIV
	ADIVU
	AREM
	AREMU

	// RV64M
	AMULW
	ADIVW
	ADIVUW
	AREMW
	AREMUW

	// RV32A
	ALRW
	ASCW
	AAMOSWAPW
	AAMOADDW
	AAMOXORW
	AAMOANDW
	AAMOORW
	AAMOMINW
	AAMOMAXW
	AAMOMINUW
	AAMOMAXUW

	// RV64A
	ALRD
	ASCD
	AAMOSWAPD
	AAMOADDD
	AAMOXORD
	AAMOANDD
	AAMOORD
	AAMOMIND
	AAMOMAXD
	AAMOMINUD
	AAMOMAXUD

	//RV32F
	AFLW
	AFSW
	AFMADDS
	AFMSUBS
	AFNMSUBS
	AFNMADDS
	AFADDS
	AFSUBS
	AFMULS
	AFDIVS
	AFSQRTS
	AFSGNJS
	AFSGNJNS
	AFMINS
	AFMAXS
	AFCVTWS
	AFCVTWUS
	AFMVXS
	AFEQS
	AFLTS
	AFLES
	AFCLASSS
	AFCVTSW
	AFCVTSWU
	AFMVSX
	AFRCSR
	AFRRM
	AFRFLAGS
	AFSCSR
	AFSRM
	AFSFLAGS
	AFSRMI
	AFSFLAGSI

	// RV64F
	AFCVTLS
	AFCVTLUS
	AFCVTSL
	AFCVTSLU

	// RV32D
	AFLD
	AFSD
	AFMADDD
	AFMSUBD
	AFNMSUBD
	AFNMADDD
	AFADDD
	AFSUBD
	AFMULD
	AFDIVD
	AFSQRTD
	AFSGNJD
	AFSGNJND
	AFMIND
	AFMAXD
	AFCVTWD
	AFCVTWUD
	//AFMVXD
	AFEQD
	AFLTD
	AFLED
	AFCLASSD
	AFCVTDW
	AFCVTDWU
	//AFMVDX

	// RV64D
	AFCVTLD
	AFCVTLUD
	AFMVXD
	AFCVTDL
	AFCVTDLU
	AFMVDX
)
