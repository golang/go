// cmd/7c/7.out.h  from Vita Nuova.
// https://code.google.com/p/ken-cc/source/browse/src/cmd/7c/7.out.h
//
// 	Copyright © 1994-1999 Lucent Technologies Inc. All rights reserved.
// 	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
// 	Portions Copyright © 1997-1999 Vita Nuova Limited
// 	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
// 	Portions Copyright © 2004,2006 Bruce Ellis
// 	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
// 	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
// 	Portions Copyright © 2009 The Go Authors. All rights reserved.
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

package arm64

import "cmd/internal/obj"

const (
	NSNAME = 8
	NSYM   = 50
	NREG   = 32 /* number of general registers */
	NFREG  = 32 /* number of floating point registers */
)

// General purpose registers, kept in the low bits of Prog.Reg.
const (
	// integer
	REG_R0 = obj.RBaseARM64 + iota
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

	// scalar floating point
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

	// SIMD
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

	// The EQ in
	// 	CSET	EQ, R0
	// is encoded as TYPE_REG, even though it's not really a register.
	COND_EQ
	COND_NE
	COND_HS
	COND_LO
	COND_MI
	COND_PL
	COND_VS
	COND_VC
	COND_HI
	COND_LS
	COND_GE
	COND_LT
	COND_GT
	COND_LE
	COND_AL
	COND_NV

	REG_RSP = REG_V31 + 32 // to differentiate ZR/SP, REG_RSP&0x1f = 31
)

// Not registers, but flags that can be combined with regular register
// constants to indicate extended register conversion. When checking,
// you should subtract obj.RBaseARM64 first. From this difference, bit 11
// indicates extended register, bits 8-10 select the conversion mode.
const REG_EXT = obj.RBaseARM64 + 1<<11

const (
	REG_UXTB = REG_EXT + iota<<8
	REG_UXTH
	REG_UXTW
	REG_UXTX
	REG_SXTB
	REG_SXTH
	REG_SXTW
	REG_SXTX
)

// Special registers, after subtracting obj.RBaseARM64, bit 12 indicates
// a special register and the low bits select the register.
const (
	REG_SPECIAL = obj.RBaseARM64 + 1<<12 + iota
	REG_DAIF
	REG_NZCV
	REG_FPSR
	REG_FPCR
	REG_SPSR_EL1
	REG_ELR_EL1
	REG_SPSR_EL2
	REG_ELR_EL2
	REG_CurrentEL
	REG_SP_EL0
	REG_SPSel
	REG_DAIFSet
	REG_DAIFClr
)

// Register assignments:
//
// compiler allocates R0 up as temps
// compiler allocates register variables R7-R25
// compiler allocates external registers R26 down
//
// compiler allocates register variables F7-F26
// compiler allocates external registers F26 down
const (
	REGMIN = REG_R7  // register variables allocated from here to REGMAX
	REGRT1 = REG_R16 // ARM64 IP0, for external linker, runtime, duffzero and duffcopy
	REGRT2 = REG_R17 // ARM64 IP1, for external linker, runtime, duffcopy
	REGPR  = REG_R18 // ARM64 platform register, unused in the Go toolchain
	REGMAX = REG_R25

	REGCTXT = REG_R26 // environment for closures
	REGTMP  = REG_R27 // reserved for liblink
	REGG    = REG_R28 // G
	REGFP   = REG_R29 // frame pointer, unused in the Go toolchain
	REGLINK = REG_R30

	// ARM64 uses R31 as both stack pointer and zero register,
	// depending on the instruction. To differentiate RSP from ZR,
	// we use a different numeric value for REGZERO and REGSP.
	REGZERO = REG_R31
	REGSP   = REG_RSP

	FREGRET  = REG_F0
	FREGMIN  = REG_F7  // first register variable
	FREGMAX  = REG_F26 // last register variable for 7g only
	FREGEXT  = REG_F26 // first external register
	FREGZERO = REG_F28 // both float and double
	FREGHALF = REG_F29 // double
	FREGONE  = REG_F30 // double
	FREGTWO  = REG_F31 // double
)

const (
	BIG = 2048 - 8
)

const (
	/* mark flags */
	LABEL = 1 << iota
	LEAF
	FLOAT
	BRANCH
	LOAD
	FCMP
	SYNC
	LIST
	FOLL
	NOSCHED
)

const (
	C_NONE   = iota
	C_REG    // R0..R30
	C_RSP    // R0..R30, RSP
	C_FREG   // F0..F31
	C_VREG   // V0..V31
	C_PAIR   // (Rn, Rm)
	C_SHIFT  // Rn<<2
	C_EXTREG // Rn.UXTB<<3
	C_SPR    // REG_NZCV
	C_COND   // EQ, NE, etc

	C_ZCON     // $0 or ZR
	C_ADDCON0  // 12-bit unsigned, unshifted
	C_ADDCON   // 12-bit unsigned, shifted left by 0 or 12
	C_MOVCON   // generated by a 16-bit constant, optionally inverted and/or shifted by multiple of 16
	C_BITCON   // bitfield and logical immediate masks
	C_ABCON    // could be C_ADDCON or C_BITCON
	C_MBCON    // could be C_MOVCON or C_BITCON
	C_LCON     // 32-bit constant
	C_VCON     // 64-bit constant
	C_FCON     // floating-point constant
	C_VCONADDR // 64-bit memory address

	C_AACON // ADDCON offset in auto constant $a(FP)
	C_LACON // 32-bit offset in auto constant $a(FP)
	C_AECON // ADDCON offset in extern constant $e(SB)

	// TODO(aram): only one branch class should be enough
	C_SBRA // for TYPE_BRANCH
	C_LBRA

	C_NPAUTO   // -512 <= x < 0, 0 mod 8
	C_NSAUTO   // -256 <= x < 0
	C_PSAUTO   // 0 to 255
	C_PPAUTO   // 0 to 504, 0 mod 8
	C_UAUTO4K  // 0 to 4095
	C_UAUTO8K  // 0 to 8190, 0 mod 2
	C_UAUTO16K // 0 to 16380, 0 mod 4
	C_UAUTO32K // 0 to 32760, 0 mod 8
	C_UAUTO64K // 0 to 65520, 0 mod 16
	C_LAUTO    // any other 32-bit constant

	C_SEXT1  // 0 to 4095, direct
	C_SEXT2  // 0 to 8190
	C_SEXT4  // 0 to 16380
	C_SEXT8  // 0 to 32760
	C_SEXT16 // 0 to 65520
	C_LEXT

	// TODO(aram): s/AUTO/INDIR/
	C_ZOREG  // 0(R)
	C_NPOREG // mirror NPAUTO, etc
	C_NSOREG
	C_PSOREG
	C_PPOREG
	C_UOREG4K
	C_UOREG8K
	C_UOREG16K
	C_UOREG32K
	C_UOREG64K
	C_LOREG

	C_ADDR // TODO(aram): explain difference from C_VCONADDR

	// The GOT slot for a symbol in -dynlink mode.
	C_GOTADDR

	// TLS "var" in local exec mode: will become a constant offset from
	// thread local base that is ultimately chosen by the program linker.
	C_TLS_LE

	// TLS "var" in initial exec mode: will become a memory address (chosen
	// by the program linker) that the dynamic linker will fill with the
	// offset from the thread local base.
	C_TLS_IE

	C_ROFF // register offset (including register extended)

	C_GOK
	C_TEXTSIZE
	C_NCLASS // must be last
)

const (
	C_XPRE  = 1 << 6 // match arm.C_WBIT, so Prog.String know how to print it
	C_XPOST = 1 << 5 // match arm.C_PBIT, so Prog.String know how to print it
)

//go:generate go run ../stringer.go -i $GOFILE -o anames.go -p arm64

const (
	AADC = obj.ABaseARM64 + obj.A_ARCHSPECIFIC + iota
	AADCS
	AADCSW
	AADCW
	AADD
	AADDS
	AADDSW
	AADDW
	AADR
	AADRP
	AAND
	AANDS
	AANDSW
	AANDW
	AASR
	AASRW
	AAT
	ABFI
	ABFIW
	ABFM
	ABFMW
	ABFXIL
	ABFXILW
	ABIC
	ABICS
	ABICSW
	ABICW
	ABRK
	ACBNZ
	ACBNZW
	ACBZ
	ACBZW
	ACCMN
	ACCMNW
	ACCMP
	ACCMPW
	ACINC
	ACINCW
	ACINV
	ACINVW
	ACLREX
	ACLS
	ACLSW
	ACLZ
	ACLZW
	ACMN
	ACMNW
	ACMP
	ACMPW
	ACNEG
	ACNEGW
	ACRC32B
	ACRC32CB
	ACRC32CH
	ACRC32CW
	ACRC32CX
	ACRC32H
	ACRC32W
	ACRC32X
	ACSEL
	ACSELW
	ACSET
	ACSETM
	ACSETMW
	ACSETW
	ACSINC
	ACSINCW
	ACSINV
	ACSINVW
	ACSNEG
	ACSNEGW
	ADC
	ADCPS1
	ADCPS2
	ADCPS3
	ADMB
	ADRPS
	ADSB
	AEON
	AEONW
	AEOR
	AEORW
	AERET
	AEXTR
	AEXTRW
	AHINT
	AHLT
	AHVC
	AIC
	AISB
	ALDAR
	ALDARB
	ALDARH
	ALDARW
	ALDAXP
	ALDAXPW
	ALDAXR
	ALDAXRB
	ALDAXRH
	ALDAXRW
	ALDP
	ALDXR
	ALDXRB
	ALDXRH
	ALDXRW
	ALDXP
	ALDXPW
	ALSL
	ALSLW
	ALSR
	ALSRW
	AMADD
	AMADDW
	AMNEG
	AMNEGW
	AMOVK
	AMOVKW
	AMOVN
	AMOVNW
	AMOVZ
	AMOVZW
	AMRS
	AMSR
	AMSUB
	AMSUBW
	AMUL
	AMULW
	AMVN
	AMVNW
	ANEG
	ANEGS
	ANEGSW
	ANEGW
	ANGC
	ANGCS
	ANGCSW
	ANGCW
	AORN
	AORNW
	AORR
	AORRW
	APRFM
	APRFUM
	ARBIT
	ARBITW
	AREM
	AREMW
	AREV
	AREV16
	AREV16W
	AREV32
	AREVW
	AROR
	ARORW
	ASBC
	ASBCS
	ASBCSW
	ASBCW
	ASBFIZ
	ASBFIZW
	ASBFM
	ASBFMW
	ASBFX
	ASBFXW
	ASDIV
	ASDIVW
	ASEV
	ASEVL
	ASMADDL
	ASMC
	ASMNEGL
	ASMSUBL
	ASMULH
	ASMULL
	ASTXR
	ASTXRB
	ASTXRH
	ASTXP
	ASTXPW
	ASTXRW
	ASTLP
	ASTLPW
	ASTLR
	ASTLRB
	ASTLRH
	ASTLRW
	ASTLXP
	ASTLXPW
	ASTLXR
	ASTLXRB
	ASTLXRH
	ASTLXRW
	ASTP
	ASUB
	ASUBS
	ASUBSW
	ASUBW
	ASVC
	ASXTB
	ASXTBW
	ASXTH
	ASXTHW
	ASXTW
	ASYS
	ASYSL
	ATBNZ
	ATBZ
	ATLBI
	ATST
	ATSTW
	AUBFIZ
	AUBFIZW
	AUBFM
	AUBFMW
	AUBFX
	AUBFXW
	AUDIV
	AUDIVW
	AUMADDL
	AUMNEGL
	AUMSUBL
	AUMULH
	AUMULL
	AUREM
	AUREMW
	AUXTB
	AUXTH
	AUXTW
	AUXTBW
	AUXTHW
	AWFE
	AWFI
	AYIELD
	AMOVB
	AMOVBU
	AMOVH
	AMOVHU
	AMOVW
	AMOVWU
	AMOVD
	AMOVNP
	AMOVNPW
	AMOVP
	AMOVPD
	AMOVPQ
	AMOVPS
	AMOVPSW
	AMOVPW
	ABEQ
	ABNE
	ABCS
	ABHS
	ABCC
	ABLO
	ABMI
	ABPL
	ABVS
	ABVC
	ABHI
	ABLS
	ABGE
	ABLT
	ABGT
	ABLE
	AFABSD
	AFABSS
	AFADDD
	AFADDS
	AFCCMPD
	AFCCMPED
	AFCCMPS
	AFCCMPES
	AFCMPD
	AFCMPED
	AFCMPES
	AFCMPS
	AFCVTSD
	AFCVTDS
	AFCVTZSD
	AFCVTZSDW
	AFCVTZSS
	AFCVTZSSW
	AFCVTZUD
	AFCVTZUDW
	AFCVTZUS
	AFCVTZUSW
	AFDIVD
	AFDIVS
	AFMOVD
	AFMOVS
	AFMULD
	AFMULS
	AFNEGD
	AFNEGS
	AFSQRTD
	AFSQRTS
	AFSUBD
	AFSUBS
	ASCVTFD
	ASCVTFS
	ASCVTFWD
	ASCVTFWS
	AUCVTFD
	AUCVTFS
	AUCVTFWD
	AUCVTFWS
	AWORD
	ADWORD
	AFCSELS
	AFCSELD
	AFMAXS
	AFMINS
	AFMAXD
	AFMIND
	AFMAXNMS
	AFMAXNMD
	AFNMULS
	AFNMULD
	AFRINTNS
	AFRINTND
	AFRINTPS
	AFRINTPD
	AFRINTMS
	AFRINTMD
	AFRINTZS
	AFRINTZD
	AFRINTAS
	AFRINTAD
	AFRINTXS
	AFRINTXD
	AFRINTIS
	AFRINTID
	AFMADDS
	AFMADDD
	AFMSUBS
	AFMSUBD
	AFNMADDS
	AFNMADDD
	AFNMSUBS
	AFNMSUBD
	AFMINNMS
	AFMINNMD
	AFCVTDH
	AFCVTHS
	AFCVTHD
	AFCVTSH
	AAESD
	AAESE
	AAESIMC
	AAESMC
	ASHA1C
	ASHA1H
	ASHA1M
	ASHA1P
	ASHA1SU0
	ASHA1SU1
	ASHA256H
	ASHA256H2
	ASHA256SU0
	ASHA256SU1
	ALAST
	AB  = obj.AJMP
	ABL = obj.ACALL
)
