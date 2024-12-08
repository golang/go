// cmd/7c/7.out.h  from Vita Nuova.
// https://bitbucket.org/plan9-from-bell-labs/9-cc/src/master/src/cmd/7c/7.out.h
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

	REG_RSP = REG_V31 + 32 // to differentiate ZR/SP, REG_RSP&0x1f = 31
)

// bits 0-4 indicates register: Vn
// bits 5-8 indicates arrangement: <T>
const (
	REG_ARNG = obj.RBaseARM64 + 1<<10 + iota<<9 // Vn.<T>
	REG_ELEM                                    // Vn.<T>[index]
	REG_ELEM_END
)

// Not registers, but flags that can be combined with regular register
// constants to indicate extended register conversion. When checking,
// you should subtract obj.RBaseARM64 first. From this difference, bit 11
// indicates extended register, bits 8-10 select the conversion mode.
// REG_LSL is the index shift specifier, bit 9 indicates shifted offset register.
const REG_LSL = obj.RBaseARM64 + 1<<9
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
// SYSREG_END is the last item in the automatically generated system register
// declaration, and it is defined in the sysRegEnc.go file.
// Define the special register after REG_SPECIAL, the first value of it should be
// REG_{name} = SYSREG_END + iota.
const (
	REG_SPECIAL = obj.RBaseARM64 + 1<<12
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
	REGRT1 = REG_R16 // ARM64 IP0, external linker may use as a scratch register in trampoline
	REGRT2 = REG_R17 // ARM64 IP1, external linker may use as a scratch register in trampoline
	REGPR  = REG_R18 // ARM64 platform register, unused in the Go toolchain
	REGMAX = REG_R25

	REGCTXT = REG_R26 // environment for closures
	REGTMP  = REG_R27 // reserved for liblink
	REGG    = REG_R28 // G
	REGFP   = REG_R29 // frame pointer
	REGLINK = REG_R30

	// ARM64 uses R31 as both stack pointer and zero register,
	// depending on the instruction. To differentiate RSP from ZR,
	// we use a different numeric value for REGZERO and REGSP.
	REGZERO = REG_R31
	REGSP   = REG_RSP

	FREGRET = REG_F0
	FREGMIN = REG_F7  // first register variable
	FREGMAX = REG_F26 // last register variable for 7g only
	FREGEXT = REG_F26 // first external register
)

// http://infocenter.arm.com/help/topic/com.arm.doc.ecm0665627/abi_sve_aadwarf_100985_0000_00_en.pdf
var ARM64DWARFRegisters = map[int16]int16{
	REG_R0:  0,
	REG_R1:  1,
	REG_R2:  2,
	REG_R3:  3,
	REG_R4:  4,
	REG_R5:  5,
	REG_R6:  6,
	REG_R7:  7,
	REG_R8:  8,
	REG_R9:  9,
	REG_R10: 10,
	REG_R11: 11,
	REG_R12: 12,
	REG_R13: 13,
	REG_R14: 14,
	REG_R15: 15,
	REG_R16: 16,
	REG_R17: 17,
	REG_R18: 18,
	REG_R19: 19,
	REG_R20: 20,
	REG_R21: 21,
	REG_R22: 22,
	REG_R23: 23,
	REG_R24: 24,
	REG_R25: 25,
	REG_R26: 26,
	REG_R27: 27,
	REG_R28: 28,
	REG_R29: 29,
	REG_R30: 30,

	// floating point
	REG_F0:  64,
	REG_F1:  65,
	REG_F2:  66,
	REG_F3:  67,
	REG_F4:  68,
	REG_F5:  69,
	REG_F6:  70,
	REG_F7:  71,
	REG_F8:  72,
	REG_F9:  73,
	REG_F10: 74,
	REG_F11: 75,
	REG_F12: 76,
	REG_F13: 77,
	REG_F14: 78,
	REG_F15: 79,
	REG_F16: 80,
	REG_F17: 81,
	REG_F18: 82,
	REG_F19: 83,
	REG_F20: 84,
	REG_F21: 85,
	REG_F22: 86,
	REG_F23: 87,
	REG_F24: 88,
	REG_F25: 89,
	REG_F26: 90,
	REG_F27: 91,
	REG_F28: 92,
	REG_F29: 93,
	REG_F30: 94,
	REG_F31: 95,

	// SIMD
	REG_V0:  64,
	REG_V1:  65,
	REG_V2:  66,
	REG_V3:  67,
	REG_V4:  68,
	REG_V5:  69,
	REG_V6:  70,
	REG_V7:  71,
	REG_V8:  72,
	REG_V9:  73,
	REG_V10: 74,
	REG_V11: 75,
	REG_V12: 76,
	REG_V13: 77,
	REG_V14: 78,
	REG_V15: 79,
	REG_V16: 80,
	REG_V17: 81,
	REG_V18: 82,
	REG_V19: 83,
	REG_V20: 84,
	REG_V21: 85,
	REG_V22: 86,
	REG_V23: 87,
	REG_V24: 88,
	REG_V25: 89,
	REG_V26: 90,
	REG_V27: 91,
	REG_V28: 92,
	REG_V29: 93,
	REG_V30: 94,
	REG_V31: 95,
}

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

//go:generate go run ../mkcnames.go -i a.out.go -o anames7.go -p arm64
const (
	// optab is sorted based on the order of these constants
	// and the first match is chosen.
	// The more specific class needs to come earlier.
	C_NONE   = iota + 1 // starting from 1, leave unclassified Addr's class as 0
	C_REG               // R0..R30
	C_ZREG              // R0..R30, ZR
	C_RSP               // R0..R30, RSP
	C_FREG              // F0..F31
	C_VREG              // V0..V31
	C_PAIR              // (Rn, Rm)
	C_SHIFT             // Rn<<2
	C_EXTREG            // Rn.UXTB[<<3]
	C_SPR               // REG_NZCV
	C_COND              // condition code, EQ, NE, etc.
	C_SPOP              // special operand, PLDL1KEEP, VMALLE1IS, etc.
	C_ARNG              // Vn.<T>
	C_ELEM              // Vn.<T>[index]
	C_LIST              // [V1, V2, V3]

	C_ZCON     // $0
	C_ABCON0   // could be C_ADDCON0 or C_BITCON
	C_ADDCON0  // 12-bit unsigned, unshifted
	C_ABCON    // could be C_ADDCON or C_BITCON
	C_AMCON    // could be C_ADDCON or C_MOVCON
	C_ADDCON   // 12-bit unsigned, shifted left by 0 or 12
	C_MBCON    // could be C_MOVCON or C_BITCON
	C_MOVCON   // generated by a 16-bit constant, optionally inverted and/or shifted by multiple of 16
	C_BITCON   // bitfield and logical immediate masks
	C_ADDCON2  // 24-bit constant
	C_LCON     // 32-bit constant
	C_MOVCON2  // a constant that can be loaded with one MOVZ/MOVN and one MOVK
	C_MOVCON3  // a constant that can be loaded with one MOVZ/MOVN and two MOVKs
	C_VCON     // 64-bit constant
	C_FCON     // floating-point constant
	C_VCONADDR // 64-bit memory address

	C_AACON  // ADDCON offset in auto constant $a(FP)
	C_AACON2 // 24-bit offset in auto constant $a(FP)
	C_LACON  // 32-bit offset in auto constant $a(FP)
	C_AECON  // ADDCON offset in extern constant $e(SB)

	// TODO(aram): only one branch class should be enough
	C_SBRA // for TYPE_BRANCH
	C_LBRA

	C_ZAUTO       // 0(RSP)
	C_NSAUTO_16   // -256 <= x < 0, 0 mod 16
	C_NSAUTO_8    // -256 <= x < 0, 0 mod 8
	C_NSAUTO_4    // -256 <= x < 0, 0 mod 4
	C_NSAUTO      // -256 <= x < 0
	C_NPAUTO_16   // -512 <= x < 0, 0 mod 16
	C_NPAUTO      // -512 <= x < 0, 0 mod 8
	C_NQAUTO_16   // -1024 <= x < 0, 0 mod 16
	C_NAUTO4K     // -4095 <= x < 0
	C_PSAUTO_16   // 0 to 255, 0 mod 16
	C_PSAUTO_8    // 0 to 255, 0 mod 8
	C_PSAUTO_4    // 0 to 255, 0 mod 4
	C_PSAUTO      // 0 to 255
	C_PPAUTO_16   // 0 to 504, 0 mod 16
	C_PPAUTO      // 0 to 504, 0 mod 8
	C_PQAUTO_16   // 0 to 1008, 0 mod 16
	C_UAUTO4K_16  // 0 to 4095, 0 mod 16
	C_UAUTO4K_8   // 0 to 4095, 0 mod 8
	C_UAUTO4K_4   // 0 to 4095, 0 mod 4
	C_UAUTO4K_2   // 0 to 4095, 0 mod 2
	C_UAUTO4K     // 0 to 4095
	C_UAUTO8K_16  // 0 to 8190, 0 mod 16
	C_UAUTO8K_8   // 0 to 8190, 0 mod 8
	C_UAUTO8K_4   // 0 to 8190, 0 mod 4
	C_UAUTO8K     // 0 to 8190, 0 mod 2  + C_PSAUTO
	C_UAUTO16K_16 // 0 to 16380, 0 mod 16
	C_UAUTO16K_8  // 0 to 16380, 0 mod 8
	C_UAUTO16K    // 0 to 16380, 0 mod 4 + C_PSAUTO
	C_UAUTO32K_16 // 0 to 32760, 0 mod 16 + C_PSAUTO
	C_UAUTO32K    // 0 to 32760, 0 mod 8 + C_PSAUTO
	C_UAUTO64K    // 0 to 65520, 0 mod 16 + C_PSAUTO
	C_LAUTOPOOL   // any other constant up to 64 bits (needs pool literal)
	C_LAUTO       // any other constant up to 64 bits

	C_SEXT1  // 0 to 4095, direct
	C_SEXT2  // 0 to 8190
	C_SEXT4  // 0 to 16380
	C_SEXT8  // 0 to 32760
	C_SEXT16 // 0 to 65520
	C_LEXT

	C_ZOREG     // 0(R)
	C_NSOREG_16 // must mirror C_NSAUTO_16, etc
	C_NSOREG_8
	C_NSOREG_4
	C_NSOREG
	C_NPOREG_16
	C_NPOREG
	C_NQOREG_16
	C_NOREG4K
	C_PSOREG_16
	C_PSOREG_8
	C_PSOREG_4
	C_PSOREG
	C_PPOREG_16
	C_PPOREG
	C_PQOREG_16
	C_UOREG4K_16
	C_UOREG4K_8
	C_UOREG4K_4
	C_UOREG4K_2
	C_UOREG4K
	C_UOREG8K_16
	C_UOREG8K_8
	C_UOREG8K_4
	C_UOREG8K
	C_UOREG16K_16
	C_UOREG16K_8
	C_UOREG16K
	C_UOREG32K_16
	C_UOREG32K
	C_UOREG64K
	C_LOREGPOOL
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
	AAESD
	AAESE
	AAESIMC
	AAESMC
	AAND
	AANDS
	AANDSW
	AANDW
	AASR
	AASRW
	AAT
	ABCC
	ABCS
	ABEQ
	ABFI
	ABFIW
	ABFM
	ABFMW
	ABFXIL
	ABFXILW
	ABGE
	ABGT
	ABHI
	ABHS
	ABIC
	ABICS
	ABICSW
	ABICW
	ABLE
	ABLO
	ABLS
	ABLT
	ABMI
	ABNE
	ABPL
	ABRK
	ABVC
	ABVS
	ACASAD
	ACASALB
	ACASALD
	ACASALH
	ACASALW
	ACASAW
	ACASB
	ACASD
	ACASH
	ACASLD
	ACASLW
	ACASPD
	ACASPW
	ACASW
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
	ADWORD
	AEON
	AEONW
	AEOR
	AEORW
	AERET
	AEXTR
	AEXTRW
	AFABSD
	AFABSS
	AFADDD
	AFADDS
	AFCCMPD
	AFCCMPED
	AFCCMPES
	AFCCMPS
	AFCMPD
	AFCMPED
	AFCMPES
	AFCMPS
	AFCSELD
	AFCSELS
	AFCVTDH
	AFCVTDS
	AFCVTHD
	AFCVTHS
	AFCVTSD
	AFCVTSH
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
	AFLDPD
	AFLDPQ
	AFLDPS
	AFMADDD
	AFMADDS
	AFMAXD
	AFMAXNMD
	AFMAXNMS
	AFMAXS
	AFMIND
	AFMINNMD
	AFMINNMS
	AFMINS
	AFMOVD
	AFMOVQ
	AFMOVS
	AFMSUBD
	AFMSUBS
	AFMULD
	AFMULS
	AFNEGD
	AFNEGS
	AFNMADDD
	AFNMADDS
	AFNMSUBD
	AFNMSUBS
	AFNMULD
	AFNMULS
	AFRINTAD
	AFRINTAS
	AFRINTID
	AFRINTIS
	AFRINTMD
	AFRINTMS
	AFRINTND
	AFRINTNS
	AFRINTPD
	AFRINTPS
	AFRINTXD
	AFRINTXS
	AFRINTZD
	AFRINTZS
	AFSQRTD
	AFSQRTS
	AFSTPD
	AFSTPQ
	AFSTPS
	AFSUBD
	AFSUBS
	AHINT
	AHLT
	AHVC
	AIC
	AISB
	ALDADDAB
	ALDADDAD
	ALDADDAH
	ALDADDALB
	ALDADDALD
	ALDADDALH
	ALDADDALW
	ALDADDAW
	ALDADDB
	ALDADDD
	ALDADDH
	ALDADDLB
	ALDADDLD
	ALDADDLH
	ALDADDLW
	ALDADDW
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
	ALDCLRAB
	ALDCLRAD
	ALDCLRAH
	ALDCLRALB
	ALDCLRALD
	ALDCLRALH
	ALDCLRALW
	ALDCLRAW
	ALDCLRB
	ALDCLRD
	ALDCLRH
	ALDCLRLB
	ALDCLRLD
	ALDCLRLH
	ALDCLRLW
	ALDCLRW
	ALDEORAB
	ALDEORAD
	ALDEORAH
	ALDEORALB
	ALDEORALD
	ALDEORALH
	ALDEORALW
	ALDEORAW
	ALDEORB
	ALDEORD
	ALDEORH
	ALDEORLB
	ALDEORLD
	ALDEORLH
	ALDEORLW
	ALDEORW
	ALDORAB
	ALDORAD
	ALDORAH
	ALDORALB
	ALDORALD
	ALDORALH
	ALDORALW
	ALDORAW
	ALDORB
	ALDORD
	ALDORH
	ALDORLB
	ALDORLD
	ALDORLH
	ALDORLW
	ALDORW
	ALDP
	ALDPSW
	ALDPW
	ALDXP
	ALDXPW
	ALDXR
	ALDXRB
	ALDXRH
	ALDXRW
	ALSL
	ALSLW
	ALSR
	ALSRW
	AMADD
	AMADDW
	AMNEG
	AMNEGW
	AMOVB
	AMOVBU
	AMOVD
	AMOVH
	AMOVHU
	AMOVK
	AMOVKW
	AMOVN
	AMOVNW
	AMOVP
	AMOVPD
	AMOVPQ
	AMOVPS
	AMOVPSW
	AMOVPW
	AMOVW
	AMOVWU
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
	ANOOP
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
	ASCVTFD
	ASCVTFS
	ASCVTFWD
	ASCVTFWS
	ASDIV
	ASDIVW
	ASEV
	ASEVL
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
	ASHA512H
	ASHA512H2
	ASHA512SU0
	ASHA512SU1
	ASMADDL
	ASMC
	ASMNEGL
	ASMSUBL
	ASMULH
	ASMULL
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
	ASTPW
	ASTXP
	ASTXPW
	ASTXR
	ASTXRB
	ASTXRH
	ASTXRW
	ASUB
	ASUBS
	ASUBSW
	ASUBW
	ASVC
	ASWPAB
	ASWPAD
	ASWPAH
	ASWPALB
	ASWPALD
	ASWPALH
	ASWPALW
	ASWPAW
	ASWPB
	ASWPD
	ASWPH
	ASWPLB
	ASWPLD
	ASWPLH
	ASWPLW
	ASWPW
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
	AUCVTFD
	AUCVTFS
	AUCVTFWD
	AUCVTFWS
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
	AUXTBW
	AUXTH
	AUXTHW
	AUXTW
	AVADD
	AVADDP
	AVADDV
	AVAND
	AVBCAX
	AVBIF
	AVBIT
	AVBSL
	AVCMEQ
	AVCMTST
	AVCNT
	AVDUP
	AVEOR
	AVEOR3
	AVEXT
	AVFMLA
	AVFMLS
	AVLD1
	AVLD1R
	AVLD2
	AVLD2R
	AVLD3
	AVLD3R
	AVLD4
	AVLD4R
	AVMOV
	AVMOVD
	AVMOVI
	AVMOVQ
	AVMOVS
	AVORR
	AVPMULL
	AVPMULL2
	AVRAX1
	AVRBIT
	AVREV16
	AVREV32
	AVREV64
	AVSHL
	AVSLI
	AVSRI
	AVST1
	AVST2
	AVST3
	AVST4
	AVSUB
	AVTBL
	AVTBX
	AVTRN1
	AVTRN2
	AVUADDLV
	AVUADDW
	AVUADDW2
	AVUMAX
	AVUMIN
	AVUSHLL
	AVUSHLL2
	AVUSHR
	AVUSRA
	AVUXTL
	AVUXTL2
	AVUZP1
	AVUZP2
	AVXAR
	AVZIP1
	AVZIP2
	AWFE
	AWFI
	AWORD
	AYIELD
	ALAST
	AB  = obj.AJMP
	ABL = obj.ACALL
)

const (
	// shift types
	SHIFT_LL  = 0 << 22
	SHIFT_LR  = 1 << 22
	SHIFT_AR  = 2 << 22
	SHIFT_ROR = 3 << 22
)

// Arrangement for ARM64 SIMD instructions
const (
	// arrangement types
	ARNG_8B = iota
	ARNG_16B
	ARNG_1D
	ARNG_4H
	ARNG_8H
	ARNG_2S
	ARNG_4S
	ARNG_2D
	ARNG_1Q
	ARNG_B
	ARNG_H
	ARNG_S
	ARNG_D
)

//go:generate stringer -type SpecialOperand -trimprefix SPOP_
type SpecialOperand int

const (
	// PRFM
	SPOP_PLDL1KEEP SpecialOperand = iota     // must be the first one
	SPOP_BEGIN     SpecialOperand = iota - 1 // set as the lower bound
	SPOP_PLDL1STRM
	SPOP_PLDL2KEEP
	SPOP_PLDL2STRM
	SPOP_PLDL3KEEP
	SPOP_PLDL3STRM
	SPOP_PLIL1KEEP
	SPOP_PLIL1STRM
	SPOP_PLIL2KEEP
	SPOP_PLIL2STRM
	SPOP_PLIL3KEEP
	SPOP_PLIL3STRM
	SPOP_PSTL1KEEP
	SPOP_PSTL1STRM
	SPOP_PSTL2KEEP
	SPOP_PSTL2STRM
	SPOP_PSTL3KEEP
	SPOP_PSTL3STRM

	// TLBI
	SPOP_VMALLE1IS
	SPOP_VAE1IS
	SPOP_ASIDE1IS
	SPOP_VAAE1IS
	SPOP_VALE1IS
	SPOP_VAALE1IS
	SPOP_VMALLE1
	SPOP_VAE1
	SPOP_ASIDE1
	SPOP_VAAE1
	SPOP_VALE1
	SPOP_VAALE1
	SPOP_IPAS2E1IS
	SPOP_IPAS2LE1IS
	SPOP_ALLE2IS
	SPOP_VAE2IS
	SPOP_ALLE1IS
	SPOP_VALE2IS
	SPOP_VMALLS12E1IS
	SPOP_IPAS2E1
	SPOP_IPAS2LE1
	SPOP_ALLE2
	SPOP_VAE2
	SPOP_ALLE1
	SPOP_VALE2
	SPOP_VMALLS12E1
	SPOP_ALLE3IS
	SPOP_VAE3IS
	SPOP_VALE3IS
	SPOP_ALLE3
	SPOP_VAE3
	SPOP_VALE3
	SPOP_VMALLE1OS
	SPOP_VAE1OS
	SPOP_ASIDE1OS
	SPOP_VAAE1OS
	SPOP_VALE1OS
	SPOP_VAALE1OS
	SPOP_RVAE1IS
	SPOP_RVAAE1IS
	SPOP_RVALE1IS
	SPOP_RVAALE1IS
	SPOP_RVAE1OS
	SPOP_RVAAE1OS
	SPOP_RVALE1OS
	SPOP_RVAALE1OS
	SPOP_RVAE1
	SPOP_RVAAE1
	SPOP_RVALE1
	SPOP_RVAALE1
	SPOP_RIPAS2E1IS
	SPOP_RIPAS2LE1IS
	SPOP_ALLE2OS
	SPOP_VAE2OS
	SPOP_ALLE1OS
	SPOP_VALE2OS
	SPOP_VMALLS12E1OS
	SPOP_RVAE2IS
	SPOP_RVALE2IS
	SPOP_IPAS2E1OS
	SPOP_RIPAS2E1
	SPOP_RIPAS2E1OS
	SPOP_IPAS2LE1OS
	SPOP_RIPAS2LE1
	SPOP_RIPAS2LE1OS
	SPOP_RVAE2OS
	SPOP_RVALE2OS
	SPOP_RVAE2
	SPOP_RVALE2
	SPOP_ALLE3OS
	SPOP_VAE3OS
	SPOP_VALE3OS
	SPOP_RVAE3IS
	SPOP_RVALE3IS
	SPOP_RVAE3OS
	SPOP_RVALE3OS
	SPOP_RVAE3
	SPOP_RVALE3

	// DC
	SPOP_IVAC
	SPOP_ISW
	SPOP_CSW
	SPOP_CISW
	SPOP_ZVA
	SPOP_CVAC
	SPOP_CVAU
	SPOP_CIVAC
	SPOP_IGVAC
	SPOP_IGSW
	SPOP_IGDVAC
	SPOP_IGDSW
	SPOP_CGSW
	SPOP_CGDSW
	SPOP_CIGSW
	SPOP_CIGDSW
	SPOP_GVA
	SPOP_GZVA
	SPOP_CGVAC
	SPOP_CGDVAC
	SPOP_CGVAP
	SPOP_CGDVAP
	SPOP_CGVADP
	SPOP_CGDVADP
	SPOP_CIGVAC
	SPOP_CIGDVAC
	SPOP_CVAP
	SPOP_CVADP

	// PSTATE fields
	SPOP_DAIFSet
	SPOP_DAIFClr

	// Condition code, EQ, NE, etc. Their relative order to EQ is matter.
	SPOP_EQ
	SPOP_NE
	SPOP_HS
	SPOP_LO
	SPOP_MI
	SPOP_PL
	SPOP_VS
	SPOP_VC
	SPOP_HI
	SPOP_LS
	SPOP_GE
	SPOP_LT
	SPOP_GT
	SPOP_LE
	SPOP_AL
	SPOP_NV
	// Condition code end.

	SPOP_END
)
