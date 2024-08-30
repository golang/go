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

package ppc64

import "cmd/internal/obj"

//go:generate go run ../stringer.go -i $GOFILE -o anames.go -p ppc64

/*
 * powerpc 64
 */
const (
	NSNAME = 8
	NSYM   = 50
	NREG   = 32 /* number of general registers */
	NFREG  = 32 /* number of floating point registers */
)

const (
	/* RBasePPC64 = 4096 */
	/* R0=4096 ... R31=4127 */
	REG_R0 = obj.RBasePPC64 + iota
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

	// CR bits. Use Book 1, chapter 2 naming for bits. Keep aligned to 32
	REG_CR0LT
	REG_CR0GT
	REG_CR0EQ
	REG_CR0SO
	REG_CR1LT
	REG_CR1GT
	REG_CR1EQ
	REG_CR1SO
	REG_CR2LT
	REG_CR2GT
	REG_CR2EQ
	REG_CR2SO
	REG_CR3LT
	REG_CR3GT
	REG_CR3EQ
	REG_CR3SO
	REG_CR4LT
	REG_CR4GT
	REG_CR4EQ
	REG_CR4SO
	REG_CR5LT
	REG_CR5GT
	REG_CR5EQ
	REG_CR5SO
	REG_CR6LT
	REG_CR6GT
	REG_CR6EQ
	REG_CR6SO
	REG_CR7LT
	REG_CR7GT
	REG_CR7EQ
	REG_CR7SO

	/* Align FPR and VSR vectors such that when masked with 0x3F they produce
	   an equivalent VSX register. */
	/* F0=4160 ... F31=4191 */
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

	/* V0=4192 ... V31=4223 */
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

	/* VS0=4224 ... VS63=4287 */
	REG_VS0
	REG_VS1
	REG_VS2
	REG_VS3
	REG_VS4
	REG_VS5
	REG_VS6
	REG_VS7
	REG_VS8
	REG_VS9
	REG_VS10
	REG_VS11
	REG_VS12
	REG_VS13
	REG_VS14
	REG_VS15
	REG_VS16
	REG_VS17
	REG_VS18
	REG_VS19
	REG_VS20
	REG_VS21
	REG_VS22
	REG_VS23
	REG_VS24
	REG_VS25
	REG_VS26
	REG_VS27
	REG_VS28
	REG_VS29
	REG_VS30
	REG_VS31
	REG_VS32
	REG_VS33
	REG_VS34
	REG_VS35
	REG_VS36
	REG_VS37
	REG_VS38
	REG_VS39
	REG_VS40
	REG_VS41
	REG_VS42
	REG_VS43
	REG_VS44
	REG_VS45
	REG_VS46
	REG_VS47
	REG_VS48
	REG_VS49
	REG_VS50
	REG_VS51
	REG_VS52
	REG_VS53
	REG_VS54
	REG_VS55
	REG_VS56
	REG_VS57
	REG_VS58
	REG_VS59
	REG_VS60
	REG_VS61
	REG_VS62
	REG_VS63

	REG_CR0
	REG_CR1
	REG_CR2
	REG_CR3
	REG_CR4
	REG_CR5
	REG_CR6
	REG_CR7

	// MMA accumulator registers, these shadow VSR 0-31
	// e.g MMAx shadows VSRx*4-VSRx*4+3 or
	//     MMA0 shadows VSR0-VSR3
	REG_A0
	REG_A1
	REG_A2
	REG_A3
	REG_A4
	REG_A5
	REG_A6
	REG_A7

	REG_MSR
	REG_FPSCR
	REG_CR

	REG_SPECIAL = REG_CR0

	REG_CRBIT0 = REG_CR0LT // An alias for a Condition Register bit 0

	REG_SPR0 = obj.RBasePPC64 + 1024 // first of 1024 registers

	REG_XER = REG_SPR0 + 1
	REG_LR  = REG_SPR0 + 8
	REG_CTR = REG_SPR0 + 9

	REGZERO = REG_R0 /* set to zero */
	REGSP   = REG_R1
	REGSB   = REG_R2
	REGRET  = REG_R3
	REGARG  = -1      /* -1 disables passing the first argument in register */
	REGRT1  = REG_R20 /* reserved for runtime, duffzero and duffcopy */
	REGRT2  = REG_R21 /* reserved for runtime, duffcopy */
	REGMIN  = REG_R7  /* register variables allocated from here to REGMAX */
	REGCTXT = REG_R11 /* context for closures */
	REGTLS  = REG_R13 /* C ABI TLS base pointer */
	REGMAX  = REG_R27
	REGEXT  = REG_R30 /* external registers allocated from here down */
	REGG    = REG_R30 /* G */
	REGTMP  = REG_R31 /* used by the linker */
	FREGRET = REG_F0
	FREGMIN = REG_F17 /* first register variable */
	FREGMAX = REG_F26 /* last register variable for 9g only */
	FREGEXT = REG_F26 /* first external register */
)

// OpenPOWER ABI for Linux Supplement Power Architecture 64-Bit ELF V2 ABI
// https://openpowerfoundation.org/?resource_lib=64-bit-elf-v2-abi-specification-power-architecture
var PPC64DWARFRegisters = map[int16]int16{}

func init() {
	// f assigns dwarfregister[from:to] = (base):(to-from+base)
	f := func(from, to, base int16) {
		for r := int16(from); r <= to; r++ {
			PPC64DWARFRegisters[r] = r - from + base
		}
	}
	f(REG_R0, REG_R31, 0)
	f(REG_F0, REG_F31, 32)
	f(REG_V0, REG_V31, 77)
	f(REG_CR0, REG_CR7, 68)

	f(REG_VS0, REG_VS31, 32)  // overlaps F0-F31
	f(REG_VS32, REG_VS63, 77) // overlaps V0-V31
	PPC64DWARFRegisters[REG_LR] = 65
	PPC64DWARFRegisters[REG_CTR] = 66
	PPC64DWARFRegisters[REG_XER] = 76
}

/*
 * GENERAL:
 *
 * compiler allocates R3 up as temps
 * compiler allocates register variables R7-R27
 * compiler allocates external registers R30 down
 *
 * compiler allocates register variables F17-F26
 * compiler allocates external registers F26 down
 */
const (
	BIG = 32768 - 8
)

const (
	/* mark flags */
	LABEL    = 1 << 0
	LEAF     = 1 << 1
	FLOAT    = 1 << 2
	BRANCH   = 1 << 3
	LOAD     = 1 << 4
	FCMP     = 1 << 5
	SYNC     = 1 << 6
	LIST     = 1 << 7
	FOLL     = 1 << 8
	NOSCHED  = 1 << 9
	PFX_X64B = 1 << 10 // A prefixed instruction crossing a 64B boundary
)

// Values for use in branch instruction BC
// BC B0,BI,label
// BO is type of branch + likely bits described below
// BI is CR value + branch type
// ex: BEQ CR2,label is BC 12,10,label
//   12 = BO_BCR
//   10 = BI_CR2 + BI_EQ

const (
	BI_CR0 = 0
	BI_CR1 = 4
	BI_CR2 = 8
	BI_CR3 = 12
	BI_CR4 = 16
	BI_CR5 = 20
	BI_CR6 = 24
	BI_CR7 = 28
	BI_LT  = 0
	BI_GT  = 1
	BI_EQ  = 2
	BI_FU  = 3
)

// Common values for the BO field.

const (
	BO_ALWAYS  = 20 // branch unconditionally
	BO_BCTR    = 16 // decrement ctr, branch on ctr != 0
	BO_NOTBCTR = 18 // decrement ctr, branch on ctr == 0
	BO_BCR     = 12 // branch on cr value
	BO_BCRBCTR = 8  // decrement ctr, branch on ctr != 0 and cr value
	BO_NOTBCR  = 4  // branch on not cr value
)

// Bit settings from the CR

const (
	C_COND_LT = iota // 0 result is negative
	C_COND_GT        // 1 result is positive
	C_COND_EQ        // 2 result is zero
	C_COND_SO        // 3 summary overflow or FP compare w/ NaN
)

const (
	C_NONE     = iota
	C_REGP     /* An even numbered gpr which can be used a gpr pair argument */
	C_REG      /* Any gpr register */
	C_FREGP    /* An even numbered fpr which can be used a fpr pair argument */
	C_FREG     /* Any fpr register */
	C_VREG     /* Any vector register */
	C_VSREGP   /* An even numbered vsx register which can be used as a vsx register pair argument */
	C_VSREG    /* Any vector-scalar register */
	C_CREG     /* The condition registor (CR) */
	C_CRBIT    /* A single bit of the CR register (0-31) */
	C_SPR      /* special processor register */
	C_AREG     /* MMA accumulator register */
	C_ZCON     /* The constant zero */
	C_U1CON    /* 1 bit unsigned constant */
	C_U2CON    /* 2 bit unsigned constant */
	C_U3CON    /* 3 bit unsigned constant */
	C_U4CON    /* 4 bit unsigned constant */
	C_U5CON    /* 5 bit unsigned constant */
	C_U8CON    /* 8 bit unsigned constant */
	C_U15CON   /* 15 bit unsigned constant */
	C_S16CON   /* 16 bit signed constant */
	C_U16CON   /* 16 bit unsigned constant */
	C_16CON    /* Any constant which fits into 16 bits. Can be signed or unsigned */
	C_U31CON   /* 31 bit unsigned constant */
	C_S32CON   /* 32 bit signed constant */
	C_U32CON   /* 32 bit unsigned constant */
	C_32CON    /* Any constant which fits into 32 bits. Can be signed or unsigned */
	C_S34CON   /* 34 bit signed constant */
	C_64CON    /* Any constant which fits into 64 bits. Can be signed or unsigned */
	C_SACON    /* $n(REG) where n <= int16 */
	C_LACON    /* $n(REG) where n <= int32 */
	C_DACON    /* $n(REG) where n <= int64 */
	C_BRA      /* A short offset argument to a branching instruction */
	C_BRAPIC   /* Like C_BRA, but requires an extra NOP for potential TOC restore by the linker. */
	C_ZOREG    /* An $0+reg memory op */
	C_SOREG    /* An $n+reg memory arg where n is a 16 bit signed offset */
	C_LOREG    /* An $n+reg memory arg where n is a 32 bit signed offset */
	C_XOREG    /* An reg+reg memory arg */
	C_FPSCR    /* The fpscr register */
	C_LR       /* The link register */
	C_CTR      /* The count register */
	C_ANY      /* Any argument */
	C_GOK      /* A non-matched argument */
	C_ADDR     /* A symbolic memory location */
	C_TLS_LE   /* A thread local, local-exec, type memory arg */
	C_TLS_IE   /* A thread local, initial-exec, type memory arg */
	C_TEXTSIZE /* An argument with Type obj.TYPE_TEXTSIZE */

	C_NCLASS /* must be the last */
)

const (
	AADD = obj.ABasePPC64 + obj.A_ARCHSPECIFIC + iota
	AADDCC
	AADDIS
	AADDV
	AADDVCC
	AADDC
	AADDCCC
	AADDCV
	AADDCVCC
	AADDME
	AADDMECC
	AADDMEVCC
	AADDMEV
	AADDE
	AADDECC
	AADDEVCC
	AADDEV
	AADDZE
	AADDZECC
	AADDZEVCC
	AADDZEV
	AADDEX
	AAND
	AANDCC
	AANDN
	AANDNCC
	AANDISCC
	ABC
	ABCL
	ABEQ
	ABGE // not LT = G/E/U
	ABGT
	ABLE // not GT = L/E/U
	ABLT
	ABNE  // not EQ = L/G/U
	ABVC  // Branch if float not unordered (also branch on not summary overflow)
	ABVS  // Branch if float unordered (also branch on summary overflow)
	ABDNZ // Decrement CTR, and branch if CTR != 0
	ABDZ  // Decrement CTR, and branch if CTR == 0
	ACMP
	ACMPU
	ACMPEQB
	ACNTLZW
	ACNTLZWCC
	ACRAND
	ACRANDN
	ACREQV
	ACRNAND
	ACRNOR
	ACROR
	ACRORN
	ACRXOR
	ADIVW
	ADIVWCC
	ADIVWVCC
	ADIVWV
	ADIVWU
	ADIVWUCC
	ADIVWUVCC
	ADIVWUV
	AMODUD
	AMODUW
	AMODSD
	AMODSW
	AEQV
	AEQVCC
	AEXTSB
	AEXTSBCC
	AEXTSH
	AEXTSHCC
	AFABS
	AFABSCC
	AFADD
	AFADDCC
	AFADDS
	AFADDSCC
	AFCMPO
	AFCMPU
	AFCTIW
	AFCTIWCC
	AFCTIWZ
	AFCTIWZCC
	AFDIV
	AFDIVCC
	AFDIVS
	AFDIVSCC
	AFMADD
	AFMADDCC
	AFMADDS
	AFMADDSCC
	AFMOVD
	AFMOVDCC
	AFMOVDU
	AFMOVS
	AFMOVSU
	AFMOVSX
	AFMOVSZ
	AFMSUB
	AFMSUBCC
	AFMSUBS
	AFMSUBSCC
	AFMUL
	AFMULCC
	AFMULS
	AFMULSCC
	AFNABS
	AFNABSCC
	AFNEG
	AFNEGCC
	AFNMADD
	AFNMADDCC
	AFNMADDS
	AFNMADDSCC
	AFNMSUB
	AFNMSUBCC
	AFNMSUBS
	AFNMSUBSCC
	AFRSP
	AFRSPCC
	AFSUB
	AFSUBCC
	AFSUBS
	AFSUBSCC
	AISEL
	AMOVMW
	ALBAR
	ALHAR
	ALSW
	ALWAR
	ALWSYNC
	AMOVDBR
	AMOVWBR
	AMOVB
	AMOVBU
	AMOVBZ
	AMOVBZU
	AMOVH
	AMOVHBR
	AMOVHU
	AMOVHZ
	AMOVHZU
	AMOVW
	AMOVWU
	AMOVFL
	AMOVCRFS
	AMTFSB0
	AMTFSB0CC
	AMTFSB1
	AMTFSB1CC
	AMULHW
	AMULHWCC
	AMULHWU
	AMULHWUCC
	AMULLW
	AMULLWCC
	AMULLWVCC
	AMULLWV
	ANAND
	ANANDCC
	ANEG
	ANEGCC
	ANEGVCC
	ANEGV
	ANOR
	ANORCC
	AOR
	AORCC
	AORN
	AORNCC
	AORIS
	AREM
	AREMU
	ARFI
	ARLWMI
	ARLWMICC
	ARLWNM
	ARLWNMCC
	ACLRLSLWI
	ASLW
	ASLWCC
	ASRW
	ASRAW
	ASRAWCC
	ASRWCC
	ASTBCCC
	ASTHCCC
	ASTSW
	ASTWCCC
	ASUB
	ASUBCC
	ASUBVCC
	ASUBC
	ASUBCCC
	ASUBCV
	ASUBCVCC
	ASUBME
	ASUBMECC
	ASUBMEVCC
	ASUBMEV
	ASUBV
	ASUBE
	ASUBECC
	ASUBEV
	ASUBEVCC
	ASUBZE
	ASUBZECC
	ASUBZEVCC
	ASUBZEV
	ASYNC
	AXOR
	AXORCC
	AXORIS

	ADCBF
	ADCBI
	ADCBST
	ADCBT
	ADCBTST
	ADCBZ
	AEIEIO
	AICBI
	AISYNC
	APTESYNC
	ATLBIE
	ATLBIEL
	ATLBSYNC
	ATW

	ASYSCALL
	AWORD

	ARFCI

	AFCPSGN
	AFCPSGNCC
	/* optional on 32-bit */
	AFRES
	AFRESCC
	AFRIM
	AFRIMCC
	AFRIP
	AFRIPCC
	AFRIZ
	AFRIZCC
	AFRIN
	AFRINCC
	AFRSQRTE
	AFRSQRTECC
	AFSEL
	AFSELCC
	AFSQRT
	AFSQRTCC
	AFSQRTS
	AFSQRTSCC

	/* 64-bit */

	ACNTLZD
	ACNTLZDCC
	ACMPW /* CMP with L=0 */
	ACMPWU
	ACMPB
	AFTDIV
	AFTSQRT
	ADIVD
	ADIVDCC
	ADIVDE
	ADIVDECC
	ADIVDEU
	ADIVDEUCC
	ADIVDVCC
	ADIVDV
	ADIVDU
	ADIVDUCC
	ADIVDUVCC
	ADIVDUV
	AEXTSW
	AEXTSWCC
	/* AFCFIW; AFCFIWCC */
	AFCFID
	AFCFIDCC
	AFCFIDU
	AFCFIDUCC
	AFCFIDS
	AFCFIDSCC
	AFCTID
	AFCTIDCC
	AFCTIDZ
	AFCTIDZCC
	ALDAR
	AMOVD
	AMOVDU
	AMOVWZ
	AMOVWZU
	AMULHD
	AMULHDCC
	AMULHDU
	AMULHDUCC
	AMULLD
	AMULLDCC
	AMULLDVCC
	AMULLDV
	ARFID
	ARLDMI
	ARLDMICC
	ARLDIMI
	ARLDIMICC
	ARLDC
	ARLDCCC
	ARLDCR
	ARLDCRCC
	ARLDICR
	ARLDICRCC
	ARLDCL
	ARLDCLCC
	ARLDICL
	ARLDICLCC
	ARLDIC
	ARLDICCC
	ACLRLSLDI
	AROTL
	AROTLW
	ASLBIA
	ASLBIE
	ASLBMFEE
	ASLBMFEV
	ASLBMTE
	ASLD
	ASLDCC
	ASRD
	ASRAD
	ASRADCC
	ASRDCC
	AEXTSWSLI
	AEXTSWSLICC
	ASTDCCC
	ATD
	ASETB

	/* 64-bit pseudo operation */
	ADWORD
	AREMD
	AREMDU

	/* more 64-bit operations */
	AHRFID
	APOPCNTD
	APOPCNTW
	APOPCNTB
	ACNTTZW
	ACNTTZWCC
	ACNTTZD
	ACNTTZDCC
	ACOPY
	APASTECC
	ADARN
	AMADDHD
	AMADDHDU
	AMADDLD

	/* Vector */
	ALVEBX
	ALVEHX
	ALVEWX
	ALVX
	ALVXL
	ALVSL
	ALVSR
	ASTVEBX
	ASTVEHX
	ASTVEWX
	ASTVX
	ASTVXL
	AVAND
	AVANDC
	AVNAND
	AVOR
	AVORC
	AVNOR
	AVXOR
	AVEQV
	AVADDUM
	AVADDUBM
	AVADDUHM
	AVADDUWM
	AVADDUDM
	AVADDUQM
	AVADDCU
	AVADDCUQ
	AVADDCUW
	AVADDUS
	AVADDUBS
	AVADDUHS
	AVADDUWS
	AVADDSS
	AVADDSBS
	AVADDSHS
	AVADDSWS
	AVADDE
	AVADDEUQM
	AVADDECUQ
	AVSUBUM
	AVSUBUBM
	AVSUBUHM
	AVSUBUWM
	AVSUBUDM
	AVSUBUQM
	AVSUBCU
	AVSUBCUQ
	AVSUBCUW
	AVSUBUS
	AVSUBUBS
	AVSUBUHS
	AVSUBUWS
	AVSUBSS
	AVSUBSBS
	AVSUBSHS
	AVSUBSWS
	AVSUBE
	AVSUBEUQM
	AVSUBECUQ
	AVMULESB
	AVMULOSB
	AVMULEUB
	AVMULOUB
	AVMULESH
	AVMULOSH
	AVMULEUH
	AVMULOUH
	AVMULESW
	AVMULOSW
	AVMULEUW
	AVMULOUW
	AVMULUWM
	AVPMSUM
	AVPMSUMB
	AVPMSUMH
	AVPMSUMW
	AVPMSUMD
	AVMSUMUDM
	AVR
	AVRLB
	AVRLH
	AVRLW
	AVRLD
	AVS
	AVSLB
	AVSLH
	AVSLW
	AVSL
	AVSLO
	AVSRB
	AVSRH
	AVSRW
	AVSR
	AVSRO
	AVSLD
	AVSRD
	AVSA
	AVSRAB
	AVSRAH
	AVSRAW
	AVSRAD
	AVSOI
	AVSLDOI
	AVCLZ
	AVCLZB
	AVCLZH
	AVCLZW
	AVCLZD
	AVPOPCNT
	AVPOPCNTB
	AVPOPCNTH
	AVPOPCNTW
	AVPOPCNTD
	AVCMPEQ
	AVCMPEQUB
	AVCMPEQUBCC
	AVCMPEQUH
	AVCMPEQUHCC
	AVCMPEQUW
	AVCMPEQUWCC
	AVCMPEQUD
	AVCMPEQUDCC
	AVCMPGT
	AVCMPGTUB
	AVCMPGTUBCC
	AVCMPGTUH
	AVCMPGTUHCC
	AVCMPGTUW
	AVCMPGTUWCC
	AVCMPGTUD
	AVCMPGTUDCC
	AVCMPGTSB
	AVCMPGTSBCC
	AVCMPGTSH
	AVCMPGTSHCC
	AVCMPGTSW
	AVCMPGTSWCC
	AVCMPGTSD
	AVCMPGTSDCC
	AVCMPNEZB
	AVCMPNEZBCC
	AVCMPNEB
	AVCMPNEBCC
	AVCMPNEH
	AVCMPNEHCC
	AVCMPNEW
	AVCMPNEWCC
	AVPERM
	AVPERMXOR
	AVPERMR
	AVBPERMQ
	AVBPERMD
	AVSEL
	AVSPLTB
	AVSPLTH
	AVSPLTW
	AVSPLTISB
	AVSPLTISH
	AVSPLTISW
	AVCIPH
	AVCIPHER
	AVCIPHERLAST
	AVNCIPH
	AVNCIPHER
	AVNCIPHERLAST
	AVSBOX
	AVSHASIGMA
	AVSHASIGMAW
	AVSHASIGMAD
	AVMRGEW
	AVMRGOW
	AVCLZLSBB
	AVCTZLSBB

	/* VSX */
	ALXV
	ALXVL
	ALXVLL
	ALXVD2X
	ALXVW4X
	ALXVH8X
	ALXVB16X
	ALXVX
	ALXVDSX
	ASTXV
	ASTXVL
	ASTXVLL
	ASTXVD2X
	ASTXVW4X
	ASTXVH8X
	ASTXVB16X
	ASTXVX
	ALXSDX
	ASTXSDX
	ALXSIWAX
	ALXSIWZX
	ASTXSIWX
	AMFVSRD
	AMFFPRD
	AMFVRD
	AMFVSRWZ
	AMFVSRLD
	AMTVSRD
	AMTFPRD
	AMTVRD
	AMTVSRWA
	AMTVSRWZ
	AMTVSRDD
	AMTVSRWS
	AXXLAND
	AXXLANDC
	AXXLEQV
	AXXLNAND
	AXXLOR
	AXXLORC
	AXXLNOR
	AXXLORQ
	AXXLXOR
	AXXSEL
	AXXMRGHW
	AXXMRGLW
	AXXSPLTW
	AXXSPLTIB
	AXXPERM
	AXXPERMDI
	AXXSLDWI
	AXXBRQ
	AXXBRD
	AXXBRW
	AXXBRH
	AXSCVDPSP
	AXSCVSPDP
	AXSCVDPSPN
	AXSCVSPDPN
	AXVCVDPSP
	AXVCVSPDP
	AXSCVDPSXDS
	AXSCVDPSXWS
	AXSCVDPUXDS
	AXSCVDPUXWS
	AXSCVSXDDP
	AXSCVUXDDP
	AXSCVSXDSP
	AXSCVUXDSP
	AXVCVDPSXDS
	AXVCVDPSXWS
	AXVCVDPUXDS
	AXVCVDPUXWS
	AXVCVSPSXDS
	AXVCVSPSXWS
	AXVCVSPUXDS
	AXVCVSPUXWS
	AXVCVSXDDP
	AXVCVSXWDP
	AXVCVUXDDP
	AXVCVUXWDP
	AXVCVSXDSP
	AXVCVSXWSP
	AXVCVUXDSP
	AXVCVUXWSP
	AXSMAXJDP
	AXSMINJDP
	ALASTAOUT // The last instruction in this list. Also the first opcode generated by ppc64map.

	// aliases
	ABR   = obj.AJMP
	ABL   = obj.ACALL
	ALAST = ALASTGEN // The final enumerated instruction value + 1. This is used to size the oprange table.
)
