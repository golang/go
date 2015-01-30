// cmd/9c/9.out.h from Vita Nuova.
//
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

/*
 * powerpc 64
 */
enum
{
	NSNAME = 8,
	NSYM = 50,
	NREG = 32,	/* number of general registers */
	NFREG = 32,	/* number of floating point registers */
};

#include "../ld/textflag.h"

// avoid conflict with ucontext.h. sigh.
#undef REG_R0
#undef REG_R1
#undef REG_R2
#undef REG_R3
#undef REG_R4
#undef REG_R5
#undef REG_R6
#undef REG_R7
#undef REG_R8
#undef REG_R9
#undef REG_R10
#undef REG_R11
#undef REG_R12
#undef REG_R13
#undef REG_R14
#undef REG_R15
#undef REG_R16
#undef REG_R17
#undef REG_R18
#undef REG_R19
#undef REG_R20
#undef REG_R21
#undef REG_R22
#undef REG_R23
#undef REG_R24
#undef REG_R25
#undef REG_R26
#undef REG_R27
#undef REG_R28
#undef REG_R29
#undef REG_R30
#undef REG_R31
#define REG_R0 GO_REG_R0
#define REG_R1 GO_REG_R1
#define REG_R2 GO_REG_R2
#define REG_R3 GO_REG_R3
#define REG_R4 GO_REG_R4
#define REG_R5 GO_REG_R5
#define REG_R6 GO_REG_R6
#define REG_R7 GO_REG_R7
#define REG_R8 GO_REG_R8
#define REG_R9 GO_REG_R9
#define REG_R10 GO_REG_R10
#define REG_R11 GO_REG_R11
#define REG_R12 GO_REG_R12
#define REG_R13 GO_REG_R13
#define REG_R14 GO_REG_R14
#define REG_R15 GO_REG_R15
#define REG_R16 GO_REG_R16
#define REG_R17 GO_REG_R17
#define REG_R18 GO_REG_R18
#define REG_R19 GO_REG_R19
#define REG_R20 GO_REG_R20
#define REG_R21 GO_REG_R21
#define REG_R22 GO_REG_R22
#define REG_R23 GO_REG_R23
#define REG_R24 GO_REG_R24
#define REG_R25 GO_REG_R25
#define REG_R26 GO_REG_R26
#define REG_R27 GO_REG_R27
#define REG_R28 GO_REG_R28
#define REG_R29 GO_REG_R29
#define REG_R30 GO_REG_R30
#define REG_R31 GO_REG_R31

enum
{
	REG_R0 = 32,
	REG_R1,
	REG_R2,
	REG_R3,
	REG_R4,
	REG_R5,
	REG_R6,
	REG_R7,
	REG_R8,
	REG_R9,
	REG_R10,
	REG_R11,
	REG_R12,
	REG_R13,
	REG_R14,
	REG_R15,
	REG_R16,
	REG_R17,
	REG_R18,
	REG_R19,
	REG_R20,
	REG_R21,
	REG_R22,
	REG_R23,
	REG_R24,
	REG_R25,
	REG_R26,
	REG_R27,
	REG_R28,
	REG_R29,
	REG_R30,
	REG_R31,

	REG_F0 = 64,
	REG_F1,
	REG_F2,
	REG_F3,
	REG_F4,
	REG_F5,
	REG_F6,
	REG_F7,
	REG_F8,
	REG_F9,
	REG_F10,
	REG_F11,
	REG_F12,
	REG_F13,
	REG_F14,
	REG_F15,
	REG_F16,
	REG_F17,
	REG_F18,
	REG_F19,
	REG_F20,
	REG_F21,
	REG_F22,
	REG_F23,
	REG_F24,
	REG_F25,
	REG_F26,
	REG_F27,
	REG_F28,
	REG_F29,
	REG_F30,
	REG_F31,
	
	REG_SPECIAL = 96,

	REG_C0 = 96,
	REG_C1,
	REG_C2,
	REG_C3,
	REG_C4,
	REG_C5,
	REG_C6,
	REG_C7,
	
	REG_MSR = 104,
	REG_FPSCR,
	REG_CR,

	REG_SPR0 = 1024, // first of 1024 registers
	REG_DCR0 = 2048, // first of 1024 registers
	
	REG_XER = REG_SPR0 + 1,
	REG_LR = REG_SPR0 + 8,
	REG_CTR = REG_SPR0 + 9,

	REGZERO		= REG_R0,	/* set to zero */
	REGSP		= REG_R1,
	REGSB		= REG_R2,
	REGRET		= REG_R3,
	REGARG		= -1,	/* -1 disables passing the first argument in register */
	REGRT1		= REG_R3,	/* reserved for runtime, duffzero and duffcopy */
	REGRT2		= REG_R4,	/* reserved for runtime, duffcopy */
	REGMIN		= REG_R7,	/* register variables allocated from here to REGMAX */
	REGCTXT		= REG_R11,	/* context for closures */
	REGTLS		= REG_R13,	/* C ABI TLS base pointer */
	REGMAX		= REG_R27,
	REGEXT		= REG_R30,	/* external registers allocated from here down */
	REGG		= REG_R30,	/* G */
	REGTMP		= REG_R31,	/* used by the linker */

	FREGRET		= REG_F0,
	FREGMIN		= REG_F17,	/* first register variable */
	FREGMAX		= REG_F26,	/* last register variable for 9g only */
	FREGEXT		= REG_F26,	/* first external register */
	FREGCVI		= REG_F27,	/* floating conversion constant */
	FREGZERO	= REG_F28,	/* both float and double */
	FREGHALF	= REG_F29,	/* double */
	FREGONE		= REG_F30,	/* double */
	FREGTWO		= REG_F31	/* double */
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
};

enum {
	BIG = 32768-8,
};

enum {
/* mark flags */
	LABEL		= 1<<0,
	LEAF		= 1<<1,
	FLOAT		= 1<<2,
	BRANCH		= 1<<3,
	LOAD		= 1<<4,
	FCMP		= 1<<5,
	SYNC		= 1<<6,
	LIST		= 1<<7,
	FOLL		= 1<<8,
	NOSCHED		= 1<<9,
};

enum
{
	C_NONE,
	C_REG,
	C_FREG,
	C_CREG,
	C_SPR,		/* special processor register */
	C_ZCON,
	C_SCON,		/* 16 bit signed */
	C_UCON,		/* 32 bit signed, low 16 bits 0 */
	C_ADDCON,	/* -0x8000 <= v < 0 */
	C_ANDCON,	/* 0 < v <= 0xFFFF */
	C_LCON,		/* other 32 */
	C_DCON,		/* other 64 (could subdivide further) */
	C_SACON,	/* $n(REG) where n <= int16 */
	C_SECON,
	C_LACON,	/* $n(REG) where int16 < n <= int32 */
	C_LECON,
	C_DACON,	/* $n(REG) where int32 < n */
	C_SBRA,
	C_LBRA,
	C_SAUTO,
	C_LAUTO,
	C_SEXT,
	C_LEXT,
	C_ZOREG,
	C_SOREG,
	C_LOREG,
	C_FPSCR,
	C_MSR,
	C_XER,
	C_LR,
	C_CTR,
	C_ANY,
	C_GOK,
	C_ADDR,
	C_TEXTSIZE,

	C_NCLASS,	/* must be the last */
};

enum
{
	AADD = A_ARCHSPECIFIC,
	AADDCC,
	AADDV,
	AADDVCC,
	AADDC,
	AADDCCC,
	AADDCV,
	AADDCVCC,
	AADDME,
	AADDMECC,
	AADDMEVCC,
	AADDMEV,
	AADDE,
	AADDECC,
	AADDEVCC,
	AADDEV,
	AADDZE,
	AADDZECC,
	AADDZEVCC,
	AADDZEV,
	AAND,
	AANDCC,
	AANDN,
	AANDNCC,
	ABC,
	ABCL,
	ABEQ,
	ABGE,
	ABGT,
	ABLE,
	ABLT,
	ABNE,
	ABVC,
	ABVS,
	ACMP,
	ACMPU,
	ACNTLZW,
	ACNTLZWCC,
	ACRAND,
	ACRANDN,
	ACREQV,
	ACRNAND,
	ACRNOR,
	ACROR,
	ACRORN,
	ACRXOR,
	ADIVW,
	ADIVWCC,
	ADIVWVCC,
	ADIVWV,
	ADIVWU,
	ADIVWUCC,
	ADIVWUVCC,
	ADIVWUV,
	AEQV,
	AEQVCC,
	AEXTSB,
	AEXTSBCC,
	AEXTSH,
	AEXTSHCC,
	AFABS,
	AFABSCC,
	AFADD,
	AFADDCC,
	AFADDS,
	AFADDSCC,
	AFCMPO,
	AFCMPU,
	AFCTIW,
	AFCTIWCC,
	AFCTIWZ,
	AFCTIWZCC,
	AFDIV,
	AFDIVCC,
	AFDIVS,
	AFDIVSCC,
	AFMADD,
	AFMADDCC,
	AFMADDS,
	AFMADDSCC,
	AFMOVD,
	AFMOVDCC,
	AFMOVDU,
	AFMOVS,
	AFMOVSU,
	AFMSUB,
	AFMSUBCC,
	AFMSUBS,
	AFMSUBSCC,
	AFMUL,
	AFMULCC,
	AFMULS,
	AFMULSCC,
	AFNABS,
	AFNABSCC,
	AFNEG,
	AFNEGCC,
	AFNMADD,
	AFNMADDCC,
	AFNMADDS,
	AFNMADDSCC,
	AFNMSUB,
	AFNMSUBCC,
	AFNMSUBS,
	AFNMSUBSCC,
	AFRSP,
	AFRSPCC,
	AFSUB,
	AFSUBCC,
	AFSUBS,
	AFSUBSCC,
	AMOVMW,
	ALSW,
	ALWAR,
	AMOVWBR,
	AMOVB,
	AMOVBU,
	AMOVBZ,
	AMOVBZU,
	AMOVH,
	AMOVHBR,
	AMOVHU,
	AMOVHZ,
	AMOVHZU,
	AMOVW,
	AMOVWU,
	AMOVFL,
	AMOVCRFS,
	AMTFSB0,
	AMTFSB0CC,
	AMTFSB1,
	AMTFSB1CC,
	AMULHW,
	AMULHWCC,
	AMULHWU,
	AMULHWUCC,
	AMULLW,
	AMULLWCC,
	AMULLWVCC,
	AMULLWV,
	ANAND,
	ANANDCC,
	ANEG,
	ANEGCC,
	ANEGVCC,
	ANEGV,
	ANOR,
	ANORCC,
	AOR,
	AORCC,
	AORN,
	AORNCC,
	AREM,
	AREMCC,
	AREMV,
	AREMVCC,
	AREMU,
	AREMUCC,
	AREMUV,
	AREMUVCC,
	ARFI,
	ARLWMI,
	ARLWMICC,
	ARLWNM,
	ARLWNMCC,
	ASLW,
	ASLWCC,
	ASRW,
	ASRAW,
	ASRAWCC,
	ASRWCC,
	ASTSW,
	ASTWCCC,
	ASUB,
	ASUBCC,
	ASUBVCC,
	ASUBC,
	ASUBCCC,
	ASUBCV,
	ASUBCVCC,
	ASUBME,
	ASUBMECC,
	ASUBMEVCC,
	ASUBMEV,
	ASUBV,
	ASUBE,
	ASUBECC,
	ASUBEV,
	ASUBEVCC,
	ASUBZE,
	ASUBZECC,
	ASUBZEVCC,
	ASUBZEV,
	ASYNC,
	AXOR,
	AXORCC,

	ADCBF,
	ADCBI,
	ADCBST,
	ADCBT,
	ADCBTST,
	ADCBZ,
	AECIWX,
	AECOWX,
	AEIEIO,
	AICBI,
	AISYNC,
	APTESYNC,
	ATLBIE,
	ATLBIEL,
	ATLBSYNC,
	ATW,

	ASYSCALL,
	AWORD,

	ARFCI,

	/* optional on 32-bit */
	AFRES,
	AFRESCC,
	AFRSQRTE,
	AFRSQRTECC,
	AFSEL,
	AFSELCC,
	AFSQRT,
	AFSQRTCC,
	AFSQRTS,
	AFSQRTSCC,

	/* 64-bit */
	
	ACNTLZD,
	ACNTLZDCC,
	ACMPW,	/* CMP with L=0 */
	ACMPWU,
	ADIVD,
	ADIVDCC,
	ADIVDVCC,
	ADIVDV,
	ADIVDU,
	ADIVDUCC,
	ADIVDUVCC,
	ADIVDUV,
	AEXTSW,
	AEXTSWCC,
	/* AFCFIW; AFCFIWCC */
	AFCFID,
	AFCFIDCC,
	AFCTID,
	AFCTIDCC,
	AFCTIDZ,
	AFCTIDZCC,
	ALDAR,
	AMOVD,
	AMOVDU,
	AMOVWZ,
	AMOVWZU,
	AMULHD,
	AMULHDCC,
	AMULHDU,
	AMULHDUCC,
	AMULLD,
	AMULLDCC,
	AMULLDVCC,
	AMULLDV,
	ARFID,
	ARLDMI,
	ARLDMICC,
	ARLDC,
	ARLDCCC,
	ARLDCR,
	ARLDCRCC,
	ARLDCL,
	ARLDCLCC,
	ASLBIA,
	ASLBIE,
	ASLBMFEE,
	ASLBMFEV,
	ASLBMTE,
	ASLD,
	ASLDCC,
	ASRD,
	ASRAD,
	ASRADCC,
	ASRDCC,
	ASTDCCC,
	ATD,

	/* 64-bit pseudo operation */
	ADWORD,
	AREMD,
	AREMDCC,
	AREMDV,
	AREMDVCC,
	AREMDU,
	AREMDUCC,
	AREMDUV,
	AREMDUVCC,

	/* more 64-bit operations */
	AHRFID,

	ALAST,
	
	// aliases
	ABR = AJMP,
	ABL = ACALL,
	ARETURN = ARET,
};

/*
 * this is the ranlib header
 */
#define	SYMDEF	"__.GOSYMDEF"
