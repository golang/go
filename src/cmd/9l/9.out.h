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

enum
{
	REGZERO		= 0,	/* set to zero */
	REGSP		= 1,
	REGSB		= 2,
	REGRET		= 3,
	REGARG		= -1,	/* -1 disables passing the first argument in register */
	REGRT1		= 3,	/* reserved for runtime, duffzero and duffcopy */
	REGRT2		= 4,	/* reserved for runtime, duffcopy */
	REGMIN		= 7,	/* register variables allocated from here to REGMAX */
	REGENV		= 11,	/* environment for closures */
	REGTLS		= 13,	/* C ABI TLS base pointer */
	REGMAX		= 27,
	REGEXT		= 30,	/* external registers allocated from here down */
	REGG		= 30,	/* G */
	REGTMP		= 31,	/* used by the linker */

	FREGRET		= 0,
	FREGMIN		= 17,	/* first register variable */
	FREGMAX		= 26,	/* last register variable for 9g only */
	FREGEXT		= 26,	/* first external register */
	FREGCVI		= 27,	/* floating conversion constant */
	FREGZERO	= 28,	/* both float and double */
	FREGHALF	= 29,	/* double */
	FREGONE		= 30,	/* double */
	FREGTWO		= 31	/* double */
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

	C_NCLASS,	/* must be the last */
};

enum
{
	AXXX,
	AADD,
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
	ABL,
	ABLE,
	ABLT,
	ABNE,
	ABR,
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
	ADATA,
	AGLOBL,
	AGOK,
	AHISTORY,
	ANAME,
	ANOP,
	ARETURN,
	ATEXT,
	AWORD,
	AEND,
	ADYNT,
	AINIT,
	ASIGNAME,

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

	AUNDEF,
	AUSEFIELD,
	ATYPE,
	AFUNCDATA,
	APCDATA,
	ACHECKNIL,
	AVARDEF,
	AVARKILL,
	ADUFFCOPY,
	ADUFFZERO,

	ALAST
};

/* type/name */
enum
{
	D_GOK	= 0,
	D_NONE,

/* name */
	D_EXTERN,
	D_STATIC,
	D_AUTO,
	D_PARAM,

/* type */
	D_BRANCH,
	D_OREG,
	D_CONST,
	D_FCONST,
	D_SCONST,
	D_REG,
	D_FPSCR,
	D_MSR,
	D_FREG,
	D_CREG,
	D_SPR,
	D_OPT,	/* branch/trap option */
	D_FILE,
	D_FILE1,
	D_DCR,	/* device control register */
	D_DCONST,
	D_ADDR, // not used, use D_CONST with non-empty sym.

	D_LAST,

/* reg names for 9g OREGISTER */
	D_R0 = 0, // type is D_REG
	D_F0 = D_R0+NREG, // type is D_FREG

/* reg names in offset field iff type is D_SPR */
	D_XER	= 1,
	D_LR	= 8,
	D_CTR	= 9
	/* and many supervisor level registers */
};

/*
 * this is the ranlib header
 */
#define	SYMDEF	"__.GOSYMDEF"
