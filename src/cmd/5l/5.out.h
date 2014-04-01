// Inferno utils/5c/5.out.h
// http://code.google.com/p/inferno-os/source/browse/utils/5c/5.out.h
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
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

#define	NSNAME		8
#define	NSYM		50
#define	NREG		16
#include "../ld/textflag.h"

#define	REGRET		0
/* -1 disables use of REGARG */
#define	REGARG		-1
/* compiler allocates R1 up as temps */
/* compiler allocates register variables R3 up */
#define	REGEXT		10
/* these two registers are declared in runtime.h */
#define REGG        (REGEXT-0)
#define REGM        (REGEXT-1)
/* compiler allocates external registers R10 down */
#define	REGTMP		11
#define	REGSP		13
#define	REGLINK		14
#define	REGPC		15

#define	NFREG		16
#define	FREGRET		0
#define	FREGEXT		7
#define	FREGTMP		15
/* compiler allocates register variables F0 up */
/* compiler allocates external registers F7 down */

enum	as
{
	AXXX,

	AAND,
	AEOR,
	ASUB,
	ARSB,
	AADD,
	AADC,
	ASBC,
	ARSC,
	ATST,
	ATEQ,
	ACMP,
	ACMN,
	AORR,
	ABIC,

	AMVN,

	AB,
	ABL,

/*
 * Do not reorder or fragment the conditional branch
 * opcodes, or the predication code will break
 */
	ABEQ,
	ABNE,
	ABCS,
	ABHS,
	ABCC,
	ABLO,
	ABMI,
	ABPL,
	ABVS,
	ABVC,
	ABHI,
	ABLS,
	ABGE,
	ABLT,
	ABGT,
	ABLE,

	AMOVWD,
	AMOVWF,
	AMOVDW,
	AMOVFW,
	AMOVFD,
	AMOVDF,
	AMOVF,
	AMOVD,

	ACMPF,
	ACMPD,
	AADDF,
	AADDD,
	ASUBF,
	ASUBD,
	AMULF,
	AMULD,
	ADIVF,
	ADIVD,
	ASQRTF,
	ASQRTD,
	AABSF,
	AABSD,

	ASRL,
	ASRA,
	ASLL,
	AMULU,
	ADIVU,
	AMUL,
	ADIV,
	AMOD,
	AMODU,

	AMOVB,
	AMOVBS,
	AMOVBU,
	AMOVH,
	AMOVHS,
	AMOVHU,
	AMOVW,
	AMOVM,
	ASWPBU,
	ASWPW,

	ANOP,
	ARFE,
	ASWI,
	AMULA,

	ADATA,
	AGLOBL,
	AGOK,
	AHISTORY,
	ANAME,
	ARET,
	ATEXT,
	AWORD,
	ADYNT_,
	AINIT_,
	ABCASE,
	ACASE,

	AEND,

	AMULL,
	AMULAL,
	AMULLU,
	AMULALU,

	ABX,
	ABXRET,
	ADWORD,

	ASIGNAME,

	ALDREX,
	ASTREX,
	
	ALDREXD,
	ASTREXD,

	APLD,

	AUNDEF,

	ACLZ,

	AMULWT,
	AMULWB,
	AMULAWT,
	AMULAWB,
	
	AUSEFIELD,
	ATYPE,
	AFUNCDATA,
	APCDATA,
	ACHECKNIL,
	AVARDEF,
	AVARKILL,

	AMRC, // MRC/MCR

	ALAST,
};

/* scond byte */
#define	C_SCOND	((1<<4)-1)
#define	C_SBIT	(1<<4)
#define	C_PBIT	(1<<5)
#define	C_WBIT	(1<<6)
#define	C_FBIT	(1<<7)	/* psr flags-only */
#define	C_UBIT	(1<<7)	/* up bit, unsigned bit */

#define C_SCOND_EQ	0
#define C_SCOND_NE	1
#define C_SCOND_HS	2
#define C_SCOND_LO	3
#define C_SCOND_MI	4
#define C_SCOND_PL	5
#define C_SCOND_VS	6
#define C_SCOND_VC	7
#define C_SCOND_HI	8
#define C_SCOND_LS	9
#define C_SCOND_GE	10
#define C_SCOND_LT	11
#define C_SCOND_GT	12
#define C_SCOND_LE	13
#define C_SCOND_NONE	14
#define C_SCOND_NV	15

/* D_SHIFT type */
#define SHIFT_LL		0<<5
#define SHIFT_LR		1<<5
#define SHIFT_AR		2<<5
#define SHIFT_RR		3<<5

enum
{
/* type/name */
	D_GOK = 0,
	D_NONE = 1,

/* type */
	D_BRANCH = (D_NONE+1),
	D_OREG = (D_NONE+2),
	D_CONST = (D_NONE+7),
	D_FCONST = (D_NONE+8),
	D_SCONST = (D_NONE+9),
	D_PSR = (D_NONE+10),
	D_REG = (D_NONE+12),
	D_FREG = (D_NONE+13),
	D_FILE = (D_NONE+16),
	D_OCONST = (D_NONE+17),
	D_FILE1 = (D_NONE+18),

	D_SHIFT = (D_NONE+19),
	D_FPCR = (D_NONE+20),
	D_REGREG = (D_NONE+21), // (reg, reg)
	D_ADDR = (D_NONE+22),

	D_SBIG = (D_NONE+23),
	D_CONST2 = (D_NONE+24),

	D_REGREG2 = (D_NONE+25), // reg, reg

/* name */
	D_EXTERN = (D_NONE+3),
	D_STATIC = (D_NONE+4),
	D_AUTO = (D_NONE+5),
	D_PARAM = (D_NONE+6),

/* internal only */
	D_SIZE = (D_NONE+40),
	D_PCREL = (D_NONE+41),
	D_GOTOFF = (D_NONE+42), // R_ARM_GOTOFF
	D_PLT0 = (D_NONE+43), // R_ARM_PLT32, 1st inst: add ip, pc, #0xNN00000
	D_PLT1 = (D_NONE+44), // R_ARM_PLT32, 2nd inst: add ip, ip, #0xNN000
	D_PLT2 = (D_NONE+45), // R_ARM_PLT32, 3rd inst: ldr pc, [ip, #0xNNN]!
	D_CALL = (D_NONE+46), // R_ARM_PLT32/R_ARM_CALL/R_ARM_JUMP24, bl xxxxx or b yyyyy
	D_TLS = (D_NONE+47), // R_ARM_TLS_LE32/R_ARM_TLS_IE32
};

/*
 * this is the ranlib header
 */
#define	SYMDEF	"__.GOSYMDEF"
