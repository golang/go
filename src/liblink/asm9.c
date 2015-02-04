// cmd/9l/optab.c, cmd/9l/asmout.c from Vita Nuova.
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

// Instruction layout.

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>
#include "../cmd/9l/9.out.h"
#include "../runtime/stack.h"

enum {
	FuncAlign = 8,
};

enum {
	r0iszero = 1,
};

typedef	struct	Optab	Optab;

struct	Optab
{
	short	as;
	uchar	a1;
	uchar	a2;
	uchar	a3;
	uchar	a4;
	char	type;
	char	size;
	char	param;
};

static Optab	optab[] = {
	{ ATEXT,	C_LEXT,	C_NONE, C_NONE, 	C_TEXTSIZE, 	 0, 0, 0 },
	{ ATEXT,	C_LEXT,	C_NONE, C_LCON, 	C_TEXTSIZE, 	 0, 0, 0 },
	{ ATEXT,	C_ADDR,	C_NONE, C_NONE, 	C_TEXTSIZE, 	 0, 0, 0 },
	{ ATEXT,	C_ADDR,	C_NONE, C_LCON, 	C_TEXTSIZE, 	 0, 0, 0 },

	/* move register */
	{ AMOVD,	C_REG,	C_NONE, C_NONE, 	C_REG,		 1, 4, 0 },
	{ AMOVB,	C_REG,	C_NONE, C_NONE, 	C_REG,		12, 4, 0 },
	{ AMOVBZ,	C_REG,	C_NONE, C_NONE, 	C_REG,		13, 4, 0 },
	{ AMOVW,	C_REG,	C_NONE, C_NONE, 	C_REG,		 12, 4, 0 },
	{ AMOVWZ,	C_REG,	C_NONE, C_NONE, 	C_REG,		 13, 4, 0 },

	{ AADD,		C_REG,	C_REG, C_NONE, 	C_REG,		 2, 4, 0 },
	{ AADD,		C_REG,	C_NONE, C_NONE, 	C_REG,		 2, 4, 0 },
	{ AADD,		C_ADDCON,C_REG, C_NONE, 	C_REG,		 4, 4, 0 },
	{ AADD,		C_ADDCON,C_NONE, C_NONE, C_REG,		 4, 4, 0 },
	{ AADD,		C_UCON,	C_REG, C_NONE, 	C_REG,		20, 4, 0 },
	{ AADD,		C_UCON,	C_NONE, C_NONE, 	C_REG,		20, 4, 0 },
	{ AADD,		C_LCON,	C_REG, C_NONE, 	C_REG,		22, 12, 0 },
	{ AADD,		C_LCON,	C_NONE, C_NONE, 	C_REG,		22, 12, 0 },

	{ AADDC,	C_REG,	C_REG, C_NONE, 	C_REG,		 2, 4, 0 },
	{ AADDC,	C_REG,	C_NONE, C_NONE, 	C_REG,		 2, 4, 0 },
	{ AADDC,	C_ADDCON,C_REG, C_NONE, 	C_REG,		 4, 4, 0 },
	{ AADDC,	C_ADDCON,C_NONE, C_NONE, C_REG,		 4, 4, 0 },
	{ AADDC,	C_LCON,	C_REG, C_NONE, 	C_REG,		22, 12, 0 },
	{ AADDC,	C_LCON,	C_NONE, C_NONE, 	C_REG,		22, 12, 0 },

	{ AAND,		C_REG,	C_REG, C_NONE, 	C_REG,		6, 4, 0 },	/* logical, no literal */
	{ AAND,		C_REG,	C_NONE, C_NONE, 	C_REG,		6, 4, 0 },
	{ AANDCC,	C_REG,	C_REG, C_NONE, 	C_REG,		6, 4, 0 },
	{ AANDCC,	C_REG,	C_NONE, C_NONE, 	C_REG,		6, 4, 0 },

	{ AANDCC,	C_ANDCON,C_NONE, C_NONE, C_REG,		58, 4, 0 },
	{ AANDCC,	C_ANDCON,C_REG, C_NONE, 	C_REG,		58, 4, 0 },
	{ AANDCC,	C_UCON,	C_NONE, C_NONE, 	C_REG,		59, 4, 0 },
	{ AANDCC,	C_UCON,	C_REG, C_NONE, 	C_REG,		59, 4, 0 },
	{ AANDCC,	C_LCON,	C_NONE, C_NONE, 	C_REG,		23, 12, 0 },
	{ AANDCC,	C_LCON,	C_REG, C_NONE, 	C_REG,		23, 12, 0 },

	{ AMULLW,	C_REG,	C_REG, C_NONE, 	C_REG,		 2, 4, 0 },
	{ AMULLW,	C_REG,	C_NONE, C_NONE, 	C_REG,		 2, 4, 0 },
	{ AMULLW,	C_ADDCON,C_REG, C_NONE, 	C_REG,		 4, 4, 0 },
	{ AMULLW,	C_ADDCON,C_NONE, C_NONE, C_REG,		 4, 4, 0 },
	{ AMULLW,	C_ANDCON,C_REG, C_NONE, 	C_REG,		 4, 4, 0 },
	{ AMULLW,	C_ANDCON,	C_NONE, C_NONE,	C_REG,	 4, 4, 0 },
	{ AMULLW,	C_LCON,	C_REG,	C_NONE,	C_REG,		22, 12, 0},
	{ AMULLW,	C_LCON,	C_NONE,	C_NONE,	C_REG,		22, 12, 0},

	{ ASUBC,	C_REG,	C_REG, C_NONE, 	C_REG,		 10, 4, 0 },
	{ ASUBC,	C_REG,	C_NONE, C_NONE, 	C_REG,		 10, 4, 0 },
	{ ASUBC,	C_REG,	C_NONE, C_ADDCON, 	C_REG,	 27, 4, 0 },
	{ ASUBC,	C_REG,	C_NONE,	C_LCON,	C_REG,		28, 12, 0},

	{ AOR,		C_REG,	C_REG, C_NONE, 	C_REG,		6, 4, 0 },	/* logical, literal not cc (or/xor) */
	{ AOR,		C_REG,	C_NONE, C_NONE, 	C_REG,		6, 4, 0 },
	{ AOR,		C_ANDCON, C_NONE, C_NONE,  C_REG,	58, 4, 0 },
	{ AOR,		C_ANDCON, C_REG, C_NONE,  C_REG,		58, 4, 0 },
	{ AOR,		C_UCON, C_NONE, C_NONE,  C_REG,		59, 4, 0 },
	{ AOR,		C_UCON, C_REG, C_NONE,  C_REG,		59, 4, 0 },
	{ AOR,		C_LCON,	C_NONE, C_NONE, 	C_REG,		23, 12, 0 },
	{ AOR,		C_LCON,	C_REG, C_NONE, 	C_REG,		23, 12, 0 },

	{ ADIVW,	C_REG,	C_REG, C_NONE, 	C_REG,		 2, 4, 0 },	/* op r1[,r2],r3 */
	{ ADIVW,	C_REG,	C_NONE, C_NONE, 	C_REG,		 2, 4, 0 },
	{ ASUB,	C_REG,	C_REG, C_NONE, 	C_REG,		 10, 4, 0 },	/* op r2[,r1],r3 */
	{ ASUB,	C_REG,	C_NONE, C_NONE, 	C_REG,		 10, 4, 0 },

	{ ASLW,	C_REG,	C_NONE, C_NONE, 	C_REG,		 6, 4, 0 },
	{ ASLW,	C_REG,	C_REG, C_NONE, 	C_REG,		 6, 4, 0 },
	{ ASLD,	C_REG,	C_NONE, C_NONE, 	C_REG,		 6, 4, 0 },
	{ ASLD,	C_REG,	C_REG, C_NONE, 	C_REG,		 6, 4, 0 },
	{ ASLD,	C_SCON,	C_REG, C_NONE,	C_REG,		25, 4, 0 },
	{ ASLD,	C_SCON,	C_NONE, C_NONE,	C_REG,		25, 4, 0 },
	{ ASLW,	C_SCON,	C_REG, C_NONE, 	C_REG,		57, 4, 0 },
	{ ASLW,	C_SCON,	C_NONE, C_NONE, 	C_REG,		57, 4, 0 },

	{ ASRAW,	C_REG,	C_NONE, C_NONE, 	C_REG,		 6, 4, 0 },
	{ ASRAW,	C_REG,	C_REG, C_NONE, 	C_REG,		 6, 4, 0 },
	{ ASRAW,	C_SCON,	C_REG, C_NONE, 	C_REG,		56, 4, 0 },
	{ ASRAW,	C_SCON,	C_NONE, C_NONE, 	C_REG,		56, 4, 0 },
	{ ASRAD,	C_REG,	C_NONE, C_NONE, 	C_REG,		 6, 4, 0 },
	{ ASRAD,	C_REG,	C_REG, C_NONE, 	C_REG,		 6, 4, 0 },
	{ ASRAD,	C_SCON,	C_REG, C_NONE, 	C_REG,		56, 4, 0 },
	{ ASRAD,	C_SCON,	C_NONE, C_NONE, 	C_REG,		56, 4, 0 },

	{ ARLWMI,	C_SCON, C_REG, C_LCON, 	C_REG,		62, 4, 0 },
	{ ARLWMI,	C_REG,	C_REG, C_LCON, 	C_REG,		63, 4, 0 },
	{ ARLDMI,	C_SCON,	C_REG, C_LCON,	C_REG,		30, 4, 0 },

	{ ARLDC,	C_SCON,	C_REG, C_LCON,	C_REG,		29, 4, 0 },
	{ ARLDCL,	C_SCON,	C_REG, C_LCON,	C_REG,		29, 4, 0 },
	{ ARLDCL,	C_REG,	C_REG,	C_LCON,	C_REG,		14, 4, 0 },
	{ ARLDCL, C_REG,	C_NONE,	C_LCON,	C_REG,		14, 4, 0 },

	{ AFADD,	C_FREG,	C_NONE, C_NONE, 	C_FREG,		 2, 4, 0 },
	{ AFADD,	C_FREG,	C_REG, C_NONE, 	C_FREG,		 2, 4, 0 },
	{ AFABS,	C_FREG,	C_NONE, C_NONE, 	C_FREG,		33, 4, 0 },
	{ AFABS,	C_NONE,	C_NONE, C_NONE, 	C_FREG,		33, 4, 0 },
	{ AFMOVD,	C_FREG,	C_NONE, C_NONE, 	C_FREG,		33, 4, 0 },

	{ AFMADD,	C_FREG,	C_REG, C_FREG, 	C_FREG,		 34, 4, 0 },
	{ AFMUL,	C_FREG,	C_NONE, C_NONE, 	C_FREG,		 32, 4, 0 },
	{ AFMUL,	C_FREG,	C_REG, C_NONE, 	C_FREG,		 32, 4, 0 },

	/* store, short offset */
	{ AMOVD,	C_REG,	C_REG, C_NONE, 	C_ZOREG,	 7, 4, REGZERO },
	{ AMOVW,	C_REG,	C_REG, C_NONE, 	C_ZOREG,	 7, 4, REGZERO },
	{ AMOVWZ,	C_REG,	C_REG, C_NONE, 	C_ZOREG,	 7, 4, REGZERO },
	{ AMOVBZ,	C_REG,	C_REG, C_NONE, 	C_ZOREG,	 7, 4, REGZERO },
	{ AMOVBZU,	C_REG,	C_REG, C_NONE, 	C_ZOREG,	 7, 4, REGZERO },
	{ AMOVB,	C_REG,	C_REG, C_NONE, 	C_ZOREG,	 7, 4, REGZERO },
	{ AMOVBU,	C_REG,	C_REG, C_NONE, 	C_ZOREG,	 7, 4, REGZERO },
	{ AMOVD,	C_REG,	C_NONE, C_NONE, 	C_SEXT,		 7, 4, REGSB },
	{ AMOVW,	C_REG,	C_NONE, C_NONE, 	C_SEXT,		 7, 4, REGSB },
	{ AMOVWZ,	C_REG,	C_NONE, C_NONE, 	C_SEXT,		 7, 4, REGSB },
	{ AMOVBZ,	C_REG,	C_NONE, C_NONE, 	C_SEXT,		 7, 4, REGSB },
	{ AMOVB,	C_REG,	C_NONE, C_NONE, 	C_SEXT,		 7, 4, REGSB },
	{ AMOVD,	C_REG,	C_NONE, C_NONE, 	C_SAUTO,	 7, 4, REGSP },
	{ AMOVW,	C_REG,	C_NONE, C_NONE, 	C_SAUTO,	 7, 4, REGSP },
	{ AMOVWZ,	C_REG,	C_NONE, C_NONE, 	C_SAUTO,	 7, 4, REGSP },
	{ AMOVBZ,	C_REG,	C_NONE, C_NONE, 	C_SAUTO,	 7, 4, REGSP },
	{ AMOVB,	C_REG,	C_NONE, C_NONE, 	C_SAUTO,	 7, 4, REGSP },
	{ AMOVD,	C_REG,	C_NONE, C_NONE, 	C_SOREG,	 7, 4, REGZERO },
	{ AMOVW,	C_REG,	C_NONE, C_NONE, 	C_SOREG,	 7, 4, REGZERO },
	{ AMOVWZ,	C_REG,	C_NONE, C_NONE, 	C_SOREG,	 7, 4, REGZERO },
	{ AMOVBZ,	C_REG,	C_NONE, C_NONE, 	C_SOREG,	 7, 4, REGZERO },
	{ AMOVBZU,	C_REG,	C_NONE, C_NONE, 	C_SOREG,	 7, 4, REGZERO },
	{ AMOVB,	C_REG,	C_NONE, C_NONE, 	C_SOREG,	 7, 4, REGZERO },
	{ AMOVBU,	C_REG,	C_NONE, C_NONE, 	C_SOREG,	 7, 4, REGZERO },

	/* load, short offset */
	{ AMOVD,	C_ZOREG,C_REG, C_NONE, 	C_REG,		 8, 4, REGZERO },
	{ AMOVW,	C_ZOREG,C_REG, C_NONE, 	C_REG,		 8, 4, REGZERO },
	{ AMOVWZ,	C_ZOREG,C_REG, C_NONE, 	C_REG,		 8, 4, REGZERO },
	{ AMOVBZ,	C_ZOREG,C_REG, C_NONE, 	C_REG,		 8, 4, REGZERO },
	{ AMOVBZU,	C_ZOREG,C_REG, C_NONE, 	C_REG,		 8, 4, REGZERO },
	{ AMOVB,	C_ZOREG,C_REG, C_NONE, 	C_REG,		9, 8, REGZERO },
	{ AMOVBU,	C_ZOREG,C_REG, C_NONE, 	C_REG,		9, 8, REGZERO },
	{ AMOVD,	C_SEXT,	C_NONE, C_NONE, 	C_REG,		 8, 4, REGSB },
	{ AMOVW,	C_SEXT,	C_NONE, C_NONE, 	C_REG,		 8, 4, REGSB },
	{ AMOVWZ,	C_SEXT,	C_NONE, C_NONE, 	C_REG,		 8, 4, REGSB },
	{ AMOVBZ,	C_SEXT,	C_NONE, C_NONE, 	C_REG,		 8, 4, REGSB },
	{ AMOVB,	C_SEXT,	C_NONE, C_NONE, 	C_REG,		9, 8, REGSB },
	{ AMOVD,	C_SAUTO,C_NONE, C_NONE, 	C_REG,		 8, 4, REGSP },
	{ AMOVW,	C_SAUTO,C_NONE, C_NONE, 	C_REG,		 8, 4, REGSP },
	{ AMOVWZ,	C_SAUTO,C_NONE, C_NONE, 	C_REG,		 8, 4, REGSP },
	{ AMOVBZ,	C_SAUTO,C_NONE, C_NONE, 	C_REG,		 8, 4, REGSP },
	{ AMOVB,	C_SAUTO,C_NONE, C_NONE, 	C_REG,		9, 8, REGSP },
	{ AMOVD,	C_SOREG,C_NONE, C_NONE, 	C_REG,		 8, 4, REGZERO },
	{ AMOVW,	C_SOREG,C_NONE, C_NONE, 	C_REG,		 8, 4, REGZERO },
	{ AMOVWZ,	C_SOREG,C_NONE, C_NONE, 	C_REG,		 8, 4, REGZERO },
	{ AMOVBZ,	C_SOREG,C_NONE, C_NONE, 	C_REG,		 8, 4, REGZERO },
	{ AMOVBZU,	C_SOREG,C_NONE, C_NONE, 	C_REG,		 8, 4, REGZERO },
	{ AMOVB,	C_SOREG,C_NONE, C_NONE, 	C_REG,		9, 8, REGZERO },
	{ AMOVBU,	C_SOREG,C_NONE, C_NONE, 	C_REG,		9, 8, REGZERO },

	/* store, long offset */
	{ AMOVD,	C_REG,	C_NONE, C_NONE, 	C_LEXT,		35, 8, REGSB },
	{ AMOVW,	C_REG,	C_NONE, C_NONE, 	C_LEXT,		35, 8, REGSB },
	{ AMOVWZ,	C_REG,	C_NONE, C_NONE, 	C_LEXT,		35, 8, REGSB },
	{ AMOVBZ,	C_REG,	C_NONE, C_NONE, 	C_LEXT,		35, 8, REGSB },
	{ AMOVB,	C_REG,	C_NONE, C_NONE, 	C_LEXT,		35, 8, REGSB },
	{ AMOVD,	C_REG,	C_NONE, C_NONE, 	C_LAUTO,	35, 8, REGSP },
	{ AMOVW,	C_REG,	C_NONE, C_NONE, 	C_LAUTO,	35, 8, REGSP },
	{ AMOVWZ,	C_REG,	C_NONE, C_NONE, 	C_LAUTO,	35, 8, REGSP },
	{ AMOVBZ,	C_REG,	C_NONE, C_NONE, 	C_LAUTO,	35, 8, REGSP },
	{ AMOVB,	C_REG,	C_NONE, C_NONE, 	C_LAUTO,	35, 8, REGSP },
	{ AMOVD,	C_REG,	C_NONE, C_NONE, 	C_LOREG,	35, 8, REGZERO },
	{ AMOVW,	C_REG,	C_NONE, C_NONE, 	C_LOREG,	35, 8, REGZERO },
	{ AMOVWZ,	C_REG,	C_NONE, C_NONE, 	C_LOREG,	35, 8, REGZERO },
	{ AMOVBZ,	C_REG,	C_NONE, C_NONE, 	C_LOREG,	35, 8, REGZERO },
	{ AMOVB,	C_REG,	C_NONE, C_NONE, 	C_LOREG,	35, 8, REGZERO },
	{ AMOVD,	C_REG,	C_NONE, C_NONE, 	C_ADDR,		74, 8, 0 },
	{ AMOVW,	C_REG,	C_NONE, C_NONE, 	C_ADDR,		74, 8, 0 },
	{ AMOVWZ,	C_REG,	C_NONE, C_NONE, 	C_ADDR,		74, 8, 0 },
	{ AMOVBZ,	C_REG,	C_NONE, C_NONE, 	C_ADDR,		74, 8, 0 },
	{ AMOVB,	C_REG,	C_NONE, C_NONE, 	C_ADDR,		74, 8, 0 },

	/* load, long offset */
	{ AMOVD,	C_LEXT,	C_NONE, C_NONE, 	C_REG,		36, 8, REGSB },
	{ AMOVW,	C_LEXT,	C_NONE, C_NONE, 	C_REG,		36, 8, REGSB },
	{ AMOVWZ,	C_LEXT,	C_NONE, C_NONE, 	C_REG,		36, 8, REGSB },
	{ AMOVBZ,	C_LEXT,	C_NONE, C_NONE, 	C_REG,		36, 8, REGSB },
	{ AMOVB,	C_LEXT,	C_NONE, C_NONE, 	C_REG,		37, 12, REGSB },
	{ AMOVD,	C_LAUTO,C_NONE, C_NONE, 	C_REG,		36, 8, REGSP },
	{ AMOVW,	C_LAUTO,C_NONE, C_NONE, 	C_REG,		36, 8, REGSP },
	{ AMOVWZ,	C_LAUTO,C_NONE, C_NONE, 	C_REG,		36, 8, REGSP },
	{ AMOVBZ,	C_LAUTO,C_NONE, C_NONE, 	C_REG,		36, 8, REGSP },
	{ AMOVB,	C_LAUTO,C_NONE, C_NONE, 	C_REG,		37, 12, REGSP },
	{ AMOVD,	C_LOREG,C_NONE, C_NONE, 	C_REG,		36, 8, REGZERO },
	{ AMOVW,	C_LOREG,C_NONE, C_NONE, 	C_REG,		36, 8, REGZERO },
	{ AMOVWZ,	C_LOREG,C_NONE, C_NONE, 	C_REG,		36, 8, REGZERO },
	{ AMOVBZ,	C_LOREG,C_NONE, C_NONE, 	C_REG,		36, 8, REGZERO },
	{ AMOVB,	C_LOREG,C_NONE, C_NONE, 	C_REG,		37, 12, REGZERO },
	{ AMOVD,	C_ADDR,	C_NONE, C_NONE, 	C_REG,		75, 8, 0 },
	{ AMOVW,	C_ADDR,	C_NONE, C_NONE, 	C_REG,		75, 8, 0 },
	{ AMOVWZ,	C_ADDR,	C_NONE, C_NONE, 	C_REG,		75, 8, 0 },
	{ AMOVBZ,	C_ADDR,	C_NONE, C_NONE, 	C_REG,		75, 8, 0 },
	{ AMOVB,	C_ADDR,	C_NONE, C_NONE, 	C_REG,		76, 12, 0 },

	/* load constant */
	{ AMOVD,	C_SECON,C_NONE, C_NONE, 	C_REG,		 3, 4, REGSB },
	{ AMOVD,	C_SACON,C_NONE, C_NONE, 	C_REG,		 3, 4, REGSP },
	{ AMOVD,	C_LECON,C_NONE, C_NONE, 	C_REG,		26, 8, REGSB }, 
	{ AMOVD,	C_LACON,C_NONE, C_NONE, 	C_REG,		26, 8, REGSP },
	{ AMOVD,	C_ADDCON,C_NONE, C_NONE, C_REG,		 3, 4, REGZERO },
	{ AMOVW,	C_SECON,C_NONE, C_NONE, 	C_REG,		 3, 4, REGSB },	/* TO DO: check */
	{ AMOVW,	C_SACON,C_NONE, C_NONE, 	C_REG,		 3, 4, REGSP },
	{ AMOVW,	C_LECON,C_NONE, C_NONE, 	C_REG,		26, 8, REGSB }, 
	{ AMOVW,	C_LACON,C_NONE, C_NONE, 	C_REG,		26, 8, REGSP },
	{ AMOVW,	C_ADDCON,C_NONE, C_NONE, C_REG,		 3, 4, REGZERO },
	{ AMOVWZ,	C_SECON,C_NONE, C_NONE, 	C_REG,		 3, 4, REGSB },	/* TO DO: check */
	{ AMOVWZ,	C_SACON,C_NONE, C_NONE, 	C_REG,		 3, 4, REGSP },
	{ AMOVWZ,	C_LECON,C_NONE, C_NONE, 	C_REG,		26, 8, REGSB }, 
	{ AMOVWZ,	C_LACON,C_NONE, C_NONE, 	C_REG,		26, 8, REGSP },
	{ AMOVWZ,	C_ADDCON,C_NONE, C_NONE, C_REG,		 3, 4, REGZERO },

	/* load unsigned/long constants (TO DO: check) */
	{ AMOVD,	C_UCON, C_NONE, C_NONE,  C_REG,		3, 4, REGZERO },
	{ AMOVD,	C_LCON,	C_NONE, C_NONE, 	C_REG,		19, 8, 0 },
	{ AMOVW,	C_UCON, C_NONE, C_NONE,  C_REG,		3, 4, REGZERO },
	{ AMOVW,	C_LCON,	C_NONE, C_NONE, 	C_REG,		19, 8, 0 },
	{ AMOVWZ,	C_UCON, C_NONE, C_NONE,  C_REG,		3, 4, REGZERO },
	{ AMOVWZ,	C_LCON,	C_NONE, C_NONE, 	C_REG,		19, 8, 0 },

	{ AMOVHBR,	C_ZOREG,	C_REG, C_NONE, C_REG,		45, 4, 0 },
	{ AMOVHBR,	C_ZOREG, C_NONE, C_NONE, C_REG,	45, 4, 0 },
	{ AMOVHBR,	C_REG,	C_REG, C_NONE,	C_ZOREG,		44, 4, 0 },
	{ AMOVHBR,	C_REG,	C_NONE, C_NONE,	C_ZOREG,		44, 4, 0 },

	{ ASYSCALL,	C_NONE,	C_NONE, C_NONE, 	C_NONE,		 5, 4, 0 },
	{ ASYSCALL,	C_REG,	C_NONE, C_NONE, 	C_NONE,		 77, 12, 0 },
	{ ASYSCALL,	C_SCON,	C_NONE, C_NONE, 	C_NONE,		 77, 12, 0 },

	{ ABEQ,		C_NONE,	C_NONE, C_NONE, 	C_SBRA,		16, 4, 0 },
	{ ABEQ,		C_CREG,	C_NONE, C_NONE, 	C_SBRA,		16, 4, 0 },

	{ ABR,		C_NONE,	C_NONE, C_NONE, 	C_LBRA,		11, 4, 0 },

	{ ABC,		C_SCON,	C_REG, C_NONE, 	C_SBRA,		16, 4, 0 },
	{ ABC,		C_SCON, C_REG, C_NONE, 	C_LBRA,		17, 4, 0 },

	{ ABR,		C_NONE,	C_NONE, C_NONE, 	C_LR,		18, 4, 0 },
	{ ABR,		C_NONE,	C_NONE, C_NONE, 	C_CTR,		18, 4, 0 },
	{ ABR,		C_REG,	C_NONE, C_NONE, 	C_CTR,		18, 4, 0 },
	{ ABR,		C_NONE,	C_NONE, C_NONE, 	C_ZOREG,		15, 8, 0 },

	{ ABC,		C_NONE,	C_REG, C_NONE, 	C_LR,		18, 4, 0 },
	{ ABC,		C_NONE,	C_REG, C_NONE, 	C_CTR,		18, 4, 0 },
	{ ABC,		C_SCON,	C_REG, C_NONE, 	C_LR,		18, 4, 0 },
	{ ABC,		C_SCON,	C_REG, C_NONE, 	C_CTR,		18, 4, 0 },
	{ ABC,		C_NONE,	C_NONE, C_NONE, 	C_ZOREG,		15, 8, 0 },

	{ AFMOVD,	C_SEXT,	C_NONE, C_NONE, 	C_FREG,		8, 4, REGSB },
	{ AFMOVD,	C_SAUTO,C_NONE, C_NONE, 	C_FREG,		8, 4, REGSP },
	{ AFMOVD,	C_SOREG,C_NONE, C_NONE, 	C_FREG,		8, 4, REGZERO },

	{ AFMOVD,	C_LEXT,	C_NONE, C_NONE, 	C_FREG,		36, 8, REGSB },
	{ AFMOVD,	C_LAUTO,C_NONE, C_NONE, 	C_FREG,		36, 8, REGSP },
	{ AFMOVD,	C_LOREG,C_NONE, C_NONE, 	C_FREG,		36, 8, REGZERO },
	{ AFMOVD,	C_ADDR,	C_NONE, C_NONE, 	C_FREG,		75, 8, 0 },

	{ AFMOVD,	C_FREG,	C_NONE, C_NONE, 	C_SEXT,		7, 4, REGSB },
	{ AFMOVD,	C_FREG,	C_NONE, C_NONE, 	C_SAUTO,	7, 4, REGSP },
	{ AFMOVD,	C_FREG,	C_NONE, C_NONE, 	C_SOREG,	7, 4, REGZERO },

	{ AFMOVD,	C_FREG,	C_NONE, C_NONE, 	C_LEXT,		35, 8, REGSB },
	{ AFMOVD,	C_FREG,	C_NONE, C_NONE, 	C_LAUTO,	35, 8, REGSP },
	{ AFMOVD,	C_FREG,	C_NONE, C_NONE, 	C_LOREG,	35, 8, REGZERO },
	{ AFMOVD,	C_FREG,	C_NONE, C_NONE, 	C_ADDR,		74, 8, 0 },

	{ ASYNC,		C_NONE,	C_NONE, C_NONE, 	C_NONE,		46, 4, 0 },
	{ AWORD,	C_LCON,	C_NONE, C_NONE, 	C_NONE,		40, 4, 0 },
	{ ADWORD,	C_LCON,	C_NONE, C_NONE, C_NONE,	31, 8, 0 },
	{ ADWORD,	C_DCON,	C_NONE, C_NONE, C_NONE,	31, 8, 0 },

	{ AADDME,	C_REG,	C_NONE, C_NONE, 	C_REG,		47, 4, 0 },

	{ AEXTSB,	C_REG,	C_NONE, C_NONE, 	C_REG,		48, 4, 0 },
	{ AEXTSB,	C_NONE,	C_NONE, C_NONE, 	C_REG,		48, 4, 0 },

	{ ANEG,		C_REG,	C_NONE, C_NONE, 	C_REG,		47, 4, 0 },
	{ ANEG,		C_NONE,	C_NONE, C_NONE, 	C_REG,		47, 4, 0 },

	{ AREM,		C_REG,	C_NONE, C_NONE, 	C_REG,		50, 12, 0 },
	{ AREM,		C_REG,	C_REG, C_NONE, 	C_REG,		50, 12, 0 },
	{ AREMU,		C_REG,	C_NONE, C_NONE, 	C_REG,		50, 16, 0 },
	{ AREMU,		C_REG,	C_REG, C_NONE, 	C_REG,		50, 16, 0 },
	{ AREMD,		C_REG,	C_NONE, C_NONE, 	C_REG,		51, 12, 0 },
	{ AREMD,		C_REG,	C_REG, C_NONE, 	C_REG,		51, 12, 0 },
	{ AREMDU,		C_REG,	C_NONE, C_NONE, 	C_REG,		51, 12, 0 },
	{ AREMDU,		C_REG,	C_REG, C_NONE, 	C_REG,		51, 12, 0 },

	{ AMTFSB0,	C_SCON,	C_NONE, C_NONE, 	C_NONE,		52, 4, 0 },
	{ AMOVFL, C_FPSCR, C_NONE, C_NONE,	C_FREG,		53, 4, 0 },
	{ AMOVFL, C_FREG, C_NONE, C_NONE,	C_FPSCR,		64, 4, 0 },
	{ AMOVFL, C_FREG, C_NONE, C_LCON,	C_FPSCR,		64, 4, 0 },
	{ AMOVFL,	C_LCON, C_NONE, C_NONE,	C_FPSCR,		65, 4, 0 },

	{ AMOVD,	C_MSR,	C_NONE, C_NONE, 	C_REG,		54, 4, 0 },		/* mfmsr */
	{ AMOVD,	C_REG,	C_NONE, C_NONE, 	C_MSR,		54, 4, 0 },		/* mtmsrd */
	{ AMOVWZ,	C_REG,	C_NONE, C_NONE, 	C_MSR,		54, 4, 0 },		/* mtmsr */

	/* 64-bit special registers */
	{ AMOVD,	C_REG,	C_NONE, C_NONE, 	C_SPR,		66, 4, 0 },
	{ AMOVD,	C_REG,	C_NONE, C_NONE, 	C_LR,		66, 4, 0 },
	{ AMOVD,	C_REG,	C_NONE, C_NONE, 	C_CTR,		66, 4, 0 },
	{ AMOVD,	C_REG,	C_NONE, C_NONE, 	C_XER,		66, 4, 0 },
	{ AMOVD,	C_SPR,	C_NONE, C_NONE, 	C_REG,		66, 4, 0 },
	{ AMOVD,	C_LR,	C_NONE, C_NONE, 	C_REG,		66, 4, 0 },
	{ AMOVD,	C_CTR,	C_NONE, C_NONE, 	C_REG,		66, 4, 0 },
	{ AMOVD,	C_XER,	C_NONE, C_NONE, 	C_REG,		66, 4, 0 },

	/* 32-bit special registers (gloss over sign-extension or not?) */
	{ AMOVW,	C_REG,	C_NONE, C_NONE, 	C_SPR,		66, 4, 0 },
	{ AMOVW,	C_REG,	C_NONE, C_NONE, 	C_CTR,		66, 4, 0 },
	{ AMOVW,	C_REG,	C_NONE, C_NONE, 	C_XER,		66, 4, 0 },
	{ AMOVW,	C_SPR,	C_NONE, C_NONE, 	C_REG,		66, 4, 0 },
	{ AMOVW,	C_XER,	C_NONE, C_NONE, 	C_REG,		66, 4, 0 },

	{ AMOVWZ,	C_REG,	C_NONE, C_NONE, 	C_SPR,		66, 4, 0 },
	{ AMOVWZ,	C_REG,	C_NONE, C_NONE, 	C_CTR,		66, 4, 0 },
	{ AMOVWZ,	C_REG,	C_NONE, C_NONE, 	C_XER,		66, 4, 0 },
	{ AMOVWZ,	C_SPR,	C_NONE, C_NONE, 	C_REG,		66, 4, 0 },
	{ AMOVWZ,	C_XER,	C_NONE, C_NONE, 	C_REG,		66, 4, 0 },

	{ AMOVFL,	C_FPSCR, C_NONE, C_NONE, 	C_CREG,		73, 4, 0 },
	{ AMOVFL,	C_CREG,	C_NONE, C_NONE, 	C_CREG,		67, 4, 0 },
	{ AMOVW,	C_CREG,	C_NONE,	C_NONE,		C_REG,		68, 4, 0 },
	{ AMOVWZ,	C_CREG,	C_NONE,	C_NONE,		C_REG,		68, 4, 0 },
	{ AMOVFL,	C_REG, C_NONE, C_LCON, C_CREG,		69, 4, 0 },
	{ AMOVFL,	C_REG, C_NONE, C_NONE, C_CREG,		69, 4, 0 },
	{ AMOVW,	C_REG, C_NONE, C_NONE, C_CREG,		69, 4, 0 },
	{ AMOVWZ,	C_REG, C_NONE, C_NONE, C_CREG,		69, 4, 0 },

	{ ACMP,	C_REG,	C_NONE, C_NONE, 	C_REG,	70, 4, 0 },
	{ ACMP,	C_REG,	C_REG, C_NONE, 	C_REG,	70, 4, 0 },
	{ ACMP,	C_REG,	C_NONE, C_NONE,	C_ADDCON,	71, 4, 0 },
	{ ACMP,	C_REG,	C_REG, C_NONE,	C_ADDCON,	71, 4, 0 },

	{ ACMPU,	C_REG,	C_NONE, C_NONE, 	C_REG,	70, 4, 0 },
	{ ACMPU,	C_REG,	C_REG, C_NONE, 	C_REG,	70, 4, 0 },
	{ ACMPU,	C_REG,	C_NONE, C_NONE,	C_ANDCON,	71, 4, 0 },
	{ ACMPU,	C_REG,	C_REG, C_NONE,	C_ANDCON,	71, 4, 0 },

	{ AFCMPO,	C_FREG,	C_NONE, C_NONE, 	C_FREG,	70, 4, 0 },
	{ AFCMPO,	C_FREG,	C_REG, C_NONE, 	C_FREG,	70, 4, 0 },

	{ ATW,		C_LCON,	C_REG, C_NONE, 	C_REG,		60, 4, 0 },
	{ ATW,		C_LCON,	C_REG, C_NONE, 	C_ADDCON,	61, 4, 0 },

	{ ADCBF,	C_ZOREG, C_NONE, C_NONE,  C_NONE,	43, 4, 0 },
	{ ADCBF,	C_ZOREG, C_REG, C_NONE,  C_NONE,	43, 4, 0 },

	{ AECOWX,	C_REG,	C_REG, C_NONE, 	C_ZOREG,	44, 4, 0 },
	{ AECIWX,	C_ZOREG, C_REG, C_NONE,  C_REG,		45, 4, 0 },
	{ AECOWX,	C_REG,	C_NONE, C_NONE, 	C_ZOREG,	44, 4, 0 },
	{ AECIWX,	C_ZOREG, C_NONE, C_NONE,  C_REG,		45, 4, 0 },

	{ AEIEIO,	C_NONE,	C_NONE, C_NONE, 	C_NONE,		46, 4, 0 },
	{ ATLBIE,	C_REG, C_NONE, C_NONE,		C_NONE,		49, 4, 0 },
	{ ATLBIE,	C_SCON, C_NONE, C_NONE,	C_REG,	49, 4, 0 },
	{ ASLBMFEE, C_REG, C_NONE, C_NONE,	C_REG,	55, 4, 0 },
	{ ASLBMTE, C_REG, C_NONE, C_NONE,	C_REG,	55, 4, 0 },

	{ ASTSW,	C_REG,	C_NONE, C_NONE, 	C_ZOREG,	44, 4, 0 },
	{ ASTSW,	C_REG,	C_NONE, C_LCON, 	C_ZOREG,	41, 4, 0 },
	{ ALSW,	C_ZOREG, C_NONE, C_NONE,  C_REG,		45, 4, 0 },
	{ ALSW,	C_ZOREG, C_NONE, C_LCON,  C_REG,		42, 4, 0 },

	{ AUNDEF,	C_NONE, C_NONE, C_NONE, C_NONE, 78, 4, 0 },
	{ AUSEFIELD,	C_ADDR,	C_NONE,	C_NONE, C_NONE,	0, 0, 0 },
	{ APCDATA,	C_LCON,	C_NONE,	C_NONE, C_LCON,	0, 0, 0 },
	{ AFUNCDATA,	C_SCON,	C_NONE,	C_NONE, C_ADDR,	0, 0, 0 },
	{ ANOP,		C_NONE, C_NONE, C_NONE, C_NONE, 0, 0, 0 },

	{ ADUFFZERO,	C_NONE,	C_NONE, C_NONE,	C_LBRA,	11, 4, 0 },  // same as ABR/ABL
	{ ADUFFCOPY,	C_NONE,	C_NONE, C_NONE,	C_LBRA,	11, 4, 0 },  // same as ABR/ABL

	{ AXXX,		C_NONE,	C_NONE, C_NONE, 	C_NONE,		 0, 4, 0 },
};

static int ocmp(const void *, const void *);
static int cmp(int, int);
static void buildop(Link*);
static void prasm(Prog *);
static int isint32(vlong);
static int isuint32(uvlong);
static int aclass(Link*, Addr*);
static Optab* oplook(Link*, Prog*);
static void asmout(Link*, Prog*, Optab*, uint32*);
static vlong vregoff(Link*, Addr*);
static int32 regoff(Link*, Addr*);
static int32 oprrr(Link*, int);
static int32 opirr(Link*, int);
static int32 opload(Link*, int);
static int32 opstore(Link*, int);
static int32 oploadx(Link*, int);
static int32 opstorex(Link*, int);
static int getmask(uchar*, uint32);
static void maskgen(Link*, Prog*, uchar*, uint32);
static int getmask64(uchar*, uvlong);
static void maskgen64(Link*, Prog*, uchar*, uvlong);
static uint32 loadu32(int, vlong);
static void addaddrreloc(Link*, LSym*, uint32*, uint32*);

typedef struct Oprang Oprang;
struct Oprang
{
	Optab*	start;
	Optab*	stop;
};

static Oprang oprange[ALAST];

static uchar	xcmp[C_NCLASS][C_NCLASS];


void
span9(Link *ctxt, LSym *cursym)
{
	Prog *p, *q;
	Optab *o;
	int m, bflag;
	vlong c, otxt;
	uint32 out[6];
	int32 i, j;
	uchar *bp, *cast;

	p = cursym->text;
	if(p == nil || p->link == nil) // handle external functions and ELF section symbols
		return;
	ctxt->cursym = cursym;
	ctxt->autosize = p->to.offset + 8;

	if(oprange[AANDN].start == nil)
 		buildop(ctxt);

	c = 0;	
	p->pc = c;

	for(p = p->link; p != nil; p = p->link) {
		ctxt->curp = p;
		p->pc = c;
		o = oplook(ctxt, p);
		m = o->size;
		if(m == 0) {
			if(p->as != ANOP && p->as != AFUNCDATA && p->as != APCDATA)
				ctxt->diag("zero-width instruction\n%P", p);
			continue;
		}
		c += m;
	}
	cursym->size = c;

	/*
	 * if any procedure is large enough to
	 * generate a large SBRA branch, then
	 * generate extra passes putting branches
	 * around jmps to fix. this is rare.
	 */
	bflag = 1;
	while(bflag) {
		if(ctxt->debugvlog)
			Bprint(ctxt->bso, "%5.2f span1\n", cputime());
		bflag = 0;
		c = 0;
		for(p = cursym->text->link; p != nil; p = p->link) {
			p->pc = c;
			o = oplook(ctxt, p);

			// very large conditional branches
			if((o->type == 16 || o->type == 17) && p->pcond) {
				otxt = p->pcond->pc - c;
				if(otxt < -(1L<<15)+10 || otxt >= (1L<<15)-10) {
					q = emallocz(sizeof(Prog));
					q->link = p->link;
					p->link = q;
					q->as = ABR;
					q->to.type = TYPE_BRANCH;
					q->pcond = p->pcond;
					p->pcond = q;
					q = emallocz(sizeof(Prog));
					q->link = p->link;
					p->link = q;
					q->as = ABR;
					q->to.type = TYPE_BRANCH;
					q->pcond = q->link->link;
					//addnop(p->link);
					//addnop(p);
					bflag = 1;
				}
			}

			m = o->size;
			if(m == 0) {
				if(p->as != ANOP && p->as != AFUNCDATA && p->as != APCDATA)
					ctxt->diag("zero-width instruction\n%P", p);
				continue;
			}
			c += m;
		}
		cursym->size = c;
	}

	c += -c&(FuncAlign-1);
	cursym->size = c;

	/*
	 * lay out the code, emitting code and data relocations.
	 */
	if(ctxt->tlsg == nil)
		ctxt->tlsg = linklookup(ctxt, "runtime.tlsg", 0);

	symgrow(ctxt, cursym, cursym->size);

	bp = cursym->p;
	for(p = cursym->text->link; p != nil; p = p->link) {
		ctxt->pc = p->pc;
		ctxt->curp = p;
		o = oplook(ctxt, p);
		if(o->size > 4*nelem(out))
			sysfatal("out array in span9 is too small, need at least %d for %P", o->size/4, p);
		asmout(ctxt, p, o, out);
		for(i=0; i<o->size/4; i++) {
			cast = (uchar*)&out[i];
			for(j=0; j<4; j++)
				*bp++ = cast[inuxi4[j]];
		}
	}
}

static int
isint32(vlong v)
{
	return (int32)v == v;
}

static int
isuint32(uvlong v)
{
	return (uint32)v == v;
}

static int
aclass(Link *ctxt, Addr *a)
{
	LSym *s;

	switch(a->type) {
	case TYPE_NONE:
		return C_NONE;

	case TYPE_REG:
		if(REG_R0 <= a->reg && a->reg <= REG_R31)
			return C_REG;
		if(REG_F0 <= a->reg && a->reg <= REG_F31)
			return C_FREG;
		if(REG_C0 <= a->reg && a->reg <= REG_C7 || a->reg == REG_CR)
			return C_CREG;
		if(REG_SPR0 <= a->reg && a->reg <= REG_SPR0+1023) {
			switch(a->reg) {
			case REG_LR:
				return C_LR;
			case REG_XER:
				return C_XER;
			case REG_CTR:
				return C_CTR;
			}
			return C_SPR;
		}
		if(REG_DCR0 <= a->reg && a->reg <= REG_DCR0+1023)
			return C_SPR;
		if(a->reg == REG_FPSCR)
			return C_FPSCR;
		if(a->reg == REG_MSR)
			return C_MSR;
		return C_GOK;

	case TYPE_MEM:
		switch(a->name) {
		case NAME_EXTERN:
		case NAME_STATIC:
			if(a->sym == nil)
				break;
			ctxt->instoffset = a->offset;
			if(a->sym != nil) // use relocation
				return C_ADDR;
			return C_LEXT;
		case NAME_AUTO:
			ctxt->instoffset = ctxt->autosize + a->offset;
			if(ctxt->instoffset >= -BIG && ctxt->instoffset < BIG)
				return C_SAUTO;
			return C_LAUTO;
		case NAME_PARAM:
			ctxt->instoffset = ctxt->autosize + a->offset + 8L;
			if(ctxt->instoffset >= -BIG && ctxt->instoffset < BIG)
				return C_SAUTO;
			return C_LAUTO;
		case TYPE_NONE:
			ctxt->instoffset = a->offset;
			if(ctxt->instoffset == 0)
				return C_ZOREG;
			if(ctxt->instoffset >= -BIG && ctxt->instoffset < BIG)
				return C_SOREG;
			return C_LOREG;
		}
		return C_GOK;

	case TYPE_TEXTSIZE:
		return C_TEXTSIZE;

	case TYPE_CONST:
	case TYPE_ADDR:
		switch(a->name) {
		case TYPE_NONE:
			ctxt->instoffset = a->offset;
			if(a->reg != 0) {
				if(-BIG <= ctxt->instoffset && ctxt->instoffset <= BIG)
					return C_SACON;
				if(isint32(ctxt->instoffset))
					return C_LACON;
				return C_DACON;
			}
			goto consize;

		case NAME_EXTERN:
		case NAME_STATIC:
			s = a->sym;
			if(s == nil)
				break;
			if(s->type == SCONST) {
				ctxt->instoffset = s->value + a->offset;
				goto consize;
			}
			ctxt->instoffset = s->value + a->offset;
			/* not sure why this barfs */
			return C_LCON;

		case NAME_AUTO:
			ctxt->instoffset = ctxt->autosize + a->offset;
			if(ctxt->instoffset >= -BIG && ctxt->instoffset < BIG)
				return C_SACON;
			return C_LACON;

		case NAME_PARAM:
			ctxt->instoffset = ctxt->autosize + a->offset + 8L;
			if(ctxt->instoffset >= -BIG && ctxt->instoffset < BIG)
				return C_SACON;
			return C_LACON;
		}
		return C_GOK;

	consize:
		if(ctxt->instoffset >= 0) {
			if(ctxt->instoffset == 0)
				return C_ZCON;
			if(ctxt->instoffset <= 0x7fff)
				return C_SCON;
			if(ctxt->instoffset <= 0xffff)
				return C_ANDCON;
			if((ctxt->instoffset & 0xffff) == 0 && isuint32(ctxt->instoffset))	/* && (instoffset & (1<<31)) == 0) */
				return C_UCON;
			if(isint32(ctxt->instoffset) || isuint32(ctxt->instoffset))
				return C_LCON;
			return C_DCON;
		}
		if(ctxt->instoffset >= -0x8000)
			return C_ADDCON;
		if((ctxt->instoffset & 0xffff) == 0 && isint32(ctxt->instoffset))
			return C_UCON;
		if(isint32(ctxt->instoffset))
			return C_LCON;
		return C_DCON;

	case TYPE_BRANCH:
		return C_SBRA;
	}
	return C_GOK;
}

static void
prasm(Prog *p)
{
	print("%P\n", p);
}

static Optab*
oplook(Link *ctxt, Prog *p)
{
	int a1, a2, a3, a4, r;
	uchar *c1, *c3, *c4;
	Optab *o, *e;

	a1 = p->optab;
	if(a1)
		return optab+(a1-1);
	a1 = p->from.class;
	if(a1 == 0) {
		a1 = aclass(ctxt, &p->from) + 1;
		p->from.class = a1;
	}
	a1--;
	a3 = p->from3.class;
	if(a3 == 0) {
		a3 = aclass(ctxt, &p->from3) + 1;
		p->from3.class = a3;
	}
	a3--;
	a4 = p->to.class;
	if(a4 == 0) {
		a4 = aclass(ctxt, &p->to) + 1;
		p->to.class = a4;
	}
	a4--;
	a2 = C_NONE;
	if(p->reg != 0)
		a2 = C_REG;
//print("oplook %P %d %d %d %d\n", p, a1, a2, a3, a4);
	r = p->as;
	o = oprange[r].start;
	if(o == 0)
		o = oprange[r].stop; /* just generate an error */
	e = oprange[r].stop;
	c1 = xcmp[a1];
	c3 = xcmp[a3];
	c4 = xcmp[a4];
	for(; o<e; o++)
		if(o->a2 == a2)
		if(c1[o->a1])
		if(c3[o->a3])
		if(c4[o->a4]) {
			p->optab = (o-optab)+1;
			return o;
		}
	ctxt->diag("illegal combination %A %^ %^ %^ %^",
		p->as, a1, a2, a3, a4);
	prasm(p);
	if(o == 0)
		o = optab;
	return o;
}

static int
cmp(int a, int b)
{

	if(a == b)
		return 1;
	switch(a) {
	case C_LCON:
		if(b == C_ZCON || b == C_SCON || b == C_UCON || b == C_ADDCON || b == C_ANDCON)
			return 1;
		break;
	case C_ADDCON:
		if(b == C_ZCON || b == C_SCON)
			return 1;
		break;
	case C_ANDCON:
		if(b == C_ZCON || b == C_SCON)
			return 1;
		break;
	case C_SPR:
		if(b == C_LR || b == C_XER || b == C_CTR)
			return 1;
		break;
	case C_UCON:
		if(b == C_ZCON)
			return 1;
		break;
	case C_SCON:
		if(b == C_ZCON)
			return 1;
		break;
	case C_LACON:
		if(b == C_SACON)
			return 1;
		break;
	case C_LBRA:
		if(b == C_SBRA)
			return 1;
		break;
	case C_LEXT:
		if(b == C_SEXT)
			return 1;
		break;
	case C_LAUTO:
		if(b == C_SAUTO)
			return 1;
		break;
	case C_REG:
		if(b == C_ZCON)
			return r0iszero;
		break;
	case C_LOREG:
		if(b == C_ZOREG || b == C_SOREG)
			return 1;
		break;
	case C_SOREG:
		if(b == C_ZOREG)
			return 1;
		break;

	case C_ANY:
		return 1;
	}
	return 0;
}

static int
ocmp(const void *a1, const void *a2)
{
	const Optab *p1, *p2;
	int n;

	p1 = a1;
	p2 = a2;
	n = p1->as - p2->as;
	if(n)
		return n;
	n = p1->a1 - p2->a1;
	if(n)
		return n;
	n = p1->a2 - p2->a2;
	if(n)
		return n;
	n = p1->a3 - p2->a3;
	if(n)
		return n;
	n = p1->a4 - p2->a4;
	if(n)
		return n;
	return 0;
}

static void
buildop(Link *ctxt)
{
	int i, n, r;

	for(i=0; i<C_NCLASS; i++)
		for(n=0; n<C_NCLASS; n++)
			xcmp[i][n] = cmp(n, i);
	for(n=0; optab[n].as != AXXX; n++)
		;
	qsort(optab, n, sizeof(optab[0]), ocmp);
	for(i=0; i<n; i++) {
		r = optab[i].as;
		oprange[r].start = optab+i;
		while(optab[i].as == r)
			i++;
		oprange[r].stop = optab+i;
		i--;
		
		switch(r)
		{
		default:
			ctxt->diag("unknown op in build: %A", r);
			sysfatal("bad code");
		case ADCBF:	/* unary indexed: op (b+a); op (b) */
			oprange[ADCBI] = oprange[r];
			oprange[ADCBST] = oprange[r];
			oprange[ADCBT] = oprange[r];
			oprange[ADCBTST] = oprange[r];
			oprange[ADCBZ] = oprange[r];
			oprange[AICBI] = oprange[r];
			break;
		case AECOWX:	/* indexed store: op s,(b+a); op s,(b) */
			oprange[ASTWCCC] = oprange[r];
			oprange[ASTDCCC] = oprange[r];
			break;
		case AREM:	/* macro */
			oprange[AREMCC] = oprange[r];
			oprange[AREMV] = oprange[r];
			oprange[AREMVCC] = oprange[r];
			break;
		case AREMU:
			oprange[AREMU] = oprange[r];
			oprange[AREMUCC] = oprange[r];
			oprange[AREMUV] = oprange[r];
			oprange[AREMUVCC] = oprange[r];
			break;
		case AREMD:
			oprange[AREMDCC] = oprange[r];
			oprange[AREMDV] = oprange[r];
			oprange[AREMDVCC] = oprange[r];
			break;
		case AREMDU:
			oprange[AREMDU] = oprange[r];
			oprange[AREMDUCC] = oprange[r];
			oprange[AREMDUV] = oprange[r];
			oprange[AREMDUVCC] = oprange[r];
			break;
		case ADIVW:	/* op Rb[,Ra],Rd */
			oprange[AMULHW] = oprange[r];
			oprange[AMULHWCC] = oprange[r];
			oprange[AMULHWU] = oprange[r];
			oprange[AMULHWUCC] = oprange[r];
			oprange[AMULLWCC] = oprange[r];
			oprange[AMULLWVCC] = oprange[r];
			oprange[AMULLWV] = oprange[r];
			oprange[ADIVWCC] = oprange[r];
			oprange[ADIVWV] = oprange[r];
			oprange[ADIVWVCC] = oprange[r];
			oprange[ADIVWU] = oprange[r];
			oprange[ADIVWUCC] = oprange[r];
			oprange[ADIVWUV] = oprange[r];
			oprange[ADIVWUVCC] = oprange[r];
			oprange[AADDCC] = oprange[r];
			oprange[AADDCV] = oprange[r];
			oprange[AADDCVCC] = oprange[r];
			oprange[AADDV] = oprange[r];
			oprange[AADDVCC] = oprange[r];
			oprange[AADDE] = oprange[r];
			oprange[AADDECC] = oprange[r];
			oprange[AADDEV] = oprange[r];
			oprange[AADDEVCC] = oprange[r];
			oprange[ACRAND] = oprange[r];
			oprange[ACRANDN] = oprange[r];
			oprange[ACREQV] = oprange[r];
			oprange[ACRNAND] = oprange[r];
			oprange[ACRNOR] = oprange[r];
			oprange[ACROR] = oprange[r];
			oprange[ACRORN] = oprange[r];
			oprange[ACRXOR] = oprange[r];
			oprange[AMULHD] = oprange[r];
			oprange[AMULHDCC] = oprange[r];
			oprange[AMULHDU] = oprange[r];
			oprange[AMULHDUCC] = oprange[r];
			oprange[AMULLD] = oprange[r];
			oprange[AMULLDCC] = oprange[r];
			oprange[AMULLDVCC] = oprange[r];
			oprange[AMULLDV] = oprange[r];
			oprange[ADIVD] = oprange[r];
			oprange[ADIVDCC] = oprange[r];
			oprange[ADIVDVCC] = oprange[r];
			oprange[ADIVDV] = oprange[r];
			oprange[ADIVDU] = oprange[r];
			oprange[ADIVDUCC] = oprange[r];
			oprange[ADIVDUVCC] = oprange[r];
			oprange[ADIVDUCC] = oprange[r];
			break;
		case AMOVBZ:	/* lbz, stz, rlwm(r/r), lhz, lha, stz, and x variants */
			oprange[AMOVH] = oprange[r];
			oprange[AMOVHZ] = oprange[r];
			break;
		case AMOVBZU:	/* lbz[x]u, stb[x]u, lhz[x]u, lha[x]u, sth[u]x, ld[x]u, std[u]x */
			oprange[AMOVHU] = oprange[r];
			oprange[AMOVHZU] = oprange[r];
			oprange[AMOVWU] = oprange[r];
			oprange[AMOVWZU] = oprange[r];
			oprange[AMOVDU] = oprange[r];
			oprange[AMOVMW] = oprange[r];
			break;
		case AAND:	/* logical op Rb,Rs,Ra; no literal */
			oprange[AANDN] = oprange[r];
			oprange[AANDNCC] = oprange[r];
			oprange[AEQV] = oprange[r];
			oprange[AEQVCC] = oprange[r];
			oprange[ANAND] = oprange[r];
			oprange[ANANDCC] = oprange[r];
			oprange[ANOR] = oprange[r];
			oprange[ANORCC] = oprange[r];
			oprange[AORCC] = oprange[r];
			oprange[AORN] = oprange[r];
			oprange[AORNCC] = oprange[r];
			oprange[AXORCC] = oprange[r];
			break;
		case AADDME:	/* op Ra, Rd */
			oprange[AADDMECC] = oprange[r];
			oprange[AADDMEV] = oprange[r];
			oprange[AADDMEVCC] = oprange[r];
			oprange[AADDZE] = oprange[r];
			oprange[AADDZECC] = oprange[r];
			oprange[AADDZEV] = oprange[r];
			oprange[AADDZEVCC] = oprange[r];
			oprange[ASUBME] = oprange[r];
			oprange[ASUBMECC] = oprange[r];
			oprange[ASUBMEV] = oprange[r];
			oprange[ASUBMEVCC] = oprange[r];
			oprange[ASUBZE] = oprange[r];
			oprange[ASUBZECC] = oprange[r];
			oprange[ASUBZEV] = oprange[r];
			oprange[ASUBZEVCC] = oprange[r];
			break;
		case AADDC:
			oprange[AADDCCC] = oprange[r];
			break;
		case ABEQ:
			oprange[ABGE] = oprange[r];
			oprange[ABGT] = oprange[r];
			oprange[ABLE] = oprange[r];
			oprange[ABLT] = oprange[r];
			oprange[ABNE] = oprange[r];
			oprange[ABVC] = oprange[r];
			oprange[ABVS] = oprange[r];
			break;
		case ABR:
			oprange[ABL] = oprange[r];
			break;
		case ABC:
			oprange[ABCL] = oprange[r];
			break;
		case AEXTSB:	/* op Rs, Ra */
			oprange[AEXTSBCC] = oprange[r];
			oprange[AEXTSH] = oprange[r];
			oprange[AEXTSHCC] = oprange[r];
			oprange[ACNTLZW] = oprange[r];
			oprange[ACNTLZWCC] = oprange[r];
			oprange[ACNTLZD] = oprange[r];
			oprange[AEXTSW] = oprange[r];
			oprange[AEXTSWCC] = oprange[r];
			oprange[ACNTLZDCC] = oprange[r];
			break;
		case AFABS:	/* fop [s,]d */
			oprange[AFABSCC] = oprange[r];
			oprange[AFNABS] = oprange[r];
			oprange[AFNABSCC] = oprange[r];
			oprange[AFNEG] = oprange[r];
			oprange[AFNEGCC] = oprange[r];
			oprange[AFRSP] = oprange[r];
			oprange[AFRSPCC] = oprange[r];
			oprange[AFCTIW] = oprange[r];
			oprange[AFCTIWCC] = oprange[r];
			oprange[AFCTIWZ] = oprange[r];
			oprange[AFCTIWZCC] = oprange[r];
			oprange[AFCTID] = oprange[r];
			oprange[AFCTIDCC] = oprange[r];
			oprange[AFCTIDZ] = oprange[r];
			oprange[AFCTIDZCC] = oprange[r];
			oprange[AFCFID] = oprange[r];
			oprange[AFCFIDCC] = oprange[r];
			oprange[AFRES] = oprange[r];
			oprange[AFRESCC] = oprange[r];
			oprange[AFRSQRTE] = oprange[r];
			oprange[AFRSQRTECC] = oprange[r];
			oprange[AFSQRT] = oprange[r];
			oprange[AFSQRTCC] = oprange[r];
			oprange[AFSQRTS] = oprange[r];
			oprange[AFSQRTSCC] = oprange[r];
			break;
		case AFADD:
			oprange[AFADDS] = oprange[r];
			oprange[AFADDCC] = oprange[r];
			oprange[AFADDSCC] = oprange[r];
			oprange[AFDIV] = oprange[r];
			oprange[AFDIVS] = oprange[r];
			oprange[AFDIVCC] = oprange[r];
			oprange[AFDIVSCC] = oprange[r];
			oprange[AFSUB] = oprange[r];
			oprange[AFSUBS] = oprange[r];
			oprange[AFSUBCC] = oprange[r];
			oprange[AFSUBSCC] = oprange[r];
			break;
		case AFMADD:
			oprange[AFMADDCC] = oprange[r];
			oprange[AFMADDS] = oprange[r];
			oprange[AFMADDSCC] = oprange[r];
			oprange[AFMSUB] = oprange[r];
			oprange[AFMSUBCC] = oprange[r];
			oprange[AFMSUBS] = oprange[r];
			oprange[AFMSUBSCC] = oprange[r];
			oprange[AFNMADD] = oprange[r];
			oprange[AFNMADDCC] = oprange[r];
			oprange[AFNMADDS] = oprange[r];
			oprange[AFNMADDSCC] = oprange[r];
			oprange[AFNMSUB] = oprange[r];
			oprange[AFNMSUBCC] = oprange[r];
			oprange[AFNMSUBS] = oprange[r];
			oprange[AFNMSUBSCC] = oprange[r];
			oprange[AFSEL] = oprange[r];
			oprange[AFSELCC] = oprange[r];
			break;
		case AFMUL:
			oprange[AFMULS] = oprange[r];
			oprange[AFMULCC] = oprange[r];
			oprange[AFMULSCC] = oprange[r];
			break;
		case AFCMPO:
			oprange[AFCMPU] = oprange[r];
			break;
		case AMTFSB0:
			oprange[AMTFSB0CC] = oprange[r];
			oprange[AMTFSB1] = oprange[r];
			oprange[AMTFSB1CC] = oprange[r];
			break;
		case ANEG:	/* op [Ra,] Rd */
			oprange[ANEGCC] = oprange[r];
			oprange[ANEGV] = oprange[r];
			oprange[ANEGVCC] = oprange[r];
			break;
		case AOR:	/* or/xor Rb,Rs,Ra; ori/xori $uimm,Rs,Ra; oris/xoris $uimm,Rs,Ra */
			oprange[AXOR] = oprange[r];
			break;
		case ASLW:
			oprange[ASLWCC] = oprange[r];
			oprange[ASRW] = oprange[r];
			oprange[ASRWCC] = oprange[r];
			break;
		case ASLD:
			oprange[ASLDCC] = oprange[r];
			oprange[ASRD] = oprange[r];
			oprange[ASRDCC] = oprange[r];
			break;
		case ASRAW:	/* sraw Rb,Rs,Ra; srawi sh,Rs,Ra */
			oprange[ASRAWCC] = oprange[r];
			break;
		case ASRAD:	/* sraw Rb,Rs,Ra; srawi sh,Rs,Ra */
			oprange[ASRADCC] = oprange[r];
			break;
		case ASUB:	/* SUB Ra,Rb,Rd => subf Rd,ra,rb */
			oprange[ASUB] = oprange[r];
			oprange[ASUBCC] = oprange[r];
			oprange[ASUBV] = oprange[r];
			oprange[ASUBVCC] = oprange[r];
			oprange[ASUBCCC] = oprange[r];
			oprange[ASUBCV] = oprange[r];
			oprange[ASUBCVCC] = oprange[r];
			oprange[ASUBE] = oprange[r];
			oprange[ASUBECC] = oprange[r];
			oprange[ASUBEV] = oprange[r];
			oprange[ASUBEVCC] = oprange[r];
			break;
		case ASYNC:
			oprange[AISYNC] = oprange[r];
			oprange[APTESYNC] = oprange[r];
			oprange[ATLBSYNC] = oprange[r];
			break;
		case ARLWMI:
			oprange[ARLWMICC] = oprange[r];
			oprange[ARLWNM] = oprange[r];
			oprange[ARLWNMCC] = oprange[r];
			break;
		case ARLDMI:
			oprange[ARLDMICC] = oprange[r];
			break;
		case ARLDC:
			oprange[ARLDCCC] = oprange[r];
			break;
		case ARLDCL:
			oprange[ARLDCR] = oprange[r];
			oprange[ARLDCLCC] = oprange[r];
			oprange[ARLDCRCC] = oprange[r];
			break;
		case AFMOVD:
			oprange[AFMOVDCC] = oprange[r];
			oprange[AFMOVDU] = oprange[r];
			oprange[AFMOVS] = oprange[r];
			oprange[AFMOVSU] = oprange[r];
			break;
		case AECIWX:
			oprange[ALWAR] = oprange[r];
			oprange[ALDAR] = oprange[r];
			break;
		case ASYSCALL:	/* just the op; flow of control */
			oprange[ARFI] = oprange[r];
			oprange[ARFCI] = oprange[r];
			oprange[ARFID] = oprange[r];
			oprange[AHRFID] = oprange[r];
			break;
		case AMOVHBR:
			oprange[AMOVWBR] = oprange[r];
			break;
		case ASLBMFEE:
			oprange[ASLBMFEV] = oprange[r];
			break;
		case ATW:
			oprange[ATD] = oprange[r];
			break;
		case ATLBIE:
			oprange[ASLBIE] = oprange[r];
			oprange[ATLBIEL] = oprange[r];
			break;
		case AEIEIO:
			oprange[ASLBIA] = oprange[r];
			break;
		case ACMP:
			oprange[ACMPW] = oprange[r];
			break;
		case ACMPU:
			oprange[ACMPWU] = oprange[r];
			break;
		case AADD:
		case AANDCC:	/* and. Rb,Rs,Ra; andi. $uimm,Rs,Ra; andis. $uimm,Rs,Ra */
		case ALSW:
		case AMOVW:	/* load/store/move word with sign extension; special 32-bit move; move 32-bit literals */
		case AMOVWZ:	/* load/store/move word with zero extension; move 32-bit literals  */
		case AMOVD:	/* load/store/move 64-bit values, including 32-bit literals with/without sign-extension */
		case AMOVB:	/* macro: move byte with sign extension */
		case AMOVBU:	/* macro: move byte with sign extension & update */
		case AMOVFL:
		case AMULLW:	/* op $s[,r2],r3; op r1[,r2],r3; no cc/v */
		case ASUBC:	/* op r1,$s,r3; op r1[,r2],r3 */
		case ASTSW:
		case ASLBMTE:
		case AWORD:
		case ADWORD:
		case ANOP:
		case ATEXT:
		case AUNDEF:
		case AUSEFIELD:
		case AFUNCDATA:
		case APCDATA:
		case ADUFFZERO:
		case ADUFFCOPY:
			break;
		}
	}
}

uint32
OPVCC(uint32 o, uint32 xo, uint32 oe, uint32 rc)
{
	return o<<26 | xo<<1 | oe<<10 | rc&1;
}

uint32
OPCC(uint32 o, uint32 xo, uint32 rc)
{
	return OPVCC(o, xo, 0, rc);
}

uint32
OP(uint32 o, uint32 xo)
{
	return OPVCC(o, xo, 0, 0);
}

/* the order is dest, a/s, b/imm for both arithmetic and logical operations */
uint32
AOP_RRR(uint32 op, uint32 d, uint32 a, uint32 b)
{
	return op | (d&31)<<21 | (a&31)<<16 | (b&31)<<11;
}

uint32
AOP_IRR(uint32 op, uint32 d, uint32 a, uint32 simm)
{
	return op | (d&31)<<21 | (a&31)<<16 | (simm&0xFFFF);
}

uint32
LOP_RRR(uint32 op, uint32 a, uint32 s, uint32 b)
{
	return op | (s&31)<<21 | (a&31)<<16 | (b&31)<<11;
}

uint32
LOP_IRR(uint32 op, uint32 a, uint32 s, uint32 uimm)
{
	return op | (s&31)<<21 | (a&31)<<16 | (uimm&0xFFFF);
}

uint32
OP_BR(uint32 op, uint32 li, uint32 aa)
{
	return op | li&0x03FFFFFC | aa<<1;
}

uint32
OP_BC(uint32 op, uint32 bo, uint32 bi, uint32 bd, uint32 aa)
{
	return op | (bo&0x1F)<<21 | (bi&0x1F)<<16 | bd&0xFFFC | aa<<1;
}

uint32
OP_BCR(uint32 op, uint32 bo, uint32 bi)
{
	return op | (bo&0x1F)<<21 | (bi&0x1F)<<16;
}

uint32
OP_RLW(uint32 op, uint32 a, uint32 s, uint32 sh, uint32 mb, uint32 me)
{
	return op | (s&31)<<21 | (a&31)<<16 | (sh&31)<<11 | (mb&31)<<6 | (me&31)<<1;
}

enum {
	/* each rhs is OPVCC(_, _, _, _) */
	OP_ADD =	31<<26 | 266<<1 | 0<<10 | 0,
	OP_ADDI =	14<<26 | 0<<1 | 0<<10 | 0,
	OP_ADDIS = 15<<26 | 0<<1 | 0<<10 | 0,
	OP_ANDI =	28<<26 | 0<<1 | 0<<10 | 0,
	OP_EXTSB =	31<<26 | 954<<1 | 0<<10 | 0,
	OP_EXTSH = 31<<26 | 922<<1 | 0<<10 | 0,
	OP_EXTSW = 31<<26 | 986<<1 | 0<<10 | 0,
	OP_MCRF =	19<<26 | 0<<1 | 0<<10 | 0,
	OP_MCRFS = 63<<26 | 64<<1 | 0<<10 | 0,
	OP_MCRXR = 31<<26 | 512<<1 | 0<<10 | 0,
	OP_MFCR =	31<<26 | 19<<1 | 0<<10 | 0,
	OP_MFFS =	63<<26 | 583<<1 | 0<<10 | 0,
	OP_MFMSR = 31<<26 | 83<<1 | 0<<10 | 0,
	OP_MFSPR = 31<<26 | 339<<1 | 0<<10 | 0,
	OP_MFSR =	31<<26 | 595<<1 | 0<<10 | 0,
	OP_MFSRIN =	31<<26 | 659<<1 | 0<<10 | 0,
	OP_MTCRF = 31<<26 | 144<<1 | 0<<10 | 0,
	OP_MTFSF = 63<<26 | 711<<1 | 0<<10 | 0,
	OP_MTFSFI = 63<<26 | 134<<1 | 0<<10 | 0,
	OP_MTMSR = 31<<26 | 146<<1 | 0<<10 | 0,
	OP_MTMSRD = 31<<26 | 178<<1 | 0<<10 | 0,
	OP_MTSPR = 31<<26 | 467<<1 | 0<<10 | 0,
	OP_MTSR =	31<<26 | 210<<1 | 0<<10 | 0,
	OP_MTSRIN =	31<<26 | 242<<1 | 0<<10 | 0,
	OP_MULLW = 31<<26 | 235<<1 | 0<<10 | 0,
	OP_MULLD = 31<<26 | 233<<1 | 0<<10 | 0,
	OP_OR =	31<<26 | 444<<1 | 0<<10 | 0,
	OP_ORI =	24<<26 | 0<<1 | 0<<10 | 0,
	OP_ORIS =	25<<26 | 0<<1 | 0<<10 | 0,
	OP_RLWINM =	21<<26 | 0<<1 | 0<<10 | 0,
	OP_SUBF =	31<<26 | 40<<1 | 0<<10 | 0,
	OP_RLDIC =	30<<26 | 4<<1 | 0<<10 | 0,
	OP_RLDICR =	30<<26 | 2<<1 | 0<<10 | 0,
	OP_RLDICL =	30<<26 | 0<<1 | 0<<10 | 0,
};

int
oclass(Addr *a)
{
	return a->class - 1;
}

// add R_ADDRPOWER relocation to symbol s for the two instructions o1 and o2.
static void
addaddrreloc(Link *ctxt, LSym *s, uint32 *o1, uint32 *o2)
{
	Reloc *rel;

	rel = addrel(ctxt->cursym);
	rel->off = ctxt->pc;
	rel->siz = 8;
	rel->sym = s;
	rel->add = ((uvlong)*o1<<32) | (uint32)*o2;
	rel->type = R_ADDRPOWER;
}

/*
 * 32-bit masks
 */
static int
getmask(uchar *m, uint32 v)
{
	int i;

	m[0] = m[1] = 0;
	if(v != ~(uint32)0 && v & (1<<31) && v & 1){	/* MB > ME */
		if(getmask(m, ~v)){
			i = m[0]; m[0] = m[1]+1; m[1] = i-1;
			return 1;
		}
		return 0;
	}
	for(i=0; i<32; i++)
		if(v & (1<<(31-i))){
			m[0] = i;
			do {
				m[1] = i;
			} while(++i<32 && (v & (1<<(31-i))) != 0);
			for(; i<32; i++)
				if(v & (1<<(31-i)))
					return 0;
			return 1;
		}
	return 0;
}

static void
maskgen(Link *ctxt, Prog *p, uchar *m, uint32 v)
{
	if(!getmask(m, v))
		ctxt->diag("cannot generate mask #%lux\n%P", v, p);
}

/*
 * 64-bit masks (rldic etc)
 */
static int
getmask64(uchar *m, uvlong v)
{
	int i;

	m[0] = m[1] = 0;
	for(i=0; i<64; i++)
		if(v & ((uvlong)1<<(63-i))){
			m[0] = i;
			do {
				m[1] = i;
			} while(++i<64 && (v & ((uvlong)1<<(63-i))) != 0);
			for(; i<64; i++)
				if(v & ((uvlong)1<<(63-i)))
					return 0;
			return 1;
		}
	return 0;
}

static void
maskgen64(Link *ctxt, Prog *p, uchar *m, uvlong v)
{
	if(!getmask64(m, v))
		ctxt->diag("cannot generate mask #%llux\n%P", v, p);
}

static uint32
loadu32(int r, vlong d)
{
	int32 v;

	v = d>>16;
	if(isuint32(d))
		return LOP_IRR(OP_ORIS, r, REGZERO, v);
	return AOP_IRR(OP_ADDIS, r, REGZERO, v);
}

static uint16
high16adjusted(int32 d)
{
	if(d & 0x8000)
		return (d>>16) + 1;
	return d>>16;
}

static void
asmout(Link *ctxt, Prog *p, Optab *o, uint32 *out)
{
	uint32 o1, o2, o3, o4, o5;
	int32 v, t;
	vlong d;
	int r, a;
	uchar mask[2];
	Reloc *rel;

	o1 = 0;
	o2 = 0;
	o3 = 0;
	o4 = 0;
	o5 = 0;

//print("%P => case %d\n", p, o->type);
	switch(o->type) {
	default:
		ctxt->diag("unknown type %d", o->type);
		prasm(p);
		break;

	case 0:		/* pseudo ops */
		break;

	case 1:		/* mov r1,r2 ==> OR Rs,Rs,Ra */
		if(p->to.reg == REGZERO && p->from.type == TYPE_CONST) {
			v = regoff(ctxt, &p->from);
			if(r0iszero && v != 0) {
				//nerrors--;
				ctxt->diag("literal operation on R0\n%P", p);
			}
			o1 = LOP_IRR(OP_ADDI, REGZERO, REGZERO, v);
			break;
		}
		o1 = LOP_RRR(OP_OR, p->to.reg, p->from.reg, p->from.reg);
		break;

	case 2:		/* int/cr/fp op Rb,[Ra],Rd */
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		o1 = AOP_RRR(oprrr(ctxt, p->as), p->to.reg, r, p->from.reg);
		break;

	case 3:		/* mov $soreg/addcon/ucon, r ==> addis/addi $i,reg',r */
		d = vregoff(ctxt, &p->from);
		v = d;
		r = p->from.reg;
		if(r == 0)
			r = o->param;
		if(r0iszero && p->to.reg == 0 && (r != 0 || v != 0))
			ctxt->diag("literal operation on R0\n%P", p);
		a = OP_ADDI;
		if(o->a1 == C_UCON) {
			if((d&0xffff) != 0)
				sysfatal("invalid handling of %P", p);
			v >>= 16;
			if(r == REGZERO && isuint32(d)){
				o1 = LOP_IRR(OP_ORIS, p->to.reg, REGZERO, v);
				break;
			}
			a = OP_ADDIS;
		} else {
			if((int16)d != d)
				sysfatal("invalid handling of %P", p);
		}
		o1 = AOP_IRR(a, p->to.reg, r, v);
		break;

	case 4:		/* add/mul $scon,[r1],r2 */
		v = regoff(ctxt, &p->from);
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		if(r0iszero && p->to.reg == 0)
			ctxt->diag("literal operation on R0\n%P", p);
		if((int16)v != v)
			sysfatal("mishandled instruction %P", p);
		o1 = AOP_IRR(opirr(ctxt, p->as), p->to.reg, r, v);
		break;

	case 5:		/* syscall */
		o1 = oprrr(ctxt, p->as);
		break;

	case 6:		/* logical op Rb,[Rs,]Ra; no literal */
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		o1 = LOP_RRR(oprrr(ctxt, p->as), p->to.reg, r, p->from.reg);
		break;

	case 7:		/* mov r, soreg ==> stw o(r) */
		r = p->to.reg;
		if(r == 0)
			r = o->param;
		v = regoff(ctxt, &p->to);
		if(p->to.type == TYPE_MEM && p->reg != 0) {
			if(v)
				ctxt->diag("illegal indexed instruction\n%P", p);
			o1 = AOP_RRR(opstorex(ctxt, p->as), p->from.reg, p->reg, r);
		} else {
			if((int16)v != v)
				sysfatal("mishandled instruction %P", p);	
			o1 = AOP_IRR(opstore(ctxt, p->as), p->from.reg, r, v);
		}
		break;

	case 8:		/* mov soreg, r ==> lbz/lhz/lwz o(r) */
		r = p->from.reg;
		if(r == 0)
			r = o->param;
		v = regoff(ctxt, &p->from);
		if(p->from.type == TYPE_MEM && p->reg != 0) {
			if(v)
				ctxt->diag("illegal indexed instruction\n%P", p);
			o1 = AOP_RRR(oploadx(ctxt, p->as), p->to.reg, p->reg, r);
		} else {
			if((int16)v != v)
				sysfatal("mishandled instruction %P", p);
			o1 = AOP_IRR(opload(ctxt, p->as), p->to.reg, r, v);
		}
		break;

	case 9:		/* movb soreg, r ==> lbz o(r),r2; extsb r2,r2 */
		r = p->from.reg;
		if(r == 0)
			r = o->param;
		v = regoff(ctxt, &p->from);
		if(p->from.type == TYPE_MEM && p->reg != 0) {
			if(v)
				ctxt->diag("illegal indexed instruction\n%P", p);
			o1 = AOP_RRR(oploadx(ctxt, p->as), p->to.reg, p->reg, r);
		} else
			o1 = AOP_IRR(opload(ctxt, p->as), p->to.reg, r, v);
		o2 = LOP_RRR(OP_EXTSB, p->to.reg, p->to.reg, 0);
		break;

	case 10:		/* sub Ra,[Rb],Rd => subf Rd,Ra,Rb */
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		o1 = AOP_RRR(oprrr(ctxt, p->as), p->to.reg, p->from.reg, r);
		break;

	case 11:	/* br/bl lbra */
		v = 0;
		if(p->pcond) {
			v = p->pcond->pc - p->pc;
			if(v & 03) {
				ctxt->diag("odd branch target address\n%P", p);
				v &= ~03;
			}
			if(v < -(1L<<25) || v >= (1L<<24))
				ctxt->diag("branch too far\n%P", p);
		}
		o1 = OP_BR(opirr(ctxt, p->as), v, 0);
		if(p->to.sym != nil) {
			rel = addrel(ctxt->cursym);
			rel->off = ctxt->pc;
			rel->siz = 4;
			rel->sym = p->to.sym;
			v += p->to.offset;
			if(v & 03) {
				ctxt->diag("odd branch target address\n%P", p);
				v &= ~03;
			}
			rel->add = v;
			rel->type = R_CALLPOWER;
		}
		break;

	case 12:	/* movb r,r (extsb); movw r,r (extsw) */
		if(p->to.reg == REGZERO && p->from.type == TYPE_CONST) {
			v = regoff(ctxt, &p->from);
			if(r0iszero && v != 0) {
				ctxt->diag("literal operation on R0\n%P", p);
			}
			o1 = LOP_IRR(OP_ADDI, REGZERO, REGZERO, v);
			break;
		}
		if(p->as == AMOVW)
			o1 = LOP_RRR(OP_EXTSW, p->to.reg, p->from.reg, 0);
		else
			o1 = LOP_RRR(OP_EXTSB, p->to.reg, p->from.reg, 0);
		break;

	case 13:	/* mov[bhw]z r,r; uses rlwinm not andi. to avoid changing CC */
		if(p->as == AMOVBZ)
			o1 = OP_RLW(OP_RLWINM, p->to.reg, p->from.reg, 0, 24, 31);
		else if(p->as == AMOVH)
			o1 = LOP_RRR(OP_EXTSH, p->to.reg, p->from.reg, 0);
		else if(p->as == AMOVHZ)
			o1 = OP_RLW(OP_RLWINM, p->to.reg, p->from.reg, 0, 16, 31);
		else if(p->as == AMOVWZ)
			o1 = OP_RLW(OP_RLDIC, p->to.reg, p->from.reg, 0, 0, 0) | (1<<5);	/* MB=32 */
		else
			ctxt->diag("internal: bad mov[bhw]z\n%P", p);
		break;

	case 14:	/* rldc[lr] Rb,Rs,$mask,Ra -- left, right give different masks */
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		d = vregoff(ctxt, &p->from3);
		maskgen64(ctxt, p, mask, d);
		switch(p->as){
		case ARLDCL: case ARLDCLCC:
			a = mask[0];	/* MB */
			if(mask[1] != 63)
				ctxt->diag("invalid mask for rotate: %llux (end != bit 63)\n%P", d, p);
			break;
		case ARLDCR: case ARLDCRCC:
			a = mask[1];	/* ME */
			if(mask[0] != 0)
				ctxt->diag("invalid mask for rotate: %llux (start != 0)\n%P", d, p);
			break;
		default:
			ctxt->diag("unexpected op in rldc case\n%P", p);
			a = 0;
		}
		o1 = LOP_RRR(oprrr(ctxt, p->as), p->to.reg, r, p->from.reg);
		o1 |= (a&31L)<<6;
		if(a & 0x20)
			o1 |= 1<<5;	/* mb[5] is top bit */
		break;

	case 17:	/* bc bo,bi,lbra (same for now) */
	case 16:	/* bc bo,bi,sbra */
		a = 0;
		if(p->from.type == TYPE_CONST)
			a = regoff(ctxt, &p->from);
		r = p->reg;
		if(r == 0)
			r = 0;
		v = 0;
		if(p->pcond)
			v = p->pcond->pc - p->pc;
		if(v & 03) {
			ctxt->diag("odd branch target address\n%P", p);
			v &= ~03;
		}
		if(v < -(1L<<16) || v >= (1L<<15))
			ctxt->diag("branch too far\n%P", p);
		o1 = OP_BC(opirr(ctxt, p->as), a, r, v, 0);
		break;

	case 15:	/* br/bl (r) => mov r,lr; br/bl (lr) */
		if(p->as == ABC || p->as == ABCL)
			v = regoff(ctxt, &p->to)&31L;
		else
			v = 20;	/* unconditional */
		r = p->reg;
		if(r == 0)
			r = 0;
		o1 = AOP_RRR(OP_MTSPR, p->to.reg, 0, 0) | ((REG_LR&0x1f)<<16) | (((REG_LR>>5)&0x1f)<<11);
		o2 = OPVCC(19, 16, 0, 0);
		if(p->as == ABL || p->as == ABCL)
			o2 |= 1;
		o2 = OP_BCR(o2, v, r);
		break;

	case 18:	/* br/bl (lr/ctr); bc/bcl bo,bi,(lr/ctr) */
		if(p->as == ABC || p->as == ABCL)
			v = regoff(ctxt, &p->from)&31L;
		else
			v = 20;	/* unconditional */
		r = p->reg;
		if(r == 0)
			r = 0;
		switch(oclass(&p->to)) {
		case C_CTR:
			o1 = OPVCC(19, 528, 0, 0);
			break;
		case C_LR:
			o1 = OPVCC(19, 16, 0, 0);
			break;
		default:
			ctxt->diag("bad optab entry (18): %d\n%P", p->to.class, p);
			v = 0;
		}
		if(p->as == ABL || p->as == ABCL)
			o1 |= 1;
		o1 = OP_BCR(o1, v, r);
		break;

	case 19:	/* mov $lcon,r ==> cau+or */
		d = vregoff(ctxt, &p->from);
		if(p->from.sym == nil) {
			o1 = loadu32(p->to.reg, d);
			o2 = LOP_IRR(OP_ORI, p->to.reg, p->to.reg, (int32)d);
		} else {
			o1 = AOP_IRR(OP_ADDIS, REGTMP, REGZERO, high16adjusted(d));
			o2 = AOP_IRR(OP_ADDI, p->to.reg, REGTMP, d);
			addaddrreloc(ctxt, p->from.sym, &o1, &o2);
		}
		//if(dlm) reloc(&p->from, p->pc, 0);
		break;

	case 20:	/* add $ucon,,r */
		v = regoff(ctxt, &p->from);
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		if(p->as == AADD && (!r0iszero && p->reg == 0 || r0iszero && p->to.reg == 0))
			ctxt->diag("literal operation on R0\n%P", p);
		o1 = AOP_IRR(opirr(ctxt, p->as+ALAST), p->to.reg, r, v>>16);
		break;

	case 22:	/* add $lcon,r1,r2 ==> cau+or+add */	/* could do add/sub more efficiently */
		if(p->to.reg == REGTMP || p->reg == REGTMP)
			ctxt->diag("cant synthesize large constant\n%P", p);
		d = vregoff(ctxt, &p->from);
		o1 = loadu32(REGTMP, d);
		o2 = LOP_IRR(OP_ORI, REGTMP, REGTMP, (int32)d);
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		o3 = AOP_RRR(oprrr(ctxt, p->as), p->to.reg, REGTMP, r);
		if(p->from.sym != nil)
			ctxt->diag("%P is not supported", p);
		//if(dlm) reloc(&p->from, p->pc, 0);
		break;

	case 23:	/* and $lcon,r1,r2 ==> cau+or+and */	/* masks could be done using rlnm etc. */
		if(p->to.reg == REGTMP || p->reg == REGTMP)
			ctxt->diag("cant synthesize large constant\n%P", p);
		d = vregoff(ctxt, &p->from);
		o1 = loadu32(REGTMP, d);
		o2 = LOP_IRR(OP_ORI, REGTMP, REGTMP, (int32)d);
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		o3 = LOP_RRR(oprrr(ctxt, p->as), p->to.reg, REGTMP, r);
		if(p->from.sym != nil)
			ctxt->diag("%P is not supported", p);
		//if(dlm) reloc(&p->from, p->pc, 0);
		break;
/*24*/

	case 25:	/* sld[.] $sh,rS,rA -> rldicr[.] $sh,rS,mask(0,63-sh),rA; srd[.] -> rldicl */
		v = regoff(ctxt, &p->from);
		if(v < 0)
			v = 0;
		else if(v > 63)
			v = 63;
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		switch(p->as){
		case ASLD: case ASLDCC:
			a = 63-v;
			o1 = OP_RLDICR;
			break;
		case ASRD: case ASRDCC:
			a = v;
			v = 64-v;
			o1 = OP_RLDICL;
			break;
		default:
			ctxt->diag("unexpected op in sldi case\n%P", p);
			a = 0;
			o1 = 0;
		}
		o1 = AOP_RRR(o1, r, p->to.reg, (v&0x1F));
		o1 |= (a&31L)<<6;
		if(v & 0x20)
			o1 |= 1<<1;
		if(a & 0x20)
			o1 |= 1<<5;	/* mb[5] is top bit */
		if(p->as == ASLDCC || p->as == ASRDCC)
			o1 |= 1;	/* Rc */
		break;

	case 26:	/* mov $lsext/auto/oreg,,r2 ==> addis+addi */
		if(p->to.reg == REGTMP)
			ctxt->diag("can't synthesize large constant\n%P", p);
		v = regoff(ctxt, &p->from);
		r = p->from.reg;
		if(r == 0)
			r = o->param;
		o1 = AOP_IRR(OP_ADDIS, REGTMP, r, high16adjusted(v));
		o2 = AOP_IRR(OP_ADDI, p->to.reg, REGTMP, v);
		break;

	case 27:		/* subc ra,$simm,rd => subfic rd,ra,$simm */
		v = regoff(ctxt, &p->from3);
		r = p->from.reg;
		o1 = AOP_IRR(opirr(ctxt, p->as), p->to.reg, r, v);
		break;

	case 28:	/* subc r1,$lcon,r2 ==> cau+or+subfc */
		if(p->to.reg == REGTMP || p->from.reg == REGTMP)
			ctxt->diag("can't synthesize large constant\n%P", p);
		v = regoff(ctxt, &p->from3);
		o1 = AOP_IRR(OP_ADDIS, REGTMP, REGZERO, v>>16);
		o2 = LOP_IRR(OP_ORI, REGTMP, REGTMP, v);
		o3 = AOP_RRR(oprrr(ctxt, p->as), p->to.reg, p->from.reg, REGTMP);
		if(p->from.sym != nil)
			ctxt->diag("%P is not supported", p);
		//if(dlm) reloc(&p->from3, p->pc, 0);
		break;

	case 29:	/* rldic[lr]? $sh,s,$mask,a -- left, right, plain give different masks */
		v = regoff(ctxt, &p->from);
		d = vregoff(ctxt, &p->from3);
		maskgen64(ctxt, p, mask, d);
		switch(p->as){
		case ARLDC: case ARLDCCC:
			a = mask[0];	/* MB */
			if(mask[1] != (63-v))
				ctxt->diag("invalid mask for shift: %llux (shift %ld)\n%P", d, v, p);
			break;
		case ARLDCL: case ARLDCLCC:
			a = mask[0];	/* MB */
			if(mask[1] != 63)
				ctxt->diag("invalid mask for shift: %llux (shift %ld)\n%P", d, v, p);
			break;
		case ARLDCR: case ARLDCRCC:
			a = mask[1];	/* ME */
			if(mask[0] != 0)
				ctxt->diag("invalid mask for shift: %llux (shift %ld)\n%P", d, v, p);
			break;
		default:
			ctxt->diag("unexpected op in rldic case\n%P", p);
			a = 0;
		}
		o1 = AOP_RRR(opirr(ctxt, p->as), p->reg, p->to.reg, (v&0x1F));
		o1 |= (a&31L)<<6;
		if(v & 0x20)
			o1 |= 1<<1;
		if(a & 0x20)
			o1 |= 1<<5;	/* mb[5] is top bit */
		break;

	case 30:	/* rldimi $sh,s,$mask,a */
		v = regoff(ctxt, &p->from);
		d = vregoff(ctxt, &p->from3);
		maskgen64(ctxt, p, mask, d);
		if(mask[1] != (63-v))
			ctxt->diag("invalid mask for shift: %llux (shift %ld)\n%P", d, v, p);
		o1 = AOP_RRR(opirr(ctxt, p->as), p->reg, p->to.reg, (v&0x1F));
		o1 |= (mask[0]&31L)<<6;
		if(v & 0x20)
			o1 |= 1<<1;
		if(mask[0] & 0x20)
			o1 |= 1<<5;	/* mb[5] is top bit */
		break;

	case 31:	/* dword */
		d = vregoff(ctxt, &p->from);
		if(ctxt->arch->endian == BigEndian) {
			o1 = d>>32;
			o2 = d;
		} else {
			o1 = d;
			o2 = d>>32;
		}
		if(p->from.sym != nil) {
			rel = addrel(ctxt->cursym);
			rel->off = ctxt->pc;
			rel->siz = 8;
			rel->sym = p->from.sym;
			rel->add = p->from.offset;
			rel->type = R_ADDR;
			o1 = o2 = 0;
		}
		break;

	case 32:	/* fmul frc,fra,frd */
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		o1 = AOP_RRR(oprrr(ctxt, p->as), p->to.reg, r, 0)|((p->from.reg&31L)<<6);
		break;

	case 33:	/* fabs [frb,]frd; fmr. frb,frd */
		r = p->from.reg;
		if(oclass(&p->from) == C_NONE)
			r = p->to.reg;
		o1 = AOP_RRR(oprrr(ctxt, p->as), p->to.reg, 0, r);
		break;

	case 34:	/* FMADDx fra,frb,frc,frd (d=a*b+c); FSELx a<0? (d=b): (d=c) */
		o1 = AOP_RRR(oprrr(ctxt, p->as), p->to.reg, p->from.reg, p->reg)|((p->from3.reg&31L)<<6);
		break;

	case 35:	/* mov r,lext/lauto/loreg ==> cau $(v>>16),sb,r'; store o(r') */
		v = regoff(ctxt, &p->to);
		r = p->to.reg;
		if(r == 0)
			r = o->param;
		o1 = AOP_IRR(OP_ADDIS, REGTMP, r, high16adjusted(v));
		o2 = AOP_IRR(opstore(ctxt, p->as), p->from.reg, REGTMP, v);
		break;

	case 36:	/* mov bz/h/hz lext/lauto/lreg,r ==> lbz/lha/lhz etc */
		v = regoff(ctxt, &p->from);
		r = p->from.reg;
		if(r == 0)
			r = o->param;
		o1 = AOP_IRR(OP_ADDIS, REGTMP, r, high16adjusted(v));
		o2 = AOP_IRR(opload(ctxt, p->as), p->to.reg, REGTMP, v);
		break;

	case 37:	/* movb lext/lauto/lreg,r ==> lbz o(reg),r; extsb r */
		v = regoff(ctxt, &p->from);
		r = p->from.reg;
		if(r == 0)
			r = o->param;
		o1 = AOP_IRR(OP_ADDIS, REGTMP, r, high16adjusted(v));
		o2 = AOP_IRR(opload(ctxt, p->as), p->to.reg, REGTMP, v);
		o3 = LOP_RRR(OP_EXTSB, p->to.reg, p->to.reg, 0);
		break;

	case 40:	/* word */
		o1 = regoff(ctxt, &p->from);
		break;

	case 41:	/* stswi */
		o1 = AOP_RRR(opirr(ctxt, p->as), p->from.reg, p->to.reg, 0) | ((regoff(ctxt, &p->from3)&0x7F)<<11);
		break;

	case 42:	/* lswi */
		o1 = AOP_RRR(opirr(ctxt, p->as), p->to.reg, p->from.reg, 0) | ((regoff(ctxt, &p->from3)&0x7F)<<11);
		break;

	case 43:	/* unary indexed source: dcbf (b); dcbf (a+b) */
		r = p->reg;
		if(r == 0)
			r = 0;
		o1 = AOP_RRR(oprrr(ctxt, p->as), 0, r, p->from.reg);
		break;

	case 44:	/* indexed store */
		r = p->reg;
		if(r == 0)
			r = 0;
		o1 = AOP_RRR(opstorex(ctxt, p->as), p->from.reg, r, p->to.reg);
		break;
	case 45:	/* indexed load */
		r = p->reg;
		if(r == 0)
			r = 0;
		o1 = AOP_RRR(oploadx(ctxt, p->as), p->to.reg, r, p->from.reg);
		break;

	case 46:	/* plain op */
		o1 = oprrr(ctxt, p->as);
		break;

	case 47:	/* op Ra, Rd; also op [Ra,] Rd */
		r = p->from.reg;
		if(r == 0)
			r = p->to.reg;
		o1 = AOP_RRR(oprrr(ctxt, p->as), p->to.reg, r, 0);
		break;

	case 48:	/* op Rs, Ra */
		r = p->from.reg;
		if(r == 0)
			r = p->to.reg;
		o1 = LOP_RRR(oprrr(ctxt, p->as), p->to.reg, r, 0);
		break;

	case 49:	/* op Rb; op $n, Rb */
		if(p->from.type != TYPE_REG){	/* tlbie $L, rB */
			v = regoff(ctxt, &p->from) & 1;
			o1 = AOP_RRR(oprrr(ctxt, p->as), 0, 0, p->to.reg) | (v<<21);
		}else
			o1 = AOP_RRR(oprrr(ctxt, p->as), 0, 0, p->from.reg);
		break;

	case 50:	/* rem[u] r1[,r2],r3 */
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		v = oprrr(ctxt, p->as);
		t = v & ((1<<10)|1);	/* OE|Rc */
		o1 = AOP_RRR(v&~t, REGTMP, r, p->from.reg);
		o2 = AOP_RRR(OP_MULLW, REGTMP, REGTMP, p->from.reg);
		o3 = AOP_RRR(OP_SUBF|t, p->to.reg, REGTMP, r);
		if(p->as == AREMU) {
			o4 = o3;
			/* Clear top 32 bits */
			o3 = OP_RLW(OP_RLDIC, REGTMP, REGTMP, 0, 0, 0) | (1<<5);
		}
		break;

	case 51:	/* remd[u] r1[,r2],r3 */
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		v = oprrr(ctxt, p->as);
		t = v & ((1<<10)|1);	/* OE|Rc */
		o1 = AOP_RRR(v&~t, REGTMP, r, p->from.reg);
		o2 = AOP_RRR(OP_MULLD, REGTMP, REGTMP, p->from.reg);
		o3 = AOP_RRR(OP_SUBF|t, p->to.reg, REGTMP, r);
		break;

	case 52:	/* mtfsbNx cr(n) */
		v = regoff(ctxt, &p->from)&31L;
		o1 = AOP_RRR(oprrr(ctxt, p->as), v, 0, 0);
		break;

	case 53:	/* mffsX ,fr1 */
		o1 = AOP_RRR(OP_MFFS, p->to.reg, 0, 0);
		break;

	case 54:	/* mov msr,r1; mov r1, msr*/
		if(oclass(&p->from) == C_REG){
			if(p->as == AMOVD)
				o1 = AOP_RRR(OP_MTMSRD, p->from.reg, 0, 0);
			else
				o1 = AOP_RRR(OP_MTMSR, p->from.reg, 0, 0);
		}else
			o1 = AOP_RRR(OP_MFMSR, p->to.reg, 0, 0);
		break;

	case 55:	/* op Rb, Rd */
		o1 = AOP_RRR(oprrr(ctxt, p->as), p->to.reg, 0, p->from.reg);
		break;

	case 56:	/* sra $sh,[s,]a; srd $sh,[s,]a */
		v = regoff(ctxt, &p->from);
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		o1 = AOP_RRR(opirr(ctxt, p->as), r, p->to.reg, v&31L);
		if(p->as == ASRAD && (v&0x20))
			o1 |= 1<<1;	/* mb[5] */
		break;

	case 57:	/* slw $sh,[s,]a -> rlwinm ... */
		v = regoff(ctxt, &p->from);
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		/*
		 * Let user (gs) shoot himself in the foot. 
		 * qc has already complained.
		 *
		if(v < 0 || v > 31)
			ctxt->diag("illegal shift %ld\n%P", v, p);
		 */
		if(v < 0)
			v = 0;
		else if(v > 32)
			v = 32;
		if(p->as == ASRW || p->as == ASRWCC) {	/* shift right */
			mask[0] = v;
			mask[1] = 31;
			v = 32-v;
		} else {
			mask[0] = 0;
			mask[1] = 31-v;
		}
		o1 = OP_RLW(OP_RLWINM, p->to.reg, r, v, mask[0], mask[1]);
		if(p->as == ASLWCC || p->as == ASRWCC)
			o1 |= 1;	/* Rc */
		break;

	case 58:		/* logical $andcon,[s],a */
		v = regoff(ctxt, &p->from);
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		o1 = LOP_IRR(opirr(ctxt, p->as), p->to.reg, r, v);
		break;

	case 59:	/* or/and $ucon,,r */
		v = regoff(ctxt, &p->from);
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		o1 = LOP_IRR(opirr(ctxt, p->as+ALAST), p->to.reg, r, v>>16);	/* oris, xoris, andis */
		break;

	case 60:	/* tw to,a,b */
		r = regoff(ctxt, &p->from)&31L;
		o1 = AOP_RRR(oprrr(ctxt, p->as), r, p->reg, p->to.reg);
		break;

	case 61:	/* tw to,a,$simm */
		r = regoff(ctxt, &p->from)&31L;
		v = regoff(ctxt, &p->to);
		o1 = AOP_IRR(opirr(ctxt, p->as), r, p->reg, v);
		break;

	case 62:	/* rlwmi $sh,s,$mask,a */
		v = regoff(ctxt, &p->from);
		maskgen(ctxt, p, mask, regoff(ctxt, &p->from3));
		o1 = AOP_RRR(opirr(ctxt, p->as), p->reg, p->to.reg, v);
		o1 |= ((mask[0]&31L)<<6)|((mask[1]&31L)<<1);
		break;

	case 63:	/* rlwmi b,s,$mask,a */
		maskgen(ctxt, p, mask, regoff(ctxt, &p->from3));
		o1 = AOP_RRR(opirr(ctxt, p->as), p->reg, p->to.reg, p->from.reg);
		o1 |= ((mask[0]&31L)<<6)|((mask[1]&31L)<<1);
		break;

	case 64:	/* mtfsf fr[, $m] {,fpcsr} */
		if(p->from3.type != TYPE_NONE)
			v = regoff(ctxt, &p->from3)&255L;
		else
			v = 255;
		o1 = OP_MTFSF | (v<<17) | (p->from.reg<<11);
		break;

	case 65:	/* MOVFL $imm,FPSCR(n) => mtfsfi crfd,imm */
		if(p->to.reg == 0)
			ctxt->diag("must specify FPSCR(n)\n%P", p);
		o1 = OP_MTFSFI | ((p->to.reg&15L)<<23) | ((regoff(ctxt, &p->from)&31L)<<12);
		break;

	case 66:	/* mov spr,r1; mov r1,spr, also dcr */
		if(REG_R0 <= p->from.reg && p->from.reg <= REG_R31) {
			r = p->from.reg;
			v = p->to.reg;
			if(REG_DCR0 <= v && v <= REG_DCR0+1023)
				o1 = OPVCC(31,451,0,0);	/* mtdcr */
			else
				o1 = OPVCC(31,467,0,0); /* mtspr */
		} else {
			r = p->to.reg;
			v = p->from.reg;
			if(REG_DCR0 <= v && v <= REG_DCR0+1023)
				o1 = OPVCC(31,323,0,0);	/* mfdcr */
			else
				o1 = OPVCC(31,339,0,0);	/* mfspr */
		}
		o1 = AOP_RRR(o1, r, 0, 0) | ((v&0x1f)<<16) | (((v>>5)&0x1f)<<11);
		break;

	case 67:	/* mcrf crfD,crfS */
		if(p->from.type != TYPE_REG || p->from.reg < REG_C0 || REG_C7 < p->from.reg ||
		   p->to.type != TYPE_REG || p->to.reg < REG_C0 || REG_C7 < p->to.reg)
			ctxt->diag("illegal CR field number\n%P", p);
		o1 = AOP_RRR(OP_MCRF, ((p->to.reg&7L)<<2), ((p->from.reg&7)<<2), 0);
		break;

	case 68:	/* mfcr rD; mfocrf CRM,rD */
		if(p->from.type == TYPE_REG && REG_C0 <= p->from.reg && p->from.reg <= REG_C7) {
			v = 1<<(7-(p->to.reg&7));	/* CR(n) */
			o1 = AOP_RRR(OP_MFCR, p->to.reg, 0, 0) | (1<<20) | (v<<12);	/* new form, mfocrf */
		}else
			o1 = AOP_RRR(OP_MFCR, p->to.reg, 0, 0);	/* old form, whole register */
		break;

	case 69:	/* mtcrf CRM,rS */
		if(p->from3.type != TYPE_NONE) {
			if(p->to.reg != 0)
				ctxt->diag("can't use both mask and CR(n)\n%P", p);
			v = regoff(ctxt, &p->from3) & 0xff;
		} else {
			if(p->to.reg == 0)
				v = 0xff;	/* CR */
			else
				v = 1<<(7-(p->to.reg&7));	/* CR(n) */
		}
		o1 = AOP_RRR(OP_MTCRF, p->from.reg, 0, 0) | (v<<12);
		break;

	case 70:	/* [f]cmp r,r,cr*/
		if(p->reg == 0)
			r = 0;
		else
			r = (p->reg&7)<<2;
		o1 = AOP_RRR(oprrr(ctxt, p->as), r, p->from.reg, p->to.reg);
		break;

	case 71:	/* cmp[l] r,i,cr*/
		if(p->reg == 0)
			r = 0;
		else
			r = (p->reg&7)<<2;
		o1 = AOP_RRR(opirr(ctxt, p->as), r, p->from.reg, 0) | (regoff(ctxt, &p->to)&0xffff);
		break;

	case 72:	/* slbmte (Rb+Rs -> slb[Rb]) -> Rs, Rb */
		o1 = AOP_RRR(oprrr(ctxt, p->as), p->from.reg, 0, p->to.reg);
		break;

	case 73:	/* mcrfs crfD,crfS */
		if(p->from.type != TYPE_REG || p->from.reg != REG_FPSCR ||
		   p->to.type != TYPE_REG || p->to.reg < REG_C0 || REG_C7 < p->to.reg)
			ctxt->diag("illegal FPSCR/CR field number\n%P", p);
		o1 = AOP_RRR(OP_MCRFS, ((p->to.reg&7L)<<2), ((0&7)<<2), 0);
		break;

	case 77:	/* syscall $scon, syscall Rx */
		if(p->from.type == TYPE_CONST) {
			if(p->from.offset > BIG || p->from.offset < -BIG)
				ctxt->diag("illegal syscall, sysnum too large: %P", p);
			o1 = AOP_IRR(OP_ADDI, REGZERO, REGZERO, p->from.offset);
		} else if(p->from.type == TYPE_REG) {
			o1 = LOP_RRR(OP_OR, REGZERO, p->from.reg, p->from.reg);
		} else {
			ctxt->diag("illegal syscall: %P", p);
			o1 = 0x7fe00008; // trap always
		}
		o2 = oprrr(ctxt, p->as);
		o3 = AOP_RRR(oprrr(ctxt, AXOR), REGZERO, REGZERO, REGZERO); // XOR R0, R0
		break;

	case 78:	/* undef */
		o1 = 0; /* "An instruction consisting entirely of binary 0s is guaranteed
			   always to be an illegal instruction."  */
		break;

	/* relocation operations */

	case 74:
		v = regoff(ctxt, &p->to);
		o1 = AOP_IRR(OP_ADDIS, REGTMP, REGZERO, high16adjusted(v));
		o2 = AOP_IRR(opstore(ctxt, p->as), p->from.reg, REGTMP, v);
		addaddrreloc(ctxt, p->to.sym, &o1, &o2);
		//if(dlm) reloc(&p->to, p->pc, 1);
		break;

	case 75:
		v = regoff(ctxt, &p->from);
		o1 = AOP_IRR(OP_ADDIS, REGTMP, REGZERO, high16adjusted(v));
		o2 = AOP_IRR(opload(ctxt, p->as), p->to.reg, REGTMP, v);
		addaddrreloc(ctxt, p->from.sym, &o1, &o2);
		//if(dlm) reloc(&p->from, p->pc, 1);
		break;

	case 76:
		v = regoff(ctxt, &p->from);
		o1 = AOP_IRR(OP_ADDIS, REGTMP, REGZERO, high16adjusted(v));
		o2 = AOP_IRR(opload(ctxt, p->as), p->to.reg, REGTMP, v);
		addaddrreloc(ctxt, p->from.sym, &o1, &o2);
		o3 = LOP_RRR(OP_EXTSB, p->to.reg, p->to.reg, 0);
		//if(dlm) reloc(&p->from, p->pc, 1);
		break;

	}

	out[0] = o1;
	out[1] = o2;
	out[2] = o3;
	out[3] = o4;
	out[4] = o5;
	return;
}

static vlong
vregoff(Link *ctxt, Addr *a)
{

	ctxt->instoffset = 0;
	aclass(ctxt, a);
	return ctxt->instoffset;
}

static int32
regoff(Link *ctxt, Addr *a)
{
	return vregoff(ctxt, a);
}

static int32
oprrr(Link *ctxt, int a)
{
	switch(a) {
	case AADD:	return OPVCC(31,266,0,0);
	case AADDCC:	return OPVCC(31,266,0,1);
	case AADDV:	return OPVCC(31,266,1,0);
	case AADDVCC:	return OPVCC(31,266,1,1);
	case AADDC:	return OPVCC(31,10,0,0);
	case AADDCCC:	return OPVCC(31,10,0,1);
	case AADDCV:	return OPVCC(31,10,1,0);
	case AADDCVCC:	return OPVCC(31,10,1,1);
	case AADDE:	return OPVCC(31,138,0,0);
	case AADDECC:	return OPVCC(31,138,0,1);
	case AADDEV:	return OPVCC(31,138,1,0);
	case AADDEVCC:	return OPVCC(31,138,1,1);
	case AADDME:	return OPVCC(31,234,0,0);
	case AADDMECC:	return OPVCC(31,234,0,1);
	case AADDMEV:	return OPVCC(31,234,1,0);
	case AADDMEVCC:	return OPVCC(31,234,1,1);
	case AADDZE:	return OPVCC(31,202,0,0);
	case AADDZECC:	return OPVCC(31,202,0,1);
	case AADDZEV:	return OPVCC(31,202,1,0);
	case AADDZEVCC:	return OPVCC(31,202,1,1);

	case AAND:	return OPVCC(31,28,0,0);
	case AANDCC:	return OPVCC(31,28,0,1);
	case AANDN:	return OPVCC(31,60,0,0);
	case AANDNCC:	return OPVCC(31,60,0,1);

	case ACMP:	return OPVCC(31,0,0,0)|(1<<21);	/* L=1 */
	case ACMPU:	return OPVCC(31,32,0,0)|(1<<21);
	case ACMPW:	return OPVCC(31,0,0,0);	/* L=0 */
	case ACMPWU:	return OPVCC(31,32,0,0);

	case ACNTLZW:	return OPVCC(31,26,0,0);
	case ACNTLZWCC:	return OPVCC(31,26,0,1);
	case ACNTLZD:		return OPVCC(31,58,0,0);
	case ACNTLZDCC:	return OPVCC(31,58,0,1);

	case ACRAND:	return OPVCC(19,257,0,0);
	case ACRANDN:	return OPVCC(19,129,0,0);
	case ACREQV:	return OPVCC(19,289,0,0);
	case ACRNAND:	return OPVCC(19,225,0,0);
	case ACRNOR:	return OPVCC(19,33,0,0);
	case ACROR:	return OPVCC(19,449,0,0);
	case ACRORN:	return OPVCC(19,417,0,0);
	case ACRXOR:	return OPVCC(19,193,0,0);

	case ADCBF:	return OPVCC(31,86,0,0);
	case ADCBI:	return OPVCC(31,470,0,0);
	case ADCBST:	return OPVCC(31,54,0,0);
	case ADCBT:	return OPVCC(31,278,0,0);
	case ADCBTST:	return OPVCC(31,246,0,0);
	case ADCBZ:	return OPVCC(31,1014,0,0);

	case AREM:
	case ADIVW:	return OPVCC(31,491,0,0);
	case AREMCC:
	case ADIVWCC:	return OPVCC(31,491,0,1);
	case AREMV:
	case ADIVWV:	return OPVCC(31,491,1,0);
	case AREMVCC:
	case ADIVWVCC:	return OPVCC(31,491,1,1);
	case AREMU:
	case ADIVWU:	return OPVCC(31,459,0,0);
	case AREMUCC:
	case ADIVWUCC:	return OPVCC(31,459,0,1);
	case AREMUV:
	case ADIVWUV:	return OPVCC(31,459,1,0);
	case AREMUVCC:
	case ADIVWUVCC:	return OPVCC(31,459,1,1);

	case AREMD:
	case ADIVD:	return OPVCC(31,489,0,0);
	case AREMDCC:
	case ADIVDCC:	return OPVCC(31,489,0,1);
	case AREMDV:
	case ADIVDV:	return OPVCC(31,489,1,0);
	case AREMDVCC:
	case ADIVDVCC:	return OPVCC(31,489,1,1);
	case AREMDU:
	case ADIVDU:	return OPVCC(31,457,0,0);
	case AREMDUCC:
	case ADIVDUCC:	return OPVCC(31,457,0,1);
	case AREMDUV:
	case ADIVDUV:	return OPVCC(31,457,1,0);
	case AREMDUVCC:
	case ADIVDUVCC:	return OPVCC(31,457,1,1);

	case AEIEIO:	return OPVCC(31,854,0,0);

	case AEQV:	return OPVCC(31,284,0,0);
	case AEQVCC:	return OPVCC(31,284,0,1);

	case AEXTSB:	return OPVCC(31,954,0,0);
	case AEXTSBCC:	return OPVCC(31,954,0,1);
	case AEXTSH:	return OPVCC(31,922,0,0);
	case AEXTSHCC:	return OPVCC(31,922,0,1);
	case AEXTSW:	return OPVCC(31,986,0,0);
	case AEXTSWCC:	return OPVCC(31,986,0,1);

	case AFABS:	return OPVCC(63,264,0,0);
	case AFABSCC:	return OPVCC(63,264,0,1);
	case AFADD:	return OPVCC(63,21,0,0);
	case AFADDCC:	return OPVCC(63,21,0,1);
	case AFADDS:	return OPVCC(59,21,0,0);
	case AFADDSCC:	return OPVCC(59,21,0,1);
	case AFCMPO:	return OPVCC(63,32,0,0);
	case AFCMPU:	return OPVCC(63,0,0,0);
	case AFCFID:	return OPVCC(63,846,0,0);
	case AFCFIDCC:	return OPVCC(63,846,0,1);
	case AFCTIW:	return OPVCC(63,14,0,0);
	case AFCTIWCC:	return OPVCC(63,14,0,1);
	case AFCTIWZ:	return OPVCC(63,15,0,0);
	case AFCTIWZCC:	return OPVCC(63,15,0,1);
	case AFCTID:	return OPVCC(63,814,0,0);
	case AFCTIDCC:	return OPVCC(63,814,0,1);
	case AFCTIDZ:	return OPVCC(63,815,0,0);
	case AFCTIDZCC:	return OPVCC(63,815,0,1);
	case AFDIV:	return OPVCC(63,18,0,0);
	case AFDIVCC:	return OPVCC(63,18,0,1);
	case AFDIVS:	return OPVCC(59,18,0,0);
	case AFDIVSCC:	return OPVCC(59,18,0,1);
	case AFMADD:	return OPVCC(63,29,0,0);
	case AFMADDCC:	return OPVCC(63,29,0,1);
	case AFMADDS:	return OPVCC(59,29,0,0);
	case AFMADDSCC:	return OPVCC(59,29,0,1);
	case AFMOVS:
	case AFMOVD:	return OPVCC(63,72,0,0);	/* load */
	case AFMOVDCC:	return OPVCC(63,72,0,1);
	case AFMSUB:	return OPVCC(63,28,0,0);
	case AFMSUBCC:	return OPVCC(63,28,0,1);
	case AFMSUBS:	return OPVCC(59,28,0,0);
	case AFMSUBSCC:	return OPVCC(59,28,0,1);
	case AFMUL:	return OPVCC(63,25,0,0);
	case AFMULCC:	return OPVCC(63,25,0,1);
	case AFMULS:	return OPVCC(59,25,0,0);
	case AFMULSCC:	return OPVCC(59,25,0,1);
	case AFNABS:	return OPVCC(63,136,0,0);
	case AFNABSCC:	return OPVCC(63,136,0,1);
	case AFNEG:	return OPVCC(63,40,0,0);
	case AFNEGCC:	return OPVCC(63,40,0,1);
	case AFNMADD:	return OPVCC(63,31,0,0);
	case AFNMADDCC:	return OPVCC(63,31,0,1);
	case AFNMADDS:	return OPVCC(59,31,0,0);
	case AFNMADDSCC:	return OPVCC(59,31,0,1);
	case AFNMSUB:	return OPVCC(63,30,0,0);
	case AFNMSUBCC:	return OPVCC(63,30,0,1);
	case AFNMSUBS:	return OPVCC(59,30,0,0);
	case AFNMSUBSCC:	return OPVCC(59,30,0,1);
	case AFRES:	return OPVCC(59,24,0,0);
	case AFRESCC:	return OPVCC(59,24,0,1);
	case AFRSP:	return OPVCC(63,12,0,0);
	case AFRSPCC:	return OPVCC(63,12,0,1);
	case AFRSQRTE:	return OPVCC(63,26,0,0);
	case AFRSQRTECC:	return OPVCC(63,26,0,1);
	case AFSEL:	return OPVCC(63,23,0,0);
	case AFSELCC:	return OPVCC(63,23,0,1);
	case AFSQRT:	return OPVCC(63,22,0,0);
	case AFSQRTCC:	return OPVCC(63,22,0,1);
	case AFSQRTS:	return OPVCC(59,22,0,0);
	case AFSQRTSCC:	return OPVCC(59,22,0,1);
	case AFSUB:	return OPVCC(63,20,0,0);
	case AFSUBCC:	return OPVCC(63,20,0,1);
	case AFSUBS:	return OPVCC(59,20,0,0);
	case AFSUBSCC:	return OPVCC(59,20,0,1);

	case AICBI:	return OPVCC(31,982,0,0);
	case AISYNC:	return OPVCC(19,150,0,0);

	case AMTFSB0:	return OPVCC(63,70,0,0);
	case AMTFSB0CC:	return OPVCC(63,70,0,1);
	case AMTFSB1:	return OPVCC(63,38,0,0);
	case AMTFSB1CC:	return OPVCC(63,38,0,1);

	case AMULHW:	return OPVCC(31,75,0,0);
	case AMULHWCC:	return OPVCC(31,75,0,1);
	case AMULHWU:	return OPVCC(31,11,0,0);
	case AMULHWUCC:	return OPVCC(31,11,0,1);
	case AMULLW:	return OPVCC(31,235,0,0);
	case AMULLWCC:	return OPVCC(31,235,0,1);
	case AMULLWV:	return OPVCC(31,235,1,0);
	case AMULLWVCC:	return OPVCC(31,235,1,1);

	case AMULHD:	return OPVCC(31,73,0,0);
	case AMULHDCC:	return OPVCC(31,73,0,1);
	case AMULHDU:	return OPVCC(31,9,0,0);
	case AMULHDUCC:	return OPVCC(31,9,0,1);
	case AMULLD:	return OPVCC(31,233,0,0);
	case AMULLDCC:	return OPVCC(31,233,0,1);
	case AMULLDV:	return OPVCC(31,233,1,0);
	case AMULLDVCC:	return OPVCC(31,233,1,1);

	case ANAND:	return OPVCC(31,476,0,0);
	case ANANDCC:	return OPVCC(31,476,0,1);
	case ANEG:	return OPVCC(31,104,0,0);
	case ANEGCC:	return OPVCC(31,104,0,1);
	case ANEGV:	return OPVCC(31,104,1,0);
	case ANEGVCC:	return OPVCC(31,104,1,1);
	case ANOR:	return OPVCC(31,124,0,0);
	case ANORCC:	return OPVCC(31,124,0,1);
	case AOR:	return OPVCC(31,444,0,0);
	case AORCC:	return OPVCC(31,444,0,1);
	case AORN:	return OPVCC(31,412,0,0);
	case AORNCC:	return OPVCC(31,412,0,1);

	case ARFI:	return OPVCC(19,50,0,0);
	case ARFCI:	return OPVCC(19,51,0,0);
	case ARFID:	return OPVCC(19,18,0,0);
	case AHRFID: return OPVCC(19,274,0,0);

	case ARLWMI:	return OPVCC(20,0,0,0);
	case ARLWMICC: return OPVCC(20,0,0,1);
	case ARLWNM:	return OPVCC(23,0,0,0);
	case ARLWNMCC:	return OPVCC(23,0,0,1);

	case ARLDCL:	return OPVCC(30,8,0,0);
	case ARLDCR:	return OPVCC(30,9,0,0);

	case ASYSCALL:	return OPVCC(17,1,0,0);

	case ASLW:	return OPVCC(31,24,0,0);
	case ASLWCC:	return OPVCC(31,24,0,1);
	case ASLD:	return OPVCC(31,27,0,0);
	case ASLDCC:	return OPVCC(31,27,0,1);

	case ASRAW:	return OPVCC(31,792,0,0);
	case ASRAWCC:	return OPVCC(31,792,0,1);
	case ASRAD:	return OPVCC(31,794,0,0);
	case ASRADCC:	return OPVCC(31,794,0,1);

	case ASRW:	return OPVCC(31,536,0,0);
	case ASRWCC:	return OPVCC(31,536,0,1);
	case ASRD:	return OPVCC(31,539,0,0);
	case ASRDCC:	return OPVCC(31,539,0,1);

	case ASUB:	return OPVCC(31,40,0,0);
	case ASUBCC:	return OPVCC(31,40,0,1);
	case ASUBV:	return OPVCC(31,40,1,0);
	case ASUBVCC:	return OPVCC(31,40,1,1);
	case ASUBC:	return OPVCC(31,8,0,0);
	case ASUBCCC:	return OPVCC(31,8,0,1);
	case ASUBCV:	return OPVCC(31,8,1,0);
	case ASUBCVCC:	return OPVCC(31,8,1,1);
	case ASUBE:	return OPVCC(31,136,0,0);
	case ASUBECC:	return OPVCC(31,136,0,1);
	case ASUBEV:	return OPVCC(31,136,1,0);
	case ASUBEVCC:	return OPVCC(31,136,1,1);
	case ASUBME:	return OPVCC(31,232,0,0);
	case ASUBMECC:	return OPVCC(31,232,0,1);
	case ASUBMEV:	return OPVCC(31,232,1,0);
	case ASUBMEVCC:	return OPVCC(31,232,1,1);
	case ASUBZE:	return OPVCC(31,200,0,0);
	case ASUBZECC:	return OPVCC(31,200,0,1);
	case ASUBZEV:	return OPVCC(31,200,1,0);
	case ASUBZEVCC:	return OPVCC(31,200,1,1);

	case ASYNC:	return OPVCC(31,598,0,0);
	case APTESYNC:	return OPVCC(31,598,0,0) | (2<<21);

	case ATLBIE:	return OPVCC(31,306,0,0);
	case ATLBIEL:	return OPVCC(31,274,0,0);
	case ATLBSYNC:	return OPVCC(31,566,0,0);
	case ASLBIA:	return OPVCC(31,498,0,0);
	case ASLBIE:	return OPVCC(31,434,0,0);
	case ASLBMFEE:	return OPVCC(31,915,0,0);
	case ASLBMFEV:	return OPVCC(31,851,0,0);
	case ASLBMTE:		return OPVCC(31,402,0,0);

	case ATW:	return OPVCC(31,4,0,0);
	case ATD:	return OPVCC(31,68,0,0);

	case AXOR:	return OPVCC(31,316,0,0);
	case AXORCC:	return OPVCC(31,316,0,1);
	}
	ctxt->diag("bad r/r opcode %A", a);
	return 0;
}

static int32
opirr(Link *ctxt, int a)
{
	switch(a) {
	case AADD:	return OPVCC(14,0,0,0);
	case AADDC:	return OPVCC(12,0,0,0);
	case AADDCCC:	return OPVCC(13,0,0,0);
	case AADD+ALAST:	return OPVCC(15,0,0,0);		/* ADDIS/CAU */

	case AANDCC:	return OPVCC(28,0,0,0);
	case AANDCC+ALAST:	return OPVCC(29,0,0,0);		/* ANDIS./ANDIU. */

	case ABR:	return OPVCC(18,0,0,0);
	case ABL:	return OPVCC(18,0,0,0) | 1;
	case ADUFFZERO:	return OPVCC(18,0,0,0) | 1;
	case ADUFFCOPY:	return OPVCC(18,0,0,0) | 1;
	case ABC:	return OPVCC(16,0,0,0);
	case ABCL:	return OPVCC(16,0,0,0) | 1;

	case ABEQ:	return AOP_RRR(16<<26,12,2,0);
	case ABGE:	return AOP_RRR(16<<26,4,0,0);
	case ABGT:	return AOP_RRR(16<<26,12,1,0);
	case ABLE:	return AOP_RRR(16<<26,4,1,0);
	case ABLT:	return AOP_RRR(16<<26,12,0,0);
	case ABNE:	return AOP_RRR(16<<26,4,2,0);
	case ABVC:	return AOP_RRR(16<<26,4,3,0);
	case ABVS:	return AOP_RRR(16<<26,12,3,0);

	case ACMP:	return OPVCC(11,0,0,0)|(1<<21);	/* L=1 */
	case ACMPU:	return OPVCC(10,0,0,0)|(1<<21);
	case ACMPW:	return OPVCC(11,0,0,0);	/* L=0 */
	case ACMPWU:	return OPVCC(10,0,0,0);
	case ALSW:	return OPVCC(31,597,0,0);

	case AMULLW:	return OPVCC(7,0,0,0);

	case AOR:	return OPVCC(24,0,0,0);
	case AOR+ALAST:	return OPVCC(25,0,0,0);		/* ORIS/ORIU */

	case ARLWMI:	return OPVCC(20,0,0,0);		/* rlwimi */
	case ARLWMICC:	return OPVCC(20,0,0,1);
	case ARLDMI:	return OPVCC(30,0,0,0) | (3<<2);	/* rldimi */
	case ARLDMICC:	return OPVCC(30,0,0,1) | (3<<2);

	case ARLWNM:	return OPVCC(21,0,0,0);		/* rlwinm */
	case ARLWNMCC:	return OPVCC(21,0,0,1);

	case ARLDCL:	return OPVCC(30,0,0,0);		/* rldicl */
	case ARLDCLCC:	return OPVCC(30,0,0,1);
	case ARLDCR:	return OPVCC(30,1,0,0);		/* rldicr */
	case ARLDCRCC:	return OPVCC(30,1,0,1);
	case ARLDC:	return OPVCC(30,0,0,0) | (2<<2);
	case ARLDCCC:	return OPVCC(30,0,0,1) | (2<<2);

	case ASRAW:	return OPVCC(31,824,0,0);
	case ASRAWCC:	return OPVCC(31,824,0,1);
	case ASRAD:	return OPVCC(31,(413<<1),0,0);
	case ASRADCC:	return OPVCC(31,(413<<1),0,1);

	case ASTSW:	return OPVCC(31,725,0,0);

	case ASUBC:	return OPVCC(8,0,0,0);

	case ATW:	return OPVCC(3,0,0,0);
	case ATD:	return OPVCC(2,0,0,0);

	case AXOR:	return OPVCC(26,0,0,0);		/* XORIL */
	case AXOR+ALAST:	return OPVCC(27,0,0,0);		/* XORIU */
	}
	ctxt->diag("bad opcode i/r %A", a);
	return 0;
}

/*
 * load o(a),d
 */
static int32
opload(Link *ctxt, int a)
{
	switch(a) {
	case AMOVD:	return OPVCC(58,0,0,0);	/* ld */
	case AMOVDU:	return OPVCC(58,0,0,1);	/* ldu */
	case AMOVWZ:	return OPVCC(32,0,0,0);		/* lwz */
	case AMOVWZU:	return OPVCC(33,0,0,0);		/* lwzu */
	case AMOVW:		return OPVCC(58,0,0,0)|(1<<1);	/* lwa */
	/* no AMOVWU */
	case AMOVB:
	case AMOVBZ:	return OPVCC(34,0,0,0);		/* load */
	case AMOVBU:
	case AMOVBZU:	return OPVCC(35,0,0,0);
	case AFMOVD:	return OPVCC(50,0,0,0);
	case AFMOVDU:	return OPVCC(51,0,0,0);
	case AFMOVS:	return OPVCC(48,0,0,0);
	case AFMOVSU:	return OPVCC(49,0,0,0);
	case AMOVH:	return OPVCC(42,0,0,0);
	case AMOVHU:	return OPVCC(43,0,0,0);
	case AMOVHZ:	return OPVCC(40,0,0,0);
	case AMOVHZU:	return OPVCC(41,0,0,0);
	case AMOVMW:	return OPVCC(46,0,0,0);	/* lmw */
	}
	ctxt->diag("bad load opcode %A", a);
	return 0;
}

/*
 * indexed load a(b),d
 */
static int32
oploadx(Link *ctxt, int a)
{
	switch(a) {
	case AMOVWZ: return OPVCC(31,23,0,0);	/* lwzx */
	case AMOVWZU:	return OPVCC(31,55,0,0); /* lwzux */
	case AMOVW:	return OPVCC(31,341,0,0);	/* lwax */
	case AMOVWU:	return OPVCC(31,373,0,0);	/* lwaux */
	case AMOVB:
	case AMOVBZ: return OPVCC(31,87,0,0);	/* lbzx */
	case AMOVBU:
	case AMOVBZU: return OPVCC(31,119,0,0);	/* lbzux */
	case AFMOVD:	return OPVCC(31,599,0,0);	/* lfdx */
	case AFMOVDU:	return OPVCC(31,631,0,0);	/*  lfdux */
	case AFMOVS:	return OPVCC(31,535,0,0);	/* lfsx */
	case AFMOVSU:	return OPVCC(31,567,0,0);	/* lfsux */
	case AMOVH:	return OPVCC(31,343,0,0);	/* lhax */
	case AMOVHU:	return OPVCC(31,375,0,0);	/* lhaux */
	case AMOVHBR:	return OPVCC(31,790,0,0);	/* lhbrx */
	case AMOVWBR:	return OPVCC(31,534,0,0);	/* lwbrx */
	case AMOVHZ:	return OPVCC(31,279,0,0);	/* lhzx */
	case AMOVHZU:	return OPVCC(31,311,0,0);	/* lhzux */
	case AECIWX:	return OPVCC(31,310,0,0);	/* eciwx */
	case ALWAR:	return OPVCC(31,20,0,0);	/* lwarx */
	case ALDAR:	return OPVCC(31,84,0,0);
	case ALSW:	return OPVCC(31,533,0,0);	/* lswx */
	case AMOVD:	return OPVCC(31,21,0,0);	/* ldx */
	case AMOVDU:	return OPVCC(31,53,0,0);	/* ldux */
	}
	ctxt->diag("bad loadx opcode %A", a);
	return 0;
}

/*
 * store s,o(d)
 */
static int32
opstore(Link *ctxt, int a)
{
	switch(a) {
	case AMOVB:
	case AMOVBZ:	return OPVCC(38,0,0,0);	/* stb */
	case AMOVBU:
	case AMOVBZU:	return OPVCC(39,0,0,0);	/* stbu */
	case AFMOVD:	return OPVCC(54,0,0,0);	/* stfd */
	case AFMOVDU:	return OPVCC(55,0,0,0);	/* stfdu */
	case AFMOVS:	return OPVCC(52,0,0,0);	/* stfs */
	case AFMOVSU:	return OPVCC(53,0,0,0);	/* stfsu */
	case AMOVHZ:
	case AMOVH:	return OPVCC(44,0,0,0);	/* sth */
	case AMOVHZU:
	case AMOVHU:	return OPVCC(45,0,0,0);	/* sthu */
	case AMOVMW:	return OPVCC(47,0,0,0);	/* stmw */
	case ASTSW:	return OPVCC(31,725,0,0);	/* stswi */
	case AMOVWZ:
	case AMOVW:	return OPVCC(36,0,0,0);	/* stw */
	case AMOVWZU:
	case AMOVWU:	return OPVCC(37,0,0,0);	/* stwu */
	case AMOVD:	return OPVCC(62,0,0,0);	/* std */
	case AMOVDU:	return OPVCC(62,0,0,1);	/* stdu */
	}
	ctxt->diag("unknown store opcode %A", a);
	return 0;
}

/*
 * indexed store s,a(b)
 */
static int32
opstorex(Link *ctxt, int a)
{
	switch(a) {
	case AMOVB:
	case AMOVBZ:	return OPVCC(31,215,0,0);	/* stbx */
	case AMOVBU:
	case AMOVBZU:	return OPVCC(31,247,0,0);	/* stbux */
	case AFMOVD:	return OPVCC(31,727,0,0);	/* stfdx */
	case AFMOVDU:	return OPVCC(31,759,0,0);	/* stfdux */
	case AFMOVS:	return OPVCC(31,663,0,0);	/* stfsx */
	case AFMOVSU:	return OPVCC(31,695,0,0);	/* stfsux */
	case AMOVHZ:
	case AMOVH:	return OPVCC(31,407,0,0);	/* sthx */
	case AMOVHBR:	return OPVCC(31,918,0,0);	/* sthbrx */
	case AMOVHZU:
	case AMOVHU:	return OPVCC(31,439,0,0);	/* sthux */
	case AMOVWZ:
	case AMOVW:	return OPVCC(31,151,0,0);	/* stwx */
	case AMOVWZU:
	case AMOVWU:	return OPVCC(31,183,0,0);	/* stwux */
	case ASTSW:	return OPVCC(31,661,0,0);	/* stswx */
	case AMOVWBR:	return OPVCC(31,662,0,0);	/* stwbrx */
	case ASTWCCC:	return OPVCC(31,150,0,1);	/* stwcx. */
	case ASTDCCC:	return OPVCC(31,214,0,1);	/* stwdx. */
	case AECOWX:	return OPVCC(31,438,0,0);	/* ecowx */
	case AMOVD:	return OPVCC(31,149,0,0);	/* stdx */
	case AMOVDU:	return OPVCC(31,181,0,0);	/* stdux */
	}
	ctxt->diag("unknown storex opcode %A", a);
	return 0;
}

