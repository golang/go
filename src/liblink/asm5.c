// Inferno utils/5l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/5l/span.c
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

// Instruction layout.

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>
#include "../cmd/5l/5.out.h"
#include "../runtime/stack.h"

typedef	struct	Optab	Optab;
typedef	struct	Oprang	Oprang;
typedef	uchar	Opcross[32][2][32];

struct	Optab
{
	uchar	as;
	uchar	a1;
	char	a2;
	uchar	a3;
	uchar	type;
	char	size;
	char	param;
	char	flag;
	uchar	pcrelsiz;
};
struct	Oprang
{
	Optab*	start;
	Optab*	stop;
};

enum
{
	LFROM		= 1<<0,
	LTO		= 1<<1,
	LPOOL		= 1<<2,
	LPCREL		= 1<<3,
};

static Optab	optab[] =
{
	/* struct Optab:
	  OPCODE,	from, prog->reg, to,		 type,size,param,flag */
	{ ATEXT,	C_ADDR,	C_NONE,	C_TEXTSIZE, 	 0, 0, 0 },

	{ AADD,		C_REG,	C_REG,	C_REG,		 1, 4, 0 },
	{ AADD,		C_REG,	C_NONE,	C_REG,		 1, 4, 0 },
	{ AMOVW,	C_REG,	C_NONE,	C_REG,		 1, 4, 0 },
	{ AMVN,		C_REG,	C_NONE,	C_REG,		 1, 4, 0 },
	{ ACMP,		C_REG,	C_REG,	C_NONE,		 1, 4, 0 },

	{ AADD,		C_RCON,	C_REG,	C_REG,		 2, 4, 0 },
	{ AADD,		C_RCON,	C_NONE,	C_REG,		 2, 4, 0 },
	{ AMOVW,	C_RCON,	C_NONE,	C_REG,		 2, 4, 0 },
	{ AMVN,		C_RCON,	C_NONE,	C_REG,		 2, 4, 0 },
	{ ACMP,		C_RCON,	C_REG,	C_NONE,		 2, 4, 0 },

	{ AADD,		C_SHIFT,C_REG,	C_REG,		 3, 4, 0 },
	{ AADD,		C_SHIFT,C_NONE,	C_REG,		 3, 4, 0 },
	{ AMVN,		C_SHIFT,C_NONE,	C_REG,		 3, 4, 0 },
	{ ACMP,		C_SHIFT,C_REG,	C_NONE,		 3, 4, 0 },

	{ AMOVW,	C_RACON,C_NONE,	C_REG,		 4, 4, REGSP },

	{ AB,		C_NONE,	C_NONE,	C_SBRA,		 5, 4, 0,	LPOOL },
	{ ABL,		C_NONE,	C_NONE,	C_SBRA,		 5, 4, 0 },
	{ ABX,		C_NONE,	C_NONE,	C_SBRA,		 74, 20, 0 },
	{ ABEQ,		C_NONE,	C_NONE,	C_SBRA,		 5, 4, 0 },

	{ AB,		C_NONE,	C_NONE,	C_ROREG,	 6, 4, 0,	LPOOL },
	{ ABL,		C_NONE,	C_NONE,	C_ROREG,	 7, 4, 0 },
	{ ABL,		C_REG,	C_NONE,	C_ROREG,	 7, 4, 0 },
	{ ABX,		C_NONE,	C_NONE,	C_ROREG,	 75, 12, 0 },
	{ ABXRET,	C_NONE,	C_NONE,	C_ROREG,	 76, 4, 0 },

	{ ASLL,		C_RCON,	C_REG,	C_REG,		 8, 4, 0 },
	{ ASLL,		C_RCON,	C_NONE,	C_REG,		 8, 4, 0 },

	{ ASLL,		C_REG,	C_NONE,	C_REG,		 9, 4, 0 },
	{ ASLL,		C_REG,	C_REG,	C_REG,		 9, 4, 0 },

	{ ASWI,		C_NONE,	C_NONE,	C_NONE,		10, 4, 0 },
	{ ASWI,		C_NONE,	C_NONE,	C_LOREG,	10, 4, 0 },
	{ ASWI,		C_NONE,	C_NONE,	C_LCON,		10, 4, 0 },

	{ AWORD,	C_NONE,	C_NONE,	C_LCON,		11, 4, 0 },
	{ AWORD,	C_NONE,	C_NONE,	C_LCONADDR,	11, 4, 0 },
	{ AWORD,	C_NONE,	C_NONE,	C_ADDR,		11, 4, 0 },

	{ AMOVW,	C_NCON,	C_NONE,	C_REG,		12, 4, 0 },
	{ AMOVW,	C_LCON,	C_NONE,	C_REG,		12, 4, 0,	LFROM },
	{ AMOVW,	C_LCONADDR,	C_NONE,	C_REG,	12, 4, 0,	LFROM | LPCREL, 4},

	{ AADD,		C_NCON,	C_REG,	C_REG,		13, 8, 0 },
	{ AADD,		C_NCON,	C_NONE,	C_REG,		13, 8, 0 },
	{ AMVN,		C_NCON,	C_NONE,	C_REG,		13, 8, 0 },
	{ ACMP,		C_NCON,	C_REG,	C_NONE,		13, 8, 0 },
	{ AADD,		C_LCON,	C_REG,	C_REG,		13, 8, 0,	LFROM },
	{ AADD,		C_LCON,	C_NONE,	C_REG,		13, 8, 0,	LFROM },
	{ AMVN,		C_LCON,	C_NONE,	C_REG,		13, 8, 0,	LFROM },
	{ ACMP,		C_LCON,	C_REG,	C_NONE,		13, 8, 0,	LFROM },

	{ AMOVB,	C_REG,	C_NONE,	C_REG,		 1, 4, 0 },
	{ AMOVBS,	C_REG,	C_NONE,	C_REG,		14, 8, 0 },
	{ AMOVBU,	C_REG,	C_NONE,	C_REG,		58, 4, 0 },
	{ AMOVH,	C_REG,	C_NONE,	C_REG,		 1, 4, 0 },
	{ AMOVHS,	C_REG,	C_NONE,	C_REG,		14, 8, 0 },
	{ AMOVHU,	C_REG,	C_NONE,	C_REG,		14, 8, 0 },

	{ AMUL,		C_REG,	C_REG,	C_REG,		15, 4, 0 },
	{ AMUL,		C_REG,	C_NONE,	C_REG,		15, 4, 0 },

	{ ADIV,		C_REG,	C_REG,	C_REG,		16, 4, 0 },
	{ ADIV,		C_REG,	C_NONE,	C_REG,		16, 4, 0 },

	{ AMULL,	C_REG,	C_REG,	C_REGREG,	17, 4, 0 },
	{ AMULA,	C_REG,	C_REG,	C_REGREG2,	17, 4, 0 },

	{ AMOVW,	C_REG,	C_NONE,	C_SAUTO,	20, 4, REGSP },
	{ AMOVW,	C_REG,	C_NONE,	C_SOREG,	20, 4, 0 },
	{ AMOVB,	C_REG,	C_NONE,	C_SAUTO,	20, 4, REGSP },
	{ AMOVB,	C_REG,	C_NONE,	C_SOREG,	20, 4, 0 },
	{ AMOVBS,	C_REG,	C_NONE,	C_SAUTO,	20, 4, REGSP },
	{ AMOVBS,	C_REG,	C_NONE,	C_SOREG,	20, 4, 0 },
	{ AMOVBU,	C_REG,	C_NONE,	C_SAUTO,	20, 4, REGSP },
	{ AMOVBU,	C_REG,	C_NONE,	C_SOREG,	20, 4, 0 },

	{ AMOVW,	C_SAUTO,C_NONE,	C_REG,		21, 4, REGSP },
	{ AMOVW,	C_SOREG,C_NONE,	C_REG,		21, 4, 0 },
	{ AMOVBU,	C_SAUTO,C_NONE,	C_REG,		21, 4, REGSP },
	{ AMOVBU,	C_SOREG,C_NONE,	C_REG,		21, 4, 0 },

	{ AMOVW,	C_REG,	C_NONE,	C_LAUTO,	30, 8, REGSP,	LTO },
	{ AMOVW,	C_REG,	C_NONE,	C_LOREG,	30, 8, 0,	LTO },
	{ AMOVW,	C_REG,	C_NONE,	C_ADDR,		64, 8, 0,	LTO | LPCREL, 4 },
	{ AMOVB,	C_REG,	C_NONE,	C_LAUTO,	30, 8, REGSP,	LTO },
	{ AMOVB,	C_REG,	C_NONE,	C_LOREG,	30, 8, 0,	LTO },
	{ AMOVB,	C_REG,	C_NONE,	C_ADDR,		64, 8, 0,	LTO | LPCREL, 4 },
	{ AMOVBS,	C_REG,	C_NONE,	C_LAUTO,	30, 8, REGSP,	LTO },
	{ AMOVBS,	C_REG,	C_NONE,	C_LOREG,	30, 8, 0,	LTO },
	{ AMOVBS,	C_REG,	C_NONE,	C_ADDR,		64, 8, 0,	LTO | LPCREL, 4 },
	{ AMOVBU,	C_REG,	C_NONE,	C_LAUTO,	30, 8, REGSP,	LTO },
	{ AMOVBU,	C_REG,	C_NONE,	C_LOREG,	30, 8, 0,	LTO },
	{ AMOVBU,	C_REG,	C_NONE,	C_ADDR,		64, 8, 0,	LTO | LPCREL, 4 },

	{ AMOVW,	C_LAUTO,C_NONE,	C_REG,		31, 8, REGSP,	LFROM },
	{ AMOVW,	C_LOREG,C_NONE,	C_REG,		31, 8, 0,	LFROM },
	{ AMOVW,	C_ADDR,	C_NONE,	C_REG,		65, 8, 0,	LFROM | LPCREL, 4 },
	{ AMOVBU,	C_LAUTO,C_NONE,	C_REG,		31, 8, REGSP,	LFROM },
	{ AMOVBU,	C_LOREG,C_NONE,	C_REG,		31, 8, 0,	LFROM },
	{ AMOVBU,	C_ADDR,	C_NONE,	C_REG,		65, 8, 0,	LFROM | LPCREL, 4 },

	{ AMOVW,	C_LACON,C_NONE,	C_REG,		34, 8, REGSP,	LFROM },

	{ AMOVW,	C_PSR,	C_NONE,	C_REG,		35, 4, 0 },
	{ AMOVW,	C_REG,	C_NONE,	C_PSR,		36, 4, 0 },
	{ AMOVW,	C_RCON,	C_NONE,	C_PSR,		37, 4, 0 },

	{ AMOVM,	C_LCON,	C_NONE,	C_SOREG,	38, 4, 0 },
	{ AMOVM,	C_SOREG,C_NONE,	C_LCON,		39, 4, 0 },

	{ ASWPW,	C_SOREG,C_REG,	C_REG,		40, 4, 0 },

	{ ARFE,		C_NONE,	C_NONE,	C_NONE,		41, 4, 0 },

	{ AMOVF,	C_FREG,	C_NONE,	C_FAUTO,	50, 4, REGSP },
	{ AMOVF,	C_FREG,	C_NONE,	C_FOREG,	50, 4, 0 },

	{ AMOVF,	C_FAUTO,C_NONE,	C_FREG,		51, 4, REGSP },
	{ AMOVF,	C_FOREG,C_NONE,	C_FREG,		51, 4, 0 },

	{ AMOVF,	C_FREG,	C_NONE,	C_LAUTO,	52, 12, REGSP,	LTO },
	{ AMOVF,	C_FREG,	C_NONE,	C_LOREG,	52, 12, 0,	LTO },

	{ AMOVF,	C_LAUTO,C_NONE,	C_FREG,		53, 12, REGSP,	LFROM },
	{ AMOVF,	C_LOREG,C_NONE,	C_FREG,		53, 12, 0,	LFROM },

	{ AMOVF,	C_FREG,	C_NONE,	C_ADDR,		68, 8, 0,	LTO | LPCREL, 4 },
	{ AMOVF,	C_ADDR,	C_NONE,	C_FREG,		69, 8, 0,	LFROM | LPCREL, 4},

	{ AADDF,	C_FREG,	C_NONE,	C_FREG,		54, 4, 0 },
	{ AADDF,	C_FREG,	C_REG,	C_FREG,		54, 4, 0 },
	{ AMOVF,	C_FREG, C_NONE, C_FREG,		54, 4, 0 },

	{ AMOVW,	C_REG,	C_NONE,	C_FCR,		56, 4, 0 },
	{ AMOVW,	C_FCR,	C_NONE,	C_REG,		57, 4, 0 },

	{ AMOVW,	C_SHIFT,C_NONE,	C_REG,		59, 4, 0 },
	{ AMOVBU,	C_SHIFT,C_NONE,	C_REG,		59, 4, 0 },

	{ AMOVB,	C_SHIFT,C_NONE,	C_REG,		60, 4, 0 },
	{ AMOVBS,	C_SHIFT,C_NONE,	C_REG,		60, 4, 0 },

	{ AMOVW,	C_REG,	C_NONE,	C_SHIFT,	61, 4, 0 },
	{ AMOVB,	C_REG,	C_NONE,	C_SHIFT,	61, 4, 0 },
	{ AMOVBS,	C_REG,	C_NONE,	C_SHIFT,	61, 4, 0 },
	{ AMOVBU,	C_REG,	C_NONE,	C_SHIFT,	61, 4, 0 },

	{ ACASE,	C_REG,	C_NONE,	C_NONE,		62, 4, 0, LPCREL, 8 },
	{ ABCASE,	C_NONE, C_NONE, C_SBRA,		63, 4, 0, LPCREL, 0 },

	{ AMOVH,	C_REG,	C_NONE, C_HAUTO,	70, 4, REGSP,	0 },
	{ AMOVH,	C_REG,	C_NONE,	C_HOREG,	70, 4, 0,	0 },
	{ AMOVHS,	C_REG,	C_NONE, C_HAUTO,	70, 4, REGSP,	0 },
	{ AMOVHS,	C_REG,	C_NONE,	C_HOREG,	70, 4, 0,	0 },
	{ AMOVHU,	C_REG,	C_NONE, C_HAUTO,	70, 4, REGSP,	0 },
	{ AMOVHU,	C_REG,	C_NONE,	C_HOREG,	70, 4, 0,	0 },

	{ AMOVB,	C_HAUTO,C_NONE,	C_REG,		71, 4, REGSP,	0 },
	{ AMOVB,	C_HOREG,C_NONE,	C_REG,		71, 4, 0,	0 },
	{ AMOVBS,	C_HAUTO,C_NONE,	C_REG,		71, 4, REGSP,	0 },
	{ AMOVBS,	C_HOREG,C_NONE,	C_REG,		71, 4, 0,	0 },
	{ AMOVH,	C_HAUTO,C_NONE, C_REG,		71, 4, REGSP,	0 },
	{ AMOVH,	C_HOREG,C_NONE,	C_REG,		71, 4, 0,	0 },
	{ AMOVHS,	C_HAUTO,C_NONE, C_REG,		71, 4, REGSP,	0 },
	{ AMOVHS,	C_HOREG,C_NONE,	C_REG,		71, 4, 0,	0 },
	{ AMOVHU,	C_HAUTO,C_NONE, C_REG,		71, 4, REGSP,	0 },
	{ AMOVHU,	C_HOREG,C_NONE,	C_REG,		71, 4, 0,	0 },

	{ AMOVH,	C_REG,	C_NONE, C_LAUTO,	72, 8, REGSP,	LTO },
	{ AMOVH,	C_REG,	C_NONE,	C_LOREG,	72, 8, 0,	LTO },
	{ AMOVH,	C_REG,	C_NONE,	C_ADDR,	94, 8, 0,	LTO | LPCREL, 4 },
	{ AMOVHS,	C_REG,	C_NONE, C_LAUTO,	72, 8, REGSP,	LTO },
	{ AMOVHS,	C_REG,	C_NONE,	C_LOREG,	72, 8, 0,	LTO },
	{ AMOVHS,	C_REG,	C_NONE,	C_ADDR,	94, 8, 0,	LTO | LPCREL, 4 },
	{ AMOVHU,	C_REG,	C_NONE, C_LAUTO,	72, 8, REGSP,	LTO },
	{ AMOVHU,	C_REG,	C_NONE,	C_LOREG,	72, 8, 0,	LTO },
	{ AMOVHU,	C_REG,	C_NONE,	C_ADDR,	94, 8, 0,	LTO | LPCREL, 4 },

	{ AMOVB,	C_LAUTO,C_NONE,	C_REG,		73, 8, REGSP,	LFROM },
	{ AMOVB,	C_LOREG,C_NONE,	C_REG,		73, 8, 0,	LFROM },
	{ AMOVB,	C_ADDR,	C_NONE,	C_REG,		93, 8, 0,	LFROM | LPCREL, 4 },
	{ AMOVBS,	C_LAUTO,C_NONE,	C_REG,		73, 8, REGSP,	LFROM },
	{ AMOVBS,	C_LOREG,C_NONE,	C_REG,		73, 8, 0,	LFROM },
	{ AMOVBS,	C_ADDR,	C_NONE,	C_REG,		93, 8, 0,	LFROM | LPCREL, 4 },
	{ AMOVH,	C_LAUTO,C_NONE, C_REG,		73, 8, REGSP,	LFROM },
	{ AMOVH,	C_LOREG,C_NONE,	C_REG,		73, 8, 0,	LFROM },
	{ AMOVH,	C_ADDR,	C_NONE,	C_REG,		93, 8, 0,	LFROM | LPCREL, 4 },
	{ AMOVHS,	C_LAUTO,C_NONE, C_REG,		73, 8, REGSP,	LFROM },
	{ AMOVHS,	C_LOREG,C_NONE,	C_REG,		73, 8, 0,	LFROM },
	{ AMOVHS,	C_ADDR,	C_NONE,	C_REG,		93, 8, 0,	LFROM | LPCREL, 4 },
	{ AMOVHU,	C_LAUTO,C_NONE, C_REG,		73, 8, REGSP,	LFROM },
	{ AMOVHU,	C_LOREG,C_NONE,	C_REG,		73, 8, 0,	LFROM },
	{ AMOVHU,	C_ADDR,	C_NONE,	C_REG,		93, 8, 0,	LFROM | LPCREL, 4 },

	{ ALDREX,	C_SOREG,C_NONE,	C_REG,		77, 4, 0 },
	{ ASTREX,	C_SOREG,C_REG,	C_REG,		78, 4, 0 },

	{ AMOVF,	C_ZFCON,C_NONE,	C_FREG,		80, 8, 0 },
	{ AMOVF,	C_SFCON,C_NONE,	C_FREG,		81, 4, 0 },

	{ ACMPF,	C_FREG,	C_REG,	C_NONE,		82, 8, 0 },
	{ ACMPF,	C_FREG, C_NONE,	C_NONE,		83, 8, 0 },

	{ AMOVFW,	C_FREG,	C_NONE,	C_FREG,		84, 4, 0 },
	{ AMOVWF,	C_FREG,	C_NONE,	C_FREG,		85, 4, 0 },

	{ AMOVFW,	C_FREG,	C_NONE,	C_REG,		86, 8, 0 },
	{ AMOVWF,	C_REG,	C_NONE,	C_FREG,		87, 8, 0 },

	{ AMOVW,	C_REG,	C_NONE,	C_FREG,		88, 4, 0 },
	{ AMOVW,	C_FREG,	C_NONE,	C_REG,		89, 4, 0 },

	{ ATST,		C_REG,	C_NONE,	C_NONE,		90, 4, 0 },

	{ ALDREXD,	C_SOREG,C_NONE,	C_REG,		91, 4, 0 },
	{ ASTREXD,	C_SOREG,C_REG,	C_REG,		92, 4, 0 },

	{ APLD,		C_SOREG,C_NONE,	C_NONE,		95, 4, 0 },
	
	{ AUNDEF,		C_NONE,	C_NONE,	C_NONE,		96, 4, 0 },

	{ ACLZ,		C_REG,	C_NONE,	C_REG,		97, 4, 0 },

	{ AMULWT,	C_REG,	C_REG,	C_REG,		98, 4, 0 },
	{ AMULAWT,	C_REG,	C_REG,	C_REGREG2,		99, 4, 0 },

	{ AUSEFIELD,	C_ADDR,	C_NONE,	C_NONE, 	 0, 0, 0 },
	{ APCDATA,	C_LCON,	C_NONE,	C_LCON,		0, 0, 0 },
	{ AFUNCDATA,	C_LCON,	C_NONE,	C_ADDR,	0, 0, 0 },
	{ ANOP,		C_NONE,	C_NONE,	C_NONE,		0, 0, 0 },

	{ ADUFFZERO,	C_NONE,	C_NONE,	C_SBRA,		 5, 4, 0 },  // same as ABL
	{ ADUFFCOPY,	C_NONE,	C_NONE,	C_SBRA,		 5, 4, 0 },  // same as ABL

	{ ADATABUNDLE,	C_NONE, C_NONE, C_NONE,		100, 4, 0 },
	{ ADATABUNDLEEND,	C_NONE, C_NONE, C_NONE,		100, 0, 0 },

	{ AXXX,		C_NONE,	C_NONE,	C_NONE,		 0, 4, 0 },
};

static struct {
	uint32	start;
	uint32	size;
	uint32	extra;
} pool;

static int	checkpool(Link*, Prog*, int);
static int 	flushpool(Link*, Prog*, int, int);
static void	addpool(Link*, Prog*, Addr*);
static void	asmout(Link*, Prog*, Optab*, uint32*);
static int	asmoutnacl(Link*, int32, Prog*, Optab*, uint32*);
static Optab*	oplook(Link*, Prog*);
static uint32	oprrr(Link*, int, int);
static uint32	olr(Link*, int32, int, int, int);
static uint32	olhr(Link*, int32, int, int, int);
static uint32	olrr(Link*, int, int, int, int);
static uint32	olhrr(Link*, int, int, int, int);
static uint32	osr(Link*, int, int, int32, int, int);
static uint32	oshr(Link*, int, int32, int, int);
static uint32	ofsr(Link*, int, int, int32, int, int, Prog*);
static uint32	osrr(Link*, int, int, int, int);
static uint32	oshrr(Link*, int, int, int, int);
static uint32	omvl(Link*, Prog*, Addr*, int);
static int32	immaddr(int32);
static int	aclass(Link*, Addr*);
static int32	immrot(uint32);
static int32	immaddr(int32);
static uint32	opbra(Link*, int, int);

static	Oprang	oprange[ALAST];
static	uchar	xcmp[C_GOK+1][C_GOK+1];

static LSym *deferreturn;

static void
nocache(Prog *p)
{
	p->optab = 0;
	p->from.class = 0;
	p->to.class = 0;
}

/* size of a case statement including jump table */
static int32
casesz(Link *ctxt, Prog *p)
{
	int jt = 0;
	int32 n = 0;
	Optab *o;

	for( ; p != nil; p = p->link){
		if(p->as == ABCASE)
			jt = 1;
		else if(jt)
			break;
		o = oplook(ctxt, p);
		n += o->size;
	}
	return n;
}

static void buildop(Link*);

// Note about encoding: Prog.scond holds the condition encoding,
// but XOR'ed with C_SCOND_XOR, so that C_SCOND_NONE == 0.
// The code that shifts the value << 28 has the responsibility
// for XORing with C_SCOND_XOR too.

// asmoutnacl assembles the instruction p. It replaces asmout for NaCl.
// It returns the total number of bytes put in out, and it can change
// p->pc if extra padding is necessary.
// In rare cases, asmoutnacl might split p into two instructions.
// origPC is the PC for this Prog (no padding is taken into account).
static int
asmoutnacl(Link *ctxt, int32 origPC, Prog *p, Optab *o, uint32 *out)
{
	int size, reg;
	Prog *q;
	Addr *a, *a2;

	size = o->size;

	// instruction specific
	switch(p->as) {
	default:
		if(out != nil)
			asmout(ctxt, p, o, out);
		break;
	case ADATABUNDLE: // align to 16-byte boundary
	case ADATABUNDLEEND: // zero width instruction, just to align next instruction to 16-byte boundary
		p->pc = (p->pc+15) & ~15;
		if(out != nil)
			asmout(ctxt, p, o, out);
		break;
	case AUNDEF:
	case APLD:
		size = 4;
		if(out != nil) {
			switch(p->as) {
			case AUNDEF:
				out[0] = 0xe7fedef0; // NACL_INSTR_ARM_ABORT_NOW (UDF #0xEDE0)
				break;
			case APLD:
				out[0] = 0xe1a01001; // (MOVW R1, R1)
				break;
			}
		}
		break;
	case AB:
	case ABL:
		if(p->to.type != TYPE_MEM) {
			if(out != nil)
				asmout(ctxt, p, o, out);
		} else {
			if(p->to.offset != 0 || size != 4 || p->to.reg > REG_R15 || p->to.reg < REG_R0)
				ctxt->diag("unsupported instruction: %P", p);
			if((p->pc&15) == 12)
				p->pc += 4;
			if(out != nil) {
				out[0] = (((p->scond&C_SCOND) ^ C_SCOND_XOR)<<28) | 0x03c0013f | ((p->to.reg&15) << 12) | ((p->to.reg&15) << 16); // BIC $0xc000000f, Rx
				if(p->as == AB)
					out[1] = (((p->scond&C_SCOND) ^ C_SCOND_XOR)<<28) | 0x012fff10 | (p->to.reg&15)<<0; // BX Rx
				else // ABL
					out[1] = (((p->scond&C_SCOND) ^ C_SCOND_XOR)<<28) | 0x012fff30 | (p->to.reg&15)<<0; // BLX Rx
			}
			size = 8;
		}
		// align the last instruction (the actual BL) to the last instruction in a bundle
		if(p->as == ABL) {
			if(deferreturn == nil)
				deferreturn = linklookup(ctxt, "runtime.deferreturn", 0);
			if(p->to.sym == deferreturn)
				p->pc = ((origPC+15) & ~15) + 16 - size;
			else
				p->pc += (16 - ((p->pc+size)&15)) & 15;
		}
		break;
	case ALDREX:
	case ALDREXD:
	case AMOVB:
	case AMOVBS:
	case AMOVBU:
	case AMOVD:
	case AMOVF:
	case AMOVH:
	case AMOVHS:
	case AMOVHU:
	case AMOVM:
	case AMOVW:
	case ASTREX:
	case ASTREXD:
		if(p->to.type == TYPE_REG && p->to.reg == REG_R15 && p->from.reg == REG_R13) { // MOVW.W x(R13), PC
			if(out != nil)
				asmout(ctxt, p, o, out);
			if(size == 4) {
				if(out != nil) {
					// Note: 5c and 5g reg.c know that DIV/MOD smashes R12
					// so that this return instruction expansion is valid.
					out[0] = out[0] & ~0x3000; // change PC to R12
					out[1] = (((p->scond&C_SCOND) ^ C_SCOND_XOR)<<28) | 0x03ccc13f; // BIC $0xc000000f, R12
					out[2] = (((p->scond&C_SCOND) ^ C_SCOND_XOR)<<28) | 0x012fff1c; // BX R12
				}
				size += 8;
				if(((p->pc+size) & 15) == 4)
					p->pc += 4;
				break;
			} else {
				// if the instruction used more than 4 bytes, then it must have used a very large
				// offset to update R13, so we need to additionally mask R13.
				if(out != nil) {
					out[size/4-1] &= ~0x3000; // change PC to R12
					out[size/4] = (((p->scond&C_SCOND) ^ C_SCOND_XOR)<<28) | 0x03cdd103; // BIC $0xc0000000, R13
					out[size/4+1] = (((p->scond&C_SCOND) ^ C_SCOND_XOR)<<28) | 0x03ccc13f; // BIC $0xc000000f, R12
					out[size/4+2] = (((p->scond&C_SCOND) ^ C_SCOND_XOR)<<28) | 0x012fff1c; // BX R12
				}
				// p->pc+size is only ok at 4 or 12 mod 16.
				if((p->pc+size)%8 == 0)
					p->pc += 4;
				size += 12;
				break;
			}
		}

		if(p->to.type == TYPE_REG && p->to.reg == REG_R15)
			ctxt->diag("unsupported instruction (move to another register and use indirect jump instead): %P", p);

		if(p->to.type == TYPE_MEM && p->to.reg == REG_R13 && (p->scond & C_WBIT) && size > 4) {
			// function prolog with very large frame size: MOVW.W R14,-100004(R13)
			// split it into two instructions:
			// 	ADD $-100004, R13
			// 	MOVW R14, 0(R13)
			q = emallocz(sizeof(Prog));
			p->scond &= ~C_WBIT;
			*q = *p;
			a = &p->to;
			if(p->to.type == TYPE_MEM)
				a2 = &q->to;
			else
				a2 = &q->from;
			nocache(q);
			nocache(p);
			// insert q after p
			q->link = p->link;
			p->link = q;
			q->pcond = nil;
			// make p into ADD $X, R13
			p->as = AADD;
			p->from = *a;
			p->from.reg = 0;
			p->from.type = TYPE_CONST;
			p->to = zprog.to;
			p->to.type = TYPE_REG;
			p->to.reg = REG_R13;
			// make q into p but load/store from 0(R13)
			q->spadj = 0;
			*a2 = zprog.from;
			a2->type = TYPE_MEM;
			a2->reg = REG_R13;
			a2->sym = nil;
			a2->offset = 0;
			size = oplook(ctxt, p)->size;
			break;
		}

		if((p->to.type == TYPE_MEM && p->to.reg != REG_R13 && p->to.reg != REG_R9) || // MOVW Rx, X(Ry), y != 13 && y != 9
		   (p->from.type == TYPE_MEM && p->from.reg != REG_R13 && p->from.reg != REG_R9)) { // MOVW X(Rx), Ry, x != 13 && x != 9
			if(p->to.type == TYPE_MEM)
				a = &p->to;
			else
				a = &p->from;
			reg = a->reg;
			if(size == 4) {
				// if addr.reg == 0, then it is probably load from x(FP) with small x, no need to modify.
				if(reg == 0) {
					if(out != nil)
						asmout(ctxt, p, o, out);
				} else {
					if(out != nil)
						out[0] = (((p->scond&C_SCOND) ^ C_SCOND_XOR)<<28) | 0x03c00103 | ((reg&15) << 16) | ((reg&15) << 12); // BIC $0xc0000000, Rx
					if((p->pc&15) == 12)
						p->pc += 4;
					size += 4;
					if(out != nil)
						asmout(ctxt, p, o, &out[1]);
				}
				break;
			} else {
				// if a load/store instruction takes more than 1 word to implement, then
				// we need to seperate the instruction into two:
				// 1. explicitly load the address into R11.
				// 2. load/store from R11.
				// This won't handle .W/.P, so we should reject such code.
				if(p->scond & (C_PBIT|C_WBIT))
					ctxt->diag("unsupported instruction (.P/.W): %P", p);
				q = emallocz(sizeof(Prog));
				*q = *p;
				if(p->to.type == TYPE_MEM)
					a2 = &q->to;
				else
					a2 = &q->from;
				nocache(q);
				nocache(p);
				// insert q after p
				q->link = p->link;
				p->link = q;
				q->pcond = nil;
				// make p into MOVW $X(R), R11
				p->as = AMOVW;
				p->from = *a;
				p->from.type = TYPE_ADDR;
				p->to = zprog.to;
				p->to.type = TYPE_REG;
				p->to.reg = REG_R11;
				// make q into p but load/store from 0(R11)
				*a2 = zprog.from;
				a2->type = TYPE_MEM;
				a2->reg = REG_R11;
				a2->sym = nil;
				a2->offset = 0;
				size = oplook(ctxt, p)->size;
				break;
			}
		} else if(out != nil)
			asmout(ctxt, p, o, out);
		break;
	}

	// destination register specific
	if(p->to.type == TYPE_REG) {
		switch(p->to.reg) {
		case REG_R9:
			ctxt->diag("invalid instruction, cannot write to R9: %P", p);
			break;
		case REG_R13:
			if(out != nil)
				out[size/4] = 0xe3cdd103; // BIC $0xc0000000, R13
			if(((p->pc+size) & 15) == 0)
				p->pc += 4;
			size += 4;
			break;
		}
	}
	return size;
}

void
span5(Link *ctxt, LSym *cursym)
{
	Prog *p, *op;
	Optab *o;
	int m, bflag, i, v, times;
	int32 c, opc;
	uint32 out[6+3];
	uchar *bp;

	p = cursym->text;
	if(p == nil || p->link == nil) // handle external functions and ELF section symbols
		return;
 
 	if(oprange[AAND].start == nil)
 		buildop(ctxt);

 	ctxt->cursym = cursym;

	ctxt->autosize = p->to.offset + 4;
	c = 0;

	for(op = p, p = p->link; p != nil || ctxt->blitrl != nil; op = p, p = p->link) {
		if(p == nil) {
		       	if(checkpool(ctxt, op, 0)) {
				p = op;
				continue;
			}
			// can't happen: blitrl is not nil, but checkpool didn't flushpool
			ctxt->diag("internal inconsistency");
			break;
		}
		ctxt->curp = p;
		p->pc = c;
		o = oplook(ctxt, p);
		if(ctxt->headtype != Hnacl) {
			m = o->size;
		} else {
			m = asmoutnacl(ctxt, c, p, o, nil);
			c = p->pc; // asmoutnacl might change pc for alignment
			o = oplook(ctxt, p); // asmoutnacl might change p in rare cases
		}
		if(m % 4 != 0 || p->pc % 4 != 0) {
			ctxt->diag("!pc invalid: %P size=%d", p, m);
		}
		// must check literal pool here in case p generates many instructions
		if(ctxt->blitrl){
			i = m;
			if(p->as == ACASE)
				i = casesz(ctxt, p);
			if(checkpool(ctxt, op, i)) {
				p = op;
				continue;
			}
		}
		if(m == 0 && (p->as != AFUNCDATA && p->as != APCDATA && p->as != ADATABUNDLEEND && p->as != ANOP)) {
			ctxt->diag("zero-width instruction\n%P", p);
			continue;
		}
		switch(o->flag & (LFROM|LTO|LPOOL)) {
		case LFROM:
			addpool(ctxt, p, &p->from);
			break;
		case LTO:
			addpool(ctxt, p, &p->to);
			break;
		case LPOOL:
			if ((p->scond&C_SCOND) == C_SCOND_NONE)
				flushpool(ctxt, p, 0, 0);
			break;
		}
		if(p->as==AMOVW && p->to.type==TYPE_REG && p->to.reg==REGPC && (p->scond&C_SCOND) == C_SCOND_NONE)
			flushpool(ctxt, p, 0, 0);
		c += m;
	}
	cursym->size = c;

	/*
	 * if any procedure is large enough to
	 * generate a large SBRA branch, then
	 * generate extra passes putting branches
	 * around jmps to fix. this is rare.
	 */
	times = 0;
	do {
		if(ctxt->debugvlog)
			Bprint(ctxt->bso, "%5.2f span1\n", cputime());
		bflag = 0;
		c = 0;
		times++;
		cursym->text->pc = 0; // force re-layout the code.
		for(p = cursym->text; p != nil; p = p->link) {
			ctxt->curp = p;
			o = oplook(ctxt,p);
			if(c > p->pc)
				p->pc = c;
/* very large branches
			if(o->type == 6 && p->pcond) {
				otxt = p->pcond->pc - c;
				if(otxt < 0)
					otxt = -otxt;
				if(otxt >= (1L<<17) - 10) {
					q = emallocz(sizeof(Prog));
					q->link = p->link;
					p->link = q;
					q->as = AB;
					q->to.type = TYPE_BRANCH;
					q->pcond = p->pcond;
					p->pcond = q;
					q = emallocz(sizeof(Prog));
					q->link = p->link;
					p->link = q;
					q->as = AB;
					q->to.type = TYPE_BRANCH;
					q->pcond = q->link->link;
					bflag = 1;
				}
			}
 */
			opc = p->pc;
			if(ctxt->headtype != Hnacl)
				m = o->size;
			else
				m = asmoutnacl(ctxt, c, p, o, nil);
			if(p->pc != opc) {
				bflag = 1;
				//print("%P pc changed %d to %d in iter. %d\n", p, opc, (int32)p->pc, times);
			}
			c = p->pc + m;
			if(m % 4 != 0 || p->pc % 4 != 0) {
				ctxt->diag("pc invalid: %P size=%d", p, m);
			}
			if(m/4 > nelem(out))
				ctxt->diag("instruction size too large: %d > %d", m/4, nelem(out));
			if(m == 0 && (p->as != AFUNCDATA && p->as != APCDATA && p->as != ADATABUNDLEEND && p->as != ANOP)) {
				if(p->as == ATEXT) {
					ctxt->autosize = p->to.offset + 4;
					continue;
				}
				ctxt->diag("zero-width instruction\n%P", p);
				continue;
			}
		}
		cursym->size = c;
	} while(bflag);
	if(c % 4 != 0) {
		ctxt->diag("sym->size=%d, invalid", c);
	}

	/*
	 * lay out the code.  all the pc-relative code references,
	 * even cross-function, are resolved now;
	 * only data references need to be relocated.
	 * with more work we could leave cross-function
	 * code references to be relocated too, and then
	 * perhaps we'd be able to parallelize the span loop above.
	 */
	if(ctxt->tlsg == nil)
		ctxt->tlsg = linklookup(ctxt, "runtime.tlsg", 0);

	p = cursym->text;
	ctxt->autosize = p->to.offset + 4;
	symgrow(ctxt, cursym, cursym->size);

	bp = cursym->p;
	c = p->pc; // even p->link might need extra padding
	for(p = p->link; p != nil; p = p->link) {
		ctxt->pc = p->pc;
		ctxt->curp = p;
		o = oplook(ctxt, p);
		opc = p->pc;
		if(ctxt->headtype != Hnacl) {
			asmout(ctxt, p, o, out);
			m = o->size;
		} else {
			m = asmoutnacl(ctxt, c, p, o, out);
			if(opc != p->pc)
				ctxt->diag("asmoutnacl broken: pc changed (%d->%d) in last stage: %P", opc, (int32)p->pc, p);
		}
		if(m % 4 != 0 || p->pc % 4 != 0) {
			ctxt->diag("final stage: pc invalid: %P size=%d", p, m);
		}
		if(c > p->pc)
			ctxt->diag("PC padding invalid: want %#lld, has %#d: %P", p->pc, c, p);
		while(c != p->pc) {
			// emit 0xe1a00000 (MOVW R0, R0)
			*bp++ = 0x00;
			*bp++ = 0x00;
			*bp++ = 0xa0;
			*bp++ = 0xe1;
			c += 4;
		}
		for(i=0; i<m/4; i++) {
			v = out[i];
			*bp++ = v;
			*bp++ = v>>8;
			*bp++ = v>>16;
			*bp++ = v>>24;
		}
		c += m;
	}
}

/*
 * when the first reference to the literal pool threatens
 * to go out of range of a 12-bit PC-relative offset,
 * drop the pool now, and branch round it.
 * this happens only in extended basic blocks that exceed 4k.
 */
static int
checkpool(Link *ctxt, Prog *p, int sz)
{
	if(pool.size >= 0xff0 || immaddr((p->pc+sz+4)+4+(12+pool.size) - (pool.start+8)) == 0)
		return flushpool(ctxt, p, 1, 0);
	else if(p->link == nil)
		return flushpool(ctxt, p, 2, 0);
	return 0;
}

static int
flushpool(Link *ctxt, Prog *p, int skip, int force)
{
	Prog *q;

	if(ctxt->blitrl) {
		if(skip){
			if(0 && skip==1)print("note: flush literal pool at %llux: len=%ud ref=%ux\n", p->pc+4, pool.size, pool.start);
			q = emallocz(sizeof(Prog));
			q->as = AB;
			q->to.type = TYPE_BRANCH;
			q->pcond = p->link;
			q->link = ctxt->blitrl;
			q->lineno = p->lineno;
			ctxt->blitrl = q;
		}
		else if(!force && (p->pc+(12+pool.size)-pool.start < 2048)) // 12 take into account the maximum nacl literal pool alignment padding size
			return 0;
		if(ctxt->headtype == Hnacl && pool.size % 16 != 0) {
			// if pool is not multiple of 16 bytes, add an alignment marker
			q = emallocz(sizeof(Prog));
			q->as = ADATABUNDLEEND;
			ctxt->elitrl->link = q;
			ctxt->elitrl = q;
		}
		ctxt->elitrl->link = p->link;
		p->link = ctxt->blitrl;
		// BUG(minux): how to correctly handle line number for constant pool entries?
		// for now, we set line number to the last instruction preceding them at least
		// this won't bloat the .debug_line tables
		while(ctxt->blitrl) {
			ctxt->blitrl->lineno = p->lineno;
			ctxt->blitrl = ctxt->blitrl->link;
		}
		ctxt->blitrl = 0;	/* BUG: should refer back to values until out-of-range */
		ctxt->elitrl = 0;
		pool.size = 0;
		pool.start = 0;
		pool.extra = 0;
		return 1;
	}
	return 0;
}

static void
addpool(Link *ctxt, Prog *p, Addr *a)
{
	Prog *q, t;
	int c;

	c = aclass(ctxt, a);

	t = zprog;
	t.as = AWORD;

	switch(c) {
	default:
		t.to.offset = a->offset;
		t.to.sym = a->sym;
		t.to.type = a->type;
		t.to.name = a->name;
		
		if(ctxt->flag_shared && t.to.sym != nil)
			t.pcrel = p;
		break;

	case C_SROREG:
	case C_LOREG:
	case C_ROREG:
	case C_FOREG:
	case C_SOREG:
	case C_HOREG:
	case C_FAUTO:
	case C_SAUTO:
	case C_LAUTO:
	case C_LACON:
		t.to.type = TYPE_CONST;
		t.to.offset = ctxt->instoffset;
		break;
	}

	if(t.pcrel == nil) {
		for(q = ctxt->blitrl; q != nil; q = q->link)	/* could hash on t.t0.offset */
			if(q->pcrel == nil && memcmp(&q->to, &t.to, sizeof(t.to)) == 0) {
				p->pcond = q;
				return;
			}
	}

	if(ctxt->headtype == Hnacl && pool.size%16 == 0) {
		// start a new data bundle
		q = emallocz(sizeof(Prog));
		*q = zprog;
		q->as = ADATABUNDLE;
		q->pc = pool.size;
		pool.size += 4;
		if(ctxt->blitrl == nil) {
			ctxt->blitrl = q;
			pool.start = p->pc;
		} else {
			ctxt->elitrl->link = q;
		}
		ctxt->elitrl = q;
	}

	q = emallocz(sizeof(Prog));
	*q = t;
	q->pc = pool.size;

	if(ctxt->blitrl == nil) {
		ctxt->blitrl = q;
		pool.start = p->pc;
	} else
		ctxt->elitrl->link = q;
	ctxt->elitrl = q;
	pool.size += 4;

	p->pcond = q;
}

static int32
regoff(Link *ctxt, Addr *a)
{

	ctxt->instoffset = 0;
	aclass(ctxt, a);
	return ctxt->instoffset;
}

static int32
immrot(uint32 v)
{
	int i;

	for(i=0; i<16; i++) {
		if((v & ~0xff) == 0)
			return (i<<8) | v | (1<<25);
		v = (v<<2) | (v>>30);
	}
	return 0;
}

static int32
immaddr(int32 v)
{
	if(v >= 0 && v <= 0xfff)
		return (v & 0xfff) |
			(1<<24) |	/* pre indexing */
			(1<<23);	/* pre indexing, up */
	if(v >= -0xfff && v < 0)
		return (-v & 0xfff) |
			(1<<24);	/* pre indexing */
	return 0;
}

static int
immfloat(int32 v)
{
	return (v & 0xC03) == 0;	/* offset will fit in floating-point load/store */
}

static int
immhalf(int32 v)
{
	if(v >= 0 && v <= 0xff)
		return v|
			(1<<24)|	/* pre indexing */
			(1<<23);	/* pre indexing, up */
	if(v >= -0xff && v < 0)
		return (-v & 0xff)|
			(1<<24);	/* pre indexing */
	return 0;
}

static int aconsize(Link *ctxt);

static int
aclass(Link *ctxt, Addr *a)
{
	LSym *s;
	int t;

	switch(a->type) {
	case TYPE_NONE:
		return C_NONE;

	case TYPE_REG:
		if(REG_R0 <= a->reg && a->reg <= REG_R15)
			return C_REG;
		if(REG_F0 <= a->reg && a->reg <= REG_F15)
			return C_FREG;
		if(a->reg == REG_FPSR || a->reg == REG_FPCR)
			return C_FCR;
		if(a->reg == REG_CPSR || a->reg == REG_SPSR)
			return C_PSR;
		return C_GOK;

	case TYPE_REGREG:
		return C_REGREG;

	case TYPE_REGREG2:
		return C_REGREG2;

	case TYPE_SHIFT:
		return C_SHIFT;

	case TYPE_MEM:
		switch(a->name) {
		case NAME_EXTERN:
		case NAME_STATIC:
			if(a->sym == 0 || a->sym->name == 0) {
				print("null sym external\n");
				return C_GOK;
			}
			ctxt->instoffset = 0;	// s.b. unused but just in case
			return C_ADDR;

		case NAME_AUTO:
			ctxt->instoffset = ctxt->autosize + a->offset;
			t = immaddr(ctxt->instoffset);
			if(t){
				if(immhalf(ctxt->instoffset)) {
					if(immfloat(t))
						return C_HFAUTO;
					return C_HAUTO;
				}
				if(immfloat(t))
					return C_FAUTO;
				return C_SAUTO;
			}
			return C_LAUTO;

		case NAME_PARAM:
			ctxt->instoffset = ctxt->autosize + a->offset + 4L;
			t = immaddr(ctxt->instoffset);
			if(t){
				if(immhalf(ctxt->instoffset)) {
					if(immfloat(t))
						return C_HFAUTO;
					return C_HAUTO;
				}
				if(immfloat(t))
					return C_FAUTO;
				return C_SAUTO;
			}
			return C_LAUTO;
		case TYPE_NONE:
			ctxt->instoffset = a->offset;
			t = immaddr(ctxt->instoffset);
			if(t) {
				if(immhalf(ctxt->instoffset)) {		 /* n.b. that it will also satisfy immrot */
					if(immfloat(t))
						return C_HFOREG;
					return C_HOREG;
				}
				if(immfloat(t))
					return C_FOREG; /* n.b. that it will also satisfy immrot */
				t = immrot(ctxt->instoffset);
				if(t)
					return C_SROREG;
				if(immhalf(ctxt->instoffset))
					return C_HOREG;
				return C_SOREG;
			}
			t = immrot(ctxt->instoffset);
			if(t)
				return C_ROREG;
			return C_LOREG;
		}
		return C_GOK;

	case TYPE_FCONST:
		if(chipzero5(ctxt, a->u.dval) >= 0)
			return C_ZFCON;
		if(chipfloat5(ctxt, a->u.dval) >= 0)
			return C_SFCON;
		return C_LFCON;

	case TYPE_TEXTSIZE:
		return C_TEXTSIZE;

	case TYPE_CONST:
	case TYPE_ADDR:
		switch(a->name) {

		case TYPE_NONE:
			ctxt->instoffset = a->offset;
			if(a->reg != 0)
				return aconsize(ctxt);

			t = immrot(ctxt->instoffset);
			if(t)
				return C_RCON;
			t = immrot(~ctxt->instoffset);
			if(t)
				return C_NCON;
			return C_LCON;

		case NAME_EXTERN:
		case NAME_STATIC:
			s = a->sym;
			if(s == nil)
				break;
			ctxt->instoffset = 0;	// s.b. unused but just in case
			return C_LCONADDR;

		case NAME_AUTO:
			ctxt->instoffset = ctxt->autosize + a->offset;
			return aconsize(ctxt);

		case NAME_PARAM:
			ctxt->instoffset = ctxt->autosize + a->offset + 4L;
			return aconsize(ctxt);
		}
		return C_GOK;

	case TYPE_BRANCH:
		return C_SBRA;
	}
	return C_GOK;
}

static int
aconsize(Link *ctxt)
{
	int t;

	t = immrot(ctxt->instoffset);
	if(t)
		return C_RACON;
	return C_LACON;
}

static void
prasm(Prog *p)
{
	print("%P\n", p);
}

static Optab*
oplook(Link *ctxt, Prog *p)
{
	int a1, a2, a3, r;
	uchar *c1, *c3;
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
	a3 = p->to.class;
	if(a3 == 0) {
		a3 = aclass(ctxt, &p->to) + 1;
		p->to.class = a3;
	}
	a3--;
	a2 = C_NONE;
	if(p->reg != 0)
		a2 = C_REG;
	r = p->as;
	o = oprange[r].start;
	if(o == 0) {
		o = oprange[r].stop; /* just generate an error */
	}
	if(0 /*debug['O']*/) {
		print("oplook %A %^ %^ %^\n",
			(int)p->as, a1, a2, a3);
		print("		%d %d\n", p->from.type, p->to.type);
	}
	e = oprange[r].stop;
	c1 = xcmp[a1];
	c3 = xcmp[a3];
	for(; o<e; o++)
		if(o->a2 == a2)
		if(c1[o->a1])
		if(c3[o->a3]) {
			p->optab = (o-optab)+1;
			return o;
		}
	ctxt->diag("illegal combination %P; %^ %^ %^, %d %d",
		p, a1, a2, a3, p->from.type, p->to.type);
	ctxt->diag("from %d %d to %d %d\n", p->from.type, p->from.name, p->to.type, p->to.name);
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
		if(b == C_RCON || b == C_NCON)
			return 1;
		break;
	case C_LACON:
		if(b == C_RACON)
			return 1;
		break;
	case C_LFCON:
		if(b == C_ZFCON || b == C_SFCON)
			return 1;
		break;

	case C_HFAUTO:
		return b == C_HAUTO || b == C_FAUTO;
	case C_FAUTO:
	case C_HAUTO:
		return b == C_HFAUTO;
	case C_SAUTO:
		return cmp(C_HFAUTO, b);
	case C_LAUTO:
		return cmp(C_SAUTO, b);

	case C_HFOREG:
		return b == C_HOREG || b == C_FOREG;
	case C_FOREG:
	case C_HOREG:
		return b == C_HFOREG;
	case C_SROREG:
		return cmp(C_SOREG, b) || cmp(C_ROREG, b);
	case C_SOREG:
	case C_ROREG:
		return b == C_SROREG || cmp(C_HFOREG, b);
	case C_LOREG:
		return cmp(C_SROREG, b);

	case C_LBRA:
		if(b == C_SBRA)
			return 1;
		break;

	case C_HREG:
		return cmp(C_SP, b) || cmp(C_PC, b);

	}
	return 0;
}

static int
ocmp(const void *a1, const void *a2)
{
	Optab *p1, *p2;
	int n;

	p1 = (Optab*)a1;
	p2 = (Optab*)a2;
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
	return 0;
}

static void
buildop(Link *ctxt)
{
	int i, n, r;

	for(i=0; i<C_GOK; i++)
		for(n=0; n<C_GOK; n++)
			xcmp[i][n] = cmp(n, i);
	for(n=0; optab[n].as != AXXX; n++) {
		if((optab[n].flag & LPCREL) != 0) {
			if(ctxt->flag_shared)
				optab[n].size += optab[n].pcrelsiz;
			else
				optab[n].flag &= ~LPCREL;
		}
	}
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
		case AADD:
			oprange[AAND] = oprange[r];
			oprange[AEOR] = oprange[r];
			oprange[ASUB] = oprange[r];
			oprange[ARSB] = oprange[r];
			oprange[AADC] = oprange[r];
			oprange[ASBC] = oprange[r];
			oprange[ARSC] = oprange[r];
			oprange[AORR] = oprange[r];
			oprange[ABIC] = oprange[r];
			break;
		case ACMP:
			oprange[ATEQ] = oprange[r];
			oprange[ACMN] = oprange[r];
			break;
		case AMVN:
			break;
		case ABEQ:
			oprange[ABNE] = oprange[r];
			oprange[ABCS] = oprange[r];
			oprange[ABHS] = oprange[r];
			oprange[ABCC] = oprange[r];
			oprange[ABLO] = oprange[r];
			oprange[ABMI] = oprange[r];
			oprange[ABPL] = oprange[r];
			oprange[ABVS] = oprange[r];
			oprange[ABVC] = oprange[r];
			oprange[ABHI] = oprange[r];
			oprange[ABLS] = oprange[r];
			oprange[ABGE] = oprange[r];
			oprange[ABLT] = oprange[r];
			oprange[ABGT] = oprange[r];
			oprange[ABLE] = oprange[r];
			break;
		case ASLL:
			oprange[ASRL] = oprange[r];
			oprange[ASRA] = oprange[r];
			break;
		case AMUL:
			oprange[AMULU] = oprange[r];
			break;
		case ADIV:
			oprange[AMOD] = oprange[r];
			oprange[AMODU] = oprange[r];
			oprange[ADIVU] = oprange[r];
			break;
		case AMOVW:
		case AMOVB:
		case AMOVBS:
		case AMOVBU:
		case AMOVH:
		case AMOVHS:
		case AMOVHU:
			break;
		case ASWPW:
			oprange[ASWPBU] = oprange[r];
			break;
		case AB:
		case ABL:
		case ABX:
		case ABXRET:
		case ADUFFZERO:
		case ADUFFCOPY:
		case ASWI:
		case AWORD:
		case AMOVM:
		case ARFE:
		case ATEXT:
		case AUSEFIELD:
		case ACASE:
		case ABCASE:
		case ATYPE:
			break;
		case AADDF:
			oprange[AADDD] = oprange[r];
			oprange[ASUBF] = oprange[r];
			oprange[ASUBD] = oprange[r];
			oprange[AMULF] = oprange[r];
			oprange[AMULD] = oprange[r];
			oprange[ADIVF] = oprange[r];
			oprange[ADIVD] = oprange[r];
			oprange[ASQRTF] = oprange[r];
			oprange[ASQRTD] = oprange[r];
			oprange[AMOVFD] = oprange[r];
			oprange[AMOVDF] = oprange[r];
			oprange[AABSF] = oprange[r];
			oprange[AABSD] = oprange[r];
			break;

		case ACMPF:
			oprange[ACMPD] = oprange[r];
			break;

		case AMOVF:
			oprange[AMOVD] = oprange[r];
			break;

		case AMOVFW:
			oprange[AMOVDW] = oprange[r];
			break;

		case AMOVWF:
			oprange[AMOVWD] = oprange[r];
			break;

		case AMULL:
			oprange[AMULAL] = oprange[r];
			oprange[AMULLU] = oprange[r];
			oprange[AMULALU] = oprange[r];
			break;

		case AMULWT:
			oprange[AMULWB] = oprange[r];
			break;

		case AMULAWT:
			oprange[AMULAWB] = oprange[r];
			break;

		case AMULA:
		case ALDREX:
		case ASTREX:
		case ALDREXD:
		case ASTREXD:
		case ATST:
		case APLD:
		case AUNDEF:
		case ACLZ:
		case AFUNCDATA:
		case APCDATA:
		case ANOP:
		case ADATABUNDLE:
		case ADATABUNDLEEND:
			break;
		}
	}
}

static uint32 mov(Link*, Prog*);

static void
asmout(Link *ctxt, Prog *p, Optab *o, uint32 *out)
{
	uint32 o1, o2, o3, o4, o5, o6;
	int32 v;
	int r, rf, rt, rt2;
	Reloc *rel;

ctxt->printp = p;
	o1 = 0;
	o2 = 0;
	o3 = 0;
	o4 = 0;
	o5 = 0;
	o6 = 0;
	ctxt->armsize += o->size;
if(0 /*debug['P']*/) print("%ux: %P	type %d\n", (uint32)(p->pc), p, o->type);
	switch(o->type) {
	default:
		ctxt->diag("unknown asm %d", o->type);
		prasm(p);
		break;

	case 0:		/* pseudo ops */
if(0 /*debug['G']*/) print("%ux: %s: arm %d\n", (uint32)(p->pc), p->from.sym->name, p->from.sym->fnptr);
		break;

	case 1:		/* op R,[R],R */
		o1 = oprrr(ctxt, p->as, p->scond);
		rf = p->from.reg;
		rt = p->to.reg;
		r = p->reg;
		if(p->to.type == TYPE_NONE)
			rt = 0;
		if(p->as == AMOVB || p->as == AMOVH || p->as == AMOVW || p->as == AMVN)
			r = 0;
		else
		if(r == 0)
			r = rt;
		o1 |= ((rf&15)<<0) | ((r&15)<<16) | ((rt&15)<<12);
		break;

	case 2:		/* movbu $I,[R],R */
		aclass(ctxt, &p->from);
		o1 = oprrr(ctxt, p->as, p->scond);
		o1 |= immrot(ctxt->instoffset);
		rt = p->to.reg;
		r = p->reg;
		if(p->to.type == TYPE_NONE)
			rt = 0;
		if(p->as == AMOVW || p->as == AMVN)
			r = 0;
		else if(r == 0)
			r = rt;
		o1 |= ((r&15)<<16) | ((rt&15)<<12);
		break;

	case 3:		/* add R<<[IR],[R],R */
		o1 = mov(ctxt, p);
		break;

	case 4:		/* add $I,[R],R */
		aclass(ctxt, &p->from);
		o1 = oprrr(ctxt, AADD, p->scond);
		o1 |= immrot(ctxt->instoffset);
		r = p->from.reg;
		if(r == 0)
			r = o->param;
		o1 |= (r&15) << 16;
		o1 |= (p->to.reg&15) << 12;
		break;

	case 5:		/* bra s */
		o1 = opbra(ctxt, p->as, p->scond);
		v = -8;
		if(p->to.sym != nil) {
			rel = addrel(ctxt->cursym);
			rel->off = ctxt->pc;
			rel->siz = 4;
			rel->sym = p->to.sym;
			v += p->to.offset;
			rel->add = o1 | ((v >> 2) & 0xffffff);
			rel->type = R_CALLARM;
			break;
		}
		if(p->pcond != nil)
			v = (p->pcond->pc - ctxt->pc) - 8;
		o1 |= (v >> 2) & 0xffffff;
		break;

	case 6:		/* b ,O(R) -> add $O,R,PC */
		aclass(ctxt, &p->to);
		o1 = oprrr(ctxt, AADD, p->scond);
		o1 |= immrot(ctxt->instoffset);
		o1 |= (p->to.reg&15) << 16;
		o1 |= (REGPC&15) << 12;
		break;

	case 7:		/* bl (R) -> blx R */
		aclass(ctxt, &p->to);
		if(ctxt->instoffset != 0)
			ctxt->diag("%P: doesn't support BL offset(REG) where offset != 0", p);
		o1 = oprrr(ctxt, ABL, p->scond);
		o1 |= (p->to.reg&15) << 0;
		rel = addrel(ctxt->cursym);
		rel->off = ctxt->pc;
		rel->siz = 0;
		rel->type = R_CALLIND;
		break;

	case 8:		/* sll $c,[R],R -> mov (R<<$c),R */
		aclass(ctxt, &p->from);
		o1 = oprrr(ctxt, p->as, p->scond);
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		o1 |= (r&15) << 0;
		o1 |= (ctxt->instoffset&31) << 7;
		o1 |= (p->to.reg&15) << 12;
		break;

	case 9:		/* sll R,[R],R -> mov (R<<R),R */
		o1 = oprrr(ctxt, p->as, p->scond);
		r = p->reg;
		if(r == 0)
			r = p->to.reg;
		o1 |= (r&15) << 0;
		o1 |= ((p->from.reg&15) << 8) | (1<<4);
		o1 |= (p->to.reg&15) << 12;
		break;

	case 10:	/* swi [$con] */
		o1 = oprrr(ctxt, p->as, p->scond);
		if(p->to.type != TYPE_NONE) {
			aclass(ctxt, &p->to);
			o1 |= ctxt->instoffset & 0xffffff;
		}
		break;

	case 11:	/* word */
		aclass(ctxt, &p->to);
		o1 = ctxt->instoffset;
		if(p->to.sym != nil) {
			// This case happens with words generated
			// in the PC stream as part of the literal pool.
			rel = addrel(ctxt->cursym);
			rel->off = ctxt->pc;
			rel->siz = 4;
			rel->sym = p->to.sym;
			rel->add = p->to.offset;
			
			// runtime.tlsg is special.
			// Its "address" is the offset from the TLS thread pointer
			// to the thread-local g and m pointers.
			// Emit a TLS relocation instead of a standard one if it's
			// typed STLSBSS.
			if(rel->sym == ctxt->tlsg && ctxt->tlsg->type == STLSBSS) {
				rel->type = R_TLS;
				if(ctxt->flag_shared)
					rel->add += ctxt->pc - p->pcrel->pc - 8 - rel->siz;
				rel->xadd = rel->add;
				rel->xsym = rel->sym;
			} else if(ctxt->flag_shared) {
				rel->type = R_PCREL;
				rel->add += ctxt->pc - p->pcrel->pc - 8;
			} else
				rel->type = R_ADDR;
			o1 = 0;
		}
		break;

	case 12:	/* movw $lcon, reg */
		o1 = omvl(ctxt, p, &p->from, p->to.reg);
		if(o->flag & LPCREL) {
			o2 = oprrr(ctxt, AADD, p->scond) | (p->to.reg&15) << 0 | (REGPC&15) << 16 | (p->to.reg&15) << 12;
		}
		break;

	case 13:	/* op $lcon, [R], R */
		o1 = omvl(ctxt, p, &p->from, REGTMP);
		if(!o1)
			break;
		o2 = oprrr(ctxt, p->as, p->scond);
		o2 |= (REGTMP&15);
		r = p->reg;
		if(p->as == AMOVW || p->as == AMVN)
			r = 0;
		else if(r == 0)
			r = p->to.reg;
		o2 |= (r&15) << 16;
		if(p->to.type != TYPE_NONE)
			o2 |= (p->to.reg&15) << 12;
		break;

	case 14:	/* movb/movbu/movh/movhu R,R */
		o1 = oprrr(ctxt, ASLL, p->scond);

		if(p->as == AMOVBU || p->as == AMOVHU)
			o2 = oprrr(ctxt, ASRL, p->scond);
		else
			o2 = oprrr(ctxt, ASRA, p->scond);

		r = p->to.reg;
		o1 |= ((p->from.reg&15)<<0)|((r&15)<<12);
		o2 |= (r&15)|((r&15)<<12);
		if(p->as == AMOVB || p->as == AMOVBS || p->as == AMOVBU) {
			o1 |= (24<<7);
			o2 |= (24<<7);
		} else {
			o1 |= (16<<7);
			o2 |= (16<<7);
		}
		break;

	case 15:	/* mul r,[r,]r */
		o1 = oprrr(ctxt, p->as, p->scond);
		rf = p->from.reg;
		rt = p->to.reg;
		r = p->reg;
		if(r == 0)
			r = rt;
		if(rt == r) {
			r = rf;
			rf = rt;
		}
		if(0)
		if(rt == r || rf == (REGPC&15) || r == (REGPC&15) || rt == (REGPC&15)) {
			ctxt->diag("bad registers in MUL");
			prasm(p);
		}
		o1 |= ((rf&15)<<8) | ((r&15)<<0) | ((rt&15)<<16);
		break;


	case 16:	/* div r,[r,]r */
		o1 = 0xf << 28;
		o2 = 0;
		break;

	case 17:
		o1 = oprrr(ctxt, p->as, p->scond);
		rf = p->from.reg;
		rt = p->to.reg;
		rt2 = p->to.offset;
		r = p->reg;
		o1 |= ((rf&15)<<8) | ((r&15)<<0) | ((rt&15)<<16) | ((rt2&15)<<12);
		break;

	case 20:	/* mov/movb/movbu R,O(R) */
		aclass(ctxt, &p->to);
		r = p->to.reg;
		if(r == 0)
			r = o->param;
		o1 = osr(ctxt, p->as, p->from.reg, ctxt->instoffset, r, p->scond);
		break;

	case 21:	/* mov/movbu O(R),R -> lr */
		aclass(ctxt, &p->from);
		r = p->from.reg;
		if(r == 0)
			r = o->param;
		o1 = olr(ctxt, ctxt->instoffset, r, p->to.reg, p->scond);
		if(p->as != AMOVW)
			o1 |= 1<<22;
		break;

	case 30:	/* mov/movb/movbu R,L(R) */
		o1 = omvl(ctxt, p, &p->to, REGTMP);
		if(!o1)
			break;
		r = p->to.reg;
		if(r == 0)
			r = o->param;
		o2 = osrr(ctxt, p->from.reg, REGTMP&15, r, p->scond);
		if(p->as != AMOVW)
			o2 |= 1<<22;
		break;

	case 31:	/* mov/movbu L(R),R -> lr[b] */
		o1 = omvl(ctxt, p, &p->from, REGTMP);
		if(!o1)
			break;
		r = p->from.reg;
		if(r == 0)
			r = o->param;
		o2 = olrr(ctxt, REGTMP&15, r, p->to.reg, p->scond);
		if(p->as == AMOVBU || p->as == AMOVBS || p->as == AMOVB)
			o2 |= 1<<22;
		break;

	case 34:	/* mov $lacon,R */
		o1 = omvl(ctxt, p, &p->from, REGTMP);
		if(!o1)
			break;

		o2 = oprrr(ctxt, AADD, p->scond);
		o2 |= (REGTMP&15);
		r = p->from.reg;
		if(r == 0)
			r = o->param;
		o2 |= (r&15) << 16;
		if(p->to.type != TYPE_NONE)
			o2 |= (p->to.reg&15) << 12;
		break;

	case 35:	/* mov PSR,R */
		o1 = (2<<23) | (0xf<<16) | (0<<0);
		o1 |= ((p->scond & C_SCOND) ^ C_SCOND_XOR) << 28;
		o1 |= (p->from.reg & 1) << 22;
		o1 |= (p->to.reg&15) << 12;
		break;

	case 36:	/* mov R,PSR */
		o1 = (2<<23) | (0x29f<<12) | (0<<4);
		if(p->scond & C_FBIT)
			o1 ^= 0x010 << 12;
		o1 |= ((p->scond & C_SCOND) ^ C_SCOND_XOR) << 28;
		o1 |= (p->to.reg & 1) << 22;
		o1 |= (p->from.reg&15) << 0;
		break;

	case 37:	/* mov $con,PSR */
		aclass(ctxt, &p->from);
		o1 = (2<<23) | (0x29f<<12) | (0<<4);
		if(p->scond & C_FBIT)
			o1 ^= 0x010 << 12;
		o1 |= ((p->scond & C_SCOND) ^ C_SCOND_XOR) << 28;
		o1 |= immrot(ctxt->instoffset);
		o1 |= (p->to.reg & 1) << 22;
		o1 |= (p->from.reg&15) << 0;
		break;

	case 38:
	case 39:
		switch(o->type) {
		case 38:	/* movm $con,oreg -> stm */
			o1 = (0x4 << 25);
			o1 |= p->from.offset & 0xffff;
			o1 |= (p->to.reg&15) << 16;
			aclass(ctxt, &p->to);
			break;
	
		case 39:	/* movm oreg,$con -> ldm */
			o1 = (0x4 << 25) | (1 << 20);
			o1 |= p->to.offset & 0xffff;
			o1 |= (p->from.reg&15) << 16;
			aclass(ctxt, &p->from);
			break;
		}
		if(ctxt->instoffset != 0)
			ctxt->diag("offset must be zero in MOVM; %P", p);
		o1 |= ((p->scond & C_SCOND) ^ C_SCOND_XOR) << 28;
		if(p->scond & C_PBIT)
			o1 |= 1 << 24;
		if(p->scond & C_UBIT)
			o1 |= 1 << 23;
		if(p->scond & C_SBIT)
			o1 |= 1 << 22;
		if(p->scond & C_WBIT)
			o1 |= 1 << 21;
		break;

	case 40:	/* swp oreg,reg,reg */
		aclass(ctxt, &p->from);
		if(ctxt->instoffset != 0)
			ctxt->diag("offset must be zero in SWP");
		o1 = (0x2<<23) | (0x9<<4);
		if(p->as != ASWPW)
			o1 |= 1 << 22;
		o1 |= (p->from.reg&15) << 16;
		o1 |= (p->reg&15) << 0;
		o1 |= (p->to.reg&15) << 12;
		o1 |= ((p->scond & C_SCOND) ^ C_SCOND_XOR) << 28;
		break;

	case 41:	/* rfe -> movm.s.w.u 0(r13),[r15] */
		o1 = 0xe8fd8000;
		break;

	case 50:	/* floating point store */
		v = regoff(ctxt, &p->to);
		r = p->to.reg;
		if(r == 0)
			r = o->param;
		o1 = ofsr(ctxt, p->as, p->from.reg, v, r, p->scond, p);
		break;

	case 51:	/* floating point load */
		v = regoff(ctxt, &p->from);
		r = p->from.reg;
		if(r == 0)
			r = o->param;
		o1 = ofsr(ctxt, p->as, p->to.reg, v, r, p->scond, p) | (1<<20);
		break;

	case 52:	/* floating point store, int32 offset UGLY */
		o1 = omvl(ctxt, p, &p->to, REGTMP);
		if(!o1)
			break;
		r = p->to.reg;
		if(r == 0)
			r = o->param;
		o2 = oprrr(ctxt, AADD, p->scond) | ((REGTMP&15) << 12) | ((REGTMP&15) << 16) | ((r&15) << 0);
		o3 = ofsr(ctxt, p->as, p->from.reg, 0, REGTMP, p->scond, p);
		break;

	case 53:	/* floating point load, int32 offset UGLY */
		o1 = omvl(ctxt, p, &p->from, REGTMP);
		if(!o1)
			break;
		r = p->from.reg;
		if(r == 0)
			r = o->param;
		o2 = oprrr(ctxt, AADD, p->scond) | ((REGTMP&15) << 12) | ((REGTMP&15) << 16) | ((r&15) << 0);
		o3 = ofsr(ctxt, p->as, p->to.reg, 0, (REGTMP&15), p->scond, p) | (1<<20);
		break;

	case 54:	/* floating point arith */
		o1 = oprrr(ctxt, p->as, p->scond);
		rf = p->from.reg;
		rt = p->to.reg;
		r = p->reg;
		if(r == 0) {
			r = rt;
			if(p->as == AMOVF || p->as == AMOVD || p->as == AMOVFD || p->as == AMOVDF ||
				p->as == ASQRTF || p->as == ASQRTD || p->as == AABSF || p->as == AABSD)
				r = 0;
		}
		o1 |= ((rf&15)<<0) | ((r&15)<<16) | ((rt&15)<<12);
		break;

	case 56:	/* move to FP[CS]R */
		o1 = (((p->scond & C_SCOND) ^ C_SCOND_XOR) << 28) | (0xe << 24) | (1<<8) | (1<<4);
		o1 |= (((p->to.reg&1)+1)<<21) | ((p->from.reg&15) << 12);
		break;

	case 57:	/* move from FP[CS]R */
		o1 = (((p->scond & C_SCOND) ^ C_SCOND_XOR) << 28) | (0xe << 24) | (1<<8) | (1<<4);
		o1 |= (((p->from.reg&1)+1)<<21) | ((p->to.reg&15)<<12) | (1<<20);
		break;
	case 58:	/* movbu R,R */
		o1 = oprrr(ctxt, AAND, p->scond);
		o1 |= immrot(0xff);
		rt = p->to.reg;
		r = p->from.reg;
		if(p->to.type == TYPE_NONE)
			rt = 0;
		if(r == 0)
			r = rt;
		o1 |= ((r&15)<<16) | ((rt&15)<<12);
		break;

	case 59:	/* movw/bu R<<I(R),R -> ldr indexed */
		if(p->from.reg == 0) {
			if(p->as != AMOVW)
				ctxt->diag("byte MOV from shifter operand");
			o1 = mov(ctxt, p);
			break;
		}
		if(p->from.offset&(1<<4))
			ctxt->diag("bad shift in LDR");
		o1 = olrr(ctxt, p->from.offset, p->from.reg, p->to.reg, p->scond);
		if(p->as == AMOVBU)
			o1 |= 1<<22;
		break;

	case 60:	/* movb R(R),R -> ldrsb indexed */
		if(p->from.reg == 0) {
			ctxt->diag("byte MOV from shifter operand");
			o1 = mov(ctxt, p);
			break;
		}
		if(p->from.offset&(~0xf))
			ctxt->diag("bad shift in LDRSB");
		o1 = olhrr(ctxt, p->from.offset, p->from.reg, p->to.reg, p->scond);
		o1 ^= (1<<5)|(1<<6);
		break;

	case 61:	/* movw/b/bu R,R<<[IR](R) -> str indexed */
		if(p->to.reg == 0)
			ctxt->diag("MOV to shifter operand");
		o1 = osrr(ctxt, p->from.reg, p->to.offset, p->to.reg, p->scond);
		if(p->as == AMOVB || p->as == AMOVBS || p->as == AMOVBU)
			o1 |= 1<<22;
		break;

	case 62:	/* case R -> movw	R<<2(PC),PC */
		if(o->flag & LPCREL) {
			o1 = oprrr(ctxt, AADD, p->scond) | immrot(1) | (p->from.reg&15) << 16 | (REGTMP&15) << 12;
			o2 = olrr(ctxt, REGTMP&15, REGPC, REGTMP, p->scond);
			o2 |= 2<<7;
			o3 = oprrr(ctxt, AADD, p->scond) | (REGTMP&15) | (REGPC&15) << 16 | (REGPC&15) << 12;
		} else {
			o1 = olrr(ctxt, p->from.reg&15, REGPC, REGPC, p->scond);
			o1 |= 2<<7;
		}
		break;

	case 63:	/* bcase */
		if(p->pcond != nil) {
			rel = addrel(ctxt->cursym);
			rel->off = ctxt->pc;
			rel->siz = 4;
			if(p->to.sym != nil && p->to.sym->type != 0) {
				rel->sym = p->to.sym;
				rel->add = p->to.offset;
			} else {
				rel->sym = ctxt->cursym;
				rel->add = p->pcond->pc;
			}
			if(o->flag & LPCREL) {
				rel->type = R_PCREL;
				rel->add += ctxt->pc - p->pcrel->pc - 16 + rel->siz;
			} else
				rel->type = R_ADDR;
			o1 = 0;
		}
		break;

	/* reloc ops */
	case 64:	/* mov/movb/movbu R,addr */
		o1 = omvl(ctxt, p, &p->to, REGTMP);
		if(!o1)
			break;
		o2 = osr(ctxt, p->as, p->from.reg, 0, REGTMP, p->scond);
		if(o->flag & LPCREL) {
			o3 = o2;
			o2 = oprrr(ctxt, AADD, p->scond) | (REGTMP&15) | (REGPC&15) << 16 | (REGTMP&15) << 12;
		}
		break;

	case 65:	/* mov/movbu addr,R */
		o1 = omvl(ctxt, p, &p->from, REGTMP);
		if(!o1)
			break;
		o2 = olr(ctxt, 0, REGTMP, p->to.reg, p->scond);
		if(p->as == AMOVBU || p->as == AMOVBS || p->as == AMOVB)
			o2 |= 1<<22;
		if(o->flag & LPCREL) {
			o3 = o2;
			o2 = oprrr(ctxt, AADD, p->scond) | (REGTMP&15) | (REGPC&15) << 16 | (REGTMP&15) << 12;
		}
		break;

	case 68:	/* floating point store -> ADDR */
		o1 = omvl(ctxt, p, &p->to, REGTMP);
		if(!o1)
			break;
		o2 = ofsr(ctxt, p->as, p->from.reg, 0, REGTMP, p->scond, p);
		if(o->flag & LPCREL) {
			o3 = o2;
			o2 = oprrr(ctxt, AADD, p->scond) | (REGTMP&15) | (REGPC&15) << 16 | (REGTMP&15) << 12;
		}
		break;

	case 69:	/* floating point load <- ADDR */
		o1 = omvl(ctxt, p, &p->from, REGTMP);
		if(!o1)
			break;
		o2 = ofsr(ctxt, p->as, p->to.reg, 0, (REGTMP&15), p->scond, p) | (1<<20);
		if(o->flag & LPCREL) {
			o3 = o2;
			o2 = oprrr(ctxt, AADD, p->scond) | (REGTMP&15) | (REGPC&15) << 16 | (REGTMP&15) << 12;
		}
		break;

	/* ArmV4 ops: */
	case 70:	/* movh/movhu R,O(R) -> strh */
		aclass(ctxt, &p->to);
		r = p->to.reg;
		if(r == 0)
			r = o->param;
		o1 = oshr(ctxt, p->from.reg, ctxt->instoffset, r, p->scond);
		break;
	case 71:	/* movb/movh/movhu O(R),R -> ldrsb/ldrsh/ldrh */
		aclass(ctxt, &p->from);
		r = p->from.reg;
		if(r == 0)
			r = o->param;
		o1 = olhr(ctxt, ctxt->instoffset, r, p->to.reg, p->scond);
		if(p->as == AMOVB || p->as == AMOVBS)
			o1 ^= (1<<5)|(1<<6);
		else if(p->as == AMOVH || p->as == AMOVHS)
			o1 ^= (1<<6);
		break;
	case 72:	/* movh/movhu R,L(R) -> strh */
		o1 = omvl(ctxt, p, &p->to, REGTMP);
		if(!o1)
			break;
		r = p->to.reg;
		if(r == 0)
			r = o->param;
		o2 = oshrr(ctxt, p->from.reg, REGTMP&15, r, p->scond);
		break;
	case 73:	/* movb/movh/movhu L(R),R -> ldrsb/ldrsh/ldrh */
		o1 = omvl(ctxt, p, &p->from, REGTMP);
		if(!o1)
			break;
		r = p->from.reg;
		if(r == 0)
			r = o->param;
		o2 = olhrr(ctxt, REGTMP&15, r, p->to.reg, p->scond);
		if(p->as == AMOVB || p->as == AMOVBS)
			o2 ^= (1<<5)|(1<<6);
		else if(p->as == AMOVH || p->as == AMOVHS)
			o2 ^= (1<<6);
		break;
	case 74:	/* bx $I */
		ctxt->diag("ABX $I");
		break;
	case 75:	/* bx O(R) */
		aclass(ctxt, &p->to);
		if(ctxt->instoffset != 0)
			ctxt->diag("non-zero offset in ABX");
/*
		o1 = 	oprrr(ctxt, AADD, p->scond) | immrot(0) | ((REGPC&15)<<16) | ((REGLINK&15)<<12);	// mov PC, LR
		o2 = (((p->scond&C_SCOND) ^ C_SCOND_XOR)<<28) | (0x12fff<<8) | (1<<4) | ((p->to.reg&15) << 0);		// BX R
*/
		// p->to.reg may be REGLINK
		o1 = oprrr(ctxt, AADD, p->scond);
		o1 |= immrot(ctxt->instoffset);
		o1 |= (p->to.reg&15) << 16;
		o1 |= (REGTMP&15) << 12;
		o2 = oprrr(ctxt, AADD, p->scond) | immrot(0) | ((REGPC&15)<<16) | ((REGLINK&15)<<12);	// mov PC, LR
		o3 = (((p->scond&C_SCOND) ^ C_SCOND_XOR)<<28) | (0x12fff<<8) | (1<<4) | (REGTMP&15);		// BX Rtmp
		break;
	case 76:	/* bx O(R) when returning from fn*/
		ctxt->diag("ABXRET");
		break;
	case 77:	/* ldrex oreg,reg */
		aclass(ctxt, &p->from);
		if(ctxt->instoffset != 0)
			ctxt->diag("offset must be zero in LDREX");
		o1 = (0x19<<20) | (0xf9f);
		o1 |= (p->from.reg&15) << 16;
		o1 |= (p->to.reg&15) << 12;
		o1 |= ((p->scond & C_SCOND) ^ C_SCOND_XOR) << 28;
		break;
	case 78:	/* strex reg,oreg,reg */
		aclass(ctxt, &p->from);
		if(ctxt->instoffset != 0)
			ctxt->diag("offset must be zero in STREX");
		o1 = (0x18<<20) | (0xf90);
		o1 |= (p->from.reg&15) << 16;
		o1 |= (p->reg&15) << 0;
		o1 |= (p->to.reg&15) << 12;
		o1 |= ((p->scond & C_SCOND) ^ C_SCOND_XOR) << 28;
		break;
	case 80:	/* fmov zfcon,freg */
		if(p->as == AMOVD) {
			o1 = 0xeeb00b00;	// VMOV imm 64
			o2 = oprrr(ctxt, ASUBD, p->scond);
		} else {
			o1 = 0x0eb00a00;	// VMOV imm 32
			o2 = oprrr(ctxt, ASUBF, p->scond);
		}
		v = 0x70;	// 1.0
		r = (p->to.reg&15) << 0;

		// movf $1.0, r
		o1 |= ((p->scond & C_SCOND) ^ C_SCOND_XOR) << 28;
		o1 |= (r&15) << 12;
		o1 |= (v&0xf) << 0;
		o1 |= (v&0xf0) << 12;

		// subf r,r,r
		o2 |= ((r&15)<<0) | ((r&15)<<16) | ((r&15)<<12);
		break;
	case 81:	/* fmov sfcon,freg */
		o1 = 0x0eb00a00;		// VMOV imm 32
		if(p->as == AMOVD)
			o1 = 0xeeb00b00;	// VMOV imm 64
		o1 |= ((p->scond & C_SCOND) ^ C_SCOND_XOR) << 28;
		o1 |= (p->to.reg&15) << 12;
		v = chipfloat5(ctxt, p->from.u.dval);
		o1 |= (v&0xf) << 0;
		o1 |= (v&0xf0) << 12;
		break;
	case 82:	/* fcmp freg,freg, */
		o1 = oprrr(ctxt, p->as, p->scond);
		o1 |= ((p->reg&15)<<12) | ((p->from.reg&15)<<0);
		o2 = 0x0ef1fa10;	// VMRS R15
		o2 |= ((p->scond & C_SCOND) ^ C_SCOND_XOR) << 28;
		break;
	case 83:	/* fcmp freg,, */
		o1 = oprrr(ctxt, p->as, p->scond);
		o1 |= ((p->from.reg&15)<<12) | (1<<16);
		o2 = 0x0ef1fa10;	// VMRS R15
		o2 |= ((p->scond & C_SCOND) ^ C_SCOND_XOR) << 28;
		break;
	case 84:	/* movfw freg,freg - truncate float-to-fix */
		o1 = oprrr(ctxt, p->as, p->scond);
		o1 |= ((p->from.reg&15)<<0);
		o1 |= ((p->to.reg&15)<<12);
		break;
	case 85:	/* movwf freg,freg - fix-to-float */
		o1 = oprrr(ctxt, p->as, p->scond);
		o1 |= ((p->from.reg&15)<<0);
		o1 |= ((p->to.reg&15)<<12);
		break;
	case 86:	/* movfw freg,reg - truncate float-to-fix */
		// macro for movfw freg,FTMP; movw FTMP,reg
		o1 = oprrr(ctxt, p->as, p->scond);
		o1 |= ((p->from.reg&15)<<0);
		o1 |= ((FREGTMP&15)<<12);
		o2 = oprrr(ctxt, AMOVFW+ALAST, p->scond);
		o2 |= ((FREGTMP&15)<<16);
		o2 |= ((p->to.reg&15)<<12);
		break;
	case 87:	/* movwf reg,freg - fix-to-float */
		// macro for movw reg,FTMP; movwf FTMP,freg
		o1 = oprrr(ctxt, AMOVWF+ALAST, p->scond);
		o1 |= ((p->from.reg&15)<<12);
		o1 |= ((FREGTMP&15)<<16);
		o2 = oprrr(ctxt, p->as, p->scond);
		o2 |= ((FREGTMP&15)<<0);
		o2 |= ((p->to.reg&15)<<12);
		break;
	case 88:	/* movw reg,freg  */
		o1 = oprrr(ctxt, AMOVWF+ALAST, p->scond);
		o1 |= ((p->from.reg&15)<<12);
		o1 |= ((p->to.reg&15)<<16);
		break;
	case 89:	/* movw freg,reg  */
		o1 = oprrr(ctxt, AMOVFW+ALAST, p->scond);
		o1 |= ((p->from.reg&15)<<16);
		o1 |= ((p->to.reg&15)<<12);
		break;
	case 90:	/* tst reg  */
		o1 = oprrr(ctxt, ACMP+ALAST, p->scond);
		o1 |= (p->from.reg&15)<<16;
		break;
	case 91:	/* ldrexd oreg,reg */
		aclass(ctxt, &p->from);
		if(ctxt->instoffset != 0)
			ctxt->diag("offset must be zero in LDREX");
		o1 = (0x1b<<20) | (0xf9f);
		o1 |= (p->from.reg&15) << 16;
		o1 |= (p->to.reg&15) << 12;
		o1 |= ((p->scond & C_SCOND) ^ C_SCOND_XOR) << 28;
		break;
	case 92:	/* strexd reg,oreg,reg */
		aclass(ctxt, &p->from);
		if(ctxt->instoffset != 0)
			ctxt->diag("offset must be zero in STREX");
		o1 = (0x1a<<20) | (0xf90);
		o1 |= (p->from.reg&15) << 16;
		o1 |= (p->reg&15) << 0;
		o1 |= (p->to.reg&15) << 12;
		o1 |= ((p->scond & C_SCOND) ^ C_SCOND_XOR) << 28;
		break;
	case 93:	/* movb/movh/movhu addr,R -> ldrsb/ldrsh/ldrh */
		o1 = omvl(ctxt, p, &p->from, REGTMP);
		if(!o1)
			break;
		o2 = olhr(ctxt, 0, REGTMP, p->to.reg, p->scond);
		if(p->as == AMOVB || p->as == AMOVBS)
			o2 ^= (1<<5)|(1<<6);
		else if(p->as == AMOVH || p->as == AMOVHS)
			o2 ^= (1<<6);
		if(o->flag & LPCREL) {
			o3 = o2;
			o2 = oprrr(ctxt, AADD, p->scond) | (REGTMP&15) | (REGPC&15) << 16 | (REGTMP&15) << 12;
		}
		break;
	case 94:	/* movh/movhu R,addr -> strh */
		o1 = omvl(ctxt, p, &p->to, REGTMP);
		if(!o1)
			break;
		o2 = oshr(ctxt, p->from.reg, 0, REGTMP, p->scond);
		if(o->flag & LPCREL) {
			o3 = o2;
			o2 = oprrr(ctxt, AADD, p->scond) | (REGTMP&15) | (REGPC&15) << 16 | (REGTMP&15) << 12;
		}
		break;
	case 95:	/* PLD off(reg) */
		o1 = 0xf5d0f000;
		o1 |= (p->from.reg&15) << 16;
		if(p->from.offset < 0) {
			o1 &= ~(1 << 23);
			o1 |= (-p->from.offset) & 0xfff;
		} else
			o1 |= p->from.offset & 0xfff;
		break;
	case 96:	/* UNDEF */
		// This is supposed to be something that stops execution.
		// It's not supposed to be reached, ever, but if it is, we'd
		// like to be able to tell how we got there.  Assemble as
		// 0xf7fabcfd which is guaranteed to raise undefined instruction
		// exception.
		o1 = 0xf7fabcfd;
		break;
	case 97:	/* CLZ Rm, Rd */
 		o1 = oprrr(ctxt, p->as, p->scond);
 		o1 |= (p->to.reg&15) << 12;
 		o1 |= (p->from.reg&15) << 0;
		break;
	case 98:	/* MULW{T,B} Rs, Rm, Rd */
		o1 = oprrr(ctxt, p->as, p->scond);
		o1 |= (p->to.reg&15) << 16;
		o1 |= (p->from.reg&15) << 8;
		o1 |= (p->reg&15) << 0;
		break;
	case 99:	/* MULAW{T,B} Rs, Rm, Rn, Rd */
		o1 = oprrr(ctxt, p->as, p->scond);
		o1 |= (p->to.reg&15) << 12;
		o1 |= (p->from.reg&15) << 8;
		o1 |= (p->reg&15) << 0;
		o1 |= (p->to.offset&15) << 16;
		break;
	case 100:
		// DATABUNDLE: BKPT $0x5be0, signify the start of NaCl data bundle;
		// DATABUNDLEEND: zero width alignment marker
		if(p->as == ADATABUNDLE)
			o1 = 0xe125be70;
		break;
	}
	
	out[0] = o1;
	out[1] = o2;
	out[2] = o3;
	out[3] = o4;
	out[4] = o5;
	out[5] = o6;
	return;
}

static uint32
mov(Link *ctxt, Prog *p)
{
	uint32 o1;
	int rt, r;

	aclass(ctxt, &p->from);
	o1 = oprrr(ctxt, p->as, p->scond);
	o1 |= p->from.offset;
	rt = p->to.reg;
	if(p->to.type == TYPE_NONE)
		rt = 0;
	r = p->reg;
	if(p->as == AMOVW || p->as == AMVN)
		r = 0;
	else if(r == 0)
		r = rt;
	o1 |= ((r&15)<<16) | ((rt&15)<<12);
	return o1;
}

static uint32
oprrr(Link *ctxt, int a, int sc)
{
	uint32 o;

	o = ((sc & C_SCOND) ^ C_SCOND_XOR) << 28;
	if(sc & C_SBIT)
		o |= 1 << 20;
	if(sc & (C_PBIT|C_WBIT))
		ctxt->diag(".nil/.W on dp instruction");
	switch(a) {
	case AMULU:
	case AMUL:	return o | (0x0<<21) | (0x9<<4);
	case AMULA:	return o | (0x1<<21) | (0x9<<4);
	case AMULLU:	return o | (0x4<<21) | (0x9<<4);
	case AMULL:	return o | (0x6<<21) | (0x9<<4);
	case AMULALU:	return o | (0x5<<21) | (0x9<<4);
	case AMULAL:	return o | (0x7<<21) | (0x9<<4);
	case AAND:	return o | (0x0<<21);
	case AEOR:	return o | (0x1<<21);
	case ASUB:	return o | (0x2<<21);
	case ARSB:	return o | (0x3<<21);
	case AADD:	return o | (0x4<<21);
	case AADC:	return o | (0x5<<21);
	case ASBC:	return o | (0x6<<21);
	case ARSC:	return o | (0x7<<21);
	case ATST:	return o | (0x8<<21) | (1<<20);
	case ATEQ:	return o | (0x9<<21) | (1<<20);
	case ACMP:	return o | (0xa<<21) | (1<<20);
	case ACMN:	return o | (0xb<<21) | (1<<20);
	case AORR:	return o | (0xc<<21);
	case AMOVB:
	case AMOVH:
	case AMOVW:	return o | (0xd<<21);
	case ABIC:	return o | (0xe<<21);
	case AMVN:	return o | (0xf<<21);
	case ASLL:	return o | (0xd<<21) | (0<<5);
	case ASRL:	return o | (0xd<<21) | (1<<5);
	case ASRA:	return o | (0xd<<21) | (2<<5);
	case ASWI:	return o | (0xf<<24);

	case AADDD:	return o | (0xe<<24) | (0x3<<20) | (0xb<<8) | (0<<4);
	case AADDF:	return o | (0xe<<24) | (0x3<<20) | (0xa<<8) | (0<<4);
	case ASUBD:	return o | (0xe<<24) | (0x3<<20) | (0xb<<8) | (4<<4);
	case ASUBF:	return o | (0xe<<24) | (0x3<<20) | (0xa<<8) | (4<<4);
	case AMULD:	return o | (0xe<<24) | (0x2<<20) | (0xb<<8) | (0<<4);
	case AMULF:	return o | (0xe<<24) | (0x2<<20) | (0xa<<8) | (0<<4);
	case ADIVD:	return o | (0xe<<24) | (0x8<<20) | (0xb<<8) | (0<<4);
	case ADIVF:	return o | (0xe<<24) | (0x8<<20) | (0xa<<8) | (0<<4);
	case ASQRTD:	return o | (0xe<<24) | (0xb<<20) | (1<<16) | (0xb<<8) | (0xc<<4);
	case ASQRTF:	return o | (0xe<<24) | (0xb<<20) | (1<<16) | (0xa<<8) | (0xc<<4);
	case AABSD:	return o | (0xe<<24) | (0xb<<20) | (0<<16) | (0xb<<8) | (0xc<<4);
	case AABSF:	return o | (0xe<<24) | (0xb<<20) | (0<<16) | (0xa<<8) | (0xc<<4);
	case ACMPD:	return o | (0xe<<24) | (0xb<<20) | (4<<16) | (0xb<<8) | (0xc<<4);
	case ACMPF:	return o | (0xe<<24) | (0xb<<20) | (4<<16) | (0xa<<8) | (0xc<<4);

	case AMOVF:	return o | (0xe<<24) | (0xb<<20) | (0<<16) | (0xa<<8) | (4<<4);
	case AMOVD:	return o | (0xe<<24) | (0xb<<20) | (0<<16) | (0xb<<8) | (4<<4);

	case AMOVDF:	return o | (0xe<<24) | (0xb<<20) | (7<<16) | (0xa<<8) | (0xc<<4) |
			(1<<8);	// dtof
	case AMOVFD:	return o | (0xe<<24) | (0xb<<20) | (7<<16) | (0xa<<8) | (0xc<<4) |
			(0<<8);	// dtof

	case AMOVWF:
			if((sc & C_UBIT) == 0)
				o |= 1<<7;	/* signed */
			return o | (0xe<<24) | (0xb<<20) | (8<<16) | (0xa<<8) | (4<<4) |
				(0<<18) | (0<<8);	// toint, double
	case AMOVWD:
			if((sc & C_UBIT) == 0)
				o |= 1<<7;	/* signed */
			return o | (0xe<<24) | (0xb<<20) | (8<<16) | (0xa<<8) | (4<<4) |
				(0<<18) | (1<<8);	// toint, double

	case AMOVFW:
			if((sc & C_UBIT) == 0)
				o |= 1<<16;	/* signed */
			return o | (0xe<<24) | (0xb<<20) | (8<<16) | (0xa<<8) | (4<<4) |
				(1<<18) | (0<<8) | (1<<7);	// toint, double, trunc
	case AMOVDW:
			if((sc & C_UBIT) == 0)
				o |= 1<<16;	/* signed */
			return o | (0xe<<24) | (0xb<<20) | (8<<16) | (0xa<<8) | (4<<4) |
				(1<<18) | (1<<8) | (1<<7);	// toint, double, trunc

	case AMOVWF+ALAST:	// copy WtoF
		return o | (0xe<<24) | (0x0<<20) | (0xb<<8) | (1<<4);
	case AMOVFW+ALAST:	// copy FtoW
		return o | (0xe<<24) | (0x1<<20) | (0xb<<8) | (1<<4);
	case ACMP+ALAST:	// cmp imm
		return o | (0x3<<24) | (0x5<<20);

	case ACLZ:
		// CLZ doesn't support .nil
		return (o & (0xf<<28)) | (0x16f<<16) | (0xf1<<4);

	case AMULWT:
		return (o & (0xf<<28)) | (0x12 << 20) | (0xe<<4);
	case AMULWB:
		return (o & (0xf<<28)) | (0x12 << 20) | (0xa<<4);
	case AMULAWT:
		return (o & (0xf<<28)) | (0x12 << 20) | (0xc<<4);
	case AMULAWB:
		return (o & (0xf<<28)) | (0x12 << 20) | (0x8<<4);

	case ABL: // BLX REG
		return (o & (0xf<<28)) | (0x12fff3 << 4);
	}
	ctxt->diag("bad rrr %d", a);
	prasm(ctxt->curp);
	return 0;
}

static uint32
opbra(Link *ctxt, int a, int sc)
{

	if(sc & (C_SBIT|C_PBIT|C_WBIT))
		ctxt->diag(".nil/.nil/.W on bra instruction");
	sc &= C_SCOND;
	sc ^= C_SCOND_XOR;
	if(a == ABL || a == ADUFFZERO || a == ADUFFCOPY)
		return (sc<<28)|(0x5<<25)|(0x1<<24);
	if(sc != 0xe)
		ctxt->diag(".COND on bcond instruction");
	switch(a) {
	case ABEQ:	return (0x0<<28)|(0x5<<25);
	case ABNE:	return (0x1<<28)|(0x5<<25);
	case ABCS:	return (0x2<<28)|(0x5<<25);
	case ABHS:	return (0x2<<28)|(0x5<<25);
	case ABCC:	return (0x3<<28)|(0x5<<25);
	case ABLO:	return (0x3<<28)|(0x5<<25);
	case ABMI:	return (0x4<<28)|(0x5<<25);
	case ABPL:	return (0x5<<28)|(0x5<<25);
	case ABVS:	return (0x6<<28)|(0x5<<25);
	case ABVC:	return (0x7<<28)|(0x5<<25);
	case ABHI:	return (0x8<<28)|(0x5<<25);
	case ABLS:	return (0x9<<28)|(0x5<<25);
	case ABGE:	return (0xa<<28)|(0x5<<25);
	case ABLT:	return (0xb<<28)|(0x5<<25);
	case ABGT:	return (0xc<<28)|(0x5<<25);
	case ABLE:	return (0xd<<28)|(0x5<<25);
	case AB:	return (0xe<<28)|(0x5<<25);
	}
	ctxt->diag("bad bra %A", a);
	prasm(ctxt->curp);
	return 0;
}

static uint32
olr(Link *ctxt, int32 v, int b, int r, int sc)
{
	uint32 o;

	if(sc & C_SBIT)
		ctxt->diag(".nil on LDR/STR instruction");
	o = ((sc & C_SCOND) ^ C_SCOND_XOR) << 28;
	if(!(sc & C_PBIT))
		o |= 1 << 24;
	if(!(sc & C_UBIT))
		o |= 1 << 23;
	if(sc & C_WBIT)
		o |= 1 << 21;
	o |= (1<<26) | (1<<20);
	if(v < 0) {
		if(sc & C_UBIT)
			ctxt->diag(".U on neg offset");
		v = -v;
		o ^= 1 << 23;
	}
	if(v >= (1<<12) || v < 0)
		ctxt->diag("literal span too large: %d (R%d)\n%P", v, b, ctxt->printp);
	o |= v;
	o |= (b&15) << 16;
	o |= (r&15) << 12;
	return o;
}

static uint32
olhr(Link *ctxt, int32 v, int b, int r, int sc)
{
	uint32 o;

	if(sc & C_SBIT)
		ctxt->diag(".nil on LDRH/STRH instruction");
	o = ((sc & C_SCOND) ^ C_SCOND_XOR) << 28;
	if(!(sc & C_PBIT))
		o |= 1 << 24;
	if(sc & C_WBIT)
		o |= 1 << 21;
	o |= (1<<23) | (1<<20)|(0xb<<4);
	if(v < 0) {
		v = -v;
		o ^= 1 << 23;
	}
	if(v >= (1<<8) || v < 0)
		ctxt->diag("literal span too large: %d (R%d)\n%P", v, b, ctxt->printp);
	o |= (v&0xf)|((v>>4)<<8)|(1<<22);
	o |= (b&15) << 16;
	o |= (r&15) << 12;
	return o;
}

static uint32
osr(Link *ctxt, int a, int r, int32 v, int b, int sc)
{
	uint32 o;

	o = olr(ctxt, v, b, r, sc) ^ (1<<20);
	if(a != AMOVW)
		o |= 1<<22;
	return o;
}

static uint32
oshr(Link *ctxt, int r, int32 v, int b, int sc)
{
	uint32 o;

	o = olhr(ctxt, v, b, r, sc) ^ (1<<20);
	return o;
}


static uint32
osrr(Link *ctxt, int r, int i, int b, int sc)
{

	return olr(ctxt, i, b, r, sc) ^ ((1<<25) | (1<<20));
}

static uint32
oshrr(Link *ctxt, int r, int i, int b, int sc)
{
	return olhr(ctxt, i, b, r, sc) ^ ((1<<22) | (1<<20));
}

static uint32
olrr(Link *ctxt, int i, int b, int r, int sc)
{

	return olr(ctxt, i, b, r, sc) ^ (1<<25);
}

static uint32
olhrr(Link *ctxt, int i, int b, int r, int sc)
{
	return olhr(ctxt, i, b, r, sc) ^ (1<<22);
}

static uint32
ofsr(Link *ctxt, int a, int r, int32 v, int b, int sc, Prog *p)
{
	uint32 o;

	if(sc & C_SBIT)
		ctxt->diag(".nil on FLDR/FSTR instruction");
	o = ((sc & C_SCOND) ^ C_SCOND_XOR) << 28;
	if(!(sc & C_PBIT))
		o |= 1 << 24;
	if(sc & C_WBIT)
		o |= 1 << 21;
	o |= (6<<25) | (1<<24) | (1<<23) | (10<<8);
	if(v < 0) {
		v = -v;
		o ^= 1 << 23;
	}
	if(v & 3)
		ctxt->diag("odd offset for floating point op: %d\n%P", v, p);
	else
	if(v >= (1<<10) || v < 0)
		ctxt->diag("literal span too large: %d\n%P", v, p);
	o |= (v>>2) & 0xFF;
	o |= (b&15) << 16;
	o |= (r&15) << 12;

	switch(a) {
	default:
		ctxt->diag("bad fst %A", a);
	case AMOVD:
		o |= 1 << 8;
	case AMOVF:
		break;
	}
	return o;
}

static uint32
omvl(Link *ctxt, Prog *p, Addr *a, int dr)
{
	int32 v;
	uint32 o1;
	if(!p->pcond) {
		aclass(ctxt, a);
		v = immrot(~ctxt->instoffset);
		if(v == 0) {
			ctxt->diag("missing literal");
			prasm(p);
			return 0;
		}
		o1 = oprrr(ctxt, AMVN, p->scond&C_SCOND);
		o1 |= v;
		o1 |= (dr&15) << 12;
	} else {
		v = p->pcond->pc - p->pc - 8;
		o1 = olr(ctxt, v, REGPC, dr, p->scond&C_SCOND);
	}
	return o1;
}

int
chipzero5(Link *ctxt, float64 e)
{
	// We use GOARM=7 to gate the use of VFPv3 vmov (imm) instructions.
	if(ctxt->goarm < 7 || e != 0)
		return -1;
	return 0;
}

int
chipfloat5(Link *ctxt, float64 e)
{
	int n;
	ulong h1;
	uint32 l, h;
	uint64 ei;

	// We use GOARM=7 to gate the use of VFPv3 vmov (imm) instructions.
	if(ctxt->goarm < 7)
		goto no;

	memmove(&ei, &e, 8);
	l = (uint32)ei;
	h = (uint32)(ei>>32);

	if(l != 0 || (h&0xffff) != 0)
		goto no;
	h1 = h & 0x7fc00000;
	if(h1 != 0x40000000 && h1 != 0x3fc00000)
		goto no;
	n = 0;

	// sign bit (a)
	if(h & 0x80000000)
		n |= 1<<7;

	// exp sign bit (b)
	if(h1 == 0x3fc00000)
		n |= 1<<6;

	// rest of exp and mantissa (cd-efgh)
	n |= (h >> 16) & 0x3f;

//print("match %.8lux %.8lux %d\n", l, h, n);
	return n;

no:
	return -1;
}
