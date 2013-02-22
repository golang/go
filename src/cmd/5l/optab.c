// Inferno utils/5l/optab.c
// http://code.google.com/p/inferno-os/source/browse/utils/5l/optab.c
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

#include	"l.h"

Optab	optab[] =
{
	/* struct Optab:
	  OPCODE,	from, prog->reg, to,		 type,size,param,flag */
	{ ATEXT,	C_ADDR,	C_NONE,	C_LCON, 	 0, 0, 0 },
	{ ATEXT,	C_ADDR,	C_REG,	C_LCON, 	 0, 0, 0 },

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
	{ ABL,		C_NONE,	C_NONE,	C_ROREG,	 7, 8, 0 },
	{ ABL,		C_REG,	C_NONE,	C_ROREG,	 7, 8, 0 },
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

	{ AMOVB,	C_REG,	C_NONE,	C_REG,		14, 8, 0 },
	{ AMOVBU,	C_REG,	C_NONE,	C_REG,		58, 4, 0 },
	{ AMOVH,	C_REG,	C_NONE,	C_REG,		14, 8, 0 },
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

	{ AMOVW,	C_REG,	C_NONE,	C_SHIFT,	61, 4, 0 },
	{ AMOVB,	C_REG,	C_NONE,	C_SHIFT,	61, 4, 0 },
	{ AMOVBU,	C_REG,	C_NONE,	C_SHIFT,	61, 4, 0 },

	{ ACASE,	C_REG,	C_NONE,	C_NONE,		62, 4, 0, LPCREL, 8 },
	{ ABCASE,	C_NONE, C_NONE, C_SBRA,		63, 4, 0 },

	{ AMOVH,	C_REG,	C_NONE, C_HAUTO,	70, 4, REGSP,	0 },
	{ AMOVH,	C_REG,	C_NONE,	C_HOREG,	70, 4, 0,	0 },
	{ AMOVHU,	C_REG,	C_NONE, C_HAUTO,	70, 4, REGSP,	0 },
	{ AMOVHU,	C_REG,	C_NONE,	C_HOREG,	70, 4, 0,	0 },

	{ AMOVB,	C_HAUTO,C_NONE,	C_REG,		71, 4, REGSP,	0 },
	{ AMOVB,	C_HOREG,C_NONE,	C_REG,		71, 4, 0,	0 },
	{ AMOVH,	C_HAUTO,C_NONE, C_REG,		71, 4, REGSP,	0 },
	{ AMOVH,	C_HOREG,C_NONE,	C_REG,		71, 4, 0,	0 },
	{ AMOVHU,	C_HAUTO,C_NONE, C_REG,		71, 4, REGSP,	0 },
	{ AMOVHU,	C_HOREG,C_NONE,	C_REG,		71, 4, 0,	0 },

	{ AMOVH,	C_REG,	C_NONE, C_LAUTO,	72, 8, REGSP,	LTO },
	{ AMOVH,	C_REG,	C_NONE,	C_LOREG,	72, 8, 0,	LTO },
	{ AMOVH,	C_REG,	C_NONE,	C_ADDR,	94, 8, 0,	LTO | LPCREL, 4 },
	{ AMOVHU,	C_REG,	C_NONE, C_LAUTO,	72, 8, REGSP,	LTO },
	{ AMOVHU,	C_REG,	C_NONE,	C_LOREG,	72, 8, 0,	LTO },
	{ AMOVHU,	C_REG,	C_NONE,	C_ADDR,	94, 8, 0,	LTO | LPCREL, 4 },

	{ AMOVB,	C_LAUTO,C_NONE,	C_REG,		73, 8, REGSP,	LFROM },
	{ AMOVB,	C_LOREG,C_NONE,	C_REG,		73, 8, 0,	LFROM },
	{ AMOVB,	C_ADDR,	C_NONE,	C_REG,		93, 8, 0,	LFROM | LPCREL, 4 },
	{ AMOVH,	C_LAUTO,C_NONE, C_REG,		73, 8, REGSP,	LFROM },
	{ AMOVH,	C_LOREG,C_NONE,	C_REG,		73, 8, 0,	LFROM },
	{ AMOVH,	C_ADDR,	C_NONE,	C_REG,		93, 8, 0,	LFROM | LPCREL, 4 },
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

	{ AXXX,		C_NONE,	C_NONE,	C_NONE,		 0, 4, 0 },
};
