// Inferno utils/5l/thumb.c
// http://code.google.com/p/inferno-os/source/browse/utils/5l/thumb.c
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

#include "l.h"

static long thumboprr(int);
static long thumboprrr(int, int);
static long thumbopirr(int , int);
static long thumbopri(int);
static long thumbophh(int);
static long thumbopbra(int);
static long thumbopmv(int, int);
static void lowreg(Prog *, int);
static void mult(Prog *, int, int);
static void numr(Prog *, int, int, int);
static void regis(Prog *, int, int, int);
static void dis(int, int);

// build a constant using neg, add and shift - only worth it if < 6 bytes */
static int
immbuildcon(int c, Prog *p)
{
	int n = 0;

	USED(p);
	if(c >= 0 && c <= 255)
		return 0;			// mv
	if(c >= -255 && c < 0)	// mv, neg
		return 1;
	if(c >= 256 && c <= 510)	// mv, add
		return 1;
	if(c < 0)
		return 0;
	while(!(c & 1)){
		n++;
		c >>= 1;
	}
	if(c >= 0 && c <= 255)	// mv, lsl
		return 1;
	return 0;
}

// positive 5 bit offset from register - O(R)
// positive 8 bit offset from register - mov O, R then [R, R]
// otherwise O goes in literal pool - mov O1(PC), R then [R, R]
static int
immoreg(int off, Prog *p)
{
	int v = 1;
	int as = p->as;

	if(off < 0)
		return C_GOREG;
	if(as == AMOVW)
		v = 4;
	else if(as == AMOVH || as == AMOVHU)
		v = 2;
	else if(as == AMOVB || as == AMOVBU)
		v = 1;
	else
		diag("bad op in immoreg");
	if(off/v <= 31)
		return C_SOREG;
	if(off <= 255)
		return C_LOREG;
	return C_GOREG;
}

// positive 8 bit - mov O, R then 0(R)
// otherwise O goes in literal pool - mov O1(PC), R then 0(R)
static int
immacon(int off, Prog *p, int t1, int t2)
{
	USED(p);
	if(off < 0)
		return t2;
	if(off <= 255)
		return t1;
	return t2;
}

// unsigned 8 bit in words
static int
immauto(int off, Prog *p)
{
	if(p->as != AMOVW)
		diag("bad op in immauto");
	mult(p, off, 4);
	if(off >= 0 && off <= 1020)
		return C_SAUTO;
	return C_LAUTO;
}

static int
immsmall(int off, Prog *p, int t1, int t2, int t3)
{
	USED(p);
	if(off >= 0 && off <= 7)
		return t1;
	if(off >= 0 && off <= 255)
		return t2;
	return t3;
}

static int
immcon(int off, Prog *p)
{
	int as = p->as;

	if(as == ASLL || as == ASRL || as == ASRA)
		return C_SCON;
	if(p->to.type == D_REG && p->to.reg == REGSP){
		if(as == AADD || as == ASUB){
			if(off >= 0 && off <= 508)
				return C_SCON;
			if(as == ASUB){
				p->as = AADD;
				p->from.offset = -p->from.offset;
			}
			return C_LCON;
		}
		diag("unknown type in immcon");
	}
	if(as == AADD || as == ASUB){
		if(p->reg != NREG)
			return immsmall(off, p, C_SCON, C_LCON, C_GCON);
		return immacon(off, p, C_SCON, C_LCON);
	}
	if(as == AMOVW && p->from.type == D_CONST && p->to.type == D_REG && immbuildcon(off, p))
		return C_BCON;
	if(as == ACMP && p->from.type == D_CONST && immbuildcon(off, p))
		return C_BCON;
	if(as == ACMP || as == AMOVW)
		return immacon(off, p, C_SCON, C_LCON);
	return C_LCON;
}

int
thumbaclass(Adr *a, Prog *p)
{
	Sym *s;
	int t;

	switch(a->type) {
	case D_NONE:
		return C_NONE;
	case D_REG:
		if(a->reg == REGSP)
			return C_SP;
		if(a->reg == REGPC)
			return C_PC;
		if(a->reg >= 8)
			return C_HREG;
		return C_REG;
	case D_SHIFT:
		diag("D_SHIFT in thumbaclass");
		return C_SHIFT;
	case D_FREG:
		diag("D_FREG in thumbaclass");
		return C_FREG;
	case D_FPCR:
		diag("D_FPCR in thumbaclass");
		return C_FCR;
	case D_OREG:
		switch(a->name) {
		case D_EXTERN:
		case D_STATIC:
			if(a->sym == 0 || a->sym->name == 0) {
				print("null sym external\n");
				print("%D\n", a);
				return C_GOK;
			}
			t = a->sym->type;
			if(t == 0 || t == SXREF) {
				diag("undefined external: %s in %s\n",
					a->sym->name, TNAME);
				a->sym->type = SDATA;
			}
			instoffset = a->sym->value + a->offset + INITDAT;
			return C_LEXT;	/* INITDAT unknown at this stage */
			// return immacon(instoffset, p, C_SEXT, C_LEXT);
		case D_AUTO:
			instoffset = autosize + a->offset;
			return immauto(instoffset, p);
		case D_PARAM:
			instoffset = autosize + a->offset + 4L;
// print("D_PARAM %s %d+%d+%d = %d\n", a->sym != S ? a->sym->name : "noname", autosize, a->offset, 4, autosize+a->offset+4);
			return immauto(instoffset, p);
		case D_NONE:
			instoffset = a->offset;
			if(a->reg == REGSP)
				return immauto(instoffset, p);
			else
				return immoreg(instoffset, p);
		}
		return C_GOK;
	case D_PSR:
		diag("D_PSR in thumbaclass");
		return C_PSR;
	case D_OCONST:
		switch(a->name) {
		case D_EXTERN:
		case D_STATIC:
			s = a->sym;
			t = s->type;
			if(t == 0 || t == SXREF) {
				diag("undefined external: %s in %s\n",
					s->name, TNAME);
				s->type = SDATA;
			}
			instoffset = s->value + a->offset + INITDAT;
			if(s->type == STEXT || s->type == SLEAF){
				instoffset = s->value + a->offset;
#ifdef CALLEEBX
				instoffset += fnpinc(s);
#else
				if(s->thumb)
					instoffset++;	// T bit
#endif
				return C_LCON;
			}
			return C_LCON;	/* INITDAT unknown at this stage */
			// return immcon(instoffset, p);
		}
		return C_GOK;
	case D_FCONST:
		diag("D_FCONST in thumaclass");
		return C_FCON;
	case D_CONST:
		switch(a->name) {
		case D_NONE:
			instoffset = a->offset;
			if(a->reg != NREG)
				goto aconsize;
			return immcon(instoffset, p);
		case D_EXTERN:
		case D_STATIC:
			s = a->sym;
			if(s == S)
				break;
			t = s->type;
			switch(t) {
			case 0:
			case SXREF:
				diag("undefined external: %s in %s\n",
					s->name, TNAME);
				s->type = SDATA;
				break;
			case SCONST:
			case STEXT:
			case SLEAF:
				instoffset = s->value + a->offset;
#ifdef CALLEEBX
				instoffset += fnpinc(s);
#else
				if(s->thumb)
					instoffset++;	// T bit
#endif
				return C_LCON;
			}
			instoffset = s->value + a->offset + INITDAT;
			return C_LCON;	/* INITDAT unknown at this stage */
			// return immcon(instoffset, p);
		case D_AUTO:
			instoffset = autosize + a->offset;
			goto aconsize;
		case D_PARAM:
			instoffset = autosize + a->offset + 4L;
		aconsize:
			if(p->from.reg == REGSP || p->from.reg == NREG)
				return instoffset >= 0 && instoffset < 1024 ? C_SACON : C_GACON;
			else if(p->from.reg == p->to.reg)
				return immacon(instoffset, p, C_SACON, C_GACON);
			return immsmall(instoffset, p, C_SACON, C_LACON, C_GACON);
		}
		return C_GOK;
	case D_BRANCH: {
		int v, va;

		p->align = 0;
		v = -4;
		va = 0;
		if(p->cond != P){
			v = (p->cond->pc - p->pc) - 4;
			va = p->cond->pc;
		}
		instoffset = v;
		if(p->as == AB){
			if(v >= -2048 && v <= 2046)
				return C_SBRA;
			p->align = 4;
			instoffset = va;
			return C_LBRA;
		}
		if(p->as == ABL){
#ifdef CALLEEBX
			int e;

			if((e = fninc(p->to.sym))) {
				v += e;
				va += e;
				instoffset += e;
			}		
#endif
			if(v >= -4194304 && v <= 4194302)
				return C_SBRA;
			p->align = 2;
			instoffset = va;
			return C_LBRA;
		}
		if(p->as == ABX){
			v = va;
			if(v >= 0 && v <= 255)
				return C_SBRA;
			p->align = 2;
			instoffset = va;
			return C_LBRA;
		}
		if(v >= -256 && v <= 254)
			return C_SBRA;
		if(v >= -(2048-2) && v <= (2046+2))
			return C_LBRA;
		p->align = 2;
		instoffset = va;
		return C_GBRA;
	}
	}
	return C_GOK;
}

// as a1 a2 a3 type size param lit vers
Optab thumboptab[] =
{
	{ ATEXT,		C_LEXT,		C_NONE,		C_LCON,		0,	0,	0 },
	{ ATEXT,		C_LEXT,		C_REG,		C_LCON,		0,	0,	0 },
	{ AMVN,		C_REG,		C_NONE,		C_REG,		1,	2,	0 },
	{ ASRL,		C_REG,		C_NONE,		C_REG,		1,	2,	0 },
	{ ACMP,		C_REG,		C_REG,		C_NONE,		1,	2,	0 },
	{ ACMN,		C_REG,		C_REG,		C_NONE,		1,	2,	0 },
	{ AADD,		C_REG,		C_REG,		C_REG,		2,	2,	0 },
	{ AADD,		C_REG,		C_NONE,		C_REG,		2,	2,	0 },
	{ AADD,		C_SCON,		C_REG,		C_REG,		3,	2,	0 },
	{ AADD,		C_LCON,		C_REG,		C_REG,		49,	4,	0 },
	{ AADD,		C_GCON,		C_REG,		C_REG,		36,	4,	0,	LFROM },
	// { AADD,		C_LCON,		C_NONE,		C_REG,		3,	2,	0,	LFROM },
	{ ASRL,		C_SCON,		C_REG,		C_REG,		4,	2,	0 },
	{ ASRL,		C_SCON,		C_NONE,		C_REG,		4,	2,	0 },
	{ AADD,		C_SCON,		C_NONE,		C_REG,		5,	2,	0 },
	{ AADD,		C_LCON,		C_NONE,		C_REG,		37,	4,	0,	LFROM },
	{ ACMP,		C_SCON,		C_REG,		C_NONE,		5,	2,	0 },
	{ ACMP,		C_BCON,		C_REG,		C_NONE,		48,	6,	0 },
	{ ACMP,		C_LCON,		C_REG,		C_NONE,		39,	4,	0,	LFROM },
	{ AMOVW,		C_SCON,		C_NONE,		C_REG,		5,	2,	0 },
	{ AMOVW,		C_BCON,		C_NONE,		C_REG,		47,	4,	0 },
	{ AMOVW,		C_LCON,		C_NONE,		C_REG,		38,	2,	0,	LFROM },
	// { AADD,		C_LCON,		C_PC,		C_REG,		6,	2,	0,	LFROM },
	// { AADD,		C_LCON,		C_SP,		C_REG,		6,	2,	0,	LFROM },
	{ AADD,		C_SCON,		C_NONE,		C_SP,		7,	2,	0 },
	{ AADD,		C_LCON,		C_NONE,		C_SP,		40,	4,	0,	LFROM },
	{ AADD,		C_REG,		C_NONE,		C_HREG,		8,	2,	0 },
	{ AADD,		C_HREG,		C_NONE,		C_REG,		8,	2,	0 },
	{ AADD,		C_HREG,		C_NONE,		C_HREG,		8,	2,	0 },
	{ AMOVW,		C_REG,		C_NONE,		C_HREG,		8,	2,	0 },
	{ AMOVW,		C_HREG,		C_NONE,		C_REG,		8,	2,	0 },
	{ AMOVW,		C_HREG,		C_NONE,		C_HREG,		8,	2,	0 },
	{ ACMP,		C_REG,		C_HREG,		C_NONE,		8,	2,	0 },
	{ ACMP,		C_HREG,		C_REG,		C_NONE,		8,	2,	0 },
	{ ACMP,		C_HREG,		C_HREG,		C_NONE,		8,	2,	0 },
	{ AB,			C_NONE,		C_NONE,		C_SBRA,		9,	2,	0,	LPOOL },
	{ ABEQ,		C_NONE,		C_NONE,		C_SBRA,		10,	2,	0 },
	{ ABL,		C_NONE,		C_NONE,		C_SBRA,		11,	4,	0 },
	{ ABX,		C_NONE,		C_NONE,		C_SBRA,		12,	10,	0 },
	{ AB,			C_NONE,		C_NONE,		C_LBRA,		41,	8,	0,	LPOOL },
	{ ABEQ,		C_NONE,		C_NONE,		C_LBRA,		46,	4,	0 },
	{ ABL,		C_NONE,		C_NONE,		C_LBRA,		43,	14,	0 },
	{ ABX,		C_NONE,		C_NONE,		C_LBRA,		44,	14,	0 },
	{ ABEQ,		C_NONE,		C_NONE,		C_GBRA,		42,  10, 	0 },
	// { AB,		C_NONE,		C_NONE,		C_SOREG,		13,	0,	0 },
	// { ABL,		C_NONE,		C_NONE,		C_SOREG,		14,	0,	0 },
	{ ABL,		C_NONE,		C_NONE,		C_REG,		51,	4,	0 },
	{ ABX,		C_NONE,		C_NONE,		C_REG,		15,	8,	0 },
	{ ABX,		C_NONE,		C_NONE,		C_HREG,		15,	8,	0 },
	{ ABXRET,		C_NONE,		C_NONE,		C_REG,		45,	2,	0 },
	{ ABXRET,		C_NONE,		C_NONE,		C_HREG,		45,	2,	0 },
	{ ASWI,		C_NONE,		C_NONE,		C_LCON,		16,	2,	0 },
	{ AWORD,		C_NONE,		C_NONE,		C_LCON,		17,	4,	0 },
	{ AWORD,		C_NONE,		C_NONE,		C_GCON,		17,	4,	0 },
	{ AWORD,		C_NONE,		C_NONE,		C_LEXT,		17,	4, 	0 },
	{ ADWORD,	C_LCON,		C_NONE,		C_LCON,		50,	8,	0 },
	{ AMOVW,		C_SAUTO,		C_NONE,		C_REG,		18,	2,	REGSP },
	{ AMOVW,		C_LAUTO,		C_NONE,		C_REG,		33,	6,	0,	LFROM  },
	// { AMOVW,		C_OFFPC,		C_NONE,		C_REG,		18,	2,	REGPC,	LFROM  },
	{ AMOVW,		C_SEXT,		C_NONE,		C_REG,		30,	4,	0 },
	{ AMOVW,		C_SOREG,		C_NONE,		C_REG,		19,	2,	0 },
	{ AMOVHU,	C_SEXT,		C_NONE,		C_REG,		30,	4,	0 },
	{ AMOVHU,	C_SOREG,		C_NONE,		C_REG,		19,	2,	0 },
	{ AMOVBU,	C_SEXT,		C_NONE,		C_REG,		30,	4,	0 },
	{ AMOVBU,	C_SOREG,		C_NONE,		C_REG,		19,	2,	0 },
	{ AMOVW,		C_REG,		C_NONE,		C_SAUTO,		20,	2,	0 },
	{ AMOVW,		C_REG,		C_NONE,		C_LAUTO,		34,	6,	0,	LTO },
	{ AMOVW,		C_REG,		C_NONE,		C_SEXT,		31,	4,	0 },
	{ AMOVW,		C_REG,		C_NONE,		C_SOREG,		21,	2,	0 },
	{ AMOVH,		C_REG,		C_NONE,		C_SEXT,		31,	4,	0 },
	{ AMOVH,		C_REG,		C_NONE,		C_SOREG,		21,	2,	0 },
	{ AMOVB,		C_REG,		C_NONE,		C_SEXT,		31,	4,	0 },
	{ AMOVB,		C_REG,		C_NONE,		C_SOREG,		21,	2,	0 },
	{ AMOVHU,	C_REG,		C_NONE,		C_SEXT,		31,	4,	0 },
	{ AMOVHU,	C_REG,		C_NONE,		C_SOREG,		21,	2,	0 },
	{ AMOVBU,	C_REG,		C_NONE,		C_SEXT,		31,	4,	0 },
	{ AMOVBU,	C_REG,		C_NONE,		C_SOREG,		21,	2,	0 },
	{ AMOVW,		C_REG,		C_NONE,		C_REG,		22,	2,	0 },
	{ AMOVB,		C_REG,		C_NONE,		C_REG,		23,	4,	0 },
	{ AMOVH,		C_REG,		C_NONE,		C_REG,		23,	4,	0 },
	{ AMOVBU,	C_REG,		C_NONE,		C_REG,		23,	4,	0 },
	{ AMOVHU,	C_REG,		C_NONE,		C_REG,		23,	4,	0 },
	{ AMOVH,		C_SEXT,		C_NONE,		C_REG,		32,	6,	0 },
	{ AMOVH,		C_SOREG,		C_NONE,		C_REG,		24,	4,	0 },
	{ AMOVB,		C_SEXT,		C_NONE,		C_REG,		32,	6,	0 },
	{ AMOVB,		C_SOREG,		C_NONE,		C_REG,		24,	4,	0 },
	{ AMOVW,		C_SACON,	C_NONE,		C_REG,		25,	2,	0 },
	{ AMOVW,		C_LACON,	C_NONE,		C_REG,		35,	4,	0 },
	{ AMOVW,		C_GACON,	C_NONE,		C_REG,		35,	4,	0,	LFROM },
	{ AMOVM,		C_LCON,		C_NONE,		C_REG,		26,	2,	0 },
	{ AMOVM,		C_REG,		C_NONE,		C_LCON,		27,	2,	0 },
	{ AMOVW,		C_LOREG,		C_NONE,		C_REG,		28,	4,	0 },
	{ AMOVH,		C_LOREG,		C_NONE,		C_REG,		28,	4,	0 },
	{ AMOVB,		C_LOREG,		C_NONE,		C_REG,		28,	4,	0 },
	{ AMOVHU,	C_LOREG,		C_NONE,		C_REG,		28,	4,	0 },
	{ AMOVBU,	C_LOREG,		C_NONE,		C_REG,		28,	4,	0 },
	{ AMOVW,		C_REG,		C_NONE,		C_LOREG,		29,	4,	0 },
	{ AMOVH,		C_REG,		C_NONE,		C_LOREG,		29,	4,	0 },
	{ AMOVB,		C_REG,		C_NONE,		C_LOREG,		29,	4,	0 },
	{ AMOVHU,	C_REG,		C_NONE,		C_LOREG,		29,	4,	0 },
	{ AMOVBU,	C_REG,		C_NONE,		C_LOREG,		29,	4,	0 },
	{ AMOVW,		C_GOREG,		C_NONE,		C_REG,		28,	4,	0,	LFROM },
	{ AMOVH,		C_GOREG,		C_NONE,		C_REG,		28,	4,	0,	LFROM },
	{ AMOVB,		C_GOREG,		C_NONE,		C_REG,		28,	4,	0,	LFROM },
	{ AMOVHU,	C_GOREG,		C_NONE,		C_REG,		28,	4,	0,	LFROM },
	{ AMOVBU,	C_GOREG,		C_NONE,		C_REG,		28,	4,	0,	LFROM },
	{ AMOVW,		C_REG,		C_NONE,		C_GOREG,		29,	4,	0,	LTO },
	{ AMOVH,		C_REG,		C_NONE,		C_GOREG,		29,	4,	0,	LTO },
	{ AMOVB,		C_REG,		C_NONE,		C_GOREG,		29,	4,	0,	LTO },
	{ AMOVHU,	C_REG,		C_NONE,		C_GOREG,		29,	4,	0,	LTO },
	{ AMOVBU,	C_REG,		C_NONE,		C_GOREG,		29,	4,	0,	LTO },
	{ AMOVW,		C_LEXT,		C_NONE,		C_REG,		30,	4,	0,	LFROM },
	{ AMOVH,		C_LEXT,		C_NONE,		C_REG,		32,	6,	0,	LFROM },
	{ AMOVB,		C_LEXT,		C_NONE,		C_REG,		32,	6,	0,	LFROM },
	{ AMOVHU,	C_LEXT,		C_NONE,		C_REG,		30,	4,	0,	LFROM },
	{ AMOVBU,	C_LEXT,		C_NONE,		C_REG,		30,	4,	0,	LFROM },
	{ AMOVW,		C_REG,		C_NONE,		C_LEXT,		31,	4,	0,	LTO },
	{ AMOVH,		C_REG,		C_NONE,		C_LEXT,		31,	4,	0,	LTO },
	{ AMOVB,		C_REG,		C_NONE,		C_LEXT,		31,	4,	0,	LTO },
	{ AMOVHU,	C_REG,		C_NONE,		C_LEXT,		31,	4,	0,	LTO },
	{ AMOVBU,	C_REG,		C_NONE,		C_LEXT,		31,	4,	0,	LTO },

	{ AXXX,		C_NONE,		C_NONE,		C_NONE,		0,	2,	0 },
};

#define OPCNTSZ	52
int opcount[OPCNTSZ];

// is this too pessimistic ?
int
brextra(Prog *p)
{
	int c;

	// +2 is for padding
	if(p->as == ATEXT)
		return 0-0+2;
	if(!isbranch(p))
		diag("bad op in brextra()");
	c = thumbaclass(&p->to, p);
	switch(p->as){
		case AB:
			if(c != C_SBRA)
				return 0;
			return 8-2+2;
		case ABL:
			if(c != C_SBRA)
				return 0;
			return 14-4+2;
		case ABX:
			if(c == C_REG || c == C_HREG)
				return 0;
#ifdef CALLEEBX
			diag("ABX $I in brextra");
#endif
			if(c != C_SBRA)
				return 0;
			return 14-10+2;
		default:
			if(c == C_GBRA)
				return 0;
			if(c == C_LBRA)
				return 10-4+2;
			return 10-2+2;
	}
	return 0;
}

#define high(r)	((r)>=8)

static long
mv(Prog *p, int r, int off)
{
	int v, o;
	if(p != nil && p->cond != nil){	// in literal pool
		v = p->cond->pc - p->pc - 4;
		if(p->cond->pc & 3)
			diag("mv: bad literal pool alignment");
		if(v & 3)
			v += 2;	// ensure M(4) offset
		mult(p, v, 4);
		off = v/4;
		numr(p, off, 0, 255);
		o = 0x9<<11;
	}
	else{
		numr(p, off, 0, 255);
		o = 0x4<<11;
	}
	o |= (r<<8) | off;
	return o;
}

static void
mvcon(Prog *p, int r, int c, long *o1, long *o2)
{
	int op = 0, n = 0;

	if(c >= 0 && c <= 255)
		diag("bad c in mvcon");
	if(c >= -255 && c < 0)	// mv, neg
		c = -c;
	else if(c >= 256 && c <= 510){	// mv, add
		n = rand()%(511-c) + (c-255);
		c -= n;
		// n = c-255;
		// c = 255;
		op = AADD;
	}
	else{
		if(c < 0)
			diag("-ve in mvcon");
		while(!(c & 1)){
			n++;
			c >>= 1;
		}
		if(c >= 0 && c <= 255)	// mv, lsl
			op = ASLL;
		else
			diag("bad shift in mvcon");
	}
	*o1 = mv(p, r, c);
	switch(op){
		case 0:
			*o2 = (1<<14) | (9<<6) | (r<<3) | r;
			break;
		case AADD:
			*o2 = (6<<11) | (r<<8) | n;
			break;
		case ASLL:
			*o2 = (n<<6) | (r<<3) | r;
			break;
	}
}

static long
mvlh(int rs, int rd)
{
	int o = 0x46<<8;

	if(high(rs)){
		rs -= 8;
		o |= 1<<6;
	}
	if(high(rd)){
		rd -= 8;
		o |= 1<<7;
	}
	o |= (rs<<3) | rd;
	return o;
}

void
thumbbuildop()
{
	int i, n, r;
	Optab *optab = thumboptab;
	Oprang *oprange = thumboprange;

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
		case AMVN:
			oprange[AADC] = oprange[r];
			oprange[ASBC] = oprange[r];
			oprange[AMUL] = oprange[r];
			oprange[AAND] = oprange[r];
			oprange[AEOR] = oprange[r];
			oprange[AORR] = oprange[r];
			oprange[ABIC] = oprange[r];
			oprange[AMULU] = oprange[r];
			break;
		case ACMN:
			oprange[ATST] = oprange[r];
			break;
		case ASRL:
			oprange[ASRA] = oprange[r];
			oprange[ASLL] = oprange[r];
			break;
		case AADD:
			oprange[ASUB] = oprange[r];
			break;
		}
	}
}

void
thumbasmout(Prog *p, Optab *o)
{
	long o1, o2, o3, o4, o5, o6, o7, v;
	int r, rf, rt;

	rf = p->from.reg;
	rt = p->to.reg;
	r = p->reg;
	o1 = o2 = o3 = o4 = o5 = o6 = o7 = 0;
if(debug['P']) print("%ulx: %P	type %d %d\n", (ulong)(p->pc), p, o->type, p->align);
	opcount[o->type] += o->size;
	switch(o->type) {
	default:
		diag("unknown asm %d", o->type);
		prasm(p);
		break;
	case 0:		/* pseudo ops */
if(debug['G']) print("%ulx: %s: thumb\n", (ulong)(p->pc), p->from.sym->name);
		break;
	case 1:		/* op R, -, R or op R, R, - */
		o1 = thumboprr(p->as);
		if(rt == NREG)
			rt = r;
		lowreg(p, rf);
		lowreg(p, rt);
		o1 |= (0x10<<10) | (rf<<3) | rt;
		break;
	case 2:		/* add/sub R, R, R or add/sub R, -, R */
		o1 = p->as == AADD ? 0x0<<9 : 0x1<<9;
		if(r == NREG)
			r = rt;
		lowreg(p, rf);
		lowreg(p, r);
		lowreg(p, rt);
		o1 |= (0x6<<10) | (rf<<6) | (r<<3) | rt;
		break;
	case 3:		/* add/sub $I, R, R or add/sub $I, -, R */
		thumbaclass(&p->from, p);
		o1 = p->as == AADD ? 0x0<<9 : 0x1<<9;
		if(r == NREG)
			r = rt;
		numr(p, instoffset, 0, 7);
		lowreg(p, r);
		lowreg(p, rt);
		o1 |= (0x7<<10) | (instoffset<<6) | (r<<3) | rt;
		break;
	case 4:		/* shift $I, R, R or shift $I, -, R */
		thumbaclass(&p->from, p);
		if(instoffset < 0)
			diag("negative shift in thumbasmout");
		instoffset %= 32;
		o1 = thumbopri(p->as);
		if(r == NREG)
			r = rt;
		numr(p, instoffset, 0, 31);
		lowreg(p, r);
		lowreg(p, rt);
		o1 |= (0x0<<13) | (instoffset<<6) | (r<<3) | rt;
		break;
	case 5:		/* add/sub/mov $I, -, R or cmp $I, R, - */
		thumbaclass(&p->from, p);
		o1 = thumbopri(p->as);	
		if(rt == NREG)
			rt = r;
		numr(p, instoffset, 0, 255);
		lowreg(p, rt);
		o1 |= (0x1<<13) | (rt<<8) | instoffset;
		break;
	case 6:		/* add $I, PC/SP, R */
		if(p->as == ASUB)
			diag("subtract in add $I, PC/SP, R");
		thumbaclass(&p->from, p);
		o1 = r == REGSP ? 0x1<<11 : 0x0<<11;
		numr(p, instoffset, 0, 255);
		regis(p, r, REGSP, REGPC);
		lowreg(p, rt);
		o1 |= (0xa<<12) | (rt<<8) | instoffset;
		break;
	case 7:		/* add, sub $I, SP */
		thumbaclass(&p->from, p);
		o1 = p->as == AADD ? 0x0<<7 : 0x1<<7;
		numr(p, instoffset, 0, 508);
		mult(p, instoffset, 4);
		regis(p, rt, REGSP, REGSP);
		o1 |= (0xb0<<8) | (instoffset>>2);
		break;
	case 8:		/* add/mov/cmp R, R where at least 1 reg is high */
		o1 = 0;
		if(rt == NREG)
			rt = r;
		if(high(rf)){
			o1 |= 1<<6;
			rf -= 8;
		}
		if(high(rt)){
			o1 |= 2<<6;
			rt -= 8;
		}
		if(o1 == 0)
			diag("no high register(%P)", p);
		o1 |= thumbophh(p->as);
		o1 |= (0x11<<10) | (rf<<3) | rt;
		break;
	case 9:		/* B	$I */
		thumbaclass(&p->to, p);
		numr(p, instoffset, -2048, 2046);
		o1 = (0x1c<<11) | ((instoffset>>1)&0x7ff);
		break;
	case 10:		/* Bcc $I */
		thumbaclass(&p->to, p);
		numr(p, instoffset, -256, 254);
		o1 = thumbopbra(p->as);
		o1 |= (0xd<<12) | ((instoffset>>1)&0xff);
		break;
	case 11:		/* BL $I */
		thumbaclass(&p->to, p);
		numr(p, instoffset, -4194304, 4194302);
		o1 = (0x1e<<11) | ((instoffset>>12)&0x7ff);
		o2 = (0x1f<<11) | ((instoffset>>1)&0x7ff);
		break;
	case 12:		/* BX $I */
#ifdef CALLEEBX
		diag("BX $I case");
#endif
		thumbaclass(&p->to, p);
		if(p->to.sym->thumb)
			instoffset  |= 1;	// T bit
		o1 = mvlh(REGPC, REGTMPT);
		o2 = (0x6<<11) | (REGTMPT<<8) | 7;	// add 7, RTMP	(T bit + PC offset)
		o3 = mvlh(REGTMPT, REGLINK);
		o4 = mv(nil, REGTMPT, instoffset);
		o5 = (0x11c<<6) | (REGTMPT<<3);
		// o1 = mv(nil, REGTMPT, v);
		// o2 = (0x11b<<6) | (REGPC<<3) | REGLINK;
		// o3 = (0x11c<<6) | (REGTMPT<<3);
		break;
	case 13:		/* B O(R)  */
		diag("B O(R)");
		break;
	case 14:		/* BL O(R) */
		diag("BL O(R)");
		break;
	case 15:		/* BX R */
		o1 = mvlh(REGPC, REGTMPT);
		o2 = (0x6<<11) | (REGTMPT<<8) | 5;	// add 5, RTMP (T bit + PC offset)
		o3 = mvlh(REGTMPT, REGLINK);
		o4 = 0;
		if(high(rt)){
			rt -= 8;
			o4 |= 1<<6;
		}
		o4 |= (0x8e<<7) | (rt<<3);
		// o1 = (0x11c<<6) | (rt<<3);
		break;
	case 16:		/* SWI $I */
		thumbaclass(&p->to, p);
		numr(p, instoffset, 0, 255);
		o1 = (0xdf<<8) | instoffset;
		break;
	case 17:		/* AWORD */
		thumbaclass(&p->to, p);
		o1 = instoffset&0xffff;
		o2 = (instoffset>>16)&0xffff;
		break;
	case 18:		/* AMOVW O(SP), R and AMOVW O(PC), R */
		thumbaclass(&p->from, p);
		rf = o->param;
		o1 = rf == REGSP ? 0x13<<11 : 0x9<<11;
		regis(p, rf, REGSP, REGPC);
		lowreg(p, rt);
		mult(p, instoffset, 4);
		numr(p, instoffset/4, 0, 255);
		o1 |= (rt<<8) | (instoffset/4);
		break;
	case 19:		/* AMOVW... O(R), R */
		thumbaclass(&p->from, p);
		o1 = thumbopmv(p->as, 1);
		v = 4;
		if(p->as == AMOVHU)
			v = 2;
		else if(p->as == AMOVBU)
			v = 1;
		mult(p, instoffset, v);
		lowreg(p, rf);
		lowreg(p, rt);
		numr(p, instoffset/v, 0, 31);
		o1 |= ((instoffset/v)<<6) | (rf<<3) | rt;
		break;
	case 20:		/* AMOVW R, O(SP) */
		thumbaclass(&p->to, p);
		o1 = 0x12<<11;
		if(rt != NREG) regis(p, rt, REGSP, REGSP);
		lowreg(p, rf);
		mult(p, instoffset, 4);
		numr(p, instoffset/4, 0, 255);
		o1 |= (rf<<8) | (instoffset/4);
		break;
	case 21:		/* AMOVW... R, O(R) */
		thumbaclass(&p->to, p);
		o1 = thumbopmv(p->as, 0);
		v = 4;
		if(p->as == AMOVHU || p->as == AMOVH)
			v = 2;
		else if(p->as == AMOVBU || p->as == AMOVB)
			v = 1;
		lowreg(p, rf);
		lowreg(p, rt);
		mult(p, instoffset, v);
		numr(p, instoffset/v, 0, 31);
		o1 |= ((instoffset/v)<<6) | (rt<<3) | rf;
		break;
	case 22:		/* AMOVW R, R -> ASLL $0, R, R */
		o1 = thumbopri(ASLL);
		lowreg(p, rf);
		lowreg(p, rt);
		o1 |= (0x0<<13) | (rf<<3) | rt;
		break;
	case 23:		/* AMOVB/AMOVH/AMOVBU/AMOVHU R, R */
		o1 = thumbopri(ASLL);
		o2 = p->as == AMOVB || p->as == AMOVH ? thumbopri(ASRA) : thumbopri(ASRL);
		v = p->as == AMOVB || p->as == AMOVBU ? 24 : 16;
		lowreg(p, rf);
		lowreg(p, rt);
		o1 |= (0x0<<13) | (v<<6) | (rf<<3) | rt;
		o2 |= (0x0<<13) | (v<<6) | (rt<<3) | rt;
		break;
	case 24:	/* AMOVH/AMOVB O(R), R -> AMOVH/AMOVB [R, R], R */
		thumbaclass(&p->from, p);
		lowreg(p, rf);
		lowreg(p, rt);
		if(rf == rt)
			r = REGTMPT;
		else
			r = rt;
		if(p->as == AMOVB)
			numr(p, instoffset, 0, 31);
		else{
			mult(p, instoffset, 2);
			numr(p, instoffset, 0, 62);
		}
		o1 = mv(p, r, instoffset);
		o2 = p->as == AMOVH ? 0x2f<<9 : 0x2b<<9;
		o2 |= (r<<6) | (rf<<3) | rt;
		break;
	case 25:	/* MOVW $sacon, R */
		thumbaclass(&p->from, p);
// print("25: %d %d %d %d\n", instoffset, rf, r, rt);
		if(rf == NREG)
			rf = REGSP;
		lowreg(p, rt);
		if(rf == REGSP){
			mult(p, instoffset, 4);
			numr(p, instoffset>>2, 0, 255);
			o1 = (0x15<<11) | (rt<<8) | (instoffset>>2);	// add $O, SP, R
		}
		else if(rf == rt){
			numr(p, instoffset, 0, 255);
			o1 = (0x6<<11) | (rt<<8) | instoffset;		// add $O, R
		}
		else{
			lowreg(p, rf);
			numr(p, instoffset, 0, 7);
			o1 = (0xe<<9) | (instoffset<<6) | (rf<<3) | rt;	// add $O, Rs, Rd
		}
		break;
	case 26:	/* AMOVM $c, oreg -> stmia */
		lowreg(p, rt);
		numr(p, p->from.offset, -256, 255);
		o1 = (0x18<<11) | (rt<<8) | (p->from.offset&0xff);
		break;
	case 27:	/* AMOVM oreg, $c ->ldmia */
		lowreg(p, rf);
		numr(p, p->to.offset, -256, 256);
		o1 = (0x19<<11) | (rf<<8) | (p->to.offset&0xff);
		break;
	case 28:	/* AMOV* O(R), R -> AMOV* [R, R], R 	(offset large)	*/
		thumbaclass(&p->from, p);
		lowreg(p, rf);
		lowreg(p, rt);
		if(rf == rt)
			r = REGTMPT;
		else
			r = rt;
		o1 = mv(p, r, instoffset);
		o2 = thumboprrr(p->as, 1);
		o2 |= (r<<6) | (rf<<3) | rt;
		break;
	case 29:	/* AMOV* R, O(R) -> AMOV* R, [R, R]	(offset large)	*/
		thumbaclass(&p->to, p);
		lowreg(p, rf);
		lowreg(p, rt);
		if(rt == REGTMPT){	// used as tmp reg
			if(instoffset >= 0 && instoffset <= 255){
				o1 = (1<<13) | (2<<11) | (rt<<8) | instoffset;	// add $O, R7
				o2 = thumbopirr(p->as, 0);
				o2 |= (0<<6) | (rt<<3) | rf;					// mov* R, 0(R)
			}
			else
				diag("big offset - case 29");
		}
		else{
			o1 = mv(p, REGTMPT, instoffset);
			o2 = thumboprrr(p->as, 0);
			o2 |= (REGTMPT<<6) | (rt<<3) | rf;
		}
		break;
	case 30:		/* AMOVW... *addr, R */
		thumbaclass(&p->from, p);
		o1 = mv(p, rt, instoffset);		// MOV addr, rtmp
		o2 = thumbopmv(p->as, 1);
		lowreg(p, rt);
		o2 |= (rt<<3) | rt;			// MOV* 0(rtmp), R
		break;
	case 31:		/* AMOVW... R, *addr */
		thumbaclass(&p->to, p);
		o1 = mv(p, REGTMPT, instoffset);
		o2 = thumbopmv(p->as, 0);
		lowreg(p, rf);
		o2 |= (REGTMPT<<3) | rf;
		break;
	case 32:	/* AMOVH/AMOVB *addr, R -> AMOVH/AMOVB [R, R], R */
		thumbaclass(&p->from, p);
		o1 = mv(p, rt, instoffset);
		lowreg(p, rt);
		o2 = mv(nil, REGTMPT, 0);
		o3 = p->as == AMOVH ? 0x2f<<9 : 0x2b<<9;
		o3 |= (REGTMPT<<6) | (rt<<3) | rt;
		break;
	case 33:	/* AMOVW O(SP), R	(O large) */
		thumbaclass(&p->from, p);
		lowreg(p, rt);
		o1 = mv(p, rt, instoffset);
		o2 = (0x111<<6) | (REGSP-8)<<3 | rt;	// add SP, rt
		o3 = thumbopmv(p->as, 1);
		o3 |= (rt<<3) | rt;
		break;
	case 34:	/* AMOVW R, O(SP)	(O large) */
		thumbaclass(&p->to, p);
		lowreg(p, rf);
		o1 = mv(p, REGTMPT, instoffset);
		o2 = (0x111<<6) | (REGSP-8)<<3 | REGTMPT;	// add SP, REGTMP
		o3 = thumbopmv(p->as, 0);
		o3 |= (REGTMPT<<3) | rf;
		break;
	case 35:	/* AMOVW $lacon, R */
		thumbaclass(&p->from, p);
		lowreg(p, rt);
		if(rf == NREG)
			rf = REGSP;
		if(rf == rt)
			rf = r = REGTMPT;
		else
			r = rt;
// print("35: io=%d rf=%d rt=%d\n", instoffset, rf, rt);
		o1 = mv(p, r, instoffset);		// mov O, Rd
		if(high(rf))
			o2 = (0x44<<8) | (0x1<<6) | ((rf-8)<<3) | rt;	// add Rs, Rd
		else
			o2 = (0x6<<10) | (rf<<6) | (rt<<3) | rt;		// add Rs, Rd
		break;
	case 36:	/* AADD/ASUB $i, r, r when $i too big */
		thumbaclass(&p->from, p);
		lowreg(p, r);
		lowreg(p, rt);
		o1 = mv(p, REGTMPT, instoffset);
		o2 = p->as == AADD ? 0xc<<9 : 0xd<<9;
		o2 |= (REGTMPT<<6) | (r<<3) | rt;
		break;
	case 37:	/* AADD/ASUB $i, r when $i too big */
		thumbaclass(&p->from, p);
		lowreg(p, rt);
		o1 = mv(p, REGTMPT, instoffset);
		o2 = p->as == AADD ? 0xc<<9 : 0xd<<9;
		o2 |= (REGTMPT<<6) | (rt<<3) | rt;
		break;
	case 38:	/* AMOVW $i, r when $i too big */
		thumbaclass(&p->from, p);
		lowreg(p, rt);
		o1 = mv(p, rt, instoffset);
		break;
	case 39:	/* ACMP $i, r when $i too big */
		thumbaclass(&p->from, p);
		lowreg(p, r);
		o1 = mv(p, REGTMPT, instoffset);
		o2 = (0x10a<<6) | (REGTMPT<<3) | r;
		break;
	case 40:		/* add, sub $I, SP when $I large*/
		thumbaclass(&p->from, p);
		if(p->as == ASUB)
			instoffset = -instoffset;
		o1 = mv(p, REGTMPT, instoffset);
		o2 = (0x112<<6) | (REGTMPT<<3) | (REGSP-8);
		regis(p, rt, REGSP, REGSP);
		break;
	case	41:		/* BL LBRA */
		thumbaclass(&p->to, p);
		o1 = (0x9<<11) | (REGTMPT<<8);	// mov 0(pc), r7
		o2 = mvlh(REGTMPT, REGPC);		// mov r7, pc
		o3 = instoffset&0xffff;			// $lab
		o4 = (instoffset>>16)&0xffff;
		break;
	case 42:		/* Bcc GBRA */
		thumbaclass(&p->to, p);
		o1 = (0xd<<12) | thumbopbra(relinv(p->as)) | (6>>1);		// bccnot 
		// ab lbra
		o2 = (0x9<<11) | (REGTMPT<<8);	// mov 0(pc), r7
		o3 = mvlh(REGTMPT, REGPC);		// mov r7, pc
		o4 = instoffset&0xffff;			// $lab
		o5 = (instoffset>>16)&0xffff;
		break;
	case 43:		/* BL LBRA */
		thumbaclass(&p->to, p);
		o1 = mvlh(REGPC, REGTMPT);						// mov pc, r7
		o2 = (0x6<<11) | (REGTMPT<<8) | 10;				// add 10, r7
		o3 = mvlh(REGTMPT, REGLINK);					// mov r7, lr
		o4 = (0x9<<11) | (REGTMPT<<8);					// mov o(pc), r7
		o5 = mvlh(REGTMPT, REGPC);						// mov r7, pc
		o6 = instoffset&0xffff;							// $lab
		o7 = (instoffset>>16)&0xffff;
		break;
	case 44:		/* BX LBRA */
#ifdef CALLEEBX
		diag("BX LBRA case");
#endif
		thumbaclass(&p->to, p);
		if(p->to.sym->thumb)
			instoffset  |= 1;	// T bit
		o1 = mvlh(REGPC, REGTMPT);						// mov pc, r7
		o2 = (0x6<<11) | (REGTMPT<<8) | 11;				// add 11, r7
		o3 = mvlh(REGTMPT, REGLINK);					// mov r7, lr
		o4 = (0x9<<11) | (REGTMPT<<8);					// mov o(pc), r7
		o5 = (0x11c<<6) | (REGTMPT<<3);					// bx r7
		o6 = instoffset&0xffff;							// $lab
		o7 = (instoffset>>16)&0xffff;
		break;
	case 45:	/* BX R when returning from fn */
		o1 = 0;
		if(high(rt)){
			rt -= 8;
			o1 |= 1<<6;
		}
		o1 |= (0x8e<<7) | (rt<<3);
		break;
	case 46:		/* Bcc LBRA */
		thumbaclass(&p->to, p);
		o1 = (0xd<<12) | thumbopbra(relinv(p->as)) | (0>>1);		// bccnot 
		// ab lbra
		instoffset -= 2;
		numr(p, instoffset, -2048, 2046);
		o2 = (0x1c<<11) | ((instoffset>>1)&0x7ff);
		break;
	case 47:	/* mov $i, R where $i can be built */
		thumbaclass(&p->from, p);
		mvcon(p, rt, instoffset, &o1, &o2);
		break;
	case 48: /* ACMP $i, r when $i built up */
		thumbaclass(&p->from, p);
		lowreg(p, r);
		mvcon(p, REGTMPT, instoffset, &o1, &o2);
		o3 = (0x10a<<6) | (REGTMPT<<3) | r;
		break;
	case 49:	/* AADD $i, r, r when $i is between 0 and 255 - could merge with case 36 */
		thumbaclass(&p->from, p);
		lowreg(p, r);
		lowreg(p, rt);
		numr(p, instoffset, 0, 255);
		o1 = mv(p, REGTMPT, instoffset);
		o2 = p->as == AADD ? 0xc<<9 : 0xd<<9;
		o2 |= (REGTMPT<<6) | (r<<3) | rt;
		break;
	case 50:		/* ADWORD */
		thumbaclass(&p->from, p);
		o1 = instoffset&0xffff;
		o2 = (instoffset>>16)&0xffff;
		thumbaclass(&p->to, p);
		o3 = instoffset&0xffff;
		o4 = (instoffset>>16)&0xffff;
		break;
	case 51:	/* BL r */
		o1 = mvlh(REGPC, REGLINK);	// mov pc, lr
		o2 = mvlh(rt, REGPC);		// mov r, pc
		break;
	}

	v = p->pc;
	switch(o->size) {
	default:
		if(debug['a'])
			Bprint(&bso, " %.8lux:\t\t%P\n", v, p);
		break;
	case 2:
		if(debug['a'])
			Bprint(&bso, " %.8lux: %.8lux\t%P\n", v, o1, p);
		hputl(o1);
		break;
	case 4:
		if(debug['a'])
			Bprint(&bso, " %.8lux: %.8lux %.8lux\t%P\n", v, o1, o2, p);
		hputl(o1);
		hputl(o2);
		break;
	case 6:
		if(debug['a'])
			Bprint(&bso, "%.8lux: %.8lux %.8lux %.8lux\t%P\n", v, o1, o2, o3, p);
		hputl(o1);
		hputl(o2);
		hputl(o3);
		break;
	case 8:
		if(debug['a'])
			Bprint(&bso, "%.8lux: %.8lux %.8lux %.8lux %.8lux\t%P\n", v, o1, o2, o3, o4, p);
		hputl(o1);
		hputl(o2);
		hputl(o3);
		hputl(o4);
		break;
	case 10:
		if(debug['a'])
			Bprint(&bso, "%.8lux: %.8lux %.8lux %.8lux %.8lux %.8lux\t%P\n", v, o1, o2, o3, o4, o5, p);
		hputl(o1);
		hputl(o2);
		hputl(o3);
		hputl(o4);
		hputl(o5);
		break;
	case 12:
		if(debug['a'])
			Bprint(&bso, "%.8lux: %.8lux %.8lux %.8lux %.8lux %.8lux %.8lux\t%P\n", v, o1, o2, o3, o4, o5, o6, p);
		hputl(o1);
		hputl(o2);
		hputl(o3);
		hputl(o4);
		hputl(o5);
		hputl(o6);
		break;
	case 14:
		if(debug['a'])
			Bprint(&bso, "%.8lux: %.8lux %.8lux %.8lux %.8lux %.8lux %.8lux %.8lux\t%P\n", v, o1, o2, o3, o4, o5, o6, o7, p);
		hputl(o1);
		hputl(o2);
		hputl(o3);
		hputl(o4);
		hputl(o5);
		hputl(o6);
		hputl(o7);
		break;
	}
	if(debug['G']){
		if(o->type == 17){
			print("%lx:	word %ld\n", p->pc, (o2<<16)+o1);
			return;
		}
		if(o->type == 50){
			print("%lx:	word %ld\n", p->pc, (o2<<16)+o1);
			print("%lx:	word %ld\n", p->pc, (o4<<16)+o3);
			return;
		}
		if(o->size > 0) dis(o1, p->pc);
		if(o->size > 2) dis(o2, p->pc+2);
		if(o->size > 4) dis(o3, p->pc+4);
		if(o->size > 6) dis(o4, p->pc+6);
		if(o->size > 8) dis(o5, p->pc+8);
		if(o->size > 10) dis(o6, p->pc+10);
		if(o->size > 12) dis(o7, p->pc+12);
		// if(o->size > 14) dis(o8, p->pc+14);
	}
}

static long
thumboprr(int a)
{
	switch(a) {
	case AMVN:	return 0xf<<6;
	case ACMP:	return 0xa<<6;
	case ACMN:	return 0xb<<6;
	case ATST:	return 0x8<<6;
	case AADC:	return 0x5<<6;
	case ASBC:	return 0x6<<6;
	case AMUL:
	case AMULU:	return 0xd<<6;
	case AAND:	return 0x0<<6;
	case AEOR:	return 0x1<<6;
	case AORR:	return 0xc<<6;
	case ABIC:	return 0xe<<6;
	case ASRL:	return 0x3<<6;
	case ASRA:	return 0x4<<6;
	case ASLL:	return 0x2<<6;
	}
	diag("bad thumbop oprr %d", a);
	prasm(curp);
	return 0;
}

static long
thumbopirr(int a, int ld)
{
	if(ld)
		diag("load in thumbopirr");
	switch(a){
		case AMOVW:	return 0xc<<11;
		case AMOVH:
		case AMOVHU:	return 0x10<<11;
		case AMOVB:
		case AMOVBU:	return 0xe<<11;
	}
	return 0;
}
	
static long
thumboprrr(int a, int ld)
{
	if(ld){
		switch(a){
		case AMOVW:	return 0x2c<<9;
		case AMOVH:	return 0x2f<<9;
		case AMOVB:	return 0x2b<<9;
		case AMOVHU:	return 0x2d<<9;
		case AMOVBU:	return 0x2e<<9;
		}
	}
	else{
		switch(a){
		case AMOVW:	return 0x28<<9;
		case AMOVHU:
		case AMOVH:	return 0x29<<9;
		case AMOVBU:
		case AMOVB:	return 0x2a<<9;
		}
	}
	diag("bad thumbop oprrr %d", a);
	prasm(curp);
	return 0;
}

static long
thumbopri(int a)
{
	switch(a) {
	case ASRL:	return 0x1<<11;
	case ASRA:	return 0x2<<11;
	case ASLL:	return 0x0<<11;
	case AADD:	return 0x2<<11;
	case ASUB:	return 0x3<<11;
	case AMOVW:	return 0x0<<11;
	case ACMP:	return 0x1<<11;
	}
	diag("bad thumbop opri %d", a);
	prasm(curp);
	return 0;
}

static long
thumbophh(int a)
{
	switch(a) {
	case AADD:	return 0x0<<8;
	case AMOVW:	return 0x2<<8;
	case ACMP:	return 0x1<<8;
	}
	diag("bad thumbop ophh %d", a);
	prasm(curp);
	return 0;
}

static long
thumbopbra(int a)
{
	switch(a) {
	case ABEQ:	return 0x0<<8;
	case ABNE:	return 0x1<<8;
	case ABCS:	return 0x2<<8;
	case ABHS:	return 0x2<<8;
	case ABCC:	return 0x3<<8;
	case ABLO:	return 0x3<<8;
	case ABMI:	return 0x4<<8;
	case ABPL:	return 0x5<<8;
	case ABVS:	return 0x6<<8;
	case ABVC:	return 0x7<<8;
	case ABHI:	return 0x8<<8;
	case ABLS:	return 0x9<<8;
	case ABGE:	return 0xa<<8;
	case ABLT:	return 0xb<<8;
	case ABGT:	return 0xc<<8;
	case ABLE:	return 0xd<<8;
	}
	diag("bad thumbop opbra %d", a);
	prasm(curp);
	return 0;
}

static long
thumbopmv(int a, int ld)
{
	switch(a) {
	case AMOVW: 	return (ld ? 0xd : 0xc)<<11;
	case AMOVH:
	case AMOVHU:	return (ld ? 0x11: 0x10)<<11;
	case AMOVB:
	case AMOVBU:	return (ld ? 0xf : 0xe)<<11;
	}
	diag("bad thumbop opmv %d", a);
	prasm(curp);
	return 0;
}

static void 
lowreg(Prog *p, int r)
{
	if(high(r))
		diag("high reg [%P]", p);
}

static void
mult(Prog *p, int n, int m)
{
	if(m*(n/m) != n)
		diag("%d not M(%d) [%P]", n, m, p);
}

static void 
numr(Prog *p, int n, int min, int max)
{
	if(n < min || n > max)
		diag("%d not in %d-%d [%P]", n, min, max, p);
}

static void 
regis(Prog *p, int r, int r1, int r2)
{
	if(r != r1 && r != r2)
		diag("reg %d not %d or %d [%P]", r, r1, r2, p);
}

void
hputl(int n)
{
	cbp[1] = n>>8;
	cbp[0] = n;
	cbp += 2;
	cbc -= 2;
	if(cbc <= 0)
		cflush();
}

void
thumbcount()
{
	int i, c = 0, t = 0;

	for (i = 0; i < OPCNTSZ; i++)
		t += opcount[i];
	if(t == 0)
		return;
	for (i = 0; i < OPCNTSZ; i++){
		c += opcount[i];
		print("%d:	%d %d %d%%\n", i, opcount[i], c, (opcount[i]*100+t/2)/t);
	}
}
	
char *op1[] = { "lsl", "lsr", "asr" };
char *op2[] = { "add", "sub" };
char *op3[] = { "movw", "cmp", "add", "sub" };
char *op4[] = { "and", "eor", "lsl", "lsr", "asr", "adc", "sbc", "ror",
		        "tst", "neg", "cmp", "cmpn", "or", "mul", "bitc", "movn" };
char *op5[] = { "add", "cmp", "movw", "bx" };
char *op6[] = { "smovw", "smovh", "smovb", "lmovb", "lmovw", "lmovhu", "lmovbu", "lmovh" };
char *op7[] = { "smovw", "lmovw", "smovb", "lmovbu" };
char *op8[] = { "smovh", "lmovhu" };
char *op9[] = { "smovw", "lmovw" };
char *op10[] = { "push", "pop" };
char *op11[] = { "stmia", "ldmia" };

char *cond[] = { "eq", "ne", "hs", "lo", "mi", "pl", "vs", "vc",
			 "hi", "ls", "ge", "lt", "gt", "le", "al", "nv" };

#define B(h, l)		bits(i, h, l)
#define IMM(h, l)	B(h, l)
#define REG(h, l)	reg(B(h, l))
#define LHREG(h, l, lh)	lhreg(B(h, l), B(lh, lh))
#define COND(h, l)	cond[B(h, l)]
#define OP1(h, l)	op1[B(h, l)]
#define OP2(h, l)	op2[B(h, l)]
#define OP3(h, l)	op3[B(h, l)]
#define OP4(h, l)	op4[B(h, l)]
#define OP5(h, l)	op5[B(h, l)]
#define OP6(h, l)	op6[B(h, l)]
#define OP7(h, l)	op7[B(h, l)]
#define OP8(h, l)	op8[B(h, l)]
#define OP9(h, l)	op9[B(h, l)]
#define OP10(h, l)	op10[B(h, l)]
#define OP11(h, l)	op11[B(h, l)]
#define SBZ(h, l)	if(IMM(h, l) != 0) diag("%x: %x bits %d,%d not zero", pc, i, h, l)
#define SNBZ(h, l)	if(IMM(h, l) == 0) diag("%x: %x bits %d,%d zero", pc, i, h, l)
#define SBO(h, l)	if(IMM(h, l) != 1) diag("%x: %x bits %d,%d not one", pc, i, h, l)

static int
bits(int i, int h, int l)
{
	if(h < l)
		diag("h < l in bits");
	return (i&(((1<<(h-l+1))-1)<<l))>>l;
}

static char *
reg(int r)
{
	static char s[4][4];
	static int i = 0;

	if(r < 0 || r > 7)
		diag("register %d out of range", r);
	i++;
	if(i == 4)
		i = 0;
	sprint(s[i], "r%d", r);
	return s[i];
}

static char *regnames[] = { "sp", "lr", "pc" };

static char *
lhreg(int r, int lh)
{
	static char s[4][4];
	static int i = 0;

	if(lh == 0)
		return reg(r);
	if(r < 0 || r > 7)
		diag("high register %d out of range", r);
	i++;
	if(i == 4)
		i = 0;
	if(r >= 5)
		sprint(s[i], "%s", regnames[r-5]);
	else
		sprint(s[i], "r%d", r+8);
	return s[i];
}
	
static void
illegal(int i, int pc)
{
	diag("%x: %x illegal instruction", pc, i);
}

static void
dis(int i, int pc)
{
	static int lasto;
	int o, l;
	char *op;

	print("%x: %x:	", pc, i);
	if(i&0xffff0000)
		illegal(i, pc);
	o = B(15, 13);
	switch(o){
	case 0:
		o = B(12, 11);
		switch(o){
			case 0:
			case 1:
			case 2:
				print("%s	%d, %s, %s\n", OP1(12, 11), IMM(10, 6), REG(5, 3), REG(2, 0));
				return;
			case 3:
				if(B(10, 10) == 0)
					print("%s	%s, %s, %s\n", OP2(9, 9), REG(8, 6), REG(5, 3), REG(2, 0));
				else
					print("%s	%d, %s, %s\n", OP2(9, 9), IMM(8, 6), REG(5, 3), REG(2, 0));
				return;
		}
	case 1:
		print("%s	%d, %s\n", OP3(12, 11), IMM(7, 0), REG(10, 8));
		return;
	case 2:
		o = B(12, 10);
		if(o == 0){
			print("%s	%s, %s\n", OP4(9, 6), REG(5, 3), REG(2, 0));
			return;
		}
		if(o == 1){
			o = B(9, 8);
			if(o == 3){
				SBZ(7, 7);
				SBZ(2, 0);
				print("%s	%s\n", OP5(9, 8), LHREG(5, 3, 6));
				return;
			}
			SNBZ(7, 6);
			print("%s	%s, %s\n", OP5(9, 8), LHREG(5, 3, 6), LHREG(2, 0, 7));
			return;
		}
		if(o == 2 || o == 3){
			print("movw	%d(pc)[%x], %s\n", 4*IMM(7, 0), 4*IMM(7, 0)+pc+4, REG(10, 8));
			return;
		}
		op = OP6(11, 9);
		if(*op == 'l')
			print("%s	[%s, %s], %s\n", op+1, REG(8, 6), REG(5, 3), REG(2, 0));
		else
			print("%s	%s, [%s, %s]\n", op+1, REG(2, 0), REG(8, 6), REG(5, 3));
		return;
	case 3:
		op = OP7(12, 11);
		if(B(12, 11) == 0 || B(12,11) == 1)
			l = 4;
		else
			l = 1;
		if(*op == 'l')
			print("%s	%d(%s), %s\n", op+1, l*IMM(10, 6), REG(5, 3), REG(2, 0));
		else
			print("%s	%s, %d(%s)\n", op+1, REG(2, 0), l*IMM(10, 6), REG(5, 3));
		return;
	case 4:
		if(B(12, 12) == 0){
			op = OP8(11, 11);
			if(*op == 'l')
				print("%s	%d(%s), %s\n", op+1, 2*IMM(10, 6), REG(5, 3), REG(2, 0));
			else
				print("%s	%s, %d(%s)\n", op+1, REG(2, 0), 2*IMM(10, 6), REG(5, 3));
			return;
		}
		op = OP9(11, 11);
		if(*op == 'l')
			print("%s	%d(sp), %s\n", op+1, 4*IMM(7, 0), REG(10, 8));
		else
			print("%s	%s, %d(sp)\n", op+1, REG(10, 8), 4*IMM(7, 0));
		return;
	case 5:
		if(B(12, 12) == 0){
			if(B(11, 11) == 0)
				print("add	%d, pc, %s\n", 4*IMM(7, 0), REG(10, 8));
			else
				print("add	%d, sp, %s\n", 4*IMM(7, 0), REG(10, 8));
			return;
		}
		if(B(11, 8) == 0){
			print("%s	%d, sp\n", OP2(7, 7), 4*IMM(6, 0));
			return;
		}
		SBO(10, 10);
		SBZ(9, 9);
		if(B(8, 8) == 0)
			print("%s	sp, %d\n", OP10(11, 11), IMM(7, 0));
		else
			print("%s	sp, %d|15\n", OP10(11, 11), IMM(7, 0));
		return;
	case 6:
		if(B(12, 12) == 0){
			print("%s	%s, %d\n", OP11(11, 11), REG(10, 8), IMM(7, 0));
			return;
		}
		if(B(11, 8) == 0xf){
			print("swi	%d\n", IMM(7, 0));
			return;
		}
		o = IMM(7, 0);
		if(o&0x80)
			o |= 0xffffff00;
		o = pc+4+(o<<1);
		print("b%s	%x\n", COND(11, 8), o);
		return;
	case 7:
		o = B(12, 11);
		switch(o){
			case 0:
				o = IMM(10, 0);
				if(o&0x400)
					o |= 0xfffff800;
				o = pc+4+(o<<1);
				print("b	%x\n", o);
				return;
			case 1:
				illegal(i, pc);
				return;
			case 2:
				lasto = IMM(10, 0);
				print("bl\n");
				return;
			case 3:
				if(lasto&0x400)
					lasto |= 0xfffff800;
				o = IMM(10, 0);
				o = (pc-2)+4+(o<<1)+(lasto<<12);
				print("bl %x\n", o);
				return;
		}
	}
}
