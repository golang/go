// Inferno libmach/5db.c
// http://code.google.com/p/inferno-os/source/browse/utils/libmach/5db.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.
//	Power PC support Copyright © 1995-2004 C H Forsyth (forsyth@terzarima.net).
//	Portions Copyright © 1997-1999 Vita Nuova Limited.
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).
//	Revisions Copyright © 2000-2004 Lucent Technologies Inc. and others.
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

#include <u.h>
#include <libc.h>
#include <bio.h>
#include "ureg_arm.h"
#include <mach.h>

static int debug = 0;

#define	BITS(a, b)	((1<<(b+1))-(1<<a))

#define LSR(v, s)	((ulong)(v) >> (s))
#define ASR(v, s)	((long)(v) >> (s))
#define ROR(v, s)	(LSR((v), (s)) | (((v) & ((1 << (s))-1)) << (32 - (s))))



typedef struct	Instr	Instr;
struct	Instr
{
	Map	*map;
	ulong	w;
	ulong	addr;
	uchar	op;			/* super opcode */

	uchar	cond;			/* bits 28-31 */
	uchar	store;			/* bit 20 */

	uchar	rd;			/* bits 12-15 */
	uchar	rn;			/* bits 16-19 */
	uchar	rs;			/* bits 0-11 (shifter operand) */

	long	imm;			/* rotated imm */
	char*	curr;			/* fill point in buffer */
	char*	end;			/* end of buffer */
	char*	err;			/* error message */
};

typedef struct Opcode Opcode;
struct Opcode
{
	char*	o;
	void	(*fmt)(Opcode*, Instr*);
	ulong	(*foll)(Map*, Rgetter, Instr*, ulong);
	char*	a;
};

static	void	format(char*, Instr*, char*);
static	char	FRAMENAME[] = ".frame";

/*
 * Arm-specific debugger interface
 */

extern	char	*armexcep(Map*, Rgetter);
static	int	armfoll(Map*, uvlong, Rgetter, uvlong*);
static	int	arminst(Map*, uvlong, char, char*, int);
static	int	armdas(Map*, uvlong, char*, int);
static	int	arminstlen(Map*, uvlong);

/*
 *	Debugger interface
 */
Machdata armmach =
{
	{0, 0, 0, 0xD},		/* break point */
	4,			/* break point size */

	leswab,			/* short to local byte order */
	leswal,			/* long to local byte order */
	leswav,			/* long to local byte order */
	risctrace,		/* C traceback */
	riscframe,		/* Frame finder */
	armexcep,			/* print exception */
	0,			/* breakpoint fixup */
	0,			/* single precision float printer */
	0,			/* double precision float printer */
	armfoll,		/* following addresses */
	arminst,		/* print instruction */
	armdas,			/* dissembler */
	arminstlen,		/* instruction size */
};

char*
armexcep(Map *map, Rgetter rget)
{
	long c;

	c = (*rget)(map, "TYPE");
	switch (c&0x1f) {
	case 0x11:
		return "Fiq interrupt";
	case 0x12:
		return "Mirq interrupt";
	case 0x13:
		return "SVC/SWI Exception";
	case 0x17:
		return "Prefetch Abort/Data Abort";
	case 0x18:
		return "Data Abort";
	case 0x1b:
		return "Undefined instruction/Breakpoint";
	case 0x1f:
		return "Sys trap";
	default:
		return "Undefined trap";
	}
}

static
char*	cond[16] =
{
	"EQ",	"NE",	"CS",	"CC",
	"MI",	"PL",	"VS",	"VC",
	"HI",	"LS",	"GE",	"LT",
	"GT",	"LE",	0,	"NV"
};

static
char*	shtype[4] =
{
	"<<",	">>",	"->",	"@>"
};

static
char *hb[4] =
{
	"???",	"HU", "B", "H"
};

static
char*	addsub[2] =
{
	"-",	"+",
};

int
armclass(long w)
{
	int op;

	op = (w >> 25) & 0x7;
	switch(op) {
	case 0:	/* data processing r,r,r */
		op = ((w >> 4) & 0xf);
		if(op == 0x9) {
			op = 48+16;		/* mul */
			if(w & (1<<24)) {
				op += 2;
				if(w & (1<<22))
					op++;	/* swap */
				break;
			}
			if(w & (1<<21))
				op++;		/* mla */
			break;
		}
		if ((op & 0x9) == 0x9)		/* ld/st byte/half s/u */
		{
			op = (48+16+4) + ((w >> 22) & 0x1) + ((w >> 19) & 0x2);
			break;
		}
		op = (w >> 21) & 0xf;
		if(w & (1<<4))
			op += 32;
		else
		if(w & (31<<7))
			op += 16;
		break;
	case 1:	/* data processing i,r,r */
		op = (48) + ((w >> 21) & 0xf);
		break;
	case 2:	/* load/store byte/word i(r) */
		op = (48+24) + ((w >> 22) & 0x1) + ((w >> 19) & 0x2);
		break;
	case 3:	/* load/store byte/word (r)(r) */
		op = (48+24+4) + ((w >> 22) & 0x1) + ((w >> 19) & 0x2);
		break;
	case 4:	/* block data transfer (r)(r) */
		op = (48+24+4+4) + ((w >> 20) & 0x1);
		break;
	case 5:	/* branch / branch link */
		op = (48+24+4+4+2) + ((w >> 24) & 0x1);
		break;
	case 7:	/* coprocessor crap */
		op = (48+24+4+4+2+2) + ((w >> 3) & 0x2) + ((w >> 20) & 0x1);
		break;
	default:
		op = (48+24+4+4+2+2+4);
		break;
	}
	return op;
}

static int
decode(Map *map, ulong pc, Instr *i)
{
	uint32 w;

	if(get4(map, pc, &w) < 0) {
		werrstr("can't read instruction: %r");
		return -1;
	}
	i->w = w;
	i->addr = pc;
	i->cond = (w >> 28) & 0xF;
	i->op = armclass(w);
	i->map = map;
	return 1;
}

static void
bprint(Instr *i, char *fmt, ...)
{
	va_list arg;

	va_start(arg, fmt);
	i->curr = vseprint(i->curr, i->end, fmt, arg);
	va_end(arg);
}

static int
plocal(Instr *i)
{
	char *reg;
	Symbol s;
	char *fn;
	int class;
	int offset;

	if(!findsym(i->addr, CTEXT, &s)) {
		if(debug)fprint(2,"fn not found @%lux: %r\n", i->addr);
		return 0;
	}
	fn = s.name;
	if (!findlocal(&s, FRAMENAME, &s)) {
		if(debug)fprint(2,"%s.%s not found @%s: %r\n", fn, FRAMENAME, s.name);
			return 0;
	}
	if(s.value > i->imm) {
		class = CAUTO;
		offset = s.value-i->imm;
		reg = "(SP)";
	} else {
		class = CPARAM;
		offset = i->imm-s.value-4;
		reg = "(FP)";
	}
	if(!getauto(&s, offset, class, &s)) {
		if(debug)fprint(2,"%s %s not found @%ux: %r\n", fn,
			class == CAUTO ? " auto" : "param", offset);
		return 0;
	}
	bprint(i, "%s%c%d%s", s.name, class == CPARAM ? '+' : '-', s.value, reg);
	return 1;
}

/*
 * Print value v as name[+offset]
 */
int
gsymoff(char *buf, int n, long v, int space)
{
	Symbol s;
	int r;
	long delta;

	r = delta = 0;		/* to shut compiler up */
	if (v) {
		r = findsym(v, space, &s);
		if (r)
			delta = v-s.value;
		if (delta < 0)
			delta = -delta;
	}
	if (v == 0 || r == 0 || delta >= 4096)
		return snprint(buf, n, "#%lux", v);
	if (strcmp(s.name, ".string") == 0)
		return snprint(buf, n, "#%lux", v);
	if (!delta)
		return snprint(buf, n, "%s", s.name);
	if (s.type != 't' && s.type != 'T')
		return snprint(buf, n, "%s+%llux", s.name, v-s.value);
	else
		return snprint(buf, n, "#%lux", v);
}

static void
armdps(Opcode *o, Instr *i)
{
	i->store = (i->w >> 20) & 1;
	i->rn = (i->w >> 16) & 0xf;
	i->rd = (i->w >> 12) & 0xf;
	i->rs = (i->w >> 0) & 0xf;
	if(i->rn == 15 && i->rs == 0) {
		if(i->op == 8) {
			format("MOVW", i,"CPSR, R%d");
			return;
		} else
		if(i->op == 10) {
			format("MOVW", i,"SPSR, R%d");
			return;
		}
	} else
	if(i->rn == 9 && i->rd == 15) {
		if(i->op == 9) {
			format("MOVW", i, "R%s, CPSR");
			return;
		} else
		if(i->op == 11) {
			format("MOVW", i, "R%s, SPSR");
			return;
		}
	}
	format(o->o, i, o->a);
}

static void
armdpi(Opcode *o, Instr *i)
{
	ulong v;
	int c;

	v = (i->w >> 0) & 0xff;
	c = (i->w >> 8) & 0xf;
	while(c) {
		v = (v<<30) | (v>>2);
		c--;
	}
	i->imm = v;
	i->store = (i->w >> 20) & 1;
	i->rn = (i->w >> 16) & 0xf;
	i->rd = (i->w >> 12) & 0xf;
	i->rs = i->w&0x0f;

		/* RET is encoded as ADD #0,R14,R15 */
	if((i->w & 0x0fffffff) == 0x028ef000){
		format("RET%C", i, "");
		return;
	}
	if((i->w & 0x0ff0ffff) == 0x0280f000){
		format("B%C", i, "0(R%n)");
		return;
	}
	format(o->o, i, o->a);
}

static void
armsdti(Opcode *o, Instr *i)
{
	ulong v;

	v = i->w & 0xfff;
	if(!(i->w & (1<<23)))
		v = -v;
	i->store = ((i->w >> 23) & 0x2) | ((i->w >>21) & 0x1);
	i->imm = v;
	i->rn = (i->w >> 16) & 0xf;
	i->rd = (i->w >> 12) & 0xf;
		/* RET is encoded as LW.P x,R13,R15 */
	if ((i->w & 0x0ffff000) == 0x049df000)
	{
		format("RET%C%p", i, "%I");
		return;
	}
	format(o->o, i, o->a);
}

/* arm V4 ld/st halfword, signed byte */
static void
armhwby(Opcode *o, Instr *i)
{
	i->store = ((i->w >> 23) & 0x2) | ((i->w >>21) & 0x1);
	i->imm = (i->w & 0xf) | ((i->w >> 8) & 0xf);
	if (!(i->w & (1 << 23)))
		i->imm = - i->imm;
	i->rn = (i->w >> 16) & 0xf;
	i->rd = (i->w >> 12) & 0xf;
	i->rs = (i->w >> 0) & 0xf;
	format(o->o, i, o->a);
}

static void
armsdts(Opcode *o, Instr *i)
{
	i->store = ((i->w >> 23) & 0x2) | ((i->w >>21) & 0x1);
	i->rs = (i->w >> 0) & 0xf;
	i->rn = (i->w >> 16) & 0xf;
	i->rd = (i->w >> 12) & 0xf;
	format(o->o, i, o->a);
}

static void
armbdt(Opcode *o, Instr *i)
{
	i->store = (i->w >> 21) & 0x3;		/* S & W bits */
	i->rn = (i->w >> 16) & 0xf;
	i->imm = i->w & 0xffff;
	if(i->w == 0xe8fd8000)
		format("RFE", i, "");
	else
		format(o->o, i, o->a);
}

/*
static void
armund(Opcode *o, Instr *i)
{
	format(o->o, i, o->a);
}

static void
armcdt(Opcode *o, Instr *i)
{
	format(o->o, i, o->a);
}
*/

static void
armunk(Opcode *o, Instr *i)
{
	format(o->o, i, o->a);
}

static void
armb(Opcode *o, Instr *i)
{
	ulong v;

	v = i->w & 0xffffff;
	if(v & 0x800000)
		v |= ~0xffffff;
	i->imm = (v<<2) + i->addr + 8;
	format(o->o, i, o->a);
}

static void
armco(Opcode *o, Instr *i)		/* coprocessor instructions */
{
	int op, p, cp;

	char buf[1024];

	i->rn = (i->w >> 16) & 0xf;
	i->rd = (i->w >> 12) & 0xf;
	i->rs = i->w&0xf;
	cp = (i->w >> 8) & 0xf;
	p = (i->w >> 5) & 0x7;
	if(i->w&(1<<4)) {
		op = (i->w >> 21) & 0x07;
		snprint(buf, sizeof(buf), "#%x, #%x, R%d, C(%d), C(%d), #%x\n", cp, op, i->rd, i->rn, i->rs, p);
	} else {
		op = (i->w >> 20) & 0x0f;
		snprint(buf, sizeof(buf), "#%x, #%x, C(%d), C(%d), C(%d), #%x\n", cp, op, i->rd, i->rn, i->rs, p);
	}
	format(o->o, i, buf);
}

static int
armcondpass(Map *map, Rgetter rget, uchar cond)
{
	ulong psr;
	uchar n;
	uchar z;
	uchar c;
	uchar v;

	psr = rget(map, "PSR");
	n = (psr >> 31) & 1;
	z = (psr >> 30) & 1;
	c = (psr >> 29) & 1;
	v = (psr >> 28) & 1;

	switch(cond) {
		case 0:		return z;
		case 1:		return !z;
		case 2:		return c;
		case 3:		return !c;
		case 4:		return n;
		case 5:		return !n;
		case 6:		return v;
		case 7:		return !v;
		case 8:		return c && !z;
		case 9:		return !c || z;
		case 10:	return n == v;
		case 11:	return n != v;
		case 12:	return !z && (n == v);
		case 13:	return z && (n != v);
		case 14:	return 1;
		case 15:	return 0;
	}
	return 0;
}

static ulong
armshiftval(Map *map, Rgetter rget, Instr *i)
{
	if(i->w & (1 << 25)) {				/* immediate */
		ulong imm = i->w & BITS(0, 7);
		ulong s = (i->w & BITS(8, 11)) >> 7; /* this contains the *2 */
		return ROR(imm, s);
	} else {
		char buf[8];
		ulong v;
		ulong s = (i->w & BITS(7,11)) >> 7;

		sprint(buf, "R%ld", i->w & 0xf);
		v = rget(map, buf);

		switch((i->w & BITS(4, 6)) >> 4) {
		case 0:					/* LSLIMM */
			return v << s;
		case 1:					/* LSLREG */
			sprint(buf, "R%lud", s >> 1);
			s = rget(map, buf) & 0xFF;
			if(s >= 32) return 0;
			return v << s;
		case 2:					/* LSRIMM */
			return LSR(v, s);
		case 3:					/* LSRREG */
			sprint(buf, "R%ld", s >> 1);
			s = rget(map, buf) & 0xFF;
			if(s >= 32) return 0;
			return LSR(v, s);
		case 4:					/* ASRIMM */
			if(s == 0) {
				if((v & (1U<<31)) == 0)
					return 0;
				return 0xFFFFFFFF;
			}
			return ASR(v, s);
		case 5:					/* ASRREG */
			sprint(buf, "R%ld", s >> 1);
			s = rget(map, buf) & 0xFF;
			if(s >= 32) {
				if((v & (1U<<31)) == 0)
					return 0;
				return 0xFFFFFFFF;
			}
			return ASR(v, s);
		case 6:					/* RORIMM */
			if(s == 0) {
				ulong c = (rget(map, "PSR") >> 29) & 1;

				return (c << 31) | LSR(v, 1);
			}
			return ROR(v, s);
		case 7:					/* RORREG */
			sprint(buf, "R%ld", (s>>1)&0xF);
			s = rget(map, buf);
			if(s == 0 || (s & 0xF) == 0)
				return v;
			return ROR(v, s & 0xF);
		}
	}
	return 0;
}

static int
nbits(ulong v)
{
	int n = 0;
	int i;

	for(i=0; i < 32 ; i++) {
		if(v & 1) ++n;
		v >>= 1;
	}

	return n;
}

static ulong
armmaddr(Map *map, Rgetter rget, Instr *i)
{
	ulong v;
	ulong nb;
	char buf[8];
	ulong rn;

	rn = (i->w >> 16) & 0xf;
	sprint(buf,"R%ld", rn);

	v = rget(map, buf);
	nb = nbits(i->w & ((1 << 15) - 1));

	switch((i->w >> 23) & 3) {
		case 0: return (v - (nb*4)) + 4;
		case 1: return v;
		case 2: return v - (nb*4);
		case 3: return v + 4;
	}
	return 0;
}

static ulong
armaddr(Map *map, Rgetter rget, Instr *i)
{
	char buf[8];
	ulong rn;

	sprint(buf, "R%ld", (i->w >> 16) & 0xf);
	rn = rget(map, buf);

	if((i->w & (1<<24)) == 0) {			/* POSTIDX */
		sprint(buf, "R%ld", rn);
		return rget(map, buf);
	}

	if((i->w & (1<<25)) == 0) {			/* OFFSET */
		sprint(buf, "R%ld", rn);
		if(i->w & (1U<<23))
			return rget(map, buf) + (i->w & BITS(0,11));
		return rget(map, buf) - (i->w & BITS(0,11));
	} else {					/* REGOFF */
		ulong index = 0;
		uchar c;
		uchar rm;

		sprint(buf, "R%ld", i->w & 0xf);
		rm = rget(map, buf);

		switch((i->w & BITS(5,6)) >> 5) {
		case 0: index = rm << ((i->w & BITS(7,11)) >> 7);	break;
		case 1: index = LSR(rm, ((i->w & BITS(7,11)) >> 7));	break;
		case 2: index = ASR(rm, ((i->w & BITS(7,11)) >> 7));	break;
		case 3:
			if((i->w & BITS(7,11)) == 0) {
				c = (rget(map, "PSR") >> 29) & 1;
				index = c << 31 | LSR(rm, 1);
			} else {
				index = ROR(rm, ((i->w & BITS(7,11)) >> 7));
			}
			break;
		}
		if(i->w & (1<<23))
			return rn + index;
		return rn - index;
	}
}

static ulong
armfadd(Map *map, Rgetter rget, Instr *i, ulong pc)
{
	char buf[8];
	int r;

	r = (i->w >> 12) & 0xf;
	if(r != 15 || !armcondpass(map, rget, (i->w >> 28) & 0xf))
		return pc+4;

	r = (i->w >> 16) & 0xf;
	sprint(buf, "R%d", r);

	return rget(map, buf) + armshiftval(map, rget, i);
}

static ulong
armfmovm(Map *map, Rgetter rget, Instr *i, ulong pc)
{
	uint32 v;
	ulong addr;

	v = i->w & 1<<15;
	if(!v || !armcondpass(map, rget, (i->w>>28)&0xf))
		return pc+4;

	addr = armmaddr(map, rget, i) + nbits(i->w & BITS(0,15));
	if(get4(map, addr, &v) < 0) {
		werrstr("can't read addr: %r");
		return -1;
	}
	return v;
}

static ulong
armfbranch(Map *map, Rgetter rget, Instr *i, ulong pc)
{
	if(!armcondpass(map, rget, (i->w >> 28) & 0xf))
		return pc+4;

	return pc + (((signed long)i->w << 8) >> 6) + 8;
}

static ulong
armfmov(Map *map, Rgetter rget, Instr *i, ulong pc)
{
	ulong rd;
	uint32 v;

	rd = (i->w >> 12) & 0xf;
	if(rd != 15 || !armcondpass(map, rget, (i->w>>28)&0xf))
		return pc+4;

	 /* LDR */
	/* BUG: Needs LDH/B, too */
	if(((i->w>>26)&0x3) == 1) {
		if(get4(map, armaddr(map, rget, i), &v) < 0) {
			werrstr("can't read instruction: %r");
			return pc+4;
		}
		return v;
	}

	 /* MOV */
	return armshiftval(map, rget, i);
}

static Opcode opcodes[] =
{
	"AND%C%S",	armdps, 0,	"R%s,R%n,R%d",
	"EOR%C%S",	armdps, 0,	"R%s,R%n,R%d",
	"SUB%C%S",	armdps, 0,	"R%s,R%n,R%d",
	"RSB%C%S",	armdps, 0,	"R%s,R%n,R%d",
	"ADD%C%S",	armdps, armfadd,	"R%s,R%n,R%d",
	"ADC%C%S",	armdps, 0,	"R%s,R%n,R%d",
	"SBC%C%S",	armdps, 0,	"R%s,R%n,R%d",
	"RSC%C%S",	armdps, 0,	"R%s,R%n,R%d",
	"TST%C%S",	armdps, 0,	"R%s,R%n",
	"TEQ%C%S",	armdps, 0,	"R%s,R%n",
	"CMP%C%S",	armdps, 0,	"R%s,R%n",
	"CMN%C%S",	armdps, 0,	"R%s,R%n",
	"ORR%C%S",	armdps, 0,	"R%s,R%n,R%d",
	"MOVW%C%S",	armdps, armfmov,	"R%s,R%d",
	"BIC%C%S",	armdps, 0,	"R%s,R%n,R%d",
	"MVN%C%S",	armdps, 0,	"R%s,R%d",

/* 16 */
	"AND%C%S",	armdps, 0,	"(R%s%h%m),R%n,R%d",
	"EOR%C%S",	armdps, 0,	"(R%s%h%m),R%n,R%d",
	"SUB%C%S",	armdps, 0,	"(R%s%h%m),R%n,R%d",
	"RSB%C%S",	armdps, 0,	"(R%s%h%m),R%n,R%d",
	"ADD%C%S",	armdps, armfadd,	"(R%s%h%m),R%n,R%d",
	"ADC%C%S",	armdps, 0,	"(R%s%h%m),R%n,R%d",
	"SBC%C%S",	armdps, 0,	"(R%s%h%m),R%n,R%d",
	"RSC%C%S",	armdps, 0,	"(R%s%h%m),R%n,R%d",
	"TST%C%S",	armdps, 0,	"(R%s%h%m),R%n",
	"TEQ%C%S",	armdps, 0,	"(R%s%h%m),R%n",
	"CMP%C%S",	armdps, 0,	"(R%s%h%m),R%n",
	"CMN%C%S",	armdps, 0,	"(R%s%h%m),R%n",
	"ORR%C%S",	armdps, 0,	"(R%s%h%m),R%n,R%d",
	"MOVW%C%S",	armdps, armfmov,	"(R%s%h%m),R%d",
	"BIC%C%S",	armdps, 0,	"(R%s%h%m),R%n,R%d",
	"MVN%C%S",	armdps, 0,	"(R%s%h%m),R%d",

/* 32 */
	"AND%C%S",	armdps, 0,	"(R%s%hR%M),R%n,R%d",
	"EOR%C%S",	armdps, 0,	"(R%s%hR%M),R%n,R%d",
	"SUB%C%S",	armdps, 0,	"(R%s%hR%M),R%n,R%d",
	"RSB%C%S",	armdps, 0,	"(R%s%hR%M),R%n,R%d",
	"ADD%C%S",	armdps, armfadd,	"(R%s%hR%M),R%n,R%d",
	"ADC%C%S",	armdps, 0,	"(R%s%hR%M),R%n,R%d",
	"SBC%C%S",	armdps, 0,	"(R%s%hR%M),R%n,R%d",
	"RSC%C%S",	armdps, 0,	"(R%s%hR%M),R%n,R%d",
	"TST%C%S",	armdps, 0,	"(R%s%hR%M),R%n",
	"TEQ%C%S",	armdps, 0,	"(R%s%hR%M),R%n",
	"CMP%C%S",	armdps, 0,	"(R%s%hR%M),R%n",
	"CMN%C%S",	armdps, 0,	"(R%s%hR%M),R%n",
	"ORR%C%S",	armdps, 0,	"(R%s%hR%M),R%n,R%d",
	"MOVW%C%S",	armdps, armfmov,	"(R%s%hR%M),R%d",
	"BIC%C%S",	armdps, 0,	"(R%s%hR%M),R%n,R%d",
	"MVN%C%S",	armdps, 0,	"(R%s%hR%M),R%d",

/* 48 */
	"AND%C%S",	armdpi, 0,	"$#%i,R%n,R%d",
	"EOR%C%S",	armdpi, 0,	"$#%i,R%n,R%d",
	"SUB%C%S",	armdpi, 0,	"$#%i,R%n,R%d",
	"RSB%C%S",	armdpi, 0,	"$#%i,R%n,R%d",
	"ADD%C%S",	armdpi, armfadd,	"$#%i,R%n,R%d",
	"ADC%C%S",	armdpi, 0,	"$#%i,R%n,R%d",
	"SBC%C%S",	armdpi, 0,	"$#%i,R%n,R%d",
	"RSC%C%S",	armdpi, 0,	"$#%i,R%n,R%d",
	"TST%C%S",	armdpi, 0,	"$#%i,R%n",
	"TEQ%C%S",	armdpi, 0,	"$#%i,R%n",
	"CMP%C%S",	armdpi, 0,	"$#%i,R%n",
	"CMN%C%S",	armdpi, 0,	"$#%i,R%n",
	"ORR%C%S",	armdpi, 0,	"$#%i,R%n,R%d",
	"MOVW%C%S",	armdpi, armfmov,	"$#%i,R%d",
	"BIC%C%S",	armdpi, 0,	"$#%i,R%n,R%d",
	"MVN%C%S",	armdpi, 0,	"$#%i,R%d",

/* 48+16 */
	"MUL%C%S",	armdpi, 0,	"R%s,R%M,R%n",
	"MULA%C%S",	armdpi, 0,	"R%s,R%M,R%n,R%d",
	"SWPW",		armdpi, 0,	"R%s,(R%n),R%d",
	"SWPB",		armdpi, 0,	"R%s,(R%n),R%d",

/* 48+16+4 */
	"MOV%u%C%p",	armhwby, 0,	"R%d,(R%n%UR%M)",
	"MOV%u%C%p",	armhwby, 0,	"R%d,%I",
	"MOV%u%C%p",	armhwby, armfmov,	"(R%n%UR%M),R%d",
	"MOV%u%C%p",	armhwby, armfmov,	"%I,R%d",

/* 48+24 */
	"MOVW%C%p",	armsdti, 0,	"R%d,%I",
	"MOVB%C%p",	armsdti, 0,	"R%d,%I",
	"MOVW%C%p",	armsdti, armfmov,	"%I,R%d",
	"MOVBU%C%p",	armsdti, armfmov,	"%I,R%d",

	"MOVW%C%p",	armsdts, 0,	"R%d,(R%s%h%m)(R%n)",
	"MOVB%C%p",	armsdts, 0,	"R%d,(R%s%h%m)(R%n)",
	"MOVW%C%p",	armsdts, armfmov,	"(R%s%h%m)(R%n),R%d",
	"MOVBU%C%p",	armsdts, armfmov,	"(R%s%h%m)(R%n),R%d",

	"MOVM%C%P%a",	armbdt, armfmovm,		"[%r],(R%n)",
	"MOVM%C%P%a",	armbdt, armfmovm,		"(R%n),[%r]",

	"B%C",		armb, armfbranch,		"%b",
	"BL%C",		armb, armfbranch,		"%b",

	"CDP%C",	armco, 0,		"",
	"CDP%C",	armco, 0,		"",
	"MCR%C",	armco, 0,		"",
	"MRC%C",	armco, 0,		"",

	"UNK",		armunk, 0,	"",
};

static void
gaddr(Instr *i)
{
	*i->curr++ = '$';
	i->curr += gsymoff(i->curr, i->end-i->curr, i->imm, CANY);
}

static	char *mode[] = { 0, "IA", "DB", "IB" };
static	char *pw[] = { "P", "PW", 0, "W" };
static	char *sw[] = { 0, "W", "S", "SW" };

static void
format(char *mnemonic, Instr *i, char *f)
{
	int j, k, m, n;
	int g;
	char *fmt;

	if(mnemonic)
		format(0, i, mnemonic);
	if(f == 0)
		return;
	if(mnemonic)
		if(i->curr < i->end)
			*i->curr++ = '\t';
	for ( ; *f && i->curr < i->end; f++) {
		if(*f != '%') {
			*i->curr++ = *f;
			continue;
		}
		switch (*++f) {

		case 'C':	/* .CONDITION */
			if(cond[i->cond])
				bprint(i, ".%s", cond[i->cond]);
			break;

		case 'S':	/* .STORE */
			if(i->store)
				bprint(i, ".S");
			break;

		case 'P':	/* P & U bits for block move */
			n = (i->w >>23) & 0x3;
			if (mode[n])
				bprint(i, ".%s", mode[n]);
			break;

		case 'p':	/* P & W bits for single data xfer*/
			if (pw[i->store])
				bprint(i, ".%s", pw[i->store]);
			break;

		case 'a':	/* S & W bits for single data xfer*/
			if (sw[i->store])
				bprint(i, ".%s", sw[i->store]);
			break;

		case 's':
			bprint(i, "%d", i->rs & 0xf);
			break;

		case 'M':
			bprint(i, "%d", (i->w>>8) & 0xf);
			break;

		case 'm':
			bprint(i, "%d", (i->w>>7) & 0x1f);
			break;

		case 'h':
			bprint(i, shtype[(i->w>>5) & 0x3]);
			break;

		case 'u':		/* Signed/unsigned Byte/Halfword */
			bprint(i, hb[(i->w>>5) & 0x3]);
			break;

		case 'I':
			if (i->rn == 13) {
				if (plocal(i))
					break;
			}
			g = 0;
			fmt = "#%lx(R%d)";
			if (i->rn == 15) {
				/* convert load of offset(PC) to a load immediate */
				uint32 x;
				if (get4(i->map, i->addr+i->imm+8, &x) > 0)
				{
					i->imm = (int32)x;
					g = 1;
					fmt = "";
				}
			}
			if (mach->sb)
			{
				if (i->rd == 11) {
					uint32 nxti;

					if (get4(i->map, i->addr+4, &nxti) > 0) {
						if ((nxti & 0x0e0f0fff) == 0x060c000b) {
							i->imm += mach->sb;
							g = 1;
							fmt = "-SB";
						}
					}
				}
				if (i->rn == 12)
				{
					i->imm += mach->sb;
					g = 1;
					fmt = "-SB(SB)";
				}
			}
			if (g)
			{
				gaddr(i);
				bprint(i, fmt, i->rn);
			}
			else
				bprint(i, fmt, i->imm, i->rn);
			break;
		case 'U':		/* Add/subtract from base */
			bprint(i, addsub[(i->w >> 23) & 1]);
			break;

		case 'n':
			bprint(i, "%d", i->rn);
			break;

		case 'd':
			bprint(i, "%d", i->rd);
			break;

		case 'i':
			bprint(i, "%lux", i->imm);
			break;

		case 'b':
			i->curr += symoff(i->curr, i->end-i->curr,
				i->imm, CTEXT);
			break;

		case 'g':
			i->curr += gsymoff(i->curr, i->end-i->curr,
				i->imm, CANY);
			break;

		case 'r':
			n = i->imm&0xffff;
			j = 0;
			k = 0;
			while(n) {
				m = j;
				while(n&0x1) {
					j++;
					n >>= 1;
				}
				if(j != m) {
					if(k)
						bprint(i, ",");
					if(j == m+1)
						bprint(i, "R%d", m);
					else
						bprint(i, "R%d-R%d", m, j-1);
					k = 1;
				}
				j++;
				n >>= 1;
			}
			break;

		case '\0':
			*i->curr++ = '%';
			return;

		default:
			bprint(i, "%%%c", *f);
			break;
		}
	}
	*i->curr = 0;
}

static int
printins(Map *map, ulong pc, char *buf, int n)
{
	Instr i;

	i.curr = buf;
	i.end = buf+n-1;
	if(decode(map, pc, &i) < 0)
		return -1;

	(*opcodes[i.op].fmt)(&opcodes[i.op], &i);
	return 4;
}

static int
arminst(Map *map, uvlong pc, char modifier, char *buf, int n)
{
	USED(modifier);
	return printins(map, pc, buf, n);
}

static int
armdas(Map *map, uvlong pc, char *buf, int n)
{
	Instr i;

	i.curr = buf;
	i.end = buf+n;
	if(decode(map, pc, &i) < 0)
		return -1;
	if(i.end-i.curr > 8)
		i.curr = _hexify(buf, i.w, 7);
	*i.curr = 0;
	return 4;
}

static int
arminstlen(Map *map, uvlong pc)
{
	Instr i;

	if(decode(map, pc, &i) < 0)
		return -1;
	return 4;
}

static int
armfoll(Map *map, uvlong pc, Rgetter rget, uvlong *foll)
{
	ulong d;
	Instr i;

	if(decode(map, pc, &i) < 0)
		return -1;

	if(opcodes[i.op].foll) {
		d = (*opcodes[i.op].foll)(map, rget, &i, pc);
		if(d == -1)
			return -1;
	} else
		d = pc+4;

	foll[0] = d;
	return 1;
}
