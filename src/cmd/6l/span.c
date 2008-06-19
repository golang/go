// Inferno utils/6l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/span.c
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

static int	rexflag;
static int	asmode;

void
span(void)
{
	Prog *p, *q;
	long v;
	vlong c, idat;
	int m, n, again;

	xdefine("etext", STEXT, 0L);
	idat = INITDAT;
	for(p = firstp; p != P; p = p->link) {
		if(p->as == ATEXT)
			curtext = p;
		n = 0;
		if(p->to.type == D_BRANCH)
			if(p->pcond == P)
				p->pcond = p;
		if((q = p->pcond) != P)
			if(q->back != 2)
				n = 1;
		p->back = n;
		if(p->as == AADJSP) {
			p->to.type = D_SP;
			v = -p->from.offset;
			p->from.offset = v;
			p->as = p->mode != 64? AADDL: AADDQ;
			if(v < 0) {
				p->as = p->mode != 64? ASUBL: ASUBQ;
				v = -v;
				p->from.offset = v;
			}
			if(v == 0)
				p->as = ANOP;
		}
	}
	n = 0;

start:
	if(debug['v'])
		Bprint(&bso, "%5.2f span\n", cputime());
	Bflush(&bso);
	c = INITTEXT;
	for(p = firstp; p != P; p = p->link) {
		if(p->as == ATEXT)
			curtext = p;
		if(p->to.type == D_BRANCH)
			if(p->back)
				p->pc = c;
		asmins(p);
		p->pc = c;
		if(p->as == ATEXT)
			p->pc += SOFmark;	// skip the stack marker
		m = andptr-and;
		p->mark = m;
		c += m;
	}

loop:
	n++;
	if(debug['v'])
		Bprint(&bso, "%5.2f span %d\n", cputime(), n);
	Bflush(&bso);
	if(n > 50) {
		print("span must be looping\n");
		errorexit();
	}
	again = 0;
	c = INITTEXT;
	for(p = firstp; p != P; p = p->link) {
		if(p->as == ATEXT)
			curtext = p;
		if(p->to.type == D_BRANCH || p->back & 0100) {
			if(p->back)
				p->pc = c;
			asmins(p);
			m = andptr-and;
			if(m != p->mark) {
				p->mark = m;
				again++;
			}
		}
		p->pc = c;
		if(p->as == ATEXT)
			p->pc += SOFmark;	// skip the stack marker
		c += p->mark;
	}
	if(again) {
		textsize = c;
		goto loop;
	}
	if(INITRND) {
		INITDAT = rnd(c, INITRND);
		if(INITDAT != idat) {
			idat = INITDAT;
			goto start;
		}
	}
	xdefine("etext", STEXT, c);
	if(debug['v'])
		Bprint(&bso, "etext = %llux\n", c);
	Bflush(&bso);
	for(p = textp; p != P; p = p->pcond)
		p->from.sym->value = p->pc;
	textsize = c - INITTEXT;
}

void
xdefine(char *p, int t, vlong v)
{
	Sym *s;

	s = lookup(p, 0);
	if(s->type == 0 || s->type == SXREF) {
		s->type = t;
		s->value = v;
	}
	if(s->type == STEXT && s->value == 0)
		s->value = v;
}

void
putsymb(char *s, int t, vlong v, int ver)
{
	int i, f, l;

	if(t == 'f')
		s++;
	l = 4;
	if(!debug['8']){
		lput(v>>32);
		l = 8;
	}
	lput(v);
	if(ver)
		t += 'a' - 'A';
	cput(t+0x80);			/* 0x80 is variable length */

	if(t == 'Z' || t == 'z') {
		cput(s[0]);
		for(i=1; s[i] != 0 || s[i+1] != 0; i += 2) {
			cput(s[i]);
			cput(s[i+1]);
		}
		cput(0);
		cput(0);
		i++;
	}
	else {
		for(i=0; s[i]; i++)
			cput(s[i]);
		cput(0);
	}
	symsize += l + 1 + i + 1;

	if(debug['n']) {
		if(t == 'z' || t == 'Z') {
			Bprint(&bso, "%c %.8llux ", t, v);
			for(i=1; s[i] != 0 || s[i+1] != 0; i+=2) {
				f = ((s[i]&0xff) << 8) | (s[i+1]&0xff);
				Bprint(&bso, "/%x", f);
			}
			Bprint(&bso, "\n");
			return;
		}
		if(ver)
			Bprint(&bso, "%c %.8llux %s<%d>\n", t, v, s, ver);
		else
			Bprint(&bso, "%c %.8llux %s\n", t, v, s);
	}
}

void
asmsym(void)
{
	Prog *p;
	Auto *a;
	Sym *s;
	int h;

	s = lookup("etext", 0);
	if(s->type == STEXT)
		putsymb(s->name, 'T', s->value, s->version);

	for(h=0; h<NHASH; h++)
		for(s=hash[h]; s!=S; s=s->link)
			switch(s->type) {
			case SCONST:
				putsymb(s->name, 'D', s->value, s->version);
				continue;

			case SDATA:
				putsymb(s->name, 'D', s->value+INITDAT, s->version);
				continue;

			case SBSS:
				putsymb(s->name, 'B', s->value+INITDAT, s->version);
				continue;

			case SFILE:
				putsymb(s->name, 'f', s->value, s->version);
				continue;
			}

	for(p=textp; p!=P; p=p->pcond) {
		s = p->from.sym;
		if(s->type != STEXT)
			continue;

		/* filenames first */
		for(a=p->to.autom; a; a=a->link)
			if(a->type == D_FILE)
				putsymb(a->asym->name, 'z', a->aoffset, 0);
			else
			if(a->type == D_FILE1)
				putsymb(a->asym->name, 'Z', a->aoffset, 0);

		putsymb(s->name, 'T', s->value, s->version);

		/* frame, auto and param after */
		putsymb(".frame", 'm', p->to.offset+8, 0);

		for(a=p->to.autom; a; a=a->link)
			if(a->type == D_AUTO)
				putsymb(a->asym->name, 'a', -a->aoffset, 0);
			else
			if(a->type == D_PARAM)
				putsymb(a->asym->name, 'p', a->aoffset, 0);
	}
	if(debug['v'] || debug['n'])
		Bprint(&bso, "symsize = %lud\n", symsize);
	Bflush(&bso);
}

void
asmlc(void)
{
	vlong oldpc;
	Prog *p;
	long oldlc, v, s;

	oldpc = INITTEXT+SOFmark;
	oldlc = 0;
	for(p = firstp; p != P; p = p->link) {
		if(p->line == oldlc || p->as == ATEXT || p->as == ANOP) {
			if(p->as == ATEXT)
				curtext = p;
			if(debug['L'])
				Bprint(&bso, "%6llux %P\n",
					p->pc, p);
			continue;
		}
		if(debug['L'])
			Bprint(&bso, "\t\t%6ld", lcsize);
		v = (p->pc - oldpc) / MINLC;
		while(v) {
			s = 127;
			if(v < 127)
				s = v;
			cput(s+128);	/* 129-255 +pc */
			if(debug['L'])
				Bprint(&bso, " pc+%ld*%d(%ld)", s, MINLC, s+128);
			v -= s;
			lcsize++;
		}
		s = p->line - oldlc;
		oldlc = p->line;
		oldpc = p->pc + MINLC;
		if(s > 64 || s < -64) {
			cput(0);	/* 0 vv +lc */
			cput(s>>24);
			cput(s>>16);
			cput(s>>8);
			cput(s);
			if(debug['L']) {
				if(s > 0)
					Bprint(&bso, " lc+%ld(%d,%ld)\n",
						s, 0, s);
				else
					Bprint(&bso, " lc%ld(%d,%ld)\n",
						s, 0, s);
				Bprint(&bso, "%6llux %P\n",
					p->pc, p);
			}
			lcsize += 5;
			continue;
		}
		if(s > 0) {
			cput(0+s);	/* 1-64 +lc */
			if(debug['L']) {
				Bprint(&bso, " lc+%ld(%ld)\n", s, 0+s);
				Bprint(&bso, "%6llux %P\n",
					p->pc, p);
			}
		} else {
			cput(64-s);	/* 65-128 -lc */
			if(debug['L']) {
				Bprint(&bso, " lc%ld(%ld)\n", s, 64-s);
				Bprint(&bso, "%6llux %P\n",
					p->pc, p);
			}
		}
		lcsize++;
	}
	while(lcsize & 1) {
		s = 129;
		cput(s);
		lcsize++;
	}
	if(debug['v'] || debug['L'])
		Bprint(&bso, "lcsize = %ld\n", lcsize);
	Bflush(&bso);
}

int
oclass(Adr *a)
{
	vlong v;
	long l;

	if(a->type >= D_INDIR || a->index != D_NONE) {
		if(a->index != D_NONE && a->scale == 0) {
			if(a->type == D_ADDR) {
				switch(a->index) {
				case D_EXTERN:
				case D_STATIC:
					return Yi32;	/* TO DO: Yi64 */
				case D_AUTO:
				case D_PARAM:
					return Yiauto;
				}
				return Yxxx;
			}
			return Ycol;
		}
		return Ym;
	}
	switch(a->type)
	{
	case D_AL:
		return Yal;

	case D_AX:
		return Yax;

/*
	case D_SPB:
*/
	case D_BPB:
	case D_SIB:
	case D_DIB:
	case D_R8B:
	case D_R9B:
	case D_R10B:
	case D_R11B:
	case D_R12B:
	case D_R13B:
	case D_R14B:
	case D_R15B:
		if(asmode != 64)
			return Yxxx;
	case D_DL:
	case D_BL:
	case D_AH:
	case D_CH:
	case D_DH:
	case D_BH:
		return Yrb;

	case D_CL:
		return Ycl;

	case D_CX:
		return Ycx;

	case D_DX:
	case D_BX:
		return Yrx;

	case D_R8:	/* not really Yrl */
	case D_R9:
	case D_R10:
	case D_R11:
	case D_R12:
	case D_R13:
	case D_R14:
	case D_R15:
		if(asmode != 64)
			return Yxxx;
	case D_SP:
	case D_BP:
	case D_SI:
	case D_DI:
		return Yrl;

	case D_F0+0:
		return	Yf0;

	case D_F0+1:
	case D_F0+2:
	case D_F0+3:
	case D_F0+4:
	case D_F0+5:
	case D_F0+6:
	case D_F0+7:
		return	Yrf;

	case D_M0+0:
	case D_M0+1:
	case D_M0+2:
	case D_M0+3:
	case D_M0+4:
	case D_M0+5:
	case D_M0+6:
	case D_M0+7:
		return	Ymr;

	case D_X0+0:
	case D_X0+1:
	case D_X0+2:
	case D_X0+3:
	case D_X0+4:
	case D_X0+5:
	case D_X0+6:
	case D_X0+7:
	case D_X0+8:
	case D_X0+9:
	case D_X0+10:
	case D_X0+11:
	case D_X0+12:
	case D_X0+13:
	case D_X0+14:
	case D_X0+15:
		return	Yxr;

	case D_NONE:
		return Ynone;

	case D_CS:	return	Ycs;
	case D_SS:	return	Yss;
	case D_DS:	return	Yds;
	case D_ES:	return	Yes;
	case D_FS:	return	Yfs;
	case D_GS:	return	Ygs;

	case D_GDTR:	return	Ygdtr;
	case D_IDTR:	return	Yidtr;
	case D_LDTR:	return	Yldtr;
	case D_MSW:	return	Ymsw;
	case D_TASK:	return	Ytask;

	case D_CR+0:	return	Ycr0;
	case D_CR+1:	return	Ycr1;
	case D_CR+2:	return	Ycr2;
	case D_CR+3:	return	Ycr3;
	case D_CR+4:	return	Ycr4;
	case D_CR+5:	return	Ycr5;
	case D_CR+6:	return	Ycr6;
	case D_CR+7:	return	Ycr7;
	case D_CR+8:	return	Ycr8;

	case D_DR+0:	return	Ydr0;
	case D_DR+1:	return	Ydr1;
	case D_DR+2:	return	Ydr2;
	case D_DR+3:	return	Ydr3;
	case D_DR+4:	return	Ydr4;
	case D_DR+5:	return	Ydr5;
	case D_DR+6:	return	Ydr6;
	case D_DR+7:	return	Ydr7;

	case D_TR+0:	return	Ytr0;
	case D_TR+1:	return	Ytr1;
	case D_TR+2:	return	Ytr2;
	case D_TR+3:	return	Ytr3;
	case D_TR+4:	return	Ytr4;
	case D_TR+5:	return	Ytr5;
	case D_TR+6:	return	Ytr6;
	case D_TR+7:	return	Ytr7;

	case D_EXTERN:
	case D_STATIC:
	case D_AUTO:
	case D_PARAM:
		return Ym;

	case D_CONST:
	case D_ADDR:
		if(a->sym == S) {
			v = a->offset;
			if(v == 0)
				return Yi0;
			if(v == 1)
				return Yi1;
			if(v >= -128 && v <= 127)
				return Yi8;
			l = v;
			if((vlong)l == v)
				return Ys32;	/* can sign extend */
			if((v>>32) == 0)
				return Yi32;	/* unsigned */
			return Yi64;
		}
		return Yi32;	/* TO DO: D_ADDR as Yi64 */

	case D_BRANCH:
		return Ybr;
	}
	return Yxxx;
}

void
asmidx(Adr *a, int base)
{
	int i;

	switch(a->index) {
	default:
		goto bad;

	case D_NONE:
		i = 4 << 3;
		goto bas;

	case D_R8:
	case D_R9:
	case D_R10:
	case D_R11:
	case D_R12:
	case D_R13:
	case D_R14:
	case D_R15:
		if(asmode != 64)
			goto bad;
	case D_AX:
	case D_CX:
	case D_DX:
	case D_BX:
	case D_BP:
	case D_SI:
	case D_DI:
		i = reg[(int)a->index] << 3;
		break;
	}
	switch(a->scale) {
	default:
		goto bad;
	case 1:
		break;
	case 2:
		i |= (1<<6);
		break;
	case 4:
		i |= (2<<6);
		break;
	case 8:
		i |= (3<<6);
		break;
	}
bas:
	switch(base) {
	default:
		goto bad;
	case D_NONE:	/* must be mod=00 */
		i |= 5;
		break;
	case D_R8:
	case D_R9:
	case D_R10:
	case D_R11:
	case D_R12:
	case D_R13:
	case D_R14:
	case D_R15:
		if(asmode != 64)
			goto bad;
	case D_AX:
	case D_CX:
	case D_DX:
	case D_BX:
	case D_SP:
	case D_BP:
	case D_SI:
	case D_DI:
		i |= reg[base];
		break;
	}
	*andptr++ = i;
	return;
bad:
	diag("asmidx: bad address %D", a);
	*andptr++ = 0;
	return;
}

static void
put4(long v)
{
	if(dlm && curp != P && reloca != nil){
		dynreloc(reloca->sym, curp->pc + andptr - &and[0], 1);
		reloca = nil;
	}
	andptr[0] = v;
	andptr[1] = v>>8;
	andptr[2] = v>>16;
	andptr[3] = v>>24;
	andptr += 4;
}

static void
put8(vlong v)
{
	if(dlm && curp != P && reloca != nil){
		dynreloc(reloca->sym, curp->pc + andptr - &and[0], 1);	/* TO DO */
		reloca = nil;
	}
	andptr[0] = v;
	andptr[1] = v>>8;
	andptr[2] = v>>16;
	andptr[3] = v>>24;
	andptr[4] = v>>32;
	andptr[5] = v>>40;
	andptr[6] = v>>48;
	andptr[7] = v>>56;
	andptr += 8;
}

vlong
vaddr(Adr *a)
{
	int t;
	vlong v;
	Sym *s;

	t = a->type;
	v = a->offset;
	if(t == D_ADDR)
		t = a->index;
	switch(t) {
	case D_STATIC:
	case D_EXTERN:
		s = a->sym;
		if(s != nil) {
			if(dlm && curp != P)
				reloca = a;
			switch(s->type) {
			case SUNDEF:
				ckoff(s, v);
			case STEXT:
			case SCONST:
				if((uvlong)s->value < (uvlong)INITTEXT)
					v += INITTEXT;	/* TO DO */
				v += s->value;
				break;
			default:
				v += INITDAT + s->value;
			}
		}
	}
	return v;
}

static void
asmandsz(Adr *a, int r, int rex, int m64)
{
	long v;
	int t;
	Adr aa;

	rex &= (0x40 | Rxr);
	v = a->offset;
	t = a->type;
	if(a->index != D_NONE) {
		if(t >= D_INDIR) {
			t -= D_INDIR;
			rexflag |= (regrex[(int)a->index] & Rxx) | (regrex[t] & Rxb) | rex;
			if(t == D_NONE) {
				*andptr++ = (0 << 6) | (4 << 0) | (r << 3);
				asmidx(a, t);
				put4(v);
				return;
			}
			if(v == 0 && t != D_BP && t != D_R13) {
				*andptr++ = (0 << 6) | (4 << 0) | (r << 3);
				asmidx(a, t);
				return;
			}
			if(v >= -128 && v < 128) {
				*andptr++ = (1 << 6) | (4 << 0) | (r << 3);
				asmidx(a, t);
				*andptr++ = v;
				return;
			}
			*andptr++ = (2 << 6) | (4 << 0) | (r << 3);
			asmidx(a, t);
			put4(v);
			return;
		}
		switch(t) {
		default:
			goto bad;
		case D_STATIC:
		case D_EXTERN:
			aa.type = D_NONE+D_INDIR;
			break;
		case D_AUTO:
		case D_PARAM:
			aa.type = D_SP+D_INDIR;
			break;
		}
		aa.offset = vaddr(a);
		aa.index = a->index;
		aa.scale = a->scale;
		asmandsz(&aa, r, rex, m64);
		return;
	}
	if(t >= D_AL && t <= D_X0+15) {
		if(v)
			goto bad;
		*andptr++ = (3 << 6) | (reg[t] << 0) | (r << 3);
		rexflag |= (regrex[t] & (0x40 | Rxb)) | rex;
		return;
	}
	if(t >= D_INDIR) {
		t -= D_INDIR;
		rexflag |= (regrex[t] & Rxb) | rex;
		if(t == D_NONE) {
			if(asmode != 64){
				*andptr++ = (0 << 6) | (5 << 0) | (r << 3);
				put4(v);
				return;
			}
			/* temporary */
			*andptr++ = (0 <<  6) | (4 << 0) | (r << 3);	/* sib present */
			*andptr++ = (0 << 6) | (4 << 3) | (5 << 0);	/* DS:d32 */
			put4(v);
			return;
		}
		if(t == D_SP || t == D_R12) {
			if(v == 0) {
				*andptr++ = (0 << 6) | (reg[t] << 0) | (r << 3);
				asmidx(a, t);
				return;
			}
			if(v >= -128 && v < 128) {
				*andptr++ = (1 << 6) | (reg[t] << 0) | (r << 3);
				asmidx(a, t);
				*andptr++ = v;
				return;
			}
			*andptr++ = (2 << 6) | (reg[t] << 0) | (r << 3);
			asmidx(a, t);
			put4(v);
			return;
		}
		if(t >= D_AX && t <= D_R15) {
			if(v == 0 && t != D_BP && t != D_R13) {
				*andptr++ = (0 << 6) | (reg[t] << 0) | (r << 3);
				return;
			}
			if(v >= -128 && v < 128) {
				andptr[0] = (1 << 6) | (reg[t] << 0) | (r << 3);
				andptr[1] = v;
				andptr += 2;
				return;
			}
			*andptr++ = (2 << 6) | (reg[t] << 0) | (r << 3);
			put4(v);
			return;
		}
		goto bad;
	}
	switch(a->type) {
	default:
		goto bad;
	case D_STATIC:
	case D_EXTERN:
		aa.type = D_NONE+D_INDIR;
		break;
	case D_AUTO:
	case D_PARAM:
		aa.type = D_SP+D_INDIR;
		break;
	}
	aa.index = D_NONE;
	aa.scale = 1;
	aa.offset = vaddr(a);
	asmandsz(&aa, r, rex, m64);
	return;
bad:
	diag("asmand: bad address %D", a);
	return;
}

void
asmand(Adr *a, Adr *ra)
{
	asmandsz(a, reg[ra->type], regrex[ra->type], 0);
}

void
asmando(Adr *a, int o)
{
	asmandsz(a, o, 0, 0);
}

static void
bytereg(Adr *a)
{
	if(a->index == D_NONE && (a->type >= D_AX && a->type <= D_R15))
		a->type = D_AL + (a->type-D_AX);
}

#define	E	0xff
Movtab	ymovtab[] =
{
/* push */
	{APUSHL,	Ycs,	Ynone,	0,	0x0e,E,0,0},
	{APUSHL,	Yss,	Ynone,	0,	0x16,E,0,0},
	{APUSHL,	Yds,	Ynone,	0,	0x1e,E,0,0},
	{APUSHL,	Yes,	Ynone,	0,	0x06,E,0,0},
	{APUSHL,	Yfs,	Ynone,	0,	0x0f,0xa0,E,0},
	{APUSHL,	Ygs,	Ynone,	0,	0x0f,0xa8,E,0},
	{APUSHQ,	Yfs,	Ynone,	0,	0x0f,0xa0,E,0},
	{APUSHQ,	Ygs,	Ynone,	0,	0x0f,0xa8,E,0},

	{APUSHW,	Ycs,	Ynone,	0,	Pe,0x0e,E,0},
	{APUSHW,	Yss,	Ynone,	0,	Pe,0x16,E,0},
	{APUSHW,	Yds,	Ynone,	0,	Pe,0x1e,E,0},
	{APUSHW,	Yes,	Ynone,	0,	Pe,0x06,E,0},
	{APUSHW,	Yfs,	Ynone,	0,	Pe,0x0f,0xa0,E},
	{APUSHW,	Ygs,	Ynone,	0,	Pe,0x0f,0xa8,E},

/* pop */
	{APOPL,	Ynone,	Yds,	0,	0x1f,E,0,0},
	{APOPL,	Ynone,	Yes,	0,	0x07,E,0,0},
	{APOPL,	Ynone,	Yss,	0,	0x17,E,0,0},
	{APOPL,	Ynone,	Yfs,	0,	0x0f,0xa1,E,0},
	{APOPL,	Ynone,	Ygs,	0,	0x0f,0xa9,E,0},
	{APOPQ,	Ynone,	Yfs,	0,	0x0f,0xa1,E,0},
	{APOPQ,	Ynone,	Ygs,	0,	0x0f,0xa9,E,0},

	{APOPW,	Ynone,	Yds,	0,	Pe,0x1f,E,0},
	{APOPW,	Ynone,	Yes,	0,	Pe,0x07,E,0},
	{APOPW,	Ynone,	Yss,	0,	Pe,0x17,E,0},
	{APOPW,	Ynone,	Yfs,	0,	Pe,0x0f,0xa1,E},
	{APOPW,	Ynone,	Ygs,	0,	Pe,0x0f,0xa9,E},

/* mov seg */
	{AMOVW,	Yes,	Yml,	1,	0x8c,0,0,0},
	{AMOVW,	Ycs,	Yml,	1,	0x8c,1,0,0},
	{AMOVW,	Yss,	Yml,	1,	0x8c,2,0,0},
	{AMOVW,	Yds,	Yml,	1,	0x8c,3,0,0},
	{AMOVW,	Yfs,	Yml,	1,	0x8c,4,0,0},
	{AMOVW,	Ygs,	Yml,	1,	0x8c,5,0,0},

	{AMOVW,	Yml,	Yes,	2,	0x8e,0,0,0},
	{AMOVW,	Yml,	Ycs,	2,	0x8e,1,0,0},
	{AMOVW,	Yml,	Yss,	2,	0x8e,2,0,0},
	{AMOVW,	Yml,	Yds,	2,	0x8e,3,0,0},
	{AMOVW,	Yml,	Yfs,	2,	0x8e,4,0,0},
	{AMOVW,	Yml,	Ygs,	2,	0x8e,5,0,0},

/* mov cr */
	{AMOVL,	Ycr0,	Yml,	3,	0x0f,0x20,0,0},
	{AMOVL,	Ycr2,	Yml,	3,	0x0f,0x20,2,0},
	{AMOVL,	Ycr3,	Yml,	3,	0x0f,0x20,3,0},
	{AMOVL,	Ycr4,	Yml,	3,	0x0f,0x20,4,0},
	{AMOVL,	Ycr8,	Yml,	3,	0x0f,0x20,8,0},
	{AMOVQ,	Ycr0,	Yml,	3,	0x0f,0x20,0,0},
	{AMOVQ,	Ycr2,	Yml,	3,	0x0f,0x20,2,0},
	{AMOVQ,	Ycr3,	Yml,	3,	0x0f,0x20,3,0},
	{AMOVQ,	Ycr4,	Yml,	3,	0x0f,0x20,4,0},
	{AMOVQ,	Ycr8,	Yml,	3,	0x0f,0x20,8,0},

	{AMOVL,	Yml,	Ycr0,	4,	0x0f,0x22,0,0},
	{AMOVL,	Yml,	Ycr2,	4,	0x0f,0x22,2,0},
	{AMOVL,	Yml,	Ycr3,	4,	0x0f,0x22,3,0},
	{AMOVL,	Yml,	Ycr4,	4,	0x0f,0x22,4,0},
	{AMOVL,	Yml,	Ycr8,	4,	0x0f,0x22,8,0},
	{AMOVQ,	Yml,	Ycr0,	4,	0x0f,0x22,0,0},
	{AMOVQ,	Yml,	Ycr2,	4,	0x0f,0x22,2,0},
	{AMOVQ,	Yml,	Ycr3,	4,	0x0f,0x22,3,0},
	{AMOVQ,	Yml,	Ycr4,	4,	0x0f,0x22,4,0},
	{AMOVQ,	Yml,	Ycr8,	4,	0x0f,0x22,8,0},

/* mov dr */
	{AMOVL,	Ydr0,	Yml,	3,	0x0f,0x21,0,0},
	{AMOVL,	Ydr6,	Yml,	3,	0x0f,0x21,6,0},
	{AMOVL,	Ydr7,	Yml,	3,	0x0f,0x21,7,0},
	{AMOVQ,	Ydr0,	Yml,	3,	0x0f,0x21,0,0},
	{AMOVQ,	Ydr6,	Yml,	3,	0x0f,0x21,6,0},
	{AMOVQ,	Ydr7,	Yml,	3,	0x0f,0x21,7,0},

	{AMOVL,	Yml,	Ydr0,	4,	0x0f,0x23,0,0},
	{AMOVL,	Yml,	Ydr6,	4,	0x0f,0x23,6,0},
	{AMOVL,	Yml,	Ydr7,	4,	0x0f,0x23,7,0},
	{AMOVQ,	Yml,	Ydr0,	4,	0x0f,0x23,0,0},
	{AMOVQ,	Yml,	Ydr6,	4,	0x0f,0x23,6,0},
	{AMOVQ,	Yml,	Ydr7,	4,	0x0f,0x23,7,0},

/* mov tr */
	{AMOVL,	Ytr6,	Yml,	3,	0x0f,0x24,6,0},
	{AMOVL,	Ytr7,	Yml,	3,	0x0f,0x24,7,0},

	{AMOVL,	Yml,	Ytr6,	4,	0x0f,0x26,6,E},
	{AMOVL,	Yml,	Ytr7,	4,	0x0f,0x26,7,E},

/* lgdt, sgdt, lidt, sidt */
	{AMOVL,	Ym,	Ygdtr,	4,	0x0f,0x01,2,0},
	{AMOVL,	Ygdtr,	Ym,	3,	0x0f,0x01,0,0},
	{AMOVL,	Ym,	Yidtr,	4,	0x0f,0x01,3,0},
	{AMOVL,	Yidtr,	Ym,	3,	0x0f,0x01,1,0},
	{AMOVQ,	Ym,	Ygdtr,	4,	0x0f,0x01,2,0},
	{AMOVQ,	Ygdtr,	Ym,	3,	0x0f,0x01,0,0},
	{AMOVQ,	Ym,	Yidtr,	4,	0x0f,0x01,3,0},
	{AMOVQ,	Yidtr,	Ym,	3,	0x0f,0x01,1,0},

/* lldt, sldt */
	{AMOVW,	Yml,	Yldtr,	4,	0x0f,0x00,2,0},
	{AMOVW,	Yldtr,	Yml,	3,	0x0f,0x00,0,0},

/* lmsw, smsw */
	{AMOVW,	Yml,	Ymsw,	4,	0x0f,0x01,6,0},
	{AMOVW,	Ymsw,	Yml,	3,	0x0f,0x01,4,0},

/* ltr, str */
	{AMOVW,	Yml,	Ytask,	4,	0x0f,0x00,3,0},
	{AMOVW,	Ytask,	Yml,	3,	0x0f,0x00,1,0},

/* load full pointer */
	{AMOVL,	Yml,	Ycol,	5,	0,0,0,0},
	{AMOVW,	Yml,	Ycol,	5,	Pe,0,0,0},

/* double shift */
	{ASHLL,	Ycol,	Yml,	6,	0xa4,0xa5,0,0},
	{ASHRL,	Ycol,	Yml,	6,	0xac,0xad,0,0},
	{ASHLQ,	Ycol,	Yml,	6,	Pw,0xa4,0xa5,0},
	{ASHRQ,	Ycol,	Yml,	6,	Pw,0xac,0xad,0},
	{ASHLW,	Ycol,	Yml,	6,	Pe,0xa4,0xa5,0},
	{ASHRW,	Ycol,	Yml,	6,	Pe,0xac,0xad,0},
	0
};

int
isax(Adr *a)
{

	switch(a->type) {
	case D_AX:
	case D_AL:
	case D_AH:
	case D_INDIR+D_AX:
		return 1;
	}
	if(a->index == D_AX)
		return 1;
	return 0;
}

void
subreg(Prog *p, int from, int to)
{

	if(debug['Q'])
		print("\n%P	s/%R/%R/\n", p, from, to);

	if(p->from.type == from)
		p->from.type = to;
	if(p->to.type == from)
		p->to.type = to;

	if(p->from.index == from)
		p->from.index = to;
	if(p->to.index == from)
		p->to.index = to;

	from += D_INDIR;
	if(p->from.type == from)
		p->from.type = to+D_INDIR;
	if(p->to.type == from)
		p->to.type = to+D_INDIR;

	if(debug['Q'])
		print("%P\n", p);
}

static int
mediaop(Optab *o, int op, int osize, int z)
{
	switch(op){
	case Pm:
	case Pe:
	case Pf2:
	case Pf3:
		if(osize != 1){
			if(op != Pm)
				*andptr++ = op;
			*andptr++ = Pm;
			op = o->op[++z];
			break;
		}
	default:
		if(andptr == and || andptr[-1] != Pm)
			*andptr++ = Pm;
		break;
	}
	*andptr++ = op;
	return z;
}

void
doasm(Prog *p)
{
	Optab *o;
	Prog *q, pp;
	uchar *t;
	Movtab *mo;
	int z, op, ft, tt, xo, l;
	vlong v;

	o = opindex[p->as];
	if(o == nil) {
		diag("asmins: missing op %P", p);
		return;
	}
	ft = oclass(&p->from) * Ymax;
	tt = oclass(&p->to) * Ymax;
	t = o->ytab;
	if(t == 0) {
		diag("asmins: noproto %P", p);
		return;
	}
	xo = o->op[0] == 0x0f;
	for(z=0; *t; z+=t[3]+xo,t+=4)
		if(ycover[ft+t[0]])
		if(ycover[tt+t[1]])
			goto found;
	goto domov;

found:
	switch(o->prefix) {
	case Pq:	/* 16 bit escape and opcode escape */
		*andptr++ = Pe;
		*andptr++ = Pm;
		break;

	case Pf2:	/* xmm opcode escape */
	case Pf3:
		*andptr++ = o->prefix;
		*andptr++ = Pm;
		break;

	case Pm:	/* opcode escape */
		*andptr++ = Pm;
		break;

	case Pe:	/* 16 bit escape */
		*andptr++ = Pe;
		break;

	case Pw:	/* 64-bit escape */
		if(p->mode != 64)
			diag("asmins: illegal 64: %P", p);
		rexflag |= Pw;
		break;

	case Pb:	/* botch */
		bytereg(&p->from);
		bytereg(&p->to);
		break;

	case P32:	/* 32 bit but illegal if 64-bit mode */
		if(p->mode == 64)
			diag("asmins: illegal in 64-bit mode: %P", p);
		break;

	case Py:	/* 64-bit only, no prefix */
		if(p->mode != 64)
			diag("asmins: illegal in %d-bit mode: %P", p->mode, p);
		break;
	}
	v = vaddr(&p->from);
	op = o->op[z];
	if(op == 0x0f) {
		*andptr++ = op;
		op = o->op[++z];
	}
	switch(t[2]) {
	default:
		diag("asmins: unknown z %d %P", t[2], p);
		return;

	case Ztext:
		v = p->to.offset;
		if(v < 0)
			v = 0;

		// eleven bytes of buried stack offset
		*andptr++ = v>>3;
		*andptr++ = v>>11;
		*andptr++ = v>>19;
		for(v=0; v<SOFmark-3; v++)
			*andptr++ = "\xa7\xf1\xd9\x2a\x82\xc8\xd8\xfe"[v];
		break;

	case Zpseudo:
		break;

	case Zlit:
		for(; op = o->op[z]; z++)
			*andptr++ = op;
		break;

	case Zmb_r:
		bytereg(&p->from);
		/* fall through */
	case Zm_r:
		*andptr++ = op;
		asmand(&p->from, &p->to);
		break;

	case Zm_r_xm:
		mediaop(o, op, t[3], z);
		asmand(&p->from, &p->to);
		break;

	case Zm_r_xm_nr:
		rexflag = 0;
		mediaop(o, op, t[3], z);
		asmand(&p->from, &p->to);
		break;

	case Zm_r_i_xm:
		mediaop(o, op, t[3], z);
		asmand(&p->from, &p->to);
		*andptr++ = p->to.offset;
		break;

	case Zm_r_3d:
		*andptr++ = 0x0f;
		*andptr++ = 0x0f;
		asmand(&p->from, &p->to);
		*andptr++ = op;
		break;

	case Zibm_r:
		*andptr++ = op;
		asmand(&p->from, &p->to);
		*andptr++ = p->to.offset;
		break;

	case Zaut_r:
		*andptr++ = 0x8d;	/* leal */
		if(p->from.type != D_ADDR)
			diag("asmins: Zaut sb type ADDR");
		p->from.type = p->from.index;
		p->from.index = D_NONE;
		asmand(&p->from, &p->to);
		p->from.index = p->from.type;
		p->from.type = D_ADDR;
		break;

	case Zm_o:
		*andptr++ = op;
		asmando(&p->from, o->op[z+1]);
		break;

	case Zr_m:
		*andptr++ = op;
		asmand(&p->to, &p->from);
		break;

	case Zr_m_xm:
		mediaop(o, op, t[3], z);
		asmand(&p->to, &p->from);
		break;

	case Zr_m_xm_nr:
		rexflag = 0;
		mediaop(o, op, t[3], z);
		asmand(&p->to, &p->from);
		break;

	case Zr_m_i_xm:
		mediaop(o, op, t[3], z);
		asmand(&p->to, &p->from);
		*andptr++ = p->from.offset;
		break;

	case Zo_m:
		*andptr++ = op;
		asmando(&p->to, o->op[z+1]);
		break;

	case Zo_m64:
		*andptr++ = op;
		asmandsz(&p->to, o->op[z+1], 0, 1);
		break;

	case Zm_ibo:
		v = vaddr(&p->to);
		*andptr++ = op;
		asmando(&p->from, o->op[z+1]);
		*andptr++ = v;
		break;

	case Zibo_m:
		*andptr++ = op;
		asmando(&p->to, o->op[z+1]);
		*andptr++ = v;
		break;

	case Zibo_m_xm:
		z = mediaop(o, op, t[3], z);
		asmando(&p->to, o->op[z+1]);
		*andptr++ = v;
		break;

	case Z_ib:
		v = vaddr(&p->to);
	case Zib_:
		*andptr++ = op;
		*andptr++ = v;
		break;

	case Zib_rp:
		rexflag |= regrex[p->to.type] & (Rxb|0x40);
		*andptr++ = op + reg[p->to.type];
		*andptr++ = v;
		break;

	case Zil_rp:
		rexflag |= regrex[p->to.type] & Rxb;
		*andptr++ = op + reg[p->to.type];
		if(o->prefix == Pe) {
			*andptr++ = v;
			*andptr++ = v>>8;
		}
		else
			put4(v);
		break;

	case Zo_iw:
		*andptr++ = op;
		if(p->from.type != D_NONE){
			*andptr++ = v;
			*andptr++ = v>>8;
		}
		break;

	case Ziq_rp:
		l = v>>32;
		if(l == 0){
			//p->mark |= 0100;
			//print("zero: %llux %P\n", v, p);
			rexflag &= ~(0x40|Rxw);
			rexflag |= regrex[p->to.type] & Rxb;
			*andptr++ = 0xb8 + reg[p->to.type];
			put4(v);
		}else if(l == -1 && (v&((uvlong)1<<31))!=0){	/* sign extend */
			//p->mark |= 0100;
			//print("sign: %llux %P\n", v, p);
			*andptr ++ = 0xc7;
			asmando(&p->to, 0);
			put4(v);
		}else{	/* need all 8 */
			//print("all: %llux %P\n", v, p);
			rexflag |= regrex[p->to.type] & Rxb;
			*andptr++ = op + reg[p->to.type];
			put8(v);
		}
		break;

	case Zib_rr:
		*andptr++ = op;
		asmand(&p->to, &p->to);
		*andptr++ = v;
		break;

	case Z_il:
		v = vaddr(&p->to);
	case Zil_:
		*andptr++ = op;
		if(o->prefix == Pe) {
			*andptr++ = v;
			*andptr++ = v>>8;
		}
		else
			put4(v);
		break;

	case Zm_ilo:
		v = vaddr(&p->to);
		*andptr++ = op;
		asmando(&p->from, o->op[z+1]);
		if(o->prefix == Pe) {
			*andptr++ = v;
			*andptr++ = v>>8;
		}
		else
			put4(v);
		break;

	case Zilo_m:
		*andptr++ = op;
		asmando(&p->to, o->op[z+1]);
		if(o->prefix == Pe) {
			*andptr++ = v;
			*andptr++ = v>>8;
		}
		else
			put4(v);
		break;

	case Zil_rr:
		*andptr++ = op;
		asmand(&p->to, &p->to);
		if(o->prefix == Pe) {
			*andptr++ = v;
			*andptr++ = v>>8;
		}
		else
			put4(v);
		break;

	case Z_rp:
		rexflag |= regrex[p->to.type] & (Rxb|0x40);
		*andptr++ = op + reg[p->to.type];
		break;

	case Zrp_:
		rexflag |= regrex[p->from.type] & (Rxb|0x40);
		*andptr++ = op + reg[p->from.type];
		break;

	case Zclr:
		*andptr++ = op;
		asmand(&p->to, &p->to);
		break;

	case Zbr:
		q = p->pcond;
		if(q) {
			v = q->pc - p->pc - 2;
			if(v >= -128 && v <= 127) {
				*andptr++ = op;
				*andptr++ = v;
			} else {
				v -= 6-2;
				*andptr++ = 0x0f;
				*andptr++ = o->op[z+1];
				*andptr++ = v;
				*andptr++ = v>>8;
				*andptr++ = v>>16;
				*andptr++ = v>>24;
			}
		}
		break;

	case Zcall:
		q = p->pcond;
		if(q) {
			v = q->pc - p->pc - 5;
			if(dlm && curp != P && p->to.sym->type == SUNDEF){
				/* v = 0 - p->pc - 5; */
				v = 0;
				ckoff(p->to.sym, v);
				v += p->to.sym->value;
				dynreloc(p->to.sym, p->pc+1, 0);
			}
			*andptr++ = op;
			*andptr++ = v;
			*andptr++ = v>>8;
			*andptr++ = v>>16;
			*andptr++ = v>>24;
		}
		break;

	case Zjmp:
		q = p->pcond;
		if(q) {
			v = q->pc - p->pc - 2;
			if(v >= -128 && v <= 127) {
				*andptr++ = op;
				*andptr++ = v;
			} else {
				v -= 5-2;
				*andptr++ = o->op[z+1];
				*andptr++ = v;
				*andptr++ = v>>8;
				*andptr++ = v>>16;
				*andptr++ = v>>24;
			}
		}
		break;

	case Zloop:
		q = p->pcond;
		if(q) {
			v = q->pc - p->pc - 2;
			if(v < -128 && v > 127)
				diag("loop too far: %P", p);
			*andptr++ = op;
			*andptr++ = v;
		}
		break;

	case Zbyte:
		*andptr++ = v;
		if(op > 1) {
			*andptr++ = v>>8;
			if(op > 2) {
				*andptr++ = v>>16;
				*andptr++ = v>>24;
				if(op > 4) {
					*andptr++ = v>>32;
					*andptr++ = v>>40;
					*andptr++ = v>>48;
					*andptr++ = v>>56;
				}
			}
		}
		break;
	}
	return;

domov:
	for(mo=ymovtab; mo->as; mo++)
		if(p->as == mo->as)
		if(ycover[ft+mo->ft])
		if(ycover[tt+mo->tt]){
			t = mo->op;
			goto mfound;
		}
bad:
	if(p->mode != 64){
		/*
		 * here, the assembly has failed.
		 * if its a byte instruction that has
		 * unaddressable registers, try to
		 * exchange registers and reissue the
		 * instruction with the operands renamed.
		 */
		pp = *p;
		z = p->from.type;
		if(z >= D_BP && z <= D_DI) {
			if(isax(&p->to)) {
				*andptr++ = 0x87;			/* xchg lhs,bx */
				asmando(&p->from, reg[D_BX]);
				subreg(&pp, z, D_BX);
				doasm(&pp);
				*andptr++ = 0x87;			/* xchg lhs,bx */
				asmando(&p->from, reg[D_BX]);
			} else {
				*andptr++ = 0x90 + reg[z];		/* xchg lsh,ax */
				subreg(&pp, z, D_AX);
				doasm(&pp);
				*andptr++ = 0x90 + reg[z];		/* xchg lsh,ax */
			}
			return;
		}
		z = p->to.type;
		if(z >= D_BP && z <= D_DI) {
			if(isax(&p->from)) {
				*andptr++ = 0x87;			/* xchg rhs,bx */
				asmando(&p->to, reg[D_BX]);
				subreg(&pp, z, D_BX);
				doasm(&pp);
				*andptr++ = 0x87;			/* xchg rhs,bx */
				asmando(&p->to, reg[D_BX]);
			} else {
				*andptr++ = 0x90 + reg[z];		/* xchg rsh,ax */
				subreg(&pp, z, D_AX);
				doasm(&pp);
				*andptr++ = 0x90 + reg[z];		/* xchg rsh,ax */
			}
			return;
		}
	}
	diag("doasm: notfound from=%ux to=%ux %P", p->from.type, p->to.type, p);
	return;

mfound:
	switch(mo->code) {
	default:
		diag("asmins: unknown mov %d %P", mo->code, p);
		break;

	case 0:	/* lit */
		for(z=0; t[z]!=E; z++)
			*andptr++ = t[z];
		break;

	case 1:	/* r,m */
		*andptr++ = t[0];
		asmando(&p->to, t[1]);
		break;

	case 2:	/* m,r */
		*andptr++ = t[0];
		asmando(&p->from, t[1]);
		break;

	case 3:	/* r,m - 2op */
		*andptr++ = t[0];
		*andptr++ = t[1];
		asmando(&p->to, t[2]);
		rexflag |= regrex[p->from.type] & (Rxr|0x40);
		break;

	case 4:	/* m,r - 2op */
		*andptr++ = t[0];
		*andptr++ = t[1];
		asmando(&p->from, t[2]);
		rexflag |= regrex[p->to.type] & (Rxr|0x40);
		break;

	case 5:	/* load full pointer, trash heap */
		if(t[0])
			*andptr++ = t[0];
		switch(p->to.index) {
		default:
			goto bad;
		case D_DS:
			*andptr++ = 0xc5;
			break;
		case D_SS:
			*andptr++ = 0x0f;
			*andptr++ = 0xb2;
			break;
		case D_ES:
			*andptr++ = 0xc4;
			break;
		case D_FS:
			*andptr++ = 0x0f;
			*andptr++ = 0xb4;
			break;
		case D_GS:
			*andptr++ = 0x0f;
			*andptr++ = 0xb5;
			break;
		}
		asmand(&p->from, &p->to);
		break;

	case 6:	/* double shift */
		if(t[0] == Pw){
			if(p->mode != 64)
				diag("asmins: illegal 64: %P", p);
			rexflag |= Pw;
			t++;
		}else if(t[0] == Pe){
			*andptr++ = Pe;
			t++;
		}
		z = p->from.type;
		switch(z) {
		default:
			goto bad;
		case D_CONST:
			*andptr++ = 0x0f;
			*andptr++ = t[0];
			asmandsz(&p->to, reg[(int)p->from.index], regrex[(int)p->from.index], 0);
			*andptr++ = p->from.offset;
			break;
		case D_CL:
		case D_CX:
			*andptr++ = 0x0f;
			*andptr++ = t[1];
			asmandsz(&p->to, reg[(int)p->from.index], regrex[(int)p->from.index], 0);
			break;
		}
		break;
	}
}

void
asmins(Prog *p)
{
	int n, np, c;

	rexflag = 0;
	andptr = and;
	asmode = p->mode;
	doasm(p);
	if(rexflag){
		/*
		 * as befits the whole approach of the architecture,
		 * the rex prefix must appear before the first opcode byte
		 * (and thus after any 66/67/f2/f3 prefix bytes, but
		 * before the 0f opcode escape!), or it might be ignored.
		 * note that the handbook often misleadingly shows 66/f2/f3 in `opcode'.
		 */
		if(p->mode != 64)
			diag("asmins: illegal in mode %d: %P", p->mode, p);
		n = andptr - and;
		for(np = 0; np < n; np++) {
			c = and[np];
			if(c != 0x66 && c != 0xf2 && c != 0xf3 && c != 0x67)
				break;
		}
		memmove(and+np+1, and+np, n-np);
		and[np] = 0x40 | rexflag;
		andptr++;
	}
}

enum{
	ABSD = 0,
	ABSU = 1,
	RELD = 2,
	RELU = 3,
};

int modemap[4] = { 0, 1, -1, 2, };

typedef struct Reloc Reloc;

struct Reloc
{
	int n;
	int t;
	uchar *m;
	ulong *a;
};

Reloc rels;

static void
grow(Reloc *r)
{
	int t;
	uchar *m, *nm;
	ulong *a, *na;

	t = r->t;
	r->t += 64;
	m = r->m;
	a = r->a;
	r->m = nm = malloc(r->t*sizeof(uchar));
	r->a = na = malloc(r->t*sizeof(ulong));
	memmove(nm, m, t*sizeof(uchar));
	memmove(na, a, t*sizeof(ulong));
	free(m);
	free(a);
}

void
dynreloc(Sym *s, ulong v, int abs)
{
	int i, k, n;
	uchar *m;
	ulong *a;
	Reloc *r;

	if(s->type == SUNDEF)
		k = abs ? ABSU : RELU;
	else
		k = abs ? ABSD : RELD;
	/* Bprint(&bso, "R %s a=%ld(%lx) %d\n", s->name, v, v, k); */
	k = modemap[k];
	r = &rels;
	n = r->n;
	if(n >= r->t)
		grow(r);
	m = r->m;
	a = r->a;
	for(i = n; i > 0; i--){
		if(v < a[i-1]){	/* happens occasionally for data */
			m[i] = m[i-1];
			a[i] = a[i-1];
		}
		else
			break;
	}
	m[i] = k;
	a[i] = v;
	r->n++;
}

static int
sput(char *s)
{
	char *p;

	p = s;
	while(*s)
		cput(*s++);
	cput(0);
	return s-p+1;
}

void
asmdyn()
{
	int i, n, t, c;
	Sym *s;
	ulong la, ra, *a;
	vlong off;
	uchar *m;
	Reloc *r;

	cflush();
	off = seek(cout, 0, 1);
	lput(0);
	t = 0;
	lput(imports);
	t += 4;
	for(i = 0; i < NHASH; i++)
		for(s = hash[i]; s != S; s = s->link)
			if(s->type == SUNDEF){
				lput(s->sig);
				t += 4;
				t += sput(s->name);
			}

	la = 0;
	r = &rels;
	n = r->n;
	m = r->m;
	a = r->a;
	lput(n);
	t += 4;
	for(i = 0; i < n; i++){
		ra = *a-la;
		if(*a < la)
			diag("bad relocation order");
		if(ra < 256)
			c = 0;
		else if(ra < 65536)
			c = 1;
		else
			c = 2;
		cput((c<<6)|*m++);
		t++;
		if(c == 0){
			cput(ra);
			t++;
		}
		else if(c == 1){
			wput(ra);
			t += 2;
		}
		else{
			lput(ra);
			t += 4;
		}
		la = *a++;
	}

	cflush();
	seek(cout, off, 0);
	lput(t);

	if(debug['v']){
		Bprint(&bso, "import table entries = %d\n", imports);
		Bprint(&bso, "export table entries = %d\n", exports);
	}
}
