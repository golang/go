// Inferno utils/8l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/8l/span.c
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

#include	"l.h"
#include	"../ld/lib.h"
#include	"../ld/elf.h"

static int32	vaddr(Adr*, Reloc*);

void
span1(Sym *s)
{
	Prog *p, *q;
	int32 c, v, loop;
	uchar *bp;
	int n, m, i;

	cursym = s;

	for(p = s->text; p != P; p = p->link) {
		p->back = 2;	// use short branches first time through
		if((q = p->pcond) != P && (q->back & 2))
			p->back |= 1;	// backward jump

		if(p->as == AADJSP) {
			p->to.type = D_SP;
			v = -p->from.offset;
			p->from.offset = v;
			p->as = AADDL;
			if(v < 0) {
				p->as = ASUBL;
				v = -v;
				p->from.offset = v;
			}
			if(v == 0)
				p->as = ANOP;
		}
	}
	
	n = 0;
	do {
		loop = 0;
		memset(s->r, 0, s->nr*sizeof s->r[0]);
		s->nr = 0;
		s->np = 0;
		c = 0;
		for(p = s->text; p != P; p = p->link) {
			p->pc = c;

			// process forward jumps to p
			for(q = p->comefrom; q != P; q = q->forwd) {
				v = p->pc - (q->pc + q->mark);
				if(q->back & 2)	{	// short
					if(v > 127) {
						loop++;
						q->back ^= 2;
					}
					if(q->as == AJCXZW)
						s->p[q->pc+2] = v;
					else
						s->p[q->pc+1] = v;
				} else {
					bp = s->p + q->pc + q->mark - 4;
					*bp++ = v;
					*bp++ = v>>8;
					*bp++ = v>>16;
					*bp = v>>24;
				}	
			}
			p->comefrom = P;

			asmins(p);
			p->pc = c;
			m = andptr-and;
			symgrow(s, p->pc+m);
			memmove(s->p+p->pc, and, m);
			p->mark = m;
			c += m;
		}
		if(++n > 20) {
			diag("span must be looping");
			errorexit();
		}
	} while(loop);
	s->size = c;

	if(debug['a'] > 1) {
		print("span1 %s %d (%d tries)\n %.6ux", s->name, s->size, n, 0);
		for(i=0; i<s->np; i++) {
			print(" %.2ux", s->p[i]);
			if(i%16 == 15)
				print("\n  %.6ux", i+1);
		}
		if(i%16)
			print("\n");
	
		for(i=0; i<s->nr; i++) {
			Reloc *r;
			
			r = &s->r[i];
			print(" rel %#.4ux/%d %s%+d\n", r->off, r->siz, r->sym->name, r->add);
		}
	}
}

void
span(void)
{
	Prog *p, *q;
	int32 v;
	int n;

	if(debug['v'])
		Bprint(&bso, "%5.2f span\n", cputime());

	// NOTE(rsc): If we get rid of the globals we should
	// be able to parallelize these iterations.
	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		if(cursym->text == nil || cursym->text->link == nil)
			continue;

		// TODO: move into span1
		for(p = cursym->text; p != P; p = p->link) {
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
				p->as = AADDL;
				if(v < 0) {
					p->as = ASUBL;
					v = -v;
					p->from.offset = v;
				}
				if(v == 0)
					p->as = ANOP;
			}
		}
		span1(cursym);
	}
}

void
xdefine(char *p, int t, int32 v)
{
	Sym *s;

	s = lookup(p, 0);
	s->type = t;
	s->value = v;
	s->reachable = 1;
	s->special = 1;
}

void
instinit(void)
{
	int i;

	for(i=1; optab[i].as; i++)
		if(i != optab[i].as) {
			diag("phase error in optab: at %A found %A", i, optab[i].as);
			errorexit();
		}
	maxop = i;

	for(i=0; i<Ymax; i++)
		ycover[i*Ymax + i] = 1;

	ycover[Yi0*Ymax + Yi8] = 1;
	ycover[Yi1*Ymax + Yi8] = 1;

	ycover[Yi0*Ymax + Yi32] = 1;
	ycover[Yi1*Ymax + Yi32] = 1;
	ycover[Yi8*Ymax + Yi32] = 1;

	ycover[Yal*Ymax + Yrb] = 1;
	ycover[Ycl*Ymax + Yrb] = 1;
	ycover[Yax*Ymax + Yrb] = 1;
	ycover[Ycx*Ymax + Yrb] = 1;
	ycover[Yrx*Ymax + Yrb] = 1;

	ycover[Yax*Ymax + Yrx] = 1;
	ycover[Ycx*Ymax + Yrx] = 1;

	ycover[Yax*Ymax + Yrl] = 1;
	ycover[Ycx*Ymax + Yrl] = 1;
	ycover[Yrx*Ymax + Yrl] = 1;

	ycover[Yf0*Ymax + Yrf] = 1;

	ycover[Yal*Ymax + Ymb] = 1;
	ycover[Ycl*Ymax + Ymb] = 1;
	ycover[Yax*Ymax + Ymb] = 1;
	ycover[Ycx*Ymax + Ymb] = 1;
	ycover[Yrx*Ymax + Ymb] = 1;
	ycover[Yrb*Ymax + Ymb] = 1;
	ycover[Ym*Ymax + Ymb] = 1;

	ycover[Yax*Ymax + Yml] = 1;
	ycover[Ycx*Ymax + Yml] = 1;
	ycover[Yrx*Ymax + Yml] = 1;
	ycover[Yrl*Ymax + Yml] = 1;
	ycover[Ym*Ymax + Yml] = 1;

	ycover[Yax*Ymax + Ymm] = 1;
	ycover[Ycx*Ymax + Ymm] = 1;
	ycover[Yrx*Ymax + Ymm] = 1;
	ycover[Yrl*Ymax + Ymm] = 1;
	ycover[Ym*Ymax + Ymm] = 1;
	ycover[Ymr*Ymax + Ymm] = 1;

	ycover[Ym*Ymax + Yxm] = 1;
	ycover[Yxr*Ymax + Yxm] = 1;

	for(i=0; i<D_NONE; i++) {
		reg[i] = -1;
		if(i >= D_AL && i <= D_BH)
			reg[i] = (i-D_AL) & 7;
		if(i >= D_AX && i <= D_DI)
			reg[i] = (i-D_AX) & 7;
		if(i >= D_F0 && i <= D_F0+7)
			reg[i] = (i-D_F0) & 7;
		if(i >= D_X0 && i <= D_X0+7)
			reg[i] = (i-D_X0) & 7;
	}
}

int
prefixof(Adr *a)
{
	switch(a->type) {
	case D_INDIR+D_CS:
		return 0x2e;
	case D_INDIR+D_DS:
		return 0x3e;
	case D_INDIR+D_ES:
		return 0x26;
	case D_INDIR+D_FS:
		return 0x64;
	case D_INDIR+D_GS:
		return 0x65;
	}
	return 0;
}

int
oclass(Adr *a)
{
	int32 v;

	if((a->type >= D_INDIR && a->type < 2*D_INDIR) || a->index != D_NONE) {
		if(a->index != D_NONE && a->scale == 0) {
			if(a->type == D_ADDR) {
				switch(a->index) {
				case D_EXTERN:
				case D_STATIC:
					return Yi32;
				case D_AUTO:
				case D_PARAM:
					return Yiauto;
				}
				return Yxxx;
			}
			//if(a->type == D_INDIR+D_ADDR)
			//	print("*Ycol\n");
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

	case D_CL:
	case D_DL:
	case D_BL:
	case D_AH:
	case D_CH:
	case D_DH:
	case D_BH:
		return Yrb;

	case D_CX:
		return Ycx;

	case D_DX:
	case D_BX:
		return Yrx;

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

	case D_X0+0:
	case D_X0+1:
	case D_X0+2:
	case D_X0+3:
	case D_X0+4:
	case D_X0+5:
	case D_X0+6:
	case D_X0+7:
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
	case D_CONST2:
	case D_ADDR:
		if(a->sym == S) {
			v = a->offset;
			if(v == 0)
				return Yi0;
			if(v == 1)
				return Yi1;
			if(v >= -128 && v <= 127)
				return Yi8;
		}
		return Yi32;

	case D_BRANCH:
		return Ybr;
	}
	return Yxxx;
}

void
asmidx(int scale, int index, int base)
{
	int i;

	switch(index) {
	default:
		goto bad;

	case D_NONE:
		i = 4 << 3;
		goto bas;

	case D_AX:
	case D_CX:
	case D_DX:
	case D_BX:
	case D_BP:
	case D_SI:
	case D_DI:
		i = reg[index] << 3;
		break;
	}
	switch(scale) {
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
	diag("asmidx: bad address %d,%d,%d", scale, index, base);
	*andptr++ = 0;
	return;
}

static void
put4(int32 v)
{
	andptr[0] = v;
	andptr[1] = v>>8;
	andptr[2] = v>>16;
	andptr[3] = v>>24;
	andptr += 4;
}

static void
relput4(Prog *p, Adr *a)
{
	vlong v;
	Reloc rel, *r;
	
	v = vaddr(a, &rel);
	if(rel.siz != 0) {
		if(rel.siz != 4)
			diag("bad reloc");
		r = addrel(cursym);
		*r = rel;
		r->off = p->pc + andptr - and;
	}
	put4(v);
}

int32
symaddr(Sym *s)
{
	if(!s->reachable)
		diag("unreachable symbol in symaddr - %s", s->name);
	return s->value;
}

static int32
vaddr(Adr *a, Reloc *r)
{
	int t;
	int32 v;
	Sym *s;
	
	if(r != nil)
		memset(r, 0, sizeof *r);

	t = a->type;
	v = a->offset;
	if(t == D_ADDR)
		t = a->index;
	switch(t) {
	case D_STATIC:
	case D_EXTERN:
		s = a->sym;
		if(s != nil) {
			if(!s->reachable)
				sysfatal("unreachable symbol in vaddr - %s", s->name);
			if(r == nil) {
				diag("need reloc for %D", a);
				errorexit();
			}
			r->type = D_ADDR;
			r->siz = 4;
			r->off = -1;
			r->sym = s;
			r->add = v;
			v = 0;
		}
	}
	return v;
}

static int
istls(Adr *a)
{
	if(HEADTYPE == Hlinux)
		return a->index == D_GS;
	return a->type == D_INDIR+D_GS;
}

void
asmand(Adr *a, int r)
{
	int32 v;
	int t, scale;
	Reloc rel;

	v = a->offset;
	t = a->type;
	rel.siz = 0;
	if(a->index != D_NONE && a->index != D_FS && a->index != D_GS) {
		if(t < D_INDIR || t >= 2*D_INDIR) {
			switch(t) {
			default:
				goto bad;
			case D_STATIC:
			case D_EXTERN:
				t = D_NONE;
				v = vaddr(a, &rel);
				break;
			case D_AUTO:
			case D_PARAM:
				t = D_SP;
				break;
			}
		} else
			t -= D_INDIR;

		if(t == D_NONE) {
			*andptr++ = (0 << 6) | (4 << 0) | (r << 3);
			asmidx(a->scale, a->index, t);
			goto putrelv;
		}
		if(v == 0 && rel.siz == 0 && t != D_BP) {
			*andptr++ = (0 << 6) | (4 << 0) | (r << 3);
			asmidx(a->scale, a->index, t);
			return;
		}
		if(v >= -128 && v < 128 && rel.siz == 0) {
			*andptr++ = (1 << 6) | (4 << 0) | (r << 3);
			asmidx(a->scale, a->index, t);
			*andptr++ = v;
			return;
		}
		*andptr++ = (2 << 6) | (4 << 0) | (r << 3);
		asmidx(a->scale, a->index, t);
		goto putrelv;
	}
	if(t >= D_AL && t <= D_F7 || t >= D_X0 && t <= D_X7) {
		if(v)
			goto bad;
		*andptr++ = (3 << 6) | (reg[t] << 0) | (r << 3);
		return;
	}
	
	scale = a->scale;
	if(t < D_INDIR || t >= 2*D_INDIR) {
		switch(a->type) {
		default:
			goto bad;
		case D_STATIC:
		case D_EXTERN:
			t = D_NONE;
			v = vaddr(a, &rel);
			break;
		case D_AUTO:
		case D_PARAM:
			t = D_SP;
			break;
		}
		scale = 1;
	} else
		t -= D_INDIR;

	if(t == D_NONE || (D_CS <= t && t <= D_GS)) {
		*andptr++ = (0 << 6) | (5 << 0) | (r << 3);
		goto putrelv;
	}
	if(t == D_SP) {
		if(v == 0 && rel.siz == 0) {
			*andptr++ = (0 << 6) | (4 << 0) | (r << 3);
			asmidx(scale, D_NONE, t);
			return;
		}
		if(v >= -128 && v < 128 && rel.siz == 0) {
			*andptr++ = (1 << 6) | (4 << 0) | (r << 3);
			asmidx(scale, D_NONE, t);
			*andptr++ = v;
			return;
		}
		*andptr++ = (2 << 6) | (4 << 0) | (r << 3);
		asmidx(scale, D_NONE, t);
		goto putrelv;
	}
	if(t >= D_AX && t <= D_DI) {
		if(v == 0 && rel.siz == 0 && t != D_BP) {
			*andptr++ = (0 << 6) | (reg[t] << 0) | (r << 3);
			return;
		}
		if(v >= -128 && v < 128 && rel.siz == 0 && a->index != D_FS && a->index != D_GS) {
			andptr[0] = (1 << 6) | (reg[t] << 0) | (r << 3);
			andptr[1] = v;
			andptr += 2;
			return;
		}
		*andptr++ = (2 << 6) | (reg[t] << 0) | (r << 3);
		goto putrelv;
	}
	goto bad;

putrelv:
	if(rel.siz != 0) {
		Reloc *r;
		
		if(rel.siz != 4) {
			diag("bad rel");
			goto bad;
		}
		r = addrel(cursym);
		*r = rel;
		r->off = curp->pc + andptr - and;
	} else if(iself && linkmode == LinkExternal && istls(a) && HEADTYPE != Hopenbsd) {
		Reloc *r;
		Sym *s;

		r = addrel(cursym);
		r->off = curp->pc + andptr - and;
		r->add = 0;
		r->xadd = 0;
		r->siz = 4;
		r->type = D_TLS;
		if(a->offset == tlsoffset+0)
			s = lookup("runtime.g", 0);
		else
			s = lookup("runtime.m", 0);
		s->type = STLSBSS;
		s->reachable = 1;
		s->hide = 1;
		s->size = PtrSize;
		r->sym = s;
		r->xsym = s;
		v = 0;
	}

	put4(v);
	return;

bad:
	diag("asmand: bad address %D", a);
	return;
}

#define	E	0xff
uchar	ymovtab[] =
{
/* push */
	APUSHL,	Ycs,	Ynone,	0,	0x0e,E,0,0,
	APUSHL,	Yss,	Ynone,	0,	0x16,E,0,0,
	APUSHL,	Yds,	Ynone,	0,	0x1e,E,0,0,
	APUSHL,	Yes,	Ynone,	0,	0x06,E,0,0,
	APUSHL,	Yfs,	Ynone,	0,	0x0f,0xa0,E,0,
	APUSHL,	Ygs,	Ynone,	0,	0x0f,0xa8,E,0,

	APUSHW,	Ycs,	Ynone,	0,	Pe,0x0e,E,0,
	APUSHW,	Yss,	Ynone,	0,	Pe,0x16,E,0,
	APUSHW,	Yds,	Ynone,	0,	Pe,0x1e,E,0,
	APUSHW,	Yes,	Ynone,	0,	Pe,0x06,E,0,
	APUSHW,	Yfs,	Ynone,	0,	Pe,0x0f,0xa0,E,
	APUSHW,	Ygs,	Ynone,	0,	Pe,0x0f,0xa8,E,

/* pop */
	APOPL,	Ynone,	Yds,	0,	0x1f,E,0,0,
	APOPL,	Ynone,	Yes,	0,	0x07,E,0,0,
	APOPL,	Ynone,	Yss,	0,	0x17,E,0,0,
	APOPL,	Ynone,	Yfs,	0,	0x0f,0xa1,E,0,
	APOPL,	Ynone,	Ygs,	0,	0x0f,0xa9,E,0,

	APOPW,	Ynone,	Yds,	0,	Pe,0x1f,E,0,
	APOPW,	Ynone,	Yes,	0,	Pe,0x07,E,0,
	APOPW,	Ynone,	Yss,	0,	Pe,0x17,E,0,
	APOPW,	Ynone,	Yfs,	0,	Pe,0x0f,0xa1,E,
	APOPW,	Ynone,	Ygs,	0,	Pe,0x0f,0xa9,E,

/* mov seg */
	AMOVW,	Yes,	Yml,	1,	0x8c,0,0,0,
	AMOVW,	Ycs,	Yml,	1,	0x8c,1,0,0,
	AMOVW,	Yss,	Yml,	1,	0x8c,2,0,0,
	AMOVW,	Yds,	Yml,	1,	0x8c,3,0,0,
	AMOVW,	Yfs,	Yml,	1,	0x8c,4,0,0,
	AMOVW,	Ygs,	Yml,	1,	0x8c,5,0,0,

	AMOVW,	Yml,	Yes,	2,	0x8e,0,0,0,
	AMOVW,	Yml,	Ycs,	2,	0x8e,1,0,0,
	AMOVW,	Yml,	Yss,	2,	0x8e,2,0,0,
	AMOVW,	Yml,	Yds,	2,	0x8e,3,0,0,
	AMOVW,	Yml,	Yfs,	2,	0x8e,4,0,0,
	AMOVW,	Yml,	Ygs,	2,	0x8e,5,0,0,

/* mov cr */
	AMOVL,	Ycr0,	Yml,	3,	0x0f,0x20,0,0,
	AMOVL,	Ycr2,	Yml,	3,	0x0f,0x20,2,0,
	AMOVL,	Ycr3,	Yml,	3,	0x0f,0x20,3,0,
	AMOVL,	Ycr4,	Yml,	3,	0x0f,0x20,4,0,

	AMOVL,	Yml,	Ycr0,	4,	0x0f,0x22,0,0,
	AMOVL,	Yml,	Ycr2,	4,	0x0f,0x22,2,0,
	AMOVL,	Yml,	Ycr3,	4,	0x0f,0x22,3,0,
	AMOVL,	Yml,	Ycr4,	4,	0x0f,0x22,4,0,

/* mov dr */
	AMOVL,	Ydr0,	Yml,	3,	0x0f,0x21,0,0,
	AMOVL,	Ydr6,	Yml,	3,	0x0f,0x21,6,0,
	AMOVL,	Ydr7,	Yml,	3,	0x0f,0x21,7,0,

	AMOVL,	Yml,	Ydr0,	4,	0x0f,0x23,0,0,
	AMOVL,	Yml,	Ydr6,	4,	0x0f,0x23,6,0,
	AMOVL,	Yml,	Ydr7,	4,	0x0f,0x23,7,0,

/* mov tr */
	AMOVL,	Ytr6,	Yml,	3,	0x0f,0x24,6,0,
	AMOVL,	Ytr7,	Yml,	3,	0x0f,0x24,7,0,

	AMOVL,	Yml,	Ytr6,	4,	0x0f,0x26,6,E,
	AMOVL,	Yml,	Ytr7,	4,	0x0f,0x26,7,E,

/* lgdt, sgdt, lidt, sidt */
	AMOVL,	Ym,	Ygdtr,	4,	0x0f,0x01,2,0,
	AMOVL,	Ygdtr,	Ym,	3,	0x0f,0x01,0,0,
	AMOVL,	Ym,	Yidtr,	4,	0x0f,0x01,3,0,
	AMOVL,	Yidtr,	Ym,	3,	0x0f,0x01,1,0,

/* lldt, sldt */
	AMOVW,	Yml,	Yldtr,	4,	0x0f,0x00,2,0,
	AMOVW,	Yldtr,	Yml,	3,	0x0f,0x00,0,0,

/* lmsw, smsw */
	AMOVW,	Yml,	Ymsw,	4,	0x0f,0x01,6,0,
	AMOVW,	Ymsw,	Yml,	3,	0x0f,0x01,4,0,

/* ltr, str */
	AMOVW,	Yml,	Ytask,	4,	0x0f,0x00,3,0,
	AMOVW,	Ytask,	Yml,	3,	0x0f,0x00,1,0,

/* load full pointer */
	AMOVL,	Yml,	Ycol,	5,	0,0,0,0,
	AMOVW,	Yml,	Ycol,	5,	Pe,0,0,0,

/* double shift */
	ASHLL,	Ycol,	Yml,	6,	0xa4,0xa5,0,0,
	ASHRL,	Ycol,	Yml,	6,	0xac,0xad,0,0,

/* extra imul */
	AIMULW,	Yml,	Yrl,	7,	Pq,0xaf,0,0,
	AIMULL,	Yml,	Yrl,	7,	Pm,0xaf,0,0,
	0
};

// byteswapreg returns a byte-addressable register (AX, BX, CX, DX)
// which is not referenced in a->type.
// If a is empty, it returns BX to account for MULB-like instructions
// that might use DX and AX.
int
byteswapreg(Adr *a)
{
	int cana, canb, canc, cand;

	cana = canb = canc = cand = 1;

	switch(a->type) {
	case D_NONE:
		cana = cand = 0;
		break;
	case D_AX:
	case D_AL:
	case D_AH:
	case D_INDIR+D_AX:
		cana = 0;
		break;
	case D_BX:
	case D_BL:
	case D_BH:
	case D_INDIR+D_BX:
		canb = 0;
		break;
	case D_CX:
	case D_CL:
	case D_CH:
	case D_INDIR+D_CX:
		canc = 0;
		break;
	case D_DX:
	case D_DL:
	case D_DH:
	case D_INDIR+D_DX:
		cand = 0;
		break;
	}
	switch(a->index) {
	case D_AX:
		cana = 0;
		break;
	case D_BX:
		canb = 0;
		break;
	case D_CX:
		canc = 0;
		break;
	case D_DX:
		cand = 0;
		break;
	}
	if(cana)
		return D_AX;
	if(canb)
		return D_BX;
	if(canc)
		return D_CX;
	if(cand)
		return D_DX;

	diag("impossible byte register");
	errorexit();
	return 0;
}

void
subreg(Prog *p, int from, int to)
{

	if(debug['Q'])
		print("\n%P	s/%R/%R/\n", p, from, to);

	if(p->from.type == from) {
		p->from.type = to;
		p->ft = 0;
	}
	if(p->to.type == from) {
		p->to.type = to;
		p->tt = 0;
	}

	if(p->from.index == from) {
		p->from.index = to;
		p->ft = 0;
	}
	if(p->to.index == from) {
		p->to.index = to;
		p->tt = 0;
	}

	from += D_INDIR;
	if(p->from.type == from) {
		p->from.type = to+D_INDIR;
		p->ft = 0;
	}
	if(p->to.type == from) {
		p->to.type = to+D_INDIR;
		p->tt = 0;
	}

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
	int z, op, ft, tt, breg;
	int32 v, pre;
	Reloc rel, *r;
	Adr *a;
	
	curp = p;	// TODO

	pre = prefixof(&p->from);
	if(pre)
		*andptr++ = pre;
	pre = prefixof(&p->to);
	if(pre)
		*andptr++ = pre;

	if(p->ft == 0)
		p->ft = oclass(&p->from);
	if(p->tt == 0)
		p->tt = oclass(&p->to);

	ft = p->ft * Ymax;
	tt = p->tt * Ymax;
	o = &optab[p->as];
	t = o->ytab;
	if(t == 0) {
		diag("asmins: noproto %P", p);
		return;
	}
	for(z=0; *t; z+=t[3],t+=4)
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

	case Pb:	/* botch */
		break;
	}

	op = o->op[z];
	switch(t[2]) {
	default:
		diag("asmins: unknown z %d %P", t[2], p);
		return;

	case Zpseudo:
		break;

	case Zlit:
		for(; op = o->op[z]; z++)
			*andptr++ = op;
		break;

	case Zlitm_r:
		for(; op = o->op[z]; z++)
			*andptr++ = op;
		asmand(&p->from, reg[p->to.type]);
		break;

	case Zm_r:
		*andptr++ = op;
		asmand(&p->from, reg[p->to.type]);
		break;

	case Zm2_r:
		*andptr++ = op;
		*andptr++ = o->op[z+1];
		asmand(&p->from, reg[p->to.type]);
		break;

	case Zm_r_xm:
		mediaop(o, op, t[3], z);
		asmand(&p->from, reg[p->to.type]);
		break;

	case Zm_r_i_xm:
		mediaop(o, op, t[3], z);
		asmand(&p->from, reg[p->to.type]);
		*andptr++ = p->to.offset;
		break;

	case Zibm_r:
		while ((op = o->op[z++]) != 0)
			*andptr++ = op;
		asmand(&p->from, reg[p->to.type]);
		*andptr++ = p->to.offset;
		break;

	case Zaut_r:
		*andptr++ = 0x8d;	/* leal */
		if(p->from.type != D_ADDR)
			diag("asmins: Zaut sb type ADDR");
		p->from.type = p->from.index;
		p->from.index = D_NONE;
		p->ft = 0;
		asmand(&p->from, reg[p->to.type]);
		p->from.index = p->from.type;
		p->from.type = D_ADDR;
		p->ft = 0;
		break;

	case Zm_o:
		*andptr++ = op;
		asmand(&p->from, o->op[z+1]);
		break;

	case Zr_m:
		*andptr++ = op;
		asmand(&p->to, reg[p->from.type]);
		break;

	case Zr_m_xm:
		mediaop(o, op, t[3], z);
		asmand(&p->to, reg[p->from.type]);
		break;

	case Zr_m_i_xm:
		mediaop(o, op, t[3], z);
		asmand(&p->to, reg[p->from.type]);
		*andptr++ = p->from.offset;
		break;

	case Zo_m:
		*andptr++ = op;
		asmand(&p->to, o->op[z+1]);
		break;

	case Zm_ibo:
		*andptr++ = op;
		asmand(&p->from, o->op[z+1]);
		*andptr++ = vaddr(&p->to, nil);
		break;

	case Zibo_m:
		*andptr++ = op;
		asmand(&p->to, o->op[z+1]);
		*andptr++ = vaddr(&p->from, nil);
		break;

	case Z_ib:
	case Zib_:
		if(t[2] == Zib_)
			a = &p->from;
		else
			a = &p->to;
		v = vaddr(a, nil);
		*andptr++ = op;
		*andptr++ = v;
		break;

	case Zib_rp:
		*andptr++ = op + reg[p->to.type];
		*andptr++ = vaddr(&p->from, nil);
		break;

	case Zil_rp:
		*andptr++ = op + reg[p->to.type];
		if(o->prefix == Pe) {
			v = vaddr(&p->from, nil);
			*andptr++ = v;
			*andptr++ = v>>8;
		}
		else
			relput4(p, &p->from);
		break;

	case Zib_rr:
		*andptr++ = op;
		asmand(&p->to, reg[p->to.type]);
		*andptr++ = vaddr(&p->from, nil);
		break;

	case Z_il:
	case Zil_:
		if(t[2] == Zil_)
			a = &p->from;
		else
			a = &p->to;
		*andptr++ = op;
		if(o->prefix == Pe) {
			v = vaddr(a, nil);
			*andptr++ = v;
			*andptr++ = v>>8;
		}
		else
			relput4(p, a);
		break;

	case Zm_ilo:
	case Zilo_m:
		*andptr++ = op;
		if(t[2] == Zilo_m) {
			a = &p->from;
			asmand(&p->to, o->op[z+1]);
		} else {
			a = &p->to;
			asmand(&p->from, o->op[z+1]);
		}
		if(o->prefix == Pe) {
			v = vaddr(a, nil);
			*andptr++ = v;
			*andptr++ = v>>8;
		}
		else
			relput4(p, a);
		break;

	case Zil_rr:
		*andptr++ = op;
		asmand(&p->to, reg[p->to.type]);
		if(o->prefix == Pe) {
			v = vaddr(&p->from, nil);
			*andptr++ = v;
			*andptr++ = v>>8;
		}
		else
			relput4(p, &p->from);
		break;

	case Z_rp:
		*andptr++ = op + reg[p->to.type];
		break;

	case Zrp_:
		*andptr++ = op + reg[p->from.type];
		break;

	case Zclr:
		*andptr++ = op;
		asmand(&p->to, reg[p->to.type]);
		break;
	
	case Zcall:
		q = p->pcond;
		if(q == nil) {
			diag("call without target");
			errorexit();
		}
		if(q->as != ATEXT) {
			// Could handle this case by making D_PCREL
			// record the Prog* instead of the Sym*, but let's
			// wait until the need arises.
			diag("call of non-TEXT %P", q);
			errorexit();
		}
		*andptr++ = op;
		r = addrel(cursym);
		r->off = p->pc + andptr - and;
		r->type = D_PCREL;
		r->siz = 4;
		r->sym = q->from.sym;
		put4(0);
		break;

	case Zbr:
	case Zjmp:
	case Zloop:
		q = p->pcond;
		if(q == nil) {
			diag("jmp/branch/loop without target");
			errorexit();
		}
		if(q->as == ATEXT) {
			// jump out of function
			if(t[2] == Zbr) {
				diag("branch to ATEXT");
				errorexit();
			}
			*andptr++ = o->op[z+1];
			r = addrel(cursym);
			r->off = p->pc + andptr - and;
			r->sym = q->from.sym;
			r->type = D_PCREL;
			r->siz = 4;
			put4(0);
			break;
		}
		
		// Assumes q is in this function.
		// TODO: Check in input, preserve in brchain.
		
		// Fill in backward jump now.
		if(p->back & 1) {
			v = q->pc - (p->pc + 2);
			if(v >= -128) {
				if(p->as == AJCXZW)
					*andptr++ = 0x67;
				*andptr++ = op;
				*andptr++ = v;
			} else if(t[2] == Zloop) {
				diag("loop too far: %P", p);
			} else {
				v -= 5-2;
				if(t[2] == Zbr) {
					*andptr++ = 0x0f;
					v--;
				}
				*andptr++ = o->op[z+1];
				*andptr++ = v;
				*andptr++ = v>>8;
				*andptr++ = v>>16;
				*andptr++ = v>>24;
			}
			break;
		}

		// Annotate target; will fill in later.
		p->forwd = q->comefrom;
		q->comefrom = p;
		if(p->back & 2)	{ // short
			if(p->as == AJCXZW)
				*andptr++ = 0x67;
			*andptr++ = op;
			*andptr++ = 0;
		} else if(t[2] == Zloop) {
			diag("loop too far: %P", p);
		} else {
			if(t[2] == Zbr)
				*andptr++ = 0x0f;
			*andptr++ = o->op[z+1];
			*andptr++ = 0;
			*andptr++ = 0;
			*andptr++ = 0;
			*andptr++ = 0;
		}
		break;

	case Zcallcon:
	case Zjmpcon:
		if(t[2] == Zcallcon)
			*andptr++ = op;
		else
			*andptr++ = o->op[z+1];
		r = addrel(cursym);
		r->off = p->pc + andptr - and;
		r->type = D_PCREL;
		r->siz = 4;
		r->add = p->to.offset;
		put4(0);
		break;
	
	case Zcallind:
		*andptr++ = op;
		*andptr++ = o->op[z+1];
		r = addrel(cursym);
		r->off = p->pc + andptr - and;
		r->type = D_ADDR;
		r->siz = 4;
		r->add = p->to.offset;
		r->sym = p->to.sym;
		put4(0);
		break;

	case Zbyte:
		v = vaddr(&p->from, &rel);
		if(rel.siz != 0) {
			rel.siz = op;
			r = addrel(cursym);
			*r = rel;
			r->off = p->pc + andptr - and;
		}
		*andptr++ = v;
		if(op > 1) {
			*andptr++ = v>>8;
			if(op > 2) {
				*andptr++ = v>>16;
				*andptr++ = v>>24;
			}
		}
		break;

	case Zmov:
		goto domov;
	}
	return;

domov:
	for(t=ymovtab; *t; t+=8)
		if(p->as == t[0])
		if(ycover[ft+t[1]])
		if(ycover[tt+t[2]])
			goto mfound;
bad:
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
		if((breg = byteswapreg(&p->to)) != D_AX) {
			*andptr++ = 0x87;			/* xchg lhs,bx */
			asmand(&p->from, reg[breg]);
			subreg(&pp, z, breg);
			doasm(&pp);
			*andptr++ = 0x87;			/* xchg lhs,bx */
			asmand(&p->from, reg[breg]);
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
		if((breg = byteswapreg(&p->from)) != D_AX) {
			*andptr++ = 0x87;			/* xchg rhs,bx */
			asmand(&p->to, reg[breg]);
			subreg(&pp, z, breg);
			doasm(&pp);
			*andptr++ = 0x87;			/* xchg rhs,bx */
			asmand(&p->to, reg[breg]);
		} else {
			*andptr++ = 0x90 + reg[z];		/* xchg rsh,ax */
			subreg(&pp, z, D_AX);
			doasm(&pp);
			*andptr++ = 0x90 + reg[z];		/* xchg rsh,ax */
		}
		return;
	}
	diag("doasm: notfound t2=%ux from=%ux to=%ux %P", t[2], p->from.type, p->to.type, p);
	return;

mfound:
	switch(t[3]) {
	default:
		diag("asmins: unknown mov %d %P", t[3], p);
		break;

	case 0:	/* lit */
		for(z=4; t[z]!=E; z++)
			*andptr++ = t[z];
		break;

	case 1:	/* r,m */
		*andptr++ = t[4];
		asmand(&p->to, t[5]);
		break;

	case 2:	/* m,r */
		*andptr++ = t[4];
		asmand(&p->from, t[5]);
		break;

	case 3:	/* r,m - 2op */
		*andptr++ = t[4];
		*andptr++ = t[5];
		asmand(&p->to, t[6]);
		break;

	case 4:	/* m,r - 2op */
		*andptr++ = t[4];
		*andptr++ = t[5];
		asmand(&p->from, t[6]);
		break;

	case 5:	/* load full pointer, trash heap */
		if(t[4])
			*andptr++ = t[4];
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
		asmand(&p->from, reg[p->to.type]);
		break;

	case 6:	/* double shift */
		z = p->from.type;
		switch(z) {
		default:
			goto bad;
		case D_CONST:
			*andptr++ = 0x0f;
			*andptr++ = t[4];
			asmand(&p->to, reg[p->from.index]);
			*andptr++ = p->from.offset;
			break;
		case D_CL:
		case D_CX:
			*andptr++ = 0x0f;
			*andptr++ = t[5];
			asmand(&p->to, reg[p->from.index]);
			break;
		}
		break;

	case 7: /* imul rm,r */
		if(t[4] == Pq) {
			*andptr++ = Pe;
			*andptr++ = Pm;
		} else
			*andptr++ = t[4];
		*andptr++ = t[5];
		asmand(&p->from, reg[p->to.type]);
		break;
	}
}

void
asmins(Prog *p)
{
	andptr = and;
	doasm(p);
	if(andptr > and+sizeof and) {
		print("and[] is too short - %ld byte instruction\n", andptr - and);
		errorexit();
	}
}
