// cmd/9l/noop.c, cmd/9l/pass.c, cmd/9l/span.c from Vita Nuova.
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

// +build ignore

#include	"l.h"

void
noops(void)
{
	Prog *p, *p1, *q, *q1;
	int o, mov, aoffset, curframe, curbecome, maxbecome;

	/*
	 * find leaf subroutines
	 * become sizes
	 * frame sizes
	 * strip NOPs
	 * expand RET
	 * expand BECOME pseudo
	 */

	if(debug['v'])
		Bprint(&bso, "%5.2f noops\n", cputime());
	Bflush(&bso);

	curframe = 0;
	curbecome = 0;
	maxbecome = 0;
	curtext = 0;
	q = P;
	for(p = firstp; p != P; p = p->link) {

		/* find out how much arg space is used in this TEXT */
		if(p->to.type == D_OREG && p->to.reg == REGSP)
			if(p->to.offset > curframe)
				curframe = p->to.offset;

		switch(p->as) {
		/* too hard, just leave alone */
		case ATEXT:
			if(curtext && curtext->from.sym) {
				curtext->from.sym->frame = curframe;
				curtext->from.sym->become = curbecome;
				if(curbecome > maxbecome)
					maxbecome = curbecome;
			}
			curframe = 0;
			curbecome = 0;

			q = p;
			p->mark |= LABEL|LEAF|SYNC;
			if(p->link)
				p->link->mark |= LABEL;
			curtext = p;
			break;

		case ANOR:
			q = p;
			if(p->to.type == D_REG)
				if(p->to.reg == REGZERO)
					p->mark |= LABEL|SYNC;
			break;

		case ALWAR:
		case ASTWCCC:
		case AECIWX:
		case AECOWX:
		case AEIEIO:
		case AICBI:
		case AISYNC:
		case ATLBIE:
		case ATLBIEL:
		case ASLBIA:
		case ASLBIE:
		case ASLBMFEE:
		case ASLBMFEV:
		case ASLBMTE:
		case ADCBF:
		case ADCBI:
		case ADCBST:
		case ADCBT:
		case ADCBTST:
		case ADCBZ:
		case ASYNC:
		case ATLBSYNC:
		case APTESYNC:
		case ATW:
		case AWORD:
		case ARFI:
		case ARFCI:
		case ARFID:
		case AHRFID:
			q = p;
			p->mark |= LABEL|SYNC;
			continue;

		case AMOVW:
		case AMOVWZ:
		case AMOVD:
			q = p;
			switch(p->from.type) {
			case D_MSR:
			case D_SPR:
			case D_FPSCR:
			case D_CREG:
			case D_DCR:
				p->mark |= LABEL|SYNC;
			}
			switch(p->to.type) {
			case D_MSR:
			case D_SPR:
			case D_FPSCR:
			case D_CREG:
			case D_DCR:
				p->mark |= LABEL|SYNC;
			}
			continue;

		case AFABS:
		case AFABSCC:
		case AFADD:
		case AFADDCC:
		case AFCTIW:
		case AFCTIWCC:
		case AFCTIWZ:
		case AFCTIWZCC:
		case AFDIV:
		case AFDIVCC:
		case AFMADD:
		case AFMADDCC:
		case AFMOVD:
		case AFMOVDU:
		/* case AFMOVDS: */
		case AFMOVS:
		case AFMOVSU:
		/* case AFMOVSD: */
		case AFMSUB:
		case AFMSUBCC:
		case AFMUL:
		case AFMULCC:
		case AFNABS:
		case AFNABSCC:
		case AFNEG:
		case AFNEGCC:
		case AFNMADD:
		case AFNMADDCC:
		case AFNMSUB:
		case AFNMSUBCC:
		case AFRSP:
		case AFRSPCC:
		case AFSUB:
		case AFSUBCC:
			q = p;
			p->mark |= FLOAT;
			continue;

		case ABL:
		case ABCL:
			if(curtext != P)
				curtext->mark &= ~LEAF;

		case ABC:
		case ABEQ:
		case ABGE:
		case ABGT:
		case ABLE:
		case ABLT:
		case ABNE:
		case ABR:
		case ABVC:
		case ABVS:

			p->mark |= BRANCH;
			q = p;
			q1 = p->cond;
			if(q1 != P) {
				while(q1->as == ANOP) {
					q1 = q1->link;
					p->cond = q1;
				}
				if(!(q1->mark & LEAF))
					q1->mark |= LABEL;
			} else
				p->mark |= LABEL;
			q1 = p->link;
			if(q1 != P)
				q1->mark |= LABEL;
			continue;

		case AFCMPO:
		case AFCMPU:
			q = p;
			p->mark |= FCMP|FLOAT;
			continue;

		case ARETURN:
			/* special form of RETURN is BECOME */
			if(p->from.type == D_CONST)
				if(p->from.offset > curbecome)
					curbecome = p->from.offset;

			q = p;
			if(p->link != P)
				p->link->mark |= LABEL;
			continue;

		case ANOP:
			q1 = p->link;
			q->link = q1;		/* q is non-nop */
			q1->mark |= p->mark;
			continue;

		default:
			q = p;
			continue;
		}
	}
	if(curtext && curtext->from.sym) {
		curtext->from.sym->frame = curframe;
		curtext->from.sym->become = curbecome;
		if(curbecome > maxbecome)
			maxbecome = curbecome;
	}

	if(debug['b'])
		print("max become = %d\n", maxbecome);
	xdefine("ALEFbecome", STEXT, maxbecome);

	curtext = 0;
	for(p = firstp; p != P; p = p->link) {
		switch(p->as) {
		case ATEXT:
			curtext = p;
			break;

		case ABL:	/* ABCL? */
			if(curtext != P && curtext->from.sym != S && curtext->to.offset >= 0) {
				o = maxbecome - curtext->from.sym->frame;
				if(o <= 0)
					break;
				/* calling a become or calling a variable */
				if(p->to.sym == S || p->to.sym->become) {
					curtext->to.offset += o;
					if(debug['b']) {
						curp = p;
						print("%D calling %D increase %d\n",
							&curtext->from, &p->to, o);
					}
				}
			}
			break;
		}
	}

	curtext = P;
	for(p = firstp; p != P; p = p->link) {
		o = p->as;
		switch(o) {
		case ATEXT:
			mov = AMOVD;
			aoffset = 0;
			curtext = p;
			autosize = p->to.offset + 8;
			if((p->mark & LEAF) && autosize <= 8)
				autosize = 0;
			else
				if(autosize & 4)
					autosize += 4;
			p->to.offset = autosize - 8;

			q = p;
			if(autosize) {
				/* use MOVDU to adjust R1 when saving R31, if autosize is small */
				if(!(curtext->mark & LEAF) && autosize >= -BIG && autosize <= BIG) {
					mov = AMOVDU;
					aoffset = -autosize;
				} else {
					q = prg();
					q->as = AADD;
					q->line = p->line;
					q->from.type = D_CONST;
					q->from.offset = -autosize;
					q->to.type = D_REG;
					q->to.reg = REGSP;

					q->link = p->link;
					p->link = q;
				}
			} else
			if(!(curtext->mark & LEAF)) {
				if(debug['v'])
					Bprint(&bso, "save suppressed in: %s\n",
						curtext->from.sym->name);
				curtext->mark |= LEAF;
			}

			if(curtext->mark & LEAF) {
				if(curtext->from.sym)
					curtext->from.sym->type = SLEAF;
				break;
			}

			q1 = prg();
			q1->as = mov;
			q1->line = p->line;
			q1->from.type = D_REG;
			q1->from.reg = REGTMP;
			q1->to.type = D_OREG;
			q1->to.offset = aoffset;
			q1->to.reg = REGSP;

			q1->link = q->link;
			q->link = q1;

			q1 = prg();
			q1->as = AMOVD;
			q1->line = p->line;
			q1->from.type = D_SPR;
			q1->from.offset = D_LR;
			q1->to.type = D_REG;
			q1->to.reg = REGTMP;

			q1->link = q->link;
			q->link = q1;
			break;

		case ARETURN:
			if(p->from.type == D_CONST)
				goto become;
			if(curtext->mark & LEAF) {
				if(!autosize) {
					p->as = ABR;
					p->from = zprg.from;
					p->to.type = D_SPR;
					p->to.offset = D_LR;
					p->mark |= BRANCH;
					break;
				}

				p->as = AADD;
				p->from.type = D_CONST;
				p->from.offset = autosize;
				p->to.type = D_REG;
				p->to.reg = REGSP;

				q = prg();
				q->as = ABR;
				q->line = p->line;
				q->to.type = D_SPR;
				q->to.offset = D_LR;
				q->mark |= BRANCH;

				q->link = p->link;
				p->link = q;
				break;
			}

			p->as = AMOVD;
			p->from.type = D_OREG;
			p->from.offset = 0;
			p->from.reg = REGSP;
			p->to.type = D_REG;
			p->to.reg = REGTMP;

			q = prg();
			q->as = AMOVD;
			q->line = p->line;
			q->from.type = D_REG;
			q->from.reg = REGTMP;
			q->to.type = D_SPR;
			q->to.offset = D_LR;

			q->link = p->link;
			p->link = q;
			p = q;

			if(autosize) {
				q = prg();
				q->as = AADD;
				q->line = p->line;
				q->from.type = D_CONST;
				q->from.offset = autosize;
				q->to.type = D_REG;
				q->to.reg = REGSP;

				q->link = p->link;
				p->link = q;
			}

			q1 = prg();
			q1->as = ABR;
			q1->line = p->line;
			q1->to.type = D_SPR;
			q1->to.offset = D_LR;
			q1->mark |= BRANCH;

			q1->link = q->link;
			q->link = q1;
			break;

		become:
			if(curtext->mark & LEAF) {

				q = prg();
				q->line = p->line;
				q->as = ABR;
				q->from = zprg.from;
				q->to = p->to;
				q->cond = p->cond;
				q->link = p->link;
				q->mark |= BRANCH;
				p->link = q;

				p->as = AADD;
				p->from = zprg.from;
				p->from.type = D_CONST;
				p->from.offset = autosize;
				p->to = zprg.to;
				p->to.type = D_REG;
				p->to.reg = REGSP;

				break;
			}
			q = prg();
			q->line = p->line;
			q->as = ABR;
			q->from = zprg.from;
			q->to = p->to;
			q->cond = p->cond;
			q->mark |= BRANCH;
			q->link = p->link;
			p->link = q;

			q = prg();
			q->line = p->line;
			q->as = AADD;
			q->from.type = D_CONST;
			q->from.offset = autosize;
			q->to.type = D_REG;
			q->to.reg = REGSP;
			q->link = p->link;
			p->link = q;

			q = prg();
			q->line = p->line;
			q->as = AMOVD;
			q->line = p->line;
			q->from.type = D_REG;
			q->from.reg = REGTMP;
			q->to.type = D_SPR;
			q->to.offset = D_LR;
			q->link = p->link;
			p->link = q;

			p->as = AMOVD;
			p->from = zprg.from;
			p->from.type = D_OREG;
			p->from.offset = 0;
			p->from.reg = REGSP;
			p->to = zprg.to;
			p->to.type = D_REG;
			p->to.reg = REGTMP;

			break;
		}
	}

	if(debug['Q'] == 0)
		return;

	curtext = P;
	q = P;		/* p - 1 */
	q1 = firstp;	/* top of block */
	o = 0;		/* count of instructions */
	for(p = firstp; p != P; p = p1) {
		p1 = p->link;
		o++;
		if(p->mark & NOSCHED){
			if(q1 != p){
				sched(q1, q);
			}
			for(; p != P; p = p->link){
				if(!(p->mark & NOSCHED))
					break;
				q = p;
			}
			p1 = p;
			q1 = p;
			o = 0;
			continue;
		}
		if(p->mark & (LABEL|SYNC)) {
			if(q1 != p)
				sched(q1, q);
			q1 = p;
			o = 1;
		}
		if(p->mark & (BRANCH|SYNC)) {
			sched(q1, p);
			q1 = p1;
			o = 0;
		}
		if(o >= NSCHED) {
			sched(q1, p);
			q1 = p1;
			o = 0;
		}
		q = p;
	}
}

void
addnop(Prog *p)
{
	Prog *q;

	q = prg();
	q->as = AOR;
	q->line = p->line;
	q->from.type = D_REG;
	q->from.reg = REGZERO;
	q->to.type = D_REG;
	q->to.reg = REGZERO;

	q->link = p->link;
	p->link = q;
}

#include	"l.h"

void
dodata(void)
{
	int i, t;
	Sym *s;
	Prog *p, *p1;
	vlong orig, orig1, v;

	if(debug['v'])
		Bprint(&bso, "%5.2f dodata\n", cputime());
	Bflush(&bso);
	for(p = datap; p != P; p = p->link) {
		s = p->from.sym;
		if(p->as == ADYNT || p->as == AINIT)
			s->value = dtype;
		if(s->type == SBSS)
			s->type = SDATA;
		if(s->type != SDATA)
			diag("initialize non-data (%d): %s\n%P",
				s->type, s->name, p);
		v = p->from.offset + p->reg;
		if(v > s->value)
			diag("initialize bounds (%lld): %s\n%P",
				s->value, s->name, p);
	}

	/*
	 * pass 1
	 *	assign 'small' variables to data segment
	 *	(rational is that data segment is more easily
	 *	 addressed through offset on REGSB)
	 */
	orig = 0;
	for(i=0; i<NHASH; i++)
	for(s = hash[i]; s != S; s = s->link) {
		t = s->type;
		if(t != SDATA && t != SBSS)
			continue;
		v = s->value;
		if(v == 0) {
			diag("%s: no size", s->name);
			v = 1;
		}
		v = rnd(v, 4);
		s->value = v;
		if(v > MINSIZ)
			continue;
		if(v >= 8)
			orig = rnd(orig, 8);
		s->value = orig;
		orig += v;
		s->type = SDATA1;
	}
	orig1 = orig;

	/*
	 * pass 2
	 *	assign 'data' variables to data segment
	 */
	for(i=0; i<NHASH; i++)
	for(s = hash[i]; s != S; s = s->link) {
		t = s->type;
		if(t != SDATA) {
			if(t == SDATA1)
				s->type = SDATA;
			continue;
		}
		v = s->value;
		if(v >= 8)
			orig = rnd(orig, 8);
		s->value = orig;
		orig += v;
		s->type = SDATA1;
	}

	if(orig)
		orig = rnd(orig, 8);
	datsize = orig;

	/*
	 * pass 3
	 *	everything else to bss segment
	 */
	for(i=0; i<NHASH; i++)
	for(s = hash[i]; s != S; s = s->link) {
		if(s->type != SBSS)
			continue;
		v = s->value;
		if(v >= 8)
			orig = rnd(orig, 8);
		s->value = orig;
		orig += v;
	}
	if(orig)
		orig = rnd(orig, 8);
	bsssize = orig-datsize;

	/*
	 * pass 4
	 *	add literals to all large values.
	 *	at this time:
	 *		small data is allocated DATA
	 *		large data is allocated DATA1
	 *		large bss is allocated BSS
	 *	the new literals are loaded between
	 *	small data and large data.
	 */
	orig = 0;
	for(p = firstp; p != P; p = p->link) {
		if(p->as != AMOVW)
			continue;
		if(p->from.type != D_CONST)
			continue;
		if(s = p->from.sym) {
			t = s->type;
			if(t != SDATA && t != SDATA1 && t != SBSS)
				continue;
			t = p->from.name;
			if(t != D_EXTERN && t != D_STATIC)
				continue;
			v = s->value + p->from.offset;
			if(v >= 0 && v <= 0xffff)
				continue;
			if(!strcmp(s->name, "setSB"))
				continue;
			/* size should be 19 max */
			if(strlen(s->name) >= 10)	/* has loader address */ 
				sprint(literal, "$%p.%llux", s, p->from.offset);
			else
				sprint(literal, "$%s.%d.%llux", s->name, s->version, p->from.offset);
		} else {
			if(p->from.name != D_NONE)
				continue;
			if(p->from.reg != NREG)
				continue;
			v = p->from.offset;
			if(v >= -0x7fff-1 && v <= 0x7fff)
				continue;
			if(!(v & 0xffff))
				continue;
			if(v)
				continue;	/* quicker to build it than load it */
			/* size should be 9 max */
			sprint(literal, "$%llux", v);
		}
		s = lookup(literal, 0);
		if(s->type == 0) {
			s->type = SDATA;
			s->value = orig1+orig;
			orig += 4;
			p1 = prg();
			p1->as = ADATA;
			p1->line = p->line;
			p1->from.type = D_OREG;
			p1->from.sym = s;
			p1->from.name = D_EXTERN;
			p1->reg = 4;
			p1->to = p->from;
			p1->link = datap;
			datap = p1;
		}
		if(s->type != SDATA)
			diag("literal not data: %s", s->name);
		p->from.type = D_OREG;
		p->from.sym = s;
		p->from.name = D_EXTERN;
		p->from.offset = 0;
		continue;
	}
	while(orig & 7)
		orig++;
	/*
	 * pass 5
	 *	re-adjust offsets
	 */
	for(i=0; i<NHASH; i++)
	for(s = hash[i]; s != S; s = s->link) {
		t = s->type;
		if(t == SBSS) {
			s->value += orig;
			continue;
		}
		if(t == SDATA1) {
			s->type = SDATA;
			s->value += orig;
			continue;
		}
	}
	datsize += orig;
	xdefine("setSB", SDATA, 0+BIG);
	xdefine("bdata", SDATA, 0);
	xdefine("edata", SDATA, datsize);
	xdefine("end", SBSS, datsize+bsssize);
	xdefine("etext", STEXT, 0);
}

void
undef(void)
{
	int i;
	Sym *s;

	for(i=0; i<NHASH; i++)
	for(s = hash[i]; s != S; s = s->link)
		if(s->type == SXREF)
			diag("%s: not defined", s->name);
}

int
relinv(int a)
{

	switch(a) {
	case ABEQ:	return ABNE;
	case ABNE:	return ABEQ;

	case ABGE:	return ABLT;
	case ABLT:	return ABGE;

	case ABGT:	return ABLE;
	case ABLE:	return ABGT;

	case ABVC:	return ABVS;
	case ABVS:	return ABVC;
	}
	return 0;
}

void
follow(void)
{

	if(debug['v'])
		Bprint(&bso, "%5.2f follow\n", cputime());
	Bflush(&bso);

	firstp = prg();
	lastp = firstp;

	xfol(textp);

	firstp = firstp->link;
	lastp->link = P;
}

void
xfol(Prog *p)
{
	Prog *q, *r;
	int a, b, i;

loop:
	if(p == P)
		return;
	a = p->as;
	if(a == ATEXT)
		curtext = p;
	if(a == ABR) {
		q = p->cond;
		if((p->mark&NOSCHED) || q && (q->mark&NOSCHED)){
			p->mark |= FOLL;
			lastp->link = p;
			lastp = p;
			p = p->link;
			xfol(p);
			p = q;
			if(p && !(p->mark & FOLL))
				goto loop;
			return;
		}
		if(q != P) {
			p->mark |= FOLL;
			p = q;
			if(!(p->mark & FOLL))
				goto loop;
		}
	}
	if(p->mark & FOLL) {
		for(i=0,q=p; i<4; i++,q=q->link) {
			if(q == lastp || (q->mark&NOSCHED))
				break;
			b = 0;		/* set */
			a = q->as;
			if(a == ANOP) {
				i--;
				continue;
			}
			if(a == ABR || a == ARETURN || a == ARFI || a == ARFCI || a == ARFID || a == AHRFID)
				goto copy;
			if(!q->cond || (q->cond->mark&FOLL))
				continue;
			b = relinv(a);
			if(!b)
				continue;
		copy:
			for(;;) {
				r = prg();
				*r = *p;
				if(!(r->mark&FOLL))
					print("cant happen 1\n");
				r->mark |= FOLL;
				if(p != q) {
					p = p->link;
					lastp->link = r;
					lastp = r;
					continue;
				}
				lastp->link = r;
				lastp = r;
				if(a == ABR || a == ARETURN || a == ARFI || a == ARFCI || a == ARFID || a == AHRFID)
					return;
				r->as = b;
				r->cond = p->link;
				r->link = p->cond;
				if(!(r->link->mark&FOLL))
					xfol(r->link);
				if(!(r->cond->mark&FOLL))
					print("cant happen 2\n");
				return;
			}
		}

		a = ABR;
		q = prg();
		q->as = a;
		q->line = p->line;
		q->to.type = D_BRANCH;
		q->to.offset = p->pc;
		q->cond = p;
		p = q;
	}
	p->mark |= FOLL;
	lastp->link = p;
	lastp = p;
	if(a == ABR || a == ARETURN || a == ARFI || a == ARFCI || a == ARFID || a == AHRFID){
		if(p->mark & NOSCHED){
			p = p->link;
			goto loop;
		}
		return;
	}
	if(p->cond != P)
	if(a != ABL && p->link != P) {
		xfol(p->link);
		p = p->cond;
		if(p == P || (p->mark&FOLL))
			return;
		goto loop;
	}
	p = p->link;
	goto loop;
}

void
patch(void)
{
	long c;
	Prog *p, *q;
	Sym *s;
	int a;
	vlong vexit;

	if(debug['v'])
		Bprint(&bso, "%5.2f patch\n", cputime());
	Bflush(&bso);
	mkfwd();
	s = lookup("exit", 0);
	vexit = s->value;
	for(p = firstp; p != P; p = p->link) {
		a = p->as;
		if(a == ATEXT)
			curtext = p;
		if((a == ABL || a == ARETURN) && p->to.sym != S) {
			s = p->to.sym;
			if(s->type != STEXT && s->type != SUNDEF) {
				diag("undefined: %s\n%P", s->name, p);
				s->type = STEXT;
				s->value = vexit;
			}
			if(s->type == SUNDEF){
				p->to.offset = 0;
				p->cond = UP;
			}
			else
				p->to.offset = s->value;
			p->to.type = D_BRANCH;
		}
		if(p->to.type != D_BRANCH || p->cond == UP)
			continue;
		c = p->to.offset;
		for(q = firstp; q != P;) {
			if(q->forwd != P)
			if(c >= q->forwd->pc) {
				q = q->forwd;
				continue;
			}
			if(c == q->pc)
				break;
			q = q->link;
		}
		if(q == P) {
			diag("branch out of range %ld\n%P", c, p);
			p->to.type = D_NONE;
		}
		p->cond = q;
	}

	for(p = firstp; p != P; p = p->link) {
		if(p->as == ATEXT)
			curtext = p;
		p->mark = 0;	/* initialization for follow */
		if(p->cond != P && p->cond != UP) {
			p->cond = brloop(p->cond);
			if(p->cond != P)
			if(p->to.type == D_BRANCH)
				p->to.offset = p->cond->pc;
		}
	}
}

#define	LOG	5
void
mkfwd(void)
{
	Prog *p;
	long dwn[LOG], cnt[LOG], i;
	Prog *lst[LOG];

	for(i=0; i<LOG; i++) {
		if(i == 0)
			cnt[i] = 1; else
			cnt[i] = LOG * cnt[i-1];
		dwn[i] = 1;
		lst[i] = P;
	}
	i = 0;
	for(p = firstp; p != P; p = p->link) {
		if(p->as == ATEXT)
			curtext = p;
		i--;
		if(i < 0)
			i = LOG-1;
		p->forwd = P;
		dwn[i]--;
		if(dwn[i] <= 0) {
			dwn[i] = cnt[i];
			if(lst[i] != P)
				lst[i]->forwd = p;
			lst[i] = p;
		}
	}
}

Prog*
brloop(Prog *p)
{
	Prog *q;
	int c;

	for(c=0; p!=P;) {
		if(p->as != ABR || (p->mark&NOSCHED))
			return p;
		q = p->cond;
		if(q <= p) {
			c++;
			if(q == p || c > 5000)
				break;
		}
		p = q;
	}
	return P;
}

vlong
atolwhex(char *s)
{
	vlong n;
	int f;

	n = 0;
	f = 0;
	while(*s == ' ' || *s == '\t')
		s++;
	if(*s == '-' || *s == '+') {
		if(*s++ == '-')
			f = 1;
		while(*s == ' ' || *s == '\t')
			s++;
	}
	if(s[0]=='0' && s[1]){
		if(s[1]=='x' || s[1]=='X'){
			s += 2;
			for(;;){
				if(*s >= '0' && *s <= '9')
					n = n*16 + *s++ - '0';
				else if(*s >= 'a' && *s <= 'f')
					n = n*16 + *s++ - 'a' + 10;
				else if(*s >= 'A' && *s <= 'F')
					n = n*16 + *s++ - 'A' + 10;
				else
					break;
			}
		} else
			while(*s >= '0' && *s <= '7')
				n = n*8 + *s++ - '0';
	} else
		while(*s >= '0' && *s <= '9')
			n = n*10 + *s++ - '0';
	if(f)
		n = -n;
	return n;
}

vlong
rnd(vlong v, long r)
{
	vlong c;

	if(r <= 0)
		return v;
	v += r - 1;
	c = v % r;
	if(c < 0)
		c += r;
	v -= c;
	return v;
}

void
import(void)
{
	int i;
	Sym *s;

	for(i = 0; i < NHASH; i++)
		for(s = hash[i]; s != S; s = s->link)
			if(s->sig != 0 && s->type == SXREF && (nimports == 0 || s->subtype == SIMPORT)){
				undefsym(s);
				Bprint(&bso, "IMPORT: %s sig=%lux v=%lld\n", s->name, s->sig, s->value);
				if(debug['S'])
					s->sig = 0;
			}
}

void
ckoff(Sym *s, vlong v)
{
	if(v < 0 || v >= 1<<Roffset)
		diag("relocation offset %lld for %s out of range", v, s->name);
}

static Prog*
newdata(Sym *s, int o, int w, int t)
{
	Prog *p;

	p = prg();
	p->link = datap;
	datap = p;
	p->as = ADATA;
	p->reg = w;
	p->from.type = D_OREG;
	p->from.name = t;
	p->from.sym = s;
	p->from.offset = o;
	p->to.type = D_CONST;
	p->to.name = D_NONE;
	return p;
}

void
export(void)
{
	int i, j, n, off, nb, sv, ne;
	Sym *s, *et, *str, **esyms;
	Prog *p;
	char buf[NSNAME], *t;

	n = 0;
	for(i = 0; i < NHASH; i++)
		for(s = hash[i]; s != S; s = s->link)
			if(s->type != SXREF && s->type != SUNDEF && (nexports == 0 && s->sig != 0 || s->subtype == SEXPORT || allexport))
				n++;
	esyms = malloc(n*sizeof(Sym*));
	ne = n;
	n = 0;
	for(i = 0; i < NHASH; i++)
		for(s = hash[i]; s != S; s = s->link)
			if(s->type != SXREF && s->type != SUNDEF && (nexports == 0 && s->sig != 0 || s->subtype == SEXPORT || allexport))
				esyms[n++] = s;
	for(i = 0; i < ne-1; i++)
		for(j = i+1; j < ne; j++)
			if(strcmp(esyms[i]->name, esyms[j]->name) > 0){
				s = esyms[i];
				esyms[i] = esyms[j];
				esyms[j] = s;
			}

	nb = 0;
	off = 0;
	et = lookup(EXPTAB, 0);
	if(et->type != 0 && et->type != SXREF)
		diag("%s already defined", EXPTAB);
	et->type = SDATA;
	str = lookup(".string", 0);
	if(str->type == 0)
		str->type = SDATA;
	sv = str->value;
	for(i = 0; i < ne; i++){
		s = esyms[i];
		Bprint(&bso, "EXPORT: %s sig=%lux t=%d\n", s->name, s->sig, s->type);

		/* signature */
		p = newdata(et, off, sizeof(long), D_EXTERN);
		off += sizeof(long);
		p->to.offset = s->sig;

		/* address */
		p = newdata(et, off, sizeof(long), D_EXTERN);
		off += sizeof(long);		/* TO DO: bug */
		p->to.name = D_EXTERN;
		p->to.sym = s;

		/* string */
		t = s->name;
		n = strlen(t)+1;
		for(;;){
			buf[nb++] = *t;
			sv++;
			if(nb >= NSNAME){
				p = newdata(str, sv-NSNAME, NSNAME, D_STATIC);
				p->to.type = D_SCONST;
				memmove(p->to.sval, buf, NSNAME);
				nb = 0;
			}
			if(*t++ == 0)
				break;
		}

		/* name */
		p = newdata(et, off, sizeof(long), D_EXTERN);
		off += sizeof(long);
		p->to.name = D_STATIC;
		p->to.sym = str;
		p->to.offset = sv-n;
	}

	if(nb > 0){
		p = newdata(str, sv-nb, nb, D_STATIC);
		p->to.type = D_SCONST;
		memmove(p->to.sval, buf, nb);
	}

	for(i = 0; i < 3; i++){
		newdata(et, off, sizeof(long), D_EXTERN);
		off += sizeof(long);
	}
	et->value = off;
	if(sv == 0)
		sv = 1;
	str->value = sv;
	exports = ne;
	free(esyms);
}

#include	"l.h"

void
span(void)
{
	Prog *p, *q;
	Sym *setext;
	Optab *o;
	int m, bflag;
	vlong c, otxt;

	if(debug['v'])
		Bprint(&bso, "%5.2f span\n", cputime());
	Bflush(&bso);

	bflag = 0;
	c = INITTEXT;
	otxt = c;
	for(p = firstp; p != P; p = p->link) {
		p->pc = c;
		o = oplook(p);
		m = o->size;
		if(m == 0) {
			if(p->as == ATEXT) {
				curtext = p;
				autosize = p->to.offset + 8;
				if(p->from3.type == D_CONST) {
					if(p->from3.offset & 3)
						diag("illegal origin\n%P", p);
					if(c > p->from3.offset)
						diag("passed origin (#%llux)\n%P", c, p);
					else
						c = p->from3.offset;
					p->pc = c;
				}
				if(p->from.sym != S)
					p->from.sym->value = c;
				/* need passes to resolve branches? */
				if(c-otxt >= (1L<<15))
					bflag = c;
				otxt = c;
				continue;
			}
			if(p->as != ANOP)
				diag("zero-width instruction\n%P", p);
			continue;
		}
		c += m;
	}

	/*
	 * if any procedure is large enough to
	 * generate a large SBRA branch, then
	 * generate extra passes putting branches
	 * around jmps to fix. this is rare.
	 */
	while(bflag) {
		if(debug['v'])
			Bprint(&bso, "%5.2f span1\n", cputime());
		bflag = 0;
		c = INITTEXT;
		for(p = firstp; p != P; p = p->link) {
			p->pc = c;
			o = oplook(p);
			if((o->type == 16 || o->type == 17) && p->cond) {
				otxt = p->cond->pc - c;
				if(otxt < -(1L<<16)+10 || otxt >= (1L<<15)-10) {
					q = prg();
					q->link = p->link;
					p->link = q;
					q->as = ABR;
					q->to.type = D_BRANCH;
					q->cond = p->cond;
					p->cond = q;
					q = prg();
					q->link = p->link;
					p->link = q;
					q->as = ABR;
					q->to.type = D_BRANCH;
					q->cond = q->link->link;
					addnop(p->link);
					addnop(p);
					bflag = 1;
				}
			}
			m = o->size;
			if(m == 0) {
				if(p->as == ATEXT) {
					curtext = p;
					autosize = p->to.offset + 8;
					if(p->from.sym != S)
						p->from.sym->value = c;
					continue;
				}
				if(p->as != ANOP)
					diag("zero-width instruction\n%P", p);
				continue;
			}
			c += m;
		}
	}

	c = rnd(c, 8);

	setext = lookup("etext", 0);
	if(setext != S) {
		setext->value = c;
		textsize = c - INITTEXT;
	}
	if(INITRND)
		INITDAT = rnd(c, INITRND);
	if(debug['v'])
		Bprint(&bso, "tsize = %llux\n", textsize);
	Bflush(&bso);
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
}

vlong
vregoff(Adr *a)
{

	instoffset = 0;
	aclass(a);
	return instoffset;
}

long
regoff(Adr *a)
{
	return vregoff(a);
}

int
isint32(vlong v)
{
	long l;

	l = v;
	return (vlong)l == v;
}

int
isuint32(uvlong v)
{
	ulong l;

	l = v;
	return (uvlong)l == v;
}

int
aclass(Adr *a)
{
	Sym *s;
	int t;

	switch(a->type) {
	case D_NONE:
		return C_NONE;

	case D_REG:
		return C_REG;

	case D_FREG:
		return C_FREG;

	case D_CREG:
		return C_CREG;

	case D_SPR:
		if(a->offset == D_LR)
			return C_LR;
		if(a->offset == D_XER)
			return C_XER;
		if(a->offset == D_CTR)
			return C_CTR;
		return C_SPR;

	case D_DCR:
		return C_SPR;

	case D_FPSCR:
		return C_FPSCR;

	case D_MSR:
		return C_MSR;

	case D_OREG:
		switch(a->name) {
		case D_EXTERN:
		case D_STATIC:
			if(a->sym == S)
				break;
			t = a->sym->type;
			if(t == 0 || t == SXREF) {
				diag("undefined external: %s in %s",
					a->sym->name, TNAME);
				a->sym->type = SDATA;
			}
			if(dlm){
				instoffset = a->sym->value + a->offset;
				switch(a->sym->type){
				case STEXT:
				case SLEAF:
				case SCONST:
				case SUNDEF:
					break;
				default:
					instoffset += INITDAT;
				}
				return C_ADDR;
			}
			instoffset = a->sym->value + a->offset - BIG;
			if(instoffset >= -BIG && instoffset < BIG)
				return C_SEXT;
			return C_LEXT;
		case D_AUTO:
			instoffset = autosize + a->offset;
			if(instoffset >= -BIG && instoffset < BIG)
				return C_SAUTO;
			return C_LAUTO;
		case D_PARAM:
			instoffset = autosize + a->offset + 8L;
			if(instoffset >= -BIG && instoffset < BIG)
				return C_SAUTO;
			return C_LAUTO;
		case D_NONE:
			instoffset = a->offset;
			if(instoffset == 0)
				return C_ZOREG;
			if(instoffset >= -BIG && instoffset < BIG)
				return C_SOREG;
			return C_LOREG;
		}
		return C_GOK;

	case D_OPT:
		instoffset = a->offset & 31L;
		if(a->name == D_NONE)
			return C_SCON;
		return C_GOK;

	case D_CONST:
		switch(a->name) {

		case D_NONE:
			instoffset = a->offset;
		consize:
			if(instoffset >= 0) {
				if(instoffset == 0)
					return C_ZCON;
				if(instoffset <= 0x7fff)
					return C_SCON;
				if(instoffset <= 0xffff)
					return C_ANDCON;
				if((instoffset & 0xffff) == 0 && isuint32(instoffset))	/* && (instoffset & (1<<31)) == 0) */
					return C_UCON;
				if(isint32(instoffset) || isuint32(instoffset))
					return C_LCON;
				return C_DCON;
			}
			if(instoffset >= -0x8000)
				return C_ADDCON;
			if((instoffset & 0xffff) == 0 && isint32(instoffset))
				return C_UCON;
			if(isint32(instoffset))
				return C_LCON;
			return C_DCON;

		case D_EXTERN:
		case D_STATIC:
			s = a->sym;
			if(s == S)
				break;
			t = s->type;
			if(t == 0 || t == SXREF) {
				diag("undefined external: %s in %s",
					s->name, TNAME);
				s->type = SDATA;
			}
			if(s->type == STEXT || s->type == SLEAF || s->type == SUNDEF) {
				instoffset = s->value + a->offset;
				return C_LCON;
			}
			if(s->type == SCONST) {
				instoffset = s->value + a->offset;
				if(dlm)
					return C_LCON;
				goto consize;
			}
			if(!dlm){
				instoffset = s->value + a->offset - BIG;
				if(instoffset >= -BIG && instoffset < BIG && instoffset != 0)
					return C_SECON;
			}
			instoffset = s->value + a->offset + INITDAT;
			if(dlm)
				return C_LCON;
			/* not sure why this barfs */
			return C_LCON;
		/*
			if(instoffset == 0)
				return C_ZCON;
			if(instoffset >= -0x8000 && instoffset <= 0xffff)
				return C_SCON;
			if((instoffset & 0xffff) == 0)
				return C_UCON;
			return C_LCON;
		*/

		case D_AUTO:
			instoffset = autosize + a->offset;
			if(instoffset >= -BIG && instoffset < BIG)
				return C_SACON;
			return C_LACON;

		case D_PARAM:
			instoffset = autosize + a->offset + 8L;
			if(instoffset >= -BIG && instoffset < BIG)
				return C_SACON;
			return C_LACON;
		}
		return C_GOK;

	case D_BRANCH:
		return C_SBRA;
	}
	return C_GOK;
}

Optab*
oplook(Prog *p)
{
	int a1, a2, a3, a4, r;
	char *c1, *c3, *c4;
	Optab *o, *e;

	a1 = p->optab;
	if(a1)
		return optab+(a1-1);
	a1 = p->from.class;
	if(a1 == 0) {
		a1 = aclass(&p->from) + 1;
		p->from.class = a1;
	}
	a1--;
	a3 = p->from3.class;
	if(a3 == 0) {
		a3 = aclass(&p->from3) + 1;
		p->from3.class = a3;
	}
	a3--;
	a4 = p->to.class;
	if(a4 == 0) {
		a4 = aclass(&p->to) + 1;
		p->to.class = a4;
	}
	a4--;
	a2 = C_NONE;
	if(p->reg != NREG)
		a2 = C_REG;
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
	diag("illegal combination %A %R %R %R %R",
		p->as, a1, a2, a3, a4);
	if(1||!debug['a'])
		prasm(p);
	if(o == 0)
		errorexit();
	return o;
}

int
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

int
ocmp(void *a1, void *a2)
{
	Optab *p1, *p2;
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

void
buildop(void)
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
			diag("unknown op in build: %A", r);
			errorexit();
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
			break;
		case AREM:	/* macro */
			oprange[AREMCC] = oprange[r];
			oprange[AREMV] = oprange[r];
			oprange[AREMVCC] = oprange[r];
			oprange[AREMU] = oprange[r];
			oprange[AREMUCC] = oprange[r];
			oprange[AREMUV] = oprange[r];
			oprange[AREMUVCC] = oprange[r];
			break;
		case AREMD:
			oprange[AREMDCC] = oprange[r];
			oprange[AREMDV] = oprange[r];
			oprange[AREMDVCC] = oprange[r];
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
			break;
		}
	}
}

enum{
	ABSD = 0,
	ABSU = 1,
	RELD = 2,
	RELU = 3,
};

int modemap[8] = { 0, 1, -1, 2, 3, 4, 5, 6};

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
dynreloc(Sym *s, long v, int abs, int split, int sext)
{
	int i, k, n;
	uchar *m;
	ulong *a;
	Reloc *r;

	if(v&3)
		diag("bad relocation address");
	v >>= 2;
	if(s->type == SUNDEF)
		k = abs ? ABSU : RELU;
	else
		k = abs ? ABSD : RELD;
	if(split)
		k += 4;
	if(sext)
		k += 2;
	/* Bprint(&bso, "R %s a=%ld(%lx) %d\n", s->name, a, a, k); */
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
