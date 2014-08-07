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
