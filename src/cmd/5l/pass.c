// Inferno utils/5l/pass.c
// http://code.google.com/p/inferno-os/source/browse/utils/5l/pass.c
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

// Code and data passes.

#include	"l.h"
#include	"../ld/lib.h"

static void xfol(Prog*, Prog**);

Prog*
brchain(Prog *p)
{
	int i;

	for(i=0; i<20; i++) {
		if(p == P || p->as != AB)
			return p;
		p = p->cond;
	}
	return P;
}

int
relinv(int a)
{
	switch(a) {
	case ABEQ:	return ABNE;
	case ABNE:	return ABEQ;
	case ABCS:	return ABCC;
	case ABHS:	return ABLO;
	case ABCC:	return ABCS;
	case ABLO:	return ABHS;
	case ABMI:	return ABPL;
	case ABPL:	return ABMI;
	case ABVS:	return ABVC;
	case ABVC:	return ABVS;
	case ABHI:	return ABLS;
	case ABLS:	return ABHI;
	case ABGE:	return ABLT;
	case ABLT:	return ABGE;
	case ABGT:	return ABLE;
	case ABLE:	return ABGT;
	}
	diag("unknown relation: %s", anames[a]);
	return a;
}

void
follow(void)
{
	Prog *firstp, *lastp;

	if(debug['v'])
		Bprint(&bso, "%5.2f follow\n", cputime());
	Bflush(&bso);

	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		firstp = prg();
		lastp = firstp;
		xfol(cursym->text, &lastp);
		lastp->link = nil;
		cursym->text = firstp->link;
	}
}

static void
xfol(Prog *p, Prog **last)
{
	Prog *q, *r;
	int a, i;

loop:
	if(p == P)
		return;
	a = p->as;
	if(a == AB) {
		q = p->cond;
		if(q != P && q->as != ATEXT) {
			p->mark |= FOLL;
			p = q;
			if(!(p->mark & FOLL))
				goto loop;
		}
	}
	if(p->mark & FOLL) {
		for(i=0,q=p; i<4; i++,q=q->link) {
			if(q == *last || q == nil)
				break;
			a = q->as;
			if(a == ANOP) {
				i--;
				continue;
			}
			if(a == AB || (a == ARET && q->scond == 14) || a == ARFE || a == AUNDEF)
				goto copy;
			if(q->cond == P || (q->cond->mark&FOLL))
				continue;
			if(a != ABEQ && a != ABNE)
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
					(*last)->link = r;
					*last = r;
					continue;
				}
				(*last)->link = r;
				*last = r;
				if(a == AB || (a == ARET && q->scond == 14) || a == ARFE || a == AUNDEF)
					return;
				r->as = ABNE;
				if(a == ABNE)
					r->as = ABEQ;
				r->cond = p->link;
				r->link = p->cond;
				if(!(r->link->mark&FOLL))
					xfol(r->link, last);
				if(!(r->cond->mark&FOLL))
					print("cant happen 2\n");
				return;
			}
		}
		a = AB;
		q = prg();
		q->as = a;
		q->line = p->line;
		q->to.type = D_BRANCH;
		q->to.offset = p->pc;
		q->cond = p;
		p = q;
	}
	p->mark |= FOLL;
	(*last)->link = p;
	*last = p;
	if(a == AB || (a == ARET && p->scond == 14) || a == ARFE || a == AUNDEF){
		return;
	}
	if(p->cond != P)
	if(a != ABL && a != ABX && p->link != P) {
		q = brchain(p->link);
		if(a != ATEXT && a != ABCASE)
		if(q != P && (q->mark&FOLL)) {
			p->as = relinv(a);
			p->link = p->cond;
			p->cond = q;
		}
		xfol(p->link, last);
		q = brchain(p->cond);
		if(q == P)
			q = p->cond;
		if(q->mark&FOLL) {
			p->cond = q;
			return;
		}
		p = q;
		goto loop;
	}
	p = p->link;
	goto loop;
}

void
patch(void)
{
	int32 c, vexit;
	Prog *p, *q;
	Sym *s;
	int a;

	if(debug['v'])
		Bprint(&bso, "%5.2f patch\n", cputime());
	Bflush(&bso);
	mkfwd();
	s = lookup("exit", 0);
	vexit = s->value;
	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		for(p = cursym->text; p != P; p = p->link) {
			a = p->as;
			if((a == ABL || a == ABX || a == AB || a == ARET) &&
			   p->to.type != D_BRANCH && p->to.sym != S) {
				s = p->to.sym;
				if(s->text == nil)
					continue;
				switch(s->type&SMASK) {
				default:
					diag("undefined: %s", s->name);
					s->type = STEXT;
					s->value = vexit;
					continue;	// avoid more error messages
				case STEXT:
					p->to.offset = s->value;
					p->to.type = D_BRANCH;
					p->cond = s->text;
					continue;
				}
			}
			if(p->to.type != D_BRANCH)
				continue;
			c = p->to.offset;
			for(q = cursym->text; q != P;) {
				if(c == q->pc)
					break;
				if(q->forwd != P && c >= q->forwd->pc)
					q = q->forwd;
				else
					q = q->link;
			}
			if(q == P) {
				diag("branch out of range %d\n%P", c, p);
				p->to.type = D_NONE;
			}
			p->cond = q;
		}
	}

	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		for(p = cursym->text; p != P; p = p->link) {
			if(p->cond != P) {
				p->cond = brloop(p->cond);
				if(p->cond != P)
				if(p->to.type == D_BRANCH)
					p->to.offset = p->cond->pc;
			}
		}
	}
}

Prog*
brloop(Prog *p)
{
	Prog *q;
	int c;

	for(c=0; p!=P;) {
		if(p->as != AB)
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

int32
atolwhex(char *s)
{
	int32 n;
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

int32
rnd(int32 v, int32 r)
{
	int32 c;

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
dozerostk(void)
{
	Prog *p, *pl;
	int32 autoffset;

	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		if(cursym->text == nil || cursym->text->link == nil)
			continue;				
		p = cursym->text;
		autoffset = p->to.offset;
		if(autoffset < 0)
			autoffset = 0;
		if(autoffset && !(p->reg&NOSPLIT)) {
			// MOVW $4(R13), R1
			p = appendp(p);
			p->as = AMOVW;
			p->from.type = D_CONST;
			p->from.reg = 13;
			p->from.offset = 4;
			p->to.type = D_REG;
			p->to.reg = 1;

			// MOVW $n(R13), R2
			p = appendp(p);
			p->as = AMOVW;
			p->from.type = D_CONST;
			p->from.reg = 13;
			p->from.offset = 4 + autoffset;
			p->to.type = D_REG;
			p->to.reg = 2;

			// MOVW $0, R3
			p = appendp(p);
			p->as = AMOVW;
			p->from.type = D_CONST;
			p->from.offset = 0;
			p->to.type = D_REG;
			p->to.reg = 3;

			// L:
			//	MOVW.P R3, 0(R1) +4
			//	CMP R1, R2
			//	BNE L
			p = pl = appendp(p);
			p->as = AMOVW;
			p->from.type = D_REG;
			p->from.reg = 3;
			p->to.type = D_OREG;
			p->to.reg = 1;
			p->to.offset = 4;
			p->scond |= C_PBIT;

			p = appendp(p);
			p->as = ACMP;
			p->from.type = D_REG;
			p->from.reg = 1;
			p->reg = 2;

			p = appendp(p);
			p->as = ABNE;
			p->to.type = D_BRANCH;
			p->cond = pl;
		}
	}
}
