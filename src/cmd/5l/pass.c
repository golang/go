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

void
dodata(void)
{
	int i, t;
	Sym *s;
	Prog *p;
	int32 orig, v;

	if(debug['v'])
		Bprint(&bso, "%5.2f dodata\n", cputime());
	Bflush(&bso);
	for(p = datap; p != P; p = p->link) {
		s = p->from.sym;
		if(s->type == SBSS)
			s->type = SDATA;
		if(s->type != SDATA && s->type != SELFDATA)
			diag("initialize non-data (%d): %s\n%P",
				s->type, s->name, p);
		v = p->from.offset + p->reg;
		if(v > s->value)
			diag("initialize bounds (%d/%d): %s\n%P",
				v, s->value, s->name, p);
		if((s->type == SBSS || s->type == SDATA) && (p->to.type == D_CONST || p->to.type == D_OCONST) && (p->to.name == D_EXTERN || p->to.name == D_STATIC)){
			s = p->to.sym;
			if(s != S && (s->type == STEXT || s->type == SLEAF || s->type == SCONST || s->type == SXREF))
				s->fnptr = 1;
		}
	}

	if(debug['t']) {
		/*
		 * pull out string constants
		 */
		for(p = datap; p != P; p = p->link) {
			s = p->from.sym;
			if(p->to.type == D_SCONST)
				s->type = SSTRING;
		}
	}
	
	/*
	 * pass 0
	 * assign elf data - must be segregated from real data
	 */
	orig = 0;
	for(i=0; i<NHASH; i++)
	for(s = hash[i]; s != S; s = s->hash) {
		if(!s->reachable || s->type != SELFDATA)
			continue;
		v = s->value;
		while(v & 3)
			v++;
		s->size = v;
		s->value = orig;
		orig += v;
	}
	elfdatsize = orig;

	/*
	 * pass 1
	 *	assign 'small' variables to data segment
	 *	(rational is that data segment is more easily
	 *	 addressed through offset on R12)
	 */
	for(i=0; i<NHASH; i++)
	for(s = hash[i]; s != S; s = s->hash) {
		t = s->type;
		if(t != SDATA && t != SBSS)
			continue;
		v = s->value;
		if(v == 0) {
			diag("%s: no size", s->name);
			v = 1;
		}
		while(v & 3)
			v++;
		s->size = v;
		s->value = v;
		if(v > MINSIZ)
			continue;
		s->value = orig;
		orig += v;
		s->type = SDATA1;
	}

	/*
	 * pass 2
	 *	assign large 'data' variables to data segment
	 */
	for(i=0; i<NHASH; i++)
	for(s = hash[i]; s != S; s = s->hash) {
		t = s->type;
		if(t != SDATA) {
			if(t == SDATA1)
				s->type = SDATA;
			continue;
		}
		v = s->value;
		s->size = v;
		s->value = orig;
		orig += v;
	}

	while(orig & 7)
		orig++;
	datsize = orig;

	/*
	 * pass 3
	 *	everything else to bss segment
	 */
	for(i=0; i<NHASH; i++)
	for(s = hash[i]; s != S; s = s->hash) {
		if(s->type != SBSS)
			continue;
		v = s->value;
		s->size = v;
		s->value = orig;
		orig += v;
	}
	while(orig & 7)
		orig++;
	bsssize = orig-datsize;

	xdefine("setR12", SDATA, 0L+BIG);
	xdefine("bdata", SDATA, 0L);
	xdefine("data", SBSS, 0);
	xdefine("edata", SDATA, datsize);
	xdefine("end", SBSS, datsize+bsssize);
	xdefine("etext", STEXT, 0L);

	if(debug['s'])
		xdefine("symdat", SFIXED, 0);
	else
		xdefine("symdat", SFIXED, SYMDATVA);
}

void
undef(void)
{
	int i;
	Sym *s;

	for(i=0; i<NHASH; i++)
	for(s = hash[i]; s != S; s = s->hash)
		if(s->type == SXREF)
			diag("%s: not defined", s->name);
}

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
	setarch(p);
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
			if(a == AB || (a == ARET && q->scond == 14) || a == ARFE)
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
				if(a == AB || (a == ARET && q->scond == 14) || a == ARFE)
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
	if(a == AB || (a == ARET && p->scond == 14) || a == ARFE){
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
			setarch(p);
			a = p->as;
			if(seenthumb && a == ABL){
				// if((s = p->to.sym) != S && (s1 = curtext->from.sym) != S)
				//	print("%s calls %s\n", s1->name, s->name);
				 if((s = p->to.sym) != S && s->thumb != cursym->thumb)
					s->foreign = 1;
			}
			if((a == ABL || a == ABX || a == AB || a == ARET) &&
			   p->to.type != D_BRANCH && p->to.sym != S) {
				s = p->to.sym;
				switch(s->type) {
				default:
					diag("undefined: %s", s->name);
					s->type = STEXT;
					s->value = vexit;
					continue;	// avoid more error messages
				case STEXT:
					p->to.offset = s->value;
					p->to.type = D_BRANCH;
					break;
				}
			}
			if(p->to.type != D_BRANCH)
				continue;
			c = p->to.offset;
			for(q = textp->text; q != P;) {
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
			setarch(p);
			a = p->as;
			if(seenthumb && a == ABL) {
#ifdef CALLEEBX
				if(0)
					{}
#else
				if((s = p->to.sym) != S && (s->foreign || s->fnptr))
					p->as = ABX;
#endif
				else if(p->to.type == D_OREG)
					p->as = ABX;
			}
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

static void
fused(Adr *a, Prog *p, Prog *ct)
{
	Sym *s = a->sym;
	Use *u;

	if(s == S)
		return;
	if(a->type == D_OREG || a->type == D_OCONST || a->type == D_CONST){
		if(a->name == D_EXTERN || a->name == D_STATIC){
			u = malloc(sizeof(Use));
			u->p = p;
			u->ct = ct;
			u->link = s->use;
			s->use = u;
		}
	}
	else if(a->type == D_BRANCH){
		u = malloc(sizeof(Use));
		u->p = p;
		u->ct = ct;
		u->link = s->use;
		s->use = u;
	}
}

static int
ckfpuse(Prog *p, Prog *ct, Sym *fp, Sym *r)
{
	int reg;

	USED(fp);
	USED(ct);
	if(p->from.sym == r && p->as == AMOVW && (p->from.type == D_CONST || p->from.type == D_OREG) && p->reg == NREG && p->to.type == D_REG){
		reg = p->to.reg;
		for(p = p->link; p != P && p->as != ATEXT; p = p->link){
			if((p->as == ABL || p->as == ABX) && p->to.type == D_OREG && p->to.reg == reg)
				return 1;
			if(!debug['F'] && (isbranch(p) || p->as == ARET)){
				// print("%s: branch %P in %s\n", fp->name, p, ct->from.sym->name);
				return 0;
			}
			if((p->from.type == D_REG || p->from.type == D_OREG) && p->from.reg == reg){
				if(!debug['F'] && p->to.type != D_REG){
					// print("%s: store %P in %s\n", fp->name, p, ct->from.sym->name);
					return 0;
				}
				reg = p->to.reg;
			}
		}
	}
	// print("%s: no MOVW O(R), R\n", fp->name);
	return debug['F'];
}

static void
setfpuse(Prog *p, Sym *fp, Sym *r)
{
	int reg;

	if(p->from.sym == r && p->as == AMOVW && (p->from.type == D_CONST || p->from.type == D_OREG) && p->reg == NREG && p->to.type == D_REG){
		reg = p->to.reg;
		for(p = p->link; p != P && p->as != ATEXT; p = p->link){
			if((p->as == ABL || p->as == ABX) && p->to.type == D_OREG && p->to.reg == reg){
				fp->fnptr = 0;
				p->as = ABL;	// safe to do so
// print("simplified %s call\n", fp->name);
				break;
			}
			if(!debug['F'] && (isbranch(p) || p->as == ARET))
				diag("bad setfpuse call");
			if((p->from.type == D_REG || p->from.type == D_OREG) && p->from.reg == reg){
				if(!debug['F'] && p->to.type != D_REG)
					diag("bad setfpuse call");
				reg = p->to.reg;
			}
		}
	}
}

static int
cksymuse(Sym *s, int t)
{
	Prog *p;

	for(p = datap; p != P; p = p->link){
		if(p->from.sym == s && p->to.sym != nil && strcmp(p->to.sym->name, ".string") != 0 && p->to.sym->thumb != t){
			// print("%s %s %d %d ", p->from.sym->name, p->to.sym->name, p->to.sym->thumb, t);
			return 0;
		}
	}
	return 1;
}

/* check the use of s at the given point */
static int
ckuse(Sym *s, Sym *s0, Use *u)
{
	Sym *s1;

	s1 = u->p->from.sym;
// print("ckuse %s %s %s\n", s->name, s0->name, s1 ? s1->name : "nil");
	if(u->ct == nil){	/* in data area */
		if(s0 == s && !cksymuse(s1, s0->thumb)){
			// print("%s: cksymuse fails\n", s0->name);
			return 0;
		}
		for(u = s1->use; u != U; u = u->link)
			if(!ckuse(s1, s0, u))
				return 0;
	}
	else{		/* in text area */
		if(u->ct->from.sym->thumb != s0->thumb){
			// print("%s(%d): foreign call %s(%d)\n", s0->name, s0->thumb, u->ct->from.sym->name, u->ct->from.sym->thumb);
			return 0;
		}
		return ckfpuse(u->p, u->ct, s0, s);
	}
	return 1;
}

static void
setuse(Sym *s, Sym *s0, Use *u)
{
	Sym *s1;

	s1 = u->p->from.sym;
	if(u->ct == nil){	/* in data area */
		for(u = s1->use; u != U; u = u->link)
			setuse(s1, s0, u);
	}
	else{		/* in text area */
		setfpuse(u->p, s0, s);
	}
}

/* detect BX O(R) which can be done as BL O(R) */
void
fnptrs(void)
{
	int i;
	Sym *s;
	Prog *p;
	Use *u;

	for(i=0; i<NHASH; i++){
		for(s = hash[i]; s != S; s = s->hash){
			if(s->fnptr && (s->type == STEXT || s->type == SLEAF || s->type == SCONST)){
				// print("%s : fnptr %d %d\n", s->name, s->thumb, s->foreign);
			}
		}
	}
	/* record use of syms */
	for(cursym = textp; cursym != nil; cursym = cursym->next)
	for(p = cursym->text; p != P; p = p->link) {
		if(p->as != ATEXT) {
			fused(&p->from, p, cursym->text);
			fused(&p->to, p, cursym->text);
		}
	}
	for(p = datap; p != P; p = p->link)
		fused(&p->to, p, nil);

	/* now look for fn ptrs */
	for(i=0; i<NHASH; i++){
		for(s = hash[i]; s != S; s = s->hash){
			if(s->fnptr && (s->type == STEXT || s->type == SLEAF || s->type == SCONST)){
				for(u = s->use; u != U; u = u->link){
					if(!ckuse(s, s, u))
						break;
				}
				if(u == U){		// can simplify
					for(u = s->use; u != U; u = u->link)
						setuse(s, s, u);
				}
			}
		}
	}

	/*  now free Use structures */
}

Prog*
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
	s->data = p;
	return p;
}
