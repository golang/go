// Inferno utils/5c/peep.c
// http://code.google.com/p/inferno-os/source/browse/utils/5c/peep.c
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


#include <u.h>
#include <libc.h>
#include "gg.h"
#include "opt.h"

static int	xtramodes(Graph*, Flow*, Adr*);
static int	shortprop(Flow *r);
static int	subprop(Flow*);
static int	copyprop(Graph*, Flow*);
static int	copy1(Adr*, Adr*, Flow*, int);
static int	copyas(Adr*, Adr*);
static int	copyau(Adr*, Adr*);
static int	copysub(Adr*, Adr*, Adr*, int);
static int	copysub1(Prog*, Adr*, Adr*, int);
static Flow*	findpre(Flow *r, Adr *v);
static int	copyau1(Prog *p, Adr *v);
static int	isdconst(Addr *a);
static int	isfloatreg(Addr*);

static uint32	gactive;

// UNUSED
int	shiftprop(Flow *r);
void	constprop(Adr *c1, Adr *v1, Flow *r);
void	predicate(Graph*);


void
peep(Prog *firstp)
{
	Flow *r;
	Graph *g;
	Prog *p;
	int t;

	g = flowstart(firstp, sizeof(Flow));
	if(g == nil)
		return;
	gactive = 0;

loop1:
	if(debug['P'] && debug['v'])
		dumpit("loop1", g->start, 0);

	t = 0;
	for(r=g->start; r!=nil; r=r->link) {
		p = r->prog;
		switch(p->as) {
		case ASLL:
		case ASRL:
		case ASRA:
			/*
			 * elide shift into TYPE_SHIFT operand of subsequent instruction
			 */
//			if(shiftprop(r)) {
//				excise(r);
//				t++;
//				break;
//			}
			break;

		case AMOVB:
		case AMOVH:
		case AMOVW:
		case AMOVF:
		case AMOVD:
			if(regtyp(&p->from))
			if(p->from.type == p->to.type && isfloatreg(&p->from) == isfloatreg(&p->to))
			if(p->scond == C_SCOND_NONE) {
				if(copyprop(g, r)) {
					excise(r);
					t++;
					break;
				}
				if(subprop(r) && copyprop(g, r)) {
					excise(r);
					t++;
					break;
				}
			}
			break;

		case AMOVHS:
		case AMOVHU:
		case AMOVBS:
		case AMOVBU:
			if(p->from.type == TYPE_REG) {
				if(shortprop(r))
					t++;
			}
			break;

#ifdef NOTDEF
XXX
			if(p->scond == C_SCOND_NONE)
			if(regtyp(&p->to))
			if(isdconst(&p->from)) {
				constprop(&p->from, &p->to, r->s1);
			}
			break;
#endif
		}
	}
	if(t)
		goto loop1;

	for(r=g->start; r!=nil; r=r->link) {
		p = r->prog;
		switch(p->as) {
		case AEOR:
			/*
			 * EOR -1,x,y => MVN x,y
			 */
			if(isdconst(&p->from) && p->from.offset == -1) {
				p->as = AMVN;
				p->from.type = TYPE_REG;
				if(p->reg != 0)
					p->from.reg = p->reg;
				else
					p->from.reg = p->to.reg;
				p->reg = 0;
			}
			break;
		}
	}

	for(r=g->start; r!=nil; r=r->link) {
		p = r->prog;
		switch(p->as) {
		case AMOVW:
		case AMOVB:
		case AMOVBS:
		case AMOVBU:
			if(p->from.type == TYPE_MEM && p->from.offset == 0)
				xtramodes(g, r, &p->from);
			else
			if(p->to.type == TYPE_MEM && p->to.offset == 0)
				xtramodes(g, r, &p->to);
			else
				continue;
			break;
//		case ACMP:
//			/*
//			 * elide CMP $0,x if calculation of x can set condition codes
//			 */
//			if(isdconst(&p->from) || p->from.offset != 0)
//				continue;
//			r2 = r->s1;
//			if(r2 == nil)
//				continue;
//			t = r2->prog->as;
//			switch(t) {
//			default:
//				continue;
//			case ABEQ:
//			case ABNE:
//			case ABMI:
//			case ABPL:
//				break;
//			case ABGE:
//				t = ABPL;
//				break;
//			case ABLT:
//				t = ABMI;
//				break;
//			case ABHI:
//				t = ABNE;
//				break;
//			case ABLS:
//				t = ABEQ;
//				break;
//			}
//			r1 = r;
//			do
//				r1 = uniqp(r1);
//			while (r1 != nil && r1->prog->as == ANOP);
//			if(r1 == nil)
//				continue;
//			p1 = r1->prog;
//			if(p1->to.type != TYPE_REG)
//				continue;
//			if(p1->to.reg != p->reg)
//			if(!(p1->as == AMOVW && p1->from.type == TYPE_REG && p1->from.reg == p->reg))
//				continue;
//
//			switch(p1->as) {
//			default:
//				continue;
//			case AMOVW:
//				if(p1->from.type != TYPE_REG)
//					continue;
//			case AAND:
//			case AEOR:
//			case AORR:
//			case ABIC:
//			case AMVN:
//			case ASUB:
//			case ARSB:
//			case AADD:
//			case AADC:
//			case ASBC:
//			case ARSC:
//				break;
//			}
//			p1->scond |= C_SBIT;
//			r2->prog->as = t;
//			excise(r);
//			continue;
		}
	}

//	predicate(g);

	flowend(g);
}

int
regtyp(Adr *a)
{
	return a->type == TYPE_REG && (REG_R0 <= a->reg && a->reg <= REG_R15 || REG_F0 <= a->reg && a->reg <= REG_F15);
}

/*
 * the idea is to substitute
 * one register for another
 * from one MOV to another
 *	MOV	a, R0
 *	ADD	b, R0	/ no use of R1
 *	MOV	R0, R1
 * would be converted to
 *	MOV	a, R1
 *	ADD	b, R1
 *	MOV	R1, R0
 * hopefully, then the former or latter MOV
 * will be eliminated by copy propagation.
 */
static int
subprop(Flow *r0)
{
	Prog *p;
	Adr *v1, *v2;
	Flow *r;
	int t;
	ProgInfo info;

	p = r0->prog;
	v1 = &p->from;
	if(!regtyp(v1))
		return 0;
	v2 = &p->to;
	if(!regtyp(v2))
		return 0;
	for(r=uniqp(r0); r!=nil; r=uniqp(r)) {
		if(uniqs(r) == nil)
			break;
		p = r->prog;
		if(p->as == AVARDEF || p->as == AVARKILL)
			continue;
		proginfo(&info, p);
		if(info.flags & Call)
			return 0;

		if((info.flags & CanRegRead) && p->to.type == TYPE_REG) {
			info.flags |= RegRead;
			info.flags &= ~(CanRegRead | RightRead);
			p->reg = p->to.reg;
		}

		switch(p->as) {
		case AMULLU:
		case AMULA:
		case AMVN:
			return 0;
		}
		
		if((info.flags & (RightRead|RightWrite)) == RightWrite) {
			if(p->to.type == v1->type)
			if(p->to.reg == v1->reg)
			if(p->scond == C_SCOND_NONE)
				goto gotit;
		}

		if(copyau(&p->from, v2) ||
		   copyau1(p, v2) ||
		   copyau(&p->to, v2))
			break;
		if(copysub(&p->from, v1, v2, 0) ||
		   copysub1(p, v1, v2, 0) ||
		   copysub(&p->to, v1, v2, 0))
			break;
	}
	return 0;

gotit:
	copysub(&p->to, v1, v2, 1);
	if(debug['P']) {
		print("gotit: %D->%D\n%P", v1, v2, r->prog);
		if(p->from.type == v2->type)
			print(" excise");
		print("\n");
	}
	for(r=uniqs(r); r!=r0; r=uniqs(r)) {
		p = r->prog;
		copysub(&p->from, v1, v2, 1);
		copysub1(p, v1, v2, 1);
		copysub(&p->to, v1, v2, 1);
		if(debug['P'])
			print("%P\n", r->prog);
	}
	t = v1->reg;
	v1->reg = v2->reg;
	v2->reg = t;
	if(debug['P'])
		print("%P last\n", r->prog);
	return 1;
}

/*
 * The idea is to remove redundant copies.
 *	v1->v2	F=0
 *	(use v2	s/v2/v1/)*
 *	set v1	F=1
 *	use v2	return fail
 *	-----------------
 *	v1->v2	F=0
 *	(use v2	s/v2/v1/)*
 *	set v1	F=1
 *	set v2	return success
 */
static int
copyprop(Graph *g, Flow *r0)
{
	Prog *p;
	Adr *v1, *v2;

	USED(g);
	p = r0->prog;
	v1 = &p->from;
	v2 = &p->to;
	if(copyas(v1, v2))
		return 1;
	gactive++;
	return copy1(v1, v2, r0->s1, 0);
}

static int
copy1(Adr *v1, Adr *v2, Flow *r, int f)
{
	int t;
	Prog *p;

	if(r->active == gactive) {
		if(debug['P'])
			print("act set; return 1\n");
		return 1;
	}
	r->active = gactive;
	if(debug['P'])
		print("copy %D->%D f=%d\n", v1, v2, f);
	for(; r != nil; r = r->s1) {
		p = r->prog;
		if(debug['P'])
			print("%P", p);
		if(!f && uniqp(r) == nil) {
			f = 1;
			if(debug['P'])
				print("; merge; f=%d", f);
		}
		t = copyu(p, v2, nil);
		switch(t) {
		case 2:	/* rar, can't split */
			if(debug['P'])
				print("; %Drar; return 0\n", v2);
			return 0;

		case 3:	/* set */
			if(debug['P'])
				print("; %Dset; return 1\n", v2);
			return 1;

		case 1:	/* used, substitute */
		case 4:	/* use and set */
			if(f) {
				if(!debug['P'])
					return 0;
				if(t == 4)
					print("; %Dused+set and f=%d; return 0\n", v2, f);
				else
					print("; %Dused and f=%d; return 0\n", v2, f);
				return 0;
			}
			if(copyu(p, v2, v1)) {
				if(debug['P'])
					print("; sub fail; return 0\n");
				return 0;
			}
			if(debug['P'])
				print("; sub%D/%D", v2, v1);
			if(t == 4) {
				if(debug['P'])
					print("; %Dused+set; return 1\n", v2);
				return 1;
			}
			break;
		}
		if(!f) {
			t = copyu(p, v1, nil);
			if(!f && (t == 2 || t == 3 || t == 4)) {
				f = 1;
				if(debug['P'])
					print("; %Dset and !f; f=%d", v1, f);
			}
		}
		if(debug['P'])
			print("\n");
		if(r->s2)
			if(!copy1(v1, v2, r->s2, f))
				return 0;
	}
	return 1;
}

// UNUSED
/*
 * The idea is to remove redundant constants.
 *	$c1->v1
 *	($c1->v2 s/$c1/v1)*
 *	set v1  return
 * The v1->v2 should be eliminated by copy propagation.
 */
void
constprop(Adr *c1, Adr *v1, Flow *r)
{
	Prog *p;

	if(debug['P'])
		print("constprop %D->%D\n", c1, v1);
	for(; r != nil; r = r->s1) {
		p = r->prog;
		if(debug['P'])
			print("%P", p);
		if(uniqp(r) == nil) {
			if(debug['P'])
				print("; merge; return\n");
			return;
		}
		if(p->as == AMOVW && copyas(&p->from, c1)) {
				if(debug['P'])
					print("; sub%D/%D", &p->from, v1);
				p->from = *v1;
		} else if(copyu(p, v1, nil) > 1) {
			if(debug['P'])
				print("; %Dset; return\n", v1);
			return;
		}
		if(debug['P'])
			print("\n");
		if(r->s2)
			constprop(c1, v1, r->s2);
	}
}

/*
 * shortprop eliminates redundant zero/sign extensions.
 *
 *   MOVBS x, R
 *   <no use R>
 *   MOVBS R, R'
 *
 * changed to
 *
 *   MOVBS x, R
 *   ...
 *   MOVB  R, R' (compiled to mov)
 *
 * MOVBS above can be a MOVBS, MOVBU, MOVHS or MOVHU.
 */
static int
shortprop(Flow *r)
{
	Prog *p, *p1;
	Flow *r1;

	p = r->prog;
	r1 = findpre(r, &p->from);
	if(r1 == nil)
		return 0;

	p1 = r1->prog;
	if(p1->as == p->as) {
		// Two consecutive extensions.
		goto gotit;
	}

	if(p1->as == AMOVW && isdconst(&p1->from)
	   && p1->from.offset >= 0 && p1->from.offset < 128) {
		// Loaded an immediate.
		goto gotit;
	}

	return 0;

gotit:
	if(debug['P'])
		print("shortprop\n%P\n%P", p1, p);
	switch(p->as) {
	case AMOVBS:
	case AMOVBU:
		p->as = AMOVB;
		break;
	case AMOVHS:
	case AMOVHU:
		p->as = AMOVH;
		break;
	}
	if(debug['P'])
		print(" => %A\n", p->as);
	return 1;
}

// UNUSED
/*
 * ASLL x,y,w
 * .. (not use w, not set x y w)
 * AXXX w,a,b (a != w)
 * .. (not use w)
 * (set w)
 * ----------- changed to
 * ..
 * AXXX (x<<y),a,b
 * ..
 */
#define FAIL(msg) { if(debug['P']) print("\t%s; FAILURE\n", msg); return 0; }
/*c2go void FAIL(char*); */

int
shiftprop(Flow *r)
{
	Flow *r1;
	Prog *p, *p1, *p2;
	int n, o;
	Adr a;

	p = r->prog;
	if(p->to.type != TYPE_REG)
		FAIL("BOTCH: result not reg");
	n = p->to.reg;
	a = zprog.from;
	if(p->reg != 0 && p->reg != p->to.reg) {
		a.type = TYPE_REG;
		a.reg = p->reg;
	}
	if(debug['P'])
		print("shiftprop\n%P", p);
	r1 = r;
	for(;;) {
		/* find first use of shift result; abort if shift operands or result are changed */
		r1 = uniqs(r1);
		if(r1 == nil)
			FAIL("branch");
		if(uniqp(r1) == nil)
			FAIL("merge");
		p1 = r1->prog;
		if(debug['P'])
			print("\n%P", p1);
		switch(copyu(p1, &p->to, nil)) {
		case 0:	/* not used or set */
			if((p->from.type == TYPE_REG && copyu(p1, &p->from, nil) > 1) ||
			   (a.type == TYPE_REG && copyu(p1, &a, nil) > 1))
				FAIL("args modified");
			continue;
		case 3:	/* set, not used */
			FAIL("BOTCH: noref");
		}
		break;
	}
	/* check whether substitution can be done */
	switch(p1->as) {
	default:
		FAIL("non-dpi");
	case AAND:
	case AEOR:
	case AADD:
	case AADC:
	case AORR:
	case ASUB:
	case ASBC:
	case ARSB:
	case ARSC:
		if(p1->reg == n || (p1->reg == 0 && p1->to.type == TYPE_REG && p1->to.reg == n)) {
			if(p1->from.type != TYPE_REG)
				FAIL("can't swap");
			p1->reg = p1->from.reg;
			p1->from.reg = n;
			switch(p1->as) {
			case ASUB:
				p1->as = ARSB;
				break;
			case ARSB:
				p1->as = ASUB;
				break;
			case ASBC:
				p1->as = ARSC;
				break;
			case ARSC:
				p1->as = ASBC;
				break;
			}
			if(debug['P'])
				print("\t=>%P", p1);
		}
	case ABIC:
	case ATST:
	case ACMP:
	case ACMN:
		if(p1->reg == n)
			FAIL("can't swap");
		if(p1->reg == 0 && p1->to.reg == n)
			FAIL("shift result used twice");
//	case AMVN:
		if(p1->from.type == TYPE_SHIFT)
			FAIL("shift result used in shift");
		if(p1->from.type != TYPE_REG || p1->from.reg != n)
			FAIL("BOTCH: where is it used?");
		break;
	}
	/* check whether shift result is used subsequently */
	p2 = p1;
	if(p1->to.reg != n)
	for (;;) {
		r1 = uniqs(r1);
		if(r1 == nil)
			FAIL("inconclusive");
		p1 = r1->prog;
		if(debug['P'])
			print("\n%P", p1);
		switch(copyu(p1, &p->to, nil)) {
		case 0:	/* not used or set */
			continue;
		case 3: /* set, not used */
			break;
		default:/* used */
			FAIL("reused");
		}
		break;
	}

	/* make the substitution */
	p2->from.reg = 0;
	o = p->reg;
	if(o == 0)
		o = p->to.reg;
	o &= 15;

	switch(p->from.type){
	case TYPE_CONST:
		o |= (p->from.offset&0x1f)<<7;
		break;
	case TYPE_REG:
		o |= (1<<4) | ((p->from.reg&15)<<8);
		break;
	}
	switch(p->as){
	case ASLL:
		o |= 0<<5;
		break;
	case ASRL:
		o |= 1<<5;
		break;
	case ASRA:
		o |= 2<<5;
		break;
	}
	p2->from = zprog.from;
	p2->from.type = TYPE_SHIFT;
	p2->from.offset = o;
	if(debug['P'])
		print("\t=>%P\tSUCCEED\n", p2);
	return 1;
}

/*
 * findpre returns the last instruction mentioning v
 * before r. It must be a set, and there must be
 * a unique path from that instruction to r.
 */
static Flow*
findpre(Flow *r, Adr *v)
{
	Flow *r1;

	for(r1=uniqp(r); r1!=nil; r=r1,r1=uniqp(r)) {
		if(uniqs(r1) != r)
			return nil;
		switch(copyu(r1->prog, v, nil)) {
		case 1: /* used */
		case 2: /* read-alter-rewrite */
			return nil;
		case 3: /* set */
		case 4: /* set and used */
			return r1;
		}
	}
	return nil;
}

/*
 * findinc finds ADD instructions with a constant
 * argument which falls within the immed_12 range.
 */
static Flow*
findinc(Flow *r, Flow *r2, Adr *v)
{
	Flow *r1;
	Prog *p;


	for(r1=uniqs(r); r1!=nil && r1!=r2; r=r1,r1=uniqs(r)) {
		if(uniqp(r1) != r)
			return nil;
		switch(copyu(r1->prog, v, nil)) {
		case 0: /* not touched */
			continue;
		case 4: /* set and used */
			p = r1->prog;
			if(p->as == AADD)
			if(isdconst(&p->from))
			if(p->from.offset > -4096 && p->from.offset < 4096)
				return r1;
		default:
			return nil;
		}
	}
	return nil;
}

static int
nochange(Flow *r, Flow *r2, Prog *p)
{
	Adr a[3];
	int i, n;

	if(r == r2)
		return 1;
	n = 0;
	if(p->reg != 0 && p->reg != p->to.reg) {
		a[n].type = TYPE_REG;
		a[n++].reg = p->reg;
	}
	switch(p->from.type) {
	case TYPE_SHIFT:
		a[n].type = TYPE_REG;
		a[n++].reg = REG_R0 + (p->from.offset&0xf);
	case TYPE_REG:
		a[n].type = TYPE_REG;
		a[n++].reg = p->from.reg;
	}
	if(n == 0)
		return 1;
	for(; r!=nil && r!=r2; r=uniqs(r)) {
		p = r->prog;
		for(i=0; i<n; i++)
			if(copyu(p, &a[i], nil) > 1)
				return 0;
	}
	return 1;
}

static int
findu1(Flow *r, Adr *v)
{
	for(; r != nil; r = r->s1) {
		if(r->active)
			return 0;
		r->active = 1;
		switch(copyu(r->prog, v, nil)) {
		case 1: /* used */
		case 2: /* read-alter-rewrite */
		case 4: /* set and used */
			return 1;
		case 3: /* set */
			return 0;
		}
		if(r->s2)
			if (findu1(r->s2, v))
				return 1;
	}
	return 0;
}

static int
finduse(Graph *g, Flow *r, Adr *v)
{
	Flow *r1;

	for(r1=g->start; r1!=nil; r1=r1->link)
		r1->active = 0;
	return findu1(r, v);
}

/*
 * xtramodes enables the ARM post increment and
 * shift offset addressing modes to transform
 *   MOVW   0(R3),R1
 *   ADD    $4,R3,R3
 * into
 *   MOVW.P 4(R3),R1
 * and 
 *   ADD    R0,R1
 *   MOVBU  0(R1),R0
 * into 
 *   MOVBU  R0<<0(R1),R0
 */
static int
xtramodes(Graph *g, Flow *r, Adr *a)
{
	Flow *r1, *r2, *r3;
	Prog *p, *p1;
	Adr v;

	p = r->prog;
	v = *a;
	v.type = TYPE_REG;
	r1 = findpre(r, &v);
	if(r1 != nil) {
		p1 = r1->prog;
		if(p1->to.type == TYPE_REG && p1->to.reg == v.reg)
		switch(p1->as) {
		case AADD:
			if(p1->scond & C_SBIT)
				// avoid altering ADD.S/ADC sequences.
				break;
			if(p1->from.type == TYPE_REG ||
			   (p1->from.type == TYPE_SHIFT && (p1->from.offset&(1<<4)) == 0 &&
			    ((p->as != AMOVB && p->as != AMOVBS) || (a == &p->from && (p1->from.offset&~0xf) == 0))) ||
			   ((p1->from.type == TYPE_ADDR || p1->from.type == TYPE_CONST) &&
			    p1->from.offset > -4096 && p1->from.offset < 4096))
			if(nochange(uniqs(r1), r, p1)) {
				if(a != &p->from || v.reg != p->to.reg)
				if (finduse(g, r->s1, &v)) {
					if(p1->reg == 0 || p1->reg == v.reg)
						/* pre-indexing */
						p->scond |= C_WBIT;
					else return 0;
				}
				switch (p1->from.type) {
				case TYPE_REG:
					/* register offset */
					if(nacl)
						return 0;
					*a = zprog.from;
					a->type = TYPE_SHIFT;
					a->offset = p1->from.reg&15;
					break;
				case TYPE_SHIFT:
					/* scaled register offset */
					if(nacl)
						return 0;
					*a = zprog.from;
					a->type = TYPE_SHIFT;
				case TYPE_CONST:
				case TYPE_ADDR:
					/* immediate offset */
					a->offset = p1->from.offset;
					break;
				}
				if(p1->reg != 0)
					a->reg = p1->reg;
				excise(r1);
				return 1;
			}
			break;
		case AMOVW:
			if(p1->from.type == TYPE_REG)
			if((r2 = findinc(r1, r, &p1->from)) != nil) {
			for(r3=uniqs(r2); r3->prog->as==ANOP; r3=uniqs(r3))
				;
			if(r3 == r) {
				/* post-indexing */
				p1 = r2->prog;
				a->reg = p1->to.reg;
				a->offset = p1->from.offset;
				p->scond |= C_PBIT;
				if(!finduse(g, r, &r1->prog->to))
					excise(r1);
				excise(r2);
				return 1;
			}
			}
			break;
		}
	}
	if(a != &p->from || a->reg != p->to.reg)
	if((r1 = findinc(r, nil, &v)) != nil) {
		/* post-indexing */
		p1 = r1->prog;
		a->offset = p1->from.offset;
		p->scond |= C_PBIT;
		excise(r1);
		return 1;
	}
	return 0;
}

/*
 * return
 * 1 if v only used (and substitute),
 * 2 if read-alter-rewrite
 * 3 if set
 * 4 if set and used
 * 0 otherwise (not touched)
 */
int
copyu(Prog *p, Adr *v, Adr *s)
{
	switch(p->as) {

	default:
		print("copyu: can't find %A\n", p->as);
		return 2;

	case AMOVM:
		if(v->type != TYPE_REG)
			return 0;
		if(p->from.type == TYPE_CONST) {	/* read reglist, read/rar */
			if(s != nil) {
				if(p->from.offset&(1<<v->reg))
					return 1;
				if(copysub(&p->to, v, s, 1))
					return 1;
				return 0;
			}
			if(copyau(&p->to, v)) {
				if(p->scond&C_WBIT)
					return 2;
				return 1;
			}
			if(p->from.offset&(1<<v->reg))
				return 1;
		} else {			/* read/rar, write reglist */
			if(s != nil) {
				if(p->to.offset&(1<<v->reg))
					return 1;
				if(copysub(&p->from, v, s, 1))
					return 1;
				return 0;
			}
			if(copyau(&p->from, v)) {
				if(p->scond&C_WBIT)
					return 2;
				if(p->to.offset&(1<<v->reg))
					return 4;
				return 1;
			}
			if(p->to.offset&(1<<v->reg))
				return 3;
		}
		return 0;

	case ANOP:	/* read,, write */
	case AMOVW:
	case AMOVF:
	case AMOVD:
	case AMOVH:
	case AMOVHS:
	case AMOVHU:
	case AMOVB:
	case AMOVBS:
	case AMOVBU:
	case AMOVFW:
	case AMOVWF:
	case AMOVDW:
	case AMOVWD:
	case AMOVFD:
	case AMOVDF:
		if(p->scond&(C_WBIT|C_PBIT))
		if(v->type == TYPE_REG) {
			if(p->from.type == TYPE_MEM || p->from.type == TYPE_SHIFT) {
				if(p->from.reg == v->reg)
					return 2;
			} else {
		  		if(p->to.reg == v->reg)
					return 2;
			}
		}
		if(s != nil) {
			if(copysub(&p->from, v, s, 1))
				return 1;
			if(!copyas(&p->to, v))
				if(copysub(&p->to, v, s, 1))
					return 1;
			return 0;
		}
		if(copyas(&p->to, v)) {
			if(p->scond != C_SCOND_NONE)
				return 2;
			if(copyau(&p->from, v))
				return 4;
			return 3;
		}
		if(copyau(&p->from, v))
			return 1;
		if(copyau(&p->to, v))
			return 1;
		return 0;

	case AMULLU:	/* read, read, write, write */
	case AMULL:
	case AMULA:
	case AMVN:
		return 2;

	case AADD:	/* read, read, write */
	case AADC:
	case ASUB:
	case ASBC:
	case ARSB:
	case ASLL:
	case ASRL:
	case ASRA:
	case AORR:
	case AAND:
	case AEOR:
	case AMUL:
	case AMULU:
	case ADIV:
	case ADIVU:
	case AMOD:
	case AMODU:
	case AADDF:
	case AADDD:
	case ASUBF:
	case ASUBD:
	case AMULF:
	case AMULD:
	case ADIVF:
	case ADIVD:

	case ACHECKNIL: /* read */
	case ACMPF:	/* read, read, */
	case ACMPD:
	case ACMP:
	case ACMN:
	case ACASE:
	case ATST:	/* read,, */
		if(s != nil) {
			if(copysub(&p->from, v, s, 1))
				return 1;
			if(copysub1(p, v, s, 1))
				return 1;
			if(!copyas(&p->to, v))
				if(copysub(&p->to, v, s, 1))
					return 1;
			return 0;
		}
		if(copyas(&p->to, v)) {
			if(p->scond != C_SCOND_NONE)
				return 2;
			if(p->reg == 0)
				p->reg = p->to.reg;
			if(copyau(&p->from, v))
				return 4;
			if(copyau1(p, v))
				return 4;
			return 3;
		}
		if(copyau(&p->from, v))
			return 1;
		if(copyau1(p, v))
			return 1;
		if(copyau(&p->to, v))
			return 1;
		return 0;

	case ABEQ:	/* read, read */
	case ABNE:
	case ABCS:
	case ABHS:
	case ABCC:
	case ABLO:
	case ABMI:
	case ABPL:
	case ABVS:
	case ABVC:
	case ABHI:
	case ABLS:
	case ABGE:
	case ABLT:
	case ABGT:
	case ABLE:
		if(s != nil) {
			if(copysub(&p->from, v, s, 1))
				return 1;
			return copysub1(p, v, s, 1);
		}
		if(copyau(&p->from, v))
			return 1;
		if(copyau1(p, v))
			return 1;
		return 0;

	case AB:	/* funny */
		if(s != nil) {
			if(copysub(&p->to, v, s, 1))
				return 1;
			return 0;
		}
		if(copyau(&p->to, v))
			return 1;
		return 0;

	case ARET:	/* funny */
		if(s != nil)
			return 1;
		return 3;

	case ABL:	/* funny */
		if(v->type == TYPE_REG) {
			// TODO(rsc): REG_R0 and REG_F0 used to be
			// (when register numbers started at 0) exregoffset and exfregoffset,
			// which are unset entirely. 
			// It's strange that this handles R0 and F0 differently from the other
			// registers. Possible failure to optimize?
			if(REG_R0 < v->reg && v->reg <= REGEXT)
				return 2;
			if(v->reg == REGARG)
				return 2;
			if(REG_F0 < v->reg && v->reg <= FREGEXT)
				return 2;
		}
		if(p->from.type == TYPE_REG && v->type == TYPE_REG && p->from.reg == v->reg)
			return 2;

		if(s != nil) {
			if(copysub(&p->to, v, s, 1))
				return 1;
			return 0;
		}
		if(copyau(&p->to, v))
			return 4;
		return 3;
	case ADUFFZERO:
		// R0 is zero, used by DUFFZERO, cannot be substituted.
		// R1 is ptr to memory, used and set, cannot be substituted.
		if(v->type == TYPE_REG) {
			if(v->reg == REGALLOC_R0)
				return 1;
			if(v->reg == REGALLOC_R0+1)
				return 2;
		}
		return 0;
	case ADUFFCOPY:
		// R0 is scratch, set by DUFFCOPY, cannot be substituted.
		// R1, R2 areptr to src, dst, used and set, cannot be substituted.
		if(v->type == TYPE_REG) {
			if(v->reg == REGALLOC_R0)
				return 3;
			if(v->reg == REGALLOC_R0+1 || v->reg == REGALLOC_R0+2)
				return 2;
		}
		return 0;
			
	case ATEXT:	/* funny */
		if(v->type == TYPE_REG)
			if(v->reg == REGARG)
				return 3;
		return 0;

	case APCDATA:
	case AFUNCDATA:
	case AVARDEF:
	case AVARKILL:
		return 0;
	}
}

/*
 * direct reference,
 * could be set/use depending on
 * semantics
 */
static int
copyas(Adr *a, Adr *v)
{

	if(regtyp(v)) {
		if(a->type == v->type)
		if(a->reg == v->reg)
			return 1;
	} else
	if(v->type == TYPE_CONST) {		/* for constprop */
		if(a->type == v->type)
		if(a->name == v->name)
		if(a->sym == v->sym)
		if(a->reg == v->reg)
		if(a->offset == v->offset)
			return 1;
	}
	return 0;
}

int
sameaddr(Adr *a, Adr *v)
{
	if(a->type != v->type)
		return 0;
	if(regtyp(v) && a->reg == v->reg)
		return 1;
	// TODO(rsc): Change v->type to v->name and enable.
	//if(v->type == NAME_AUTO || v->type == NAME_PARAM) {
	//	if(v->offset == a->offset)
	//		return 1;
	//}
	return 0;
}

/*
 * either direct or indirect
 */
static int
copyau(Adr *a, Adr *v)
{

	if(copyas(a, v))
		return 1;
	if(v->type == TYPE_REG) {
		if(a->type == TYPE_ADDR && a->reg != 0) {
			if(a->reg == v->reg)
				return 1;
		} else
		if(a->type == TYPE_MEM) {
			if(a->reg == v->reg)
				return 1;
		} else
		if(a->type == TYPE_REGREG || a->type == TYPE_REGREG2) {
			if(a->reg == v->reg)
				return 1;
			if(a->offset == v->reg)
				return 1;
		} else
		if(a->type == TYPE_SHIFT) {
			if((a->offset&0xf) == v->reg - REG_R0)
				return 1;
			if((a->offset&(1<<4)) && ((a->offset>>8)&0xf) == v->reg - REG_R0)
				return 1;
		}
	}
	return 0;
}

/*
 * compare v to the center
 * register in p (p->reg)
 */
static int
copyau1(Prog *p, Adr *v)
{
	if(v->type == TYPE_REG && v->reg == 0)
		return 0;
	return p->reg == v->reg;
}

/*
 * substitute s for v in a
 * return failure to substitute
 */
static int
copysub(Adr *a, Adr *v, Adr *s, int f)
{

	if(f)
	if(copyau(a, v)) {
		if(a->type == TYPE_SHIFT) {
			if((a->offset&0xf) == v->reg - REG_R0)
				a->offset = (a->offset&~0xf)|(s->reg&0xf);
			if((a->offset&(1<<4)) && ((a->offset>>8)&0xf) == v->reg - REG_R0)
				a->offset = (a->offset&~(0xf<<8))|((s->reg&0xf)<<8);
		} else
		if(a->type == TYPE_REGREG || a->type == TYPE_REGREG2) {
			if(a->offset == v->reg)
				a->offset = s->reg;
			if(a->reg == v->reg)
				a->reg = s->reg;
		} else
			a->reg = s->reg;
	}
	return 0;
}

static int
copysub1(Prog *p1, Adr *v, Adr *s, int f)
{

	if(f)
	if(copyau1(p1, v))
		p1->reg = s->reg;
	return 0;
}

struct {
	int opcode;
	int notopcode;
	int scond;
	int notscond;
} predinfo[]  = {
	{ ABEQ,	ABNE,	0x0,	0x1, },
	{ ABNE,	ABEQ,	0x1,	0x0, },
	{ ABCS,	ABCC,	0x2,	0x3, },
	{ ABHS,	ABLO,	0x2,	0x3, },
	{ ABCC,	ABCS,	0x3,	0x2, },
	{ ABLO,	ABHS,	0x3,	0x2, },
	{ ABMI,	ABPL,	0x4,	0x5, },
	{ ABPL,	ABMI,	0x5,	0x4, },
	{ ABVS,	ABVC,	0x6,	0x7, },
	{ ABVC,	ABVS,	0x7,	0x6, },
	{ ABHI,	ABLS,	0x8,	0x9, },
	{ ABLS,	ABHI,	0x9,	0x8, },
	{ ABGE,	ABLT,	0xA,	0xB, },
	{ ABLT,	ABGE,	0xB,	0xA, },
	{ ABGT,	ABLE,	0xC,	0xD, },
	{ ABLE,	ABGT,	0xD,	0xC, },
};

typedef struct {
	Flow *start;
	Flow *last;
	Flow *end;
	int len;
} Joininfo;

enum {
	Join,
	Split,
	End,
	Branch,
	Setcond,
	Toolong
};

enum {
	Falsecond,
	Truecond,
	Delbranch,
	Keepbranch
};

static int
isbranch(Prog *p)
{
	return (ABEQ <= p->as) && (p->as <= ABLE);
}

static int
predicable(Prog *p)
{
	switch(p->as) {
	case ANOP:
	case AXXX:
	case ADATA:
	case AGLOBL:
	case ATEXT:
	case AWORD:
	case ABCASE:
	case ACASE:
		return 0;
	}
	if(isbranch(p))
		return 0;
	return 1;
}

/*
 * Depends on an analysis of the encodings performed by 5l.
 * These seem to be all of the opcodes that lead to the "S" bit
 * being set in the instruction encodings.
 *
 * C_SBIT may also have been set explicitly in p->scond.
 */
static int
modifiescpsr(Prog *p)
{
	switch(p->as) {
	case AMULLU:
	case AMULA:
	case AMULU:
	case ADIVU:

	case ATEQ:
	case ACMN:
	case ATST:
	case ACMP:
	case AMUL:
	case ADIV:
	case AMOD:
	case AMODU:
	case ABL:
		return 1;
	}
	if(p->scond & C_SBIT)
		return 1;
	return 0;
}

/*
 * Find the maximal chain of instructions starting with r which could
 * be executed conditionally
 */
static int
joinsplit(Flow *r, Joininfo *j)
{
	j->start = r;
	j->last = r;
	j->len = 0;
	do {
		if (r->p2 && (r->p1 || r->p2->p2link)) {
			j->end = r;
			return Join;
		}
		if (r->s1 && r->s2) {
			j->end = r;
			return Split;
		}
		j->last = r;
		if (r->prog->as != ANOP)
			j->len++;
		if (!r->s1 && !r->s2) {
			j->end = r->link;
			return End;
		}
		if (r->s2) {
			j->end = r->s2;
			return Branch;
		}
		if (modifiescpsr(r->prog)) {
			j->end = r->s1;
			return Setcond;
		}
		r = r->s1;
	} while (j->len < 4);
	j->end = r;
	return Toolong;
}

static Flow*
successor(Flow *r)
{
	if(r->s1)
		return r->s1;
	else
		return r->s2;
}

static void
applypred(Flow *rstart, Joininfo *j, int cond, int branch)
{
	int pred;
	Flow *r;

	if(j->len == 0)
		return;
	if(cond == Truecond)
		pred = predinfo[rstart->prog->as - ABEQ].scond;
	else
		pred = predinfo[rstart->prog->as - ABEQ].notscond;

	for(r = j->start;; r = successor(r)) {
		if(r->prog->as == AB) {
			if(r != j->last || branch == Delbranch)
				excise(r);
			else {
				if(cond == Truecond)
					r->prog->as = predinfo[rstart->prog->as - ABEQ].opcode;
				else
					r->prog->as = predinfo[rstart->prog->as - ABEQ].notopcode;
			}
		}
		else
		if(predicable(r->prog))
			r->prog->scond = (r->prog->scond&~C_SCOND)|pred;
		if(r->s1 != r->link) {
			r->s1 = r->link;
			r->link->p1 = r;
		}
		if(r == j->last)
			break;
	}
}

void
predicate(Graph *g)
{
	Flow *r;
	int t1, t2;
	Joininfo j1, j2;

	for(r=g->start; r!=nil; r=r->link) {
		if (isbranch(r->prog)) {
			t1 = joinsplit(r->s1, &j1);
			t2 = joinsplit(r->s2, &j2);
			if(j1.last->link != j2.start)
				continue;
			if(j1.end == j2.end)
			if((t1 == Branch && (t2 == Join || t2 == Setcond)) ||
			   (t2 == Join && (t1 == Join || t1 == Setcond))) {
				applypred(r, &j1, Falsecond, Delbranch);
				applypred(r, &j2, Truecond, Delbranch);
				excise(r);
				continue;
			}
			if(t1 == End || t1 == Branch) {
				applypred(r, &j1, Falsecond, Keepbranch);
				excise(r);
				continue;
			}
		}
	}
}

static int
isdconst(Addr *a)
{
	return a->type == TYPE_CONST;
}

static int
isfloatreg(Addr *a)
{
	return REG_F0 <= a->reg && a->reg <= REG_F15;
}

int
stackaddr(Addr *a)
{
	return regtyp(a) && a->reg == REGSP;
}

int
smallindir(Addr *a, Addr *reg)
{
	return reg->type == TYPE_REG && a->type == TYPE_MEM &&
		a->reg == reg->reg &&
		0 <= a->offset && a->offset < 4096;
}
