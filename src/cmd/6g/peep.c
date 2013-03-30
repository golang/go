// Derived from Inferno utils/6c/peep.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/peep.c
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

static void	conprop(Reg *r);
static void elimshortmov(Reg *r);
static int prevl(Reg *r, int reg);
static void pushback(Reg *r);
static int regconsttyp(Adr*);

// do we need the carry bit
static int
needc(Prog *p)
{
	while(p != P) {
		switch(p->as) {
		case AADCL:
		case AADCQ:
		case ASBBL:
		case ASBBQ:
		case ARCRB:
		case ARCRW:
		case ARCRL:
		case ARCRQ:
			return 1;
		case AADDB:
		case AADDW:
		case AADDL:
		case AADDQ:
		case ASUBB:
		case ASUBW:
		case ASUBL:
		case ASUBQ:
		case AJMP:
		case ARET:
		case ACALL:
			return 0;
		default:
			if(p->to.type == D_BRANCH)
				return 0;
		}
		p = p->link;
	}
	return 0;
}

static Reg*
rnops(Reg *r)
{
	Prog *p;
	Reg *r1;

	if(r != R)
	for(;;) {
		p = r->prog;
		if(p->as != ANOP || p->from.type != D_NONE || p->to.type != D_NONE)
			break;
		r1 = uniqs(r);
		if(r1 == R)
			break;
		r = r1;
	}
	return r;
}

void
peep(void)
{
	Reg *r, *r1, *r2;
	Prog *p, *p1;
	int t;

	/*
	 * complete R structure
	 */
	t = 0;
	for(r=firstr; r!=R; r=r1) {
		r1 = r->link;
		if(r1 == R)
			break;
		p = r->prog->link;
		while(p != r1->prog)
		switch(p->as) {
		default:
			r2 = rega();
			r->link = r2;
			r2->link = r1;

			r2->prog = p;
			p->reg = r2;

			r2->p1 = r;
			r->s1 = r2;
			r2->s1 = r1;
			r1->p1 = r2;

			r = r2;
			t++;

		case ADATA:
		case AGLOBL:
		case ANAME:
		case ASIGNAME:
		case ALOCALS:
		case ATYPE:
			p = p->link;
		}
	}
	
	// byte, word arithmetic elimination.
	elimshortmov(r);

	// constant propagation
	// find MOV $con,R followed by
	// another MOV $con,R without
	// setting R in the interim
	for(r=firstr; r!=R; r=r->link) {
		p = r->prog;
		switch(p->as) {
		case ALEAL:
		case ALEAQ:
			if(regtyp(&p->to))
			if(p->from.sym != S)
				conprop(r);
			break;

		case AMOVB:
		case AMOVW:
		case AMOVL:
		case AMOVQ:
		case AMOVSS:
		case AMOVSD:
			if(regtyp(&p->to))
			if(p->from.type == D_CONST)
				conprop(r);
			break;
		}
	}

loop1:
	if(debug['P'] && debug['v'])
		dumpit("loop1", firstr);

	t = 0;
	for(r=firstr; r!=R; r=r->link) {
		p = r->prog;
		switch(p->as) {
		case AMOVL:
		case AMOVQ:
		case AMOVSS:
		case AMOVSD:
			if(regtyp(&p->to))
			if(regtyp(&p->from)) {
				if(copyprop(r)) {
					excise(r);
					t++;
				} else
				if(subprop(r) && copyprop(r)) {
					excise(r);
					t++;
				}
			}
			break;

		case AMOVBLZX:
		case AMOVWLZX:
		case AMOVBLSX:
		case AMOVWLSX:
			if(regtyp(&p->to)) {
				r1 = rnops(uniqs(r));
				if(r1 != R) {
					p1 = r1->prog;
					if(p->as == p1->as && p->to.type == p1->from.type){
						p1->as = AMOVL;
						t++;
					}
				}
			}
			break;

		case AMOVBQSX:
		case AMOVBQZX:
		case AMOVWQSX:
		case AMOVWQZX:
		case AMOVLQSX:
		case AMOVLQZX:
		case AMOVQL:
			if(regtyp(&p->to)) {
				r1 = rnops(uniqs(r));
				if(r1 != R) {
					p1 = r1->prog;
					if(p->as == p1->as && p->to.type == p1->from.type){
						p1->as = AMOVQ;
						t++;
					}
				}
			}
			break;

		case AADDL:
		case AADDQ:
		case AADDW:
			if(p->from.type != D_CONST || needc(p->link))
				break;
			if(p->from.offset == -1){
				if(p->as == AADDQ)
					p->as = ADECQ;
				else
				if(p->as == AADDL)
					p->as = ADECL;
				else
					p->as = ADECW;
				p->from = zprog.from;
				break;
			}
			if(p->from.offset == 1){
				if(p->as == AADDQ)
					p->as = AINCQ;
				else if(p->as == AADDL)
					p->as = AINCL;
				else
					p->as = AINCW;
				p->from = zprog.from;
				break;
			}
			break;

		case ASUBL:
		case ASUBQ:
		case ASUBW:
			if(p->from.type != D_CONST || needc(p->link))
				break;
			if(p->from.offset == -1) {
				if(p->as == ASUBQ)
					p->as = AINCQ;
				else
				if(p->as == ASUBL)
					p->as = AINCL;
				else
					p->as = AINCW;
				p->from = zprog.from;
				break;
			}
			if(p->from.offset == 1){
				if(p->as == ASUBQ)
					p->as = ADECQ;
				else
				if(p->as == ASUBL)
					p->as = ADECL;
				else
					p->as = ADECW;
				p->from = zprog.from;
				break;
			}
			break;
		}
	}
	if(t)
		goto loop1;

	// MOVLQZX removal.
	// The MOVLQZX exists to avoid being confused for a
	// MOVL that is just copying 32-bit data around during
	// copyprop.  Now that copyprop is done, remov MOVLQZX R1, R2
	// if it is dominated by an earlier ADDL/MOVL/etc into R1 that
	// will have already cleared the high bits.
	//
	// MOVSD removal.
	// We never use packed registers, so a MOVSD between registers
	// can be replaced by MOVAPD, which moves the pair of float64s
	// instead of just the lower one.  We only use the lower one, but
	// the processor can do better if we do moves using both.
	for(r=firstr; r!=R; r=r->link) {
		p = r->prog;
		if(p->as == AMOVLQZX)
		if(regtyp(&p->from))
		if(p->from.type == p->to.type)
		if(prevl(r, p->from.type))
			excise(r);
		
		if(p->as == AMOVSD)
		if(regtyp(&p->from))
		if(regtyp(&p->to))
			p->as = AMOVAPD;
	}

	// load pipelining
	// push any load from memory as early as possible
	// to give it time to complete before use.
	for(r=firstr; r!=R; r=r->link) {
		p = r->prog;
		switch(p->as) {
		case AMOVB:
		case AMOVW:
		case AMOVL:
		case AMOVQ:
		case AMOVLQZX:
			if(regtyp(&p->to) && !regconsttyp(&p->from))
				pushback(r);
		}
	}
}

static void
pushback(Reg *r0)
{
	Reg *r, *b;
	Prog *p0, *p, t;
	
	b = R;
	p0 = r0->prog;
	for(r=uniqp(r0); r!=R && uniqs(r)!=R; r=uniqp(r)) {
		p = r->prog;
		if(p->as != ANOP) {
			if(!regconsttyp(&p->from) || !regtyp(&p->to))
				break;
			if(copyu(p, &p0->to, A) || copyu(p0, &p->to, A))
				break;
		}
		if(p->as == ACALL)
			break;
		b = r;
	}
	
	if(b == R) {
		if(debug['v']) {
			print("no pushback: %P\n", r0->prog);
			if(r)
				print("\t%P [%d]\n", r->prog, uniqs(r)!=R);
		}
		return;
	}

	if(debug['v']) {
		print("pushback\n");
		for(r=b;; r=r->link) {
			print("\t%P\n", r->prog);
			if(r == r0)
				break;
		}
	}

	t = *r0->prog;
	for(r=uniqp(r0);; r=uniqp(r)) {
		p0 = r->link->prog;
		p = r->prog;
		p0->as = p->as;
		p0->lineno = p->lineno;
		p0->from = p->from;
		p0->to = p->to;

		if(r == b)
			break;
	}
	p0 = r->prog;
	p0->as = t.as;
	p0->lineno = t.lineno;
	p0->from = t.from;
	p0->to = t.to;

	if(debug['v']) {
		print("\tafter\n");
		for(r=b;; r=r->link) {
			print("\t%P\n", r->prog);
			if(r == r0)
				break;
		}
	}
}

void
excise(Reg *r)
{
	Prog *p;

	p = r->prog;
	if(debug['P'] && debug['v'])
		print("%P ===delete===\n", p);

	p->as = ANOP;
	p->from = zprog.from;
	p->to = zprog.to;

	ostats.ndelmov++;
}

Reg*
uniqp(Reg *r)
{
	Reg *r1;

	r1 = r->p1;
	if(r1 == R) {
		r1 = r->p2;
		if(r1 == R || r1->p2link != R)
			return R;
	} else
		if(r->p2 != R)
			return R;
	return r1;
}

Reg*
uniqs(Reg *r)
{
	Reg *r1;

	r1 = r->s1;
	if(r1 == R) {
		r1 = r->s2;
		if(r1 == R)
			return R;
	} else
		if(r->s2 != R)
			return R;
	return r1;
}

int
regtyp(Adr *a)
{
	int t;

	t = a->type;
	if(t >= D_AX && t <= D_R15)
		return 1;
	if(t >= D_X0 && t <= D_X0+15)
		return 1;
	return 0;
}

// movb elimination.
// movb is simulated by the linker
// when a register other than ax, bx, cx, dx
// is used, so rewrite to other instructions
// when possible.  a movb into a register
// can smash the entire 32-bit register without
// causing any trouble.
static void
elimshortmov(Reg *r)
{
	Prog *p;

	USED(r);
	for(r=firstr; r!=R; r=r->link) {
		p = r->prog;
		if(regtyp(&p->to)) {
			switch(p->as) {
			case AINCB:
			case AINCW:
				p->as = AINCQ;
				break;
			case ADECB:
			case ADECW:
				p->as = ADECQ;
				break;
			case ANEGB:
			case ANEGW:
				p->as = ANEGQ;
				break;
			case ANOTB:
			case ANOTW:
				p->as = ANOTQ;
				break;
			}
			if(regtyp(&p->from) || p->from.type == D_CONST) {
				// move or artihmetic into partial register.
				// from another register or constant can be movl.
				// we don't switch to 64-bit arithmetic if it can
				// change how the carry bit is set (and the carry bit is needed).
				switch(p->as) {
				case AMOVB:
				case AMOVW:
					p->as = AMOVQ;
					break;
				case AADDB:
				case AADDW:
					if(!needc(p->link))
						p->as = AADDQ;
					break;
				case ASUBB:
				case ASUBW:
					if(!needc(p->link))
						p->as = ASUBQ;
					break;
				case AMULB:
				case AMULW:
					p->as = AMULQ;
					break;
				case AIMULB:
				case AIMULW:
					p->as = AIMULQ;
					break;
				case AANDB:
				case AANDW:
					p->as = AANDQ;
					break;
				case AORB:
				case AORW:
					p->as = AORQ;
					break;
				case AXORB:
				case AXORW:
					p->as = AXORQ;
					break;
				case ASHLB:
				case ASHLW:
					p->as = ASHLQ;
					break;
				}
			} else if(p->from.type >= D_NONE) {
				// explicit zero extension, but don't
				// do that if source is a byte register
				// (only AH can occur and it's forbidden).
				switch(p->as) {
				case AMOVB:
					p->as = AMOVBQZX;
					break;
				case AMOVW:
					p->as = AMOVWQZX;
					break;
				}
			}
		}
	}
}

static int
regconsttyp(Adr *a)
{
	if(regtyp(a))
		return 1;
	switch(a->type) {
	case D_CONST:
	case D_FCONST:
	case D_SCONST:
	case D_ADDR:
		return 1;
	}
	return 0;
}

// is reg guaranteed to be truncated by a previous L instruction?
static int
prevl(Reg *r0, int reg)
{
	Prog *p;
	Reg *r;

	for(r=uniqp(r0); r!=R; r=uniqp(r)) {
		p = r->prog;
		if(p->to.type == reg) {
			switch(p->as) {
			case AADDL:
			case AANDL:
			case ADECL:
			case ADIVL:
			case AIDIVL:
			case AIMULL:
			case AINCL:
			case AMOVL:
			case AMULL:
			case AORL:
			case ARCLL:
			case ARCRL:
			case AROLL:
			case ARORL:
			case ASALL:
			case ASARL:
			case ASHLL:
			case ASHRL:
			case ASUBL:
			case AXORL:
				return 1;
			}
			return 0;
		}
	}
	return 0;
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
int
subprop(Reg *r0)
{
	Prog *p;
	Adr *v1, *v2;
	Reg *r;
	int t;

	if(debug['P'] && debug['v'])
		print("subprop %P\n", r0->prog);
	p = r0->prog;
	v1 = &p->from;
	if(!regtyp(v1)) {
		if(debug['P'] && debug['v'])
			print("\tnot regtype %D; return 0\n", v1);
		return 0;
	}
	v2 = &p->to;
	if(!regtyp(v2)) {
		if(debug['P'] && debug['v'])
			print("\tnot regtype %D; return 0\n", v2);
		return 0;
	}
	for(r=uniqp(r0); r!=R; r=uniqp(r)) {
		if(debug['P'] && debug['v'])
			print("\t? %P\n", r->prog);
		if(uniqs(r) == R) {
			if(debug['P'] && debug['v'])
				print("\tno unique successor\n");
			break;
		}
		p = r->prog;
		switch(p->as) {
		case ACALL:
			if(debug['P'] && debug['v'])
				print("\tfound %P; return 0\n", p);
			return 0;

		case AIMULL:
		case AIMULQ:
		case AIMULW:
			if(p->to.type != D_NONE)
				break;
			goto giveup;

		case ARCLB:
		case ARCLL:
		case ARCLQ:
		case ARCLW:
		case ARCRB:
		case ARCRL:
		case ARCRQ:
		case ARCRW:
		case AROLB:
		case AROLL:
		case AROLQ:
		case AROLW:
		case ARORB:
		case ARORL:
		case ARORQ:
		case ARORW:
		case ASALB:
		case ASALL:
		case ASALQ:
		case ASALW:
		case ASARB:
		case ASARL:
		case ASARQ:
		case ASARW:
		case ASHLB:
		case ASHLL:
		case ASHLQ:
		case ASHLW:
		case ASHRB:
		case ASHRL:
		case ASHRQ:
		case ASHRW:
			if(p->from.type == D_CONST)
				break;
			goto giveup;

		case ADIVB:
		case ADIVL:
		case ADIVQ:
		case ADIVW:
		case AIDIVB:
		case AIDIVL:
		case AIDIVQ:
		case AIDIVW:
		case AIMULB:
		case AMULB:
		case AMULL:
		case AMULQ:
		case AMULW:

		case AREP:
		case AREPN:

		case ACWD:
		case ACDQ:
		case ACQO:

		case ASTOSB:
		case ASTOSL:
		case ASTOSQ:
		case AMOVSB:
		case AMOVSL:
		case AMOVSQ:
		giveup:
			if(debug['P'] && debug['v'])
				print("\tfound %P; return 0\n", p);
			return 0;

		case AMOVL:
		case AMOVQ:
		case AMOVSS:
		case AMOVSD:
			if(p->to.type == v1->type)
				goto gotit;
			break;
		}
		if(copyau(&p->from, v2) ||
		   copyau(&p->to, v2)) {
		   	if(debug['P'] && debug['v'])
		   		print("\tcopyau %D failed\n", v2);
			break;
		}
		if(copysub(&p->from, v1, v2, 0) ||
		   copysub(&p->to, v1, v2, 0)) {
		   	if(debug['P'] && debug['v'])
		   		print("\tcopysub failed\n");
			break;
		}
	}
	if(debug['P'] && debug['v'])
		print("\tran off end; return 0\n");
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
		copysub(&p->to, v1, v2, 1);
		if(debug['P'])
			print("%P\n", r->prog);
	}
	t = v1->type;
	v1->type = v2->type;
	v2->type = t;
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
int
copyprop(Reg *r0)
{
	Prog *p;
	Adr *v1, *v2;
	Reg *r;

	if(debug['P'] && debug['v'])
		print("copyprop %P\n", r0->prog);
	p = r0->prog;
	v1 = &p->from;
	v2 = &p->to;
	if(copyas(v1, v2))
		return 1;
	for(r=firstr; r!=R; r=r->link)
		r->active = 0;
	return copy1(v1, v2, r0->s1, 0);
}

int
copy1(Adr *v1, Adr *v2, Reg *r, int f)
{
	int t;
	Prog *p;

	if(r->active) {
		if(debug['P'])
			print("act set; return 1\n");
		return 1;
	}
	r->active = 1;
	if(debug['P'])
		print("copy %D->%D f=%d\n", v1, v2, f);
	for(; r != R; r = r->s1) {
		p = r->prog;
		if(debug['P'])
			print("%P", p);
		if(!f && uniqp(r) == R) {
			f = 1;
			if(debug['P'])
				print("; merge; f=%d", f);
		}
		t = copyu(p, v2, A);
		switch(t) {
		case 2:	/* rar, cant split */
			if(debug['P'])
				print("; %D rar; return 0\n", v2);
			return 0;

		case 3:	/* set */
			if(debug['P'])
				print("; %D set; return 1\n", v2);
			return 1;

		case 1:	/* used, substitute */
		case 4:	/* use and set */
			if(f) {
				if(!debug['P'])
					return 0;
				if(t == 4)
					print("; %D used+set and f=%d; return 0\n", v2, f);
				else
					print("; %D used and f=%d; return 0\n", v2, f);
				return 0;
			}
			if(copyu(p, v2, v1)) {
				if(debug['P'])
					print("; sub fail; return 0\n");
				return 0;
			}
			if(debug['P'])
				print("; sub %D/%D", v2, v1);
			if(t == 4) {
				if(debug['P'])
					print("; %D used+set; return 1\n", v2);
				return 1;
			}
			break;
		}
		if(!f) {
			t = copyu(p, v1, A);
			if(!f && (t == 2 || t == 3 || t == 4)) {
				f = 1;
				if(debug['P'])
					print("; %D set and !f; f=%d", v1, f);
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
		if(debug['P'])
			print("unknown op %A\n", p->as);
		/* SBBL; ADCL; FLD1; SAHF */
		return 2;


	case ANEGB:
	case ANEGW:
	case ANEGL:
	case ANEGQ:
	case ANOTB:
	case ANOTW:
	case ANOTL:
	case ANOTQ:
		if(copyas(&p->to, v))
			return 2;
		break;

	case ALEAL:	/* lhs addr, rhs store */
	case ALEAQ:
		if(copyas(&p->from, v))
			return 2;


	case ANOP:	/* rhs store */
	case AMOVL:
	case AMOVQ:
	case AMOVBLSX:
	case AMOVBLZX:
	case AMOVBQSX:
	case AMOVBQZX:
	case AMOVLQSX:
	case AMOVLQZX:
	case AMOVWLSX:
	case AMOVWLZX:
	case AMOVWQSX:
	case AMOVWQZX:
	case AMOVQL:

	case AMOVSS:
	case AMOVSD:
	case ACVTSD2SL:
	case ACVTSD2SQ:
	case ACVTSD2SS:
	case ACVTSL2SD:
	case ACVTSL2SS:
	case ACVTSQ2SD:
	case ACVTSQ2SS:
	case ACVTSS2SD:
	case ACVTSS2SL:
	case ACVTSS2SQ:
	case ACVTTSD2SL:
	case ACVTTSD2SQ:
	case ACVTTSS2SL:
	case ACVTTSS2SQ:
		if(copyas(&p->to, v)) {
			if(s != A)
				return copysub(&p->from, v, s, 1);
			if(copyau(&p->from, v))
				return 4;
			return 3;
		}
		goto caseread;

	case ARCLB:
	case ARCLL:
	case ARCLQ:
	case ARCLW:
	case ARCRB:
	case ARCRL:
	case ARCRQ:
	case ARCRW:
	case AROLB:
	case AROLL:
	case AROLQ:
	case AROLW:
	case ARORB:
	case ARORL:
	case ARORQ:
	case ARORW:
	case ASALB:
	case ASALL:
	case ASALQ:
	case ASALW:
	case ASARB:
	case ASARL:
	case ASARQ:
	case ASARW:
	case ASHLB:
	case ASHLL:
	case ASHLQ:
	case ASHLW:
	case ASHRB:
	case ASHRL:
	case ASHRQ:
	case ASHRW:
		if(copyas(&p->to, v))
			return 2;
		if(copyas(&p->from, v))
			if(p->from.type == D_CX)
				return 2;
		goto caseread;

	case AADDB:	/* rhs rar */
	case AADDL:
	case AADDQ:
	case AADDW:
	case AANDB:
	case AANDL:
	case AANDQ:
	case AANDW:
	case ADECL:
	case ADECQ:
	case ADECW:
	case AINCL:
	case AINCQ:
	case AINCW:
	case ASUBB:
	case ASUBL:
	case ASUBQ:
	case ASUBW:
	case AORB:
	case AORL:
	case AORQ:
	case AORW:
	case AXORB:
	case AXORL:
	case AXORQ:
	case AXORW:
	case AMOVB:
	case AMOVW:

	case AADDSD:
	case AADDSS:
	case ACMPSD:
	case ACMPSS:
	case ADIVSD:
	case ADIVSS:
	case AMAXSD:
	case AMAXSS:
	case AMINSD:
	case AMINSS:
	case AMULSD:
	case AMULSS:
	case ARCPSS:
	case ARSQRTSS:
	case ASQRTSD:
	case ASQRTSS:
	case ASUBSD:
	case ASUBSS:
	case AXORPD:
		if(copyas(&p->to, v))
			return 2;
		goto caseread;

	case ACMPL:	/* read only */
	case ACMPW:
	case ACMPB:
	case ACMPQ:

	case ACOMISD:
	case ACOMISS:
	case AUCOMISD:
	case AUCOMISS:
	caseread:
		if(s != A) {
			if(copysub(&p->from, v, s, 1))
				return 1;
			return copysub(&p->to, v, s, 1);
		}
		if(copyau(&p->from, v))
			return 1;
		if(copyau(&p->to, v))
			return 1;
		break;

	case AJGE:	/* no reference */
	case AJNE:
	case AJLE:
	case AJEQ:
	case AJHI:
	case AJLS:
	case AJMI:
	case AJPL:
	case AJGT:
	case AJLT:
	case AJCC:
	case AJCS:

	case AADJSP:
	case AWAIT:
	case ACLD:
		break;

	case AIMULL:
	case AIMULQ:
	case AIMULW:
		if(p->to.type != D_NONE) {
			if(copyas(&p->to, v))
				return 2;
			goto caseread;
		}

	case ADIVB:
	case ADIVL:
	case ADIVQ:
	case ADIVW:
	case AIDIVB:
	case AIDIVL:
	case AIDIVQ:
	case AIDIVW:
	case AIMULB:
	case AMULB:
	case AMULL:
	case AMULQ:
	case AMULW:

	case ACWD:
	case ACDQ:
	case ACQO:
		if(v->type == D_AX || v->type == D_DX)
			return 2;
		goto caseread;

	case AREP:
	case AREPN:
		if(v->type == D_CX)
			return 2;
		goto caseread;

	case AMOVSB:
	case AMOVSL:
	case AMOVSQ:
		if(v->type == D_DI || v->type == D_SI)
			return 2;
		goto caseread;

	case ASTOSB:
	case ASTOSL:
	case ASTOSQ:
		if(v->type == D_AX || v->type == D_DI)
			return 2;
		goto caseread;

	case AJMP:	/* funny */
		if(s != A) {
			if(copysub(&p->to, v, s, 1))
				return 1;
			return 0;
		}
		if(copyau(&p->to, v))
			return 1;
		return 0;

	case ARET:	/* funny */
		if(s != A)
			return 1;
		return 3;

	case ACALL:	/* funny */
		if(REGEXT && v->type <= REGEXT && v->type > exregoffset)
			return 2;
		if(REGARG >= 0 && v->type == (uchar)REGARG)
			return 2;
		if(v->type == p->from.type)
			return 2;

		if(s != A) {
			if(copysub(&p->to, v, s, 1))
				return 1;
			return 0;
		}
		if(copyau(&p->to, v))
			return 4;
		return 3;

	case ATEXT:	/* funny */
		if(REGARG >= 0 && v->type == (uchar)REGARG)
			return 3;
		return 0;
	}
	return 0;
}

/*
 * direct reference,
 * could be set/use depending on
 * semantics
 */
int
copyas(Adr *a, Adr *v)
{
	if(a->type != v->type)
		return 0;
	if(regtyp(v))
		return 1;
	if(v->type == D_AUTO || v->type == D_PARAM)
		if(v->offset == a->offset)
			return 1;
	return 0;
}

/*
 * either direct or indirect
 */
int
copyau(Adr *a, Adr *v)
{

	if(copyas(a, v)) {
		if(debug['P'] && debug['v'])
			print("\tcopyau: copyas returned 1\n");
		return 1;
	}
	if(regtyp(v)) {
		if(a->type-D_INDIR == v->type) {
			if(debug['P'] && debug['v'])
				print("\tcopyau: found indir use - return 1\n");
			return 1;
		}
		if(a->index == v->type) {
			if(debug['P'] && debug['v'])
				print("\tcopyau: found index use - return 1\n");
			return 1;
		}
	}
	return 0;
}

/*
 * substitute s for v in a
 * return failure to substitute
 */
int
copysub(Adr *a, Adr *v, Adr *s, int f)
{
	int t;

	if(copyas(a, v)) {
		t = s->type;
		if(t >= D_AX && t <= D_R15 || t >= D_X0 && t <= D_X0+15) {
			if(f)
				a->type = t;
		}
		return 0;
	}
	if(regtyp(v)) {
		t = v->type;
		if(a->type == t+D_INDIR) {
			if((s->type == D_BP || s->type == D_R13) && a->index != D_NONE)
				return 1;	/* can't use BP-base with index */
			if(f)
				a->type = s->type+D_INDIR;
//			return 0;
		}
		if(a->index == t) {
			if(f)
				a->index = s->type;
			return 0;
		}
		return 0;
	}
	return 0;
}

static void
conprop(Reg *r0)
{
	Reg *r;
	Prog *p, *p0;
	int t;
	Adr *v0;

	p0 = r0->prog;
	v0 = &p0->to;
	r = r0;

loop:
	r = uniqs(r);
	if(r == R || r == r0)
		return;
	if(uniqp(r) == R)
		return;

	p = r->prog;
	t = copyu(p, v0, A);
	switch(t) {
	case 0:	// miss
	case 1:	// use
		goto loop;

	case 2:	// rar
	case 4:	// use and set
		break;

	case 3:	// set
		if(p->as == p0->as)
		if(p->from.type == p0->from.type)
		if(p->from.node == p0->from.node)
		if(p->from.offset == p0->from.offset)
		if(p->from.scale == p0->from.scale)
		if(p->from.u.vval == p0->from.u.vval)
		if(p->from.index == p0->from.index) {
			excise(r);
			goto loop;
		}
		break;
	}
}
