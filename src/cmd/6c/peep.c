// Inferno utils/6c/peep.c
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

#include "gc.h"

static int
needc(Prog *p)
{
	while(p != P) {
		switch(p->as) {
		case AADCL:
		case AADCQ:
		case ASBBL:
		case ASBBQ:
		case ARCRL:
		case ARCRQ:
			return 1;
		case AADDL:
		case AADDQ:
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
	for(;;){
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
			p = p->link;
		}
	}

	pc = 0;	/* speculating it won't kill */

loop1:

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
				else if(p->as == AADDL)
					p->as = ADECL;
				else
					p->as = ADECW;
				p->from = zprog.from;
			}
			else if(p->from.offset == 1){
				if(p->as == AADDQ)
					p->as = AINCQ;
				else if(p->as == AADDL)
					p->as = AINCL;
				else
					p->as = AINCW;
				p->from = zprog.from;
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
				else if(p->as == ASUBL)
					p->as = AINCL;
				else
					p->as = AINCW;
				p->from = zprog.from;
			}
			else if(p->from.offset == 1){
				if(p->as == ASUBQ)
					p->as = ADECQ;
				else if(p->as == ASUBL)
					p->as = ADECL;
				else
					p->as = ADECW;
				p->from = zprog.from;
			}
			break;
		}
	}
	if(t)
		goto loop1;
}

void
excise(Reg *r)
{
	Prog *p;

	p = r->prog;
	p->as = ANOP;
	p->from = zprog.from;
	p->to = zprog.to;
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
regtyp(Addr *a)
{
	int t;

	t = a->type;
	if(t >= D_AX && t <= D_R15)
		return 1;
	if(t >= D_X0 && t <= D_X0+15)
		return 1;
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
	Addr *v1, *v2;
	Reg *r;
	int t;

	p = r0->prog;
	v1 = &p->from;
	if(!regtyp(v1))
		return 0;
	v2 = &p->to;
	if(!regtyp(v2))
		return 0;
	for(r=uniqp(r0); r!=R; r=uniqp(r)) {
		if(uniqs(r) == R)
			break;
		p = r->prog;
		switch(p->as) {
		case ACALL:
			return 0;

		case AIMULL:
		case AIMULQ:
		case AIMULW:
			if(p->to.type != D_NONE)
				break;
			goto giveup;

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
		case AMOVQL:
		giveup:
			return 0;

		case AMOVL:
		case AMOVQ:
			if(p->to.type == v1->type)
				goto gotit;
			break;
		}
		if(copyau(&p->from, v2) ||
		   copyau(&p->to, v2))
			break;
		if(copysub(&p->from, v1, v2, 0) ||
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
	Addr *v1, *v2;
	Reg *r;

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
copy1(Addr *v1, Addr *v2, Reg *r, int f)
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
		case 2:	/* rar, can't split */
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
copyu(Prog *p, Addr *v, Addr *s)
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

	case APREFETCHT0:
	case APREFETCHT1:
	case APREFETCHT2:
	case APREFETCHNTA:

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
		if(v->type == REGRET || v->type == FREGRET)
			return 2;
		if(s != A)
			return 1;
		return 3;

	case ACALL:	/* funny */
		if(REGARG >= 0 && v->type == (uchar)REGARG)
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
copyas(Addr *a, Addr *v)
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
copyau(Addr *a, Addr *v)
{

	if(copyas(a, v))
		return 1;
	if(regtyp(v)) {
		if(a->type-D_INDIR == v->type)
			return 1;
		if(a->index == v->type)
			return 1;
	}
	return 0;
}

/*
 * substitute s for v in a
 * return failure to substitute
 */
int
copysub(Addr *a, Addr *v, Addr *s, int f)
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
