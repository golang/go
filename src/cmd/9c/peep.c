// cmd/9c/peep.c from Vita Nuova.
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

#include "gc.h"

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

loop1:
	t = 0;
	for(r=firstr; r!=R; r=r->link) {
		p = r->prog;
		if(p->as == AMOVW || p->as == AMOVD || p->as == AFMOVS || p->as == AFMOVD)
		if(regtyp(&p->to)) {
			if(regtyp(&p->from))
			if(p->from.type == p->to.type) {
				if(copyprop(r)) {
					excise(r);
					t++;
				} else
				if(subprop(r) && copyprop(r)) {
					excise(r);
					t++;
				}
			}
			if(regzer(&p->from))
			if(p->to.type == D_REG) {
				p->from.type = D_REG;
				p->from.reg = REGZERO;
				if(copyprop(r)) {
					excise(r);
					t++;
				} else
				if(subprop(r) && copyprop(r)) {
					excise(r);
					t++;
				}
			}
		}
	}
	if(t)
		goto loop1;
	/*
	 * look for MOVB x,R; MOVB R,R
	 */
	for(r=firstr; r!=R; r=r->link) {
		p = r->prog;
		switch(p->as) {
		default:
			continue;
		case AMOVH:
		case AMOVHZ:
		case AMOVB:
		case AMOVBZ:
		case AMOVW:
		case AMOVWZ:
			if(p->to.type != D_REG)
				continue;
			break;
		}
		r1 = r->link;
		if(r1 == R)
			continue;
		p1 = r1->prog;
		if(p1->as != p->as)
			continue;
		if(p1->from.type != D_REG || p1->from.reg != p->to.reg)
			continue;
		if(p1->to.type != D_REG || p1->to.reg != p->to.reg)
			continue;
		excise(r1);
	}

	if(debug['D'] > 1)
		return;	/* allow following code improvement to be suppressed */

	/*
	 * look for OP x,y,R; CMP R, $0 -> OPCC x,y,R
	 * when OP can set condition codes correctly
	 */
	for(r=firstr; r!=R; r=r->link) {
		p = r->prog;
		switch(p->as) {
		case ACMP:
		case ACMPW:		/* always safe? */
			if(!regzer(&p->to))
				continue;
			r1 = r->s1;
			if(r1 == R)
				continue;
			switch(r1->prog->as) {
			default:
				continue;
			case ABCL:
			case ABC:
				/* the conditions can be complex and these are currently little used */
				continue;
			case ABEQ:
			case ABGE:
			case ABGT:
			case ABLE:
			case ABLT:
			case ABNE:
			case ABVC:
			case ABVS:
				break;
			}
			r1 = r;
			do
				r1 = uniqp(r1);
			while (r1 != R && r1->prog->as == ANOP);
			if(r1 == R)
				continue;
			p1 = r1->prog;
			if(p1->to.type != D_REG || p1->to.reg != p->from.reg)
				continue;
			switch(p1->as) {
			case ASUB:
			case AADD:
			case AXOR:
			case AOR:
				/* irregular instructions */
				if(p1->from.type == D_CONST)
					continue;
				break;
			}
			switch(p1->as) {
			default:
				continue;
			case AMOVW:
			case AMOVD:
				if(p1->from.type != D_REG)
					continue;
				continue;
			case AANDCC:
			case AANDNCC:
			case AORCC:
			case AORNCC:
			case AXORCC:
			case ASUBCC:
			case ASUBECC:
			case ASUBMECC:
			case ASUBZECC:
			case AADDCC:
			case AADDCCC:
			case AADDECC:
			case AADDMECC:
			case AADDZECC:
			case ARLWMICC:
			case ARLWNMCC:
				t = p1->as;
				break;
			/* don't deal with floating point instructions for now */
/*
			case AFABS:	t = AFABSCC; break;
			case AFADD:	t = AFADDCC; break;
			case AFADDS:	t = AFADDSCC; break;
			case AFCTIW:	t = AFCTIWCC; break;
			case AFCTIWZ:	t = AFCTIWZCC; break;
			case AFDIV:	t = AFDIVCC; break;
			case AFDIVS:	t = AFDIVSCC; break;
			case AFMADD:	t = AFMADDCC; break;
			case AFMADDS:	t = AFMADDSCC; break;
			case AFMOVD:	t = AFMOVDCC; break;
			case AFMSUB:	t = AFMSUBCC; break;
			case AFMSUBS:	t = AFMSUBSCC; break;
			case AFMUL:	t = AFMULCC; break;
			case AFMULS:	t = AFMULSCC; break;
			case AFNABS:	t = AFNABSCC; break;
			case AFNEG:	t = AFNEGCC; break;
			case AFNMADD:	t = AFNMADDCC; break;
			case AFNMADDS:	t = AFNMADDSCC; break;
			case AFNMSUB:	t = AFNMSUBCC; break;
			case AFNMSUBS:	t = AFNMSUBSCC; break;
			case AFRSP:	t = AFRSPCC; break;
			case AFSUB:	t = AFSUBCC; break;
			case AFSUBS:	t = AFSUBSCC; break;
			case ACNTLZW:	t = ACNTLZWCC; break;
			case AMTFSB0:	t = AMTFSB0CC; break;
			case AMTFSB1:	t = AMTFSB1CC; break;
*/
			case AADD:	t = AADDCC; break;
			case AADDV:	t = AADDVCC; break;
			case AADDC:	t = AADDCCC; break;
			case AADDCV:	t = AADDCVCC; break;
			case AADDME:	t = AADDMECC; break;
			case AADDMEV:	t = AADDMEVCC; break;
			case AADDE:	t = AADDECC; break;
			case AADDEV:	t = AADDEVCC; break;
			case AADDZE:	t = AADDZECC; break;
			case AADDZEV:	t = AADDZEVCC; break;
			case AAND:	t = AANDCC; break;
			case AANDN:	t = AANDNCC; break;
			case ADIVW:	t = ADIVWCC; break;
			case ADIVWV:	t = ADIVWVCC; break;
			case ADIVWU:	t = ADIVWUCC; break;
			case ADIVWUV:	t = ADIVWUVCC; break;
			case ADIVD:	t = ADIVDCC; break;
			case ADIVDV:	t = ADIVDVCC; break;
			case ADIVDU:	t = ADIVDUCC; break;
			case ADIVDUV:	t = ADIVDUVCC; break;
			case AEQV:	t = AEQVCC; break;
			case AEXTSB:	t = AEXTSBCC; break;
			case AEXTSH:	t = AEXTSHCC; break;
			case AEXTSW:	t = AEXTSWCC; break;
			case AMULHW:	t = AMULHWCC; break;
			case AMULHWU:	t = AMULHWUCC; break;
			case AMULLW:	t = AMULLWCC; break;
			case AMULLWV:	t = AMULLWVCC; break;
			case AMULHD:	t = AMULHDCC; break;
			case AMULHDU:	t = AMULHDUCC; break;
			case AMULLD:	t = AMULLDCC; break;
			case AMULLDV:	t = AMULLDVCC; break;
			case ANAND:	t = ANANDCC; break;
			case ANEG:	t = ANEGCC; break;
			case ANEGV:	t = ANEGVCC; break;
			case ANOR:	t = ANORCC; break;
			case AOR:	t = AORCC; break;
			case AORN:	t = AORNCC; break;
			case AREM:	t = AREMCC; break;
			case AREMV:	t = AREMVCC; break;
			case AREMU:	t = AREMUCC; break;
			case AREMUV:	t = AREMUVCC; break;
			case AREMD:	t = AREMDCC; break;
			case AREMDV:	t = AREMDVCC; break;
			case AREMDU:	t = AREMDUCC; break;
			case AREMDUV:	t = AREMDUVCC; break;
			case ARLWMI:	t = ARLWMICC; break;
			case ARLWNM:	t = ARLWNMCC; break;
			case ASLW:	t = ASLWCC; break;
			case ASRAW:	t = ASRAWCC; break;
			case ASRW:	t = ASRWCC; break;
			case ASLD:	t = ASLDCC; break;
			case ASRAD:	t = ASRADCC; break;
			case ASRD:	t = ASRDCC; break;
			case ASUB:	t = ASUBCC; break;
			case ASUBV:	t = ASUBVCC; break;
			case ASUBC:	t = ASUBCCC; break;
			case ASUBCV:	t = ASUBCVCC; break;
			case ASUBME:	t = ASUBMECC; break;
			case ASUBMEV:	t = ASUBMEVCC; break;
			case ASUBE:	t = ASUBECC; break;
			case ASUBEV:	t = ASUBEVCC; break;
			case ASUBZE:	t = ASUBZECC; break;
			case ASUBZEV:	t = ASUBZEVCC; break;
			case AXOR:	t = AXORCC; break;
				break;
			}
			if(debug['D'])
				print("cmp %P; %P -> ", p1, p);
			p1->as = t;
			if(debug['D'])
				print("%P\n", p1);
			excise(r);
			continue;
		}
	}
}

void
excise(Reg *r)
{
	Prog *p;

	p = r->prog;
	p->as = ANOP;
	p->from = zprog.from;
	p->from3 = zprog.from3;
	p->to = zprog.to;
	p->reg = zprog.reg; /**/
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

/*
 * if the system forces R0 to be zero,
 * convert references to $0 to references to R0.
 */
regzer(Adr *a)
{
	if(R0ISZERO) {
		if(a->type == D_CONST)
			if(a->sym == S)
				if(a->offset == 0)
					return 1;
		if(a->type == D_REG)
			if(a->reg == REGZERO)
				return 1;
	}
	return 0;
}

regtyp(Adr *a)
{

	if(a->type == D_REG) {
		if(!R0ISZERO || a->reg != REGZERO)
			return 1;
		return 0;
	}
	if(a->type == D_FREG)
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
	Adr *v1, *v2;
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
		case ABL:
			return 0;

		case AADD:
		case AADDC:
		case AADDCC:
		case AADDE:
		case AADDECC:
		case ASUB:
		case ASUBCC:
		case ASUBC:
		case ASUBCCC:
		case ASUBE:
		case ASUBECC:
		case ASLW:
		case ASRW:
		case ASRWCC:
		case ASRAW:
		case ASRAWCC:
		case ASLD:
		case ASRD:
		case ASRAD:
		case AOR:
		case AORCC:
		case AORN:
		case AORNCC:
		case AAND:
		case AANDCC:
		case AANDN:
		case AANDNCC:
		case ANAND:
		case ANANDCC:
		case ANOR:
		case ANORCC:
		case AXOR:
		case AXORCC:
		case AMULHW:
		case AMULHWU:
		case AMULLW:
		case AMULLD:
		case ADIVW:
		case ADIVWU:
		case ADIVD:
		case ADIVDU:
		case AREM:
		case AREMU:
		case AREMD:
		case AREMDU:
		case ARLWNM:
		case ARLWNMCC:

		case AFADD:
		case AFADDS:
		case AFSUB:
		case AFSUBS:
		case AFMUL:
		case AFMULS:
		case AFDIV:
		case AFDIVS:
			if(p->to.type == v1->type)
			if(p->to.reg == v1->reg) {
				if(p->reg == NREG)
					p->reg = p->to.reg;
				goto gotit;
			}
			break;

		case AADDME:
		case AADDMECC:
		case AADDZE:
		case AADDZECC:
		case ASUBME:
		case ASUBMECC:
		case ASUBZE:
		case ASUBZECC:
		case ANEG:
		case ANEGCC:
		case AFNEG:
		case AFNEGCC:
		case AFMOVS:
		case AFMOVD:
		case AMOVW:
		case AMOVD:
			if(p->to.type == v1->type)
			if(p->to.reg == v1->reg)
				goto gotit;
			break;
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
int
copyprop(Reg *r0)
{
	Prog *p;
	Adr *v1, *v2;
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
			t = copyu(p, v1, A);
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
			print(" (???)");
		return 2;


	case ANOP:	/* read, write */
	case AMOVH:
	case AMOVHZ:
	case AMOVB:
	case AMOVBZ:
	case AMOVW:
	case AMOVWZ:
	case AMOVD:

	case ANEG:
	case ANEGCC:
	case AADDME:
	case AADDMECC:
	case AADDZE:
	case AADDZECC:
	case ASUBME:
	case ASUBMECC:
	case ASUBZE:
	case ASUBZECC:

	case AFCTIW:
	case AFCTIWZ:
	case AFMOVS:
	case AFMOVD:
	case AFRSP:
	case AFNEG:
	case AFNEGCC:
		if(s != A) {
			if(copysub(&p->from, v, s, 1))
				return 1;
			if(!copyas(&p->to, v))
				if(copysub(&p->to, v, s, 1))
					return 1;
			return 0;
		}
		if(copyas(&p->to, v)) {
			if(copyau(&p->from, v))
				return 4;
			return 3;
		}
		if(copyau(&p->from, v))
			return 1;
		if(copyau(&p->to, v))
			return 1;
		return 0;

	case ARLWMI:	/* read read rar */
	case ARLWMICC:
		if(copyas(&p->to, v))
			return 2;
		/* fall through */

	case AADD:	/* read read write */
	case AADDC:
	case AADDE:
	case ASUB:
	case ASLW:
	case ASRW:
	case ASRAW:
	case ASLD:
	case ASRD:
	case ASRAD:
	case AOR:
	case AORCC:
	case AORN:
	case AORNCC:
	case AAND:
	case AANDCC:
	case AANDN:
	case AANDNCC:
	case ANAND:
	case ANANDCC:
	case ANOR:
	case ANORCC:
	case AXOR:
	case AMULHW:
	case AMULHWU:
	case AMULLW:
	case AMULLD:
	case ADIVW:
	case ADIVD:
	case ADIVWU:
	case ADIVDU:
	case AREM:
	case AREMU:
	case AREMD:
	case AREMDU:
	case ARLWNM:
	case ARLWNMCC:

	case AFADDS:
	case AFADD:
	case AFSUBS:
	case AFSUB:
	case AFMULS:
	case AFMUL:
	case AFDIVS:
	case AFDIV:
		if(s != A) {
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
			if(p->reg == NREG)
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

	case ABEQ:
	case ABGT:
	case ABGE:
	case ABLT:
	case ABLE:
	case ABNE:
	case ABVC:
	case ABVS:
		break;

	case ACMP:	/* read read */
	case ACMPU:
	case ACMPW:
	case ACMPWU:
	case AFCMPO:
	case AFCMPU:
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

	case ABR:	/* funny */
		if(s != A) {
			if(copysub(&p->to, v, s, 1))
				return 1;
			return 0;
		}
		if(copyau(&p->to, v))
			return 1;
		return 0;

	case ARETURN:	/* funny */
		if(v->type == D_REG)
			if(v->reg == REGRET)
				return 2;
		if(v->type == D_FREG)
			if(v->reg == FREGRET)
				return 2;

	case ABL:	/* funny */
		if(v->type == D_REG) {
			if(v->reg <= REGEXT && v->reg > exregoffset)
				return 2;
			if(v->reg == REGARG)
				return 2;
		}
		if(v->type == D_FREG) {
			if(v->reg <= FREGEXT && v->reg > exfregoffset)
				return 2;
		}

		if(s != A) {
			if(copysub(&p->to, v, s, 1))
				return 1;
			return 0;
		}
		if(copyau(&p->to, v))
			return 4;
		return 3;

	case ATEXT:	/* funny */
		if(v->type == D_REG)
			if(v->reg == REGARG)
				return 3;
		return 0;
	}
	return 0;
}

int
a2type(Prog *p)
{

	switch(p->as) {
	case AADD:
	case AADDC:
	case AADDCC:
	case AADDCCC:
	case AADDE:
	case AADDECC:
	case AADDME:
	case AADDMECC:
	case AADDZE:
	case AADDZECC:
	case ASUB:
	case ASUBC:
	case ASUBCC:
	case ASUBCCC:
	case ASUBE:
	case ASUBECC:
	case ASUBME:
	case ASUBMECC:
	case ASUBZE:
	case ASUBZECC:
	case ASLW:
	case ASLWCC:
	case ASRW:
	case ASRWCC:
	case ASRAW:
	case ASRAWCC:
	case ASLD:
	case ASLDCC:
	case ASRD:
	case ASRDCC:
	case ASRAD:
	case ASRADCC:
	case AOR:
	case AORCC:
	case AORN:
	case AORNCC:
	case AAND:
	case AANDCC:
	case AANDN:
	case AANDNCC:
	case AXOR:
	case AXORCC:
	case ANEG:
	case ANEGCC:
	case AMULHW:
	case AMULHWU:
	case AMULLW:
	case AMULLWCC:
	case ADIVW:
	case ADIVWCC:
	case ADIVWU:
	case ADIVWUCC:
	case AREM:
	case AREMCC:
	case AREMU:
	case AREMUCC:
	case AMULLD:
	case AMULLDCC:
	case ADIVD:
	case ADIVDCC:
	case ADIVDU:
	case ADIVDUCC:
	case AREMD:
	case AREMDCC:
	case AREMDU:
	case AREMDUCC:
	case ANAND:
	case ANANDCC:
	case ANOR:
	case ANORCC:
	case ARLWMI:
	case ARLWMICC:
	case ARLWNM:
	case ARLWNMCC:
		return D_REG;

	case AFADDS:
	case AFADDSCC:
	case AFADD:
	case AFADDCC:
	case AFSUBS:
	case AFSUBSCC:
	case AFSUB:
	case AFSUBCC:
	case AFMULS:
	case AFMULSCC:
	case AFMUL:
	case AFMULCC:
	case AFDIVS:
	case AFDIVSCC:
	case AFDIV:
	case AFDIVCC:
	case AFNEG:
	case AFNEGCC:
		return D_FREG;
	}
	return D_NONE;
}

/*
 * direct reference,
 * could be set/use depending on
 * semantics
 */
int
copyas(Adr *a, Adr *v)
{

	if(regtyp(v))
		if(a->type == v->type)
		if(a->reg == v->reg)
			return 1;
	return 0;
}

/*
 * either direct or indirect
 */
int
copyau(Adr *a, Adr *v)
{

	if(copyas(a, v))
		return 1;
	if(v->type == D_REG)
		if(a->type == D_OREG)
			if(v->reg == a->reg)
				return 1;
	return 0;
}

int
copyau1(Prog *p, Adr *v)
{

	if(regtyp(v))
		if(p->from.type == v->type || p->to.type == v->type)
		if(p->reg == v->reg) {
			if(a2type(p) != v->type)
				print("botch a2type %P\n", p);
			return 1;
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

	if(f)
	if(copyau(a, v))
		a->reg = s->reg;
	return 0;
}

int
copysub1(Prog *p1, Adr *v, Adr *s, int f)
{

	if(f)
	if(copyau1(p1, v))
		p1->reg = s->reg;
	return 0;
}
