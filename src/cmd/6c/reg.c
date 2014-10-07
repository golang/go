// Inferno utils/6c/reg.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/reg.c
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

static	void	fixjmp(Reg*);

Reg*
rega(void)
{
	Reg *r;

	r = freer;
	if(r == R) {
		r = alloc(sizeof(*r));
	} else
		freer = r->link;

	*r = zreg;
	return r;
}

int
rcmp(const void *a1, const void *a2)
{
	Rgn *p1, *p2;
	int c1, c2;

	p1 = (Rgn*)a1;
	p2 = (Rgn*)a2;
	c1 = p2->cost;
	c2 = p1->cost;
	if(c1 -= c2)
		return c1;
	return p2->varno - p1->varno;
}

void
regopt(Prog *p)
{
	Reg *r, *r1, *r2;
	Prog *p1;
	int i, z;
	int32 initpc, val, npc;
	uint32 vreg;
	Bits bit;
	struct
	{
		int32	m;
		int32	c;
		Reg*	p;
	} log5[6], *lp;

	firstr = R;
	lastr = R;
	nvar = 0;
	regbits = RtoB(D_SP) | RtoB(D_AX) | RtoB(D_X0);
	for(z=0; z<BITS; z++) {
		externs.b[z] = 0;
		params.b[z] = 0;
		consts.b[z] = 0;
		addrs.b[z] = 0;
	}

	/*
	 * pass 1
	 * build aux data structure
	 * allocate pcs
	 * find use and set of variables
	 */
	val = 5L * 5L * 5L * 5L * 5L;
	lp = log5;
	for(i=0; i<5; i++) {
		lp->m = val;
		lp->c = 0;
		lp->p = R;
		val /= 5L;
		lp++;
	}
	val = 0;
	for(; p != P; p = p->link) {
		switch(p->as) {
		case ADATA:
		case AGLOBL:
		case ANAME:
		case ASIGNAME:
		case AFUNCDATA:
			continue;
		}
		r = rega();
		if(firstr == R) {
			firstr = r;
			lastr = r;
		} else {
			lastr->link = r;
			r->p1 = lastr;
			lastr->s1 = r;
			lastr = r;
		}
		r->prog = p;
		r->pc = val;
		val++;

		lp = log5;
		for(i=0; i<5; i++) {
			lp->c--;
			if(lp->c <= 0) {
				lp->c = lp->m;
				if(lp->p != R)
					lp->p->log5 = r;
				lp->p = r;
				(lp+1)->c = 0;
				break;
			}
			lp++;
		}

		r1 = r->p1;
		if(r1 != R)
		switch(r1->prog->as) {
		case ARET:
		case AJMP:
		case AIRETL:
		case AIRETQ:
			r->p1 = R;
			r1->s1 = R;
		}

		bit = mkvar(r, &p->from);
		if(bany(&bit))
		switch(p->as) {
		/*
		 * funny
		 */
		case ALEAL:
		case ALEAQ:
			for(z=0; z<BITS; z++)
				addrs.b[z] |= bit.b[z];
			break;

		/*
		 * left side read
		 */
		default:
			for(z=0; z<BITS; z++)
				r->use1.b[z] |= bit.b[z];
			break;
		}

		bit = mkvar(r, &p->to);
		if(bany(&bit))
		switch(p->as) {
		default:
			diag(Z, "reg: unknown op: %A", p->as);
			break;

		/*
		 * right side read
		 */
		case ACMPB:
		case ACMPL:
		case ACMPQ:
		case ACMPW:
		case APREFETCHT0:
		case APREFETCHT1:
		case APREFETCHT2:
		case APREFETCHNTA:
		case ACOMISS:
		case ACOMISD:
		case AUCOMISS:
		case AUCOMISD:
			for(z=0; z<BITS; z++)
				r->use2.b[z] |= bit.b[z];
			break;

		/*
		 * right side write
		 */
		case ANOP:
		case AMOVL:
		case AMOVQ:
		case AMOVB:
		case AMOVW:
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
			for(z=0; z<BITS; z++)
				r->set.b[z] |= bit.b[z];
			break;

		/*
		 * right side read+write
		 */
		case AADDB:
		case AADDL:
		case AADDQ:
		case AADDW:
		case AANDB:
		case AANDL:
		case AANDQ:
		case AANDW:
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
		case ASALB:
		case ASALL:
		case ASALQ:
		case ASALW:
		case ASARB:
		case ASARL:
		case ASARQ:
		case ASARW:
		case AROLB:
		case AROLL:
		case AROLQ:
		case AROLW:
		case ARORB:
		case ARORL:
		case ARORQ:
		case ARORW:
		case ASHLB:
		case ASHLL:
		case ASHLQ:
		case ASHLW:
		case ASHRB:
		case ASHRL:
		case ASHRQ:
		case ASHRW:
		case AIMULL:
		case AIMULQ:
		case AIMULW:
		case ANEGL:
		case ANEGQ:
		case ANOTL:
		case ANOTQ:
		case AADCL:
		case AADCQ:
		case ASBBL:
		case ASBBQ:

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
			for(z=0; z<BITS; z++) {
				r->set.b[z] |= bit.b[z];
				r->use2.b[z] |= bit.b[z];
			}
			break;

		/*
		 * funny
		 */
		case ACALL:
			for(z=0; z<BITS; z++)
				addrs.b[z] |= bit.b[z];
			break;
		}

		switch(p->as) {
		case AIMULL:
		case AIMULQ:
		case AIMULW:
			if(p->to.type != D_NONE)
				break;

		case AIDIVB:
		case AIDIVL:
		case AIDIVQ:
		case AIDIVW:
		case AIMULB:
		case ADIVB:
		case ADIVL:
		case ADIVQ:
		case ADIVW:
		case AMULB:
		case AMULL:
		case AMULQ:
		case AMULW:

		case ACWD:
		case ACDQ:
		case ACQO:
			r->regu |= RtoB(D_AX) | RtoB(D_DX);
			break;

		case AREP:
		case AREPN:
		case ALOOP:
		case ALOOPEQ:
		case ALOOPNE:
			r->regu |= RtoB(D_CX);
			break;

		case AMOVSB:
		case AMOVSL:
		case AMOVSQ:
		case AMOVSW:
		case ACMPSB:
		case ACMPSL:
		case ACMPSQ:
		case ACMPSW:
			r->regu |= RtoB(D_SI) | RtoB(D_DI);
			break;

		case ASTOSB:
		case ASTOSL:
		case ASTOSQ:
		case ASTOSW:
		case ASCASB:
		case ASCASL:
		case ASCASQ:
		case ASCASW:
			r->regu |= RtoB(D_AX) | RtoB(D_DI);
			break;

		case AINSB:
		case AINSL:
		case AINSW:
		case AOUTSB:
		case AOUTSL:
		case AOUTSW:
			r->regu |= RtoB(D_DI) | RtoB(D_DX);
			break;
		}
	}
	if(firstr == R)
		return;
	initpc = pc - val;
	npc = val;

	/*
	 * pass 2
	 * turn branch references to pointers
	 * build back pointers
	 */
	for(r = firstr; r != R; r = r->link) {
		p = r->prog;
		if(p->to.type == D_BRANCH) {
			val = p->to.offset - initpc;
			r1 = firstr;
			while(r1 != R) {
				r2 = r1->log5;
				if(r2 != R && val >= r2->pc) {
					r1 = r2;
					continue;
				}
				if(r1->pc == val)
					break;
				r1 = r1->link;
			}
			if(r1 == R) {
				nearln = p->lineno;
				diag(Z, "ref not found\n%P", p);
				continue;
			}
			if(r1 == r) {
				nearln = p->lineno;
				diag(Z, "ref to self\n%P", p);
				continue;
			}
			r->s2 = r1;
			r->p2link = r1->p2;
			r1->p2 = r;
		}
	}
	if(debug['R']) {
		p = firstr->prog;
		print("\n%L %D\n", p->lineno, &p->from);
	}

	/*
	 * pass 2.1
	 * fix jumps
	 */
	fixjmp(firstr);

	/*
	 * pass 2.5
	 * find looping structure
	 */
	for(r = firstr; r != R; r = r->link)
		r->active = 0;
	change = 0;
	loopit(firstr, npc);
	if(debug['R'] && debug['v']) {
		print("\nlooping structure:\n");
		for(r = firstr; r != R; r = r->link) {
			print("%d:%P", r->loop, r->prog);
			for(z=0; z<BITS; z++)
				bit.b[z] = r->use1.b[z] |
					   r->use2.b[z] |
					   r->set.b[z];
			if(bany(&bit)) {
				print("\t");
				if(bany(&r->use1))
					print(" u1=%B", r->use1);
				if(bany(&r->use2))
					print(" u2=%B", r->use2);
				if(bany(&r->set))
					print(" st=%B", r->set);
			}
			print("\n");
		}
	}

	/*
	 * pass 3
	 * iterate propagating usage
	 * 	back until flow graph is complete
	 */
loop1:
	change = 0;
	for(r = firstr; r != R; r = r->link)
		r->active = 0;
	for(r = firstr; r != R; r = r->link)
		if(r->prog->as == ARET)
			prop(r, zbits, zbits);
loop11:
	/* pick up unreachable code */
	i = 0;
	for(r = firstr; r != R; r = r1) {
		r1 = r->link;
		if(r1 && r1->active && !r->active) {
			prop(r, zbits, zbits);
			i = 1;
		}
	}
	if(i)
		goto loop11;
	if(change)
		goto loop1;


	/*
	 * pass 4
	 * iterate propagating register/variable synchrony
	 * 	forward until graph is complete
	 */
loop2:
	change = 0;
	for(r = firstr; r != R; r = r->link)
		r->active = 0;
	synch(firstr, zbits);
	if(change)
		goto loop2;


	/*
	 * pass 5
	 * isolate regions
	 * calculate costs (paint1)
	 */
	r = firstr;
	if(r) {
		for(z=0; z<BITS; z++)
			bit.b[z] = (r->refahead.b[z] | r->calahead.b[z]) &
			  ~(externs.b[z] | params.b[z] | addrs.b[z] | consts.b[z]);
		if(bany(&bit)) {
			nearln = r->prog->lineno;
			warn(Z, "used and not set: %B", bit);
			if(debug['R'] && !debug['w'])
				print("used and not set: %B\n", bit);
		}
	}
	if(debug['R'] && debug['v'])
		print("\nprop structure:\n");
	for(r = firstr; r != R; r = r->link)
		r->act = zbits;
	rgp = region;
	nregion = 0;
	for(r = firstr; r != R; r = r->link) {
		if(debug['R'] && debug['v']) {
			print("%P\t", r->prog);
			if(bany(&r->set))
				print("s:%B ", r->set);
			if(bany(&r->refahead))
				print("ra:%B ", r->refahead);
			if(bany(&r->calahead))
				print("ca:%B ", r->calahead);
			print("\n");
		}
		for(z=0; z<BITS; z++)
			bit.b[z] = r->set.b[z] &
			  ~(r->refahead.b[z] | r->calahead.b[z] | addrs.b[z]);
		if(bany(&bit)) {
			nearln = r->prog->lineno;
			warn(Z, "set and not used: %B", bit);
			if(debug['R'])
				print("set and not used: %B\n", bit);
			excise(r);
		}
		for(z=0; z<BITS; z++)
			bit.b[z] = LOAD(r) & ~(r->act.b[z] | addrs.b[z]);
		while(bany(&bit)) {
			i = bnum(bit);
			rgp->enter = r;
			rgp->varno = i;
			change = 0;
			if(debug['R'] && debug['v'])
				print("\n");
			paint1(r, i);
			bit.b[i/32] &= ~(1L<<(i%32));
			if(change <= 0) {
				if(debug['R'])
					print("%L$%d: %B\n",
						r->prog->lineno, change, blsh(i));
				continue;
			}
			rgp->cost = change;
			nregion++;
			if(nregion >= NRGN)
				fatal(Z, "too many regions");
			rgp++;
		}
	}
	qsort(region, nregion, sizeof(region[0]), rcmp);

	/*
	 * pass 6
	 * determine used registers (paint2)
	 * replace code (paint3)
	 */
	rgp = region;
	for(i=0; i<nregion; i++) {
		bit = blsh(rgp->varno);
		vreg = paint2(rgp->enter, rgp->varno);
		vreg = allreg(vreg, rgp);
		if(debug['R']) {
			print("%L$%d %R: %B\n",
				rgp->enter->prog->lineno,
				rgp->cost,
				rgp->regno,
				bit);
		}
		if(rgp->regno != 0)
			paint3(rgp->enter, rgp->varno, vreg, rgp->regno);
		rgp++;
	}
	/*
	 * pass 7
	 * peep-hole on basic block
	 */
	if(!debug['R'] || debug['P'])
		peep();

	/*
	 * pass 8
	 * recalculate pc
	 */
	val = initpc;
	for(r = firstr; r != R; r = r1) {
		r->pc = val;
		p = r->prog;
		p1 = P;
		r1 = r->link;
		if(r1 != R)
			p1 = r1->prog;
		for(; p != p1; p = p->link) {
			switch(p->as) {
			default:
				val++;
				break;

			case ANOP:
			case ADATA:
			case AGLOBL:
			case ANAME:
			case ASIGNAME:
			case AFUNCDATA:
				break;
			}
		}
	}
	pc = val;

	/*
	 * fix up branches
	 */
	if(debug['R'])
		if(bany(&addrs))
			print("addrs: %B\n", addrs);

	r1 = 0; /* set */
	for(r = firstr; r != R; r = r->link) {
		p = r->prog;
		if(p->to.type == D_BRANCH) {
			p->to.offset = r->s2->pc;
			p->to.u.branch = r->s2->prog;
		}
		r1 = r;
	}

	/*
	 * last pass
	 * eliminate nops
	 * free aux structures
	 */
	for(p = firstr->prog; p != P; p = p->link){
		while(p->link && p->link->as == ANOP)
			p->link = p->link->link;
	}
	if(r1 != R) {
		r1->link = freer;
		freer = firstr;
	}
}

/*
 * add mov b,rn
 * just after r
 */
void
addmove(Reg *r, int bn, int rn, int f)
{
	Prog *p, *p1;
	Addr *a;
	Var *v;

	p1 = alloc(sizeof(*p1));
	*p1 = zprog;
	p = r->prog;

	p1->link = p->link;
	p->link = p1;
	p1->lineno = p->lineno;

	v = var + bn;

	a = &p1->to;
	a->sym = v->sym;
	a->offset = v->offset;
	a->etype = v->etype;
	a->type = v->name;

	p1->as = AMOVL;
	if(v->etype == TCHAR || v->etype == TUCHAR)
		p1->as = AMOVB;
	if(v->etype == TSHORT || v->etype == TUSHORT)
		p1->as = AMOVW;
	if(v->etype == TVLONG || v->etype == TUVLONG || (v->etype == TIND && ewidth[TIND] == 8))
		p1->as = AMOVQ;
	if(v->etype == TFLOAT)
		p1->as = AMOVSS;
	if(v->etype == TDOUBLE)
		p1->as = AMOVSD;

	p1->from.type = rn;
	if(!f) {
		p1->from = *a;
		*a = zprog.from;
		a->type = rn;
		if(v->etype == TUCHAR)
			p1->as = AMOVB;
		if(v->etype == TUSHORT)
			p1->as = AMOVW;
	}
	if(debug['R'])
		print("%P\t.a%P\n", p, p1);
}

uint32
doregbits(int r)
{
	uint32 b;

	b = 0;
	if(r >= D_INDIR)
		r -= D_INDIR;
	if(r >= D_AX && r <= D_R15)
		b |= RtoB(r);
	else
	if(r >= D_AL && r <= D_R15B)
		b |= RtoB(r-D_AL+D_AX);
	else
	if(r >= D_AH && r <= D_BH)
		b |= RtoB(r-D_AH+D_AX);
	else
	if(r >= D_X0 && r <= D_X0+15)
		b |= FtoB(r);
	return b;
}

Bits
mkvar(Reg *r, Addr *a)
{
	Var *v;
	int i, t, n, et, z;
	int32 o;
	Bits bit;
	LSym *s;

	/*
	 * mark registers used
	 */
	t = a->type;
	r->regu |= doregbits(t);
	r->regu |= doregbits(a->index);

	switch(t) {
	default:
		goto none;
	case D_ADDR:
		a->type = a->index;
		bit = mkvar(r, a);
		for(z=0; z<BITS; z++)
			addrs.b[z] |= bit.b[z];
		a->type = t;
		goto none;
	case D_EXTERN:
	case D_STATIC:
	case D_PARAM:
	case D_AUTO:
		n = t;
		break;
	}
	s = a->sym;
	if(s == nil)
		goto none;
	if(s->name[0] == '.')
		goto none;
	et = a->etype;
	o = a->offset;
	v = var;
	for(i=0; i<nvar; i++) {
		if(s == v->sym)
		if(n == v->name)
		if(o == v->offset)
			goto out;
		v++;
	}
	if(nvar >= NVAR)
		fatal(Z, "variable not optimized: %s", s->name);
	i = nvar;
	nvar++;
	v = &var[i];
	v->sym = s;
	v->offset = o;
	v->name = n;
	v->etype = et;
	if(debug['R'])
		print("bit=%2d et=%2d %D\n", i, et, a);

out:
	bit = blsh(i);
	if(n == D_EXTERN || n == D_STATIC)
		for(z=0; z<BITS; z++)
			externs.b[z] |= bit.b[z];
	if(n == D_PARAM)
		for(z=0; z<BITS; z++)
			params.b[z] |= bit.b[z];
	if(v->etype != et || !(typechlpfd[et] || typev[et]))	/* funny punning */
		for(z=0; z<BITS; z++)
			addrs.b[z] |= bit.b[z];
	return bit;

none:
	return zbits;
}

void
prop(Reg *r, Bits ref, Bits cal)
{
	Reg *r1, *r2;
	int z;

	for(r1 = r; r1 != R; r1 = r1->p1) {
		for(z=0; z<BITS; z++) {
			ref.b[z] |= r1->refahead.b[z];
			if(ref.b[z] != r1->refahead.b[z]) {
				r1->refahead.b[z] = ref.b[z];
				change++;
			}
			cal.b[z] |= r1->calahead.b[z];
			if(cal.b[z] != r1->calahead.b[z]) {
				r1->calahead.b[z] = cal.b[z];
				change++;
			}
		}
		switch(r1->prog->as) {
		case ACALL:
			for(z=0; z<BITS; z++) {
				cal.b[z] |= ref.b[z] | externs.b[z];
				ref.b[z] = 0;
			}
			break;

		case ATEXT:
			for(z=0; z<BITS; z++) {
				cal.b[z] = 0;
				ref.b[z] = 0;
			}
			break;

		case ARET:
			for(z=0; z<BITS; z++) {
				cal.b[z] = externs.b[z];
				ref.b[z] = 0;
			}
		}
		for(z=0; z<BITS; z++) {
			ref.b[z] = (ref.b[z] & ~r1->set.b[z]) |
				r1->use1.b[z] | r1->use2.b[z];
			cal.b[z] &= ~(r1->set.b[z] | r1->use1.b[z] | r1->use2.b[z]);
			r1->refbehind.b[z] = ref.b[z];
			r1->calbehind.b[z] = cal.b[z];
		}
		if(r1->active)
			break;
		r1->active = 1;
	}
	for(; r != r1; r = r->p1)
		for(r2 = r->p2; r2 != R; r2 = r2->p2link)
			prop(r2, r->refbehind, r->calbehind);
}

/*
 * find looping structure
 *
 * 1) find reverse postordering
 * 2) find approximate dominators,
 *	the actual dominators if the flow graph is reducible
 *	otherwise, dominators plus some other non-dominators.
 *	See Matthew S. Hecht and Jeffrey D. Ullman,
 *	"Analysis of a Simple Algorithm for Global Data Flow Problems",
 *	Conf.  Record of ACM Symp. on Principles of Prog. Langs, Boston, Massachusetts,
 *	Oct. 1-3, 1973, pp.  207-217.
 * 3) find all nodes with a predecessor dominated by the current node.
 *	such a node is a loop head.
 *	recursively, all preds with a greater rpo number are in the loop
 */
int32
postorder(Reg *r, Reg **rpo2r, int32 n)
{
	Reg *r1;

	r->rpo = 1;
	r1 = r->s1;
	if(r1 && !r1->rpo)
		n = postorder(r1, rpo2r, n);
	r1 = r->s2;
	if(r1 && !r1->rpo)
		n = postorder(r1, rpo2r, n);
	rpo2r[n] = r;
	n++;
	return n;
}

int32
rpolca(int32 *idom, int32 rpo1, int32 rpo2)
{
	int32 t;

	if(rpo1 == -1)
		return rpo2;
	while(rpo1 != rpo2){
		if(rpo1 > rpo2){
			t = rpo2;
			rpo2 = rpo1;
			rpo1 = t;
		}
		while(rpo1 < rpo2){
			t = idom[rpo2];
			if(t >= rpo2)
				fatal(Z, "bad idom");
			rpo2 = t;
		}
	}
	return rpo1;
}

int
doms(int32 *idom, int32 r, int32 s)
{
	while(s > r)
		s = idom[s];
	return s == r;
}

int
loophead(int32 *idom, Reg *r)
{
	int32 src;

	src = r->rpo;
	if(r->p1 != R && doms(idom, src, r->p1->rpo))
		return 1;
	for(r = r->p2; r != R; r = r->p2link)
		if(doms(idom, src, r->rpo))
			return 1;
	return 0;
}

void
loopmark(Reg **rpo2r, int32 head, Reg *r)
{
	if(r->rpo < head || r->active == head)
		return;
	r->active = head;
	r->loop += LOOP;
	if(r->p1 != R)
		loopmark(rpo2r, head, r->p1);
	for(r = r->p2; r != R; r = r->p2link)
		loopmark(rpo2r, head, r);
}

void
loopit(Reg *r, int32 nr)
{
	Reg *r1;
	int32 i, d, me;

	if(nr > maxnr) {
		rpo2r = alloc(nr * sizeof(Reg*));
		idom = alloc(nr * sizeof(int32));
		maxnr = nr;
	}

	d = postorder(r, rpo2r, 0);
	if(d > nr)
		fatal(Z, "too many reg nodes");
	nr = d;
	for(i = 0; i < nr / 2; i++){
		r1 = rpo2r[i];
		rpo2r[i] = rpo2r[nr - 1 - i];
		rpo2r[nr - 1 - i] = r1;
	}
	for(i = 0; i < nr; i++)
		rpo2r[i]->rpo = i;

	idom[0] = 0;
	for(i = 0; i < nr; i++){
		r1 = rpo2r[i];
		me = r1->rpo;
		d = -1;
		if(r1->p1 != R && r1->p1->rpo < me)
			d = r1->p1->rpo;
		for(r1 = r1->p2; r1 != nil; r1 = r1->p2link)
			if(r1->rpo < me)
				d = rpolca(idom, d, r1->rpo);
		idom[i] = d;
	}

	for(i = 0; i < nr; i++){
		r1 = rpo2r[i];
		r1->loop++;
		if(r1->p2 != R && loophead(idom, r1))
			loopmark(rpo2r, i, r1);
	}
}

void
synch(Reg *r, Bits dif)
{
	Reg *r1;
	int z;

	for(r1 = r; r1 != R; r1 = r1->s1) {
		for(z=0; z<BITS; z++) {
			dif.b[z] = (dif.b[z] &
				~(~r1->refbehind.b[z] & r1->refahead.b[z])) |
					r1->set.b[z] | r1->regdiff.b[z];
			if(dif.b[z] != r1->regdiff.b[z]) {
				r1->regdiff.b[z] = dif.b[z];
				change++;
			}
		}
		if(r1->active)
			break;
		r1->active = 1;
		for(z=0; z<BITS; z++)
			dif.b[z] &= ~(~r1->calbehind.b[z] & r1->calahead.b[z]);
		if(r1->s2 != R)
			synch(r1->s2, dif);
	}
}

uint32
allreg(uint32 b, Rgn *r)
{
	Var *v;
	int i;

	v = var + r->varno;
	r->regno = 0;
	switch(v->etype) {

	default:
		diag(Z, "unknown etype %d/%d", bitno(b), v->etype);
		break;

	case TCHAR:
	case TUCHAR:
	case TSHORT:
	case TUSHORT:
	case TINT:
	case TUINT:
	case TLONG:
	case TULONG:
	case TVLONG:
	case TUVLONG:
	case TIND:
	case TARRAY:
		i = BtoR(~b);
		if(i && r->cost > 0) {
			r->regno = i;
			return RtoB(i);
		}
		break;

	case TDOUBLE:
	case TFLOAT:
		i = BtoF(~b);
		if(i && r->cost > 0) {
			r->regno = i;
			return FtoB(i);
		}
		break;
	}
	return 0;
}

void
paint1(Reg *r, int bn)
{
	Reg *r1;
	Prog *p;
	int z;
	uint32 bb;

	z = bn/32;
	bb = 1L<<(bn%32);
	if(r->act.b[z] & bb)
		return;
	for(;;) {
		if(!(r->refbehind.b[z] & bb))
			break;
		r1 = r->p1;
		if(r1 == R)
			break;
		if(!(r1->refahead.b[z] & bb))
			break;
		if(r1->act.b[z] & bb)
			break;
		r = r1;
	}

	if(LOAD(r) & ~(r->set.b[z]&~(r->use1.b[z]|r->use2.b[z])) & bb) {
		change -= CLOAD * r->loop;
		if(debug['R'] && debug['v'])
			print("%d%P\td %B $%d\n", r->loop,
				r->prog, blsh(bn), change);
	}
	for(;;) {
		r->act.b[z] |= bb;
		p = r->prog;

		if(r->use1.b[z] & bb) {
			change += CREF * r->loop;
			if(debug['R'] && debug['v'])
				print("%d%P\tu1 %B $%d\n", r->loop,
					p, blsh(bn), change);
		}

		if((r->use2.b[z]|r->set.b[z]) & bb) {
			change += CREF * r->loop;
			if(debug['R'] && debug['v'])
				print("%d%P\tu2 %B $%d\n", r->loop,
					p, blsh(bn), change);
		}

		if(STORE(r) & r->regdiff.b[z] & bb) {
			change -= CLOAD * r->loop;
			if(debug['R'] && debug['v'])
				print("%d%P\tst %B $%d\n", r->loop,
					p, blsh(bn), change);
		}

		if(r->refbehind.b[z] & bb)
			for(r1 = r->p2; r1 != R; r1 = r1->p2link)
				if(r1->refahead.b[z] & bb)
					paint1(r1, bn);

		if(!(r->refahead.b[z] & bb))
			break;
		r1 = r->s2;
		if(r1 != R)
			if(r1->refbehind.b[z] & bb)
				paint1(r1, bn);
		r = r->s1;
		if(r == R)
			break;
		if(r->act.b[z] & bb)
			break;
		if(!(r->refbehind.b[z] & bb))
			break;
	}
}

uint32
regset(Reg *r, uint32 bb)
{
	uint32 b, set;
	Addr v;
	int c;

	set = 0;
	v = zprog.from;
	while(b = bb & ~(bb-1)) {
		v.type = b & 0xFFFF? BtoR(b): BtoF(b);
		if(v.type == 0)
			diag(Z, "zero v.type for %#ux", b);
		c = copyu(r->prog, &v, A);
		if(c == 3)
			set |= b;
		bb &= ~b;
	}
	return set;
}

uint32
reguse(Reg *r, uint32 bb)
{
	uint32 b, set;
	Addr v;
	int c;

	set = 0;
	v = zprog.from;
	while(b = bb & ~(bb-1)) {
		v.type = b & 0xFFFF? BtoR(b): BtoF(b);
		c = copyu(r->prog, &v, A);
		if(c == 1 || c == 2 || c == 4)
			set |= b;
		bb &= ~b;
	}
	return set;
}

uint32
paint2(Reg *r, int bn)
{
	Reg *r1;
	int z;
	uint32 bb, vreg, x;

	z = bn/32;
	bb = 1L << (bn%32);
	vreg = regbits;
	if(!(r->act.b[z] & bb))
		return vreg;
	for(;;) {
		if(!(r->refbehind.b[z] & bb))
			break;
		r1 = r->p1;
		if(r1 == R)
			break;
		if(!(r1->refahead.b[z] & bb))
			break;
		if(!(r1->act.b[z] & bb))
			break;
		r = r1;
	}
	for(;;) {
		r->act.b[z] &= ~bb;

		vreg |= r->regu;

		if(r->refbehind.b[z] & bb)
			for(r1 = r->p2; r1 != R; r1 = r1->p2link)
				if(r1->refahead.b[z] & bb)
					vreg |= paint2(r1, bn);

		if(!(r->refahead.b[z] & bb))
			break;
		r1 = r->s2;
		if(r1 != R)
			if(r1->refbehind.b[z] & bb)
				vreg |= paint2(r1, bn);
		r = r->s1;
		if(r == R)
			break;
		if(!(r->act.b[z] & bb))
			break;
		if(!(r->refbehind.b[z] & bb))
			break;
	}

	bb = vreg;
	for(; r; r=r->s1) {
		x = r->regu & ~bb;
		if(x) {
			vreg |= reguse(r, x);
			bb |= regset(r, x);
		}
	}
	return vreg;
}

void
paint3(Reg *r, int bn, int32 rb, int rn)
{
	Reg *r1;
	Prog *p;
	int z;
	uint32 bb;

	z = bn/32;
	bb = 1L << (bn%32);
	if(r->act.b[z] & bb)
		return;
	for(;;) {
		if(!(r->refbehind.b[z] & bb))
			break;
		r1 = r->p1;
		if(r1 == R)
			break;
		if(!(r1->refahead.b[z] & bb))
			break;
		if(r1->act.b[z] & bb)
			break;
		r = r1;
	}

	if(LOAD(r) & ~(r->set.b[z] & ~(r->use1.b[z]|r->use2.b[z])) & bb)
		addmove(r, bn, rn, 0);
	for(;;) {
		r->act.b[z] |= bb;
		p = r->prog;

		if(r->use1.b[z] & bb) {
			if(debug['R'])
				print("%P", p);
			addreg(&p->from, rn);
			if(debug['R'])
				print("\t.c%P\n", p);
		}
		if((r->use2.b[z]|r->set.b[z]) & bb) {
			if(debug['R'])
				print("%P", p);
			addreg(&p->to, rn);
			if(debug['R'])
				print("\t.c%P\n", p);
		}

		if(STORE(r) & r->regdiff.b[z] & bb)
			addmove(r, bn, rn, 1);
		r->regu |= rb;

		if(r->refbehind.b[z] & bb)
			for(r1 = r->p2; r1 != R; r1 = r1->p2link)
				if(r1->refahead.b[z] & bb)
					paint3(r1, bn, rb, rn);

		if(!(r->refahead.b[z] & bb))
			break;
		r1 = r->s2;
		if(r1 != R)
			if(r1->refbehind.b[z] & bb)
				paint3(r1, bn, rb, rn);
		r = r->s1;
		if(r == R)
			break;
		if(r->act.b[z] & bb)
			break;
		if(!(r->refbehind.b[z] & bb))
			break;
	}
}

void
addreg(Addr *a, int rn)
{

	a->sym = 0;
	a->offset = 0;
	a->type = rn;
}

int32
RtoB(int r)
{

	if(r < D_AX || r > D_R15)
		return 0;
	return 1L << (r-D_AX);
}

int
BtoR(int32 b)
{

	b &= 0xffffL;
	if(nacl)
		b &= ~((1<<(D_BP-D_AX)) | (1<<(D_R15-D_AX)));
	if(b == 0)
		return 0;
	return bitno(b) + D_AX;
}

/*
 *	bit	reg
 *	16	X5
 *	17	X6
 *	18	X7
 */
int32
FtoB(int f)
{
	if(f < FREGMIN || f > FREGEXT)
		return 0;
	return 1L << (f - FREGMIN + 16);
}

int
BtoF(int32 b)
{

	b &= 0x70000L;
	if(b == 0)
		return 0;
	return bitno(b) - 16 + FREGMIN;
}

/* what instruction does a JMP to p eventually land on? */
static Reg*
chasejmp(Reg *r, int *jmploop)
{
	int n;

	n = 0;
	for(; r; r=r->s2) {
		if(r->prog->as != AJMP || r->prog->to.type != D_BRANCH)
			break;
		if(++n > 10) {
			*jmploop = 1;
			break;
		}
	}
	return r;
}

/* mark all code reachable from firstp as alive */
static void
mark(Reg *firstr)
{
	Reg *r;
	Prog *p;

	for(r=firstr; r; r=r->link) {
		if(r->active)
			break;
		r->active = 1;
		p = r->prog;
		if(p->as != ACALL && p->to.type == D_BRANCH)
			mark(r->s2);
		if(p->as == AJMP || p->as == ARET || p->as == AUNDEF)
			break;
	}
}

/*
 * the code generator depends on being able to write out JMP
 * instructions that it can jump to now but fill in later.
 * the linker will resolve them nicely, but they make the code
 * longer and more difficult to follow during debugging.
 * remove them.
 */
static void
fixjmp(Reg *firstr)
{
	int jmploop;
	Reg *r;
	Prog *p;

	if(debug['R'] && debug['v'])
		print("\nfixjmp\n");

	// pass 1: resolve jump to AJMP, mark all code as dead.
	jmploop = 0;
	for(r=firstr; r; r=r->link) {
		p = r->prog;
		if(debug['R'] && debug['v'])
			print("%04d %P\n", (int)r->pc, p);
		if(p->as != ACALL && p->to.type == D_BRANCH && r->s2 && r->s2->prog->as == AJMP) {
			r->s2 = chasejmp(r->s2, &jmploop);
			p->to.offset = r->s2->pc;
			p->to.u.branch = r->s2->prog;
			if(debug['R'] && debug['v'])
				print("->%P\n", p);
		}
		r->active = 0;
	}
	if(debug['R'] && debug['v'])
		print("\n");

	// pass 2: mark all reachable code alive
	mark(firstr);

	// pass 3: delete dead code (mostly JMPs).
	for(r=firstr; r; r=r->link) {
		if(!r->active) {
			p = r->prog;
			if(p->link == P && p->as == ARET && r->p1 && r->p1->prog->as != ARET) {
				// This is the final ARET, and the code so far doesn't have one.
				// Let it stay.
			} else {
				if(debug['R'] && debug['v'])
					print("del %04d %P\n", (int)r->pc, p);
				p->as = ANOP;
			}
		}
	}

	// pass 4: elide JMP to next instruction.
	// only safe if there are no jumps to JMPs anymore.
	if(!jmploop) {
		for(r=firstr; r; r=r->link) {
			p = r->prog;
			if(p->as == AJMP && p->to.type == D_BRANCH && r->s2 == r->link) {
				if(debug['R'] && debug['v'])
					print("del %04d %P\n", (int)r->pc, p);
				p->as = ANOP;
			}
		}
	}

	// fix back pointers.
	for(r=firstr; r; r=r->link) {
		r->p2 = R;
		r->p2link = R;
	}
	for(r=firstr; r; r=r->link) {
		if(r->s2) {
			r->p2link = r->s2->p2;
			r->s2->p2 = r;
		}
	}

	if(debug['R'] && debug['v']) {
		print("\n");
		for(r=firstr; r; r=r->link)
			print("%04d %P\n", (int)r->pc, r->prog);
		print("\n");
	}
}

