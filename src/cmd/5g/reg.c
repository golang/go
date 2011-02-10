// Inferno utils/5c/reg.c
// http://code.google.com/p/inferno-os/source/browse/utils/5g/reg.c
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


#include "gg.h"
#include "opt.h"

#define	P2R(p)	(Reg*)(p->reg)

	void	addsplits(void);
	int	noreturn(Prog *p);
static	int	first	= 0;

Reg*
rega(void)
{
	Reg *r;

	r = freer;
	if(r == R) {
		r = mal(sizeof(*r));
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

static void
setoutvar(void)
{
	Type *t;
	Node *n;
	Addr a;
	Iter save;
	Bits bit;
	int z;

	t = structfirst(&save, getoutarg(curfn->type));
	while(t != T) {
		n = nodarg(t, 1);
		a = zprog.from;
		naddr(n, &a, 0);
		bit = mkvar(R, &a);
		for(z=0; z<BITS; z++)
			ovar.b[z] |= bit.b[z];
		t = structnext(&save);
	}
//if(bany(b))
//print("ovars = %Q\n", &ovar);
}

void
excise(Reg *r)
{
	Prog *p;

	p = r->prog;
	p->as = ANOP;
	p->scond = zprog.scond;
	p->from = zprog.from;
	p->to = zprog.to;
	p->reg = zprog.reg;
}

static void
setaddrs(Bits bit)
{
	int i, n;
	Var *v;
	Sym *s;

	while(bany(&bit)) {
		// convert each bit to a variable
		i = bnum(bit);
		s = var[i].sym;
		n = var[i].name;
		bit.b[i/32] &= ~(1L<<(i%32));

		// disable all pieces of that variable
		for(i=0; i<nvar; i++) {
			v = var+i;
			if(v->sym == s && v->name == n)
				v->addr = 2;
		}
	}
}

void
regopt(Prog *firstp)
{
	Reg *r, *r1;
	Prog *p;
	int i, z, nr;
	uint32 vreg;
	Bits bit;

	if(first == 0) {
		fmtinstall('Q', Qconv);
	}

	first++;
	if(debug['K']) {
		if(first != 13)
			return;
//		debug['R'] = 2;
//		debug['P'] = 2;
		print("optimizing %S\n", curfn->nname->sym);
	}

	// count instructions
	nr = 0;
	for(p=firstp; p!=P; p=p->link)
		nr++;

	// if too big dont bother
	if(nr >= 10000) {
//		print("********** %S is too big (%d)\n", curfn->nname->sym, nr);
		return;
	}

	r1 = R;
	firstr = R;
	lastr = R;
	nvar = 0;
	regbits = RtoB(REGSP)|RtoB(REGLINK)|RtoB(REGPC);
	for(z=0; z<BITS; z++) {
		externs.b[z] = 0;
		params.b[z] = 0;
		consts.b[z] = 0;
		addrs.b[z] = 0;
		ovar.b[z] = 0;
	}

	// build list of return variables
	setoutvar();

	/*
	 * pass 1
	 * build aux data structure
	 * allocate pcs
	 * find use and set of variables
	 */
	nr = 0;
	for(p=firstp; p != P; p = p->link) {
		switch(p->as) {
		case ADATA:
		case AGLOBL:
		case ANAME:
		case ASIGNAME:
			continue;
		}
		r = rega();
		nr++;
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
		p->regp = r;

		r1 = r->p1;
		if(r1 != R) {
			switch(r1->prog->as) {
			case ARET:
			case AB:
			case ARFE:
				r->p1 = R;
				r1->s1 = R;
			}
		}

		/*
		 * left side always read
		 */
		bit = mkvar(r, &p->from);
		for(z=0; z<BITS; z++)
			r->use1.b[z] |= bit.b[z];

		/*
		 * right side depends on opcode
		 */
		bit = mkvar(r, &p->to);
		if(bany(&bit))
		switch(p->as) {
		default:
			yyerror("reg: unknown op: %A", p->as);
			break;

		/*
		 * right side write
		 */
		case ANOP:
		case AMOVB:
		case AMOVBU:
		case AMOVH:
		case AMOVHU:
		case AMOVW:
		case AMOVF:
		case AMOVD:
			for(z=0; z<BITS; z++)
				r->set.b[z] |= bit.b[z];
			break;

		/*
		 * funny
		 */
		case ABL:
			setaddrs(bit);
			break;
		}

		if(p->as == AMOVM) {
			z = p->to.offset;
			if(p->from.type == D_CONST)
				z = p->from.offset;
			for(i=0; z; i++) {
				if(z&1)
					regbits |= RtoB(i);
				z >>= 1;
			}
		}
	}
	if(firstr == R)
		return;

	for(i=0; i<nvar; i++) {
		Var *v = var+i;
		if(v->addr) {
			bit = blsh(i);
			for(z=0; z<BITS; z++)
				addrs.b[z] |= bit.b[z];
		}

//		print("bit=%2d addr=%d et=%-6E w=%-2d s=%S + %lld\n",
//			i, v->addr, v->etype, v->width, v->sym, v->offset);
	}

	/*
	 * pass 2
	 * turn branch references to pointers
	 * build back pointers
	 */
	for(r=firstr; r!=R; r=r->link) {
		p = r->prog;
		if(p->to.type == D_BRANCH) {
			if(p->to.branch == P)
				fatal("pnil %P", p);
			r1 = p->to.branch->regp;
			if(r1 == R)
				fatal("rnil %P", p);
			if(r1 == r) {
				//fatal("ref to self %P", p);
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
		print("	addr = %Q\n", addrs);
	}

	/*
	 * pass 2.5
	 * find looping structure
	 */
	for(r = firstr; r != R; r = r->link)
		r->active = 0;
	change = 0;
	loopit(firstr, nr);

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

	addsplits();

	if(debug['R'] > 1) {
		print("\nprop structure:\n");
		for(r = firstr; r != R; r = r->link) {
			print("%d:%P", r->loop, r->prog);
			for(z=0; z<BITS; z++) {
				bit.b[z] = r->set.b[z] |
					r->refahead.b[z] | r->calahead.b[z] |
					r->refbehind.b[z] | r->calbehind.b[z] |
					r->use1.b[z] | r->use2.b[z];
				bit.b[z] &= ~addrs.b[z];
			}

			if(bany(&bit)) {
				print("\t");
				if(bany(&r->use1))
					print(" u1=%Q", r->use1);
				if(bany(&r->use2))
					print(" u2=%Q", r->use2);
				if(bany(&r->set))
					print(" st=%Q", r->set);
				if(bany(&r->refahead))
					print(" ra=%Q", r->refahead);
				if(bany(&r->calahead))
					print(" ca=%Q", r->calahead);
				if(bany(&r->refbehind))
					print(" rb=%Q", r->refbehind);
				if(bany(&r->calbehind))
					print(" cb=%Q", r->calbehind);
			}
			print("\n");
		}
	}

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
		if(bany(&bit) & !r->refset) {
			// should never happen - all variables are preset
			if(debug['w'])
				print("%L: used and not set: %Q\n", r->prog->lineno, bit);
			r->refset = 1;
		}
	}

	for(r = firstr; r != R; r = r->link)
		r->act = zbits;
	rgp = region;
	nregion = 0;
	for(r = firstr; r != R; r = r->link) {
		for(z=0; z<BITS; z++)
			bit.b[z] = r->set.b[z] &
			  ~(r->refahead.b[z] | r->calahead.b[z] | addrs.b[z]);
		if(bany(&bit) && !r->refset) {
			if(debug['w'])
				print("%L: set and not used: %Q\n", r->prog->lineno, bit);
			r->refset = 1;
			excise(r);
		}
		for(z=0; z<BITS; z++)
			bit.b[z] = LOAD(r) & ~(r->act.b[z] | addrs.b[z]);
		while(bany(&bit)) {
			i = bnum(bit);
			rgp->enter = r;
			rgp->varno = i;
			change = 0;
			if(debug['R'] > 1)
				print("\n");
			paint1(r, i);
			bit.b[i/32] &= ~(1L<<(i%32));
			if(change <= 0) {
				if(debug['R'])
					print("%L $%d: %Q\n",
						r->prog->lineno, change, blsh(i));
				continue;
			}
			rgp->cost = change;
			nregion++;
			if(nregion >= NRGN) {
				if(debug['R'] > 1)
					print("too many regions\n");
				goto brk;
			}
			rgp++;
		}
	}
brk:
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
			if(rgp->regno >= NREG)
				print("%L $%d F%d: %Q\n",
					rgp->enter->prog->lineno,
					rgp->cost,
					rgp->regno-NREG,
					bit);
			else
				print("%L $%d R%d: %Q\n",
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
	if(!debug['R'] || debug['P']) {
		peep();
	}

	/*
	 * last pass
	 * eliminate nops
	 * free aux structures
	 * adjust the stack pointer
	 *	MOVW.W 	R1,-12(R13)			<<- start
	 *	MOVW   	R0,R1
	 *	MOVW   	R1,8(R13)
	 *	MOVW   	$0,R1
	 *	MOVW   	R1,4(R13)
	 *	BL     	,runtime.newproc+0(SB)
	 *	MOVW   	&ft+-32(SP),R7			<<- adjust
	 *	MOVW   	&j+-40(SP),R6			<<- adjust
	 *	MOVW   	autotmp_0003+-24(SP),R5		<<- adjust
	 *	MOVW   	$12(R13),R13			<<- finish
	 */
	vreg = 0;
	for(p = firstp; p != P; p = p->link) {
		while(p->link != P && p->link->as == ANOP)
			p->link = p->link->link;
		if(p->to.type == D_BRANCH)
			while(p->to.branch != P && p->to.branch->as == ANOP)
				p->to.branch = p->to.branch->link;
		if(p->as == AMOVW && p->to.reg == 13) {
			if(p->scond & C_WBIT) {
				vreg = -p->to.offset;		// in adjust region
//				print("%P adjusting %d\n", p, vreg);
				continue;
			}
			if(p->from.type == D_CONST && p->to.type == D_REG) {
				if(p->from.offset != vreg)
					print("in and out different\n");
//				print("%P finish %d\n", p, vreg);
				vreg = 0;	// done adjust region
				continue;
			}

//			print("%P %d %d from type\n", p, p->from.type, D_CONST);
//			print("%P %d %d to type\n\n", p, p->to.type, D_REG);
		}

		if(p->as == AMOVW && vreg != 0) {
			if(p->from.sym != S)
			if(p->from.name == D_AUTO || p->from.name == D_PARAM) {
				p->from.offset += vreg;
//				print("%P adjusting from %d %d\n", p, vreg, p->from.type);
			}
			if(p->to.sym != S)
			if(p->to.name == D_AUTO || p->to.name == D_PARAM) {
				p->to.offset += vreg;
//				print("%P adjusting to %d %d\n", p, vreg, p->from.type);
			}
		}
	}
	if(r1 != R) {
		r1->link = freer;
		freer = firstr;
	}

}

void
addsplits(void)
{
	Reg *r, *r1;
	int z, i;
	Bits bit;

	for(r = firstr; r != R; r = r->link) {
		if(r->loop > 1)
			continue;
		if(r->prog->as == ABL)
			continue;
		for(r1 = r->p2; r1 != R; r1 = r1->p2link) {
			if(r1->loop <= 1)
				continue;
			for(z=0; z<BITS; z++)
				bit.b[z] = r1->calbehind.b[z] &
					(r->refahead.b[z] | r->use1.b[z] | r->use2.b[z]) &
					~(r->calahead.b[z] & addrs.b[z]);
			while(bany(&bit)) {
				i = bnum(bit);
				bit.b[i/32] &= ~(1L << (i%32));
			}
		}
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
	Adr *a;
	Var *v;

	p1 = mal(sizeof(*p1));
	*p1 = zprog;
	p = r->prog;

	p1->link = p->link;
	p->link = p1;
	p1->lineno = p->lineno;

	v = var + bn;

	a = &p1->to;
	a->sym = v->sym;
	a->name = v->name;
	a->offset = v->offset;
	a->etype = v->etype;
	a->type = D_OREG;
	if(a->etype == TARRAY || a->sym == S)
		a->type = D_CONST;

	if(v->addr)
		fatal("addmove: shouldnt be doing this %A\n", a);

	switch(v->etype) {
	default:
		print("What is this %E\n", v->etype);

	case TINT8:
		p1->as = AMOVB;
		break;
	case TBOOL:
	case TUINT8:
		p1->as = AMOVBU;
		break;
	case TINT16:
		p1->as = AMOVH;
		break;
	case TUINT16:
		p1->as = AMOVHU;
		break;
	case TINT32:
	case TUINT32:
	case TPTR32:
		p1->as = AMOVW;
		break;
	case TFLOAT32:
		p1->as = AMOVF;
		break;
	case TFLOAT64:
		p1->as = AMOVD;
		break;
	}

	p1->from.type = D_REG;
	p1->from.reg = rn;
	if(rn >= NREG) {
		p1->from.type = D_FREG;
		p1->from.reg = rn-NREG;
	}
	if(!f) {
		p1->from = *a;
		*a = zprog.from;
		a->type = D_REG;
		a->reg = rn;
		if(rn >= NREG) {
			a->type = D_FREG;
			a->reg = rn-NREG;
		}
		if(v->etype == TUINT8 || v->etype == TBOOL)
			p1->as = AMOVBU;
		if(v->etype == TUINT16)
			p1->as = AMOVHU;
	}
	if(debug['R'])
		print("%P\t.a%P\n", p, p1);
}

static int
overlap(int32 o1, int w1, int32 o2, int w2)
{
	int32 t1, t2;

	t1 = o1+w1;
	t2 = o2+w2;

	if(!(t1 > o2 && t2 > o1))
		return 0;

	return 1;
}

Bits
mkvar(Reg *r, Adr *a)
{
	Var *v;
	int i, t, n, et, z, w, flag;
	int32 o;
	Bits bit;
	Sym *s;

	// mark registers used
	t = a->type;
	n = D_NONE;

	flag = 0;
//	if(a->pun)
//		flag = 1;

	switch(t) {
	default:
		print("type %d %d %D\n", t, a->name, a);
		goto none;

	case D_NONE:
	case D_FCONST:
	case D_BRANCH:
		break;

	case D_CONST:
		flag = 1;
		goto onereg;

	case D_REGREG:
		if(a->offset != NREG)
			r->regu |= RtoB(a->offset);
		goto onereg;

	case D_REG:
	case D_SHIFT:
	case D_OREG:
	onereg:
		if(a->reg != NREG)
			r->regu |= RtoB(a->reg);
		break;

	case D_FREG:
		if(a->reg != NREG)
			r->regu |= FtoB(a->reg);
		break;
	}

	switch(a->name) {
	default:
		goto none;

	case D_EXTERN:
	case D_STATIC:
	case D_AUTO:
	case D_PARAM:
		n = a->name;
		break;
	}

	s = a->sym;
	if(s == S)
		goto none;
	if(s->name[0] == '.')
		goto none;
	et = a->etype;
	o = a->offset;
	w = a->width;

	for(i=0; i<nvar; i++) {
		v = var+i;
		if(v->sym == s && v->name == n) {
			if(v->offset == o)
			if(v->etype == et)
			if(v->width == w)
				if(!flag)
					return blsh(i);

			// if they overlaps, disable both
			if(overlap(v->offset, v->width, o, w)) {
				v->addr = 1;
				flag = 1;
			}
		}
	}

	switch(et) {
	case 0:
	case TFUNC:
	case TARRAY:
	case TSTRING:
		goto none;
	}

	if(nvar >= NVAR) {
		if(debug['w'] > 1 && s)
			fatal("variable not optimized: %D", a);
		goto none;
	}

	i = nvar;
	nvar++;
	v = var+i;
	v->sym = s;
	v->offset = o;
	v->name = n;
//	v->gotype = a->gotype;
	v->etype = et;
	v->width = w;
	v->addr = flag;		// funny punning

	if(debug['R'])
		print("bit=%2d et=%E pun=%d %D\n", i, et, flag, a);

	bit = blsh(i);
	if(n == D_EXTERN || n == D_STATIC)
		for(z=0; z<BITS; z++)
			externs.b[z] |= bit.b[z];
	if(n == D_PARAM)
		for(z=0; z<BITS; z++)
			params.b[z] |= bit.b[z];

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
		case ABL:
			if(noreturn(r1->prog))
				break;
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
				cal.b[z] = externs.b[z] | ovar.b[z];
				ref.b[z] = 0;
			}
			break;
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
				fatal("bad idom");
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
		rpo2r = mal(nr * sizeof(Reg*));
		idom = mal(nr * sizeof(int32));
		maxnr = nr;
	}
	d = postorder(r, rpo2r, 0);
	if(d > nr)
		fatal("too many reg nodes");
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
		fatal("unknown etype %d/%E", bitno(b), v->etype);
		break;

	case TINT8:
	case TUINT8:
	case TINT16:
	case TUINT16:
	case TINT32:
	case TUINT32:
	case TINT:
	case TUINT:
	case TUINTPTR:
	case TBOOL:
	case TPTR32:
		i = BtoR(~b);
		if(i && r->cost >= 0) {
			r->regno = i;
			return RtoB(i);
		}
		break;

	case TFLOAT32:
	case TFLOAT64:
		i = BtoF(~b);
		if(i && r->cost >= 0) {
			r->regno = i+NREG;
			return FtoB(i);
		}
		break;

	case TINT64:
	case TUINT64:
	case TPTR64:
	case TINTER:
	case TSTRUCT:
	case TARRAY:
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

	if(LOAD(r) & ~(r->set.b[z] & ~(r->use1.b[z]|r->use2.b[z])) & bb) {
		change -= CLOAD * r->loop;
		if(debug['R'] > 1)
			print("%d%P\td %Q $%d\n", r->loop,
				r->prog, blsh(bn), change);
	}
	for(;;) {
		r->act.b[z] |= bb;
		p = r->prog;

		if(r->use1.b[z] & bb) {
			change += CREF * r->loop;
			if(debug['R'] > 1)
				print("%d%P\tu1 %Q $%d\n", r->loop,
					p, blsh(bn), change);
		}

		if((r->use2.b[z]|r->set.b[z]) & bb) {
			change += CREF * r->loop;
			if(debug['R'] > 1)
				print("%d%P\tu2 %Q $%d\n", r->loop,
					p, blsh(bn), change);
		}

		if(STORE(r) & r->regdiff.b[z] & bb) {
			change -= CLOAD * r->loop;
			if(debug['R'] > 1)
				print("%d%P\tst %Q $%d\n", r->loop,
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
paint2(Reg *r, int bn)
{
	Reg *r1;
	int z;
	uint32 bb, vreg;

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
addreg(Adr *a, int rn)
{

	if(a->type == D_CONST)
		fatal("addreg: cant do this %D %d\n", a, rn);

	a->sym = 0;
	a->name = D_NONE;
	a->type = D_REG;
	a->reg = rn;
	if(rn >= NREG) {
		a->type = D_FREG;
		a->reg = rn-NREG;
	}
}

/*
 *	bit	reg
 *	0	R0
 *	1	R1
 *	...	...
 *	10	R10
 */
int32
RtoB(int r)
{

	if(r < 2 || r >= REGTMP-2)	// excluded R9 and R10 for m and g
		return 0;
	return 1L << r;
}

int
BtoR(int32 b)
{
	b &= 0x01fcL;	// excluded R9 and R10 for m and g
	if(b == 0)
		return 0;
	return bitno(b);
}

/*
 *	bit	reg
 *	18	F2
 *	19	F3
 *	...	...
 *	23	F7
 */
int32
FtoB(int f)
{

	if(f < 2 || f > NFREG-1)
		return 0;
	return 1L << (f + 16);
}

int
BtoF(int32 b)
{

	b &= 0xfc0000L;
	if(b == 0)
		return 0;
	return bitno(b) - 16;
}

static Sym*	symlist[10];

int
noreturn(Prog *p)
{
	Sym *s;
	int i;

	if(symlist[0] == S) {
		symlist[0] = pkglookup("panicindex", runtimepkg);
		symlist[1] = pkglookup("panicslice", runtimepkg);
		symlist[2] = pkglookup("throwinit", runtimepkg);
		symlist[3] = pkglookup("panic", runtimepkg);
	}

	s = p->to.sym;
	if(s == S)
		return 0;
	for(i=0; symlist[i]!=S; i++)
		if(s == symlist[i])
			return 1;
	return 0;
}

void
dumpone(Reg *r)
{
	int z;
	Bits bit;

	print("%d:%P", r->loop, r->prog);
	for(z=0; z<BITS; z++)
		bit.b[z] =
			r->set.b[z] |
			r->use1.b[z] |
			r->use2.b[z] |
			r->refbehind.b[z] |
			r->refahead.b[z] |
			r->calbehind.b[z] |
			r->calahead.b[z] |
			r->regdiff.b[z] |
			r->act.b[z] |
				0;
//	if(bany(&bit)) {
//		print("\t");
//		if(bany(&r->set))
//			print(" s:%Q", r->set);
//		if(bany(&r->use1))
//			print(" u1:%Q", r->use1);
//		if(bany(&r->use2))
//			print(" u2:%Q", r->use2);
//		if(bany(&r->refbehind))
//			print(" rb:%Q ", r->refbehind);
//		if(bany(&r->refahead))
//			print(" ra:%Q ", r->refahead);
//		if(bany(&r->calbehind))
//			print("cb:%Q ", r->calbehind);
//		if(bany(&r->calahead))
//			print(" ca:%Q ", r->calahead);
//		if(bany(&r->regdiff))
//			print(" d:%Q ", r->regdiff);
//		if(bany(&r->act))
//			print(" a:%Q ", r->act);
//	}
	print("\n");
}

void
dumpit(char *str, Reg *r0)
{
	Reg *r, *r1;

	print("\n%s\n", str);
	for(r = r0; r != R; r = r->link) {
		dumpone(r);
		r1 = r->p2;
		if(r1 != R) {
			print("	pred:");
			for(; r1 != R; r1 = r1->p2link)
				print(" %.4ud", r1->prog->loc);
			print("\n");
		}
//		r1 = r->s1;
//		if(r1 != R) {
//			print("	succ:");
//			for(; r1 != R; r1 = r1->s1)
//				print(" %.4ud", r1->prog->loc);
//			print("\n");
//		}
	}
}
