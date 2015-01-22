// Inferno utils/5c/reg.c
// http://code.google.com/p/inferno-os/source/browse/utils/5c/reg.c
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

#define	NREGVAR	32
#define	REGBITS	((uint64)0xffffffffull)
/*c2go enum {
	NREGVAR = 32,
	REGBITS = 0xffffffff,
};
*/

	void	addsplits(void);
static	Reg*	firstr;
static	int	first	= 1;

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
excise(Flow *r)
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
	Node *node;

	while(bany(&bit)) {
		// convert each bit to a variable
		i = bnum(bit);
		node = var[i].node;
		n = var[i].name;
		biclr(&bit, i);

		// disable all pieces of that variable
		for(i=0; i<nvar; i++) {
			v = var+i;
			if(v->node == node && v->name == n)
				v->addr = 2;
		}
	}
}

static char* regname[] = {
	".R0",
	".R1",
	".R2",
	".R3",
	".R4",
	".R5",
	".R6",
	".R7",
	".R8",
	".R9",
	".R10",
	".R11",
	".R12",
	".R13",
	".R14",
	".R15",
	".F0",
	".F1",
	".F2",
	".F3",
	".F4",
	".F5",
	".F6",
	".F7",
	".F8",
	".F9",
	".F10",
	".F11",
	".F12",
	".F13",
	".F14",
	".F15",
};

static Node* regnodes[NREGVAR];

static void walkvardef(Node *n, Reg *r, int active);

void
regopt(Prog *firstp)
{
	Reg *r, *r1;
	Prog *p;
	Graph *g;
	int i, z, active;
	uint32 vreg;
	Bits bit;
	ProgInfo info;

	if(first) {
		fmtinstall('Q', Qconv);
		first = 0;
	}

	mergetemp(firstp);

	/*
	 * control flow is more complicated in generated go code
	 * than in generated c code.  define pseudo-variables for
	 * registers, so we have complete register usage information.
	 */
	nvar = NREGVAR;
	memset(var, 0, NREGVAR*sizeof var[0]);
	for(i=0; i<NREGVAR; i++) {
		if(regnodes[i] == N)
			regnodes[i] = newname(lookup(regname[i]));
		var[i].node = regnodes[i];
	}

	regbits = RtoB(REGSP)|RtoB(REGLINK)|RtoB(REGPC);
	for(z=0; z<BITS; z++) {
		externs.b[z] = 0;
		params.b[z] = 0;
		consts.b[z] = 0;
		addrs.b[z] = 0;
		ivar.b[z] = 0;
		ovar.b[z] = 0;
	}

	/*
	 * pass 1
	 * build aux data structure
	 * allocate pcs
	 * find use and set of variables
	 */
	g = flowstart(firstp, sizeof(Reg));
	if(g == nil) {
		for(i=0; i<nvar; i++)
			var[i].node->opt = nil;
		return;
	}

	firstr = (Reg*)g->start;

	for(r = firstr; r != R; r = (Reg*)r->f.link) {
		p = r->f.prog;
		if(p->as == AVARDEF || p->as == AVARKILL)
			continue;
		proginfo(&info, p);

		// Avoid making variables for direct-called functions.
		if(p->as == ABL && p->to.name == D_EXTERN)
			continue;

		bit = mkvar(r, &p->from);
		if(info.flags & LeftRead)
			for(z=0; z<BITS; z++)
				r->use1.b[z] |= bit.b[z];
		if(info.flags & LeftAddr)
			setaddrs(bit);

		if(info.flags & RegRead) {	
			if(p->from.type != D_FREG)
				r->use1.b[0] |= RtoB(p->reg);
			else
				r->use1.b[0] |= FtoB(p->reg);
		}

		if(info.flags & (RightAddr | RightRead | RightWrite)) {
			bit = mkvar(r, &p->to);
			if(info.flags & RightAddr)
				setaddrs(bit);
			if(info.flags & RightRead)
				for(z=0; z<BITS; z++)
					r->use2.b[z] |= bit.b[z];
			if(info.flags & RightWrite)
				for(z=0; z<BITS; z++)
					r->set.b[z] |= bit.b[z];
		}

		/* the mod/div runtime routines smash R12 */
		if(p->as == ADIV || p->as == ADIVU || p->as == AMOD || p->as == AMODU)
			r->set.b[0] |= RtoB(12);
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

		if(debug['R'] && debug['v'])
			print("bit=%2d addr=%d et=%-6E w=%-2d s=%N + %lld\n",
				i, v->addr, v->etype, v->width, v->node, v->offset);
	}

	if(debug['R'] && debug['v'])
		dumpit("pass1", &firstr->f, 1);

	/*
	 * pass 2
	 * find looping structure
	 */
	flowrpo(g);

	if(debug['R'] && debug['v'])
		dumpit("pass2", &firstr->f, 1);

	/*
	 * pass 2.5
	 * iterate propagating fat vardef covering forward
	 * r->act records vars with a VARDEF since the last CALL.
	 * (r->act will be reused in pass 5 for something else,
	 * but we'll be done with it by then.)
	 */
	active = 0;
	for(r = firstr; r != R; r = (Reg*)r->f.link) {
		r->f.active = 0;
		r->act = zbits;
	}
	for(r = firstr; r != R; r = (Reg*)r->f.link) {
		p = r->f.prog;
		if(p->as == AVARDEF && isfat(((Node*)(p->to.node))->type) && ((Node*)(p->to.node))->opt != nil) {
			active++;
			walkvardef(p->to.node, r, active);
		}
	}

	/*
	 * pass 3
	 * iterate propagating usage
	 * 	back until flow graph is complete
	 */
loop1:
	change = 0;
	for(r = firstr; r != R; r = (Reg*)r->f.link)
		r->f.active = 0;
	for(r = firstr; r != R; r = (Reg*)r->f.link)
		if(r->f.prog->as == ARET)
			prop(r, zbits, zbits);
loop11:
	/* pick up unreachable code */
	i = 0;
	for(r = firstr; r != R; r = r1) {
		r1 = (Reg*)r->f.link;
		if(r1 && r1->f.active && !r->f.active) {
			prop(r, zbits, zbits);
			i = 1;
		}
	}
	if(i)
		goto loop11;
	if(change)
		goto loop1;

	if(debug['R'] && debug['v'])
		dumpit("pass3", &firstr->f, 1);


	/*
	 * pass 4
	 * iterate propagating register/variable synchrony
	 * 	forward until graph is complete
	 */
loop2:
	change = 0;
	for(r = firstr; r != R; r = (Reg*)r->f.link)
		r->f.active = 0;
	synch(firstr, zbits);
	if(change)
		goto loop2;

	addsplits();

	if(debug['R'] && debug['v'])
		dumpit("pass4", &firstr->f, 1);

	if(debug['R'] > 1) {
		print("\nprop structure:\n");
		for(r = firstr; r != R; r = (Reg*)r->f.link) {
			print("%d:%P", r->f.loop, r->f.prog);
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
	 * pass 4.5
	 * move register pseudo-variables into regu.
	 */
	for(r = firstr; r != R; r = (Reg*)r->f.link) {
		r->regu = (r->refbehind.b[0] | r->set.b[0]) & REGBITS;

		r->set.b[0] &= ~REGBITS;
		r->use1.b[0] &= ~REGBITS;
		r->use2.b[0] &= ~REGBITS;
		r->refbehind.b[0] &= ~REGBITS;
		r->refahead.b[0] &= ~REGBITS;
		r->calbehind.b[0] &= ~REGBITS;
		r->calahead.b[0] &= ~REGBITS;
		r->regdiff.b[0] &= ~REGBITS;
		r->act.b[0] &= ~REGBITS;
	}

	if(debug['R'] && debug['v'])
		dumpit("pass4.5", &firstr->f, 1);

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
		if(bany(&bit) && !r->f.refset) {
			// should never happen - all variables are preset
			if(debug['w'])
				print("%L: used and not set: %Q\n", r->f.prog->lineno, bit);
			r->f.refset = 1;
		}
	}

	for(r = firstr; r != R; r = (Reg*)r->f.link)
		r->act = zbits;
	rgp = region;
	nregion = 0;
	for(r = firstr; r != R; r = (Reg*)r->f.link) {
		for(z=0; z<BITS; z++)
			bit.b[z] = r->set.b[z] &
			  ~(r->refahead.b[z] | r->calahead.b[z] | addrs.b[z]);
		if(bany(&bit) && !r->f.refset) {
			if(debug['w'])
				print("%L: set and not used: %Q\n", r->f.prog->lineno, bit);
			r->f.refset = 1;
			excise(&r->f);
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
			biclr(&bit, i);
			if(change <= 0) {
				if(debug['R'])
					print("%L $%d: %Q\n",
						r->f.prog->lineno, change, blsh(i));
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

	if(debug['R'] && debug['v'])
		dumpit("pass5", &firstr->f, 1);

	/*
	 * pass 6
	 * determine used registers (paint2)
	 * replace code (paint3)
	 */
	rgp = region;
	if(debug['R'] && debug['v'])
		print("\nregisterizing\n");
	for(i=0; i<nregion; i++) {
		if(debug['R'] && debug['v'])
			print("region %d: cost %d varno %d enter %lld\n", i, rgp->cost, rgp->varno, rgp->enter->f.prog->pc);
		bit = blsh(rgp->varno);
		vreg = paint2(rgp->enter, rgp->varno, 0);
		vreg = allreg(vreg, rgp);
		if(debug['R']) {
			if(rgp->regno >= NREG)
				print("%L $%d F%d: %Q\n",
					rgp->enter->f.prog->lineno,
					rgp->cost,
					rgp->regno-NREG,
					bit);
			else
				print("%L $%d R%d: %Q\n",
					rgp->enter->f.prog->lineno,
					rgp->cost,
					rgp->regno,
					bit);
		}
		if(rgp->regno != 0)
			paint3(rgp->enter, rgp->varno, vreg, rgp->regno);
		rgp++;
	}

	/*
	 * free aux structures. peep allocates new ones.
	 */
	for(i=0; i<nvar; i++)
		var[i].node->opt = nil;
	flowend(g);
	firstr = R;

	if(debug['R'] && debug['v']) {
		// Rebuild flow graph, since we inserted instructions
		g = flowstart(firstp, sizeof(Reg));
		firstr = (Reg*)g->start;
		dumpit("pass6", &firstr->f, 1);
		flowend(g);
		firstr = R;
	}

	/*
	 * pass 7
	 * peep-hole on basic block
	 */
	if(!debug['R'] || debug['P']) {
		peep(firstp);
	}

	if(debug['R'] && debug['v'])
		dumpit("pass7", &firstr->f, 1);

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
			while(p->to.u.branch != P && p->to.u.branch->as == ANOP)
				p->to.u.branch = p->to.u.branch->link;
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
			if(p->from.sym != nil)
			if(p->from.name == D_AUTO || p->from.name == D_PARAM) {
				p->from.offset += vreg;
//				print("%P adjusting from %d %d\n", p, vreg, p->from.type);
			}
			if(p->to.sym != nil)
			if(p->to.name == D_AUTO || p->to.name == D_PARAM) {
				p->to.offset += vreg;
//				print("%P adjusting to %d %d\n", p, vreg, p->from.type);
			}
		}
	}
}

static void
walkvardef(Node *n, Reg *r, int active)
{
	Reg *r1, *r2;
	int bn;
	Var *v;
	
	for(r1=r; r1!=R; r1=(Reg*)r1->f.s1) {
		if(r1->f.active == active)
			break;
		r1->f.active = active;
		if(r1->f.prog->as == AVARKILL && r1->f.prog->to.node == n)
			break;
		for(v=n->opt; v!=nil; v=v->nextinnode) {
			bn = v - var;
			biset(&r1->act, bn);
		}
		if(r1->f.prog->as == ABL)
			break;
	}

	for(r2=r; r2!=r1; r2=(Reg*)r2->f.s1)
		if(r2->f.s2 != nil)
			walkvardef(n, (Reg*)r2->f.s2, active);
}

void
addsplits(void)
{
	Reg *r, *r1;
	int z, i;
	Bits bit;

	for(r = firstr; r != R; r = (Reg*)r->f.link) {
		if(r->f.loop > 1)
			continue;
		if(r->f.prog->as == ABL)
			continue;
		if(r->f.prog->as == ADUFFZERO)
			continue;
		if(r->f.prog->as == ADUFFCOPY)
			continue;
		for(r1 = (Reg*)r->f.p2; r1 != R; r1 = (Reg*)r1->f.p2link) {
			if(r1->f.loop <= 1)
				continue;
			for(z=0; z<BITS; z++)
				bit.b[z] = r1->calbehind.b[z] &
					(r->refahead.b[z] | r->use1.b[z] | r->use2.b[z]) &
					~(r->calahead.b[z] & addrs.b[z]);
			while(bany(&bit)) {
				i = bnum(bit);
				biclr(&bit, i);
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
	Prog *p, *p1, *p2;
	Adr *a;
	Var *v;

	p1 = mal(sizeof(*p1));
	*p1 = zprog;
	p = r->f.prog;
	
	// If there's a stack fixup coming (after BL newproc or BL deferproc),
	// delay the load until after the fixup.
	p2 = p->link;
	if(p2 && p2->as == AMOVW && p2->from.type == D_CONST && p2->from.reg == REGSP && p2->to.reg == REGSP && p2->to.type == D_REG)
		p = p2;

	p1->link = p->link;
	p->link = p1;
	p1->lineno = p->lineno;

	v = var + bn;

	a = &p1->to;
	a->name = v->name;
	a->node = v->node;
	a->sym = linksym(v->node->sym);
	a->offset = v->offset;
	a->etype = v->etype;
	a->type = D_OREG;
	if(a->etype == TARRAY || a->sym == nil)
		a->type = D_CONST;

	if(v->addr)
		fatal("addmove: shouldn't be doing this %A\n", a);

	switch(v->etype) {
	default:
		print("What is this %E\n", v->etype);

	case TINT8:
		p1->as = AMOVBS;
		break;
	case TBOOL:
	case TUINT8:
//print("movbu %E %d %S\n", v->etype, bn, v->sym);
		p1->as = AMOVBU;
		break;
	case TINT16:
		p1->as = AMOVHS;
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
	Node *node;

	// mark registers used
	t = a->type;

	flag = 0;
	switch(t) {
	default:
		print("type %d %d %D\n", t, a->name, a);
		goto none;

	case D_NONE:
	case D_FCONST:
	case D_BRANCH:
		break;


	case D_REGREG:
	case D_REGREG2:
		bit = zbits;
		if(a->offset != NREG)
			bit.b[0] |= RtoB(a->offset);
		if(a->reg != NREG)
			bit.b[0] |= RtoB(a->reg);
		return bit;

	case D_CONST:
	case D_REG:
	case D_SHIFT:
		if(a->reg != NREG) {
			bit = zbits;
			bit.b[0] = RtoB(a->reg);
			return bit;
		}
		break;

	case D_OREG:
		if(a->reg != NREG) {
			if(a == &r->f.prog->from)
				r->use1.b[0] |= RtoB(a->reg);
			else
				r->use2.b[0] |= RtoB(a->reg);
			if(r->f.prog->scond & (C_PBIT|C_WBIT))
				r->set.b[0] |= RtoB(a->reg);
		}
		break;

	case D_FREG:
		if(a->reg != NREG) {
			bit = zbits;
			bit.b[0] = FtoB(a->reg);
			return bit;
		}
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

	node = a->node;
	if(node == N || node->op != ONAME || node->orig == N)
		goto none;
	node = node->orig;
	if(node->orig != node)
		fatal("%D: bad node", a);
	if(node->sym == S || node->sym->name[0] == '.')
		goto none;
	et = a->etype;
	o = a->offset;
	w = a->width;
	if(w < 0)
		fatal("bad width %d for %D", w, a);

	for(i=0; i<nvar; i++) {
		v = var+i;
		if(v->node == node && v->name == n) {
			if(v->offset == o)
			if(v->etype == et)
			if(v->width == w)
				if(!flag)
					return blsh(i);

			// if they overlap, disable both
			if(overlap(v->offset, v->width, o, w)) {
				v->addr = 1;
				flag = 1;
			}
		}
	}

	switch(et) {
	case 0:
	case TFUNC:
		goto none;
	}

	if(nvar >= NVAR) {
		if(debug['w'] > 1 && node)
			fatal("variable not optimized: %D", a);
		
		// If we're not tracking a word in a variable, mark the rest as
		// having its address taken, so that we keep the whole thing
		// live at all calls. otherwise we might optimize away part of
		// a variable but not all of it.
		for(i=0; i<nvar; i++) {
			v = var+i;
			if(v->node == node)
				v->addr = 1;
		}
		goto none;
	}

	i = nvar;
	nvar++;
//print("var %d %E %D %S\n", i, et, a, s);
	v = var+i;
	v->offset = o;
	v->name = n;
	v->etype = et;
	v->width = w;
	v->addr = flag;		// funny punning
	v->node = node;
	
	// node->opt is the head of a linked list
	// of Vars within the given Node, so that
	// we can start at a Var and find all the other
	// Vars in the same Go variable.
	v->nextinnode = node->opt;
	node->opt = v;
	
	bit = blsh(i);
	if(n == D_EXTERN || n == D_STATIC)
		for(z=0; z<BITS; z++)
			externs.b[z] |= bit.b[z];
	if(n == D_PARAM)
		for(z=0; z<BITS; z++)
			params.b[z] |= bit.b[z];

	if(node->class == PPARAM)
		for(z=0; z<BITS; z++)
			ivar.b[z] |= bit.b[z];
	if(node->class == PPARAMOUT)
		for(z=0; z<BITS; z++)
			ovar.b[z] |= bit.b[z];

	// Treat values with their address taken as live at calls,
	// because the garbage collector's liveness analysis in ../gc/plive.c does.
	// These must be consistent or else we will elide stores and the garbage
	// collector will see uninitialized data.
	// The typical case where our own analysis is out of sync is when the
	// node appears to have its address taken but that code doesn't actually
	// get generated and therefore doesn't show up as an address being
	// taken when we analyze the instruction stream.
	// One instance of this case is when a closure uses the same name as
	// an outer variable for one of its own variables declared with :=.
	// The parser flags the outer variable as possibly shared, and therefore
	// sets addrtaken, even though it ends up not being actually shared.
	// If we were better about _ elision, _ = &x would suffice too.
	// The broader := in a closure problem is mentioned in a comment in
	// closure.c:/^typecheckclosure and dcl.c:/^oldname.
	if(node->addrtaken)
		v->addr = 1;

	// Disable registerization for globals, because:
	// (1) we might panic at any time and we want the recovery code
	// to see the latest values (issue 1304).
	// (2) we don't know what pointers might point at them and we want
	// loads via those pointers to see updated values and vice versa (issue 7995).
	//
	// Disable registerization for results if using defer, because the deferred func
	// might recover and return, causing the current values to be used.
	if(node->class == PEXTERN || (hasdefer && node->class == PPARAMOUT))
		v->addr = 1;

	if(debug['R'])
		print("bit=%2d et=%2E w=%d+%d %#N %D flag=%d\n", i, et, o, w, node, a, v->addr);

	return bit;

none:
	return zbits;
}

void
prop(Reg *r, Bits ref, Bits cal)
{
	Reg *r1, *r2;
	int z, i, j;
	Var *v, *v1;

	for(r1 = r; r1 != R; r1 = (Reg*)r1->f.p1) {
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
		switch(r1->f.prog->as) {
		case ABL:
			if(noreturn(r1->f.prog))
				break;

			// Mark all input variables (ivar) as used, because that's what the
			// liveness bitmaps say. The liveness bitmaps say that so that a
			// panic will not show stale values in the parameter dump.
			// Mark variables with a recent VARDEF (r1->act) as used,
			// so that the optimizer flushes initializations to memory,
			// so that if a garbage collection happens during this CALL,
			// the collector will see initialized memory. Again this is to
			// match what the liveness bitmaps say.
			for(z=0; z<BITS; z++) {
				cal.b[z] |= ref.b[z] | externs.b[z] | ivar.b[z] | r1->act.b[z];
				ref.b[z] = 0;
			}
			
			// cal.b is the current approximation of what's live across the call.
			// Every bit in cal.b is a single stack word. For each such word,
			// find all the other tracked stack words in the same Go variable
			// (struct/slice/string/interface) and mark them live too.
			// This is necessary because the liveness analysis for the garbage
			// collector works at variable granularity, not at word granularity.
			// It is fundamental for slice/string/interface: the garbage collector
			// needs the whole value, not just some of the words, in order to
			// interpret the other bits correctly. Specifically, slice needs a consistent
			// ptr and cap, string needs a consistent ptr and len, and interface
			// needs a consistent type word and data word.
			for(z=0; z<BITS; z++) {
				if(cal.b[z] == 0)
					continue;
				for(i=0; i<64; i++) {
					if(z*64+i >= nvar || ((cal.b[z]>>i)&1) == 0)
						continue;
					v = var+z*64+i;
					if(v->node->opt == nil) // v represents fixed register, not Go variable
						continue;

					// v->node->opt is the head of a linked list of Vars
					// corresponding to tracked words from the Go variable v->node.
					// Walk the list and set all the bits.
					// For a large struct this could end up being quadratic:
					// after the first setting, the outer loop (for z, i) would see a 1 bit
					// for all of the remaining words in the struct, and for each such
					// word would go through and turn on all the bits again.
					// To avoid the quadratic behavior, we only turn on the bits if
					// v is the head of the list or if the head's bit is not yet turned on.
					// This will set the bits at most twice, keeping the overall loop linear.
					v1 = v->node->opt;
					j = v1 - var;
					if(v == v1 || !btest(&cal, j)) {
						for(; v1 != nil; v1 = v1->nextinnode) {
							j = v1 - var;
							biset(&cal, j);
						}
					}
				}
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
		if(r1->f.active)
			break;
		r1->f.active = 1;
	}
	for(; r != r1; r = (Reg*)r->f.p1)
		for(r2 = (Reg*)r->f.p2; r2 != R; r2 = (Reg*)r2->f.p2link)
			prop(r2, r->refbehind, r->calbehind);
}

void
synch(Reg *r, Bits dif)
{
	Reg *r1;
	int z;

	for(r1 = r; r1 != R; r1 = (Reg*)r1->f.s1) {
		for(z=0; z<BITS; z++) {
			dif.b[z] = (dif.b[z] &
				~(~r1->refbehind.b[z] & r1->refahead.b[z])) |
					r1->set.b[z] | r1->regdiff.b[z];
			if(dif.b[z] != r1->regdiff.b[z]) {
				r1->regdiff.b[z] = dif.b[z];
				change++;
			}
		}
		if(r1->f.active)
			break;
		r1->f.active = 1;
		for(z=0; z<BITS; z++)
			dif.b[z] &= ~(~r1->calbehind.b[z] & r1->calahead.b[z]);
		if(r1->f.s2 != nil)
			synch((Reg*)r1->f.s2, dif);
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
	uint64 bb;

	z = bn/64;
	bb = 1LL<<(bn%64);
	if(r->act.b[z] & bb)
		return;
	for(;;) {
		if(!(r->refbehind.b[z] & bb))
			break;
		r1 = (Reg*)r->f.p1;
		if(r1 == R)
			break;
		if(!(r1->refahead.b[z] & bb))
			break;
		if(r1->act.b[z] & bb)
			break;
		r = r1;
	}

	if(LOAD(r) & ~(r->set.b[z] & ~(r->use1.b[z]|r->use2.b[z])) & bb) {
		change -= CLOAD * r->f.loop;
		if(debug['R'] > 1)
			print("%d%P\td %Q $%d\n", r->f.loop,
				r->f.prog, blsh(bn), change);
	}
	for(;;) {
		r->act.b[z] |= bb;
		p = r->f.prog;


		if(r->f.prog->as != ANOP) { // don't give credit for NOPs
			if(r->use1.b[z] & bb) {
				change += CREF * r->f.loop;
				if(debug['R'] > 1)
					print("%d%P\tu1 %Q $%d\n", r->f.loop,
						p, blsh(bn), change);
			}
			if((r->use2.b[z]|r->set.b[z]) & bb) {
				change += CREF * r->f.loop;
				if(debug['R'] > 1)
					print("%d%P\tu2 %Q $%d\n", r->f.loop,
						p, blsh(bn), change);
			}
		}

		if(STORE(r) & r->regdiff.b[z] & bb) {
			change -= CLOAD * r->f.loop;
			if(debug['R'] > 1)
				print("%d%P\tst %Q $%d\n", r->f.loop,
					p, blsh(bn), change);
		}

		if(r->refbehind.b[z] & bb)
			for(r1 = (Reg*)r->f.p2; r1 != R; r1 = (Reg*)r1->f.p2link)
				if(r1->refahead.b[z] & bb)
					paint1(r1, bn);

		if(!(r->refahead.b[z] & bb))
			break;
		r1 = (Reg*)r->f.s2;
		if(r1 != R)
			if(r1->refbehind.b[z] & bb)
				paint1(r1, bn);
		r = (Reg*)r->f.s1;
		if(r == R)
			break;
		if(r->act.b[z] & bb)
			break;
		if(!(r->refbehind.b[z] & bb))
			break;
	}
}

uint32
paint2(Reg *r, int bn, int depth)
{
	Reg *r1;
	int z;
	uint64 bb, vreg;

	z = bn/64;
	bb = 1LL << (bn%64);
	vreg = regbits;
	if(!(r->act.b[z] & bb))
		return vreg;
	for(;;) {
		if(!(r->refbehind.b[z] & bb))
			break;
		r1 = (Reg*)r->f.p1;
		if(r1 == R)
			break;
		if(!(r1->refahead.b[z] & bb))
			break;
		if(!(r1->act.b[z] & bb))
			break;
		r = r1;
	}
	for(;;) {
		if(debug['R'] && debug['v'])
			print("  paint2 %d %P\n", depth, r->f.prog);

		r->act.b[z] &= ~bb;

		vreg |= r->regu;

		if(r->refbehind.b[z] & bb)
			for(r1 = (Reg*)r->f.p2; r1 != R; r1 = (Reg*)r1->f.p2link)
				if(r1->refahead.b[z] & bb)
					vreg |= paint2(r1, bn, depth+1);

		if(!(r->refahead.b[z] & bb))
			break;
		r1 = (Reg*)r->f.s2;
		if(r1 != R)
			if(r1->refbehind.b[z] & bb)
				vreg |= paint2(r1, bn, depth+1);
		r = (Reg*)r->f.s1;
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
paint3(Reg *r, int bn, uint32 rb, int rn)
{
	Reg *r1;
	Prog *p;
	int z;
	uint64 bb;

	z = bn/64;
	bb = 1LL << (bn%64);
	if(r->act.b[z] & bb)
		return;
	for(;;) {
		if(!(r->refbehind.b[z] & bb))
			break;
		r1 = (Reg*)r->f.p1;
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
		p = r->f.prog;

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
			for(r1 = (Reg*)r->f.p2; r1 != R; r1 = (Reg*)r1->f.p2link)
				if(r1->refahead.b[z] & bb)
					paint3(r1, bn, rb, rn);

		if(!(r->refahead.b[z] & bb))
			break;
		r1 = (Reg*)r->f.s2;
		if(r1 != R)
			if(r1->refbehind.b[z] & bb)
				paint3(r1, bn, rb, rn);
		r = (Reg*)r->f.s1;
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
	a->sym = nil;
	a->node = nil;
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
 *	12  R12
 */
uint32
RtoB(int r)
{
	if(r >= REGTMP-2 && r != 12)	// excluded R9 and R10 for m and g, but not R12
		return 0;
	return 1L << r;
}

int
BtoR(uint32 b)
{
	// TODO Allow R0 and R1, but be careful with a 0 return
	// TODO Allow R9. Only R10 is reserved now (just g, not m).
	b &= 0x11fcL;	// excluded R9 and R10 for m and g, but not R12
	if(b == 0)
		return 0;
	return bitno(b);
}

/*
 *	bit	reg
 *	18	F2
 *	19	F3
 *	...	...
 *	31	F15
 */
uint32
FtoB(int f)
{

	if(f < 2 || f > NFREG-1)
		return 0;
	return 1L << (f + 16);
}

int
BtoF(uint32 b)
{

	b &= 0xfffc0000L;
	if(b == 0)
		return 0;
	return bitno(b) - 16;
}

void
dumpone(Flow *f, int isreg)
{
	int z;
	Bits bit;
	Reg *r;

	print("%d:%P", f->loop, f->prog);
	if(isreg) {
		r = (Reg*)f;
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
		if(bany(&bit)) {
			print("\t");
			if(bany(&r->set))
				print(" s:%Q", r->set);
			if(bany(&r->use1))
				print(" u1:%Q", r->use1);
			if(bany(&r->use2))
				print(" u2:%Q", r->use2);
			if(bany(&r->refbehind))
				print(" rb:%Q ", r->refbehind);
			if(bany(&r->refahead))
				print(" ra:%Q ", r->refahead);
			if(bany(&r->calbehind))
				print(" cb:%Q ", r->calbehind);
			if(bany(&r->calahead))
				print(" ca:%Q ", r->calahead);
			if(bany(&r->regdiff))
				print(" d:%Q ", r->regdiff);
			if(bany(&r->act))
				print(" a:%Q ", r->act);
		}
	}
	print("\n");
}

void
dumpit(char *str, Flow *r0, int isreg)
{
	Flow *r, *r1;

	print("\n%s\n", str);
	for(r = r0; r != nil; r = r->link) {
		dumpone(r, isreg);
		r1 = r->p2;
		if(r1 != nil) {
			print("	pred:");
			for(; r1 != nil; r1 = r1->p2link)
				print(" %.4ud", (int)r1->prog->pc);
			if(r->p1 != nil)
				print(" (and %.4ud)", (int)r->p1->prog->pc);
			else
				print(" (only)");
			print("\n");
		}
		// Print successors if it's not just the next one
		if(r->s1 != r->link || r->s2 != nil) {
			print("	succ:");
			if(r->s1 != nil)
				print(" %.4ud", (int)r->s1->prog->pc);
			if(r->s2 != nil)
				print(" %.4ud", (int)r->s2->prog->pc);
			print("\n");
		}
	}
}
