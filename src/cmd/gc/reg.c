// Derived from Inferno utils/6c/reg.c
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

#include <u.h>
#include <libc.h>
#include "go.h"
#include "popt.h"

static	Flow*	firstf;
static	int	first	= 1;

static void	addmove(Flow*, int, int, int);
static Bits	mkvar(Flow*, Adr*);
static void	prop(Flow*, Bits, Bits);
static void	synch(Flow*, Bits);
static uint64	allreg(uint64, Rgn*);
static void	paint1(Flow*, int);
static uint64	paint2(Flow*, int, int);
static void	paint3(Flow*, int, uint64, int);
static void	addreg(Adr*, int);

static int
rcmp(const void *a1, const void *a2)
{
	Rgn *p1, *p2;

	p1 = (Rgn*)a1;
	p2 = (Rgn*)a2;
	if(p1->cost != p2->cost)
		return p2->cost - p1->cost;
	if(p1->varno != p2->varno)
		return p2->varno - p1->varno;
	if(p1->enter != p2->enter)
		return p2->enter->id - p1->enter->id;
	return 0;
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

static Node* regnodes[64];

static void walkvardef(Node *n, Flow *r, int active);

void
regopt(Prog *firstp)
{
	Flow *f, *f1;
	Reg *r;
	Prog *p;
	Graph *g;
	ProgInfo info;
	int i, z, active;
	uint64 vreg, usedreg;
	uint64 mask;
	int nreg;
	char **regnames;
	Bits bit;
	Rgn *rgp;

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
	regnames = thearch.regnames(&nreg);
	nvar = nreg;
	memset(var, 0, nreg*sizeof var[0]);
	for(i=0; i<nreg; i++) {
		if(regnodes[i] == N)
			regnodes[i] = newname(lookup(regnames[i]));
		var[i].node = regnodes[i];
	}

	regbits = thearch.excludedregs();
	externs = zbits;
	params = zbits;
	consts = zbits;
	addrs = zbits;
	ivar = zbits;
	ovar = zbits;

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

	firstf = g->start;

	for(f = firstf; f != nil; f = f->link) {
		p = f->prog;
		if(p->as == AVARDEF || p->as == AVARKILL)
			continue;
		thearch.proginfo(&info, p);

		// Avoid making variables for direct-called functions.
		if(p->as == ACALL && p->to.type == TYPE_MEM && p->to.name == NAME_EXTERN)
			continue;

		// from vs to doesn't matter for registers.
		r = (Reg*)f->data;
		r->use1.b[0] |= info.reguse | info.regindex;
		r->set.b[0] |= info.regset;

		bit = mkvar(f, &p->from);
		if(bany(&bit)) {
			if(info.flags & LeftAddr)
				setaddrs(bit);
			if(info.flags & LeftRead)
				for(z=0; z<BITS; z++)
					r->use1.b[z] |= bit.b[z];
			if(info.flags & LeftWrite)
				for(z=0; z<BITS; z++)
					r->set.b[z] |= bit.b[z];
		}

		// Compute used register for reg
		if(info.flags & RegRead)
			r->use1.b[0] |= thearch.RtoB(p->reg);

		// Currently we never generate three register forms.
		// If we do, this will need to change.
		if(p->from3.type != TYPE_NONE)
			fatal("regopt not implemented for from3");

		bit = mkvar(f, &p->to);
		if(bany(&bit)) {	
			if(info.flags & RightAddr)
				setaddrs(bit);
			if(info.flags & RightRead)
				for(z=0; z<BITS; z++)
					r->use2.b[z] |= bit.b[z];
			if(info.flags & RightWrite)
				for(z=0; z<BITS; z++)
					r->set.b[z] |= bit.b[z];
		}
	}

	for(i=0; i<nvar; i++) {
		Var *v;
		v = var+i;
		if(v->addr) {
			bit = blsh(i);
			for(z=0; z<BITS; z++)
				addrs.b[z] |= bit.b[z];
		}

		if(debug['R'] && debug['v'])
			print("bit=%2d addr=%d et=%E w=%-2d s=%N + %lld\n",
				i, v->addr, v->etype, v->width, v->node, v->offset);
	}

	if(debug['R'] && debug['v'])
		dumpit("pass1", firstf, 1);

	/*
	 * pass 2
	 * find looping structure
	 */
	flowrpo(g);

	if(debug['R'] && debug['v'])
		dumpit("pass2", firstf, 1);

	/*
	 * pass 2.5
	 * iterate propagating fat vardef covering forward
	 * r->act records vars with a VARDEF since the last CALL.
	 * (r->act will be reused in pass 5 for something else,
	 * but we'll be done with it by then.)
	 */
	active = 0;
	for(f = firstf; f != nil; f = f->link) {
		f->active = 0;
		r = (Reg*)f->data;
		r->act = zbits;
	}
	for(f = firstf; f != nil; f = f->link) {
		p = f->prog;
		if(p->as == AVARDEF && isfat(((Node*)(p->to.node))->type) && ((Node*)(p->to.node))->opt != nil) {
			active++;
			walkvardef(p->to.node, f, active);
		}
	}

	/*
	 * pass 3
	 * iterate propagating usage
	 * 	back until flow graph is complete
	 */
loop1:
	change = 0;
	for(f = firstf; f != nil; f = f->link)
		f->active = 0;
	for(f = firstf; f != nil; f = f->link)
		if(f->prog->as == ARET)
			prop(f, zbits, zbits);
loop11:
	/* pick up unreachable code */
	i = 0;
	for(f = firstf; f != nil; f = f1) {
		f1 = f->link;
		if(f1 && f1->active && !f->active) {
			prop(f, zbits, zbits);
			i = 1;
		}
	}
	if(i)
		goto loop11;
	if(change)
		goto loop1;

	if(debug['R'] && debug['v'])
		dumpit("pass3", firstf, 1);

	/*
	 * pass 4
	 * iterate propagating register/variable synchrony
	 * 	forward until graph is complete
	 */
loop2:
	change = 0;
	for(f = firstf; f != nil; f = f->link)
		f->active = 0;
	synch(firstf, zbits);
	if(change)
		goto loop2;

	if(debug['R'] && debug['v'])
		dumpit("pass4", firstf, 1);

	/*
	 * pass 4.5
	 * move register pseudo-variables into regu.
	 */
	if(nreg == 64)
		mask = ~0ULL; // can't rely on C to shift by 64
	else
		mask = (1ULL<<nreg) - 1;
	for(f = firstf; f != nil; f = f->link) {
		r = (Reg*)f->data;
		r->regu = (r->refbehind.b[0] | r->set.b[0]) & mask;
		r->set.b[0] &= ~mask;
		r->use1.b[0] &= ~mask;
		r->use2.b[0] &= ~mask;
		r->refbehind.b[0] &= ~mask;
		r->refahead.b[0] &= ~mask;
		r->calbehind.b[0] &= ~mask;
		r->calahead.b[0] &= ~mask;
		r->regdiff.b[0] &= ~mask;
		r->act.b[0] &= ~mask;
	}

	if(debug['R'] && debug['v'])
		dumpit("pass4.5", firstf, 1);

	/*
	 * pass 5
	 * isolate regions
	 * calculate costs (paint1)
	 */
	f = firstf;
	if(f) {
		r = (Reg*)f->data;
		for(z=0; z<BITS; z++)
			bit.b[z] = (r->refahead.b[z] | r->calahead.b[z]) &
			  ~(externs.b[z] | params.b[z] | addrs.b[z] | consts.b[z]);
		if(bany(&bit) && !f->refset) {
			// should never happen - all variables are preset
			if(debug['w'])
				print("%L: used and not set: %Q\n", f->prog->lineno, bit);
			f->refset = 1;
		}
	}
	for(f = firstf; f != nil; f = f->link)
		((Reg*)f->data)->act = zbits;
	nregion = 0;
	for(f = firstf; f != nil; f = f->link) {
		r = (Reg*)f->data;
		for(z=0; z<BITS; z++)
			bit.b[z] = r->set.b[z] &
			  ~(r->refahead.b[z] | r->calahead.b[z] | addrs.b[z]);
		if(bany(&bit) && !f->refset) {
			if(debug['w'])
				print("%L: set and not used: %Q\n", f->prog->lineno, bit);
			f->refset = 1;
			thearch.excise(f);
		}
		for(z=0; z<BITS; z++)
			bit.b[z] = LOAD(r) & ~(r->act.b[z] | addrs.b[z]);
		while(bany(&bit)) {
			i = bnum(bit);
			change = 0;
			paint1(f, i);
			biclr(&bit, i);
			if(change <= 0)
				continue;
			if(nregion >= NRGN) {
				if(debug['R'] && debug['v'])
					print("too many regions\n");
				goto brk;
			}
			rgp = &region[nregion];
			rgp->enter = f;
			rgp->varno = i;
			rgp->cost = change;
			nregion++;
		}
	}
brk:
	qsort(region, nregion, sizeof(region[0]), rcmp);

	if(debug['R'] && debug['v'])
		dumpit("pass5", firstf, 1);

	/*
	 * pass 6
	 * determine used registers (paint2)
	 * replace code (paint3)
	 */
	if(debug['R'] && debug['v'])
		print("\nregisterizing\n");
	for(i=0; i<nregion; i++) {
		rgp = &region[i];
		if(debug['R'] && debug['v'])
			print("region %d: cost %d varno %d enter %lld\n", i, rgp->cost, rgp->varno, rgp->enter->prog->pc);
		bit = blsh(rgp->varno);
		usedreg = paint2(rgp->enter, rgp->varno, 0);
		vreg = allreg(usedreg, rgp);
		if(rgp->regno != 0) {
			if(debug['R'] && debug['v']) {
				Var *v;

				v = var + rgp->varno;
				print("registerize %N+%lld (bit=%2d et=%E) in %R usedreg=%#llx vreg=%#llx\n",
						v->node, v->offset, rgp->varno, v->etype, rgp->regno, usedreg, vreg);
			}
			paint3(rgp->enter, rgp->varno, vreg, rgp->regno);
		}
	}

	/*
	 * free aux structures. peep allocates new ones.
	 */
	for(i=0; i<nvar; i++)
		var[i].node->opt = nil;
	flowend(g);
	firstf = nil;

	if(debug['R'] && debug['v']) {
		// Rebuild flow graph, since we inserted instructions
		g = flowstart(firstp, 0);
		firstf = g->start;
		dumpit("pass6", firstf, 0);
		flowend(g);
		firstf = nil;
	}

	/*
	 * pass 7
	 * peep-hole on basic block
	 */
	if(!debug['R'] || debug['P'])
		thearch.peep(firstp);

	/*
	 * eliminate nops
	 */
	for(p=firstp; p!=P; p=p->link) {
		while(p->link != P && p->link->as == ANOP)
			p->link = p->link->link;
		if(p->to.type == TYPE_BRANCH)
			while(p->to.u.branch != P && p->to.u.branch->as == ANOP)
				p->to.u.branch = p->to.u.branch->link;
	}

	if(debug['R']) {
		if(ostats.ncvtreg ||
		   ostats.nspill ||
		   ostats.nreload ||
		   ostats.ndelmov ||
		   ostats.nvar ||
		   ostats.naddr ||
		   0)
			print("\nstats\n");

		if(ostats.ncvtreg)
			print("	%4d cvtreg\n", ostats.ncvtreg);
		if(ostats.nspill)
			print("	%4d spill\n", ostats.nspill);
		if(ostats.nreload)
			print("	%4d reload\n", ostats.nreload);
		if(ostats.ndelmov)
			print("	%4d delmov\n", ostats.ndelmov);
		if(ostats.nvar)
			print("	%4d var\n", ostats.nvar);
		if(ostats.naddr)
			print("	%4d addr\n", ostats.naddr);

		memset(&ostats, 0, sizeof(ostats));
	}
}

static void
walkvardef(Node *n, Flow *f, int active)
{
	Flow *f1, *f2;
	int bn;
	Var *v;
	
	for(f1=f; f1!=nil; f1=f1->s1) {
		if(f1->active == active)
			break;
		f1->active = active;
		if(f1->prog->as == AVARKILL && f1->prog->to.node == n)
			break;
		for(v=n->opt; v!=nil; v=v->nextinnode) {
			bn = v->id;
			biset(&((Reg*)f1->data)->act, bn);
		}
		if(f1->prog->as == ACALL)
			break;
	}

	for(f2=f; f2!=f1; f2=f2->s1)
		if(f2->s2 != nil)
			walkvardef(n, f2->s2, active);
}

/*
 * add mov b,rn
 * just after r
 */
static void
addmove(Flow *r, int bn, int rn, int f)
{
	Prog *p, *p1;
	Adr *a;
	Var *v;

	p1 = mal(sizeof(*p1));
	clearp(p1);
	p1->pc = 9999;

	p = r->prog;
	p1->link = p->link;
	p->link = p1;
	p1->lineno = p->lineno;

	v = var + bn;

	a = &p1->to;
	a->offset = v->offset;
	a->etype = v->etype;
	a->type = TYPE_MEM;
	a->name = v->name;
	a->node = v->node;
	a->sym = linksym(v->node->sym);
	/* NOTE(rsc): 9g did
	if(a->etype == TARRAY)
		a->type = TYPE_ADDR;
	else if(a->sym == nil)
		a->type = TYPE_CONST;
	*/

	p1->as = thearch.optoas(OAS, types[(uchar)v->etype]);
	// TODO(rsc): Remove special case here.
	if((thearch.thechar == '9' || thearch.thechar == '5') && v->etype == TBOOL)
		p1->as = thearch.optoas(OAS, types[TUINT8]);
	p1->from.type = TYPE_REG;
	p1->from.reg = rn;
	p1->from.name = NAME_NONE;
	if(!f) {
		p1->from = *a;
		*a = zprog.from;
		a->type = TYPE_REG;
		a->reg = rn;
	}
	if(debug['R'] && debug['v'])
		print("%P ===add=== %P\n", p, p1);
	ostats.nspill++;
}

static int
overlap(int64 o1, int w1, int64 o2, int w2)
{
	int64 t1, t2;

	t1 = o1+w1;
	t2 = o2+w2;

	if(!(t1 > o2 && t2 > o1))
		return 0;

	return 1;
}

static Bits
mkvar(Flow *f, Adr *a)
{
	Var *v;
	int i, n, et, z, flag;
	int64 w;
	uint64 regu;
	int64 o;
	Bits bit;
	Node *node;
	Reg *r;
	

	/*
	 * mark registers used
	 */
	if(a->type == TYPE_NONE)
		goto none;

	r = (Reg*)f->data;
	r->use1.b[0] |= thearch.doregbits(a->index); // TODO: Use RtoB

	switch(a->type) {
	default:
		regu = thearch.doregbits(a->reg) | thearch.RtoB(a->reg); // TODO: Use RtoB
		if(regu == 0)
			goto none;
		bit = zbits;
		bit.b[0] = regu;
		return bit;

	case TYPE_ADDR:
		// TODO(rsc): Remove special case here.
		if(thearch.thechar == '9' || thearch.thechar == '5')
			goto memcase;
		a->type = TYPE_MEM;
		bit = mkvar(f, a);
		setaddrs(bit);
		a->type = TYPE_ADDR;
		ostats.naddr++;
		goto none;

	case TYPE_MEM:
	memcase:
		if(r != R) {
			r->use1.b[0] |= thearch.RtoB(a->reg);
			/* NOTE: 5g did
				if(r->f.prog->scond & (C_PBIT|C_WBIT))
					r->set.b[0] |= RtoB(a->reg);
			*/
		}
		switch(a->name) {
		default:
			goto none;
		case NAME_EXTERN:
		case NAME_STATIC:
		case NAME_PARAM:
		case NAME_AUTO:
			n = a->name;
			break;
		}
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
		fatal("bad width %lld for %D", w, a);

	flag = 0;
	for(i=0; i<nvar; i++) {
		v = var+i;
		if(v->node == node && v->name == n) {
			if(v->offset == o)
			if(v->etype == et)
			if(v->width == w) {
				// TODO(rsc): Remove special case for arm here.
				if(!flag || thearch.thechar != '5')
					return blsh(i);
			}

			// if they overlap, disable both
			if(overlap(v->offset, v->width, o, w)) {
//				print("disable overlap %s %d %d %d %d, %E != %E\n", s->name, v->offset, v->width, o, w, v->etype, et);
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
		if(debug['w'] > 1 && node != N)
			fatal("variable not optimized: %#N", node);
		
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
	v = var+i;
	v->id = i;
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
	if(n == NAME_EXTERN || n == NAME_STATIC)
		for(z=0; z<BITS; z++)
			externs.b[z] |= bit.b[z];
	if(n == NAME_PARAM)
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
		print("bit=%2d et=%E w=%lld+%lld %#N %D flag=%d\n", i, et, o, w, node, a, v->addr);
	ostats.nvar++;

	return bit;

none:
	return zbits;
}

static void
prop(Flow *f, Bits ref, Bits cal)
{
	Flow *f1, *f2;
	Reg *r, *r1;
	int z, i;
	Var *v, *v1;

	for(f1 = f; f1 != nil; f1 = f1->p1) {
		r1 = (Reg*)f1->data;
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
		switch(f1->prog->as) {
		case ACALL:
			if(noreturn(f1->prog))
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
					if(v == v1 || !btest(&cal, v1->id)) {
						for(; v1 != nil; v1 = v1->nextinnode) {
							biset(&cal, v1->id);
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
		if(f1->active)
			break;
		f1->active = 1;
	}

	for(; f != f1; f = f->p1) {
		r = (Reg*)f->data;
		for(f2 = f->p2; f2 != nil; f2 = f2->p2link)
			prop(f2, r->refbehind, r->calbehind);
	}
}

static void
synch(Flow *f, Bits dif)
{
	Flow *f1;
	Reg *r1;
	int z;

	for(f1 = f; f1 != nil; f1 = f1->s1) {
		r1 = (Reg*)f1->data;
		for(z=0; z<BITS; z++) {
			dif.b[z] = (dif.b[z] &
				~(~r1->refbehind.b[z] & r1->refahead.b[z])) |
					r1->set.b[z] | r1->regdiff.b[z];
			if(dif.b[z] != r1->regdiff.b[z]) {
				r1->regdiff.b[z] = dif.b[z];
				change++;
			}
		}
		if(f1->active)
			break;
		f1->active = 1;
		for(z=0; z<BITS; z++)
			dif.b[z] &= ~(~r1->calbehind.b[z] & r1->calahead.b[z]);
		if(f1->s2 != nil)
			synch(f1->s2, dif);
	}
}

static uint64
allreg(uint64 b, Rgn *r)
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
	case TINT64:
	case TUINT64:
	case TINT:
	case TUINT:
	case TUINTPTR:
	case TBOOL:
	case TPTR32:
	case TPTR64:
		i = thearch.BtoR(~b);
		if(i && r->cost > 0) {
			r->regno = i;
			return thearch.RtoB(i);
		}
		break;

	case TFLOAT32:
	case TFLOAT64:
		i = thearch.BtoF(~b);
		if(i && r->cost > 0) {
			r->regno = i;
			return thearch.FtoB(i);
		}
		break;
	}
	return 0;
}

static void
paint1(Flow *f, int bn)
{
	Flow *f1;
	Reg *r, *r1;
	int z;
	uint64 bb;

	z = bn/64;
	bb = 1LL<<(bn%64);
	r = (Reg*)f->data;
	if(r->act.b[z] & bb)
		return;
	for(;;) {
		if(!(r->refbehind.b[z] & bb))
			break;
		f1 = f->p1;
		if(f1 == nil)
			break;
		r1 = (Reg*)f1->data;
		if(!(r1->refahead.b[z] & bb))
			break;
		if(r1->act.b[z] & bb)
			break;
		f = f1;
		r = r1;
	}

	if(LOAD(r) & ~(r->set.b[z]&~(r->use1.b[z]|r->use2.b[z])) & bb) {
		change -= CLOAD * f->loop;
	}
	for(;;) {
		r->act.b[z] |= bb;

		if(f->prog->as != ANOP) { // don't give credit for NOPs
			if(r->use1.b[z] & bb)
				change += CREF * f->loop;
			if((r->use2.b[z]|r->set.b[z]) & bb)
				change += CREF * f->loop;
		}

		if(STORE(r) & r->regdiff.b[z] & bb) {
			change -= CLOAD * f->loop;
		}

		if(r->refbehind.b[z] & bb)
			for(f1 = f->p2; f1 != nil; f1 = f1->p2link)
				if(((Reg*)f1->data)->refahead.b[z] & bb)
					paint1(f1, bn);

		if(!(r->refahead.b[z] & bb))
			break;
		f1 = f->s2;
		if(f1 != nil)
			if(((Reg*)f1->data)->refbehind.b[z] & bb)
				paint1(f1, bn);
		f = f->s1;
		if(f == nil)
			break;
		r = (Reg*)f->data;
		if(r->act.b[z] & bb)
			break;
		if(!(r->refbehind.b[z] & bb))
			break;
	}
}

static uint64
paint2(Flow *f, int bn, int depth)
{
	Flow *f1;
	Reg *r, *r1;
	int z;
	uint64 bb, vreg;

	z = bn/64;
	bb = 1LL << (bn%64);
	vreg = regbits;
	r = (Reg*)f->data;
	if(!(r->act.b[z] & bb))
		return vreg;
	for(;;) {
		if(!(r->refbehind.b[z] & bb))
			break;
		f1 = f->p1;
		if(f1 == nil)
			break;
		r1 = (Reg*)f1->data;
		if(!(r1->refahead.b[z] & bb))
			break;
		if(!(r1->act.b[z] & bb))
			break;
		f = f1;
		r = r1;
	}
	for(;;) {
		if(debug['R'] && debug['v'])
			print("  paint2 %d %P\n", depth, f->prog);

		r->act.b[z] &= ~bb;

		vreg |= r->regu;

		if(r->refbehind.b[z] & bb)
			for(f1 = f->p2; f1 != nil; f1 = f1->p2link)
				if(((Reg*)f1->data)->refahead.b[z] & bb)
					vreg |= paint2(f1, bn, depth+1);

		if(!(r->refahead.b[z] & bb))
			break;
		f1 = f->s2;
		if(f1 != nil)
			if(((Reg*)f1->data)->refbehind.b[z] & bb)
				vreg |= paint2(f1, bn, depth+1);
		f = f->s1;
		if(f == nil)
			break;
		r = (Reg*)f->data;
		if(!(r->act.b[z] & bb))
			break;
		if(!(r->refbehind.b[z] & bb))
			break;
	}

	return vreg;
}

static void
paint3(Flow *f, int bn, uint64 rb, int rn)
{
	Flow *f1;
	Reg *r, *r1;
	Prog *p;
	int z;
	uint64 bb;

	z = bn/64;
	bb = 1LL << (bn%64);
	r = (Reg*)f->data;
	if(r->act.b[z] & bb)
		return;
	for(;;) {
		if(!(r->refbehind.b[z] & bb))
			break;
		f1 = f->p1;
		if(f1 == nil)
			break;
		r1 = (Reg*)f1->data;
		if(!(r1->refahead.b[z] & bb))
			break;
		if(r1->act.b[z] & bb)
			break;
		f = f1;
		r = r1;
	}

	if(LOAD(r) & ~(r->set.b[z] & ~(r->use1.b[z]|r->use2.b[z])) & bb)
		addmove(f, bn, rn, 0);
	for(;;) {
		r->act.b[z] |= bb;
		p = f->prog;

		if(r->use1.b[z] & bb) {
			if(debug['R'] && debug['v'])
				print("%P", p);
			addreg(&p->from, rn);
			if(debug['R'] && debug['v'])
				print(" ===change== %P\n", p);
		}
		if((r->use2.b[z]|r->set.b[z]) & bb) {
			if(debug['R'] && debug['v'])
				print("%P", p);
			addreg(&p->to, rn);
			if(debug['R'] && debug['v'])
				print(" ===change== %P\n", p);
		}

		if(STORE(r) & r->regdiff.b[z] & bb)
			addmove(f, bn, rn, 1);
		r->regu |= rb;

		if(r->refbehind.b[z] & bb)
			for(f1 = f->p2; f1 != nil; f1 = f1->p2link)
				if(((Reg*)f1->data)->refahead.b[z] & bb)
					paint3(f1, bn, rb, rn);

		if(!(r->refahead.b[z] & bb))
			break;
		f1 = f->s2;
		if(f1 != nil)
			if(((Reg*)f1->data)->refbehind.b[z] & bb)
				paint3(f1, bn, rb, rn);
		f = f->s1;
		if(f == nil)
			break;
		r = (Reg*)f->data;
		if(r->act.b[z] & bb)
			break;
		if(!(r->refbehind.b[z] & bb))
			break;
	}
}

static void
addreg(Adr *a, int rn)
{
	a->sym = nil;
	a->node = nil;
	a->offset = 0;
	a->type = TYPE_REG;
	a->reg = rn;
	a->name = 0;

	ostats.ncvtreg++;
}

void
dumpone(Flow *f, int isreg)
{
	int z;
	Bits bit;
	Reg *r;

	print("%d:%P", f->loop, f->prog);
	if(isreg) {	
		r = (Reg*)f->data;
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
