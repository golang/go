// Inferno utils/6c/sgen.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/sgen.c
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
#include "../../runtime/funcdata.h"

int
hasdotdotdot(Type *t)
{
	for(t=t->down; t!=T; t=t->down)
		if(t->etype == TDOT)
			return 1;
	return 0;
}

vlong
argsize(int doret)
{
	Type *t;
	int32 s;

//print("t=%T\n", thisfn);
	s = 0;
	if(hasdotdotdot(thisfn))
		s = align(s, thisfn->link, Aarg0, nil);
	for(t=thisfn->down; t!=T; t=t->down) {
		switch(t->etype) {
		case TVOID:
			break;
		case TDOT:
			if((textflag & NOSPLIT) == 0)
				yyerror("function takes ... without textflag NOSPLIT");
			return ArgsSizeUnknown;
		default:
			s = align(s, t, Aarg1, nil);
			s = align(s, t, Aarg2, nil);
			break;
		}
//print("	%d %T\n", s, t);
	}
	if(thechar == '6')
		s = (s+7) & ~7;
	else
		s = (s+3) & ~3;
	if(doret && thisfn->link->etype != TVOID) {
		s = align(s, thisfn->link, Aarg1, nil);
		s = align(s, thisfn->link, Aarg2, nil);
		if(thechar == '6')
			s = (s+7) & ~7;
		else
			s = (s+3) & ~3;
	}
	return s;
}

void
codgen(Node *n, Node *nn)
{
	Prog *sp;
	Node *n1, nod, nod1;

	cursafe = 0;
	curarg = 0;
	maxargsafe = 0;

	/*
	 * isolate name
	 */
	for(n1 = nn;; n1 = n1->left) {
		if(n1 == Z) {
			diag(nn, "can't find function name");
			return;
		}
		if(n1->op == ONAME)
			break;
	}
	nearln = nn->lineno;

	p = gtext(n1->sym, stkoff);
	p->from.sym->cfunc = 1;
	sp = p;

	/*
	 * isolate first argument
	 */
	if(REGARG >= 0) {
		if(typesuv[thisfn->link->etype]) {
			nod1 = *nodret->left;
			nodreg(&nod, &nod1, REGARG);
			gmove(&nod, &nod1);
		} else
		if(firstarg && typechlp[firstargtype->etype]) {
			nod1 = *nodret->left;
			nod1.sym = firstarg;
			nod1.type = firstargtype;
			nod1.xoffset = align(0, firstargtype, Aarg1, nil);
			nod1.etype = firstargtype->etype;
			nodreg(&nod, &nod1, REGARG);
			gmove(&nod, &nod1);
		}
	}

	canreach = 1;
	warnreach = 1;
	gen(n);
	if(canreach && thisfn->link->etype != TVOID)
		diag(Z, "no return at end of function: %s", n1->sym->name);
	noretval(3);
	gbranch(ORETURN);

	if(!debug['N'] || debug['R'] || debug['P'])
		regopt(sp);

	if(thechar=='6' || thechar=='7')	/* [sic] */
		maxargsafe = xround(maxargsafe, 8);
	sp->to.offset += maxargsafe;
}

void
supgen(Node *n)
{
	int owarn;
	long spc;
	Prog *sp;

	if(n == Z)
		return;
	suppress++;
	owarn = warnreach;
	warnreach = 0;
	spc = pc;
	sp = lastp;
	gen(n);
	lastp = sp;
	pc = spc;
	sp->link = nil;
	suppress--;
	warnreach = owarn;
}

void
gen(Node *n)
{
	Node *l, nod, nod1;
	Prog *sp, *spc, *spb;
	Case *cn;
	long sbc, scc;
	int snbreak, sncontin;
	int f, o, oldreach;

loop:
	if(n == Z)
		return;
	nearln = n->lineno;
	o = n->op;
	if(debug['G'])
		if(o != OLIST)
			print("%L %O\n", nearln, o);

	if(!canreach) {
		switch(o) {
		case OLABEL:
		case OCASE:
		case OLIST:
		case OBREAK:
		case OFOR:
		case OWHILE:
		case ODWHILE:
			/* all handled specially - see switch body below */
			break;
		default:
			if(warnreach) {
				warn(n, "unreachable code %O", o);
				warnreach = 0;
			}
		}
	}

	switch(o) {

	default:
		complex(n);
		cgen(n, Z);
		break;

	case OLIST:
		gen(n->left);

	rloop:
		n = n->right;
		goto loop;

	case ORETURN:
		canreach = 0;
		warnreach = !suppress;
		complex(n);
		if(n->type == T)
			break;
		l = n->left;
		if(l == Z) {
			noretval(3);
			gbranch(ORETURN);
			break;
		}
		if(typecmplx[n->type->etype] && !hasdotdotdot(thisfn)) {
			regret(&nod, n, thisfn, 2);
			sugen(l, &nod, n->type->width);
			noretval(3);
			gbranch(ORETURN);
			break;
		}
		if(typecmplx[n->type->etype]) {
			sugen(l, nodret, n->type->width);
			noretval(3);
			gbranch(ORETURN);
			break;
		}
		regret(&nod1, n, thisfn, 2);
		nod = nod1;
		if(nod.op != OREGISTER)
			regalloc(&nod, n, Z);
		cgen(l, &nod);
		if(nod1.op != OREGISTER)
			gmove(&nod, &nod1);
		regfree(&nod);
		if(typefd[n->type->etype])
			noretval(1);
		else
			noretval(2);
		gbranch(ORETURN);
		break;

	case OLABEL:
		canreach = 1;
		l = n->left;
		if(l) {
			l->pc = pc;
			if(l->label)
				patch(l->label, pc);
		}
		gbranch(OGOTO);	/* prevent self reference in reg */
		patch(p, pc);
		goto rloop;

	case OGOTO:
		canreach = 0;
		warnreach = !suppress;
		n = n->left;
		if(n == Z)
			return;
		if(n->complex == 0) {
			diag(Z, "label undefined: %s", n->sym->name);
			return;
		}
		if(suppress)
			return;
		gbranch(OGOTO);
		if(n->pc) {
			patch(p, n->pc);
			return;
		}
		if(n->label)
			patch(n->label, pc-1);
		n->label = p;
		return;

	case OCASE:
		canreach = 1;
		l = n->left;
		if(cases == C)
			diag(n, "case/default outside a switch");
		if(l == Z) {
			newcase();
			cases->val = 0;
			cases->def = 1;
			cases->label = pc;
			cases->isv = 0;
			goto rloop;
		}
		complex(l);
		if(l->type == T)
			goto rloop;
		if(l->op == OCONST)
		if(typeword[l->type->etype] && l->type->etype != TIND) {
			newcase();
			cases->val = l->vconst;
			cases->def = 0;
			cases->label = pc;
			cases->isv = typev[l->type->etype];
			goto rloop;
		}
		diag(n, "case expression must be integer constant");
		goto rloop;

	case OSWITCH:
		l = n->left;
		complex(l);
		if(l->type == T)
			break;
		if(!typechlvp[l->type->etype] || l->type->etype == TIND) {
			diag(n, "switch expression must be integer");
			break;
		}

		gbranch(OGOTO);		/* entry */
		sp = p;

		cn = cases;
		cases = C;
		newcase();

		sbc = breakpc;
		breakpc = pc;
		snbreak = nbreak;
		nbreak = 0;
		gbranch(OGOTO);
		spb = p;

		gen(n->right);		/* body */
		if(canreach){
			gbranch(OGOTO);
			patch(p, breakpc);
			nbreak++;
		}

		patch(sp, pc);
		doswit(l);
		patch(spb, pc);

		cases = cn;
		breakpc = sbc;
		canreach = nbreak!=0;
		if(canreach == 0)
			warnreach = !suppress;
		nbreak = snbreak;
		break;

	case OWHILE:
	case ODWHILE:
		l = n->left;
		gbranch(OGOTO);		/* entry */
		sp = p;

		scc = continpc;
		continpc = pc;
		gbranch(OGOTO);
		spc = p;

		sbc = breakpc;
		breakpc = pc;
		snbreak = nbreak;
		nbreak = 0;
		gbranch(OGOTO);
		spb = p;

		patch(spc, pc);
		if(n->op == OWHILE)
			patch(sp, pc);
		bcomplex(l, Z);		/* test */
		patch(p, breakpc);
		if(l->op != OCONST || vconst(l) == 0)
			nbreak++;

		if(n->op == ODWHILE)
			patch(sp, pc);
		gen(n->right);		/* body */
		gbranch(OGOTO);
		patch(p, continpc);

		patch(spb, pc);
		continpc = scc;
		breakpc = sbc;
		canreach = nbreak!=0;
		if(canreach == 0)
			warnreach = !suppress;
		nbreak = snbreak;
		break;

	case OFOR:
		l = n->left;
		if(!canreach && l->right->left && warnreach) {
			warn(n, "unreachable code FOR");
			warnreach = 0;
		}
		gen(l->right->left);	/* init */
		gbranch(OGOTO);		/* entry */
		sp = p;

		/*
		 * if there are no incoming labels in the
		 * body and the top's not reachable, warn
		 */
		if(!canreach && warnreach && deadheads(n)) {
			warn(n, "unreachable code %O", o);
			warnreach = 0;
		}

		scc = continpc;
		continpc = pc;
		gbranch(OGOTO);
		spc = p;

		sbc = breakpc;
		breakpc = pc;
		snbreak = nbreak;
		nbreak = 0;
		sncontin = ncontin;
		ncontin = 0;
		gbranch(OGOTO);
		spb = p;

		patch(spc, pc);
		gen(l->right->right);	/* inc */
		patch(sp, pc);
		if(l->left != Z) {	/* test */
			bcomplex(l->left, Z);
			patch(p, breakpc);
			if(l->left->op != OCONST || vconst(l->left) == 0)
				nbreak++;
		}
		canreach = 1;
		gen(n->right);		/* body */
		if(canreach){
			gbranch(OGOTO);
			patch(p, continpc);
			ncontin++;
		}
		if(!ncontin && l->right->right && warnreach) {
			warn(l->right->right, "unreachable FOR inc");
			warnreach = 0;
		}

		patch(spb, pc);
		continpc = scc;
		breakpc = sbc;
		canreach = nbreak!=0;
		if(canreach == 0)
			warnreach = !suppress;
		nbreak = snbreak;
		ncontin = sncontin;
		break;

	case OCONTINUE:
		if(continpc < 0) {
			diag(n, "continue not in a loop");
			break;
		}
		gbranch(OGOTO);
		patch(p, continpc);
		ncontin++;
		canreach = 0;
		warnreach = !suppress;
		break;

	case OBREAK:
		if(breakpc < 0) {
			diag(n, "break not in a loop");
			break;
		}
		/*
		 * Don't complain about unreachable break statements.
		 * There are breaks hidden in yacc's output and some people
		 * write return; break; in their switch statements out of habit.
		 * However, don't confuse the analysis by inserting an
		 * unreachable reference to breakpc either.
		 */
		if(!canreach)
			break;
		gbranch(OGOTO);
		patch(p, breakpc);
		nbreak++;
		canreach = 0;
		warnreach = !suppress;
		break;

	case OIF:
		l = n->left;
		if(bcomplex(l, n->right)) {
			if(typefd[l->type->etype])
				f = !l->fconst;
			else
				f = !l->vconst;
			if(debug['c'])
				print("%L const if %s\n", nearln, f ? "false" : "true");
			if(f) {
				canreach = 1;
				supgen(n->right->left);
				oldreach = canreach;
				canreach = 1;
				gen(n->right->right);
				/*
				 * treat constant ifs as regular ifs for
				 * reachability warnings.
				 */
				if(!canreach && oldreach && debug['w'] < 2)
					warnreach = 0;
			}
			else {
				canreach = 1;
				gen(n->right->left);
				oldreach = canreach;
				canreach = 1;
				supgen(n->right->right);
				/*
				 * treat constant ifs as regular ifs for
				 * reachability warnings.
				 */
				if(!oldreach && canreach && debug['w'] < 2)
					warnreach = 0;
				canreach = oldreach;
			}
		}
		else {
			sp = p;
			canreach = 1;
			if(n->right->left != Z)
				gen(n->right->left);
			oldreach = canreach;
			canreach = 1;
			if(n->right->right != Z) {
				gbranch(OGOTO);
				patch(sp, pc);
				sp = p;
				gen(n->right->right);
			}
			patch(sp, pc);
			canreach = canreach || oldreach;
			if(canreach == 0)
				warnreach = !suppress;
		}
		break;

	case OSET:
	case OUSED:
	case OPREFETCH:
		usedset(n->left, o);
		break;
	}
}

void
usedset(Node *n, int o)
{
	if(n->op == OLIST) {
		usedset(n->left, o);
		usedset(n->right, o);
		return;
	}
	complex(n);
	if(o == OPREFETCH) {
		gprefetch(n);
		return;
	}
	switch(n->op) {
	case OADDR:	/* volatile */
		gins(ANOP, n, Z);
		break;
	case ONAME:
		if(o == OSET)
			gins(ANOP, Z, n);
		else
			gins(ANOP, n, Z);
		break;
	}
}

int
bcomplex(Node *n, Node *c)
{
	Node *b, nod;

	complex(n);
	if(n->type != T)
	if(tcompat(n, T, n->type, tnot))
		n->type = T;
	if(n->type == T) {
		gbranch(OGOTO);
		return 0;
	}
	if(c != Z && n->op == OCONST && deadheads(c))
		return 1;
	if(typev[n->type->etype] && machcap(Z)) {
		b = &nod;
		b->op = ONE;
		b->left = n;
		b->right = new(0, Z, Z);
		*b->right = *nodconst(0);
		b->right->type = n->type;
		b->type = types[TLONG];
		n = b;
	}
	bool64(n);
	boolgen(n, 1, Z);
	return 0;
}
