// Inferno utils/5c/sgen.c
// http://code.google.com/p/inferno-os/source/browse/utils/5c/sgen.c
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

int32
argsize(void)
{
	Type *t;
	int32 s;

//print("t=%T\n", thisfn);
	s = 0;
	for(t=thisfn->down; t!=T; t=t->down) {
		switch(t->etype) {
		case TVOID:
			break;
		case TDOT:
			s += 64;
			break;
		default:
			s = align(s, t, Aarg1);
			s = align(s, t, Aarg2);
			break;
		}
//print("	%d %T\n", s, t);
	}
	return (s+7) & ~7;
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
			diag(nn, "cant find function name");
			return;
		}
		if(n1->op == ONAME)
			break;
	}
	nearln = nn->lineno;
	gpseudo(ATEXT, n1->sym, nodconst(stkoff));
	p->to.type = D_CONST2;
	p->to.offset2 = argsize();

	sp = p;

	/*
	 * isolate first argument
	 */
	if(REGARG >= 0) {
		if(typesuv[thisfn->link->etype]) {
			nod1 = *nodret->left;
			nodreg(&nod, &nod1, REGARG);
			gopcode(OAS, &nod, Z, &nod1);
		} else
		if(firstarg && typechlp[firstargtype->etype]) {
			nod1 = *nodret->left;
			nod1.sym = firstarg;
			nod1.type = firstargtype;
			nod1.xoffset = align(0, firstargtype, Aarg1);
			nod1.etype = firstargtype->etype;
			nodreg(&nod, &nod1, REGARG);
			gopcode(OAS, &nod, Z, &nod1);
		}
	}

	retok = 0;
	gen(n);
	if(!retok)
		if(thisfn->link->etype != TVOID)
			warn(Z, "no return at end of function: %s", n1->sym->name);
	noretval(3);
	gbranch(ORETURN);

	if(!debug['N'] || debug['R'] || debug['P'])
		regopt(sp);

	sp->to.offset += maxargsafe;
}

void
supgen(Node *n)
{
	int32 spc;
	Prog *sp;

	if(n == Z)
		return;
	suppress++;
	spc = pc;
	sp = lastp;
	gen(n);
	lastp = sp;
	pc = spc;
	sp->link = nil;
	suppress--;
}

void
gen(Node *n)
{
	Node *l, nod;
	Prog *sp, *spc, *spb;
	Case *cn;
	int32 sbc, scc;
	int o, f;

loop:
	if(n == Z)
		return;
	nearln = n->lineno;
	o = n->op;
	if(debug['G'])
		if(o != OLIST)
			print("%L %O\n", nearln, o);

	retok = 0;
	switch(o) {

	default:
		complex(n);
		cgen(n, Z, 0);
		break;

	case OLIST:
		gen(n->left);

	rloop:
		n = n->right;
		goto loop;

	case ORETURN:
		retok = 1;
		complex(n);
		if(n->type == T)
			break;
		l = n->left;
		if(l == Z) {
			noretval(3);
			gbranch(ORETURN);
			break;
		}
		if(typesuv[n->type->etype]) {
			sugen(l, nodret, n->type->width);
			noretval(3);
			gbranch(ORETURN);
			break;
		}
		regret(&nod, n);
		cgen(l, &nod, 0);
		regfree(&nod);
		if(typefd[n->type->etype])
			noretval(1);
		else
			noretval(2);
		gbranch(ORETURN);
		break;

	case OLABEL:
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
		retok = 1;
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
		l = n->left;
		if(cases == C)
			diag(n, "case/default outside a switch");
		if(l == Z) {
			cas();
			cases->val = 0;
			cases->def = 1;
			cases->label = pc;
			goto rloop;
		}
		complex(l);
		if(l->type == T)
			goto rloop;
		if(l->op == OCONST)
		if(typechl[l->type->etype]) {
			cas();
			cases->val = l->vconst;
			cases->def = 0;
			cases->label = pc;
			goto rloop;
		}
		diag(n, "case expression must be integer constant");
		goto rloop;

	case OSWITCH:
		l = n->left;
		complex(l);
		if(l->type == T)
			break;
		if(!typechl[l->type->etype]) {
			diag(n, "switch expression must be integer");
			break;
		}

		gbranch(OGOTO);		/* entry */
		sp = p;

		cn = cases;
		cases = C;
		cas();

		sbc = breakpc;
		breakpc = pc;
		gbranch(OGOTO);
		spb = p;

		gen(n->right);
		gbranch(OGOTO);
		patch(p, breakpc);

		patch(sp, pc);
		regalloc(&nod, l, Z);
		nod.type = types[TLONG];
		cgen(l, &nod, 0);
		doswit(&nod);
		regfree(&nod);
		patch(spb, pc);

		cases = cn;
		breakpc = sbc;
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
		gbranch(OGOTO);
		spb = p;

		patch(spc, pc);
		if(n->op == OWHILE)
			patch(sp, pc);
		bcomplex(l, Z);		/* test */
		patch(p, breakpc);

		if(n->op == ODWHILE)
			patch(sp, pc);
		gen(n->right);		/* body */
		gbranch(OGOTO);
		patch(p, continpc);

		patch(spb, pc);
		continpc = scc;
		breakpc = sbc;
		break;

	case OFOR:
		l = n->left;
		gen(l->right->left);	/* init */
		gbranch(OGOTO);		/* entry */
		sp = p;

		scc = continpc;
		continpc = pc;
		gbranch(OGOTO);
		spc = p;

		sbc = breakpc;
		breakpc = pc;
		gbranch(OGOTO);
		spb = p;

		patch(spc, pc);
		gen(l->right->right);	/* inc */
		patch(sp, pc);
		if(l->left != Z) {	/* test */
			bcomplex(l->left, Z);
			patch(p, breakpc);
		}
		gen(n->right);		/* body */
		gbranch(OGOTO);
		patch(p, continpc);

		patch(spb, pc);
		continpc = scc;
		breakpc = sbc;
		break;

	case OCONTINUE:
		if(continpc < 0) {
			diag(n, "continue not in a loop");
			break;
		}
		gbranch(OGOTO);
		patch(p, continpc);
		break;

	case OBREAK:
		if(breakpc < 0) {
			diag(n, "break not in a loop");
			break;
		}
		gbranch(OGOTO);
		patch(p, breakpc);
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
				supgen(n->right->left);
				gen(n->right->right);
			}
			else {
				gen(n->right->left);
				supgen(n->right->right);
			}
		}
		else {
			sp = p;
			if(n->right->left != Z)
				gen(n->right->left);
			if(n->right->right != Z) {
				gbranch(OGOTO);
				patch(sp, pc);
				sp = p;
				gen(n->right->right);
			}
			patch(sp, pc);
		}
		break;

	case OSET:
	case OUSED:
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

void
noretval(int n)
{

	if(n & 1) {
		gins(ANOP, Z, Z);
		p->to.type = D_REG;
		p->to.reg = REGRET;
	}
	if(n & 2) {
		gins(ANOP, Z, Z);
		p->to.type = D_FREG;
		p->to.reg = FREGRET;
	}
}

/*
 *	calculate addressability as follows
 *		CONST ==> 20		$value
 *		NAME ==> 10		name
 *		REGISTER ==> 11		register
 *		INDREG ==> 12		*[(reg)+offset]
 *		&10 ==> 2		$name
 *		ADD(2, 20) ==> 2	$name+offset
 *		ADD(3, 20) ==> 3	$(reg)+offset
 *		&12 ==> 3		$(reg)+offset
 *		*11 ==> 11		??
 *		*2 ==> 10		name
 *		*3 ==> 12		*(reg)+offset
 *	calculate complexity (number of registers)
 */
void
xcom(Node *n)
{
	Node *l, *r;
	int t;

	if(n == Z)
		return;
	l = n->left;
	r = n->right;
	n->addable = 0;
	n->complex = 0;
	switch(n->op) {
	case OCONST:
		n->addable = 20;
		return;

	case OREGISTER:
		n->addable = 11;
		return;

	case OINDREG:
		n->addable = 12;
		return;

	case ONAME:
		n->addable = 10;
		return;

	case OADDR:
		xcom(l);
		if(l->addable == 10)
			n->addable = 2;
		if(l->addable == 12)
			n->addable = 3;
		break;

	case OIND:
		xcom(l);
		if(l->addable == 11)
			n->addable = 12;
		if(l->addable == 3)
			n->addable = 12;
		if(l->addable == 2)
			n->addable = 10;
		break;

	case OADD:
		xcom(l);
		xcom(r);
		if(l->addable == 20) {
			if(r->addable == 2)
				n->addable = 2;
			if(r->addable == 3)
				n->addable = 3;
		}
		if(r->addable == 20) {
			if(l->addable == 2)
				n->addable = 2;
			if(l->addable == 3)
				n->addable = 3;
		}
		break;

	case OASLMUL:
	case OASMUL:
		xcom(l);
		xcom(r);
		t = vlog(r);
		if(t >= 0) {
			n->op = OASASHL;
			r->vconst = t;
			r->type = types[TINT];
		}
		break;

	case OMUL:
	case OLMUL:
		xcom(l);
		xcom(r);
		t = vlog(r);
		if(t >= 0) {
			n->op = OASHL;
			r->vconst = t;
			r->type = types[TINT];
		}
		t = vlog(l);
		if(t >= 0) {
			n->op = OASHL;
			n->left = r;
			n->right = l;
			r = l;
			l = n->left;
			r->vconst = t;
			r->type = types[TINT];
		}
		break;

	case OASLDIV:
		xcom(l);
		xcom(r);
		t = vlog(r);
		if(t >= 0) {
			n->op = OASLSHR;
			r->vconst = t;
			r->type = types[TINT];
		}
		break;

	case OLDIV:
		xcom(l);
		xcom(r);
		t = vlog(r);
		if(t >= 0) {
			n->op = OLSHR;
			r->vconst = t;
			r->type = types[TINT];
		}
		break;

	case OASLMOD:
		xcom(l);
		xcom(r);
		t = vlog(r);
		if(t >= 0) {
			n->op = OASAND;
			r->vconst--;
		}
		break;

	case OLMOD:
		xcom(l);
		xcom(r);
		t = vlog(r);
		if(t >= 0) {
			n->op = OAND;
			r->vconst--;
		}
		break;

	default:
		if(l != Z)
			xcom(l);
		if(r != Z)
			xcom(r);
		break;
	}
	if(n->addable >= 10)
		return;

	if(l != Z)
		n->complex = l->complex;
	if(r != Z) {
		if(r->complex == n->complex)
			n->complex = r->complex+1;
		else
		if(r->complex > n->complex)
			n->complex = r->complex;
	}
	if(n->complex == 0)
		n->complex++;

	if(com64(n))
		return;

	switch(n->op) {
	case OFUNC:
		n->complex = FNX;
		break;

	case OADD:
	case OXOR:
	case OAND:
	case OOR:
	case OEQ:
	case ONE:
		/*
		 * immediate operators, make const on right
		 */
		if(l->op == OCONST) {
			n->left = r;
			n->right = l;
		}
		break;
	}
}

int
bcomplex(Node *n, Node *c)
{

	complex(n);
	if(n->type != T)
	if(tcompat(n, T, n->type, tnot))
		n->type = T;
	if(n->type != T) {
		if(c != Z && n->op == OCONST && deadheads(c))
			return 1;
		bool64(n);
		boolgen(n, 1, Z);
	} else
		gbranch(OGOTO);
	return 0;
}
