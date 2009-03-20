// Inferno utils/8c/sgen.c
// http://code.google.com/p/inferno-os/source/browse/utils/8c/sgen.c
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

	/*
	 * isolate first argument
	 */
	if(REGARG) {
		if(typesuv[thisfn->link->etype]) {
			nod1 = *nodret->left;
			nodreg(&nod, &nod1, REGARG);
			gmove(&nod, &nod1);
		} else
		if(firstarg && typechlp[firstargtype->etype]) {
			nod1 = *nodret->left;
			nod1.sym = firstarg;
			nod1.type = firstargtype;
			nod1.xoffset = align(0, firstargtype, Aarg1);
			nod1.etype = firstargtype->etype;
			nodreg(&nod, &nod1, REGARG);
			gmove(&nod, &nod1);
		}
	}

	sp = p;
	retok = 0;
	gen(n);
	if(!retok)
		if(thisfn->link->etype != TVOID)
			warn(Z, "no return at end of function: %s", n1->sym->name);
	noretval(3);
	if(thisfn && thisfn->link && typefd[thisfn->link->etype])
		gins(AFLDZ, Z, Z);
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
	int f, o;

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
		cgen(n, Z);
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
			if(typefd[n->type->etype])
				gins(AFLDZ, Z, Z);
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
		cgen(l, &nod);
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
			l->xoffset = pc;
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
		if(n->xoffset) {
			patch(p, n->xoffset);
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
		cgen(l, &nod);
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
		p->to.type = REGRET;
	}
	if(n & 2) {
		gins(ANOP, Z, Z);
		p->to.type = FREGRET;
	}
}

/* welcome to commute */
static void
commute(Node *n)
{
	Node *l, *r;

	l = n->left;
	r = n->right;
	if(r->complex > l->complex) {
		n->left = r;
		n->right = l;
	}
}

void
indexshift(Node *n)
{
	int g;

	if(!typechlp[n->type->etype])
		return;
	simplifyshift(n);
	if(n->op == OASHL && n->right->op == OCONST){
		g = vconst(n->right);
		if(g >= 0 && g < 4)
			n->addable = 7;
	}
}

/*
 *	calculate addressability as follows
 *		NAME ==> 10/11		name+value(SB/SP)
 *		REGISTER ==> 12		register
 *		CONST ==> 20		$value
 *		*(20) ==> 21		value
 *		&(10) ==> 13		$name+value(SB)
 *		&(11) ==> 1		$name+value(SP)
 *		(13) + (20) ==> 13	fold constants
 *		(1) + (20) ==> 1	fold constants
 *		*(13) ==> 10		back to name
 *		*(1) ==> 11		back to name
 *
 *		(20) * (X) ==> 7	multiplier in indexing
 *		(X,7) + (13,1) ==> 8	adder in indexing (addresses)
 *		(8) ==> &9(OINDEX)	index, almost addressable
 *
 *	calculate complexity (number of registers)
 */
void
xcom(Node *n)
{
	Node *l, *r;
	int g;

	if(n == Z)
		return;
	l = n->left;
	r = n->right;
	n->complex = 0;
	n->addable = 0;
	switch(n->op) {
	case OCONST:
		n->addable = 20;
		break;

	case ONAME:
		n->addable = 10;
		if(n->class == CPARAM || n->class == CAUTO)
			n->addable = 11;
		break;

	case OEXREG:
		n->addable = 10;
		break;

	case OREGISTER:
		n->addable = 12;
		break;

	case OINDREG:
		n->addable = 12;
		break;

	case OADDR:
		xcom(l);
		if(l->addable == 10)
			n->addable = 13;
		else
		if(l->addable == 11)
			n->addable = 1;
		break;

	case OADD:
		xcom(l);
		xcom(r);
		if(n->type->etype != TIND)
			break;

		switch(r->addable) {
		case 20:
			switch(l->addable) {
			case 1:
			case 13:
			commadd:
				l->type = n->type;
				*n = *l;
				l = new(0, Z, Z);
				*l = *(n->left);
				l->xoffset += r->vconst;
				n->left = l;
				r = n->right;
				goto brk;
			}
			break;

		case 1:
		case 13:
		case 10:
		case 11:
			/* l is the base, r is the index */
			if(l->addable != 20)
				n->addable = 8;
			break;
		}
		switch(l->addable) {
		case 20:
			switch(r->addable) {
			case 13:
			case 1:
				r = n->left;
				l = n->right;
				n->left = l;
				n->right = r;
				goto commadd;
			}
			break;

		case 13:
		case 1:
		case 10:
		case 11:
			/* r is the base, l is the index */
			if(r->addable != 20)
				n->addable = 8;
			break;
		}
		if(n->addable == 8 && !side(n)) {
			indx(n);
			l = new1(OINDEX, idx.basetree, idx.regtree);
			l->scale = idx.scale;
			l->addable = 9;
			l->complex = l->right->complex;
			l->type = l->left->type;
			n->op = OADDR;
			n->left = l;
			n->right = Z;
			n->addable = 8;
			break;
		}
		break;

	case OINDEX:
		xcom(l);
		xcom(r);
		n->addable = 9;
		break;

	case OIND:
		xcom(l);
		if(l->op == OADDR) {
			l = l->left;
			l->type = n->type;
			*n = *l;
			return;
		}
		switch(l->addable) {
		case 20:
			n->addable = 21;
			break;
		case 1:
			n->addable = 11;
			break;
		case 13:
			n->addable = 10;
			break;
		}
		break;

	case OASHL:
		xcom(l);
		xcom(r);
		indexshift(n);
		break;

	case OMUL:
	case OLMUL:
		xcom(l);
		xcom(r);
		g = vlog(l);
		if(g >= 0) {
			n->left = r;
			n->right = l;
			l = r;
			r = n->right;
		}
		g = vlog(r);
		if(g >= 0) {
			n->op = OASHL;
			r->vconst = g;
			r->type = types[TINT];
			indexshift(n);
			break;
		}
commute(n);
		break;

	case OASLDIV:
		xcom(l);
		xcom(r);
		g = vlog(r);
		if(g >= 0) {
			n->op = OASLSHR;
			r->vconst = g;
			r->type = types[TINT];
		}
		break;

	case OLDIV:
		xcom(l);
		xcom(r);
		g = vlog(r);
		if(g >= 0) {
			n->op = OLSHR;
			r->vconst = g;
			r->type = types[TINT];
			indexshift(n);
			break;
		}
		break;

	case OASLMOD:
		xcom(l);
		xcom(r);
		g = vlog(r);
		if(g >= 0) {
			n->op = OASAND;
			r->vconst--;
		}
		break;

	case OLMOD:
		xcom(l);
		xcom(r);
		g = vlog(r);
		if(g >= 0) {
			n->op = OAND;
			r->vconst--;
		}
		break;

	case OASMUL:
	case OASLMUL:
		xcom(l);
		xcom(r);
		g = vlog(r);
		if(g >= 0) {
			n->op = OASASHL;
			r->vconst = g;
		}
		break;

	case OLSHR:
	case OASHR:
		xcom(l);
		xcom(r);
		indexshift(n);
		break;

	default:
		if(l != Z)
			xcom(l);
		if(r != Z)
			xcom(r);
		break;
	}
brk:
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

	case OLMOD:
	case OMOD:
	case OLMUL:
	case OLDIV:
	case OMUL:
	case ODIV:
	case OASLMUL:
	case OASLDIV:
	case OASLMOD:
	case OASMUL:
	case OASDIV:
	case OASMOD:
		if(r->complex >= l->complex) {
			n->complex = l->complex + 3;
			if(r->complex > n->complex)
				n->complex = r->complex;
		} else {
			n->complex = r->complex + 3;
			if(l->complex > n->complex)
				n->complex = l->complex;
		}
		break;

	case OLSHR:
	case OASHL:
	case OASHR:
	case OASLSHR:
	case OASASHL:
	case OASASHR:
		if(r->complex >= l->complex) {
			n->complex = l->complex + 2;
			if(r->complex > n->complex)
				n->complex = r->complex;
		} else {
			n->complex = r->complex + 2;
			if(l->complex > n->complex)
				n->complex = l->complex;
		}
		break;

	case OADD:
	case OXOR:
	case OAND:
	case OOR:
		/*
		 * immediate operators, make const on right
		 */
		if(l->op == OCONST) {
			n->left = r;
			n->right = l;
		}
		break;

	case OEQ:
	case ONE:
	case OLE:
	case OLT:
	case OGE:
	case OGT:
	case OHI:
	case OHS:
	case OLO:
	case OLS:
		/*
		 * compare operators, make const on left
		 */
		if(r->op == OCONST) {
			n->left = r;
			n->right = l;
			n->op = invrel[relindex(n->op)];
		}
		break;
	}
}

void
indx(Node *n)
{
	Node *l, *r;

	if(debug['x'])
		prtree(n, "indx");

	l = n->left;
	r = n->right;
	if(l->addable == 1 || l->addable == 13 || r->complex > l->complex) {
		n->right = l;
		n->left = r;
		l = r;
		r = n->right;
	}
	if(l->addable != 7) {
		idx.regtree = l;
		idx.scale = 1;
	} else
	if(l->right->addable == 20) {
		idx.regtree = l->left;
		idx.scale = 1 << l->right->vconst;
	} else
	if(l->left->addable == 20) {
		idx.regtree = l->right;
		idx.scale = 1 << l->left->vconst;
	} else
		diag(n, "bad index");

	idx.basetree = r;
	if(debug['x']) {
		print("scale = %d\n", idx.scale);
		prtree(idx.regtree, "index");
		prtree(idx.basetree, "base");
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
	if(n->type != T) {
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
			cgen64(b, Z);
			return 0;
		}
		bool64(n);
		boolgen(n, 1, Z);
	} else
		gbranch(OGOTO);
	return 0;
}
