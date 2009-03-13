// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"

enum
{
	Snorm		= 0,
	Strue,
	Sfalse,
	Stype,
};
Node*	binarysw(Node *t, Iter *save, Node *name);

/*
 * walktype
 */
Type*
sw0(Node *c, Type *place, int arg)
{
	Node *r;

	if(c == N)
		return T;
	switch(c->op) {
	default:
		if(arg == Stype) {
			yyerror("inappropriate case for a type switch");
			return T;
		}
		walktype(c, Erv);
		return T;
	case OTYPESW:
		if(arg != Stype)
			yyerror("inappropriate type case");
		return T;
	case OAS:
		break;
	}
	walktype(c->left, Elv);

	r = c->right;
	if(c == N)
		return T;

	switch(r->op) {
	default:
		goto bad;
	case ORECV:
		// <-chan
		walktype(r->left, Erv);
		if(!istype(r->left->type, TCHAN))
			goto bad;
		break;
	case OINDEX:
		// map[e]
		walktype(r->left, Erv);
		if(!istype(r->left->type, TMAP))
			goto bad;
		break;
	case ODOTTYPE:
		// interface.(type)
		walktype(r->left, Erv);
		if(!istype(r->left->type, TINTER))
			goto bad;
		break;
	}
	c->type = types[TBOOL];
	if(arg != Strue)
		goto bad;
	return T;

bad:
	yyerror("inappropriate assignment in a case statement");
	return T;
}

/*
 * return the first type
 */
Type*
sw1(Node *c, Type *place, int arg)
{
	if(place == T)
		return c->type;
	return place;
}

/*
 * return a suitable type
 */
Type*
sw2(Node *c, Type *place, int arg)
{
	return types[TINT];	// botch
}

/*
 * check that switch type
 * is compat with all the cases
 */
Type*
sw3(Node *c, Type *place, int arg)
{
	if(place == T)
		return c->type;
	if(c->type == T)
		c->type = place;
	convlit(c, place);
	if(!ascompat(place, c->type))
		badtype(OSWITCH, place, c->type);
	return place;
}

/*
 * over all cases, call paramenter function.
 * four passes of these are used to allocate
 * types to cases and switch
 */
Type*
walkcases(Node *sw, Type*(*call)(Node*, Type*, int arg), int arg)
{
	Iter save;
	Node *n;
	Type *place;
	int32 lno;

	lno = setlineno(sw);
	place = call(sw->ntest, T, arg);

	n = listfirst(&save, &sw->nbody->left);
	if(n == N || n->op == OEMPTY)
		return T;

loop:
	if(n == N) {
		lineno = lno;
		return place;
	}

	if(n->op != OCASE)
		fatal("walkcases: not case %O\n", n->op);

	if(n->left != N) {
		setlineno(n->left);
		place = call(n->left, place, arg);
	}
	n = listnext(&save);
	goto loop;
}

Node*
newlabel()
{
	static int label;

	label++;
	snprint(namebuf, sizeof(namebuf), "%.6d", label);
	return newname(lookup(namebuf));
}

/*
 * build separate list of statements and cases
 * make labels between cases and statements
 * deal with fallthrough, break, unreachable statements
 */
void
casebody(Node *sw)
{
	Iter save;
	Node *os, *oc, *t, *c;
	Node *cas, *stat, *def;
	Node *go, *br;
	int32 lno;

	lno = setlineno(sw);
	t = listfirst(&save, &sw->nbody);
	if(t == N || t->op == OEMPTY) {
		sw->nbody = nod(OLIST, N, N);
		return;
	}

	cas = N;	// cases
	stat = N;	// statements
	def = N;	// defaults
	os = N;		// last statement
	oc = N;		// last case
	br = nod(OBREAK, N, N);

loop:
	if(t == N) {
		if(oc == N && os != N)
			yyerror("first switch statement must be a case");

		stat = list(stat, br);
		cas = list(cas, def);

		sw->nbody = nod(OLIST, rev(cas), rev(stat));
//dump("case", sw->nbody->left);
//dump("stat", sw->nbody->right);
		lineno = lno;
		return;
	}

	lno = setlineno(t);

	switch(t->op) {
	case OXCASE:
		t->op = OCASE;
		if(oc == N && os != N)
			yyerror("first switch statement must be a case");

		// botch - shouldnt fall thru declaration
		if(os != N && os->op == OXFALL)
			os->op = OFALL;
		else
			stat = list(stat, br);

		go = nod(OGOTO, newlabel(), N);

		c = t->left;
		if(c == N) {
			if(def != N)
				yyerror("more than one default case");

			// reuse original default case
			t->right = go;
			def = t;
		}

		// expand multi-valued cases
		for(; c!=N; c=c->right) {
			if(c->op != OLIST) {
				// reuse original case
				t->left = c;
				t->right = go;
				cas = list(cas, t);
				break;
			}
			cas = list(cas, nod(OCASE, c->left, go));
		}
		stat = list(stat, nod(OLABEL, go->left, N));
		oc = t;
		os = N;
		break;

	default:
		stat = list(stat, t);
		os = t;
		break;
	}
	t = listnext(&save);
	goto loop;
}

/*
 * rebulid case statements into if .. goto
 */
void
exprswitch(Node *sw, int arg)
{
	Iter save;
	Node *name, *bool, *cas;
	Node *t, *a;
//dump("exprswitch before", sw->nbody->left);

	cas = N;
	name = N;
	bool = N;

	if(arg != Strue && arg != Sfalse) {
		name = nod(OXXX, N, N);
		tempname(name, sw->ntest->type);
		cas = nod(OAS, name, sw->ntest);
	}

	t = listfirst(&save, &sw->nbody->left);

loop:
	if(t == N) {
		sw->nbody->left = rev(cas);
//dump("exprswitch after", sw->nbody->left);
		return;
	}

	if(t->left == N) {
		cas = list(cas, t->right);		// goto default
		t = listnext(&save);
		goto loop;
	}

	// pull out the dcl in case this
	// variable is allocated on the heap.
	// this should be done better to prevent
	// multiple (unused) heap allocations per switch.
	if(t->ninit != N && t->ninit->op == ODCL) {
//dump("exprswitch case init", t->ninit);
		cas = list(cas, t->ninit);
		t->ninit = N;
	}

	if(t->left->op == OAS) {
		if(bool == N) {
			bool = nod(OXXX, N, N);
			tempname(bool, types[TBOOL]);
		}
//dump("oas", t);
		t->left->left = nod(OLIST, t->left->left, bool);
		cas = list(cas, t->left);		// v,bool = rhs

		a = nod(OIF, N, N);
		a->nbody = t->right;			// then goto l
		a->ntest = bool;
		if(arg != Strue)
			a->ntest = nod(ONOT, bool, N);
		cas = list(cas, a);			// if bool goto l

		t = listnext(&save);
		goto loop;
	}


	switch(arg) {
	default:
		// not bool const
		a = binarysw(t, &save, name);
		if(a != N)
			break;

		a = nod(OIF, N, N);
		a->ntest = nod(OEQ, name, t->left);	// if name == val
		a->nbody = t->right;			// then goto l
		break;

	case Strue:
		a = nod(OIF, N, N);
		a->ntest = t->left;			// if val
		a->nbody = t->right;			// then goto l
		break;

	case Sfalse:
		a = nod(OIF, N, N);
		a->ntest = nod(ONOT, t->left, N);	// if !val
		a->nbody = t->right;			// then goto l
		break;
	}
	cas = list(cas, a);

	t = listnext(&save);
	goto loop;
}

/*
 * convert switch of the form
 *	switch v := i.(type) { case t1: ..; case t2: ..; }
 * into if statements
 */
void
typeswitch(Node *sw)
{
	Iter save;
	Node *face, *bool, *cas;
	Node *t, *a, *b;

//dump("typeswitch", sw);

	walktype(sw->ntest->right, Erv);
	if(!istype(sw->ntest->right->type, TINTER)) {
		yyerror("type switch must be on an interface");
		return;
	}
	walkcases(sw, sw0, Stype);

	/*
	 * predeclare variables for the interface var
	 * and the boolean var
	 */
	face = nod(OXXX, N, N);
	tempname(face, sw->ntest->right->type);
	cas = nod(OAS, face, sw->ntest->right);

	bool = nod(OXXX, N, N);
	tempname(bool, types[TBOOL]);

	t = listfirst(&save, &sw->nbody->left);

loop:
	if(t == N) {
		sw->nbody->left = rev(cas);
		walkstate(sw->nbody);
//dump("done", sw->nbody->left);
		return;
	}

	if(t->left == N) {
		cas = list(cas, t->right);		// goto default
		t = listnext(&save);
		goto loop;
	}
	if(t->left->op != OTYPESW) {
		t = listnext(&save);
		goto loop;
	}

	// pull out the dcl in case this
	// variable is allocated on the heap.
	// this should be done better to prevent
	// multiple (unused) heap allocations per switch.
	// not worth doing now -- make a binary search
	// on contents of signature instead.
	if(t->ninit != N && t->ninit->op == ODCL) {
//dump("typeswitch case init", t->ninit);
		cas = list(cas, t->ninit);
		t->ninit = N;
	}

	a = t->left->left;		// var
	a = nod(OLIST, a, bool);	// var,bool

	b = nod(ODOTTYPE, face, N);
	b->type = t->left->left->type;	// interface.(type)

	a = nod(OAS, a, b);		// var,bool = interface.(type)
	cas = list(cas, a);

	a = nod(OIF, N, N);
	a->ntest = bool;
	a->nbody = t->right;		// if bool { goto l }
	cas = list(cas, a);

	t = listnext(&save);
	goto loop;
}

void
walkswitch(Node *sw)
{
	Type *t;
	int arg;

//dump("walkswitch", sw);

	/*
	 * reorder the body into (OLIST, cases, statements)
	 * cases have OGOTO into statements.
	 * both have inserted OBREAK statements
	 */
	walkstate(sw->ninit);
	if(sw->ntest == N)
		sw->ntest = nodbool(1);
	casebody(sw);

	/*
	 * classify the switch test
	 * Strue or Sfalse if the test is a bool constant
	 *	this allows cases to be map/chan/interface assignments
	 *	as well as (boolean) expressions
	 * Stype if the test is v := interface.(type)
	 *	this forces all cases to be types
	 * Snorm otherwise
	 *	all cases are expressions
	 */
	if(sw->ntest->op == OTYPESW) {
		typeswitch(sw);
		return;
	}
	arg = Snorm;
	if(isconst(sw->ntest, CTBOOL)) {
		arg = Strue;
		if(sw->ntest->val.u.xval == 0)
			arg = Sfalse;
	}

	/*
	 * init statement is nothing important
	 */
	walktype(sw->ntest, Erv);
//print("after walkwalks\n");

	/*
	 * pass 0,1,2,3
	 * walk the cases as appropriate for switch type
	 */
	walkcases(sw, sw0, arg);
	t = sw->ntest->type;
	if(t == T)
		t = walkcases(sw, sw1, arg);
	if(t == T)
		t = walkcases(sw, sw2, arg);
	if(t == T)
		return;
	walkcases(sw, sw3, arg);
	convlit(sw->ntest, t);
//print("after walkcases\n");

	/*
	 * convert the switch into OIF statements
	 */
	exprswitch(sw, arg);
	walkstate(sw->nbody);
//print("normal done\n");
}

/*
 * binary search on cases
 */
enum
{
	Ncase	= 4,	// needed to binary search
};

typedef	struct	Case	Case;
struct	Case
{
	Node*	node;		// points at case statement
	Case*	link;		// linked list to link
};
#define	C	((Case*)nil)

int
iscaseconst(Node *t)
{
	if(t == N || t->left == N)
		return 0;
	switch(consttype(t->left)) {
	case CTFLT:
	case CTINT:
	case CTSTR:
		return 1;
	}
	return 0;
}

int
countcase(Node *t, Iter save)
{
	int n;

	// note that the iter is by value,
	// so cases are not really consumed
	for(n=0;; n++) {
		if(!iscaseconst(t))
			return n;
		t = listnext(&save);
	}
}

Case*
csort(Case *l, int(*f)(Case*, Case*))
{
	Case *l1, *l2, *le;

	if(l == C || l->link == C)
		return l;

	l1 = l;
	l2 = l;
	for(;;) {
		l2 = l2->link;
		if(l2 == C)
			break;
		l2 = l2->link;
		if(l2 == C)
			break;
		l1 = l1->link;
	}

	l2 = l1->link;
	l1->link = C;
	l1 = csort(l, f);
	l2 = csort(l2, f);

	/* set up lead element */
	if((*f)(l1, l2) < 0) {
		l = l1;
		l1 = l1->link;
	} else {
		l = l2;
		l2 = l2->link;
	}
	le = l;

	for(;;) {
		if(l1 == C) {
			while(l2) {
				le->link = l2;
				le = l2;
				l2 = l2->link;
			}
			le->link = C;
			break;
		}
		if(l2 == C) {
			while(l1) {
				le->link = l1;
				le = l1;
				l1 = l1->link;
			}
			break;
		}
		if((*f)(l1, l2) < 0) {
			le->link = l1;
			le = l1;
			l1 = l1->link;
		} else {
			le->link = l2;
			le = l2;
			l2 = l2->link;
		}
	}
	le->link = C;
	return l;
}

int
casecmp(Case *c1, Case *c2)
{
	int ct;
	Node *n1, *n2;

	n1 = c1->node->left;
	n2 = c2->node->left;

	ct = n1->val.ctype;
	if(ct != n2->val.ctype)
		fatal("casecmp1");

	switch(ct) {
	case CTFLT:
		return mpcmpfltflt(n1->val.u.fval, n2->val.u.fval);
	case CTINT:
		return mpcmpfixfix(n1->val.u.xval, n2->val.u.xval);
	case CTSTR:
		return cmpslit(n1, n2);
	}

	fatal("casecmp2");
	return 0;
}

Node*
constsw(Case *c0, int ncase, Node *name)
{
	Node *cas, *a;
	Case *c;
	int i, n;

	// small number do sequentially
	if(ncase < Ncase) {
		cas = N;
		for(i=0; i<ncase; i++) {
			a = nod(OIF, N, N);
			a->ntest = nod(OEQ, name, c0->node->left);
			a->nbody = c0->node->right;
			cas = list(cas, a);
			c0 = c0->link;
		}
		return rev(cas);
	}

	// find center and recur
	c = c0;
	n = ncase>>1;
	for(i=0; i<n; i++)
		c = c->link;

	a = nod(OIF, N, N);
	a->ntest = nod(OLE, name, c->node->left);
	a->nbody = constsw(c0, n+1, name);	// include center
	a->nelse = constsw(c->link, ncase-n-1, name);	// exclude center
	return a;
}

Node*
binarysw(Node *t, Iter *save, Node *name)
{
	Case *c, *c1;
	int i, ncase;
	Node *a;

	ncase = countcase(t, *save);
	if(ncase < Ncase)
		return N;

	c = C;
	for(i=1; i<ncase; i++) {
		c1 = mal(sizeof(*c1));
		c1->link = c;
		c1->node = t;
		c = c1;

		t = listnext(save);
	}

	// last one shouldnt consume the iter
	c1 = mal(sizeof(*c1));
	c1->link = c;
	c1->node = t;
	c = c1;

	c = csort(c, casecmp);
	a = constsw(c, ncase, name);
//dump("bin", a);
	return a;
}
