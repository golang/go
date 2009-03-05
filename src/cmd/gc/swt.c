// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"

/*
 * walktype
 */
Type*
sw0(Node *c, Type *place)
{
	Node *r;

	if(c == N)
		return T;
	if(c->op != OAS) {
		walktype(c, Erv);
		return T;
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
	return T;

bad:
	yyerror("inappropriate assignment in a case statement");
	return T;
}

/*
 * return the first type
 */
Type*
sw1(Node *c, Type *place)
{
	if(place == T)
		return c->type;
	return place;
}

/*
 * return a suitable type
 */
Type*
sw2(Node *c, Type *place)
{
	return types[TINT];	// botch
}

/*
 * check that switch type
 * is compat with all the cases
 */
Type*
sw3(Node *c, Type *place)
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
walkcases(Node *sw, Type*(*call)(Node*, Type*))
{
	Iter save;
	Node *n;
	Type *place;
	int32 lno;

	lno = setlineno(sw);
	place = call(sw->ntest, T);

	n = listfirst(&save, &sw->nbody->left);
	if(n->op == OEMPTY)
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
		place = call(n->left, place);
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
prepsw(Node *sw)
{
	Iter save;
	Node *name, *cas;
	Node *t, *a;
	int bool;

	bool = 0;
	if(whatis(sw->ntest) == Wlitbool) {
		bool = 1;		// true
		if(sw->ntest->val.u.xval == 0)
			bool = 2;	// false
	}

	cas = N;
	name = N;
	if(bool == 0) {
		name = nod(OXXX, N, N);
		tempname(name, sw->ntest->type);
		cas = nod(OAS, name, sw->ntest);
	}

	t = listfirst(&save, &sw->nbody->left);

loop:
	if(t == N) {
		sw->nbody->left = rev(cas);
		walkstate(sw->nbody->left);
//dump("case", sw->nbody->left);
		return;
	}

	if(t->left == N) {
		cas = list(cas, t->right);		// goto default
		t = listnext(&save);
		goto loop;
	}

	a = nod(OIF, N, N);
	a->nbody = t->right;				// then goto l

	switch(bool) {
	default:
		// not bool const
		a->ntest = nod(OEQ, name, t->left);	// if name == val
		break;

	case 1:
		// bool true
		a->ntest = t->left;			// if val
		break;

	case 2:
		// bool false
		a->ntest = nod(ONOT, t->left, N);	// if !val
		break;
	}
	cas = list(cas, a);

	t = listnext(&save);
	goto loop;
}

void
walkswitch(Node *n)
{
	Type *t;

	casebody(n);
	if(n->ntest == N)
		n->ntest = booltrue;

	walkstate(n->ninit);
	walktype(n->ntest, Erv);
	walkstate(n->nbody);

	// walktype
	walkcases(n, sw0);

	// find common type
	t = n->ntest->type;
	if(t == T)
		t = walkcases(n, sw1);

	// if that fails pick a type
	if(t == T)
		t = walkcases(n, sw2);

	// set the type on all literals
	if(t != T) {
		walkcases(n, sw3);
		convlit(n->ntest, t);
		prepsw(n);
	}
}
