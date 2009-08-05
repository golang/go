// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * range
 */

#include "go.h"

void
typecheckrange(Node *n)
{
	int op, et;
	Type *t, *t1, *t2;
	Node *v1, *v2;
	NodeList *ll;

	// delicate little dance.  see typecheckas2
	for(ll=n->list; ll; ll=ll->next)
		if(ll->n->defn != n)
			typecheck(&ll->n, Erv);

	typecheck(&n->right, Erv);
	if((t = n->right->type) == T)
		goto out;
	n->type = t;

	switch(t->etype) {
	default:
		yyerror("cannot range over %+N", n->right);
		goto out;

	case TARRAY:
		t1 = types[TINT];
		t2 = t->type;
		break;

	case TMAP:
		t1 = t->down;
		t2 = t->type;
		break;

	case TCHAN:
		t1 = t->type;
		t2 = nil;
		if(count(n->list) == 2)
			goto toomany;
		break;

	case TSTRING:
		t1 = types[TINT];
		t2 = types[TINT];
		break;
	}

	if(count(n->list) > 2) {
	toomany:
		yyerror("too many variables in range");
	}

	v1 = n->list->n;
	v2 = N;
	if(n->list->next)
		v2 = n->list->next->n;

	if(v1->defn == n)
		v1->type = t1;
	else if(v1->type != T && checkconv(t1, v1->type, 0, &op, &et) < 0)
		yyerror("cannot assign type %T to %+N", t1, v1);
	if(v2) {
		if(v2->defn == n)
			v2->type = t2;
		else if(v2->type != T && checkconv(t2, v2->type, 0, &op, &et) < 0)
			yyerror("cannot assign type %T to %+N", t1, v1);
	}

out:
	typechecklist(n->nbody, Etop);

	// second half of dance
	n->typecheck = 1;
	for(ll=n->list; ll; ll=ll->next)
		if(ll->n->typecheck == 0)
			typecheck(&ll->n, Erv);
}

void
walkrange(Node *n)
{
	Node *ohv1, *hv1, *hv2;	// hidden (old) val 1, 2
	Node *ha, *hit;	// hidden aggregate, iterator
	Node *a, *v1, *v2;	// not hidden aggregate, val 1, 2
	Node *fn;
	NodeList *body, *init;
	Type *th, *t;

	t = n->type;
	init = nil;

	a = n->right;
	if(t->etype == TSTRING && !eqtype(t, types[TSTRING])) {
		a = nod(OCONV, n->right, N);
		a->type = types[TSTRING];
	}
	ha = nod(OXXX, N, N);
	tempname(ha, a->type);
	init = list(init, nod(OAS, ha, a));

	v1 = n->list->n;
	hv1 = N;

	v2 = N;
	if(n->list->next)
		v2 = n->list->next->n;
	hv2 = N;

	switch(t->etype) {
	default:
		fatal("walkrange");

	case TARRAY:
		hv1 = nod(OXXX, N, n);
		tempname(hv1, v1->type);

		init = list(init, nod(OAS, hv1, N));
		n->ntest = nod(OLT, hv1, nod(OLEN, ha, N));
		n->nincr = nod(OASOP, hv1, nodintconst(1));
		n->nincr->etype = OADD;
		body = list1(nod(OAS, v1, hv1));
		if(v2)
			body = list(body, nod(OAS, v2, nod(OINDEX, ha, hv1)));
		break;

	case TMAP:
		th = typ(TARRAY);
		th->type = ptrto(types[TUINT8]);
		th->bound = (sizeof(struct Hiter) + widthptr - 1) / widthptr;
		hit = nod(OXXX, N, N);
		tempname(hit, th);

		fn = syslook("mapiterinit", 1);
		argtype(fn, t->down);
		argtype(fn, t->type);
		argtype(fn, th);
		init = list(init, mkcall1(fn, T, nil, ha, nod(OADDR, hit, N)));
		n->ntest = nod(ONE, nod(OINDEX, hit, nodintconst(0)), nodnil());

		fn = syslook("mapiternext", 1);
		argtype(fn, th);
		n->nincr = mkcall1(fn, T, nil, nod(OADDR, hit, N));

		if(v2 == N) {
			fn = syslook("mapiter1", 1);
			argtype(fn, th);
			argtype(fn, t->down);
			a = nod(OAS, v1, mkcall1(fn, t->down, nil, nod(OADDR, hit, N)));
		} else {
			fn = syslook("mapiter2", 1);
			argtype(fn, th);
			argtype(fn, t->down);
			argtype(fn, t->type);
			a = nod(OAS2, N, N);
			a->list = list(list1(v1), v2);
			a->rlist = list1(mkcall1(fn, getoutargx(fn->type), nil, nod(OADDR, hit, N)));
		}
		body = list1(a);
		break;

	case TCHAN:
		hv1 = nod(OXXX, N, n);
		tempname(hv1, v1->type);

		n->ntest = nod(ONOT, nod(OCLOSED, ha, N), N);
		n->ntest->ninit = list1(nod(OAS, hv1, nod(ORECV, ha, N)));
		body = list1(nod(OAS, v1, hv1));
		break;

	case TSTRING:
		ohv1 = nod(OXXX, N, N);
		tempname(ohv1, types[TINT]);

		hv1 = nod(OXXX, N, N);
		tempname(hv1, types[TINT]);
		init = list(init, nod(OAS, hv1, N));

		if(v2 == N)
			a = nod(OAS, hv1, mkcall("stringiter", types[TINT], nil, ha, hv1));
		else {
			hv2 = nod(OXXX, N, N);
			tempname(hv2, types[TINT]);
			a = nod(OAS2, N, N);
			a->list = list(list1(hv1), hv2);
			fn = syslook("stringiter2", 0);
			a->rlist = list1(mkcall1(fn, getoutargx(fn->type), nil, ha, hv1));
		}
		n->ntest = nod(ONE, hv1, nodintconst(0));
		n->ntest->ninit = list(list1(nod(OAS, ohv1, hv1)), a);

		body = list1(nod(OAS, v1, ohv1));
		if(v2 != N)
			body = list(body, nod(OAS, v2, hv2));
		break;
	}

	n->op = OFOR;
	typechecklist(init, Etop);
	n->ninit = concat(n->ninit, init);
	typechecklist(n->ntest->ninit, Etop);
	typecheck(&n->ntest, Erv);
	typecheck(&n->nincr, Etop);
	typechecklist(body, Etop);
	n->nbody = concat(body, n->nbody);
	walkstmt(&n);
}

