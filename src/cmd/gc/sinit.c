// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"

static struct
{
	Node*	list;
	Node*	mapname;
	Type*	type;
} xxx;

/*
 * the init code (thru initfix) reformats the
 *	var = ...
 * statements, rewriting the automatic
 * variables with the static variables.
 * this allows the code generator to
 * generate DATA statements instead
 * of assignment statements.
 * it is quadradic, may need to change.
 * it is extremely fragile knowing exactly
 * how the code from (struct|array|map)lit
 * will look. ideally the lit routines could
 * write the code in this form, but ...
 */

void
initlin(Node* n)
{
	if(n == N)
		return;
	initlin(n->ninit);
	switch(n->op) {
	default:
		print("o = %O\n", n->op);
		n->ninit = N;
		xxx.list = list(xxx.list, n);
		break;

	case OCALL:
		// call to mapassign1
		n->ninit = N;
		xxx.list = list(xxx.list, n);
		break;

	case OAS:
		n->ninit = N;
		xxx.list = list(xxx.list, n);
		break;

	case OLIST:
		initlin(n->left);
		initlin(n->right);
		break;
	}
}

int
inittmp(Node *n)
{
	if(n != N)
	if(n->op == ONAME)
	if(n->sym != S)
	if(n->class == PAUTO)
	if(memcmp(n->sym->name, "autotmp_", 8) == 0)
		return 1;
	return 0;
}

int
sametmp(Node *n1, Node *n2)
{
	if(inittmp(n1))
	if(n1->xoffset == n2->xoffset)
		return 1;
	return 0;
}

int
indsametmp(Node *n1, Node *n2)
{
	if(n1->op == OIND)
	if(inittmp(n1->left))
	if(n1->left->xoffset == n2->xoffset)
		return 1;
	return 0;
}

Node*
findarg(Node *n, char *arg, char *fn)
{
	Iter param;
	Node *a;

	if(n == N || n->op != OCALL ||
	   n->left == N || n->left->sym == S ||
	   strcmp(n->left->sym->name, fn) != 0)
		return N;

	a = listfirst(&param, &n->right);
	while(a != N) {
		if(a->op == OAS &&
		   a->left != N && a->right != N &&
		   a->left->op == OINDREG &&
		   a->left->sym != S)
			if(strcmp(a->left->sym->name, arg) == 0)
				return a->right;
		a = listnext(&param);
	}
	return N;
}

Node*
slicerewrite(Node *n)
{
	Node *nel;
	Type *t;
	int b;
	Node *a;

	// call to newarray - find nel argument
	nel = findarg(n, "nel", "newarray");
	if(nel == N || !isslice(n->type))
		goto no;

	b = mpgetfix(nel->val.u.xval);
	t = shallow(n->type);
	t->bound = b;

	// special hack for zero-size array
	// invent an l-value to point at
	if(b == 0)
		a = staticname(types[TBOOL]);
	else
		a = staticname(t);

	a = nod(OCOMPSLICE, a, N);
	a->type = n->type;
	return a;

no:
	return N;
}

Node*
maprewrite(Node *n)
{
	Node *nel;
	Type *ta, *tb;
	Node *a;

	// call to newarray - find nel argument
	nel = findarg(n, "hint", "newmap");
	if(nel == N)
		goto no;
	ta = n->type;
	if(ta->etype != TMAP)
		goto no;

	// create a new type from map[index]value
	//	[0]struct { a index; b value) }

	tb = typ(TFIELD);
	tb->type = ta->down;
	tb->sym = lookup("key");
	tb->nname = newname(tb->sym);
	tb->down = typ(TFIELD);
	tb->down->type = ta->type;
	tb->down->sym = lookup("val");
	tb->down->nname = newname(tb->down->sym);

	ta = typ(TSTRUCT);
	ta->type = tb;

	tb = typ(TARRAY);
	tb->type = ta;
	tb->bound = 0;

	dowidth(tb);

	a = staticname(tb);
	a = nod(OCOMPMAP, a, N);
	a->type = n->type;

	// save stuff for this iteration
	xxx.mapname = a->left;
	xxx.type = tb;

	return a;

no:
	return N;
}

// convert the call to mapassign1
// into static[i].key = k, static[i].val = v
Node*
mapindex(Node *n)
{
	Node *index, *val, *key, *a, *b;

	// pull all the primatives
	key = findarg(n, "key", "mapassign1");
	val = findarg(n, "val", "mapassign1");
	index = nodintconst(xxx.type->bound);
	xxx.type->bound++;
	dowidth(xxx.type);

	// build tree
	a = nod(OINDEX, xxx.mapname, index);
	a = nod(ODOT, a, newname(lookup("key")));
	a = nod(OAS, a, key);

	b = nod(OINDEX, xxx.mapname, index);
	b = nod(ODOT, b, newname(lookup("val")));
	b = nod(OAS, b, val);

	a = nod(OLIST, a, b);
	walktype(a, Etop);

	return a;
}

// for a copy out reference, A = B,
// look through the whole structure
// and substitute references of B to A.
// some rewrite goes on also.
int
initsub(Node *n, Node *nam)
{
	Iter iter;
	Node *r, *w;
	int any;

	any = 0;
	r = listfirst(&iter, &xxx.list);
	while(r != N) {
		switch(r->op) {
		case OAS:
		case OEMPTY:
			if(r->left != N)
			switch(r->left->op) {
			case ONAME:
				if(sametmp(r->left, nam)) {
					any = 1;
					r->left = n;

					w = slicerewrite(r->right);
					if(w != N) {
						n = w->left;	// from now on use fixed array
						r->right = w;
						break;
					}

					w = maprewrite(r->right);
					if(w != N) {
						n = w->left;	// from now on use fixed array
						r->right = w;
						break;
					}
				}
				break;
			case ODOT:
				if(sametmp(r->left->left, nam)) {
					any = 1;
					r->left->left = n;
				}
				if(indsametmp(r->left->left, nam)) {
					any = 1;
					r->left->left->left = n;
				}
				break;
			case OINDEX:
				if(sametmp(r->left->left, nam)) {
					any = 1;
					r->left->left = n;
				}
				if(indsametmp(r->left->left, nam)) {
					any = 1;
					r->left->left->left = n;
				}
				break;
			}
			break;
		case OCALL:
			// call to mapassign1
			// look through the parameters
			w = findarg(r, "hmap", "mapassign1");
			if(w == N)
				break;
			if(sametmp(w, nam)) {
				any = 1;
				*r = *mapindex(r);
			}
			if(indsametmp(w, nam)) {
fatal("indirect map index");
				any = 1;
				w->right->left = n;
			}
			break;
		}
		r = listnext(&iter);
	}
	return any;
}

Node*
initfix(Node* n)
{
	Iter iter;
	Node *r;

//dump("prelin", n);

	xxx.list = N;
	initlin(n);
	xxx.list = rev(xxx.list);
if(1)
return xxx.list;

if(debug['A'])
dump("preinitfix", xxx.list);

	// look for the copy-out reference
	r = listfirst(&iter, &xxx.list);
	while(r != N) {
		if(r->op == OAS)
		if(inittmp(r->right)) {
			if(initsub(r->left, r->right))
				r->op = OEMPTY;
		}
		r = listnext(&iter);
	}
if(debug['A'])
dump("postinitfix", xxx.list);
	return xxx.list;
}
