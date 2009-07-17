// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"

static struct
{
	NodeList*	list;
	Node*	mapname;
	Type*	type;
} xxx;

enum
{
	TC_xxx,

	TC_unknown,		// class
	TC_struct,
	TC_array,
	TC_slice,
	TC_map,

	TS_start,		// state
	TS_middle,
	TS_end,
};

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

static int
typeclass(Type *t)
{
	if(t != T)
	switch(t->etype) {
	case TSTRUCT:
		return TC_struct;
	case TARRAY:
		if(t->bound >= 0)
			return TC_array;
		return TC_slice;
	case TMAP:
		return TC_map;
	}
	return TC_unknown;
}

void
initlin(NodeList *l)
{
	Node *n;

	for(; l; l=l->next) {
		n = l->n;
		initlin(n->ninit);
		n->ninit = nil;
		xxx.list = list(xxx.list, n);
		switch(n->op) {
		default:
			print("o = %O\n", n->op);
			break;

		case OCALL:
			// call to mapassign1
		case OAS:
			break;
		}
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

Node*
findarg(Node *n, char *arg, char *fn)
{
	Node *a;
	NodeList *l;

	if(n == N || n->op != OCALL ||
	   n->left == N || n->left->sym == S ||
	   strcmp(n->left->sym->name, fn) != 0)
		return N;

	for(l=n->list; l; l=l->next) {
		a = l->n;
		if(a->op == OAS &&
		   a->left != N && a->right != N &&
		   a->left->op == OINDREG &&
		   a->left->sym != S)
			if(strcmp(a->left->sym->name, arg) == 0)
				return a->right;
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

	while(n->op == OCONVNOP)
		n = n->left;

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
	Node *index, *val, *key, *a, *b, *r;

	// pull all the primatives
	key = findarg(n, "key", "mapassign1");
	if(key == N)
		return N;
	val = findarg(n, "val", "mapassign1");
	if(val == N)
		return N;
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

	r = liststmt(list(list1(a), b));
	walkstmt(r);
	return r;
}

// for a copy out reference, A = B,
// look through the whole structure
// and substitute references of B to A.
// some rewrite goes on also.
void
initsub(Node *n, Node *nam)
{
	Node *r, *w, *c;
	NodeList *l;
	int class, state;

	// we could probably get a little more
	// out of this if we allow minimal simple
	// expression on the right (eg OADDR-ONAME)
	if(n->op != ONAME)
		return;

	class = typeclass(nam->type);
	state = TS_start;

	switch(class) {
	case TC_struct:
		goto str;
	case TC_array:
		goto ary;
	case TC_slice:
		goto sli;
	case TC_map:
		goto map;
	}
	return;

str:
	for(l=xxx.list; l; l=l->next) {
		r = l->n;
		if(r->op != OAS && r->op != OEMPTY)
			continue;

		// optional first usage "nam = N"
		if(r->right == N && sametmp(r->left, nam)) {
			if(state != TS_start) {
				dump("", r);
				fatal("initsub: str-first and state=%d", state);
			}
			state = TS_middle;
			r->op = OEMPTY;
			continue;
		}

		// last usage "n = nam"
		if(r->left != N && sametmp(r->right, nam)) {
			if(state == TS_end) {
				dump("", r);
				fatal("initsub: str-last and state=%d", state);
			}
			state = TS_end;
			r->op = OEMPTY;
			continue;
		}

		// middle usage "(nam DOT name) AS expr"
		if(r->left->op != ODOT || !sametmp(r->left->left, nam))
			continue;
		if(state == TS_end) {
			dump("", r);
			fatal("initsub: str-middle and state=%d", state);
		}
		state = TS_middle;
		r->left->left = n;
	}
	return;

ary:
	for(l=xxx.list; l; l=l->next) {
		r = l->n;
		if(r->op != OAS && r->op != OEMPTY)
			continue;

		// optional first usage "nam = N"
		if(r->right == N && sametmp(r->left, nam)) {
			if(state != TS_start) {
				dump("", r);
				fatal("initsub: ary-first and state=%d", state);
			}
			state = TS_middle;
			r->op = OEMPTY;
			continue;
		}

		// last usage "n = nam"
		if(r->left != N && sametmp(r->right, nam)) {
			if(state == TS_end) {
				dump("", r);
				fatal("initsub: ary-last and state=%d", state);
			}
			state = TS_end;
			r->op = OEMPTY;
			continue;
		}

		// middle usage "(nam INDEX literal) = expr"
		if(r->left->op != OINDEX || !sametmp(r->left->left, nam))
			continue;
		if(state == TS_end) {
			dump("", r);
			fatal("initsub: ary-middle and state=%d", state);
		}
		state = TS_middle;
		r->left->left = n;
	}
	return;

sli:
	w = N;
	for(l=xxx.list; l; l=l->next) {
		r = l->n;
		if(r->op != OAS && r->op != OEMPTY)
			continue;

		// first usage "nam = (newarray CALL args)"
		if(r->right != N && sametmp(r->left, nam)) {
			w = slicerewrite(r->right);
			if(w == N)
				continue;
			if(state != TS_start) {
				dump("", r);
				fatal("initsub: ary-first and state=%d", state);
			}
			state = TS_middle;
			r->right = w;
			r->left = n;
			continue;
		}

		// last usage "n = nam"
		if(r->left != N && sametmp(r->right, nam)) {
			if(state != TS_middle) {
				dump("", r);
				fatal("initsub: ary-last and state=%d", state);
			}
			state = TS_end;
			r->op = OEMPTY;
			continue;
		}

		// middle usage "(nam INDEX literal) = expr"
		if(r->left->op != OINDEX || !sametmp(r->left->left, nam))
			continue;
		if(state != TS_middle) {
			dump("", r);
			fatal("initsub: ary-middle and state=%d", state);
		}
		state = TS_middle;
		r->left->left = w->left;
	}
	return;

map:
return;
	w = N;
	for(l=xxx.list; l; l=l->next) {
		r = l->n;
		if(r->op == OCALL) {
			// middle usage "(CALL mapassign1 key, val, map)"
			c = mapindex(r);
			if(c == N)
				continue;
			state = TS_middle;
			*r = *c;
			continue;
		}
		if(r->op != OAS && r->op != OEMPTY)
			continue;

		// first usage "nam = (newmap CALL args)"
		if(r->right != N && sametmp(r->left, nam)) {
			w = maprewrite(r->right);
			if(w == N)
				continue;
			if(state != TS_start) {
				dump("", r);
				fatal("initsub: ary-first and state=%d", state);
			}
			state = TS_middle;
			r->right = w;
			r->left = n;
			continue;
		}

		// last usage "n = nam"
		if(r->left != N && sametmp(r->right, nam)) {
			if(state != TS_middle) {
				dump("", r);
				fatal("initsub: ary-last and state=%d", state);
			}
			state = TS_end;
			r->op = OEMPTY;
			continue;
		}
	}
	return;

}

NodeList*
initfix(NodeList *l)
{
	Node *r;

	xxx.list = nil;
	initlin(l);

if(0)
return xxx.list;

	// look for the copy-out reference
	for(l=xxx.list; l; l=l->next) {
		r = l->n;
		if(r->op == OAS)
		if(inittmp(r->right))
			initsub(r->left, r->right);
	}
	return xxx.list;
}
