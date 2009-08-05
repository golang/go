// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go.h"

/*
 * look for
 *	unsafe.Sizeof
 *	unsafe.Offsetof
 * rewrite with a constant
 */
Node*
unsafenmagic(Node *fn, NodeList *args)
{
	Node *r, *n;
	Sym *s;
	Type *t, *tr;
	long v;
	Val val;

	if(fn == N || fn->op != ONAME || (s = fn->sym) == S)
		goto no;
	if(strcmp(s->package, "unsafe") != 0)
		goto no;

	if(args == nil) {
		yyerror("missing argument for %S", s);
		goto no;
	}
	r = args->n;

	n = nod(OLITERAL, N, N);
	if(strcmp(s->name, "Sizeof") == 0) {
		typecheck(&r, Erv);
		tr = r->type;
		if(r->op == OLITERAL && r->val.ctype == CTSTR)
			tr = types[TSTRING];
		if(tr == T)
			goto no;
		v = tr->width;
		goto yes;
	}
	if(strcmp(s->name, "Offsetof") == 0) {
		if(r->op != ODOT && r->op != ODOTPTR)
			goto no;
		typecheck(&r, Erv);
		v = r->xoffset;
		goto yes;
	}
	if(strcmp(s->name, "Alignof") == 0) {
		typecheck(&r, Erv);
		tr = r->type;
		if(r->op == OLITERAL && r->val.ctype == CTSTR)
			tr = types[TSTRING];
		if(tr == T)
			goto no;

		// make struct { byte; T; }
		t = typ(TSTRUCT);
		t->type = typ(TFIELD);
		t->type->type = types[TUINT8];
		t->type->down = typ(TFIELD);
		t->type->down->type = tr;
		// compute struct widths
		dowidth(t);

		// the offset of T is its required alignment
		v = t->type->down->width;
		goto yes;
	}

no:
	return N;

yes:
	if(args->next != nil)
		yyerror("extra arguments for %S", s);
	// any side effects disappear; ignore init
	val.ctype = CTINT;
	val.u.xval = mal(sizeof(*n->val.u.xval));
	mpmovecfix(val.u.xval, v);
	n = nod(OLITERAL, N, N);
	n->val = val;
	n->type = types[TINT];
	return n;
}
