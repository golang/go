// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include "go.h"

/*
 * look for
 *	unsafe.Sizeof
 *	unsafe.Offsetof
 *	unsafe.Alignof
 * rewrite with a constant
 */
Node*
unsafenmagic(Node *nn)
{
	Node *r, *n;
	Sym *s;
	Type *t, *tr;
	long v;
	Val val;
	Node *fn;
	NodeList *args;

	fn = nn->left;
	args = nn->list;

	if(safemode || fn == N || fn->op != ONAME || (s = fn->sym) == S)
		goto no;
	if(s->pkg != unsafepkg)
		goto no;

	if(args == nil) {
		yyerror("missing argument for %S", s);
		goto no;
	}
	r = args->n;

	if(strcmp(s->name, "Sizeof") == 0) {
		typecheck(&r, Erv);
		defaultlit(&r, T);
		tr = r->type;
		if(tr == T)
			goto bad;
		dowidth(tr);
		v = tr->width;
		goto yes;
	}
	if(strcmp(s->name, "Offsetof") == 0) {
		typecheck(&r, Erv);
		if(r->op != ODOT && r->op != ODOTPTR)
			goto bad;
		typecheck(&r, Erv);
		v = r->xoffset;
		goto yes;
	}
	if(strcmp(s->name, "Alignof") == 0) {
		typecheck(&r, Erv);
		defaultlit(&r, T);
		tr = r->type;
		if(tr == T)
			goto bad;

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

bad:
	yyerror("invalid expression %N", nn);
	v = 0;
	goto ret;

yes:
	if(args->next != nil)
		yyerror("extra arguments for %S", s);
ret:
	// any side effects disappear; ignore init
	val.ctype = CTINT;
	val.u.xval = mal(sizeof(*n->val.u.xval));
	mpmovecfix(val.u.xval, v);
	n = nod(OLITERAL, N, N);
	n->orig = nn;
	n->val = val;
	n->type = types[TUINTPTR];
	nn->type = types[TUINTPTR];
	return n;
}

int
isunsafebuiltin(Node *n)
{
	if(n == N || n->op != ONAME || n->sym == S || n->sym->pkg != unsafepkg)
		return 0;
	if(strcmp(n->sym->name, "Sizeof") == 0)
		return 1;
	if(strcmp(n->sym->name, "Offsetof") == 0)
		return 1;
	if(strcmp(n->sym->name, "Alignof") == 0)
		return 1;
	return 0;
}
