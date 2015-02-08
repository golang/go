// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include "go.h"

/*
 * a function named init is a special case.
 * it is called by the initialization before
 * main is run. to make it unique within a
 * package and also uncallable, the name,
 * normally "pkg.init", is altered to "pkg.init·1".
 */
Sym*
renameinit(void)
{
	static int initgen;

	snprint(namebuf, sizeof(namebuf), "init·%d", ++initgen);
	return lookup(namebuf);
}

/*
 * hand-craft the following initialization code
 *	var initdone· uint8 				(1)
 *	func init()					(2)
 *		if initdone· != 0 {			(3)
 *			if initdone· == 2		(4)
 *				return
 *			throw();			(5)
 *		}
 *		initdone· = 1;				(6)
 *		// over all matching imported symbols
 *			<pkg>.init()			(7)
 *		{ <init stmts> }			(8)
 *		init·<n>() // if any			(9)
 *		initdone· = 2;				(10)
 *		return					(11)
 *	}
 */
static int
anyinit(NodeList *n)
{
	uint32 h;
	Sym *s;
	NodeList *l;

	// are there any interesting init statements
	for(l=n; l; l=l->next) {
		switch(l->n->op) {
		case ODCLFUNC:
		case ODCLCONST:
		case ODCLTYPE:
		case OEMPTY:
			break;
		case OAS:
			if(isblank(l->n->left) && candiscard(l->n->right))
				break;
			// fall through
		default:
			return 1;
		}
	}

	// is this main
	if(strcmp(localpkg->name, "main") == 0)
		return 1;

	// is there an explicit init function
	snprint(namebuf, sizeof(namebuf), "init·1");
	s = lookup(namebuf);
	if(s->def != N)
		return 1;

	// are there any imported init functions
	for(h=0; h<NHASH; h++)
	for(s = hash[h]; s != S; s = s->link) {
		if(s->name[0] != 'i' || strcmp(s->name, "init") != 0)
			continue;
		if(s->def == N)
			continue;
		return 1;
	}

	// then none
	return 0;
}

void
fninit(NodeList *n)
{
	int i;
	Node *gatevar;
	Node *a, *b, *fn;
	NodeList *r;
	uint32 h;
	Sym *s, *initsym;

	if(debug['A']) {
		// sys.go or unsafe.go during compiler build
		return;
	}

	n = initfix(n);
	if(!anyinit(n))
		return;

	r = nil;

	// (1)
	snprint(namebuf, sizeof(namebuf), "initdone·");
	gatevar = newname(lookup(namebuf));
	addvar(gatevar, types[TUINT8], PEXTERN);

	// (2)
	maxarg = 0;
	snprint(namebuf, sizeof(namebuf), "init");

	fn = nod(ODCLFUNC, N, N);
	initsym = lookup(namebuf);
	fn->nname = newname(initsym);
	fn->nname->defn = fn;
	fn->nname->ntype = nod(OTFUNC, N, N);
	declare(fn->nname, PFUNC);
	funchdr(fn);

	// (3)
	a = nod(OIF, N, N);
	a->ntest = nod(ONE, gatevar, nodintconst(0));
	r = list(r, a);

	// (4)
	b = nod(OIF, N, N);
	b->ntest = nod(OEQ, gatevar, nodintconst(2));
	b->nbody = list1(nod(ORETURN, N, N));
	a->nbody = list1(b);

	// (5)
	b = syslook("throwinit", 0);
	b = nod(OCALL, b, N);
	a->nbody = list(a->nbody, b);

	// (6)
	a = nod(OAS, gatevar, nodintconst(1));
	r = list(r, a);

	// (7)
	for(h=0; h<NHASH; h++)
	for(s = hash[h]; s != S; s = s->link) {
		if(s->name[0] != 'i' || strcmp(s->name, "init") != 0)
			continue;
		if(s->def == N)
			continue;
		if(s == initsym)
			continue;

		// could check that it is fn of no args/returns
		a = nod(OCALL, s->def, N);
		r = list(r, a);
	}

	// (8)
	r = concat(r, n);

	// (9)
	// could check that it is fn of no args/returns
	for(i=1;; i++) {
		snprint(namebuf, sizeof(namebuf), "init·%d", i);
		s = lookup(namebuf);
		if(s->def == N)
			break;
		a = nod(OCALL, s->def, N);
		r = list(r, a);
	}

	// (10)
	a = nod(OAS, gatevar, nodintconst(2));
	r = list(r, a);

	// (11)
	a = nod(ORETURN, N, N);
	r = list(r, a);
	exportsym(fn->nname);

	fn->nbody = r;
	funcbody(fn);

	curfn = fn;
	typecheck(&fn, Etop);
	typechecklist(r, Etop);
	curfn = nil;
	funccompile(fn);
}
