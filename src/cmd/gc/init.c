// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go.h"

/*
 * a function named init is a special case.
 * it is called by the initialization before
 * main is run. to make it unique within a
 * package and also uncallable, the name,
 * normally "pkg.init", is altered to "pkg.init·filename".
 */
Node*
renameinit(Node *n)
{
	Sym *s;

	s = n->sym;
	if(s == S)
		return n;
	if(strcmp(s->name, "init") != 0)
		return n;

	snprint(namebuf, sizeof(namebuf), "init·%s", filename);
	s = lookup(namebuf);
	return newname(s);
}

/*
 * hand-craft the following initialization code
 *	var initdone·<file> uint8 			(1)
 *	func	Init·<file>()				(2)
 *		if initdone·<file> {			(3)
 *			if initdone·<file> == 2		(4)
 *				return
 *			throw();			(5)
 *		}
 *		initdone.<file>++;			(6)
 *		// over all matching imported symbols
 *			<pkg>.init·<file>()		(7)
 *		{ <init stmts> }			(8)
 *		init·<file>()	// if any		(9)
 *		initdone.<file>++;			(10)
 *		return					(11)
 *	}
 */
int
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
		default:
			return 1;
		}
	}

	// is this main
	if(strcmp(package, "main") == 0)
		return 1;

	// is there an explicit init function
	snprint(namebuf, sizeof(namebuf), "init·%s", filename);
	s = lookup(namebuf);
	if(s->def != N)
		return 1;

	// are there any imported init functions
	for(h=0; h<NHASH; h++)
	for(s = hash[h]; s != S; s = s->link) {
		if(s->name[0] != 'I' || strncmp(s->name, "Init·", 6) != 0)
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
	Node *gatevar;
	Node *a, *b, *fn;
	NodeList *r;
	uint32 h;
	Sym *s, *initsym;

	if(strcmp(package, "PACKAGE") == 0) {
		// sys.go or unsafe.go during compiler build
		return;
	}

	n = initfix(n);
	if(!anyinit(n))
		return;

	r = nil;

	// (1)
	snprint(namebuf, sizeof(namebuf), "initdone·%s", filename);
	gatevar = newname(lookup(namebuf));
	addvar(gatevar, types[TUINT8], PEXTERN);

	// (2)

	maxarg = 0;

	snprint(namebuf, sizeof(namebuf), "Init·%s", filename);

	// this is a botch since we need a known name to
	// call the top level init function out of rt0
	if(strcmp(package, "main") == 0)
		snprint(namebuf, sizeof(namebuf), "init");

	fn = nod(ODCLFUNC, N, N);
	initsym = lookup(namebuf);
	fn->nname = newname(initsym);
	fn->nname->ntype = nod(OTFUNC, N, N);
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
	a = nod(OASOP, gatevar, nodintconst(1));
	a->etype = OADD;
	r = list(r, a);

	// (7)
	for(h=0; h<NHASH; h++)
	for(s = hash[h]; s != S; s = s->link) {
		if(s->name[0] != 'I' || strncmp(s->name, "Init·", 6) != 0)
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
	r = concat(r, initfix(n));

	// (9)
	// could check that it is fn of no args/returns
	snprint(namebuf, sizeof(namebuf), "init·%s", filename);
	s = lookup(namebuf);
	if(s->def != N) {
		a = nod(OCALL, s->def, N);
		r = list(r, a);
	}

	// (10)
	a = nod(OASOP, gatevar, nodintconst(1));
	a->etype = OADD;
	r = list(r, a);

	// (11)
	a = nod(ORETURN, N, N);
	r = list(r, a);

	exportsym(fn->nname);

	fn->nbody = r;

//dump("b", fn);
//dump("r", fn->nbody);

	initflag = 1;	// flag for loader static initialization
	funcbody(fn);
	typecheck(&fn, Etop);
	funccompile(fn);
	initflag = 0;
}


