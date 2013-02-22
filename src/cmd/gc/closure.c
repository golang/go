// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * function literals aka closures
 */

#include <u.h>
#include <libc.h>
#include "go.h"

void
closurehdr(Node *ntype)
{
	Node *n, *name, *a, *orig;
	NodeList *l;

	n = nod(OCLOSURE, N, N);
	n->ntype = ntype;
	n->funcdepth = funcdepth;

	funchdr(n);

	// steal ntype's argument names and
	// leave a fresh copy in their place.
	// references to these variables need to
	// refer to the variables in the external
	// function declared below; see walkclosure.
	n->list = ntype->list;
	n->rlist = ntype->rlist;
	ntype->list = nil;
	ntype->rlist = nil;
	for(l=n->list; l; l=l->next) {
		name = l->n->left;
		if(name)
			name = newname(name->sym);
		a = nod(ODCLFIELD, name, l->n->right);
		a->isddd = l->n->isddd;
		if(name)
			name->isddd = a->isddd;
		ntype->list = list(ntype->list, a);
	}
	for(l=n->rlist; l; l=l->next) {
		name = l->n->left;
		if(name) {
			orig = name->orig;  // preserve the meaning of orig == N (anonymous PPARAMOUT)
			name = newname(name->sym);
			name->orig = orig;
		}
		ntype->rlist = list(ntype->rlist, nod(ODCLFIELD, name, l->n->right));
	}
}

Node*
closurebody(NodeList *body)
{
	Node *func, *v;
	NodeList *l;

	if(body == nil)
		body = list1(nod(OEMPTY, N, N));

	func = curfn;
	func->nbody = body;
	funcbody(func);

	// closure-specific variables are hanging off the
	// ordinary ones in the symbol table; see oldname.
	// unhook them.
	// make the list of pointers for the closure call.
	for(l=func->cvars; l; l=l->next) {
		v = l->n;
		v->closure->closure = v->outer;
		v->heapaddr = nod(OADDR, oldname(v->sym), N);
	}

	return func;
}

static Node* makeclosure(Node *func, int nowrap);

void
typecheckclosure(Node *func, int top)
{
	Node *oldfn;
	NodeList *l;
	Node *v;

	oldfn = curfn;
	typecheck(&func->ntype, Etype);
	func->type = func->ntype->type;
	
	// Type check the body now, but only if we're inside a function.
	// At top level (in a variable initialization: curfn==nil) we're not
	// ready to type check code yet; we'll check it later, because the
	// underlying closure function we create is added to xtop.
	if(curfn && func->type != T) {
		curfn = func;
		typechecklist(func->nbody, Etop);
		curfn = oldfn;
	}

	// type check the & of closed variables outside the closure,
	// so that the outer frame also grabs them and knows they
	// escape.
	func->enter = nil;
	for(l=func->cvars; l; l=l->next) {
		v = l->n;
		if(v->type == T) {
			// if v->type is nil, it means v looked like it was
			// going to be used in the closure but wasn't.
			// this happens because when parsing a, b, c := f()
			// the a, b, c gets parsed as references to older
			// a, b, c before the parser figures out this is a
			// declaration.
			v->op = 0;
			continue;
		}
		// For a closure that is called in place, but not
		// inside a go statement, avoid moving variables to the heap.
		if ((top & (Ecall|Eproc)) == Ecall)
			v->heapaddr->etype = 1;
		typecheck(&v->heapaddr, Erv);
		func->enter = list(func->enter, v->heapaddr);
		v->heapaddr = N;
	}

	// Create top-level function 
	xtop = list(xtop, makeclosure(func, func->cvars==nil || (top&Ecall)));
}

static Node*
makeclosure(Node *func, int nowrap)
{
	Node *xtype, *v, *addr, *xfunc, *cv;
	NodeList *l, *body;
	static int closgen;
	char *p;
	int offset;

	/*
	 * wrap body in external function
	 * that begins by reading closure parameters.
	 */
	xtype = nod(OTFUNC, N, N);
	xtype->list = func->list;
	xtype->rlist = func->rlist;

	// create the function
	xfunc = nod(ODCLFUNC, N, N);
	snprint(namebuf, sizeof namebuf, "func·%.3d", ++closgen);
	xfunc->nname = newname(lookup(namebuf));
	xfunc->nname->sym->flags |= SymExported; // disable export
	xfunc->nname->ntype = xtype;
	xfunc->nname->defn = xfunc;
	declare(xfunc->nname, PFUNC);
	xfunc->nname->funcdepth = func->funcdepth;
	xfunc->funcdepth = func->funcdepth;
	
	// declare variables holding addresses taken from closure
	// and initialize in entry prologue.
	body = nil;
	offset = widthptr;
	for(l=func->cvars; l; l=l->next) {
		v = l->n;
		if(v->op == 0)
			continue;
		addr = nod(ONAME, N, N);
		p = smprint("&%s", v->sym->name);
		addr->sym = lookup(p);
		free(p);
		addr->ntype = nod(OIND, typenod(v->type), N);
		addr->class = PAUTO;
		addr->addable = 1;
		addr->ullman = 1;
		addr->used = 1;
		addr->curfn = xfunc;
		xfunc->dcl = list(xfunc->dcl, addr);
		v->heapaddr = addr;
		cv = nod(OCLOSUREVAR, N, N);
		cv->type = ptrto(v->type);
		cv->xoffset = offset;
		body = list(body, nod(OAS, addr, cv));
		offset += widthptr;
	}
	typechecklist(body, Etop);
	walkstmtlist(body);
	xfunc->enter = body;

	xfunc->nbody = func->nbody;
	xfunc->dcl = concat(func->dcl, xfunc->dcl);
	if(xfunc->nbody == nil)
		fatal("empty body - won't generate any code");
	typecheck(&xfunc, Etop);

	xfunc->closure = func;
	func->closure = xfunc;
	
	func->nbody = nil;
	func->list = nil;
	func->rlist = nil;

	return xfunc;
}

Node*
walkclosure(Node *func, NodeList **init)
{
	Node *clos, *typ;
	NodeList *l;
	char buf[20];
	int narg;

	// If no closure vars, don't bother wrapping.
	if(func->cvars == nil)
		return func->closure->nname;

	// Create closure in the form of a composite literal.
	// supposing the closure captures an int i and a string s
	// and has one float64 argument and no results,
	// the generated code looks like:
	//
	//	clos = &struct{F uintptr; A0 *int; A1 *string}{func·001, &i, &s}
	//
	// The use of the struct provides type information to the garbage
	// collector so that it can walk the closure. We could use (in this case)
	// [3]unsafe.Pointer instead, but that would leave the gc in the dark.
	// The information appears in the binary in the form of type descriptors;
	// the struct is unnamed so that closures in multiple packages with the
	// same struct type can share the descriptor.

	narg = 0;
	typ = nod(OTSTRUCT, N, N);
	typ->list = list1(nod(ODCLFIELD, newname(lookup("F")), typenod(types[TUINTPTR])));
	for(l=func->cvars; l; l=l->next) {
		if(l->n->op == 0)
			continue;
		snprint(buf, sizeof buf, "A%d", narg++);
		typ->list = list(typ->list, nod(ODCLFIELD, newname(lookup(buf)), l->n->heapaddr->ntype));
	}

	clos = nod(OCOMPLIT, N, nod(OIND, typ, N));
	clos->right->implicit = 1;
	clos->list = concat(list1(nod(OCFUNC, func->closure->nname, N)), func->enter);

	// Force type conversion from *struct to the func type.
	clos = nod(OCONVNOP, clos, N);
	clos->type = func->type;
	
	typecheck(&clos, Erv);
	walkexpr(&clos, init);

	return clos;
}

// Special case for closures that get called in place.
// Optimize runtime.closure(X, __func__xxxx_, .... ) away
// to __func__xxxx_(Y ....).
// On entry, expect n->op == OCALL, n->left->op == OCLOSURE.
void
walkcallclosure(Node *n, NodeList **init)
{
	USED(init);
	if (n->op != OCALLFUNC || n->left->op != OCLOSURE) {
		dump("walkcallclosure", n);
		fatal("abuse of walkcallclosure");
	}

	// New arg list for n. First the closure-args
	// and then the original parameter list.
	n->list = concat(n->left->enter, n->list);
	n->left = n->left->closure->nname;
	dowidth(n->left->type);
	n->type = getoutargx(n->left->type);
	// for a single valued function, pull the field type out of the struct
	if (n->type && n->type->type && !n->type->type->down)
		n->type = n->type->type->type;
}
