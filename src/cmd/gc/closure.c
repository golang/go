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
	Node *n, *name, *a;
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
		if(name)
			name = newname(name->sym);
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
	func->endlineno = lineno;
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

static Node* makeclosure(Node *func);

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
	xtop = list(xtop, makeclosure(func));
}

static Node*
makeclosure(Node *func)
{
	Node *xtype, *v, *addr, *xfunc, *cv;
	NodeList *l, *body;
	static int closgen;
	char *p;
	vlong offset;

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
	xfunc->endlineno = func->endlineno;
	
	// declare variables holding addresses taken from closure
	// and initialize in entry prologue.
	body = nil;
	offset = widthptr;
	xfunc->needctxt = func->cvars != nil;
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
	clos->esc = func->esc;
	clos->right->implicit = 1;
	clos->list = concat(list1(nod(OCFUNC, func->closure->nname, N)), func->enter);

	// Force type conversion from *struct to the func type.
	clos = nod(OCONVNOP, clos, N);
	clos->type = func->type;

	typecheck(&clos, Erv);
	// typecheck will insert a PTRLIT node under CONVNOP,
	// tag it with escape analysis result.
	clos->left->esc = func->esc;
	// non-escaping temp to use, if any.
	// orderexpr did not compute the type; fill it in now.
	if(func->alloc != N) {
		func->alloc->type = clos->left->left->type;
		func->alloc->orig->type = func->alloc->type;
		clos->left->right = func->alloc;
		func->alloc = N;
	}
	walkexpr(&clos, init);

	return clos;
}

static Node *makepartialcall(Node*, Type*, Node*);

void
typecheckpartialcall(Node *fn, Node *sym)
{
	switch(fn->op) {
	case ODOTINTER:
	case ODOTMETH:
		break;
	default:
		fatal("invalid typecheckpartialcall");
	}

	// Create top-level function.
	fn->nname = makepartialcall(fn, fn->type, sym);
	fn->right = sym;
	fn->op = OCALLPART;
	fn->type = fn->nname->type;
}

static Node*
makepartialcall(Node *fn, Type *t0, Node *meth)
{
	Node *ptr, *n, *fld, *call, *xtype, *xfunc, *cv, *savecurfn;
	Type *rcvrtype, *basetype, *t;
	NodeList *body, *l, *callargs, *retargs;
	char *p;
	Sym *sym;
	Pkg *spkg;
	static Pkg* gopkg;
	int i, ddd;

	// TODO: names are not right
	rcvrtype = fn->left->type;
	if(exportname(meth->sym->name))
		p = smprint("%-hT.%s·fm", rcvrtype, meth->sym->name);
	else
		p = smprint("%-hT.(%-S)·fm", rcvrtype, meth->sym);
	basetype = rcvrtype;
	if(isptr[rcvrtype->etype])
		basetype = basetype->type;
	if(basetype->etype != TINTER && basetype->sym == S)
		fatal("missing base type for %T", rcvrtype);

	spkg = nil;
	if(basetype->sym != S)
		spkg = basetype->sym->pkg;
	if(spkg == nil) {
		if(gopkg == nil)
			gopkg = mkpkg(newstrlit("go"));
		spkg = gopkg;
	}
	sym = pkglookup(p, spkg);
	free(p);
	if(sym->flags & SymUniq)
		return sym->def;
	sym->flags |= SymUniq;
	
	savecurfn = curfn;
	curfn = N;

	xtype = nod(OTFUNC, N, N);
	i = 0;
	l = nil;
	callargs = nil;
	ddd = 0;
	xfunc = nod(ODCLFUNC, N, N);
	curfn = xfunc;
	for(t = getinargx(t0)->type; t; t = t->down) {
		snprint(namebuf, sizeof namebuf, "a%d", i++);
		n = newname(lookup(namebuf));
		n->class = PPARAM;
		xfunc->dcl = list(xfunc->dcl, n);
		callargs = list(callargs, n);
		fld = nod(ODCLFIELD, n, typenod(t->type));
		if(t->isddd) {
			fld->isddd = 1;
			ddd = 1;
		}
		l = list(l, fld);
	}
	xtype->list = l;
	i = 0;
	l = nil;
	retargs = nil;
	for(t = getoutargx(t0)->type; t; t = t->down) {
		snprint(namebuf, sizeof namebuf, "r%d", i++);
		n = newname(lookup(namebuf));
		n->class = PPARAMOUT;
		xfunc->dcl = list(xfunc->dcl, n);
		retargs = list(retargs, n);
		l = list(l, nod(ODCLFIELD, n, typenod(t->type)));
	}
	xtype->rlist = l;

	xfunc->dupok = 1;
	xfunc->nname = newname(sym);
	xfunc->nname->sym->flags |= SymExported; // disable export
	xfunc->nname->ntype = xtype;
	xfunc->nname->defn = xfunc;
	declare(xfunc->nname, PFUNC);

	// Declare and initialize variable holding receiver.
	body = nil;
	xfunc->needctxt = 1;
	cv = nod(OCLOSUREVAR, N, N);
	cv->xoffset = widthptr;
	cv->type = rcvrtype;
	if(cv->type->align > widthptr)
		cv->xoffset = cv->type->align;
	ptr = nod(ONAME, N, N);
	ptr->sym = lookup("rcvr");
	ptr->class = PAUTO;
	ptr->addable = 1;
	ptr->ullman = 1;
	ptr->used = 1;
	ptr->curfn = xfunc;
	xfunc->dcl = list(xfunc->dcl, ptr);
	if(isptr[rcvrtype->etype] || isinter(rcvrtype)) {
		ptr->ntype = typenod(rcvrtype);
		body = list(body, nod(OAS, ptr, cv));
	} else {
		ptr->ntype = typenod(ptrto(rcvrtype));
		body = list(body, nod(OAS, ptr, nod(OADDR, cv, N)));
	}

	call = nod(OCALL, nod(OXDOT, ptr, meth), N);
	call->list = callargs;
	call->isddd = ddd;
	if(t0->outtuple == 0) {
		body = list(body, call);
	} else {
		n = nod(OAS2, N, N);
		n->list = retargs;
		n->rlist = list1(call);
		body = list(body, n);
		n = nod(ORETURN, N, N);
		body = list(body, n);
	}

	xfunc->nbody = body;

	typecheck(&xfunc, Etop);
	sym->def = xfunc;
	xtop = list(xtop, xfunc);
	curfn = savecurfn;

	return xfunc;
}

Node*
walkpartialcall(Node *n, NodeList **init)
{
	Node *clos, *typ;

	// Create closure in the form of a composite literal.
	// For x.M with receiver (x) type T, the generated code looks like:
	//
	//	clos = &struct{F uintptr; R T}{M.T·f, x}
	//
	// Like walkclosure above.

	if(isinter(n->left->type)) {
		// Trigger panic for method on nil interface now.
		// Otherwise it happens in the wrapper and is confusing.
		n->left = cheapexpr(n->left, init);
		checknil(n->left, init);
	}

	typ = nod(OTSTRUCT, N, N);
	typ->list = list1(nod(ODCLFIELD, newname(lookup("F")), typenod(types[TUINTPTR])));
	typ->list = list(typ->list, nod(ODCLFIELD, newname(lookup("R")), typenod(n->left->type)));

	clos = nod(OCOMPLIT, N, nod(OIND, typ, N));
	clos->esc = n->esc;
	clos->right->implicit = 1;
	clos->list = list1(nod(OCFUNC, n->nname->nname, N));
	clos->list = list(clos->list, n->left);

	// Force type conversion from *struct to the func type.
	clos = nod(OCONVNOP, clos, N);
	clos->type = n->type;

	typecheck(&clos, Erv);
	// typecheck will insert a PTRLIT node under CONVNOP,
	// tag it with escape analysis result.
	clos->left->esc = n->esc;
	// non-escaping temp to use, if any.
	// orderexpr did not compute the type; fill it in now.
	if(n->alloc != N) {
		n->alloc->type = clos->left->left->type;
		n->alloc->orig->type = n->alloc->type;
		clos->left->right = n->alloc;
		n->alloc = N;
	}
	walkexpr(&clos, init);

	return clos;
}
