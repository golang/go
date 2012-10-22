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
	Node *xtype, *v, *addr, *xfunc;
	NodeList *l;
	static int closgen;
	char *p;

	/*
	 * wrap body in external function
	 * with extra closure parameters.
	 */
	xtype = nod(OTFUNC, N, N);

	// each closure variable has a corresponding
	// address parameter.
	for(l=func->cvars; l; l=l->next) {
		v = l->n;
		if(v->op == 0)
			continue;
		addr = nod(ONAME, N, N);
		p = smprint("&%s", v->sym->name);
		addr->sym = lookup(p);
		free(p);
		addr->ntype = nod(OIND, typenod(v->type), N);
		addr->class = PPARAM;
		addr->addable = 1;
		addr->ullman = 1;

		v->heapaddr = addr;

		xtype->list = list(xtype->list, nod(ODCLFIELD, addr, addr->ntype));
	}

	// then a dummy arg where the closure's caller pc sits
	if (!nowrap)
		xtype->list = list(xtype->list, nod(ODCLFIELD, N, typenod(types[TUINTPTR])));

	// then the function arguments
	xtype->list = concat(xtype->list, func->list);
	xtype->rlist = concat(xtype->rlist, func->rlist);

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
	xfunc->nbody = func->nbody;
	xfunc->dcl = func->dcl;
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
	int narg;
	Node *xtype, *xfunc, *call, *clos;
	NodeList *l, *in;

	// no closure vars, don't bother wrapping
	if(func->cvars == nil)
		return func->closure->nname;

	/*
	 * wrap body in external function
	 * with extra closure parameters.
	 */

	// create the function
	xfunc = func->closure;
	xtype = xfunc->nname->ntype;

	// prepare call of sys.closure that turns external func into func literal value.
	clos = syslook("closure", 1);
	clos->type = T;
	clos->ntype = nod(OTFUNC, N, N);
	in = list1(nod(ODCLFIELD, N, typenod(types[TINT])));	// siz
	in = list(in, nod(ODCLFIELD, N, xtype));
	narg = 0;
	for(l=func->cvars; l; l=l->next) {
		if(l->n->op == 0)
			continue;
		narg++;
		in = list(in, nod(ODCLFIELD, N, l->n->heapaddr->ntype));
	}
	clos->ntype->list = in;
	clos->ntype->rlist = list1(nod(ODCLFIELD, N, typenod(func->type)));
	typecheck(&clos, Erv);

	call = nod(OCALL, clos, N);
	if(narg*widthptr > 100)
		yyerror("closure needs too many variables; runtime will reject it");
	in = list1(nodintconst(narg*widthptr));
	in = list(in, xfunc->nname);
	in = concat(in, func->enter);
	call->list = in;

	typecheck(&call, Erv);
	walkexpr(&call, init);
	return call;
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
