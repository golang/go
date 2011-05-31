// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * function literals aka closures
 */

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
	l = func->dcl;
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

void
typecheckclosure(Node *func)
{
	Node *oldfn;
	NodeList *l;
	Node *v;

	oldfn = curfn;
	typecheck(&func->ntype, Etype);
	func->type = func->ntype->type;
	if(func->type != T) {
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
		typecheck(&v->heapaddr, Erv);
		func->enter = list(func->enter, v->heapaddr);
		v->heapaddr = N;
	}
}

Node*
walkclosure(Node *func, NodeList **init)
{
	int narg;
	Node *xtype, *v, *addr, *xfunc, *call, *clos;
	NodeList *l, *in;
	static int closgen;
	char *p;

	/*
	 * wrap body in external function
	 * with extra closure parameters.
	 */
	xtype = nod(OTFUNC, N, N);

	// each closure variable has a corresponding
	// address parameter.
	narg = 0;
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
		narg++;

		v->heapaddr = addr;

		xtype->list = list(xtype->list, nod(ODCLFIELD, addr, addr->ntype));
	}

	// then a dummy arg where the closure's caller pc sits
	xtype->list = list(xtype->list, nod(ODCLFIELD, N, typenod(types[TUINTPTR])));

	// then the function arguments
	xtype->list = concat(xtype->list, func->list);
	xtype->rlist = concat(xtype->rlist, func->rlist);

	// create the function
	xfunc = nod(ODCLFUNC, N, N);
	snprint(namebuf, sizeof namebuf, "_func_%.3d", ++closgen);
	xfunc->nname = newname(lookup(namebuf));
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
	closures = list(closures, xfunc);

	// prepare call of sys.closure that turns external func into func literal value.
	clos = syslook("closure", 1);
	clos->type = T;
	clos->ntype = nod(OTFUNC, N, N);
	in = list1(nod(ODCLFIELD, N, typenod(types[TINT])));	// siz
	in = list(in, nod(ODCLFIELD, N, xtype));
	for(l=func->cvars; l; l=l->next) {
		if(l->n->op == 0)
			continue;
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
	Node *z;
	NodeList *ll, *cargs;

	walkexpr(&n->left, init);
	cargs =	n->left    // FUNC runtime.closure
		->list     // arguments
		->next     // skip first
		->next;    // skip second

	n->left = n->left  // FUNC runtime.closure
		->list     // arguments
		->next     // skip first
		->n        // AS (to indreg) 
		->right;   // argument  == the generated function 

	// New arg list for n. First the closure-args, stolen from
	// runtime.closure's 3rd and following,
	ll = nil;
	for (; cargs; cargs = cargs->next)
		ll = list(ll, cargs->n->right);  // cargs->n is the OAS(INDREG, arg)

	// then an extra zero, to fill the dummy return pointer slot,
	z = nod(OXXX, N, N);
	nodconst(z, types[TUINTPTR], 0);
	z->typecheck = 1;
	ll = list(ll, z);

	// and finally the original parameter list.
	n->list = concat(ll, n->list);
}
