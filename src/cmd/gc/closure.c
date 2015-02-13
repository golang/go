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
		v->outerexpr = oldname(v->sym);
	}

	return func;
}

static Node* makeclosure(Node *func);

void
typecheckclosure(Node *func, int top)
{
	Node *oldfn, *n;
	NodeList *l;
	int olddd;

	for(l=func->cvars; l; l=l->next) {
		n = l->n->closure;
		if(!n->captured) {
			n->captured = 1;
			if(n->decldepth == 0)
				fatal("typecheckclosure: var %hN does not have decldepth assigned", n);
			// Ignore assignments to the variable in straightline code
			// preceding the first capturing by a closure.
			if(n->decldepth == decldepth)
				n->assigned = 0;
		}
	}

	for(l=func->dcl; l; l=l->next)
		if(l->n->op == ONAME && (l->n->class == PPARAM || l->n->class == PPARAMOUT))
			l->n->decldepth = 1;

	oldfn = curfn;
	typecheck(&func->ntype, Etype);
	func->type = func->ntype->type;
	func->top = top;

	// Type check the body now, but only if we're inside a function.
	// At top level (in a variable initialization: curfn==nil) we're not
	// ready to type check code yet; we'll check it later, because the
	// underlying closure function we create is added to xtop.
	if(curfn && func->type != T) {
		curfn = func;
		olddd = decldepth;
		decldepth = 1;
		typechecklist(func->nbody, Etop);
		decldepth = olddd;
		curfn = oldfn;
	}

	// Create top-level function 
	xtop = list(xtop, makeclosure(func));
}

static Node*
makeclosure(Node *func)
{
	Node *xtype, *xfunc;
	static int closgen;

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

// capturevars is called in a separate phase after all typechecking is done.
// It decides whether each variable captured by a closure should be captured
// by value or by reference.
// We use value capturing for values <= 128 bytes that are never reassigned
// after capturing (effectively constant).
void
capturevars(Node *xfunc)
{
	Node *func, *v, *outer;
	NodeList *l;
	int lno;

	lno = lineno;
	lineno = xfunc->lineno;

	func = xfunc->closure;
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
			v->op = OXXX;
			continue;
		}

		// type check the & of closed variables outside the closure,
		// so that the outer frame also grabs them and knows they escape.
		dowidth(v->type);
		outer = v->outerexpr;
		v->outerexpr = N;
		// out parameters will be assigned to implicitly upon return.
		if(outer->class != PPARAMOUT && !v->closure->addrtaken && !v->closure->assigned && v->type->width <= 128)
			v->byval = 1;
		else {
			v->closure->addrtaken = 1;
			outer = nod(OADDR, outer, N);
		}
		if(debug['m'] > 1) {
			Sym *name;
			char *how;
			name = nil;
			if(v->curfn && v->curfn->nname)
				name = v->curfn->nname->sym;
			how = "ref";
			if(v->byval)
				how = "value";
			warnl(v->lineno, "%S capturing by %s: %S (addr=%d assign=%d width=%d)",
				name, how,
				v->sym, v->closure->addrtaken, v->closure->assigned, (int32)v->type->width);
		}
		typecheck(&outer, Erv);
		func->enter = list(func->enter, outer);
	}

	lineno = lno;
}

// transformclosure is called in a separate phase after escape analysis.
// It transform closure bodies to properly reference captured variables.
void
transformclosure(Node *xfunc)
{
	Node *func, *cv, *addr, *v, *f;
	NodeList *l, *body;
	Type **param, *fld;
	vlong offset;
	int lno, nvar;

	lno = lineno;
	lineno = xfunc->lineno;
	func = xfunc->closure;

	if(func->top&Ecall) {
		// If the closure is directly called, we transform it to a plain function call
		// with variables passed as args. This avoids allocation of a closure object.
		// Here we do only a part of the transformation. Walk of OCALLFUNC(OCLOSURE)
		// will complete the transformation later.
		// For illustration, the following closure:
		//	func(a int) {
		//		println(byval)
		//		byref++
		//	}(42)
		// becomes:
		//	func(a int, byval int, &byref *int) {
		//		println(byval)
		//		(*&byref)++
		//	}(42, byval, &byref)

		// f is ONAME of the actual function.
		f = xfunc->nname;
		// Get pointer to input arguments and rewind to the end.
		// We are going to append captured variables to input args.
		param = &getinargx(f->type)->type;
		for(; *param; param = &(*param)->down) {
		}
		for(l=func->cvars; l; l=l->next) {
			v = l->n;
			if(v->op == OXXX)
				continue;
			fld = typ(TFIELD);
			fld->funarg = 1;
			if(v->byval) {
				// If v is captured by value, we merely downgrade it to PPARAM.
				v->class = PPARAM;
				v->ullman = 1;
				fld->nname = v;
			} else {
				// If v of type T is captured by reference,
				// we introduce function param &v *T
				// and v remains PPARAMREF with &v heapaddr
				// (accesses will implicitly deref &v).
				snprint(namebuf, sizeof namebuf, "&%s", v->sym->name);
				addr = newname(lookup(namebuf));
				addr->type = ptrto(v->type);
				addr->class = PPARAM;
				v->heapaddr = addr;
				fld->nname = addr;
			}
			fld->type = fld->nname->type;
			fld->sym = fld->nname->sym;
			// Declare the new param and append it to input arguments.
			xfunc->dcl = list(xfunc->dcl, fld->nname);
			*param = fld;
			param = &fld->down;
		}
		// Recalculate param offsets.
		if(f->type->width > 0)
			fatal("transformclosure: width is already calculated");
		dowidth(f->type);
		xfunc->type = f->type; // update type of ODCLFUNC
	} else {
		// The closure is not called, so it is going to stay as closure.
		nvar = 0;
		body = nil;
		offset = widthptr;
		for(l=func->cvars; l; l=l->next) {
			v = l->n;
			if(v->op == OXXX)
				continue;
			nvar++;
			// cv refers to the field inside of closure OSTRUCTLIT.
			cv = nod(OCLOSUREVAR, N, N);
			cv->type = v->type;
			if(!v->byval)
				cv->type = ptrto(v->type);
			offset = rnd(offset, cv->type->align);
			cv->xoffset = offset;
			offset += cv->type->width;

			if(v->byval && v->type->width <= 2*widthptr && thearch.thechar == '6') {
				//  If it is a small variable captured by value, downgrade it to PAUTO.
				// This optimization is currently enabled only for amd64, see:
				// https://github.com/golang/go/issues/9865
				v->class = PAUTO;
				v->ullman = 1;
				xfunc->dcl = list(xfunc->dcl, v);
				body = list(body, nod(OAS, v, cv));
			} else {
				// Declare variable holding addresses taken from closure
				// and initialize in entry prologue.
				snprint(namebuf, sizeof namebuf, "&%s", v->sym->name);
				addr = newname(lookup(namebuf));
				addr->ntype = nod(OIND, typenod(v->type), N);
				addr->class = PAUTO;
				addr->used = 1;
				addr->curfn = xfunc;
				xfunc->dcl = list(xfunc->dcl, addr);
				v->heapaddr = addr;
				if(v->byval)
					cv = nod(OADDR, cv, N);
				body = list(body, nod(OAS, addr, cv));
			}
		}
		typechecklist(body, Etop);
		walkstmtlist(body);
		xfunc->enter = body;
		xfunc->needctxt = nvar > 0;
	}

	lineno = lno;
}

Node*
walkclosure(Node *func, NodeList **init)
{
	Node *clos, *typ, *typ1, *v;
	NodeList *l;

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

	typ = nod(OTSTRUCT, N, N);
	typ->list = list1(nod(ODCLFIELD, newname(lookup("F")), typenod(types[TUINTPTR])));
	for(l=func->cvars; l; l=l->next) {
		v = l->n;
		if(v->op == OXXX)
			continue;
		typ1 = typenod(v->type);
		if(!v->byval)
			typ1 = nod(OIND, typ1, N);
		typ->list = list(typ->list, nod(ODCLFIELD, newname(v->sym), typ1));
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
