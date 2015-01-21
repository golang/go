// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// The inlining facility makes 2 passes: first caninl determines which
// functions are suitable for inlining, and for those that are it
// saves a copy of the body. Then inlcalls walks each function body to
// expand calls to inlinable functions.
//
// The debug['l'] flag controls the agressiveness. Note that main() swaps level 0 and 1,
// making 1 the default and -l disable.  -ll and more is useful to flush out bugs.
// These additional levels (beyond -l) may be buggy and are not supported.
//      0: disabled
//      1: 40-nodes leaf functions, oneliners, lazy typechecking (default)
//      2: early typechecking of all imported bodies 
//      3: allow variadic functions
//      4: allow non-leaf functions , (breaks runtime.Caller)
//      5: transitive inlining
//
//  At some point this may get another default and become switch-offable with -N.
//
//  The debug['m'] flag enables diagnostic output.  a single -m is useful for verifying
//  which calls get inlined or not, more is for debugging, and may go away at any point.
//
// TODO:
//   - inline functions with ... args
//   - handle T.meth(f()) with func f() (t T, arg, arg, )

#include <u.h>
#include <libc.h>
#include "go.h"

// Used by caninl.
static Node*	inlcopy(Node *n);
static NodeList* inlcopylist(NodeList *ll);
static int	ishairy(Node *n, int *budget);
static int	ishairylist(NodeList *ll, int *budget); 

// Used by inlcalls
static void	inlnodelist(NodeList *l);
static void	inlnode(Node **np);
static void	mkinlcall(Node **np, Node *fn, int isddd);
static Node*	inlvar(Node *n);
static Node*	retvar(Type *n, int i);
static Node*	argvar(Type *n, int i);
static Node*	newlabel(void);
static Node*	inlsubst(Node *n);
static NodeList* inlsubstlist(NodeList *l);

static void	setlno(Node*, int);

// Used during inlsubst[list]
static Node *inlfn;		// function currently being inlined
static Node *inlretlabel;	// target of the goto substituted in place of a return
static NodeList *inlretvars;	// temp out variables

// Get the function's package.  For ordinary functions it's on the ->sym, but for imported methods
// the ->sym can be re-used in the local package, so peel it off the receiver's type.
static Pkg*
fnpkg(Node *fn)
{
	Type *rcvr;
	
	if(fn->type->thistuple) {
		// method
		rcvr = getthisx(fn->type)->type->type;
		if(isptr[rcvr->etype])
			rcvr = rcvr->type;
		if(!rcvr->sym)
			fatal("receiver with no sym: [%S] %lN  (%T)", fn->sym, fn, rcvr);
		return rcvr->sym->pkg;
	}
	// non-method
	return fn->sym->pkg;
}

// Lazy typechecking of imported bodies.  For local functions, caninl will set ->typecheck
// because they're a copy of an already checked body. 
void
typecheckinl(Node *fn)
{
	Node *savefn;
	Pkg *pkg;
	int save_safemode, lno;

	lno = setlineno(fn);

	// typecheckinl is only for imported functions;
	// their bodies may refer to unsafe as long as the package
	// was marked safe during import (which was checked then).
	// the ->inl of a local function has been typechecked before caninl copied it.
	pkg = fnpkg(fn);
	if (pkg == localpkg || pkg == nil)
		return; // typecheckinl on local function

	if (debug['m']>2)
		print("typecheck import [%S] %lN { %#H }\n", fn->sym, fn, fn->inl);

	save_safemode = safemode;
	safemode = 0;

	savefn = curfn;
	curfn = fn;
	typechecklist(fn->inl, Etop);
	curfn = savefn;

	safemode = save_safemode;

	lineno = lno;
}

// Caninl determines whether fn is inlineable.
// If so, caninl saves fn->nbody in fn->inl and substitutes it with a copy.
// fn and ->nbody will already have been typechecked.
void
caninl(Node *fn)
{
	Node *savefn;
	Type *t;
	int budget;

	if(fn->op != ODCLFUNC)
		fatal("caninl %N", fn);
	if(!fn->nname)
		fatal("caninl no nname %+N", fn);

	// If fn has no body (is defined outside of Go), cannot inline it.
	if(fn->nbody == nil)
		return;

	if(fn->typecheck == 0)
		fatal("caninl on non-typechecked function %N", fn);

	// can't handle ... args yet
	if(debug['l'] < 3)
		for(t=fn->type->type->down->down->type; t; t=t->down)
			if(t->isddd)
				return;

	budget = 40;  // allowed hairyness
	if(ishairylist(fn->nbody, &budget))
		return;

	savefn = curfn;
	curfn = fn;

	fn->nname->inl = fn->nbody;
	fn->nbody = inlcopylist(fn->nname->inl);
	fn->nname->inldcl = inlcopylist(fn->nname->defn->dcl);

	// hack, TODO, check for better way to link method nodes back to the thing with the ->inl
	// this is so export can find the body of a method
	fn->type->nname = fn->nname;

	if(debug['m'] > 1)
		print("%L: can inline %#N as: %#T { %#H }\n", fn->lineno, fn->nname, fn->type, fn->nname->inl);
	else if(debug['m'])
		print("%L: can inline %N\n", fn->lineno, fn->nname);

	curfn = savefn;
}

// Look for anything we want to punt on.
static int
ishairylist(NodeList *ll, int* budget)
{
	for(;ll;ll=ll->next)
		if(ishairy(ll->n, budget))
			return 1;
	return 0;
}

static int
ishairy(Node *n, int *budget)
{
	if(!n)
		return 0;

	// Things that are too hairy, irrespective of the budget
	switch(n->op) {
	case OCALL:
	case OCALLFUNC:
	case OCALLINTER:
	case OCALLMETH:
	case OPANIC:
	case ORECOVER:
		if(debug['l'] < 4)
			return 1;
		break;

	case OCLOSURE:
	case OCALLPART:
	case ORANGE:
	case OFOR:
	case OSELECT:
	case OSWITCH:
	case OPROC:
	case ODEFER:
	case ODCLTYPE:  // can't print yet
	case ODCLCONST:  // can't print yet
	case ORETJMP:
		return 1;

		break;
	}

	(*budget)--;

	return  *budget < 0 ||
		ishairy(n->left, budget) ||
		ishairy(n->right, budget) ||
		ishairylist(n->list, budget) ||
		ishairylist(n->rlist, budget) ||
		ishairylist(n->ninit, budget) ||
		ishairy(n->ntest, budget) ||
		ishairy(n->nincr, budget) ||
		ishairylist(n->nbody, budget) ||
		ishairylist(n->nelse, budget);
}

// Inlcopy and inlcopylist recursively copy the body of a function.
// Any name-like node of non-local class is marked for re-export by adding it to
// the exportlist.
static NodeList*
inlcopylist(NodeList *ll)
{
	NodeList *l;

	l = nil;
	for(; ll; ll=ll->next)
		l = list(l, inlcopy(ll->n));
	return l;
}

static Node*
inlcopy(Node *n)
{
	Node *m;

	if(n == N)
		return N;

	switch(n->op) {
	case ONAME:
	case OTYPE:
	case OLITERAL:
		return n;
	}

	m = nod(OXXX, N, N);
	*m = *n;
	m->inl = nil;
	m->left	  = inlcopy(n->left);
	m->right  = inlcopy(n->right);
	m->list   = inlcopylist(n->list);
	m->rlist  = inlcopylist(n->rlist);
	m->ninit  = inlcopylist(n->ninit);
	m->ntest  = inlcopy(n->ntest);
	m->nincr  = inlcopy(n->nincr);
	m->nbody  = inlcopylist(n->nbody);
	m->nelse  = inlcopylist(n->nelse);

	return m;
}


// Inlcalls/nodelist/node walks fn's statements and expressions and substitutes any
// calls made to inlineable functions.  This is the external entry point.
void
inlcalls(Node *fn)
{
	Node *savefn;

	savefn = curfn;
	curfn = fn;
	inlnode(&fn);
	if(fn != curfn)
		fatal("inlnode replaced curfn");
	curfn = savefn;
}

// Turn an OINLCALL into a statement.
static void
inlconv2stmt(Node *n)
{
	n->op = OBLOCK;
	// n->ninit stays
	n->list = n->nbody;
	n->nbody = nil;
	n->rlist = nil;
}

// Turn an OINLCALL into a single valued expression.
static void
inlconv2expr(Node **np)
{
	Node *n, *r;
	n = *np;
	r = n->rlist->n;
	addinit(&r, concat(n->ninit, n->nbody));
	*np = r;
}

// Turn the rlist (with the return values) of the OINLCALL in
// n into an expression list lumping the ninit and body
// containing the inlined statements on the first list element so
// order will be preserved Used in return, oas2func and call
// statements.
static NodeList*
inlconv2list(Node *n)
{
	NodeList *l;

	if(n->op != OINLCALL || n->rlist == nil)
		fatal("inlconv2list %+N\n", n);
	
	l = n->rlist;
	addinit(&l->n, concat(n->ninit, n->nbody));
	return l;
} 
 
static void
inlnodelist(NodeList *l)
{
	for(; l; l=l->next)
		inlnode(&l->n);
}

// inlnode recurses over the tree to find inlineable calls, which will
// be turned into OINLCALLs by mkinlcall.  When the recursion comes
// back up will examine left, right, list, rlist, ninit, ntest, nincr,
// nbody and nelse and use one of the 4 inlconv/glue functions above
// to turn the OINLCALL into an expression, a statement, or patch it
// in to this nodes list or rlist as appropriate.
// NOTE it makes no sense to pass the glue functions down the
// recursion to the level where the OINLCALL gets created because they
// have to edit /this/ n, so you'd have to push that one down as well,
// but then you may as well do it here.  so this is cleaner and
// shorter and less complicated.
static void
inlnode(Node **np)
{
	Node *n;
	NodeList *l;
	int lno;

	if(*np == nil)
		return;

	n = *np;
	
	switch(n->op) {
	case ODEFER:
	case OPROC:
		// inhibit inlining of their argument
		switch(n->left->op) {
		case OCALLFUNC:
		case OCALLMETH:
			n->left->etype = n->op;
		}

	case OCLOSURE:
		// TODO do them here (or earlier),
		// so escape analysis can avoid more heapmoves.
		return;
	}

	lno = setlineno(n);

	inlnodelist(n->ninit);
	for(l=n->ninit; l; l=l->next)
		if(l->n->op == OINLCALL)
			inlconv2stmt(l->n);

	inlnode(&n->left);
	if(n->left && n->left->op == OINLCALL)
		inlconv2expr(&n->left);

	inlnode(&n->right);
	if(n->right && n->right->op == OINLCALL)
		inlconv2expr(&n->right);

	inlnodelist(n->list);
	switch(n->op) {
	case OBLOCK:
		for(l=n->list; l; l=l->next)
			if(l->n->op == OINLCALL)
				inlconv2stmt(l->n);
		break;

	case ORETURN:
	case OCALLFUNC:
	case OCALLMETH:
	case OCALLINTER:
	case OAPPEND:
	case OCOMPLEX:
		// if we just replaced arg in f(arg()) or return arg with an inlined call
		// and arg returns multiple values, glue as list
		if(count(n->list) == 1 && n->list->n->op == OINLCALL && count(n->list->n->rlist) > 1) {
			n->list = inlconv2list(n->list->n);
			break;
		}

		// fallthrough
	default:
		for(l=n->list; l; l=l->next)
			if(l->n->op == OINLCALL)
				inlconv2expr(&l->n);
	}

	inlnodelist(n->rlist);
	switch(n->op) {
	case OAS2FUNC:
		if(n->rlist->n->op == OINLCALL) {
			n->rlist = inlconv2list(n->rlist->n);
			n->op = OAS2;
			n->typecheck = 0;
			typecheck(np, Etop);
			break;
		}

		// fallthrough
	default:
		for(l=n->rlist; l; l=l->next)
			if(l->n->op == OINLCALL)
				inlconv2expr(&l->n);

	}

	inlnode(&n->ntest);
	if(n->ntest && n->ntest->op == OINLCALL)
		inlconv2expr(&n->ntest);

	inlnode(&n->nincr);
	if(n->nincr && n->nincr->op == OINLCALL)
		inlconv2stmt(n->nincr);

	inlnodelist(n->nbody);
	for(l=n->nbody; l; l=l->next)
		if(l->n->op == OINLCALL)
			inlconv2stmt(l->n);

	inlnodelist(n->nelse);
	for(l=n->nelse; l; l=l->next)
		if(l->n->op == OINLCALL)
			inlconv2stmt(l->n);

	// with all the branches out of the way, it is now time to
	// transmogrify this node itself unless inhibited by the
	// switch at the top of this function.
	switch(n->op) {
	case OCALLFUNC:
	case OCALLMETH:
		if (n->etype == OPROC || n->etype == ODEFER)
			return;
	}

	switch(n->op) {
	case OCALLFUNC:
		if(debug['m']>3)
			print("%L:call to func %+N\n", n->lineno, n->left);
		if(n->left->inl)	// normal case
			mkinlcall(np, n->left, n->isddd);
		else if(n->left->op == ONAME && n->left->left && n->left->left->op == OTYPE && n->left->right &&  n->left->right->op == ONAME)  // methods called as functions
			if(n->left->sym->def)
				mkinlcall(np, n->left->sym->def, n->isddd);
		break;

	case OCALLMETH:
		if(debug['m']>3)
			print("%L:call to meth %lN\n", n->lineno, n->left->right);
		// typecheck should have resolved ODOTMETH->type, whose nname points to the actual function.
		if(n->left->type == T) 
			fatal("no function type for [%p] %+N\n", n->left, n->left);

		if(n->left->type->nname == N) 
			fatal("no function definition for [%p] %+T\n", n->left->type, n->left->type);

		mkinlcall(np, n->left->type->nname, n->isddd);

		break;
	}
	
	lineno = lno;
}

static void	mkinlcall1(Node **np, Node *fn, int isddd);

static void
mkinlcall(Node **np, Node *fn, int isddd)
{
	int save_safemode;
	Pkg *pkg;

	save_safemode = safemode;

	// imported functions may refer to unsafe as long as the
	// package was marked safe during import (already checked).
	pkg = fnpkg(fn);
	if(pkg != localpkg && pkg != nil)
		safemode = 0;
	mkinlcall1(np, fn, isddd);
	safemode = save_safemode;
}

static Node*
tinlvar(Type *t)
{
	if(t->nname && !isblank(t->nname)) {
		if(!t->nname->inlvar)
			fatal("missing inlvar for %N\n", t->nname);
		return t->nname->inlvar;
	}
	typecheck(&nblank, Erv | Easgn);
	return nblank;
}

static int inlgen;

// if *np is a call, and fn is a function with an inlinable body, substitute *np with an OINLCALL.
// On return ninit has the parameter assignments, the nbody is the
// inlined function body and list, rlist contain the input, output
// parameters.
static void
mkinlcall1(Node **np, Node *fn, int isddd)
{
	int i;
	int chkargcount;
	Node *n, *call, *saveinlfn, *as, *m;
	NodeList *dcl, *ll, *ninit, *body;
	Type *t;
	// For variadic fn.
	int variadic, varargcount, multiret;
	Node *vararg;
	NodeList *varargs;
	Type *varargtype, *vararrtype;

	if (fn->inl == nil)
		return;

	if (fn == curfn || fn->defn == curfn)
		return;

	if(debug['l']<2)
		typecheckinl(fn);

	n = *np;

	// Bingo, we have a function node, and it has an inlineable body
	if(debug['m']>1)
		print("%L: inlining call to %S %#T { %#H }\n", n->lineno, fn->sym, fn->type, fn->inl);
	else if(debug['m'])
		print("%L: inlining call to %N\n", n->lineno, fn);

	if(debug['m']>2)
		print("%L: Before inlining: %+N\n", n->lineno, n);

	saveinlfn = inlfn;
	inlfn = fn;

	ninit = n->ninit;

//dumplist("ninit pre", ninit);

	if(fn->defn) // local function
		dcl = fn->inldcl;
	else // imported function
		dcl = fn->dcl;

	inlretvars = nil;
	i = 0;
	// Make temp names to use instead of the originals
	for(ll = dcl; ll; ll=ll->next) {
		if(ll->n->class == PPARAMOUT)  // return values handled below.
			continue;
		if(ll->n->op == ONAME) {
			ll->n->inlvar = inlvar(ll->n);
			// Typecheck because inlvar is not necessarily a function parameter.
			typecheck(&ll->n->inlvar, Erv);
			if ((ll->n->class&~PHEAP) != PAUTO)
				ninit = list(ninit, nod(ODCL, ll->n->inlvar, N));  // otherwise gen won't emit the allocations for heapallocs
		}
	}

	// temporaries for return values.
	for(t = getoutargx(fn->type)->type; t; t = t->down) {
		if(t != T && t->nname != N && !isblank(t->nname)) {
			m = inlvar(t->nname);
			typecheck(&m, Erv);
			t->nname->inlvar = m;
		} else {
			// anonymous return values, synthesize names for use in assignment that replaces return
			m = retvar(t, i++);
		}
		ninit = list(ninit, nod(ODCL, m, N));
		inlretvars = list(inlretvars, m);
	}

	// assign receiver.
	if(fn->type->thistuple && n->left->op == ODOTMETH) {
		// method call with a receiver.
		t = getthisx(fn->type)->type;
		if(t != T && t->nname != N && !isblank(t->nname) && !t->nname->inlvar)
			fatal("missing inlvar for %N\n", t->nname);
		if(!n->left->left)
			fatal("method call without receiver: %+N", n);
		if(t == T)
			fatal("method call unknown receiver type: %+N", n);
		as = nod(OAS, tinlvar(t), n->left->left);
		if(as != N) {
			typecheck(&as, Etop);
			ninit = list(ninit, as);
		}
	}

	// check if inlined function is variadic.
	variadic = 0;
	varargtype = T;
	varargcount = 0;
	for(t=fn->type->type->down->down->type; t; t=t->down) {
		if(t->isddd) {
			variadic = 1;
			varargtype = t->type;
		}
	}
	// but if argument is dotted too forget about variadicity.
	if(variadic && isddd)
		variadic = 0;

	// check if argument is actually a returned tuple from call.
	multiret = 0;
	if(n->list && !n->list->next) {
		switch(n->list->n->op) {
		case OCALL:
		case OCALLFUNC:
		case OCALLINTER:
		case OCALLMETH:
			if(n->list->n->left->type->outtuple > 1)
				multiret = n->list->n->left->type->outtuple-1;
		}
	}

	if(variadic) {
		varargcount = count(n->list) + multiret;
		if(n->left->op != ODOTMETH)
			varargcount -= fn->type->thistuple;
		varargcount -= fn->type->intuple - 1;
	}

	// assign arguments to the parameters' temp names
	as = nod(OAS2, N, N);
	as->rlist = n->list;
	ll = n->list;

	// TODO: if len(nlist) == 1 but multiple args, check that n->list->n is a call?
	if(fn->type->thistuple && n->left->op != ODOTMETH) {
		// non-method call to method
		if(!n->list)
			fatal("non-method call to method without first arg: %+N", n);
		// append receiver inlvar to LHS.
		t = getthisx(fn->type)->type;
		if(t != T && t->nname != N && !isblank(t->nname) && !t->nname->inlvar)
			fatal("missing inlvar for %N\n", t->nname);
		if(t == T)
			fatal("method call unknown receiver type: %+N", n);
		as->list = list(as->list, tinlvar(t));
		ll = ll->next; // track argument count.
	}

	// append ordinary arguments to LHS.
	chkargcount = n->list && n->list->next;
	vararg = N;    // the slice argument to a variadic call
	varargs = nil; // the list of LHS names to put in vararg.
	if(!chkargcount) {
		// 0 or 1 expression on RHS.
		for(t = getinargx(fn->type)->type; t; t=t->down) {
			if(variadic && t->isddd) {
				vararg = tinlvar(t);
				for(i=0; i<varargcount && ll; i++) {
					m = argvar(varargtype, i);
					varargs = list(varargs, m);
					as->list = list(as->list, m);
				}
				break;
			}
			as->list = list(as->list, tinlvar(t));
		}
	} else {
		// match arguments except final variadic (unless the call is dotted itself)
		for(t = getinargx(fn->type)->type; t;) {
			if(!ll)
				break;
			if(variadic && t->isddd)
				break;
			as->list = list(as->list, tinlvar(t));
			t=t->down;
			ll=ll->next;
		}
		// match varargcount arguments with variadic parameters.
		if(variadic && t && t->isddd) {
			vararg = tinlvar(t);
			for(i=0; i<varargcount && ll; i++) {
				m = argvar(varargtype, i);
				varargs = list(varargs, m);
				as->list = list(as->list, m);
				ll=ll->next;
			}
			if(i==varargcount)
				t=t->down;
		}
		if(ll || t)
			fatal("arg count mismatch: %#T  vs %,H\n",  getinargx(fn->type), n->list);
	}

	if (as->rlist) {
		typecheck(&as, Etop);
		ninit = list(ninit, as);
	}

	// turn the variadic args into a slice.
	if(variadic) {
		as = nod(OAS, vararg, N);
		if(!varargcount) {
			as->right = nodnil();
			as->right->type = varargtype;
		} else {
			vararrtype = typ(TARRAY);
			vararrtype->type = varargtype->type;
			vararrtype->bound = varargcount;

			as->right = nod(OCOMPLIT, N, typenod(varargtype));
			as->right->list = varargs;
			as->right = nod(OSLICE, as->right, nod(OKEY, N, N));
		}
		typecheck(&as, Etop);
		ninit = list(ninit, as);
	}

	// zero the outparams
	for(ll = inlretvars; ll; ll=ll->next) {
		as = nod(OAS, ll->n, N);
		typecheck(&as, Etop);
		ninit = list(ninit, as);
	}

	inlretlabel = newlabel();
	inlgen++;
	body = inlsubstlist(fn->inl);

	body = list(body, nod(OGOTO, inlretlabel, N));	// avoid 'not used' when function doesnt have return
	body = list(body, nod(OLABEL, inlretlabel, N));

	typechecklist(body, Etop);
//dumplist("ninit post", ninit);

	call = nod(OINLCALL, N, N);
	call->ninit = ninit;
	call->nbody = body;
	call->rlist = inlretvars;
	call->type = n->type;
	call->typecheck = 1;

	setlno(call, n->lineno);
//dumplist("call body", body);

	*np = call;

	inlfn =	saveinlfn;

	// transitive inlining
	// TODO do this pre-expansion on fn->inl directly.  requires
	// either supporting exporting statemetns with complex ninits
	// or saving inl and making inlinl
	if(debug['l'] >= 5) {
		body = fn->inl;
		fn->inl = nil;	// prevent infinite recursion
		inlnodelist(call->nbody);
		for(ll=call->nbody; ll; ll=ll->next)
			if(ll->n->op == OINLCALL)
				inlconv2stmt(ll->n);
		fn->inl = body;
	}

	if(debug['m']>2)
		print("%L: After inlining %+N\n\n", n->lineno, *np);

}

// Every time we expand a function we generate a new set of tmpnames,
// PAUTO's in the calling functions, and link them off of the
// PPARAM's, PAUTOS and PPARAMOUTs of the called function. 
static Node*
inlvar(Node *var)
{
	Node *n;

	if(debug['m']>3)
		print("inlvar %+N\n", var);

	n = newname(var->sym);
	n->type = var->type;
	n->class = PAUTO;
	n->used = 1;
	n->curfn = curfn;   // the calling function, not the called one
	n->addrtaken = var->addrtaken;

	// Esc pass wont run if we're inlining into a iface wrapper.
	// Luckily, we can steal the results from the target func.
	// If inlining a function defined in another package after
	// escape analysis is done, treat all local vars as escaping.
	// See issue 9537.
	if(var->esc == EscHeap || (inl_nonlocal && var->op == ONAME))
		addrescapes(n);

	curfn->dcl = list(curfn->dcl, n);
	return n;
}

// Synthesize a variable to store the inlined function's results in.
static Node*
retvar(Type *t, int i)
{
	Node *n;

	snprint(namebuf, sizeof(namebuf), "~r%d", i);
	n = newname(lookup(namebuf));
	n->type = t->type;
	n->class = PAUTO;
	n->used = 1;
	n->curfn = curfn;   // the calling function, not the called one
	curfn->dcl = list(curfn->dcl, n);
	return n;
}

// Synthesize a variable to store the inlined function's arguments
// when they come from a multiple return call.
static Node*
argvar(Type *t, int i)
{
	Node *n;

	snprint(namebuf, sizeof(namebuf), "~arg%d", i);
	n = newname(lookup(namebuf));
	n->type = t->type;
	n->class = PAUTO;
	n->used = 1;
	n->curfn = curfn;   // the calling function, not the called one
	curfn->dcl = list(curfn->dcl, n);
	return n;
}

static Node*
newlabel(void)
{
	Node *n;
	static int label;
	
	label++;
	snprint(namebuf, sizeof(namebuf), ".inlret%.6d", label);
	n = newname(lookup(namebuf));
	n->etype = 1;  // flag 'safe' for escape analysis (no backjumps)
	return n;
}

// inlsubst and inlsubstlist recursively copy the body of the saved
// pristine ->inl body of the function while substituting references
// to input/output parameters with ones to the tmpnames, and
// substituting returns with assignments to the output.
static NodeList*
inlsubstlist(NodeList *ll)
{
	NodeList *l;

	l = nil;
	for(; ll; ll=ll->next)
		l = list(l, inlsubst(ll->n));
	return l;
}

static Node*
inlsubst(Node *n)
{
	char *p;
	Node *m, *as;
	NodeList *ll;

	if(n == N)
		return N;

	switch(n->op) {
	case ONAME:
		if(n->inlvar) { // These will be set during inlnode
			if (debug['m']>2)
				print ("substituting name %+N  ->  %+N\n", n, n->inlvar);
			return n->inlvar;
		}
		if (debug['m']>2)
			print ("not substituting name %+N\n", n);
		return n;

	case OLITERAL:
	case OTYPE:
		return n;

	case ORETURN:
		// Since we don't handle bodies with closures, this return is guaranteed to belong to the current inlined function.

//		dump("Return before substitution", n);
		m = nod(OGOTO, inlretlabel, N);
		m->ninit  = inlsubstlist(n->ninit);

		if(inlretvars && n->list) {
			as = nod(OAS2, N, N);
			// shallow copy or OINLCALL->rlist will be the same list, and later walk and typecheck may clobber that.
			for(ll=inlretvars; ll; ll=ll->next)
				as->list = list(as->list, ll->n);
			as->rlist = inlsubstlist(n->list);
			typecheck(&as, Etop);
			m->ninit = list(m->ninit, as);
		}

		typechecklist(m->ninit, Etop);
		typecheck(&m, Etop);
//		dump("Return after substitution", m);
		return m;
	
	case OGOTO:
	case OLABEL:
		m = nod(OXXX, N, N);
		*m = *n;
		m->ninit = nil;
		p = smprint("%s·%d", n->left->sym->name, inlgen);	
		m->left = newname(lookup(p));
		free(p);
		return m;	
	}


	m = nod(OXXX, N, N);
	*m = *n;
	m->ninit = nil;
	
	if(n->op == OCLOSURE)
		fatal("cannot inline function containing closure: %+N", n);

	m->left	  = inlsubst(n->left);
	m->right  = inlsubst(n->right);
	m->list	  = inlsubstlist(n->list);
	m->rlist  = inlsubstlist(n->rlist);
	m->ninit  = concat(m->ninit, inlsubstlist(n->ninit));
	m->ntest  = inlsubst(n->ntest);
	m->nincr  = inlsubst(n->nincr);
	m->nbody  = inlsubstlist(n->nbody);
	m->nelse  = inlsubstlist(n->nelse);

	return m;
}

// Plaster over linenumbers
static void
setlnolist(NodeList *ll, int lno)
{
	for(;ll;ll=ll->next)
		setlno(ll->n, lno);
}

static void
setlno(Node *n, int lno)
{
	if(!n)
		return;

	// don't clobber names, unless they're freshly synthesized
	if(n->op != ONAME || n->lineno == 0)
		n->lineno = lno;
	
	setlno(n->left, lno);
	setlno(n->right, lno);
	setlnolist(n->list, lno);
	setlnolist(n->rlist, lno);
	setlnolist(n->ninit, lno);
	setlno(n->ntest, lno);
	setlno(n->nincr, lno);
	setlnolist(n->nbody, lno);
	setlnolist(n->nelse, lno);
}
