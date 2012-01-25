// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// The inlining facility makes 2 passes: first caninl determines which
// functions are suitable for inlining, and for those that are it
// saves a copy of the body. Then inlcalls walks each function body to
// expand calls to inlinable functions.
//
// TODO:
//   - inline functions with ... args
//   - handle T.meth(f()) with func f() (t T, arg, arg, )
//   - (limited) recursive inlining
//   - it would be nice if func max(x, y int) { if x > y { return x }; return y } would be inlineable

#include <u.h>
#include <libc.h>
#include "go.h"

// Used by caninl.
static Node*	inlcopy(Node *n);
static NodeList* inlcopylist(NodeList *ll);
static int	ishairy(Node *n);
static int	ishairylist(NodeList *ll); 

// Used by inlcalls
static void	inlnodelist(NodeList *l);
static void	inlnode(Node **np);
static void	mkinlcall(Node **np, Node *fn);
static Node*	inlvar(Node *n);
static Node*	retvar(Type *n, int i);
static Node*	newlabel(void);
static Node*	inlsubst(Node *n);
static NodeList* inlsubstlist(NodeList *ll);

static void	setlno(Node*, int);

// Used during inlsubst[list]
static Node *inlfn;		// function currently being inlined
static Node *inlretlabel;	// target of the goto substituted in place of a return
static NodeList *inlretvars;	// temp out variables


void
typecheckinl(Node *fn)
{
	Node *savefn;

	if (debug['m']>2)
		print("typecheck import [%S] %lN { %#H }\n", fn->sym, fn, fn->inl);

	savefn = curfn;
	curfn = fn;
	importpkg = fn->sym->pkg;
	typechecklist(fn->inl, Etop);
	importpkg = nil;
	curfn = savefn;
}

// Caninl determines whether fn is inlineable. Currently that means:
// fn is exactly 1 statement, either a return or an assignment, and
// some temporary constraints marked TODO.  If fn is inlineable, saves
// fn->nbody in fn->inl and substitutes it with a copy.
void
caninl(Node *fn)
{
	Node *savefn;
	Type *t;

	if(fn->op != ODCLFUNC)
		fatal("caninl %N", fn);
	if(!fn->nname)
		fatal("caninl no nname %+N", fn);

	// exactly 1 statement
	if(fn->nbody == nil || fn->nbody->next != nil)
		return;

	// the single statement should be a return, an assignment or empty.
	switch(fn->nbody->n->op) {
	default:
		return;
	case ORETURN:
	case OAS:
	case OAS2:
	case OEMPTY:
		break;
	}

	// can't handle ... args yet
	for(t=fn->type->type->down->down->type; t; t=t->down)
		if(t->isddd)
			return;

	// TODO Anything non-trivial
	if(ishairy(fn))
		return;

	savefn = curfn;
	curfn = fn;

	fn->nname->inl = fn->nbody;
	fn->nbody = inlcopylist(fn->nname->inl);

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
ishairylist(NodeList *ll)
{
	for(;ll;ll=ll->next)
		if(ishairy(ll->n))
			return 1;
	return 0;
}

static int
ishairy(Node *n)
{
	if(!n)
		return 0;

	// Some of these are implied by the single-assign-or-return condition in caninl,
	// but they may stay even if that one is relaxed.
	switch(n->op) {
	case OCALL:
	case OCALLFUNC:
	case OCALLINTER:
	case OCALLMETH:
	case OCLOSURE:	// TODO too hard to inlvar the PARAMREFs
	case OIF:
	case ORANGE:
	case OFOR:
	case OSELECT:
	case OSWITCH:
	case OPROC:
	case ODEFER:
		return 1;
	}

	return  ishairy(n->left) ||
		ishairy(n->right) ||
		ishairylist(n->list) ||
		ishairylist(n->rlist) ||
		ishairylist(n->ninit) ||
		ishairy(n->ntest) ||
		ishairy(n->nincr) ||
		ishairylist(n->nbody) ||
		ishairylist(n->nelse);
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

// Turn the OINLCALL in n->list into an expression list on n.
// Used in return and call statements.
static void
inlgluelist(Node *n)
{
	Node *c;

	c = n->list->n;  // this is the OINLCALL
	n->ninit = concat(n->ninit, c->ninit);
	n->ninit = concat(n->ninit, c->nbody);
	n->list  = c->rlist;
} 

// Turn the OINLCALL in n->rlist->n into an expression list on n.
// Used in OAS2FUNC.
static void
inlgluerlist(Node *n)
{
	Node *c;

	c = n->rlist->n;  // this is the OINLCALL
	n->ninit = concat(n->ninit, c->ninit);
	n->ninit = concat(n->ninit, c->nbody);
	n->rlist = c->rlist;
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
		// TODO do them here instead of in lex.c phase 6b, so escape analysis
		// can avoid more heapmoves.
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
		if(count(n->list) == 1 && curfn->type->outtuple > 1 && n->list->n->op == OINLCALL) {
			inlgluelist(n);
			break;
		}
		
		goto list_dflt;

	case OCALLMETH:
	case OCALLINTER:
	case OCALLFUNC:
		// if we just replaced arg in f(arg()) with an inlined call
		// and arg returns multiple values, glue as list
		if(count(n->list) == 1 && n->list->n->op == OINLCALL && count(n->list->n->rlist) > 1) {
			inlgluelist(n);
			break;
		}

		// fallthrough
	default:
	list_dflt:
		for(l=n->list; l; l=l->next)
			if(l->n->op == OINLCALL)
				inlconv2expr(&l->n);
	}

	inlnodelist(n->rlist);
	switch(n->op) {
	case OAS2FUNC:
		if(n->rlist->n->op == OINLCALL) {
			inlgluerlist(n);
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
			mkinlcall(np, n->left);
		else if(n->left->op == ONAME && n->left->left && n->left->left->op == OTYPE && n->left->right &&  n->left->right->op == ONAME)  // methods called as functions
			if(n->left->sym->def)
				mkinlcall(np, n->left->sym->def);
		break;

	case OCALLMETH:
		if(debug['m']>3)
			print("%L:call to meth %lN\n", n->lineno, n->left->right);
		// typecheck should have resolved ODOTMETH->type, whose nname points to the actual function.
		if(n->left->type == T) 
			fatal("no function type for [%p] %+N\n", n->left, n->left);

		if(n->left->type->nname == N) 
			fatal("no function definition for [%p] %+T\n", n->left->type, n->left->type);

		mkinlcall(np, n->left->type->nname);

		break;
	}
	
	lineno = lno;
}

// if *np is a call, and fn is a function with an inlinable body, substitute *np with an OINLCALL.
// On return ninit has the parameter assignments, the nbody is the
// inlined function body and list, rlist contain the input, output
// parameters.
static void
mkinlcall(Node **np, Node *fn)
{
	int i;
	Node *n, *call, *saveinlfn, *as, *m;
	NodeList *dcl, *ll, *ninit, *body;
	Type *t;

	if (fn->inl == nil)
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

	if (fn->defn) // local function
		dcl = fn->defn->dcl;
	else // imported function
		dcl = fn->dcl;

	inlretvars = nil;
	i = 0;
	// Make temp names to use instead of the originals
	for(ll = dcl; ll; ll=ll->next)
		if(ll->n->op == ONAME) {
			ll->n->inlvar = inlvar(ll->n);
			ninit = list(ninit, nod(ODCL, ll->n->inlvar, N));  // otherwise gen won't emit the allocations for heapallocs
			if (ll->n->class == PPARAMOUT)  // we rely on the order being correct here
				inlretvars = list(inlretvars, ll->n->inlvar);
		}

	// anonymous return values, synthesize names for use in assignment that replaces return
	if(inlretvars == nil && fn->type->outtuple > 0)
		for(t = getoutargx(fn->type)->type; t; t = t->down) {
			m = retvar(t, i++);
			ninit = list(ninit, nod(ODCL, m, N));
			inlretvars = list(inlretvars, m);
		}

	// assign arguments to the parameters' temp names
	as = N;
	if(fn->type->thistuple) {
		t = getthisx(fn->type)->type;
		if(t != T && t->nname != N && !isblank(t->nname) && !t->nname->inlvar)
			fatal("missing inlvar for %N\n", t->nname);

		if(n->left->op == ODOTMETH) {
			if(!n->left->left)
				fatal("method call without receiver: %+N", n);
			if(t == T)
				fatal("method call unknown receiver type: %+N", n);
			if(t->nname != N && !isblank(t->nname))
				as = nod(OAS, t->nname->inlvar, n->left->left);
			else
				as = nod(OAS, temp(t->type), n->left->left);
		} else {  // non-method call to method
			if (!n->list)
				fatal("non-method call to method without first arg: %+N", n);
			if(t != T && t->nname != N && !isblank(t->nname))
				as = nod(OAS, t->nname->inlvar, n->list->n);
		}

		if(as != N) {
			typecheck(&as, Etop);
			ninit = list(ninit, as);
		}
	}

	as = nod(OAS2, N, N);
	if(fn->type->intuple > 1 && n->list && !n->list->next) {
		// TODO check that n->list->n is a call?
		// TODO: non-method call to T.meth(f()) where f returns t, args...
		as->rlist = n->list;
		for(t = getinargx(fn->type)->type; t; t=t->down) {
			if(t->nname && !isblank(t->nname)) {
				if(!t->nname->inlvar)
					fatal("missing inlvar for %N\n", t->nname);
				as->list = list(as->list, t->nname->inlvar);
			} else {
				as->list = list(as->list, temp(t->type));
			}
		}		
	} else {
		ll = n->list;
		if(fn->type->thistuple && n->left->op != ODOTMETH) // non method call to method
			ll=ll->next;  // was handled above in if(thistuple)

		for(t = getinargx(fn->type)->type; t && ll; t=t->down) {
			if(t->nname && !isblank(t->nname)) {
				if(!t->nname->inlvar)
					fatal("missing inlvar for %N\n", t->nname);
				as->list = list(as->list, t->nname->inlvar);
				as->rlist = list(as->rlist, ll->n);
			}
			ll=ll->next;
		}
		if(ll || t)
			fatal("arg count mismatch: %#T  vs %,H\n",  getinargx(fn->type), n->list);
	}

	if (as->rlist) {
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
	body = inlsubstlist(fn->inl);

	body = list(body, nod(OGOTO, inlretlabel, N));	// avoid 'not used' when function doesnt have return
	body = list(body, nod(OLABEL, inlretlabel, N));

	typechecklist(body, Etop);

	call = nod(OINLCALL, N, N);
	call->ninit = ninit;
	call->nbody = body;
	call->rlist = inlretvars;
	call->type = n->type;
	call->typecheck = 1;

	setlno(call, n->lineno);

	*np = call;

	inlfn =	saveinlfn;
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
	curfn->dcl = list(curfn->dcl, n);
	return n;
}

// Synthesize a variable to store the inlined function's results in.
static Node*
retvar(Type *t, int i)
{
	Node *n;

	snprint(namebuf, sizeof(namebuf), ".r%d", i);
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
