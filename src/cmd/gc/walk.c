// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"

static	Node*	walkprint(Node*, NodeList**, int);
static	Node*	conv(Node*, Type*);
static	Node*	mapfn(char*, Type*);
static	Node*	makenewvar(Type*, NodeList**, Node**);
static	Node*	ascompatee1(int, Node*, Node*, NodeList**);
static	NodeList*	ascompatee(int, NodeList*, NodeList*, NodeList**);
static	NodeList*	ascompatet(int, NodeList*, Type**, int, NodeList**);
static	NodeList*	ascompatte(int, Type**, NodeList*, int, NodeList**);
static	Node*	convas(Node*, NodeList**);
static	void	heapmoves(void);
static	NodeList*	paramstoheap(Type **argin, int out);
static	NodeList*	reorder1(NodeList*);
static	NodeList*	reorder3(NodeList*);

static	NodeList*	walkdefstack;

// can this code branch reach the end
// without an undcontitional RETURN
// this is hard, so it is conservative
static int
walkret(NodeList *l)
{
	Node *n;

loop:
	while(l && l->next)
		l = l->next;
	if(l == nil)
		return 1;

	// at this point, we have the last
	// statement of the function
	n = l->n;
	switch(n->op) {
	case OBLOCK:
		l = n->list;
		goto loop;

	case OGOTO:
	case ORETURN:
	case OPANIC:
		return 0;
		break;
	}

	// all other statements
	// will flow to the end
	return 1;
}

void
walk(Node *fn)
{
	char s[50];
	NodeList *l;
	Node *n;
	int lno;

	curfn = fn;
	if(debug['W']) {
		snprint(s, sizeof(s), "\nbefore %S", curfn->nname->sym);
		dumplist(s, curfn->nbody);
	}
	if(curfn->type->outtuple)
		if(walkret(curfn->nbody))
			yyerror("function ends without a return statement");
	typechecklist(curfn->nbody, Etop);
	lno = lineno;
	for(l=fn->dcl; l; l=l->next) {
		n = l->n;
		if(n->op != ONAME || n->class != PAUTO)
			continue;
		lineno = n->lineno;
		typecheck(&n, Erv | Easgn);	// only needed for unused variables
		if(!n->used && n->sym->name[0] != '&' && !nsyntaxerrors)
			yyerror("%S declared and not used", n->sym);
	}
	lineno = lno;
	if(nerrors != 0)
		return;
	walkstmtlist(curfn->nbody);
	if(debug['W']) {
		snprint(s, sizeof(s), "after walk %S", curfn->nname->sym);
		dumplist(s, curfn->nbody);
	}
	heapmoves();
	if(debug['W'] && curfn->enter != nil) {
		snprint(s, sizeof(s), "enter %S", curfn->nname->sym);
		dumplist(s, curfn->enter);
	}
}

static int nwalkdeftype;
static NodeList *methodqueue;

static void
domethod(Node *n)
{
	Node *nt;

	nt = n->type->nname;
	typecheck(&nt, Etype);
	if(nt->type == T) {
		// type check failed; leave empty func
		n->type->etype = TFUNC;
		n->type->nod = N;
		return;
	}
	*n->type = *nt->type;
	n->type->nod = N;
	checkwidth(n->type);
}

static void
walkdeftype(Node *n)
{
	int maplineno, embedlineno, lno;
	Type *t;
	NodeList *l;

	nwalkdeftype++;
	lno = lineno;
	setlineno(n);
	n->type->sym = n->sym;
	n->typecheck = 1;
	typecheck(&n->ntype, Etype);
	if((t = n->ntype->type) == T) {
		n->diag = 1;
		goto ret;
	}

	// copy new type and clear fields
	// that don't come along
	maplineno = n->type->maplineno;
	embedlineno = n->type->embedlineno;
	*n->type = *t;
	t = n->type;
	t->sym = n->sym;
	t->local = n->local;
	t->vargen = n->vargen;
	t->siggen = 0;
	t->method = nil;
	t->nod = N;
	t->printed = 0;
	t->deferwidth = 0;

	// double-check use of type as map key.
	if(maplineno) {
		lineno = maplineno;
		maptype(n->type, types[TBOOL]);
	}
	if(embedlineno) {
		lineno = embedlineno;
		if(isptr[t->etype])
			yyerror("embedded type cannot be a pointer");
	}

ret:
	lineno = lno;

	// if there are no type definitions going on, it's safe to
	// try to resolve the method types for the interfaces
	// we just read.
	if(nwalkdeftype == 1) {
		while((l = methodqueue) != nil) {
			methodqueue = nil;
			for(; l; l=l->next)
				domethod(l->n);
		}
	}
	nwalkdeftype--;
}

void
queuemethod(Node *n)
{
	if(nwalkdeftype == 0) {
		domethod(n);
		return;
	}
	methodqueue = list(methodqueue, n);
}

Node*
walkdef(Node *n)
{
	int lno;
	NodeList *init;
	Node *e;
	Type *t;
	NodeList *l;

	lno = lineno;
	setlineno(n);

	if(n->op == ONONAME) {
		if(!n->diag) {
			n->diag = 1;
			if(n->lineno != 0)
				lineno = n->lineno;
			yyerror("undefined: %S", n->sym);
		}
		return n;
	}

	if(n->walkdef == 1)
		return n;

	l = mal(sizeof *l);
	l->n = n;
	l->next = walkdefstack;
	walkdefstack = l;

	if(n->walkdef == 2) {
		flusherrors();
		print("walkdef loop:");
		for(l=walkdefstack; l; l=l->next)
			print(" %S", l->n->sym);
		print("\n");
		fatal("walkdef loop");
	}
	n->walkdef = 2;

	if(n->type != T || n->sym == S)	// builtin or no name
		goto ret;

	init = nil;
	switch(n->op) {
	default:
		fatal("walkdef %O", n->op);

	case OLITERAL:
		if(n->ntype != N) {
			typecheck(&n->ntype, Etype);
			n->type = n->ntype->type;
			n->ntype = N;
			if(n->type == T) {
				n->diag = 1;
				goto ret;
			}
		}
		e = n->defn;
		n->defn = N;
		if(e == N) {
			lineno = n->lineno;
			dump("walkdef nil defn", n);
			yyerror("xxx");
		}
		typecheck(&e, Erv | Eiota);
		if(e->type != T && e->op != OLITERAL) {
			yyerror("const initializer must be constant");
			goto ret;
		}
		t = n->type;
		if(t != T) {
			convlit(&e, t);
			if(!okforconst[t->etype])
				yyerror("invalid constant type %T", t);
		}
		n->val = e->val;
		n->type = e->type;
		break;

	case ONAME:
		if(n->ntype != N) {
			typecheck(&n->ntype, Etype);
			n->type = n->ntype->type;
			if(n->type == T) {
				n->diag = 1;
				goto ret;
			}
		}
		if(n->type != T)
			break;
		if(n->defn == N) {
			if(n->etype != 0)	// like OPRINTN
				break;
			if(nerrors > 0) {
				// Can have undefined variables in x := foo
				// that make x have an n->ndefn == nil.
				// If there are other errors anyway, don't
				// bother adding to the noise.
				break;
			}
			fatal("var without type, init: %S", n->sym);
		}
		if(n->defn->op == ONAME) {
			typecheck(&n->defn, Erv);
			n->type = n->defn->type;
			break;
		}
		typecheck(&n->defn, Etop);	// fills in n->type
		break;

	case OTYPE:
		if(curfn)
			defercheckwidth();
		n->walkdef = 1;
		n->type = typ(TFORW);
		n->type->sym = n->sym;
		walkdeftype(n);
		if(curfn)
			resumecheckwidth();
		break;

	case OPACK:
		// nothing to see here
		break;
	}

ret:
	if(walkdefstack->n != n)
		fatal("walkdefstack mismatch");
	l = walkdefstack;
	walkdefstack = l->next;

	lineno = lno;
	n->walkdef = 1;
	return n;
}

void
walkstmtlist(NodeList *l)
{
	for(; l; l=l->next)
		walkstmt(&l->n);
}

static int
samelist(NodeList *a, NodeList *b)
{
	for(; a && b; a=a->next, b=b->next)
		if(a->n != b->n)
			return 0;
	return a == b;
}

static int
paramoutheap(Node *fn)
{
	NodeList *l;
	
	for(l=fn->dcl; l; l=l->next) {
		switch(l->n->class) {
		case PPARAMOUT|PHEAP:
			return 1;
		case PAUTO:
		case PAUTO|PHEAP:
			// stop early - parameters are over
			return 0;
		}
	}
	return 0;
}

void
walkstmt(Node **np)
{
	NodeList *init;
	NodeList *ll, *rl;
	int cl, lno;
	Node *n, *f;

	n = *np;
	if(n == N)
		return;

	lno = lineno;
	setlineno(n);

	switch(n->op) {
	default:
		if(n->op == ONAME)
			yyerror("%S is not a top level statement", n->sym);
		else
			yyerror("%O is not a top level statement", n->op);
		dump("nottop", n);
		break;

	case OASOP:
	case OAS:
	case OAS2:
	case OAS2DOTTYPE:
	case OAS2RECV:
	case OAS2FUNC:
	case OAS2MAPW:
	case OAS2MAPR:
	case OCLOSE:
	case OCLOSED:
	case OCOPY:
	case OCALLMETH:
	case OCALLINTER:
	case OCALL:
	case OCALLFUNC:
	case OSEND:
	case ORECV:
	case OPRINT:
	case OPRINTN:
	case OPANIC:
	case OEMPTY:
	case ORECOVER:
		if(n->typecheck == 0)
			fatal("missing typecheck");
		init = n->ninit;
		n->ninit = nil;
		walkexpr(&n, &init);
		n->ninit = concat(init, n->ninit);
		break;

	case OBREAK:
	case ODCL:
	case OCONTINUE:
	case OFALL:
	case OGOTO:
	case OLABEL:
	case ODCLCONST:
	case ODCLTYPE:
		break;

	case OBLOCK:
		walkstmtlist(n->list);
		break;

	case OXCASE:
		yyerror("case statement out of place");
		n->op = OCASE;
	case OCASE:
		walkstmt(&n->right);
		break;

	case ODEFER:
		hasdefer = 1;
		switch(n->left->op) {
		case OPRINT:
		case OPRINTN:
			walkexprlist(n->left->list, &n->ninit);
			n->left = walkprint(n->left, &n->ninit, 1);
			break;
		default:
			walkexpr(&n->left, &n->ninit);
			break;
		}
		break;

	case OFOR:
		walkstmtlist(n->ninit);
		if(n->ntest != N) {
			walkstmtlist(n->ntest->ninit);
			init = n->ntest->ninit;
			n->ntest->ninit = nil;
			walkexpr(&n->ntest, &init);
			n->ntest->ninit = concat(init, n->ntest->ninit);
		}
		walkstmt(&n->nincr);
		walkstmtlist(n->nbody);
		break;

	case OIF:
		walkstmtlist(n->ninit);
		walkexpr(&n->ntest, &n->ninit);
		walkstmtlist(n->nbody);
		walkstmtlist(n->nelse);
		break;

	case OPROC:
		walkexpr(&n->left, &n->ninit);
		break;

	case ORETURN:
		walkexprlist(n->list, &n->ninit);
		if(n->list == nil)
			break;
		if((curfn->type->outnamed && count(n->list) > 1) || paramoutheap(curfn)) {
			// assign to the function out parameters,
			// so that reorder3 can fix up conflicts
			rl = nil;
			for(ll=curfn->dcl; ll != nil; ll=ll->next) {
				cl = ll->n->class & ~PHEAP;
				if(cl == PAUTO)
					break;
				if(cl == PPARAMOUT)
					rl = list(rl, ll->n);
			}
			if(samelist(rl, n->list)) {
				// special return in disguise
				n->list = nil;
				break;
			}
			if(count(n->list) == 1 && count(rl) > 1) {
				// OAS2FUNC in disguise
				f = n->list->n;
				if(f->op != OCALLFUNC && f->op != OCALLMETH && f->op != OCALLINTER)
					fatal("expected return of call, have %#N", f);
				n->list = concat(list1(f), ascompatet(n->op, rl, &f->type, 0, &n->ninit));
				break;
			}
			ll = ascompatee(n->op, rl, n->list, &n->ninit);
			n->list = reorder3(ll);
			break;
		}
		ll = ascompatte(n->op, getoutarg(curfn->type), n->list, 1, &n->ninit);
		n->list = ll;
		break;

	case OSELECT:
		walkselect(n);
		break;

	case OSWITCH:
		walkswitch(n);
		break;

	case ORANGE:
		walkrange(n);
		break;

	case OXFALL:
		yyerror("fallthrough statement out of place");
		n->op = OFALL;
		break;
	}

	*np = n;
}


/*
 * walk the whole tree of the body of an
 * expression or simple statement.
 * the types expressions are calculated.
 * compile-time constants are evaluated.
 * complex side effects like statements are appended to init
 */

void
walkexprlist(NodeList *l, NodeList **init)
{
	for(; l; l=l->next)
		walkexpr(&l->n, init);
}

void
walkexprlistsafe(NodeList *l, NodeList **init)
{
	for(; l; l=l->next) {
		l->n = safeexpr(l->n, init);
		walkexpr(&l->n, init);
	}
}

void
walkexpr(Node **np, NodeList **init)
{
	Node *r, *l, *var, *a;
	NodeList *ll, *lr, *lpost;
	Type *t;
	int et;
	int32 lno;
	Node *n, *fn;
	char buf[100], *p;

	n = *np;

	if(n == N)
		return;

	if(init == &n->ninit) {
		// not okay to use n->ninit when walking n,
		// because we might replace n with some other node
		// and would lose the init list.
		fatal("walkexpr init == &n->ninit");
	}

	// annoying case - not typechecked
	if(n->op == OKEY) {
		walkexpr(&n->left, init);
		walkexpr(&n->right, init);
		return;
	}

	lno = setlineno(n);

	if(debug['w'] > 1)
		dump("walk-before", n);

	if(n->typecheck != 1) {
		dump("missed typecheck", n);
		fatal("missed typecheck");
	}

	t = T;
	et = Txxx;

	switch(n->op) {
	default:
		dump("walk", n);
		fatal("walkexpr: switch 1 unknown op %N", n);
		goto ret;

	case OTYPE:
	case ONONAME:
	case OINDREG:
	case OEMPTY:
		goto ret;

	case ONOT:
	case OMINUS:
	case OPLUS:
	case OCOM:
	case OREAL:
	case OIMAG:
	case ODOT:
	case ODOTPTR:
	case ODOTMETH:
	case ODOTINTER:
	case OIND:
		walkexpr(&n->left, init);
		goto ret;

	case OLEN:
	case OCAP:
		walkexpr(&n->left, init);
		
		// replace len(*[10]int) with 10.
		// delayed until now to preserve side effects.
		t = n->left->type;
		if(isptr[t->etype])
			t = t->type;
		if(isfixedarray(t)) {
			safeexpr(n->left, init);
			nodconst(n, n->type, t->bound);
			n->typecheck = 1;
		}
		goto ret;
	
	case OLSH:
	case ORSH:
	case OAND:
	case OOR:
	case OXOR:
	case OANDAND:
	case OOROR:
	case OSUB:
	case OMUL:
	case OEQ:
	case ONE:
	case OLT:
	case OLE:
	case OGE:
	case OGT:
	case OADD:
	case OCMPLX:
		walkexpr(&n->left, init);
		walkexpr(&n->right, init);
		goto ret;

	case OPRINT:
	case OPRINTN:
		walkexprlist(n->list, init);
		n = walkprint(n, init, 0);
		goto ret;

	case OPANIC:
		n = mkcall("panic", T, init, n->left);
		goto ret;

	case ORECOVER:
		n = mkcall("recover", n->type, init, nod(OADDR, nodfp, N));
		goto ret;

	case OLITERAL:
		n->addable = 1;
		goto ret;

	case ONAME:
		if(!(n->class & PHEAP) && n->class != PPARAMREF)
			n->addable = 1;
		goto ret;

	case OCALLINTER:
		t = n->left->type;
		if(n->list && n->list->n->op == OAS)
			goto ret;
		walkexpr(&n->left, init);
		walkexprlist(n->list, init);
		ll = ascompatte(n->op, getinarg(t), n->list, 0, init);
		n->list = reorder1(ll);
		goto ret;

	case OCALLFUNC:
		t = n->left->type;
		if(n->list && n->list->n->op == OAS)
			goto ret;
		walkexpr(&n->left, init);
		walkexprlist(n->list, init);
		ll = ascompatte(n->op, getinarg(t), n->list, 0, init);
		n->list = reorder1(ll);
		if(isselect(n)) {
			// special prob with selectsend and selectrecv:
			// if chan is nil, they don't know big the channel
			// element is and therefore don't know how to find
			// the output bool, so we clear it before the call.
			Node *b;
			b = nodbool(0);
			typecheck(&b, Erv);
			lr = ascompatte(n->op, getoutarg(t), list1(b), 0, init);
			n->list = concat(n->list, lr);
		}
		goto ret;

	case OCALLMETH:
		t = n->left->type;
		if(n->list && n->list->n->op == OAS)
			goto ret;
		walkexpr(&n->left, init);
		walkexprlist(n->list, init);
		ll = ascompatte(n->op, getinarg(t), n->list, 0, init);
		lr = ascompatte(n->op, getthis(t), list1(n->left->left), 0, init);
		ll = concat(ll, lr);
		n->left->left = N;
		ullmancalc(n->left);
		n->list = reorder1(ll);
		goto ret;

	case OAS:
		*init = concat(*init, n->ninit);
		n->ninit = nil;
		walkexpr(&n->left, init);
		n->left = safeexpr(n->left, init);
		if(oaslit(n, init))
			goto ret;
		walkexpr(&n->right, init);
		l = n->left;
		r = n->right;
		if(l == N || r == N)
			goto ret;
		r = ascompatee1(n->op, l, r, init);
		if(r != N) {
			r->dodata = n->dodata;
			n = r;
		}
		goto ret;

	case OAS2:
		*init = concat(*init, n->ninit);
		n->ninit = nil;
		walkexprlistsafe(n->list, init);
		walkexprlistsafe(n->rlist, init);
		ll = ascompatee(OAS, n->list, n->rlist, init);
		ll = reorder3(ll);
		n = liststmt(ll);
		goto ret;

	case OAS2FUNC:
	as2func:
		// a,b,... = fn()
		*init = concat(*init, n->ninit);
		n->ninit = nil;
		r = n->rlist->n;
		walkexprlistsafe(n->list, init);
		walkexpr(&r, init);
		l = n->list->n;
		
		// all the really hard stuff - explicit function calls and so on -
		// is gone, but map assignments remain.
		// if there are map assignments here, assign via
		// temporaries, because ascompatet assumes
		// the targets can be addressed without function calls
		// and map index has an implicit one.
		lpost = nil;
		if(l->op == OINDEXMAP) {
			var = nod(OXXX, N, N);
			tempname(var, l->type);
			n->list->n = var;
			a = nod(OAS, l, var);
			typecheck(&a, Etop);
			lpost = list(lpost, a);
		}
		l = n->list->next->n;
		if(l->op == OINDEXMAP) {
			var = nod(OXXX, N, N);
			tempname(var, l->type);
			n->list->next->n = var;
			a = nod(OAS, l, var);
			typecheck(&a, Etop);
			lpost = list(lpost, a);
		}
		ll = ascompatet(n->op, n->list, &r->type, 0, init);
		walkexprlist(lpost, init);
		n = liststmt(concat(concat(list1(r), ll), lpost));
		goto ret;

	case OAS2RECV:
		// a,b = <-c
		*init = concat(*init, n->ninit);
		n->ninit = nil;
		r = n->rlist->n;
		walkexprlistsafe(n->list, init);
		walkexpr(&r->left, init);
		fn = chanfn("chanrecv2", 2, r->left->type);
		r = mkcall1(fn, getoutargx(fn->type), init, r->left);
		n->rlist->n = r;
		n->op = OAS2FUNC;
		goto as2func;

	case OAS2MAPR:
		// a,b = m[i];
		*init = concat(*init, n->ninit);
		n->ninit = nil;
		r = n->rlist->n;
		walkexprlistsafe(n->list, init);
		walkexpr(&r->left, init);
		fn = mapfn("mapaccess2", r->left->type);
		r = mkcall1(fn, getoutargx(fn->type), init, r->left, r->right);
		n->rlist = list1(r);
		n->op = OAS2FUNC;
		goto as2func;

	case OAS2MAPW:
		// map[] = a,b - mapassign2
		// a,b = m[i];
		*init = concat(*init, n->ninit);
		n->ninit = nil;
		walkexprlistsafe(n->list, init);
		l = n->list->n;
		t = l->left->type;
		n = mkcall1(mapfn("mapassign2", t), T, init, l->left, l->right, n->rlist->n, n->rlist->next->n);
		goto ret;

	case OAS2DOTTYPE:
		// a,b = i.(T)
		*init = concat(*init, n->ninit);
		n->ninit = nil;
		r = n->rlist->n;
		walkexprlistsafe(n->list, init);
		r->op = ODOTTYPE2;
		walkexpr(&r, init);
		ll = ascompatet(n->op, n->list, &r->type, 0, init);
		n = liststmt(concat(list1(r), ll));
		goto ret;

	case ODOTTYPE:
	case ODOTTYPE2:
		// Build name of function: assertI2E2 etc.
		strcpy(buf, "assert");
		p = buf+strlen(buf);
		if(isnilinter(n->left->type))
			*p++ = 'E';
		else
			*p++ = 'I';
		*p++ = '2';
		if(isnilinter(n->type))
			*p++ = 'E';
		else if(isinter(n->type))
			*p++ = 'I';
		else
			*p++ = 'T';
		if(n->op == ODOTTYPE2)
			*p++ = '2';
		*p = '\0';
	
		fn = syslook(buf, 1);
		ll = list1(typename(n->type));
		ll = list(ll, n->left);
		argtype(fn, n->left->type);
		argtype(fn, n->type);
		n = nod(OCALL, fn, N);
		n->list = ll;
		typecheck(&n, Erv | Efnstruct);
		walkexpr(&n, init);
		goto ret;

	case OCONVIFACE:
		// Build name of function: convI2E etc.
		// Not all names are possible
		// (e.g., we'll never generate convE2E or convE2I).
		walkexpr(&n->left, init);
		strcpy(buf, "conv");
		p = buf+strlen(buf);
		if(isnilinter(n->left->type))
			*p++ = 'E';
		else if(isinter(n->left->type))
			*p++ = 'I';
		else
			*p++ = 'T';
		*p++ = '2';
		if(isnilinter(n->type))
			*p++ = 'E';
		else
			*p++ = 'I';
		*p = '\0';
		
		fn = syslook(buf, 1);
		ll = nil;
		if(!isinter(n->left->type))
			ll = list(ll, typename(n->left->type));
		if(!isnilinter(n->type))
			ll = list(ll, typename(n->type));
		ll = list(ll, n->left);
		argtype(fn, n->left->type);
		argtype(fn, n->type);
		dowidth(fn->type);
		n = nod(OCALL, fn, N);
		n->list = ll;
		typecheck(&n, Erv);
		walkexpr(&n, init);
		goto ret;

	case OCONV:
	case OCONVNOP:
		if(thechar == '5') {
			if(isfloat[n->left->type->etype] &&
			   (n->type->etype == TINT64 || n->type->etype == TUINT64)) {
				n = mkcall("float64toint64", n->type, init, conv(n->left, types[TFLOAT64]));
				goto ret;
			}
			if((n->left->type->etype == TINT64 || n->left->type->etype == TUINT64) &&
			   isfloat[n->type->etype]) {
				n = mkcall("int64tofloat64", n->type, init, conv(n->left, types[TINT64]));
				goto ret;
			}
		}
		walkexpr(&n->left, init);
		goto ret;

	case OASOP:
		n->left = safeexpr(n->left, init);
		walkexpr(&n->left, init);
		l = n->left;
		walkexpr(&n->right, init);
		if(n->etype == OANDNOT) {
			n->etype = OAND;
			n->right = nod(OCOM, n->right, N);
			typecheck(&n->right, Erv);
		}

		/*
		 * on 32-bit arch, rewrite 64-bit ops into l = l op r.
		 * on 386, rewrite float ops into l = l op r.
		 * everywhere, rewrite map ops into l = l op r.
		 * everywhere, rewrite string += into l = l op r.
		 * TODO(rsc): Maybe this rewrite should be done always?
		 */
		et = n->left->type->etype;
		if((widthptr == 4 && (et == TUINT64 || et == TINT64)) ||
		   (thechar == '8' && isfloat[et]) ||
		   l->op == OINDEXMAP ||
		   et == TSTRING) {
			l = safeexpr(n->left, init);
			a = l;
			if(a->op == OINDEXMAP) {
				// map index has "lhs" bit set in a->etype.
				// make a copy so we can clear it on the rhs.
				a = nod(OXXX, N, N);
				*a = *l;
				a->etype = 0;
			}
			r = nod(OAS, l, nod(n->etype, a, n->right));
			typecheck(&r, Etop);
			walkexpr(&r, init);
			n = r;
		}
		goto ret;

	case OANDNOT:
		walkexpr(&n->left, init);
		walkexpr(&n->right, init);
		n->op = OAND;
		n->right = nod(OCOM, n->right, N);
		typecheck(&n->right, Erv);
		goto ret;

	case ODIV:
	case OMOD:
		walkexpr(&n->left, init);
		walkexpr(&n->right, init);
		/*
		 * rewrite complex div into function call.
		 */
		et = n->left->type->etype;
		if(iscomplex[et] && n->op == ODIV) {
			n = mkcall("complex128div", n->type, init,
				conv(n->left, types[TCOMPLEX128]),
				conv(n->right, types[TCOMPLEX128]));
			goto ret;
		}
		/*
		 * rewrite div and mod into function calls
		 * on 32-bit architectures.
		 */
		if(widthptr > 4 || (et != TUINT64 && et != TINT64))
			goto ret;
		if(et == TINT64)
			strcpy(namebuf, "int64");
		else
			strcpy(namebuf, "uint64");
		if(n->op == ODIV)
			strcat(namebuf, "div");
		else
			strcat(namebuf, "mod");
		n = mkcall(namebuf, n->type, init,
			conv(n->left, types[et]), conv(n->right, types[et]));
		goto ret;

	case OINDEX:
		walkexpr(&n->left, init);
		walkexpr(&n->right, init);

		// if range of type cannot exceed static array bound,
		// disable bounds check
		if(!isslice(n->left->type))
		if(n->right->type->width < 4)
		if((1<<(8*n->right->type->width)) <= n->left->type->bound)
			n->etype = 1;

		goto ret;

	case OINDEXMAP:
		if(n->etype == 1)
			goto ret;
		t = n->left->type;
		n = mkcall1(mapfn("mapaccess1", t), t->type, init, n->left, n->right);
		goto ret;

	case ORECV:
		walkexpr(&n->left, init);
		walkexpr(&n->right, init);
		n = mkcall1(chanfn("chanrecv1", 2, n->left->type), n->type, init, n->left);
		goto ret;

	case OSLICE:
		walkexpr(&n->left, init);
		n->left = safeexpr(n->left, init);
		walkexpr(&n->right->left, init);
		n->right->left = safeexpr(n->right->left, init);
		walkexpr(&n->right->right, init);
		n->right->right = safeexpr(n->right->right, init);
		// dynamic slice
		// sliceslice(old []any, lb int, hb int, width int) (ary []any)
		// sliceslice1(old []any, lb int, width int) (ary []any)
		t = n->type;
		if(n->right->right != N) {
			fn = syslook("sliceslice", 1);
			argtype(fn, t->type);			// any-1
			argtype(fn, t->type);			// any-2
			n = mkcall1(fn, t, init,
				n->left,
				conv(n->right->left, types[TINT]),
				conv(n->right->right, types[TINT]),
				nodintconst(t->type->width));
		} else {
			fn = syslook("sliceslice1", 1);
			argtype(fn, t->type);			// any-1
			argtype(fn, t->type);			// any-2
			n = mkcall1(fn, t, init,
				n->left,
				conv(n->right->left, types[TINT]),
				nodintconst(t->type->width));
		}
		goto ret;

	case OSLICEARR:
		walkexpr(&n->left, init);
		n->left = safeexpr(n->left, init);
		walkexpr(&n->right->left, init);
		n->right->left = safeexpr(n->right->left, init);
		walkexpr(&n->right->right, init);
		n->right->right = safeexpr(n->right->right, init);
		// static slice
		// slicearray(old *any, nel int, lb int, hb int, width int) (ary []any)
		t = n->type;
		fn = syslook("slicearray", 1);
		argtype(fn, n->left->type);	// any-1
		argtype(fn, t->type);			// any-2
		if(n->right->right == N)
			r = nodintconst(n->left->type->bound);
		else
			r = conv(n->right->right, types[TINT]);
		n = mkcall1(fn, t, init,
			nod(OADDR, n->left, N), nodintconst(n->left->type->bound),
			conv(n->right->left, types[TINT]),
			r,
			nodintconst(t->type->width));
		goto ret;

	case OCONVSLICE:
		// slicearray(old *any, nel int, lb int, hb int, width int) (ary []any)
		fn = syslook("slicearray", 1);
		argtype(fn, n->left->type->type);		// any-1
		argtype(fn, n->type->type);			// any-2
		n = mkcall1(fn, n->type, init, n->left,
			nodintconst(n->left->type->type->bound),
			nodintconst(0),
			nodintconst(n->left->type->type->bound),
			nodintconst(n->type->type->width));
		goto ret;

	case OADDR:;
		Node *nvar, *nstar;

		// turn &Point(1, 2) or &[]int(1, 2) or &[...]int(1, 2) into allocation.
		// initialize with
		//	nvar := new(*Point);
		//	*nvar = Point(1, 2);
		// and replace expression with nvar
		switch(n->left->op) {
		case OARRAYLIT:
		case OMAPLIT:
		case OSTRUCTLIT:
			nvar = makenewvar(n->type, init, &nstar);
			anylit(0, n->left, nstar, init);
			n = nvar;
			goto ret;
		}

		walkexpr(&n->left, init);
		goto ret;

	case ONEW:
		n = callnew(n->type->type);
		goto ret;

	case OCMPSTR:
		// If one argument to the comparison is an empty string,
		// comparing the lengths instead will yield the same result
		// without the function call.
		if((isconst(n->left, CTSTR) && n->left->val.u.sval->len == 0) ||
		   (isconst(n->right, CTSTR) && n->right->val.u.sval->len == 0)) {
			r = nod(n->etype, nod(OLEN, n->left, N), nod(OLEN, n->right, N));
			typecheck(&r, Erv);
			walkexpr(&r, init);
			n = r;
			goto ret;
		}

		// s + "badgerbadgerbadger" == "badgerbadgerbadger"
		if((n->etype == OEQ || n->etype == ONE) &&
		   isconst(n->right, CTSTR) &&
		   n->left->op == OADDSTR && isconst(n->left->right, CTSTR) &&
		   cmpslit(n->right, n->left->right) == 0) {
			r = nod(n->etype, nod(OLEN, n->left->left, N), nodintconst(0));
			typecheck(&r, Erv);
			walkexpr(&r, init);
			n = r;
			goto ret;
		}

		// sys_cmpstring(s1, s2) :: 0
		r = mkcall("cmpstring", types[TINT], init,
			conv(n->left, types[TSTRING]),
			conv(n->right, types[TSTRING]));
		r = nod(n->etype, r, nodintconst(0));
		typecheck(&r, Erv);
		n = r;
		goto ret;

	case OADDSTR:
		// sys_catstring(s1, s2)
		n = mkcall("catstring", n->type, init,
			conv(n->left, types[TSTRING]),
			conv(n->right, types[TSTRING]));
		goto ret;

	case OSLICESTR:
		// sys_slicestring(s, lb, hb)
		if(n->right->right) {
			n = mkcall("slicestring", n->type, init,
				conv(n->left, types[TSTRING]),
				conv(n->right->left, types[TINT]),
				conv(n->right->right, types[TINT]));
		} else {
			n = mkcall("slicestring1", n->type, init,
				conv(n->left, types[TSTRING]),
				conv(n->right->left, types[TINT]));
		}
		goto ret;

	case OINDEXSTR:
		// TODO(rsc): should be done in back end
		// sys_indexstring(s, i)
		n = mkcall("indexstring", n->type, init,
			conv(n->left, types[TSTRING]),
			conv(n->right, types[TINT]));
		goto ret;

	case OCOPY:
		fn = syslook("slicecopy", 1);
		argtype(fn, n->left->type);
		argtype(fn, n->right->type);
		n = mkcall1(fn, n->type, init,
			n->left, n->right,
			nodintconst(n->left->type->type->width));
		goto ret;

	case OCLOSE:
		// cannot use chanfn - closechan takes any, not chan any
		fn = syslook("closechan", 1);
		argtype(fn, n->left->type);
		n = mkcall1(fn, T, init, n->left);
		goto ret;

	case OCLOSED:
		// cannot use chanfn - closechan takes any, not chan any
		fn = syslook("closedchan", 1);
		argtype(fn, n->left->type);
		n = mkcall1(fn, n->type, init, n->left);
		goto ret;

	case OMAKECHAN:
		n = mkcall1(chanfn("makechan", 1, n->type), n->type, init,
			typename(n->type->type),
			conv(n->left, types[TINT64]));
		goto ret;

	case OMAKEMAP:
		t = n->type;

		fn = syslook("makemap", 1);
		argtype(fn, t->down);	// any-1
		argtype(fn, t->type);	// any-2

		n = mkcall1(fn, n->type, init,
			typename(t->down),	// key type
			typename(t->type),		// value type
			conv(n->left, types[TINT64]));
		goto ret;

	case OMAKESLICE:
		// makeslice(t *Type, nel int64, max int64) (ary []any)
		l = n->left;
		r = n->right;
		if(r == nil)
			l = r = safeexpr(l, init);
		t = n->type;
		fn = syslook("makeslice", 1);
		argtype(fn, t->type);			// any-1
		n = mkcall1(fn, n->type, init,
			typename(n->type),
			conv(l, types[TINT64]),
			conv(r, types[TINT64]));
		goto ret;

	case ORUNESTR:
		// sys_intstring(v)
		n = mkcall("intstring", n->type, init,
			conv(n->left, types[TINT64]));
		goto ret;

	case OARRAYBYTESTR:
		// slicebytetostring([]byte) string;
		n = mkcall("slicebytetostring", n->type, init, n->left);
		goto ret;

	case OARRAYRUNESTR:
		// sliceinttostring([]int) string;
		n = mkcall("sliceinttostring", n->type, init, n->left);
		goto ret;

	case OSTRARRAYBYTE:
		// stringtoslicebyte(string) []byte;
		n = mkcall("stringtoslicebyte", n->type, init, n->left);
		goto ret;

	case OSTRARRAYRUNE:
		// stringtosliceint(string) []int
		n = mkcall("stringtosliceint", n->type, init, n->left);
		goto ret;

	case OCMPIFACE:
		// ifaceeq(i1 any-1, i2 any-2) (ret bool);
		if(!eqtype(n->left->type, n->right->type))
			fatal("ifaceeq %O %T %T", n->op, n->left->type, n->right->type);
		if(isnilinter(n->left->type))
			fn = syslook("efaceeq", 1);
		else
			fn = syslook("ifaceeq", 1);
		argtype(fn, n->right->type);
		argtype(fn, n->left->type);
		r = mkcall1(fn, n->type, init, n->left, n->right);
		if(n->etype == ONE) {
			r = nod(ONOT, r, N);
			typecheck(&r, Erv);
		}
		n = r;
		goto ret;

	case OARRAYLIT:
	case OMAPLIT:
	case OSTRUCTLIT:
		nvar = nod(OXXX, N, N);
		tempname(nvar, n->type);
		anylit(0, n, nvar, init);
		n = nvar;
		goto ret;

	case OSEND:
		n = mkcall1(chanfn("chansend1", 2, n->left->type), T, init, n->left, n->right);
		goto ret;

	case OSENDNB:
		n = mkcall1(chanfn("chansend2", 2, n->left->type), n->type, init, n->left, n->right);
		goto ret;

	case OCLOSURE:
		n = walkclosure(n, init);
		goto ret;
	}
	fatal("missing switch %O", n->op);

ret:
	if(debug['w'] && n != N)
		dump("walk", n);

	ullmancalc(n);
	lineno = lno;
	*np = n;
}

static Node*
makenewvar(Type *t, NodeList **init, Node **nstar)
{
	Node *nvar, *nas;

	nvar = nod(OXXX, N, N);
	tempname(nvar, t);
	nas = nod(OAS, nvar, callnew(t->type));
	typecheck(&nas, Etop);
	walkexpr(&nas, init);
	*init = list(*init, nas);

	*nstar = nod(OIND, nvar, N);
	typecheck(nstar, Erv);
	return nvar;
}

static Node*
ascompatee1(int op, Node *l, Node *r, NodeList **init)
{
	return convas(nod(OAS, l, r), init);
}

static NodeList*
ascompatee(int op, NodeList *nl, NodeList *nr, NodeList **init)
{
	NodeList *ll, *lr, *nn;

	/*
	 * check assign expression list to
	 * a expression list. called in
	 *	expr-list = expr-list
	 */

	// ensure order of evaluation for function calls
	for(ll=nl; ll; ll=ll->next)
		ll->n = safeexpr(ll->n, init);
	for(lr=nr; lr; lr=lr->next)
		lr->n = safeexpr(lr->n, init);

	nn = nil;
	for(ll=nl, lr=nr; ll && lr; ll=ll->next, lr=lr->next)
		nn = list(nn, ascompatee1(op, ll->n, lr->n, init));

	// cannot happen: caller checked that lists had same length
	if(ll || lr)
		yyerror("error in shape across %O", op);
	return nn;
}

/*
 * l is an lv and rt is the type of an rv
 * return 1 if this implies a function call
 * evaluating the lv or a function call
 * in the conversion of the types
 */
static int
fncall(Node *l, Type *rt)
{
	if(l->ullman >= UINF)
		return 1;
	if(eqtype(l->type, rt))
		return 0;
	return 1;
}

static NodeList*
ascompatet(int op, NodeList *nl, Type **nr, int fp, NodeList **init)
{
	Node *l, *tmp, *a;
	NodeList *ll;
	Type *r;
	Iter saver;
	int ucount;
	NodeList *nn, *mm;

	/*
	 * check assign type list to
	 * a expression list. called in
	 *	expr-list = func()
	 */
	r = structfirst(&saver, nr);
	nn = nil;
	mm = nil;
	ucount = 0;
	for(ll=nl; ll; ll=ll->next) {
		if(r == T)
			break;
		l = ll->n;
		if(isblank(l)) {
			r = structnext(&saver);
			continue;
		}

		// any lv that causes a fn call must be
		// deferred until all the return arguments
		// have been pulled from the output arguments
		if(fncall(l, r->type)) {
			tmp = nod(OXXX, N, N);
			tempname(tmp, r->type);
			typecheck(&tmp, Erv);
			a = nod(OAS, l, tmp);
			a = convas(a, init);
			mm = list(mm, a);
			l = tmp;
		}

		a = nod(OAS, l, nodarg(r, fp));
		a = convas(a, init);
		ullmancalc(a);
		if(a->ullman >= UINF)
			ucount++;
		nn = list(nn, a);
		r = structnext(&saver);
	}

	if(ll != nil || r != T)
		yyerror("assignment count mismatch: %d = %d",
			count(nl), structcount(*nr));
	if(ucount)
		fatal("reorder2: too many function calls evaluating parameters");
	return concat(nn, mm);
}

 /*
 * package all the arguments that match a ... T parameter into a []T.
 */
static NodeList*
mkdotargslice(NodeList *lr0, NodeList *nn, Type *l, int fp, NodeList **init)
{
	Node *a, *n;
	Type *tslice;

	tslice = typ(TARRAY);
	tslice->type = l->type->type;
	tslice->bound = -1;

	n = nod(OCOMPLIT, N, typenod(tslice));
	n->list = lr0;
	typecheck(&n, Erv);
	if(n->type == T)
		fatal("mkdotargslice: typecheck failed");
	walkexpr(&n, init);

	a = nod(OAS, nodarg(l, fp), n);
	nn = list(nn, convas(a, init));
	return nn;
}

/*
 * helpers for shape errors
 */
static void
dumptypes(Type **nl, char *what)
{
	int first;
	Type *l;
	Iter savel;

	l = structfirst(&savel, nl);
	print("\t");
	first = 1;
	for(l = structfirst(&savel, nl); l != T; l = structnext(&savel)) {
		if(first)
			first = 0;
		else
			print(", ");
		print("%T", l);
	}
	if(first)
		print("[no arguments %s]", what);
	print("\n");
}

static void
dumpnodetypes(NodeList *l, char *what)
{
	int first;
	Node *r;

	print("\t");
	first = 1;
	for(; l; l=l->next) {
		r = l->n;
		if(first)
			first = 0;
		else
			print(", ");
		print("%T", r->type);
	}
	if(first)
		print("[no arguments %s]", what);
	print("\n");
}

/*
 * check assign expression list to
 * a type list. called in
 *	return expr-list
 *	func(expr-list)
 */
static NodeList*
ascompatte(int op, Type **nl, NodeList *lr, int fp, NodeList **init)
{
	Type *l, *ll;
	Node *r, *a;
	NodeList *nn, *lr0, *alist;
	Iter savel;

	lr0 = lr;
	l = structfirst(&savel, nl);
	r = N;
	if(lr)
		r = lr->n;
	nn = nil;
	
	// f(g()) where g has multiple return values
	if(r != N && lr->next == nil && r->type->etype == TSTRUCT && r->type->funarg) {
		// optimization - can do block copy
		if(eqtypenoname(r->type, *nl)) {
			a = nodarg(*nl, fp);
			a->type = r->type;
			nn = list1(convas(nod(OAS, a, r), init));
			goto ret;
		}
		
		// conversions involved.
		// copy into temporaries.
		alist = nil;
		for(l=structfirst(&savel, &r->type); l; l=structnext(&savel)) {
			a = nod(OXXX, N, N);
			tempname(a, l->type);
			alist = list(alist, a);
		}
		a = nod(OAS2, N, N);
		a->list = alist;
		a->rlist = lr;
		typecheck(&a, Etop);
		walkstmt(&a);
		*init = list(*init, a);
		lr = alist;
		r = lr->n;
		l = structfirst(&savel, nl);
	}

loop:
	if(l != T && l->isddd) {
		// the ddd parameter must be last
		ll = structnext(&savel);
		if(ll != T)
			yyerror("... must be last argument");

		// special case --
		// only if we are assigning a single ddd
		// argument to a ddd parameter then it is
		// passed thru unencapsulated
		if(r != N && lr->next == nil && r->isddd && eqtype(l->type, r->type)) {
			a = nod(OAS, nodarg(l, fp), r);
			a = convas(a, init);
			nn = list(nn, a);
			goto ret;
		}

		// normal case -- make a slice of all
		// remaining arguments and pass it to
		// the ddd parameter.
		nn = mkdotargslice(lr, nn, l, fp, init);
		goto ret;
	}

	if(l == T || r == N) {
		if(l != T || r != N) {
			if(l != T)
				yyerror("not enough arguments to %O", op);
			else
				yyerror("too many arguments to %O", op);
			dumptypes(nl, "expected");
			dumpnodetypes(lr0, "given");
		}
		goto ret;
	}

	a = nod(OAS, nodarg(l, fp), r);
	a = convas(a, init);
	nn = list(nn, a);

	l = structnext(&savel);
	r = N;
	lr = lr->next;
	if(lr != nil)
		r = lr->n;
	goto loop;

ret:
	for(lr=nn; lr; lr=lr->next)
		lr->n->typecheck = 1;
	return nn;
}

// generate code for print
static Node*
walkprint(Node *nn, NodeList **init, int defer)
{
	Node *r;
	Node *n;
	NodeList *l, *all;
	Node *on;
	Type *t;
	int notfirst, et, op;
	NodeList *calls, *intypes, *args;
	Fmt fmt;

	on = nil;
	op = nn->op;
	all = nn->list;
	calls = nil;
	notfirst = 0;
	intypes = nil;
	args = nil;

	memset(&fmt, 0, sizeof fmt);
	if(defer) {
		// defer print turns into defer printf with format string
		fmtstrinit(&fmt);
		intypes = list(intypes, nod(ODCLFIELD, N, typenod(types[TSTRING])));
		args = list1(nod(OXXX, N, N));
	}

	for(l=all; l; l=l->next) {
		if(notfirst) {
			if(defer)
				fmtprint(&fmt, " ");
			else
				calls = list(calls, mkcall("printsp", T, init));
		}
		notfirst = op == OPRINTN;

		n = l->n;
		if(n->op == OLITERAL) {
			switch(n->val.ctype) {
			case CTINT:
				defaultlit(&n, types[TINT64]);
				break;
			case CTFLT:
				defaultlit(&n, types[TFLOAT64]);
				break;
			}
		}
		if(n->op != OLITERAL && n->type && n->type->etype == TIDEAL)
			defaultlit(&n, types[TINT64]);
		defaultlit(&n, nil);
		l->n = n;
		if(n->type == T || n->type->etype == TFORW)
			continue;

		t = n->type;
		et = n->type->etype;
		if(isinter(n->type)) {
			if(defer) {
				if(isnilinter(n->type))
					fmtprint(&fmt, "%%e");
				else
					fmtprint(&fmt, "%%i");
			} else {
				if(isnilinter(n->type))
					on = syslook("printeface", 1);
				else
					on = syslook("printiface", 1);
				argtype(on, n->type);		// any-1
			}
		} else if(isptr[et] || et == TCHAN || et == TMAP || et == TFUNC) {
			if(defer) {
				fmtprint(&fmt, "%%p");
			} else {
				on = syslook("printpointer", 1);
				argtype(on, n->type);	// any-1
			}
		} else if(isslice(n->type)) {
			if(defer) {
				fmtprint(&fmt, "%%a");
			} else {
				on = syslook("printslice", 1);
				argtype(on, n->type);	// any-1
			}
		} else if(isint[et]) {
			if(defer) {
				if(et == TUINT64)
					fmtprint(&fmt, "%%U");
				else {
					fmtprint(&fmt, "%%D");
					t = types[TINT64];
				}
			} else {
				if(et == TUINT64)
					on = syslook("printuint", 0);
				else
					on = syslook("printint", 0);
			}
		} else if(isfloat[et]) {
			if(defer) {
				fmtprint(&fmt, "%%f");
				t = types[TFLOAT64];
			} else
				on = syslook("printfloat", 0);
		} else if(iscomplex[et]) {
			if(defer) {
				fmtprint(&fmt, "%%C");
				t = types[TCOMPLEX128];
			} else
				on = syslook("printcomplex", 0);
		} else if(et == TBOOL) {
			if(defer)
				fmtprint(&fmt, "%%t");
			else
				on = syslook("printbool", 0);
		} else if(et == TSTRING) {
			if(defer)
				fmtprint(&fmt, "%%S");
			else
				on = syslook("printstring", 0);
		} else {
			badtype(OPRINT, n->type, T);
			continue;
		}

		if(!defer) {
			t = *getinarg(on->type);
			if(t != nil)
				t = t->type;
			if(t != nil)
				t = t->type;
		}

		if(!eqtype(t, n->type)) {
			n = nod(OCONV, n, N);
			n->type = t;
		}

		if(defer) {
			intypes = list(intypes, nod(ODCLFIELD, N, typenod(t)));
			args = list(args, n);
		} else {
			r = nod(OCALL, on, N);
			r->list = list1(n);
			calls = list(calls, r);
		}
	}

	if(defer) {
		if(op == OPRINTN)
			fmtprint(&fmt, "\n");
		on = syslook("printf", 1);
		on->type = functype(nil, intypes, nil);
		args->n = nod(OLITERAL, N, N);
		args->n->val.ctype = CTSTR;
		args->n->val.u.sval = strlit(fmtstrflush(&fmt));
		r = nod(OCALL, on, N);
		r->list = args;
		typecheck(&r, Etop);
		walkexpr(&r, init);
	} else {
		if(op == OPRINTN)
			calls = list(calls, mkcall("printnl", T, nil));
		typechecklist(calls, Etop);
		walkexprlist(calls, init);

		r = nod(OEMPTY, N, N);
		typecheck(&r, Etop);
		walkexpr(&r, init);
		r->ninit = calls;
	}
	return r;
}

Node*
callnew(Type *t)
{
	Node *fn;

	dowidth(t);
	fn = syslook("mal", 1);
	argtype(fn, t);
	return mkcall1(fn, ptrto(t), nil, nodintconst(t->width));
}

static Node*
convas(Node *n, NodeList **init)
{
	Node *l, *r;
	Type *lt, *rt;

	if(n->op != OAS)
		fatal("convas: not OAS %O", n->op);
	n->typecheck = 1;

	lt = T;
	rt = T;

	l = n->left;
	r = n->right;
	if(l == N || r == N)
		goto out;

	lt = l->type;
	rt = r->type;
	if(lt == T || rt == T)
		goto out;

	if(isblank(n->left)) {
		defaultlit(&n->right, T);
		goto out;
	}

	if(n->left->op == OINDEXMAP) {
		n = mkcall1(mapfn("mapassign1", n->left->left->type), T, init,
			n->left->left, n->left->right, n->right);
		goto out;
	}
	
	if(eqtype(lt, rt))
		goto out;
	
	n->right = assignconv(r, lt, "assignment");
	walkexpr(&n->right, init);

out:
	ullmancalc(n);
	return n;
}

/*
 * from ascompat[te]
 * evaluating actual function arguments.
 *	f(a,b)
 * if there is exactly one function expr,
 * then it is done first. otherwise must
 * make temp variables
 */
NodeList*
reorder1(NodeList *all)
{
	Node *f, *a, *n;
	NodeList *l, *r, *g;
	int c, d, t;

	c = 0;	// function calls
	t = 0;	// total parameters

	for(l=all; l; l=l->next) {
		n = l->n;
		t++;
		ullmancalc(n);
		if(n->ullman >= UINF)
			c++;
	}
	if(c == 0 || t == 1)
		return all;

	g = nil;	// fncalls assigned to tempnames
	f = N;	// last fncall assigned to stack
	r = nil;	// non fncalls and tempnames assigned to stack
	d = 0;
	for(l=all; l; l=l->next) {
		n = l->n;
		if(n->ullman < UINF) {
			r = list(r, n);
			continue;
		}
		d++;
		if(d == c) {
			f = n;
			continue;
		}

		// make assignment of fncall to tempname
		a = nod(OXXX, N, N);
		tempname(a, n->right->type);
		a = nod(OAS, a, n->right);
		g = list(g, a);

		// put normal arg assignment on list
		// with fncall replaced by tempname
		n->right = a->left;
		r = list(r, n);
	}

	if(f != N)
		g = list(g, f);
	return concat(g, r);
}

/*
 * from ascompat[ee]
 *	a,b = c,d
 * simultaneous assignment. there cannot
 * be later use of an earlier lvalue.
 */

static int
vmatch2(Node *l, Node *r)
{
	NodeList *ll;

	/*
	 * isolate all right sides
	 */
	if(r == N)
		return 0;
	switch(r->op) {
	case ONAME:
		// match each right given left
		if(l == r)
			return 1;
	case OLITERAL:
		return 0;
	}
	if(vmatch2(l, r->left))
		return 1;
	if(vmatch2(l, r->right))
		return 1;
	for(ll=r->list; ll; ll=ll->next)
		if(vmatch2(l, ll->n))
			return 1;
	return 0;
}

int
vmatch1(Node *l, Node *r)
{
	NodeList *ll;

	/*
	 * isolate all left sides
	 */
	if(l == N || r == N)
		return 0;
	switch(l->op) {
	case ONAME:
		switch(l->class) {
		case PPARAM:
		case PPARAMREF:
		case PAUTO:
			break;
		default:
			// assignment to non-stack variable
			// must be delayed if right has function calls.
			if(r->ullman >= UINF)
				return 1;
			break;
		}
		return vmatch2(l, r);
	case OLITERAL:
		return 0;
	}
	if(vmatch1(l->left, r))
		return 1;
	if(vmatch1(l->right, r))
		return 1;
	for(ll=l->list; ll; ll=ll->next)
		if(vmatch1(ll->n, r))
			return 1;
	return 0;
}

NodeList*
reorder3(NodeList *all)
{
	Node *n1, *n2, *q;
	int c1, c2;
	NodeList *l1, *l2, *r;

	r = nil;
	for(l1=all, c1=0; l1; l1=l1->next, c1++) {
		n1 = l1->n;
		for(l2=all, c2=0; l2; l2=l2->next, c2++) {
			n2 = l2->n;
			if(c2 > c1) {
				if(vmatch1(n1->left, n2->right)) {
					// delay assignment to n1->left
					q = nod(OXXX, N, N);
					tempname(q, n1->right->type);
					q = nod(OAS, n1->left, q);
					n1->left = q->right;
					r = list(r, q);
					break;
				}
			}
		}
	}
	return concat(all, r);
}

/*
 * walk through argin parameters.
 * generate and return code to allocate
 * copies of escaped parameters to the heap.
 */
static NodeList*
paramstoheap(Type **argin, int out)
{
	Type *t;
	Iter savet;
	Node *v;
	NodeList *nn;

	nn = nil;
	for(t = structfirst(&savet, argin); t != T; t = structnext(&savet)) {
		v = t->nname;
		if(v == N && out && hasdefer) {
			// Defer might stop a panic and show the
			// return values as they exist at the time of panic.
			// Make sure to zero them on entry to the function.
			nn = list(nn, nod(OAS, nodarg(t, 1), N));
		}
		if(v == N || !(v->class & PHEAP))
			continue;

		// generate allocation & copying code
		if(v->alloc == nil)
			v->alloc = callnew(v->type);
		nn = list(nn, nod(OAS, v->heapaddr, v->alloc));
		if((v->class & ~PHEAP) != PPARAMOUT)
			nn = list(nn, nod(OAS, v, v->stackparam));
	}
	return nn;
}

/*
 * walk through argout parameters copying back to stack
 */
static NodeList*
returnsfromheap(Type **argin)
{
	Type *t;
	Iter savet;
	Node *v;
	NodeList *nn;

	nn = nil;
	for(t = structfirst(&savet, argin); t != T; t = structnext(&savet)) {
		v = t->nname;
		if(v == N || v->class != (PHEAP|PPARAMOUT))
			continue;
		nn = list(nn, nod(OAS, v->stackparam, v));
	}
	return nn;
}

/*
 * take care of migrating any function in/out args
 * between the stack and the heap.  adds code to
 * curfn's before and after lists.
 */
static void
heapmoves(void)
{
	NodeList *nn;

	nn = paramstoheap(getthis(curfn->type), 0);
	nn = concat(nn, paramstoheap(getinarg(curfn->type), 0));
	nn = concat(nn, paramstoheap(getoutarg(curfn->type), 1));
	curfn->enter = concat(curfn->enter, nn);
	curfn->exit = returnsfromheap(getoutarg(curfn->type));
}

static Node*
vmkcall(Node *fn, Type *t, NodeList **init, va_list va)
{
	int i, n;
	Node *r;
	NodeList *args;

	if(fn->type == T || fn->type->etype != TFUNC)
		fatal("mkcall %#N %T", fn, fn->type);

	args = nil;
	n = fn->type->intuple;
	for(i=0; i<n; i++)
		args = list(args, va_arg(va, Node*));

	r = nod(OCALL, fn, N);
	r->list = args;
	if(fn->type->outtuple > 0)
		typecheck(&r, Erv | Efnstruct);
	else
		typecheck(&r, Etop);
	walkexpr(&r, init);
	r->type = t;
	return r;
}

Node*
mkcall(char *name, Type *t, NodeList **init, ...)
{
	Node *r;
	va_list va;

	va_start(va, init);
	r = vmkcall(syslook(name, 0), t, init, va);
	va_end(va);
	return r;
}

Node*
mkcall1(Node *fn, Type *t, NodeList **init, ...)
{
	Node *r;
	va_list va;

	va_start(va, init);
	r = vmkcall(fn, t, init, va);
	va_end(va);
	return r;
}

static Node*
conv(Node *n, Type *t)
{
	if(eqtype(n->type, t))
		return n;
	n = nod(OCONV, n, N);
	n->type = t;
	typecheck(&n, Erv);
	return n;
}

Node*
chanfn(char *name, int n, Type *t)
{
	Node *fn;
	int i;

	if(t->etype != TCHAN)
		fatal("chanfn %T", t);
	fn = syslook(name, 1);
	for(i=0; i<n; i++)
		argtype(fn, t->type);
	return fn;
}

static Node*
mapfn(char *name, Type *t)
{
	Node *fn;

	if(t->etype != TMAP)
		fatal("mapfn %T", t);
	fn = syslook(name, 1);
	argtype(fn, t->down);
	argtype(fn, t->type);
	argtype(fn, t->down);
	argtype(fn, t->type);
	return fn;
}
