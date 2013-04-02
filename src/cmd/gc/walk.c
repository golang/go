// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	<u.h>
#include	<libc.h>
#include	"go.h"

static	Node*	walkprint(Node*, NodeList**, int);
static	Node*	mapfn(char*, Type*);
static	Node*	mapfndel(char*, Type*);
static	Node*	ascompatee1(int, Node*, Node*, NodeList**);
static	NodeList*	ascompatee(int, NodeList*, NodeList*, NodeList**);
static	NodeList*	ascompatet(int, NodeList*, Type**, int, NodeList**);
static	NodeList*	ascompatte(int, Node*, int, Type**, NodeList*, int, NodeList**);
static	Node*	convas(Node*, NodeList**);
static	void	heapmoves(void);
static	NodeList*	paramstoheap(Type **argin, int out);
static	NodeList*	reorder1(NodeList*);
static	NodeList*	reorder3(NodeList*);
static	Node*	addstr(Node*, NodeList**);
static	Node*	appendslice(Node*, NodeList**);
static	Node*	append(Node*, NodeList**);
static	Node*	sliceany(Node*, NodeList**);
static	void	walkcompare(Node**, NodeList**);
static	void	walkrotate(Node**);
static	void	walkmul(Node**, NodeList**);
static	void	walkdiv(Node**, NodeList**);
static	int	bounded(Node*, int64);
static	Mpint	mpzero;

void
walk(Node *fn)
{
	char s[50];
	NodeList *l;
	int lno;

	curfn = fn;

	if(debug['W']) {
		snprint(s, sizeof(s), "\nbefore %S", curfn->nname->sym);
		dumplist(s, curfn->nbody);
	}

	lno = lineno;

	// Final typecheck for any unused variables.
	// It's hard to be on the heap when not-used, but best to be consistent about &~PHEAP here and below.
	for(l=fn->dcl; l; l=l->next)
		if(l->n->op == ONAME && (l->n->class&~PHEAP) == PAUTO)
			typecheck(&l->n, Erv | Easgn);

	// Propagate the used flag for typeswitch variables up to the NONAME in it's definition.
	for(l=fn->dcl; l; l=l->next)
		if(l->n->op == ONAME && (l->n->class&~PHEAP) == PAUTO && l->n->defn && l->n->defn->op == OTYPESW && l->n->used)
			l->n->defn->left->used++;
	
	for(l=fn->dcl; l; l=l->next) {
		if(l->n->op != ONAME || (l->n->class&~PHEAP) != PAUTO || l->n->sym->name[0] == '&' || l->n->used)
			continue;
		if(l->n->defn && l->n->defn->op == OTYPESW) {
			if(l->n->defn->left->used)
				continue;
			lineno = l->n->defn->left->lineno;
			yyerror("%S declared and not used", l->n->sym);
			l->n->defn->left->used = 1; // suppress repeats
		} else {
			lineno = l->n->lineno;
			yyerror("%S declared and not used", l->n->sym);
		}
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
		case PPARAMOUT:
		case PPARAMOUT|PHEAP:
			return l->n->addrtaken;
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
	int cl;
	Node *n, *f;

	n = *np;
	if(n == N)
		return;

	setlineno(n);

	walkstmtlist(n->ninit);

	switch(n->op) {
	default:
		if(n->op == ONAME)
			yyerror("%S is not a top level statement", n->sym);
		else
			yyerror("%O is not a top level statement", n->op);
		dump("nottop", n);
		break;

	case OAS:
	case OASOP:
	case OAS2:
	case OAS2DOTTYPE:
	case OAS2RECV:
	case OAS2FUNC:
	case OAS2MAPR:
	case OCLOSE:
	case OCOPY:
	case OCALLMETH:
	case OCALLINTER:
	case OCALL:
	case OCALLFUNC:
	case ODELETE:
	case OSEND:
	case ORECV:
	case OPRINT:
	case OPRINTN:
	case OPANIC:
	case OEMPTY:
	case ORECOVER:
		if(n->typecheck == 0)
			fatal("missing typecheck: %+N", n);
		init = n->ninit;
		n->ninit = nil;
		walkexpr(&n, &init);
		addinit(&n, init);
		break;

	case OBREAK:
	case ODCL:
	case OCONTINUE:
	case OFALL:
	case OGOTO:
	case OLABEL:
	case ODCLCONST:
	case ODCLTYPE:
	case OCHECKNOTNIL:
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
		if(n->ntest != N) {
			walkstmtlist(n->ntest->ninit);
			init = n->ntest->ninit;
			n->ntest->ninit = nil;
			walkexpr(&n->ntest, &init);
			addinit(&n->ntest, init);
		}
		walkstmt(&n->nincr);
		walkstmtlist(n->nbody);
		break;

	case OIF:
		walkexpr(&n->ntest, &n->ninit);
		walkstmtlist(n->nbody);
		walkstmtlist(n->nelse);
		break;

	case OPROC:
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
					fatal("expected return of call, have %N", f);
				n->list = concat(list1(f), ascompatet(n->op, rl, &f->type, 0, &n->ninit));
				break;
			}

			// move function calls out, to make reorder3's job easier.
			walkexprlistsafe(n->list, &n->ninit);
			ll = ascompatee(n->op, rl, n->list, &n->ninit);
			n->list = reorder3(ll);
			break;
		}
		ll = ascompatte(n->op, nil, 0, getoutarg(curfn->type), n->list, 1, &n->ninit);
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

	if(n->op == ONAME)
		fatal("walkstmt ended up with name: %+N", n);
	
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
	int64 v;
	int32 lno;
	Node *n, *fn, *n1, *n2;
	Sym *sym;
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

	if(n->ninit != nil) {
		walkstmtlist(n->ninit);
		*init = concat(*init, n->ninit);
		n->ninit = nil;
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

	if(n->typecheck != 1)
		fatal("missed typecheck: %+N\n", n);

	switch(n->op) {
	default:
		dump("walk", n);
		fatal("walkexpr: switch 1 unknown op %+hN", n);
		break;

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
	case ODOTMETH:
	case ODOTINTER:
		walkexpr(&n->left, init);
		goto ret;

	case OIND:
		if(n->left->type->type->width == 0) {
			n->left = cheapexpr(n->left, init);
			checknotnil(n->left, init);
		}
		walkexpr(&n->left, init);
		goto ret;

	case ODOT:
		usefield(n);
		walkexpr(&n->left, init);
		goto ret;

	case ODOTPTR:
		usefield(n);
		if(n->op == ODOTPTR && n->left->type->type->width == 0) {
			n->left = cheapexpr(n->left, init);
			checknotnil(n->left, init);
		}
		walkexpr(&n->left, init);
		goto ret;

	case OEFACE:
		walkexpr(&n->left, init);
		walkexpr(&n->right, init);
		goto ret;

	case OITAB:
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
		walkexpr(&n->left, init);
		walkexpr(&n->right, init);
	shiftwalked:
		t = n->left->type;
		n->bounded = bounded(n->right, 8*t->width);
		if(debug['m'] && n->etype && !isconst(n->right, CTINT))
			warn("shift bounds check elided");
		goto ret;

	case OAND:
	case OSUB:
	case OHMUL:
	case OLT:
	case OLE:
	case OGE:
	case OGT:
	case OADD:
	case OCOMPLEX:
	case OLROT:
		walkexpr(&n->left, init);
		walkexpr(&n->right, init);
		goto ret;

	case OOR:
	case OXOR:
		walkexpr(&n->left, init);
		walkexpr(&n->right, init);
		walkrotate(&n);
		goto ret;

	case OEQ:
	case ONE:
		walkexpr(&n->left, init);
		walkexpr(&n->right, init);
		walkcompare(&n, init);
		goto ret;

	case OANDAND:
	case OOROR:
		walkexpr(&n->left, init);
		// cannot put side effects from n->right on init,
		// because they cannot run before n->left is checked.
		// save elsewhere and store on the eventual n->right.
		ll = nil;
		walkexpr(&n->right, &ll);
		addinit(&n->right, ll);
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

	case OCLOSUREVAR:
	case OCFUNC:
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
		ll = ascompatte(n->op, n, n->isddd, getinarg(t), n->list, 0, init);
		n->list = reorder1(ll);
		goto ret;

	case OCALLFUNC:
		t = n->left->type;
		if(n->list && n->list->n->op == OAS)
			goto ret;

		walkexpr(&n->left, init);
		walkexprlist(n->list, init);

		ll = ascompatte(n->op, n, n->isddd, getinarg(t), n->list, 0, init);
		n->list = reorder1(ll);
		goto ret;

	case OCALLMETH:
		t = n->left->type;
		if(n->list && n->list->n->op == OAS)
			goto ret;
		walkexpr(&n->left, init);
		walkexprlist(n->list, init);
		ll = ascompatte(n->op, n, 0, getthis(t), list1(n->left->left), 0, init);
		lr = ascompatte(n->op, n, n->isddd, getinarg(t), n->list, 0, init);
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
		if(n->left != N && n->right != N) {
			r = convas(nod(OAS, n->left, n->right), init);
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
			var = temp(l->type);
			n->list->n = var;
			a = nod(OAS, l, var);
			typecheck(&a, Etop);
			lpost = list(lpost, a);
		}
		l = n->list->next->n;
		if(l->op == OINDEXMAP) {
			var = temp(l->type);
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
		*init = concat(*init, n->ninit);
		n->ninit = nil;
		r = n->rlist->n;
		walkexprlistsafe(n->list, init);
		walkexpr(&r->left, init);
		fn = chanfn("chanrecv2", 2, r->left->type);
		r = mkcall1(fn, getoutargx(fn->type), init, typename(r->left->type), r->left);
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
		t = r->left->type;
		p = nil;
		if(t->type->width <= 128) { // Check ../../pkg/runtime/hashmap.c:MAXVALUESIZE before changing.
			switch(simsimtype(t->down)) {
			case TINT32:
			case TUINT32:
				p = "mapaccess2_fast32";
				break;
			case TINT64:
			case TUINT64:
				p = "mapaccess2_fast64";
				break;
			case TSTRING:
				p = "mapaccess2_faststr";
				break;
			}
		}
		if(p != nil) {
			// from:
			//   a,b = m[i]
			// to:
			//   var,b = mapaccess2_fast*(t, m, i)
			//   a = *var
			a = n->list->n;
			var = temp(ptrto(t->type));
			var->typecheck = 1;

			fn = mapfn(p, t);
			r = mkcall1(fn, getoutargx(fn->type), init, typename(t), r->left, r->right);
			n->rlist = list1(r);
			n->op = OAS2FUNC;
			n->list->n = var;
			walkexpr(&n, init);
			*init = list(*init, n);

			n = nod(OAS, a, nod(OIND, var, N));
			typecheck(&n, Etop);
			walkexpr(&n, init);
			goto ret;
		}
		fn = mapfn("mapaccess2", t);
		r = mkcall1(fn, getoutargx(fn->type), init, typename(t), r->left, r->right);
		n->rlist = list1(r);
		n->op = OAS2FUNC;
		goto as2func;

	case ODELETE:
		*init = concat(*init, n->ninit);
		n->ninit = nil;
		l = n->list->n;
		r = n->list->next->n;
		t = l->type;
		n = mkcall1(mapfndel("mapdelete", t), t->down, init, typename(t), l, r);
		goto ret;

	case OAS2DOTTYPE:
		// a,b = i.(T)
		*init = concat(*init, n->ninit);
		n->ninit = nil;
		r = n->rlist->n;
		walkexprlistsafe(n->list, init);
		if(isblank(n->list->n) && !isinter(r->type)) {
			strcpy(buf, "assert");
			p = buf+strlen(buf);
			if(isnilinter(r->left->type))
				*p++ = 'E';
			else
				*p++ = 'I';
			*p++ = '2';
			*p++ = 'T';
			*p++ = 'O';
			*p++ = 'K';
			*p = '\0';
			
			fn = syslook(buf, 1);
			ll = list1(typename(r->type));
			ll = list(ll, r->left);
			argtype(fn, r->left->type);
			n1 = nod(OCALL, fn, N);
			n1->list = ll;
			n = nod(OAS, n->list->next->n, n1);
			typecheck(&n, Etop);
			walkexpr(&n, init);
			goto ret;
		}

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
		walkexpr(&n->left, init);

		// Optimize convT2E as a two-word copy when T is uintptr-shaped.
		if(!isinter(n->left->type) && isnilinter(n->type) &&
		   (n->left->type->width == widthptr) &&
		   isint[simsimtype(n->left->type)]) {
			l = nod(OEFACE, typename(n->left->type), n->left);
			l->type = n->type;
			l->typecheck = n->typecheck;
			n = l;
			goto ret;
		}

		// Build name of function: convI2E etc.
		// Not all names are possible
		// (e.g., we'll never generate convE2E or convE2I).
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
		if(!isinter(n->left->type) && !isnilinter(n->type)){
			sym = pkglookup(smprint("%-T.%-T", n->left->type, n->type), itabpkg);
			if(sym->def == N) {
				l = nod(ONAME, N, N);
				l->sym = sym;
				l->type = ptrto(types[TUINT8]);
				l->addable = 1;
				l->class = PEXTERN;
				l->xoffset = 0;
				sym->def = l;
				ggloblsym(sym, widthptr, 1, 0);
			}
			l = nod(OADDR, sym->def, N);
			l->addable = 1;
			ll = list(ll, l);

			if(n->left->type->width == widthptr &&
		   	   isint[simsimtype(n->left->type)]) {
				/* For pointer types, we can make a special form of optimization
				 *
				 * These statements are put onto the expression init list:
				 * 	Itab *tab = atomicloadtype(&cache);
				 * 	if(tab == nil)
				 * 		tab = typ2Itab(type, itype, &cache);
				 *
				 * The CONVIFACE expression is replaced with this:
				 * 	OEFACE{tab, ptr};
				 */
				l = temp(ptrto(types[TUINT8]));

				n1 = nod(OAS, l, sym->def);
				typecheck(&n1, Etop);
				*init = list(*init, n1);

				fn = syslook("typ2Itab", 1);
				n1 = nod(OCALL, fn, N);
				n1->list = ll;
				typecheck(&n1, Erv);
				walkexpr(&n1, init);

				n2 = nod(OIF, N, N);
				n2->ntest = nod(OEQ, l, nodnil());
				n2->nbody = list1(nod(OAS, l, n1));
				n2->likely = -1;
				typecheck(&n2, Etop);
				*init = list(*init, n2);

				l = nod(OEFACE, l, n->left);
				l->typecheck = n->typecheck; 
				l->type = n->type;
				n = l;
				goto ret;
			}
		}
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
			if(isfloat[n->left->type->etype]) {
				if(n->type->etype == TINT64) {
					n = mkcall("float64toint64", n->type, init, conv(n->left, types[TFLOAT64]));
					goto ret;
				}
				if(n->type->etype == TUINT64) {
					n = mkcall("float64touint64", n->type, init, conv(n->left, types[TFLOAT64]));
					goto ret;
				}
			}
			if(isfloat[n->type->etype]) {
				if(n->left->type->etype == TINT64) {
					n = mkcall("int64tofloat64", n->type, init, conv(n->left, types[TINT64]));
					goto ret;
				}
				if(n->left->type->etype == TUINT64) {
					n = mkcall("uint64tofloat64", n->type, init, conv(n->left, types[TUINT64]));
					goto ret;
				}
			}
		}
		walkexpr(&n->left, init);
		goto ret;

	case OASOP:
		if(n->etype == OANDNOT) {
			n->etype = OAND;
			n->right = nod(OCOM, n->right, N);
			typecheck(&n->right, Erv);
		}
		n->left = safeexpr(n->left, init);
		walkexpr(&n->left, init);
		l = n->left;
		walkexpr(&n->right, init);

		/*
		 * on 32-bit arch, rewrite 64-bit ops into l = l op r.
		 * on 386, rewrite float ops into l = l op r.
		 * everywhere, rewrite map ops into l = l op r.
		 * everywhere, rewrite string += into l = l op r.
		 * everywhere, rewrite integer/complex /= into l = l op r.
		 * TODO(rsc): Maybe this rewrite should be done always?
		 */
		et = n->left->type->etype;
		if((widthptr == 4 && (et == TUINT64 || et == TINT64)) ||
		   (thechar == '8' && isfloat[et]) ||
		   l->op == OINDEXMAP ||
		   et == TSTRING ||
		   (!isfloat[et] && n->etype == ODIV) ||
		   n->etype == OMOD) {
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
			goto ret;
		}
		if(n->etype == OLSH || n->etype == ORSH)
			goto shiftwalked;
		goto ret;

	case OANDNOT:
		walkexpr(&n->left, init);
		n->op = OAND;
		n->right = nod(OCOM, n->right, N);
		typecheck(&n->right, Erv);
		walkexpr(&n->right, init);
		goto ret;

	case OMUL:
		walkexpr(&n->left, init);
		walkexpr(&n->right, init);
		walkmul(&n, init);
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
			t = n->type;
			n = mkcall("complex128div", types[TCOMPLEX128], init,
				conv(n->left, types[TCOMPLEX128]),
				conv(n->right, types[TCOMPLEX128]));
			n = conv(n, t);
			goto ret;
		}
		// Nothing to do for float divisions.
		if(isfloat[et])
			goto ret;

		// Try rewriting as shifts or magic multiplies.
		walkdiv(&n, init);

		/*
		 * rewrite 64-bit div and mod into function calls
		 * on 32-bit architectures.
		 */
		switch(n->op) {
		case OMOD:
		case ODIV:
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
			break;
		default:
			break;
		}
		goto ret;

	case OINDEX:
		walkexpr(&n->left, init);
		// save the original node for bounds checking elision.
		// If it was a ODIV/OMOD walk might rewrite it.
		r = n->right;
		walkexpr(&n->right, init);

		// if range of type cannot exceed static array bound,
		// disable bounds check.
		if(n->bounded)
			goto ret;
		t = n->left->type;
		if(t != T && isptr[t->etype])
			t = t->type;
		if(isfixedarray(t)) {
			n->bounded = bounded(r, t->bound);
			if(debug['m'] && n->bounded && !isconst(n->right, CTINT))
				warn("index bounds check elided");
			if(smallintconst(n->right) && !n->bounded)
				yyerror("index out of bounds");
		} else if(isconst(n->left, CTSTR)) {
			n->bounded = bounded(r, n->left->val.u.sval->len);
			if(debug['m'] && n->bounded && !isconst(n->right, CTINT))
				warn("index bounds check elided");
			if(smallintconst(n->right)) {
				if(!n->bounded)
					yyerror("index out of bounds");
				else {
					// replace "abc"[2] with 'b'.
					// delayed until now because "abc"[2] is not
					// an ideal constant.
					v = mpgetfix(n->right->val.u.xval);
					nodconst(n, n->type, n->left->val.u.sval->s[v]);
					n->typecheck = 1;
				}
			}
		}

		if(isconst(n->right, CTINT))
		if(mpcmpfixfix(n->right->val.u.xval, &mpzero) < 0 ||
		   mpcmpfixfix(n->right->val.u.xval, maxintval[TINT]) > 0)
			yyerror("index out of bounds");
		goto ret;

	case OINDEXMAP:
		if(n->etype == 1)
			goto ret;

		t = n->left->type;
		p = nil;
		if(t->type->width <= 128) {  // Check ../../pkg/runtime/hashmap.c:MAXVALUESIZE before changing.
			switch(simsimtype(t->down)) {
			case TINT32:
			case TUINT32:
				p = "mapaccess1_fast32";
				break;
			case TINT64:
			case TUINT64:
				p = "mapaccess1_fast64";
				break;
			case TSTRING:
				p = "mapaccess1_faststr";
				break;
			}
		}
		if(p != nil) {
			// use fast version.  The fast versions return a pointer to the value - we need
			// to dereference it to get the result.
			n = mkcall1(mapfn(p, t), ptrto(t->type), init, typename(t), n->left, n->right);
			n = nod(OIND, n, N);
			n->type = t->type;
			n->typecheck = 1;
		} else {
			// no fast version for this key
			n = mkcall1(mapfn("mapaccess1", t), t->type, init, typename(t), n->left, n->right);
		}
		goto ret;

	case ORECV:
		walkexpr(&n->left, init);
		walkexpr(&n->right, init);
		n = mkcall1(chanfn("chanrecv1", 2, n->left->type), n->type, init, typename(n->left->type), n->left);
		goto ret;

	case OSLICE:
		if(n->right != N && n->right->left == N && n->right->right == N) { // noop
			walkexpr(&n->left, init);
			n = n->left;
			goto ret;
		}
		// fallthrough
	case OSLICEARR:
	case OSLICESTR:
		if(n->right == N) // already processed
			goto ret;

		walkexpr(&n->left, init);
		// cgen_slice can't handle string literals as source
		// TODO the OINDEX case is a bug elsewhere that needs to be traced.  it causes a crash on ([2][]int{ ... })[1][lo:hi]
		if((n->op == OSLICESTR && n->left->op == OLITERAL) || (n->left->op == OINDEX))
			n->left = copyexpr(n->left, n->left->type, init);
		else
			n->left = safeexpr(n->left, init);
		walkexpr(&n->right->left, init);
		n->right->left = safeexpr(n->right->left, init);
		walkexpr(&n->right->right, init);
		n->right->right = safeexpr(n->right->right, init);
		n = sliceany(n, init);  // chops n->right, sets n->list
		goto ret;

	case OADDR:
		walkexpr(&n->left, init);
		goto ret;

	case ONEW:
		if(n->esc == EscNone && n->type->type->width < (1<<16)) {
			r = temp(n->type->type);
			r = nod(OAS, r, N);  // zero temp
			typecheck(&r, Etop);
			*init = list(*init, r);
			r = nod(OADDR, r->left, N);
			typecheck(&r, Erv);
			n = r;
		} else {
			n = callnew(n->type->type);
		}
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
			r->type = n->type;
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
			r->type = n->type;
			n = r;
			goto ret;
		}

		if(n->etype == OEQ || n->etype == ONE) {
			// prepare for rewrite below
			n->left = cheapexpr(n->left, init);
			n->right = cheapexpr(n->right, init);

			r = mkcall("eqstring", types[TBOOL], init,
				conv(n->left, types[TSTRING]),
				conv(n->right, types[TSTRING]));

			// quick check of len before full compare for == or !=
			if(n->etype == OEQ) {
				// len(left) == len(right) && eqstring(left, right)
				r = nod(OANDAND, nod(OEQ, nod(OLEN, n->left, N), nod(OLEN, n->right, N)), r);
			} else {
				// len(left) != len(right) || !eqstring(left, right)
				r = nod(ONOT, r, N);
				r = nod(OOROR, nod(ONE, nod(OLEN, n->left, N), nod(OLEN, n->right, N)), r);
			}
			typecheck(&r, Erv);
			walkexpr(&r, nil);
		} else {
			// sys_cmpstring(s1, s2) :: 0
			r = mkcall("cmpstring", types[TINT], init,
				conv(n->left, types[TSTRING]),
				conv(n->right, types[TSTRING]));
			r = nod(n->etype, r, nodintconst(0));
		}

		typecheck(&r, Erv);
		if(n->type->etype != TBOOL) fatal("cmp %T", n->type);
		r->type = n->type;
		n = r;
		goto ret;

	case OADDSTR:
		n = addstr(n, init);
		goto ret;
	
	case OAPPEND:
		if(n->isddd) {
			if(istype(n->type->type, TUINT8) && istype(n->list->next->n->type, TSTRING))
				n = mkcall("appendstr", n->type, init, typename(n->type), n->list->n, n->list->next->n);
			else
				n = appendslice(n, init);
		}
		else
			n = append(n, init);
		goto ret;

	case OCOPY:
		if(n->right->type->etype == TSTRING)
			fn = syslook("slicestringcopy", 1);
		else
			fn = syslook("copy", 1);
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

	case OMAKECHAN:
		n = mkcall1(chanfn("makechan", 1, n->type), n->type, init,
			typename(n->type),
			conv(n->left, types[TINT64]));
		goto ret;

	case OMAKEMAP:
		t = n->type;

		fn = syslook("makemap", 1);
		argtype(fn, t->down);	// any-1
		argtype(fn, t->type);	// any-2

		n = mkcall1(fn, n->type, init,
			typename(n->type),
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
		// slicerunetostring([]rune) string;
		n = mkcall("slicerunetostring", n->type, init, n->left);
		goto ret;

	case OSTRARRAYBYTE:
		// stringtoslicebyte(string) []byte;
		n = mkcall("stringtoslicebyte", n->type, init, conv(n->left, types[TSTRING]));
		goto ret;

	case OSTRARRAYRUNE:
		// stringtoslicerune(string) []rune
		n = mkcall("stringtoslicerune", n->type, init, n->left);
		goto ret;

	case OCMPIFACE:
		// ifaceeq(i1 any-1, i2 any-2) (ret bool);
		if(!eqtype(n->left->type, n->right->type))
			fatal("ifaceeq %O %T %T", n->op, n->left->type, n->right->type);
		if(isnilinter(n->left->type))
			fn = syslook("efaceeq", 1);
		else
			fn = syslook("ifaceeq", 1);

		n->right = cheapexpr(n->right, init);
		n->left = cheapexpr(n->left, init);
		argtype(fn, n->right->type);
		argtype(fn, n->left->type);
		r = mkcall1(fn, n->type, init, n->left, n->right);
		if(n->etype == ONE)
			r = nod(ONOT, r, N);
		
		// check itable/type before full compare.
		if(n->etype == OEQ)
			r = nod(OANDAND, nod(OEQ, nod(OITAB, n->left, N), nod(OITAB, n->right, N)), r);
		else
			r = nod(OOROR, nod(ONE, nod(OITAB, n->left, N), nod(OITAB, n->right, N)), r);
		typecheck(&r, Erv);
		walkexpr(&r, init);
		r->type = n->type;
		n = r;
		goto ret;

	case OARRAYLIT:
	case OMAPLIT:
	case OSTRUCTLIT:
	case OPTRLIT:
		var = temp(n->type);
		anylit(0, n, var, init);
		n = var;
		goto ret;

	case OSEND:
		n = mkcall1(chanfn("chansend1", 2, n->left->type), T, init, typename(n->left->type), n->left, n->right);
		goto ret;

	case OCLOSURE:
		n = walkclosure(n, init);
		goto ret;
	
	case OCALLPART:
		n = walkpartialcall(n, init);
		goto ret;
	}
	fatal("missing switch %O", n->op);

ret:
	ullmancalc(n);

	if(debug['w'] && n != N)
		dump("walk", n);

	lineno = lno;
	*np = n;
}

static Node*
ascompatee1(int op, Node *l, Node *r, NodeList **init)
{
	Node *n;
	USED(op);
	
	// convas will turn map assigns into function calls,
	// making it impossible for reorder3 to work.
	n = nod(OAS, l, r);
	if(l->op == OINDEXMAP)
		return n;

	return convas(n, init);
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
	for(ll=nl, lr=nr; ll && lr; ll=ll->next, lr=lr->next) {
		// Do not generate 'x = x' during return. See issue 4014.
		if(op == ORETURN && ll->n == lr->n)
			continue;
		nn = list(nn, ascompatee1(op, ll->n, lr->n, init));
	}

	// cannot happen: caller checked that lists had same length
	if(ll || lr)
		yyerror("error in shape across %+H %O %+H / %d %d [%s]", nl, op, nr, count(nl), count(nr), curfn->nname->sym->name);
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
	if(l->ullman >= UINF || l->op == OINDEXMAP)
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

	USED(op);

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
			tmp = temp(r->type);
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
		yyerror("ascompatet: assignment count mismatch: %d = %d",
			count(nl), structcount(*nr));

	if(ucount)
		fatal("ascompatet: too many function calls evaluating parameters");
	return concat(nn, mm);
}

 /*
 * package all the arguments that match a ... T parameter into a []T.
 */
static NodeList*
mkdotargslice(NodeList *lr0, NodeList *nn, Type *l, int fp, NodeList **init, int esc)
{
	Node *a, *n;
	Type *tslice;

	tslice = typ(TARRAY);
	tslice->type = l->type->type;
	tslice->bound = -1;

	if(count(lr0) == 0) {
		n = nodnil();
		n->type = tslice;
	} else {
		n = nod(OCOMPLIT, N, typenod(tslice));
		n->list = lr0;
		n->esc = esc;
		typecheck(&n, Erv);
		if(n->type == T)
			fatal("mkdotargslice: typecheck failed");
		walkexpr(&n, init);
	}

	a = nod(OAS, nodarg(l, fp), n);
	nn = list(nn, convas(a, init));
	return nn;
}

/*
 * helpers for shape errors
 */
static char*
dumptypes(Type **nl, char *what)
{
	int first;
	Type *l;
	Iter savel;
	Fmt fmt;

	fmtstrinit(&fmt);
	fmtprint(&fmt, "\t");
	first = 1;
	for(l = structfirst(&savel, nl); l != T; l = structnext(&savel)) {
		if(first)
			first = 0;
		else
			fmtprint(&fmt, ", ");
		fmtprint(&fmt, "%T", l);
	}
	if(first)
		fmtprint(&fmt, "[no arguments %s]", what);
	return fmtstrflush(&fmt);
}

static char*
dumpnodetypes(NodeList *l, char *what)
{
	int first;
	Node *r;
	Fmt fmt;

	fmtstrinit(&fmt);
	fmtprint(&fmt, "\t");
	first = 1;
	for(; l; l=l->next) {
		r = l->n;
		if(first)
			first = 0;
		else
			fmtprint(&fmt, ", ");
		fmtprint(&fmt, "%T", r->type);
	}
	if(first)
		fmtprint(&fmt, "[no arguments %s]", what);
	return fmtstrflush(&fmt);
}

/*
 * check assign expression list to
 * a type list. called in
 *	return expr-list
 *	func(expr-list)
 */
static NodeList*
ascompatte(int op, Node *call, int isddd, Type **nl, NodeList *lr, int fp, NodeList **init)
{
	int esc;
	Type *l, *ll;
	Node *r, *a;
	NodeList *nn, *lr0, *alist;
	Iter savel;
	char *l1, *l2;

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
			a = temp(l->type);
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
		if(r != N && lr->next == nil && isddd && eqtype(l->type, r->type)) {
			a = nod(OAS, nodarg(l, fp), r);
			a = convas(a, init);
			nn = list(nn, a);
			goto ret;
		}

		// normal case -- make a slice of all
		// remaining arguments and pass it to
		// the ddd parameter.
		esc = EscUnknown;
		if(call->right)
			esc = call->right->esc;
		nn = mkdotargslice(lr, nn, l, fp, init, esc);
		goto ret;
	}

	if(l == T || r == N) {
		if(l != T || r != N) {
			l1 = dumptypes(nl, "expected");
			l2 = dumpnodetypes(lr0, "given");
			if(l != T)
				yyerror("not enough arguments to %O\n%s\n%s", op, l1, l2);
			else
				yyerror("too many arguments to %O\n%s\n%s", op, l1, l2);
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
			case CTRUNE:
				defaultlit(&n, runetype);
				break;
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
		} else if(isptr[et] || et == TCHAN || et == TMAP || et == TFUNC || et == TUNSAFEPTR) {
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
		on = syslook("goprintf", 1);
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
	fn = syslook("new", 1);
	argtype(fn, t);
	return mkcall1(fn, ptrto(t), nil, typename(t));
}

static Node*
convas(Node *n, NodeList **init)
{
	Type *lt, *rt;

	if(n->op != OAS)
		fatal("convas: not OAS %O", n->op);

	n->typecheck = 1;

	if(n->left == N || n->right == N)
		goto out;

	lt = n->left->type;
	rt = n->right->type;
	if(lt == T || rt == T)
		goto out;

	if(isblank(n->left)) {
		defaultlit(&n->right, T);
		goto out;
	}

	if(n->left->op == OINDEXMAP) {
		n = mkcall1(mapfn("mapassign1", n->left->left->type), T, init,
			typename(n->left->left->type),
			n->left->left, n->left->right, n->right);
		goto out;
	}

	if(eqtype(lt, rt))
		goto out;

	n->right = assignconv(n->right, lt, "assignment");
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
static NodeList*
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
		a = temp(n->right->type);
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

static void reorder3save(Node**, NodeList*, NodeList*, NodeList**);
static int aliased(Node*, NodeList*, NodeList*);

/*
 * from ascompat[ee]
 *	a,b = c,d
 * simultaneous assignment. there cannot
 * be later use of an earlier lvalue.
 *
 * function calls have been removed.
 */
static NodeList*
reorder3(NodeList *all)
{
	NodeList *list, *early, *mapinit;
	Node *l;

	// If a needed expression may be affected by an
	// earlier assignment, make an early copy of that
	// expression and use the copy instead.
	early = nil;
	mapinit = nil;
	for(list=all; list; list=list->next) {
		l = list->n->left;

		// Save subexpressions needed on left side.
		// Drill through non-dereferences.
		for(;;) {
			if(l->op == ODOT || l->op == OPAREN) {
				l = l->left;
				continue;
			}
			if(l->op == OINDEX && isfixedarray(l->left->type)) {
				reorder3save(&l->right, all, list, &early);
				l = l->left;
				continue;
			}
			break;
		}
		switch(l->op) {
		default:
			fatal("reorder3 unexpected lvalue %#O", l->op);
		case ONAME:
			break;
		case OINDEX:
		case OINDEXMAP:
			reorder3save(&l->left, all, list, &early);
			reorder3save(&l->right, all, list, &early);
			if(l->op == OINDEXMAP)
				list->n = convas(list->n, &mapinit);
			break;
		case OIND:
		case ODOTPTR:
			reorder3save(&l->left, all, list, &early);
		}

		// Save expression on right side.
		reorder3save(&list->n->right, all, list, &early);
	}

	early = concat(mapinit, early);
	return concat(early, all);
}

static int vmatch2(Node*, Node*);
static int varexpr(Node*);

/*
 * if the evaluation of *np would be affected by the 
 * assignments in all up to but not including stop,
 * copy into a temporary during *early and
 * replace *np with that temp.
 */
static void
reorder3save(Node **np, NodeList *all, NodeList *stop, NodeList **early)
{
	Node *n, *q;

	n = *np;
	if(!aliased(n, all, stop))
		return;
	
	q = temp(n->type);
	q = nod(OAS, q, n);
	typecheck(&q, Etop);
	*early = list(*early, q);
	*np = q->left;
}

/*
 * what's the outer value that a write to n affects?
 * outer value means containing struct or array.
 */
static Node*
outervalue(Node *n)
{	
	for(;;) {
		if(n->op == ODOT || n->op == OPAREN) {
			n = n->left;
			continue;
		}
		if(n->op == OINDEX && isfixedarray(n->left->type)) {
			n = n->left;
			continue;
		}
		break;
	}
	return n;
}

/*
 * Is it possible that the computation of n might be
 * affected by writes in as up to but not including stop?
 */
static int
aliased(Node *n, NodeList *all, NodeList *stop)
{
	int memwrite, varwrite;
	Node *a;
	NodeList *l;

	if(n == N)
		return 0;

	// Look for obvious aliasing: a variable being assigned
	// during the all list and appearing in n.
	// Also record whether there are any writes to main memory.
	// Also record whether there are any writes to variables
	// whose addresses have been taken.
	memwrite = 0;
	varwrite = 0;
	for(l=all; l!=stop; l=l->next) {
		a = outervalue(l->n->left);
		if(a->op != ONAME) {
			memwrite = 1;
			continue;
		}
		switch(n->class) {
		default:
			varwrite = 1;
			continue;
		case PAUTO:
		case PPARAM:
		case PPARAMOUT:
			if(n->addrtaken) {
				varwrite = 1;
				continue;
			}
			if(vmatch2(a, n)) {
				// Direct hit.
				return 1;
			}
		}
	}

	// The variables being written do not appear in n.
	// However, n might refer to computed addresses
	// that are being written.
	
	// If no computed addresses are affected by the writes, no aliasing.
	if(!memwrite && !varwrite)
		return 0;

	// If n does not refer to computed addresses
	// (that is, if n only refers to variables whose addresses
	// have not been taken), no aliasing.
	if(varexpr(n))
		return 0;

	// Otherwise, both the writes and n refer to computed memory addresses.
	// Assume that they might conflict.
	return 1;
}

/*
 * does the evaluation of n only refer to variables
 * whose addresses have not been taken?
 * (and no other memory)
 */
static int
varexpr(Node *n)
{
	if(n == N)
		return 1;

	switch(n->op) {
	case OLITERAL:	
		return 1;
	case ONAME:
		switch(n->class) {
		case PAUTO:
		case PPARAM:
		case PPARAMOUT:
			if(!n->addrtaken)
				return 1;
		}
		return 0;

	case OADD:
	case OSUB:
	case OOR:
	case OXOR:
	case OMUL:
	case ODIV:
	case OMOD:
	case OLSH:
	case ORSH:
	case OAND:
	case OANDNOT:
	case OPLUS:
	case OMINUS:
	case OCOM:
	case OPAREN:
	case OANDAND:
	case OOROR:
	case ODOT:  // but not ODOTPTR
	case OCONV:
	case OCONVNOP:
	case OCONVIFACE:
	case ODOTTYPE:
		return varexpr(n->left) && varexpr(n->right);
	}

	// Be conservative.
	return 0;
}

/*
 * is the name l mentioned in r?
 */
static int
vmatch2(Node *l, Node *r)
{
	NodeList *ll;

	if(r == N)
		return 0;
	switch(r->op) {
	case ONAME:
		// match each right given left
		return l == r;
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

/*
 * is any name mentioned in l also mentioned in r?
 * called by sinit.c
 */
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
		if(v && v->sym && v->sym->name[0] == '~')
			v = N;
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
	int32 lno;

	lno = lineno;
	lineno = curfn->lineno;
	nn = paramstoheap(getthis(curfn->type), 0);
	nn = concat(nn, paramstoheap(getinarg(curfn->type), 0));
	nn = concat(nn, paramstoheap(getoutarg(curfn->type), 1));
	curfn->enter = concat(curfn->enter, nn);
	lineno = curfn->endlineno;
	curfn->exit = returnsfromheap(getoutarg(curfn->type));
	lineno = lno;
}

static Node*
vmkcall(Node *fn, Type *t, NodeList **init, va_list va)
{
	int i, n;
	Node *r;
	NodeList *args;

	if(fn->type == T || fn->type->etype != TFUNC)
		fatal("mkcall %N %T", fn, fn->type);

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

Node*
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

static Node*
mapfndel(char *name, Type *t)
{
	Node *fn;

	if(t->etype != TMAP)
		fatal("mapfn %T", t);
	fn = syslook(name, 1);
	argtype(fn, t->down);
	argtype(fn, t->type);
	argtype(fn, t->down);
	return fn;
}

static Node*
addstr(Node *n, NodeList **init)
{
	Node *r, *cat, *typstr;
	NodeList *in, *args;
	int i, count;

	count = 0;
	for(r=n; r->op == OADDSTR; r=r->left)
		count++;	// r->right
	count++;	// r

	// prepare call of runtime.catstring of type int, string, string, string
	// with as many strings as we have.
	cat = syslook("concatstring", 1);
	cat->type = T;
	cat->ntype = nod(OTFUNC, N, N);
	in = list1(nod(ODCLFIELD, N, typenod(types[TINT])));	// count
	typstr = typenod(types[TSTRING]);
	for(i=0; i<count; i++)
		in = list(in, nod(ODCLFIELD, N, typstr));
	cat->ntype->list = in;
	cat->ntype->rlist = list1(nod(ODCLFIELD, N, typstr));

	args = nil;
	for(r=n; r->op == OADDSTR; r=r->left)
		args = concat(list1(conv(r->right, types[TSTRING])), args);
	args = concat(list1(conv(r, types[TSTRING])), args);
	args = concat(list1(nodintconst(count)), args);

	r = nod(OCALL, cat, N);
	r->list = args;
	typecheck(&r, Erv);
	walkexpr(&r, init);
	r->type = n->type;

	return r;
}

static Node*
appendslice(Node *n, NodeList **init)
{
	Node *f;

	f = syslook("appendslice", 1);
	argtype(f, n->type);
	argtype(f, n->type->type);
	argtype(f, n->type);
	return mkcall1(f, n->type, init, typename(n->type), n->list->n, n->list->next->n);
}

// expand append(src, a [, b]* ) to
//
//   init {
//     s := src
//     const argc = len(args) - 1
//     if cap(s) - len(s) < argc {
//	    s = growslice(s, argc)
//     }
//     n := len(s)
//     s = s[:n+argc]
//     s[n] = a
//     s[n+1] = b
//     ...
//   }
//   s
static Node*
append(Node *n, NodeList **init)
{
	NodeList *l, *a;
	Node *nsrc, *ns, *nn, *na, *nx, *fn;
	int argc;

	walkexprlistsafe(n->list, init);

	// walkexprlistsafe will leave OINDEX (s[n]) alone if both s
	// and n are name or literal, but those may index the slice we're
	// modifying here.  Fix explicitly.
	for(l=n->list; l; l=l->next)
		l->n = cheapexpr(l->n, init);

	nsrc = n->list->n;
	argc = count(n->list) - 1;
	if (argc < 1) {
		return nsrc;
	}

	l = nil;

	ns = temp(nsrc->type);
	l = list(l, nod(OAS, ns, nsrc));  // s = src

	na = nodintconst(argc);		// const argc
	nx = nod(OIF, N, N);		// if cap(s) - len(s) < argc
	nx->ntest = nod(OLT, nod(OSUB, nod(OCAP, ns, N), nod(OLEN, ns, N)), na);

	fn = syslook("growslice", 1);	//   growslice(<type>, old []T, n int64) (ret []T)
	argtype(fn, ns->type->type);	// 1 old []any
	argtype(fn, ns->type->type);	// 2 ret []any

	nx->nbody = list1(nod(OAS, ns, mkcall1(fn,  ns->type, &nx->ninit,
					       typename(ns->type),
					       ns,
					       conv(na, types[TINT64]))));
	l = list(l, nx);

	nn = temp(types[TINT]);
	l = list(l, nod(OAS, nn, nod(OLEN, ns, N)));	 // n = len(s)

	nx = nod(OSLICE, ns, nod(OKEY, N, nod(OADD, nn, na)));	 // ...s[:n+argc]
	nx->bounded = 1;
	l = list(l, nod(OAS, ns, nx));			// s = s[:n+argc]

	for (a = n->list->next;	 a != nil; a = a->next) {
		nx = nod(OINDEX, ns, nn);		// s[n] ...
		nx->bounded = 1;
		l = list(l, nod(OAS, nx, a->n));	// s[n] = arg
		if (a->next != nil)
			l = list(l, nod(OAS, nn, nod(OADD, nn, nodintconst(1))));  // n = n + 1
	}

	typechecklist(l, Etop);
	walkstmtlist(l);
	*init = concat(*init, l);
	return ns;
}


// Generate frontend part for OSLICE[ARR|STR]
// 
static	Node*
sliceany(Node* n, NodeList **init)
{
	int bounded;
	Node *src, *lb, *hb, *bound, *chk, *chk1, *chk2;
	int64 lbv, hbv, bv, w;
	Type *bt;

//	print("before sliceany: %+N\n", n);

	src = n->left;
	lb = n->right->left;
	hb = n->right->right;

	bounded = n->etype;
	
	if(n->op == OSLICESTR)
		bound = nod(OLEN, src, N);
	else
		bound = nod(OCAP, src, N);

	typecheck(&bound, Erv);
	walkexpr(&bound, init);  // if src is an array, bound will be a const now.

	// static checks if possible
	bv = 1LL<<50;
	if(isconst(bound, CTINT)) {
		if(!smallintconst(bound))
			yyerror("array len too large");
		else
			bv = mpgetfix(bound->val.u.xval);
	}

	if(isconst(hb, CTINT)) {
		hbv = mpgetfix(hb->val.u.xval);
		if(hbv < 0 || hbv > bv) {
			yyerror("slice index out of bounds");
			hbv = -1;
		}
	}
	if(isconst(lb, CTINT)) {
		lbv = mpgetfix(lb->val.u.xval);
		if(lbv < 0 || lbv > bv) {
			yyerror("slice index out of bounds");
			lbv = -1;
		}
		if(lbv == 0)
			lb = N;
	}

	// dynamic checks convert all bounds to unsigned to save us the bound < 0 comparison
	// generate
	//     if hb > bound || lb > hb { panicslice() }
	chk = N;
	chk1 = N;
	chk2 = N;

	bt = types[simtype[TUINT]];
	if(hb != N && hb->type->width > 4)
		bt = types[TUINT64];
	if(lb != N && lb->type->width > 4)
		bt = types[TUINT64];

	bound = cheapexpr(conv(bound, bt), init);

	if(hb != N) {
		hb = cheapexpr(conv(hb, bt), init);
		if(!bounded)
			chk1 = nod(OLT, bound, hb);  
	} else if(n->op == OSLICEARR) {
		hb = bound;
	} else {
		hb = nod(OLEN, src, N);
		typecheck(&hb, Erv);
		walkexpr(&hb, init);
		hb = cheapexpr(conv(hb, bt), init);
	}

	if(lb != N) {
		lb = cheapexpr(conv(lb, bt), init);
		if(!bounded)
			chk2 = nod(OLT, hb, lb);  
	}

	if(chk1 != N || chk2 != N) {
		chk = nod(OIF, N, N);
		chk->nbody = list1(mkcall("panicslice", T, init));
		if(chk1 != N)
			chk->ntest = chk1;
		if(chk2 != N) {
			if(chk->ntest == N)
				chk->ntest = chk2;
			else
				chk->ntest = nod(OOROR, chk->ntest, chk2);
		}
		typecheck(&chk, Etop);
		walkstmt(&chk);
		*init = concat(*init, chk->ninit);
		chk->ninit = nil;
		*init = list(*init, chk);
	}
	
	// prepare new cap, len and offs for backend cgen_slice
	// cap = bound [ - lo ]
	n->right = N;
	n->list = nil;
	if(lb == N)
		bound = conv(bound, types[simtype[TUINT]]);
	else
		bound = nod(OSUB, conv(bound, types[simtype[TUINT]]), conv(lb, types[simtype[TUINT]]));
	typecheck(&bound, Erv);
	walkexpr(&bound, init);
	n->list = list(n->list, bound);

	// len = hi [ - lo]
	if(lb == N)
		hb = conv(hb, types[simtype[TUINT]]);
	else
		hb = nod(OSUB, conv(hb, types[simtype[TUINT]]), conv(lb, types[simtype[TUINT]]));
	typecheck(&hb, Erv);
	walkexpr(&hb, init);
	n->list = list(n->list, hb);

	// offs = [width *] lo, but omit if zero
	if(lb != N) {
		if(n->op == OSLICESTR)
			w = 1;
		else
			w = n->type->type->width;
		lb = conv(lb, types[TUINTPTR]);
		if(w > 1)
			lb = nod(OMUL, nodintconst(w), lb);
		typecheck(&lb, Erv);
		walkexpr(&lb, init);
		n->list = list(n->list, lb);
	}

//	print("after sliceany: %+N\n", n);

	return n;
}

static Node*
eqfor(Type *t)
{
	int a;
	Node *n;
	Node *ntype;
	Sym *sym;

	// Should only arrive here with large memory or
	// a struct/array containing a non-memory field/element.
	// Small memory is handled inline, and single non-memory
	// is handled during type check (OCMPSTR etc).
	a = algtype1(t, nil);
	if(a != AMEM && a != -1)
		fatal("eqfor %T", t);

	if(a == AMEM) {
		n = syslook("memequal", 1);
		argtype(n, t);
		argtype(n, t);
		return n;
	}

	sym = typesymprefix(".eq", t);
	n = newname(sym);
	n->class = PFUNC;
	ntype = nod(OTFUNC, N, N);
	ntype->list = list(ntype->list, nod(ODCLFIELD, N, typenod(ptrto(types[TBOOL]))));
	ntype->list = list(ntype->list, nod(ODCLFIELD, N, typenod(types[TUINTPTR])));
	ntype->list = list(ntype->list, nod(ODCLFIELD, N, typenod(ptrto(t))));
	ntype->list = list(ntype->list, nod(ODCLFIELD, N, typenod(ptrto(t))));
	typecheck(&ntype, Etype);
	n->type = ntype->type;
	return n;
}

static int
countfield(Type *t)
{
	Type *t1;
	int n;
	
	n = 0;
	for(t1=t->type; t1!=T; t1=t1->down)
		n++;
	return n;
}

static void
walkcompare(Node **np, NodeList **init)
{
	Node *n, *l, *r, *fn, *call, *a, *li, *ri, *expr;
	int andor, i;
	Type *t, *t1;
	static Node *tempbool;
	
	n = *np;
	
	// Must be comparison of array or struct.
	// Otherwise back end handles it.
	t = n->left->type;
	switch(t->etype) {
	default:
		return;
	case TARRAY:
		if(isslice(t))
			return;
		break;
	case TSTRUCT:
		break;
	}
	
	if(!islvalue(n->left) || !islvalue(n->right))
		goto hard;

	l = temp(ptrto(t));
	a = nod(OAS, l, nod(OADDR, n->left, N));
	a->right->etype = 1;  // addr does not escape
	typecheck(&a, Etop);
	*init = list(*init, a);

	r = temp(ptrto(t));
	a = nod(OAS, r, nod(OADDR, n->right, N));
	a->right->etype = 1;  // addr does not escape
	typecheck(&a, Etop);
	*init = list(*init, a);

	expr = N;
	andor = OANDAND;
	if(n->op == ONE)
		andor = OOROR;

	if(t->etype == TARRAY &&
		t->bound <= 4 &&
		issimple[t->type->etype]) {
		// Four or fewer elements of a basic type.
		// Unroll comparisons.
		for(i=0; i<t->bound; i++) {
			li = nod(OINDEX, l, nodintconst(i));
			ri = nod(OINDEX, r, nodintconst(i));
			a = nod(n->op, li, ri);
			if(expr == N)
				expr = a;
			else
				expr = nod(andor, expr, a);
		}
		if(expr == N)
			expr = nodbool(n->op == OEQ);
		typecheck(&expr, Erv);
		walkexpr(&expr, init);
		expr->type = n->type;
		*np = expr;
		return;
	}
	
	if(t->etype == TSTRUCT && countfield(t) <= 4) {
		// Struct of four or fewer fields.
		// Inline comparisons.
		for(t1=t->type; t1; t1=t1->down) {
			if(isblanksym(t1->sym))
				continue;
			li = nod(OXDOT, l, newname(t1->sym));
			ri = nod(OXDOT, r, newname(t1->sym));
			a = nod(n->op, li, ri);
			if(expr == N)
				expr = a;
			else
				expr = nod(andor, expr, a);
		}
		if(expr == N)
			expr = nodbool(n->op == OEQ);
		typecheck(&expr, Erv);
		walkexpr(&expr, init);
		expr->type = n->type;
		*np = expr;
		return;
	}
	
	// Chose not to inline, but still have addresses.
	// Call equality function directly.
	// The equality function requires a bool pointer for
	// storing its address, because it has to be callable
	// from C, and C can't access an ordinary Go return value.
	// To avoid creating many temporaries, cache one per function.
	if(tempbool == N || tempbool->curfn != curfn)
		tempbool = temp(types[TBOOL]);
	
	call = nod(OCALL, eqfor(t), N);
	a = nod(OADDR, tempbool, N);
	a->etype = 1;  // does not escape
	call->list = list(call->list, a);
	call->list = list(call->list, nodintconst(t->width));
	call->list = list(call->list, l);
	call->list = list(call->list, r);
	typecheck(&call, Etop);
	walkstmt(&call);
	*init = list(*init, call);

	// tempbool cannot be used directly as multiple comparison
	// expressions may exist in the same statement. Create another
	// temporary to hold the value (its address is not taken so it can
	// be optimized away).
	r = temp(types[TBOOL]);
	a = nod(OAS, r, tempbool);
	typecheck(&a, Etop);
	walkstmt(&a);
	*init = list(*init, a);

	if(n->op != OEQ)
		r = nod(ONOT, r, N);
	typecheck(&r, Erv);
	walkexpr(&r, init);
	*np = r;
	return;

hard:
	// Cannot take address of one or both of the operands.
	// Instead, pass directly to runtime helper function.
	// Easier on the stack than passing the address
	// of temporary variables, because we are better at reusing
	// the argument space than temporary variable space.
	fn = syslook("equal", 1);
	l = n->left;
	r = n->right;
	argtype(fn, n->left->type);
	argtype(fn, n->left->type);
	r = mkcall1(fn, n->type, init, typename(n->left->type), l, r);
	if(n->op == ONE) {
		r = nod(ONOT, r, N);
		typecheck(&r, Erv);
	}
	*np = r;
	return;
}

static int
samecheap(Node *a, Node *b)
{
	Node *ar, *br;
	while(a != N && b != N && a->op == b->op) {
		switch(a->op) {
		default:
			return 0;
		case ONAME:
			return a == b;
		case ODOT:
		case ODOTPTR:
			ar = a->right;
			br = b->right;
			if(ar->op != ONAME || br->op != ONAME || ar->sym != br->sym)
				return 0;
			break;
		case OINDEX:
			ar = a->right;
			br = b->right;
			if(!isconst(ar, CTINT) || !isconst(br, CTINT) || mpcmpfixfix(ar->val.u.xval, br->val.u.xval) != 0)
				return 0;
			break;
		}
		a = a->left;
		b = b->left;
	}
	return 0;
}

static void
walkrotate(Node **np)
{
	int w, sl, sr, s;
	Node *l, *r;
	Node *n;
	
	n = *np;

	// Want << | >> or >> | << or << ^ >> or >> ^ << on unsigned value.
	l = n->left;
	r = n->right;
	if((n->op != OOR && n->op != OXOR) ||
	   (l->op != OLSH && l->op != ORSH) ||
	   (r->op != OLSH && r->op != ORSH) ||
	   n->type == T || issigned[n->type->etype] ||
	   l->op == r->op) {
		return;
	}

	// Want same, side effect-free expression on lhs of both shifts.
	if(!samecheap(l->left, r->left))
		return;
	
	// Constants adding to width?
	w = l->type->width * 8;
	if(smallintconst(l->right) && smallintconst(r->right)) {
		if((sl=mpgetfix(l->right->val.u.xval)) >= 0 && (sr=mpgetfix(r->right->val.u.xval)) >= 0 && sl+sr == w)
			goto yes;
		return;
	}
	
	// TODO: Could allow s and 32-s if s is bounded (maybe s&31 and 32-s&31).
	return;
	
yes:
	// Rewrite left shift half to left rotate.
	if(l->op == OLSH)
		n = l;
	else
		n = r;
	n->op = OLROT;
	
	// Remove rotate 0 and rotate w.
	s = mpgetfix(n->right->val.u.xval);
	if(s == 0 || s == w)
		n = n->left;

	*np = n;
	return;
}

/*
 * walkmul rewrites integer multiplication by powers of two as shifts.
 */
static void
walkmul(Node **np, NodeList **init)
{
	Node *n, *nl, *nr;
	int pow, neg, w;
	
	n = *np;
	if(!isint[n->type->etype])
		return;

	if(n->right->op == OLITERAL) {
		nl = n->left;
		nr = n->right;
	} else if(n->left->op == OLITERAL) {
		nl = n->right;
		nr = n->left;
	} else
		return;

	neg = 0;

	// x*0 is 0 (and side effects of x).
	if(mpgetfix(nr->val.u.xval) == 0) {
		cheapexpr(nl, init);
		nodconst(n, n->type, 0);
		goto ret;
	}

	// nr is a constant.
	pow = powtwo(nr);
	if(pow < 0)
		return;
	if(pow >= 1000) {
		// negative power of 2, like -16
		neg = 1;
		pow -= 1000;
	}

	w = nl->type->width*8;
	if(pow+1 >= w)// too big, shouldn't happen
		return;

	nl = cheapexpr(nl, init);

	if(pow == 0) {
		// x*1 is x
		n = nl;
		goto ret;
	}
	
	n = nod(OLSH, nl, nodintconst(pow));

ret:
	if(neg)
		n = nod(OMINUS, n, N);

	typecheck(&n, Erv);
	walkexpr(&n, init);
	*np = n;
}

/*
 * walkdiv rewrites division by a constant as less expensive
 * operations.
 */
static void
walkdiv(Node **np, NodeList **init)
{
	Node *n, *nl, *nr, *nc;
	Node *n1, *n2, *n3, *n4;
	int pow; // if >= 0, nr is 1<<pow
	int s; // 1 if nr is negative.
	int w;
	Type *twide;
	Magic m;

	n = *np;
	if(n->right->op != OLITERAL)
		return;
	// nr is a constant.
	nl = cheapexpr(n->left, init);
	nr = n->right;

	// special cases of mod/div
	// by a constant
	w = nl->type->width*8;
	s = 0;
	pow = powtwo(nr);
	if(pow >= 1000) {
		// negative power of 2
		s = 1;
		pow -= 1000;
	}

	if(pow+1 >= w) {
		// divisor too large.
		return;
	}
	if(pow < 0) {
		goto divbymul;
	}

	switch(pow) {
	case 0:
		if(n->op == OMOD) {
			// nl % 1 is zero.
			nodconst(n, n->type, 0);
		} else if(s) {
			// divide by -1
			n->op = OMINUS;
			n->right = N;
		} else {
			// divide by 1
			n = nl;
		}
		break;
	default:
		if(issigned[n->type->etype]) {
			if(n->op == OMOD) {
				// signed modulo 2^pow is like ANDing
				// with the last pow bits, but if nl < 0,
				// nl & (2^pow-1) is (nl+1)%2^pow - 1.
				nc = nod(OXXX, N, N);
				nodconst(nc, types[simtype[TUINT]], w-1);
				n1 = nod(ORSH, nl, nc); // n1 = -1 iff nl < 0.
				if(pow == 1) {
					typecheck(&n1, Erv);
					n1 = cheapexpr(n1, init);
					// n = (nl+ε)&1 -ε where ε=1 iff nl<0.
					n2 = nod(OSUB, nl, n1);
					nc = nod(OXXX, N, N);
					nodconst(nc, nl->type, 1);
					n3 = nod(OAND, n2, nc);
					n = nod(OADD, n3, n1);
				} else {
					// n = (nl+ε)&(nr-1) - ε where ε=2^pow-1 iff nl<0.
					nc = nod(OXXX, N, N);
					nodconst(nc, nl->type, (1LL<<pow)-1);
					n2 = nod(OAND, n1, nc); // n2 = 2^pow-1 iff nl<0.
					typecheck(&n2, Erv);
					n2 = cheapexpr(n2, init);

					n3 = nod(OADD, nl, n2);
					n4 = nod(OAND, n3, nc);
					n = nod(OSUB, n4, n2);
				}
				break;
			} else {
				// arithmetic right shift does not give the correct rounding.
				// if nl >= 0, nl >> n == nl / nr
				// if nl < 0, we want to add 2^n-1 first.
				nc = nod(OXXX, N, N);
				nodconst(nc, types[simtype[TUINT]], w-1);
				n1 = nod(ORSH, nl, nc); // n1 = -1 iff nl < 0.
				if(pow == 1) {
					// nl+1 is nl-(-1)
					n->left = nod(OSUB, nl, n1);
				} else {
					// Do a logical right right on -1 to keep pow bits.
					nc = nod(OXXX, N, N);
					nodconst(nc, types[simtype[TUINT]], w-pow);
					n2 = nod(ORSH, conv(n1, tounsigned(nl->type)), nc);
					n->left = nod(OADD, nl, conv(n2, nl->type));
				}
				// n = (nl + 2^pow-1) >> pow
				n->op = ORSH;
				nc = nod(OXXX, N, N);
				nodconst(nc, types[simtype[TUINT]], pow);
				n->right = nc;
				n->typecheck = 0;
			}
			if(s)
				n = nod(OMINUS, n, N);
			break;
		}
		nc = nod(OXXX, N, N);
		if(n->op == OMOD) {
			// n = nl & (nr-1)
			n->op = OAND;
			nodconst(nc, nl->type, mpgetfix(nr->val.u.xval)-1);
		} else {
			// n = nl >> pow
			n->op = ORSH;
			nodconst(nc, types[simtype[TUINT]], pow);
		}
		n->typecheck = 0;
		n->right = nc;
		break;
	}
	goto ret;

divbymul:
	// try to do division by multiply by (2^w)/d
	// see hacker's delight chapter 10
	// TODO: support 64-bit magic multiply here.
	m.w = w;
	if(issigned[nl->type->etype]) {
		m.sd = mpgetfix(nr->val.u.xval);
		smagic(&m);
	} else {
		m.ud = mpgetfix(nr->val.u.xval);
		umagic(&m);
	}
	if(m.bad)
		return;

	// We have a quick division method so use it
	// for modulo too.
	if(n->op == OMOD)
		goto longmod;

	switch(simtype[nl->type->etype]) {
	default:
		return;

	case TUINT8:
	case TUINT16:
	case TUINT32:
		// n1 = nl * magic >> w (HMUL)
		nc = nod(OXXX, N, N);
		nodconst(nc, nl->type, m.um);
		n1 = nod(OMUL, nl, nc);
		typecheck(&n1, Erv);
		n1->op = OHMUL;
		if(m.ua) {
			// Select a Go type with (at least) twice the width.
			switch(simtype[nl->type->etype]) {
			default:
				return;
			case TUINT8:
			case TUINT16:
				twide = types[TUINT32];
				break;
			case TUINT32:
				twide = types[TUINT64];
				break;
			case TINT8:
			case TINT16:
				twide = types[TINT32];
				break;
			case TINT32:
				twide = types[TINT64];
				break;
			}

			// add numerator (might overflow).
			// n2 = (n1 + nl)
			n2 = nod(OADD, conv(n1, twide), conv(nl, twide));

			// shift by m.s
			nc = nod(OXXX, N, N);
			nodconst(nc, types[TUINT], m.s);
			n = conv(nod(ORSH, n2, nc), nl->type);
		} else {
			// n = n1 >> m.s
			nc = nod(OXXX, N, N);
			nodconst(nc, types[TUINT], m.s);
			n = nod(ORSH, n1, nc);
		}
		break;

	case TINT8:
	case TINT16:
	case TINT32:
		// n1 = nl * magic >> w
		nc = nod(OXXX, N, N);
		nodconst(nc, nl->type, m.sm);
		n1 = nod(OMUL, nl, nc);
		typecheck(&n1, Erv);
		n1->op = OHMUL;
		if(m.sm < 0) {
			// add the numerator.
			n1 = nod(OADD, n1, nl);
		}
		// shift by m.s
		nc = nod(OXXX, N, N);
		nodconst(nc, types[TUINT], m.s);
		n2 = conv(nod(ORSH, n1, nc), nl->type);
		// add 1 iff n1 is negative.
		nc = nod(OXXX, N, N);
		nodconst(nc, types[TUINT], w-1);
		n3 = nod(ORSH, nl, nc); // n4 = -1 iff n1 is negative.
		n = nod(OSUB, n2, n3);
		// apply sign.
		if(m.sd < 0)
			n = nod(OMINUS, n, N);
		break;
	}
	goto ret;

longmod:
	// rewrite as A%B = A - (A/B*B).
	n1 = nod(ODIV, nl, nr);
	n2 = nod(OMUL, n1, nr);
	n = nod(OSUB, nl, n2);
	goto ret;

ret:
	typecheck(&n, Erv);
	walkexpr(&n, init);
	*np = n;
}

// return 1 if integer n must be in range [0, max), 0 otherwise
static int
bounded(Node *n, int64 max)
{
	int64 v;
	int32 bits;
	int sign;

	if(n->type == T || !isint[n->type->etype])
		return 0;

	sign = issigned[n->type->etype];
	bits = 8*n->type->width;

	if(smallintconst(n)) {
		v = mpgetfix(n->val.u.xval);
		return 0 <= v && v < max;
	}

	switch(n->op) {
	case OAND:
		v = -1;
		if(smallintconst(n->left)) {
			v = mpgetfix(n->left->val.u.xval);
		} else if(smallintconst(n->right)) {
			v = mpgetfix(n->right->val.u.xval);
		}
		if(0 <= v && v < max)
			return 1;
		break;

	case OMOD:
		if(!sign && smallintconst(n->right)) {
			v = mpgetfix(n->right->val.u.xval);
			if(0 <= v && v <= max)
				return 1;
		}
		break;
	
	case ODIV:
		if(!sign && smallintconst(n->right)) {
			v = mpgetfix(n->right->val.u.xval);
			while(bits > 0 && v >= 2) {
				bits--;
				v >>= 1;
			}
		}
		break;
	
	case ORSH:
		if(!sign && smallintconst(n->right)) {
			v = mpgetfix(n->right->val.u.xval);
			if(v > bits)
				return 1;
			bits -= v;
		}
		break;
	}
	
	if(!sign && bits <= 62 && (1LL<<bits) <= max)
		return 1;
	
	return 0;
}

void
usefield(Node *n)
{
	Type *field, *l;

	if(!fieldtrack_enabled)
		return;

	switch(n->op) {
	default:
		fatal("usefield %O", n->op);
	case ODOT:
	case ODOTPTR:
		break;
	}
	
	field = n->paramfld;
	if(field == T)
		fatal("usefield %T %S without paramfld", n->left->type, n->right->sym);
	if(field->note == nil || strstr(field->note->s, "go:\"track\"") == nil)
		return;

	// dedup on list
	if(field->lastfn == curfn)
		return;
	field->lastfn = curfn;
	field->outer = n->left->type;
	if(isptr[field->outer->etype])
		field->outer = field->outer->type;
	if(field->outer->sym == S)
		yyerror("tracked field must be in named struct type");
	if(!exportname(field->sym->name))
		yyerror("tracked field must be exported (upper case)");

	l = typ(0);
	l->type = field;
	l->down = curfn->paramfld;
	curfn->paramfld = l;
}

static int
candiscardlist(NodeList *l)
{
	for(; l; l=l->next)
		if(!candiscard(l->n))
			return 0;
	return 1;
}

int
candiscard(Node *n)
{
	if(n == N)
		return 1;
	
	switch(n->op) {
	default:
		return 0;

	case ONAME:
	case ONONAME:
	case OTYPE:
	case OPACK:
	case OLITERAL:
	case OADD:
	case OSUB:
	case OOR:
	case OXOR:
	case OADDSTR:
	case OADDR:
	case OANDAND:
	case OARRAYBYTESTR:
	case OARRAYRUNESTR:
	case OSTRARRAYBYTE:
	case OSTRARRAYRUNE:
	case OCAP:
	case OCMPIFACE:
	case OCMPSTR:
	case OCOMPLIT:
	case OMAPLIT:
	case OSTRUCTLIT:
	case OARRAYLIT:
	case OPTRLIT:
	case OCONV:
	case OCONVIFACE:
	case OCONVNOP:
	case ODOT:
	case OEQ:
	case ONE:
	case OLT:
	case OLE:
	case OGT:
	case OGE:
	case OKEY:
	case OLEN:
	case OMUL:
	case OLSH:
	case ORSH:
	case OAND:
	case OANDNOT:
	case ONEW:
	case ONOT:
	case OCOM:
	case OPLUS:
	case OMINUS:
	case OOROR:
	case OPAREN:
	case ORUNESTR:
	case OREAL:
	case OIMAG:
	case OCOMPLEX:
		// Discardable as long as the subpieces are.
		break;

	case ODIV:
	case OMOD:
		// Discardable as long as we know it's not division by zero.
		if(isconst(n->right, CTINT) && mpcmpfixc(n->right->val.u.xval, 0) != 0)
			break;
		if(isconst(n->right, CTFLT) && mpcmpfltc(n->right->val.u.fval, 0) != 0)
			break;
		return 0;

	case OMAKECHAN:
	case OMAKEMAP:
		// Discardable as long as we know it won't fail because of a bad size.
		if(isconst(n->left, CTINT) && mpcmpfixc(n->left->val.u.xval, 0) == 0)
			break;
		return 0;
	
	case OMAKESLICE:
		// Difficult to tell what sizes are okay.
		return 0;		
	}
	
	if(!candiscard(n->left) ||
	   !candiscard(n->right) ||
	   !candiscard(n->ntest) ||
	   !candiscard(n->nincr) ||
	   !candiscardlist(n->ninit) ||
	   !candiscardlist(n->nbody) ||
	   !candiscardlist(n->nelse) ||
	   !candiscardlist(n->list) ||
	   !candiscardlist(n->rlist)) {
		return 0;
	}
	
	return 1;
}
