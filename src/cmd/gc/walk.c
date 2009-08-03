// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"

static	Node*	walkprint(Node*, NodeList**);
static	Node*	mkcall(char*, Type*, NodeList**, ...);
static	Node*	mkcall1(Node*, Type*, NodeList**, ...);
static	Node*	conv(Node*, Type*);
static	Node*	chanfn(char*, int, Type*);
static	Node*	mapfn(char*, Type*);
static	Node*	makenewvar(Type*, NodeList**, Node**);
enum
{
	Inone,
	I2T,
	I2T2,
	I2I,
	I2Ix,
	I2I2,
	T2I,
	I2Isame,
	E2T,
	E2T2,
	E2I,
	E2I2,
	I2E,
	I2E2,
	T2E,
	E2Esame,
};

// can this code branch reach the end
// without an undcontitional RETURN
// this is hard, so it is conservative
int
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
		return 0;

	case OCALL:
		if(n->left->op == ONAME) {
			switch(n->left->etype) {
			case OPANIC:
			case OPANICN:
				return 0;
			}
		}
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

	curfn = fn;
	if(debug['W']) {
		snprint(s, sizeof(s), "\nbefore %S", curfn->nname->sym);
		dumplist(s, curfn->nbody);
	}
	if(curfn->type->outtuple)
		if(walkret(curfn->nbody))
			yyerror("function ends without a return statement");
	typechecklist(curfn->nbody, Etop);
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
gettype(Node **np, NodeList **init)
{
	if(debug['W'])
		dump("\nbefore gettype", *np);
	typecheck(np, Erv);
	if(debug['W'])
		dump("after gettype", *np);
}

void
walkdeflist(NodeList *l)
{
	for(; l; l=l->next)
		walkdef(l->n);
}

void
walkdef(Node *n)
{
	int lno;
	NodeList *init;
	Node *e;
	Type *t;

	lno = lineno;
	setlineno(n);

	if(n->op == ONONAME) {
		if(!n->diag) {
			n->diag = 1;
			yyerror("undefined: %S", n->sym);
		}
		return;
	}

	if(n->walkdef == 1)
		return;
	if(n->walkdef == 2) {
		// TODO(rsc): better loop message
		fatal("loop");
	}
	n->walkdef = 2;

	if(n->type != T || n->sym == S)	// builtin or no name
		goto ret;


	init = nil;
	switch(n->op) {
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
		typecheck(&e, Erv);
		if(e->op != OLITERAL) {
			yyerror("const initializer must be constant");
			goto ret;
		}
		t = n->type;
		if(t != T)
			convlit(&e, t);
		n->val = e->val;
		n->type = e->type;
		break;
	}

ret:
	lineno = lno;
	n->walkdef = 1;
}

void
walkstmtlist(NodeList *l)
{
	for(; l; l=l->next)
		walkstmt(&l->n);
}

void
walkstmt(Node **np)
{
	NodeList *init;
	NodeList *ll;
	int lno;
	Node *n;

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

	case OAPPENDSTR:
	case OASOP:
	case OAS:
	case OAS2:
	case OCLOSE:
	case OCLOSED:
	case OCALLMETH:
	case OCALLINTER:
	case OCALL:
	case OCALLFUNC:
	case OSEND:
	case ORECV:
	case OPRINT:
	case OPRINTN:
	case OPANIC:
	case OPANICN:
	case OEMPTY:
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
		walkexpr(&n->left, &n->ninit);
		break;

	case OFOR:
		walkstmtlist(n->ninit);
		walkbool(&n->ntest);
		walkstmt(&n->nincr);
		walkstmtlist(n->nbody);
		break;

	case OIF:
		walkstmtlist(n->ninit);
		walkbool(&n->ntest);
		walkstmtlist(n->nbody);
		walkstmtlist(n->nelse);
		break;

	case OPROC:
		walkexpr(&n->left, &n->ninit);
		break;

	case ORETURN:
		walkexprlist(n->list, &n->ninit);
		if(curfn->type->outnamed && n->list == nil) {
			// print("special return\n");
			break;
		}
		ll = ascompatte(n->op, getoutarg(curfn->type), n->list, 1, &n->ninit);
		n->list = reorder4(ll);
		break;

	case OSELECT:
		walkselect(n);
		break;

	case OSWITCH:
		walkswitch(n);
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
walkexpr(Node **np, NodeList **init)
{
	Node *r, *l;
	NodeList *ll, *lr;
	Type *t;
	int et, cl, cr;
	int32 lno;
	Node *n, *fn;

	n = *np;

	if(n == N)
		return;

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
	case OLEN:
	case OCAP:
	case ODOT:
	case ODOTPTR:
	case ODOTMETH:
	case ODOTINTER:
	case OIND:
		walkexpr(&n->left, init);
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
		walkexpr(&n->left, init);
		walkexpr(&n->right, init);
		goto ret;

	case OPRINT:
	case OPRINTN:
	case OPANIC:
	case OPANICN:
		walkexprlist(n->list, init);
		n = walkprint(n, init);
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
		walkexpr(&n->right, init);
		l = n->left;
		r = n->right;
		if(l == N || r == N)
			goto ret;
		r = ascompatee1(n->op, l, r, init);
		if(r != N)
			n = r;
		goto ret;

	case OAS2:
		*init = concat(*init, n->ninit);
		n->ninit = nil;

		walkexprlist(n->list, init);

		cl = count(n->list);
		cr = count(n->rlist);
		if(cl == cr) {
		multias:
			walkexprlist(n->rlist, init);
			ll = ascompatee(OAS, n->list, n->rlist, init);
			ll = reorder3(ll);
			n = liststmt(ll);
			goto ret;
		}

		l = n->list->n;
		r = n->rlist->n;

		// count mismatch - special cases
		switch(r->op) {
		case OCALLMETH:
		case OCALLINTER:
		case OCALLFUNC:
		case OCALL:
			if(cr == 1) {
				// a,b,... = fn()
				walkexpr(&r, init);
				if(r->type == T || r->type->etype != TSTRUCT)
					break;
				ll = ascompatet(n->op, n->list, &r->type, 0, init);
				n = liststmt(concat(list1(r), ll));
				goto ret;
			}
			break;

		case OINDEXMAP:
			if(cl == 2 && cr == 1) {
				// a,b = map[] - mapaccess2
				walkexpr(&r->left, init);
				l = mapop(n, init);
				if(l == N)
					break;
				n = l;
				goto ret;
			}
			break;

		case ORECV:
			if(cl == 2 && cr == 1) {
				// a,b = <chan - chanrecv2
				walkexpr(&r->left, init);
				if(!istype(r->left->type, TCHAN))
					break;
				l = chanop(n, init);
				if(l == N)
					break;
				n = l;
				goto ret;
			}
			break;

		case ODOTTYPE:
			walkdottype(r, init);
			if(cl == 2 && cr == 1) {
				// a,b = i.(T)
				if(r->left == N)
					break;
				et = ifaceas1(r->type, r->left->type, 1);
				switch(et) {
				case I2Isame:
				case E2Esame:
					n->rlist = list(list1(r->left), nodbool(1));
					typechecklist(n->rlist, Erv);
					goto multias;
				case I2E:
					n->list = list(list1(n->right), nodbool(1));
					typechecklist(n->rlist, Erv);
					goto multias;
				case I2T:
					et = I2T2;
					break;
				case I2Ix:
					et = I2I2;
					break;
				case E2I:
					et = E2I2;
					break;
				case E2T:
					et = E2T2;
					break;
				default:
					et = Inone;
					break;
				}
				if(et == Inone)
					break;
				r = ifacecvt(r->type, r->left, et, init);
				ll = ascompatet(n->op, n->list, &r->type, 0, init);
				n = liststmt(concat(list1(r), ll));
				goto ret;
			}
			break;
		}

		switch(l->op) {
		case OINDEXMAP:
			if(cl == 1 && cr == 2) {
				// map[] = a,b - mapassign2
				l = mapop(n, init);
				if(l == N)
					break;
				n = l;
				goto ret;
			}
			break;
		}
		if(l->diag == 0) {
			l->diag = 1;
			yyerror("assignment count mismatch: %d = %d", cl, cr);
		}
		goto ret;

	case ODOTTYPE:
		walkdottype(n, init);
		walkconv(&n, init);
		goto ret;

	case OCONV:
	case OCONVNOP:
		walkexpr(&n->left, init);
		goto ret;

	case OASOP:
		walkexpr(&n->left, init);
		l = n->left;
		if(l->op == OINDEXMAP)
			n = mapop(n, init);
		walkexpr(&n->right, init);
		if(n->etype == OANDNOT) {
			n->etype = OAND;
			n->right = nod(OCOM, n->right, N);
			n->right->type = n->right->left->type;
			goto ret;
		}

		/*
		 * on 32-bit arch, rewrite 64-bit ops into l = l op r
		 */
		et = n->left->type->etype;
		if(widthptr == 4 && (et == TUINT64 || et == TINT64)) {
			l = saferef(n->left, init);
			r = nod(OAS, l, nod(n->etype, l, n->right));
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
		n->right->type = n->right->left->type;
		goto ret;

	case ODIV:
	case OMOD:
		/*
		 * rewrite div and mod into function calls
		 * on 32-bit architectures.
		 */
		walkexpr(&n->left, init);
		walkexpr(&n->right, init);
		et = n->left->type->etype;
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
		walkexpr(&n->right->left, init);
		walkexpr(&n->right->right, init);
		// dynamic slice
		// arraysliced(old []any, lb int, hb int, width int) (ary []any)
		t = n->type;
		fn = syslook("arraysliced", 1);
		argtype(fn, t->type);			// any-1
		argtype(fn, t->type);			// any-2
		n = mkcall1(fn, t, init,
			n->left,
			conv(n->right->left, types[TINT]),
			conv(n->right->right, types[TINT]),
			nodintconst(t->type->width));
		goto ret;

	case OSLICEARR:
		walkexpr(&n->left, init);
		walkexpr(&n->right->left, init);
		walkexpr(&n->right->right, init);
		// static slice
		// arrayslices(old *any, nel int, lb int, hb int, width int) (ary []any)
		t = n->type;
		fn = syslook("arrayslices", 1);
		argtype(fn, n->left->type);	// any-1
		argtype(fn, t->type);			// any-2
		n = mkcall1(fn, t, init,
			nod(OADDR, n->left, N), nodintconst(t->bound),
			conv(n->right->left, types[TINT]),
			conv(n->right->right, types[TINT]),
			nodintconst(t->type->width));
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
			nvar = makenewvar(n->type, init, &nstar);
			arraylit(n->left, nstar, init);
			n = nvar;
			goto ret;

		case OMAPLIT:
			nvar = makenewvar(n->type, init, &nstar);
			maplit(n->left, nstar, init);
			n = nvar;
			goto ret;


		case OSTRUCTLIT:
			nvar = makenewvar(n->type, init, &nstar);
			structlit(n->left, nstar, init);
			n = nvar;
			goto ret;
		}

		walkexpr(&n->left, init);
		goto ret;

	case ONEW:
		n = callnew(n->type->type);
		goto ret;

	case OCMPSTR:
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

	case OAPPENDSTR:
		// s1 = sys_catstring(s1, s2)
		if(n->etype != OADD)
			fatal("walkasopstring: not add");
		r = mkcall("catstring", n->left->type, init,
			conv(n->left, types[TSTRING]),
			conv(n->right, types[TSTRING]));
		r = nod(OAS, n->left, r);
		n = r;
		goto ret;

	case OSLICESTR:
		// sys_slicestring(s, lb, hb)
		n = mkcall("slicestring", n->type, init,
			conv(n->left, types[TSTRING]),
			conv(n->right->left, types[TINT]),
			conv(n->right->right, types[TINT]));
		goto ret;

	case OINDEXSTR:
		// TODO(rsc): should be done in back end
		// sys_indexstring(s, i)
		n = mkcall("indexstring", n->type, init,
			conv(n->left, types[TSTRING]),
			conv(n->right, types[TINT]));
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
		n = mkcall1(chanfn("newchan", 1, n->type), n->type, init,
			nodintconst(n->type->type->width),
			nodintconst(algtype(n->type->type)),
			conv(n->left, types[TINT]));
		goto ret;

	case OMAKEMAP:
		t = n->type;

		fn = syslook("newmap", 1);
		argtype(fn, t->down);	// any-1
		argtype(fn, t->type);	// any-2

		n = mkcall1(fn, n->type, init,
			nodintconst(t->down->width),	// key width
			nodintconst(t->type->width),		// val width
			nodintconst(algtype(t->down)),	// key algorithm
			nodintconst(algtype(t->type)),		// val algorithm
			conv(n->left, types[TINT]));
		goto ret;

	case OMAKESLICE:
		// newarray(nel int, max int, width int) (ary []any)
		t = n->type;
		fn = syslook("newarray", 1);
		argtype(fn, t->type);			// any-1
		n = mkcall1(fn, n->type, nil,
			conv(n->left, types[TINT]),
			conv(n->right, types[TINT]),
			nodintconst(t->type->width));
		goto ret;

	case ORUNESTR:
		// sys_intstring(v)
		n = mkcall("intstring", n->type, init, conv(n->left, types[TINT64]));	// TODO(rsc): int64?!
		goto ret;

	case OARRAYBYTESTR:
		// arraystring([]byte) string;
		n = mkcall("arraystring", n->type, init, n->left);
		goto ret;

	case OARRAYRUNESTR:
		// arraystring([]byte) string;
		n = mkcall("arraystringi", n->type, init, n->left);
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
		n = arraylit(n, N, init);
		goto ret;

	case OMAPLIT:
		n = maplit(n, N, init);
		goto ret;

	case OSTRUCTLIT:
		n = structlit(n, N, init);
		goto ret;

	case OSEND:
		n = mkcall1(chanfn("chansend1", 2, n->left->type), T, init, n->left, n->right);
		goto ret;

	case OSENDNB:
		n = mkcall1(chanfn("chansend2", 2, n->left->type), n->type, init, n->left, n->right);
		goto ret;

	case OCONVIFACE:
		walkexpr(&n->left, init);
		n = ifacecvt(n->type, n->left, n->etype, init);
		goto ret;

	case OCONVSLICE:
		// arrays2d(old *any, nel int) (ary []any)
		fn = syslook("arrays2d", 1);
		argtype(fn, n->left->type->type);		// any-1
		argtype(fn, n->type->type);			// any-2
		n = mkcall1(fn, n->type, init, n->left, nodintconst(n->left->type->type->bound));
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

Node*
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

void
walkbool(Node **np)
{
	Node *n;

	n = *np;
	if(n == N)
		return;
	walkexpr(np, &n->ninit);
	defaultlit(np, T);
	n = *np;
	if(n->type != T && !eqtype(n->type, types[TBOOL]))
		yyerror("IF and FOR require a boolean type");
}

void
walkdottype(Node *n, NodeList **init)
{
	walkexpr(&n->left, init);
	if(n->left == N)
		return;
	if(n->right != N) {
		walkexpr(&n->right, init);
		n->type = n->right->type;
		n->right = N;
	}
}

void
walkconv(Node **np, NodeList **init)
{
	int et;
	char *what;
	Type *t;
	Node *l;
	Node *n;

	n = *np;
	t = n->type;
	if(t == T)
		return;
	walkexpr(&n->left, init);
	l = n->left;
	if(l == N)
		return;
	if(l->type == T)
		return;

	// if using .(T), interface assertion.
	if(n->op == ODOTTYPE) {
		et = ifaceas1(t, l->type, 1);
		if(et == I2Isame || et == E2Esame) {
			n->op = OCONVNOP;
			return;
		}
		if(et != Inone) {
			n = ifacecvt(t, l, et, init);
			*np = n;
			return;
		}
		goto bad;
	}

	fatal("walkconv");

bad:
	if(n->diag)
		return;
	n->diag = 1;
	if(n->op == ODOTTYPE)
		what = "type assertion";
	else
		what = "conversion";
	if(l->type != T)
		yyerror("invalid %s: %T to %T", what, l->type, t);
}

Node*
selcase(Node *n, Node *var, NodeList **init)
{
	Node *a, *r, *c;
	Type *t;

	if(n->list == nil)
		goto dflt;
	c = n->list->n;
	if(c->op == ORECV)
		goto recv;

	walkexpr(&c->left, init);		// chan
	walkexpr(&c->right, init);	// elem

	t = fixchan(c->left->type);
	if(t == T)
		return N;

	if(!(t->chan & Csend)) {
		yyerror("cannot send on %T", t);
		return N;
	}

	convlit(&c->right, t->type);
	if(!ascompat(t->type, c->right->type)) {
		badtype(c->op, t->type, c->right->type);
		return N;
	}

	// selectsend(sel *byte, hchan *chan any, elem any) (selected bool);
	a = mkcall1(chanfn("selectsend", 2, t), types[TBOOL], init, var, c->left, c->right);
	goto out;

recv:
	if(c->right != N)
		goto recv2;

	walkexpr(&c->left, init);		// chan

	t = fixchan(c->left->type);
	if(t == T)
		return N;

	if(!(t->chan & Crecv)) {
		yyerror("cannot receive from %T", t);
		return N;
	}

	// selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
	a = mkcall1(chanfn("selectrecv", 2, t), types[TBOOL], init, var, c->left, nodnil());
	goto out;

recv2:
	walkexpr(&c->right, init);	// chan

	t = fixchan(c->right->type);
	if(t == T)
		return N;

	if(!(t->chan & Crecv)) {
		yyerror("cannot receive from %T", t);
		return N;
	}

	walkexpr(&c->left, init);

	// selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
	a = mkcall1(chanfn("selectrecv", 2, t), types[TBOOL], init, var, c->right, nod(OADDR, c->left, N));
	goto out;

dflt:
	// selectdefault(sel *byte);
	a = mkcall("selectdefault", types[TBOOL], init, var);
	goto out;

out:
	r = nod(OIF, N, N);
	r->ntest = a;

	return r;
}

/*
 * enumerate the special cases
 * of the case statement:
 *	case v := <-chan		// select and switch
 */
Node*
selectas(Node *name, Node *expr, NodeList **init)
{
	Type *t;

	if(expr == N || expr->op != ORECV)
		goto bad;

	walkexpr(&expr->left, init);
	t = expr->left->type;
	if(t == T)
		goto bad;
	if(t->etype != TCHAN)
		goto bad;
	t = t->type;
	return old2new(name, t, init);

bad:
	return name;
}

void
walkselect(Node *sel)
{
	Node *n, *l, *oc, *on, *r;
	Node *var, *def;
	NodeList *res, *bod, *nbod, *init, *ln;
	int count, op;
	int32 lno;

	lno = setlineno(sel);

	init = nil;

	// generate sel-struct
	var = nod(OXXX, N, N);
	tempname(var, ptrto(types[TUINT8]));

	if(sel->list == nil) {
		yyerror("empty select");
		return;
	}

	count = 0;	// number of cases
	res = nil;	// entire select body
	bod = nil;	// body of each case
	oc = N;		// last case
	def = N;	// default case
	for(ln=sel->list; ln; ln=ln->next) {
		n = ln->n;
		setlineno(n);
		if(n->op != OXCASE)
			fatal("walkselect %O", n->op);

		count++;
		l = N;
		if(n->list == nil) {
			op = ORECV;	// actual value not used
			if(def != N)
				yyerror("repeated default; first at %L", def->lineno);
			def = n;
		} else {
			l = n->list->n;
			op = l->op;
			if(n->list->next) {
				yyerror("select cases cannot be lists");
				continue;
			}
		}

		nbod = nil;
		switch(op) {
		default:
			yyerror("select cases must be send, recv or default %O", op);
			continue;

		case OAS:
			// convert new syntax (a=recv(chan)) to (recv(a,chan))
			if(l->right == N || l->right->op != ORECV) {
				yyerror("select cases must be send, recv or default %O", l->right->op);
				break;
			}
			r = l->right;	// rcv
			r->right = r->left;
			r->left = l->left;
			n->list->n = r;

			// convert case x := foo: body
			// to case tmp := foo: x := tmp; body.
			// if x escapes and must be allocated
			// on the heap, this delays the allocation
			// until after the select has chosen this branch.
			if(n->ninit != nil && n->ninit->n->op == ODCL) {
				on = nod(OXXX, N, N);
				tempname(on, l->left->type);
				on->sym = lookup("!tmpselect!");
				r->left = on;
				on = nod(OAS, l->left, on);
				typecheck(&on, Etop);
				nbod = list(n->ninit, on);
				n->ninit = nil;
			}
			break;

		case OSEND:
		case OSENDNB:
		case ORECV:
			break;
		}

		nbod = concat(nbod, n->nbody);
		nbod = list(nbod, nod(OBREAK, N, N));
		n->nbody = nil;

		oc = selcase(n, var, &init);
		if(oc != N) {
			oc->nbody = nbod;
			res = list(res, oc);
		}
	}
	setlineno(sel);

	// selectgo(sel *byte);
	res = list(res, mkcall("selectgo", T, nil, var));

	// newselect(size uint32) (sel *byte);
	r = nod(OAS, var, mkcall("newselect", var->type, nil, nodintconst(count)));
	typecheck(&r, Etop);
	typechecklist(res, Etop);

	sel->ninit = list1(r);
	sel->nbody = res;
	sel->left = N;

	walkstmtlist(sel->ninit);
	walkstmtlist(sel->nbody);
//dump("sel", sel);

	sel->ninit = concat(sel->ninit, init);
	lineno = lno;
}

Node*
ascompatee1(int op, Node *l, Node *r, NodeList **init)
{
	Node *a;

	/*
	 * check assign expression to
	 * a expression. called in
	 *	expr = expr
	 */
	if(l->type != T && l->type->etype == TFORW)
		return N;
	if(r->type != T && r->type->etype ==TFORW)
		return N;
	convlit(&r, l->type);
	if(!ascompat(l->type, r->type)) {
		badtype(op, l->type, r->type);
		return N;
	}
	a = nod(OAS, l, r);
	a = convas(a, init);
	return a;
}

NodeList*
ascompatee(int op, NodeList *nl, NodeList *nr, NodeList **init)
{
	NodeList *ll, *lr, *nn;

	/*
	 * check assign expression list to
	 * a expression list. called in
	 *	expr-list = expr-list
	 */
	nn = nil;
	for(ll=nl, lr=nr; ll && lr; ll=ll->next, lr=lr->next)
		nn = list(nn, ascompatee1(op, ll->n, lr->n, init));

	// cannot happen: caller checked that lists had same length
	if(ll || lr)
		yyerror("error in shape across %O", op);
	return nn;
}

/*
 * n is an lv and t is the type of an rv
 * return 1 if this implies a function call
 * evaluating the lv or a function call
 * in the conversion of the types
 */
int
fncall(Node *l, Type *rt)
{
	if(l->ullman >= UINF)
		return 1;
	if(eqtype(l->type, rt))
		return 0;
	return 1;
}

NodeList*
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
		if(!ascompat(l->type, r->type)) {
			badtype(op, l->type, r->type);
			return nil;
		}

		// any lv that causes a fn call must be
		// deferred until all the return arguments
		// have been pulled from the output arguments
		if(fncall(l, r->type)) {
			tmp = nod(OXXX, N, N);
			tempname(tmp, r->type);
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
		yyerror("reorder2: too many function calls evaluating parameters");
	return concat(nn, mm);
}

/*
 * make a tsig for the structure
 * carrying the ... arguments
 */
Type*
sigtype(Type *st)
{
	Sym *s;
	Type *t;
	static int sigdddgen;

	dowidth(st);

	sigdddgen++;
	snprint(namebuf, sizeof(namebuf), "dsigddd_%d", sigdddgen);
	s = lookup(namebuf);
	t = newtype(s);
	t = dodcltype(t);
	updatetype(t, st);
	t->local = 1;
	return t;
}

/*
 * package all the arguments that
 * match a ... parameter into an
 * automatic structure.
 * then call the ... arg (interface)
 * with a pointer to the structure.
 */
NodeList*
mkdotargs(NodeList *lr0, NodeList *nn, Type *l, int fp, NodeList **init)
{
	Node *r;
	Type *t, *st, *ft;
	Node *a, *var;
	NodeList *lr, *n;

	n = nil;			// list of assignments

	st = typ(TSTRUCT);	// generated structure
	ft = T;			// last field
	for(lr=lr0; lr; lr=lr->next) {
		r = lr->n;
		if(r->op == OLITERAL && r->val.ctype == CTNIL) {
			if(r->type == T || r->type->etype == TNIL) {
				yyerror("inappropriate use of nil in ... argument");
				return nil;
			}
		}
		defaultlit(&r, T);
		lr->n = r;
		if(r->type == T)	// type check failed
			return nil;

		// generate the next structure field
		t = typ(TFIELD);
		t->type = r->type;
		if(ft == T)
			st->type = t;
		else
			ft->down = t;
		ft = t;

		a = nod(OAS, N, r);
		n = list(n, a);
	}

	// make a named type for the struct
	st = sigtype(st);
	dowidth(st);

	// now we have the size, make the struct
	var = nod(OXXX, N, N);
	tempname(var, st);
	var->sym = lookup(".ddd");

	// assign the fields to the struct.
	// use the init list so that reorder1 doesn't reorder
	// these assignments after the interface conversion
	// below.
	t = st->type;
	for(lr=n; lr; lr=lr->next) {
		r = lr->n;
		r->left = nod(OXXX, N, N);
		*r->left = *var;
		r->left->type = r->right->type;
		r->left->xoffset += t->width;
		typecheck(&r, Etop);
		walkexpr(&r, init);
		lr->n = r;
		t = t->down;
	}
	*init = concat(*init, n);

	// last thing is to put assignment
	// of the structure to the DDD parameter
	a = nod(OAS, nodarg(l, fp), var);
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
NodeList*
ascompatte(int op, Type **nl, NodeList *lr, int fp, NodeList **init)
{
	Type *l, *ll;
	Node *r, *a;
	NodeList *nn, *lr0;
	Iter savel, peekl;

	lr0 = lr;
	l = structfirst(&savel, nl);
	r = N;
	if(lr)
		r = lr->n;
	nn = nil;

	// 1 to many
	peekl = savel;
	if(l != T && r != N
	&& structnext(&peekl) != T
	&& lr->next == nil
	&& eqtypenoname(r->type, *nl)) {
		// clumsy check for differently aligned structs.
		// now that output structs are aligned separately
		// from the input structs, should never happen.
		if(r->type->width != (*nl)->width)
			fatal("misaligned multiple return\n\t%T\n\t%T", r->type, *nl);
		a = nodarg(*nl, fp);
		a->type = r->type;
		nn = list1(convas(nod(OAS, a, r), init));
		goto ret;
	}

loop:
	if(l != T && isddd(l->type)) {
		// the ddd parameter must be last
		ll = structnext(&savel);
		if(ll != T)
			yyerror("... must be last argument");

		// special case --
		// only if we are assigning a single ddd
		// argument to a ddd parameter then it is
		// passed thru unencapsulated
		if(r != N && lr->next == nil && isddd(r->type)) {
			a = nod(OAS, nodarg(l, fp), r);
			a = convas(a, init);
			nn = list(nn, a);
			goto ret;
		}

		// normal case -- make a structure of all
		// remaining arguments and pass a pointer to
		// it to the ddd parameter (empty interface)
		nn = mkdotargs(lr, nn, l, fp, init);
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

	if(!ascompat(l->type, r->type)) {
		badtype(op, l->type, r->type);
		return nil;
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

/*
 * can we assign var of type src to var of type dst?
 * return 0 if not, 1 if conversion is trivial, 2 if conversion is non-trivial.
 */
int
ascompat(Type *dst, Type *src)
{
	if(eqtype(dst, src))
		return 1;

	if(dst == T || src == T)
		return 0;

	if(dst->etype == TFORWINTER || dst->etype == TFORWSTRUCT || dst->etype == TFORW)
		return 0;
	if(src->etype == TFORWINTER || src->etype == TFORWSTRUCT || src->etype == TFORW)
		return 0;

	// interfaces go through even if names don't match
	if(isnilinter(dst) || isnilinter(src))
		return 2;

	if(isinter(dst) && isinter(src))
		return 2;

	if(isinter(dst) && methtype(src))
		return 2;

	if(isinter(src) && methtype(dst))
		return 2;

	// otherwise, if concrete types have names, they must match
	if(dst->sym && src->sym && dst != src)
		return 0;

	if(dst->etype == TCHAN && src->etype == TCHAN) {
		if(!eqtype(dst->type, src->type))
			return 0;
		if(dst->chan & ~src->chan)
			return 0;
		return 1;
	}

	if(isslice(dst)
	&& isptr[src->etype]
	&& isfixedarray(src->type)
	&& eqtype(dst->type, src->type->type))
		return 2;

	return 0;
}

// generate code for print
static Node*
walkprint(Node *nn, NodeList **init)
{
	Node *r;
	Node *n;
	NodeList *l, *all;
	Node *on;
	Type *t;
	int notfirst, et, op;
	NodeList *calls;

	op = nn->op;
	all = nn->list;
	calls = nil;
	notfirst = 0;

	for(l=all; l; l=l->next) {
		if(notfirst)
			calls = list(calls, mkcall("printsp", T, init));
		notfirst = op == OPRINTN || op == OPANICN;

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

		et = n->type->etype;
		if(isinter(n->type)) {
			if(isnilinter(n->type))
				on = syslook("printeface", 1);
			else
				on = syslook("printiface", 1);
			argtype(on, n->type);		// any-1
		} else if(isptr[et] || et == TCHAN || et == TMAP || et == TFUNC) {
			on = syslook("printpointer", 1);
			argtype(on, n->type);	// any-1
		} else if(isslice(n->type)) {
			on = syslook("printarray", 1);
			argtype(on, n->type);	// any-1
		} else if(isint[et]) {
			if(et == TUINT64)
				on = syslook("printuint", 0);
			else
				on = syslook("printint", 0);
		} else if(isfloat[et]) {
			on = syslook("printfloat", 0);
		} else if(et == TBOOL) {
			on = syslook("printbool", 0);
		} else if(et == TSTRING) {
			on = syslook("printstring", 0);
		} else {
			badtype(OPRINT, n->type, T);
			continue;
		}

		t = *getinarg(on->type);
		if(t != nil)
			t = t->type;
		if(t != nil)
			t = t->type;

		if(!eqtype(t, n->type)) {
			n = nod(OCONV, n, N);
			n->type = t;
		}
		r = nod(OCALL, on, N);
		r->list = list1(n);
		calls = list(calls, r);
	}

	if(op == OPRINTN)
		calls = list(calls, mkcall("printnl", T, nil));
	typechecklist(calls, Etop);
	walkexprlist(calls, init);

	if(op == OPANIC || op == OPANICN)
		r = mkcall("panicl", T, nil);
	else
		r = nod(OEMPTY, N, N);
	typecheck(&r, Etop);
	walkexpr(&r, init);
	r->ninit = calls;
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

Type*
fixmap(Type *t)
{
	if(t == T)
		goto bad;
	if(t->etype != TMAP)
		goto bad;
	if(t->down == T || t->type == T)
		goto bad;

	dowidth(t->down);
	dowidth(t->type);

	return t;

bad:
	yyerror("not a map: %lT", t);
	return T;
}

Type*
fixchan(Type *t)
{
	if(t == T)
		goto bad;
	if(t->etype != TCHAN)
		goto bad;
	if(t->type == T)
		goto bad;

	dowidth(t->type);

	return t;

bad:
	yyerror("not a channel: %lT", t);
	return T;
}

Node*
mapop(Node *n, NodeList **init)
{
	Node *r, *a, *l;
	Type *t;
	Node *fn;
	int cl, cr;
	NodeList *args;

	r = n;
	switch(n->op) {
	default:
		fatal("mapop: unknown op %O", n->op);

	case OAS:
		// mapassign1(hmap map[any-1]any-2, key any-3, val any-4);
		if(n->left->op != OINDEXMAP)
			goto shape;

		t = fixmap(n->left->left->type);
		if(t == T)
			break;

		r = mkcall1(mapfn("mapassign1", t), T, init, n->left->left, n->left->right, n->right);
		break;

	case OAS2:
		cl = count(n->list);
		cr = count(n->rlist);

		if(cl == 1 && cr == 2)
			goto assign2;
		if(cl == 2 && cr == 1)
			goto access2;
		goto shape;

	assign2:
		// mapassign2(hmap map[any]any, key any, val any, pres bool);
		l = n->list->n;
		if(l->op != OINDEXMAP)
			goto shape;

		t = fixmap(l->left->type);
		if(t == T)
			break;

		r = mkcall1(mapfn("mapassign2", t), T, init, l->left, l->right, n->rlist->n, n->rlist->next->n);
		break;

	access2:
		// mapaccess2(hmap map[any-1]any-2, key any-3) (val-4 any, pres bool);

//dump("access2", n);
		r = n->rlist->n;
		if(r->op != OINDEXMAP)
			goto shape;

		t = fixmap(r->left->type);
		if(t == T)
			break;

		args = list1(r->left);		// map
		args = list(args, r->right);		// key

		fn = mapfn("mapaccess2", t);
		a = mkcall1(fn, getoutargx(fn->type), init, r->left, r->right);
		n->rlist = list1(a);
		typecheck(&n, Etop);
		walkexpr(&n, init);
		r = n;
		break;

	case OASOP:
		// rewrite map[index] op= right
		// into tmpi := index; map[tmpi] = map[tmpi] op right
		// TODO(rsc): does this double-evaluate map?

		t = n->left->left->type;
		a = nod(OXXX, N, N);
		tempname(a, t->down);			// tmpi
		r = nod(OAS, a, n->left->right);	// tmpi := index
		n->left->right = a;			// m[tmpi]
		typecheck(&r, Etop);
		walkexpr(&r, init);
		*init = list(*init, r);

		a = nod(OXXX, N, N);
		*a = *n->left;		// copy of map[tmpi]
		a->etype = 0;
		a = nod(n->etype, a, n->right);		// m[tmpi] op right
		r = nod(OAS, n->left, a);		// map[tmpi] = map[tmpi] op right
		typecheck(&r, Etop);
		walkexpr(&r, init);
		break;
	}
	return r;

shape:
	dump("shape", n);
	fatal("mapop: %O", n->op);
	return N;
}

Node*
chanop(Node *n, NodeList **init)
{
	Node *r, *fn;
	Type *t;
	int cl, cr;

	r = n;
	switch(n->op) {
	default:
		fatal("chanop: unknown op %O", n->op);

	case OAS2:
		cl = count(n->list);
		cr = count(n->rlist);

		if(cl != 2 || cr != 1 || n->rlist->n->op != ORECV)
			goto shape;

		// chanrecv2(hchan *chan any) (elem any, pres bool);
		r = n->rlist->n;
		defaultlit(&r->left, T);
		t = fixchan(r->left->type);
		if(t == T)
			break;
		if(!(t->chan & Crecv)) {
			yyerror("cannot receive from %T", t);
			break;
		}

		fn = chanfn("chanrecv2", 2, t);
		r = mkcall1(fn, getoutargx(fn->type), init, r->left);
		n->rlist->n = r;
		r = n;
		walkexpr(&r, init);
		break;
	}
	return r;

shape:
	fatal("chanop: %O", n->op);
	return N;
}


Type*
fixarray(Type *t)
{

	if(t == T)
		goto bad;
	if(t->etype != TARRAY)
		goto bad;
	if(t->type == T)
		goto bad;
	dowidth(t);
	return t;

bad:
	yyerror("not an array: %lT", t);
	return T;

}

/*
 * assigning src to dst involving interfaces?
 * return op to use.
 */
int
ifaceas1(Type *dst, Type *src, int explicit)
{
	if(src == T || dst == T)
		return Inone;

	if(explicit && !isinter(src))
		yyerror("cannot use .(T) on non-interface type %T", src);

	if(isinter(dst)) {
		if(isinter(src)) {
			if(isnilinter(dst)) {
				if(isnilinter(src))
					return E2Esame;
				return I2E;
			}
			if(eqtype(dst, src))
				return I2Isame;
			ifacecheck(dst, src, lineno, explicit);
			if(isnilinter(src))
				return E2I;
			if(explicit)
				return I2Ix;
			return I2I;
		}
		if(isnilinter(dst))
			return T2E;
		ifacecheck(dst, src, lineno, explicit);
		return T2I;
	}
	if(isinter(src)) {
		ifacecheck(dst, src, lineno, explicit);
		if(isnilinter(src))
			return E2T;
		return I2T;
	}
	return Inone;
}

/*
 * treat convert T to T as noop
 */
int
ifaceas(Type *dst, Type *src, int explicit)
{
	int et;

	et = ifaceas1(dst, src, explicit);
	if(et == I2Isame || et == E2Esame)
		et = Inone;
	return et;
}

static	char*
ifacename[] =
{
	[I2T]		= "ifaceI2T",
	[I2T2]		= "ifaceI2T2",
	[I2I]		= "ifaceI2I",
	[I2Ix]		= "ifaceI2Ix",
	[I2I2]		= "ifaceI2I2",
	[I2Isame]	= "ifaceI2Isame",
	[E2T]		= "ifaceE2T",
	[E2T2]		= "ifaceE2T2",
	[E2I]		= "ifaceE2I",
	[E2I2]		= "ifaceE2I2",
	[I2E]		= "ifaceI2E",
	[I2E2]		= "ifaceI2E2",
	[T2I]		= "ifaceT2I",
	[T2E]		= "ifaceT2E",
	[E2Esame]	= "ifaceE2Esame",
};

Node*
ifacecvt(Type *tl, Node *n, int et, NodeList **init)
{
	Type *tr;
	Node *r, *on;
	NodeList *args;

	tr = n->type;

	switch(et) {
	default:
		fatal("ifacecvt: unknown op %d\n", et);

	case I2Isame:
	case E2Esame:
		return n;

	case T2I:
		// ifaceT2I(sigi *byte, sigt *byte, elem any) (ret any);
		args = list1(typename(tl));	// sigi
		args = list(args, typename(tr));	// sigt
		args = list(args, n);	// elem

		on = syslook("ifaceT2I", 1);
		argtype(on, tr);
		argtype(on, tl);
		dowidth(on->type);
		break;

	case I2T:
	case I2T2:
	case I2I:
	case I2Ix:
	case I2I2:
	case E2T:
	case E2T2:
	case E2I:
	case E2I2:
		// iface[IT]2[IT][2](sigt *byte, iface any) (ret any[, ok bool]);
		args = list1(typename(tl));	// sigi or sigt
		args = list(args, n);		// iface

		on = syslook(ifacename[et], 1);
		argtype(on, tr);
		argtype(on, tl);
		break;

	case I2E:
		// TODO(rsc): Should do this in back end, without a call.
		// ifaceI2E(elem any) (ret any);
		args = list1(n);	// elem

		on = syslook("ifaceI2E", 1);
		argtype(on, tr);
		argtype(on, tl);
		break;

	case T2E:
		// TODO(rsc): Should do this in back end for pointer case, without a call.
		// ifaceT2E(sigt *byte, elem any) (ret any);
		args = list1(typename(tr));	// sigt
		args = list(args, n);		// elem

		on = syslook("ifaceT2E", 1);
		argtype(on, tr);
		argtype(on, tl);
		break;
	}

	r = nod(OCALL, on, N);
	r->list = args;
	typecheck(&r, Erv);
	walkexpr(&r, init);
	return r;
}

Node*
convas(Node *n, NodeList **init)
{
	Node *l, *r;
	Type *lt, *rt;
	int et;

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

	if(n->left->op == OINDEXMAP) {
		n = mapop(n, init);
		goto out;
	}

	if(eqtype(lt, rt))
		goto out;

	et = ifaceas(lt, rt, 0);
	if(et != Inone) {
		n->right = ifacecvt(lt, r, et, init);
		goto out;
	}

	if(ascompat(lt, rt))
		goto out;

	badtype(n->op, lt, rt);

out:
	ullmancalc(n);
	return n;
}

int
colasname(Node *n)
{
	// TODO(rsc): can probably simplify
	// once late-binding of names goes in
	switch(n->op) {
	case ONAME:
	case ONONAME:
	case OPACK:
		break;
	case OTYPE:
	case OLITERAL:
		if(n->sym != S)
			break;
		// fallthrough
	default:
		return 0;
	}
	return 1;
}

Node*
old2new(Node *n, Type *t, NodeList **init)
{
	Node *l;

	if(!colasname(n)) {
		yyerror("left side of := must be a name");
		return n;
	}
	if(t != T && t->funarg) {
		yyerror("use of multi func value as single value in :=");
		return n;
	}
	l = newname(n->sym);
	dodclvar(l, t, init);
	return l;
}

static Node*
mixedoldnew(Node *n, Type *t)
{
	n = nod(OXXX, n, N);
	n->type = t;
	return n;
}

static NodeList*
checkmixed(NodeList *nl, NodeList **init)
{
	Node *a, *l;
	NodeList *ll, *n;
	Type *t;
	int ntot, nred;

	// first pass, check if it is a special
	// case of new and old declarations

	ntot = 0;	// number assignments
	nred = 0;	// number redeclarations
	for(ll=nl; ll; ll=ll->next) {
		l = ll->n;
		t = l->type;
		l = l->left;

		if(!colasname(l))
			goto allnew;
		if(l->sym->block == block) {
			if(!eqtype(l->type, t))
				goto allnew;
			nred++;
		}
		ntot++;
	}

	// test for special case
	// a) multi-assignment (ntot>1)
	// b) at least one redeclaration (red>0)
	// c) not all redeclarations (nred!=ntot)
	if(nred == 0 || ntot <= 1 || nred == ntot)
		goto allnew;

	n = nil;
	for(ll=nl; ll; ll=ll->next) {
		l = ll->n;
		t = l->type;
		l = l->left;

		a = l;
		if(l->sym->block != block)
			a = old2new(l, t, init);

		n = list(n, a);
	}
	return n;

allnew:
	// same as original
	n = nil;
	for(ll=nl; ll; ll=ll->next) {
		l = ll->n;
		t = l->type;
		l = l->left;

		a = old2new(l, t, init);
		n = list(n, a);
	}
	return n;
}

Node*
colas(NodeList *ll, NodeList *lr)
{
	Node *l, *r, *a, *nl, *nr;
	Iter savet;
	NodeList *init, *savel, *saver, *n;
	Type *t;
	int cl, cr;

	/* nl is an expression list.
	 * nr is an expression list.
	 * return a newname-list from
	 * types derived from the rhs.
	 */
	cr = count(lr);
	cl = count(ll);
	init = nil;
	n = nil;

	/* check calls early, to give better message for a := f() */
	if(cr == 1) {
		nr = lr->n;
		switch(nr->op) {
		case OCALL:
		case OCALLFUNC:
			if(nr->left->op == ONAME && nr->left->etype != 0)
				break;
			typecheck(&nr->left, Erv | Etype | Ecall);
			walkexpr(&nr->left, &init);
			if(nr->left->op == OTYPE)
				break;
			goto call;
		case OCALLMETH:
		case OCALLINTER:
			typecheck(&nr->left, Erv);
			walkexpr(&nr->left, &init);
		call:
			convlit(&nr->left, types[TFUNC]);
			t = nr->left->type;
			if(t == T)
				goto outl;	// error already printed
			if(t->etype == tptr)
				t = t->type;
			if(t == T || t->etype != TFUNC) {
				yyerror("cannot call %T", t);
				goto outl;
			}
			if(t->outtuple != cl) {
				cr = t->outtuple;
				goto badt;
			}
			// finish call - first half above
			t = structfirst(&savet, getoutarg(t));
			if(t == T)
				goto outl;
			for(savel=ll; savel; savel=savel->next) {
				l = savel->n;
				a = mixedoldnew(l, t->type);
				n = list(n, a);
				t = structnext(&savet);
			}
			n = checkmixed(n, &init);
			goto out;
		}
	}
	if(cl != cr) {
		if(cr == 1) {
			nr = lr->n;
			goto multi;
		}
		goto badt;
	}

	for(savel=ll, saver=lr; savel != nil; savel=savel->next, saver=saver->next) {
		l = savel->n;
		r = saver->n;

		typecheck(&r, Erv);
		defaultlit(&r, T);
		saver->n = r;
		a = mixedoldnew(l, r->type);
		n = list(n, a);
	}
	n = checkmixed(n, &init);
	goto out;

multi:
	typecheck(&nr, Erv);
	lr->n = nr;

	/*
	 * there is a list on the left
	 * and a mono on the right.
	 * go into the right to get
	 * individual types for the left.
	 */
	switch(nr->op) {
	default:
		goto badt;

	case OINDEXMAP:
		// check if rhs is a map index.
		// if so, types are valuetype,bool
		if(cl != 2)
			goto badt;
		walkexpr(&nr->left, &init);
		t = nr->left->type;
		a = mixedoldnew(ll->n, t->type);
		n = list1(a);
		a = mixedoldnew(ll->next->n, types[TBOOL]);
		n = list(n, a);
		n = checkmixed(n, &init);
		break;

	case ODOTTYPE:
		// a,b := i.(T)
		walkdottype(nr, &init);
		if(cl != 2)
			goto badt;
		// a,b = iface
		a = mixedoldnew(ll->n, nr->type);
		n = list1(a);
		a = mixedoldnew(ll->next->n, types[TBOOL]);
		n = list(n, a);
		n = checkmixed(n, &init);
		break;

	case ORECV:
		if(cl != 2)
			goto badt;
		walkexpr(&nr->left, &init);
		t = nr->left->type;
		if(!istype(t, TCHAN))
			goto badt;
		a = mixedoldnew(ll->n, t->type);
		n = list1(a);
		a = mixedoldnew(ll->next->n, types[TBOOL]);
		n = list(n, a);
		n = checkmixed(n, &init);
		break;
	}
	goto out;

badt:
	nl = ll->n;
	if(nl->diag == 0) {
		nl->diag = 1;
		yyerror("assignment count mismatch: %d = %d %#N", cl, cr, lr->n);
	}
outl:
	n = ll;

out:
	// n is the lhs of the assignment.
	// init holds the list of declarations.
	a = nod(OAS2, N, N);
	a->list = n;
	a->rlist = lr;
	a->ninit = init;
	a->colas = 1;
	return a;
}

/*
 * rewrite a range statement
 * k and v are names/new_names
 * m is an array or map
 * local is 0 (meaning =) or 1 (meaning :=)
 */
Node*
dorange(Node *nn)
{
	Node *k, *v, *m;
	Node *n, *hv, *hc, *ha, *hk, *ohk, *on, *r, *a, *as;
	NodeList *init, *args;
	Type *t, *th;
	int local;
	NodeList *nl;

	if(nn->op != ORANGE)
		fatal("dorange not ORANGE");

	nl = nn->list;
	k = nl->n;
	if((nl = nl->next) != nil) {
		v = nl->n;
		nl = nl->next;
	} else
		v = N;
	if(nl != nil)
		yyerror("too many variables in range");

	n = nod(OFOR, N, N);
	init = nil;

	typecheck(&nn->right, Erv);
	m = nn->right;
	local = nn->etype;

	t = m->type;
	if(t == T)
		goto out;
	if(t->etype == TARRAY)
		goto ary;
	if(t->etype == TMAP)
		goto map;
	if(t->etype == TCHAN)
		goto chan;
	if(t->etype == TSTRING)
		goto strng;

	yyerror("range must be over map/array/chan/string");
	goto out;

ary:
	hk = nod(OXXX, N, N);		// hidden key
	tempname(hk, types[TINT]);

	ha = nod(OXXX, N, N);		// hidden array
	tempname(ha, t);

	a = nod(OAS, hk, nodintconst(0));
	init = list(init, a);

	a = nod(OAS, ha, m);
	init = list(init, a);

	n->ntest = nod(OLT, hk, nod(OLEN, ha, N));
	n->nincr = nod(OASOP, hk, nodintconst(1));
	n->nincr->etype = OADD;

	if(local)
		k = old2new(k, hk->type, &init);
	n->nbody = list1(nod(OAS, k, hk));

	if(v != N) {
		if(local)
			v = old2new(v, t->type, &init);
		n->nbody = list(n->nbody,
			nod(OAS, v, nod(OINDEX, ha, hk)) );
	}
	goto out;

map:
	th = typ(TARRAY);
	th->type = ptrto(types[TUINT8]);
	th->bound = (sizeof(struct Hiter) + types[tptr]->width - 1) /
			types[tptr]->width;
	hk = nod(OXXX, N, N);		// hidden iterator
	tempname(hk, th);		// hashmap hash_iter

	on = syslook("mapiterinit", 1);
	argtype(on, t->down);
	argtype(on, t->type);
	argtype(on, th);
	a = nod(OADDR, hk, N);
	r = nod(OCALL, on, N);
	r->list = list(list1(m), a);

	init = list(init, r);

	r = nod(OINDEX, hk, nodintconst(0));
	a = nod(OLITERAL, N, N);
	a->val.ctype = CTNIL;
	a->type = types[TNIL];
	r = nod(ONE, r, a);
	n->ntest = r;

	on = syslook("mapiternext", 1);
	argtype(on, th);
	r = nod(OADDR, hk, N);
	args = list1(r);
	r = nod(OCALL, on, N);
	r->list = args;
	n->nincr = r;

	if(local)
		k = old2new(k, t->down, &init);
	if(v == N) {
		on = syslook("mapiter1", 1);
		argtype(on, th);
		argtype(on, t->down);
		r = nod(OADDR, hk, N);
		args = list1(r);
		r = nod(OCALL, on, N);
		r->list = args;
		n->nbody = list1(nod(OAS, k, r));
		goto out;
	}
	if(local)
		v = old2new(v, t->type, &init);
	on = syslook("mapiter2", 1);
	argtype(on, th);
	argtype(on, t->down);
	argtype(on, t->type);
	r = nod(OADDR, hk, N);
	args = list1(r);
	r = nod(OCALL, on, N);
	r->list = args;
	as = nod(OAS2, N, N);
	as->list = list(list1(k), v);
	as->rlist = list1(r);
	n->nbody = list1(as);
	goto out;

chan:
	if(v != N)
		yyerror("chan range can only have one variable");

	hc = nod(OXXX, N, N);	// hidden chan
	tempname(hc, t);

	hv = nod(OXXX, N, N);	// hidden value
	tempname(hv, t->type);

	a = nod(OAS, hc, m);
	init = list(init, a);

	a = nod(ORECV, hc, N);
	a = nod(OAS, hv, a);
	init = list(init, a);

	a = nod(OCLOSED, N, N);
	a->list = list1(hc);
	n->ntest = nod(ONOT, a, N);
	n->nincr = nod(OAS, hv, nod(ORECV, hc, N));

	if(local)
		k = old2new(k, hv->type, &init);
	n->nbody = list1(nod(OAS, k, hv));

	goto out;

strng:
	hk = nod(OXXX, N, N);		// hidden key
	tempname(hk, types[TINT]);

	ohk = nod(OXXX, N, N);		// old hidden key
	tempname(ohk, types[TINT]);

	ha = nod(OXXX, N, N);		// hidden string
	tempname(ha, types[TSTRING]);

	hv = N;
	if(v != N) {
		hv = nod(OXXX, N, N);		// hidden value
		tempname(hv, types[TINT]);
	}

	if(local) {
		k = old2new(k, types[TINT], &init);
		if(v != N)
			v = old2new(v, types[TINT], &init);
	}

	// ha = s
	a = nod(OCONV, m, N);
	a->type = ha->type;
	a = nod(OAS, ha, a);
	init = list(init, a);

	// ohk = 0
	a = nod(OAS, ohk, nodintconst(0));
	init = list(init, a);

	// hk[,hv] = stringiter(ha,hk)
	if(v != N) {
		// hk,v = stringiter2(ha, hk)
		on = syslook("stringiter2", 0);
		a = nod(OCALL, on, N);
		a->list = list(list1(ha), nodintconst(0));
		as = nod(OAS2, N, N);
		as->list = list(list1(hk), hv);
		as->rlist = list1(a);
		a = as;
	} else {
		// hk = stringiter(ha, hk)
		on = syslook("stringiter", 0);
		a = nod(OCALL, on, N);
		a->list = list(list1(ha), nodintconst(0));
		a = nod(OAS, hk, a);
	}
	init = list(init, a);

	// while(hk != 0)
	n->ntest = nod(ONE, hk, nodintconst(0));

	// hk[,hv] = stringiter(ha,hk)
	if(v != N) {
		// hk,hv = stringiter2(ha, hk)
		on = syslook("stringiter2", 0);
		a = nod(OCALL, on, N);
		a->list = list(list1(ha), hk);
		as = nod(OAS2, N, N);
		as->list = list(list1(hk), hv);
		as->rlist = list1(a);
		a = as;
	} else {
		// hk = stringiter(ha, hk)
		on = syslook("stringiter", 0);
		a = nod(OCALL, on, N);
		a->list = list(list1(ha), hk);
		a = nod(OAS, hk, a);
	}
	n->nincr = a;

	// k,ohk[,v] = ohk,hk,[,hv]
	a = nod(OAS, k, ohk);
	n->nbody = list1(a);
	a = nod(OAS, ohk, hk);
	n->nbody = list(n->nbody, a);
	if(v != N) {
		a = nod(OAS, v, hv);
		n->nbody = list(n->nbody, a);
	}

out:
	n->ninit = concat(n->ninit, init);
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
	int c, t;

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
	f = N;	// one fncall assigned to stack
	r = nil;	// non fncalls and tempnames assigned to stack

	for(l=all; l; l=l->next) {
		n = l->n;
		ullmancalc(n);
		if(n->ullman < UINF) {
			r = list(r, n);
			continue;
		}
		if(f == N) {
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
int
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
	if(l == N)
		return 0;
	switch(l->op) {
	case ONAME:
		// match each left with all rights
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

NodeList*
reorder4(NodeList *ll)
{
	/*
	 * from ascompat[te]
	 *	return c,d
	 * return expression assigned to output
	 * parameters. there may be no problems.
	 *
	 * TODO(rsc): i don't believe that.
	 *	func f() (a, b int) {
	 *		a, b = 1, 2;
	 *		return b, a;
	 *	}
	 */
	return ll;
}

Node*
structlit(Node *n, Node *var, NodeList **init)
{
	Type *t;
	Node *r, *a;
	NodeList *nl;

	t = n->type;
	if(t->etype != TSTRUCT)
		fatal("structlit: not struct");

	if(var == N) {
		var = nod(OXXX, N, N);
		tempname(var, t);
	}

	nl = n->list;

	if(count(n->list) < structcount(t)) {
		a = nod(OAS, var, N);
		typecheck(&a, Etop);
		walkexpr(&a, init);
		*init = list(*init, a);
	}

	for(; nl; nl=nl->next) {
		r = nl->n;

		// build list of var.field = expr
		a = nod(ODOT, var, newname(r->left->sym));
		a = nod(OAS, a, r->right);
		typecheck(&a, Etop);
		walkexpr(&a, init);
		*init = list(*init, a);
	}
	return var;
}

Node*
arraylit(Node *n, Node *var, NodeList **init)
{
	Type *t;
	Node *r, *a;
	NodeList *l;

	t = n->type;

	if(var == N) {
		var = nod(OXXX, N, N);
		tempname(var, t);
	}

	if(t->bound < 0) {
		// slice
		a = nod(OMAKE, N, N);
		a->list = list(list1(typenod(t)), n->right);
		a = nod(OAS, var, a);
		typecheck(&a, Etop);
		walkexpr(&a, init);
		*init = list(*init, a);
	} else {
		// if entire array isnt initialized,
		// then clear the array
		if(count(n->list) < t->bound) {
			a = nod(OAS, var, N);
			typecheck(&a, Etop);
			walkexpr(&a, init);
			*init = list(*init, a);
		}
	}

	for(l=n->list; l; l=l->next) {
		r = l->n;
		// build list of var[c] = expr
		a = nod(OINDEX, var, r->left);
		a = nod(OAS, a, r->right);
		typecheck(&a, Etop);
		walkexpr(&a, init);	// add any assignments in r to top
		*init = list(*init, a);
	}

	return var;
}

Node*
maplit(Node *n, Node *var, NodeList **init)
{
	Type *t;
	Node *r, *a;
	Node* hash[101];
	NodeList *l;
	int nerr;

	nerr = nerrors;
	t = n->type;
	if(t->etype != TMAP)
		fatal("maplit: not map");

	if(var == N) {
		var = nod(OXXX, N, N);
		tempname(var, t);
	}

	a = nod(OMAKE, N, N);
	a->list = list1(typenod(t));
	a = nod(OAS, var, a);
	typecheck(&a, Etop);
	walkexpr(&a, init);
	*init = list(*init, a);

	memset(hash, 0, sizeof(hash));
	for(l=n->list; l; l=l->next) {
		r = l->n;
		// build list of var[c] = expr
		a = nod(OINDEX, var, r->left);
		a = nod(OAS, a, r->right);
		typecheck(&a, Etop);
		walkexpr(&a, init);
		if(nerr != nerrors)
			break;

		*init = list(*init, a);
	}
	return var;
}

/*
 * walk through argin parameters.
 * generate and return code to allocate
 * copies of escaped parameters to the heap.
 */
NodeList*
paramstoheap(Type **argin)
{
	Type *t;
	Iter savet;
	Node *v;
	NodeList *nn;

	nn = nil;
	for(t = structfirst(&savet, argin); t != T; t = structnext(&savet)) {
		v = t->nname;
		if(v == N || !(v->class & PHEAP))
			continue;

		// generate allocation & copying code
		nn = list(nn, nod(OAS, v->heapaddr, v->alloc));
		nn = list(nn, nod(OAS, v, v->stackparam));
	}
	return nn;
}

/*
 * take care of migrating any function in/out args
 * between the stack and the heap.  adds code to
 * curfn's before and after lists.
 */
void
heapmoves(void)
{
	NodeList *nn;

	nn = paramstoheap(getthis(curfn->type));
	nn = concat(nn, paramstoheap(getinarg(curfn->type)));
	curfn->enter = concat(curfn->enter, nn);
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
		typecheck(&r, Erv);
	else
		typecheck(&r, Etop);
	walkexpr(&r, init);
	r->type = t;
	return r;
}

static Node*
mkcall(char *name, Type *t, NodeList **init, ...)
{
	Node *r;
	va_list va;

	va_start(va, init);
	r = vmkcall(syslook(name, 0), t, init, va);
	va_end(va);
	return r;
}

static Node*
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

static Node*
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
