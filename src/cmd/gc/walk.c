// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"

static	Type*	sw1(Node*, Type*);
static	Type*	sw2(Node*, Type*);
static	Type*	sw3(Node*, Type*);
static	Node*	curfn;

void
walk(Node *fn)
{
	if(debug['W'])
		dump("fn-before", fn->nbody);
	curfn = fn;
	walktype(fn->nbody, Etop);
	if(debug['W'])
		dump("fn", fn->nbody);
}

void
walktype(Node *n, int top)
{
	Node *r, *l;
	Type *t;
	Sym *s;
	long lno;
	int et;

	/*
	 * walk the whole tree of the body of a function.
	 * the types expressions are calculated.
	 * compile-time constants are evaluated.
	 */

	lno = dynlineno;
	if(top == Exxx || top == Eyyy) {
		dump("", n);
		fatal("walktype: top=%d", top);
	}

loop:
	if(n == N)
		goto ret;
	if(n->op != ONAME)
		dynlineno = n->lineno;	// for diagnostics

	if(debug['w'] > 1 && top == Etop && n->op != OLIST)
		dump("walk-before", n);

	t = T;
	et = Txxx;

	switch(n->op) {
	default:
		fatal("walktype: switch 1 unknown op %N", n);
		goto ret;

	case OPRINT:
		if(top != Etop)
			goto nottop;
		walktype(n->left, Erv);
		*n = *prcompat(n->left);
		goto ret;

	case OPANIC:
		if(top != Etop)
			goto nottop;
		walktype(n->left, Erv);
		*n = *nod(OLIST, prcompat(n->left), nodpanic(n->lineno));
		goto ret;

	case OLITERAL:
		if(top != Erv)
			goto nottop;
		n->addable = 1;
		ullmancalc(n);
		goto ret;

	case ONAME:
		if(top == Etop)
			goto nottop;
		n->addable = 1;
		ullmancalc(n);
		if(n->type == T) {
			s = n->sym;
			if(s->undef == 0) {
				yyerror("walktype: %N undeclared", n);
				s->undef = 1;
			}
		}
		goto ret;

	case OLIST:
		walktype(n->left, top);
		n = n->right;
		goto loop;

	case OFOR:
		if(top != Etop)
			goto nottop;
		walktype(n->ninit, Etop);
		walktype(n->ntest, Erv);
		walktype(n->nincr, Etop);
		n = n->nbody;
		goto loop;

	case OSWITCH:
		if(top != Etop)
			goto nottop;

		if(n->ntest == N)
			n->ntest = booltrue;
		walktype(n->ninit, Etop);
		walktype(n->ntest, Erv);
		walktype(n->nbody, Etop);

		// find common type
		if(n->ntest->type == T)
			n->ntest->type = walkswitch(n, sw1);

		// if that fails pick a type
		if(n->ntest->type == T)
			n->ntest->type = walkswitch(n, sw2);

		// set the type on all literals
		if(n->ntest->type != T)
			walkswitch(n, sw3);
		walktype(n->ntest, Erv);	// BOTCH is this right
		walktype(n->nincr, Erv);
		goto ret;

	case OEMPTY:
		if(top != Etop)
			goto nottop;
		goto ret;

	case OIF:
		if(top != Etop)
			goto nottop;
		walktype(n->ninit, Etop);
		walktype(n->ntest, Erv);
		walktype(n->nelse, Etop);
		n = n->nbody;
		goto loop;

	case OCALLMETH:
	case OCALLINTER:
	case OCALL:
		if(top == Elv)
			goto nottop;

		n->ullman = UINF;
		if(n->type != T)
			goto ret;

		walktype(n->left, Erv);
		if(n->left == N)
			goto ret;

		t = n->left->type;
		if(t == T)
			goto ret;

		dowidth(t);
		if(n->left->op == ODOTMETH)
			n->op = OCALLMETH;
		if(n->left->op == ODOTINTER)
			n->op = OCALLINTER;

		if(isptr[t->etype])
			t = t->type;

		if(t->etype != TFUNC) {
			yyerror("call of a non-function %T", t);
			goto ret;
		}

		n->type = *getoutarg(t);
		if(t->outtuple == 1)
			n->type = n->type->type->type;

		walktype(n->right, Erv);

		switch(n->op) {
		default:
			fatal("walk: op: %O", n->op);

		case OCALLINTER:
			l = ascompatte(n->op, getinarg(t), &n->right, 0);
			n->right = reorder1(l);
			break;

		case OCALL:
			l = ascompatte(n->op, getinarg(t), &n->right, 0);
			n->right = reorder1(l);
			break;

		case OCALLMETH:
			// add this-pointer to the arg list
			l = ascompatte(n->op, getinarg(t), &n->right, 0);
			r = ascompatte(n->op, getthis(t), &n->left->left, 0);
			if(l != N)
				r = nod(OLIST, r, l);
			n->right = reorder1(r);
			break;
		}
		goto ret;

	case OAS:
		if(top != Etop)
			goto nottop;

		l = n->left;
		r = n->right;
		if(l == N)
			goto ret;

		walktype(l, Elv);
		walktype(r, Erv);
		if(l == N || l->type == T)
			goto ret;

		convlit(r, l->type);
		if(r == N || r->type == T)
			goto ret;

		if(r->op == OCALL && l->op == OLIST) {
			l = ascompatet(n->op, &n->left, &r->type, 0);
			if(l != N) {
				*n = *nod(OLIST, r, reorder2(l));
			}
			goto ret;
		}

		l = ascompatee(n->op, &n->left, &n->right);
		if(l != N)
			*n = *reorder3(l);
		goto ret;

	case OBREAK:
	case OCONTINUE:
	case OGOTO:
	case OLABEL:
		if(top != Etop)
			goto nottop;
		goto ret;

	case OXCASE:
		if(top != Etop)
			goto nottop;
		yyerror("case statement out of place");
		n->op = OCASE;

	case OCASE:
		if(top != Etop)
			goto nottop;
		walktype(n->left, Erv);
		n = n->right;
		goto loop;

	case OXFALL:
		if(top != Etop)
			goto nottop;
		yyerror("fallthrough statement out of place");
		n->op = OFALL;

	case OFALL:
	case OINDREG:
		goto ret;

	case OS2I:
	case OI2S:
	case OI2I:
		if(top != Erv)
			goto nottop;
		n->addable = 0;
		walktype(n->left, Erv);
		goto ret;

	case OCONV:
		if(top != Erv)
			goto nottop;
		walktype(n->left, Erv);
		if(n->left == N)
			goto ret;

		convlit(n->left, n->type);

		// nil conversion
		if(eqtype(n->type, n->left->type, 0)) {
			if(n->left->op != ONAME)
				*n = *n->left;
			goto ret;
		}

		// simple fix-float
		if(n->left->type != T)
		if(isint[n->left->type->etype] || isfloat[n->left->type->etype])
		if(isint[n->type->etype] || isfloat[n->type->etype]) {
			evconst(n);
			goto ret;
		}

		// to string
		if(isptrto(n->type, TSTRING)) {
			if(isint[n->left->type->etype]) {
				*n = *stringop(n, top);
				goto ret;
			}
			if(isbytearray(n->left->type) != 0) {
				n->op = OARRAY;
				*n = *stringop(n, top);
				goto ret;
			}
		}

		if(n->type->etype == TARRAY) {
			arrayconv(n->type, n->left);
			goto ret;
		}

		badtype(n->op, n->left->type, n->type);
		goto ret;

	case ORETURN:
		if(top != Etop)
			goto nottop;
		walktype(n->left, Erv);
		l = ascompatte(n->op, getoutarg(curfn->type), &n->left, 1);
		if(l != N)
			n->left = reorder4(l);
		goto ret;

	case ONOT:
		if(top != Erv)
			goto nottop;
		walktype(n->left, Erv);
		if(n->left == N || n->left->type == T)
			goto ret;
		et = n->left->type->etype;
		break;

	case OASOP:
		if(top != Etop)
			goto nottop;
		walktype(n->left, Elv);
		goto com;

	case OLSH:
	case ORSH:
	case OMOD:
	case OAND:
	case OOR:
	case OXOR:
	case OANDAND:
	case OOROR:
	case OEQ:
	case ONE:
	case OLT:
	case OLE:
	case OGE:
	case OGT:
	case OADD:
	case OSUB:
	case OMUL:
	case ODIV:
		if(top != Erv)
			goto nottop;
		walktype(n->left, Erv);

	com:
		walktype(n->right, Erv);
		if(n->left == N || n->right == N)
			goto ret;
		convlit(n->left, n->right->type);
		convlit(n->right, n->left->type);
		evconst(n);
		if(n->op == OLITERAL)
			goto ret;
		if(n->left->type == T || n->right->type == T)
			goto ret;
		if(!ascompat(n->left->type, n->right->type))
			goto badt;

		switch(n->op) {
		case OEQ:
		case ONE:
		case OLT:
		case OLE:
		case OGE:
		case OGT:
		case OADD:
		case OASOP:
			if(isptrto(n->left->type, TSTRING)) {
				*n = *stringop(n, top);
				goto ret;
			}
		}
		break;

	case OMINUS:
	case OPLUS:
	case OCOM:
		if(top != Erv)
			goto nottop;
		walktype(n->left, Erv);
		if(n->left == N)
			goto ret;
		evconst(n);
		ullmancalc(n);
		if(n->op == OLITERAL)
			goto ret;
		break;

	case OLEN:
		if(top != Erv)
			goto nottop;
		walktype(n->left, Erv);
		evconst(n);
		t = n->left->type;
		if(t != T && isptr[t->etype])
			t = t->type;
		if(t == T)
			goto ret;
		switch(t->etype) {
		default:
			goto badt;
		case TSTRING:
			break;
		}
		n->type = types[TINT32];
		goto ret;

	case OINDEX:
	case OINDEXPTR:
		if(top == Etop)
			goto nottop;

		walktype(n->left, top);
		walktype(n->right, Erv);

		if(n->left == N || n->right == N)
			goto ret;

		defaultlit(n->left);
		t = n->left->type;
		if(t == T)
			goto ret;

		// left side is indirect
		if(isptr[t->etype]) {
			t = t->type;
			n->op = OINDEXPTR;
		}

		switch(t->etype) {
		default:
			goto badt;

		case TMAP:

print("top=%d type %lT", top, t);
dump("index", n);
			// right side must map type
			if(n->right->type == T) {
				convlit(n->right, t->down);
				if(n->right->type == T)
					break;
			}
			if(!eqtype(n->right->type, t->down, 0))
				goto badt;
			if(n->op != OINDEXPTR)
				goto badt;
			n->op = OINDEX;
			n->type = t->type;
			if(top == Erv)
*n = *mapop(n, top);
			break;

		case TSTRING:
			// right side must be an int
			if(top != Erv)
				goto nottop;
			if(n->right->type == T) {
				convlit(n->right, types[TINT32]);
				if(n->right->type == T)
					break;
			}
			if(!isint[n->right->type->etype])
				goto badt;
			*n = *stringop(n, top);
			break;
			
		case TARRAY:
		case TDARRAY:
			// right side must be an int
			if(n->right->type == T) {
				convlit(n->right, types[TINT32]);
				if(n->right->type == T)
					break;
			}
			if(!isint[n->right->type->etype])
				goto badt;

			n->type = t->type;
			break;
		}
		goto ret;

	case OSLICE:
		if(top == Etop)
			goto nottop;

		walktype(n->left, top);
		walktype(n->right, Erv);
		if(n->left == N || n->right == N)
			goto ret;
		if(isptrto(n->left->type, TSTRING)) {
			*n = *stringop(n, top);
			goto ret;
		}
		badtype(OSLICE, n->left->type, T);
		goto ret;

	case ODOT:
	case ODOTPTR:
	case ODOTMETH:
	case ODOTINTER:
		if(top == Etop)
			goto nottop;
		walkdot(n, top);
		goto ret;

	case OADDR:
		if(top != Erv)
			goto nottop;
		walktype(n->left, Elv);
		if(n->left == N)
			goto ret;
		t = n->left->type;
		if(t == T)
			goto ret;
		n->type = ptrto(t);
		goto ret;

	case OIND:
		if(top == Etop)
			goto nottop;
		walktype(n->left, top);
		if(n->left == N)
			goto ret;
		t = n->left->type;
		if(t == T)
			goto ret;
		if(!isptr[t->etype])
			goto badt;
		n->type = t->type;
		goto ret;

	case ONEW:
		if(top != Erv)
			goto nottop;
		*n = *newcompat(n);
		goto ret;
	}

/*
 * ======== second switch ========
 */

	switch(n->op) {
	default:
		fatal("walktype: switch 2 unknown op %N", n);
		goto ret;

	case OASOP:
		break;

	case ONOT:
	case OANDAND:
	case OOROR:
		et = n->left->type->etype;
		if(et != TBOOL)
			goto badt;
		t = types[TBOOL];
		break;

	case OEQ:
	case ONE:
		et = n->left->type->etype;
		if(!okforeq[et])
			goto badt;
		t = types[TBOOL];
		break;

	case OLT:
	case OLE:
	case OGE:
	case OGT:
		et = n->left->type->etype;
		if(!okforadd[et])
			if(!isptrto(n->left->type, TSTRING))
				goto badt;
		t = types[TBOOL];
		break;

	case OADD:
	case OSUB:
	case OMUL:
	case ODIV:
	case OPLUS:
		et = n->left->type->etype;
		if(!okforadd[et])
			goto badt;
		break;

	case OMINUS:
		et = n->left->type->etype;
		if(!okforadd[et])
			goto badt;
		if(!isfloat[et])
			break;

		l = nod(OLITERAL, N, N);
		l->val.ctype = CTFLT;
		l->val.dval = 0;

		l = nod(OSUB, l, n->left);
		*n = *l;
		walktype(n, Erv);
		goto ret;

	case OLSH:
	case ORSH:
	case OAND:
	case OOR:
	case OXOR:
	case OMOD:
	case OCOM:
		et = n->left->type->etype;
		if(!okforand[et])
			goto badt;
		break;
	}

	if(t == T)
		t = n->left->type;
	n->type = t;
	goto ret;

nottop:
	dump("bad top", n);
	fatal("walktype: top=%d %O", top, n->op);
	goto ret;

badt:
	if(n->right == N) {
		if(n->left == N) {
			badtype(n->op, T, T);
			goto ret;
		}
		badtype(n->op, n->left->type, T);
		goto ret;
	}
	badtype(n->op, n->left->type, n->right->type);
	goto ret;

ret:
	if(debug['w'] && top == Etop && n != N)
		dump("walk", n);

	ullmancalc(n);
	dynlineno = lno;
}

/*
 * return the first type
 */
Type*
sw1(Node *c, Type *place)
{
	if(place == T)
		return c->type;
	return place;
}

/*
 * return a suitable type
 */
Type*
sw2(Node *c, Type *place)
{
	return types[TINT32];	// botch
}

/*
 * check that selected type
 * is compat with all the cases
 */
Type*
sw3(Node *c, Type *place)
{
	if(place == T)
		return c->type;
	if(c->type == T)
		c->type = place;
	convlit(c, place);
	if(!ascompat(place, c->type))
		badtype(OSWITCH, place, c->type);
	return place;
}

Type*
walkswitch(Node *sw, Type*(*call)(Node*, Type*))
{
	Node *n, *c;
	Type *place;
	place = call(sw->ntest, T);

	n = sw->nbody;
	if(n->op == OLIST)
		n = n->left;
	if(n->op == OEMPTY)
		return;

	for(; n!=N; n=n->right) {
		if(n->op != OCASE)
			fatal("walkswitch: not case %O\n", n->op);
		for(c=n->left; c!=N; c=c->right) {
			if(c->op != OLIST) {
				place = call(c, place);
				break;
			}
			place = call(c->left, place);
		}
	}
	return place;
}

int
casebody(Node *n)
{
	Node *oc, *ot, *t;
	Iter save;


	/*
	 * look to see if statements at top level have
	 * case labels attached to them. convert the illegal
	 * ops XFALL and XCASE into legal ops FALL and CASE.
	 * all unconverted ops will thus be caught as illegal
	 */

	oc = N;		// last case statement
	ot = N;		// last statement (look for XFALL)
	t = listfirst(&save, &n);

loop:
	if(t == N) {
		/* empty switch */
		if(oc == N)
			return 0;
		return 1;
	}
	if(t->op == OXCASE) {
		/* rewrite and link top level cases */
		t->op = OCASE;
		if(oc != N)
			oc->right = t;
		oc = t;

		/* rewrite top fall that preceed case */
		if(ot != N && ot->op == OXFALL)
			ot->op = OFALL;
	}

	/* if first statement is not case */
	if(oc == N)
		return 0;

	ot = t;
	t = listnext(&save);
	goto loop;
}

/*
 * allowable type combinations for
 * normal binary operations.
 */
Type*
lookdot(Node *n, Type *t, int d)
{
	Type *f, *r, *c;
	Sym *s;

	r = T;
	s = n->sym;
	if(d > 0)
		goto deep;

	for(f=t->type; f!=T; f=f->down) {
		if(f->sym == S)
			continue;
		if(f->sym != s)
			continue;
		if(r != T) {
			yyerror("ambiguous DOT reference %s", s->name);
			break;
		}
		r = f;
	}
	return r;

deep:
	/* deeper look after shallow failed */
	for(f=t->type; f!=T; f=f->down) {
		// only look at unnamed sub-structures
		// BOTCH no such thing -- all are assigned temp names
		if(f->sym != S)
			continue;
		c = f->type;
		if(c->etype != TSTRUCT)
			continue;
		c = lookdot(n, c, d-1);
		if(c == T)
			continue;
		if(r != T) {
			yyerror("ambiguous unnamed DOT reference %s", s->name);
			break;
		}
		r = c;
	}
	return r;
}

void
walkdot(Node *n, int top)
{
	Node *mn;
	Type *t, *f;
	int i;

if(debug['T'])
print("%L walkdot %O %d\n", n->op, top);

	if(n->left == N || n->right == N)
		return;

	walktype(n->left, Erv);
	if(n->right->op != ONAME) {
		yyerror("rhs of . must be a name");
		return;
	}

	t = n->left->type;
	if(t == T)
		return;

	if(isptr[t->etype]) {
		t = t->type;
		if(t == T)
			return;
		n->op = ODOTPTR;
	}

	if(n->right->op != ONAME)
		fatal("walkdot: not name %O", n->right->op);

	switch(t->etype) {
	default:
		badtype(ODOT, t, T);
		return;

	case TSTRUCT:
	case TINTER:
		for(i=0; i<5; i++) {
			f = lookdot(n->right, t, i);
			if(f != T)
				break;
		}

		// look up the field as TYPE_name
		// for a mothod. botch this should
		// be done better.
		if(f == T && t->etype == TSTRUCT) {
			mn = methodname(n->right, t);
			for(i=0; i<5; i++) {
				f = lookdot(mn, t, i);
				if(f != T)
					break;
			}
		}

		if(f == T) {
			yyerror("undefined DOT reference %N", n->right);
			break;
		}

		n->xoffset = f->width;
		n->right = f->nname;		// substitute real name
		n->type = f->type;
		if(n->type->etype == TFUNC) {
			n->op = ODOTMETH;
			if(t->etype == TINTER) {
				n->op = ODOTINTER;
			}
		}
		break;
	}
}


Node*
ascompatee(int op, Node **nl, Node **nr)
{
	Node *l, *r, *nn, *a;
	Iter savel, saver;

	/*
	 * check assign expression list to
	 * a expression list. called in
	 *	expr-list = expr-list
	 */
	l = listfirst(&savel, nl);
	r = listfirst(&saver, nr);
	nn = N;
	

loop:
	if(l == N || r == N) {
		if(l != r)
			yyerror("error in shape across assignment");
		return rev(nn);
	}

	convlit(r, l->type);
	if(!ascompat(l->type, r->type)) {
		badtype(op, l->type, r->type);
		return N;
	}

	a = nod(OAS, l, r);
	a = convas(a);
	if(nn == N)
		nn = a;
	else
		nn = nod(OLIST, a, nn);

	l = listnext(&savel);
	r = listnext(&saver);
	goto loop;
}

Node*
ascompatet(int op, Node **nl, Type **nr, int fp)
{
	Node *l, *nn, *a;
	Type *r;
	Iter savel, saver;

	/*
	 * check assign type list to
	 * a expression list. called in
	 *	expr-list = func()
	 */
	l = listfirst(&savel, nl);
	r = structfirst(&saver, nr);
	nn = N;

loop:
	if(l == N || r == T) {
		if(l != N || r != T)
			yyerror("error in shape across assignment");
		return rev(nn);
	}

	if(!ascompat(l->type, r->type)) {
		badtype(op, l->type, r->type);
		return N;
	}

	a = nod(OAS, l, nodarg(r, fp));
	a = convas(a);
	if(nn == N)
		nn = a;
	else
		nn = nod(OLIST, a, nn);

	l = listnext(&savel);
	r = structnext(&saver);

	goto loop;
}

Node*
ascompatte(int op, Type **nl, Node **nr, int fp)
{
	Type *l;
	Node *r, *nn, *a;
	Iter savel, saver;

	/*
	 * check assign expression list to
	 * a type list. called in
	 *	return expr-list
	 *	func(expr-list)
	 */
	l = structfirst(&savel, nl);
	r = listfirst(&saver, nr);
	nn = N;

loop:
	if(l == T || r == N) {
		if(l != T || r != N)
			yyerror("error in shape across assignment");
		return rev(nn);
	}
	convlit(r, l->type);
	if(!ascompat(l->type, r->type)) {
		badtype(op, l->type, r->type);
		return N;
	}

	a = nod(OAS, nodarg(l, fp), r);
	a = convas(a);
	if(nn == N)
		nn = a;
	else
		nn = nod(OLIST, a, nn);

	l = structnext(&savel);
	r = listnext(&saver);

	goto loop;
}

/*
 * can we assign var of type t2 to var of type t1
 */
int
ascompat(Type *t1, Type *t2)
{
	if(eqtype(t1, t2, 0))
		return 1;

//	if(eqtype(t1, nilptr, 0))
//		return 1;
//	if(eqtype(t2, nilptr, 0))
//		return 1;

	if(isinter(t1))
		if(isptrto(t2, TSTRUCT) || isinter(t2))
			return 1;

	if(isinter(t2))
		if(isptrto(t1, TSTRUCT))
			return 1;

	return 0;
}

Node*
prcompat(Node *n)
{
	Node *l, *r;
	Type *t;
	Iter save;
	int w;
	char *name;
	Node *on;

	r = N;
	l = listfirst(&save, &n);

loop:
	if(l == N) {
		walktype(r, Etop);
		return r;
	}

	w = whatis(l);
	switch(w) {
	default:
		badtype(n->op, l->type, T);
		l = listnext(&save);
		goto loop;
	case Wlitint:
	case Wtint:
		name = "printint";
		break;
	case Wlitfloat:
	case Wtfloat:
		name = "printfloat";
		break;
	case Wlitbool:
	case Wtbool:
		name = "printbool";
		break;
	case Wlitstr:
	case Wtstr:
		name = "printstring";
		break;
	}

	on = syslook(name, 0);
	t = *getinarg(on->type);
	if(t != nil)
		t = t->type;
	if(t != nil)
		t = t->type;

	if(!eqtype(t, l->type, 0)) {
		l = nod(OCONV, l, N);
		l->type = t;
	}

	if(r == N)
		r = nod(OCALL, on, l);
	else
		r = nod(OLIST, r, nod(OCALL, on, l));

	l = listnext(&save);
	goto loop;
}

Node*
nodpanic(long lineno)
{
	Node *n, *on;

	on = syslook("panicl", 0);
	n = nodintconst(lineno);
	n = nod(OCALL, on, n);
	walktype(n, Etop);
	return n;
}

Node*
newcompat(Node *n)
{
	Node *r, *on;
	Type *t;

	t = n->type;
	if(t == T || !isptr[t->etype] || t->type == T)
		fatal("newcompat: type should be pointer %lT", t);

	t = t->type;
	if(t->etype == TMAP) {
		r = mapop(n, Erv);
		return r;
	}

	if(n->left != N)
		yyerror("dont know what new(,e) means");

	dowidth(t);

	on = syslook("mal", 1);

	argtype(on, t);

	r = nodintconst(t->width);
	r = nod(OCALL, on, r);
	walktype(r, Erv);

//	r = nod(OCONV, r, N);
	r->type = n->type;

	return r;
}

Node*
stringop(Node *n, int top)
{
	Node *r, *c, *on;
	long lno, l;

	lno = dynlineno;
	dynlineno = n->lineno;

	switch(n->op) {
	default:
		fatal("stringop: unknown op %E", n->op);

	case OEQ:
	case ONE:
	case OGE:
	case OGT:
	case OLE:
	case OLT:
		// sys_cmpstring(s1, s2) :: 0
		on = syslook("cmpstring", 0);
		r = nod(OLIST, n->left, n->right);
		r = nod(OCALL, on, r);
		c = nodintconst(0);
		r = nod(n->op, r, c);
		break;

	case OADD:
		// sys_catstring(s1, s2)
		on = syslook("catstring", 0);
		r = nod(OLIST, n->left, n->right);
		r = nod(OCALL, on, r);
		break;

	case OASOP:
		// sys_catstring(s1, s2)
		switch(n->etype) {
		default:
			fatal("stringop: unknown op %E-%E", n->op, n->etype);

		case OADD:
			// s1 = sys_catstring(s1, s2)
			if(n->etype != OADD)
				fatal("stringop: not cat");
			r = nod(OLIST, n->left, n->right);
			on = syslook("catstring", 0);
			r = nod(OCALL, on, r);
			r = nod(OAS, n->left, r);
			break;
		}
		break;

	case OSLICE:
		// sys_slicestring(s, lb, hb)
		r = nod(OCONV, n->right->left, N);
		r->type = types[TINT32];

		c = nod(OCONV, n->right->right, N);
		c->type = types[TINT32];

		r = nod(OLIST, r, c);
		r = nod(OLIST, n->left, r);
		on = syslook("slicestring", 0);
		r = nod(OCALL, on, r);
		break;

	case OINDEXPTR:
		// sys_indexstring(s, i)
		r = nod(OCONV, n->right, N);
		r->type = types[TINT32];
		r = nod(OLIST, n->left, r);
		on = syslook("indexstring", 0);
		r = nod(OCALL, on, r);
		break;

	case OCONV:
		// sys_intstring(v)
		r = nod(OCONV, n->left, N);
		r->type = types[TINT64];
		on = syslook("intstring", 0);
		r = nod(OCALL, on, r);
		break;

	case OARRAY:
		// byteastring(a, l)
		c = nodintconst(0);
		r = nod(OINDEX, n->left, c);
		r = nod(OADDR, r, N);

		l = isbytearray(n->left->type);
		c = nodintconst(l-1);

		r = nod(OLIST, r, c);
		on = syslook("byteastring", 0);
		r = nod(OCALL, on, r);
		break;
	}

	walktype(r, top);
	dynlineno = lno;
	return r;
}

Type*
fixmap(Type *tm)
{
	Type *t;

	t = tm->type;
	if(t == T) {
		fatal("fixmap: t nil");
		return T;
	}

	if(t->etype != TMAP) {
		fatal("fixmap: %O not map");
		return T;
	}

	if(t->down == T || t->type == T) {
		fatal("fixmap: map key/value types are nil");
		return T;
	}

	dowidth(t->down);
	dowidth(t->type);

	return t;
}

static int
algtype(Type *t)
{
	int a;

	a = 100;
	if(issimple[t->etype])
		a = 0;		// simple mem
	else
	if(isptrto(t, TSTRING))
		a = 1;		// string
	else
	if(isptr[t->etype])
		a = 2;		// pointer
	else
	if(isinter(t))
		a = 3;		// interface
	else
		fatal("algtype: cant find type %T", t);
	return a;
}

Node*
mapop(Node *n, int top)
{
	long lno;
	Node *r, *a;
	Type *t;
	Node *on;
	int alg1, alg2;

	lno = dynlineno;
	dynlineno = n->lineno;

//dump("mapop", n);

	r = n;
	switch(n->op) {
	default:
		fatal("mapop: unknown op %E", n->op);

	case ONEW:
		if(top != Erv)
			goto nottop;

		// newmap(keysize uint32, valsize uint32,
		//	keyalg uint32, valalg uint32,
		//	hint uint32) (hmap *map[any-1]any-2);

		t = fixmap(n->type);
		if(t == T)
			break;

		a = n->left;				// hint
		if(n->left == N)
			a = nodintconst(0);
		r = a;
		a = nodintconst(algtype(t->type));	// val algorithm
		r = nod(OLIST, a, r);
		a = nodintconst(algtype(t->down));	// key algorithm
		r = nod(OLIST, a, r);
		a = nodintconst(t->type->width);	// val width
		r = nod(OLIST, a, r);
		a = nodintconst(t->down->width);	// key width
		r = nod(OLIST, a, r);

		on = syslook("newmap", 1);

		argtype(on, t->down);	// any-1
		argtype(on, t->type);	// any-2

		r = nod(OCALL, on, r);
		walktype(r, top);
		r->type = n->type;
		break;

	case OINDEX:
		if(top != Erv)
			goto nottop;
dump("access start", n);
		// mapaccess1(hmap *map[any]any, key any) (val any);

		t = fixmap(n->left->type);
		if(t == T)
			break;

		convlit(n->right, t->down);

		if(!eqtype(n->right->type, t->down, 0)) {
			badtype(n->op, n->right->type, t->down);
			break;
		}

		a = n->right;				// key
		if(!isptr[t->down->etype]) {
			a = nod(OADDR, a, N);
			a->type = ptrto(t);
		}
		r = a;
		a = n->left;				// map
		r = nod(OLIST, a, r);

		on = syslook("mapaccess1", 1);

		argtype(on, t->down);	// any-1
		argtype(on, t->type);	// any-2
		argtype(on, t->down);	// any-3
		argtype(on, t->type);	// any-4

		r = nod(OCALL, on, r);
		walktype(r, Erv);
		r->type = t->type;
dump("access finish", r);
		break;

		// mapaccess2(hmap *map[any]any, key any) (val any, pres bool);

		t = fixmap(n->left->type);
		if(t == T)
			break;

		convlit(n->right, t->down);

		if(!eqtype(n->right->type, t->down, 0)) {
			badtype(n->op, n->right->type, t->down);
			break;
		}

		a = n->right;				// key
		if(!isptr[t->down->etype]) {
			a = nod(OADDR, a, N);
			a->type = ptrto(t);
		}
		r = a;
		a = n->left;				// map
		r = nod(OLIST, a, r);

		on = syslook("mapaccess2", 1);

		argtype(on, t->down);	// any-1
		argtype(on, t->type);	// any-2
		argtype(on, t->down);	// any-3
		argtype(on, t->type);	// any-4

		r = nod(OCALL, on, r);
		walktype(r, Erv);
		r->type = t->type;
		break;

	case OAS:
		if(top != Elv)
			goto nottop;
		if(n->left->op != OINDEX)
			fatal("mapos: AS left not OINDEX");

		// mapassign1(hmap *map[any-1]any-2, key any-3, val any-4);

		t = fixmap(n->left->left->type);
		if(t == T)
			break;

		a = n->right;				// val
		r = a;
		a = n->left->right;			// key
		r = nod(OLIST, a, r);
		a = n->left->left;			// map
		r = nod(OLIST, a, r);

		on = syslook("mapassign1", 1);

		argtype(on, t->down);	// any-1
		argtype(on, t->type);	// any-2
		argtype(on, t->down);	// any-3
		argtype(on, t->type);	// any-4

		r = nod(OCALL, on, r);
		walktype(r, Erv);
		break;

/* BOTCH get 2nd version attached */
		if(top != Elv)
			goto nottop;
		if(n->left->op != OINDEX)
			fatal("mapos: AS left not OINDEX");

		// mapassign2(hmap *map[any]any, key any, val any, pres bool);

		t = fixmap(n->left->left->type);
		if(t == T)
			break;

		a = n->right;				// pres
		r = a;
		a = n->right;				// val
		r =nod(OLIST, a, r);
		a = n->left->right;			// key
		r = nod(OLIST, a, r);
		a = n->left->left;			// map
		r = nod(OLIST, a, r);

		on = syslook("mapassign2", 1);

		argtype(on, t->down);	// any-1
		argtype(on, t->type);	// any-2
		argtype(on, t->down);	// any-3
		argtype(on, t->type);	// any-4

		r = nod(OCALL, on, r);
		walktype(r, Erv);
		break;

	}
//dump("mapop return", r);
	dynlineno = lno;
	return r;

nottop:
	dump("bad top", n);
	fatal("mapop: top=%d %O", top, n->op);
	return N;
}

void
diagnamed(Type *t)
{
	if(isinter(t))
		if(t->sym == S)
			yyerror("interface type must be named");
	if(isptrto(t, TSTRUCT))
		if(t->type == T || t->type->sym == S)
			yyerror("structure type must be named");
}

Node*
convas(Node *n)
{
	int o;
	Node *l, *r;
	Type *lt, *rt;

	if(n->op != OAS)
		fatal("convas: not as %O", n->op);

	ullmancalc(n);
	l = n->left;
	r = n->right;
	if(l == N || r == N)
		return n;

	lt = l->type;
	rt = r->type;
	if(lt == T || rt == T)
		return n;

	if(n->left->op == OINDEX)
	if(isptrto(n->left->left->type, TMAP)) {
		*n = *mapop(n, Elv);
		return n;
	}

	if(n->left->op == OINDEXPTR)
	if(n->left->left->type->etype == TMAP) {
		*n = *mapop(n, Elv);
		return n;
	}

	if(eqtype(lt, rt, 0))
		return n;

	if(isinter(lt)) {
		if(isptrto(rt, TSTRUCT)) {
			o = OS2I;
			goto ret;
		}
		if(isinter(rt)) {
			o = OI2I;
			goto ret;
		}
	}

	if(isptrto(lt, TSTRUCT)) {
		if(isinter(rt)) {
			o = OI2S;
			goto ret;
		}
	}

	badtype(n->op, lt, rt);
	return n;

ret:
	diagnamed(lt);
	diagnamed(rt);

	n->right = nod(o, r, N);
	n->right->type = l->type;
	walktype(n, Etop);
	return n;
}

void
arrayconv(Type *t, Node *n)
{
	int c;
	Iter save;
	Node *l;

	l = listfirst(&save, &n);
	c = 0;

loop:
	if(l == N) {
		if(t->bound == 0)
			t->bound = c;
		if(t->bound == 0 || t->bound < c)
			yyerror("error with array convert bounds");
		return;
	}

	c++;
	walktype(l, Erv);
	convlit(l, t->type);
	if(!ascompat(l->type, t->type))
		badtype(OARRAY, l->type, t->type);
	l = listnext(&save);
	goto loop;
}

Node*
reorder1(Node *n)
{
	Iter save;
	Node *l, *r, *f, *a, *g;
	int c, t;

	/*
	 * from ascompat[te]
	 * evaluating actual function arguments.
	 *	f(a,b)
	 * if there is exactly one function expr,
	 * then it is done first. otherwise must
	 * make temp variables
	 */

	l = listfirst(&save, &n);
	c = 0;	// function calls
	t = 0;	// total parameters

loop1:
	if(l == N) {
		if(c == 0 || t == 1)
			return n;
		goto pass2;
	}
	if(l->op == OLIST)
		fatal("reorder1 OLIST");

	t++;
	if(l->ullman >= UINF)
		c++;
	l = listnext(&save);
	goto loop1;

pass2:
	l = listfirst(&save, &n);
	g = N;	// fncalls assigned to tempnames
	f = N;	// one fncall assigned to stack
	r = N;	// non fncalls and tempnames assigned to stack

loop2:
	if(l == N) {
		r = rev(r);
		g = rev(g);
		if(g != N)
			f = nod(OLIST, g, f);
		r = nod(OLIST, f, r);
		return r;
	}
	if(l->ullman < UINF) {
		if(r == N)
			r = l;
		else
			r = nod(OLIST, l, r);
		goto more;
	}
	if(f == N) {
		f = l;
		goto more;
	}

	// make assignment of fncall to tempname
	a = nod(OXXX, N, N);
	tempname(a, l->right->type);
	a = nod(OAS, a, l->right);

	if(g == N)
		g = a;
	else
		g = nod(OLIST, a, g);

	// put normal arg assignment on list
	// with fncall replaced by tempname
	l->right = a->left;
	if(r == N)
		r = l;
	else
		r = nod(OLIST, l, r);

more:
	l = listnext(&save);
	goto loop2;
}

Node*
reorder2(Node *n)
{
	Iter save;
	Node *l;
	int c;

	/*
	 * from ascompat[et]
	 *	a,b = f()
	 * return of a multi.
	 * there can be no function calls at all,
	 * or they will over-write the return values.
	 */

	l = listfirst(&save, &n);
	c = 0;

loop1:
	if(l == N) {
		if(c > 0)
			yyerror("reorder2: too many funcation calls evaluating parameters");
		return n;
	}
	if(l->op == OLIST)
		fatal("reorder2 OLIST");

	if(l->ullman >= UINF)
		c++;
	l = listnext(&save);
	goto loop1;
}

Node*
reorder3(Node *n)
{
	/*
	 * from ascompat[ee]
	 *	a,b = c,d
	 * simultaneous assignment. there can be
	 * later use of an earlier lvalue.
	 */
	return n;
}

Node*
reorder4(Node *n)
{
	/*
	 * from ascompat[te]
	 *	return c,d
	 * return expression assigned to output
	 * parameters. there may be no problems.
	 */
	return n;
}
