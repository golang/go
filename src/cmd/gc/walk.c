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
	curfn = fn;
	walktype(fn->nbody, 1);
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

loop:
	if(n == N)
		goto ret;
	if(n->op != ONAME)
		dynlineno = n->lineno;	// for diagnostics

	t = T;
	et = Txxx;

	switch(n->op) {
	default:
		fatal("walktype: switch 1 unknown op %N", n);
		goto ret;

	case OPRINT:
		walktype(n->left, 0);
		*n = *prcompat(n->left);
		goto ret;

	case OPANIC:
		walktype(n->left, 0);
		*n = *nod(OLIST, prcompat(n->left), nodpanic(n->lineno));
		goto ret;

	case OLITERAL:
		n->addable = 1;
		ullmancalc(n);
		goto ret;

	case ONAME:
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
		if(!top)
			goto nottop;
		walktype(n->ninit, 1);
		walktype(n->ntest, 1);
		walktype(n->nincr, 1);
		n = n->nbody;
		goto loop;

	case OSWITCH:
		if(!top)
			goto nottop;

		if(n->ntest == N)
			n->ntest = booltrue;
		walktype(n->ninit, 1);
		walktype(n->ntest, 1);
		walktype(n->nbody, 1);

		// find common type
		if(n->ntest->type == T)
			n->ntest->type = walkswitch(n->ntest, n->nbody, sw1);

		// if that fails pick a type
		if(n->ntest->type == T)
			n->ntest->type = walkswitch(n->ntest, n->nbody, sw2);

		// set the type on all literals
		if(n->ntest->type != T)
			walkswitch(n->ntest, n->nbody, sw3);

		walktype(n->ntest, 1);

		n = n->nincr;
		goto loop;

	case OEMPTY:
		if(!top)
			goto nottop;
		goto ret;

	case OIF:
		if(!top)
			goto nottop;
		walktype(n->ninit, 1);
		walktype(n->ntest, 1);
		walktype(n->nelse, 1);
		n = n->nbody;
		goto loop;

	case OCALLMETH:
	case OCALLINTER:
	case OCALL:
		n->ullman = UINF;
		if(n->type != T)
			goto ret;

		walktype(n->left, 0);
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

		walktype(n->right, 0);

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
		if(!top)
			goto nottop;

		r = n->right;
		if(r == N)
			goto ret;
		l = n->left;
		if(l == N)
			goto ret;

		if(r->op == OCALL && l->op == OLIST) {
			walktype(l, 0);
			walktype(r, 0);
			l = ascompatet(n->op, &n->left, &r->type, 0);
			if(l != N) {
				*n = *nod(OLIST, r, reorder2(l));
			}
			goto ret;
		}

		walktype(l, 0);
		walktype(r, 0);
		l = ascompatee(n->op, &n->left, &n->right);
		if(l != N)
			*n = *reorder3(l);
		goto ret;

	case OBREAK:
	case OCONTINUE:
	case OGOTO:
	case OLABEL:
		goto ret;

	case OXCASE:
		yyerror("case statement out of place");
		n->op = OCASE;

	case OCASE:
		n = n->left;
		goto loop;

	case OXFALL:
		yyerror("fallthrough statement out of place");
		n->op = OFALL;

	case OFALL:
	case OINDREG:
		goto ret;

	case OS2I:
	case OI2S:
	case OI2I:
		n->addable = 0;
		walktype(n->left, 0);
		goto ret;

	case OCONV:
		walktype(n->left, 0);
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
				*n = *stringop(n);
				goto ret;
			}
			if(isbytearray(n->left->type) != 0) {
				n->op = OARRAY;
				*n = *stringop(n);
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
		walktype(n->left, 0);
		l = ascompatte(n->op, getoutarg(curfn->type), &n->left, 1);
		if(l != N)
			n->left = reorder4(l);
		goto ret;

	case ONOT:
		walktype(n->left, 0);
		if(n->left == N || n->left->type == T)
			goto ret;
		et = n->left->type->etype;
		break;

	case OASOP:
		if(!top)
			goto nottop;

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
		walktype(n->left, 0);
		walktype(n->right, 0);
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
				*n = *stringop(n);
				goto ret;
			}
		}
		break;

	case OMINUS:
	case OPLUS:
	case OCOM:
		walktype(n->left, 0);
		if(n->left == N)
			goto ret;
		evconst(n);
		ullmancalc(n);
		if(n->op == OLITERAL)
			goto ret;
		break;

	case OLEN:
		walktype(n->left, 0);
		evconst(n);
		ullmancalc(n);
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
		walktype(n->left, 0);
		walktype(n->right, 0);
		ullmancalc(n);
		if(n->left == N || n->right == N)
			goto ret;
		t = n->left->type;
		if(t == T)
			goto ret;

		// map
		if(isptrto(t, TMAP)) {
			fatal("index map");
			goto ret;
		}

		// right side must be an int
		if(n->right->type == T)
			convlit(n->right, types[TINT32]);
		if(n->left->type == T || n->right->type == T)
			goto ret;
		if(!isint[n->right->type->etype])
			goto badt;

		// left side is string
		if(isptrto(t, TSTRING)) {
			*n = *stringop(n);
			goto ret;
		}

		// left side is array
		if(isptr[t->etype]) {
			t = t->type;
			n->op = OINDEXPTR;
		}
		if(t->etype != TARRAY && t->etype != TDARRAY)
			goto badt;
		n->type = t->type;
		goto ret;

	case OSLICE:
		walktype(n->left, 0);
		walktype(n->right, 0);
		if(n->left == N || n->right == N)
			goto ret;
		if(isptrto(n->left->type, TSTRING)) {
			*n = *stringop(n);
			goto ret;
		}
		badtype(OSLICE, n->left->type, T);
		goto ret;

	case ODOT:
	case ODOTPTR:
	case ODOTMETH:
	case ODOTINTER:
		walkdot(n);
		goto ret;

	case OADDR:
		walktype(n->left, 0);
		if(n->left == N)
			goto ret;
		t = n->left->type;
		if(t == T)
			goto ret;
		n->type = ptrto(t);
		goto ret;

	case OIND:
		walktype(n->left, 0);
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
		walktype(n, 0);
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
	fatal("walktype: not top %O", n->op);
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
walkswitch(Node *test, Node *body, Type*(*call)(Node*, Type*))
{
	Node *n, *c;
	Type *place;

	place = call(test, T);

	n = body;
	if(n->op == OLIST)
		n = n->left;

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

	if(t->op != OXCASE)
		return 0;

loop:
	if(t == N) {
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

	/* if first statement is not case then return 0 */
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
walkdot(Node *n)
{
	Node *mn;
	Type *t, *f;
	int i;

	if(n->left == N || n->right == N)
		return;

	walktype(n->left, 0);
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
	Sym *s;

	r = N;
	l = listfirst(&save, &n);

loop:
	if(l == N) {
		walktype(r, 1);
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

	s = pkglookup(name, "sys");
	if(s == S || s->oname == N)
		fatal("prcompat: cant find sys_%s", name);

	t = *getinarg(s->oname->type);
	if(t != nil)
		t = t->type;
	if(t != nil)
		t = t->type;

	if(!eqtype(t, l->type, 0)) {
		l = nod(OCONV, l, N);
		l->type = t;
	}

	if(r == N)
		r = nod(OCALL, s->oname, l);
	else
		r = nod(OLIST, r, nod(OCALL, s->oname, l));

	l = listnext(&save);
	goto loop;
}

Node*
nodpanic(long lineno)
{
	Sym *s;
	char *name;
	Node *n;

	name = "panicl";
	s = pkglookup(name, "sys");
	if(s == S || s->oname == N)
		fatal("prcompat: cant find sys_%s", name);

	n = nodintconst(lineno);
	n = nod(OCALL, s->oname, n);
	walktype(n, 1);
	return n;
}

Node*
newcompat(Node *n)
{
	Node *r;
	Type *t;
	Sym *s;

	if(n->left != N)
		yyerror("dont know what new(,e) means");
	t = n->type;
	if(t == T || !isptr[t->etype])
		fatal("NEW sb pointer %lT", t);

	dowidth(t->type);

	s = pkglookup("mal", "sys");
	if(s == S || s->oname == N)
		fatal("newcompat: cant find sys_mal");

	r = nodintconst(t->type->width);
	r = nod(OCALL, s->oname, r);
	walktype(r, 0);

//	r = nod(OCONV, r, N);
	r->type = t;

	return r;
}

Node*
stringop(Node *n)
{
	Node *r, *c;
	Sym *s;
	long lno;
	long l;

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
		s = pkglookup("cmpstring", "sys");
		if(s == S || s->oname == N)
			fatal("stringop: cant find sys_cmpstring");

		r = nod(OLIST, n->left, n->right);
		r = nod(OCALL, s->oname, r);
		c = nodintconst(0);
		r = nod(n->op, r, c);
		break;

	case OADD:
		// sys_catstring(s1, s2)
		s = pkglookup("catstring", "sys");
		if(s == S || s->oname == N)
			fatal("stringop: cant find sys_catstring");
		r = nod(OLIST, n->left, n->right);
		r = nod(OCALL, s->oname, r);
		break;

	case OASOP:
		// sys_catstring(s1, s2)
		switch(n->etype) {
		default:
			fatal("stringop: unknown op %E-%E", n->op, n->etype);

		case OADD:
			// s1 = sys_catstring(s1, s2)
			s = pkglookup("catstring", "sys");
			if(s == S || s->oname == N || n->etype != OADD)
				fatal("stringop: cant find sys_catstring");
			r = nod(OLIST, n->left, n->right);
			r = nod(OCALL, s->oname, r);
			r = nod(OAS, n->left, r);
			break;
		}
		break;

	case OSLICE:
		// sys_slicestring(s, lb, hb)
		s = pkglookup("slicestring", "sys");
		if(s == S || s->oname == N)
			fatal("stringop: cant find sys_slicestring");

		r = nod(OCONV, n->right->left, N);
		r->type = types[TINT32];

		c = nod(OCONV, n->right->right, N);
		c->type = types[TINT32];

		r = nod(OLIST, r, c);

		r = nod(OLIST, n->left, r);

		r = nod(OCALL, s->oname, r);
		break;

	case OINDEX:
		// sys_indexstring(s, i)
		s = pkglookup("indexstring", "sys");
		if(s == S || s->oname == N)
			fatal("stringop: cant find sys_indexstring");

		r = nod(OCONV, n->right, N);
		r->type = types[TINT32];

		r = nod(OLIST, n->left, r);
		r = nod(OCALL, s->oname, r);
		break;

	case OCONV:
		// sys_intstring(v)
		s = pkglookup("intstring", "sys");
		if(s == S || s->oname == N)
			fatal("stringop: cant find sys_intstring");

		r = nod(OCONV, n->left, N);
		r->type = types[TINT64];

		r = nod(OCALL, s->oname, r);
		break;

	case OARRAY:
		// byteastring(a, l)
		s = pkglookup("byteastring", "sys");
		if(s == S || s->oname == N)
			fatal("stringop: cant find sys_byteastring");

		c = nodintconst(0);
		r = nod(OINDEX, n->left, c);
		r = nod(OADDR, r, N);

		l = isbytearray(n->left->type);
		c = nodintconst(l-1);

		r = nod(OLIST, r, c);
		r = nod(OCALL, s->oname, r);
		break;
	}

	walktype(r, 1);
	dynlineno = lno;
	return r;
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

	l = n->left;
	r = n->right;
	if(l == N || r == N)
		return n;

	lt = l->type;
	rt = r->type;
	if(lt == T || rt == T)
		return n;

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
	walktype(n, 1);
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
	walktype(l, 0);
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
	Node *l, *r, *f;
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
		if(c > 1) {
			yyerror("reorder1: too many funcation calls evaluating parameters");
			return n;
		}
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
	f = N;	// isolated function call
	r = N;	// rest of them

loop2:
	if(l == N) {
		if(r == N || f == N)
			fatal("reorder1 not nil 1");
		r = nod(OLIST, f, r);
		return rev(r);
	}
	if(l->ullman >= UINF) {
		if(f != N)
			fatal("reorder1 not nil 2");
		f = l;
	} else
	if(r == N)
		r = l;
	else
		r = nod(OLIST, l, r);

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
