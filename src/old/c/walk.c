// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"

static	Node*	sw1(Node*, Node*);
static	Node*	sw2(Node*, Node*);
static	Node*	sw3(Node*, Node*);
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
	Node *t, *r;
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

	t = N;
	et = Txxx;

	switch(n->op) {
	default:
		fatal("walktype: switch 1 unknown op %N", n);
		goto ret;

	case OPANIC:
	case OPRINT:
		walktype(n->left, 0);
		prcompat(&n->left);
		goto ret;

	case OLITERAL:
		n->addable = 1;
		ullmancalc(n);
		goto ret;

	case ONAME:
		n->addable = 1;
		ullmancalc(n);
		if(n->type == N) {
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
		if(n->ntest->type == N)
			n->ntest->type = walkswitch(n->ntest, n->nbody, sw1);

		// if that fails pick a type
		if(n->ntest->type == N)
			n->ntest->type = walkswitch(n->ntest, n->nbody, sw2);

		// set the type on all literals
		if(n->ntest->type != N)
			walkswitch(n->ntest, n->nbody, sw3);

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

	case OCALL:
	case OCALLPTR:
	case OCALLMETH:
	case OCALLINTER:
		walktype(n->left, 0);
		if(n->left == N)
			goto ret;
		t = n->left->type;
		if(t == N)
			goto ret;

		if(n->left->op == ODOTMETH)
			n->op = OCALLMETH;
		if(n->left->op == ODOTINTER)
			n->op = OCALLINTER;

		if(t->etype == TPTR) {
			t = t->type;
			n->op = OCALLPTR;
		}

		if(t->etype != TFUNC) {
			yyerror("call of a non-function %T", t);
			goto ret;
		}

		n->type = *getoutarg(t);
		switch(t->outtuple) {
		default:
			n->kaka = PCALL_MULTI;
			if(!top)
				yyerror("function call must be single valued (%d)", et);
			break;
		case 0:
			n->kaka = PCALL_NIL;
			break;
		case 1:
			n->kaka = PCALL_SINGLE;
			n->type = n->type->type->type;
			break;
		}

		r = n->right;
		walktype(r, 0);
		ascompatte(n->op, getinarg(t), &n->right);
		goto ret;

	case OAS:
		if(!top)
			goto nottop;

		n->kaka = PAS_SINGLE;
		r = n->left;
		if(r != N && r->op == OLIST)
			n->kaka = PAS_MULTI;

		walktype(r, 0);

		r = n->right;
		if(r == N)
			goto ret;

		if(r->op == OCALL && n->kaka == PAS_MULTI) {
			walktype(r, 1);
			if(r->kaka == PCALL_MULTI) {
				ascompatet(n->op, &n->left, &r->type);
				n->kaka = PAS_CALLM;
				goto ret;
			}
		}

		walktype(n->right, 0);
		ascompatee(n->op, &n->left, &n->right);

		if(n->kaka == PAS_SINGLE) {
			t = n->right->type;
			if(t != N && t->etype == TSTRUCT)
				n->kaka = PAS_STRUCT;
		}
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
		goto ret;

	case OCONV:
		walktype(n->left, 0);
		if(n->left == N)
			goto ret;
		convlit(n->left, n->type);
		if(eqtype(n->type, n->left->type, 0))
			*n = *n->left;
		goto ret;

	case ORETURN:
		walktype(n->left, 0);
		ascompatte(n->op, getoutarg(curfn->type), &n->left);
		goto ret;

	case ONOT:
		walktype(n->left, 0);
		if(n->left == N || n->left->type == N)
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
	case OCAT:
		walktype(n->left, 0);
		walktype(n->right, 0);
		if(n->left == N || n->right == N)
			goto ret;
		convlit(n->left, n->right->type);
		convlit(n->right, n->left->type);
		evconst(n);
		if(n->op == OLITERAL)
			goto ret;
		if(n->left->type == N || n->right->type == N)
			goto ret;
		if(!ascompat(n->left->type, n->right->type))
			goto badt;
		break;

	case OPLUS:
	case OMINUS:
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
		if(t != N && t->etype == TPTR)
			t = t->type;
		if(t == N)
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
	case OINDEXSTR:
	case OINDEXMAP:
	case OINDEXPTRMAP:
		walktype(n->left, 0);
		walktype(n->right, 0);
		ullmancalc(n);
		if(n->left == N || n->right == N)
			goto ret;
		t = n->left->type;
		if(t == N)
			goto ret;

		// map - left and right sides must match
		if(t->etype == TMAP || isptrto(t, TMAP)) {
			n->ullman = UINF;
			n->op = OINDEXMAP;
			if(isptrto(t, TMAP)) {
				n->op = OINDEXPTRMAP;
				t = t->type;
				if(t == N)
					goto ret;
			}
			convlit(n->right, t->down);
			if(!ascompat(t->down, n->right->type))
				goto badt;
			n->type = t->type;
			goto ret;
		}

		// right side must be an int
		if(n->right->type == N)
			convlit(n->right, types[TINT32]);
		if(n->left->type == N || n->right->type == N)
			goto ret;
		if(!isint[n->right->type->etype])
			goto badt;

		// left side is string
		if(isptrto(t, TSTRING)) {
			n->op = OINDEXSTR;
			n->type = types[TUINT8];
			goto ret;
		}

		// left side is ptr to string
		if(isptrto(t, TPTR) && isptrto(t->type, TSTRING)) {
			n->op = OINDEXPTRSTR;
			n->type = types[TUINT8];
			goto ret;
		}

		// left side is array
		if(t->etype == TPTR) {
			t = t->type;
			n->op = OINDEXPTR;
		}
		if(t->etype != TARRAY && t->etype != TDARRAY)
			goto badt;
		n->type = t->type;
		goto ret;

	case OSLICE:
		walkslice(n);
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
		if(t == N)
			goto ret;
		n->type = ptrto(t);
		goto ret;

	case OIND:
		walktype(n->left, 0);
		if(n->left == N)
			goto ret;
		t = n->left->type;
		if(t == N)
			goto ret;
		if(t->etype != TPTR)
			goto badt;
		n->type = t->type;
		goto ret;

	case ONEW:
		if(n->left != N)
			yyerror("dont know what new(,e) means");
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

	case OCAT:
	case OADD:
		if(isptrto(n->left->type, TSTRING)) {
			n->op = OCAT;
			break;
		}

	case OSUB:
	case OMUL:
	case ODIV:
	case OPLUS:
	case OMINUS:
		et = n->left->type->etype;
		if(!okforadd[et])
			goto badt;
		break;

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

	if(t == N)
		t = n->left->type;
	n->type = t;
	ullmancalc(n);
	goto ret;

nottop:
	fatal("walktype: not top %O", n->op);

badt:
	if(n->right == N) {
		if(n->left == N) {
			badtype(n->op, N, N);
			goto ret;
		}
		badtype(n->op, n->left->type, N);
		goto ret;
	}
	badtype(n->op, n->left->type, n->right->type);
	goto ret;

ret:
	dynlineno = lno;
}

/*
 * return the first type
 */
Node*
sw1(Node *c, Node *place)
{
	if(place == N)
		return c->type;
	return place;
}

/*
 * return a suitable type
 */
Node*
sw2(Node *c, Node *place)
{
	return types[TINT32];	// botch
}

/*
 * check that selected type
 * is compat with all the cases
 */
Node*
sw3(Node *c, Node *place)
{
	if(place == N)
		return c->type;
	if(c->type == N)
		c->type = place;
	convlit(c, place);
	if(!ascompat(place, c->type))
		badtype(OSWITCH, place, c->type);
	return place;
}

Node*
walkswitch(Node *test, Node *body, Node*(*call)(Node*, Node*))
{
	Node *n, *c;
	Node *place;

	place = call(test, N);

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

Node*
lookdot(Node *n, Node *t, int d)
{
	Node *r, *f, *c;
	Sym *s;
	int o;

	r = N;
	s = n->sym;
	if(d > 0)
		goto deep;

	o = 0;
	for(f=t->type; f!=N; f=f->down) {
		f->kaka = o;
		o++;

		if(f->sym == S)
			continue;
		if(f->sym != s)
			continue;
		if(r != N) {
			yyerror("ambiguous DOT reference %s", s->name);
			break;
		}
		r = f;
	}
	return r;

deep:
	/* deeper look after shallow failed */
	for(f=t->type; f!=N; f=f->down) {
		// only look at unnamed sub-structures
		// BOTCH no such thing -- all are assigned temp names
		if(f->sym != S)
			continue;
		c = f->type;
		if(c->etype != TSTRUCT)
			continue;
		c = lookdot(n, c, d-1);
		if(c == N)
			continue;
		if(r != N) {
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
	Node *t, *f;
	int i;

	if(n->left == N || n->right == N)
		return;

	walktype(n->left, 0);
	if(n->right->op != ONAME) {
		yyerror("rhs of . must be a name");
		return;
	}

	t = n->left->type;
	if(t == N)
		return;

	if(t->etype == TPTR) {
		t = t->type;
		if(t == N)
			return;
		n->op = ODOTPTR;
	}

	if(n->right->op != ONAME)
		fatal("walkdot: not name %O", n->right->op);

	switch(t->etype) {
	default:
		badtype(ODOT, t, N);
		return;

	case TSTRUCT:
	case TINTER:
		for(i=0; i<5; i++) {
			f = lookdot(n->right, t, i);
			if(f != N)
				break;
		}
		if(f == N) {
			yyerror("undefined DOT reference %N", n->right);
			break;
		}
		n->right = f->nname;		// substitute real name
		n->type = f->type;
		if(n->type->etype == TFUNC) {
			n->op = ODOTMETH;
			if(t->etype == TINTER) {
				n->op = ODOTINTER;
				n->kaka = f->kaka;
			}
		}
		break;
	}
}

void
walkslice(Node *n)
{
	Node *l, *r;

	if(n->left == N || n->right == N)
		return;
	if(n->right->op != OLIST)
		fatal("slice not a list");

	walktype(n->left, 0);
	if(isptrto(n->left->type, TSTRING)) {
		n->op = OSLICESTR;
		goto ok;
	}
	if(isptrto(n->left->type->type, TPTR) && isptrto(n->left->type->type, TSTRING)) {
		n->op = OSLICEPTRSTR;
		goto ok;
	}

	badtype(OSLICE, n->left->type, N);
	return;

ok:
	// check for type errors
	walktype(n->right, 0);
	l = n->right->left;
	r = n->right->right;
	convlit(l, types[TINT32]);
	convlit(r, types[TINT32]);
	if(l == N || r == N ||
	   l->type == N || r->type == N)
		return;
	if(!isint[l->type->etype] || !isint[l->type->etype]) {
		badtype(OSLICE, l->type, r->type);
		return;
	}

	// now convert to int32
	n->right->left = nod(OCONV, n->right->left, N);
	n->right->left->type = types[TINT32];
	n->right->right = nod(OCONV, n->right->right, N);
	n->right->right->type = types[TINT32];
	walktype(n->right, 0);

	n->type = n->left->type;
}

/*
 * test tuple type list against each other
 * called in four contexts
 *	1. a,b = c,d		...ee
 *	2. a,b = fn()		...et
 *	3. call(fn())		...tt
 *	4. call(a,b)		...te
 */
void
ascompatee(int op, Node **nl, Node **nr)
{
	Node *l, *r;
	Iter savel, saver;
	int sa, na;

	l = listfirst(&savel, nl);
	r = listfirst(&saver, nr);
	na = 0;	// number of assignments - looking for multi
	sa = 0;	// one of the assignments is a structure assignment

loop:
	if(l == N || r == N) {
		if(l != r)
			yyerror("error in shape across assignment");
		if(sa != 0 && na > 1)
			yyerror("cant do multi-struct assignments");
		return;
	}

	convlit(r, l->type);

	if(!ascompat(l->type, r->type)) {
		badtype(op, l->type, r->type);
		return;
	}
	if(l->type != N && l->type->etype == TSTRUCT)
		sa = 1;

	l = listnext(&savel);
	r = listnext(&saver);
	na++;
	goto loop;
}

void
ascompatet(int op, Node **nl, Node **nr)
{
	Node *l, *r;
	Iter savel, saver;

	l = listfirst(&savel, nl);
	r = structfirst(&saver, nr);

loop:
	if(l == N || r == N) {
		if(l != r)
			yyerror("error in shape across assignment");
		return;
	}

	if(!ascompat(l->type, r->type)) {
		badtype(op, l->type, r->type);
		return;
	}

	l = listnext(&savel);
	r = structnext(&saver);

	goto loop;
}

void
ascompatte(int op, Node **nl, Node **nr)
{
	Node *l, *r;
	Iter savel, saver;

	l = structfirst(&savel, nl);
	r = listfirst(&saver, nr);

loop:
	if(l == N || r == N) {
		if(l != r)
			yyerror("error in shape across assignment");
		return;
	}

	convlit(r, l->type);

	if(!ascompat(l->type, r->type)) {
		badtype(op, l->type, r->type);
		return;
	}

	l = structnext(&savel);
	r = listnext(&saver);

	goto loop;
}

void
ascompattt(int op, Node **nl, Node **nr)
{
	Node *l, *r;
	Iter savel, saver;

	l = structfirst(&savel, nl);
	r = structfirst(&saver, nr);

loop:
	if(l == N || r == N) {
		if(l != r)
			yyerror("error in shape across assignment");
		return;
	}

	if(!ascompat(l->type, r->type)) {
		badtype(op, l->type, r->type);
		return;
	}

	l = structnext(&savel);
	r = structnext(&saver);

	goto loop;
}

/*
 * can we assign var of type t2 to var of type t1
 */
int
ascompat(Node *t1, Node *t2)
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

void
prcompat(Node **n)
{
	Node *l, *t;
	Iter save;
	int w;

	l = listfirst(&save, n);

loop:
	if(l == N)
		return;

	t = N;
	w = whatis(l);
	switch(w) {
	default:
		badtype((*n)->op, l->type, N);
		break;
	case Wtint:
	case Wtfloat:
	case Wtbool:
	case Wtstr:
		break;
	case Wlitint:
		t = types[TINT32];
		break;
	case Wlitfloat:
		t = types[TFLOAT64];
		break;
	case Wlitbool:
		t = types[TBOOL];
		break;
	case Wlitstr:
		t = types[TSTRING];
		break;
	}

	if(t != N)
		convlit(l, t);

	l = listnext(&save);
	goto loop;
}
