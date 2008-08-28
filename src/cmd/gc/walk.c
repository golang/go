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

int
isselect(Node *n)
{
	Sym *s;

	if(n == N)
		return 0;
	n = n->left;
	s = pkglookup("selectsend", "sys");
	if(s == n->sym)
		return 1;
	s = pkglookup("selectrecv", "sys");
	if(s == n->sym)
		return 1;
	return 0;
}

void
walktype(Node *n, int top)
{
	Node *r, *l;
	Type *t;
	Sym *s;
	int et, cl, cr;
	int32 lno;

	lno = setlineno(n);

	/*
	 * walk the whole tree of the body of a function.
	 * the types expressions are calculated.
	 * compile-time constants are evaluated.
	 */

	if(top == Exxx || top == Eyyy) {
		dump("", n);
		fatal("walktype: bad top=%d", top);
	}

loop:
	if(n == N)
		goto ret;

	setlineno(n);

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
		*n = *list(prcompat(n->left), nodpanic(n->lineno));
		goto ret;

	case OLITERAL:
		if(top != Erv)
			goto nottop;
		n->addable = 1;
		goto ret;

	case ONONAME:
		s = n->sym;
		if(s->undef == 0) {
			s->undef = 1;
			yyerror("%S: undefined", s);
			goto ret;
		}
		if(top == Etop)
			goto nottop;
		goto ret;

	case ONAME:
		if(top == Etop)
			goto nottop;
		n->addable = 1;
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
		walkbool(n->ntest);
		walktype(n->nincr, Etop);
		n = n->nbody;
		goto loop;

	case OSWITCH:
		if(top != Etop)
			goto nottop;

		if(!casebody(n->nbody))
			yyerror("switch statement must have case labels");

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

	case OSELECT:
		if(top != Etop)
			goto nottop;

		walkselect(n);
		goto ret;

	case OSCASE:
		if(top != Etop)
			goto nottop;
//		walktype(n->left, Erv);	SPECIAL
		n = n->right;
		goto loop;

	case OEMPTY:
		if(top != Etop)
			goto nottop;
		goto ret;

	case OIF:
		if(top != Etop)
			goto nottop;
		walktype(n->ninit, Etop);
		walkbool(n->ntest);
		walktype(n->nelse, Etop);
		n = n->nbody;
		goto loop;

	case OPROC:
		if(top != Etop)
			goto nottop;
		walktype(n->left, Etop);
		goto ret;

	case OCALLMETH:
	case OCALLINTER:
	case OCALL:
		if(top == Elv)
			goto nottop;

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
			if(isselect(n)) {
				// clear output bool - special prob with selectsend
				r = ascompatte(n->op, getoutarg(t), &boolfalse, 0);
				n->right = list(n->right, r);
			}
			break;

		case OCALLMETH:
			l = ascompatte(n->op, getinarg(t), &n->right, 0);
			r = ascompatte(n->op, getthis(t), &n->left->left, 0);
			l = list(r, l);
			n->left->left = N;
			ullmancalc(n->left);
			n->right = reorder1(l);
			break;
		}
		goto ret;

	case OAS:
		if(top != Etop)
			goto nottop;

		l = n->left;
		r = n->right;
		walktype(l, Elv);
		if(l == N || r == N)
			goto ret;

		cl = listcount(l);
		cr = listcount(r);

		if(cl == cr) {
			walktype(r, Erv);
			l = ascompatee(n->op, &n->left, &n->right);
			if(l != N)
				*n = *reorder3(l);
			goto ret;
		}

		switch(r->op) {

		case OCALLMETH:
		case OCALLINTER:
		case OCALL:
			if(cr == 1) {
				// a,b,... = fn()
				walktype(r, Erv);
				l = ascompatet(n->op, &n->left, &r->type, 0);
				if(l != N) {
					*n = *list(r, reorder2(l));
				}
				goto ret;
			}
			break;

		case OINDEX:
		case OINDEXPTR:
			if(cl == 2 && cr == 1) {
				// a,b = map[] - mapaccess2
				if(!isptrto(r->left->type, TMAP))
					break;
				l = mapop(n, top);
				if(l == N)
					break;
				*n = *l;
				goto ret;
			}
			break;

		case ORECV:
			if(cl == 2 && cr == 1) {
				// a,b = <chan - chanrecv2
				walktype(r->left, Erv);
				if(!isptrto(r->left->type, TCHAN))
					break;
				l = chanop(n, top);
				if(l == N)
					break;
				*n = *l;
				goto ret;
			}
			break;
		}

		switch(l->op) {
		case OINDEX:
		case OINDEXPTR:
			if(cl == 1 && cr == 2) {
				// map[] = a,b - mapassign2
				if(!isptrto(l->left->type, TMAP))
					break;
				l = mapop(n, top);
				if(l == N)
					break;
				*n = *l;
				goto ret;
			}
			break;
		}

		yyerror("bad shape across assignment - cr=%d cl=%d\n", cr, cl);
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

		l = n->left;
		if(l == N)
			goto ret;
		t = n->type;
		if(t == T)
			goto ret;

		convlit(l, t);

		// nil conversion
		if(eqtype(t, l->type, 0)) {
			if(l->op != ONAME)
				*n = *l;
			goto ret;
		}

		// simple fix-float
		if(l->type != T)
		if(isint[l->type->etype] || isfloat[l->type->etype])
		if(isint[t->etype] || isfloat[t->etype]) {
			evconst(n);
			goto ret;
		}

		// to string
		if(isptrto(t, TSTRING)) {
			if(isint[l->type->etype]) {
				*n = *stringop(n, top);
				goto ret;
			}
			if(isbytearray(l->type) != 0) {
				n->op = OARRAY;
				*n = *stringop(n, top);
				goto ret;
			}
		}

		// convert dynamic to static generated by ONEW
		if(isptrto(t, TARRAY) && isptrto(l->type, TDARRAY))
			goto ret;

//		if(t->etype == TARRAY) {
//			arrayconv(t, l);
//			goto ret;
//		}

		r = isandss(n->type, l);
		if(r != N) {
			*n = *r;
			goto ret;
		}
		badtype(n->op, l->type, t);
		goto ret;

	case ORETURN:
		if(top != Etop)
			goto nottop;
		walktype(n->left, Erv);
		if(curfn->type->outnamed && n->left == N) {
			// print("special return\n");
			goto ret;
		}
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
		l = n->left;
		if(l->op != OINDEX)
			goto com;
		if(!isptrto(l->left->type, TMAP))
			goto com;
		*n = *mapop(n, top);
		goto loop;

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
		evconst(n);
		if(n->op == OLITERAL)
			goto ret;
		convlit(n->left, n->right->type);
		convlit(n->right, n->left->type);
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
		case TMAP:
		case TDARRAY:
			break;
		case TARRAY:
			nodconst(n, types[TINT32], t->bound);
			break;
		}
		n->type = types[TINT32];
		goto ret;

	case OCAP:
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
		case TDARRAY:
			break;
		case TARRAY:
			nodconst(n, types[TINT32], t->bound);
			break;
		}
		n->type = types[TINT32];
		goto ret;

	case OINDEX:
	case OINDEXPTR:
		if(top == Etop)
			goto nottop;

		walktype(n->left, Erv);
		walktype(n->right, Erv);

		if(n->left == N || n->right == N)
			goto ret;

		defaultlit(n->left);
		t = n->left->type;
		if(t == T)
			goto ret;

// BOTCH - convert each index opcode
// to look like this and get rid of OINDEXPTR
		if(isptr[t->etype])
		if(isptrto(t, TSTRING) || isptrto(t->type, TSTRING)) {
			// right side must be an int
			if(top != Erv)
				goto nottop;
			if(n->right->type == T) {
				convlit(n->right, types[TINT32]);
				if(n->right->type == T)
					goto ret;
			}
			if(!isint[n->right->type->etype])
				goto badt;
			*n = *stringop(n, top);
			goto ret;
		}

		// left side is indirect
		if(isptr[t->etype]) {
			t = t->type;
			n->op = OINDEXPTR;
		}

		switch(t->etype) {
		default:
			goto badt;

		case TMAP:
			// right side must be map type
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

		case TDARRAY:
		case TARRAY:
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

	case OSEND:
		if(top == Elv)
			goto nottop;
		walktype(n->left, Erv);		// chan
		walktype(n->right, Erv);	// e
		*n = *chanop(n, top);
		goto ret;

	case ORECV:
		if(top == Elv)
			goto nottop;
		if(n->right == N) {
			walktype(n->left, Erv);	// chan
			*n = *chanop(n, top);	// returns e blocking
			goto ret;
		}
		walktype(n->left, Elv);		// e
		walktype(n->right, Erv);	// chan
		*n = *chanop(n, top);		// returns bool non-blocking
		goto ret;

	case OSLICE:
		if(top == Etop)
			goto nottop;

		walktype(n->left, top);
		walktype(n->right, Erv);
		if(n->left == N || n->right == N)
			goto ret;
		t = n->left->type;
		if(isptr[t->etype])
			t = t->type;
		if(t->etype == TSTRING) {
			*n = *stringop(n, top);
			goto ret;
		}
		if(t->etype == TDARRAY || t->etype == TARRAY) {
			*n = *arrayop(n, top);
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
		l->val.u.fval = mal(sizeof(*l->val.u.fval));
		l->val.ctype = CTFLT;
		mpmovecflt(l->val.u.fval, 0.0);

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
	lineno = lno;
}

void
walkbool(Node *n)
{
	walktype(n, Erv);
	if(n != N && n->type != T)
		if(!eqtype(n->type, types[TBOOL], 0))
			yyerror("IF and FOR require a boolean type");
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
 * check that switch type
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

	setlineno(sw);

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
				setlineno(c);
				place = call(c, place);
				break;
			}
			setlineno(c);
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

Node*
selcase(Node *n, Node *var)
{
	Node *a, *r, *on, *c;
	Type *t;
	Iter iter;

	c = n->left;
	if(c->op == ORECV)
		goto recv;

	walktype(c->left, Erv);		// chan
	walktype(c->right, Erv);	// elem

	t = fixchan(c->left->type);
	if(t == T)
		return;

	convlit(c->right, t->type);
	if(!ascompat(t->type, c->right->type)) {
		badtype(c->op, t->type, c->right->type);
		return;
	}

	// selectsend(sel *byte, hchan *chan any, elem any) (selected bool);
	on = syslook("selectsend", 1);
	argtype(on, t->type);
	argtype(on, t->type);

	a = c->right;			// elem
	r = a;
	a = c->left;			// chan
	r = list(a, r);
	a = var;			// sel-var
	r = list(a, r);

	goto out;

recv:
	if(c->right != N)
		goto recv2;

	walktype(c->left, Erv);		// chan

	t = fixchan(c->left->type);
	if(t == T)
		return;

	// selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
	on = syslook("selectrecv", 1);
	argtype(on, t->type);
	argtype(on, t->type);

	a = c->left;			// nil elem
	a = nod(OLITERAL, N, N);
	a->val.ctype = CTNIL;

	r = a;
	a = c->left;			// chan
	r = list(a, r);
	a = var;			// sel-var
	r = list(a, r);
	goto out;

recv2:	
	walktype(c->right, Erv);	// chan

	t = fixchan(c->right->type);
	if(t == T)
		return;

	walktype(c->left, Elv);	// elem
	convlit(c->left, t->type);
	if(!ascompat(t->type, c->left->type)) {
		badtype(c->op, t->type, c->left->type);
		return;
	}

	// selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
	on = syslook("selectrecv", 1);
	argtype(on, t->type);
	argtype(on, t->type);

	a = c->left;			// elem
	a = nod(OADDR, a, N);
	r = a;
	a = c->right;			// chan
	r = list(a, r);
	a = var;			// sel-var
	r = list(a, r);

out:
	a = nod(OCALL, on, r);
	r = nod(OIF, N, N);
	r->ntest = a;

	return r;
}

void
walkselect(Node *sel)
{
	Iter iter;
	Node *n, *oc, *on, *r;
	Node *var, *bod, *res;
	int count;
	int32 lno;

	lno = setlineno(sel);

	// generate sel-struct
	var = nod(OXXX, N, N);
	tempname(var, ptrto(types[TUINT8]));

	n = listfirst(&iter, &sel->left);
	if(n == N || n->op != OXCASE)
		yyerror("first select statement must be a case");

	count = 0;	// number of cases
	res = N;	// entire select body
	bod = N;	// body of each case
	oc = N;		// last case

	for(count=0; n!=N; n=listnext(&iter)) {
		setlineno(n);

		switch(n->op) {
		default:
			bod = list(bod, n);
			break;

		case OXCASE:
			switch(n->left->op) {
			default:
				yyerror("select cases must be send or recv");
				break;

			case OSEND:
			case ORECV:
				if(oc != N) {
					bod = list(bod, nod(OBREAK, N, N));
					oc->nbody = rev(bod);
				}
				oc = selcase(n, var);
				res = list(res, oc);
				break;
			}
			bod = N;
			count++;
			break;
		}
	}
	if(oc != N) {
		bod = list(bod, nod(OBREAK, N, N));
		oc->nbody = rev(bod);
	}
	setlineno(sel);

	// selectgo(sel *byte);
	on = syslook("selectgo", 0);
	r = nod(OCALL, on, var);		// sel-var
	res = list(res, r);

	// newselect(size uint32) (sel *byte);
	on = syslook("newselect", 0);

	r = nod(OXXX, N, N);
	nodconst(r, types[TINT32], count);	// count
	r = nod(OCALL, on, r);
	r = nod(OAS, var, r);

	sel->ninit = r;
	sel->nbody = rev(res);
	sel->left = N;

	walktype(sel->ninit, Etop);
	walktype(sel->nbody, Etop);

//dump("sel", sel);

	lineno = lno;
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
			yyerror("error in shape across %O", op);
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
		nn = list(a, nn);

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
			yyerror("error in shape across %O", op);
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
		nn = list(a, nn);

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
			yyerror("error in shape across %O", op);
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
		nn = list(a, nn);

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

	if(isptrto(t1, TDARRAY))
		if(isptrto(t2, TARRAY))
			return 1;
	return 0;
}

Node*
prcompat(Node *n)
{
	Node *l, *r;
	Node *on;
	Type *t;
	Iter save;
	int w;
	char *name;

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
		if(!isptr[l->type->etype]) {
			badtype(n->op, l->type, T);
			l = listnext(&save);
			goto loop;
		}
		on = syslook("printpointer", 1);
		argtype(on, l->type->type);	// any-1
		break;

	case Wlitint:
	case Wtint:
		on = syslook("printint", 0);
		break;
	case Wlitfloat:
	case Wtfloat:
		on = syslook("printfloat", 0);
		break;
	case Wlitbool:
	case Wtbool:
		on = syslook("printbool", 0);
		break;
	case Wlitstr:
	case Wtstr:
		on = syslook("printstring", 0);
		break;
	}

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
		r = list(r, nod(OCALL, on, l));

	l = listnext(&save);
	goto loop;
}

Node*
nodpanic(int32 lineno)
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
	switch(t->etype) {
	case TMAP:
		r = mapop(n, Erv);
		return r;

	case TCHAN:
		r = chanop(n, Erv);
		return r;

	case TDARRAY:
	case TARRAY:
		r = arrayop(n, Erv);
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
	int32 l;

	switch(n->op) {
	default:
		fatal("stringop: unknown op %O", n->op);

	case OEQ:
	case ONE:
	case OGE:
	case OGT:
	case OLE:
	case OLT:
		// sys_cmpstring(s1, s2) :: 0
		on = syslook("cmpstring", 0);
		r = list(n->left, n->right);
		r = nod(OCALL, on, r);
		c = nodintconst(0);
		r = nod(n->op, r, c);
		break;

	case OADD:
		// sys_catstring(s1, s2)
		on = syslook("catstring", 0);
		r = list(n->left, n->right);
		r = nod(OCALL, on, r);
		break;

	case OASOP:
		// sys_catstring(s1, s2)
		switch(n->etype) {
		default:
			fatal("stringop: unknown op %O-%O", n->op, n->etype);

		case OADD:
			// s1 = sys_catstring(s1, s2)
			if(n->etype != OADD)
				fatal("stringop: not cat");
			r = list(n->left, n->right);
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

		r = list(r, c);
		r = list(n->left, r);
		on = syslook("slicestring", 0);
		r = nod(OCALL, on, r);
		break;

	case OINDEX:
		// sys_indexstring(s, i)
		c = n->left;
		if(isptrto(c->type->type, TSTRING)) {
			// lhs is string or *string
			c = nod(OIND, c, N);
			c->type = c->left->type->type;
		}
		r = nod(OCONV, n->right, N);
		r->type = types[TINT32];
		r = list(c, r);
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

		r = list(r, c);
		on = syslook("byteastring", 0);
		r = nod(OCALL, on, r);
		break;
	}

	walktype(r, top);
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
		fatal("fixmap: %lT not map", tm);
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

Type*
fixchan(Type *tm)
{
	Type *t;

	t = tm->type;
	if(t == T) {
		fatal("fixchan: t nil");
		return T;
	}

	if(t->etype != TCHAN) {
		fatal("fixchan: %lT not chan", tm);
		return T;
	}

	if(t->type == T) {
		fatal("fixchan: chan element type is nil");
		return T;
	}

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
	Node *r, *a;
	Type *t;
	Node *on;
	int alg1, alg2, cl, cr;

//dump("mapop", n);

	r = n;
	switch(n->op) {
	default:
		fatal("mapop: unknown op %O", n->op);

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
		r = list(a, r);
		a = nodintconst(algtype(t->down));	// key algorithm
		r = list(a, r);
		a = nodintconst(t->type->width);	// val width
		r = list(a, r);
		a = nodintconst(t->down->width);	// key width
		r = list(a, r);

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
//		if(!isptr[t->down->etype]) {
//			a = nod(OADDR, a, N);
//			a->type = ptrto(t);
//		}

		r = a;
		a = n->left;				// map
		r = list(a, r);

		on = syslook("mapaccess1", 1);

		argtype(on, t->down);	// any-1
		argtype(on, t->type);	// any-2
		argtype(on, t->down);	// any-3
		argtype(on, t->type);	// any-4

		r = nod(OCALL, on, r);
		walktype(r, Erv);
		r->type = t->type;
		break;

	case OAS:
		cl = listcount(n->left);
		cr = listcount(n->right);

		if(cl == 1 && cr == 2)
			goto assign2;
		if(cl == 2 && cr == 1)
			goto access2;
		if(cl != 1 || cr != 1)
			goto shape;

		// mapassign1(hmap *map[any-1]any-2, key any-3, val any-4);

//dump("assign1", n);
		if(n->left->op != OINDEX)
			goto shape;

		t = fixmap(n->left->left->type);
		if(t == T)
			break;

		a = n->right;				// val
		r = a;
		a = n->left->right;			// key
		r = list(a, r);
		a = n->left->left;			// map
		r = list(a, r);

		on = syslook("mapassign1", 1);

		argtype(on, t->down);	// any-1
		argtype(on, t->type);	// any-2
		argtype(on, t->down);	// any-3
		argtype(on, t->type);	// any-4

		r = nod(OCALL, on, r);
		walktype(r, Erv);
		break;

	assign2:
		// mapassign2(hmap *map[any]any, key any, val any, pres bool);

//dump("assign2", n);
		if(n->left->op != OINDEX)
			goto shape;

		t = fixmap(n->left->left->type);
		if(t == T)
			break;

		a = n->right->right;			// pres
		r = a;
		a = n->right->left;			// val
		r =list(a, r);
		a = n->left->right;			// key
		r = list(a, r);
		a = n->left->left;			// map
		r = list(a, r);

		on = syslook("mapassign2", 1);

		argtype(on, t->down);	// any-1
		argtype(on, t->type);	// any-2
		argtype(on, t->down);	// any-3
		argtype(on, t->type);	// any-4

		r = nod(OCALL, on, r);
		walktype(r, Erv);
		break;

	access2:
		// mapaccess2(hmap *map[any-1]any-2, key any-3) (val-4 any, pres bool);

//dump("access2", n);
		if(n->right->op != OINDEX)
			goto shape;

		t = fixmap(n->right->left->type);
		if(t == T)
			break;

		a = n->right->right;			// key
		r = a;
		a = n->right->left;			// map
		r = list(a, r);

		on = syslook("mapaccess2", 1);

		argtype(on, t->down);	// any-1
		argtype(on, t->type);	// any-2
		argtype(on, t->down);	// any-3
		argtype(on, t->type);	// any-4

		n->right = nod(OCALL, on, r);
		walktype(n, Etop);
		r = n;
		break;

	case OASOP:
		// rewrite map[index] op= right
		// into tmpi := index; map[tmpi] = map[tmpi] op right

		t = n->left->left->type->type;
		a = nod(OXXX, N, N);
		tempname(a, t->down);			// tmpi
		r = nod(OAS, a, n->left->right);	// tmpi := index
		n->left->right = a;			// m[tmpi]

		a = nod(OXXX, N, N);
		*a = *n->left;				// copy of map[tmpi]
		a = nod(n->etype, a, n->right);		// m[tmpi] op right
		a = nod(OAS, n->left, a);		// map[tmpi] = map[tmpi] op right
		r = nod(OLIST, r, a);
	}
	return r;

shape:
	dump("shape", n);
	fatal("mapop: cl=%d cr=%d, %O", top, n->op);
	return N;

nottop:
	dump("bad top", n);
	fatal("mapop: top=%d %O", top, n->op);
	return N;
}

Node*
chanop(Node *n, int top)
{
	Node *r, *a;
	Type *t;
	Node *on;
	int alg, cl, cr;

//dump("chanop", n);

	r = n;
	switch(n->op) {
	default:
		fatal("chanop: unknown op %O", n->op);

	case ONEW:
		// newchan(elemsize uint32, elemalg uint32,
		//	hint uint32) (hmap *chan[any-1]);

		t = fixchan(n->type);
		if(t == T)
			break;

		a = n->left;				// hint
		if(n->left == N)
			a = nodintconst(0);
		r = a;
		a = nodintconst(algtype(t->type));	// elem algorithm
		r = list(a, r);
		a = nodintconst(t->type->width);	// elem width
		r = list(a, r);

		on = syslook("newchan", 1);
		argtype(on, t->type);	// any-1

		r = nod(OCALL, on, r);
		walktype(r, top);
		r->type = n->type;
		break;

	case OAS:
		cl = listcount(n->left);
		cr = listcount(n->right);

		if(cl != 2 || cr != 1 || n->right->op != ORECV)
			goto shape;

		// chanrecv2(hchan *chan any) (elem any, pres bool);
		t = fixchan(n->right->left->type);
		if(t == T)
			break;

		a = n->right->left;			// chan
		r = a;

		on = syslook("chanrecv2", 1);

		argtype(on, t->type);	// any-1
		argtype(on, t->type);	// any-2
		r = nod(OCALL, on, r);
		n->right = r;
		r = n;
		walktype(r, Etop);
		break;

	case ORECV:
		if(n->right != N)
			goto recv2;

		// chanrecv1(hchan *chan any) (elem any);

		t = fixchan(n->left->type);
		if(t == T)
			break;

		a = n->left;			// chan
		r = a;

		on = syslook("chanrecv1", 1);

		argtype(on, t->type);	// any-1
		argtype(on, t->type);	// any-2
		r = nod(OCALL, on, r);
		walktype(r, Erv);
		break;

	recv2:
		// chanrecv3(hchan *chan any, *elem any) (pres bool);
		t = fixchan(n->right->type);
		if(t == T)
			break;

		a = n->right;			// chan
		r = a;
		a = n->left;			// elem
		if(a == N) {
			a = nod(OLITERAL, N, N);
			a->val.ctype = CTNIL;
		} else
			a = nod(OADDR, a, N);

		on = syslook("chanrecv3", 1);

		argtype(on, t->type);	// any-1
		argtype(on, t->type);	// any-2

		r = nod(OCALL, on, r);
		n->right = r;
		r = n;
		walktype(r, Etop);
		break;

	case OSEND:
		t = fixchan(n->left->type);
		if(t == T)
			break;
		if(top != Etop)
			goto send2;

		// chansend1(hchan *chan any, elem any);
		t = fixchan(n->left->type);
		if(t == T)
			break;

		a = n->right;			// e
		r = a;
		a = n->left;			// chan
		r = list(a, r);

		on = syslook("chansend1", 1);
		argtype(on, t->type);	// any-1
		argtype(on, t->type);	// any-2
		r = nod(OCALL, on, r);
		walktype(r, top);
		break;

	send2:
		// chansend2(hchan *chan any, val any) (pres bool);
		t = fixchan(n->left->type);
		if(t == T)
			break;

		a = n->right;			// e
		r = a;
		a = n->left;			// chan
		r = list(a, r);

		on = syslook("chansend2", 1);
		argtype(on, t->type);	// any-1
		argtype(on, t->type);	// any-2
		r = nod(OCALL, on, r);
		walktype(r, top);
		break;
	}
	return r;

shape:
	fatal("chanop: %O", n->op);
	return N;
}

Type*
fixarray(Type *tm)
{
	Type *t;

	t = tm->type;
	if(t == T) {
		fatal("fixarray: t nil");
		return T;
	}

	if(t->etype != TDARRAY && t->etype != TARRAY) {
		fatal("fixarray: %lT not array", tm);
		return T;
	}

	if(t->type == T) {
		fatal("fixarray: array element type is nil");
		return T;
	}

	dowidth(t->type);

	return t;
}

Node*
arrayop(Node *n, int top)
{
	Node *r, *a;
	Type *t;
	Node *on;
	Iter save;

	r = n;
	switch(n->op) {
	default:
		fatal("darrayop: unknown op %O", n->op);

	case ONEW:
		// newarray(nel uint32, max uint32, width uint32) (ary *[]any)
		t = fixarray(n->type);

		a = nodintconst(t->type->width);	// width
		a = nod(OCONV, a, N);
		a->type = types[TUINT32];
		r = a;

		a = listfirst(&save, &n->left);		// max
		if(a == N)
			a = nodintconst(0);
		a = nod(OCONV, a, N);
		a->type = types[TUINT32];
		r = list(a, r);

		a = t->dbound;				// nel
		if(a == N)
			a = nodintconst(t->bound);
		a = nod(OCONV, a, N);
		a->type = types[TUINT32];
		r = list(a, r);

		on = syslook("newarray", 1);
		argtype(on, t->type);			// any-1
		r = nod(OCALL, on, r);

		walktype(r, top);
		if(t->etype == TARRAY) {
			// single case when we can convert a dynamic
			// array pointer to a static array pointer
			// saves making a sys function to alloc a static
			r = nod(OCONV, r, N);
			r->type = ptrto(t);
		}
		break;

	case OAS:
		// arrays2d(old *any, nel uint32) (ary *[]any)
		t = fixarray(n->right->type);

		a = nodintconst(t->bound);		// nel
		a = nod(OCONV, a, N);
		a->type = types[TUINT32];
		r = a;

		a = n->right;				// old
		r = list(a, r);

		on = syslook("arrays2d", 1);
		argtype(on, n->right->type->type);	// any-1
		argtype(on, t->type);			// any-2
		r = nod(OCALL, on, r);

		walktype(r, top);
		n->right = r;
		return n;

	case OSLICE:
		if(isptrto(n->left->type, TARRAY))
			goto slicestatic;

		// arrayslices(old *[]any, lb uint32, hb uint32, width uint32) (ary *[]any)
		t = fixarray(n->left->type);

		a = nodintconst(t->type->width);	// width
		a = nod(OCONV, a, N);
		a->type = types[TUINT32];
		r = a;

		a = nod(OCONV, n->right->right, N);	// hb
		a->type = types[TUINT32];
		r = list(a, r);

		a = nod(OCONV, n->right->left, N);	// lb
		a->type = types[TINT32];
		r = list(a, r);

		a = n->left;				// old
		r = list(a, r);

		on = syslook("arraysliced", 1);
		argtype(on, t->type);			// any-1
		argtype(on, t->type);			// any-2
		r = nod(OCALL, on, r);
		walktype(r, top);
		break;

	slicestatic:
		// arrayslices(old *any, nel uint32, lb uint32, hb uint32, width uint32) (ary *[]any)
		t = fixarray(n->left->type);

		a = nodintconst(t->type->width);	// width
		a = nod(OCONV, a, N);
		a->type = types[TUINT32];
		r = a;

		a = nod(OCONV, n->right->right, N);	// hb
		a->type = types[TUINT32];
		r = list(a, r);

		a = nod(OCONV, n->right->left, N);	// lb
		a->type = types[TINT32];
		r = list(a, r);

		a = nodintconst(t->bound);		// nel
		a = nod(OCONV, a, N);
		a->type = types[TINT32];
		r = list(a, r);

		a = n->left;				// old
		r = list(a, r);

		on = syslook("arrayslices", 1);
		argtype(on, t);				// any-1
		argtype(on, t->type);			// any-2
		r = nod(OCALL, on, r);
		walktype(r, top);
		break;
	}
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
isandss(Type *lt, Node *r)
{
	Type *rt;
	Node *n;
	int o;

	rt = r->type;
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

	return N;

ret:
	diagnamed(lt);
	diagnamed(rt);

	n = nod(o, r, N);
	n->type = lt;
	return n;
}

Node*
convas(Node *n)
{
	Node *l, *r;
	Type *lt, *rt;

	if(n->op != OAS)
		fatal("convas: not OAS %O", n->op);

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

	if(n->left->op == OSEND)
	if(n->left->type != T) {
		*n = *chanop(n, Elv);
		return n;
	}

	if(eqtype(lt, rt, 0))
		return n;

	r = isandss(lt, r);
	if(r != N) {
		n->right = r;
		walktype(n, Etop);
		return n;
	}

	if(isptrto(lt, TDARRAY) && isptrto(rt, TARRAY)) {
		if(!eqtype(lt->type->type, rt->type->type, 0))
			goto bad;
		*n = *arrayop(n, Etop);
		return n;
	}

	if(ascompat(lt, rt))
		return n;

bad:
	badtype(n->op, lt, rt);
	return n;
}

//void
//arrayconv(Type *t, Node *n)
//{
//	int c;
//	Iter save;
//	Node *l;
//
//	l = listfirst(&save, &n);
//	c = 0;
//
//loop:
//	if(l == N) {
//		if(t->bound == 0)
//			t->bound = c;
//		if(t->bound == 0 || t->bound < c)
//			yyerror("error with array convert bounds");
//		return;
//	}
//
//	c++;
//	walktype(l, Erv);
//	convlit(l, t->type);
//	if(!ascompat(l->type, t->type))
//		badtype(OARRAY, l->type, t->type);
//	l = listnext(&save);
//	goto loop;
//}

Node*
old2new(Node *n, Type *t)
{
	Node *l;

	if(n->op != ONAME && n->op != ONONAME) {
		yyerror("left side of := must be a name");
		return n;
	}
	l = newname(n->sym);
	dodclvar(l, t);
	return l;
}

Node*
colas(Node *nl, Node *nr)
{
	Iter savel, saver;
	Node *l, *r, *a, *n;
	Type *t;
	int cl, cr;

	/* nl is an expression list.
	 * nr is an expression list.
	 * return a newname-list from
	 * types derived from the rhs.
	 */
	n = N;
	cr = listcount(nr);
	cl = listcount(nl);
	if(cl != cr) {
		if(cr == 1)
			goto multi;
		goto badt;
	}

	l = listfirst(&savel, &nl);
	r = listfirst(&saver, &nr);

	while(l != N) {
		walktype(r, Erv);
		defaultlit(r);
		a = old2new(l, r->type);
		if(n == N)
			n = a;
		else
			n = list(n, a);

		l = listnext(&savel);
		r = listnext(&saver);
	}
	n = rev(n);
	return n;

multi:
	/*
	 * there is a list on the left
	 * and a mono on the right.
	 * go into the right to get
	 * individual types for the left.
	 */
	switch(nr->op) {
	default:
		goto badt;

	case OCALLMETH:
	case OCALLINTER:
	case OCALL:
		walktype(nr->left, Erv);
		t = nr->left->type;
		if(t == T || t->etype != TFUNC)
			goto badt;
		if(t->outtuple != cl)
			goto badt;

		l = listfirst(&savel, &nl);
		t = structfirst(&saver, getoutarg(t));
		while(l != N) {
			a = old2new(l, t->type);
			if(n == N)
				n = a;
			else
				n = list(n, a);
			l = listnext(&savel);
			t = structnext(&saver);
		}
		break;

	case OINDEX:
	case OINDEXPTR:
		// check if rhs is a map index.
		// if so, types are bool,maptype
		if(cl != 2)
			goto badt;
		walktype(nr->left, Elv);
		t = nr->left->type;
		if(t != T && isptr[t->etype])
			t = t->type;
		if(t == T || t->etype != TMAP)
			goto badt;

		a = old2new(nl->left, t->type);
		n = a;
		a = old2new(nl->right, types[TBOOL]);
		n = list(n, a);
		break;

	case ORECV:
		if(cl != 2)
			goto badt;
		walktype(nr->left, Erv);
		t = nr->left->type;
		if(!isptrto(t, TCHAN))
			goto badt;
		a = old2new(nl->left, t->type->type);
		n = a;
		a = old2new(nl->right, types[TBOOL]);
		n = list(n, a);
	}
	n = rev(n);
	return n;

badt:
	yyerror("shape error across :=");
	return nl;
}

/*
 * from ascompat[te]
 * evaluating actual function arguments.
 *	f(a,b)
 * if there is exactly one function expr,
 * then it is done first. otherwise must
 * make temp variables
 */
Node*
reorder1(Node *n)
{
	Iter save;
	Node *l, *r, *f, *a, *g;
	int c, t;

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
			f = list(g, f);
		r = list(f, r);
		return r;
	}
	if(l->ullman < UINF) {
		if(r == N)
			r = l;
		else
			r = list(l, r);
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
		g = list(a, g);

	// put normal arg assignment on list
	// with fncall replaced by tempname
	l->right = a->left;
	if(r == N)
		r = l;
	else
		r = list(l, r);

more:
	l = listnext(&save);
	goto loop2;
}

/*
 * from ascompat[et]
 *	a,b = f()
 * return of a multi.
 * there can be no function calls at all,
 * or they will over-write the return values.
 */
Node*
reorder2(Node *n)
{
	Iter save;
	Node *l;
	int c;

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

/*
 * from ascompat[ee]
 *	a,b = c,d
 * simultaneous assignment. there cannot
 * be later use of an earlier lvalue.
 */
int
vmatch2(Node *l, Node *r)
{

loop:
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
	if(vmatch2(l, r->right))
		return 1;
	r = r->left;
	goto loop;
}

int
vmatch1(Node *l, Node *r)
{

loop:
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
	if(vmatch1(l->right, r))
		return 1;
	l = l->left;
	goto loop;
}

Node*
reorder3(Node *n)
{
	Iter save1, save2;
	Node *l1, *l2, *q, *r;
	int c1, c2;

	r = N;

	l1 = listfirst(&save1, &n);
	c1 = 0;

	while(l1 != N) {
		l2 = listfirst(&save2, &n);
		c2 = 0;
		while(l2 != N) {
			if(c2 > c1) {
				if(vmatch1(l1->left, l2->right)) {
					q = nod(OXXX, N, N);
					tempname(q, l2->right->type);
					q = nod(OAS, l1->left, q);
					l1->left = q->right;
					if(r == N)
						r = q;
					else
						r = list(r, q);
					break;
				}
			}
			l2 = listnext(&save2);
			c2++;
		}
		l1 = listnext(&save1);
		c1++;
	}
	if(r == N)
		return n;

	q = N;
	l1 = listfirst(&save1, &n);
	while(l1 != N) {
		if(q == N)
			q = l1;
		else
			q = list(q, l1);
		l1 = listnext(&save1);
	}

	r = rev(r);
	l1 = listfirst(&save1, &r);
	while(l1 != N) {
		if(q == N)
			q = l1;
		else
			q = list(q, l1);
		l1 = listnext(&save1);
	}

	q = rev(q);
//dump("res", q);
	return q;
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
