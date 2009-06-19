// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"

static	Node*	curfn;

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
walkret(Node *n)
{

loop:
	if(n != N)
	switch(n->op) {
	case OLIST:
		if(n->right == N) {
			n = n->left;
			goto loop;
		}
		n = n->right;
		goto loop;

	// at this point, we have the last
	// statement of the function

	case OGOTO:
	case OPANIC:
	case ORETURN:
		return 0;
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
		dump(s, curfn->nbody);
	}
	if(curfn->type->outtuple)
		if(walkret(curfn->nbody))
			yyerror("function ends without a return statement");
	if(addtop != N) {
		dump("addtop", addtop);
		fatal("addtop in walk");
	}
	walkstate(curfn->nbody);
	if(debug['W']) {
		snprint(s, sizeof(s), "after walk %S", curfn->nname->sym);
		dump(s, curfn->nbody);
	}
	heapmoves();
	if(debug['W'] && curfn->enter != N) {
		snprint(s, sizeof(s), "enter %S", curfn->nname->sym);
		dump(s, curfn->enter);
	}
}

void
addtotop(Node *n)
{
	Node *l;

	while(addtop != N) {
		l = addtop;
		addtop = N;
		walktype(l, Etop);
		n->ninit = list(n->ninit, l);
	}
}

void
gettype(Node *n, Node *a)
{
	if(debug['W'])
		dump("\nbefore gettype", n);
	walktype(n, Erv);
	if(a == N && addtop != N)
		fatal("gettype: addtop");
	addtotop(a);
	if(debug['W'])
		dump("after gettype", n);
}

void
walkstate(Node *n)
{
	Node *more;

loop:
	if(n == N)
		return;

	more = N;
	switch(n->op) {
	case ONAME:	// one only; lineno isn't right for right now
	case OPACK:
	case OTYPE:
	case OLITERAL:
		break;
	default:
		lineno = n->lineno;
	}

	switch(n->op) {

	case OLIST:
		walkstate(n->left);
		more = n->right;
		break;

	default:
		if(n->op == ONAME)
			yyerror("walkstate: %S not a top level statement", n->sym);
		else
			yyerror("walkstate: %O not a top level statement", n->op);

	case OASOP:
	case OAS:
	case OCLOSE:
	case OCLOSED:
	case OCALLMETH:
	case OCALLINTER:
	case OCALL:
	case ODCL:
	case OSEND:
	case ORECV:
	case OPRINT:
	case OPRINTN:
	case OPANIC:
	case OPANICN:
	case OFOR:
	case OIF:
	case OSWITCH:
	case OSELECT:
	case OEMPTY:
	case OBREAK:
	case OCONTINUE:
	case OGOTO:
	case OLABEL:
	case OFALL:
	case OXCASE:
	case OCASE:
	case OXFALL:
	case ORETURN:
	case OPROC:
	case ODEFER:
		walktype(n, Etop);
		break;
	}

	addtotop(n);

	if(more != N) {
		n = more;
		goto loop;
	}
}

void
indir(Node *nl, Node *nr)
{
	if(nr != N && nl != nr)
		*nl = *nr;
}

void
implicitstar(Node **nn)
{
	Type *t;
	Node *n;

	// insert implicit * if needed
	n = *nn;
	t = n->type;
	if(t == T || !isptr[t->etype])
		return;
	t = t->type;
	if(t == T)
		return;
	switch(t->etype) {
	case TMAP:
	case TSTRING:
	case TARRAY:
	case TINTER:
		break;
	default:
		return;
	}
	n = nod(OIND, n, N);
	walktype(n, Elv);
	*nn = n;
}

void
walktype(Node *n, int top)
{
	Node *r, *l;
	Type *t;
	Sym *s;
	int et, cl, cr;
	int32 lno;

	if(n == N)
		return;
	lno = setlineno(n);

	/*
	 * walk the whole tree of the body of a function.
	 * the types expressions are calculated.
	 * compile-time constants are evaluated.
	 */

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
		dump("walk", n);
		fatal("walktype: switch 1 unknown op %N", n);
		goto ret;

	case ODCL:
		goto ret;

	case OLIST:
	case OKEY:
		walktype(n->left, top);
		n = n->right;
		goto loop;

	case OPRINT:
		if(top != Etop)
			goto nottop;
		walktype(n->left, Erv);
		indir(n, prcompat(n->left, 0));
		goto ret;

	case OPRINTN:
		if(top != Etop)
			goto nottop;
		walktype(n->left, Erv);
		indir(n, prcompat(n->left, 1));
		goto ret;

	case OPANIC:
		if(top != Etop)
			goto nottop;
		walktype(n->left, Erv);
		indir(n, list(prcompat(n->left, 0), nodpanic(n->lineno)));
		goto ret;

	case OPANICN:
		if(top != Etop)
			goto nottop;
		walktype(n->left, Erv);
		indir(n, list(prcompat(n->left, 2), nodpanic(n->lineno)));
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
		if(!(n->class & PHEAP) && n->class != PPARAMREF)
			n->addable = 1;
		if(n->type == T) {
			s = n->sym;
			if(s->undef == 0) {
				if(n->etype != 0)
					yyerror("walktype: %S must be called", s);
				else
					yyerror("walktype: %S undeclared", s);
				s->undef = 1;
			}
		}
		goto ret;

	case OFOR:
		if(top != Etop)
			goto nottop;
		walkstate(n->ninit);
		walkbool(n->ntest);
		walkstate(n->nincr);
		walkstate(n->nbody);
		goto ret;

	case OSWITCH:
		if(top != Etop)
			goto nottop;

		walkswitch(n);
		goto ret;

	case OSELECT:
		if(top != Etop)
			goto nottop;
		walkselect(n);
		goto ret;

	case OIF:
		if(top != Etop)
			goto nottop;
		walkstate(n->ninit);
		walkbool(n->ntest);
		walkstate(n->nbody);
		walkstate(n->nelse);
		goto ret;

	case ODEFER:
		hasdefer = 1;
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

		if(n->left == N)
			goto ret;

		walktype(n->left, Erv);
		defaultlit(n->left, T);

		t = n->left->type;
		if(t == T)
			goto ret;

		if(n->left->op == ODOTMETH)
			n->op = OCALLMETH;
		if(n->left->op == ODOTINTER)
			n->op = OCALLINTER;

		if(t->etype != TFUNC) {
			yyerror("call of a non-function: %T", t);
			goto ret;
		}

		dowidth(t);
		n->type = *getoutarg(t);
		switch(t->outtuple) {
		case 0:
			if(top == Erv) {
				yyerror("function requires a return type");
				n->type = types[TINT];
			}
			break;

		case 1:
			n->type = n->type->type->type;
			break;
		}

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
				Node *b;
				b = nodbool(0);
				r = ascompatte(n->op, getoutarg(t), &b, 0);
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

		addtop = list(addtop, n->ninit);
		n->ninit = N;

		l = n->left;
		r = n->right;
		walktype(l, Elv);
		if(l == N || r == N)
			goto ret;

		cl = listcount(l);
		cr = listcount(r);

		if(cl == cr) {
		simpleas:
			walktype(r, Erv);
			l = ascompatee(n->op, &n->left, &n->right);
			if(l != N)
				indir(n, reorder3(l));
			goto ret;
		}

		switch(r->op) {

		case OCALLMETH:
		case OCALLINTER:
		case OCALL:
			if(cr == 1) {
				// a,b,... = fn()
				walktype(r, Erv);
				if(r->type == T || r->type->etype != TSTRUCT)
					break;
				l = ascompatet(n->op, &n->left, &r->type, 0);
				if(l != N)
					indir(n, list(r, l));
				goto ret;
			}
			break;

		case OINDEX:
			if(cl == 2 && cr == 1) {
				// a,b = map[] - mapaccess2
				walktype(r->left, Erv);
				implicitstar(&r->left);
				if(!istype(r->left->type, TMAP))
					break;
				l = mapop(n, top);
				if(l == N)
					break;
				indir(n, l);
				goto ret;
			}
			break;

		case ORECV:
			if(cl == 2 && cr == 1) {
				// a,b = <chan - chanrecv2
				walktype(r->left, Erv);
				if(!istype(r->left->type, TCHAN))
					break;
				l = chanop(n, top);
				if(l == N)
					break;
				indir(n, l);
				goto ret;
			}
			break;

		case ODOTTYPE:
			if(cl == 2 && cr == 1) {
				// a,b = i.(T)
				walktype(r->left, Erv);
				if(r->left == N)
					break;
				et = ifaceas1(r->type, r->left->type, 1);
				switch(et) {
				case I2Isame:
				case E2Esame:
					n->right = nod(OLIST, r->left, nodbool(1));
					goto simpleas;
				case I2E:
					n->right = nod(OLIST, n->right, nodbool(1));
					goto simpleas;
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
				r = ifacecvt(r->type, r->left, et);
				l = ascompatet(n->op, &n->left, &r->type, 0);
				if(l != N)
					indir(n, list(r, l));
				goto ret;
			}
			break;
		}

		switch(l->op) {
		case OINDEX:
			if(cl == 1 && cr == 2) {
				// map[] = a,b - mapassign2
				if(!istype(l->left->type, TMAP))
					break;
				l = mapop(n, top);
				if(l == N)
					break;
				indir(n, l);
				goto ret;
			}
			break;
		}
		if(l->diag == 0) {
			l->diag = 1;
			yyerror("assignment count mismatch: %d = %d", cl, cr);
		}
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
		walkstate(n->right);
		goto ret;

	case OXFALL:
		if(top != Etop)
			goto nottop;
		yyerror("fallthrough statement out of place");
		n->op = OFALL;

	case OFALL:
	case OINDREG:
	case OEMPTY:
		goto ret;

	case OCONV:
	case ODOTTYPE:
		if(top != Erv)
			goto nottop;
		walkconv(n);
		goto ret;

	case OCOMPMAP:
	case OCOMPSLICE:
		goto ret;

	case OCOMPOS:
		t = n->type;
		if(t == T)
			goto ret;
		l = n->left;
		if(l == N)
			goto ret;

		switch(t->etype) {
		default:
			yyerror("invalid type for composite literal: %T", t);
			goto ret;

		case TSTRUCT:
			r = structlit(n, N);
			break;

		case TARRAY:
			r = arraylit(n, N);
			break;

		case TMAP:
			r = maplit(n, N);
			break;
		}
		indir(n, r);
//		walktype(l, Erv);
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
		evconst(n);
		if(n->op == OLITERAL)
			goto ret;
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
		if(l->op == OINDEX && istype(l->left->type, TMAP))
			indir(n, mapop(n, top));
		if(n->etype == OLSH || n->etype == ORSH)
			goto shft;
		goto com;

	case OLSH:
	case ORSH:
		if(top != Erv)
			goto nottop;
		walktype(n->left, Erv);

	shft:
		walktype(n->right, Erv);
		if(n->left == N || n->right == N)
			goto ret;
		evconst(n);
		if(n->op == OLITERAL)
			goto ret;
		// do NOT defaultlit n->left.
		// let parent defaultlit or convlit instead.
		defaultlit(n->right, types[TUINT]);
		if(n->left->type == T || n->right->type == T)
			goto ret;
		if(issigned[n->right->type->etype])
			goto badt;
		// check of n->left->type happens in second switch.
		break;

	case OMOD:
	case OAND:
	case OANDNOT:
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
		defaultlit2(n->left, n->right);
		if(n->left->type == T || n->right->type == T)
			goto ret;
		if(!eqtype(n->left->type, n->right->type))
			goto badt;

		switch(n->op) {
		case OANDNOT:
			n->op = OAND;
			n->right = nod(OCOM, n->right, N);
			n->right->type = n->right->left->type;
			break;

		case OASOP:
			if(n->etype == OANDNOT) {
				n->etype = OAND;
				n->right = nod(OCOM, n->right, N);
				n->right->type = n->right->left->type;
				break;
			}

		case OEQ:
		case ONE:
		case OLT:
		case OLE:
		case OGE:
		case OGT:
		case OADD:
			if(istype(n->left->type, TSTRING)) {
				indir(n, stringop(n, top));
				goto ret;
			}
			break;
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
		defaultlit(n->left, T);
		implicitstar(&n->left);
		t = n->left->type;
		if(t == T)
			goto ret;
		switch(t->etype) {
		default:
			goto badt;
		case TSTRING:
			if(isconst(n->left, CTSTR))
				nodconst(n, types[TINT], n->left->val.u.sval->len);
			break;
		case TMAP:
			break;
		case TARRAY:
			if(t->bound >= 0)
				nodconst(n, types[TINT], t->bound);
			break;
		}
		n->type = types[TINT];
		goto ret;

	case OCAP:
		if(top != Erv)
			goto nottop;
		walktype(n->left, Erv);
		defaultlit(n->left, T);
		implicitstar(&n->left);
		t = n->left->type;
		if(t == T)
			goto ret;
		switch(t->etype) {
		default:
			goto badt;
		case TARRAY:
			if(t->bound >= 0)
				nodconst(n, types[TINT], t->bound);
			break;
		}
		n->type = types[TINT];
		goto ret;

	case OINDEX:
		if(top == Etop)
			goto nottop;

		walktype(n->left, Erv);
		walktype(n->right, Erv);

		if(n->left == N || n->right == N)
			goto ret;

		defaultlit(n->left, T);
		implicitstar(&n->left);

		t = n->left->type;
		if(t == T)
			goto ret;

		switch(t->etype) {
		default:
			defaultlit(n->right, T);
			goto badt;

		case TSTRING:
			// right side must be an int
			if(top != Erv)
				goto nottop;
			defaultlit(n->right, types[TINT]);
			if(n->right->type == T)
				break;
			if(!isint[n->right->type->etype])
				goto badt;
			indir(n, stringop(n, top));
			break;

		case TMAP:
			// right side must be map type
			defaultlit(n->right, t->down);
			if(n->right->type == T)
				break;
			if(!eqtype(n->right->type, t->down))
				goto badt;
			n->type = t->type;
			if(top == Erv)
				indir(n, mapop(n, top));
			break;

		case TARRAY:
			// right side must be an int
			defaultlit(n->right, types[TINT]);
			if(n->right->type == T)
				break;
			if(!isint[n->right->type->etype])
				goto badt;
			n->type = t->type;
			break;
		}
		goto ret;

	case OCLOSE:
		if(top != Etop)
			goto nottop;
		walktype(n->left, Erv);		// chan
		indir(n, chanop(n, top));
		goto ret;

	case OCLOSED:
		if(top == Elv)
			goto nottop;
		walktype(n->left, Erv);		// chan
		indir(n, chanop(n, top));
		goto ret;

	case OSEND:
		if(top == Elv)
			goto nottop;
		walktype(n->left, Erv);	// chan
		walktype(n->right, Erv);	// e
		indir(n, chanop(n, top));
		goto ret;

	case ORECV:
		if(top == Elv)
			goto nottop;
		if(n->right == N) {
			walktype(n->left, Erv);		// chan
			indir(n, chanop(n, top));	// returns e blocking
			goto ret;
		}
		walktype(n->left, Elv);		// e
		walktype(n->right, Erv);	// chan
		indir(n, chanop(n, top));	// returns bool non-blocking
		goto ret;

	case OSLICE:
		if(top == Etop)
			goto nottop;

		walktype(n->left, top);
		walktype(n->right, Erv);
		if(n->left == N || n->right == N)
			goto ret;
		defaultlit(n->left, T);
		defaultlit(n->right->left, types[TUINT]);
		defaultlit(n->right->right, types[TUINT]);
		implicitstar(&n->left);
		t = n->left->type;
		if(t == T)
			goto ret;
		if(t->etype == TSTRING) {
			indir(n, stringop(n, top));
			goto ret;
		}
		if(t->etype == TARRAY) {
			indir(n, arrayop(n, top));
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
		defaultlit(n->left, T);
		walkdot(n);
		goto ret;

	case OADDR:
		if(top != Erv)
			goto nottop;
		defaultlit(n->left, T);
		if(n->left->op == OCOMPOS && n->left->type != T) {
			Node *nvar, *nas, *nstar;

			// turn &Point(1, 2) or &[]int(1, 2) or &[...]int(1, 2) into allocation.
			// initialize with
			//	nvar := new(*Point);
			//	*nvar = Point(1, 2);
			// and replace expression with nvar

			nvar = nod(OXXX, N, N);
			tempname(nvar, ptrto(n->left->type));

			nas = nod(OAS, nvar, callnew(n->left->type));
			walktype(nas, Etop);
			addtop = list(addtop, nas);

			nstar = nod(OIND, nvar, N);
			nstar->type = n->left->type;

			switch(n->left->type->etype) {
			case TSTRUCT:
				structlit(n->left, nstar);
				break;
			case TARRAY:
				arraylit(n->left, nstar);
				break;
			case TMAP:
				maplit(n->left, nstar);
				break;
			default:
				goto badlit;
			}

//			walktype(n->left->left, Erv);
			indir(n, nvar);
			goto ret;
		}

	badlit:
		if(istype(n->left->type, TFUNC) && n->left->class == PFUNC) {
			if(!n->diag) {
				n->diag = 1;
				yyerror("cannot take address of function");
			}
		}
		if(n->left == N)
			goto ret;
		walktype(n->left, Elv);
		t = n->left->type;
		if(t == T)
			goto ret;
		addrescapes(n->left);
		n->type = ptrto(t);
		goto ret;

	case OIND:
		if(top == Etop)
			goto nottop;
		if(top == Elv)	// even if n is lvalue, n->left is rvalue
			top = Erv;
		walktype(n->left, top);
		defaultlit(n->left, T);
		if(n->left == N)
			goto ret;
		t = n->left->type;
		if(t == T)
			goto ret;
		if(!isptr[t->etype])
			goto badt;
		n->type = t->type;
		goto ret;

	case OMAKE:
		if(top != Erv)
			goto nottop;
		l = n->left;
		if(l == N) {
			yyerror("missing argument to make");
			goto ret;
		}
		indir(n, makecompat(n));
		goto ret;

	case ONEW:
		if(top != Erv)
			goto nottop;
		l = n->left;
		if(l == N)
			yyerror("missing argument to new");
		else if(n->right != N)
			yyerror("too many arguments to new");
		else if(l->op != OTYPE)
			yyerror("argument to new must be type");
		else if((t = l->type) == T)
			;
		else
			indir(n, callnew(t));
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
		if(n->left->type == T)
			goto ret;
		et = n->left->type->etype;
		if(et != TBOOL)
			goto badt;
		t = types[TBOOL];
		break;

	case OEQ:
	case ONE:
		if(n->left->type == T)
			goto ret;
		et = n->left->type->etype;
		if(!okforeq[et] && !isslice(n->left->type))
			goto badt;
		if(isinter(n->left->type)) {
			indir(n, ifaceop(n));
			goto ret;
		}
		t = types[TBOOL];
		break;

	case OLT:
	case OLE:
	case OGE:
	case OGT:
		if(n->left->type == T)
			goto ret;
		et = n->left->type->etype;
		if(!okforadd[et] && et != TSTRING)
			goto badt;
		t = types[TBOOL];
		break;

	case OADD:
	case OSUB:
	case OMUL:
	case ODIV:
	case OPLUS:
		if(n->left->type == T)
			goto ret;
		et = n->left->type->etype;
		if(!okforadd[et])
			goto badt;
		break;

	case OMINUS:
		if(n->left->type == T)
			goto ret;
		et = n->left->type->etype;
		if(!okforadd[et])
			goto badt;
		if(isfloat[et]) {
			// TODO(rsc): Can do this more efficiently,
			// but OSUB is wrong.  Should be in back end anyway.
			indir(n, nod(OMUL, n->left, nodintconst(-1)));
			walktype(n, Erv);
			goto ret;
		}
		break;

	case OLSH:
	case ORSH:
	case OAND:
	case OANDNOT:
	case OOR:
	case OXOR:
	case OMOD:
	case OCOM:
		if(n->left->type == T)
			goto ret;
		et = n->left->type->etype;
		if(et != TIDEAL && !okforand[et])
			goto badt;
		break;
	}

	/*
	 * rewrite div and mod into function calls
	 * on 32-bit architectures.
	 */
	switch(n->op) {
	case ODIV:
	case OMOD:
		et = n->left->type->etype;
		if(widthptr > 4 || (et != TUINT64 && et != TINT64))
			break;
		if(et == TINT64)
			strcpy(namebuf, "int64");
		else
			strcpy(namebuf, "uint64");
		if(n->op == ODIV)
			strcat(namebuf, "div");
		else
			strcat(namebuf, "mod");
		l = syslook(namebuf, 0);
		n->left = nod(OCONV, n->left, N);
		n->left->type = types[et];
		n->right = nod(OCONV, n->right, N);
		n->right->type = types[et];
		r = nod(OCALL, l, list(n->left, n->right));
		r = nod(OCONV, r, N);
		r->type = n->left->left->type;
		walktype(r, Erv);
		indir(n, r);
		goto ret;

	case OASOP:
		et = n->left->type->etype;
		if(widthptr > 4 || (et != TUINT64 && et != TINT64))
			break;
		l = saferef(n->left);
		r = nod(OAS, l, nod(n->etype, l, n->right));
		walktype(r, Etop);
		indir(n, r);
		goto ret;
	}

	if(t == T)
		t = n->left->type;
	n->type = t;
	goto ret;

nottop:
	if(n->diag)
		goto ret;
	n->diag = 1;
	switch(top) {
	default:
		yyerror("didn't expect %O here", n->op);
		break;
	case Etop:
		yyerror("operation %O not allowed in statement context", n->op);
		break;
	case Elv:
		yyerror("operation %O not allowed in assignment context", n->op);
		break;
	case Erv:
		yyerror("operation %O not allowed in expression context", n->op);
		break;
	}
	goto ret;

badt:
	if(n->diag)
		goto ret;
	n->diag = 1;
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
	defaultlit(n, T);
	addtotop(n);
	if(n != N && n->type != T)
		if(!eqtype(n->type, types[TBOOL]))
			yyerror("IF and FOR require a boolean type");
}

void
walkconv(Node *n)
{
	int et;
	char *what;
	Type *t;
	Node *l;

	t = n->type;
	if(t == T)
		return;
	l = n->left;
	if(l == N)
		return;
	walktype(l, Erv);
	if(l->type == T)
		return;

	// if using .(T), interface assertion.
	if(n->op == ODOTTYPE) {
		// interface conversion
		defaultlit(l, T);
		if(!isinter(l->type))
			yyerror("type assertion requires interface on left, have %T", l->type);
		et = ifaceas1(t, l->type, 1);
		if(et == I2Isame || et == E2Esame) {
			n->op = OCONV;
			goto nop;
		}
		if(et != Inone) {
			indir(n, ifacecvt(t, l, et));
			return;
		}
		goto bad;
	}

	// otherwise, conversion.
	convlit1(l, t, 1);
	if(l->type == T)
		return;

	// no-op conversion
	if(cvttype(t, l->type) == 1) {
	nop:
		if(l->op != ONAME) {
			indir(n, l);
			n->type = t;
		}
		return;
	}

	// simple fix-float
	if(isint[l->type->etype] || isfloat[l->type->etype])
	if(isint[t->etype] || isfloat[t->etype]) {
		evconst(n);
		return;
	}

	// to string
	if(l->type != T)
	if(istype(t, TSTRING)) {
		et = l->type->etype;
		if(isint[et]) {
			indir(n, stringop(n, Erv));
			return;
		}

		// can convert []byte and *[10]byte
		if((isptr[et] && isfixedarray(l->type->type) && istype(l->type->type->type, TUINT8))
		|| (isslice(l->type) && istype(l->type->type, TUINT8))) {
			n->op = OARRAY;
			indir(n, stringop(n, Erv));
			return;
		}

		// can convert []int and *[10]int
		if((isptr[et] && isfixedarray(l->type->type) && istype(l->type->type->type, TINT))
		|| (isslice(l->type) && istype(l->type->type, TINT))) {
			n->op = OARRAY;
			indir(n, stringop(n, Erv));
			return;
		}
	}

	// convert dynamic to static generated by ONEW/OMAKE
	if(isfixedarray(t) && isslice(l->type))
		return;

	// convert static array to dynamic array
	if(isslice(t) && isptr[l->type->etype] && isfixedarray(l->type->type)) {
		if(eqtype(t->type->type, l->type->type->type->type)) {
			indir(n, arrayop(n, Erv));
			return;
		}
	}

	// convert to unsafe.pointer
	if(isptrto(n->type, TANY)) {
		if(isptr[l->type->etype])
			return;
		if(l->type->etype == TUINTPTR)
			return;
	}

	// convert from unsafe.pointer
	if(isptrto(l->type, TANY)) {
		if(isptr[t->etype])
			return;
		if(t->etype == TUINTPTR)
			return;
	}

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
	else
	if(n->left->op == OLIST)
		yyerror("invalid type for composite literal: %T", t);
}

Node*
selcase(Node *n, Node *var)
{
	Node *a, *r, *on, *c;
	Type *t;

	if(n->left == N)
		goto dflt;
	c = n->left;
	if(c->op == ORECV)
		goto recv;

	walktype(c->left, Erv);		// chan
	walktype(c->right, Erv);	// elem

	t = fixchan(c->left->type);
	if(t == T)
		return N;

	if(!(t->chan & Csend)) {
		yyerror("cannot send on %T", t);
		return N;
	}

	convlit(c->right, t->type);
	if(!ascompat(t->type, c->right->type)) {
		badtype(c->op, t->type, c->right->type);
		return N;
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
		return N;

	if(!(t->chan & Crecv)) {
		yyerror("cannot receive from %T", t);
		return N;
	}

	// selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
	on = syslook("selectrecv", 1);
	argtype(on, t->type);
	argtype(on, t->type);

	a = c->left;			// nil elem
	a = nod(OLITERAL, N, N);
	a->val.ctype = CTNIL;
	a->type = types[TNIL];

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
		return N;

	if(!(t->chan & Crecv)) {
		yyerror("cannot receive from %T", t);
		return N;
	}

	walktype(c->left, Elv);	// elem
	convlit(c->left, t->type);
	if(!ascompat(t->type, c->left->type)) {
		badtype(c->op, t->type, c->left->type);
		return N;
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
	goto out;

dflt:
	// selectdefault(sel *byte);
	on = syslook("selectdefault", 0);
	a = var;
	r = a;				// sel-var
	goto out;

out:
	a = nod(OCALL, on, r);
	r = nod(OIF, N, N);
	r->ntest = a;

	return r;
}

/*
 * enumerate the special cases
 * of the case statement:
 *	case v := <-chan		// select and switch
 *	case v := map[]			// switch
 *	case v := interface.(TYPE)	// switch
 */
Node*
selectas(Node *name, Node *expr)
{
	Node *a;
	Type *t;

	if(expr == N)
		goto bad;
	switch(expr->op) {
	default:
//dump("case", expr);
		goto bad;

	case ORECV:
		walktype(expr->left, Erv);
		t = expr->left->type;
		if(t == T)
			goto bad;
		if(t->etype != TCHAN)
			goto bad;
		t = t->type;
		break;

	case OINDEX:
		walktype(expr->left, Erv);
		walktype(expr->right, Erv);
		t = expr->left->type;
		if(t == T)
			goto bad;
		if(t->etype != TMAP)
			goto bad;
		t = t->type;
		break;

	case ODOTTYPE:
		walktype(expr->left, Erv);
		t = expr->left->type;
		if(t == T)
			goto bad;
		if(t->etype != TINTER)
			goto bad;
		t = expr->type;
		break;
	}
	a = old2new(name, t);
	return a;

bad:
	return name;
}

void
walkselect(Node *sel)
{
	Iter iter, iter1;
	Node *n, *n1, *l, *oc, *on, *r;
	Node *var, *bod, *nbod, *res, *def;
	int count, op;
	int32 lno;

	lno = setlineno(sel);

	// generate sel-struct
	var = nod(OXXX, N, N);
	tempname(var, ptrto(types[TUINT8]));

	n = listfirst(&iter, &sel->left);
	if(n == N || n->op == OEMPTY) {
		yyerror("empty select");
		return;
	}

	count = 0;	// number of cases
	res = N;	// entire select body
	bod = N;	// body of each case
	oc = N;		// last case
	def = N;	// default case

	for(; n!=N; n=listnext(&iter)) {
		setlineno(n);
		if(n->op != OXCASE)
			fatal("walkselect %O", n->op);

		count++;
		if(n->left == N) {
			op = ORECV;	// actual value not used
			if(def != N)
				yyerror("repeated default; first at %L", def->lineno);
			def = n;
		} else
			op = n->left->op;

		nbod = N;
		switch(op) {
		default:
			yyerror("select cases must be send, recv or default");
			continue;

		case OAS:
			// convert new syntax (a=recv(chan)) to (recv(a,chan))
			l = n->left;
			if(l->right == N || l->right->op != ORECV) {
				yyerror("select cases must be send, recv or default");
				break;
			}
			r = l->right;	// rcv
			r->right = r->left;
			r->left = l->left;
			n->left = r;

			// convert case x := foo: body
			// to case tmp := foo: x := tmp; body.
			// if x escapes and must be allocated
			// on the heap, this delays the allocation
			// until after the select has chosen this branch.
			if(n->ninit != N && n->ninit->op == ODCL) {
				on = nod(OXXX, N, N);
				tempname(on, l->left->type);
				on->sym = lookup("!tmpselect!");
				r->left = on;
				nbod = nod(OAS, l->left, on);
				nbod->ninit = n->ninit;
				n->ninit = N;
			}
			break;

		case OSEND:
		case ORECV:
			break;
		}

		for(n1 = listfirst(&iter1, &n->nbody); n1 != N; n1 = listnext(&iter1))
			nbod = list(nbod, n1);
		nbod = list(nbod, nod(OBREAK, N, N));
		n->nbody = N;

		oc = selcase(n, var);
		if(oc != N) {
			oc->nbody = rev(nbod);
			res = list(res, oc);
		}
	}
	setlineno(sel);

	// selectgo(sel *byte);
	on = syslook("selectgo", 0);
	r = nod(OCALL, on, var);		// sel-var
	res = list(res, r);

	// newselect(size uint32) (sel *byte);
	on = syslook("newselect", 0);

	r = nod(OXXX, N, N);
	nodconst(r, types[TINT], count);	// count
	r = nod(OCALL, on, r);
	r = nod(OAS, var, r);

	sel->ninit = r;
	sel->nbody = rev(res);
	sel->left = N;

	walkstate(sel->ninit);
	walkstate(sel->nbody);

//dump("sel", sel);

	lineno = lno;
}

Type*
lookdot1(Sym *s, Type *t, Type *f)
{
	Type *r;

	r = T;
	for(; f!=T; f=f->down) {
		if(f->sym == S)
			continue;
		if(f->sym != s)
			continue;
		if(r != T) {
			yyerror("ambiguous DOT reference %T.%S", t, s);
			break;
		}
		r = f;
	}
	return r;
}

int
lookdot(Node *n, Type *t)
{
	Type *f1, *f2, *tt, *rcvr;
	Sym *s;

	s = n->right->sym;

	f1 = T;
	if(t->etype == TSTRUCT || t->etype == TINTER)
		f1 = lookdot1(s, t, t->type);

	f2 = methtype(n->left->type);
	if(f2 != T)
		f2 = lookdot1(s, f2, f2->method);

	if(f1 != T) {
		if(f2 != T)
			yyerror("ambiguous DOT reference %S as both field and method",
				n->right->sym);
		n->right = f1->nname;		// substitute real name
		n->xoffset = f1->width;
		n->type = f1->type;
		if(t->etype == TINTER) {
			implicitstar(&n->left);
			n->op = ODOTINTER;
		}
		return 1;
	}

	if(f2 != T) {
		tt = n->left->type;
		rcvr = getthisx(f2->type)->type->type;
		if(!eqtype(rcvr, tt)) {
			if(rcvr->etype == tptr && eqtype(rcvr->type, tt)) {
				walktype(n->left, Elv);
				addrescapes(n->left);
				n->left = nod(OADDR, n->left, N);
				n->left->type = ptrto(tt);
			} else if(tt->etype == tptr && eqtype(tt->type, rcvr)) {
				n->left = nod(OIND, n->left, N);
				n->left->type = tt->type;
			} else {
				// method is attached to wrong type?
				fatal("method mismatch: %T for %T", rcvr, tt);
			}
		}
		n->right = methodname(n->right, n->left->type);
		n->xoffset = f2->width;
		n->type = f2->type;
		n->op = ODOTMETH;
		return 1;
	}

	return 0;
}

void
walkdot(Node *n)
{
	Type *t;

	addtop = list(addtop, n->ninit);
	n->ninit = N;

	if(n->left == N || n->right == N)
		return;
	switch(n->op) {
	case ODOTINTER:
	case ODOTMETH:
		return;	// already done
	}

	walktype(n->left, Erv);
	if(n->right->op != ONAME) {
		yyerror("rhs of . must be a name");
		return;
	}

	t = n->left->type;
	if(t == T)
		return;

	// as a structure field or pointer to structure field
	if(isptr[t->etype]) {
		t = t->type;
		if(t == T)
			return;
		n->op = ODOTPTR;
	}

	if(!lookdot(n, t))
		yyerror("undefined DOT %S on %T", n->right->sym, n->left->type);
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
		// cannot happen: caller checked that lists had same length
		if(l != r)
			yyerror("error in shape across %O", op);
		return rev(nn);
	}

	convlit(r, l->type);
	if(!ascompat(l->type, r->type)) {
		badtype(op, l->type, r->type);
		return N;
	}
	if(l->op == ONAME && l->class == PFUNC)
		yyerror("cannot assign to function");

	a = nod(OAS, l, r);
	a = convas(a);
	nn = list(a, nn);

	l = listnext(&savel);
	r = listnext(&saver);
	goto loop;
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

Node*
ascompatet(int op, Node **nl, Type **nr, int fp)
{
	Node *l, *nn, *mm, *tmp, *a;
	Type *r;
	Iter savel, saver;
	int ucount;

	/*
	 * check assign type list to
	 * a expression list. called in
	 *	expr-list = func()
	 */
	l = listfirst(&savel, nl);
	r = structfirst(&saver, nr);
	nn = N;
	mm = N;
	ucount = 0;

loop:
	if(l == N || r == T) {
		if(l != N || r != T)
			yyerror("assignment count mismatch: %d = %d",
				listcount(*nl), structcount(*nr));
		if(ucount)
			yyerror("reorder2: too many function calls evaluating parameters");
		if(mm != N) {
			mm = rev(mm);
			for(l=listfirst(&savel, &mm); l!=N; l=listnext(&savel))
				nn = list(nn, l);
		}
		return rev(nn);
	}

	if(!ascompat(l->type, r->type)) {
		badtype(op, l->type, r->type);
		return N;
	}

	// any lv that causes a fn call must be
	// deferred until all the return arguments
	// have been pulled from the output arguments
	if(fncall(l, r->type)) {
		tmp = nod(OXXX, N, N);
		tempname(tmp, r->type);
		a = nod(OAS, l, tmp);
		a = convas(a);
		mm = list(mm, a);
		l = tmp;
	}

	a = nod(OAS, l, nodarg(r, fp));
	a = convas(a);
	ullmancalc(a);
	if(a->ullman >= UINF)
		ucount++;
	nn = list(nn, a);

	l = listnext(&savel);
	r = structnext(&saver);

	goto loop;
}

/*
 * make a tsig for the structure
 * carrying the ... arguments
 */
Type*
sigtype(Type *st)
{
	Dcl *x;
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

	// record internal type for signature generation
	x = mal(sizeof(*x));
	x->op = OTYPE;
	x->dsym = s;
	x->dtype = t;
	x->forw = signatlist;
	x->block = block;
	signatlist = x;

	return t;
}

/*
 * package all the arguments that
 * match a ... parameter into an
 * automatic structure.
 * then call the ... arg (interface)
 * with a pointer to the structure.
 */
Node*
mkdotargs(Node *r, Node *rr, Iter *saver, Node *nn, Type *l, int fp)
{
	Type *t, *st, *ft;
	Node *a, *n, *var;
	Iter saven;

	n = N;			// list of assignments

	st = typ(TSTRUCT);	// generated structure
	ft = T;			// last field
	while(r != N) {
		if(r->op == OLITERAL && r->val.ctype == CTNIL) {
			if(r->type == T || r->type->etype == TNIL) {
				yyerror("inappropriate use of nil in ... argument");
				return N;
			}
		}
		defaultlit(r, T);
		if(r->type == T)	// type check failed
			return N;

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
		if(rr != N) {
			r = rr;
			rr = N;
		} else
			r = listnext(saver);
	}

	// make a named type for the struct
	st = sigtype(st);
	dowidth(st);

	// now we have the size, make the struct
	var = nod(OXXX, N, N);
	tempname(var, st);
	var->sym = lookup(".ddd");

	// assign the fields to the struct.
	// use addtop so that reorder1 doesn't reorder
	// these assignments after the interface conversion
	// below.
	n = rev(n);
	r = listfirst(&saven, &n);
	t = st->type;
	while(r != N) {
		r->left = nod(OXXX, N, N);
		*r->left = *var;
		r->left->type = r->right->type;
		r->left->xoffset += t->width;
		addtop = list(addtop, r);
		r = listnext(&saven);
		t = t->down;
	}

	// last thing is to put assignment
	// of the structure to the DDD parameter
	a = nod(OAS, nodarg(l, fp), var);
	nn = list(convas(a), nn);

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
dumpnodetypes(Node **nr, char *what)
{
	int first;
	Node *r;
	Iter saver;

	print("\t");
	first = 1;
	for(r = listfirst(&saver, nr); r != N; r = listnext(&saver)) {
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
Node*
ascompatte(int op, Type **nl, Node **nr, int fp)
{
	Type *l, *ll;
	Node *r, *rr, *nn, *a;
	Iter savel, saver, peekl, peekr;

	l = structfirst(&savel, nl);
	r = listfirst(&saver, nr);
	nn = N;

	// 1 to many
	peekl = savel;
	peekr = saver;
	if(l != T && r != N
	&& structnext(&peekl) != T
	&& listnext(&peekr) == N
	&& eqtypenoname(r->type, *nl)) {
		// TODO(rsc): clumsy check for differently aligned structs.
		// need to handle eventually, but this keeps us
		// from inserting bugs
		if(r->type->width != (*nl)->width) {
			fprint(2, "oops: %T %d %T %d\n", r->type, r->type->width, (*nl), (*nl)->width);
			yyerror("misaligned multiple return (6g's fault)");
		}
		a = nodarg(*nl, fp);
		a->type = r->type;
		return convas(nod(OAS, a, r));
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
		rr = listnext(&saver);
		if(r != N && rr == N && isddd(r->type)) {
			a = nod(OAS, nodarg(l, fp), r);
			a = convas(a);
			nn = list(a, nn);
			return rev(nn);
		}

		// normal case -- make a structure of all
		// remaining arguments and pass a pointer to
		// it to the ddd parameter (empty interface)
		nn = mkdotargs(r, rr, &saver, nn, l, fp);

		return rev(nn);
	}

	if(l == T || r == N) {
		if(l != T || r != N) {
			if(l != T)
				yyerror("not enough arguments to %O", op);
			else
				yyerror("too many arguments to %O", op);
			dumptypes(nl, "expected");
			dumpnodetypes(nr, "given");
		}
		return rev(nn);
	}
	convlit(r, l->type);
	if(!ascompat(l->type, r->type)) {
		badtype(op, l->type, r->type);
		return N;
	}


	a = nod(OAS, nodarg(l, fp), r);
	a = convas(a);
	nn = list(a, nn);

	l = structnext(&savel);
	r = listnext(&saver);

	goto loop;
}

/*
 * do the export rules allow writing to this type?
 * cannot be implicitly assigning to any type with
 * an unavailable field.
 */
int
exportasok(Type *t)
{
	Type *f;
	Sym *s;

	if(t == T)
		return 1;
	switch(t->etype) {
	default:
		// most types can't contain others; they're all fine.
		break;
	case TSTRUCT:
		for(f=t->type; f; f=f->down) {
			if(f->etype != TFIELD)
				fatal("structas: not field");
			s = f->sym;
			// s == nil doesn't happen for embedded fields (they get the type symbol).
			// it only happens for fields in a ... struct.
			if(s != nil && !exportname(s->name) && strcmp(package, s->package) != 0) {
				yyerror("implicit assignment of %T field '%s'", t, s->name);
				return 0;
			}
			if(!exportasok(f->type))
				return 0;
		}
		break;

	case TARRAY:
		if(t->bound < 0)	// slices are pointers; that's fine
			break;
		if(!exportasok(t->type))
			return 0;
		break;
	}
	return 1;
}

/*
 * can we assign var of type src to var of type dst?
 * return 0 if not, 1 if conversion is trivial, 2 if conversion is non-trivial.
 */
int
ascompat(Type *dst, Type *src)
{
	if(eqtype(dst, src)) {
		exportasok(src);
		return 1;
	}

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
//	fmt = 0: print
//	fmt = 1: println
//	fmt = 2: panicln (like println but no trailing newline)
Node*
prcompat(Node *n, int fmt)
{
	Node *l, *r;
	Node *on;
	Type *t;
	Iter save;
	int notfirst, et;

	r = N;
	l = listfirst(&save, &n);
	notfirst = 0;

loop:
	if(l == N) {
		if(fmt == 1) {
			on = syslook("printnl", 0);
			r = list(r, nod(OCALL, on, N));
		}
		walktype(r, Etop);
		return r;
	}

	if(notfirst) {
		on = syslook("printsp", 0);
		r = list(r, nod(OCALL, on, N));
	}

	walktype(l, Erv);
	if(l->op == OLITERAL) {
		switch(l->val.ctype) {
		case CTINT:
			defaultlit(l, types[TINT64]);
			break;
		case CTFLT:
			defaultlit(l, types[TFLOAT64]);
			break;
		}
	}
	if(l->type == T)
		goto out;

	et = l->type->etype;
	if(isinter(l->type)) {
		if(isnilinter(l->type))
			on = syslook("printeface", 1);
		else
			on = syslook("printiface", 1);
		argtype(on, l->type);		// any-1
	} else if(isptr[et] || et == TCHAN || et == TMAP || et == TFUNC) {
		on = syslook("printpointer", 1);
		argtype(on, l->type);	// any-1
	} else if(isslice(l->type)) {
		on = syslook("printarray", 1);
		argtype(on, l->type);	// any-1
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
		badtype(OPRINT, l->type, T);
		l = listnext(&save);
		goto loop;
	}

	t = *getinarg(on->type);
	if(t != nil)
		t = t->type;
	if(t != nil)
		t = t->type;

	if(!eqtype(t, l->type)) {
		l = nod(OCONV, l, N);
		l->type = t;
	}

	r = list(r, nod(OCALL, on, l));

out:
	notfirst = fmt;
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
makecompat(Node *n)
{
	Type *t;
	Node *l, *r;

	l = n->left;
	r = N;
	if(l->op == OLIST) {
		r = l->right;
		l = l->left;
	}
	if(l->op != OTYPE) {
		yyerror("cannot make(expr)");
		return n;
	}
	t = l->type;
	n->type = t;
	n->left = r;
	n->right = N;

	if(t != T)
	switch(t->etype) {
	case TARRAY:
		if(!isslice(t))
			goto bad;
		return arrayop(n, Erv);
	case TMAP:
		return mapop(n, Erv);
	case TCHAN:
		return chanop(n, Erv);
	}

bad:
	if(!n->diag) {
		n->diag = 1;
		yyerror("cannot make(%T)", t);
	}
	return n;
}

Node*
callnew(Type *t)
{
	Node *r, *on;

	dowidth(t);
	on = syslook("mal", 1);
	argtype(on, t);
	r = nodintconst(t->width);
	r = nod(OCALL, on, r);
	walktype(r, Erv);
	return r;
}

Node*
stringop(Node *n, int top)
{
	Node *r, *c, *on;

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
		r = nod(OCONV, n->left, N);
		r->type = types[TSTRING];
		c = nod(OCONV, n->right, N);
		c->type = types[TSTRING];
		r = nod(OCALL, on, list(r, c));
		c = nodintconst(0);
		r = nod(n->op, r, c);
		break;

	case OADD:
		// sys_catstring(s1, s2)
		on = syslook("catstring", 0);
		r = nod(OCONV, n->left, N);
		r->type = types[TSTRING];
		c = nod(OCONV, n->right, N);
		c->type = types[TSTRING];
		r = nod(OCALL, on, list(r, c));
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
			on = syslook("catstring", 0);
			r = nod(OCONV, n->left, N);
			r->type = types[TSTRING];
			c = nod(OCONV, n->right, N);
			c->type = types[TSTRING];
			r = nod(OCALL, on, list(r, c));
			r = nod(OAS, n->left, r);
			break;
		}
		break;

	case OSLICE:
		// sys_slicestring(s, lb, hb)
		r = nod(OCONV, n->right->left, N);
		r->type = types[TINT];

		c = nod(OCONV, n->right->right, N);
		c->type = types[TINT];

		r = list(r, c);
		r = list(n->left, r);
		on = syslook("slicestring", 0);
		r = nod(OCALL, on, r);
		break;

	case OINDEX:
		// sys_indexstring(s, i)
		r = nod(OCONV, n->right, N);
		r->type = types[TINT];
		r = list(n->left, r);
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
		r = n->left;
		if(r->type != T && r->type->type != T) {
			if(istype(r->type->type, TINT) || istype(r->type->type->type, TINT)) {
				// arraystring([]byte) string;
				on = syslook("arraystringi", 0);
				r = nod(OCALL, on, r);
				break;
			}
		}

		// arraystring([]byte) string;
		on = syslook("arraystring", 0);
		r = nod(OCALL, on, r);
		break;
	}

	walktype(r, top);
	return r;
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
mapop(Node *n, int top)
{
	Node *r, *a;
	Type *t;
	Node *on;
	int cl, cr;

	r = n;
	switch(n->op) {
	default:
		fatal("mapop: unknown op %O", n->op);

	case OMAKE:
		cl = listcount(n->left);
		if(cl > 1)
			yyerror("too many arguments to make map");

		if(top != Erv)
			goto nottop;

		// newmap(keysize int, valsize int,
		//	keyalg int, valalg int,
		//	hint int) (hmap map[any-1]any-2);

		t = fixmap(n->type);
		if(t == T)
			break;

		a = n->left;				// hint
		if(cl != 1)
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
		// mapaccess1(hmap map[any]any, key any) (val any);

		t = fixmap(n->left->type);
		if(t == T)
			break;

		convlit(n->right, t->down);

		if(!eqtype(n->right->type, t->down)) {
			badtype(n->op, n->right->type, t->down);
			break;
		}

		a = n->right;				// key
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

		// mapassign1(hmap map[any-1]any-2, key any-3, val any-4);
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
		walktype(r, Etop);
		break;

	assign2:
		// mapassign2(hmap map[any]any, key any, val any, pres bool);
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
		walktype(r, Etop);
		break;

	access2:
		// mapaccess2(hmap map[any-1]any-2, key any-3) (val-4 any, pres bool);

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

		t = n->left->left->type;
		a = nod(OXXX, N, N);
		tempname(a, t->down);			// tmpi
		r = nod(OAS, a, n->left->right);	// tmpi := index
		n->left->right = a;			// m[tmpi]

		a = nod(OXXX, N, N);
		indir(a, n->left);			// copy of map[tmpi]
		a = nod(n->etype, a, n->right);		// m[tmpi] op right
		a = nod(OAS, n->left, a);		// map[tmpi] = map[tmpi] op right
		r = nod(OLIST, r, a);
		walktype(r, Etop);
		break;
	}
	return r;

shape:
	dump("shape", n);
	fatal("mapop: cl=%d cr=%d, %O", top, n->op);
	return N;

nottop:
	yyerror("didn't expect %O here", n->op);
	return N;
}

Node*
chanop(Node *n, int top)
{
	Node *r, *a;
	Type *t;
	Node *on;
	int cl, cr;

//dump("chanop", n);

	r = n;
	switch(n->op) {
	default:
		fatal("chanop: unknown op %O", n->op);

	case OCLOSE:
		// closechan(hchan *chan any);
		t = fixchan(n->left->type);
		if(t == T)
			break;

		a = n->left;			// chan
		r = a;

		on = syslook("closechan", 1);
		argtype(on, t);	// any-1

		r = nod(OCALL, on, r);
		walktype(r, top);
		r->type = n->type;
		break;

	case OCLOSED:
		// closedchan(hchan *chan any) bool;
		t = fixchan(n->left->type);
		if(t == T)
			break;

		a = n->left;			// chan
		r = a;

		on = syslook("closedchan", 1);
		argtype(on, t);	// any-1

		r = nod(OCALL, on, r);
		walktype(r, top);
		n->type = r->type;
		break;

	case OMAKE:
		cl = listcount(n->left);
		if(cl > 1)
			yyerror("too many arguments to make chan");

		// newchan(elemsize int, elemalg int,
		//	hint int) (hmap *chan[any-1]);

		t = fixchan(n->type);
		if(t == T)
			break;

		a = nodintconst(0);
		if(cl == 1) {
			// async buf size
			a = nod(OCONV, n->left, N);
			a->type = types[TINT];
		}

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
		defaultlit(n->right->left, T);
		t = fixchan(n->right->left->type);
		if(t == T)
			break;

		if(!(t->chan & Crecv)) {
			yyerror("cannot receive from %T", t);
			break;
		}

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
		defaultlit(n->left, T);
		t = fixchan(n->left->type);
		if(t == T)
			break;

		if(!(t->chan & Crecv)) {
			yyerror("cannot receive from %T", t);
			break;
		}

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
			a->type = types[TNIL];
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
		if(!(t->chan & Csend)) {
			yyerror("cannot send to %T", t);
			break;
		}

		if(top != Etop)
			goto send2;

		// chansend1(hchan *chan any, elem any);
		a = n->right;			// e
		r = a;
		a = n->left;			// chan
		r = list(a, r);

		on = syslook("chansend1", 1);
		argtype(on, t->type);	// any-1
		argtype(on, t->type);	// any-2
		r = nod(OCALL, on, r);
		walktype(r, Etop);
		break;

	send2:
		// chansend2(hchan *chan any, val any) (pres bool);
		a = n->right;			// e
		r = a;
		a = n->left;			// chan
		r = list(a, r);

		on = syslook("chansend2", 1);
		argtype(on, t->type);	// any-1
		argtype(on, t->type);	// any-2
		r = nod(OCALL, on, r);
		walktype(r, Etop);
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

Node*
arrayop(Node *n, int top)
{
	Node *r, *a;
	Type *t, *tl;
	Node *on;
	Iter save;
	int cl;

	r = n;
	switch(n->op) {
	default:
		fatal("darrayop: unknown op %O", n->op);

	case OCONV:
		// arrays2d(old *any, nel int) (ary []any)
		if(n->left->type == T || !isptr[n->left->type->etype])
			break;
		t = fixarray(n->left->type->type);
		tl = fixarray(n->type);
		if(t == T || tl == T)
			break;

		a = nodintconst(t->bound);		// nel
		a = nod(OCONV, a, N);
		a->type = types[TINT];
		r = a;

		r = list(n->left, r);				// old

		on = syslook("arrays2d", 1);
		argtype(on, t);				// any-1
		argtype(on, tl->type);			// any-2
		r = nod(OCALL, on, r);
		n->left = r;
		walktype(n, top);
		return n;

	case OAS:
		r = nod(OCONV, n->right, N);
		r->type = n->left->type;
		n->right = arrayop(r, Erv);
		return n;

	case OMAKE:
		cl = listcount(n->left);
		if(cl > 2)
			yyerror("too many arguments to make array");

		// newarray(nel int, max int, width int) (ary []any)
		t = fixarray(n->type);
		if(t == T)
			break;

		a = nodintconst(t->type->width);	// width
		a = nod(OCONV, a, N);
		a->type = types[TINT];
		r = a;

		a = listfirst(&save, &n->left);		// max
		a = listnext(&save);
		if(a == N)
			a = nodintconst(0);
		a = nod(OCONV, a, N);
		a->type = types[TINT];
		r = list(a, r);

		a = listfirst(&save, &n->left);		// nel
		if(a == N) {
			yyerror("new slice must have size");
			a = nodintconst(1);
		}
		a = nod(OCONV, a, N);
		a->type = types[TINT];
		r = list(a, r);

		on = syslook("newarray", 1);
		argtype(on, t->type);			// any-1
		r = nod(OCALL, on, r);

		walktype(r, top);
		r->type = t;	// if t had a name, going through newarray lost it
		break;

	case OSLICE:
		// arrayslices(old any, nel int, lb int, hb int, width int) (ary []any)
		// arraysliced(old []any, lb int, hb int, width int) (ary []any)

		t = fixarray(n->left->type);
		if(t == T)
			break;

		a = nodintconst(t->type->width);	// width
		a = nod(OCONV, a, N);
		a->type = types[TINT];
		r = a;

		a = nod(OCONV, n->right->right, N);	// hb
		a->type = types[TINT];
		r = list(a, r);

		a = nod(OCONV, n->right->left, N);	// lb
		a->type = types[TINT];
		r = list(a, r);

		if(t->bound >= 0) {
			// static slice
			a = nodintconst(t->bound);		// nel
			a = nod(OCONV, a, N);
			a->type = types[TINT];
			r = list(a, r);

			a = nod(OADDR, n->left, N);		// old
			r = list(a, r);

			on = syslook("arrayslices", 1);
			argtype(on, t);				// any-1
			argtype(on, t->type);			// any-2
		} else {
			// dynamic slice
			a = n->left;				// old
			r = list(a, r);

			on = syslook("arraysliced", 1);
			argtype(on, t->type);			// any-1
			argtype(on, t->type);			// any-2
		}
		r = nod(OCALL, on, r);
		walktype(r, top);
		break;
	}
	return r;
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
ifacecvt(Type *tl, Node *n, int et)
{
	Type *tr;
	Node *r, *a, *on;

	tr = n->type;

	switch(et) {
	default:
		fatal("ifacecvt: unknown op %d\n", et);

	case T2I:
		// ifaceT2I(sigi *byte, sigt *byte, elem any) (ret any);

		a = n;				// elem
		r = a;

		a = nod(OADDR, signame(tr), N);	// sigt
		r = list(a, r);

		a = nod(OADDR, signame(tl), N);	// sigi
		r = list(a, r);

		on = syslook("ifaceT2I", 1);
		argtype(on, tr);
		argtype(on, tl);
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
		a = n;				// interface
		r = a;

		a = nod(OADDR, signame(tl), N);	// sigi or sigt
		r = list(a, r);

		on = syslook(ifacename[et], 1);
		argtype(on, tr);
		argtype(on, tl);
		break;

	case I2E:
		// TODO(rsc): Should do this in back end, without a call.
		// ifaceI2E(elem any) (ret any);
		a = n;				// interface
		r = a;
		on = syslook("ifaceI2E", 1);
		argtype(on, tr);
		argtype(on, tl);
		break;

	case T2E:
		// TODO(rsc): Should do this in back end for pointer case, without a call.
		// ifaceT2E(sigt *byte, elem any) (ret any);
		a = n;				// elem
		r = a;

		a = nod(OADDR, signame(tr), N);	// sigt
		r = list(a, r);

		on = syslook("ifaceT2E", 1);
		argtype(on, tr);
		argtype(on, tl);
		break;
	}

	r = nod(OCALL, on, r);
	walktype(r, Erv);
	return r;
}

Node*
ifaceop(Node *n)
{
	Node *r, *a, *on;

	switch(n->op) {
	default:
		fatal("ifaceop %O", n->op);

	case OEQ:
	case ONE:
		// ifaceeq(i1 any-1, i2 any-2) (ret bool);
		a = n->right;				// i2
		r = a;

		a = n->left;				// i1
		r = list(a, r);

		if(!eqtype(n->left->type, n->right->type))
			fatal("ifaceop %O %T %T", n->op, n->left->type, n->right->type);
		if(isnilinter(n->left->type))
			on = syslook("efaceeq", 1);
		else
			on = syslook("ifaceeq", 1);
		argtype(on, n->right->type);
		argtype(on, n->left->type);

		r = nod(OCALL, on, r);
		if(n->op == ONE)
			r = nod(ONOT, r, N);

		walktype(r, Erv);
		return r;
	}

	r = nod(OCALL, on, r);
	walktype(r, Erv);
	return r;
}

Node*
convas(Node *n)
{
	Node *l, *r;
	Type *lt, *rt;
	int et;

	if(n->op != OAS)
		fatal("convas: not OAS %O", n->op);

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

	if(n->left->op == OINDEX)
	if(istype(n->left->left->type, TMAP)) {
		indir(n, mapop(n, Elv));
		goto out;
	}

	if(n->left->op == OSEND)
	if(n->left->type != T) {
		indir(n, chanop(n, Elv));
		goto out;
	}

	if(eqtype(lt, rt))
		goto out;

	et = ifaceas(lt, rt, 0);
	if(et != Inone) {
		n->right = ifacecvt(lt, r, et);
		goto out;
	}

	if(isslice(lt) && isptr[rt->etype] && isfixedarray(rt->type)) {
		if(!eqtype(lt->type->type, rt->type->type->type))
			goto bad;
		indir(n, arrayop(n, Etop));
		goto out;
	}

	if(ascompat(lt, rt))
		goto out;

bad:
	badtype(n->op, lt, rt);

out:
	ullmancalc(n);
	return n;
}

int
colasname(Node *n)
{
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
old2new(Node *n, Type *t)
{
	Node *l;

	if(!colasname(n)) {
		yyerror("left side of := must be a name");
		return n;
	}
	l = newname(n->sym);
	dodclvar(l, t);
	return l;
}

static Node*
mixedoldnew(Node *n, Type *t)
{
	n = nod(OXXX, n, N);
	n->type = t;
	return n;
}

static Node*
checkmixed(Node *nl)
{
	Iter save;
	Node *l, *n, *a;
	Type *t;
	int ntot, nred;

	nl = rev(nl);

	// first pass, check if it is a special
	// case of new and old declarations

	ntot = 0;	// number assignments
	nred = 0;	// number redeclarations
	l = listfirst(&save, &nl);
	while(l != N) {
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
		l = listnext(&save);
	}

	// test for special case
	// a) multi-assignment (ntot>1)
	// b) at least one redeclaration (red>0)
	// c) not all redeclarations (nred!=ntot)
	if(nred == 0 || ntot <= 1 || nred == ntot)
		goto allnew;

	n = N;
	l = listfirst(&save, &nl);
	while(l != N) {
		t = l->type;
		l = l->left;

		a = l;
		if(l->sym->block != block)
			a = old2new(l, t);

		n = list(n, a);
		l = listnext(&save);
	}
	return rev(n);

allnew:
	// same as original
	n = N;
	l = listfirst(&save, &nl);
	while(l != N) {
		t = l->type;
		l = l->left;

		a = old2new(l, t);
		n = list(n, a);
		l = listnext(&save);
	}
	return rev(n);
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

	/* check calls early, to give better message for a := f() */
	if(cr == 1) {
		switch(nr->op) {
		case OCALLMETH:
		case OCALLINTER:
		case OCALL:
			walktype(nr->left, Erv);
			convlit(nr->left, types[TFUNC]);
			t = nr->left->type;
			if(t == T)
				return nl;	// error already printed
			if(t->etype == tptr)
				t = t->type;
			if(t == T || t->etype != TFUNC) {
				yyerror("cannot call %T", t);
				return nl;
			}
			if(t->outtuple != cl) {
				cr = t->outtuple;
				goto badt;
			}
			// finish call - first half above
			l = listfirst(&savel, &nl);
			t = structfirst(&saver, getoutarg(t));
			while(l != N) {
				a = mixedoldnew(l, t->type);
				n = list(n, a);
				l = listnext(&savel);
				t = structnext(&saver);
			}
			n = checkmixed(n);
			return n;
		}
	}
	if(cl != cr) {
		if(cr == 1)
			goto multi;
		goto badt;
	}

	l = listfirst(&savel, &nl);
	r = listfirst(&saver, &nr);

	while(l != N) {
		walktype(r, Erv);
		defaultlit(r, T);
		a = mixedoldnew(l, r->type);
		n = list(n, a);

		l = listnext(&savel);
		r = listnext(&saver);
	}
	n = checkmixed(n);
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

	case OINDEX:
		// check if rhs is a map index.
		// if so, types are valuetype,bool
		if(cl != 2)
			goto badt;
		walktype(nr->left, Erv);
		implicitstar(&nr->left);
		t = nr->left->type;
		if(!istype(t, TMAP))
			goto badt;
		a = mixedoldnew(nl->left, t->type);
		n = a;
		a = mixedoldnew(nl->right, types[TBOOL]);
		n = list(n, a);
		n = checkmixed(n);
		break;

	case ODOTTYPE:
		// a,b := i.(T)
		if(cl != 2)
			goto badt;
		walktype(nr->left, Erv);
		if(!isinter(nr->left->type))
			goto badt;
		// a,b = iface
		a = mixedoldnew(nl->left, nr->type);
		n = a;
		a = mixedoldnew(nl->right, types[TBOOL]);
		n = list(n, a);
		n = checkmixed(n);
		break;

	case ORECV:
		if(cl != 2)
			goto badt;
		walktype(nr->left, Erv);
		t = nr->left->type;
		if(!istype(t, TCHAN))
			goto badt;
		a = mixedoldnew(nl->left, t->type);
		n = a;
		a = mixedoldnew(nl->right, types[TBOOL]);
		n = list(n, a);
		n = checkmixed(n);
		break;
	}
	n = rev(n);
	return n;

badt:
	if(nl->diag == 0) {
		nl->diag = 1;
		yyerror("assignment count mismatch: %d = %d", cl, cr);
	}
	return nl;
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
	Node *k, *v, *m, *init;
	Node *n, *hv, *hc, *ha, *hk, *ohk, *on, *r, *a;
	Type *t, *th;
	int local;

	if(nn->op != ORANGE)
		fatal("dorange not ORANGE");

	k = nn->left;
	v = N;
	if(k->op == OLIST) {
		v = k->right;
		k = k->left;
	}

	n = nod(OFOR, N, N);
	init = N;

	walktype(nn->right, Erv);
	implicitstar(&nn->right);
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
		k = old2new(k, hk->type);
	n->nbody = nod(OAS, k, hk);

	if(v != N) {
		if(local)
			v = old2new(v, t->type);
		n->nbody = list(n->nbody,
			nod(OAS, v, nod(OINDEX, ha, hk)) );
	}
	addtotop(n);
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
	r = nod(OADDR, hk, N);
	r = list(m, r);
	r = nod(OCALL, on, r);

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
	r = nod(OCALL, on, r);
	n->nincr = r;

	if(local)
		k = old2new(k, t->down);
	if(v == N) {
		on = syslook("mapiter1", 1);
		argtype(on, th);
		argtype(on, t->down);
		r = nod(OADDR, hk, N);
		r = nod(OCALL, on, r);
		n->nbody = nod(OAS, k, r);
		goto out;
	}
	if(local)
		v = old2new(v, t->type);
	on = syslook("mapiter2", 1);
	argtype(on, th);
	argtype(on, t->down);
	argtype(on, t->type);
	r = nod(OADDR, hk, N);
	r = nod(OCALL, on, r);
	n->nbody = nod(OAS, nod(OLIST, k, v), r);
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

	n->ntest = nod(ONOT, nod(OCLOSED, hc, N), N);
	n->nincr = nod(OAS, hv, nod(ORECV, hc, N));

	if(local)
		k = old2new(k, hv->type);
	n->nbody = nod(OAS, k, hv);
	addtotop(n);
	goto out;

strng:
	hk = nod(OXXX, N, N);		// hidden key
	tempname(hk, types[TINT]);

	ohk = nod(OXXX, N, N);		// old hidden key
	tempname(ohk, types[TINT]);

	ha = nod(OXXX, N, N);		// hidden string
	tempname(ha, t);

	hv = N;
	if(v != N) {
		hv = nod(OXXX, N, N);		// hidden value
		tempname(hv, types[TINT]);
	}

	if(local) {
		k = old2new(k, types[TINT]);
		if(v != N)
			v = old2new(v, types[TINT]);
	}

	// ha = s
	a = nod(OAS, ha, m);
	init = list(init, a);

	// ohk = 0
	a = nod(OAS, ohk, nodintconst(0));
	init = list(init, a);

	// hk[,hv] = stringiter(ha,hk)
	if(v != N) {
		// hk,v = stringiter2(ha, hk)
		on = syslook("stringiter2", 0);
//		argtype(on, v->type);
		a = list(ha, nodintconst(0));
		a = nod(OCALL, on, a);
		a = nod(OAS, list(hk, hv), a);
	} else {
		// hk = stringiter(ha, hk)
		on = syslook("stringiter", 0);
		a = list(ha, nodintconst(0));
		a = nod(OCALL, on, a);
		a = nod(OAS, hk, a);
	}
	init = list(init, a);

	// while(hk != 0)
	n->ntest = nod(ONE, hk, nodintconst(0));

	// hk[,hv] = stringiter(ha,hk)
	if(v != N) {
		// hk,hv = stringiter2(ha, hk)
		on = syslook("stringiter2", 0);
		a = list(ha, hk);
		a = nod(OCALL, on, a);
		a = nod(OAS, list(hk, hv), a);
	} else {
		// hk = stringiter(ha, hk)
		on = syslook("stringiter", 0);
		a = list(ha, hk);
		a = nod(OCALL, on, a);
		a = nod(OAS, hk, a);
	}
	n->nincr = list(n->nincr, a);

	// k,ohk[,v] = ohk,hk,[,hv]
	a = nod(OAS, k, ohk);
	n->nbody = a;
	a = nod(OAS, ohk, hk);
	n->nbody = list(n->nbody, a);
	if(v != N) {
		a = nod(OAS, v, hv);
		n->nbody = list(n->nbody, a);
	}

	addtotop(n);
	goto out;

out:
	n->ninit = list(n->ninit, init);
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
	ullmancalc(l);
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
		f = list(g, f);
		r = list(f, r);
		return r;
	}
	ullmancalc(l);
	if(l->ullman < UINF) {
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
	g = list(a, g);

	// put normal arg assignment on list
	// with fncall replaced by tempname
	l->right = a->left;
	r = list(l, r);

more:
	l = listnext(&save);
	goto loop2;
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
					tempname(q, l1->right->type);
					q = nod(OAS, l1->left, q);
					l1->left = q->right;
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
		q = list(q, l1);
		l1 = listnext(&save1);
	}

	r = rev(r);
	l1 = listfirst(&save1, &r);
	while(l1 != N) {
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

static void
fielddup(Node *n, Node *hash[], ulong nhash)
{
	uint h;
	char *s;
	Node *a;

	if(n->op != ONAME)
		fatal("fielddup: not ONAME");
	s = n->sym->name;
	h = stringhash(s)%nhash;
	for(a=hash[h]; a!=N; a=a->ntest) {
		if(strcmp(a->sym->name, s) == 0) {
			yyerror("duplicate field name in struct literal: %s", s);
			return;
		}
	}
	n->ntest = hash[h];
	hash[h] = n;
}

Node*
structlit(Node *n, Node *var)
{
	Iter savel, saver;
	Type *l, *t;
	Node *r, *a;
	Node* hash[101];
	int nerr;

	nerr = nerrors;
	t = n->type;
	if(t->etype != TSTRUCT)
		fatal("structlit: not struct");

	if(var == N) {
		var = nod(OXXX, N, N);
		tempname(var, t);
	}

	r = listfirst(&saver, &n->left);
	if(r != N && r->op == OEMPTY)
		r = N;
	if(r == N)
		goto keyval;	// must clear structure

	if(r->op == OKEY)
		goto keyval;
	l = structfirst(&savel, &n->type);

	while(r != N) {
		// assignment to every field
		if(l == T)
			break;
		if(r->op == OKEY) {
			yyerror("mixture of value and field:value initializers");
			return var;
		}

		// build list of var.field = expr
		a = nod(ODOT, var, newname(l->sym));
		a = nod(OAS, a, r);
		walktype(a, Etop);
		if(nerr != nerrors)
			return var;
		addtop = list(addtop, a);

		l = structnext(&savel);
		r = listnext(&saver);
	}
	if(l != T)
		yyerror("struct literal expect expr of type %T", l);
	if(r != N)
		yyerror("struct literal too many expressions");
	return var;

keyval:
	memset(hash, 0, sizeof(hash));
	a = nod(OAS, var, N);
	addtop = list(addtop, a);

	while(r != N) {
		// assignment to field:value elements
		if(r->op != OKEY) {
			yyerror("mixture of field:value and value initializers");
			break;
		}

		// build list of var.field = expr
		a = nod(ODOT, var, newname(r->left->sym));
		fielddup(a->right, hash, nelem(hash));
		if(nerr != nerrors)
			break;

		a = nod(OAS, a, r->right);
		walktype(a, Etop);
		if(nerr != nerrors)
			break;

		addtop = list(addtop, a);

		r = listnext(&saver);
	}
	return var;
}

static void
indexdup(Node *n, Node *hash[], ulong nhash)
{
	uint h;
	Node *a;
	ulong b, c;

	if(n->op != OLITERAL)
		fatal("indexdup: not OLITERAL");

	b = mpgetfix(n->val.u.xval);
	h = b%nhash;
	for(a=hash[h]; a!=N; a=a->ntest) {
		c = mpgetfix(a->val.u.xval);
		if(b == c) {
			yyerror("duplicate index in array literal: %ld", b);
			return;
		}
	}
	n->ntest = hash[h];
	hash[h] = n;
}

Node*
arraylit(Node *n, Node *var)
{
	Iter saver;
	Type *t;
	Node *r, *a;
	long ninit, b;
	Node* hash[101];
	int nerr;

	nerr = nerrors;
	t = n->type;
	if(t->etype != TARRAY)
		fatal("arraylit: not array");

	// find max index
	ninit = 0;
	b = 0;

	r = listfirst(&saver, &n->left);
	if(r != N && r->op == OEMPTY)
		r = N;

	while(r != N) {
		if(r->op == OKEY) {
			evconst(r->left);
			b = nonnegconst(r->left);
		}
		b++;
		if(b > ninit)
			ninit = b;
		r = listnext(&saver);
	}

	b = t->bound;
	if(b == -100) {
		// flag for [...]
		b = ninit;
		if(var == N)
			t = shallow(t);
		t->bound = b;
	}

	if(var == N) {
		var = nod(OXXX, N, N);
		tempname(var, t);
	}

	if(b < 0) {
		// slice
		a = nod(OMAKE, nod(OLIST, typenod(t), nodintconst(ninit)), N);
		a = nod(OAS, var, a);
		addtop = list(addtop, a);
	} else {
		// if entire array isnt initialized,
		// then clear the array
		if(ninit < b) {
			a = nod(OAS, var, N);
			addtop = list(addtop, a);
		}
	}

	b = 0;
	r = listfirst(&saver, &n->left);
	if(r != N && r->op == OEMPTY)
		r = N;

	memset(hash, 0, sizeof(hash));
	while(r != N) {
		// build list of var[c] = expr
		if(r->op == OKEY) {
			b = nonnegconst(r->left);
			if(b < 0) {
				yyerror("array index must be non-negative integer");
				break;
			}
			r = r->right;
		}

		if(t->bound >= 0 && b > t->bound) {
			yyerror("array index out of bounds");
			break;
		}

		a = nodintconst(b);
		indexdup(a, hash, nelem(hash));
		if(nerr != nerrors)
			break;

		a = nod(OINDEX, var, a);
		a = nod(OAS, a, r);
		walktype(a, Etop);	// add any assignments in r to addtop
		if(nerr != nerrors)
			break;

		addtop = list(addtop, a);
		b++;

		r = listnext(&saver);
	}
	return var;
}

static void
keydup(Node *n, Node *hash[], ulong nhash)
{
	uint h;
	ulong b;
	double d;
	int i;
	Node *a;
	Node cmp;
	char *s;

	evconst(n);
	if(n->op != OLITERAL)
		return;	// we dont check variables

	switch(n->val.ctype) {
	default:	// unknown, bool, nil
		b = 23;
		break;
	case CTINT:
		b = mpgetfix(n->val.u.xval);
		break;
	case CTFLT:
		d = mpgetflt(n->val.u.fval);
		s = (char*)&d;
		b = 0;
		for(i=sizeof(d); i>0; i--)
			b = b*PRIME1 + *s++;
		break;
	case CTSTR:
		b = 0;
		s = n->val.u.sval->s;
		for(i=n->val.u.sval->len; i>0; i--)
			b = b*PRIME1 + *s++;
		break;
	}

	h = b%nhash;
	memset(&cmp, 0, sizeof(cmp));
	for(a=hash[h]; a!=N; a=a->ntest) {
		cmp.op = OEQ;
		cmp.left = n;
		cmp.right = a;
		evconst(&cmp);
		b = cmp.val.u.bval;
		if(b) {
			// too lazy to print the literal
			yyerror("duplicate key in map literal");
			return;
		}
	}
	n->ntest = hash[h];
	hash[h] = n;
}

Node*
maplit(Node *n, Node *var)
{
	Iter saver;
	Type *t;
	Node *r, *a;
	Node* hash[101];
	int nerr;

	nerr = nerrors;
	t = n->type;
	if(t->etype != TMAP)
		fatal("maplit: not map");

	if(var == N) {
		var = nod(OXXX, N, N);
		tempname(var, t);
	}

	a = nod(OMAKE, typenod(t), N);
	a = nod(OAS, var, a);
	addtop = list(addtop, a);

	r = listfirst(&saver, &n->left);
	if(r != N && r->op == OEMPTY)
		r = N;

	memset(hash, 0, sizeof(hash));
	while(r != N) {
		if(r == N)
			break;

		if(r->op != OKEY) {
			yyerror("map literal must have key:value pairs");
			break;
		}

		// build list of var[c] = expr
		keydup(r->left, hash, nelem(hash));
		if(nerr != nerrors)
			break;

		a = nod(OINDEX, var, r->left);
		a = nod(OAS, a, r->right);
		walktype(a, Etop);
		if(nerr != nerrors)
			break;

		addtop = list(addtop, a);

		r = listnext(&saver);
	}
	return var;
}

/*
 * the address of n has been taken and might be used after
 * the current function returns.  mark any local vars
 * as needing to move to the heap.
 */
void
addrescapes(Node *n)
{
	char buf[100];
	switch(n->op) {
	default:
		// probably a type error already.
		// dump("addrescapes", n);
		break;

	case ONAME:
		if(n->noescape)
			break;
		switch(n->class) {
		case PPARAMOUT:
			yyerror("cannot take address of out parameter %s", n->sym->name);
			break;
		case PAUTO:
		case PPARAM:
			// if func param, need separate temporary
			// to hold heap pointer.
			if(n->class == PPARAM) {
				// expression to refer to stack copy
				n->stackparam = nod(OPARAM, n, N);
				n->stackparam->type = n->type;
				n->stackparam->addable = 1;
				n->stackparam->xoffset = n->xoffset;
			}

			n->class |= PHEAP;
			n->addable = 0;
			n->ullman = 2;
			n->alloc = callnew(n->type);
			n->xoffset = 0;

			// create stack variable to hold pointer to heap
			n->heapaddr = nod(0, N, N);
			tempname(n->heapaddr, ptrto(n->type));
			snprint(buf, sizeof buf, "&%S", n->sym);
			n->heapaddr->sym = lookup(buf);
			break;
		}
		break;

	case OIND:
	case ODOTPTR:
		break;

	case ODOT:
	case OINDEX:
		// ODOTPTR has already been introduced,
		// so these are the non-pointer ODOT and OINDEX.
		// In &x[0], if x is a slice, then x does not
		// escape--the pointer inside x does, but that
		// is always a heap pointer anyway.
		if(!isslice(n->left->type))
			addrescapes(n->left);
		break;
	}
}

/*
 * walk through argin parameters.
 * generate and return code to allocate
 * copies of escaped parameters to the heap.
 */
Node*
paramstoheap(Type **argin)
{
	Type *t;
	Iter savet;
	Node *v, *nn;

	nn = N;
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
	Node *nn;

	nn = paramstoheap(getthis(curfn->type));
	nn = list(nn, paramstoheap(getinarg(curfn->type)));
	curfn->enter = list(curfn->enter, nn);
}
