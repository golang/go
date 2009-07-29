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
	walkexpr(np, Erv, init);
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
			walkexpr(&n->ntype, Etype, &init);
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
		walkexpr(&e, Erv, &init);
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

	case OASOP:
	case OAS:
	case OAS2:
	case OCLOSE:
	case OCLOSED:
	case OCALLMETH:
	case OCALLINTER:
	case OCALL:
	case OSEND:
	case ORECV:
	case OPRINT:
	case OPRINTN:
	case OPANIC:
	case OPANICN:
	case OEMPTY:
		init = n->ninit;
		n->ninit = nil;
		walkexpr(&n, Etop, &init);
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
		walkexpr(&n->left, Etop, &n->ninit);
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
		walkexpr(&n->left, Etop, &n->ninit);
		break;

	case ORETURN:
		walkexprlist(n->list, Erv, &n->ninit);
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
	if(!isfixedarray(t))
		return;
	n = nod(OIND, n, N);
	walkexpr(&n, Elv, nil);
	*nn = n;
}

void
typechecklist(NodeList *l, int top)
{
	for(; l; l=l->next)
		typecheck(&l->n, top);
}

/*
 * type check the whole tree of an expression.
 * calculates expression types.
 * evaluates compile time constants.
 * marks variables that escape the local frame.
 * rewrites n->op to be more specific in some cases.
 * replaces *np with a new pointer in some cases.
 * returns the final value of *np as a convenience.
 */
Node*
typecheck(Node **np, int top)
{
	int et, et1, et2;
	Node *n, *l, *r;
	int lno, ok;
	Type *t;

	n = *np;
	if(n == N || n->typecheck == 1)
		return n;
	if(n->typecheck == 2)
		fatal("typecheck loop");
	n->typecheck = 2;

	if(n->sym && n->walkdef != 1)
		walkdef(n);

	lno = setlineno(n);

	ok = 0;
	switch(n->op) {
	default:
		// until typecheck is complete, do nothing.
		goto ret;
		dump("typecheck", n);
		fatal("typecheck %O", n->op);

	/*
	 * names
	 */
	case OLITERAL:
		ok |= Erv;
		goto ret;

	case ONONAME:
		ok |= Elv | Erv;
		goto ret;

	case ONAME:
		if(n->etype != 0) {
			yyerror("must call builtin %S", n->sym);
			goto error;
		}
		ok |= Erv;
		if(n->class != PFUNC)
			ok |= Elv;
		goto ret;

	/*
	 * types (OIND is with exprs)
	 */
	case OTYPE:
		ok |= Etype;
		if(n->type == T)
			goto error;
		break;

	case OTARRAY:
		ok |= Etype;
		t = typ(TARRAY);
		l = n->left;
		r = n->right;
		if(l == nil) {
			t->bound = -1;
		} else {
			typecheck(&l, Erv | Etype);
			walkexpr(&l, Erv | Etype, &n->ninit);	// TODO: remove
			switch(l->op) {
			default:
				yyerror("invalid array bound %O", l->op);
				goto error;

			case OLITERAL:
				if(consttype(l) == CTINT) {
					t->bound = mpgetfix(l->val.u.xval);
					if(t->bound < 0) {
						yyerror("array bound must be non-negative");
						goto error;
					}
				}
				break;

			case OTYPE:
				if(l->type == T)
					goto error;
				if(l->type->etype != TDDD) {
					yyerror("invalid array bound %T", l->type);
					goto error;
				}
				t->bound = -100;
				break;
			}
		}
		typecheck(&r, Etype);
		if(r->type == T)
			goto error;
		t->type = r->type;
		n->op = OTYPE;
		n->type = t;
		n->left = N;
		n->right = N;
		checkwidth(t);
		break;

	case OTMAP:
		ok |= Etype;
		l = typecheck(&n->left, Etype);
		r = typecheck(&n->right, Etype);
		if(l->type == T || r->type == T)
			goto error;
		n->op = OTYPE;
		n->type = maptype(l->type, r->type);
		n->left = N;
		n->right = N;
		break;

	case OTCHAN:
		ok |= Etype;
		l = typecheck(&n->left, Etype);
		if(l->type == T)
			goto error;
		t = typ(TCHAN);
		t->type = l->type;
		t->chan = n->etype;
		n->op = OTYPE;
		n->type = t;
		n->left = N;
		n->etype = 0;
		break;

	case OTSTRUCT:
		ok |= Etype;
		n->op = OTYPE;
		n->type = dostruct(n->list, TSTRUCT);
		if(n->type == T)
			goto error;
		n->list = nil;
		break;

	case OTINTER:
		ok |= Etype;
		n->op = OTYPE;
		n->type = dostruct(n->list, TINTER);
		if(n->type == T)
			goto error;
		n->type = sortinter(n->type);
		break;

	case OTFUNC:
		ok |= Etype;
		n->op = OTYPE;
		n->type = functype(n->left, n->list, n->rlist);
		if(n->type == T)
			goto error;
		break;

	/*
	 * exprs
	 */
	case OADD:
	case OAND:
	case OANDAND:
	case OANDNOT:
	case ODIV:
	case OEQ:
	case OGE:
	case OGT:
	case OLE:
	case OLT:
	case OMOD:
	case OMUL:
	case ONE:
	case OOR:
	case OOROR:
	case OSUB:
	case OXOR:
		ok |= Erv;
		l = typecheck(&n->left, Erv | Eideal);
		r = typecheck(&n->right, Erv | Eideal);
		if(l->type == T || r->type == T)
			goto error;
		et1 = l->type->etype;
		et2 = r->type->etype;
		if(et1 == TIDEAL || et1 == TNIL || et2 == TIDEAL || et2 == TNIL)
		if(et1 != TIDEAL && et1 != TNIL || et2 != TIDEAL && et2 != TNIL) {
			// ideal mixed with non-ideal
			defaultlit2(&l, &r);
			n->left = l;
			n->right = r;
		}
		t = l->type;
		if(t->etype == TIDEAL)
			t = r->type;
		et = t->etype;
		if(et == TIDEAL)
			et = TINT;
		if(t->etype != TIDEAL && !eqtype(l->type, r->type)) {
		badbinary:
			yyerror("invalid operation: %#N", n);
			goto error;
		}
		if(!okfor[n->op][et])
			goto badbinary;
		// okfor allows any array == array;
		// restrict to slice == nil and nil == slice.
		if(l->type->etype == TARRAY && !isslice(l->type))
			goto badbinary;
		if(r->type->etype == TARRAY && !isslice(r->type))
			goto badbinary;
		if(isslice(l->type) && !isnil(l) && !isnil(r))
			goto badbinary;
		evconst(n);
		goto ret;

	case OCOM:
	case OMINUS:
	case ONOT:
	case OPLUS:
		ok |= Erv;
		l = typecheck(&n->left, Erv | Eideal);
		walkexpr(&n->left, Erv | Eideal, &n->ninit);	// TODO: remove
		if((t = l->type) == T)
			goto error;
		if(!okfor[n->op][t->etype]) {
			yyerror("invalid operation: %#O %T", n->op, t);
			goto error;
		}
		n->type = t;
		goto ret;

	/*
	 * type or expr
	 */
	case OIND:
		typecheck(&n->left, top | Etype);
		if(n->left->op == OTYPE) {
			ok |= Etype;
			n->op = OTYPE;
			n->type = ptrto(n->left->type);
			n->left = N;
			goto ret;
		}

		// TODO: OIND expression type checking
		goto ret;

	}

ret:
	evconst(n);
	if(n->op == OTYPE && !(top & Etype)) {
		yyerror("type %T is not an expression", n->type);
		goto error;
	}
	if((top & (Elv|Erv|Etype)) == Etype && n->op != OTYPE) {
		yyerror("%O is not a type", n->op);
		goto error;
	}

	/* TODO
	if(n->type == T)
		fatal("typecheck nil type");
	*/
	goto out;

error:
	n->type = T;

out:
	lineno = lno;
	n->typecheck = 1;
	*np = n;
	return n;
}


/*
 * walk the whole tree of the body of an
 * expression or simple statement.
 * the types expressions are calculated.
 * compile-time constants are evaluated.
 * complex side effects like statements are appended to init
 */

void
walkexprlist(NodeList *l, int top, NodeList **init)
{
	for(; l; l=l->next)
		walkexpr(&l->n, top, init);
}

void
walkexpr(Node **np, int top, NodeList **init)
{
	Node *r, *l;
	NodeList *ll, *lr;
	Type *t;
	Sym *s;
	int et, cl, cr, typeok, op;
	int32 lno;
	Node *n;

	n = *np;

	if(n == N)
		return;

	lno = setlineno(n);
	typeok = top & Etype;
	top &= ~Etype;

	if(debug['w'] > 1 && top == Etop)
		dump("walk-before", n);

	if(n->typecheck != 1)
		typecheck(&n, top | typeok);

reswitch:
	t = T;
	et = Txxx;

	switch(n->op) {
	default:
		dump("walk", n);
		fatal("walkexpr: switch 1 unknown op %N", n);
		goto ret;

	case OTYPE:
		goto ret;

	case OKEY:
		walkexpr(&n->left, top | typeok, init);
		walkexpr(&n->right, top | typeok, init);
		goto ret;

	case OPRINT:
		if(top != Etop)
			goto nottop;
		walkexprlist(n->list, Erv, init);
		n = prcompat(n->list, 0, 0);
//dump("prcompat", n);
		goto ret;

	case OPRINTN:
		if(top != Etop)
			goto nottop;
		walkexprlist(n->list, Erv, init);
		n = prcompat(n->list, 1, 0);
		goto ret;

	case OPANIC:
		if(top != Etop)
			goto nottop;
		walkexprlist(n->list, Erv, init);
		n = prcompat(n->list, 0, 1);
		goto ret;

	case OPANICN:
		if(top != Etop)
			goto nottop;
		walkexprlist(n->list, Erv, init);
		n = prcompat(n->list, 2, 1);
		goto ret;

	case OLITERAL:
		if(!(top & Erv))
			goto nottop;
		n->addable = 1;
		goto ret;

	case ONONAME:
		s = n->sym;
		if(n->diag == 0) {
			s->undef = 1;
			n->diag = 1;
			yyerror("undefined: %S", s);
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
					yyerror("walkexpr: %S must be called", s, init);
				else
					yyerror("walkexpr: %S undeclared", s, init);
				s->undef = 1;
			}
		}
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

		if(n->left->op == ONAME && n->left->etype != 0) {
			// builtin OLEN, OCAP, etc.
			n->op = n->left->etype;
			n->left = N;
//dump("do", n);
			goto reswitch;
		}

		walkexpr(&n->left, Erv | Etype, init);
		defaultlit(&n->left, T);

		t = n->left->type;
		if(t == T)
			goto ret;

		if(n->left->op == ODOTMETH)
			n->op = OCALLMETH;
		if(n->left->op == ODOTINTER)
			n->op = OCALLINTER;
		if(n->left->op == OTYPE) {
			n->op = OCONV;
			if(!(top & Erv))
				goto nottop;
			// turn CALL(type, arg) into CONV(arg) w/ type.
			n->type = n->left->type;
			if(n->list == nil) {
				yyerror("missing argument in type conversion");
				goto ret;
			}
			if(n->list->next != nil) {
				yyerror("too many arguments in type conversion");
				goto ret;
			}
			n->left = n->list->n;
			n->list = nil;
			goto reswitch;
		}

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
			if(n->type != T && n->type->type != T && n->type->type->type != T)
				n->type = n->type->type->type;
			break;
		}

		walkexprlist(n->list, Erv, init);

		switch(n->op) {
		default:
			fatal("walk: op: %O", n->op);

		case OCALLINTER:
			ll = ascompatte(n->op, getinarg(t), n->list, 0, init);
			n->list = reorder1(ll);
			break;

		case OCALL:
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
			break;

		case OCALLMETH:
			ll = ascompatte(n->op, getinarg(t), n->list, 0, init);
			lr = ascompatte(n->op, getthis(t), list1(n->left->left), 0, init);
			ll = concat(ll, lr);
			n->left->left = N;
			ullmancalc(n->left);
			n->list = reorder1(ll);
			break;
		}
		goto ret;

	case OAS:
		if(top != Etop)
			goto nottop;
		*init = concat(*init, n->ninit);
		n->ninit = nil;
		walkexpr(&n->left, Elv, init);
		walkexpr(&n->right, Erv, init);
		l = n->left;
		r = n->right;
		if(l == N || r == N)
			goto ret;
		r = ascompatee1(n->op, l, r, init);
		if(r != N)
			n = r;
		goto ret;

	case OAS2:
		if(top != Etop)
			goto nottop;
		*init = concat(*init, n->ninit);
		n->ninit = nil;

		walkexprlist(n->list, Elv, init);

		cl = count(n->list);
		cr = count(n->rlist);
		if(cl == cr) {
		multias:
			walkexprlist(n->rlist, Erv, init);
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
		case OCALL:
			if(cr == 1) {
				// a,b,... = fn()
				walkexpr(&r, Erv, init);
				if(r->type == T || r->type->etype != TSTRUCT)
					break;
				ll = ascompatet(n->op, n->list, &r->type, 0, init);
				n = liststmt(concat(list1(r), ll));
				goto ret;
			}
			break;

		case OINDEX:
			if(cl == 2 && cr == 1) {
				// a,b = map[] - mapaccess2
				walkexpr(&r->left, Erv, init);
				implicitstar(&r->left);
				if(!istype(r->left->type, TMAP))
					break;
				l = mapop(n, top, init);
				if(l == N)
					break;
				n = l;
				goto ret;
			}
			break;

		case ORECV:
			if(cl == 2 && cr == 1) {
				// a,b = <chan - chanrecv2
				walkexpr(&r->left, Erv, init);
				if(!istype(r->left->type, TCHAN))
					break;
				l = chanop(n, top, init);
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
					goto multias;
				case I2E:
					n->list = list(list1(n->right), nodbool(1));
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
				r = ifacecvt(r->type, r->left, et);
				ll = ascompatet(n->op, n->list, &r->type, 0, init);
				n = liststmt(concat(list1(r), ll));
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
				l = mapop(n, top, init);
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

	case OINDREG:
	case OEMPTY:
		goto ret;

	case ODOTTYPE:
		walkdottype(n, init);
		// fall through
	case OCONV:
		if(!(top & Erv))
			goto nottop;
		walkconv(&n, init);
		goto ret;

	case OCONVNOP:
		goto ret;

	case OCOMPMAP:
	case OCOMPSLICE:
		goto ret;

	case OCOMPOS:
		walkexpr(&n->right, Etype, init);
		t = n->right->type;
		n->type = t;
		if(t == T)
			goto ret;

		switch(t->etype) {
		default:
			yyerror("invalid type for composite literal: %T", t);
			goto ret;

		case TSTRUCT:
			r = structlit(n, N, init);
			break;

		case TARRAY:
			r = arraylit(n, N, init);
			break;

		case TMAP:
			r = maplit(n, N, init);
			break;
		}
		n = r;
		goto ret;

	case ONOT:
		if(!(top & Erv))
			goto nottop;
		if(n->op == OLITERAL)
			goto ret;
		walkexpr(&n->left, Erv, init);
		if(n->left == N || n->left->type == T)
			goto ret;
		et = n->left->type->etype;
		break;

	case OASOP:
		if(top != Etop)
			goto nottop;
		walkexpr(&n->left, Elv, init);
		l = n->left;
		if(l->op == OINDEX && istype(l->left->type, TMAP))
			n = mapop(n, top, init);
		if(n->etype == OLSH || n->etype == ORSH)
			goto shft;
		goto com;

	case OLSH:
	case ORSH:
		if(!(top & Erv))
			goto nottop;
		walkexpr(&n->left, Erv, init);

	shft:
		walkexpr(&n->right, Erv, init);
		if(n->left == N || n->right == N)
			goto ret;
		evconst(n);
		if(n->op == OLITERAL)
			goto ret;
		// do NOT defaultlit n->left.
		// let parent defaultlit or convlit instead.
		defaultlit(&n->right, types[TUINT]);
		if(n->left->type == T || n->right->type == T)
			goto ret;
		et = n->right->type->etype;
		if(issigned[et] || !isint[et])
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
		if(!(top & Erv))
			goto nottop;
		walkexpr(&n->left, Erv, init);

	com:
		walkexpr(&n->right, Erv, init);
		if(n->left == N || n->right == N)
			goto ret;
		evconst(n);
		if(n->op == OLITERAL)
			goto ret;
		defaultlit2(&n->left, &n->right);
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
			if(istype(n->left->type, TSTRING)) {
				n = stringop(n, top, init);
				goto ret;
			}
			break;

		case OEQ:
		case ONE:
		case OLT:
		case OLE:
		case OGE:
		case OGT:
		case OADD:
			if(istype(n->left->type, TSTRING)) {
				n = stringop(n, top, nil);
				goto ret;
			}
			break;
		}
		break;

	case OMINUS:
	case OPLUS:
	case OCOM:
		if(!(top & Erv))
			goto nottop;
		walkexpr(&n->left, Erv, init);
		if(n->left == N)
			goto ret;
		if(n->op == OLITERAL)
			goto ret;
		break;

	case OLEN:
		if(!(top & Erv))
			goto nottop;
		if(n->left == N) {
			if(n->list == nil) {
				yyerror("missing argument to len");
				goto ret;
			}
			if(n->list->next)
				yyerror("too many arguments to len");
			n->left = n->list->n;
		}
		walkexpr(&n->left, Erv, init);
		defaultlit(&n->left, T);
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
		if(!(top & Erv))
			goto nottop;
		if(n->left == N) {
			if(n->list == nil) {
				yyerror("missing argument to cap");
				goto ret;
			}
			if(n->list->next)
				yyerror("too many arguments to cap");
			n->left = n->list->n;
		}
		walkexpr(&n->left, Erv, init);
		defaultlit(&n->left, T);
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

		walkexpr(&n->left, Erv, init);
		walkexpr(&n->right, Erv, init);

		if(n->left == N || n->right == N)
			goto ret;

		defaultlit(&n->left, T);
		implicitstar(&n->left);

		t = n->left->type;
		if(t == T)
			goto ret;

		switch(t->etype) {
		default:
			defaultlit(&n->right, T);
			goto badt;

		case TSTRING:
			// right side must be an int
			if(!(top & Erv))
				goto nottop;
			defaultlit(&n->right, types[TINT]);
			if(n->right->type == T)
				break;
			if(!isint[n->right->type->etype])
				goto badt;
			n = stringop(n, top, nil);
			break;

		case TMAP:
			// right side must be map type
			defaultlit(&n->right, t->down);
			if(n->right->type == T)
				break;
			if(!eqtype(n->right->type, t->down))
				goto badt;
			n->type = t->type;
			if(top == Erv)
				n = mapop(n, top, nil);
			break;

		case TARRAY:
			// right side must be an int
			defaultlit(&n->right, types[TINT]);
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
		walkexpr(&n->left, Erv, init);		// chan
		n = chanop(n, top, nil);
		goto ret;

	case OCLOSED:
		if(top == Elv)
			goto nottop;
		walkexpr(&n->left, Erv, init);		// chan
		n = chanop(n, top, nil);
		goto ret;

	case OSEND:
		if(top == Elv)
			goto nottop;
		walkexpr(&n->left, Erv, init);	// chan
		walkexpr(&n->right, Erv, init);	// e
		n = chanop(n, top, nil);
		goto ret;

	case ORECV:
		if(top == Elv)
			goto nottop;
		if(n->right == N) {
			walkexpr(&n->left, Erv, init);		// chan
			n = chanop(n, top, init);	// returns e blocking
			goto ret;
		}
		walkexpr(&n->left, Elv, init);		// e
		walkexpr(&n->right, Erv, init);	// chan
		n = chanop(n, top, nil);	// returns bool non-blocking
		goto ret;

	case OSLICE:
		if(top == Etop)
			goto nottop;

		walkexpr(&n->left, top, init);
		walkexpr(&n->right->left, Erv, init);
		walkexpr(&n->right->right, Erv, init);
		if(n->left == N || n->right == N)
			goto ret;
		defaultlit(&n->left, T);
		defaultlit(&n->right->left, types[TUINT]);
		defaultlit(&n->right->right, types[TUINT]);
		implicitstar(&n->left);
		t = n->left->type;
		if(t == T)
			goto ret;
		if(t->etype == TSTRING) {
			n = stringop(n, top, nil);
			goto ret;
		}
		if(t->etype == TARRAY) {
			n = arrayop(n, top);
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
		defaultlit(&n->left, T);
		walkdot(n, init);
		goto ret;

	case OADDR:
		if(!(top & Erv))
			goto nottop;
		defaultlit(&n->left, T);
		if(n->left->op == OCOMPOS) {
			walkexpr(&n->left->right, Etype, init);
			n->left->type = n->left->right->type;
			if(n->left->type == T)
				goto ret;

			Node *nvar, *nas, *nstar;

			// turn &Point(1, 2) or &[]int(1, 2) or &[...]int(1, 2) into allocation.
			// initialize with
			//	nvar := new(*Point);
			//	*nvar = Point(1, 2);
			// and replace expression with nvar

			nvar = nod(OXXX, N, N);
			tempname(nvar, ptrto(n->left->type));

			nas = nod(OAS, nvar, callnew(n->left->type));
			walkexpr(&nas, Etop, init);
			*init = list(*init, nas);

			nstar = nod(OIND, nvar, N);
			nstar->type = n->left->type;

			switch(n->left->type->etype) {
			case TSTRUCT:
				structlit(n->left, nstar, init);
				break;
			case TARRAY:
				arraylit(n->left, nstar, init);
				break;
			case TMAP:
				maplit(n->left, nstar, init);
				break;
			default:
				goto badlit;
			}

//			walkexpr(&n->left->left, Erv, init);
			n = nvar;
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
		walkexpr(&n->left, Elv, init);
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
		if(n->left == N)
			goto ret;
		walkexpr(&n->left, top | Etype, init);
		defaultlit(&n->left, T);
		if(n->left->op == OTYPE) {
			n->op = OTYPE;
			n->type = ptrto(n->left->type);
			goto ret;
		}
		t = n->left->type;
		if(t == T)
			goto ret;
		if(!isptr[t->etype])
			goto badt;
		n->type = t->type;
		goto ret;

	case OMAKE:
		if(!(top & Erv))
			goto nottop;
		n = makecompat(n);
		goto ret;

	case ONEW:
		if(!(top & Erv))
			goto nottop;
		if(n->list == nil) {
			yyerror("missing argument to new");
			goto ret;
		}
		if(n->list->next)
			yyerror("too many arguments to new");
		walkexpr(&n->list->n, Etype, init);
		l = n->list->n;
		if((t = l->type) == T)
			;
		else
			n = callnew(t);
		goto ret;
	}

/*
 * ======== second switch ========
 */

	op = n->op;
	if(op == OASOP)
		op = n->etype;
	switch(op) {
	default:
		fatal("walkexpr: switch 2 unknown op %N", n, init);
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
			n = ifaceop(n);
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
		if(!okforarith[et] && et != TSTRING)
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
		if(!okforarith[et])
			goto badt;
		break;

	case OMINUS:
		if(n->left->type == T)
			goto ret;
		et = n->left->type->etype;
		if(!okforarith[et])
			goto badt;
		if(isfloat[et]) {
			// TODO(rsc): Can do this more efficiently,
			// but OSUB is wrong.  Should be in back end anyway.
			n = nod(OMUL, n->left, nodintconst(-1));
			walkexpr(&n, Erv, init);
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
		r = nod(OCALL, l, N);
		r->list = list(list1(n->left), n->right);
		r = nod(OCONV, r, N);
		r->type = n->left->left->type;
		walkexpr(&r, Erv, init);
		n = r;
		goto ret;

	case OASOP:
		et = n->left->type->etype;
		if(widthptr > 4 || (et != TUINT64 && et != TINT64))
			break;
		l = saferef(n->left, init);
		r = nod(OAS, l, nod(n->etype, l, n->right));
		walkexpr(&r, Etop, init);
		n = r;
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
	switch((top | typeok) & ~Eideal) {
	default:
		yyerror("didn't expect %O here [top=%d]", n->op, top);
		break;
	case Etype:
		yyerror("operation %O not allowed in type context", n->op);
		break;
	case Etop:
		yyerror("operation %O not allowed in statement context", n->op);
		break;
	case Elv:
		yyerror("operation %O not allowed in assignment context", n->op);
		break;
	case Erv:
	case Erv | Etype:
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
	op = n->op;
	if(op == OASOP)
		op = n->etype;
	badtype(op, n->left->type, n->right->type);
	goto ret;

ret:
	if(debug['w'] && top == Etop && n != N)
		dump("walk", n);

	if(typeok && top == 0) {	// must be type
		if(n->op != OTYPE) {
			if(n->sym) {
				if(!n->sym->undef)
					yyerror("%S is not a type", n->sym);
			} else {
				yyerror("expr %O is not type", n->op);
				n->op = OTYPE;	// leads to fewer errors later
				n->type = T;
			}
		}
	}
	if(!typeok && n->op == OTYPE)
		yyerror("cannot use type %T as expr", n->type);

	ullmancalc(n);
	lineno = lno;
	*np = n;
}

void
walkbool(Node **np)
{
	Node *n;

	n = *np;
	if(n == N)
		return;
	walkexpr(np, Erv, &n->ninit);
	defaultlit(np, T);
	n = *np;
	if(n->type != T && !eqtype(n->type, types[TBOOL]))
		yyerror("IF and FOR require a boolean type");
}

void
walkdottype(Node *n, NodeList **init)
{
	walkexpr(&n->left, Erv, init);
	if(n->left == N)
		return;
	defaultlit(&n->left, T);
	if(!isinter(n->left->type))
		yyerror("type assertion requires interface on left, have %T", n->left->type);
	if(n->right != N) {
		walkexpr(&n->right, Etype, init);
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
	walkexpr(&n->left, Erv, init);
	l = n->left;
	if(l == N)
		return;
	if(l->type == T)
		return;

	// if using .(T), interface assertion.
	if(n->op == ODOTTYPE) {
		et = ifaceas1(t, l->type, 1);
		if(et == I2Isame || et == E2Esame)
			goto nop;
		if(et != Inone) {
			n = ifacecvt(t, l, et);
			*np = n;
			return;
		}
		goto bad;
	}

	// otherwise, conversion.
	convlit1(&n->left, t, 1);
	l = n->left;
	if(l->type == T)
		return;

	// no-op conversion
	if(cvttype(t, l->type) == 1) {
	nop:
		if(l->op == OLITERAL) {
			*n = *l;
			n->type = t;
			return;
		}
		// leave OCONV node in place
		// in case tree gets walked again.
		// back end will ignore.
		n->op = OCONVNOP;
		return;
	}

	// to/from interface.
	// ifaceas1 will generate a good error
	// if the conversion is invalid.
	if(t->etype == TINTER || l->type->etype == TINTER) {
		n = ifacecvt(t, l, ifaceas1(t, l->type, 0));
		*np = n;
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
			n = stringop(n, Erv, nil);
			*np = n;
			return;
		}

		// can convert []byte and *[10]byte
		if((isptr[et] && isfixedarray(l->type->type) && istype(l->type->type->type, TUINT8))
		|| (isslice(l->type) && istype(l->type->type, TUINT8))) {
			n->op = OARRAY;
			n = stringop(n, Erv, nil);
			*np = n;
			return;
		}

		// can convert []int and *[10]int
		if((isptr[et] && isfixedarray(l->type->type) && istype(l->type->type->type, TINT))
		|| (isslice(l->type) && istype(l->type->type, TINT))) {
			n->op = OARRAY;
			n = stringop(n, Erv, nil);
			*np = n;
			return;
		}
	}

	// convert dynamic to static generated by ONEW/OMAKE
	if(isfixedarray(t) && isslice(l->type))
		return;

	// convert static array to dynamic array
	if(isslice(t) && isptr[l->type->etype] && isfixedarray(l->type->type)) {
		if(eqtype(t->type->type, l->type->type->type->type)) {
			n = arrayop(n, Erv);
			*np = n;
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
}

Node*
selcase(Node *n, Node *var, NodeList **init)
{
	Node *a, *r, *on, *c;
	Type *t;
	NodeList *args;

	if(n->list == nil)
		goto dflt;
	c = n->list->n;
	if(c->op == ORECV)
		goto recv;

	walkexpr(&c->left, Erv, init);		// chan
	walkexpr(&c->right, Erv, init);	// elem

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
	on = syslook("selectsend", 1);
	argtype(on, t->type);
	argtype(on, t->type);

	a = var;			// sel-var
	args = list1(a);
	a = c->left;			// chan
	args = list(args, a);
	a = c->right;			// elem
	args = list(args, a);
	goto out;

recv:
	if(c->right != N)
		goto recv2;

	walkexpr(&c->left, Erv, init);		// chan

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

	a = var;			// sel-var
	args = list1(a);

	a = c->left;			// chan
	args = list(args, a);

	a = c->left;			// nil elem
	a = nod(OLITERAL, N, N);
	a->val.ctype = CTNIL;
	a->type = types[TNIL];
	args = list(args, a);
	goto out;

recv2:
	walkexpr(&c->right, Erv, init);	// chan

	t = fixchan(c->right->type);
	if(t == T)
		return N;

	if(!(t->chan & Crecv)) {
		yyerror("cannot receive from %T", t);
		return N;
	}

	walkexpr(&c->left, Elv, init);	// check elem
	convlit(&c->left, t->type);
	if(!ascompat(t->type, c->left->type)) {
		badtype(c->op, t->type, c->left->type);
		return N;
	}

	// selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
	on = syslook("selectrecv", 1);
	argtype(on, t->type);
	argtype(on, t->type);

	a = var;			// sel-var
	args = list1(a);

	a = c->right;			// chan
	args = list(args, a);

	a = c->left;			// elem
	a = nod(OADDR, a, N);
	args = list(args, a);
	goto out;

dflt:
	// selectdefault(sel *byte);
	on = syslook("selectdefault", 0);
	a = var;
	args = list1(a);
	goto out;

out:
	a = nod(OCALL, on, N);
	a->list = args;
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

	walkexpr(&expr->left, Erv, init);
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
	NodeList *args, *res, *bod, *nbod, *init, *ln;
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
				nbod = list(n->ninit, nod(OAS, l->left, on));
				n->ninit = nil;
			}
			break;

		case OSEND:
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
	on = syslook("selectgo", 0);
	r = nod(OCALL, on, N);
	r->list = list1(var);		// sel-var
	res = list(res, r);

	// newselect(size uint32) (sel *byte);
	on = syslook("newselect", 0);

	r = nod(OXXX, N, N);
	nodconst(r, types[TINT], count);	// count
	args = list1(r);
	r = nod(OCALL, on, N);
	r->list = args;
	r = nod(OAS, var, r);

	sel->ninit = list1(r);
	sel->nbody = res;
	sel->left = N;

	walkstmtlist(sel->ninit);
	walkstmtlist(sel->nbody);
//dump("sel", sel);

	sel->ninit = concat(sel->ninit, init);
	lineno = lno;
}

Type*
lookdot1(Sym *s, Type *t, Type *f)
{
	Type *r;

	r = T;
	for(; f!=T; f=f->down) {
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
		n->xoffset = f1->width;
		n->type = f1->type;
		if(t->etype == TINTER) {
			if(isptr[n->left->type->etype]) {
				n->left = nod(OIND, n->left, N);	// implicitstar
				walkexpr(&n->left, Elv, nil);
			}
			n->op = ODOTINTER;
		}
		return 1;
	}

	if(f2 != T) {
		tt = n->left->type;
		rcvr = getthisx(f2->type)->type->type;
		if(!eqtype(rcvr, tt)) {
			if(rcvr->etype == tptr && eqtype(rcvr->type, tt)) {
				walkexpr(&n->left, Elv, nil);
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
walkdot(Node *n, NodeList **init)
{
	Type *t;

	walkexprlist(n->ninit, Etop, init);
	if(n->ninit != nil) {
		*init = concat(*init, n->ninit);
		n->ninit = nil;
	}

	if(n->left == N || n->right == N)
		return;
	switch(n->op) {
	case ODOTINTER:
	case ODOTMETH:
		return;	// already done
	}

	walkexpr(&n->left, Erv, init);
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

	if(!lookdot(n, t)) {
		if(!n->diag) {
			n->diag = 1;
			yyerror("undefined: %T field %S", n->left->type, n->right->sym);
		}
	}
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
	convlit(&r, l->type);
	if(!ascompat(l->type, r->type)) {
		badtype(op, l->type, r->type);
		return N;
	}
	if(l->op == ONAME && l->class == PFUNC)
		yyerror("cannot assign to function");

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
		walkexpr(&r, Etop, init);
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
		return list1(convas(nod(OAS, a, r), init));
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
			return nn;
		}

		// normal case -- make a structure of all
		// remaining arguments and pass a pointer to
		// it to the ddd parameter (empty interface)
		return mkdotargs(lr, nn, l, fp, init);
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
		return nn;
	}
	convlit(&r, l->type);
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
Node*
prcompat(NodeList *all, int fmt, int dopanic)
{
	Node *r;
	Node *n;
	NodeList *l;
	Node *on;
	Type *t;
	int notfirst, et;
	NodeList *calls;

	calls = nil;
	notfirst = 0;

	for(l=all; l; l=l->next) {
		if(notfirst) {
			on = syslook("printsp", 0);
			calls = list(calls, nod(OCALL, on, N));
		}
		notfirst = fmt;

		walkexpr(&l->n, Erv, nil);
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
		defaultlit(&n, nil);
		l->n = n;
		if(n->type == T)
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

	if(fmt == 1 && !dopanic) {
		on = syslook("printnl", 0);
		calls = list(calls, nod(OCALL, on, N));
	}
	walkexprlist(calls, Etop, nil);

	if(dopanic)
		r = nodpanic(0);
	else
		r = nod(OEMPTY, N, N);
	walkexpr(&r, Etop, nil);
	r->ninit = calls;
	return r;
}

Node*
nodpanic(int32 lineno)
{
	Node *n, *on;
	NodeList *args;

	on = syslook("panicl", 0);
	n = nodintconst(lineno);
	args = list1(n);
	n = nod(OCALL, on, N);
	n->list = args;
	walkexpr(&n, Etop, nil);
	return n;
}

Node*
makecompat(Node *n)
{
	Type *t;
	Node *l, *r;
	NodeList *args, *init;

//dump("makecompat", n);
	args = n->list;
	if(args == nil) {
		yyerror("make requires type argument");
		return n;
	}
	r = N;
	l = args->n;
	args = args->next;
	init = nil;
	walkexpr(&l, Etype, &init);
	if(l->op != OTYPE) {
		yyerror("cannot make(expr)");
		return n;
	}
	t = l->type;
	n->type = t;
	n->list = args;

	if(t != T)
	switch(t->etype) {
	case TARRAY:
		if(!isslice(t))
			goto bad;
		return arrayop(n, Erv);
	case TMAP:
		return mapop(n, Erv, nil);
	case TCHAN:
		return chanop(n, Erv, nil);
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
	NodeList *args;

	dowidth(t);
	on = syslook("mal", 1);
	argtype(on, t);
	r = nodintconst(t->width);
	args = list1(r);
	r = nod(OCALL, on, N);
	r->list = args;
	walkexpr(&r, Erv, nil);
	return r;
}

Node*
stringop(Node *n, int top, NodeList **init)
{
	Node *r, *c, *on;
	NodeList *args;

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
		args = list1(r);
		c = nod(OCONV, n->right, N);
		c->type = types[TSTRING];
		args = list(args, c);
		r = nod(OCALL, on, N);
		r->list = args;
		c = nodintconst(0);
		r = nod(n->op, r, c);
		break;

	case OADD:
		// sys_catstring(s1, s2)
		on = syslook("catstring", 0);
		r = nod(OCONV, n->left, N);
		r->type = types[TSTRING];
		args = list1(r);
		c = nod(OCONV, n->right, N);
		c->type = types[TSTRING];
		args = list(args, c);
		r = nod(OCALL, on, N);
		r->list = args;
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
			args = list1(r);
			c = nod(OCONV, n->right, N);
			c->type = types[TSTRING];
			args = list(args, c);
			r = nod(OCALL, on, N);
			r->list = args;
			r = nod(OAS, n->left, r);
			break;
		}
		break;

	case OSLICE:
		r = nod(OCONV, n->left, N);
		r->type = types[TSTRING];
		args = list1(r);

		// sys_slicestring(s, lb, hb)
		r = nod(OCONV, n->right->left, N);
		r->type = types[TINT];
		args = list(args, r);

		c = nod(OCONV, n->right->right, N);
		c->type = types[TINT];
		args = list(args, c);

		on = syslook("slicestring", 0);
		r = nod(OCALL, on, N);
		r->list = args;
		break;

	case OINDEX:
		// sys_indexstring(s, i)
		r = nod(OCONV, n->left, N);
		r->type = types[TSTRING];
		args = list1(r);

		r = nod(OCONV, n->right, N);
		r->type = types[TINT];
		args = list(args, r);
		on = syslook("indexstring", 0);
		r = nod(OCALL, on, N);
		r->list = args;
		break;

	case OCONV:
		// sys_intstring(v)
		r = nod(OCONV, n->left, N);
		r->type = types[TINT64];
		args = list1(r);
		on = syslook("intstring", 0);
		r = nod(OCALL, on, N);
		r->list = args;
		break;

	case OARRAY:
		// arraystring([]byte) string;
		on = syslook("arraystring", 0);
		r = n->left;

		if(r->type != T && r->type->type != T) {
			if(istype(r->type->type, TINT) || istype(r->type->type->type, TINT)) {
				// arraystring([]byte) string;
				on = syslook("arraystringi", 0);
			}
		}
		args = list1(r);
		r = nod(OCALL, on, N);
		r->list = args;
		break;
	}

	walkexpr(&r, top, init);
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
mapop(Node *n, int top, NodeList **init)
{
	Node *r, *a, *l;
	Type *t;
	Node *on;
	int cl, cr;
	NodeList *args;

	r = n;
	switch(n->op) {
	default:
		fatal("mapop: unknown op %O", n->op);

	case OMAKE:
		cl = count(n->list);
		if(cl > 1)
			yyerror("too many arguments to make map");

		if(!(top & Erv))
			goto nottop;

		// newmap(keysize int, valsize int,
		//	keyalg int, valalg int,
		//	hint int) (hmap map[any-1]any-2);

		t = fixmap(n->type);
		if(t == T)
			break;

		a = nodintconst(t->down->width);	// key width
		args = list1(a);
		a = nodintconst(t->type->width);	// val width
		args = list(args, a);
		a = nodintconst(algtype(t->down));	// key algorithm
		args = list(args, a);
		a = nodintconst(algtype(t->type));	// val algorithm
		args = list(args, a);

		if(cl == 1)
			a = n->list->n;				// hint
		else
			a = nodintconst(0);
		args = list(args, a);

		on = syslook("newmap", 1);

		argtype(on, t->down);	// any-1
		argtype(on, t->type);	// any-2

		r = nod(OCALL, on, N);
		r->list = args;
		walkexpr(&r, top, nil);
		r->type = n->type;
		break;

	case OINDEX:
		if(!(top & Erv))
			goto nottop;
		// mapaccess1(hmap map[any]any, key any) (val any);

		t = fixmap(n->left->type);
		if(t == T)
			break;

		convlit(&n->right, t->down);

		if(!eqtype(n->right->type, t->down)) {
			badtype(n->op, n->right->type, t->down);
			break;
		}

		a = n->left;				// map
		args = list1(a);
		a = n->right;				// key
		args = list(args, a);

		on = syslook("mapaccess1", 1);

		argtype(on, t->down);	// any-1
		argtype(on, t->type);	// any-2
		argtype(on, t->down);	// any-3
		argtype(on, t->type);	// any-4

		r = nod(OCALL, on, N);
		r->list = args;
		walkexpr(&r, Erv, nil);
		r->type = t->type;
		break;

	case OAS:
		// mapassign1(hmap map[any-1]any-2, key any-3, val any-4);
		if(n->left->op != OINDEX)
			goto shape;

		t = fixmap(n->left->left->type);
		if(t == T)
			break;

		a = n->left->left;			// map
		args = list1(a);
		a = n->left->right;			// key
		args = list(args, a);
		a = n->right;				// val
		args = list(args, a);

		on = syslook("mapassign1", 1);

		argtype(on, t->down);	// any-1
		argtype(on, t->type);	// any-2
		argtype(on, t->down);	// any-3
		argtype(on, t->type);	// any-4

		r = nod(OCALL, on, N);
		r->list = args;
		walkexpr(&r, Etop, init);
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
		if(l->op != OINDEX)
			goto shape;

		t = fixmap(l->left->type);
		if(t == T)
			break;

		args = list1(l->left);			// map
		args = list(args, l->right);		// key
		args = list(args, n->rlist->n);		// val
		args = list(args, n->rlist->next->n);	// pres

		on = syslook("mapassign2", 1);

		argtype(on, t->down);	// any-1
		argtype(on, t->type);	// any-2
		argtype(on, t->down);	// any-3
		argtype(on, t->type);	// any-4

		r = nod(OCALL, on, N);
		r->list = args;
		walkexpr(&r, Etop, init);
		break;

	access2:
		// mapaccess2(hmap map[any-1]any-2, key any-3) (val-4 any, pres bool);

//dump("access2", n);
		r = n->rlist->n;
		if(r->op != OINDEX)
			goto shape;

		t = fixmap(r->left->type);
		if(t == T)
			break;

		args = list1(r->left);		// map
		args = list(args, r->right);		// key

		on = syslook("mapaccess2", 1);

		argtype(on, t->down);	// any-1
		argtype(on, t->type);	// any-2
		argtype(on, t->down);	// any-3
		argtype(on, t->type);	// any-4

		a = nod(OCALL, on, N);
		a->list = args;
		n->rlist = list1(a);
		walkexpr(&n, Etop, init);
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
		walkexpr(&r, Etop, init);
		*init = list(*init, r);

		a = nod(OXXX, N, N);
		*a = *n->left;		// copy of map[tmpi]
		a = nod(n->etype, a, n->right);		// m[tmpi] op right
		r = nod(OAS, n->left, a);		// map[tmpi] = map[tmpi] op right
		walkexpr(&r, Etop, init);
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
chanop(Node *n, int top, NodeList **init)
{
	Node *r, *a, *on;
	NodeList *args;
	Type *t;
	int cl, cr;

	r = n;
	switch(n->op) {
	default:
		fatal("chanop: unknown op %O", n->op);

	case OCLOSE:
		cl = count(n->list);
		if(cl > 1)
			yyerror("too many arguments to close");
		else if(cl < 1)
			yyerror("missing argument to close");
		n->left = n->list->n;

		// closechan(hchan *chan any);
		t = fixchan(n->left->type);
		if(t == T)
			break;

		a = n->left;			// chan
		args = list1(a);

		on = syslook("closechan", 1);
		argtype(on, t);	// any-1

		r = nod(OCALL, on, N);
		r->list = args;
		walkexpr(&r, top, nil);
		r->type = n->type;
		break;

	case OCLOSED:
		cl = count(n->list);
		if(cl > 1)
			yyerror("too many arguments to closed");
		else if(cl < 1)
			yyerror("missing argument to closed");
		n->left = n->list->n;

		// closedchan(hchan *chan any) bool;
		t = fixchan(n->left->type);
		if(t == T)
			break;

		a = n->left;			// chan
		args = list1(a);

		on = syslook("closedchan", 1);
		argtype(on, t);	// any-1

		r = nod(OCALL, on, N);
		r->list = args;
		walkexpr(&r, top, nil);
		n->type = r->type;
		break;

	case OMAKE:
		cl = count(n->list);
		if(cl > 1)
			yyerror("too many arguments to make chan");

		// newchan(elemsize int, elemalg int,
		//	hint int) (hmap *chan[any-1]);

		t = fixchan(n->type);
		if(t == T)
			break;

		a = nodintconst(t->type->width);	// elem width
		args = list1(a);
		a = nodintconst(algtype(t->type));	// elem algorithm
		args = list(args, a);
		a = nodintconst(0);
		if(cl == 1) {
			// async buf size
			a = nod(OCONV, n->list->n, N);
			a->type = types[TINT];
		}
		args = list(args, a);

		on = syslook("newchan", 1);
		argtype(on, t->type);	// any-1

		r = nod(OCALL, on, N);
		r->list = args;
		walkexpr(&r, top, nil);
		r->type = n->type;
		break;

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

		a = r->left;			// chan
		args = list1(a);

		on = syslook("chanrecv2", 1);

		argtype(on, t->type);	// any-1
		argtype(on, t->type);	// any-2
		r = nod(OCALL, on, N);
		r->list = args;
		n->rlist->n = r;
		r = n;
		walkexpr(&r, Etop, init);
		break;

	case ORECV:
		// should not happen - nonblocking is OAS w/ ORECV now.
		if(n->right != N) {
			dump("recv2", n);
			fatal("chanop recv2");
		}

		// chanrecv1(hchan *chan any) (elem any);
		defaultlit(&n->left, T);
		t = fixchan(n->left->type);
		if(t == T)
			break;

		if(!(t->chan & Crecv)) {
			yyerror("cannot receive from %T", t);
			break;
		}

		a = n->left;			// chan
		args = list1(a);

		on = syslook("chanrecv1", 1);

		argtype(on, t->type);	// any-1
		argtype(on, t->type);	// any-2
		r = nod(OCALL, on, N);
		r->list = args;
		walkexpr(&r, Erv, nil);
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
		a = n->left;			// chan
		args = list1(a);
		a = n->right;			// e
		args = list(args, a);

		on = syslook("chansend1", 1);
		argtype(on, t->type);	// any-1
		argtype(on, t->type);	// any-2
		r = nod(OCALL, on, N);
		r->list = args;
		walkexpr(&r, Etop, nil);
		break;

	send2:
		// chansend2(hchan *chan any, val any) (pres bool);
		a = n->left;			// chan
		args = list1(a);
		a = n->right;			// e
		args = list(args, a);

		on = syslook("chansend2", 1);
		argtype(on, t->type);	// any-1
		argtype(on, t->type);	// any-2
		r = nod(OCALL, on, N);
		r->list = args;
		walkexpr(&r, Etop, nil);
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
	NodeList *args;
	Type *t, *tl;
	Node *on;
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

		args = list1(n->left);	// old

		a = nodintconst(t->bound);		// nel
		a = nod(OCONV, a, N);
		a->type = types[TINT];
		args = list(args, a);

		on = syslook("arrays2d", 1);
		argtype(on, t);				// any-1
		argtype(on, tl->type);			// any-2
		r = nod(OCALL, on, N);
		r->list = args;
		n->left = r;
		walkexpr(&n, top, nil);
		return n;

	case OAS:
		r = nod(OCONV, n->right, N);
		r->type = n->left->type;
		n->right = arrayop(r, Erv);
		return n;

	case OMAKE:
		cl = count(n->list);
		if(cl > 2)
			yyerror("too many arguments to make array");

		// newarray(nel int, max int, width int) (ary []any)
		t = fixarray(n->type);
		if(t == T)
			break;

		// nel
		a = n->list->n;
		if(a == N) {
			yyerror("new slice must have size");
			a = nodintconst(1);
		}
		a = nod(OCONV, a, N);
		a->type = types[TINT];
		args = list1(a);

		// max
		if(cl < 2)
			a = nodintconst(0);
		else
			a = n->list->next->n;
		a = nod(OCONV, a, N);
		a->type = types[TINT];
		args = list(args, a);

		// width
		a = nodintconst(t->type->width);	// width
		a = nod(OCONV, a, N);
		a->type = types[TINT];
		args = list(args, a);

		on = syslook("newarray", 1);
		argtype(on, t->type);			// any-1
		r = nod(OCALL, on, N);
		r->list = args;
		walkexpr(&r, top, nil);
		r->type = t;	// if t had a name, going through newarray lost it
		break;

	case OSLICE:
		// arrayslices(old any, nel int, lb int, hb int, width int) (ary []any)
		// arraysliced(old []any, lb int, hb int, width int) (ary []any)

		t = fixarray(n->left->type);
		if(t == T)
			break;

		if(t->bound >= 0) {
			// static slice
			a = nod(OADDR, n->left, N);		// old
			args = list1(a);

			a = nodintconst(t->bound);		// nel
			a = nod(OCONV, a, N);
			a->type = types[TINT];
			args = list(args, a);

			on = syslook("arrayslices", 1);
			argtype(on, t);				// any-1
			argtype(on, t->type);			// any-2
		} else {
			// dynamic slice
			a = n->left;				// old
			args = list1(a);

			on = syslook("arraysliced", 1);
			argtype(on, t->type);			// any-1
			argtype(on, t->type);			// any-2
		}

		a = nod(OCONV, n->right->left, N);	// lb
		a->type = types[TINT];
		args = list(args, a);

		a = nod(OCONV, n->right->right, N);	// hb
		a->type = types[TINT];
		args = list(args, a);

		a = nodintconst(t->type->width);	// width
		a = nod(OCONV, a, N);
		a->type = types[TINT];
		args = list(args, a);

		r = nod(OCALL, on, N);
		r->list = args;
		walkexpr(&r, top, nil);
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
	Node *r, *on;
	NodeList *args;

	tr = n->type;

	switch(et) {
	default:
		fatal("ifacecvt: unknown op %d\n", et);

	case T2I:
		// ifaceT2I(sigi *byte, sigt *byte, elem any) (ret any);
		args = list1(typename(tl));	// sigi
		args = list(args, typename(tr));	// sigt
		args = list(args, n);	// elem

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
	walkexpr(&r, Erv, nil);
	return r;
}

Node*
ifaceop(Node *n)
{
	Node *r, *on;
	NodeList *args;

	switch(n->op) {
	default:
		fatal("ifaceop %O", n->op);

	case OEQ:
	case ONE:
		// ifaceeq(i1 any-1, i2 any-2) (ret bool);
		args = list1(n->left);		// i1
		args = list(args, n->right);	// i2

		if(!eqtype(n->left->type, n->right->type))
			fatal("ifaceop %O %T %T", n->op, n->left->type, n->right->type);
		if(isnilinter(n->left->type))
			on = syslook("efaceeq", 1);
		else
			on = syslook("ifaceeq", 1);
		argtype(on, n->right->type);
		argtype(on, n->left->type);

		r = nod(OCALL, on, N);
		r->list = args;
		if(n->op == ONE)
			r = nod(ONOT, r, N);
		walkexpr(&r, Erv, nil);
		return r;
	}
}

Node*
convas(Node *n, NodeList **init)
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
		n = mapop(n, Elv, init);
		goto out;
	}

	if(n->left->op == OSEND)
	if(n->left->type != T) {
		n = chanop(n, Elv, init);
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
		n = arrayop(n, Etop);
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
			if(nr->left->op == ONAME && nr->left->etype != 0)
				break;
			walkexpr(&nr->left, Erv | Etype, &init);
			if(nr->left->op == OTYPE)
				break;
			goto call;
		case OCALLMETH:
		case OCALLINTER:
			walkexpr(&nr->left, Erv, &init);
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

		walkexpr(&r, Erv, &init);
		defaultlit(&r, T);
		saver->n = r;
		a = mixedoldnew(l, r->type);
		n = list(n, a);
	}
	n = checkmixed(n, &init);
	goto out;

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
		walkexpr(&nr->left, Erv, &init);
		implicitstar(&nr->left);
		t = nr->left->type;
		if(!istype(t, TMAP))
			goto badt;
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
		walkexpr(&nr->left, Erv, &init);
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
		yyerror("assignment count mismatch: %d = %d", cl, cr);
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

	walkexpr(&nn->right, Erv, &init);
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
structlit(Node *n, Node *var, NodeList **init)
{
	Iter savel;
	Type *l, *t;
	Node *r, *a;
	Node* hash[101];
	NodeList *nl;
	int nerr;

	nerr = nerrors;
	t = n->type;
	if(t->etype != TSTRUCT)
		fatal("structlit: not struct");

	if(var == N) {
		var = nod(OXXX, N, N);
		tempname(var, t);
	}

	nl = n->list;
	if(nl == nil || nl->n->op == OKEY)
		goto keyval;

	l = structfirst(&savel, &n->type);
	for(; nl; nl=nl->next) {
		r = nl->n;
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
		walkexpr(&a, Etop, init);
		if(nerr != nerrors)
			return var;
		*init = list(*init, a);

		l = structnext(&savel);
	}
	if(l != T)
		yyerror("struct literal expect expr of type %T", l);
	if(nl != nil)
		yyerror("struct literal too many expressions");
	return var;

keyval:
	memset(hash, 0, sizeof(hash));
	a = nod(OAS, var, N);
	walkexpr(&a, Etop, init);
	*init = list(*init, a);

	for(; nl; nl=nl->next) {
		r = nl->n;

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
		walkexpr(&a, Etop, init);
		if(nerr != nerrors)
			break;

		*init = list(*init, a);
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
arraylit(Node *n, Node *var, NodeList **init)
{
	Type *t;
	Node *r, *a;
	NodeList *l;
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

	for(l=n->list; l; l=l->next) {
		r = l->n;
		if(r->op == OKEY) {
			evconst(r->left);
			b = nonnegconst(r->left);
		}
		b++;
		if(b > ninit)
			ninit = b;
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
		a = nod(OMAKE, N, N);
		a->list = list(list1(typenod(t)), nodintconst(ninit));
		a = nod(OAS, var, a);
		walkexpr(&a, Etop, init);
		*init = list(*init, a);
	} else {
		// if entire array isnt initialized,
		// then clear the array
		if(ninit < b) {
			a = nod(OAS, var, N);
			walkexpr(&a, Etop, init);
			*init = list(*init, a);
		}
	}

	b = 0;
	memset(hash, 0, sizeof(hash));
	for(l=n->list; l; l=l->next) {
		r = l->n;
		// build list of var[c] = expr
		if(r->op == OKEY) {
			b = nonnegconst(r->left);
			if(b < 0) {
				yyerror("array index must be non-negative constant");
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
		walkexpr(&a, Etop, init);	// add any assignments in r to top
		if(nerr != nerrors)
			break;

		*init = list(*init, a);
		b++;
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
	walkexpr(&a, Etop, init);
	*init = list(*init, a);

	memset(hash, 0, sizeof(hash));
	for(l=n->list; l; l=l->next) {
		r = l->n;
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
		walkexpr(&a, Etop, init);
		if(nerr != nerrors)
			break;

		*init = list(*init, a);
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
