// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go.h"

static void	implicitstar(Node**);
static int	onearg(Node*);
static int	lookdot(Node*, Type*);
static int	convert(Node**, Type*, int);

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
	int et, et1, et2, op, nerr, len;
	NodeList *ll;
	Node *n, *l, *r;
	NodeList *args;
	int i, lno, ok;
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

reswitch:
	ok = 0;
	switch(n->op) {
	default:
		// until typecheck is complete, do nothing.
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
			switch(l->op) {
			default:
				yyerror("invalid array bound %#N", l);
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
	 * type or expr
	 */
	case OIND:
		l = typecheck(&n->left, top | Etype);
		if((t = l->type) == T)
			goto error;
		if(l->op == OTYPE) {
			ok |= Etype;
			n->op = OTYPE;
			n->type = ptrto(l->type);
			n->left = N;
			goto ret;
		}
		if(!isptr[t->etype]) {
			yyerror("invalid indirect %#N (non-pointer type %T)", n, t);
			goto error;
		}
		n->type = t->type;
		goto ret;

	/*
	 * arithmetic exprs
	 */
	case OASOP:
		ok |= Etop;
		l = typecheck(&n->left, Elv);
		r = typecheck(&n->right, Erv);
		if(l->type == T || r->type == T)
			goto error;
		op = n->etype;
		goto arith;

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
	case OLSH:
	case ORSH:
	case OMOD:
	case OMUL:
	case ONE:
	case OOR:
	case OOROR:
	case OSUB:
	case OXOR:
		ok |= Erv;
		l = typecheck(&n->left, Erv);
		r = typecheck(&n->right, Erv);
		if(l->type == T || r->type == T)
			goto error;
		op = n->op;
	arith:
		if(op == OLSH || op == ORSH)
			goto shift;
		// ideal mixed with non-ideal
		defaultlit2(&l, &r, 0);
		n->left = l;
		n->right = r;
		t = l->type;
		if(t->etype == TIDEAL)
			t = r->type;
		et = t->etype;
		if(et == TIDEAL)
			et = TINT;
		if(t->etype != TIDEAL && !eqtype(l->type, r->type)) {
		badbinary:
			yyerror("invalid operation: %#N (type %T %#O %T)", n, l->type, op, r->type);
			goto error;
		}
		if(!okfor[op][et])
			goto badbinary;
		// okfor allows any array == array;
		// restrict to slice == nil and nil == slice.
		if(l->type->etype == TARRAY && !isslice(l->type))
			goto badbinary;
		if(r->type->etype == TARRAY && !isslice(r->type))
			goto badbinary;
		if(isslice(l->type) && !isnil(l) && !isnil(r))
			goto badbinary;
		t = l->type;
		if(iscmp[n->op])
			t = types[TBOOL];
		n->type = t;
		goto ret;

	shift:
		defaultlit(&r, types[TUINT]);
		n->right = r;
		t = r->type;
		if(!isint[t->etype] || issigned[t->etype]) {
			yyerror("invalid operation: %#N (shift count type %T)", n, r->type);
			goto error;
		}
		// no defaultlit for left
		// the outer context gives the type
		n->type = l->type;
		goto ret;

	case OCOM:
	case OMINUS:
	case ONOT:
	case OPLUS:
		ok |= Erv;
		l = typecheck(&n->left, Erv);
		if((t = l->type) == T)
			goto error;
		if(!okfor[n->op][t->etype]) {
			yyerror("invalid operation: %#O %T", n->op, t);
			goto error;
		}
		n->type = t;
		goto ret;

	/*
	 * exprs
	 */
	case OADDR:
		l = typecheck(&n->left, Elv);
		if((t = l->type) == T)
			goto error;
		n->type = ptrto(t);
		goto ret;

	case OCOMPOS:
		l = typecheck(&n->right /* sic */, Etype /* TODO | Edotarray */);
		if((t = l->type) == T)
			goto error;
		nerr = nerrors;
		switch(t->etype) {
		case TARRAY:
			len = 0;
			i = 0;
			for(ll=n->list; ll; ll=ll->next) {
				l = ll->n;
				if(l->op == OKEY) {
					typecheck(&l->left, Erv);
					evconst(l->left);
					i = nonnegconst(l->left);
					typecheck(&l->right, Erv);
					defaultlit(&l->right, t->type);
					// TODO more forceful conversion of l->right
				} else {
					typecheck(&ll->n, Erv);
					defaultlit(&ll->n, t->type);
					// TODO more forceful conversion
				}
				i++;
				if(i > len) {
					len = i;
					if(t->bound >= 0 && len > t->bound) {
						setlineno(l);
						yyerror("array index out of bounds");
						t->bound = -1;	// no more errors
					}
				}
			}
			if(t->bound == -100)
				t->bound = len;
			break;

		case TMAP:
			for(ll=n->list; ll; ll=ll->next) {
				l = ll->n;
				if(l->op != OKEY) {
					yyerror("missing key in map literal");
					continue;
				}
				typecheck(&l->left, Erv);
				typecheck(&l->right, Erv);
				defaultlit(&l->left, t->down);
				defaultlit(&l->right, t->type);
				// TODO more forceful
			}
			break;

		case TSTRUCT:
		//	fatal("compos %T", t);
			;
		}
		if(nerr != nerrors)
			goto error;
		n->type = t;
		goto ret;

	case ODOT:
		l = typecheck(&n->left, Erv);
		if((t = l->type) == T)
			goto error;
		if(n->right->op != ONAME) {
			yyerror("rhs of . must be a name");	// impossible
			goto error;
		}
		if(isptr[t->etype]) {
			t = t->type;
			if(t == T)
				goto error;
			n->op = ODOTPTR;
		}
		if(!lookdot(n, t)) {
			yyerror("%#N undefined (%S in type %T)", n, n->right->sym, t);
			goto error;
		}
		switch(n->op) {
		case ODOTINTER:
		case ODOTMETH:
			ok |= Ecall;
			break;
		default:
			ok |= Erv;
			// TODO ok |= Elv sometimes
			break;
		}
		goto ret;

	case ODOTTYPE:
		typecheck(&n->left, Erv);
		defaultlit(&n->left, T);
		l = n->left;
		if((t = l->type) == T)
			goto error;
		if(!isinter(t)) {
			yyerror("invalid type assertion: %#N (non-interface type %T on left)", n, t);
			goto error;
		}
		if(n->right != N) {
			typecheck(&n->right, Etype);
			n->type = n->right->type;
			n->right = N;
			if(n->type == T)
				goto error;
		}
		goto ret;

	case OINDEX:
		typecheck(&n->left, Erv);
		defaultlit(&n->left, T);
		implicitstar(&n->left);
		l = n->left;
		typecheck(&n->right, Erv);
		r = n->right;
		if((t = l->type) == T || r->type == T)
			goto error;
		switch(t->etype) {
		default:
			yyerror("invalid operation: %#N (index of type %T)", n, t);
			goto error;

		case TARRAY:
			ok |= Erv | Elv;
			defaultlit(&n->right, types[TUINT]);
			n->type = t->type;
			break;

		case TMAP:
			ok |= Erv | Elv;
			defaultlit(&n->right, t->down);
			n->type = t->type;
			break;

		case TSTRING:
			ok |= Erv;
			defaultlit(&n->right, types[TUINT]);
			n->type = types[TUINT8];
			break;
		}
		goto ret;

	case ORECV:
		typecheck(&n->left, Erv);
		defaultlit(&n->left, T);
		l = n->left;
		if((t = l->type) == T)
			goto error;
		if(t->etype != TCHAN) {
			yyerror("invalid operation: %#N (recv from non-chan type %T)", n, t);
			goto error;
		}
		if(!(t->chan & Crecv)) {
			yyerror("invalid operation: %#N (recv from send-only type %T)", n, t);
			goto error;
		}
		n->type = t->type;
		ok |= Erv;
		goto ret;

	case OSEND:
		l = typecheck(&n->left, Erv);
		typecheck(&n->right, Erv);
		defaultlit(&n->left, T);
		l = n->left;
		if((t = l->type) == T)
			goto error;
		if(!(t->chan & Csend)) {
			yyerror("invalid operation: %#N (send to recv-only type %T)", n, t);
			goto error;
		}
		defaultlit(&n->right, t->type);
		r = n->right;
		if((t = r->type) == T)
			goto error;
		// TODO: more aggressive
		ok |= Etop | Erv;
		n->type = types[TBOOL];
		goto ret;

	case OSLICE:
		ok |= Erv;
		typecheck(&n->left, top);
		typecheck(&n->right->left, Erv);
		typecheck(&n->right->right, Erv);
		defaultlit(&n->left, T);
		defaultlit(&n->right->left, types[TUINT]);
		defaultlit(&n->right->right, types[TUINT]);
		implicitstar(&n->left);
		if(n->right->left == N || n->right->right == N) {
			yyerror("missing slice bounds?");
			goto error;
		}
		if((t = n->right->left->type) == T)
			goto error;
		if(!isint[t->etype]) {
			yyerror("invalid array index %#N (type %T)", n->right->left, t);
			goto error;
		}
		if((t = n->right->right->type) == T)
			goto error;
		if(!isint[t->etype]) {
			yyerror("invalid array index %#N (type %T)", n->right->right, t);
			goto error;
		}
		l = n->left;
		if((t = l->type) == T)
			goto error;
		// TODO(rsc): 64-bit slice index needs to be checked
		// for overflow in generated code
		switch(t->etype) {
		default:
			yyerror("invalid operation: %#N (slice of type %T)", n, t);
			goto error;

		case TARRAY:
			ok |= Elv;
			n = arrayop(n, Erv);
			break;

		case TSTRING:
			n = stringop(n, Erv, nil);
			break;
		}
		goto ret;

	/*
	 * call and call like
	 */
	case OCALL:
		l = n->left;
		if(l->op == ONAME && l->etype != 0) {
			// builtin: OLEN, OCAP, etc.
			n->op = l->etype;
			n->left = n->right;
			n->right = N;
			goto reswitch;
		}
		l = typecheck(&n->left, Erv | Etype | Ecall);
		typechecklist(n->list, Erv);
		if((t = l->type) == T)
{
yyerror("skip %#N", n);
			goto error;
}
		if(l->op == OTYPE) {
			ok |= Erv;
			// turn CALL(type, arg) into CONV(arg) w/ type
			n->left = N;
			if(onearg(n) < 0)
				goto error;
			n->op = OCONV;
			n->type = l->type;
			goto doconv;
		}
		// TODO: check args
		if(t->outtuple == 0) {
			ok |= Etop;
			goto ret;
		}
		if(t->outtuple == 1) {
			ok |= Erv;
			t = getoutargx(l->type)->type;
			if(t->etype == TFIELD)
				t = t->type;
			n->type = t;
			goto ret;
		}
		// multiple return
		// ok |= Emulti;
		n->type = getoutargx(l->type);
		goto ret;

	case OCAP:
	case OLEN:
		if(onearg(n) < 0)
			goto error;
		typecheck(&n->left, Erv);
		defaultlit(&n->left, T);
		implicitstar(&n->left);
		l = n->left;
		if((t = l->type) == T)
			goto error;
		switch(n->op) {
		case OCAP:
			if(!okforcap[t->etype])
				goto badcall1;
			break;
		case OLEN:
			if(!okforlen[t->etype])
				goto badcall1;
			break;
		}
		// might be constant
		switch(t->etype) {
		case TSTRING:
			if(isconst(l, CTSTR))
				nodconst(n, types[TINT], l->val.u.sval->len);
			break;
		case TARRAY:
			if(t->bound >= 0)
				nodconst(n, types[TINT], t->bound);
			break;
		}
		n->type = types[TINT];
		goto ret;

	case OCLOSED:
		ok |= Erv;
	case OCLOSE:
		ok |= Etop;
		if(onearg(n) < 0)
			goto error;
		typecheck(&n->left, Erv);
		defaultlit(&n->left, T);
		l = n->left;
		if((t = l->type) == T)
			goto error;
		if(t->etype != TCHAN) {
			yyerror("invalid operation: %#N (non-chan type %T)", n, t);
			goto error;
		}
		goto ret;

	case OCONV:
	doconv:
		typecheck(&n->left, Erv);
		defaultlit(&n->left, n->type);
		if((t = n->left->type) == T)
			goto error;
		switch(convert(&n->left, n->type, 1)) {
		case -1:
			goto error;
		case 0:
			n = n->left;
			break;
		case OCONV:
			break;
		}
		goto ret;

	case OMAKE:
		args = n->list;
		if(args == nil) {
			yyerror("missing argument to make");
			goto error;
		}
		l = args->n;
		args = args->next;
		typecheck(&l, Etype);
		if((t = l->type) == T)
			goto error;

		switch(t->etype) {
		default:
		badmake:
			yyerror("cannot make type %T", t);
			goto error;

		case TARRAY:
			if(!isslice(t))
				goto badmake;
			if(args == nil) {
				yyerror("missing len argument to make(%T)", t);
				goto error;
			}
			l = args->n;
			args = args->next;
			typecheck(&l, Erv);
			defaultlit(&l, types[TUINT]);
			r = N;
			if(args != nil) {
				r = args->n;
				args = args->next;
				typecheck(&r, Erv);
				defaultlit(&r, types[TUINT]);
			}
			if(l->type == T || (r && r->type == T))
				goto error;
			if(!isint[l->type->etype]) {
				yyerror("non-integer len argument to make(%T)", t);
				goto error;
			}
			if(r && !isint[r->type->etype]) {
				yyerror("non-integer cap argument to make(%T)", t);
				goto error;
			}
			n->left = l;
			n->right = r;
			break;

		case TMAP:
			if(args != nil) {
				l = args->n;
				args = args->next;
				typecheck(&l, Erv);
				defaultlit(&l, types[TUINT]);
				if(l->type == T)
					goto error;
				if(!isint[l->type->etype]) {
					yyerror("non-integer size argument to make(%T)", t);
					goto error;
				}
				n->left = l;
			}
			break;

		case TCHAN:
			l = N;
			if(args != nil) {
				l = args->n;
				args = args->next;
				typecheck(&l, Erv);
				defaultlit(&l, types[TUINT]);
				if(l->type == T)
					goto error;
				if(!isint[l->type->etype]) {
					yyerror("non-integer buffer argument to make(%T)", t);
					goto error;
				}
				n->left = l;
			}
			break;
		}
		if(args != nil) {
			yyerror("too many arguments to make(%T)", t);
			goto error;
		}
		n->type = t;
		goto ret;

	case ONEW:
		args = n->list;
		if(args == nil) {
			yyerror("missing argument to new");
			goto error;
		}
		l = args->n;
		typecheck(&l, Etype);
		if((t = l->type) == T)
			goto error;
		if(args->next != nil) {
			yyerror("too many arguments to new(%T)", t);
			goto error;
		}
		n->left = l;
		n->type = ptrto(t);
		goto ret;

	case OPANIC:
	case OPANICN:
	case OPRINT:
	case OPRINTN:
		typechecklist(n->list, Erv);
		goto ret;

	/*
	 * statements
	 */
	case OAS:
		typecheck(&n->left, Elv);
		typecheck(&n->right, Erv);
		goto ret;

	case OAS2:
		typechecklist(n->list, Elv);
		typechecklist(n->rlist, Erv);
		goto ret;

	case OBREAK:
	case OCONTINUE:
	case ODCL:
	case OEMPTY:
	case OGOTO:
	case OLABEL:
	case OXFALL:
		goto ret;

	case ODEFER:
	case OPROC:
		typecheck(&n->left, Etop);
		goto ret;

	case OFOR:
		typechecklist(n->ninit, Etop);
		typecheck(&n->ntest, Erv);	// TODO Ebool
		typecheck(&n->nincr, Etop);
		typechecklist(n->nbody, Etop);
		goto ret;

	case OIF:
		typechecklist(n->ninit, Etop);
		typecheck(&n->ntest, Erv);	// TODO Ebool
		typechecklist(n->nbody, Etop);
		typechecklist(n->nelse, Etop);
		goto ret;

	case ORETURN:
		typechecklist(n->list, Erv);
		// TODO convert
		goto ret;

	case OSELECT:
		typechecklist(n->ninit, Etop);
		typecheck(&n->ntest, Erv);
		typechecklist(n->list, Etop);
		goto ret;

	case OSWITCH:
		typechecklist(n->ninit, Etop);
		typecheck(&n->ntest, Erv);
		typechecklist(n->list, Etop);
		goto ret;

	case OTYPECASE:
		typecheck(&n->left, Elv);
		goto ret;

	case OTYPESW:
		typecheck(&n->right, Erv);
		goto ret;

	case OXCASE:
		typechecklist(n->list, Erv);
		typechecklist(n->nbody, Etop);
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
	if((ok & Ecall) && !(top & Ecall)) {
		yyerror("must call method %#N", n);
		goto error;
	}

	/* TODO
	if(n->type == T)
		fatal("typecheck nil type");
	*/
	goto out;

badcall1:
	yyerror("invalid argument %#N (type %T) for %#O", n->left, n->left->type, n->op);
	goto error;

error:
	n->type = T;

out:
	lineno = lno;
	n->typecheck = 1;
	*np = n;
	return n;
}

static void
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
	typecheck(&n, Erv);
	*nn = n;
}

static int
onearg(Node *n)
{
	if(n->left != N)
		return 0;
	if(n->list == nil) {
		yyerror("missing argument to %#O - %#N", n->op, n);
		return -1;
	}
	n->left = n->list->n;
	if(n->list->next != nil) {
		yyerror("too many arguments to %#O", n->op);
		n->list = nil;
		return -1;
	}
	n->list = nil;
	return 0;
}

static Type*
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

static int
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
				typecheck(&n->left, Elv);
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
				typecheck(&n->left, Elv);
				addrescapes(n->left);
				n->left = nod(OADDR, n->left, N);
				typecheck(&n->left, Erv);
			} else if(tt->etype == tptr && eqtype(tt->type, rcvr)) {
				n->left = nod(OIND, n->left, N);
				typecheck(&n->left, Erv);
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

/*
 * try to convert *np to t.
 * explicit means conversion like int64(n).
 * not explicit means assignment, return, or function call parameter.
 * return -1 for failure, 0 if OCONV node not needed, 1 if OCONV is needed.
 */
static int
convert(Node **np, Type *t, int explicit)
{
	int et;
	Node *n, *n1;
	Type *tt;

	n = *np;

	if(n->type == t)
		return 0;

	if(eqtype(n->type, t))
		return OCONV;

	// XXX wtf?
	convlit1(&n, t, explicit);
	if(n->type == T)
		return -1;

	// no-op conversion
	if(cvttype(t, n->type) == 1) {
	nop:
		if(n->op == OLITERAL) {
			// can convert literal in place
			n1 = nod(OXXX, N, N);
			*n1 = *n;
			n1->type = t;
			*np = n1;
			return 0;
		}
		return OCONV;
	}

	if(!explicit) {
		yyerror("cannot use %#N (type %T) as type %T", n, n->type, t);
		return -1;
	}

	// simple fix-float
	if(isint[n->type->etype] || isfloat[n->type->etype])
	if(isint[t->etype] || isfloat[t->etype]) {
		// evconst(n);	// XXX is this needed?
		return OCONV;
	}

	// to/from interface.
	// ifaceas1 will generate a good error if the conversion fails.
	if(t->etype == TINTER || n->type->etype == TINTER) {
		n = ifacecvt(t, n, ifaceas1(t, n->type, 0));
		n->type = t;
		*np = n;
		return 0;
	}

	// to string
	if(istype(t, TSTRING)) {
		// integer rune
		et = n->type->etype;
		if(isint[et]) {
		//	xxx;
			return OCONVRUNE;
		}

		// []byte and *[10]byte -> string
		tt = T;
		if(isptr[et] && isfixedarray(n->type->type))
			tt = n->type->type->type;
		else if(isslice(n->type))
			tt = n->type->type;
		if(tt) {
			if(tt->etype == TUINT8)
				return OCONVSTRB;
			if(tt->etype == TINT)
				return OCONVSTRI;
		}
	}

	// convert static array to slice
	if(isslice(t) && isptr[n->type->etype] && isfixedarray(n->type->type)
	&& eqtype(t->type, n->type->type->type))
		return OCONVA2S;

	// convert to unsafe pointer
	if(isptrto(t, TANY)
	&& (isptr[n->type->etype] || n->type->etype == TUINTPTR))
		return OCONV;

	// convert from unsafe pointer
	if(isptrto(n->type, TANY)
	&& (isptr[t->etype] || t->etype == TUINTPTR))
		return OCONV;

	yyerror("cannot convert %#N (type %T) to type %T", n, n->type, t);
	return -1;
}
