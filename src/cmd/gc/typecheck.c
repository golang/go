// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * type check the whole tree of an expression.
 * calculates expression types.
 * evaluates compile time constants.
 * marks variables that escape the local frame.
 * rewrites n->op to be more specific in some cases.
 * sets n->walk to walking function.
 *
 * TODO:
 *	trailing ... section of function calls
 */

#include "go.h"

static void	implicitstar(Node**);
static int	onearg(Node*);
static int	lookdot(Node*, Type*, int);
static void	typecheckaste(int, Type*, NodeList*, char*);
static int	exportassignok(Type*, char*);
static Type*	lookdot1(Sym *s, Type *t, Type *f, int);
static int	nokeys(NodeList*);
static void	typecheckcomplit(Node**);
static void	addrescapes(Node*);
static void	typecheckas2(Node*);
static void	typecheckas(Node*);
static void	typecheckfunc(Node*);
static void	checklvalue(Node*, char*);
static void	checkassign(Node*);
static void	checkassignlist(NodeList*);
static void	toslice(Node**);

void
typechecklist(NodeList *l, int top)
{
	for(; l; l=l->next)
		typecheck(&l->n, top);
}

/*
 * type check node *np.
 * replaces *np with a new pointer in some cases.
 * returns the final value of *np as a convenience.
 */
Node*
typecheck(Node **np, int top)
{
	int et, op;
	Node *n, *l, *r;
	NodeList *args;
	int lno, ok, ntop;
	Type *t;
	Sym *sym;
	Val v;

	// cannot type check until all the source has been parsed
	if(!typecheckok)
		fatal("early typecheck");

	n = *np;
	if(n == N)
		return N;

	// Skip typecheck if already done.
	// But re-typecheck ONAME/OTYPE/OLITERAL/OPACK node in case context has changed.
	if(n->typecheck == 1) {
		switch(n->op) {
		case ONAME:
		case OTYPE:
		case OLITERAL:
		case OPACK:
			break;
		default:
			return n;
		}
	}

	if(n->typecheck == 2)
		fatal("typecheck loop");
	n->typecheck = 2;

redo:
	lno = setlineno(n);
	if(n->sym) {
		walkdef(n);
		if(n->op == ONONAME)
			goto error;
	}

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
		if(n->type == T && n->val.ctype == CTSTR)
			n->type = idealstring;
		goto ret;

	case ONONAME:
		ok |= Erv;
		goto ret;

	case ONAME:
		if(n->etype != 0) {
			ok |= Ecall;
			goto ret;
		}
		if(!(top & Easgn)) {
			// not a write to the variable
			if(isblank(n)) {
				yyerror("cannot use _ as value");
				goto error;
			}
			n->used = 1;
		}
		ok |= Erv;
		goto ret;

	case OPACK:
		yyerror("use of package %S not in selector", n->sym);
		goto error;

	case OIOTA:
		// looked like iota during parsing but might
		// have been redefined.  decide.
		if(n->left->op != ONONAME)
			n = n->left;
		else
			n = n->right;
		goto redo;

	case ODDD:
		break;

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
			t->bound = -1;	// slice
		} else if(l->op == ODDD) {
			t->bound = -100;	// to be filled in
		} else {
			l = typecheck(&n->left, Erv);
			switch(consttype(l)) {
			case CTINT:
				v = l->val;
				break;
			case CTFLT:
				v = toint(l->val);
				break;
			default:
				yyerror("invalid array bound %#N", l);
				goto error;
			}
			t->bound = mpgetfix(v.u.xval);
			if(t->bound < 0) {
				yyerror("array bound must be non-negative");
				goto error;
			} else
				overflow(v, types[TINT]);
		}
		typecheck(&r, Etype);
		if(r->type == T)
			goto error;
		t->type = r->type;
		n->op = OTYPE;
		n->type = t;
		n->left = N;
		n->right = N;
		if(t->bound != -100)
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
		ntop = Erv | Etype;
		if(!(top & Eaddr))
			ntop |= Eindir;
		l = typecheck(&n->left, ntop);
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
			yyerror("invalid indirect of %+N", n->left);
			goto error;
		}
		ok |= Erv;
		n->type = t->type;
		goto ret;

	/*
	 * arithmetic exprs
	 */
	case OASOP:
		ok |= Etop;
		l = typecheck(&n->left, Erv);
		checkassign(n->left);
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
		l = typecheck(&n->left, Erv | (top & Eiota));
		r = typecheck(&n->right, Erv | (top & Eiota));
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
		if(l->type == T || r->type == T)
			goto error;
		t = l->type;
		if(t->etype == TIDEAL)
			t = r->type;
		et = t->etype;
		if(et == TIDEAL)
			et = TINT;
		if(t->etype != TIDEAL && !eqtype(l->type, r->type)) {
		badbinary:
			defaultlit2(&l, &r, 1);
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
		if(iscmp[n->op]) {
			evconst(n);
			t = types[TBOOL];
			if(n->op != OLITERAL) {
				defaultlit2(&l, &r, 1);
				n->left = l;
				n->right = r;
			}
		}
		if(et == TSTRING) {
			if(iscmp[n->op]) {
				n->etype = n->op;
				n->op = OCMPSTR;
			} else if(n->op == OASOP)
				n->op = OAPPENDSTR;
			else if(n->op == OADD)
				n->op = OADDSTR;
		}
		if(et == TINTER) {
			if(l->op == OLITERAL && l->val.ctype == CTNIL) {
				// swap for back end
				n->left = r;
				n->right = l;
			} else if(r->op == OLITERAL && r->val.ctype == CTNIL) {
				// leave alone for back end
			} else {
				n->etype = n->op;
				n->op = OCMPIFACE;
			}
		}
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
		t = l->type;
		if(t != T && t->etype != TIDEAL && !isint[t->etype]) {
			yyerror("invalid operation: %#N (shift of type %T)", n, t);
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
		l = typecheck(&n->left, Erv | (top & Eiota));
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
		ok |= Erv;
		typecheck(&n->left, Erv | Eaddr);
		if(n->left->type == T)
			goto error;
		switch(n->left->op) {
		case OMAPLIT:
		case OSTRUCTLIT:
		case OARRAYLIT:
			break;
		default:
			checklvalue(n->left, "take the address of");
		}
		defaultlit(&n->left, T);
		l = n->left;
		if((t = l->type) == T)
			goto error;
		if(!(top & Eindir))
			addrescapes(n->left);
		n->type = ptrto(t);
		goto ret;

	case OCOMPLIT:
		ok |= Erv;
		typecheckcomplit(&n);
		if(n->type == T)
			goto error;
		goto ret;

	case OXDOT:
		n = adddot(n);
		n->op = ODOT;
		// fall through
	case ODOT:
		typecheck(&n->left, Erv|Etype);
		defaultlit(&n->left, T);
		l = n->left;
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
			checkwidth(t);
		}
		sym = n->right->sym;
		if(!lookdot(n, t, 0)) {
			if(lookdot(n, t, 1))
				yyerror("%#N undefined (cannot refer to unexported field %S)", n, n->right->sym);
			else
				yyerror("%#N undefined (type %T has no field %S)", n, t, n->right->sym);
			goto error;
		}
		if(l->op == OTYPE) {
			if(n->type->etype != TFUNC || n->type->thistuple != 1) {
				yyerror("type %T has no method %hS", n->left->type, sym);
				n->type = T;
				goto error;
			}
			if(t->etype == TINTER) {
				yyerror("method expression on interface not implemented");
				n->type = T;
				goto error;
			}
			n->op = ONAME;
			n->sym = methodsym(sym, l->type);
			n->type = methodfunc(n->type, 1);
			n->xoffset = 0;
			getinargx(n->type)->type->type = l->type;	// fix up receiver
			n->class = PFUNC;
			ok = Erv;
			goto ret;
		}
		switch(n->op) {
		case ODOTINTER:
		case ODOTMETH:
			ok |= Ecall;
			break;
		default:
			ok |= Erv;
			break;
		}
		goto ret;

	case ODOTTYPE:
		ok |= Erv;
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
		ok |= Erv;
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
			defaultlit(&n->right, types[TUINT]);
			if(n->right->type != T && !isint[n->right->type->etype])
				yyerror("non-integer array index %#N", n->right);
			n->type = t->type;
			break;

		case TMAP:
			n->etype = 0;
			defaultlit(&n->right, t->down);
			if(n->right->type != T && !eqtype(n->right->type, t->down))
				yyerror("invalid map index %#N - need type %T", n->right, t->down);
			n->type = t->type;
			n->op = OINDEXMAP;
			break;

		case TSTRING:
			defaultlit(&n->right, types[TUINT]);
			if(n->right->type != T && !isint[n->right->type->etype])
				yyerror("non-integer string index %#N", n->right);
			n->type = types[TUINT8];
			n->op = OINDEXSTR;
			break;
		}
		goto ret;

	case ORECV:
		ok |= Etop | Erv;
		typecheck(&n->left, Erv);
		defaultlit(&n->left, T);
		l = n->left;
		if((t = l->type) == T)
			goto error;
		if(t->etype != TCHAN) {
			yyerror("invalid operation: %#N (receive from non-chan type %T)", n, t);
			goto error;
		}
		if(!(t->chan & Crecv)) {
			yyerror("invalid operation: %#N (receive from send-only type %T)", n, t);
			goto error;
		}
		n->type = t->type;
		goto ret;

	case OSEND:
		ok |= Etop | Erv;
		l = typecheck(&n->left, Erv);
		typecheck(&n->right, Erv);
		defaultlit(&n->left, T);
		l = n->left;
		if((t = l->type) == T)
			goto error;
		if(!(t->chan & Csend)) {
			yyerror("invalid operation: %#N (send to receive-only type %T)", n, t);
			goto error;
		}
		defaultlit(&n->right, t->type);
		r = n->right;
		if((t = r->type) == T)
			goto error;
		// TODO: more aggressive
		n->etype = 0;
		n->type = T;
		if(top & Erv) {
			n->op = OSENDNB;
			n->type = types[TBOOL];
		}
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
		if(n->right->left == N) {
			yyerror("missing slice bounds?");
			goto error;
		}
		if((t = n->right->left->type) == T)
			goto error;
		if(!isint[t->etype]) {
			yyerror("invalid slice index %#N (type %T)", n->right->left, t);
			goto error;
		}
		if(n->right->right != N) {
			if((t = n->right->right->type) == T)
				goto error;
			if(!isint[t->etype]) {
				yyerror("invalid slice index %#N (type %T)", n->right->right, t);
				goto error;
			}
		}
		l = n->left;
		if((t = l->type) == T)
			goto error;
		// TODO(rsc): 64-bit slice index needs to be checked
		// for overflow in generated code
		if(istype(t, TSTRING)) {
			n->type = t;
			n->op = OSLICESTR;
			goto ret;
		}
		if(isfixedarray(t)) {
			n->type = typ(TARRAY);
			n->type->type = t->type;
			n->type->bound = -1;
			dowidth(n->type);
			n->op = OSLICEARR;
			goto ret;
		}
		if(isslice(t)) {
			n->type = t;
			goto ret;
		}
		yyerror("cannot slice %#N (type %T)", l, t);
		goto error;

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
		if(l->op == ONAME && (r = unsafenmagic(l, n->list)) != N) {
			n = r;
			goto reswitch;
		}
		typecheck(&n->left, Erv | Etype | Ecall);
		defaultlit(&n->left, T);
		l = n->left;
		if(l->op == OTYPE) {
			// pick off before type-checking arguments
			ok |= Erv;
			// turn CALL(type, arg) into CONV(arg) w/ type
			n->left = N;
			if(onearg(n) < 0)
				goto error;
			n->op = OCONV;
			n->type = l->type;
			goto doconv;
		}

		if(count(n->list) == 1)
			typecheck(&n->list->n, Erv | Efnstruct);
		else
			typechecklist(n->list, Erv);
		if((t = l->type) == T)
			goto error;
		checkwidth(t);

		switch(l->op) {
		case ODOTINTER:
			n->op = OCALLINTER;
			break;

		case ODOTMETH:
			n->op = OCALLMETH;
			typecheckaste(OCALL, getthisx(t), list1(l->left), "method receiver");
			break;

		default:
			n->op = OCALLFUNC;
			if(t->etype != TFUNC) {
				yyerror("cannot call non-function %#N (type %T)", l, t);
				goto error;
			}
			break;
		}
		typecheckaste(OCALL, getinargx(t), n->list, "function argument");
		ok |= Etop;
		if(t->outtuple == 0)
			goto ret;
		ok |= Erv;
		if(t->outtuple == 1) {
			t = getoutargx(l->type)->type;
			if(t == T)
				goto error;
			if(t->etype == TFIELD)
				t = t->type;
			n->type = t;
			goto ret;
		}
		// multiple return
		if(!(top & (Efnstruct | Etop))) {
			yyerror("multiple-value %#N() in single-value context", l);
			goto ret;
		}
		n->type = getoutargx(l->type);
		goto ret;

	case OCAP:
	case OLEN:
		ok |= Erv;
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
	case OCLOSE:
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
		if(n->op == OCLOSED) {
			n->type = types[TBOOL];
			ok |= Erv;
		} else
			ok |= Etop;
		goto ret;

	case OCOPY:
		ok |= Etop|Erv;
		args = n->list;
		if(args == nil || args->next == nil) {
			yyerror("missing arguments to copy");
			goto error;
		}
		if(args->next->next != nil) {
			yyerror("too many arguments to copy");
			goto error;
		}
		n->left = args->n;
		n->right = args->next->n;
		n->type = types[TINT];
		typecheck(&n->left, Erv);
		typecheck(&n->right, Erv);
		if(n->left->type == T || n->right->type == T)
			goto error;
		toslice(&n->left);
		toslice(&n->right);
		defaultlit(&n->left, T);
		defaultlit(&n->right, T);
		if(!isslice(n->left->type) || !isslice(n->right->type)) {
			if(!isslice(n->left->type) && !isslice(n->right->type))
				yyerror("arguments to copy must be array pointer or slice; have %lT, %lT", n->left->type, n->right->type);
			else if(!isslice(n->left->type))
				yyerror("first argument to copy should be array pointer or slice; have %lT", n->left->type);
			else
				yyerror("second argument to copy should be array pointer or slice; have %lT", n->right->type);
			goto error;
		}
		if(!eqtype(n->left->type, n->right->type)) {
			yyerror("arguments to copy have different element types %lT and %lT", n->left->type, n->right->type);
			goto error;
		}
		goto ret;

	case OCONV:
	doconv:
		ok |= Erv;
		typecheck(&n->left, Erv | (top & Eindir));
		convlit1(&n->left, n->type, 1);
		if((t = n->left->type) == T || n->type == T)
			goto error;
		n = typecheckconv(n, n->left, n->type, 1, "conversion");
		if(n->type == T)
			goto error;
		goto ret;

	case OMAKE:
		ok |= Erv;
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
			if(r == N)
				r = nodintconst(0);
			n->left = l;
			n->right = r;
			n->op = OMAKESLICE;
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
			} else
				n->left = nodintconst(0);
			n->op = OMAKEMAP;
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
			} else
				n->left = nodintconst(0);
			n->op = OMAKECHAN;
			break;
		}
		if(args != nil) {
			yyerror("too many arguments to make(%T)", t);
			n->op = OMAKE;
			goto error;
		}
		n->type = t;
		goto ret;

	case ONEW:
		ok |= Erv;
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
		ok |= Etop;
		typechecklist(n->list, Erv);
		goto ret;

	case OCLOSURE:
		ok |= Erv;
		typecheckclosure(n);
		if(n->type == T)
			goto error;
		goto ret;

	/*
	 * statements
	 */
	case OAS:
		ok |= Etop;
		typecheckas(n);
		goto ret;

	case OAS2:
		ok |= Etop;
		typecheckas2(n);
		goto ret;

	case OBREAK:
	case OCONTINUE:
	case ODCL:
	case OEMPTY:
	case OGOTO:
	case OLABEL:
	case OXFALL:
		ok |= Etop;
		goto ret;

	case ODEFER:
	case OPROC:
		ok |= Etop;
		typecheck(&n->left, Etop);
		goto ret;

	case OFOR:
		ok |= Etop;
		typechecklist(n->ninit, Etop);
		typecheck(&n->ntest, Erv);
		if(n->ntest != N && (t = n->ntest->type) != T && t->etype != TBOOL)
			yyerror("non-bool %+N used as for condition", n->ntest);
		typecheck(&n->nincr, Etop);
		typechecklist(n->nbody, Etop);
		goto ret;

	case OIF:
		ok |= Etop;
		typechecklist(n->ninit, Etop);
		typecheck(&n->ntest, Erv);
		if(n->ntest != N && (t = n->ntest->type) != T && t->etype != TBOOL)
			yyerror("non-bool %+N used as if condition", n->ntest);
		typechecklist(n->nbody, Etop);
		typechecklist(n->nelse, Etop);
		goto ret;

	case ORETURN:
		ok |= Etop;
		typechecklist(n->list, Erv | Efnstruct);
		if(curfn->type->outnamed && n->list == nil)
			goto ret;
		typecheckaste(ORETURN, getoutargx(curfn->type), n->list, "return argument");
		goto ret;

	case OSELECT:
		ok |= Etop;
		typecheckselect(n);
		goto ret;

	case OSWITCH:
		ok |= Etop;
		typecheckswitch(n);
		goto ret;

	case ORANGE:
		ok |= Etop;
		typecheckrange(n);
		goto ret;

	case OTYPECASE:
		ok |= Etop | Erv;
		typecheck(&n->left, Erv);
		goto ret;

	case OTYPESW:
		yyerror("use of .(type) outside type switch");
		goto error;

	case OXCASE:
		ok |= Etop;
		typechecklist(n->list, Erv);
		typechecklist(n->nbody, Etop);
		goto ret;

	case ODCLFUNC:
		ok |= Etop;
		typecheckfunc(n);
		goto ret;

	case ODCLCONST:
		ok |= Etop;
		typecheck(&n->left, Erv);
		goto ret;

	case ODCLTYPE:
		ok |= Etop;
		typecheck(&n->left, Etype);
		if(!incannedimport)
			checkwidth(n->left->type);
		goto ret;
	}

ret:
	t = n->type;
	if(t && !t->funarg && n->op != OTYPE) {
		switch(t->etype) {
		case TFUNC:	// might have TANY; wait until its called
		case TANY:
		case TFORW:
		case TIDEAL:
		case TNIL:
		case TBLANK:
			break;
		case TARRAY:
			if(t->bound == -100) {
				yyerror("use of [...] array outside of array literal");
				t->bound = 1;
			}
		default:
			checkwidth(t);
		}
	}

	evconst(n);
	if(n->op == OTYPE && !(top & Etype)) {
		yyerror("type %T is not an expression", n->type);
		goto error;
	}
	if((top & (Erv|Etype)) == Etype && n->op != OTYPE) {
		yyerror("%#N is not a type", n);
		goto error;
	}
	if((ok & Ecall) && !(top & Ecall)) {
		yyerror("must call %#N", n);
		goto error;
	}
	// TODO(rsc): simplify
	if((top & (Ecall|Erv|Etype)) && !(top & Etop) && !(ok & (Erv|Etype|Ecall))) {
		yyerror("%#N used as value", n);
		goto error;
	}
	if((top & Etop) && !(top & (Ecall|Erv|Etype)) && !(ok & Etop)) {
		yyerror("%#N not used", n);
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

static void
toslice(Node **nn)
{
	Node *n;
	Type *t;

	n = *nn;
	if(n->type == T)
		return;
	if(isptr[n->type->etype] && isfixedarray(n->type->type)) {
		// convert to slice
		t = typ(TARRAY);
		t->bound = -1;
		t->type = n->type->type->type;
		n = typecheckconv(nil, n, t, 0, "conversion of array pointer to slice");
		*nn = n;
	}
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
lookdot1(Sym *s, Type *t, Type *f, int dostrcmp)
{
	Type *r;

	r = T;
	for(; f!=T; f=f->down) {
		if(dostrcmp && strcmp(f->sym->name, s->name) == 0)
			return f;
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
lookdot(Node *n, Type *t, int dostrcmp)
{
	Type *f1, *f2, *tt, *rcvr;
	Sym *s;

	s = n->right->sym;

	dowidth(t);
	f1 = T;
	if(t->etype == TSTRUCT || t->etype == TINTER)
		f1 = lookdot1(s, t, t->type, dostrcmp);

	f2 = methtype(n->left->type);
	if(f2 != T)
		f2 = lookdot1(s, f2, f2->method, dostrcmp);

	if(f1 != T) {
		if(f2 != T)
			yyerror("ambiguous DOT reference %S as both field and method",
				n->right->sym);
		n->xoffset = f1->width;
		n->type = f1->type;
		if(t->etype == TINTER) {
			if(isptr[n->left->type->etype]) {
				n->left = nod(OIND, n->left, N);	// implicitstar
				typecheck(&n->left, Erv);
			}
			n->op = ODOTINTER;
		}
		return 1;
	}

	if(f2 != T) {
		tt = n->left->type;
		dowidth(tt);
		rcvr = getthisx(f2->type)->type->type;
		if(n->left->op != OTYPE && !eqtype(rcvr, tt)) {
			if(rcvr->etype == tptr && eqtype(rcvr->type, tt)) {
				checklvalue(n->left, "call pointer method on");
				addrescapes(n->left);
				n->left = nod(OADDR, n->left, N);
				typecheck(&n->left, Etype|Erv);
			} else if(tt->etype == tptr && eqtype(tt->type, rcvr)) {
				n->left = nod(OIND, n->left, N);
				typecheck(&n->left, Etype|Erv);
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

static int
nokeys(NodeList *l)
{
	for(; l; l=l->next)
		if(l->n->op == OKEY)
			return 0;
	return 1;
}

/*
 * check implicit or explicit conversion from node type nt to type t.
 */
int
checkconv(Type *nt, Type *t, int explicit, int *op, int *et, char *desc)
{
	*op = OCONV;
	*et = 0;

	// preexisting error
	if(t == T || t->etype == TFORW)
		return 0;

	/*
	 * implicit conversions
	 */
	if(nt == T)
		return 0;

	if(t->etype == TBLANK) {
		*op = OCONVNOP;
		return 0;
	}

	if(eqtype(t, nt)) {
		exportassignok(t, desc);
		*op = OCONVNOP;
		if(!explicit || t == nt)
			return 0;
		return 1;
	}

	// interfaces are not subject to the name restrictions below.
	// accept anything involving interfaces and let ifacecvt
	// generate a good message.  some messages have to be
	// delayed anyway.
	// TODO(rsc): now that everything is delayed for whole-package
	// compilation, the messages could be generated right here.
	if(isnilinter(t) || isnilinter(nt) || isinter(t) || isinter(nt)) {
		*et = ifaceas1(t, nt, 0);
		*op = OCONVIFACE;
		return 1;
	}

	// otherwise, if concrete types have names, they must match.
	if(!explicit && t->sym && nt->sym && t != nt)
		return -1;

	// channel must not lose directionality
	if(t->etype == TCHAN && nt->etype == TCHAN) {
		if(t->chan & ~nt->chan)
			return -1;
		if(eqtype(t->type, nt->type)) {
			*op = OCONVNOP;
			return 1;
		}
	}

	// array to slice
	if(isslice(t) && isptr[nt->etype] && isfixedarray(nt->type)
	&& eqtype(t->type, nt->type->type)) {
		*op = OCONVSLICE;
		return 1;
	}

	/*
	 * explicit conversions
	 */
	if(!explicit)
		return -1;

	// same representation
	if(cvttype(t, nt)) {
		*op = OCONVNOP;
		return 1;
	}

	// simple fix-float
	if(isint[t->etype] || isfloat[t->etype])
	if(isint[nt->etype] || isfloat[nt->etype])
		return 1;

	// to string
	if(istype(t, TSTRING)) {
		// integer rune
		if(isint[nt->etype]) {
			*op = ORUNESTR;
			return 1;
		}

		// *[10]byte -> string
		// in preparation for next step
		if(isptr[nt->etype] && isfixedarray(nt->type)) {
			switch(nt->type->type->etype) {
			case TUINT8:
				*op = OARRAYBYTESTR;
				return 1;
			case TINT:
				*op = OARRAYRUNESTR;
				return 1;
			}
		}

		// []byte -> string
		if(isslice(nt)) {
			switch(nt->type->etype) {
			case TUINT8:
				*op = OARRAYBYTESTR;
				return 1;
			case TINT:
				*op = OARRAYRUNESTR;
				return 1;
			}
		}
	}

	// convert to unsafe pointer
	if(isptrto(t, TANY)
	&& (isptr[nt->etype] || nt->etype == TUINTPTR))
		return 1;

	// convert from unsafe pointer
	if(isptrto(nt, TANY)
	&& (isptr[t->etype] || t->etype == TUINTPTR))
		return 1;

	return -1;
}

Node*
typecheckconv(Node *nconv, Node *n, Type *t, int explicit, char *desc)
{
	int et, op;
	Node *n1;
	char *prefix;

	convlit1(&n, t, explicit);
	if(n->type == T)
		return n;


	if(n->op == OLITERAL)
	if(explicit || isideal(n->type))
	if(cvttype(t, n->type)) {
		// can convert literal in place
		// TODO(rsc) is this needed?
		n1 = nod(OXXX, N, N);
		*n1 = *n;
		n1->type = t;
		return n1;
	}

	prefix = "";
	if(desc != nil)
		prefix = " in ";
	else
		desc = "";
	switch(checkconv(n->type, t, explicit, &op, &et, desc)) {
	case -1:
		if(explicit)
			yyerror("cannot convert %+N to type %T%s%s", n, t, prefix, desc);
		else
			yyerror("cannot use %+N as type %T%s%s", n, t, prefix, desc);
		return n;

	case 0:
		if(nconv) {
			nconv->op = OCONVNOP;
			return nconv;
		}
		return n;
	}

	if(op == OCONVIFACE)
		defaultlit(&n, T);

	if(nconv == N)
		nconv = nod(OCONV, n, N);
	nconv->op = op;
	nconv->etype = et;
	nconv->type = t;
	nconv->typecheck = 1;
	return nconv;
}

/*
 * typecheck assignment: type list = expression list
 */
static void
typecheckaste(int op, Type *tstruct, NodeList *nl, char *desc)
{
	Type *t, *tl, *tn;
	Node *n;
	int lno;

	lno = lineno;

	if(tstruct->broke)
		goto out;

	if(nl != nil && nl->next == nil && (n = nl->n)->type != T)
	if(n->type->etype == TSTRUCT && n->type->funarg) {
		setlineno(n);
		tn = n->type->type;
		for(tl=tstruct->type; tl; tl=tl->down) {
			int xx, yy;
			if(tl->isddd) {
				// TODO(rsc): delete if (but not body) in DDD cleanup.
				if(tl->type->etype != TINTER)
					for(; tn; tn=tn->down)
						if(checkconv(tn->type, tl->type->type, 0, &xx, &yy, desc) < 0)
							yyerror("cannot use %T as type %T in %s", tn->type, tl->type->type, desc);
				goto out;
			}
			if(tn == T) {
				yyerror("not enough arguments to %#O", op);
				goto out;
			}
			if(checkconv(tn->type, tl->type, 0, &xx, &yy, desc) < 0)
				yyerror("cannot use type %T as type %T in %s", tn->type, tl->type, desc);
			tn = tn->down;
		}
		if(tn != T)
			yyerror("too many arguments to %#O", op);
		goto out;
	}

	for(tl=tstruct->type; tl; tl=tl->down) {
		t = tl->type;
		if(tl->isddd) {
			if(nl != nil && nl->n->isddd && !eqtype(nl->n->type, t)) {
				// TODO(rsc): This is not actually illegal but will
				// help catch bugs.
				yyerror("cannot pass %+N as %T (... mismatch)", nl->n, tl);
			}
			if(nl != nil && nl->next == nil && nl->n->isddd && eqtype(nl->n->type, t))
				goto out;
			for(; nl; nl=nl->next) {
				int xx, yy;
				setlineno(nl->n);
				defaultlit(&nl->n, t->type);
				// TODO(rsc): drop first if in DDD cleanup
				if(t->etype != TINTER)
				if(checkconv(nl->n->type, t->type, 0, &xx, &yy, desc) < 0)
					yyerror("cannot use %+N as type %T in %s", nl->n, t->type, desc);					
			}
			goto out;
		}
		if(nl == nil) {
			yyerror("not enough arguments to %#O", op);
			goto out;
		}
		n = nl->n;
		setlineno(nl->n);
		if(n->type != T)
			nl->n = typecheckconv(nil, n, t, 0, desc);
		nl = nl->next;
	}
	if(nl != nil) {
		yyerror("too many arguments to %#O", op);
		goto out;
	}

out:
	lineno = lno;
}

/*
 * do the export rules allow writing to this type?
 * cannot be implicitly assigning to any type with
 * an unavailable field.
 */
static int
exportassignok(Type *t, char *desc)
{
	Type *f;
	Sym *s;

	if(t == T)
		return 1;
	if(t->trecur)
		return 1;
	t->trecur = 1;

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
			if(s != nil && !exportname(s->name) && s->pkg != localpkg) {
				char *prefix;
				
				prefix = "";
				if(desc != nil)
					prefix = " in ";
				else
					desc = "";
				yyerror("implicit assignment of %T field '%s'%s%s", t, s->name, prefix, desc);
				goto no;
			}
			if(!exportassignok(f->type, desc))
				goto no;
		}
		break;

	case TARRAY:
		if(t->bound < 0)	// slices are pointers; that's fine
			break;
		if(!exportassignok(t->type, desc))
			goto no;
		break;
	}
	t->trecur = 0;
	return 1;

no:
	t->trecur = 0;
	return 0;
}


/*
 * type check composite
 */

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

static void
typecheckcomplit(Node **np)
{
	int bad, i, len, nerr;
	Node *l, *n, *hash[101];
	NodeList *ll;
	Type *t, *f;
	Sym *s;

	n = *np;

	memset(hash, 0, sizeof hash);

	l = typecheck(&n->right /* sic */, Etype);
	if((t = l->type) == T)
		goto error;
	nerr = nerrors;
	switch(t->etype) {
	default:
		yyerror("invalid type for composite literal: %T", t);
		n->type = T;
		break;

	case TARRAY:
		len = 0;
		i = 0;
		for(ll=n->list; ll; ll=ll->next) {
			l = ll->n;
			if(l->op == OKEY) {
				typecheck(&l->left, Erv);
				evconst(l->left);
				i = nonnegconst(l->left);
				if(i < 0) {
					yyerror("array index must be non-negative integer constant");
					i = -(1<<30);	// stay negative for a while
				}
				typecheck(&l->right, Erv);
				defaultlit(&l->right, t->type);
				l->right = typecheckconv(nil, l->right, t->type, 0, "array index");
			} else {
				typecheck(&ll->n, Erv);
				defaultlit(&ll->n, t->type);
				ll->n = typecheckconv(nil, ll->n, t->type, 0, "array index");
				ll->n = nod(OKEY, nodintconst(i), ll->n);
				ll->n->left->type = types[TINT];
				ll->n->left->typecheck = 1;
			}
			if(i >= 0)
				indexdup(ll->n->left, hash, nelem(hash));
			i++;
			if(i > len) {
				len = i;
				if(t->bound >= 0 && len > t->bound) {
					setlineno(l);
					yyerror("array index %d out of bounds [0:%d]", len, t->bound);
					t->bound = -1;	// no more errors
				}
			}
		}
		if(t->bound == -100)
			t->bound = len;
		if(t->bound < 0)
			n->right = nodintconst(len);
		n->op = OARRAYLIT;
		break;

	case TMAP:
		for(ll=n->list; ll; ll=ll->next) {
			l = ll->n;
			if(l->op != OKEY) {
				typecheck(&ll->n, Erv);
				yyerror("missing key in map literal");
				continue;
			}
			typecheck(&l->left, Erv);
			typecheck(&l->right, Erv);
			defaultlit(&l->left, t->down);
			defaultlit(&l->right, t->type);
			l->left = typecheckconv(nil, l->left, t->down, 0, "map key");
			l->right = typecheckconv(nil, l->right, t->type, 0, "map value");
			keydup(l->left, hash, nelem(hash));
		}
		n->op = OMAPLIT;
		break;

	case TSTRUCT:
		bad = 0;
		if(n->list != nil && nokeys(n->list)) {
			// simple list of variables
			f = t->type;
			for(ll=n->list; ll; ll=ll->next) {
				typecheck(&ll->n, Erv);
				if(f == nil) {
					if(!bad++)
						yyerror("too many values in struct initializer");
					continue;
				}
				s = f->sym;
				if(s != nil && !exportname(s->name) && s->pkg != localpkg)
					yyerror("implicit assignment of %T field '%s' in struct literal", t, s->name);
				ll->n = typecheckconv(nil, ll->n, f->type, 0, "field value");
				ll->n = nod(OKEY, newname(f->sym), ll->n);
				ll->n->left->typecheck = 1;
				f = f->down;
			}
			if(f != nil)
				yyerror("too few values in struct initializer");
		} else {
			// keyed list
			for(ll=n->list; ll; ll=ll->next) {
				l = ll->n;
				if(l->op != OKEY) {
					if(!bad++)
						yyerror("mixture of field:value and value initializers");
					typecheck(&ll->n, Erv);
					continue;
				}
				s = l->left->sym;
				if(s == S) {
					yyerror("invalid field name %#N in struct initializer", l->left);
					typecheck(&l->right, Erv);
					continue;
				}
				l->left = newname(s);
				l->left->typecheck = 1;
				f = lookdot1(s, t, t->type, 0);
				typecheck(&l->right, Erv);
				if(f == nil) {
					yyerror("unknown %T field '%s' in struct literal", t, s->name);
					continue;
				}
				s = f->sym;
				fielddup(newname(s), hash, nelem(hash));
				l->right = typecheckconv(nil, l->right, f->type, 0, "field value");
			}
		}
		n->op = OSTRUCTLIT;
		break;
	}
	if(nerr != nerrors)
		goto error;
	n->type = t;

	*np = n;
	return;

error:
	n->type = T;
	*np = n;
}

/*
 * the address of n has been taken and might be used after
 * the current function returns.  mark any local vars
 * as needing to move to the heap.
 */
static void
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
			// the function type has already been checked
			// (we're in the function body)
			// so the param already has a valid xoffset.
			if(n->class == PPARAM) {
				// expression to refer to stack copy
				n->stackparam = nod(OPARAM, n, N);
				n->stackparam->type = n->type;
				n->stackparam->addable = 1;
				if(n->xoffset == BADWIDTH)
					fatal("addrescapes before param assignment");
				n->stackparam->xoffset = n->xoffset;
				n->xoffset = 0;
			}

			n->class |= PHEAP;
			n->addable = 0;
			n->ullman = 2;
			n->xoffset = 0;

			// create stack variable to hold pointer to heap
			n->heapaddr = nod(ONAME, N, N);
			n->heapaddr->type = ptrto(n->type);
			snprint(buf, sizeof buf, "&%S", n->sym);
			n->heapaddr->sym = lookup(buf);
			n->heapaddr->class = PHEAP-1;	// defer tempname to allocparams
			curfn->dcl = list(curfn->dcl, n->heapaddr);
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
 * lvalue etc
 */
int
islvalue(Node *n)
{
	switch(n->op) {
	case OINDEX:
	case OIND:
	case ODOTPTR:
		return 1;
	case ODOT:
		return islvalue(n->left);
	case ONAME:
		if(n->class == PFUNC)
			return 0;
		return 1;
	}
	return 0;
}

static void
checklvalue(Node *n, char *verb)
{
	if(!islvalue(n))
		yyerror("cannot %s %#N", verb, n);
}

static void
checkassign(Node *n)
{
	if(islvalue(n))
		return;
	if(n->op == OINDEXMAP) {
		n->etype = 1;
		return;
	}
	yyerror("cannot assign to %#N", n);
}

static void
checkassignlist(NodeList *l)
{
	for(; l; l=l->next)
		checkassign(l->n);
}

/*
 * type check assignment.
 * if this assignment is the definition of a var on the left side,
 * fill in the var's type.
 */

static void
typecheckas(Node *n)
{
	// delicate little dance.
	// the definition of n may refer to this assignment
	// as its definition, in which case it will call typecheckas.
	// in that case, do not call typecheck back, or it will cycle.
	// if the variable has a type (ntype) then typechecking
	// will not look at defn, so it is okay (and desirable,
	// so that the conversion below happens).
	if(n->left->defn != n || n->left->ntype)
		typecheck(&n->left, Erv | Easgn);

	checkassign(n->left);
	typecheck(&n->right, Erv);
	if(n->right && n->right->type != T) {
		if(n->left->type != T)
			n->right = typecheckconv(nil, n->right, n->left->type, 0, "assignment");
		else
			exportassignok(n->right->type, "assignment");
	}
	if(n->left->defn == n && n->left->ntype == N) {
		defaultlit(&n->right, T);
		n->left->type = n->right->type;
	}

	// second half of dance.
	// now that right is done, typecheck the left
	// just to get it over with.  see dance above.
	n->typecheck = 1;
	if(n->left->typecheck == 0)
		typecheck(&n->left, Erv | Easgn);
}

static void
typecheckas2(Node *n)
{
	int cl, cr, op, et;
	NodeList *ll, *lr;
	Node *l, *r;
	Iter s;
	Type *t;

	for(ll=n->list; ll; ll=ll->next) {
		// delicate little dance.
		if(ll->n->defn != n || ll->n->ntype)
			typecheck(&ll->n, Erv | Easgn);
	}
	cl = count(n->list);
	cr = count(n->rlist);
	checkassignlist(n->list);
	if(cl > 1 && cr == 1)
		typecheck(&n->rlist->n, Erv | Efnstruct);
	else
		typechecklist(n->rlist, Erv);

	if(cl == cr) {
		// easy
		for(ll=n->list, lr=n->rlist; ll; ll=ll->next, lr=lr->next) {
			if(ll->n->type != T && lr->n->type != T)
				lr->n = typecheckconv(nil, lr->n, ll->n->type, 0, nil);
			if(ll->n->defn == n && ll->n->ntype == N) {
				defaultlit(&lr->n, T);
				ll->n->type = lr->n->type;
			}
		}
		goto out;
	}


	l = n->list->n;
	r = n->rlist->n;

	// m[i] = x, ok
	if(cl == 1 && cr == 2 && l->op == OINDEXMAP) {
		if(l->type == T)
			goto out;
		n->op = OAS2MAPW;
		n->rlist->n = typecheckconv(nil, r, l->type, 0, nil);
		r = n->rlist->next->n;
		n->rlist->next->n = typecheckconv(nil, r, types[TBOOL], 0, nil);
		goto out;
	}

	// x,y,z = f()
	if(cr == 1) {
		if(r->type == T)
			goto out;
		switch(r->op) {
		case OCALLMETH:
		case OCALLINTER:
		case OCALLFUNC:
			if(r->type->etype != TSTRUCT || r->type->funarg == 0)
				break;
			cr = structcount(r->type);
			if(cr != cl)
				goto mismatch;
			n->op = OAS2FUNC;
			t = structfirst(&s, &r->type);
			for(ll=n->list; ll; ll=ll->next) {
				if(ll->n->type != T)
					if(checkconv(t->type, ll->n->type, 0, &op, &et, nil) < 0)
						yyerror("cannot assign type %T to %+N", t->type, ll->n);
				if(ll->n->defn == n && ll->n->ntype == N)
					ll->n->type = t->type;
				t = structnext(&s);
			}
			goto out;
		}
	}

	// x, ok = y
	if(cl == 2 && cr == 1) {
		if(r->type == T)
			goto out;
		switch(r->op) {
		case OINDEXMAP:
			n->op = OAS2MAPR;
			goto common;
		case ORECV:
			n->op = OAS2RECV;
			goto common;
		case ODOTTYPE:
			n->op = OAS2DOTTYPE;
		common:
			if(l->type != T && checkconv(r->type, l->type, 0, &op, &et, nil) < 0)
				yyerror("cannot assign %+N to %+N", r, l);
			if(l->defn == n)
				l->type = r->type;
			l = n->list->next->n;
			if(l->type != T && checkconv(types[TBOOL], l->type, 0, &op, &et, nil) < 0)
				yyerror("cannot assign bool value to %+N", l);
			if(l->defn == n && l->ntype == N)
				l->type = types[TBOOL];
			goto out;
		}
	}

mismatch:
	yyerror("assignment count mismatch: %d = %d", cl, cr);

out:
	// second half of dance
	n->typecheck = 1;
	for(ll=n->list; ll; ll=ll->next)
		if(ll->n->typecheck == 0)
			typecheck(&ll->n, Erv | Easgn);
}

/*
 * type check function definition
 */
static void
typecheckfunc(Node *n)
{
	Type *t, *rcvr;

//dump("nname", n->nname);
	typecheck(&n->nname, Erv | Easgn);
	if((t = n->nname->type) == T)
		return;
	n->type = t;

	rcvr = getthisx(t)->type;
	if(rcvr != nil && n->shortname != N && !isblank(n->shortname))
		addmethod(n->shortname->sym, t, 1);
}
