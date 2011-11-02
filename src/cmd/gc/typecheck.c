// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * type check the whole tree of an expression.
 * calculates expression types.
 * evaluates compile time constants.
 * marks variables that escape the local frame.
 * rewrites n->op to be more specific in some cases.
 */

#include <u.h>
#include <libc.h>
#include "go.h"

static void	implicitstar(Node**);
static int	onearg(Node*, char*, ...);
static int	twoarg(Node*);
static int	lookdot(Node*, Type*, int);
static int	looktypedot(Node*, Type*, int);
static void	typecheckaste(int, Node*, int, Type*, NodeList*, char*);
static Type*	lookdot1(Sym *s, Type *t, Type *f, int);
static int	nokeys(NodeList*);
static void	typecheckcomplit(Node**);
static void	typecheckas2(Node*);
static void	typecheckas(Node*);
static void	typecheckfunc(Node*);
static void	checklvalue(Node*, char*);
static void	checkassign(Node*);
static void	checkassignlist(NodeList*);
static void	stringtoarraylit(Node**);
static Node*	resolve(Node*);
static Type*	getforwtype(Node*);

static	NodeList*	typecheckdefstack;

/*
 * resolve ONONAME to definition, if any.
 */
static Node*
resolve(Node *n)
{
	Node *r;

	if(n != N && n->op == ONONAME && (r = n->sym->def) != N) {
		if(r->op != OIOTA)
			n = r;
		else if(n->iota >= 0)
			n = nodintconst(n->iota);
	}
	return n;
}

void
typechecklist(NodeList *l, int top)
{
	for(; l; l=l->next)
		typecheck(&l->n, top);
}

static char* _typekind[] = {
	[TINT]		= "int",
	[TUINT]		= "uint",
	[TINT8]		= "int8",
	[TUINT8]	= "uint8",
	[TINT16]	= "int16",
	[TUINT16]	= "uint16",
	[TINT32]	= "int32",
	[TUINT32]	= "uint32",
	[TINT64]	= "int64",
	[TUINT64]	= "uint64",
	[TUINTPTR]	= "uintptr",
	[TCOMPLEX64]	= "complex64",
	[TCOMPLEX128]	= "complex128",
	[TFLOAT32]	= "float32",
	[TFLOAT64]	= "float64",
	[TBOOL]		= "bool",
	[TSTRING]	= "string",
	[TPTR32]	= "pointer",
	[TPTR64]	= "pointer",
	[TSTRUCT]	= "struct",
	[TINTER]	= "interface",
	[TCHAN]		= "chan",
	[TMAP]		= "map",
	[TARRAY]	= "array",
	[TFUNC]		= "func",
	[TNIL]		= "nil",
	[TIDEAL]	= "ideal number",
};

static char*
typekind(int et)
{
	static char buf[50];
	char *s;
	
	if(0 <= et && et < nelem(_typekind) && (s=_typekind[et]) != nil)
		return s;
	snprint(buf, sizeof buf, "etype=%d", et);
	return buf;
}

/*
 * type check node *np.
 * replaces *np with a new pointer in some cases.
 * returns the final value of *np as a convenience.
 */
Node*
typecheck(Node **np, int top)
{
	int et, aop, op, ptr;
	Node *n, *l, *r;
	NodeList *args;
	int lno, ok, ntop;
	Type *t, *tp, *ft, *missing, *have;
	Sym *sym;
	Val v;
	char *why;

	// cannot type check until all the source has been parsed
	if(!typecheckok)
		fatal("early typecheck");

	n = *np;
	if(n == N)
		return N;
	
	lno = setlineno(n);

	// Skip over parens.
	while(n->op == OPAREN)
		n = n->left;

	// Resolve definition of name and value of iota lazily.
	n = resolve(n);

	*np = n;

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
			lineno = lno;
			return n;
		}
	}

	if(n->typecheck == 2) {
		yyerror("typechecking loop involving %N", n);
		lineno = lno;
		return n;
	}
	n->typecheck = 2;

	if(n->sym) {
		if(n->op == ONAME && n->etype != 0 && !(top & Ecall)) {
			yyerror("use of builtin %S not in function call", n->sym);
			goto error;
		}

		// a dance to handle forward-declared recursive pointer types.
		if(n->op == OTYPE && (ft = getforwtype(n->ntype)) != T)
			defertypecopy(n, ft);

		typecheckdef(n);
		n->realtype = n->type;
		if(n->op == ONONAME)
			goto error;
	}
	*np = n;

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

	case OTPAREN:
		ok |= Etype;
		l = typecheck(&n->left, Etype);
		if(l->type == T)
			goto error;
		n->op = OTYPE;
		n->type = l->type;
		n->left = N;
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
			if(!(top&Ecomplit))
				yyerror("use of [...] array outside of array literal");
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
				yyerror("invalid array bound %N", l);
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
		if(!(top & Eaddr))  		// The *x in &*x is not an indirect.
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
			yyerror("invalid indirect of %lN", n->left);
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
		if(iscmp[n->op] && t->etype != TIDEAL && !eqtype(l->type, r->type)) {
			// comparison is okay as long as one side is
			// assignable to the other.  convert so they have
			// the same type.  (the only conversion that isn't
			// a no-op is concrete == interface.)
			if(r->type->etype != TBLANK && (aop = assignop(l->type, r->type, nil)) != 0) {
				l = nod(aop, l, N);
				l->type = r->type;
				l->typecheck = 1;
				n->left = l;
				t = l->type;
			} else if(l->type->etype != TBLANK && (aop = assignop(r->type, l->type, nil)) != 0) {
				r = nod(aop, r, N);
				r->type = l->type;
				r->typecheck = 1;
				n->right = r;
				t = r->type;
			}
			et = t->etype;
		}
		if(t->etype != TIDEAL && !eqtype(l->type, r->type)) {
			defaultlit2(&l, &r, 1);
			yyerror("invalid operation: %N (mismatched types %T and %T)", n, l->type, r->type);
			goto error;
		}
		if(!okfor[op][et]) {
		notokfor:
			yyerror("invalid operation: %N (operator %O not defined on %s)", n, op, typekind(et));
			goto error;
		}
		// okfor allows any array == array;
		// restrict to slice == nil and nil == slice.
		if(l->type->etype == TARRAY && !isslice(l->type))
			goto notokfor;
		if(r->type->etype == TARRAY && !isslice(r->type))
			goto notokfor;
		if(isslice(l->type) && !isnil(l) && !isnil(r)) {
			yyerror("invalid operation: %N (slice can only be compared to nil)", n);
			goto error;
		}
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
			} else if(n->op == OADD)
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
			yyerror("invalid operation: %N (shift count type %T, must be unsigned integer)", n, r->type);
			goto error;
		}
		t = l->type;
		if(t != T && t->etype != TIDEAL && !isint[t->etype]) {
			yyerror("invalid operation: %N (shift of type %T)", n, t);
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
			yyerror("invalid operation: %O %T", n->op, t);
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
		for(l=n->left; l->op == ODOT; l=l->left)
			l->addrtaken = 1;
		l->addrtaken = 1;
		defaultlit(&n->left, T);
		l = n->left;
		if((t = l->type) == T)
			goto error;
		// top&Eindir means this is &x in *&x.  (or the arg to built-in print)
		// n->etype means code generator flagged it as non-escaping.
		if(debug['s'] && !(top & Eindir) && !n->etype)
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
		sym = n->right->sym;
		if(l->op == OTYPE) {
			if(!looktypedot(n, t, 0)) {
				if(looktypedot(n, t, 1))
					yyerror("%N undefined (cannot refer to unexported method %S)", n, n->right->sym);
				else
					yyerror("%N undefined (type %T has no method %S)", n, t, n->right->sym);
				goto error;
			}
			if(n->type->etype != TFUNC || n->type->thistuple != 1) {
				yyerror("type %T has no method %hS", n->left->type, sym);
				n->type = T;
				goto error;
			}
			n->op = ONAME;
			n->sym = methodsym(sym, l->type, 0);
			n->type = methodfunc(n->type, l->type);
			n->xoffset = 0;
			n->class = PFUNC;
			ok = Erv;
			goto ret;
		}
		tp = t;
		if(isptr[t->etype] && t->type->etype != TINTER) {
			t = t->type;
			if(t == T)
				goto error;
			n->op = ODOTPTR;
			checkwidth(t);
		}
		if(!lookdot(n, t, 0)) {
			if(lookdot(n, t, 1))
				yyerror("%N undefined (cannot refer to unexported field or method %S)", n, n->right->sym);
			else
				yyerror("%N undefined (type %T has no field or method %S)", n, tp, n->right->sym);
			goto error;
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
			yyerror("invalid type assertion: %N (non-interface type %T on left)", n, t);
			goto error;
		}
		if(n->right != N) {
			typecheck(&n->right, Etype);
			n->type = n->right->type;
			n->right = N;
			if(n->type == T)
				goto error;
		}
		if(n->type != T && n->type->etype != TINTER)
		if(!implements(n->type, t, &missing, &have, &ptr)) {
			if(have)
				yyerror("impossible type assertion: %lN cannot have dynamic type %T"
					" (wrong type for %S method)\n\thave %S%hT\n\twant %S%hT",
					l, n->type, missing->sym, have->sym, have->type,
					missing->sym, missing->type);
			else
				yyerror("impossible type assertion: %lN cannot have dynamic type %T"
					" (missing %S method)", l, n->type, missing->sym);
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
			yyerror("invalid operation: %N (index of type %T)", n, t);
			goto error;

		case TARRAY:
			defaultlit(&n->right, T);
			if(n->right->type != T && !isint[n->right->type->etype])
				yyerror("non-integer array index %N", n->right);
			n->type = t->type;
			break;

		case TMAP:
			n->etype = 0;
			defaultlit(&n->right, t->down);
			if(n->right->type != T)
				n->right = assignconv(n->right, t->down, "map index");
			n->type = t->type;
			n->op = OINDEXMAP;
			break;

		case TSTRING:
			defaultlit(&n->right, types[TUINT]);
			if(n->right->type != T && !isint[n->right->type->etype])
				yyerror("non-integer string index %N", n->right);
			n->type = types[TUINT8];
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
			yyerror("invalid operation: %N (receive from non-chan type %T)", n, t);
			goto error;
		}
		if(!(t->chan & Crecv)) {
			yyerror("invalid operation: %N (receive from send-only type %T)", n, t);
			goto error;
		}
		n->type = t->type;
		goto ret;

	case OSEND:
		if(top & Erv) {
			yyerror("send statement %N used as value; use select for non-blocking send", n);
			goto error;
		}
		ok |= Etop | Erv;
		l = typecheck(&n->left, Erv);
		typecheck(&n->right, Erv);
		defaultlit(&n->left, T);
		l = n->left;
		if((t = l->type) == T)
			goto error;
		if(t->etype != TCHAN) {
			yyerror("invalid operation: %N (send to non-chan type %T)", n, t);
			goto error;
		}
		if(!(t->chan & Csend)) {
			yyerror("invalid operation: %N (send to receive-only type %T)", n, t);
			goto error;
		}
		defaultlit(&n->right, t->type);
		r = n->right;
		if(r->type == T)
			goto error;
		r = assignconv(r, l->type->type, "send");
		// TODO: more aggressive
		n->etype = 0;
		n->type = T;
		goto ret;

	case OSLICE:
		ok |= Erv;
		typecheck(&n->left, top);
		typecheck(&n->right->left, Erv);
		typecheck(&n->right->right, Erv);
		defaultlit(&n->left, T);
		defaultlit(&n->right->left, T);
		defaultlit(&n->right->right, T);
		if(isfixedarray(n->left->type)) {
			n->left = nod(OADDR, n->left, N);
			n->left->implicit = 1;
			typecheck(&n->left, top);
		}
		if(n->right->left != N) {
			if((t = n->right->left->type) == T)
				goto error;
			if(!isint[t->etype]) {
				yyerror("invalid slice index %N (type %T)", n->right->left, t);
				goto error;
			}
		}
		if(n->right->right != N) {
			if((t = n->right->right->type) == T)
				goto error;
			if(!isint[t->etype]) {
				yyerror("invalid slice index %N (type %T)", n->right->right, t);
				goto error;
			}
		}
		l = n->left;
		if((t = l->type) == T)
			goto error;
		if(istype(t, TSTRING)) {
			n->type = t;
			n->op = OSLICESTR;
			goto ret;
		}
		if(isptr[t->etype] && isfixedarray(t->type)) {
			n->type = typ(TARRAY);
			n->type->type = t->type->type;
			n->type->bound = -1;
			dowidth(n->type);
			n->op = OSLICEARR;
			goto ret;
		}
		if(isslice(t)) {
			n->type = t;
			goto ret;
		}
		yyerror("cannot slice %N (type %T)", l, t);
		goto error;

	/*
	 * call and call like
	 */
	case OCALL:
		l = n->left;
		if(l->op == ONAME && (r = unsafenmagic(n)) != N) {
			if(n->isddd)
				yyerror("invalid use of ... with builtin %N", l);
			n = r;
			goto reswitch;
		}
		typecheck(&n->left, Erv | Etype | Ecall |(top&Eproc));
		l = n->left;
		if(l->op == ONAME && l->etype != 0) {
			if(n->isddd && l->etype != OAPPEND)
				yyerror("invalid use of ... with builtin %N", l);
			// builtin: OLEN, OCAP, etc.
			n->op = l->etype;
			n->left = n->right;
			n->right = N;
			goto reswitch;
		}
		defaultlit(&n->left, T);
		l = n->left;
		if(l->op == OTYPE) {
			if(n->isddd || l->type->bound == -100)
				yyerror("invalid use of ... in type conversion", l);
			// pick off before type-checking arguments
			ok |= Erv;
			// turn CALL(type, arg) into CONV(arg) w/ type
			n->left = N;
			n->op = OCONV;
			n->type = l->type;
			if(onearg(n, "conversion to %T", l->type) < 0)
				goto error;
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
			// typecheckaste was used here but there wasn't enough
			// information further down the call chain to know if we
			// were testing a method receiver for unexported fields.
			// It isn't necessary, so just do a sanity check.
			tp = getthisx(t)->type->type;
			if(l->left == N || !eqtype(l->left->type, tp))
				fatal("method receiver");
			break;

		default:
			n->op = OCALLFUNC;
			if(t->etype != TFUNC) {
				yyerror("cannot call non-function %N (type %T)", l, t);
				goto error;
			}
			break;
		}
		typecheckaste(OCALL, n->left, n->isddd, getinargx(t), n->list, "function argument");
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
			yyerror("multiple-value %N() in single-value context", l);
			goto ret;
		}
		n->type = getoutargx(l->type);
		goto ret;

	case OCAP:
	case OLEN:
	case OREAL:
	case OIMAG:
		ok |= Erv;
		if(onearg(n, "%O", n->op) < 0)
			goto error;
		typecheck(&n->left, Erv);
		defaultlit(&n->left, T);
		implicitstar(&n->left);
		l = n->left;
		t = l->type;
		if(t == T)
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
		case OREAL:
		case OIMAG:
			if(!iscomplex[t->etype])
				goto badcall1;
			if(isconst(l, CTCPLX)){
				if(n->op == OREAL)
					n = nodfltconst(&l->val.u.cval->real);
				else
					n = nodfltconst(&l->val.u.cval->imag);
			}
			n->type = types[cplxsubtype(t->etype)];
			goto ret;
		}
		// might be constant
		switch(t->etype) {
		case TSTRING:
			if(isconst(l, CTSTR)) {
				r = nod(OXXX, N, N);
				nodconst(r, types[TINT], l->val.u.sval->len);
				r->orig = n;
				n = r;
			}
			break;
		case TARRAY:
			if(t->bound >= 0 && l->op == ONAME) {
				r = nod(OXXX, N, N);
				nodconst(r, types[TINT], t->bound);
				r->orig = n;
				n = r;
			}
			break;
		}
		n->type = types[TINT];
		goto ret;

	case OCOMPLEX:
		ok |= Erv;
		if(twoarg(n) < 0)
			goto error;
		l = typecheck(&n->left, Erv | (top & Eiota));
		r = typecheck(&n->right, Erv | (top & Eiota));
		if(l->type == T || r->type == T)
			goto error;
		defaultlit2(&l, &r, 0);
		n->left = l;
		n->right = r;
		if(l->type->etype != r->type->etype) {
		badcmplx:
			yyerror("invalid operation: %N (complex of types %T, %T)", n, l->type, r->type);
			goto error;
		}
		switch(l->type->etype) {
		default:
			goto badcmplx;
		case TIDEAL:
			t = types[TIDEAL];
			break;
		case TFLOAT32:
			t = types[TCOMPLEX64];
			break;
		case TFLOAT64:
			t = types[TCOMPLEX128];
			break;
		}
		if(l->op == OLITERAL && r->op == OLITERAL) {
			// make it a complex literal
			n = nodcplxlit(l->val, r->val);
		}
		n->type = t;
		goto ret;

	case OCLOSE:
		if(onearg(n, "%O", n->op) < 0)
			goto error;
		typecheck(&n->left, Erv);
		defaultlit(&n->left, T);
		l = n->left;
		if((t = l->type) == T)
			goto error;
		if(t->etype != TCHAN) {
			yyerror("invalid operation: %N (non-chan type %T)", n, t);
			goto error;
		}
		if(!(t->chan & Csend)) {
			yyerror("invalid operation: %#N (cannot close receive-only channel)", n);
			goto error;
		}
		ok |= Etop;
		goto ret;

	case ODELETE:
		args = n->list;
		if(args == nil) {
			yyerror("missing arguments to delete");
			goto error;
		}
		if(args->next == nil) {
			yyerror("missing second (key) argument to delete");
			goto error;
		}
		if(args->next->next != nil) {
			yyerror("too many arguments to delete");
			goto error;
		}
		ok |= Etop;
		typechecklist(args, Erv);
		l = args->n;
		r = args->next->n;
		if(l->type != T && l->type->etype != TMAP) {
			yyerror("first argument to delete must be map; have %lT", l->type);
			goto error;
		}
		args->next->n = assignconv(r, l->type->down, "delete");
		goto ret;

	case OAPPEND:
		ok |= Erv;
		args = n->list;
		if(args == nil) {
			yyerror("missing arguments to append");
			goto error;
		}
		typechecklist(args, Erv);
		if((t = args->n->type) == T)
			goto error;
		n->type = t;
		if(!isslice(t)) {
			yyerror("first argument to append must be slice; have %lT", t);
			goto error;
		}
		if(n->isddd) {
			if(args->next == nil) {
				yyerror("cannot use ... on first argument to append");
				goto error;
			}
			if(args->next->next != nil) {
				yyerror("too many arguments to append");
				goto error;
			}
			if(istype(t->type, TUINT8) && istype(args->next->n->type, TSTRING)) {
				defaultlit(&args->next->n, types[TSTRING]);
				goto ret;
			}
			args->next->n = assignconv(args->next->n, t->orig, "append");
			goto ret;
		}
		for(args=args->next; args != nil; args=args->next) {
			if(args->n->type == T)
				continue;
			args->n = assignconv(args->n, t->type, "append");
		}
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
		n->list = nil;
		n->type = types[TINT];
		typecheck(&n->left, Erv);
		typecheck(&n->right, Erv);
		if(n->left->type == T || n->right->type == T)
			goto error;
		defaultlit(&n->left, T);
		defaultlit(&n->right, T);

		// copy([]byte, string)
		if(isslice(n->left->type) && n->right->type->etype == TSTRING) {
			if(eqtype(n->left->type->type, bytetype))
				goto ret;
			yyerror("arguments to copy have different element types: %lT and string", n->left->type);
			goto error;
		}

		if(!isslice(n->left->type) || !isslice(n->right->type)) {
			if(!isslice(n->left->type) && !isslice(n->right->type))
				yyerror("arguments to copy must be slices; have %lT, %lT", n->left->type, n->right->type);
			else if(!isslice(n->left->type))
				yyerror("first argument to copy should be slice; have %lT", n->left->type);
			else
				yyerror("second argument to copy should be slice or string; have %lT", n->right->type);
			goto error;
		}
		if(!eqtype(n->left->type->type, n->right->type->type)) {
			yyerror("arguments to copy have different element types: %lT and %lT", n->left->type, n->right->type);
			goto error;
		}
		goto ret;

	case OCONV:
	doconv:
		ok |= Erv;
		typecheck(&n->left, Erv | (top & (Eindir | Eiota)));
		convlit1(&n->left, n->type, 1);
		if((t = n->left->type) == T || n->type == T)
			goto error;
		if((n->op = convertop(t, n->type, &why)) == 0) {
			yyerror("cannot convert %lN to type %T%s", n->left, n->type, why);
			n->op = OCONV;
		}
		switch(n->op) {
		case OCONVNOP:
			if(n->left->op == OLITERAL) {
				n->op = OLITERAL;
				n->val = n->left->val;
			}
			break;
		case OSTRARRAYBYTE:
		case OSTRARRAYRUNE:
			if(n->left->op == OLITERAL)
				stringtoarraylit(&n);
			break;
		}
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
			defaultlit(&l, types[TINT]);
			r = N;
			if(args != nil) {
				r = args->n;
				args = args->next;
				typecheck(&r, Erv);
				defaultlit(&r, types[TINT]);
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
			n->op = OMAKESLICE;
			break;

		case TMAP:
			if(args != nil) {
				l = args->n;
				args = args->next;
				typecheck(&l, Erv);
				defaultlit(&l, types[TINT]);
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
				defaultlit(&l, types[TINT]);
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

	case OPRINT:
	case OPRINTN:
		ok |= Etop;
		typechecklist(n->list, Erv | Eindir);  // Eindir: address does not escape
		goto ret;

	case OPANIC:
		ok |= Etop;
		if(onearg(n, "panic") < 0)
			goto error;
		typecheck(&n->left, Erv);
		defaultlit(&n->left, types[TINTER]);
		if(n->left->type == T)
			goto error;
		goto ret;
	
	case ORECOVER:
		ok |= Erv|Etop;
		if(n->list != nil) {
			yyerror("too many arguments to recover");
			goto error;
		}
		n->type = types[TINTER];
		goto ret;

	case OCLOSURE:
		ok |= Erv;
		typecheckclosure(n, top);
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
		ok |= Etop;
		typecheck(&n->left, Etop);
		goto ret;

	case OPROC:
		ok |= Etop;
		typecheck(&n->left, Etop|Eproc);
		goto ret;

	case OFOR:
		ok |= Etop;
		typechecklist(n->ninit, Etop);
		typecheck(&n->ntest, Erv);
		if(n->ntest != N && (t = n->ntest->type) != T && t->etype != TBOOL)
			yyerror("non-bool %lN used as for condition", n->ntest);
		typecheck(&n->nincr, Etop);
		typechecklist(n->nbody, Etop);
		goto ret;

	case OIF:
		ok |= Etop;
		typechecklist(n->ninit, Etop);
		typecheck(&n->ntest, Erv);
		if(n->ntest != N && (t = n->ntest->type) != T && t->etype != TBOOL)
			yyerror("non-bool %lN used as if condition", n->ntest);
		typechecklist(n->nbody, Etop);
		typechecklist(n->nelse, Etop);
		goto ret;

	case ORETURN:
		ok |= Etop;
		typechecklist(n->list, Erv | Efnstruct);
		if(curfn == N) {
			yyerror("return outside function");
			goto error;
		}
		if(curfn->type->outnamed && n->list == nil)
			goto ret;
		typecheckaste(ORETURN, nil, 0, getoutargx(curfn->type), n->list, "return argument");
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
		default:
			checkwidth(t);
		}
	}

	// TODO(rsc): should not need to check importpkg,
	// but reflect mentions unsafe.Pointer.
	if(safemode && !incannedimport && !importpkg && t && t->etype == TUNSAFEPTR)
		yyerror("cannot use unsafe.Pointer");

	evconst(n);
	if(n->op == OTYPE && !(top & Etype)) {
		yyerror("type %T is not an expression", n->type);
		goto error;
	}
	if((top & (Erv|Etype)) == Etype && n->op != OTYPE) {
		yyerror("%N is not a type", n);
		goto error;
	}
	if((ok & Ecall) && !(top & Ecall)) {
		yyerror("method %N is not an expression, must be called", n);
		goto error;
	}
	// TODO(rsc): simplify
	if((top & (Ecall|Erv|Etype)) && !(top & Etop) && !(ok & (Erv|Etype|Ecall))) {
		yyerror("%N used as value", n);
		goto error;
	}
	if((top & Etop) && !(top & (Ecall|Erv|Etype)) && !(ok & Etop)) {
		if(n->diag == 0) {
			yyerror("%N not used", n);
			n->diag = 1;
		}
		goto error;
	}

	/* TODO
	if(n->type == T)
		fatal("typecheck nil type");
	*/
	goto out;

badcall1:
	yyerror("invalid argument %lN for %O", n->left, n->op);
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

	// insert implicit * if needed for fixed array
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
onearg(Node *n, char *f, ...)
{
	va_list arg;
	char *p;

	if(n->left != N)
		return 0;
	if(n->list == nil) {
		va_start(arg, f);
		p = vsmprint(f, arg);
		va_end(arg);
		yyerror("missing argument to %s: %N", p, n);
		return -1;
	}
	if(n->list->next != nil) {
		va_start(arg, f);
		p = vsmprint(f, arg);
		va_end(arg);
		yyerror("too many arguments to %s: %N", p, n);
		n->left = n->list->n;
		n->list = nil;
		return -1;
	}
	n->left = n->list->n;
	n->list = nil;
	return 0;
}

static int
twoarg(Node *n)
{
	if(n->left != N)
		return 0;
	if(n->list == nil) {
		yyerror("missing argument to %O - %N", n->op, n);
		return -1;
	}
	n->left = n->list->n;
	if(n->list->next == nil) {
		yyerror("missing argument to %O - %N", n->op, n);
		n->list = nil;
		return -1;
	}
	if(n->list->next->next != nil) {
		yyerror("too many arguments to %O - %N", n->op, n);
		n->list = nil;
		return -1;
	}
	n->right = n->list->next->n;
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
looktypedot(Node *n, Type *t, int dostrcmp)
{
	Type *f1, *f2, *tt;
	Sym *s;
	
	s = n->right->sym;

	if(t->etype == TINTER) {
		f1 = lookdot1(s, t, t->type, dostrcmp);
		if(f1 == T)
			return 0;

		if(f1->width == BADWIDTH)
			fatal("lookdot badwidth %T %p", f1, f1);
		n->right = methodname(n->right, t);
		n->xoffset = f1->width;
		n->type = f1->type;
		n->op = ODOTINTER;
		return 1;
	}

	tt = t;
	if(t->sym == S && isptr[t->etype])
		tt = t->type;

	f2 = methtype(tt);
	if(f2 == T)
		return 0;

	expandmeth(f2->sym, f2);
	f2 = lookdot1(s, f2, f2->xmethod, dostrcmp);
	if(f2 == T)
		return 0;

	// disallow T.m if m requires *T receiver
	if(isptr[getthisx(f2->type)->type->type->etype]
	&& !isptr[t->etype]
	&& f2->embedded != 2
	&& !isifacemethod(f2->type)) {
		yyerror("invalid method expression %N (needs pointer receiver: (*%T).%hS)", n, t, f2->sym);
		return 0;
	}

	n->right = methodname(n->right, t);
	n->xoffset = f2->width;
	n->type = f2->type;
	n->op = ODOTMETH;
	return 1;
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

	f2 = T;
	if(n->left->type == t || n->left->type->sym == S) {
		f2 = methtype(t);
		if(f2 != T) {
			// Use f2->method, not f2->xmethod: adddot has
			// already inserted all the necessary embedded dots.
			f2 = lookdot1(s, f2, f2->method, dostrcmp);
		}
	}

	if(f1 != T) {
		if(f2 != T)
			yyerror("ambiguous DOT reference %S as both field and method",
				n->right->sym);
		if(f1->width == BADWIDTH)
			fatal("lookdot badwidth %T %p", f1, f1);
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
		if(!eqtype(rcvr, tt)) {
			if(rcvr->etype == tptr && eqtype(rcvr->type, tt)) {
				checklvalue(n->left, "call pointer method on");
				if(debug['s'])
					addrescapes(n->left);
				n->left = nod(OADDR, n->left, N);
				n->left->implicit = 1;
				typecheck(&n->left, Etype|Erv);
			} else if(tt->etype == tptr && eqtype(tt->type, rcvr)) {
				n->left = nod(OIND, n->left, N);
				n->left->implicit = 1;
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
 * typecheck assignment: type list = expression list
 */
static void
typecheckaste(int op, Node *call, int isddd, Type *tstruct, NodeList *nl, char *desc)
{
	Type *t, *tl, *tn;
	Node *n;
	int lno;
	char *why;

	lno = lineno;

	if(tstruct->broke)
		goto out;

	if(nl != nil && nl->next == nil && (n = nl->n)->type != T)
	if(n->type->etype == TSTRUCT && n->type->funarg) {
		tn = n->type->type;
		for(tl=tstruct->type; tl; tl=tl->down) {
			if(tl->isddd) {
				for(; tn; tn=tn->down) {
					exportassignok(tn->type, desc);
					if(assignop(tn->type, tl->type->type, &why) == 0) {
						if(call != N)
							yyerror("cannot use %T as type %T in argument to %N%s", tn->type, tl->type, call, why);
						else
							yyerror("cannot use %T as type %T in %s%s", tn->type, tl->type, desc, why);
					}
				}
				goto out;
			}
			if(tn == T)
				goto notenough;
			exportassignok(tn->type, desc);
			if(assignop(tn->type, tl->type, &why) == 0) {
				if(call != N)
					yyerror("cannot use %T as type %T in argument to %N%s", tn->type, tl->type, call, why);
				else
					yyerror("cannot use %T as type %T in %s%s", tn->type, tl->type, desc, why);
			}
			tn = tn->down;
		}
		if(tn != T)
			goto toomany;
		goto out;
	}

	for(tl=tstruct->type; tl; tl=tl->down) {
		t = tl->type;
		if(tl->isddd) {
			if(isddd) {
				if(nl == nil)
					goto notenough;
				if(nl->next != nil)
					goto toomany;
				n = nl->n;
				setlineno(n);
				if(n->type != T)
					nl->n = assignconv(n, t, desc);
				goto out;
			}
			for(; nl; nl=nl->next) {
				n = nl->n;
				setlineno(nl->n);
				if(n->type != T)
					nl->n = assignconv(n, t->type, desc);
			}
			goto out;
		}
		if(nl == nil)
			goto notenough;
		n = nl->n;
		setlineno(n);
		if(n->type != T)
			nl->n = assignconv(n, t, desc);
		nl = nl->next;
	}
	if(nl != nil)
		goto toomany;
	if(isddd) {
		if(call != N)
			yyerror("invalid use of ... in call to %N", call);
		else
			yyerror("invalid use of ... in %O", op);
	}

out:
	lineno = lno;
	return;

notenough:
	if(call != N)
		yyerror("not enough arguments in call to %N", call);
	else
		yyerror("not enough arguments to %O", op);
	goto out;

toomany:
	if(call != N)
		yyerror("too many arguments in call to %N", call);
	else
		yyerror("too many arguments to %O", op);
	goto out;
}

/*
 * do the export rules allow writing to this type?
 * cannot be implicitly assigning to any type with
 * an unavailable field.
 */
int
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
				yyerror("implicit assignment of unexported field '%s' of %T%s%s", s->name, t, prefix, desc);
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

static int
prime(ulong h, ulong sr)
{
	ulong n;

	for(n=3; n<=sr; n+=2)
		if(h%n == 0)
			return 0;
	return 1;
}

static ulong
inithash(Node *n, Node ***hash, Node **autohash, ulong nautohash)
{
	ulong h, sr;
	NodeList *ll;
	int i;

	// count the number of entries
	h = 0;
	for(ll=n->list; ll; ll=ll->next)
		h++;

	// if the auto hash table is
	// large enough use it.
	if(h <= nautohash) {
		*hash = autohash;
		memset(*hash, 0, nautohash * sizeof(**hash));
		return nautohash;
	}

	// make hash size odd and 12% larger than entries
	h += h/8;
	h |= 1;

	// calculate sqrt of h
	sr = h/2;
	for(i=0; i<5; i++)
		sr = (sr + h/sr)/2;

	// check for primeality
	while(!prime(h, sr))
		h += 2;

	// build and return a throw-away hash table
	*hash = mal(h * sizeof(**hash));
	memset(*hash, 0, h * sizeof(**hash));
	return h;
}

static void
typecheckcomplit(Node **np)
{
	int bad, i, len, nerr;
	Node *l, *n, **hash;
	NodeList *ll;
	Type *t, *f, *pushtype;
	Sym *s;
	int32 lno;
	ulong nhash;
	Node *autohash[101];

	n = *np;
	lno = lineno;

	if(n->right == N) {
		if(n->list != nil)
			setlineno(n->list->n);
		yyerror("missing type in composite literal");
		goto error;
	}

	setlineno(n->right);
	l = typecheck(&n->right /* sic */, Etype|Ecomplit);
	if((t = l->type) == T)
		goto error;
	nerr = nerrors;

	// can omit type on composite literal values if the outer
	// composite literal is array, slice, or map, and the 
	// element type is itself a struct, array, slice, or map.
	pushtype = T;
	if(t->etype == TARRAY || t->etype == TMAP) {
		pushtype = t->type;
		if(pushtype != T) {
			switch(pushtype->etype) {
			case TSTRUCT:
			case TARRAY:
			case TMAP:
				break;
			default:
				pushtype = T;
				break;
			}
		}
	}

	switch(t->etype) {
	default:
		yyerror("invalid type for composite literal: %T", t);
		n->type = T;
		break;

	case TARRAY:
		nhash = inithash(n, &hash, autohash, nelem(autohash));

		len = 0;
		i = 0;
		for(ll=n->list; ll; ll=ll->next) {
			l = ll->n;
			setlineno(l);
			if(l->op != OKEY) {
				l = nod(OKEY, nodintconst(i), l);
				l->left->type = types[TINT];
				l->left->typecheck = 1;
				ll->n = l;
			}

			typecheck(&l->left, Erv);
			evconst(l->left);
			i = nonnegconst(l->left);
			if(i < 0) {
				yyerror("array index must be non-negative integer constant");
				i = -(1<<30);	// stay negative for a while
			}
			if(i >= 0)
				indexdup(l->left, hash, nhash);
			i++;
			if(i > len) {
				len = i;
				if(t->bound >= 0 && len > t->bound) {
					setlineno(l);
					yyerror("array index %d out of bounds [0:%d]", len, t->bound);
					t->bound = -1;	// no more errors
				}
			}

			if(l->right->op == OCOMPLIT && l->right->right == N && pushtype != T)
				l->right->right = typenod(pushtype);
			typecheck(&l->right, Erv);
			defaultlit(&l->right, t->type);
			l->right = assignconv(l->right, t->type, "array element");
		}
		if(t->bound == -100)
			t->bound = len;
		if(t->bound < 0)
			n->right = nodintconst(len);
		n->op = OARRAYLIT;
		break;

	case TMAP:
		nhash = inithash(n, &hash, autohash, nelem(autohash));

		for(ll=n->list; ll; ll=ll->next) {
			l = ll->n;
			setlineno(l);
			if(l->op != OKEY) {
				typecheck(&ll->n, Erv);
				yyerror("missing key in map literal");
				continue;
			}

			typecheck(&l->left, Erv);
			defaultlit(&l->left, t->down);
			l->left = assignconv(l->left, t->down, "map key");
			keydup(l->left, hash, nhash);

			if(l->right->op == OCOMPLIT && l->right->right == N && pushtype != T)
				l->right->right = typenod(pushtype);
			typecheck(&l->right, Erv);
			defaultlit(&l->right, t->type);
			l->right = assignconv(l->right, t->type, "map value");
		}
		n->op = OMAPLIT;
		break;

	case TSTRUCT:
		bad = 0;
		if(n->list != nil && nokeys(n->list)) {
			// simple list of variables
			f = t->type;
			for(ll=n->list; ll; ll=ll->next) {
				setlineno(ll->n);
				typecheck(&ll->n, Erv);
				if(f == nil) {
					if(!bad++)
						yyerror("too many values in struct initializer");
					continue;
				}
				s = f->sym;
				if(s != nil && !exportname(s->name) && s->pkg != localpkg)
					yyerror("implicit assignment of unexported field '%s' in %T literal", s->name, t);
				ll->n = assignconv(ll->n, f->type, "field value");
				ll->n = nod(OKEY, newname(f->sym), ll->n);
				ll->n->left->type = f;
				ll->n->left->typecheck = 1;
				f = f->down;
			}
			if(f != nil)
				yyerror("too few values in struct initializer");
		} else {
			nhash = inithash(n, &hash, autohash, nelem(autohash));

			// keyed list
			for(ll=n->list; ll; ll=ll->next) {
				l = ll->n;
				setlineno(l);
				if(l->op != OKEY) {
					if(!bad++)
						yyerror("mixture of field:value and value initializers");
					typecheck(&ll->n, Erv);
					continue;
				}
				s = l->left->sym;
				if(s == S) {
					yyerror("invalid field name %N in struct initializer", l->left);
					typecheck(&l->right, Erv);
					continue;
				}
				// Sym might have resolved to name in other top-level
				// package, because of import dot.  Redirect to correct sym
				// before we do the lookup.
				if(s->pkg != localpkg)
					s = lookup(s->name);
				f = lookdot1(s, t, t->type, 0);
				typecheck(&l->right, Erv);
				if(f == nil) {
					yyerror("unknown %T field '%s' in struct literal", t, s->name);
					continue;
				}
				l->left = newname(s);
				l->left->typecheck = 1;
				l->left->type = f;
				s = f->sym;
				fielddup(newname(s), hash, nhash);
				l->right = assignconv(l->right, f->type, "field value");
			}
		}
		n->op = OSTRUCTLIT;
		break;
	}
	if(nerr != nerrors)
		goto error;
	n->type = t;

	*np = n;
	lineno = lno;
	return;

error:
	n->type = T;
	*np = n;
	lineno = lno;
}

/*
 * lvalue etc
 */
int
islvalue(Node *n)
{
	switch(n->op) {
	case OINDEX:
		if(isfixedarray(n->left->type))
			return islvalue(n->left);
		if(n->left->type != T && n->left->type->etype == TSTRING)
			return 0;
		// fall through
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
		yyerror("cannot %s %N", verb, n);
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
	yyerror("cannot assign to %N", n);
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
	n->left = resolve(n->left);
	if(n->left->defn != n || n->left->ntype)
		typecheck(&n->left, Erv | Easgn);

	checkassign(n->left);
	typecheck(&n->right, Erv);
	if(n->right && n->right->type != T) {
		if(n->left->type != T)
			n->right = assignconv(n->right, n->left->type, "assignment");
		else if(!isblank(n->left))
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
checkassignto(Type *src, Node *dst)
{
	char *why;

	if(assignop(src, dst->type, &why) == 0) {
		yyerror("cannot assign %T to %lN in multiple assignment%s", src, dst, why);
		return;
	}
	exportassignok(dst->type, "multiple assignment");
}

static void
typecheckas2(Node *n)
{
	int cl, cr;
	NodeList *ll, *lr;
	Node *l, *r, *rr;
	Iter s;
	Type *t;

	for(ll=n->list; ll; ll=ll->next) {
		// delicate little dance.
		ll->n = resolve(ll->n);
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
				lr->n = assignconv(lr->n, ll->n->type, "assignment");
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
		n->rlist->n = assignconv(r, l->type, "assignment");
		rr = n->rlist->next->n;
		n->rlist->next->n = assignconv(rr, types[TBOOL], "assignment");
		if(isconst(rr, CTBOOL) && !rr->val.u.bval) {
			n->op = ODELETE;
			n->list = list(list1(l->left), l->right);
			n->right = n->rlist->n;
			n->rlist = nil;
		}
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
					checkassignto(t->type, ll->n);
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
			n->right = n->rlist->n;
			goto common;
		case ODOTTYPE:
			n->op = OAS2DOTTYPE;
			r->op = ODOTTYPE2;
		common:
			if(l->type != T)
				checkassignto(r->type, l);
			if(l->defn == n)
				l->type = r->type;
			l = n->list->next->n;
			if(l->type != T)
				checkassignto(types[TBOOL], l);
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

	typecheck(&n->nname, Erv | Easgn);
	if((t = n->nname->type) == T)
		return;
	n->type = t;

	rcvr = getthisx(t)->type;
	if(rcvr != nil && n->shortname != N && !isblank(n->shortname))
		addmethod(n->shortname->sym, t, 1);
}

static void
stringtoarraylit(Node **np)
{
	int32 i;
	NodeList *l;
	Strlit *s;
	char *p, *ep;
	Rune r;
	Node *nn, *n;

	n = *np;
	if(n->left->op != OLITERAL || n->left->val.ctype != CTSTR)
		fatal("stringtoarraylit %N", n);

	s = n->left->val.u.sval;
	l = nil;
	p = s->s;
	ep = s->s + s->len;
	i = 0;
	if(n->type->type->etype == TUINT8) {
		// raw []byte
		while(p < ep)
			l = list(l, nod(OKEY, nodintconst(i++), nodintconst((uchar)*p++)));
	} else {
		// utf-8 []rune
		while(p < ep) {
			p += chartorune(&r, p);
			l = list(l, nod(OKEY, nodintconst(i++), nodintconst(r)));
		}
	}
	nn = nod(OCOMPLIT, N, typenod(n->type));
	nn->list = l;
	typecheck(&nn, Erv);
	*np = nn;
}

static Type*
getforwtype(Node *n)
{
	Node *f1, *f2;

	for(f2=n; ; n=n->ntype) {
		if((n = resolve(n)) == N || n->op != OTYPE)
			return T;

		if(n->type != T && n->type->etype == TFORW)
			return n->type;

		// Check for ntype cycle.
		if((f2 = resolve(f2)) != N && (f1 = resolve(f2->ntype)) != N) {
			f2 = resolve(f1->ntype);
			if(f1 == n || f2 == n)
				return T;
		}
	}
}

static int ntypecheckdeftype;
static NodeList *methodqueue;

static void
domethod(Node *n)
{
	Node *nt;

	nt = n->type->nname;
	typecheck(&nt, Etype);
	if(nt->type == T) {
		// type check failed; leave empty func
		n->type->etype = TFUNC;
		n->type->nod = N;
		return;
	}
	*n->type = *nt->type;
	n->type->nod = N;
	checkwidth(n->type);
}

typedef struct NodeTypeList NodeTypeList;
struct NodeTypeList {
	Node *n;
	Type *t;
	NodeTypeList *next;
};

static	NodeTypeList	*dntq;
static	NodeTypeList	*dntend;

void
defertypecopy(Node *n, Type *t)
{
	NodeTypeList *ntl;

	if(n == N || t == T)
		return;

	ntl = mal(sizeof *ntl);
	ntl->n = n;
	ntl->t = t;
	ntl->next = nil;

	if(dntq == nil)
		dntq = ntl;
	else
		dntend->next = ntl;

	dntend = ntl;
}

void
resumetypecopy(void)
{
	NodeTypeList *l;

	for(l=dntq; l; l=l->next)
		copytype(l->n, l->t);
}

void
copytype(Node *n, Type *t)
{
	*n->type = *t;

	t = n->type;
	t->sym = n->sym;
	t->local = n->local;
	t->vargen = n->vargen;
	t->siggen = 0;
	t->method = nil;
	t->nod = N;
	t->printed = 0;
	t->deferwidth = 0;
}

static void
typecheckdeftype(Node *n)
{
	int maplineno, embedlineno, lno;
	Type *t;
	NodeList *l;

	ntypecheckdeftype++;
	lno = lineno;
	setlineno(n);
	n->type->sym = n->sym;
	n->typecheck = 1;
	typecheck(&n->ntype, Etype);
	if((t = n->ntype->type) == T) {
		n->diag = 1;
		goto ret;
	}
	if(n->type == T) {
		n->diag = 1;
		goto ret;
	}

	maplineno = n->type->maplineno;
	embedlineno = n->type->embedlineno;

	// copy new type and clear fields
	// that don't come along.
	// anything zeroed here must be zeroed in
	// typedcl2 too.
	copytype(n, t);

	// double-check use of type as map key.
	if(maplineno) {
		lineno = maplineno;
		maptype(n->type, types[TBOOL]);
	}
	if(embedlineno) {
		lineno = embedlineno;
		if(isptr[t->etype])
			yyerror("embedded type cannot be a pointer");
	}

ret:
	lineno = lno;

	// if there are no type definitions going on, it's safe to
	// try to resolve the method types for the interfaces
	// we just read.
	if(ntypecheckdeftype == 1) {
		while((l = methodqueue) != nil) {
			methodqueue = nil;
			for(; l; l=l->next)
				domethod(l->n);
		}
	}
	ntypecheckdeftype--;
}

void
queuemethod(Node *n)
{
	if(ntypecheckdeftype == 0) {
		domethod(n);
		return;
	}
	methodqueue = list(methodqueue, n);
}

Node*
typecheckdef(Node *n)
{
	int lno;
	Node *e;
	Type *t;
	NodeList *l;

	lno = lineno;
	setlineno(n);

	if(n->op == ONONAME) {
		if(!n->diag) {
			n->diag = 1;
			if(n->lineno != 0)
				lineno = n->lineno;
			yyerror("undefined: %S", n->sym);
		}
		return n;
	}

	if(n->walkdef == 1)
		return n;

	l = mal(sizeof *l);
	l->n = n;
	l->next = typecheckdefstack;
	typecheckdefstack = l;

	if(n->walkdef == 2) {
		flusherrors();
		print("typecheckdef loop:");
		for(l=typecheckdefstack; l; l=l->next)
			print(" %S", l->n->sym);
		print("\n");
		fatal("typecheckdef loop");
	}
	n->walkdef = 2;

	if(n->type != T || n->sym == S)	// builtin or no name
		goto ret;

	switch(n->op) {
	default:
		fatal("typecheckdef %O", n->op);

	case OGOTO:
	case OLABEL:
		// not really syms
		break;

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
			dump("typecheckdef nil defn", n);
			yyerror("xxx");
		}
		typecheck(&e, Erv | Eiota);
		if(e->type != T && e->op != OLITERAL) {
			yyerror("const initializer must be constant");
			goto ret;
		}
		if(isconst(e, CTNIL)) {
			yyerror("const initializer cannot be nil");
			goto ret;
		}
		t = n->type;
		if(t != T) {
			if(!okforconst[t->etype]) {
				yyerror("invalid constant type %T", t);
				goto ret;
			}
			if(!isideal(e->type) && !eqtype(t, e->type)) {
				yyerror("cannot use %lN as type %T in const initializer", e, t);
				goto ret;
			}
			convlit(&e, t);
		}
		n->val = e->val;
		n->type = e->type;
		break;

	case ONAME:
		if(n->ntype != N) {
			typecheck(&n->ntype, Etype);
			n->type = n->ntype->type;

			if(n->type == T) {
				n->diag = 1;
				goto ret;
			}
		}
		if(n->type != T)
			break;
		if(n->defn == N) {
			if(n->etype != 0)	// like OPRINTN
				break;
			if(nsavederrors+nerrors > 0) {
				// Can have undefined variables in x := foo
				// that make x have an n->ndefn == nil.
				// If there are other errors anyway, don't
				// bother adding to the noise.
				break;
			}
			fatal("var without type, init: %S", n->sym);
		}
		if(n->defn->op == ONAME) {
			typecheck(&n->defn, Erv);
			n->type = n->defn->type;
			break;
		}
		typecheck(&n->defn, Etop);	// fills in n->type
		break;

	case OTYPE:
		if(curfn)
			defercheckwidth();
		n->walkdef = 1;
		n->type = typ(TFORW);
		n->type->sym = n->sym;
		typecheckdeftype(n);
		if(curfn)
			resumecheckwidth();
		break;

	case OPACK:
		// nothing to see here
		break;
	}

ret:
	if(typecheckdefstack->n != n)
		fatal("typecheckdefstack mismatch");
	l = typecheckdefstack;
	typecheckdefstack = l->next;

	lineno = lno;
	n->walkdef = 1;
	return n;
}
