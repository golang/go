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
static Type*	lookdot1(Node*, Sym *s, Type *t, Type *f, int);
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
static void	checkdefergo(Node*);
static int	checkmake(Type*, char*, Node*);
static int	checksliceindex(Node*, Node*, Type*);
static int	checksliceconst(Node*, Node*);

static	NodeList*	typecheckdefstack;

/*
 * resolve ONONAME to definition, if any.
 */
static Node*
resolve(Node *n)
{
	Node *r;

	if(n != N && n->op == ONONAME && n->sym != S && (r = n->sym->def) != N) {
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
	[TUNSAFEPTR]	= "unsafe.Pointer",
	[TSTRUCT]	= "struct",
	[TINTER]	= "interface",
	[TCHAN]		= "chan",
	[TMAP]		= "map",
	[TARRAY]	= "array",
	[TFUNC]		= "func",
	[TNIL]		= "nil",
	[TIDEAL]	= "untyped number",
};

static char*
typekind(Type *t)
{
	int et;
	static char buf[50];
	char *s;
	
	if(isslice(t))
		return "slice";
	et = t->etype;
	if(0 <= et && et < nelem(_typekind) && (s=_typekind[et]) != nil)
		return s;
	snprint(buf, sizeof buf, "etype=%d", et);
	return buf;
}

/*
 * sprint_depchain prints a dependency chain
 * of nodes into fmt.
 * It is used by typecheck in the case of OLITERAL nodes
 * to print constant definition loops.
 */
static void
sprint_depchain(Fmt *fmt, NodeList *stack, Node *cur, Node *first)
{
	NodeList *l;

	for(l = stack; l; l=l->next) {
		if(l->n->op == cur->op) {
			if(l->n != first)
				sprint_depchain(fmt, l->next, l->n, first);
			fmtprint(fmt, "\n\t%L: %N uses %N", l->n->lineno, l->n, cur);
			return;
		}
	}
}

/*
 * type check node *np.
 * replaces *np with a new pointer in some cases.
 * returns the final value of *np as a convenience.
 */
static void typecheck1(Node **, int);
Node*
typecheck(Node **np, int top)
{
	Node *n;
	int lno;
	Fmt fmt;
	NodeList *l;
	static NodeList *tcstack, *tcfree;

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
		// Typechecking loop. Trying printing a meaningful message,
		// otherwise a stack trace of typechecking.
		switch(n->op) {
		case ONAME:
			// We can already diagnose variables used as types.
			if((top & (Erv|Etype)) == Etype)
				yyerror("%N is not a type", n);
			break;
		case OLITERAL:
			if((top & (Erv|Etype)) == Etype) {
				yyerror("%N is not a type", n);
				break;
			}
			fmtstrinit(&fmt);
			sprint_depchain(&fmt, tcstack, n, n);
			yyerrorl(n->lineno, "constant definition loop%s", fmtstrflush(&fmt));
			break;
		}
		if(nsavederrors+nerrors == 0) {
			fmtstrinit(&fmt);
			for(l=tcstack; l; l=l->next)
				fmtprint(&fmt, "\n\t%L %N", l->n->lineno, l->n);
			yyerror("typechecking loop involving %N%s", n, fmtstrflush(&fmt));
		}
		lineno = lno;
		return n;
	}
	n->typecheck = 2;

	if(tcfree != nil) {
		l = tcfree;
		tcfree = l->next;
	} else
		l = mal(sizeof *l);
	l->next = tcstack;
	l->n = n;
	tcstack = l;

	typecheck1(&n, top);
	*np = n;
	n->typecheck = 1;

	if(tcstack != l)
		fatal("typecheck stack out of sync");
	tcstack = l->next;
	l->next = tcfree;
	tcfree = l;

	lineno = lno;
	return n;
}

/*
 * does n contain a call or receive operation?
 */
static int callrecvlist(NodeList*);

static int
callrecv(Node *n)
{
	if(n == nil)
		return 0;
	
	switch(n->op) {
	case OCALL:
	case OCALLMETH:
	case OCALLINTER:
	case OCALLFUNC:
	case ORECV:
	case OCAP:
	case OLEN:
	case OCOPY:
	case ONEW:
	case OAPPEND:
	case ODELETE:
		return 1;
	}

	return callrecv(n->left) ||
		callrecv(n->right) ||
		callrecv(n->ntest) ||
		callrecv(n->nincr) ||
		callrecvlist(n->ninit) ||
		callrecvlist(n->nbody) ||
		callrecvlist(n->nelse) ||
		callrecvlist(n->list) ||
		callrecvlist(n->rlist);
}

static int
callrecvlist(NodeList *l)
{
	for(; l; l=l->next)
		if(callrecv(l->n))
			return 1;
	return 0;
}

// indexlit implements typechecking of untyped values as
// array/slice indexes. It is equivalent to defaultlit
// except for constants of numerical kind, which are acceptable
// whenever they can be represented by a value of type int.
static void
indexlit(Node **np)
{
	Node *n;

	n = *np;
	if(n == N || !isideal(n->type))
		return;
	switch(consttype(n)) {
	case CTINT:
	case CTRUNE:
	case CTFLT:
	case CTCPLX:
		defaultlit(np, types[TINT]);
		break;
	}
	defaultlit(np, T);
}

static void
typecheck1(Node **np, int top)
{
	int et, aop, op, ptr;
	Node *n, *l, *r, *lo, *mid, *hi;
	NodeList *args;
	int ok, ntop;
	Type *t, *tp, *missing, *have, *badtype;
	Val v;
	char *why, *desc, descbuf[64];
	vlong x;
	
	n = *np;

	if(n->sym) {
		if(n->op == ONAME && n->etype != 0 && !(top & Ecall)) {
			yyerror("use of builtin %S not in function call", n->sym);
			goto error;
		}

		typecheckdef(n);
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
		if(!(top &Ecall) && isunsafebuiltin(n)) {
			yyerror("%N is not an expression, must be called", n);
			goto error;
		}
		ok |= Erv;
		goto ret;

	case OPACK:
		yyerror("use of package %S without selector", n->sym);
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
	
	case OTARRAY:
		ok |= Etype;
		t = typ(TARRAY);
		l = n->left;
		r = n->right;
		if(l == nil) {
			t->bound = -1;	// slice
		} else if(l->op == ODDD) {
			t->bound = -100;	// to be filled in
			if(!(top&Ecomplit) && !n->diag) {
				t->broke = 1;
				n->diag = 1;
				yyerror("use of [...] array outside of array literal");
			}
		} else {
			l = typecheck(&n->left, Erv);
			switch(consttype(l)) {
			case CTINT:
			case CTRUNE:
				v = l->val;
				break;
			case CTFLT:
				v = toint(l->val);
				break;
			default:
				if(l->type != T && isint[l->type->etype] && l->op != OLITERAL)
					yyerror("non-constant array bound %N", l);
				else
					yyerror("invalid array bound %N", l);
				goto error;
			}
			t->bound = mpgetfix(v.u.xval);
			if(doesoverflow(v, types[TINT])) {
				yyerror("array bound is too large"); 
				goto error;
			} else if(t->bound < 0) {
				yyerror("array bound must be non-negative");
				goto error;
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
		n->type = tostruct(n->list);
		if(n->type == T || n->type->broke)
			goto error;
		n->list = nil;
		break;

	case OTINTER:
		ok |= Etype;
		n->op = OTYPE;
		n->type = tointerface(n->list);
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
		ntop |= top & Ecomplit;
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
			if(top & (Erv | Etop)) {
				yyerror("invalid indirect of %lN", n->left);
				goto error;
			}
			goto ret;
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
			// the same type.
			//
			// the only conversion that isn't a no-op is concrete == interface.
			// in that case, check comparability of the concrete type.
			if(r->type->etype != TBLANK && (aop = assignop(l->type, r->type, nil)) != 0) {
				if(isinter(r->type) && !isinter(l->type) && algtype1(l->type, nil) == ANOEQ) {
					yyerror("invalid operation: %N (operator %O not defined on %s)", n, op, typekind(l->type));
					goto error;
				}
				l = nod(aop, l, N);
				l->type = r->type;
				l->typecheck = 1;
				n->left = l;
				t = l->type;
			} else if(l->type->etype != TBLANK && (aop = assignop(r->type, l->type, nil)) != 0) {
				if(isinter(l->type) && !isinter(r->type) && algtype1(r->type, nil) == ANOEQ) {
					yyerror("invalid operation: %N (operator %O not defined on %s)", n, op, typekind(r->type));
					goto error;
				}
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
			if(n->op == OASOP && n->implicit) {
				yyerror("invalid operation: %N (non-numeric type %T)", n, l->type);
				goto error;
			}
			yyerror("invalid operation: %N (mismatched types %T and %T)", n, l->type, r->type);
			goto error;
		}
		if(!okfor[op][et]) {
			yyerror("invalid operation: %N (operator %O not defined on %s)", n, op, typekind(t));
			goto error;
		}
		// okfor allows any array == array, map == map, func == func.
		// restrict to slice/map/func == nil and nil == slice/map/func.
		if(isfixedarray(l->type) && algtype1(l->type, nil) == ANOEQ) {
			yyerror("invalid operation: %N (%T cannot be compared)", n, l->type);
			goto error;
		}
		if(isslice(l->type) && !isnil(l) && !isnil(r)) {
			yyerror("invalid operation: %N (slice can only be compared to nil)", n);
			goto error;
		}
		if(l->type->etype == TMAP && !isnil(l) && !isnil(r)) {
			yyerror("invalid operation: %N (map can only be compared to nil)", n);
			goto error;
		}
		if(l->type->etype == TFUNC && !isnil(l) && !isnil(r)) {
			yyerror("invalid operation: %N (func can only be compared to nil)", n);
			goto error;
		}
		if(l->type->etype == TSTRUCT && algtype1(l->type, &badtype) == ANOEQ) {
			yyerror("invalid operation: %N (struct containing %T cannot be compared)", n, badtype);
			goto error;
		}
		
		t = l->type;
		if(iscmp[n->op]) {
			evconst(n);
			t = idealbool;
			if(n->op != OLITERAL) {
				defaultlit2(&l, &r, 1);
				n->left = l;
				n->right = r;
			}
		// non-comparison operators on ideal bools should make them lose their ideal-ness
		} else if(t == idealbool)
			t = types[TBOOL];

		if(et == TSTRING) {
			if(iscmp[n->op]) {
				n->etype = n->op;
				n->op = OCMPSTR;
			} else if(n->op == OADD) {
				// create OADDSTR node with list of strings in x + y + z + (w + v) + ...
				n->op = OADDSTR;
				if(l->op == OADDSTR)
					n->list = l->list;
				else
					n->list = list1(l);
				if(r->op == OADDSTR)
					n->list = concat(n->list, r->list);
				else
					n->list = list(n->list, r);
				n->left = N;
				n->right = N;
			}
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

		if((op == ODIV || op == OMOD) && isconst(r, CTINT))
		if(mpcmpfixc(r->val.u.xval, 0) == 0) {
			yyerror("division by zero");
			goto error;
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
		checklvalue(n->left, "take the address of");
		r = outervalue(n->left);
		for(l = n->left; l != r; l = l->left)
			l->addrtaken = 1;
		if(l->orig != l && l->op == ONAME)
			fatal("found non-orig name node %N", l);
		l->addrtaken = 1;
		defaultlit(&n->left, T);
		l = n->left;
		if((t = l->type) == T)
			goto error;
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
		if(n->left == N)
			goto error;
		// fall through
	case ODOT:
		typecheck(&n->left, Erv|Etype);
		defaultlit(&n->left, T);
		if((t = n->left->type) == T)
			goto error;
		if(n->right->op != ONAME) {
			yyerror("rhs of . must be a name");	// impossible
			goto error;
		}
		r = n->right;

		if(n->left->op == OTYPE) {
			if(!looktypedot(n, t, 0)) {
				if(looktypedot(n, t, 1))
					yyerror("%N undefined (cannot refer to unexported method %S)", n, n->right->sym);
				else
					yyerror("%N undefined (type %T has no method %S)", n, t, n->right->sym);
				goto error;
			}
			if(n->type->etype != TFUNC || n->type->thistuple != 1) {
				yyerror("type %T has no method %hS", n->left->type, n->right->sym);
				n->type = T;
				goto error;
			}
			n->op = ONAME;
			n->sym = n->right->sym;
			n->type = methodfunc(n->type, n->left->type);
			n->xoffset = 0;
			n->class = PFUNC;
			ok = Erv;
			goto ret;
		}
		if(isptr[t->etype] && t->type->etype != TINTER) {
			t = t->type;
			if(t == T)
				goto error;
			n->op = ODOTPTR;
			checkwidth(t);
		}
		if(isblank(n->right)) {
			yyerror("cannot refer to blank field or method");
			goto error;
		}
		if(!lookdot(n, t, 0)) {
			if(lookdot(n, t, 1))
				yyerror("%N undefined (cannot refer to unexported field or method %S)", n, n->right->sym);
			else
				yyerror("%N undefined (type %T has no field or method %S)", n, n->left->type, n->right->sym);
			goto error;
		}
		switch(n->op) {
		case ODOTINTER:
		case ODOTMETH:
			if(top&Ecall)
				ok |= Ecall;
			else {
				typecheckpartialcall(n, r);
				ok |= Erv;
			}
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
			if(have && have->sym == missing->sym)
				yyerror("impossible type assertion:\n\t%T does not implement %T (wrong type for %S method)\n"
					"\t\thave %S%hhT\n\t\twant %S%hhT", n->type, t, missing->sym,
					have->sym, have->type, missing->sym, missing->type);
			else if(ptr)
				yyerror("impossible type assertion:\n\t%T does not implement %T (%S method has pointer receiver)",
					n->type, t, missing->sym);
			else if(have)
				yyerror("impossible type assertion:\n\t%T does not implement %T (missing %S method)\n"
					"\t\thave %S%hhT\n\t\twant %S%hhT", n->type, t, missing->sym,
					have->sym, have->type, missing->sym, missing->type);
			else
				yyerror("impossible type assertion:\n\t%T does not implement %T (missing %S method)",
					n->type, t, missing->sym);
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
			yyerror("invalid operation: %N (type %T does not support indexing)", n, t);
			goto error;


		case TSTRING:
		case TARRAY:
			indexlit(&n->right);
			if(t->etype == TSTRING)
				n->type = types[TUINT8];
			else
				n->type = t->type;
			why = "string";
			if(t->etype == TARRAY) {
				if(isfixedarray(t))
					why = "array";
				else
					why = "slice";
			}
			if(n->right->type != T && !isint[n->right->type->etype]) {
				yyerror("non-integer %s index %N", why, n->right);
				break;
			}
			if(isconst(n->right, CTINT)) {
				x = mpgetfix(n->right->val.u.xval);
				if(x < 0)
					yyerror("invalid %s index %N (index must be non-negative)", why, n->right);
				else if(isfixedarray(t) && t->bound > 0 && x >= t->bound)
					yyerror("invalid array index %N (out of bounds for %d-element array)", n->right, t->bound);
				else if(isconst(n->left, CTSTR) && x >= n->left->val.u.sval->len)
					yyerror("invalid string index %N (out of bounds for %d-byte string)", n->right, n->left->val.u.sval->len);
				else if(mpcmpfixfix(n->right->val.u.xval, maxintval[TINT]) > 0)
					yyerror("invalid %s index %N (index too large)", why, n->right);
			}
			break;

		case TMAP:
			n->etype = 0;
			defaultlit(&n->right, t->down);
			if(n->right->type != T)
				n->right = assignconv(n->right, t->down, "map index");
			n->type = t->type;
			n->op = OINDEXMAP;
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
		ok |= Etop;
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
		n->right = assignconv(r, l->type->type, "send");
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
		indexlit(&n->right->left);
		indexlit(&n->right->right);
		l = n->left;
		if(isfixedarray(l->type)) {
			if(!islvalue(n->left)) {
				yyerror("invalid operation %N (slice of unaddressable value)", n);
				goto error;
			}
			n->left = nod(OADDR, n->left, N);
			n->left->implicit = 1;
			typecheck(&n->left, Erv);
			l = n->left;
		}
		if((t = l->type) == T)
			goto error;
		tp = nil;
		if(istype(t, TSTRING)) {
			n->type = t;
			n->op = OSLICESTR;
		} else if(isptr[t->etype] && isfixedarray(t->type)) {
			tp = t->type;
			n->type = typ(TARRAY);
			n->type->type = tp->type;
			n->type->bound = -1;
			dowidth(n->type);
			n->op = OSLICEARR;
		} else if(isslice(t)) {
			n->type = t;
		} else {
			yyerror("cannot slice %N (type %T)", l, t);
			goto error;
		}
		if((lo = n->right->left) != N && checksliceindex(l, lo, tp) < 0)
			goto error;
		if((hi = n->right->right) != N && checksliceindex(l, hi, tp) < 0)
			goto error;
		if(checksliceconst(lo, hi) < 0)
			goto error;
		goto ret;

	case OSLICE3:
		ok |= Erv;
		typecheck(&n->left, top);
		typecheck(&n->right->left, Erv);
		typecheck(&n->right->right->left, Erv);
		typecheck(&n->right->right->right, Erv);
		defaultlit(&n->left, T);
		indexlit(&n->right->left);
		indexlit(&n->right->right->left);
		indexlit(&n->right->right->right);
		l = n->left;
		if(isfixedarray(l->type)) {
			if(!islvalue(n->left)) {
				yyerror("invalid operation %N (slice of unaddressable value)", n);
				goto error;
			}
			n->left = nod(OADDR, n->left, N);
			n->left->implicit = 1;
			typecheck(&n->left, Erv);
			l = n->left;
		}
		if((t = l->type) == T)
			goto error;
		tp = nil;
		if(istype(t, TSTRING)) {
			yyerror("invalid operation %N (3-index slice of string)", n);
			goto error;
		}
		if(isptr[t->etype] && isfixedarray(t->type)) {
			tp = t->type;
			n->type = typ(TARRAY);
			n->type->type = tp->type;
			n->type->bound = -1;
			dowidth(n->type);
			n->op = OSLICE3ARR;
		} else if(isslice(t)) {
			n->type = t;
		} else {
			yyerror("cannot slice %N (type %T)", l, t);
			goto error;
		}
		if((lo = n->right->left) != N && checksliceindex(l, lo, tp) < 0)
			goto error;
		if((mid = n->right->right->left) != N && checksliceindex(l, mid, tp) < 0)
			goto error;
		if((hi = n->right->right->right) != N && checksliceindex(l, hi, tp) < 0)
			goto error;
		if(checksliceconst(lo, hi) < 0 || checksliceconst(lo, mid) < 0 || checksliceconst(mid, hi) < 0)
			goto error;
		goto ret;

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
		n->diag |= n->left->diag;
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
			if(n->isddd || l->type->bound == -100) {
				if(!l->type->broke)
					yyerror("invalid use of ... in type conversion", l);
				n->diag = 1;
			}
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

		if(count(n->list) == 1 && !n->isddd)
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
		if(snprint(descbuf, sizeof descbuf, "argument to %N", n->left) < sizeof descbuf)
			desc = descbuf;
		else
			desc = "function argument";
		typecheckaste(OCALL, n->left, n->isddd, getinargx(t), n->list, desc);
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
				r = n;
				if(n->op == OREAL)
					n = nodfltconst(&l->val.u.cval->real);
				else
					n = nodfltconst(&l->val.u.cval->imag);
				n->orig = r;
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
			if(t->bound < 0) // slice
				break;
			if(callrecv(l)) // has call or receive
				break;
			r = nod(OXXX, N, N);
			nodconst(r, types[TINT], t->bound);
			r->orig = n;
			n = r;
			break;
		}
		n->type = types[TINT];
		goto ret;

	case OCOMPLEX:
		ok |= Erv;
		if(count(n->list) == 1) {
			typechecklist(n->list, Efnstruct);
			t = n->list->n->left->type;
			if(t->outtuple != 2) {
				yyerror("invalid operation: complex expects two arguments, %N returns %d results", n->list->n, t->outtuple);
				goto error;
			}
			t = n->list->n->type->type;
			l = t->nname;
			r = t->down->nname;
		} else {
			if(twoarg(n) < 0)
				goto error;
			l = typecheck(&n->left, Erv | (top & Eiota));
			r = typecheck(&n->right, Erv | (top & Eiota));
			if(l->type == T || r->type == T)
				goto error;
			defaultlit2(&l, &r, 0);
			if(l->type == T || r->type == T)
				goto error;
			n->left = l;
			n->right = r;
		}
		if(!eqtype(l->type, r->type)) {
			yyerror("invalid operation: %N (mismatched types %T and %T)", n, l->type, r->type);
			goto error;
		}
		switch(l->type->etype) {
		default:
			yyerror("invalid operation: %N (arguments have type %T, expected floating-point)", n, l->type, r->type);
			goto error;
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
			r = nodcplxlit(l->val, r->val);
			r->orig = n;
			n = r;
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
			yyerror("invalid operation: %N (cannot close receive-only channel)", n);
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

		if(count(args) == 1 && !n->isddd)
			typecheck(&args->n, Erv | Efnstruct);
		else
			typechecklist(args, Erv);

		if((t = args->n->type) == T)
			goto error;

		// Unpack multiple-return result before type-checking.
		if(istype(t, TSTRUCT)) {
			t = t->type;
			if(istype(t, TFIELD))
				t = t->type;
		}

		n->type = t;
		if(!isslice(t)) {
			if(isconst(args->n, CTNIL)) {
				yyerror("first argument to append must be typed slice; have untyped nil", t);
				goto error;
			}
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
		if(n->left->type == T || n->right->type == T)
			goto error;

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
		saveorignode(n);
		typecheck(&n->left, Erv | (top & (Eindir | Eiota)));
		convlit1(&n->left, n->type, 1);
		if((t = n->left->type) == T || n->type == T)
			goto error;
		if((n->op = convertop(t, n->type, &why)) == 0) {
			if(!n->diag && !n->type->broke) {
				yyerror("cannot convert %lN to type %T%s", n->left, n->type, why);
				n->diag = 1;
			}
			n->op = OCONV;
		}
		switch(n->op) {
		case OCONVNOP:
			if(n->left->op == OLITERAL) {
				r = nod(OXXX, N, N);
				n->op = OCONV;
				n->orig = r;
				*r = *n;
				n->op = OLITERAL;
				n->val = n->left->val;
			}
			break;
		case OSTRARRAYBYTE:
			// do not use stringtoarraylit.
			// generated code and compiler memory footprint is better without it.
			break;
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
		n->list = nil;
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
			r = N;
			if(args != nil) {
				r = args->n;
				args = args->next;
				typecheck(&r, Erv);
			}
			if(l->type == T || (r && r->type == T))
				goto error;
			et = checkmake(t, "len", l) < 0;
			et |= r && checkmake(t, "cap", r) < 0;
			if(et)
				goto error;
			if(isconst(l, CTINT) && r && isconst(r, CTINT) && mpcmpfixfix(l->val.u.xval, r->val.u.xval) > 0) {
				yyerror("len larger than cap in make(%T)", t);
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
				if(checkmake(t, "size", l) < 0)
					goto error;
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
				if(checkmake(t, "buffer", l) < 0)
					goto error;
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
		for(args=n->list; args; args=args->next) {
			// Special case for print: int constant is int64, not int.
			if(isconst(args->n, CTINT))
				defaultlit(&args->n, types[TINT64]);
			else
				defaultlit(&args->n, T);
		}
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
	
	case OITAB:
		ok |= Erv;
		typecheck(&n->left, Erv);
		if((t = n->left->type) == T)
			goto error;
		if(t->etype != TINTER)
			fatal("OITAB of %T", t);
		n->type = ptrto(types[TUINTPTR]);
		goto ret;

	case OSPTR:
		ok |= Erv;
		typecheck(&n->left, Erv);
		if((t = n->left->type) == T)
			goto error;
		if(!isslice(t) && t->etype != TSTRING)
			fatal("OSPTR of %T", t);
		if(t->etype == TSTRING)
			n->type = ptrto(types[TUINT8]);
		else
			n->type = ptrto(t->type);
		goto ret;

	case OCLOSUREVAR:
		ok |= Erv;
		goto ret;
	
	case OCFUNC:
		ok |= Erv;
		typecheck(&n->left, Erv);
		n->type = types[TUINTPTR];
		goto ret;

	case OCONVNOP:
		ok |= Erv;
		typecheck(&n->left, Erv);
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
	case OVARKILL:
		ok |= Etop;
		goto ret;

	case ODEFER:
		ok |= Etop;
		typecheck(&n->left, Etop|Erv);
		if(!n->left->diag)
			checkdefergo(n);
		goto ret;

	case OPROC:
		ok |= Etop;
		typecheck(&n->left, Etop|Eproc|Erv);
		checkdefergo(n);
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
		if(count(n->list) == 1)
			typechecklist(n->list, Erv | Efnstruct);
		else
			typechecklist(n->list, Erv);
		if(curfn == N) {
			yyerror("return outside function");
			goto error;
		}
		if(curfn->type->outnamed && n->list == nil)
			goto ret;
		typecheckaste(ORETURN, nil, 0, getoutargx(curfn->type), n->list, "return argument");
		goto ret;
	
	case ORETJMP:
		ok |= Etop;
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

	if(safemode && !incannedimport && !importpkg && !compiling_wrappers && t && t->etype == TUNSAFEPTR)
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
	// TODO(rsc): simplify
	if((top & (Ecall|Erv|Etype)) && !(top & Etop) && !(ok & (Erv|Etype|Ecall))) {
		yyerror("%N used as value", n);
		goto error;
	}
	if((top & Etop) && !(top & (Ecall|Erv|Etype)) && !(ok & Etop)) {
		if(n->diag == 0) {
			yyerror("%N evaluated but not used", n);
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
	*np = n;
}

static int
checksliceindex(Node *l, Node *r, Type *tp)
{
	Type *t;

	if((t = r->type) == T)
		return -1;
	if(!isint[t->etype]) {
		yyerror("invalid slice index %N (type %T)", r, t);
		return -1;
	}
	if(r->op == OLITERAL) {
		if(mpgetfix(r->val.u.xval) < 0) {
			yyerror("invalid slice index %N (index must be non-negative)", r);
			return -1;
		} else if(tp != nil && tp->bound > 0 && mpgetfix(r->val.u.xval) > tp->bound) {
			yyerror("invalid slice index %N (out of bounds for %d-element array)", r, tp->bound);
			return -1;
		} else if(isconst(l, CTSTR) && mpgetfix(r->val.u.xval) > l->val.u.sval->len) {
			yyerror("invalid slice index %N (out of bounds for %d-byte string)", r, l->val.u.sval->len);
			return -1;
		} else if(mpcmpfixfix(r->val.u.xval, maxintval[TINT]) > 0) {
			yyerror("invalid slice index %N (index too large)", r);
			return -1;
		}
	}
	return 0;
}

static int
checksliceconst(Node *lo, Node *hi)
{
	if(lo != N && hi != N && lo->op == OLITERAL && hi->op == OLITERAL
	   && mpcmpfixfix(lo->val.u.xval, hi->val.u.xval) > 0) {
		yyerror("invalid slice index: %N > %N", lo, hi);
		return -1;
	}
	return 0;
}

static void
checkdefergo(Node *n)
{
	char *what;
	
	what = "defer";
	if(n->op == OPROC)
		what = "go";

	switch(n->left->op) {
	case OCALLINTER:
	case OCALLMETH:
	case OCALLFUNC:
	case OCLOSE:
	case OCOPY:
	case ODELETE:
	case OPANIC:
	case OPRINT:
	case OPRINTN:
	case ORECOVER:
		// ok
		break;
	case OAPPEND:
	case OCAP:
	case OCOMPLEX:
	case OIMAG:
	case OLEN:
	case OMAKE:
	case OMAKESLICE:
	case OMAKECHAN:
	case OMAKEMAP:
	case ONEW:
	case OREAL:
	case OLITERAL: // conversion or unsafe.Alignof, Offsetof, Sizeof
		if(n->left->orig != N && n->left->orig->op == OCONV)
			goto conv;
		yyerror("%s discards result of %N", what, n->left);
		break;
	default:
	conv:
		// type is broken or missing, most likely a method call on a broken type
		// we will warn about the broken type elsewhere. no need to emit a potentially confusing error
		if(n->left->type == T || n->left->type->broke)
			break;

		if(!n->diag) {
			// The syntax made sure it was a call, so this must be
			// a conversion.
			n->diag = 1;
			yyerror("%s requires function call, not conversion", what);
		}
		break;
	}
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
	n->implicit = 1;
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
lookdot1(Node *errnode, Sym *s, Type *t, Type *f, int dostrcmp)
{
	Type *r;

	r = T;
	for(; f!=T; f=f->down) {
		if(dostrcmp && strcmp(f->sym->name, s->name) == 0)
			return f;
		if(f->sym != s)
			continue;
		if(r != T) {
			if(errnode)
				yyerror("ambiguous selector %N", errnode);
			else if(isptr[t->etype])
				yyerror("ambiguous selector (%T).%S", t, s);
			else
				yyerror("ambiguous selector %T.%S", t, s);
			break;
		}
		r = f;
	}
	return r;
}

static int
looktypedot(Node *n, Type *t, int dostrcmp)
{
	Type *f1, *f2;
	Sym *s;
	
	s = n->right->sym;

	if(t->etype == TINTER) {
		f1 = lookdot1(n, s, t, t->type, dostrcmp);
		if(f1 == T)
			return 0;

		n->right = methodname(n->right, t);
		n->xoffset = f1->width;
		n->type = f1->type;
		n->op = ODOTINTER;
		return 1;
	}

	// Find the base type: methtype will fail if t
	// is not of the form T or *T.
	f2 = methtype(t, 0);
	if(f2 == T)
		return 0;

	expandmeth(f2);
	f2 = lookdot1(n, s, f2, f2->xmethod, dostrcmp);
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

static Type*
derefall(Type* t)
{
	while(t && t->etype == tptr)
		t = t->type;
	return t;
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
		f1 = lookdot1(n, s, t, t->type, dostrcmp);

	f2 = T;
	if(n->left->type == t || n->left->type->sym == S) {
		f2 = methtype(t, 0);
		if(f2 != T) {
			// Use f2->method, not f2->xmethod: adddot has
			// already inserted all the necessary embedded dots.
			f2 = lookdot1(n, s, f2, f2->method, dostrcmp);
		}
	}

	if(f1 != T) {
		if(f2 != T)
			yyerror("%S is both field and method",
				n->right->sym);
		if(f1->width == BADWIDTH)
			fatal("lookdot badwidth %T %p", f1, f1);
		n->xoffset = f1->width;
		n->type = f1->type;
		n->paramfld = f1;
		if(t->etype == TINTER) {
			if(isptr[n->left->type->etype]) {
				n->left = nod(OIND, n->left, N);	// implicitstar
				n->left->implicit = 1;
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
				n->left = nod(OADDR, n->left, N);
				n->left->implicit = 1;
				typecheck(&n->left, Etype|Erv);
			} else if(tt->etype == tptr && rcvr->etype != tptr && eqtype(tt->type, rcvr)) {
				n->left = nod(OIND, n->left, N);
				n->left->implicit = 1;
				typecheck(&n->left, Etype|Erv);
			} else if(tt->etype == tptr && tt->type->etype == tptr && eqtype(derefall(tt), derefall(rcvr))) {
				yyerror("calling method %N with receiver %lN requires explicit dereference", n->right, n->left);
				while(tt->etype == tptr) {
					// Stop one level early for method with pointer receiver.
					if(rcvr->etype == tptr && tt->type->etype != tptr)
						break;
					n->left = nod(OIND, n->left, N);
					n->left->implicit = 1;
					typecheck(&n->left, Etype|Erv);
					tt = tt->type;
				}
			} else {
				fatal("method mismatch: %T for %T", rcvr, tt);
			}
		}
		n->right = methodname(n->right, n->left->type);
		n->xoffset = f2->width;
		n->type = f2->type;
//		print("lookdot found [%p] %T\n", f2->type, f2->type);
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

static int
hasddd(Type *t)
{
	Type *tl;

	for(tl=t->type; tl; tl=tl->down) {
		if(tl->isddd)
			return 1;
	}
	return 0;
}

static int
downcount(Type *t)
{
	Type *tl;
	int n;

	n = 0;
	for(tl=t->type; tl; tl=tl->down) {
		n++;
	}
	return n;
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
	int n1, n2;

	lno = lineno;

	if(tstruct->broke)
		goto out;

	n = N;
	if(nl != nil && nl->next == nil && (n = nl->n)->type != T)
	if(n->type->etype == TSTRUCT && n->type->funarg) {
		if(!hasddd(tstruct)) {
			n1 = downcount(tstruct);
			n2 = downcount(n->type);
			if(n2 > n1)
				goto toomany;
			if(n2 < n1)
				goto notenough;
		}
		
		tn = n->type->type;
		for(tl=tstruct->type; tl; tl=tl->down) {
			if(tl->isddd) {
				for(; tn; tn=tn->down) {
					if(assignop(tn->type, tl->type->type, &why) == 0) {
						if(call != N)
							yyerror("cannot use %T as type %T in argument to %N%s", tn->type, tl->type->type, call, why);
						else
							yyerror("cannot use %T as type %T in %s%s", tn->type, tl->type->type, desc, why);
					}
				}
				goto out;
			}
			if(tn == T)
				goto notenough;
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

	n1 = downcount(tstruct);
	n2 = count(nl);
	if(!hasddd(tstruct)) {
		if(n2 > n1)
			goto toomany;
		if(n2 < n1)
			goto notenough;
	}
	else {
		if(!isddd) {
			if(n2 < n1-1)
				goto notenough;
		} else {
			if(n2 > n1)
				goto toomany;
			if(n2 < n1)
				goto notenough;
		}
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
	if(n == N || !n->diag) {
		if(call != N)
			yyerror("not enough arguments in call to %N", call);
		else
			yyerror("not enough arguments to %O", op);
		if(n != N)
			n->diag = 1;
	}
	goto out;

toomany:
	if(call != N)
		yyerror("too many arguments in call to %N", call);
	else
		yyerror("too many arguments to %O", op);
	goto out;
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
	Node *a, *orign;
	Node cmp;
	char *s;

	orign = n;
	if(n->op == OCONVIFACE)
		n = n->left;
	evconst(n);
	if(n->op != OLITERAL)
		return;	// we dont check variables

	switch(n->val.ctype) {
	default:	// unknown, bool, nil
		b = 23;
		break;
	case CTINT:
	case CTRUNE:
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
		b = 0;
		if(a->op == OCONVIFACE && orign->op == OCONVIFACE) {
			if(eqtype(a->left->type, n->type)) {
				cmp.right = a->left;
				evconst(&cmp);
				b = cmp.val.u.bval;
			}
		} else if(eqtype(a->type, n->type)) {
			cmp.right = a;
			evconst(&cmp);
			b = cmp.val.u.bval;
		}
		if(b) {
			yyerror("duplicate key %N in map literal", n);
			return;
		}
	}
	orign->ntest = hash[h];
	hash[h] = orign;
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

static int
iscomptype(Type *t)
{
	switch(t->etype) {
	case TARRAY:
	case TSTRUCT:
	case TMAP:
		return 1;
	case TPTR32:
	case TPTR64:
		switch(t->type->etype) {
		case TARRAY:
		case TSTRUCT:
		case TMAP:
			return 1;
		}
		break;
	}
	return 0;
}

static void
pushtype(Node *n, Type *t)
{
	if(n == N || n->op != OCOMPLIT || !iscomptype(t))
		return;
	
	if(n->right == N) {
		n->right = typenod(t);
		n->implicit = 1;  // don't print
		n->right->implicit = 1;  // * is okay
	}
	else if(debug['s']) {
		typecheck(&n->right, Etype);
		if(n->right->type != T && eqtype(n->right->type, t))
			print("%lL: redundant type: %T\n", n->lineno, t);
	}
}

static void
typecheckcomplit(Node **np)
{
	int bad, i, nerr;
	int64 len;
	Node *l, *n, *norig, *r, **hash;
	NodeList *ll;
	Type *t, *f;
	Sym *s, *s1;
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

	// Save original node (including n->right)
	norig = nod(n->op, N, N);
	*norig = *n;

	setlineno(n->right);
	l = typecheck(&n->right /* sic */, Etype|Ecomplit);
	if((t = l->type) == T)
		goto error;
	nerr = nerrors;
	n->type = t;

	if(isptr[t->etype]) {
		// For better or worse, we don't allow pointers as the composite literal type,
		// except when using the &T syntax, which sets implicit on the OIND.
		if(!n->right->implicit) {
			yyerror("invalid pointer type %T for composite literal (use &%T instead)", t, t->type);
			goto error;
		}
		// Also, the underlying type must be a struct, map, slice, or array.
		if(!iscomptype(t)) {
			yyerror("invalid pointer type %T for composite literal", t);
			goto error;
		}
		t = t->type;
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
			if(i < 0 && !l->left->diag) {
				yyerror("array index must be non-negative integer constant");
				l->left->diag = 1;
				i = -(1<<30);	// stay negative for a while
			}
			if(i >= 0)
				indexdup(l->left, hash, nhash);
			i++;
			if(i > len) {
				len = i;
				if(t->bound >= 0 && len > t->bound) {
					setlineno(l);
					yyerror("array index %lld out of bounds [0:%lld]", len-1, t->bound);
					t->bound = -1;	// no more errors
				}
			}

			r = l->right;
			pushtype(r, t->type);
			typecheck(&r, Erv);
			defaultlit(&r, t->type);
			l->right = assignconv(r, t->type, "array element");
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
			if (l->left->op != OCONV)
				keydup(l->left, hash, nhash);

			r = l->right;
			pushtype(r, t->type);
			typecheck(&r, Erv);
			defaultlit(&r, t->type);
			l->right = assignconv(r, t->type, "map value");
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
				// No pushtype allowed here.  Must name fields for that.
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
				if(s->pkg != localpkg && exportname(s->name)) {
					s1 = lookup(s->name);
					if(s1->origpkg == s->pkg)
						s = s1;
				}
				f = lookdot1(nil, s, t, t->type, 0);
				if(f == nil) {
					yyerror("unknown %T field '%S' in struct literal", t, s);
					continue;
				}
				l->left = newname(s);
				l->left->typecheck = 1;
				l->left->type = f;
				s = f->sym;
				fielddup(newname(s), hash, nhash);
				r = l->right;
				// No pushtype allowed here.  Tried and rejected.
				typecheck(&r, Erv);
				l->right = assignconv(r, f->type, "field value");
			}
		}
		n->op = OSTRUCTLIT;
		break;
	}
	if(nerr != nerrors)
		goto error;
	
	n->orig = norig;
	if(isptr[n->type->etype]) {
		n = nod(OPTRLIT, n, N);
		n->typecheck = 1;
		n->type = n->left->type;
		n->left->type = t;
		n->left->typecheck = 1;
	}

	n->orig = norig;
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
	case OCLOSUREVAR:
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

	// have already complained about n being undefined
	if(n->op == ONONAME)
		return;

	yyerror("cannot assign to %N", n);
}

static void
checkassignlist(NodeList *l)
{
	for(; l; l=l->next)
		checkassign(l->n);
}

// Check whether l and r are the same side effect-free expression,
// so that it is safe to reuse one instead of computing both.
static int
samesafeexpr(Node *l, Node *r)
{
	if(l->op != r->op || !eqtype(l->type, r->type))
		return 0;
	
	switch(l->op) {
	case ONAME:
	case OCLOSUREVAR:
		return l == r;
	
	case ODOT:
	case ODOTPTR:
		return l->right != nil && r->right != nil && l->right->sym == r->right->sym && samesafeexpr(l->left, r->left);
	
	case OIND:
		return samesafeexpr(l->left, r->left);
	
	case OINDEX:
		return samesafeexpr(l->left, r->left) && samesafeexpr(l->right, r->right);
	}
	
	return 0;
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
	
	// Recognize slices being updated in place, for better code generation later.
	// Don't rewrite if using race detector, to avoid needing to teach race detector
	// about this optimization.
	if(n->left && n->left->op != OINDEXMAP && n->right && !flag_race) {
		switch(n->right->op) {
		case OSLICE:
		case OSLICE3:
		case OSLICESTR:
			// For x = x[0:y], x can be updated in place, without touching pointer.
			// TODO(rsc): Reenable once it is actually updated in place without touching the pointer.
			if(0 && samesafeexpr(n->left, n->right->left) && (n->right->right->left == N || iszero(n->right->right->left)))
				n->right->reslice = 1;
			break;
		
		case OAPPEND:
			// For x = append(x, ...), x can be updated in place when there is capacity,
			// without touching the pointer; otherwise the emitted code to growslice
			// can take care of updating the pointer, and only in that case.
			// TODO(rsc): Reenable once the emitted code does update the pointer.
			if(0 && n->right->list != nil && samesafeexpr(n->left, n->right->list->n))
				n->right->reslice = 1;
			break;
		}
	}
}

static void
checkassignto(Type *src, Node *dst)
{
	char *why;

	if(assignop(src, dst->type, &why) == 0) {
		yyerror("cannot assign %T to %lN in multiple assignment%s", src, dst, why);
		return;
	}
}

static void
typecheckas2(Node *n)
{
	int cl, cr;
	NodeList *ll, *lr;
	Node *l, *r;
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
		yyerror("assignment count mismatch: %d = %d (use delete)", cl, cr);
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
				if(t->type != T && ll->n->type != T)
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
			if(l->type != T && l->type->etype != TBOOL)
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
	t->nname = n->nname;
	rcvr = getthisx(t)->type;
	if(rcvr != nil && n->shortname != N && !isblank(n->shortname))
		addmethod(n->shortname->sym, t, 1, n->nname->nointerface);
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


static int ntypecheckdeftype;
static NodeList *methodqueue;

static void
domethod(Node *n)
{
	Node *nt;
	Type *t;

	nt = n->type->nname;
	typecheck(&nt, Etype);
	if(nt->type == T) {
		// type check failed; leave empty func
		n->type->etype = TFUNC;
		n->type->nod = N;
		return;
	}
	
	// If we have
	//	type I interface {
	//		M(_ int)
	//	}
	// then even though I.M looks like it doesn't care about the
	// value of its argument, a specific implementation of I may
	// care.  The _ would suppress the assignment to that argument
	// while generating a call, so remove it.
	for(t=getinargx(nt->type)->type; t; t=t->down) {
		if(t->sym != nil && strcmp(t->sym->name, "_") == 0)
			t->sym = nil;
	}

	*n->type = *nt->type;
	n->type->nod = N;
	checkwidth(n->type);
}

static NodeList *mapqueue;

void
copytype(Node *n, Type *t)
{
	int maplineno, embedlineno, lno;
	NodeList *l;

	if(t->etype == TFORW) {
		// This type isn't computed yet; when it is, update n.
		t->copyto = list(t->copyto, n);
		return;
	}

	maplineno = n->type->maplineno;
	embedlineno = n->type->embedlineno;

	l = n->type->copyto;
	*n->type = *t;

	t = n->type;
	t->sym = n->sym;
	t->local = n->local;
	t->vargen = n->vargen;
	t->siggen = 0;
	t->method = nil;
	t->xmethod = nil;
	t->nod = N;
	t->printed = 0;
	t->deferwidth = 0;
	t->copyto = nil;
	
	// Update nodes waiting on this type.
	for(; l; l=l->next)
		copytype(l->n, t);

	// Double-check use of type as embedded type.
	lno = lineno;
	if(embedlineno) {
		lineno = embedlineno;
		if(isptr[t->etype])
			yyerror("embedded type cannot be a pointer");
	}
	lineno = lno;
	
	// Queue check for map until all the types are done settling.
	if(maplineno) {
		t->maplineno = maplineno;
		mapqueue = list(mapqueue, n);
	}
}

static void
typecheckdeftype(Node *n)
{
	int lno;
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
		n->type = T;
		goto ret;
	}
	if(n->type == T) {
		n->diag = 1;
		goto ret;
	}

	// copy new type and clear fields
	// that don't come along.
	// anything zeroed here must be zeroed in
	// typedcl2 too.
	copytype(n, t);

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
		for(l=mapqueue; l; l=l->next) {
			lineno = l->n->type->maplineno;
			maptype(l->n->type, types[TBOOL]);
		}
		lineno = lno;
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
	int lno, nerrors0;
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
		if(isconst(e, CTNIL)) {
			yyerror("const initializer cannot be nil");
			goto ret;
		}
		if(e->type != T && e->op != OLITERAL || !isgoconst(e)) {
			if(!e->diag) {
				yyerror("const initializer %N is not a constant", e);
				e->diag = 1;
			}
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
		nerrors0 = nerrors;
		typecheckdeftype(n);
		if(n->type->etype == TFORW && nerrors > nerrors0) {
			// Something went wrong during type-checking,
			// but it was reported. Silence future errors.
			n->type->broke = 1;
		}
		if(curfn)
			resumecheckwidth();
		break;

	case OPACK:
		// nothing to see here
		break;
	}

ret:
	if(n->op != OLITERAL && n->type != T && isideal(n->type))
		fatal("got %T for %N", n->type, n);
	if(typecheckdefstack->n != n)
		fatal("typecheckdefstack mismatch");
	l = typecheckdefstack;
	typecheckdefstack = l->next;

	lineno = lno;
	n->walkdef = 1;
	return n;
}

static int
checkmake(Type *t, char *arg, Node *n)
{
	if(n->op == OLITERAL) {
		switch(n->val.ctype) {
		case CTINT:
		case CTRUNE:
		case CTFLT:
		case CTCPLX:
			n->val = toint(n->val);
			if(mpcmpfixc(n->val.u.xval, 0) < 0) {
				yyerror("negative %s argument in make(%T)", arg, t);
				return -1;
			}
			if(mpcmpfixfix(n->val.u.xval, maxintval[TINT]) > 0) {
				yyerror("%s argument too large in make(%T)", arg, t);
				return -1;
			}
			
			// Delay defaultlit until after we've checked range, to avoid
			// a redundant "constant NNN overflows int" error.
			defaultlit(&n, types[TINT]);
			return 0;
		default:
		       	break;
		}
	}

	if(!isint[n->type->etype] && n->type->etype != TIDEAL) {
		yyerror("non-integer %s argument in make(%T) - %T", arg, t, n->type);
		return -1;
	}

	// Defaultlit still necessary for non-constant: n might be 1<<k.
	defaultlit(&n, types[TINT]);

	return 0;
}

static void	markbreaklist(NodeList*, Node*);

static void
markbreak(Node *n, Node *implicit)
{
	Label *lab;

	if(n == N)
		return;

	switch(n->op) {
	case OBREAK:
		if(n->left == N) {
			if(implicit)
				implicit->hasbreak = 1;
		} else {
			lab = n->left->sym->label;
			if(lab != L)
				lab->def->hasbreak = 1;
		}
		break;
	
	case OFOR:
	case OSWITCH:
	case OTYPESW:
	case OSELECT:
	case ORANGE:
		implicit = n;
		// fall through
	
	default:
		markbreak(n->left, implicit);
		markbreak(n->right, implicit);
		markbreak(n->ntest, implicit);
		markbreak(n->nincr, implicit);
		markbreaklist(n->ninit, implicit);
		markbreaklist(n->nbody, implicit);
		markbreaklist(n->nelse, implicit);
		markbreaklist(n->list, implicit);
		markbreaklist(n->rlist, implicit);
		break;
	}
}

static void
markbreaklist(NodeList *l, Node *implicit)
{
	Node *n;
	Label *lab;

	for(; l; l=l->next) {
		n = l->n;
		if(n->op == OLABEL && l->next && n->defn == l->next->n) {
			switch(n->defn->op) {
			case OFOR:
			case OSWITCH:
			case OTYPESW:
			case OSELECT:
			case ORANGE:
				lab = mal(sizeof *lab);
				lab->def = n->defn;
				n->left->sym->label = lab;
				markbreak(n->defn, n->defn);
				n->left->sym->label = L;
				l = l->next;
				continue;
			}
		}
		markbreak(n, implicit);
	}
}

static int
isterminating(NodeList *l, int top)
{
	int def;
	Node *n;

	if(l == nil)
		return 0;
	if(top) {
		while(l->next && l->n->op != OLABEL)
			l = l->next;
		markbreaklist(l, nil);
	}
	while(l->next)
		l = l->next;
	n = l->n;

	if(n == N)
		return 0;

	switch(n->op) {
	// NOTE: OLABEL is treated as a separate statement,
	// not a separate prefix, so skipping to the last statement
	// in the block handles the labeled statement case by
	// skipping over the label. No case OLABEL here.

	case OBLOCK:
		return isterminating(n->list, 0);

	case OGOTO:
	case ORETURN:
	case ORETJMP:
	case OPANIC:
	case OXFALL:
		return 1;

	case OFOR:
		if(n->ntest != N)
			return 0;
		if(n->hasbreak)
			return 0;
		return 1;

	case OIF:
		return isterminating(n->nbody, 0) && isterminating(n->nelse, 0);

	case OSWITCH:
	case OTYPESW:
	case OSELECT:
		if(n->hasbreak)
			return 0;
		def = 0;
		for(l=n->list; l; l=l->next) {
			if(!isterminating(l->n->nbody, 0))
				return 0;
			if(l->n->list == nil) // default
				def = 1;
		}
		if(n->op != OSELECT && !def)
			return 0;
		return 1;
	}
	
	return 0;
}

void
checkreturn(Node *fn)
{
	if(fn->type->outtuple && fn->nbody != nil)
		if(!isterminating(fn->nbody, 1))
			yyerrorl(fn->endlineno, "missing return at end of function");
}
