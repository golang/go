// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	<u.h>
#include	<libc.h>
#include	"go.h"
#define	TUP(x,y)	(((x)<<16)|(y))
/*c2go int TUP(int, int); */

static	Val	tocplx(Val);
static	Val	toflt(Val);
static	Val	tostr(Val);
static	Val	copyval(Val);
static	void	cmplxmpy(Mpcplx*, Mpcplx*);
static	void	cmplxdiv(Mpcplx*, Mpcplx*);

/*
 * truncate float literal fv to 32-bit or 64-bit precision
 * according to type; return truncated value.
 */
Mpflt*
truncfltlit(Mpflt *oldv, Type *t)
{
	double d;
	Mpflt *fv;
	Val v;

	if(t == T)
		return oldv;

	memset(&v, 0, sizeof v);
	v.ctype = CTFLT;
	v.u.fval = oldv;
	overflow(v, t);

	fv = mal(sizeof *fv);
	*fv = *oldv;

	// convert large precision literal floating
	// into limited precision (float64 or float32)
	switch(t->etype) {
	case TFLOAT64:
		d = mpgetflt(fv);
		mpmovecflt(fv, d);
		break;

	case TFLOAT32:
		d = mpgetflt32(fv);
		mpmovecflt(fv, d);

		break;
	}
	return fv;
}

/*
 * convert n, if literal, to type t.
 * implicit conversion.
 */
void
convlit(Node **np, Type *t)
{
	convlit1(np, t, 0);
}

/*
 * convert n, if literal, to type t.
 * return a new node if necessary
 * (if n is a named constant, can't edit n->type directly).
 */
void
convlit1(Node **np, Type *t, int explicit)
{
	int ct, et;
	Node *n, *nn;

	n = *np;
	if(n == N || t == T || n->type == T || isideal(t) || n->type == t)
		return;
	if(!explicit && !isideal(n->type))
		return;

	if(n->op == OLITERAL) {
		nn = nod(OXXX, N, N);
		*nn = *n;
		n = nn;
		*np = n;
	}

	switch(n->op) {
	default:
		if(n->type == idealbool) {
			if(t->etype == TBOOL)
				n->type = t;
			else
				n->type = types[TBOOL];
		}
		if(n->type->etype == TIDEAL) {
			convlit(&n->left, t);
			convlit(&n->right, t);
			n->type = t;
		}
		return;
	case OLITERAL:
		// target is invalid type for a constant?  leave alone.
		if(!okforconst[t->etype] && n->type->etype != TNIL) {
			defaultlit(&n, T);
			*np = n;
			return;
		}
		break;
	case OLSH:
	case ORSH:
		convlit1(&n->left, t, explicit && isideal(n->left->type));
		t = n->left->type;
		if(t != T && t->etype == TIDEAL && n->val.ctype != CTINT)
			n->val = toint(n->val);
		if(t != T && !isint[t->etype]) {
			yyerror("invalid operation: %N (shift of type %T)", n, t);
			t = T;
		}
		n->type = t;
		return;
	case OCOMPLEX:
		if(n->type->etype == TIDEAL) {
			switch(t->etype) {
			default:
				// If trying to convert to non-complex type,
				// leave as complex128 and let typechecker complain.
				t = types[TCOMPLEX128];
				//fallthrough
			case TCOMPLEX128:
				n->type = t;
				convlit(&n->left, types[TFLOAT64]);
				convlit(&n->right, types[TFLOAT64]);
				break;
			case TCOMPLEX64:
				n->type = t;
				convlit(&n->left, types[TFLOAT32]);
				convlit(&n->right, types[TFLOAT32]);
				break;
			}
		}
		return;
	}

	// avoided repeated calculations, errors
	if(eqtype(n->type, t))
		return;

	ct = consttype(n);
	if(ct < 0)
		goto bad;

	et = t->etype;
	if(et == TINTER) {
		if(ct == CTNIL && n->type == types[TNIL]) {
			n->type = t;
			return;
		}
		defaultlit(np, T);
		return;
	}

	switch(ct) {
	default:
		goto bad;

	case CTNIL:
		switch(et) {
		default:
			n->type = T;
			goto bad;

		case TSTRING:
			// let normal conversion code handle it
			return;

		case TARRAY:
			if(!isslice(t))
				goto bad;
			break;

		case TPTR32:
		case TPTR64:
		case TINTER:
		case TMAP:
		case TCHAN:
		case TFUNC:
		case TUNSAFEPTR:
			break;

		case TUINTPTR:
			// A nil literal may be converted to uintptr
			// if it is an unsafe.Pointer
			if(n->type->etype == TUNSAFEPTR) {
				n->val.u.xval = mal(sizeof(*n->val.u.xval));
				mpmovecfix(n->val.u.xval, 0);
				n->val.ctype = CTINT;
			} else
				goto bad;
		}
		break;

	case CTSTR:
	case CTBOOL:
		if(et != n->type->etype)
			goto bad;
		break;

	case CTINT:
	case CTRUNE:
	case CTFLT:
	case CTCPLX:
		ct = n->val.ctype;
		if(isint[et]) {
			switch(ct) {
			default:
				goto bad;
			case CTCPLX:
			case CTFLT:
			case CTRUNE:
				n->val = toint(n->val);
				// flowthrough
			case CTINT:
				overflow(n->val, t);
				break;
			}
		} else
		if(isfloat[et]) {
			switch(ct) {
			default:
				goto bad;
			case CTCPLX:
			case CTINT:
			case CTRUNE:
				n->val = toflt(n->val);
				// flowthrough
			case CTFLT:
				n->val.u.fval = truncfltlit(n->val.u.fval, t);
				break;
			}
		} else
		if(iscomplex[et]) {
			switch(ct) {
			default:
				goto bad;
			case CTFLT:
			case CTINT:
			case CTRUNE:
				n->val = tocplx(n->val);
				break;
			case CTCPLX:
				overflow(n->val, t);
				break;
			}
		} else
		if(et == TSTRING && (ct == CTINT || ct == CTRUNE) && explicit)
			n->val = tostr(n->val);
		else
			goto bad;
		break;
	}
	n->type = t;
	return;

bad:
	if(!n->diag) {
		if(!t->broke)
			yyerror("cannot convert %N to type %T", n, t);
		n->diag = 1;
	}
	if(isideal(n->type)) {
		defaultlit(&n, T);
		*np = n;
	}
	return;
}

static Val
copyval(Val v)
{
	Mpint *i;
	Mpflt *f;
	Mpcplx *c;

	switch(v.ctype) {
	case CTINT:
	case CTRUNE:
		i = mal(sizeof(*i));
		mpmovefixfix(i, v.u.xval);
		v.u.xval = i;
		break;
	case CTFLT:
		f = mal(sizeof(*f));
		mpmovefltflt(f, v.u.fval);
		v.u.fval = f;
		break;
	case CTCPLX:
		c = mal(sizeof(*c));
		mpmovefltflt(&c->real, &v.u.cval->real);
		mpmovefltflt(&c->imag, &v.u.cval->imag);
		v.u.cval = c;
		break;
	}
	return v;
}

static Val
tocplx(Val v)
{
	Mpcplx *c;

	switch(v.ctype) {
	case CTINT:
	case CTRUNE:
		c = mal(sizeof(*c));
		mpmovefixflt(&c->real, v.u.xval);
		mpmovecflt(&c->imag, 0.0);
		v.ctype = CTCPLX;
		v.u.cval = c;
		break;
	case CTFLT:
		c = mal(sizeof(*c));
		mpmovefltflt(&c->real, v.u.fval);
		mpmovecflt(&c->imag, 0.0);
		v.ctype = CTCPLX;
		v.u.cval = c;
		break;
	}
	return v;
}

static Val
toflt(Val v)
{
	Mpflt *f;

	switch(v.ctype) {
	case CTINT:
	case CTRUNE:
		f = mal(sizeof(*f));
		mpmovefixflt(f, v.u.xval);
		v.ctype = CTFLT;
		v.u.fval = f;
		break;
	case CTCPLX:
		f = mal(sizeof(*f));
		mpmovefltflt(f, &v.u.cval->real);
		if(mpcmpfltc(&v.u.cval->imag, 0) != 0)
			yyerror("constant %#F%+#Fi truncated to real", &v.u.cval->real, &v.u.cval->imag);
		v.ctype = CTFLT;
		v.u.fval = f;
		break;
	}
	return v;
}

Val
toint(Val v)
{
	Mpint *i;

	switch(v.ctype) {
	case CTRUNE:
		v.ctype = CTINT;
		break;
	case CTFLT:
		i = mal(sizeof(*i));
		if(mpmovefltfix(i, v.u.fval) < 0)
			yyerror("constant %#F truncated to integer", v.u.fval);
		v.ctype = CTINT;
		v.u.xval = i;
		break;
	case CTCPLX:
		i = mal(sizeof(*i));
		if(mpmovefltfix(i, &v.u.cval->real) < 0)
			yyerror("constant %#F%+#Fi truncated to integer", &v.u.cval->real, &v.u.cval->imag);
		if(mpcmpfltc(&v.u.cval->imag, 0) != 0)
			yyerror("constant %#F%+#Fi truncated to real", &v.u.cval->real, &v.u.cval->imag);
		v.ctype = CTINT;
		v.u.xval = i;
		break;
	}
	return v;
}

int
doesoverflow(Val v, Type *t)
{
	switch(v.ctype) {
	case CTINT:
	case CTRUNE:
		if(!isint[t->etype])
			fatal("overflow: %T integer constant", t);
		if(mpcmpfixfix(v.u.xval, minintval[t->etype]) < 0 ||
		   mpcmpfixfix(v.u.xval, maxintval[t->etype]) > 0)
			return 1;
		break;
	case CTFLT:
		if(!isfloat[t->etype])
			fatal("overflow: %T floating-point constant", t);
		if(mpcmpfltflt(v.u.fval, minfltval[t->etype]) <= 0 ||
		   mpcmpfltflt(v.u.fval, maxfltval[t->etype]) >= 0)
			return 1;
		break;
	case CTCPLX:
		if(!iscomplex[t->etype])
			fatal("overflow: %T complex constant", t);
		if(mpcmpfltflt(&v.u.cval->real, minfltval[t->etype]) <= 0 ||
		   mpcmpfltflt(&v.u.cval->real, maxfltval[t->etype]) >= 0 ||
		   mpcmpfltflt(&v.u.cval->imag, minfltval[t->etype]) <= 0 ||
		   mpcmpfltflt(&v.u.cval->imag, maxfltval[t->etype]) >= 0)
			return 1;
		break;
	}
	return 0;
}

void
overflow(Val v, Type *t)
{
	// v has already been converted
	// to appropriate form for t.
	if(t == T || t->etype == TIDEAL)
		return;

	if(!doesoverflow(v, t))
		return;

	switch(v.ctype) {
	case CTINT:
	case CTRUNE:
		yyerror("constant %B overflows %T", v.u.xval, t);
		break;
	case CTFLT:
		yyerror("constant %#F overflows %T", v.u.fval, t);
		break;
	case CTCPLX:
		yyerror("constant %#F overflows %T", v.u.fval, t);
		break;
	}
}

static Val
tostr(Val v)
{
	Rune rune;
	int l;
	Strlit *s;

	switch(v.ctype) {
	case CTINT:
	case CTRUNE:
		if(mpcmpfixfix(v.u.xval, minintval[TINT]) < 0 ||
		   mpcmpfixfix(v.u.xval, maxintval[TINT]) > 0)
			yyerror("overflow in int -> string");
		rune = mpgetfix(v.u.xval);
		l = runelen(rune);
		s = mal(sizeof(*s)+l);
		s->len = l;
		runetochar((char*)s->s, &rune);
		memset(&v, 0, sizeof v);
		v.ctype = CTSTR;
		v.u.sval = s;
		break;

	case CTFLT:
		yyerror("no float -> string");

	case CTNIL:
		memset(&v, 0, sizeof v);
		v.ctype = CTSTR;
		v.u.sval = mal(sizeof *s);
		break;
	}
	return v;
}

int
consttype(Node *n)
{
	if(n == N || n->op != OLITERAL)
		return -1;
	return n->val.ctype;
}

int
isconst(Node *n, int ct)
{
	int t;
	
	t = consttype(n);
	// If the caller is asking for CTINT, allow CTRUNE too.
	// Makes life easier for back ends.
	return t == ct || (ct == CTINT && t == CTRUNE);
}

static Node*
saveorig(Node *n)
{
	Node *n1;

	if(n == n->orig) {
		// duplicate node for n->orig.
		n1 = nod(OLITERAL, N, N);
		n->orig = n1;
		*n1 = *n;
	}
	return n->orig;
}

/*
 * if n is constant, rewrite as OLITERAL node.
 */
void
evconst(Node *n)
{
	Node *nl, *nr, *norig;
	int32 len;
	Strlit *str;
	int wl, wr, lno, et;
	Val v, rv;
	Mpint b;
	NodeList *l1, *l2;

	// pick off just the opcodes that can be
	// constant evaluated.
	switch(n->op) {
	default:
		return;
	case OADD:
	case OAND:
	case OANDAND:
	case OANDNOT:
	case OARRAYBYTESTR:
	case OCOM:
	case ODIV:
	case OEQ:
	case OGE:
	case OGT:
	case OLE:
	case OLSH:
	case OLT:
	case OMINUS:
	case OMOD:
	case OMUL:
	case ONE:
	case ONOT:
	case OOR:
	case OOROR:
	case OPLUS:
	case ORSH:
	case OSUB:
	case OXOR:
		break;
	case OCONV:
		if(n->type == T)
			return;
		if(!okforconst[n->type->etype] && n->type->etype != TNIL)
			return;
		break;
	
	case OADDSTR:
		// merge adjacent constants in the argument list.
		for(l1=n->list; l1 != nil; l1= l1->next) {
			if(isconst(l1->n, CTSTR) && l1->next != nil && isconst(l1->next->n, CTSTR)) {
				l2 = l1;
				len = 0;
				while(l2 != nil && isconst(l2->n, CTSTR)) {
					nr = l2->n;
					len += nr->val.u.sval->len;
					l2 = l2->next;
				}
				// merge from l1 up to but not including l2
				str = mal(sizeof(*str) + len);
				str->len = len;
				len = 0;
				l2 = l1;
				while(l2 != nil && isconst(l2->n, CTSTR)) {
					nr = l2->n;
					memmove(str->s+len, nr->val.u.sval->s, nr->val.u.sval->len);
					len += nr->val.u.sval->len;
					l2 = l2->next;
				}
				nl = nod(OXXX, N, N);
				*nl = *l1->n;
				nl->orig = nl;
				nl->val.ctype = CTSTR;
				nl->val.u.sval = str;
				l1->n = nl;
				l1->next = l2;
			}
		}
		// fix list end pointer.
		for(l2=n->list; l2 != nil; l2=l2->next)
			n->list->end = l2;
		// collapse single-constant list to single constant.
		if(count(n->list) == 1 && isconst(n->list->n, CTSTR)) {
			n->op = OLITERAL;
			n->val = n->list->n->val;
		}
		return;
	}

	nl = n->left;
	if(nl == N || nl->type == T)
		return;
	if(consttype(nl) < 0)
		return;
	wl = nl->type->etype;
	if(isint[wl] || isfloat[wl] || iscomplex[wl])
		wl = TIDEAL;

	nr = n->right;
	if(nr == N)
		goto unary;
	if(nr->type == T)
		return;
	if(consttype(nr) < 0)
		return;
	wr = nr->type->etype;
	if(isint[wr] || isfloat[wr] || iscomplex[wr])
		wr = TIDEAL;

	// check for compatible general types (numeric, string, etc)
	if(wl != wr)
		goto illegal;

	// check for compatible types.
	switch(n->op) {
	default:
		// ideal const mixes with anything but otherwise must match.
		if(nl->type->etype != TIDEAL) {
			defaultlit(&nr, nl->type);
			n->right = nr;
		}
		if(nr->type->etype != TIDEAL) {
			defaultlit(&nl, nr->type);
			n->left = nl;
		}
		if(nl->type->etype != nr->type->etype)
			goto illegal;
		break;

	case OLSH:
	case ORSH:
		// right must be unsigned.
		// left can be ideal.
		defaultlit(&nr, types[TUINT]);
		n->right = nr;
		if(nr->type && (issigned[nr->type->etype] || !isint[nr->type->etype]))
			goto illegal;
		if(nl->val.ctype != CTRUNE)
			nl->val = toint(nl->val);
		nr->val = toint(nr->val);
		break;
	}

	// copy numeric value to avoid modifying
	// n->left, in case someone still refers to it (e.g. iota).
	v = nl->val;
	if(wl == TIDEAL)
		v = copyval(v);

	rv = nr->val;

	// convert to common ideal
	if(v.ctype == CTCPLX || rv.ctype == CTCPLX) {
		v = tocplx(v);
		rv = tocplx(rv);
	}
	if(v.ctype == CTFLT || rv.ctype == CTFLT) {
		v = toflt(v);
		rv = toflt(rv);
	}

	// Rune and int turns into rune.
	if(v.ctype == CTRUNE && rv.ctype == CTINT)
		rv.ctype = CTRUNE;
	if(v.ctype == CTINT && rv.ctype == CTRUNE) {
		if(n->op == OLSH || n->op == ORSH)
			rv.ctype = CTINT;
		else
			v.ctype = CTRUNE;
	}

	if(v.ctype != rv.ctype) {
		// Use of undefined name as constant?
		if((v.ctype == 0 || rv.ctype == 0) && nerrors > 0)
			return;
		fatal("constant type mismatch %T(%d) %T(%d)", nl->type, v.ctype, nr->type, rv.ctype);
	}

	// run op
	switch(TUP(n->op, v.ctype)) {
	default:
		goto illegal;
	case TUP(OADD, CTINT):
	case TUP(OADD, CTRUNE):
		mpaddfixfix(v.u.xval, rv.u.xval, 0);
		break;
	case TUP(OSUB, CTINT):
	case TUP(OSUB, CTRUNE):
		mpsubfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(OMUL, CTINT):
	case TUP(OMUL, CTRUNE):
		mpmulfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(ODIV, CTINT):
	case TUP(ODIV, CTRUNE):
		if(mpcmpfixc(rv.u.xval, 0) == 0) {
			yyerror("division by zero");
			mpmovecfix(v.u.xval, 1);
			break;
		}
		mpdivfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(OMOD, CTINT):
	case TUP(OMOD, CTRUNE):
		if(mpcmpfixc(rv.u.xval, 0) == 0) {
			yyerror("division by zero");
			mpmovecfix(v.u.xval, 1);
			break;
		}
		mpmodfixfix(v.u.xval, rv.u.xval);
		break;

	case TUP(OLSH, CTINT):
	case TUP(OLSH, CTRUNE):
		mplshfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(ORSH, CTINT):
	case TUP(ORSH, CTRUNE):
		mprshfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(OOR, CTINT):
	case TUP(OOR, CTRUNE):
		mporfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(OAND, CTINT):
	case TUP(OAND, CTRUNE):
		mpandfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(OANDNOT, CTINT):
	case TUP(OANDNOT, CTRUNE):
		mpandnotfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(OXOR, CTINT):
	case TUP(OXOR, CTRUNE):
		mpxorfixfix(v.u.xval, rv.u.xval);
		break;

	case TUP(OADD, CTFLT):
		mpaddfltflt(v.u.fval, rv.u.fval);
		break;
	case TUP(OSUB, CTFLT):
		mpsubfltflt(v.u.fval, rv.u.fval);
		break;
	case TUP(OMUL, CTFLT):
		mpmulfltflt(v.u.fval, rv.u.fval);
		break;
	case TUP(ODIV, CTFLT):
		if(mpcmpfltc(rv.u.fval, 0) == 0) {
			yyerror("division by zero");
			mpmovecflt(v.u.fval, 1.0);
			break;
		}
		mpdivfltflt(v.u.fval, rv.u.fval);
		break;
	case TUP(OMOD, CTFLT):
		// The default case above would print 'ideal % ideal',
		// which is not quite an ideal error.
		if(!n->diag) {
			yyerror("illegal constant expression: floating-point %% operation");
			n->diag = 1;
		}
		return;

	case TUP(OADD, CTCPLX):
		mpaddfltflt(&v.u.cval->real, &rv.u.cval->real);
		mpaddfltflt(&v.u.cval->imag, &rv.u.cval->imag);
		break;
	case TUP(OSUB, CTCPLX):
		mpsubfltflt(&v.u.cval->real, &rv.u.cval->real);
		mpsubfltflt(&v.u.cval->imag, &rv.u.cval->imag);
		break;
	case TUP(OMUL, CTCPLX):
		cmplxmpy(v.u.cval, rv.u.cval);
		break;
	case TUP(ODIV, CTCPLX):
		if(mpcmpfltc(&rv.u.cval->real, 0) == 0 &&
		   mpcmpfltc(&rv.u.cval->imag, 0) == 0) {
			yyerror("complex division by zero");
			mpmovecflt(&rv.u.cval->real, 1.0);
			mpmovecflt(&rv.u.cval->imag, 0.0);
			break;
		}
		cmplxdiv(v.u.cval, rv.u.cval);
		break;

	case TUP(OEQ, CTNIL):
		goto settrue;
	case TUP(ONE, CTNIL):
		goto setfalse;

	case TUP(OEQ, CTINT):
	case TUP(OEQ, CTRUNE):
		if(mpcmpfixfix(v.u.xval, rv.u.xval) == 0)
			goto settrue;
		goto setfalse;
	case TUP(ONE, CTINT):
	case TUP(ONE, CTRUNE):
		if(mpcmpfixfix(v.u.xval, rv.u.xval) != 0)
			goto settrue;
		goto setfalse;
	case TUP(OLT, CTINT):
	case TUP(OLT, CTRUNE):
		if(mpcmpfixfix(v.u.xval, rv.u.xval) < 0)
			goto settrue;
		goto setfalse;
	case TUP(OLE, CTINT):
	case TUP(OLE, CTRUNE):
		if(mpcmpfixfix(v.u.xval, rv.u.xval) <= 0)
			goto settrue;
		goto setfalse;
	case TUP(OGE, CTINT):
	case TUP(OGE, CTRUNE):
		if(mpcmpfixfix(v.u.xval, rv.u.xval) >= 0)
			goto settrue;
		goto setfalse;
	case TUP(OGT, CTINT):
	case TUP(OGT, CTRUNE):
		if(mpcmpfixfix(v.u.xval, rv.u.xval) > 0)
			goto settrue;
		goto setfalse;

	case TUP(OEQ, CTFLT):
		if(mpcmpfltflt(v.u.fval, rv.u.fval) == 0)
			goto settrue;
		goto setfalse;
	case TUP(ONE, CTFLT):
		if(mpcmpfltflt(v.u.fval, rv.u.fval) != 0)
			goto settrue;
		goto setfalse;
	case TUP(OLT, CTFLT):
		if(mpcmpfltflt(v.u.fval, rv.u.fval) < 0)
			goto settrue;
		goto setfalse;
	case TUP(OLE, CTFLT):
		if(mpcmpfltflt(v.u.fval, rv.u.fval) <= 0)
			goto settrue;
		goto setfalse;
	case TUP(OGE, CTFLT):
		if(mpcmpfltflt(v.u.fval, rv.u.fval) >= 0)
			goto settrue;
		goto setfalse;
	case TUP(OGT, CTFLT):
		if(mpcmpfltflt(v.u.fval, rv.u.fval) > 0)
			goto settrue;
		goto setfalse;

	case TUP(OEQ, CTCPLX):
		if(mpcmpfltflt(&v.u.cval->real, &rv.u.cval->real) == 0 &&
		   mpcmpfltflt(&v.u.cval->imag, &rv.u.cval->imag) == 0)
			goto settrue;
		goto setfalse;
	case TUP(ONE, CTCPLX):
		if(mpcmpfltflt(&v.u.cval->real, &rv.u.cval->real) != 0 ||
		   mpcmpfltflt(&v.u.cval->imag, &rv.u.cval->imag) != 0)
			goto settrue;
		goto setfalse;

	case TUP(OEQ, CTSTR):
		if(cmpslit(nl, nr) == 0)
			goto settrue;
		goto setfalse;
	case TUP(ONE, CTSTR):
		if(cmpslit(nl, nr) != 0)
			goto settrue;
		goto setfalse;
	case TUP(OLT, CTSTR):
		if(cmpslit(nl, nr) < 0)
			goto settrue;
		goto setfalse;
	case TUP(OLE, CTSTR):
		if(cmpslit(nl, nr) <= 0)
			goto settrue;
		goto setfalse;
	case TUP(OGE, CTSTR):
		if(cmpslit(nl, nr) >= 0l)
			goto settrue;
		goto setfalse;
	case TUP(OGT, CTSTR):
		if(cmpslit(nl, nr) > 0)
			goto settrue;
		goto setfalse;

	case TUP(OOROR, CTBOOL):
		if(v.u.bval || rv.u.bval)
			goto settrue;
		goto setfalse;
	case TUP(OANDAND, CTBOOL):
		if(v.u.bval && rv.u.bval)
			goto settrue;
		goto setfalse;
	case TUP(OEQ, CTBOOL):
		if(v.u.bval == rv.u.bval)
			goto settrue;
		goto setfalse;
	case TUP(ONE, CTBOOL):
		if(v.u.bval != rv.u.bval)
			goto settrue;
		goto setfalse;
	}
	goto ret;

unary:
	// copy numeric value to avoid modifying
	// nl, in case someone still refers to it (e.g. iota).
	v = nl->val;
	if(wl == TIDEAL)
		v = copyval(v);

	switch(TUP(n->op, v.ctype)) {
	default:
		if(!n->diag) {
			yyerror("illegal constant expression %O %T", n->op, nl->type);
			n->diag = 1;
		}
		return;

	case TUP(OCONV, CTNIL):
	case TUP(OARRAYBYTESTR, CTNIL):
		if(n->type->etype == TSTRING) {
			v = tostr(v);
			nl->type = n->type;
			break;
		}
		// fall through
	case TUP(OCONV, CTINT):
	case TUP(OCONV, CTRUNE):
	case TUP(OCONV, CTFLT):
	case TUP(OCONV, CTSTR):
		convlit1(&nl, n->type, 1);
		v = nl->val;
		break;

	case TUP(OPLUS, CTINT):
	case TUP(OPLUS, CTRUNE):
		break;
	case TUP(OMINUS, CTINT):
	case TUP(OMINUS, CTRUNE):
		mpnegfix(v.u.xval);
		break;
	case TUP(OCOM, CTINT):
	case TUP(OCOM, CTRUNE):
		et = Txxx;
		if(nl->type != T)
			et = nl->type->etype;

		// calculate the mask in b
		// result will be (a ^ mask)
		switch(et) {
		default:
			// signed guys change sign
			mpmovecfix(&b, -1);
			break;

		case TUINT8:
		case TUINT16:
		case TUINT32:
		case TUINT64:
		case TUINT:
		case TUINTPTR:
			// unsigned guys invert their bits
			mpmovefixfix(&b, maxintval[et]);
			break;
		}
		mpxorfixfix(v.u.xval, &b);
		break;

	case TUP(OPLUS, CTFLT):
		break;
	case TUP(OMINUS, CTFLT):
		mpnegflt(v.u.fval);
		break;

	case TUP(OPLUS, CTCPLX):
		break;
	case TUP(OMINUS, CTCPLX):
		mpnegflt(&v.u.cval->real);
		mpnegflt(&v.u.cval->imag);
		break;

	case TUP(ONOT, CTBOOL):
		if(!v.u.bval)
			goto settrue;
		goto setfalse;
	}

ret:
	norig = saveorig(n);
	*n = *nl;
	// restore value of n->orig.
	n->orig = norig;
	n->val = v;

	// check range.
	lno = setlineno(n);
	overflow(v, n->type);
	lineno = lno;

	// truncate precision for non-ideal float.
	if(v.ctype == CTFLT && n->type->etype != TIDEAL)
		n->val.u.fval = truncfltlit(v.u.fval, n->type);
	return;

settrue:
	norig = saveorig(n);
	*n = *nodbool(1);
	n->orig = norig;
	return;

setfalse:
	norig = saveorig(n);
	*n = *nodbool(0);
	n->orig = norig;
	return;

illegal:
	if(!n->diag) {
		yyerror("illegal constant expression: %T %O %T",
			nl->type, n->op, nr->type);
		n->diag = 1;
	}
	return;
}

Node*
nodlit(Val v)
{
	Node *n;

	n = nod(OLITERAL, N, N);
	n->val = v;
	switch(v.ctype) {
	default:
		fatal("nodlit ctype %d", v.ctype);
	case CTSTR:
		n->type = idealstring;
		break;
	case CTBOOL:
		n->type = idealbool;
		break;
	case CTINT:
	case CTRUNE:
	case CTFLT:
	case CTCPLX:
		n->type = types[TIDEAL];
		break;
	case CTNIL:
		n->type = types[TNIL];
		break;
	}
	return n;
}

Node*
nodcplxlit(Val r, Val i)
{
	Node *n;
	Mpcplx *c;

	r = toflt(r);
	i = toflt(i);

	c = mal(sizeof(*c));
	n = nod(OLITERAL, N, N);
	n->type = types[TIDEAL];
	n->val.u.cval = c;
	n->val.ctype = CTCPLX;

	if(r.ctype != CTFLT || i.ctype != CTFLT)
		fatal("nodcplxlit ctype %d/%d", r.ctype, i.ctype);

	mpmovefltflt(&c->real, r.u.fval);
	mpmovefltflt(&c->imag, i.u.fval);
	return n;
}

// idealkind returns a constant kind like consttype
// but for an arbitrary "ideal" (untyped constant) expression.
static int
idealkind(Node *n)
{
	int k1, k2;

	if(n == N || !isideal(n->type))
		return CTxxx;

	switch(n->op) {
	default:
		return CTxxx;
	case OLITERAL:
		return n->val.ctype;
	case OADD:
	case OAND:
	case OANDNOT:
	case OCOM:
	case ODIV:
	case OMINUS:
	case OMOD:
	case OMUL:
	case OSUB:
	case OXOR:
	case OOR:
	case OPLUS:
		// numeric kinds.
		k1 = idealkind(n->left);
		k2 = idealkind(n->right);
		if(k1 > k2)
			return k1;
		else
			return k2;
	case OREAL:
	case OIMAG:
		return CTFLT;
	case OCOMPLEX:
		return CTCPLX;
	case OADDSTR:
		return CTSTR;
	case OANDAND:
	case OEQ:
	case OGE:
	case OGT:
	case OLE:
	case OLT:
	case ONE:
	case ONOT:
	case OOROR:
	case OCMPSTR:
	case OCMPIFACE:
		return CTBOOL;
	case OLSH:
	case ORSH:
		// shifts (beware!).
		return idealkind(n->left);
	}
}

void
defaultlit(Node **np, Type *t)
{
	int lno;
	int ctype;
	Node *n, *nn;
	Type *t1;

	n = *np;
	if(n == N || !isideal(n->type))
		return;

	if(n->op == OLITERAL) {
		nn = nod(OXXX, N, N);
		*nn = *n;
		n = nn;
		*np = n;
	}

	lno = setlineno(n);
	ctype = idealkind(n);
	switch(ctype) {
	default:
		if(t != T) {
			convlit(np, t);
			return;
		}
		if(n->val.ctype == CTNIL) {
			lineno = lno;
			if(!n->diag) {
				yyerror("use of untyped nil");
				n->diag = 1;
			}
			n->type = T;
			break;
		}
		if(n->val.ctype == CTSTR) {
			t1 = types[TSTRING];
			convlit(np, t1);
			break;
		}
		yyerror("defaultlit: unknown literal: %N", n);
		break;
	case CTxxx:
		fatal("defaultlit: idealkind is CTxxx: %+N", n);
		break;
	case CTBOOL:
		t1 = types[TBOOL];
		if(t != T && t->etype == TBOOL)
			t1 = t;
		convlit(np, t1);
		break;
	case CTINT:
		t1 = types[TINT];
		goto num;
	case CTRUNE:
		t1 = runetype;
		goto num;
	case CTFLT:
		t1 = types[TFLOAT64];
		goto num;
	case CTCPLX:
		t1 = types[TCOMPLEX128];
		goto num;
	}
	lineno = lno;
	return;

num:
	if(t != T) {
		if(isint[t->etype]) {
			t1 = t;
			n->val = toint(n->val);
		}
		else
		if(isfloat[t->etype]) {
			t1 = t;
			n->val = toflt(n->val);
		}
		else
		if(iscomplex[t->etype]) {
			t1 = t;
			n->val = tocplx(n->val);
		}
	}
	overflow(n->val, t1);
	convlit(np, t1);
	lineno = lno;
	return;
}

/*
 * defaultlit on both nodes simultaneously;
 * if they're both ideal going in they better
 * get the same type going out.
 * force means must assign concrete (non-ideal) type.
 */
void
defaultlit2(Node **lp, Node **rp, int force)
{
	Node *l, *r;
	int lkind, rkind;

	l = *lp;
	r = *rp;
	if(l->type == T || r->type == T)
		return;
	if(!isideal(l->type)) {
		convlit(rp, l->type);
		return;
	}
	if(!isideal(r->type)) {
		convlit(lp, r->type);
		return;
	}
	if(!force)
		return;
	if(l->type->etype == TBOOL) {
		convlit(lp, types[TBOOL]);
		convlit(rp, types[TBOOL]);
	}
	lkind = idealkind(l);
	rkind = idealkind(r);
	if(lkind == CTCPLX || rkind == CTCPLX) {
		convlit(lp, types[TCOMPLEX128]);
		convlit(rp, types[TCOMPLEX128]);
		return;
	}
	if(lkind == CTFLT || rkind == CTFLT) {
		convlit(lp, types[TFLOAT64]);
		convlit(rp, types[TFLOAT64]);
		return;
	}

	if(lkind == CTRUNE || rkind == CTRUNE) {
		convlit(lp, runetype);
		convlit(rp, runetype);
		return;
	}

	convlit(lp, types[TINT]);
	convlit(rp, types[TINT]);
}

int
cmpslit(Node *l, Node *r)
{
	int32 l1, l2, i, m;
	uchar *s1, *s2;

	l1 = l->val.u.sval->len;
	l2 = r->val.u.sval->len;
	s1 = (uchar*)l->val.u.sval->s;
	s2 = (uchar*)r->val.u.sval->s;

	m = l1;
	if(l2 < m)
		m = l2;

	for(i=0; i<m; i++) {
		if(s1[i] == s2[i])
			continue;
		if(s1[i] > s2[i])
			return +1;
		return -1;
	}
	if(l1 == l2)
		return 0;
	if(l1 > l2)
		return +1;
	return -1;
}

int
smallintconst(Node *n)
{
	if(n->op == OLITERAL && isconst(n, CTINT) && n->type != T)
	switch(simtype[n->type->etype]) {
	case TINT8:
	case TUINT8:
	case TINT16:
	case TUINT16:
	case TINT32:
	case TUINT32:
	case TBOOL:
	case TPTR32:
		return 1;
	case TIDEAL:
	case TINT64:
	case TUINT64:
	case TPTR64:
		if(mpcmpfixfix(n->val.u.xval, minintval[TINT32]) < 0
		|| mpcmpfixfix(n->val.u.xval, maxintval[TINT32]) > 0)
			break;
		return 1;
	}
	return 0;
}

long
nonnegconst(Node *n)
{
	if(n->op == OLITERAL && n->type != T)
	switch(simtype[n->type->etype]) {
	case TINT8:
	case TUINT8:
	case TINT16:
	case TUINT16:
	case TINT32:
	case TUINT32:
	case TINT64:
	case TUINT64:
	case TIDEAL:
		// check negative and 2^31
		if(mpcmpfixfix(n->val.u.xval, minintval[TUINT32]) < 0
		|| mpcmpfixfix(n->val.u.xval, maxintval[TINT32]) > 0)
			break;
		return mpgetfix(n->val.u.xval);
	}
	return -1;
}

/*
 * convert x to type et and back to int64
 * for sign extension and truncation.
 */
static int64
iconv(int64 x, int et)
{
	switch(et) {
	case TINT8:
		x = (int8)x;
		break;
	case TUINT8:
		x = (uint8)x;
		break;
	case TINT16:
		x = (int16)x;
		break;
	case TUINT16:
		x = (uint64)x;
		break;
	case TINT32:
		x = (int32)x;
		break;
	case TUINT32:
		x = (uint32)x;
		break;
	case TINT64:
	case TUINT64:
		break;
	}
	return x;
}

/*
 * convert constant val to type t; leave in con.
 * for back end.
 */
void
convconst(Node *con, Type *t, Val *val)
{
	int64 i;
	int tt;

	tt = simsimtype(t);

	// copy the constant for conversion
	nodconst(con, types[TINT8], 0);
	con->type = t;
	con->val = *val;

	if(isint[tt]) {
		con->val.ctype = CTINT;
		con->val.u.xval = mal(sizeof *con->val.u.xval);
		switch(val->ctype) {
		default:
			fatal("convconst ctype=%d %lT", val->ctype, t);
		case CTINT:
		case CTRUNE:
			i = mpgetfix(val->u.xval);
			break;
		case CTBOOL:
			i = val->u.bval;
			break;
		case CTNIL:
			i = 0;
			break;
		}
		i = iconv(i, tt);
		mpmovecfix(con->val.u.xval, i);
		return;
	}

	if(isfloat[tt]) {
		con->val = toflt(con->val);
		if(con->val.ctype != CTFLT)
			fatal("convconst ctype=%d %T", con->val.ctype, t);
		if(tt == TFLOAT32)
			con->val.u.fval = truncfltlit(con->val.u.fval, t);
		return;
	}

	if(iscomplex[tt]) {
		con->val = tocplx(con->val);
		if(tt == TCOMPLEX64) {
			con->val.u.cval->real = *truncfltlit(&con->val.u.cval->real, types[TFLOAT32]);
			con->val.u.cval->imag = *truncfltlit(&con->val.u.cval->imag, types[TFLOAT32]);
		}
		return;
	}

	fatal("convconst %lT constant", t);

}

// complex multiply v *= rv
//	(a, b) * (c, d) = (a*c - b*d, b*c + a*d)
static void
cmplxmpy(Mpcplx *v, Mpcplx *rv)
{
	Mpflt ac, bd, bc, ad;

	mpmovefltflt(&ac, &v->real);
	mpmulfltflt(&ac, &rv->real);	// ac

	mpmovefltflt(&bd, &v->imag);
	mpmulfltflt(&bd, &rv->imag);	// bd

	mpmovefltflt(&bc, &v->imag);
	mpmulfltflt(&bc, &rv->real);	// bc

	mpmovefltflt(&ad, &v->real);
	mpmulfltflt(&ad, &rv->imag);	// ad

	mpmovefltflt(&v->real, &ac);
	mpsubfltflt(&v->real, &bd);	// ac-bd

	mpmovefltflt(&v->imag, &bc);
	mpaddfltflt(&v->imag, &ad);	// bc+ad
}

// complex divide v /= rv
//	(a, b) / (c, d) = ((a*c + b*d), (b*c - a*d))/(c*c + d*d)
static void
cmplxdiv(Mpcplx *v, Mpcplx *rv)
{
	Mpflt ac, bd, bc, ad, cc_plus_dd;

	mpmovefltflt(&cc_plus_dd, &rv->real);
	mpmulfltflt(&cc_plus_dd, &rv->real);	// cc

	mpmovefltflt(&ac, &rv->imag);
	mpmulfltflt(&ac, &rv->imag);		// dd

	mpaddfltflt(&cc_plus_dd, &ac);		// cc+dd

	mpmovefltflt(&ac, &v->real);
	mpmulfltflt(&ac, &rv->real);		// ac

	mpmovefltflt(&bd, &v->imag);
	mpmulfltflt(&bd, &rv->imag);		// bd

	mpmovefltflt(&bc, &v->imag);
	mpmulfltflt(&bc, &rv->real);		// bc

	mpmovefltflt(&ad, &v->real);
	mpmulfltflt(&ad, &rv->imag);		// ad

	mpmovefltflt(&v->real, &ac);
	mpaddfltflt(&v->real, &bd);		// ac+bd
	mpdivfltflt(&v->real, &cc_plus_dd);	// (ac+bd)/(cc+dd)

	mpmovefltflt(&v->imag, &bc);
	mpsubfltflt(&v->imag, &ad);		// bc-ad
	mpdivfltflt(&v->imag, &cc_plus_dd);	// (bc+ad)/(cc+dd)
}

static int hascallchan(Node*);

// Is n a Go language constant (as opposed to a compile-time constant)?
// Expressions derived from nil, like string([]byte(nil)), while they
// may be known at compile time, are not Go language constants.
// Only called for expressions known to evaluated to compile-time
// constants.
int
isgoconst(Node *n)
{
	Node *l;
	Type *t;

	if(n->orig != N)
		n = n->orig;

	switch(n->op) {
	case OADD:
	case OADDSTR:
	case OAND:
	case OANDAND:
	case OANDNOT:
	case OCOM:
	case ODIV:
	case OEQ:
	case OGE:
	case OGT:
	case OLE:
	case OLSH:
	case OLT:
	case OMINUS:
	case OMOD:
	case OMUL:
	case ONE:
	case ONOT:
	case OOR:
	case OOROR:
	case OPLUS:
	case ORSH:
	case OSUB:
	case OXOR:
	case OIOTA:
	case OCOMPLEX:
	case OREAL:
	case OIMAG:
		if(isgoconst(n->left) && (n->right == N || isgoconst(n->right)))
			return 1;
		break;

	case OCONV:
		if(okforconst[n->type->etype] && isgoconst(n->left))
			return 1;
		break;

	case OLEN:
	case OCAP:
		l = n->left;
		if(isgoconst(l))
			return 1;
		// Special case: len/cap is constant when applied to array or
		// pointer to array when the expression does not contain
		// function calls or channel receive operations.
		t = l->type;
		if(t != T && isptr[t->etype])
			t = t->type;
		if(isfixedarray(t) && !hascallchan(l))
			return 1;
		break;

	case OLITERAL:
		if(n->val.ctype != CTNIL)
			return 1;
		break;

	case ONAME:
		l = n->sym->def;
		if(l && l->op == OLITERAL && n->val.ctype != CTNIL)
			return 1;
		break;
	
	case ONONAME:
		if(n->sym->def != N && n->sym->def->op == OIOTA)
			return 1;
		break;
	
	case OCALL:
		// Only constant calls are unsafe.Alignof, Offsetof, and Sizeof.
		l = n->left;
		while(l->op == OPAREN)
			l = l->left;
		if(l->op != ONAME || l->sym->pkg != unsafepkg)
			break;
		if(strcmp(l->sym->name, "Alignof") == 0 ||
		   strcmp(l->sym->name, "Offsetof") == 0 ||
		   strcmp(l->sym->name, "Sizeof") == 0)
			return 1;
		break;		
	}

	//dump("nonconst", n);
	return 0;
}

static int
hascallchan(Node *n)
{
	NodeList *l;

	if(n == N)
		return 0;
	switch(n->op) {
	case OAPPEND:
	case OCALL:
	case OCALLFUNC:
	case OCALLINTER:
	case OCALLMETH:
	case OCAP:
	case OCLOSE:
	case OCOMPLEX:
	case OCOPY:
	case ODELETE:
	case OIMAG:
	case OLEN:
	case OMAKE:
	case ONEW:
	case OPANIC:
	case OPRINT:
	case OPRINTN:
	case OREAL:
	case ORECOVER:
	case ORECV:
		return 1;
	}
	
	if(hascallchan(n->left) ||
	   hascallchan(n->right))
		return 1;
	
	for(l=n->list; l; l=l->next)
		if(hascallchan(l->n))
			return 1;
	for(l=n->rlist; l; l=l->next)
		if(hascallchan(l->n))
			return 1;

	return 0;
}
