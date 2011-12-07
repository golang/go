// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	<u.h>
#include	<libc.h>
#include	"go.h"
#define	TUP(x,y)	(((x)<<16)|(y))

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
	float f;
	Mpflt *fv;

	if(t == T)
		return oldv;

	fv = mal(sizeof *fv);
	*fv = *oldv;

	// convert large precision literal floating
	// into limited precision (float64 or float32)
	// botch -- this assumes that compiler fp
	//    has same precision as runtime fp
	switch(t->etype) {
	case TFLOAT64:
		d = mpgetflt(fv);
		mpmovecflt(fv, d);
		break;

	case TFLOAT32:
		d = mpgetflt(fv);
		f = d;
		d = f;
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
		}
		break;

	case CTSTR:
	case CTBOOL:
		if(et != n->type->etype)
			goto bad;
		break;

	case CTINT:
	case CTFLT:
	case CTCPLX:
		ct = n->val.ctype;
		if(isint[et]) {
			switch(ct) {
			default:
				goto bad;
			case CTCPLX:
			case CTFLT:
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
				n->val = toflt(n->val);
				// flowthrough
			case CTFLT:
				overflow(n->val, t);
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
				n->val = tocplx(n->val);
				break;
			case CTCPLX:
				overflow(n->val, t);
				break;
			}
		} else
		if(et == TSTRING && ct == CTINT && explicit)
			n->val = tostr(n->val);
		else
			goto bad;
		break;
	}
	n->type = t;
	return;

bad:
	if(!n->diag) {
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

void
overflow(Val v, Type *t)
{
	// v has already been converted
	// to appropriate form for t.
	if(t == T || t->etype == TIDEAL)
		return;
	switch(v.ctype) {
	case CTINT:
		if(!isint[t->etype])
			fatal("overflow: %T integer constant", t);
		if(mpcmpfixfix(v.u.xval, minintval[t->etype]) < 0 ||
		   mpcmpfixfix(v.u.xval, maxintval[t->etype]) > 0)
			yyerror("constant %B overflows %T", v.u.xval, t);
		break;
	case CTFLT:
		if(!isfloat[t->etype])
			fatal("overflow: %T floating-point constant", t);
		if(mpcmpfltflt(v.u.fval, minfltval[t->etype]) <= 0 ||
		   mpcmpfltflt(v.u.fval, maxfltval[t->etype]) >= 0)
			yyerror("constant %#F overflows %T", v.u.fval, t);
		break;
	case CTCPLX:
		if(!iscomplex[t->etype])
			fatal("overflow: %T complex constant", t);
		if(mpcmpfltflt(&v.u.cval->real, minfltval[t->etype]) <= 0 ||
		   mpcmpfltflt(&v.u.cval->real, maxfltval[t->etype]) >= 0 ||
		   mpcmpfltflt(&v.u.cval->imag, minfltval[t->etype]) <= 0 ||
		   mpcmpfltflt(&v.u.cval->imag, maxfltval[t->etype]) >= 0)
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
	return consttype(n) == ct;
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

	// pick off just the opcodes that can be
	// constant evaluated.
	switch(n->op) {
	default:
		return;
	case OADD:
	case OADDSTR:
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
	if(v.ctype != rv.ctype) {
		// Use of undefined name as constant?
		if((v.ctype == 0 || rv.ctype == 0) && nerrors > 0)
			return;
		fatal("constant type mismatch %T(%d) %T(%d)", nl->type, v.ctype, nr->type, rv.ctype);
	}

	// run op
	switch(TUP(n->op, v.ctype)) {
	default:
	illegal:
		if(!n->diag) {
			yyerror("illegal constant expression: %T %O %T",
				nl->type, n->op, nr->type);
			n->diag = 1;
		}
		return;

	case TUP(OADD, CTINT):
		mpaddfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(OSUB, CTINT):
		mpsubfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(OMUL, CTINT):
		mpmulfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(ODIV, CTINT):
		if(mpcmpfixc(rv.u.xval, 0) == 0) {
			yyerror("division by zero");
			mpmovecfix(v.u.xval, 1);
			break;
		}
		mpdivfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(OMOD, CTINT):
		if(mpcmpfixc(rv.u.xval, 0) == 0) {
			yyerror("division by zero");
			mpmovecfix(v.u.xval, 1);
			break;
		}
		mpmodfixfix(v.u.xval, rv.u.xval);
		break;

	case TUP(OLSH, CTINT):
		mplshfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(ORSH, CTINT):
		mprshfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(OOR, CTINT):
		mporfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(OAND, CTINT):
		mpandfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(OANDNOT, CTINT):
		mpandnotfixfix(v.u.xval, rv.u.xval);
		break;
	case TUP(OXOR, CTINT):
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
		if(mpcmpfixfix(v.u.xval, rv.u.xval) == 0)
			goto settrue;
		goto setfalse;
	case TUP(ONE, CTINT):
		if(mpcmpfixfix(v.u.xval, rv.u.xval) != 0)
			goto settrue;
		goto setfalse;
	case TUP(OLT, CTINT):
		if(mpcmpfixfix(v.u.xval, rv.u.xval) < 0)
			goto settrue;
		goto setfalse;
	case TUP(OLE, CTINT):
		if(mpcmpfixfix(v.u.xval, rv.u.xval) <= 0)
			goto settrue;
		goto setfalse;
	case TUP(OGE, CTINT):
		if(mpcmpfixfix(v.u.xval, rv.u.xval) >= 0)
			goto settrue;
		goto setfalse;
	case TUP(OGT, CTINT):
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
	case TUP(OADDSTR, CTSTR):
		len = v.u.sval->len + rv.u.sval->len;
		str = mal(sizeof(*str) + len);
		str->len = len;
		memcpy(str->s, v.u.sval->s, v.u.sval->len);
		memcpy(str->s+v.u.sval->len, rv.u.sval->s, rv.u.sval->len);
		str->len = len;
		v.u.sval = str;
		break;

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
	case TUP(OCONV, CTFLT):
	case TUP(OCONV, CTSTR):
		convlit1(&nl, n->type, 1);
		break;

	case TUP(OPLUS, CTINT):
		break;
	case TUP(OMINUS, CTINT):
		mpnegfix(v.u.xval);
		break;
	case TUP(OCOM, CTINT):
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
	if(n == n->orig) {
		// duplicate node for n->orig.
		norig = nod(OLITERAL, N, N);
		*norig = *n;
	} else
		norig = n->orig;
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
	*n = *nodbool(1);
	return;

setfalse:
	*n = *nodbool(0);
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

// TODO(rsc): combine with convlit
void
defaultlit(Node **np, Type *t)
{
	int lno;
	Node *n, *nn;

	n = *np;
	if(n == N || !isideal(n->type))
		return;

	switch(n->op) {
	case OLITERAL:
		nn = nod(OXXX, N, N);
		*nn = *n;
		n = nn;
		*np = n;
		break;
	case OLSH:
	case ORSH:
		defaultlit(&n->left, t);
		t = n->left->type;
		if(t != T && !isint[t->etype]) {
			yyerror("invalid operation: %N (shift of type %T)", n, t);
			t = T;
		}
		n->type = t;
		return;
	default:
		if(n->left == N) {
			dump("defaultlit", n);
			fatal("defaultlit");
		}
		// n is ideal, so left and right must both be ideal.
		// n has not been computed as a constant value,
		// so either left or right must not be constant.
		// The only 'ideal' non-constant expressions are shifts.  Ugh.
		// If one of these is a shift and the other is not, use that type.
		// When compiling x := 1<<i + 3.14, this means we try to push
		// the float64 down into the 1<<i, producing the correct error
		// (cannot shift float64).
		if(t == T && (n->right->op == OLSH || n->right->op == ORSH)) {
			defaultlit(&n->left, T);
			defaultlit(&n->right, n->left->type);
		} else if(t == T && (n->left->op == OLSH || n->left->op == ORSH)) {
			defaultlit(&n->right, T);
			defaultlit(&n->left, n->right->type);
		} else {
			defaultlit(&n->left, t);
			defaultlit(&n->right, t);
		}
		if(n->type == idealbool || n->type == idealstring)
			n->type = types[n->type->etype];
		else
			n->type = n->left->type;
		return;
	}

	lno = setlineno(n);
	switch(n->val.ctype) {
	default:
		if(t != T) {
			convlit(np, t);
			break;
		}
		if(n->val.ctype == CTNIL) {
			lineno = lno;
			yyerror("use of untyped nil");
			n->type = T;
			break;
		}
		if(n->val.ctype == CTSTR) {
			n->type = types[TSTRING];
			break;
		}
		yyerror("defaultlit: unknown literal: %N", n);
		break;
	case CTBOOL:
		n->type = types[TBOOL];
		if(t != T && t->etype == TBOOL)
			n->type = t;
		break;
	case CTINT:
		n->type = types[TINT];
		goto num;
	case CTFLT:
		n->type = types[TFLOAT64];
		goto num;
	case CTCPLX:
		n->type = types[TCOMPLEX128];
		goto num;
	num:
		if(t != T) {
			if(isint[t->etype]) {
				n->type = t;
				n->val = toint(n->val);
			}
			else
			if(isfloat[t->etype]) {
				n->type = t;
				n->val = toflt(n->val);
			}
			else
			if(iscomplex[t->etype]) {
				n->type = t;
				n->val = tocplx(n->val);
			}
		}
		overflow(n->val, n->type);
		break;
	}
	lineno = lno;
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
	if(isconst(l, CTCPLX) || isconst(r, CTCPLX)) {
		convlit(lp, types[TCOMPLEX128]);
		convlit(rp, types[TCOMPLEX128]);
		return;
	}
	if(isconst(l, CTFLT) || isconst(r, CTFLT)) {
		convlit(lp, types[TFLOAT64]);
		convlit(rp, types[TFLOAT64]);
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
	if(n->op == OLITERAL && n->val.ctype == CTINT && n->type != T)
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
