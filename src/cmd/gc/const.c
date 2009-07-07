// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"
#define	TUP(x,y)	(((x)<<16)|(y))

static Val toflt(Val);
static Val toint(Val);
static Val tostr(Val);
static void overflow(Val, Type*);
static Val copyval(Val);

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
convlit(Node *n, Type *t)
{
	convlit1(n, t, 0);
}

/*
 * convert n, if literal, to type t.
 */
void
convlit1(Node *n, Type *t, int explicit)
{
	int et, ct;

	if(n == N || t == T || n->type == T)
		return;
	et = t->etype;
	if(et == TIDEAL || et == TNIL)
		return;

	switch(n->op) {
	default:
		return;
	case OLITERAL:
		break;
	case OLSH:
	case ORSH:
		convlit(n->left, t);
		n->type = n->left->type;
		return;
	}
	// avoided repeated calculations, errors
	if(cvttype(n->type, t) == 1) {
		n->type = t;
		return;
	}

	ct = consttype(n);
	if(ct < 0)
		goto bad;

	if(et == TINTER) {
		if(ct == CTNIL && n->type == types[TNIL]) {
			n->type = t;
			return;
		}
		defaultlit(n, T);
		return;
	}

	// if already has non-ideal type, cannot change implicitly
	if(!explicit) {
		switch(n->type->etype) {
		case TIDEAL:
		case TNIL:
			break;
		case TSTRING:
			if(n->type == idealstring)
				break;
			// fall through
		default:
			goto bad;
		}
	}

	switch(ct) {
	default:
		goto bad;

	case CTNIL:
		switch(et) {
		default:
			goto bad;

		case TSTRING:
			// let normal conversion code handle it
			return;

		case TPTR32:
		case TPTR64:
		case TINTER:
		case TARRAY:
		case TMAP:
		case TCHAN:
		case TFUNC:
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
		ct = n->val.ctype;
		if(isint[et]) {
			if(ct == CTFLT)
				n->val = toint(n->val);
			else if(ct != CTINT)
				goto bad;
			overflow(n->val, t);
		} else if(isfloat[et]) {
			if(ct == CTINT)
				n->val = toflt(n->val);
			else if(ct != CTFLT)
				goto bad;
			overflow(n->val, t);
			n->val.u.fval = truncfltlit(n->val.u.fval, t);
		} else if(et == TSTRING && ct == CTINT && explicit)
			n->val = tostr(n->val);
		else
			goto bad;
	}
	n->type = t;
	return;

bad:
	if(n->type->etype == TIDEAL)
		defaultlit(n, T);
	yyerror("cannot convert %T constant to %T", n->type, t);
	n->diag = 1;
	return;
}

static Val
copyval(Val v)
{
	Mpint *i;
	Mpflt *f;

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
	}
	return v;
}

static Val
toflt(Val v)
{
	Mpflt *f;

	if(v.ctype == CTINT) {
		f = mal(sizeof(*f));
		mpmovefixflt(f, v.u.xval);
		v.ctype = CTFLT;
		v.u.fval = f;
	}
	return v;
}

static Val
toint(Val v)
{
	Mpint *i;

	if(v.ctype == CTFLT) {
		i = mal(sizeof(*i));
		if(mpmovefltfix(i, v.u.fval) < 0)
			yyerror("constant %#F truncated to integer", v.u.fval);
		v.ctype = CTINT;
		v.u.xval = i;
	}
	return v;
}

static void
overflow(Val v, Type *t)
{
	// v has already been converted
	// to appropriate form for t.
	if(t == T || t->etype == TIDEAL)
		return;
	switch(v.ctype) {
	case CTINT:
		if(mpcmpfixfix(v.u.xval, minintval[t->etype]) < 0
		|| mpcmpfixfix(v.u.xval, maxintval[t->etype]) > 0)
			yyerror("constant %B overflows %T", v.u.xval, t);
		break;
	case CTFLT:
		if(mpcmpfltflt(v.u.fval, minfltval[t->etype]) < 0
		|| mpcmpfltflt(v.u.fval, maxfltval[t->etype]) > 0)
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
		if(mpcmpfixfix(v.u.xval, minintval[TINT]) < 0
		|| mpcmpfixfix(v.u.xval, maxintval[TINT]) > 0)
			yyerror("overflow in int -> string");
		rune = mpgetfix(v.u.xval);
		l = runelen(rune);
		s = mal(sizeof(*s)+l);
		s->len = l;
		runetochar((char*)s->s, &rune);
		v.ctype = CTSTR;
		v.u.sval = s;
		break;

	case CTFLT:
		yyerror("no float -> string");
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
	Node *nl, *nr;
	int32 len;
	Strlit *str;
	int wl, wr, lno, et;
	Val v;
	Mpint b;

	nl = n->left;
	if(nl == N || nl->type == T)
		return;
	if(consttype(nl) < 0)
		return;
	wl = nl->type->etype;
	if(isint[wl] || isfloat[wl])
		wl = TIDEAL;

	nr = n->right;
	if(nr == N)
		goto unary;
	if(nr->type == T)
		return;
	if(consttype(nr) < 0)
		return;
	wr = nr->type->etype;
	if(isint[wr] || isfloat[wr])
		wr = TIDEAL;

	// check for compatible general types (numeric, string, etc)
	if(wl != wr)
		goto illegal;

	// check for compatible types.
	switch(n->op) {
	default:
		// ideal const mixes with anything but otherwise must match.
		if(nl->type->etype != TIDEAL)
			defaultlit(nr, nl->type);
		if(nr->type->etype != TIDEAL)
			defaultlit(nl, nr->type);
		if(nl->type->etype != nr->type->etype)
			goto illegal;
		break;

	case OLSH:
	case ORSH:
		// right must be unsigned.
		// left can be ideal.
		defaultlit(nr, types[TUINT]);
		break;
	}

	// copy numeric value to avoid modifying
	// n->left, in case someone still refers to it (e.g. iota).
	v = nl->val;
	if(wl == TIDEAL)
		v = copyval(v);

	// since wl == wr,
	// the only way v.ctype != nr->val.ctype
	// is when one is CTINT and the other CTFLT.
	// make both CTFLT.
	if(v.ctype != nr->val.ctype) {
		v = toflt(v);
		nr->val = toflt(nr->val);
	}

	// run op
	switch(TUP(n->op, v.ctype)) {
	default:
	illegal:
		yyerror("illegal constant expression %T %O %T",
			nl->type, n->op, nr->type);
		n->diag = 1;
		return;

	case TUP(OADD, CTINT):
		mpaddfixfix(v.u.xval, nr->val.u.xval);
		break;
	case TUP(OSUB, CTINT):
		mpsubfixfix(v.u.xval, nr->val.u.xval);
		break;
	case TUP(OMUL, CTINT):
		mpmulfixfix(v.u.xval, nr->val.u.xval);
		break;
	case TUP(ODIV, CTINT):
		if(mpcmpfixc(nr->val.u.xval, 0) == 0) {
			yyerror("division by zero");
			mpmovecfix(v.u.xval, 1);
			break;
		}
		mpdivfixfix(v.u.xval, nr->val.u.xval);
		break;
	case TUP(OMOD, CTINT):
		if(mpcmpfixc(nr->val.u.xval, 0) == 0) {
			yyerror("division by zero");
			mpmovecfix(v.u.xval, 1);
			break;
		}
		mpmodfixfix(v.u.xval, nr->val.u.xval);
		break;

	case TUP(OLSH, CTINT):
		mplshfixfix(v.u.xval, nr->val.u.xval);
		break;
	case TUP(ORSH, CTINT):
		mprshfixfix(v.u.xval, nr->val.u.xval);
		break;
	case TUP(OOR, CTINT):
		mporfixfix(v.u.xval, nr->val.u.xval);
		break;
	case TUP(OAND, CTINT):
		mpandfixfix(v.u.xval, nr->val.u.xval);
		break;
	case TUP(OANDNOT, CTINT):
		mpandnotfixfix(v.u.xval, nr->val.u.xval);
		break;
	case TUP(OXOR, CTINT):
		mpxorfixfix(v.u.xval, nr->val.u.xval);
		break;
	case TUP(OADD, CTFLT):
		mpaddfltflt(v.u.fval, nr->val.u.fval);
		break;
	case TUP(OSUB, CTFLT):
		mpsubfltflt(v.u.fval, nr->val.u.fval);
		break;
	case TUP(OMUL, CTFLT):
		mpmulfltflt(v.u.fval, nr->val.u.fval);
		break;
	case TUP(ODIV, CTFLT):
		if(mpcmpfltc(nr->val.u.fval, 0) == 0) {
			yyerror("division by zero");
			mpmovecflt(v.u.fval, 1.0);
			break;
		}
		mpdivfltflt(v.u.fval, nr->val.u.fval);
		break;

	case TUP(OEQ, CTNIL):
		goto settrue;
	case TUP(ONE, CTNIL):
		goto setfalse;

	case TUP(OEQ, CTINT):
		if(mpcmpfixfix(v.u.xval, nr->val.u.xval) == 0)
			goto settrue;
		goto setfalse;
	case TUP(ONE, CTINT):
		if(mpcmpfixfix(v.u.xval, nr->val.u.xval) != 0)
			goto settrue;
		goto setfalse;
	case TUP(OLT, CTINT):
		if(mpcmpfixfix(v.u.xval, nr->val.u.xval) < 0)
			goto settrue;
		goto setfalse;
	case TUP(OLE, CTINT):
		if(mpcmpfixfix(v.u.xval, nr->val.u.xval) <= 0)
			goto settrue;
		goto setfalse;
	case TUP(OGE, CTINT):
		if(mpcmpfixfix(v.u.xval, nr->val.u.xval) >= 0)
			goto settrue;
		goto setfalse;
	case TUP(OGT, CTINT):
		if(mpcmpfixfix(v.u.xval, nr->val.u.xval) > 0)
			goto settrue;
		goto setfalse;

	case TUP(OEQ, CTFLT):
		if(mpcmpfltflt(v.u.fval, nr->val.u.fval) == 0)
			goto settrue;
		goto setfalse;
	case TUP(ONE, CTFLT):
		if(mpcmpfltflt(v.u.fval, nr->val.u.fval) != 0)
			goto settrue;
		goto setfalse;
	case TUP(OLT, CTFLT):
		if(mpcmpfltflt(v.u.fval, nr->val.u.fval) < 0)
			goto settrue;
		goto setfalse;
	case TUP(OLE, CTFLT):
		if(mpcmpfltflt(v.u.fval, nr->val.u.fval) <= 0)
			goto settrue;
		goto setfalse;
	case TUP(OGE, CTFLT):
		if(mpcmpfltflt(v.u.fval, nr->val.u.fval) >= 0)
			goto settrue;
		goto setfalse;
	case TUP(OGT, CTFLT):
		if(mpcmpfltflt(v.u.fval, nr->val.u.fval) > 0)
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
	case TUP(OADD, CTSTR):
		len = v.u.sval->len + nr->val.u.sval->len;
		str = mal(sizeof(*str) + len);
		str->len = len;
		memcpy(str->s, v.u.sval->s, v.u.sval->len);
		memcpy(str->s+v.u.sval->len, nr->val.u.sval->s, nr->val.u.sval->len);
		str->len = len;
		v.u.sval = str;
		break;

	case TUP(OOROR, CTBOOL):
		if(v.u.bval || nr->val.u.bval)
			goto settrue;
		goto setfalse;
	case TUP(OANDAND, CTBOOL):
		if(v.u.bval && nr->val.u.bval)
			goto settrue;
		goto setfalse;
	case TUP(OEQ, CTBOOL):
		if(v.u.bval == nr->val.u.bval)
			goto settrue;
		goto setfalse;
	case TUP(ONE, CTBOOL):
		if(v.u.bval != nr->val.u.bval)
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
		yyerror("illegal constant expression %O %T", n->op, nl->type);
		return;

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

	case TUP(ONOT, CTBOOL):
		if(!v.u.bval)
			goto settrue;
		goto setfalse;
	}

ret:
	// rewrite n in place.
	*n = *nl;
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
		n->type = types[TSTRING];
		break;
	case CTBOOL:
		n->type = types[TBOOL];
		break;
	case CTINT:
	case CTFLT:
		n->type = types[TIDEAL];
		break;
	case CTNIL:
		n->type = types[TNIL];
		break;
	}
	return n;
}

void
defaultlit(Node *n, Type *t)
{
	int lno;

	if(n == N)
		return;
	if(n->type == T || n->type->etype != TIDEAL)
		return;

	switch(n->op) {
	case OLITERAL:
		break;
	case OLSH:
	case ORSH:
		defaultlit(n->left, t);
		n->type = n->left->type;
		return;
	}

	lno = lineno;
	lineno = n->lineno;
	switch(n->val.ctype) {
	default:
		yyerror("defaultlit: unknown literal: %N", n);
		break;
	case CTINT:
		n->type = types[TINT];
		if(t != T) {
			if(isint[t->etype])
				n->type = t;
			else if(isfloat[t->etype]) {
				n->type = t;
				n->val = toflt(n->val);
			}
		}
		overflow(n->val, n->type);
		break;
	case CTFLT:
		n->type = types[TFLOAT];
		if(t != T) {
			if(isfloat[t->etype])
				n->type = t;
			else if(isint[t->etype]) {
				n->type = t;
				n->val = toint(n->val);
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
 */
void
defaultlit2(Node *l, Node *r)
{
	if(l->type == T || r->type == T)
		return;
	if(l->type->etype != TIDEAL && l->type->etype != TNIL) {
		convlit(r, l->type);
		return;
	}
	if(r->type->etype != TIDEAL && r->type->etype != TNIL) {
		convlit(l, r->type);
		return;
	}
	if(isconst(l, CTFLT) || isconst(r, CTFLT)) {
		convlit(l, types[TFLOAT]);
		convlit(r, types[TFLOAT]);
		return;
	}
	convlit(l, types[TINT]);
	convlit(r, types[TINT]);
}

int
cmpslit(Node *l, Node *r)
{
	int32 l1, l2, i, m;
	char *s1, *s2;

	l1 = l->val.u.sval->len;
	l2 = r->val.u.sval->len;
	s1 = l->val.u.sval->s;
	s2 = r->val.u.sval->s;

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
	if(n->op == OLITERAL)
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
	}
	return 0;
}

long
nonnegconst(Node *n)
{
	if(n->op == OLITERAL)
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
int64
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
		if(con->val.ctype == CTINT) {
			con->val.ctype = CTFLT;
			con->val.u.fval = mal(sizeof *con->val.u.fval);
			mpmovefixflt(con->val.u.fval, val->u.xval);
		}
		if(con->val.ctype != CTFLT)
			fatal("convconst ctype=%d %T", con->val.ctype, t);
		if(!isfloat[tt]) {
			// easy to handle, but can it happen?
			fatal("convconst CTINT %T", t);
		}
		if(tt == TFLOAT32)
			con->val.u.fval = truncfltlit(con->val.u.fval, t);
		return;
	}

	fatal("convconst %lT constant", t);

}
