// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"
#define	TUP(x,y)	(((x)<<16)|(y))

void
truncfltlit(Mpflt *fv, Type *t)
{
	double d;
	float f;

	if(t == T)
		return;

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
}

void
convlit1(Node *n, Type *t, int conv)
{
	int et, wt;

	if(n == N || t == T)
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

	et = t->etype;
	wt = whatis(n);

	switch(wt) {
	default:
		goto bad1;

	case Wlitnil:
		switch(et) {
		default:
			goto bad1;

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

	case Wlitstr:
		if(isnilinter(t)) {
			defaultlit(n);
			return;
		}
		if(et == TSTRING)
			break;
		goto bad1;

	case Wlitbool:
		if(isnilinter(t)) {
			defaultlit(n);
			return;
		}
		if(et == TBOOL)
			break;
		goto bad1;

	case Wlitint:
		if(isnilinter(t)) {
			defaultlit(n);
			return;
		}
		if(isint[et]) {
			// int to int
			if(mpcmpfixfix(n->val.u.xval, minintval[et]) < 0)
				goto bad2;
			if(mpcmpfixfix(n->val.u.xval, maxintval[et]) > 0)
				goto bad2;
			break;
		}
		if(isfloat[et]) {
			// int to float
			Mpint *xv;
			Mpflt *fv;

			xv = n->val.u.xval;
			if(mpcmpfixflt(xv, minfltval[et]) < 0)
				goto bad2;
			if(mpcmpfixflt(xv, maxfltval[et]) > 0)
				goto bad2;
			fv = mal(sizeof(*n->val.u.fval));
			n->val.u.fval = fv;
			mpmovefixflt(fv, xv);
			n->val.ctype = CTFLT;
			truncfltlit(fv, t);
			break;
		}
		if(!conv)
			goto bad1;

		// only done as string(CONST)
		if(et == TSTRING) {
			Rune rune;
			int l;
			String *s;

			rune = mpgetfix(n->val.u.xval);
			l = runelen(rune);
			s = mal(sizeof(*s)+l);
			s->len = l;
			runetochar((char*)(s->s), &rune);

			n->val.u.sval = s;
			n->val.ctype = CTSTR;
			break;
		}
		goto bad1;

	case Wlitfloat:
		if(isnilinter(t)) {
			defaultlit(n);
			return;
		}
		if(isint[et]) {
			// float to int
			Mpflt *fv;

			fv = n->val.u.fval;
			if(mpcmpfltfix(fv, minintval[et]) < 0)
				goto bad2;
			if(mpcmpfltfix(fv, maxintval[et]) > 0)
				goto bad2;
			if(floor(mpgetflt(fv)) != mpgetflt(fv))
				goto bad3;
			n->val.u.xval = mal(sizeof(*n->val.u.xval));
			mpmovefltfix(n->val.u.xval, fv);
			n->val.ctype = CTINT;
			break;
		}
		if(isfloat[et]) {
			// float to float
			Mpflt *fv;

			fv = n->val.u.fval;
			if(mpcmpfltflt(fv, minfltval[et]) < 0)
				goto bad2;
			if(mpcmpfltflt(fv, maxfltval[et]) > 0)
				goto bad2;
			truncfltlit(fv, t);
			break;
		}
		goto bad1;
	}
	n->type = t;

	return;

bad1:
	yyerror("illegal conversion of %W to %T", wt, t);
	return;

bad2:
	yyerror("overflow converting constant to %T", t);
	return;

bad3:
	yyerror("cannot convert non-integer constant to %T", t);
	return;
}

void
convlit(Node *n, Type *t)
{
	convlit1(n, t, 0);
}

void
evconst(Node *n)
{
	Node *nl, *nr;
	int32 len;
	String *str;
	int wl, wr;
	Mpint *xval;
	Mpflt *fval;

	xval = nil;
	fval = nil;

	nl = n->left;
	if(nl == N)
		return;

	wl = whatis(nl);
	switch(wl) {
	default:
		return;

	case Wlitint:
	case Wlitfloat:
	case Wlitbool:
	case Wlitstr:
	case Wlitnil:
		break;
	}

	nr = n->right;
	if(nr == N)
		goto unary;

	wr = whatis(nr);
	switch(wr) {
	default:
		return;

	case Wlitint:
	case Wlitfloat:
	case Wlitbool:
	case Wlitstr:
	case Wlitnil:
		break;
	}

	if(wl != wr) {
		if(wl == Wlitfloat && wr == Wlitint) {
			xval = nr->val.u.xval;
			nr->val.u.fval = mal(sizeof(*nr->val.u.fval));
			mpmovefixflt(nr->val.u.fval, xval);
			nr->val.ctype = CTFLT;
			wr = whatis(nr);
		} else
		if(wl == Wlitint && wr == Wlitfloat) {
			xval = nl->val.u.xval;
			nl->val.u.fval = mal(sizeof(*nl->val.u.fval));
			mpmovefixflt(nl->val.u.fval, xval);
			nl->val.ctype = CTFLT;
			wl = whatis(nl);
		} else {
			yyerror("illegal combination of literals %O %W, %W", n->op, wl, wr);
			return;
		}
	}

	// dance to not modify left side
	// this is because iota will reuse it
	if(wl == Wlitint) {
		xval = mal(sizeof(*xval));
		mpmovefixfix(xval, nl->val.u.xval);
	} else
	if(wl == Wlitfloat) {
		fval = mal(sizeof(*fval));
		mpmovefltflt(fval, nl->val.u.fval);
	}

	switch(TUP(n->op, wl)) {
	default:
		yyerror("illegal literal %O %W", n->op, wl);
		return;

	case TUP(OADD, Wlitint):
		mpaddfixfix(xval, nr->val.u.xval);
		break;
	case TUP(OSUB, Wlitint):
		mpsubfixfix(xval, nr->val.u.xval);
		break;
	case TUP(OMUL, Wlitint):
		mpmulfixfix(xval, nr->val.u.xval);
		break;
	case TUP(ODIV, Wlitint):
		mpdivfixfix(xval, nr->val.u.xval);
		break;
	case TUP(OMOD, Wlitint):
		mpmodfixfix(xval, nr->val.u.xval);
		break;

	case TUP(OLSH, Wlitint):
		mplshfixfix(xval, nr->val.u.xval);
		break;
	case TUP(ORSH, Wlitint):
		mprshfixfix(xval, nr->val.u.xval);
		break;
	case TUP(OOR, Wlitint):
		mporfixfix(xval, nr->val.u.xval);
		break;
	case TUP(OAND, Wlitint):
		mpandfixfix(xval, nr->val.u.xval);
		break;
	case TUP(OANDNOT, Wlitint):
		mpandnotfixfix(xval, nr->val.u.xval);
		break;
	case TUP(OXOR, Wlitint):
		mpxorfixfix(xval, nr->val.u.xval);
		break;

	case TUP(OADD, Wlitfloat):
		mpaddfltflt(fval, nr->val.u.fval);
		break;
	case TUP(OSUB, Wlitfloat):
		mpsubfltflt(fval, nr->val.u.fval);
		break;
	case TUP(OMUL, Wlitfloat):
		mpmulfltflt(fval, nr->val.u.fval);
		break;
	case TUP(ODIV, Wlitfloat):
		mpdivfltflt(fval, nr->val.u.fval);
		break;

	case TUP(OEQ, Wlitnil):
		goto settrue;
	case TUP(ONE, Wlitnil):
		goto setfalse;

	case TUP(OEQ, Wlitint):
		if(mpcmpfixfix(xval, nr->val.u.xval) == 0)
			goto settrue;
		goto setfalse;
	case TUP(ONE, Wlitint):
		if(mpcmpfixfix(xval, nr->val.u.xval) != 0)
			goto settrue;
		goto setfalse;
	case TUP(OLT, Wlitint):
		if(mpcmpfixfix(xval, nr->val.u.xval) < 0)
			goto settrue;
		goto setfalse;
	case TUP(OLE, Wlitint):
		if(mpcmpfixfix(xval, nr->val.u.xval) <= 0)
			goto settrue;
		goto setfalse;
	case TUP(OGE, Wlitint):
		if(mpcmpfixfix(xval, nr->val.u.xval) >= 0)
			goto settrue;
		goto setfalse;
	case TUP(OGT, Wlitint):
		if(mpcmpfixfix(xval, nr->val.u.xval) > 0)
			goto settrue;
		goto setfalse;

	case TUP(OEQ, Wlitfloat):
		if(mpcmpfltflt(fval, nr->val.u.fval) == 0)
			goto settrue;
		goto setfalse;
	case TUP(ONE, Wlitfloat):
		if(mpcmpfltflt(fval, nr->val.u.fval) != 0)
			goto settrue;
		goto setfalse;
	case TUP(OLT, Wlitfloat):
		if(mpcmpfltflt(fval, nr->val.u.fval) < 0)
			goto settrue;
		goto setfalse;
	case TUP(OLE, Wlitfloat):
		if(mpcmpfltflt(fval, nr->val.u.fval) <= 0)
			goto settrue;
		goto setfalse;
	case TUP(OGE, Wlitfloat):
		if(mpcmpfltflt(fval, nr->val.u.fval) >= 0)
			goto settrue;
		goto setfalse;
	case TUP(OGT, Wlitfloat):
		if(mpcmpfltflt(fval, nr->val.u.fval) > 0)
			goto settrue;
		goto setfalse;

	case TUP(OEQ, Wlitstr):
		if(cmpslit(nl, nr) == 0)
			goto settrue;
		goto setfalse;
	case TUP(ONE, Wlitstr):
		if(cmpslit(nl, nr) != 0)
			goto settrue;
		goto setfalse;
	case TUP(OLT, Wlitstr):
		if(cmpslit(nl, nr) < 0)
			goto settrue;
		goto setfalse;
	case TUP(OLE, Wlitstr):
		if(cmpslit(nl, nr) <= 0)
			goto settrue;
		goto setfalse;
	case TUP(OGE, Wlitstr):
		if(cmpslit(nl, nr) >= 0l)
			goto settrue;
		goto setfalse;
	case TUP(OGT, Wlitstr):
		if(cmpslit(nl, nr) > 0)
			goto settrue;
		goto setfalse;
	case TUP(OADD, Wlitstr):
		len = nl->val.u.sval->len + nr->val.u.sval->len;
		str = mal(sizeof(*str) + len);
		str->len = len;
		memcpy(str->s, nl->val.u.sval->s, nl->val.u.sval->len);
		memcpy(str->s+nl->val.u.sval->len, nr->val.u.sval->s, nr->val.u.sval->len);
		str->len = len;
		nl->val.u.sval = str;
		break;

	case TUP(OOROR, Wlitbool):
		if(nl->val.u.bval || nr->val.u.bval)
			goto settrue;
		goto setfalse;
	case TUP(OANDAND, Wlitbool):
		if(nl->val.u.bval && nr->val.u.bval)
			goto settrue;
		goto setfalse;
	}
	goto ret;

settrue:
	*n = *booltrue;
	return;

setfalse:
	*n = *boolfalse;
	return;

unary:
	if(wl == Wlitint) {
		xval = mal(sizeof(*xval));
		mpmovefixfix(xval, nl->val.u.xval);
	} else
	if(wl == Wlitfloat) {
		fval = mal(sizeof(*fval));
		mpmovefltflt(fval, nl->val.u.fval);
	}

	switch(TUP(n->op, wl)) {
	default:
		yyerror("illegal combination of literals %O %d", n->op, wl);
		return;

	case TUP(OPLUS, Wlitint):
		break;
	case TUP(OMINUS, Wlitint):
		mpnegfix(xval);
		break;
	case TUP(OCOM, Wlitint):
		mpcomfix(xval);
		break;

	case TUP(OPLUS, Wlitfloat):
		break;
	case TUP(OMINUS, Wlitfloat):
		mpnegflt(fval);
		break;

	case TUP(ONOT, Wlitbool):
		if(nl->val.u.bval)
			goto settrue;
		goto setfalse;
	}

ret:
	*n = *nl;

	// second half of dance
	if(wl == Wlitint) {
		n->val.u.xval = xval;
	} else
	if(wl == Wlitfloat) {
		n->val.u.fval = fval;
		truncfltlit(fval, n->type);
	}
}

void
defaultlit(Node *n)
{
	if(n == N)
		return;
	if(n->type != T)
		return;
	if(n->op != OLITERAL)
		return;

	switch(n->val.ctype) {
	default:
		yyerror("defaultlit: unknown literal: %N", n);
		break;
	case CTINT:
	case CTSINT:
	case CTUINT:
		n->type = types[TINT];
		break;
	case CTFLT:
		n->type = types[TFLOAT];
		break;
	case CTBOOL:
		n->type = types[TBOOL];
		break;
	case CTSTR:
		n->type = types[TSTRING];
		break;
	}
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
