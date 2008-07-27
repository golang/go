// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"
#define	TUP(x,y)	(((x)<<16)|(y))

void
convlit(Node *n, Type *t)
{
	int et;
	Node *n1;

	if(n == N || n->op != OLITERAL || t == T)
		return;

	et = t->etype;
	switch(whatis(n)) {
	default:
		goto bad1;

	case Wlitnil:
		if(!isptr[et] && et != TINTER)
			goto bad1;
		if(isptrto(t, TSTRING)) {
			n->val.sval = mal(8);
			n->val.ctype = CTSTR;
		}
		break;

	case Wlitstr:
		if(isptrto(t, TSTRING))
			break;
		goto bad1;

	case Wlitbool:
		if(et == TBOOL)
			break;
		goto bad1;

	case Wlitint:
		if(isptrto(t, TSTRING)) {
			Rune rune;
			int l;
			String *s;

			rune = n->val.vval;
			l = runelen(rune);
			s = mal(sizeof(*s)+l);
			s->len = l;
			runetochar((char*)(s->s), &rune);

			n->val.sval = s;
			n->val.ctype = CTSTR;
			break;
		}
		if(isint[et]) {
			if(n->val.vval < minintval[et])
				goto bad2;
			if(n->val.vval > maxintval[et])
				goto bad2;
			break;
		}
		if(isfloat[et]) {
			if(n->val.vval < minfloatval[et])
				goto bad2;
			if(n->val.vval > maxfloatval[et])
				goto bad2;
			n->val.dval = n->val.vval;
			n->val.ctype = CTFLT;
			break;
		}
		goto bad1;

	case Wlitfloat:
		if(isint[et]) {
			if(n->val.dval < minintval[et])
				goto bad2;
			if(n->val.dval > maxintval[et])
				goto bad2;
			n->val.vval = n->val.dval;
			n->val.ctype = CTINT;
			break;
		}
		if(isfloat[et]) {
			if(n->val.dval < minfloatval[et])
				goto bad2;
			if(n->val.dval > maxfloatval[et])
				goto bad2;
			break;
		}
		goto bad1;
	}
	n->type = t;
	return;

bad1:
	yyerror("illegal conversion of constant to %T", t);
	return;

bad2:
	yyerror("overflow converting constant to %T", t);
	return;
}

void
evconst(Node *n)
{
	Node *nl, *nr;
	long len;
	String *str;
	int wl, wr;

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
		break;
	}

	if(wl != wr) {
		if(wl == Wlitfloat && wr == Wlitint) {
			nr->val.dval = nr->val.vval;
			nr->val.ctype = CTFLT;
			wr = whatis(nr);
		} else
		if(wl == Wlitint && wr == Wlitfloat) {
			nl->val.dval = nl->val.vval;
			nl->val.ctype = CTFLT;
			wl = whatis(nl);
		} else {
			yyerror("illegal combination of literals %d %d", nl->etype, nr->etype);
			return;
		}
	}

	switch(TUP(n->op, wl)) {
	default:
		yyerror("illegal combination of literals %O %d", n->op, wl);
		return;

	case TUP(OADD, Wlitint):
		nl->val.vval += nr->val.vval;
		break;
	case TUP(OSUB, Wlitint):
		nl->val.vval -= nr->val.vval;
		break;
	case TUP(OMUL, Wlitint):
		nl->val.vval *= nr->val.vval;
		break;
	case TUP(ODIV, Wlitint):
		nl->val.vval /= nr->val.vval;
		break;
	case TUP(OMOD, Wlitint):
		nl->val.vval %= nr->val.vval;
		break;
	case TUP(OLSH, Wlitint):
		nl->val.vval <<= nr->val.vval;
		break;
	case TUP(ORSH, Wlitint):
		nl->val.vval >>= nr->val.vval;
		break;
	case TUP(OOR, Wlitint):
		nl->val.vval |= nr->val.vval;
		break;
	case TUP(OAND, Wlitint):
		nl->val.vval &= nr->val.vval;
		break;
	case TUP(OXOR, Wlitint):
		nl->val.vval ^= nr->val.vval;
		break;

	case TUP(OADD, Wlitfloat):
		nl->val.dval += nr->val.dval;
		break;
	case TUP(OSUB, Wlitfloat):
		nl->val.dval -= nr->val.dval;
		break;
	case TUP(OMUL, Wlitfloat):
		nl->val.dval *= nr->val.dval;
		break;
	case TUP(ODIV, Wlitfloat):
		nl->val.dval /= nr->val.dval;
		break;

	case TUP(OEQ, Wlitint):
		if(nl->val.vval == nr->val.vval)
			goto settrue;
		goto setfalse;
	case TUP(ONE, Wlitint):
		if(nl->val.vval != nr->val.vval)
			goto settrue;
		goto setfalse;
	case TUP(OLT, Wlitint):
		if(nl->val.vval < nr->val.vval)
			goto settrue;
		goto setfalse;
	case TUP(OLE, Wlitint):
		if(nl->val.vval <= nr->val.vval)
			goto settrue;
		goto setfalse;
	case TUP(OGE, Wlitint):
		if(nl->val.vval >= nr->val.vval)
			goto settrue;
		goto setfalse;
	case TUP(OGT, Wlitint):
		if(nl->val.vval > nr->val.vval)
			goto settrue;
		goto setfalse;

	case TUP(OEQ, Wlitfloat):
		if(nl->val.dval == nr->val.dval)
			goto settrue;
		goto setfalse;
	case TUP(ONE, Wlitfloat):
		if(nl->val.dval != nr->val.dval)
			goto settrue;
		goto setfalse;
	case TUP(OLT, Wlitfloat):
		if(nl->val.dval < nr->val.dval)
			goto settrue;
		goto setfalse;
	case TUP(OLE, Wlitfloat):
		if(nl->val.dval <= nr->val.dval)
			goto settrue;
		goto setfalse;
	case TUP(OGE, Wlitfloat):
		if(nl->val.dval >= nr->val.dval)
			goto settrue;
		goto setfalse;
	case TUP(OGT, Wlitfloat):
		if(nl->val.dval > nr->val.dval)
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
		len = nl->val.sval->len + nr->val.sval->len;
		str = mal(sizeof(*str) + len);
		str->len = len;
		memcpy(str->s, nl->val.sval->s, nl->val.sval->len);
		memcpy(str->s+nl->val.sval->len, nr->val.sval->s, nr->val.sval->len);
		str->len = len;
		nl->val.sval = str;
		break;

	case TUP(OOROR, Wlitbool):
		if(nl->val.vval || nr->val.vval)
			goto settrue;
		goto setfalse;
	case TUP(OANDAND, Wlitbool):
		if(nl->val.vval && nr->val.vval)
			goto settrue;
		goto setfalse;
	}
	*n = *nl;
	return;

settrue:
	*n = *booltrue;
	return;

setfalse:
	*n = *boolfalse;
	return;

unary:
	switch(TUP(n->op, wl)) {
	default:
		yyerror("illegal combination of literals %O %d", n->op, wl);
		return;

	case TUP(OPLUS, Wlitint):
		nl->val.vval = +nl->val.vval;
		break;
	case TUP(OMINUS, Wlitint):
		nl->val.vval = -nl->val.vval;
		break;
	case TUP(OCOM, Wlitint):
		nl->val.vval = ~nl->val.vval;
		break;

	case TUP(OPLUS, Wlitfloat):
		nl->val.dval = +nl->val.dval;
		break;
	case TUP(OMINUS, Wlitfloat):
		nl->val.dval = -nl->val.dval;
		break;

	case TUP(ONOT, Wlitbool):
		if(nl->val.vval)
			goto settrue;
		goto setfalse;
	}
	*n = *nl;
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
		n->type = types[TINT32];
		break;
	case CTFLT:
		n->type = types[TFLOAT64];
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
	long l1, l2, i, m;
	char *s1, *s2;

	l1 = l->val.sval->len;
	l2 = r->val.sval->len;
	s1 = l->val.sval->s;
	s2 = r->val.sval->s;

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
