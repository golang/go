// Inferno utils/cc/funct.c
// http://code.google.com/p/inferno-os/source/browse/utils/cc/funct.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include	<u.h>
#include	"cc.h"

typedef	struct	Ftab	Ftab;
struct	Ftab
{
	char	op;
	char*	name;
	char	typ;
};
typedef	struct	Gtab	Gtab;
struct	Gtab
{
	char	etype;
	char*	name;
};

Ftab	ftabinit[OEND];
Gtab	gtabinit[NALLTYPES];

int
isfunct(Node *n)
{
	Type *t, *t1;
	Funct *f;
	Node *l;
	Sym *s;
	int o;

	o = n->op;
	if(n->left == Z)
		goto no;
	t = n->left->type;
	if(t == T)
		goto no;
	f = t->funct;

	switch(o) {
	case OAS:	// put cast on rhs
	case OASI:
	case OASADD:
	case OASAND:
	case OASASHL:
	case OASASHR:
	case OASDIV:
	case OASLDIV:
	case OASLMOD:
	case OASLMUL:
	case OASLSHR:
	case OASMOD:
	case OASMUL:
	case OASOR:
	case OASSUB:
	case OASXOR:
		if(n->right == Z)
			goto no;
		t1 = n->right->type;
		if(t1 == T)
			goto no;
		if(t1->funct == f)
			break;

		l = new(OXXX, Z, Z);
		*l = *n->right;

		n->right->left = l;
		n->right->right = Z;
		n->right->type = t;
		n->right->op = OCAST;

		if(!isfunct(n->right))
			prtree(n, "isfunc !");
		break;

	case OCAST:	// t f(T) or T f(t)
		t1 = n->type;
		if(t1 == T)
			goto no;
		if(f != nil) {
			s = f->castfr[t1->etype];
			if(s == S)
				goto no;
			n->right = n->left;
			goto build;
		}
		f = t1->funct;
		if(f != nil) {
			s = f->castto[t->etype];
			if(s == S)
				goto no;
			n->right = n->left;
			goto build;
		}
		goto no;
	}

	if(f == nil)
		goto no;
	s = f->sym[o];
	if(s == S)
		goto no;

	/*
	 * the answer is yes,
	 * now we rewrite the node
	 * and give diagnostics
	 */
	switch(o) {
	default:
		diag(n, "isfunct op missing %O\n", o);
		goto bad;

	case OADD:	// T f(T, T)
	case OAND:
	case OASHL:
	case OASHR:
	case ODIV:
	case OLDIV:
	case OLMOD:
	case OLMUL:
	case OLSHR:
	case OMOD:
	case OMUL:
	case OOR:
	case OSUB:
	case OXOR:

	case OEQ:	// int f(T, T)
	case OGE:
	case OGT:
	case OHI:
	case OHS:
	case OLE:
	case OLO:
	case OLS:
	case OLT:
	case ONE:
		if(n->right == Z)
			goto bad;
		t1 = n->right->type;
		if(t1 == T)
			goto bad;
		if(t1->funct != f)
			goto bad;
		n->right = new(OLIST, n->left, n->right);
		break;

	case OAS:	// structure copies done by the compiler
	case OASI:
		goto no;

	case OASADD:	// T f(T*, T)
	case OASAND:
	case OASASHL:
	case OASASHR:
	case OASDIV:
	case OASLDIV:
	case OASLMOD:
	case OASLMUL:
	case OASLSHR:
	case OASMOD:
	case OASMUL:
	case OASOR:
	case OASSUB:
	case OASXOR:
		if(n->right == Z)
			goto bad;
		t1 = n->right->type;
		if(t1 == T)
			goto bad;
		if(t1->funct != f)
			goto bad;
		n->right = new(OLIST, new(OADDR, n->left, Z), n->right);
		break;

	case OPOS:	// T f(T)
	case ONEG:
	case ONOT:
	case OCOM:
		n->right = n->left;
		break;


	}

build:
	l = new(ONAME, Z, Z);
	l->sym = s;
	l->type = s->type;
	l->etype = s->type->etype;
	l->xoffset = s->offset;
	l->class = s->class;
	tcomo(l, 0);

	n->op = OFUNC;
	n->left = l;
	n->type = l->type->link;
	if(tcompat(n, T, l->type, tfunct))
		goto bad;
	if(tcoma(n->left, n->right, l->type->down, 1))
		goto bad;
	return 1;

no:
	return 0;

bad:
	diag(n, "can't rewrite typestr for op %O\n", o);
	prtree(n, "isfunct");
	n->type = T;
	return 1;
}

void
dclfunct(Type *t, Sym *s)
{
	Funct *f;
	Node *n;
	Type *f1, *f2, *f3, *f4;
	int o, i, c;
	char str[100];

	if(t->funct)
		return;

	// recognize generated tag of dorm _%d_
	if(t->tag == S)
		goto bad;
	for(i=0; c = t->tag->name[i]; i++) {
		if(c == '_') {
			if(i == 0 || t->tag->name[i+1] == 0)
				continue;
			break;
		}
		if(c < '0' || c > '9')
			break;
	}
	if(c == 0)
		goto bad;

	f = alloc(sizeof(*f));
	for(o=0; o<nelem(f->sym); o++)
		f->sym[o] = S;

	t->funct = f;

	f1 = typ(TFUNC, t);
	f1->down = copytyp(t);
	f1->down->down = t;

	f2 = typ(TFUNC, types[TINT]);
	f2->down = copytyp(t);
	f2->down->down = t;

	f3 = typ(TFUNC, t);
	f3->down = typ(TIND, t);
	f3->down->down = t;

	f4 = typ(TFUNC, t);
	f4->down = t;

	for(i=0;; i++) {
		o = ftabinit[i].op;
		if(o == OXXX)
			break;
		sprint(str, "%s_%s_", t->tag->name, ftabinit[i].name);
		n = new(ONAME, Z, Z);
		n->sym = slookup(str);
		f->sym[o] = n->sym;
		switch(ftabinit[i].typ) {
		default:
			diag(Z, "dclfunct op missing %d\n", ftabinit[i].typ);
			break;

		case 1:	// T f(T,T)	+
			dodecl(xdecl, CEXTERN, f1, n);
			break;

		case 2:	// int f(T,T)	==
			dodecl(xdecl, CEXTERN, f2, n);
			break;

		case 3:	// void f(T*,T)	+=
			dodecl(xdecl, CEXTERN, f3, n);
			break;

		case 4:	// T f(T)	~
			dodecl(xdecl, CEXTERN, f4, n);
			break;
		}
	}
	for(i=0;; i++) {
		o = gtabinit[i].etype;
		if(o == TXXX)
			break;

		/*
		 * OCAST types T1 _T2_T1_(T2)
		 */
		sprint(str, "_%s%s_", gtabinit[i].name, t->tag->name);
		n = new(ONAME, Z, Z);
		n->sym = slookup(str);
		f->castto[o] = n->sym;

		f1 = typ(TFUNC, t);
		f1->down = types[o];
		dodecl(xdecl, CEXTERN, f1, n);

		sprint(str, "%s_%s_", t->tag->name, gtabinit[i].name);
		n = new(ONAME, Z, Z);
		n->sym = slookup(str);
		f->castfr[o] = n->sym;

		f1 = typ(TFUNC, types[o]);
		f1->down = t;
		dodecl(xdecl, CEXTERN, f1, n);
	}
	return;
bad:
	diag(Z, "dclfunct bad %T %s\n", t, s->name);
}

Gtab	gtabinit[NALLTYPES] =
{
	TCHAR,		"c",
	TUCHAR,		"uc",
	TSHORT,		"h",
	TUSHORT,	"uh",
	TINT,		"i",
	TUINT,		"ui",
	TLONG,		"l",
	TULONG,		"ul",
	TVLONG,		"v",
	TUVLONG,	"uv",
	TFLOAT,		"f",
	TDOUBLE,	"d",
	TXXX
};

Ftab	ftabinit[OEND] =
{
	OADD,		"add",		1,
	OAND,		"and",		1,
	OASHL,		"ashl",		1,
	OASHR,		"ashr",		1,
	ODIV,		"div",		1,
	OLDIV,		"ldiv",		1,
	OLMOD,		"lmod",		1,
	OLMUL,		"lmul",		1,
	OLSHR,		"lshr",		1,
	OMOD,		"mod",		1,
	OMUL,		"mul",		1,
	OOR,		"or",		1,
	OSUB,		"sub",		1,
	OXOR,		"xor",		1,

	OEQ,		"eq",		2,
	OGE,		"ge",		2,
	OGT,		"gt",		2,
	OHI,		"hi",		2,
	OHS,		"hs",		2,
	OLE,		"le",		2,
	OLO,		"lo",		2,
	OLS,		"ls",		2,
	OLT,		"lt",		2,
	ONE,		"ne",		2,

	OASADD,		"asadd",	3,
	OASAND,		"asand",	3,
	OASASHL,	"asashl",	3,
	OASASHR,	"asashr",	3,
	OASDIV,		"asdiv",	3,
	OASLDIV,	"asldiv",	3,
	OASLMOD,	"aslmod",	3,
	OASLMUL,	"aslmul",	3,
	OASLSHR,	"aslshr",	3,
	OASMOD,		"asmod",	3,
	OASMUL,		"asmul",	3,
	OASOR,		"asor",		3,
	OASSUB,		"assub",	3,
	OASXOR,		"asxor",	3,

	OPOS,		"pos",		4,
	ONEG,		"neg",		4,
	OCOM,		"com",		4,
	ONOT,		"not",		4,

//	OPOSTDEC,
//	OPOSTINC,
//	OPREDEC,
//	OPREINC,

	OXXX,
};

//	Node*	nodtestv;

//	Node*	nodvpp;
//	Node*	nodppv;
//	Node*	nodvmm;
//	Node*	nodmmv;
