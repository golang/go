// Inferno utils/cc/com64.c
// http://code.google.com/p/inferno-os/source/browse/utils/cc/com64.c
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

#include <u.h>
#include "cc.h"

/*
 * this is machine depend, but it is totally
 * common on all of the 64-bit symulating machines.
 */

#define	FNX	100	/* botch -- redefinition */

Node*	nodaddv;
Node*	nodsubv;
Node*	nodmulv;
Node*	noddivv;
Node*	noddivvu;
Node*	nodmodv;
Node*	nodmodvu;
Node*	nodlshv;
Node*	nodrshav;
Node*	nodrshlv;
Node*	nodandv;
Node*	nodorv;
Node*	nodxorv;
Node*	nodnegv;
Node*	nodcomv;

Node*	nodtestv;
Node*	nodeqv;
Node*	nodnev;
Node*	nodlev;
Node*	nodltv;
Node*	nodgev;
Node*	nodgtv;
Node*	nodhiv;
Node*	nodhsv;
Node*	nodlov;
Node*	nodlsv;

Node*	nodf2v;
Node*	nodd2v;
Node*	nodp2v;
Node*	nodsi2v;
Node*	nodui2v;
Node*	nodsl2v;
Node*	nodul2v;
Node*	nodsh2v;
Node*	noduh2v;
Node*	nodsc2v;
Node*	noduc2v;

Node*	nodv2f;
Node*	nodv2d;
Node*	nodv2ui;
Node*	nodv2si;
Node*	nodv2ul;
Node*	nodv2sl;
Node*	nodv2uh;
Node*	nodv2sh;
Node*	nodv2uc;
Node*	nodv2sc;

Node*	nodvpp;
Node*	nodppv;
Node*	nodvmm;
Node*	nodmmv;

Node*	nodvasop;

char	etconv[NALLTYPES];	/* for _vasop */
Init	initetconv[] =
{
	TCHAR,		1,	0,
	TUCHAR,		2,	0,
	TSHORT,		3,	0,
	TUSHORT,	4,	0,
	TLONG,		5,	0,
	TULONG,		6,	0,
	TVLONG,		7,	0,
	TUVLONG,	8,	0,
	TINT,		9,	0,
	TUINT,		10,	0,
	-1,		0,	0,
};

Node*
fvn(char *name, int type)
{
	Node *n;

	n = new(ONAME, Z, Z);
	n->sym = slookup(name);
	n->sym->sig = SIGINTERN;
	if(fntypes[type] == 0)
		fntypes[type] = typ(TFUNC, types[type]);
	n->type = fntypes[type];
	n->etype = type;
	n->class = CGLOBL;
	n->addable = 10;
	n->complex = 0;
	return n;
}

void
com64init(void)
{
	Init *p;

	nodaddv = fvn("_addv", TVLONG);
	nodsubv = fvn("_subv", TVLONG);
	nodmulv = fvn("_mulv", TVLONG);
	noddivv = fvn("_divv", TVLONG);
	noddivvu = fvn("_divvu", TVLONG);
	nodmodv = fvn("_modv", TVLONG);
	nodmodvu = fvn("_modvu", TVLONG);
	nodlshv = fvn("_lshv", TVLONG);
	nodrshav = fvn("_rshav", TVLONG);
	nodrshlv = fvn("_rshlv", TVLONG);
	nodandv = fvn("_andv", TVLONG);
	nodorv = fvn("_orv", TVLONG);
	nodxorv = fvn("_xorv", TVLONG);
	nodnegv = fvn("_negv", TVLONG);
	nodcomv = fvn("_comv", TVLONG);

	nodtestv = fvn("_testv", TLONG);
	nodeqv = fvn("_eqv", TLONG);
	nodnev = fvn("_nev", TLONG);
	nodlev = fvn("_lev", TLONG);
	nodltv = fvn("_ltv", TLONG);
	nodgev = fvn("_gev", TLONG);
	nodgtv = fvn("_gtv", TLONG);
	nodhiv = fvn("_hiv", TLONG);
	nodhsv = fvn("_hsv", TLONG);
	nodlov = fvn("_lov", TLONG);
	nodlsv = fvn("_lsv", TLONG);

	nodf2v = fvn("_f2v", TVLONG);
	nodd2v = fvn("_d2v", TVLONG);
	nodp2v = fvn("_p2v", TVLONG);
	nodsi2v = fvn("_si2v", TVLONG);
	nodui2v = fvn("_ui2v", TVLONG);
	nodsl2v = fvn("_sl2v", TVLONG);
	nodul2v = fvn("_ul2v", TVLONG);
	nodsh2v = fvn("_sh2v", TVLONG);
	noduh2v = fvn("_uh2v", TVLONG);
	nodsc2v = fvn("_sc2v", TVLONG);
	noduc2v = fvn("_uc2v", TVLONG);

	nodv2f = fvn("_v2f", TFLOAT);
	nodv2d = fvn("_v2d", TDOUBLE);
	nodv2sl = fvn("_v2sl", TLONG);
	nodv2ul = fvn("_v2ul", TULONG);
	nodv2si = fvn("_v2si", TINT);
	nodv2ui = fvn("_v2ui", TUINT);
	nodv2sh = fvn("_v2sh", TSHORT);
	nodv2uh = fvn("_v2ul", TUSHORT);
	nodv2sc = fvn("_v2sc", TCHAR);
	nodv2uc = fvn("_v2uc", TUCHAR);

	nodvpp = fvn("_vpp", TVLONG);
	nodppv = fvn("_ppv", TVLONG);
	nodvmm = fvn("_vmm", TVLONG);
	nodmmv = fvn("_mmv", TVLONG);

	nodvasop = fvn("_vasop", TVLONG);

	for(p = initetconv; p->code >= 0; p++)
		etconv[p->code] = p->value;
}

int
com64(Node *n)
{
	Node *l, *r, *a, *t;
	int lv, rv;

	if(n->type == 0)
		return 0;

	l = n->left;
	r = n->right;

	lv = 0;
	if(l && l->type && typev[l->type->etype])
		lv = 1;
	rv = 0;
	if(r && r->type && typev[r->type->etype])
		rv = 1;

	if(lv) {
		switch(n->op) {
		case OEQ:
			a = nodeqv;
			goto setbool;
		case ONE:
			a = nodnev;
			goto setbool;
		case OLE:
			a = nodlev;
			goto setbool;
		case OLT:
			a = nodltv;
			goto setbool;
		case OGE:
			a = nodgev;
			goto setbool;
		case OGT:
			a = nodgtv;
			goto setbool;
		case OHI:
			a = nodhiv;
			goto setbool;
		case OHS:
			a = nodhsv;
			goto setbool;
		case OLO:
			a = nodlov;
			goto setbool;
		case OLS:
			a = nodlsv;
			goto setbool;

		case OANDAND:
		case OOROR:
			if(machcap(n))
				return 1;

			if(rv) {
				r = new(OFUNC, nodtestv, r);
				n->right = r;
				r->complex = FNX;
				r->op = OFUNC;
				r->type = types[TLONG];
			}

		case OCOND:
		case ONOT:
			if(machcap(n))
				return 1;

			l = new(OFUNC, nodtestv, l);
			n->left = l;
			l->complex = FNX;
			l->op = OFUNC;
			l->type = types[TLONG];
			n->complex = FNX;
			return 1;
		}
	}

	if(rv) {
		if(machcap(n))
			return 1;
		switch(n->op) {
		case OANDAND:
		case OOROR:
			r = new(OFUNC, nodtestv, r);
			n->right = r;
			r->complex = FNX;
			r->op = OFUNC;
			r->type = types[TLONG];
			return 1;
		}
	}

	if(typev[n->type->etype]) {
		if(machcap(n))
			return 1;
		switch(n->op) {
		default:
			diag(n, "unknown vlong %O", n->op);
		case OFUNC:
			n->complex = FNX;
		case ORETURN:
		case OAS:
		case OIND:
			return 1;
		case OADD:
			a = nodaddv;
			goto setbop;
		case OSUB:
			a = nodsubv;
			goto setbop;
		case OMUL:
		case OLMUL:
			a = nodmulv;
			goto setbop;
		case ODIV:
			a = noddivv;
			goto setbop;
		case OLDIV:
			a = noddivvu;
			goto setbop;
		case OMOD:
			a = nodmodv;
			goto setbop;
		case OLMOD:
			a = nodmodvu;
			goto setbop;
		case OASHL:
			a = nodlshv;
			goto setbop;
		case OASHR:
			a = nodrshav;
			goto setbop;
		case OLSHR:
			a = nodrshlv;
			goto setbop;
		case OAND:
			a = nodandv;
			goto setbop;
		case OOR:
			a = nodorv;
			goto setbop;
		case OXOR:
			a = nodxorv;
			goto setbop;
		case OPOSTINC:
			a = nodvpp;
			goto setvinc;
		case OPOSTDEC:
			a = nodvmm;
			goto setvinc;
		case OPREINC:
			a = nodppv;
			goto setvinc;
		case OPREDEC:
			a = nodmmv;
			goto setvinc;
		case ONEG:
			a = nodnegv;
			goto setfnx;
		case OCOM:
			a = nodcomv;
			goto setfnx;
		case OCAST:
			switch(l->type->etype) {
			case TCHAR:
				a = nodsc2v;
				goto setfnxl;
			case TUCHAR:
				a = noduc2v;
				goto setfnxl;
			case TSHORT:
				a = nodsh2v;
				goto setfnxl;
			case TUSHORT:
				a = noduh2v;
				goto setfnxl;
			case TINT:
				a = nodsi2v;
				goto setfnx;
			case TUINT:
				a = nodui2v;
				goto setfnx;
			case TLONG:
				a = nodsl2v;
				goto setfnx;
			case TULONG:
				a = nodul2v;
				goto setfnx;
			case TFLOAT:
				a = nodf2v;
				goto setfnx;
			case TDOUBLE:
				a = nodd2v;
				goto setfnx;
			case TIND:
				a = nodp2v;
				goto setfnx;
			}
			diag(n, "unknown %T->vlong cast", l->type);
			return 1;
		case OASADD:
			a = nodaddv;
			goto setasop;
		case OASSUB:
			a = nodsubv;
			goto setasop;
		case OASMUL:
		case OASLMUL:
			a = nodmulv;
			goto setasop;
		case OASDIV:
			a = noddivv;
			goto setasop;
		case OASLDIV:
			a = noddivvu;
			goto setasop;
		case OASMOD:
			a = nodmodv;
			goto setasop;
		case OASLMOD:
			a = nodmodvu;
			goto setasop;
		case OASASHL:
			a = nodlshv;
			goto setasop;
		case OASASHR:
			a = nodrshav;
			goto setasop;
		case OASLSHR:
			a = nodrshlv;
			goto setasop;
		case OASAND:
			a = nodandv;
			goto setasop;
		case OASOR:
			a = nodorv;
			goto setasop;
		case OASXOR:
			a = nodxorv;
			goto setasop;
		}
	}

	if(typefd[n->type->etype] && l && l->op == OFUNC) {
		switch(n->op) {
		case OASADD:
		case OASSUB:
		case OASMUL:
		case OASLMUL:
		case OASDIV:
		case OASLDIV:
		case OASMOD:
		case OASLMOD:
		case OASASHL:
		case OASASHR:
		case OASLSHR:
		case OASAND:
		case OASOR:
		case OASXOR:
			if(l->right && typev[l->right->etype]) {
				diag(n, "sorry float <asop> vlong not implemented\n");
			}
		}
	}

	if(n->op == OCAST) {
		if(l->type && typev[l->type->etype]) {
			if(machcap(n))
				return 1;
			switch(n->type->etype) {
			case TDOUBLE:
				a = nodv2d;
				goto setfnx;
			case TFLOAT:
				a = nodv2f;
				goto setfnx;
			case TLONG:
				a = nodv2sl;
				goto setfnx;
			case TULONG:
				a = nodv2ul;
				goto setfnx;
			case TINT:
				a = nodv2si;
				goto setfnx;
			case TUINT:
				a = nodv2ui;
				goto setfnx;
			case TSHORT:
				a = nodv2sh;
				goto setfnx;
			case TUSHORT:
				a = nodv2uh;
				goto setfnx;
			case TCHAR:
				a = nodv2sc;
				goto setfnx;
			case TUCHAR:
				a = nodv2uc;
				goto setfnx;
			case TIND:	// small pun here
				a = nodv2ul;
				goto setfnx;
			}
			diag(n, "unknown vlong->%T cast", n->type);
			return 1;
		}
	}

	return 0;

setbop:
	n->left = a;
	n->right = new(OLIST, l, r);
	n->complex = FNX;
	n->op = OFUNC;
	return 1;

setfnxl:
	l = new(OCAST, l, 0);
	l->type = types[TLONG];
	l->complex = l->left->complex;

setfnx:
	n->left = a;
	n->right = l;
	n->complex = FNX;
	n->op = OFUNC;
	return 1;

setvinc:
	n->left = a;
	l = new(OADDR, l, Z);
	l->type = typ(TIND, l->left->type);
	n->right = new(OLIST, l, r);
	n->complex = FNX;
	n->op = OFUNC;
	return 1;

setbool:
	if(machcap(n))
		return 1;
	n->left = a;
	n->right = new(OLIST, l, r);
	n->complex = FNX;
	n->op = OFUNC;
	n->type = types[TLONG];
	return 1;

setasop:
	if(l->op == OFUNC) {
		l = l->right;
		goto setasop;
	}

	t = new(OCONST, 0, 0);
	t->vconst = etconv[l->type->etype];
	t->type = types[TLONG];
	t->addable = 20;
	r = new(OLIST, t, r);

	t = new(OADDR, a, 0);
	t->type = typ(TIND, a->type);
	r = new(OLIST, t, r);

	t = new(OADDR, l, 0);
	t->type = typ(TIND, l->type);
	r = new(OLIST, t, r);

	n->left = nodvasop;
	n->right = r;
	n->complex = FNX;
	n->op = OFUNC;

	return 1;
}

void
bool64(Node *n)
{
	Node *n1;

	if(machcap(Z))
		return;
	if(typev[n->type->etype]) {
		n1 = new(OXXX, 0, 0);
		*n1 = *n;

		n->right = n1;
		n->left = nodtestv;
		n->complex = FNX;
		n->addable = 0;
		n->op = OFUNC;
		n->type = types[TLONG];
	}
}

/*
 * more machine depend stuff.
 * this is common for 8,16,32,64 bit machines.
 * this is common for ieee machines.
 */
double
convvtof(vlong v)
{
	double d;

	d = v;		/* BOTCH */
	return d;
}

vlong
convftov(double d)
{
	vlong v;


	v = d;		/* BOTCH */
	return v;
}

double
convftox(double d, int et)
{

	if(!typefd[et])
		diag(Z, "bad type in castftox %s", tnames[et]);
	return d;
}

vlong
convvtox(vlong c, int et)
{
	int n;

	n = 8 * ewidth[et];
	c &= MASK(n);
	if(!typeu[et])
		if(c & SIGN(n))
			c |= ~MASK(n);
	return c;
}
