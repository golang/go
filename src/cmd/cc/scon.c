// Inferno utils/cc/scon.c
// http://code.google.com/p/inferno-os/source/browse/utils/cc/scon.c
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

static Node*
acast(Type *t, Node *n)
{
	if(n->type->etype != t->etype || n->op == OBIT) {
		n = new1(OCAST, n, Z);
		if(nocast(n->left->type, t))
			*n = *n->left;
		n->type = t;
	}
	return n;
}


void
evconst(Node *n)
{
	Node *l, *r;
	int et, isf;
	vlong v;
	double d;

	if(n == Z || n->type == T)
		return;

	et = n->type->etype;
	isf = typefd[et];

	l = n->left;
	r = n->right;

	d = 0;
	v = 0;

	switch(n->op) {
	default:
		return;

	case ONEG:
		if(isf)
			d = -l->fconst;
		else
			v = -l->vconst;
		break;

	case OCOM:
		v = ~l->vconst;
		break;

	case OCAST:
		if(et == TVOID)
			return;
		et = l->type->etype;
		if(isf) {
			if(typefd[et])
				d = l->fconst;
			else
				d = l->vconst;
		} else {
			if(typefd[et])
				v = l->fconst;
			else
				v = convvtox(l->vconst, n->type->etype);
		}
		break;

	case OCONST:
		break;

	case OADD:
		if(isf)
			d = l->fconst + r->fconst;
		else {
			v = l->vconst + r->vconst;
		}
		break;

	case OSUB:
		if(isf)
			d = l->fconst - r->fconst;
		else
			v = l->vconst - r->vconst;
		break;

	case OMUL:
		if(isf)
			d = l->fconst * r->fconst;
		else {
			v = l->vconst * r->vconst;
		}
		break;

	case OLMUL:
		v = (uvlong)l->vconst * (uvlong)r->vconst;
		break;


	case ODIV:
		if(vconst(r) == 0) {
			warn(n, "divide by zero");
			return;
		}
		if(isf)
			d = l->fconst / r->fconst;
		else
			v = l->vconst / r->vconst;
		break;

	case OLDIV:
		if(vconst(r) == 0) {
			warn(n, "divide by zero");
			return;
		}
		v = (uvlong)l->vconst / (uvlong)r->vconst;
		break;

	case OMOD:
		if(vconst(r) == 0) {
			warn(n, "modulo by zero");
			return;
		}
		v = l->vconst % r->vconst;
		break;

	case OLMOD:
		if(vconst(r) == 0) {
			warn(n, "modulo by zero");
			return;
		}
		v = (uvlong)l->vconst % (uvlong)r->vconst;
		break;

	case OAND:
		v = l->vconst & r->vconst;
		break;

	case OOR:
		v = l->vconst | r->vconst;
		break;

	case OXOR:
		v = l->vconst ^ r->vconst;
		break;

	case OLSHR:
		if(l->type->width != sizeof(uvlong))
			v = ((uvlong)l->vconst & 0xffffffffULL) >> r->vconst;
		else
			v = (uvlong)l->vconst >> r->vconst;
		break;

	case OASHR:
		v = l->vconst >> r->vconst;
		break;

	case OASHL:
		v = (uvlong)l->vconst << r->vconst;
		break;

	case OLO:
		v = (uvlong)l->vconst < (uvlong)r->vconst;
		break;

	case OLT:
		if(typefd[l->type->etype])
			v = l->fconst < r->fconst;
		else
			v = l->vconst < r->vconst;
		break;

	case OHI:
		v = (uvlong)l->vconst > (uvlong)r->vconst;
		break;

	case OGT:
		if(typefd[l->type->etype])
			v = l->fconst > r->fconst;
		else
			v = l->vconst > r->vconst;
		break;

	case OLS:
		v = (uvlong)l->vconst <= (uvlong)r->vconst;
		break;

	case OLE:
		if(typefd[l->type->etype])
			v = l->fconst <= r->fconst;
		else
			v = l->vconst <= r->vconst;
		break;

	case OHS:
		v = (uvlong)l->vconst >= (uvlong)r->vconst;
		break;

	case OGE:
		if(typefd[l->type->etype])
			v = l->fconst >= r->fconst;
		else
			v = l->vconst >= r->vconst;
		break;

	case OEQ:
		if(typefd[l->type->etype])
			v = l->fconst == r->fconst;
		else
			v = l->vconst == r->vconst;
		break;

	case ONE:
		if(typefd[l->type->etype])
			v = l->fconst != r->fconst;
		else
			v = l->vconst != r->vconst;
		break;

	case ONOT:
		if(typefd[l->type->etype])
			v = !l->fconst;
		else
			v = !l->vconst;
		break;

	case OANDAND:
		if(typefd[l->type->etype])
			v = l->fconst && r->fconst;
		else
			v = l->vconst && r->vconst;
		break;

	case OOROR:
		if(typefd[l->type->etype])
			v = l->fconst || r->fconst;
		else
			v = l->vconst || r->vconst;
		break;
	}
	if(isf) {
		n->fconst = d;
	} else {
		n->vconst = convvtox(v, n->type->etype);
	}
	n->oldop = n->op;
	n->op = OCONST;
}

void
acom(Node *n)
{
	Type *t;
	Node *l, *r;
	int i;

	switch(n->op)
	{

	case ONAME:
	case OCONST:
	case OSTRING:
	case OINDREG:
	case OREGISTER:
		return;

	case ONEG:
		l = n->left;
		if(addo(n) && addo(l))
			break;
		acom(l);
		return;

	case OADD:
	case OSUB:
	case OMUL:
		l = n->left;
		r = n->right;
		if(addo(n)) {
			if(addo(r))
				break;
			if(addo(l))
				break;
		}
		acom(l);
		acom(r);
		return;

	default:
		l = n->left;
		r = n->right;
		if(l != Z)
			acom(l);
		if(r != Z)
			acom(r);
		return;
	}

	/* bust terms out */
	t = n->type;
	term[0].mult = 0;
	term[0].node = Z;
	nterm = 1;
	acom1(1, n);
	if(debug['m'])
	for(i=0; i<nterm; i++) {
		print("%d %3lld ", i, term[i].mult);
		prtree1(term[i].node, 1, 0);
	}
	if(nterm < NTERM)
		acom2(n, t);
	n->type = t;
}

int
acomcmp1(const void *a1, const void *a2)
{
	vlong c1, c2;
	Term *t1, *t2;

	t1 = (Term*)a1;
	t2 = (Term*)a2;
	c1 = t1->mult;
	if(c1 < 0)
		c1 = -c1;
	c2 = t2->mult;
	if(c2 < 0)
		c2 = -c2;
	if(c1 > c2)
		return 1;
	if(c1 < c2)
		return -1;
	c1 = 1;
	if(t1->mult < 0)
		c1 = 0;
	c2 = 1;
	if(t2->mult < 0)
		c2 = 0;
	if(c2 -= c1)
		return c2;
	if(t2 > t1)
		return 1;
	return -1;
}

int
acomcmp2(const void *a1, const void *a2)
{
	vlong c1, c2;
	Term *t1, *t2;

	t1 = (Term*)a1;
	t2 = (Term*)a2;
	c1 = t1->mult;
	c2 = t2->mult;
	if(c1 > c2)
		return 1;
	if(c1 < c2)
		return -1;
	if(t2 > t1)
		return 1;
	return -1;
}

void
acom2(Node *n, Type *t)
{
	Node *l, *r;
	Term trm[NTERM];
	int et, nt, i, j;
	vlong c1, c2;

	/*
	 * copy into automatic
	 */
	c2 = 0;
	nt = nterm;
	for(i=0; i<nt; i++)
		trm[i] = term[i];
	/*
	 * recur on subtrees
	 */
	j = 0;
	for(i=1; i<nt; i++) {
		c1 = trm[i].mult;
		if(c1 == 0)
			continue;
		l = trm[i].node;
		if(l != Z) {
			j = 1;
			acom(l);
		}
	}
	c1 = trm[0].mult;
	if(j == 0) {
		n->oldop = n->op;
		n->op = OCONST;
		n->vconst = c1;
		return;
	}
	et = t->etype;

	/*
	 * prepare constant term,
	 * combine it with an addressing term
	 */
	if(c1 != 0) {
		l = new1(OCONST, Z, Z);
		l->type = t;
		l->vconst = c1;
		trm[0].mult = 1;
		for(i=1; i<nt; i++) {
			if(trm[i].mult != 1)
				continue;
			r = trm[i].node;
			if(r->op != OADDR)
				continue;
			r->type = t;
			l = new1(OADD, r, l);
			l->type = t;
			trm[i].mult = 0;
			break;
		}
		trm[0].node = l;
	}
	/*
	 * look for factorable terms
	 * c1*i + c1*c2*j -> c1*(i + c2*j)
	 */
	qsort(trm+1, nt-1, sizeof(trm[0]), acomcmp1);
	for(i=nt-1; i>=0; i--) {
		c1 = trm[i].mult;
		if(c1 < 0)
			c1 = -c1;
		if(c1 <= 1)
			continue;
		for(j=i+1; j<nt; j++) {
			c2 = trm[j].mult;
			if(c2 < 0)
				c2 = -c2;
			if(c2 <= 1)
				continue;
			if(c2 % c1)
				continue;
			r = trm[j].node;
			if(r->type->etype != et)
				r = acast(t, r);
			c2 = trm[j].mult/trm[i].mult;
			if(c2 != 1 && c2 != -1) {
				r = new1(OMUL, r, new(OCONST, Z, Z));
				r->type = t;
				r->right->type = t;
				r->right->vconst = c2;
			}
			l = trm[i].node;
			if(l->type->etype != et)
				l = acast(t, l);
			r = new1(OADD, l, r);
			r->type = t;
			if(c2 == -1)
				r->op = OSUB;
			trm[i].node = r;
			trm[j].mult = 0;
		}
	}
	if(debug['m']) {
		print("\n");
		for(i=0; i<nt; i++) {
			print("%d %3lld ", i, trm[i].mult);
			prtree1(trm[i].node, 1, 0);
		}
	}

	/*
	 * put it all back together
	 */
	qsort(trm+1, nt-1, sizeof(trm[0]), acomcmp2);
	l = Z;
	for(i=nt-1; i>=0; i--) {
		c1 = trm[i].mult;
		if(c1 == 0)
			continue;
		r = trm[i].node;
		if(r->type->etype != et || r->op == OBIT)
			r = acast(t, r);
		if(c1 != 1 && c1 != -1) {
			r = new1(OMUL, r, new(OCONST, Z, Z));
			r->type = t;
			r->right->type = t;
			if(c1 < 0) {
				r->right->vconst = -c1;
				c1 = -1;
			} else {
				r->right->vconst = c1;
				c1 = 1;
			}
		}
		if(l == Z) {
			l = r;
			c2 = c1;
			continue;
		}
		if(c1 < 0)
			if(c2 < 0)
				l = new1(OADD, l, r);
			else
				l = new1(OSUB, l, r);
		else
			if(c2 < 0) {
				l = new1(OSUB, r, l);
				c2 = 1;
			} else
				l = new1(OADD, l, r);
		l->type = t;
	}
	if(c2 < 0) {
		r = new1(OCONST, 0, 0);
		r->vconst = 0;
		r->type = t;
		l = new1(OSUB, r, l);
		l->type = t;
	}
	*n = *l;
}

void
acom1(vlong v, Node *n)
{
	Node *l, *r;

	if(v == 0 || nterm >= NTERM)
		return;
	if(!addo(n)) {
		if(n->op == OCONST)
		if(!typefd[n->type->etype]) {
			term[0].mult += v*n->vconst;
			return;
		}
		term[nterm].mult = v;
		term[nterm].node = n;
		nterm++;
		return;
	}
	switch(n->op) {

	case OCAST:
		acom1(v, n->left);
		break;

	case ONEG:
		acom1(-v, n->left);
		break;

	case OADD:
		acom1(v, n->left);
		acom1(v, n->right);
		break;

	case OSUB:
		acom1(v, n->left);
		acom1(-v, n->right);
		break;

	case OMUL:
		l = n->left;
		r = n->right;
		if(l->op == OCONST)
		if(!typefd[n->type->etype]) {
			acom1(v*l->vconst, r);
			break;
		}
		if(r->op == OCONST)
		if(!typefd[n->type->etype]) {
			acom1(v*r->vconst, l);
			break;
		}
		break;

	default:
		diag(n, "not addo");
	}
}

int
addo(Node *n)
{

	if(n != Z)
	if(!typefd[n->type->etype])
	if(!typev[n->type->etype] || ewidth[TVLONG] == ewidth[TIND])
	switch(n->op) {

	case OCAST:
		if(nilcast(n->left->type, n->type))
			return 1;
		break;

	case ONEG:
	case OADD:
	case OSUB:
		return 1;

	case OMUL:
		if(n->left->op == OCONST)
			return 1;
		if(n->right->op == OCONST)
			return 1;
	}
	return 0;
}
