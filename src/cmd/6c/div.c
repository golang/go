// Inferno utils/6c/div.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/div.c
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

#include "gc.h"

/*
 * Based on: Granlund, T.; Montgomery, P.L.
 * "Division by Invariant Integers using Multiplication".
 * SIGPLAN Notices, Vol. 29, June 1994, page 61.
 */

#define	TN(n)	((uvlong)1 << (n))
#define	T31	TN(31)
#define	T32	TN(32)

int
multiplier(uint32 d, int p, uvlong *mp)
{
	int l;
	uvlong mlo, mhi, tlo, thi;

	l = topbit(d - 1) + 1;
	mlo = (((TN(l) - d) << 32) / d) + T32;
	if(l + p == 64)
		mhi = (((TN(l) + 1 - d) << 32) / d) + T32;
	else
		mhi = (TN(32 + l) + TN(32 + l - p)) / d;
	/*assert(mlo < mhi);*/
	while(l > 0) {
		tlo = mlo >> 1;
		thi = mhi >> 1;
		if(tlo == thi)
			break;
		mlo = tlo;
		mhi = thi;
		l--;
	}
	*mp = mhi;
	return l;
}

int
sdiv(uint32 d, uint32 *mp, int *sp)
{
	int s;
	uvlong m;

	s = multiplier(d, 32 - 1, &m);
	*mp = m;
	*sp = s;
	if(m >= T31)
		return 1;
	else
		return 0;
}

int
udiv(uint32 d, uint32 *mp, int *sp, int *pp)
{
	int p, s;
	uvlong m;

	s = multiplier(d, 32, &m);
	p = 0;
	if(m >= T32) {
		while((d & 1) == 0) {
			d >>= 1;
			p++;
		}
		s = multiplier(d, 32 - p, &m);
	}
	*mp = m;
	*pp = p;
	if(m >= T32) {
		/*assert(p == 0);*/
		*sp = s - 1;
		return 1;
	}
	else {
		*sp = s;
		return 0;
	}
}

void
sdivgen(Node *l, Node *r, Node *ax, Node *dx)
{
	int a, s;
	uint32 m;
	vlong c;

	c = r->vconst;
	if(c < 0)
		c = -c;
	a = sdiv(c, &m, &s);
//print("a=%d i=%d s=%d m=%ux\n", a, (long)r->vconst, s, m);
	gins(AMOVL, nodconst(m), ax);
	gins(AIMULL, l, Z);
	gins(AMOVL, l, ax);
	if(a)
		gins(AADDL, ax, dx);
	gins(ASHRL, nodconst(31), ax);
	gins(ASARL, nodconst(s), dx);
	gins(AADDL, ax, dx);
	if(r->vconst < 0)
		gins(ANEGL, Z, dx);
}

void
udivgen(Node *l, Node *r, Node *ax, Node *dx)
{
	int a, s, t;
	uint32 m;
	Node nod;

	a = udiv(r->vconst, &m, &s, &t);
//print("a=%ud i=%d p=%d s=%d m=%ux\n", a, (long)r->vconst, t, s, m);
	if(t != 0) {
		gins(AMOVL, l, ax);
		gins(ASHRL, nodconst(t), ax);
		gins(AMOVL, nodconst(m), dx);
		gins(AMULL, dx, Z);
	}
	else if(a) {
		if(l->op != OREGISTER) {
			regalloc(&nod, l, Z);
			gins(AMOVL, l, &nod);
			l = &nod;
		}
		gins(AMOVL, nodconst(m), ax);
		gins(AMULL, l, Z);
		gins(AADDL, l, dx);
		gins(ARCRL, nodconst(1), dx);
		if(l == &nod)
			regfree(l);
	}
	else {
		gins(AMOVL, nodconst(m), ax);
		gins(AMULL, l, Z);
	}
	if(s != 0)
		gins(ASHRL, nodconst(s), dx);
}

void
sext(Node *d, Node *s, Node *l)
{
	if(s->reg == D_AX && !nodreg(d, Z, D_DX)) {
		reg[D_DX]++;
		gins(ACDQ, Z, Z);
	}
	else {
		regalloc(d, l, Z);
		gins(AMOVL, s, d);
		gins(ASARL, nodconst(31), d);
	}
}

void
sdiv2(int32 c, int v, Node *l, Node *n)
{
	Node nod;

	if(v > 0) {
		if(v > 1) {
			sext(&nod, n, l);
			gins(AANDL, nodconst((1 << v) - 1), &nod);
			gins(AADDL, &nod, n);
			regfree(&nod);
		}
		else {
			gins(ACMPL, n, nodconst(0x80000000));
			gins(ASBBL, nodconst(-1), n);
		}
		gins(ASARL, nodconst(v), n);
	}
	if(c < 0)
		gins(ANEGL, Z, n);
}

void
smod2(int32 c, int v, Node *l, Node *n)
{
	Node nod;

	if(c == 1) {
		zeroregm(n);
		return;
	}

	sext(&nod, n, l);
	if(v == 0) {
		zeroregm(n);
		gins(AXORL, &nod, n);
		gins(ASUBL, &nod, n);
	}
	else if(v > 1) {
		gins(AANDL, nodconst((1 << v) - 1), &nod);
		gins(AADDL, &nod, n);
		gins(AANDL, nodconst((1 << v) - 1), n);
		gins(ASUBL, &nod, n);
	}
	else {
		gins(AANDL, nodconst(1), n);
		gins(AXORL, &nod, n);
		gins(ASUBL, &nod, n);
	}
	regfree(&nod);
}
