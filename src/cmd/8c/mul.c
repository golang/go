// Inferno utils/8c/mul.c
// http://code.google.com/p/inferno-os/source/browse/utils/8c/mul.c
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

typedef struct	Malg	Malg;
typedef struct	Mparam	Mparam;

struct	Malg
{
	schar	vals[10];
};

struct	Mparam
{
	uint32	value;
	schar	alg;
	char	neg;
	char	shift;
	char	arg;
	schar	off;
};

static	Mparam	multab[32];
static	int	mulptr;

static	Malg	malgs[]	=
{
	{0, 100},
	{-1, 1, 100},
	{-9, -5, -3, 3, 5, 9, 100},
	{6, 10, 12, 18, 20, 24, 36, 40, 72, 100},
	{-8, -4, -2, 2, 4, 8, 100},
};

/*
 * return position of lowest 1
 */
int
lowbit(uint32 v)
{
	int s, i;
	uint32 m;

	s = 0;
	m = 0xFFFFFFFFUL;
	for(i = 16; i > 0; i >>= 1) {
		m >>= i;
		if((v & m) == 0) {
			v >>= i;
			s += i;
		}
	}
	return s;
}

void
genmuladd(Node *d, Node *s, int m, Node *a)
{
	Node nod;

	nod.op = OINDEX;
	nod.left = a;
	nod.right = s;
	nod.scale = m;
	nod.type = types[TIND];
	nod.xoffset = 0;
	xcom(&nod);
	gopcode(OADDR, d->type, &nod, d);
}

void
mulparam(uint32 m, Mparam *mp)
{
	int c, i, j, n, o, q, s;
	int bc, bi, bn, bo, bq, bs, bt;
	schar *p;
	int32 u;
	uint32 t;

	bc = bq = 10;
	bi = bn = bo = bs = bt = 0;
	for(i = 0; i < nelem(malgs); i++) {
		for(p = malgs[i].vals, j = 0; (o = p[j]) < 100; j++)
		for(s = 0; s < 2; s++) {
			c = 10;
			q = 10;
			u = m - o;
			if(u == 0)
				continue;
			if(s) {
				o = -o;
				if(o > 0)
					continue;
				u = -u;
			}
			n = lowbit(u);
			t = (uint32)u >> n;
			switch(i) {
			case 0:
				if(t == 1) {
					c = s + 1;
					q = 0;
					break;
				}
				switch(t) {
				case 3:
				case 5:
				case 9:
					c = s + 1;
					if(n)
						c++;
					q = 0;
					break;
				}
				if(s)
					break;
				switch(t) {
				case 15:
				case 25:
				case 27:
				case 45:
				case 81:
					c = 2;
					if(n)
						c++;
					q = 1;
					break;
				}
				break;
			case 1:
				if(t == 1) {
					c = 3;
					q = 3;
					break;
				}
				switch(t) {
				case 3:
				case 5:
				case 9:
					c = 3;
					q = 2;
					break;
				}
				break;
			case 2:
				if(t == 1) {
					c = 3;
					q = 2;
					break;
				}
				break;
			case 3:
				if(s)
					break;
				if(t == 1) {
					c = 3;
					q = 1;
					break;
				}
				break;
			case 4:
				if(t == 1) {
					c = 3;
					q = 0;
					break;
				}
				break;
			}
			if(c < bc || (c == bc && q > bq)) {
				bc = c;
				bi = i;
				bn = n;
				bo = o;
				bq = q;
				bs = s;
				bt = t;
			}
		}
	}
	mp->value = m;
	if(bc <= 3) {
		mp->alg = bi;
		mp->shift = bn;
		mp->off = bo;
		mp->neg = bs;
		mp->arg = bt;
	}
	else
		mp->alg = -1;
}

int
m0(int a)
{
	switch(a) {
	case -2:
	case 2:
		return 2;
	case -3:
	case 3:
		return 2;
	case -4:
	case 4:
		return 4;
	case -5:
	case 5:
		return 4;
	case 6:
		return 2;
	case -8:
	case 8:
		return 8;
	case -9:
	case 9:
		return 8;
	case 10:
		return 4;
	case 12:
		return 2;
	case 15:
		return 2;
	case 18:
		return 8;
	case 20:
		return 4;
	case 24:
		return 2;
	case 25:
		return 4;
	case 27:
		return 2;
	case 36:
		return 8;
	case 40:
		return 4;
	case 45:
		return 4;
	case 72:
		return 8;
	case 81:
		return 8;
	}
	diag(Z, "bad m0");
	return 0;
}

int
m1(int a)
{
	switch(a) {
	case 15:
		return 4;
	case 25:
		return 4;
	case 27:
		return 8;
	case 45:
		return 8;
	case 81:
		return 8;
	}
	diag(Z, "bad m1");
	return 0;
}

int
m2(int a)
{
	switch(a) {
	case 6:
		return 2;
	case 10:
		return 2;
	case 12:
		return 4;
	case 18:
		return 2;
	case 20:
		return 4;
	case 24:
		return 8;
	case 36:
		return 4;
	case 40:
		return 8;
	case 72:
		return 8;
	}
	diag(Z, "bad m2");
	return 0;
}

void
shiftit(Type *t, Node *s, Node *d)
{
	int32 c;

	c = (int32)s->vconst & 31;
	switch(c) {
	case 0:
		break;
	case 1:
		gopcode(OADD, t, d, d);
		break;
	default:
		gopcode(OASHL, t, s, d);
	}
}

static int
mulgen1(uint32 v, Node *n)
{
	int i, o;
	Mparam *p;
	Node nod, nods;

	for(i = 0; i < nelem(multab); i++) {
		p = &multab[i];
		if(p->value == v)
			goto found;
	}

	p = &multab[mulptr];
	if(++mulptr == nelem(multab))
		mulptr = 0;

	mulparam(v, p);

found:
//	print("v=%.x a=%d n=%d s=%d g=%d o=%d \n", p->value, p->alg, p->neg, p->shift, p->arg, p->off);
	if(p->alg < 0)
		return 0;

	nods = *nodconst(p->shift);

	o = OADD;
	if(p->alg > 0) {
		regalloc(&nod, n, Z);
		if(p->off < 0)
			o = OSUB;
	}

	switch(p->alg) {
	case 0:
		switch(p->arg) {
		case 1:
			shiftit(n->type, &nods, n);
			break;
		case 15:
		case 25:
		case 27:
		case 45:
		case 81:
			genmuladd(n, n, m1(p->arg), n);
			/* fall thru */
		case 3:
		case 5:
		case 9:
			genmuladd(n, n, m0(p->arg), n);
			shiftit(n->type, &nods, n);
			break;
		default:
			goto bad;
		}
		if(p->neg == 1)
			gins(ANEGL, Z, n);
		break;
	case 1:
		switch(p->arg) {
		case 1:
			gmove(n, &nod);
			shiftit(n->type, &nods, &nod);
			break;
		case 3:
		case 5:
		case 9:
			genmuladd(&nod, n, m0(p->arg), n);
			shiftit(n->type, &nods, &nod);
			break;
		default:
			goto bad;
		}
		if(p->neg)
			gopcode(o, n->type, &nod, n);
		else {
			gopcode(o, n->type, n, &nod);
			gmove(&nod, n);
		}
		break;
	case 2:
		genmuladd(&nod, n, m0(p->off), n);
		shiftit(n->type, &nods, n);
		goto comop;
	case 3:
		genmuladd(&nod, n, m0(p->off), n);
		shiftit(n->type, &nods, n);
		genmuladd(n, &nod, m2(p->off), n);
		break;
	case 4:
		genmuladd(&nod, n, m0(p->off), nodconst(0));
		shiftit(n->type, &nods, n);
		goto comop;
	default:
		diag(Z, "bad mul alg");
		break;
	comop:
		if(p->neg) {
			gopcode(o, n->type, n, &nod);
			gmove(&nod, n);
		}
		else
			gopcode(o, n->type, &nod, n);
	}

	if(p->alg > 0)
		regfree(&nod);

	return 1;

bad:
	diag(Z, "mulgen botch");
	return 1;
}

void
mulgen(Type *t, Node *r, Node *n)
{
	if(!mulgen1(r->vconst, n))
		gopcode(OMUL, t, r, n);
}
