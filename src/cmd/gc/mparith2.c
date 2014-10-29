// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	<u.h>
#include	<libc.h>
#include	"go.h"

//
// return the significant
// words of the argument
//
static int
mplen(Mpint *a)
{
	int i, n;
	long *a1;

	n = -1;
	a1 = &a->a[0];
	for(i=0; i<Mpprec; i++) {
		if(*a1++ != 0)
			n = i;
	}
	return n+1;
}

//
// left shift mpint by one
// ignores sign
//
static void
mplsh(Mpint *a, int quiet)
{
	long *a1, x;
	int i, c;

	c = 0;
	a1 = &a->a[0];
	for(i=0; i<Mpprec; i++) {
		x = (*a1 << 1) + c;
		c = 0;
		if(x >= Mpbase) {
			x -= Mpbase;
			c = 1;
		}
		*a1++ = x;
	}
	a->ovf = c;
	if(a->ovf && !quiet)
		yyerror("constant shift overflow");
}

//
// left shift mpint by Mpscale
// ignores sign
//
static void
mplshw(Mpint *a, int quiet)
{
	long *a1;
	int i;

	a1 = &a->a[Mpprec-1];
	if(*a1) {
		a->ovf = 1;
		if(!quiet)
			yyerror("constant shift overflow");
	}
	for(i=1; i<Mpprec; i++) {
		a1[0] = a1[-1];
		a1--;
	}
	a1[0] = 0;
}

//
// right shift mpint by one
// ignores sign and overflow
//
static void
mprsh(Mpint *a)
{
	long *a1, x, lo;
	int i, c;

	c = 0;
	lo = a->a[0] & 1;
	a1 = &a->a[Mpprec];
	for(i=0; i<Mpprec; i++) {
		x = *--a1;
		*a1 = (x + c) >> 1;
		c = 0;
		if(x & 1)
			c = Mpbase;
	}
	if(a->neg && lo != 0)
		mpaddcfix(a, -1);
}

//
// right shift mpint by Mpscale
// ignores sign and overflow
//
static void
mprshw(Mpint *a)
{
	long *a1, lo;
	int i;

	lo = a->a[0];
	a1 = &a->a[0];
	for(i=1; i<Mpprec; i++) {
		a1[0] = a1[1];
		a1++;
	}
	a1[0] = 0;
	if(a->neg && lo != 0)
		mpaddcfix(a, -1);
}

//
// return the sign of (abs(a)-abs(b))
//
static int
mpcmp(Mpint *a, Mpint *b)
{
	long x, *a1, *b1;
	int i;

	if(a->ovf || b->ovf) {
		if(nsavederrors+nerrors == 0)
			yyerror("ovf in cmp");
		return 0;
	}

	a1 = &a->a[0] + Mpprec;
	b1 = &b->a[0] + Mpprec;

	for(i=0; i<Mpprec; i++) {
		x = *--a1 - *--b1;
		if(x > 0)
			return +1;
		if(x < 0)
			return -1;
	}
	return 0;
}

//
// negate a
// ignore sign and ovf
//
static void
mpneg(Mpint *a)
{
	long x, *a1;
	int i, c;

	a1 = &a->a[0];
	c = 0;
	for(i=0; i<Mpprec; i++) {
		x = -*a1 -c;
		c = 0;
		if(x < 0) {
			x += Mpbase;
			c = 1;
		}
		*a1++ = x;
	}
}

// shift left by s (or right by -s)
void
mpshiftfix(Mpint *a, int s)
{
	if(s >= 0) {
		while(s >= Mpscale) {
			mplshw(a, 0);
			s -= Mpscale;
		}
		while(s > 0) {
			mplsh(a, 0);
			s--;
		}
	} else {
		s = -s;
		while(s >= Mpscale) {
			mprshw(a);
			s -= Mpscale;
		}
		while(s > 0) {
			mprsh(a);
			s--;
		}
	}
}

/// implements fix arihmetic

void
mpaddfixfix(Mpint *a, Mpint *b, int quiet)
{
	int i, c;
	long x, *a1, *b1;

	if(a->ovf || b->ovf) {
		if(nsavederrors+nerrors == 0)
			yyerror("ovf in mpaddxx");
		a->ovf = 1;
		return;
	}

	c = 0;
	a1 = &a->a[0];
	b1 = &b->a[0];
	if(a->neg != b->neg)
		goto sub;

	// perform a+b
	for(i=0; i<Mpprec; i++) {
		x = *a1 + *b1++ + c;
		c = 0;
		if(x >= Mpbase) {
			x -= Mpbase;
			c = 1;
		}
		*a1++ = x;
	}
	a->ovf = c;
	if(a->ovf && !quiet)
		yyerror("constant addition overflow");

	return;

sub:
	// perform a-b
	switch(mpcmp(a, b)) {
	case 0:
		mpmovecfix(a, 0);
		break;

	case 1:
		for(i=0; i<Mpprec; i++) {
			x = *a1 - *b1++ - c;
			c = 0;
			if(x < 0) {
				x += Mpbase;
				c = 1;
			}
			*a1++ = x;
		}
		break;

	case -1:
		a->neg ^= 1;
		for(i=0; i<Mpprec; i++) {
			x = *b1++ - *a1 - c;
			c = 0;
			if(x < 0) {
				x += Mpbase;
				c = 1;
			}
			*a1++ = x;
		}
		break;
	}
}

void
mpmulfixfix(Mpint *a, Mpint *b)
{

	int i, j, na, nb;
	long *a1, x;
	Mpint s, q;

	if(a->ovf || b->ovf) {
		if(nsavederrors+nerrors == 0)
			yyerror("ovf in mpmulfixfix");
		a->ovf = 1;
		return;
	}

	// pick the smaller
	// to test for bits
	na = mplen(a);
	nb = mplen(b);
	if(na > nb) {
		mpmovefixfix(&s, a);
		a1 = &b->a[0];
		na = nb;
	} else {
		mpmovefixfix(&s, b);
		a1 = &a->a[0];
	}
	s.neg = 0;

	mpmovecfix(&q, 0);
	for(i=0; i<na; i++) {
		x = *a1++;
		for(j=0; j<Mpscale; j++) {
			if(x & 1) {
				if(s.ovf) {
					q.ovf = 1;
					goto out;
				}
				mpaddfixfix(&q, &s, 1);
				if(q.ovf)
					goto out;
			}
			mplsh(&s, 1);
			x >>= 1;
		}
	}

out:
	q.neg = a->neg ^ b->neg;
	mpmovefixfix(a, &q);
	if(a->ovf)
		yyerror("constant multiplication overflow");
}

void
mpmulfract(Mpint *a, Mpint *b)
{

	int i, j;
	long *a1, x;
	Mpint s, q;

	if(a->ovf || b->ovf) {
		if(nsavederrors+nerrors == 0)
			yyerror("ovf in mpmulflt");
		a->ovf = 1;
		return;
	}

	mpmovefixfix(&s, b);
	a1 = &a->a[Mpprec];
	s.neg = 0;
	mpmovecfix(&q, 0);

	x = *--a1;
	if(x != 0)
		yyerror("mpmulfract not normal");

	for(i=0; i<Mpprec-1; i++) {
		x = *--a1;
		if(x == 0) {
			mprshw(&s);
			continue;
		}
		for(j=0; j<Mpscale; j++) {
			x <<= 1;
			if(x & Mpbase)
				mpaddfixfix(&q, &s, 1);
			mprsh(&s);
		}
	}

	q.neg = a->neg ^ b->neg;
	mpmovefixfix(a, &q);
	if(a->ovf)
		yyerror("constant multiplication overflow");
}

void
mporfixfix(Mpint *a, Mpint *b)
{
	int i;
	long x, *a1, *b1;

	x = 0;
	if(a->ovf || b->ovf) {
		if(nsavederrors+nerrors == 0)
			yyerror("ovf in mporfixfix");
		mpmovecfix(a, 0);
		a->ovf = 1;
		return;
	}
	if(a->neg) {
		a->neg = 0;
		mpneg(a);
	}
	if(b->neg)
		mpneg(b);

	a1 = &a->a[0];
	b1 = &b->a[0];
	for(i=0; i<Mpprec; i++) {
		x = *a1 | *b1++;
		*a1++ = x;
	}

	if(b->neg)
		mpneg(b);
	if(x & Mpsign) {
		a->neg = 1;
		mpneg(a);
	}
}

void
mpandfixfix(Mpint *a, Mpint *b)
{
	int i;
	long x, *a1, *b1;

	x = 0;
	if(a->ovf || b->ovf) {
		if(nsavederrors+nerrors == 0)
			yyerror("ovf in mpandfixfix");
		mpmovecfix(a, 0);
		a->ovf = 1;
		return;
	}
	if(a->neg) {
		a->neg = 0;
		mpneg(a);
	}
	if(b->neg)
		mpneg(b);

	a1 = &a->a[0];
	b1 = &b->a[0];
	for(i=0; i<Mpprec; i++) {
		x = *a1 & *b1++;
		*a1++ = x;
	}

	if(b->neg)
		mpneg(b);
	if(x & Mpsign) {
		a->neg = 1;
		mpneg(a);
	}
}

void
mpandnotfixfix(Mpint *a, Mpint *b)
{
	int i;
	long x, *a1, *b1;

	x = 0;
	if(a->ovf || b->ovf) {
		if(nsavederrors+nerrors == 0)
			yyerror("ovf in mpandnotfixfix");
		mpmovecfix(a, 0);
		a->ovf = 1;
		return;
	}
	if(a->neg) {
		a->neg = 0;
		mpneg(a);
	}
	if(b->neg)
		mpneg(b);

	a1 = &a->a[0];
	b1 = &b->a[0];
	for(i=0; i<Mpprec; i++) {
		x = *a1 & ~*b1++;
		*a1++ = x;
	}

	if(b->neg)
		mpneg(b);
	if(x & Mpsign) {
		a->neg = 1;
		mpneg(a);
	}
}

void
mpxorfixfix(Mpint *a, Mpint *b)
{
	int i;
	long x, *a1, *b1;

	x = 0;
	if(a->ovf || b->ovf) {
		if(nsavederrors+nerrors == 0)
			yyerror("ovf in mporfixfix");
		mpmovecfix(a, 0);
		a->ovf = 1;
		return;
	}
	if(a->neg) {
		a->neg = 0;
		mpneg(a);
	}
	if(b->neg)
		mpneg(b);

	a1 = &a->a[0];
	b1 = &b->a[0];
	for(i=0; i<Mpprec; i++) {
		x = *a1 ^ *b1++;
		*a1++ = x;
	}

	if(b->neg)
		mpneg(b);
	if(x & Mpsign) {
		a->neg = 1;
		mpneg(a);
	}
}

void
mplshfixfix(Mpint *a, Mpint *b)
{
	vlong s;

	if(a->ovf || b->ovf) {
		if(nsavederrors+nerrors == 0)
			yyerror("ovf in mporfixfix");
		mpmovecfix(a, 0);
		a->ovf = 1;
		return;
	}
	s = mpgetfix(b);
	if(s < 0 || s >= Mpprec*Mpscale) {
		yyerror("stupid shift: %lld", s);
		mpmovecfix(a, 0);
		return;
	}

	mpshiftfix(a, s);
}

void
mprshfixfix(Mpint *a, Mpint *b)
{
	vlong s;

	if(a->ovf || b->ovf) {
		if(nsavederrors+nerrors == 0)
			yyerror("ovf in mprshfixfix");
		mpmovecfix(a, 0);
		a->ovf = 1;
		return;
	}
	s = mpgetfix(b);
	if(s < 0 || s >= Mpprec*Mpscale) {
		yyerror("stupid shift: %lld", s);
		if(a->neg)
			mpmovecfix(a, -1);
		else
			mpmovecfix(a, 0);
		return;
	}

	mpshiftfix(a, -s);
}

void
mpnegfix(Mpint *a)
{
	a->neg ^= 1;
}

vlong
mpgetfix(Mpint *a)
{
	vlong v;

	if(a->ovf) {
		if(nsavederrors+nerrors == 0)
			yyerror("constant overflow");
		return 0;
	}

	v = (uvlong)a->a[0];
	v |= (uvlong)a->a[1] << Mpscale;
	v |= (uvlong)a->a[2] << (Mpscale+Mpscale);
	if(a->neg)
		v = -(uvlong)v;
	return v;
}

void
mpmovecfix(Mpint *a, vlong c)
{
	int i;
	long *a1;
	vlong x;

	a->neg = 0;
	a->ovf = 0;

	x = c;
	if(x < 0) {
		a->neg = 1;
		x = -(uvlong)x;
	}

	a1 = &a->a[0];
	for(i=0; i<Mpprec; i++) {
		*a1++ = x&Mpmask;
		x >>= Mpscale;
	}
}

void
mpdivmodfixfix(Mpint *q, Mpint *r, Mpint *n, Mpint *d)
{
	int i, ns, ds;

	ns = n->neg;
	ds = d->neg;
	n->neg = 0;
	d->neg = 0;

	mpmovefixfix(r, n);
	mpmovecfix(q, 0);

	// shift denominator until it
	// is larger than numerator
	for(i=0; i<Mpprec*Mpscale; i++) {
		if(mpcmp(d, r) > 0)
			break;
		mplsh(d, 1);
	}

	// if it never happens
	// denominator is probably zero
	if(i >= Mpprec*Mpscale) {
		q->ovf = 1;
		r->ovf = 1;
		n->neg = ns;
		d->neg = ds;
		yyerror("constant division overflow");
		return;
	}

	// shift denominator back creating
	// quotient a bit at a time
	// when done the remaining numerator
	// will be the remainder
	for(; i>0; i--) {
		mplsh(q, 1);
		mprsh(d);
		if(mpcmp(d, r) <= 0) {
			mpaddcfix(q, 1);
			mpsubfixfix(r, d);
		}
	}

	n->neg = ns;
	d->neg = ds;
	r->neg = ns;
	q->neg = ns^ds;
}

static int
mpiszero(Mpint *a)
{
	long *a1;
	int i;
	a1 = &a->a[0] + Mpprec;
	for(i=0; i<Mpprec; i++) {
		if(*--a1 != 0)
			return 0;
	}
	return 1;
}

void
mpdivfract(Mpint *a, Mpint *b)
{
	Mpint n, d;
	int i, j, neg;
	long *a1, x;

	mpmovefixfix(&n, a);	// numerator
	mpmovefixfix(&d, b);	// denominator
	a1 = &a->a[Mpprec];	// quotient

	neg = n.neg ^ d.neg;
	n.neg = 0;
	d.neg = 0;
	for(i=0; i<Mpprec; i++) {
		x = 0;
		for(j=0; j<Mpscale; j++) {
			x <<= 1;
			if(mpcmp(&d, &n) <= 0) {
				if(!mpiszero(&d))
					x |= 1;
				mpsubfixfix(&n, &d);
			}
			mprsh(&d);
		}
		*--a1 = x;
	}
	a->neg = neg;
}

int
mptestfix(Mpint *a)
{
	Mpint b;
	int r;

	mpmovecfix(&b, 0);
	r = mpcmp(a, &b);
	if(a->neg) {
		if(r > 0)
			return -1;
		if(r < 0)
			return +1;
	}
	return r;
}
