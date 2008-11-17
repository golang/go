// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go.h"

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
// ignores sign and overflow
//
static void
mplsh(Mpint *a)
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
}

//
// left shift mpint by Mpscale
// ignores sign and overflow
//
static void
mplshw(Mpint *a)
{
	long *a1;
	int i;

	a1 = &a->a[Mpprec-1];
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
		warn("ovf in cmp");
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

/// implements fix arihmetic

void
mpaddfixfix(Mpint *a, Mpint *b)
{
	int i, c;
	long x, *a1, *b1;

	if(a->ovf || b->ovf) {
		warn("ovf in mpaddxx");
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
	if(a->ovf)
		warn("set ovf in mpaddxx");

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
		warn("ovf in mpmulfixfix");
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
			if(x & 1)
				mpaddfixfix(&q, &s);
			mplsh(&s);
			x >>= 1;
		}
	}

	q.neg = a->neg ^ b->neg;
	mpmovefixfix(a, &q);
	if(a->ovf)
		warn("set ovf in mpmulfixfix");
}

void
mporfixfix(Mpint *a, Mpint *b)
{
	int i;
	long x, *a1, *b1;

	if(a->ovf || b->ovf) {
		warn("ovf in mporfixfix");
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

	if(a->ovf || b->ovf) {
		warn("ovf in mpandfixfix");
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
mpxorfixfix(Mpint *a, Mpint *b)
{
	int i;
	long x, *a1, *b1;

	if(a->ovf || b->ovf) {
		warn("ovf in mporfixfix");
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
		warn("ovf in mporfixfix");
		mpmovecfix(a, 0);
		a->ovf = 1;
		return;
	}
	s = mpgetfix(b);
	if(s < 0 || s >= Mpprec*Mpscale) {
		warn("stupid shift: %lld", s);
		mpmovecfix(a, 0);
		return;
	}

	while(s >= Mpscale) {
		mplshw(a);
		s -= Mpscale;
	}
	while(s > 0) {
		mplsh(a);
		s--;
	}
}

void
mprshfixfix(Mpint *a, Mpint *b)
{
	vlong s;

	if(a->ovf || b->ovf) {
		warn("ovf in mprshfixfix");
		mpmovecfix(a, 0);
		a->ovf = 1;
		return;
	}
	s = mpgetfix(b);
	if(s < 0 || s >= Mpprec*Mpscale) {
		warn("stupid shift: %lld", s);
		if(a->neg)
			mpmovecfix(a, -1);
		else
			mpmovecfix(a, 0);
		return;
	}

	while(s >= Mpscale) {
		mprshw(a);
		s -= Mpscale;
	}
	while(s > 0) {
		mprsh(a);
		s--;
	}
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
		warn("ovf in mpgetfix");
		return 0;
	}

	v = (vlong)a->a[0];
	v |= (vlong)a->a[1] << Mpscale;
	v |= (vlong)a->a[2] << (Mpscale+Mpscale);
	if(a->neg)
		v = -v;
	return v;
}

double
mpgetfixflt(Mpint *a)
{
	// answer might not fit in intermediate vlong, so format
	// to string and then let the string routine convert.
	char buf[1000];

	snprint(buf, sizeof buf, "%B", a);
	return strtod(buf, nil);
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
		x = -x;
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
	int i;

	mpmovefixfix(r, n);
	mpmovecfix(q, 0);

	// shift denominator until it
	// is larger than numerator
	for(i=0; i<Mpprec*Mpscale; i++) {
		if(mpcmp(d, r) > 0)
			break;
		mplsh(d);
	}

	// if it never happens
	// denominator is probably zero
	if(i >= Mpprec*Mpscale) {
		q->ovf = 1;
		r->ovf = 1;
		warn("set ovf in mpdivmodfixfix");
		return;
	}

	// shift denominator back creating
	// quotient a bit at a time
	// when done the remaining numerator
	// will be the remainder
	for(; i>0; i--) {
		mplsh(q);
		mprsh(d);
		if(mpcmp(d, r) <= 0) {
			mpaddcfix(q, 1);
			mpsubfixfix(r, d);
		}
	}
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
