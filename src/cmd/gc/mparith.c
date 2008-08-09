// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go.h"

int
mpcmpfixfix(Mpint *a, Mpint *b)
{
	if(a->val > b->val)
		return +1;
	if(a->val < b->val)
		return -1;
	return 0;
}

int
mpcmpfixc(Mpint *b, vlong c)
{
	Mpint a;

	mpmovecfix(&a, c);
	return mpcmpfixfix(&a, b);
}

int
mpcmpfltflt(Mpflt *a, Mpflt *b)
{
	if(a->val > b->val)
		return +1;
	if(a->val < b->val)
		return -1;
	return 0;
}

int
mpcmpfltc(Mpint *b, double c)
{
	Mpint a;

	mpmovecflt(&a, c);
	return mpcmpfltflt(&a, b);
}

void
mpaddfixfix(Mpint *a, Mpint *b)
{
	a->val += b->val;
}

void
mpsubfixfix(Mpint *a, Mpint *b)
{
	a->val -= b->val;
}

void
mpmulfixfix(Mpint *a, Mpint *b)
{
	a->val *= b->val;
}

void
mpdivfixfix(Mpint *a, Mpint *b)
{
	a->val /= b->val;
}

void
mpmodfixfix(Mpint *a, Mpint *b)
{
	a->val %= b->val;
}

void
mporfixfix(Mpint *a, Mpint *b)
{
	a->val |= b->val;
}

void
mpandfixfix(Mpint *a, Mpint *b)
{
	a->val &= b->val;
}

void
mpxorfixfix(Mpint *a, Mpint *b)
{
	a->val ^= b->val;
}

void
mplshfixfix(Mpint *a, Mpint *b)
{
	a->val <<= b->val;
}

void
mprshfixfix(Mpint *a, Mpint *b)
{
	a->val >>= b->val;
}

void
mpnegfix(Mpint *a)
{
	a->val = -a->val;
}

void
mpcomfix(Mpint *a)
{
	a->val = ~a->val;
}

void
mpaddfltflt(Mpflt *a, Mpflt *b)
{
	a->val += b->val;
}

void
mpsubfltflt(Mpflt *a, Mpflt *b)
{
	a->val -= b->val;
}

void
mpmulfltflt(Mpflt *a, Mpflt *b)
{
	a->val *= b->val;
}

void
mpdivfltflt(Mpflt *a, Mpflt *b)
{
	a->val /= b->val;
}

vlong
mpgetfix(Mpint *a)
{
	return a->val;
}

double
mpgetflt(Mpflt *a)
{
	return a->val;
}

void
mpmovefixfix(Mpint *a, Mpint *b)
{
	*a = *b;
}

void
mpmovefltflt(Mpflt *a, Mpflt *b)
{
	*a = *b;
}

void
mpmovefixflt(Mpflt *a, Mpint *b)
{
	a->val = b->val;
}

void
mpmovecfix(Mpint *a, vlong c)
{
	a->val = c;
}

void
mpmovecflt(Mpflt *a, double c)
{
	a->val = c;
}

void
mpmovefltfix(Mpint *a, Mpflt *b)
{
	a->val = b->val;
}

void
mpnegflt(Mpflt *a)
{
	a->val = -a->val;
}

void
mpaddcflt(Mpflt *a, double c)
{
	Mpflt b;

	mpmovecflt(&b, c);
	mpaddfltflt(a, &b);
}

void
mpmulcflt(Mpflt *a, double c)
{
	Mpflt b;

	mpmovecflt(&b, c);
	mpmulfltflt(a, &b);
}

void
mpaddcfix(Mpint *a, vlong c)
{
	Mpint b;

	mpmovecfix(&b, c);
	mpaddfixfix(a, &b);
}

void
mpmulcfix(Mpint *a, vlong c)
{
	Mpint b;

	mpmovecfix(&b, c);
	mpmulfixfix(a, &b);
}

//
// power of ten
//
static	double
tentab[] = { 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9 };

static double
dppow10(int n)
{
	int i;

	if(n < 0)
		return 1.0/dppow10(-n);

	if(n < nelem(tentab))
		return tentab[n];

	i = n/2;
	return dppow10(i) * dppow10(n-i);
}

//
// floating point input
// required syntax is [+-]d*[.]d*[e[+-]d*]
//
void
mpatoflt(Mpflt *a, char *as)
{
	int dp, c, f, ef, ex, zer;
	char *s;

	s = as;
	dp = 0;		/* digits after decimal point */
	f = 0;		/* sign */
	ex = 0;		/* exponent */
	zer = 1;	/* zero */

	mpmovecflt(a, 0);
	for(;;) {
		switch(c = *s++) {
		default:
			goto bad;

		case '-':
			f = 1;

		case ' ':
		case  '\t':
		case  '+':
			continue;

		case '.':
			dp = 1;
			continue;

		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7':
		case '8':
		case '9':
			zer = 0;

		case '0':
			mpmulcflt(a, 10);
			mpaddcflt(a, c-'0');
			if(dp)
				dp++;
			continue;

		case 'E':
		case 'e':
			ex = 0;
			ef = 0;
			for(;;) {
				c = *s++;
				if(c == '+' || c == ' ' || c == '\t')
					continue;
				if(c == '-') {
					ef = 1;
					continue;
				}
				if(c >= '0' && c <= '9') {
					ex = ex*10 + (c-'0');
					continue;
				}
				break;
			}
			if(ef)
				ex = -ex;

		case 0:
			break;
		}
		break;
	}

	if(dp)
		dp--;
	if(mpcmpfltc(a, 0.0) != 0)
		mpmulcflt(a, dppow10(ex-dp));
	if(f)
		mpnegflt(a);
	return;

bad:
	warn("set ovf in mpatof");
	mpmovecflt(a, 0.0);
}

//
// fixed point input
// required syntax is [+-][0[x]]d*
// 
void
mpatofix(Mpint *a, char *as)
{

	int c, f;
	char *s;

	s = as;
	f = 0;
	mpmovecfix(a, 0);

	c = *s++;
	switch(c) {
	case '-':
		f = 1;

	case '+':
		c = *s++;
		if(c != '0')
			break;

	case '0':
		goto oct;
	}

	while(c) {
		if(c >= '0' && c <= '9') {
			mpmulcfix(a, 10);
			mpaddcfix(a, c-'0');
			c = *s++;
			continue;
		}
		goto bad;
	}
	goto out;

oct:
	c = *s++;
	if(c == 'x' || c == 'X')
		goto hex;
	while(c) {
		if(c >= '0' && c <= '7') {
			mpmulcfix(a, 8);
			mpaddcfix(a, c-'0');
			c = *s++;
			continue;
		}
		goto bad;
	}
	goto out;

hex:
	c = *s++;
	while(c) {
		if(c >= '0' && c <= '9') {
			mpmulcfix(a, 16);
			mpaddcfix(a, c-'0');
			c = *s++;
			continue;
		}
		if(c >= 'a' && c <= 'f') {
			mpmulcfix(a, 16);
			mpaddcfix(a, c+10-'a');
			c = *s++;
			continue;
		}
		if(c >= 'A' && c <= 'F') {
			mpmulcfix(a, 16);
			mpaddcfix(a, c+10-'A');
			c = *s++;
			continue;
		}
		goto bad;
	}

out:
	if(f)
		mpnegfix(a);
	return;

bad:
	warn("set ovf in mpatov: %s", as);
	mpmovecfix(a, 0);
}
