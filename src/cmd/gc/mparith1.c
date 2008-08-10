// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go.h"

/// uses arihmetic

int
mpcmpfixfix(Mpint *a, Mpint *b)
{
	Mpint c;

	mpmovefixfix(&c, a);
	mpsubfixfix(&c, b);
	return mptestfix(&c);
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
	Mpflt c;

	mpmovefltflt(&c, a);
	mpsubfltflt(&c, b);
	return mptestflt(&c);
}

int
mpcmpfltc(Mpflt *b, double c)
{
	Mpflt a;

	mpmovecflt(&a, c);
	return mpcmpfltflt(&a, b);
}

void
mpsubfixfix(Mpint *a, Mpint *b)
{
	mpnegfix(b);
	mpaddfixfix(a, b);
	mpnegfix(b);
}

void
mpsubfltflt(Mpflt *a, Mpflt *b)
{
	mpnegflt(b);
	mpaddfltflt(a, b);
	mpnegflt(b);
}

void
mpaddcfix(Mpint *a, vlong c)
{
	Mpint b;

	mpmovecfix(&b, c);
	mpaddfixfix(a, &b);
}

void
mpaddcflt(Mpflt *a, double c)
{
	Mpflt b;

	mpmovecflt(&b, c);
	mpaddfltflt(a, &b);
}

void
mpmulcfix(Mpint *a, vlong c)
{
	Mpint b;

	mpmovecfix(&b, c);
	mpmulfixfix(a, &b);
}

void
mpmulcflt(Mpflt *a, double c)
{
	Mpflt b;

	mpmovecflt(&b, c);
	mpmulfltflt(a, &b);
}

void
mpdivfixfix(Mpint *a, Mpint *b)
{
	Mpint q, r;

	mpdivmodfixfix(&q, &r, a, b);
	mpmovefixfix(a, &q);
}

void
mpmodfixfix(Mpint *a, Mpint *b)
{
	Mpint q, r;

	mpdivmodfixfix(&q, &r, a, b);
	mpmovefixfix(a, &r);
}

void
mpcomfix(Mpint *a)
{
	Mpint b;

	mpmovecfix(&b, 1);
	mpnegfix(a);
	mpsubfixfix(a, &b);
}

void
mpmovefixflt(Mpflt *a, Mpint *b)
{
	mpmovecflt(a, mpgetfix(b));
}

void
mpmovefltfix(Mpint *a, Mpflt *b)
{
	mpmovecfix(a, mpgetflt(b));
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

	mpmovecflt(a, 0.0);
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
