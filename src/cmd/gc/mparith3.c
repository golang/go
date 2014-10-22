// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	<u.h>
#include	<libc.h>
#include	"go.h"

/*
 * returns the leading non-zero
 * word of the number
 */
int
sigfig(Mpflt *a)
{
	int i;

	for(i=Mpprec-1; i>=0; i--)
		if(a->val.a[i] != 0)
			break;
//print("sigfig %d %d\n", i-z+1, z);
	return i+1;
}

/*
 * sets the exponent.
 * a too large exponent is an error.
 * a too small exponent rounds the number to zero.
 */
void
mpsetexp(Mpflt *a, int exp) {
	if((short)exp != exp) {
		if(exp > 0) {
			yyerror("float constant is too large");
			a->exp = 0x7fff;
		}
		else {
			mpmovecflt(a, 0);
		}
	}
	else {
		a->exp = exp;
	}
}

/*
 * shifts the leading non-zero
 * word of the number to Mpnorm
 */
void
mpnorm(Mpflt *a)
{
	int s, os;
	long x;

	os = sigfig(a);
	if(os == 0) {
		// zero
		a->exp = 0;
		a->val.neg = 0;
		return;
	}

	// this will normalize to the nearest word
	x = a->val.a[os-1];
	s = (Mpnorm-os) * Mpscale;

	// further normalize to the nearest bit
	for(;;) {
		x <<= 1;
		if(x & Mpbase)
			break;
		s++;
		if(x == 0) {
			// this error comes from trying to
			// convert an Inf or something
			// where the initial x=0x80000000
			s = (Mpnorm-os) * Mpscale;
			break;
		}
	}

	mpshiftfix(&a->val, s);
	mpsetexp(a, a->exp-s);
}

/// implements float arihmetic

void
mpaddfltflt(Mpflt *a, Mpflt *b)
{
	int sa, sb, s;
	Mpflt c;

	if(Mpdebug)
		print("\n%F + %F", a, b);

	sa = sigfig(a);
	if(sa == 0) {
		mpmovefltflt(a, b);
		goto out;
	}

	sb = sigfig(b);
	if(sb == 0)
		goto out;

	s = a->exp - b->exp;
	if(s > 0) {
		// a is larger, shift b right
		mpmovefltflt(&c, b);
		mpshiftfix(&c.val, -s);
		mpaddfixfix(&a->val, &c.val, 0);
		goto out;
	}
	if(s < 0) {
		// b is larger, shift a right
		mpshiftfix(&a->val, s);
		mpsetexp(a, a->exp-s);
		mpaddfixfix(&a->val, &b->val, 0);
		goto out;
	}
	mpaddfixfix(&a->val, &b->val, 0);

out:
	mpnorm(a);
	if(Mpdebug)
		print(" = %F\n\n", a);
}

void
mpmulfltflt(Mpflt *a, Mpflt *b)
{
	int sa, sb;

	if(Mpdebug)
		print("%F\n * %F\n", a, b);

	sa = sigfig(a);
	if(sa == 0) {
		// zero
		a->exp = 0;
		a->val.neg = 0;
		return;
	}

	sb = sigfig(b);
	if(sb == 0) {
		// zero
		mpmovefltflt(a, b);
		return;
	}

	mpmulfract(&a->val, &b->val);
	mpsetexp(a, (a->exp + b->exp) + Mpscale*Mpprec - Mpscale - 1);

	mpnorm(a);
	if(Mpdebug)
		print(" = %F\n\n", a);
}

void
mpdivfltflt(Mpflt *a, Mpflt *b)
{
	int sa, sb;
	Mpflt c;

	if(Mpdebug)
		print("%F\n / %F\n", a, b);

	sb = sigfig(b);
	if(sb == 0) {
		// zero and ovfl
		a->exp = 0;
		a->val.neg = 0;
		a->val.ovf = 1;
		yyerror("constant division by zero");
		return;
	}

	sa = sigfig(a);
	if(sa == 0) {
		// zero
		a->exp = 0;
		a->val.neg = 0;
		return;
	}

	// adjust b to top
	mpmovefltflt(&c, b);
	mpshiftfix(&c.val, Mpscale);

	// divide
	mpdivfract(&a->val, &c.val);
	mpsetexp(a, (a->exp-c.exp) - Mpscale*(Mpprec-1) + 1);

	mpnorm(a);
	if(Mpdebug)
		print(" = %F\n\n", a);
}

static double
mpgetfltN(Mpflt *a, int prec, int bias)
{
	int s, i, e, minexp;
	uvlong v;
	double f;

	if(a->val.ovf && nsavederrors+nerrors == 0)
		yyerror("mpgetflt ovf");

	s = sigfig(a);
	if(s == 0)
		return 0;

	if(s != Mpnorm) {
		yyerror("mpgetflt norm");
		mpnorm(a);
	}

	while((a->val.a[Mpnorm-1] & Mpsign) == 0) {
		mpshiftfix(&a->val, 1);
		mpsetexp(a, a->exp-1);	// can set 'a' to zero
		s = sigfig(a);
		if(s == 0)
			return 0;
	}

	// pick up the mantissa, a rounding bit, and a tie-breaking bit in a uvlong
	s = prec+2;
	v = 0;
	for(i=Mpnorm-1; s>=Mpscale; i--) {
		v = (v<<Mpscale) | a->val.a[i];
		s -= Mpscale;
	}
	if(s > 0) {
		v = (v<<s) | (a->val.a[i]>>(Mpscale-s));
		if((a->val.a[i]&((1<<(Mpscale-s))-1)) != 0)
			v |= 1;
		i--;
	}
	for(; i >= 0; i--) {
		if(a->val.a[i] != 0)
			v |= 1;
	}

	// gradual underflow
	e = Mpnorm*Mpscale + a->exp - prec;
	minexp = bias+1-prec+1;
	if(e < minexp) {
		s = minexp - e;
		if(s > prec+1)
			s = prec+1;
		if((v & ((1ULL<<s)-1)) != 0)
			v |= 1ULL<<s;
		v >>= s;
		e = minexp;
	}
	
	// round to even
	v |= (v&4)>>2;
	v += v&1;
	v >>= 2;

	f = (double)(v);
	f = ldexp(f, e);

	if(a->val.neg)
		f = -f;

	return f;
}

double
mpgetflt(Mpflt *a)
{
	return mpgetfltN(a, 53, -1023);
}

double
mpgetflt32(Mpflt *a)
{
	return mpgetfltN(a, 24, -127);
}

void
mpmovecflt(Mpflt *a, double c)
{
	int i;
	double f;
	long l;

	if(Mpdebug)
		print("\nconst %g", c);
	mpmovecfix(&a->val, 0);
	a->exp = 0;
	if(c == 0)
		goto out;
	if(c < 0) {
		a->val.neg = 1;
		c = -c;
	}

	f = frexp(c, &i);
	a->exp = i;

	for(i=0; i<10; i++) {
		f = f*Mpbase;
		l = floor(f);
		f = f - l;
		a->exp -= Mpscale;
		a->val.a[0] = l;
		if(f == 0)
			break;
		mpshiftfix(&a->val, Mpscale);
	}

out:
	mpnorm(a);
	if(Mpdebug)
		print(" = %F\n", a);
}

void
mpnegflt(Mpflt *a)
{
	a->val.neg ^= 1;
}

int
mptestflt(Mpflt *a)
{
	int s;

	if(Mpdebug)
		print("\n%F?", a);
	s = sigfig(a);
	if(s != 0) {
		s = +1;
		if(a->val.neg)
			s = -1;
	}
	if(Mpdebug)
		print(" = %d\n", s);
	return s;
}
