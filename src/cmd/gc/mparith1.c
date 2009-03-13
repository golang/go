// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"

/// uses arithmetic

int
mpcmpfixflt(Mpint *a, Mpflt *b)
{
	char buf[500];
	Mpflt c;

	snprint(buf, sizeof(buf), "%B", a);
	mpatoflt(&c, buf);
	return mpcmpfltflt(&c, b);
}

int
mpcmpfltfix(Mpflt *a, Mpint *b)
{
	char buf[500];
	Mpflt c;

	snprint(buf, sizeof(buf), "%B", b);
	mpatoflt(&c, buf);
	return mpcmpfltflt(a, &c);
}

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
	mpnegfix(a);
	mpaddfixfix(a, b);
	mpnegfix(a);
}

void
mpsubfltflt(Mpflt *a, Mpflt *b)
{
	mpnegflt(a);
	mpaddfltflt(a, b);
	mpnegflt(a);
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
	a->val = *b;
	a->exp = 0;
	mpnorm(a);
}

// convert (truncate) b to a.
// return -1 (but still convert) if b was non-integer.
int
mpmovefltfix(Mpint *a, Mpflt *b)
{
	Mpflt f;
	*a = b->val;
	mpshiftfix(a, b->exp);
	if(b->exp < 0) {
		f.val = *a;
		f.exp = 0;
		mpnorm(&f);
		if(mpcmpfltflt(b, &f) != 0)
			return -1;
	}
	return 0;
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

static	double	tab[] = { 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7 };
static void
mppow10flt(Mpflt *a, int p)
{
	if(p < nelem(tab)) {
		mpmovecflt(a, tab[p]);
		return;
	}
	mppow10flt(a, p>>1);
	mpmulfltflt(a, a);
	if(p & 1)
		mpmulcflt(a, 10);
}

//
// floating point input
// required syntax is [+-]d*[.]d*[e[+-]d*]
//
void
mpatoflt(Mpflt *a, char *as)
{
	Mpflt b;
	int dp, c, f, ef, ex, eb, zer;
	char *s;

	s = as;
	dp = 0;		/* digits after decimal point */
	f = 0;		/* sign */
	ex = 0;		/* exponent */
	eb = 0;		/* binary point */
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

		case 'P':
		case 'p':
			eb = 1;

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

	if(eb) {
		if(dp)
			goto bad;
		a->exp += ex;
		goto out;
	}

	if(dp)
		dp--;
	if(mpcmpfltc(a, 0.0) != 0) {
		if(ex >= dp) {
			mppow10flt(&b, ex-dp);
			mpmulfltflt(a, &b);
		} else {
			mppow10flt(&b, dp-ex);
			mpdivfltflt(a, &b);
		}
	}

out:
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

int
Bconv(Fmt *fp)
{
	char buf[500], *p;
	Mpint *xval, q, r, ten;
	int f;

	xval = va_arg(fp->args, Mpint*);
	mpmovefixfix(&q, xval);
	f = 0;
	if(mptestfix(&q) < 0) {
		f = 1;
		mpnegfix(&q);
	}
	mpmovecfix(&ten, 10);

	p = &buf[sizeof(buf)];
	*--p = 0;
	for(;;) {
		mpdivmodfixfix(&q, &r, &q, &ten);
		*--p = mpgetfix(&r) + '0';
		if(mptestfix(&q) <= 0)
			break;
	}
	if(f)
		*--p = '-';
	return fmtstrcpy(fp, p);
}

int
Fconv(Fmt *fp)
{
	char buf[500];
	Mpflt *fvp, fv;
	double d;

	fvp = va_arg(fp->args, Mpflt*);
	if(fp->flags & FmtSharp) {
		// alternate form - decimal for error messages.
		// for well in range, convert to double and use print's %g
		if(-900 < fvp->exp && fvp->exp < 900) {
			d = mpgetflt(fvp);
			return fmtprint(fp, "%g", d);
		}
		// TODO(rsc): for well out of range, print
		// an approximation like 1.234e1000
	}

	if(sigfig(fvp) == 0) {
		snprint(buf, sizeof(buf), "0p+0");
		goto out;
	}
	fv = *fvp;

	while(fv.val.a[0] == 0) {
		mpshiftfix(&fv.val, -Mpscale);
		fv.exp += Mpscale;
	}
	while((fv.val.a[0]&1) == 0) {
		mpshiftfix(&fv.val, -1);
		fv.exp += 1;
	}

	if(fv.exp >= 0) {
		snprint(buf, sizeof(buf), "%Bp+%d", &fv.val, fv.exp);
		goto out;
	}
	snprint(buf, sizeof(buf), "%Bp-%d", &fv.val, -fv.exp);

out:
	return fmtstrcpy(fp, buf);
}
