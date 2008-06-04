// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	<u.h>
#include	<libc.h>

int	mpatof(char*, double*);
int	mpatov(char *s, vlong *v);

enum
{
	Mpscale	= 29,		/* safely smaller than bits in a long */
	Mpprec	= 36,		/* Mpscale*Mpprec sb > largest fp exp */
	Mpbase	= 1L<<Mpscale,
};

typedef
struct
{
	long	a[Mpprec];
	char	ovf;
} Mp;

static	void	mpint(Mp*, int);
static	void	mppow(Mp*, int, int);
static	void	mpmul(Mp*, int);
static	void	mpadd(Mp*, Mp*);
static	int	mptof(Mp*, double*);

/*
 * convert a string, s, to floating in *d
 * return conversion overflow.
 * required syntax is [+-]d*[.]d*[e[+-]d*]
 */
int
mpatof(char *s, double *d)
{
	Mp a, b;
	int dp, c, f, ef, ex, zer;
	double d1, d2;

	dp = 0;		/* digits after decimal point */
	f = 0;		/* sign */
	ex = 0;		/* exponent */
	zer = 1;	/* zero */
	memset(&a, 0, sizeof(a));
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
			mpint(&b, c-'0');
			mpmul(&a, 10);
			mpadd(&a, &b);
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
	if(a.ovf)
		goto bad;
	if(zer) {
		*d = 0;
		return 0;
	}
	if(dp)
		dp--;
	dp -= ex;
	if(dp > 0) {
		/*
		 * must divide by 10**dp
		 */
		if(mptof(&a, &d1))
			goto bad;

		/*
		 * trial exponent of 8**dp
		 * 8 (being between 5 and 10)
		 * should pick up all underflows
		 * in the division of 5**dp.
		 */
		d2 = frexp(d1, &ex);
		d2 = ldexp(d2, ex-3*dp);
		if(d2 == 0)
			goto bad;

		/*
		 * decompose each 10 into 5*2.
		 * create 5**dp in fixed point
		 * and then play with the exponent
		 * for the remaining 2**dp.
		 * note that 5**dp will overflow
		 * with as few as 134 input digits.
		 */
		mpint(&a, 1);
		mppow(&a, 5, dp);
		if(mptof(&a, &d2))
			goto bad;
		d1 = frexp(d1/d2, &ex);
		d1 = ldexp(d1, ex-dp);
		if(d1 == 0)
			goto bad;
	} else {
		/*
		 * must multiply by 10**|dp| --
		 * just do it in fixed point.
		 */
		mppow(&a, 10, -dp);
		if(mptof(&a, &d1))
			goto bad;
	}
	if(f)
		d1 = -d1;
	*d = d1;
	return 0;

bad:
	return 1;
}

/*
 * convert a to floating in *d
 * return conversion overflow
 */
static int
mptof(Mp *a, double *d)
{
	double f, g;
	long x, *a1;
	int i;

	if(a->ovf)
		return 1;
	a1 = a->a;
	f = ldexp(*a1++, 0);
	for(i=Mpscale; i<Mpprec*Mpscale; i+=Mpscale)
		if(x = *a1++) {
			g = ldexp(x, i);
			/*
			 * NOTE: the test (g==0) is plan9
			 * specific. ansi compliant overflow
			 * is signaled by HUGE and errno==ERANGE.
			 * change this for your particular ldexp.
			 */
			if(g == 0)
				return 1;
			f += g;		/* this could bomb! */
		}
	*d = f;
	return 0;
}

/*
 * return a += b
 */
static void
mpadd(Mp *a, Mp *b)
{
	int i, c;
	long x, *a1, *b1;

	if(b->ovf)
		a->ovf = 1;
	if(a->ovf)
		return;
	c = 0;
	a1 = a->a;
	b1 = b->a;
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
}

/*
 * return a = c
 */
static void
mpint(Mp *a, int c)
{

	memset(a, 0, sizeof(*a));
	a->a[0] = c;
}

/*
 * return a *= c
 */
static void
mpmul(Mp *a, int c)
{
	Mp p;
	int b;
	memmove(&p, a, sizeof(p));
	if(!(c & 1))
		memset(a, 0, sizeof(*a));
	c &= ~1;
	for(b=2; c; b<<=1) {
		mpadd(&p, &p);
		if(c & b) {
			mpadd(a, &p);
			c &= ~b;
		}
	}
}

/*
 * return a *= b**e
 */
static void
mppow(Mp *a, int b, int e)
{
	int b1;

	b1 = b*b;
	b1 = b1*b1;
	while(e >= 4) {
		mpmul(a, b1);
		e -= 4;
		if(a->ovf)
			return;
	}
	while(e > 0) {
		mpmul(a, b);
		e--;
	}
}

/*
 * convert a string, s, to vlong in *v
 * return conversion overflow.
 * required syntax is [0[x]]d*
 */
int
mpatov(char *s, vlong *v)
{
	vlong n, nn;
	int c;
	n = 0;
	c = *s;
	if(c == '0')
		goto oct;
	while(c = *s++) {
		if(c >= '0' && c <= '9')
			nn = n*10 + c-'0';
		else
			goto bad;
		if(n < 0 && nn >= 0)
			goto bad;
		n = nn;
	}
	goto out;
oct:
	s++;
	c = *s;
	if(c == 'x' || c == 'X')
		goto hex;
	while(c = *s++) {
		if(c >= '0' || c <= '7')
			nn = n*8 + c-'0';
		else
			goto bad;
		if(n < 0 && nn >= 0)
			goto bad;
		n = nn;
	}
	goto out;
hex:
	s++;
	while(c = *s++) {
		if(c >= '0' && c <= '9')
			c += 0-'0';
		else
		if(c >= 'a' && c <= 'f')
			c += 10-'a';
		else
		if(c >= 'A' && c <= 'F')
			c += 10-'A';
		else
			goto bad;
		nn = n*16 + c;
		if(n < 0 && nn >= 0)
			goto bad;
		n = nn;
	}
out:
	*v = n;
	return 0;

bad:
	*v = ~0;
	return 1;
}
