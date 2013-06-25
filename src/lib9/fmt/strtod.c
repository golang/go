/*
 * The authors of this software are Rob Pike and Ken Thompson,
 * with contributions from Mike Burrows and Sean Dorward.
 *
 *     Copyright (c) 2002-2006 by Lucent Technologies.
 *     Portions Copyright (c) 2004 Google Inc.
 * 
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR LUCENT TECHNOLOGIES 
 * NOR GOOGLE INC MAKE ANY REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING 
 * THE MERCHANTABILITY OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 */

#include <u.h>
#include <errno.h>
#include <libc.h>
#include "fmtdef.h"

static ulong
umuldiv(ulong a, ulong b, ulong c)
{
	double d;

	d = ((double)a * (double)b) / (double)c;
	if(d >= 4294967295.)
		d = 4294967295.;
	return (ulong)d;
}

/*
 * This routine will convert to arbitrary precision
 * floating point entirely in multi-precision fixed.
 * The answer is the closest floating point number to
 * the given decimal number. Exactly half way are
 * rounded ala ieee rules.
 * Method is to scale input decimal between .500 and .999...
 * with external power of 2, then binary search for the
 * closest mantissa to this decimal number.
 * Nmant is is the required precision. (53 for ieee dp)
 * Nbits is the max number of bits/word. (must be <= 28)
 * Prec is calculated - the number of words of fixed mantissa.
 */
enum
{
	Nbits	= 28,				/* bits safely represented in a ulong */
	Nmant	= 53,				/* bits of precision required */
	Prec	= (Nmant+Nbits+1)/Nbits,	/* words of Nbits each to represent mantissa */
	Sigbit	= 1<<(Prec*Nbits-Nmant),	/* first significant bit of Prec-th word */
	Ndig	= 1500,
	One	= (ulong)(1<<Nbits),
	Half	= (ulong)(One>>1),
	Maxe	= 310,

	Fsign	= 1<<0,		/* found - */
	Fesign	= 1<<1,		/* found e- */
	Fdpoint	= 1<<2,		/* found . */

	S0	= 0,		/* _		_S0	+S1	#S2	.S3 */
	S1,			/* _+		#S2	.S3 */
	S2,			/* _+#		#S2	.S4	eS5 */
	S3,			/* _+.		#S4 */
	S4,			/* _+#.#	#S4	eS5 */
	S5,			/* _+#.#e	+S6	#S7 */
	S6,			/* _+#.#e+	#S7 */
	S7			/* _+#.#e+#	#S7 */
};

static	int	xcmp(char*, char*);
static	int	fpcmp(char*, ulong*);
static	void	frnorm(ulong*);
static	void	divascii(char*, int*, int*, int*);
static	void	mulascii(char*, int*, int*, int*);

typedef	struct	Tab	Tab;
struct	Tab
{
	int	bp;
	int	siz;
	char*	cmp;
};

double
fmtstrtod(const char *as, char **aas)
{
	int na, ex, dp, bp, c, i, flag, state;
	ulong low[Prec], hig[Prec], mid[Prec];
	double d;
	char *s, a[Ndig];

	flag = 0;	/* Fsign, Fesign, Fdpoint */
	na = 0;		/* number of digits of a[] */
	dp = 0;		/* na of decimal point */
	ex = 0;		/* exonent */

	state = S0;
	for(s=(char*)as;; s++) {
		c = *s;
		if(c >= '0' && c <= '9') {
			switch(state) {
			case S0:
			case S1:
			case S2:
				state = S2;
				break;
			case S3:
			case S4:
				state = S4;
				break;

			case S5:
			case S6:
			case S7:
				state = S7;
				ex = ex*10 + (c-'0');
				continue;
			}
			if(na == 0 && c == '0') {
				dp--;
				continue;
			}
			if(na < Ndig-50)
				a[na++] = (char)c;
			continue;
		}
		switch(c) {
		case '\t':
		case '\n':
		case '\v':
		case '\f':
		case '\r':
		case ' ':
			if(state == S0)
				continue;
			break;
		case '-':
			if(state == S0)
				flag |= Fsign;
			else
				flag |= Fesign;
		case '+':
			if(state == S0)
				state = S1;
			else
			if(state == S5)
				state = S6;
			else
				break;	/* syntax */
			continue;
		case '.':
			flag |= Fdpoint;
			dp = na;
			if(state == S0 || state == S1) {
				state = S3;
				continue;
			}
			if(state == S2) {
				state = S4;
				continue;
			}
			break;
		case 'e':
		case 'E':
			if(state == S2 || state == S4) {
				state = S5;
				continue;
			}
			break;
		}
		break;
	}

	/*
	 * clean up return char-pointer
	 */
	switch(state) {
	case S0:
		if(xcmp(s, "nan") == 0) {
			if(aas != nil)
				*aas = s+3;
			goto retnan;
		}
	case S1:
		if(xcmp(s, "infinity") == 0) {
			if(aas != nil)
				*aas = s+8;
			goto retinf;
		}
		if(xcmp(s, "inf") == 0) {
			if(aas != nil)
				*aas = s+3;
			goto retinf;
		}
	case S3:
		if(aas != nil)
			*aas = (char*)as;
		goto ret0;	/* no digits found */
	case S6:
		s--;		/* back over +- */
	case S5:
		s--;		/* back over e */
		break;
	}
	if(aas != nil)
		*aas = s;

	if(flag & Fdpoint)
	while(na > 0 && a[na-1] == '0')
		na--;
	if(na == 0)
		goto ret0;	/* zero */
	a[na] = 0;
	if(!(flag & Fdpoint))
		dp = na;
	if(flag & Fesign)
		ex = -ex;
	dp += ex;
	if(dp < -Maxe){
		errno = ERANGE;
		goto ret0;	/* underflow by exp */
	} else
	if(dp > +Maxe)
		goto retinf;	/* overflow by exp */

	/*
	 * normalize the decimal ascii number
	 * to range .[5-9][0-9]* e0
	 */
	bp = 0;		/* binary exponent */
	while(dp > 0)
		divascii(a, &na, &dp, &bp);
	while(dp < 0 || a[0] < '5')
		mulascii(a, &na, &dp, &bp);

	/* close approx by naive conversion */
	mid[0] = 0;
	mid[1] = 1;
	for(i=0; (c=a[i]) != '\0'; i++) {
		mid[0] = mid[0]*10 + (ulong)(c-'0');
		mid[1] = mid[1]*10;
		if(i >= 8)
			break;
	}
	low[0] = umuldiv(mid[0], One, mid[1]);
	hig[0] = umuldiv(mid[0]+1, One, mid[1]);
	for(i=1; i<Prec; i++) {
		low[i] = 0;
		hig[i] = One-1;
	}

	/* binary search for closest mantissa */
	for(;;) {
		/* mid = (hig + low) / 2 */
		c = 0;
		for(i=0; i<Prec; i++) {
			mid[i] = hig[i] + low[i];
			if(c)
				mid[i] += One;
			c = mid[i] & 1;
			mid[i] >>= 1;
		}
		frnorm(mid);

		/* compare */
		c = fpcmp(a, mid);
		if(c > 0) {
			c = 1;
			for(i=0; i<Prec; i++)
				if(low[i] != mid[i]) {
					c = 0;
					low[i] = mid[i];
				}
			if(c)
				break;	/* between mid and hig */
			continue;
		}
		if(c < 0) {
			for(i=0; i<Prec; i++)
				hig[i] = mid[i];
			continue;
		}

		/* only hard part is if even/odd roundings wants to go up */
		c = mid[Prec-1] & (Sigbit-1);
		if(c == Sigbit/2 && (mid[Prec-1]&Sigbit) == 0)
			mid[Prec-1] -= (ulong)c;
		break;	/* exactly mid */
	}

	/* normal rounding applies */
	c = mid[Prec-1] & (Sigbit-1);
	mid[Prec-1] -= (ulong)c;
	if(c >= Sigbit/2) {
		mid[Prec-1] += Sigbit;
		frnorm(mid);
	}
	goto out;

ret0:
	return 0;

retnan:
	return __NaN();

retinf:
	/*
	 * Unix strtod requires these.  Plan 9 would return Inf(0) or Inf(-1). */
	errno = ERANGE;
	if(flag & Fsign)
		return -HUGE_VAL;
	return HUGE_VAL;

out:
	d = 0;
	for(i=0; i<Prec; i++)
		d = d*One + (double)mid[i];
	if(flag & Fsign)
		d = -d;
	d = ldexp(d, bp - Prec*Nbits);
	if(d == 0){	/* underflow */
		errno = ERANGE;
	}
	return d;
}

static void
frnorm(ulong *f)
{
	int i;
	ulong c;

	c = 0;
	for(i=Prec-1; i>0; i--) {
		f[i] += c;
		c = f[i] >> Nbits;
		f[i] &= One-1;
	}
	f[0] += c;
}

static int
fpcmp(char *a, ulong* f)
{
	ulong tf[Prec];
	int i, d, c;

	for(i=0; i<Prec; i++)
		tf[i] = f[i];

	for(;;) {
		/* tf *= 10 */
		for(i=0; i<Prec; i++)
			tf[i] = tf[i]*10;
		frnorm(tf);
		d = (int)(tf[0] >> Nbits) + '0';
		tf[0] &= One-1;

		/* compare next digit */
		c = *a;
		if(c == 0) {
			if('0' < d)
				return -1;
			if(tf[0] != 0)
				goto cont;
			for(i=1; i<Prec; i++)
				if(tf[i] != 0)
					goto cont;
			return 0;
		}
		if(c > d)
			return +1;
		if(c < d)
			return -1;
		a++;
	cont:;
	}
}

static void
divby(char *a, int *na, int b)
{
	int n, c;
	char *p;

	p = a;
	n = 0;
	while(n>>b == 0) {
		c = *a++;
		if(c == 0) {
			while(n) {
				c = n*10;
				if(c>>b)
					break;
				n = c;
			}
			goto xx;
		}
		n = n*10 + c-'0';
		(*na)--;
	}
	for(;;) {
		c = n>>b;
		n -= c<<b;
		*p++ = (char)(c + '0');
		c = *a++;
		if(c == 0)
			break;
		n = n*10 + c-'0';
	}
	(*na)++;
xx:
	while(n) {
		n = n*10;
		c = n>>b;
		n -= c<<b;
		*p++ = (char)(c + '0');
		(*na)++;
	}
	*p = 0;
}

static	Tab	tab1[] =
{
	 1,  0, "",
	 3,  1, "7",
	 6,  2, "63",
	 9,  3, "511",
	13,  4, "8191",
	16,  5, "65535",
	19,  6, "524287",
	23,  7, "8388607",
	26,  8, "67108863",
	27,  9, "134217727",
};

static void
divascii(char *a, int *na, int *dp, int *bp)
{
	int b, d;
	Tab *t;

	d = *dp;
	if(d >= (int)(nelem(tab1)))
		d = (int)(nelem(tab1))-1;
	t = tab1 + d;
	b = t->bp;
	if(memcmp(a, t->cmp, (size_t)t->siz) > 0)
		d--;
	*dp -= d;
	*bp += b;
	divby(a, na, b);
}

static void
mulby(char *a, char *p, char *q, int b)
{
	int n, c;

	n = 0;
	*p = 0;
	for(;;) {
		q--;
		if(q < a)
			break;
		c = *q - '0';
		c = (c<<b) + n;
		n = c/10;
		c -= n*10;
		p--;
		*p = (char)(c + '0');
	}
	while(n) {
		c = n;
		n = c/10;
		c -= n*10;
		p--;
		*p = (char)(c + '0');
	}
}

static	Tab	tab2[] =
{
	 1,  1, "",				/* dp = 0-0 */
	 3,  3, "125",
	 6,  5, "15625",
	 9,  7, "1953125",
	13, 10, "1220703125",
	16, 12, "152587890625",
	19, 14, "19073486328125",
	23, 17, "11920928955078125",
	26, 19, "1490116119384765625",
	27, 19, "7450580596923828125",		/* dp 8-9 */
};

static void
mulascii(char *a, int *na, int *dp, int *bp)
{
	char *p;
	int d, b;
	Tab *t;

	d = -*dp;
	if(d >= (int)(nelem(tab2)))
		d = (int)(nelem(tab2))-1;
	t = tab2 + d;
	b = t->bp;
	if(memcmp(a, t->cmp, (size_t)t->siz) < 0)
		d--;
	p = a + *na;
	*bp -= b;
	*dp += d;
	*na += d;
	mulby(a, p+d, p, b);
}

static int
xcmp(char *a, char *b)
{
	int c1, c2;

	while((c1 = *b++) != '\0') {
		c2 = *a++;
		if(isupper(c2))
			c2 = tolower(c2);
		if(c1 != c2)
			return 1;
	}
	return 0;
}
