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

/* Copyright (c) 2002-2006 Lucent Technologies; see LICENSE */
#include <u.h>
#include <errno.h>
#include <libc.h>
#include "fmtdef.h"

enum
{
	FDIGIT	= 30,
	FDEFLT	= 6,
	NSIGNIF	= 17
};

/*
 * first few powers of 10, enough for about 1/2 of the
 * total space for doubles.
 */
static double pows10[] =
{
	  1e0,   1e1,   1e2,   1e3,   1e4,   1e5,   1e6,   1e7,   1e8,   1e9,
	 1e10,  1e11,  1e12,  1e13,  1e14,  1e15,  1e16,  1e17,  1e18,  1e19,
	 1e20,  1e21,  1e22,  1e23,  1e24,  1e25,  1e26,  1e27,  1e28,  1e29,
	 1e30,  1e31,  1e32,  1e33,  1e34,  1e35,  1e36,  1e37,  1e38,  1e39,
	 1e40,  1e41,  1e42,  1e43,  1e44,  1e45,  1e46,  1e47,  1e48,  1e49,
	 1e50,  1e51,  1e52,  1e53,  1e54,  1e55,  1e56,  1e57,  1e58,  1e59,
	 1e60,  1e61,  1e62,  1e63,  1e64,  1e65,  1e66,  1e67,  1e68,  1e69,
	 1e70,  1e71,  1e72,  1e73,  1e74,  1e75,  1e76,  1e77,  1e78,  1e79,
	 1e80,  1e81,  1e82,  1e83,  1e84,  1e85,  1e86,  1e87,  1e88,  1e89,
	 1e90,  1e91,  1e92,  1e93,  1e94,  1e95,  1e96,  1e97,  1e98,  1e99,
	1e100, 1e101, 1e102, 1e103, 1e104, 1e105, 1e106, 1e107, 1e108, 1e109,
	1e110, 1e111, 1e112, 1e113, 1e114, 1e115, 1e116, 1e117, 1e118, 1e119,
	1e120, 1e121, 1e122, 1e123, 1e124, 1e125, 1e126, 1e127, 1e128, 1e129,
	1e130, 1e131, 1e132, 1e133, 1e134, 1e135, 1e136, 1e137, 1e138, 1e139,
	1e140, 1e141, 1e142, 1e143, 1e144, 1e145, 1e146, 1e147, 1e148, 1e149,
	1e150, 1e151, 1e152, 1e153, 1e154, 1e155, 1e156, 1e157, 1e158, 1e159,
};

#undef	pow10
#define	npows10 ((int)(sizeof(pows10)/sizeof(pows10[0])))
#define	pow10(x)  fmtpow10(x)

static double
pow10(int n)
{
	double d;
	int neg;

	neg = 0;
	if(n < 0){
		neg = 1;
		n = -n;
	}

	if(n < npows10)
		d = pows10[n];
	else{
		d = pows10[npows10-1];
		for(;;){
			n -= npows10 - 1;
			if(n < npows10){
				d *= pows10[n];
				break;
			}
			d *= pows10[npows10 - 1];
		}
	}
	if(neg)
		return 1./d;
	return d;
}

/*
 * add 1 to the decimal integer string a of length n.
 * if 99999 overflows into 10000, return 1 to tell caller
 * to move the virtual decimal point.
 */
static int
xadd1(char *a, int n)
{
	char *b;
	int c;

	if(n < 0 || n > NSIGNIF)
		return 0;
	for(b = a+n-1; b >= a; b--) {
		c = *b + 1;
		if(c <= '9') {
			*b = c;
			return 0;
		}
		*b = '0';
	}
	/*
	 * need to overflow adding digit.
	 * shift number down and insert 1 at beginning.
	 * decimal is known to be 0s or we wouldn't
	 * have gotten this far.  (e.g., 99999+1 => 00000)
	 */
	a[0] = '1';
	return 1;
}

/*
 * subtract 1 from the decimal integer string a.
 * if 10000 underflows into 09999, make it 99999
 * and return 1 to tell caller to move the virtual
 * decimal point.  this way, xsub1 is inverse of xadd1.
 */
static int
xsub1(char *a, int n)
{
	char *b;
	int c;

	if(n < 0 || n > NSIGNIF)
		return 0;
	for(b = a+n-1; b >= a; b--) {
		c = *b - 1;
		if(c >= '0') {
			if(c == '0' && b == a) {
				/*
				 * just zeroed the top digit; shift everyone up.
				 * decimal is known to be 9s or we wouldn't
				 * have gotten this far.  (e.g., 10000-1 => 09999)
				 */
				*b = '9';
				return 1;
			}
			*b = c;
			return 0;
		}
		*b = '9';
	}
	/*
	 * can't get here.  the number a is always normalized
	 * so that it has a nonzero first digit.
	 */
	abort();
}

/*
 * format exponent like sprintf(p, "e%+02d", e)
 */
static void
xfmtexp(char *p, int e, int ucase)
{
	char se[9];
	int i;

	*p++ = ucase ? 'E' : 'e';
	if(e < 0) {
		*p++ = '-';
		e = -e;
	} else
		*p++ = '+';
	i = 0;
	while(e) {
		se[i++] = e % 10 + '0';
		e /= 10;
	}
	while(i < 2)
		se[i++] = '0';
	while(i > 0)
		*p++ = se[--i];
	*p++ = '\0';
}

/*
 * compute decimal integer m, exp such that:
 *	f = m*10^exp
 *	m is as short as possible with losing exactness
 * assumes special cases (NaN, +Inf, -Inf) have been handled.
 */
static void
xdtoa(double f, char *s, int *exp, int *neg, int *ns)
{
	int c, d, e2, e, ee, i, ndigit, oerrno;
	char tmp[NSIGNIF+10];
	double g;

	oerrno = errno; /* in case strtod smashes errno */

	/*
	 * make f non-negative.
	 */
	*neg = 0;
	if(f < 0) {
		f = -f;
		*neg = 1;
	}

	/*
	 * must handle zero specially.
	 */
	if(f == 0){
		*exp = 0;
		s[0] = '0';
		s[1] = '\0';
		*ns = 1;
		return;
	}

	/*
	 * find g,e such that f = g*10^e.
	 * guess 10-exponent using 2-exponent, then fine tune.
	 */
	frexp(f, &e2);
	e = (int)(e2 * .301029995664);
	g = f * pow10(-e);
	while(g < 1) {
		e--;
		g = f * pow10(-e);
	}
	while(g >= 10) {
		e++;
		g = f * pow10(-e);
	}

	/*
	 * convert NSIGNIF digits as a first approximation.
	 */
	for(i=0; i<NSIGNIF; i++) {
		d = (int)g;
		s[i] = d+'0';
		g = (g-d) * 10;
	}
	s[i] = 0;

	/*
	 * adjust e because s is 314159... not 3.14159...
	 */
	e -= NSIGNIF-1;
	xfmtexp(s+NSIGNIF, e, 0);

	/*
	 * adjust conversion until strtod(s) == f exactly.
	 */
	for(i=0; i<10; i++) {
		g = strtod(s, nil);
		if(f > g) {
			if(xadd1(s, NSIGNIF)) {
				/* gained a digit */
				e--;
				xfmtexp(s+NSIGNIF, e, 0);
			}
			continue;
		}
		if(f < g) {
			if(xsub1(s, NSIGNIF)) {
				/* lost a digit */
				e++;
				xfmtexp(s+NSIGNIF, e, 0);
			}
			continue;
		}
		break;
	}

	/*
	 * play with the decimal to try to simplify.
	 */

	/*
	 * bump last few digits up to 9 if we can
	 */
	for(i=NSIGNIF-1; i>=NSIGNIF-3; i--) {
		c = s[i];
		if(c != '9') {
			s[i] = '9';
			g = strtod(s, nil);
			if(g != f) {
				s[i] = c;
				break;
			}
		}
	}

	/*
	 * add 1 in hopes of turning 9s to 0s
	 */
	if(s[NSIGNIF-1] == '9') {
		strcpy(tmp, s);
		ee = e;
		if(xadd1(tmp, NSIGNIF)) {
			ee--;
			xfmtexp(tmp+NSIGNIF, ee, 0);
		}
		g = strtod(tmp, nil);
		if(g == f) {
			strcpy(s, tmp);
			e = ee;
		}
	}

	/*
	 * bump last few digits down to 0 as we can.
	 */
	for(i=NSIGNIF-1; i>=NSIGNIF-3; i--) {
		c = s[i];
		if(c != '0') {
			s[i] = '0';
			g = strtod(s, nil);
			if(g != f) {
				s[i] = c;
				break;
			}
		}
	}

	/*
	 * remove trailing zeros.
	 */
	ndigit = NSIGNIF;
	while(ndigit > 1 && s[ndigit-1] == '0'){
		e++;
		--ndigit;
	}
	s[ndigit] = 0;
	*exp = e;
	*ns = ndigit;
	errno = oerrno;
}

#ifdef PLAN9PORT
static char *special[] = { "NaN", "NaN", "+Inf", "+Inf", "-Inf", "-Inf" };
#else
static char *special[] = { "nan", "NAN", "inf", "INF", "-inf", "-INF" };
#endif

int
__efgfmt(Fmt *fmt)
{
	char buf[NSIGNIF+10], *dot, *digits, *p, *s, suf[10], *t;
	double f;
	int c, chr, dotwid, e, exp, fl, ndigits, neg, newndigits;
	int pad, point, prec, realchr, sign, sufwid, ucase, wid, z1, z2;
	Rune r, *rs, *rt;

	if(fmt->flags&FmtLong)
		f = va_arg(fmt->args, long double);
	else
		f = va_arg(fmt->args, double);

	/*
	 * extract formatting flags
	 */
	fl = fmt->flags;
	fmt->flags = 0;
	prec = FDEFLT;
	if(fl & FmtPrec)
		prec = fmt->prec;
	chr = fmt->r;
	ucase = 0;
	switch(chr) {
	case 'A':
	case 'E':
	case 'F':
	case 'G':
		chr += 'a'-'A';
		ucase = 1;
		break;
	}

	/*
	 * pick off special numbers.
	 */
	if(__isNaN(f)) {
		s = special[0+ucase];
	special:
		fmt->flags = fl & (FmtWidth|FmtLeft);
		return __fmtcpy(fmt, s, strlen(s), strlen(s));
	}
	if(__isInf(f, 1)) {
		s = special[2+ucase];
		goto special;
	}
	if(__isInf(f, -1)) {
		s = special[4+ucase];
		goto special;
	}

	/*
	 * get exact representation.
	 */
	digits = buf;
	xdtoa(f, digits, &exp, &neg, &ndigits);

	/*
	 * get locale's decimal point.
	 */
	dot = fmt->decimal;
	if(dot == nil)
		dot = ".";
	dotwid = utflen(dot);

	/*
	 * now the formatting fun begins.
	 * compute parameters for actual fmt:
	 *
	 *	pad: number of spaces to insert before/after field.
	 *	z1: number of zeros to insert before digits
	 *	z2: number of zeros to insert after digits
	 *	point: number of digits to print before decimal point
	 *	ndigits: number of digits to use from digits[]
	 *	suf: trailing suffix, like "e-5"
	 */
	realchr = chr;
	switch(chr){
	case 'g':
		/*
		 * convert to at most prec significant digits. (prec=0 means 1)
		 */
		if(prec == 0)
			prec = 1;
		if(ndigits > prec) {
			if(digits[prec] >= '5' && xadd1(digits, prec))
				exp++;
			exp += ndigits-prec;
			ndigits = prec;
		}

		/*
		 * extra rules for %g (implemented below):
		 *	trailing zeros removed after decimal unless FmtSharp.
		 *	decimal point only if digit follows.
		 */

		/* fall through to %e */
	default:
	case 'e':
		/*
		 * one significant digit before decimal, no leading zeros.
		 */
		point = 1;
		z1 = 0;

		/*
		 * decimal point is after ndigits digits right now.
		 * slide to be after first.
		 */
		e  = exp + (ndigits-1);

		/*
		 * if this is %g, check exponent and convert prec
		 */
		if(realchr == 'g') {
			if(-4 <= e && e < prec)
				goto casef;
			prec--;	/* one digit before decimal; rest after */
		}

		/*
		 * compute trailing zero padding or truncate digits.
		 */
		if(1+prec >= ndigits)
			z2 = 1+prec - ndigits;
		else {
			/*
			 * truncate digits
			 */
			assert(realchr != 'g');
			newndigits = 1+prec;
			if(digits[newndigits] >= '5' && xadd1(digits, newndigits)) {
				/*
				 * had 999e4, now have 100e5
				 */
				e++;
			}
			ndigits = newndigits;
			z2 = 0;
		}
		xfmtexp(suf, e, ucase);
		sufwid = strlen(suf);
		break;

	casef:
	case 'f':
		/*
		 * determine where digits go with respect to decimal point
		 */
		if(ndigits+exp > 0) {
			point = ndigits+exp;
			z1 = 0;
		} else {
			point = 1;
			z1 = 1 + -(ndigits+exp);
		}

		/*
		 * %g specifies prec = number of significant digits
		 * convert to number of digits after decimal point
		 */
		if(realchr == 'g')
			prec += z1 - point;

		/*
		 * compute trailing zero padding or truncate digits.
		 */
		if(point+prec >= z1+ndigits)
			z2 = point+prec - (z1+ndigits);
		else {
			/*
			 * truncate digits
			 */
			assert(realchr != 'g');
			newndigits = point+prec - z1;
			if(newndigits < 0) {
				z1 += newndigits;
				newndigits = 0;
			} else if(newndigits == 0) {
				/* perhaps round up */
				if(digits[0] >= '5'){
					digits[0] = '1';
					newndigits = 1;
					goto newdigit;
				}
			} else if(digits[newndigits] >= '5' && xadd1(digits, newndigits)) {
				/*
				 * digits was 999, is now 100; make it 1000
				 */
				digits[newndigits++] = '0';
			newdigit:
				/*
				 * account for new digit
				 */
				if(z1)	/* 0.099 => 0.100 or 0.99 => 1.00*/
					z1--;
				else	/* 9.99 => 10.00 */
					point++;
			}
			z2 = 0;
			ndigits = newndigits;
		}
		sufwid = 0;
		break;
	}

	/*
	 * if %g is given without FmtSharp, remove trailing zeros.
	 * must do after truncation, so that e.g. print %.3g 1.001
	 * produces 1, not 1.00.  sorry, but them's the rules.
	 */
	if(realchr == 'g' && !(fl & FmtSharp)) {
		if(z1+ndigits+z2 >= point) {
			if(z1+ndigits < point)
				z2 = point - (z1+ndigits);
			else{
				z2 = 0;
				while(z1+ndigits > point && digits[ndigits-1] == '0')
					ndigits--;
			}
		}
	}

	/*
	 * compute width of all digits and decimal point and suffix if any
	 */
	wid = z1+ndigits+z2;
	if(wid > point)
		wid += dotwid;
	else if(wid == point){
		if(fl & FmtSharp)
			wid += dotwid;
		else
			point++;	/* do not print any decimal point */
	}
	wid += sufwid;

	/*
	 * determine sign
	 */
	sign = 0;
	if(neg)
		sign = '-';
	else if(fl & FmtSign)
		sign = '+';
	else if(fl & FmtSpace)
		sign = ' ';
	if(sign)
		wid++;

	/*
	 * compute padding
	 */
	pad = 0;
	if((fl & FmtWidth) && fmt->width > wid)
		pad = fmt->width - wid;
	if(pad && !(fl & FmtLeft) && (fl & FmtZero)){
		z1 += pad;
		point += pad;
		pad = 0;
	}

	/*
	 * format the actual field.  too bad about doing this twice.
	 */
	if(fmt->runes){
		if(pad && !(fl & FmtLeft) && __rfmtpad(fmt, pad) < 0)
			return -1;
		rt = (Rune*)fmt->to;
		rs = (Rune*)fmt->stop;
		if(sign)
			FMTRCHAR(fmt, rt, rs, sign);
		while(z1>0 || ndigits>0 || z2>0) {
			if(z1 > 0){
				z1--;
				c = '0';
			}else if(ndigits > 0){
				ndigits--;
				c = *digits++;
			}else{
				z2--;
				c = '0';
			}
			FMTRCHAR(fmt, rt, rs, c);
			if(--point == 0) {
				for(p = dot; *p; ){
					p += chartorune(&r, p);
					FMTRCHAR(fmt, rt, rs, r);
				}
			}
		}
		fmt->nfmt += rt - (Rune*)fmt->to;
		fmt->to = rt;
		if(sufwid && __fmtcpy(fmt, suf, sufwid, sufwid) < 0)
			return -1;
		if(pad && (fl & FmtLeft) && __rfmtpad(fmt, pad) < 0)
			return -1;
	}else{
		if(pad && !(fl & FmtLeft) && __fmtpad(fmt, pad) < 0)
			return -1;
		t = (char*)fmt->to;
		s = (char*)fmt->stop;
		if(sign)
			FMTCHAR(fmt, t, s, sign);
		while(z1>0 || ndigits>0 || z2>0) {
			if(z1 > 0){
				z1--;
				c = '0';
			}else if(ndigits > 0){
				ndigits--;
				c = *digits++;
			}else{
				z2--;
				c = '0';
			}
			FMTCHAR(fmt, t, s, c);
			if(--point == 0)
				for(p=dot; *p; p++)
					FMTCHAR(fmt, t, s, *p);
		}
		fmt->nfmt += t - (char*)fmt->to;
		fmt->to = t;
		if(sufwid && __fmtcpy(fmt, suf, sufwid, sufwid) < 0)
			return -1;
		if(pad && (fl & FmtLeft) && __fmtpad(fmt, pad) < 0)
			return -1;
	}
	return 0;
}

