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
#include <libc.h>
#include "fmtdef.h"

/* format the output into f->to and return the number of characters fmted  */
int
dofmt(Fmt *f, char *fmt)
{
	Rune rune, *rt, *rs;
	Rune r;
	char *t, *s;
	int n, nfmt;

	nfmt = f->nfmt;
	for(;;){
		if(f->runes){
			rt = (Rune*)f->to;
			rs = (Rune*)f->stop;
			while((r = (Rune)*(uchar*)fmt) && r != '%'){
				if(r < Runeself)
					fmt++;
				else{
					fmt += chartorune(&rune, fmt);
					r = rune;
				}
				FMTRCHAR(f, rt, rs, r);
			}
			fmt++;
			f->nfmt += (int)(rt - (Rune *)f->to);
			f->to = rt;
			if(!r)
				return f->nfmt - nfmt;
			f->stop = rs;
		}else{
			t = (char*)f->to;
			s = (char*)f->stop;
			while((r = (Rune)*(uchar*)fmt) && r != '%'){
				if(r < Runeself){
					FMTCHAR(f, t, s, r);
					fmt++;
				}else{
					n = chartorune(&rune, fmt);
					if(t + n > s){
						t = (char*)__fmtflush(f, t, n);
						if(t != nil)
							s = (char*)f->stop;
						else
							return -1;
					}
					while(n--)
						*t++ = *fmt++;
				}
			}
			fmt++;
			f->nfmt += (int)(t - (char *)f->to);
			f->to = t;
			if(!r)
				return f->nfmt - nfmt;
			f->stop = s;
		}

		fmt = (char*)__fmtdispatch(f, fmt, 0);
		if(fmt == nil)
			return -1;
	}
}

void *
__fmtflush(Fmt *f, void *t, int len)
{
	if(f->runes)
		f->nfmt += (int)((Rune*)t - (Rune*)f->to);
	else
		f->nfmt += (int)((char*)t - (char *)f->to);
	f->to = t;
	if(f->flush == 0 || (*f->flush)(f) == 0 || (char*)f->to + len > (char*)f->stop){
		f->stop = f->to;
		return nil;
	}
	return f->to;
}

/*
 * put a formatted block of memory sz bytes long of n runes into the output buffer,
 * left/right justified in a field of at least f->width characters (if FmtWidth is set)
 */
int
__fmtpad(Fmt *f, int n)
{
	char *t, *s;
	int i;

	t = (char*)f->to;
	s = (char*)f->stop;
	for(i = 0; i < n; i++)
		FMTCHAR(f, t, s, ' ');
	f->nfmt += (int)(t - (char *)f->to);
	f->to = t;
	return 0;
}

int
__rfmtpad(Fmt *f, int n)
{
	Rune *t, *s;
	int i;

	t = (Rune*)f->to;
	s = (Rune*)f->stop;
	for(i = 0; i < n; i++)
		FMTRCHAR(f, t, s, ' ');
	f->nfmt += (int)(t - (Rune *)f->to);
	f->to = t;
	return 0;
}

int
__fmtcpy(Fmt *f, const void *vm, int n, int sz)
{
	Rune *rt, *rs, r;
	char *t, *s, *m, *me;
	ulong fl;
	int nc, w;

	m = (char*)vm;
	me = m + sz;
	fl = f->flags;
	w = 0;
	if(fl & FmtWidth)
		w = f->width;
	if((fl & FmtPrec) && n > f->prec)
		n = f->prec;
	if(f->runes){
		if(!(fl & FmtLeft) && __rfmtpad(f, w - n) < 0)
			return -1;
		rt = (Rune*)f->to;
		rs = (Rune*)f->stop;
		for(nc = n; nc > 0; nc--){
			r = *(uchar*)m;
			if(r < Runeself)
				m++;
			else if((me - m) >= UTFmax || fullrune(m, (int)(me-m)))
				m += chartorune(&r, m);
			else
				break;
			FMTRCHAR(f, rt, rs, r);
		}
		f->nfmt += (int)(rt - (Rune *)f->to);
		f->to = rt;
		if(fl & FmtLeft && __rfmtpad(f, w - n) < 0)
			return -1;
	}else{
		if(!(fl & FmtLeft) && __fmtpad(f, w - n) < 0)
			return -1;
		t = (char*)f->to;
		s = (char*)f->stop;
		for(nc = n; nc > 0; nc--){
			r = *(uchar*)m;
			if(r < Runeself)
				m++;
			else if((me - m) >= UTFmax || fullrune(m, (int)(me-m)))
				m += chartorune(&r, m);
			else
				break;
			FMTRUNE(f, t, s, r);
		}
		f->nfmt += (int)(t - (char *)f->to);
		f->to = t;
		if(fl & FmtLeft && __fmtpad(f, w - n) < 0)
			return -1;
	}
	return 0;
}

int
__fmtrcpy(Fmt *f, const void *vm, int n)
{
	Rune r, *m, *me, *rt, *rs;
	char *t, *s;
	ulong fl;
	int w;

	m = (Rune*)vm;
	fl = f->flags;
	w = 0;
	if(fl & FmtWidth)
		w = f->width;
	if((fl & FmtPrec) && n > f->prec)
		n = f->prec;
	if(f->runes){
		if(!(fl & FmtLeft) && __rfmtpad(f, w - n) < 0)
			return -1;
		rt = (Rune*)f->to;
		rs = (Rune*)f->stop;
		for(me = m + n; m < me; m++)
			FMTRCHAR(f, rt, rs, *m);
		f->nfmt += (int)(rt - (Rune *)f->to);
		f->to = rt;
		if(fl & FmtLeft && __rfmtpad(f, w - n) < 0)
			return -1;
	}else{
		if(!(fl & FmtLeft) && __fmtpad(f, w - n) < 0)
			return -1;
		t = (char*)f->to;
		s = (char*)f->stop;
		for(me = m + n; m < me; m++){
			r = *m;
			FMTRUNE(f, t, s, r);
		}
		f->nfmt += (int)(t - (char *)f->to);
		f->to = t;
		if(fl & FmtLeft && __fmtpad(f, w - n) < 0)
			return -1;
	}
	return 0;
}

/* fmt out one character */
int
__charfmt(Fmt *f)
{
	char x[1];

	x[0] = (char)va_arg(f->args, int);
	f->prec = 1;
	return __fmtcpy(f, (const char*)x, 1, 1);
}

/* fmt out one rune */
int
__runefmt(Fmt *f)
{
	Rune x[1];

	x[0] = (Rune)va_arg(f->args, int);
	return __fmtrcpy(f, (const void*)x, 1);
}

/* public helper routine: fmt out a null terminated string already in hand */
int
fmtstrcpy(Fmt *f, char *s)
{
	int i, j;

	if(!s)
		return __fmtcpy(f, "<nil>", 5, 5);
	/* if precision is specified, make sure we don't wander off the end */
	if(f->flags & FmtPrec){
#ifdef PLAN9PORT
		Rune r;
		i = 0;
		for(j=0; j<f->prec && s[i]; j++)
			i += chartorune(&r, s+i);
#else
		/* ANSI requires precision in bytes, not Runes */
		for(i=0; i<f->prec; i++)
			if(s[i] == 0)
				break;
		j = utfnlen(s, i);	/* won't print partial at end */
#endif
		return __fmtcpy(f, s, j, i);
	}
	return __fmtcpy(f, s, utflen(s), (int)strlen(s));
}

/* fmt out a null terminated utf string */
int
__strfmt(Fmt *f)
{
	char *s;

	s = va_arg(f->args, char *);
	return fmtstrcpy(f, s);
}

/* public helper routine: fmt out a null terminated rune string already in hand */
int
fmtrunestrcpy(Fmt *f, Rune *s)
{
	Rune *e;
	int n, p;

	if(!s)
		return __fmtcpy(f, "<nil>", 5, 5);
	/* if precision is specified, make sure we don't wander off the end */
	if(f->flags & FmtPrec){
		p = f->prec;
		for(n = 0; n < p; n++)
			if(s[n] == 0)
				break;
	}else{
		for(e = s; *e; e++)
			;
		n = (int)(e - s);
	}
	return __fmtrcpy(f, s, n);
}

/* fmt out a null terminated rune string */
int
__runesfmt(Fmt *f)
{
	Rune *s;

	s = va_arg(f->args, Rune *);
	return fmtrunestrcpy(f, s);
}

/* fmt a % */
int
__percentfmt(Fmt *f)
{
	Rune x[1];

	x[0] = f->r;
	f->prec = 1;
	return __fmtrcpy(f, (const void*)x, 1);
}

/* fmt an integer */
int
__ifmt(Fmt *f)
{
	char buf[140], *p, *conv;
	/* 140: for 64 bits of binary + 3-byte sep every 4 digits */
	uvlong vu;
	ulong fl, u;
	int neg, base, i, n, w, isv;
	int ndig, len, excess, bytelen;
	char *grouping;
	char *thousands;

	neg = 0;
	fl = f->flags;
	isv = 0;
	vu = 0;
	u = 0;
#ifndef PLAN9PORT
	/*
	 * Unsigned verbs for ANSI C
	 */
	switch(f->r){
	case 'o':
	case 'p':
	case 'u':
	case 'x':
	case 'X':
		fl |= FmtUnsigned;
		fl &= ~(FmtSign|FmtSpace);
		break;
	}
#endif
	if(f->r == 'p'){
		u = (uintptr)va_arg(f->args, void*);
		f->r = 'x';
		fl |= FmtUnsigned;
	}else if(fl & FmtVLong){
		isv = 1;
		if(fl & FmtUnsigned)
			vu = va_arg(f->args, uvlong);
		else
			vu = (uvlong)va_arg(f->args, vlong);
	}else if(fl & FmtLong){
		if(fl & FmtUnsigned)
			u = va_arg(f->args, ulong);
		else
			u = (ulong)va_arg(f->args, long);
	}else if(fl & FmtByte){
		if(fl & FmtUnsigned)
			u = (uchar)va_arg(f->args, int);
		else
			u = (ulong)(char)va_arg(f->args, int);
	}else if(fl & FmtShort){
		if(fl & FmtUnsigned)
			u = (ushort)va_arg(f->args, int);
		else
			u = (ulong)(short)va_arg(f->args, int);
	}else{
		if(fl & FmtUnsigned)
			u = va_arg(f->args, uint);
		else
			u = (ulong)va_arg(f->args, int);
	}
	conv = "0123456789abcdef";
	grouping = "\4";	/* for hex, octal etc. (undefined by spec but nice) */
	thousands = f->thousands;
	switch(f->r){
	case 'd':
	case 'i':
	case 'u':
		base = 10;
		grouping = f->grouping;
		break;
	case 'X':
		conv = "0123456789ABCDEF";
		/* fall through */
	case 'x':
		base = 16;
		thousands = ":";
		break;
	case 'b':
		base = 2;
		thousands = ":";
		break;
	case 'o':
		base = 8;
		break;
	default:
		return -1;
	}
	if(!(fl & FmtUnsigned)){
		if(isv && (vlong)vu < 0){
			vu = (uvlong)-(vlong)vu;
			neg = 1;
		}else if(!isv && (long)u < 0){
			u = (ulong)-(long)u;
			neg = 1;
		}
	}
	p = buf + sizeof buf - 1;
	n = 0;	/* in runes */
	excess = 0;	/* number of bytes > number runes */
	ndig = 0;
	len = utflen(thousands);
	bytelen = (int)strlen(thousands);
	if(isv){
		while(vu){
			i = (int)(vu % (uvlong)base);
			vu /= (uvlong)base;
			if((fl & FmtComma) && n % 4 == 3){
				*p-- = ',';
				n++;
			}
			if((fl & FmtApost) && __needsep(&ndig, &grouping)){
				n += len;
				excess += bytelen - len;
				p -= bytelen;
				memmove(p+1, thousands, (size_t)bytelen);
			}
			*p-- = conv[i];
			n++;
		}
	}else{
		while(u){
			i = (int)(u % (ulong)base);
			u /= (ulong)base;
			if((fl & FmtComma) && n % 4 == 3){
				*p-- = ',';
				n++;
			}
			if((fl & FmtApost) && __needsep(&ndig, &grouping)){
				n += len;
				excess += bytelen - len;
				p -= bytelen;
				memmove(p+1, thousands, (size_t)bytelen);
			}
			*p-- = conv[i];
			n++;
		}
	}
	if(n == 0){
		/*
		 * "The result of converting a zero value with
		 * a precision of zero is no characters."  - ANSI
		 *
		 * "For o conversion, # increases the precision, if and only if
		 * necessary, to force the first digit of the result to be a zero
		 * (if the value and precision are both 0, a single 0 is printed)." - ANSI
		 */
		if(!(fl & FmtPrec) || f->prec != 0 || (f->r == 'o' && (fl & FmtSharp))){
			*p-- = '0';
			n = 1;
			if(fl & FmtApost)
				__needsep(&ndig, &grouping);
		}
	}
	for(w = f->prec; n < w && p > buf+3; n++){
		if((fl & FmtApost) && __needsep(&ndig, &grouping)){
			n += len;
			excess += bytelen - len;
			p -= bytelen;
			memmove(p+1, thousands, (size_t)bytelen);
		}
		*p-- = '0';
	}
	if(neg || (fl & (FmtSign|FmtSpace)))
		n++;
	if(fl & FmtSharp){
		if(base == 16)
			n += 2;
		else if(base == 8){
			if(p[1] == '0')
				fl &= ~(ulong)FmtSharp;
			else
				n++;
		}
	}
	if((fl & FmtZero) && !(fl & (FmtLeft|FmtPrec))){
		w = 0;
		if(fl & FmtWidth)
			w = f->width;
		for(; n < w && p > buf+3; n++){
			if((fl & FmtApost) && __needsep(&ndig, &grouping)){
				n += len;
				excess += bytelen - len;
				p -= bytelen;
				memmove(p+1, thousands, (size_t)bytelen);
			}
			*p-- = '0';
		}
		f->flags &= ~(ulong)FmtWidth;
	}
	if(fl & FmtSharp){
		if(base == 16)
			*p-- = (char)f->r;
		if(base == 16 || base == 8)
			*p-- = '0';
	}
	if(neg)
		*p-- = '-';
	else if(fl & FmtSign)
		*p-- = '+';
	else if(fl & FmtSpace)
		*p-- = ' ';
	f->flags &= ~(ulong)FmtPrec;
	return __fmtcpy(f, p + 1, n, n + excess);
}

int
__countfmt(Fmt *f)
{
	void *p;
	ulong fl;

	fl = f->flags;
	p = va_arg(f->args, void*);
	if(fl & FmtVLong){
		*(vlong*)p = f->nfmt;
	}else if(fl & FmtLong){
		*(long*)p = f->nfmt;
	}else if(fl & FmtByte){
		*(char*)p = (char)f->nfmt;
	}else if(fl & FmtShort){
		*(short*)p = (short)f->nfmt;
	}else{
		*(int*)p = f->nfmt;
	}
	return 0;
}

int
__flagfmt(Fmt *f)
{
	switch(f->r){
	case ',':
		f->flags |= FmtComma;
		break;
	case '-':
		f->flags |= FmtLeft;
		break;
	case '+':
		f->flags |= FmtSign;
		break;
	case '#':
		f->flags |= FmtSharp;
		break;
	case '\'':
		f->flags |= FmtApost;
		break;
	case ' ':
		f->flags |= FmtSpace;
		break;
	case 'u':
		f->flags |= FmtUnsigned;
		break;
	case 'h':
		if(f->flags & FmtShort)
			f->flags |= FmtByte;
		f->flags |= FmtShort;
		break;
	case 'L':
		f->flags |= FmtLDouble;
		break;
	case 'l':
		if(f->flags & FmtLong)
			f->flags |= FmtVLong;
		f->flags |= FmtLong;
		break;
	}
	return 1;
}

/* default error format */
int
__badfmt(Fmt *f)
{
	char x[2+UTFmax];
	int n;

	x[0] = '%';
	n = 1 + runetochar(x+1, &f->r);
	x[n++] = '%';
	f->prec = n;
	__fmtcpy(f, (const void*)x, n, n);
	return 0;
}
