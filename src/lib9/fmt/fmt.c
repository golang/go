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

enum
{
	Maxfmt = 64
};

typedef struct Convfmt Convfmt;
struct Convfmt
{
	int	c;
	volatile	Fmts	fmt;	/* for spin lock in fmtfmt; avoids race due to write order */
};

static struct
{
	/* lock by calling __fmtlock, __fmtunlock */
	int	nfmt;
	Convfmt	fmt[Maxfmt];
} fmtalloc;

static Convfmt knownfmt[] = {
	' ',	__flagfmt,
	'#',	__flagfmt,
	'%',	__percentfmt,
	'\'',	__flagfmt,
	'+',	__flagfmt,
	',',	__flagfmt,
	'-',	__flagfmt,
	'C',	__runefmt,	/* Plan 9 addition */
	'E',	__efgfmt,
#ifndef PLAN9PORT
	'F',	__efgfmt,	/* ANSI only */
#endif
	'G',	__efgfmt,
#ifndef PLAN9PORT
	'L',	__flagfmt,	/* ANSI only */
#endif
	'S',	__runesfmt,	/* Plan 9 addition */
	'X',	__ifmt,
	'b',	__ifmt,		/* Plan 9 addition */
	'c',	__charfmt,
	'd',	__ifmt,
	'e',	__efgfmt,
	'f',	__efgfmt,
	'g',	__efgfmt,
	'h',	__flagfmt,
#ifndef PLAN9PORT
	'i',	__ifmt,		/* ANSI only */
#endif
	'l',	__flagfmt,
	'n',	__countfmt,
	'o',	__ifmt,
	'p',	__ifmt,
	'r',	__errfmt,
	's',	__strfmt,
#ifdef PLAN9PORT
	'u',	__flagfmt,
#else
	'u',	__ifmt,
#endif
	'x',	__ifmt,
	0,	nil,
};


int	(*fmtdoquote)(int);

/*
 * __fmtlock() must be set
 */
static int
__fmtinstall(int c, Fmts f)
{
	Convfmt *p, *ep;

	if(c<=0 || c>=65536)
		return -1;
	if(!f)
		f = __badfmt;

	ep = &fmtalloc.fmt[fmtalloc.nfmt];
	for(p=fmtalloc.fmt; p<ep; p++)
		if(p->c == c)
			break;

	if(p == &fmtalloc.fmt[Maxfmt])
		return -1;

	p->fmt = f;
	if(p == ep){	/* installing a new format character */
		fmtalloc.nfmt++;
		p->c = c;
	}

	return 0;
}

int
fmtinstall(int c, int (*f)(Fmt*))
{
	int ret;

	__fmtlock();
	ret = __fmtinstall(c, f);
	__fmtunlock();
	return ret;
}

static Fmts
fmtfmt(int c)
{
	Convfmt *p, *ep;

	ep = &fmtalloc.fmt[fmtalloc.nfmt];
	for(p=fmtalloc.fmt; p<ep; p++)
		if(p->c == c){
			while(p->fmt == nil)	/* loop until value is updated */
				;
			return p->fmt;
		}

	/* is this a predefined format char? */
	__fmtlock();
	for(p=knownfmt; p->c; p++)
		if(p->c == c){
			__fmtinstall(p->c, p->fmt);
			__fmtunlock();
			return p->fmt;
		}
	__fmtunlock();

	return __badfmt;
}

void*
__fmtdispatch(Fmt *f, void *fmt, int isrunes)
{
	Rune rune, r;
	int i, n;

	f->flags = 0;
	f->width = f->prec = 0;

	for(;;){
		if(isrunes){
			r = *(Rune*)fmt;
			fmt = (Rune*)fmt + 1;
		}else{
			fmt = (char*)fmt + chartorune(&rune, (char*)fmt);
			r = rune;
		}
		f->r = r;
		switch(r){
		case '\0':
			return nil;
		case '.':
			f->flags |= FmtWidth|FmtPrec;
			continue;
		case '0':
			if(!(f->flags & FmtWidth)){
				f->flags |= FmtZero;
				continue;
			}
			/* fall through */
		case '1': case '2': case '3': case '4':
		case '5': case '6': case '7': case '8': case '9':
			i = 0;
			while(r >= '0' && r <= '9'){
				i = i * 10 + r - '0';
				if(isrunes){
					r = *(Rune*)fmt;
					fmt = (Rune*)fmt + 1;
				}else{
					r = *(char*)fmt;
					fmt = (char*)fmt + 1;
				}
			}
			if(isrunes)
				fmt = (Rune*)fmt - 1;
			else
				fmt = (char*)fmt - 1;
		numflag:
			if(f->flags & FmtWidth){
				f->flags |= FmtPrec;
				f->prec = i;
			}else{
				f->flags |= FmtWidth;
				f->width = i;
			}
			continue;
		case '*':
			i = va_arg(f->args, int);
			if(i < 0){
				/*
				 * negative precision =>
				 * ignore the precision.
				 */
				if(f->flags & FmtPrec){
					f->flags &= ~FmtPrec;
					f->prec = 0;
					continue;
				}
				i = -i;
				f->flags |= FmtLeft;
			}
			goto numflag;
		}
		n = (*fmtfmt(r))(f);
		if(n < 0)
			return nil;
		if(n == 0)
			return fmt;
	}
}
