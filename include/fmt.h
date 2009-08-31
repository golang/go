#ifndef _FMT_H_
#define _FMT_H_ 1
#if defined(__cplusplus)
extern "C" {
#endif
/*
 * The authors of this software are Rob Pike and Ken Thompson.
 *              Copyright (c) 2002 by Lucent Technologies.
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR LUCENT TECHNOLOGIES MAKE ANY
 * REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
 * OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 */

#include <stdarg.h>
#include <utf.h>

typedef struct Fmt	Fmt;
struct Fmt{
	unsigned char	runes;		/* output buffer is runes or chars? */
	void	*start;			/* of buffer */
	void	*to;			/* current place in the buffer */
	void	*stop;			/* end of the buffer; overwritten if flush fails */
	int	(*flush)(Fmt *);	/* called when to == stop */
	void	*farg;			/* to make flush a closure */
	int	nfmt;			/* num chars formatted so far */
	va_list	args;			/* args passed to dofmt */
	Rune	r;			/* % format Rune */
	int	width;
	int	prec;
	unsigned long	flags;
	char	*decimal;	/* decimal point; cannot be "" */

	/* For %'d */
	char *thousands;	/* separator for thousands */

	/*
	 * Each char is an integer indicating #digits before next separator. Values:
	 *	\xFF: no more grouping (or \x7F; defined to be CHAR_MAX in POSIX)
	 *	\x00: repeat previous indefinitely
	 *	\x**: count that many
	 */
	char	*grouping;		/* descriptor of separator placement */
};

enum{
	FmtWidth	= 1,
	FmtLeft		= FmtWidth << 1,
	FmtPrec		= FmtLeft << 1,
	FmtSharp	= FmtPrec << 1,
	FmtSpace	= FmtSharp << 1,
	FmtSign		= FmtSpace << 1,
	FmtApost		= FmtSign << 1,
	FmtZero		= FmtApost << 1,
	FmtUnsigned	= FmtZero << 1,
	FmtShort	= FmtUnsigned << 1,
	FmtLong		= FmtShort << 1,
	FmtVLong	= FmtLong << 1,
	FmtComma	= FmtVLong << 1,
	FmtByte		= FmtComma << 1,
	FmtLDouble	= FmtByte << 1,

	FmtFlag		= FmtLDouble << 1
};

extern	int	(*fmtdoquote)(int);

/* Edit .+1,/^$/ | cfn $PLAN9/src/lib9/fmt/?*.c | grep -v static |grep -v __ */
int		dofmt(Fmt *f, char *fmt);
int		dorfmt(Fmt *f, const Rune *fmt);
double		fmtcharstod(int(*f)(void*), void *vp);
int		fmtfdflush(Fmt *f);
int		fmtfdinit(Fmt *f, int fd, char *buf, int size);
int		fmtinstall(int c, int (*f)(Fmt*));
int		fmtnullinit(Fmt*);
void		fmtlocaleinit(Fmt*, char*, char*, char*);
int		fmtprint(Fmt *f, char *fmt, ...);
int		fmtrune(Fmt *f, int r);
int		fmtrunestrcpy(Fmt *f, Rune *s);
int		fmtstrcpy(Fmt *f, char *s);
char*		fmtstrflush(Fmt *f);
int		fmtstrinit(Fmt *f);
double		fmtstrtod(const char *as, char **aas);
int		fmtvprint(Fmt *f, char *fmt, va_list args);
int		fprint(int fd, char *fmt, ...);
int		print(char *fmt, ...);
void		quotefmtinstall(void);
int		quoterunestrfmt(Fmt *f);
int		quotestrfmt(Fmt *f);
Rune*		runefmtstrflush(Fmt *f);
int		runefmtstrinit(Fmt *f);
Rune*		runeseprint(Rune *buf, Rune *e, char *fmt, ...);
Rune*		runesmprint(char *fmt, ...);
int		runesnprint(Rune *buf, int len, char *fmt, ...);
int		runesprint(Rune *buf, char *fmt, ...);
Rune*		runevseprint(Rune *buf, Rune *e, char *fmt, va_list args);
Rune*		runevsmprint(char *fmt, va_list args);
int		runevsnprint(Rune *buf, int len, char *fmt, va_list args);
char*		seprint(char *buf, char *e, char *fmt, ...);
char*		smprint(char *fmt, ...);
int		snprint(char *buf, int len, char *fmt, ...);
int		sprint(char *buf, char *fmt, ...);
int		vfprint(int fd, char *fmt, va_list args);
char*		vseprint(char *buf, char *e, char *fmt, va_list args);
char*		vsmprint(char *fmt, va_list args);
int		vsnprint(char *buf, int len, char *fmt, va_list args);

#if defined(__cplusplus)
}
#endif
#endif
