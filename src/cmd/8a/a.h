// Inferno utils/8a/a.h
// http://code.google.com/p/inferno-os/source/browse/utils/8a/a.h
//
//	Copyright © 1994-1999 Lucent Technologies Inc.	All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <bio.h>
#include <link.h>
#include "../8l/8.out.h"

#ifndef	EXTERN
#define	EXTERN	extern
#endif

#undef	getc
#undef	ungetc
#undef	BUFSIZ

#define	getc	ccgetc
#define	ungetc	ccungetc

typedef	struct	Sym	Sym;
typedef	struct	Ref	Ref;
typedef	struct	Io	Io;
typedef	struct	Addr2	Addr2;

#define	MAXALIGN	7
#define	FPCHIP		1
#define	NSYMB		500
#define	BUFSIZ		8192
#define	HISTSZ		20
#ifndef	EOF
#define	EOF		(-1)
#endif
#define	IGN		(-2)
#define	GETC()		((--fi.c < 0)? filbuf(): *fi.p++ & 0xff)
#define	NHASH		503
#define	STRINGSZ	200
#define	NMACRO		10

struct	Sym
{
	Sym*	link;
	Ref*	ref;
	char*	macro;
	int32	value;
	ushort	type;
	char	*name;
	char	sym;
};
#define	S	((Sym*)0)

struct	Ref
{
	int	class;
};

EXTERN struct
{
	char*	p;
	int	c;
} fi;

struct	Io
{
	Io*	link;
	char	b[BUFSIZ];
	char*	p;
	short	c;
	short	f;
};
#define	I	((Io*)0)

struct	Addr2
{
	Addr	from;
	Addr	to;
};

enum
{
	CLAST,
	CMACARG,
	CMACRO,
	CPREPROC,
};

EXTERN	int	debug[256];
EXTERN	Sym*	hash[NHASH];
EXTERN	char**	Dlist;
EXTERN	int	nDlist;
EXTERN	int	newflag;
EXTERN	char*	hunk;
EXTERN	char**	include;
EXTERN	Io*	iofree;
EXTERN	Io*	ionext;
EXTERN	Io*	iostack;
EXTERN	int32	lineno;
EXTERN	int	nerrors;
EXTERN	int32	nhunk;
EXTERN	int	ninclude;
EXTERN	int32	nsymb;
EXTERN	Addr	nullgen;
EXTERN	char*	outfile;
EXTERN	int	pass;
EXTERN	int32	pc;
EXTERN	int	peekc;
EXTERN	int32	stmtline;
EXTERN	int	sym;
EXTERN	char*	symb;
EXTERN	int	thechar;
EXTERN	char*	thestring;
EXTERN	int32	thunk;
EXTERN	Biobuf	obuf;
EXTERN	Link*	ctxt;
EXTERN	Biobuf	bstdout;

void*	alloc(int32);
void*	allocn(void*, int32, int32);
void	ensuresymb(int32);
void	errorexit(void);
void	pushio(void);
void	newio(void);
void	newfile(char*, int);
Sym*	slookup(char*);
Sym*	lookup(void);
void	syminit(Sym*);
int32	yylex(void);
int	getc(void);
int	getnsc(void);
void	unget(int);
int	escchar(int);
void	cinit(void);
void	checkscale(int);
void	pinit(char*);
void	cclean(void);
int	isreg(Addr*);
void	outcode(int, Addr2*);
void	outhist(void);
int	filbuf(void);
Sym*	getsym(void);
void	domacro(void);
void	macund(void);
void	macdef(void);
void	macexpand(Sym*, char*);
void	macinc(void);
void	macprag(void);
void	maclin(void);
void	macif(int);
void	macend(void);
void	dodefine(char*);
void	prfile(int32);
void	linehist(char*, int);
void	gethunk(void);
void	yyerror(char*, ...);
int	yyparse(void);
void	setinclude(char*);
int	assemble(char*);
