// Inferno utils/5l/l.h
// http://code.google.com/p/inferno-os/source/browse/utils/5l/l.h
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
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

#include	<u.h>
#include	<libc.h>
#include	<bio.h>
#include	<link.h>
#include	"5.out.h"

enum
{
	thechar = '5',
	PtrSize = 4,
	IntSize = 4,
	RegSize = 4,
	MaxAlign = 8,	// max data alignment
	FuncAlign = 4  // single-instruction alignment
};

#ifndef	EXTERN
#define	EXTERN	extern
#endif

#define	P		((Prog*)0)
#define	S		((LSym*)0)

enum
{
/* mark flags */
	FOLL		= 1<<0,
	LABEL		= 1<<1,
	LEAF		= 1<<2,

	MINLC	= 4,
};

EXTERN	int32	autosize;
EXTERN	LSym*	datap;
EXTERN	int	debug[128];
EXTERN	char*	noname;
EXTERN	Prog*	lastp;
EXTERN	int32	lcsize;
EXTERN	char	literal[32];
EXTERN	int	nerrors;
EXTERN	int32	instoffset;
EXTERN	char*	rpath;
EXTERN	uint32	stroffset;
EXTERN	int32	symsize;
EXTERN	int	armsize;

#pragma	varargck	type	"I"	uint32*

int	Iconv(Fmt *fp);
void	adddynlib(char *lib);
void	adddynrel(LSym *s, Reloc *r);
void	adddynrela(LSym *rel, LSym *s, Reloc *r);
void	adddynsym(Link *ctxt, LSym *s);
int	archreloc(Reloc *r, LSym *s, vlong *val);
void	asmb(void);
int	elfreloc1(Reloc *r, vlong sectoff);
void	elfsetupplt(void);
void	listinit(void);
int	machoreloc1(Reloc *r, vlong sectoff);
void	main(int argc, char *argv[]);
int32	rnd(int32 v, int32 r);

/* Native is little-endian */
#define	LPUT(a)	lputl(a)
#define	WPUT(a)	wputl(a)
#define	VPUT(a)	abort()

/* Used by ../ld/dwarf.c */
enum
{
	DWARFREGSP = 13
};
