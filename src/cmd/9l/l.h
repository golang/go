// cmd/9l/l.h from Vita Nuova.
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2008 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2008 Lucent Technologies Inc. and others
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
#include	"9.out.h"

#ifndef	EXTERN
#define	EXTERN	extern
#endif

enum
{
	thechar = '9',
	PtrSize = 8,
	IntSize = 8,
	RegSize = 8,
	MaxAlign = 32,	// max data alignment
	FuncAlign = 8
};

#define	P		((Prog*)0)
#define	S		((LSym*)0)

enum
{
	FPCHIP		= 1,
	STRINGSZ	= 200,
	MAXHIST		= 20,				/* limit of path elements for history symbols */
	DATBLK		= 1024,
	NHASH		= 10007,
	NHUNK		= 100000,
	MINSIZ		= 64,
	NENT		= 100,
	NSCHED		= 20,
	MINLC		= 4,

	Roffset	= 22,		/* no. bits for offset in relocation address */
	Rindex	= 10		/* no. bits for index in relocation address */
};

EXTERN	int32	autosize;
EXTERN	LSym*	datap;
EXTERN	int	debug[128];
EXTERN	int32	lcsize;
EXTERN	char	literal[32];
EXTERN	int	nerrors;
EXTERN	vlong	instoffset;
EXTERN	char*	rpath;
EXTERN	vlong	pc;
EXTERN	int32	symsize;
EXTERN	int32	staticgen;
EXTERN	Prog*	lastp;
EXTERN	vlong	textsize;

void	asmb(void);
void	adddynlib(char *lib);
void	adddynrel(LSym *s, Reloc *r);
void	adddynsym(Link *ctxt, LSym *s);
int	archreloc(Reloc *r, LSym *s, vlong *val);
vlong	archrelocvariant(Reloc *r, LSym *s, vlong t);
void	listinit(void);
vlong	rnd(vlong, int32);

#define	LPUT(a)	(ctxt->arch->endian == BigEndian ? lputb(a):lputl(a))
#define	WPUT(a)	(ctxt->arch->endian == BigEndian ? wputb(a):wputl(a))
#define	VPUT(a)	(ctxt->arch->endian == BigEndian ? vputb(a):vputl(a))

/* Used by ../ld/dwarf.c */
enum
{
	DWARFREGSP = 1
};
