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

/* do not undefine this - code will be removed eventually */
#define	CALLEEBX

#define	dynptrsize	0

#define	P		((Prog*)0)
#define	S		((LSym*)0)
#define	TNAME		(ctxt->cursym?ctxt->cursym->name:noname)

#define SIGNINTERN	(1729*325*1729)

typedef	struct	Count	Count;
struct	Count
{
	int32	count;
	int32	outof;
};

enum
{
/* mark flags */
	FOLL		= 1<<0,
	LABEL		= 1<<1,
	LEAF		= 1<<2,

	STRINGSZ	= 200,
	MINSIZ		= 64,
	NENT		= 100,
	MAXIO		= 8192,
	MAXHIST		= 40,	/* limit of path elements for history symbols */
	MINLC	= 4,

	C_NONE		= 0,
	C_REG,
	C_REGREG,
	C_REGREG2,
	C_SHIFT,
	C_FREG,
	C_PSR,
	C_FCR,

	C_RCON,		/* 0xff rotated */
	C_NCON,		/* ~RCON */
	C_SCON,		/* 0xffff */
	C_LCON,
	C_LCONADDR,
	C_ZFCON,
	C_SFCON,
	C_LFCON,

	C_RACON,
	C_LACON,

	C_SBRA,
	C_LBRA,

	C_HAUTO,	/* halfword insn offset (-0xff to 0xff) */
	C_FAUTO,	/* float insn offset (0 to 0x3fc, word aligned) */
	C_HFAUTO,	/* both H and F */
	C_SAUTO,	/* -0xfff to 0xfff */
	C_LAUTO,

	C_HOREG,
	C_FOREG,
	C_HFOREG,
	C_SOREG,
	C_ROREG,
	C_SROREG,	/* both nil and R */
	C_LOREG,

	C_PC,
	C_SP,
	C_HREG,

	C_ADDR,		/* reference to relocatable address */

	C_GOK,
};

#ifndef COFFCVT

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
void	cput(int32 c);
int	elfreloc1(Reloc *r, vlong sectoff);
void	elfsetupplt(void);
void	hput(int32 l);
void	listinit(void);
void	lput(int32 l);
int	machoreloc1(Reloc *r, vlong sectoff);
void	main(int argc, char *argv[]);
void	noops(void);
void	nopstat(char *f, Count *c);
int32	rnd(int32 v, int32 r);
void	wput(int32 l);

/* Native is little-endian */
#define	LPUT(a)	lputl(a)
#define	WPUT(a)	wputl(a)
#define	VPUT(a)	abort()

#endif

/* Used by ../ld/dwarf.c */
enum
{
	DWARFREGSP = 13
};
