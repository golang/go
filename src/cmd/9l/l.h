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
#include	"../9c/9.out.h"
#include	"../8l/elf.h"

#ifndef	EXTERN
#define	EXTERN	extern
#endif

#define	LIBNAMELEN	300

typedef	struct	Adr	Adr;
typedef	struct	Sym	Sym;
typedef	struct	Autom	Auto;
typedef	struct	Prog	Prog;
typedef	struct	Optab	Optab;

#define	P		((Prog*)0)
#define	S		((Sym*)0)
#define	TNAME		(curtext&&curtext->from.sym?curtext->from.sym->name:noname)

struct	Adr
{
	union
	{
		vlong	u0offset;
		char	u0sval[NSNAME];
		Ieee	u0ieee;
	}u0;
	Sym	*sym;
	Auto	*autom;
	char	type;
	uchar	reg;
	char	name;
	char	class;
};

#define	offset	u0.u0offset
#define	sval	u0.u0sval
#define	ieee	u0.u0ieee

struct	Prog
{
	Adr	from;
	Adr	from3;	/* fma and rlwm */
	Adr	to;
	Prog	*forwd;
	Prog	*cond;
	Prog	*link;
	vlong	pc;
	long	regused;
	short	line;
	short	mark;
	short	optab;		/* could be uchar */
	short	as;
	char	reg;
};
struct	Sym
{
	char	*name;
	short	type;
	short	version;
	short	become;
	short	frame;
	uchar	subtype;
	ushort	file;
	vlong	value;
	long	sig;
	Sym	*link;
};
struct	Autom
{
	Sym	*sym;
	Auto	*link;
	vlong	aoffset;
	short	type;
};
struct	Optab
{
	short	as;
	char	a1;
	char	a2;
	char	a3;
	char	a4;
	char	type;
	char	size;
	char	param;
};
struct
{
	Optab*	start;
	Optab*	stop;
} oprange[ALAST];

enum
{
	FPCHIP		= 1,
	BIG		= 32768-8,
	STRINGSZ	= 200,
	MAXIO		= 8192,
	MAXHIST		= 20,				/* limit of path elements for history symbols */
	DATBLK		= 1024,
	NHASH		= 10007,
	NHUNK		= 100000,
	MINSIZ		= 64,
	NENT		= 100,
	NSCHED		= 20,

/* mark flags */
	LABEL		= 1<<0,
	LEAF		= 1<<1,
	FLOAT		= 1<<2,
	BRANCH		= 1<<3,
	LOAD		= 1<<4,
	FCMP		= 1<<5,
	SYNC		= 1<<6,
	LIST		= 1<<7,
	FOLL		= 1<<8,
	NOSCHED		= 1<<9,

	STEXT		= 1,
	SDATA,
	SBSS,
	SDATA1,
	SXREF,
	SLEAF,
	SFILE,
	SCONST,
	SUNDEF,

	SIMPORT,
	SEXPORT,

	C_NONE		= 0,
	C_REG,
	C_FREG,
	C_CREG,
	C_SPR,		/* special processor register */
	C_ZCON,
	C_SCON,		/* 16 bit signed */
	C_UCON,		/* low 16 bits 0 */
	C_ADDCON,	/* -0x8000 <= v < 0 */
	C_ANDCON,	/* 0 < v <= 0xFFFF */
	C_LCON,		/* other 32 */
	C_DCON,		/* other 64 (could subdivide further) */
	C_SACON,
	C_SECON,
	C_LACON,
	C_LECON,
	C_SBRA,
	C_LBRA,
	C_SAUTO,
	C_LAUTO,
	C_SEXT,
	C_LEXT,
	C_ZOREG,
	C_SOREG,
	C_LOREG,
	C_FPSCR,
	C_MSR,
	C_XER,
	C_LR,
	C_CTR,
	C_ANY,
	C_GOK,
	C_ADDR,

	C_NCLASS,

	Roffset	= 22,		/* no. bits for offset in relocation address */
	Rindex	= 10		/* no. bits for index in relocation address */
};

EXTERN union
{
	struct
	{
		uchar	obuf[MAXIO];			/* output buffer */
		uchar	ibuf[MAXIO];			/* input buffer */
	} u;
	char	dbuf[1];
} buf;

#define	cbuf	u.obuf
#define	xbuf	u.ibuf

EXTERN	long	HEADR;			/* length of header */
EXTERN	int	HEADTYPE;		/* type of header */
EXTERN	vlong	INITDAT;		/* data location */
EXTERN	long	INITRND;		/* data round above text location */
EXTERN	vlong	INITTEXT;		/* text location */
EXTERN	long	INITTEXTP;		/* text location (physical) */
EXTERN	char*	INITENTRY;		/* entry point */
EXTERN	long	autosize;
EXTERN	Biobuf	bso;
EXTERN	long	bsssize;
EXTERN	int	cbc;
EXTERN	uchar*	cbp;
EXTERN	int	cout;
EXTERN	Auto*	curauto;
EXTERN	Auto*	curhist;
EXTERN	Prog*	curp;
EXTERN	Prog*	curtext;
EXTERN	Prog*	datap;
EXTERN	Prog*	prog_movsw;
EXTERN	Prog*	prog_movdw;
EXTERN	Prog*	prog_movws;
EXTERN	Prog*	prog_movwd;
EXTERN	vlong	datsize;
EXTERN	char	debug[128];
EXTERN	Prog*	firstp;
EXTERN	uchar	fnuxi8[8];
EXTERN	uchar	fnuxi4[4];
EXTERN	Sym*	hash[NHASH];
EXTERN	Sym*	histfrog[MAXHIST];
EXTERN	int	histfrogp;
EXTERN	int	histgen;
EXTERN	char*	library[50];
EXTERN	char*	libraryobj[50];
EXTERN	int	libraryp;
EXTERN	int	xrefresolv;
EXTERN	char*	hunk;
EXTERN	uchar	inuxi1[1];
EXTERN	uchar	inuxi2[2];
EXTERN	uchar	inuxi4[4];
EXTERN	uchar	inuxi8[8];
EXTERN	Prog*	lastp;
EXTERN	long	lcsize;
EXTERN	char	literal[32];
EXTERN	int	nerrors;
EXTERN	long	nhunk;
EXTERN	char*	noname;
EXTERN	vlong	instoffset;
EXTERN	char*	outfile;
EXTERN	vlong	pc;
EXTERN	int	r0iszero;
EXTERN	long	symsize;
EXTERN	long	staticgen;
EXTERN	Prog*	textp;
EXTERN	vlong	textsize;
EXTERN	long	tothunk;
EXTERN	char	xcmp[C_NCLASS][C_NCLASS];
EXTERN	int	version;
EXTERN	Prog	zprg;
EXTERN	int	dtype;

EXTERN	int	doexp, dlm;
EXTERN	int	imports, nimports;
EXTERN	int	exports, nexports, allexport;
EXTERN	char*	EXPTAB;
EXTERN	Prog	undefp;

#define	UP	(&undefp)

extern	Optab	optab[];
extern	char*	anames[];
extern	char*	cnames[];

int	Aconv(Fmt*);
int	Dconv(Fmt*);
int	Nconv(Fmt*);
int	Pconv(Fmt*);
int	Sconv(Fmt*);
int	Rconv(Fmt*);
int	aclass(Adr*);
void	addhist(long, int);
void	histtoauto(void);
void	addlibpath(char*);
void	addnop(Prog*);
void	append(Prog*, Prog*);
void	asmb(void);
void	asmdyn(void);
void	asmlc(void);
int	asmout(Prog*, Optab*, int);
void	asmsym(void);
vlong	atolwhex(char*);
Prog*	brloop(Prog*);
void	buildop(void);
void	cflush(void);
void	ckoff(Sym*, vlong);
int	cmp(int, int);
void	cput(long);
int	compound(Prog*);
double	cputime(void);
void	datblk(long, long);
void	diag(char*, ...);
void	dodata(void);
void	doprof1(void);
void	doprof2(void);
void	dynreloc(Sym*, long, int, int, int);
vlong	entryvalue(void);
void	errorexit(void);
void	exchange(Prog*);
void	export(void);
int	fileexists(char*);
int	find1(long, int);
char*	findlib(char*);
void	follow(void);
void	gethunk(void);
double	ieeedtod(Ieee*);
long	ieeedtof(Ieee*);
void	import(void);
int	isint32(vlong);
int	isuint32(uvlong);
int	isnop(Prog*);
void	ldobj(int, long, char*);
void	loadlib(void);
void	listinit(void);
void	initmuldiv(void);
Sym*	lookup(char*, int);
void	llput(vlong);
void	llputl(vlong);
void	lput(long);
void	lputl(long);
void	mkfwd(void);
void*	mysbrk(ulong);
void	names(void);
void	nocache(Prog*);
void	noops(void);
void	nopout(Prog*);
void	nuxiinit(void);
void	objfile(char*);
int	ocmp(void*, void*);
long	opcode(int);
Optab*	oplook(Prog*);
void	patch(void);
void	prasm(Prog*);
void	prepend(Prog*, Prog*);
Prog*	prg(void);
int	pseudo(Prog*);
void	putsymb(char*, int, vlong, int);
void	readundefs(char*, int);
long	regoff(Adr*);
int	relinv(int);
vlong	rnd(vlong, long);
void	sched(Prog*, Prog*);
void	span(void);
void	strnput(char*, int);
void	undef(void);
void	undefsym(Sym*);
vlong	vregoff(Adr*);
void	wput(long);
void	wputl(long);
void	xdefine(char*, int, vlong);
void	xfol(Prog*);
void	zerosig(char*);

#pragma	varargck	type	"D"	Adr*
#pragma	varargck	type	"N"	Adr*
#pragma	varargck	type	"P"	Prog*
#pragma	varargck	type	"R"	int
#pragma	varargck	type	"A"	int
#pragma	varargck	type	"S"	char*
#pragma	varargck	argpos	diag 1
