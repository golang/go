// Inferno utils/6l/l.h
// http://code.google.com/p/inferno-os/source/browse/utils/6l/l.h
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
#include	"../6l/6.out.h"
#include	"compat.h"

#ifndef	EXTERN
#define	EXTERN	extern
#endif

#define	P		((Prog*)0)
#define	S		((Sym*)0)
#define	TNAME		(curtext?curtext->from.sym->name:noname)
#define	cput(c)\
	{ *cbp++ = c;\
	if(--cbc <= 0)\
		cflush(); }

typedef	struct	Adr	Adr;
typedef	struct	Prog	Prog;
typedef	struct	Sym	Sym;
typedef	struct	Auto	Auto;
typedef	struct	Optab	Optab;
typedef	struct	Movtab	Movtab;

struct	Adr
{
	union
	{
		vlong	u0offset;
		char	u0scon[8];
		Prog	*u0cond;	/* not used, but should be D_BRANCH */
		Ieee	u0ieee;
		char	*u0sbig;
	} u0;
	union
	{
		Auto*	u1autom;
		Sym*	u1sym;
	} u1;
	short	type;
	char	index;
	char	scale;
	Sym*	gotype;
};

#define	offset	u0.u0offset
#define	scon	u0.u0scon
#define	cond	u0.u0cond
#define	ieee	u0.u0ieee
#define	sbig	u0.u0sbig

#define	autom	u1.u1autom
#define	sym	u1.u1sym

struct	Prog
{
	Adr	from;
	Adr	to;
	Prog	*forwd;
	Prog*	link;
	Prog*	dlink;
	Prog*	pcond;	/* work on this */
	vlong	pc;
	int32	line;
	uchar	mark;	/* work on these */
	uchar	back;

	short	as;
	char	width;		/* fake for DATA */
	char	mode;	/* 16, 32, or 64 */
};
struct	Auto
{
	Sym*	asym;
	Auto*	link;
	int32	aoffset;
	short	type;
	Sym*	gotype;
};
struct	Sym
{
	char	*name;
	short	type;
	short	version;
	short	become;
	short	frame;
	ushort	file;
	uchar	subtype;
	uchar	dupok;
	uchar	reachable;
	vlong	value;
	vlong	size;
	int32	sig;
	Sym*	link;
	Prog*	text;
	Prog*	data;
	Sym*	gotype;
};
struct	Optab
{
	short	as;
	uchar*	ytab;
	uchar	prefix;
	uchar	op[20];
};
struct	Movtab
{
	short	as;
	uchar	ft;
	uchar	tt;
	uchar	code;
	uchar	op[4];
};

enum
{
	Sxxx,
	STEXT		= 1,
	SDATA,
	SBSS,
	SDATA1,
	SXREF,
	SFILE,
	SCONST,
	SUNDEF,

	SIMPORT,
	SEXPORT,

	NHASH		= 10007,
	NHUNK		= 100000,
	MINSIZ		= 8,
	STRINGSZ	= 200,
	MINLC		= 1,
	MAXIO		= 8192,
	MAXHIST		= 20,				/* limit of path elements for history symbols */

	Yxxx		= 0,
	Ynone,
	Yi0,
	Yi1,
	Yi8,
	Ys32,
	Yi32,
	Yi64,
	Yiauto,
	Yal,
	Ycl,
	Yax,
	Ycx,
	Yrb,
	Yrl,
	Yrf,
	Yf0,
	Yrx,
	Ymb,
	Yml,
	Ym,
	Ybr,
	Ycol,

	Ycs,	Yss,	Yds,	Yes,	Yfs,	Ygs,
	Ygdtr,	Yidtr,	Yldtr,	Ymsw,	Ytask,
	Ycr0,	Ycr1,	Ycr2,	Ycr3,	Ycr4,	Ycr5,	Ycr6,	Ycr7,	Ycr8,
	Ydr0,	Ydr1,	Ydr2,	Ydr3,	Ydr4,	Ydr5,	Ydr6,	Ydr7,
	Ytr0,	Ytr1,	Ytr2,	Ytr3,	Ytr4,	Ytr5,	Ytr6,	Ytr7,	Yrl32,	Yrl64,
	Ymr, Ymm,
	Yxr, Yxm,
	Ymax,

	Zxxx		= 0,

	Zlit,
	Z_rp,
	Zbr,
	Zcall,
	Zib_,
	Zib_rp,
	Zibo_m,
	Zibo_m_xm,
	Zil_,
	Zil_rp,
	Ziq_rp,
	Zilo_m,
	Ziqo_m,
	Zjmp,
	Zloop,
	Zo_iw,
	Zm_o,
	Zm_r,
	Zm_r_xm,
	Zm_r_i_xm,
	Zm_r_3d,
	Zm_r_xm_nr,
	Zr_m_xm_nr,
	Zibm_r,	/* mmx1,mmx2/mem64,imm8 */
	Zmb_r,
	Zaut_r,
	Zo_m,
	Zo_m64,
	Zpseudo,
	Zr_m,
	Zr_m_xm,
	Zr_m_i_xm,
	Zrp_,
	Z_ib,
	Z_il,
	Zm_ibo,
	Zm_ilo,
	Zib_rr,
	Zil_rr,
	Zclr,
	Zbyte,
	Zmax,

	Px		= 0,
	P32		= 0x32,	/* 32-bit only */
	Pe		= 0x66,	/* operand escape */
	Pm		= 0x0f,	/* 2byte opcode escape */
	Pq		= 0xff,	/* both escape */
	Pb		= 0xfe,	/* byte operands */
	Pf2		= 0xf2,	/* xmm escape 1 */
	Pf3		= 0xf3,	/* xmm escape 2 */
	Pw		= 0x48,	/* Rex.w */
	Py		= 0x80,	/* defaults to 64-bit mode */

	Rxf		= 1<<9,	/* internal flag for Rxr on from */
	Rxt		= 1<<8,	/* internal flag for Rxr on to */
	Rxw		= 1<<3,	/* =1, 64-bit operand size */
	Rxr		= 1<<2,	/* extend modrm reg */
	Rxx		= 1<<1,	/* extend sib index */
	Rxb		= 1<<0,	/* extend modrm r/m, sib base, or opcode reg */

	Roffset	= 22,		/* no. bits for offset in relocation address */
	Rindex	= 10,		/* no. bits for index in relocation address */
	Maxand	= 10,		/* in -a output width of the byte codes */
};

EXTERN union
{
	struct
	{
		char	obuf[MAXIO];			/* output buffer */
		uchar	ibuf[MAXIO];			/* input buffer */
	} u;
	char	dbuf[1];
} buf;

#define	cbuf	u.obuf
#define	xbuf	u.ibuf

#pragma	varargck	type	"A"	uint
#pragma	varargck	type	"D"	Adr*
#pragma	varargck	type	"P"	Prog*
#pragma	varargck	type	"R"	int
#pragma	varargck	type	"S"	char*

EXTERN	int32	HEADR;
EXTERN	int32	HEADTYPE;
EXTERN	vlong	INITDAT;
EXTERN	int32	INITRND;
EXTERN	vlong	INITTEXT;
EXTERN	char*	INITENTRY;		/* entry point */
EXTERN	char*	LIBDIR;
EXTERN	Biobuf	bso;
EXTERN	int32	bsssize;
EXTERN	int	cbc;
EXTERN	char*	cbp;
EXTERN	char*	pcstr;
EXTERN	int	cout;
EXTERN	Auto*	curauto;
EXTERN	Auto*	curhist;
EXTERN	Prog*	curp;
EXTERN	Prog*	curtext;
EXTERN	Prog*	datap;
EXTERN	Prog*	edatap;
EXTERN	vlong	datsize;
EXTERN	char	debug[128];
EXTERN	char	literal[32];
EXTERN	Prog*	etextp;
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
EXTERN	char	ycover[Ymax*Ymax];
EXTERN	uchar*	andptr;
EXTERN	uchar*	rexptr;
EXTERN	uchar	and[30];
EXTERN	int	reg[D_NONE];
EXTERN	int	regrex[D_NONE+1];
EXTERN	Prog*	lastp;
EXTERN	int32	lcsize;
EXTERN	int	nerrors;
EXTERN	int32	nhunk;
EXTERN	int32	nsymbol;
EXTERN	char*	noname;
EXTERN	char*	outfile;
EXTERN	vlong	pc;
EXTERN	int32	spsize;
EXTERN	Sym*	symlist;
EXTERN	int32	symsize;
EXTERN	Prog*	textp;
EXTERN	vlong	textsize;
EXTERN	int32	thunk;
EXTERN	int	version;
EXTERN	Prog	zprg;
EXTERN	int	dtype;
EXTERN	char*	paramspace;

EXTERN	Adr*	reloca;
EXTERN	int	doexp;		// export table
EXTERN	int	dlm;		// dynamically loadable module
EXTERN	int	imports, nimports;
EXTERN	int	exports, nexports;
EXTERN	char*	EXPTAB;
EXTERN	Prog	undefp;
EXTERN	vlong	textstksiz;
EXTERN	vlong	textarg;
extern	char	thechar;

#define	UP	(&undefp)

extern	Optab	optab[];
extern	Optab*	opindex[];
extern	char*	anames[];

int	Aconv(Fmt*);
int	Dconv(Fmt*);
int	Pconv(Fmt*);
int	Rconv(Fmt*);
int	Sconv(Fmt*);
void	addhist(int32, int);
void	addstackmark(void);
Prog*	appendp(Prog*);
vlong	addstring(Sym*, char*);
vlong	adduint32(Sym*, uint32);
vlong	adduint64(Sym*, uint64);
vlong	addaddr(Sym*, Sym*);
vlong	addsize(Sym*, Sym*);
void	asmb(void);
void	asmdyn(void);
void	asmins(Prog*);
void	asmlc(void);
void	asmsp(void);
void	asmsym(void);
vlong	atolwhex(char*);
Prog*	brchain(Prog*);
Prog*	brloop(Prog*);
void	buildop(void);
void	cflush(void);
void	ckoff(Sym*, int32);
Prog*	copyp(Prog*);
double	cputime(void);
void	datblk(int32, int32);
void	deadcode(void);
void	diag(char*, ...);
void	dobss(void);
void	dodata(void);
void	doelf(void);
void	doinit(void);
void	doprof1(void);
void	doprof2(void);
void	dostkoff(void);
void	dynreloc(Sym*, uint32, int);
vlong	entryvalue(void);
void	errorexit(void);
void	export(void);
int	find1(int32, int);
int	find2(int32, int);
void	follow(void);
void	addstachmark(void);
void	gethunk(void);
void	gotypestrings(void);
void	histtoauto(void);
double	ieeedtod(Ieee*);
int32	ieeedtof(Ieee*);
void	import(void);
void	ldobj(Biobuf*, int64, char*);
void	ldpkg(Biobuf*, int64, char*);
void	loadlib(void);
void	listinit(void);
Sym*	lookup(char*, int);
void	lputb(int32);
void	lputl(int32);
void	main(int, char*[]);
void	mkfwd(void);
void*	mysbrk(uint32);
Prog*	newdata(Sym*, int, int, int);
Prog*	newtext(Prog*, Sym*);
void	nopout(Prog*);
void	nuxiinit(void);
void	objfile(char*);
int	opsize(Prog*);
void	patch(void);
Prog*	prg(void);
void	parsetextconst(vlong);
void	readundefs(char*, int);
int	relinv(int);
int32	reuse(Prog*, Sym*);
vlong	rnd(vlong, vlong);
void	span(void);
void	strnput(char*, int);
void	undef(void);
void	undefsym(Sym*);
vlong	vaddr(Adr*);
vlong	symaddr(Sym*);
void	vputl(uint64);
void	wputb(uint16);
void	wputl(uint16);
void	xdefine(char*, int, vlong);
void	xfol(Prog*);
void	zaddr(Biobuf*, Adr*, Sym*[]);
void	zerosig(char*);

void	machseg(char*, vlong, vlong, vlong, vlong, uint32, uint32, uint32, uint32);
void	machsymseg(uint32, uint32);
void	machsect(char*, char*, vlong, vlong, uint32, uint32, uint32, uint32, uint32);
void	machstack(vlong);
void	machdylink(void);
uint32	machheadr(void);

/* Native is little-endian */
#define	LPUT(a)	lputl(a)
#define	WPUT(a)	wputl(a)
#define	VPUT(a)	vputl(a)

#pragma	varargck	type	"D"	Adr*
#pragma	varargck	type	"P"	Prog*
#pragma	varargck	type	"R"	int
#pragma	varargck	type	"A"	int
#pragma	varargck	argpos	diag 1
