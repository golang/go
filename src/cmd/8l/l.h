// Inferno utils/8l/l.h
// http://code.google.com/p/inferno-os/source/browse/utils/8l/l.h
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
#include	"8.out.h"

#ifndef	EXTERN
#define	EXTERN	extern
#endif

enum
{
	thechar = '8',
	PtrSize = 4,
	IntSize = 4,
	MaxAlign = 32,	// max data alignment
	FuncAlign = 16
};

#define	P		((Prog*)0)
#define	S		((Sym*)0)
#define	TNAME		(cursym?cursym->name:noname)

typedef	struct	Adr	Adr;
typedef	struct	Prog	Prog;
typedef	struct	Sym	Sym;
typedef	struct	Auto	Auto;
typedef	struct	Optab	Optab;
typedef	struct	Reloc	Reloc;

struct	Adr
{
	union
	{
		int32	u0offset;
		char	u0scon[8];
		Prog	*u0cond;	/* not used, but should be D_BRANCH */
		Ieee	u0ieee;
		char	*u0sbig;
	} u0;
	Sym*	sym;
	short	type;
	uchar	index;
	char	scale;
	int32	offset2;
};

#define	offset	u0.u0offset
#define	scon	u0.u0scon
#define	cond	u0.u0cond
#define	ieee	u0.u0ieee
#define	sbig	u0.u0sbig

struct	Reloc
{
	int32	off;
	uchar	siz;
	uchar	done;
	int32	type;
	int32	add;
	int32	xadd;
	Sym*	sym;
	Sym*	xsym;
};

struct	Prog
{
	Adr	from;
	Adr	to;
	Prog*	forwd;
	Prog*	comefrom;
	Prog*	link;
	Prog*	pcond;	/* work on this */
	int32	pc;
	int32	spadj;
	int32	line;
	short	as;
	char	width;		/* fake for DATA */
	char	ft;		/* oclass cache */
	char	tt;
	uchar	mark;	/* work on these */
	uchar	back;
	uchar	bigjmp;
};
#define	datasize	from.scale
#define	textflag	from.scale
#define	iscall(p)	((p)->as == ACALL)

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
	char*	name;
	char*	extname;	// name used in external object files
	short	type;
	short	version;
	uchar	dupok;
	uchar	reachable;
	uchar	cgoexport;
	uchar	special;
	uchar	stkcheck;
	uchar	hide;
	int32	value;
	int32	size;
	int32	sig;
	int32	dynid;
	int32	plt;
	int32	got;
	int32	align;	// if non-zero, required alignment in bytes
	int32	elfsym;
	int32	locals;	// size of stack frame locals area
	int32	args;	// size of stack frame incoming arguments area
	Sym*	hash;	// in hash table
	Sym*	allsym;	// in all symbol list
	Sym*	next;	// in text or data list
	Sym*	sub;	// in sub list
	Sym*	outer;	// container of sub
	Sym*	gotype;
	Sym*	reachparent;
	Sym*	queue;
	char*	file;
	char*	dynimplib;
	char*	dynimpvers;
	struct Section*	sect;
	
	// STEXT
	Auto*	autom;
	Prog*	text;
	
	// SDATA, SBSS
	uchar*	p;
	int32	np;
	int32	maxp;
	Reloc*	r;
	int32	nr;
	int32	maxr;
	int 	rel_ro;
};
struct	Optab
{
	short	as;
	uchar*	ytab;
	uchar	prefix;
	uchar	op[12];
};

enum
{
	MINSIZ		= 4,
	STRINGSZ	= 200,
	MINLC		= 1,
	MAXIO		= 8192,
	MAXHIST		= 20,				/* limit of path elements for history symbols */

	Yxxx		= 0,
	Ynone,
	Yi0,
	Yi1,
	Yi8,
	Yi32,
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
	Ycr0,	Ycr1,	Ycr2,	Ycr3,	Ycr4,	Ycr5,	Ycr6,	Ycr7,
	Ydr0,	Ydr1,	Ydr2,	Ydr3,	Ydr4,	Ydr5,	Ydr6,	Ydr7,
	Ytr0,	Ytr1,	Ytr2,	Ytr3,	Ytr4,	Ytr5,	Ytr6,	Ytr7,
	Ymr, Ymm,
	Yxr, Yxm,
	Ymax,

	Zxxx		= 0,

	Zlit,
	Zlitm_r,
	Z_rp,
	Zbr,
	Zcall,
	Zcallcon,
	Zcallind,
	Zib_,
	Zib_rp,
	Zibo_m,
	Zil_,
	Zil_rp,
	Zilo_m,
	Zjmp,
	Zjmpcon,
	Zloop,
	Zm_o,
	Zm_r,
	Zm2_r,
	Zm_r_xm,
	Zm_r_i_xm,
	Zaut_r,
	Zo_m,
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
	Zibm_r,	/* mmx1,mmx2/mem64,imm8 */
	Zbyte,
	Zmov,
	Zmax,

	Px		= 0,
	Pe		= 0x66,	/* operand escape */
	Pm		= 0x0f,	/* 2byte opcode escape */
	Pq		= 0xff,	/* both escape */
	Pb		= 0xfe,	/* byte operands */
	Pf2		= 0xf2,	/* xmm escape 1 */
	Pf3		= 0xf3,	/* xmm escape 2 */
};

#pragma	varargck	type	"A"	int
#pragma	varargck	type	"D"	Adr*
#pragma	varargck	type	"I"	uchar*
#pragma	varargck	type	"P"	Prog*
#pragma	varargck	type	"R"	int
#pragma	varargck	type	"S"	char*
#pragma	varargck	type	"Y"	Sym*
#pragma	varargck	type	"Z"	char*
#pragma	varargck	type	"i"	char*

EXTERN	int32	HEADR;
EXTERN	int32	HEADTYPE;
EXTERN	int32	INITRND;
EXTERN	int32	INITTEXT;
EXTERN	int32	INITDAT;
EXTERN	char*	INITENTRY;		/* entry point */
EXTERN	char*	LIBINITENTRY;		/* shared library entry point */
EXTERN	char*	pcstr;
EXTERN	Auto*	curauto;
EXTERN	Auto*	curhist;
EXTERN	Prog*	curp;
EXTERN	Sym*	cursym;
EXTERN	Sym*	datap;
EXTERN	int	debug[128];
EXTERN	char	literal[32];
EXTERN	Sym*	etextp;
EXTERN	Prog*	firstp;
EXTERN	uchar	ycover[Ymax*Ymax];
EXTERN	uchar*	andptr;
EXTERN	uchar	and[100];
EXTERN	char	reg[D_NONE];
EXTERN	int32	lcsize;
EXTERN	int	maxop;
EXTERN	int	nerrors;
EXTERN	char*	noname;
EXTERN	int32	pc;
EXTERN	char*	rpath;
EXTERN	int32	spsize;
EXTERN	Sym*	symlist;
EXTERN	int32	symsize;
EXTERN	Sym*	textp;
EXTERN	int32	textsize;
EXTERN	int	version;
EXTERN	Prog	zprg;
EXTERN	int	dtype;
EXTERN	int	tlsoffset;
EXTERN	Sym*	adrgotype;	// type symbol on last Adr read
EXTERN	Sym*	fromgotype;	// type symbol on last p->from read

extern	Optab	optab[];
extern	char*	anames[];

int	Aconv(Fmt*);
int	Dconv(Fmt*);
int	Iconv(Fmt*);
int	Pconv(Fmt*);
int	Rconv(Fmt*);
int	Sconv(Fmt*);
void	addhist(int32, int);
Prog*	appendp(Prog*);
void	asmb(void);
void	asmdyn(void);
void	asmins(Prog*);
void	asmsym(void);
int32	atolwhex(char*);
Prog*	brchain(Prog*);
Prog*	brloop(Prog*);
void	cflush(void);
Prog*	copyp(Prog*);
vlong	cpos(void);
double	cputime(void);
void	diag(char*, ...);
void	dodata(void);
void	doelf(void);
void	doprof1(void);
void	doprof2(void);
void	dostkoff(void);
int32	entryvalue(void);
void	follow(void);
void	instinit(void);
void	listinit(void);
Sym*	lookup(char*, int);
void	lputb(int32);
void	lputl(int32);
void	vputl(uint64);
void	strnput(char*, int);
void	main(int, char*[]);
void*	mal(uint32);
int	opsize(Prog*);
void	patch(void);
Prog*	prg(void);
int	relinv(int);
int32	rnd(int32, int32);
void	s8put(char*);
void	span(void);
void	undef(void);
int32	symaddr(Sym*);
void	wput(ushort);
void	wputl(ushort);
void	xdefine(char*, int, int32);

uint32	machheadr(void);
vlong		addaddr(Sym *s, Sym *t);
vlong		addsize(Sym *s, Sym *t);
vlong		addstring(Sym *s, char *str);
vlong		adduint16(Sym *s, uint16 v);
vlong		adduint32(Sym *s, uint32 v);
vlong		adduint64(Sym *s, uint64 v);
vlong		adduint8(Sym *s, uint8 v);
vlong		adduintxx(Sym *s, uint64 v, int wid);

/*
 *	go.c
 */
void	deadcode(void);

/* Native is little-endian */
#define	LPUT(a)	lputl(a)
#define	WPUT(a)	wputl(a)
#define	VPUT(a)	vputl(a)

/* Used by ../ld/dwarf.c */
enum
{
	DWARFREGSP = 4
};
