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
#include	"../5l/5.out.h"

enum
{
	PtrSize = 4
};

#ifndef	EXTERN
#define	EXTERN	extern
#endif

/* do not undefine this - code will be removed eventually */
#define	CALLEEBX

typedef	struct	Adr	Adr;
typedef	struct	Sym	Sym;
typedef	struct	Autom	Auto;
typedef	struct	Prog	Prog;
typedef	struct	Optab	Optab;
typedef	struct	Oprang	Oprang;
typedef	uchar	Opcross[32][2][32];
typedef	struct	Count	Count;
typedef	struct	Use	Use;

#define	P		((Prog*)0)
#define	S		((Sym*)0)
#define	U		((Use*)0)
#define	TNAME		(curtext&&curtext->from.sym?curtext->from.sym->name:noname)

struct	Adr
{
	union
	{
		int32	u0offset;
		char*	u0sval;
		Ieee*	u0ieee;
		char*	u0sbig;
	} u0;
	union
	{
		Auto*	u1autom;
		Sym*	u1sym;
	} u1;
	char	type;
	uchar	index; // not used on arm, required by ld/go.c
	char	reg;
	char	name;
	int32	offset2; // argsize
	char	class;
	Sym*	gotype;
};

#define	offset	u0.u0offset
#define	sval	u0.u0sval
#define	ieee	u0.u0ieee
#define	sbig	u0.u0sbig

#define	autom	u1.u1autom
#define	sym	u1.u1sym

struct	Prog
{
	Adr	from;
	Adr	to;
	union
	{
		int32	u0regused;
		Prog*	u0forwd;
	} u0;
	Prog*	cond;
	Prog*	link;
	Prog*	dlink;
	int32	pc;
	int32	line;
	uchar	mark;
	uchar	optab;
	uchar	as;
	uchar	scond;
	uchar	reg;
	uchar	align;
};
#define	regused	u0.u0regused
#define	forwd	u0.u0forwd

struct	Sym
{
	char	*name;
	short	type;
	short	version;
	short	become;
	short	frame;
	uchar	subtype;
	uchar	reachable;
	int32	value;
	int32	sig;
	uchar	used;
	uchar	thumb;	// thumb code
	uchar	foreign;	// called by arm if thumb, by thumb if arm
	uchar	fnptr;	// used as fn ptr
	Use*		use;
	Sym*	link;
	Prog*	text;
	Prog*	data;
	Sym*	gotype;
	char*	file;
	char*	dynldname;
	char*	dynldlib;
};

#define SIGNINTERN	(1729*325*1729)

struct	Autom
{
	Sym*	asym;
	Auto*	link;
	int32	aoffset;
	short	type;
	Sym*	gotype;
};
struct	Optab
{
	char	as;
	uchar	a1;
	char	a2;
	uchar	a3;
	uchar	type;
	char	size;
	char	param;
	char	flag;
};
struct	Oprang
{
	Optab*	start;
	Optab*	stop;
};
struct	Count
{
	int32	count;
	int32	outof;
};
struct	Use
{
	Prog*	p;	/* use */
	Prog*	ct;	/* curtext */
	Use*		link;
};

enum
{
	Sxxx,

	STEXT		= 1,
	SDATA,
	SBSS,
	SDATA1,
	SXREF,
	SLEAF,
	SFILE,
	SCONST,
	SSTRING,
	SUNDEF,
	SREMOVED,

	SIMPORT,
	SEXPORT,

	LFROM		= 1<<0,
	LTO		= 1<<1,
	LPOOL		= 1<<2,
	V4		= 1<<3,	/* arm v4 arch */

	C_NONE		= 0,
	C_REG,
	C_REGREG,
	C_SHIFT,
	C_FREG,
	C_PSR,
	C_FCR,

	C_RCON,		/* 0xff rotated */
	C_NCON,		/* ~RCON */
	C_SCON,		/* 0xffff */
	C_BCON,		/* thumb */
	C_LCON,
	C_FCON,
	C_GCON,		/* thumb */

	C_RACON,
	C_SACON,	/* thumb */
	C_LACON,
	C_GACON,	/* thumb */

	C_RECON,
	C_LECON,

	C_SBRA,
	C_LBRA,
	C_GBRA,		/* thumb */

	C_HAUTO,	/* halfword insn offset (-0xff to 0xff) */
	C_FAUTO,	/* float insn offset (0 to 0x3fc, word aligned) */
	C_HFAUTO,	/* both H and F */
	C_SAUTO,	/* -0xfff to 0xfff */
	C_LAUTO,

	C_HEXT,
	C_FEXT,
	C_HFEXT,
	C_SEXT,
	C_LEXT,

	C_HOREG,
	C_FOREG,
	C_HFOREG,
	C_SOREG,
	C_ROREG,
	C_SROREG,	/* both S and R */
	C_LOREG,
	C_GOREG,		/* thumb */

	C_PC,
	C_SP,
	C_HREG,
	C_OFFPC,		/* thumb */

	C_ADDR,		/* relocatable address */

	C_GOK,

/* mark flags */
	FOLL		= 1<<0,
	LABEL		= 1<<1,
	LEAF		= 1<<2,

	BIG		= (1<<12)-4,
	STRINGSZ	= 200,
	NHASH		= 10007,
	NHUNK		= 100000,
	MINSIZ		= 64,
	NENT		= 100,
	MAXIO		= 8192,
	MAXHIST		= 20,	/* limit of path elements for history symbols */

	Roffset	= 22,		/* no. bits for offset in relocation address */
	Rindex	= 10,		/* no. bits for index in relocation address */
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

#define	setarch(p)		if((p)->as==ATEXT) thumb=(p)->reg&ALLTHUMBS
#define	setthumb(p)	if((p)->as==ATEXT) seenthumb|=(p)->reg&ALLTHUMBS

#ifndef COFFCVT

EXTERN	int32	HEADR;			/* length of header */
EXTERN	int	HEADTYPE;		/* type of header */
EXTERN	int32	INITDAT;		/* data location */
EXTERN	int32	INITRND;		/* data round above text location */
EXTERN	int32	INITTEXT;		/* text location */
EXTERN	char*	INITENTRY;		/* entry point */
EXTERN	int32	autosize;
EXTERN	Biobuf	bso;
EXTERN	int32	bsssize;
EXTERN	int	cbc;
EXTERN	uchar*	cbp;
EXTERN	int	cout;
EXTERN	Auto*	curauto;
EXTERN	Auto*	curhist;
EXTERN	Prog*	curp;
EXTERN	Prog*	curtext;
EXTERN	Prog*	datap;
EXTERN	int32	datsize;
EXTERN	char	debug[128];
EXTERN	Prog*	edatap;
EXTERN	Prog*	etextp;
EXTERN	Prog*	firstp;
EXTERN	char*	noname;
EXTERN	int	xrefresolv;
EXTERN	Prog*	lastp;
EXTERN	int32	lcsize;
EXTERN	char	literal[32];
EXTERN	int	nerrors;
EXTERN	int32	instoffset;
EXTERN	Opcross	opcross[8];
EXTERN	Oprang	oprange[ALAST];
EXTERN	Oprang	thumboprange[ALAST];
EXTERN	char*	outfile;
EXTERN	int32	pc;
EXTERN	uchar	repop[ALAST];
EXTERN	uint32	stroffset;
EXTERN	int32	symsize;
EXTERN	Prog*	textp;
EXTERN	int32	textsize;
EXTERN	int	version;
EXTERN	char	xcmp[C_GOK+1][C_GOK+1];
EXTERN	Prog	zprg;
EXTERN	int	dtype;
EXTERN	int	armv4;
EXTERN	int	thumb;
EXTERN	int	seenthumb;
EXTERN	int	armsize;

EXTERN	int	doexp, dlm;
EXTERN	int	imports, nimports;
EXTERN	int	exports, nexports;
EXTERN	char*	EXPTAB;
EXTERN	Prog	undefp;

#define	UP	(&undefp)

extern	char*	anames[];
extern	Optab	optab[];
extern	Optab	thumboptab[];

void	addpool(Prog*, Adr*);
EXTERN	Prog*	blitrl;
EXTERN	Prog*	elitrl;

void	initdiv(void);
EXTERN	Prog*	prog_div;
EXTERN	Prog*	prog_divu;
EXTERN	Prog*	prog_mod;
EXTERN	Prog*	prog_modu;

#pragma	varargck	type	"A"	int
#pragma	varargck	type	"C"	int
#pragma	varargck	type	"D"	Adr*
#pragma	varargck	type	"N"	Adr*
#pragma	varargck	type	"P"	Prog*
#pragma	varargck	type	"S"	char*

int	Aconv(Fmt*);
int	Cconv(Fmt*);
int	Dconv(Fmt*);
int	Nconv(Fmt*);
int	Oconv(Fmt*);
int	Pconv(Fmt*);
int	Sconv(Fmt*);
int	aclass(Adr*);
int	thumbaclass(Adr*, Prog*);
void	addhist(int32, int);
Prog*	appendp(Prog*);
void	asmb(void);
void	asmdyn(void);
void	asmlc(void);
void	asmthumbmap(void);
void	asmout(Prog*, Optab*);
void	thumbasmout(Prog*, Optab*);
void	asmsym(void);
int32	atolwhex(char*);
Prog*	brloop(Prog*);
void	buildop(void);
void	thumbbuildop(void);
void	buildrep(int, int);
void	cflush(void);
void	ckoff(Sym*, int32);
int	chipfloat(Ieee*);
int	cmp(int, int);
int	compound(Prog*);
double	cputime(void);
void	datblk(int32, int32, int);
void	diag(char*, ...);
void	divsig(void);
void	dodata(void);
void	doprof1(void);
void	doprof2(void);
void	dynreloc(Sym*, int32, int);
int32	entryvalue(void);
void	exchange(Prog*);
void	export(void);
void	follow(void);
void	hputl(int);
void	import(void);
int	isnop(Prog*);
void	listinit(void);
Sym*	lookup(char*, int);
void	cput(int);
void	hput(int32);
void	lput(int32);
void	lputl(int32);
void	mkfwd(void);
void*	mysbrk(uint32);
void	names(void);
Prog*	newdata(Sym *s, int o, int w, int t);
void	nocache(Prog*);
int	ocmp(const void*, const void*);
int32	opirr(int);
Optab*	oplook(Prog*);
int32	oprrr(int, int);
int32	olr(int32, int, int, int);
int32	olhr(int32, int, int, int);
int32	olrr(int, int, int, int);
int32	olhrr(int, int, int, int);
int32	osr(int, int, int32, int, int);
int32	oshr(int, int32, int, int);
int32	ofsr(int, int, int32, int, int, Prog*);
int32	osrr(int, int, int, int);
int32	oshrr(int, int, int, int);
int32	omvl(Prog*, Adr*, int);
void	patch(void);
void	prasm(Prog*);
void	prepend(Prog*, Prog*);
Prog*	prg(void);
int	pseudo(Prog*);
void	putsymb(char*, int, int32, int);
int32	regoff(Adr*);
int	relinv(int);
int32	rnd(int32, int32);
void	span(void);
void	strnput(char*, int);
void	undef(void);
void	wput(int32);
void    wputl(ushort w);
void	xdefine(char*, int, int32);
void	xfol(Prog*);
void	noops(void);
int32	immrot(uint32);
int32	immaddr(int32);
int32	opbra(int, int);
int	brextra(Prog*);
int	isbranch(Prog*);
int	fnpinc(Sym *);
int	fninc(Sym *);
void	thumbcount(void);
void reachable(void);
void fnptrs(void);

uint32	linuxheadr(void);
void	linuxphdr(int type, int flags, vlong foff,
	vlong vaddr, vlong paddr,
	vlong filesize, vlong memsize, vlong align);
void	linuxshdr(char *name, uint32 type, vlong flags, vlong addr, vlong off,
	vlong size, uint32 link, uint32 info, vlong align, vlong entsize);
int	linuxstrtable(void);

/*
 *	go.c
 */
void	deadcode(void);
char*	gotypefor(char *name);
void	ldpkg(Biobuf *f, int64 len, char *filename);

#endif
