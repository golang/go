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
#include	"5.out.h"

enum
{
	thechar = '5',
	PtrSize = 4,
	IntSize = 4,
	MaxAlign = 8,	// max data alignment
	FuncAlign = 4  // single-instruction alignment
};

#ifndef	EXTERN
#define	EXTERN	extern
#endif

/* do not undefine this - code will be removed eventually */
#define	CALLEEBX

#define	dynptrsize	0

typedef	struct	Adr	Adr;
typedef	struct	Sym	Sym;
typedef	struct	Autom	Auto;
typedef	struct	Prog	Prog;
typedef	struct	Reloc	Reloc;
typedef	struct	Optab	Optab;
typedef	struct	Oprang	Oprang;
typedef	uchar	Opcross[32][2][32];
typedef	struct	Count	Count;

#define	P		((Prog*)0)
#define	S		((Sym*)0)
#define	TNAME		(cursym?cursym->name:noname)

struct	Adr
{
	union
	{
		struct {
			int32	u0offset;
			int32	u0offset2; // argsize
		} u0off;
		char*	u0sval;
		Ieee	u0ieee;
		char*	u0sbig;
	} u0;
	Sym*	sym;
	Sym*	gotype;
	char	type;
	char	reg;
	char	name;
	char	class;
};

#define	offset	u0.u0off.u0offset
#define	offset2	u0.u0off.u0offset2
#define	sval	u0.u0sval
#define	scon	sval
#define	ieee	u0.u0ieee
#define	sbig	u0.u0sbig

struct	Reloc
{
	int32	off;
	uchar	siz;
	uchar	done;
	int16	type;
	int32	add;
	int32	xadd;
	Sym*	sym;
	Sym*	xsym;
};

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
	Prog*	pcrel;
	int32	pc;
	int32	line;
	int32	spadj;
	uchar	mark;
	uchar	optab;
	uchar	as;
	uchar	scond;
	uchar	reg;
	uchar	align;	// unused
};

#define	regused	u0.u0regused
#define	forwd	u0.u0forwd
#define	datasize	reg
#define	textflag	reg

#define	iscall(p)	((p)->as == ABL)

struct	Sym
{
	char*	name;
	char*	extname;	// name used in external object files
	short	type;
	short	version;
	uchar	dupok;
	uchar	reachable;
	uchar	cgoexport;
	uchar	leaf;
	int32	dynid;
	int32	plt;
	int32	got;
	int32	value;
	int32	sig;
	int32	size;
	int32	align;	// if non-zero, required alignment in bytes
	int32	elfsym;
	int32	locals;	// size of stack frame locals area
	int32	args;	// size of stack frame incoming arguments area
	uchar	special;
	uchar	fnptr;	// used as fn ptr
	uchar	stkcheck;
	uchar	hide;
	Sym*	hash;	// in hash table
	Sym*	allsym;	// in all symbol list
	Sym*	next;	// in text or data list
	Sym*	sub;	// in SSUB list
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
	uchar	pcrelsiz;
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

enum
{
	LFROM		= 1<<0,
	LTO		= 1<<1,
	LPOOL		= 1<<2,
	LPCREL		= 1<<3,

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
	C_SROREG,	/* both S and R */
	C_LOREG,

	C_PC,
	C_SP,
	C_HREG,

	C_ADDR,		/* reference to relocatable address */

	C_GOK,

/* mark flags */
	FOLL		= 1<<0,
	LABEL		= 1<<1,
	LEAF		= 1<<2,

	STRINGSZ	= 200,
	MINSIZ		= 64,
	NENT		= 100,
	MAXIO		= 8192,
	MAXHIST		= 20,	/* limit of path elements for history symbols */
	MINLC	= 4,
};

#ifndef COFFCVT

EXTERN	int32	HEADR;			/* length of header */
EXTERN	int	HEADTYPE;		/* type of header */
EXTERN	int32	INITDAT;		/* data location */
EXTERN	int32	INITRND;		/* data round above text location */
EXTERN	int32	INITTEXT;		/* text location */
EXTERN	char*	INITENTRY;		/* entry point */
EXTERN	char*	LIBINITENTRY;		/* shared library entry point */
EXTERN	int32	autosize;
EXTERN	Auto*	curauto;
EXTERN	Auto*	curhist;
EXTERN	Prog*	curp;
EXTERN	Sym*	cursym;
EXTERN	Sym*	datap;
EXTERN	int	debug[128];
EXTERN	Sym*	etextp;
EXTERN	char*	noname;
EXTERN	Prog*	lastp;
EXTERN	int32	lcsize;
EXTERN	char	literal[32];
EXTERN	int	nerrors;
EXTERN	int32	instoffset;
EXTERN	Opcross	opcross[8];
EXTERN	Oprang	oprange[ALAST];
EXTERN	char*	outfile;
EXTERN	int32	pc;
EXTERN	uchar	repop[ALAST];
EXTERN	char*	interpreter;
EXTERN	char*	rpath;
EXTERN	uint32	stroffset;
EXTERN	int32	symsize;
EXTERN	Sym*	textp;
EXTERN	int	version;
EXTERN	char	xcmp[C_GOK+1][C_GOK+1];
EXTERN	Prog	zprg;
EXTERN	int	dtype;
EXTERN	int	tlsoffset;
EXTERN	int	armsize;
EXTERN	int	goarm;
EXTERN	Sym*	adrgotype;	// type symbol on last Adr read
EXTERN	Sym*	fromgotype;	// type symbol on last p->from read

extern	char*	anames[];
extern	Optab	optab[];

void	addpool(Prog*, Adr*);
EXTERN	Prog*	blitrl;
EXTERN	Prog*	elitrl;

EXTERN	int	goarm;

void	initdiv(void);
EXTERN	Prog*	prog_div;
EXTERN	Prog*	prog_divu;
EXTERN	Prog*	prog_mod;
EXTERN	Prog*	prog_modu;

#pragma	varargck	type	"A"	int
#pragma	varargck	type	"C"	int
#pragma	varargck	type	"D"	Adr*
#pragma	varargck	type	"I"	uchar*
#pragma	varargck	type	"N"	Adr*
#pragma	varargck	type	"P"	Prog*
#pragma	varargck	type	"S"	char*
#pragma	varargck	type	"Z"	char*
#pragma	varargck	type	"i"	char*

int	Aconv(Fmt*);
int	Cconv(Fmt*);
int	Dconv(Fmt*);
int	Iconv(Fmt*);
int	Nconv(Fmt*);
int	Oconv(Fmt*);
int	Pconv(Fmt*);
int	Sconv(Fmt*);
int	aclass(Adr*);
void	addhist(int32, int);
Prog*	appendp(Prog*);
void	asmb(void);
void	asmout(Prog*, Optab*, int32*);
int32	atolwhex(char*);
Prog*	brloop(Prog*);
void	buildop(void);
void	buildrep(int, int);
void	cflush(void);
int	chipzero(Ieee*);
int	chipfloat(Ieee*);
int	cmp(int, int);
int	compound(Prog*);
double	cputime(void);
void	diag(char*, ...);
void	divsig(void);
void	dodata(void);
void	doprof1(void);
void	doprof2(void);
int32	entryvalue(void);
void	exchange(Prog*);
void	follow(void);
void	hputl(int);
int	isnop(Prog*);
void	listinit(void);
Sym*	lookup(char*, int);
void	cput(int);
void	hput(int32);
void	lput(int32);
void	lputb(int32);
void	lputl(int32);
void*	mysbrk(uint32);
void	names(void);
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
int32	regoff(Adr*);
int	relinv(int);
int32	rnd(int32, int32);
void	softfloat(void);
void	span(void);
void	strnput(char*, int);
int32	symaddr(Sym*);
void	undef(void);
void	vputb(uint64);
void	vputl(uint64);
void	wputb(uint16);
void	wput(int32);
void    wputl(ushort w);
void	xdefine(char*, int, int32);
void	noops(void);
int32	immrot(uint32);
int32	immaddr(int32);
int32	opbra(int, int);
int	brextra(Prog*);
int	isbranch(Prog*);
void	fnptrs(void);
void	doelf(void);
void	dozerostk(void); // used by -Z

vlong		addaddr(Sym *s, Sym *t);
vlong		addsize(Sym *s, Sym *t);
vlong		addstring(Sym *s, char *str);
vlong		adduint16(Sym *s, uint16 v);
vlong		adduint32(Sym *s, uint32 v);
vlong		adduint64(Sym *s, uint64 v);
vlong		adduint8(Sym *s, uint8 v);
vlong		adduintxx(Sym *s, uint64 v, int wid);

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
