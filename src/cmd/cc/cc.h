// Inferno utils/cc/cc.h
// http://code.google.com/p/inferno-os/source/browse/utils/cc/cc.h
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

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <ctype.h>
#include "compat.h"

#pragma	lib	"../cc/cc.a$O"

#ifndef	EXTERN
#define EXTERN	extern
#endif

typedef	struct	Node	Node;
typedef	struct	Sym	Sym;
typedef	struct	Type	Type;
typedef	struct	Funct	Funct;
typedef	struct	Decl	Decl;
typedef	struct	Io	Io;
typedef	struct	Hist	Hist;
typedef	struct	Term	Term;
typedef	struct	Init	Init;
typedef	struct	Bits	Bits;

#define	NHUNK		50000L
#define	BUFSIZ		8192
#define	NSYMB		500
#define	NHASH		1024
#define	STRINGSZ	200
#define	HISTSZ		20
#define YYMAXDEPTH	500
#define	NTERM		10
#define	MAXALIGN	7

#define	SIGN(n)		((vlong)1<<(n-1))
#define	MASK(n)		(SIGN(n)|(SIGN(n)-1))

#define	BITS	5
#define	NVAR	(BITS*sizeof(ulong)*8)
struct	Bits
{
	ulong	b[BITS];
};

struct	Node
{
	Node*	left;
	Node*	right;
	void*	label;
	long	pc;
	int	reg;
	long	xoffset;
	double	fconst;		/* fp constant */
	vlong	vconst;		/* non fp const */
	char*	cstring;	/* character string */
	ushort*	rstring;	/* rune string */

	Sym*	sym;
	Type*	type;
	long	lineno;
	uchar	op;
	uchar	oldop;
	uchar	xcast;
	uchar	class;
	uchar	etype;
	uchar	complex;
	uchar	addable;
	uchar	scale;
	uchar	garb;
};
#define	Z	((Node*)0)

struct	Sym
{
	Sym*	link;
	Type*	type;
	Type*	suetag;
	Type*	tenum;
	char*	macro;
	long	varlineno;
	long	offset;
	vlong	vconst;
	double	fconst;
	Node*	label;
	ushort	lexical;
	char	*name;
	ushort	block;
	ushort	sueblock;
	uchar	class;
	uchar	sym;
	uchar	aused;
	uchar	sig;
};
#define	S	((Sym*)0)

enum{
	SIGNONE = 0,
	SIGDONE = 1,
	SIGINTERN = 2,

	SIGNINTERN = 1729*325*1729,
};

struct	Decl
{
	Decl*	link;
	Sym*	sym;
	Type*	type;
	long	varlineno;
	long	offset;
	short	val;
	ushort	block;
	uchar	class;
	uchar	aused;
};
#define	D	((Decl*)0)

struct	Type
{
	Sym*	sym;
	Sym*	tag;
	Funct*	funct;
	Type*	link;
	Type*	down;
	long	width;
	long	offset;
	long	lineno;
	uchar	shift;
	uchar	nbits;
	uchar	etype;
	uchar	garb;
};

#define	T	((Type*)0)
#define	NODECL	((void(*)(int, Type*, Sym*))0)

struct	Init			/* general purpose initialization */
{
	int	code;
	ulong	value;
	char*	s;
};

EXTERN struct
{
	char*	p;
	int	c;
} fi;

struct	Io
{
	Io*	link;
	char*	p;
	char	b[BUFSIZ];
	short	c;
	short	f;
};
#define	I	((Io*)0)

struct	Hist
{
	Hist*	link;
	char*	name;
	long	line;
	long	offset;
};
#define	H	((Hist*)0)
EXTERN Hist*	hist;

struct	Term
{
	vlong	mult;
	Node	*node;
};

enum
{
	Axxx,
	Ael1,
	Ael2,
	Asu2,
	Aarg0,
	Aarg1,
	Aarg2,
	Aaut3,
	NALIGN,
};

enum
{
	DMARK,
	DAUTO,
	DSUE,
	DLABEL,
};
enum
{
	OXXX,
	OADD,
	OADDR,
	OAND,
	OANDAND,
	OARRAY,
	OAS,
	OASI,
	OASADD,
	OASAND,
	OASASHL,
	OASASHR,
	OASDIV,
	OASHL,
	OASHR,
	OASLDIV,
	OASLMOD,
	OASLMUL,
	OASLSHR,
	OASMOD,
	OASMUL,
	OASOR,
	OASSUB,
	OASXOR,
	OBIT,
	OBREAK,
	OCASE,
	OCAST,
	OCOMMA,
	OCOND,
	OCONST,
	OCONTINUE,
	ODIV,
	ODOT,
	ODOTDOT,
	ODWHILE,
	OENUM,
	OEQ,
	OFOR,
	OFUNC,
	OGE,
	OGOTO,
	OGT,
	OHI,
	OHS,
	OIF,
	OIND,
	OINDREG,
	OINIT,
	OLABEL,
	OLDIV,
	OLE,
	OLIST,
	OLMOD,
	OLMUL,
	OLO,
	OLS,
	OLSHR,
	OLT,
	OMOD,
	OMUL,
	ONAME,
	ONE,
	ONOT,
	OOR,
	OOROR,
	OPOSTDEC,
	OPOSTINC,
	OPREDEC,
	OPREINC,
	OPROTO,
	OREGISTER,
	ORETURN,
	OSET,
	OSIGN,
	OSIZE,
	OSTRING,
	OLSTRING,
	OSTRUCT,
	OSUB,
	OSWITCH,
	OUNION,
	OUSED,
	OWHILE,
	OXOR,
	ONEG,
	OCOM,
	OPOS,
	OELEM,

	OTST,		/* used in some compilers */
	OINDEX,
	OFAS,
	OREGPAIR,

	OEND
};
enum
{
	TXXX,
	TCHAR,
	TUCHAR,
	TSHORT,
	TUSHORT,
	TINT,
	TUINT,
	TLONG,
	TULONG,
	TVLONG,
	TUVLONG,
	TFLOAT,
	TDOUBLE,
	TIND,
	TFUNC,
	TARRAY,
	TVOID,
	TSTRUCT,
	TUNION,
	TENUM,
	NTYPE,

	TAUTO	= NTYPE,
	TEXTERN,
	TSTATIC,
	TTYPEDEF,
	TTYPESTR,
	TREGISTER,
	TCONSTNT,
	TVOLATILE,
	TUNSIGNED,
	TSIGNED,
	TDOT,
	TFILE,
	TOLD,
	NALLTYPES,
};
enum
{
	CXXX,
	CAUTO,
	CEXTERN,
	CGLOBL,
	CSTATIC,
	CLOCAL,
	CTYPEDEF,
	CTYPESTR,
	CPARAM,
	CSELEM,
	CLABEL,
	CEXREG,
	NCTYPES,
};
enum
{
	GXXX		= 0,
	GCONSTNT	= 1<<0,
	GVOLATILE	= 1<<1,
	NGTYPES		= 1<<2,

	GINCOMPLETE	= 1<<2,
};
enum
{
	BCHAR		= 1L<<TCHAR,
	BUCHAR		= 1L<<TUCHAR,
	BSHORT		= 1L<<TSHORT,
	BUSHORT		= 1L<<TUSHORT,
	BINT		= 1L<<TINT,
	BUINT		= 1L<<TUINT,
	BLONG		= 1L<<TLONG,
	BULONG		= 1L<<TULONG,
	BVLONG		= 1L<<TVLONG,
	BUVLONG		= 1L<<TUVLONG,
	BFLOAT		= 1L<<TFLOAT,
	BDOUBLE		= 1L<<TDOUBLE,
	BIND		= 1L<<TIND,
	BFUNC		= 1L<<TFUNC,
	BARRAY		= 1L<<TARRAY,
	BVOID		= 1L<<TVOID,
	BSTRUCT		= 1L<<TSTRUCT,
	BUNION		= 1L<<TUNION,
	BENUM		= 1L<<TENUM,
	BFILE		= 1L<<TFILE,
	BDOT		= 1L<<TDOT,
	BCONSTNT	= 1L<<TCONSTNT,
	BVOLATILE	= 1L<<TVOLATILE,
	BUNSIGNED	= 1L<<TUNSIGNED,
	BSIGNED		= 1L<<TSIGNED,
	BAUTO		= 1L<<TAUTO,
	BEXTERN		= 1L<<TEXTERN,
	BSTATIC		= 1L<<TSTATIC,
	BTYPEDEF	= 1L<<TTYPEDEF,
	BTYPESTR	= 1L<<TTYPESTR,
	BREGISTER	= 1L<<TREGISTER,

	BINTEGER	= BCHAR|BUCHAR|BSHORT|BUSHORT|BINT|BUINT|
				BLONG|BULONG|BVLONG|BUVLONG,
	BNUMBER		= BINTEGER|BFLOAT|BDOUBLE,

/* these can be overloaded with complex types */

	BCLASS		= BAUTO|BEXTERN|BSTATIC|BTYPEDEF|BTYPESTR|BREGISTER,
	BGARB		= BCONSTNT|BVOLATILE,
};

struct	Funct
{
	Sym*	sym[OEND];
	Sym*	castto[NTYPE];
	Sym*	castfr[NTYPE];
};

EXTERN struct
{
	Type*	tenum;		/* type of entire enum */
	Type*	cenum;		/* type of current enum run */
	vlong	lastenum;	/* value of current enum */
	double	floatenum;	/* value of current enum */
} en;

EXTERN	int	autobn;
EXTERN	long	autoffset;
EXTERN	int	blockno;
EXTERN	Decl*	dclstack;
EXTERN	char	debug[256];
EXTERN	Hist*	ehist;
EXTERN	long	firstbit;
EXTERN	Sym*	firstarg;
EXTERN	Type*	firstargtype;
EXTERN	Decl*	firstdcl;
EXTERN	int	fperror;
EXTERN	Sym*	hash[NHASH];
EXTERN	char*	hunk;
EXTERN	char*	include[20];
EXTERN	Io*	iofree;
EXTERN	Io*	ionext;
EXTERN	Io*	iostack;
EXTERN	long	lastbit;
EXTERN	char	lastclass;
EXTERN	Type*	lastdcl;
EXTERN	long	lastfield;
EXTERN	Type*	lasttype;
EXTERN	long	lineno;
EXTERN	long	nearln;
EXTERN	int	nerrors;
EXTERN	int	newflag;
EXTERN	long	nhunk;
EXTERN	int	ninclude;
EXTERN	Node*	nodproto;
EXTERN	Node*	nodcast;
EXTERN	Biobuf	outbuf;
EXTERN	Biobuf	diagbuf;
EXTERN	char*	outfile;
EXTERN	char*	pathname;
EXTERN	int	peekc;
EXTERN	long	stkoff;
EXTERN	Type*	strf;
EXTERN	Type*	strl;
EXTERN	char	symb[NSYMB];
EXTERN	Sym*	symstring;
EXTERN	int	taggen;
EXTERN	Type*	tfield;
EXTERN	Type*	tufield;
EXTERN	int	thechar;
EXTERN	char*	thestring;
EXTERN	Type*	thisfn;
EXTERN	long	thunk;
EXTERN	Type*	types[NTYPE];
EXTERN	Type*	fntypes[NTYPE];
EXTERN	Node*	initlist;
EXTERN	Term	term[NTERM];
EXTERN	int	nterm;
EXTERN	int	packflg;
EXTERN	int	fproundflg;
EXTERN	int	profileflg;
EXTERN	int	ncontin;
EXTERN	int	canreach;
EXTERN	int	warnreach;
EXTERN	Bits	zbits;

extern	char	*onames[], *tnames[], *gnames[];
extern	char	*cnames[], *qnames[], *bnames[];
extern	uchar	tab[NTYPE][NTYPE];
extern	uchar	comrel[], invrel[], logrel[];
extern	long	ncast[], tadd[], tand[];
extern	long	targ[], tasadd[], tasign[], tcast[];
extern	long	tdot[], tfunct[], tindir[], tmul[];
extern	long	tnot[], trel[], tsub[];

extern	uchar	typeaf[];
extern	uchar	typefd[];
extern	uchar	typei[];
extern	uchar	typesu[];
extern	uchar	typesuv[];
extern	uchar	typeu[];
extern	uchar	typev[];
extern	uchar	typec[];
extern	uchar	typeh[];
extern	uchar	typeil[];
extern	uchar	typeilp[];
extern	uchar	typechl[];
extern	uchar	typechlv[];
extern	uchar	typechlvp[];
extern	uchar	typechlp[];
extern	uchar	typechlpfd[];

EXTERN	uchar*	typeword;
EXTERN	uchar*	typecmplx;

extern	ulong	thash1;
extern	ulong	thash2;
extern	ulong	thash3;
extern	ulong	thash[];

/*
 *	compat.c/unix.c/windows.c
 */
int	mywait(int*);
int	mycreat(char*, int);
int	systemtype(int);
int	pathchar(void);
int	myaccess(char*);
char*	mygetwd(char*, int);
int	myexec(char*, char*[]);
int	mydup(int, int);
int	myfork(void);
int	mypipe(int*);
void*	mysbrk(ulong);

/*
 *	parser
 */
int	yyparse(void);
int	mpatov(char*, vlong*);

/*
 *	lex.c
 */
void*	allocn(void*, long, long);
void*	alloc(long);
void	cinit(void);
int	compile(char*, char**, int);
void	errorexit(void);
int	filbuf(void);
int	getc(void);
long	getr(void);
int	getnsc(void);
Sym*	lookup(void);
void	main(int, char*[]);
void	newfile(char*, int);
void	newio(void);
void	pushio(void);
long	escchar(long, int, int);
Sym*	slookup(char*);
void	syminit(Sym*);
void	unget(int);
long	yylex(void);
int	Lconv(Fmt*);
int	Tconv(Fmt*);
int	FNconv(Fmt*);
int	Oconv(Fmt*);
int	Qconv(Fmt*);
int	VBconv(Fmt*);
void	setinclude(char*);

/*
 * mac.c
 */
void	dodefine(char*);
void	domacro(void);
Sym*	getsym(void);
long	getnsn(void);
void	linehist(char*, int);
void	macdef(void);
void	macprag(void);
void	macend(void);
void	macexpand(Sym*, char*);
void	macif(int);
void	macinc(void);
void	maclin(void);
void	macund(void);

/*
 * dcl.c
 */
Node*	doinit(Sym*, Type*, long, Node*);
Type*	tcopy(Type*);
Node*	init1(Sym*, Type*, long, int);
Node*	newlist(Node*, Node*);
void	adecl(int, Type*, Sym*);
int	anyproto(Node*);
void	argmark(Node*, int);
void	dbgdecl(Sym*);
Node*	dcllabel(Sym*, int);
Node*	dodecl(void(*)(int, Type*, Sym*), int, Type*, Node*);
Sym*	mkstatic(Sym*);
void	doenum(Sym*, Node*);
void	snap(Type*);
Type*	dotag(Sym*, int, int);
void	edecl(int, Type*, Sym*);
Type*	fnproto(Node*);
Type*	fnproto1(Node*);
void	markdcl(void);
Type*	paramconv(Type*, int);
void	pdecl(int, Type*, Sym*);
Decl*	push(void);
Decl*	push1(Sym*);
Node*	revertdcl(void);
long	xround(long, int);
int	rsametype(Type*, Type*, int, int);
int	sametype(Type*, Type*);
ulong	sign(Sym*);
ulong	signature(Type*);
void	suallign(Type*);
void	tmerge(Type*, Sym*);
void	walkparam(Node*, int);
void	xdecl(int, Type*, Sym*);
Node*	contig(Sym*, Node*, long);

/*
 * com.c
 */
void	ccom(Node*);
void	complex(Node*);
int	tcom(Node*);
int	tcoma(Node*, Node*, Type*, int);
int	tcomd(Node*);
int	tcomo(Node*, int);
int	tcomx(Node*);
int	tlvalue(Node*);
void	constas(Node*, Type*, Type*);

/*
 * con.c
 */
void	acom(Node*);
void	acom1(vlong, Node*);
void	acom2(Node*, Type*);
int	acomcmp1(const void*, const void*);
int	acomcmp2(const void*, const void*);
int	addo(Node*);
void	evconst(Node*);

/*
 * funct.c
 */
int	isfunct(Node*);
void	dclfunct(Type*, Sym*);

/*
 * sub.c
 */
void	arith(Node*, int);
int	deadheads(Node*);
Type*	dotsearch(Sym*, Type*, Node*, long*);
long	dotoffset(Type*, Type*, Node*);
void	gethunk(void);
Node*	invert(Node*);
int	bitno(long);
void	makedot(Node*, Type*, long);
int	mixedasop(Type*, Type*);
Node*	new(int, Node*, Node*);
Node*	new1(int, Node*, Node*);
int	nilcast(Type*, Type*);
int	nocast(Type*, Type*);
void	prtree(Node*, char*);
void	prtree1(Node*, int, int);
void	relcon(Node*, Node*);
int	relindex(int);
int	simpleg(long);
Type*	garbt(Type*, long);
int	simplec(long);
Type*	simplet(long);
int	stcompat(Node*, Type*, Type*, long[]);
int	tcompat(Node*, Type*, Type*, long[]);
void	tinit(void);
Type*	typ(int, Type*);
Type*	copytyp(Type*);
void	typeext(Type*, Node*);
void	typeext1(Type*, Node*);
int	side(Node*);
int	vconst(Node*);
int	xlog2(uvlong);
int	vlog(Node*);
int	topbit(ulong);
void	simplifyshift(Node*);
long	typebitor(long, long);
void	diag(Node*, char*, ...);
void	warn(Node*, char*, ...);
void	yyerror(char*, ...);
void	fatal(Node*, char*, ...);

/*
 * acid.c
 */
void	acidtype(Type*);
void	acidvar(Sym*);

/*
 * pickle.c
 */
void	pickletype(Type*);

/*
 * bits.c
 */
Bits	bor(Bits, Bits);
Bits	band(Bits, Bits);
Bits	bnot(Bits);
int	bany(Bits*);
int	bnum(Bits);
Bits	blsh(uint);
int	beq(Bits, Bits);
int	bset(Bits, uint);

/*
 * dpchk.c
 */
void	dpcheck(Node*);
void	arginit(void);
void	pragvararg(void);
void	pragpack(void);
void	pragfpround(void);
void pragprofile(void);
void	pragincomplete(void);

/*
 * calls to machine depend part
 */
void	codgen(Node*, Node*);
void	gclean(void);
void	gextern(Sym*, Node*, long, long);
void	ginit(void);
long	outstring(char*, long);
long	outlstring(ushort*, long);
void	sextern(Sym*, Node*, long, long);
void	xcom(Node*);
long	exreg(Type*);
long	align(long, Type*, int);
long	maxround(long, long);

extern	schar	ewidth[];

/*
 * com64
 */
int	com64(Node*);
void	com64init(void);
void	bool64(Node*);
double	convvtof(vlong);
vlong	convftov(double);
double	convftox(double, int);
vlong	convvtox(vlong, int);

/*
 * machcap
 */
int	machcap(Node*);

#pragma	varargck	argpos	warn	2
#pragma	varargck	argpos	diag	2
#pragma	varargck	argpos	yyerror	1

#pragma	varargck	type	"F"	Node*
#pragma	varargck	type	"L"	long
#pragma	varargck	type	"Q"	long
#pragma	varargck	type	"O"	int
#pragma	varargck	type	"T"	Type*
#pragma	varargck	type	"|"	int
