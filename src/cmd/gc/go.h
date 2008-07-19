// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
todo:
	1. dyn arrays
	2. multi
	3. block 0
tothinkabout:
	2. argument in import
*/

#include	<u.h>
#include	<libc.h>
#include	<bio.h>
#include	"compat.h"

#ifndef	EXTERN
#define EXTERN	extern
#endif
enum
{
	NHUNK		= 50000,
	BUFSIZ		= 8192,
	NSYMB		= 500,
	NHASH		= 1024,
	STRINGSZ	= 200,
	YYMAXDEPTH	= 500,
	MAXALIGN	= 7,
	UINF		= 100,
	HISTSZ		= 10,

	PRIME1		= 3,
	PRIME2		= 10007,
	PRIME3		= 10009,
	PRIME4		= 10037,
	PRIME5		= 10039,
	PRIME6		= 10061,
	PRIME7		= 10067,
	PRIME8		= 10079,
	PRIME9		= 10091,
};

/*
 * note this is the representation
 * of the compilers string literals,
 * it happens to also be the runtime
 * representation, ignoring sizes and
 * alignment, but that may change.
 */
typedef	struct	String	String;
struct	String
{
	long	len;
	char	s[3];	// variable
};

typedef	struct	Val	Val;
struct	Val
{
	int	ctype;
	double	dval;
	vlong	vval;
	String*	sval;
};

typedef	struct	Sym	Sym;
typedef	struct	Node	Node;
typedef	struct	Type	Type;

struct	Type
{
	uchar	etype;
	uchar	chan;
	uchar	recur;		// to detect loops
	uchar	trecur;		// to detect loops

	// TFUNCT
	uchar	thistuple;
	uchar	outtuple;
	uchar	intuple;
	uchar	outnamed;

	Sym*	sym;
	long	vargen;		// unique name for OTYPE/ONAME

	// most nodes
	Type*	type;
	vlong	width;		// offset in TFIELD, width in all others

	// TFIELD
	Type*	down;		// also used in TMAP

	// TPTR
	Type*	nforw;

	// TFUNCT
	Node*	nname;
	vlong	argwid;

	// TARRAY
	long	bound;
};
#define	T	((Type*)0)

struct	Node
{
	uchar	op;
	uchar	ullman;		// sethi/ullman number
	uchar	addable;	// type of addressability - 0 is not addressable
	uchar	trecur;		// to detect loops
	uchar	etype;		// op for OASOP, etype for OTYPE, exclam for export
	uchar	class;		// PPARAM, PAUTO, PEXTERN, PSTATIC
	uchar	method;		// OCALLMETH name
	uchar	iota;		// OLITERAL made from iota

	// most nodes
	Node*	left;
	Node*	right;
	Type*	type;

	// for-body
	Node*	ninit;
	Node*	ntest;
	Node*	nincr;
	Node*	nbody;

	// if-body
	Node*	nelse;

	// cases
	Node*	ncase;

	// func
	Node*	nname;

	// OLITERAL
	Val	val;

	Sym*	osym;		// import
	Sym*	fsym;		// import
	Sym*	psym;		// import
	Sym*	sym;		// various
	long	vargen;		// unique name for OTYPE/ONAME
	long	lineno;
	vlong	xoffset;
};
#define	N	((Node*)0)

struct	Sym
{
	ushort	tblock;
	ushort	vblock;

	uchar	undef;		// a diagnostic has been generated
	uchar	export;		// marked as export
	uchar	exported;	// has been exported
	uchar	sym;		// huffman encoding in object file

	char*	opackage;	// original package name
	char*	package;	// package name
	char*	name;		// variable name
	Node*	oname;		// ONAME node if a var
	Type*	otype;		// TYPE node if a type
	Node*	oconst;		// OLITERAL node if a const
	Type*	forwtype;	// TPTR iff forward declared
	void*	label;		// pointer to Prog* of label
	vlong	offset;		// stack location if automatic
	long	lexical;
	long	vargen;		// unique variable number
	Sym*	link;
};
#define	S	((Sym*)0)

typedef	struct	Dcl	Dcl;
struct	Dcl
{
	uchar	op;
	Sym*	dsym;		// for printing only
	Node*	dnode;		// oname
	Type*	dtype;		// otype
	long	lineno;

	Dcl*	forw;
	Dcl*	back;		// sentinel has pointer to last
};
#define	D	((Dcl*)0)

typedef	struct	Iter	Iter;
struct	Iter
{
	int	done;
	Type*	tfunc;
	Type*	t;
	Node**	an;
	Node*	n;
};

typedef	struct	Hist	Hist;
struct	Hist
{
	Hist*	link;
	char*	name;
	long	line;
	long	offset;
};
#define	H	((Hist*)0)

enum
{
	OXXX,

	OTYPE, OCONST, OVAR, OEXPORT, OIMPORT,

	ONAME, ONONAME,
	ODOT, ODOTPTR, ODOTMETH, ODOTINTER,
	ODCLFUNC, ODCLFIELD, ODCLARG,
	OLIST, OCMP,
	OPTR, OARRAY,
	ORETURN, OFOR, OIF, OSWITCH, OI2S, OS2I, OI2I,
	OAS, OASOP, OCASE, OXCASE, OFALL, OXFALL,
	OGOTO, OPROC, ONEW, OPANIC, OPRINT, OEMPTY,

	OOROR,
	OANDAND,
	OEQ, ONE, OLT, OLE, OGE, OGT,
	OADD, OSUB, OOR, OXOR,
	OMUL, ODIV, OMOD, OLSH, ORSH, OAND,
	OLEN,
	OFUNC,
	OLABEL,
	OBREAK,
	OCONTINUE,
	OADDR,
	OIND,
	OCALL, OCALLMETH, OCALLINTER,
	OINDEX, OINDEXPTR, OSLICE,
	ONOT, OCOM, OPLUS, OMINUS, OSEND, ORECV,
	OLITERAL, OREGISTER, OINDREG,
	OCONV,
	OBAD,

	OEND,
};
enum
{
	Txxx,			// 0

	TINT8,	TUINT8,		// 1
	TINT16,	TUINT16,
	TINT32,	TUINT32,
	TINT64,	TUINT64,

	TFLOAT32,		// 9
	TFLOAT64,
	TFLOAT80,

	TBOOL,			// 12

	TPTR32, TPTR64,		// 13

	TFUNC,
	TARRAY,
	TDARRAY,
	TSTRUCT,
	TCHAN,
	TMAP,
	TINTER,
	TFORW,
	TFIELD,
	TANY,
	TSTRING,

	NTYPE,			// 26
};
enum
{
	CTxxx,

	CTINT,
	CTSINT,
	CTUINT,
	CTFLT,

	CTSTR,
	CTBOOL,
	CTNIL,
};

enum
{
	/* indications for whatis() */
	Wnil	= 0,
	Wtnil,

	Wtfloat,
	Wtint,
	Wtbool,
	Wtstr,

	Wlitfloat,
	Wlitint,
	Wlitbool,
	Wlitstr,
	Wlitnil,

	Wtunkn,
};

enum
{
	/* types of channel */
	Cxxx,
	Cboth,
	Crecv,
	Csend,
};

enum
{
	Pxxx,

	PEXTERN,	// declaration context
	PAUTO,
	PPARAM,
	PSTATIC,
};

enum
{
	Exxx,
	Eyyy,
	Etop,		// evaluated at statement level
	Elv,		// evaluated in lvalue context
	Erv,		// evaluated in rvalue context
};

typedef	struct	Io	Io;
struct	Io
{
	char*	infile;
	Biobuf*	bin;
	long	ilineno;
	int	peekc;
	char*	cp;	// used for content when bin==nil
};

EXTERN	Io	curio;
EXTERN	Io	pushedio;
EXTERN	long	lineno;
EXTERN	char*	pathname;
EXTERN	Hist*	hist;
EXTERN	Hist*	ehist;


EXTERN	char*	infile;
EXTERN	char*	outfile;
EXTERN	char*	package;
EXTERN	Biobuf*	bout;
EXTERN	int	nerrors;
EXTERN	char	namebuf[NSYMB];
EXTERN	char	debug[256];
EXTERN	Sym*	hash[NHASH];
EXTERN	Sym*	dclstack;
EXTERN	Sym*	b0stack;
EXTERN	Sym*	pkgmyname;	// my name for package
EXTERN	Sym*	pkgimportname;	// package name from imported package
EXTERN	int	tptr;		// either TPTR32 or TPTR64
extern	char*	sysimport;
EXTERN	char*	filename;	// name to uniqify names

EXTERN	Type*	types[NTYPE];
EXTERN	uchar	isptr[NTYPE];
EXTERN	uchar	isint[NTYPE];
EXTERN	uchar	isfloat[NTYPE];
EXTERN	uchar	issigned[NTYPE];
EXTERN	uchar	issimple[NTYPE];
EXTERN	uchar	okforeq[NTYPE];
EXTERN	uchar	okforadd[NTYPE];
EXTERN	uchar	okforand[NTYPE];
EXTERN	double	minfloatval[NTYPE];
EXTERN	double	maxfloatval[NTYPE];
EXTERN	vlong	minintval[NTYPE];
EXTERN	vlong	maxintval[NTYPE];

EXTERN	Dcl*	autodcl;
EXTERN	Dcl*	paramdcl;
EXTERN	Dcl*	externdcl;
EXTERN	Dcl*	exportlist;
EXTERN	int	dclcontext;	// PEXTERN/PAUTO
EXTERN	int	importflag;
EXTERN	int	inimportsys;

EXTERN	Node*	booltrue;
EXTERN	Node*	boolfalse;
EXTERN	ulong	iota;
EXTERN	Node*	lastconst;
EXTERN	long	vargen;
EXTERN	long	exportgen;
EXTERN	long	maxarg;
EXTERN	long	stksize;
EXTERN	ushort	blockgen;		// max block number
EXTERN	ushort	block;			// current block number

EXTERN	Node*	retnil;
EXTERN	Node*	fskel;

EXTERN	char*	context;
EXTERN	int	thechar;
EXTERN	char*	thestring;
EXTERN	char*	hunk;
EXTERN	long	nhunk;
EXTERN	long	thunk;

/*
 *	y.tab.c
 */
int	yyparse(void);

/*
 *	lex.c
 */
int	mainlex(int, char*[]);
void	setfilename(char*);
void	importfile(Val*);
void	cannedimports(void);
void	unimportfile();
long	yylex(void);
void	lexinit(void);
char*	lexname(int);
long	getr(void);
int	getnsc(void);
int	escchar(int, int*, vlong*);
int	getc(void);
void	ungetc(int);
void	mkpackage(char*);

/*
 *	mpatof.c
 */
int	mpatof(char*, double*);
int	mpatov(char*, vlong*);

/*
 *	subr.c
 */
void	myexit(int);
void*	mal(long);
void*	remal(void*, long, long);
void	errorexit(void);
ulong	stringhash(char*);
Sym*	lookup(char*);
Sym*	pkglookup(char*, char*);
void	yyerror(char*, ...);
void	warn(char*, ...);
void	fatal(char*, ...);
void	linehist(char*, long);
long	setlineno(Node*);
Node*	nod(int, Node*, Node*);
Node*	list(Node*, Node*);
Type*	typ(int);
Dcl*	dcl(void);
Node*	rev(Node*);
Node*	unrev(Node*);
void	dodump(Node*, int);
void	dump(char*, Node*);
Type*	aindex(Node*, Type*);
int	isnil(Node*);
int	isptrto(Type*, int);
int	isinter(Type*);
int	isbytearray(Type*);
int	eqtype(Type*, Type*, int);
void	argtype(Node*, Type*);
int	eqargs(Type*, Type*);
ulong	typehash(Type*, int);
void	frame(int);
Node*	literal(long);
Node*	dobad(void);
Node*	nodintconst(long);
void	ullmancalc(Node*);
void	badtype(int, Type*, Type*);
Type*	ptrto(Type*);
Node*	cleanidlist(Node*);
Node*	syslook(char*, int);
Node*	treecopy(Node*);

Type**	getthis(Type*);
Type**	getoutarg(Type*);
Type**	getinarg(Type*);

Type*	getthisx(Type*);
Type*	getoutargx(Type*);
Type*	getinargx(Type*);

Node*	listfirst(Iter*, Node**);
Node*	listnext(Iter*);
Type*	structfirst(Iter*, Type**);
Type*	structnext(Iter*);
Type*	funcfirst(Iter*, Type*);
Type*	funcnext(Iter*);

int	Econv(Fmt*);
int	Jconv(Fmt*);
int	Lconv(Fmt*);
int	Oconv(Fmt*);
int	Sconv(Fmt*);
int	Tconv(Fmt*);
int	Nconv(Fmt*);
int	Zconv(Fmt*);

/*
 *	dcl.c
 */
void	dodclvar(Node*, Type*);
void	dodcltype(Type*, Type*);
void	dodclconst(Node*, Node*);
void	defaultlit(Node*);
int	listcount(Node*);
Node*	methodname(Node*, Type*);
Type*	functype(Node*, Node*, Node*);
char*	thistypenam(Node*);
void	funcnam(Type*, char*);
void	funchdr(Node*);
void	funcargs(Type*);
void	funcbody(Node*);
Type*	dostruct(Node*, int);
Type**	stotype(Node*, Type**);
Type*	sortinter(Type*);
void	markdcl(void);
void	popdcl(void);
void	poptodcl(void);
void	dumpdcl(char*);
void	markdclstack(void);
void	testdclstack(void);
Sym*	pushdcl(Sym*);
void	addvar(Node*, Type*, int);
void	addtyp(Type*, Type*, int);
Node*	fakethis(void);
Node*	newname(Sym*);
Node*	oldname(Sym*);
Type*	newtype(Sym*);
Type*	oldtype(Sym*);
Type*	forwdcl(Sym*);
void	fninit(Node*);

/*
 *	export.c
 */
void	renamepkg(Node*);
void	markexport(Node*);
void	dumpe(Sym*);
void	dumpexport(void);
void	dumpexporttype(Sym*);
void	dumpexportvar(Sym*);
void	dumpexportconst(Sym*);
void	doimportv1(Node*, Node*);
void	doimportc1(Node*, Val*);
void	doimportc2(Node*, Node*, Val*);
void	doimport1(Node*, Node*, Node*);
void	doimport2(Node*, Val*, Node*);
void	doimport3(Node*, Node*);
void	doimport4(Node*, Node*);
void	doimport5(Node*, Val*);
void	doimport6(Node*, Node*);
void	doimport7(Node*, Node*);
void	doimport8(Node*, Val*, Node*);

/*
 *	walk.c
 */
void	walk(Node*);
void	walktype(Node*, int);
void	walkbool(Node*);
Type*	walkswitch(Node*, Type*(*)(Node*, Type*));
int	casebody(Node*);
int	whatis(Node*);
void	walkdot(Node*, int);
Node*	ascompatee(int, Node**, Node**);
Node*	ascompatet(int, Node**, Type**, int);
Node*	ascompatte(int, Type**, Node**, int);
int	ascompat(Type*, Type*);
Node*	prcompat(Node*);
Node*	nodpanic(long);
Node*	newcompat(Node*);
Node*	stringop(Node*, int);
Node*	mapop(Node*, int);
Node*	chanop(Node*, int);
Node*	convas(Node*);
void	arrayconv(Type*, Node*);
Node*	colas(Node*, Node*);
Node*	reorder1(Node*);
Node*	reorder2(Node*);
Node*	reorder3(Node*);
Node*	reorder4(Node*);

/*
 *	const.c
 */
void	convlit(Node*, Type*);
void	evconst(Node*);
int	cmpslit(Node *l, Node *r);

/*
 *	gen.c/gsubr.c/obj.c
 */
void	belexinit(int);
void	besetptr(void);
vlong	convvtox(vlong, int);
void	compile(Node*);
void	proglist(void);
int	optopop(int);
void	dumpobj(void);
void	dowidth(Type*);
void	argspace(long);
Node*	nodarg(Type*, int);
void	nodconst(Node*, Type*, vlong);
