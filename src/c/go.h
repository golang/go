// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
todo:
	1. dyn arrays
	2. multi
	3. block 0
tothinkabout:
	1. alias name name.name.name
	2. argument in import
*/

#include	<u.h>
#include	<libc.h>
#include	<bio.h>

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

/* note this is the representation
 * of the compilers string literals,
 * it happens to also be the runtime
 * representation, but that may change */
typedef	struct	String	String;
struct	String
{
	long	len;
	uchar	s[3];	// variable
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
struct	Node
{
	int	op;

	// most nodes
	Node*	left;
	Node*	right;
	Node*	type;

	// for-body
	Node*	ninit;
	Node*	ntest;
	Node*	nincr;
	Node*	nbody;

	// if-body
	Node*	nelse;

	// OTYPE-TFIELD
	Node*	down;		// also used in TMAP
	Node*	uberstruct;

	// cases
	Node*	ncase;

	// OTYPE-TPTR
	Node*	nforw;

	// OTYPE-TFUNCT
	Node*	this;
	Node*	argout;
	Node*	argin;
	Node*	nname;
	int	thistuple;
	int	outtuple;
	int	intuple;

	// OTYPE-TARRAY
	long	bound;

	// OLITERAL
	Val	val;

	Sym*	osym;		// import
	Sym*	fsym;		// import
	Sym*	psym;		// import
	Sym*	sym;		// various
	uchar	ullman;		// sethi/ullman number
	uchar	addable;	// type of addressability - 0 is not addressable
	uchar	recur;		// to detect loops
	uchar	trecur;		// to detect loops
	uchar	etype;		// is an op for OASOP, is etype for OTYPE
	uchar	chan;
	uchar	kaka;
	uchar	multi;		// type of assignment or call
	long	vargen;		// unique name for OTYPE/ONAME
	long	lineno;
};
#define	N	((Node*)0)

struct	Sym
{
	char*	opackage;	// original package name
	char*	package;	// package name
	char*	name;		// variable name
	Node*	oname;		// ONAME node if a var
	Node*	otype;		// OTYPE node if a type
	Node*	oconst;		// OLITERAL node if a const
	Node*	forwtype;	// OTYPE/TPTR iff foreward declared
	void*	label;		// pointer to Prog* of label
	long	lexical;
	long	vargen;		// unique variable number
	uchar	undef;		// a diagnostic has been generated
	uchar	export;		// marked as export
	uchar	exported;	// has been exported
	Sym*	link;
};
#define	S	((Sym*)0)

typedef	struct	Dcl	Dcl;
struct	Dcl
{
	int	op;		// ONAME for var, OTYPE for type, Oxxx for const
	Sym*	dsym;		// for printing only
	Node*	dnode;		// otype or oname
	long	lineno;

	Dcl*	forw;
	Dcl*	back;		// sentinel has pointer to last
};
#define	D	((Dcl*)0)

typedef	struct	Iter	Iter;
struct	Iter
{
	int	done;
	Node**	an;
	Node*	n;
};

enum
{
	OXXX,

	OTYPE, OCONST, OVAR, OEXPORT, OIMPORT,

	ONAME,
	ODOT, ODOTPTR, ODOTMETH, ODOTINTER,
	ODCLFUNC, ODCLFIELD, ODCLARG,
	OLIST,
	OPTR, OARRAY,
	ORETURN, OFOR, OIF, OSWITCH,
	OAS, OASOP, OCASE, OXCASE, OFALL, OXFALL,
	OGOTO, OPROC, ONEW, OPANIC, OPRINT, OEMPTY,

	OOROR,
	OANDAND,
	OEQ, ONE, OLT, OLE, OGE, OGT,
	OADD, OSUB, OOR, OXOR, OCAT,
	OMUL, ODIV, OMOD, OLSH, ORSH, OAND,
	ODEC, OINC,
	OLEN,
	OFUNC,
	OLABEL,
	OBREAK,
	OCONTINUE,
	OADDR,
	OIND,
	OCALL, OCALLPTR, OCALLMETH, OCALLINTER,
	OINDEX, OINDEXSTR, OINDEXMAP,
	OINDEXPTR, OINDEXPTRSTR, OINDEXPTRMAP,
	OSLICE, OSLICESTR, OSLICEPTRSTR,
	ONOT, OCOM, OPLUS, OMINUS, OSEND, ORECV,
	OLITERAL,
	OCONV,
	OBAD,

	OEND,
};
enum
{
	Txxx,

	TINT8,	TUINT8,
	TINT16,	TUINT16,
	TINT32,	TUINT32,
	TINT64,	TUINT64,

	TFLOAT32,
	TFLOAT64,
	TFLOAT80,

	TBOOL,

	TPTR,
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

	NTYPE,
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

	PCALL_NIL,	// no return value
	PCALL_SINGLE,	// single return value
	PCALL_MULTI,	// multiple return values

	PAS_SINGLE,	// top level walk->gen hints for OAS
	PAS_MULTI,	// multiple values
	PAS_CALLM,	// multiple values from a call
	PAS_STRUCT,	// structure assignment
};

typedef	struct	Io	Io;
struct	Io
{
	char*	infile;
	Biobuf*	bin;
	long	lineno;
	int	peekc;
};

EXTERN	Io	curio;
EXTERN	Io	pushedio;

EXTERN	char*	outfile;
EXTERN	char*	package;
EXTERN	Biobuf*	bout;
EXTERN	int	nerrors;
EXTERN	char	namebuf[NSYMB];
EXTERN	char	debug[256];
EXTERN	long	dynlineno;
EXTERN	Sym*	hash[NHASH];
EXTERN	Sym*	dclstack;
EXTERN	Sym*	b0stack;
EXTERN	Sym*	pkgmyname;	// my name for package

EXTERN	Node*	types[NTYPE];
EXTERN	uchar	isint[NTYPE];
EXTERN	uchar	isfloat[NTYPE];
EXTERN	uchar	okforeq[NTYPE];
EXTERN	uchar	okforadd[NTYPE];
EXTERN	uchar	okforand[NTYPE];
EXTERN	double	minfloatval[NTYPE];
EXTERN	double	maxfloatval[NTYPE];
EXTERN	vlong	minintval[NTYPE];
EXTERN	vlong	maxintval[NTYPE];

EXTERN	Dcl*	autodcl;
EXTERN	Dcl*	externdcl;
EXTERN	Dcl*	exportlist;
EXTERN	int	dclcontext;	// PEXTERN/PAUTO
EXTERN	int	importflag;

EXTERN	Node*	booltrue;
EXTERN	Node*	boolfalse;
EXTERN	ulong	iota;
EXTERN	long	vargen;
EXTERN	long	exportgen;

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
int	main(int, char*[]);
void	importfile(Val*);
void	unimportfile();
long	yylex(void);
void	lexinit(void);
char*	lexname(int);
long	getr(void);
int	getnsc(void);
long	escchar(long, int*);
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
Node*	nod(int, Node*, Node*);
Dcl*	dcl(void);
Node*	rev(Node*);
Node*	unrev(Node*);
void	dodump(Node*, int);
void	dump(char*, Node*);
Node*	aindex(Node*, Node*);
int	isnil(Node*);
int	isptrto(Node*, int);
int	isinter(Node*);
int	isbytearray(Node*);
int	eqtype(Node*, Node*, int);
int	eqargs(Node*, Node*);
ulong	typehash(Node*, int);
void	frame(int);
Node*	literal(long);
Node*	dobad(void);
void	ullmancalc(Node*);
void	badtype(int, Node*, Node*);
Node*	ptrto(Node*);
Node*	cleanidlist(Node*);

Node**	getthis(Node*);
Node**	getoutarg(Node*);
Node**	getinarg(Node*);

Node*	getthisx(Node*);
Node*	getoutargx(Node*);
Node*	getinargx(Node*);

Node*	listfirst(Iter*, Node**);
Node*	listnext(Iter*);
Node*	structfirst(Iter*, Node**);
Node*	structnext(Iter*);

int	Econv(Fmt*);
int	Jconv(Fmt*);
int	Oconv(Fmt*);
int	Sconv(Fmt*);
int	Tconv(Fmt*);
int	Nconv(Fmt*);
int	Zconv(Fmt*);

/*
 *	dcl.c
 */
void	dodclvar(Node*, Node*);
void	dodcltype(Node*, Node*);
void	dodclconst(Node*, Node*);
void	defaultlit(Node*);
int	listcount(Node*);
Node*	functype(Node*, Node*, Node*);
char*	thistypenam(Node*);
void	funcnam(Node*, char*);
void	funchdr(Node*);
void	funcargs(Node*);
void	funcbody(Node*);
Node*	dostruct(Node*, int);
Node**	stotype(Node*, Node**, Node*);
Node*	sortinter(Node*);
void	markdcl(char*);
void	popdcl(char*);
void	poptodcl(void);
void	markdclstack(void);
void	testdclstack(void);
Sym*	pushdcl(Sym*);
void	addvar(Node*, Node*, int);
void	addtyp(Node*, Node*, int);
Node*	newname(Sym*);
Node*	oldname(Sym*);
Node*	newtype(Sym*);
Node*	oldtype(Sym*);
Node*	forwdcl(Sym*);

/*
 *	export.c
 */
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

/*
 *	walk.c
 */
void	walk(Node*);
void	walktype(Node*, int);
Node*	walkswitch(Node*, Node*, Node*(*)(Node*, Node*));
int	casebody(Node*);
int	whatis(Node*);
void	walkdot(Node*);
void	walkslice(Node*);
void	ascompatee(int, Node**, Node**);
void	ascompatet(int, Node**, Node**);
void	ascompatte(int, Node**, Node**);
void	ascompattt(int, Node**, Node**);
int	ascompat(Node*, Node*);
void	prcompat(Node**);

/*
 *	const.c
 */
void	convlit(Node*, Node*);
void	evconst(Node*);
int	cmpslit(Node *l, Node *r);

/*
 *	gen.c/gsubr.c/obj.c
 */
void	belexinit(int);
vlong	convvtox(vlong, int);
void	compile(Node*);
void	proglist(void);
void	dumpobj(void);
int	optopop(int);
