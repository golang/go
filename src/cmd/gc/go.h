// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
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

	AUNK		= 100,
	// these values are known by runtime
	ASIMP		= 0,
	ASTRING,
	APTR,
	AINTER,
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
	int32	len;
	char	s[3];	// variable
};

/*
 * note this is the runtime representation
 * of the compilers arrays. it is probably
 * insafe to use it this way, but it puts
 * all the changes in one place.
 */
typedef	struct	Array	Array;
struct	Array
{				// must not move anything
	uchar	array[8];	// pointer to data
	uint32	nel;		// number of elements
	uint32	cap;		// allocated number of elements
	uchar	b;		// actual array - may not be contig
};

enum
{
	Mpscale	= 29,		/* safely smaller than bits in a long */
	Mpprec	= 10,		/* Mpscale*Mpprec is max number of bits */
	Mpbase	= 1L<<Mpscale,
	Mpsign	= Mpbase >> 1,
	Mpmask	= Mpbase -1,
	Debug	= 1,
};

typedef	struct	Mpint	Mpint;
struct	Mpint
{
	vlong	val;
	long	a[Mpprec];
	uchar	neg;
	uchar	ovf;
};

typedef	struct	Mpflt	Mpflt;
struct	Mpflt
{
	double	val;
	uchar	ovf;
};

typedef	struct	Val	Val;
struct	Val
{
	short	ctype;
	union
	{
		short	reg;		// OREGISTER
		short	bval;		// bool value CTBOOL
		Mpint*	xval;		// int CTINT
		Mpflt*	fval;		// float CTFLT
		String*	sval;		// string CTSTR
	} u;
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
	uchar	methptr;	// all methods are pointers to this type

	// TFUNCT
	uchar	thistuple;
	uchar	outtuple;
	uchar	intuple;
	uchar	outnamed;

	Type*	method;

	Sym*	sym;
	int32	vargen;		// unique name for OTYPE/ONAME

	Node*	nname;
	vlong	argwid;

	// most nodes
	Type*	type;
	vlong	width;		// offset in TFIELD, width in all others

	// TFIELD
	Type*	down;		// also used in TMAP

	// TPTR
	Type*	nforw;

	// TARRAY
	int32	bound;		// negative is dynamic array
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

	// OLITERAL/OREGISTER
	Val	val;

	Sym*	osym;		// import
	Sym*	fsym;		// import
	Sym*	psym;		// import
	Sym*	sym;		// various
	int32	vargen;		// unique name for OTYPE/ONAME
	int32	lineno;
	vlong	xoffset;
};
#define	N	((Node*)0)

struct	Sym
{
	ushort	tblock;
	ushort	vblock;

	uchar	undef;		// a diagnostic has been generated
	uchar	export;		// marked as export
	uchar	exported;	// exported
	uchar	sym;		// huffman encoding in object file
	uchar	local;		// created in this file

	char*	opackage;	// original package name
	char*	package;	// package name
	char*	name;		// variable name
	Node*	oname;		// ONAME node if a var
	Type*	otype;		// TYPE node if a type
	Node*	oconst;		// OLITERAL node if a const
	Type*	forwtype;	// TPTR iff forward declared
	vlong	offset;		// stack location if automatic
	int32	lexical;
	int32	vargen;		// unique variable number
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
	int32	lineno;

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
	int32	line;
	int32	offset;
};
#define	H	((Hist*)0)

enum
{
	OXXX,

	OTYPE, OCONST, OVAR, OIMPORT,

	ONAME, ONONAME,
	ODOT, ODOTPTR, ODOTMETH, ODOTINTER,
	ODCLFUNC, ODCLFIELD, ODCLARG,
	OLIST, OCMP, OPTR, OARRAY,
	ORETURN, OFOR, OIF, OSWITCH,
	OAS, OASOP, OCASE, OXCASE, OFALL, OXFALL,
	OGOTO, OPROC, ONEW, OEMPTY, OSELECT,
	OLEN, OCAP, OPANIC, OPANICN, OPRINT, OPRINTN, OTYPEOF,

	OOROR,
	OANDAND,
	OEQ, ONE, OLT, OLE, OGE, OGT,
	OADD, OSUB, OOR, OXOR,
	OMUL, ODIV, OMOD, OLSH, ORSH, OAND,
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
	OCONV, OKEY,
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
	T_old_DARRAY,
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
	int32	ilineno;
	int	peekc;
	char*	cp;	// used for content when bin==nil
};

EXTERN	Io	curio;
EXTERN	Io	pushedio;
EXTERN	int32	lineno;
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
EXTERN	int	exportadj;	// declaration is being exported

EXTERN	Type*	types[NTYPE];
EXTERN	uchar	isptr[NTYPE];
EXTERN	uchar	isint[NTYPE];
EXTERN	uchar	isfloat[NTYPE];
EXTERN	uchar	issigned[NTYPE];
EXTERN	uchar	issimple[NTYPE];
EXTERN	uchar	okforeq[NTYPE];
EXTERN	uchar	okforadd[NTYPE];
EXTERN	uchar	okforand[NTYPE];

EXTERN	Mpint*	minintval[NTYPE];
EXTERN	Mpint*	maxintval[NTYPE];
EXTERN	Mpflt*	minfltval[NTYPE];
EXTERN	Mpflt*	maxfltval[NTYPE];

EXTERN	Dcl*	autodcl;
EXTERN	Dcl*	paramdcl;
EXTERN	Dcl*	externdcl;
EXTERN	Dcl*	exportlist;
EXTERN	Dcl*	signatlist;
EXTERN	int	dclcontext;	// PEXTERN/PAUTO
EXTERN	int	importflag;
EXTERN	int	inimportsys;

EXTERN	Node*	booltrue;
EXTERN	Node*	boolfalse;
EXTERN	uint32	iota;
EXTERN	Node*	lastconst;
EXTERN	int32	vargen;
EXTERN	int32	exportgen;
EXTERN	int32	maxarg;
EXTERN	int32	stksize;
EXTERN	ushort	blockgen;		// max block number
EXTERN	ushort	block;			// current block number

EXTERN	Node*	retnil;
EXTERN	Node*	fskel;

EXTERN	char*	context;
EXTERN	int	thechar;
EXTERN	char*	thestring;
EXTERN	char*	hunk;
EXTERN	int32	nhunk;
EXTERN	int32	thunk;

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
int32	yylex(void);
void	lexinit(void);
char*	lexname(int);
int32	getr(void);
int	getnsc(void);
int	escchar(int, int*, vlong*);
int	getc(void);
void	ungetc(int);
void	mkpackage(char*);

/*
 *	mparith1.c
 */
int	mpcmpfixflt(Mpint *a, Mpflt *b);
int	mpcmpfltfix(Mpflt *a, Mpint *b);
int	mpcmpfixfix(Mpint *a, Mpint *b);
int	mpcmpfixc(Mpint *b, vlong c);
int	mpcmpfltflt(Mpflt *a, Mpflt *b);
int	mpcmpfltc(Mpflt *b, double c);
void	mpsubfixfix(Mpint *a, Mpint *b);
void	mpsubfltflt(Mpflt *a, Mpflt *b);
void	mpaddcfix(Mpint *a, vlong c);
void	mpaddcflt(Mpflt *a, double c);
void	mpmulcfix(Mpint *a, vlong c);
void	mpmulcflt(Mpflt *a, double c);
void	mpdivfixfix(Mpint *a, Mpint *b);
void	mpmodfixfix(Mpint *a, Mpint *b);
void	mpatofix(Mpint *a, char *s);
void	mpatoflt(Mpflt *a, char *s);
void	mpmovefltfix(Mpint *a, Mpflt *b);
void	mpmovefixflt(Mpflt *a, Mpint *b);
int	Bconv(Fmt*);

/*
 *	mparith2.c
 */
void	mpmovefixfix(Mpint *a, Mpint *b);
void	mpmovecfix(Mpint *a, vlong v);
int	mptestfix(Mpint *a);
void	mpaddfixfix(Mpint *a, Mpint *b);
void	mpmulfixfix(Mpint *a, Mpint *b);
void	mpdivmodfixfix(Mpint *q, Mpint *r, Mpint *n, Mpint *d);
void	mpnegfix(Mpint *a);
void	mpandfixfix(Mpint *a, Mpint *b);
void	mplshfixfix(Mpint *a, Mpint *b);
void	mporfixfix(Mpint *a, Mpint *b);
void	mprshfixfix(Mpint *a, Mpint *b);
void	mpxorfixfix(Mpint *a, Mpint *b);
void	mpcomfix(Mpint *a);
vlong	mpgetfix(Mpint *a);

/*
 *	mparith3.c
 */
void	mpmovefltflt(Mpflt *a, Mpflt *b);
void	mpmovecflt(Mpflt *a, double f);
int	mptestflt(Mpflt *a);
void	mpaddfltflt(Mpflt *a, Mpflt *b);
void	mpmulfltflt(Mpflt *a, Mpflt *b);
void	mpdivfltflt(Mpflt *a, Mpflt *b);
void	mpnegflt(Mpflt *a);
double	mpgetflt(Mpflt *a);

/*
 *	subr.c
 */
void	myexit(int);
void*	mal(int32);
void*	remal(void*, int32, int32);
void	errorexit(void);
uint32	stringhash(char*);
Sym*	lookup(char*);
Sym*	pkglookup(char*, char*);
void	yyerror(char*, ...);
void	warn(char*, ...);
void	fatal(char*, ...);
void	linehist(char*, int32);
int32	setlineno(Node*);
Node*	nod(int, Node*, Node*);
Node*	list(Node*, Node*);
Type*	typ(int);
Dcl*	dcl(void);
int	algtype(Type*);
Node*	rev(Node*);
Node*	unrev(Node*);
Node*	appendr(Node*, Node*);
void	dodump(Node*, int);
void	dump(char*, Node*);
Type*	aindex(Node*, Type*);
int	isnil(Node*);
int	isptrto(Type*, int);
int	isptrarray(Type*);
int	isptrdarray(Type*);
int	isinter(Type*);
int	ismethod(Type*);
Sym*	signame(Type*);
int	bytearraysz(Type*);
int	eqtype(Type*, Type*, int);
void	argtype(Node*, Type*);
int	eqargs(Type*, Type*);
uint32	typehash(Type*, int);
void	frame(int);
Node*	literal(int32);
Node*	dobad(void);
Node*	nodintconst(int32);
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
void	addmethod(Node*, Type*, int);
Node*	methodname(Node*, Type*);
Type*	functype(Node*, Node*, Node*);
char*	thistypenam(Node*);
void	funcnam(Type*, char*);
Node*	renameinit(Node*);
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
Node*	nametoanondcl(Node*);
Node*	nametodcl(Node*, Type*);
Node*	anondcl(Type*);

/*
 *	export.c
 */
void	renamepkg(Node*);
void	exportsym(Sym*);
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
void	doimport9(Sym*, Node*);

/*
 *	walk.c
 */
void	addtotop(Node*);
void	gettype(Node*, Node*);
void	walk(Node*);
void	walkstate(Node*);
void	walktype(Node*, int);
void	walkas(Node*);
void	walkbool(Node*);
Type*	walkswitch(Node*, Type*(*)(Node*, Type*));
int	casebody(Node*);
void	walkselect(Node*);
int	whatis(Node*);
void	walkdot(Node*);
Node*	ascompatee(int, Node**, Node**);
Node*	ascompatet(int, Node**, Type**, int);
Node*	ascompatte(int, Type**, Node**, int);
int	ascompat(Type*, Type*);
Node*	prcompat(Node*, int);
Node*	nodpanic(int32);
Node*	newcompat(Node*);
Node*	stringop(Node*, int);
Type*	fixmap(Type*);
Node*	mapop(Node*, int);
Type*	fixchan(Type*);
Node*	chanop(Node*, int);
Node*	arrayop(Node*, int);
Node*	ifaceop(Type*, Node*, int);
int	isandss(Type*, Node*);
Node*	convas(Node*);
void	arrayconv(Type*, Node*);
Node*	colas(Node*, Node*);
Node*	reorder1(Node*);
Node*	reorder2(Node*);
Node*	reorder3(Node*);
Node*	reorder4(Node*);
Node*	structlit(Node*);
Node*	arraylit(Node*);
Node*	maplit(Node*);
Node*	selectas(Node*, Node*);
Node*	old2new(Node*, Type*);

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
void	argspace(int32);
Node*	nodarg(Type*, int);
void	nodconst(Node*, Type*, vlong);
Type*	deep(Type*);
