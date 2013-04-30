// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	<bio.h>

#undef OAPPEND

// avoid <ctype.h>
#undef isblank
#define isblank goisblank

#ifndef	EXTERN
#define	EXTERN	extern
#endif

#undef	BUFSIZ

// The parser's maximum stack size.
// We have to use a #define macro here since yacc
// or bison will check for its definition and use
// a potentially smaller value if it is undefined.
#define YYMAXDEPTH 500

enum
{
	NHUNK		= 50000,
	BUFSIZ		= 8192,
	NSYMB		= 500,
	NHASH		= 1024,
	STRINGSZ	= 200,
	MAXALIGN	= 7,
	UINF		= 100,
	HISTSZ		= 10,

	PRIME1		= 3,

	AUNK		= 100,

	// These values are known by runtime.
	// The MEMx and NOEQx values must run in parallel.  See algtype.
	AMEM		= 0,
	AMEM0,
	AMEM8,
	AMEM16,
	AMEM32,
	AMEM64,
	AMEM128,
	ANOEQ,
	ANOEQ0,
	ANOEQ8,
	ANOEQ16,
	ANOEQ32,
	ANOEQ64,
	ANOEQ128,
	ASTRING,
	AINTER,
	ANILINTER,
	ASLICE,
	AFLOAT32,
	AFLOAT64,
	ACPLX64,
	ACPLX128,

	BADWIDTH	= -1000000000,
};

extern vlong	MAXWIDTH;

/*
 * note this is the representation
 * of the compilers string literals,
 * it is not the runtime representation
 */
typedef	struct	Strlit	Strlit;
struct	Strlit
{
	int32	len;
	char	s[3];	// variable
};

enum
{
	Mpscale	= 29,		// safely smaller than bits in a long
	Mpprec	= 16,		// Mpscale*Mpprec is max number of bits
	Mpnorm	= Mpprec - 1,	// significant words in a normalized float
	Mpbase	= 1L << Mpscale,
	Mpsign	= Mpbase >> 1,
	Mpmask	= Mpbase - 1,
	Mpdebug	= 0,
};

typedef	struct	Mpint	Mpint;
struct	Mpint
{
	long	a[Mpprec];
	uchar	neg;
	uchar	ovf;
};

typedef	struct	Mpflt	Mpflt;
struct	Mpflt
{
	Mpint	val;
	short	exp;
};

typedef	struct	Mpcplx	Mpcplx;
struct	Mpcplx
{
	Mpflt	real;
	Mpflt	imag;
};

typedef	struct	Val	Val;
struct	Val
{
	short	ctype;
	union
	{
		short	reg;		// OREGISTER
		short	bval;		// bool value CTBOOL
		Mpint*	xval;		// int CTINT, rune CTRUNE
		Mpflt*	fval;		// float CTFLT
		Mpcplx*	cval;		// float CTCPLX
		Strlit*	sval;		// string CTSTR
	} u;
};

typedef	struct	Pkg Pkg;
typedef	struct	Sym	Sym;
typedef	struct	Node	Node;
typedef	struct	NodeList	NodeList;
typedef	struct	Type	Type;
typedef	struct	Label	Label;

struct	Type
{
	uchar	etype;
	uchar	nointerface;
	uchar	chan;
	uchar	trecur;		// to detect loops
	uchar	printed;
	uchar	embedded;	// TFIELD embedded type
	uchar	siggen;
	uchar	funarg;		// on TSTRUCT and TFIELD
	uchar	copyany;
	uchar	local;		// created in this file
	uchar	deferwidth;
	uchar	broke;  	// broken type definition.
	uchar	isddd;		// TFIELD is ... argument
	uchar	align;

	Node*	nod;		// canonical OTYPE node
	Type*	orig;		// original type (type literal or predefined type)
	int		lineno;

	// TFUNC
	int	thistuple;
	int	outtuple;
	int	intuple;
	uchar	outnamed;

	Type*	method;
	Type*	xmethod;

	Sym*	sym;
	int32	vargen;		// unique name for OTYPE/ONAME

	Node*	nname;
	vlong	argwid;

	// most nodes
	Type*	type;   	// actual type for TFIELD, element type for TARRAY, TCHAN, TMAP, TPTRxx
	vlong	width;  	// offset in TFIELD, width in all others

	// TFIELD
	Type*	down;		// next struct field, also key type in TMAP
	Type*	outer;		// outer struct
	Strlit*	note;		// literal string annotation

	// TARRAY
	vlong	bound;		// negative is dynamic array

	int32	maplineno;	// first use of TFORW as map key
	int32	embedlineno;	// first use of TFORW as embedded type
	
	// for TFORW, where to copy the eventual value to
	NodeList	*copyto;
	
	// for usefield
	Node	*lastfn;
};
#define	T	((Type*)0)

typedef struct InitEntry InitEntry;
typedef struct InitPlan InitPlan;

struct InitEntry
{
	vlong xoffset;  // struct, array only
	Node *key;  // map only
	Node *expr;
};

struct InitPlan
{
	vlong lit;  // bytes of initialized non-zero literals
	vlong zero;  // bytes of zeros
	vlong expr;  // bytes of run-time computed expressions

	InitEntry *e;
	int len;
	int cap;
};

enum
{
	EscUnknown,
	EscHeap,
	EscScope,
	EscNone,
	EscReturn,
	EscNever,
	EscBits = 4,
	EscMask = (1<<EscBits) - 1,
};

struct	Node
{
	// Tree structure.
	// Generic recursive walks should follow these fields.
	Node*	left;
	Node*	right;
	Node*	ntest;
	Node*	nincr;
	NodeList*	ninit;
	NodeList*	nbody;
	NodeList*	nelse;
	NodeList*	list;
	NodeList*	rlist;

	uchar	op;
	uchar	nointerface;
	uchar	ullman;		// sethi/ullman number
	uchar	addable;	// type of addressability - 0 is not addressable
	uchar	trecur;		// to detect loops
	uchar	etype;		// op for OASOP, etype for OTYPE, exclam for export
	uchar	bounded;	// bounds check unnecessary
	uchar	class;		// PPARAM, PAUTO, PEXTERN, etc
	uchar	method;		// OCALLMETH name
	uchar	embedded;	// ODCLFIELD embedded type
	uchar	colas;		// OAS resulting from :=
	uchar	diag;		// already printed error about this
	uchar	noescape;	// func arguments do not escape
	uchar	builtin;	// built-in name, like len or close
	uchar	walkdef;
	uchar	typecheck;
	uchar	local;
	uchar	dodata;
	uchar	initorder;
	uchar	used;
	uchar	isddd;
	uchar	readonly;
	uchar	implicit;
	uchar	addrtaken;	// address taken, even if not moved to heap
	uchar	dupok;	// duplicate definitions ok (for func)
	schar	likely; // likeliness of if statement
	uchar	hasbreak;	// has break statement
	uint	esc;		// EscXXX
	int	funcdepth;

	// most nodes
	Type*	type;
	Node*	orig;		// original form, for printing, and tracking copies of ONAMEs

	// func
	Node*	nname;
	Node*	shortname;
	NodeList*	enter;
	NodeList*	exit;
	NodeList*	cvars;	// closure params
	NodeList*	dcl;	// autodcl for this func/closure
	NodeList*	inl;	// copy of the body for use in inlining

	// OLITERAL/OREGISTER
	Val	val;

	// ONAME
	Node*	ntype;
	Node*	defn;	// ONAME: initializing assignment; OLABEL: labeled statement
	Node*	pack;	// real package for import . names
	Node*	curfn;	// function for local variables
	Type*	paramfld; // TFIELD for this PPARAM; also for ODOT, curfn

	// ONAME func param with PHEAP
	Node*	heapaddr;	// temp holding heap address of param
	Node*	stackparam;	// OPARAM node referring to stack copy of param
	Node*	alloc;	// allocation call

	// ONAME closure param with PPARAMREF
	Node*	outer;	// outer PPARAMREF in nested closure
	Node*	closure;	// ONAME/PHEAP <-> ONAME/PPARAMREF

	// ONAME substitute while inlining
	Node* inlvar;

	// OPACK
	Pkg*	pkg;
	
	// OARRAYLIT, OMAPLIT, OSTRUCTLIT.
	InitPlan*	initplan;

	// Escape analysis.
	NodeList* escflowsrc;	// flow(this, src)
	NodeList* escretval;	// on OCALLxxx, list of dummy return values
	int	escloopdepth;	// -1: global, 0: return variables, 1:function top level, increased inside function for every loop or label to mark scopes

	Sym*	sym;		// various
	int32	vargen;		// unique name for OTYPE/ONAME
	int32	lineno;
	int32	endlineno;
	vlong	xoffset;
	vlong	stkdelta;	// offset added by stack frame compaction phase.
	int32	ostk;
	int32	iota;
	uint32	walkgen;
	int32	esclevel;
};
#define	N	((Node*)0)

/*
 * Every node has a walkgen field.
 * If you want to do a traversal of a node graph that
 * might contain duplicates and want to avoid
 * visiting the same nodes twice, increment walkgen
 * before starting.  Then before processing a node, do
 *
 *	if(n->walkgen == walkgen)
 *		return;
 *	n->walkgen = walkgen;
 *
 * Such a walk cannot call another such walk recursively,
 * because of the use of the global walkgen.
 */
EXTERN	uint32	walkgen;

struct	NodeList
{
	Node*	n;
	NodeList*	next;
	NodeList*	end;
};

enum
{
	SymExport	= 1<<0,	// to be exported
	SymPackage	= 1<<1,
	SymExported	= 1<<2,	// already written out by export
	SymUniq		= 1<<3,
	SymSiggen	= 1<<4,
	SymGcgen	= 1<<5,
};

struct	Sym
{
	ushort	lexical;
	uchar	flags;
	uchar	sym;		// huffman encoding in object file
	Sym*	link;
	int32	npkg;	// number of imported packages with this name
	uint32	uniqgen;

	// saved and restored by dcopy
	Pkg*	pkg;
	char*	name;		// variable name
	Node*	def;		// definition: ONAME OTYPE OPACK or OLITERAL
	Label*	label;	// corresponding label (ephemeral)
	int32	block;		// blocknumber to catch redeclaration
	int32	lastlineno;	// last declaration for diagnostic
	Pkg*	origpkg;	// original package for . import
};
#define	S	((Sym*)0)

EXTERN	Sym*	dclstack;

struct	Pkg
{
	char*	name;		// package name
	Strlit*	path;		// string literal used in import statement
	Sym*	pathsym;
	char*	prefix;		// escaped path for use in symbol table
	Pkg*	link;
	uchar	imported;	// export data of this package was parsed
	char	exported;	// import line written in export data
	char	direct;	// imported directly
	char	safe;	// whether the package is marked as safe
};

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

// Node ops.
enum
{
	OXXX,

	// names
	ONAME,	// var, const or func name
	ONONAME,	// unnamed arg or return value: f(int, string) (int, error) { etc }
	OTYPE,	// type name
	OPACK,	// import
	OLITERAL, // literal

	// expressions
	OADD,	// x + y
	OSUB,	// x - y
	OOR,	// x | y
	OXOR,	// x ^ y
	OADDSTR,	// s + "foo"
	OADDR,	// &x
	OANDAND,	// b0 && b1
	OAPPEND,	// append
	OARRAYBYTESTR,	// string(bytes)
	OARRAYRUNESTR,	// string(runes)
	OSTRARRAYBYTE,	// []byte(s)
	OSTRARRAYRUNE,	// []rune(s)
	OAS,	// x = y or x := y
	OAS2,	// x, y, z = xx, yy, zz
	OAS2FUNC,	// x, y = f()
	OAS2RECV,	// x, ok = <-c
	OAS2MAPR,	// x, ok = m["foo"]
	OAS2DOTTYPE,	// x, ok = I.(int)
	OASOP,	// x += y
	OCALL,	// function call, method call or type conversion, possibly preceded by defer or go.
	OCALLFUNC,	// f()
	OCALLMETH,	// t.Method()
	OCALLINTER,	// err.Error()
	OCALLPART,	// t.Method (without ())
	OCAP,	// cap
	OCLOSE,	// close
	OCLOSURE,	// f = func() { etc }
	OCMPIFACE,	// err1 == err2
	OCMPSTR,	// s1 == s2
	OCOMPLIT,	// composite literal, typechecking may convert to a more specific OXXXLIT.
	OMAPLIT,	// M{"foo":3, "bar":4}
	OSTRUCTLIT,	// T{x:3, y:4}
	OARRAYLIT,	// [2]int{3, 4}
	OPTRLIT,	// &T{x:3, y:4}
	OCONV,	// var i int; var u uint; i = int(u)
	OCONVIFACE,	// I(t)
	OCONVNOP,	// type Int int; var i int; var j Int; i = int(j)
	OCOPY,	// copy
	ODCL,	// var x int
	ODCLFUNC,	// func f() or func (r) f()
	ODCLFIELD,	// struct field, interface field, or func/method argument/return value.
	ODCLCONST,	// const pi = 3.14
	ODCLTYPE,	// type Int int
	ODELETE,	// delete
	ODOT,	// t.x
	ODOTPTR,	// p.x that is implicitly (*p).x
	ODOTMETH,	// t.Method
	ODOTINTER,	// err.Error
	OXDOT,	// t.x, typechecking may convert to a more specific ODOTXXX.
	ODOTTYPE,	// e = err.(MyErr)
	ODOTTYPE2,	// e, ok = err.(MyErr)
	OEQ,	// x == y
	ONE,	// x != y
	OLT,	// x < y
	OLE,	// x <= y
	OGE,	// x >= y
	OGT,	// x > y
	OIND,	// *p
	OINDEX,	// a[i]
	OINDEXMAP,	// m[s]
	OKEY,	// The x:3 in t{x:3, y:4}, the 1:2 in a[1:2], the 2:20 in [3]int{2:20}, etc.
	OPARAM,	// The on-stack copy of a parameter or return value that escapes.
	OLEN,	// len
	OMAKE,	// make, typechecking may convert to a more specfic OMAKEXXX.
	OMAKECHAN,	// make(chan int)
	OMAKEMAP,	// make(map[string]int)
	OMAKESLICE,	// make([]int, 0)
	OMUL,	// x * y
	ODIV,	// x / y
	OMOD,	// x % y
	OLSH,	// x << u
	ORSH,	// x >> u
	OAND,	// x & y
	OANDNOT,	// x &^ y
	ONEW,	// new
	ONOT,	// !b
	OCOM,	// ^x
	OPLUS,	// +x
	OMINUS,	// -y
	OOROR,	// b1 || b2
	OPANIC,	// panic
	OPRINT,	// print
	OPRINTN,	// println
	OPAREN,	// (x)
	OSEND,	// c <- x
	OSLICE,	// v[1:2], typechecking may convert to a more specfic OSLICEXXX.
	OSLICEARR,	// a[1:2]
	OSLICESTR,	// s[1:2]
	ORECOVER,	// recover
	ORECV,	// <-c
	ORUNESTR,	// string(i)
	OSELRECV,	// case x = <-c:
	OSELRECV2,	// case x, ok = <-c:
	OIOTA,	// iota
	OREAL,	// real
	OIMAG,	// imag
	OCOMPLEX,	// complex

	// statements
	OBLOCK,	// block of code
	OBREAK,	// break
	OCASE,	// case, after being verified by swt.c's casebody.
	OXCASE,	// case, before verification.
	OCONTINUE,	// continue
	ODEFER,	// defer
	OEMPTY,	// no-op
	OFALL,	// fallthrough, after being verified by swt.c's casebody.
	OXFALL,	// fallthrough, before verification.
	OFOR,	// for
	OGOTO,	// goto
	OIF,	// if
	OLABEL,	// label:
	OPROC,	// go
	ORANGE,	// range
	ORETURN,	// return
	OSELECT,	// select
	OSWITCH,	// switch x
	OTYPESW,	// switch err.(type)

	// types
	OTCHAN,	// chan int
	OTMAP,	// map[string]int
	OTSTRUCT,	// struct{}
	OTINTER,	// interface{}
	OTFUNC,	// func()
	OTARRAY,	// []int, [8]int, [N]int or [...]int
	OTPAREN,	// (T)

	// misc
	ODDD,	// func f(args ...int) or f(l...) or var a = [...]int{0, 1, 2}.
	ODDDARG,	// func f(args ...int), introduced by escape analysis.
	OINLCALL,	// intermediary representation of an inlined call.
	OEFACE,	// itable and data words of an empty-interface value.
	OITAB,	// itable word of an interface value.
	OCLOSUREVAR, // variable reference at beginning of closure function
	OCFUNC,	// reference to c function pointer (not go func value)
	OCHECKNOTNIL, // emit code to ensure pointer/interface not nil

	// arch-specific registers
	OREGISTER,	// a register, such as AX.
	OINDREG,	// offset plus indirect of a register, such as 8(SP).

	// 386/amd64-specific opcodes
	OCMP,	// compare: ACMP.
	ODEC,	// decrement: ADEC.
	OINC,	// increment: AINC.
	OEXTEND,	// extend: ACWD/ACDQ/ACQO.
	OHMUL, // high mul: AMUL/AIMUL for unsigned/signed (OMUL uses AIMUL for both).
	OLROT,	// left rotate: AROL.
	ORROTC, // right rotate-carry: ARCR.

	OEND,
};

enum
{
	Txxx,			// 0

	TINT8,	TUINT8,		// 1
	TINT16,	TUINT16,
	TINT32,	TUINT32,
	TINT64,	TUINT64,
	TINT, TUINT, TUINTPTR,

	TCOMPLEX64,		// 12
	TCOMPLEX128,

	TFLOAT32,		// 14
	TFLOAT64,

	TBOOL,			// 16

	TPTR32, TPTR64,		// 17

	TFUNC,			// 19
	TARRAY,
	T_old_DARRAY,
	TSTRUCT,		// 22
	TCHAN,
	TMAP,
	TINTER,			// 25
	TFORW,
	TFIELD,
	TANY,
	TSTRING,
	TUNSAFEPTR,

	// pseudo-types for literals
	TIDEAL,			// 31
	TNIL,
	TBLANK,

	// pseudo-type for frame layout
	TFUNCARGS,
	TCHANARGS,
	TINTERMETH,

	NTYPE,
};

enum
{
	CTxxx,

	CTINT,
	CTRUNE,
	CTFLT,
	CTCPLX,
	CTSTR,
	CTBOOL,
	CTNIL,
};

enum
{
	/* types of channel */
	/* must match ../../pkg/nreflect/type.go:/Chandir */
	Cxxx,
	Crecv = 1<<0,
	Csend = 1<<1,
	Cboth = Crecv | Csend,
};

// declaration context
enum
{
	Pxxx,

	PEXTERN,	// global variable
	PAUTO,		// local variables
	PPARAM,		// input arguments
	PPARAMOUT,	// output results
	PPARAMREF,	// closure variable reference
	PFUNC,		// global function

	PDISCARD,	// discard during parse of duplicate import

	PHEAP = 1<<7,	// an extra bit to identify an escaped variable
};

enum
{
	Etop = 1<<1,		// evaluated at statement level
	Erv = 1<<2,		// evaluated in value context
	Etype = 1<<3,
	Ecall = 1<<4,		// call-only expressions are ok
	Efnstruct = 1<<5,	// multivalue function returns are ok
	Eiota = 1<<6,		// iota is ok
	Easgn = 1<<7,		// assigning to expression
	Eindir = 1<<8,		// indirecting through expression
	Eaddr = 1<<9,		// taking address of expression
	Eproc = 1<<10,		// inside a go statement
	Ecomplit = 1<<11,	// type in composite literal
};

#define	BITS	5
#define	NVAR	(BITS*sizeof(uint32)*8)

typedef	struct	Bits	Bits;
struct	Bits
{
	uint32	b[BITS];
};

EXTERN	Bits	zbits;

typedef	struct	Var	Var;
struct	Var
{
	vlong	offset;
	Node*	node;
	int	width;
	char	name;
	char	etype;
	char	addr;
};

EXTERN	Var	var[NVAR];

typedef	struct	Typedef	Typedef;
struct	Typedef
{
	char*	name;
	int	etype;
	int	sameas;
};

extern	Typedef	typedefs[];

typedef	struct	Sig	Sig;
struct	Sig
{
	char*	name;
	Pkg*	pkg;
	Sym*	isym;
	Sym*	tsym;
	Type*	type;
	Type*	mtype;
	int32	offset;
	Sig*	link;
};

typedef	struct	Io	Io;
struct	Io
{
	char*	infile;
	Biobuf*	bin;
	int32	ilineno;
	int	nlsemi;
	int	eofnl;
	int	peekc;
	int	peekc1;	// second peekc for ...
	char*	cp;	// used for content when bin==nil
	int	importsafe;
};

typedef	struct	Dlist	Dlist;
struct	Dlist
{
	Type*	field;
};

typedef	struct	Idir	Idir;
struct Idir
{
	Idir*	link;
	char*	dir;
};

/*
 * argument passing to/from
 * smagic and umagic
 */
typedef	struct	Magic Magic;
struct	Magic
{
	int	w;	// input for both - width
	int	s;	// output for both - shift
	int	bad;	// output for both - unexpected failure

	// magic multiplier for signed literal divisors
	int64	sd;	// input - literal divisor
	int64	sm;	// output - multiplier

	// magic multiplier for unsigned literal divisors
	uint64	ud;	// input - literal divisor
	uint64	um;	// output - multiplier
	int	ua;	// output - adder
};

typedef struct	Prog Prog;
#pragma incomplete Prog

struct	Label
{
	uchar	used;
	Sym*	sym;
	Node*	def;
	NodeList*	use;
	Label*	link;
	
	// for use during gen
	Prog*	gotopc;	// pointer to unresolved gotos
	Prog*	labelpc;	// pointer to code
	Prog*	breakpc;	// pointer to code
	Prog*	continpc;	// pointer to code
};
#define	L	((Label*)0)

/*
 * note this is the runtime representation
 * of the compilers arrays.
 *
 * typedef	struct
 * {				// must not move anything
 *	uchar	array[8];	// pointer to data
 *	uchar	nel[4];		// number of elements
 *	uchar	cap[4];		// allocated number of elements
 * } Array;
 */
EXTERN	int	Array_array;	// runtime offsetof(Array,array) - same for String
EXTERN	int	Array_nel;	// runtime offsetof(Array,nel) - same for String
EXTERN	int	Array_cap;	// runtime offsetof(Array,cap)
EXTERN	int	sizeof_Array;	// runtime sizeof(Array)


/*
 * note this is the runtime representation
 * of the compilers strings.
 *
 * typedef	struct
 * {				// must not move anything
 *	uchar	array[8];	// pointer to data
 *	uchar	nel[4];		// number of elements
 * } String;
 */
EXTERN	int	sizeof_String;	// runtime sizeof(String)

EXTERN	Dlist	dotlist[10];	// size is max depth of embeddeds

EXTERN	Io	curio;
EXTERN	Io	pushedio;
EXTERN	int32	lexlineno;
EXTERN	int32	lineno;
EXTERN	int32	prevlineno;
EXTERN	char*	pathname;
EXTERN	Hist*	hist;
EXTERN	Hist*	ehist;

EXTERN	char*	infile;
EXTERN	char*	outfile;
EXTERN	Biobuf*	bout;
EXTERN	int	nerrors;
EXTERN	int	nsavederrors;
EXTERN	int	nsyntaxerrors;
EXTERN	int	safemode;
EXTERN	char	namebuf[NSYMB];
EXTERN	char	lexbuf[NSYMB];
EXTERN	char	litbuf[NSYMB];
EXTERN	int	debug[256];
EXTERN	Sym*	hash[NHASH];
EXTERN	Sym*	importmyname;	// my name for package
EXTERN	Pkg*	localpkg;	// package being compiled
EXTERN	Pkg*	importpkg;	// package being imported
EXTERN	Pkg*	structpkg;	// package that declared struct, during import
EXTERN	Pkg*	builtinpkg;	// fake package for builtins
EXTERN	Pkg*	gostringpkg;	// fake pkg for Go strings
EXTERN	Pkg*	itabpkg;	// fake pkg for itab cache
EXTERN	Pkg*	runtimepkg;	// package runtime
EXTERN	Pkg*	racepkg;	// package runtime/race
EXTERN	Pkg*	stringpkg;	// fake package for C strings
EXTERN	Pkg*	typepkg;	// fake package for runtime type info (headers)
EXTERN	Pkg*	typelinkpkg;	// fake package for runtime type info (data)
EXTERN	Pkg*	weaktypepkg;	// weak references to runtime type info
EXTERN	Pkg*	unsafepkg;	// package unsafe
EXTERN	Pkg*	trackpkg;	// fake package for field tracking
EXTERN	Pkg*	phash[128];
EXTERN	int	tptr;		// either TPTR32 or TPTR64
extern	char*	runtimeimport;
extern	char*	unsafeimport;
EXTERN	char*	myimportpath;
EXTERN	Idir*	idirs;
EXTERN	char*	localimport;

EXTERN	Type*	types[NTYPE];
EXTERN	Type*	idealstring;
EXTERN	Type*	idealbool;
EXTERN	Type*	bytetype;
EXTERN	Type*	runetype;
EXTERN	Type*	errortype;
EXTERN	uchar	simtype[NTYPE];
EXTERN	uchar	isptr[NTYPE];
EXTERN	uchar	isforw[NTYPE];
EXTERN	uchar	isint[NTYPE];
EXTERN	uchar	isfloat[NTYPE];
EXTERN	uchar	iscomplex[NTYPE];
EXTERN	uchar	issigned[NTYPE];
EXTERN	uchar	issimple[NTYPE];

EXTERN	uchar	okforeq[NTYPE];
EXTERN	uchar	okforadd[NTYPE];
EXTERN	uchar	okforand[NTYPE];
EXTERN	uchar	okfornone[NTYPE];
EXTERN	uchar	okforcmp[NTYPE];
EXTERN	uchar	okforbool[NTYPE];
EXTERN	uchar	okforcap[NTYPE];
EXTERN	uchar	okforlen[NTYPE];
EXTERN	uchar	okforarith[NTYPE];
EXTERN	uchar	okforconst[NTYPE];
EXTERN	uchar*	okfor[OEND];
EXTERN	uchar	iscmp[OEND];

EXTERN	Mpint*	minintval[NTYPE];
EXTERN	Mpint*	maxintval[NTYPE];
EXTERN	Mpflt*	minfltval[NTYPE];
EXTERN	Mpflt*	maxfltval[NTYPE];

EXTERN	NodeList*	xtop;
EXTERN	NodeList*	externdcl;
EXTERN	NodeList*	closures;
EXTERN	NodeList*	exportlist;
EXTERN	NodeList*	importlist;	// imported functions and methods with inlinable bodies
EXTERN	NodeList*	funcsyms;
EXTERN	int	dclcontext;		// PEXTERN/PAUTO
EXTERN	int	incannedimport;
EXTERN	int	statuniqgen;		// name generator for static temps
EXTERN	int	loophack;

EXTERN	int32	iota;
EXTERN	NodeList*	lastconst;
EXTERN	Node*	lasttype;
EXTERN	vlong	maxarg;
EXTERN	vlong	stksize;		// stack size for current frame
EXTERN	int32	blockgen;		// max block number
EXTERN	int32	block;			// current block number
EXTERN	int	hasdefer;		// flag that curfn has defer statetment

EXTERN	Node*	curfn;

EXTERN	int	widthptr;
EXTERN	int	widthint;

EXTERN	Node*	typesw;
EXTERN	Node*	nblank;

extern	int	thechar;
extern	char*	thestring;
EXTERN	int  	use_sse;

EXTERN	char*	hunk;
EXTERN	int32	nhunk;
EXTERN	int32	thunk;

EXTERN	int	funcdepth;
EXTERN	int	typecheckok;
EXTERN	int	compiling_runtime;
EXTERN	int	compiling_wrappers;
EXTERN	int	pure_go;
EXTERN	int	flag_race;
EXTERN	int	flag_largemodel;
EXTERN	int	noescape;

EXTERN	int	nointerface;
EXTERN	int	fieldtrack_enabled;

/*
 *	y.tab.c
 */
int	yyparse(void);

/*
 *	align.c
 */
int	argsize(Type *t);
void	checkwidth(Type *t);
void	defercheckwidth(void);
void	dowidth(Type *t);
void	resumecheckwidth(void);
vlong	rnd(vlong o, vlong r);
void	typeinit(void);

/*
 *	bits.c
 */
int	Qconv(Fmt *fp);
Bits	band(Bits a, Bits b);
int	bany(Bits *a);
int	beq(Bits a, Bits b);
int	bitno(int32 b);
Bits	blsh(uint n);
Bits	bnot(Bits a);
int	bnum(Bits a);
Bits	bor(Bits a, Bits b);
int	bset(Bits a, uint n);

/*
 *	closure.c
 */
Node*	closurebody(NodeList *body);
void	closurehdr(Node *ntype);
void	typecheckclosure(Node *func, int top);
Node*	walkclosure(Node *func, NodeList **init);
void	typecheckpartialcall(Node*, Node*);
Node*	walkpartialcall(Node*, NodeList**);

/*
 *	const.c
 */
int	cmpslit(Node *l, Node *r);
int	consttype(Node *n);
void	convconst(Node *con, Type *t, Val *val);
void	convlit(Node **np, Type *t);
void	convlit1(Node **np, Type *t, int explicit);
void	defaultlit(Node **np, Type *t);
void	defaultlit2(Node **lp, Node **rp, int force);
void	evconst(Node *n);
int	isconst(Node *n, int ct);
int	isgoconst(Node *n);
Node*	nodcplxlit(Val r, Val i);
Node*	nodlit(Val v);
long	nonnegconst(Node *n);
int	doesoverflow(Val v, Type *t);
void	overflow(Val v, Type *t);
int	smallintconst(Node *n);
Val	toint(Val v);
Mpflt*	truncfltlit(Mpflt *oldv, Type *t);

/*
 *	cplx.c
 */
void	complexadd(int op, Node *nl, Node *nr, Node *res);
void	complexbool(int op, Node *nl, Node *nr, int true, int likely, Prog *to);
void	complexgen(Node *n, Node *res);
void	complexminus(Node *nl, Node *res);
void	complexmove(Node *f, Node *t);
void	complexmul(Node *nl, Node *nr, Node *res);
int	complexop(Node *n, Node *res);
void	nodfconst(Node *n, Type *t, Mpflt* fval);

/*
 *	dcl.c
 */
void	addmethod(Sym *sf, Type *t, int local, int nointerface);
void	addvar(Node *n, Type *t, int ctxt);
NodeList*	checkarglist(NodeList *all, int input);
Node*	colas(NodeList *left, NodeList *right, int32 lno);
void	colasdefn(NodeList *left, Node *defn);
NodeList*	constiter(NodeList *vl, Node *t, NodeList *cl);
Node*	dclname(Sym *s);
void	declare(Node *n, int ctxt);
void	dumpdcl(char *st);
Node*	embedded(Sym *s);
Node*	fakethis(void);
void	funcbody(Node *n);
void	funccompile(Node *n, int isclosure);
void	funchdr(Node *n);
Type*	functype(Node *this, NodeList *in, NodeList *out);
void	ifacedcl(Node *n);
int	isifacemethod(Type *f);
void	markdcl(void);
Node*	methodname(Node *n, Type *t);
Node*	methodname1(Node *n, Node *t);
Sym*	methodsym(Sym *nsym, Type *t0, int iface);
Node*	newname(Sym *s);
Node*	oldname(Sym *s);
void	popdcl(void);
void	poptodcl(void);
void	redeclare(Sym *s, char *where);
void	testdclstack(void);
Type*	tointerface(NodeList *l);
Type*	tostruct(NodeList *l);
Node*	typedcl0(Sym *s);
Node*	typedcl1(Node *n, Node *t, int local);
Node*	typenod(Type *t);
NodeList*	variter(NodeList *vl, Node *t, NodeList *el);
Sym*	funcsym(Sym*);

/*
 *	esc.c
 */
void	escapes(NodeList*);

/*
 *	export.c
 */
void	autoexport(Node *n, int ctxt);
void	dumpexport(void);
int	exportname(char *s);
void	exportsym(Node *n);
void    importconst(Sym *s, Type *t, Node *n);
void	importimport(Sym *s, Strlit *z);
Sym*    importsym(Sym *s, int op);
void    importtype(Type *pt, Type *t);
void    importvar(Sym *s, Type *t);
Type*	pkgtype(Sym *s);

/*
 *	fmt.c
 */
void	fmtinstallgo(void);
void	dump(char *s, Node *n);
void	dumplist(char *s, NodeList *l);

/*
 *	gen.c
 */
void	addrescapes(Node *n);
void	cgen_as(Node *nl, Node *nr);
void	cgen_callmeth(Node *n, int proc);
void	cgen_eface(Node* n, Node* res);
void	cgen_slice(Node* n, Node* res);
void	clearlabels(void);
void	checklabels(void);
int	dotoffset(Node *n, int64 *oary, Node **nn);
void	gen(Node *n);
void	genlist(NodeList *l);
Node*	sysfunc(char *name);
void	tempname(Node *n, Type *t);
Node*	temp(Type*);

/*
 *	init.c
 */
void	fninit(NodeList *n);
Sym*	renameinit(void);

/*
 *	inl.c
 */
void	caninl(Node *fn);
void	inlcalls(Node *fn);
void	typecheckinl(Node *fn);

/*
 *	lex.c
 */
void	cannedimports(char *file, char *cp);
void	importfile(Val *f, int line);
char*	lexname(int lex);
char*	expstring(void);
void	mkpackage(char* pkgname);
void	unimportfile(void);
int32	yylex(void);
extern	int	windows;
extern	int	yylast;
extern	int	yyprev;

/*
 *	mparith1.c
 */
int	Bconv(Fmt *fp);
int	Fconv(Fmt *fp);
void	mpaddcfix(Mpint *a, vlong c);
void	mpaddcflt(Mpflt *a, double c);
void	mpatofix(Mpint *a, char *as);
void	mpatoflt(Mpflt *a, char *as);
int	mpcmpfixc(Mpint *b, vlong c);
int	mpcmpfixfix(Mpint *a, Mpint *b);
int	mpcmpfixflt(Mpint *a, Mpflt *b);
int	mpcmpfltc(Mpflt *b, double c);
int	mpcmpfltfix(Mpflt *a, Mpint *b);
int	mpcmpfltflt(Mpflt *a, Mpflt *b);
void	mpcomfix(Mpint *a);
void	mpdivfixfix(Mpint *a, Mpint *b);
void	mpmodfixfix(Mpint *a, Mpint *b);
void	mpmovefixfix(Mpint *a, Mpint *b);
void	mpmovefixflt(Mpflt *a, Mpint *b);
int	mpmovefltfix(Mpint *a, Mpflt *b);
void	mpmovefltflt(Mpflt *a, Mpflt *b);
void	mpmulcfix(Mpint *a, vlong c);
void	mpmulcflt(Mpflt *a, double c);
void	mpsubfixfix(Mpint *a, Mpint *b);
void	mpsubfltflt(Mpflt *a, Mpflt *b);

/*
 *	mparith2.c
 */
void	mpaddfixfix(Mpint *a, Mpint *b, int);
void	mpandfixfix(Mpint *a, Mpint *b);
void	mpandnotfixfix(Mpint *a, Mpint *b);
void	mpdivfract(Mpint *a, Mpint *b);
void	mpdivmodfixfix(Mpint *q, Mpint *r, Mpint *n, Mpint *d);
vlong	mpgetfix(Mpint *a);
void	mplshfixfix(Mpint *a, Mpint *b);
void	mpmovecfix(Mpint *a, vlong c);
void	mpmulfixfix(Mpint *a, Mpint *b);
void	mpmulfract(Mpint *a, Mpint *b);
void	mpnegfix(Mpint *a);
void	mporfixfix(Mpint *a, Mpint *b);
void	mprshfixfix(Mpint *a, Mpint *b);
void	mpshiftfix(Mpint *a, int s);
int	mptestfix(Mpint *a);
void	mpxorfixfix(Mpint *a, Mpint *b);

/*
 *	mparith3.c
 */
void	mpaddfltflt(Mpflt *a, Mpflt *b);
void	mpdivfltflt(Mpflt *a, Mpflt *b);
double	mpgetflt(Mpflt *a);
void	mpmovecflt(Mpflt *a, double c);
void	mpmulfltflt(Mpflt *a, Mpflt *b);
void	mpnegflt(Mpflt *a);
void	mpnorm(Mpflt *a);
int	mptestflt(Mpflt *a);
int	sigfig(Mpflt *a);

/*
 *	obj.c
 */
void	Bputname(Biobuf *b, Sym *s);
int	duint16(Sym *s, int off, uint16 v);
int	duint32(Sym *s, int off, uint32 v);
int	duint64(Sym *s, int off, uint64 v);
int	duint8(Sym *s, int off, uint8 v);
int	duintptr(Sym *s, int off, uint64 v);
int	dsname(Sym *s, int off, char *dat, int ndat);
void	dumpobj(void);
void	ieeedtod(uint64 *ieee, double native);
Sym*	stringsym(char*, int);

/*
 *	order.c
 */
void	order(Node *fn);

/*
 *	range.c
 */
void	typecheckrange(Node *n);
void	walkrange(Node *n);

/*
 *	reflect.c
 */
void	dumptypestructs(void);
Type*	methodfunc(Type *f, Type*);
Node*	typename(Type *t);
Sym*	typesym(Type *t);
Sym*	typenamesym(Type *t);
Sym*	tracksym(Type *t);
Sym*	typesymprefix(char *prefix, Type *t);
int	haspointers(Type *t);
void	usefield(Node*);

/*
 *	select.c
 */
void	typecheckselect(Node *sel);
void	walkselect(Node *sel);

/*
 *	sinit.c
 */
void	anylit(int, Node *n, Node *var, NodeList **init);
int	gen_as_init(Node *n);
NodeList*	initfix(NodeList *l);
int	oaslit(Node *n, NodeList **init);
int	stataddr(Node *nam, Node *n);

/*
 *	subr.c
 */
Node*	adddot(Node *n);
int	adddot1(Sym *s, Type *t, int d, Type **save, int ignorecase);
void	addinit(Node**, NodeList*);
Type*	aindex(Node *b, Type *t);
int	algtype(Type *t);
int	algtype1(Type *t, Type **bad);
void	argtype(Node *on, Type *t);
Node*	assignconv(Node *n, Type *t, char *context);
int	assignop(Type *src, Type *dst, char **why);
void	badtype(int o, Type *tl, Type *tr);
int	brcom(int a);
int	brrev(int a);
NodeList*	concat(NodeList *a, NodeList *b);
int	convertop(Type *src, Type *dst, char **why);
Node*	copyexpr(Node*, Type*, NodeList**);
int	count(NodeList *l);
int	cplxsubtype(int et);
int	eqtype(Type *t1, Type *t2);
int	eqtypenoname(Type *t1, Type *t2);
void	errorexit(void);
void	expandmeth(Type *t);
void	fatal(char *fmt, ...);
void	flusherrors(void);
void	frame(int context);
Type*	funcfirst(Iter *s, Type *t);
Type*	funcnext(Iter *s);
void	genwrapper(Type *rcvr, Type *method, Sym *newnam, int iface);
void	genhash(Sym *sym, Type *t);
void	geneq(Sym *sym, Type *t);
Type**	getinarg(Type *t);
Type*	getinargx(Type *t);
Type**	getoutarg(Type *t);
Type*	getoutargx(Type *t);
Type**	getthis(Type *t);
Type*	getthisx(Type *t);
int	implements(Type *t, Type *iface, Type **missing, Type **have, int *ptr);
void	importdot(Pkg *opkg, Node *pack);
int	is64(Type *t);
int	isbadimport(Strlit *s);
int	isblank(Node *n);
int	isblanksym(Sym *s);
int	isfixedarray(Type *t);
int	isideal(Type *t);
int	isinter(Type *t);
int	isnil(Node *n);
int	isnilinter(Type *t);
int	isptrto(Type *t, int et);
int	isslice(Type *t);
int	istype(Type *t, int et);
void	linehist(char *file, int32 off, int relative);
NodeList*	list(NodeList *l, Node *n);
NodeList*	list1(Node *n);
void	listsort(NodeList**, int(*f)(Node*, Node*));
Node*	liststmt(NodeList *l);
NodeList*	listtreecopy(NodeList *l);
Sym*	lookup(char *name);
void*	mal(int32 n);
Type*	maptype(Type *key, Type *val);
Type*	methtype(Type *t, int mustname);
Pkg*	mkpkg(Strlit *path);
Sym*	ngotype(Node *n);
int	noconv(Type *t1, Type *t2);
Node*	nod(int op, Node *nleft, Node *nright);
Node*	nodbool(int b);
void	nodconst(Node *n, Type *t, int64 v);
Node*	nodintconst(int64 v);
Node*	nodfltconst(Mpflt *v);
Node*	nodnil(void);
int	parserline(void);
Sym*	pkglookup(char *name, Pkg *pkg);
int	powtwo(Node *n);
Type*	ptrto(Type *t);
void*	remal(void *p, int32 on, int32 n);
Sym*	restrictlookup(char *name, Pkg *pkg);
Node*	safeexpr(Node *n, NodeList **init);
void	saveerrors(void);
Node*	cheapexpr(Node *n, NodeList **init);
Node*	localexpr(Node *n, Type *t, NodeList **init);
void	saveorignode(Node *n);
int32	setlineno(Node *n);
void	setmaxarg(Type *t);
Type*	shallow(Type *t);
int	simsimtype(Type *t);
void	smagic(Magic *m);
Type*	sortinter(Type *t);
uint32	stringhash(char *p);
Strlit*	strlit(char *s);
int	structcount(Type *t);
Type*	structfirst(Iter *s, Type **nn);
Type*	structnext(Iter *s);
Node*	syslook(char *name, int copy);
Type*	tounsigned(Type *t);
Node*	treecopy(Node *n);
Type*	typ(int et);
uint32	typehash(Type *t);
void	ullmancalc(Node *n);
void	umagic(Magic *m);
void	warn(char *fmt, ...);
void	warnl(int line, char *fmt, ...);
void	yyerror(char *fmt, ...);
void	yyerrorl(int line, char *fmt, ...);

/*
 *	swt.c
 */
void	typecheckswitch(Node *n);
void	walkswitch(Node *sw);

/*
 *	typecheck.c
 */
int	islvalue(Node *n);
Node*	typecheck(Node **np, int top);
void	typechecklist(NodeList *l, int top);
Node*	typecheckdef(Node *n);
void	copytype(Node *n, Type *t);
void	checkreturn(Node*);
void	queuemethod(Node *n);

/*
 *	unsafe.c
 */
int	isunsafebuiltin(Node *n);
Node*	unsafenmagic(Node *n);

/*
 *	walk.c
 */
Node*	callnew(Type *t);
Node*	chanfn(char *name, int n, Type *t);
Node*	mkcall(char *name, Type *t, NodeList **init, ...);
Node*	mkcall1(Node *fn, Type *t, NodeList **init, ...);
int	vmatch1(Node *l, Node *r);
void	walk(Node *fn);
void	walkexpr(Node **np, NodeList **init);
void	walkexprlist(NodeList *l, NodeList **init);
void	walkexprlistsafe(NodeList *l, NodeList **init);
void	walkstmt(Node **np);
void	walkstmtlist(NodeList *l);
Node*	conv(Node*, Type*);
int	candiscard(Node*);

/*
 *	arch-specific ggen.c/gsubr.c/gobj.c/pgen.c
 */
#define	P	((Prog*)0)

typedef	struct	Plist	Plist;
struct	Plist
{
	Node*	name;
	Prog*	firstpc;
	int	recur;
	Plist*	link;
};

EXTERN	Plist*	plist;
EXTERN	Plist*	plast;

EXTERN	Prog*	continpc;
EXTERN	Prog*	breakpc;
EXTERN	Prog*	pc;
EXTERN	Prog*	firstpc;
EXTERN	Prog*	retpc;

EXTERN	Node*	nodfp;

int	anyregalloc(void);
void	betypeinit(void);
void	bgen(Node *n, int true, int likely, Prog *to);
void	checkref(Node *n, int force);
void	checknotnil(Node*, NodeList**);
void	cgen(Node*, Node*);
void	cgen_asop(Node *n);
void	cgen_call(Node *n, int proc);
void	cgen_callinter(Node *n, Node *res, int proc);
void	cgen_ret(Node *n);
void	clearfat(Node *n);
void	compile(Node*);
void	defframe(Prog*);
int	dgostringptr(Sym*, int off, char *str);
int	dgostrlitptr(Sym*, int off, Strlit*);
int	dstringptr(Sym *s, int off, char *str);
int	dsymptr(Sym *s, int off, Sym *x, int xoff);
int	duintxx(Sym *s, int off, uint64 v, int wid);
void	dumpdata(void);
void	dumpfuncs(void);
void	fixautoused(Prog*);
void	gdata(Node*, Node*, int);
void	gdatacomplex(Node*, Mpcplx*);
void	gdatastring(Node*, Strlit*);
void	genembedtramp(Type*, Type*, Sym*, int iface);
void	ggloblnod(Node *nam);
void	ggloblsym(Sym *s, int32 width, int dupok, int rodata);
Prog*	gjmp(Prog*);
void	gused(Node*);
int	isfat(Type*);
void	markautoused(Prog*);
Plist*	newplist(void);
Node*	nodarg(Type*, int);
void	nopout(Prog*);
void	patch(Prog*, Prog*);
Prog*	unpatch(Prog*);
void	zfile(Biobuf *b, char *p, int n);
void	zhist(Biobuf *b, int line, vlong offset);
void	zname(Biobuf *b, Sym *s, int t);

#pragma	varargck	type	"A"	int
#pragma	varargck	type	"B"	Mpint*
#pragma	varargck	type	"D"	Addr*
#pragma	varargck	type	"lD"	Addr*
#pragma	varargck	type	"E"	int
#pragma	varargck	type	"E"	uint
#pragma	varargck	type	"F"	Mpflt*
#pragma	varargck	type	"H"	NodeList*
#pragma	varargck	type	"J"	Node*
#pragma	varargck	type	"lL"	int
#pragma	varargck	type	"lL"	uint
#pragma	varargck	type	"L"	int
#pragma	varargck	type	"L"	uint
#pragma	varargck	type	"N"	Node*
#pragma	varargck	type	"lN"	Node*
#pragma	varargck	type	"O"	uint
#pragma	varargck	type	"P"	Prog*
#pragma	varargck	type	"Q"	Bits
#pragma	varargck	type	"R"	int
#pragma	varargck	type	"S"	Sym*
#pragma	varargck	type	"lS"	Sym*
#pragma	varargck	type	"T"	Type*
#pragma	varargck	type	"lT"	Type*
#pragma	varargck	type	"V"	Val*
#pragma	varargck	type	"Y"	char*
#pragma	varargck	type	"Z"	Strlit*

/*
 *	racewalk.c
 */
void	racewalk(Node *fn);
