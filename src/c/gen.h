// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef	EXTERN
#define EXTERN	extern
#endif

typedef	struct	Prog	Prog;
typedef	struct	Addr	Addr;

struct	Addr
{
	int	type;
	Node*	node;
	Prog*	branch;
};

enum
{
	AXXX		= 0,
	ANONE,
	ANODE,
	ABRANCH,
};

struct	Prog
{
	int	op;			// opcode
	int	pt;
	int	pt1;
	int	param;
	long	lineno;			// source line
	long	loc;			// program counter for print
	int	mark;
	Addr	addr;			// operand
	Prog*	link;
};
#define	P	((Prog*)0)

typedef	struct	Plist	Plist;
struct	Plist
{
	Node*	name;
	Dcl*	locals;
	Prog*	firstpc;
	int	recur;
	Plist*	link;
};

typedef	struct	Sig	Sig;
struct Sig
{
	char*	fun;
	ulong	hash;
	int	offset;
	Sig*	link;
};

enum
{
	PTxxx,

	PTINT8		= TINT8,
	PTUINT8		= TUINT8,
	PTINT16		= TINT16,
	PTUINT16	= TUINT16,
	PTINT32		= TINT32,
	PTUINT32	= TUINT32,
	PTINT64		= TINT64,
	PTUINT64	= TUINT64,
	PTFLOAT32	= TFLOAT32,
	PTFLOAT64	= TFLOAT64,
	PTFLOAT80	= TFLOAT80,
	PTBOOL		= TBOOL,
	PTPTR		= TPTR,
	PTSTRUCT	= TSTRUCT,
	PTINTER		= TINTER,
	PTARRAY		= TARRAY,
	PTSTRING	= TSTRING,
	PTCHAN		= TCHAN,
	PTMAP		= TMAP,

	PTNIL		= NTYPE,
	PTADDR,
	PTERROR,

	NPTYPE,
};

enum
{
	PXXX		= 0,

	PERROR, PPANIC, PPRINT, PGOTO, PGOTOX,

	PCMP, PTEST, PNEW, PLEN,
	PCALL1, PCALL2, PCALLI2, PCALLM2, PCALLF2, PCALL3, PRETURN,

	PBEQ, PBNE,
	PBLT, PBLE, PBGE, PBGT,
	PBTRUE, PBFALSE,

	PLOAD, PLOADI,
	PSTORE, PSTOREI,
	PSTOREZ, PSTOREZI,
	PCONV, PADDR, PADDO, PINDEX, PINDEXZ,
	PSLICE,

	PADD, PSUB, PMUL, PDIV, PLSH, PRSH, PMOD,
	PAND, POR, PXOR, PCAT,

	PMINUS, PCOM,

	PEND,
};

typedef	struct	Case Case;
struct	Case
{
	Prog*	sprog;
	Node*	scase;
	Case*	slink;
};
#define	C	((Case*)0)

EXTERN	Prog*	continpc;
EXTERN	Prog*	breakpc;
EXTERN	Prog*	pc;
EXTERN	Prog*	firstpc;
EXTERN	Plist*	plist;
EXTERN	Plist*	plast;
EXTERN	Biobuf*	bout;
EXTERN	long	dynloc;

/*
 * gen.c
 */
void	compile(Node*);
void	proglist(void);
void	gen(Node*);
void	cgen(Node*);
void	agen(Node*);
void	bgen(Node*, int, Prog*);
void	swgen(Node*);
Node*	lookdot(Node*, Node*, int);
int	usesptr(Node*);
void	inarggen(void);
void	cgen_as(Node*, Node*, int, int);
void	cgen_asop(Node*, Node*, int);
void	cgen_ret(Node*);
void	cgen_call(Node*, int);
void	cgen_callret(Node*, Node*);
void	genprint(Node*);
int	needconvert(Node*, Node*);
void	genconv(Node*, Node*);
void	genindex(Node*);

/*
 * gsubr.c
 */
int	Aconv(Fmt*);
int	Pconv(Fmt*);
void	proglist(void);
Prog*	gbranch(int, Node*);
void	patch(Prog*, Prog*);
Prog*	prog(int);
Node*	tempname(Node*);
Prog*	gopcode(int, int, Node*);
Prog*	gopcodet(int, Node*, Node*);
void	gaddoffset(Node*);
void	gconv(int, int);
int	conv2pt(Node*);
void	belexinit(int);
vlong	convvtox(vlong, int);
int	brcom(int);
int	brrev(int);
void	fnparam(Node*, int, int);
Sig*	lsort(Sig*, int(*)(Sig*, Sig*));

/*
 * obj.c
 */
void	dumpobj(void);
void	litrl(Prog*);
void	obj(Prog*);
void	follow(Prog*);
Prog*	gotochain(Prog*);
int	Xconv(Fmt*);
int	Rconv(Fmt*);
int	Qconv(Fmt*);
int	Dconv(Fmt*);
int	Cconv(Fmt*);
void	dumpexterns(void);
void	dumpfunct(Plist*);
void	dumpsignatures(void);
void	doframe(Dcl*, char*);
void	docall1(Prog*);
void	docall2(Prog*);
void	docalli2(Prog*);
void	docallm2(Prog*);
void	docallf2(Prog*);
void	docall3(Prog*);
void	doconv(Prog*);
char*	getfmt(int);
void	dumpmethods(void);
