// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


#include <u.h>
#include <libc.h>

#include "../gc/go.h"
#include "../6l/6.out.h"

#ifndef	EXTERN
#define EXTERN	extern
#endif

typedef	struct	Prog	Prog;
typedef	struct	Addr	Addr;

struct	Addr
{
	vlong	offset;
	double	dval;
	Prog*	branch;
	char	sval[NSNAME];

	Sym*	sym;
	uchar	type;
	uchar	index;
	uchar	etype;
	uchar	scale;	/* doubles as width in DATA op */
};
#define	A	((Addr*)0)

struct	Prog
{
	short	as;		// opcode
	uint32	loc;		// pc offset in this func
	uint32	lineno;		// source line that generated this
	Addr	from;		// src address
	Addr	to;		// dst address
	Prog*	link;		// next instruction in this func
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
	char*	name;
	Sym*	sym;
	uint32	hash;
	int32	perm;
	int32	offset;
	Sig*	link;
};

typedef	struct	Case Case;
struct	Case
{
	Prog*	sprog;
	Node*	scase;
	Case*	slink;
};
#define	C	((Case*)0)

typedef	struct	Pool Pool;
struct	Pool
{
	String*	sval;
	Pool*	link;
};

typedef	struct	Label Label;
struct	Label
{
	uchar	op;		// OFOR/OGOTO/OLABEL
	Sym*	sym;
	Prog*	label;		// pointer to code
	Prog*	breakpc;	// pointer to code
	Prog*	continpc;	// pointer to code
	Label*	link;
};
#define	L	((Label*)0)

EXTERN	Prog*	continpc;
EXTERN	Prog*	breakpc;
EXTERN	Prog*	pc;
EXTERN	Prog*	firstpc;
EXTERN	Plist*	plist;
EXTERN	Plist*	plast;
EXTERN	Pool*	poolist;
EXTERN	Pool*	poolast;
EXTERN	Biobuf*	bout;
EXTERN	int32	dynloc;
EXTERN	uchar	reg[D_NONE];
EXTERN	ushort	txt[NTYPE*NTYPE];
EXTERN	int32	maxround;
EXTERN	int32	widthptr;
EXTERN	Sym*	symstringo;	// string objects
EXTERN	int32	stringo;	// size of string objects
EXTERN	int32	pcloc;		// instruction counter
EXTERN	String	emptystring;
extern	char*	anames[];
EXTERN	Hist*	hist;
EXTERN	Prog	zprog;
EXTERN	Label*	labellist;
EXTERN	Label*	findlab(Sym*);
EXTERN	Node*	curfn;
EXTERN	Node*	newproc;
EXTERN	Node*	throwindex;
EXTERN	Node*	throwreturn;

/*
 * gen.c
 */
void	compile(Node*);
void	proglist(void);
void	gen(Node*, Label*);
void	swgen(Node*);
void	selgen(Node*);
Node*	lookdot(Node*, Node*, int);
void	inarggen(void);
void	agen_inter(Node*, Node*);
void	cgen_as(Node*, Node*, int);
void	cgen_asop(Node*);
void	cgen_ret(Node*);
void	cgen_call(Node*, int);
void	cgen_callmeth(Node*, int);
void	cgen_callinter(Node*, Node*, int);
void	cgen_proc(Node*);
void	cgen_callret(Node*, Node*);
void	cgen_div(int, Node*, Node*, Node*);
void	cgen_shift(int, Node*, Node*, Node*);
void	genpanic(void);
int	needconvert(Type*, Type*);
void	genconv(Type*, Type*);
void	allocparams(void);
void	checklabels();

/*
 * cgen
 */
void	cgen(Node*, Node*);
void	agen(Node*, Node*);
void	igen(Node*, Node*, Node*);
vlong	fieldoffset(Type*, Node*);
void	bgen(Node*, int, Prog*);
void	sgen(Node*, Node*, int32);
void	gmove(Node*, Node*);
Prog*	gins(int, Node*, Node*);
int	samaddr(Node*, Node*);
void	naddr(Node*, Addr*);
void	cgen_aret(Node*, Node*);

/*
 * gsubr.c
 */
void	clearp(Prog*);
void	proglist(void);
Prog*	gbranch(int, Type*);
void	patch(Prog*, Prog*);
Prog*	prog(int);
void	gaddoffset(Node*);
void	gconv(int, int);
int	conv2pt(Type*);
void	belexinit(int);
vlong	convvtox(vlong, int);
int	brcom(int);
int	brrev(int);
void	fnparam(Type*, int, int);
Sig*	lsort(Sig*, int(*)(Sig*, Sig*));
Prog*	gop(int, Node*, Node*, Node*);
void	setconst(Addr*, vlong);
void	setaddr(Addr*, Node*);
int	optoas(int, Type*);
void	ginit(void);
void	gclean(void);
void	regalloc(Node*, Type*, Node*);
void	regfree(Node*);
void	regsalloc(Node*, Type*);	// replace w tmpvar
void	regret(Node*, Type*);
Node*	nodarg(Type*, int);
void	nodreg(Node*, Type*, int);
void	nodindreg(Node*, Type*, int);
void	nodconst(Node*, Type*, vlong);
Sym*	signame(Type*);
void	nodtypesig(Node*, Type*);
void	gconreg(int, vlong, int);
void	buildtxt(void);
void	stringpool(Node*);
void	tempname(Node*, Type*);
Plist*	newplist(void);
int	isfat(Type*);
void	setmaxarg(Type*);

/*
 * list.c
 */
int	Aconv(Fmt*);
int	Dconv(Fmt*);
int	Pconv(Fmt*);
int	Rconv(Fmt*);
int	Yconv(Fmt*);
void	listinit(void);

/*
 * obj
 */
void	zname(Biobuf*, Sym*, int);
void	zaddr(Biobuf*, Addr*, int);
void	ieeedtod(Ieee*, double);
void	dumpstrings(void);
void	dumpsignatures(void);
void	outhist(Biobuf*);

/*
 * align
 */
void	dowidth(Type*);
uint32	rnd(uint32, uint32);
