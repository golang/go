// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef	EXTERN
#define	EXTERN	extern
#endif

#include "../gc/go.h"
#include "../6l/6.out.h"

typedef	struct	Addr	Addr;

struct	Addr
{
	vlong	offset;
	
	union {
		double	dval;
		vlong	vval;
		Prog*	branch;
		char	sval[NSNAME];
	} u;

	Sym*	gotype;
	Sym*	sym;
	Node*	node;
	int64	width;
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
	void*	reg;		// pointer to containing Reg struct
};

#define TEXTFLAG from.scale

EXTERN	int32	dynloc;
EXTERN	uchar	reg[D_NONE];
EXTERN	int32	pcloc;		// instruction counter
EXTERN	Strlit	emptystring;
extern	char*	anames[];
EXTERN	Prog	zprog;
EXTERN	Node*	newproc;
EXTERN	Node*	deferproc;
EXTERN	Node*	deferreturn;
EXTERN	Node*	panicindex;
EXTERN	Node*	panicslice;
EXTERN	Node*	throwreturn;
extern	vlong	unmappedzero;

/*
 * ggen.c
 */
void	compile(Node*);
void	proglist(void);
void	gen(Node*);
Node*	lookdot(Node*, Node*, int);
void	cgen_as(Node*, Node*);
void	cgen_callmeth(Node*, int);
void	cgen_callinter(Node*, Node*, int);
void	cgen_proc(Node*, int);
void	cgen_callret(Node*, Node*);
void	cgen_div(int, Node*, Node*, Node*);
void	cgen_bmul(int, Node*, Node*, Node*);
void	cgen_hmul(Node*, Node*, Node*);
void	cgen_shift(int, int, Node*, Node*, Node*);
void	cgen_dcl(Node*);
int	needconvert(Type*, Type*);
void	genconv(Type*, Type*);
void	allocparams(void);
void	checklabels(void);
void	ginscall(Node*, int);
int	gen_as_init(Node*);
void	clearslim(Node*);

/*
 * cgen.c
 */
void	agen(Node*, Node*);
void	agenr(Node*, Node*, Node*);
void	cgenr(Node*, Node*, Node*);
void	igen(Node*, Node*, Node*);
vlong	fieldoffset(Type*, Node*);
void	sgen(Node*, Node*, int64);
void	gmove(Node*, Node*);
Prog*	gins(int, Node*, Node*);
int	samaddr(Node*, Node*);
void	naddr(Node*, Addr*, int);
void	cgen_aret(Node*, Node*);
void	restx(Node*, Node*);
void	savex(int, Node*, Node*, Node*, Type*);
int	componentgen(Node*, Node*);

/*
 * gsubr.c
 */
void	clearp(Prog*);
void	proglist(void);
Prog*	gbranch(int, Type*, int);
Prog*	prog(int);
void	gconv(int, int);
int	conv2pt(Type*);
vlong	convvtox(vlong, int);
void	fnparam(Type*, int, int);
Prog*	gop(int, Node*, Node*, Node*);
int	optoas(int, Type*);
void	ginit(void);
void	gclean(void);
void	regalloc(Node*, Type*, Node*);
void	regfree(Node*);
Node*	nodarg(Type*, int);
void	nodreg(Node*, Type*, int);
void	nodindreg(Node*, Type*, int);
void	gconreg(int, vlong, int);
void	ginscon(int, vlong, Node*);
void	buildtxt(void);
Plist*	newplist(void);
int	isfat(Type*);
void	sudoclean(void);
int	sudoaddable(int, Node*, Addr*);
void	afunclit(Addr*, Node*);
void	nodfconst(Node*, Type*, Mpflt*);
void	gtrack(Sym*);

/*
 * cplx.c
 */
int	complexop(Node*, Node*);
void	complexmove(Node*, Node*);
void	complexgen(Node*, Node*);

/*
 * gobj.c
 */
void	datastring(char*, int, Addr*);
void	datagostring(Strlit*, Addr*);

/*
 * list.c
 */
int	Aconv(Fmt*);
int	Dconv(Fmt*);
int	Pconv(Fmt*);
int	Rconv(Fmt*);
int	Yconv(Fmt*);
void	listinit(void);

void	zaddr(Biobuf*, Addr*, int, int);

#pragma	varargck	type	"D"	Addr*
#pragma	varargck	type	"lD"	Addr*
