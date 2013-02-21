// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef	EXTERN
#define	EXTERN	extern
#endif

#include "../gc/go.h"
#include "../8l/8.out.h"

typedef	struct	Addr	Addr;

struct	Addr
{
	int32	offset;
	int32	offset2;

	union {
		double	dval;
		vlong	vval;
		Prog*	branch;
		char	sval[NSNAME];
	} u;

	Sym*	gotype;
	Sym*	sym;
	Node*	node;
	int	width;
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

// foptoas flags
enum
{
	Frev = 1<<0,
	Fpop = 1<<1,
	Fpop2 = 1<<2,
};

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
EXTERN	int	maxstksize;
extern	uint32	unmappedzero;


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
void	cgen_float(Node*, Node*);
void	bgen_float(Node *n, int true, int likely, Prog *to);
void	cgen_dcl(Node*);
int	needconvert(Type*, Type*);
void	genconv(Type*, Type*);
void	allocparams(void);
void	checklabels(void);
void	ginscall(Node*, int);

/*
 * cgen.c
 */
void	agen(Node*, Node*);
void	igen(Node*, Node*, Node*);
vlong	fieldoffset(Type*, Node*);
void	sgen(Node*, Node*, int64);
void	gmove(Node*, Node*);
Prog*	gins(int, Node*, Node*);
int	samaddr(Node*, Node*);
void	naddr(Node*, Addr*, int);
void	cgen_aret(Node*, Node*);
Node*	ncon(uint32);
void	mgen(Node*, Node*, Node*);
void	mfree(Node*);
int	componentgen(Node*, Node*);

/*
 * cgen64.c
 */
void	cmp64(Node*, Node*, int, int, Prog*);
void	cgen64(Node*, Node*);

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
int	foptoas(int, Type*, int);
void	ginit(void);
void	gclean(void);
void	regalloc(Node*, Type*, Node*);
void	regfree(Node*);
Node*	nodarg(Type*, int);
void	nodreg(Node*, Type*, int);
void	nodindreg(Node*, Type*, int);
void	nodconst(Node*, Type*, int64);
void	gconreg(int, vlong, int);
void	buildtxt(void);
Plist*	newplist(void);
int	isfat(Type*);
void	sudoclean(void);
int	sudoaddable(int, Node*, Addr*);
int	dotaddable(Node*, Node*);
void	afunclit(Addr*, Node*);
void	split64(Node*, Node*, Node*);
void	splitclean(void);
void	nswap(Node*, Node*);
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
