// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef	EXTERN
#define	EXTERN	extern
#endif

#include "../gc/go.h"
#include "../5l/5.out.h"

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

	Sym*	sym;
	Sym*	gotype;
	Node*	node;
	int	width;
	uchar	type;
	char	name;
	uchar	reg;
	uchar	etype;
};
#define	A	((Addr*)0)

struct	Prog
{
	uint32	loc;		// pc offset in this func
	uint32	lineno;		// source line that generated this
	Prog*	link;		// next instruction in this func
	void*	regp;		// points to enclosing Reg struct
	short	as;		// opcode
	uchar	reg;		// doubles as width in DATA op
	uchar	scond;
	Addr	from;		// src address
	Addr	to;		// dst address
};

#define TEXTFLAG reg

#define REGALLOC_R0 0
#define REGALLOC_RMAX REGEXT
#define REGALLOC_F0 (REGALLOC_RMAX+1)
#define REGALLOC_FMAX (REGALLOC_F0 + FREGEXT)

EXTERN	int32	dynloc;
EXTERN	uchar	reg[REGALLOC_FMAX+1];
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
extern	long	unmappedzero;
EXTERN	int	maxstksize;

/*
 * gen.c
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
void	cgen_dcl(Node*);
int	needconvert(Type*, Type*);
void	genconv(Type*, Type*);
void	allocparams(void);
void	checklabels(void);
void	ginscall(Node*, int);

/*
 * cgen
 */
void	agen(Node*, Node*);
Prog* cgenindex(Node *, Node *, int);
void	igen(Node*, Node*, Node*);
void agenr(Node *n, Node *a, Node *res);
vlong	fieldoffset(Type*, Node*);
void	sgen(Node*, Node*, int64);
void	gmove(Node*, Node*);
Prog*	gins(int, Node*, Node*);
int	samaddr(Node*, Node*);
void	raddr(Node *n, Prog *p);
Prog*	gcmp(int, Node*, Node*);
Prog*	gshift(int as, Node *lhs, int32 stype, int32 sval, Node *rhs);
Prog *	gregshift(int as, Node *lhs, int32 stype, Node *reg, Node *rhs);
void	naddr(Node*, Addr*, int);
void	cgen_aret(Node*, Node*);
void	cgen_hmul(Node*, Node*, Node*);
void	cgen_shift(int, int, Node*, Node*, Node*);
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
void	ginit(void);
void	gclean(void);
void	regalloc(Node*, Type*, Node*);
void	regfree(Node*);
Node*	nodarg(Type*, int);
void	nodreg(Node*, Type*, int);
void	nodindreg(Node*, Type*, int);
void	buildtxt(void);
Plist*	newplist(void);
int	isfat(Type*);
int	dotaddable(Node*, Node*);
void	sudoclean(void);
int	sudoaddable(int, Node*, Addr*, int*);
void	afunclit(Addr*, Node*);
void	datagostring(Strlit*, Addr*);
void	split64(Node*, Node*, Node*);
void	splitclean(void);
Node*	ncon(uint32 i);
void	gtrack(Sym*);

/*
 * obj.c
 */
void	datastring(char*, int, Addr*);

/*
 * list.c
 */
int	Aconv(Fmt*);
int	Cconv(Fmt*);
int	Dconv(Fmt*);
int	Mconv(Fmt*);
int	Pconv(Fmt*);
int	Rconv(Fmt*);
int	Yconv(Fmt*);
void	listinit(void);

void	zaddr(Biobuf*, Addr*, int, int);

#pragma	varargck	type	"D"	Addr*
#pragma	varargck	type	"M"	Addr*
