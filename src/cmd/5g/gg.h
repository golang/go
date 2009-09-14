// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>

#include "../gc/go.h"
#include "../5l/5.out.h"

#ifndef	EXTERN
#define EXTERN	extern
#endif

typedef	struct	Addr	Addr;

struct	Addr
{
	int32	offset;
	int32	offset2;
	double	dval;
	Prog*	branch;
	char	sval[NSNAME];

	Sym*	sym;
	int	width;
	uchar	type;
	char	name;
	char	reg;
	uchar	etype;
};
#define	A	((Addr*)0)

struct	Prog
{
	short	as;			// opcode
	uint32	loc;		// pc offset in this func
	uint32	lineno;		// source line that generated this
	Addr	from;		// src address
	Addr	to;			// dst address
	Prog*	link;		// next instruction in this func
	char	reg;		// doubles as width in DATA op
	uchar	scond;
};

#define REGALLOC_R0 0
#define REGALLOC_RMAX REGEXT
#define REGALLOC_F0 (REGALLOC_RMAX+1)
#define REGALLOC_FMAX (REGALLOC_F0 + FREGEXT)

EXTERN	Biobuf*	bout;
EXTERN	int32	dynloc;
EXTERN	uchar	reg[REGALLOC_FMAX];
EXTERN	int32	pcloc;		// instruction counter
EXTERN	Strlit	emptystring;
extern	char*	anames[];
EXTERN	Hist*	hist;
EXTERN	Prog	zprog;
EXTERN	Node*	curfn;
EXTERN	Node*	newproc;
EXTERN	Node*	deferproc;
EXTERN	Node*	deferreturn;
EXTERN	Node*	throwindex;
EXTERN	Node*	throwreturn;
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
void	checklabels();
void	ginscall(Node*, int);

/*
 * cgen
 */
void	agen(Node*, Node*);
void	igen(Node*, Node*, Node*);
vlong	fieldoffset(Type*, Node*);
void	bgen(Node*, int, Prog*);
void	sgen(Node*, Node*, int32);
void	gmove(Node*, Node*);
Prog*	gins(int, Node*, Node*);
int	samaddr(Node*, Node*);
void	raddr(Node *n, Prog *p);
void	naddr(Node*, Addr*);
void	cgen_aret(Node*, Node*);

/*
 * cgen64.c
 */
void	cmp64(Node*, Node*, int, Prog*);
void	cgen64(Node*, Node*);

/*
 * gsubr.c
 */
void	clearp(Prog*);
void	proglist(void);
Prog*	gbranch(int, Type*);
Prog*	prog(int);
void	gaddoffset(Node*);
void	gconv(int, int);
int	conv2pt(Type*);
vlong	convvtox(vlong, int);
void	fnparam(Type*, int, int);
Prog*	gop(int, Node*, Node*, Node*);
void	setconst(Addr*, vlong);
void	setaddr(Addr*, Node*);
int	optoas(int, Type*);
void	ginit(void);
void	gclean(void);
void	regalloc(Node*, Type*, Node*);
void	regfree(Node*);
void	tempalloc(Node*, Type*);
void	tempfree(Node*);
Node*	nodarg(Type*, int);
void	nodreg(Node*, Type*, int);
void	nodindreg(Node*, Type*, int);
void	buildtxt(void);
Plist*	newplist(void);
int	isfat(Type*);
int	dotaddable(Node*, Node*);
void	sudoclean(void);
int	sudoaddable(int, Node*, Addr*, int*);
void	afunclit(Addr*);
void	datagostring(Strlit*, Addr*);
void	split64(Node*, Node*, Node*);
void	splitclean(void);

/*
 * obj.c
 */
void	datastring(char*, int, Addr*);

/*
 * list.c
 */
int	Aconv(Fmt*);
int	Dconv(Fmt*);
int	Mconv(Fmt*);
int	Pconv(Fmt*);
int	Rconv(Fmt*);
int	Yconv(Fmt*);
void	listinit(void);

void	zaddr(Biobuf*, Addr*, int);
