// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef	EXTERN
#define	EXTERN	extern
#endif

#include "../gc/go.h"
#include "../5l/5.out.h"

enum
{
	REGALLOC_R0 = REG_R0,
	REGALLOC_RMAX = REGEXT,
	REGALLOC_F0 = REG_F0,
	REGALLOC_FMAX = FREGEXT,
};

EXTERN	uchar	reg[REGALLOC_FMAX+1];
extern	long	unmappedzero;

/*
 * gen.c
 */
void	compile(Node*);
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
void	listinit(void);

void	zaddr(Biobuf*, Addr*, int, int);

void afunclit(Addr*, Node*);
int anyregalloc(void);
void betypeinit(void);
void bgen(Node*, int, int, Prog*);
void cgen(Node*, Node*);
void cgen_call(Node*, int);
void cgen_callinter(Node*, Node*, int);
void cgen_ret(Node*);
void clearfat(Node*);
void clearp(Prog*);
void defframe(Prog*);
int dgostringptr(Sym*, int, char*);
int dgostrlitptr(Sym*, int, Strlit*);
int dsname(Sym*, int, char*, int);
int dsymptr(Sym*, int, Sym*, int);
void dumpdata(void);
void dumpit(char*, Flow*, int);
void excise(Flow*);
void expandchecks(Prog*);
void fixautoused(Prog*);
void gclean(void);
void	gdata(Node*, Node*, int);
void	gdatacomplex(Node*, Mpcplx*);
void	gdatastring(Node*, Strlit*);
void	ggloblnod(Node *nam);
void	ggloblsym(Sym *s, int32 width, int8 flags);
void ginit(void);
Prog*	gins(int, Node*, Node*);
void	ginscall(Node*, int);
Prog*	gjmp(Prog*);
void gtrack(Sym*);
void	gused(Node*);
void	igen(Node*, Node*, Node*);
int isfat(Type*);
void linkarchinit(void);
void markautoused(Prog*);
void naddr(Node*, Addr*, int);
Plist* newplist(void);
Node* nodarg(Type*, int);
void patch(Prog*, Prog*);
void proginfo(ProgInfo*, Prog*);
void regalloc(Node*, Type*, Node*);
void regfree(Node*);
void regopt(Prog*);
int regtyp(Addr*);
int sameaddr(Addr*, Addr*);
int smallindir(Addr*, Addr*);
int stackaddr(Addr*);
Prog* unpatch(Prog*);

/*
 * reg.c
 */
uint64 excludedregs(void);
uint64 RtoB(int);
uint64 FtoB(int);
int BtoR(uint64);
int BtoF(uint64);
uint64 doregbits(int);
char** regnames(int*);

/*
 * peep.c
 */
void peep(Prog*);
