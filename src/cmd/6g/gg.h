// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef	EXTERN
#define	EXTERN	extern
#endif

#include "../gc/go.h"
#include "../6l/6.out.h"

EXTERN	uchar	reg[MAXREG];
EXTERN	Node*	panicdiv;
extern	vlong	unmappedzero;
extern	int	addptr;
extern	int	cmpptr;
extern	int	movptr;
extern	int	leaptr;

/*
 * ggen.c
 */
void	compile(Node*);
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
void	fixlargeoffset(Node *n);

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
