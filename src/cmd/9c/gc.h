// cmd/9c/gc.h from Vita Nuova.
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2008 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2008 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include	<u.h>
#include	"../cc/cc.h"
#include	"../9l/9.out.h"

/*
 * 9c/powerpc64
 */
#define	SZ_CHAR		1
#define	SZ_SHORT	2
#define	SZ_INT		4
#define	SZ_LONG		4
#define	SZ_IND		8
#define	SZ_FLOAT	4
#define	SZ_VLONG	8
#define	SZ_DOUBLE	8
#define	FNX		100

typedef	struct	Case	Case;
typedef	struct	C1	C1;
typedef	struct	Multab	Multab;
typedef	struct	Hintab	Hintab;
typedef	struct	Reg	Reg;
typedef	struct	Rgn	Rgn;

#define	A	((Adr*)0)

#define	INDEXED	9
#define	P	((Prog*)0)

struct	Case
{
	Case*	link;
	vlong	val;
	int32	label;
	char	def;
	char isv;
};
#define	C	((Case*)0)

struct	C1
{
	vlong	val;
	int32	label;
};

struct	Multab
{
	int32	val;
	char	code[20];
};

struct	Hintab
{
	ushort	val;
	char	hint[10];
};

struct	Reg
{
	int32	pc;
	int32	rpo;		/* reverse post ordering */

	Bits	set;
	Bits	use1;
	Bits	use2;

	Bits	refbehind;
	Bits	refahead;
	Bits	calbehind;
	Bits	calahead;
	Bits	regdiff;
	Bits	act;

	int32	regu;
	int32	loop;		/* could be shorter */

	union
	{
		Reg*	log5;
		int32	active;
	};
	Reg*	p1;
	Reg*	p2;
	Reg*	p2link;
	Reg*	s1;
	Reg*	s2;
	Reg*	link;
	Prog*	prog;
};
#define	R	((Reg*)0)

#define	NRGN	600
struct	Rgn
{
	Reg*	enter;
	short	cost;
	short	varno;
	short	regno;
};

EXTERN	int32	breakpc;
EXTERN	int32	nbreak;
EXTERN	Case*	cases;
EXTERN	Node	constnode;
EXTERN	Node	fconstnode;
EXTERN	Node	vconstnode;
EXTERN	int32	continpc;
EXTERN	int32	curarg;
EXTERN	int32	cursafe;
EXTERN	Prog*	lastp;
extern	int	hintabsize;
EXTERN	int32	maxargsafe;
EXTERN	Multab	multab[20];
EXTERN	int	mnstring;
EXTERN	int	retok;
EXTERN	Node*	nodrat;
EXTERN	Node*	nodret;
EXTERN	Node*	nodsafe;
EXTERN	int32	nrathole;
EXTERN	int32	nstring;
EXTERN	Prog*	p;
EXTERN	int32	pc;
EXTERN	Node	regnode;
EXTERN	Node	qregnode;
EXTERN	char	string[NSNAME];
EXTERN	Sym*	symrathole;
EXTERN	Node	znode;
EXTERN	Prog	zprog;
EXTERN	int	reg[NREG+NREG];
EXTERN	int32	exregoffset;
EXTERN	int32	exfregoffset;
EXTERN	uchar	typechlpv[NTYPE];

#define	BLOAD(r)	band(bnot(r->refbehind), r->refahead)
#define	BSTORE(r)	band(bnot(r->calbehind), r->calahead)
#define	LOAD(r)		(~r->refbehind.b[z] & r->refahead.b[z])
#define	STORE(r)	(~r->calbehind.b[z] & r->calahead.b[z])

#define	bset(a,n)	((a).b[(n)/32]&(1L<<(n)%32))

#define	CLOAD	5
#define	CREF	5
#define	CINF	1000
#define	LOOP	3

EXTERN	Rgn	region[NRGN];
EXTERN	Rgn*	rgp;
EXTERN	int	nregion;
EXTERN	int	nvar;

EXTERN	Bits	externs;
EXTERN	Bits	params;
EXTERN	Bits	consts;
EXTERN	Bits	addrs;

EXTERN	int32	regbits;
EXTERN	int32	exregbits;

EXTERN	int	change;
EXTERN	int	suppress;

EXTERN	Reg*	firstr;
EXTERN	Reg*	lastr;
EXTERN	Reg	zreg;
EXTERN	Reg*	freer;
EXTERN	Var	var[NVAR];
EXTERN	int32*	idom;
EXTERN	Reg**	rpo2r;
EXTERN	int32	maxnr;

#define	R0ISZERO	(debug['0']==0)

extern	char*	anames[];
extern	Hintab	hintab[];

/*
 * sgen.c
 */
void	codgen(Node*, Node*);
void	gen(Node*);
void	usedset(Node*, int);
void	noretval(int);
void	xcom(Node*);
int	bcomplex(Node*, Node*);
Prog*	gtext(Sym*, int32);
vlong	argsize(void);

/*
 * cgen.c
 */
void	cgen(Node*, Node*);
void	reglcgen(Node*, Node*, Node*);
void	lcgen(Node*, Node*);
void	bcgen(Node*, int);
void	boolgen(Node*, int, Node*);
void	sugen(Node*, Node*, int32);
void	layout(Node*, Node*, int, int, Node*);

/*
 * txt.c
 */
void	ginit(void);
void	gclean(void);
void	nextpc(void);
void	gargs(Node*, Node*, Node*);
void	garg1(Node*, Node*, Node*, int, Node**);
Node*	nodconst(int32);
Node*	nod32const(vlong);
Node*	nodfconst(double);
Node*	nodgconst(vlong v, Type *t);
void	nodreg(Node*, Node*, int);
void	regret(Node*, Node*);
void	regalloc(Node*, Node*, Node*);
void	regfree(Node*);
void	regialloc(Node*, Node*, Node*);
void	regsalloc(Node*, Node*);
void	regaalloc1(Node*, Node*);
void	regaalloc(Node*, Node*);
void	regind(Node*, Node*);
void	gprep(Node*, Node*);
void	raddr(Node*, Prog*);
void	naddr(Node*, Addr*);
void	gmove(Node*, Node*);
void	gins(int a, Node*, Node*);
void	gopcode(int, Node*, Node*, Node*);
int	samaddr(Node*, Node*);
void	gbranch(int);
int	immconst(Node*);
void	patch(Prog*, int32);
int	sconst(Node*);
int	sval(int32);
int	uconst(Node*);
void	gpseudo(int, Sym*, Node*);
void	gprefetch(Node*);
void	gpcdata(int, int);

/*
 * swt.c
 */
int	swcmp(const void*, const void*);
void	doswit(Node*);
void	swit1(C1*, int, int32, Node*);
void	swit2(C1*, int, int32, Node*, Node*);
void	newcase(void);
void	bitload(Node*, Node*, Node*, Node*, Node*);
void	bitstore(Node*, Node*, Node*, Node*, Node*);
int32	outstring(char*, int32);
int	mulcon(Node*, Node*);
Multab*	mulcon0(Node*, int32);
int	mulcon1(Node*, int32, Node*);
void	nullwarn(Node*, Node*);
void	sextern(Sym*, Node*, int32, int32);
void	gextern(Sym*, Node*, int32, int32);
void	outcode(void);

/*
 * list
 */
void	listinit(void);
int	Pconv(Fmt*);
int	Aconv(Fmt*);
int	Dconv(Fmt*);
int	Sconv(Fmt*);
int	Nconv(Fmt*);
int	Bconv(Fmt*);

/*
 * reg.c
 */
Reg*	rega(void);
int	rcmp(const void*, const void*);
void	regopt(Prog*);
void	addmove(Reg*, int, int, int);
Bits	mkvar(Addr*, int);
void	prop(Reg*, Bits, Bits);
void	loopit(Reg*, int32);
void	synch(Reg*, Bits);
uint32	allreg(uint32, Rgn*);
void	paint1(Reg*, int);
uint32	paint2(Reg*, int);
void	paint3(Reg*, int, int32, int);
void	addreg(Addr*, int);

/*
 * peep.c
 */
void	peep(void);
void	excise(Reg*);
Reg*	uniqp(Reg*);
Reg*	uniqs(Reg*);
int	regtyp(Addr*);
int	regzer(Addr*);
int	anyvar(Addr*);
int	subprop(Reg*);
int	copyprop(Reg*);
int	copy1(Addr*, Addr*, Reg*, int);
int	copyu(Prog*, Addr*, Addr*);

int	copyas(Addr*, Addr*);
int	copyau(Addr*, Addr*);
int	copyau1(Prog*, Addr*);
int	copysub(Addr*, Addr*, Addr*, int);
int	copysub1(Prog*, Addr*, Addr*, int);

int32	RtoB(int);
int32	FtoB(int);
int	BtoR(int32);
int	BtoF(int32);

/*
 * com64.c
 */
int	com64(Node*);
void	com64init(void);
void	bool64(Node*);

#pragma	varargck	type	"A"	int
#pragma	varargck	type	"B"	Bits
#pragma	varargck	type	"D"	Addr*
#pragma	varargck	type	"N"	Addr*
#pragma	varargck	type	"P"	Prog*
#pragma	varargck	type	"S"	char*
