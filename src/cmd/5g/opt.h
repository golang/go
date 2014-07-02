// Inferno utils/5c/gc.h
// http://code.google.com/p/inferno-os/source/browse/utils/5c/gc.h
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
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

#include	"../gc/popt.h"

#define	Z	N
#define	Adr	Addr

#define	D_HI	D_NONE
#define	D_LO	D_NONE

#define	BLOAD(r)	band(bnot(r->refbehind), r->refahead)
#define	BSTORE(r)	band(bnot(r->calbehind), r->calahead)
#define	LOAD(r)		(~r->refbehind.b[z] & r->refahead.b[z])
#define	STORE(r)	(~r->calbehind.b[z] & r->calahead.b[z])

#define	CLOAD	5
#define	CREF	5
#define	CINF	1000
#define	LOOP	3

typedef	struct	Reg	Reg;
typedef	struct	Rgn	Rgn;

/*c2go
extern Node *Z;
enum
{
	D_HI = D_NONE,
	D_LO = D_NONE,
	CLOAD = 5,
	CREF = 5,
	CINF = 1000,
	LOOP = 3,
};

uint32 BLOAD(Reg*);
uint32 BSTORE(Reg*);
uint32 LOAD(Reg*);
uint32 STORE(Reg*);
*/

// A Reg is a wrapper around a single Prog (one instruction) that holds
// register optimization information while the optimizer runs.
// r->prog is the instruction.
// r->prog->opt points back to r.
struct	Reg
{
	Flow	f;

	Bits	set;  		// variables written by this instruction.
	Bits	use1; 		// variables read by prog->from.
	Bits	use2; 		// variables read by prog->to.

	Bits	refbehind;
	Bits	refahead;
	Bits	calbehind;
	Bits	calahead;
	Bits	regdiff;
	Bits	act;

	int32	regu;		// register used bitmap
};
#define	R	((Reg*)0)
/*c2go extern Reg *R; */

#define	NRGN	600
/*c2go enum { NRGN = 600 }; */
struct	Rgn
{
	Reg*	enter;
	short	cost;
	short	varno;
	short	regno;
};

EXTERN	int32	exregoffset;		// not set
EXTERN	int32	exfregoffset;		// not set
EXTERN	Reg	zreg;
EXTERN	Reg*	freer;
EXTERN	Reg**	rpo2r;
EXTERN	Rgn	region[NRGN];
EXTERN	Rgn*	rgp;
EXTERN	int	nregion;
EXTERN	int	nvar;
EXTERN	int32	regbits;
EXTERN	int32	exregbits;
EXTERN	Bits	externs;
EXTERN	Bits	params;
EXTERN	Bits	consts;
EXTERN	Bits	addrs;
EXTERN	Bits	ivar;
EXTERN	Bits	ovar;
EXTERN	int	change;
EXTERN	int32	maxnr;
EXTERN	int32*	idom;

EXTERN	struct
{
	int32	ncvtreg;
	int32	nspill;
	int32	nreload;
	int32	ndelmov;
	int32	nvar;
	int32	naddr;
} ostats;

/*
 * reg.c
 */
Reg*	rega(void);
int	rcmp(const void*, const void*);
void	regopt(Prog*);
void	addmove(Reg*, int, int, int);
Bits	mkvar(Reg *r, Adr *a);
void	prop(Reg*, Bits, Bits);
void	synch(Reg*, Bits);
uint32	allreg(uint32, Rgn*);
void	paint1(Reg*, int);
uint32	paint2(Reg*, int);
void	paint3(Reg*, int, int32, int);
void	addreg(Adr*, int);
void	dumpit(char *str, Flow *r0, int);

/*
 * peep.c
 */
void	peep(Prog*);
void	excise(Flow*);
int	copyu(Prog*, Adr*, Adr*);

int32	RtoB(int);
int32	FtoB(int);
int	BtoR(int32);
int	BtoF(int32);

/*
 * prog.c
 */
typedef struct ProgInfo ProgInfo;
struct ProgInfo
{
	uint32 flags; // the bits below
};

enum
{
	// Pseudo-op, like TEXT, GLOBL, TYPE, PCDATA, FUNCDATA.
	Pseudo = 1<<1,
	
	// There's nothing to say about the instruction,
	// but it's still okay to see.
	OK = 1<<2,

	// Size of right-side write, or right-side read if no write.
	SizeB = 1<<3,
	SizeW = 1<<4,
	SizeL = 1<<5,
	SizeQ = 1<<6,
	SizeF = 1<<7, // float aka float32
	SizeD = 1<<8, // double aka float64

	// Left side: address taken, read, write.
	LeftAddr = 1<<9,
	LeftRead = 1<<10,
	LeftWrite = 1<<11,
	
	// Register in middle; never written.
	RegRead = 1<<12,
	CanRegRead = 1<<13,
	
	// Right side: address taken, read, write.
	RightAddr = 1<<14,
	RightRead = 1<<15,
	RightWrite = 1<<16,

	// Instruction kinds
	Move = 1<<17, // straight move
	Conv = 1<<18, // size conversion
	Cjmp = 1<<19, // conditional jump
	Break = 1<<20, // breaks control flow (no fallthrough)
	Call = 1<<21, // function call
	Jump = 1<<22, // jump
	Skip = 1<<23, // data instruction
};

void proginfo(ProgInfo*, Prog*);

// To allow use of AJMP and ACALL in ../gc/popt.c.
enum
{
	AJMP = AB,
	ACALL = ABL,
};
