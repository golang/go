// Derived from Inferno utils/6c/gc.h
// http://code.google.com/p/inferno-os/source/browse/utils/6c/gc.h
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

	Bits	set;  		// regopt variables written by this instruction.
	Bits	use1; 		// regopt variables read by prog->from.
	Bits	use2; 		// regopt variables read by prog->to.

	// refahead/refbehind are the regopt variables whose current
	// value may be used in the following/preceding instructions
	// up to a CALL (or the value is clobbered).
	Bits	refbehind;
	Bits	refahead;
	// calahead/calbehind are similar, but for variables in
	// instructions that are reachable after hitting at least one
	// CALL.
	Bits	calbehind;
	Bits	calahead;
	Bits	regdiff;
	Bits	act;

	uint64	regu;		// register used bitmap
};
#define	R	((Reg*)0)
/*c2go extern Reg *R; */

#define	NRGN	600
/*c2go enum { NRGN = 600 }; */

// A Rgn represents a single regopt variable over a region of code
// where a register could potentially be dedicated to that variable.
// The code encompassed by a Rgn is defined by the flow graph,
// starting at enter, flood-filling forward while varno is refahead
// and backward while varno is refbehind, and following branches.  A
// single variable may be represented by multiple disjoint Rgns and
// each Rgn may choose a different register for that variable.
// Registers are allocated to regions greedily in order of descending
// cost.
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
EXTERN	Rgn	region[NRGN];
EXTERN	Rgn*	rgp;
EXTERN	int	nregion;
EXTERN	int	nvar;
EXTERN	int32	regbits;
EXTERN	int32	exregbits;		// TODO(austin) not used; remove
EXTERN	Bits	externs;
EXTERN	Bits	params;
EXTERN	Bits	consts;
EXTERN	Bits	addrs;
EXTERN	Bits	ivar;
EXTERN	Bits	ovar;
EXTERN	int	change;
EXTERN	int32	maxnr;

EXTERN	struct
{
	int32	ncvtreg;
	int32	nspill;
	int32	ndelmov;
	int32	nvar;
} ostats;

/*
 * reg.c
 */
int	rcmp(const void*, const void*);
void	regopt(Prog*);
void	addmove(Reg*, int, int, int);
Bits	mkvar(Reg*, Adr*);
void	prop(Reg*, Bits, Bits);
void	synch(Reg*, Bits);
uint64	allreg(uint64, Rgn*);
void	paint1(Reg*, int);
uint64	paint2(Reg*, int, int);
void	paint3(Reg*, int, uint64, int);
void	addreg(Adr*, int);
void	dumpone(Flow*, int);
void	dumpit(char*, Flow*, int);

/*
 * peep.c
 */
void	peep(Prog*);
void	excise(Flow*);
int	copyu(Prog*, Adr*, Adr*);

uint64	RtoB(int);
uint64	FtoB(int);
int	BtoR(uint64);
int	BtoF(uint64);

/*
 * prog.c
 */
typedef struct ProgInfo ProgInfo;
struct ProgInfo
{
	uint32 flags; // the bits below
	uint64 reguse; // registers implicitly used by this instruction
	uint64 regset; // registers implicitly set by this instruction
	uint64 regindex; // registers used by addressing mode
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

	// Left side (Prog.from): address taken, read, write.
	LeftAddr = 1<<9,
	LeftRead = 1<<10,
	LeftWrite = 1<<11,

	// Register in middle (Prog.reg); only ever read.
	RegRead = 1<<12,
	CanRegRead = 1<<13,

	// Right side (Prog.to): address taken, read, write.
	RightAddr = 1<<14,
	RightRead = 1<<15,
	RightWrite = 1<<16,

	// Instruction updates whichever of from/to is type D_OREG
	PostInc = 1<<17,

	// Instruction kinds
	Move = 1<<18, // straight move
	Conv = 1<<19, // size conversion
	Cjmp = 1<<20, // conditional jump
	Break = 1<<21, // breaks control flow (no fallthrough)
	Call = 1<<22, // function call
	Jump = 1<<23, // jump
	Skip = 1<<24, // data instruction
};

void proginfo(ProgInfo*, Prog*);

// Many Power ISA arithmetic and logical instructions come in four
// standard variants.  These bits let us map between variants.
enum {
	V_CC = 1<<0,		// xCC (affect CR field 0 flags)
	V_V  = 1<<1,		// xV (affect SO and OV flags)
};

int as2variant(int);
int variant2as(int, int);

// To allow use of AJMP, ACALL, ARET in ../gc/popt.c.
enum
{
	AJMP = ABR,
	ACALL = ABL,
	ARET = ARETURN,
};
