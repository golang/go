// Inferno utils/8c/cgen64.c
// http://code.google.com/p/inferno-os/source/browse/utils/8c/cgen64.c
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

#include "gc.h"

void
zeroregm(Node *n)
{
	gins(AMOVL, nodconst(0), n);
}

/* do we need to load the address of a vlong? */
int
vaddr(Node *n, int a)
{
	switch(n->op) {
	case ONAME:
		if(a)
			return 1;
		return !(n->class == CEXTERN || n->class == CGLOBL || n->class == CSTATIC);

	case OCONST:
	case OREGISTER:
	case OINDREG:
		return 1;
	}
	return 0;
}

int32
hi64v(Node *n)
{
	if(align(0, types[TCHAR], Aarg1, nil))	/* isbigendian */
		return (int32)(n->vconst) & ~0L;
	else
		return (int32)((uvlong)n->vconst>>32) & ~0L;
}

int32
lo64v(Node *n)
{
	if(align(0, types[TCHAR], Aarg1, nil))	/* isbigendian */
		return (int32)((uvlong)n->vconst>>32) & ~0L;
	else
		return (int32)(n->vconst) & ~0L;
}

Node *
hi64(Node *n)
{
	return nodconst(hi64v(n));
}

Node *
lo64(Node *n)
{
	return nodconst(lo64v(n));
}

static Node *
anonreg(void)
{
	Node *n;

	n = new(OREGISTER, Z, Z);
	n->reg = D_NONE;
	n->type = types[TLONG];
	return n;
}

static Node *
regpair(Node *n, Node *t)
{
	Node *r;

	if(n != Z && n->op == OREGPAIR)
		return n;
	r = new(OREGPAIR, anonreg(), anonreg());
	if(n != Z)
		r->type = n->type;
	else
		r->type = t->type;
	return r;
}

static void
evacaxdx(Node *r)
{
	Node nod1, nod2;

	if(r->reg == D_AX || r->reg == D_DX) {
		reg[D_AX]++;
		reg[D_DX]++;
		/*
		 * this is just an optim that should
		 * check for spill
		 */
		r->type = types[TULONG];
		regalloc(&nod1, r, Z);
		nodreg(&nod2, Z, r->reg);
		gins(AMOVL, &nod2, &nod1);
		regfree(r);
		r->reg = nod1.reg;
		reg[D_AX]--;
		reg[D_DX]--;
	}
}

/* lazy instantiation of register pair */
static int
instpair(Node *n, Node *l)
{
	int r;

	r = 0;
	if(n->left->reg == D_NONE) {
		if(l != Z) {
			n->left->reg = l->reg;
			r = 1;
		}
		else
			regalloc(n->left, n->left, Z);
	}
	if(n->right->reg == D_NONE)
		regalloc(n->right, n->right, Z);
	return r;
}

static void
zapreg(Node *n)
{
	if(n->reg != D_NONE) {
		regfree(n);
		n->reg = D_NONE;
	}
}

static void
freepair(Node *n)
{
	regfree(n->left);
	regfree(n->right);
}

/* n is not OREGPAIR, nn is */
void
loadpair(Node *n, Node *nn)
{
	Node nod;

	instpair(nn, Z);
	if(n->op == OCONST) {
		gins(AMOVL, lo64(n), nn->left);
		n->xoffset += SZ_LONG;
		gins(AMOVL, hi64(n), nn->right);
		n->xoffset -= SZ_LONG;
		return;
	}
	if(!vaddr(n, 0)) {
		/* steal the right register for the laddr */
		nod = regnode;
		nod.reg = nn->right->reg;
		lcgen(n, &nod);
		n = &nod;
		regind(n, n);
		n->xoffset = 0;
	}
	gins(AMOVL, n, nn->left);
	n->xoffset += SZ_LONG;
	gins(AMOVL, n, nn->right);
	n->xoffset -= SZ_LONG;
}

/* n is OREGPAIR, nn is not */
static void
storepair(Node *n, Node *nn, int f)
{
	Node nod;

	if(!vaddr(nn, 0)) {
		reglcgen(&nod, nn, Z);
		nn = &nod;
	}
	gins(AMOVL, n->left, nn);
	nn->xoffset += SZ_LONG;
	gins(AMOVL, n->right, nn);
	nn->xoffset -= SZ_LONG;
	if(nn == &nod)
		regfree(&nod);
	if(f)
		freepair(n);
}

enum
{
/* 4 only, see WW */
	WNONE	= 0,
	WCONST,
	WADDR,
	WHARD,
};

static int
whatof(Node *n, int a)
{
	if(n->op == OCONST)
		return WCONST;
	return !vaddr(n, a) ? WHARD : WADDR;
}

/* can upgrade an extern to addr for AND */
static int
reduxv(Node *n)
{
	return lo64v(n) == 0 || hi64v(n) == 0;
}

int
cond(int op)
{
	switch(op) {
	case OANDAND:
	case OOROR:
	case ONOT:
		return 1;

	case OEQ:
	case ONE:
	case OLE:
	case OLT:
	case OGE:
	case OGT:
	case OHI:
	case OHS:
	case OLO:
	case OLS:
		return 1;
	}
	return 0;
}

/*
 * for a func operand call it and then return
 * the safe node
 */
static Node *
vfunc(Node *n, Node *nn)
{
	Node *t;

	if(n->op != OFUNC)
		return n;
	t = new(0, Z, Z);
	if(nn == Z || nn == nodret)
		nn = n;
	regsalloc(t, nn);
	sugen(n, t, 8);
	return t;
}

/* try to steal a reg */
static int
getreg(Node **np, Node *t, int r)
{
	Node *n, *p;

	n = *np;
	if(n->reg == r) {
		p = new(0, Z, Z);
		regalloc(p, n, Z);
		gins(AMOVL, n, p);
		*t = *n;
		*np = p;
		return 1;
	}
	return 0;
}

static Node *
snarfreg(Node *n, Node *t, int r, Node *d, Node *c)
{
	if(n == Z || n->op != OREGPAIR || (!getreg(&n->left, t, r) && !getreg(&n->right, t, r))) {
		if(nodreg(t, Z, r)) {
			regalloc(c, d, Z);
			gins(AMOVL, t, c);
			reg[r]++;
			return c;
		}
		reg[r]++;
	}
	return Z;
}

enum
{
	Vstart	= OEND,

	Vgo,
	Vamv,
	Vmv,
	Vzero,
	Vop,
	Vopx,
	Vins,
	Vins0,
	Vinsl,
	Vinsr,
	Vinsla,
	Vinsra,
	Vinsx,
	Vmul,
	Vshll,
	VT,
	VF,
	V_l_lo_f,
	V_l_hi_f,
	V_l_lo_t,
	V_l_hi_t,
	V_l_lo_u,
	V_l_hi_u,
	V_r_lo_f,
	V_r_hi_f,
	V_r_lo_t,
	V_r_hi_t,
	V_r_lo_u,
	V_r_hi_u,
	Vspazz,
	Vend,

	V_T0,
	V_T1,
	V_F0,
	V_F1,

	V_a0,
	V_a1,
	V_f0,
	V_f1,

	V_p0,
	V_p1,
	V_p2,
	V_p3,
	V_p4,

	V_s0,
	V_s1,
	V_s2,
	V_s3,
	V_s4,

	C00,
	C01,
	C31,
	C32,

	O_l_lo,
	O_l_hi,
	O_r_lo,
	O_r_hi,
	O_t_lo,
	O_t_hi,
	O_l,
	O_r,
	O_l_rp,
	O_r_rp,
	O_t_rp,
	O_r0,
	O_r1,
	O_Zop,

	O_a0,
	O_a1,

	V_C0,
	V_C1,

	V_S0,
	V_S1,

	VOPS	= 5,
	VLEN	= 5,
	VARGS	= 2,

	S00	= 0,
	Sc0,
	Sc1,
	Sc2,
	Sac3,
	Sac4,
	S10,

	SAgen	= 0,
	SAclo,
	SAc32,
	SAchi,
	SAdgen,
	SAdclo,
	SAdc32,
	SAdchi,

	B0c	= 0,
	Bca,
	Bac,

	T0i	= 0,
	Tii,

	Bop0	= 0,
	Bop1,
};

/*
 * _testv:
 * 	CMPL	lo,$0
 * 	JNE	true
 * 	CMPL	hi,$0
 * 	JNE	true
 * 	GOTO	false
 * false:
 * 	GOTO	code
 * true:
 * 	GOTO	patchme
 * code:
 */

static uchar	testi[][VLEN] =
{
	{Vop, ONE, O_l_lo, C00},
	{V_s0, Vop, ONE, O_l_hi, C00},
	{V_s1, Vgo, V_s2, Vgo, V_s3},
	{VF, V_p0, V_p1, VT, V_p2},
	{Vgo, V_p3},
	{VT, V_p0, V_p1, VF, V_p2},
	{Vend},
};

/* shift left general case */
static uchar	shll00[][VLEN] =
{
	{Vop, OGE, O_r, C32},
	{V_s0, Vinsl, ASHLL, O_r, O_l_rp},
	{Vins, ASHLL, O_r, O_l_lo, Vgo},
	{V_p0, V_s0},
	{Vins, ASHLL, O_r, O_l_lo},
	{Vins, AMOVL, O_l_lo, O_l_hi},
	{Vzero, O_l_lo, V_p0, Vend},
};

/* shift left rp, const < 32 */
static uchar	shllc0[][VLEN] =
{
	{Vinsl, ASHLL, O_r, O_l_rp},
	{Vshll, O_r, O_l_lo, Vend},
};

/* shift left rp, const == 32 */
static uchar	shllc1[][VLEN] =
{
	{Vins, AMOVL, O_l_lo, O_l_hi},
	{Vzero, O_l_lo, Vend},
};

/* shift left rp, const > 32 */
static uchar	shllc2[][VLEN] =
{
	{Vshll, O_r, O_l_lo},
	{Vins, AMOVL, O_l_lo, O_l_hi},
	{Vzero, O_l_lo, Vend},
};

/* shift left addr, const == 32 */
static uchar	shllac3[][VLEN] =
{
	{Vins, AMOVL, O_l_lo, O_t_hi},
	{Vzero, O_t_lo, Vend},
};

/* shift left addr, const > 32 */
static uchar	shllac4[][VLEN] =
{
	{Vins, AMOVL, O_l_lo, O_t_hi},
	{Vshll, O_r, O_t_hi},
	{Vzero, O_t_lo, Vend},
};

/* shift left of constant */
static uchar	shll10[][VLEN] =
{
	{Vop, OGE, O_r, C32},
	{V_s0, Vins, AMOVL, O_l_lo, O_t_lo},
	{Vins, AMOVL, O_l_hi, O_t_hi},
	{Vinsl, ASHLL, O_r, O_t_rp},
	{Vins, ASHLL, O_r, O_t_lo, Vgo},
	{V_p0, V_s0},
	{Vins, AMOVL, O_l_lo, O_t_hi},
	{V_l_lo_t, Vins, ASHLL, O_r, O_t_hi},
	{Vzero, O_t_lo, V_p0, Vend},
};

static uchar	(*shlltab[])[VLEN] =
{
	shll00,
	shllc0,
	shllc1,
	shllc2,
	shllac3,
	shllac4,
	shll10,
};

/* shift right general case */
static uchar	shrl00[][VLEN] =
{
	{Vop, OGE, O_r, C32},
	{V_s0, Vinsr, ASHRL, O_r, O_l_rp},
	{Vins, O_a0, O_r, O_l_hi, Vgo},
	{V_p0, V_s0},
	{Vins, O_a0, O_r, O_l_hi},
	{Vins, AMOVL, O_l_hi, O_l_lo},
	{V_T1, Vzero, O_l_hi},
	{V_F1, Vins, ASARL, C31, O_l_hi},
	{V_p0, Vend},
};

/* shift right rp, const < 32 */
static uchar	shrlc0[][VLEN] =
{
	{Vinsr, ASHRL, O_r, O_l_rp},
	{Vins, O_a0, O_r, O_l_hi, Vend},
};

/* shift right rp, const == 32 */
static uchar	shrlc1[][VLEN] =
{
	{Vins, AMOVL, O_l_hi, O_l_lo},
	{V_T1, Vzero, O_l_hi},
	{V_F1, Vins, ASARL, C31, O_l_hi},
	{Vend},
};

/* shift right rp, const > 32 */
static uchar	shrlc2[][VLEN] =
{
	{Vins, O_a0, O_r, O_l_hi},
	{Vins, AMOVL, O_l_hi, O_l_lo},
	{V_T1, Vzero, O_l_hi},
	{V_F1, Vins, ASARL, C31, O_l_hi},
	{Vend},
};

/* shift right addr, const == 32 */
static uchar	shrlac3[][VLEN] =
{
	{Vins, AMOVL, O_l_hi, O_t_lo},
	{V_T1, Vzero, O_t_hi},
	{V_F1, Vins, AMOVL, O_t_lo, O_t_hi},
	{V_F1, Vins, ASARL, C31, O_t_hi},
	{Vend},
};

/* shift right addr, const > 32 */
static uchar	shrlac4[][VLEN] =
{
	{Vins, AMOVL, O_l_hi, O_t_lo},
	{Vins, O_a0, O_r, O_t_lo},
	{V_T1, Vzero, O_t_hi},
	{V_F1, Vins, AMOVL, O_t_lo, O_t_hi},
	{V_F1, Vins, ASARL, C31, O_t_hi},
	{Vend},
};

/* shift right of constant */
static uchar	shrl10[][VLEN] =
{
	{Vop, OGE, O_r, C32},
	{V_s0, Vins, AMOVL, O_l_lo, O_t_lo},
	{Vins, AMOVL, O_l_hi, O_t_hi},
	{Vinsr, ASHRL, O_r, O_t_rp},
	{Vins, O_a0, O_r, O_t_hi, Vgo},
	{V_p0, V_s0},
	{Vins, AMOVL, O_l_hi, O_t_lo},
	{V_l_hi_t, Vins, O_a0, O_r, O_t_lo},
	{V_l_hi_u, V_S1},
	{V_T1, Vzero, O_t_hi, V_p0},
	{V_F1, Vins, AMOVL, O_t_lo, O_t_hi},
	{V_F1, Vins, ASARL, C31, O_t_hi},
	{Vend},
};

static uchar	(*shrltab[])[VLEN] =
{
	shrl00,
	shrlc0,
	shrlc1,
	shrlc2,
	shrlac3,
	shrlac4,
	shrl10,
};

/* shift asop left general case */
static uchar	asshllgen[][VLEN] =
{
	{V_a0, V_a1},
	{Vop, OGE, O_r, C32},
	{V_s0, Vins, AMOVL, O_l_lo, O_r0},
	{Vins, AMOVL, O_l_hi, O_r1},
	{Vinsla, ASHLL, O_r, O_r0},
	{Vins, ASHLL, O_r, O_r0},
	{Vins, AMOVL, O_r1, O_l_hi},
	{Vins, AMOVL, O_r0, O_l_lo, Vgo},
	{V_p0, V_s0},
	{Vins, AMOVL, O_l_lo, O_r0},
	{Vzero, O_l_lo},
	{Vins, ASHLL, O_r, O_r0},
	{Vins, AMOVL, O_r0, O_l_hi, V_p0},
	{V_f0, V_f1, Vend},
};

/* shift asop left, const < 32 */
static uchar	asshllclo[][VLEN] =
{
	{V_a0, V_a1},
	{Vins, AMOVL, O_l_lo, O_r0},
	{Vins, AMOVL, O_l_hi, O_r1},
	{Vinsla, ASHLL, O_r, O_r0},
	{Vshll, O_r, O_r0},
	{Vins, AMOVL, O_r1, O_l_hi},
	{Vins, AMOVL, O_r0, O_l_lo},
	{V_f0, V_f1, Vend},
};

/* shift asop left, const == 32 */
static uchar	asshllc32[][VLEN] =
{
	{V_a0},
	{Vins, AMOVL, O_l_lo, O_r0},
	{Vzero, O_l_lo},
	{Vins, AMOVL, O_r0, O_l_hi},
	{V_f0, Vend},
};

/* shift asop left, const > 32 */
static uchar	asshllchi[][VLEN] =
{
	{V_a0},
	{Vins, AMOVL, O_l_lo, O_r0},
	{Vzero, O_l_lo},
	{Vshll, O_r, O_r0},
	{Vins, AMOVL, O_r0, O_l_hi},
	{V_f0, Vend},
};

/* shift asop dest left general case */
static uchar	asdshllgen[][VLEN] =
{
	{Vop, OGE, O_r, C32},
	{V_s0, Vins, AMOVL, O_l_lo, O_t_lo},
	{Vins, AMOVL, O_l_hi, O_t_hi},
	{Vinsl, ASHLL, O_r, O_t_rp},
	{Vins, ASHLL, O_r, O_t_lo},
	{Vins, AMOVL, O_t_hi, O_l_hi},
	{Vins, AMOVL, O_t_lo, O_l_lo, Vgo},
	{V_p0, V_s0},
	{Vins, AMOVL, O_l_lo, O_t_hi},
	{Vzero, O_l_lo},
	{Vins, ASHLL, O_r, O_t_hi},
	{Vzero, O_t_lo},
	{Vins, AMOVL, O_t_hi, O_l_hi, V_p0},
	{Vend},
};

/* shift asop dest left, const < 32 */
static uchar	asdshllclo[][VLEN] =
{
	{Vins, AMOVL, O_l_lo, O_t_lo},
	{Vins, AMOVL, O_l_hi, O_t_hi},
	{Vinsl, ASHLL, O_r, O_t_rp},
	{Vshll, O_r, O_t_lo},
	{Vins, AMOVL, O_t_hi, O_l_hi},
	{Vins, AMOVL, O_t_lo, O_l_lo},
	{Vend},
};

/* shift asop dest left, const == 32 */
static uchar	asdshllc32[][VLEN] =
{
	{Vins, AMOVL, O_l_lo, O_t_hi},
	{Vzero, O_t_lo},
	{Vins, AMOVL, O_t_hi, O_l_hi},
	{Vins, AMOVL, O_t_lo, O_l_lo},
	{Vend},
};

/* shift asop dest, const > 32 */
static uchar	asdshllchi[][VLEN] =
{
	{Vins, AMOVL, O_l_lo, O_t_hi},
	{Vzero, O_t_lo},
	{Vshll, O_r, O_t_hi},
	{Vins, AMOVL, O_t_lo, O_l_lo},
	{Vins, AMOVL, O_t_hi, O_l_hi},
	{Vend},
};

static uchar	(*asshlltab[])[VLEN] =
{
	asshllgen,
	asshllclo,
	asshllc32,
	asshllchi,
	asdshllgen,
	asdshllclo,
	asdshllc32,
	asdshllchi,
};

/* shift asop right general case */
static uchar	asshrlgen[][VLEN] =
{
	{V_a0, V_a1},
	{Vop, OGE, O_r, C32},
	{V_s0, Vins, AMOVL, O_l_lo, O_r0},
	{Vins, AMOVL, O_l_hi, O_r1},
	{Vinsra, ASHRL, O_r, O_r0},
	{Vinsx, Bop0, O_r, O_r1},
	{Vins, AMOVL, O_r0, O_l_lo},
	{Vins, AMOVL, O_r1, O_l_hi, Vgo},
	{V_p0, V_s0},
	{Vins, AMOVL, O_l_hi, O_r0},
	{Vinsx, Bop0, O_r, O_r0},
	{V_T1, Vzero, O_l_hi},
	{Vins, AMOVL, O_r0, O_l_lo},
	{V_F1, Vins, ASARL, C31, O_r0},
	{V_F1, Vins, AMOVL, O_r0, O_l_hi},
	{V_p0, V_f0, V_f1, Vend},
};

/* shift asop right, const < 32 */
static uchar	asshrlclo[][VLEN] =
{
	{V_a0, V_a1},
	{Vins, AMOVL, O_l_lo, O_r0},
	{Vins, AMOVL, O_l_hi, O_r1},
	{Vinsra, ASHRL, O_r, O_r0},
	{Vinsx, Bop0, O_r, O_r1},
	{Vins, AMOVL, O_r0, O_l_lo},
	{Vins, AMOVL, O_r1, O_l_hi},
	{V_f0, V_f1, Vend},
};

/* shift asop right, const == 32 */
static uchar	asshrlc32[][VLEN] =
{
	{V_a0},
	{Vins, AMOVL, O_l_hi, O_r0},
	{V_T1, Vzero, O_l_hi},
	{Vins, AMOVL, O_r0, O_l_lo},
	{V_F1, Vins, ASARL, C31, O_r0},
	{V_F1, Vins, AMOVL, O_r0, O_l_hi},
	{V_f0, Vend},
};

/* shift asop right, const > 32 */
static uchar	asshrlchi[][VLEN] =
{
	{V_a0},
	{Vins, AMOVL, O_l_hi, O_r0},
	{V_T1, Vzero, O_l_hi},
	{Vinsx, Bop0, O_r, O_r0},
	{Vins, AMOVL, O_r0, O_l_lo},
	{V_F1, Vins, ASARL, C31, O_r0},
	{V_F1, Vins, AMOVL, O_r0, O_l_hi},
	{V_f0, Vend},
};

/* shift asop dest right general case */
static uchar	asdshrlgen[][VLEN] =
{
	{Vop, OGE, O_r, C32},
	{V_s0, Vins, AMOVL, O_l_lo, O_t_lo},
	{Vins, AMOVL, O_l_hi, O_t_hi},
	{Vinsr, ASHRL, O_r, O_t_rp},
	{Vinsx, Bop0, O_r, O_t_hi},
	{Vins, AMOVL, O_t_lo, O_l_lo},
	{Vins, AMOVL, O_t_hi, O_l_hi, Vgo},
	{V_p0, V_s0},
	{Vins, AMOVL, O_l_hi, O_t_lo},
	{V_T1, Vzero, O_t_hi},
	{Vinsx, Bop0, O_r, O_t_lo},
	{V_F1, Vins, AMOVL, O_t_lo, O_t_hi},
	{V_F1, Vins, ASARL, C31, O_t_hi},
	{Vins, AMOVL, O_t_hi, O_l_hi, V_p0},
	{Vend},
};

/* shift asop dest right, const < 32 */
static uchar	asdshrlclo[][VLEN] =
{
	{Vins, AMOVL, O_l_lo, O_t_lo},
	{Vins, AMOVL, O_l_hi, O_t_hi},
	{Vinsr, ASHRL, O_r, O_t_rp},
	{Vinsx, Bop0, O_r, O_t_hi},
	{Vins, AMOVL, O_t_lo, O_l_lo},
	{Vins, AMOVL, O_t_hi, O_l_hi},
	{Vend},
};

/* shift asop dest right, const == 32 */
static uchar	asdshrlc32[][VLEN] =
{
	{Vins, AMOVL, O_l_hi, O_t_lo},
	{V_T1, Vzero, O_t_hi},
	{V_F1, Vins, AMOVL, O_t_lo, O_t_hi},
	{V_F1, Vins, ASARL, C31, O_t_hi},
	{Vins, AMOVL, O_t_lo, O_l_lo},
	{Vins, AMOVL, O_t_hi, O_l_hi},
	{Vend},
};

/* shift asop dest, const > 32 */
static uchar	asdshrlchi[][VLEN] =
{
	{Vins, AMOVL, O_l_hi, O_t_lo},
	{V_T1, Vzero, O_t_hi},
	{Vinsx, Bop0, O_r, O_t_lo},
	{V_T1, Vins, AMOVL, O_t_hi, O_l_hi},
	{V_T1, Vins, AMOVL, O_t_lo, O_l_lo},
	{V_F1, Vins, AMOVL, O_t_lo, O_t_hi},
	{V_F1, Vins, ASARL, C31, O_t_hi},
	{V_F1, Vins, AMOVL, O_t_lo, O_l_lo},
	{V_F1, Vins, AMOVL, O_t_hi, O_l_hi},
	{Vend},
};

static uchar	(*asshrltab[])[VLEN] =
{
	asshrlgen,
	asshrlclo,
	asshrlc32,
	asshrlchi,
	asdshrlgen,
	asdshrlclo,
	asdshrlc32,
	asdshrlchi,
};

static uchar	shrlargs[]	= { ASHRL, 1 };
static uchar	sarlargs[]	= { ASARL, 0 };

/* ++ -- */
static uchar	incdec[][VLEN] =
{
	{Vinsx, Bop0, C01, O_l_lo},
	{Vinsx, Bop1, C00, O_l_hi, Vend},
};

/* ++ -- *p */
static uchar	incdecpre[][VLEN] =
{
	{Vins, AMOVL, O_l_lo, O_t_lo},
	{Vins, AMOVL, O_l_hi, O_t_hi},
	{Vinsx, Bop0, C01, O_t_lo},
	{Vinsx, Bop1, C00, O_t_hi},
	{Vins, AMOVL, O_t_lo, O_l_lo},
	{Vins, AMOVL, O_t_hi, O_l_hi, Vend},
};

/* *p ++ -- */
static uchar	incdecpost[][VLEN] =
{
	{Vins, AMOVL, O_l_lo, O_t_lo},
	{Vins, AMOVL, O_l_hi, O_t_hi},
	{Vinsx, Bop0, C01, O_l_lo},
	{Vinsx, Bop1, C00, O_l_hi, Vend},
};

/* binop rp, rp */
static uchar	binop00[][VLEN] =
{
	{Vinsx, Bop0, O_r_lo, O_l_lo},
	{Vinsx, Bop1, O_r_hi, O_l_hi, Vend},
	{Vend},
};

/* binop rp, addr */
static uchar	binoptmp[][VLEN] =
{
	{V_a0, Vins, AMOVL, O_r_lo, O_r0},
	{Vinsx, Bop0, O_r0, O_l_lo},
	{Vins, AMOVL, O_r_hi, O_r0},
	{Vinsx, Bop1, O_r0, O_l_hi},
	{V_f0, Vend},
};

/* binop t = *a op *b */
static uchar	binop11[][VLEN] =
{
	{Vins, AMOVL, O_l_lo, O_t_lo},
	{Vinsx, Bop0, O_r_lo, O_t_lo},
	{Vins, AMOVL, O_l_hi, O_t_hi},
	{Vinsx, Bop1, O_r_hi, O_t_hi, Vend},
};

/* binop t = rp +- c */
static uchar	add0c[][VLEN] =
{
	{V_r_lo_t, Vinsx, Bop0, O_r_lo, O_l_lo},
	{V_r_lo_f, Vamv, Bop0, Bop1},
	{Vinsx, Bop1, O_r_hi, O_l_hi},
	{Vend},
};

/* binop t = rp & c */
static uchar	and0c[][VLEN] =
{
	{V_r_lo_t, Vinsx, Bop0, O_r_lo, O_l_lo},
	{V_r_lo_f, Vins, AMOVL, C00, O_l_lo},
	{V_r_hi_t, Vinsx, Bop1, O_r_hi, O_l_hi},
	{V_r_hi_f, Vins, AMOVL, C00, O_l_hi},
	{Vend},
};

/* binop t = rp | c */
static uchar	or0c[][VLEN] =
{
	{V_r_lo_t, Vinsx, Bop0, O_r_lo, O_l_lo},
	{V_r_hi_t, Vinsx, Bop1, O_r_hi, O_l_hi},
	{Vend},
};

/* binop t = c - rp */
static uchar	sub10[][VLEN] =
{
	{V_a0, Vins, AMOVL, O_l_lo, O_r0},
	{Vinsx, Bop0, O_r_lo, O_r0},
	{Vins, AMOVL, O_l_hi, O_r_lo},
	{Vinsx, Bop1, O_r_hi, O_r_lo},
	{Vspazz, V_f0, Vend},
};

/* binop t = c + *b */
static uchar	addca[][VLEN] =
{
	{Vins, AMOVL, O_r_lo, O_t_lo},
	{V_l_lo_t, Vinsx, Bop0, O_l_lo, O_t_lo},
	{V_l_lo_f, Vamv, Bop0, Bop1},
	{Vins, AMOVL, O_r_hi, O_t_hi},
	{Vinsx, Bop1, O_l_hi, O_t_hi},
	{Vend},
};

/* binop t = c & *b */
static uchar	andca[][VLEN] =
{
	{V_l_lo_t, Vins, AMOVL, O_r_lo, O_t_lo},
	{V_l_lo_t, Vinsx, Bop0, O_l_lo, O_t_lo},
	{V_l_lo_f, Vzero, O_t_lo},
	{V_l_hi_t, Vins, AMOVL, O_r_hi, O_t_hi},
	{V_l_hi_t, Vinsx, Bop1, O_l_hi, O_t_hi},
	{V_l_hi_f, Vzero, O_t_hi},
	{Vend},
};

/* binop t = c | *b */
static uchar	orca[][VLEN] =
{
	{Vins, AMOVL, O_r_lo, O_t_lo},
	{V_l_lo_t, Vinsx, Bop0, O_l_lo, O_t_lo},
	{Vins, AMOVL, O_r_hi, O_t_hi},
	{V_l_hi_t, Vinsx, Bop1, O_l_hi, O_t_hi},
	{Vend},
};

/* binop t = c - *b */
static uchar	subca[][VLEN] =
{
	{Vins, AMOVL, O_l_lo, O_t_lo},
	{Vins, AMOVL, O_l_hi, O_t_hi},
	{Vinsx, Bop0, O_r_lo, O_t_lo},
	{Vinsx, Bop1, O_r_hi, O_t_hi},
	{Vend},
};

/* binop t = *a +- c */
static uchar	addac[][VLEN] =
{
	{Vins, AMOVL, O_l_lo, O_t_lo},
	{V_r_lo_t, Vinsx, Bop0, O_r_lo, O_t_lo},
	{V_r_lo_f, Vamv, Bop0, Bop1},
	{Vins, AMOVL, O_l_hi, O_t_hi},
	{Vinsx, Bop1, O_r_hi, O_t_hi},
	{Vend},
};

/* binop t = *a | c */
static uchar	orac[][VLEN] =
{
	{Vins, AMOVL, O_l_lo, O_t_lo},
	{V_r_lo_t, Vinsx, Bop0, O_r_lo, O_t_lo},
	{Vins, AMOVL, O_l_hi, O_t_hi},
	{V_r_hi_t, Vinsx, Bop1, O_r_hi, O_t_hi},
	{Vend},
};

/* binop t = *a & c */
static uchar	andac[][VLEN] =
{
	{V_r_lo_t, Vins, AMOVL, O_l_lo, O_t_lo},
	{V_r_lo_t, Vinsx, Bop0, O_r_lo, O_t_lo},
	{V_r_lo_f, Vzero, O_t_lo},
	{V_r_hi_t, Vins, AMOVL, O_l_hi, O_t_hi},
	{V_r_hi_t, Vinsx, Bop0, O_r_hi, O_t_hi},
	{V_r_hi_f, Vzero, O_t_hi},
	{Vend},
};

static uchar	ADDargs[]	= { AADDL, AADCL };
static uchar	ANDargs[]	= { AANDL, AANDL };
static uchar	ORargs[]	= { AORL, AORL };
static uchar	SUBargs[]	= { ASUBL, ASBBL };
static uchar	XORargs[]	= { AXORL, AXORL };

static uchar	(*ADDtab[])[VLEN] =
{
	add0c, addca, addac,
};

static uchar	(*ANDtab[])[VLEN] =
{
	and0c, andca, andac,
};

static uchar	(*ORtab[])[VLEN] =
{
	or0c, orca, orac,
};

static uchar	(*SUBtab[])[VLEN] =
{
	add0c, subca, addac,
};

/* mul of const32 */
static uchar	mulc32[][VLEN] =
{
	{V_a0, Vop, ONE, O_l_hi, C00},
	{V_s0, Vins, AMOVL, O_r_lo, O_r0},
	{Vins, AMULL, O_r0, O_Zop},
	{Vgo, V_p0, V_s0},
	{Vins, AMOVL, O_l_hi, O_r0},
	{Vmul, O_r_lo, O_r0},
	{Vins, AMOVL, O_r_lo, O_l_hi},
	{Vins, AMULL, O_l_hi, O_Zop},
	{Vins, AADDL, O_r0, O_l_hi},
	{V_f0, V_p0, Vend},
};

/* mul of const64 */
static uchar	mulc64[][VLEN] =
{
	{V_a0, Vins, AMOVL, O_r_hi, O_r0},
	{Vop, OOR, O_l_hi, O_r0},
	{Vop, ONE, O_r0, C00},
	{V_s0, Vins, AMOVL, O_r_lo, O_r0},
	{Vins, AMULL, O_r0, O_Zop},
	{Vgo, V_p0, V_s0},
	{Vmul, O_r_lo, O_l_hi},
	{Vins, AMOVL, O_l_lo, O_r0},
	{Vmul, O_r_hi, O_r0},
	{Vins, AADDL, O_l_hi, O_r0},
	{Vins, AMOVL, O_r_lo, O_l_hi},
	{Vins, AMULL, O_l_hi, O_Zop},
	{Vins, AADDL, O_r0, O_l_hi},
	{V_f0, V_p0, Vend},
};

/* mul general */
static uchar	mull[][VLEN] =
{
	{V_a0, Vins, AMOVL, O_r_hi, O_r0},
	{Vop, OOR, O_l_hi, O_r0},
	{Vop, ONE, O_r0, C00},
	{V_s0, Vins, AMOVL, O_r_lo, O_r0},
	{Vins, AMULL, O_r0, O_Zop},
	{Vgo, V_p0, V_s0},
	{Vins, AIMULL, O_r_lo, O_l_hi},
	{Vins, AMOVL, O_l_lo, O_r0},
	{Vins, AIMULL, O_r_hi, O_r0},
	{Vins, AADDL, O_l_hi, O_r0},
	{Vins, AMOVL, O_r_lo, O_l_hi},
	{Vins, AMULL, O_l_hi, O_Zop},
	{Vins, AADDL, O_r0, O_l_hi},
	{V_f0, V_p0, Vend},
};

/* cast rp l to rp t */
static uchar	castrp[][VLEN] =
{
	{Vmv, O_l, O_t_lo},
	{VT, Vins, AMOVL, O_t_lo, O_t_hi},
	{VT, Vins, ASARL, C31, O_t_hi},
	{VF, Vzero, O_t_hi},
	{Vend},
};

/* cast rp l to addr t */
static uchar	castrpa[][VLEN] =
{
	{VT, V_a0, Vmv, O_l, O_r0},
	{VT, Vins, AMOVL, O_r0, O_t_lo},
	{VT, Vins, ASARL, C31, O_r0},
	{VT, Vins, AMOVL, O_r0, O_t_hi},
	{VT, V_f0},
	{VF, Vmv, O_l, O_t_lo},
	{VF, Vzero, O_t_hi},
	{Vend},
};

static uchar	netab0i[][VLEN] =
{
	{Vop, ONE, O_l_lo, O_r_lo},
	{V_s0, Vop, ONE, O_l_hi, O_r_hi},
	{V_s1, Vgo, V_s2, Vgo, V_s3},
	{VF, V_p0, V_p1, VT, V_p2},
	{Vgo, V_p3},
	{VT, V_p0, V_p1, VF, V_p2},
	{Vend},
};

static uchar	netabii[][VLEN] =
{
	{V_a0, Vins, AMOVL, O_l_lo, O_r0},
	{Vop, ONE, O_r0, O_r_lo},
	{V_s0, Vins, AMOVL, O_l_hi, O_r0},
	{Vop, ONE, O_r0, O_r_hi},
	{V_s1, Vgo, V_s2, Vgo, V_s3},
	{VF, V_p0, V_p1, VT, V_p2},
	{Vgo, V_p3},
	{VT, V_p0, V_p1, VF, V_p2},
	{V_f0, Vend},
};

static uchar	cmptab0i[][VLEN] =
{
	{Vopx, Bop0, O_l_hi, O_r_hi},
	{V_s0, Vins0, AJNE},
	{V_s1, Vopx, Bop1, O_l_lo, O_r_lo},
	{V_s2, Vgo, V_s3, Vgo, V_s4},
	{VT, V_p1, V_p3},
	{VF, V_p0, V_p2},
	{Vgo, V_p4},
	{VT, V_p0, V_p2},
	{VF, V_p1, V_p3},
	{Vend},
};

static uchar	cmptabii[][VLEN] =
{
	{V_a0, Vins, AMOVL, O_l_hi, O_r0},
	{Vopx, Bop0, O_r0, O_r_hi},
	{V_s0, Vins0, AJNE},
	{V_s1, Vins, AMOVL, O_l_lo, O_r0},
	{Vopx, Bop1, O_r0, O_r_lo},
	{V_s2, Vgo, V_s3, Vgo, V_s4},
	{VT, V_p1, V_p3},
	{VF, V_p0, V_p2},
	{Vgo, V_p4},
	{VT, V_p0, V_p2},
	{VF, V_p1, V_p3},
	{V_f0, Vend},
};

static uchar	(*NEtab[])[VLEN] =
{
	netab0i, netabii,
};

static uchar	(*cmptab[])[VLEN] =
{
	cmptab0i, cmptabii,
};

static uchar	GEargs[]	= { OGT, OHS };
static uchar	GTargs[]	= { OGT, OHI };
static uchar	HIargs[]	= { OHI, OHI };
static uchar	HSargs[]	= { OHI, OHS };

/* Big Generator */
static void
biggen(Node *l, Node *r, Node *t, int true, uchar code[][VLEN], uchar *a)
{
	int i, j, g, oc, op, lo, ro, to, xo, *xp;
	Type *lt;
	Prog *pr[VOPS];
	Node *ot, *tl, *tr, tmps[2];
	uchar *c, (*cp)[VLEN], args[VARGS];

	if(a != nil)
		memmove(args, a, VARGS);
//print("biggen %d %d %d\n", args[0], args[1], args[2]);
//if(l) prtree(l, "l");
//if(r) prtree(r, "r");
//if(t) prtree(t, "t");
	lo = ro = to = 0;
	cp = code;

	for (;;) {
		c = *cp++;
		g = 1;
		i = 0;
//print("code %d %d %d %d %d\n", c[0], c[1], c[2], c[3], c[4]);
		for(;;) {
			switch(op = c[i]) {
			case Vgo:
				if(g)
					gbranch(OGOTO);
				i++;
				break;

			case Vamv:
				i += 3;
				if(i > VLEN) {
					diag(l, "bad Vop");
					return;
				}
				if(g)
					args[c[i - 1]] = args[c[i - 2]];
				break;

			case Vzero:
				i += 2;
				if(i > VLEN) {
					diag(l, "bad Vop");
					return;
				}
				j = i - 1;
				goto op;

			case Vspazz:	// nasty hack to save a reg in SUB
//print("spazz\n");
				if(g) {
//print("hi %R lo %R t %R\n", r->right->reg, r->left->reg, tmps[0].reg);
					ot = r->right;
					r->right = r->left;
					tl = new(0, Z, Z);
					*tl = tmps[0];
					r->left = tl;
					tmps[0] = *ot;
//print("hi %R lo %R t %R\n", r->right->reg, r->left->reg, tmps[0].reg);
				}
				i++;
				break;

			case Vmv:
			case Vmul:
			case Vshll:
				i += 3;
				if(i > VLEN) {
					diag(l, "bad Vop");
					return;
				}
				j = i - 2;
				goto op;

			case Vins0:
				i += 2;
				if(i > VLEN) {
					diag(l, "bad Vop");
					return;
				}
				gins(c[i - 1], Z, Z);
				break;

			case Vop:
			case Vopx:
			case Vins:
			case Vinsl:
			case Vinsr:
			case Vinsla:
			case Vinsra:
			case Vinsx:
				i += 4;
				if(i > VLEN) {
					diag(l, "bad Vop");
					return;
				}
				j = i - 2;
				goto op;

			op:
				if(!g)
					break;
				tl = Z;
				tr = Z;
				for(; j < i; j++) {
					switch(c[j]) {
					case C00:
						ot = nodconst(0);
						break;
					case C01:
						ot = nodconst(1);
						break;
					case C31:
						ot = nodconst(31);
						break;
					case C32:
						ot = nodconst(32);
						break;

					case O_l:
					case O_l_lo:
						ot = l; xp = &lo; xo = 0;
						goto op0;
					case O_l_hi:
						ot = l; xp = &lo; xo = SZ_LONG;
						goto op0;
					case O_r:
					case O_r_lo:
						ot = r; xp = &ro; xo = 0;
						goto op0;
					case O_r_hi:
						ot = r; xp = &ro; xo = SZ_LONG;
						goto op0;
					case O_t_lo:
						ot = t; xp = &to; xo = 0;
						goto op0;
					case O_t_hi:
						ot = t; xp = &to; xo = SZ_LONG;
						goto op0;
					case O_l_rp:
						ot = l;
						break;
					case O_r_rp:
						ot = r;
						break;
					case O_t_rp:
						ot = t;
						break;
					case O_r0:
					case O_r1:
						ot = &tmps[c[j] - O_r0];
						break;
					case O_Zop:
						ot = Z;
						break;

					op0:
						switch(ot->op) {
						case OCONST:
							if(xo)
								ot = hi64(ot);
							else
								ot = lo64(ot);
							break;
						case OREGPAIR:
							if(xo)
								ot = ot->right;
							else
								ot = ot->left;
							break;
						case OREGISTER:
							break;
						default:
							if(xo != *xp) {
								ot->xoffset += xo - *xp;
								*xp = xo;
							}
						}
						break;
					
					default:
						diag(l, "bad V_lop");
						return;
					}
					if(tl == nil)
						tl = ot;
					else
						tr = ot;
				}
				if(op == Vzero) {
					zeroregm(tl);
					break;
				}
				oc = c[i - 3];
				if(op == Vinsx || op == Vopx) {
//print("%d -> %d\n", oc, args[oc]);
					oc = args[oc];
				}
				else {
					switch(oc) {
					case O_a0:
					case O_a1:
						oc = args[oc - O_a0];
						break;
					}
				}
				switch(op) {
				case Vmul:
					mulgen(tr->type, tl, tr);
					break;
				case Vmv:
					gmove(tl, tr);
					break;
				case Vshll:
					shiftit(tr->type, tl, tr);
					break;
				case Vop:
				case Vopx:
					gopcode(oc, types[TULONG], tl, tr);
					break;
				case Vins:
				case Vinsx:
					gins(oc, tl, tr);
					break;
				case Vinsl:
					gins(oc, tl, tr->right);
					p->from.index = tr->left->reg;
					break;
				case Vinsr:
					gins(oc, tl, tr->left);
					p->from.index = tr->right->reg;
					break;
				case Vinsla:
					gins(oc, tl, tr + 1);
					p->from.index = tr->reg;
					break;
				case Vinsra:
					gins(oc, tl, tr);
					p->from.index = (tr + 1)->reg;
					break;
				}
				break;

			case VT:
				g = true;
				i++;
				break;
			case VF:
				g = !true;
				i++;
				break;

			case V_T0: case V_T1:
				g = args[op - V_T0];
				i++;
				break;

			case V_F0: case V_F1:
				g = !args[op - V_F0];
				i++;
				break;

			case V_C0: case V_C1:
				if(g)
					args[op - V_C0] = 0;
				i++;
				break;

			case V_S0: case V_S1:
				if(g)
					args[op - V_S0] = 1;
				i++;
				break;

			case V_l_lo_f:
				g = lo64v(l) == 0;
				i++;
				break;
			case V_l_hi_f:
				g = hi64v(l) == 0;
				i++;
				break;
			case V_l_lo_t:
				g = lo64v(l) != 0;
				i++;
				break;
			case V_l_hi_t:
				g = hi64v(l) != 0;
				i++;
				break;
			case V_l_lo_u:
				g = lo64v(l) >= 0;
				i++;
				break;
			case V_l_hi_u:
				g = hi64v(l) >= 0;
				i++;
				break;
			case V_r_lo_f:
				g = lo64v(r) == 0;
				i++;
				break;
			case V_r_hi_f:
				g = hi64v(r) == 0;
				i++;
				break;
			case V_r_lo_t:
				g = lo64v(r) != 0;
				i++;
				break;
			case V_r_hi_t:
				g = hi64v(r) != 0;
				i++;
				break;
			case V_r_lo_u:
				g = lo64v(r) >= 0;
				i++;
				break;
			case V_r_hi_u:
				g = hi64v(r) >= 0;
				i++;
				break;

			case Vend:
				goto out;

			case V_a0: case V_a1:
				if(g) {
					lt = l->type;
					l->type = types[TULONG];
					regalloc(&tmps[op - V_a0], l, Z);
					l->type = lt;
				}
				i++;
				break;

			case V_f0: case V_f1:
				if(g)
					regfree(&tmps[op - V_f0]);
				i++;
				break;

			case V_p0: case V_p1: case V_p2: case V_p3: case V_p4:
				if(g)
					patch(pr[op - V_p0], pc);
				i++;
				break;

			case V_s0: case V_s1: case V_s2: case V_s3: case V_s4:
				if(g)
					pr[op - V_s0] = p;
				i++;
				break;

			default:
				diag(l, "bad biggen: %d", op);
				return;
			}
			if(i == VLEN || c[i] == 0)
				break;
		}
	}
out:
	if(lo)
		l->xoffset -= lo;
	if(ro)
		r->xoffset -= ro;
	if(to)
		t->xoffset -= to;
}

int
cgen64(Node *n, Node *nn)
{
	Type *dt;
	uchar *args, (*cp)[VLEN], (**optab)[VLEN];
	int li, ri, lri, dr, si, m, op, sh, cmp, true;
	Node *c, *d, *l, *r, *t, *s, nod1, nod2, nod3, nod4, nod5;

	if(debug['g']) {
		prtree(nn, "cgen64 lhs");
		prtree(n, "cgen64");
		print("AX = %d\n", reg[D_AX]);
	}
	cmp = 0;
	sh = 0;

	switch(n->op) {
	case ONEG:
		d = regpair(nn, n);
		sugen(n->left, d, 8);
		gins(ANOTL, Z, d->right);
		gins(ANEGL, Z, d->left);
		gins(ASBBL, nodconst(-1), d->right);
		break;

	case OCOM:
		if(!vaddr(n->left, 0) || !vaddr(nn, 0))
			d = regpair(nn, n);
		else
			return 0;
		sugen(n->left, d, 8);
		gins(ANOTL, Z, d->left);
		gins(ANOTL, Z, d->right);
		break;

	case OADD:
		optab = ADDtab;
		args = ADDargs;
		goto twoop;
	case OAND:
		optab = ANDtab;
		args = ANDargs;
		goto twoop;
	case OOR:
		optab = ORtab;
		args = ORargs;
		goto twoop;
	case OSUB:
		optab = SUBtab;
		args = SUBargs;
		goto twoop;
	case OXOR:
		optab = ORtab;
		args = XORargs;
		goto twoop;
	case OASHL:
		sh = 1;
		args = nil;
		optab = shlltab;
		goto twoop;
	case OLSHR:
		sh = 1;
		args = shrlargs;
		optab = shrltab;
		goto twoop;
	case OASHR:
		sh = 1;
		args = sarlargs;
		optab = shrltab;
		goto twoop;
	case OEQ:
		cmp = 1;
		args = nil;
		optab = nil;
		goto twoop;
	case ONE:
		cmp = 1;
		args = nil;
		optab = nil;
		goto twoop;
	case OLE:
		cmp = 1;
		args = nil;
		optab = nil;
		goto twoop;
	case OLT:
		cmp = 1;
		args = nil;
		optab = nil;
		goto twoop;
	case OGE:
		cmp = 1;
		args = nil;
		optab = nil;
		goto twoop;
	case OGT:
		cmp = 1;
		args = nil;
		optab = nil;
		goto twoop;
	case OHI:
		cmp = 1;
		args = nil;
		optab = nil;
		goto twoop;
	case OHS:
		cmp = 1;
		args = nil;
		optab = nil;
		goto twoop;
	case OLO:
		cmp = 1;
		args = nil;
		optab = nil;
		goto twoop;
	case OLS:
		cmp = 1;
		args = nil;
		optab = nil;
		goto twoop;

twoop:
		dr = nn != Z && nn->op == OREGPAIR;
		l = vfunc(n->left, nn);
		if(sh)
			r = n->right;
		else
			r = vfunc(n->right, nn);

		li = l->op == ONAME || l->op == OINDREG || l->op == OCONST;
		ri = r->op == ONAME || r->op == OINDREG || r->op == OCONST;

#define	IMM(l, r)	((l) | ((r) << 1))

		lri = IMM(li, ri);

		/* find out what is so easy about some operands */
		if(li)
			li = whatof(l, sh | cmp);
		if(ri)
			ri = whatof(r, cmp);

		if(sh)
			goto shift;

		if(cmp)
			goto cmp;

		/* evaluate hard subexps, stealing nn if possible. */
		switch(lri) {
		case IMM(0, 0):
		bin00:
			if(l->complex > r->complex) {
				if(dr)
					t = nn;
				else
					t = regpair(Z, n);
				sugen(l, t, 8);
				l = t;
				t = regpair(Z, n);
				sugen(r, t, 8);
				r = t;
			}
			else {
				t = regpair(Z, n);
				sugen(r, t, 8);
				r = t;
				if(dr)
					t = nn;
				else
					t = regpair(Z, n);
				sugen(l, t, 8);
				l = t;
			}
			break;
		case IMM(0, 1):
			if(dr)
				t = nn;
			else
				t = regpair(Z, n);
			sugen(l, t, 8);
			l = t;
			break;
		case IMM(1, 0):
			if(n->op == OSUB && l->op == OCONST && hi64v(l) == 0) {
				lri = IMM(0, 0);
				goto bin00;
			}
			if(dr)
				t = nn;
			else
				t = regpair(Z, n);
			sugen(r, t, 8);
			r = t;
			break;
		case IMM(1, 1):
			break;
		}

#define	WW(l, r)	((l) | ((r) << 2))
		d = Z;
		dt = nn->type;
		nn->type = types[TLONG];

		switch(lri) {
		case IMM(0, 0):
			biggen(l, r, Z, 0, binop00, args);
			break;
		case IMM(0, 1):
			switch(ri) {
			case WNONE:
				diag(r, "bad whatof\n");
				break;
			case WCONST:
				biggen(l, r, Z, 0, optab[B0c], args);
				break;
			case WHARD:
				reglcgen(&nod2, r, Z);
				r = &nod2;
				/* fall thru */
			case WADDR:
				biggen(l, r, Z, 0, binoptmp, args);
				if(ri == WHARD)
					regfree(r);
				break;
			}
			break;
		case IMM(1, 0):
			if(n->op == OSUB) {
				switch(li) {
				case WNONE:
					diag(l, "bad whatof\n");
					break;
				case WHARD:
					reglcgen(&nod2, l, Z);
					l = &nod2;
					/* fall thru */
				case WADDR:
				case WCONST:
					biggen(l, r, Z, 0, sub10, args);
					break;
				}
				if(li == WHARD)
					regfree(l);
			}
			else {
				switch(li) {
				case WNONE:
					diag(l, "bad whatof\n");
					break;
				case WCONST:
					biggen(r, l, Z, 0, optab[B0c], args);
					break;
				case WHARD:
					reglcgen(&nod2, l, Z);
					l = &nod2;
					/* fall thru */
				case WADDR:
					biggen(r, l, Z, 0, binoptmp, args);
					if(li == WHARD)
						regfree(l);
					break;
				}
			}
			break;
		case IMM(1, 1):
			switch(WW(li, ri)) {
			case WW(WCONST, WHARD):
				if(r->op == ONAME && n->op == OAND && reduxv(l))
					ri = WADDR;
				break;
			case WW(WHARD, WCONST):
				if(l->op == ONAME && n->op == OAND && reduxv(r))
					li = WADDR;
				break;
			}
			if(li == WHARD) {
				reglcgen(&nod3, l, Z);
				l = &nod3;
			}
			if(ri == WHARD) {
				reglcgen(&nod2, r, Z);
				r = &nod2;
			}
			d = regpair(nn, n);
			instpair(d, Z);
			switch(WW(li, ri)) {
			case WW(WCONST, WADDR):
			case WW(WCONST, WHARD):
				biggen(l, r, d, 0, optab[Bca], args);
				break;

			case WW(WADDR, WCONST):
			case WW(WHARD, WCONST):
				biggen(l, r, d, 0, optab[Bac], args);
				break;

			case WW(WADDR, WADDR):
			case WW(WADDR, WHARD):
			case WW(WHARD, WADDR):
			case WW(WHARD, WHARD):
				biggen(l, r, d, 0, binop11, args);
				break;

			default:
				diag(r, "bad whatof pair %d %d\n", li, ri);
				break;
			}
			if(li == WHARD)
				regfree(l);
			if(ri == WHARD)
				regfree(r);
			break;
		}

		nn->type = dt;

		if(d != Z)
			goto finished;

		switch(lri) {
		case IMM(0, 0):
			freepair(r);
			/* fall thru */;
		case IMM(0, 1):
			if(!dr)
				storepair(l, nn, 1);
			break;
		case IMM(1, 0):
			if(!dr)
				storepair(r, nn, 1);
			break;
		case IMM(1, 1):
			break;
		}
		return 1;

	shift:
		c = Z;

		/* evaluate hard subexps, stealing nn if possible. */
		/* must also secure CX.  not as many optims as binop. */
		switch(lri) {
		case IMM(0, 0):
		imm00:
			if(l->complex + 1 > r->complex) {
				if(dr)
					t = nn;
				else
					t = regpair(Z, l);
				sugen(l, t, 8);
				l = t;
				t = &nod1;
				c = snarfreg(l, t, D_CX, r, &nod2);
				cgen(r, t);
				r = t;
			}
			else {
				t = &nod1;
				c = snarfreg(nn, t, D_CX, r, &nod2);
				cgen(r, t);
				r = t;
				if(dr)
					t = nn;
				else
					t = regpair(Z, l);
				sugen(l, t, 8);
				l = t;
			}
			break;
		case IMM(0, 1):
		imm01:
			if(ri != WCONST) {
				lri = IMM(0, 0);
				goto imm00;
			}
			if(dr)
				t = nn;
			else
				t = regpair(Z, n);
			sugen(l, t, 8);
			l = t;
			break;
		case IMM(1, 0):
		imm10:
			if(li != WCONST) {
				lri = IMM(0, 0);
				goto imm00;
			}
			t = &nod1;
			c = snarfreg(nn, t, D_CX, r, &nod2);
			cgen(r, t);
			r = t;
			break;
		case IMM(1, 1):
			if(ri != WCONST) {
				lri = IMM(1, 0);
				goto imm10;
			}
			if(li == WHARD) {
				lri = IMM(0, 1);
				goto imm01;
			}
			break;
		}

		d = Z;

		switch(lri) {
		case IMM(0, 0):
			biggen(l, r, Z, 0, optab[S00], args);
			break;
		case IMM(0, 1):
			switch(ri) {
			case WNONE:
			case WADDR:
			case WHARD:
				diag(r, "bad whatof\n");
				break;
			case WCONST:
				m = r->vconst & 63;
				s = nodconst(m);
				if(m < 32)
					cp = optab[Sc0];
				else if(m == 32)
					cp = optab[Sc1];
				else
					cp = optab[Sc2];
				biggen(l, s, Z, 0, cp, args);
				break;
			}
			break;
		case IMM(1, 0):
			/* left is const */
			d = regpair(nn, n);
			instpair(d, Z);
			biggen(l, r, d, 0, optab[S10], args);
			regfree(r);
			break;
		case IMM(1, 1):
			d = regpair(nn, n);
			instpair(d, Z);
			switch(WW(li, ri)) {
			case WW(WADDR, WCONST):
				m = r->vconst & 63;
				s = nodconst(m);
				if(m < 32) {
					loadpair(l, d);
					l = d;
					cp = optab[Sc0];
				}
				else if(m == 32)
					cp = optab[Sac3];
				else
					cp = optab[Sac4];
				biggen(l, s, d, 0, cp, args);
				break;

			default:
				diag(r, "bad whatof pair %d %d\n", li, ri);
				break;
			}
			break;
		}

		if(c != Z) {
			gins(AMOVL, c, r);
			regfree(c);
		}

		if(d != Z)
			goto finished;

		switch(lri) {
		case IMM(0, 0):
			regfree(r);
			/* fall thru */
		case IMM(0, 1):
			if(!dr)
				storepair(l, nn, 1);
			break;
		case IMM(1, 0):
			regfree(r);
			break;
		case IMM(1, 1):
			break;
		}
		return 1;

	cmp:
		op = n->op;
		/* evaluate hard subexps */
		switch(lri) {
		case IMM(0, 0):
			if(l->complex > r->complex) {
				t = regpair(Z, l);
				sugen(l, t, 8);
				l = t;
				t = regpair(Z, r);
				sugen(r, t, 8);
				r = t;
			}
			else {
				t = regpair(Z, r);
				sugen(r, t, 8);
				r = t;
				t = regpair(Z, l);
				sugen(l, t, 8);
				l = t;
			}
			break;
		case IMM(1, 0):
			t = r;
			r = l;
			l = t;
			ri = li;
			op = invrel[relindex(op)];
			/* fall thru */
		case IMM(0, 1):
			t = regpair(Z, l);
			sugen(l, t, 8);
			l = t;
			break;
		case IMM(1, 1):
			break;
		}

		true = 1;
		optab = cmptab;
		switch(op) {
		case OEQ:
			optab = NEtab;
			true = 0;
			break;
		case ONE:
			optab = NEtab;
			break;
		case OLE:
			args = GTargs;
			true = 0;
			break;
		case OGT:
			args = GTargs;
			break;
		case OLS:
			args = HIargs;
			true = 0;
			break;
		case OHI:
			args = HIargs;
			break;
		case OLT:
			args = GEargs;
			true = 0;
			break;
		case OGE:
			args = GEargs;
			break;
		case OLO:
			args = HSargs;
			true = 0;
			break;
		case OHS:
			args = HSargs;
			break;
		default:
			diag(n, "bad cmp\n");
			SET(optab);
		}

		switch(lri) {
		case IMM(0, 0):
			biggen(l, r, Z, true, optab[T0i], args);
			break;
		case IMM(0, 1):
		case IMM(1, 0):
			switch(ri) {
			case WNONE:
				diag(l, "bad whatof\n");
				break;
			case WCONST:
				biggen(l, r, Z, true, optab[T0i], args);
				break;
			case WHARD:
				reglcgen(&nod2, r, Z);
				r = &nod2;
				/* fall thru */
			case WADDR:
				biggen(l, r, Z, true, optab[T0i], args);
				if(ri == WHARD)
					regfree(r);
				break;
			}
			break;
		case IMM(1, 1):
			if(li == WHARD) {
				reglcgen(&nod3, l, Z);
				l = &nod3;
			}
			if(ri == WHARD) {
				reglcgen(&nod2, r, Z);
				r = &nod2;
			}
			biggen(l, r, Z, true, optab[Tii], args);
			if(li == WHARD)
				regfree(l);
			if(ri == WHARD)
				regfree(r);
			break;
		}

		switch(lri) {
		case IMM(0, 0):
			freepair(r);
			/* fall thru */;
		case IMM(0, 1):
		case IMM(1, 0):
			freepair(l);
			break;
		case IMM(1, 1):
			break;
		}
		return 1;

	case OASMUL:
	case OASLMUL:
		m = 0;
		goto mulop;

	case OMUL:
	case OLMUL:
		m = 1;
		goto mulop;

	mulop:
		dr = nn != Z && nn->op == OREGPAIR;
		l = vfunc(n->left, nn);
		r = vfunc(n->right, nn);
		if(r->op != OCONST) {
			if(l->complex > r->complex) {
				if(m) {
					t = l;
					l = r;
					r = t;
				}
				else if(!vaddr(l, 1)) {
					reglcgen(&nod5, l, Z);
					l = &nod5;
					evacaxdx(l);
				}
			}
			t = regpair(Z, n);
			sugen(r, t, 8);
			r = t;
			evacaxdx(r->left);
			evacaxdx(r->right);
			if(l->complex <= r->complex && !m && !vaddr(l, 1)) {
				reglcgen(&nod5, l, Z);
				l = &nod5;
				evacaxdx(l);
			}
		}
		if(dr)
			t = nn;
		else
			t = regpair(Z, n);
		c = Z;
		d = Z;
		if(!nodreg(&nod1, t->left, D_AX)) {
			if(t->left->reg != D_AX){
				t->left->reg = D_AX;
				reg[D_AX]++;
			}else if(reg[D_AX] == 0)
				fatal(Z, "vlong mul AX botch");
		}
		if(!nodreg(&nod2, t->right, D_DX)) {
			if(t->right->reg != D_DX){
				t->right->reg = D_DX;
				reg[D_DX]++;
			}else if(reg[D_DX] == 0)
				fatal(Z, "vlong mul DX botch");
		}
		if(m)
			sugen(l, t, 8);
		else
			loadpair(l, t);
		if(t->left->reg != D_AX) {
			c = &nod3;
			regsalloc(c, t->left);
			gmove(&nod1, c);
			gmove(t->left, &nod1);
			zapreg(t->left);
		}
		if(t->right->reg != D_DX) {
			d = &nod4;
			regsalloc(d, t->right);
			gmove(&nod2, d);
			gmove(t->right, &nod2);
			zapreg(t->right);
		}
		if(c != Z || d != Z) {
			s = regpair(Z, n);
			s->left = &nod1;
			s->right = &nod2;
		}
		else
			s = t;
		if(r->op == OCONST) {
			if(hi64v(r) == 0)
				biggen(s, r, Z, 0, mulc32, nil);
			else
				biggen(s, r, Z, 0, mulc64, nil);
		}
		else
			biggen(s, r, Z, 0, mull, nil);
		instpair(t, Z);
		if(c != Z) {
			gmove(&nod1, t->left);
			gmove(&nod3, &nod1);
		}
		if(d != Z) {
			gmove(&nod2, t->right);
			gmove(&nod4, &nod2);
		}
		if(r->op == OREGPAIR)
			freepair(r);
		if(!m)
			storepair(t, l, 0);
		if(l == &nod5)
			regfree(l);
		if(!dr) {
			if(nn != Z)
				storepair(t, nn, 1);
			else
				freepair(t);
		}
		return 1;

	case OASADD:
		args = ADDargs;
		goto vasop;
	case OASAND:
		args = ANDargs;
		goto vasop;
	case OASOR:
		args = ORargs;
		goto vasop;
	case OASSUB:
		args = SUBargs;
		goto vasop;
	case OASXOR:
		args = XORargs;
		goto vasop;

	vasop:
		l = n->left;
		r = n->right;
		dr = nn != Z && nn->op == OREGPAIR;
		m = 0;
		if(l->complex > r->complex) {
			if(!vaddr(l, 1)) {
				reglcgen(&nod1, l, Z);
				l = &nod1;
			}
			if(!vaddr(r, 1) || nn != Z || r->op == OCONST) {
				if(dr)
					t = nn;
				else
					t = regpair(Z, r);
				sugen(r, t, 8);
				r = t;
				m = 1;
			}
		}
		else {
			if(!vaddr(r, 1) || nn != Z || r->op == OCONST) {
				if(dr)
					t = nn;
				else
					t = regpair(Z, r);
				sugen(r, t, 8);
				r = t;
				m = 1;
			}
			if(!vaddr(l, 1)) {
				reglcgen(&nod1, l, Z);
				l = &nod1;
			}
		}
		if(nn != Z) {
			if(n->op == OASSUB)
				biggen(l, r, Z, 0, sub10, args);
			else
				biggen(r, l, Z, 0, binoptmp, args);
			storepair(r, l, 0);
		}
		else {
			if(m)
				biggen(l, r, Z, 0, binop00, args);
			else
				biggen(l, r, Z, 0, binoptmp, args);
		}
		if(l == &nod1)
			regfree(&nod1);
		if(m) {
			if(nn == Z)
				freepair(r);
			else if(!dr)
				storepair(r, nn, 1);
		}
		return 1;

	case OASASHL:
		args = nil;
		optab = asshlltab;
		goto assh;
	case OASLSHR:
		args = shrlargs;
		optab = asshrltab;
		goto assh;
	case OASASHR:
		args = sarlargs;
		optab = asshrltab;
		goto assh;

	assh:
		c = Z;
		l = n->left;
		r = n->right;
		if(r->op == OCONST) {
			m = r->vconst & 63;
			if(m < 32)
				m = SAclo;
			else if(m == 32)
				m = SAc32;
			else
				m = SAchi;
		}
		else
			m = SAgen;
		if(l->complex > r->complex) {
			if(!vaddr(l, 0)) {
				reglcgen(&nod1, l, Z);
				l = &nod1;
			}
			if(m == SAgen) {
				t = &nod2;
				if(l->reg == D_CX) {
					regalloc(t, r, Z);
					gmove(l, t);
					l->reg = t->reg;
					t->reg = D_CX;
				}
				else
					c = snarfreg(nn, t, D_CX, r, &nod3);
				cgen(r, t);
				r = t;
			}
		}
		else {
			if(m == SAgen) {
				t = &nod2;
				c = snarfreg(nn, t, D_CX, r, &nod3);
				cgen(r, t);
				r = t;
			}
			if(!vaddr(l, 0)) {
				reglcgen(&nod1, l, Z);
				l = &nod1;
			}
		}

		if(nn != Z) {
			m += SAdgen - SAgen;
			d = regpair(nn, n);
			instpair(d, Z);
			biggen(l, r, d, 0, optab[m], args);
			if(l == &nod1) {
				regfree(&nod1);
				l = Z;
			}
			if(r == &nod2 && c == Z) {
				regfree(&nod2);
				r = Z;
			}
			if(d != nn)
				storepair(d, nn, 1);
		}
		else
			biggen(l, r, Z, 0, optab[m], args);

		if(c != Z) {
			gins(AMOVL, c, r);
			regfree(c);
		}
		if(l == &nod1)
			regfree(&nod1);
		if(r == &nod2)
			regfree(&nod2);
		return 1;

	case OPOSTINC:
		args = ADDargs;
		cp = incdecpost;
		goto vinc;
	case OPOSTDEC:
		args = SUBargs;
		cp = incdecpost;
		goto vinc;
	case OPREINC:
		args = ADDargs;
		cp = incdecpre;
		goto vinc;
	case OPREDEC:
		args = SUBargs;
		cp = incdecpre;
		goto vinc;

	vinc:
		l = n->left;
		if(!vaddr(l, 1)) {
			reglcgen(&nod1, l, Z);
			l = &nod1;
		}
		
		if(nn != Z) {
			d = regpair(nn, n);
			instpair(d, Z);
			biggen(l, Z, d, 0, cp, args);
			if(l == &nod1) {
				regfree(&nod1);
				l = Z;
			}
			if(d != nn)
				storepair(d, nn, 1);
		}
		else
			biggen(l, Z, Z, 0, incdec, args);

		if(l == &nod1)
			regfree(&nod1);
		return 1;

	case OCAST:
		l = n->left;
		if(typev[l->type->etype]) {
			if(!vaddr(l, 1)) {
				if(l->complex + 1 > nn->complex) {
					d = regpair(Z, l);
					sugen(l, d, 8);
					if(!vaddr(nn, 1)) {
						reglcgen(&nod1, nn, Z);
						r = &nod1;
					}
					else
						r = nn;
				}
				else {
					if(!vaddr(nn, 1)) {
						reglcgen(&nod1, nn, Z);
						r = &nod1;
					}
					else
						r = nn;
					d = regpair(Z, l);
					sugen(l, d, 8);
				}
//				d->left->type = r->type;
				d->left->type = types[TLONG];
				gmove(d->left, r);
				freepair(d);
			}
			else {
				if(nn->op != OREGISTER && !vaddr(nn, 1)) {
					reglcgen(&nod1, nn, Z);
					r = &nod1;
				}
				else
					r = nn;
//				l->type = r->type;
				l->type = types[TLONG];
				gmove(l, r);
			}
			if(r != nn)
				regfree(r);
		}
		else {
			if(typeu[l->type->etype] || cond(l->op))
				si = TUNSIGNED;
			else
				si = TSIGNED;
			regalloc(&nod1, l, Z);
			cgen(l, &nod1);
			if(nn->op == OREGPAIR) {
				m = instpair(nn, &nod1);
				biggen(&nod1, Z, nn, si == TSIGNED, castrp, nil);
			}
			else {
				m = 0;
				if(!vaddr(nn, si != TSIGNED)) {
					dt = nn->type;
					nn->type = types[TLONG];
					reglcgen(&nod2, nn, Z);
					nn->type = dt;
					nn = &nod2;
				}
				dt = nn->type;
				nn->type = types[TLONG];
				biggen(&nod1, Z, nn, si == TSIGNED, castrpa, nil);
				nn->type = dt;
				if(nn == &nod2)
					regfree(&nod2);
			}
			if(!m)
				regfree(&nod1);
		}
		return 1;

	default:
		if(n->op == OREGPAIR) {
			storepair(n, nn, 1);
			return 1;
		}
		if(nn->op == OREGPAIR) {
			loadpair(n, nn);
			return 1;
		}
		return 0;
	}
finished:
	if(d != nn)
		storepair(d, nn, 1);
	return 1;
}

void
testv(Node *n, int true)
{
	Type *t;
	Node *nn, nod;

	switch(n->op) {
	case OINDREG:
	case ONAME:
		biggen(n, Z, Z, true, testi, nil);
		break;

	default:
		n = vfunc(n, n);
		if(n->addable >= INDEXED) {
			t = n->type;
			n->type = types[TLONG];
			reglcgen(&nod, n, Z);
			n->type = t;
			n = &nod;
			biggen(n, Z, Z, true, testi, nil);
			if(n == &nod)
				regfree(n);
		}
		else {
			nn = regpair(Z, n);
			sugen(n, nn, 8);
			biggen(nn, Z, Z, true, testi, nil);
			freepair(nn);
		}
	}
}
