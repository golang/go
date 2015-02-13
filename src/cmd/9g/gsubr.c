// Derived from Inferno utils/6c/txt.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/txt.c
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

#include <u.h>
#include <libc.h>
#include "gg.h"
#include "../../runtime/funcdata.h"

// TODO(rsc): Can make this bigger if we move
// the text segment up higher in 6l for all GOOS.
// At the same time, can raise StackBig in ../../runtime/stack.h.
vlong unmappedzero = 4096;

static	int	resvd[] =
{
	REGZERO,
	REGSP,	// reserved for SP
	// We need to preserve the C ABI TLS pointer because sigtramp
	// may happen during C code and needs to access the g.  C
	// clobbers REGG, so if Go were to clobber REGTLS, sigtramp
	// won't know which convention to use.  By preserving REGTLS,
	// we can just retrieve g from TLS when we aren't sure.
	REGTLS,
	// TODO(austin): Consolidate REGTLS and REGG?
	REGG,
	REGTMP,	// REGTMP
	FREGCVI,
	FREGZERO,
	FREGHALF,
	FREGONE,
	FREGTWO,
};

void
ginit(void)
{
	int i;

	for(i=0; i<nelem(reg); i++)
		reg[i] = 1;
	for(i=0; i<NREG+NFREG; i++)
		reg[i] = 0;

	for(i=0; i<nelem(resvd); i++)
		reg[resvd[i] - REG_R0]++;
}

static	uintptr	regpc[nelem(reg)];

void
gclean(void)
{
	int i;

	for(i=0; i<nelem(resvd); i++)
		reg[resvd[i] - REG_R0]--;

	for(i=0; i<nelem(reg); i++)
		if(reg[i])
			yyerror("reg %R left allocated, %p\n", i+REG_R0, regpc[i]);
}

int
anyregalloc(void)
{
	int i, j;

	for(i=0; i<nelem(reg); i++) {
		if(reg[i] == 0)
			goto ok;
		for(j=0; j<nelem(resvd); j++)
			if(resvd[j] == i)
				goto ok;
		return 1;
	ok:;
	}
	return 0;
}

/*
 * allocate register of type t, leave in n.
 * if o != N, o is desired fixed register.
 * caller must regfree(n).
 */
void
regalloc(Node *n, Type *t, Node *o)
{
	int i, et;
	int fixfree, fltfree;

	if(t == T)
		fatal("regalloc: t nil");
	et = simtype[t->etype];

	if(debug['r']) {
		fixfree = 0;
		fltfree = 0;
		for(i = REG_R0; i < REG_F31; i++)
			if(reg[i - REG_R0] == 0) {
				if(i < REG_F0)
					fixfree++;
				else
					fltfree++;
			}
		print("regalloc fix %d flt %d free\n", fixfree, fltfree);
	}

	switch(et) {
	case TINT8:
	case TUINT8:
	case TINT16:
	case TUINT16:
	case TINT32:
	case TUINT32:
	case TINT64:
	case TUINT64:
	case TPTR32:
	case TPTR64:
	case TBOOL:
		if(o != N && o->op == OREGISTER) {
			i = o->val.u.reg;
			if(i >= REGMIN && i <= REGMAX)
				goto out;
		}
		for(i=REGMIN; i<=REGMAX; i++)
			if(reg[i - REG_R0] == 0) {
				regpc[i - REG_R0] = (uintptr)getcallerpc(&n);
				goto out;
			}
		flusherrors();
		for(i=REG_R0; i<REG_R0+NREG; i++)
			print("R%d %p\n", i, regpc[i - REG_R0]);
		fatal("out of fixed registers");

	case TFLOAT32:
	case TFLOAT64:
		if(o != N && o->op == OREGISTER) {
			i = o->val.u.reg;
			if(i >= FREGMIN && i <= FREGMAX)
				goto out;
		}
		for(i=FREGMIN; i<=FREGMAX; i++)
			if(reg[i - REG_R0] == 0) {
				regpc[i - REG_R0] = (uintptr)getcallerpc(&n);
				goto out;
			}
		flusherrors();
		for(i=REG_F0; i<REG_F0+NREG; i++)
			print("F%d %p\n", i, regpc[i - REG_R0]);
		fatal("out of floating registers");

	case TCOMPLEX64:
	case TCOMPLEX128:
		tempname(n, t);
		return;
	}
	fatal("regalloc: unknown type %T", t);
	return;

out:
	reg[i - REG_R0]++;
	nodreg(n, t, i);
}

void
regfree(Node *n)
{
	int i;

	if(n->op == ONAME)
		return;
	if(n->op != OREGISTER && n->op != OINDREG)
		fatal("regfree: not a register");
	i = n->val.u.reg - REG_R0;
	if(i == REGSP - REG_R0)
		return;
	if(i < 0 || i >= nelem(reg))
		fatal("regfree: reg out of range");
	if(reg[i] <= 0)
		fatal("regfree: reg not allocated");
	reg[i]--;
	if(reg[i] == 0)
		regpc[i] = 0;
}

/*
 * generate
 *	as $c, n
 */
void
ginscon(int as, vlong c, Node *n2)
{
	Node n1, ntmp;

	nodconst(&n1, types[TINT64], c);

	if(as != AMOVD && (c < -BIG || c > BIG)) {
		// cannot have more than 16-bit of immediate in ADD, etc.
		// instead, MOV into register first.
		regalloc(&ntmp, types[TINT64], N);
		gins(AMOVD, &n1, &ntmp);
		gins(as, &ntmp, n2);
		regfree(&ntmp);
		return;
	}
	gins(as, &n1, n2);
}

/*
 * generate
 *	as n, $c (CMP/CMPU)
 */
void
ginscon2(int as, Node *n2, vlong c)
{
	Node n1, ntmp;

	nodconst(&n1, types[TINT64], c);

	switch(as) {
	default:
		fatal("ginscon2");
	case ACMP:
		if(-BIG <= c && c <= BIG) {
			gins(as, n2, &n1);
			return;
		}
		break;
	case ACMPU:
		if(0 <= c && c <= 2*BIG) {
			gins(as, n2, &n1);
			return;
		}
		break;
	}
	// MOV n1 into register first
	regalloc(&ntmp, types[TINT64], N);
	gins(AMOVD, &n1, &ntmp);
	gins(as, n2, &ntmp);
	regfree(&ntmp);
}

#define	CASE(a,b)	(((a)<<16)|((b)<<0))
/*c2go int CASE(int, int); */

/*
 * set up nodes representing 2^63
 */
Node bigi;
Node bigf;

void
bignodes(void)
{
	static int did;

	if(did)
		return;
	did = 1;

	nodconst(&bigi, types[TUINT64], 1);
	mpshiftfix(bigi.val.u.xval, 63);

	bigf = bigi;
	bigf.type = types[TFLOAT64];
	bigf.val.ctype = CTFLT;
	bigf.val.u.fval = mal(sizeof *bigf.val.u.fval);
	mpmovefixflt(bigf.val.u.fval, bigi.val.u.xval);
}

/*
 * generate move:
 *	t = f
 * hard part is conversions.
 */
void
gmove(Node *f, Node *t)
{
	int a, ft, tt;
	Type *cvt;
	Node r1, r2, r3, con;
	Prog *p1, *p2;

	if(debug['M'])
		print("gmove %lN -> %lN\n", f, t);

	ft = simsimtype(f->type);
	tt = simsimtype(t->type);
	cvt = t->type;

	if(iscomplex[ft] || iscomplex[tt]) {
		complexmove(f, t);
		return;
	}

	// cannot have two memory operands
	if(ismem(f) && ismem(t))
		goto hard;

	// convert constant to desired type
	if(f->op == OLITERAL) {
		switch(tt) {
		default:
			convconst(&con, t->type, &f->val);
			break;

		case TINT32:
		case TINT16:
		case TINT8:
			convconst(&con, types[TINT64], &f->val);
			regalloc(&r1, con.type, t);
			gins(AMOVD, &con, &r1);
			gmove(&r1, t);
			regfree(&r1);
			return;

		case TUINT32:
		case TUINT16:
		case TUINT8:
			convconst(&con, types[TUINT64], &f->val);
			regalloc(&r1, con.type, t);
			gins(AMOVD, &con, &r1);
			gmove(&r1, t);
			regfree(&r1);
			return;
		}

		f = &con;
		ft = tt;	// so big switch will choose a simple mov

		// constants can't move directly to memory.
		if(ismem(t)) {
			goto hard;
			// float constants come from memory.
			//if(isfloat[tt])
			//	goto hard;

			// 64-bit immediates are also from memory.
			//if(isint[tt])
			//	goto hard;
			//// 64-bit immediates are really 32-bit sign-extended
			//// unless moving into a register.
			//if(isint[tt]) {
			//	if(mpcmpfixfix(con.val.u.xval, minintval[TINT32]) < 0)
			//		goto hard;
			//	if(mpcmpfixfix(con.val.u.xval, maxintval[TINT32]) > 0)
			//		goto hard;
			//}
		}
	}

	// value -> value copy, only one memory operand.
	// figure out the instruction to use.
	// break out of switch for one-instruction gins.
	// goto rdst for "destination must be register".
	// goto hard for "convert to cvt type first".
	// otherwise handle and return.

	switch(CASE(ft, tt)) {
	default:
		fatal("gmove %lT -> %lT", f->type, t->type);

	/*
	 * integer copy and truncate
	 */
	case CASE(TINT8, TINT8):	// same size
	case CASE(TUINT8, TINT8):
	case CASE(TINT16, TINT8):	// truncate
	case CASE(TUINT16, TINT8):
	case CASE(TINT32, TINT8):
	case CASE(TUINT32, TINT8):
	case CASE(TINT64, TINT8):
	case CASE(TUINT64, TINT8):
		a = AMOVB;
		break;

	case CASE(TINT8, TUINT8):	// same size
	case CASE(TUINT8, TUINT8):
	case CASE(TINT16, TUINT8):	// truncate
	case CASE(TUINT16, TUINT8):
	case CASE(TINT32, TUINT8):
	case CASE(TUINT32, TUINT8):
	case CASE(TINT64, TUINT8):
	case CASE(TUINT64, TUINT8):
		a = AMOVBZ;
		break;

	case CASE(TINT16, TINT16):	// same size
	case CASE(TUINT16, TINT16):
	case CASE(TINT32, TINT16):	// truncate
	case CASE(TUINT32, TINT16):
	case CASE(TINT64, TINT16):
	case CASE(TUINT64, TINT16):
		a = AMOVH;
		break;

	case CASE(TINT16, TUINT16):	// same size
	case CASE(TUINT16, TUINT16):
	case CASE(TINT32, TUINT16):	// truncate
	case CASE(TUINT32, TUINT16):
	case CASE(TINT64, TUINT16):
	case CASE(TUINT64, TUINT16):
		a = AMOVHZ;
		break;

	case CASE(TINT32, TINT32):	// same size
	case CASE(TUINT32, TINT32):
	case CASE(TINT64, TINT32):	// truncate
	case CASE(TUINT64, TINT32):
		a = AMOVW;
		break;

	case CASE(TINT32, TUINT32):	// same size
	case CASE(TUINT32, TUINT32):
	case CASE(TINT64, TUINT32):
	case CASE(TUINT64, TUINT32):
		a = AMOVWZ;
		break;

	case CASE(TINT64, TINT64):	// same size
	case CASE(TINT64, TUINT64):
	case CASE(TUINT64, TINT64):
	case CASE(TUINT64, TUINT64):
		a = AMOVD;
		break;

	/*
	 * integer up-conversions
	 */
	case CASE(TINT8, TINT16):	// sign extend int8
	case CASE(TINT8, TUINT16):
	case CASE(TINT8, TINT32):
	case CASE(TINT8, TUINT32):
	case CASE(TINT8, TINT64):
	case CASE(TINT8, TUINT64):
		a = AMOVB;
		goto rdst;

	case CASE(TUINT8, TINT16):	// zero extend uint8
	case CASE(TUINT8, TUINT16):
	case CASE(TUINT8, TINT32):
	case CASE(TUINT8, TUINT32):
	case CASE(TUINT8, TINT64):
	case CASE(TUINT8, TUINT64):
		a = AMOVBZ;
		goto rdst;

	case CASE(TINT16, TINT32):	// sign extend int16
	case CASE(TINT16, TUINT32):
	case CASE(TINT16, TINT64):
	case CASE(TINT16, TUINT64):
		a = AMOVH;
		goto rdst;

	case CASE(TUINT16, TINT32):	// zero extend uint16
	case CASE(TUINT16, TUINT32):
	case CASE(TUINT16, TINT64):
	case CASE(TUINT16, TUINT64):
		a = AMOVHZ;
		goto rdst;

	case CASE(TINT32, TINT64):	// sign extend int32
	case CASE(TINT32, TUINT64):
		a = AMOVW;
		goto rdst;

	case CASE(TUINT32, TINT64):	// zero extend uint32
	case CASE(TUINT32, TUINT64):
		a = AMOVWZ;
		goto rdst;

	/*
	* float to integer
	*/
	case CASE(TFLOAT32, TINT32):
	case CASE(TFLOAT64, TINT32):
	case CASE(TFLOAT32, TINT64):
	case CASE(TFLOAT64, TINT64):
	case CASE(TFLOAT32, TINT16):
	case CASE(TFLOAT32, TINT8):
	case CASE(TFLOAT32, TUINT16):
	case CASE(TFLOAT32, TUINT8):
	case CASE(TFLOAT64, TINT16):
	case CASE(TFLOAT64, TINT8):
	case CASE(TFLOAT64, TUINT16):
	case CASE(TFLOAT64, TUINT8):
	case CASE(TFLOAT32, TUINT32):
	case CASE(TFLOAT64, TUINT32):
	case CASE(TFLOAT32, TUINT64):
	case CASE(TFLOAT64, TUINT64):
		//warn("gmove: convert float to int not implemented: %N -> %N\n", f, t);
		//return;
		// algorithm is:
		//	if small enough, use native float64 -> int64 conversion.
		//	otherwise, subtract 2^63, convert, and add it back.
		bignodes();
		regalloc(&r1, types[ft], f);
		gmove(f, &r1);
		if(tt == TUINT64) {
			regalloc(&r2, types[TFLOAT64], N);
			gmove(&bigf, &r2);
			gins(AFCMPU, &r1, &r2);
			p1 = gbranch(optoas(OLT, types[TFLOAT64]), T, +1);
			gins(AFSUB, &r2, &r1);
			patch(p1, pc);
			regfree(&r2);
		}
		regalloc(&r2, types[TFLOAT64], N);
		regalloc(&r3, types[TINT64], t);
		gins(AFCTIDZ, &r1, &r2);
		p1 = gins(AFMOVD, &r2, N);
		p1->to.type = TYPE_MEM;
		p1->to.reg = REGSP;
		p1->to.offset = -8;
		p1 = gins(AMOVD, N, &r3);
		p1->from.type = TYPE_MEM;
		p1->from.reg = REGSP;
		p1->from.offset = -8;
		regfree(&r2);
		regfree(&r1);
		if(tt == TUINT64) {
			p1 = gbranch(optoas(OLT, types[TFLOAT64]), T, +1); // use CR0 here again
			nodreg(&r1, types[TINT64], REGTMP);
			gins(AMOVD, &bigi, &r1);
			gins(AADD, &r1, &r3);
			patch(p1, pc);
		}
		gmove(&r3, t);
		regfree(&r3);
		return;

	/*
	 * integer to float
	 */
	case CASE(TINT32, TFLOAT32):
	case CASE(TINT32, TFLOAT64):
	case CASE(TINT64, TFLOAT32):
	case CASE(TINT64, TFLOAT64):
	case CASE(TINT16, TFLOAT32):
	case CASE(TINT16, TFLOAT64):
	case CASE(TINT8, TFLOAT32):
	case CASE(TINT8, TFLOAT64):
	case CASE(TUINT16, TFLOAT32):
	case CASE(TUINT16, TFLOAT64):
	case CASE(TUINT8, TFLOAT32):
	case CASE(TUINT8, TFLOAT64):
	case CASE(TUINT32, TFLOAT32):
	case CASE(TUINT32, TFLOAT64):
	case CASE(TUINT64, TFLOAT32):
	case CASE(TUINT64, TFLOAT64):
		//warn("gmove: convert int to float not implemented: %N -> %N\n", f, t);
		//return;
		// algorithm is:
		//	if small enough, use native int64 -> uint64 conversion.
		//	otherwise, halve (rounding to odd?), convert, and double.
		bignodes();
		regalloc(&r1, types[TINT64], N);
		gmove(f, &r1);
		if(ft == TUINT64) {
			nodreg(&r2, types[TUINT64], REGTMP);
			gmove(&bigi, &r2);
			gins(ACMPU, &r1, &r2);
			p1 = gbranch(optoas(OLT, types[TUINT64]), T, +1);
			p2 = gins(ASRD, N, &r1);
			p2->from.type = TYPE_CONST;
			p2->from.offset = 1;
			patch(p1, pc);
		}
		regalloc(&r2, types[TFLOAT64], t);
		p1 = gins(AMOVD, &r1, N);
		p1->to.type = TYPE_MEM;
		p1->to.reg = REGSP;
		p1->to.offset = -8;
		p1 = gins(AFMOVD, N, &r2);
		p1->from.type = TYPE_MEM;
		p1->from.reg = REGSP;
		p1->from.offset = -8;
		gins(AFCFID, &r2, &r2);
		regfree(&r1);
		if(ft == TUINT64) {
			p1 = gbranch(optoas(OLT, types[TUINT64]), T, +1); // use CR0 here again
			nodreg(&r1, types[TFLOAT64], FREGTWO);
			gins(AFMUL, &r1, &r2);
			patch(p1, pc);
		}
		gmove(&r2, t);
		regfree(&r2);
		return;

	/*
	 * float to float
	 */
	case CASE(TFLOAT32, TFLOAT32):
		a = AFMOVS;
		break;

	case CASE(TFLOAT64, TFLOAT64):
		a = AFMOVD;
		break;

	case CASE(TFLOAT32, TFLOAT64):
		a = AFMOVS;
		goto rdst;

	case CASE(TFLOAT64, TFLOAT32):
		a = AFRSP;
		goto rdst;
	}

	gins(a, f, t);
	return;

rdst:
	// requires register destination
	regalloc(&r1, t->type, t);
	gins(a, f, &r1);
	gmove(&r1, t);
	regfree(&r1);
	return;

hard:
	// requires register intermediate
	regalloc(&r1, cvt, t);
	gmove(f, &r1);
	gmove(&r1, t);
	regfree(&r1);
	return;
}

/*
 * generate one instruction:
 *	as f, t
 */
Prog*
gins(int as, Node *f, Node *t)
{
	int32 w;
	Prog *p;
	Addr af, at;

	// TODO(austin): Add self-move test like in 6g (but be careful
	// of truncation moves)

	memset(&af, 0, sizeof af);
	memset(&at, 0, sizeof at);
	if(f != N)
		naddr(f, &af, 1);
	if(t != N)
		naddr(t, &at, 1);
	p = prog(as);
	if(f != N)
		p->from = af;
	if(t != N)
		p->to = at;
	if(debug['g'])
		print("%P\n", p);

	w = 0;
	switch(as) {
	case AMOVB:
	case AMOVBU:
	case AMOVBZ:
	case AMOVBZU:
		w = 1;
		break;
	case AMOVH:
	case AMOVHU:
	case AMOVHZ:
	case AMOVHZU:
		w = 2;
		break;
	case AMOVW:
	case AMOVWU:
	case AMOVWZ:
	case AMOVWZU:
		w = 4;
		break;
	case AMOVD:
	case AMOVDU:
		if(af.type == TYPE_CONST || af.type == TYPE_ADDR)
			break;
		w = 8;
		break;
	}
	if(w != 0 && ((f != N && af.width < w) || (t != N && at.type != TYPE_REG && at.width > w))) {
		dump("f", f);
		dump("t", t);
		fatal("bad width: %P (%d, %d)\n", p, af.width, at.width);
	}

	return p;
}

void
fixlargeoffset(Node *n)
{
	Node a;

	if(n == N)
		return;
	if(n->op != OINDREG)
		return;
	if(n->val.u.reg == REGSP) // stack offset cannot be large
		return;
	if(n->xoffset != (int32)n->xoffset) {
		// TODO(minux): offset too large, move into R31 and add to R31 instead.
		// this is used only in test/fixedbugs/issue6036.go.
		fatal("offset too large: %N", n);
		a = *n;
		a.op = OREGISTER;
		a.type = types[tptr];
		a.xoffset = 0;
		cgen_checknil(&a);
		ginscon(optoas(OADD, types[tptr]), n->xoffset, &a);
		n->xoffset = 0;
	}
}

/*
 * return Axxx for Oxxx on type t.
 */
int
optoas(int op, Type *t)
{
	int a;

	if(t == T)
		fatal("optoas: t is nil");

	a = AXXX;
	switch(CASE(op, simtype[t->etype])) {
	default:
		fatal("optoas: no entry for op=%O type=%T", op, t);
		break;

	case CASE(OEQ, TBOOL):
	case CASE(OEQ, TINT8):
	case CASE(OEQ, TUINT8):
	case CASE(OEQ, TINT16):
	case CASE(OEQ, TUINT16):
	case CASE(OEQ, TINT32):
	case CASE(OEQ, TUINT32):
	case CASE(OEQ, TINT64):
	case CASE(OEQ, TUINT64):
	case CASE(OEQ, TPTR32):
	case CASE(OEQ, TPTR64):
	case CASE(OEQ, TFLOAT32):
	case CASE(OEQ, TFLOAT64):
		a = ABEQ;
		break;

	case CASE(ONE, TBOOL):
	case CASE(ONE, TINT8):
	case CASE(ONE, TUINT8):
	case CASE(ONE, TINT16):
	case CASE(ONE, TUINT16):
	case CASE(ONE, TINT32):
	case CASE(ONE, TUINT32):
	case CASE(ONE, TINT64):
	case CASE(ONE, TUINT64):
	case CASE(ONE, TPTR32):
	case CASE(ONE, TPTR64):
	case CASE(ONE, TFLOAT32):
	case CASE(ONE, TFLOAT64):
		a = ABNE;
		break;

	case CASE(OLT, TINT8):	// ACMP
	case CASE(OLT, TINT16):
	case CASE(OLT, TINT32):
	case CASE(OLT, TINT64):
	case CASE(OLT, TUINT8):	// ACMPU
	case CASE(OLT, TUINT16):
	case CASE(OLT, TUINT32):
	case CASE(OLT, TUINT64):
	case CASE(OLT, TFLOAT32): // AFCMPU
	case CASE(OLT, TFLOAT64):
		a = ABLT;
		break;

	case CASE(OLE, TINT8):	// ACMP
	case CASE(OLE, TINT16):
	case CASE(OLE, TINT32):
	case CASE(OLE, TINT64):
	case CASE(OLE, TUINT8):	// ACMPU
	case CASE(OLE, TUINT16):
	case CASE(OLE, TUINT32):
	case CASE(OLE, TUINT64):
	case CASE(OLE, TFLOAT32): // AFCMPU
	case CASE(OLE, TFLOAT64):
		a = ABLE;
		break;

	case CASE(OGT, TINT8):
	case CASE(OGT, TINT16):
	case CASE(OGT, TINT32):
	case CASE(OGT, TINT64):
	case CASE(OGT, TUINT8):
	case CASE(OGT, TUINT16):
	case CASE(OGT, TUINT32):
	case CASE(OGT, TUINT64):
	case CASE(OGT, TFLOAT32):
	case CASE(OGT, TFLOAT64):
		a = ABGT;
		break;

	case CASE(OGE, TINT8):
	case CASE(OGE, TINT16):
	case CASE(OGE, TINT32):
	case CASE(OGE, TINT64):
	case CASE(OGE, TUINT8):
	case CASE(OGE, TUINT16):
	case CASE(OGE, TUINT32):
	case CASE(OGE, TUINT64):
	case CASE(OGE, TFLOAT32):
	case CASE(OGE, TFLOAT64):
		a = ABGE;
		break;

	case CASE(OCMP, TBOOL):
	case CASE(OCMP, TINT8):
	case CASE(OCMP, TINT16):
	case CASE(OCMP, TINT32):
	case CASE(OCMP, TPTR32):
	case CASE(OCMP, TINT64):
		a = ACMP;
		break;

	case CASE(OCMP, TUINT8):
	case CASE(OCMP, TUINT16):
	case CASE(OCMP, TUINT32):
	case CASE(OCMP, TUINT64):
	case CASE(OCMP, TPTR64):
		a = ACMPU;
		break;

	case CASE(OCMP, TFLOAT32):
	case CASE(OCMP, TFLOAT64):
		a = AFCMPU;
		break;

	case CASE(OAS, TBOOL):
	case CASE(OAS, TINT8):
		a = AMOVB;
		break;

	case CASE(OAS, TUINT8):
		a = AMOVBZ;
		break;

	case CASE(OAS, TINT16):
		a = AMOVH;
		break;

	case CASE(OAS, TUINT16):
		a = AMOVHZ;
		break;

	case CASE(OAS, TINT32):
		a = AMOVW;
		break;

	case CASE(OAS, TUINT32):
	case CASE(OAS, TPTR32):
		a = AMOVWZ;
		break;

	case CASE(OAS, TINT64):
	case CASE(OAS, TUINT64):
	case CASE(OAS, TPTR64):
		a = AMOVD;
		break;

	case CASE(OAS, TFLOAT32):
		a = AFMOVS;
		break;

	case CASE(OAS, TFLOAT64):
		a = AFMOVD;
		break;

	case CASE(OADD, TINT8):
	case CASE(OADD, TUINT8):
	case CASE(OADD, TINT16):
	case CASE(OADD, TUINT16):
	case CASE(OADD, TINT32):
	case CASE(OADD, TUINT32):
	case CASE(OADD, TPTR32):
	case CASE(OADD, TINT64):
	case CASE(OADD, TUINT64):
	case CASE(OADD, TPTR64):
		a = AADD;
		break;

	case CASE(OADD, TFLOAT32):
		a = AFADDS;
		break;

	case CASE(OADD, TFLOAT64):
		a = AFADD;
		break;

	case CASE(OSUB, TINT8):
	case CASE(OSUB, TUINT8):
	case CASE(OSUB, TINT16):
	case CASE(OSUB, TUINT16):
	case CASE(OSUB, TINT32):
	case CASE(OSUB, TUINT32):
	case CASE(OSUB, TPTR32):
	case CASE(OSUB, TINT64):
	case CASE(OSUB, TUINT64):
	case CASE(OSUB, TPTR64):
		a = ASUB;
		break;

	case CASE(OSUB, TFLOAT32):
		a = AFSUBS;
		break;

	case CASE(OSUB, TFLOAT64):
		a = AFSUB;
		break;

	case CASE(OMINUS, TINT8):
	case CASE(OMINUS, TUINT8):
	case CASE(OMINUS, TINT16):
	case CASE(OMINUS, TUINT16):
	case CASE(OMINUS, TINT32):
	case CASE(OMINUS, TUINT32):
	case CASE(OMINUS, TPTR32):
	case CASE(OMINUS, TINT64):
	case CASE(OMINUS, TUINT64):
	case CASE(OMINUS, TPTR64):
		a = ANEG;
		break;

	case CASE(OAND, TINT8):
	case CASE(OAND, TUINT8):
	case CASE(OAND, TINT16):
	case CASE(OAND, TUINT16):
	case CASE(OAND, TINT32):
	case CASE(OAND, TUINT32):
	case CASE(OAND, TPTR32):
	case CASE(OAND, TINT64):
	case CASE(OAND, TUINT64):
	case CASE(OAND, TPTR64):
		a = AAND;
		break;

	case CASE(OOR, TINT8):
	case CASE(OOR, TUINT8):
	case CASE(OOR, TINT16):
	case CASE(OOR, TUINT16):
	case CASE(OOR, TINT32):
	case CASE(OOR, TUINT32):
	case CASE(OOR, TPTR32):
	case CASE(OOR, TINT64):
	case CASE(OOR, TUINT64):
	case CASE(OOR, TPTR64):
		a = AOR;
		break;

	case CASE(OXOR, TINT8):
	case CASE(OXOR, TUINT8):
	case CASE(OXOR, TINT16):
	case CASE(OXOR, TUINT16):
	case CASE(OXOR, TINT32):
	case CASE(OXOR, TUINT32):
	case CASE(OXOR, TPTR32):
	case CASE(OXOR, TINT64):
	case CASE(OXOR, TUINT64):
	case CASE(OXOR, TPTR64):
		a = AXOR;
		break;

	// TODO(minux): handle rotates
	//case CASE(OLROT, TINT8):
	//case CASE(OLROT, TUINT8):
	//case CASE(OLROT, TINT16):
	//case CASE(OLROT, TUINT16):
	//case CASE(OLROT, TINT32):
	//case CASE(OLROT, TUINT32):
	//case CASE(OLROT, TPTR32):
	//case CASE(OLROT, TINT64):
	//case CASE(OLROT, TUINT64):
	//case CASE(OLROT, TPTR64):
	//	a = 0//???; RLDC?
	//	break;

	case CASE(OLSH, TINT8):
	case CASE(OLSH, TUINT8):
	case CASE(OLSH, TINT16):
	case CASE(OLSH, TUINT16):
	case CASE(OLSH, TINT32):
	case CASE(OLSH, TUINT32):
	case CASE(OLSH, TPTR32):
	case CASE(OLSH, TINT64):
	case CASE(OLSH, TUINT64):
	case CASE(OLSH, TPTR64):
		a = ASLD;
		break;

	case CASE(ORSH, TUINT8):
	case CASE(ORSH, TUINT16):
	case CASE(ORSH, TUINT32):
	case CASE(ORSH, TPTR32):
	case CASE(ORSH, TUINT64):
	case CASE(ORSH, TPTR64):
		a = ASRD;
		break;

	case CASE(ORSH, TINT8):
	case CASE(ORSH, TINT16):
	case CASE(ORSH, TINT32):
	case CASE(ORSH, TINT64):
		a = ASRAD;
		break;

	// TODO(minux): handle rotates
	//case CASE(ORROTC, TINT8):
	//case CASE(ORROTC, TUINT8):
	//case CASE(ORROTC, TINT16):
	//case CASE(ORROTC, TUINT16):
	//case CASE(ORROTC, TINT32):
	//case CASE(ORROTC, TUINT32):
	//case CASE(ORROTC, TINT64):
	//case CASE(ORROTC, TUINT64):
	//	a = 0//??? RLDC??
	//	break;

	case CASE(OHMUL, TINT64):
		a = AMULHD;
		break;
	case CASE(OHMUL, TUINT64):
	case CASE(OHMUL, TPTR64):
		a = AMULHDU;
		break;

	case CASE(OMUL, TINT8):
	case CASE(OMUL, TINT16):
	case CASE(OMUL, TINT32):
	case CASE(OMUL, TINT64):
		a = AMULLD;
		break;

	case CASE(OMUL, TUINT8):
	case CASE(OMUL, TUINT16):
	case CASE(OMUL, TUINT32):
	case CASE(OMUL, TPTR32):
		// don't use word multiply, the high 32-bit are undefined.
		// fallthrough
	case CASE(OMUL, TUINT64):
	case CASE(OMUL, TPTR64):
		a = AMULLD; // for 64-bit multiplies, signedness doesn't matter.
		break;

	case CASE(OMUL, TFLOAT32):
		a = AFMULS;
		break;

	case CASE(OMUL, TFLOAT64):
		a = AFMUL;
		break;

	case CASE(ODIV, TINT8):
	case CASE(ODIV, TINT16):
	case CASE(ODIV, TINT32):
	case CASE(ODIV, TINT64):
		a = ADIVD;
		break;

	case CASE(ODIV, TUINT8):
	case CASE(ODIV, TUINT16):
	case CASE(ODIV, TUINT32):
	case CASE(ODIV, TPTR32):
	case CASE(ODIV, TUINT64):
	case CASE(ODIV, TPTR64):
		a = ADIVDU;
		break;

	case CASE(ODIV, TFLOAT32):
		a = AFDIVS;
		break;

	case CASE(ODIV, TFLOAT64):
		a = AFDIV;
		break;

	}
	return a;
}

enum
{
	ODynam		= 1<<0,
	OAddable	= 1<<1,
};

int
xgen(Node *n, Node *a, int o)
{
	// TODO(minux)
	USED(n); USED(a); USED(o);
	return -1;
}

void
sudoclean(void)
{
	return;
}

/*
 * generate code to compute address of n,
 * a reference to a (perhaps nested) field inside
 * an array or struct.
 * return 0 on failure, 1 on success.
 * on success, leaves usable address in a.
 *
 * caller is responsible for calling sudoclean
 * after successful sudoaddable,
 * to release the register used for a.
 */
int
sudoaddable(int as, Node *n, Addr *a)
{
	// TODO(minux)
	USED(as); USED(n);
	memset(a, 0, sizeof *a);
	return 0;
}
