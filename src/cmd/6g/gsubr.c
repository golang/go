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

#include "gg.h"

void
clearp(Prog *p)
{
	p->as = AEND;
	p->from.type = D_NONE;
	p->from.index = D_NONE;
	p->to.type = D_NONE;
	p->to.index = D_NONE;
	p->loc = pcloc;
	pcloc++;
}

Prog*
prog(int as)
{
	Prog *p;

	p = pc;
	pc = mal(sizeof(*pc));

	clearp(pc);

	if(lineno == 0) {
		if(debug['K'])
			warn("prog: line 0");
	}

	p->as = as;
	p->lineno = lineno;
	p->link = pc;
	return p;
}

Prog*
gbranch(int as, Type *t)
{
	Prog *p;

	p = prog(as);
	p->to.type = D_BRANCH;
	p->to.branch = P;
	return p;
}

void
patch(Prog *p, Prog *to)
{
	if(p->to.type != D_BRANCH)
		fatal("patch: not a branch");
	p->to.branch = to;
	p->to.offset = to->loc;
}

Plist*
newplist(void)
{
	Plist *pl;

	pl = mal(sizeof(*pl));
	if(plist == nil)
		plist = pl;
	else
		plast->link = pl;
	plast = pl;

	pc = mal(sizeof(*pc));
	clearp(pc);
	pl->firstpc = pc;

	return pl;
}

void
ginit(void)
{
	int i;

	for(i=0; i<nelem(reg); i++)
		reg[i] = 1;
	for(i=D_AX; i<=D_R15; i++)
		reg[i] = 0;
	for(i=D_X0; i<=D_X7; i++)
		reg[i] = 0;
	reg[D_SP]++;
}

void
gclean(void)
{
	int i;

	reg[D_SP]--;
	for(i=D_AX; i<=D_R15; i++)
		if(reg[i])
			yyerror("reg %R left allocated\n", i);
	for(i=D_X0; i<=D_X7; i++)
		if(reg[i])
			yyerror("reg %R left allocated\n", i);
}

void
regalloc(Node *n, Type *t, Node *o)
{
	int i, et;

	if(t == T)
		fatal("regalloc: t nil");
	et = simtype[t->etype];
	
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
			if(i >= D_AX && i <= D_R15)
				goto out;
		}
		for(i=D_AX; i<=D_R15; i++)
			if(reg[i] == 0)
				goto out;

		yyerror("out of fixed registers");
		goto err;

	case TFLOAT32:
	case TFLOAT64:
	case TFLOAT80:
		if(o != N && o->op == OREGISTER) {
			i = o->val.u.reg;
			if(i >= D_X0 && i <= D_X7)
				goto out;
		}
		for(i=D_X0; i<=D_X7; i++)
			if(reg[i] == 0)
				goto out;
		yyerror("out of floating registers");
		goto err;
	}
	yyerror("regalloc: unknown type %T", t);

err:
	nodreg(n, t, 0);
	return;

out:
	reg[i]++;
	nodreg(n, t, i);
}

void
regfree(Node *n)
{
	int i;

	if(n->op != OREGISTER && n->op != OINDREG)
		fatal("regfree: not a register");
	i = n->val.u.reg;
	if(i < 0 || i >= sizeof(reg))
		fatal("regfree: reg out of range");
	if(reg[i] <= 0)
		fatal("regfree: reg not allocated");
	reg[i]--;
}

void
regret(Node *n, Type *t)
{
	if(t == T)
		fatal("regret: t nil");
	fatal("regret");
}

void
nodreg(Node *n, Type *t, int r)
{
	if(t == T)
		fatal("nodreg: t nil");

	memset(n, 0, sizeof(*n));
	n->op = OREGISTER;
	n->addable = 1;
	ullmancalc(n);
	n->val.u.reg = r;
	n->type = t;
}

void
nodindreg(Node *n, Type *t, int r)
{
	nodreg(n, t, r);
	n->op = OINDREG;
}

Node*
nodarg(Type *t, int fp)
{
	Node *n;

	if(t->etype != TFIELD)
		fatal("nodarg: not field %T", t);

	n = nod(ONAME, N, N);
	n->type = t->type;
	n->sym = t->sym;
	if(t->width == BADWIDTH)
		fatal("nodarg: offset not computed for %T", t);
	n->xoffset = t->width;
	n->addable = 1;

	switch(fp) {
	case 0:		// output arg
		n->op = OINDREG;
		n->val.u.reg = D_SP;
		break;

	case 1:		// input arg
		n->class = PPARAM;
		break;

	case 2:		// offset output arg
fatal("shpuldnt be used");
		n->op = OINDREG;
		n->val.u.reg = D_SP;
		n->xoffset += types[tptr]->width;
		break;
	}
	return n;
}

void
nodconst(Node *n, Type *t, vlong v)
{
	memset(n, 0, sizeof(*n));
	n->op = OLITERAL;
	n->addable = 1;
	ullmancalc(n);
	n->val.u.xval = mal(sizeof(*n->val.u.xval));
	mpmovecfix(n->val.u.xval, v);
	n->val.ctype = CTINT;
	n->type = t;

	switch(t->etype) {
	case TFLOAT32:
	case TFLOAT64:
	case TFLOAT80:
		fatal("nodconst: bad type %T", t);

	case TPTR32:
	case TPTR64:
	case TUINT8:
	case TUINT16:
	case TUINT32:
	case TUINT64:
		n->val.ctype = CTUINT;
		break;
	}
}

void
gconreg(int as, vlong c, int reg)
{
	Node n1, n2;

	nodconst(&n1, types[TINT64], c);
	nodreg(&n2, types[TINT64], reg);
	gins(as, &n1, &n2);
}

#define	CASE(a,b)	(((a)<<16)|((b)<<0))

void
gmove(Node *f, Node *t)
{
	int ft, tt, t64, a;
	Node nod, nod1, nod2, nod3, nodc;
	Prog *p1, *p2;

	ft = simtype[f->type->etype];
	tt = simtype[t->type->etype];

	t64 = 0;
	if(tt == TINT64 || tt == TUINT64 || tt == TPTR64)
		t64 = 1;

	if(debug['M'])
		print("gop: %O %O[%E],%O[%E]\n", OAS,
			f->op, ft, t->op, tt);
	if(isfloat[ft] && f->op == OCONST) {
		/* TO DO: pick up special constants, possibly preloaded */
		if(mpgetflt(f->val.u.fval) == 0.0) {
			regalloc(&nod, t->type, t);
			gins(AXORPD, &nod, &nod);
			gmove(&nod, t);
			regfree(&nod);
			return;
		}
	}
/*
 * load
 */
	if(f->op == ONAME || f->op == OINDREG ||
	   f->op == OIND || f->op == OINDEX)
	switch(ft) {
	case TINT8:
		a = AMOVBLSX;
		if(t64)
			a = AMOVBQSX;
		goto ld;
	case TBOOL:
	case TUINT8:
		a = AMOVBLZX;
		if(t64)
			a = AMOVBQZX;
		goto ld;
	case TINT16:
		a = AMOVWLSX;
		if(t64)
			a = AMOVWQSX;
		goto ld;
	case TUINT16:
		a = AMOVWLZX;
		if(t64)
			a = AMOVWQZX;
		goto ld;
	case TINT32:
		if(isfloat[tt]) {
			regalloc(&nod, t->type, t);
			if(tt == TFLOAT64)
				a = ACVTSL2SD;
			else
				a = ACVTSL2SS;
			gins(a, f, &nod);
			gmove(&nod, t);
			regfree(&nod);
			return;
		}
		a = AMOVL;
		if(t64)
			a = AMOVLQSX;
		goto ld;
	case TUINT32:
	case TPTR32:
		a = AMOVL;
		if(t64)
			a = AMOVLQZX;
		goto ld;
	case TINT64:
		if(isfloat[tt]) {
			regalloc(&nod, t->type, t);
			if(tt == TFLOAT64)
				a = ACVTSQ2SD;
			else
				a = ACVTSQ2SS;
			gins(a, f, &nod);
			gmove(&nod, t);
			regfree(&nod);
			return;
		}
	case TUINT64:
	case TPTR64:
		a = AMOVQ;

	ld:
		regalloc(&nod, f->type, t);
		nod.type = t64? types[TINT64]: types[TINT32];
		gins(a, f, &nod);
		gmove(&nod, t);
		regfree(&nod);
		return;

	case TFLOAT32:
		a = AMOVSS;
		goto fld;
	case TFLOAT64:
		a = AMOVSD;
	fld:
		regalloc(&nod, f->type, t);
		if(tt != TFLOAT64 && tt != TFLOAT32){	/* TO DO: why is this here */
			dump("odd tree", f);
			nod.type = t64? types[TINT64]: types[TINT32];
		}
		gins(a, f, &nod);
		gmove(&nod, t);
		regfree(&nod);
		return;
	}

/*
 * store
 */
	if(t->op == ONAME || t->op == OINDREG ||
	   t->op == OIND || t->op == OINDEX)
	switch(tt) {
	case TBOOL:
	case TINT8:
	case TUINT8:
		a = AMOVB;
		goto st;
	case TINT16:
	case TUINT16:
		a = AMOVW;
		goto st;
	case TINT32:
	case TUINT32:
	case TPTR32:
		a = AMOVL;
		goto st;
	case TINT64:
	case TUINT64:
	case TPTR64:
		a = AMOVQ;
		goto st;

	st:
		if(f->op == OCONST) {
			gins(a, f, t);
			return;
		}
	fst:
		regalloc(&nod, t->type, f);
		gmove(f, &nod);
		gins(a, &nod, t);
		regfree(&nod);
		return;

	case TFLOAT32:
		a = AMOVSS;
		goto fst;
	case TFLOAT64:
		a = AMOVSD;
		goto fst;
	}

/*
 * convert
 */
	switch(CASE(ft, tt)) {
	default:
/*
 * integer to integer
 ********
 *		a = AGOK;	break;

 *	case CASE(TBOOL, TBOOL):
 *	case CASE(TINT8, TBOOL):
 *	case CASE(TUINT8, TBOOL):
 *	case CASE(TINT16, TBOOL):
 *	case CASE(TUINT16, TBOOL):
 *	case CASE(TINT32, TBOOL):
 *	case CASE(TUINT32, TBOOL):
 *	case CASE(TPTR64, TBOOL):

 *	case CASE(TBOOL, TINT8):
 *	case CASE(TINT8, TINT8):
 *	case CASE(TUINT8, TINT8):
 *	case CASE(TINT16, TINT8):
 *	case CASE(TUINT16, TINT8):
 *	case CASE(TINT32, TINT8):
 *	case CASE(TUINT32, TINT8):
 *	case CASE(TPTR64, TINT8):

 *	case CASE(TBOOL, TUINT8):
 *	case CASE(TINT8, TUINT8):
 *	case CASE(TUINT8, TUINT8):
 *	case CASE(TINT16, TUINT8):
 *	case CASE(TUINT16, TUINT8):
 *	case CASE(TINT32, TUINT8):
 *	case CASE(TUINT32, TUINT8):
 *	case CASE(TPTR64, TUINT8):

 *	case CASE(TINT16, TINT16):
 *	case CASE(TUINT16, TINT16):
 *	case CASE(TINT32, TINT16):
 *	case CASE(TUINT32, TINT16):
 *	case CASE(TPTR64, TINT16):

 *	case CASE(TINT16, TUINT16):
 *	case CASE(TUINT16, TUINT16):
 *	case CASE(TINT32, TUINT16):
 *	case CASE(TUINT32, TUINT16):
 *	case CASE(TPTR64, TUINT16):

 *	case CASE(TINT64, TUINT):
 *	case CASE(TINT64, TUINT32):
 *	case CASE(TUINT64, TUINT32):
 *****/
		a = AMOVL;
		break;

	case CASE(TINT64, TINT8):
	case CASE(TINT64, TINT16):
	case CASE(TINT64, TINT32):
	case CASE(TUINT64, TINT8):
	case CASE(TUINT64, TINT16):
	case CASE(TUINT64, TINT32):
		a = AMOVLQSX;		// this looks bad
		break;

	case CASE(TINT32, TINT64):
	case CASE(TINT32, TPTR64):
		a = AMOVLQSX;
		break;

	case CASE(TUINT32, TINT64):
	case CASE(TUINT32, TUINT64):
	case CASE(TUINT32, TPTR64):
	case CASE(TPTR32, TINT64):
	case CASE(TPTR32, TUINT64):
	case CASE(TPTR32, TPTR64):
		a = AMOVLQZX;
		break;

	case CASE(TPTR64, TINT64):
	case CASE(TINT64, TINT64):
	case CASE(TUINT64, TINT64):
	case CASE(TINT64, TUINT64):
	case CASE(TUINT64, TUINT64):
	case CASE(TPTR64, TUINT64):
	case CASE(TINT64, TPTR64):
	case CASE(TUINT64, TPTR64):
	case CASE(TPTR64, TPTR64):
		a = AMOVQ;
		break;

	case CASE(TINT16, TINT32):
	case CASE(TINT16, TUINT32):
		a = AMOVWLSX;
//		if(f->op == OCONST) {
//			f->val.vval &= 0xffff;
//			if(f->val.vval & 0x8000)
//				f->val.vval |= 0xffff0000;
//			a = AMOVL;
//		}
		break;

	case CASE(TINT16, TINT64):
	case CASE(TINT16, TUINT64):
	case CASE(TINT16, TPTR64):
		a = AMOVWQSX;
//		if(f->op == OCONST) {
//			f->val.vval &= 0xffff;
//			if(f->val.vval & 0x8000){
//				f->val.vval |= 0xffff0000;
//				f->val.vval |= (vlong)~0 << 32;
//			}
//			a = AMOVL;
//		}
		break;

	case CASE(TUINT16, TINT32):
	case CASE(TUINT16, TUINT32):
		a = AMOVWLZX;
//		if(f->op == OCONST) {
//			f->val.vval &= 0xffff;
//			a = AMOVL;
//		}
		break;

	case CASE(TUINT16, TINT64):
	case CASE(TUINT16, TUINT64):
	case CASE(TUINT16, TPTR64):
		a = AMOVWQZX;
//		if(f->op == OCONST) {
//			f->val.vval &= 0xffff;
//			a = AMOVL;	/* MOVL also zero-extends to 64 bits */
//		}
		break;

	case CASE(TINT8, TINT16):
	case CASE(TINT8, TUINT16):
	case CASE(TINT8, TINT32):
	case CASE(TINT8, TUINT32):
		a = AMOVBLSX;
//		if(f->op == OCONST) {
//			f->val.vval &= 0xff;
//			if(f->val.vval & 0x80)
//				f->val.vval |= 0xffffff00;
//			a = AMOVL;
//		}
		break;

	case CASE(TINT8, TINT64):
	case CASE(TINT8, TUINT64):
	case CASE(TINT8, TPTR64):
		a = AMOVBQSX;
//		if(f->op == OCONST) {
//			f->val.vval &= 0xff;
//			if(f->val.vval & 0x80){
//				f->val.vval |= 0xffffff00;
//				f->val.vval |= (vlong)~0 << 32;
//			}
//			a = AMOVQ;
//		}
		break;

	case CASE(TBOOL, TINT16):
	case CASE(TBOOL, TUINT16):
	case CASE(TBOOL, TINT32):
	case CASE(TBOOL, TUINT32):
	case CASE(TUINT8, TINT16):
	case CASE(TUINT8, TUINT16):
	case CASE(TUINT8, TINT32):
	case CASE(TUINT8, TUINT32):
		a = AMOVBLZX;
//		if(f->op == OCONST) {
//			f->val.vval &= 0xff;
//			a = AMOVL;
//		}
		break;

	case CASE(TBOOL, TINT64):
	case CASE(TBOOL, TUINT64):
	case CASE(TBOOL, TPTR64):
	case CASE(TUINT8, TINT64):
	case CASE(TUINT8, TUINT64):
	case CASE(TUINT8, TPTR64):
		a = AMOVBQZX;
//		if(f->op == OCONST) {
//			f->val.vval &= 0xff;
//			a = AMOVL;	/* zero-extends to 64-bits */
//		}
		break;

/*
 * float to fix
 */
	case CASE(TFLOAT32, TINT8):
	case CASE(TFLOAT32, TINT16):
	case CASE(TFLOAT32, TINT32):
		regalloc(&nod, t->type, N);
		gins(ACVTTSS2SL, f, &nod);
		gmove(&nod, t);
		regfree(&nod);
		return;

	case CASE(TFLOAT32, TBOOL):
	case CASE(TFLOAT32, TUINT8):
	case CASE(TFLOAT32, TUINT16):
	case CASE(TFLOAT32, TUINT32):
	case CASE(TFLOAT32, TINT64):
	case CASE(TFLOAT32, TUINT64):
	case CASE(TFLOAT32, TPTR64):
		regalloc(&nod, t->type, N);
		gins(ACVTTSS2SQ, f, &nod);
		gmove(&nod, t);
		regfree(&nod);
		return;

	case CASE(TFLOAT64, TINT8):
	case CASE(TFLOAT64, TINT16):
	case CASE(TFLOAT64, TINT32):
		regalloc(&nod, t->type, N);
		gins(ACVTTSD2SL, f, &nod);
		gmove(&nod, t);
		regfree(&nod);
		return;

	case CASE(TFLOAT64, TBOOL):
	case CASE(TFLOAT64, TUINT8):
	case CASE(TFLOAT64, TUINT16):
	case CASE(TFLOAT64, TUINT32):
	case CASE(TFLOAT64, TINT64):
	case CASE(TFLOAT64, TUINT64):
	case CASE(TFLOAT64, TPTR64):
		regalloc(&nod, t->type, N);
		gins(ACVTTSD2SQ, f, &nod);
		gmove(&nod, t);
		regfree(&nod);
		return;

/*
 * uvlong to float
 */
	case CASE(TUINT64, TFLOAT64):
	case CASE(TUINT64, TFLOAT32):
		a = ACVTSQ2SS;
		if(tt == TFLOAT64)
			a = ACVTSQ2SD;
		regalloc(&nod, f->type, f);
		gmove(f, &nod);
		regalloc(&nod1, t->type, t);
		nodconst(&nodc, types[TUINT64], 0);
		gins(ACMPQ, &nod, &nodc);
		p1 = gbranch(AJLT, T);
		gins(a, &nod, &nod1);
		p2 = gbranch(AJMP, T);
		patch(p1, pc);
		regalloc(&nod2, f->type, N);
		regalloc(&nod3, f->type, N);
		gmove(&nod, &nod2);
		nodconst(&nodc, types[TUINT64], 1);
		gins(ASHRQ, &nodc, &nod2);
		gmove(&nod, &nod3);
		gins(AANDL, &nodc, &nod3);
		gins(AORQ, &nod3, &nod2);
		gins(a, &nod2, &nod1);
		gins(tt == TFLOAT64? AADDSD: AADDSS, &nod1, &nod1);
		regfree(&nod2);
		regfree(&nod3);
		patch(p2, pc);
		regfree(&nod);
		regfree(&nod1);
		return;

	case CASE(TUINT32, TFLOAT64):
	case CASE(TUINT32, TFLOAT32):
		a = ACVTSQ2SS;
		if(tt == TFLOAT64)
			a = ACVTSQ2SD;
		regalloc(&nod, f->type, f);
		gins(AMOVLQZX, f, &nod);
		regalloc(&nod1, t->type, t);
		gins(a, &nod, &nod1);
		gmove(&nod1, t);
		regfree(&nod);
		regfree(&nod1);
		return;

/*
 * fix to float
 */
	case CASE(TINT64, TFLOAT32):
	case CASE(TPTR64, TFLOAT32):
		regalloc(&nod, t->type, t);
		gins(ACVTSQ2SS, f, &nod);
		gmove(&nod, t);
		regfree(&nod);
		return;

	case CASE(TINT64, TFLOAT64):
	case CASE(TPTR64, TFLOAT64):
		regalloc(&nod, t->type, t);
		gins(ACVTSQ2SD, f, &nod);
		gmove(&nod, t);
		regfree(&nod);
		return;

	case CASE(TBOOL, TFLOAT32):
	case CASE(TINT8, TFLOAT32):
	case CASE(TUINT8, TFLOAT32):
	case CASE(TINT16, TFLOAT32):
	case CASE(TUINT16, TFLOAT32):
	case CASE(TINT32, TFLOAT32):
		regalloc(&nod, t->type, t);
		gins(ACVTSL2SS, f, &nod);
		gmove(&nod, t);
		regfree(&nod);
		return;

	case CASE(TBOOL, TFLOAT64):
	case CASE(TINT8, TFLOAT64):
	case CASE(TUINT8, TFLOAT64):
	case CASE(TINT16, TFLOAT64):
	case CASE(TUINT16, TFLOAT64):
	case CASE(TINT32, TFLOAT64):
		regalloc(&nod, t->type, t);
		gins(ACVTSL2SD, f, &nod);
		gmove(&nod, t);
		regfree(&nod);
		return;

/*
 * float to float
 */
	case CASE(TFLOAT32, TFLOAT32):
		a = AMOVSS;
		break;
	case CASE(TFLOAT64, TFLOAT32):
		a = ACVTSD2SS;
		break;
	case CASE(TFLOAT32, TFLOAT64):
		a = ACVTSS2SD;
		break;
	case CASE(TFLOAT64, TFLOAT64):
		a = AMOVSD;
		break;
	}
	if(a == AMOVQ ||
	   a == AMOVSD ||
	   a == AMOVSS ||
	   (a == AMOVL && f->type->width == t->type->width))	/* TO DO: check AMOVL */
		if(samaddr(f, t))
			return;
	gins(a, f, t);
}

void
regsalloc(Node *f, Type *t)
{
	fatal("regsalloc");
}

int
samaddr(Node *f, Node *t)
{

	if(f->op != t->op)
		return 0;

	switch(f->op) {
	case OREGISTER:
		if(f->val.u.reg != t->val.u.reg)
			break;
		return 1;
	}
	return 0;
}

Prog*
gins(int as, Node *f, Node *t)
{
//	Node nod;
//	int32 v;
	Prog *p;

//	if(f != N && f->op == OINDEX) {
//		regalloc(&nod, &regnode, Z);
//		v = constnode.vconst;
//		cgen(f->right, &nod);
//		constnode.vconst = v;
//		idx.reg = nod.reg;
//		regfree(&nod);
//	}
//	if(t != N && t->op == OINDEX) {
//		regalloc(&nod, &regnode, Z);
//		v = constnode.vconst;
//		cgen(t->right, &nod);
//		constnode.vconst = v;
//		idx.reg = nod.reg;
//		regfree(&nod);
//	}

	p = prog(as);
	if(f != N)
		naddr(f, &p->from);
	if(t != N)
		naddr(t, &p->to);
	if(debug['g'])
		print("%P\n", p);
	return p;
}

void
naddr(Node *n, Addr *a)
{

	a->type = D_NONE;
	if(n == N)
		return;

	switch(n->op) {
	default:
		fatal("naddr: bad %O %D", n->op, a);
		break;

	case OREGISTER:
		a->type = n->val.u.reg;
		a->sym = S;
		break;

//	case OINDEX:
//	case OIND:
//		naddr(n->left, a);
//		if(a->type >= D_AX && a->type <= D_DI)
//			a->type += D_INDIR;
//		else
//		if(a->type == D_CONST)
//			a->type = D_NONE+D_INDIR;
//		else
//		if(a->type == D_ADDR) {
//			a->type = a->index;
//			a->index = D_NONE;
//		} else
//			goto bad;
//		if(n->op == OINDEX) {
//			a->index = idx.reg;
//			a->scale = n->scale;
//		}
//		break;

	case OINDREG:
		a->type = n->val.u.reg+D_INDIR;
		a->sym = n->sym;
		a->offset = n->xoffset;
		break;

	case ONAME:
		a->etype = 0;
		if(n->type != T)
			a->etype = n->type->etype;
		a->offset = n->xoffset;
		a->sym = n->sym;
		if(a->sym == S)
			a->sym = lookup(".noname");
		if(n->method) {
			if(n->type != T)
			if(n->type->sym != S)
			if(n->type->sym->opackage != nil)
				a->sym = pkglookup(a->sym->name, n->type->sym->opackage);
		}

		switch(n->class) {
		default:
			fatal("naddr: ONAME class %S %d\n", n->sym, n->class);
		case PEXTERN:
			a->type = D_EXTERN;
			break;
		case PAUTO:
			a->type = D_AUTO;
			break;
		case PPARAM:
			a->type = D_PARAM;
			break;
		case PSTATIC:
			a->type = D_STATIC;
			break;
		}
		break;

	case OLITERAL:
		switch(n->val.ctype) {
		default:
			fatal("naddr: const %lT", n->type);
			break;
		case CTFLT:
			a->type = D_FCONST;
			a->dval = mpgetflt(n->val.u.fval);
			break;
		case CTINT:
		case CTSINT:
		case CTUINT:
			a->sym = S;
			a->type = D_CONST;
			a->offset = mpgetfix(n->val.u.xval);
			break;
		case CTSTR:
			a->etype = n->etype;
			a->sym = symstringo;
			a->type = D_ADDR;
			a->index = D_STATIC;
			a->offset = symstringo->offset;
			stringpool(n);
			break;
		case CTBOOL:
			a->sym = S;
			a->type = D_CONST;
			a->offset = n->val.u.bval;
			break;
		case CTNIL:
			a->sym = S;
			a->type = D_CONST;
			a->offset = 0;
			break;
		}
		break;

	case OADDR:
		naddr(n->left, a);
		if(a->type >= D_INDIR) {
			a->type -= D_INDIR;
			break;
		}
		if(a->type == D_EXTERN || a->type == D_STATIC ||
		   a->type == D_AUTO || a->type == D_PARAM)
			if(a->index == D_NONE) {
				a->index = a->type;
				a->type = D_ADDR;
				break;
			}
		fatal("naddr: OADDR\n");

//	case OADD:
//		if(n->right->op == OLITERAL) {
//			v = n->right->vconst;
//			naddr(n->left, a);
//		} else
//		if(n->left->op == OLITERAL) {
//			v = n->left->vconst;
//			naddr(n->right, a);
//		} else
//			goto bad;
//		a->offset += v;
//		break;

	}
}

int
optoas(int op, Type *t)
{
	int a;

	if(t == T)
		fatal("optoas: t is nil");

	a = AGOK;
	switch(CASE(op, simtype[t->etype])) {
	default:
		fatal("optoas: no entry %O-%T", op, t);
		break;

	case CASE(OADDR, TPTR32):
		a = ALEAL;
		break;

	case CASE(OADDR, TPTR64):
		a = ALEAQ;
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
		a = AJEQ;
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
		a = AJNE;
		break;

	case CASE(OLT, TINT8):
	case CASE(OLT, TINT16):
	case CASE(OLT, TINT32):
	case CASE(OLT, TINT64):
		a = AJLT;
		break;

	case CASE(OLT, TUINT8):
	case CASE(OLT, TUINT16):
	case CASE(OLT, TUINT32):
	case CASE(OLT, TUINT64):
	case CASE(OGT, TFLOAT32):
	case CASE(OGT, TFLOAT64):
		a = AJCS;
		break;

	case CASE(OLE, TINT8):
	case CASE(OLE, TINT16):
	case CASE(OLE, TINT32):
	case CASE(OLE, TINT64):
		a = AJLE;
		break;

	case CASE(OLE, TUINT8):
	case CASE(OLE, TUINT16):
	case CASE(OLE, TUINT32):
	case CASE(OLE, TUINT64):
	case CASE(OGE, TFLOAT32):
	case CASE(OGE, TFLOAT64):
		a = AJLS;
		break;

	case CASE(OGT, TINT8):
	case CASE(OGT, TINT16):
	case CASE(OGT, TINT32):
	case CASE(OGT, TINT64):
		a = AJGT;
		break;

	case CASE(OGT, TUINT8):
	case CASE(OGT, TUINT16):
	case CASE(OGT, TUINT32):
	case CASE(OGT, TUINT64):
	case CASE(OLT, TFLOAT32):
	case CASE(OLT, TFLOAT64):
		a = AJHI;
		break;

	case CASE(OGE, TINT8):
	case CASE(OGE, TINT16):
	case CASE(OGE, TINT32):
	case CASE(OGE, TINT64):
		a = AJGE;
		break;

	case CASE(OGE, TUINT8):
	case CASE(OGE, TUINT16):
	case CASE(OGE, TUINT32):
	case CASE(OGE, TUINT64):
	case CASE(OLE, TFLOAT32):
	case CASE(OLE, TFLOAT64):
		a = AJCC;
		break;

	case CASE(OCMP, TBOOL):
	case CASE(OCMP, TINT8):
	case CASE(OCMP, TUINT8):
		a = ACMPB;
		break;

	case CASE(OCMP, TINT16):
	case CASE(OCMP, TUINT16):
		a = ACMPW;
		break;

	case CASE(OCMP, TINT32):
	case CASE(OCMP, TUINT32):
	case CASE(OCMP, TPTR32):
		a = ACMPL;
		break;

	case CASE(OCMP, TINT64):
	case CASE(OCMP, TUINT64):
	case CASE(OCMP, TPTR64):
		a = ACMPQ;
		break;

	case CASE(OCMP, TFLOAT32):
		a = AUCOMISS;
		break;

	case CASE(OCMP, TFLOAT64):
		a = AUCOMISD;
		break;

	case CASE(OADD, TINT8):
	case CASE(OADD, TUINT8):
		a = AADDB;
		break;

	case CASE(OADD, TINT16):
	case CASE(OADD, TUINT16):
		a = AADDW;
		break;

	case CASE(OADD, TINT32):
	case CASE(OADD, TUINT32):
	case CASE(OADD, TPTR32):
		a = AADDL;
		break;

	case CASE(OADD, TINT64):
	case CASE(OADD, TUINT64):
	case CASE(OADD, TPTR64):
		a = AADDQ;
		break;

	case CASE(OADD, TFLOAT32):
		a = AADDSS;
		break;

	case CASE(OADD, TFLOAT64):
		a = AADDSD;
		break;

	case CASE(OSUB, TINT8):
	case CASE(OSUB, TUINT8):
		a = ASUBB;
		break;

	case CASE(OSUB, TINT16):
	case CASE(OSUB, TUINT16):
		a = ASUBW;
		break;

	case CASE(OSUB, TINT32):
	case CASE(OSUB, TUINT32):
	case CASE(OSUB, TPTR32):
		a = ASUBL;
		break;

	case CASE(OSUB, TINT64):
	case CASE(OSUB, TUINT64):
	case CASE(OSUB, TPTR64):
		a = ASUBQ;
		break;

	case CASE(OSUB, TFLOAT32):
		a = ASUBSS;
		break;

	case CASE(OSUB, TFLOAT64):
		a = ASUBSD;
		break;

	case CASE(OINC, TINT8):
	case CASE(OINC, TUINT8):
		a = AINCB;
		break;

	case CASE(OINC, TINT16):
	case CASE(OINC, TUINT16):
		a = AINCW;
		break;

	case CASE(OINC, TINT32):
	case CASE(OINC, TUINT32):
	case CASE(OINC, TPTR32):
		a = AINCL;
		break;

	case CASE(OINC, TINT64):
	case CASE(OINC, TUINT64):
	case CASE(OINC, TPTR64):
		a = AINCQ;
		break;

	case CASE(ODEC, TINT8):
	case CASE(ODEC, TUINT8):
		a = ADECB;
		break;

	case CASE(ODEC, TINT16):
	case CASE(ODEC, TUINT16):
		a = ADECW;
		break;

	case CASE(ODEC, TINT32):
	case CASE(ODEC, TUINT32):
	case CASE(ODEC, TPTR32):
		a = ADECL;
		break;

	case CASE(ODEC, TINT64):
	case CASE(ODEC, TUINT64):
	case CASE(ODEC, TPTR64):
		a = ADECQ;
		break;

	case CASE(OMINUS, TINT8):
	case CASE(OMINUS, TUINT8):
		a = ANEGB;
		break;

	case CASE(OMINUS, TINT16):
	case CASE(OMINUS, TUINT16):
		a = ANEGW;
		break;

	case CASE(OMINUS, TINT32):
	case CASE(OMINUS, TUINT32):
	case CASE(OMINUS, TPTR32):
		a = ANEGL;
		break;

	case CASE(OMINUS, TINT64):
	case CASE(OMINUS, TUINT64):
	case CASE(OMINUS, TPTR64):
		a = ANEGQ;
		break;

	case CASE(OAND, TINT8):
	case CASE(OAND, TUINT8):
		a = AANDB;
		break;

	case CASE(OAND, TINT16):
	case CASE(OAND, TUINT16):
		a = AANDW;
		break;

	case CASE(OAND, TINT32):
	case CASE(OAND, TUINT32):
	case CASE(OAND, TPTR32):
		a = AANDL;
		break;

	case CASE(OAND, TINT64):
	case CASE(OAND, TUINT64):
	case CASE(OAND, TPTR64):
		a = AANDQ;
		break;

	case CASE(OOR, TINT8):
	case CASE(OOR, TUINT8):
		a = AORB;
		break;

	case CASE(OOR, TINT16):
	case CASE(OOR, TUINT16):
		a = AORW;
		break;

	case CASE(OOR, TINT32):
	case CASE(OOR, TUINT32):
	case CASE(OOR, TPTR32):
		a = AORL;
		break;

	case CASE(OOR, TINT64):
	case CASE(OOR, TUINT64):
	case CASE(OOR, TPTR64):
		a = AORQ;
		break;

	case CASE(OXOR, TINT8):
	case CASE(OXOR, TUINT8):
		a = AXORB;
		break;

	case CASE(OXOR, TINT16):
	case CASE(OXOR, TUINT16):
		a = AXORW;
		break;

	case CASE(OXOR, TINT32):
	case CASE(OXOR, TUINT32):
	case CASE(OXOR, TPTR32):
		a = AXORL;
		break;

	case CASE(OXOR, TINT64):
	case CASE(OXOR, TUINT64):
	case CASE(OXOR, TPTR64):
		a = AXORQ;
		break;

	case CASE(OLSH, TINT8):
	case CASE(OLSH, TUINT8):
		a = ASHLB;
		break;

	case CASE(OLSH, TINT16):
	case CASE(OLSH, TUINT16):
		a = ASHLW;
		break;

	case CASE(OLSH, TINT32):
	case CASE(OLSH, TUINT32):
	case CASE(OLSH, TPTR32):
		a = ASHLL;
		break;

	case CASE(OLSH, TINT64):
	case CASE(OLSH, TUINT64):
	case CASE(OLSH, TPTR64):
		a = ASHLQ;
		break;

	case CASE(ORSH, TUINT8):
		a = ASHRB;
		break;

	case CASE(ORSH, TUINT16):
		a = ASHRW;
		break;

	case CASE(ORSH, TUINT32):
	case CASE(ORSH, TPTR32):
		a = ASHRL;
		break;

	case CASE(ORSH, TUINT64):
	case CASE(ORSH, TPTR64):
		a = ASHRQ;
		break;

	case CASE(ORSH, TINT8):
		a = ASARB;
		break;

	case CASE(ORSH, TINT16):
		a = ASARW;
		break;

	case CASE(ORSH, TINT32):
		a = ASARL;
		break;

	case CASE(ORSH, TINT64):
		a = ASARQ;
		break;

	case CASE(OMUL, TINT8):
	case CASE(OMUL, TUINT8):
		a = AIMULB;
		break;

	case CASE(OMUL, TINT16):
	case CASE(OMUL, TUINT16):
		a = AIMULW;
		break;

	case CASE(OMUL, TINT32):
	case CASE(OMUL, TUINT32):
	case CASE(OMUL, TPTR32):
		a = AIMULL;
		break;

	case CASE(OMUL, TINT64):
	case CASE(OMUL, TUINT64):
	case CASE(OMUL, TPTR64):
		a = AIMULQ;
		break;

	case CASE(OMUL, TFLOAT32):
		a = AMULSS;
		break;

	case CASE(OMUL, TFLOAT64):
		a = AMULSD;
		break;

	case CASE(ODIV, TINT8):
	case CASE(OMOD, TINT8):
		a = AIDIVB;
		break;

	case CASE(ODIV, TUINT8):
	case CASE(OMOD, TUINT8):
		a = ADIVB;
		break;

	case CASE(ODIV, TINT16):
	case CASE(OMOD, TINT16):
		a = AIDIVW;
		break;

	case CASE(ODIV, TUINT16):
	case CASE(OMOD, TUINT16):
		a = ADIVW;
		break;

	case CASE(ODIV, TINT32):
	case CASE(OMOD, TINT32):
		a = AIDIVL;
		break;

	case CASE(ODIV, TUINT32):
	case CASE(ODIV, TPTR32):
	case CASE(OMOD, TUINT32):
	case CASE(OMOD, TPTR32):
		a = ADIVL;
		break;

	case CASE(ODIV, TINT64):
	case CASE(OMOD, TINT64):
		a = AIDIVQ;
		break;

	case CASE(ODIV, TUINT64):
	case CASE(ODIV, TPTR64):
	case CASE(OMOD, TUINT64):
	case CASE(OMOD, TPTR64):
		a = ADIVQ;
		break;

	case CASE(OFOR, TINT16):
		a = ACWD;
		break;

	case CASE(OFOR, TINT32):
		a = ACDQ;
		break;

	case CASE(OFOR, TINT64):
		a = ACQO;
		break;

	case CASE(ODIV, TFLOAT32):
		a = ADIVSS;
		break;

	case CASE(ODIV, TFLOAT64):
		a = ADIVSD;
		break;

	}
	return a;
}

int
isfat(Type *t)
{
	if(t != T)
	switch(t->etype) {
	case TSTRUCT:
	case TARRAY:
	case TINTER:	// maybe remove later
	case TDDD:	// maybe remove later
		return 1;
	}
	return 0;
}

/*
 * return unsigned(op)
 * eg GT -> HS
 */
int
brunsigned(int a)
{
	switch(a) {
	case AJLT:	return AJGE;
	case AJGT:	return AJLE;
	case AJLE:	return AJGT;
	case AJGE:	return AJLT;
	}
	return a;
}

/*
 * return !(op)
 * eg == <=> !=
 */
int
brcom(int a)
{
	switch(a) {
	case OEQ:	return ONE;
	case ONE:	return OEQ;
	case OLT:	return OGE;
	case OGT:	return OLE;
	case OLE:	return OGT;
	case OGE:	return OLT;
	}
	fatal("brcom: no com for %A\n", a);
	return a;
}

/*
 * return reverse(op)
 * eg a op b <=> b r(op) a
 */
int
brrev(int a)
{
	switch(a) {
	case OEQ:	return OEQ;
	case ONE:	return ONE;
	case OLT:	return OGT;
	case OGT:	return OLT;
	case OLE:	return OGE;
	case OGE:	return OLE;
	}
	fatal("brcom: no rev for %A\n", a);
	return a;
}

/*
 * make a new off the books
 */
void
tempname(Node *n, Type *t)
{
	Sym *s;
	uint32 w;

	if(t == T) {
		yyerror("tempname called with nil type");
		t = types[TINT32];
	}

	s = lookup("!tmpname!");

	memset(n, 0, sizeof(*n));
	n->op = ONAME;
	n->sym = s;
	n->type = t;
	n->etype = t->etype;
	n->class = PAUTO;
	n->addable = 1;
	n->ullman = 1;

	dowidth(t);
	w = t->width;
	stksize += w;
	stksize = rnd(stksize, w);
	n->xoffset = -stksize;
}

void
stringpool(Node *n)
{
	Pool *p;
	int w;

	if(n->op != OLITERAL || n->val.ctype != CTSTR) {
		if(n->val.ctype == CTNIL)
			return;
		fatal("stringpool: not string %N", n);
	}

	p = mal(sizeof(*p));

	p->sval = n->val.u.sval;
	p->link = nil;

	if(poolist == nil)
		poolist = p;
	else
		poolast->link = p;
	poolast = p;

	w = types[TINT32]->width;
	symstringo->offset += w;		// len
	symstringo->offset += p->sval->len;	// str[len]
	symstringo->offset = rnd(symstringo->offset, w);
}

Sig*
lsort(Sig *l, int(*f)(Sig*, Sig*))
{
	Sig *l1, *l2, *le;

	if(l == 0 || l->link == 0)
		return l;

	l1 = l;
	l2 = l;
	for(;;) {
		l2 = l2->link;
		if(l2 == 0)
			break;
		l2 = l2->link;
		if(l2 == 0)
			break;
		l1 = l1->link;
	}

	l2 = l1->link;
	l1->link = 0;
	l1 = lsort(l, f);
	l2 = lsort(l2, f);

	/* set up lead element */
	if((*f)(l1, l2) < 0) {
		l = l1;
		l1 = l1->link;
	} else {
		l = l2;
		l2 = l2->link;
	}
	le = l;

	for(;;) {
		if(l1 == 0) {
			while(l2) {
				le->link = l2;
				le = l2;
				l2 = l2->link;
			}
			le->link = 0;
			break;
		}
		if(l2 == 0) {
			while(l1) {
				le->link = l1;
				le = l1;
				l1 = l1->link;
			}
			break;
		}
		if((*f)(l1, l2) < 0) {
			le->link = l1;
			le = l1;
			l1 = l1->link;
		} else {
			le->link = l2;
			le = l2;
			l2 = l2->link;
		}
	}
	le->link = 0;
	return l;
}

void
setmaxarg(Type *t)
{
	Type *to;
	int32 w;

	to = *getoutarg(t);
	w = to->width;
	if(w > maxarg)
		maxarg = w;
}
