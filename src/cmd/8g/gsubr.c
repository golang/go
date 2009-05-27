// Derived from Inferno utils/8c/txt.c
// http://code.google.com/p/inferno-os/source/browse/utils/8c/txt.c
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

#define	CASE(a,b)	(((a)<<16)|((b)<<0))

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

/*
 * generate and return proc with p->as = as,
 * linked into program.  pc is next instruction.
 */
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

/*
 * generate a branch.
 * t is ignored.
 */
Prog*
gbranch(int as, Type *t)
{
	Prog *p;

	p = prog(as);
	p->to.type = D_BRANCH;
	p->to.branch = P;
	return p;
}

/*
 * patch previous branch to jump to to.
 */
void
patch(Prog *p, Prog *to)
{
	if(p->to.type != D_BRANCH)
		fatal("patch: not a branch");
	p->to.branch = to;
	p->to.offset = to->loc;
}

/*
 * start a new Prog list.
 */
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
gused(Node *n)
{
	gins(ANOP, n, N);	// used
}

Prog*
gjmp(Prog *to)
{
	Prog *p;

	p = gbranch(AJMP, T);
	if(to != P)
		patch(p, to);
	return p;
}

void
ggloblnod(Node *nam, int32 width)
{
	Prog *p;

	p = gins(AGLOBL, nam, N);
	p->lineno = nam->lineno;
	p->to.sym = S;
	p->to.type = D_CONST;
	p->to.offset = width;
}

void
ggloblsym(Sym *s, int32 width, int dupok)
{
	Prog *p;

	p = gins(AGLOBL, N, N);
	p->from.type = D_EXTERN;
	p->from.index = D_NONE;
	p->from.sym = s;
	p->to.type = D_CONST;
	p->to.index = D_NONE;
	p->to.offset = width;
	if(dupok)
		p->from.scale = DUPOK;
}

int
isfat(Type *t)
{
	if(t != T)
	switch(t->etype) {
	case TSTRUCT:
	case TARRAY:
	case TSTRING:
	case TINTER:	// maybe remove later
	case TDDD:	// maybe remove later
		return 1;
	}
	return 0;
}

/*
 * naddr of func generates code for address of func.
 * if using opcode that can take address implicitly,
 * call afunclit to fix up the argument.
 */
void
afunclit(Addr *a)
{
	if(a->type == D_ADDR && a->index == D_EXTERN) {
		a->type = D_EXTERN;
		a->index = D_NONE;
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

	a = AGOK;
	switch(CASE(op, simtype[t->etype])) {
	default:
		fatal("optoas: no entry %O-%T", op, t);
		break;

	case CASE(OADDR, TPTR32):
		a = ALEAL;
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

	case CASE(OAS, TBOOL):
	case CASE(OAS, TINT8):
	case CASE(OAS, TUINT8):
		a = AMOVB;
		break;

	case CASE(OAS, TINT16):
	case CASE(OAS, TUINT16):
		a = AMOVW;
		break;

	case CASE(OAS, TINT32):
	case CASE(OAS, TUINT32):
	case CASE(OAS, TPTR32):
		a = AMOVL;
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

	case CASE(ORSH, TINT8):
		a = ASARB;
		break;

	case CASE(ORSH, TINT16):
		a = ASARW;
		break;

	case CASE(ORSH, TINT32):
		a = ASARL;
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

	case CASE(OEXTEND, TINT16):
		a = ACWD;
		break;

	case CASE(OEXTEND, TINT32):
		a = ACDQ;
		break;
	}
	return a;
}

static	int	resvd[] =
{
//	D_DI,	// for movstring
//	D_SI,	// for movstring

	D_AX,	// for divide
	D_CX,	// for shift
	D_DX,	// for divide
	D_SP,	// for stack
};

void
ginit(void)
{
	int i;

	for(i=0; i<nelem(reg); i++)
		reg[i] = 1;
	for(i=D_AX; i<=D_DI; i++)
		reg[i] = 0;

	// TODO: Use MMX ?
	for(i=D_F0; i<=D_F7; i++)
		reg[i] = 0;

	for(i=0; i<nelem(resvd); i++)
		reg[resvd[i]]++;
}

void
gclean(void)
{
	int i;

	for(i=0; i<nelem(resvd); i++)
		reg[resvd[i]]--;

	for(i=D_AX; i<=D_DI; i++)
		if(reg[i])
			yyerror("reg %R left allocated\n", i);
	for(i=D_F0; i<=D_F7; i++)
		if(reg[i])
			yyerror("reg %R left allocated\n", i);
}

ulong regpc[D_NONE];

/*
 * allocate register of type t, leave in n.
 * if o != N, o is desired fixed register.
 * caller must regfree(n).
 */
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
			if(i >= D_AX && i <= D_DI)
				goto out;
		}
		for(i=D_AX; i<=D_DI; i++)
			if(reg[i] == 0)
				goto out;

		fprint(2, "registers allocated at\n");
		for(i=D_AX; i<=D_DI; i++)
			fprint(2, "\t%R\t%#lux\n", i, regpc[i]);
		yyerror("out of fixed registers");
		goto err;

	case TFLOAT32:
	case TFLOAT64:
		if(o != N && o->op == OREGISTER) {
			i = o->val.u.reg;
			if(i >= D_F0 && i <= D_F7)
				goto out;
		}
		for(i=D_F0; i<=D_F7; i++)
			if(reg[i] == 0)
				goto out;
		yyerror("out of floating registers");
		goto err;
	}
	yyerror("regalloc: unknown type %T", t);
	i = 0;

err:
	nodreg(n, t, 0);
	return;

out:
	if(reg[i] == 0) {
		regpc[i] = getcallerpc(&n);
		if(i == D_AX || i == D_CX || i == D_DX || i == D_SP) {
			dump("regalloc-o", o);
			fatal("regalloc %R", i);
		}
	}
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
	if(reg[i] == 0 && (i == D_AX || i == D_CX || i == D_DX || i == D_SP))
		fatal("regfree %R", i);
}

void
tempalloc(Node *n, Type *t)
{
	int w;

	dowidth(t);

	memset(n, 0, sizeof(*n));
	n->op = ONAME;
	n->sym = S;
	n->type = t;
	n->etype = t->etype;
	n->class = PAUTO;
	n->addable = 1;
	n->ullman = 1;
	n->noescape = 1;
	n->ostk = stksize;

	w = t->width;
	stksize += w;
	stksize = rnd(stksize, w);
	n->xoffset = -stksize;
	if(stksize > maxstksize)
		maxstksize = stksize;
}

void
tempfree(Node *n)
{
	if(n->xoffset != -stksize)
		fatal("tempfree %lld %d", -n->xoffset, stksize);
	stksize = n->ostk;
}

/*
 * initialize n to be register r of type t.
 */
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

/*
 * initialize n to be indirect of register r; n is type t.
 */
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
	Type *first;
	Iter savet;

	// entire argument struct, not just one arg
	switch(t->etype) {
	default:
		fatal("nodarg %T", t);

	case TSTRUCT:
		if(!t->funarg)
			fatal("nodarg: TSTRUCT but not funarg");
		n = nod(ONAME, N, N);
		n->sym = lookup(".args");
		n->type = t;
		first = structfirst(&savet, &t);
		if(first == nil)
			fatal("nodarg: bad struct");
		if(first->width == BADWIDTH)
			fatal("nodarg: offset not computed for %T", t);
		n->xoffset = first->width;
		n->addable = 1;
		break;

	case TFIELD:
		n = nod(ONAME, N, N);
		n->type = t->type;
		n->sym = t->sym;
		if(t->width == BADWIDTH)
			fatal("nodarg: offset not computed for %T", t);
		n->xoffset = t->width;
		n->addable = 1;
		break;
	}

	switch(fp) {
	default:
		fatal("nodarg %T %d", t, fp);

	case 0:		// output arg
		n->op = OINDREG;
		n->val.u.reg = D_SP;
		break;

	case 1:		// input arg
		n->class = PPARAM;
		break;
	}
	return n;
}

/*
 * generate
 *	as $c, reg
 */
void
gconreg(int as, vlong c, int reg)
{
	Node n1, n2;

	nodconst(&n1, types[TINT64], c);
	nodreg(&n2, types[TINT64], reg);
	gins(as, &n1, &n2);
}

/*
 * generate move:
 *	t = f
 * f may be in memory,
 * t is known to be a 32-bit register.
 */
void
gload(Node *f, Node *t)
{
	int a, ft;

	ft = simtype[f->type->etype];

	switch(ft) {
	default:
		fatal("gload %T", f->type);
	case TINT8:
		a = AMOVBLSX;
		if(isconst(f, CTINT) || isconst(f, CTBOOL))
			a = AMOVL;
		break;
	case TBOOL:
	case TUINT8:
		a = AMOVBLZX;
		if(isconst(f, CTINT) || isconst(f, CTBOOL))
			a = AMOVL;
		break;
	case TINT16:
		a = AMOVWLSX;
		if(isconst(f, CTINT) || isconst(f, CTBOOL))
			a = AMOVL;
		break;
	case TUINT16:
		a = AMOVWLZX;
		if(isconst(f, CTINT))
			a = AMOVL;
		break;
	case TINT32:
	case TUINT32:
	case TPTR32:
		a = AMOVL;
		break;
	case TINT64:
	case TUINT64:
		a = AMOVL;	// truncating
		break;
	}

	gins(a, f, t);
}

/*
 * generate move:
 *	t = f
 * f is known to be a 32-bit register.
 * t may be in memory.
 */
void
gstore(Node *f, Node *t)
{
	int a, ft, tt;
	Node nod, adr;

	ft = simtype[f->type->etype];
	tt = simtype[t->type->etype];

	switch(tt) {
	default:
		fatal("gstore %T", t->type);
	case TINT8:
	case TBOOL:
	case TUINT8:
		a = AMOVB;
		break;
	case TINT16:
	case TUINT16:
		a = AMOVW;
		break;
	case TINT32:
	case TUINT32:
	case TPTR32:
		a = AMOVL;
		break;
	case TINT64:
	case TUINT64:
		if(t->op == OREGISTER)
			fatal("gstore %T %O", t->type, t->op);
		memset(&adr, 0, sizeof adr);
		igen(t, &adr, N);
		t = &adr;
		t->xoffset += 4;
		switch(ft) {
		default:
			fatal("gstore %T -> %T", f, t);
			break;
		case TINT32:
			nodreg(&nod, types[TINT32], D_AX);
			gins(AMOVL, f, &nod);
			gins(ACDQ, N, N);
			nodreg(&nod, types[TINT32], D_DX);
			gins(AMOVL, &nod, t);
			break;
		case TUINT32:
			gins(AMOVL, nodintconst(0), t);
			break;
		}
		t->xoffset -= 4;
		a = AMOVL;
	}

	gins(a, f, t);
	if(t == &adr)
		regfree(&adr);
}

void
gmove(Node *f, Node *t)
{
	int ft, tt, t64, a;
	Node nod;

	ft = simtype[f->type->etype];
	tt = simtype[t->type->etype];

	a = AGOK;

	t64 = 0;
	if(tt == TINT64 || tt == TUINT64 || tt == TPTR64)
		t64 = 1;

	if(debug['M'])
		print("gop: %O %O[%E],%O[%E]\n", OAS,
			f->op, ft, t->op, tt);
	if(isfloat[ft] && f->op == OCONST) {
		fatal("fp");
		/* TO DO: pick up special constants, possibly preloaded */
	//F
	/*
		if(mpgetflt(f->val.u.fval) == 0.0) {
			regalloc(&nod, t->type, t);
			gins(AXORPD, &nod, &nod);
			gmove(&nod, t);
			regfree(&nod);
			return;
		}
	*/
	}

	if(is64(types[ft]) && isconst(f, CTINT)) {
		f->type = types[TINT32];	// XXX check constant value, choose correct type
		ft = TINT32;
	}

	if(is64(types[ft]) && is64(types[tt])) {
		sgen(f, t, 8);
		return;
	}

	regalloc(&nod, types[TINT32], t);
	gload(f, &nod);
	gstore(&nod, t);
	regfree(&nod);
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
/*
 * generate one instruction:
 *	as f, t
 */
Prog*
gins(int as, Node *f, Node *t)
{
	Prog *p;

	// generating AMOVL BX, BX is just dumb.
	if(f != N && t != N && samaddr(f, t) && as == AMOVL)
		return nil;

	p = prog(as);
	if(f != N)
		naddr(f, &p->from);
	if(t != N)
		naddr(t, &p->to);
	if(debug['g'])
		print("%P\n", p);
	return p;
}

/*
 * generate code to compute n;
 * make a refer to result.
 */
void
naddr(Node *n, Addr *a)
{
	a->scale = 0;
	a->index = D_NONE;
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

	case OINDREG:
		a->type = n->val.u.reg+D_INDIR;
		a->sym = n->sym;
		a->offset = n->xoffset;
		break;

	case OPARAM:
		// n->left is PHEAP ONAME for stack parameter.
		// compute address of actual parameter on stack.
		a->etype = n->left->type->etype;
		a->offset = n->xoffset;
		a->sym = n->left->sym;
		a->type = D_PARAM;
		break;

	case ONAME:
		a->etype = 0;
		if(n->type != T)
			a->etype = simtype[n->type->etype];
		a->offset = n->xoffset;
		a->sym = n->sym;
		if(a->sym == S)
			a->sym = lookup(".noname");
		if(n->method) {
			if(n->type != T)
			if(n->type->sym != S)
			if(n->type->sym->package != nil)
				a->sym = pkglookup(a->sym->name, n->type->sym->package);
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
		case PPARAMOUT:
			a->type = D_PARAM;
			break;
		case PFUNC:
			a->index = D_EXTERN;
			a->type = D_ADDR;
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
			a->sym = S;
			a->type = D_CONST;
			a->offset = mpgetfix(n->val.u.xval);
			break;
		case CTSTR:
			datagostring(n->val.u.sval, a);
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

void
sudoclean(void)
{
}

int
sudoaddable(int as, Node *n, Addr *a)
{
	return 0;
}
