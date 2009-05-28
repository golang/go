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

#define FCASE(a, b, c)  (((a)<<16)|((b)<<8)|(c))
int
foptoas(int op, Type *t, int flg)
{
	int et;

	et = t->etype;

	// clear Frev if unneeded
	switch(op) {
	case OADD:
	case OMUL:
		flg &= ~Frev;
		break;
	}

	switch(FCASE(op, et, flg)) {
	case FCASE(OADD, TFLOAT32, 0):
		return AFADDF;
	case FCASE(OADD, TFLOAT64, 0):
		return AFADDD;
	case FCASE(OADD, TFLOAT64, Fpop):
		return AFADDDP;

	case FCASE(OSUB, TFLOAT32, 0):
		return AFSUBF;
	case FCASE(OSUB, TFLOAT32, Frev):
		return AFSUBRF;

	case FCASE(OSUB, TFLOAT64, 0):
		return AFSUBD;
	case FCASE(OSUB, TFLOAT64, Frev):
		return AFSUBRD;
	case FCASE(OSUB, TFLOAT64, Fpop):
		return AFSUBDP;
	case FCASE(OSUB, TFLOAT64, Fpop|Frev):
		return AFSUBRDP;

	case FCASE(OMUL, TFLOAT32, 0):
		return AFMULF;
	case FCASE(OMUL, TFLOAT64, 0):
		return AFMULD;
	case FCASE(OMUL, TFLOAT64, Fpop):
		return AFMULDP;

	case FCASE(ODIV, TFLOAT32, 0):
		return AFDIVF;
	case FCASE(ODIV, TFLOAT32, Frev):
		return AFDIVRF;

	case FCASE(ODIV, TFLOAT64, 0):
		return AFDIVD;
	case FCASE(ODIV, TFLOAT64, Frev):
		return AFDIVRD;
	case FCASE(ODIV, TFLOAT64, Fpop):
		return AFDIVDP;
	case FCASE(ODIV, TFLOAT64, Fpop|Frev):
		return AFDIVRDP;

	case FCASE(OCMP, TFLOAT32, 0):
		return AFCOMF;
	case FCASE(OCMP, TFLOAT32, Fpop):
		return AFCOMFP;
	case FCASE(OCMP, TFLOAT64, 0):
		return AFCOMD;
	case FCASE(OCMP, TFLOAT64, Fpop):
		return AFCOMDP;
	case FCASE(OCMP, TFLOAT64, Fpop2):
		return AFCOMDPP;
	}

	fatal("foptoas %O %T %#x", op, t, flg);
	return 0;
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

ulong regpc[D_NONE];

void
gclean(void)
{
	int i;

	for(i=0; i<nelem(resvd); i++)
		reg[resvd[i]]--;

	for(i=D_AX; i<=D_DI; i++)
		if(reg[i])
			yyerror("reg %R left allocated at %lux\n", i, regpc[i]);
	for(i=D_F0; i<=D_F7; i++)
		if(reg[i])
			yyerror("reg %R left allocated\n", i);
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
		regpc[i] = (ulong)__builtin_return_address(0);
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
 * Is this node a memory operand?
 */
int
ismem(Node *n)
{
	switch(n->op) {
	case OINDREG:
	case ONAME:
	case OPARAM:
		return 1;
	}
	return 0;
}

Node sclean[10];
int nsclean;

/*
 * n is a 64-bit value.  fill in lo and hi to refer to its 32-bit halves.
 */
void
split64(Node *n, Node *lo, Node *hi)
{
	Node n1;
	int64 i;

	if(!is64(n->type))
		fatal("split64 %T", n->type);

	sclean[nsclean].op = OEMPTY;
	if(nsclean >= nelem(sclean))
		fatal("split64 clean");
	nsclean++;
	switch(n->op) {
	default:
		if(!dotaddable(n, &n1)) {
			igen(n, &n1, N);
			sclean[nsclean-1] = n1;
		}
		n = &n1;
		// fall through
	case ONAME:
	case OINDREG:
		*lo = *n;
		*hi = *n;
		lo->type = types[TUINT32];
		if(n->type->etype == TINT64)
			hi->type = types[TINT32];
		else
			hi->type = types[TUINT32];
		hi->xoffset += 4;
		break;

	case OLITERAL:
		convconst(&n1, n->type, &n->val);
		i = mpgetfix(n1.val.u.xval);
		nodconst(lo, types[TUINT32], (uint32)i);
		i >>= 32;
		if(n->type->etype == TINT64)
			nodconst(hi, types[TINT32], (int32)i);
		else
			nodconst(hi, types[TUINT32], (uint32)i);
		break;
	}
}

void
splitclean(void)
{
	if(nsclean <= 0)
		fatal("splitclean");
	nsclean--;
	if(sclean[nsclean].op != OEMPTY)
		regfree(&sclean[nsclean]);
}

void
gmove(Node *f, Node *t)
{
	int a, ft, tt;
	Type *cvt;
	Node r1, r2, flo, fhi, tlo, thi, con;

	if(debug['M'])
		print("gmove %N -> %N\n", f, t);

	ft = simsimtype(f->type);
	tt = simsimtype(t->type);
	cvt = t->type;

	// cannot have two memory operands;
	// except 64-bit, which always copies via registers anyway.
	if(ismem(f) && ismem(t) && !is64(f->type) && !is64(t->type))
		goto hard;

	// convert constant to desired type
	if(f->op == OLITERAL) {
		convconst(&con, t->type, &f->val);
		f = &con;
		ft = tt;	// so big switch will choose a simple mov

		// some constants can't move directly to memory.
		if(ismem(t)) {
			// float constants come from memory.
			if(isfloat[tt])
				goto hard;
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
		fatal("gmove %N -> %N", f, t);

	/*
	 * integer copy and truncate
	 */
	case CASE(TINT8, TINT8):	// same size
	case CASE(TINT8, TUINT8):
	case CASE(TUINT8, TINT8):
	case CASE(TUINT8, TUINT8):
	case CASE(TINT16, TINT8):	// truncate
	case CASE(TUINT16, TINT8):
	case CASE(TINT32, TINT8):
	case CASE(TUINT32, TINT8):
//	case CASE(TINT64, TINT8):
//	case CASE(TUINT64, TINT8):
	case CASE(TINT16, TUINT8):
	case CASE(TUINT16, TUINT8):
	case CASE(TINT32, TUINT8):
	case CASE(TUINT32, TUINT8):
//	case CASE(TINT64, TUINT8):
//	case CASE(TUINT64, TUINT8):
		a = AMOVB;
		break;

	case CASE(TINT16, TINT16):	// same size
	case CASE(TINT16, TUINT16):
	case CASE(TUINT16, TINT16):
	case CASE(TUINT16, TUINT16):
	case CASE(TINT32, TINT16):	// truncate
	case CASE(TUINT32, TINT16):
//	case CASE(TINT64, TINT16):
//	case CASE(TUINT64, TINT16):
	case CASE(TINT32, TUINT16):
	case CASE(TUINT32, TUINT16):
//	case CASE(TINT64, TUINT16):
//	case CASE(TUINT64, TUINT16):
		a = AMOVW;
		break;

	case CASE(TINT32, TINT32):	// same size
	case CASE(TINT32, TUINT32):
	case CASE(TUINT32, TINT32):
	case CASE(TUINT32, TUINT32):
//	case CASE(TINT64, TINT32):	// truncate
//	case CASE(TUINT64, TINT32):
//	case CASE(TINT64, TUINT32):
//	case CASE(TUINT64, TUINT32):
		a = AMOVL;
		break;

	case CASE(TINT64, TINT64):	// same size
	case CASE(TINT64, TUINT64):
	case CASE(TUINT64, TINT64):
	case CASE(TUINT64, TUINT64):
		split64(f, &flo, &fhi);
		split64(t, &tlo, &thi);
		if(f->op == OLITERAL) {
			gins(AMOVL, &flo, &tlo);
			gins(AMOVL, &fhi, &thi);
		} else {
			regalloc(&r1, types[TUINT32], N);
			regalloc(&r2, types[TUINT32], N);
			gins(AMOVL, &flo, &r1);
			gins(AMOVL, &fhi, &r2);
			gins(AMOVL, &r1, &tlo);
			gins(AMOVL, &r2, &thi);
			regfree(&r2);
			regfree(&r1);
		}
		splitclean();
		splitclean();
		return;

	/*
	 * integer up-conversions
	 */
	case CASE(TINT8, TINT16):	// sign extend int8
	case CASE(TINT8, TUINT16):
		a = AMOVBWSX;
		goto rdst;
	case CASE(TINT8, TINT32):
	case CASE(TINT8, TUINT32):
		a = AMOVBLSX;
		goto rdst;
//	case CASE(TINT8, TINT64):
//	case CASE(TINT8, TUINT64):
//		a = AMOVBQSX;
//		goto rdst;

	case CASE(TUINT8, TINT16):	// zero extend uint8
	case CASE(TUINT8, TUINT16):
		a = AMOVBWZX;
		goto rdst;
	case CASE(TUINT8, TINT32):
	case CASE(TUINT8, TUINT32):
		a = AMOVBLZX;
		goto rdst;
//	case CASE(TUINT8, TINT64):
//	case CASE(TUINT8, TUINT64):
//		a = AMOVBQZX;
//		goto rdst;

	case CASE(TINT16, TINT32):	// sign extend int16
	case CASE(TINT16, TUINT32):
		a = AMOVWLSX;
		goto rdst;
//	case CASE(TINT16, TINT64):
//	case CASE(TINT16, TUINT64):
//		a = AMOVWQSX;
//		goto rdst;

	case CASE(TUINT16, TINT32):	// zero extend uint16
	case CASE(TUINT16, TUINT32):
		a = AMOVWLZX;
		goto rdst;
//	case CASE(TUINT16, TINT64):
//	case CASE(TUINT16, TUINT64):
//		a = AMOVWQZX;
//		goto rdst;

	case CASE(TINT32, TINT64):	// sign extend int32
	case CASE(TINT32, TUINT64):
		split64(t, &tlo, &thi);
		nodreg(&flo, tlo.type, D_AX);
		nodreg(&fhi, thi.type, D_DX);
		gmove(f, &flo);
		gins(ACDQ, N, N);
		gins(AMOVL, &flo, &tlo);
		gins(AMOVL, &fhi, &thi);
		return;

//	case CASE(TUINT32, TINT64):	// zero extend uint32
//	case CASE(TUINT32, TUINT64):
//		// AMOVL into a register zeros the top of the register,
//		// so this is not always necessary, but if we rely on AMOVL
//		// the optimizer is almost certain to screw with us.
//		a = AMOVLQZX;
//		goto rdst;

	/*
	* float to integer
	*
	case CASE(TFLOAT32, TINT16):
	case CASE(TFLOAT32, TINT32):
	case CASE(TFLOAT32, TINT64):
	case CASE(TFLOAT64, TINT16):
	case CASE(TFLOAT64, TINT32):
	case CASE(TFLOAT64, TINT64):
		if(ft == TFLOAT32)
			gins(AFMOVF, f, &f0);
		else
			gins(AFMOVD, f, &f0);
		if(tt == TINT16)
			gins(AFMOVWP, &f0, t);
		else if(tt == TINT32)
			gins(AFMOVLP, &f0, t);
		else
			gins(AFMOVVP, &f0, t);
		return;

	case CASE(TFLOAT32, TINT8):
	case CASE(TFLOAT32, TUINT16):
	case CASE(TFLOAT32, TUINT8):
	case CASE(TFLOAT64, TINT8):
	case CASE(TFLOAT64, TUINT16):
	case CASE(TFLOAT64, TUINT8):
		// convert via int32.
		cvt = types[TINT32];
		goto hard;

	case CASE(TFLOAT32, TUINT32):
	case CASE(TFLOAT64, TUINT32):
		// could potentially convert via int64.
		cvt = types[TINT64];
		goto hard;

	case CASE(TFLOAT32, TUINT64):
	case CASE(TFLOAT64, TUINT64):
		if(ft == TFLOAT32)
			gins(AFMOVF, f, &f0);
		else
			gins(AFMOVD, f, &f0);
		// algorithm is:
		//	if small enough, use native float64 -> int64 conversion.
		//	otherwise, subtract 2^63, convert, and add it back.
		bignodes();
		regalloc(&r1, types[ft], N);
		regalloc(&r2, types[ft], N);
		gins(optoas(OCMP, f->type), &bigf, &r1);
		p1 = gbranch(optoas(OLE, f->type), T);
		gins(a, &r1, &r2);
		p2 = gbranch(AJMP, T);
		patch(p1, pc);
		gins(optoas(OAS, f->type), &bigf, &r3);
		gins(optoas(OSUB, f->type), &r3, &r1);
		gins(a, &r1, &r2);
		gins(AMOVQ, &bigi, &r4);
		gins(AXORQ, &r4, &r2);
		patch(p2, pc);
		gmove(&r2, t);
		regfree(&r4);
		regfree(&r3);
		regfree(&r2);
		regfree(&r1);
		fatal("lazy");
		return;
	*/
	/*
	 * integer to float
	 *
	case CASE(TINT32, TFLOAT32):
		a = ACVTSL2SS;
		goto rdst;


	case CASE(TINT32, TFLOAT64):
		a = ACVTSL2SD;
		goto rdst;

	case CASE(TINT64, TFLOAT32):
		a = ACVTSQ2SS;
		goto rdst;

	case CASE(TINT64, TFLOAT64):
		a = ACVTSQ2SD;
		goto rdst;

	case CASE(TINT16, TFLOAT32):
	case CASE(TINT16, TFLOAT64):
	case CASE(TINT8, TFLOAT32):
	case CASE(TINT8, TFLOAT64):
	case CASE(TUINT16, TFLOAT32):
	case CASE(TUINT16, TFLOAT64):
	case CASE(TUINT8, TFLOAT32):
	case CASE(TUINT8, TFLOAT64):
		// convert via int32
		cvt = types[TINT32];
		goto hard;

	case CASE(TUINT32, TFLOAT32):
	case CASE(TUINT32, TFLOAT64):
		// convert via int64.
		cvt = types[TINT64];
		goto hard;

	case CASE(TUINT64, TFLOAT32):
	case CASE(TUINT64, TFLOAT64):
		// algorithm is:
		//	if small enough, use native int64 -> uint64 conversion.
		//	otherwise, halve (rounding to odd?), convert, and double.
		a = ACVTSQ2SS;
		if(tt == TFLOAT64)
			a = ACVTSQ2SD;
		nodconst(&zero, types[TUINT64], 0);
		nodconst(&one, types[TUINT64], 1);
		regalloc(&r1, f->type, f);
		regalloc(&r2, t->type, t);
		regalloc(&r3, f->type, N);
		regalloc(&r4, f->type, N);
		gmove(f, &r1);
		gins(ACMPQ, &r1, &zero);
		p1 = gbranch(AJLT, T);
		gins(a, &r1, &r2);
		p2 = gbranch(AJMP, T);
		patch(p1, pc);
		gmove(&r1, &r3);
		gins(ASHRQ, &one, &r3);
		gmove(&r1, &r4);
		gins(AANDL, &one, &r4);
		gins(AORQ, &r4, &r3);
		gins(a, &r3, &r2);
		gins(optoas(OADD, t->type), &r2, &r2);
		patch(p2, pc);
		gmove(&r2, t);
		regfree(&r4);
		regfree(&r3);
		regfree(&r2);
		regfree(&r1);
		return;
	*/
	/*
	 * float to float
	 *
	case CASE(TFLOAT32, TFLOAT32):
		a = AMOVSS;
		break;

	case CASE(TFLOAT64, TFLOAT64):
		a = AMOVSD;
		break;

	case CASE(TFLOAT32, TFLOAT64):
		a = ACVTSS2SD;
		goto rdst;

	case CASE(TFLOAT64, TFLOAT32):
		a = ACVTSD2SS;
		goto rdst;
	*/
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

	switch(as) {
	case AMOVB:
	case AMOVW:
	case AMOVL:
		if(f != N && t != N && samaddr(f, t))
			return nil;
	}

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

int
dotaddable(Node *n, Node *n1)
{
	int o, oary[10];
	Node *nn;

	if(n->op != ODOT)
		return 0;

	o = dotoffset(n, oary, &nn);
	if(nn != N && nn->addable && o == 1 && oary[0] >= 0) {
		*n1 = *nn;
		n1->type = n->type;
		n1->xoffset += oary[0];
		return 1;
	}
	return 0;
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
