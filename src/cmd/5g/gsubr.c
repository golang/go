// Derived from Inferno utils/5c/txt.c
// http://code.google.com/p/inferno-os/source/browse/utils/5c/txt.c
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

	p = gbranch(AB, T);
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
			if(i >= REGALLOC_R0 && i <= REGALLOC_RMAX)
				goto out;
		}
		for(i=REGALLOC_R0; i<=REGALLOC_RMAX; i++)
			if(reg[i] == 0)
				goto out;

		yyerror("out of fixed registers");
		goto err;

	case TFLOAT32:
	case TFLOAT64:
	case TFLOAT80:
		if(o != N && o->op == OREGISTER) {
			i = o->val.u.reg;
			if(i >= REGALLOC_F0 && i <= REGALLOC_FMAX)
				goto out;
		}
		for(i=REGALLOC_F0; i<=REGALLOC_FMAX; i++)
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
//print("tempalloc %d -> %d from %p\n", n->ostk, n->xoffset, __builtin_return_address(0));
	if(stksize > maxstksize)
		maxstksize = stksize;
}

void
tempfree(Node *n)
{
//print("tempfree %d\n", n->xoffset);
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
	if(t->etype == TSTRUCT && t->funarg) {
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
		goto fp;
	}

	if(t->etype != TFIELD)
		fatal("nodarg: not field %T", t);

	n = nod(ONAME, N, N);
	n->type = t->type;
	n->sym = t->sym;
	if(t->width == BADWIDTH)
		fatal("nodarg: offset not computed for %T", t);
	n->xoffset = t->width;
	n->addable = 1;

fp:
	switch(fp) {
	case 0:		// output arg
		n->op = OINDREG;
		n->val.u.reg = REGRET;
		break;

	case 1:		// input arg
		n->class = PPARAM;
		break;

	case 2:		// offset output arg
fatal("shouldnt be used");
		n->op = OINDREG;
		n->val.u.reg = REGSP;
		n->xoffset += types[tptr]->width;
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

#define	CASE(a,b)	(((a)<<16)|((b)<<0))

void
gmove(Node *f, Node *t)
{
	int a, ft, tt;
	Type *cvt;
	Node r1, r2, t1, t2, flo, fhi, tlo, thi, con, f0, f1, ax, dx, cx;
	Prog *p1, *p2, *p3;

	if(debug['M'])
		print("gmove %N -> %N\n", f, t);

	ft = simsimtype(f->type);
	tt = simsimtype(t->type);
	cvt = t->type;

	// cannot have two integer memory operands;
	// except 64-bit, which always copies via registers anyway.
	// TODO(kaib): re-enable check
// 	if(isint[ft] && isint[tt] && !is64(f->type) && !is64(t->type) && ismem(f) && ismem(t))
// 		goto hard;

	// convert constant to desired type
	if(f->op == OLITERAL) {
		if(tt == TFLOAT32)
			convconst(&con, types[TFLOAT64], &f->val);
		else
			convconst(&con, t->type, &f->val);
		f = &con;
		ft = simsimtype(con.type);

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
		goto fatal;

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
	case CASE(TINT16, TUINT8):
	case CASE(TUINT16, TUINT8):
	case CASE(TINT32, TUINT8):
	case CASE(TUINT32, TUINT8):
		a = AMOVB;
		break;

	case CASE(TINT64, TINT8):	// truncate low word
	case CASE(TUINT64, TINT8):
	case CASE(TINT64, TUINT8):
	case CASE(TUINT64, TUINT8):
		fatal("gmove INT64,INT8 not implemented");
// 		split64(f, &flo, &fhi);
// 		nodreg(&r1, t->type, D_AX);
// 		gins(AMOVB, &flo, &r1);
// 		gins(AMOVB, &r1, t);
// 		splitclean();
		return;

	case CASE(TINT16, TINT16):	// same size
	case CASE(TINT16, TUINT16):
	case CASE(TUINT16, TINT16):
	case CASE(TUINT16, TUINT16):
	case CASE(TINT32, TINT16):	// truncate
	case CASE(TUINT32, TINT16):
	case CASE(TINT32, TUINT16):
	case CASE(TUINT32, TUINT16):
		a = AMOVH;
		break;

	case CASE(TINT64, TINT16):	// truncate low word
	case CASE(TUINT64, TINT16):
	case CASE(TINT64, TUINT16):
	case CASE(TUINT64, TUINT16):
		fatal("gmove INT64,INT16 not implemented");
// 		split64(f, &flo, &fhi);
// 		nodreg(&r1, t->type, D_AX);
// 		gins(AMOVW, &flo, &r1);
// 		gins(AMOVW, &r1, t);
// 		splitclean();
		return;

	case CASE(TINT32, TINT32):	// same size
	case CASE(TINT32, TUINT32):
	case CASE(TUINT32, TINT32):
	case CASE(TUINT32, TUINT32):
		a = AMOVW;
		break;

	case CASE(TINT64, TINT32):	// truncate
	case CASE(TUINT64, TINT32):
	case CASE(TINT64, TUINT32):
	case CASE(TUINT64, TUINT32):
		fatal("gmove INT64,INT32 not implemented");
// 		split64(f, &flo, &fhi);
// 		nodreg(&r1, t->type, D_AX);
// 		gins(AMOVL, &flo, &r1);
// 		gins(AMOVL, &r1, t);
// 		splitclean();
		return;

	case CASE(TINT64, TINT64):	// same size
	case CASE(TINT64, TUINT64):
	case CASE(TUINT64, TINT64):
	case CASE(TUINT64, TUINT64):
		fatal("gmove INT64,INT64 not implemented");
// 		split64(f, &flo, &fhi);
// 		split64(t, &tlo, &thi);
// 		if(f->op == OLITERAL) {
// 			gins(AMOVL, &flo, &tlo);
// 			gins(AMOVL, &fhi, &thi);
// 		} else {
// 			nodreg(&r1, t->type, D_AX);
// 			nodreg(&r2, t->type, D_DX);
// 			gins(AMOVL, &flo, &r1);
// 			gins(AMOVL, &fhi, &r2);
// 			gins(AMOVL, &r1, &tlo);
// 			gins(AMOVL, &r2, &thi);
// 		}
// 		splitclean();
// 		splitclean();
		return;

	/*
	 * integer up-conversions
	 */
// 	case CASE(TINT8, TINT16):	// sign extend int8
// 	case CASE(TINT8, TUINT16):
// 		a = AMOVBWSX;
// 		goto rdst;
// 	case CASE(TINT8, TINT32):
// 	case CASE(TINT8, TUINT32):
// 		a = AMOVBLSX;
// 		goto rdst;
// 	case CASE(TINT8, TINT64):	// convert via int32
// 	case CASE(TINT8, TUINT64):
// 		cvt = types[TINT32];
// 		goto hard;

// 	case CASE(TUINT8, TINT16):	// zero extend uint8
// 	case CASE(TUINT8, TUINT16):
// 		a = AMOVBWZX;
// 		goto rdst;
// 	case CASE(TUINT8, TINT32):
// 	case CASE(TUINT8, TUINT32):
// 		a = AMOVBLZX;
// 		goto rdst;
// 	case CASE(TUINT8, TINT64):	// convert via uint32
// 	case CASE(TUINT8, TUINT64):
// 		cvt = types[TUINT32];
// 		goto hard;

// 	case CASE(TINT16, TINT32):	// sign extend int16
// 	case CASE(TINT16, TUINT32):
// 		a = AMOVWLSX;
// 		goto rdst;
// 	case CASE(TINT16, TINT64):	// convert via int32
// 	case CASE(TINT16, TUINT64):
// 		cvt = types[TINT32];
// 		goto hard;

// 	case CASE(TUINT16, TINT32):	// zero extend uint16
// 	case CASE(TUINT16, TUINT32):
// 		a = AMOVWLZX;
// 		goto rdst;
// 	case CASE(TUINT16, TINT64):	// convert via uint32
// 	case CASE(TUINT16, TUINT64):
// 		cvt = types[TUINT32];
// 		goto hard;

// 	case CASE(TINT32, TINT64):	// sign extend int32
// 	case CASE(TINT32, TUINT64):
// 		fatal("gmove TINT32,INT64 not implemented");
// // 		split64(t, &tlo, &thi);
// // 		nodreg(&flo, tlo.type, D_AX);
// // 		nodreg(&fhi, thi.type, D_DX);
// // 		gmove(f, &flo);
// // 		gins(ACDQ, N, N);
// // 		gins(AMOVL, &flo, &tlo);
// // 		gins(AMOVL, &fhi, &thi);
// // 		splitclean();
// 		return;

// 	case CASE(TUINT32, TINT64):	// zero extend uint32
// 	case CASE(TUINT32, TUINT64):
// 		fatal("gmove TUINT32,INT64 not implemented");
// // 		split64(t, &tlo, &thi);
// // 		gmove(f, &tlo);
// // 		gins(AMOVL, ncon(0), &thi);
// // 		splitclean();
// 		return;

// 	/*
// 	* float to integer
// 	*/
// 	case CASE(TFLOAT32, TINT16):
// 	case CASE(TFLOAT32, TINT32):
// 	case CASE(TFLOAT32, TINT64):
// 	case CASE(TFLOAT64, TINT16):
// 	case CASE(TFLOAT64, TINT32):
// 	case CASE(TFLOAT64, TINT64):
// 		if(t->op == OREGISTER)
// 			goto hardmem;
// 		nodreg(&r1, types[ft], D_F0);
// 		if(ft == TFLOAT32)
// 			gins(AFMOVF, f, &r1);
// 		else
// 			gins(AFMOVD, f, &r1);

// 		// set round to zero mode during conversion
// 		tempalloc(&t1, types[TUINT16]);
// 		tempalloc(&t2, types[TUINT16]);
// 		gins(AFSTCW, N, &t1);
// 		gins(AMOVW, ncon(0xf7f), &t2);
// 		gins(AFLDCW, &t2, N);
// 		if(tt == TINT16)
// 			gins(AFMOVWP, &r1, t);
// 		else if(tt == TINT32)
// 			gins(AFMOVLP, &r1, t);
// 		else
// 			gins(AFMOVVP, &r1, t);
// 		gins(AFLDCW, &t1, N);
// 		tempfree(&t2);
// 		tempfree(&t1);
// 		return;

// 	case CASE(TFLOAT32, TINT8):
// 	case CASE(TFLOAT32, TUINT16):
// 	case CASE(TFLOAT32, TUINT8):
// 	case CASE(TFLOAT64, TINT8):
// 	case CASE(TFLOAT64, TUINT16):
// 	case CASE(TFLOAT64, TUINT8):
// 		// convert via int32.
// 		tempalloc(&t1, types[TINT32]);
// 		gmove(f, &t1);
// 		switch(tt) {
// 		default:
// 			fatal("gmove %T", t);
// 		case TINT8:
// 			gins(ACMPL, &t1, ncon(-0x80));
// 			p1 = gbranch(optoas(OLT, types[TINT32]), T);
// 			gins(ACMPL, &t1, ncon(0x7f));
// 			p2 = gbranch(optoas(OGT, types[TINT32]), T);
// 			p3 = gbranch(AJMP, T);
// 			patch(p1, pc);
// 			patch(p2, pc);
// 			gmove(ncon(-0x80), &t1);
// 			patch(p3, pc);
// 			gmove(&t1, t);
// 			break;
// 		case TUINT8:
// 			gins(ATESTL, ncon(0xffffff00), &t1);
// 			p1 = gbranch(AJEQ, T);
// 			gins(AMOVB, ncon(0), &t1);
// 			patch(p1, pc);
// 			gmove(&t1, t);
// 			break;
// 		case TUINT16:
// 			gins(ATESTL, ncon(0xffff0000), &t1);
// 			p1 = gbranch(AJEQ, T);
// 			gins(AMOVW, ncon(0), &t1);
// 			patch(p1, pc);
// 			gmove(&t1, t);
// 			break;
// 		}
// 		tempfree(&t1);
// 		return;

// 	case CASE(TFLOAT32, TUINT32):
// 	case CASE(TFLOAT64, TUINT32):
// 		// convert via int64.
// 		tempalloc(&t1, types[TINT64]);
// 		gmove(f, &t1);
// 		split64(&t1, &tlo, &thi);
// 		gins(ACMPL, &thi, ncon(0));
// 		p1 = gbranch(AJEQ, T);
// 		gins(AMOVL, ncon(0), &tlo);
// 		patch(p1, pc);
// 		gmove(&tlo, t);
// 		splitclean();
// 		tempfree(&t1);
// 		return;

// 	case CASE(TFLOAT32, TUINT64):
// 	case CASE(TFLOAT64, TUINT64):
// 		bignodes();
// 		nodreg(&f0, types[ft], D_F0);
// 		nodreg(&f1, types[ft], D_F0 + 1);
// 		nodreg(&ax, types[TUINT16], D_AX);

// 		gmove(f, &f0);

// 		// if 0 > v { answer = 0 }
// 		gmove(&zerof, &f0);
// 		gins(AFUCOMP, &f0, &f1);
// 		gins(AFSTSW, N, &ax);
// 		gins(ASAHF, N, N);
// 		p1 = gbranch(optoas(OGT, types[tt]), T);
// 		// if 1<<64 <= v { answer = 0 too }
// 		gmove(&two64f, &f0);
// 		gins(AFUCOMP, &f0, &f1);
// 		gins(AFSTSW, N, &ax);
// 		gins(ASAHF, N, N);
// 		p2 = gbranch(optoas(OGT, types[tt]), T);
// 		patch(p1, pc);
// 		gins(AFMOVVP, &f0, t);	// don't care about t, but will pop the stack
// 		split64(t, &tlo, &thi);
// 		gins(AMOVL, ncon(0), &tlo);
// 		gins(AMOVL, ncon(0), &thi);
// 		splitclean();
// 		p1 = gbranch(AJMP, T);
// 		patch(p2, pc);

// 		// in range; algorithm is:
// 		//	if small enough, use native float64 -> int64 conversion.
// 		//	otherwise, subtract 2^63, convert, and add it back.

// 		// set round to zero mode during conversion
// 		tempalloc(&t1, types[TUINT16]);
// 		tempalloc(&t2, types[TUINT16]);
// 		gins(AFSTCW, N, &t1);
// 		gins(AMOVW, ncon(0xf7f), &t2);
// 		gins(AFLDCW, &t2, N);
// 		tempfree(&t2);

// 		// actual work
// 		gmove(&two63f, &f0);
// 		gins(AFUCOMP, &f0, &f1);
// 		gins(AFSTSW, N, &ax);
// 		gins(ASAHF, N, N);
// 		p2 = gbranch(optoas(OLE, types[tt]), T);
// 		gins(AFMOVVP, &f0, t);
// 		p3 = gbranch(AJMP, T);
// 		patch(p2, pc);
// 		gmove(&two63f, &f0);
// 		gins(AFSUBDP, &f0, &f1);
// 		gins(AFMOVVP, &f0, t);
// 		split64(t, &tlo, &thi);
// 		gins(AXORL, ncon(0x80000000), &thi);	// + 2^63
// 		patch(p3, pc);
// 		patch(p1, pc);
// 		splitclean();

// 		// restore rounding mode
// 		gins(AFLDCW, &t1, N);
// 		tempfree(&t1);
// 		return;

// 	/*
// 	 * integer to float
// 	 */
// 	case CASE(TINT16, TFLOAT32):
// 	case CASE(TINT16, TFLOAT64):
// 	case CASE(TINT32, TFLOAT32):
// 	case CASE(TINT32, TFLOAT64):
// 	case CASE(TINT64, TFLOAT32):
// 	case CASE(TINT64, TFLOAT64):
// 		fatal("gmove TINT,TFLOAT not implemented");
// // 		if(t->op != OREGISTER)
// // 			goto hard;
// // 		if(f->op == OREGISTER) {
// // 			cvt = f->type;
// // 			goto hardmem;
// // 		}
// // 		switch(ft) {
// // 		case TINT16:
// // 			a = AFMOVW;
// // 			break;
// // 		case TINT32:
// // 			a = AFMOVL;
// // 			break;
// // 		default:
// // 			a = AFMOVV;
// // 			break;
// // 		}
// 		break;

// 	case CASE(TINT8, TFLOAT32):
// 	case CASE(TINT8, TFLOAT64):
// 	case CASE(TUINT16, TFLOAT32):
// 	case CASE(TUINT16, TFLOAT64):
// 	case CASE(TUINT8, TFLOAT32):
// 	case CASE(TUINT8, TFLOAT64):
// 		// convert via int32 memory
// 		cvt = types[TINT32];
// 		goto hardmem;

// 	case CASE(TUINT32, TFLOAT32):
// 	case CASE(TUINT32, TFLOAT64):
// 		// convert via int64 memory
// 		cvt = types[TINT64];
// 		goto hardmem;

// 	case CASE(TUINT64, TFLOAT32):
// 	case CASE(TUINT64, TFLOAT64):
// 		// algorithm is:
// 		//	if small enough, use native int64 -> uint64 conversion.
// 		//	otherwise, halve (rounding to odd?), convert, and double.
// 		nodreg(&ax, types[TUINT32], D_AX);
// 		nodreg(&dx, types[TUINT32], D_DX);
// 		nodreg(&cx, types[TUINT32], D_CX);
// 		tempalloc(&t1, f->type);
// 		split64(&t1, &tlo, &thi);
// 		gmove(f, &t1);
// 		gins(ACMPL, &thi, ncon(0));
// 		p1 = gbranch(AJLT, T);
// 		// native
// 		t1.type = types[TINT64];
// 		gmove(&t1, t);
// 		p2 = gbranch(AJMP, T);
// 		// simulated
// 		patch(p1, pc);
// 		gmove(&tlo, &ax);
// 		gmove(&thi, &dx);
// 		p1 = gins(ASHRL, ncon(1), &ax);
// 		p1->from.index = D_DX;	// double-width shift DX -> AX
// 		p1->from.scale = 0;
// 		gins(ASETCC, N, &cx);
// 		gins(AORB, &cx, &ax);
// 		gins(ASHRL, ncon(1), &dx);
// 		gmove(&dx, &thi);
// 		gmove(&ax, &tlo);
// 		nodreg(&r1, types[tt], D_F0);
// 		nodreg(&r2, types[tt], D_F0 + 1);
// 		gmove(&t1, &r1);	// t1.type is TINT64 now, set above
// 		gins(AFMOVD, &r1, &r1);
// 		gins(AFADDDP, &r1, &r2);
// 		gmove(&r1, t);
// 		patch(p2, pc);
// 		splitclean();
// 		tempfree(&t1);
// 		return;

// 	/*
// 	 * float to float
// 	 */
// 	case CASE(TFLOAT32, TFLOAT32):
// 	case CASE(TFLOAT64, TFLOAT64):
// 		// The way the code generator uses floating-point
// 		// registers, a move from F0 to F0 is intended as a no-op.
// 		// On the x86, it's not: it pushes a second copy of F0
// 		// on the floating point stack.  So toss it away here.
// 		// Also, F0 is the *only* register we ever evaluate
// 		// into, so we should only see register/register as F0/F0.
// 		if(f->op == OREGISTER && t->op == OREGISTER) {
// 			if(f->val.u.reg != D_F0 || t->val.u.reg != D_F0)
// 				goto fatal;
// 			return;
// 		}
// 		if(ismem(f) && ismem(t))
// 			goto hard;
// 		a = AFMOVF;
// 		if(ft == TFLOAT64)
// 			a = AFMOVD;
// 		if(ismem(t)) {
// 			if(f->op != OREGISTER || f->val.u.reg != D_F0)
// 				fatal("gmove %N", f);
// 			a = AFMOVFP;
// 			if(ft == TFLOAT64)
// 				a = AFMOVDP;
// 		}
// 		break;

// 	case CASE(TFLOAT32, TFLOAT64):
// 		if(f->op == OREGISTER && t->op == OREGISTER) {
// 			if(f->val.u.reg != D_F0 || t->val.u.reg != D_F0)
// 				goto fatal;
// 			return;
// 		}
// 		if(f->op == OREGISTER)
// 			gins(AFMOVDP, f, t);
// 		else
// 			gins(AFMOVF, f, t);
// 		return;

// 	case CASE(TFLOAT64, TFLOAT32):
// 		if(f->op == OREGISTER && t->op == OREGISTER) {
// 			tempalloc(&r1, types[TFLOAT32]);
// 			gins(AFMOVFP, f, &r1);
// 			gins(AFMOVF, &r1, t);
// 			tempfree(&r1);
// 			return;
// 		}
// 		if(f->op == OREGISTER)
// 			gins(AFMOVFP, f, t);
// 		else
// 			gins(AFMOVD, f, t);
// 		return;
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

hardmem:
	// requires memory intermediate
	tempalloc(&r1, cvt);
	gmove(f, &r1);
	gmove(&r1, t);
	tempfree(&r1);
	return;

fatal:
	// should not happen
	fatal("gmove %N -> %N", f, t);
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
		a->type = D_OREG;
		if (n->val.u.reg <= REGALLOC_RMAX)
			a->reg = n->val.u.reg;
		else
			a->reg = n->val.u.reg - REGALLOC_F0;
		a->sym = S;
		break;

	case OINDEX:
	case OIND:
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

//	case OINDREG:
//		a->type = n->val.u.reg+D_INDIR;
//		a->sym = n->sym;
//		a->offset = n->xoffset;
//		break;

//	case OPARAM:
//		// n->left is PHEAP ONAME for stack parameter.
//		// compute address of actual parameter on stack.
//		a->etype = simtype[n->left->type->etype];
//		a->width = n->left->type->width;
//		a->offset = n->xoffset;
//		a->sym = n->left->sym;
//		a->type = D_PARAM;
//		break;

	case ONAME:
		a->etype = 0;
		a->width = 0;
		if(n->type != T) {
			a->etype = simtype[n->type->etype];
			a->width = n->type->width;
		}
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

//	case OADDR:
//		naddr(n->left, a);
//		if(a->type >= D_INDIR) {
//			a->type -= D_INDIR;
//			break;
//		}
//		if(a->type == D_EXTERN || a->type == D_STATIC ||
//		   a->type == D_AUTO || a->type == D_PARAM)
//			if(a->index == D_NONE) {
//				a->index = a->type;
//				a->type = D_ADDR;
//				break;
//			}
//		fatal("naddr: OADDR\n");

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

/*	case CASE(OADDR, TPTR32):
		a = ALEAL;
		break;

	case CASE(OADDR, TPTR64):
		a = ALEAQ;
		break;
*/
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

	case CASE(OLT, TINT8):
	case CASE(OLT, TINT16):
	case CASE(OLT, TINT32):
	case CASE(OLT, TINT64):
		a = ABLT;
		break;

	case CASE(OLT, TUINT8):
	case CASE(OLT, TUINT16):
	case CASE(OLT, TUINT32):
	case CASE(OLT, TUINT64):
	case CASE(OGT, TFLOAT32):
	case CASE(OGT, TFLOAT64):
		a = ABCS;
		break;

	case CASE(OLE, TINT8):
	case CASE(OLE, TINT16):
	case CASE(OLE, TINT32):
	case CASE(OLE, TINT64):
		a = ABLE;
		break;

	case CASE(OLE, TUINT8):
	case CASE(OLE, TUINT16):
	case CASE(OLE, TUINT32):
	case CASE(OLE, TUINT64):
	case CASE(OGE, TFLOAT32):
	case CASE(OGE, TFLOAT64):
		a = ABLS;
		break;

	case CASE(OGT, TINT8):
	case CASE(OGT, TINT16):
	case CASE(OGT, TINT32):
	case CASE(OGT, TINT64):
		a = ABGT;
		break;

	case CASE(OGT, TUINT8):
	case CASE(OGT, TUINT16):
	case CASE(OGT, TUINT32):
	case CASE(OGT, TUINT64):
	case CASE(OLT, TFLOAT32):
	case CASE(OLT, TFLOAT64):
		a = ABHI;
		break;

	case CASE(OGE, TINT8):
	case CASE(OGE, TINT16):
	case CASE(OGE, TINT32):
	case CASE(OGE, TINT64):
		a = ABGE;
		break;

	case CASE(OGE, TUINT8):
	case CASE(OGE, TUINT16):
	case CASE(OGE, TUINT32):
	case CASE(OGE, TUINT64):
	case CASE(OLE, TFLOAT32):
	case CASE(OLE, TFLOAT64):
		a = ABCC;
		break;

	case CASE(OCMP, TBOOL):
	case CASE(OCMP, TINT8):
	case CASE(OCMP, TUINT8):
		a = ACMP;
		break;

//	case CASE(OCMP, TINT16):
//	case CASE(OCMP, TUINT16):
//		a = ACMPW;
//		break;

//	case CASE(OCMP, TINT32):
//	case CASE(OCMP, TUINT32):
//	case CASE(OCMP, TPTR32):
//		a = ACMPL;
//		break;

//	case CASE(OCMP, TINT64):
//	case CASE(OCMP, TUINT64):
//	case CASE(OCMP, TPTR64):
//		a = ACMPQ;
//		break;

//	case CASE(OCMP, TFLOAT32):
//		a = AUCOMISS;
//		break;

//	case CASE(OCMP, TFLOAT64):
//		a = AUCOMISD;
//		break;

//	case CASE(OAS, TBOOL):
//	case CASE(OAS, TINT8):
//	case CASE(OAS, TUINT8):
//		a = AMOVB;
//		break;

//	case CASE(OAS, TINT16):
//	case CASE(OAS, TUINT16):
//		a = AMOVW;
//		break;

//	case CASE(OAS, TINT32):
//	case CASE(OAS, TUINT32):
//	case CASE(OAS, TPTR32):
//		a = AMOVL;
//		break;

//	case CASE(OAS, TINT64):
//	case CASE(OAS, TUINT64):
//	case CASE(OAS, TPTR64):
//		a = AMOVQ;
//		break;

//	case CASE(OAS, TFLOAT32):
//		a = AMOVSS;
//		break;

//	case CASE(OAS, TFLOAT64):
//		a = AMOVSD;
//		break;

	case CASE(OADD, TINT8):
	case CASE(OADD, TUINT8):
	case CASE(OADD, TINT16):
	case CASE(OADD, TUINT16):
	case CASE(OADD, TINT32):
	case CASE(OADD, TUINT32):
	case CASE(OADD, TPTR32):
		a = AADD;
		break;

//	case CASE(OADD, TINT64):
//	case CASE(OADD, TUINT64):
//	case CASE(OADD, TPTR64):
//		a = AADDQ;
//		break;

//	case CASE(OADD, TFLOAT32):
//		a = AADDSS;
//		break;

//	case CASE(OADD, TFLOAT64):
//		a = AADDSD;
//		break;

//	case CASE(OSUB, TINT8):
//	case CASE(OSUB, TUINT8):
//		a = ASUBB;
//		break;

//	case CASE(OSUB, TINT16):
//	case CASE(OSUB, TUINT16):
//		a = ASUBW;
//		break;

//	case CASE(OSUB, TINT32):
//	case CASE(OSUB, TUINT32):
//	case CASE(OSUB, TPTR32):
//		a = ASUBL;
//		break;

//	case CASE(OSUB, TINT64):
//	case CASE(OSUB, TUINT64):
//	case CASE(OSUB, TPTR64):
//		a = ASUBQ;
//		break;

//	case CASE(OSUB, TFLOAT32):
//		a = ASUBSS;
//		break;

//	case CASE(OSUB, TFLOAT64):
//		a = ASUBSD;
//		break;

//	case CASE(OMINUS, TINT8):
//	case CASE(OMINUS, TUINT8):
//		a = ANEGB;
//		break;

//	case CASE(OMINUS, TINT16):
//	case CASE(OMINUS, TUINT16):
//		a = ANEGW;
//		break;

//	case CASE(OMINUS, TINT32):
//	case CASE(OMINUS, TUINT32):
//	case CASE(OMINUS, TPTR32):
//		a = ANEGL;
//		break;

//	case CASE(OMINUS, TINT64):
//	case CASE(OMINUS, TUINT64):
//	case CASE(OMINUS, TPTR64):
//		a = ANEGQ;
//		break;

//	case CASE(OAND, TINT8):
//	case CASE(OAND, TUINT8):
//		a = AANDB;
//		break;

//	case CASE(OAND, TINT16):
//	case CASE(OAND, TUINT16):
//		a = AANDW;
//		break;

//	case CASE(OAND, TINT32):
//	case CASE(OAND, TUINT32):
//	case CASE(OAND, TPTR32):
//		a = AANDL;
//		break;

//	case CASE(OAND, TINT64):
//	case CASE(OAND, TUINT64):
//	case CASE(OAND, TPTR64):
//		a = AANDQ;
//		break;

//	case CASE(OOR, TINT8):
//	case CASE(OOR, TUINT8):
//		a = AORB;
//		break;

//	case CASE(OOR, TINT16):
//	case CASE(OOR, TUINT16):
//		a = AORW;
//		break;

//	case CASE(OOR, TINT32):
//	case CASE(OOR, TUINT32):
//	case CASE(OOR, TPTR32):
//		a = AORL;
//		break;

//	case CASE(OOR, TINT64):
//	case CASE(OOR, TUINT64):
//	case CASE(OOR, TPTR64):
//		a = AORQ;
//		break;

//	case CASE(OXOR, TINT8):
//	case CASE(OXOR, TUINT8):
//		a = AXORB;
//		break;

//	case CASE(OXOR, TINT16):
//	case CASE(OXOR, TUINT16):
//		a = AXORW;
//		break;

//	case CASE(OXOR, TINT32):
//	case CASE(OXOR, TUINT32):
//	case CASE(OXOR, TPTR32):
//		a = AXORL;
//		break;

//	case CASE(OXOR, TINT64):
//	case CASE(OXOR, TUINT64):
//	case CASE(OXOR, TPTR64):
//		a = AXORQ;
//		break;

//	case CASE(OLSH, TINT8):
//	case CASE(OLSH, TUINT8):
//		a = ASHLB;
//		break;

//	case CASE(OLSH, TINT16):
//	case CASE(OLSH, TUINT16):
//		a = ASHLW;
//		break;

//	case CASE(OLSH, TINT32):
//	case CASE(OLSH, TUINT32):
//	case CASE(OLSH, TPTR32):
//		a = ASHLL;
//		break;

//	case CASE(OLSH, TINT64):
//	case CASE(OLSH, TUINT64):
//	case CASE(OLSH, TPTR64):
//		a = ASHLQ;
//		break;

//	case CASE(ORSH, TUINT8):
//		a = ASHRB;
//		break;

//	case CASE(ORSH, TUINT16):
//		a = ASHRW;
//		break;

//	case CASE(ORSH, TUINT32):
//	case CASE(ORSH, TPTR32):
//		a = ASHRL;
//		break;

//	case CASE(ORSH, TUINT64):
//	case CASE(ORSH, TPTR64):
//		a = ASHRQ;
//		break;

//	case CASE(ORSH, TINT8):
//		a = ASARB;
//		break;

//	case CASE(ORSH, TINT16):
//		a = ASARW;
//		break;

//	case CASE(ORSH, TINT32):
//		a = ASARL;
//		break;

//	case CASE(ORSH, TINT64):
//		a = ASARQ;
//		break;

//	case CASE(OMUL, TINT8):
//	case CASE(OMUL, TUINT8):
//		a = AIMULB;
//		break;

//	case CASE(OMUL, TINT16):
//	case CASE(OMUL, TUINT16):
//		a = AIMULW;
//		break;

//	case CASE(OMUL, TINT32):
//	case CASE(OMUL, TUINT32):
//	case CASE(OMUL, TPTR32):
//		a = AIMULL;
//		break;

//	case CASE(OMUL, TINT64):
//	case CASE(OMUL, TUINT64):
//	case CASE(OMUL, TPTR64):
//		a = AIMULQ;
//		break;

//	case CASE(OMUL, TFLOAT32):
//		a = AMULSS;
//		break;

//	case CASE(OMUL, TFLOAT64):
//		a = AMULSD;
//		break;

//	case CASE(ODIV, TINT8):
//	case CASE(OMOD, TINT8):
//		a = AIDIVB;
//		break;

//	case CASE(ODIV, TUINT8):
//	case CASE(OMOD, TUINT8):
//		a = ADIVB;
//		break;

//	case CASE(ODIV, TINT16):
//	case CASE(OMOD, TINT16):
//		a = AIDIVW;
//		break;

//	case CASE(ODIV, TUINT16):
//	case CASE(OMOD, TUINT16):
//		a = ADIVW;
//		break;

//	case CASE(ODIV, TINT32):
//	case CASE(OMOD, TINT32):
//		a = AIDIVL;
//		break;

//	case CASE(ODIV, TUINT32):
//	case CASE(ODIV, TPTR32):
//	case CASE(OMOD, TUINT32):
//	case CASE(OMOD, TPTR32):
//		a = ADIVL;
//		break;

//	case CASE(ODIV, TINT64):
//	case CASE(OMOD, TINT64):
//		a = AIDIVQ;
//		break;

//	case CASE(ODIV, TUINT64):
//	case CASE(ODIV, TPTR64):
//	case CASE(OMOD, TUINT64):
//	case CASE(OMOD, TPTR64):
//		a = ADIVQ;
//		break;

//	case CASE(OEXTEND, TINT16):
//		a = ACWD;
//		break;

//	case CASE(OEXTEND, TINT32):
//		a = ACDQ;
//		break;

//	case CASE(OEXTEND, TINT64):
//		a = ACQO;
//		break;

//	case CASE(ODIV, TFLOAT32):
//		a = ADIVSS;
//		break;

//	case CASE(ODIV, TFLOAT64):
//		a = ADIVSD;
//		break;

	}
	return a;
}

enum
{
	ODynam	= 1<<0,
	OPtrto	= 1<<1,
};

static	Node	clean[20];
static	int	cleani = 0;

void
sudoclean(void)
{
	if(clean[cleani-1].op != OEMPTY)
		regfree(&clean[cleani-1]);
	if(clean[cleani-2].op != OEMPTY)
		regfree(&clean[cleani-2]);
	cleani -= 2;
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
	int o, i, w;
	int oary[10];
	int64 v;
	Node n1, n2, n3, *nn, *l, *r;
	Node *reg, *reg1;
	Prog *p1;
	Type *t;

	if(n->type == T)
		return 0;

	switch(n->op) {
	case OLITERAL:
		if(n->val.ctype != CTINT)
			break;
		v = mpgetfix(n->val.u.xval);
		if(v >= 32000 || v <= -32000)
			break;
		goto lit;

	case ODOT:
	case ODOTPTR:
		cleani += 2;
		reg = &clean[cleani-1];
		reg1 = &clean[cleani-2];
		reg->op = OEMPTY;
		reg1->op = OEMPTY;
		goto odot;

	case OINDEX:
		cleani += 2;
		reg = &clean[cleani-1];
		reg1 = &clean[cleani-2];
		reg->op = OEMPTY;
		reg1->op = OEMPTY;
		goto oindex;
	}
	return 0;

lit:
	fatal("sudoaddable lit not implemented");
//	switch(as) {
//	default:
//		return 0;
//	case AADDB: case AADDW: case AADDL: case AADDQ:
//	case ASUBB: case ASUBW: case ASUBL: case ASUBQ:
//	case AANDB: case AANDW: case AANDL: case AANDQ:
//	case AORB:  case AORW:  case AORL:  case AORQ:
//	case AXORB: case AXORW: case AXORL: case AXORQ:
//	case AINCB: case AINCW: case AINCL: case AINCQ:
//	case ADECB: case ADECW: case ADECL: case ADECQ:
//	case AMOVB: case AMOVW: case AMOVL: case AMOVQ:
//		break;
//	}

//	cleani += 2;
//	reg = &clean[cleani-1];
//	reg1 = &clean[cleani-2];
//	reg->op = OEMPTY;
//	reg1->op = OEMPTY;
//	naddr(n, a);
//	goto yes;

odot:
	o = dotoffset(n, oary, &nn);
	if(nn == N)
		goto no;

	if(nn->addable && o == 1 && oary[0] >= 0) {
		// directly addressable set of DOTs
		n1 = *nn;
		n1.type = n->type;
		n1.xoffset += oary[0];
		naddr(&n1, a);
		goto yes;
	}

	regalloc(reg, types[tptr], N);
	n1 = *reg;
	n1.op = OINDREG;
	if(oary[0] >= 0) {
		agen(nn, reg);
		n1.xoffset = oary[0];
	} else {
		cgen(nn, reg);
		n1.xoffset = -(oary[0]+1);
	}

	fatal("sudoaddable odot not implemented");
//	for(i=1; i<o; i++) {
//		if(oary[i] >= 0)
//			fatal("cant happen");
//		gins(AMOVQ, &n1, reg);
//		n1.xoffset = -(oary[i]+1);
//	}

	a->type = D_NONE;
	a->index = D_NONE;
	naddr(&n1, a);
	goto yes;

oindex:
	l = n->left;
	r = n->right;
	if(l->ullman >= UINF && r->ullman >= UINF)
		goto no;

	// set o to type of array
	o = 0;
	if(isptr[l->type->etype]) {
		o += OPtrto;
		if(l->type->type->etype != TARRAY)
			fatal("not ptr ary");
		if(l->type->type->bound < 0)
			o += ODynam;
	} else {
		if(l->type->etype != TARRAY)
			fatal("not ary");
		if(l->type->bound < 0)
			o += ODynam;
	}

	w = n->type->width;
	if(isconst(r, CTINT))
		goto oindex_const;

	switch(w) {
	default:
		goto no;
	case 1:
	case 2:
	case 4:
	case 8:
		break;
	}

	// load the array (reg)
	if(l->ullman > r->ullman) {
		regalloc(reg, types[tptr], N);
		if(o & OPtrto)
			cgen(l, reg);
		else
			agen(l, reg);
	}

	// load the index (reg1)
	t = types[TUINT64];
	if(issigned[r->type->etype])
		t = types[TINT64];
	regalloc(reg1, t, N);
	regalloc(&n3, r->type, reg1);
	cgen(r, &n3);
	gmove(&n3, reg1);
	regfree(&n3);

	// load the array (reg)
	if(l->ullman <= r->ullman) {
		regalloc(reg, types[tptr], N);
		if(o & OPtrto)
			cgen(l, reg);
		else
			agen(l, reg);
	}

	// check bounds
	if(!debug['B']) {
		if(o & ODynam) {
			n2 = *reg;
			n2.op = OINDREG;
			n2.type = types[tptr];
			n2.xoffset = Array_nel;
		} else {
			nodconst(&n2, types[TUINT64], l->type->bound);
			if(o & OPtrto)
				nodconst(&n2, types[TUINT64], l->type->type->bound);
		}
		gins(optoas(OCMP, types[TUINT32]), reg1, &n2);
		p1 = gbranch(optoas(OLT, types[TUINT32]), T);
		ginscall(throwindex, 0);
		patch(p1, pc);
	}

	if(o & ODynam) {
		n2 = *reg;
		n2.op = OINDREG;
		n2.type = types[tptr];
		n2.xoffset = Array_array;
		gmove(&n2, reg);
	}

	fatal("sudoaddable oindex not implemented");
//	naddr(reg1, a);
//	a->offset = 0;
//	a->scale = w;
//	a->index = a->type;
//	a->type = reg->val.u.reg + D_INDIR;

	goto yes;

oindex_const:
	// index is constant
	// can check statically and
	// can multiply by width statically

	regalloc(reg, types[tptr], N);
	if(o & OPtrto)
		cgen(l, reg);
	else
		agen(l, reg);

	v = mpgetfix(r->val.u.xval);
	if(o & ODynam) {

		if(!debug['B']) {
			n1 = *reg;
			n1.op = OINDREG;
			n1.type = types[tptr];
			n1.xoffset = Array_nel;
			nodconst(&n2, types[TUINT64], v);
			gins(optoas(OCMP, types[TUINT32]), &n1, &n2);
			p1 = gbranch(optoas(OGT, types[TUINT32]), T);
			ginscall(throwindex, 0);
			patch(p1, pc);
		}

		n1 = *reg;
		n1.op = OINDREG;
		n1.type = types[tptr];
		n1.xoffset = Array_array;
		gmove(&n1, reg);

	} else
	if(!debug['B']) {
		if(v < 0) {
			yyerror("out of bounds on array");
		} else
		if(o & OPtrto) {
			if(v >= l->type->type->bound)
				yyerror("out of bounds on array");
		} else
		if(v >= l->type->bound) {
			yyerror("out of bounds on array");
		}
	}

	n2 = *reg;
	n2.op = OINDREG;
	n2.xoffset = v*w;
	a->type = D_NONE;
	a->index = D_NONE;
	naddr(&n2, a);
	goto yes;

yes:
	return 1;

no:
	sudoclean();
	return 0;
}
