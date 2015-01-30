// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#undef	EXTERN
#define	EXTERN
#include <u.h>
#include <libc.h>
#include "gg.h"
#include "opt.h"

static Prog *appendpp(Prog *p, int as, int ftype, int freg, vlong foffset, int ttype, int treg, vlong toffset);
static Prog *zerorange(Prog *p, vlong frame, vlong lo, vlong hi);

void
defframe(Prog *ptxt)
{
	uint32 frame;
	Prog *p;
	vlong hi, lo;
	NodeList *l;
	Node *n;

	// fill in argument size, stack size
	ptxt->to.type = TYPE_TEXTSIZE;
	ptxt->to.u.argsize = rnd(curfn->type->argwid, widthptr);
	frame = rnd(stksize+maxarg, widthreg);
	ptxt->to.offset = frame;
	
	// insert code to zero ambiguously live variables
	// so that the garbage collector only sees initialized values
	// when it looks for pointers.
	p = ptxt;
	lo = hi = 0;
	// iterate through declarations - they are sorted in decreasing xoffset order.
	for(l=curfn->dcl; l != nil; l = l->next) {
		n = l->n;
		if(!n->needzero)
			continue;
		if(n->class != PAUTO)
			fatal("needzero class %d", n->class);
		if(n->type->width % widthptr != 0 || n->xoffset % widthptr != 0 || n->type->width == 0)
			fatal("var %lN has size %d offset %d", n, (int)n->type->width, (int)n->xoffset);

		if(lo != hi && n->xoffset + n->type->width >= lo - 2*widthreg) {
			// merge with range we already have
			lo = n->xoffset;
			continue;
		}
		// zero old range
		p = zerorange(p, frame, lo, hi);

		// set new range
		hi = n->xoffset + n->type->width;
		lo = n->xoffset;
	}
	// zero final range
	zerorange(p, frame, lo, hi);
}

static Prog*
zerorange(Prog *p, vlong frame, vlong lo, vlong hi)
{
	vlong cnt, i;
	Prog *p1;
	Node *f;

	cnt = hi - lo;
	if(cnt == 0)
		return p;
	if(cnt < 4*widthptr) {
		for(i = 0; i < cnt; i += widthptr)
			p = appendpp(p, AMOVD, TYPE_REG, REGZERO, 0, TYPE_MEM, REGSP, 8+frame+lo+i);
	} else if(cnt <= 128*widthptr) {
		p = appendpp(p, AADD, TYPE_CONST, 0, 8+frame+lo-8, TYPE_REG, REGRT1, 0);
		p->reg = REGSP;
		p = appendpp(p, ADUFFZERO, TYPE_NONE, 0, 0, TYPE_MEM, 0, 0);
		f = sysfunc("duffzero");
		naddr(f, &p->to, 1);
		afunclit(&p->to, f);
		p->to.offset = 4*(128-cnt/widthptr);
	} else {
		p = appendpp(p, AMOVD, TYPE_CONST, 0, 8+frame+lo-8, TYPE_REG, REGTMP, 0);
		p = appendpp(p, AADD, TYPE_REG, REGTMP, 0, TYPE_REG, REGRT1, 0);
		p->reg = REGSP;
		p = appendpp(p, AMOVD, TYPE_CONST, 0, cnt, TYPE_REG, REGTMP, 0);
		p = appendpp(p, AADD, TYPE_REG, REGTMP, 0, TYPE_REG, REGRT2, 0);
		p->reg = REGRT1;
		p1 = p = appendpp(p, AMOVDU, TYPE_REG, REGZERO, 0, TYPE_MEM, REGRT1, widthptr);
		p = appendpp(p, ACMP, TYPE_REG, REGRT1, 0, TYPE_REG, REGRT2, 0);
		p = appendpp(p, ABNE, TYPE_NONE, 0, 0, TYPE_BRANCH, 0, 0);
		patch(p, p1);
	}
	return p;
}

static Prog*
appendpp(Prog *p, int as, int ftype, int freg, vlong foffset, int ttype, int treg, vlong toffset)
{
	Prog *q;
	q = mal(sizeof(*q));
	clearp(q);
	q->as = as;
	q->lineno = p->lineno;
	q->from.type = ftype;
	q->from.reg = freg;
	q->from.offset = foffset;
	q->to.type = ttype;
	q->to.reg = treg;
	q->to.offset = toffset;
	q->link = p->link;
	p->link = q;
	return q;
}

// Sweep the prog list to mark any used nodes.
void
markautoused(Prog *p)
{
	for (; p; p = p->link) {
		if (p->as == ATYPE || p->as == AVARDEF || p->as == AVARKILL)
			continue;

		if (p->from.node)
			((Node*)(p->from.node))->used = 1;

		if (p->to.node)
			((Node*)(p->to.node))->used = 1;
	}
}

// Fixup instructions after allocauto (formerly compactframe) has moved all autos around.
void
fixautoused(Prog *p)
{
	Prog **lp;

	for (lp=&p; (p=*lp) != P; ) {
		if (p->as == ATYPE && p->from.node && p->from.name == NAME_AUTO && !((Node*)(p->from.node))->used) {
			*lp = p->link;
			continue;
		}
		if ((p->as == AVARDEF || p->as == AVARKILL) && p->to.node && !((Node*)(p->to.node))->used) {
			// Cannot remove VARDEF instruction, because - unlike TYPE handled above -
			// VARDEFs are interspersed with other code, and a jump might be using the
			// VARDEF as a target. Replace with a no-op instead. A later pass will remove
			// the no-ops.
			nopout(p);
			continue;
		}
		if (p->from.name == NAME_AUTO && p->from.node)
			p->from.offset += ((Node*)(p->from.node))->stkdelta;

		if (p->to.name == NAME_AUTO && p->to.node)
			p->to.offset += ((Node*)(p->to.node))->stkdelta;

		lp = &p->link;
	}
}

/*
 * generate: BL reg, f
 * where both reg and f are registers.
 * On power, f must be moved to CTR first.
 */
static void
ginsBL(Node *reg, Node *f)
{
	Prog *p;
	p = gins(AMOVD, f, N);
	p->to.type = TYPE_REG;
	p->to.reg = REG_CTR;
	p = gins(ABL, reg, N);
	p->to.type = TYPE_REG;
	p->to.reg = REG_CTR;
}

/*
 * generate:
 *	call f
 *	proc=-1	normal call but no return
 *	proc=0	normal call
 *	proc=1	goroutine run in new proc
 *	proc=2	defer call save away stack
  *	proc=3	normal call to C pointer (not Go func value)
 */
void
ginscall(Node *f, int proc)
{
	Prog *p;
	Node reg, con, reg2;
	Node r1;
	int32 extra;

	if(f->type != T) {
		extra = 0;
		if(proc == 1 || proc == 2)
			extra = 2 * widthptr;
		setmaxarg(f->type, extra);
	}

	switch(proc) {
	default:
		fatal("ginscall: bad proc %d", proc);
		break;

	case 0:	// normal call
	case -1:	// normal call but no return
		if(f->op == ONAME && f->class == PFUNC) {
			if(f == deferreturn) {
				// Deferred calls will appear to be returning to
				// the CALL deferreturn(SB) that we are about to emit.
				// However, the stack trace code will show the line
				// of the instruction byte before the return PC. 
				// To avoid that being an unrelated instruction,
				// insert a ppc64 NOP that we will have the right line number.
				// The ppc64 NOP is really or r0, r0, r0; use that description
				// because the NOP pseudo-instruction would be removed by
				// the linker.
				nodreg(&reg, types[TINT], REG_R0);
				gins(AOR, &reg, &reg);
			}
			p = gins(ABL, N, f);
			afunclit(&p->to, f);
			if(proc == -1 || noreturn(p))
				gins(AUNDEF, N, N);
			break;
		}
		nodreg(&reg, types[tptr], REGENV);
		nodreg(&r1, types[tptr], REG_R3);
		gmove(f, &reg);
		reg.op = OINDREG;
		gmove(&reg, &r1);
		reg.op = OREGISTER;
		ginsBL(&reg, &r1);
		break;
	
	case 3:	// normal call of c function pointer
		ginsBL(N, f);
		break;

	case 1:	// call in new proc (go)
	case 2:	// deferred call (defer)
		nodconst(&con, types[TINT64], argsize(f->type));
		nodreg(&reg, types[TINT64], REG_R3);
		nodreg(&reg2, types[TINT64], REG_R4);
		gmove(f, &reg);

		gmove(&con, &reg2);
		p = gins(AMOVW, &reg2, N);
		p->to.type = TYPE_MEM;
		p->to.reg = REGSP;
		p->to.offset = 8;

		p = gins(AMOVD, &reg, N);
		p->to.type = TYPE_MEM;
		p->to.reg = REGSP;
		p->to.offset = 16;

		if(proc == 1)
			ginscall(newproc, 0);
		else {
			if(!hasdefer)
				fatal("hasdefer=0 but has defer");
			ginscall(deferproc, 0);
		}

		if(proc == 2) {
			nodreg(&reg, types[TINT64], REG_R3);
			p = gins(ACMP, &reg, N);
			p->to.type = TYPE_REG;
			p->to.reg = REG_R0;
			p = gbranch(ABEQ, T, +1);
			cgen_ret(N);
			patch(p, pc);
		}
		break;
	}
}

/*
 * n is call to interface method.
 * generate res = n.
 */
void
cgen_callinter(Node *n, Node *res, int proc)
{
	Node *i, *f;
	Node tmpi, nodi, nodo, nodr, nodsp;
	Prog *p;

	i = n->left;
	if(i->op != ODOTINTER)
		fatal("cgen_callinter: not ODOTINTER %O", i->op);

	f = i->right;		// field
	if(f->op != ONAME)
		fatal("cgen_callinter: not ONAME %O", f->op);

	i = i->left;		// interface

	if(!i->addable) {
		tempname(&tmpi, i->type);
		cgen(i, &tmpi);
		i = &tmpi;
	}

	genlist(n->list);		// assign the args

	// i is now addable, prepare an indirected
	// register to hold its address.
	igen(i, &nodi, res);		// REG = &inter

	nodindreg(&nodsp, types[tptr], REGSP);
	nodsp.xoffset = widthptr;
	if(proc != 0)
		nodsp.xoffset += 2 * widthptr; // leave room for size & fn
	nodi.type = types[tptr];
	nodi.xoffset += widthptr;
	cgen(&nodi, &nodsp);	// {8 or 24}(SP) = 8(REG) -- i.data

	regalloc(&nodo, types[tptr], res);
	nodi.type = types[tptr];
	nodi.xoffset -= widthptr;
	cgen(&nodi, &nodo);	// REG = 0(REG) -- i.tab
	regfree(&nodi);

	regalloc(&nodr, types[tptr], &nodo);
	if(n->left->xoffset == BADWIDTH)
		fatal("cgen_callinter: badwidth");
	cgen_checknil(&nodo); // in case offset is huge
	nodo.op = OINDREG;
	nodo.xoffset = n->left->xoffset + 3*widthptr + 8;
	if(proc == 0) {
		// plain call: use direct c function pointer - more efficient
		cgen(&nodo, &nodr);	// REG = 32+offset(REG) -- i.tab->fun[f]
		proc = 3;
	} else {
		// go/defer. generate go func value.
		p = gins(AMOVD, &nodo, &nodr);	// REG = &(32+offset(REG)) -- i.tab->fun[f]
		p->from.type = TYPE_ADDR;
	}

	nodr.type = n->left->type;
	ginscall(&nodr, proc);

	regfree(&nodr);
	regfree(&nodo);
}

/*
 * generate function call;
 *	proc=0	normal call
 *	proc=1	goroutine run in new proc
 *	proc=2	defer call save away stack
 */
void
cgen_call(Node *n, int proc)
{
	Type *t;
	Node nod, afun;

	if(n == N)
		return;

	if(n->left->ullman >= UINF) {
		// if name involves a fn call
		// precompute the address of the fn
		tempname(&afun, types[tptr]);
		cgen(n->left, &afun);
	}

	genlist(n->list);		// assign the args
	t = n->left->type;

	// call tempname pointer
	if(n->left->ullman >= UINF) {
		regalloc(&nod, types[tptr], N);
		cgen_as(&nod, &afun);
		nod.type = t;
		ginscall(&nod, proc);
		regfree(&nod);
		return;
	}

	// call pointer
	if(n->left->op != ONAME || n->left->class != PFUNC) {
		regalloc(&nod, types[tptr], N);
		cgen_as(&nod, n->left);
		nod.type = t;
		ginscall(&nod, proc);
		regfree(&nod);
		return;
	}

	// call direct
	n->left->method = 1;
	ginscall(n->left, proc);
}

/*
 * call to n has already been generated.
 * generate:
 *	res = return value from call.
 */
void
cgen_callret(Node *n, Node *res)
{
	Node nod;
	Type *fp, *t;
	Iter flist;

	t = n->left->type;
	if(t->etype == TPTR32 || t->etype == TPTR64)
		t = t->type;

	fp = structfirst(&flist, getoutarg(t));
	if(fp == T)
		fatal("cgen_callret: nil");

	memset(&nod, 0, sizeof(nod));
	nod.op = OINDREG;
	nod.val.u.reg = REGSP;
	nod.addable = 1;

	nod.xoffset = fp->width + widthptr; // +widthptr: saved LR at 0(R1)
	nod.type = fp->type;
	cgen_as(res, &nod);
}

/*
 * call to n has already been generated.
 * generate:
 *	res = &return value from call.
 */
void
cgen_aret(Node *n, Node *res)
{
	Node nod1, nod2;
	Type *fp, *t;
	Iter flist;

	t = n->left->type;
	if(isptr[t->etype])
		t = t->type;

	fp = structfirst(&flist, getoutarg(t));
	if(fp == T)
		fatal("cgen_aret: nil");

	memset(&nod1, 0, sizeof(nod1));
	nod1.op = OINDREG;
	nod1.val.u.reg = REGSP;
	nod1.addable = 1;

	nod1.xoffset = fp->width + widthptr; // +widthptr: saved lr at 0(SP)
	nod1.type = fp->type;

	if(res->op != OREGISTER) {
		regalloc(&nod2, types[tptr], res);
		agen(&nod1, &nod2);
		gins(AMOVD, &nod2, res);
		regfree(&nod2);
	} else
		agen(&nod1, res);
}

/*
 * generate return.
 * n->left is assignments to return values.
 */
void
cgen_ret(Node *n)
{
	Prog *p;

	if(n != N)
		genlist(n->list);		// copy out args
	if(hasdefer)
		ginscall(deferreturn, 0);
	genlist(curfn->exit);
	p = gins(ARET, N, N);
	if(n != N && n->op == ORETJMP) {
		p->to.name = NAME_EXTERN;
		p->to.type = TYPE_ADDR;
		p->to.sym = linksym(n->left->sym);
	}
}

void
cgen_asop(Node *n)
{
	USED(n);
	fatal("cgen_asop"); // no longer used
}

int
samereg(Node *a, Node *b)
{
	if(a == N || b == N)
		return 0;
	if(a->op != OREGISTER)
		return 0;
	if(b->op != OREGISTER)
		return 0;
	if(a->val.u.reg != b->val.u.reg)
		return 0;
	return 1;
}

/*
 * generate division.
 * generates one of:
 *	res = nl / nr
 *	res = nl % nr
 * according to op.
 */
void
dodiv(int op, Node *nl, Node *nr, Node *res)
{
	int a, check;
	Type *t, *t0;
	Node tl, tr, tl2, tr2, nm1, nz, tm;
	Prog *p1, *p2;

	// Have to be careful about handling
	// most negative int divided by -1 correctly.
	// The hardware will generate undefined result.
	// Also need to explicitly trap on division on zero,
	// the hardware will silently generate undefined result.
	// DIVW will leave unpredicable result in higher 32-bit,
	// so always use DIVD/DIVDU.
	t = nl->type;
	t0 = t;
	check = 0;
	if(issigned[t->etype]) {
		check = 1;
		if(isconst(nl, CTINT) && mpgetfix(nl->val.u.xval) != -(1ULL<<(t->width*8-1)))
			check = 0;
		else if(isconst(nr, CTINT) && mpgetfix(nr->val.u.xval) != -1)
			check = 0;
	}
	if(t->width < 8) {
		if(issigned[t->etype])
			t = types[TINT64];
		else
			t = types[TUINT64];
		check = 0;
	}

	a = optoas(ODIV, t);

	regalloc(&tl, t0, N);
	regalloc(&tr, t0, N);
	if(nl->ullman >= nr->ullman) {
		cgen(nl, &tl);
		cgen(nr, &tr);
	} else {
		cgen(nr, &tr);
		cgen(nl, &tl);
	}
	if(t != t0) {
		// Convert
		tl2 = tl;
		tr2 = tr;
		tl.type = t;
		tr.type = t;
		gmove(&tl2, &tl);
		gmove(&tr2, &tr);
	}

	// Handle divide-by-zero panic.
	p1 = gins(optoas(OCMP, t), &tr, N);
	p1->to.type = TYPE_REG;
	p1->to.reg = REGZERO;
	p1 = gbranch(optoas(ONE, t), T, +1);
	if(panicdiv == N)
		panicdiv = sysfunc("panicdivide");
	ginscall(panicdiv, -1);
	patch(p1, pc);

	if(check) {
		nodconst(&nm1, t, -1);
		gins(optoas(OCMP, t), &tr, &nm1);
		p1 = gbranch(optoas(ONE, t), T, +1);
		if(op == ODIV) {
			// a / (-1) is -a.
			gins(optoas(OMINUS, t), N, &tl);
			gmove(&tl, res);
		} else {
			// a % (-1) is 0.
			nodconst(&nz, t, 0);
			gmove(&nz, res);
		}
		p2 = gbranch(AJMP, T, 0);
		patch(p1, pc);
	}
	p1 = gins(a, &tr, &tl);
	if(op == ODIV) {
		regfree(&tr);
		gmove(&tl, res);
	} else {
		// A%B = A-(A/B*B)
		regalloc(&tm, t, N);
		// patch div to use the 3 register form
		// TODO(minux): add gins3?
		p1->reg = p1->to.reg;
		p1->to.reg = tm.val.u.reg;
		gins(optoas(OMUL, t), &tr, &tm);
		regfree(&tr);
		gins(optoas(OSUB, t), &tm, &tl);
		regfree(&tm);
		gmove(&tl, res);
	}
	regfree(&tl);
	if(check)
		patch(p2, pc);
}

/*
 * generate division according to op, one of:
 *	res = nl / nr
 *	res = nl % nr
 */
void
cgen_div(int op, Node *nl, Node *nr, Node *res)
{
	Node n1, n2, n3;
	int w, a;
	Magic m;

	// TODO(minux): enable division by magic multiply (also need to fix longmod below)
	//if(nr->op != OLITERAL)
		goto longdiv;
	w = nl->type->width*8;

	// Front end handled 32-bit division. We only need to handle 64-bit.
	// try to do division by multiply by (2^w)/d
	// see hacker's delight chapter 10
	switch(simtype[nl->type->etype]) {
	default:
		goto longdiv;

	case TUINT64:
		m.w = w;
		m.ud = mpgetfix(nr->val.u.xval);
		umagic(&m);
		if(m.bad)
			break;
		if(op == OMOD)
			goto longmod;

		cgenr(nl, &n1, N);
		nodconst(&n2, nl->type, m.um);
		regalloc(&n3, nl->type, res);
		cgen_hmul(&n1, &n2, &n3);

		if(m.ua) {
			// need to add numerator accounting for overflow
			gins(optoas(OADD, nl->type), &n1, &n3);
			nodconst(&n2, nl->type, 1);
			gins(optoas(ORROTC, nl->type), &n2, &n3);
			nodconst(&n2, nl->type, m.s-1);
			gins(optoas(ORSH, nl->type), &n2, &n3);
		} else {
			nodconst(&n2, nl->type, m.s);
			gins(optoas(ORSH, nl->type), &n2, &n3);	// shift dx
		}

		gmove(&n3, res);
		regfree(&n1);
		regfree(&n3);
		return;

	case TINT64:
		m.w = w;
		m.sd = mpgetfix(nr->val.u.xval);
		smagic(&m);
		if(m.bad)
			break;
		if(op == OMOD)
			goto longmod;

		cgenr(nl, &n1, res);
		nodconst(&n2, nl->type, m.sm);
		regalloc(&n3, nl->type, N);
		cgen_hmul(&n1, &n2, &n3);

		if(m.sm < 0) {
			// need to add numerator
			gins(optoas(OADD, nl->type), &n1, &n3);
		}

		nodconst(&n2, nl->type, m.s);
		gins(optoas(ORSH, nl->type), &n2, &n3);	// shift n3

		nodconst(&n2, nl->type, w-1);
		gins(optoas(ORSH, nl->type), &n2, &n1);	// -1 iff num is neg
		gins(optoas(OSUB, nl->type), &n1, &n3);	// added

		if(m.sd < 0) {
			// this could probably be removed
			// by factoring it into the multiplier
			gins(optoas(OMINUS, nl->type), N, &n3);
		}

		gmove(&n3, res);
		regfree(&n1);
		regfree(&n3);
		return;
	}
	goto longdiv;

longdiv:
	// division and mod using (slow) hardware instruction
	dodiv(op, nl, nr, res);
	return;

longmod:
	// mod using formula A%B = A-(A/B*B) but
	// we know that there is a fast algorithm for A/B
	regalloc(&n1, nl->type, res);
	cgen(nl, &n1);
	regalloc(&n2, nl->type, N);
	cgen_div(ODIV, &n1, nr, &n2);
	a = optoas(OMUL, nl->type);
	if(w == 8) {
		// use 2-operand 16-bit multiply
		// because there is no 2-operand 8-bit multiply
		//a = AIMULW;
	}
	if(!smallintconst(nr)) {
		regalloc(&n3, nl->type, N);
		cgen(nr, &n3);
		gins(a, &n3, &n2);
		regfree(&n3);
	} else
		gins(a, nr, &n2);
	gins(optoas(OSUB, nl->type), &n2, &n1);
	gmove(&n1, res);
	regfree(&n1);
	regfree(&n2);
}

/*
 * generate high multiply:
 *   res = (nl*nr) >> width
 */
void
cgen_hmul(Node *nl, Node *nr, Node *res)
{
	int w;
	Node n1, n2, *tmp;
	Type *t;
	Prog *p;

	// largest ullman on left.
	if(nl->ullman < nr->ullman) {
		tmp = nl;
		nl = nr;
		nr = tmp;
	}
	t = nl->type;
	w = t->width * 8;
	cgenr(nl, &n1, res);
	cgenr(nr, &n2, N);
	switch(simtype[t->etype]) {
	case TINT8:
	case TINT16:
	case TINT32:
		gins(optoas(OMUL, t), &n2, &n1);
		p = gins(ASRAD, N, &n1);
		p->from.type = TYPE_CONST;
		p->from.offset = w;
		break;
	case TUINT8:
	case TUINT16:
	case TUINT32:
		gins(optoas(OMUL, t), &n2, &n1);
		p = gins(ASRD, N, &n1);
		p->from.type = TYPE_CONST;
		p->from.offset = w;
		break;
	case TINT64:
	case TUINT64:
		if(issigned[t->etype])
			p = gins(AMULHD, &n2, &n1);
		else
			p = gins(AMULHDU, &n2, &n1);
		break;
	default:
		fatal("cgen_hmul %T", t);
		break;
	}
	cgen(&n1, res);
	regfree(&n1);
	regfree(&n2);
}

/*
 * generate shift according to op, one of:
 *	res = nl << nr
 *	res = nl >> nr
 */
void
cgen_shift(int op, int bounded, Node *nl, Node *nr, Node *res)
{
	Node n1, n2, n3, n4, n5;
	int a;
	Prog *p1;
	uvlong sc;
	Type *tcount;

	a = optoas(op, nl->type);

	if(nr->op == OLITERAL) {
		regalloc(&n1, nl->type, res);
		cgen(nl, &n1);
		sc = mpgetfix(nr->val.u.xval);
		if(sc >= nl->type->width*8) {
			// large shift gets 2 shifts by width-1
			nodconst(&n3, types[TUINT32], nl->type->width*8-1);
			gins(a, &n3, &n1);
			gins(a, &n3, &n1);
		} else
			gins(a, nr, &n1);
		gmove(&n1, res);
		regfree(&n1);
		goto ret;
	}

	if(nl->ullman >= UINF) {
		tempname(&n4, nl->type);
		cgen(nl, &n4);
		nl = &n4;
	}
	if(nr->ullman >= UINF) {
		tempname(&n5, nr->type);
		cgen(nr, &n5);
		nr = &n5;
	}

	// Allow either uint32 or uint64 as shift type,
	// to avoid unnecessary conversion from uint32 to uint64
	// just to do the comparison.
	tcount = types[simtype[nr->type->etype]];
	if(tcount->etype < TUINT32)
		tcount = types[TUINT32];

	regalloc(&n1, nr->type, N);		// to hold the shift type in CX
	regalloc(&n3, tcount, &n1);	// to clear high bits of CX

	regalloc(&n2, nl->type, res);
	if(nl->ullman >= nr->ullman) {
		cgen(nl, &n2);
		cgen(nr, &n1);
		gmove(&n1, &n3);
	} else {
		cgen(nr, &n1);
		gmove(&n1, &n3);
		cgen(nl, &n2);
	}
	regfree(&n3);

	// test and fix up large shifts
	if(!bounded) {
		nodconst(&n3, tcount, nl->type->width*8);
		gins(optoas(OCMP, tcount), &n1, &n3);
		p1 = gbranch(optoas(OLT, tcount), T, +1);
		if(op == ORSH && issigned[nl->type->etype]) {
			nodconst(&n3, types[TUINT32], nl->type->width*8-1);
			gins(a, &n3, &n2);
		} else {
			nodconst(&n3, nl->type, 0);
			gmove(&n3, &n2);
		}
		patch(p1, pc);
	}

	gins(a, &n1, &n2);

	gmove(&n2, res);

	regfree(&n1);
	regfree(&n2);

ret:
	;
}

void
clearfat(Node *nl)
{
	uint64 w, c, q, t, boff;
	Node dst, end, r0, *f;
	Prog *p, *pl;

	/* clear a fat object */
	if(debug['g']) {
		print("clearfat %N (%T, size: %lld)\n", nl, nl->type, nl->type->width);
	}

	w = nl->type->width;
	// Avoid taking the address for simple enough types.
	//if(componentgen(N, nl))
	//	return;

	c = w % 8;	// bytes
	q = w / 8;	// dwords

	if(reg[REGRT1] > 0)
		fatal("R%d in use during clearfat", REGRT1);

	nodreg(&r0, types[TUINT64], REG_R0); // r0 is always zero
	nodreg(&dst, types[tptr], REGRT1);
	reg[REGRT1]++;
	agen(nl, &dst);

	if(q > 128) {
		p = gins(ASUB, N, &dst);
		p->from.type = TYPE_CONST;
		p->from.offset = 8;

		regalloc(&end, types[tptr], N);
		p = gins(AMOVD, &dst, &end);
		p->from.type = TYPE_ADDR;
		p->from.offset = q*8;

		p = gins(AMOVDU, &r0, &dst);
		p->to.type = TYPE_MEM;
		p->to.offset = 8;
		pl = p;

		p = gins(ACMP, &dst, &end);
		patch(gbranch(ABNE, T, 0), pl);

		regfree(&end);
		// The loop leaves R3 on the last zeroed dword
		boff = 8;
	} else if(q >= 4) {
		p = gins(ASUB, N, &dst);
		p->from.type = TYPE_CONST;
		p->from.offset = 8;
		f = sysfunc("duffzero");
		p = gins(ADUFFZERO, N, f);
		afunclit(&p->to, f);
		// 4 and 128 = magic constants: see ../../runtime/asm_ppc64x.s
		p->to.offset = 4*(128-q);
		// duffzero leaves R3 on the last zeroed dword
		boff = 8;
	} else {
		for(t = 0; t < q; t++) {
			p = gins(AMOVD, &r0, &dst);
			p->to.type = TYPE_MEM;
			p->to.offset = 8*t;
		}
		boff = 8*q;
	}

	for(t = 0; t < c; t++) {
		p = gins(AMOVB, &r0, &dst);
		p->to.type = TYPE_MEM;
		p->to.offset = t+boff;
	}
	reg[REGRT1]--;
}

// Called after regopt and peep have run.
// Expand CHECKNIL pseudo-op into actual nil pointer check.
void
expandchecks(Prog *firstp)
{
	Prog *p, *p1, *p2;

	for(p = firstp; p != P; p = p->link) {
		if(debug_checknil && ctxt->debugvlog)
			print("expandchecks: %P\n", p);
		if(p->as != ACHECKNIL)
			continue;
		if(debug_checknil && p->lineno > 1) // p->lineno==1 in generated wrappers
			warnl(p->lineno, "generated nil check");
		if(p->from.type != TYPE_REG)
			fatal("invalid nil check %P\n", p);
		/*
		// check is
		//	TD $4, R0, arg (R0 is always zero)
		// eqv. to:
		// 	tdeq r0, arg
		// NOTE: this needs special runtime support to make SIGTRAP recoverable.
		reg = p->from.reg;
		p->as = ATD;
		p->from = p->to = p->from3 = zprog.from;
		p->from.type = TYPE_CONST;
		p->from.offset = 4;
		p->from.reg = 0;
		p->reg = REG_R0;
		p->to.type = TYPE_REG;
		p->to.reg = reg;
		*/
		// check is
		//	CMP arg, R0
		//	BNE 2(PC) [likely]
		//	MOVD R0, 0(R0)
		p1 = mal(sizeof *p1);
		p2 = mal(sizeof *p2);
		clearp(p1);
		clearp(p2);
		p1->link = p2;
		p2->link = p->link;
		p->link = p1;
		p1->lineno = p->lineno;
		p2->lineno = p->lineno;
		p1->pc = 9999;
		p2->pc = 9999;
		p->as = ACMP;
		p->to.type = TYPE_REG;
		p->to.reg = REGZERO;
		p1->as = ABNE;
		//p1->from.type = TYPE_CONST;
		//p1->from.offset = 1; // likely
		p1->to.type = TYPE_BRANCH;
		p1->to.u.branch = p2->link;
		// crash by write to memory address 0.
		p2->as = AMOVD;
		p2->from.type = TYPE_REG;
		p2->from.reg = REG_R0;
		p2->to.type = TYPE_MEM;
		p2->to.reg = REG_R0;
		p2->to.offset = 0;
	}
}
