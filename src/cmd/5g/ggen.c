// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#undef	EXTERN
#define	EXTERN
#include <u.h>
#include <libc.h>
#include "gg.h"
#include "opt.h"

static Prog* appendpp(Prog*, int, int, int, int32, int, int, int32);
static Prog *zerorange(Prog *p, vlong frame, vlong lo, vlong hi, uint32 *r0);

void
defframe(Prog *ptxt)
{
	uint32 frame, r0;
	Prog *p;
	vlong hi, lo;
	NodeList *l;
	Node *n;

	// fill in argument size
	ptxt->to.type = D_CONST2;
	ptxt->to.offset2 = rnd(curfn->type->argwid, widthptr);

	// fill in final stack size
	frame = rnd(stksize+maxarg, widthptr);
	ptxt->to.offset = frame;
	
	// insert code to contain ambiguously live variables
	// so that garbage collector only sees initialized values
	// when it looks for pointers.
	p = ptxt;
	lo = hi = 0;
	r0 = 0;
	for(l=curfn->dcl; l != nil; l = l->next) {
		n = l->n;
		if(!n->needzero)
			continue;
		if(n->class != PAUTO)
			fatal("needzero class %d", n->class);
		if(n->type->width % widthptr != 0 || n->xoffset % widthptr != 0 || n->type->width == 0)
			fatal("var %lN has size %d offset %d", n, (int)n->type->width, (int)n->xoffset);
		if(lo != hi && n->xoffset + n->type->width >= lo - 2*widthptr) {
			// merge with range we already have
			lo = rnd(n->xoffset, widthptr);
			continue;
		}
		// zero old range
		p = zerorange(p, frame, lo, hi, &r0);

		// set new range
		hi = n->xoffset + n->type->width;
		lo = n->xoffset;
	}
	// zero final range
	zerorange(p, frame, lo, hi, &r0);
}

static Prog*
zerorange(Prog *p, vlong frame, vlong lo, vlong hi, uint32 *r0)
{
	vlong cnt, i;
	Prog *p1;
	Node *f;

	cnt = hi - lo;
	if(cnt == 0)
		return p;
	if(*r0 == 0) {
		p = appendpp(p, AMOVW, D_CONST, NREG, 0, D_REG, 0, 0);
		*r0 = 1;
	}
	if(cnt < 4*widthptr) {
		for(i = 0; i < cnt; i += widthptr) 
			p = appendpp(p, AMOVW, D_REG, 0, 0, D_OREG, REGSP, 4+frame+lo+i);
	} else if(!nacl && (cnt <= 128*widthptr)) {
		p = appendpp(p, AADD, D_CONST, NREG, 4+frame+lo, D_REG, 1, 0);
		p->reg = REGSP;
		p = appendpp(p, ADUFFZERO, D_NONE, NREG, 0, D_OREG, NREG, 0);
		f = sysfunc("duffzero");
		naddr(f, &p->to, 1);
		afunclit(&p->to, f);
		p->to.offset = 4*(128-cnt/widthptr);
	} else {
		p = appendpp(p, AADD, D_CONST, NREG, 4+frame+lo, D_REG, 1, 0);
		p->reg = REGSP;
		p = appendpp(p, AADD, D_CONST, NREG, cnt, D_REG, 2, 0);
		p->reg = 1;
		p1 = p = appendpp(p, AMOVW, D_REG, 0, 0, D_OREG, 1, 4);
		p->scond |= C_PBIT;
		p = appendpp(p, ACMP, D_REG, 1, 0, D_NONE, 0, 0);
		p->reg = 2;
		p = appendpp(p, ABNE, D_NONE, NREG, 0, D_BRANCH, NREG, 0);
		patch(p, p1);
	}
	return p;
}

static Prog*	
appendpp(Prog *p, int as, int ftype, int freg, int32 foffset, int ttype, int treg, int32 toffset)	
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
markautoused(Prog* p)
{
	for (; p; p = p->link) {
		if (p->as == ATYPE || p->as == AVARDEF || p->as == AVARKILL)
			continue;

		if (p->from.node)
			p->from.node->used = 1;

		if (p->to.node)
			p->to.node->used = 1;
	}
}

// Fixup instructions after allocauto (formerly compactframe) has moved all autos around.
void
fixautoused(Prog* p)
{
	Prog **lp;

	for (lp=&p; (p=*lp) != P; ) {
		if (p->as == ATYPE && p->from.node && p->from.name == D_AUTO && !p->from.node->used) {
			*lp = p->link;
			continue;
		}
		if ((p->as == AVARDEF || p->as == AVARKILL) && p->to.node && !p->to.node->used) {
			// Cannot remove VARDEF instruction, because - unlike TYPE handled above -
			// VARDEFs are interspersed with other code, and a jump might be using the
			// VARDEF as a target. Replace with a no-op instead. A later pass will remove
			// the no-ops.
			p->to.type = D_NONE;
			p->to.node = N;
			p->as = ANOP;
			continue;
		}

		if (p->from.name == D_AUTO && p->from.node)
			p->from.offset += p->from.node->stkdelta;

		if (p->to.name == D_AUTO && p->to.node)
			p->to.offset += p->to.node->stkdelta;

		lp = &p->link;
	}
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
	Node n1, r, r1, con;
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
				// the BL deferreturn(SB) that we are about to emit.
				// However, the stack trace code will show the line
				// of the instruction before that return PC. 
				// To avoid that instruction being an unrelated instruction,
				// insert a NOP so that we will have the right line number.
				// ARM NOP 0x00000000 is really AND.EQ R0, R0, R0.
				// Use the latter form because the NOP pseudo-instruction
				// would be removed by the linker.
				nodreg(&r, types[TINT], 0);
				p = gins(AAND, &r, &r);
				p->scond = C_SCOND_EQ;
			}
			p = gins(ABL, N, f);
			afunclit(&p->to, f);
			if(proc == -1 || noreturn(p))
				gins(AUNDEF, N, N);
			break;
		}
		nodreg(&r, types[tptr], 7);
		nodreg(&r1, types[tptr], 1);
		gmove(f, &r);
		r.op = OINDREG;
		gmove(&r, &r1);
		r.op = OREGISTER;
		r1.op = OINDREG;
		gins(ABL, &r, &r1);
		break;

	case 3:	// normal call of c function pointer
		gins(ABL, N, f);
		break;

	case 1:	// call in new proc (go)
	case 2:	// deferred call (defer)
		regalloc(&r, types[tptr], N);
		nodconst(&con, types[TINT32], argsize(f->type));
		gins(AMOVW, &con, &r);
		p = gins(AMOVW, &r, N);
		p->to.type = D_OREG;
		p->to.reg = REGSP;
		p->to.offset = 4;

		memset(&n1, 0, sizeof n1);
		n1.op = OADDR;
		n1.left = f;
		gins(AMOVW, &n1, &r);
		p = gins(AMOVW, &r, N);
		p->to.type = D_OREG;
		p->to.reg = REGSP;
		p->to.offset = 8;

		regfree(&r);

		if(proc == 1)
			ginscall(newproc, 0);
		else
			ginscall(deferproc, 0);

		if(proc == 2) {
			nodconst(&con, types[TINT32], 0);
			p = gins(ACMP, &con, N);
			p->reg = 0;
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
	int r;
	Node *i, *f;
	Node tmpi, nodo, nodr, nodsp;
	Prog *p;

	i = n->left;
	if(i->op != ODOTINTER)
		fatal("cgen_callinter: not ODOTINTER %O", i->op);

	f = i->right;		// field
	if(f->op != ONAME)
		fatal("cgen_callinter: not ONAME %O", f->op);

	i = i->left;		// interface

	// Release res register during genlist and cgen,
	// which might have their own function calls.
	r = -1;
	if(res != N && (res->op == OREGISTER || res->op == OINDREG)) {
		r = res->val.u.reg;
		reg[r]--;
	}

	if(!i->addable) {
		tempname(&tmpi, i->type);
		cgen(i, &tmpi);
		i = &tmpi;
	}

	genlist(n->list);			// args
	if(r >= 0)
		reg[r]++;

	regalloc(&nodr, types[tptr], res);
	regalloc(&nodo, types[tptr], &nodr);
	nodo.op = OINDREG;

	agen(i, &nodr);		// REG = &inter

	nodindreg(&nodsp, types[tptr], REGSP);
	nodsp.xoffset = widthptr;
	if(proc != 0)
		nodsp.xoffset += 2 * widthptr; // leave room for size & fn
	nodo.xoffset += widthptr;
	cgen(&nodo, &nodsp);	// {4 or 12}(SP) = 4(REG) -- i.data

	nodo.xoffset -= widthptr;
	cgen(&nodo, &nodr);	// REG = 0(REG) -- i.tab
	cgen_checknil(&nodr); // in case offset is huge

	nodo.xoffset = n->left->xoffset + 3*widthptr + 8;
	
	if(proc == 0) {
		// plain call: use direct c function pointer - more efficient
		cgen(&nodo, &nodr);	// REG = 20+offset(REG) -- i.tab->fun[f]
		nodr.op = OINDREG;
		proc = 3;
	} else {
		// go/defer. generate go func value.
		p = gins(AMOVW, &nodo, &nodr);
		p->from.type = D_CONST;	// REG = &(20+offset(REG)) -- i.tab->fun[f]
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
		goto ret;
	}

	// call pointer
	if(n->left->op != ONAME || n->left->class != PFUNC) {
		regalloc(&nod, types[tptr], N);
		cgen_as(&nod, n->left);
		nod.type = t;
		ginscall(&nod, proc);
		regfree(&nod);
		goto ret;
	}

	// call direct
	n->left->method = 1;
	ginscall(n->left, proc);


ret:
	;
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

	nod.xoffset = fp->width + 4; // +4: saved lr at 0(SP)
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

	nod1.xoffset = fp->width + 4; // +4: saved lr at 0(SP)
	nod1.type = fp->type;

	if(res->op != OREGISTER) {
		regalloc(&nod2, types[tptr], res);
		agen(&nod1, &nod2);
		gins(AMOVW, &nod2, res);
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
		p->to.name = D_EXTERN;
		p->to.type = D_CONST;
		p->to.sym = linksym(n->left->sym);
	}
}

/*
 * generate += *= etc.
 */
void
cgen_asop(Node *n)
{
	Node n1, n2, n3, n4;
	Node *nl, *nr;
	Prog *p1;
	Addr addr;
	int a, w;

	nl = n->left;
	nr = n->right;

	if(nr->ullman >= UINF && nl->ullman >= UINF) {
		tempname(&n1, nr->type);
		cgen(nr, &n1);
		n2 = *n;
		n2.right = &n1;
		cgen_asop(&n2);
		goto ret;
	}

	if(!isint[nl->type->etype])
		goto hard;
	if(!isint[nr->type->etype])
		goto hard;
	if(is64(nl->type) || is64(nr->type))
		goto hard64;

	switch(n->etype) {
	case OADD:
	case OSUB:
	case OXOR:
	case OAND:
	case OOR:
		a = optoas(n->etype, nl->type);
		if(nl->addable) {
			if(smallintconst(nr))
				n3 = *nr;
			else {
				regalloc(&n3, nr->type, N);
				cgen(nr, &n3);
			}
			regalloc(&n2, nl->type, N);
			cgen(nl, &n2);
			gins(a, &n3, &n2);
			cgen(&n2, nl);
			regfree(&n2);
			if(n3.op != OLITERAL)
				regfree(&n3);
			goto ret;
		}
		if(nr->ullman < UINF)
		if(sudoaddable(a, nl, &addr, &w)) {
			w = optoas(OAS, nl->type);
			regalloc(&n2, nl->type, N);
			p1 = gins(w, N, &n2);
			p1->from = addr;
			regalloc(&n3, nr->type, N);
			cgen(nr, &n3);
			gins(a, &n3, &n2);
			p1 = gins(w, &n2, N);
			p1->to = addr;
			regfree(&n2);
			regfree(&n3);
			sudoclean();
			goto ret;
		}
	}

hard:
	n2.op = 0;
	n1.op = 0;
	if(nr->op == OLITERAL) {
		// don't allocate a register for literals.
	} else if(nr->ullman >= nl->ullman || nl->addable) {
		regalloc(&n2, nr->type, N);
		cgen(nr, &n2);
		nr = &n2;
	} else {
		tempname(&n2, nr->type);
		cgen(nr, &n2);
		nr = &n2;
	}
	if(!nl->addable) {
		igen(nl, &n1, N);
		nl = &n1;
	}

	n3 = *n;
	n3.left = nl;
	n3.right = nr;
	n3.op = n->etype;

	regalloc(&n4, nl->type, N);
	cgen(&n3, &n4);
	gmove(&n4, nl);

	if(n1.op)
		regfree(&n1);
	if(n2.op == OREGISTER)
		regfree(&n2);
	regfree(&n4);
	goto ret;

hard64:
	if(nr->ullman > nl->ullman) {
		tempname(&n2, nr->type);
		cgen(nr, &n2);
		igen(nl, &n1, N);
	} else {
		igen(nl, &n1, N);
		tempname(&n2, nr->type);
		cgen(nr, &n2);
	}

	n3 = *n;
	n3.left = &n1;
	n3.right = &n2;
	n3.op = n->etype;

	cgen(&n3, &n1);

ret:
	;
}

int
samereg(Node *a, Node *b)
{
	if(a->op != OREGISTER)
		return 0;
	if(b->op != OREGISTER)
		return 0;
	if(a->val.u.reg != b->val.u.reg)
		return 0;
	return 1;
}

/*
 * generate high multiply
 *  res = (nl * nr) >> wordsize
 */
void
cgen_hmul(Node *nl, Node *nr, Node *res)
{
	int w;
	Node n1, n2, *tmp;
	Type *t;
	Prog *p;

	if(nl->ullman < nr->ullman) {
		tmp = nl;
		nl = nr;
		nr = tmp;
	}
	t = nl->type;
	w = t->width * 8;
	regalloc(&n1, t, res);
	cgen(nl, &n1);
	regalloc(&n2, t, N);
	cgen(nr, &n2);
	switch(simtype[t->etype]) {
	case TINT8:
	case TINT16:
		gins(optoas(OMUL, t), &n2, &n1);
		gshift(AMOVW, &n1, SHIFT_AR, w, &n1);
		break;
	case TUINT8:
	case TUINT16:
		gins(optoas(OMUL, t), &n2, &n1);
		gshift(AMOVW, &n1, SHIFT_LR, w, &n1);
		break;
	case TINT32:
	case TUINT32:
		// perform a long multiplication.
		if(issigned[t->etype])
			p = gins(AMULL, &n2, N);
		else
			p = gins(AMULLU, &n2, N);
		// n2 * n1 -> (n1 n2)
		p->reg = n1.val.u.reg;
		p->to.type = D_REGREG;
		p->to.reg = n1.val.u.reg;
		p->to.offset = n2.val.u.reg;
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
	Node n1, n2, n3, nt, t, lo, hi;
	int w, v;
	Prog *p1, *p2, *p3;
	Type *tr;
	uvlong sc;

	USED(bounded);
	if(nl->type->width > 4)
		fatal("cgen_shift %T", nl->type);

	w = nl->type->width * 8;

	if(op == OLROT) {
		v = mpgetfix(nr->val.u.xval);
		regalloc(&n1, nl->type, res);
		if(w == 32) {
			cgen(nl, &n1);
			gshift(AMOVW, &n1, SHIFT_RR, w-v, &n1);
		} else {
			regalloc(&n2, nl->type, N);
			cgen(nl, &n2);
			gshift(AMOVW, &n2, SHIFT_LL, v, &n1);
			gshift(AORR, &n2, SHIFT_LR, w-v, &n1);
			regfree(&n2);
			// Ensure sign/zero-extended result.
			gins(optoas(OAS, nl->type), &n1, &n1);
		}
		gmove(&n1, res);
		regfree(&n1);
		return;
	}

	if(nr->op == OLITERAL) {
		regalloc(&n1, nl->type, res);
		cgen(nl, &n1);
		sc = mpgetfix(nr->val.u.xval);
		if(sc == 0) {
			// nothing to do
		} else if(sc >= nl->type->width*8) {
			if(op == ORSH && issigned[nl->type->etype])
				gshift(AMOVW, &n1, SHIFT_AR, w, &n1);
			else
				gins(AEOR, &n1, &n1);
		} else {
			if(op == ORSH && issigned[nl->type->etype])
				gshift(AMOVW, &n1, SHIFT_AR, sc, &n1);
			else if(op == ORSH)
				gshift(AMOVW, &n1, SHIFT_LR, sc, &n1);
			else // OLSH
				gshift(AMOVW, &n1, SHIFT_LL, sc, &n1);
		}
		if(w < 32 && op == OLSH)
			gins(optoas(OAS, nl->type), &n1, &n1);
		gmove(&n1, res);
		regfree(&n1);
		return;
	}

	tr = nr->type;
	if(tr->width > 4) {
		tempname(&nt, nr->type);
		if(nl->ullman >= nr->ullman) {
			regalloc(&n2, nl->type, res);
			cgen(nl, &n2);
			cgen(nr, &nt);
			n1 = nt;
		} else {
			cgen(nr, &nt);
			regalloc(&n2, nl->type, res);
			cgen(nl, &n2);
		}
		split64(&nt, &lo, &hi);
		regalloc(&n1, types[TUINT32], N);
		regalloc(&n3, types[TUINT32], N);
		gmove(&lo, &n1);
		gmove(&hi, &n3);
		splitclean();
		gins(ATST, &n3, N);
		nodconst(&t, types[TUINT32], w);
		p1 = gins(AMOVW, &t, &n1);
		p1->scond = C_SCOND_NE;
		tr = types[TUINT32];
		regfree(&n3);
	} else {
		if(nl->ullman >= nr->ullman) {
			regalloc(&n2, nl->type, res);
			cgen(nl, &n2);
			regalloc(&n1, nr->type, N);
			cgen(nr, &n1);
		} else {
			regalloc(&n1, nr->type, N);
			cgen(nr, &n1);
			regalloc(&n2, nl->type, res);
			cgen(nl, &n2);
		}
	}

	// test for shift being 0
	gins(ATST, &n1, N);
	p3 = gbranch(ABEQ, T, -1);

	// test and fix up large shifts
	// TODO: if(!bounded), don't emit some of this.
	regalloc(&n3, tr, N);
	nodconst(&t, types[TUINT32], w);
	gmove(&t, &n3);
	gcmp(ACMP, &n1, &n3);
	if(op == ORSH) {
		if(issigned[nl->type->etype]) {
			p1 = gshift(AMOVW, &n2, SHIFT_AR, w-1, &n2);
			p2 = gregshift(AMOVW, &n2, SHIFT_AR, &n1, &n2);
		} else {
			p1 = gins(AEOR, &n2, &n2);
			p2 = gregshift(AMOVW, &n2, SHIFT_LR, &n1, &n2);
		}
		p1->scond = C_SCOND_HS;
		p2->scond = C_SCOND_LO;
	} else {
		p1 = gins(AEOR, &n2, &n2);
		p2 = gregshift(AMOVW, &n2, SHIFT_LL, &n1, &n2);
		p1->scond = C_SCOND_HS;
		p2->scond = C_SCOND_LO;
	}
	regfree(&n3);

	patch(p3, pc);
	// Left-shift of smaller word must be sign/zero-extended.
	if(w < 32 && op == OLSH)
		gins(optoas(OAS, nl->type), &n2, &n2);
	gmove(&n2, res);

	regfree(&n1);
	regfree(&n2);
}

void
clearfat(Node *nl)
{
	uint32 w, c, q;
	Node dst, nc, nz, end, r0, r1, *f;
	Prog *p, *pl;

	/* clear a fat object */
	if(debug['g'])
		dump("\nclearfat", nl);

	w = nl->type->width;
	// Avoid taking the address for simple enough types.
	if(componentgen(N, nl))
		return;

	c = w % 4;	// bytes
	q = w / 4;	// quads

	r0.op = OREGISTER;
	r0.val.u.reg = REGALLOC_R0;
	r1.op = OREGISTER;
	r1.val.u.reg = REGALLOC_R0 + 1;
	regalloc(&dst, types[tptr], &r1);
	agen(nl, &dst);
	nodconst(&nc, types[TUINT32], 0);
	regalloc(&nz, types[TUINT32], &r0);
	cgen(&nc, &nz);

	if(q > 128) {
		regalloc(&end, types[tptr], N);
		p = gins(AMOVW, &dst, &end);
		p->from.type = D_CONST;
		p->from.offset = q*4;

		p = gins(AMOVW, &nz, &dst);
		p->to.type = D_OREG;
		p->to.offset = 4;
		p->scond |= C_PBIT;
		pl = p;

		p = gins(ACMP, &dst, N);
		raddr(&end, p);
		patch(gbranch(ABNE, T, 0), pl);

		regfree(&end);
	} else if(q >= 4 && !nacl) {
		f = sysfunc("duffzero");
		p = gins(ADUFFZERO, N, f);
		afunclit(&p->to, f);
		// 4 and 128 = magic constants: see ../../runtime/asm_arm.s
		p->to.offset = 4*(128-q);
	} else
	while(q > 0) {
		p = gins(AMOVW, &nz, &dst);
		p->to.type = D_OREG;
		p->to.offset = 4;
 		p->scond |= C_PBIT;
//print("1. %P\n", p);
		q--;
	}

	while(c > 0) {
		p = gins(AMOVB, &nz, &dst);
		p->to.type = D_OREG;
		p->to.offset = 1;
 		p->scond |= C_PBIT;
//print("2. %P\n", p);
		c--;
	}
	regfree(&dst);
	regfree(&nz);
}

// Called after regopt and peep have run.
// Expand CHECKNIL pseudo-op into actual nil pointer check.
void
expandchecks(Prog *firstp)
{
	int reg;
	Prog *p, *p1;

	for(p = firstp; p != P; p = p->link) {
		if(p->as != ACHECKNIL)
			continue;
		if(debug_checknil && p->lineno > 1) // p->lineno==1 in generated wrappers
			warnl(p->lineno, "generated nil check");
		if(p->from.type != D_REG)
			fatal("invalid nil check %P", p);
		reg = p->from.reg;
		// check is
		//	CMP arg, $0
		//	MOV.EQ arg, 0(arg)
		p1 = mal(sizeof *p1);
		clearp(p1);
		p1->link = p->link;
		p->link = p1;
		p1->lineno = p->lineno;
		p1->pc = 9999;
		p1->as = AMOVW;
		p1->from.type = D_REG;
		p1->from.reg = reg;
		p1->to.type = D_OREG;
		p1->to.reg = reg;
		p1->to.offset = 0;
		p1->scond = C_SCOND_EQ;
		p->as = ACMP;
		p->from.type = D_CONST;
		p->from.reg = NREG;
		p->from.offset = 0;
		p->reg = reg;
	}
}
