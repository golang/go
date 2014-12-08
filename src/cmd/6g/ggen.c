// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#undef	EXTERN
#define	EXTERN
#include <u.h>
#include <libc.h>
#include "gg.h"
#include "opt.h"

static Prog *appendpp(Prog*, int, int, vlong, int, vlong);
static Prog *zerorange(Prog *p, vlong frame, vlong lo, vlong hi, uint32 *ax);

void
defframe(Prog *ptxt)
{
	uint32 frame, ax;
	Prog *p;
	vlong hi, lo;
	NodeList *l;
	Node *n;

	// fill in argument size
	ptxt->to.offset = rnd(curfn->type->argwid, widthptr);

	// fill in final stack size
	ptxt->to.offset <<= 32;
	frame = rnd(stksize+maxarg, widthreg);
	ptxt->to.offset |= frame;
	
	// insert code to zero ambiguously live variables
	// so that the garbage collector only sees initialized values
	// when it looks for pointers.
	p = ptxt;
	lo = hi = 0;
	ax = 0;
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
		p = zerorange(p, frame, lo, hi, &ax);

		// set new range
		hi = n->xoffset + n->type->width;
		lo = n->xoffset;
	}
	// zero final range
	zerorange(p, frame, lo, hi, &ax);
}

static Prog*
zerorange(Prog *p, vlong frame, vlong lo, vlong hi, uint32 *ax)
{
	vlong cnt, i;

	cnt = hi - lo;
	if(cnt == 0)
		return p;
	if(*ax == 0) {
		p = appendpp(p, AMOVQ, D_CONST, 0, D_AX, 0);
		*ax = 1;
	}
	if(cnt % widthreg != 0) {
		// should only happen with nacl
		if(cnt % widthptr != 0)
			fatal("zerorange count not a multiple of widthptr %d", cnt);
		p = appendpp(p, AMOVL, D_AX, 0, D_SP+D_INDIR, frame+lo);
		lo += widthptr;
		cnt -= widthptr;
	}
	if(cnt <= 4*widthreg) {
		for(i = 0; i < cnt; i += widthreg) {
			p = appendpp(p, AMOVQ, D_AX, 0, D_SP+D_INDIR, frame+lo+i);
		}
	} else if(!nacl && (cnt <= 128*widthreg)) {
		p = appendpp(p, leaptr, D_SP+D_INDIR, frame+lo, D_DI, 0);
		p = appendpp(p, ADUFFZERO, D_NONE, 0, D_ADDR, 2*(128-cnt/widthreg));
		p->to.sym = linksym(pkglookup("duffzero", runtimepkg));
	} else {
		p = appendpp(p, AMOVQ, D_CONST, cnt/widthreg, D_CX, 0);
		p = appendpp(p, leaptr, D_SP+D_INDIR, frame+lo, D_DI, 0);
		p = appendpp(p, AREP, D_NONE, 0, D_NONE, 0);
		p = appendpp(p, ASTOSQ, D_NONE, 0, D_NONE, 0);
	}
	return p;
}

static Prog*	
appendpp(Prog *p, int as, int ftype, vlong foffset, int ttype, vlong toffset)	
{
	Prog *q;
	q = mal(sizeof(*q));	
	clearp(q);	
	q->as = as;	
	q->lineno = p->lineno;	
	q->from.type = ftype;	
	q->from.offset = foffset;	
	q->to.type = ttype;	
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
fixautoused(Prog *p)
{
	Prog **lp;

	for (lp=&p; (p=*lp) != P; ) {
		if (p->as == ATYPE && p->from.node && p->from.type == D_AUTO && !p->from.node->used) {
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
		if (p->from.type == D_AUTO && p->from.node)
			p->from.offset += p->from.node->stkdelta;

		if (p->to.type == D_AUTO && p->to.node)
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
	Node reg, stk;
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
				// insert an x86 NOP that we will have the right line number.
				// x86 NOP 0x90 is really XCHG AX, AX; use that description
				// because the NOP pseudo-instruction would be removed by
				// the linker.
				nodreg(&reg, types[TINT], D_AX);
				gins(AXCHGL, &reg, &reg);
			}
			p = gins(ACALL, N, f);
			afunclit(&p->to, f);
			if(proc == -1 || noreturn(p))
				gins(AUNDEF, N, N);
			break;
		}
		nodreg(&reg, types[tptr], D_DX);
		nodreg(&r1, types[tptr], D_BX);
		gmove(f, &reg);
		reg.op = OINDREG;
		gmove(&reg, &r1);
		reg.op = OREGISTER;
		gins(ACALL, &reg, &r1);
		break;
	
	case 3:	// normal call of c function pointer
		gins(ACALL, N, f);
		break;

	case 1:	// call in new proc (go)
	case 2:	// deferred call (defer)
		memset(&stk, 0, sizeof(stk));
		stk.op = OINDREG;
		stk.val.u.reg = D_SP;
		stk.xoffset = 0;

		if(widthptr == 8) {
			// size of arguments at 0(SP)
			ginscon(AMOVQ, argsize(f->type), &stk);

			// FuncVal* at 8(SP)
			stk.xoffset = widthptr;
			nodreg(&reg, types[TINT64], D_AX);
			gmove(f, &reg);
			gins(AMOVQ, &reg, &stk);
		} else {
			// size of arguments at 0(SP)
			ginscon(AMOVL, argsize(f->type), &stk);

			// FuncVal* at 4(SP)
			stk.xoffset = widthptr;
			nodreg(&reg, types[TINT32], D_AX);
			gmove(f, &reg);
			gins(AMOVL, &reg, &stk);
		}

		if(proc == 1)
			ginscall(newproc, 0);
		else {
			if(!hasdefer)
				fatal("hasdefer=0 but has defer");
			ginscall(deferproc, 0);
		}
		if(proc == 2) {
			nodreg(&reg, types[TINT32], D_AX);
			gins(ATESTL, &reg, &reg);
			p = gbranch(AJEQ, T, +1);
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

	nodindreg(&nodsp, types[tptr], D_SP);
        nodsp.xoffset = 0;
	if(proc != 0)
		nodsp.xoffset += 2 * widthptr; // leave room for size & fn
	nodi.type = types[tptr];
	nodi.xoffset += widthptr;
	cgen(&nodi, &nodsp);	// {0, 8(nacl), or 16}(SP) = 8(REG) -- i.data

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
		gins(ALEAQ, &nodo, &nodr);	// REG = &(32+offset(REG)) -- i.tab->fun[f]
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
	nod.val.u.reg = D_SP;
	nod.addable = 1;

	nod.xoffset = fp->width;
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
	nod1.val.u.reg = D_SP;
	nod1.addable = 1;

	nod1.xoffset = fp->width;
	nod1.type = fp->type;

	if(res->op != OREGISTER) {
		regalloc(&nod2, types[tptr], res);
		gins(leaptr, &nod1, &nod2);
		gins(movptr, &nod2, res);
		regfree(&nod2);
	} else
		gins(leaptr, &nod1, res);
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
		p->to.type = D_EXTERN;
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
	int a;

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

	switch(n->etype) {
	case OADD:
		if(smallintconst(nr))
		if(mpgetfix(nr->val.u.xval) == 1) {
			a = optoas(OINC, nl->type);
			if(nl->addable) {
				gins(a, N, nl);
				goto ret;
			}
			if(sudoaddable(a, nl, &addr)) {
				p1 = gins(a, N, N);
				p1->to = addr;
				sudoclean();
				goto ret;
			}
		}
		break;

	case OSUB:
		if(smallintconst(nr))
		if(mpgetfix(nr->val.u.xval) == 1) {
			a = optoas(ODEC, nl->type);
			if(nl->addable) {
				gins(a, N, nl);
				goto ret;
			}
			if(sudoaddable(a, nl, &addr)) {
				p1 = gins(a, N, N);
				p1->to = addr;
				sudoclean();
				goto ret;
			}
		}
		break;
	}

	switch(n->etype) {
	case OADD:
	case OSUB:
	case OXOR:
	case OAND:
	case OOR:
		a = optoas(n->etype, nl->type);
		if(nl->addable) {
			if(smallintconst(nr)) {
				gins(a, nr, nl);
				goto ret;
			}
			regalloc(&n2, nr->type, N);
			cgen(nr, &n2);
			gins(a, &n2, nl);
			regfree(&n2);
			goto ret;
		}
		if(nr->ullman < UINF)
		if(sudoaddable(a, nl, &addr)) {
			if(smallintconst(nr)) {
				p1 = gins(a, nr, N);
				p1->to = addr;
				sudoclean();
				goto ret;
			}
			regalloc(&n2, nr->type, N);
			cgen(nr, &n2);
			p1 = gins(a, &n2, N);
			p1->to = addr;
			regfree(&n2);
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

ret:
	;
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
	Node n3, n4;
	Type *t, *t0;
	Node ax, dx, ax1, n31, oldax, olddx;
	Prog *p1, *p2;

	// Have to be careful about handling
	// most negative int divided by -1 correctly.
	// The hardware will trap.
	// Also the byte divide instruction needs AH,
	// which we otherwise don't have to deal with.
	// Easiest way to avoid for int8, int16: use int32.
	// For int32 and int64, use explicit test.
	// Could use int64 hw for int32.
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
	if(t->width < 4) {
		if(issigned[t->etype])
			t = types[TINT32];
		else
			t = types[TUINT32];
		check = 0;
	}
	a = optoas(op, t);

	regalloc(&n3, t0, N);
	if(nl->ullman >= nr->ullman) {
		savex(D_AX, &ax, &oldax, res, t0);
		cgen(nl, &ax);
		regalloc(&ax, t0, &ax);	// mark ax live during cgen
		cgen(nr, &n3);
		regfree(&ax);
	} else {
		cgen(nr, &n3);
		savex(D_AX, &ax, &oldax, res, t0);
		cgen(nl, &ax);
	}
	if(t != t0) {
		// Convert
		ax1 = ax;
		n31 = n3;
		ax.type = t;
		n3.type = t;
		gmove(&ax1, &ax);
		gmove(&n31, &n3);
	}

	p2 = P;
	if(nacl) {
		// Native Client does not relay the divide-by-zero trap
		// to the executing program, so we must insert a check
		// for ourselves.
		nodconst(&n4, t, 0);
		gins(optoas(OCMP, t), &n3, &n4);
		p1 = gbranch(optoas(ONE, t), T, +1);
		if(panicdiv == N)
			panicdiv = sysfunc("panicdivide");
		ginscall(panicdiv, -1);
		patch(p1, pc);
	}
	if(check) {
		nodconst(&n4, t, -1);
		gins(optoas(OCMP, t), &n3, &n4);
		p1 = gbranch(optoas(ONE, t), T, +1);
		if(op == ODIV) {
			// a / (-1) is -a.
			gins(optoas(OMINUS, t), N, &ax);
			gmove(&ax, res);
		} else {
			// a % (-1) is 0.
			nodconst(&n4, t, 0);
			gmove(&n4, res);
		}
		p2 = gbranch(AJMP, T, 0);
		patch(p1, pc);
	}
	savex(D_DX, &dx, &olddx, res, t);
	if(!issigned[t->etype]) {
		nodconst(&n4, t, 0);
		gmove(&n4, &dx);
	} else
		gins(optoas(OEXTEND, t), N, N);
	gins(a, &n3, N);
	regfree(&n3);
	if(op == ODIV)
		gmove(&ax, res);
	else
		gmove(&dx, res);
	restx(&dx, &olddx);
	if(check)
		patch(p2, pc);
	restx(&ax, &oldax);
}

/*
 * register dr is one of the special ones (AX, CX, DI, SI, etc.).
 * we need to use it.  if it is already allocated as a temporary
 * (r > 1; can only happen if a routine like sgen passed a
 * special as cgen's res and then cgen used regalloc to reuse
 * it as its own temporary), then move it for now to another
 * register.  caller must call restx to move it back.
 * the move is not necessary if dr == res, because res is
 * known to be dead.
 */
void
savex(int dr, Node *x, Node *oldx, Node *res, Type *t)
{
	int r;

	r = reg[dr];

	// save current ax and dx if they are live
	// and not the destination
	memset(oldx, 0, sizeof *oldx);
	nodreg(x, t, dr);
	if(r > 1 && !samereg(x, res)) {
		regalloc(oldx, types[TINT64], N);
		x->type = types[TINT64];
		gmove(x, oldx);
		x->type = t;
		oldx->ostk = r;	// squirrel away old r value
		reg[dr] = 1;
	}
}

void
restx(Node *x, Node *oldx)
{
	if(oldx->op != 0) {
		x->type = types[TINT64];
		reg[x->val.u.reg] = oldx->ostk;
		gmove(oldx, x);
		regfree(oldx);
	}
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

	if(nr->op != OLITERAL)
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
		a = AIMULW;
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
	Type *t;
	int a;
	Node n1, n2, ax, dx, *tmp;

	t = nl->type;
	a = optoas(OHMUL, t);
	if(nl->ullman < nr->ullman) {
		tmp = nl;
		nl = nr;
		nr = tmp;
	}
	cgenr(nl, &n1, res);
	cgenr(nr, &n2, N);
	nodreg(&ax, t, D_AX);
	gmove(&n1, &ax);
	gins(a, &n2, N);
	regfree(&n2);
	regfree(&n1);

	if(t->width == 1) {
		// byte multiply behaves differently.
		nodreg(&ax, t, D_AH);
		nodreg(&dx, t, D_DX);
		gmove(&ax, &dx);
	}
	nodreg(&dx, t, D_DX);
	gmove(&dx, res);
}

/*
 * generate shift according to op, one of:
 *	res = nl << nr
 *	res = nl >> nr
 */
void
cgen_shift(int op, int bounded, Node *nl, Node *nr, Node *res)
{
	Node n1, n2, n3, n4, n5, cx, oldcx;
	int a, rcx;
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

	rcx = reg[D_CX];
	nodreg(&n1, types[TUINT32], D_CX);
	
	// Allow either uint32 or uint64 as shift type,
	// to avoid unnecessary conversion from uint32 to uint64
	// just to do the comparison.
	tcount = types[simtype[nr->type->etype]];
	if(tcount->etype < TUINT32)
		tcount = types[TUINT32];

	regalloc(&n1, nr->type, &n1);		// to hold the shift type in CX
	regalloc(&n3, tcount, &n1);	// to clear high bits of CX

	nodreg(&cx, types[TUINT64], D_CX);
	memset(&oldcx, 0, sizeof oldcx);
	if(rcx > 0 && !samereg(&cx, res)) {
		regalloc(&oldcx, types[TUINT64], N);
		gmove(&cx, &oldcx);
	}
	cx.type = tcount;

	if(samereg(&cx, res))
		regalloc(&n2, nl->type, N);
	else
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

	if(oldcx.op != 0) {
		cx.type = types[TUINT64];
		gmove(&oldcx, &cx);
		regfree(&oldcx);
	}

	gmove(&n2, res);

	regfree(&n1);
	regfree(&n2);

ret:
	;
}

/*
 * generate byte multiply:
 *	res = nl * nr
 * there is no 2-operand byte multiply instruction so
 * we do a full-width multiplication and truncate afterwards.
 */
void
cgen_bmul(int op, Node *nl, Node *nr, Node *res)
{
	Node n1, n2, n1b, n2b, *tmp;
	Type *t;
	int a;

	// largest ullman on left.
	if(nl->ullman < nr->ullman) {
		tmp = nl;
		nl = nr;
		nr = tmp;
	}

	// generate operands in "8-bit" registers.
	regalloc(&n1b, nl->type, res);
	cgen(nl, &n1b);
	regalloc(&n2b, nr->type, N);
	cgen(nr, &n2b);

	// perform full-width multiplication.
	t = types[TUINT64];
	if(issigned[nl->type->etype])
		t = types[TINT64];
	nodreg(&n1, t, n1b.val.u.reg);
	nodreg(&n2, t, n2b.val.u.reg);
	a = optoas(op, t);
	gins(a, &n2, &n1);

	// truncate.
	gmove(&n1, res);
	regfree(&n1b);
	regfree(&n2b);
}

void
clearfat(Node *nl)
{
	int64 w, c, q;
	Node n1, oldn1, ax, oldax, di, z;
	Prog *p;

	/* clear a fat object */
	if(debug['g'])
		dump("\nclearfat", nl);

	w = nl->type->width;
	// Avoid taking the address for simple enough types.
	if(componentgen(N, nl))
		return;

	c = w % 8;	// bytes
	q = w / 8;	// quads

	if(q < 4) {
		// Write sequence of MOV 0, off(base) instead of using STOSQ.
		// The hope is that although the code will be slightly longer,
		// the MOVs will have no dependencies and pipeline better
		// than the unrolled STOSQ loop.
		// NOTE: Must use agen, not igen, so that optimizer sees address
		// being taken. We are not writing on field boundaries.
		agenr(nl, &n1, N);
		n1.op = OINDREG;
		nodconst(&z, types[TUINT64], 0);
		while(q-- > 0) {
			n1.type = z.type;
			gins(AMOVQ, &z, &n1);
			n1.xoffset += 8;
		}
		if(c >= 4) {
			nodconst(&z, types[TUINT32], 0);
			n1.type = z.type;
			gins(AMOVL, &z, &n1);
			n1.xoffset += 4;
			c -= 4;
		}
		nodconst(&z, types[TUINT8], 0);
		while(c-- > 0) {
			n1.type = z.type;
			gins(AMOVB, &z, &n1);
			n1.xoffset++;
		}
		regfree(&n1);
		return;
	}

	savex(D_DI, &n1, &oldn1, N, types[tptr]);
	agen(nl, &n1);

	savex(D_AX, &ax, &oldax, N, types[tptr]);
	gconreg(AMOVL, 0, D_AX);

	if(q > 128 || nacl) {
		gconreg(movptr, q, D_CX);
		gins(AREP, N, N);	// repeat
		gins(ASTOSQ, N, N);	// STOQ AL,*(DI)+
	} else {
		p = gins(ADUFFZERO, N, N);
		p->to.type = D_ADDR;
		p->to.sym = linksym(pkglookup("duffzero", runtimepkg));
		// 2 and 128 = magic constants: see ../../runtime/asm_amd64.s
		p->to.offset = 2*(128-q);
	}

	z = ax;
	di = n1;
	if(w >= 8 && c >= 4) {
		di.op = OINDREG;
		di.type = z.type = types[TINT64];
		p = gins(AMOVQ, &z, &di);
		p->to.scale = 1;
		p->to.offset = c-8;
	} else if(c >= 4) {
		di.op = OINDREG;
		di.type = z.type = types[TINT32];
		p = gins(AMOVL, &z, &di);
		if(c > 4) {
			p = gins(AMOVL, &z, &di);
			p->to.scale = 1;
			p->to.offset = c-4;
		}
	} else
	while(c > 0) {
		gins(ASTOSB, N, N);	// STOB AL,*(DI)+
		c--;
	}

	restx(&n1, &oldn1);
	restx(&ax, &oldax);
}

// Called after regopt and peep have run.
// Expand CHECKNIL pseudo-op into actual nil pointer check.
void
expandchecks(Prog *firstp)
{
	Prog *p, *p1, *p2;

	for(p = firstp; p != P; p = p->link) {
		if(p->as != ACHECKNIL)
			continue;
		if(debug_checknil && p->lineno > 1) // p->lineno==1 in generated wrappers
			warnl(p->lineno, "generated nil check");
		// check is
		//	CMP arg, $0
		//	JNE 2(PC) (likely)
		//	MOV AX, 0
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
		p->as = cmpptr;
		p->to.type = D_CONST;
		p->to.offset = 0;
		p1->as = AJNE;
		p1->from.type = D_CONST;
		p1->from.offset = 1; // likely
		p1->to.type = D_BRANCH;
		p1->to.u.branch = p2->link;
		// crash by write to memory address 0.
		// if possible, since we know arg is 0, use 0(arg),
		// which will be shorter to encode than plain 0.
		p2->as = AMOVL;
		p2->from.type = D_AX;
		if(regtyp(&p->from))
			p2->to.type = p->from.type + D_INDIR;
		else
			p2->to.type = D_INDIR+D_NONE;
		p2->to.offset = 0;
	}
}
