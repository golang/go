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
	vlong lo, hi;
	NodeList *l;
	Node *n;

	// fill in argument size
	ptxt->to.offset2 = rnd(curfn->type->argwid, widthptr);

	// fill in final stack size
	frame = rnd(stksize+maxarg, widthptr);
	ptxt->to.offset = frame;
	
	// insert code to zero ambiguously live variables
	// so that the garbage collector only sees initialized values
	// when it looks for pointers.
	p = ptxt;
	hi = 0;
	lo = hi;
	ax = 0;
	for(l=curfn->dcl; l != nil; l = l->next) {
		n = l->n;
		if(!n->needzero)
			continue;
		if(n->class != PAUTO)
			fatal("needzero class %d", n->class);
		if(n->type->width % widthptr != 0 || n->xoffset % widthptr != 0 || n->type->width == 0)
			fatal("var %lN has size %d offset %d", n, (int)n->type->width, (int)n->xoffset);
		if(lo != hi && n->xoffset + n->type->width == lo - 2*widthptr) {
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
		p = appendpp(p, AMOVL, D_CONST, 0, D_AX, 0);
		*ax = 1;
	}
	if(cnt <= 4*widthreg) {
		for(i = 0; i < cnt; i += widthreg) {
			p = appendpp(p, AMOVL, D_AX, 0, D_SP+D_INDIR, frame+lo+i);
		}
	} else if(!nacl && cnt <= 128*widthreg) {
		p = appendpp(p, ALEAL, D_SP+D_INDIR, frame+lo, D_DI, 0);
		p = appendpp(p, ADUFFZERO, D_NONE, 0, D_ADDR, 1*(128-cnt/widthreg));
		p->to.sym = linksym(pkglookup("duffzero", runtimepkg));
	} else {
		p = appendpp(p, AMOVL, D_CONST, cnt/widthreg, D_CX, 0);
		p = appendpp(p, ALEAL, D_SP+D_INDIR, frame+lo, D_DI, 0);
		p = appendpp(p, AREP, D_NONE, 0, D_NONE, 0);
		p = appendpp(p, ASTOSL, D_NONE, 0, D_NONE, 0);
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
fixautoused(Prog* p)
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

void
clearfat(Node *nl)
{
	uint32 w, c, q;
	Node n1, z;
	Prog *p;

	/* clear a fat object */
	if(debug['g'])
		dump("\nclearfat", nl);

	w = nl->type->width;
	// Avoid taking the address for simple enough types.
	if(componentgen(N, nl))
		return;

	c = w % 4;	// bytes
	q = w / 4;	// quads

	if(q < 4) {
		// Write sequence of MOV 0, off(base) instead of using STOSL.
		// The hope is that although the code will be slightly longer,
		// the MOVs will have no dependencies and pipeline better
		// than the unrolled STOSL loop.
		// NOTE: Must use agen, not igen, so that optimizer sees address
		// being taken. We are not writing on field boundaries.
		regalloc(&n1, types[tptr], N);
		agen(nl, &n1);
		n1.op = OINDREG;
		nodconst(&z, types[TUINT64], 0);
		while(q-- > 0) {
			n1.type = z.type;
			gins(AMOVL, &z, &n1);
			n1.xoffset += 4;
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

	nodreg(&n1, types[tptr], D_DI);
	agen(nl, &n1);
	gconreg(AMOVL, 0, D_AX);

	if(q > 128 || (q >= 4 && nacl)) {
		gconreg(AMOVL, q, D_CX);
		gins(AREP, N, N);	// repeat
		gins(ASTOSL, N, N);	// STOL AL,*(DI)+
	} else if(q >= 4) {
		p = gins(ADUFFZERO, N, N);
		p->to.type = D_ADDR;
		p->to.sym = linksym(pkglookup("duffzero", runtimepkg));
		// 1 and 128 = magic constants: see ../../runtime/asm_386.s
		p->to.offset = 1*(128-q);
	} else
	while(q > 0) {
		gins(ASTOSL, N, N);	// STOL AL,*(DI)+
		q--;
	}

	while(c > 0) {
		gins(ASTOSB, N, N);	// STOB AL,*(DI)+
		c--;
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
	Node reg, r1, con;

	if(f->type != T)
		setmaxarg(f->type);

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
				// because the NOP pseudo-instruction will be removed by
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
		nodreg(&reg, types[TINT32], D_CX);
		gins(APUSHL, f, N);
		nodconst(&con, types[TINT32], argsize(f->type));
		gins(APUSHL, &con, N);
		if(proc == 1)
			ginscall(newproc, 0);
		else
			ginscall(deferproc, 0);
		gins(APOPL, N, &reg);
		gins(APOPL, N, &reg);
		if(proc == 2) {
			nodreg(&reg, types[TINT64], D_AX);
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
	nodi.type = types[tptr];
	nodi.xoffset += widthptr;
	cgen(&nodi, &nodsp);	// 0(SP) = 4(REG) -- i.data

	regalloc(&nodo, types[tptr], res);
	nodi.type = types[tptr];
	nodi.xoffset -= widthptr;
	cgen(&nodi, &nodo);	// REG = 0(REG) -- i.tab
	regfree(&nodi);

	regalloc(&nodr, types[tptr], &nodo);
	if(n->left->xoffset == BADWIDTH)
		fatal("cgen_callinter: badwidth");
	cgen_checknil(&nodo);
	nodo.op = OINDREG;
	nodo.xoffset = n->left->xoffset + 3*widthptr + 8;
	
	if(proc == 0) {
		// plain call: use direct c function pointer - more efficient
		cgen(&nodo, &nodr);	// REG = 20+offset(REG) -- i.tab->fun[f]
		proc = 3;
	} else {
		// go/defer. generate go func value.
		gins(ALEAL, &nodo, &nodr);	// REG = &(20+offset(REG)) -- i.tab->fun[f]
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
		gins(ALEAL, &nod1, &nod2);
		gins(AMOVL, &nod2, res);
		regfree(&nod2);
	} else
		gins(ALEAL, &nod1, res);
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
	if(is64(nl->type) || is64(nr->type))
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
	if(nr->ullman >= nl->ullman || nl->addable) {
		mgen(nr, &n2, N);
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

	mgen(&n3, &n4, N);
	gmove(&n4, nl);

	if(n1.op)
		regfree(&n1);
	mfree(&n2);
	mfree(&n4);

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
 * generate division.
 * caller must set:
 *	ax = allocated AX register
 *	dx = allocated DX register
 * generates one of:
 *	res = nl / nr
 *	res = nl % nr
 * according to op.
 */
void
dodiv(int op, Node *nl, Node *nr, Node *res, Node *ax, Node *dx)
{
	int check;
	Node n1, t1, t2, t3, t4, n4, nz;
	Type *t, *t0;
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
		if(isconst(nl, CTINT) && mpgetfix(nl->val.u.xval) != -1LL<<(t->width*8-1))
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

	tempname(&t1, t);
	tempname(&t2, t);
	if(t0 != t) {
		tempname(&t3, t0);
		tempname(&t4, t0);
		cgen(nl, &t3);
		cgen(nr, &t4);
		// Convert.
		gmove(&t3, &t1);
		gmove(&t4, &t2);
	} else {
		cgen(nl, &t1);
		cgen(nr, &t2);
	}

	if(!samereg(ax, res) && !samereg(dx, res))
		regalloc(&n1, t, res);
	else
		regalloc(&n1, t, N);
	gmove(&t2, &n1);
	gmove(&t1, ax);
	p2 = P;
	if(nacl) {
		// Native Client does not relay the divide-by-zero trap
		// to the executing program, so we must insert a check
		// for ourselves.
		nodconst(&n4, t, 0);
		gins(optoas(OCMP, t), &n1, &n4);
		p1 = gbranch(optoas(ONE, t), T, +1);
		if(panicdiv == N)
			panicdiv = sysfunc("panicdivide");
		ginscall(panicdiv, -1);
		patch(p1, pc);
	}
	if(check) {
		nodconst(&n4, t, -1);
		gins(optoas(OCMP, t), &n1, &n4);
		p1 = gbranch(optoas(ONE, t), T, +1);
		if(op == ODIV) {
			// a / (-1) is -a.
			gins(optoas(OMINUS, t), N, ax);
			gmove(ax, res);
		} else {
			// a % (-1) is 0.
			nodconst(&n4, t, 0);
			gmove(&n4, res);
		}
		p2 = gbranch(AJMP, T, 0);
		patch(p1, pc);
	}
	if(!issigned[t->etype]) {
		nodconst(&nz, t, 0);
		gmove(&nz, dx);
	} else
		gins(optoas(OEXTEND, t), N, N);
	gins(optoas(op, t), &n1, N);
	regfree(&n1);

	if(op == ODIV)
		gmove(ax, res);
	else
		gmove(dx, res);
	if(check)
		patch(p2, pc);
}

static void
savex(int dr, Node *x, Node *oldx, Node *res, Type *t)
{
	int r;

	r = reg[dr];
	nodreg(x, types[TINT32], dr);

	// save current ax and dx if they are live
	// and not the destination
	memset(oldx, 0, sizeof *oldx);
	if(r > 0 && !samereg(x, res)) {
		tempname(oldx, types[TINT32]);
		gmove(x, oldx);
	}

	regalloc(x, t, x);
}

static void
restx(Node *x, Node *oldx)
{
	regfree(x);

	if(oldx->op != 0) {
		x->type = types[TINT32];
		gmove(oldx, x);
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
	Node ax, dx, oldax, olddx;
	Type *t;

	if(is64(nl->type))
		fatal("cgen_div %T", nl->type);

	if(issigned[nl->type->etype])
		t = types[TINT32];
	else
		t = types[TUINT32];
	savex(D_AX, &ax, &oldax, res, t);
	savex(D_DX, &dx, &olddx, res, t);
	dodiv(op, nl, nr, res, &ax, &dx);
	restx(&dx, &olddx);
	restx(&ax, &oldax);
}

/*
 * generate shift according to op, one of:
 *	res = nl << nr
 *	res = nl >> nr
 */
void
cgen_shift(int op, int bounded, Node *nl, Node *nr, Node *res)
{
	Node n1, n2, nt, cx, oldcx, hi, lo;
	int a, w;
	Prog *p1, *p2;
	uvlong sc;

	if(nl->type->width > 4)
		fatal("cgen_shift %T", nl->type);

	w = nl->type->width * 8;

	a = optoas(op, nl->type);

	if(nr->op == OLITERAL) {
		tempname(&n2, nl->type);
		cgen(nl, &n2);
		regalloc(&n1, nl->type, res);
		gmove(&n2, &n1);
		sc = mpgetfix(nr->val.u.xval);
		if(sc >= nl->type->width*8) {
			// large shift gets 2 shifts by width-1
			gins(a, ncon(w-1), &n1);
			gins(a, ncon(w-1), &n1);
		} else
			gins(a, nr, &n1);
		gmove(&n1, res);
		regfree(&n1);
		return;
	}

	memset(&oldcx, 0, sizeof oldcx);
	nodreg(&cx, types[TUINT32], D_CX);
	if(reg[D_CX] > 1 && !samereg(&cx, res)) {
		tempname(&oldcx, types[TUINT32]);
		gmove(&cx, &oldcx);
	}

	if(nr->type->width > 4) {
		tempname(&nt, nr->type);
		n1 = nt;
	} else {
		nodreg(&n1, types[TUINT32], D_CX);
		regalloc(&n1, nr->type, &n1);		// to hold the shift type in CX
	}

	if(samereg(&cx, res))
		regalloc(&n2, nl->type, N);
	else
		regalloc(&n2, nl->type, res);
	if(nl->ullman >= nr->ullman) {
		cgen(nl, &n2);
		cgen(nr, &n1);
	} else {
		cgen(nr, &n1);
		cgen(nl, &n2);
	}

	// test and fix up large shifts
	if(bounded) {
		if(nr->type->width > 4) {
			// delayed reg alloc
			nodreg(&n1, types[TUINT32], D_CX);
			regalloc(&n1, types[TUINT32], &n1);		// to hold the shift type in CX
			split64(&nt, &lo, &hi);
			gmove(&lo, &n1);
			splitclean();
		}
	} else {
		if(nr->type->width > 4) {
			// delayed reg alloc
			nodreg(&n1, types[TUINT32], D_CX);
			regalloc(&n1, types[TUINT32], &n1);		// to hold the shift type in CX
			split64(&nt, &lo, &hi);
			gmove(&lo, &n1);
			gins(optoas(OCMP, types[TUINT32]), &hi, ncon(0));
			p2 = gbranch(optoas(ONE, types[TUINT32]), T, +1);
			gins(optoas(OCMP, types[TUINT32]), &n1, ncon(w));
			p1 = gbranch(optoas(OLT, types[TUINT32]), T, +1);
			splitclean();
			patch(p2, pc);
		} else {
			gins(optoas(OCMP, nr->type), &n1, ncon(w));
			p1 = gbranch(optoas(OLT, types[TUINT32]), T, +1);
		}
		if(op == ORSH && issigned[nl->type->etype]) {
			gins(a, ncon(w-1), &n2);
		} else {
			gmove(ncon(0), &n2);
		}
		patch(p1, pc);
	}
	gins(a, &n1, &n2);

	if(oldcx.op != 0)
		gmove(&oldcx, &cx);

	gmove(&n2, res);

	regfree(&n1);
	regfree(&n2);
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
	Node n1, n2, nt, *tmp;
	Type *t;
	int a;

	// copy from byte to full registers
	t = types[TUINT32];
	if(issigned[nl->type->etype])
		t = types[TINT32];

	// largest ullman on left.
	if(nl->ullman < nr->ullman) {
		tmp = nl;
		nl = nr;
		nr = tmp;
	}

	tempname(&nt, nl->type);
	cgen(nl, &nt);
	regalloc(&n1, t, res);
	cgen(nr, &n1);
	regalloc(&n2, t, N);
	gmove(&nt, &n2);
	a = optoas(op, t);
	gins(a, &n2, &n1);
	regfree(&n2);
	gmove(&n1, res);
	regfree(&n1);
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
	Node n1, n2, ax, dx;

	t = nl->type;
	a = optoas(OHMUL, t);
	// gen nl in n1.
	tempname(&n1, t);
	cgen(nl, &n1);
	// gen nr in n2.
	regalloc(&n2, t, res);
	cgen(nr, &n2);

	// multiply.
	nodreg(&ax, t, D_AX);
	gmove(&n2, &ax);
	gins(a, &n1, N);
	regfree(&n2);

	if(t->width == 1) {
		// byte multiply behaves differently.
		nodreg(&ax, t, D_AH);
		nodreg(&dx, t, D_DX);
		gmove(&ax, &dx);
	}
	nodreg(&dx, t, D_DX);
	gmove(&dx, res);
}

static void cgen_float387(Node *n, Node *res);
static void cgen_floatsse(Node *n, Node *res);

/*
 * generate floating-point operation.
 */
void
cgen_float(Node *n, Node *res)
{
	Node *nl;
	Node n1, n2;
	Prog *p1, *p2, *p3;

	nl = n->left;
	switch(n->op) {
	case OEQ:
	case ONE:
	case OLT:
	case OLE:
	case OGE:
		p1 = gbranch(AJMP, T, 0);
		p2 = pc;
		gmove(nodbool(1), res);
		p3 = gbranch(AJMP, T, 0);
		patch(p1, pc);
		bgen(n, 1, 0, p2);
		gmove(nodbool(0), res);
		patch(p3, pc);
		return;

	case OPLUS:
		cgen(nl, res);
		return;

	case OCONV:
		if(eqtype(n->type, nl->type) || noconv(n->type, nl->type)) {
			cgen(nl, res);
			return;
		}

		tempname(&n2, n->type);
		mgen(nl, &n1, res);
		gmove(&n1, &n2);
		gmove(&n2, res);
		mfree(&n1);
		return;
	}

	if(use_sse)
		cgen_floatsse(n, res);
	else
		cgen_float387(n, res);
}

// floating-point.  387 (not SSE2)
static void
cgen_float387(Node *n, Node *res)
{
	Node f0, f1;
	Node *nl, *nr;

	nl = n->left;
	nr = n->right;
	nodreg(&f0, nl->type, D_F0);
	nodreg(&f1, n->type, D_F0+1);
	if(nr != N)
		goto flt2;

	// unary
	cgen(nl, &f0);
	if(n->op != OCONV && n->op != OPLUS)
		gins(foptoas(n->op, n->type, 0), N, N);
	gmove(&f0, res);
	return;

flt2:	// binary
	if(nl->ullman >= nr->ullman) {
		cgen(nl, &f0);
		if(nr->addable)
			gins(foptoas(n->op, n->type, 0), nr, &f0);
		else {
			cgen(nr, &f0);
			gins(foptoas(n->op, n->type, Fpop), &f0, &f1);
		}
	} else {
		cgen(nr, &f0);
		if(nl->addable)
			gins(foptoas(n->op, n->type, Frev), nl, &f0);
		else {
			cgen(nl, &f0);
			gins(foptoas(n->op, n->type, Frev|Fpop), &f0, &f1);
		}
	}
	gmove(&f0, res);
	return;

}

static void
cgen_floatsse(Node *n, Node *res)
{
	Node *nl, *nr, *r;
	Node n1, n2, nt;
	int a;

	nl = n->left;
	nr = n->right;
	switch(n->op) {
	default:
		dump("cgen_floatsse", n);
		fatal("cgen_floatsse %O", n->op);
		return;

	case OMINUS:
	case OCOM:
		nr = nodintconst(-1);
		convlit(&nr, n->type);
		a = foptoas(OMUL, nl->type, 0);
		goto sbop;

	// symmetric binary
	case OADD:
	case OMUL:
		a = foptoas(n->op, nl->type, 0);
		goto sbop;

	// asymmetric binary
	case OSUB:
	case OMOD:
	case ODIV:
		a = foptoas(n->op, nl->type, 0);
		goto abop;
	}

sbop:	// symmetric binary
	if(nl->ullman < nr->ullman || nl->op == OLITERAL) {
		r = nl;
		nl = nr;
		nr = r;
	}

abop:	// asymmetric binary
	if(nl->ullman >= nr->ullman) {
		tempname(&nt, nl->type);
		cgen(nl, &nt);
		mgen(nr, &n2, N);
		regalloc(&n1, nl->type, res);
		gmove(&nt, &n1);
		gins(a, &n2, &n1);
		gmove(&n1, res);
		regfree(&n1);
		mfree(&n2);
	} else {
		regalloc(&n2, nr->type, res);
		cgen(nr, &n2);
		regalloc(&n1, nl->type, N);
		cgen(nl, &n1);
		gins(a, &n2, &n1);
		regfree(&n2);
		gmove(&n1, res);
		regfree(&n1);
	}
	return;
}

void
bgen_float(Node *n, int true, int likely, Prog *to)
{
	int et, a;
	Node *nl, *nr, *r;
	Node n1, n2, n3, tmp, t1, t2, ax;
	Prog *p1, *p2;

	nl = n->left;
	nr = n->right;
	a = n->op;
	if(!true) {
		// brcom is not valid on floats when NaN is involved.
		p1 = gbranch(AJMP, T, 0);
		p2 = gbranch(AJMP, T, 0);
		patch(p1, pc);
		// No need to avoid re-genning ninit.
		bgen_float(n, 1, -likely, p2);
		patch(gbranch(AJMP, T, 0), to);
		patch(p2, pc);
		return;
	}

	if(use_sse)
		goto sse;
	else
		goto x87;

x87:
	a = brrev(a);	// because the args are stacked
	if(a == OGE || a == OGT) {
		// only < and <= work right with NaN; reverse if needed
		r = nr;
		nr = nl;
		nl = r;
		a = brrev(a);
	}

	nodreg(&tmp, nr->type, D_F0);
	nodreg(&n2, nr->type, D_F0 + 1);
	nodreg(&ax, types[TUINT16], D_AX);
	et = simsimtype(nr->type);
	if(et == TFLOAT64) {
		if(nl->ullman > nr->ullman) {
			cgen(nl, &tmp);
			cgen(nr, &tmp);
			gins(AFXCHD, &tmp, &n2);
		} else {
			cgen(nr, &tmp);
			cgen(nl, &tmp);
		}
		gins(AFUCOMIP, &tmp, &n2);
		gins(AFMOVDP, &tmp, &tmp);	// annoying pop but still better than STSW+SAHF
	} else {
		// TODO(rsc): The moves back and forth to memory
		// here are for truncating the value to 32 bits.
		// This handles 32-bit comparison but presumably
		// all the other ops have the same problem.
		// We need to figure out what the right general
		// solution is, besides telling people to use float64.
		tempname(&t1, types[TFLOAT32]);
		tempname(&t2, types[TFLOAT32]);
		cgen(nr, &t1);
		cgen(nl, &t2);
		gmove(&t2, &tmp);
		gins(AFCOMFP, &t1, &tmp);
		gins(AFSTSW, N, &ax);
		gins(ASAHF, N, N);
	}

	goto ret;

sse:
	if(!nl->addable) {
		tempname(&n1, nl->type);
		cgen(nl, &n1);
		nl = &n1;
	}
	if(!nr->addable) {
		tempname(&tmp, nr->type);
		cgen(nr, &tmp);
		nr = &tmp;
	}
	regalloc(&n2, nr->type, N);
	gmove(nr, &n2);
	nr = &n2;

	if(nl->op != OREGISTER) {
		regalloc(&n3, nl->type, N);
		gmove(nl, &n3);
		nl = &n3;
	}

	if(a == OGE || a == OGT) {
		// only < and <= work right with NaN; reverse if needed
		r = nr;
		nr = nl;
		nl = r;
		a = brrev(a);
	}

	gins(foptoas(OCMP, nr->type, 0), nl, nr);
	if(nl->op == OREGISTER)
		regfree(nl);
	regfree(nr);

ret:
	if(a == OEQ) {
		// neither NE nor P
		p1 = gbranch(AJNE, T, -likely);
		p2 = gbranch(AJPS, T, -likely);
		patch(gbranch(AJMP, T, 0), to);
		patch(p1, pc);
		patch(p2, pc);
	} else if(a == ONE) {
		// either NE or P
		patch(gbranch(AJNE, T, likely), to);
		patch(gbranch(AJPS, T, likely), to);
	} else
		patch(gbranch(optoas(a, nr->type), T, likely), to);

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
		p->as = ACMPL;
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
