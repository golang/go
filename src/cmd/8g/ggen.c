// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#undef	EXTERN
#define	EXTERN
#include <u.h>
#include <libc.h>
#include "gg.h"
#include "opt.h"

void
defframe(Prog *ptxt)
{
	// fill in argument size
	ptxt->to.offset2 = rnd(curfn->type->argwid, widthptr);

	// fill in final stack size
	if(stksize > maxstksize)
		maxstksize = stksize;
	ptxt->to.offset = rnd(maxstksize+maxarg, widthptr);
	maxstksize = 0;
}

// Sweep the prog list to mark any used nodes.
void
markautoused(Prog* p)
{
	for (; p; p = p->link) {
		if (p->from.type == D_AUTO && p->from.node)
			p->from.node->used = 1;

		if (p->to.type == D_AUTO && p->to.node)
			p->to.node->used = 1;
	}
}

// Fixup instructions after compactframe has moved all autos around.
void
fixautoused(Prog* p)
{
	for (; p; p = p->link) {
		if (p->from.type == D_AUTO && p->from.node)
			p->from.offset += p->from.node->stkdelta;

		if (p->to.type == D_AUTO && p->to.node)
			p->to.offset += p->to.node->stkdelta;
	}
}

void
clearfat(Node *nl)
{
	uint32 w, c, q;
	Node n1;

	/* clear a fat object */
	if(debug['g'])
		dump("\nclearfat", nl);

	w = nl->type->width;
	// Avoid taking the address for simple enough types.
	if(componentgen(N, nl))
		return;

	c = w % 4;	// bytes
	q = w / 4;	// quads

	gconreg(AMOVL, 0, D_AX);
	nodreg(&n1, types[tptr], D_DI);
	agen(nl, &n1);

	if(q >= 4) {
		gconreg(AMOVL, q, D_CX);
		gins(AREP, N, N);	// repeat
		gins(ASTOSL, N, N);	// STOL AL,*(DI)+
	} else
	while(q > 0) {
		gins(ASTOSL, N, N);	// STOL AL,*(DI)+
		q--;
	}

	if(c >= 4) {
		gconreg(AMOVL, c, D_CX);
		gins(AREP, N, N);	// repeat
		gins(ASTOSB, N, N);	// STOB AL,*(DI)+
	} else
	while(c > 0) {
		gins(ASTOSB, N, N);	// STOB AL,*(DI)+
		c--;
	}
}

/*
 * generate:
 *	call f
 *	proc=0	normal call
 *	proc=1	goroutine run in new proc
 *	proc=2	defer call save away stack
 */
void
ginscall(Node *f, int proc)
{
	Prog *p;
	Node reg, con;

	switch(proc) {
	default:
		fatal("ginscall: bad proc %d", proc);
		break;

	case 0:	// normal call
	case -1:	// normal call but no return
		p = gins(ACALL, N, f);
		afunclit(&p->to);
		if(proc == -1 || noreturn(p))
			gins(AUNDEF, N, N);
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
			patch(gbranch(AJNE, T, -1), retpc);
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
	nodo.op = OINDREG;
	nodo.xoffset = n->left->xoffset + 3*widthptr + 8;
	cgen(&nodo, &nodr);	// REG = 20+offset(REG) -- i.tab->fun[f]

	// BOTCH nodr.type = fntype;
	nodr.type = n->left->type;
	ginscall(&nodr, proc);

	regfree(&nodr);
	regfree(&nodo);

	setmaxarg(n->left->type);
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

	setmaxarg(t);

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
	genlist(n->list);		// copy out args
	if(retpc)
		gjmp(retpc);
	else
		gins(ARET, N, N);
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
	Node n1, n2, *tmp;
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

	regalloc(&n1, t, res);
	cgen(nl, &n1);
	regalloc(&n2, t, N);
	cgen(nr, &n2);
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
		nodreg(&dx, t, D_DL);
		gmove(&ax, &dx);
	}
	nodreg(&dx, t, D_DX);
	gmove(&dx, res);
}

