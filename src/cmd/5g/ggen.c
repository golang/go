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
	ptxt->to.type = D_CONST2;
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
		if (p->as == ATYPE)
			continue;

		if (p->from.name == D_AUTO && p->from.node)
			p->from.node->used = 1;

		if (p->to.name == D_AUTO && p->to.node)
			p->to.node->used = 1;
	}
}

// Fixup instructions after compactframe has moved all autos around.
void
fixautoused(Prog* p)
{
	Prog **lp;

	for (lp=&p; (p=*lp) != P; ) {
		if (p->as == ATYPE && p->from.node && p->from.name == D_AUTO && !p->from.node->used) {
			*lp = p->link;
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

	switch(proc) {
	default:
		fatal("ginscall: bad proc %d", proc);
		break;

	case 0:	// normal call
	case -1:	// normal call but no return
		if(f->op == ONAME && f->class == PFUNC) {
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
		p = gins(AMOVW, N, &r);
		p->from.type = D_OREG;
		p->from.reg = REGSP;
		
		p = gins(AMOVW, &r, N);
		p->to.type = D_OREG;
		p->to.reg = REGSP;
		p->to.offset = -12;
		p->scond |= C_WBIT;

		memset(&n1, 0, sizeof n1);
		n1.op = OADDR;
		n1.left = f;
		gins(AMOVW, &n1, &r);

		p = gins(AMOVW, &r, N);
		p->to.type = D_OREG;
		p->to.reg = REGSP;
		p->to.offset = 8;

		nodconst(&con, types[TINT32], argsize(f->type));
		gins(AMOVW, &con, &r);
		p = gins(AMOVW, &r, N);
		p->to.type = D_OREG;
		p->to.reg = REGSP;
		p->to.offset = 4;
		regfree(&r);

		if(proc == 1)
			ginscall(newproc, 0);
		else
			ginscall(deferproc, 0);

		nodreg(&r, types[tptr], 1);
		p = gins(AMOVW, N, N);
		p->from.type = D_CONST;
		p->from.reg = REGSP;
		p->from.offset = 12;
		p->to.reg = REGSP;
		p->to.type = D_REG;

		if(proc == 2) {
			nodconst(&con, types[TINT32], 0);
			p = gins(ACMP, &con, N);
			p->reg = 0;
			patch(gbranch(ABNE, T, -1), retpc);
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
	nodsp.xoffset = 4;
	nodo.xoffset += widthptr;
	cgen(&nodo, &nodsp);	// 4(SP) = 4(REG) -- i.data

	nodo.xoffset -= widthptr;
	cgen(&nodo, &nodr);	// REG = 0(REG) -- i.tab

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
	genlist(n->list);		// copy out args
	if(hasdefer || curfn->exit)
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
	gmove(&n2, res);

	regfree(&n1);
	regfree(&n2);
}

void
clearfat(Node *nl)
{
	uint32 w, c, q;
	Node dst, nc, nz, end;
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

	regalloc(&dst, types[tptr], N);
	agen(nl, &dst);
	nodconst(&nc, types[TUINT32], 0);
	regalloc(&nz, types[TUINT32], 0);
	cgen(&nc, &nz);

	if(q >= 4) {
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
		p = gins(AMOVBU, &nz, &dst);
		p->to.type = D_OREG;
		p->to.offset = 1;
 		p->scond |= C_PBIT;
//print("2. %P\n", p);
		c--;
	}
	regfree(&dst);
	regfree(&nz);
}
