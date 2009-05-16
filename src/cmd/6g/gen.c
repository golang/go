// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


#undef	EXTERN
#define	EXTERN
#include "gg.h"
#include "opt.h"

void
compile(Node *fn)
{
	Plist *pl;
	Node nod1;
	Prog *ptxt;
	int32 lno;
	Type *t;
	Iter save;

	if(newproc == N) {
		newproc = sysfunc("newproc");
		deferproc = sysfunc("deferproc");
		deferreturn = sysfunc("deferreturn");
		throwindex = sysfunc("throwindex");
		throwreturn = sysfunc("throwreturn");
	}

	if(fn->nbody == N)
		return;

	// set up domain for labels
	labellist = L;

	lno = setlineno(fn);

	curfn = fn;
	dowidth(curfn->type);

	if(curfn->type->outnamed) {
		// add clearing of the output parameters
		t = structfirst(&save, getoutarg(curfn->type));
		while(t != T) {
			if(t->nname != N)
				curfn->nbody = list(nod(OAS, t->nname, N), curfn->nbody);
			t = structnext(&save);
		}
	}

	hasdefer = 0;
	walk(curfn);
	if(nerrors != 0)
		goto ret;

	allocparams();

	continpc = P;
	breakpc = P;

	pl = newplist();
	pl->name = curfn->nname;
	pl->locals = autodcl;

	nodconst(&nod1, types[TINT32], 0);
	ptxt = gins(ATEXT, curfn->nname, &nod1);
	afunclit(&ptxt->from);

	ginit();
	gen(curfn->enter);
	gen(curfn->nbody);
	gclean();
	checklabels();

	if(curfn->type->outtuple != 0)
		ginscall(throwreturn, 0);

	if(hasdefer)
		ginscall(deferreturn, 0);
	pc->as = ARET;	// overwrite AEND
	pc->lineno = lineno;

	if(!debug['N'] || debug['R'] || debug['P'])
		regopt(ptxt);

	// fill in argument size
	ptxt->to.offset = rnd(curfn->type->argwid, maxround);

	// fill in final stack size
	ptxt->to.offset <<= 32;
	ptxt->to.offset |= rnd(stksize+maxarg, maxround);

	if(debug['f'])
		frame(0);

ret:
	lineno = lno;
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
		p = gins(ACALL, N, f);
		afunclit(&p->to);
		break;

	case 1:	// call in new proc (go)
	case 2:	// defered call (defer)
		nodreg(&reg, types[TINT64], D_AX);
		gins(APUSHQ, f, N);
		nodconst(&con, types[TINT32], argsize(f->type));
		gins(APUSHQ, &con, N);
		if(proc == 1)
			ginscall(newproc, 0);
		else
			ginscall(deferproc, 0);
		gins(APOPQ, N, &reg);
		gins(APOPQ, N, &reg);
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
	Node tmpi, nodo, nodr, nodsp;

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

	gen(n->right);			// args

	regalloc(&nodr, types[tptr], res);
	regalloc(&nodo, types[tptr], &nodr);
	nodo.op = OINDREG;

	agen(i, &nodr);		// REG = &inter

	nodindreg(&nodsp, types[tptr], D_SP);
	nodo.xoffset += widthptr;
	cgen(&nodo, &nodsp);	// 0(SP) = 8(REG) -- i.s

	nodo.xoffset -= widthptr;
	cgen(&nodo, &nodr);	// REG = 0(REG) -- i.m

	nodo.xoffset = n->left->xoffset + 4*widthptr;
	cgen(&nodo, &nodr);	// REG = 32+offset(REG) -- i.m->fun[f]

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

	gen(n->right);		// assign the args
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
		gins(ALEAQ, &nod1, &nod2);
		gins(AMOVQ, &nod2, res);
		regfree(&nod2);
	} else
		gins(ALEAQ, &nod1, res);
}

/*
 * generate return.
 * n->left is assignments to return values.
 */
void
cgen_ret(Node *n)
{
	gen(n->left);		// copy out args
	if(hasdefer)
		ginscall(deferreturn, 0);
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
	if(nr->ullman > nl->ullman) {
		regalloc(&n2, nr->type, N);
		cgen(nr, &n2);
		igen(nl, &n1, N);
	} else {
		igen(nl, &n1, N);
		regalloc(&n2, nr->type, N);
		cgen(nr, &n2);
	}

	n3 = *n;
	n3.left = &n1;
	n3.right = &n2;
	n3.op = n->etype;

	regalloc(&n4, nl->type, N);
	cgen(&n3, &n4);
	gmove(&n4, &n1);

	regfree(&n1);
	regfree(&n2);
	regfree(&n4);

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
	int a;
	Node n3, n4;
	Type *t;

	t = nl->type;
	if(t->width == 1) {
		if(issigned[t->etype])
			t = types[TINT32];
		else
			t = types[TUINT32];
	}
	a = optoas(op, t);

	regalloc(&n3, nr->type, N);
	if(nl->ullman >= nr->ullman) {
		cgen(nl, ax);
		if(!issigned[t->etype]) {
			nodconst(&n4, t, 0);
			gmove(&n4, dx);
		} else
			gins(optoas(OEXTEND, t), N, N);
		cgen(nr, &n3);
	} else {
		cgen(nr, &n3);
		cgen(nl, ax);
		if(!issigned[t->etype]) {
			nodconst(&n4, t, 0);
			gmove(&n4, dx);
		} else
			gins(optoas(OEXTEND, t), N, N);
	}
	gins(a, &n3, N);
	regfree(&n3);

	if(op == ODIV)
		gmove(ax, res);
	else
		gmove(dx, res);
}

/*
 * generate division according to op, one of:
 *	res = nl / nr
 *	res = nl % nr
 */
void
cgen_div(int op, Node *nl, Node *nr, Node *res)
{
	Node ax, dx;
	int rax, rdx;

	rax = reg[D_AX];
	rdx = reg[D_DX];

	nodreg(&ax, types[TINT64], D_AX);
	nodreg(&dx, types[TINT64], D_DX);
	regalloc(&ax, nl->type, &ax);
	regalloc(&dx, nl->type, &dx);

	dodiv(op, nl, nr, res, &ax, &dx);

	regfree(&ax);
	regfree(&dx);
}

/*
 * generate shift according to op, one of:
 *	res = nl << nr
 *	res = nl >> nr
 */
void
cgen_shift(int op, Node *nl, Node *nr, Node *res)
{
	Node n1, n2, n3;
	int a;
	Prog *p1;

	a = optoas(op, nl->type);

	if(nr->op == OLITERAL) {
		regalloc(&n1, nl->type, res);
		cgen(nl, &n1);
		if(mpgetfix(nr->val.u.xval) >= nl->type->width*8) {
			// large shift gets 2 shifts by width
			nodconst(&n3, types[TUINT32], nl->type->width*8-1);
			gins(a, &n3, &n1);
			gins(a, &n3, &n1);
		} else
			gins(a, nr, &n1);
		gmove(&n1, res);
		regfree(&n1);
		goto ret;
	}

	nodreg(&n1, types[TUINT32], D_CX);
	regalloc(&n1, nr->type, &n1);		// to hold the shift type in CX
	regalloc(&n3, types[TUINT64], &n1);	// to clear high bits of CX

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
	nodconst(&n3, types[TUINT64], nl->type->width*8);
	gins(optoas(OCMP, types[TUINT64]), &n1, &n3);
	p1 = gbranch(optoas(OLT, types[TUINT64]), T);
	if(op == ORSH && issigned[nl->type->etype]) {
		nodconst(&n3, types[TUINT32], nl->type->width*8-1);
		gins(a, &n3, &n2);
	} else {
		nodconst(&n3, nl->type, 0);
		gmove(&n3, &n2);
	}
	patch(p1, pc);
	gins(a, &n1, &n2);

	gmove(&n2, res);

	regfree(&n1);
	regfree(&n2);

ret:
	;
}

/*
 * generate byte multiply:
 *	res = nl * nr
 * no byte multiply instruction so have to do
 * 16-bit multiply and take bottom half.
 */
void
cgen_bmul(int op, Node *nl, Node *nr, Node *res)
{
	Node n1b, n2b, n1w, n2w;
	Type *t;
	int a;

	if(nl->ullman >= nr->ullman) {
		regalloc(&n1b, nl->type, res);
		cgen(nl, &n1b);
		regalloc(&n2b, nr->type, N);
		cgen(nr, &n2b);
	} else {
		regalloc(&n2b, nr->type, N);
		cgen(nr, &n2b);
		regalloc(&n1b, nl->type, res);
		cgen(nl, &n1b);
	}

	// copy from byte to short registers
	t = types[TUINT16];
	if(issigned[nl->type->etype])
		t = types[TINT16];

	regalloc(&n2w, t, &n2b);
	cgen(&n2b, &n2w);

	regalloc(&n1w, t, &n1b);
	cgen(&n1b, &n1w);

	a = optoas(op, t);
	gins(a, &n2w, &n1w);
	cgen(&n1w, &n1b);
	cgen(&n1b, res);

	regfree(&n1w);
	regfree(&n2w);
	regfree(&n1b);
	regfree(&n2b);
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
	c = w % 8;	// bytes
	q = w / 8;	// quads

	gconreg(AMOVQ, 0, D_AX);
	nodreg(&n1, types[tptr], D_DI);
	agen(nl, &n1);

	if(q >= 4) {
		gconreg(AMOVQ, q, D_CX);
		gins(AREP, N, N);	// repeat
		gins(ASTOSQ, N, N);	// STOQ AL,*(DI)+
	} else
	while(q > 0) {
		gins(ASTOSQ, N, N);	// STOQ AL,*(DI)+
		q--;
	}

	if(c >= 4) {
		gconreg(AMOVQ, c, D_CX);
		gins(AREP, N, N);	// repeat
		gins(ASTOSB, N, N);	// STOB AL,*(DI)+
	} else
	while(c > 0) {
		gins(ASTOSB, N, N);	// STOB AL,*(DI)+
		c--;
	}
}

int
gen_as_init(Node *nr, Node *nl)
{
	return 0;
}
