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
		throwslice = sysfunc("throwslice");
		throwreturn = sysfunc("throwreturn");
	}

	if(fn->nbody == nil)
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
				curfn->nbody = concat(list1(nod(OAS, t->nname, N)), curfn->nbody);
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

	nodconst(&nod1, types[TINT32], 0);
	ptxt = gins(ATEXT, curfn->nname, &nod1);
	afunclit(&ptxt->from);

	ginit();
	genlist(curfn->enter);
	genlist(curfn->nbody);
	gclean();
	checklabels();
	if(nerrors != 0)
		goto ret;

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
		else {
			if(!hasdefer)
				fatal("hasdefer=0 but has defer");
			ginscall(deferproc, 0);
		}
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

	genlist(n->list);		// assign the args

	regalloc(&nodr, types[tptr], res);
	regalloc(&nodo, types[tptr], &nodr);
	nodo.op = OINDREG;

	agen(i, &nodr);		// REG = &inter

	nodindreg(&nodsp, types[tptr], D_SP);
	nodo.xoffset += widthptr;
	cgen(&nodo, &nodsp);	// 0(SP) = 8(REG) -- i.s

	nodo.xoffset -= widthptr;
	cgen(&nodo, &nodr);	// REG = 0(REG) -- i.m

	nodo.xoffset = n->left->xoffset + 3*widthptr + 8;
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
	genlist(n->list);		// copy out args
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
	ax->type = t;
	dx->type = t;

	regalloc(&n3, t, N);
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

static void
savex(int dr, Node *x, Node *oldx, Node *res, Type *t)
{
	int r;

	r = reg[dr];
	nodreg(x, types[TINT64], dr);

	// save current ax and dx if they are live
	// and not the destination
	memset(oldx, 0, sizeof *oldx);
	if(r > 0 && !samereg(x, res)) {
		regalloc(oldx, types[TINT64], N);
		gmove(x, oldx);
	}

	regalloc(x, t, x);
}

static void
restx(Node *x, Node *oldx)
{
	regfree(x);

	if(oldx->op != 0) {
		x->type = types[TINT64];
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
	Node ax, dx, oldax, olddx;
	Node n1, n2, n3, savl, savr;
	int n, w, s, a;
	Magic m;

	if(nl->ullman >= UINF) {
		tempname(&savl, nl->type);
		cgen(nl, &savl);
		nl = &savl;
	}
	if(nr->ullman >= UINF) {
		tempname(&savr, nr->type);
		cgen(nr, &savr);
		nr = &savr;
	}

	if(nr->op != OLITERAL)
		goto longdiv;

	// special cases of mod/div
	// by a constant
	w = nl->type->width*8;
	s = 0;
	n = powtwo(nr);
	if(n >= 1000) {
		// negative power of 2
		s = 1;
		n -= 1000;
	}

	if(n+1 >= w) {
		// just sign bit
		goto longdiv;
	}

	if(n < 0)
		goto divbymul;
	switch(n) {
	case 0:
		// divide by 1
		regalloc(&n1, nl->type, res);
		cgen(nl, &n1);
		if(op == OMOD) {
			gins(optoas(OXOR, nl->type), &n1, &n1);
		} else
		if(s)
			gins(optoas(OMINUS, nl->type), N, &n1);
		gmove(&n1, res);
		regfree(&n1);
		return;
	case 1:
		// divide by 2
		if(op == OMOD) {
			if(issigned[nl->type->etype])
				goto longmod;
			regalloc(&n1, nl->type, res);
			cgen(nl, &n1);
			nodconst(&n2, nl->type, 1);
			gins(optoas(OAND, nl->type), &n2, &n1);
			gmove(&n1, res);
			regfree(&n1);
			return;
		}
		regalloc(&n1, nl->type, res);
		cgen(nl, &n1);
		if(!issigned[nl->type->etype])
			break;

		// develop -1 iff nl is negative
		regalloc(&n2, nl->type, N);
		gmove(&n1, &n2);
		nodconst(&n3, nl->type, w-1);
		gins(optoas(ORSH, nl->type), &n3, &n2);
		gins(optoas(OSUB, nl->type), &n2, &n1);
		regfree(&n2);
		break;
	default:
		if(op == OMOD) {
			if(issigned[nl->type->etype])
				goto longmod;
			regalloc(&n1, nl->type, res);
			cgen(nl, &n1);
			nodconst(&n2, nl->type, mpgetfix(nr->val.u.xval)-1);
			if(!smallintconst(&n2)) {
				regalloc(&n3, nl->type, N);
				gmove(&n2, &n3);
				gins(optoas(OAND, nl->type), &n3, &n1);
				regfree(&n3);
			} else
				gins(optoas(OAND, nl->type), &n2, &n1);
			gmove(&n1, res);
			regfree(&n1);
			return;
		}
		regalloc(&n1, nl->type, res);
		cgen(nl, &n1);
		if(!issigned[nl->type->etype])
			break;

		// develop (2^k)-1 iff nl is negative
		regalloc(&n2, nl->type, N);
		gmove(&n1, &n2);
		nodconst(&n3, nl->type, w-1);
		gins(optoas(ORSH, nl->type), &n3, &n2);
		nodconst(&n3, nl->type, w-n);
		gins(optoas(ORSH, tounsigned(nl->type)), &n3, &n2);
		gins(optoas(OADD, nl->type), &n2, &n1);
		regfree(&n2);
		break;
	}
	nodconst(&n2, nl->type, n);
	gins(optoas(ORSH, nl->type), &n2, &n1);
	if(s)
		gins(optoas(OMINUS, nl->type), N, &n1);
	gmove(&n1, res);
	regfree(&n1);
	return;

divbymul:
	// try to do division by multiply by (2^w)/d
	// see hacker's delight chapter 10
	switch(simtype[nl->type->etype]) {
	default:
		goto longdiv;

	case TUINT8:
	case TUINT16:
	case TUINT32:
	case TUINT64:
		m.w = w;
		m.ud = mpgetfix(nr->val.u.xval);
		umagic(&m);
		if(m.bad)
			break;
		if(op == OMOD)
			goto longmod;

		savex(D_AX, &ax, &oldax, res, nl->type);
		savex(D_DX, &dx, &olddx, res, nl->type);

		regalloc(&n1, nl->type, N);
		cgen(nl, &n1);				// num -> reg(n1)

		nodconst(&n2, nl->type, m.um);
		gmove(&n2, &ax);			// const->ax

		gins(optoas(OHMUL, nl->type), &n1, N);	// imul reg
		if(w == 8) {
			// fix up 8-bit multiply
			Node ah, dl;
			nodreg(&ah, types[TUINT8], D_AH);
			nodreg(&dl, types[TUINT8], D_DL);
			gins(AMOVB, &ah, &dl);
		}

		if(m.ua) {
			// need to add numerator accounting for overflow
			gins(optoas(OADD, nl->type), &n1, &dx);
			nodconst(&n2, nl->type, 1);
			gins(optoas(ORRC, nl->type), &n2, &dx);
			nodconst(&n2, nl->type, m.s-1);
			gins(optoas(ORSH, nl->type), &n2, &dx);
		} else {
			nodconst(&n2, nl->type, m.s);
			gins(optoas(ORSH, nl->type), &n2, &dx);	// shift dx
		}


		regfree(&n1);
		gmove(&dx, res);

		restx(&ax, &oldax);
		restx(&dx, &olddx);
		return;

	case TINT8:
	case TINT16:
	case TINT32:
	case TINT64:
		m.w = w;
		m.sd = mpgetfix(nr->val.u.xval);
		smagic(&m);
		if(m.bad)
			break;
		if(op == OMOD)
			goto longmod;

		savex(D_AX, &ax, &oldax, res, nl->type);
		savex(D_DX, &dx, &olddx, res, nl->type);

		regalloc(&n1, nl->type, N);
		cgen(nl, &n1);				// num -> reg(n1)

		nodconst(&n2, nl->type, m.sm);
		gmove(&n2, &ax);			// const->ax

		gins(optoas(OHMUL, nl->type), &n1, N);	// imul reg
		if(w == 8) {
			// fix up 8-bit multiply
			Node ah, dl;
			nodreg(&ah, types[TUINT8], D_AH);
			nodreg(&dl, types[TUINT8], D_DL);
			gins(AMOVB, &ah, &dl);
		}

		if(m.sm < 0) {
			// need to add numerator
			gins(optoas(OADD, nl->type), &n1, &dx);
		}

		nodconst(&n2, nl->type, m.s);
		gins(optoas(ORSH, nl->type), &n2, &dx);	// shift dx

		nodconst(&n2, nl->type, w-1);
		gins(optoas(ORSH, nl->type), &n2, &n1);	// -1 iff num is neg
		gins(optoas(OSUB, nl->type), &n1, &dx);	// added

		if(m.sd < 0) {
			// this could probably be removed
			// by factoring it into the multiplier
			gins(optoas(OMINUS, nl->type), N, &dx);
		}

		regfree(&n1);
		gmove(&dx, res);

		restx(&ax, &oldax);
		restx(&dx, &olddx);
		return;
	}
	goto longdiv;

longdiv:
	// division and mod using (slow) hardware instruction
	savex(D_AX, &ax, &oldax, res, nl->type);
	savex(D_DX, &dx, &olddx, res, nl->type);
	dodiv(op, nl, nr, res, &ax, &dx);
	restx(&ax, &oldax);
	restx(&dx, &olddx);
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
 * generate shift according to op, one of:
 *	res = nl << nr
 *	res = nl >> nr
 */
void
cgen_shift(int op, Node *nl, Node *nr, Node *res)
{
	Node n1, n2, n3, n4, n5, cx, oldcx;
	int a, rcx;
	Prog *p1;
	uvlong sc;

	a = optoas(op, nl->type);

	if(nr->op == OLITERAL) {
		regalloc(&n1, nl->type, res);
		cgen(nl, &n1);
		sc = mpgetfix(nr->val.u.xval);
		if(sc >= nl->type->width*8) {
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
	regalloc(&n1, nr->type, &n1);		// to hold the shift type in CX
	regalloc(&n3, types[TUINT64], &n1);	// to clear high bits of CX

	nodreg(&cx, types[TUINT64], D_CX);
	memset(&oldcx, 0, sizeof oldcx);
	if(rcx > 0 && !samereg(&cx, res)) {
		regalloc(&oldcx, types[TUINT64], N);
		gmove(&cx, &oldcx);
	}

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

	if(oldcx.op != 0) {
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
 * no 2-operand byte multiply instruction so have to do
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
getlit(Node *lit)
{
	if(smallintconst(lit))
		return mpgetfix(lit->val.u.xval);
	return -1;
}

int
stataddr(Node *nam, Node *n)
{
	int l;

	if(n == N)
		goto no;

	switch(n->op) {
	case ONAME:
		*nam = *n;
		return n->addable;

	case ODOT:
		if(!stataddr(nam, n->left))
			break;
		nam->xoffset += n->xoffset;
		nam->type = n->type;
		return 1;

	case OINDEX:
		if(n->left->type->bound < 0)
			break;
		if(!stataddr(nam, n->left))
			break;
		l = getlit(n->right);
		if(l < 0)
			break;
		nam->xoffset += l*n->type->width;
		nam->type = n->type;
		return 1;
	}

no:
	return 0;
}

int
gen_as_init(Node *nr, Node *nl)
{
	Node nam, nod1;
	Prog *p;

	if(!initflag)
		goto no;

	if(nr == N) {
		if(!stataddr(&nam, nl))
			goto no;
		if(nam.class != PEXTERN)
			goto no;
		return 1;
	}

	if(nr->op == OCOMPSLICE) {
		// create a slice pointing to an array
		if(!stataddr(&nam, nl)) {
			dump("stataddr", nl);
			goto no;
		}

		p = gins(ADATA, &nam, nr->left);
		p->from.scale = types[tptr]->width;
		p->to.index = p->to.type;
		p->to.type = D_ADDR;
//print("%P\n", p);

		nodconst(&nod1, types[TINT32], nr->left->type->bound);
		p = gins(ADATA, &nam, &nod1);
		p->from.scale = types[TINT32]->width;
		p->from.offset += types[tptr]->width;
//print("%P\n", p);

		p = gins(ADATA, &nam, &nod1);
		p->from.scale = types[TINT32]->width;
		p->from.offset += types[tptr]->width+types[TINT32]->width;

		goto yes;
	}

	if(nr->op == OCOMPMAP) {
		goto yes;
	}

	if(nr->type == T ||
	   !eqtype(nl->type, nr->type))
		goto no;

	if(!stataddr(&nam, nl))
		goto no;
	if(nam.class != PEXTERN)
		goto no;

	switch(nr->op) {
	default:
		goto no;

	case OLITERAL:
		goto lit;
	}

no:
	return 0;

lit:
	switch(nr->type->etype) {
	default:
		goto no;

	case TBOOL:
		if(memcmp(nam.sym->name, "initdone·", 9) == 0)
			goto no;
	case TINT8:
	case TUINT8:
	case TINT16:
	case TUINT16:
	case TINT32:
	case TUINT32:
	case TINT64:
	case TUINT64:
	case TINT:
	case TUINT:
	case TFLOAT32:
	case TFLOAT64:
	case TFLOAT:
		p = gins(ADATA, &nam, nr);
		p->from.scale = nr->type->width;
		break;

	case TSTRING:
		p = gins(ADATA, &nam, N);
		datastring(nr->val.u.sval->s, nr->val.u.sval->len, &p->to);
		p->from.scale = types[tptr]->width;
		p->to.index = p->to.type;
		p->to.type = D_ADDR;
//print("%P\n", p);

		nodconst(&nod1, types[TINT32], nr->val.u.sval->len);
		p = gins(ADATA, &nam, &nod1);
		p->from.scale = types[TINT32]->width;
		p->from.offset += types[tptr]->width;
//print("%P\n", p);
		break;
	}

yes:
//dump("\ngen_as_init", nl);
//dump("", nr);
//print("%P\n", p);
	return 1;
}

static int
regcmp(Node *ra, Node *rb)
{
	return ra->local - rb->local;
}

void
getargs(NodeList *nn, Node *reg, int n)
{
	NodeList *l;
	int i;

	l = nn;
	for(i=0; i<n; i++) {
		if(!smallintconst(l->n->right) && !isslice(l->n->right->type)) {
			regalloc(reg+i, l->n->right->type, N);
			cgen(l->n->right, reg+i);
		} else
			reg[i] = *l->n->right;
		if(reg[i].local != 0)
			yyerror("local used");
		reg[i].local = l->n->left->xoffset;
		l = l->next;
	}
	qsort((void*)reg, n, sizeof(*reg), regcmp);
	for(i=0; i<n; i++)
		reg[i].local = 0;
}

void
cmpandthrow(Node *nl, Node *nr)
{
	vlong cl, cr;
	Prog *p1;
	int op;
	Node *c;

	op = OLE;
	if(smallintconst(nl)) {
		cl = mpgetfix(nl->val.u.xval);
		if(cl == 0)
			return;
		if(smallintconst(nr)) {
			cr = mpgetfix(nr->val.u.xval);
			if(cl > cr)
				ginscall(throwslice, 0);
			return;
		}

		// put the constant on the right
		op = brrev(op);
		c = nl;
		nl = nr;
		nr = c;
	}

	gins(optoas(OCMP, types[TUINT32]), nl, nr);
	p1 = gbranch(optoas(op, types[TUINT32]), T);
	ginscall(throwslice, 0);
	patch(p1, pc);
}

int
sleasy(Node *n)
{
	if(n->op != ONAME)
		return 0;
	if(!n->addable)
		return 0;
return 0;
	return 1;
}

// generate inline code for
//	slicearray
//	sliceslice
//	arraytoslice
int
cgen_inline(Node *n, Node *res)
{
	Node nodes[10];
	Node n1, n2;
	vlong v;
	int i;

	if(n->op != OCALLFUNC)
		goto no;
	if(!n->left->addable)
		goto no;
	if(!sleasy(res))
		goto no;
	if(strcmp(n->left->sym->package, "sys") != 0)
		goto no;
	if(strcmp(n->left->sym->name, "slicearray") == 0)
		goto slicearray;
	if(strcmp(n->left->sym->name, "sliceslice") == 0)
		goto sliceslice;
	if(strcmp(n->left->sym->name, "arraytoslice") == 0)
		goto arraytoslice;
	goto no;

slicearray:
	getargs(n->list, nodes, 5);

	// if(hb[3] > nel[1]) goto throw
	cmpandthrow(nodes+3, nodes+1);

	// if(lb[2] > hb[3]) goto throw
	cmpandthrow(nodes+2, nodes+3);

	// len = hb[3] - lb[2] (destroys hb)
	n2 = *res;
	n2.xoffset += Array_nel;

	if(smallintconst(nodes+3) && smallintconst(nodes+2)) {
		v = mpgetfix((nodes+3)->val.u.xval) -
			mpgetfix((nodes+2)->val.u.xval);
		nodconst(&n1, types[TUINT32], v);
		gins(optoas(OAS, types[TUINT32]), &n1, &n2);
	} else {
		regalloc(&n1, types[TUINT32], nodes+3);
		gmove(nodes+3, &n1);
		if(!smallintconst(nodes+2) || mpgetfix((nodes+2)->val.u.xval) != 0)
			gins(optoas(OSUB, types[TUINT32]), nodes+2, &n1);
		gins(optoas(OAS, types[TUINT32]), &n1, &n2);
		regfree(&n1);
	}

	// cap = nel[1] - lb[2] (destroys nel)
	n2 = *res;
	n2.xoffset += Array_cap;

	if(smallintconst(nodes+1) && smallintconst(nodes+2)) {
		v = mpgetfix((nodes+1)->val.u.xval) -
			mpgetfix((nodes+2)->val.u.xval);
		nodconst(&n1, types[TUINT32], v);
		gins(optoas(OAS, types[TUINT32]), &n1, &n2);
	} else {
		regalloc(&n1, types[TUINT32], nodes+1);
		gmove(nodes+1, &n1);
		if(!smallintconst(nodes+2) || mpgetfix((nodes+2)->val.u.xval) != 0)
			gins(optoas(OSUB, types[TUINT32]), nodes+2, &n1);
		gins(optoas(OAS, types[TUINT32]), &n1, &n2);
		regfree(&n1);
	}

	// ary = old[0] + (lb[2] * width[4]) (destroys old)
	n2 = *res;
	n2.xoffset += Array_array;

	if(smallintconst(nodes+2) && smallintconst(nodes+4)) {
		v = mpgetfix((nodes+2)->val.u.xval) *
			mpgetfix((nodes+4)->val.u.xval);
		if(v != 0) {
			nodconst(&n1, types[tptr], v);
			gins(optoas(OADD, types[tptr]), &n1, nodes+0);
		}
	} else {
		regalloc(&n1, types[tptr], nodes+2);
		gmove(nodes+2, &n1);
		if(!smallintconst(nodes+4) || mpgetfix((nodes+4)->val.u.xval) != 1)
			gins(optoas(OMUL, types[tptr]), nodes+4, &n1);
		gins(optoas(OADD, types[tptr]), &n1, nodes+0);
		regfree(&n1);
	}
	gins(optoas(OAS, types[tptr]), nodes+0, &n2);

	for(i=0; i<5; i++) {
		if((nodes+i)->op == OREGISTER)
			regfree(nodes+i);
	}
	return 1;

sliceslice:
	getargs(n->list, nodes, 4);
	if(!sleasy(&nodes[0])) {
		for(i=0; i<4; i++) {
			if(nodes[i].op == OREGISTER)
				regfree(&nodes[i]);
		}
		goto no;
	}

	// if(hb[2] > old.cap[0]) goto throw;
	n2 = nodes[0];
	n2.xoffset += Array_cap;
	cmpandthrow(nodes+2, &n2);

	// if(lb[1] > hb[2]) goto throw;
	cmpandthrow(&nodes[1], &nodes[2]);

	// ret.len = hb[2]-lb[1]; (destroys hb[2])
	n2 = *res;
	n2.xoffset += Array_nel;

	if(smallintconst(&nodes[2]) && smallintconst(&nodes[1])) {
		v = mpgetfix((nodes+2)->val.u.xval) -
			mpgetfix(nodes[2].val.u.xval);
		nodconst(&n1, types[TUINT32], v);
		gins(optoas(OAS, types[TUINT32]), &n1, &n2);
	} else {
		regalloc(&n1, types[TUINT32], nodes+2);
		gmove(nodes+2, &n1);
		if(!smallintconst(&nodes[1]) || mpgetfix(nodes[1].val.u.xval) != 0)
			gins(optoas(OSUB, types[TUINT32]), &nodes[1], &n1);
		gins(optoas(OAS, types[TUINT32]), &n1, &n2);
		regfree(&n1);
	}

	// ret.cap = old.cap[0]-lb[1]; (uses hb[2])
	n2 = nodes[0];
	n2.xoffset += Array_cap;

	regalloc(&n1, types[TUINT32], nodes+2);
	gins(optoas(OAS, types[TUINT32]), &n2, &n1);
	if(!smallintconst(nodes+1) || mpgetfix((nodes+1)->val.u.xval) != 0)
		gins(optoas(OSUB, types[TUINT32]), nodes+1, &n1);

	n2 = *res;
	n2.xoffset += Array_cap;
	gins(optoas(OAS, types[TUINT32]), &n1, &n2);
	regfree(&n1);

	// ret.array = old.array[0]+lb[1]*width[3]; (uses lb)
	n2 = nodes[0];
	n2.xoffset += Array_array;

	regalloc(&n1, types[tptr], nodes+1);
	if(smallintconst(nodes+1) && smallintconst(nodes+3)) {
		gins(optoas(OAS, types[tptr]), &n2, &n1);
		v = mpgetfix((nodes+1)->val.u.xval) *
			mpgetfix((nodes+3)->val.u.xval);
		if(v != 0) {
			nodconst(&n2, types[tptr], v);
			gins(optoas(OADD, types[tptr]), &n2, &n1);
		}
	} else {
		gmove(nodes+1, &n1);
		if(!smallintconst(nodes+3) || mpgetfix((nodes+3)->val.u.xval) != 1)
			gins(optoas(OMUL, types[tptr]), nodes+3, &n1);
		gins(optoas(OADD, types[tptr]), &n2, &n1);
	}

	n2 = *res;
	n2.xoffset += Array_array;
	gins(optoas(OAS, types[tptr]), &n1, &n2);
	regfree(&n1);

	for(i=0; i<4; i++) {
		if((nodes+i)->op == OREGISTER)
			regfree(nodes+i);
	}
	return 1;

arraytoslice:
	// ret.len = nel;
	// ret.cap = nel;
	// ret.array = old;
	goto no;

no:
	return 0;
}
