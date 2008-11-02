// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


#undef	EXTERN
#define	EXTERN
#include "gg.h"

enum
{
	// random unused opcode
	AJMPX	= AADDPD,
};

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
	newproc = nod(ONAME, N, N);
	newproc->sym = pkglookup("newproc", "sys");
	newproc->class = PEXTERN;
	newproc->addable = 1;
	newproc->ullman = 0;
}

if(throwindex == N) {
	throwindex = nod(ONAME, N, N);
	throwindex->sym = pkglookup("throwindex", "sys");
	throwindex->class = PEXTERN;
	throwindex->addable = 1;
	throwindex->ullman = 0;
}

if(throwreturn == N) {
	throwreturn = nod(ONAME, N, N);
	throwreturn->sym = pkglookup("throwreturn", "sys");
	throwreturn->class = PEXTERN;
	throwreturn->addable = 1;
	throwreturn->ullman = 0;
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
			if(t->nname != N && t->nname->sym->name[0] != '_') {
				curfn->nbody = list(nod(OAS, t->nname, N), curfn->nbody);
			}
			t = structnext(&save);
		}
	}

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

//	inarggen();

	ginit();
	gen(curfn->nbody, L);
	gclean();
	checklabels();

	if(curfn->type->outtuple != 0) {
		gins(ACALL, N, throwreturn);
	}

	pc->as = ARET;	// overwrite AEND
	pc->lineno = lineno;

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

void
allocparams(void)
{
	Dcl *d;
	Node *n;
	uint32 w;

	/*
	 * allocate (set xoffset) the stack
	 * slots for all automatics.
	 * allocated starting at -w down.
	 */
	for(d=autodcl; d!=D; d=d->forw) {
		if(d->op != ONAME)
			continue;

		n = d->dnode;
		if(n->class != PAUTO)
			continue;

		dowidth(n->type);
		w = n->type->width;
		stksize += w;
		stksize = rnd(stksize, w);

		n->xoffset = -stksize;
	}
}

/*
 * compile statements
 */
void
gen(Node *n, Label *labloop)
{
	int32 lno;
	Prog *scontin, *sbreak;
	Prog *p1, *p2, *p3;
	Node *l;
	Label *lab;

	lno = setlineno(n);

loop:
	if(n == N)
		goto ret;
	if(n->ninit)
		gen(n->ninit, L);

	setlineno(n);

	switch(n->op) {
	default:
		fatal("gen: unknown op %N", n);
		break;

	case OLIST:
		l = n->left;
		gen(l, L);
		if(l != N && l->op == OLABEL) {
			// call the next statement with a label
			l = n->right;
			if(l != N) {
				if(l->op != OLIST) {
					gen(l, labellist);
					break;
				}
				gen(l->left, labellist);
				n = l->right;
				labloop = L;
				goto loop;
			}
		}
		n = n->right;
		labloop = L;
		goto loop;

	case OPANIC:
		genpanic();
		break;

	case OCASE:
	case OFALL:
	case OXCASE:
	case OXFALL:
	case OEMPTY:
		break;

	case OLABEL:
		lab = mal(sizeof(*lab));
		lab->link = labellist;
		labellist = lab;
		lab->sym = n->left->sym;

		lab->op = OLABEL;
		lab->label = pc;
		break;

	case OGOTO:
		lab = mal(sizeof(*lab));
		lab->link = labellist;
		labellist = lab;
		lab->sym = n->left->sym;

		lab->op = OGOTO;
		lab->label = pc;
		gbranch(AJMP, T);
		break;

	case OBREAK:
		if(n->left != N) {
			lab = findlab(n->left->sym);
			if(lab == L || lab->breakpc == P) {
				yyerror("break label is not defined: %S", n->left->sym);
				break;
			}
			patch(gbranch(AJMP, T), lab->breakpc);
			break;
		}

		if(breakpc == P) {
			yyerror("break is not in a loop");
			break;
		}
		patch(gbranch(AJMP, T), breakpc);
		break;

	case OCONTINUE:
		if(n->left != N) {
			lab = findlab(n->left->sym);
			if(lab == L || lab->continpc == P) {
				yyerror("continue label is not defined: %S", n->left->sym);
				break;
			}
			patch(gbranch(AJMP, T), lab->continpc);
			break;
		}

		if(continpc == P) {
			yyerror("gen: continue is not in a loop");
			break;
		}
		patch(gbranch(AJMP, T), continpc);
		break;

	case OFOR:
		p1 = gbranch(AJMP, T);			// 		goto test
		sbreak = breakpc;
		breakpc = gbranch(AJMP, T);		// break:	goto done
		scontin = continpc;
		continpc = pc;
		gen(n->nincr, L);				// contin:	incr
		patch(p1, pc);				// test:
		if(n->ntest != N)
			if(n->ntest->ninit != N)
				gen(n->ntest->ninit, L);
		bgen(n->ntest, 0, breakpc);		//		if(!test) goto break
		if(labloop != L) {
			labloop->op = OFOR;
			labloop->continpc = continpc;
			labloop->breakpc = breakpc;
		}
		gen(n->nbody, L);			//		body
		patch(gbranch(AJMP, T), continpc);	//		goto contin
		patch(breakpc, pc);			// done:
		continpc = scontin;
		breakpc = sbreak;
		break;

	case OIF:
		p1 = gbranch(AJMP, T);			//		goto test
		p2 = gbranch(AJMP, T);			// p2:		goto else
		patch(p1, pc);				// test:
		if(n->ntest != N)
			if(n->ntest->ninit != N)
				gen(n->ntest->ninit, L);
		bgen(n->ntest, 0, p2);			// 		if(!test) goto p2
		gen(n->nbody, L);			//		then
		p3 = gbranch(AJMP, T);			//		goto done
		patch(p2, pc);				// else:
		gen(n->nelse, L);			//		else
		patch(p3, pc);				// done:
		break;

	case OSWITCH:
		p1 = gbranch(AJMP, T);			// 		goto test
		sbreak = breakpc;
		breakpc = gbranch(AJMP, T);		// break:	goto done
		patch(p1, pc);				// test:
		if(labloop != L) {
			labloop->op = OFOR;
			labloop->breakpc = breakpc;
		}
		swgen(n);				//		switch(test) body
		patch(breakpc, pc);			// done:
		breakpc = sbreak;
		break;

	case OSELECT:
		sbreak = breakpc;
		p1 = gbranch(AJMP, T);			// 		goto test
		breakpc = gbranch(AJMP, T);		// break:	goto done
		patch(p1, pc);				// test:
		if(labloop != L) {
			labloop->op = OFOR;
			labloop->breakpc = breakpc;
		}
		gen(n->nbody, L);			//		select() body
		patch(breakpc, pc);			// done:
		breakpc = sbreak;
		break;

	case OASOP:
		cgen_asop(n);
		break;

	case OAS:
		cgen_as(n->left, n->right, n->op);
		break;

	case OCALLMETH:
		cgen_callmeth(n, 0);
		break;

	case OCALLINTER:
		cgen_callinter(n, N, 0);
		break;

	case OCALL:
		cgen_call(n, 0);
		break;

	case OPROC:
		cgen_proc(n);
		break;

	case ORETURN:
		cgen_ret(n);
		break;
	}

ret:
	lineno = lno;
}

void
swgen(Node *n)
{
	Node *c1, *c2;
	Node n1, tmp;
	Case *s0, *se, *s;
	Prog *p1, *dflt;
	int32 lno;
	int any;
	Iter save1, save2;

// botch - put most of this code in
// walk. gen binary search for
// sequence of constant cases

	lno = setlineno(n);

	p1 = gbranch(AJMP, T);
	s0 = C;
	se = C;

	// walk thru the body placing breaks
	// and labels into the case statements

	any = 0;
	dflt = P;
	c1 = listfirst(&save1, &n->nbody);
	while(c1 != N) {
		setlineno(c1);
		if(c1->op != OCASE) {
			if(s0 == C && dflt == P)
				yyerror("unreachable statements in a switch");
			gen(c1, L);

			any = 1;
			if(c1->op == OFALL)
				any = 0;
			c1 = listnext(&save1);
			continue;
		}

		// put in the break between cases
		if(any)
			patch(gbranch(AJMP, T), breakpc);
		any = 1;

		// over case expressions
		c2 = listfirst(&save2, &c1->left);
		if(c2 == N)
			dflt = pc;

		while(c2 != N) {
			s = mal(sizeof(*s));
			if(s0 == C)
				s0 = s;
			else
				se->slink = s;
			se = s;

			s->scase = c2;		// case expression
			s->sprog = pc;		// where to go

			c2 = listnext(&save2);
		}

		c1 = listnext(&save1);
	}

	lineno = lno;

	if(any)
		patch(gbranch(AJMP, T), breakpc);

	patch(p1, pc);

	if(n->ntest != N)
		if(n->ntest->ninit != N)
			gen(n->ntest->ninit, L);
	tempname(&tmp, n->ntest->type);
	cgen(n->ntest, &tmp);

	for(s=s0; s!=C; s=s->slink) {
		setlineno(s->scase);
		memset(&n1, 0, sizeof(n1));
		n1.op = OEQ;
		n1.left = &tmp;
		n1.right = s->scase;
		walktype(&n1, Erv);
		bgen(&n1, 1, s->sprog);
	}
	if(dflt != P) {
		patch(gbranch(AJMP, T), dflt);
		goto ret;
	}
	patch(gbranch(AJMP, T), breakpc);

ret:
	lineno = lno;
}

void
inarggen(void)
{
	fatal("inarggen");
}

void
genpanic(void)
{
	Node n1, n2;
	Prog *p;

	nodconst(&n1, types[TINT64], 0xf0);
	nodreg(&n2, types[TINT64], D_AX);
	gins(AMOVL, &n1, &n2);
	p = pc;
	gins(AMOVQ, &n2, N);
	p->to.type = D_INDIR+D_AX;
}

int
argsize(Type *t)
{
	Iter save;
	Type *fp;
	int w, x;

	w = 0;

	fp = structfirst(&save, getoutarg(t));
	while(fp != T) {
		x = fp->width + fp->type->width;
		if(x > w)
			w = x;
		fp = structnext(&save);
	}

	fp = funcfirst(&save, t);
	while(fp != T) {
		x = fp->width + fp->type->width;
		if(x > w)
			w = x;
		fp = funcnext(&save);
	}

	w = (w+7) & ~7;
	return w;
}

void
ginscall(Node *f, int proc)
{
	Node reg, con;

	if(proc) {
		nodreg(&reg, types[TINT64], D_AX);
		if(f->op != OREGISTER) {
			gins(ALEAQ, f, &reg);
			gins(APUSHQ, &reg, N);
		} else
			gins(APUSHQ, f, N);
		nodconst(&con, types[TINT32], argsize(f->type));
		gins(APUSHQ, &con, N);
		gins(ACALL, N, newproc);
		gins(APOPQ, N, &reg);
		gins(APOPQ, N, &reg);
		return;
	}
	gins(ACALL, N, f);
}

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

	gen(n->right, L);		// args

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
	ginscall(&nodr, proc);

	regfree(&nodr);
	regfree(&nodo);

	setmaxarg(n->left->type);
}

void
cgen_callmeth(Node *n, int proc)
{
	Node *l;

	// generate a rewrite for method call
	// (p.f)(...) goes to (f)(p,...)

	l = n->left;
	if(l->op != ODOTMETH)
		fatal("cgen_callmeth: not dotmethod: %N");

	n->op = OCALL;
	n->left = n->left->right;
	n->left->type = l->type;

	if(n->left->op == ONAME)
		n->left->class = PEXTERN;
	cgen_call(n, proc);
}

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
		if(isptr[n->left->type->etype])
			cgen(n->left, &afun);
		else
			agen(n->left, &afun);
	}

	gen(n->right, L);	// assign the args
	t = n->left->type;
	if(isptr[t->etype])
		t = t->type;

	setmaxarg(t);

	// call tempname pointer
	if(n->left->ullman >= UINF) {
		regalloc(&nod, types[tptr], N);
		cgen_as(&nod, &afun, 0);
		nod.type = t;
		ginscall(&nod, proc);
		regfree(&nod);
		goto ret;
	}

	// call pointer
	if(isptr[n->left->type->etype]) {
		regalloc(&nod, types[tptr], N);
		cgen_as(&nod, n->left, 0);
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

void
cgen_proc(Node *n)
{
	switch(n->left->op) {
	default:
		fatal("cgen_proc: unknown call %O", n->left->op);

	case OCALLMETH:
		cgen_callmeth(n->left, 1);
		break;

	case OCALLINTER:
		cgen_callinter(n->left, N, 1);
		break;

	case OCALL:
		cgen_call(n->left, 1);
		break;
	}

}

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
	cgen_as(res, &nod, 0);
}

void
cgen_aret(Node *n, Node *res)
{
	Node nod1;
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

	gins(ALEAQ, &nod1, res);
}

void
cgen_ret(Node *n)
{
	gen(n->left, L);	// copy out args
	gins(ARET, N, N);
}

void
cgen_asop(Node *n)
{
	Node n1, n2, n3, n4;
	Node *nl, *nr;

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

void
cgen_as(Node *nl, Node *nr, int op)
{
	Node nc, n1;
	Type *tl;
	uint32 w, c;

	if(nl == N)
		return;

	if(nr == N || isnil(nr)) {
		if(nl->op == OLIST) {
			cgen_as(nl->left, nr, op);
			cgen_as(nl->right, nr, op);
			return;
		}
		tl = nl->type;
		if(tl == T)
			return;
		if(isfat(tl)) {
			/* clear a fat object */
			if(debug['g'])
				dump("\nclearfat", nl);

			if(nl->type->width < 0)
				fatal("clearfat %T %lld", nl->type, nl->type->width);
			w = nl->type->width;

			if(w > 0)
				gconreg(AMOVQ, 0, D_AX);

			if(w > 0) {
				nodreg(&n1, types[tptr], D_DI);
				agen(nl, &n1);
				gins(ACLD, N, N);	// clear direction flag
			}

			c = w / 8;
			if(c > 0) {
				gconreg(AMOVQ, c, D_CX);
				gins(AREP, N, N);	// repeat
				gins(ASTOSQ, N, N);	// STOQ AL,*(DI)+
			}

			c = w % 8;
			if(c > 0) {
				gconreg(AMOVQ, c, D_CX);
				gins(AREP, N, N);	// repeat
				gins(ASTOSB, N, N);	// STOB AL,*(DI)+
			}
			goto ret;
		}

		/* invent a "zero" for the rhs */
		nr = &nc;
		memset(nr, 0, sizeof(*nr));
		switch(tl->etype) {
		default:
			fatal("cgen_as: tl %T", tl);
			break;

		case TINT:
		case TUINT:
		case TINT8:
		case TUINT8:
		case TINT16:
		case TUINT16:
		case TINT32:
		case TUINT32:
		case TINT64:
		case TUINT64:
		case TUINTPTR:
			nr->val.u.xval = mal(sizeof(*nr->val.u.xval));
			mpmovecfix(nr->val.u.xval, 0);
			nr->val.ctype = CTINT;
			break;

		case TFLOAT:
		case TFLOAT32:
		case TFLOAT64:
		case TFLOAT80:
			nr->val.u.fval = mal(sizeof(*nr->val.u.fval));
			mpmovecflt(nr->val.u.fval, 0.0);
			nr->val.ctype = CTFLT;
			break;

		case TBOOL:
			nr->val.u.bval = 0;
			nr->val.ctype = CTBOOL;
			break;

		case TPTR32:
		case TPTR64:
			nr->val.ctype = CTNIL;
			break;

		}
		nr->op = OLITERAL;
		nr->type = tl;
		nr->addable = 1;
		ullmancalc(nr);
	}

	tl = nl->type;
	if(tl == T)
		return;

	cgen(nr, nl);

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

void
dodiv(int op, Node *nl, Node *nr, Node *res, Node *ax, Node *dx)
{
	int a;
	Node n3;
	Type *t;

	t = nl->type;
	if(t->width == 1) {
		if(issigned[t->etype])
			t = types[TINT32];
		else
			t = types[TUINT32];
	}
	a = optoas(op, t);

	if(!issigned[t->etype]) {
		nodconst(&n3, t, 0);
		gmove(&n3, dx);
	}

	regalloc(&n3, nr->type, N);
	if(nl->ullman >= nr->ullman) {
		cgen(nl, ax);
		if(issigned[t->etype])
			gins(optoas(OFOR, t), N, N);
		cgen(nr, &n3);
		gins(a, &n3, N);
	} else {
		cgen(nr, &n3);
		cgen(nl, ax);
		if(issigned[t->etype])
			gins(optoas(OFOR, t), N, N);
		gins(a, &n3, N);
	}
	regfree(&n3);

	if(op == ODIV)
		gmove(ax, res);
	else
		gmove(dx, res);
}

/*
 * this is hard because divide
 * is done in a fixed numerator
 * of combined DX:AX registers
 */
void
cgen_div(int op, Node *nl, Node *nr, Node *res)
{
	Node ax, dx, n3, tmpax, tmpdx;
	int rax, rdx;

	rax = reg[D_AX];
	rdx = reg[D_DX];

	nodreg(&ax, types[TINT64], D_AX);
	nodreg(&dx, types[TINT64], D_DX);
	regalloc(&ax, nl->type, &ax);
	regalloc(&dx, nl->type, &dx);

	// clean out the AX register
	if(rax && !samereg(res, &ax)) {
		if(rdx && !samereg(res, &dx)) {
			regalloc(&tmpdx, types[TINT64], N);
			regalloc(&tmpax, types[TINT64], N);
			regalloc(&n3, nl->type, N);		// dest for div

			gins(AMOVQ, &dx, &tmpdx);
			gins(AMOVQ, &ax, &tmpax);
			dodiv(op, nl, nr, &n3, &ax, &dx);
			gins(AMOVQ, &tmpax, &ax);
			gins(AMOVQ, &tmpdx, &dx);
			gmove(&n3, res);

			regfree(&tmpdx);
			regfree(&tmpax);
			regfree(&n3);
			goto ret;
		}
		regalloc(&tmpax, types[TINT64], N);
		regalloc(&n3, nl->type, N);		// dest for div

		gins(AMOVQ, &ax, &tmpax);
		dodiv(op, nl, nr, &n3, &ax, &dx);
		gins(AMOVQ, &tmpax, &ax);
		gmove(&n3, res);

		regfree(&tmpax);
		regfree(&n3);
		goto ret;
	}

	// clean out the DX register
	if(rdx && !samereg(res, &dx)) {
		regalloc(&tmpdx, types[TINT64], N);
		regalloc(&n3, nl->type, N);		// dest for div

		gins(AMOVQ, &dx, &tmpdx);
		dodiv(op, nl, nr, &n3, &ax, &dx);
		gins(AMOVQ, &tmpdx, &dx);
		gmove(&n3, res);

		regfree(&tmpdx);
		regfree(&n3);
		goto ret;
	}
	dodiv(op, nl, nr, res, &ax, &dx);

ret:
	regfree(&ax);
	regfree(&dx);
}

/*
 * this is hard because shift
 * count is either constant
 * or the CL register
 */
void
cgen_shift(int op, Node *nl, Node *nr, Node *res)
{
	Node n1, n2, n3;
	int a, rcl;
	Prog *p1;

	a = optoas(op, nl->type);

	if(nr->op == OLITERAL) {
		regalloc(&n1, nl->type, res);
		cgen(nl, &n1);
		gins(a, nr, &n1);
		gmove(&n1, res);
		regfree(&n1);
		goto ret;
	}

	rcl = reg[D_CX];

	nodreg(&n1, types[TINT64], D_CX);
	regalloc(&n1, nr->type, &n1);

	// clean out the CL register
	if(rcl) {
		regalloc(&n2, types[TINT64], N);
		gins(AMOVQ, &n1, &n2);
		regfree(&n1);

		reg[D_CX] = 0;
		if(samereg(res, &n1))
			cgen_shift(op, nl, nr, &n2);
		else
			cgen_shift(op, nl, nr, res);
		reg[D_CX] = rcl;

		gins(AMOVQ, &n2, &n1);
		regfree(&n2);
		goto ret;
	}

	regalloc(&n2, nl->type, res);	// can one shift the CL register
	if(nl->ullman >= nr->ullman) {
		cgen(nl, &n2);
		cgen(nr, &n1);
	} else {
		cgen(nr, &n1);
		cgen(nl, &n2);
	}
	// test and fix up large shifts
	nodconst(&n3, types[TUINT32], nl->type->width*8);
	gins(optoas(OCMP, types[TUINT32]), &n1, &n3);
	p1 = gbranch(optoas(OLT, types[TUINT32]), T);
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

void
checklabels(void)
{
	Label *l, *m;
	Sym *s;

//	// print the label list
//	for(l=labellist; l!=L; l=l->link) {
//		print("lab %O %S\n", l->op, l->sym);
//	}

	for(l=labellist; l!=L; l=l->link) {
	switch(l->op) {
		case OFOR:
		case OLABEL:
			// these are definitions -
			s = l->sym;
			for(m=labellist; m!=L; m=m->link) {
				if(m->sym != s)
					continue;
				switch(m->op) {
				case OFOR:
				case OLABEL:
					// these are definitions -
					// look for redefinitions
					if(l != m)
						yyerror("label %S redefined", s);
					break;
				case OGOTO:
					// these are references -
					// patch to definition
					patch(m->label, l->label);
					m->sym = S;	// mark done
					break;
				}
			}
		}
	}

	// diagnostic for all undefined references
	for(l=labellist; l!=L; l=l->link)
		if(l->op == OGOTO && l->sym != S)
			yyerror("label %S not defined", l->sym);
}

Label*
findlab(Sym *s)
{
	Label *l;

	for(l=labellist; l!=L; l=l->link) {
		if(l->sym != s)
			continue;
		if(l->op != OFOR)
			continue;
		return l;
	}
	return L;
}
