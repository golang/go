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
		if (p->from.name == D_AUTO && p->from.node)
			p->from.node->used++;

		if (p->to.name == D_AUTO && p->to.node)
			p->to.node->used++;
	}
}

// Fixup instructions after compactframe has moved all autos around.
void
fixautoused(Prog* p)
{
	for (; p; p = p->link) {
		if (p->from.name == D_AUTO && p->from.node)
			p->from.offset += p->from.node->stkdelta;

		if (p->to.name == D_AUTO && p->to.node)
			p->to.offset += p->to.node->stkdelta;
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
	Node n1, r, con;

	switch(proc) {
	default:
		fatal("ginscall: bad proc %d", proc);
		break;

	case 0:	// normal call
		p = gins(ABL, N, f);
		afunclit(&p->to);
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
			patch(gbranch(ABNE, T), retpc);
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
			regalloc(&n3, nr->type, N);
			cgen(nr, &n3);
			regalloc(&n2, nl->type, N);
			cgen(nl, &n2);
			gins(a, &n3, &n2);
			cgen(&n2, nl);
			regfree(&n2);
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
	if(nr->ullman >= nl->ullman || nl->addable) {
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
 * generate shift according to op, one of:
 *	res = nl << nr
 *	res = nl >> nr
 */
void
cgen_shift(int op, Node *nl, Node *nr, Node *res)
{
	Node n1, n2, n3, nt, t, lo, hi;
	int w;
	Prog *p1, *p2, *p3;
	Type *tr;
	uvlong sc;

	if(nl->type->width > 4)
		fatal("cgen_shift %T", nl->type);

	w = nl->type->width * 8;

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
	p3 = gbranch(ABEQ, T);

	// test and fix up large shifts
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
		patch(gbranch(ABNE, T), pl);

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

static int
regcmp(const void *va, const void *vb)
{
	Node *ra, *rb;

	ra = (Node*)va;
	rb = (Node*)vb;
	return ra->local - rb->local;
}

static	Prog*	throwpc;

// We're only going to bother inlining if we can
// convert all the arguments to 32 bits safely.  Can we?
static int
fix64(NodeList *nn, int n)
{
	NodeList *l;
	Node *r;
	int i;
	
	l = nn;
	for(i=0; i<n; i++) {
		r = l->n->right;
		if(is64(r->type) && !smallintconst(r)) {
			if(r->op == OCONV)
				r = r->left;
			if(is64(r->type))
				return 0;
		}
		l = l->next;
	}
	return 1;
}

void
getargs(NodeList *nn, Node *reg, int n)
{
	NodeList *l;
	int i;

	throwpc = nil;

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
	vlong cl;
	Prog *p1;
	int op;
	Node *c, n1, n2;

	op = OLE;
	if(smallintconst(nl)) {
		cl = mpgetfix(nl->val.u.xval);
		if(cl == 0)
			return;
		if(smallintconst(nr))
			return;

		// put the constant on the right
		op = brrev(op);
		c = nl;
		nl = nr;
		nr = c;
	}

	n1.op = OXXX;
	if(nr->op != OREGISTER) {
		regalloc(&n1, types[TUINT32], N);
		gmove(nr, &n1);
		nr = &n1;
	}
	n2.op = OXXX;
	if(nl->op != OREGISTER) {
		regalloc(&n2, types[TUINT32], N);
		gmove(nl, &n2);
		nl = &n2;
	}
	gcmp(optoas(OCMP, types[TUINT32]), nl, nr);
	if(nr == &n1)
		regfree(&n1);
	if(nl == &n2)
		regfree(&n2);
	if(throwpc == nil) {
		p1 = gbranch(optoas(op, types[TUINT32]), T);
		throwpc = pc;
		ginscall(panicslice, 0);
		patch(p1, pc);
	} else {
		op = brcom(op);
		p1 = gbranch(optoas(op, types[TUINT32]), T);
		patch(p1, throwpc);
	}
}

int
sleasy(Node *n)
{
	if(n->op != ONAME)
		return 0;
	if(!n->addable)
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
	Node nodes[5];
	Node n1, n2, n3, nres, ntemp;
	vlong v;
	int i, narg;

	if(n->op != OCALLFUNC)
		goto no;
	if(!n->left->addable)
		goto no;
	if(n->left->sym == S)
		goto no;
	if(n->left->sym->pkg != runtimepkg)
		goto no;
	if(strcmp(n->left->sym->name, "slicearray") == 0)
		goto slicearray;
	if(strcmp(n->left->sym->name, "sliceslice") == 0) {
		narg = 4;
		goto sliceslice;
	}
	if(strcmp(n->left->sym->name, "sliceslice1") == 0) {
		narg = 3;
		goto sliceslice;
	}
	goto no;

slicearray:
	if(!sleasy(res))
		goto no;
	if(!fix64(n->list, 5))
		goto no;
	getargs(n->list, nodes, 5);

	// if(hb[3] > nel[1]) goto throw
	cmpandthrow(&nodes[3], &nodes[1]);

	// if(lb[2] > hb[3]) goto throw
	cmpandthrow(&nodes[2], &nodes[3]);

	// len = hb[3] - lb[2] (destroys hb)
	n2 = *res;
	n2.type = types[TUINT32];
	n2.xoffset += Array_nel;

	if(smallintconst(&nodes[3]) && smallintconst(&nodes[2])) {
		v = mpgetfix(nodes[3].val.u.xval) -
			mpgetfix(nodes[2].val.u.xval);
		nodconst(&n1, types[TUINT32], v);
		gmove(&n1, &n2);
	} else {
		regalloc(&n1, types[TUINT32], &nodes[3]);
		gmove(&nodes[3], &n1);
		if(!smallintconst(&nodes[2]) || mpgetfix(nodes[2].val.u.xval) != 0)
			gins(optoas(OSUB, types[TUINT32]), &nodes[2], &n1);
		gmove(&n1, &n2);
		regfree(&n1);
	}

	// cap = nel[1] - lb[2] (destroys nel)
	n2 = *res;
	n2.type = types[TUINT32];
	n2.xoffset += Array_cap;

	if(smallintconst(&nodes[1]) && smallintconst(&nodes[2])) {
		v = mpgetfix(nodes[1].val.u.xval) -
			mpgetfix(nodes[2].val.u.xval);
		nodconst(&n1, types[TUINT32], v);
		gmove(&n1, &n2);
	} else {
		regalloc(&n1, types[TUINT32], &nodes[1]);
		gmove(&nodes[1], &n1);
		if(!smallintconst(&nodes[2]) || mpgetfix(nodes[2].val.u.xval) != 0)
			gins(optoas(OSUB, types[TUINT32]), &nodes[2], &n1);
		gmove(&n1, &n2);
		regfree(&n1);
	}

	// if slice could be too big, dereference to
	// catch nil array pointer.
	if(nodes[0].op == OREGISTER && nodes[0].type->type->width >= unmappedzero) {
		n2 = nodes[0];
		n2.xoffset = 0;
		n2.op = OINDREG;
		n2.type = types[TUINT8];
		regalloc(&n1, types[TUINT32], N);
		gins(AMOVB, &n2, &n1);
		regfree(&n1);
	}

	// ary = old[0] + (lb[2] * width[4]) (destroys old)
	n2 = *res;
	n2.type = types[tptr];
	n2.xoffset += Array_array;

	if(smallintconst(&nodes[2]) && smallintconst(&nodes[4])) {
		v = mpgetfix(nodes[2].val.u.xval) *
			mpgetfix(nodes[4].val.u.xval);
		if(v != 0) {
			nodconst(&n1, types[tptr], v);
			gins(optoas(OADD, types[tptr]), &n1, &nodes[0]);
		}
	} else {
		regalloc(&n1, types[tptr], &nodes[2]);
		gmove(&nodes[2], &n1);
		if(!smallintconst(&nodes[4]) || mpgetfix(nodes[4].val.u.xval) != 1) {
			regalloc(&n3, types[tptr], N);
			gmove(&nodes[4], &n3);
			gins(optoas(OMUL, types[tptr]), &n3, &n1);
			regfree(&n3);
		}
		gins(optoas(OADD, types[tptr]), &n1, &nodes[0]);
		regfree(&n1);
	}
	gmove(&nodes[0], &n2);

	for(i=0; i<5; i++) {
		if(nodes[i].op == OREGISTER)
			regfree(&nodes[i]);
	}
	return 1;

sliceslice:
	if(!fix64(n->list, narg))
		goto no;
	ntemp.op = OXXX;
	if(!sleasy(n->list->n->right)) {
		Node *n0;
		
		n0 = n->list->n->right;
		tempname(&ntemp, res->type);
		cgen(n0, &ntemp);
		n->list->n->right = &ntemp;
		getargs(n->list, nodes, narg);
		n->list->n->right = n0;
	} else
		getargs(n->list, nodes, narg);

	nres = *res;		// result
	if(!sleasy(res)) {
		if(ntemp.op == OXXX)
			tempname(&ntemp, res->type);
		nres = ntemp;
	}

	if(narg == 3) {	// old[lb:]
		// move width to where it would be for old[lb:hb]
		nodes[3] = nodes[2];
		nodes[2].op = OXXX;
		
		// if(lb[1] > old.nel[0]) goto throw;
		n2 = nodes[0];
		n2.xoffset += Array_nel;
		n2.type = types[TUINT32];
		cmpandthrow(&nodes[1], &n2);

		// ret.nel = old.nel[0]-lb[1];
		n2 = nodes[0];
		n2.type = types[TUINT32];
		n2.xoffset += Array_nel;
	
		regalloc(&n1, types[TUINT32], N);
		gmove(&n2, &n1);
		if(!smallintconst(&nodes[1]) || mpgetfix(nodes[1].val.u.xval) != 0)
			gins(optoas(OSUB, types[TUINT32]), &nodes[1], &n1);
	
		n2 = nres;
		n2.type = types[TUINT32];
		n2.xoffset += Array_nel;
		gmove(&n1, &n2);
		regfree(&n1);
	} else {	// old[lb:hb]
		// if(hb[2] > old.cap[0]) goto throw;
		n2 = nodes[0];
		n2.xoffset += Array_cap;
		n2.type = types[TUINT32];
		cmpandthrow(&nodes[2], &n2);

		// if(lb[1] > hb[2]) goto throw;
		cmpandthrow(&nodes[1], &nodes[2]);

		// ret.len = hb[2]-lb[1]; (destroys hb[2])
		n2 = nres;
		n2.type = types[TUINT32];
		n2.xoffset += Array_nel;
	
		if(smallintconst(&nodes[2]) && smallintconst(&nodes[1])) {
			v = mpgetfix(nodes[2].val.u.xval) -
				mpgetfix(nodes[1].val.u.xval);
			nodconst(&n1, types[TUINT32], v);
			gmove(&n1, &n2);
		} else {
			regalloc(&n1, types[TUINT32], &nodes[2]);
			gmove(&nodes[2], &n1);
			if(!smallintconst(&nodes[1]) || mpgetfix(nodes[1].val.u.xval) != 0)
				gins(optoas(OSUB, types[TUINT32]), &nodes[1], &n1);
			gmove(&n1, &n2);
			regfree(&n1);
		}
	}

	// ret.cap = old.cap[0]-lb[1]; (uses hb[2])
	n2 = nodes[0];
	n2.type = types[TUINT32];
	n2.xoffset += Array_cap;

	regalloc(&n1, types[TUINT32], &nodes[2]);
	gmove(&n2, &n1);
	if(!smallintconst(&nodes[1]) || mpgetfix(nodes[1].val.u.xval) != 0)
		gins(optoas(OSUB, types[TUINT32]), &nodes[1], &n1);

	n2 = nres;
	n2.type = types[TUINT32];
	n2.xoffset += Array_cap;
	gmove(&n1, &n2);
	regfree(&n1);

	// ret.array = old.array[0]+lb[1]*width[3]; (uses lb[1])
	n2 = nodes[0];
	n2.type = types[tptr];
	n2.xoffset += Array_array;
	regalloc(&n3, types[tptr], N);
	gmove(&n2, &n3);

	regalloc(&n1, types[tptr], &nodes[1]);
	if(smallintconst(&nodes[1]) && smallintconst(&nodes[3])) {
		gmove(&n2, &n1);
		v = mpgetfix(nodes[1].val.u.xval) *
			mpgetfix(nodes[3].val.u.xval);
		if(v != 0) {
			nodconst(&n2, types[tptr], v);
			gins(optoas(OADD, types[tptr]), &n3, &n1);
		}
	} else {
		gmove(&nodes[1], &n1);
		if(!smallintconst(&nodes[3]) || mpgetfix(nodes[3].val.u.xval) != 1) {
			regalloc(&n2, types[tptr], N);
			gmove(&nodes[3], &n2);
			gins(optoas(OMUL, types[tptr]), &n2, &n1);
			regfree(&n2);
		}
		gins(optoas(OADD, types[tptr]), &n3, &n1);
	}
	regfree(&n3);

	n2 = nres;
	n2.type = types[tptr];
	n2.xoffset += Array_array;
	gmove(&n1, &n2);
	regfree(&n1);

	for(i=0; i<4; i++) {
		if(nodes[i].op == OREGISTER)
			regfree(&nodes[i]);
	}

	if(!sleasy(res)) {
		cgen(&nres, res);
	}
	return 1;

no:
	return 0;
}
