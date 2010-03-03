// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "gg.h"

static void	subnode(Node *nr, Node *ni, Node *nc);
static void	negate(Node *n);
static void	zero(Node *n);
static int	isimag1i(Node*);

#define	CASE(a,b)	(((a)<<16)|((b)<<0))

/*
 * generate:
 *	res = n;
 * simplifies and calls gmove.
 * perm is
 *	0 (r,i) -> (r,i)
 *	1 (r,i) -> (-i,r)   *1i
 *	2 (r,i) -> (i,-r)   /1i
 */
void
complexmove(Node *f, Node *t, int perm)
{
	int ft, tt;
	Node n1, n2, n3, n4, nc;

	if(debug['g']) {
		dump("\ncomplex-f", f);
		dump("complex-t", t);
	}

	if(!t->addable)
		fatal("to no addable");

	ft = simsimtype(f->type);
	tt = simsimtype(t->type);
	switch(CASE(ft,tt)) {

	default:
		fatal("complexmove: unknown conversion: %T -> %T\n",
			f->type, t->type);

	case CASE(TCOMPLEX64,TCOMPLEX64):
	case CASE(TCOMPLEX64,TCOMPLEX128):
	case CASE(TCOMPLEX128,TCOMPLEX64):
	case CASE(TCOMPLEX128,TCOMPLEX128):
		// complex to complex move/convert
		// make from addable
		if(!f->addable) {
			tempname(&n1, f->type);
			complexmove(f, &n1, 0);
			f = &n1;
		}

		subnode(&n1, &n2, f);
		subnode(&n3, &n4, t);

		// perform the permutations.
		switch(perm) {
		case 0:	// r,i => r,i
			gmove(&n1, &n3);
			gmove(&n2, &n4);
			break;
		case 1: // r,i => -i,r
			regalloc(&nc, n3.type, N);
			gmove(&n2, &nc);
			negate(&nc);
			gmove(&n1, &n4);
			gmove(&nc, &n3);
			regfree(&nc);
			break;
		case 2: // r,i => i,-r
			regalloc(&nc, n4.type, N);
			gmove(&n1, &nc);
			negate(&nc);
			gmove(&n2, &n3);
			gmove(&nc, &n4);
			regfree(&nc);
			break;
		}
		break;

	case CASE(TFLOAT32,TCOMPLEX64):
	case CASE(TFLOAT32,TCOMPLEX128):
	case CASE(TFLOAT64,TCOMPLEX64):
	case CASE(TFLOAT64,TCOMPLEX128):
		// float to complex goes to real part

		regalloc(&n1, types[ft], N);
		cgen(f, &n1);
		subnode(&n3, &n4, t);

		// perform the permutations.
		switch(perm) {
		case 0:	// no permutations
			gmove(&n1, &n3);
			zero(&n4);
			break;
		case 1:
			gmove(&n1, &n4);
			zero(&n3);
			break;
		case 2:
			negate(&n1);
			gmove(&n1, &n4);
			zero(&n3);
			break;
		}
		regfree(&n1);
		break;
	}
}

int
complexop(Node *n, Node *res)
{
	if(n != N && n->type != T)
	if(iscomplex[n->type->etype]) {
		switch(n->op) {
		case OCONV:
		case OADD:
		case OSUB:
		case OMUL:
		case ODIV:
		case OMINUS:
			goto yes;
		}
//dump("complexop no", n);
	}
	return 0;

yes:
	return 1;
}

void
complexgen(Node *n, Node *res)
{
	Node *nl, *nr;
	Node n1, n2, n3, n4, n5, n6;
	Node ra, rb, rc, rd;
	int tl, tr;

	if(debug['g']) {
		dump("\ncomplex-n", n);
		dump("complex-res", res);
	}

	// perform conversion from n to res
	tl = simsimtype(res->type);
	tl = cplxsubtype(tl);
	tr = simsimtype(n->type);
	tr = cplxsubtype(tr);
	if(tl != tr) {
		if(!n->addable) {
			tempname(&n1, n->type);
			complexmove(n, &n1, 0);
			n = &n1;
		}
		complexmove(n, res, 0);
		return;
	}

	nl = n->left;
	if(nl == N)
		return;
	nr = n->right;

	// make both sides addable in ullman order
	if(nr != N) {
		if(nl->ullman > nr->ullman && !nl->addable) {
			tempname(&n1, nl->type);
			complexgen(nl, &n1);
			nl = &n1;
		}
		if(!nr->addable) {
			tempname(&n2, nr->type);
			complexgen(nr, &n2);
			nr = &n2;
		}
	}
	if(!nl->addable) {
		tempname(&n1, nl->type);
		complexgen(nl, &n1);
		nl = &n1;
	}

	switch(n->op) {
	default:
		fatal("opcode %O", n->op);
		break;

	case OCONV:
		complexmove(nl, res, 0);
		break;

	case OMINUS:
		subnode(&n1, &n2, nl);
		subnode(&n5, &n6, res);

		regalloc(&ra, n5.type, N);
		gmove(&n1, &ra);
		negate(&ra);
		gmove(&ra, &n5);
		regfree(&ra);

		regalloc(&ra, n5.type, N);
		gmove(&n2, &ra);
		negate(&ra);
		gmove(&ra, &n6);
		regfree(&ra);
		break;

	case OADD:
	case OSUB:

		subnode(&n1, &n2, nl);
		subnode(&n3, &n4, nr);
		subnode(&n5, &n6, res);

		regalloc(&ra, n5.type, N);
		gmove(&n1, &ra);
		gins(optoas(n->op, n5.type), &n3, &ra);
		gmove(&ra, &n5);
		regfree(&ra);

		regalloc(&ra, n6.type, N);
		gmove(&n2, &ra);
		gins(optoas(n->op, n6.type), &n4, &ra);
		gmove(&ra, &n6);
		regfree(&ra);
		break;

	case OMUL:
		if(isimag1i(nr)) {
			complexmove(nl, res, 1);
			break;
		}
		if(isimag1i(nl)) {
			complexmove(nr, res, 1);
			break;
		}

		subnode(&n1, &n2, nl);
		subnode(&n3, &n4, nr);
		subnode(&n5, &n6, res);

		regalloc(&ra, n5.type, N);
		regalloc(&rb, n5.type, N);
		regalloc(&rc, n6.type, N);
		regalloc(&rd, n6.type, N);

		gmove(&n1, &ra);
		gmove(&n3, &rc);
		gins(optoas(OMUL, n5.type), &rc, &ra);	// ra = a*c
		
		gmove(&n2, &rb);
		gmove(&n4, &rd);
		gins(optoas(OMUL, n5.type), &rd, &rb);	// rb = b*d
		gins(optoas(OSUB, n5.type), &rb, &ra);	// ra = (a*c - b*d)

		gins(optoas(OMUL, n5.type), &n2, &rc);	// rc = b*c
		gins(optoas(OMUL, n5.type), &n1, &rd);	// rd = a*d
		gins(optoas(OADD, n5.type), &rd, &rc);	// rc = (b*c + a*d)

		gmove(&ra, &n5);
		gmove(&rc, &n6);

		regfree(&ra);
		regfree(&rb);
		regfree(&rc);
		regfree(&rd);
		break;

	case ODIV:
		if(isimag1i(nr)) {
			complexmove(nl, res, 2);
			break;
		}

		subnode(&n1, &n2, nl);
		subnode(&n3, &n4, nr);
		subnode(&n5, &n6, res);

		regalloc(&ra, n5.type, N);
		regalloc(&rb, n5.type, N);
		regalloc(&rc, n6.type, N);
		regalloc(&rd, n6.type, N);

		gmove(&n1, &ra);
		gmove(&n3, &rc);
		gins(optoas(OMUL, n5.type), &rc, &ra);	// ra = a*c
		
		gmove(&n2, &rb);
		gmove(&n4, &rd);
		gins(optoas(OMUL, n5.type), &rd, &rb);	// rb = b*d
		gins(optoas(OADD, n5.type), &rb, &ra);	// ra = (a*c + b*d)

		gins(optoas(OMUL, n5.type), &n2, &rc);	// rc = b*c
		gins(optoas(OMUL, n5.type), &n1, &rd);	// rd = a*d
		gins(optoas(OSUB, n5.type), &rd, &rc);	// rc = (b*c - a*d)

		gmove(&n3, &rb);
		gins(optoas(OMUL, n5.type), &rb, &rb);	// rb = c*c
		gmove(&n4, &rd);
		gins(optoas(OMUL, n5.type), &rd, &rd);	// rd = d*d
		gins(optoas(OADD, n5.type), &rd, &rb);	// rb = (c*c + d*d)

		gins(optoas(ODIV, n5.type), &rb, &ra);	// ra = (a*c + b*d)/(c*c + d*d)
		gins(optoas(ODIV, n5.type), &rb, &rc);	// rc = (b*c - a*d)/(c*c + d*d)

		gmove(&ra, &n5);
		gmove(&rc, &n6);

		regfree(&ra);
		regfree(&rb);
		regfree(&rc);
		regfree(&rd);
		break;
	}
}

void
complexbool(int op, Node *nl, Node *nr, int true, Prog *to)
{
	Node n1, n2, n3, n4;
	Node na, nb, nc;

	// make both sides addable in ullman order
	if(nr != N) {
		if(nl->ullman > nr->ullman && !nl->addable) {
			tempname(&n1, nl->type);
			complexgen(nl, &n1);
			nl = &n1;
		}
		if(!nr->addable) {
			tempname(&n2, nr->type);
			complexgen(nr, &n2);
			nr = &n2;
		}
	}
	if(!nl->addable) {
		tempname(&n1, nl->type);
		complexgen(nl, &n1);
		nl = &n1;
	}

	// build tree
	// real(l) == real(r) && imag(l) == imag(r)

	subnode(&n1, &n2, nl);
	subnode(&n3, &n4, nr);

	memset(&na, 0, sizeof(na));
	na.op = OANDAND;
	na.left = &nb;
	na.right = &nc;
	na.type = types[TBOOL];

	memset(&nb, 0, sizeof(na));
	nb.op = OEQ;
	nb.left = &n1;
	nb.right = &n3;
	nb.type = types[TBOOL];

	memset(&nc, 0, sizeof(na));
	nc.op = OEQ;
	nc.left = &n2;
	nc.right = &n4;
	nc.type = types[TBOOL];

	if(op == ONE)
		true = !true;

	bgen(&na, true, to);
}

int
cplxsubtype(int et)
{
	if(et == TCOMPLEX64)
		return TFLOAT32;
	if(et == TCOMPLEX128)
		return TFLOAT64;
	fatal("cplxsubtype: %E\n", et);
	return 0;
}

void
nodfconst(Node *n, Type *t, Mpflt* fval)
{
	memset(n, 0, sizeof(*n));
	n->op = OLITERAL;
	n->addable = 1;
	ullmancalc(n);
	n->val.u.fval = fval;
	n->val.ctype = CTFLT;
	n->type = t;

	if(!isfloat[t->etype])
		fatal("nodfconst: bad type %T", t);
}

static int
isimag1i(Node *n)
{
	if(n != N)
	if(n->op == OLITERAL)
	if(n->val.ctype == CTCPLX)
	if(mpgetflt(&n->val.u.cval->real) == 0.0)
	if(mpgetflt(&n->val.u.cval->imag) == 1.0)
		return 1;
	return 0;
}

// break addable nc-complex into nr-real and ni-imaginary
static void
subnode(Node *nr, Node *ni, Node *nc)
{
	int tc;
	Type *t;

	if(!nc->addable)
		fatal("subnode not addable");

	tc = simsimtype(nc->type);
	tc = cplxsubtype(tc);
	t = types[tc];

	if(nc->op == OLITERAL) {
		nodfconst(nr, t, &nc->val.u.cval->real);
		nodfconst(ni, t, &nc->val.u.cval->imag);
		return;
	}

	*nr = *nc;
	nr->type = t;

	*ni = *nc;
	ni->type = t;
	ni->xoffset += t->width;
}

// generate code to negate register nr
static void
negate(Node *nr)
{
	Node nc;
	Mpflt fval;

	memset(&nc, 0, sizeof(nc));
	nc.op = OLITERAL;
	nc.addable = 1;
	ullmancalc(&nc);
	nc.val.u.fval = &fval;
	nc.val.ctype = CTFLT;
	nc.type = nr->type;

	mpmovecflt(nc.val.u.fval, -1.0);
	gins(optoas(OMUL, nr->type), &nc, nr);
}

// generate code to zero addable dest nr
static void
zero(Node *nr)
{
	Node nc;
	Mpflt fval;

	memset(&nc, 0, sizeof(nc));
	nc.op = OLITERAL;
	nc.addable = 1;
	ullmancalc(&nc);
	nc.val.u.fval = &fval;
	nc.val.ctype = CTFLT;
	nc.type = nr->type;

	mpmovecflt(nc.val.u.fval, 0.0);

	gmove(&nc, nr);
}
