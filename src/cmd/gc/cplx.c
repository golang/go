// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "gg.h"

static	void	subnode(Node *nr, Node *ni, Node *nc);
static	void	zero(Node *n);
static	void	minus(Node *nl, Node *res);
	void	complexminus(Node*, Node*);
	void	complexadd(int op, Node*, Node*, Node*);
	void	complexmul(Node*, Node*, Node*);

#define	CASE(a,b)	(((a)<<16)|((b)<<0))

/*
 * generate:
 *	res = n;
 * simplifies and calls gmove.
 */
void
complexmove(Node *f, Node *t)
{
	int ft, tt;
	Node n1, n2, n3, n4;

	if(debug['g']) {
		dump("\ncomplexmove-f", f);
		dump("complexmove-t", t);
	}

	if(!t->addable)
		fatal("complexmove: to not addable");

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
			complexmove(f, &n1);
			f = &n1;
		}

		subnode(&n1, &n2, f);
		subnode(&n3, &n4, t);

		cgen(&n1, &n3);
		cgen(&n2, &n4);
		break;

	// these are depricated
	case CASE(TFLOAT32,TCOMPLEX64):
	case CASE(TFLOAT32,TCOMPLEX128):
	case CASE(TFLOAT64,TCOMPLEX64):
	case CASE(TFLOAT64,TCOMPLEX128):
		// float to complex goes to real part

		subnode(&n1, &n2, t);
		cgen(f, &n1);
		zero(&n2);
		break;
	}
}

int
complexop(Node *n, Node *res)
{
	if(n != N && n->type != T)
	if(iscomplex[n->type->etype]) {
		goto maybe;
	}
	if(res != N && res->type != T)
	if(iscomplex[res->type->etype]) {
		goto maybe;
	}

	if(n->op == OREAL || n->op == OIMAG)
		goto yes;

	goto no;

maybe:
	switch(n->op) {
	case OCONV:	// implemented ops
	case OADD:
	case OSUB:
	case OMUL:
	case OMINUS:
	case OCMPLX:
	case OREAL:
	case OIMAG:
		goto yes;

	case ODOT:
	case ODOTPTR:
	case OINDEX:
	case OIND:
	case ONAME:
		goto yes;
	}

no:
//dump("\ncomplex-no", n);
	return 0;
yes:
//dump("\ncomplex-yes", n);
	return 1;
}

void
complexgen(Node *n, Node *res)
{
	Node *nl, *nr;
	Node tnl, tnr;
	Node n1, n2, tmp;
	int tl, tr;

	if(debug['g']) {
		dump("\ncomplexgen-n", n);
		dump("complexgen-res", res);
	}

	// pick off float/complex opcodes
	switch(n->op) {
	case OCMPLX:
		subnode(&n1, &n2, res);
		tempname(&tmp, n1.type);
		cgen(n->left, &tmp);
		cgen(n->right, &n2);
		cgen(&tmp, &n1);
		return;

	case OREAL:
		subnode(&n1, &n2, n->left);
		cgen(&n1, res);
		return;

	case OIMAG:
		subnode(&n1, &n2, n->left);
		cgen(&n2, res);
		return;
	}

	// perform conversion from n to res
	tl = simsimtype(res->type);
	tl = cplxsubtype(tl);
	tr = simsimtype(n->type);
	tr = cplxsubtype(tr);
	if(tl != tr) {
		if(!n->addable) {
			tempname(&n1, n->type);
			complexmove(n, &n1);
			n = &n1;
		}
		complexmove(n, res);
		return;
	}

	if(!res->addable) {
		igen(res, &n1, N);
		cgen(n, &n1);
		regfree(&n1);
		return;
	}
	if(n->addable) {
		complexmove(n, res);
		return;
	}

	switch(n->op) {
	default:
		dump("complexgen: unknown op", n);
		fatal("complexgen: unknown op %O", n->op);

	case ODOT:
	case ODOTPTR:
	case OINDEX:
	case OIND:
	case ONAME:	// PHEAP or PPARAMREF var
		igen(n, &n1, res);
		complexmove(&n1, res);
		regfree(&n1);
		return;

	case OCONV:
	case OADD:
	case OSUB:
	case OMUL:
	case OMINUS:
	case OCMPLX:
	case OREAL:
	case OIMAG:
		break;
	}

	nl = n->left;
	if(nl == N)
		return;
	nr = n->right;

	// make both sides addable in ullman order
	if(nr != N) {
		if(nl->ullman > nr->ullman && !nl->addable) {
			tempname(&tnl, nl->type);
			cgen(nl, &tnl);
			nl = &tnl;
		}
		if(!nr->addable) {
			tempname(&tnr, nr->type);
			cgen(nr, &tnr);
			nr = &tnr;
		}
	}
	if(!nl->addable) {
		tempname(&tnl, nl->type);
		cgen(nl, &tnl);
		nl = &tnl;
	}

	switch(n->op) {
	default:
		fatal("complexgen: unknown op %O", n->op);
		break;

	case OCONV:
		complexmove(nl, res);
		break;

	case OMINUS:
		complexminus(nl, res);
		break;

	case OADD:
	case OSUB:
		complexadd(n->op, nl, nr, res);
		break;

	case OMUL:
		complexmul(nl, nr, res);
		break;
	// ODIV call a runtime function
	}
}

void
complexbool(int op, Node *nl, Node *nr, int true, Prog *to)
{
	Node tnl, tnr;
	Node n1, n2, n3, n4;
	Node na, nb, nc;

	// make both sides addable in ullman order
	if(nr != N) {
		if(nl->ullman > nr->ullman && !nl->addable) {
			tempname(&tnl, nl->type);
			cgen(nl, &tnl);
			nl = &tnl;
		}
		if(!nr->addable) {
			tempname(&tnr, nr->type);
			cgen(nr, &tnr);
			nr = &tnr;
		}
	}
	if(!nl->addable) {
		tempname(&tnl, nl->type);
		cgen(nl, &tnl);
		nl = &tnl;
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

	cgen(&nc, nr);
}

// generate code res = -nl
static void
minus(Node *nl, Node *res)
{
	Node ra;

	memset(&ra, 0, sizeof(ra));
	ra.op = OMINUS;
	ra.left = nl;
	ra.type = nl->type;
	cgen(&ra, res);
}

// build and execute tree
//	real(res) = -real(nl)
//	imag(res) = -imag(nl)
void
complexminus(Node *nl, Node *res)
{
	Node n1, n2, n5, n6;

	subnode(&n1, &n2, nl);
	subnode(&n5, &n6, res);

	minus(&n1, &n5);
	minus(&n2, &n6);
}


// build and execute tree
//	real(res) = real(nl) op real(nr)
//	imag(res) = imag(nl) op imag(nr)
void
complexadd(int op, Node *nl, Node *nr, Node *res)
{
	Node n1, n2, n3, n4, n5, n6;
	Node ra;

	subnode(&n1, &n2, nl);
	subnode(&n3, &n4, nr);
	subnode(&n5, &n6, res);

	memset(&ra, 0, sizeof(ra));
	ra.op = op;
	ra.left = &n1;
	ra.right = &n3;
	ra.type = n1.type;
	cgen(&ra, &n5);

	memset(&ra, 0, sizeof(ra));
	ra.op = op;
	ra.left = &n2;
	ra.right = &n4;
	ra.type = n2.type;
	cgen(&ra, &n6);
}

// build and execute tree
//	tmp       = real(nl)*real(nr) - imag(nl)*imag(nr)
//	imag(res) = real(nl)*imag(nr) + imag(nl)*real(nr)
//	real(res) = tmp
void
complexmul(Node *nl, Node *nr, Node *res)
{
	Node n1, n2, n3, n4, n5, n6;
	Node rm1, rm2, ra, tmp;

	subnode(&n1, &n2, nl);
	subnode(&n3, &n4, nr);
	subnode(&n5, &n6, res);
	tempname(&tmp, n5.type);

	// real part -> tmp
	memset(&rm1, 0, sizeof(ra));
	rm1.op = OMUL;
	rm1.left = &n1;
	rm1.right = &n3;
	rm1.type = n1.type;

	memset(&rm2, 0, sizeof(ra));
	rm2.op = OMUL;
	rm2.left = &n2;
	rm2.right = &n4;
	rm2.type = n2.type;

	memset(&ra, 0, sizeof(ra));
	ra.op = OSUB;
	ra.left = &rm1;
	ra.right = &rm2;
	ra.type = rm1.type;
	cgen(&ra, &tmp);

	// imag part
	memset(&rm1, 0, sizeof(ra));
	rm1.op = OMUL;
	rm1.left = &n1;
	rm1.right = &n4;
	rm1.type = n1.type;

	memset(&rm2, 0, sizeof(ra));
	rm2.op = OMUL;
	rm2.left = &n2;
	rm2.right = &n3;
	rm2.type = n2.type;

	memset(&ra, 0, sizeof(ra));
	ra.op = OADD;
	ra.left = &rm1;
	ra.right = &rm2;
	ra.type = rm1.type;
	cgen(&ra, &n6);

	// tmp ->real part
	cgen(&tmp, &n5);
}
