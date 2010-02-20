// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "gg.h"

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
	Node n1, n2;

	if(debug['g']) {
		dump("\ncomplex-f", f);
		dump("complex-t", t);
	}

	if(!t->addable)
		fatal("to no addable");

	ft = cplxsubtype(simsimtype(f->type));
	tt = cplxsubtype(simsimtype(t->type));

	// copy halfs of complex literal
	if(f->op == OLITERAL) {
		// real part
		nodfconst(&n1, types[ft], &f->val.u.cval->real);
		n2 = *t;
		n2.type = types[tt];
		gmove(&n1, &n2);

		// imag part
		nodfconst(&n1, types[ft], &f->val.u.cval->imag);
		n2.xoffset += n2.type->width;
		gmove(&n1, &n2);
		return;
	}

	// make from addable
	if(!f->addable) {
		tempname(&n1, f->type);
		complexgen(f, &n1);
		f = &n1;
	}

	// real part
	n1 = *f;
	n1.type = types[ft];

	n2 = *t;
	n2.type = types[tt];

	gmove(&n1, &n2);

	// imag part
	n1.xoffset += n1.type->width;
	n2.xoffset += n2.type->width;
	gmove(&n1, &n2);

}

void
complexgen(Node *n, Node *res)
{
	Node *nl, *nr;
	Node n1, n2;
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
		tempname(&n1, n->type);
		complexgen(n, &n1);
		complexmove(&n1, n);
		return;
	}

	nl = n->left;
	if(nl == N)
		return;

	nr = n->right;
	if(nr != N) {
		// make both sides addable in ullman order
		if(nl->ullman > nr->ullman) {
			if(!nl->addable) {
				tempname(&n1, nl->type);
				complexgen(nl, &n1);
				nl = &n1;
			}
			if(!nr->addable) {
				tempname(&n1, nr->type);
				complexgen(nr, &n2);
				nr = &n2;
			}
		} else {
			if(!nr->addable) {
				tempname(&n1, nr->type);
				complexgen(nr, &n2);
				nr = &n2;
			}
			if(!nl->addable) {
				tempname(&n1, nl->type);
				complexgen(nl, &n1);
				nl = &n1;
			}
		}
	}

	switch(n->op) {
	default:
		fatal("opcode %O", n->op);
		break;

	case OADD:
	case OSUB:
	case OMUL:
	case ODIV:
		if(nr == N)
			fatal("");
		fatal("opcode %O", n->op);
		break;

	case OMINUS:
		fatal("opcode %O", n->op);
		break;

	case OCONV:
		complexmove(nl, res);
		break;
	}
}
