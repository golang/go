// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * static initialization
 */

#include	"go.h"

static void
init1(Node *n, NodeList **out)
{
	NodeList *l;

	if(n == N)
		return;
	init1(n->left, out);
	init1(n->right, out);
	for(l=n->list; l; l=l->next)
		init1(l->n, out);

	if(n->op != ONAME)
		return;
	switch(n->class) {
	case PEXTERN:
	case PFUNC:
		break;
	default:
		return;
	}

	if(n->initorder == 1)
		return;
	if(n->initorder == 2)
		fatal("init loop");

	// make sure that everything n depends on is initialized.
	// n->defn is an assignment to n
	n->initorder = 2;
	if(n->defn != N) {
		switch(n->defn->op) {
		default:
			goto bad;

		case ODCLFUNC:
			for(l=n->defn->nbody; l; l=l->next)
				init1(l->n, out);
			break;

		case OAS:
			if(n->defn->left != n)
				goto bad;
			n->defn->dodata = 1;
			init1(n->defn->right, out);
			if(debug['j'])
				print("%S\n", n->sym);
			*out = list(*out, n->defn);
			break;
		}
	}
	n->initorder = 1;
	return;

bad:
	dump("defn", n->defn);
	fatal("init1: bad defn");
}

static void
initreorder(NodeList *l, NodeList **out)
{
	Node *n;

	for(; l; l=l->next) {
		n = l->n;
		switch(n->op) {
		case ODCLFUNC:
		case ODCLCONST:
		case ODCLTYPE:
			continue;
		}
		initreorder(n->ninit, out);
		n->ninit = nil;
		init1(n, out);
	}
}

NodeList*
initfix(NodeList *l)
{
	NodeList *lout;

	lout = nil;
	initreorder(l, &lout);
	return lout;
}
