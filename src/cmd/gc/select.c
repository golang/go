// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * select
 */

#include "go.h"

void
typecheckselect(Node *sel)
{
	Node *ncase, *n, *def;
	NodeList *l;
	int lno, count;

	def = nil;
	lno = setlineno(sel);
	count = 0;
	typechecklist(sel->ninit, Etop);
	for(l=sel->list; l; l=l->next) {
		count++;
		ncase = l->n;
		setlineno(ncase);
		if(ncase->op != OXCASE)
			fatal("typecheckselect %O", ncase->op);

		if(ncase->list == nil) {
			// default
			if(def != N)
				yyerror("multiple defaults in select (first at %L)", def->lineno);
			else
				def = ncase;
		} else if(ncase->list->next) {
			yyerror("select cases cannot be lists");
		} else {
			n = typecheck(&ncase->list->n, Etop);
			ncase->left = n;
			ncase->list = nil;
			setlineno(n);
			switch(n->op) {
			default:
				yyerror("select case must be receive, send or assign recv");;
				break;

			case OAS:
				// convert x = <-c into OSELRECV(x, c)
				if(n->right->op != ORECV) {
					yyerror("select assignment must have receive on right hand side");
					break;
				}
				n->op = OSELRECV;
				n->right = n->right->left;
				break;

			case ORECV:
				// convert <-c into OSELRECV(N, c)
				n->op = OSELRECV;
				n->right = n->left;
				n->left = N;
				break;

			case OSEND:
				break;
			}
		}
		typechecklist(ncase->nbody, Etop);
	}
	sel->xoffset = count;
	if(count == 0)
		yyerror("empty select");
	lineno = lno;
}

void
walkselect(Node *sel)
{
	int lno;
	Node *n, *ncase, *r, *a, *tmp, *var;
	NodeList *l, *init;

	lno = setlineno(sel);
	init = sel->ninit;
	sel->ninit = nil;

	// generate sel-struct
	var = nod(OXXX, N, N);
	tempname(var, ptrto(types[TUINT8]));
	r = nod(OAS, var, mkcall("newselect", var->type, nil, nodintconst(sel->xoffset)));
	typecheck(&r, Etop);
	init = list(init, r);

	if(sel->list == nil)
		fatal("double walkselect");	// already rewrote

	// register cases
	for(l=sel->list; l; l=l->next) {
		ncase = l->n;
		n = ncase->left;
		r = nod(OIF, N, N);
		r->nbody = ncase->ninit;
		ncase->ninit = nil;
		if(n != nil) {
			r->nbody = concat(r->nbody, n->ninit);
			n->ninit = nil;
		}
		if(n == nil) {
			// selectdefault(sel *byte);
			r->ntest = mkcall("selectdefault", types[TBOOL], &init, var);
		} else if(n->op == OSEND) {
			// selectsend(sel *byte, hchan *chan any, elem any) (selected bool);
			r->ntest = mkcall1(chanfn("selectsend", 2, n->left->type), types[TBOOL], &init, var, n->left, n->right);
		} else if(n->op == OSELRECV) {
			tmp = N;
			if(n->left == N)
				a = nodnil();
			else {
				// introduce temporary until we're sure this will succeed.
				tmp = nod(OXXX, N, N);
				tempname(tmp, n->right->type->type);
				a = nod(OADDR, tmp, N);
			}
			// selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
			r->ntest = mkcall1(chanfn("selectrecv", 2, n->right->type), types[TBOOL], &init, var, n->right, a);
			if(tmp != N) {
				a = nod(OAS, n->left, tmp);
				typecheck(&a, Etop);
				r->nbody = list(r->nbody, a);
			}
		} else
			fatal("select %O", n->op);
		r->nbody = concat(r->nbody, ncase->nbody);
		r->nbody = list(r->nbody, nod(OBREAK, N, N));
		init = list(init, r);
	}

	// run the select
	init = list(init, mkcall("selectgo", T, nil, var));
	sel->nbody = init;
	sel->list = nil;
	walkstmtlist(init);

	lineno = lno;
}
