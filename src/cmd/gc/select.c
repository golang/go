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
				yyerror("select case must be receive, send or assign recv");
				break;

			case OAS:
				// convert x = <-c into OSELRECV(x, <-c).
				// remove implicit conversions; the eventual assignment
				// will reintroduce them.
				if((n->right->op == OCONVNOP || n->right->op == OCONVIFACE) && n->right->implicit)
					n->right = n->right->left;

				if(n->right->op != ORECV) {
					yyerror("select assignment must have receive on right hand side");
					break;
				}
				n->op = OSELRECV;
				break;

			case ORECV:
				// convert <-c into OSELRECV(N, <-c)
				n = nod(OSELRECV, N, n);
				ncase->left = n;
				break;

			case OSEND:
				break;
			}
		}
		typechecklist(ncase->nbody, Etop);
	}
	sel->xoffset = count;
	lineno = lno;
}

void
walkselect(Node *sel)
{
	int lno, i;
	Node *n, *r, *a, *tmp, *var, *cas, *dflt, *ch;
	NodeList *l, *init;
	
	if(sel->list == nil && sel->xoffset != 0)
		fatal("double walkselect");	// already rewrote
	
	lno = setlineno(sel);
	i = count(sel->list);
	
	// optimization: zero-case select
	if(i == 0) {
		sel->nbody = list1(mkcall("block", nil, nil));
		goto out;
	}

	// optimization: one-case select: single op.
	if(i == 1) {
		cas = sel->list->n;
		l = cas->ninit;
		if(cas->left != N) {  // not default:
			n = cas->left;
			l = concat(l, n->ninit);
			n->ninit = nil;
			switch(n->op) {
			default:
				fatal("select %O", n->op);

			case OSEND:
				ch = cheapexpr(n->left, &l);
				n->left = ch;
				break;

			case OSELRECV:
				r = n->right;
				ch = cheapexpr(r->left, &l);
				r->left = ch;

				if(n->left == N)
					n = r;
				else {
					n = nod(OAS, n->left, r);
					typecheck(&n, Etop);
				}
				break;
			}

			// if ch == nil { block() }; n;
			a = nod(OIF, N, N);
			a->ntest = nod(OEQ, ch, nodnil());
			a->nbody = list1(mkcall("block", nil, &l));
			typecheck(&a, Etop);
			l = list(l, a);
			l = list(l, n);
		}
		l = concat(l, cas->nbody);
		sel->nbody = l;
		goto out;
	}

	// introduce temporary variables for OSELRECV where needed.
	// this rewrite is used by both the general code and the next optimization.
	for(l=sel->list; l; l=l->next) {
		cas = l->n;
		n = cas->left;
		if(n == N)
			continue;
		switch(n->op) {
		case OSELRECV:
			ch = n->right->left;

			// If we can use the address of the target without
			// violating addressability or order of operations, do so.
			// Otherwise introduce a temporary.
			// Also introduce a temporary for := variables that escape,
			// so that we can delay the heap allocation until the case
			// is selected.
			if(n->left == N || isblank(n->left))
				n->left = nodnil();
			else if(n->left->op == ONAME &&
					(!n->colas || (n->class&PHEAP) == 0) &&
					convertop(ch->type->type, n->left->type, nil) == OCONVNOP) {
				n->left = nod(OADDR, n->left, N);
				n->left->etype = 1;  // pointer does not escape
				typecheck(&n->left, Erv);
			} else {
				tmp = nod(OXXX, N, N);
				tempname(tmp, ch->type->type);
				a = nod(OADDR, tmp, N);
				a->etype = 1;  // pointer does not escape
				typecheck(&a, Erv);
				r = nod(OAS, n->left, tmp);
				typecheck(&r, Etop);
				cas->nbody = concat(n->ninit, cas->nbody);
				n->ninit = nil;
				cas->nbody = concat(list1(r), cas->nbody);
				n->left = a;
			}
		}
	}

	// optimization: two-case select but one is default: single non-blocking op.
	if(i == 2 && (sel->list->n->left == nil || sel->list->next->n->left == nil)) {
		if(sel->list->n->left == nil) {
			cas = sel->list->next->n;
			dflt = sel->list->n;
		} else {
			dflt = sel->list->next->n;
			cas = sel->list->n;
		}
		
		n = cas->left;
		r = nod(OIF, N, N);
		r->ninit = cas->ninit;
		switch(n->op) {
		default:
			fatal("select %O", n->op);

		case OSEND:
			// if c != nil && selectnbsend(c, v) { body } else { default body }
			ch = cheapexpr(n->left, &r->ninit);
			r->ntest = nod(OANDAND, nod(ONE, ch, nodnil()),
				mkcall1(chanfn("selectnbsend", 2, ch->type),
					types[TBOOL], &r->ninit, ch, n->right));
			break;
			
		case OSELRECV:
			// if c != nil && selectnbrecv(&v, c) { body } else { default body }
			r = nod(OIF, N, N);
			r->ninit = cas->ninit;
			ch = cheapexpr(n->right->left, &r->ninit);
			r->ntest = nod(OANDAND, nod(ONE, ch, nodnil()),
				mkcall1(chanfn("selectnbrecv", 2, ch->type),
					types[TBOOL], &r->ninit, n->left, ch));
			break;
		}
		typecheck(&r->ntest, Erv);
		r->nbody = cas->nbody;
		r->nelse = concat(dflt->ninit, dflt->nbody);
		sel->nbody = list1(r);
		goto out;
	}		

	init = sel->ninit;
	sel->ninit = nil;

	// generate sel-struct
	var = nod(OXXX, N, N);
	tempname(var, ptrto(types[TUINT8]));
	r = nod(OAS, var, mkcall("newselect", var->type, nil, nodintconst(sel->xoffset)));
	typecheck(&r, Etop);
	init = list(init, r);

	// register cases
	for(l=sel->list; l; l=l->next) {
		cas = l->n;
		n = cas->left;
		r = nod(OIF, N, N);
		r->nbody = cas->ninit;
		cas->ninit = nil;
		if(n != nil) {
			r->nbody = concat(r->nbody, n->ninit);
			n->ninit = nil;
		}
		if(n == nil) {
			// selectdefault(sel *byte);
			r->ntest = mkcall("selectdefault", types[TBOOL], &init, var);
		} else {
			switch(n->op) {
			default:
				fatal("select %O", n->op);
	
			case OSEND:
				// selectsend(sel *byte, hchan *chan any, elem any) (selected bool);
				r->ntest = mkcall1(chanfn("selectsend", 2, n->left->type), types[TBOOL],
					&init, var, n->left, n->right);
				break;
			case OSELRECV:
				// selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
				r->ntest = mkcall1(chanfn("selectrecv", 2, n->right->left->type), types[TBOOL],
					&init, var, n->right->left, n->left);
				break;
			}
		}
		r->nbody = concat(r->nbody, cas->nbody);
		r->nbody = list(r->nbody, nod(OBREAK, N, N));
		init = list(init, r);
	}

	// run the select
	init = list(init, mkcall("selectgo", T, nil, var));
	sel->nbody = init;

out:
	sel->list = nil;
	walkstmtlist(sel->nbody);
	lineno = lno;
}
