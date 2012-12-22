// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * select
 */

#include <u.h>
#include <libc.h>
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

			case OAS2RECV:
				// convert x, ok = <-c into OSELRECV2(x, <-c) with ntest=ok
				if(n->rlist->n->op != ORECV) {
					yyerror("select assignment must have receive on right hand side");
					break;
				}
				n->op = OSELRECV2;
				n->left = n->list->n;
				n->ntest = n->list->next->n;
				n->right = n->rlist->n;
				n->rlist = nil;
				break;

			case ORECV:
				// convert <-c into OSELRECV(N, <-c)
				n = nod(OSELRECV, N, n);
				n->typecheck = 1;
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
		setlineno(cas);
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
			
			case OSELRECV2:
				r = n->right;
				ch = cheapexpr(r->left, &l);
				r->left = ch;
				
				a = nod(OAS2, N, N);
				a->list = n->list;
				a->rlist = list1(n->right);
				n = a;
				typecheck(&n, Etop);
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
		setlineno(cas);
		n = cas->left;
		if(n == N)
			continue;
		switch(n->op) {
		case OSELRECV:
		case OSELRECV2:
			ch = n->right->left;

			// If we can use the address of the target without
			// violating addressability or order of operations, do so.
			// Otherwise introduce a temporary.
			// Also introduce a temporary for := variables that escape,
			// so that we can delay the heap allocation until the case
			// is selected.
			if(n->op == OSELRECV2) {
				if(n->ntest == N || isblank(n->ntest))
					n->ntest = nodnil();
				else if(n->ntest->op == ONAME &&
						(!n->colas || (n->ntest->class&PHEAP) == 0) &&
						convertop(types[TBOOL], n->ntest->type, nil) == OCONVNOP) {
					n->ntest = nod(OADDR, n->ntest, N);
					n->ntest->etype = 1;  // pointer does not escape
					typecheck(&n->ntest, Erv);
				} else {
					tmp = temp(types[TBOOL]);
					a = nod(OADDR, tmp, N);
					a->etype = 1;  // pointer does not escape
					typecheck(&a, Erv);
					r = nod(OAS, n->ntest, tmp);
					typecheck(&r, Etop);
					cas->nbody = concat(list1(r), cas->nbody);
					n->ntest = a;
				}
			}

			if(n->left == N || isblank(n->left))
				n->left = nodnil();
			else if(n->left->op == ONAME &&
					(!n->colas || (n->left->class&PHEAP) == 0) &&
					convertop(ch->type->type, n->left->type, nil) == OCONVNOP) {
				n->left = nod(OADDR, n->left, N);
				n->left->etype = 1;  // pointer does not escape
				typecheck(&n->left, Erv);
			} else {
				tmp = temp(ch->type->type);
				a = nod(OADDR, tmp, N);
				a->etype = 1;  // pointer does not escape
				typecheck(&a, Erv);
				r = nod(OAS, n->left, tmp);
				typecheck(&r, Etop);
				cas->nbody = concat(list1(r), cas->nbody);
				n->left = a;
			}
			
			cas->nbody = concat(n->ninit, cas->nbody);
			n->ninit = nil;
			break;
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
		setlineno(n);
		r = nod(OIF, N, N);
		r->ninit = cas->ninit;
		switch(n->op) {
		default:
			fatal("select %O", n->op);

		case OSEND:
			// if c != nil && selectnbsend(c, v) { body } else { default body }
			ch = cheapexpr(n->left, &r->ninit);
			r->ntest = mkcall1(chanfn("selectnbsend", 2, ch->type),
					types[TBOOL], &r->ninit, typename(ch->type), ch, n->right);
			break;
			
		case OSELRECV:
			// if c != nil && selectnbrecv(&v, c) { body } else { default body }
			r = nod(OIF, N, N);
			r->ninit = cas->ninit;
			ch = cheapexpr(n->right->left, &r->ninit);
			r->ntest = mkcall1(chanfn("selectnbrecv", 2, ch->type),
					types[TBOOL], &r->ninit, typename(ch->type), n->left, ch);
			break;

		case OSELRECV2:
			// if c != nil && selectnbrecv2(&v, c) { body } else { default body }
			r = nod(OIF, N, N);
			r->ninit = cas->ninit;
			ch = cheapexpr(n->right->left, &r->ninit);
			r->ntest = mkcall1(chanfn("selectnbrecv2", 2, ch->type),
					types[TBOOL], &r->ninit, typename(ch->type), n->left, n->ntest, ch);
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
	setlineno(sel);
	var = temp(ptrto(types[TUINT8]));
	r = nod(OAS, var, mkcall("newselect", var->type, nil, nodintconst(sel->xoffset)));
	typecheck(&r, Etop);
	init = list(init, r);

	// register cases
	for(l=sel->list; l; l=l->next) {
		cas = l->n;
		setlineno(cas);
		n = cas->left;
		r = nod(OIF, N, N);
		r->ninit = cas->ninit;
		cas->ninit = nil;
		if(n != nil) {
			r->ninit = concat(r->ninit, n->ninit);
			n->ninit = nil;
		}
		if(n == nil) {
			// selectdefault(sel *byte);
			r->ntest = mkcall("selectdefault", types[TBOOL], &r->ninit, var);
		} else {
			switch(n->op) {
			default:
				fatal("select %O", n->op);
	
			case OSEND:
				// selectsend(sel *byte, hchan *chan any, elem *any) (selected bool);
				n->left = localexpr(safeexpr(n->left, &r->ninit), n->left->type, &r->ninit);
				n->right = localexpr(n->right, n->left->type->type, &r->ninit);
				n->right = nod(OADDR, n->right, N);
				n->right->etype = 1;  // pointer does not escape
				typecheck(&n->right, Erv);
				r->ntest = mkcall1(chanfn("selectsend", 2, n->left->type), types[TBOOL],
					&r->ninit, var, n->left, n->right);
				break;

			case OSELRECV:
				// selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
				r->ntest = mkcall1(chanfn("selectrecv", 2, n->right->left->type), types[TBOOL],
					&r->ninit, var, n->right->left, n->left);
				break;

			case OSELRECV2:
				// selectrecv2(sel *byte, hchan *chan any, elem *any, received *bool) (selected bool);
				r->ntest = mkcall1(chanfn("selectrecv2", 2, n->right->left->type), types[TBOOL],
					&r->ninit, var, n->right->left, n->left, n->ntest);
				break;
			}
		}
		r->nbody = concat(r->nbody, cas->nbody);
		r->nbody = list(r->nbody, nod(OBREAK, N, N));
		init = list(init, r);
	}

	// run the select
	setlineno(sel);
	init = list(init, mkcall("selectgo", T, nil, var));
	sel->nbody = init;

out:
	sel->list = nil;
	walkstmtlist(sel->nbody);
	lineno = lno;
}
