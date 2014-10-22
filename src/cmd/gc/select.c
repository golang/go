// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * select
 */

#include <u.h>
#include <libc.h>
#include "go.h"

static Type* selecttype(int32 size);

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
				n->list = nil;
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
	Node *n, *r, *a, *var, *selv, *cas, *dflt, *ch;
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
	// TODO(rsc): Reenable optimization once order.c can handle it.
	// golang.org/issue/7672.
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
				// ok already
				ch = n->left;
				break;

			case OSELRECV:
				ch = n->right->left;
			Selrecv1:
				if(n->left == N)
					n = n->right;
				else
					n->op = OAS;
				break;
			
			case OSELRECV2:
				ch = n->right->left;
				if(n->ntest == N)
					goto Selrecv1;
				if(n->left == N) {
					typecheck(&nblank, Erv | Easgn);
					n->left = nblank;
				}
				n->op = OAS2;
				n->list = list(list1(n->left), n->ntest);
				n->rlist = list1(n->right);
				n->right = N;
				n->left = N;
				n->ntest = N;
				n->typecheck = 0;
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

	// convert case value arguments to addresses.
	// this rewrite is used by both the general code and the next optimization.
	for(l=sel->list; l; l=l->next) {
		cas = l->n;
		setlineno(cas);
		n = cas->left;
		if(n == N)
			continue;
		switch(n->op) {
		case OSEND:
			n->right = nod(OADDR, n->right, N);
			typecheck(&n->right, Erv);
			break;
		case OSELRECV:
		case OSELRECV2:
			if(n->op == OSELRECV2 && n->ntest == N)
				n->op = OSELRECV;
			if(n->op == OSELRECV2) {
				n->ntest = nod(OADDR, n->ntest, N);
				typecheck(&n->ntest, Erv);
			}
			if(n->left == N)
				n->left = nodnil();
			else {
				n->left = nod(OADDR, n->left, N);
				typecheck(&n->left, Erv);
			}			
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
			// if selectnbsend(c, v) { body } else { default body }
			ch = n->left;
			r->ntest = mkcall1(chanfn("selectnbsend", 2, ch->type),
					types[TBOOL], &r->ninit, typename(ch->type), ch, n->right);
			break;
			
		case OSELRECV:
			// if c != nil && selectnbrecv(&v, c) { body } else { default body }
			r = nod(OIF, N, N);
			r->ninit = cas->ninit;
			ch = n->right->left;
			r->ntest = mkcall1(chanfn("selectnbrecv", 2, ch->type),
					types[TBOOL], &r->ninit, typename(ch->type), n->left, ch);
			break;

		case OSELRECV2:
			// if c != nil && selectnbrecv2(&v, c) { body } else { default body }
			r = nod(OIF, N, N);
			r->ninit = cas->ninit;
			ch = n->right->left;
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
	selv = temp(selecttype(sel->xoffset));
	r = nod(OAS, selv, N);
	typecheck(&r, Etop);
	init = list(init, r);
	var = conv(conv(nod(OADDR, selv, N), types[TUNSAFEPTR]), ptrto(types[TUINT8]));
	r = mkcall("newselect", T, nil, var, nodintconst(selv->type->width), nodintconst(sel->xoffset));
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
		// selv is no longer alive after use.
		r->nbody = list(r->nbody, nod(OVARKILL, selv, N));
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

// Keep in sync with src/runtime/chan.h.
static Type*
selecttype(int32 size)
{
	Node *sel, *sudog, *scase, *arr;

	// TODO(dvyukov): it's possible to generate SudoG and Scase only once
	// and then cache; and also cache Select per size.
	sudog = nod(OTSTRUCT, N, N);
	sudog->list = list(sudog->list, nod(ODCLFIELD, newname(lookup("g")), typenod(ptrto(types[TUINT8]))));
	sudog->list = list(sudog->list, nod(ODCLFIELD, newname(lookup("selectdone")), typenod(ptrto(types[TUINT8]))));
	sudog->list = list(sudog->list, nod(ODCLFIELD, newname(lookup("link")), typenod(ptrto(types[TUINT8]))));
	sudog->list = list(sudog->list, nod(ODCLFIELD, newname(lookup("prev")), typenod(ptrto(types[TUINT8]))));
	sudog->list = list(sudog->list, nod(ODCLFIELD, newname(lookup("elem")), typenod(ptrto(types[TUINT8]))));
	sudog->list = list(sudog->list, nod(ODCLFIELD, newname(lookup("releasetime")), typenod(types[TUINT64])));
	sudog->list = list(sudog->list, nod(ODCLFIELD, newname(lookup("nrelease")), typenod(types[TINT32])));
	sudog->list = list(sudog->list, nod(ODCLFIELD, newname(lookup("waitlink")), typenod(ptrto(types[TUINT8]))));
	typecheck(&sudog, Etype);
	sudog->type->noalg = 1;
	sudog->type->local = 1;

	scase = nod(OTSTRUCT, N, N);
	scase->list = list(scase->list, nod(ODCLFIELD, newname(lookup("elem")), typenod(ptrto(types[TUINT8]))));
	scase->list = list(scase->list, nod(ODCLFIELD, newname(lookup("chan")), typenod(ptrto(types[TUINT8]))));
	scase->list = list(scase->list, nod(ODCLFIELD, newname(lookup("pc")), typenod(types[TUINTPTR])));
	scase->list = list(scase->list, nod(ODCLFIELD, newname(lookup("kind")), typenod(types[TUINT16])));
	scase->list = list(scase->list, nod(ODCLFIELD, newname(lookup("so")), typenod(types[TUINT16])));
	scase->list = list(scase->list, nod(ODCLFIELD, newname(lookup("receivedp")), typenod(ptrto(types[TUINT8]))));
	scase->list = list(scase->list, nod(ODCLFIELD, newname(lookup("releasetime")), typenod(types[TUINT64])));
	typecheck(&scase, Etype);
	scase->type->noalg = 1;
	scase->type->local = 1;

	sel = nod(OTSTRUCT, N, N);
	sel->list = list(sel->list, nod(ODCLFIELD, newname(lookup("tcase")), typenod(types[TUINT16])));
	sel->list = list(sel->list, nod(ODCLFIELD, newname(lookup("ncase")), typenod(types[TUINT16])));
	sel->list = list(sel->list, nod(ODCLFIELD, newname(lookup("pollorder")), typenod(ptrto(types[TUINT8]))));
	sel->list = list(sel->list, nod(ODCLFIELD, newname(lookup("lockorder")), typenod(ptrto(types[TUINT8]))));
	arr = nod(OTARRAY, nodintconst(size), scase);
	sel->list = list(sel->list, nod(ODCLFIELD, newname(lookup("scase")), arr));
	arr = nod(OTARRAY, nodintconst(size), typenod(ptrto(types[TUINT8])));
	sel->list = list(sel->list, nod(ODCLFIELD, newname(lookup("lockorderarr")), arr));
	arr = nod(OTARRAY, nodintconst(size), typenod(types[TUINT16]));
	sel->list = list(sel->list, nod(ODCLFIELD, newname(lookup("pollorderarr")), arr));
	typecheck(&sel, Etype);
	sel->type->noalg = 1;
	sel->type->local = 1;

	return sel->type;
}
