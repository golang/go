// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * portable half of code generator.
 * mainly statements and control flow.
 */

#include "go.h"

Node*
sysfunc(char *name)
{
	Node *n;

	n = newname(pkglookup(name, "sys"));
	n->class = PFUNC;
	return n;
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
		if(n->class & PHEAP)
			w = widthptr;
		stksize += w;
		stksize = rnd(stksize, w);

		n->xoffset = -stksize;
	}
}

void
newlab(int op, Sym *s)
{
	Label *lab;

	lab = mal(sizeof(*lab));
	lab->link = labellist;
	labellist = lab;

	lab->sym = s;
	lab->op = op;
	lab->label = pc;
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

/*
 * compile statements
 */
void
genlist(NodeList *l)
{
	for(; l; l=l->next)
		gen(l->n);
}

void
gen(Node *n)
{
	int32 lno;
	Prog *scontin, *sbreak;
	Prog *p1, *p2, *p3;
	Label *lab;

	lno = setlineno(n);

	if(n == N)
		goto ret;

	p3 = pc;	// save pc for loop labels
	if(n->ninit)
		genlist(n->ninit);

	setlineno(n);

	switch(n->op) {
	default:
		fatal("gen: unknown op %N", n);
		break;

	case OCASE:
	case OFALL:
	case OXCASE:
	case OXFALL:
	case OEMPTY:
		break;

	case OBLOCK:
		genlist(n->list);
		break;

	case OLABEL:
		newlab(OLABEL, n->left->sym);
		break;

	case OGOTO:
		newlab(OGOTO, n->left->sym);
		gjmp(P);
		break;

	case OBREAK:
		if(n->left != N) {
			for(lab=labellist; lab!=L; lab=lab->link) {
				if(lab->breakpc != P) {
					gjmp(lab->breakpc);
					break;
				}
			}
			if(lab == L)
				yyerror("break label not defined: %S", n->left->sym);
			break;
		}
		if(breakpc == P) {
			yyerror("break is not in a loop");
			break;
		}
		gjmp(breakpc);
		break;

	case OCONTINUE:
		if(n->left != N) {
			for(lab=labellist; lab!=L; lab=lab->link) {
				if(lab->continpc != P) {
					gjmp(lab->continpc);
					break;
				}
			}
			if(lab == L)
				yyerror("break label not defined: %S", n->left->sym);
			break;
		}

		if(continpc == P) {
			yyerror("gen: continue is not in a loop");
			break;
		}
		gjmp(continpc);
		break;

	case OFOR:
		sbreak = breakpc;
		p1 = gjmp(P);			// 		goto test
		breakpc = gjmp(P);		// break:	goto done
		scontin = continpc;
		continpc = pc;

		// define break and cotinue labels
		for(lab=labellist; lab!=L; lab=lab->link) {
			if(lab->label != p3)
				break;
			if(lab->op == OLABEL) {
				lab->breakpc = breakpc;
				lab->continpc = continpc;
			}
		}

		gen(n->nincr);				// contin:	incr
		patch(p1, pc);				// test:
		if(n->ntest != N)
			if(n->ntest->ninit != nil)
				genlist(n->ntest->ninit);
		bgen(n->ntest, 0, breakpc);		//		if(!test) goto break
		genlist(n->nbody);				//		body
		gjmp(continpc);
		patch(breakpc, pc);			// done:
		continpc = scontin;
		breakpc = sbreak;
		break;

	case OIF:
		p1 = gjmp(P);			//		goto test
		p2 = gjmp(P);			// p2:		goto else
		patch(p1, pc);				// test:
		if(n->ntest != N)
			if(n->ntest->ninit != nil)
				genlist(n->ntest->ninit);
		bgen(n->ntest, 0, p2);			// 		if(!test) goto p2
		genlist(n->nbody);				//		then
		p3 = gjmp(P);			//		goto done
		patch(p2, pc);				// else:
		genlist(n->nelse);				//		else
		patch(p3, pc);				// done:
		break;

	case OSWITCH:
		sbreak = breakpc;
		p1 = gjmp(P);			// 		goto test
		breakpc = gjmp(P);		// break:	goto done

		// define break label
		for(lab=labellist; lab!=L; lab=lab->link) {
			if(lab->label != p3)
				break;
			if(lab->op == OLABEL) {
				lab->breakpc = breakpc;
			}
		}

		patch(p1, pc);				// test:
		genlist(n->nbody);				//		switch(test) body
		patch(breakpc, pc);			// done:
		breakpc = sbreak;
		break;

	case OSELECT:
		sbreak = breakpc;
		p1 = gjmp(P);			// 		goto test
		breakpc = gjmp(P);		// break:	goto done

		// define break label
		for(lab=labellist; lab!=L; lab=lab->link) {
			if(lab->label != p3)
				break;
			if(lab->op == OLABEL) {
				lab->breakpc = breakpc;
			}
		}

		patch(p1, pc);				// test:
		genlist(n->nbody);				//		select() body
		patch(breakpc, pc);			// done:
		breakpc = sbreak;
		break;

	case OASOP:
		cgen_asop(n);
		break;

	case ODCL:
		cgen_dcl(n->left);
		break;

	case OAS:
		cgen_as(n->left, n->right);
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
		cgen_proc(n, 1);
		break;

	case ODEFER:
		cgen_proc(n, 2);
		break;

	case ORETURN:
		cgen_ret(n);
		break;
	}

ret:
	lineno = lno;
}

/*
 * generate call to non-interface method
 *	proc=0	normal call
 *	proc=1	goroutine run in new proc
 *	proc=2	defer call save away stack
 */
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
		n->left->class = PFUNC;
	cgen_call(n, proc);
}

/*
 * generate code to start new proc running call n.
 */
void
cgen_proc(Node *n, int proc)
{
	switch(n->left->op) {
	default:
		fatal("cgen_proc: unknown call %O", n->left->op);

	case OCALLMETH:
		cgen_callmeth(n->left, proc);
		break;

	case OCALLINTER:
		cgen_callinter(n->left, N, proc);
		break;

	case OCALL:
		cgen_call(n->left, proc);
		break;
	}

}

/*
 * generate declaration.
 * nothing to do for on-stack automatics,
 * but might have to allocate heap copy
 * for escaped variables.
 */
void
cgen_dcl(Node *n)
{
	if(debug['g'])
		dump("\ncgen-dcl", n);
	if(n->op != ONAME) {
		dump("cgen_dcl", n);
		fatal("cgen_dcl");
	}
	if(!(n->class & PHEAP))
		return;
	cgen_as(n->heapaddr, n->alloc);
}

/*
 * generate assignment:
 *	nl = nr
 * nr == N means zero nl.
 */
void
cgen_as(Node *nl, Node *nr)
{
	Node nc;
	Type *tl;
	int iszer;

	if(nl == N)
		return;

	if(debug['g']) {
		dump("cgen_as", nl);
		dump("cgen_as = ", nr);
	}

	iszer = 0;
	if(nr == N || isnil(nr)) {
		// externals and heaps should already be clear
		if(nr == N) {
			if(nl->class == PEXTERN)
				return;
			if(nl->class & PHEAP)
				return;
			if(gen_as_init(nr, nl))
				return;
		}

		tl = nl->type;
		if(tl == T)
			return;
		if(isfat(tl)) {
			clearfat(nl);
			goto ret;
		}

		/* invent a "zero" for the rhs */
		iszer = 1;
		nr = &nc;
		memset(nr, 0, sizeof(*nr));
		switch(simtype[tl->etype]) {
		default:
			fatal("cgen_as: tl %T", tl);
			break;

		case TINT8:
		case TUINT8:
		case TINT16:
		case TUINT16:
		case TINT32:
		case TUINT32:
		case TINT64:
		case TUINT64:
			nr->val.u.xval = mal(sizeof(*nr->val.u.xval));
			mpmovecfix(nr->val.u.xval, 0);
			nr->val.ctype = CTINT;
			break;

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
	if(iszer && nl->addable)
		gused(nl);

ret:
	;
}
