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

	n = newname(pkglookup(name, "runtime"));
	n->class = PFUNC;
	return n;
}

void
allocparams(void)
{
	NodeList *l;
	Node *n;
	uint32 w;
	Sym *s;
	int lno;

	if(stksize < 0)
		fatal("allocparams not during code generation");

	/*
	 * allocate (set xoffset) the stack
	 * slots for all automatics.
	 * allocated starting at -w down.
	 */
	lno = lineno;
	for(l=curfn->dcl; l; l=l->next) {
		n = l->n;
		if(n->op == ONAME && n->class == PHEAP-1) {
			// heap address variable; finish the job
			// started in addrescapes.
			s = n->sym;
			tempname(n, n->type);
			n->sym = s;
		}
		if(n->op != ONAME || n->class != PAUTO)
			continue;
		if(n->type == T)
			continue;
		dowidth(n->type);
		w = n->type->width;
		if(w >= 100000000)
			fatal("bad width");
		stksize += w;
		stksize = rnd(stksize, w);
		n->xoffset = -stksize;
	}
	lineno = lno;
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
	case ODCLCONST:
	case ODCLFUNC:
	case ODCLTYPE:
		break;

	case OEMPTY:
		// insert no-op so that
		//	L:; for { }
		// does not treat L as a label for the loop.
		if(labellist && labellist->label == p3)
			gused(N);
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
				if(lab->sym == n->left->sym) {
					if(lab->breakpc == P)
						yyerror("invalid break label %S", n->left->sym);
					gjmp(lab->breakpc);
					goto donebreak;
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
	donebreak:
		break;

	case OCONTINUE:
		if(n->left != N) {
			for(lab=labellist; lab!=L; lab=lab->link) {
				if(lab->sym == n->left->sym) {
					if(lab->continpc == P)
						yyerror("invalid continue label %S", n->left->sym);
					gjmp(lab->continpc);
					goto donecont;
				}
			}
			if(lab == L)
				yyerror("continue label not defined: %S", n->left->sym);
			break;
		}

		if(continpc == P) {
			yyerror("continue is not in a loop");
			break;
		}
		gjmp(continpc);
	donecont:
		break;

	case OFOR:
		sbreak = breakpc;
		p1 = gjmp(P);			//		goto test
		breakpc = gjmp(P);		// break:	goto done
		scontin = continpc;
		continpc = pc;

		// define break and continue labels
		if((lab = labellist) != L && lab->label == p3 && lab->op == OLABEL) {
			lab->breakpc = breakpc;
			lab->continpc = continpc;
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
		bgen(n->ntest, 0, p2);			//		if(!test) goto p2
		genlist(n->nbody);				//		then
		p3 = gjmp(P);			//		goto done
		patch(p2, pc);				// else:
		genlist(n->nelse);				//		else
		patch(p3, pc);				// done:
		break;

	case OSWITCH:
		sbreak = breakpc;
		p1 = gjmp(P);			//		goto test
		breakpc = gjmp(P);		// break:	goto done

		// define break label
		if((lab = labellist) != L && lab->label == p3 && lab->op == OLABEL)
			lab->breakpc = breakpc;

		patch(p1, pc);				// test:
		genlist(n->nbody);				//		switch(test) body
		patch(breakpc, pc);			// done:
		breakpc = sbreak;
		break;

	case OSELECT:
		sbreak = breakpc;
		p1 = gjmp(P);			//		goto test
		breakpc = gjmp(P);		// break:	goto done

		// define break label
		if((lab = labellist) != L && lab->label == p3 && lab->op == OLABEL)
			lab->breakpc = breakpc;

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
		if(gen_as_init(n))
			break;
		cgen_as(n->left, n->right);
		break;

	case OCALLMETH:
		cgen_callmeth(n, 0);
		break;

	case OCALLINTER:
		cgen_callinter(n, N, 0);
		break;

	case OCALLFUNC:
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

	n->op = OCALLFUNC;
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

	case OCALLFUNC:
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
 * generate discard of value
 */
void
cgen_discard(Node *nr)
{
	Node tmp;

	if(nr == N)
		return;

	switch(nr->op) {
	case ONAME:
		if(!(nr->class & PHEAP))
			gused(nr);
		break;

	// unary
	case OADD:
	case OAND:
	case ODIV:
	case OEQ:
	case OGE:
	case OGT:
	case OLE:
	case OLSH:
	case OLT:
	case OMOD:
	case OMUL:
	case ONE:
	case OOR:
	case ORSH:
	case OSUB:
	case OXOR:
		cgen_discard(nr->left);
		cgen_discard(nr->right);
		break;

	// binary
	case OCAP:
	case OCOM:
	case OLEN:
	case OMINUS:
	case ONOT:
	case OPLUS:
		cgen_discard(nr->left);
		break;

	// special enough to just evaluate
	default:
		tempname(&tmp, nr->type);
		cgen_as(&tmp, nr);
		gused(&tmp);
	}
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

	if(isblank(nl)) {
		cgen_discard(nr);
		return;
	}

	iszer = 0;
	if(nr == N || isnil(nr)) {
		// externals and heaps should already be clear
		if(nr == N) {
			if(nl->class == PEXTERN)
				return;
			if(nl->class & PHEAP)
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

/*
 * gather series of offsets
 * >=0 is direct addressed field
 * <0 is pointer to next field (+1)
 */
int
dotoffset(Node *n, int *oary, Node **nn)
{
	int i;

	switch(n->op) {
	case ODOT:
		if(n->xoffset == BADWIDTH) {
			dump("bad width in dotoffset", n);
			fatal("bad width in dotoffset");
		}
		i = dotoffset(n->left, oary, nn);
		if(i > 0) {
			if(oary[i-1] >= 0)
				oary[i-1] += n->xoffset;
			else
				oary[i-1] -= n->xoffset;
			break;
		}
		if(i < 10)
			oary[i++] = n->xoffset;
		break;

	case ODOTPTR:
		if(n->xoffset == BADWIDTH) {
			dump("bad width in dotoffset", n);
			fatal("bad width in dotoffset");
		}
		i = dotoffset(n->left, oary, nn);
		if(i < 10)
			oary[i++] = -(n->xoffset+1);
		break;

	default:
		*nn = n;
		return 0;
	}
	if(i >= 10)
		*nn = N;
	return i;
}

/*
 * make a new off the books
 */
void
tempname(Node *n, Type *t)
{
	Sym *s;
	uint32 w;

	if(stksize < 0)
		fatal("tempname not during code generation");

	if(t == T) {
		yyerror("tempname called with nil type");
		t = types[TINT32];
	}

	// give each tmp a different name so that there
	// a chance to registerizer them
	snprint(namebuf, sizeof(namebuf), "autotmp_%.4d", statuniqgen);
	statuniqgen++;
	s = lookup(namebuf);

	memset(n, 0, sizeof(*n));
	n->op = ONAME;
	n->sym = s;
	n->type = t;
	n->class = PAUTO;
	n->addable = 1;
	n->ullman = 1;
	n->noescape = 1;

	dowidth(t);
	w = t->width;
	stksize += w;
	stksize = rnd(stksize, w);
	n->xoffset = -stksize;
}
