// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#undef	EXTERN
#define	EXTERN
#include "gg.h"
#include "opt.h"

static void compactframe(Prog* p);

void
compile(Node *fn)
{
	Plist *pl;
	Node nod1, *n;
	Prog *ptxt;
	int32 lno;
	Type *t;
	Iter save;
	vlong oldstksize;

	if(newproc == N) {
		newproc = sysfunc("newproc");
		deferproc = sysfunc("deferproc");
		deferreturn = sysfunc("deferreturn");
		panicindex = sysfunc("panicindex");
		panicslice = sysfunc("panicslice");
		throwreturn = sysfunc("throwreturn");
	}

	if(fn->nbody == nil)
		return;

	// set up domain for labels
	clearlabels();

	lno = setlineno(fn);

	curfn = fn;
	dowidth(curfn->type);

	if(curfn->type->outnamed) {
		// add clearing of the output parameters
		t = structfirst(&save, getoutarg(curfn->type));
		while(t != T) {
			if(t->nname != N) {
				n = nod(OAS, t->nname, N);
				typecheck(&n, Etop);
				curfn->nbody = concat(list1(n), curfn->nbody);
			}
			t = structnext(&save);
		}
	}

	hasdefer = 0;
	walk(curfn);
	if(nerrors != 0 || isblank(curfn->nname))
		goto ret;

	allocparams();

	continpc = P;
	breakpc = P;

	pl = newplist();
	pl->name = curfn->nname;

	setlineno(curfn);

	nodconst(&nod1, types[TINT32], 0);
	ptxt = gins(ATEXT, curfn->nname, &nod1);
	afunclit(&ptxt->from);

	ginit();
	genlist(curfn->enter);

	retpc = nil;
	if(hasdefer || curfn->exit) {
		Prog *p1;

		p1 = gjmp(nil);
		retpc = gjmp(nil);
		patch(p1, pc);
	}

	genlist(curfn->nbody);
	gclean();
	checklabels();
	if(nerrors != 0)
		goto ret;
	if(curfn->endlineno)
		lineno = curfn->endlineno;

	if(curfn->type->outtuple != 0)
		ginscall(throwreturn, 0);

	if(retpc)
		patch(retpc, pc);
	ginit();
	if(hasdefer)
		ginscall(deferreturn, 0);
	if(curfn->exit)
		genlist(curfn->exit);
	gclean();
	if(nerrors != 0)
		goto ret;
	pc->as = ARET;	// overwrite AEND
	pc->lineno = lineno;

	if(!debug['N'] || debug['R'] || debug['P']) {
		regopt(ptxt);
	}

	oldstksize = stksize;
	compactframe(ptxt);
	if(0)
		print("compactframe: %ld to %ld\n", oldstksize, stksize);

	defframe(ptxt);

	if(0)
		frame(0);

ret:
	lineno = lno;
}


// Sort the list of stack variables.  autos after anything else,
// within autos, unused after used, and within used on reverse alignment.
// non-autos sort on offset.
static int
cmpstackvar(Node *a, Node *b)
{
	if (a->class != b->class)
		return (a->class == PAUTO) ? 1 : -1;
	if (a->class != PAUTO)
		return a->xoffset - b->xoffset;
	if ((a->used == 0) != (b->used == 0))
		return b->used - a->used;
	return b->type->align - a->type->align;

}

static void
compactframe(Prog* ptxt)
{
	NodeList *ll;
	Node* n;
	Prog *p;
	uint32 w;

	if (stksize == 0)
		return;

	// Mark the PAUTO's unused.
	for(ll=curfn->dcl; ll != nil; ll=ll->next)
		if (ll->n->class == PAUTO && ll->n->op == ONAME)
			ll->n->used = 0;

	// Sweep the prog list to mark any used nodes.
	for (p = ptxt; p; p = p->link) {
		if (p->from.type == D_AUTO && p->from.node)
			p->from.node->used++;

		if (p->to.type == D_AUTO && p->to.node)
			p->to.node->used++;
	}

	listsort(&curfn->dcl, cmpstackvar);

	// Unused autos are at the end, chop 'em off.
	ll = curfn->dcl;
	n = ll->n;
	if (n->class == PAUTO && n->op == ONAME && !n->used) {
		curfn->dcl = nil;
		stksize = 0;
		return;
	}

	for(ll = curfn->dcl; ll->next != nil; ll=ll->next) {
		n = ll->next->n;
		if (n->class == PAUTO && n->op == ONAME && !n->used) {
			ll->next = nil;
			curfn->dcl->end = ll;
			break;
		}
	}

	// Reassign stack offsets of the locals that are still there.
	stksize = 0;
	for(ll = curfn->dcl; ll != nil; ll=ll->next) {
		n = ll->n;
		// TODO find out where the literal autos come from
		if (n->class != PAUTO || n->op != ONAME)
			continue;

		w = n->type->width;
		if((w >= MAXWIDTH) || (w < 1))
			fatal("bad width");
		stksize += w;
		stksize = rnd(stksize, n->type->align);
		if(thechar == '5')
			stksize = rnd(stksize, widthptr);
		n->stkdelta = -stksize - n->xoffset;
	}

	// Fixup instructions.
	for (p = ptxt; p; p = p->link) {
		if (p->from.type == D_AUTO && p->from.node)
			p->from.offset += p->from.node->stkdelta;

		if (p->to.type == D_AUTO && p->to.node)
			p->to.offset += p->to.node->stkdelta;
	}

	// The debug information needs accurate offsets on the symbols.
	for(ll = curfn->dcl ;ll != nil; ll=ll->next) {
		if (ll->n->class != PAUTO || ll->n->op != ONAME)
			continue;
		ll->n->xoffset += ll->n->stkdelta;
		ll->n->stkdelta = 0;
	}
}
