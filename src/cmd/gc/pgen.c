// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	<u.h>
#include	<libc.h>
#include	"gg.h"
#include	"opt.h"

static void allocauto(Prog* p);

void
compile(Node *fn)
{
	Plist *pl;
	Node nod1, *n;
	Prog *plocals, *ptxt, *p, *p1;
	int32 lno;
	Type *t;
	Iter save;
	vlong oldstksize;
	NodeList *l;

	if(newproc == N) {
		newproc = sysfunc("newproc");
		deferproc = sysfunc("deferproc");
		deferreturn = sysfunc("deferreturn");
		panicindex = sysfunc("panicindex");
		panicslice = sysfunc("panicslice");
		throwreturn = sysfunc("throwreturn");
	}

	lno = setlineno(fn);

	if(fn->nbody == nil) {
		if(pure_go || memcmp(fn->nname->sym->name, "init·", 6) == 0)
			yyerror("missing function body", fn);
		goto ret;
	}

	saveerrors();

	// set up domain for labels
	clearlabels();

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
	
	order(curfn);
	if(nerrors != 0)
		goto ret;
	
	hasdefer = 0;
	walk(curfn);
	if(nerrors != 0)
		goto ret;
	if(flag_race)
		racewalk(curfn);
	if(nerrors != 0)
		goto ret;

	continpc = P;
	breakpc = P;

	pl = newplist();
	pl->name = curfn->nname;

	setlineno(curfn);

	nodconst(&nod1, types[TINT32], 0);
	ptxt = gins(ATEXT, isblank(curfn->nname) ? N : curfn->nname, &nod1);
	if(fn->dupok)
		ptxt->TEXTFLAG = DUPOK;
	afunclit(&ptxt->from, curfn->nname);

	ginit();

	plocals = gins(ALOCALS, N, N);

	for(t=curfn->paramfld; t; t=t->down)
		gtrack(tracksym(t->type));

	for(l=fn->dcl; l; l=l->next) {
		n = l->n;
		if(n->op != ONAME) // might be OTYPE or OLITERAL
			continue;
		switch(n->class) {
		case PAUTO:
		case PPARAM:
		case PPARAMOUT:
			nodconst(&nod1, types[TUINTPTR], l->n->type->width);
			p = gins(ATYPE, l->n, &nod1);
			p->from.gotype = ngotype(l->n);
			break;
		}
	}

	genlist(curfn->enter);

	retpc = nil;
	if(hasdefer || curfn->exit) {
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
	allocauto(ptxt);

	plocals->to.type = D_CONST;
	plocals->to.offset = stksize;

	if(0)
		print("allocauto: %lld to %lld\n", oldstksize, (vlong)stksize);

	setlineno(curfn);
	if((int64)stksize+maxarg > (1ULL<<31))
		yyerror("stack frame too large (>2GB)");

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
	if (a->class != PAUTO) {
		if (a->xoffset < b->xoffset)
			return -1;
		if (a->xoffset > b->xoffset)
			return 1;
		return 0;
	}
	if ((a->used == 0) != (b->used == 0))
		return b->used - a->used;
	return b->type->align - a->type->align;

}

// TODO(lvd) find out where the PAUTO/OLITERAL nodes come from.
static void
allocauto(Prog* ptxt)
{
	NodeList *ll;
	Node* n;
	vlong w;

	if(curfn->dcl == nil)
		return;

	// Mark the PAUTO's unused.
	for(ll=curfn->dcl; ll != nil; ll=ll->next)
		if (ll->n->class == PAUTO)
			ll->n->used = 0;

	markautoused(ptxt);

	listsort(&curfn->dcl, cmpstackvar);

	// Unused autos are at the end, chop 'em off.
	ll = curfn->dcl;
	n = ll->n;
	if (n->class == PAUTO && n->op == ONAME && !n->used) {
		// No locals used at all
		curfn->dcl = nil;
		stksize = 0;
		fixautoused(ptxt);
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
		if (n->class != PAUTO || n->op != ONAME)
			continue;

		dowidth(n->type);
		w = n->type->width;
		if(w >= MAXWIDTH || w < 0)
			fatal("bad width");
		stksize += w;
		stksize = rnd(stksize, n->type->align);
		if(thechar == '5')
			stksize = rnd(stksize, widthptr);
		if(stksize >= (1ULL<<31)) {
			setlineno(curfn);
			yyerror("stack frame too large (>2GB)");
		}
		n->stkdelta = -stksize - n->xoffset;
	}

	fixautoused(ptxt);

	// The debug information needs accurate offsets on the symbols.
	for(ll = curfn->dcl ;ll != nil; ll=ll->next) {
		if (ll->n->class != PAUTO || ll->n->op != ONAME)
			continue;
		ll->n->xoffset += ll->n->stkdelta;
		ll->n->stkdelta = 0;
	}
}
