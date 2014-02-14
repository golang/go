// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// "Portable" code generation.
// Compiled separately for 5g, 6g, and 8g, so allowed to use gg.h, opt.h.
// Must code to the intersection of the three back ends.

#include	<u.h>
#include	<libc.h>
#include	"gg.h"
#include	"opt.h"
#include	"../../pkg/runtime/funcdata.h"

static void allocauto(Prog* p);

static Sym*
makefuncdatasym(char *namefmt, int64 funcdatakind)
{
	Node nod;
	Node *pnod;
	Sym *sym;
	static int32 nsym;

	snprint(namebuf, sizeof(namebuf), namefmt, nsym++);
	sym = lookup(namebuf);
	pnod = newname(sym);
	pnod->class = PEXTERN;
	nodconst(&nod, types[TINT32], funcdatakind);
	gins(AFUNCDATA, &nod, pnod);
	return sym;
}

void
gvardef(Node *n)
{
	if(n == N)
		fatal("gvardef nil");
	switch(n->class) {
	case PAUTO:
	case PPARAM:
	case PPARAMOUT:
		gins(AVARDEF, N, n);
	}
}

static void
removevardef(Prog *firstp)
{
	Prog *p;

	for(p = firstp; p != P; p = p->link) {
		while(p->link != P && p->link->as == AVARDEF)
			p->link = p->link->link;
		if(p->to.type == D_BRANCH)
			while(p->to.u.branch != P && p->to.u.branch->as == AVARDEF)
				p->to.u.branch = p->to.u.branch->link;
	}
}

void
compile(Node *fn)
{
	Plist *pl;
	Node nod1, *n;
	Prog *ptxt, *p, *p1;
	int32 lno;
	Type *t;
	Iter save;
	vlong oldstksize;
	NodeList *l;
	Sym *gcargs;
	Sym *gclocals;
	Sym *gcdead;

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
	pl->name = linksym(curfn->nname->sym);

	setlineno(curfn);

	nodconst(&nod1, types[TINT32], 0);
	ptxt = gins(ATEXT, isblank(curfn->nname) ? N : curfn->nname, &nod1);
	if(fn->dupok)
		ptxt->TEXTFLAG |= DUPOK;
	if(fn->wrapper)
		ptxt->TEXTFLAG |= WRAPPER;

	// Clumsy but important.
	// See test/recover.go for test cases and src/pkg/reflect/value.go
	// for the actual functions being considered.
	if(myimportpath != nil && strcmp(myimportpath, "reflect") == 0) {
		if(strcmp(curfn->nname->sym->name, "callReflect") == 0 || strcmp(curfn->nname->sym->name, "callMethod") == 0)
			ptxt->TEXTFLAG |= WRAPPER;
	}	
	
	afunclit(&ptxt->from, curfn->nname);

	ginit();

	gcargs = makefuncdatasym("gcargs·%d", FUNCDATA_ArgsPointerMaps);
	gclocals = makefuncdatasym("gclocals·%d", FUNCDATA_LocalsPointerMaps);
	// TODO(cshapiro): emit the dead value map when the garbage collector
	// pre-verification pass is checked in.  It is otherwise harmless to
	// emit this information if it is not used but it does cost RSS at
	// compile time.  At present, the amount of additional RSS is
	// substantial enough to affect our smallest build machines.
	if(0)
		gcdead = makefuncdatasym("gcdead·%d", FUNCDATA_DeadValueMaps);
	else
		gcdead = nil;

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
			p->from.gotype = linksym(ngotype(l->n));
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
	if(hasdefer) {
		ginscall(deferreturn, 0);
		// deferreturn pretends to have one uintptr argument.
		// Reserve space for it so stack scanner is happy.
		if(maxarg < widthptr)
			maxarg = widthptr;
	}
	if(curfn->exit)
		genlist(curfn->exit);
	gclean();
	if(nerrors != 0)
		goto ret;

	pc->as = ARET;	// overwrite AEND
	pc->lineno = lineno;

	fixjmp(ptxt);
	if(!debug['N'] || debug['R'] || debug['P']) {
		regopt(ptxt);
		nilopt(ptxt);
	}
	expandchecks(ptxt);

	oldstksize = stksize;
	allocauto(ptxt);

	if(0)
		print("allocauto: %lld to %lld\n", oldstksize, (vlong)stksize);
	USED(oldstksize);

	setlineno(curfn);
	if((int64)stksize+maxarg > (1ULL<<31)) {
		yyerror("stack frame too large (>2GB)");
		goto ret;
	}

	// Emit garbage collection symbols.
	liveness(curfn, ptxt, gcargs, gclocals, gcdead);

	defframe(ptxt);

	if(0)
		frame(0);

	// Remove leftover instrumentation from the instruction stream.
	removevardef(ptxt);
ret:
	lineno = lno;
}

// Sort the list of stack variables. Autos after anything else,
// within autos, unused after used, within used, things with
// pointers first, zeroed things first, and then decreasing size.
// Because autos are laid out in decreasing addresses
// on the stack, pointers first, zeroed things first and decreasing size
// really means, in memory, things with pointers needing zeroing at
// the top of the stack and increasing in size.
// Non-autos sort on offset.
static int
cmpstackvar(Node *a, Node *b)
{
	int ap, bp;

	if (a->class != b->class)
		return (a->class == PAUTO) ? +1 : -1;
	if (a->class != PAUTO) {
		if (a->xoffset < b->xoffset)
			return -1;
		if (a->xoffset > b->xoffset)
			return +1;
		return 0;
	}
	if ((a->used == 0) != (b->used == 0))
		return b->used - a->used;

	ap = haspointers(a->type);
	bp = haspointers(b->type);
	if(ap != bp)
		return bp - ap;

	ap = a->needzero;
	bp = b->needzero;
	if(ap != bp)
		return bp - ap;

	if(a->type->width < b->type->width)
		return +1;
	if(a->type->width > b->type->width)
		return -1;

	return strcmp(a->sym->name, b->sym->name);
}

// TODO(lvd) find out where the PAUTO/OLITERAL nodes come from.
static void
allocauto(Prog* ptxt)
{
	NodeList *ll;
	Node* n;
	vlong w;

	stksize = 0;
	stkptrsize = 0;
	stkzerosize = 0;

	if(curfn->dcl == nil)
		return;

	// Mark the PAUTO's unused.
	for(ll=curfn->dcl; ll != nil; ll=ll->next)
		if (ll->n->class == PAUTO)
			ll->n->used = 0;

	markautoused(ptxt);

	if(precisestack_enabled) {
		// TODO: Remove when liveness analysis sets needzero instead.
		for(ll=curfn->dcl; ll != nil; ll=ll->next)
			if(ll->n->class == PAUTO)
				ll->n->needzero = 1; // ll->n->addrtaken;
	}

	listsort(&curfn->dcl, cmpstackvar);

	// Unused autos are at the end, chop 'em off.
	ll = curfn->dcl;
	n = ll->n;
	if (n->class == PAUTO && n->op == ONAME && !n->used) {
		// No locals used at all
		curfn->dcl = nil;
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
		if(haspointers(n->type)) {
			stkptrsize = stksize;
			if(n->needzero)
				stkzerosize = stksize;
		}
		if(thechar == '5')
			stksize = rnd(stksize, widthptr);
		if(stksize >= (1ULL<<31)) {
			setlineno(curfn);
			yyerror("stack frame too large (>2GB)");
		}
		n->stkdelta = -stksize - n->xoffset;
	}
	stksize = rnd(stksize, widthptr);
	stkptrsize = rnd(stkptrsize, widthptr);
	stkzerosize = rnd(stkzerosize, widthptr);

	fixautoused(ptxt);

	// The debug information needs accurate offsets on the symbols.
	for(ll = curfn->dcl; ll != nil; ll=ll->next) {
		if (ll->n->class != PAUTO || ll->n->op != ONAME)
			continue;
		ll->n->xoffset += ll->n->stkdelta;
		ll->n->stkdelta = 0;
	}
}

static void movelargefn(Node*);

void
movelarge(NodeList *l)
{
	for(; l; l=l->next)
		if(l->n->op == ODCLFUNC)
			movelargefn(l->n);
}

static void
movelargefn(Node *fn)
{
	NodeList *l;
	Node *n;

	for(l=fn->dcl; l != nil; l=l->next) {
		n = l->n;
		if(n->class == PAUTO && n->type != T && n->type->width > MaxStackVarSize)
			addrescapes(n);
	}
}

void
cgen_checknil(Node *n)
{
	Node reg;

	if(disable_checknil)
		return;
	// Ideally we wouldn't see any TUINTPTR here, but we do.
	if(n->type == T || (!isptr[n->type->etype] && n->type->etype != TUINTPTR && n->type->etype != TUNSAFEPTR)) {
		dump("checknil", n);
		fatal("bad checknil");
	}
	if((thechar == '5' && n->op != OREGISTER) || !n->addable) {
		regalloc(&reg, types[tptr], n);
		cgen(n, &reg);
		gins(ACHECKNIL, &reg, N);
		regfree(&reg);
		return;
	}
	gins(ACHECKNIL, n, N);
}
