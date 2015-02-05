// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// "Portable" code generation.
// Compiled separately for 5g, 6g, and 8g, so allowed to use gg.h, opt.h.
// Must code to the intersection of the three back ends.

#include	<u.h>
#include	<libc.h>
#include	"md5.h"
#include	"go.h"
//#include	"opt.h"
#include	"../../runtime/funcdata.h"
#include	"../ld/textflag.h"

static void allocauto(Prog* p);
static void emitptrargsmap(void);

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
	thearch.gins(AFUNCDATA, &nod, pnod);
	return sym;
}

// gvardef inserts a VARDEF for n into the instruction stream.
// VARDEF is an annotation for the liveness analysis, marking a place
// where a complete initialization (definition) of a variable begins.
// Since the liveness analysis can see initialization of single-word
// variables quite easy, gvardef is usually only called for multi-word
// or 'fat' variables, those satisfying isfat(n->type).
// However, gvardef is also called when a non-fat variable is initialized
// via a block move; the only time this happens is when you have
//	return f()
// for a function with multiple return values exactly matching the return
// types of the current function.
//
// A 'VARDEF x' annotation in the instruction stream tells the liveness
// analysis to behave as though the variable x is being initialized at that
// point in the instruction stream. The VARDEF must appear before the
// actual (multi-instruction) initialization, and it must also appear after
// any uses of the previous value, if any. For example, if compiling:
//
//	x = x[1:]
//
// it is important to generate code like:
//
//	base, len, cap = pieces of x[1:]
//	VARDEF x
//	x = {base, len, cap}
//
// If instead the generated code looked like:
//
//	VARDEF x
//	base, len, cap = pieces of x[1:]
//	x = {base, len, cap}
//
// then the liveness analysis would decide the previous value of x was
// unnecessary even though it is about to be used by the x[1:] computation.
// Similarly, if the generated code looked like:
//
//	base, len, cap = pieces of x[1:]
//	x = {base, len, cap}
//	VARDEF x
//
// then the liveness analysis will not preserve the new value of x, because
// the VARDEF appears to have "overwritten" it.
//
// VARDEF is a bit of a kludge to work around the fact that the instruction
// stream is working on single-word values but the liveness analysis
// wants to work on individual variables, which might be multi-word
// aggregates. It might make sense at some point to look into letting
// the liveness analysis work on single-word values as well, although
// there are complications around interface values, slices, and strings,
// all of which cannot be treated as individual words.
//
// VARKILL is the opposite of VARDEF: it marks a value as no longer needed,
// even if its address has been taken. That is, a VARKILL annotation asserts
// that its argument is certainly dead, for use when the liveness analysis
// would not otherwise be able to deduce that fact.

static void
gvardefx(Node *n, int as)
{
	if(n == N)
		fatal("gvardef nil");
	if(n->op != ONAME) {
		yyerror("gvardef %#O; %N", n->op, n);
		return;
	}
	switch(n->class) {
	case PAUTO:
	case PPARAM:
	case PPARAMOUT:
		thearch.gins(as, N, n);
	}
}

void
gvardef(Node *n)
{
	gvardefx(n, AVARDEF);
}

void
gvarkill(Node *n)
{
	gvardefx(n, AVARKILL);
}

static void
removevardef(Prog *firstp)
{
	Prog *p;

	for(p = firstp; p != P; p = p->link) {
		while(p->link != P && (p->link->as == AVARDEF || p->link->as == AVARKILL))
			p->link = p->link->link;
		if(p->to.type == TYPE_BRANCH)
			while(p->to.u.branch != P && (p->to.u.branch->as == AVARDEF || p->to.u.branch->as == AVARKILL))
				p->to.u.branch = p->to.u.branch->link;
	}
}

static void
gcsymdup(Sym *s)
{
	LSym *ls;
	uint64 lo, hi;
	
	ls = linksym(s);
	if(ls->nr > 0)
		fatal("cannot rosymdup %s with relocations", ls->name);
	MD5 d;
	md5reset(&d);
	md5write(&d, ls->p, ls->np);
	lo = md5sum(&d, &hi);
	ls->name = smprint("gclocals·%016llux%016llux", lo, hi);
	ls->dupok = 1;
}

void
compile(Node *fn)
{
	Plist *pl;
	Node nod1, *n;
	Prog *ptxt, *p;
	int32 lno;
	Type *t;
	Iter save;
	vlong oldstksize;
	NodeList *l;
	Node *nam;
	Sym *gcargs;
	Sym *gclocals;

	if(newproc == N) {
		newproc = sysfunc("newproc");
		deferproc = sysfunc("deferproc");
		deferreturn = sysfunc("deferreturn");
		panicindex = sysfunc("panicindex");
		panicslice = sysfunc("panicslice");
		throwreturn = sysfunc("throwreturn");
	}

	lno = setlineno(fn);

	curfn = fn;
	dowidth(curfn->type);

	if(fn->nbody == nil) {
		if(pure_go || strncmp(fn->nname->sym->name, "init·", 6) == 0) {
			yyerror("missing function body", fn);
			goto ret;
		}
		if(debug['A'])
			goto ret;
		emitptrargsmap();
		goto ret;
	}

	saveerrors();

	// set up domain for labels
	clearlabels();

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
	nam = curfn->nname;
	if(isblank(nam))
		nam = N;
	ptxt = thearch.gins(ATEXT, nam, &nod1);
	if(fn->dupok)
		ptxt->from3.offset |= DUPOK;
	if(fn->wrapper)
		ptxt->from3.offset |= WRAPPER;
	if(fn->needctxt)
		ptxt->from3.offset |= NEEDCTXT;
	if(fn->nosplit)
		ptxt->from3.offset |= NOSPLIT;

	// Clumsy but important.
	// See test/recover.go for test cases and src/reflect/value.go
	// for the actual functions being considered.
	if(myimportpath != nil && strcmp(myimportpath, "reflect") == 0) {
		if(strcmp(curfn->nname->sym->name, "callReflect") == 0 || strcmp(curfn->nname->sym->name, "callMethod") == 0)
			ptxt->from3.offset |= WRAPPER;
	}	
	
	afunclit(&ptxt->from, curfn->nname);

	thearch.ginit();

	gcargs = makefuncdatasym("gcargs·%d", FUNCDATA_ArgsPointerMaps);
	gclocals = makefuncdatasym("gclocals·%d", FUNCDATA_LocalsPointerMaps);

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
			p = thearch.gins(ATYPE, l->n, &nod1);
			p->from.gotype = linksym(ngotype(l->n));
			break;
		}
	}

	genlist(curfn->enter);
	genlist(curfn->nbody);
	thearch.gclean();
	checklabels();
	if(nerrors != 0)
		goto ret;
	if(curfn->endlineno)
		lineno = curfn->endlineno;

	if(curfn->type->outtuple != 0)
		thearch.ginscall(throwreturn, 0);

	thearch.ginit();
	// TODO: Determine when the final cgen_ret can be omitted. Perhaps always?
	thearch.cgen_ret(nil);
	if(hasdefer) {
		// deferreturn pretends to have one uintptr argument.
		// Reserve space for it so stack scanner is happy.
		if(maxarg < widthptr)
			maxarg = widthptr;
	}
	thearch.gclean();
	if(nerrors != 0)
		goto ret;

	pc->as = ARET;	// overwrite AEND
	pc->lineno = lineno;

	fixjmp(ptxt);
	if(!debug['N'] || debug['R'] || debug['P']) {
		regopt(ptxt);
		nilopt(ptxt);
	}
	thearch.expandchecks(ptxt);

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
	liveness(curfn, ptxt, gcargs, gclocals);
	gcsymdup(gcargs);
	gcsymdup(gclocals);

	thearch.defframe(ptxt);

	if(debug['f'])
		frame(0);

	// Remove leftover instrumentation from the instruction stream.
	removevardef(ptxt);
ret:
	lineno = lno;
}

static void
emitptrargsmap(void)
{
	int nptr, nbitmap, j, off;
	vlong xoffset;
	Bvec *bv;
	Sym *sym;
	
	sym = lookup(smprint("%s.args_stackmap", curfn->nname->sym->name));

	nptr = curfn->type->argwid / widthptr;
	bv = bvalloc(nptr*2);
	nbitmap = 1;
	if(curfn->type->outtuple > 0)
		nbitmap = 2;
	off = duint32(sym, 0, nbitmap);
	off = duint32(sym, off, bv->n);
	if(curfn->type->thistuple > 0) {
		xoffset = 0;
		twobitwalktype1(getthisx(curfn->type), &xoffset, bv);
	}
	if(curfn->type->intuple > 0) {
		xoffset = 0;
		twobitwalktype1(getinargx(curfn->type), &xoffset, bv);
	}
	for(j = 0; j < bv->n; j += 32)
		off = duint32(sym, off, bv->b[j/32]);
	if(curfn->type->outtuple > 0) {
		xoffset = 0;
		twobitwalktype1(getoutargx(curfn->type), &xoffset, bv);
		for(j = 0; j < bv->n; j += 32)
			off = duint32(sym, off, bv->b[j/32]);
	}
	ggloblsym(sym, off, RODATA);
	free(bv);
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

	if (a->class != b->class) {
		if(a->class == PAUTO)
			return +1;
		return -1;
	}
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
		if(w >= thearch.MAXWIDTH || w < 0)
			fatal("bad width");
		stksize += w;
		stksize = rnd(stksize, n->type->align);
		if(haspointers(n->type))
			stkptrsize = stksize;
		if(thearch.thechar == '5' || thearch.thechar == '9')
			stksize = rnd(stksize, widthptr);
		if(stksize >= (1ULL<<31)) {
			setlineno(curfn);
			yyerror("stack frame too large (>2GB)");
		}
		n->stkdelta = -stksize - n->xoffset;
	}
	stksize = rnd(stksize, widthreg);
	stkptrsize = rnd(stkptrsize, widthreg);

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
	// Ideally we wouldn't see any integer types here, but we do.
	if(n->type == T || (!isptr[n->type->etype] && !isint[n->type->etype] && n->type->etype != TUNSAFEPTR)) {
		dump("checknil", n);
		fatal("bad checknil");
	}
	if(((thearch.thechar == '5' || thearch.thechar == '9') && n->op != OREGISTER) || !n->addable || n->op == OLITERAL) {
		thearch.regalloc(&reg, types[tptr], n);
		thearch.cgen(n, &reg);
		thearch.gins(ACHECKNIL, &reg, N);
		thearch.regfree(&reg);
		return;
	}
	thearch.gins(ACHECKNIL, n, N);
}
