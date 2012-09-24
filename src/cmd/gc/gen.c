// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * portable half of code generator.
 * mainly statements and control flow.
 */

#include <u.h>
#include <libc.h>
#include "go.h"

static void	cgen_dcl(Node *n);
static void	cgen_proc(Node *n, int proc);
static void	checkgoto(Node*, Node*);

static Label *labellist;
static Label *lastlabel;

Node*
sysfunc(char *name)
{
	Node *n;

	n = newname(pkglookup(name, runtimepkg));
	n->class = PFUNC;
	return n;
}

/*
 * the address of n has been taken and might be used after
 * the current function returns.  mark any local vars
 * as needing to move to the heap.
 */
void
addrescapes(Node *n)
{
	char buf[100];
	Node *oldfn;

	switch(n->op) {
	default:
		// probably a type error already.
		// dump("addrescapes", n);
		break;

	case ONAME:
		if(n == nodfp)
			break;

		// if this is a tmpname (PAUTO), it was tagged by tmpname as not escaping.
		// on PPARAM it means something different.
		if(n->class == PAUTO && n->esc == EscNever)
			break;

		if(debug['N'] && n->esc != EscUnknown)
			fatal("without escape analysis, only PAUTO's should have esc: %N", n);

		switch(n->class) {
		case PPARAMREF:
			addrescapes(n->defn);
			break;
		case PPARAM:
		case PPARAMOUT:
			// if func param, need separate temporary
			// to hold heap pointer.
			// the function type has already been checked
			// (we're in the function body)
			// so the param already has a valid xoffset.

			// expression to refer to stack copy
			n->stackparam = nod(OPARAM, n, N);
			n->stackparam->type = n->type;
			n->stackparam->addable = 1;
			if(n->xoffset == BADWIDTH)
				fatal("addrescapes before param assignment");
			n->stackparam->xoffset = n->xoffset;
			// fallthrough

		case PAUTO:
			n->class |= PHEAP;
			n->addable = 0;
			n->ullman = 2;
			n->xoffset = 0;

			// create stack variable to hold pointer to heap
			oldfn = curfn;
			curfn = n->curfn;
			n->heapaddr = temp(ptrto(n->type));
			snprint(buf, sizeof buf, "&%S", n->sym);
			n->heapaddr->sym = lookup(buf);
			n->heapaddr->orig->sym = n->heapaddr->sym;
			if(!debug['N'])
				n->esc = EscHeap;
			if(debug['m'])
				print("%L: moved to heap: %N\n", n->lineno, n);
			curfn = oldfn;
			break;
		}
		break;

	case OIND:
	case ODOTPTR:
		break;

	case ODOT:
	case OINDEX:
		// ODOTPTR has already been introduced,
		// so these are the non-pointer ODOT and OINDEX.
		// In &x[0], if x is a slice, then x does not
		// escape--the pointer inside x does, but that
		// is always a heap pointer anyway.
		if(!isslice(n->left->type))
			addrescapes(n->left);
		break;
	}
}

void
clearlabels(void)
{
	Label *l;

	for(l=labellist; l!=L; l=l->link)
		l->sym->label = L;
	
	labellist = L;
	lastlabel = L;
}

static Label*
newlab(Node *n)
{
	Sym *s;
	Label *lab;
	
	s = n->left->sym;
	if((lab = s->label) == L) {
		lab = mal(sizeof(*lab));
		if(lastlabel == nil)
			labellist = lab;
		else
			lastlabel->link = lab;
		lastlabel = lab;
		lab->sym = s;
		s->label = lab;
	}
	
	if(n->op == OLABEL) {
		if(lab->def != N)
			yyerror("label %S already defined at %L", s, lab->def->lineno);
		else
			lab->def = n;
	} else
		lab->use = list(lab->use, n);

	return lab;
}

void
checklabels(void)
{
	Label *lab;
	NodeList *l;

	for(lab=labellist; lab!=L; lab=lab->link) {
		if(lab->def == N) {
			for(l=lab->use; l; l=l->next)
				yyerrorl(l->n->lineno, "label %S not defined", lab->sym);
			continue;
		}
		if(lab->use == nil && !lab->used) {
			yyerrorl(lab->def->lineno, "label %S defined and not used", lab->sym);
			continue;
		}
		if(lab->gotopc != P)
			fatal("label %S never resolved", lab->sym);
		for(l=lab->use; l; l=l->next)
			checkgoto(l->n, lab->def);
	}
}

static void
checkgoto(Node *from, Node *to)
{
	int nf, nt;
	Sym *block, *dcl, *fs, *ts;
	int lno;

	if(from->sym == to->sym)
		return;

	nf = 0;
	for(fs=from->sym; fs; fs=fs->link)
		nf++;
	nt = 0;
	for(fs=to->sym; fs; fs=fs->link)
		nt++;
	fs = from->sym;
	for(; nf > nt; nf--)
		fs = fs->link;
	if(fs != to->sym) {
		lno = lineno;
		setlineno(from);

		// decide what to complain about.
		// prefer to complain about 'into block' over declarations,
		// so scan backward to find most recent block or else dcl.
		block = S;
		dcl = S;
		ts = to->sym;
		for(; nt > nf; nt--) {
			if(ts->pkg == nil)
				block = ts;
			else
				dcl = ts;
			ts = ts->link;
		}
		while(ts != fs) {
			if(ts->pkg == nil)
				block = ts;
			else
				dcl = ts;
			ts = ts->link;
			fs = fs->link;
		}

		if(block)
			yyerror("goto %S jumps into block starting at %L", from->left->sym, block->lastlineno);
		else
			yyerror("goto %S jumps over declaration of %S at %L", from->left->sym, dcl, dcl->lastlineno);
		lineno = lno;
	}
}

static Label*
stmtlabel(Node *n)
{
	Label *lab;

	if(n->sym != S)
	if((lab = n->sym->label) != L)
	if(lab->def != N)
	if(lab->def->defn == n)
		return lab;
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
	int32 wasregalloc;

	lno = setlineno(n);
	wasregalloc = anyregalloc();

	if(n == N)
		goto ret;

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
		break;

	case OBLOCK:
		genlist(n->list);
		break;

	case OLABEL:
		lab = newlab(n);

		// if there are pending gotos, resolve them all to the current pc.
		for(p1=lab->gotopc; p1; p1=p2) {
			p2 = unpatch(p1);
			patch(p1, pc);
		}
		lab->gotopc = P;
		if(lab->labelpc == P)
			lab->labelpc = pc;

		if(n->defn) {
			switch(n->defn->op) {
			case OFOR:
			case OSWITCH:
			case OSELECT:
				// so stmtlabel can find the label
				n->defn->sym = lab->sym;
			}
		}
		break;

	case OGOTO:
		// if label is defined, emit jump to it.
		// otherwise save list of pending gotos in lab->gotopc.
		// the list is linked through the normal jump target field
		// to avoid a second list.  (the jumps are actually still
		// valid code, since they're just going to another goto
		// to the same label.  we'll unwind it when we learn the pc
		// of the label in the OLABEL case above.)
		lab = newlab(n);
		if(lab->labelpc != P)
			gjmp(lab->labelpc);
		else
			lab->gotopc = gjmp(lab->gotopc);
		break;

	case OBREAK:
		if(n->left != N) {
			lab = n->left->sym->label;
			if(lab == L) {
				yyerror("break label not defined: %S", n->left->sym);
				break;
			}
			lab->used = 1;
			if(lab->breakpc == P) {
				yyerror("invalid break label %S", n->left->sym);
				break;
			}
			gjmp(lab->breakpc);
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
			lab = n->left->sym->label;
			if(lab == L) {
				yyerror("continue label not defined: %S", n->left->sym);
				break;
			}
			lab->used = 1;
			if(lab->continpc == P) {
				yyerror("invalid continue label %S", n->left->sym);
				break;
			}
			gjmp(lab->continpc);
			break;
		}
		if(continpc == P) {
			yyerror("continue is not in a loop");
			break;
		}
		gjmp(continpc);
		break;

	case OFOR:
		sbreak = breakpc;
		p1 = gjmp(P);			//		goto test
		breakpc = gjmp(P);		// break:	goto done
		scontin = continpc;
		continpc = pc;

		// define break and continue labels
		if((lab = stmtlabel(n)) != L) {
			lab->breakpc = breakpc;
			lab->continpc = continpc;
		}
		gen(n->nincr);				// contin:	incr
		patch(p1, pc);				// test:
		bgen(n->ntest, 0, -1, breakpc);		//		if(!test) goto break
		genlist(n->nbody);				//		body
		gjmp(continpc);
		patch(breakpc, pc);			// done:
		continpc = scontin;
		breakpc = sbreak;
		if(lab) {
			lab->breakpc = P;
			lab->continpc = P;
		}
		break;

	case OIF:
		p1 = gjmp(P);			//		goto test
		p2 = gjmp(P);			// p2:		goto else
		patch(p1, pc);				// test:
		bgen(n->ntest, 0, -n->likely, p2);		//		if(!test) goto p2
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
		if((lab = stmtlabel(n)) != L)
			lab->breakpc = breakpc;

		patch(p1, pc);				// test:
		genlist(n->nbody);				//		switch(test) body
		patch(breakpc, pc);			// done:
		breakpc = sbreak;
		if(lab != L)
			lab->breakpc = P;
		break;

	case OSELECT:
		sbreak = breakpc;
		p1 = gjmp(P);			//		goto test
		breakpc = gjmp(P);		// break:	goto done

		// define break label
		if((lab = stmtlabel(n)) != L)
			lab->breakpc = breakpc;

		patch(p1, pc);				// test:
		genlist(n->nbody);				//		select() body
		patch(breakpc, pc);			// done:
		breakpc = sbreak;
		if(lab != L)
			lab->breakpc = P;
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
	if(anyregalloc() != wasregalloc) {
		dump("node", n);
		fatal("registers left allocated");
	}

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
static void
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
static void
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
	if(n->alloc == nil)
		n->alloc = callnew(n->type);
	cgen_as(n->heapaddr, n->alloc);
}

/*
 * generate discard of value
 */
static void
cgen_discard(Node *nr)
{
	Node tmp;

	if(nr == N)
		return;

	switch(nr->op) {
	case ONAME:
		if(!(nr->class & PHEAP) && nr->class != PEXTERN && nr->class != PFUNC && nr->class != PPARAMREF)
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

	if(debug['g']) {
		dump("cgen_as", nl);
		dump("cgen_as = ", nr);
	}

	while(nr != N && nr->op == OCONVNOP)
		nr = nr->left;

	if(nl == N || isblank(nl)) {
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

		case TCOMPLEX64:
		case TCOMPLEX128:
			nr->val.u.cval = mal(sizeof(*nr->val.u.cval));
			mpmovecflt(&nr->val.u.cval->real, 0.0);
			mpmovecflt(&nr->val.u.cval->imag, 0.0);
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
 * generate:
 *	res = iface{typ, data}
 * n->left is typ
 * n->right is data
 */
void
cgen_eface(Node *n, Node *res)
{
	/* 
	 * the right node of an eface may contain function calls that uses res as an argument,
	 * so it's important that it is done first
	 */
	Node dst;
	dst = *res;
	dst.type = types[tptr];
	dst.xoffset += widthptr;
	cgen(n->right, &dst);
	dst.xoffset -= widthptr;
	cgen(n->left, &dst);
}

/*
 * generate:
 *	res = s[lo, hi];
 * n->left is s
 * n->list is (cap(s)-lo(TUINT), hi-lo(TUINT)[, lo*width(TUINTPTR)])
 * caller (cgen) guarantees res is an addable ONAME.
 */
void
cgen_slice(Node *n, Node *res)
{
	Node src, dst, *cap, *len, *offs, *add;

	cap = n->list->n;
	len = n->list->next->n;
	offs = N;
	if(n->list->next->next)
		offs = n->list->next->next->n;

	// dst.len = hi [ - lo ]
	dst = *res;
	dst.xoffset += Array_nel;
	dst.type = types[simtype[TUINT]];
	cgen(len, &dst);

	if(n->op != OSLICESTR) {
		// dst.cap = cap [ - lo ]
		dst = *res;
		dst.xoffset += Array_cap;
		dst.type = types[simtype[TUINT]];
		cgen(cap, &dst);
	}

	// dst.array = src.array  [ + lo *width ]
	dst = *res;
	dst.xoffset += Array_array;
	dst.type = types[TUINTPTR];

	if(n->op == OSLICEARR) {
		if(!isptr[n->left->type->etype])
			fatal("slicearr is supposed to work on pointer: %+N\n", n);
		checkref(n->left);
	}

	src = *n->left;
	src.xoffset += Array_array;
	src.type = types[TUINTPTR];

	if(offs == N) {
		cgen(&src, &dst);
	} else {
		add = nod(OADD, &src, offs);
		typecheck(&add, Erv);
		cgen(add, &dst);
	}
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
tempname(Node *nn, Type *t)
{
	Node *n;
	Sym *s;

	if(curfn == N)
		fatal("no curfn for tempname");

	if(t == T) {
		yyerror("tempname called with nil type");
		t = types[TINT32];
	}

	// give each tmp a different name so that there
	// a chance to registerizer them
	snprint(namebuf, sizeof(namebuf), "autotmp_%.4d", statuniqgen);
	statuniqgen++;
	s = lookup(namebuf);
	n = nod(ONAME, N, N);
	n->sym = s;
	s->def = n;
	n->type = t;
	n->class = PAUTO;
	n->addable = 1;
	n->ullman = 1;
	n->esc = EscNever;
	n->curfn = curfn;
	curfn->dcl = list(curfn->dcl, n);

	dowidth(t);
	n->xoffset = 0;
	*nn = *n;
}

Node*
temp(Type *t)
{
	Node *n;
	
	n = nod(OXXX, N, N);
	tempname(n, t);
	n->sym->def->used = 1;
	return n;
}
