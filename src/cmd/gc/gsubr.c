// Derived from Inferno utils/6c/txt.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/txt.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <u.h>
#include <libc.h>
#include "go.h"
#include "../../runtime/funcdata.h"
#include "../ld/textflag.h"

void
ggloblnod(Node *nam)
{
	Prog *p;

	p = thearch.gins(AGLOBL, nam, N);
	p->lineno = nam->lineno;
	p->from.sym->gotype = linksym(ngotype(nam));
	p->to.sym = nil;
	p->to.type = TYPE_CONST;
	p->to.offset = nam->type->width;
	if(nam->readonly)
		p->from3.offset = RODATA;
	if(nam->type != T && !haspointers(nam->type))
		p->from3.offset |= NOPTR;
}

void
gtrack(Sym *s)
{
	Prog *p;
	
	p = thearch.gins(AUSEFIELD, N, N);
	p->from.type = TYPE_MEM;
	p->from.name = NAME_EXTERN;
	p->from.sym = linksym(s);
}

void
ggloblsym(Sym *s, int32 width, int8 flags)
{
	Prog *p;

	p = thearch.gins(AGLOBL, N, N);
	p->from.type = TYPE_MEM;
	p->from.name = NAME_EXTERN;
	p->from.sym = linksym(s);
	p->to.type = TYPE_CONST;
	p->to.offset = width;
	p->from3.offset = flags;
}

void
clearp(Prog *p)
{
	nopout(p);
	p->as = AEND;
	p->pc = pcloc;
	pcloc++;
}

static int ddumped;
static Prog *dfirst;
static Prog *dpc;

/*
 * generate and return proc with p->as = as,
 * linked into program. pc is next instruction.
 */
Prog*
prog(int as)
{
	Prog *p;

	if(as == ADATA || as == AGLOBL) {
		if(ddumped)
			fatal("already dumped data");
		if(dpc == nil) {
			dpc = mal(sizeof(*dpc));
			dfirst = dpc;
		}
		p = dpc;
		dpc = mal(sizeof(*dpc));
		p->link = dpc;
	} else {
		p = pc;
		pc = mal(sizeof(*pc));
		clearp(pc);
		p->link = pc;
	}

	if(lineno == 0) {
		if(debug['K'])
			warn("prog: line 0");
	}

	p->as = as;
	p->lineno = lineno;
	return p;
}

void
dumpdata(void)
{
	ddumped = 1;
	if(dfirst == nil)
		return;
	newplist();
	*pc = *dfirst;
	pc = dpc;
	clearp(pc);
}

/*
 * generate a branch.
 * t is ignored.
 * likely values are for branch prediction:
 *	-1 unlikely
 *	0 no opinion
 *	+1 likely
 */
Prog*
gbranch(int as, Type *t, int likely)
{
	Prog *p;
	
	USED(t);

	p = prog(as);
	p->to.type = TYPE_BRANCH;
	p->to.u.branch = P;
	if(as != AJMP && likely != 0 && thearch.thechar != '9') {
		p->from.type = TYPE_CONST;
		p->from.offset = likely > 0;
	}
	return p;
}

/*
 * patch previous branch to jump to to.
 */
void
patch(Prog *p, Prog *to)
{
	if(p->to.type != TYPE_BRANCH)
		fatal("patch: not a branch");
	p->to.u.branch = to;
	p->to.offset = to->pc;
}

Prog*
unpatch(Prog *p)
{
	Prog *q;

	if(p->to.type != TYPE_BRANCH)
		fatal("unpatch: not a branch");
	q = p->to.u.branch;
	p->to.u.branch = P;
	p->to.offset = 0;
	return q;
}

/*
 * start a new Prog list.
 */
Plist*
newplist(void)
{
	Plist *pl;

	pl = linknewplist(ctxt);

	pc = mal(sizeof(*pc));
	clearp(pc);
	pl->firstpc = pc;

	return pl;
}

void
gused(Node *n)
{
	thearch.gins(ANOP, n, N);	// used
}

Prog*
gjmp(Prog *to)
{
	Prog *p;

	p = gbranch(AJMP, T, 0);
	if(to != P)
		patch(p, to);
	return p;
}

int
isfat(Type *t)
{
	if(t != T)
	switch(t->etype) {
	case TSTRUCT:
	case TARRAY:
	case TSTRING:
	case TINTER:	// maybe remove later
		return 1;
	}
	return 0;
}

/*
 * naddr of func generates code for address of func.
 * if using opcode that can take address implicitly,
 * call afunclit to fix up the argument.
 */
void
afunclit(Addr *a, Node *n)
{
	if(a->type == TYPE_ADDR && a->name == NAME_EXTERN) {
		a->type = TYPE_MEM;
		a->sym = linksym(n->sym);
	}
}

/*
 * initialize n to be register r of type t.
 */
void
nodreg(Node *n, Type *t, int r)
{
	if(t == T)
		fatal("nodreg: t nil");

	memset(n, 0, sizeof(*n));
	n->op = OREGISTER;
	n->addable = 1;
	ullmancalc(n);
	n->val.u.reg = r;
	n->type = t;
}

/*
 * initialize n to be indirect of register r; n is type t.
 */
void
nodindreg(Node *n, Type *t, int r)
{
	nodreg(n, t, r);
	n->op = OINDREG;
}

/*
 * Is this node a memory operand?
 */
int
ismem(Node *n)
{
	switch(n->op) {
	case OITAB:
	case OSPTR:
	case OLEN:
	case OCAP:
	case OINDREG:
	case ONAME:
	case OPARAM:
	case OCLOSUREVAR:
		return 1;
	case OADDR:
		return thearch.thechar == '6' || thearch.thechar == '9'; // because 6g uses PC-relative addressing; TODO(rsc): not sure why 9g too
	}
	return 0;
}

// Sweep the prog list to mark any used nodes.
void
markautoused(Prog* p)
{
	for (; p; p = p->link) {
		if (p->as == ATYPE || p->as == AVARDEF || p->as == AVARKILL)
			continue;

		if (p->from.node)
			((Node*)(p->from.node))->used = 1;

		if (p->to.node)
			((Node*)(p->to.node))->used = 1;
	}
}

// Fixup instructions after allocauto (formerly compactframe) has moved all autos around.
void
fixautoused(Prog *p)
{
	Prog **lp;

	for (lp=&p; (p=*lp) != P; ) {
		if (p->as == ATYPE && p->from.node && p->from.name == NAME_AUTO && !((Node*)(p->from.node))->used) {
			*lp = p->link;
			continue;
		}
		if ((p->as == AVARDEF || p->as == AVARKILL) && p->to.node && !((Node*)(p->to.node))->used) {
			// Cannot remove VARDEF instruction, because - unlike TYPE handled above -
			// VARDEFs are interspersed with other code, and a jump might be using the
			// VARDEF as a target. Replace with a no-op instead. A later pass will remove
			// the no-ops.
			nopout(p);
			continue;
		}
		if (p->from.name == NAME_AUTO && p->from.node)
			p->from.offset += ((Node*)(p->from.node))->stkdelta;

		if (p->to.name == NAME_AUTO && p->to.node)
			p->to.offset += ((Node*)(p->to.node))->stkdelta;

		lp = &p->link;
	}
}

int
samereg(Node *a, Node *b)
{
	if(a == N || b == N)
		return 0;
	if(a->op != OREGISTER)
		return 0;
	if(b->op != OREGISTER)
		return 0;
	if(a->val.u.reg != b->val.u.reg)
		return 0;
	return 1;
}

Node*
nodarg(Type *t, int fp)
{
	Node *n;
	NodeList *l;
	Type *first;
	Iter savet;

	// entire argument struct, not just one arg
	if(t->etype == TSTRUCT && t->funarg) {
		n = nod(ONAME, N, N);
		n->sym = lookup(".args");
		n->type = t;
		first = structfirst(&savet, &t);
		if(first == nil)
			fatal("nodarg: bad struct");
		if(first->width == BADWIDTH)
			fatal("nodarg: offset not computed for %T", t);
		n->xoffset = first->width;
		n->addable = 1;
		goto fp;
	}

	if(t->etype != TFIELD)
		fatal("nodarg: not field %T", t);
	
	if(fp == 1) {
		for(l=curfn->dcl; l; l=l->next) {
			n = l->n;
			if((n->class == PPARAM || n->class == PPARAMOUT) && !isblanksym(t->sym) && n->sym == t->sym)
				return n;
		}
	}

	n = nod(ONAME, N, N);
	n->type = t->type;
	n->sym = t->sym;
	
	if(t->width == BADWIDTH)
		fatal("nodarg: offset not computed for %T", t);
	n->xoffset = t->width;
	n->addable = 1;
	n->orig = t->nname;

fp:
	// Rewrite argument named _ to __,
	// or else the assignment to _ will be
	// discarded during code generation.
	if(isblank(n))
		n->sym = lookup("__");

	switch(fp) {
	case 0:		// output arg
		n->op = OINDREG;
		n->val.u.reg = thearch.REGSP;
		if(thearch.thechar == '5')
			n->xoffset += 4;
		if(thearch.thechar == '9')
			n->xoffset += 8;
		break;

	case 1:		// input arg
		n->class = PPARAM;
		break;

	case 2:		// offset output arg
fatal("shouldn't be used");
		n->op = OINDREG;
		n->val.u.reg = thearch.REGSP;
		n->xoffset += types[tptr]->width;
		break;
	}
	n->typecheck = 1;
	return n;
}

/*
 * generate code to compute n;
 * make a refer to result.
 */
void
naddr(Node *n, Addr *a, int canemitcode)
{
	Sym *s;

	*a = zprog.from;
	if(n == N)
		return;

	if(n->type != T && n->type->etype != TIDEAL) {
		// TODO(rsc): This is undone by the selective clearing of width below,
		// to match architectures that were not as aggressive in setting width
		// during naddr. Those widths must be cleared to avoid triggering
		// failures in gins when it detects real but heretofore latent (and one
		// hopes innocuous) type mismatches.
		// The type mismatches should be fixed and the clearing below removed.
		dowidth(n->type);
		a->width = n->type->width;
	}

	switch(n->op) {
	default:
		fatal("naddr: bad %O %D", n->op, a);
		break;

	case OREGISTER:
		a->type = TYPE_REG;
		a->reg = n->val.u.reg;
		a->sym = nil;
		if(thearch.thechar == '8') // TODO(rsc): Never clear a->width.
			a->width = 0;
		break;

	case OINDREG:
		a->type = TYPE_MEM;
		a->reg = n->val.u.reg;
		a->sym = linksym(n->sym);
		a->offset = n->xoffset;
		if(a->offset != (int32)a->offset)
			yyerror("offset %lld too large for OINDREG", a->offset);
		if(thearch.thechar == '8') // TODO(rsc): Never clear a->width.
			a->width = 0;
		break;

	case OPARAM:
		// n->left is PHEAP ONAME for stack parameter.
		// compute address of actual parameter on stack.
		a->etype = simtype[n->left->type->etype];
		a->width = n->left->type->width;
		a->offset = n->xoffset;
		a->sym = linksym(n->left->sym);
		a->type = TYPE_MEM;
		a->name = NAME_PARAM;
		a->node = n->left->orig;
		break;
	
	case OCLOSUREVAR:
		if(!curfn->needctxt)
			fatal("closurevar without needctxt");
		a->type = TYPE_MEM;
		a->reg = thearch.REGCTXT;
		a->sym = nil;
		a->offset = n->xoffset;
		break;
	
	case OCFUNC:
		naddr(n->left, a, canemitcode);
		a->sym = linksym(n->left->sym);
		break;

	case ONAME:
		a->etype = 0;
		if(n->type != T)
			a->etype = simtype[n->type->etype];
		a->offset = n->xoffset;
		s = n->sym;
		a->node = n->orig;
		//if(a->node >= (Node*)&n)
		//	fatal("stack node");
		if(s == S)
			s = lookup(".noname");
		if(n->method) {
			if(n->type != T)
			if(n->type->sym != S)
			if(n->type->sym->pkg != nil)
				s = pkglookup(s->name, n->type->sym->pkg);
		}

		a->type = TYPE_MEM;
		switch(n->class) {
		default:
			fatal("naddr: ONAME class %S %d\n", n->sym, n->class);
		case PEXTERN:
			a->name = NAME_EXTERN;
			break;
		case PAUTO:
			a->name = NAME_AUTO;
			break;
		case PPARAM:
		case PPARAMOUT:
			a->name = NAME_PARAM;
			break;
		case PFUNC:
			a->name = NAME_EXTERN;
			a->type = TYPE_ADDR;
			a->width = widthptr;
			s = funcsym(s);			
			break;
		}
		a->sym = linksym(s);
		break;

	case OLITERAL:
		if(thearch.thechar == '8')
			a->width = 0;
		switch(n->val.ctype) {
		default:
			fatal("naddr: const %lT", n->type);
			break;
		case CTFLT:
			a->type = TYPE_FCONST;
			a->u.dval = mpgetflt(n->val.u.fval);
			break;
		case CTINT:
		case CTRUNE:
			a->sym = nil;
			a->type = TYPE_CONST;
			a->offset = mpgetfix(n->val.u.xval);
			break;
		case CTSTR:
			datagostring(n->val.u.sval, a);
			break;
		case CTBOOL:
			a->sym = nil;
			a->type = TYPE_CONST;
			a->offset = n->val.u.bval;
			break;
		case CTNIL:
			a->sym = nil;
			a->type = TYPE_CONST;
			a->offset = 0;
			break;
		}
		break;

	case OADDR:
		naddr(n->left, a, canemitcode);
		a->etype = tptr;
		if(thearch.thechar != '5' && thearch.thechar != '9') // TODO(rsc): Do this even for arm, ppc64.
			a->width = widthptr;
		if(a->type != TYPE_MEM)
			fatal("naddr: OADDR %D (from %O)", a, n->left->op);
		a->type = TYPE_ADDR;
		break;
	
	case OITAB:
		// itable of interface value
		naddr(n->left, a, canemitcode);
		if(a->type == TYPE_CONST && a->offset == 0)
			break;  // itab(nil)
		a->etype = tptr;
		a->width = widthptr;
		break;

	case OSPTR:
		// pointer in a string or slice
		naddr(n->left, a, canemitcode);
		if(a->type == TYPE_CONST && a->offset == 0)
			break;	// ptr(nil)
		a->etype = simtype[tptr];
		a->offset += Array_array;
		a->width = widthptr;
		break;

	case OLEN:
		// len of string or slice
		naddr(n->left, a, canemitcode);
		if(a->type == TYPE_CONST && a->offset == 0)
			break;	// len(nil)
		a->etype = simtype[TUINT];
		if(thearch.thechar == '9')
			a->etype = simtype[TINT];
		a->offset += Array_nel;
		if(thearch.thechar != '5') // TODO(rsc): Do this even on arm.
			a->width = widthint;
		break;

	case OCAP:
		// cap of string or slice
		naddr(n->left, a, canemitcode);
		if(a->type == TYPE_CONST && a->offset == 0)
			break;	// cap(nil)
		a->etype = simtype[TUINT];
		if(thearch.thechar == '9')
			a->etype = simtype[TINT];
		a->offset += Array_cap;
		if(thearch.thechar != '5') // TODO(rsc): Do this even on arm.
			a->width = widthint;
		break;

//	case OADD:
//		if(n->right->op == OLITERAL) {
//			v = n->right->vconst;
//			naddr(n->left, a, canemitcode);
//		} else
//		if(n->left->op == OLITERAL) {
//			v = n->left->vconst;
//			naddr(n->right, a, canemitcode);
//		} else
//			goto bad;
//		a->offset += v;
//		break;

	}
}
