// Derived from Inferno utils/5c/txt.c
// http://code.google.com/p/inferno-os/source/browse/utils/5c/txt.c
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
#include "gg.h"
#include "../../runtime/funcdata.h"

// TODO(rsc): Can make this bigger if we move
// the text segment up higher in 5l for all GOOS.
// At the same time, can raise StackBig in ../../runtime/stack.h.
long unmappedzero = 4096;

void
clearp(Prog *p)
{
	p->as = AEND;
	p->reg = NREG;
	p->scond = C_SCOND_NONE;
	p->from.type = D_NONE;
	p->from.name = D_NONE;
	p->from.reg = NREG;
	p->to.type = D_NONE;
	p->to.name = D_NONE;
	p->to.reg = NREG;
	p->pc = pcloc;
	pcloc++;
}

static int ddumped;
static Prog *dfirst;
static Prog *dpc;

/*
 * generate and return proc with p->as = as,
 * linked into program.  pc is next instruction.
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
		p->reg = 0;  // used for flags
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
	USED(likely);  // TODO: record this for linker

	p = prog(as);
	p->to.type = D_BRANCH;
	p->to.u.branch = P;
	return p;
}

/*
 * patch previous branch to jump to to.
 */
void
patch(Prog *p, Prog *to)
{
	if(p->to.type != D_BRANCH)
		fatal("patch: not a branch");
	p->to.u.branch = to;
	p->to.offset = to->pc;
}

Prog*
unpatch(Prog *p)
{
	Prog *q;

	if(p->to.type != D_BRANCH)
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
	gins(ANOP, n, N);	// used
}

Prog*
gjmp(Prog *to)
{
	Prog *p;

	p = gbranch(AB, T, 0);
	if(to != P)
		patch(p, to);
	return p;
}

void
ggloblnod(Node *nam)
{
	Prog *p;

	p = gins(AGLOBL, nam, N);
	p->lineno = nam->lineno;
	p->from.sym->gotype = linksym(ngotype(nam));
	p->to.sym = nil;
	p->to.type = D_CONST;
	p->to.offset = nam->type->width;
	if(nam->readonly)
		p->reg = RODATA;
	if(nam->type != T && !haspointers(nam->type))
		p->reg |= NOPTR;
}

void
ggloblsym(Sym *s, int32 width, int8 flags)
{
	Prog *p;

	p = gins(AGLOBL, N, N);
	p->from.type = D_OREG;
	p->from.name = D_EXTERN;
	p->from.sym = linksym(s);
	p->to.type = D_CONST;
	p->to.name = D_NONE;
	p->to.offset = width;
	p->reg = flags;
}

void
gtrack(Sym *s)
{
	Prog *p;
	
	p = gins(AUSEFIELD, N, N);
	p->from.type = D_OREG;
	p->from.name = D_EXTERN;
	p->from.sym = linksym(s);
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
 * also fix up direct register references to be D_OREG.
 */
void
afunclit(Addr *a, Node *n)
{
	if(a->type == D_CONST && a->name == D_EXTERN || a->type == D_REG) {
		a->type = D_OREG;
		if(n->op == ONAME)
			a->sym = linksym(n->sym);
	}
}

static	int	resvd[] =
{
	9,     // reserved for m
	10,    // reserved for g
	REGSP, // reserved for SP
};

void
ginit(void)
{
	int i;

	for(i=0; i<nelem(reg); i++)
		reg[i] = 0;
	for(i=0; i<nelem(resvd); i++)
		reg[resvd[i]]++;
}

void
gclean(void)
{
	int i;

	for(i=0; i<nelem(resvd); i++)
		reg[resvd[i]]--;

	for(i=0; i<nelem(reg); i++)
		if(reg[i])
			yyerror("reg %R left allocated\n", i);
}

int32
anyregalloc(void)
{
	int i, j;

	for(i=0; i<nelem(reg); i++) {
		if(reg[i] == 0)
			goto ok;
		for(j=0; j<nelem(resvd); j++)
			if(resvd[j] == i)
				goto ok;
		return 1;
	ok:;
	}
	return 0;
}

uintptr regpc[REGALLOC_FMAX+1];

/*
 * allocate register of type t, leave in n.
 * if o != N, o is desired fixed register.
 * caller must regfree(n).
 */
void
regalloc(Node *n, Type *t, Node *o)
{
	int i, et, fixfree, floatfree;

	if(0 && debug['r']) {
		fixfree = 0;
		for(i=REGALLOC_R0; i<=REGALLOC_RMAX; i++)
			if(reg[i] == 0)
				fixfree++;
		floatfree = 0;
		for(i=REGALLOC_F0; i<=REGALLOC_FMAX; i++)
			if(reg[i] == 0)
				floatfree++;
		print("regalloc fix %d float %d\n", fixfree, floatfree);
	}

	if(t == T)
		fatal("regalloc: t nil");
	et = simtype[t->etype];
	if(is64(t))
		fatal("regalloc: 64 bit type %T");

	switch(et) {
	case TINT8:
	case TUINT8:
	case TINT16:
	case TUINT16:
	case TINT32:
	case TUINT32:
	case TPTR32:
	case TBOOL:
		if(o != N && o->op == OREGISTER) {
			i = o->val.u.reg;
			if(i >= REGALLOC_R0 && i <= REGALLOC_RMAX)
				goto out;
		}
		for(i=REGALLOC_R0; i<=REGALLOC_RMAX; i++)
			if(reg[i] == 0) {
				regpc[i] = (uintptr)getcallerpc(&n);
				goto out;
			}
		print("registers allocated at\n");
		for(i=REGALLOC_R0; i<=REGALLOC_RMAX; i++)
			print("%d %p\n", i, regpc[i]);
		fatal("out of fixed registers");
		goto err;

	case TFLOAT32:
	case TFLOAT64:
		if(o != N && o->op == OREGISTER) {
			i = o->val.u.reg;
			if(i >= REGALLOC_F0 && i <= REGALLOC_FMAX)
				goto out;
		}
		for(i=REGALLOC_F0; i<=REGALLOC_FMAX; i++)
			if(reg[i] == 0)
				goto out;
		fatal("out of floating point registers");
		goto err;

	case TCOMPLEX64:
	case TCOMPLEX128:
		tempname(n, t);
		return;
	}
	yyerror("regalloc: unknown type %T", t);

err:
	nodreg(n, t, 0);
	return;

out:
	reg[i]++;
	nodreg(n, t, i);
}

void
regfree(Node *n)
{
	int i, fixfree, floatfree;

	if(0 && debug['r']) {
		fixfree = 0;
		for(i=REGALLOC_R0; i<=REGALLOC_RMAX; i++)
			if(reg[i] == 0)
				fixfree++;
		floatfree = 0;
		for(i=REGALLOC_F0; i<=REGALLOC_FMAX; i++)
			if(reg[i] == 0)
				floatfree++;
		print("regalloc fix %d float %d\n", fixfree, floatfree);
	}

	if(n->op == ONAME)
		return;
	if(n->op != OREGISTER && n->op != OINDREG)
		fatal("regfree: not a register");
	i = n->val.u.reg;
	if(i == REGSP)
		return;
	if(i < 0 || i >= nelem(reg) || i >= nelem(regpc))
		fatal("regfree: reg out of range");
	if(reg[i] <= 0)
		fatal("regfree: reg %R not allocated", i);
	reg[i]--;
	if(reg[i] == 0)
		regpc[i] = 0;
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
	default:
		fatal("nodarg %T %d", t, fp);

	case 0:		// output arg for calling another function
		n->op = OINDREG;
		n->val.u.reg = REGSP;
		n->xoffset += 4;
		break;

	case 1:		// input arg to current function
		n->class = PPARAM;
		break;
	}
	n->typecheck = 1;
	return n;
}

/*
 * return constant i node.
 * overwritten by next call, but useful in calls to gins.
 */
Node*
ncon(uint32 i)
{
	static Node n;

	if(n.type == T)
		nodconst(&n, types[TUINT32], 0);
	mpmovecfix(n.val.u.xval, i);
	return &n;
}

/*
 * Is this node a memory operand?
 */
int
ismem(Node *n)
{
	switch(n->op) {
	case OINDREG:
	case ONAME:
	case OPARAM:
	case OCLOSUREVAR:
		return 1;
	}
	return 0;
}

Node sclean[10];
int nsclean;

/*
 * n is a 64-bit value.  fill in lo and hi to refer to its 32-bit halves.
 */
void
split64(Node *n, Node *lo, Node *hi)
{
	Node n1;
	int64 i;

	if(!is64(n->type))
		fatal("split64 %T", n->type);

	if(nsclean >= nelem(sclean))
		fatal("split64 clean");
	sclean[nsclean].op = OEMPTY;
	nsclean++;
	switch(n->op) {
	default:
		if(!dotaddable(n, &n1)) {
			igen(n, &n1, N);
			sclean[nsclean-1] = n1;
		}
		n = &n1;
		goto common;
	case ONAME:
		if(n->class == PPARAMREF) {
			cgen(n->heapaddr, &n1);
			sclean[nsclean-1] = n1;
			// fall through.
			n = &n1;
		}
		goto common;
	case OINDREG:
	common:
		*lo = *n;
		*hi = *n;
		lo->type = types[TUINT32];
		if(n->type->etype == TINT64)
			hi->type = types[TINT32];
		else
			hi->type = types[TUINT32];
		hi->xoffset += 4;
		break;

	case OLITERAL:
		convconst(&n1, n->type, &n->val);
		i = mpgetfix(n1.val.u.xval);
		nodconst(lo, types[TUINT32], (uint32)i);
		i >>= 32;
		if(n->type->etype == TINT64)
			nodconst(hi, types[TINT32], (int32)i);
		else
			nodconst(hi, types[TUINT32], (uint32)i);
		break;
	}
}

void
splitclean(void)
{
	if(nsclean <= 0)
		fatal("splitclean");
	nsclean--;
	if(sclean[nsclean].op != OEMPTY)
		regfree(&sclean[nsclean]);
}

#define	CASE(a,b)	(((a)<<16)|((b)<<0))
/*c2go int CASE(int, int); */

void
gmove(Node *f, Node *t)
{
	int a, ft, tt, fa, ta;
	Type *cvt;
	Node r1, r2, flo, fhi, tlo, thi, con;
	Prog *p1;

	if(debug['M'])
		print("gmove %N -> %N\n", f, t);

	ft = simsimtype(f->type);
	tt = simsimtype(t->type);
	cvt = t->type;

	if(iscomplex[ft] || iscomplex[tt]) {
		complexmove(f, t);
		return;
	}

	// cannot have two memory operands;
	// except 64-bit, which always copies via registers anyway.
	if(!is64(f->type) && !is64(t->type) && ismem(f) && ismem(t))
		goto hard;

	// convert constant to desired type
	if(f->op == OLITERAL) {
		switch(tt) {
		default:
			convconst(&con, t->type, &f->val);
			break;

		case TINT16:
		case TINT8:
			convconst(&con, types[TINT32], &f->val);
			regalloc(&r1, con.type, t);
			gins(AMOVW, &con, &r1);
			gmove(&r1, t);
			regfree(&r1);
			return;

		case TUINT16:
		case TUINT8:
			convconst(&con, types[TUINT32], &f->val);
			regalloc(&r1, con.type, t);
			gins(AMOVW, &con, &r1);
			gmove(&r1, t);
			regfree(&r1);
			return;
		}

		f = &con;
		ft = simsimtype(con.type);

		// constants can't move directly to memory
		if(ismem(t) && !is64(t->type)) goto hard;
	}

	// value -> value copy, only one memory operand.
	// figure out the instruction to use.
	// break out of switch for one-instruction gins.
	// goto rdst for "destination must be register".
	// goto hard for "convert to cvt type first".
	// otherwise handle and return.

	switch(CASE(ft, tt)) {
	default:
		goto fatal;

	/*
	 * integer copy and truncate
	 */
	case CASE(TINT8, TINT8):	// same size
		if(!ismem(f)) {
			a = AMOVB;
			break;
		}
	case CASE(TUINT8, TINT8):
	case CASE(TINT16, TINT8):	// truncate
	case CASE(TUINT16, TINT8):
	case CASE(TINT32, TINT8):
	case CASE(TUINT32, TINT8):
		a = AMOVBS;
		break;

	case CASE(TUINT8, TUINT8):
		if(!ismem(f)) {
			a = AMOVB;
			break;
		}
	case CASE(TINT8, TUINT8):
	case CASE(TINT16, TUINT8):
	case CASE(TUINT16, TUINT8):
	case CASE(TINT32, TUINT8):
	case CASE(TUINT32, TUINT8):
		a = AMOVBU;
		break;

	case CASE(TINT64, TINT8):	// truncate low word
	case CASE(TUINT64, TINT8):
		a = AMOVBS;
		goto trunc64;

	case CASE(TINT64, TUINT8):
	case CASE(TUINT64, TUINT8):
		a = AMOVBU;
		goto trunc64;

	case CASE(TINT16, TINT16):	// same size
		if(!ismem(f)) {
			a = AMOVH;
			break;
		}
	case CASE(TUINT16, TINT16):
	case CASE(TINT32, TINT16):	// truncate
	case CASE(TUINT32, TINT16):
		a = AMOVHS;
		break;

	case CASE(TUINT16, TUINT16):
		if(!ismem(f)) {
			a = AMOVH;
			break;
		}
	case CASE(TINT16, TUINT16):
	case CASE(TINT32, TUINT16):
	case CASE(TUINT32, TUINT16):
		a = AMOVHU;
		break;

	case CASE(TINT64, TINT16):	// truncate low word
	case CASE(TUINT64, TINT16):
		a = AMOVHS;
		goto trunc64;

	case CASE(TINT64, TUINT16):
	case CASE(TUINT64, TUINT16):
		a = AMOVHU;
		goto trunc64;

	case CASE(TINT32, TINT32):	// same size
	case CASE(TINT32, TUINT32):
	case CASE(TUINT32, TINT32):
	case CASE(TUINT32, TUINT32):
		a = AMOVW;
		break;

	case CASE(TINT64, TINT32):	// truncate
	case CASE(TUINT64, TINT32):
	case CASE(TINT64, TUINT32):
	case CASE(TUINT64, TUINT32):
		split64(f, &flo, &fhi);
		regalloc(&r1, t->type, N);
		gins(AMOVW, &flo, &r1);
		gins(AMOVW, &r1, t);
		regfree(&r1);
		splitclean();
		return;

	case CASE(TINT64, TINT64):	// same size
	case CASE(TINT64, TUINT64):
	case CASE(TUINT64, TINT64):
	case CASE(TUINT64, TUINT64):
		split64(f, &flo, &fhi);
		split64(t, &tlo, &thi);
		regalloc(&r1, flo.type, N);
		regalloc(&r2, fhi.type, N);
		gins(AMOVW, &flo, &r1);
		gins(AMOVW, &fhi, &r2);
		gins(AMOVW, &r1, &tlo);
		gins(AMOVW, &r2, &thi);
		regfree(&r1);
		regfree(&r2);
		splitclean();
		splitclean();
		return;

	/*
	 * integer up-conversions
	 */
	case CASE(TINT8, TINT16):	// sign extend int8
	case CASE(TINT8, TUINT16):
	case CASE(TINT8, TINT32):
	case CASE(TINT8, TUINT32):
		a = AMOVBS;
		goto rdst;
	case CASE(TINT8, TINT64):	// convert via int32
	case CASE(TINT8, TUINT64):
		cvt = types[TINT32];
		goto hard;

	case CASE(TUINT8, TINT16):	// zero extend uint8
	case CASE(TUINT8, TUINT16):
	case CASE(TUINT8, TINT32):
	case CASE(TUINT8, TUINT32):
		a = AMOVBU;
		goto rdst;
	case CASE(TUINT8, TINT64):	// convert via uint32
	case CASE(TUINT8, TUINT64):
		cvt = types[TUINT32];
		goto hard;

	case CASE(TINT16, TINT32):	// sign extend int16
	case CASE(TINT16, TUINT32):
		a = AMOVHS;
		goto rdst;
	case CASE(TINT16, TINT64):	// convert via int32
	case CASE(TINT16, TUINT64):
		cvt = types[TINT32];
		goto hard;

	case CASE(TUINT16, TINT32):	// zero extend uint16
	case CASE(TUINT16, TUINT32):
		a = AMOVHU;
		goto rdst;
	case CASE(TUINT16, TINT64):	// convert via uint32
	case CASE(TUINT16, TUINT64):
		cvt = types[TUINT32];
		goto hard;

	case CASE(TINT32, TINT64):	// sign extend int32
	case CASE(TINT32, TUINT64):
		split64(t, &tlo, &thi);
		regalloc(&r1, tlo.type, N);
		regalloc(&r2, thi.type, N);
		gmove(f, &r1);
		p1 = gins(AMOVW, &r1, &r2);
		p1->from.type = D_SHIFT;
		p1->from.offset = 2 << 5 | 31 << 7 | r1.val.u.reg; // r1->31
		p1->from.reg = NREG;
//print("gmove: %P\n", p1);
		gins(AMOVW, &r1, &tlo);
		gins(AMOVW, &r2, &thi);
		regfree(&r1);
		regfree(&r2);
		splitclean();
		return;

	case CASE(TUINT32, TINT64):	// zero extend uint32
	case CASE(TUINT32, TUINT64):
		split64(t, &tlo, &thi);
		gmove(f, &tlo);
		regalloc(&r1, thi.type, N);
		gins(AMOVW, ncon(0), &r1);
		gins(AMOVW, &r1, &thi);
		regfree(&r1);
		splitclean();
		return;

	/*
	* float to integer
	*/
	case CASE(TFLOAT32, TINT8):
	case CASE(TFLOAT32, TUINT8):
	case CASE(TFLOAT32, TINT16):
	case CASE(TFLOAT32, TUINT16):
	case CASE(TFLOAT32, TINT32):
	case CASE(TFLOAT32, TUINT32):
//	case CASE(TFLOAT32, TUINT64):

	case CASE(TFLOAT64, TINT8):
	case CASE(TFLOAT64, TUINT8):
	case CASE(TFLOAT64, TINT16):
	case CASE(TFLOAT64, TUINT16):
	case CASE(TFLOAT64, TINT32):
	case CASE(TFLOAT64, TUINT32):
//	case CASE(TFLOAT64, TUINT64):
		fa = AMOVF;
		a = AMOVFW;
		if(ft == TFLOAT64) {
			fa = AMOVD;
			a = AMOVDW;
		}
		ta = AMOVW;
		switch(tt) {
		case TINT8:
			ta = AMOVBS;
			break;
		case TUINT8:
			ta = AMOVBU;
			break;
		case TINT16:
			ta = AMOVHS;
			break;
		case TUINT16:
			ta = AMOVHU;
			break;
		}

		regalloc(&r1, types[ft], f);
		regalloc(&r2, types[tt], t);
		gins(fa, f, &r1);	// load to fpu
		p1 = gins(a, &r1, &r1);	// convert to w
		switch(tt) {
		case TUINT8:
		case TUINT16:
		case TUINT32:
			p1->scond |= C_UBIT;
		}
		gins(AMOVW, &r1, &r2);	// copy to cpu
		gins(ta, &r2, t);	// store
		regfree(&r1);
		regfree(&r2);
		return;

	/*
	 * integer to float
	 */
	case CASE(TINT8, TFLOAT32):
	case CASE(TUINT8, TFLOAT32):
	case CASE(TINT16, TFLOAT32):
	case CASE(TUINT16, TFLOAT32):
	case CASE(TINT32, TFLOAT32):
	case CASE(TUINT32, TFLOAT32):
	case CASE(TINT8, TFLOAT64):
	case CASE(TUINT8, TFLOAT64):
	case CASE(TINT16, TFLOAT64):
	case CASE(TUINT16, TFLOAT64):
	case CASE(TINT32, TFLOAT64):
	case CASE(TUINT32, TFLOAT64):
		fa = AMOVW;
		switch(ft) {
		case TINT8:
			fa = AMOVBS;
			break;
		case TUINT8:
			fa = AMOVBU;
			break;
		case TINT16:
			fa = AMOVHS;
			break;
		case TUINT16:
			fa = AMOVHU;
			break;
		}
		a = AMOVWF;
		ta = AMOVF;
		if(tt == TFLOAT64) {
			a = AMOVWD;
			ta = AMOVD;
		}
		regalloc(&r1, types[ft], f);
		regalloc(&r2, types[tt], t);
		gins(fa, f, &r1);	// load to cpu
		gins(AMOVW, &r1, &r2);	// copy to fpu
		p1 = gins(a, &r2, &r2);	// convert
		switch(ft) {
		case TUINT8:
		case TUINT16:
		case TUINT32:
			p1->scond |= C_UBIT;
		}
		gins(ta, &r2, t);	// store
		regfree(&r1);
		regfree(&r2);
		return;

	case CASE(TUINT64, TFLOAT32):
	case CASE(TUINT64, TFLOAT64):
		fatal("gmove UINT64, TFLOAT not implemented");
		return;


	/*
	 * float to float
	 */
	case CASE(TFLOAT32, TFLOAT32):
		a = AMOVF;
		break;

	case CASE(TFLOAT64, TFLOAT64):
		a = AMOVD;
		break;

	case CASE(TFLOAT32, TFLOAT64):
		regalloc(&r1, types[TFLOAT64], t);
		gins(AMOVF, f, &r1);
		gins(AMOVFD, &r1, &r1);
		gins(AMOVD, &r1, t);
		regfree(&r1);
		return;

	case CASE(TFLOAT64, TFLOAT32):
		regalloc(&r1, types[TFLOAT64], t);
		gins(AMOVD, f, &r1);
		gins(AMOVDF, &r1, &r1);
		gins(AMOVF, &r1, t);
		regfree(&r1);
		return;
	}

	gins(a, f, t);
	return;

rdst:
	// TODO(kaib): we almost always require a register dest anyway, this can probably be
	// removed.
	// requires register destination
	regalloc(&r1, t->type, t);
	gins(a, f, &r1);
	gmove(&r1, t);
	regfree(&r1);
	return;

hard:
	// requires register intermediate
	regalloc(&r1, cvt, t);
	gmove(f, &r1);
	gmove(&r1, t);
	regfree(&r1);
	return;

trunc64:
	// truncate 64 bit integer
	split64(f, &flo, &fhi);
	regalloc(&r1, t->type, N);
	gins(a, &flo, &r1);
	gins(a, &r1, t);
	regfree(&r1);
	splitclean();
	return;

fatal:
	// should not happen
	fatal("gmove %N -> %N", f, t);
}

int
samaddr(Node *f, Node *t)
{

	if(f->op != t->op)
		return 0;

	switch(f->op) {
	case OREGISTER:
		if(f->val.u.reg != t->val.u.reg)
			break;
		return 1;
	}
	return 0;
}

/*
 * generate one instruction:
 *	as f, t
 */
Prog*
gins(int as, Node *f, Node *t)
{
//	Node nod;
//	int32 v;
	Prog *p;
	Addr af, at;

	if(f != N && f->op == OINDEX) {
		fatal("gins OINDEX not implemented");
//		regalloc(&nod, &regnode, Z);
//		v = constnode.vconst;
//		cgen(f->right, &nod);
//		constnode.vconst = v;
//		idx.reg = nod.reg;
//		regfree(&nod);
	}
	if(t != N && t->op == OINDEX) {
		fatal("gins OINDEX not implemented");
//		regalloc(&nod, &regnode, Z);
//		v = constnode.vconst;
//		cgen(t->right, &nod);
//		constnode.vconst = v;
//		idx.reg = nod.reg;
//		regfree(&nod);
	}

	memset(&af, 0, sizeof af);
	memset(&at, 0, sizeof at);
	if(f != N)
		naddr(f, &af, 1);
	if(t != N)
		naddr(t, &at, 1);
	p = prog(as);
	if(f != N)
		p->from = af;
	if(t != N)
		p->to = at;
	if(debug['g'])
		print("%P\n", p);
	return p;
}

/*
 * insert n into reg slot of p
 */
void
raddr(Node *n, Prog *p)
{
	Addr a;

	naddr(n, &a, 1);
	if(a.type != D_REG && a.type != D_FREG) {
		if(n)
			fatal("bad in raddr: %O", n->op);
		else
			fatal("bad in raddr: <null>");
		p->reg = NREG;
	} else
		p->reg = a.reg;
}

/* generate a comparison
TODO(kaib): one of the args can actually be a small constant. relax the constraint and fix call sites.
 */
Prog*
gcmp(int as, Node *lhs, Node *rhs)
{
	Prog *p;

	if(lhs->op != OREGISTER)
		fatal("bad operands to gcmp: %O %O", lhs->op, rhs->op);

	p = gins(as, rhs, N);
	raddr(lhs, p);
	return p;
}

/* generate a constant shift
 * arm encodes a shift by 32 as 0, thus asking for 0 shift is illegal.
*/
Prog*
gshift(int as, Node *lhs, int32 stype, int32 sval, Node *rhs)
{
	Prog *p;

	if(sval <= 0 || sval > 32)
		fatal("bad shift value: %d", sval);

	sval = sval&0x1f;

	p = gins(as, N, rhs);
	p->from.type = D_SHIFT;
	p->from.offset = stype | sval<<7 | lhs->val.u.reg;
	return p;
}

/* generate a register shift
*/
Prog *
gregshift(int as, Node *lhs, int32 stype, Node *reg, Node *rhs)
{
	Prog *p;
	p = gins(as, N, rhs);
	p->from.type = D_SHIFT;
	p->from.offset = stype | reg->val.u.reg << 8 | 1<<4 | lhs->val.u.reg;
	return p;
}

/*
 * generate code to compute n;
 * make a refer to result.
 */
void
naddr(Node *n, Addr *a, int canemitcode)
{
	Sym *s;

	a->type = D_NONE;
	a->name = D_NONE;
	a->reg = NREG;
	a->gotype = nil;
	a->node = N;
	a->etype = 0;
	if(n == N)
		return;

	if(n->type != T && n->type->etype != TIDEAL) {
		dowidth(n->type);
		a->width = n->type->width;
	}

	switch(n->op) {
	default:
		fatal("naddr: bad %O %D", n->op, a);
		break;

	case OREGISTER:
		if(n->val.u.reg <= REGALLOC_RMAX) {
			a->type = D_REG;
			a->reg = n->val.u.reg;
		} else {
			a->type = D_FREG;
			a->reg = n->val.u.reg - REGALLOC_F0;
		}
		a->sym = nil;
		break;

	case OINDEX:
	case OIND:
		fatal("naddr: OINDEX");
//		naddr(n->left, a);
//		if(a->type >= D_AX && a->type <= D_DI)
//			a->type += D_INDIR;
//		else
//		if(a->type == D_CONST)
//			a->type = D_NONE+D_INDIR;
//		else
//		if(a->type == D_ADDR) {
//			a->type = a->index;
//			a->index = D_NONE;
//		} else
//			goto bad;
//		if(n->op == OINDEX) {
//			a->index = idx.reg;
//			a->scale = n->scale;
//		}
//		break;

	case OINDREG:
		a->type = D_OREG;
		a->reg = n->val.u.reg;
		a->sym = linksym(n->sym);
		a->offset = n->xoffset;
		break;

	case OPARAM:
		// n->left is PHEAP ONAME for stack parameter.
		// compute address of actual parameter on stack.
		a->etype = simtype[n->left->type->etype];
		a->width = n->left->type->width;
		a->offset = n->xoffset;
		a->sym = linksym(n->left->sym);
		a->type = D_OREG;
		a->name = D_PARAM;
		a->node = n->left->orig;
		break;
	
	case OCLOSUREVAR:
		if(!curfn->needctxt)
			fatal("closurevar without needctxt");
		a->type = D_OREG;
		a->reg = 7;
		a->offset = n->xoffset;
		a->sym = nil;
		break;		

	case OCFUNC:
		naddr(n->left, a, canemitcode);
		a->sym = linksym(n->left->sym);
		break;

	case ONAME:
		a->etype = 0;
		a->width = 0;
		a->reg = NREG;
		if(n->type != T) {
			a->etype = simtype[n->type->etype];
			a->width = n->type->width;
		}
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

		a->type = D_OREG;
		switch(n->class) {
		default:
			fatal("naddr: ONAME class %S %d\n", n->sym, n->class);
		case PEXTERN:
			a->name = D_EXTERN;
			break;
		case PAUTO:
			a->name = D_AUTO;
			break;
		case PPARAM:
		case PPARAMOUT:
			a->name = D_PARAM;
			break;
		case PFUNC:
			a->name = D_EXTERN;
			a->type = D_CONST;
			s = funcsym(s);
			break;
		}
		a->sym = linksym(s);
		break;

	case OLITERAL:
		switch(n->val.ctype) {
		default:
			fatal("naddr: const %lT", n->type);
			break;
		case CTFLT:
			a->type = D_FCONST;
			a->u.dval = mpgetflt(n->val.u.fval);
			break;
		case CTINT:
		case CTRUNE:
			a->sym = nil;
			a->type = D_CONST;
			a->offset = mpgetfix(n->val.u.xval);
			break;
		case CTSTR:
			datagostring(n->val.u.sval, a);
			break;
		case CTBOOL:
			a->sym = nil;
			a->type = D_CONST;
			a->offset = n->val.u.bval;
			break;
		case CTNIL:
			a->sym = nil;
			a->type = D_CONST;
			a->offset = 0;
			break;
		}
		break;

	case OITAB:
		// itable of interface value
		naddr(n->left, a, canemitcode);
		a->etype = TINT32;
		if(a->type == D_CONST && a->offset == 0)
			break;	// len(nil)
		break;

	case OSPTR:
		// pointer in a string or slice
		naddr(n->left, a, canemitcode);
		if(a->type == D_CONST && a->offset == 0)
			break;	// ptr(nil)
		a->etype = simtype[tptr];
		a->offset += Array_array;
		a->width = widthptr;
		break;

	case OLEN:
		// len of string or slice
		naddr(n->left, a, canemitcode);
		a->etype = TINT32;
		if(a->type == D_CONST && a->offset == 0)
			break;	// len(nil)
		a->offset += Array_nel;
		break;

	case OCAP:
		// cap of string or slice
		naddr(n->left, a, canemitcode);
		a->etype = TINT32;
		if(a->type == D_CONST && a->offset == 0)
			break;	// cap(nil)
		a->offset += Array_cap;
		break;

	case OADDR:
		naddr(n->left, a, canemitcode);
		a->etype = tptr;
		switch(a->type) {
		case D_OREG:
			a->type = D_CONST;
			break;

		case D_REG:
		case D_CONST:
			break;
		
		default:
			fatal("naddr: OADDR %d\n", a->type);
		}
	}
	
	if(a->width < 0)
		fatal("naddr: bad width for %N -> %D", n, a);
}

/*
 * return Axxx for Oxxx on type t.
 */
int
optoas(int op, Type *t)
{
	int a;

	if(t == T)
		fatal("optoas: t is nil");

	a = AGOK;
	switch(CASE(op, simtype[t->etype])) {
	default:
		fatal("optoas: no entry %O-%T etype %T simtype %T", op, t, types[t->etype], types[simtype[t->etype]]);
		break;

/*	case CASE(OADDR, TPTR32):
		a = ALEAL;
		break;

	case CASE(OADDR, TPTR64):
		a = ALEAQ;
		break;
*/
	// TODO(kaib): make sure the conditional branches work on all edge cases
	case CASE(OEQ, TBOOL):
	case CASE(OEQ, TINT8):
	case CASE(OEQ, TUINT8):
	case CASE(OEQ, TINT16):
	case CASE(OEQ, TUINT16):
	case CASE(OEQ, TINT32):
	case CASE(OEQ, TUINT32):
	case CASE(OEQ, TINT64):
	case CASE(OEQ, TUINT64):
	case CASE(OEQ, TPTR32):
	case CASE(OEQ, TPTR64):
	case CASE(OEQ, TFLOAT32):
	case CASE(OEQ, TFLOAT64):
		a = ABEQ;
		break;

	case CASE(ONE, TBOOL):
	case CASE(ONE, TINT8):
	case CASE(ONE, TUINT8):
	case CASE(ONE, TINT16):
	case CASE(ONE, TUINT16):
	case CASE(ONE, TINT32):
	case CASE(ONE, TUINT32):
	case CASE(ONE, TINT64):
	case CASE(ONE, TUINT64):
	case CASE(ONE, TPTR32):
	case CASE(ONE, TPTR64):
	case CASE(ONE, TFLOAT32):
	case CASE(ONE, TFLOAT64):
		a = ABNE;
		break;

	case CASE(OLT, TINT8):
	case CASE(OLT, TINT16):
	case CASE(OLT, TINT32):
	case CASE(OLT, TINT64):
	case CASE(OLT, TFLOAT32):
	case CASE(OLT, TFLOAT64):
		a = ABLT;
		break;

	case CASE(OLT, TUINT8):
	case CASE(OLT, TUINT16):
	case CASE(OLT, TUINT32):
	case CASE(OLT, TUINT64):
		a = ABLO;
		break;

	case CASE(OLE, TINT8):
	case CASE(OLE, TINT16):
	case CASE(OLE, TINT32):
	case CASE(OLE, TINT64):
	case CASE(OLE, TFLOAT32):
	case CASE(OLE, TFLOAT64):
		a = ABLE;
		break;

	case CASE(OLE, TUINT8):
	case CASE(OLE, TUINT16):
	case CASE(OLE, TUINT32):
	case CASE(OLE, TUINT64):
		a = ABLS;
		break;

	case CASE(OGT, TINT8):
	case CASE(OGT, TINT16):
	case CASE(OGT, TINT32):
	case CASE(OGT, TINT64):
	case CASE(OGT, TFLOAT32):
	case CASE(OGT, TFLOAT64):
		a = ABGT;
		break;

	case CASE(OGT, TUINT8):
	case CASE(OGT, TUINT16):
	case CASE(OGT, TUINT32):
	case CASE(OGT, TUINT64):
		a = ABHI;
		break;

	case CASE(OGE, TINT8):
	case CASE(OGE, TINT16):
	case CASE(OGE, TINT32):
	case CASE(OGE, TINT64):
	case CASE(OGE, TFLOAT32):
	case CASE(OGE, TFLOAT64):
		a = ABGE;
		break;

	case CASE(OGE, TUINT8):
	case CASE(OGE, TUINT16):
	case CASE(OGE, TUINT32):
	case CASE(OGE, TUINT64):
		a = ABHS;
		break;

	case CASE(OCMP, TBOOL):
	case CASE(OCMP, TINT8):
	case CASE(OCMP, TUINT8):
	case CASE(OCMP, TINT16):
	case CASE(OCMP, TUINT16):
	case CASE(OCMP, TINT32):
	case CASE(OCMP, TUINT32):
	case CASE(OCMP, TPTR32):
		a = ACMP;
		break;

	case CASE(OCMP, TFLOAT32):
		a = ACMPF;
		break;

	case CASE(OCMP, TFLOAT64):
		a = ACMPD;
		break;

	case CASE(OAS, TBOOL):
		a = AMOVB;
		break;

	case CASE(OAS, TINT8):
		a = AMOVBS;
		break;

	case CASE(OAS, TUINT8):
		a = AMOVBU;
		break;

	case CASE(OAS, TINT16):
		a = AMOVHS;
		break;

	case CASE(OAS, TUINT16):
		a = AMOVHU;
		break;

	case CASE(OAS, TINT32):
	case CASE(OAS, TUINT32):
	case CASE(OAS, TPTR32):
		a = AMOVW;
		break;

	case CASE(OAS, TFLOAT32):
		a = AMOVF;
		break;

	case CASE(OAS, TFLOAT64):
		a = AMOVD;
		break;

	case CASE(OADD, TINT8):
	case CASE(OADD, TUINT8):
	case CASE(OADD, TINT16):
	case CASE(OADD, TUINT16):
	case CASE(OADD, TINT32):
	case CASE(OADD, TUINT32):
	case CASE(OADD, TPTR32):
		a = AADD;
		break;

	case CASE(OADD, TFLOAT32):
		a = AADDF;
		break;

	case CASE(OADD, TFLOAT64):
		a = AADDD;
		break;

	case CASE(OSUB, TINT8):
	case CASE(OSUB, TUINT8):
	case CASE(OSUB, TINT16):
	case CASE(OSUB, TUINT16):
	case CASE(OSUB, TINT32):
	case CASE(OSUB, TUINT32):
	case CASE(OSUB, TPTR32):
		a = ASUB;
		break;

	case CASE(OSUB, TFLOAT32):
		a = ASUBF;
		break;

	case CASE(OSUB, TFLOAT64):
		a = ASUBD;
		break;

	case CASE(OMINUS, TINT8):
	case CASE(OMINUS, TUINT8):
	case CASE(OMINUS, TINT16):
	case CASE(OMINUS, TUINT16):
	case CASE(OMINUS, TINT32):
	case CASE(OMINUS, TUINT32):
	case CASE(OMINUS, TPTR32):
		a = ARSB;
		break;

	case CASE(OAND, TINT8):
	case CASE(OAND, TUINT8):
	case CASE(OAND, TINT16):
	case CASE(OAND, TUINT16):
	case CASE(OAND, TINT32):
	case CASE(OAND, TUINT32):
	case CASE(OAND, TPTR32):
		a = AAND;
		break;

	case CASE(OOR, TINT8):
	case CASE(OOR, TUINT8):
	case CASE(OOR, TINT16):
	case CASE(OOR, TUINT16):
	case CASE(OOR, TINT32):
	case CASE(OOR, TUINT32):
	case CASE(OOR, TPTR32):
		a = AORR;
		break;

	case CASE(OXOR, TINT8):
	case CASE(OXOR, TUINT8):
	case CASE(OXOR, TINT16):
	case CASE(OXOR, TUINT16):
	case CASE(OXOR, TINT32):
	case CASE(OXOR, TUINT32):
	case CASE(OXOR, TPTR32):
		a = AEOR;
		break;

	case CASE(OLSH, TINT8):
	case CASE(OLSH, TUINT8):
	case CASE(OLSH, TINT16):
	case CASE(OLSH, TUINT16):
	case CASE(OLSH, TINT32):
	case CASE(OLSH, TUINT32):
	case CASE(OLSH, TPTR32):
		a = ASLL;
		break;

	case CASE(ORSH, TUINT8):
	case CASE(ORSH, TUINT16):
	case CASE(ORSH, TUINT32):
	case CASE(ORSH, TPTR32):
		a = ASRL;
		break;

	case CASE(ORSH, TINT8):
	case CASE(ORSH, TINT16):
	case CASE(ORSH, TINT32):
		a = ASRA;
		break;

	case CASE(OMUL, TUINT8):
	case CASE(OMUL, TUINT16):
	case CASE(OMUL, TUINT32):
	case CASE(OMUL, TPTR32):
		a = AMULU;
		break;

	case CASE(OMUL, TINT8):
	case CASE(OMUL, TINT16):
	case CASE(OMUL, TINT32):
		a = AMUL;
		break;

	case CASE(OMUL, TFLOAT32):
		a = AMULF;
		break;

	case CASE(OMUL, TFLOAT64):
		a = AMULD;
		break;

	case CASE(ODIV, TUINT8):
	case CASE(ODIV, TUINT16):
	case CASE(ODIV, TUINT32):
	case CASE(ODIV, TPTR32):
		a = ADIVU;
		break;

	case CASE(ODIV, TINT8):
	case CASE(ODIV, TINT16):
	case CASE(ODIV, TINT32):
		a = ADIV;
		break;

	case CASE(OMOD, TUINT8):
	case CASE(OMOD, TUINT16):
	case CASE(OMOD, TUINT32):
	case CASE(OMOD, TPTR32):
		a = AMODU;
		break;

	case CASE(OMOD, TINT8):
	case CASE(OMOD, TINT16):
	case CASE(OMOD, TINT32):
		a = AMOD;
		break;

//	case CASE(OEXTEND, TINT16):
//		a = ACWD;
//		break;

//	case CASE(OEXTEND, TINT32):
//		a = ACDQ;
//		break;

//	case CASE(OEXTEND, TINT64):
//		a = ACQO;
//		break;

	case CASE(ODIV, TFLOAT32):
		a = ADIVF;
		break;

	case CASE(ODIV, TFLOAT64):
		a = ADIVD;
		break;

	}
	return a;
}

enum
{
	ODynam	= 1<<0,
	OPtrto	= 1<<1,
};

static	Node	clean[20];
static	int	cleani = 0;

void
sudoclean(void)
{
	if(clean[cleani-1].op != OEMPTY)
		regfree(&clean[cleani-1]);
	if(clean[cleani-2].op != OEMPTY)
		regfree(&clean[cleani-2]);
	cleani -= 2;
}

int
dotaddable(Node *n, Node *n1)
{
	int o;
	int64 oary[10];
	Node *nn;

	if(n->op != ODOT)
		return 0;

	o = dotoffset(n, oary, &nn);
	if(nn != N && nn->addable && o == 1 && oary[0] >= 0) {
		*n1 = *nn;
		n1->type = n->type;
		n1->xoffset += oary[0];
		return 1;
	}
	return 0;
}

/*
 * generate code to compute address of n,
 * a reference to a (perhaps nested) field inside
 * an array or struct.
 * return 0 on failure, 1 on success.
 * on success, leaves usable address in a.
 *
 * caller is responsible for calling sudoclean
 * after successful sudoaddable,
 * to release the register used for a.
 */
int
sudoaddable(int as, Node *n, Addr *a, int *w)
{
	int o, i;
	int64 oary[10];
	int64 v;
	Node n1, n2, n3, n4, *nn, *l, *r;
	Node *reg, *reg1;
	Prog *p1, *p2;
	Type *t;

	if(n->type == T)
		return 0;

	switch(n->op) {
	case OLITERAL:
		if(!isconst(n, CTINT))
			break;
		v = mpgetfix(n->val.u.xval);
		if(v >= 32000 || v <= -32000)
			break;
		goto lit;

	case ODOT:
	case ODOTPTR:
		cleani += 2;
		reg = &clean[cleani-1];
		reg1 = &clean[cleani-2];
		reg->op = OEMPTY;
		reg1->op = OEMPTY;
		goto odot;

	case OINDEX:
		return 0;
		// disabled: OINDEX case is now covered by agenr
		// for a more suitable register allocation pattern.
		if(n->left->type->etype == TSTRING)
			return 0;
		cleani += 2;
		reg = &clean[cleani-1];
		reg1 = &clean[cleani-2];
		reg->op = OEMPTY;
		reg1->op = OEMPTY;
		goto oindex;
	}
	return 0;

lit:
	switch(as) {
	default:
		return 0;
	case AADD: case ASUB: case AAND: case AORR: case AEOR:
	case AMOVB: case AMOVBS: case AMOVBU:
	case AMOVH: case AMOVHS: case AMOVHU:
	case AMOVW:
		break;
	}

	cleani += 2;
	reg = &clean[cleani-1];
	reg1 = &clean[cleani-2];
	reg->op = OEMPTY;
	reg1->op = OEMPTY;
	naddr(n, a, 1);
	goto yes;

odot:
	o = dotoffset(n, oary, &nn);
	if(nn == N)
		goto no;

	if(nn->addable && o == 1 && oary[0] >= 0) {
		// directly addressable set of DOTs
		n1 = *nn;
		n1.type = n->type;
		n1.xoffset += oary[0];
		naddr(&n1, a, 1);
		goto yes;
	}

	regalloc(reg, types[tptr], N);
	n1 = *reg;
	n1.op = OINDREG;
	if(oary[0] >= 0) {
		agen(nn, reg);
		n1.xoffset = oary[0];
	} else {
		cgen(nn, reg);
		cgen_checknil(reg);
		n1.xoffset = -(oary[0]+1);
	}

	for(i=1; i<o; i++) {
		if(oary[i] >= 0)
			fatal("can't happen");
		gins(AMOVW, &n1, reg);
		cgen_checknil(reg);
		n1.xoffset = -(oary[i]+1);
	}

	a->type = D_NONE;
	a->name = D_NONE;
	n1.type = n->type;
	naddr(&n1, a, 1);
	goto yes;

oindex:
	l = n->left;
	r = n->right;
	if(l->ullman >= UINF && r->ullman >= UINF)
		goto no;

	// set o to type of array
	o = 0;
	if(isptr[l->type->etype]) {
		o += OPtrto;
		if(l->type->type->etype != TARRAY)
			fatal("not ptr ary");
		if(l->type->type->bound < 0)
			o += ODynam;
	} else {
		if(l->type->etype != TARRAY)
			fatal("not ary");
		if(l->type->bound < 0)
			o += ODynam;
	}

	*w = n->type->width;
	if(isconst(r, CTINT))
		goto oindex_const;

	switch(*w) {
	default:
		goto no;
	case 1:
	case 2:
	case 4:
	case 8:
		break;
	}

	// load the array (reg)
	if(l->ullman > r->ullman) {
		regalloc(reg, types[tptr], N);
		if(o & OPtrto) {
			cgen(l, reg);
			cgen_checknil(reg);
		} else
			agen(l, reg);
	}

	// load the index (reg1)
	t = types[TUINT32];
	if(issigned[r->type->etype])
		t = types[TINT32];
	regalloc(reg1, t, N);
	regalloc(&n3, types[TINT32], reg1);
	p2 = cgenindex(r, &n3, debug['B'] || n->bounded);
	gmove(&n3, reg1);
	regfree(&n3);

	// load the array (reg)
	if(l->ullman <= r->ullman) {
		regalloc(reg, types[tptr], N);
		if(o & OPtrto) {
			cgen(l, reg);
			cgen_checknil(reg);
		} else
			agen(l, reg);
	}

	// check bounds
	if(!debug['B']) {
		if(o & ODynam) {
			n2 = *reg;
			n2.op = OINDREG;
			n2.type = types[tptr];
			n2.xoffset = Array_nel;
		} else {
			if(o & OPtrto)
				nodconst(&n2, types[TUINT32], l->type->type->bound);
			else
				nodconst(&n2, types[TUINT32], l->type->bound);
		}
		regalloc(&n3, n2.type, N);
		cgen(&n2, &n3);
		gcmp(optoas(OCMP, types[TUINT32]), reg1, &n3);
		regfree(&n3);
		p1 = gbranch(optoas(OLT, types[TUINT32]), T, +1);
		if(p2)
			patch(p2, pc);
		ginscall(panicindex, 0);
		patch(p1, pc);
	}

	if(o & ODynam) {
		n2 = *reg;
		n2.op = OINDREG;
		n2.type = types[tptr];
		n2.xoffset = Array_array;
		gmove(&n2, reg);
	}

	switch(*w) {
	case 1:
		gins(AADD, reg1, reg);
		break;
	case 2:
		gshift(AADD, reg1, SHIFT_LL, 1, reg);
		break;
	case 4:
		gshift(AADD, reg1, SHIFT_LL, 2, reg);
		break;
	case 8:
		gshift(AADD, reg1, SHIFT_LL, 3, reg);
		break;
	}

	naddr(reg1, a, 1);
	a->type = D_OREG;
	a->reg = reg->val.u.reg;
	a->offset = 0;
	goto yes;

oindex_const:
	// index is constant
	// can check statically and
	// can multiply by width statically

	regalloc(reg, types[tptr], N);
	if(o & OPtrto) {
		cgen(l, reg);
		cgen_checknil(reg);
	} else
		agen(l, reg);

	v = mpgetfix(r->val.u.xval);
	if(o & ODynam) {
		if(!debug['B'] && !n->bounded) {
			n1 = *reg;
			n1.op = OINDREG;
			n1.type = types[tptr];
			n1.xoffset = Array_nel;
			nodconst(&n2, types[TUINT32], v);
			regalloc(&n3, types[TUINT32], N);
			cgen(&n2, &n3);
			regalloc(&n4, n1.type, N);
			cgen(&n1, &n4);
			gcmp(optoas(OCMP, types[TUINT32]), &n4, &n3);
			regfree(&n4);
			regfree(&n3);
			p1 = gbranch(optoas(OGT, types[TUINT32]), T, +1);
			ginscall(panicindex, 0);
			patch(p1, pc);
		}

		n1 = *reg;
		n1.op = OINDREG;
		n1.type = types[tptr];
		n1.xoffset = Array_array;
		gmove(&n1, reg);
	}

	n2 = *reg;
	n2.op = OINDREG;
	n2.xoffset = v * (*w);
	a->type = D_NONE;
	a->name = D_NONE;
	naddr(&n2, a, 1);
	goto yes;

yes:
	return 1;

no:
	sudoclean();
	return 0;
}
