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
#include "gg.h"

// TODO(rsc): Can make this bigger if we move
// the text segment up higher in 6l for all GOOS.
vlong unmappedzero = 4096;

void
clearp(Prog *p)
{
	p->as = AEND;
	p->from.type = D_NONE;
	p->from.index = D_NONE;
	p->to.type = D_NONE;
	p->to.index = D_NONE;
	p->loc = pcloc;
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
	p->to.type = D_BRANCH;
	p->to.u.branch = P;
	if(as != AJMP && likely != 0) {
		p->from.type = D_CONST;
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
	if(p->to.type != D_BRANCH)
		fatal("patch: not a branch");
	p->to.u.branch = to;
	p->to.offset = to->loc;
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

	pl = mal(sizeof(*pl));
	if(plist == nil)
		plist = pl;
	else
		plast->link = pl;
	plast = pl;

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

	p = gbranch(AJMP, T, 0);
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
	p->from.gotype = ngotype(nam);
	p->to.sym = S;
	p->to.type = D_CONST;
	p->to.offset = nam->type->width;
	if(nam->readonly)
		p->from.scale = RODATA;
	if(nam->type != T && !haspointers(nam->type))
		p->from.scale |= NOPTR;
}

void
gtrack(Sym *s)
{
	Prog *p;
	
	p = gins(AUSEFIELD, N, N);
	p->from.type = D_EXTERN;
	p->from.index = D_NONE;
	p->from.sym = s;
}

void
ggloblsym(Sym *s, int32 width, int dupok, int rodata)
{
	Prog *p;

	p = gins(AGLOBL, N, N);
	p->from.type = D_EXTERN;
	p->from.index = D_NONE;
	p->from.sym = s;
	p->to.type = D_CONST;
	p->to.index = D_NONE;
	p->to.offset = width;
	if(dupok)
		p->from.scale |= DUPOK;
	if(rodata)
		p->from.scale |= RODATA;
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
	if(a->type == D_ADDR && a->index == D_EXTERN) {
		a->type = D_EXTERN;
		a->index = D_NONE;
		a->sym = n->sym;
	}
}

static	int	resvd[] =
{
	D_DI,	// for movstring
	D_SI,	// for movstring

	D_AX,	// for divide
	D_CX,	// for shift
	D_DX,	// for divide
	D_SP,	// for stack
};

void
ginit(void)
{
	int i;

	for(i=0; i<nelem(reg); i++)
		reg[i] = 1;
	for(i=D_AX; i<=D_R15; i++)
		reg[i] = 0;
	for(i=D_X0; i<=D_X15; i++)
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

	for(i=D_AX; i<=D_R15; i++)
		if(reg[i])
			yyerror("reg %R left allocated\n", i);
	for(i=D_X0; i<=D_X15; i++)
		if(reg[i])
			yyerror("reg %R left allocated\n", i);
}

int32
anyregalloc(void)
{
	int i, j;

	for(i=D_AX; i<=D_R15; i++) {
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

static	uintptr	regpc[D_R15+1 - D_AX];

/*
 * allocate register of type t, leave in n.
 * if o != N, o is desired fixed register.
 * caller must regfree(n).
 */
void
regalloc(Node *n, Type *t, Node *o)
{
	int i, et;

	if(t == T)
		fatal("regalloc: t nil");
	et = simtype[t->etype];

	switch(et) {
	case TINT8:
	case TUINT8:
	case TINT16:
	case TUINT16:
	case TINT32:
	case TUINT32:
	case TINT64:
	case TUINT64:
	case TPTR32:
	case TPTR64:
	case TBOOL:
		if(o != N && o->op == OREGISTER) {
			i = o->val.u.reg;
			if(i >= D_AX && i <= D_R15)
				goto out;
		}
		for(i=D_AX; i<=D_R15; i++)
			if(reg[i] == 0) {
				regpc[i-D_AX] = (uintptr)getcallerpc(&n);
				goto out;
			}

		flusherrors();
		for(i=0; i+D_AX<=D_R15; i++)
			print("%d %p\n", i, regpc[i]);
		fatal("out of fixed registers");

	case TFLOAT32:
	case TFLOAT64:
		if(o != N && o->op == OREGISTER) {
			i = o->val.u.reg;
			if(i >= D_X0 && i <= D_X15)
				goto out;
		}
		for(i=D_X0; i<=D_X15; i++)
			if(reg[i] == 0)
				goto out;
		fatal("out of floating registers");

	case TCOMPLEX64:
	case TCOMPLEX128:
		tempname(n, t);
		return;
	}
	fatal("regalloc: unknown type %T", t);
	return;

out:
	reg[i]++;
	nodreg(n, t, i);
}

void
regfree(Node *n)
{
	int i;

	if(n->op == ONAME)
		return;
	if(n->op != OREGISTER && n->op != OINDREG)
		fatal("regfree: not a register");
	i = n->val.u.reg;
	if(i == D_SP)
		return;
	if(i < 0 || i >= nelem(reg))
		fatal("regfree: reg out of range");
	if(reg[i] <= 0)
		fatal("regfree: reg not allocated");
	reg[i]--;
	if(reg[i] == 0 && D_AX <= i && i <= D_R15)
		regpc[i - D_AX] = 0;
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
		n->val.u.reg = D_SP;
		break;

	case 1:		// input arg
		n->class = PPARAM;
		break;

	case 2:		// offset output arg
fatal("shouldnt be used");
		n->op = OINDREG;
		n->val.u.reg = D_SP;
		n->xoffset += types[tptr]->width;
		break;
	}
	n->typecheck = 1;
	return n;
}

/*
 * generate
 *	as $c, reg
 */
void
gconreg(int as, vlong c, int reg)
{
	Node nr;

	nodreg(&nr, types[TINT64], reg);
	ginscon(as, c, &nr);
}

/*
 * generate
 *	as $c, n
 */
void
ginscon(int as, vlong c, Node *n2)
{
	Node n1, ntmp;

	nodconst(&n1, types[TINT64], c);

	if(as != AMOVQ && (c < -1LL<<31 || c >= 1LL<<31)) {
		// cannot have 64-bit immediokate in ADD, etc.
		// instead, MOV into register first.
		regalloc(&ntmp, types[TINT64], N);
		gins(AMOVQ, &n1, &ntmp);
		gins(as, &ntmp, n2);
		regfree(&ntmp);
		return;
	}
	gins(as, &n1, n2);
}

#define	CASE(a,b)	(((a)<<16)|((b)<<0))

/*
 * Is this node a memory operand?
 */
int
ismem(Node *n)
{
	switch(n->op) {
	case OITAB:
	case OLEN:
	case OCAP:
	case OINDREG:
	case ONAME:
	case OPARAM:
	case OCLOSUREVAR:
		return 1;
	case OADDR:
		if(flag_largemodel)
			return 1;
		break;
	}
	return 0;
}

/*
 * set up nodes representing 2^63
 */
Node bigi;
Node bigf;

void
bignodes(void)
{
	static int did;

	if(did)
		return;
	did = 1;

	nodconst(&bigi, types[TUINT64], 1);
	mpshiftfix(bigi.val.u.xval, 63);

	bigf = bigi;
	bigf.type = types[TFLOAT64];
	bigf.val.ctype = CTFLT;
	bigf.val.u.fval = mal(sizeof *bigf.val.u.fval);
	mpmovefixflt(bigf.val.u.fval, bigi.val.u.xval);
}

/*
 * generate move:
 *	t = f
 * hard part is conversions.
 */
// TODO: lost special constants for floating point.  XORPD for 0.0?
void
gmove(Node *f, Node *t)
{
	int a, ft, tt;
	Type *cvt;
	Node r1, r2, r3, r4, zero, one, con;
	Prog *p1, *p2;

	if(debug['M'])
		print("gmove %lN -> %lN\n", f, t);

	ft = simsimtype(f->type);
	tt = simsimtype(t->type);
	cvt = t->type;

	if(iscomplex[ft] || iscomplex[tt]) {
		complexmove(f, t);
		return;
	}

	// cannot have two memory operands
	if(ismem(f) && ismem(t))
		goto hard;

	// convert constant to desired type
	if(f->op == OLITERAL) {
		convconst(&con, t->type, &f->val);
		f = &con;
		ft = tt;	// so big switch will choose a simple mov

		// some constants can't move directly to memory.
		if(ismem(t)) {
			// float constants come from memory.
			if(isfloat[tt])
				goto hard;

			// 64-bit immediates are really 32-bit sign-extended
			// unless moving into a register.
			if(isint[tt]) {
				if(mpcmpfixfix(con.val.u.xval, minintval[TINT32]) < 0)
					goto hard;
				if(mpcmpfixfix(con.val.u.xval, maxintval[TINT32]) > 0)
					goto hard;
			}
		}
	}

	// value -> value copy, only one memory operand.
	// figure out the instruction to use.
	// break out of switch for one-instruction gins.
	// goto rdst for "destination must be register".
	// goto hard for "convert to cvt type first".
	// otherwise handle and return.

	switch(CASE(ft, tt)) {
	default:
		fatal("gmove %lT -> %lT", f->type, t->type);

	/*
	 * integer copy and truncate
	 */
	case CASE(TINT8, TINT8):	// same size
	case CASE(TINT8, TUINT8):
	case CASE(TUINT8, TINT8):
	case CASE(TUINT8, TUINT8):
	case CASE(TINT16, TINT8):	// truncate
	case CASE(TUINT16, TINT8):
	case CASE(TINT32, TINT8):
	case CASE(TUINT32, TINT8):
	case CASE(TINT64, TINT8):
	case CASE(TUINT64, TINT8):
	case CASE(TINT16, TUINT8):
	case CASE(TUINT16, TUINT8):
	case CASE(TINT32, TUINT8):
	case CASE(TUINT32, TUINT8):
	case CASE(TINT64, TUINT8):
	case CASE(TUINT64, TUINT8):
		a = AMOVB;
		break;

	case CASE(TINT16, TINT16):	// same size
	case CASE(TINT16, TUINT16):
	case CASE(TUINT16, TINT16):
	case CASE(TUINT16, TUINT16):
	case CASE(TINT32, TINT16):	// truncate
	case CASE(TUINT32, TINT16):
	case CASE(TINT64, TINT16):
	case CASE(TUINT64, TINT16):
	case CASE(TINT32, TUINT16):
	case CASE(TUINT32, TUINT16):
	case CASE(TINT64, TUINT16):
	case CASE(TUINT64, TUINT16):
		a = AMOVW;
		break;

	case CASE(TINT32, TINT32):	// same size
	case CASE(TINT32, TUINT32):
	case CASE(TUINT32, TINT32):
	case CASE(TUINT32, TUINT32):
		a = AMOVL;
		break;

	case CASE(TINT64, TINT32):	// truncate
	case CASE(TUINT64, TINT32):
	case CASE(TINT64, TUINT32):
	case CASE(TUINT64, TUINT32):
		a = AMOVQL;
		break;

	case CASE(TINT64, TINT64):	// same size
	case CASE(TINT64, TUINT64):
	case CASE(TUINT64, TINT64):
	case CASE(TUINT64, TUINT64):
		a = AMOVQ;
		break;

	/*
	 * integer up-conversions
	 */
	case CASE(TINT8, TINT16):	// sign extend int8
	case CASE(TINT8, TUINT16):
		a = AMOVBWSX;
		goto rdst;
	case CASE(TINT8, TINT32):
	case CASE(TINT8, TUINT32):
		a = AMOVBLSX;
		goto rdst;
	case CASE(TINT8, TINT64):
	case CASE(TINT8, TUINT64):
		a = AMOVBQSX;
		goto rdst;

	case CASE(TUINT8, TINT16):	// zero extend uint8
	case CASE(TUINT8, TUINT16):
		a = AMOVBWZX;
		goto rdst;
	case CASE(TUINT8, TINT32):
	case CASE(TUINT8, TUINT32):
		a = AMOVBLZX;
		goto rdst;
	case CASE(TUINT8, TINT64):
	case CASE(TUINT8, TUINT64):
		a = AMOVBQZX;
		goto rdst;

	case CASE(TINT16, TINT32):	// sign extend int16
	case CASE(TINT16, TUINT32):
		a = AMOVWLSX;
		goto rdst;
	case CASE(TINT16, TINT64):
	case CASE(TINT16, TUINT64):
		a = AMOVWQSX;
		goto rdst;

	case CASE(TUINT16, TINT32):	// zero extend uint16
	case CASE(TUINT16, TUINT32):
		a = AMOVWLZX;
		goto rdst;
	case CASE(TUINT16, TINT64):
	case CASE(TUINT16, TUINT64):
		a = AMOVWQZX;
		goto rdst;

	case CASE(TINT32, TINT64):	// sign extend int32
	case CASE(TINT32, TUINT64):
		a = AMOVLQSX;
		goto rdst;

	case CASE(TUINT32, TINT64):	// zero extend uint32
	case CASE(TUINT32, TUINT64):
		// AMOVL into a register zeros the top of the register,
		// so this is not always necessary, but if we rely on AMOVL
		// the optimizer is almost certain to screw with us.
		a = AMOVLQZX;
		goto rdst;

	/*
	* float to integer
	*/
	case CASE(TFLOAT32, TINT32):
		a = ACVTTSS2SL;
		goto rdst;

	case CASE(TFLOAT64, TINT32):
		a = ACVTTSD2SL;
		goto rdst;

	case CASE(TFLOAT32, TINT64):
		a = ACVTTSS2SQ;
		goto rdst;

	case CASE(TFLOAT64, TINT64):
		a = ACVTTSD2SQ;
		goto rdst;

	case CASE(TFLOAT32, TINT16):
	case CASE(TFLOAT32, TINT8):
	case CASE(TFLOAT32, TUINT16):
	case CASE(TFLOAT32, TUINT8):
	case CASE(TFLOAT64, TINT16):
	case CASE(TFLOAT64, TINT8):
	case CASE(TFLOAT64, TUINT16):
	case CASE(TFLOAT64, TUINT8):
		// convert via int32.
		cvt = types[TINT32];
		goto hard;

	case CASE(TFLOAT32, TUINT32):
	case CASE(TFLOAT64, TUINT32):
		// convert via int64.
		cvt = types[TINT64];
		goto hard;

	case CASE(TFLOAT32, TUINT64):
	case CASE(TFLOAT64, TUINT64):
		// algorithm is:
		//	if small enough, use native float64 -> int64 conversion.
		//	otherwise, subtract 2^63, convert, and add it back.
		a = ACVTTSS2SQ;
		if(ft == TFLOAT64)
			a = ACVTTSD2SQ;
		bignodes();
		regalloc(&r1, types[ft], N);
		regalloc(&r2, types[tt], t);
		regalloc(&r3, types[ft], N);
		regalloc(&r4, types[tt], N);
		gins(optoas(OAS, f->type), f, &r1);
		gins(optoas(OCMP, f->type), &bigf, &r1);
		p1 = gbranch(optoas(OLE, f->type), T, +1);
		gins(a, &r1, &r2);
		p2 = gbranch(AJMP, T, 0);
		patch(p1, pc);
		gins(optoas(OAS, f->type), &bigf, &r3);
		gins(optoas(OSUB, f->type), &r3, &r1);
		gins(a, &r1, &r2);
		gins(AMOVQ, &bigi, &r4);
		gins(AXORQ, &r4, &r2);
		patch(p2, pc);
		gmove(&r2, t);
		regfree(&r4);
		regfree(&r3);
		regfree(&r2);
		regfree(&r1);
		return;

	/*
	 * integer to float
	 */
	case CASE(TINT32, TFLOAT32):
		a = ACVTSL2SS;
		goto rdst;


	case CASE(TINT32, TFLOAT64):
		a = ACVTSL2SD;
		goto rdst;

	case CASE(TINT64, TFLOAT32):
		a = ACVTSQ2SS;
		goto rdst;

	case CASE(TINT64, TFLOAT64):
		a = ACVTSQ2SD;
		goto rdst;

	case CASE(TINT16, TFLOAT32):
	case CASE(TINT16, TFLOAT64):
	case CASE(TINT8, TFLOAT32):
	case CASE(TINT8, TFLOAT64):
	case CASE(TUINT16, TFLOAT32):
	case CASE(TUINT16, TFLOAT64):
	case CASE(TUINT8, TFLOAT32):
	case CASE(TUINT8, TFLOAT64):
		// convert via int32
		cvt = types[TINT32];
		goto hard;

	case CASE(TUINT32, TFLOAT32):
	case CASE(TUINT32, TFLOAT64):
		// convert via int64.
		cvt = types[TINT64];
		goto hard;

	case CASE(TUINT64, TFLOAT32):
	case CASE(TUINT64, TFLOAT64):
		// algorithm is:
		//	if small enough, use native int64 -> uint64 conversion.
		//	otherwise, halve (rounding to odd?), convert, and double.
		a = ACVTSQ2SS;
		if(tt == TFLOAT64)
			a = ACVTSQ2SD;
		nodconst(&zero, types[TUINT64], 0);
		nodconst(&one, types[TUINT64], 1);
		regalloc(&r1, f->type, f);
		regalloc(&r2, t->type, t);
		regalloc(&r3, f->type, N);
		regalloc(&r4, f->type, N);
		gmove(f, &r1);
		gins(ACMPQ, &r1, &zero);
		p1 = gbranch(AJLT, T, +1);
		gins(a, &r1, &r2);
		p2 = gbranch(AJMP, T, 0);
		patch(p1, pc);
		gmove(&r1, &r3);
		gins(ASHRQ, &one, &r3);
		gmove(&r1, &r4);
		gins(AANDL, &one, &r4);
		gins(AORQ, &r4, &r3);
		gins(a, &r3, &r2);
		gins(optoas(OADD, t->type), &r2, &r2);
		patch(p2, pc);
		gmove(&r2, t);
		regfree(&r4);
		regfree(&r3);
		regfree(&r2);
		regfree(&r1);
		return;

	/*
	 * float to float
	 */
	case CASE(TFLOAT32, TFLOAT32):
		a = AMOVSS;
		break;

	case CASE(TFLOAT64, TFLOAT64):
		a = AMOVSD;
		break;

	case CASE(TFLOAT32, TFLOAT64):
		a = ACVTSS2SD;
		goto rdst;

	case CASE(TFLOAT64, TFLOAT32):
		a = ACVTSD2SS;
		goto rdst;
	}

	gins(a, f, t);
	return;

rdst:
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
	int32 w;
	Prog *p;
	Addr af, at;

//	if(f != N && f->op == OINDEX) {
//		regalloc(&nod, &regnode, Z);
//		v = constnode.vconst;
//		cgen(f->right, &nod);
//		constnode.vconst = v;
//		idx.reg = nod.reg;
//		regfree(&nod);
//	}
//	if(t != N && t->op == OINDEX) {
//		regalloc(&nod, &regnode, Z);
//		v = constnode.vconst;
//		cgen(t->right, &nod);
//		constnode.vconst = v;
//		idx.reg = nod.reg;
//		regfree(&nod);
//	}

	switch(as) {
	case AMOVB:
	case AMOVW:
	case AMOVL:
	case AMOVQ:
	case AMOVSS:
	case AMOVSD:
		if(f != N && t != N && samaddr(f, t))
			return nil;
		break;
	
	case ALEAQ:
		if(f != N && isconst(f, CTNIL)) {
			fatal("gins LEAQ nil %T", f->type);
		}
		break;
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

	w = 0;
	switch(as) {
	case AMOVB:
		w = 1;
		break;
	case AMOVW:
		w = 2;
		break;
	case AMOVL:
		w = 4;
		break;
	case AMOVQ:
		w = 8;
		break;
	}
	if(w != 0 && ((f != N && af.width < w) || (t != N && at.width > w))) {
		dump("f", f);
		dump("t", t);
		fatal("bad width: %P (%d, %d)\n", p, af.width, at.width);
	}

	return p;
}

// Generate an instruction referencing *n
// to force segv on nil pointer dereference.
void
checkref(Node *n, int force)
{
	Node m;

	if(!force && isptr[n->type->etype] && n->type->type->width < unmappedzero)
		return;

	regalloc(&m, types[TUINTPTR], n);
	cgen(n, &m);
	m.xoffset = 0;
	m.op = OINDREG;
	m.type = types[TUINT8];
	gins(ATESTB, nodintconst(0), &m);
	regfree(&m);
}

static void
checkoffset(Addr *a, int canemitcode)
{
	Prog *p;

	if(a->offset < unmappedzero)
		return;
	if(!canemitcode)
		fatal("checkoffset %#llx, cannot emit code", a->offset);

	// cannot rely on unmapped nil page at 0 to catch
	// reference with large offset. instead, emit explicit
	// test of 0(reg).
	p = gins(ATESTB, nodintconst(0), N);
	p->to = *a;
	p->to.offset = 0;
}

/*
 * generate code to compute n;
 * make a refer to result.
 */
void
naddr(Node *n, Addr *a, int canemitcode)
{
	a->scale = 0;
	a->index = D_NONE;
	a->type = D_NONE;
	a->gotype = S;
	a->node = N;
	a->width = 0;
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
		a->type = n->val.u.reg;
		a->sym = S;
		break;

//	case OINDEX:
//	case OIND:
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
		a->type = n->val.u.reg+D_INDIR;
		a->sym = n->sym;
		a->offset = n->xoffset;
		if(a->offset != (int32)a->offset)
			yyerror("offset %lld too large for OINDREG", a->offset);
		checkoffset(a, canemitcode);
		break;

	case OPARAM:
		// n->left is PHEAP ONAME for stack parameter.
		// compute address of actual parameter on stack.
		a->etype = simtype[n->left->type->etype];
		a->width = n->left->type->width;
		a->offset = n->xoffset;
		a->sym = n->left->sym;
		a->type = D_PARAM;
		a->node = n->left->orig;
		break;
	
	case OCLOSUREVAR:
		a->type = D_DX+D_INDIR;
		a->sym = S;
		a->offset = n->xoffset;
		break;
	
	case OCFUNC:
		naddr(n->left, a, canemitcode);
		a->sym = n->left->sym;
		break;

	case ONAME:
		a->etype = 0;
		if(n->type != T)
			a->etype = simtype[n->type->etype];
		a->offset = n->xoffset;
		a->sym = n->sym;
		a->node = n->orig;
		//if(a->node >= (Node*)&n)
		//	fatal("stack node");
		if(a->sym == S)
			a->sym = lookup(".noname");
		if(n->method) {
			if(n->type != T)
			if(n->type->sym != S)
			if(n->type->sym->pkg != nil)
				a->sym = pkglookup(a->sym->name, n->type->sym->pkg);
		}

		switch(n->class) {
		default:
			fatal("naddr: ONAME class %S %d\n", n->sym, n->class);
		case PEXTERN:
			a->type = D_EXTERN;
			break;
		case PAUTO:
			a->type = D_AUTO;
			break;
		case PPARAM:
		case PPARAMOUT:
			a->type = D_PARAM;
			break;
		case PFUNC:
			a->index = D_EXTERN;
			a->type = D_ADDR;
			a->width = widthptr;
			a->sym = funcsym(a->sym);
			break;
		}
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
			a->sym = S;
			a->type = D_CONST;
			a->offset = mpgetfix(n->val.u.xval);
			break;
		case CTSTR:
			datagostring(n->val.u.sval, a);
			break;
		case CTBOOL:
			a->sym = S;
			a->type = D_CONST;
			a->offset = n->val.u.bval;
			break;
		case CTNIL:
			a->sym = S;
			a->type = D_CONST;
			a->offset = 0;
			break;
		}
		break;

	case OADDR:
		naddr(n->left, a, canemitcode);
		a->width = widthptr;
		if(a->type >= D_INDIR) {
			a->type -= D_INDIR;
			break;
		}
		if(a->type == D_EXTERN || a->type == D_STATIC ||
		   a->type == D_AUTO || a->type == D_PARAM)
			if(a->index == D_NONE) {
				a->index = a->type;
				a->type = D_ADDR;
				break;
			}
		fatal("naddr: OADDR\n");
	
	case OITAB:
		// itable of interface value
		naddr(n->left, a, canemitcode);
		if(a->type == D_CONST && a->offset == 0)
			break;  // itab(nil)
		a->etype = tptr;
		a->width = widthptr;
		if(a->offset >= unmappedzero && a->offset-Array_nel < unmappedzero)
			checkoffset(a, canemitcode);
		break;

	case OLEN:
		// len of string or slice
		naddr(n->left, a, canemitcode);
		if(a->type == D_CONST && a->offset == 0)
			break;	// len(nil)
		a->etype = simtype[TUINT];
		a->offset += Array_nel;
		a->width = widthint;
		if(a->offset >= unmappedzero && a->offset-Array_nel < unmappedzero)
			checkoffset(a, canemitcode);
		break;

	case OCAP:
		// cap of string or slice
		naddr(n->left, a, canemitcode);
		if(a->type == D_CONST && a->offset == 0)
			break;	// cap(nil)
		a->etype = simtype[TUINT];
		a->offset += Array_cap;
		a->width = widthint;
		if(a->offset >= unmappedzero && a->offset-Array_cap < unmappedzero)
			checkoffset(a, canemitcode);
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
		fatal("optoas: no entry %O-%T", op, t);
		break;

	case CASE(OADDR, TPTR32):
		a = ALEAL;
		break;

	case CASE(OADDR, TPTR64):
		a = ALEAQ;
		break;

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
		a = AJEQ;
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
		a = AJNE;
		break;

	case CASE(OLT, TINT8):
	case CASE(OLT, TINT16):
	case CASE(OLT, TINT32):
	case CASE(OLT, TINT64):
		a = AJLT;
		break;

	case CASE(OLT, TUINT8):
	case CASE(OLT, TUINT16):
	case CASE(OLT, TUINT32):
	case CASE(OLT, TUINT64):
		a = AJCS;
		break;

	case CASE(OLE, TINT8):
	case CASE(OLE, TINT16):
	case CASE(OLE, TINT32):
	case CASE(OLE, TINT64):
		a = AJLE;
		break;

	case CASE(OLE, TUINT8):
	case CASE(OLE, TUINT16):
	case CASE(OLE, TUINT32):
	case CASE(OLE, TUINT64):
		a = AJLS;
		break;

	case CASE(OGT, TINT8):
	case CASE(OGT, TINT16):
	case CASE(OGT, TINT32):
	case CASE(OGT, TINT64):
		a = AJGT;
		break;

	case CASE(OGT, TUINT8):
	case CASE(OGT, TUINT16):
	case CASE(OGT, TUINT32):
	case CASE(OGT, TUINT64):
	case CASE(OLT, TFLOAT32):
	case CASE(OLT, TFLOAT64):
		a = AJHI;
		break;

	case CASE(OGE, TINT8):
	case CASE(OGE, TINT16):
	case CASE(OGE, TINT32):
	case CASE(OGE, TINT64):
		a = AJGE;
		break;

	case CASE(OGE, TUINT8):
	case CASE(OGE, TUINT16):
	case CASE(OGE, TUINT32):
	case CASE(OGE, TUINT64):
	case CASE(OLE, TFLOAT32):
	case CASE(OLE, TFLOAT64):
		a = AJCC;
		break;

	case CASE(OCMP, TBOOL):
	case CASE(OCMP, TINT8):
	case CASE(OCMP, TUINT8):
		a = ACMPB;
		break;

	case CASE(OCMP, TINT16):
	case CASE(OCMP, TUINT16):
		a = ACMPW;
		break;

	case CASE(OCMP, TINT32):
	case CASE(OCMP, TUINT32):
	case CASE(OCMP, TPTR32):
		a = ACMPL;
		break;

	case CASE(OCMP, TINT64):
	case CASE(OCMP, TUINT64):
	case CASE(OCMP, TPTR64):
		a = ACMPQ;
		break;

	case CASE(OCMP, TFLOAT32):
		a = AUCOMISS;
		break;

	case CASE(OCMP, TFLOAT64):
		a = AUCOMISD;
		break;

	case CASE(OAS, TBOOL):
	case CASE(OAS, TINT8):
	case CASE(OAS, TUINT8):
		a = AMOVB;
		break;

	case CASE(OAS, TINT16):
	case CASE(OAS, TUINT16):
		a = AMOVW;
		break;

	case CASE(OAS, TINT32):
	case CASE(OAS, TUINT32):
	case CASE(OAS, TPTR32):
		a = AMOVL;
		break;

	case CASE(OAS, TINT64):
	case CASE(OAS, TUINT64):
	case CASE(OAS, TPTR64):
		a = AMOVQ;
		break;

	case CASE(OAS, TFLOAT32):
		a = AMOVSS;
		break;

	case CASE(OAS, TFLOAT64):
		a = AMOVSD;
		break;

	case CASE(OADD, TINT8):
	case CASE(OADD, TUINT8):
		a = AADDB;
		break;

	case CASE(OADD, TINT16):
	case CASE(OADD, TUINT16):
		a = AADDW;
		break;

	case CASE(OADD, TINT32):
	case CASE(OADD, TUINT32):
	case CASE(OADD, TPTR32):
		a = AADDL;
		break;

	case CASE(OADD, TINT64):
	case CASE(OADD, TUINT64):
	case CASE(OADD, TPTR64):
		a = AADDQ;
		break;

	case CASE(OADD, TFLOAT32):
		a = AADDSS;
		break;

	case CASE(OADD, TFLOAT64):
		a = AADDSD;
		break;

	case CASE(OSUB, TINT8):
	case CASE(OSUB, TUINT8):
		a = ASUBB;
		break;

	case CASE(OSUB, TINT16):
	case CASE(OSUB, TUINT16):
		a = ASUBW;
		break;

	case CASE(OSUB, TINT32):
	case CASE(OSUB, TUINT32):
	case CASE(OSUB, TPTR32):
		a = ASUBL;
		break;

	case CASE(OSUB, TINT64):
	case CASE(OSUB, TUINT64):
	case CASE(OSUB, TPTR64):
		a = ASUBQ;
		break;

	case CASE(OSUB, TFLOAT32):
		a = ASUBSS;
		break;

	case CASE(OSUB, TFLOAT64):
		a = ASUBSD;
		break;

	case CASE(OINC, TINT8):
	case CASE(OINC, TUINT8):
		a = AINCB;
		break;

	case CASE(OINC, TINT16):
	case CASE(OINC, TUINT16):
		a = AINCW;
		break;

	case CASE(OINC, TINT32):
	case CASE(OINC, TUINT32):
	case CASE(OINC, TPTR32):
		a = AINCL;
		break;

	case CASE(OINC, TINT64):
	case CASE(OINC, TUINT64):
	case CASE(OINC, TPTR64):
		a = AINCQ;
		break;

	case CASE(ODEC, TINT8):
	case CASE(ODEC, TUINT8):
		a = ADECB;
		break;

	case CASE(ODEC, TINT16):
	case CASE(ODEC, TUINT16):
		a = ADECW;
		break;

	case CASE(ODEC, TINT32):
	case CASE(ODEC, TUINT32):
	case CASE(ODEC, TPTR32):
		a = ADECL;
		break;

	case CASE(ODEC, TINT64):
	case CASE(ODEC, TUINT64):
	case CASE(ODEC, TPTR64):
		a = ADECQ;
		break;

	case CASE(OMINUS, TINT8):
	case CASE(OMINUS, TUINT8):
		a = ANEGB;
		break;

	case CASE(OMINUS, TINT16):
	case CASE(OMINUS, TUINT16):
		a = ANEGW;
		break;

	case CASE(OMINUS, TINT32):
	case CASE(OMINUS, TUINT32):
	case CASE(OMINUS, TPTR32):
		a = ANEGL;
		break;

	case CASE(OMINUS, TINT64):
	case CASE(OMINUS, TUINT64):
	case CASE(OMINUS, TPTR64):
		a = ANEGQ;
		break;

	case CASE(OAND, TINT8):
	case CASE(OAND, TUINT8):
		a = AANDB;
		break;

	case CASE(OAND, TINT16):
	case CASE(OAND, TUINT16):
		a = AANDW;
		break;

	case CASE(OAND, TINT32):
	case CASE(OAND, TUINT32):
	case CASE(OAND, TPTR32):
		a = AANDL;
		break;

	case CASE(OAND, TINT64):
	case CASE(OAND, TUINT64):
	case CASE(OAND, TPTR64):
		a = AANDQ;
		break;

	case CASE(OOR, TINT8):
	case CASE(OOR, TUINT8):
		a = AORB;
		break;

	case CASE(OOR, TINT16):
	case CASE(OOR, TUINT16):
		a = AORW;
		break;

	case CASE(OOR, TINT32):
	case CASE(OOR, TUINT32):
	case CASE(OOR, TPTR32):
		a = AORL;
		break;

	case CASE(OOR, TINT64):
	case CASE(OOR, TUINT64):
	case CASE(OOR, TPTR64):
		a = AORQ;
		break;

	case CASE(OXOR, TINT8):
	case CASE(OXOR, TUINT8):
		a = AXORB;
		break;

	case CASE(OXOR, TINT16):
	case CASE(OXOR, TUINT16):
		a = AXORW;
		break;

	case CASE(OXOR, TINT32):
	case CASE(OXOR, TUINT32):
	case CASE(OXOR, TPTR32):
		a = AXORL;
		break;

	case CASE(OXOR, TINT64):
	case CASE(OXOR, TUINT64):
	case CASE(OXOR, TPTR64):
		a = AXORQ;
		break;

	case CASE(OLROT, TINT8):
	case CASE(OLROT, TUINT8):
		a = AROLB;
		break;

	case CASE(OLROT, TINT16):
	case CASE(OLROT, TUINT16):
		a = AROLW;
		break;

	case CASE(OLROT, TINT32):
	case CASE(OLROT, TUINT32):
	case CASE(OLROT, TPTR32):
		a = AROLL;
		break;

	case CASE(OLROT, TINT64):
	case CASE(OLROT, TUINT64):
	case CASE(OLROT, TPTR64):
		a = AROLQ;
		break;

	case CASE(OLSH, TINT8):
	case CASE(OLSH, TUINT8):
		a = ASHLB;
		break;

	case CASE(OLSH, TINT16):
	case CASE(OLSH, TUINT16):
		a = ASHLW;
		break;

	case CASE(OLSH, TINT32):
	case CASE(OLSH, TUINT32):
	case CASE(OLSH, TPTR32):
		a = ASHLL;
		break;

	case CASE(OLSH, TINT64):
	case CASE(OLSH, TUINT64):
	case CASE(OLSH, TPTR64):
		a = ASHLQ;
		break;

	case CASE(ORSH, TUINT8):
		a = ASHRB;
		break;

	case CASE(ORSH, TUINT16):
		a = ASHRW;
		break;

	case CASE(ORSH, TUINT32):
	case CASE(ORSH, TPTR32):
		a = ASHRL;
		break;

	case CASE(ORSH, TUINT64):
	case CASE(ORSH, TPTR64):
		a = ASHRQ;
		break;

	case CASE(ORSH, TINT8):
		a = ASARB;
		break;

	case CASE(ORSH, TINT16):
		a = ASARW;
		break;

	case CASE(ORSH, TINT32):
		a = ASARL;
		break;

	case CASE(ORSH, TINT64):
		a = ASARQ;
		break;

	case CASE(ORROTC, TINT8):
	case CASE(ORROTC, TUINT8):
		a = ARCRB;
		break;

	case CASE(ORROTC, TINT16):
	case CASE(ORROTC, TUINT16):
		a = ARCRW;
		break;

	case CASE(ORROTC, TINT32):
	case CASE(ORROTC, TUINT32):
		a = ARCRL;
		break;

	case CASE(ORROTC, TINT64):
	case CASE(ORROTC, TUINT64):
		a = ARCRQ;
		break;

	case CASE(OHMUL, TINT8):
	case CASE(OMUL, TINT8):
	case CASE(OMUL, TUINT8):
		a = AIMULB;
		break;

	case CASE(OHMUL, TINT16):
	case CASE(OMUL, TINT16):
	case CASE(OMUL, TUINT16):
		a = AIMULW;
		break;

	case CASE(OHMUL, TINT32):
	case CASE(OMUL, TINT32):
	case CASE(OMUL, TUINT32):
	case CASE(OMUL, TPTR32):
		a = AIMULL;
		break;

	case CASE(OHMUL, TINT64):
	case CASE(OMUL, TINT64):
	case CASE(OMUL, TUINT64):
	case CASE(OMUL, TPTR64):
		a = AIMULQ;
		break;

	case CASE(OHMUL, TUINT8):
		a = AMULB;
		break;

	case CASE(OHMUL, TUINT16):
		a = AMULW;
		break;

	case CASE(OHMUL, TUINT32):
	case CASE(OHMUL, TPTR32):
		a = AMULL;
		break;

	case CASE(OHMUL, TUINT64):
	case CASE(OHMUL, TPTR64):
		a = AMULQ;
		break;

	case CASE(OMUL, TFLOAT32):
		a = AMULSS;
		break;

	case CASE(OMUL, TFLOAT64):
		a = AMULSD;
		break;

	case CASE(ODIV, TINT8):
	case CASE(OMOD, TINT8):
		a = AIDIVB;
		break;

	case CASE(ODIV, TUINT8):
	case CASE(OMOD, TUINT8):
		a = ADIVB;
		break;

	case CASE(ODIV, TINT16):
	case CASE(OMOD, TINT16):
		a = AIDIVW;
		break;

	case CASE(ODIV, TUINT16):
	case CASE(OMOD, TUINT16):
		a = ADIVW;
		break;

	case CASE(ODIV, TINT32):
	case CASE(OMOD, TINT32):
		a = AIDIVL;
		break;

	case CASE(ODIV, TUINT32):
	case CASE(ODIV, TPTR32):
	case CASE(OMOD, TUINT32):
	case CASE(OMOD, TPTR32):
		a = ADIVL;
		break;

	case CASE(ODIV, TINT64):
	case CASE(OMOD, TINT64):
		a = AIDIVQ;
		break;

	case CASE(ODIV, TUINT64):
	case CASE(ODIV, TPTR64):
	case CASE(OMOD, TUINT64):
	case CASE(OMOD, TPTR64):
		a = ADIVQ;
		break;

	case CASE(OEXTEND, TINT16):
		a = ACWD;
		break;

	case CASE(OEXTEND, TINT32):
		a = ACDQ;
		break;

	case CASE(OEXTEND, TINT64):
		a = ACQO;
		break;

	case CASE(ODIV, TFLOAT32):
		a = ADIVSS;
		break;

	case CASE(ODIV, TFLOAT64):
		a = ADIVSD;
		break;

	}
	return a;
}

enum
{
	ODynam		= 1<<0,
	OAddable	= 1<<1,
};

static	Node	clean[20];
static	int	cleani = 0;

int
xgen(Node *n, Node *a, int o)
{
	regalloc(a, types[tptr], N);

	if(o & ODynam)
	if(n->addable)
	if(n->op != OINDREG)
	if(n->op != OREGISTER)
		return 1;

	agen(n, a);
	return 0;
}

void
sudoclean(void)
{
	if(clean[cleani-1].op != OEMPTY)
		regfree(&clean[cleani-1]);
	if(clean[cleani-2].op != OEMPTY)
		regfree(&clean[cleani-2]);
	cleani -= 2;
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
sudoaddable(int as, Node *n, Addr *a)
{
	int o, i;
	int64 oary[10];
	int64 v, w;
	Node n1, n2, n3, n4, *nn, *l, *r;
	Node *reg, *reg1;
	Prog *p1;
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
		goto oindex;
	}
	return 0;

lit:
	switch(as) {
	default:
		return 0;
	case AADDB: case AADDW: case AADDL: case AADDQ:
	case ASUBB: case ASUBW: case ASUBL: case ASUBQ:
	case AANDB: case AANDW: case AANDL: case AANDQ:
	case AORB:  case AORW:  case AORL:  case AORQ:
	case AXORB: case AXORW: case AXORL: case AXORQ:
	case AINCB: case AINCW: case AINCL: case AINCQ:
	case ADECB: case ADECW: case ADECL: case ADECQ:
	case AMOVB: case AMOVW: case AMOVL: case AMOVQ:
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
		n1.xoffset = -(oary[0]+1);
	}

	for(i=1; i<o; i++) {
		if(oary[i] >= 0)
			fatal("cant happen");
		gins(AMOVQ, &n1, reg);
		n1.xoffset = -(oary[i]+1);
	}

	a->type = D_NONE;
	a->index = D_NONE;
	naddr(&n1, a, 1);
	goto yes;

oindex:
	l = n->left;
	r = n->right;
	if(l->ullman >= UINF && r->ullman >= UINF)
		return 0;

	// set o to type of array
	o = 0;
	if(isptr[l->type->etype])
		fatal("ptr ary");
	if(l->type->etype != TARRAY)
		fatal("not ary");
	if(l->type->bound < 0)
		o |= ODynam;

	w = n->type->width;
	if(isconst(r, CTINT))
		goto oindex_const;

	switch(w) {
	default:
		return 0;
	case 1:
	case 2:
	case 4:
	case 8:
		break;
	}

	cleani += 2;
	reg = &clean[cleani-1];
	reg1 = &clean[cleani-2];
	reg->op = OEMPTY;
	reg1->op = OEMPTY;

	// load the array (reg)
	if(l->ullman > r->ullman) {
		if(xgen(l, reg, o))
			o |= OAddable;
	}

	// load the index (reg1)
	t = types[TUINT64];
	if(issigned[r->type->etype])
		t = types[TINT64];
	regalloc(reg1, t, N);
	regalloc(&n3, r->type, reg1);
	cgen(r, &n3);
	gmove(&n3, reg1);
	regfree(&n3);

	// load the array (reg)
	if(l->ullman <= r->ullman) {
		if(xgen(l, reg, o))
			o |= OAddable;
	}

	if(!(o & ODynam) && l->type->width >= unmappedzero && l->op == OIND) {
		// cannot rely on page protections to
		// catch array ptr == 0, so dereference.
		n2 = *reg;
		n2.xoffset = 0;
		n2.op = OINDREG;
		n2.type = types[TUINT8];
		gins(ATESTB, nodintconst(0), &n2);
	}

	// check bounds
	if(!debug['B'] && !n->bounded) {
		// check bounds
		n4.op = OXXX;
		t = types[simtype[TUINT]];
		if(o & ODynam) {
			if(o & OAddable) {
				n2 = *l;
				n2.xoffset += Array_nel;
				n2.type = types[simtype[TUINT]];
			} else {
				n2 = *reg;
				n2.xoffset = Array_nel;
				n2.op = OINDREG;
				n2.type = types[simtype[TUINT]];
			}
		} else {
			if(is64(r->type))
				t = types[TUINT64];
			nodconst(&n2, types[TUINT64], l->type->bound);
		}
		gins(optoas(OCMP, t), reg1, &n2);
		p1 = gbranch(optoas(OLT, t), T, +1);
		if(n4.op != OXXX)
			regfree(&n4);
		ginscall(panicindex, -1);
		patch(p1, pc);
	}

	if(o & ODynam) {
		if(o & OAddable) {
			n2 = *l;
			n2.xoffset += Array_array;
			n2.type = types[tptr];
			gmove(&n2, reg);
		} else {
			n2 = *reg;
			n2.op = OINDREG;
			n2.xoffset = Array_array;
			n2.type = types[tptr];
			gmove(&n2, reg);
		}
	}

	if(o & OAddable) {
		naddr(reg1, a, 1);
		a->offset = 0;
		a->scale = w;
		a->index = a->type;
		a->type = reg->val.u.reg + D_INDIR;
	} else {
		naddr(reg1, a, 1);
		a->offset = 0;
		a->scale = w;
		a->index = a->type;
		a->type = reg->val.u.reg + D_INDIR;
	}

	goto yes;

oindex_const:
	// index is constant
	// can check statically and
	// can multiply by width statically

	v = mpgetfix(r->val.u.xval);

	if(sudoaddable(as, l, a))
		goto oindex_const_sudo;

	cleani += 2;
	reg = &clean[cleani-1];
	reg1 = &clean[cleani-2];
	reg->op = OEMPTY;
	reg1->op = OEMPTY;

	if(o & ODynam) {
		regalloc(reg, types[tptr], N);
		agen(l, reg);
	
		if(!debug['B'] && !n->bounded) {
			n1 = *reg;
			n1.op = OINDREG;
			n1.type = types[tptr];
			n1.xoffset = Array_nel;
			nodconst(&n2, types[TUINT64], v);
			gins(optoas(OCMP, types[simtype[TUINT]]), &n1, &n2);
			p1 = gbranch(optoas(OGT, types[simtype[TUINT]]), T, +1);
			ginscall(panicindex, -1);
			patch(p1, pc);
		}

		n1 = *reg;
		n1.op = OINDREG;
		n1.type = types[tptr];
		n1.xoffset = Array_array;
		gmove(&n1, reg);

		n2 = *reg;
		n2.op = OINDREG;
		n2.xoffset = v*w;
		a->type = D_NONE;
		a->index = D_NONE;
		naddr(&n2, a, 1);
		goto yes;
	}
	
	igen(l, &n1, N);
	if(n1.op == OINDREG) {
		*reg = n1;
		reg->op = OREGISTER;
	}
	n1.xoffset += v*w;
	a->type = D_NONE;
	a->index= D_NONE;
	naddr(&n1, a, 1);
	goto yes;

oindex_const_sudo:
	if((o & ODynam) == 0) {
		// array indexed by a constant
		a->offset += v*w;
		goto yes;
	}

	// slice indexed by a constant
	if(!debug['B'] && !n->bounded) {
		a->offset += Array_nel;
		nodconst(&n2, types[TUINT64], v);
		p1 = gins(optoas(OCMP, types[simtype[TUINT]]), N, &n2);
		p1->from = *a;
		p1 = gbranch(optoas(OGT, types[simtype[TUINT]]), T, +1);
		ginscall(panicindex, -1);
		patch(p1, pc);
		a->offset -= Array_nel;
	}

	a->offset += Array_array;
	reg = &clean[cleani-1];
	if(reg->op == OEMPTY)
		regalloc(reg, types[tptr], N);

	p1 = gins(AMOVQ, N, reg);
	p1->from = *a;

	n2 = *reg;
	n2.op = OINDREG;
	n2.xoffset = v*w;
	a->type = D_NONE;
	a->index = D_NONE;
	naddr(&n2, a, 1);
	goto yes;

yes:
	return 1;

no:
	sudoclean();
	return 0;
}
