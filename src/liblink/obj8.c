// Inferno utils/8l/pass.c
// http://code.google.com/p/inferno-os/source/browse/utils/8l/pass.c
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
#include <bio.h>
#include <link.h>
#include "../cmd/8l/8.out.h"
#include "../pkg/runtime/stack.h"

static Addr noaddr = {
	.type = D_NONE,
	.index = D_NONE,
	.scale = 1,
};

static Prog zprg = {
	.back = 2,
	.as = AGOK,
	.from = {
		.type = D_NONE,
		.index = D_NONE,
		.scale = 1,
	},
	.to = {
		.type = D_NONE,
		.index = D_NONE,
		.scale = 1,
	},
};

static void
zname(Biobuf *b, LSym *s, int t)
{
	BPUTLE2(b, ANAME);	/* as */
	BPUTC(b, t);		/* type */
	BPUTC(b, s->symid);	/* sym */
	Bwrite(b, s->name, strlen(s->name)+1);
}

static void
zfile(Biobuf *b, char *p, int n)
{
	BPUTLE2(b, ANAME);
	BPUTC(b, D_FILE);
	BPUTC(b, 1);
	BPUTC(b, '<');
	Bwrite(b, p, n);
	BPUTC(b, 0);
}

static void
zaddr(Biobuf *b, Addr *a, int s, int gotype)
{
	int32 l;
	uint64 e;
	int i, t;
	char *n;

	t = 0;
	if(a->index != D_NONE || a->scale != 0)
		t |= T_INDEX;
	if(s != 0)
		t |= T_SYM;
	if(gotype != 0)
		t |= T_GOTYPE;

	switch(a->type) {

	case D_BRANCH:
		if(a->offset == 0 || a->u.branch != nil) {
			if(a->u.branch == nil)
				sysfatal("unpatched branch %D", a);
			a->offset = a->u.branch->loc;
		}

	default:
		t |= T_TYPE;

	case D_NONE:
		if(a->offset != 0)
			t |= T_OFFSET;
		if(a->offset2 != 0)
			t |= T_OFFSET2;
		break;
	case D_FCONST:
		t |= T_FCONST;
		break;
	case D_SCONST:
		t |= T_SCONST;
		break;
	}
	BPUTC(b, t);

	if(t & T_INDEX) {	/* implies index, scale */
		BPUTC(b, a->index);
		BPUTC(b, a->scale);
	}
	if(t & T_OFFSET) {	/* implies offset */
		l = a->offset;
		BPUTLE4(b, l);
	}
	if(t & T_OFFSET2) {	/* implies offset */
		l = a->offset2;
		BPUTLE4(b, l);
	}
	if(t & T_SYM)		/* implies sym */
		BPUTC(b, s);
	if(t & T_FCONST) {
		double2ieee(&e, a->u.dval);
		BPUTLE4(b, e);
		BPUTLE4(b, e >> 32);
		return;
	}
	if(t & T_SCONST) {
		n = a->u.sval;
		for(i=0; i<NSNAME; i++) {
			BPUTC(b, *n);
			n++;
		}
		return;
	}
	if(t & T_TYPE)
		BPUTC(b, a->type);
	if(t & T_GOTYPE)
		BPUTC(b, gotype);
}

static void
zhist(Biobuf *b, int line, vlong offset)
{
	Addr a;

	BPUTLE2(b, AHISTORY);
	BPUTLE4(b, line);
	zaddr(b, &noaddr, 0, 0);
	a = noaddr;
	if(offset != 0) {
		a.offset = offset;
		a.type = D_CONST;
	}
	zaddr(b, &a, 0, 0);
}

static int
symtype(Addr *a)
{
	int t;

	t = a->type;
	if(t == D_ADDR)
		t = a->index;
	return t;
}

static void
zprog(Link *ctxt, Biobuf *b, Prog *p, int sf, int gf, int st, int gt)
{
	USED(ctxt);

	BPUTLE2(b, p->as);
	BPUTLE4(b, p->lineno);
	zaddr(b, &p->from, sf, gf);
	zaddr(b, &p->to, st, gt);
}

static int
isdata(Prog *p)
{
	return p->as == ADATA || p->as == AGLOBL;
}

static int
iscall(Prog *p)
{
	return p->as == ACALL;
}

static int
datasize(Prog *p)
{
	return p->from.scale;
}

static int
textflag(Prog *p)
{
	return p->from.scale;
}

static void
settextflag(Prog *p, int f)
{
	p->from.scale = f;
}

static void
progedit(Link *ctxt, Prog *p)
{
	Prog *q;

	if(ctxt->headtype == Hwindows) {
		// Convert
		//   op	  n(GS), reg
		// to
		//   MOVL 0x14(FS), reg
		//   op	  n(reg), reg
		// The purpose of this patch is to fix some accesses
		// to extern register variables (TLS) on Windows, as
		// a different method is used to access them.
		if(p->from.type == D_INDIR+D_GS
		&& p->to.type >= D_AX && p->to.type <= D_DI) {
			q = appendp(ctxt, p);
			q->from = p->from;
			q->from.type = D_INDIR + p->to.type;
			q->to = p->to;
			q->as = p->as;
			p->as = AMOVL;
			p->from.type = D_INDIR+D_FS;
			p->from.offset = 0x14;
		}
	}
	if(ctxt->headtype == Hlinux) {
		// Running binaries under Xen requires using
		//	MOVL 0(GS), reg
		// and then off(reg) instead of saying off(GS) directly
		// when the offset is negative.
		// In external mode we just produce a reloc.
		if(p->from.type == D_INDIR+D_GS && p->from.offset < 0
		&& p->to.type >= D_AX && p->to.type <= D_DI) {
			if(ctxt->linkmode != LinkExternal) {
				q = appendp(ctxt, p);
				q->from = p->from;
				q->from.type = D_INDIR + p->to.type;
				q->to = p->to;
				q->as = p->as;
				p->as = AMOVL;
				p->from.type = D_INDIR+D_GS;
				p->from.offset = 0;
			} else {
				// Add signals to relocate.
				p->from.index = D_GS;
				p->from.scale = 1;
			}
		}
	}
	if(ctxt->headtype == Hplan9) {
		if(p->from.type == D_INDIR+D_GS
		&& p->to.type >= D_AX && p->to.type <= D_DI) {
			if(ctxt->plan9tos == nil)
				ctxt->plan9tos = linklookup(ctxt, "_tos", 0);
			q = appendp(ctxt, p);
			q->from = p->from;
			q->from.type = D_INDIR + p->to.type;
			q->to = p->to;
			q->as = p->as;
			p->as = AMOVL;
			p->from.type = D_EXTERN;
			p->from.sym = ctxt->plan9tos;
			p->from.offset = 0;
		}
	}
}

static Prog*
prg(void)
{
	Prog *p;

	p = emallocz(sizeof(*p));
	*p = zprg;
	return p;
}

static Prog*	load_g_cx(Link*, Prog*);
static Prog*	stacksplit(Link*, Prog*, int32, Prog**);

static void
addstacksplit(Link *ctxt, LSym *cursym)
{
	Prog *p, *q;
	int32 autoffset, deltasp;
	int a;

	if(ctxt->symmorestack[0] == nil)
		ctxt->symmorestack[0] = linklookup(ctxt, "runtime.morestack", 0);

	if(ctxt->headtype == Hplan9 && ctxt->plan9tos == nil)
		ctxt->plan9tos = linklookup(ctxt, "_tos", 0);

	ctxt->cursym = cursym;

	if(cursym->text == nil || cursym->text->link == nil)
		return;

	p = cursym->text;
	autoffset = p->to.offset;
	if(autoffset < 0)
		autoffset = 0;

	q = nil;

	if(!(p->from.scale & NOSPLIT) || (p->from.scale & WRAPPER)) {
		p = appendp(ctxt, p);
		p = load_g_cx(ctxt, p); // load g into CX
	}
	if(!(cursym->text->from.scale & NOSPLIT))
		p = stacksplit(ctxt, p, autoffset, &q); // emit split check

	if(autoffset) {
		p = appendp(ctxt, p);
		p->as = AADJSP;
		p->from.type = D_CONST;
		p->from.offset = autoffset;
		p->spadj = autoffset;
	} else {
		// zero-byte stack adjustment.
		// Insert a fake non-zero adjustment so that stkcheck can
		// recognize the end of the stack-splitting prolog.
		p = appendp(ctxt, p);
		p->as = ANOP;
		p->spadj = -ctxt->arch->ptrsize;
		p = appendp(ctxt, p);
		p->as = ANOP;
		p->spadj = ctxt->arch->ptrsize;
	}
	if(q != nil)
		q->pcond = p;
	deltasp = autoffset;
	
	if(cursym->text->from.scale & WRAPPER) {
		// g->panicwrap += autoffset + ctxt->arch->ptrsize;
		p = appendp(ctxt, p);
		p->as = AADDL;
		p->from.type = D_CONST;
		p->from.offset = autoffset + ctxt->arch->ptrsize;
		p->to.type = D_INDIR+D_CX;
		p->to.offset = 2*ctxt->arch->ptrsize;
	}
	
	if(ctxt->debugzerostack && autoffset && !(cursym->text->from.scale&NOSPLIT)) {
		// 8l -Z means zero the stack frame on entry.
		// This slows down function calls but can help avoid
		// false positives in garbage collection.
		p = appendp(ctxt, p);
		p->as = AMOVL;
		p->from.type = D_SP;
		p->to.type = D_DI;
		
		p = appendp(ctxt, p);
		p->as = AMOVL;
		p->from.type = D_CONST;
		p->from.offset = autoffset/4;
		p->to.type = D_CX;
		
		p = appendp(ctxt, p);
		p->as = AMOVL;
		p->from.type = D_CONST;
		p->from.offset = 0;
		p->to.type = D_AX;
		
		p = appendp(ctxt, p);
		p->as = AREP;
		
		p = appendp(ctxt, p);
		p->as = ASTOSL;
	}
	
	for(; p != nil; p = p->link) {
		a = p->from.type;
		if(a == D_AUTO)
			p->from.offset += deltasp;
		if(a == D_PARAM)
			p->from.offset += deltasp + 4;
		a = p->to.type;
		if(a == D_AUTO)
			p->to.offset += deltasp;
		if(a == D_PARAM)
			p->to.offset += deltasp + 4;

		switch(p->as) {
		default:
			continue;
		case APUSHL:
		case APUSHFL:
			deltasp += 4;
			p->spadj = 4;
			continue;
		case APUSHW:
		case APUSHFW:
			deltasp += 2;
			p->spadj = 2;
			continue;
		case APOPL:
		case APOPFL:
			deltasp -= 4;
			p->spadj = -4;
			continue;
		case APOPW:
		case APOPFW:
			deltasp -= 2;
			p->spadj = -2;
			continue;
		case ARET:
			break;
		}

		if(autoffset != deltasp)
			ctxt->diag("unbalanced PUSH/POP");

		if(cursym->text->from.scale & WRAPPER) {
			p = load_g_cx(ctxt, p);
			p = appendp(ctxt, p);
			// g->panicwrap -= autoffset + ctxt->arch->ptrsize;
			p->as = ASUBL;
			p->from.type = D_CONST;
			p->from.offset = autoffset + ctxt->arch->ptrsize;
			p->to.type = D_INDIR+D_CX;
			p->to.offset = 2*ctxt->arch->ptrsize;
			p = appendp(ctxt, p);
			p->as = ARET;
		}

		if(autoffset) {
			p->as = AADJSP;
			p->from.type = D_CONST;
			p->from.offset = -autoffset;
			p->spadj = -autoffset;
			p = appendp(ctxt, p);
			p->as = ARET;
			// If there are instructions following
			// this ARET, they come from a branch
			// with the same stackframe, so undo
			// the cleanup.
			p->spadj = +autoffset;
		}
		if(p->to.sym) // retjmp
			p->as = AJMP;
	}
}

// Append code to p to load g into cx.
// Overwrites p with the first instruction (no first appendp).
// Overwriting p is unusual but it lets use this in both the
// prologue (caller must call appendp first) and in the epilogue.
// Returns last new instruction.
static Prog*
load_g_cx(Link *ctxt, Prog *p)
{
	switch(ctxt->headtype) {
	case Hwindows:
		p->as = AMOVL;
		p->from.type = D_INDIR+D_FS;
		p->from.offset = 0x14;
		p->to.type = D_CX;

		p = appendp(ctxt, p);
		p->as = AMOVL;
		p->from.type = D_INDIR+D_CX;
		p->from.offset = 0;
		p->to.type = D_CX;
		break;
	
	case Hlinux:
		if(ctxt->linkmode != LinkExternal) {
			p->as = AMOVL;
			p->from.type = D_INDIR+D_GS;
			p->from.offset = 0;
			p->to.type = D_CX;

			p = appendp(ctxt, p);
			p->as = AMOVL;
			p->from.type = D_INDIR+D_CX;
			p->from.offset = ctxt->tlsoffset + 0;
			p->to.type = D_CX;
		} else {
			p->as = AMOVL;
			p->from.type = D_INDIR+D_GS;
			p->from.offset = ctxt->tlsoffset + 0;
			p->to.type = D_CX;
			p->from.index = D_GS;
			p->from.scale = 1;
		}
		break;
	
	case Hplan9:
		p->as = AMOVL;
		p->from.type = D_EXTERN;
		p->from.sym = ctxt->plan9tos;
		p->to.type = D_CX;
		
		p = appendp(ctxt, p);
		p->as = AMOVL;
		p->from.type = D_INDIR+D_CX;
		p->from.offset = ctxt->tlsoffset + 0;
		p->to.type = D_CX;				
		break;
	
	default:
		p->as = AMOVL;
		p->from.type = D_INDIR+D_GS;
		p->from.offset = ctxt->tlsoffset + 0;
		p->to.type = D_CX;
	}
	return p;
}

// Append code to p to check for stack split.
// Appends to (does not overwrite) p.
// Assumes g is in CX.
// Returns last new instruction.
// On return, *jmpok is the instruction that should jump
// to the stack frame allocation if no split is needed.
static Prog*
stacksplit(Link *ctxt, Prog *p, int32 framesize, Prog **jmpok)
{
	Prog *q, *q1;
	int arg;

	if(ctxt->debugstack) {
		// 8l -K means check not only for stack
		// overflow but stack underflow.
		// On underflow, INT 3 (breakpoint).
		// Underflow itself is rare but this also
		// catches out-of-sync stack guard info.
		p = appendp(ctxt, p);
		p->as = ACMPL;
		p->from.type = D_INDIR+D_CX;
		p->from.offset = 4;
		p->to.type = D_SP;

		p = appendp(ctxt, p);
		p->as = AJCC;
		p->to.type = D_BRANCH;
		p->to.offset = 4;
		q1 = p;

		p = appendp(ctxt, p);
		p->as = AINT;
		p->from.type = D_CONST;
		p->from.offset = 3;
		
		p = appendp(ctxt, p);
		p->as = ANOP;
		q1->pcond = p;
	}
	q1 = nil;

	if(framesize <= StackSmall) {
		// small stack: SP <= stackguard
		//	CMPL SP, stackguard
		p = appendp(ctxt, p);
		p->as = ACMPL;
		p->from.type = D_SP;
		p->to.type = D_INDIR+D_CX;
	} else if(framesize <= StackBig) {
		// large stack: SP-framesize <= stackguard-StackSmall
		//	LEAL -(framesize-StackSmall)(SP), AX
		//	CMPL AX, stackguard
		p = appendp(ctxt, p);
		p->as = ALEAL;
		p->from.type = D_INDIR+D_SP;
		p->from.offset = -(framesize-StackSmall);
		p->to.type = D_AX;

		p = appendp(ctxt, p);
		p->as = ACMPL;
		p->from.type = D_AX;
		p->to.type = D_INDIR+D_CX;
	} else {
		// Such a large stack we need to protect against wraparound
		// if SP is close to zero.
		//	SP-stackguard+StackGuard <= framesize + (StackGuard-StackSmall)
		// The +StackGuard on both sides is required to keep the left side positive:
		// SP is allowed to be slightly below stackguard. See stack.h.
		//
		// Preemption sets stackguard to StackPreempt, a very large value.
		// That breaks the math above, so we have to check for that explicitly.
		//	MOVL	stackguard, CX
		//	CMPL	CX, $StackPreempt
		//	JEQ	label-of-call-to-morestack
		//	LEAL	StackGuard(SP), AX
		//	SUBL	stackguard, AX
		//	CMPL	AX, $(framesize+(StackGuard-StackSmall))
		p = appendp(ctxt, p);
		p->as = AMOVL;
		p->from.type = D_INDIR+D_CX;
		p->from.offset = 0;
		p->to.type = D_SI;

		p = appendp(ctxt, p);
		p->as = ACMPL;
		p->from.type = D_SI;
		p->to.type = D_CONST;
		p->to.offset = (uint32)StackPreempt;

		p = appendp(ctxt, p);
		p->as = AJEQ;
		p->to.type = D_BRANCH;
		q1 = p;

		p = appendp(ctxt, p);
		p->as = ALEAL;
		p->from.type = D_INDIR+D_SP;
		p->from.offset = StackGuard;
		p->to.type = D_AX;
		
		p = appendp(ctxt, p);
		p->as = ASUBL;
		p->from.type = D_SI;
		p->from.offset = 0;
		p->to.type = D_AX;
		
		p = appendp(ctxt, p);
		p->as = ACMPL;
		p->from.type = D_AX;
		p->to.type = D_CONST;
		p->to.offset = framesize+(StackGuard-StackSmall);
	}		
			
	// common
	p = appendp(ctxt, p);
	p->as = AJHI;
	p->to.type = D_BRANCH;
	p->to.offset = 4;
	q = p;

	p = appendp(ctxt, p);	// save frame size in DI
	p->as = AMOVL;
	p->to.type = D_DI;
	p->from.type = D_CONST;

	// If we ask for more stack, we'll get a minimum of StackMin bytes.
	// We need a stack frame large enough to hold the top-of-stack data,
	// the function arguments+results, our caller's PC, our frame,
	// a word for the return PC of the next call, and then the StackLimit bytes
	// that must be available on entry to any function called from a function
	// that did a stack check.  If StackMin is enough, don't ask for a specific
	// amount: then we can use the custom functions and save a few
	// instructions.
	if(StackTop + ctxt->cursym->text->to.offset2 + ctxt->arch->ptrsize + framesize + ctxt->arch->ptrsize + StackLimit >= StackMin)
		p->from.offset = (framesize+7) & ~7LL;

	arg = ctxt->cursym->text->to.offset2;
	if(arg == 1) // special marker for known 0
		arg = 0;
	if(arg&3)
		ctxt->diag("misaligned argument size in stack split");
	p = appendp(ctxt, p);	// save arg size in AX
	p->as = AMOVL;
	p->to.type = D_AX;
	p->from.type = D_CONST;
	p->from.offset = arg;

	p = appendp(ctxt, p);
	p->as = ACALL;
	p->to.type = D_BRANCH;
	p->to.sym = ctxt->symmorestack[0];

	p = appendp(ctxt, p);
	p->as = AJMP;
	p->to.type = D_BRANCH;
	p->pcond = ctxt->cursym->text->link;

	if(q != nil)
		q->pcond = p->link;
	if(q1 != nil)
		q1->pcond = q->link;
	
	*jmpok = q;
	return p;
}

static void xfol(Link*, Prog*, Prog**);

static void
follow(Link *ctxt, LSym *s)
{
	Prog *firstp, *lastp;

	ctxt->cursym = s;

	firstp = ctxt->arch->prg();
	lastp = firstp;
	xfol(ctxt, s->text, &lastp);
	lastp->link = nil;
	s->text = firstp->link;
}

static int
nofollow(int a)
{
	switch(a) {
	case AJMP:
	case ARET:
	case AIRETL:
	case AIRETW:
	case AUNDEF:
		return 1;
	}
	return 0;
}

static int
pushpop(int a)
{
	switch(a) {
	case APUSHL:
	case APUSHFL:
	case APUSHW:
	case APUSHFW:
	case APOPL:
	case APOPFL:
	case APOPW:
	case APOPFW:
		return 1;
	}
	return 0;
}

static int
relinv(int a)
{

	switch(a) {
	case AJEQ:	return AJNE;
	case AJNE:	return AJEQ;
	case AJLE:	return AJGT;
	case AJLS:	return AJHI;
	case AJLT:	return AJGE;
	case AJMI:	return AJPL;
	case AJGE:	return AJLT;
	case AJPL:	return AJMI;
	case AJGT:	return AJLE;
	case AJHI:	return AJLS;
	case AJCS:	return AJCC;
	case AJCC:	return AJCS;
	case AJPS:	return AJPC;
	case AJPC:	return AJPS;
	case AJOS:	return AJOC;
	case AJOC:	return AJOS;
	}
	sysfatal("unknown relation: %s", anames8[a]);
	return 0;
}

static void
xfol(Link *ctxt, Prog *p, Prog **last)
{
	Prog *q;
	int i;
	enum as a;

loop:
	if(p == nil)
		return;
	if(p->as == AJMP)
	if((q = p->pcond) != nil && q->as != ATEXT) {
		/* mark instruction as done and continue layout at target of jump */
		p->mark = 1;
		p = q;
		if(p->mark == 0)
			goto loop;
	}
	if(p->mark) {
		/* 
		 * p goes here, but already used it elsewhere.
		 * copy up to 4 instructions or else branch to other copy.
		 */
		for(i=0,q=p; i<4; i++,q=q->link) {
			if(q == nil)
				break;
			if(q == *last)
				break;
			a = q->as;
			if(a == ANOP) {
				i--;
				continue;
			}
			if(nofollow(a) || pushpop(a))	
				break;	// NOTE(rsc): arm does goto copy
			if(q->pcond == nil || q->pcond->mark)
				continue;
			if(a == ACALL || a == ALOOP)
				continue;
			for(;;) {
				if(p->as == ANOP) {
					p = p->link;
					continue;
				}
				q = copyp(ctxt, p);
				p = p->link;
				q->mark = 1;
				(*last)->link = q;
				*last = q;
				if(q->as != a || q->pcond == nil || q->pcond->mark)
					continue;

				q->as = relinv(q->as);
				p = q->pcond;
				q->pcond = q->link;
				q->link = p;
				xfol(ctxt, q->link, last);
				p = q->link;
				if(p->mark)
					return;
				goto loop;
			}
		} /* */
		q = ctxt->arch->prg();
		q->as = AJMP;
		q->lineno = p->lineno;
		q->to.type = D_BRANCH;
		q->to.offset = p->pc;
		q->pcond = p;
		p = q;
	}
	
	/* emit p */
	p->mark = 1;
	(*last)->link = p;
	*last = p;
	a = p->as;

	/* continue loop with what comes after p */
	if(nofollow(a))
		return;
	if(p->pcond != nil && a != ACALL) {
		/*
		 * some kind of conditional branch.
		 * recurse to follow one path.
		 * continue loop on the other.
		 */
		if((q = brchain(ctxt, p->pcond)) != nil)
			p->pcond = q;
		if((q = brchain(ctxt, p->link)) != nil)
			p->link = q;
		if(p->from.type == D_CONST) {
			if(p->from.offset == 1) {
				/*
				 * expect conditional jump to be taken.
				 * rewrite so that's the fall-through case.
				 */
				p->as = relinv(a);
				q = p->link;
				p->link = p->pcond;
				p->pcond = q;
			}
		} else {
			q = p->link;
			if(q->mark)
			if(a != ALOOP) {
				p->as = relinv(a);
				p->link = p->pcond;
				p->pcond = q;
			}
		}
		xfol(ctxt, p->link, last);
		if(p->pcond->mark)
			return;
		p = p->pcond;
		goto loop;
	}
	p = p->link;
	goto loop;
}

LinkArch link386 = {
	.name = "386",

	.addstacksplit = addstacksplit,
	.assemble = span8,
	.datasize = datasize,
	.follow = follow,
	.iscall = iscall,
	.isdata = isdata,
	.ldobj = ldobj8,
	.nopout = nopout8,
	.prg = prg,
	.progedit = progedit,
	.settextflag = settextflag,
	.symtype = symtype,
	.textflag = textflag,
	.zfile = zfile,
	.zhist = zhist,
	.zname = zname,
	.zprog = zprog,

	.minlc = 1,
	.ptrsize = 4,

	.D_ADDR = D_ADDR,
	.D_BRANCH = D_BRANCH,
	.D_CONST = D_CONST,
	.D_EXTERN = D_EXTERN,
	.D_FCONST = D_FCONST,
	.D_NONE = D_NONE,
	.D_PCREL = D_PCREL,
	.D_SCONST = D_SCONST,
	.D_SIZE = D_SIZE,

	.ACALL = ACALL,
	.AFUNCDATA = AFUNCDATA,
	.AJMP = AJMP,
	.ANOP = ANOP,
	.APCDATA = APCDATA,
	.ARET = ARET,
	.ATEXT = ATEXT,
	.AUSEFIELD = AUSEFIELD,
};
