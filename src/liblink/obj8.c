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
#include "../runtime/stack.h"

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

static int
symtype(Addr *a)
{
	int t;

	t = a->type;
	if(t == D_ADDR)
		t = a->index;
	return t;
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

static int
canuselocaltls(Link *ctxt)
{
	switch(ctxt->headtype) {
	case Hlinux:
	case Hnacl:
	case Hplan9:
	case Hwindows:
		return 0;
	}
	return 1;
}

static void
progedit(Link *ctxt, Prog *p)
{
	char literal[64];
	LSym *s;
	Prog *q;
	
	// See obj6.c for discussion of TLS.
	if(canuselocaltls(ctxt)) {
		// Reduce TLS initial exec model to TLS local exec model.
		// Sequences like
		//	MOVL TLS, BX
		//	... off(BX)(TLS*1) ...
		// become
		//	NOP
		//	... off(TLS) ...
		if(p->as == AMOVL && p->from.type == D_TLS && D_AX <= p->to.type && p->to.type <= D_DI) {
			p->as = ANOP;
			p->from.type = D_NONE;
			p->to.type = D_NONE;
		}
		if(p->from.index == D_TLS && D_INDIR+D_AX <= p->from.type && p->from.type <= D_INDIR+D_DI) {
			p->from.type = D_INDIR+D_TLS;
			p->from.scale = 0;
			p->from.index = D_NONE;
		}
		if(p->to.index == D_TLS && D_INDIR+D_AX <= p->to.type && p->to.type <= D_INDIR+D_DI) {
			p->to.type = D_INDIR+D_TLS;
			p->to.scale = 0;
			p->to.index = D_NONE;
		}
	} else {
		// As a courtesy to the C compilers, rewrite TLS local exec load as TLS initial exec load.
		// The instruction
		//	MOVL off(TLS), BX
		// becomes the sequence
		//	MOVL TLS, BX
		//	MOVL off(BX)(TLS*1), BX
		// This allows the C compilers to emit references to m and g using the direct off(TLS) form.
		if(p->as == AMOVL && p->from.type == D_INDIR+D_TLS && D_AX <= p->to.type && p->to.type <= D_DI) {
			q = appendp(ctxt, p);
			q->as = p->as;
			q->from = p->from;
			q->from.type = D_INDIR + p->to.type;
			q->from.index = D_TLS;
			q->from.scale = 2; // TODO: use 1
			q->to = p->to;
			p->from.type = D_TLS;
			p->from.index = D_NONE;
			p->from.offset = 0;
		}
	}

	// TODO: Remove.
	if(ctxt->headtype == Hplan9) {
		if(p->from.scale == 1 && p->from.index == D_TLS)
			p->from.scale = 2;
		if(p->to.scale == 1 && p->to.index == D_TLS)
			p->to.scale = 2;
	}

	// Rewrite CALL/JMP/RET to symbol as D_BRANCH.
	switch(p->as) {
	case ACALL:
	case AJMP:
	case ARET:
		if((p->to.type == D_EXTERN || p->to.type == D_STATIC) && p->to.sym != nil)
			p->to.type = D_BRANCH;
		break;
	}

	// Rewrite float constants to values stored in memory.
	switch(p->as) {
	case AFMOVF:
	case AFADDF:
	case AFSUBF:
	case AFSUBRF:
	case AFMULF:
	case AFDIVF:
	case AFDIVRF:
	case AFCOMF:
	case AFCOMFP:
	case AMOVSS:
	case AADDSS:
	case ASUBSS:
	case AMULSS:
	case ADIVSS:
	case ACOMISS:
	case AUCOMISS:
		if(p->from.type == D_FCONST) {
			int32 i32;
			float32 f32;
			f32 = p->from.u.dval;
			memmove(&i32, &f32, 4);
			sprint(literal, "$f32.%08ux", (uint32)i32);
			s = linklookup(ctxt, literal, 0);
			if(s->type == 0) {
				s->type = SRODATA;
				adduint32(ctxt, s, i32);
				s->reachable = 0;
			}
			p->from.type = D_EXTERN;
			p->from.sym = s;
			p->from.offset = 0;
		}
		break;

	case AFMOVD:
	case AFADDD:
	case AFSUBD:
	case AFSUBRD:
	case AFMULD:
	case AFDIVD:
	case AFDIVRD:
	case AFCOMD:
	case AFCOMDP:
	case AMOVSD:
	case AADDSD:
	case ASUBSD:
	case AMULSD:
	case ADIVSD:
	case ACOMISD:
	case AUCOMISD:
		if(p->from.type == D_FCONST) {
			int64 i64;
			memmove(&i64, &p->from.u.dval, 8);
			sprint(literal, "$f64.%016llux", (uvlong)i64);
			s = linklookup(ctxt, literal, 0);
			if(s->type == 0) {
				s->type = SRODATA;
				adduint64(ctxt, s, i64);
				s->reachable = 0;
			}
			p->from.type = D_EXTERN;
			p->from.sym = s;
			p->from.offset = 0;
		}
		break;
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
static Prog*	stacksplit(Link*, Prog*, int32, int, Prog**);

static void
addstacksplit(Link *ctxt, LSym *cursym)
{
	Prog *p, *q, *p1, *p2;
	int32 autoffset, deltasp;
	int a;

	if(ctxt->symmorestack[0] == nil) {
		ctxt->symmorestack[0] = linklookup(ctxt, "runtime.morestack", 0);
		ctxt->symmorestack[1] = linklookup(ctxt, "runtime.morestack_noctxt", 0);
	}

	if(ctxt->headtype == Hplan9 && ctxt->plan9privates == nil)
		ctxt->plan9privates = linklookup(ctxt, "_privates", 0);

	ctxt->cursym = cursym;

	if(cursym->text == nil || cursym->text->link == nil)
		return;

	p = cursym->text;
	autoffset = p->to.offset;
	if(autoffset < 0)
		autoffset = 0;
	
	cursym->locals = autoffset;
	cursym->args = p->to.offset2;

	q = nil;

	if(!(p->from.scale & NOSPLIT) || (p->from.scale & WRAPPER)) {
		p = appendp(ctxt, p);
		p = load_g_cx(ctxt, p); // load g into CX
	}
	if(!(cursym->text->from.scale & NOSPLIT))
		p = stacksplit(ctxt, p, autoffset, !(cursym->text->from.scale&NEEDCTXT), &q); // emit split check

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
		// if(g->panic != nil && g->panic->argp == FP) g->panic->argp = bottom-of-frame
		//
		//	MOVL g_panic(CX), BX
		//	TESTL BX, BX
		//	JEQ end
		//	LEAL (autoffset+4)(SP), DI
		//	CMPL panic_argp(BX), DI
		//	JNE end
		//	MOVL SP, panic_argp(BX)
		// end:
		//	NOP
		//
		// The NOP is needed to give the jumps somewhere to land.
		// It is a liblink NOP, not an x86 NOP: it encodes to 0 instruction bytes.

		p = appendp(ctxt, p);
		p->as = AMOVL;
		p->from.type = D_INDIR+D_CX;
		p->from.offset = 4*ctxt->arch->ptrsize; // G.panic
		p->to.type = D_BX;

		p = appendp(ctxt, p);
		p->as = ATESTL;
		p->from.type = D_BX;
		p->to.type = D_BX;

		p = appendp(ctxt, p);
		p->as = AJEQ;
		p->to.type = D_BRANCH;
		p1 = p;

		p = appendp(ctxt, p);
		p->as = ALEAL;
		p->from.type = D_INDIR+D_SP;
		p->from.offset = autoffset+4;
		p->to.type = D_DI;

		p = appendp(ctxt, p);
		p->as = ACMPL;
		p->from.type = D_INDIR+D_BX;
		p->from.offset = 0; // Panic.argp
		p->to.type = D_DI;

		p = appendp(ctxt, p);
		p->as = AJNE;
		p->to.type = D_BRANCH;
		p2 = p;

		p = appendp(ctxt, p);
		p->as = AMOVL;
		p->from.type = D_SP;
		p->to.type = D_INDIR+D_BX;
		p->to.offset = 0; // Panic.argp

		p = appendp(ctxt, p);
		p->as = ANOP;
		p1->pcond = p;
		p2->pcond = p;
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
	Prog *next;

	p->as = AMOVL;
	p->from.type = D_INDIR+D_TLS;
	p->from.offset = 0;
	p->to.type = D_CX;

	next = p->link;
	progedit(ctxt, p);
	while(p->link != next)
		p = p->link;
	
	if(p->from.index == D_TLS)
		p->from.scale = 2;

	return p;
}

// Append code to p to check for stack split.
// Appends to (does not overwrite) p.
// Assumes g is in CX.
// Returns last new instruction.
// On return, *jmpok is the instruction that should jump
// to the stack frame allocation if no split is needed.
static Prog*
stacksplit(Link *ctxt, Prog *p, int32 framesize, int noctxt, Prog **jmpok)
{
	Prog *q, *q1;

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
		p->to.offset = 2*ctxt->arch->ptrsize;	// G.stackguard0
		if(ctxt->cursym->cfunc)
			p->to.offset = 3*ctxt->arch->ptrsize;	// G.stackguard1
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
		p->to.offset = 2*ctxt->arch->ptrsize;	// G.stackguard0
		if(ctxt->cursym->cfunc)
			p->to.offset = 3*ctxt->arch->ptrsize;	// G.stackguard1
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
		p->from.offset = 2*ctxt->arch->ptrsize;	// G.stackguard0
		if(ctxt->cursym->cfunc)
			p->from.offset = 3*ctxt->arch->ptrsize;	// G.stackguard1
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

	p = appendp(ctxt, p);
	p->as = ACALL;
	p->to.type = D_BRANCH;
	if(ctxt->cursym->cfunc)
		p->to.sym = linklookup(ctxt, "runtime.morestackc", 0);
	else
		p->to.sym = ctxt->symmorestack[noctxt];

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
	int a;

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
	.thechar = '8',
	.endian = LittleEndian,

	.addstacksplit = addstacksplit,
	.assemble = span8,
	.datasize = datasize,
	.follow = follow,
	.iscall = iscall,
	.isdata = isdata,
	.prg = prg,
	.progedit = progedit,
	.settextflag = settextflag,
	.symtype = symtype,
	.textflag = textflag,

	.minlc = 1,
	.ptrsize = 4,
	.regsize = 4,

	.D_ADDR = D_ADDR,
	.D_AUTO = D_AUTO,
	.D_BRANCH = D_BRANCH,
	.D_CONST = D_CONST,
	.D_EXTERN = D_EXTERN,
	.D_FCONST = D_FCONST,
	.D_NONE = D_NONE,
	.D_PARAM = D_PARAM,
	.D_SCONST = D_SCONST,
	.D_STATIC = D_STATIC,

	.ACALL = ACALL,
	.ADATA = ADATA,
	.AEND = AEND,
	.AFUNCDATA = AFUNCDATA,
	.AGLOBL = AGLOBL,
	.AJMP = AJMP,
	.ANOP = ANOP,
	.APCDATA = APCDATA,
	.ARET = ARET,
	.ATEXT = ATEXT,
	.ATYPE = ATYPE,
	.AUSEFIELD = AUSEFIELD,
};
