// Inferno utils/6l/pass.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/pass.c
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
#include "../cmd/6l/6.out.h"
#include "../pkg/runtime/stack.h"

static Addr noaddr = {
	.type = D_NONE,
	.index = D_NONE,
	.scale = 0,
};

static Prog zprg = {
	.back = 2,
	.as = AGOK,
	.from = {
		.type = D_NONE,
		.index = D_NONE,
	},
	.to = {
		.type = D_NONE,
		.index = D_NONE,
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
		if(a->offset != 0) {
			t |= T_OFFSET;
			l = a->offset;
			if((vlong)l != a->offset)
				t |= T_64;
		}
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
		if(t & T_64) {
			l = a->offset>>32;
			BPUTLE4(b, l);
		}
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

	if(ctxt->gmsym == nil)
		ctxt->gmsym = linklookup(ctxt, "runtime.tlsgm", 0);

	if(ctxt->headtype == Hwindows) { 
		// Windows
		// Convert
		//   op	  n(GS), reg
		// to
		//   MOVL 0x28(GS), reg
		//   op	  n(reg), reg
		// The purpose of this patch is to fix some accesses
		// to extern register variables (TLS) on Windows, as
		// a different method is used to access them.
		if(p->from.type == D_INDIR+D_GS
		&& p->to.type >= D_AX && p->to.type <= D_DI 
		&& p->from.offset <= 8) {
			q = appendp(ctxt, p);
			q->from = p->from;
			q->from.type = D_INDIR + p->to.type;
			q->to = p->to;
			q->as = p->as;
			p->as = AMOVQ;
			p->from.type = D_INDIR+D_GS;
			p->from.offset = 0x28;
		}
	}
	if(ctxt->headtype == Hlinux || ctxt->headtype == Hfreebsd
	|| ctxt->headtype == Hopenbsd || ctxt->headtype == Hnetbsd
	|| ctxt->headtype == Hplan9 || ctxt->headtype == Hdragonfly) {
		// ELF uses FS instead of GS.
		if(p->from.type == D_INDIR+D_GS)
			p->from.type = D_INDIR+D_FS;
		if(p->to.type == D_INDIR+D_GS)
			p->to.type = D_INDIR+D_FS;
		if(p->from.index == D_GS)
			p->from.index = D_FS;
		if(p->to.index == D_GS)
			p->to.index = D_FS;
	}
	if(!ctxt->flag_shared) {
		// Convert g() or m() accesses of the form
		//   op n(reg)(GS*1), reg
		// to
		//   op n(GS*1), reg
		if(p->from.index == D_FS || p->from.index == D_GS) {
			p->from.type = D_INDIR + p->from.index;
			p->from.index = D_NONE;
		}
		// Convert g() or m() accesses of the form
		//   op reg, n(reg)(GS*1)
		// to
		//   op reg, n(GS*1)
		if(p->to.index == D_FS || p->to.index == D_GS) {
			p->to.type = D_INDIR + p->to.index;
			p->to.index = D_NONE;
		}
		// Convert get_tls access of the form
		//   op runtime.tlsgm(SB), reg
		// to
		//   NOP
		if(ctxt->gmsym != nil && p->from.sym == ctxt->gmsym) {
			p->as = ANOP;
			p->from.type = D_NONE;
			p->to.type = D_NONE;
			p->from.sym = nil;
			p->to.sym = nil;
		}
	} else {
		// Convert TLS reads of the form
		//   op n(GS), reg
		// to
		//   MOVQ $runtime.tlsgm(SB), reg
		//   op n(reg)(GS*1), reg
		if((p->from.type == D_INDIR+D_FS || p->from.type == D_INDIR + D_GS) && p->to.type >= D_AX && p->to.type <= D_DI) {
			q = appendp(ctxt, p);
			q->to = p->to;
			q->as = p->as;
			q->from.type = D_INDIR+p->to.type;
			q->from.index = p->from.type - D_INDIR;
			q->from.scale = 1;
			q->from.offset = p->from.offset;
			p->as = AMOVQ;
			p->from.type = D_EXTERN;
			p->from.sym = ctxt->gmsym;
			p->from.offset = 0;
		}
	}
}

static char*
morename[] =
{
	"runtime.morestack00",
	"runtime.morestack10",
	"runtime.morestack01",
	"runtime.morestack11",

	"runtime.morestack8",
	"runtime.morestack16",
	"runtime.morestack24",
	"runtime.morestack32",
	"runtime.morestack40",
	"runtime.morestack48",
};

static Prog*	load_g_cx(Link*, Prog*);
static Prog*	stacksplit(Link*, Prog*, int32, int32, Prog**);

static void
parsetextconst(vlong arg, vlong *textstksiz, vlong *textarg)
{
	*textstksiz = arg & 0xffffffffLL;
	if(*textstksiz & 0x80000000LL)
		*textstksiz = -(-*textstksiz & 0xffffffffLL);

	*textarg = (arg >> 32) & 0xffffffffLL;
	if(*textarg & 0x80000000LL)
		*textarg = 0;
	*textarg = (*textarg+7) & ~7LL;
}

static void
addstacksplit(Link *ctxt, LSym *cursym)
{
	Prog *p, *q, *q1;
	int32 autoffset, deltasp;
	int a, pcsize;
	uint32 i;
	vlong textstksiz, textarg;

	if(ctxt->gmsym == nil)
		ctxt->gmsym = linklookup(ctxt, "runtime.tlsgm", 0);
	if(ctxt->symmorestack[0] == nil) {
		if(nelem(morename) > nelem(ctxt->symmorestack))
			sysfatal("Link.symmorestack needs at least %d elements", nelem(morename));
		for(i=0; i<nelem(morename); i++)
			ctxt->symmorestack[i] = linklookup(ctxt, morename[i], 0);
	}
	ctxt->cursym = cursym;

	if(cursym->text == nil || cursym->text->link == nil)
		return;				

	p = cursym->text;
	parsetextconst(p->to.offset, &textstksiz, &textarg);
	autoffset = textstksiz;
	if(autoffset < 0)
		autoffset = 0;

	if(autoffset < StackSmall && !(p->from.scale & NOSPLIT)) {
		for(q = p; q != nil; q = q->link)
			if(q->as == ACALL)
				goto noleaf;
		p->from.scale |= NOSPLIT;
	noleaf:;
	}

	if((p->from.scale & NOSPLIT) && autoffset >= StackSmall)
		ctxt->diag("nosplit func likely to overflow stack");

	q = nil;
	if(!(p->from.scale & NOSPLIT) || (p->from.scale & WRAPPER)) {
		p = appendp(ctxt, p);
		p = load_g_cx(ctxt, p); // load g into CX
	}
	if(!(cursym->text->from.scale & NOSPLIT))
		p = stacksplit(ctxt, p, autoffset, textarg, &q); // emit split check

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

	if(ctxt->debugstack > 1 && autoffset) {
		// 6l -K -K means double-check for stack overflow
		// even after calling morestack and even if the
		// function is marked as nosplit.
		p = appendp(ctxt, p);
		p->as = AMOVQ;
		p->from.type = D_INDIR+D_CX;
		p->from.offset = 0;
		p->to.type = D_BX;

		p = appendp(ctxt, p);
		p->as = ASUBQ;
		p->from.type = D_CONST;
		p->from.offset = StackSmall+32;
		p->to.type = D_BX;

		p = appendp(ctxt, p);
		p->as = ACMPQ;
		p->from.type = D_SP;
		p->to.type = D_BX;

		p = appendp(ctxt, p);
		p->as = AJHI;
		p->to.type = D_BRANCH;
		q1 = p;

		p = appendp(ctxt, p);
		p->as = AINT;
		p->from.type = D_CONST;
		p->from.offset = 3;

		p = appendp(ctxt, p);
		p->as = ANOP;
		q1->pcond = p;
	}
	
	if(ctxt->debugzerostack && autoffset && !(cursym->text->from.scale&NOSPLIT)) {
		// 6l -Z means zero the stack frame on entry.
		// This slows down function calls but can help avoid
		// false positives in garbage collection.
		p = appendp(ctxt, p);
		p->as = AMOVQ;
		p->from.type = D_SP;
		p->to.type = D_DI;
		
		p = appendp(ctxt, p);
		p->as = AMOVQ;
		p->from.type = D_CONST;
		p->from.offset = autoffset/8;
		p->to.type = D_CX;
		
		p = appendp(ctxt, p);
		p->as = AMOVQ;
		p->from.type = D_CONST;
		p->from.offset = 0;
		p->to.type = D_AX;
		
		p = appendp(ctxt, p);
		p->as = AREP;
		
		p = appendp(ctxt, p);
		p->as = ASTOSQ;
	}
	
	for(; p != nil; p = p->link) {
		pcsize = p->mode/8;
		a = p->from.type;
		if(a == D_AUTO)
			p->from.offset += deltasp;
		if(a == D_PARAM)
			p->from.offset += deltasp + pcsize;
		a = p->to.type;
		if(a == D_AUTO)
			p->to.offset += deltasp;
		if(a == D_PARAM)
			p->to.offset += deltasp + pcsize;

		switch(p->as) {
		default:
			continue;
		case APUSHL:
		case APUSHFL:
			deltasp += 4;
			p->spadj = 4;
			continue;
		case APUSHQ:
		case APUSHFQ:
			deltasp += 8;
			p->spadj = 8;
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
		case APOPQ:
		case APOPFQ:
			deltasp -= 8;
			p->spadj = -8;
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
	if(ctxt->flag_shared) {
		// Load TLS offset with MOVQ $runtime.tlsgm(SB), CX
		p->as = AMOVQ;
		p->from.type = D_EXTERN;
		p->from.sym = ctxt->gmsym;
		p->to.type = D_CX;
		p = appendp(ctxt, p);
	}
	p->as = AMOVQ;
	if(ctxt->headtype == Hlinux || ctxt->headtype == Hfreebsd
	|| ctxt->headtype == Hopenbsd || ctxt->headtype == Hnetbsd
	|| ctxt->headtype == Hplan9 || ctxt->headtype == Hdragonfly)
		// ELF uses FS
		p->from.type = D_INDIR+D_FS;
	else
		p->from.type = D_INDIR+D_GS;
	if(ctxt->flag_shared) {
		// Add TLS offset stored in CX
		p->from.index = p->from.type - D_INDIR;
		p->from.type = D_INDIR + D_CX;
	}
	p->from.offset = ctxt->tlsoffset+0;
	p->to.type = D_CX;
	if(ctxt->headtype == Hwindows) {
		// movq %gs:0x28, %rcx
		// movq (%rcx), %rcx
		p->as = AMOVQ;
		p->from.type = D_INDIR+D_GS;
		p->from.offset = 0x28;
		p->to.type = D_CX;

		p = appendp(ctxt, p);
		p->as = AMOVQ;
		p->from.type = D_INDIR+D_CX;
		p->from.offset = 0;
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
stacksplit(Link *ctxt, Prog *p, int32 framesize, int32 textarg, Prog **jmpok)
{
	Prog *q, *q1;
	uint32 moreconst1, moreconst2, i;

	if(ctxt->debugstack) {
		// 6l -K means check not only for stack
		// overflow but stack underflow.
		// On underflow, INT 3 (breakpoint).
		// Underflow itself is rare but this also
		// catches out-of-sync stack guard info

		p = appendp(ctxt, p);
		p->as = ACMPQ;
		p->from.type = D_INDIR+D_CX;
		p->from.offset = 8;
		p->to.type = D_SP;

		p = appendp(ctxt, p);
		p->as = AJHI;
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

	q = nil;
	q1 = nil;
	if(framesize <= StackSmall) {
		// small stack: SP <= stackguard
		//	CMPQ SP, stackguard
		p = appendp(ctxt, p);
		p->as = ACMPQ;
		p->from.type = D_SP;
		p->to.type = D_INDIR+D_CX;
	} else if(framesize <= StackBig) {
		// large stack: SP-framesize <= stackguard-StackSmall
		//	LEAQ -xxx(SP), AX
		//	CMPQ AX, stackguard
		p = appendp(ctxt, p);
		p->as = ALEAQ;
		p->from.type = D_INDIR+D_SP;
		p->from.offset = -(framesize-StackSmall);
		p->to.type = D_AX;

		p = appendp(ctxt, p);
		p->as = ACMPQ;
		p->from.type = D_AX;
		p->to.type = D_INDIR+D_CX;
	} else {
		// Such a large stack we need to protect against wraparound.
		// If SP is close to zero:
		//	SP-stackguard+StackGuard <= framesize + (StackGuard-StackSmall)
		// The +StackGuard on both sides is required to keep the left side positive:
		// SP is allowed to be slightly below stackguard. See stack.h.
		//
		// Preemption sets stackguard to StackPreempt, a very large value.
		// That breaks the math above, so we have to check for that explicitly.
		//	MOVQ	stackguard, CX
		//	CMPQ	CX, $StackPreempt
		//	JEQ	label-of-call-to-morestack
		//	LEAQ	StackGuard(SP), AX
		//	SUBQ	CX, AX
		//	CMPQ	AX, $(framesize+(StackGuard-StackSmall))

		p = appendp(ctxt, p);
		p->as = AMOVQ;
		p->from.type = D_INDIR+D_CX;
		p->from.offset = 0;
		p->to.type = D_SI;

		p = appendp(ctxt, p);
		p->as = ACMPQ;
		p->from.type = D_SI;
		p->to.type = D_CONST;
		p->to.offset = StackPreempt;

		p = appendp(ctxt, p);
		p->as = AJEQ;
		p->to.type = D_BRANCH;
		q1 = p;

		p = appendp(ctxt, p);
		p->as = ALEAQ;
		p->from.type = D_INDIR+D_SP;
		p->from.offset = StackGuard;
		p->to.type = D_AX;
		
		p = appendp(ctxt, p);
		p->as = ASUBQ;
		p->from.type = D_SI;
		p->to.type = D_AX;
		
		p = appendp(ctxt, p);
		p->as = ACMPQ;
		p->from.type = D_AX;
		p->to.type = D_CONST;
		p->to.offset = framesize+(StackGuard-StackSmall);
	}					

	// common
	p = appendp(ctxt, p);
	p->as = AJHI;
	p->to.type = D_BRANCH;
	q = p;

	// If we ask for more stack, we'll get a minimum of StackMin bytes.
	// We need a stack frame large enough to hold the top-of-stack data,
	// the function arguments+results, our caller's PC, our frame,
	// a word for the return PC of the next call, and then the StackLimit bytes
	// that must be available on entry to any function called from a function
	// that did a stack check.  If StackMin is enough, don't ask for a specific
	// amount: then we can use the custom functions and save a few
	// instructions.
	moreconst1 = 0;
	if(StackTop + textarg + ctxt->arch->ptrsize + framesize + ctxt->arch->ptrsize + StackLimit >= StackMin)
		moreconst1 = framesize;
	moreconst2 = textarg;
	if(moreconst2 == 1) // special marker
		moreconst2 = 0;
	if((moreconst2&7) != 0)
		ctxt->diag("misaligned argument size in stack split");
	// 4 varieties varieties (const1==0 cross const2==0)
	// and 6 subvarieties of (const1==0 and const2!=0)
	p = appendp(ctxt, p);
	if(moreconst1 == 0 && moreconst2 == 0) {
		p->as = ACALL;
		p->to.type = D_BRANCH;
		p->to.sym = ctxt->symmorestack[0];
	} else
	if(moreconst1 != 0 && moreconst2 == 0) {
		p->as = AMOVL;
		p->from.type = D_CONST;
		p->from.offset = moreconst1;
		p->to.type = D_AX;

		p = appendp(ctxt, p);
		p->as = ACALL;
		p->to.type = D_BRANCH;
		p->to.sym = ctxt->symmorestack[1];
	} else
	if(moreconst1 == 0 && moreconst2 <= 48 && moreconst2%8 == 0) {
		i = moreconst2/8 + 3;
		p->as = ACALL;
		p->to.type = D_BRANCH;
		p->to.sym = ctxt->symmorestack[i];
	} else
	if(moreconst1 == 0 && moreconst2 != 0) {
		p->as = AMOVL;
		p->from.type = D_CONST;
		p->from.offset = moreconst2;
		p->to.type = D_AX;

		p = appendp(ctxt, p);
		p->as = ACALL;
		p->to.type = D_BRANCH;
		p->to.sym = ctxt->symmorestack[2];
	} else {
		p->as = AMOVQ;
		p->from.type = D_CONST;
		p->from.offset = (uint64)moreconst2 << 32;
		p->from.offset |= moreconst1;
		p->to.type = D_AX;

		p = appendp(ctxt, p);
		p->as = ACALL;
		p->to.type = D_BRANCH;
		p->to.sym = ctxt->symmorestack[3];
	}
	
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
	case AIRETQ:
	case AIRETW:
	case ARETFL:
	case ARETFQ:
	case ARETFW:
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
	case APUSHQ:
	case APUSHFQ:
	case APUSHW:
	case APUSHFW:
	case APOPL:
	case APOPFL:
	case APOPQ:
	case APOPFQ:
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
	sysfatal("unknown relation: %s", anames6[a]);
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

static Prog*
prg(void)
{
	Prog *p;

	p = emallocz(sizeof(*p));
	*p = zprg;
	return p;
}

LinkArch linkamd64 = {
	.name = "amd64",

	.zprog = zprog,
	.zhist = zhist,
	.zfile = zfile,
	.zname = zname,
	.isdata = isdata,
	.ldobj = ldobj6,
	.nopout = nopout6,
	.symtype = symtype,
	.iscall = iscall,
	.datasize = datasize,
	.textflag = textflag,
	.settextflag = settextflag,
	.progedit = progedit,
	.prg = prg,
	.addstacksplit = addstacksplit,
	.assemble = span6,
	.follow = follow,

	.minlc = 1,
	.ptrsize = 8,

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
