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
#include "../runtime/stack.h"

static void
nopout(Prog *p)
{
	p->as = ANOP;
	p->from.type = TYPE_NONE;
	p->from.reg = 0;
	p->from.name = 0;
	p->to.type = TYPE_NONE;
	p->to.reg = 0;
	p->to.name = 0;
}

static void nacladdr(Link*, Prog*, Addr*);

static int
canuselocaltls(Link *ctxt)
{
	switch(ctxt->headtype) {
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

	// Thread-local storage references use the TLS pseudo-register.
	// As a register, TLS refers to the thread-local storage base, and it
	// can only be loaded into another register:
	//
	//         MOVQ TLS, AX
	//
	// An offset from the thread-local storage base is written off(reg)(TLS*1).
	// Semantically it is off(reg), but the (TLS*1) annotation marks this as
	// indexing from the loaded TLS base. This emits a relocation so that
	// if the linker needs to adjust the offset, it can. For example:
	//
	//         MOVQ TLS, AX
	//         MOVQ 8(AX)(TLS*1), CX // load m into CX
	// 
	// On systems that support direct access to the TLS memory, this
	// pair of instructions can be reduced to a direct TLS memory reference:
	// 
	//         MOVQ 8(TLS), CX // load m into CX
	//
	// The 2-instruction and 1-instruction forms correspond roughly to
	// ELF TLS initial exec mode and ELF TLS local exec mode, respectively.
	// 
	// We applies this rewrite on systems that support the 1-instruction form.
	// The decision is made using only the operating system (and probably
	// the -shared flag, eventually), not the link mode. If some link modes
	// on a particular operating system require the 2-instruction form,
	// then all builds for that operating system will use the 2-instruction
	// form, so that the link mode decision can be delayed to link time.
	//
	// In this way, all supported systems use identical instructions to
	// access TLS, and they are rewritten appropriately first here in
	// liblink and then finally using relocations in the linker.

	if(canuselocaltls(ctxt)) {
		// Reduce TLS initial exec model to TLS local exec model.
		// Sequences like
		//	MOVQ TLS, BX
		//	... off(BX)(TLS*1) ...
		// become
		//	NOP
		//	... off(TLS) ...
		//
		// TODO(rsc): Remove the Hsolaris special case. It exists only to
		// guarantee we are producing byte-identical binaries as before this code.
		// But it should be unnecessary.
		if((p->as == AMOVQ || p->as == AMOVL) && p->from.type == TYPE_REG && p->from.reg == REG_TLS && p->to.type == TYPE_REG && REG_AX <= p->to.reg && p->to.reg <= REG_R15 && ctxt->headtype != Hsolaris)
			nopout(p);
		if(p->from.type == TYPE_MEM && p->from.index == REG_TLS && REG_AX <= p->from.reg && p->from.reg <= REG_R15) {
			p->from.reg = REG_TLS;
			p->from.scale = 0;
			p->from.index = REG_NONE;
		}
		if(p->to.type == TYPE_MEM && p->to.index == REG_TLS && REG_AX <= p->to.reg && p->to.reg <= REG_R15) {
			p->to.reg = REG_TLS;
			p->to.scale = 0;
			p->to.index = REG_NONE;
		}
	} else {
		// As a courtesy to the C compilers, rewrite TLS local exec load as TLS initial exec load.
		// The instruction
		//	MOVQ off(TLS), BX
		// becomes the sequence
		//	MOVQ TLS, BX
		//	MOVQ off(BX)(TLS*1), BX
		// This allows the C compilers to emit references to m and g using the direct off(TLS) form.
		if((p->as == AMOVQ || p->as == AMOVL) && p->from.type == TYPE_MEM && p->from.reg == REG_TLS && p->to.type == TYPE_REG && REG_AX <= p->to.reg && p->to.reg <= REG_R15) {
			q = appendp(ctxt, p);
			q->as = p->as;
			q->from = p->from;
			q->from.type = TYPE_MEM;
			q->from.reg = p->to.reg;
			q->from.index = REG_TLS;
			q->from.scale = 2; // TODO: use 1
			q->to = p->to;
			p->from.type = TYPE_REG;
			p->from.reg = REG_TLS;
			p->from.index = REG_NONE;
			p->from.offset = 0;
		}
	}

	// TODO: Remove.
	if(ctxt->headtype == Hwindows || ctxt->headtype == Hplan9) {
		if(p->from.scale == 1 && p->from.index == REG_TLS)
			p->from.scale = 2;
		if(p->to.scale == 1 && p->to.index == REG_TLS)
			p->to.scale = 2;
	}

	if(ctxt->headtype == Hnacl) {
		nacladdr(ctxt, p, &p->from);
		nacladdr(ctxt, p, &p->to);
	}

	// Maintain information about code generation mode.
	if(ctxt->mode == 0)
		ctxt->mode = 64;
	p->mode = ctxt->mode;
	
	switch(p->as) {
	case AMODE:
		if(p->from.type == TYPE_CONST || (p->from.type == TYPE_MEM && p->from.reg == REG_NONE)) {
			switch((int)p->from.offset) {
			case 16:
			case 32:
			case 64:
				ctxt->mode = p->from.offset;
				break;
			}
		}
		nopout(p);
		break;
	}
	
	// Rewrite CALL/JMP/RET to symbol as TYPE_BRANCH.
	switch(p->as) {
	case ACALL:
	case AJMP:
	case ARET:
		if(p->to.type == TYPE_MEM && (p->to.name == NAME_EXTERN || p->to.name == NAME_STATIC) && p->to.sym != nil)
			p->to.type = TYPE_BRANCH;
		break;
	}

	// Rewrite float constants to values stored in memory.
	switch(p->as) {
	case AMOVSS:
		// Convert AMOVSS $(0), Xx to AXORPS Xx, Xx
		if(p->from.type == TYPE_FCONST)
		if(p->from.u.dval == 0)
		if(p->to.type == TYPE_REG && REG_X0 <= p->to.reg && p->to.reg <= REG_X15) {
			p->as = AXORPS;
			p->from = p->to;
			break;
		}
		// fallthrough

	case AFMOVF:
	case AFADDF:
	case AFSUBF:
	case AFSUBRF:
	case AFMULF:
	case AFDIVF:
	case AFDIVRF:
	case AFCOMF:
	case AFCOMFP:
	case AADDSS:
	case ASUBSS:
	case AMULSS:
	case ADIVSS:
	case ACOMISS:
	case AUCOMISS:
		if(p->from.type == TYPE_FCONST) {
			uint32 i32;
			float32 f32;
			f32 = p->from.u.dval;
			memmove(&i32, &f32, 4);
			sprint(literal, "$f32.%08ux", i32);
			s = linklookup(ctxt, literal, 0);
			if(s->type == 0) {
				s->type = SRODATA;
				adduint32(ctxt, s, i32);
				s->reachable = 0;
			}
			p->from.type = TYPE_MEM;
			p->from.name = NAME_EXTERN;
			p->from.sym = s;
			p->from.offset = 0;
		}
		break;

	case AMOVSD:
		// Convert AMOVSD $(0), Xx to AXORPS Xx, Xx
		if(p->from.type == TYPE_FCONST)
		if(p->from.u.dval == 0)
		if(p->to.type == TYPE_REG && REG_X0 <= p->to.reg && p->to.reg <= REG_X15) {
			p->as = AXORPS;
			p->from = p->to;
			break;
		}
		// fallthrough
	
	case AFMOVD:
	case AFADDD:
	case AFSUBD:
	case AFSUBRD:
	case AFMULD:
	case AFDIVD:
	case AFDIVRD:
	case AFCOMD:
	case AFCOMDP:
	case AADDSD:
	case ASUBSD:
	case AMULSD:
	case ADIVSD:
	case ACOMISD:
	case AUCOMISD:
		if(p->from.type == TYPE_FCONST) {
			uint64 i64;
			memmove(&i64, &p->from.u.dval, 8);
			sprint(literal, "$f64.%016llux", i64);
			s = linklookup(ctxt, literal, 0);
			if(s->type == 0) {
				s->type = SRODATA;
				adduint64(ctxt, s, i64);
				s->reachable = 0;
			}
			p->from.type = TYPE_MEM;
			p->from.name = NAME_EXTERN;
			p->from.sym = s;
			p->from.offset = 0;
		}
		break;
	}
}

static void
nacladdr(Link *ctxt, Prog *p, Addr *a)
{
	if(p->as == ALEAL || p->as == ALEAQ)
		return;
	
	if(a->reg == REG_BP) {
		ctxt->diag("invalid address: %P", p);
		return;
	}
	if(a->reg == REG_TLS)
		a->reg = REG_BP;
	if(a->type == TYPE_MEM && a->name == NAME_NONE) {
		switch(a->reg) {
		case REG_BP:
		case REG_SP:
		case REG_R15:
			// all ok
			break;
		default:
			if(a->index != REG_NONE)
				ctxt->diag("invalid address %P", p);
			a->index = a->reg;
			if(a->index != REG_NONE)
				a->scale = 1;
			a->reg = REG_R15;
			break;
		}
	}
}

static Prog*	load_g_cx(Link*, Prog*);
static Prog*	stacksplit(Link*, Prog*, int32, int32, int, Prog**);
static void	indir_cx(Link*, Addr*);

static void
preprocess(Link *ctxt, LSym *cursym)
{
	Prog *p, *q, *p1, *p2;
	int32 autoffset, deltasp;
	int a, pcsize, bpsize;
	vlong textarg;

	if(ctxt->tlsg == nil)
		ctxt->tlsg = linklookup(ctxt, "runtime.tlsg", 0);
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
	
	if(framepointer_enabled && autoffset > 0) {
		// Make room for to save a base pointer.  If autoffset == 0,
		// this might do something special like a tail jump to
		// another function, so in that case we omit this.
		bpsize = ctxt->arch->ptrsize;
		autoffset += bpsize;
		p->to.offset += bpsize;
	} else {
		bpsize = 0;
	}

	textarg = p->to.u.argsize;
	cursym->args = textarg;
	cursym->locals = p->to.offset;

	if(autoffset < StackSmall && !(p->from3.offset & NOSPLIT)) {
		for(q = p; q != nil; q = q->link) {
			if(q->as == ACALL)
				goto noleaf;
			if((q->as == ADUFFCOPY || q->as == ADUFFZERO) && autoffset >= StackSmall - 8)
				goto noleaf;
		}
		p->from3.offset |= NOSPLIT;
	noleaf:;
	}

	q = nil;
	if(!(p->from3.offset & NOSPLIT) || (p->from3.offset & WRAPPER)) {
		p = appendp(ctxt, p);
		p = load_g_cx(ctxt, p); // load g into CX
	}
	if(!(cursym->text->from3.offset & NOSPLIT))
		p = stacksplit(ctxt, p, autoffset, textarg, !(cursym->text->from3.offset&NEEDCTXT), &q); // emit split check

	if(autoffset) {
		if(autoffset%ctxt->arch->regsize != 0)
			ctxt->diag("unaligned stack size %d", autoffset);
		p = appendp(ctxt, p);
		p->as = AADJSP;
		p->from.type = TYPE_CONST;
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

	if(bpsize > 0) {
		// Save caller's BP
		p = appendp(ctxt, p);
		p->as = AMOVQ;
		p->from.type = TYPE_REG;
		p->from.reg = REG_BP;
		p->to.type = TYPE_MEM;
		p->to.reg = REG_SP;
		p->to.scale = 1;
		p->to.offset = autoffset - bpsize;

		// Move current frame to BP
		p = appendp(ctxt, p);
		p->as = ALEAQ;
		p->from.type = TYPE_MEM;
		p->from.reg = REG_SP;
		p->from.scale = 1;
		p->from.offset = autoffset - bpsize;
		p->to.type = TYPE_REG;
		p->to.reg = REG_BP;
	}
	
	if(cursym->text->from3.offset & WRAPPER) {
		// if(g->panic != nil && g->panic->argp == FP) g->panic->argp = bottom-of-frame
		//
		//	MOVQ g_panic(CX), BX
		//	TESTQ BX, BX
		//	JEQ end
		//	LEAQ (autoffset+8)(SP), DI
		//	CMPQ panic_argp(BX), DI
		//	JNE end
		//	MOVQ SP, panic_argp(BX)
		// end:
		//	NOP
		//
		// The NOP is needed to give the jumps somewhere to land.
		// It is a liblink NOP, not an x86 NOP: it encodes to 0 instruction bytes.

		p = appendp(ctxt, p);
		p->as = AMOVQ;
		p->from.type = TYPE_MEM;
		p->from.reg = REG_CX;
		p->from.offset = 4*ctxt->arch->ptrsize; // G.panic
		p->to.type = TYPE_REG;
		p->to.reg = REG_BX;
		if(ctxt->headtype == Hnacl) {
			p->as = AMOVL;
			p->from.type = TYPE_MEM;
			p->from.reg = REG_R15;
			p->from.scale = 1;
			p->from.index = REG_CX;
		}

		p = appendp(ctxt, p);
		p->as = ATESTQ;
		p->from.type = TYPE_REG;
		p->from.reg = REG_BX;
		p->to.type = TYPE_REG;
		p->to.reg = REG_BX;
		if(ctxt->headtype == Hnacl)
			p->as = ATESTL;

		p = appendp(ctxt, p);
		p->as = AJEQ;
		p->to.type = TYPE_BRANCH;
		p1 = p;

		p = appendp(ctxt, p);
		p->as = ALEAQ;
		p->from.type = TYPE_MEM;
		p->from.reg = REG_SP;
		p->from.offset = autoffset+8;
		p->to.type = TYPE_REG;
		p->to.reg = REG_DI;
		if(ctxt->headtype == Hnacl)
			p->as = ALEAL;

		p = appendp(ctxt, p);
		p->as = ACMPQ;
		p->from.type = TYPE_MEM;
		p->from.reg = REG_BX;
		p->from.offset = 0; // Panic.argp
		p->to.type = TYPE_REG;
		p->to.reg = REG_DI;
		if(ctxt->headtype == Hnacl) {
			p->as = ACMPL;
			p->from.type = TYPE_MEM;
			p->from.reg = REG_R15;
			p->from.scale = 1;
			p->from.index = REG_BX;
		}

		p = appendp(ctxt, p);
		p->as = AJNE;
		p->to.type = TYPE_BRANCH;
		p2 = p;

		p = appendp(ctxt, p);
		p->as = AMOVQ;
		p->from.type = TYPE_REG;
		p->from.reg = REG_SP;
		p->to.type = TYPE_MEM;
		p->to.reg = REG_BX;
		p->to.offset = 0; // Panic.argp
		if(ctxt->headtype == Hnacl) {
			p->as = AMOVL;
			p->to.type = TYPE_MEM;
			p->to.reg = REG_R15;
			p->to.scale = 1;
			p->to.index = REG_BX;
		}

		p = appendp(ctxt, p);
		p->as = ANOP;
		p1->pcond = p;
		p2->pcond = p;
	}

	if(ctxt->debugzerostack && autoffset && !(cursym->text->from3.offset&NOSPLIT)) {
		// 6l -Z means zero the stack frame on entry.
		// This slows down function calls but can help avoid
		// false positives in garbage collection.
		p = appendp(ctxt, p);
		p->as = AMOVQ;
		p->from.type = TYPE_REG;
		p->from.reg = REG_SP;
		p->to.type = TYPE_REG;
		p->to.reg = REG_DI;
		
		p = appendp(ctxt, p);
		p->as = AMOVQ;
		p->from.type = TYPE_CONST;
		p->from.offset = autoffset/8;
		p->to.type = TYPE_REG;
		p->to.reg = REG_CX;
		
		p = appendp(ctxt, p);
		p->as = AMOVQ;
		p->from.type = TYPE_CONST;
		p->from.offset = 0;
		p->to.type = TYPE_REG;
		p->to.reg = REG_AX;
		
		p = appendp(ctxt, p);
		p->as = AREP;
		
		p = appendp(ctxt, p);
		p->as = ASTOSQ;
	}
	
	for(; p != nil; p = p->link) {
		pcsize = p->mode/8;
		a = p->from.name;
		if(a == NAME_AUTO)
			p->from.offset += deltasp - bpsize;
		if(a == NAME_PARAM)
			p->from.offset += deltasp + pcsize;
		a = p->to.name;
		if(a == NAME_AUTO)
			p->to.offset += deltasp - bpsize;
		if(a == NAME_PARAM)
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

		if(autoffset) {
			if(bpsize > 0) {
				// Restore caller's BP
				p->as = AMOVQ;
				p->from.type = TYPE_MEM;
				p->from.reg = REG_SP;
				p->from.scale = 1;
				p->from.offset = autoffset - bpsize;
				p->to.type = TYPE_REG;
				p->to.reg = REG_BP;
				p = appendp(ctxt, p);
			}

			p->as = AADJSP;
			p->from.type = TYPE_CONST;
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

static void
indir_cx(Link *ctxt, Addr *a)
{
	if(ctxt->headtype == Hnacl) {
		a->type = TYPE_MEM;
		a->reg = REG_R15;
		a->index = REG_CX;
		a->scale = 1;
		return;
	}

	a->type = TYPE_MEM;
	a->reg = REG_CX;
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

	p->as = AMOVQ;
	if(ctxt->arch->ptrsize == 4)
		p->as = AMOVL;
	p->from.type = TYPE_MEM;
	p->from.reg = REG_TLS;
	p->from.offset = 0;
	p->to.type = TYPE_REG;
	p->to.reg = REG_CX;
	
	next = p->link;
	progedit(ctxt, p);
	while(p->link != next)
		p = p->link;
	
	if(p->from.index == REG_TLS)
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
stacksplit(Link *ctxt, Prog *p, int32 framesize, int32 textarg, int noctxt, Prog **jmpok)
{
	Prog *q, *q1;
	int cmp, lea, mov, sub;

	USED(textarg);
	cmp = ACMPQ;
	lea = ALEAQ;
	mov = AMOVQ;
	sub = ASUBQ;

	if(ctxt->headtype == Hnacl) {
		cmp = ACMPL;
		lea = ALEAL;
		mov = AMOVL;
		sub = ASUBL;
	}

	q1 = nil;
	if(framesize <= StackSmall) {
		// small stack: SP <= stackguard
		//	CMPQ SP, stackguard
		p = appendp(ctxt, p);
		p->as = cmp;
		p->from.type = TYPE_REG;
		p->from.reg = REG_SP;
		indir_cx(ctxt, &p->to);
		p->to.offset = 2*ctxt->arch->ptrsize;	// G.stackguard0
		if(ctxt->cursym->cfunc)
			p->to.offset = 3*ctxt->arch->ptrsize;	// G.stackguard1
	} else if(framesize <= StackBig) {
		// large stack: SP-framesize <= stackguard-StackSmall
		//	LEAQ -xxx(SP), AX
		//	CMPQ AX, stackguard
		p = appendp(ctxt, p);
		p->as = lea;
		p->from.type = TYPE_MEM;
		p->from.reg = REG_SP;
		p->from.offset = -(framesize-StackSmall);
		p->to.type = TYPE_REG;
		p->to.reg = REG_AX;

		p = appendp(ctxt, p);
		p->as = cmp;
		p->from.type = TYPE_REG;
		p->from.reg = REG_AX;
		indir_cx(ctxt, &p->to);
		p->to.offset = 2*ctxt->arch->ptrsize;	// G.stackguard0
		if(ctxt->cursym->cfunc)
			p->to.offset = 3*ctxt->arch->ptrsize;	// G.stackguard1
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
		p->as = mov;
		indir_cx(ctxt, &p->from);
		p->from.offset = 2*ctxt->arch->ptrsize;	// G.stackguard0
		if(ctxt->cursym->cfunc)
			p->from.offset = 3*ctxt->arch->ptrsize;	// G.stackguard1
		p->to.type = TYPE_REG;
		p->to.reg = REG_SI;

		p = appendp(ctxt, p);
		p->as = cmp;
		p->from.type = TYPE_REG;
		p->from.reg = REG_SI;
		p->to.type = TYPE_CONST;
		p->to.offset = StackPreempt;

		p = appendp(ctxt, p);
		p->as = AJEQ;
		p->to.type = TYPE_BRANCH;
		q1 = p;

		p = appendp(ctxt, p);
		p->as = lea;
		p->from.type = TYPE_MEM;
		p->from.reg = REG_SP;
		p->from.offset = StackGuard;
		p->to.type = TYPE_REG;
		p->to.reg = REG_AX;
		
		p = appendp(ctxt, p);
		p->as = sub;
		p->from.type = TYPE_REG;
		p->from.reg = REG_SI;
		p->to.type = TYPE_REG;
		p->to.reg = REG_AX;
		
		p = appendp(ctxt, p);
		p->as = cmp;
		p->from.type = TYPE_REG;
		p->from.reg = REG_AX;
		p->to.type = TYPE_CONST;
		p->to.offset = framesize+(StackGuard-StackSmall);
	}					

	// common
	p = appendp(ctxt, p);
	p->as = AJHI;
	p->to.type = TYPE_BRANCH;
	q = p;

	p = appendp(ctxt, p);
	p->as = ACALL;
	p->to.type = TYPE_BRANCH;
	if(ctxt->cursym->cfunc)
		p->to.sym = linklookup(ctxt, "runtime.morestackc", 0);
	else
		p->to.sym = ctxt->symmorestack[noctxt];
	
	p = appendp(ctxt, p);
	p->as = AJMP;
	p->to.type = TYPE_BRANCH;
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

	firstp = emallocz(sizeof(Prog));
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
		q = emallocz(sizeof(Prog));
		q->as = AJMP;
		q->lineno = p->lineno;
		q->to.type = TYPE_BRANCH;
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
		if(p->from.type == TYPE_CONST) {
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

LinkArch linkamd64 = {
	.name = "amd64",
	.thechar = '6',
	.endian = LittleEndian,

	.preprocess = preprocess,
	.assemble = span6,
	.follow = follow,
	.progedit = progedit,

	.minlc = 1,
	.ptrsize = 8,
	.regsize = 8,
};

LinkArch linkamd64p32 = {
	.name = "amd64p32",
	.thechar = '6',
	.endian = LittleEndian,

	.preprocess = preprocess,
	.assemble = span6,
	.follow = follow,
	.progedit = progedit,

	.minlc = 1,
	.ptrsize = 4,
	.regsize = 8,
};
