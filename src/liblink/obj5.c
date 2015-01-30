// Derived from Inferno utils/5c/swt.c
// http://code.google.com/p/inferno-os/source/browse/utils/5c/swt.c
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
#include "../cmd/5l/5.out.h"
#include "../runtime/stack.h"

static void
progedit(Link *ctxt, Prog *p)
{
	char literal[64];
	LSym *s;
	static LSym *tlsfallback;

	p->from.class = 0;
	p->to.class = 0;

	// Rewrite B/BL to symbol as TYPE_BRANCH.
	switch(p->as) {
	case AB:
	case ABL:
	case ADUFFZERO:
	case ADUFFCOPY:
		if(p->to.type == TYPE_MEM && (p->to.name == NAME_EXTERN || p->to.name == NAME_STATIC) && p->to.sym != nil)
			p->to.type = TYPE_BRANCH;
		break;
	}

	// Replace TLS register fetches on older ARM procesors.
	switch(p->as) {
	case AMRC:
		// Treat MRC 15, 0, <reg>, C13, C0, 3 specially.
		if((p->to.offset & 0xffff0fff) == 0xee1d0f70) {
			// Because the instruction might be rewriten to a BL which returns in R0
			// the register must be zero.
		       	if ((p->to.offset & 0xf000) != 0)
				ctxt->diag("%L: TLS MRC instruction must write to R0 as it might get translated into a BL instruction", p->lineno);

			if(ctxt->goarm < 7) {
				// Replace it with BL runtime.read_tls_fallback(SB) for ARM CPUs that lack the tls extension.
				if(tlsfallback == nil)
					tlsfallback = linklookup(ctxt, "runtime.read_tls_fallback", 0);
				// MOVW	LR, R11
				p->as = AMOVW;
				p->from.type = TYPE_REG;
				p->from.reg = REGLINK;
				p->to.type = TYPE_REG;
				p->to.reg = REGTMP;

				// BL	runtime.read_tls_fallback(SB)
				p = appendp(ctxt, p);
				p->as = ABL;
				p->to.type = TYPE_BRANCH;
				p->to.sym = tlsfallback;
				p->to.offset = 0;

				// MOVW	R11, LR
				p = appendp(ctxt, p);
				p->as = AMOVW;
				p->from.type = TYPE_REG;
				p->from.reg = REGTMP;
				p->to.type = TYPE_REG;
				p->to.reg = REGLINK;
				break;
			}
		}
		// Otherwise, MRC/MCR instructions need no further treatment.
		p->as = AWORD;
		break;
	}

	// Rewrite float constants to values stored in memory.
	switch(p->as) {
	case AMOVF:
		if(p->from.type == TYPE_FCONST && chipfloat5(ctxt, p->from.u.dval) < 0 &&
		   (chipzero5(ctxt, p->from.u.dval) < 0 || (p->scond & C_SCOND) != C_SCOND_NONE)) {
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
			p->from.sym = s;
			p->from.name = NAME_EXTERN;
			p->from.offset = 0;
		}
		break;

	case AMOVD:
		if(p->from.type == TYPE_FCONST && chipfloat5(ctxt, p->from.u.dval) < 0 &&
		   (chipzero5(ctxt, p->from.u.dval) < 0 || (p->scond & C_SCOND) != C_SCOND_NONE)) {
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
			p->from.sym = s;
			p->from.name = NAME_EXTERN;
			p->from.offset = 0;
		}
		break;
	}

	if(ctxt->flag_shared) {
		// Shared libraries use R_ARM_TLS_IE32 instead of 
		// R_ARM_TLS_LE32, replacing the link time constant TLS offset in
		// runtime.tlsg with an address to a GOT entry containing the 
		// offset. Rewrite $runtime.tlsg(SB) to runtime.tlsg(SB) to
		// compensate.
		if(ctxt->tlsg == nil)
			ctxt->tlsg = linklookup(ctxt, "runtime.tlsg", 0);

		if(p->from.type == TYPE_ADDR && p->from.name == NAME_EXTERN && p->from.sym == ctxt->tlsg)
			p->from.type = TYPE_MEM;
		if(p->to.type == TYPE_ADDR && p->to.name == NAME_EXTERN && p->to.sym == ctxt->tlsg)
			p->to.type = TYPE_MEM;
	}
}

static	Prog*	stacksplit(Link*, Prog*, int32, int);
static	void		initdiv(Link*);
static	void	softfloat(Link*, LSym*);

// Prog.mark
enum
{
	FOLL = 1<<0,
	LABEL = 1<<1,
	LEAF = 1<<2,
};

static void
linkcase(Prog *casep)
{
	Prog *p;

	for(p = casep; p != nil; p = p->link){
		if(p->as == ABCASE) {
			for(; p != nil && p->as == ABCASE; p = p->link)
				p->pcrel = casep;
			break;
		}
	}
}

static void
nocache5(Prog *p)
{
	p->optab = 0;
	p->from.class = 0;
	p->to.class = 0;
}

static void
preprocess(Link *ctxt, LSym *cursym)
{
	Prog *p, *pl, *p1, *p2, *q, *q1, *q2;
	int o;
	int32 autosize, autoffset;
	
	autosize = 0;

	if(ctxt->symmorestack[0] == nil) {
		ctxt->symmorestack[0] = linklookup(ctxt, "runtime.morestack", 0);
		ctxt->symmorestack[1] = linklookup(ctxt, "runtime.morestack_noctxt", 0);
	}

	q = nil;
	
	ctxt->cursym = cursym;

	if(cursym->text == nil || cursym->text->link == nil)
		return;				

	softfloat(ctxt, cursym);

	p = cursym->text;
	autoffset = p->to.offset;
	if(autoffset < 0)
		autoffset = 0;
	cursym->locals = autoffset;
	cursym->args = p->to.u.argsize;

	if(ctxt->debugzerostack) {
		if(autoffset && !(p->from3.offset&NOSPLIT)) {
			// MOVW $4(R13), R1
			p = appendp(ctxt, p);
			p->as = AMOVW;
			p->from.type = TYPE_ADDR;
			p->from.reg = REG_R13;
			p->from.offset = 4;
			p->to.type = TYPE_REG;
			p->to.reg = REG_R1;
	
			// MOVW $n(R13), R2
			p = appendp(ctxt, p);
			p->as = AMOVW;
			p->from.type = TYPE_ADDR;
			p->from.reg = REG_R13;
			p->from.offset = 4 + autoffset;
			p->to.type = TYPE_REG;
			p->to.reg = REG_R2;
	
			// MOVW $0, R3
			p = appendp(ctxt, p);
			p->as = AMOVW;
			p->from.type = TYPE_CONST;
			p->from.offset = 0;
			p->to.type = TYPE_REG;
			p->to.reg = REG_R3;
	
			// L:
			//	MOVW.nil R3, 0(R1) +4
			//	CMP R1, R2
			//	BNE L
			p = pl = appendp(ctxt, p);
			p->as = AMOVW;
			p->from.type = TYPE_REG;
			p->from.reg = REG_R3;
			p->to.type = TYPE_MEM;
			p->to.reg = REG_R1;
			p->to.offset = 4;
			p->scond |= C_PBIT;
	
			p = appendp(ctxt, p);
			p->as = ACMP;
			p->from.type = TYPE_REG;
			p->from.reg = REG_R1;
			p->reg = REG_R2;
	
			p = appendp(ctxt, p);
			p->as = ABNE;
			p->to.type = TYPE_BRANCH;
			p->pcond = pl;
		}
	}

	/*
	 * find leaf subroutines
	 * strip NOPs
	 * expand RET
	 * expand BECOME pseudo
	 */

	for(p = cursym->text; p != nil; p = p->link) {
		switch(p->as) {
		case ACASE:
			if(ctxt->flag_shared)
				linkcase(p);
			break;

		case ATEXT:
			p->mark |= LEAF;
			break;

		case ARET:
			break;

		case ADIV:
		case ADIVU:
		case AMOD:
		case AMODU:
			q = p;
			if(ctxt->sym_div == nil)
				initdiv(ctxt);
			cursym->text->mark &= ~LEAF;
			continue;

		case ANOP:
			q1 = p->link;
			q->link = q1;		/* q is non-nop */
			if(q1 != nil)
				q1->mark |= p->mark;
			continue;

		case ABL:
		case ABX:
		case ADUFFZERO:
		case ADUFFCOPY:
			cursym->text->mark &= ~LEAF;

		case ABCASE:
		case AB:

		case ABEQ:
		case ABNE:
		case ABCS:
		case ABHS:
		case ABCC:
		case ABLO:
		case ABMI:
		case ABPL:
		case ABVS:
		case ABVC:
		case ABHI:
		case ABLS:
		case ABGE:
		case ABLT:
		case ABGT:
		case ABLE:
			q1 = p->pcond;
			if(q1 != nil) {
				while(q1->as == ANOP) {
					q1 = q1->link;
					p->pcond = q1;
				}
			}
			break;
		}
		q = p;
	}

	for(p = cursym->text; p != nil; p = p->link) {
		o = p->as;
		switch(o) {
		case ATEXT:
			autosize = p->to.offset + 4;
			if(autosize <= 4)
			if(cursym->text->mark & LEAF) {
				p->to.offset = -4;
				autosize = 0;
			}

			if(!autosize && !(cursym->text->mark & LEAF)) {
				if(ctxt->debugvlog) {
					Bprint(ctxt->bso, "save suppressed in: %s\n",
						cursym->name);
					Bflush(ctxt->bso);
				}
				cursym->text->mark |= LEAF;
			}
			if(cursym->text->mark & LEAF) {
				cursym->leaf = 1;
				if(!autosize)
					break;
			}

			if(!(p->from3.offset & NOSPLIT))
				p = stacksplit(ctxt, p, autosize, !(cursym->text->from3.offset&NEEDCTXT)); // emit split check
			
			// MOVW.W		R14,$-autosize(SP)
			p = appendp(ctxt, p);
			p->as = AMOVW;
			p->scond |= C_WBIT;
			p->from.type = TYPE_REG;
			p->from.reg = REGLINK;
			p->to.type = TYPE_MEM;
			p->to.offset = -autosize;
			p->to.reg = REGSP;
			p->spadj = autosize;
			
			if(cursym->text->from3.offset & WRAPPER) {
				// if(g->panic != nil && g->panic->argp == FP) g->panic->argp = bottom-of-frame
				//
				//	MOVW g_panic(g), R1
				//	CMP $0, R1
				//	B.EQ end
				//	MOVW panic_argp(R1), R2
				//	ADD $(autosize+4), R13, R3
				//	CMP R2, R3
				//	B.NE end
				//	ADD $4, R13, R4
				//	MOVW R4, panic_argp(R1)
				// end:
				//	NOP
				//
				// The NOP is needed to give the jumps somewhere to land.
				// It is a liblink NOP, not an ARM NOP: it encodes to 0 instruction bytes.

				p = appendp(ctxt, p);
				p->as = AMOVW;
				p->from.type = TYPE_MEM;
				p->from.reg = REGG;
				p->from.offset = 4*ctxt->arch->ptrsize; // G.panic
				p->to.type = TYPE_REG;
				p->to.reg = REG_R1;
			
				p = appendp(ctxt, p);
				p->as = ACMP;
				p->from.type = TYPE_CONST;
				p->from.offset = 0;
				p->reg = REG_R1;
			
				p = appendp(ctxt, p);
				p->as = ABEQ;
				p->to.type = TYPE_BRANCH;
				p1 = p;
				
				p = appendp(ctxt, p);
				p->as = AMOVW;
				p->from.type = TYPE_MEM;
				p->from.reg = REG_R1;
				p->from.offset = 0; // Panic.argp
				p->to.type = TYPE_REG;
				p->to.reg = REG_R2;
			
				p = appendp(ctxt, p);
				p->as = AADD;
				p->from.type = TYPE_CONST;
				p->from.offset = autosize+4;
				p->reg = REG_R13;
				p->to.type = TYPE_REG;
				p->to.reg = REG_R3;

				p = appendp(ctxt, p);
				p->as = ACMP;
				p->from.type = TYPE_REG;
				p->from.reg = REG_R2;
				p->reg = REG_R3;

				p = appendp(ctxt, p);
				p->as = ABNE;
				p->to.type = TYPE_BRANCH;
				p2 = p;
			
				p = appendp(ctxt, p);
				p->as = AADD;
				p->from.type = TYPE_CONST;
				p->from.offset = 4;
				p->reg = REG_R13;
				p->to.type = TYPE_REG;
				p->to.reg = REG_R4;

				p = appendp(ctxt, p);
				p->as = AMOVW;
				p->from.type = TYPE_REG;
				p->from.reg = REG_R4;
				p->to.type = TYPE_MEM;
				p->to.reg = REG_R1;
				p->to.offset = 0; // Panic.argp

				p = appendp(ctxt, p);
				p->as = ANOP;
				p1->pcond = p;
				p2->pcond = p;
			}
			break;

		case ARET:
			nocache5(p);
			if(cursym->text->mark & LEAF) {
				if(!autosize) {
					p->as = AB;
					p->from = zprog.from;
					if(p->to.sym) { // retjmp
						p->to.type = TYPE_BRANCH;
					} else {
						p->to.type = TYPE_MEM;
						p->to.offset = 0;
						p->to.reg = REGLINK;
					}
					break;
				}
			}

			p->as = AMOVW;
			p->scond |= C_PBIT;
			p->from.type = TYPE_MEM;
			p->from.offset = autosize;
			p->from.reg = REGSP;
			p->to.type = TYPE_REG;
			p->to.reg = REGPC;
			// If there are instructions following
			// this ARET, they come from a branch
			// with the same stackframe, so no spadj.
			
			if(p->to.sym) { // retjmp
				p->to.reg = REGLINK;
				q2 = appendp(ctxt, p);
				q2->as = AB;
				q2->to.type = TYPE_BRANCH;
				q2->to.sym = p->to.sym;
				p->to.sym = nil;
				p = q2;
			}
			break;

		case AADD:
			if(p->from.type == TYPE_CONST && p->from.reg == 0 && p->to.type == TYPE_REG && p->to.reg == REGSP)
				p->spadj = -p->from.offset;
			break;

		case ASUB:
			if(p->from.type == TYPE_CONST && p->from.reg == 0 && p->to.type == TYPE_REG && p->to.reg == REGSP)
				p->spadj = p->from.offset;
			break;

		case ADIV:
		case ADIVU:
		case AMOD:
		case AMODU:
			if(ctxt->debugdivmod)
				break;
			if(p->from.type != TYPE_REG)
				break;
			if(p->to.type != TYPE_REG)
				break;
			q1 = p;

			/* MOV a,4(SP) */
			p = appendp(ctxt, p);
			p->as = AMOVW;
			p->lineno = q1->lineno;
			p->from.type = TYPE_REG;
			p->from.reg = q1->from.reg;
			p->to.type = TYPE_MEM;
			p->to.reg = REGSP;
			p->to.offset = 4;

			/* MOV b,REGTMP */
			p = appendp(ctxt, p);
			p->as = AMOVW;
			p->lineno = q1->lineno;
			p->from.type = TYPE_REG;
			p->from.reg = q1->reg;
			if(q1->reg == 0)
				p->from.reg = q1->to.reg;
			p->to.type = TYPE_REG;
			p->to.reg = REGTMP;
			p->to.offset = 0;

			/* CALL appropriate */
			p = appendp(ctxt, p);
			p->as = ABL;
			p->lineno = q1->lineno;
			p->to.type = TYPE_BRANCH;
			switch(o) {
			case ADIV:
				p->to.sym = ctxt->sym_div;
				break;
			case ADIVU:
				p->to.sym = ctxt->sym_divu;
				break;
			case AMOD:
				p->to.sym = ctxt->sym_mod;
				break;
			case AMODU:
				p->to.sym = ctxt->sym_modu;
				break;
			}

			/* MOV REGTMP, b */
			p = appendp(ctxt, p);
			p->as = AMOVW;
			p->lineno = q1->lineno;
			p->from.type = TYPE_REG;
			p->from.reg = REGTMP;
			p->from.offset = 0;
			p->to.type = TYPE_REG;
			p->to.reg = q1->to.reg;

			/* ADD $8,SP */
			p = appendp(ctxt, p);
			p->as = AADD;
			p->lineno = q1->lineno;
			p->from.type = TYPE_CONST;
			p->from.reg = 0;
			p->from.offset = 8;
			p->reg = 0;
			p->to.type = TYPE_REG;
			p->to.reg = REGSP;
			p->spadj = -8;

			/* Keep saved LR at 0(SP) after SP change. */
			/* MOVW 0(SP), REGTMP; MOVW REGTMP, -8!(SP) */
			/* TODO: Remove SP adjustments; see issue 6699. */
			q1->as = AMOVW;
			q1->from.type = TYPE_MEM;
			q1->from.reg = REGSP;
			q1->from.offset = 0;
			q1->reg = 0;
			q1->to.type = TYPE_REG;
			q1->to.reg = REGTMP;

			/* SUB $8,SP */
			q1 = appendp(ctxt, q1);
			q1->as = AMOVW;
			q1->from.type = TYPE_REG;
			q1->from.reg = REGTMP;
			q1->reg = 0;
			q1->to.type = TYPE_MEM;
			q1->to.reg = REGSP;
			q1->to.offset = -8;
			q1->scond |= C_WBIT;
			q1->spadj = 8;

			break;
		case AMOVW:
			if((p->scond & C_WBIT) && p->to.type == TYPE_MEM && p->to.reg == REGSP)
				p->spadj = -p->to.offset;
			if((p->scond & C_PBIT) && p->from.type == TYPE_MEM && p->from.reg == REGSP && p->to.reg != REGPC)
				p->spadj = -p->from.offset;
			if(p->from.type == TYPE_ADDR && p->from.reg == REGSP && p->to.type == TYPE_REG && p->to.reg == REGSP)
				p->spadj = -p->from.offset;
			break;
		}
	}
}

static int
isfloatreg(Addr *a)
{
	return a->type == TYPE_REG && REG_F0 <= a->reg && a->reg <= REG_F15;
}

static void
softfloat(Link *ctxt, LSym *cursym)
{
	Prog *p, *next;
	LSym *symsfloat;
	int wasfloat;

	if(ctxt->goarm > 5)
		return;

	symsfloat = linklookup(ctxt, "_sfloat", 0);

	wasfloat = 0;
	for(p = cursym->text; p != nil; p = p->link)
		if(p->pcond != nil)
			p->pcond->mark |= LABEL;
	for(p = cursym->text; p != nil; p = p->link) {
		switch(p->as) {
		case AMOVW:
			if(isfloatreg(&p->to) || isfloatreg(&p->from))
				goto soft;
			goto notsoft;

		case AMOVWD:
		case AMOVWF:
		case AMOVDW:
		case AMOVFW:
		case AMOVFD:
		case AMOVDF:
		case AMOVF:
		case AMOVD:

		case ACMPF:
		case ACMPD:
		case AADDF:
		case AADDD:
		case ASUBF:
		case ASUBD:
		case AMULF:
		case AMULD:
		case ADIVF:
		case ADIVD:
		case ASQRTF:
		case ASQRTD:
		case AABSF:
		case AABSD:
			goto soft;

		default:
			goto notsoft;
		}

	soft:
		if (!wasfloat || (p->mark&LABEL)) {
			next = emallocz(sizeof(Prog));
			*next = *p;

			// BL _sfloat(SB)
			*p = zprog;
			p->link = next;
			p->as = ABL;
 				p->to.type = TYPE_BRANCH;
			p->to.sym = symsfloat;
			p->lineno = next->lineno;

			p = next;
			wasfloat = 1;
		}
		continue;

	notsoft:
		wasfloat = 0;
	}
}

static Prog*
stacksplit(Link *ctxt, Prog *p, int32 framesize, int noctxt)
{
	// MOVW			g_stackguard(g), R1
	p = appendp(ctxt, p);
	p->as = AMOVW;
	p->from.type = TYPE_MEM;
	p->from.reg = REGG;
	p->from.offset = 2*ctxt->arch->ptrsize;	// G.stackguard0
	if(ctxt->cursym->cfunc)
		p->from.offset = 3*ctxt->arch->ptrsize;	// G.stackguard1
	p->to.type = TYPE_REG;
	p->to.reg = REG_R1;
	
	if(framesize <= StackSmall) {
		// small stack: SP < stackguard
		//	CMP	stackguard, SP
		p = appendp(ctxt, p);
		p->as = ACMP;
		p->from.type = TYPE_REG;
		p->from.reg = REG_R1;
		p->reg = REGSP;
	} else if(framesize <= StackBig) {
		// large stack: SP-framesize < stackguard-StackSmall
		//	MOVW $-framesize(SP), R2
		//	CMP stackguard, R2
		p = appendp(ctxt, p);
		p->as = AMOVW;
		p->from.type = TYPE_ADDR;
		p->from.reg = REGSP;
		p->from.offset = -framesize;
		p->to.type = TYPE_REG;
		p->to.reg = REG_R2;
		
		p = appendp(ctxt, p);
		p->as = ACMP;
		p->from.type = TYPE_REG;
		p->from.reg = REG_R1;
		p->reg = REG_R2;
	} else {
		// Such a large stack we need to protect against wraparound
		// if SP is close to zero.
		//	SP-stackguard+StackGuard < framesize + (StackGuard-StackSmall)
		// The +StackGuard on both sides is required to keep the left side positive:
		// SP is allowed to be slightly below stackguard. See stack.h.
		//	CMP $StackPreempt, R1
		//	MOVW.NE $StackGuard(SP), R2
		//	SUB.NE R1, R2
		//	MOVW.NE $(framesize+(StackGuard-StackSmall)), R3
		//	CMP.NE R3, R2
		p = appendp(ctxt, p);
		p->as = ACMP;
		p->from.type = TYPE_CONST;
		p->from.offset = (uint32)StackPreempt;
		p->reg = REG_R1;

		p = appendp(ctxt, p);
		p->as = AMOVW;
		p->from.type = TYPE_ADDR;
		p->from.reg = REGSP;
		p->from.offset = StackGuard;
		p->to.type = TYPE_REG;
		p->to.reg = REG_R2;
		p->scond = C_SCOND_NE;
		
		p = appendp(ctxt, p);
		p->as = ASUB;
		p->from.type = TYPE_REG;
		p->from.reg = REG_R1;
		p->to.type = TYPE_REG;
		p->to.reg = REG_R2;
		p->scond = C_SCOND_NE;
		
		p = appendp(ctxt, p);
		p->as = AMOVW;
		p->from.type = TYPE_ADDR;
		p->from.offset = framesize + (StackGuard - StackSmall);
		p->to.type = TYPE_REG;
		p->to.reg = REG_R3;
		p->scond = C_SCOND_NE;
		
		p = appendp(ctxt, p);
		p->as = ACMP;
		p->from.type = TYPE_REG;
		p->from.reg = REG_R3;
		p->reg = REG_R2;
		p->scond = C_SCOND_NE;
	}
	
	// MOVW.LS	R14, R3
	p = appendp(ctxt, p);
	p->as = AMOVW;
	p->scond = C_SCOND_LS;
	p->from.type = TYPE_REG;
	p->from.reg = REGLINK;
	p->to.type = TYPE_REG;
	p->to.reg = REG_R3;

	// BL.LS		runtime.morestack(SB) // modifies LR, returns with LO still asserted
	p = appendp(ctxt, p);
	p->as = ABL;
	p->scond = C_SCOND_LS;
	p->to.type = TYPE_BRANCH;
	if(ctxt->cursym->cfunc)
		p->to.sym = linklookup(ctxt, "runtime.morestackc", 0);
	else
		p->to.sym = ctxt->symmorestack[noctxt];
	
	// BLS	start
	p = appendp(ctxt, p);
	p->as = ABLS;
	p->to.type = TYPE_BRANCH;
	p->pcond = ctxt->cursym->text->link;
	
	return p;
}

static void
initdiv(Link *ctxt)
{
	if(ctxt->sym_div != nil)
		return;
	ctxt->sym_div = linklookup(ctxt, "_div", 0);
	ctxt->sym_divu = linklookup(ctxt, "_divu", 0);
	ctxt->sym_mod = linklookup(ctxt, "_mod", 0);
	ctxt->sym_modu = linklookup(ctxt, "_modu", 0);
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
relinv(int a)
{
	switch(a) {
	case ABEQ:	return ABNE;
	case ABNE:	return ABEQ;
	case ABCS:	return ABCC;
	case ABHS:	return ABLO;
	case ABCC:	return ABCS;
	case ABLO:	return ABHS;
	case ABMI:	return ABPL;
	case ABPL:	return ABMI;
	case ABVS:	return ABVC;
	case ABVC:	return ABVS;
	case ABHI:	return ABLS;
	case ABLS:	return ABHI;
	case ABGE:	return ABLT;
	case ABLT:	return ABGE;
	case ABGT:	return ABLE;
	case ABLE:	return ABGT;
	}
	sysfatal("unknown relation: %s", anames5[a]);
	return 0;
}

static void
xfol(Link *ctxt, Prog *p, Prog **last)
{
	Prog *q, *r;
	int a, i;

loop:
	if(p == nil)
		return;
	a = p->as;
	if(a == AB) {
		q = p->pcond;
		if(q != nil && q->as != ATEXT) {
			p->mark |= FOLL;
			p = q;
			if(!(p->mark & FOLL))
				goto loop;
		}
	}
	if(p->mark & FOLL) {
		for(i=0,q=p; i<4; i++,q=q->link) {
			if(q == *last || q == nil)
				break;
			a = q->as;
			if(a == ANOP) {
				i--;
				continue;
			}
			if(a == AB || (a == ARET && q->scond == C_SCOND_NONE) || a == ARFE || a == AUNDEF)
				goto copy;
			if(q->pcond == nil || (q->pcond->mark&FOLL))
				continue;
			if(a != ABEQ && a != ABNE)
				continue;
		copy:
			for(;;) {
				r = emallocz(sizeof(Prog));
				*r = *p;
				if(!(r->mark&FOLL))
					print("can't happen 1\n");
				r->mark |= FOLL;
				if(p != q) {
					p = p->link;
					(*last)->link = r;
					*last = r;
					continue;
				}
				(*last)->link = r;
				*last = r;
				if(a == AB || (a == ARET && q->scond == C_SCOND_NONE) || a == ARFE || a == AUNDEF)
					return;
				r->as = ABNE;
				if(a == ABNE)
					r->as = ABEQ;
				r->pcond = p->link;
				r->link = p->pcond;
				if(!(r->link->mark&FOLL))
					xfol(ctxt, r->link, last);
				if(!(r->pcond->mark&FOLL))
					print("can't happen 2\n");
				return;
			}
		}
		a = AB;
		q = emallocz(sizeof(Prog));
		q->as = a;
		q->lineno = p->lineno;
		q->to.type = TYPE_BRANCH;
		q->to.offset = p->pc;
		q->pcond = p;
		p = q;
	}
	p->mark |= FOLL;
	(*last)->link = p;
	*last = p;
	if(a == AB || (a == ARET && p->scond == C_SCOND_NONE) || a == ARFE || a == AUNDEF){
		return;
	}
	if(p->pcond != nil)
	if(a != ABL && a != ABX && p->link != nil) {
		q = brchain(ctxt, p->link);
		if(a != ATEXT && a != ABCASE)
		if(q != nil && (q->mark&FOLL)) {
			p->as = relinv(a);
			p->link = p->pcond;
			p->pcond = q;
		}
		xfol(ctxt, p->link, last);
		q = brchain(ctxt, p->pcond);
		if(q == nil)
			q = p->pcond;
		if(q->mark&FOLL) {
			p->pcond = q;
			return;
		}
		p = q;
		goto loop;
	}
	p = p->link;
	goto loop;
}

LinkArch linkarm = {
	.name = "arm",
	.thechar = '5',
	.endian = LittleEndian,

	.preprocess = preprocess,
	.assemble = span5,
	.follow = follow,
	.progedit = progedit,

	.minlc = 4,
	.ptrsize = 4,
	.regsize = 4,
};
