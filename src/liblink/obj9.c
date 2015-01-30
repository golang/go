// cmd/9l/noop.c, cmd/9l/pass.c, cmd/9l/span.c from Vita Nuova.
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2008 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2008 Lucent Technologies Inc. and others
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
#include "../cmd/9l/9.out.h"
#include "../runtime/stack.h"
#include "../runtime/funcdata.h"

static void
progedit(Link *ctxt, Prog *p)
{
	char literal[64];
	LSym *s;

	USED(ctxt);

	p->from.class = 0;
	p->to.class = 0;

	// Rewrite BR/BL to symbol as TYPE_BRANCH.
	switch(p->as) {
	case ABR:
	case ABL:
	case ARETURN:
	case ADUFFZERO:
	case ADUFFCOPY:
		if(p->to.sym != nil)
			p->to.type = TYPE_BRANCH;
		break;
	}

	// Rewrite float constants to values stored in memory.
	switch(p->as) {
	case AFMOVS:
		if(p->from.type == TYPE_FCONST) {
			uint32 i32;
			float32 f32;
			f32 = p->from.u.dval;
			memmove(&i32, &f32, 4);
			sprint(literal, "$f32.%08ux", i32);
			s = linklookup(ctxt, literal, 0);
			s->size = 4;
			p->from.type = TYPE_MEM;
			p->from.sym = s;
			p->from.name = NAME_EXTERN;
			p->from.offset = 0;
		}
		break;
	case AFMOVD:
		if(p->from.type == TYPE_FCONST) {
			uint64 i64;
			memmove(&i64, &p->from.u.dval, 8);
			sprint(literal, "$f64.%016llux", i64);
			s = linklookup(ctxt, literal, 0);
			s->size = 8;
			p->from.type = TYPE_MEM;
			p->from.sym = s;
			p->from.name = NAME_EXTERN;
			p->from.offset = 0;
		}
		break;
	case AMOVD:
		// Put >32-bit constants in memory and load them
		if(p->from.type == TYPE_CONST && p->from.name == NAME_NONE && p->from.reg == 0 && (int32)p->from.offset != p->from.offset) {
			sprint(literal, "$i64.%016llux", (uvlong)p->from.offset);
			s = linklookup(ctxt, literal, 0);
			s->size = 8;
			p->from.type = TYPE_MEM;
			p->from.sym = s;
			p->from.name = NAME_EXTERN;
			p->from.offset = 0;
		}
	}

	// Rewrite SUB constants into ADD.
	switch(p->as) {
	case ASUBC:
		if(p->from.type == TYPE_CONST) {
			p->from.offset = -p->from.offset;
			p->as = AADDC;
		}
		break;

	case ASUBCCC:
		if(p->from.type == TYPE_CONST) {
			p->from.offset = -p->from.offset;
			p->as = AADDCCC;
		}
		break;

	case ASUB:
		if(p->from.type == TYPE_CONST) {
			p->from.offset = -p->from.offset;
			p->as = AADD;
		}
		break;
	}
}

static Prog*	stacksplit(Link*, Prog*, int32, int);

static void
preprocess(Link *ctxt, LSym *cursym)
{
	Prog *p, *q, *p1, *p2, *q1;
	int o, mov, aoffset;
	vlong textstksiz;
	int32 autosize;

	if(ctxt->symmorestack[0] == nil) {
		ctxt->symmorestack[0] = linklookup(ctxt, "runtime.morestack", 0);
		ctxt->symmorestack[1] = linklookup(ctxt, "runtime.morestack_noctxt", 0);
		// TODO(minux): add morestack short-cuts with small fixed frame-size.
	}

	ctxt->cursym = cursym;

	if(cursym->text == nil || cursym->text->link == nil)
		return;				

	p = cursym->text;
	textstksiz = p->to.offset;
	
	cursym->args = p->to.u.argsize;
	cursym->locals = textstksiz;

	/*
	 * find leaf subroutines
	 * strip NOPs
	 * expand RET
	 * expand BECOME pseudo
	 */

	if(ctxt->debugvlog)
		Bprint(ctxt->bso, "%5.2f noops\n", cputime());
	Bflush(ctxt->bso);

	q = nil;
	for(p = cursym->text; p != nil; p = p->link) {
		switch(p->as) {
		/* too hard, just leave alone */
		case ATEXT:
			q = p;
			p->mark |= LABEL|LEAF|SYNC;
			if(p->link)
				p->link->mark |= LABEL;
			break;

		case ANOR:
			q = p;
			if(p->to.type == TYPE_REG)
				if(p->to.reg == REGZERO)
					p->mark |= LABEL|SYNC;
			break;

		case ALWAR:
		case ASTWCCC:
		case AECIWX:
		case AECOWX:
		case AEIEIO:
		case AICBI:
		case AISYNC:
		case ATLBIE:
		case ATLBIEL:
		case ASLBIA:
		case ASLBIE:
		case ASLBMFEE:
		case ASLBMFEV:
		case ASLBMTE:
		case ADCBF:
		case ADCBI:
		case ADCBST:
		case ADCBT:
		case ADCBTST:
		case ADCBZ:
		case ASYNC:
		case ATLBSYNC:
		case APTESYNC:
		case ATW:
		case AWORD:
		case ARFI:
		case ARFCI:
		case ARFID:
		case AHRFID:
			q = p;
			p->mark |= LABEL|SYNC;
			continue;

		case AMOVW:
		case AMOVWZ:
		case AMOVD:
			q = p;
			if(p->from.reg >= REG_SPECIAL || p->to.reg >= REG_SPECIAL)
				p->mark |= LABEL|SYNC;
			continue;

		case AFABS:
		case AFABSCC:
		case AFADD:
		case AFADDCC:
		case AFCTIW:
		case AFCTIWCC:
		case AFCTIWZ:
		case AFCTIWZCC:
		case AFDIV:
		case AFDIVCC:
		case AFMADD:
		case AFMADDCC:
		case AFMOVD:
		case AFMOVDU:
		/* case AFMOVDS: */
		case AFMOVS:
		case AFMOVSU:
		/* case AFMOVSD: */
		case AFMSUB:
		case AFMSUBCC:
		case AFMUL:
		case AFMULCC:
		case AFNABS:
		case AFNABSCC:
		case AFNEG:
		case AFNEGCC:
		case AFNMADD:
		case AFNMADDCC:
		case AFNMSUB:
		case AFNMSUBCC:
		case AFRSP:
		case AFRSPCC:
		case AFSUB:
		case AFSUBCC:
			q = p;
			p->mark |= FLOAT;
			continue;

		case ABL:
		case ABCL:
		case ADUFFZERO:
		case ADUFFCOPY:
			cursym->text->mark &= ~LEAF;

		case ABC:
		case ABEQ:
		case ABGE:
		case ABGT:
		case ABLE:
		case ABLT:
		case ABNE:
		case ABR:
		case ABVC:
		case ABVS:
			p->mark |= BRANCH;
			q = p;
			q1 = p->pcond;
			if(q1 != nil) {
				while(q1->as == ANOP) {
					q1 = q1->link;
					p->pcond = q1;
				}
				if(!(q1->mark & LEAF))
					q1->mark |= LABEL;
			} else
				p->mark |= LABEL;
			q1 = p->link;
			if(q1 != nil)
				q1->mark |= LABEL;
			continue;

		case AFCMPO:
		case AFCMPU:
			q = p;
			p->mark |= FCMP|FLOAT;
			continue;

		case ARETURN:
			q = p;
			if(p->link != nil)
				p->link->mark |= LABEL;
			continue;

		case ANOP:
			q1 = p->link;
			q->link = q1;		/* q is non-nop */
			q1->mark |= p->mark;
			continue;

		default:
			q = p;
			continue;
		}
	}

	autosize = 0;
	for(p = cursym->text; p != nil; p = p->link) {
		o = p->as;
		switch(o) {
		case ATEXT:
			mov = AMOVD;
			aoffset = 0;
			autosize = textstksiz + 8;
			if((p->mark & LEAF) && autosize <= 8)
				autosize = 0;
			else
				if(autosize & 4)
					autosize += 4;
			p->to.offset = autosize-8;

			if(!(p->from3.offset & NOSPLIT))
				p = stacksplit(ctxt, p, autosize, !(cursym->text->from3.offset&NEEDCTXT)); // emit split check

			q = p;
			if(autosize) {
				/* use MOVDU to adjust R1 when saving R31, if autosize is small */
				if(!(cursym->text->mark & LEAF) && autosize >= -BIG && autosize <= BIG) {
					mov = AMOVDU;
					aoffset = -autosize;
				} else {
					q = appendp(ctxt, p);
					q->as = AADD;
					q->lineno = p->lineno;
					q->from.type = TYPE_CONST;
					q->from.offset = -autosize;
					q->to.type = TYPE_REG;
					q->to.reg = REGSP;
					q->spadj = +autosize;
				}
			} else
			if(!(cursym->text->mark & LEAF)) {
				if(ctxt->debugvlog) {
					Bprint(ctxt->bso, "save suppressed in: %s\n",
						cursym->name);
					Bflush(ctxt->bso);
				}
				cursym->text->mark |= LEAF;
			}

			if(cursym->text->mark & LEAF) {
				cursym->leaf = 1;
				break;
			}

			q = appendp(ctxt, q);
			q->as = AMOVD;
			q->lineno = p->lineno;
			q->from.type = TYPE_REG;
			q->from.reg = REG_LR;
			q->to.type = TYPE_REG;
			q->to.reg = REGTMP;

			q = appendp(ctxt, q);
			q->as = mov;
			q->lineno = p->lineno;
			q->from.type = TYPE_REG;
			q->from.reg = REGTMP;
			q->to.type = TYPE_MEM;
			q->to.offset = aoffset;
			q->to.reg = REGSP;
			if(q->as == AMOVDU)
				q->spadj = -aoffset;

			if(cursym->text->from3.offset & WRAPPER) {
				// if(g->panic != nil && g->panic->argp == FP) g->panic->argp = bottom-of-frame
				//
				//	MOVD g_panic(g), R3
				//	CMP R0, R3
				//	BEQ end
				//	MOVD panic_argp(R3), R4
				//	ADD $(autosize+8), R1, R5
				//	CMP R4, R5
				//	BNE end
				//	ADD $8, R1, R6
				//	MOVD R6, panic_argp(R3)
				// end:
				//	NOP
				//
				// The NOP is needed to give the jumps somewhere to land.
				// It is a liblink NOP, not a ppc64 NOP: it encodes to 0 instruction bytes.


				q = appendp(ctxt, q);
				q->as = AMOVD;
				q->from.type = TYPE_MEM;
				q->from.reg = REGG;
				q->from.offset = 4*ctxt->arch->ptrsize; // G.panic
				q->to.type = TYPE_REG;
				q->to.reg = REG_R3;

				q = appendp(ctxt, q);
				q->as = ACMP;
				q->from.type = TYPE_REG;
				q->from.reg = REG_R0;
				q->to.type = TYPE_REG;
				q->to.reg = REG_R3;

				q = appendp(ctxt, q);
				q->as = ABEQ;
				q->to.type = TYPE_BRANCH;
				p1 = q;

				q = appendp(ctxt, q);
				q->as = AMOVD;
				q->from.type = TYPE_MEM;
				q->from.reg = REG_R3;
				q->from.offset = 0; // Panic.argp
				q->to.type = TYPE_REG;
				q->to.reg = REG_R4;

				q = appendp(ctxt, q);
				q->as = AADD;
				q->from.type = TYPE_CONST;
				q->from.offset = autosize+8;
				q->reg = REGSP;
				q->to.type = TYPE_REG;
				q->to.reg = REG_R5;

				q = appendp(ctxt, q);
				q->as = ACMP;
				q->from.type = TYPE_REG;
				q->from.reg = REG_R4;
				q->to.type = TYPE_REG;
				q->to.reg = REG_R5;

				q = appendp(ctxt, q);
				q->as = ABNE;
				q->to.type = TYPE_BRANCH;
				p2 = q;

				q = appendp(ctxt, q);
				q->as = AADD;
				q->from.type = TYPE_CONST;
				q->from.offset = 8;
				q->reg = REGSP;
				q->to.type = TYPE_REG;
				q->to.reg = REG_R6;

				q = appendp(ctxt, q);
				q->as = AMOVD;
				q->from.type = TYPE_REG;
				q->from.reg = REG_R6;
				q->to.type = TYPE_MEM;
				q->to.reg = REG_R3;
				q->to.offset = 0; // Panic.argp

				q = appendp(ctxt, q);
				q->as = ANOP;
				p1->pcond = q;
				p2->pcond = q;
			}

			break;

		case ARETURN:
			if(p->from.type == TYPE_CONST) {
				ctxt->diag("using BECOME (%P) is not supported!", p);
				break;
			}
			if(p->to.sym) { // retjmp
				p->as = ABR;
				p->to.type = TYPE_BRANCH;
				break;
			}
			if(cursym->text->mark & LEAF) {
				if(!autosize) {
					p->as = ABR;
					p->from = zprog.from;
					p->to.type = TYPE_REG;
					p->to.reg = REG_LR;
					p->mark |= BRANCH;
					break;
				}

				p->as = AADD;
				p->from.type = TYPE_CONST;
				p->from.offset = autosize;
				p->to.type = TYPE_REG;
				p->to.reg = REGSP;
				p->spadj = -autosize;

				q = emallocz(sizeof(Prog));
				q->as = ABR;
				q->lineno = p->lineno;
				q->to.type = TYPE_REG;
				q->to.reg = REG_LR;
				q->mark |= BRANCH;
				q->spadj = +autosize;

				q->link = p->link;
				p->link = q;
				break;
			}

			p->as = AMOVD;
			p->from.type = TYPE_MEM;
			p->from.offset = 0;
			p->from.reg = REGSP;
			p->to.type = TYPE_REG;
			p->to.reg = REGTMP;

			q = emallocz(sizeof(Prog));
			q->as = AMOVD;
			q->lineno = p->lineno;
			q->from.type = TYPE_REG;
			q->from.reg = REGTMP;
			q->to.type = TYPE_REG;
			q->to.reg = REG_LR;

			q->link = p->link;
			p->link = q;
			p = q;

			if(0) {
				// Debug bad returns
				q = emallocz(sizeof(Prog));
				q->as = AMOVD;
				q->lineno = p->lineno;
				q->from.type = TYPE_MEM;
				q->from.offset = 0;
				q->from.reg = REGTMP;
				q->to.type = TYPE_REG;
				q->to.reg = REGTMP;

				q->link = p->link;
				p->link = q;
				p = q;
			}

			if(autosize) {
				q = emallocz(sizeof(Prog));
				q->as = AADD;
				q->lineno = p->lineno;
				q->from.type = TYPE_CONST;
				q->from.offset = autosize;
				q->to.type = TYPE_REG;
				q->to.reg = REGSP;
				q->spadj = -autosize;

				q->link = p->link;
				p->link = q;
			}

			q1 = emallocz(sizeof(Prog));
			q1->as = ABR;
			q1->lineno = p->lineno;
			q1->to.type = TYPE_REG;
			q1->to.reg = REG_LR;
			q1->mark |= BRANCH;
			q1->spadj = +autosize;

			q1->link = q->link;
			q->link = q1;
			break;

		case AADD:
			if(p->to.type == TYPE_REG && p->to.reg == REGSP && p->from.type == TYPE_CONST)
				p->spadj = -p->from.offset;
			break;
		}
	}

/*
// instruction scheduling
	if(debug['Q'] == 0)
		return;

	curtext = nil;
	q = nil;	// p - 1
	q1 = firstp;	// top of block
	o = 0;		// count of instructions
	for(p = firstp; p != nil; p = p1) {
		p1 = p->link;
		o++;
		if(p->mark & NOSCHED){
			if(q1 != p){
				sched(q1, q);
			}
			for(; p != nil; p = p->link){
				if(!(p->mark & NOSCHED))
					break;
				q = p;
			}
			p1 = p;
			q1 = p;
			o = 0;
			continue;
		}
		if(p->mark & (LABEL|SYNC)) {
			if(q1 != p)
				sched(q1, q);
			q1 = p;
			o = 1;
		}
		if(p->mark & (BRANCH|SYNC)) {
			sched(q1, p);
			q1 = p1;
			o = 0;
		}
		if(o >= NSCHED) {
			sched(q1, p);
			q1 = p1;
			o = 0;
		}
		q = p;
	}
*/
}

static Prog*
stacksplit(Link *ctxt, Prog *p, int32 framesize, int noctxt)
{
	Prog *q, *q1;

	// MOVD	g_stackguard(g), R3
	p = appendp(ctxt, p);
	p->as = AMOVD;
	p->from.type = TYPE_MEM;
	p->from.reg = REGG;
	p->from.offset = 2*ctxt->arch->ptrsize;	// G.stackguard0
	if(ctxt->cursym->cfunc)
		p->from.offset = 3*ctxt->arch->ptrsize;	// G.stackguard1
	p->to.type = TYPE_REG;
	p->to.reg = REG_R3;

	q = nil;
	if(framesize <= StackSmall) {
		// small stack: SP < stackguard
		//	CMP	stackguard, SP
		p = appendp(ctxt, p);
		p->as = ACMPU;
		p->from.type = TYPE_REG;
		p->from.reg = REG_R3;
		p->to.type = TYPE_REG;
		p->to.reg = REGSP;
	} else if(framesize <= StackBig) {
		// large stack: SP-framesize < stackguard-StackSmall
		//	ADD $-framesize, SP, R4
		//	CMP stackguard, R4
		p = appendp(ctxt, p);
		p->as = AADD;
		p->from.type = TYPE_CONST;
		p->from.offset = -framesize;
		p->reg = REGSP;
		p->to.type = TYPE_REG;
		p->to.reg = REG_R4;

		p = appendp(ctxt, p);
		p->as = ACMPU;
		p->from.type = TYPE_REG;
		p->from.reg = REG_R3;
		p->to.type = TYPE_REG;
		p->to.reg = REG_R4;
	} else {
		// Such a large stack we need to protect against wraparound.
		// If SP is close to zero:
		//	SP-stackguard+StackGuard <= framesize + (StackGuard-StackSmall)
		// The +StackGuard on both sides is required to keep the left side positive:
		// SP is allowed to be slightly below stackguard. See stack.h.
		//
		// Preemption sets stackguard to StackPreempt, a very large value.
		// That breaks the math above, so we have to check for that explicitly.
		//	// stackguard is R3
		//	CMP	R3, $StackPreempt
		//	BEQ	label-of-call-to-morestack
		//	ADD	$StackGuard, SP, R4
		//	SUB	R3, R4
		//	MOVD	$(framesize+(StackGuard-StackSmall)), R31
		//	CMPU	R31, R4
		p = appendp(ctxt, p);
		p->as = ACMP;
		p->from.type = TYPE_REG;
		p->from.reg = REG_R3;
		p->to.type = TYPE_CONST;
		p->to.offset = StackPreempt;

		q = p = appendp(ctxt, p);
		p->as = ABEQ;
		p->to.type = TYPE_BRANCH;

		p = appendp(ctxt, p);
		p->as = AADD;
		p->from.type = TYPE_CONST;
		p->from.offset = StackGuard;
		p->reg = REGSP;
		p->to.type = TYPE_REG;
		p->to.reg = REG_R4;

		p = appendp(ctxt, p);
		p->as = ASUB;
		p->from.type = TYPE_REG;
		p->from.reg = REG_R3;
		p->to.type = TYPE_REG;
		p->to.reg = REG_R4;

		p = appendp(ctxt, p);
		p->as = AMOVD;
		p->from.type = TYPE_CONST;
		p->from.offset = framesize + StackGuard - StackSmall;
		p->to.type = TYPE_REG;
		p->to.reg = REGTMP;

		p = appendp(ctxt, p);
		p->as = ACMPU;
		p->from.type = TYPE_REG;
		p->from.reg = REGTMP;
		p->to.type = TYPE_REG;
		p->to.reg = REG_R4;
	}

	// q1: BLT	done
	q1 = p = appendp(ctxt, p);
	p->as = ABLT;
	p->to.type = TYPE_BRANCH;

	// MOVD	LR, R5
	p = appendp(ctxt, p);
	p->as = AMOVD;
	p->from.type = TYPE_REG;
	p->from.reg = REG_LR;
	p->to.type = TYPE_REG;
	p->to.reg = REG_R5;
	if(q)
		q->pcond = p;

	// BL	runtime.morestack(SB)
	p = appendp(ctxt, p);
	p->as = ABL;
	p->to.type = TYPE_BRANCH;
	if(ctxt->cursym->cfunc)
		p->to.sym = linklookup(ctxt, "runtime.morestackc", 0);
	else
		p->to.sym = ctxt->symmorestack[noctxt];

	// BR	start
	p = appendp(ctxt, p);
	p->as = ABR;
	p->to.type = TYPE_BRANCH;
	p->pcond = ctxt->cursym->text->link;

	// placeholder for q1's jump target
	p = appendp(ctxt, p);
	p->as = ANOP; // zero-width place holder
	q1->pcond = p;

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
relinv(int a)
{

	switch(a) {
	case ABEQ:	return ABNE;
	case ABNE:	return ABEQ;

	case ABGE:	return ABLT;
	case ABLT:	return ABGE;

	case ABGT:	return ABLE;
	case ABLE:	return ABGT;

	case ABVC:	return ABVS;
	case ABVS:	return ABVC;
	}
	return 0;
}

static void
xfol(Link *ctxt, Prog *p, Prog **last)
{
	Prog *q, *r;
	int a, b, i;

loop:
	if(p == nil)
		return;
	a = p->as;
	if(a == ABR) {
		q = p->pcond;
		if((p->mark&NOSCHED) || q && (q->mark&NOSCHED)){
			p->mark |= FOLL;
			(*last)->link = p;
			*last = p;
			p = p->link;
			xfol(ctxt, p, last);
			p = q;
			if(p && !(p->mark & FOLL))
				goto loop;
			return;
		}
		if(q != nil) {
			p->mark |= FOLL;
			p = q;
			if(!(p->mark & FOLL))
				goto loop;
		}
	}
	if(p->mark & FOLL) {
		for(i=0,q=p; i<4; i++,q=q->link) {
			if(q == *last || (q->mark&NOSCHED))
				break;
			b = 0;		/* set */
			a = q->as;
			if(a == ANOP) {
				i--;
				continue;
			}
			if(a == ABR || a == ARETURN || a == ARFI || a == ARFCI || a == ARFID || a == AHRFID)
				goto copy;
			if(!q->pcond || (q->pcond->mark&FOLL))
				continue;
			b = relinv(a);
			if(!b)
				continue;
		copy:
			for(;;) {
				r = emallocz(sizeof(Prog));
				*r = *p;
				if(!(r->mark&FOLL))
					print("cant happen 1\n");
				r->mark |= FOLL;
				if(p != q) {
					p = p->link;
					(*last)->link = r;
					*last = r;
					continue;
				}
				(*last)->link = r;
				*last = r;
				if(a == ABR || a == ARETURN || a == ARFI || a == ARFCI || a == ARFID || a == AHRFID)
					return;
				r->as = b;
				r->pcond = p->link;
				r->link = p->pcond;
				if(!(r->link->mark&FOLL))
					xfol(ctxt, r->link, last);
				if(!(r->pcond->mark&FOLL))
					print("cant happen 2\n");
				return;
			}
		}

		a = ABR;
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
	if(a == ABR || a == ARETURN || a == ARFI || a == ARFCI || a == ARFID || a == AHRFID){
		if(p->mark & NOSCHED){
			p = p->link;
			goto loop;
		}
		return;
	}
	if(p->pcond != nil)
	if(a != ABL && p->link != nil) {
		xfol(ctxt, p->link, last);
		p = p->pcond;
		if(p == nil || (p->mark&FOLL))
			return;
		goto loop;
	}
	p = p->link;
	goto loop;
}

LinkArch linkppc64 = {
	.name = "ppc64",
	.thechar = '9',
	.endian = BigEndian,

	.preprocess = preprocess,
	.assemble = span9,
	.follow = follow,
	.progedit = progedit,

	.minlc = 4,
	.ptrsize = 8,
	.regsize = 8,
};

LinkArch linkppc64le = {
	.name = "ppc64le",
	.thechar = '9',
	.endian = LittleEndian,

	.preprocess = preprocess,
	.assemble = span9,
	.follow = follow,
	.progedit = progedit,

	.minlc = 4,
	.ptrsize = 8,
	.regsize = 8,
};
