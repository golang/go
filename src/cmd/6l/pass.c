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

// Code and data passes.

#include	"l.h"
#include	"../ld/lib.h"
#include "../../pkg/runtime/stack.h"

static void xfol(Prog*, Prog**);

Prog*
brchain(Prog *p)
{
	int i;

	for(i=0; i<20; i++) {
		if(p == P || p->as != AJMP)
			return p;
		p = p->pcond;
	}
	return P;
}

void
follow(void)
{
	Prog *firstp, *lastp;

	if(debug['v'])
		Bprint(&bso, "%5.2f follow\n", cputime());
	Bflush(&bso);
	
	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		firstp = prg();
		lastp = firstp;
		xfol(cursym->text, &lastp);
		lastp->link = nil;
		cursym->text = firstp->link;
	}
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

static void
xfol(Prog *p, Prog **last)
{
	Prog *q;
	int i;
	enum as a;

loop:
	if(p == P)
		return;
	if(p->as == AJMP)
	if((q = p->pcond) != P && q->as != ATEXT) {
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
			if(q == P)
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
			if(q->pcond == P || q->pcond->mark)
				continue;
			if(a == ACALL || a == ALOOP)
				continue;
			for(;;) {
				if(p->as == ANOP) {
					p = p->link;
					continue;
				}
				q = copyp(p);
				p = p->link;
				q->mark = 1;
				(*last)->link = q;
				*last = q;
				if(q->as != a || q->pcond == P || q->pcond->mark)
					continue;

				q->as = relinv(q->as);
				p = q->pcond;
				q->pcond = q->link;
				q->link = p;
				xfol(q->link, last);
				p = q->link;
				if(p->mark)
					return;
				goto loop;
			}
		} /* */
		q = prg();
		q->as = AJMP;
		q->line = p->line;
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
	if(p->pcond != P && a != ACALL) {
		/*
		 * some kind of conditional branch.
		 * recurse to follow one path.
		 * continue loop on the other.
		 */
		if((q = brchain(p->pcond)) != P)
			p->pcond = q;
		if((q = brchain(p->link)) != P)
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
		xfol(p->link, last);
		if(p->pcond->mark)
			return;
		p = p->pcond;
		goto loop;
	}
	p = p->link;
	goto loop;
}

Prog*
byteq(int v)
{
	Prog *p;

	p = prg();
	p->as = ABYTE;
	p->from.type = D_CONST;
	p->from.offset = v&0xff;
	return p;
}

int
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
	diag("unknown relation: %s in %s", anames[a], TNAME);
	errorexit();
	return a;
}

void
patch(void)
{
	int32 c;
	Prog *p, *q;
	Sym *s;
	int32 vexit;

	if(debug['v'])
		Bprint(&bso, "%5.2f mkfwd\n", cputime());
	Bflush(&bso);
	mkfwd();
	if(debug['v'])
		Bprint(&bso, "%5.2f patch\n", cputime());
	Bflush(&bso);

	s = lookup("exit", 0);
	vexit = s->value;
	for(cursym = textp; cursym != nil; cursym = cursym->next)
	for(p = cursym->text; p != P; p = p->link) {
		if(HEADTYPE == Hwindows) { 
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
				q = appendp(p);
				q->from = p->from;
				q->from.type = D_INDIR + p->to.type;
				q->to = p->to;
				q->as = p->as;
				p->as = AMOVQ;
				p->from.type = D_INDIR+D_GS;
				p->from.offset = 0x28;
			}
		}
		if(HEADTYPE == Hlinux || HEADTYPE == Hfreebsd
		|| HEADTYPE == Hopenbsd || HEADTYPE == Hnetbsd
		|| HEADTYPE == Hplan9x64) {
			// ELF uses FS instead of GS.
			if(p->from.type == D_INDIR+D_GS)
				p->from.type = D_INDIR+D_FS;
			if(p->to.type == D_INDIR+D_GS)
				p->to.type = D_INDIR+D_FS;
		}
		if(p->as == ACALL || (p->as == AJMP && p->to.type != D_BRANCH)) {
			s = p->to.sym;
			if(s) {
				if(debug['c'])
					Bprint(&bso, "%s calls %s\n", TNAME, s->name);
				if((s->type&SMASK) != STEXT) {
					/* diag prints TNAME first */
					diag("undefined: %s", s->name);
					s->type = STEXT;
					s->value = vexit;
					continue;	// avoid more error messages
				}
				if(s->text == nil)
					continue;
				p->to.type = D_BRANCH;
				p->to.offset = s->text->pc;
				p->pcond = s->text;
				continue;
			}
		}
		if(p->to.type != D_BRANCH)
			continue;
		c = p->to.offset;
		for(q = cursym->text; q != P;) {
			if(c == q->pc)
				break;
			if(q->forwd != P && c >= q->forwd->pc)
				q = q->forwd;
			else
				q = q->link;
		}
		if(q == P) {
			diag("branch out of range in %s (%#ux)\n%P [%s]",
				TNAME, c, p, p->to.sym ? p->to.sym->name : "<nil>");
			p->to.type = D_NONE;
		}
		p->pcond = q;
	}

	for(cursym = textp; cursym != nil; cursym = cursym->next)
	for(p = cursym->text; p != P; p = p->link) {
		p->mark = 0;	/* initialization for follow */
		if(p->pcond != P) {
			p->pcond = brloop(p->pcond);
			if(p->pcond != P)
			if(p->to.type == D_BRANCH)
				p->to.offset = p->pcond->pc;
		}
	}
}

Prog*
brloop(Prog *p)
{
	int c;
	Prog *q;

	c = 0;
	for(q = p; q != P; q = q->pcond) {
		if(q->as != AJMP)
			break;
		c++;
		if(c >= 5000)
			return P;
	}
	return q;
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
Prog*	pmorestack[nelem(morename)];
Sym*	symmorestack[nelem(morename)];

void
dostkoff(void)
{
	Prog *p, *q, *q1;
	int32 autoffset, deltasp;
	int a, pcsize;
	uint32 moreconst1, moreconst2, i;

	for(i=0; i<nelem(morename); i++) {
		symmorestack[i] = lookup(morename[i], 0);
		if(symmorestack[i]->type != STEXT)
			diag("morestack trampoline not defined - %s", morename[i]);
		pmorestack[i] = symmorestack[i]->text;
	}

	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		if(cursym->text == nil || cursym->text->link == nil)
			continue;				

		p = cursym->text;
		parsetextconst(p->to.offset);
		autoffset = textstksiz;
		if(autoffset < 0)
			autoffset = 0;

		if(autoffset < StackSmall && !(p->from.scale & NOSPLIT)) {
			for(q = p; q != P; q = q->link)
				if(q->as == ACALL)
					goto noleaf;
			p->from.scale |= NOSPLIT;
		noleaf:;
		}

		q = P;
		if((p->from.scale & NOSPLIT) && autoffset >= StackSmall)
			diag("nosplit func likely to overflow stack");

		if(!(p->from.scale & NOSPLIT)) {
			p = appendp(p);	// load g into CX
			p->as = AMOVQ;
			if(HEADTYPE == Hlinux || HEADTYPE == Hfreebsd
			|| HEADTYPE == Hopenbsd || HEADTYPE == Hnetbsd
			|| HEADTYPE == Hplan9x64)	// ELF uses FS
				p->from.type = D_INDIR+D_FS;
			else
				p->from.type = D_INDIR+D_GS;
			p->from.offset = tlsoffset+0;
			p->to.type = D_CX;
			if(HEADTYPE == Hwindows) {
				// movq %gs:0x28, %rcx
				// movq (%rcx), %rcx
				p->as = AMOVQ;
				p->from.type = D_INDIR+D_GS;
				p->from.offset = 0x28;
				p->to.type = D_CX;

			
				p = appendp(p);
				p->as = AMOVQ;
				p->from.type = D_INDIR+D_CX;
				p->from.offset = 0;
				p->to.type = D_CX;
			}

			if(debug['K']) {
				// 6l -K means check not only for stack
				// overflow but stack underflow.
				// On underflow, INT 3 (breakpoint).
				// Underflow itself is rare but this also
				// catches out-of-sync stack guard info

				p = appendp(p);
				p->as = ACMPQ;
				p->from.type = D_INDIR+D_CX;
				p->from.offset = 8;
				p->to.type = D_SP;

				p = appendp(p);
				p->as = AJHI;
				p->to.type = D_BRANCH;
				p->to.offset = 4;
				q1 = p;

				p = appendp(p);
				p->as = AINT;
				p->from.type = D_CONST;
				p->from.offset = 3;

				p = appendp(p);
				p->as = ANOP;
				q1->pcond = p;
			}

			if(autoffset < StackBig) {  // do we need to call morestack?
				if(autoffset <= StackSmall) {
					// small stack
					p = appendp(p);
					p->as = ACMPQ;
					p->from.type = D_SP;
					p->to.type = D_INDIR+D_CX;
				} else {
					// large stack
					p = appendp(p);
					p->as = ALEAQ;
					p->from.type = D_INDIR+D_SP;
					p->from.offset = -(autoffset-StackSmall);
					p->to.type = D_AX;

					p = appendp(p);
					p->as = ACMPQ;
					p->from.type = D_AX;
					p->to.type = D_INDIR+D_CX;
				}

				// common
				p = appendp(p);
				p->as = AJHI;
				p->to.type = D_BRANCH;
				p->to.offset = 4;
				q = p;
			}

			// If we ask for more stack, we'll get a minimum of StackMin bytes.
			// We need a stack frame large enough to hold the top-of-stack data,
			// the function arguments+results, our caller's PC, our frame,
			// a word for the return PC of the next call, and then the StackLimit bytes
			// that must be available on entry to any function called from a function
			// that did a stack check.  If StackMin is enough, don't ask for a specific
			// amount: then we can use the custom functions and save a few
			// instructions.
			moreconst1 = 0;
			if(StackTop + textarg + PtrSize + autoffset + PtrSize + StackLimit >= StackMin)
				moreconst1 = autoffset;
			moreconst2 = textarg;

			// 4 varieties varieties (const1==0 cross const2==0)
			// and 6 subvarieties of (const1==0 and const2!=0)
			p = appendp(p);
			if(moreconst1 == 0 && moreconst2 == 0) {
				p->as = ACALL;
				p->to.type = D_BRANCH;
				p->pcond = pmorestack[0];
				p->to.sym = symmorestack[0];
			} else
			if(moreconst1 != 0 && moreconst2 == 0) {
				p->as = AMOVL;
				p->from.type = D_CONST;
				p->from.offset = moreconst1;
				p->to.type = D_AX;

				p = appendp(p);
				p->as = ACALL;
				p->to.type = D_BRANCH;
				p->pcond = pmorestack[1];
				p->to.sym = symmorestack[1];
			} else
			if(moreconst1 == 0 && moreconst2 <= 48 && moreconst2%8 == 0) {
				i = moreconst2/8 + 3;
				p->as = ACALL;
				p->to.type = D_BRANCH;
				p->pcond = pmorestack[i];
				p->to.sym = symmorestack[i];
			} else
			if(moreconst1 == 0 && moreconst2 != 0) {
				p->as = AMOVL;
				p->from.type = D_CONST;
				p->from.offset = moreconst2;
				p->to.type = D_AX;

				p = appendp(p);
				p->as = ACALL;
				p->to.type = D_BRANCH;
				p->pcond = pmorestack[2];
				p->to.sym = symmorestack[2];
			} else {
				p->as = AMOVQ;
				p->from.type = D_CONST;
				p->from.offset = (uint64)moreconst2 << 32;
				p->from.offset |= moreconst1;
				p->to.type = D_AX;

				p = appendp(p);
				p->as = ACALL;
				p->to.type = D_BRANCH;
				p->pcond = pmorestack[3];
				p->to.sym = symmorestack[3];
			}
		}

		if(q != P)
			q->pcond = p->link;

		if(autoffset) {
			p = appendp(p);
			p->as = AADJSP;
			p->from.type = D_CONST;
			p->from.offset = autoffset;
			p->spadj = autoffset;
			if(q != P)
				q->pcond = p;
		} else {
			// zero-byte stack adjustment.
			// Insert a fake non-zero adjustment so that stkcheck can
			// recognize the end of the stack-splitting prolog.
			p = appendp(p);
			p->as = ANOP;
			p->spadj = -PtrSize;
			p = appendp(p);
			p->as = ANOP;
			p->spadj = PtrSize;
		}
		deltasp = autoffset;

		if(debug['K'] > 1 && autoffset) {
			// 6l -KK means double-check for stack overflow
			// even after calling morestack and even if the
			// function is marked as nosplit.
			p = appendp(p);
			p->as = AMOVQ;
			p->from.type = D_INDIR+D_CX;
			p->from.offset = 0;
			p->to.type = D_BX;

			p = appendp(p);
			p->as = ASUBQ;
			p->from.type = D_CONST;
			p->from.offset = StackSmall+32;
			p->to.type = D_BX;

			p = appendp(p);
			p->as = ACMPQ;
			p->from.type = D_SP;
			p->to.type = D_BX;

			p = appendp(p);
			p->as = AJHI;
			p->to.type = D_BRANCH;
			q1 = p;

			p = appendp(p);
			p->as = AINT;
			p->from.type = D_CONST;
			p->from.offset = 3;

			p = appendp(p);
			p->as = ANOP;
			q1->pcond = p;
		}
		
		if(debug['Z'] && autoffset && !(cursym->text->from.scale&NOSPLIT)) {
			// 6l -Z means zero the stack frame on entry.
			// This slows down function calls but can help avoid
			// false positives in garbage collection.
			p = appendp(p);
			p->as = AMOVQ;
			p->from.type = D_SP;
			p->to.type = D_DI;
			
			p = appendp(p);
			p->as = AMOVQ;
			p->from.type = D_CONST;
			p->from.offset = autoffset/8;
			p->to.type = D_CX;
			
			p = appendp(p);
			p->as = AMOVQ;
			p->from.type = D_CONST;
			p->from.offset = 0;
			p->to.type = D_AX;
			
			p = appendp(p);
			p->as = AREP;
			
			p = appendp(p);
			p->as = ASTOSQ;
		}
		
		for(; p != P; p = p->link) {
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
				diag("unbalanced PUSH/POP");
	
			if(autoffset) {
				p->as = AADJSP;
				p->from.type = D_CONST;
				p->from.offset = -autoffset;
				p->spadj = -autoffset;
				p = appendp(p);
				p->as = ARET;
				// If there are instructions following
				// this ARET, they come from a branch
				// with the same stackframe, so undo
				// the cleanup.
				p->spadj = +autoffset;
			}
		}
	}
}

vlong
atolwhex(char *s)
{
	vlong n;
	int f;

	n = 0;
	f = 0;
	while(*s == ' ' || *s == '\t')
		s++;
	if(*s == '-' || *s == '+') {
		if(*s++ == '-')
			f = 1;
		while(*s == ' ' || *s == '\t')
			s++;
	}
	if(s[0]=='0' && s[1]){
		if(s[1]=='x' || s[1]=='X'){
			s += 2;
			for(;;){
				if(*s >= '0' && *s <= '9')
					n = n*16 + *s++ - '0';
				else if(*s >= 'a' && *s <= 'f')
					n = n*16 + *s++ - 'a' + 10;
				else if(*s >= 'A' && *s <= 'F')
					n = n*16 + *s++ - 'A' + 10;
				else
					break;
			}
		} else
			while(*s >= '0' && *s <= '7')
				n = n*8 + *s++ - '0';
	} else
		while(*s >= '0' && *s <= '9')
			n = n*10 + *s++ - '0';
	if(f)
		n = -n;
	return n;
}
