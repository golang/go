// Inferno utils/5l/noop.c
// http://code.google.com/p/inferno-os/source/browse/utils/5l/noop.c
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

// Code transformations.

#include	"l.h"
#include	"../ld/lib.h"

// see ../../runtime/proc.c:/StackGuard
enum
{
	StackBig = 4096,
	StackSmall = 128,
};

static	Sym*	sym_div;
static	Sym*	sym_divu;
static	Sym*	sym_mod;
static	Sym*	sym_modu;

static void
linkcase(Prog *casep)
{
	Prog *p;

	for(p = casep; p != P; p = p->link){
		if(p->as == ABCASE) {
			for(; p != P && p->as == ABCASE; p = p->link)
				p->pcrel = casep;
			break;
		}
	}
}

void
noops(void)
{
	Prog *p, *q, *q1;
	int o;
	Prog *pmorestack;
	Sym *symmorestack;

	/*
	 * find leaf subroutines
	 * strip NOPs
	 * expand RET
	 * expand BECOME pseudo
	 */

	if(debug['v'])
		Bprint(&bso, "%5.2f noops\n", cputime());
	Bflush(&bso);

	symmorestack = lookup("runtime.morestack", 0);
	if(symmorestack->type != STEXT) {
		diag("runtime·morestack not defined");
		errorexit();
	}
	pmorestack = symmorestack->text;
	pmorestack->reg |= NOSPLIT;

	q = P;
	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		for(p = cursym->text; p != P; p = p->link) {
			switch(p->as) {
			case ACASE:
				if(flag_shared)
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
				if(prog_div == P)
					initdiv();
				cursym->text->mark &= ~LEAF;
				continue;
	
			case ANOP:
				q1 = p->link;
				q->link = q1;		/* q is non-nop */
				if(q1 != P)
					q1->mark |= p->mark;
				continue;
	
			case ABL:
			case ABX:
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
				q1 = p->cond;
				if(q1 != P) {
					while(q1->as == ANOP) {
						q1 = q1->link;
						p->cond = q1;
					}
				}
				break;
			}
			q = p;
		}
	}

	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		for(p = cursym->text; p != P; p = p->link) {
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
					if(debug['v'])
						Bprint(&bso, "save suppressed in: %s\n",
							cursym->name);
					Bflush(&bso);
					cursym->text->mark |= LEAF;
				}
				if(cursym->text->mark & LEAF) {
					cursym->leaf = 1;
					if(!autosize)
						break;
				}
	
				if(p->reg & NOSPLIT) {
					q1 = prg();
					q1->as = AMOVW;
					q1->scond |= C_WBIT;
					q1->line = p->line;
					q1->from.type = D_REG;
					q1->from.reg = REGLINK;
					q1->to.type = D_OREG;
					q1->to.offset = -autosize;
					q1->to.reg = REGSP;
					q1->spadj = autosize;
					q1->link = p->link;
					p->link = q1;
				} else if (autosize < StackBig) {
					// split stack check for small functions
					// MOVW			g_stackguard(g), R1
					// CMP			R1, $-autosize(SP)
					// MOVW.LO		$autosize, R1
					// MOVW.LO		$args, R2
					// MOVW.LO		R14, R3
					// BL.LO			runtime.morestack(SB) // modifies LR
					// MOVW.W		R14,$-autosize(SP)
	
					// TODO(kaib): add more trampolines
					// TODO(kaib): put stackguard in register
					// TODO(kaib): add support for -K and underflow detection

					// MOVW			g_stackguard(g), R1
					p = appendp(p);
					p->as = AMOVW;
					p->from.type = D_OREG;
					p->from.reg = REGG;
					p->to.type = D_REG;
					p->to.reg = 1;
					
					if(autosize < StackSmall) {	
						// CMP			R1, SP
						p = appendp(p);
						p->as = ACMP;
						p->from.type = D_REG;
						p->from.reg = 1;
						p->reg = REGSP;
					} else {
						// MOVW		$-autosize(SP), R2
						// CMP	R1, R2
						p = appendp(p);
						p->as = AMOVW;
						p->from.type = D_CONST;
						p->from.reg = REGSP;
						p->from.offset = -autosize;
						p->to.type = D_REG;
						p->to.reg = 2;
						
						p = appendp(p);
						p->as = ACMP;
						p->from.type = D_REG;
						p->from.reg = 1;
						p->reg = 2;
					}

					// MOVW.LO		$autosize, R1
					p = appendp(p);
					p->as = AMOVW;
					p->scond = C_SCOND_LO;
					p->from.type = D_CONST;
					p->from.offset = autosize;
					p->to.type = D_REG;
					p->to.reg = 1;
	
					// MOVW.LO		$args, R2
					p = appendp(p);
					p->as = AMOVW;
					p->scond = C_SCOND_LO;
					p->from.type = D_CONST;
					p->from.offset = (cursym->text->to.offset2 + 3) & ~3;
					p->to.type = D_REG;
					p->to.reg = 2;
	
					// MOVW.LO	R14, R3
					p = appendp(p);
					p->as = AMOVW;
					p->scond = C_SCOND_LO;
					p->from.type = D_REG;
					p->from.reg = REGLINK;
					p->to.type = D_REG;
					p->to.reg = 3;
	
					// BL.LO		runtime.morestack(SB) // modifies LR
					p = appendp(p);
					p->as = ABL;
					p->scond = C_SCOND_LO;
					p->to.type = D_BRANCH;
					p->to.sym = symmorestack;
					p->cond = pmorestack;
	
					// MOVW.W		R14,$-autosize(SP)
					p = appendp(p);
					p->as = AMOVW;
					p->scond |= C_WBIT;
					p->from.type = D_REG;
					p->from.reg = REGLINK;
					p->to.type = D_OREG;
					p->to.offset = -autosize;
					p->to.reg = REGSP;
					p->spadj = autosize;
				} else { // > StackBig
					// MOVW		$autosize, R1
					// MOVW		$args, R2
					// MOVW		R14, R3
					// BL			runtime.morestack(SB) // modifies LR
					// MOVW.W		R14,$-autosize(SP)
	
					// MOVW		$autosize, R1
					p = appendp(p);
					p->as = AMOVW;
					p->from.type = D_CONST;
					p->from.offset = autosize;
					p->to.type = D_REG;
					p->to.reg = 1;
	
					// MOVW		$args, R2
					// also need to store the extra 4 bytes.
					p = appendp(p);
					p->as = AMOVW;
					p->from.type = D_CONST;
					p->from.offset = (cursym->text->to.offset2 + 3) & ~3;
					p->to.type = D_REG;
					p->to.reg = 2;
	
					// MOVW	R14, R3
					p = appendp(p);
					p->as = AMOVW;
					p->from.type = D_REG;
					p->from.reg = REGLINK;
					p->to.type = D_REG;
					p->to.reg = 3;
	
					// BL		runtime.morestack(SB) // modifies LR
					p = appendp(p);
					p->as = ABL;
					p->to.type = D_BRANCH;
					p->to.sym = symmorestack;
					p->cond = pmorestack;
	
					// MOVW.W		R14,$-autosize(SP)
					p = appendp(p);
					p->as = AMOVW;
					p->scond |= C_WBIT;
					p->from.type = D_REG;
					p->from.reg = REGLINK;
					p->to.type = D_OREG;
					p->to.offset = -autosize;
					p->to.reg = REGSP;
					p->spadj = autosize;
				}
				break;
	
			case ARET:
				nocache(p);
				if(cursym->text->mark & LEAF) {
					if(!autosize) {
						p->as = AB;
						p->from = zprg.from;
						p->to.type = D_OREG;
						p->to.offset = 0;
						p->to.reg = REGLINK;
						break;
					}
				}
				p->as = AMOVW;
				p->scond |= C_PBIT;
				p->from.type = D_OREG;
				p->from.offset = autosize;
				p->from.reg = REGSP;
				p->to.type = D_REG;
				p->to.reg = REGPC;
				// If there are instructions following
				// this ARET, they come from a branch
				// with the same stackframe, so no spadj.
				break;
	
			case AADD:
				if(p->from.type == D_CONST && p->from.reg == NREG && p->to.type == D_REG && p->to.reg == REGSP)
					p->spadj = -p->from.offset;
				break;

			case ASUB:
				if(p->from.type == D_CONST && p->from.reg == NREG && p->to.type == D_REG && p->to.reg == REGSP)
					p->spadj = p->from.offset;
				break;

			case ADIV:
			case ADIVU:
			case AMOD:
			case AMODU:
				if(debug['M'])
					break;
				if(p->from.type != D_REG)
					break;
				if(p->to.type != D_REG)
					break;
				q1 = p;
	
				/* MOV a,4(SP) */
				p = appendp(p);
				p->as = AMOVW;
				p->line = q1->line;
				p->from.type = D_REG;
				p->from.reg = q1->from.reg;
				p->to.type = D_OREG;
				p->to.reg = REGSP;
				p->to.offset = 4;
	
				/* MOV b,REGTMP */
				p = appendp(p);
				p->as = AMOVW;
				p->line = q1->line;
				p->from.type = D_REG;
				p->from.reg = q1->reg;
				if(q1->reg == NREG)
					p->from.reg = q1->to.reg;
				p->to.type = D_REG;
				p->to.reg = REGTMP;
				p->to.offset = 0;
	
				/* CALL appropriate */
				p = appendp(p);
				p->as = ABL;
				p->line = q1->line;
				p->to.type = D_BRANCH;
				p->cond = p;
				switch(o) {
				case ADIV:
					p->cond = prog_div;
					p->to.sym = sym_div;
					break;
				case ADIVU:
					p->cond = prog_divu;
					p->to.sym = sym_divu;
					break;
				case AMOD:
					p->cond = prog_mod;
					p->to.sym = sym_mod;
					break;
				case AMODU:
					p->cond = prog_modu;
					p->to.sym = sym_modu;
					break;
				}
	
				/* MOV REGTMP, b */
				p = appendp(p);
				p->as = AMOVW;
				p->line = q1->line;
				p->from.type = D_REG;
				p->from.reg = REGTMP;
				p->from.offset = 0;
				p->to.type = D_REG;
				p->to.reg = q1->to.reg;
	
				/* ADD $8,SP */
				p = appendp(p);
				p->as = AADD;
				p->line = q1->line;
				p->from.type = D_CONST;
				p->from.reg = NREG;
				p->from.offset = 8;
				p->reg = NREG;
				p->to.type = D_REG;
				p->to.reg = REGSP;
				p->spadj = -8;
	
				/* SUB $8,SP */
				q1->as = ASUB;
				q1->from.type = D_CONST;
				q1->from.offset = 8;
				q1->from.reg = NREG;
				q1->reg = NREG;
				q1->to.type = D_REG;
				q1->to.reg = REGSP;
				q1->spadj = 8;
	
				break;
			case AMOVW:
				if((p->scond & C_WBIT) && p->to.type == D_OREG && p->to.reg == REGSP)
					p->spadj = -p->to.offset;
				if((p->scond & C_PBIT) && p->from.type == D_OREG && p->from.reg == REGSP && p->to.reg != REGPC)
					p->spadj = -p->from.offset;
				if(p->from.type == D_CONST && p->from.reg == REGSP && p->to.type == D_REG && p->to.reg == REGSP)
					p->spadj = -p->from.offset;
				break;
			}
		}
	}
}

static void
sigdiv(char *n)
{
	Sym *s;

	s = lookup(n, 0);
	if(s->type == STEXT)
		if(s->sig == 0)
			s->sig = SIGNINTERN;
}

void
divsig(void)
{
	sigdiv("_div");
	sigdiv("_divu");
	sigdiv("_mod");
	sigdiv("_modu");
}

void
initdiv(void)
{
	Sym *s2, *s3, *s4, *s5;

	if(prog_div != P)
		return;
	sym_div = s2 = lookup("_div", 0);
	sym_divu = s3 = lookup("_divu", 0);
	sym_mod = s4 = lookup("_mod", 0);
	sym_modu = s5 = lookup("_modu", 0);
	prog_div = s2->text;
	prog_divu = s3->text;
	prog_mod = s4->text;
	prog_modu = s5->text;
	if(prog_div == P) {
		diag("undefined: %s", s2->name);
		prog_div = cursym->text;
	}
	if(prog_divu == P) {
		diag("undefined: %s", s3->name);
		prog_divu = cursym->text;
	}
	if(prog_mod == P) {
		diag("undefined: %s", s4->name);
		prog_mod = cursym->text;
	}
	if(prog_modu == P) {
		diag("undefined: %s", s5->name);
		prog_modu = cursym->text;
	}
}

void
nocache(Prog *p)
{
	p->optab = 0;
	p->from.class = 0;
	p->to.class = 0;
}
