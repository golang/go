// Derived from Inferno utils/6c/peep.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/peep.c
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
#include "../gc/popt.h"
#include "opt.h"

static int	regzer(Addr *a);
static int	subprop(Flow*);
static int	copyprop(Flow*);
static int	copy1(Addr*, Addr*, Flow*, int);
static int	copyas(Addr*, Addr*);
static int	copyau(Addr*, Addr*);
static int	copysub(Addr*, Addr*, Addr*, int);
static int	copysub1(Prog*, Addr*, Addr*, int);
static int	copyau1(Prog *p, Addr *v);
static int	copyu(Prog *p, Addr *v, Addr *s);

static uint32	gactive;

void
peep(Prog *firstp)
{
	Graph *g;
	Flow *r, *r1;
	Prog *p, *p1;
	int t;

	g = flowstart(firstp, 0);
	if(g == nil)
		return;
	gactive = 0;

loop1:
	if(debug['P'] && debug['v'])
		dumpit("loop1", g->start, 0);

	t = 0;
	for(r=g->start; r!=nil; r=r->link) {
		p = r->prog;
		// TODO(austin) Handle smaller moves.  arm and amd64
		// distinguish between moves that moves that *must*
		// sign/zero extend and moves that don't care so they
		// can eliminate moves that don't care without
		// breaking moves that do care.  This might let us
		// simplify or remove the next peep loop, too.
		if(p->as == AMOVD || p->as == AFMOVD)
		if(regtyp(&p->to)) {
			// Try to eliminate reg->reg moves
			if(regtyp(&p->from))
			if(p->from.type == p->to.type) {
				if(copyprop(r)) {
					excise(r);
					t++;
				} else
				if(subprop(r) && copyprop(r)) {
					excise(r);
					t++;
				}
			}
			// Convert uses to $0 to uses of R0 and
			// propagate R0
			if(regzer(&p->from))
			if(p->to.type == TYPE_REG) {
				p->from.type = TYPE_REG;
				p->from.reg = REGZERO;
				if(copyprop(r)) {
					excise(r);
					t++;
				} else
				if(subprop(r) && copyprop(r)) {
					excise(r);
					t++;
				}
			}
		}
	}
	if(t)
		goto loop1;

	/*
	 * look for MOVB x,R; MOVB R,R (for small MOVs not handled above)
	 */
	for(r=g->start; r!=nil; r=r->link) {
		p = r->prog;
		switch(p->as) {
		default:
			continue;
		case AMOVH:
		case AMOVHZ:
		case AMOVB:
		case AMOVBZ:
		case AMOVW:
		case AMOVWZ:
			if(p->to.type != TYPE_REG)
				continue;
			break;
		}
		r1 = r->link;
		if(r1 == nil)
			continue;
		p1 = r1->prog;
		if(p1->as != p->as)
			continue;
		if(p1->from.type != TYPE_REG || p1->from.reg != p->to.reg)
			continue;
		if(p1->to.type != TYPE_REG || p1->to.reg != p->to.reg)
			continue;
		excise(r1);
	}

	if(debug['D'] > 1)
		goto ret;	/* allow following code improvement to be suppressed */

	/*
	 * look for OP x,y,R; CMP R, $0 -> OPCC x,y,R
	 * when OP can set condition codes correctly
	 */
	for(r=g->start; r!=nil; r=r->link) {
		p = r->prog;
		switch(p->as) {
		case ACMP:
		case ACMPW:		/* always safe? */
			if(!regzer(&p->to))
				continue;
			r1 = r->s1;
			if(r1 == nil)
				continue;
			switch(r1->prog->as) {
			default:
				continue;
			case ABCL:
			case ABC:
				/* the conditions can be complex and these are currently little used */
				continue;
			case ABEQ:
			case ABGE:
			case ABGT:
			case ABLE:
			case ABLT:
			case ABNE:
			case ABVC:
			case ABVS:
				break;
			}
			r1 = r;
			do
				r1 = uniqp(r1);
			while (r1 != nil && r1->prog->as == ANOP);
			if(r1 == nil)
				continue;
			p1 = r1->prog;
			if(p1->to.type != TYPE_REG || p1->to.reg != p->from.reg)
				continue;
			switch(p1->as) {
			case ASUB:
			case AADD:
			case AXOR:
			case AOR:
				/* irregular instructions */
				if(p1->from.type == TYPE_CONST || p1->from.type == TYPE_ADDR)
					continue;
				break;
			}
			switch(p1->as) {
			default:
				continue;
			case AMOVW:
			case AMOVD:
				if(p1->from.type != TYPE_REG)
					continue;
				continue;
			case AANDCC:
			case AANDNCC:
			case AORCC:
			case AORNCC:
			case AXORCC:
			case ASUBCC:
			case ASUBECC:
			case ASUBMECC:
			case ASUBZECC:
			case AADDCC:
			case AADDCCC:
			case AADDECC:
			case AADDMECC:
			case AADDZECC:
			case ARLWMICC:
			case ARLWNMCC:
			/* don't deal with floating point instructions for now */
/*
			case AFABS:
			case AFADD:
			case AFADDS:
			case AFCTIW:
			case AFCTIWZ:
			case AFDIV:
			case AFDIVS:
			case AFMADD:
			case AFMADDS:
			case AFMOVD:
			case AFMSUB:
			case AFMSUBS:
			case AFMUL:
			case AFMULS:
			case AFNABS:
			case AFNEG:
			case AFNMADD:
			case AFNMADDS:
			case AFNMSUB:
			case AFNMSUBS:
			case AFRSP:
			case AFSUB:
			case AFSUBS:
			case ACNTLZW:
			case AMTFSB0:
			case AMTFSB1:
*/
			case AADD:
			case AADDV:
			case AADDC:
			case AADDCV:
			case AADDME:
			case AADDMEV:
			case AADDE:
			case AADDEV:
			case AADDZE:
			case AADDZEV:
			case AAND:
			case AANDN:
			case ADIVW:
			case ADIVWV:
			case ADIVWU:
			case ADIVWUV:
			case ADIVD:
			case ADIVDV:
			case ADIVDU:
			case ADIVDUV:
			case AEQV:
			case AEXTSB:
			case AEXTSH:
			case AEXTSW:
			case AMULHW:
			case AMULHWU:
			case AMULLW:
			case AMULLWV:
			case AMULHD:
			case AMULHDU:
			case AMULLD:
			case AMULLDV:
			case ANAND:
			case ANEG:
			case ANEGV:
			case ANOR:
			case AOR:
			case AORN:
			case AREM:
			case AREMV:
			case AREMU:
			case AREMUV:
			case AREMD:
			case AREMDV:
			case AREMDU:
			case AREMDUV:
			case ARLWMI:
			case ARLWNM:
			case ASLW:
			case ASRAW:
			case ASRW:
			case ASLD:
			case ASRAD:
			case ASRD:
			case ASUB:
			case ASUBV:
			case ASUBC:
			case ASUBCV:
			case ASUBME:
			case ASUBMEV:
			case ASUBE:
			case ASUBEV:
			case ASUBZE:
			case ASUBZEV:
			case AXOR:
				t = variant2as(p1->as, as2variant(p1->as) | V_CC);
				break;
			}
			if(debug['D'])
				print("cmp %P; %P -> ", p1, p);
			p1->as = t;
			if(debug['D'])
				print("%P\n", p1);
			excise(r);
			continue;
		}
	}

ret:
	flowend(g);
}

void
excise(Flow *r)
{
	Prog *p;

	p = r->prog;
	if(debug['P'] && debug['v'])
		print("%P ===delete===\n", p);
	nopout(p);
	ostats.ndelmov++;
}

/*
 * regzer returns 1 if a's value is 0 (a is R0 or $0)
 */
static int
regzer(Addr *a)
{
	if(a->type == TYPE_CONST || a->type == TYPE_ADDR)
		if(a->sym == nil && a->reg == 0)
			if(a->offset == 0)
				return 1;
	if(a->type == TYPE_REG)
		if(a->reg == REGZERO)
			return 1;
	return 0;
}

int
regtyp(Adr *a)
{
	// TODO(rsc): Floating point register exclusions?
	return a->type == TYPE_REG && REG_R0 <= a->reg && a->reg <= REG_F31 && a->reg != REGZERO;
}

/*
 * the idea is to substitute
 * one register for another
 * from one MOV to another
 *	MOV	a, R1
 *	ADD	b, R1	/ no use of R2
 *	MOV	R1, R2
 * would be converted to
 *	MOV	a, R2
 *	ADD	b, R2
 *	MOV	R2, R1
 * hopefully, then the former or latter MOV
 * will be eliminated by copy propagation.
 *
 * r0 (the argument, not the register) is the MOV at the end of the
 * above sequences.  This returns 1 if it modified any instructions.
 */
static int
subprop(Flow *r0)
{
	Prog *p;
	Addr *v1, *v2;
	Flow *r;
	int t;
	ProgInfo info;

	p = r0->prog;
	v1 = &p->from;
	if(!regtyp(v1))
		return 0;
	v2 = &p->to;
	if(!regtyp(v2))
		return 0;
	for(r=uniqp(r0); r!=nil; r=uniqp(r)) {
		if(uniqs(r) == nil)
			break;
		p = r->prog;
		if(p->as == AVARDEF || p->as == AVARKILL)
			continue;
		proginfo(&info, p);
		if(info.flags & Call)
			return 0;

		if((info.flags & (RightRead|RightWrite)) == RightWrite) {
			if(p->to.type == v1->type)
			if(p->to.reg == v1->reg)
				goto gotit;
		}

		if(copyau(&p->from, v2) ||
		   copyau1(p, v2) ||
		   copyau(&p->to, v2))
			break;
		if(copysub(&p->from, v1, v2, 0) ||
		   copysub1(p, v1, v2, 0) ||
		   copysub(&p->to, v1, v2, 0))
			break;
	}
	return 0;

gotit:
	copysub(&p->to, v1, v2, 1);
	if(debug['P']) {
		print("gotit: %D->%D\n%P", v1, v2, r->prog);
		if(p->from.type == v2->type)
			print(" excise");
		print("\n");
	}
	for(r=uniqs(r); r!=r0; r=uniqs(r)) {
		p = r->prog;
		copysub(&p->from, v1, v2, 1);
		copysub1(p, v1, v2, 1);
		copysub(&p->to, v1, v2, 1);
		if(debug['P'])
			print("%P\n", r->prog);
	}
	t = v1->reg;
	v1->reg = v2->reg;
	v2->reg = t;
	if(debug['P'])
		print("%P last\n", r->prog);
	return 1;
}

/*
 * The idea is to remove redundant copies.
 *	v1->v2	F=0
 *	(use v2	s/v2/v1/)*
 *	set v1	F=1
 *	use v2	return fail (v1->v2 move must remain)
 *	-----------------
 *	v1->v2	F=0
 *	(use v2	s/v2/v1/)*
 *	set v1	F=1
 *	set v2	return success (caller can remove v1->v2 move)
 */
static int
copyprop(Flow *r0)
{
	Prog *p;
	Addr *v1, *v2;

	p = r0->prog;
	v1 = &p->from;
	v2 = &p->to;
	if(copyas(v1, v2)) {
		if(debug['P'])
			print("eliminating self-move\n", r0->prog);
		return 1;
	}
	gactive++;
	if(debug['P'])
		print("trying to eliminate %D->%D move from:\n%P\n", v1, v2, r0->prog);
	return copy1(v1, v2, r0->s1, 0);
}

// copy1 replaces uses of v2 with v1 starting at r and returns 1 if
// all uses were rewritten.
static int
copy1(Addr *v1, Addr *v2, Flow *r, int f)
{
	int t;
	Prog *p;

	if(r->active == gactive) {
		if(debug['P'])
			print("act set; return 1\n");
		return 1;
	}
	r->active = gactive;
	if(debug['P'])
		print("copy1 replace %D with %D f=%d\n", v2, v1, f);
	for(; r != nil; r = r->s1) {
		p = r->prog;
		if(debug['P'])
			print("%P", p);
		if(!f && uniqp(r) == nil) {
			// Multiple predecessors; conservatively
			// assume v1 was set on other path
			f = 1;
			if(debug['P'])
				print("; merge; f=%d", f);
		}
		t = copyu(p, v2, nil);
		switch(t) {
		case 2:	/* rar, can't split */
			if(debug['P'])
				print("; %D rar; return 0\n", v2);
			return 0;

		case 3:	/* set */
			if(debug['P'])
				print("; %D set; return 1\n", v2);
			return 1;

		case 1:	/* used, substitute */
		case 4:	/* use and set */
			if(f) {
				if(!debug['P'])
					return 0;
				if(t == 4)
					print("; %D used+set and f=%d; return 0\n", v2, f);
				else
					print("; %D used and f=%d; return 0\n", v2, f);
				return 0;
			}
			if(copyu(p, v2, v1)) {
				if(debug['P'])
					print("; sub fail; return 0\n");
				return 0;
			}
			if(debug['P'])
				print("; sub %D->%D\n => %P", v2, v1, p);
			if(t == 4) {
				if(debug['P'])
					print("; %D used+set; return 1\n", v2);
				return 1;
			}
			break;
		}
		if(!f) {
			t = copyu(p, v1, nil);
			if(!f && (t == 2 || t == 3 || t == 4)) {
				f = 1;
				if(debug['P'])
					print("; %D set and !f; f=%d", v1, f);
			}
		}
		if(debug['P'])
			print("\n");
		if(r->s2)
			if(!copy1(v1, v2, r->s2, f))
				return 0;
	}
	return 1;
}

// If s==nil, copyu returns the set/use of v in p; otherwise, it
// modifies p to replace reads of v with reads of s and returns 0 for
// success or non-zero for failure.
//
// If s==nil, copy returns one of the following values:
// 	1 if v only used
//	2 if v is set and used in one address (read-alter-rewrite;
// 	  can't substitute)
//	3 if v is only set
//	4 if v is set in one address and used in another (so addresses
// 	  can be rewritten independently)
//	0 otherwise (not touched)
static int
copyu(Prog *p, Addr *v, Addr *s)
{
	if(p->from3.type != TYPE_NONE)
		// 9g never generates a from3
		print("copyu: from3 (%D) not implemented\n", &p->from3);

	switch(p->as) {

	default:
		print("copyu: can't find %A\n", p->as);
		return 2;

	case ANOP:	/* read p->from, write p->to */
	case AMOVH:
	case AMOVHZ:
	case AMOVB:
	case AMOVBZ:
	case AMOVW:
	case AMOVWZ:
	case AMOVD:

	case ANEG:
	case ANEGCC:
	case AADDME:
	case AADDMECC:
	case AADDZE:
	case AADDZECC:
	case ASUBME:
	case ASUBMECC:
	case ASUBZE:
	case ASUBZECC:

	case AFCTIW:
	case AFCTIWZ:
	case AFCTID:
	case AFCTIDZ:
	case AFCFID:
	case AFCFIDCC:
	case AFMOVS:
	case AFMOVD:
	case AFRSP:
	case AFNEG:
	case AFNEGCC:
		if(s != nil) {
			if(copysub(&p->from, v, s, 1))
				return 1;
			// Update only indirect uses of v in p->to
			if(!copyas(&p->to, v))
				if(copysub(&p->to, v, s, 1))
					return 1;
			return 0;
		}
		if(copyas(&p->to, v)) {
			// Fix up implicit from
			if(p->from.type == TYPE_NONE)
				p->from = p->to;
			if(copyau(&p->from, v))
				return 4;
			return 3;
		}
		if(copyau(&p->from, v))
			return 1;
		if(copyau(&p->to, v))
			// p->to only indirectly uses v
			return 1;
		return 0;

	case AMOVBU:	/* rar p->from, write p->to or read p->from, rar p->to */
	case AMOVBZU:
	case AMOVHU:
	case AMOVHZU:
	case AMOVWZU:
	case AMOVDU:
		if(p->from.type == TYPE_MEM) {
			if(copyas(&p->from, v))
				// No s!=nil check; need to fail
				// anyway in that case
				return 2;
			if(s != nil) {
				if(copysub(&p->to, v, s, 1))
					return 1;
				return 0;
			}
			if(copyas(&p->to, v))
				return 3;
		} else if (p->to.type == TYPE_MEM) {
			if(copyas(&p->to, v))
				return 2;
			if(s != nil) {
				if(copysub(&p->from, v, s, 1))
					return 1;
				return 0;
			}
			if(copyau(&p->from, v))
				return 1;
		} else {
			print("copyu: bad %P\n", p);
		}
		return 0;

	case ARLWMI:	/* read p->from, read p->reg, rar p->to */
	case ARLWMICC:
		if(copyas(&p->to, v))
			return 2;
		/* fall through */

	case AADD:	/* read p->from, read p->reg, write p->to */
	case AADDC:
	case AADDE:
	case ASUB:
	case ASLW:
	case ASRW:
	case ASRAW:
	case ASLD:
	case ASRD:
	case ASRAD:
	case AOR:
	case AORCC:
	case AORN:
	case AORNCC:
	case AAND:
	case AANDCC:
	case AANDN:
	case AANDNCC:
	case ANAND:
	case ANANDCC:
	case ANOR:
	case ANORCC:
	case AXOR:
	case AMULHW:
	case AMULHWU:
	case AMULLW:
	case AMULLD:
	case ADIVW:
	case ADIVD:
	case ADIVWU:
	case ADIVDU:
	case AREM:
	case AREMU:
	case AREMD:
	case AREMDU:
	case ARLWNM:
	case ARLWNMCC:

	case AFADDS:
	case AFADD:
	case AFSUBS:
	case AFSUB:
	case AFMULS:
	case AFMUL:
	case AFDIVS:
	case AFDIV:
		if(s != nil) {
			if(copysub(&p->from, v, s, 1))
				return 1;
			if(copysub1(p, v, s, 1))
				return 1;
			// Update only indirect uses of v in p->to
			if(!copyas(&p->to, v))
				if(copysub(&p->to, v, s, 1))
					return 1;
			return 0;
		}
		if(copyas(&p->to, v)) {
			if(p->reg == 0)
				// Fix up implicit reg (e.g., ADD
				// R3,R4 -> ADD R3,R4,R4) so we can
				// update reg and to separately.
				p->reg = p->to.reg;
			if(copyau(&p->from, v))
				return 4;
			if(copyau1(p, v))
				return 4;
			return 3;
		}
		if(copyau(&p->from, v))
			return 1;
		if(copyau1(p, v))
			return 1;
		if(copyau(&p->to, v))
			return 1;
		return 0;

	case ABEQ:
	case ABGT:
	case ABGE:
	case ABLT:
	case ABLE:
	case ABNE:
	case ABVC:
	case ABVS:
		return 0;

	case ACHECKNIL:	/* read p->from */
	case ACMP:	/* read p->from, read p->to */
	case ACMPU:
	case ACMPW:
	case ACMPWU:
	case AFCMPO:
	case AFCMPU:
		if(s != nil) {
			if(copysub(&p->from, v, s, 1))
				return 1;
			return copysub(&p->to, v, s, 1);
		}
		if(copyau(&p->from, v))
			return 1;
		if(copyau(&p->to, v))
			return 1;
		return 0;

	case ABR:	/* read p->to */
		// 9g never generates a branch to a GPR (this isn't
		// even a normal instruction; liblink turns it in to a
		// mov and a branch).
		if(s != nil) {
			if(copysub(&p->to, v, s, 1))
				return 1;
			return 0;
		}
		if(copyau(&p->to, v))
			return 1;
		return 0;

	case ARETURN:	/* funny */
		if(s != nil)
			return 0;
		// All registers die at this point, so claim
		// everything is set (and not used).
		return 3;

	case ABL:	/* funny */
		if(v->type == TYPE_REG) {
			// TODO(rsc): REG_R0 and REG_F0 used to be
			// (when register numbers started at 0) exregoffset and exfregoffset,
			// which are unset entirely. 
			// It's strange that this handles R0 and F0 differently from the other
			// registers. Possible failure to optimize?
			if(REG_R0 < v->reg && v->reg <= REGEXT)
				return 2;
			if(v->reg == REGARG)
				return 2;
			if(REG_F0 < v->reg && v->reg <= FREGEXT)
				return 2;
		}
		if(p->from.type == TYPE_REG && v->type == TYPE_REG && p->from.reg == v->reg)
			return 2;

		if(s != nil) {
			if(copysub(&p->to, v, s, 1))
				return 1;
			return 0;
		}
		if(copyau(&p->to, v))
			return 4;
		return 3;

	case ADUFFZERO:
		// R0 is zero, used by DUFFZERO, cannot be substituted.
		// R3 is ptr to memory, used and set, cannot be substituted.
		if(v->type == TYPE_REG) {
			if(v->reg == 0)
				return 1;
			if(v->reg == 3)
				return 2;
		}
		return 0;

	case ADUFFCOPY:
		// R3, R4 are ptr to src, dst, used and set, cannot be substituted.
		// R5 is scratch, set by DUFFCOPY, cannot be substituted.
		if(v->type == TYPE_REG) {
			if(v->reg == 3 || v->reg == 4)
				return 2;
			if(v->reg == 5)
				return 3;
		}
		return 0;

	case ATEXT:	/* funny */
		if(v->type == TYPE_REG)
			if(v->reg == REGARG)
				return 3;
		return 0;

	case APCDATA:
	case AFUNCDATA:
	case AVARDEF:
	case AVARKILL:
		return 0;
	}
}

// copyas returns 1 if a and v address the same register.
//
// If a is the from operand, this means this operation reads the
// register in v.  If a is the to operand, this means this operation
// writes the register in v.
static int
copyas(Addr *a, Addr *v)
{
	if(regtyp(v))
		if(a->type == v->type)
		if(a->reg == v->reg)
			return 1;
	return 0;
}

// copyau returns 1 if a either directly or indirectly addresses the
// same register as v.
//
// If a is the from operand, this means this operation reads the
// register in v.  If a is the to operand, this means the operation
// either reads or writes the register in v (if !copyas(a, v), then
// the operation reads the register in v).
static int
copyau(Addr *a, Addr *v)
{
	if(copyas(a, v))
		return 1;
	if(v->type == TYPE_REG)
		if(a->type == TYPE_MEM || (a->type == TYPE_ADDR && a->reg != 0))
			if(v->reg == a->reg)
				return 1;
	return 0;
}

// copyau1 returns 1 if p->reg references the same register as v and v
// is a direct reference.
static int
copyau1(Prog *p, Addr *v)
{
	if(regtyp(v) && v->reg != 0)
		if(p->reg == v->reg)
			return 1;
	return 0;
}

// copysub replaces v with s in a if f!=0 or indicates it if could if f==0.
// Returns 1 on failure to substitute (it always succeeds on ppc64).
static int
copysub(Addr *a, Addr *v, Addr *s, int f)
{
	if(f)
	if(copyau(a, v))
		a->reg = s->reg;
	return 0;
}

// copysub1 replaces v with s in p1->reg if f!=0 or indicates if it could if f==0.
// Returns 1 on failure to substitute (it always succeeds on ppc64).
static int
copysub1(Prog *p1, Addr *v, Addr *s, int f)
{
	if(f)
	if(copyau1(p1, v))
		p1->reg = s->reg;
	return 0;
}

int
sameaddr(Addr *a, Addr *v)
{
	if(a->type != v->type)
		return 0;
	if(regtyp(v) && a->reg == v->reg)
		return 1;
	if(v->type == NAME_AUTO || v->type == NAME_PARAM)
		if(v->offset == a->offset)
			return 1;
	return 0;
}

int
smallindir(Addr *a, Addr *reg)
{
	return reg->type == TYPE_REG && a->type == TYPE_MEM &&
		a->reg == reg->reg &&
		0 <= a->offset && a->offset < 4096;
}

int
stackaddr(Addr *a)
{
	return a->type == TYPE_REG && a->reg == REGSP;
}
