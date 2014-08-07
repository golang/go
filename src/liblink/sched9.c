// cmd/9l/sched.c from Vita Nuova.
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

// +build ignore

#include	"l.h"

enum
{
	E_ICC	= 1<<0,
	E_FCC	= 1<<1,
	E_MEM	= 1<<2,
	E_MEMSP	= 1<<3,	/* uses offset and size */
	E_MEMSB	= 1<<4,	/* uses offset and size */
	E_LR	= 1<<5,
	E_CR = 1<<6,
	E_CTR = 1<<7,
	E_XER = 1<<8,

	E_CR0 = 0xF<<0,
	E_CR1 = 0xF<<4,

	ANYMEM	= E_MEM|E_MEMSP|E_MEMSB,
	ALL	= ~0,
};

typedef	struct	Sch	Sch;
typedef	struct	Dep	Dep;

struct	Dep
{
	ulong	ireg;
	ulong	freg;
	ulong	cc;
	ulong	cr;
};
struct	Sch
{
	Prog	p;
	Dep	set;
	Dep	used;
	long	soffset;
	char	size;
	char	comp;
};

void	regused(Sch*, Prog*);
int	depend(Sch*, Sch*);
int	conflict(Sch*, Sch*);
int	offoverlap(Sch*, Sch*);
void	dumpbits(Sch*, Dep*);

void
sched(Prog *p0, Prog *pe)
{
	Prog *p, *q;
	Sch sch[NSCHED], *s, *t, *u, *se, stmp;

	if(!debug['Q'])
		return;
	/*
	 * build side structure
	 */
	s = sch;
	for(p=p0;; p=p->link) {
		memset(s, 0, sizeof(*s));
		s->p = *p;
		regused(s, p);
		if(debug['X']) {
			Bprint(&bso, "%P\tset", &s->p);
			dumpbits(s, &s->set);
			Bprint(&bso, "; used");
			dumpbits(s, &s->used);
			if(s->comp)
				Bprint(&bso, "; compound");
			if(s->p.mark & LOAD)
				Bprint(&bso, "; load");
			if(s->p.mark & BRANCH)
				Bprint(&bso, "; branch");
			if(s->p.mark & FCMP)
				Bprint(&bso, "; fcmp");
			Bprint(&bso, "\n");
		}
		s++;
		if(p == pe)
			break;
	}
	se = s;

	for(s=se-1; s>=sch; s--) {

		/*
		 * load delay. interlocked.
		 */
		if(s->p.mark & LOAD) {
			if(s >= se-1)
				continue;
			if(!conflict(s, (s+1)))
				continue;
			/*
			 * s is load, s+1 is immediate use of result
			 * t is the trial instruction to insert between s and s+1
			 */
			for(t=s-1; t>=sch; t--) {
				if(t->p.mark & BRANCH)
					goto no2;
				if(t->p.mark & FCMP)
					if((s+1)->p.mark & BRANCH)
						goto no2;
				if(t->p.mark & LOAD)
					if(conflict(t, (s+1)))
						goto no2;
				for(u=t+1; u<=s; u++)
					if(depend(u, t))
						goto no2;
				goto out2;
			no2:;
			}
			if(debug['X'])
				Bprint(&bso, "?l%P\n", &s->p);
			continue;
		out2:
			if(debug['X']) {
				Bprint(&bso, "!l%P\n", &t->p);
				Bprint(&bso, "%P\n", &s->p);
			}
			stmp = *t;
			memmove(t, t+1, (uchar*)s - (uchar*)t);
			*s = stmp;
			s--;
			continue;
		}

		/*
		 * fop2 delay.
		 */
		if(s->p.mark & FCMP) {
			if(s >= se-1)
				continue;
			if(!((s+1)->p.mark & BRANCH))
				continue;
			/* t is the trial instruction to use */
			for(t=s-1; t>=sch; t--) {
				for(u=t+1; u<=s; u++)
					if(depend(u, t))
						goto no3;
				goto out3;
			no3:;
			}
			if(debug['X'])
				Bprint(&bso, "?f%P\n", &s->p);
			continue;
		out3:
			if(debug['X']) {
				Bprint(&bso, "!f%P\n", &t->p);
				Bprint(&bso, "%P\n", &s->p);
			}
			stmp = *t;
			memmove(t, t+1, (uchar*)s - (uchar*)t);
			*s = stmp;
			s--;
			continue;
		}
	}

	/*
	 * put it all back
	 */
	for(s=sch, p=p0; s<se; s++, p=q) {
		q = p->link;
		if(q != s->p.link) {
			*p = s->p;
			p->link = q;
		}
	}
	if(debug['X'])
		Bprint(&bso, "\n");
}

void
regused(Sch *s, Prog *realp)
{
	int c, ar, ad, ld, sz, nr, upd;
	ulong m;
	Prog *p;

	p = &s->p;
	s->comp = compound(p);
	if(s->comp) {
		s->set.ireg |= 1<<REGTMP;
		s->used.ireg |= 1<<REGTMP;
	}
	ar = 0;		/* dest is really reference */
	ad = 0;		/* source/dest is really address */
	ld = 0;		/* opcode is load instruction */
	sz = 32*4;		/* size of load/store for overlap computation */
	nr = 0;	/* source/dest is not really reg */
	upd = 0;	/* move with update; changes reg */

/*
 * flags based on opcode
 */
	switch(p->as) {
	case ATEXT:
		curtext = realp;
		autosize = p->to.offset + 8;
		ad = 1;
		break;
	case ABL:
		s->set.cc |= E_LR;
		ar = 1;
		ad = 1;
		break;
	case ABR:
		ar = 1;
		ad = 1;
		break;
	case ACMP:
	case ACMPU:
	case ACMPW:
	case ACMPWU:
		s->set.cc |= E_ICC;
		if(p->reg == 0)
			s->set.cr |= E_CR0;
		else
			s->set.cr |= (0xF<<((p->reg&7)*4));
		ar = 1;
		break;
	case AFCMPO:
	case AFCMPU:
		s->set.cc |= E_FCC;
		if(p->reg == 0)
			s->set.cr |= E_CR0;
		else
			s->set.cr |= (0xF<<((p->reg&7)*4));
		ar = 1;
		break;
	case ACRAND:
	case ACRANDN:
	case ACREQV:
	case ACRNAND:
	case ACRNOR:
	case ACROR:
	case ACRORN:
	case ACRXOR:
		s->used.cr |= 1<<p->from.reg;
		s->set.cr |= 1<<p->to.reg;
		nr = 1;
		break;
	case ABCL:	/* tricky */
		s->used.cc |= E_FCC|E_ICC;
		s->used.cr = ALL;
		s->set.cc |= E_LR;
		ar = 1;
		break;
	case ABC:		/* tricky */
		s->used.cc |= E_FCC|E_ICC;
		s->used.cr = ALL;
		ar = 1;
		break;
	case ABEQ:
	case ABGE:
	case ABGT:
	case ABLE:
	case ABLT:
	case ABNE:
	case ABVC:
	case ABVS:
		s->used.cc |= E_ICC;
		s->used.cr |= E_CR0;
		ar = 1;
		break;
	case ALSW:
	case AMOVMW:
		/* could do better */
		sz = 32*4;
		ld = 1;
		break;
	case AMOVBU:
	case AMOVBZU:
		upd = 1;
		sz = 1;
		ld = 1;
		break;
	case AMOVB:
	case AMOVBZ:
		sz = 1;
		ld = 1;
		break;
	case AMOVHU:
	case AMOVHZU:
		upd = 1;
		sz = 2;
		ld = 1;
		break;
	case AMOVH:
	case AMOVHBR:
	case AMOVHZ:
		sz = 2;
		ld = 1;
		break;
	case AFMOVSU:
	case AMOVWU:
	case AMOVWZU:
		upd = 1;
		sz = 4;
		ld = 1;
		break;
	case AFMOVS:
	case AMOVW:
	case AMOVWZ:
	case AMOVWBR:
	case ALWAR:
		sz = 4;
		ld = 1;
		break;
	case AFMOVDU:
		upd = 1;
		sz = 8;
		ld = 1;
		break;
	case AFMOVD:
		sz = 8;
		ld = 1;
		break;
	case AFMOVDCC:
		sz = 8;
		ld = 1;
		s->set.cc |= E_FCC;
		s->set.cr |= E_CR1;
		break;
	case AMOVFL:
	case AMOVCRFS:
	case AMTFSB0:
	case AMTFSB0CC:
	case AMTFSB1:
	case AMTFSB1CC:
		s->set.ireg = ALL;
		s->set.freg = ALL;
		s->set.cc = ALL;
		s->set.cr = ALL;
		break;
	case AADDCC:
	case AADDVCC:
	case AADDCCC:
	case AADDCVCC:
	case AADDMECC:
	case AADDMEVCC:
	case AADDECC:
	case AADDEVCC:
	case AADDZECC:
	case AADDZEVCC:
	case AANDCC:
	case AANDNCC:
	case ACNTLZWCC:
	case ADIVWCC:
	case ADIVWVCC:
	case ADIVWUCC:
	case ADIVWUVCC:
	case AEQVCC:
	case AEXTSBCC:
	case AEXTSHCC:
	case AMULHWCC:
	case AMULHWUCC:
	case AMULLWCC:
	case AMULLWVCC:
	case ANANDCC:
	case ANEGCC:
	case ANEGVCC:
	case ANORCC:
	case AORCC:
	case AORNCC:
	case AREMCC:
	case AREMVCC:
	case AREMUCC:
	case AREMUVCC:
	case ARLWMICC:
	case ARLWNMCC:
	case ASLWCC:
	case ASRAWCC:
	case ASRWCC:
	case ASTWCCC:
	case ASUBCC:
	case ASUBVCC:
	case ASUBCCC:
	case ASUBCVCC:
	case ASUBMECC:
	case ASUBMEVCC:
	case ASUBECC:
	case ASUBEVCC:
	case ASUBZECC:
	case ASUBZEVCC:
	case AXORCC:
		s->set.cc |= E_ICC;
		s->set.cr |= E_CR0;
		break;
	case AFABSCC:
	case AFADDCC:
	case AFADDSCC:
	case AFCTIWCC:
	case AFCTIWZCC:
	case AFDIVCC:
	case AFDIVSCC:
	case AFMADDCC:
	case AFMADDSCC:
	case AFMSUBCC:
	case AFMSUBSCC:
	case AFMULCC:
	case AFMULSCC:
	case AFNABSCC:
	case AFNEGCC:
	case AFNMADDCC:
	case AFNMADDSCC:
	case AFNMSUBCC:
	case AFNMSUBSCC:
	case AFRSPCC:
	case AFSUBCC:
	case AFSUBSCC:
		s->set.cc |= E_FCC;
		s->set.cr |= E_CR1;
		break;
	}

/*
 * flags based on 'to' field
 */
	c = p->to.class;
	if(c == 0) {
		c = aclass(&p->to) + 1;
		p->to.class = c;
	}
	c--;
	switch(c) {
	default:
		print("unknown class %d %D\n", c, &p->to);

	case C_NONE:
	case C_ZCON:
	case C_SCON:
	case C_UCON:
	case C_LCON:
	case C_ADDCON:
	case C_ANDCON:
	case C_SBRA:
	case C_LBRA:
		break;
	case C_CREG:
		c = p->to.reg;
		if(c == NREG)
			s->set.cr = ALL;
		else
			s->set.cr |= (0xF << ((p->from.reg&7)*4));
		s->set.cc = ALL;
		break;
	case C_SPR:
	case C_FPSCR:
	case C_MSR:
	case C_XER:
		s->set.ireg = ALL;
		s->set.freg = ALL;
		s->set.cc = ALL;
		s->set.cr = ALL;
		break;
	case C_LR:
		s->set.cc |= E_LR;
		break;
	case C_CTR:
		s->set.cc |= E_CTR;
		break;
	case C_ZOREG:
	case C_SOREG:
	case C_LOREG:
		c = p->to.reg;
		s->used.ireg |= 1<<c;
		if(upd)
			s->set.ireg |= 1<<c;
		if(ad)
			break;
		s->size = sz;
		s->soffset = regoff(&p->to);

		m = ANYMEM;
		if(c == REGSB)
			m = E_MEMSB;
		if(c == REGSP)
			m = E_MEMSP;

		if(ar)
			s->used.cc |= m;
		else
			s->set.cc |= m;
		break;
	case C_SACON:
	case C_LACON:
		s->used.ireg |= 1<<REGSP;
		if(upd)
			s->set.ireg |= 1<<c;
		break;
	case C_SECON:
	case C_LECON:
		s->used.ireg |= 1<<REGSB;
		if(upd)
			s->set.ireg |= 1<<c;
		break;
	case C_REG:
		if(nr)
			break;
		if(ar)
			s->used.ireg |= 1<<p->to.reg;
		else
			s->set.ireg |= 1<<p->to.reg;
		break;
	case C_FREG:
		if(ar)
			s->used.freg |= 1<<p->to.reg;
		else
			s->set.freg |= 1<<p->to.reg;
		break;
	case C_SAUTO:
	case C_LAUTO:
		s->used.ireg |= 1<<REGSP;
		if(upd)
			s->set.ireg |= 1<<c;
		if(ad)
			break;
		s->size = sz;
		s->soffset = regoff(&p->to);

		if(ar)
			s->used.cc |= E_MEMSP;
		else
			s->set.cc |= E_MEMSP;
		break;
	case C_SEXT:
	case C_LEXT:
		s->used.ireg |= 1<<REGSB;
		if(upd)
			s->set.ireg |= 1<<c;
		if(ad)
			break;
		s->size = sz;
		s->soffset = regoff(&p->to);

		if(ar)
			s->used.cc |= E_MEMSB;
		else
			s->set.cc |= E_MEMSB;
		break;
	}

/*
 * flags based on 'from' field
 */
	c = p->from.class;
	if(c == 0) {
		c = aclass(&p->from) + 1;
		p->from.class = c;
	}
	c--;
	switch(c) {
	default:
		print("unknown class %d %D\n", c, &p->from);

	case C_NONE:
	case C_ZCON:
	case C_SCON:
	case C_UCON:
	case C_LCON:
	case C_ADDCON:
	case C_ANDCON:
	case C_SBRA:
	case C_LBRA:
		c = p->from.reg;
		if(c != NREG)
			s->used.ireg |= 1<<c;
		break;
	case C_CREG:
		c = p->from.reg;
		if(c == NREG)
			s->used.cr = ALL;
		else
			s->used.cr |= (0xF << ((p->from.reg&7)*4));
		s->used.cc = ALL;
		break;
	case C_SPR:
	case C_FPSCR:
	case C_MSR:
	case C_XER:
		s->set.ireg = ALL;
		s->set.freg = ALL;
		s->set.cc = ALL;
		s->set.cr = ALL;
		break;
	case C_LR:
		s->used.cc |= E_LR;
		break;
	case C_CTR:
		s->used.cc |= E_CTR;
		break;
	case C_ZOREG:
	case C_SOREG:
	case C_LOREG:
		c = p->from.reg;
		s->used.ireg |= 1<<c;
		if(ld)
			p->mark |= LOAD;
		if(ad)
			break;
		s->size = sz;
		s->soffset = regoff(&p->from);

		m = ANYMEM;
		if(c == REGSB)
			m = E_MEMSB;
		if(c == REGSP)
			m = E_MEMSP;

		s->used.cc |= m;
		break;
	case C_SACON:
	case C_LACON:
		s->used.ireg |= 1<<REGSP;
		break;
	case C_SECON:
	case C_LECON:
		s->used.ireg |= 1<<REGSB;
		break;
	case C_REG:
		if(nr)
			break;
		s->used.ireg |= 1<<p->from.reg;
		break;
	case C_FREG:
		s->used.freg |= 1<<p->from.reg;
		break;
	case C_SAUTO:
	case C_LAUTO:
		s->used.ireg |= 1<<REGSP;
		if(ld)
			p->mark |= LOAD;
		if(ad)
			break;
		s->size = sz;
		s->soffset = regoff(&p->from);

		s->used.cc |= E_MEMSP;
		break;
	case C_SEXT:
	case C_LEXT:
		s->used.ireg |= 1<<REGSB;
		if(ld)
			p->mark |= LOAD;
		if(ad)
			break;
		s->size = sz;
		s->soffset = regoff(&p->from);

		s->used.cc |= E_MEMSB;
		break;
	}
	
	c = p->reg;
	if(c != NREG) {
		if(p->from.type == D_FREG || p->to.type == D_FREG)
			s->used.freg |= 1<<c;
		else
			s->used.ireg |= 1<<c;
	}
}

/*
 * test to see if 2 instrictions can be
 * interchanged without changing semantics
 */
int
depend(Sch *sa, Sch *sb)
{
	ulong x;

	if(sa->set.ireg & (sb->set.ireg|sb->used.ireg))
		return 1;
	if(sb->set.ireg & sa->used.ireg)
		return 1;

	if(sa->set.freg & (sb->set.freg|sb->used.freg))
		return 1;
	if(sb->set.freg & sa->used.freg)
		return 1;

	if(sa->set.cr & (sb->set.cr|sb->used.cr))
		return 1;
	if(sb->set.cr & sa->used.cr)
		return 1;


	x = (sa->set.cc & (sb->set.cc|sb->used.cc)) |
		(sb->set.cc & sa->used.cc);
	if(x) {
		/*
		 * allow SB and SP to pass each other.
		 * allow SB to pass SB iff doffsets are ok
		 * anything else conflicts
		 */
		if(x != E_MEMSP && x != E_MEMSB)
			return 1;
		x = sa->set.cc | sb->set.cc |
			sa->used.cc | sb->used.cc;
		if(x & E_MEM)
			return 1;
		if(offoverlap(sa, sb))
			return 1;
	}

	return 0; 
}

int
offoverlap(Sch *sa, Sch *sb)
{

	if(sa->soffset < sb->soffset) {
		if(sa->soffset+sa->size > sb->soffset)
			return 1;
		return 0;
	}
	if(sb->soffset+sb->size > sa->soffset)
		return 1;
	return 0;
}

/*
 * test 2 adjacent instructions
 * and find out if inserted instructions
 * are desired to prevent stalls.
 * first instruction is a load instruction.
 */
int
conflict(Sch *sa, Sch *sb)
{

	if(sa->set.ireg & sb->used.ireg)
		return 1;
	if(sa->set.freg & sb->used.freg)
		return 1;
	if(sa->set.cr & sb->used.cr)
		return 1;
	return 0;
}

int
compound(Prog *p)
{
	Optab *o;

	o = oplook(p);
	if(o->size != 4)
		return 1;
	if(p->to.type == D_REG && p->to.reg == REGSB)
		return 1;
	return 0;
}

void
dumpbits(Sch *s, Dep *d)
{
	int i;

	for(i=0; i<32; i++)
		if(d->ireg & (1<<i))
			Bprint(&bso, " R%d", i);
	for(i=0; i<32; i++)
		if(d->freg & (1<<i))
			Bprint(&bso, " F%d", i);
	for(i=0; i<32; i++)
		if(d->cr & (1<<i))
			Bprint(&bso, " C%d", i);
	for(i=0; i<32; i++)
		switch(d->cc & (1<<i)) {
		default:
			break;
		case E_ICC:
			Bprint(&bso, " ICC");
			break;
		case E_FCC:
			Bprint(&bso, " FCC");
			break;
		case E_LR:
			Bprint(&bso, " LR");
			break;
		case E_CR:
			Bprint(&bso, " CR");
			break;
		case E_CTR:
			Bprint(&bso, " CTR");
			break;
		case E_XER:
			Bprint(&bso, " XER");
			break;
		case E_MEM:
			Bprint(&bso, " MEM%d", s->size);
			break;
		case E_MEMSB:
			Bprint(&bso, " SB%d", s->size);
			break;
		case E_MEMSP:
			Bprint(&bso, " SP%d", s->size);
			break;
		}
}
