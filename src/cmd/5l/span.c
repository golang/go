// Inferno utils/5l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/5l/span.c
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

// Instruction layout.

#include	"l.h"
#include	"../ld/lib.h"

static struct {
	uint32	start;
	uint32	size;
	uint32	extra;
} pool;

int	checkpool(Prog*, int);
int 	flushpool(Prog*, int, int);

int
isbranch(Prog *p)
{
	int as = p->as;
	return (as >= ABEQ && as <= ABLE) || as == AB || as == ABL || as == ABX;
}

static int
scan(Prog *op, Prog *p, int c)
{
	Prog *q;

	for(q = op->link; q != p && q != P; q = q->link){
		q->pc = c;
		c += oplook(q)->size;
		nocache(q);
	}
	return c;
}

/* size of a case statement including jump table */
static int32
casesz(Prog *p)
{
	int jt = 0;
	int32 n = 0;
	Optab *o;

	for( ; p != P; p = p->link){
		if(p->as == ABCASE)
			jt = 1;
		else if(jt)
			break;
		o = oplook(p);
		n += o->size;
	}
	return n;
}

void
span(void)
{
	Prog *p, *op;
	Optab *o;
	int m, bflag, i, v;
	int32 c, otxt, out[6];
	Section *sect;
	uchar *bp;
	Sym *sub;

	if(debug['v'])
		Bprint(&bso, "%5.2f span\n", cputime());
	Bflush(&bso);

	sect = addsection(&segtext, ".text", 05);
	lookup("text", 0)->sect = sect;
	lookup("etext", 0)->sect = sect;

	bflag = 0;
	c = INITTEXT;
	otxt = c;
	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		cursym->sect = sect;
		p = cursym->text;
		if(p == P || p->link == P) { // handle external functions and ELF section symbols
			if(cursym->type & SSUB)
				continue;
			if(cursym->align != 0)
				c = rnd(c, cursym->align);
			cursym->value = 0;
			for(sub = cursym; sub != S; sub = sub->sub) {
				sub->value += c;
				for(p = sub->text; p != P; p = p->link)
					p->pc += sub->value;
			}
			c += cursym->size;
			continue;
		}
		p->pc = c;
		cursym->value = c;

		autosize = p->to.offset + 4;
		if(p->from.sym != S)
			p->from.sym->value = c;
		/* need passes to resolve branches */
		if(c-otxt >= 1L<<17)
			bflag = 1;
		otxt = c;

		for(op = p, p = p->link; p != P; op = p, p = p->link) {
			curp = p;
			p->pc = c;
			o = oplook(p);
			m = o->size;
			// must check literal pool here in case p generates many instructions
			if(blitrl){
				if(checkpool(op, p->as == ACASE ? casesz(p) : m))
					c = p->pc = scan(op, p, c);
			}
			if(m == 0) {
				diag("zero-width instruction\n%P", p);
				continue;
			}
			switch(o->flag & (LFROM|LTO|LPOOL)) {
			case LFROM:
				addpool(p, &p->from);
				break;
			case LTO:
				addpool(p, &p->to);
				break;
			case LPOOL:
				if ((p->scond&C_SCOND) == 14)
					flushpool(p, 0, 0);
				break;
			}
			if(p->as==AMOVW && p->to.type==D_REG && p->to.reg==REGPC && (p->scond&C_SCOND) == 14)
				flushpool(p, 0, 0);
			c += m;
		}
		if(blitrl){
			if(checkpool(op, 0))
				c = scan(op, P, c);
		}
		cursym->size = c - cursym->value;
	}

	/*
	 * if any procedure is large enough to
	 * generate a large SBRA branch, then
	 * generate extra passes putting branches
	 * around jmps to fix. this is rare.
	 */
	while(bflag) {
		if(debug['v'])
			Bprint(&bso, "%5.2f span1\n", cputime());
		bflag = 0;
		c = INITTEXT;
		for(cursym = textp; cursym != nil; cursym = cursym->next) {
			if(!cursym->text || !cursym->text->link)
				continue;
			cursym->value = c;
			for(p = cursym->text; p != P; p = p->link) {
				curp = p;
				p->pc = c;
				o = oplook(p);
/* very large branches
				if(o->type == 6 && p->cond) {
					otxt = p->cond->pc - c;
					if(otxt < 0)
						otxt = -otxt;
					if(otxt >= (1L<<17) - 10) {
						q = prg();
						q->link = p->link;
						p->link = q;
						q->as = AB;
						q->to.type = D_BRANCH;
						q->cond = p->cond;
						p->cond = q;
						q = prg();
						q->link = p->link;
						p->link = q;
						q->as = AB;
						q->to.type = D_BRANCH;
						q->cond = q->link->link;
						bflag = 1;
					}
				}
 */
				m = o->size;
				if(m == 0) {
					if(p->as == ATEXT) {
						autosize = p->to.offset + 4;
						if(p->from.sym != S)
							p->from.sym->value = c;
						continue;
					}
					diag("zero-width instruction\n%P", p);
					continue;
				}
				c += m;
			}
			cursym->size = c - cursym->value;
		}
	}

	c = rnd(c, 8);
	
	/*
	 * lay out the code.  all the pc-relative code references,
	 * even cross-function, are resolved now;
	 * only data references need to be relocated.
	 * with more work we could leave cross-function
	 * code references to be relocated too, and then
	 * perhaps we'd be able to parallelize the span loop above.
	 */
	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		p = cursym->text;
		if(p == P || p->link == P)
		       continue;
		autosize = p->to.offset + 4;
		symgrow(cursym, cursym->size);
	
		bp = cursym->p;
		for(p = p->link; p != P; p = p->link) {
			pc = p->pc;
			curp = p;
			o = oplook(p);
			asmout(p, o, out);
			for(i=0; i<o->size/4; i++) {
				v = out[i];
				*bp++ = v;
				*bp++ = v>>8;
				*bp++ = v>>16;
				*bp++ = v>>24;
			}
		}
	}
	sect->vaddr = INITTEXT;
	sect->len = c - INITTEXT;
}

/*
 * when the first reference to the literal pool threatens
 * to go out of range of a 12-bit PC-relative offset,
 * drop the pool now, and branch round it.
 * this happens only in extended basic blocks that exceed 4k.
 */
int
checkpool(Prog *p, int sz)
{
	if(pool.size >= 0xffc || immaddr((p->pc+sz+4)+4+pool.size - pool.start+8) == 0)
		return flushpool(p, 1, 0);
	else if(p->link == P)
		return flushpool(p, 2, 0);
	return 0;
}

int
flushpool(Prog *p, int skip, int force)
{
	Prog *q;

	if(blitrl) {
		if(skip){
			if(0 && skip==1)print("note: flush literal pool at %ux: len=%ud ref=%ux\n", p->pc+4, pool.size, pool.start);
			q = prg();
			q->as = AB;
			q->to.type = D_BRANCH;
			q->cond = p->link;
			q->link = blitrl;
			q->line = p->line;
			blitrl = q;
		}
		else if(!force && (p->pc+pool.size-pool.start < 2048))
			return 0;
		elitrl->link = p->link;
		p->link = blitrl;
		// BUG(minux): how to correctly handle line number for constant pool entries?
		// for now, we set line number to the last instruction preceding them at least
		// this won't bloat the .debug_line tables
		while(blitrl) {
			blitrl->line = p->line;
			blitrl = blitrl->link;
		}
		blitrl = 0;	/* BUG: should refer back to values until out-of-range */
		elitrl = 0;
		pool.size = 0;
		pool.start = 0;
		pool.extra = 0;
		return 1;
	}
	return 0;
}

void
addpool(Prog *p, Adr *a)
{
	Prog *q, t;
	int c;

	c = aclass(a);

	t = zprg;
	t.as = AWORD;

	switch(c) {
	default:
		t.to = *a;
		if(flag_shared && t.to.sym != S)
			t.pcrel = p;
		break;

	case C_SROREG:
	case C_LOREG:
	case C_ROREG:
	case C_FOREG:
	case C_SOREG:
	case C_HOREG:
	case C_FAUTO:
	case C_SAUTO:
	case C_LAUTO:
	case C_LACON:
		t.to.type = D_CONST;
		t.to.offset = instoffset;
		break;
	}

	if(t.pcrel == P) {
		for(q = blitrl; q != P; q = q->link)	/* could hash on t.t0.offset */
			if(q->pcrel == P && memcmp(&q->to, &t.to, sizeof(t.to)) == 0) {
				p->cond = q;
				return;
			}
	}

	q = prg();
	*q = t;
	q->pc = pool.size;

	if(blitrl == P) {
		blitrl = q;
		pool.start = p->pc;
		q->align = 4;
	} else
		elitrl->link = q;
	elitrl = q;
	pool.size += 4;

	p->cond = q;
}

void
xdefine(char *p, int t, int32 v)
{
	Sym *s;

	s = lookup(p, 0);
	s->type = t;
	s->value = v;
	s->reachable = 1;
	s->special = 1;
}

int32
regoff(Adr *a)
{

	instoffset = 0;
	aclass(a);
	return instoffset;
}

int32
immrot(uint32 v)
{
	int i;

	for(i=0; i<16; i++) {
		if((v & ~0xff) == 0)
			return (i<<8) | v | (1<<25);
		v = (v<<2) | (v>>30);
	}
	return 0;
}

int32
immaddr(int32 v)
{
	if(v >= 0 && v <= 0xfff)
		return (v & 0xfff) |
			(1<<24) |	/* pre indexing */
			(1<<23);	/* pre indexing, up */
	if(v >= -0xfff && v < 0)
		return (-v & 0xfff) |
			(1<<24);	/* pre indexing */
	return 0;
}

int
immfloat(int32 v)
{
	return (v & 0xC03) == 0;	/* offset will fit in floating-point load/store */
}

int
immhalf(int32 v)
{
	if(v >= 0 && v <= 0xff)
		return v|
			(1<<24)|	/* pre indexing */
			(1<<23);	/* pre indexing, up */
	if(v >= -0xff && v < 0)
		return (-v & 0xff)|
			(1<<24);	/* pre indexing */
	return 0;
}

int32
symaddr(Sym *s)
{
	if(!s->reachable)
		diag("unreachable symbol in symaddr - %s", s->name);
	return s->value;
}

int
aclass(Adr *a)
{
	Sym *s;
	int t;

	switch(a->type) {
	case D_NONE:
		return C_NONE;

	case D_REG:
		return C_REG;

	case D_REGREG:
		return C_REGREG;

	case D_REGREG2:
		return C_REGREG2;

	case D_SHIFT:
		return C_SHIFT;

	case D_FREG:
		return C_FREG;

	case D_FPCR:
		return C_FCR;

	case D_OREG:
		switch(a->name) {
		case D_EXTERN:
		case D_STATIC:
			if(a->sym == 0 || a->sym->name == 0) {
				print("null sym external\n");
				print("%D\n", a);
				return C_GOK;
			}
			instoffset = 0;	// s.b. unused but just in case
			return C_ADDR;

		case D_AUTO:
			instoffset = autosize + a->offset;
			t = immaddr(instoffset);
			if(t){
				if(immhalf(instoffset))
					return immfloat(t) ? C_HFAUTO : C_HAUTO;
				if(immfloat(t))
					return C_FAUTO;
				return C_SAUTO;
			}
			return C_LAUTO;

		case D_PARAM:
			instoffset = autosize + a->offset + 4L;
			t = immaddr(instoffset);
			if(t){
				if(immhalf(instoffset))
					return immfloat(t) ? C_HFAUTO : C_HAUTO;
				if(immfloat(t))
					return C_FAUTO;
				return C_SAUTO;
			}
			return C_LAUTO;
		case D_NONE:
			instoffset = a->offset;
			t = immaddr(instoffset);
			if(t) {
				if(immhalf(instoffset))		 /* n.b. that it will also satisfy immrot */
					return immfloat(t) ? C_HFOREG : C_HOREG;
				if(immfloat(t))
					return C_FOREG; /* n.b. that it will also satisfy immrot */
				t = immrot(instoffset);
				if(t)
					return C_SROREG;
				if(immhalf(instoffset))
					return C_HOREG;
				return C_SOREG;
			}
			t = immrot(instoffset);
			if(t)
				return C_ROREG;
			return C_LOREG;
		}
		return C_GOK;

	case D_PSR:
		return C_PSR;

	case D_OCONST:
		switch(a->name) {
		case D_EXTERN:
		case D_STATIC:
			instoffset = 0;	// s.b. unused but just in case
			return C_ADDR;
		}
		return C_GOK;

	case D_FCONST:
		if(chipzero(&a->ieee) >= 0)
			return C_ZFCON;
		if(chipfloat(&a->ieee) >= 0)
			return C_SFCON;
		return C_LFCON;

	case D_CONST:
	case D_CONST2:
		switch(a->name) {

		case D_NONE:
			instoffset = a->offset;
			if(a->reg != NREG)
				goto aconsize;

			t = immrot(instoffset);
			if(t)
				return C_RCON;
			t = immrot(~instoffset);
			if(t)
				return C_NCON;
			return C_LCON;

		case D_EXTERN:
		case D_STATIC:
			s = a->sym;
			if(s == S)
				break;
			instoffset = 0;	// s.b. unused but just in case
			if(flag_shared)
				return C_LCONADDR;
			else
				return C_LCON;

		case D_AUTO:
			instoffset = autosize + a->offset;
			goto aconsize;

		case D_PARAM:
			instoffset = autosize + a->offset + 4L;
		aconsize:
			t = immrot(instoffset);
			if(t)
				return C_RACON;
			return C_LACON;
		}
		return C_GOK;

	case D_BRANCH:
		return C_SBRA;
	}
	return C_GOK;
}

Optab*
oplook(Prog *p)
{
	int a1, a2, a3, r;
	char *c1, *c3;
	Optab *o, *e;

	a1 = p->optab;
	if(a1)
		return optab+(a1-1);
	a1 = p->from.class;
	if(a1 == 0) {
		a1 = aclass(&p->from) + 1;
		p->from.class = a1;
	}
	a1--;
	a3 = p->to.class;
	if(a3 == 0) {
		a3 = aclass(&p->to) + 1;
		p->to.class = a3;
	}
	a3--;
	a2 = C_NONE;
	if(p->reg != NREG)
		a2 = C_REG;
	r = p->as;
	o = oprange[r].start;
	if(o == 0) {
		a1 = opcross[repop[r]][a1][a2][a3];
		if(a1) {
			p->optab = a1+1;
			return optab+a1;
		}
		o = oprange[r].stop; /* just generate an error */
	}
	if(debug['O']) {
		print("oplook %A %O %O %O\n",
			(int)p->as, a1, a2, a3);
		print("		%d %d\n", p->from.type, p->to.type);
	}
	e = oprange[r].stop;
	c1 = xcmp[a1];
	c3 = xcmp[a3];
	for(; o<e; o++)
		if(o->a2 == a2)
		if(c1[o->a1])
		if(c3[o->a3]) {
			p->optab = (o-optab)+1;
			return o;
		}
	diag("illegal combination %A %O %O %O, %d %d",
		p->as, a1, a2, a3, p->from.type, p->to.type);
	prasm(p);
	if(o == 0)
		o = optab;
	return o;
}

int
cmp(int a, int b)
{

	if(a == b)
		return 1;
	switch(a) {
	case C_LCON:
		if(b == C_RCON || b == C_NCON)
			return 1;
		break;
	case C_LACON:
		if(b == C_RACON)
			return 1;
		break;
	case C_LFCON:
		if(b == C_ZFCON || b == C_SFCON)
			return 1;
		break;

	case C_HFAUTO:
		return b == C_HAUTO || b == C_FAUTO;
	case C_FAUTO:
	case C_HAUTO:
		return b == C_HFAUTO;
	case C_SAUTO:
		return cmp(C_HFAUTO, b);
	case C_LAUTO:
		return cmp(C_SAUTO, b);

	case C_HFOREG:
		return b == C_HOREG || b == C_FOREG;
	case C_FOREG:
	case C_HOREG:
		return b == C_HFOREG;
	case C_SROREG:
		return cmp(C_SOREG, b) || cmp(C_ROREG, b);
	case C_SOREG:
	case C_ROREG:
		return b == C_SROREG || cmp(C_HFOREG, b);
	case C_LOREG:
		return cmp(C_SROREG, b);

	case C_LBRA:
		if(b == C_SBRA)
			return 1;
		break;

	case C_HREG:
		return cmp(C_SP, b) || cmp(C_PC, b);

	}
	return 0;
}

int
ocmp(const void *a1, const void *a2)
{
	Optab *p1, *p2;
	int n;

	p1 = (Optab*)a1;
	p2 = (Optab*)a2;
	n = p1->as - p2->as;
	if(n)
		return n;
	n = p1->a1 - p2->a1;
	if(n)
		return n;
	n = p1->a2 - p2->a2;
	if(n)
		return n;
	n = p1->a3 - p2->a3;
	if(n)
		return n;
	return 0;
}

void
buildop(void)
{
	int i, n, r;

	for(i=0; i<C_GOK; i++)
		for(n=0; n<C_GOK; n++)
			xcmp[i][n] = cmp(n, i);
	for(n=0; optab[n].as != AXXX; n++) {
		if((optab[n].flag & LPCREL) != 0) {
			if(flag_shared)
				optab[n].size += optab[n].pcrelsiz;
			else
				optab[n].flag &= ~LPCREL;
		}
	}
	qsort(optab, n, sizeof(optab[0]), ocmp);
	for(i=0; i<n; i++) {
		r = optab[i].as;
		oprange[r].start = optab+i;
		while(optab[i].as == r)
			i++;
		oprange[r].stop = optab+i;
		i--;

		switch(r)
		{
		default:
			diag("unknown op in build: %A", r);
			errorexit();
		case AADD:
			oprange[AAND] = oprange[r];
			oprange[AEOR] = oprange[r];
			oprange[ASUB] = oprange[r];
			oprange[ARSB] = oprange[r];
			oprange[AADC] = oprange[r];
			oprange[ASBC] = oprange[r];
			oprange[ARSC] = oprange[r];
			oprange[AORR] = oprange[r];
			oprange[ABIC] = oprange[r];
			break;
		case ACMP:
			oprange[ATEQ] = oprange[r];
			oprange[ACMN] = oprange[r];
			break;
		case AMVN:
			break;
		case ABEQ:
			oprange[ABNE] = oprange[r];
			oprange[ABCS] = oprange[r];
			oprange[ABHS] = oprange[r];
			oprange[ABCC] = oprange[r];
			oprange[ABLO] = oprange[r];
			oprange[ABMI] = oprange[r];
			oprange[ABPL] = oprange[r];
			oprange[ABVS] = oprange[r];
			oprange[ABVC] = oprange[r];
			oprange[ABHI] = oprange[r];
			oprange[ABLS] = oprange[r];
			oprange[ABGE] = oprange[r];
			oprange[ABLT] = oprange[r];
			oprange[ABGT] = oprange[r];
			oprange[ABLE] = oprange[r];
			break;
		case ASLL:
			oprange[ASRL] = oprange[r];
			oprange[ASRA] = oprange[r];
			break;
		case AMUL:
			oprange[AMULU] = oprange[r];
			break;
		case ADIV:
			oprange[AMOD] = oprange[r];
			oprange[AMODU] = oprange[r];
			oprange[ADIVU] = oprange[r];
			break;
		case AMOVW:
		case AMOVB:
		case AMOVBU:
		case AMOVH:
		case AMOVHU:
			break;
		case ASWPW:
			oprange[ASWPBU] = oprange[r];
			break;
		case AB:
		case ABL:
		case ABX:
		case ABXRET:
		case ASWI:
		case AWORD:
		case AMOVM:
		case ARFE:
		case ATEXT:
		case AUSEFIELD:
		case ALOCALS:
		case ACASE:
		case ABCASE:
		case ATYPE:
			break;
		case AADDF:
			oprange[AADDD] = oprange[r];
			oprange[ASUBF] = oprange[r];
			oprange[ASUBD] = oprange[r];
			oprange[AMULF] = oprange[r];
			oprange[AMULD] = oprange[r];
			oprange[ADIVF] = oprange[r];
			oprange[ADIVD] = oprange[r];
			oprange[ASQRTF] = oprange[r];
			oprange[ASQRTD] = oprange[r];
			oprange[AMOVFD] = oprange[r];
			oprange[AMOVDF] = oprange[r];
			oprange[AABSF] = oprange[r];
			oprange[AABSD] = oprange[r];
			break;

		case ACMPF:
			oprange[ACMPD] = oprange[r];
			break;

		case AMOVF:
			oprange[AMOVD] = oprange[r];
			break;

		case AMOVFW:
			oprange[AMOVDW] = oprange[r];
			break;

		case AMOVWF:
			oprange[AMOVWD] = oprange[r];
			break;

		case AMULL:
			oprange[AMULAL] = oprange[r];
			oprange[AMULLU] = oprange[r];
			oprange[AMULALU] = oprange[r];
			break;

		case AMULWT:
			oprange[AMULWB] = oprange[r];
			break;

		case AMULAWT:
			oprange[AMULAWB] = oprange[r];
			break;

		case AMULA:
		case ALDREX:
		case ASTREX:
		case ALDREXD:
		case ASTREXD:
		case ATST:
		case APLD:
		case AUNDEF:
		case ACLZ:
			break;
		}
	}
}

/*
void
buildrep(int x, int as)
{
	Opcross *p;
	Optab *e, *s, *o;
	int a1, a2, a3, n;

	if(C_NONE != 0 || C_REG != 1 || C_GOK >= 32 || x >= nelem(opcross)) {
		diag("assumptions fail in buildrep");
		errorexit();
	}
	repop[as] = x;
	p = (opcross + x);
	s = oprange[as].start;
	e = oprange[as].stop;
	for(o=e-1; o>=s; o--) {
		n = o-optab;
		for(a2=0; a2<2; a2++) {
			if(a2) {
				if(o->a2 == C_NONE)
					continue;
			} else
				if(o->a2 != C_NONE)
					continue;
			for(a1=0; a1<32; a1++) {
				if(!xcmp[a1][o->a1])
					continue;
				for(a3=0; a3<32; a3++)
					if(xcmp[a3][o->a3])
						(*p)[a1][a2][a3] = n;
			}
		}
	}
	oprange[as].start = 0;
}
*/
