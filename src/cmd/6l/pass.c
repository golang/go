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

#include	"l.h"
#include	"../ld/lib.h"

// see ../../runtime/proc.c:/StackGuard
enum
{
	StackSmall = 128,
	StackBig = 4096,
};

void
dodata(void)
{
	int i;
	Sym *s;
	Prog *p;
	int32 t, u;

	if(debug['v'])
		Bprint(&bso, "%5.2f dodata\n", cputime());
	Bflush(&bso);
	for(p = datap; p != P; p = p->link) {
		curtext = p;	// for diag messages
		s = p->from.sym;
		if(p->as == ADYNT || p->as == AINIT)
			s->value = dtype;
		if(s->type == SBSS)
			s->type = SDATA;
		if(s->type != SDATA)
			diag("initialize non-data (%d): %s\n%P",
				s->type, s->name, p);
		t = p->from.offset + p->width;
		if(t > s->value)
			diag("initialize bounds (%lld): %s\n%P",
				s->value, s->name, p);
	}
	/* allocate small guys */
	datsize = 0;
	for(i=0; i<NHASH; i++)
	for(s = hash[i]; s != S; s = s->link) {
		if(!s->reachable)
			continue;
		if(s->type != SDATA)
		if(s->type != SBSS)
			continue;
		t = s->value;
		if(t == 0 && s->name[0] != '.') {
			diag("%s: no size", s->name);
			t = 1;
		}
		t = rnd(t, 4);
		s->value = t;
		if(t > MINSIZ)
			continue;
		if(t >= 8)
			datsize = rnd(datsize, 8);
		s->size = t;
		s->value = datsize;
		datsize += t;
		s->type = SDATA1;
	}

	/* allocate the rest of the data */
	for(i=0; i<NHASH; i++)
	for(s = hash[i]; s != S; s = s->link) {
		if(!s->reachable)
			continue;
		if(s->type != SDATA) {
			if(s->type == SDATA1)
				s->type = SDATA;
			continue;
		}
		t = s->value;
		if(t >= 8)
			datsize = rnd(datsize, 8);
		s->size = t;
		s->value = datsize;
		datsize += t;
	}
	if(datsize)
		datsize = rnd(datsize, 8);

	if(debug['j']) {
		/*
		 * pad data with bss that fits up to next
		 * 8k boundary, then push data to 8k
		 */
		u = rnd(datsize, 8192);
		u -= datsize;
		for(i=0; i<NHASH; i++)
		for(s = hash[i]; s != S; s = s->link) {
			if(!s->reachable)
				continue;
			if(s->type != SBSS)
				continue;
			t = s->value;
			if(t > u)
				continue;
			u -= t;
			s->size = t;
			s->value = datsize;
			s->type = SDATA;
			datsize += t;
		}
		datsize += u;
	}
}

void
dobss(void)
{
	int i;
	Sym *s;
	int32 t;

	if(dynptrsize > 0) {
		/* dynamic pointer section between data and bss */
		datsize = rnd(datsize, 8);
	}

	/* now the bss */
	bsssize = 0;
	for(i=0; i<NHASH; i++)
	for(s = hash[i]; s != S; s = s->link) {
		if(!s->reachable)
			continue;
		if(s->type != SBSS)
			continue;
		t = s->value;
		s->size = t;
		if(t >= 8)
			bsssize = rnd(bsssize, 8);
		s->value = bsssize + dynptrsize + datsize;
		bsssize += t;
	}

	xdefine("data", SBSS, 0);
	xdefine("edata", SBSS, datsize);
	xdefine("end", SBSS, dynptrsize + bsssize + datsize);
}

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

	if(debug['v'])
		Bprint(&bso, "%5.2f follow\n", cputime());
	Bflush(&bso);
	firstp = prg();
	lastp = firstp;
	xfol(textp);
	lastp->link = P;
	firstp = firstp->link;
}

void
xfol(Prog *p)
{
	Prog *q;
	int i;
	enum as a;

loop:
	if(p == P)
		return;
	if(p->as == ATEXT)
		curtext = p;
	if(!curtext->from.sym->reachable) {
		p = p->pcond;
		goto loop;
	}
	if(p->as == AJMP)
	if((q = p->pcond) != P && q->as != ATEXT) {
		p->mark = 1;
		p = q;
		if(p->mark == 0)
			goto loop;
	}
	if(p->mark) {
		/* copy up to 4 instructions to avoid branch */
		for(i=0,q=p; i<4; i++,q=q->link) {
			if(q == P)
				break;
			if(q == lastp)
				break;
			a = q->as;
			if(a == ANOP) {
				i--;
				continue;
			}
			switch(a) {
			case AJMP:
			case ARET:
			case AIRETL:
			case AIRETQ:
			case AIRETW:
			case ARETFL:
			case ARETFQ:
			case ARETFW:

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
				goto brk;
			}
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
				lastp->link = q;
				lastp = q;
				if(q->as != a || q->pcond == P || q->pcond->mark)
					continue;

				q->as = relinv(q->as);
				p = q->pcond;
				q->pcond = q->link;
				q->link = p;
				xfol(q->link);
				p = q->link;
				if(p->mark)
					return;
				goto loop;
			}
		} /* */
	brk:;
		q = prg();
		q->as = AJMP;
		q->line = p->line;
		q->to.type = D_BRANCH;
		q->to.offset = p->pc;
		q->pcond = p;
		p = q;
	}
	p->mark = 1;
	lastp->link = p;
	lastp = p;
	a = p->as;
	if(a == AJMP || a == ARET || a == AIRETL || a == AIRETQ || a == AIRETW ||
	   a == ARETFL || a == ARETFQ || a == ARETFW)
		return;
	if(p->pcond != P)
	if(a != ACALL) {
		q = brchain(p->link);
		if(q != P && q->mark)
		if(a != ALOOP) {
			p->as = relinv(a);
			p->link = p->pcond;
			p->pcond = q;
		}
		xfol(p->link);
		q = brchain(p->pcond);
		if(q->mark) {
			p->pcond = q;
			return;
		}
		p = q;
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
	return a;
}

void
doinit(void)
{
	Sym *s;
	Prog *p;
	int x;

	for(p = datap; p != P; p = p->link) {
		x = p->to.type;
		if(x != D_EXTERN && x != D_STATIC)
			continue;
		s = p->to.sym;
		if(s->type == 0 || s->type == SXREF)
			diag("undefined %s initializer of %s",
				s->name, p->from.sym->name);
		p->to.offset += s->value;
		p->to.type = D_CONST;
		if(s->type == SDATA || s->type == SBSS)
			p->to.offset += INITDAT;
	}
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
	for(p = firstp; p != P; p = p->link) {
		if(p->as == ATEXT)
			curtext = p;
		if(p->as == ACALL || (p->as == AJMP && p->to.type != D_BRANCH)) {
			s = p->to.sym;
			if(s) {
				if(debug['c'])
					Bprint(&bso, "%s calls %s\n", TNAME, s->name);
				switch(s->type) {
				default:
					/* diag prints TNAME first */
					diag("undefined: %s", s->name);
					s->type = STEXT;
					s->value = vexit;
					continue;	// avoid more error messages
				case STEXT:
					p->to.offset = s->value;
					break;
				case SUNDEF:
					p->pcond = UP;
					p->to.offset = 0;
					break;
				}
				p->to.type = D_BRANCH;
			}
		}
		if(p->to.type != D_BRANCH || p->pcond == UP)
			continue;
		c = p->to.offset;
		for(q = firstp; q != P;) {
			if(q->forwd != P)
			if(c >= q->forwd->pc) {
				q = q->forwd;
				continue;
			}
			if(c == q->pc)
				break;
			q = q->link;
		}
		if(q == P) {
			diag("branch out of range in %s\n%P [%s]",
				TNAME, p, p->to.sym ? p->to.sym->name : "<nil>");
			p->to.type = D_NONE;
		}
		p->pcond = q;
	}

	for(p = firstp; p != P; p = p->link) {
		if(p->as == ATEXT)
			curtext = p;
		p->mark = 0;	/* initialization for follow */
		if(p->pcond != P && p->pcond != UP) {
			p->pcond = brloop(p->pcond);
			if(p->pcond != P)
			if(p->to.type == D_BRANCH)
				p->to.offset = p->pcond->pc;
		}
	}
}

#define	LOG	5
void
mkfwd(void)
{
	Prog *p;
	int i;
	int32 dwn[LOG], cnt[LOG];
	Prog *lst[LOG];

	for(i=0; i<LOG; i++) {
		if(i == 0)
			cnt[i] = 1; else
			cnt[i] = LOG * cnt[i-1];
		dwn[i] = 1;
		lst[i] = P;
	}
	i = 0;
	for(p = firstp; p != P; p = p->link) {
		if(p->as == ATEXT)
			curtext = p;
		i--;
		if(i < 0)
			i = LOG-1;
		p->forwd = P;
		dwn[i]--;
		if(dwn[i] <= 0) {
			dwn[i] = cnt[i];
			if(lst[i] != P)
				lst[i]->forwd = p;
			lst[i] = p;
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
	"runtime·morestack00",
	"runtime·morestack10",
	"runtime·morestack01",
	"runtime·morestack11",

	"runtime·morestack8",
	"runtime·morestack16",
	"runtime·morestack24",
	"runtime·morestack32",
	"runtime·morestack40",
	"runtime·morestack48",
};
Prog*	pmorestack[nelem(morename)];
Sym*	symmorestack[nelem(morename)];

void
dostkoff(void)
{
	Prog *p, *q, *q1;
	int32 autoffset, deltasp;
	int a, f, curframe, curbecome, maxbecome, pcsize;
	uint32 moreconst1, moreconst2, i;

	for(i=0; i<nelem(morename); i++) {
		symmorestack[i] = lookup(morename[i], 0);
		pmorestack[i] = P;
	}

	for(p = firstp; p != P; p = p->link) {
		if(p->as == ATEXT) {
			for(i=0; i<nelem(morename); i++) {
				if(p->from.sym == symmorestack[i]) {
					pmorestack[i] = p;
					break;
				}
			}
		}
	}

	for(i=0; i<nelem(morename); i++) {
		if(pmorestack[i] == P)
			diag("morestack trampoline not defined");
	}

	curframe = 0;
	curbecome = 0;
	maxbecome = 0;
	curtext = 0;
	for(p = firstp; p != P; p = p->link) {

		/* find out how much arg space is used in this TEXT */
		if(p->to.type == (D_INDIR+D_SP))
			if(p->to.offset > curframe)
				curframe = p->to.offset;

		switch(p->as) {
		case ATEXT:
			if(curtext && curtext->from.sym) {
				curtext->from.sym->frame = curframe;
				curtext->from.sym->become = curbecome;
				if(curbecome > maxbecome)
					maxbecome = curbecome;
			}
			curframe = 0;
			curbecome = 0;

			curtext = p;
			break;

		case ARET:
			/* special form of RET is BECOME */
			if(p->from.type == D_CONST)
				if(p->from.offset > curbecome)
					curbecome = p->from.offset;
			break;
		}
	}
	if(curtext && curtext->from.sym) {
		curtext->from.sym->frame = curframe;
		curtext->from.sym->become = curbecome;
		if(curbecome > maxbecome)
			maxbecome = curbecome;
	}

	if(debug['b'])
		print("max become = %d\n", maxbecome);
	xdefine("ALEFbecome", STEXT, maxbecome);

	curtext = 0;
	for(p = firstp; p != P; p = p->link) {
		switch(p->as) {
		case ATEXT:
			curtext = p;
			break;
		case ACALL:
			if(curtext != P && curtext->from.sym != S && curtext->to.offset >= 0) {
				f = maxbecome - curtext->from.sym->frame;
				if(f <= 0)
					break;
				/* calling a become or calling a variable */
				if(p->to.sym == S || p->to.sym->become) {
					curtext->to.offset += f;
					if(debug['b']) {
						curp = p;
						print("%D calling %D increase %d\n",
							&curtext->from, &p->to, f);
					}
				}
			}
			break;
		}
	}

	autoffset = 0;
	deltasp = 0;
	for(p = firstp; p != P; p = p->link) {
		if(p->as == ATEXT) {
			curtext = p;
			parsetextconst(p->to.offset);
			autoffset = textstksiz;
			if(autoffset < 0)
				autoffset = 0;

			q = P;
			q1 = P;
			if((p->from.scale & NOSPLIT) && autoffset >= StackSmall)
				diag("nosplit func likely to overflow stack");

			if(!(p->from.scale & NOSPLIT)) {
				if(debug['K']) {
					// 6l -K means check not only for stack
					// overflow but stack underflow.
					// On underflow, INT 3 (breakpoint).
					// Underflow itself is rare but this also
					// catches out-of-sync stack guard info

					p = appendp(p);
					p->as = ACMPQ;
					p->from.type = D_INDIR+D_R15;
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
				}

				if(autoffset < StackBig) {  // do we need to call morestack?
					if(autoffset <= StackSmall) {
						// small stack
						p = appendp(p);
						p->as = ACMPQ;
						p->from.type = D_SP;
						p->to.type = D_INDIR+D_R15;
						if(q1) {
							q1->pcond = p;
							q1 = P;
						}
					} else {
						// large stack
						p = appendp(p);
						p->as = ALEAQ;
						p->from.type = D_INDIR+D_SP;
						p->from.offset = -(autoffset-StackSmall);
						p->to.type = D_AX;
						if(q1) {
							q1->pcond = p;
							q1 = P;
						}

						p = appendp(p);
						p->as = ACMPQ;
						p->from.type = D_AX;
						p->to.type = D_INDIR+D_R15;
					}

					// common
					p = appendp(p);
					p->as = AJHI;
					p->to.type = D_BRANCH;
					p->to.offset = 4;
					q = p;
				}

				/* 160 comes from 3 calls (3*8) 4 safes (4*8) and 104 guard */
				moreconst1 = 0;
				if(autoffset+160 > 4096)
					moreconst1 = (autoffset+160) & ~7LL;
				moreconst2 = textarg;

				// 4 varieties varieties (const1==0 cross const2==0)
				// and 6 subvarieties of (const1==0 and const2!=0)
				p = appendp(p);
				if(moreconst1 == 0 && moreconst2 == 0) {
					p->as = ACALL;
					p->to.type = D_BRANCH;
					p->pcond = pmorestack[0];
					p->to.sym = symmorestack[0];
					if(q1) {
						q1->pcond = p;
						q1 = P;
					}
				} else
				if(moreconst1 != 0 && moreconst2 == 0) {
					p->as = AMOVL;
					p->from.type = D_CONST;
					p->from.offset = moreconst1;
					p->to.type = D_AX;
					if(q1) {
						q1->pcond = p;
						q1 = P;
					}

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
					if(q1) {
						q1->pcond = p;
						q1 = P;
					}
				} else
				if(moreconst1 == 0 && moreconst2 != 0) {
					p->as = AMOVL;
					p->from.type = D_CONST;
					p->from.offset = moreconst2;
					p->to.type = D_AX;
					if(q1) {
						q1->pcond = p;
						q1 = P;
					}

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
					if(q1) {
						q1->pcond = p;
						q1 = P;
					}

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
				if(q != P)
					q->pcond = p;
			}
			deltasp = autoffset;

			if(debug['K'] > 1 && autoffset) {
				// 6l -KK means double-check for stack overflow
				// even after calling morestack and even if the
				// function is marked as nosplit.
				p = appendp(p);
				p->as = AMOVQ;
				p->from.type = D_INDIR+D_R15;
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
				q1 = P;
			}
		}
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
			continue;
		case APUSHQ:
		case APUSHFQ:
			deltasp += 8;
			continue;
		case APUSHW:
		case APUSHFW:
			deltasp += 2;
			continue;
		case APOPL:
		case APOPFL:
			deltasp -= 4;
			continue;
		case APOPQ:
		case APOPFQ:
			deltasp -= 8;
			continue;
		case APOPW:
		case APOPFW:
			deltasp -= 2;
			continue;
		case ARET:
			break;
		}

		if(autoffset != deltasp)
			diag("unbalanced PUSH/POP");
		if(p->from.type == D_CONST)
			goto become;

		if(autoffset) {
			p->as = AADJSP;
			p->from.type = D_CONST;
			p->from.offset = -autoffset;

			p = appendp(p);
			p->as = ARET;
		}
		continue;

	become:
		q = p;
		p = appendp(p);
		p->as = AJMP;
		p->to = q->to;
		p->pcond = q->pcond;

		q->as = AADJSP;
		q->from = zprg.from;
		q->from.type = D_CONST;
		q->from.offset = -autoffset;
		q->to = zprg.to;
		continue;
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

void
undef(void)
{
	int i;
	Sym *s;

	for(i=0; i<NHASH; i++)
	for(s = hash[i]; s != S; s = s->link)
		if(s->type == SXREF)
			diag("%s: not defined", s->name);
}

void
import(void)
{
	int i;
	Sym *s;

	for(i = 0; i < NHASH; i++)
		for(s = hash[i]; s != S; s = s->link)
			if(s->sig != 0 && s->type == SXREF && (nimports == 0 || s->subtype == SIMPORT)){
				if(s->value != 0)
					diag("value != 0 on SXREF");
				undefsym(s);
				Bprint(&bso, "IMPORT: %s sig=%lux v=%lld\n", s->name, s->sig, s->value);
				if(debug['S'])
					s->sig = 0;
			}
}

void
ckoff(Sym *s, int32 v)
{
	if(v < 0 || v >= 1<<Roffset)
		diag("relocation offset %ld for %s out of range", v, s->name);
}

Prog*
newdata(Sym *s, int o, int w, int t)
{
	Prog *p;

	p = prg();
	if(edatap == P)
		datap = p;
	else
		edatap->link = p;
	edatap = p;
	p->as = ADATA;
	p->width = w;
	p->from.scale = w;
	p->from.type = t;
	p->from.sym = s;
	p->from.offset = o;
	p->to.type = D_CONST;
	p->dlink = s->data;
	s->data = p;
	return p;
}

Prog*
newtext(Prog *p, Sym *s)
{
	if(p == P) {
		p = prg();
		p->as = ATEXT;
		p->from.sym = s;
	}
	s->type = STEXT;
	s->text = p;
	s->value = pc;
	lastp->link = p;
	lastp = p;
	p->pc = pc++;
	if(textp == P)
		textp = p;
	else
		etextp->pcond = p;
	etextp = p;
	return p;
}

void
export(void)
{
	int i, j, n, off, nb, sv, ne;
	Sym *s, *et, *str, **esyms;
	Prog *p;
	char buf[NSNAME], *t;

	n = 0;
	for(i = 0; i < NHASH; i++)
		for(s = hash[i]; s != S; s = s->link)
			if(s->sig != 0 && s->type != SXREF &&
			   s->type != SUNDEF &&
			   (nexports == 0 || s->subtype == SEXPORT))
				n++;
	esyms = mal(n*sizeof(Sym*));
	ne = n;
	n = 0;
	for(i = 0; i < NHASH; i++)
		for(s = hash[i]; s != S; s = s->link)
			if(s->sig != 0 && s->type != SXREF &&
			   s->type != SUNDEF &&
			   (nexports == 0 || s->subtype == SEXPORT))
				esyms[n++] = s;
	for(i = 0; i < ne-1; i++)
		for(j = i+1; j < ne; j++)
			if(strcmp(esyms[i]->name, esyms[j]->name) > 0){
				s = esyms[i];
				esyms[i] = esyms[j];
				esyms[j] = s;
			}

	nb = 0;
	off = 0;
	et = lookup(EXPTAB, 0);
	if(et->type != 0 && et->type != SXREF)
		diag("%s already defined", EXPTAB);
	et->type = SDATA;
	str = lookup(".string", 0);
	if(str->type == 0)
		str->type = SDATA;
	sv = str->value;
	for(i = 0; i < ne; i++){
		s = esyms[i];
		if(debug['S'])
			s->sig = 0;
		/* Bprint(&bso, "EXPORT: %s sig=%lux t=%d\n", s->name, s->sig, s->type); */

		/* signature */
		p = newdata(et, off, sizeof(int32), D_EXTERN);
		off += sizeof(int32);
		p->to.offset = s->sig;

		/* address */
		p = newdata(et, off, sizeof(int32), D_EXTERN);
		off += sizeof(int32);
		p->to.type = D_ADDR;
		p->to.index = D_EXTERN;
		p->to.sym = s;

		/* string */
		t = s->name;
		n = strlen(t)+1;
		for(;;){
			buf[nb++] = *t;
			sv++;
			if(nb >= NSNAME){
				p = newdata(str, sv-NSNAME, NSNAME, D_STATIC);
				p->to.type = D_SCONST;
				memmove(p->to.scon, buf, NSNAME);
				nb = 0;
			}
			if(*t++ == 0)
				break;
		}

		/* name */
		p = newdata(et, off, sizeof(int32), D_EXTERN);
		off += sizeof(int32);
		p->to.type = D_ADDR;
		p->to.index = D_STATIC;
		p->to.sym = str;
		p->to.offset = sv-n;
	}

	if(nb > 0){
		p = newdata(str, sv-nb, nb, D_STATIC);
		p->to.type = D_SCONST;
		memmove(p->to.scon, buf, nb);
	}

	for(i = 0; i < 3; i++){
		newdata(et, off, sizeof(int32), D_EXTERN);
		off += sizeof(int32);
	}
	et->value = off;
	if(sv == 0)
		sv = 1;
	str->value = sv;
	exports = ne;
	free(esyms);
}
