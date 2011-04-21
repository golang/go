// Inferno utils/8l/obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/8l/obj.c
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

// Profiling.

#include	"l.h"
#include	"../ld/lib.h"

void
doprof1(void)
{
#ifdef	NOTDEF  // TODO(rsc)
	Sym *s;
	int32 n;
	Prog *p, *q;

	if(debug['v'])
		Bprint(&bso, "%5.2f profile 1\n", cputime());
	Bflush(&bso);
	s = lookup("__mcount", 0);
	n = 1;
	for(p = firstp->link; p != P; p = p->link) {
		if(p->as == ATEXT) {
			q = prg();
			q->line = p->line;
			q->link = datap;
			datap = q;
			q->as = ADATA;
			q->from.type = D_EXTERN;
			q->from.offset = n*4;
			q->from.sym = s;
			q->from.scale = 4;
			q->to = p->from;
			q->to.type = D_CONST;

			q = prg();
			q->line = p->line;
			q->pc = p->pc;
			q->link = p->link;
			p->link = q;
			p = q;
			p->as = AADDL;
			p->from.type = D_CONST;
			p->from.offset = 1;
			p->to.type = D_EXTERN;
			p->to.sym = s;
			p->to.offset = n*4 + 4;

			n += 2;
			continue;
		}
	}
	q = prg();
	q->line = 0;
	q->link = datap;
	datap = q;

	q->as = ADATA;
	q->from.type = D_EXTERN;
	q->from.sym = s;
	q->from.scale = 4;
	q->to.type = D_CONST;
	q->to.offset = n;

	s->type = SBSS;
	s->size = n*4;
#endif
}

void
doprof2(void)
{
	Sym *s2, *s4;
	Prog *p, *q, *ps2, *ps4;

	if(debug['v'])
		Bprint(&bso, "%5.2f profile 2\n", cputime());
	Bflush(&bso);

	s2 = lookup("_profin", 0);
	s4 = lookup("_profout", 0);
	if(s2->type != STEXT || s4->type != STEXT) {
		diag("_profin/_profout not defined");
		return;
	}

	ps2 = P;
	ps4 = P;
	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		p = cursym->text;
		if(p->from.sym == s2) {
			p->from.scale = 1;
			ps2 = p;
		}
		if(p->from.sym == s4) {
			p->from.scale = 1;
			ps4 = p;
		}
	}
	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		p = cursym->text;

		if(p->from.scale & NOPROF)	/* dont profile */
			continue;

		/*
		 * JMPL	profin
		 */
		q = prg();
		q->line = p->line;
		q->pc = p->pc;
		q->link = p->link;
		p->link = q;
		p = q;
		p->as = ACALL;
		p->to.type = D_BRANCH;
		p->pcond = ps2;
		p->to.sym = s2;

		for(; p; p=p->link) {
			if(p->as == ARET) {
				/*
				 * RET
				 */
				q = prg();
				q->as = ARET;
				q->from = p->from;
				q->to = p->to;
				q->link = p->link;
				p->link = q;
	
				/*
				 * JAL	profout
				 */
				p->as = ACALL;
				p->from = zprg.from;
				p->to = zprg.to;
				p->to.type = D_BRANCH;
				p->pcond = ps4;
				p->to.sym = s4;
	
				p = q;
			}
		}
	}
}
