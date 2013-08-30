// Derived from Inferno utils/8c/swt.c
// http://code.google.com/p/inferno-os/source/browse/utils/8c/swt.c
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

void
zname(Biobuf *b, Sym *s, int t)
{
	BPUTLE2(b, ANAME);	/* as */
	BPUTC(b, t);		/* type */
	BPUTC(b, s->sym);	/* sym */

	Bputname(b, s);
}

void
zfile(Biobuf *b, char *p, int n)
{
	BPUTLE2(b, ANAME);
	BPUTC(b, D_FILE);
	BPUTC(b, 1);
	BPUTC(b, '<');
	Bwrite(b, p, n);
	BPUTC(b, 0);
}

void
zhist(Biobuf *b, int line, vlong offset)
{
	Addr a;

	BPUTLE2(b, AHISTORY);
	BPUTLE4(b, line);
	zaddr(b, &zprog.from, 0, 0);
	a = zprog.to;
	if(offset != 0) {
		a.offset = offset;
		a.type = D_CONST;
	}
	zaddr(b, &a, 0, 0);
}

void
zaddr(Biobuf *b, Addr *a, int s, int gotype)
{
	int32 l;
	uint64 e;
	int i, t;
	char *n;

	t = 0;
	if(a->index != D_NONE || a->scale != 0)
		t |= T_INDEX;
	if(s != 0)
		t |= T_SYM;
	if(gotype != 0)
		t |= T_GOTYPE;

	switch(a->type) {

	case D_BRANCH:
		if(a->u.branch == nil)
			fatal("unpatched branch");
		a->offset = a->u.branch->loc;

	default:
		t |= T_TYPE;

	case D_NONE:
		if(a->offset != 0)
			t |= T_OFFSET;
		if(a->offset2 != 0)
			t |= T_OFFSET2;
		break;
	case D_FCONST:
		t |= T_FCONST;
		break;
	case D_SCONST:
		t |= T_SCONST;
		break;
	}
	BPUTC(b, t);

	if(t & T_INDEX) {	/* implies index, scale */
		BPUTC(b, a->index);
		BPUTC(b, a->scale);
	}
	if(t & T_OFFSET) {	/* implies offset */
		l = a->offset;
		BPUTLE4(b, l);
	}
	if(t & T_OFFSET2) {	/* implies offset */
		l = a->offset2;
		BPUTLE4(b, l);
	}
	if(t & T_SYM)		/* implies sym */
		BPUTC(b, s);
	if(t & T_FCONST) {
		ieeedtod(&e, a->u.dval);
		BPUTLE4(b, e);
		BPUTLE4(b, e >> 32);
		return;
	}
	if(t & T_SCONST) {
		n = a->u.sval;
		for(i=0; i<NSNAME; i++) {
			BPUTC(b, *n);
			n++;
		}
		return;
	}
	if(t & T_TYPE)
		BPUTC(b, a->type);
	if(t & T_GOTYPE)
		BPUTC(b, gotype);
}

static struct {
	struct { Sym *sym; short type; } h[NSYM];
	int sym;
} z;

static void
zsymreset(void)
{
	for(z.sym=0; z.sym<NSYM; z.sym++) {
		z.h[z.sym].sym = S;
		z.h[z.sym].type = 0;
	}
	z.sym = 1;
}

static int
zsym(Sym *s, int t, int *new)
{
	int i;

	*new = 0;
	if(s == S)
		return 0;

	i = s->sym;
	if(i < 0 || i >= NSYM)
		i = 0;
	if(z.h[i].type == t && z.h[i].sym == s)
		return i;
	i = z.sym;
	s->sym = i;
	zname(bout, s, t);
	z.h[i].sym = s;
	z.h[i].type = t;
	if(++z.sym >= NSYM)
		z.sym = 1;
	*new = 1;
	return i;
}

static int
zsymaddr(Addr *a, int *new)
{
	int t;

	t = a->type;
	if(t == D_ADDR)
		t = a->index;
	return zsym(a->sym, t, new);
}

void
dumpfuncs(void)
{
	Plist *pl;
	int sf, st, gf, gt, new;
	Sym *s;
	Prog *p;

	zsymreset();

	// fix up pc
	pcloc = 0;
	for(pl=plist; pl!=nil; pl=pl->link) {
		if(isblank(pl->name))
			continue;
		for(p=pl->firstpc; p!=P; p=p->link) {
			p->loc = pcloc;
			if(p->as != ADATA && p->as != AGLOBL)
				pcloc++;
		}
	}

	// put out functions
	for(pl=plist; pl!=nil; pl=pl->link) {
		if(isblank(pl->name))
			continue;

		// -S prints code; -SS prints code and data
		if(debug['S'] && (pl->name || debug['S']>1)) {
			s = S;
			if(pl->name != N)
				s = pl->name->sym;
			print("\n--- prog list \"%S\" ---\n", s);
			for(p=pl->firstpc; p!=P; p=p->link)
				print("%P\n", p);
		}

		for(p=pl->firstpc; p!=P; p=p->link) {
			for(;;) {
				sf = zsymaddr(&p->from, &new);
				gf = zsym(p->from.gotype, D_EXTERN, &new);
				if(new && sf == gf)
					continue;
				st = zsymaddr(&p->to, &new);
				if(new && (st == sf || st == gf))
					continue;
				gt = zsym(p->to.gotype, D_EXTERN, &new);
				if(new && (gt == sf || gt == gf || gt == st))
					continue;
				break;
			}

			BPUTLE2(bout, p->as);
			BPUTLE4(bout, p->lineno);
			zaddr(bout, &p->from, sf, gf);
			zaddr(bout, &p->to, st, gt);
		}
	}
}

int
dsname(Sym *s, int off, char *t, int n)
{
	Prog *p;

	p = gins(ADATA, N, N);
	p->from.type = D_EXTERN;
	p->from.index = D_NONE;
	p->from.offset = off;
	p->from.scale = n;
	p->from.sym = s;
	
	p->to.type = D_SCONST;
	p->to.index = D_NONE;
	memmove(p->to.u.sval, t, n);
	return off + n;
}

/*
 * make a refer to the data s, s+len
 * emitting DATA if needed.
 */
void
datastring(char *s, int len, Addr *a)
{
	Sym *sym;
	
	sym = stringsym(s, len);
	a->type = D_EXTERN;
	a->sym = sym;
	a->node = sym->def;
	a->offset = widthptr+4;  // skip header
	a->etype = TINT32;
}

/*
 * make a refer to the string sval,
 * emitting DATA if needed.
 */
void
datagostring(Strlit *sval, Addr *a)
{
	Sym *sym;
	
	sym = stringsym(sval->s, sval->len);
	a->type = D_EXTERN;
	a->sym = sym;
	a->node = sym->def;
	a->offset = 0;  // header
	a->etype = TINT32;
}

void
gdata(Node *nam, Node *nr, int wid)
{
	Prog *p;
	vlong v;

	if(nr->op == OLITERAL) {
		switch(nr->val.ctype) {
		case CTCPLX:
			gdatacomplex(nam, nr->val.u.cval);
			return;
		case CTSTR:
			gdatastring(nam, nr->val.u.sval);
			return;
		}
	}

	if(wid == 8 && is64(nr->type)) {
		v = mpgetfix(nr->val.u.xval);
		p = gins(ADATA, nam, nodintconst(v));
		p->from.scale = 4;
		p = gins(ADATA, nam, nodintconst(v>>32));
		p->from.scale = 4;
		p->from.offset += 4;
		return;
	}
	p = gins(ADATA, nam, nr);
	p->from.scale = wid;
}

void
gdatacomplex(Node *nam, Mpcplx *cval)
{
	Prog *p;
	int w;

	w = cplxsubtype(nam->type->etype);
	w = types[w]->width;

	p = gins(ADATA, nam, N);
	p->from.scale = w;
	p->to.type = D_FCONST;
	p->to.u.dval = mpgetflt(&cval->real);

	p = gins(ADATA, nam, N);
	p->from.scale = w;
	p->from.offset += w;
	p->to.type = D_FCONST;
	p->to.u.dval = mpgetflt(&cval->imag);
}

void
gdatastring(Node *nam, Strlit *sval)
{
	Prog *p;
	Node nod1;

	p = gins(ADATA, nam, N);
	datastring(sval->s, sval->len, &p->to);
	p->from.scale = types[tptr]->width;
	p->to.index = p->to.type;
	p->to.type = D_ADDR;
//print("%P\n", p);

	nodconst(&nod1, types[TINT32], sval->len);
	p = gins(ADATA, nam, &nod1);
	p->from.scale = types[TINT32]->width;
	p->from.offset += types[tptr]->width;
}

int
dstringptr(Sym *s, int off, char *str)
{
	Prog *p;

	off = rnd(off, widthptr);
	p = gins(ADATA, N, N);
	p->from.type = D_EXTERN;
	p->from.index = D_NONE;
	p->from.sym = s;
	p->from.offset = off;
	p->from.scale = widthptr;

	datastring(str, strlen(str)+1, &p->to);
	p->to.index = p->to.type;
	p->to.type = D_ADDR;
	p->to.etype = TINT32;
	off += widthptr;

	return off;
}

int
dgostrlitptr(Sym *s, int off, Strlit *lit)
{
	Prog *p;

	if(lit == nil)
		return duintptr(s, off, 0);

	off = rnd(off, widthptr);
	p = gins(ADATA, N, N);
	p->from.type = D_EXTERN;
	p->from.index = D_NONE;
	p->from.sym = s;
	p->from.offset = off;
	p->from.scale = widthptr;
	datagostring(lit, &p->to);
	p->to.index = p->to.type;
	p->to.type = D_ADDR;
	p->to.etype = TINT32;
	off += widthptr;

	return off;
}

int
dgostringptr(Sym *s, int off, char *str)
{
	int n;
	Strlit *lit;

	if(str == nil)
		return duintptr(s, off, 0);

	n = strlen(str);
	lit = mal(sizeof *lit + n);
	strcpy(lit->s, str);
	lit->len = n;
	return dgostrlitptr(s, off, lit);
}


int
duintxx(Sym *s, int off, uint64 v, int wid)
{
	Prog *p;

	off = rnd(off, wid);

	p = gins(ADATA, N, N);
	p->from.type = D_EXTERN;
	p->from.index = D_NONE;
	p->from.sym = s;
	p->from.offset = off;
	p->from.scale = wid;
	p->to.type = D_CONST;
	p->to.index = D_NONE;
	p->to.offset = v;
	off += wid;

	return off;
}

int
dsymptr(Sym *s, int off, Sym *x, int xoff)
{
	Prog *p;

	off = rnd(off, widthptr);

	p = gins(ADATA, N, N);
	p->from.type = D_EXTERN;
	p->from.index = D_NONE;
	p->from.sym = s;
	p->from.offset = off;
	p->from.scale = widthptr;
	p->to.type = D_ADDR;
	p->to.index = D_EXTERN;
	p->to.sym = x;
	p->to.offset = xoff;
	off += widthptr;

	return off;
}

void
nopout(Prog *p)
{
	p->as = ANOP;
}

