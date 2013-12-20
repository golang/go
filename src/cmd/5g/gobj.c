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
#include "gg.h"

int
dsname(Sym *sym, int off, char *t, int n)
{
	Prog *p;

	p = gins(ADATA, N, N);
	p->from.type = D_OREG;
	p->from.name = D_EXTERN;
	p->from.etype = TINT32;
	p->from.offset = off;
	p->from.reg = NREG;
	p->from.sym = linksym(sym);
	
	p->reg = n;
	
	p->to.type = D_SCONST;
	p->to.name = D_NONE;
	p->to.reg = NREG;
	p->to.offset = 0;
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
	a->type = D_OREG;
	a->name = D_EXTERN;
	a->etype = TINT32;
	a->offset = widthptr+4;  // skip header
	a->reg = NREG;
	a->sym = linksym(sym);
	a->node = sym->def;
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
	a->type = D_OREG;
	a->name = D_EXTERN;
	a->etype = TINT32;
	a->offset = 0;  // header
	a->reg = NREG;
	a->sym = linksym(sym);
	a->node = sym->def;
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
		p->reg = 4;
		p = gins(ADATA, nam, nodintconst(v>>32));
		p->reg = 4;
		p->from.offset += 4;
		return;
	}
	p = gins(ADATA, nam, nr);
	p->reg = wid;
}

void
gdatacomplex(Node *nam, Mpcplx *cval)
{
	Prog *p;
	int w;

	w = cplxsubtype(nam->type->etype);
	w = types[w]->width;

	p = gins(ADATA, nam, N);
	p->reg = w;
	p->to.type = D_FCONST;
	p->to.u.dval = mpgetflt(&cval->real);

	p = gins(ADATA, nam, N);
	p->reg = w;
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
	p->reg = types[tptr]->width;
	p->to.type = D_CONST;
	p->to.etype = TINT32;
//print("%P\n", p);

	nodconst(&nod1, types[TINT32], sval->len);
	p = gins(ADATA, nam, &nod1);
	p->reg = types[TINT32]->width;
	p->from.offset += types[tptr]->width;
}

int
dstringptr(Sym *s, int off, char *str)
{
	Prog *p;

	off = rnd(off, widthptr);
	p = gins(ADATA, N, N);
	p->from.type = D_OREG;
	p->from.name = D_EXTERN;
	p->from.sym = linksym(s);
	p->from.offset = off;
	p->reg = widthptr;

	datastring(str, strlen(str)+1, &p->to);
	p->to.type = D_CONST;
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
	p->from.type = D_OREG;
	p->from.name = D_EXTERN;
	p->from.sym = linksym(s);
	p->from.offset = off;
	p->reg = widthptr;
	datagostring(lit, &p->to);
	p->to.type = D_CONST;
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
dsymptr(Sym *s, int off, Sym *x, int xoff)
{
	Prog *p;

	off = rnd(off, widthptr);

	p = gins(ADATA, N, N);
	p->from.type = D_OREG;
	p->from.name = D_EXTERN;
	p->from.sym = linksym(s);
	p->from.offset = off;
	p->reg = widthptr;
	p->to.type = D_CONST;
	p->to.name = D_EXTERN;
	p->to.sym = linksym(x);
	p->to.offset = xoff;
	off += widthptr;

	return off;
}

void
nopout(Prog *p)
{
	p->as = ANOP;
}
