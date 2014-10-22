// Inferno utils/8c/swt.c
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

#include "gc.h"

void
swit1(C1 *q, int nc, int32 def, Node *n)
{
	Node nreg;

	if(typev[n->type->etype]) {
		regsalloc(&nreg, n);
		nreg.type = types[TVLONG];
		cgen(n, &nreg);
		swit2(q, nc, def, &nreg);
		return;
	}

	regalloc(&nreg, n, Z);
	nreg.type = types[TLONG];
	cgen(n, &nreg);
	swit2(q, nc, def, &nreg);
	regfree(&nreg);
}

void
swit2(C1 *q, int nc, int32 def, Node *n)
{
	C1 *r;
	int i;
	Prog *sp;

	if(nc < 5) {
		for(i=0; i<nc; i++) {
			if(debug['W'])
				print("case = %.8ux\n", q->val);
			gopcode(OEQ, n->type, n, nodconst(q->val));
			patch(p, q->label);
			q++;
		}
		gbranch(OGOTO);
		patch(p, def);
		return;
	}
	i = nc / 2;
	r = q+i;
	if(debug['W'])
		print("case > %.8ux\n", r->val);
	gopcode(OGT, n->type, n, nodconst(r->val));
	sp = p;
	gbranch(OGOTO);
	p->as = AJEQ;
	patch(p, r->label);
	swit2(q, i, def, n);

	if(debug['W'])
		print("case < %.8ux\n", r->val);
	patch(sp, pc);
	swit2(r+1, nc-i-1, def, n);
}

void
bitload(Node *b, Node *n1, Node *n2, Node *n3, Node *nn)
{
	int sh;
	int32 v;
	Node *l;

	/*
	 * n1 gets adjusted/masked value
	 * n2 gets address of cell
	 * n3 gets contents of cell
	 */
	l = b->left;
	if(n2 != Z) {
		regalloc(n1, l, nn);
		reglcgen(n2, l, Z);
		regalloc(n3, l, Z);
		gmove(n2, n3);
		gmove(n3, n1);
	} else {
		regalloc(n1, l, nn);
		cgen(l, n1);
	}
	if(b->type->shift == 0 && typeu[b->type->etype]) {
		v = ~0 + (1L << b->type->nbits);
		gopcode(OAND, types[TLONG], nodconst(v), n1);
	} else {
		sh = 32 - b->type->shift - b->type->nbits;
		if(sh > 0)
			gopcode(OASHL, types[TLONG], nodconst(sh), n1);
		sh += b->type->shift;
		if(sh > 0)
			if(typeu[b->type->etype])
				gopcode(OLSHR, types[TLONG], nodconst(sh), n1);
			else
				gopcode(OASHR, types[TLONG], nodconst(sh), n1);
	}
}

void
bitstore(Node *b, Node *n1, Node *n2, Node *n3, Node *nn)
{
	int32 v;
	Node nod;
	int sh;

	regalloc(&nod, b->left, Z);
	v = ~0 + (1L << b->type->nbits);
	gopcode(OAND, types[TLONG], nodconst(v), n1);
	gmove(n1, &nod);
	if(nn != Z)
		gmove(n1, nn);
	sh = b->type->shift;
	if(sh > 0)
		gopcode(OASHL, types[TLONG], nodconst(sh), &nod);
	v <<= sh;
	gopcode(OAND, types[TLONG], nodconst(~v), n3);
	gopcode(OOR, types[TLONG], n3, &nod);
	gmove(&nod, n2);

	regfree(&nod);
	regfree(n1);
	regfree(n2);
	regfree(n3);
}

int32
outstring(char *s, int32 n)
{
	int32 r;

	if(suppress)
		return nstring;
	r = nstring;
	while(n) {
		string[mnstring] = *s++;
		mnstring++;
		nstring++;
		if(mnstring >= NSNAME) {
			gpseudo(ADATA, symstring, nodconst(0L));
			p->from.offset += nstring - NSNAME;
			p->from.scale = NSNAME;
			p->to.type = D_SCONST;
			memmove(p->to.u.sval, string, NSNAME);
			mnstring = 0;
		}
		n--;
	}
	return r;
}

void
sextern(Sym *s, Node *a, int32 o, int32 w)
{
	int32 e, lw;

	for(e=0; e<w; e+=NSNAME) {
		lw = NSNAME;
		if(w-e < lw)
			lw = w-e;
		gpseudo(ADATA, s, nodconst(0L));
		p->from.offset += o+e;
		p->from.scale = lw;
		p->to.type = D_SCONST;
		memmove(p->to.u.sval, a->cstring+e, lw);
	}
}

void
gextern(Sym *s, Node *a, int32 o, int32 w)
{
	if(a->op == OCONST && typev[a->type->etype]) {
		gpseudo(ADATA, s, lo64(a));
		p->from.offset += o;
		p->from.scale = 4;
		gpseudo(ADATA, s, hi64(a));
		p->from.offset += o + 4;
		p->from.scale = 4;
		return;
	}
	gpseudo(ADATA, s, a);
	p->from.offset += o;
	p->from.scale = w;
	switch(p->to.type) {
	default:
		p->to.index = p->to.type;
		p->to.type = D_ADDR;
	case D_CONST:
	case D_FCONST:
	case D_ADDR:
		break;
	}
}

void
outcode(void)
{
	int f;
	Biobuf b;

	f = open(outfile, OWRITE);
	if(f < 0) {
		diag(Z, "cannot open %s", outfile);
		return;
	}
	Binit(&b, f, OWRITE);

	Bprint(&b, "go object %s %s %s\n", getgoos(), getgoarch(), getgoversion());
	if(pragcgobuf.to > pragcgobuf.start) {
		Bprint(&b, "\n");
		Bprint(&b, "$$  // exports\n\n");
		Bprint(&b, "$$  // local types\n\n");
		Bprint(&b, "$$  // cgo\n");
		Bprint(&b, "%s", fmtstrflush(&pragcgobuf));
		Bprint(&b, "\n$$\n\n");
	}
	Bprint(&b, "!\n");

	writeobj(ctxt, &b);
	Bterm(&b);
	close(f);
	lastp = P;
}

int32
align(int32 i, Type *t, int op, int32 *maxalign)
{
	int32 o;
	Type *v;
	int w, packw;

	o = i;
	w = 1;
	packw = 0;
	switch(op) {
	default:
		diag(Z, "unknown align opcode %d", op);
		break;

	case Asu2:	/* padding at end of a struct */
		w = *maxalign;
		if(w < 1)
			w = 1;
		if(packflg)
			packw = packflg;
		break;

	case Ael1:	/* initial align of struct element */
		for(v=t; v->etype==TARRAY; v=v->link)
			;
		if(v->etype == TSTRUCT || v->etype == TUNION)
			w = v->align;
		else {
			w = ewidth[v->etype];
			if(w == 8)
				w = 4;
		}
		if(w < 1 || w > SZ_LONG)
			fatal(Z, "align");
		if(packflg) 
			packw = packflg;
		break;

	case Ael2:	/* width of a struct element */
		o += t->width;
		break;

	case Aarg0:	/* initial passbyptr argument in arg list */
		if(typesuv[t->etype]) {
			o = align(o, types[TIND], Aarg1, nil);
			o = align(o, types[TIND], Aarg2, nil);
		}
		break;

	case Aarg1:	/* initial align of parameter */
		w = ewidth[t->etype];
		if(w <= 0 || w >= SZ_LONG) {
			w = SZ_LONG;
			break;
		}
		w = 1;		/* little endian no adjustment */
		break;

	case Aarg2:	/* width of a parameter */
		o += t->width;
		w = t->width;
		if(w > SZ_LONG)
			w = SZ_LONG;
		break;

	case Aaut3:	/* total align of automatic */
		o = align(o, t, Ael1, nil);
		o = align(o, t, Ael2, nil);
		break;
	}
	if(packw != 0 && xround(o, w) != xround(o, packw))
		diag(Z, "#pragma pack changes offset of %T", t);
	o = xround(o, w);
	if(maxalign && *maxalign < w)
		*maxalign = w;
	if(debug['A'])
		print("align %s %d %T = %d\n", bnames[op], i, t, o);
	return o;
}

int32
maxround(int32 max, int32 v)
{
	v = xround(v, SZ_LONG);
	if(v > max)
		return v;
	return max;
}
