// Inferno utils/5c/swt.c
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
	int32 v;
	Prog *sp;

	if(nc >= 3) {
		i = (q+nc-1)->val - (q+0)->val;
		if(!nacl && i > 0 && i < nc*2)
			goto direct;
	}
	if(nc < 5) {
		for(i=0; i<nc; i++) {
			if(debug['W'])
				print("case = %.8ux\n", q->val);
			gopcode(OEQ, nodconst(q->val), n, Z);
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
	gopcode(OGT, nodconst(r->val), n, Z);
	sp = p;
	gopcode(OEQ, nodconst(r->val), n, Z);	/* just gen the B.EQ */
	patch(p, r->label);
	swit2(q, i, def, n);

	if(debug['W'])
		print("case < %.8ux\n", r->val);
	patch(sp, pc);
	swit2(r+1, nc-i-1, def, n);
	return;

direct:
	v = q->val;
	if(v != 0)
		gopcode(OSUB, nodconst(v), Z, n);
	gopcode(OCASE, nodconst((q+nc-1)->val - v), n, Z);
	patch(p, def);
	for(i=0; i<nc; i++) {
		if(debug['W'])
			print("case = %.8ux\n", q->val);
		while(q->val != v) {
			nextpc();
			p->as = ABCASE;
			patch(p, def);
			v++;
		}
		nextpc();
		p->as = ABCASE;
		patch(p, q->label);
		q++;
		v++;
	}
	gbranch(OGOTO);		/* so that regopt() won't be confused */
	patch(p, def);
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
		gopcode(OAS, n2, Z, n3);
		gopcode(OAS, n3, Z, n1);
	} else {
		regalloc(n1, l, nn);
		cgen(l, n1);
	}
	if(b->type->shift == 0 && typeu[b->type->etype]) {
		v = ~0 + (1L << b->type->nbits);
		gopcode(OAND, nodconst(v), Z, n1);
	} else {
		sh = 32 - b->type->shift - b->type->nbits;
		if(sh > 0)
			gopcode(OASHL, nodconst(sh), Z, n1);
		sh += b->type->shift;
		if(sh > 0)
			if(typeu[b->type->etype])
				gopcode(OLSHR, nodconst(sh), Z, n1);
			else
				gopcode(OASHR, nodconst(sh), Z, n1);
	}
}

void
bitstore(Node *b, Node *n1, Node *n2, Node *n3, Node *nn)
{
	int32 v;
	Node nod, *l;
	int sh;

	/*
	 * n1 has adjusted/masked value
	 * n2 has address of cell
	 * n3 has contents of cell
	 */
	l = b->left;
	regalloc(&nod, l, Z);
	v = ~0 + (1L << b->type->nbits);
	gopcode(OAND, nodconst(v), Z, n1);
	gopcode(OAS, n1, Z, &nod);
	if(nn != Z)
		gopcode(OAS, n1, Z, nn);
	sh = b->type->shift;
	if(sh > 0)
		gopcode(OASHL, nodconst(sh), Z, &nod);
	v <<= sh;
	gopcode(OAND, nodconst(~v), Z, n3);
	gopcode(OOR, n3, Z, &nod);
	gopcode(OAS, &nod, Z, n2);

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
			p->reg = NSNAME;
			p->to.type = D_SCONST;
			memmove(p->to.u.sval, string, NSNAME);
			mnstring = 0;
		}
		n--;
	}
	return r;
}

int
mulcon(Node *n, Node *nn)
{
	Node *l, *r, nod1, nod2;
	Multab *m;
	int32 v, vs;
	int o;
	char code[sizeof(m->code)+2], *p;

	if(typefd[n->type->etype])
		return 0;
	l = n->left;
	r = n->right;
	if(l->op == OCONST) {
		l = r;
		r = n->left;
	}
	if(r->op != OCONST)
		return 0;
	v = convvtox(r->vconst, n->type->etype);
	if(v != r->vconst) {
		if(debug['M'])
			print("%L multiply conv: %lld\n", n->lineno, r->vconst);
		return 0;
	}
	m = mulcon0(v);
	if(!m) {
		if(debug['M'])
			print("%L multiply table: %lld\n", n->lineno, r->vconst);
		return 0;
	}
	if(debug['M'] && debug['v'])
		print("%L multiply: %d\n", n->lineno, v);

	memmove(code, m->code, sizeof(m->code));
	code[sizeof(m->code)] = 0;

	p = code;
	if(p[1] == 'i')
		p += 2;
	regalloc(&nod1, n, nn);
	cgen(l, &nod1);
	vs = v;
	regalloc(&nod2, n, Z);

loop:
	switch(*p) {
	case 0:
		regfree(&nod2);
		if(vs < 0) {
			gopcode(OAS, &nod1, Z, &nod1);
			gopcode(OSUB, &nod1, nodconst(0), nn);
		} else
			gopcode(OAS, &nod1, Z, nn);
		regfree(&nod1);
		return 1;
	case '+':
		o = OADD;
		goto addsub;
	case '-':
		o = OSUB;
	addsub:	/* number is r,n,l */
		v = p[1] - '0';
		r = &nod1;
		if(v&4)
			r = &nod2;
		n = &nod1;
		if(v&2)
			n = &nod2;
		l = &nod1;
		if(v&1)
			l = &nod2;
		gopcode(o, l, n, r);
		break;
	default: /* op is shiftcount, number is r,l */
		v = p[1] - '0';
		r = &nod1;
		if(v&2)
			r = &nod2;
		l = &nod1;
		if(v&1)
			l = &nod2;
		v = *p - 'a';
		if(v < 0 || v >= 32) {
			diag(n, "mulcon unknown op: %c%c", p[0], p[1]);
			break;
		}
		gopcode(OASHL, nodconst(v), l, r);
		break;
	}
	p += 2;
	goto loop;
}

void
sextern(Sym *s, Node *a, int32 o, int32 w)
{
	int32 e, lw;

	for(e=0; e<w; e+=NSNAME) {
		lw = NSNAME;
		if(w-e < lw)
			lw = w-e;
		gpseudo(ADATA, s, nodconst(0));
		p->from.offset += o+e;
		p->reg = lw;
		p->to.type = D_SCONST;
		memmove(p->to.u.sval, a->cstring+e, lw);
	}
}

void
gextern(Sym *s, Node *a, int32 o, int32 w)
{

	if(a->op == OCONST && typev[a->type->etype]) {
		if(isbigendian)
			gpseudo(ADATA, s, nod32const(a->vconst>>32));
		else
			gpseudo(ADATA, s, nod32const(a->vconst));
		p->from.offset += o;
		p->reg = 4;
		if(isbigendian)
			gpseudo(ADATA, s, nod32const(a->vconst));
		else
			gpseudo(ADATA, s, nod32const(a->vconst>>32));
		p->from.offset += o + 4;
		p->reg = 4;
		return;
	}
	gpseudo(ADATA, s, a);
	p->from.offset += o;
	p->reg = w;
	if(p->to.type == D_OREG)
		p->to.type = D_CONST;
}

void
outcode(void)
{
	Bprint(&outbuf, "go object %s %s %s\n", getgoos(), getgoarch(), getgoversion());
	if(pragcgobuf.to > pragcgobuf.start) {
		Bprint(&outbuf, "\n");
		Bprint(&outbuf, "$$  // exports\n\n");
		Bprint(&outbuf, "$$  // local types\n\n");
		Bprint(&outbuf, "$$  // cgo\n");
		Bprint(&outbuf, "%s", fmtstrflush(&pragcgobuf));
		Bprint(&outbuf, "\n$$\n\n");
	}
	Bprint(&outbuf, "!\n");

	writeobj(ctxt, &outbuf);
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
		o = align(o, t, Ael2, nil);
		o = align(o, t, Ael1, nil);
		w = SZ_LONG;	/* because of a pun in cc/dcl.c:contig() */
		break;
	}
	if(packw != 0 && xround(o, w) != xround(o, packw))
		diag(Z, "#pragma pack changes offset of %T", t);
	o = xround(o, w);
	if(maxalign != nil && *maxalign < w)
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
