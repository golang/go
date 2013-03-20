// Inferno utils/6c/swt.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/swt.c
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

	regalloc(&nreg, n, Z);
	if(typev[n->type->etype])
		nreg.type = types[TVLONG];
	else
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
				print("case = %.8llux\n", q->val);
			gcmp(OEQ, n, q->val);
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
		print("case > %.8llux\n", r->val);
	gcmp(OGT, n, r->val);
	sp = p;
	gbranch(OGOTO);
	p->as = AJEQ;
	patch(p, r->label);
	swit2(q, i, def, n);

	if(debug['W'])
		print("case < %.8llux\n", r->val);
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
		gopcode(OAND, tfield, nodconst(v), n1);
	} else {
		sh = 32 - b->type->shift - b->type->nbits;
		if(sh > 0)
			gopcode(OASHL, tfield, nodconst(sh), n1);
		sh += b->type->shift;
		if(sh > 0)
			if(typeu[b->type->etype])
				gopcode(OLSHR, tfield, nodconst(sh), n1);
			else
				gopcode(OASHR, tfield, nodconst(sh), n1);
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
			memmove(p->to.sval, string, NSNAME);
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
		memmove(p->to.sval, a->cstring+e, lw);
	}
}

void
gextern(Sym *s, Node *a, int32 o, int32 w)
{
	if(0 && a->op == OCONST && typev[a->type->etype]) {
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

void	zname(Biobuf*, Sym*, int);
void	zaddr(Biobuf*, Adr*, int);
void	outhist(Biobuf*);

void
outcode(void)
{
	struct { Sym *sym; short type; } h[NSYM];
	Prog *p;
	Sym *s;
	int f, sf, st, t, sym;
	Biobuf b;

	if(debug['S']) {
		for(p = firstp; p != P; p = p->link)
			if(p->as != ADATA && p->as != AGLOBL)
				pc--;
		for(p = firstp; p != P; p = p->link) {
			print("%P\n", p);
			if(p->as != ADATA && p->as != AGLOBL)
				pc++;
		}
	}

	f = open(outfile, OWRITE);
	if(f < 0) {
		diag(Z, "cannot open %s", outfile);
		return;
	}
	Binit(&b, f, OWRITE);

	Bprint(&b, "go object %s %s %s\n", getgoos(), thestring, getgoversion());
	if(pragcgobuf.to > pragcgobuf.start) {
		Bprint(&b, "\n");
		Bprint(&b, "$$  // exports\n\n");
		Bprint(&b, "$$  // local types\n\n");
		Bprint(&b, "$$  // cgo\n");
		Bprint(&b, "%s", fmtstrflush(&pragcgobuf));
		Bprint(&b, "\n$$\n\n");
	}
	Bprint(&b, "!\n");

	outhist(&b);
	for(sym=0; sym<NSYM; sym++) {
		h[sym].sym = S;
		h[sym].type = 0;
	}
	sym = 1;
	for(p = firstp; p != P; p = p->link) {
	jackpot:
		sf = 0;
		s = p->from.sym;
		while(s != S) {
			sf = s->sym;
			if(sf < 0 || sf >= NSYM)
				sf = 0;
			t = p->from.type;
			if(t == D_ADDR)
				t = p->from.index;
			if(h[sf].type == t)
			if(h[sf].sym == s)
				break;
			s->sym = sym;
			zname(&b, s, t);
			h[sym].sym = s;
			h[sym].type = t;
			sf = sym;
			sym++;
			if(sym >= NSYM)
				sym = 1;
			break;
		}
		st = 0;
		s = p->to.sym;
		while(s != S) {
			st = s->sym;
			if(st < 0 || st >= NSYM)
				st = 0;
			t = p->to.type;
			if(t == D_ADDR)
				t = p->to.index;
			if(h[st].type == t)
			if(h[st].sym == s)
				break;
			s->sym = sym;
			zname(&b, s, t);
			h[sym].sym = s;
			h[sym].type = t;
			st = sym;
			sym++;
			if(sym >= NSYM)
				sym = 1;
			if(st == sf)
				goto jackpot;
			break;
		}
		Bputc(&b, p->as);
		Bputc(&b, p->as>>8);
		Bputc(&b, p->lineno);
		Bputc(&b, p->lineno>>8);
		Bputc(&b, p->lineno>>16);
		Bputc(&b, p->lineno>>24);
		zaddr(&b, &p->from, sf);
		zaddr(&b, &p->to, st);
	}
	Bterm(&b);
	close(f);
	firstp = P;
	lastp = P;
}

void
outhist(Biobuf *b)
{
	Hist *h;
	char *p, *q, *op, c;
	Prog pg;
	int n;
	char *tofree;
	static int first = 1;
	static char *goroot, *goroot_final;

	if(first) {
		// Decide whether we need to rewrite paths from $GOROOT to $GOROOT_FINAL.
		first = 0;
		goroot = getenv("GOROOT");
		goroot_final = getenv("GOROOT_FINAL");
		if(goroot == nil)
			goroot = "";
		if(goroot_final == nil)
			goroot_final = goroot;
		if(strcmp(goroot, goroot_final) == 0) {
			goroot = nil;
			goroot_final = nil;
		}
	}

	tofree = nil;
	pg = zprog;
	pg.as = AHISTORY;
	c = pathchar();
	for(h = hist; h != H; h = h->link) {
		p = h->name;
		if(p != nil && goroot != nil) {
			n = strlen(goroot);
			if(strncmp(p, goroot, strlen(goroot)) == 0 && p[n] == '/') {
				tofree = smprint("%s%s", goroot_final, p+n);
				p = tofree;
			}
		}
		op = 0;
		if(systemtype(Windows) && p && p[1] == ':'){
			c = p[2];
		} else if(p && p[0] != c && h->offset == 0 && pathname){
			if(systemtype(Windows) && pathname[1] == ':') {
				op = p;
				p = pathname;
				c = p[2];
			} else if(pathname[0] == c){
				op = p;
				p = pathname;
			}
		}
		while(p) {
			q = utfrune(p, c);
			if(q) {
				n = q-p;
				if(n == 0){
					n = 1;	/* leading "/" */
					*p = '/';	/* don't emit "\" on windows */
				}
				q++;
			} else {
				n = strlen(p);
				q = 0;
			}
			if(n) {
				Bputc(b, ANAME);
				Bputc(b, ANAME>>8);
				Bputc(b, D_FILE);
				Bputc(b, 1);
				Bputc(b, '<');
				Bwrite(b, p, n);
				Bputc(b, 0);
			}
			p = q;
			if(p == 0 && op) {
				p = op;
				op = 0;
			}
		}
		pg.lineno = h->line;
		pg.to.type = zprog.to.type;
		pg.to.offset = h->offset;
		if(h->offset)
			pg.to.type = D_CONST;

		Bputc(b, pg.as);
		Bputc(b, pg.as>>8);
		Bputc(b, pg.lineno);
		Bputc(b, pg.lineno>>8);
		Bputc(b, pg.lineno>>16);
		Bputc(b, pg.lineno>>24);
		zaddr(b, &pg.from, 0);
		zaddr(b, &pg.to, 0);

		if(tofree) {
			free(tofree);
			tofree = nil;
		}
	}
}

void
zname(Biobuf *b, Sym *s, int t)
{
	char *n;
	uint32 sig;

	if(debug['T'] && t == D_EXTERN && s->sig != SIGDONE && s->type != types[TENUM] && s != symrathole){
		sig = sign(s);
		Bputc(b, ASIGNAME);
		Bputc(b, ASIGNAME>>8);
		Bputc(b, sig);
		Bputc(b, sig>>8);
		Bputc(b, sig>>16);
		Bputc(b, sig>>24);
		s->sig = SIGDONE;
	}
	else{
		Bputc(b, ANAME);	/* as */
		Bputc(b, ANAME>>8);	/* as */
	}
	Bputc(b, t);			/* type */
	Bputc(b, s->sym);		/* sym */
	n = s->name;
	while(*n) {
		Bputc(b, *n);
		n++;
	}
	Bputc(b, 0);
}

void
zaddr(Biobuf *b, Adr *a, int s)
{
	int32 l;
	int i, t;
	char *n;
	Ieee e;

	t = 0;
	if(a->index != D_NONE || a->scale != 0)
		t |= T_INDEX;
	if(s != 0)
		t |= T_SYM;

	switch(a->type) {
	default:
		t |= T_TYPE;
	case D_NONE:
		if(a->offset != 0) {
			t |= T_OFFSET;
			l = a->offset;
			if((vlong)l != a->offset)
				t |= T_64;
		}
		break;
	case D_FCONST:
		t |= T_FCONST;
		break;
	case D_SCONST:
		t |= T_SCONST;
		break;
	}
	Bputc(b, t);

	if(t & T_INDEX) {	/* implies index, scale */
		Bputc(b, a->index);
		Bputc(b, a->scale);
	}
	if(t & T_OFFSET) {	/* implies offset */
		l = a->offset;
		Bputc(b, l);
		Bputc(b, l>>8);
		Bputc(b, l>>16);
		Bputc(b, l>>24);
		if(t & T_64) {
			l = a->offset>>32;
			Bputc(b, l);
			Bputc(b, l>>8);
			Bputc(b, l>>16);
			Bputc(b, l>>24);
		}
	}
	if(t & T_SYM)		/* implies sym */
		Bputc(b, s);
	if(t & T_FCONST) {
		ieeedtod(&e, a->dval);
		l = e.l;
		Bputc(b, l);
		Bputc(b, l>>8);
		Bputc(b, l>>16);
		Bputc(b, l>>24);
		l = e.h;
		Bputc(b, l);
		Bputc(b, l>>8);
		Bputc(b, l>>16);
		Bputc(b, l>>24);
		return;
	}
	if(t & T_SCONST) {
		n = a->sval;
		for(i=0; i<NSNAME; i++) {
			Bputc(b, *n);
			n++;
		}
		return;
	}
	if(t & T_TYPE)
		Bputc(b, a->type);
}

int32
align(int32 i, Type *t, int op, int32 *maxalign)
{
	int32 o;
	Type *v;
	int w;

	o = i;
	w = 1;
	switch(op) {
	default:
		diag(Z, "unknown align opcode %d", op);
		break;

	case Asu2:	/* padding at end of a struct */
		w = *maxalign;
		if(w < 1)
			w = 1;
		if(packflg)
			w = packflg;
		break;

	case Ael1:	/* initial align of struct element */
		for(v=t; v->etype==TARRAY; v=v->link)
			;
		if(v->etype == TSTRUCT || v->etype == TUNION)
			w = v->align;
		else
			w = ewidth[v->etype];
		if(w < 1 || w > SZ_VLONG)
			fatal(Z, "align");
		if(packflg) 
			w = packflg;
		break;

	case Ael2:	/* width of a struct element */
		o += t->width;
		break;

	case Aarg0:	/* initial passbyptr argument in arg list */
		if(typesu[t->etype]) {
			o = align(o, types[TIND], Aarg1, nil);
			o = align(o, types[TIND], Aarg2, nil);
		}
		break;

	case Aarg1:	/* initial align of parameter */
		w = ewidth[t->etype];
		if(w <= 0 || w >= SZ_VLONG) {
			w = SZ_VLONG;
			break;
		}
		w = 1;		/* little endian no adjustment */
		break;

	case Aarg2:	/* width of a parameter */
		o += t->width;
		w = t->width;
		if(w > SZ_VLONG)
			w = SZ_VLONG;
		break;

	case Aaut3:	/* total align of automatic */
		o = align(o, t, Ael1, nil);
		o = align(o, t, Ael2, nil);
		break;
	}
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
	v = xround(v, SZ_VLONG);
	if(v > max)
		return v;
	return max;
}
