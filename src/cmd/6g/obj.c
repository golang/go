// Derived from Inferno utils/6c/swt.c
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

#include "gg.h"

void
dumpobj(void)
{
	Plist *pl;
	Prog *p;
	Dcl *d;
	Sym *s;
	Node *n;
	struct { Sym *sym; short type; } h[NSYM];
	int sf, st, t, sym;
	Node n1;

	// add nil plist w AEND
	newplist();

	bout = Bopen(outfile, OWRITE);
	if(bout == nil)
		fatal("cant open %s", outfile);

	Bprint(bout, "amd64\n");
	Bprint(bout, "  exports automatically generated from\n");
	Bprint(bout, "  %s in package \"%s\"\n", curio.infile, package);
	dumpexport();
	Bprint(bout, "\n!\n");

	outhist(bout);

	// add globals
	nodconst(&n1, types[TINT32], 0);
	for(d=externdcl; d!=D; d=d->forw) {
		if(d->op != ONAME)
			continue;

		s = d->dsym;
		if(s == S)
			fatal("external nil");
		n = d->dnode;
		if(n == N || n->type == T)
			fatal("external %S nil\n", s);

		if(n->type->etype == TFUNC)
			continue;

		dowidth(n->type);
		mpmovecfix(n1.val.u.xval, n->type->width);

		p = pc;
		gins(AGLOBL, s->oname, &n1);
		p->lineno = s->oname->lineno;
	}

	dumpstrings();
	dumpsignatures();

	for(sym=0; sym<NSYM; sym++) {
		h[sym].sym = S;
		h[sym].type = 0;
	}
	sym = 1;

	// put out functions
	for(pl=plist; pl!=nil; pl=pl->link) {

		if(debug['S']) {
			s = S;
			if(pl->name != N)
				s = pl->name->sym;
			print("\n--- prog list \"%S\" ---\n", s);
			for(p=pl->firstpc; p!=P; p=p->link)
				print("%P\n", p);
		}

		for(p=pl->firstpc; p!=P; p=p->link) {
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
				zname(bout, s, t);
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
				zname(bout, s, t);
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
			Bputc(bout, p->as);
			Bputc(bout, p->as>>8);
			Bputc(bout, p->lineno);
			Bputc(bout, p->lineno>>8);
			Bputc(bout, p->lineno>>16);
			Bputc(bout, p->lineno>>24);
			zaddr(bout, &p->from, sf);
			zaddr(bout, &p->to, st);
		}
	}
	Bterm(bout);
return;
	Bterm(bout);
}

Bputdot(Biobuf *b)
{
	// put out middle dot ·
	Bputc(b, 0xc2);
	Bputc(b, 0xb7);
}

void
zname(Biobuf *b, Sym *s, int t)
{
	char *n;

	Bputc(b, ANAME);	/* as */
	Bputc(b, ANAME>>8);	/* as */
	Bputc(b, t);		/* type */
	Bputc(b, s->sym);	/* sym */

	for(n=s->opackage; *n; n++)
		Bputc(b, *n);
	Bputdot(b);
	for(n=s->name; *n; n++)
		Bputc(b, *n);
	Bputc(b, 0);
}

void
zaddr(Biobuf *b, Addr *a, int s)
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

void
outhist(Biobuf *b)
{
	Hist *h;
	char *p, *q, *op;
	Prog pg;
	int n;

	pg = zprog;
	pg.as = AHISTORY;
	for(h = hist; h != H; h = h->link) {
		p = h->name;
		op = 0;

		if(p && p[0] != '/' && h->offset == 0 && pathname && pathname[0] == '/') {
			op = p;
			p = pathname;
		}

		while(p) {
			q = utfrune(p, '/');
			if(q) {
				n = q-p;
				if(n == 0)
					n = 1;		// leading "/"
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
	}
}

void
ieeedtod(Ieee *ieee, double native)
{
	double fr, ho, f;
	int exp;

	if(native < 0) {
		ieeedtod(ieee, -native);
		ieee->h |= 0x80000000L;
		return;
	}
	if(native == 0) {
		ieee->l = 0;
		ieee->h = 0;
		return;
	}
	fr = frexp(native, &exp);
	f = 2097152L;		/* shouldnt use fp constants here */
	fr = modf(fr*f, &ho);
	ieee->h = ho;
	ieee->h &= 0xfffffL;
	ieee->h |= (exp+1022L) << 20;
	f = 65536L;
	fr = modf(fr*f, &ho);
	ieee->l = ho;
	ieee->l <<= 16;
	ieee->l |= (int32)(fr*f);
}

void
datastring(char *s, int len)
{
	int w;
	Prog *p;
	Addr ac, ao;

	// string
	memset(&ao, 0, sizeof(ao));
	ao.type = D_STATIC;
	ao.index = D_NONE;
	ao.etype = TINT32;
	ao.sym = symstringo;
	ao.offset = 0;		// fill in

	// constant
	memset(&ac, 0, sizeof(ac));
	ac.type = D_CONST;
	ac.index = D_NONE;
	ac.offset = 0;		// fill in

	for(w=0; w<len; w+=8) {
		p = pc;
		gins(ADATA, N, N);

		// .stringo<>+oo, [NSNAME], $"xxx"
		p->from = ao;
		p->from.offset = stringo;

		p->from.scale = NSNAME;
		if(w+8 > len)
			p->from.scale = len-w;

		p->to = ac;
		p->to.type = D_SCONST;
		p->to.offset = len;
		memmove(p->to.sval, s+w, p->from.scale);
		stringo += p->from.scale;
	}
}

void
dumpstrings(void)
{
	Pool *l;
	Prog *p;
	Addr ac, ao;
	int32 wi;

	if(poolist == nil)
		return;

	memset(&ac, 0, sizeof(ac));
	memset(&ao, 0, sizeof(ao));

	// constant
	ac.type = D_CONST;
	ac.index = D_NONE;
	ac.offset = 0;			// fill in

	// string len+ptr
	ao.type = D_STATIC;
	ao.index = D_NONE;
	ao.etype = TINT32;
	ao.sym = symstringo;
	ao.offset = 0;			// fill in

	wi = types[TINT32]->width;

	// lay out (count+string)
	for(l=poolist; l!=nil; l=l->link) {

		p = pc;
		gins(ADATA, N, N);

		// .stringo<>+xx, wi, $len
		stringo = rnd(stringo, wi);
		p->from = ao;
		p->from.offset = stringo;
		p->from.scale = wi;
		p->to = ac;
		p->to.offset = l->sval->len;
		stringo += wi;

		datastring(l->sval->s, l->sval->len);
	}
}

static int
sigcmp(Sig *a, Sig *b)
{
	return strcmp(a->name, b->name);
}

void
dumpsignatures(void)
{
	Dcl *d, *x;
	Type *t, *f;
	Sym *s1, *s;
	int et, o, wi, ot;
	Sig *a, *b;
	Addr at, ao, ac, ad;
	Prog *p;
	char *sp;

	// copy externdcl list to signatlist
	for(d=externdcl; d!=D; d=d->forw) {
		if(d->op != OTYPE)
			continue;

		t = d->dtype;
		if(t == T)
			continue;

		et = t->etype;
		if(et != TSTRUCT && et != TINTER)
			continue;

		s = d->dsym;
		if(s == S)
			continue;

		if(s->name[0] == '_')
			continue;

		// if it was imported
		if(s->local == 0)
			continue;

// until i can figure out
// when/if it is used, do them all
//		// if not used and not exported
//		if(s->local == 1 && !s->export)
//			continue;

		x = mal(sizeof(*d));
		x->op = OTYPE;
		x->dsym = s;
		x->dtype = t;
		x->forw = signatlist;
		signatlist = x;
	}

	/*
	 * put all the names into a linked
	 * list so that it may be generated in sorted order.
	 * the runtime will be linear rather than quadradic
	 */

	memset(&at, 0, sizeof(at));
	memset(&ao, 0, sizeof(ao));
	memset(&ac, 0, sizeof(ac));
	memset(&ad, 0, sizeof(ad));

	// sig structure
	at.type = D_EXTERN;
	at.index = D_NONE;
	at.sym = S;			// fill in
	at.offset = 0;			// fill in

	// $string
	ao.type = D_ADDR;
	ao.index = D_STATIC;
	ao.etype = TINT32;
	ao.sym = symstringo;
	ao.offset = 0;			// fill in

	// constant
	ac.type = D_CONST;
	ac.index = D_NONE;
	ac.offset = 0;			// fill in

	// $method
	ad.type = D_ADDR;
	ad.index = D_EXTERN;
	ad.sym = S;			// fill in
	ad.offset = 0;

	wi = types[TINT32]->width;

	for(d=signatlist; d!=D; d=d->forw) {
		if(d->op != OTYPE)
			continue;

		t = d->dtype;
		et = t->etype;
		if(et != TSTRUCT && et != TINTER)
			continue;

		s = d->dsym;
		if(s == S)
			continue;

		if(s->name[0] == '_')
			continue;

		if(strcmp(s->opackage, package) != 0)
			continue;

		at.sym = signame(t);

		a = nil;
		o = 0;
		for(f=t->type; f!=T; f=f->down) {
			if(f->type->etype != TFUNC)
				continue;

			if(f->etype != TFIELD)
				fatal("dumpsignatures: not field");

			s1 = f->sym;
			if(s1 == nil)
				continue;
			if(s1->name[0] == '_')
				continue;

			b = mal(sizeof(*b));
			b->link = a;
			a = b;

			a->name = s1->name;
			sp = strchr(s1->name, '_');
			if(sp != nil)
				a->name = sp+1;
			
			a->hash = PRIME8*stringhash(a->name) + PRIME9*typehash(f->type, 0);
			a->sym = f->sym;
			a->offset = o;
			o++;
		}

		a = lsort(a, sigcmp);
		ot = 0;

		if(et == TINTER) {
			o = 0;
			for(b=a; b!=nil; b=b->link)
				o++;

			// sigi[0].name = ""
			ot = rnd(ot, maxround);
			p = pc;
			gins(ADATA, N, N);
			p->from = at;
			p->from.offset = ot;
			p->from.scale = widthptr;
			p->to = ao;
			p->to.offset = stringo;
			ot += widthptr;

			// sigi[0].hash = 0
			ot = rnd(ot, wi);
			p = pc;
			gins(ADATA, N, N);
			p->from = at;
			p->from.offset = ot;
			p->from.scale = wi;
			p->to = ac;
			p->to.offset = 0;
			ot += wi;

			// sigi[0].offset = count
			ot = rnd(ot, wi);
			p = pc;
			gins(ADATA, N, N);
			p->from = at;
			p->from.offset = ot;
			p->from.scale = wi;
			p->to = ac;
			p->to.offset = o;
			ot += wi;

			datastring("", 1);

		}

		for(b=a; b!=nil; b=b->link) {

			// sigx[++].name = "fieldname"
			ot = rnd(ot, maxround);
			p = pc;
			gins(ADATA, N, N);
			p->from = at;
			p->from.offset = ot;
			p->from.scale = widthptr;
			p->to = ao;
			p->to.offset = stringo;
			ot += widthptr;

			// sigx[++].hash = hashcode
			ot = rnd(ot, wi);
			p = pc;
			gins(ADATA, N, N);
			p->from = at;
			p->from.offset = ot;
			p->from.scale = wi;
			p->to = ac;
			p->to.offset = b->hash;
			ot += wi;

			if(et == TINTER) {
				// sigi[++].offset = offset of method
				ot = rnd(ot, wi);
				p = pc;
				gins(ADATA, N, N);
				p->from = at;
				p->from.offset = ot;
				p->from.scale = wi;
				p->to = ac;
				p->to.offset = b->offset;
				ot += wi;
			} else {
				// sigs[++].fun = &method
				ot = rnd(ot, widthptr);
				p = pc;
				gins(ADATA, N, N);
				p->from = at;
				p->from.offset = ot;
				p->from.scale = widthptr;
				p->to = ad;
				p->to.sym = b->sym;
				ot += widthptr;
			}
			datastring(b->name, strlen(b->name)+1);
		}

		// nil field name at end
		ot = rnd(ot, maxround);
		p = pc;
		gins(ADATA, N, N);
		p->from = at;
		p->from.offset = ot;
		p->from.scale = widthptr;
		p->to = ac;
		p->to.offset = 0;
		ot += widthptr;

		p = pc;
		gins(AGLOBL, N, N);
		p->from = at;
		p->to = ac;
		p->to.offset = ot;
	}

	if(stringo > 0) {
		p = pc;
		gins(AGLOBL, N, N);
		p->from = ao;
		p->to = ac;
		p->to.offset = stringo;
	}
}
