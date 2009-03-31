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

#include "gg.h"

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
zfile(Biobuf *b, char *p, int n)
{
	Bputc(b, ANAME);
	Bputc(b, ANAME>>8);
	Bputc(b, D_FILE);
	Bputc(b, 1);
	Bputc(b, '<');
	Bwrite(b, p, n);
	Bputc(b, 0);
}

void
zhist(Biobuf *b, int line, vlong offset)
{
	Addr a;

	Bputc(b, AHISTORY);
	Bputc(b, AHISTORY>>8);
	Bputc(b, line);
	Bputc(b, line>>8);
	Bputc(b, line>>16);
	Bputc(b, line>>24);
	zaddr(b, &zprog.from, 0);
	a = zprog.to;
	if(offset != 0) {
		a.offset = offset;
		a.type = D_CONST;
	}
	zaddr(b, &a, 0);
}

void
zaddr(Biobuf *b, Addr *a, int s)
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

	switch(a->type) {

	case D_BRANCH:
		a->offset = a->branch->loc;

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
	}
	if(t & T_OFFSET2) {	/* implies offset */
		l = a->offset2;
		Bputc(b, l);
		Bputc(b, l>>8);
		Bputc(b, l>>16);
		Bputc(b, l>>24);
	}
	if(t & T_SYM)		/* implies sym */
		Bputc(b, s);
	if(t & T_FCONST) {
		ieeedtod(&e, a->dval);
		l = e;
		Bputc(b, l);
		Bputc(b, l>>8);
		Bputc(b, l>>16);
		Bputc(b, l>>24);
		l = e >> 32;
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
dumpfuncs(void)
{
	Plist *pl;
	int sf, st, t, sym;
	struct { Sym *sym; short type; } h[NSYM];
	Sym *s;
	Prog *p;

	for(sym=0; sym<NSYM; sym++) {
		h[sym].sym = S;
		h[sym].type = 0;
	}
	sym = 1;

	// fix up pc
	pcloc = 0;
	for(pl=plist; pl!=nil; pl=pl->link) {
		for(p=pl->firstpc; p!=P; p=p->link) {
			p->loc = pcloc;
			pcloc++;
		}
	}

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
	p->to.type = D_ADDR;
	p->to.index = D_STATIC;
	p->to.etype = TINT32;
	p->to.sym = symstringo;
	p->to.offset = stringo;
	off += widthptr;

	datastring(str, strlen(str)+1);
	return off;
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
duint32(Sym *s, int off, uint32 v)
{
	return duintxx(s, off, v, 4);
}

int
duint16(Sym *s, int off, uint32 v)
{
	return duintxx(s, off, v, 2);
}

int
duintptr(Sym *s, int off, uint32 v)
{
	return duintxx(s, off, v, 8);
}

int
dsymptr(Sym *s, int off, Sym *x)
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
	p->to.offset = 0;
	off += widthptr;

	return off;
}


void
genembedtramp(Type *t, Sig *b)
{
	Sym *e;
	int c, d, o;
	Prog *p;
	Type *f;

	e = lookup(b->name);
	for(d=0; d<nelem(dotlist); d++) {
		c = adddot1(e, t, d, nil);
		if(c == 1)
			goto out;
	}
	fatal("genembedtramp %T.%s", t, b->name);

out:
	if(d == 0)
		return;

//	print("genembedtramp %d\n", d);
//	print("	t    = %lT\n", t);
//	print("	name = %s\n", b->name);
//	print("	sym  = %S\n", b->sym);
//	print("	hash = 0x%ux\n", b->hash);

	newplist()->name = newname(b->sym);

	//TEXT	main·S_test2(SB),7,$0
	p = pc;
	gins(ATEXT, N, N);
	p->from.type = D_EXTERN;
	p->from.sym = b->sym;
	p->to.type = D_CONST;
	p->to.offset = 0;
	p->from.scale = 7;
//print("1. %P\n", p);

	//MOVL	4(SP), AX
	p = pc;
	gins(AMOVL, N, N);
	p->from.type = D_INDIR+D_SP;
	p->from.offset = 4;
	p->to.type = D_AX;
//print("2. %P\n", p);

	o = 0;
	for(c=d-1; c>=0; c--) {
		f = dotlist[c].field;
		o += f->width;
		if(!isptr[f->type->etype])
			continue;
		//MOVL	o(AX), AX
		p = pc;
		gins(AMOVL, N, N);
		p->from.type = D_INDIR+D_AX;
		p->from.offset = o;
		p->to.type = D_AX;
//print("3. %P\n", p);
		o = 0;
	}
	if(o != 0) {
		//ADDL	$XX, AX
		p = pc;
		gins(AADDL, N, N);
		p->from.type = D_CONST;
		p->from.offset = o;
		p->to.type = D_AX;
//print("4. %P\n", p);
	}

	//MOVL	AX, 4(SP)
	p = pc;
	gins(AMOVL, N, N);
	p->from.type = D_AX;
	p->to.type = D_INDIR+D_SP;
	p->to.offset = 8;
//print("5. %P\n", p);

	f = dotlist[0].field;
	//JMP	main·*Sub_test2(SB)
	if(isptr[f->type->etype])
		f = f->type;
	p = pc;
	gins(AJMP, N, N);
	p->to.type = D_EXTERN;
	p->to.sym = methodsym(lookup(b->name), ptrto(f->type));
//print("6. %P\n", p);

	pc->as = ARET;	// overwrite AEND
}

void
nopout(Prog *p)
{
	p->as = ANOP;
}

