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

#include <u.h>
#include <libc.h>
#include "gg.h"

void
zname(Biobuf *b, Sym *s, int t)
{
	Bputc(b, ANAME);	/* as */
	Bputc(b, ANAME>>8);	/* as */
	Bputc(b, t);		/* type */
	Bputc(b, s->sym);	/* sym */

	Bputname(b, s);
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
		ieeedtod(&e, a->u.dval);
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
		n = a->u.sval;
		for(i=0; i<NSNAME; i++) {
			Bputc(b, *n);
			n++;
		}
		return;
	}
	if(t & T_TYPE)
		Bputc(b, a->type);
	if(t & T_GOTYPE)
		Bputc(b, gotype);
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

			Bputc(bout, p->as);
			Bputc(bout, p->as>>8);
			Bputc(bout, p->lineno);
			Bputc(bout, p->lineno>>8);
			Bputc(bout, p->lineno>>16);
			Bputc(bout, p->lineno>>24);
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
	a->offset = widthptr+widthint;  // skip header
	a->etype = simtype[TINT];
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

	nodconst(&nod1, types[TINT], sval->len);
	p = gins(ADATA, nam, &nod1);
	p->from.scale = widthint;
	p->from.offset += widthptr;
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
	p->to.etype = simtype[TINT];
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
	p->to.etype = simtype[TINT];
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
genembedtramp(Type *rcvr, Type *method, Sym *newnam, int iface)
{
	Sym *e;
	int c, d, mov, add, loaded;
	int64 o;
	Prog *p;
	Type *f;
	
	USED(iface);

	if(0 && debug['r'])
		print("genembedtramp %T %T %S\n", rcvr, method, newnam);

	e = method->sym;
	for(d=0; d<nelem(dotlist); d++) {
		c = adddot1(e, rcvr, d, nil, 0);
		if(c == 1)
			goto out;
	}
	fatal("genembedtramp %T.%S", rcvr, method->sym);

out:
	newplist()->name = newname(newnam);

	//TEXT	main·S_test2(SB),7,$0
	p = pc;
	gins(ATEXT, N, N);
	p->from.type = D_EXTERN;
	p->from.sym = newnam;
	p->to.type = D_CONST;
	p->to.offset = 0;
	p->from.scale = 7;
//print("1. %P\n", p);

	mov = AMOVQ;
	add = AADDQ;
	loaded = 0;
	o = 0;
	for(c=d-1; c>=0; c--) {
		f = dotlist[c].field;
		o += f->width;
		if(!isptr[f->type->etype])
			continue;
		if(!loaded) {
			loaded = 1;
			//MOVQ	8(SP), AX
			p = pc;
			gins(mov, N, N);
			p->from.type = D_INDIR+D_SP;
			p->from.offset = widthptr;
			p->to.type = D_AX;
//print("2. %P\n", p);
		}

		//MOVQ	o(AX), AX
		p = pc;
		gins(mov, N, N);
		p->from.type = D_INDIR+D_AX;
		p->from.offset = o;
		p->to.type = D_AX;
//print("3. %P\n", p);
		o = 0;
	}
	if(o != 0) {
		//ADDQ	$XX, AX
		p = pc;
		gins(add, N, N);
		p->from.type = D_CONST;
		p->from.offset = o;
		if(loaded)
			p->to.type = D_AX;
		else {
			p->to.type = D_INDIR+D_SP;
			p->to.offset = widthptr;
		}
//print("4. %P\n", p);
	}

	//MOVQ	AX, 8(SP)
	if(loaded) {
		p = pc;
		gins(mov, N, N);
		p->from.type = D_AX;
		p->to.type = D_INDIR+D_SP;
		p->to.offset = widthptr;
//print("5. %P\n", p);
	} else {
		// TODO(rsc): obviously this is unnecessary,
		// but 6l has a bug, and it can't handle
		// JMP instructions too close to the top of
		// a new function.
		gins(ANOP, N, N);
	}

	f = dotlist[0].field;
	//JMP	main·*Sub_test2(SB)
	if(isptr[f->type->etype])
		f = f->type;
	p = pc;
	gins(AJMP, N, N);
	p->to.type = D_EXTERN;
	p->to.sym = methodsym(method->sym, ptrto(f->type), 0);
//print("6. %P\n", p);

	pc->as = ARET;	// overwrite AEND
}

void
nopout(Prog *p)
{
	p->as = ANOP;
}

