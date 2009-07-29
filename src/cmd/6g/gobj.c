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
zname(Biobuf *b, Sym *s, int t)
{
	char *n;

	Bputc(b, ANAME);	/* as */
	Bputc(b, ANAME>>8);	/* as */
	Bputc(b, t);		/* type */
	Bputc(b, s->sym);	/* sym */

	for(n=s->package; *n; n++)
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
		if(a->branch == nil)
			fatal("unpatched branch");
		a->offset = a->branch->loc;

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
			if(p->as != ADATA && p->as != AGLOBL)
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

/* deferred DATA output */
static Prog *strdat;
static Prog *estrdat;
static int gflag;
static Prog *savepc;

static void
data(void)
{
	gflag = debug['g'];
	debug['g'] = 0;

	if(estrdat == nil) {
		strdat = mal(sizeof(*pc));
		clearp(strdat);
		estrdat = strdat;
	}
	if(savepc)
		fatal("data phase error");
	savepc = pc;
	pc = estrdat;
}

static void
text(void)
{
	if(!savepc)
		fatal("text phase error");
	debug['g'] = gflag;
	estrdat = pc;
	pc = savepc;
	savepc = nil;
}

void
dumpdata(void)
{
	Prog *p;

	if(estrdat == nil)
		return;
	*pc = *strdat;
	if(gflag)
		for(p=pc; p!=estrdat; p=p->link)
			print("%P\n", p);
	pc = estrdat;
}

/*
 * make a refer to the data s, s+len
 * emitting DATA if needed.
 */
void
datastring(char *s, int len, Addr *a)
{
	int w;
	Prog *p;
	Addr ac, ao;
	static int gen;
	struct {
		Strlit lit;
		char buf[100];
	} tmp;

	// string
	memset(&ao, 0, sizeof(ao));
	ao.type = D_STATIC;
	ao.index = D_NONE;
	ao.etype = TINT32;
	ao.offset = 0;		// fill in

	// constant
	memset(&ac, 0, sizeof(ac));
	ac.type = D_CONST;
	ac.index = D_NONE;
	ac.offset = 0;		// fill in

	// huge strings are made static to avoid long names.
	if(len > 100) {
		snprint(namebuf, sizeof(namebuf), ".string.%d", gen++);
		ao.sym = lookup(namebuf);
		ao.type = D_STATIC;
	} else {
		if(len > 0 && s[len-1] == '\0')
			len--;
		tmp.lit.len = len;
		memmove(tmp.lit.s, s, len);
		tmp.lit.s[len] = '\0';
		len++;
		snprint(namebuf, sizeof(namebuf), "\"%Z\"", &tmp.lit);
		ao.sym = pkglookup(namebuf, "string");
		ao.type = D_EXTERN;
	}
	*a = ao;

	// only generate data the first time.
	if(ao.sym->uniq)
		return;
	ao.sym->uniq = 1;

	data();
	for(w=0; w<len; w+=8) {
		p = pc;
		gins(ADATA, N, N);

		// DATA s+w, [NSNAME], $"xxx"
		p->from = ao;
		p->from.offset = w;

		p->from.scale = NSNAME;
		if(w+8 > len)
			p->from.scale = len-w;

		p->to = ac;
		p->to.type = D_SCONST;
		p->to.offset = len;
		memmove(p->to.sval, s+w, p->from.scale);
	}
	p = pc;
	ggloblsym(ao.sym, len, ao.type == D_EXTERN);
	if(ao.type == D_STATIC)
		p->from.type = D_STATIC;
	text();
}

/*
 * make a refer to the string sval,
 * emitting DATA if needed.
 */
void
datagostring(Strlit *sval, Addr *a)
{
	Prog *p;
	Addr ac, ao, ap;
	int32 wi, wp;
	static int gen;

	memset(&ac, 0, sizeof(ac));
	memset(&ao, 0, sizeof(ao));
	memset(&ap, 0, sizeof(ap));

	// constant
	ac.type = D_CONST;
	ac.index = D_NONE;
	ac.offset = 0;			// fill in

	// string len+ptr
	ao.type = D_STATIC;		// fill in
	ao.index = D_NONE;
	ao.etype = TINT32;
	ao.sym = nil;			// fill in

	// $string len+ptr
	datastring(sval->s, sval->len, &ap);
	ap.index = ap.type;
	ap.type = D_ADDR;
	ap.etype = TINT32;

	wi = types[TUINT32]->width;
	wp = types[tptr]->width;

	if(ap.index == D_STATIC) {
		// huge strings are made static to avoid long names
		snprint(namebuf, sizeof(namebuf), ".gostring.%d", ++gen);
		ao.sym = lookup(namebuf);
		ao.type = D_STATIC;
	} else {
		// small strings get named by their contents,
		// so that multiple modules using the same string
		// can share it.
		snprint(namebuf, sizeof(namebuf), "\"%Z\"", sval);
		ao.sym = pkglookup(namebuf, "go.string");
		ao.type = D_EXTERN;
	}

	*a = ao;
	if(ao.sym->uniq)
		return;
	ao.sym->uniq = 1;

	data();
	// DATA gostring, wp, $cstring
	p = pc;
	gins(ADATA, N, N);
	p->from = ao;
	p->from.scale = wp;
	p->to = ap;

	// DATA gostring+wp, wi, $len
	p = pc;
	gins(ADATA, N, N);
	p->from = ao;
	p->from.offset = wp;
	p->from.scale = wi;
	p->to = ac;
	p->to.offset = sval->len;

	p = pc;
	ggloblsym(ao.sym, types[TSTRING]->width, ao.type == D_EXTERN);
	if(ao.type == D_STATIC)
		p->from.type = D_STATIC;
	text();
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
genembedtramp(Type *rcvr, Type *method, Sym *newnam)
{
	Sym *e;
	int c, d, o, mov, add, loaded;
	Prog *p;
	Type *f;

	if(debug['r'])
		print("genembedtramp %T %T %S\n", rcvr, method, newnam);

	e = method->sym;
	for(d=0; d<nelem(dotlist); d++) {
		c = adddot1(e, rcvr, d, nil);
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
		p = pc;
		gins(ANOP, N, N);
	}

	f = dotlist[0].field;
	//JMP	main·*Sub_test2(SB)
	if(isptr[f->type->etype])
		f = f->type;
	p = pc;
	gins(AJMP, N, N);
	p->to.type = D_EXTERN;
	p->to.sym = methodsym(method->sym, ptrto(f->type));
//print("6. %P\n", p);

	pc->as = ARET;	// overwrite AEND
}

void
nopout(Prog *p)
{
	p->as = ANOP;
}

